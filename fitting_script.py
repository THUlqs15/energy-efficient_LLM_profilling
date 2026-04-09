#!/usr/bin/env python3
"""
Batch Latency Model Fitting Script.

Reads profiling data and fits the batch latency regression model with GPU frequency.

Usage:
    python fitting_script.py --input profiling_data.jsonl

The model fits:
    T_pd(f, B) = (1/f) * [a_p * Σl_q² + b_p * Σl_q·l_kv + c_p * Σl_q]
               + (1/f^α) * [a_d * Σl_kv + b_d * num_decode]
               + t_c

where:
    - f: GPU frequency (MHz)
    - a_p, b_p, c_p: prefill coefficients
    - a_d, b_d: decode coefficients
    - α: frequency scaling exponent for decode (0 < α < 1)
    - t_c: constant overhead
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def load_profiling_data(jsonl_path):
    """Load profiling data from JSONL file."""
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def compute_aggregate_features(batch_record):
    """
    Compute aggregate features from a batch record.

    Returns:
        dict with keys:
            - f: GPU frequency (MHz)
            - sum_lq_sq: Σl_q² for prefill
            - sum_lq_lkv: Σl_q·l_kv for prefill
            - sum_lq: Σl_q for prefill
            - sum_lkv_decode: Σl_kv for decode
            - num_decode: number of decode requests
            - wall_time_ms: measured execution time
    """
    f = batch_record["gpu_freq_mhz"]
    requests = batch_record["requests"]
    wall_time_ms = batch_record["wall_time_ms"]

    # Prefill aggregates
    sum_lq_sq = 0.0
    sum_lq_lkv = 0.0
    sum_lq = 0.0

    # Decode aggregates
    sum_lkv_decode = 0.0
    num_decode = 0

    for req in requests:
        req_type = req["type"]
        l_q = req["l_q"]
        l_kv = req["l_kv"]

        if req_type == "prefill":
            sum_lq_sq += l_q ** 2
            sum_lq_lkv += l_q * l_kv
            sum_lq += l_q
        elif req_type == "decode":
            sum_lkv_decode += l_kv
            num_decode += 1

    return {
        "f": f,
        "sum_lq_sq": sum_lq_sq,
        "sum_lq_lkv": sum_lq_lkv,
        "sum_lq": sum_lq,
        "sum_lkv_decode": sum_lkv_decode,
        "num_decode": num_decode,
        "wall_time_ms": wall_time_ms,
    }


def prepare_dataset(profiling_data):
    """
    Convert raw profiling data to feature matrix and target vector.

    Returns:
        X_raw: list of feature dicts
        y: numpy array of measured times (ms)
    """
    X_raw = []
    y = []

    for batch_record in profiling_data:
        features = compute_aggregate_features(batch_record)

        # Filter out empty batches (no prefill or decode)
        if features["sum_lq"] == 0 and features["num_decode"] == 0:
            continue

        # Filter out batches with invalid time measurements
        if features["wall_time_ms"] <= 0:
            continue

        X_raw.append(features)
        y.append(features["wall_time_ms"])

    return X_raw, np.array(y)


def fit_model_grid_search(X_raw_train, y_train, X_raw_test, y_test):
    """
    Fit model using grid search over α.

    Returns:
        dict with:
            - params: [a_p, b_p, c_p, a_d, b_d, alpha, t_c]
            - train_mape: training MAPE
            - test_mape: test MAPE
            - train_metrics: dict with MAE, RMSE, R²
            - test_metrics: dict with MAE, RMSE, R²
    """
    print("\n=== Grid Search over α ===")

    best_mape = float("inf")
    best_alpha = None
    best_params = None
    best_model = None

    # Grid search over α from 0.01 to 0.99
    alpha_range = np.arange(0.05, 1.0, 0.01)

    for alpha in alpha_range:
        # Build feature matrix for this α
        X_train = []
        for feat in X_raw_train:
            f = feat["f"]
            row = [
                feat["sum_lq_sq"] / f,           # a_p term
                feat["sum_lq_lkv"] / f,          # b_p term
                feat["sum_lq"] / f,              # c_p term
                feat["sum_lkv_decode"] / (f ** alpha),  # a_d term
                feat["num_decode"] / (f ** alpha),      # b_d term
                1.0,                             # t_c term
            ]
            X_train.append(row)

        X_train = np.array(X_train)

        # Fit linear regression (no intercept since we have t_c)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)

        # Predict on test set
        X_test = []
        for feat in X_raw_test:
            f = feat["f"]
            row = [
                feat["sum_lq_sq"] / f,
                feat["sum_lq_lkv"] / f,
                feat["sum_lq"] / f,
                feat["sum_lkv_decode"] / (f ** alpha),
                feat["num_decode"] / (f ** alpha),
                1.0,
            ]
            X_test.append(row)

        X_test = np.array(X_test)
        y_pred = model.predict(X_test)

        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        if mape < best_mape:
            best_mape = mape
            best_alpha = alpha
            best_params = model.coef_
            best_model = model

    print(f"  Best α: {best_alpha:.4f}")
    print(f"  Best test MAPE: {best_mape:.2f}%")

    # Compute final metrics
    X_train_best = []
    for feat in X_raw_train:
        f = feat["f"]
        row = [
            feat["sum_lq_sq"] / f,
            feat["sum_lq_lkv"] / f,
            feat["sum_lq"] / f,
            feat["sum_lkv_decode"] / (f ** best_alpha),
            feat["num_decode"] / (f ** best_alpha),
            1.0,
        ]
        X_train_best.append(row)

    X_train_best = np.array(X_train_best)
    y_train_pred = best_model.predict(X_train_best)

    X_test_best = []
    for feat in X_raw_test:
        f = feat["f"]
        row = [
            feat["sum_lq_sq"] / f,
            feat["sum_lq_lkv"] / f,
            feat["sum_lq"] / f,
            feat["sum_lkv_decode"] / (f ** best_alpha),
            feat["num_decode"] / (f ** best_alpha),
            1.0,
        ]
        X_test_best.append(row)

    X_test_best = np.array(X_test_best)
    y_test_pred = best_model.predict(X_test_best)

    # Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # Extract parameters: [a_p, b_p, c_p, a_d, b_d, t_c]
    a_p, b_p, c_p, a_d, b_d, t_c = best_params

    return {
        "params": [a_p, b_p, c_p, a_d, b_d, best_alpha, t_c],
        "train_mape": train_metrics["mape"],
        "test_mape": test_metrics["mape"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def refine_with_nonlinear_optimization(
    X_raw_train, y_train, X_raw_test, y_test, initial_params
):
    """
    Refine parameters using non-linear optimization.

    Args:
        initial_params: [a_p, b_p, c_p, a_d, b_d, alpha, t_c]

    Returns:
        Same dict structure as fit_model_grid_search
    """
    print("\n=== Non-linear Optimization (refinement) ===")

    def predict(params, X_raw):
        """Predict using the model."""
        a_p, b_p, c_p, a_d, b_d, alpha, t_c = params
        predictions = []

        for feat in X_raw:
            f = feat["f"]
            T_p = (1.0 / f) * (
                a_p * feat["sum_lq_sq"]
                + b_p * feat["sum_lq_lkv"]
                + c_p * feat["sum_lq"]
            )
            T_d = (1.0 / (f ** alpha)) * (
                a_d * feat["sum_lkv_decode"]
                + b_d * feat["num_decode"]
            )
            T_total = T_p + T_d + t_c
            predictions.append(T_total)

        return np.array(predictions)

    def loss(params):
        """Mean squared error loss."""
        y_pred = predict(params, X_raw_train)
        return np.mean((y_train - y_pred) ** 2)

    # Bounds: all positive except t_c can be any value
    # α must be in (0.01, 0.99)
    bounds = [
        (0, None),    # a_p
        (0, None),    # b_p
        (0, None),    # c_p
        (0, None),    # a_d
        (0, None),    # b_d
        (0.01, 0.99), # alpha
        (None, None), # t_c
    ]

    result = minimize(
        loss,
        x0=initial_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-9},
    )

    if not result.success:
        print(f"  Warning: Optimization did not converge: {result.message}")

    refined_params = result.x
    print(f"  Refined α: {refined_params[5]:.4f}")

    # Compute metrics
    y_train_pred = predict(refined_params, X_raw_train)
    y_test_pred = predict(refined_params, X_raw_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print(f"  Refined test MAPE: {test_metrics['mape']:.2f}%")

    return {
        "params": refined_params.tolist(),
        "train_mape": train_metrics["mape"],
        "test_mape": test_metrics["mape"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def print_results(result, X_raw, y):
    """Print fitted parameters and metrics."""
    a_p, b_p, c_p, a_d, b_d, alpha, t_c = result["params"]

    print("\n" + "=" * 80)
    print("FITTED PARAMETERS")
    print("=" * 80)
    print(f"a_p (prefill l_q² coeff):       {a_p:.6e}")
    print(f"b_p (prefill l_q·l_kv coeff):   {b_p:.6e}")
    print(f"c_p (prefill l_q coeff):        {c_p:.6e}")
    print(f"a_d (decode l_kv coeff):        {a_d:.6e}")
    print(f"b_d (decode constant coeff):    {b_d:.6e}")
    print(f"α (alpha, freq exponent):       {alpha:.6f}")
    print(f"t_c (constant overhead, ms):    {t_c:.6f}")

    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"{'Metric':<10} {'Train':>12} {'Test':>12}")
    print("-" * 36)
    print(f"{'MAPE (%)':<10} {result['train_mape']:>12.2f} {result['test_mape']:>12.2f}")
    print(f"{'MAE (ms)':<10} {result['train_metrics']['mae']:>12.4f} {result['test_metrics']['mae']:>12.4f}")
    print(f"{'RMSE (ms)':<10} {result['train_metrics']['rmse']:>12.4f} {result['test_metrics']['rmse']:>12.4f}")
    print(f"{'R²':<10} {result['train_metrics']['r2']:>12.4f} {result['test_metrics']['r2']:>12.4f}")

    # Data summary
    freqs = sorted(set(feat["f"] for feat in X_raw))
    all_lq = []
    all_lkv = []
    for feat in X_raw:
        if feat["sum_lq"] > 0:
            # Approximate individual l_q values (we only have aggregates)
            all_lq.append(feat["sum_lq"])
        if feat["sum_lkv_decode"] > 0:
            all_lkv.append(feat["sum_lkv_decode"] / max(feat["num_decode"], 1))

    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total batches:              {len(y)}")
    print(f"GPU frequencies profiled:   {len(freqs)}")
    print(f"Frequency range:            [{min(freqs):.0f}, {max(freqs):.0f}] MHz")
    if all_lq:
        print(f"Prefill l_q range:          [{min(all_lq):.0f}, {max(all_lq):.0f}]")
    if all_lkv:
        print(f"Decode l_kv range (avg):    [{min(all_lkv):.0f}, {max(all_lkv):.0f}]")
    print(f"Wall time range:            [{y.min():.2f}, {y.max():.2f}] ms")

    print("\n" + "=" * 80)
    print("MODEL FORMULA")
    print("=" * 80)
    print(f"T_pd(f, B) = (1/f) × [{a_p:.6e}·Σl_q² + {b_p:.6e}·Σl_q·l_kv + {c_p:.6e}·Σl_q]")
    print(f"           + (1/f^{alpha:.6f}) × [{a_d:.6e}·Σl_kv + {b_d:.6e}·num_decode]")
    print(f"           + {t_c:.6f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Fit batch latency regression model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL profiling data file"
    )
    parser.add_argument(
        "--output-params",
        type=str,
        help="Optional: output fitted parameters to JSON file"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip non-linear refinement step"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test set split ratio (default: 0.2)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Batch Latency Model Fitting")
    print("=" * 80)

    # Load data
    print(f"\nLoading profiling data from: {args.input}")
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    profiling_data = load_profiling_data(args.input)
    print(f"  Loaded {len(profiling_data)} batch records")

    # Prepare dataset
    print("\nPreparing dataset...")
    X_raw, y = prepare_dataset(profiling_data)
    print(f"  Valid samples: {len(y)}")

    if len(y) < 100:
        print(f"Warning: Only {len(y)} samples available. Need at least 100-200 for good fit.")

    # Split train/test
    print(f"\nSplitting data (test ratio: {args.test_split})...")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_split, random_state=42
    )

    X_raw_train = [X_raw[i] for i in train_idx]
    X_raw_test = [X_raw[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"  Train set: {len(y_train)} samples")
    print(f"  Test set: {len(y_test)} samples")

    # Fit model with grid search
    result = fit_model_grid_search(X_raw_train, y_train, X_raw_test, y_test)

    # Refine with non-linear optimization
    if not args.no_refine:
        result = refine_with_nonlinear_optimization(
            X_raw_train, y_train, X_raw_test, y_test, result["params"]
        )

    # Print results
    print_results(result, X_raw, y)

    # Save parameters
    if args.output_params:
        params_dict = {
            "a_p": result["params"][0],
            "b_p": result["params"][1],
            "c_p": result["params"][2],
            "a_d": result["params"][3],
            "b_d": result["params"][4],
            "alpha": result["params"][5],
            "t_c": result["params"][6],
            "train_mape": result["train_mape"],
            "test_mape": result["test_mape"],
            "train_metrics": result["train_metrics"],
            "test_metrics": result["test_metrics"],
        }

        with open(args.output_params, "w") as f:
            json.dump(params_dict, f, indent=2)

        print(f"\nParameters saved to: {args.output_params}")

    print("\n✓ Fitting complete!\n")


if __name__ == "__main__":
    main()
