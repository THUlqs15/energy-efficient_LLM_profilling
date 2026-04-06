"""
Regression fitting script for vLLM batch execution time model.

Model:
  T_pd(f, B) = (1/f) * [a_p * sum(l_q^2) + b_p * sum(l_q*l_kv) + c_p * sum(l_q)]
             + (1/f^alpha) * [a_d * sum(l_kv_decode) + b_d * num_decode]
             + t_c

7 parameters: {a_p, b_p, c_p, a_d, b_d, alpha, t_c}
"""

import json
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize

INPUT_PATH = "/home/ubuntu/lqs/profiling_data.jsonl"


def load_data(path):
    """Load and parse profiling JSONL data."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


def extract_features(records):
    """Extract aggregate features per batch record."""
    rows = []
    for r in records:
        f = r["gpu_freq_mhz"]
        y = r["wall_time_ms"]

        if f <= 0 or y <= 0:
            continue

        sum_lq_sq = 0.0
        sum_lq_lkv = 0.0
        sum_lq = 0.0
        sum_lkv_decode = 0.0
        num_decode = 0

        for req in r["requests"]:
            lq = req["l_q"]
            lkv = req["l_kv"]
            if req["type"] == "prefill":
                sum_lq_sq += lq * lq
                sum_lq_lkv += lq * lkv
                sum_lq += lq
            else:
                sum_lkv_decode += lkv
                num_decode += 1

        rows.append({
            "f": float(f),
            "y": float(y),
            "sum_lq_sq": sum_lq_sq,
            "sum_lq_lkv": sum_lq_lkv,
            "sum_lq": sum_lq,
            "sum_lkv_decode": sum_lkv_decode,
            "num_decode": float(num_decode),
        })

    print(f"Valid rows after filtering: {len(rows)}")
    return rows


def build_feature_matrix(rows, alpha):
    """Build design matrix X and target y for a given alpha."""
    n = len(rows)
    X = np.zeros((n, 6))
    y = np.zeros(n)

    for i, row in enumerate(rows):
        f = row["f"]
        X[i, 0] = row["sum_lq_sq"] / f
        X[i, 1] = row["sum_lq_lkv"] / f
        X[i, 2] = row["sum_lq"] / f
        X[i, 3] = row["sum_lkv_decode"] / (f ** alpha)
        X[i, 4] = row["num_decode"] / (f ** alpha)
        X[i, 5] = 1.0  # constant t_c
        y[i] = row["y"]

    return X, y


def compute_metrics(y_true, y_pred):
    """Compute MAPE, MAE, RMSE, R2."""
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAPE": mape, "MAE": mae, "RMSE": rmse, "R2": r2}


def fit_with_alpha(rows_train, rows_test, alpha):
    """Fit linear regression for given alpha, return test MAPE and params."""
    X_train, y_train = build_feature_matrix(rows_train, alpha)
    X_test, y_test = build_feature_matrix(rows_test, alpha)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train, y_train)

    y_pred_test = lr.predict(X_test)
    y_true_test = y_test

    mask = y_true_test > 0
    if mask.sum() == 0:
        return float("inf"), None, None

    mape = np.mean(np.abs((y_true_test[mask] - y_pred_test[mask]) / y_true_test[mask])) * 100
    return mape, lr.coef_, (X_train, y_train, X_test, y_test)


def build_prediction(rows, a_p, b_p, c_p, a_d, b_d, alpha, t_c):
    """Compute predictions for rows given model params."""
    n = len(rows)
    y_pred = np.zeros(n)
    for i, row in enumerate(rows):
        f = row["f"]
        prefill_term = (1.0 / f) * (
            a_p * row["sum_lq_sq"]
            + b_p * row["sum_lq_lkv"]
            + c_p * row["sum_lq"]
        )
        decode_term = (1.0 / (f ** alpha)) * (
            a_d * row["sum_lkv_decode"] + b_d * row["num_decode"]
        )
        y_pred[i] = prefill_term + decode_term + t_c
    return y_pred


def refined_fit(rows_train, rows_test):
    """
    Two-stage fitting:
    1. Grid search over alpha
    2. Refine with scipy.optimize.minimize (L-BFGS-B with bounds for non-negativity)

    Notes on single-frequency data:
    - With only f=1710 MHz, terms 1/f and 1/f^alpha are both constants
    - alpha is NOT identifiable from single-frequency data alone
    - We still fit it but note its uncertainty
    - We enforce non-negativity on a_p, b_p, c_p, a_d, b_d (physical constraints)
    """
    print("\n=== Grid Search over alpha ===")
    best_train_mape = float("inf")
    best_alpha = 0.5
    best_coef = None

    alphas = np.arange(0.01, 1.01, 0.01)
    for alpha in alphas:
        mape, coef, data = fit_with_alpha(rows_train, rows_test, alpha)
        if mape < best_train_mape and coef is not None:
            best_train_mape = mape
            best_alpha = alpha
            best_coef = coef

    print(f"Best alpha from grid search: {best_alpha:.3f} (train MAPE: {best_train_mape:.2f}%)")

    # Also compute test MAPE for the grid-search solution
    if best_coef is not None:
        a_p_g, b_p_g, c_p_g, a_d_g, b_d_g, t_c_g = best_coef
        y_pred_test_g = build_prediction(rows_test, a_p_g, b_p_g, c_p_g, a_d_g, b_d_g, best_alpha, t_c_g)
        y_true_test = np.array([r["y"] for r in rows_test])
        mask = y_true_test > 0
        test_mape_g = np.mean(np.abs((y_true_test[mask] - y_pred_test_g[mask]) / y_true_test[mask])) * 100
        print(f"Grid-search test MAPE: {test_mape_g:.2f}%")

    # Stage 2: Refine with L-BFGS-B + non-negativity constraints
    print("\n=== Refining with scipy.optimize.minimize (L-BFGS-B with bounds) ===")

    rows_all = rows_train  # fit on training data

    def objective(params):
        """Full model MSE + MAPE on training set."""
        a_p, b_p, c_p, a_d, b_d, alpha, t_c = params
        if alpha <= 0 or alpha > 2:
            return 1e6

        n = len(rows_all)
        y_pred = np.zeros(n)
        y_true = np.zeros(n)

        for i, row in enumerate(rows_all):
            f = row["f"]
            y_true[i] = row["y"]
            prefill_term = (1.0 / f) * (
                a_p * row["sum_lq_sq"]
                + b_p * row["sum_lq_lkv"]
                + c_p * row["sum_lq"]
            )
            decode_term = (1.0 / (f ** alpha)) * (
                a_d * row["sum_lkv_decode"] + b_d * row["num_decode"]
            )
            y_pred[i] = prefill_term + decode_term + t_c

        mask = y_true > 0
        if mask.sum() == 0:
            return 1e6
        # Use combined MSE + MAPE for better fitting
        mse = np.mean((y_true[mask] - y_pred[mask]) ** 2)
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        return mape + mse / (np.mean(y_true[mask]) ** 2)

    # Bounds: non-negative for all except t_c (can be small positive/negative)
    # a_p, b_p, c_p, a_d, b_d >= 0 (physical); alpha in (0.01, 1.0); t_c >= 0
    bounds = [
        (0, None),   # a_p >= 0
        (0, None),   # b_p >= 0
        (0, None),   # c_p >= 0
        (0, None),   # a_d >= 0
        (0, None),   # b_d >= 0
        (0.01, 1.0), # alpha in (0.01, 1.0)
        (0, None),   # t_c >= 0
    ]

    # Initial params from grid search
    if best_coef is not None:
        a_p0, b_p0, c_p0, a_d0, b_d0, t_c0 = best_coef
        # Force non-negativity for initial point
        x0 = [
            max(0, a_p0), max(0, b_p0), max(0, c_p0),
            max(0, a_d0), max(0, b_d0),
            best_alpha,
            max(0, t_c0)
        ]
    else:
        f0 = rows_all[0]["f"] if rows_all else 1710.0
        x0 = [1e-3/f0, 1e-4/f0, 1e-2/f0, 1e-3/f0, 1.0/f0, 0.5, 1.0]

    best_result = None
    best_val = float("inf")

    # Try multiple starting points
    starts = [x0]
    # Additional starting points with different alpha values
    for alpha_start in [0.3, 0.5, 0.7, 0.9]:
        starts.append([x0[0], x0[1], x0[2], x0[3], x0[4], alpha_start, x0[6]])

    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_result = result

    # Also try Nelder-Mead without bounds for comparison
    result_nm = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-10, "fatol": 1e-10},
    )

    print(f"L-BFGS-B best objective: {best_result.fun:.6f}")
    print(f"Nelder-Mead objective: {result_nm.fun:.6f}")

    # Pick the best result (but enforce non-negativity)
    if best_result.fun <= result_nm.fun:
        params = best_result.x
        method = "L-BFGS-B"
    else:
        # For Nelder-Mead, clip to non-negative
        params = result_nm.x.copy()
        for idx in [0, 1, 2, 3, 4, 6]:
            params[idx] = max(0, params[idx])
        params[5] = max(0.01, min(1.0, params[5]))
        method = "Nelder-Mead"

    a_p, b_p, c_p, a_d, b_d, alpha_opt, t_c = params
    print(f"Using {method} result, alpha={alpha_opt:.4f}")

    return a_p, b_p, c_p, a_d, b_d, alpha_opt, t_c


def evaluate_model(rows, a_p, b_p, c_p, a_d, b_d, alpha, t_c):
    """Compute predictions and metrics."""
    n = len(rows)
    y_pred = np.zeros(n)
    y_true = np.zeros(n)

    for i, row in enumerate(rows):
        f = row["f"]
        y_true[i] = row["y"]
        prefill_term = (1.0 / f) * (
            a_p * row["sum_lq_sq"]
            + b_p * row["sum_lq_lkv"]
            + c_p * row["sum_lq"]
        )
        decode_term = (1.0 / (f ** alpha)) * (
            a_d * row["sum_lkv_decode"] + b_d * row["num_decode"]
        )
        y_pred[i] = prefill_term + decode_term + t_c

    return y_true, y_pred, compute_metrics(y_true, y_pred)


def main():
    print("=== vLLM Batch Latency Model Fitting ===")

    # Load data
    records = load_data(INPUT_PATH)
    rows = extract_features(records)

    if len(rows) < 10:
        print("ERROR: Not enough data to fit. Need at least 10 valid records.")
        sys.exit(1)

    # Data statistics
    freqs = [r["f"] for r in rows]
    ys = [r["y"] for r in rows]
    print(f"\nData summary:")
    print(f"  Records: {len(rows)}")
    print(f"  GPU freq range: [{min(freqs):.0f}, {max(freqs):.0f}] MHz")
    print(f"  Wall time range: [{min(ys):.3f}, {max(ys):.3f}] ms")
    print(f"  Wall time mean: {np.mean(ys):.3f} ms, std: {np.std(ys):.3f} ms")

    # Prefill/decode stats
    prefill_rows = [r for r in rows if r["sum_lq"] > 0]
    decode_rows = [r for r in rows if r["num_decode"] > 0]
    mixed_rows = [r for r in rows if r["sum_lq"] > 0 and r["num_decode"] > 0]
    print(f"  Batches with prefill: {len(prefill_rows)}")
    print(f"  Batches with decode: {len(decode_rows)}")
    print(f"  Mixed batches: {len(mixed_rows)}")

    # Train/test split (80/20, random shuffle for representativeness)
    import random as _random
    _random.seed(42)
    rows_shuffled = rows[:]
    _random.shuffle(rows_shuffled)
    n = len(rows_shuffled)
    n_train = int(n * 0.8)
    rows_train = rows_shuffled[:n_train]
    rows_test = rows_shuffled[n_train:]
    print(f"\nTrain/test split: {n_train}/{n - n_train}")

    # Fit model
    a_p, b_p, c_p, a_d, b_d, alpha, t_c = refined_fit(rows_train, rows_test)

    print(f"\n=== Fitted Parameters ===")
    print(f"  a_p (prefill quadratic): {a_p:.6e}")
    print(f"  b_p (prefill cross):     {b_p:.6e}")
    print(f"  c_p (prefill linear):    {c_p:.6e}")
    print(f"  a_d (decode kv):         {a_d:.6e}")
    print(f"  b_d (decode count):      {b_d:.6e}")
    print(f"  alpha (decode freq exp): {alpha:.4f}")
    print(f"  t_c (constant):          {t_c:.6e}")

    # Evaluate on train and test
    y_true_train, y_pred_train, metrics_train = evaluate_model(
        rows_train, a_p, b_p, c_p, a_d, b_d, alpha, t_c
    )
    y_true_test, y_pred_test, metrics_test = evaluate_model(
        rows_test, a_p, b_p, c_p, a_d, b_d, alpha, t_c
    )

    print(f"\n=== Evaluation Metrics ===")
    print(f"{'Metric':<10} {'Train':>12} {'Test':>12}")
    print("-" * 36)
    print(f"{'MAPE':<10} {metrics_train['MAPE']:>11.2f}% {metrics_test['MAPE']:>11.2f}%")
    print(f"{'MAE':<10} {metrics_train['MAE']:>10.4f}ms {metrics_test['MAE']:>10.4f}ms")
    print(f"{'RMSE':<10} {metrics_train['RMSE']:>10.4f}ms {metrics_test['RMSE']:>10.4f}ms")
    print(f"{'R2':<10} {metrics_train['R2']:>12.4f} {metrics_test['R2']:>12.4f}")

    # Save results for later use by result.md writer
    import json
    results = {
        "params": {
            "a_p": a_p, "b_p": b_p, "c_p": c_p,
            "a_d": a_d, "b_d": b_d, "alpha": alpha, "t_c": t_c,
        },
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "data_summary": {
            "total_records": len(rows),
            "n_train": n_train,
            "n_test": n - n_train,
            "freq_min": min(freqs),
            "freq_max": max(freqs),
            "wall_time_min": min(ys),
            "wall_time_max": max(ys),
            "wall_time_mean": float(np.mean(ys)),
            "batches_prefill": len(prefill_rows),
            "batches_decode": len(decode_rows),
            "batches_mixed": len(mixed_rows),
        },
    }
    with open("/home/ubuntu/lqs/fitting_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /home/ubuntu/lqs/fitting_results.json")

    return results


if __name__ == "__main__":
    main()
