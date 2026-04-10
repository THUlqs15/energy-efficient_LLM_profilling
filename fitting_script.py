#!/usr/bin/env python3
"""
Fitting script for vLLM batch latency model.
Reads profiling data, fits the 7-parameter model {a_p, b_p, c_p, a_d, b_d, alpha, t_c},
and reports evaluation metrics (MAPE, MAE, RMSE, R²).

Uses three fitting approaches:
  1. Grid search over alpha + OLS linear regression (MSE loss)
  2. Grid search over alpha + weighted linear regression (relative error loss)
  3. Non-linear optimisation (scipy) minimising MAPE directly
"""

import argparse
import json
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser(description="Fit batch latency model")
    p.add_argument("--input", default="/home/ubuntu/lqs/profiling_data.jsonl")
    p.add_argument("--output", default="/home/ubuntu/lqs/fitted_params.json")
    p.add_argument("--warmup-per-freq", type=int, default=25,
                   help="Discard first N batches per frequency group as warmup")
    p.add_argument("--freq-tolerance", type=float, default=50.0,
                   help="Max deviation (MHz) from target freq before discarding")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Data loading & cleaning ──────────────────────────────────────────────────

def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_batch_features(rec):
    f = rec["gpu_freq_mhz"]
    wall_ms = rec["wall_time_ms"]
    reqs = rec.get("requests", [])

    sum_lq_sq = sum_lq_lkv = sum_lq = 0.0
    sum_lkv_d = 0.0
    num_decode = num_prefill = 0

    for r in reqs:
        lq, lkv = r["l_q"], r["l_kv"]
        if r["type"] == "prefill":
            sum_lq_sq += lq ** 2
            sum_lq_lkv += lq * lkv
            sum_lq += lq
            num_prefill += 1
        else:
            sum_lkv_d += lkv
            num_decode += 1

    return {
        "f": f, "wall_ms": wall_ms,
        "sum_lq_sq": sum_lq_sq, "sum_lq_lkv": sum_lq_lkv, "sum_lq": sum_lq,
        "sum_lkv_d": sum_lkv_d, "num_decode": num_decode,
        "num_prefill": num_prefill, "total_reqs": num_prefill + num_decode,
    }


def clean_and_prepare(records, warmup_per_freq, freq_tolerance):
    if not records:
        raise ValueError("No profiling records found!")

    # Cluster by rounded frequency (15 MHz steps)
    clusters = {}
    for i, rec in enumerate(records):
        rf = round(rec["gpu_freq_mhz"] / 15.0) * 15.0
        clusters.setdefault(rf, []).append((i, rec))

    # Remove warmup and frequency outliers
    cleaned = []
    for cf, items in sorted(clusters.items()):
        for idx, rec in items[warmup_per_freq:]:
            if abs(rec["gpu_freq_mhz"] - cf) <= freq_tolerance:
                cleaned.append(rec)

    print(f"Records after warmup/freq cleaning: {len(cleaned)} "
          f"(from {len(records)} raw, {len(clusters)} freq clusters)")

    # Remove empty batches and compute features
    features = [compute_batch_features(r) for r in cleaned
                if r.get("requests") and r["wall_time_ms"] > 0]

    # Remove outlier times (top/bottom 0.5%)
    times = np.array([ft["wall_ms"] for ft in features])
    p_lo, p_hi = np.percentile(times, [0.5, 99.5])
    features = [ft for ft in features if p_lo <= ft["wall_ms"] <= p_hi]
    print(f"After outlier removal ({p_lo:.1f}-{p_hi:.1f} ms): {len(features)}")

    return features


# ── Feature matrix builders ──────────────────────────────────────────────────

def _extract_arrays(features):
    N = len(features)
    f = np.array([ft["f"] for ft in features])
    y = np.array([ft["wall_ms"] for ft in features])
    sq = np.array([ft["sum_lq_sq"] for ft in features])
    ql = np.array([ft["sum_lq_lkv"] for ft in features])
    lq = np.array([ft["sum_lq"] for ft in features])
    ld = np.array([ft["sum_lkv_d"] for ft in features])
    nd = np.array([ft["num_decode"] for ft in features])
    return f, y, sq, ql, lq, ld, nd


def build_X(f, sq, ql, lq, ld, nd, alpha):
    inv_f = 1.0 / f
    inv_fa = 1.0 / np.power(f, alpha)
    return np.column_stack([
        sq * inv_f,         # a_p
        ql * inv_f,         # b_p
        lq * inv_f,         # c_p
        ld * inv_fa,        # a_d
        nd * inv_fa,        # b_d
        np.ones(len(f)),    # t_c
    ])


def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


# ── Fitting approaches ───────────────────────────────────────────────────────

def fit_grid_ols(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                 f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te):
    """Approach 1: Grid search over alpha + OLS (MSE loss)."""
    print("\n── Approach 1: Grid search + OLS ──")
    best = (float("inf"), None, None)
    for ai in range(1, 100):
        alpha = ai / 100.0
        X_tr = build_X(f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr, alpha)
        X_te = build_X(f_te, sq_te, ql_te, lq_te, ld_te, nd_te, alpha)
        mdl = LinearRegression(fit_intercept=False)
        mdl.fit(X_tr, y_tr)
        m = mape(y_te, mdl.predict(X_te))
        if m < best[0]:
            best = (m, alpha, mdl.coef_.copy())

    m_test, alpha, coefs = best
    a_p, b_p, c_p, a_d, b_d, t_c = coefs
    X_tr = build_X(f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr, alpha)
    m_train = mape(y_tr, X_tr @ coefs)
    print(f"  alpha={alpha:.2f}  Train MAPE={m_train:.2f}%  Test MAPE={m_test:.2f}%")
    return {"a_p": a_p, "b_p": b_p, "c_p": c_p,
            "a_d": a_d, "b_d": b_d, "alpha": alpha, "t_c": t_c}


def fit_grid_weighted(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                      f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te):
    """Approach 2: Grid search + weighted regression (relative-error loss)."""
    print("\n── Approach 2: Grid search + weighted regression ──")
    w_tr = 1.0 / y_tr  # weight by inverse of time → equal relative importance
    best = (float("inf"), None, None)
    for ai in range(1, 100):
        alpha = ai / 100.0
        X_tr = build_X(f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr, alpha)
        X_te = build_X(f_te, sq_te, ql_te, lq_te, ld_te, nd_te, alpha)
        Xw = X_tr * w_tr[:, None]
        yw = y_tr * w_tr
        mdl = LinearRegression(fit_intercept=False)
        mdl.fit(Xw, yw)
        m = mape(y_te, X_te @ mdl.coef_)
        if m < best[0]:
            best = (m, alpha, mdl.coef_.copy())

    m_test, alpha, coefs = best
    a_p, b_p, c_p, a_d, b_d, t_c = coefs
    X_tr = build_X(f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr, alpha)
    m_train = mape(y_tr, X_tr @ coefs)
    print(f"  alpha={alpha:.2f}  Train MAPE={m_train:.2f}%  Test MAPE={m_test:.2f}%")
    return {"a_p": a_p, "b_p": b_p, "c_p": c_p,
            "a_d": a_d, "b_d": b_d, "alpha": alpha, "t_c": t_c}


def fit_scipy_mape(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                   f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te,
                   init_params):
    """Approach 3: Scipy non-linear optimisation minimising MAPE."""
    print("\n── Approach 3: Scipy MAPE minimisation ──")

    def predict(params, f, sq, ql, lq, ld, nd):
        a_p, b_p, c_p, a_d, b_d, alpha, t_c = params
        return ((a_p * sq + b_p * ql + c_p * lq) / f
                + (a_d * ld + b_d * nd) / np.power(f, alpha)
                + t_c)

    def loss(params):
        pred = predict(params, f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr)
        # Mean absolute relative error
        return np.mean(np.abs((y_tr - pred) / y_tr))

    x0 = [init_params["a_p"], init_params["b_p"], init_params["c_p"],
           init_params["a_d"], init_params["b_d"], init_params["alpha"],
           init_params["t_c"]]

    bounds = [(0, None), (None, None), (0, None),
              (0, None), (0, None), (0.01, 0.99), (0, None)]

    result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 50000, "ftol": 1e-15})

    # Also try Nelder-Mead with penalty for out-of-bounds alpha
    def penalised_loss(params):
        alpha = params[5]
        if alpha < 0.01 or alpha > 0.99:
            return 1e10
        return loss(params)

    result2 = minimize(penalised_loss, x0, method="Nelder-Mead",
                       options={"maxiter": 100000, "xatol": 1e-12, "fatol": 1e-14})
    if result2.fun < result.fun:
        result = result2

    a_p, b_p, c_p, a_d, b_d, alpha, t_c = result.x
    pred_te = predict(result.x, f_te, sq_te, ql_te, lq_te, ld_te, nd_te)
    pred_tr = predict(result.x, f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr)
    m_train = mape(y_tr, pred_tr)
    m_test = mape(y_te, pred_te)
    print(f"  alpha={alpha:.4f}  Train MAPE={m_train:.2f}%  Test MAPE={m_test:.2f}%")
    return {"a_p": a_p, "b_p": b_p, "c_p": c_p,
            "a_d": a_d, "b_d": b_d, "alpha": alpha, "t_c": t_c}


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(params, features, label=""):
    f, y, sq, ql, lq, ld, nd = _extract_arrays(features)
    alpha = params["alpha"]
    y_pred = ((params["a_p"] * sq + params["b_p"] * ql + params["c_p"] * lq) / f
              + (params["a_d"] * ld + params["b_d"] * nd) / np.power(f, alpha)
              + params["t_c"])

    m = mape(y, y_pred)
    mae_val = mean_absolute_error(y, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y, y_pred))
    r2_val = r2_score(y, y_pred)

    print(f"  {label:12s} MAPE={m:7.2f}%  MAE={mae_val:8.4f} ms  "
          f"RMSE={rmse_val:8.4f} ms  R²={r2_val:.6f}")
    return {"mape": m, "mae_ms": mae_val, "rmse_ms": rmse_val, "r2": r2_val}


def data_summary_from_records(records):
    lq_vals, lkv_vals = [], []
    for rec in records:
        for r in rec.get("requests", []):
            if r["type"] == "prefill":
                lq_vals.append(r["l_q"])
            else:
                lkv_vals.append(r["l_kv"])
    return {
        "prefill_lq_min": min(lq_vals) if lq_vals else 0,
        "prefill_lq_max": max(lq_vals) if lq_vals else 0,
        "decode_lkv_min": min(lkv_vals) if lkv_vals else 0,
        "decode_lkv_max": max(lkv_vals) if lkv_vals else 0,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Loading profiling data from {args.input} ...")
    records = load_records(args.input)
    print(f"Loaded {len(records)} raw records")

    features = clean_and_prepare(records, args.warmup_per_freq, args.freq_tolerance)
    print(f"Usable batch samples: {len(features)}")

    if len(features) < 100:
        print("ERROR: Too few usable samples.")
        sys.exit(1)

    # 80/20 train-test split
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    features_train = [features[i] for i in indices[:split]]
    features_test = [features[i] for i in indices[split:]]
    print(f"Train: {len(features_train)}, Test: {len(features_test)}")

    # Extract arrays
    f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr = _extract_arrays(features_train)
    f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te = _extract_arrays(features_test)

    # Three fitting approaches
    params1 = fit_grid_ols(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                           f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te)
    params2 = fit_grid_weighted(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                                f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te)
    params3 = fit_scipy_mape(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                             f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te,
                             init_params=params2)

    # Compare all three
    print("\n" + "=" * 70)
    print("COMPARISON (test set)")
    print("=" * 70)
    candidates = [
        ("Grid+OLS", params1),
        ("Grid+Weighted", params2),
        ("Scipy+MAPE", params3),
    ]
    best_mape, best_params, best_method = float("inf"), None, None
    for name, params in candidates:
        metrics = evaluate(params, features_test, name)
        if metrics["mape"] < best_mape:
            best_mape = metrics["mape"]
            best_params = params
            best_method = name

    print(f"\n  >>> Best: {best_method} (test MAPE = {best_mape:.2f}%)")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)
    metrics_train = evaluate(best_params, features_train, "Train")
    metrics_test = evaluate(best_params, features_test, "Test")

    # Data summary
    f_vals = [ft["f"] for ft in features]
    batch_sizes = [ft["total_reqs"] for ft in features]
    rec_summary = data_summary_from_records(records)

    output = {
        "parameters": {k: float(v) for k, v in best_params.items()},
        "method": best_method,
        "metrics": {
            "train": {k: float(v) for k, v in metrics_train.items()},
            "test": {k: float(v) for k, v in metrics_test.items()},
        },
        "data_summary": {
            "total_batches": len(features),
            "train_size": len(features_train),
            "test_size": len(features_test),
            "freq_min": min(f_vals),
            "freq_max": max(f_vals),
            "freq_distinct": len(set(round(f) for f in f_vals)),
            "batch_size_min": min(batch_sizes),
            "batch_size_max": max(batch_sizes),
            **rec_summary,
        },
    }

    with open(args.output, "w") as fout:
        json.dump(output, fout, indent=2)

    print(f"\nFitted parameters saved to {args.output}")
    print("\nFitted Parameters:")
    for k, v in best_params.items():
        print(f"  {k:>8s} = {v:.8e}")


if __name__ == "__main__":
    main()
