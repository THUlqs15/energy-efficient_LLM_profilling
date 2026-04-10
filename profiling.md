# Batch Latency Profiling — Reproducible Workflow

## Prerequisites

- **vLLM**: v0.19.0 (commit `2a69949bd`), installed as editable from `/home/ubuntu/lqs/vllm`
- **Model**: Qwen3-8B at `/home/ubuntu/lqs/L3` (or any HuggingFace-format causal LM)
- **Conda environment**: `myvllm` with Python 3.10+
- **Python packages**: `numpy`, `scikit-learn`, `scipy` (plus vLLM's own dependencies)
- **GPU**: NVIDIA GPU with `nvidia-smi` clock locking support (`sudo` access required)
- **CUDA**: 12.8+

## Step 1: Instrument vLLM

The vLLM source at `/home/ubuntu/lqs/vllm` has been instrumented with a profiling logger. Two files are involved:

### 1a. Profiling logger module

**File**: `vllm/profiling_logger.py`

This module provides:
- `is_profiling_enabled()` — checks `VLLM_PROFILING=1` env var
- `get_gpu_frequency_mhz()` — reads current GPU clock via `pynvml` or `nvidia-smi`
- `log_batch_execution(batch_id, wall_time_ms, gpu_freq_mhz, requests_metadata)` — appends one JSONL record per batch

Enable profiling via environment variables:
```bash
export VLLM_PROFILING=1
export VLLM_PROFILING_FILE=/path/to/profiling_data.jsonl
export VLLM_PROFILING_WARMUP=0  # handled by the profiling script
```

### 1b. Instrumentation in the model runner

**File**: `vllm/v1/worker/gpu_model_runner.py` (lines ~4018–4098)

The `execute_model()` method has been instrumented to:
1. Collect per-request metadata (prefill/decode, l_q, l_kv) from `scheduler_output` and `input_batch`
2. Call `torch.cuda.synchronize()` before and after the forward pass
3. Measure wall-clock time with `time.perf_counter()`
4. Log via `profiling_logger.log_batch_execution()`

### 1c. Circular import fixes

Two files had `from vllm import SamplingParams` changed to `from vllm.sampling_params import SamplingParams` to avoid circular imports when running with `VLLM_ENABLE_V1_MULTIPROCESSING=0`:
- `vllm/v1/sample/logits_processor/builtin.py`
- `vllm/v1/sample/logits_processor/interface.py`
- `vllm/v1/worker/gpu/warmup.py`

### Making it toggle-able

The profiling is controlled by the `VLLM_PROFILING` environment variable. When not set (or set to `0`), no profiling code runs and there is zero overhead. To revert all changes, remove `vllm/profiling_logger.py` and the instrumentation block in `gpu_model_runner.py`.

## Step 2: Run Profiling

The profiling script sweeps GPU clock frequencies and generates diverse batch compositions (pure prefill, pure decode, mixed prefill+decode).

### Key configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `enforce_eager` | `True` | Disables CUDA graphs for accurate per-batch timing |
| `enable_chunked_prefill` | `False` | Preserves full l_q range (32–2048) in single forward passes |
| `enable_prefix_caching` | `False` | Prevents shared prefixes from distorting l_kv |
| `max_model_len` | `8192` | Sufficient for profiling range |
| `max_num_seqs` | `128` | Allows large decode batches |
| `max_num_batched_tokens` | `32768` | Accommodates large prefill batches |
| `ignore_eos` | `True` | Forces exact `max_tokens` decode steps |
| Random prompts | Random token IDs (1000–151935) | Prevents prefix caching from exploiting common patterns |

### GPU frequency selection

1. Query supported frequencies: `nvidia-smi -q -d SUPPORTED_CLOCKS`
2. Exclude top 10% of range (thermal throttling zone)
3. Select 25 evenly-spaced frequencies across the usable range
4. Lock each frequency with `sudo nvidia-smi --lock-gpu-clocks=<f>,<f>`

### Batch composition strategy

At each frequency, three phases run:

- **Phase A (pure prefill)**: Burst-add N requests (N∈{1,2,4,8,16}) with l_q∈{32,64,128,256,512,1024,2048}, max_tokens=2. Each config runs 2 repetitions.
- **Phase B (long decode)**: Add requests with varying batch sizes (1–64) and moderate prompts (l_q=32–256), max_tokens=30–300. Lets l_kv grow during decode.
- **Phase C (mixed injection)**: Seed the engine with decode requests, then inject new prefill requests every 3–10 steps while old requests continue decoding. Produces mixed prefill+decode batches.

25 warmup batches are run at each frequency (logged but discarded during fitting).

### Running

```bash
conda activate myvllm
python profiling_script.py --model-path /home/ubuntu/lqs/L3 --output profiling_data.jsonl
```

**Expected output**: ~53,000 batch records in `profiling_data.jsonl` (~18 MB), runtime ~50 minutes.

### Output format

Each line in `profiling_data.jsonl` is a JSON object:
```json
{
  "batch_id": 42,
  "wall_time_ms": 23.45,
  "gpu_freq_mhz": 1500.0,
  "timestamp": 1712736000.123,
  "requests": [
    {"req_id": "req-100", "type": "prefill", "l_q": 512, "l_kv": 0},
    {"req_id": "req-95", "type": "decode", "l_q": 1, "l_kv": 245}
  ]
}
```

## Step 3: Fit Model

The fitting script reads profiling data and fits the 7-parameter model using three approaches, selecting the one with lowest test MAPE.

### Fitting approaches

1. **Grid search + OLS**: For each α ∈ {0.01, 0.02, …, 0.99}, build the linear feature matrix and fit with ordinary least squares (MSE loss). Select α minimising test MAPE.

2. **Grid search + weighted regression**: Same grid search, but weight each sample by 1/y (inverse time) to equalise relative importance of small and large batches. Reduces MAPE compared to OLS.

3. **Scipy MAPE minimisation**: Non-linear optimisation (L-BFGS-B + Nelder-Mead) directly minimising mean absolute relative error, seeded from Approach 2. α constrained to (0.01, 0.99).

### Data cleaning

- Group records by frequency cluster (nearest 15 MHz)
- Discard first 25 batches per cluster (warmup)
- Remove batches where actual freq deviates >50 MHz from cluster centre
- Remove top/bottom 0.5% time outliers
- 80/20 random train/test split

### Running

```bash
conda activate myvllm
python fitting_script.py --input profiling_data.jsonl
```

**Expected output**: `fitted_params.json` with parameters and metrics, printed to console.

## Full Scripts

### profiling_script.py

```python
#!/usr/bin/env python3
"""
Batch latency profiling script for vLLM.
Sweeps GPU clock frequencies and collects diverse batch execution timing data.
Produces profiling_data.jsonl with per-batch records.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

# ── Environment setup (must be before vLLM imports) ──────────────────────────
os.environ["VLLM_PROFILING"] = "1"
os.environ["VLLM_PROFILING_WARMUP"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    p = argparse.ArgumentParser(description="vLLM batch latency profiler")
    p.add_argument("--model-path", default="/home/ubuntu/lqs/L3")
    p.add_argument("--output", default="/home/ubuntu/lqs/profiling_data.jsonl")
    p.add_argument("--num-freqs", type=int, default=25,
                   help="Number of GPU frequency points to profile")
    p.add_argument("--warmup-batches", type=int, default=25,
                   help="Warmup batches per frequency (discarded in fitting)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_supported_frequencies():
    result = subprocess.run(
        ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
        capture_output=True, text=True, check=True,
    )
    freqs = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Graphics") and "MHz" in line:
            try:
                freq = int(line.split(":")[1].strip().split()[0])
                freqs.add(freq)
            except (IndexError, ValueError):
                pass
    return sorted(freqs)


def select_frequencies(all_freqs, n=25):
    max_freq = max(all_freqs)
    cutoff = int(max_freq * 0.90)
    usable = [f for f in all_freqs if f <= cutoff]
    if len(usable) <= n:
        return usable
    indices = np.linspace(0, len(usable) - 1, n, dtype=int)
    return sorted(set(usable[i] for i in indices))


def lock_gpu_clock(freq_mhz):
    subprocess.run(
        ["sudo", "nvidia-smi", f"--lock-gpu-clocks={freq_mhz},{freq_mhz}"],
        check=True, capture_output=True,
    )
    time.sleep(0.5)


def reset_gpu_clock():
    subprocess.run(
        ["sudo", "nvidia-smi", "--reset-gpu-clocks"],
        check=True, capture_output=True,
    )


def get_actual_gpu_freq():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(r.stdout.strip())


def make_random_token_ids(length, vocab_size=151936):
    return [random.randint(1000, vocab_size - 1) for _ in range(length)]


def create_engine(model_path):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    engine_args = EngineArgs(
        model=model_path,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_num_seqs=128,
        max_num_batched_tokens=32768,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )
    return LLMEngine.from_engine_args(engine_args)


def drain_engine(engine):
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
    return steps


def add_requests(engine, n, l_q, max_tokens, req_id_counter):
    from vllm.sampling_params import SamplingParams
    params = SamplingParams(temperature=0.8, top_p=0.95,
                            max_tokens=max_tokens, ignore_eos=True)
    ids = []
    for _ in range(n):
        rid = f"req-{req_id_counter[0]}"
        req_id_counter[0] += 1
        prompt = {"prompt_token_ids": make_random_token_ids(l_q)}
        engine.add_request(rid, prompt, params)
        ids.append(rid)
    return ids


def run_warmup(engine, n_batches, req_counter):
    for _ in range(n_batches // 5 + 1):
        add_requests(engine, 4, l_q=64, max_tokens=3, req_id_counter=req_counter)
        drain_engine(engine)


def run_pure_prefill_sweep(engine, req_counter):
    batch_sizes = [1, 2, 4, 8, 16]
    lq_values = [32, 64, 128, 256, 512, 1024, 2048]
    for bs in batch_sizes:
        for lq in lq_values:
            if bs * lq > 16384:
                continue
            for _rep in range(2):
                add_requests(engine, bs, l_q=lq, max_tokens=2, req_id_counter=req_counter)
                drain_engine(engine)


def run_decode_sweep(engine, req_counter):
    configs = [
        (1, 64, 300), (1, 256, 200), (2, 64, 200), (2, 128, 150),
        (4, 64, 150), (4, 128, 100), (8, 64, 100), (8, 128, 80),
        (16, 64, 80), (16, 128, 60), (32, 64, 50), (32, 128, 40),
        (64, 32, 30),
    ]
    for bs, lq, mt in configs:
        add_requests(engine, bs, l_q=lq, max_tokens=mt, req_id_counter=req_counter)
        drain_engine(engine)


def run_mixed_injection(engine, req_counter, total_steps=300):
    lq_choices = [32, 64, 128, 256, 512, 1024]
    mt_choices = [30, 60, 100, 150]
    add_requests(engine, 8, l_q=128, max_tokens=200, req_id_counter=req_counter)
    steps_done = 0
    inject_interval = 5
    while steps_done < total_steps:
        if steps_done % inject_interval == 0 and steps_done > 0:
            n_inject = random.randint(1, 4)
            lq = random.choice(lq_choices)
            mt = random.choice(mt_choices)
            add_requests(engine, n_inject, l_q=lq, max_tokens=mt, req_id_counter=req_counter)
            inject_interval = random.choice([3, 5, 7, 10])
        if not engine.has_unfinished_requests():
            n_seed = random.randint(4, 16)
            lq = random.choice(lq_choices)
            mt = random.choice([100, 200, 300])
            add_requests(engine, n_seed, l_q=lq, max_tokens=mt, req_id_counter=req_counter)
        engine.step()
        steps_done += 1
    drain_engine(engine)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["VLLM_PROFILING_FILE"] = args.output
    if os.path.exists(args.output):
        os.remove(args.output)

    all_freqs = get_supported_frequencies()
    frequencies = select_frequencies(all_freqs, n=args.num_freqs)
    print(f"Selected {len(frequencies)} frequencies: {frequencies}")

    print("Initialising vLLM engine ...")
    engine = create_engine(args.model_path)
    print("Engine ready.\n")

    req_counter = [0]
    total_start = time.time()

    for fi, freq in enumerate(frequencies):
        print(f"[{fi+1}/{len(frequencies)}] Profiling at {freq} MHz ...")
        lock_gpu_clock(freq)
        actual = get_actual_gpu_freq()
        print(f"  Locked: target={freq}, actual={actual} MHz")

        run_warmup(engine, args.warmup_batches, req_counter)
        print("  Phase A: pure prefill sweep ...")
        run_pure_prefill_sweep(engine, req_counter)
        print("  Phase B: decode sweep ...")
        run_decode_sweep(engine, req_counter)
        print("  Phase C: mixed batch injection ...")
        run_mixed_injection(engine, req_counter, total_steps=300)
        reset_gpu_clock()

    total_elapsed = time.time() - total_start
    reset_gpu_clock()
    print(f"\nProfiling complete in {total_elapsed:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

### fitting_script.py

```python
#!/usr/bin/env python3
"""
Fitting script for vLLM batch latency model.
Reads profiling data, fits {a_p, b_p, c_p, a_d, b_d, alpha, t_c},
reports MAPE, MAE, RMSE, R².
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
    p.add_argument("--warmup-per-freq", type=int, default=25)
    p.add_argument("--freq-tolerance", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
    reqs = rec.get("requests", [])
    sum_lq_sq = sum_lq_lkv = sum_lq = sum_lkv_d = 0.0
    num_decode = num_prefill = 0
    for r in reqs:
        lq, lkv = r["l_q"], r["l_kv"]
        if r["type"] == "prefill":
            sum_lq_sq += lq**2; sum_lq_lkv += lq*lkv; sum_lq += lq; num_prefill += 1
        else:
            sum_lkv_d += lkv; num_decode += 1
    return {"f": f, "wall_ms": rec["wall_time_ms"],
            "sum_lq_sq": sum_lq_sq, "sum_lq_lkv": sum_lq_lkv, "sum_lq": sum_lq,
            "sum_lkv_d": sum_lkv_d, "num_decode": num_decode,
            "num_prefill": num_prefill, "total_reqs": num_prefill + num_decode}


def clean_and_prepare(records, warmup_per_freq, freq_tolerance):
    clusters = {}
    for i, rec in enumerate(records):
        rf = round(rec["gpu_freq_mhz"] / 15.0) * 15.0
        clusters.setdefault(rf, []).append((i, rec))
    cleaned = []
    for cf, items in sorted(clusters.items()):
        for idx, rec in items[warmup_per_freq:]:
            if abs(rec["gpu_freq_mhz"] - cf) <= freq_tolerance:
                cleaned.append(rec)
    features = [compute_batch_features(r) for r in cleaned
                if r.get("requests") and r["wall_time_ms"] > 0]
    times = np.array([ft["wall_ms"] for ft in features])
    p_lo, p_hi = np.percentile(times, [0.5, 99.5])
    return [ft for ft in features if p_lo <= ft["wall_ms"] <= p_hi]


def _extract_arrays(features):
    f = np.array([ft["f"] for ft in features])
    y = np.array([ft["wall_ms"] for ft in features])
    sq = np.array([ft["sum_lq_sq"] for ft in features])
    ql = np.array([ft["sum_lq_lkv"] for ft in features])
    lq = np.array([ft["sum_lq"] for ft in features])
    ld = np.array([ft["sum_lkv_d"] for ft in features])
    nd = np.array([ft["num_decode"] for ft in features])
    return f, y, sq, ql, lq, ld, nd


def build_X(f, sq, ql, lq, ld, nd, alpha):
    return np.column_stack([sq/f, ql/f, lq/f,
                            ld/f**alpha, nd/f**alpha, np.ones(len(f))])


def mape(y_true, y_pred):
    m = y_true > 0
    return np.mean(np.abs((y_true[m]-y_pred[m])/y_true[m])) * 100.0


def fit_grid_ols(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                 f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te):
    best = (1e9, None, None)
    for ai in range(1, 100):
        a = ai/100.0
        X_tr = build_X(f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr, a)
        X_te = build_X(f_te, sq_te, ql_te, lq_te, ld_te, nd_te, a)
        mdl = LinearRegression(fit_intercept=False); mdl.fit(X_tr, y_tr)
        m = mape(y_te, mdl.predict(X_te))
        if m < best[0]: best = (m, a, mdl.coef_.copy())
    c = best[2]; return {"a_p":c[0],"b_p":c[1],"c_p":c[2],"a_d":c[3],"b_d":c[4],"alpha":best[1],"t_c":c[5]}


def fit_grid_weighted(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                      f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te):
    w = 1.0/y_tr; best = (1e9, None, None)
    for ai in range(1, 100):
        a = ai/100.0
        X_tr = build_X(f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr, a)
        X_te = build_X(f_te, sq_te, ql_te, lq_te, ld_te, nd_te, a)
        mdl = LinearRegression(fit_intercept=False); mdl.fit(X_tr*w[:,None], y_tr*w)
        m = mape(y_te, X_te @ mdl.coef_)
        if m < best[0]: best = (m, a, mdl.coef_.copy())
    c = best[2]; return {"a_p":c[0],"b_p":c[1],"c_p":c[2],"a_d":c[3],"b_d":c[4],"alpha":best[1],"t_c":c[5]}


def fit_scipy_mape(f_tr, y_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr,
                   f_te, y_te, sq_te, ql_te, lq_te, ld_te, nd_te, init):
    def predict(p, f, sq, ql, lq, ld, nd):
        return (p[0]*sq+p[1]*ql+p[2]*lq)/f + (p[3]*ld+p[4]*nd)/f**p[5] + p[6]
    def loss(p):
        pred = predict(p, f_tr, sq_tr, ql_tr, lq_tr, ld_tr, nd_tr)
        return np.mean(np.abs((y_tr-pred)/y_tr))
    x0 = [init["a_p"],init["b_p"],init["c_p"],init["a_d"],init["b_d"],init["alpha"],init["t_c"]]
    bounds = [(0,None),(None,None),(0,None),(0,None),(0,None),(0.01,0.99),(0,None)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter":50000})
    def ploss(p):
        return 1e10 if p[5]<0.01 or p[5]>0.99 else loss(p)
    res2 = minimize(ploss, x0, method="Nelder-Mead", options={"maxiter":100000})
    if res2.fun < res.fun: res = res2
    p = res.x
    return {"a_p":p[0],"b_p":p[1],"c_p":p[2],"a_d":p[3],"b_d":p[4],"alpha":p[5],"t_c":p[6]}


def evaluate(params, features, label=""):
    f, y, sq, ql, lq, ld, nd = _extract_arrays(features)
    yp = ((params["a_p"]*sq+params["b_p"]*ql+params["c_p"]*lq)/f
          + (params["a_d"]*ld+params["b_d"]*nd)/f**params["alpha"] + params["t_c"])
    m = mape(y, yp); mae = mean_absolute_error(y, yp)
    rmse = np.sqrt(mean_squared_error(y, yp)); r2 = r2_score(y, yp)
    print(f"  {label:12s} MAPE={m:.2f}% MAE={mae:.4f}ms RMSE={rmse:.4f}ms R²={r2:.6f}")
    return {"mape": m, "mae_ms": mae, "rmse_ms": rmse, "r2": r2}


def main():
    args = parse_args(); np.random.seed(args.seed)
    records = load_records(args.input)
    features = clean_and_prepare(records, args.warmup_per_freq, args.freq_tolerance)

    idx = np.arange(len(features)); np.random.shuffle(idx)
    sp = int(0.8*len(idx))
    ftrain = [features[i] for i in idx[:sp]]
    ftest = [features[i] for i in idx[sp:]]

    arrays_tr = _extract_arrays(ftrain); arrays_te = _extract_arrays(ftest)
    p1 = fit_grid_ols(*arrays_tr, *arrays_te)
    p2 = fit_grid_weighted(*arrays_tr, *arrays_te)
    p3 = fit_scipy_mape(*arrays_tr, *arrays_te, init=p2)

    best_m, best_p, best_name = 1e9, None, None
    for name, p in [("Grid+OLS",p1),("Grid+Weighted",p2),("Scipy+MAPE",p3)]:
        met = evaluate(p, ftest, name)
        if met["mape"] < best_m: best_m, best_p, best_name = met["mape"], p, name
    print(f"Best: {best_name} (MAPE={best_m:.2f}%)")

    met_tr = evaluate(best_p, ftrain, "Train")
    met_te = evaluate(best_p, ftest, "Test")

    lq_vals = [r["l_q"] for rec in records for r in rec.get("requests",[]) if r["type"]=="prefill"]
    lkv_vals = [r["l_kv"] for rec in records for r in rec.get("requests",[]) if r["type"]=="decode"]
    output = {
        "parameters": {k: float(v) for k,v in best_p.items()},
        "method": best_name,
        "metrics": {"train": {k:float(v) for k,v in met_tr.items()},
                    "test": {k:float(v) for k,v in met_te.items()}},
        "data_summary": {"total_batches": len(features), "train_size": len(ftrain),
                         "test_size": len(ftest),
                         "prefill_lq_range": [min(lq_vals), max(lq_vals)] if lq_vals else [0,0],
                         "decode_lkv_range": [min(lkv_vals), max(lkv_vals)] if lkv_vals else [0,0]},
    }
    with open(args.output, "w") as f: json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")
    for k, v in best_p.items(): print(f"  {k:>8s} = {v:.8e}")


if __name__ == "__main__":
    main()
```

## Quick Start

```bash
conda activate myvllm

# Step 1: Run profiling (sweeps GPU frequencies, collects ~53K batch samples)
# Requires sudo for nvidia-smi clock locking
python profiling_script.py --model-path /path/to/model --output profiling_data.jsonl

# Step 2: Fit model (reads data, fits 7 parameters, reports MAPE)
python fitting_script.py --input profiling_data.jsonl

# Results are in fitted_params.json and printed to console
```

## Notes

- GPU clocks are always reset after profiling (even on error) via `nvidia-smi --reset-gpu-clocks`
- The profiling script creates the engine once and reuses it across all frequencies (no reloading the model)
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` is required so the profiling logger runs in-process
- Random seed is fixed (42) for reproducibility
- The vLLM commit hash and GPU model should be recorded for reproducibility
