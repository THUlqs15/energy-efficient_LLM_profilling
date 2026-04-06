# vLLM Batch Latency Profiling — Complete Reproducible Workflow

This document describes how to reproduce the batch execution time profiling and model fitting on a new server.

## System Requirements

- NVIDIA GPU (tested on A10G)
- CUDA 12.x or 13.x
- conda with `myvllm` environment (see setup below)
- `sudo` access for nvidia-smi clock control (optional)

## Repository Layout

```
/home/ubuntu/lqs/
├── vllm/                          # vLLM source (editable install)
│   ├── profiling_logger.py        # Profiling data collection module
│   └── vllm/
│       └── v1/worker/
│           └── gpu_model_runner.py  # Instrumented with timing hooks
├── L3/                            # Qwen3-8B model weights
├── profiling_script.py            # Data collection script
├── fitting_script.py              # Regression fitting script
├── profiling_data.jsonl           # Collected batch records (output)
├── fitting_results.json           # Fitted parameters (output)
├── result.md                      # Summary of results
└── profiling.md                   # This file
```

## Setup

### 1. Create conda environment

```bash
conda create -n myvllm python=3.10
conda activate myvllm
```

### 2. Install vLLM (editable from source)

```bash
cd /home/ubuntu/lqs/vllm
VLLM_USE_PRECOMPILED=1 pip install -e . --torch-backend=auto
```

### 3. Install additional dependencies

```bash
conda run -n myvllm pip install scikit-learn scipy
```

## Phase 1: vLLM Codebase Instrumentation

### Key files involved

| File | Purpose |
|------|---------|
| `vllm/v1/worker/gpu_model_runner.py` | Main model runner; execute_model() method |
| `vllm/v1/worker/gpu_input_batch.py` | InputBatch — per-request state |
| `vllm/v1/core/sched/output.py` | SchedulerOutput — batch descriptor |

### Where execute_model lives

The forward pass is in `GPUModelRunner.execute_model()` at line ~3770 in `gpu_model_runner.py`. The actual GPU forward pass is at line ~4034:

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

### How per-request info is extracted

From `scheduler_output.num_scheduled_tokens` (dict: req_id → l_q) and  
`input_batch.num_computed_tokens_cpu[i]` (l_kv = tokens already in KV cache) and  
`input_batch.num_prompt_tokens[i]` (total prompt tokens, used to determine prefill vs decode).

A request is **prefill** if `l_kv < num_prompt_tokens` (still processing the prompt).  
A request is **decode** if `l_kv >= num_prompt_tokens` (generating output tokens).

## Phase 2: Profiling Logger (`/home/ubuntu/lqs/vllm/profiling_logger.py`)

Toggle via environment variable:
```bash
export VLLM_PROFILING=1
export VLLM_PROFILING_OUTPUT=/home/ubuntu/lqs/profiling_data.jsonl
```

Output format (JSONL):
```json
{"batch_id": 0, "gpu_freq_mhz": 1710, "wall_time_ms": 64.2, "requests": [
    {"type": "prefill", "l_q": 128, "l_kv": 0},
    {"type": "decode", "l_q": 1, "l_kv": 256}
]}
```

The logger reads GPU frequency via:
```bash
nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits
```

## Phase 3: Instrumentation in gpu_model_runner.py

The following changes were made to `vllm/v1/worker/gpu_model_runner.py`:

### Import (after line 23)

```python
# Profiling instrumentation
try:
    import sys as _sys
    import os as _os
    _worker_dir = _os.path.abspath(__file__)
    _profiling_dir = _os.path.dirname(
        _os.path.dirname(
            _os.path.dirname(
                _os.path.dirname(_worker_dir)
            )
        )
    )
    if _profiling_dir not in _sys.path:
        _sys.path.insert(0, _profiling_dir)
    import profiling_logger as _profiling_logger
    _PROFILING_AVAILABLE = True
except ImportError:
    _PROFILING_AVAILABLE = False
```

### Timing hooks around forward pass (inside execute_model)

```python
# --- Profiling: pre-forward ---
_do_profile = _PROFILING_AVAILABLE and _profiling_logger.is_enabled()
if _do_profile:
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    _gpu_freq = _profiling_logger.get_gpu_freq_mhz()
    _req_info = _profiling_logger.extract_request_info(
        scheduler_output, self.input_batch
    )

with set_forward_context(...):
    model_output = self._model_forward(...)

# --- Profiling: post-forward ---
if _do_profile:
    torch.cuda.synchronize()
    _t1 = time.perf_counter()
    _wall_ms = (_t1 - _t0) * 1000.0
    _profiling_logger.log_batch(
        wall_time_ms=_wall_ms,
        gpu_freq_mhz=_gpu_freq,
        requests=_req_info,
    )
```

## Phase 4: GPU Frequency Control

### On systems that support clock locking

```bash
# List supported frequencies
nvidia-smi -q -d SUPPORTED_CLOCKS

# Lock to specific frequency
sudo nvidia-smi --lock-gpu-clocks=<freq>,<freq>

# Set application clocks
sudo nvidia-smi -ac <mem_freq>,<gr_freq>

# Reset
sudo nvidia-smi --reset-gpu-clocks
sudo nvidia-smi -rac
```

### On AWS A10G instances (this system)

The A10G in AWS does NOT honor clock locking — the GPU remains at 1710 MHz regardless:
```
$ sudo nvidia-smi --lock-gpu-clocks=210,210
GPU clocks set to "(gpuClkMin 210, gpuClkMax 210)"  ← command succeeds
$ nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits
1710  ← but actual clock unchanged
```

This is a known AWS restriction. To work around:
1. Collect all data at 1710 MHz (what we did)
2. Fix alpha to a theoretical value (0.5–1.0 for memory-bound decode)
3. Use a different instance type (e.g., bare metal) for multi-frequency experiments

## Phase 5: Running the Profiling Script

### Important: Path Fix

Due to the vLLM editable install, running from `/home/ubuntu/lqs/` causes Python to find `vllm` as a namespace package (the project root directory) instead of the actual package with `__init__.py`. The fix is to remove `''` from `sys.path`:

```python
import sys
if '' in sys.path:
    sys.path.remove('')
```

Always run from a different working directory:
```bash
cd /tmp && conda run -n myvllm python /home/ubuntu/lqs/profiling_script.py
```

### Command

```bash
cd /tmp
conda run -n myvllm python /home/ubuntu/lqs/profiling_script.py 2>&1 | tee /home/ubuntu/lqs/profiling_run.log
```

### What the script does

1. Detects supported GPU frequencies
2. Attempts to lock each selected frequency (6 evenly-spaced from range)
3. If clock locking fails, profiles at natural frequency
4. For each frequency session:
   - Initializes `vllm.entrypoints.llm.LLM` engine (enforce_eager=True)
   - Runs 3 phases:
     - **Phase 1**: Pure prefill batches (max_tokens=1, varying prompt sizes 8–2048 tokens)
     - **Phase 2**: Decode-heavy batches (short prompts, max_tokens=32–256, batch sizes 1–16)
     - **Phase 3**: Mixed random configurations
   - Discards first 15 batches as warmup
   - Collects 400+ samples per frequency

### Profiling data format verification

```bash
python3 -c "
import json
with open('/home/ubuntu/lqs/profiling_data.jsonl') as f:
    recs = [json.loads(l) for l in f]
print(f'Records: {len(recs)}')
print('Sample:', recs[0])
"
```

## Phase 6: Running the Fitting Script

```bash
cd /tmp
conda run -n myvllm python /home/ubuntu/lqs/fitting_script.py
```

### What the fitting script does

1. Loads profiling JSONL data
2. Computes aggregate features per batch:
   - `sum_lq_sq = Σ l_q²` (prefill requests)
   - `sum_lq_lkv = Σ l_q * l_kv` (prefill requests)
   - `sum_lq = Σ l_q` (prefill requests)
   - `sum_lkv_decode = Σ l_kv` (decode requests)
   - `num_decode = count of decode requests`
3. Random 80/20 train/test split (seed=42)
4. **Grid search** over alpha ∈ [0.01, 1.0] with LinearRegression for other params
5. **Refinement** with L-BFGS-B (bounded, non-negative coefficients) using multiple starting points
6. Reports MAPE, MAE, RMSE, R² on both splits
7. Saves results to `fitting_results.json`

### Model formula

```
T_pd(f, B) = (1/f)       * [a_p * Σ l_q²  +  b_p * Σ l_q·l_kv  +  c_p * Σ l_q]
           + (1/f^alpha)  * [a_d * Σ l_kv_decode  +  b_d * num_decode]
           + t_c
```

- f in MHz
- time in ms
- 7 parameters: {a_p, b_p, c_p, a_d, b_d, alpha, t_c}

## Complete Single-Command Reproduction

```bash
# Step 1: Collect profiling data (takes ~2 minutes)
cd /tmp && conda run -n myvllm python /home/ubuntu/lqs/profiling_script.py

# Step 2: Fit the model (takes ~1 minute)
cd /tmp && conda run -n myvllm python /home/ubuntu/lqs/fitting_script.py
```

## Troubleshooting

### "cannot import name 'SamplingParams' from 'vllm'"
Run from a directory that is NOT `/home/ubuntu/lqs/`. The profiling_script.py automatically removes `''` from sys.path to fix this, but you must use `cd /tmp` first.

### "No profiling file created"
Check that `VLLM_PROFILING=1` is set before importing vllm. The EngineCore subprocess inherits environment variables from the parent process.

### "EngineArgs.__init__() got an unexpected keyword argument"
Use `vllm.entrypoints.llm.LLM`, not `vllm.LLM`. Do not pass `disable_log_requests` or `disable_log_stats` — these are not supported in vllm 0.19.0.

### OOM (Out of Memory)
Reduce `gpu_memory_utilization` (default 0.85) or `max_model_len` (default 2048).

### GPU clock doesn't change
On AWS cloud instances, clock control is restricted. The profiling will automatically fall back to natural frequency and log the actual GPU frequency per batch record.

## Notes on Fitting Quality

### Single-frequency limitation
With only one GPU frequency, `alpha` is **not identifiable** — any alpha value gives the same predictions when f is fixed. The fitted alpha = 0.817 is plausible but unverified. To get a reliable alpha estimate, you need at least 2-3 distinct GPU frequencies.

### b_p = 0 (cross-term)
All prefill requests in our dataset have `l_kv = 0` (fresh prefills). The `b_p` term (Σ l_q·l_kv) only contributes when prefilling requests that already have cached KV context (e.g., multi-turn conversations). With fresh prompts only, b_p cannot be estimated.

### Large t_c = 57ms
The constant overhead of ~57ms per batch is larger than typical CUDA kernel overhead. In vLLM v1, the LLM engine uses an inter-process communication (IPC) architecture where the `EngineCore` runs in a separate subprocess. Scheduling decisions and result communication add measurable latency on top of pure GPU time.
