# Batch Latency Profiling — Reproducible Workflow

This document provides a complete, reproducible workflow for profiling vLLM batch execution times and fitting a latency prediction model.

## Prerequisites

- **vLLM**: Installed from source at `/home/ubuntu/lqs/vllm` (v0.19.0 or compatible)
- **Model**: Any HuggingFace-compatible LLM (tested with Qwen3)
- **Python environment**: Conda environment with:
  - PyTorch with CUDA support
  - vLLM dependencies
  - numpy, scipy, scikit-learn
- **GPU**: NVIDIA GPU with clock locking support (tested on RTX 4090)
- **Permissions**: sudo access for GPU clock locking (`nvidia-smi --lock-gpu-clocks`)

### Environment Setup

```bash
# Create conda environment (if not exists)
conda create -n myvllm python=3.10
conda activate myvllm

# Install vLLM from source
cd /home/ubuntu/lqs/vllm
pip install -e .

# Install fitting dependencies
pip install numpy scipy scikit-learn pynvml
```

## Step 1: Instrument vLLM

The vLLM source code has been instrumented to log batch execution data. The key modifications are:

### 1.1 Profiling Logger Module

**File**: `vllm/profiling_logger.py`

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Profiling logger for batch latency estimation.
Logs batch execution time, GPU frequency, and per-request metadata.
Enabled via VLLM_PROFILING environment variable.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

_profiling_enabled = os.getenv("VLLM_PROFILING", "0") == "1"
_profiling_file = os.getenv("VLLM_PROFILING_FILE", "/home/ubuntu/lqs/profiling_data.jsonl")
_batch_counter = 0
_warmup_batches = int(os.getenv("VLLM_PROFILING_WARMUP", "20"))


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled via environment variable."""
    return _profiling_enabled


def get_gpu_frequency_mhz() -> float:
    """Get current GPU graphics clock frequency in MHz."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        freq_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        pynvml.nvmlShutdown()
        return float(freq_mhz)
    except Exception:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip().split()[0])
        except Exception:
            return -1.0


def log_batch_execution(
    batch_id: int,
    wall_time_ms: float,
    gpu_freq_mhz: float,
    requests_metadata: list[dict[str, Any]],
) -> None:
    """Log a single batch execution to the profiling file."""
    global _batch_counter

    if not _profiling_enabled:
        return

    if _batch_counter < _warmup_batches:
        _batch_counter += 1
        return

    record = {
        "batch_id": batch_id,
        "wall_time_ms": wall_time_ms,
        "gpu_freq_mhz": gpu_freq_mhz,
        "requests": requests_metadata,
        "timestamp": time.time(),
    }

    try:
        Path(_profiling_file).parent.mkdir(parents=True, exist_ok=True)
        with open(_profiling_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[Profiling] Failed to log batch {batch_id}: {e}")

    _batch_counter += 1


def reset_batch_counter() -> None:
    """Reset the batch counter (e.g., when changing GPU frequency)."""
    global _batch_counter
    _batch_counter = 0
```

### 1.2 Model Runner Instrumentation

**File**: `vllm/v1/worker/gpu_model_runner.py`

Add import at top of file:
```python
from vllm import profiling_logger
```

Add instrumentation in `execute_model()` method (around line 4018):

```python
# Profiling: Prepare metadata before forward pass
if profiling_logger.is_profiling_enabled():
    profiling_gpu_freq = profiling_logger.get_gpu_frequency_mhz()
    profiling_requests_metadata = []

    for req_id in self.input_batch.req_ids[:num_reqs]:
        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            continue

        l_q = scheduler_output.num_scheduled_tokens.get(req_id, 0)
        if l_q == 0:
            continue

        l_kv = int(self.input_batch.num_computed_tokens_cpu[req_index])
        is_prefill = (l_kv == 0 or
                     any(r.req_id == req_id for r in scheduler_output.scheduled_new_reqs))
        request_type = "prefill" if is_prefill else "decode"

        profiling_requests_metadata.append({
            "req_id": req_id,
            "type": request_type,
            "l_q": l_q,
            "l_kv": l_kv,
        })

# Profiling: Sync GPU and start timer
if profiling_logger.is_profiling_enabled():
    torch.cuda.synchronize()
    profiling_start_time = time.perf_counter()

# ... [forward pass code] ...

# Profiling: Sync GPU and log execution time
if profiling_logger.is_profiling_enabled():
    torch.cuda.synchronize()
    profiling_end_time = time.perf_counter()
    profiling_wall_time_ms = (profiling_end_time - profiling_start_time) * 1000.0

    profiling_logger.log_batch_execution(
        batch_id=getattr(self, '_profiling_batch_id', 0),
        wall_time_ms=profiling_wall_time_ms,
        gpu_freq_mhz=profiling_gpu_freq,
        requests_metadata=profiling_requests_metadata,
    )
    self._profiling_batch_id = getattr(self, '_profiling_batch_id', 0) + 1
```

### 1.3 Enable Profiling (Non-Intrusive)

Profiling is controlled by environment variables:
- `VLLM_PROFILING=1` - Enable profiling
- `VLLM_PROFILING_FILE=/path/to/output.jsonl` - Output file path
- `VLLM_PROFILING_WARMUP=20` - Number of warmup batches to skip

## Step 2: Run Profiling

### Full Profiling Script

**File**: `profiling_script.py`

```python
#!/usr/bin/env python3
"""
Batch Latency Profiling Script for vLLM.
Profiles batch execution latency across GPU frequencies and batch configurations.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def query_supported_gpu_clocks():
    """Query all supported GPU graphics clock frequencies."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
            capture_output=True, text=True, check=True,
        )
        freqs = []
        for line in result.stdout.split("\n"):
            if "Graphics" in line and "MHz" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    freq_str = parts[1].strip().split()[0]
                    try:
                        freqs.append(int(freq_str))
                    except ValueError:
                        pass
        return sorted(set(freqs), reverse=True)
    except Exception as e:
        print(f"Error querying GPU clocks: {e}")
        return []


def select_representative_frequencies(all_freqs, num_freqs=8):
    """Select representative frequencies spanning the full range."""
    if len(all_freqs) <= num_freqs:
        return all_freqs
    percentiles = [0, 10, 25, 40, 60, 75, 90, 100]
    selected_indices = [int(len(all_freqs) * p / 100) for p in percentiles]
    selected_indices = [min(i, len(all_freqs) - 1) for i in selected_indices]
    seen = set()
    result = []
    for freq in [all_freqs[i] for i in selected_indices]:
        if freq not in seen:
            seen.add(freq)
            result.append(freq)
    return result


def lock_gpu_clock(freq_mhz):
    """Lock GPU clock to specified frequency using sudo."""
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "--lock-gpu-clocks", f"{freq_mhz},{freq_mhz}"],
            check=True, capture_output=True,
        )
        print(f"  Locked GPU clock to {freq_mhz} MHz")
        time.sleep(1)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        actual_freq = int(result.stdout.strip())
        if abs(actual_freq - freq_mhz) > 50:
            print(f"  Warning: Requested {freq_mhz} MHz but got {actual_freq} MHz")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error locking GPU clock: {e}")
        return False


def reset_gpu_clock():
    """Reset GPU clock to default using sudo."""
    try:
        subprocess.run(["sudo", "nvidia-smi", "--reset-gpu-clocks"],
                      check=True, capture_output=True)
        print("  Reset GPU clock to default")
    except Exception as e:
        print(f"  Error resetting GPU clock: {e}")


def generate_batch_configurations(max_prefill=16, max_decode=64):
    """Generate diverse batch configurations for profiling."""
    configs = []
    prefill_lens = [32, 64, 128, 256, 512, 1024, 2048]
    decode_kv_lens = [64, 128, 256, 512, 1024, 2048]

    # Pure prefill batches
    for num_pf in [1, 2, 4, 8, 12, 16]:
        if num_pf > max_prefill:
            continue
        for pf_len in prefill_lens:
            configs.append({
                "num_prefill": num_pf, "num_decode": 0,
                "prefill_lengths": [pf_len] * num_pf,
                "decode_kv_lengths": [],
            })

    # Pure decode batches
    for num_dec in [4, 8, 16, 32, 48, 64]:
        if num_dec > max_decode:
            continue
        for kv_len in decode_kv_lens:
            configs.append({
                "num_prefill": 0, "num_decode": num_dec,
                "prefill_lengths": [],
                "decode_kv_lengths": [kv_len] * num_dec,
            })

    # Mixed batches
    for num_pf in [1, 2, 4]:
        for num_dec in [4, 8, 16, 32]:
            if num_pf > max_prefill or num_dec > max_decode:
                continue
            for pf_len in [128, 512, 1024]:
                for kv_len in [256, 512, 1024]:
                    configs.append({
                        "num_prefill": num_pf, "num_decode": num_dec,
                        "prefill_lengths": [pf_len] * num_pf,
                        "decode_kv_lengths": [kv_len] * num_dec,
                    })

    return configs


def profile_at_frequency(freq_mhz, model_path, batch_configs, samples_per_config=3,
                         output_file="profiling_data.jsonl"):
    """Profile batches at a specific GPU frequency."""
    print(f"\n=== Profiling at {freq_mhz} MHz ===")

    if not lock_gpu_clock(freq_mhz):
        return 0

    os.environ["VLLM_PROFILING"] = "1"
    os.environ["VLLM_PROFILING_FILE"] = output_file
    os.environ["VLLM_PROFILING_WARMUP"] = "20"

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            max_model_len=4096,
            max_num_seqs=256,
            enforce_eager=True,
            gpu_memory_utilization=0.9,
        )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        total_samples = 0

        for config in batch_configs:
            prompts = []
            for pf_len in config["prefill_lengths"]:
                prompt = "The " + "quick brown fox jumps over lazy dog. " * (pf_len // 10)
                prompts.append(prompt)
            for kv_len in config["decode_kv_lengths"]:
                prompt = "The " + "word " * (kv_len // 2)
                prompts.append(prompt)

            for _ in range(samples_per_config):
                if prompts:
                    try:
                        llm.generate(prompts, sampling_params)
                        total_samples += 1
                    except Exception:
                        pass

        del llm
        torch.cuda.empty_cache()
        return total_samples

    except Exception as e:
        print(f"  Error: {e}")
        return 0
    finally:
        reset_gpu_clock()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="profiling_data.jsonl")
    parser.add_argument("--num-frequencies", type=int, default=8)
    parser.add_argument("--samples-per-config", type=int, default=3)
    args = parser.parse_args()

    all_freqs = query_supported_gpu_clocks()
    selected_freqs = select_representative_frequencies(all_freqs, args.num_frequencies)
    batch_configs = generate_batch_configurations()

    if Path(args.output).exists():
        Path(args.output).unlink()

    total_samples = 0
    for freq_mhz in selected_freqs:
        samples = profile_at_frequency(
            freq_mhz, args.model_path, batch_configs,
            args.samples_per_config, args.output
        )
        total_samples += samples

    print(f"\nTotal samples: {total_samples}")
    reset_gpu_clock()


if __name__ == "__main__":
    main()
```

## Step 3: Fit Model

### Fitting Script

**File**: `fitting_script.py`

```python
#!/usr/bin/env python3
"""
Batch Latency Model Fitting Script.
Fits: T_pd(f, B) = (1/f) × [a_p × Σl_q² + b_p × Σl_q·l_kv + c_p × Σl_q]
                 + (1/f^α) × [a_d × Σl_kv + b_d × num_decode]
                 + t_c
"""

import argparse
import json
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
    """Compute aggregate features from a batch record."""
    f = batch_record["gpu_freq_mhz"]
    requests = batch_record["requests"]
    wall_time_ms = batch_record["wall_time_ms"]

    sum_lq_sq, sum_lq_lkv, sum_lq = 0.0, 0.0, 0.0
    sum_lkv_decode, num_decode = 0.0, 0

    for req in requests:
        l_q, l_kv = req["l_q"], req["l_kv"]
        if req["type"] == "prefill":
            sum_lq_sq += l_q ** 2
            sum_lq_lkv += l_q * l_kv
            sum_lq += l_q
        else:
            sum_lkv_decode += l_kv
            num_decode += 1

    return {
        "f": f, "sum_lq_sq": sum_lq_sq, "sum_lq_lkv": sum_lq_lkv,
        "sum_lq": sum_lq, "sum_lkv_decode": sum_lkv_decode,
        "num_decode": num_decode, "wall_time_ms": wall_time_ms,
    }


def fit_model_grid_search(X_raw_train, y_train, X_raw_test, y_test):
    """Fit model using grid search over α."""
    best_mape, best_alpha, best_params = float("inf"), None, None

    for alpha in np.arange(0.05, 1.0, 0.01):
        X_train = np.array([[
            feat["sum_lq_sq"] / feat["f"],
            feat["sum_lq_lkv"] / feat["f"],
            feat["sum_lq"] / feat["f"],
            feat["sum_lkv_decode"] / (feat["f"] ** alpha),
            feat["num_decode"] / (feat["f"] ** alpha),
            1.0,
        ] for feat in X_raw_train])

        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)

        X_test = np.array([[
            feat["sum_lq_sq"] / feat["f"],
            feat["sum_lq_lkv"] / feat["f"],
            feat["sum_lq"] / feat["f"],
            feat["sum_lkv_decode"] / (feat["f"] ** alpha),
            feat["num_decode"] / (feat["f"] ** alpha),
            1.0,
        ] for feat in X_raw_test])

        y_pred = model.predict(X_test)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        if mape < best_mape:
            best_mape, best_alpha, best_params = mape, alpha, model.coef_

    return best_params, best_alpha, best_mape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-params", type=str)
    args = parser.parse_args()

    data = load_profiling_data(args.input)
    X_raw = [compute_aggregate_features(b) for b in data
             if b["wall_time_ms"] > 0]
    y = np.array([f["wall_time_ms"] for f in X_raw])

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_raw_train = [X_raw[i] for i in train_idx]
    X_raw_test = [X_raw[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    params, alpha, mape = fit_model_grid_search(X_raw_train, y_train, X_raw_test, y_test)
    a_p, b_p, c_p, a_d, b_d, t_c = params

    print(f"Fitted Parameters:")
    print(f"  a_p={a_p:.6e}, b_p={b_p:.6e}, c_p={c_p:.6e}")
    print(f"  a_d={a_d:.6e}, b_d={b_d:.6e}, alpha={alpha:.4f}, t_c={t_c:.4f}")
    print(f"  Test MAPE: {mape:.2f}%")

    if args.output_params:
        with open(args.output_params, "w") as f:
            json.dump({
                "a_p": a_p, "b_p": b_p, "c_p": c_p,
                "a_d": a_d, "b_d": b_d, "alpha": alpha, "t_c": t_c,
                "test_mape": mape,
            }, f, indent=2)


if __name__ == "__main__":
    main()
```

## Quick Start

```bash
# Activate environment
conda activate myvllm

# Run profiling (sweeps GPU frequencies, collects batch data)
# Requires sudo for GPU clock locking
PYTHONPATH=/home/ubuntu/lqs/vllm \
VLLM_PROFILING=1 \
VLLM_PROFILING_FILE=/home/ubuntu/lqs/profiling_data.jsonl \
python profiling_script.py \
    --model-path /home/ubuntu/lqs/L3 \
    --output /home/ubuntu/lqs/profiling_data.jsonl \
    --num-frequencies 8 \
    --samples-per-config 3

# Run fitting (reads data, fits parameters, reports MAPE)
python fitting_script.py \
    --input /home/ubuntu/lqs/profiling_data.jsonl \
    --output-params fitted_params.json
```

## Output File Format

### Profiling Data (JSONL)

Each line is a JSON object representing one batch execution:

```json
{
  "batch_id": 123,
  "wall_time_ms": 45.67,
  "gpu_freq_mhz": 2100.0,
  "requests": [
    {"req_id": "abc-123", "type": "prefill", "l_q": 256, "l_kv": 0},
    {"req_id": "def-456", "type": "decode", "l_q": 1, "l_kv": 512}
  ],
  "timestamp": 1712345678.123
}
```

### Fitted Parameters (JSON)

```json
{
  "a_p": 0.0,
  "b_p": 0.059,
  "c_p": 331.5,
  "a_d": 0.0,
  "b_d": 0.0,
  "alpha": 0.05,
  "t_c": 21.16,
  "test_mape": 24.47
}
```

## Troubleshooting

### GPU Clock Locking Fails
- Ensure you have sudo access
- Check if GPU supports clock locking: `nvidia-smi -q -d SUPPORTED_CLOCKS`
- Try application clocks as fallback: `sudo nvidia-smi -ac 10501,<freq>`

### No Decode Data
- The `generate()` API with `max_tokens=1` only captures prefill
- For decode data, use multi-step generation or direct `LLMEngine` instrumentation

### High MAPE
- Increase `samples_per_config` for more data
- Ensure GPU temperature is stable (throttling affects results)
- Check for other GPU workloads during profiling

### Memory Issues
- Reduce `max_num_seqs` in LLM initialization
- Lower `gpu_memory_utilization`
- Use smaller batch configurations
