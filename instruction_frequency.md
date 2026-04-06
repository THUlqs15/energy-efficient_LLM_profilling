# Batch Latency Estimator: Offline Profiling Instructions

## Objective

Profile the vLLM serving engine to collect batch execution time data, then fit a regression model that predicts batch processing latency. The model separates **prefill** and **decode** requests with distinct formulas, and explicitly accounts for **GPU clock frequency** as a variable. You must produce two deliverables: `result.md` (fitted parameters + MAPE) and `profiling.md` (reproducible workflow + scripts).

---

## Environment

- **vLLM source code**: `/home/ubuntu/lqs/vllm` (you may modify any code here for profiling purposes)
- **LLM model path**: `/home/ubuntu/lqs/L3`
- **Conda environment**: activate with `conda activate myvllm` before any vLLM-related command
- **Output directory for result.md and profiling.md**: current working directory (where you are invoked)

---

## Theoretical Model

The batch execution time model decomposes latency into a constant overhead and per-request computational cost.

### Per-Request Latency

For a request `r`, the total estimated latency is:

```
T_pd(r) = T_tilde_pd(r) + t_c
```

where `t_c` is a constant overhead (kernel launch, I/O processing, etc.), and `T_tilde_pd(r)` is the core computational latency:

```
T_tilde_pd(r) = T_tilde_p(r)   if r is a prefill request
               = T_tilde_d(r)   if r is a decode request
```

### Prefill Model (Equation 5, frequency-aware)

```
T_tilde_p(f, r) = [a_p * l_q(r)^2 + b_p * l_q(r) * l_kv(r) + c_p * l_q(r)] / f
```

- `f`: GPU clock frequency (in MHz or GHz — pick one unit and stay consistent)
- `l_q(r)`: number of tokens processed in the current forward pass (i.e., the input/prompt length for prefill)
- `l_kv(r)`: KV cache length (0 for fresh prefill with no prefix caching, or nonzero if there's shared prefix)
- Prefill is **compute-bound**, so latency scales as `1/f` (inversely proportional to frequency).
- Trainable parameters: `{a_p, b_p, c_p}`

### Decode Model (Equation 6, frequency-aware)

```
T_tilde_d(f, r) = [a_d * l_kv(r) + b_d * 1] / f^α
```

- `f`: GPU clock frequency (same unit as prefill)
- `l_kv(r)`: KV cache length (total context so far)
- Decode is **memory-bound**, so latency scales as `1/f^α` where `α ∈ (0, 1)`. The exponent `α` must also be fitted — it captures the fact that memory bandwidth does not scale linearly with GPU clock frequency.
- Trainable parameters: `{a_d, b_d, α}`

### Batch-Level Estimation (Equation 7)

For a batch `B = B_p ∪ B_d` (union of prefill set and decode set) executed at GPU frequency `f`:

```
T_pd(f, B) = Σ_{r ∈ B_p} T_tilde_p(f, r) + Σ_{r ∈ B_d} T_tilde_d(f, r) + t_c
```

Expanding:

```
T_pd(f, B) = (1/f) * [a_p * Σ l_q(r)² + b_p * Σ l_q(r)·l_kv(r) + c_p * Σ l_q(r)]   (prefill sum)
           + (1/f^α) * [a_d * Σ l_kv(r) + b_d * num_decode]                            (decode sum)
           + t_c
```

**Total trainable parameters**: `{a_p, b_p, c_p, a_d, b_d, α, t_c}` (7 parameters)

> **Note on α**: Since `α` appears as an exponent, this model is **not purely linear**. The fitting strategy must handle this — see Phase 3 for details.

---

## Step-by-Step Profiling Procedure

### Phase 1: Instrument vLLM to Collect Per-Batch Timing Data

You need to modify vLLM's source code to log the actual execution time of each batch (a single model forward pass / scheduler step), along with the metadata of every request in that batch.

**What to log for each batch execution:**

1. **Batch-level**: actual GPU execution wall-clock time (in milliseconds), using `torch.cuda.synchronize()` before and after the forward pass, then `time.perf_counter()` to measure.
2. **GPU frequency `f`**: the current GPU clock frequency at the time of this batch execution. You can read it via `nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits` or programmatically via `pynvml`. Log the frequency in MHz.
3. **Per-request in batch**: for each request (sequence) in the batch, record:
   - Whether it is a **prefill** or **decode** request
   - `l_q`: number of query tokens being processed in this step
   - `l_kv`: current KV cache length (number of tokens already in the cache before this step)

**Key instrumentation points in vLLM** (these are suggestions — explore the codebase to find the exact right locations):

- The model runner's `execute_model()` or the equivalent forward-pass entry point is the best place to measure batch execution time.
- The scheduler or sequence group metadata contains information about each request's prefill/decode status, query length, and KV cache length.
- Look at `ModelRunner`, `ModelRunnerBase`, or similar classes in `vllm/worker/` for the forward pass.
- Look at `SchedulerOutput`, `SequenceGroupMetadata`, or `ModelInput` for batch composition details.
- Be careful: measure only the GPU forward pass time, not scheduling overhead or sampling time. Use `torch.cuda.synchronize()` to ensure accurate GPU timing.

**Implementation approach:**

1. Add a profiling logger module (e.g., `vllm/profiling_logger.py`) that writes batch records to a CSV or JSON Lines file.
2. Instrument the forward pass to record timing.
3. Extract per-request metadata from the batch before or after the forward pass.
4. Write each batch's data as one record: `{batch_id, gpu_freq_mhz, wall_time_ms, requests: [{type, l_q, l_kv}, ...]}`.
5. Save the profiling data to a file like `/workspace/lqs3/LLM_scheduling/profiling_data.jsonl`.

### Phase 2: Generate Diverse Profiling Batches

Run the vLLM engine with synthetic requests designed to cover a wide range of batch compositions **and GPU frequencies**. You need diversity in:

- **GPU clock frequency `f`**: enumerate all supported frequencies. Use `nvidia-smi -q -d SUPPORTED_CLOCKS` to list them, then lock the GPU to each frequency with `nvidia-smi -lgc <freq>,<freq>` (or `nvidia-smi --lock-gpu-clocks=<freq>,<freq>`) before running the profiling batches at that frequency. After profiling at each frequency, unlock with `nvidia-smi -rgc`. You need `sudo` or root permissions for clock locking — check and handle this. If the GPU does not support arbitrary clock locking, use the available supported clock levels and profile at each one.
- **Number of prefill requests per batch**: 0 to ~16 (or the max batch size)
- **Number of decode requests per batch**: 0 to ~64 (or more)
- **Prefill query lengths** (`l_q`): vary from short (32, 64, 128) to long (512, 1024, 2048)
- **KV cache lengths** (`l_kv`): vary from 0 to large values (256, 512, 1024, 2048)
- **Mixed batches**: combinations of prefill and decode requests

**GPU Frequency Sweeping Strategy:**

1. First, query all supported GPU graphics clock frequencies:
   ```bash
   nvidia-smi -q -d SUPPORTED_CLOCKS
   ```
   This lists all valid (memory_clock, graphics_clock) pairs. Focus on the graphics clock values.

2. Select a representative subset of frequencies (at least 5-8 distinct values spanning the full range, e.g., min, 25th percentile, 50th, 75th, max). If the total number of supported frequencies is small (≤10), use all of them.

3. For each frequency `f`:
   ```bash
   nvidia-smi --lock-gpu-clocks=<f>,<f>
   # Run profiling batches at this frequency
   nvidia-smi --reset-gpu-clocks
   ```

4. Record the actual GPU frequency in every batch log entry (read it programmatically to confirm the lock is effective).

**Approach options** (choose the most practical one):

**Option A: Offline batch profiling script** (Recommended)
Write a standalone Python script that:
1. Queries supported GPU clock frequencies.
2. For each frequency, locks the GPU clock, initializes the vLLM `LLMEngine` or `LLM` class directly (not the API server).
3. Manually constructs batches with controlled compositions (you may need to directly call the model runner or manipulate the scheduler).
4. For each batch configuration, runs the forward pass multiple times (e.g., 3-5 repetitions) and records the timing along with the GPU frequency.
5. Systematically sweeps over parameter ranges × frequencies.
6. Resets GPU clocks after completion.

**Option B: Synthetic workload via API**
Start the vLLM server and send carefully crafted requests via the API to induce various batch compositions. Less controllable but simpler. You would need to restart or reconfigure the server at each frequency.

**Target**: collect at least **500-1000 batch samples per frequency level**, with diverse batch configurations. Across all frequencies, this means potentially several thousand samples total.

### Phase 3: Fit the Regression Model

After collecting profiling data, fit the model parameters. Because `α` appears as an exponent on `f`, this is **not a pure linear regression** — it requires a two-stage or non-linear optimization approach.

**Feature engineering for each batch:**

From the raw per-batch data, compute these aggregate features:

```python
# For each batch b with GPU frequency f:
f = b.gpu_freq_mhz

# Prefill aggregate features (will be divided by f)
sum_lq_sq = sum(r.l_q ** 2 for r in b.prefill_requests)        # for a_p
sum_lq_lkv = sum(r.l_q * r.l_kv for r in b.prefill_requests)   # for b_p
sum_lq = sum(r.l_q for r in b.prefill_requests)                 # for c_p

# Decode aggregate features (will be divided by f^α)
sum_lkv_decode = sum(r.l_kv for r in b.decode_requests)         # for a_d
num_decode = len(b.decode_requests)                              # for b_d

# constant 1 for t_c
```

**The full model:**

```
T_pd(f, B) = (1/f) * [a_p * sum_lq_sq + b_p * sum_lq_lkv + c_p * sum_lq]
           + (1/f^α) * [a_d * sum_lkv_decode + b_d * num_decode]
           + t_c
```

**Fitting Strategy — Two approaches (use both, compare results):**

**Approach 1: Grid search over α + linear regression** (Recommended)

Since for any fixed value of `α`, the model is linear in `{a_p, b_p, c_p, a_d, b_d, t_c}`:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

best_mape = float('inf')
best_alpha = None
best_params = None

for alpha in np.arange(0.01, 1.0, 0.01):  # grid search α in (0, 1)
    # Construct feature matrix for this α
    X = np.column_stack([
        sum_lq_sq / f,          # a_p term
        sum_lq_lkv / f,         # b_p term
        sum_lq / f,             # c_p term
        sum_lkv_decode / f**alpha,  # a_d term
        num_decode / f**alpha,      # b_d term
        np.ones(N),                 # t_c term
    ])
    y = measured_times

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    if mape < best_mape:
        best_mape = mape
        best_alpha = alpha
        best_params = model.coef_  # [a_p, b_p, c_p, a_d, b_d, t_c]
```

**Approach 2: Non-linear optimization with `scipy.optimize.minimize`**

Use this as a refinement step after Approach 1 to fine-tune α:

```python
from scipy.optimize import minimize

def predict(params, X_raw, f):
    a_p, b_p, c_p, a_d, b_d, alpha, t_c = params
    return (1/f) * (a_p * X_raw['sum_lq_sq'] + b_p * X_raw['sum_lq_lkv'] + c_p * X_raw['sum_lq']) \
         + (1/f**alpha) * (a_d * X_raw['sum_lkv_decode'] + b_d * X_raw['num_decode']) \
         + t_c

def loss(params):
    y_pred = predict(params, X_raw_train, f_train)
    return np.mean((y_train - y_pred) ** 2)  # MSE

# Initialize from Approach 1 results
x0 = [*best_params_from_approach1, best_alpha_from_approach1, best_tc]
result = minimize(loss, x0, method='Nelder-Mead', options={'maxiter': 10000})
```

Constrain `α ∈ (0, 1)` using bounds (e.g., `scipy.optimize.minimize` with `method='L-BFGS-B'` and `bounds`).

**Use `numpy`, `sklearn`, and `scipy` for fitting.**

### Phase 4: Evaluate the Model

1. Split data into **80% train / 20% test** (or use cross-validation).
2. Fit on train set.
3. Evaluate on test set using **Mean Absolute Percentage Error (MAPE)**:

```python
MAPE = (1/N) * Σ |y_true - y_pred| / y_true * 100%
```

4. Also report R², MAE, and RMSE for completeness.
5. Target MAPE: ~4.5% based on the reference paper. If significantly worse, investigate:
   - Data quality issues (outliers, warm-up effects)
   - Missing features
   - Non-linearity beyond what the model captures

### Phase 5: Validate and Iterate

- Plot predicted vs. actual execution times (scatter plot).
- Check residuals for systematic bias.
- If MAPE is too high, consider:
  - Removing outlier batches (first few warm-up batches)
  - Adding higher-order terms
  - Separating by batch size ranges
  - Checking if chunked prefill affects the model

---

## Output Deliverables

### 1. `result.md` (in current working directory)

Must contain:

```markdown
# Batch Latency Model — Fitted Results

## Model
<model name, GPU info, any relevant hardware details>

## Fitted Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a_p | ... | Prefill: quadratic term coefficient for l_q² |
| b_p | ... | Prefill: cross-term coefficient for l_q × l_kv |
| c_p | ... | Prefill: linear term coefficient for l_q |
| a_d | ... | Decode: linear term coefficient for l_kv |
| b_d | ... | Decode: constant term per decode request |
| α (alpha) | ... | Decode: frequency scaling exponent (0 < α < 1) |
| t_c | ... | Batch-level constant overhead |

## GPU Frequency Information

- GPU model: ...
- Supported frequencies profiled: [list of f values in MHz]
- Fitted α value: ... (interpretation: decode latency scales as 1/f^α)

## Evaluation Metrics

| Metric | Train | Test |
|--------|-------|------|
| MAPE   | ...%  | ...% |
| MAE    | ... ms | ... ms |
| RMSE   | ... ms | ... ms |
| R²     | ...   | ...  |

## Data Summary

- Total batches profiled: ...
- Train set size: ...
- Test set size: ...
- GPU frequency range: [min MHz, max MHz]
- Number of distinct frequencies: ...
- Prefill l_q range: [min, max]
- Decode l_kv range: [min, max]
- Batch size range: [min, max]

## Usage

To predict batch execution time at GPU frequency f (MHz):
```
T_pd(f, B) = (1/f) * (a_p * Σ l_q² + b_p * Σ l_q·l_kv + c_p * Σ l_q)
           + (1/f^α) * (a_d * Σ l_kv_decode + b_d * num_decode)
           + t_c
```
<provide the concrete formula with fitted numeric values plugged in>
```

### 2. `profiling.md` (in current working directory)

Must contain the **complete reproducible workflow** so that on a new server, one can:

1. Run a single profiling script to collect data
2. Run a single fitting script to get model parameters and MAPE

Structure:

```markdown
# Batch Latency Profiling — Reproducible Workflow

## Prerequisites
- vLLM installed (specify version/commit if relevant)
- Model path (parameterized)
- Python packages needed (numpy, sklearn, etc.)

## Step 1: Instrument vLLM
<Describe what files were modified and how, or provide a patch>
<If possible, make the profiling non-intrusive — e.g., controlled by an env variable>

## Step 2: Run Profiling
<Provide the exact script, command, and expected output file>

## Step 3: Fit Model
<Provide the exact fitting script that reads profiling data and outputs parameters + MAPE>

## Full Scripts

### profiling_script.py
<complete, self-contained script>

### fitting_script.py
<complete, self-contained script>

## Quick Start
```bash
conda activate myvllm
# Profiling script automatically sweeps GPU frequencies and batch configurations
python profiling_script.py --model-path /path/to/model --output profiling_data.jsonl
# Fitting script reads data, fits {a_p, b_p, c_p, a_d, b_d, α, t_c}, reports MAPE
python fitting_script.py --input profiling_data.jsonl
```
```

---

## Important Notes

1. **GPU warm-up**: Discard the first 10-20 batches as warm-up **at each frequency level**. GPU kernels and memory allocators need time to stabilize, especially after a clock change.
2. **torch.cuda.synchronize()**: Always synchronize before taking timestamps to get accurate GPU execution time.
3. **Avoid interference**: During profiling, ensure no other GPU workloads are running.
4. **GPU clock locking**: Always verify the clock lock took effect by reading back the actual GPU frequency (e.g., via `nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits` or `pynvml`). Some GPUs may throttle due to thermal limits — monitor temperature and power.
5. **Clock lock permissions**: `nvidia-smi --lock-gpu-clocks` typically requires root/sudo. If unavailable, use `nvidia-smi -ac <mem_clock>,<graphics_clock>` (application clocks) as a fallback, though it's less strict.
6. **Frequency unit consistency**: Choose MHz and stick with it everywhere — profiling logs, feature computation, and the fitted model. Document the unit clearly in result.md.
7. **Chunked prefill**: If vLLM uses chunked prefill (splitting long prefills across iterations), account for this — `l_q` should be the actual number of tokens processed in that specific forward pass, not the full prompt length.
8. **Memory constraints**: Stay within GPU memory. If the model is large, limit max batch sizes accordingly.
9. **Reproducibility**: Set random seeds, record the exact vLLM commit hash, and note GPU model/driver version.
10. **Restore vLLM**: After profiling, either revert your vLLM modifications or make them toggle-able via an environment variable (e.g., `VLLM_PROFILING=1`).
11. **Reset GPU clocks**: Always run `nvidia-smi --reset-gpu-clocks` after profiling to restore default behavior.
12. **Data format**: Use JSON Lines for the profiling data — one JSON object per line, each representing one batch execution. Each record must include `gpu_freq_mhz`. This is easy to parse and append to.
13. **Non-linear fitting**: Since `α` is a non-linear parameter, always validate the fitted α value: it should be in (0, 1). If the optimizer converges to α ≈ 0 or α ≈ 1, this may indicate insufficient frequency variation in the data or other issues.
