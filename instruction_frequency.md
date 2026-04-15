# Batch Latency Estimator: Offline Profiling Instructions

## Objective

Profile the vLLM serving engine to collect batch execution time data, then fit a regression model that predicts batch processing latency. The model separates **prefill** and **decode** requests with distinct formulas, and explicitly accounts for **GPU clock frequency** as a variable. You must produce two deliverables: `result.md` (fitted parameters + MAPE) and `profiling.md` (reproducible workflow + scripts).

> **Note**: If you find any scripts or data files left over from a previous profiling run (e.g., `profiling_script.py`, `profiling_engine_script.py`, `fitting_script.py`, `profiling_data.jsonl`, `fitted_params.json`, etc.), feel free to overwrite or delete them. **Do not copy or reuse their code** — the requirements in this instruction may differ from what was used before, and carrying over old configurations will reproduce the same problems.

---

## Environment

- **vLLM source code**: `/home/ubuntu/lqs/vllm` (you may modify any code here for profiling purposes)
- **LLM model path**: `/home/ubuntu/lqs/LLM_model`
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

- `f`: GPU graphics clock frequency (in MHz or GHz — pick one unit and stay consistent). **Note**: throughout this document, "GPU clock frequency" or "frequency" always refers to the **graphics clock frequency**, not the memory clock frequency. The memory clock frequency must be held constant during all profiling — see Phase 2 for details.
- `l_q(r)`: number of tokens processed in the current forward pass (i.e., the input/prompt length for prefill)
- `l_kv(r)`: KV cache length (0 for fresh prefill with no prefix caching, or nonzero if there's shared prefix)
- Prefill is **compute-bound**, so latency scales as `1/f` (inversely proportional to graphics frequency).
- Trainable parameters: `{a_p, b_p, c_p}`

### Decode Model (Equation 6, frequency-aware)

```
T_tilde_d(f, r) = [a_d * l_kv(r) + b_d * 1] / f^α
```

- `f`: GPU graphics clock frequency (same unit as prefill)
- `l_kv(r)`: KV cache length (total context so far)
- Decode is **memory-bound**, so latency scales as `1/f^α` where `α ∈ (0, 1)`. The exponent `α` must also be fitted — it captures the fact that memory bandwidth does not scale linearly with GPU graphics clock frequency.
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
2. **GPU graphics frequency `f`**: the current GPU graphics clock frequency at the time of this batch execution. You can read it via `nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits` or programmatically via `pynvml`. Log the frequency in MHz.
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

- **GPU graphics clock frequency `f`**: enumerate all supported graphics clock frequencies and sweep nearly all of them. Use `nvidia-smi -q -d SUPPORTED_CLOCKS` to list all valid (memory_clock, graphics_clock) pairs. **Fix the memory clock at its highest supported value** and iterate over graphics clock frequencies from the lowest supported value up to approximately 95% of the maximum supported graphics clock. Lock the GPU clocks with `nvidia-smi --lock-gpu-clocks=<freq>,<freq>` and set the memory clock with `nvidia-smi --lock-memory-clocks=<mem_freq>,<mem_freq>` (or use `nvidia-smi -ac <mem_freq>,<graphics_freq>`) before running the profiling batches at each graphics frequency. After profiling at each frequency, unlock with `nvidia-smi -rgc` and `nvidia-smi -rmc`. You need `sudo` or root permissions for clock locking — check and handle this.
- **Number of prefill requests per batch**: 0 to ~16 (or the max batch size)
- **Number of decode requests per batch**: 0 to ~64 (or more)
- **Prefill query lengths** (`l_q`): vary from short (32, 64, 128) to long (512, 1024, 2048, 4096). You must cover a wide range to expose the quadratic `l_q²` term.
- **KV cache lengths** (`l_kv`): vary from 0 to large values (256, 512, 1024, 2048)
- **Decode generation lengths**: use longer generation (e.g., max_tokens up to 500–1000) so that decode requests accumulate large `l_kv` values, exposing the `a_d` term.
- **Mixed batches**: combinations of prefill and decode requests **in the same batch/step**. This is critical — the model assumes batch latency is the sum of individual request contributions (`Σ prefill + Σ decode + t_c`). This additivity assumption must be validated with mixed batches where prefill and decode requests compete for GPU resources simultaneously. Pure-prefill and pure-decode batches alone cannot test this.

**GPU Frequency Sweeping Strategy:**

1. First, query all supported GPU clock frequencies:
   ```bash
   nvidia-smi -q -d SUPPORTED_CLOCKS
   ```
   This lists all valid (memory_clock, graphics_clock) pairs. Identify the **highest supported memory clock** and collect all graphics clock values available at that memory clock level.

2. **Fix the memory clock** at the highest supported value for the entire profiling session. This isolates the effect of graphics clock frequency on latency and prevents memory bandwidth from becoming a confounding variable.

3. **Sweep nearly all supported graphics clock frequencies** — from the **lowest** supported value up to approximately **95% of the maximum** supported graphics clock. Do not cherry-pick or subsample: include every supported graphics clock frequency in this range. The only frequencies to skip are those in the **top ~5%** of the range, where GPU Boost and thermal throttling make clock locking unreliable (the GPU may fluctuate between multiple frequencies instead of staying locked). For example, if the GPU supports graphics clocks from 210 MHz to 2775 MHz, profile every supported frequency from 210 MHz up to ~2636 MHz (95% of 2775). This typically results in dozens to over a hundred frequency points depending on the GPU, which greatly improves the accuracy and robustness of the fitted α parameter.

4. For each graphics frequency `f`:
   ```bash
   nvidia-smi --lock-memory-clocks=<max_mem_freq>,<max_mem_freq>
   nvidia-smi --lock-gpu-clocks=<f>,<f>
   # Run profiling batches at this frequency
   nvidia-smi --reset-gpu-clocks
   nvidia-smi --reset-memory-clocks
   ```

5. Record the actual GPU graphics frequency in every batch log entry (read it programmatically to confirm the lock is effective). Also verify that the memory clock remains at the expected fixed value.

**Approach options** (choose the most practical one):

**Option A: Offline batch profiling script** (Recommended)
Write a standalone Python script that:
1. Queries supported GPU clock frequencies (both memory and graphics clocks). Identifies the highest supported memory clock and all graphics clocks available at that memory clock level.
2. Locks the memory clock at the highest supported value. For each graphics frequency (from lowest to ~95% of max), locks the GPU graphics clock, initializes the vLLM `LLMEngine` or `LLM` class directly (not the API server).
3. Manually constructs batches with controlled compositions (you may need to directly call the model runner or manipulate the scheduler).
4. For each batch configuration, runs the forward pass multiple times (e.g., 3-5 repetitions) and records the timing along with the GPU frequency.
5. Systematically sweeps over parameter ranges × frequencies.
6. Resets both GPU graphics clocks and memory clocks after completion.

**Critical engine configuration requirements:**
- **`enable_chunked_prefill=False`**: You MUST disable chunked prefill. With chunked prefill enabled, long prompts are split into tiny chunks (often <20 tokens), which makes all `l_q` values extremely small and prevents the model from learning the quadratic `a_p * l_q²` term. The whole point of profiling is to observe how latency scales with varying `l_q` up to thousands of tokens.
- **`enable_prefix_caching=False`**: You MUST disable automatic prefix caching (APC). With APC enabled, repeated or similar prompts hit the prefix cache, causing `l_kv > 0` for what should be fresh prefill requests and reducing the effective `l_q`. This corrupts the prefill measurements.
- **`enforce_eager=True`**: Disable CUDA Graph to get accurate per-batch timing.
- **Use truly randomized prompts**: Do not use repetitive patterns like "The quick brown fox..." repeated N times. Instead, generate prompts with diverse, random vocabulary so that prefix caching (even if accidentally re-enabled) cannot exploit common prefixes between requests.

**Batch composition strategy — continuous request injection:**

The profiling script must produce three types of batches: pure prefill, pure decode, and **mixed prefill+decode**. To achieve mixed batches, you must **inject new requests while old requests are still decoding** — do NOT wait for all requests to finish before adding new ones. If you use a loop like `while has_unfinished: step()` followed by adding a new batch of requests, the engine will never see prefill and decode in the same step.

Instead, use a step-driven loop that periodically injects new requests:

```python
# WRONG — produces only pure prefill or pure decode batches:
for iteration in range(...):
    engine.add_request(...)       # add all requests
    while engine.has_unfinished_requests():
        engine.step()             # run to completion → no mixing

# CORRECT — produces mixed batches:
for step in range(total_steps):
    if should_inject_new_request(step):
        engine.add_request(...)   # inject while old requests are decoding
    engine.step()                 # batch may contain both prefill + decode
```

Aim for a roughly even mix: ~1/3 of batches should be pure prefill, ~1/3 pure decode, ~1/3 mixed. You can control this by tuning the injection interval relative to each request's `max_tokens`.

Additionally, vary the **batch sizes** (1, 2, 4, 8, 16, 32, 64 requests per batch). Different batch sizes are essential to separate the per-batch constant overhead `t_c` from per-request costs — single-request batches alone cannot disentangle these.

**Option B: Synthetic workload via API**
Start the vLLM server and send carefully crafted requests via the API to induce various batch compositions. Less controllable but simpler. You would need to restart or reconfigure the server at each frequency.

**Target**: collect at least **200-500 batch samples per frequency level**, with diverse batch configurations. Since you are sweeping nearly all supported graphics frequencies (potentially dozens to 100+ frequency levels), adjust the per-frequency sample count to keep total profiling time manageable — the total dataset should be at least **10,000-50,000 samples**. Having many frequency points is more valuable for fitting α than having a very large number of samples per frequency.

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
5. If MAPE is high (e.g., >10%), investigate potential issues:
   - Data quality issues (outliers, warm-up effects)
   - Missing features
   - Non-linearity beyond what the model captures
   - Verify that all 7 parameters have non-trivial fitted values (especially `a_p`, `b_p`, `a_d`). If any compute-related parameter is exactly 0, it likely means the profiling data lacked sufficient variation in that dimension (e.g., `l_q` range too narrow, or decode sequences too short).

### Phase 5: Validate and Iterate

- Plot predicted vs. actual execution times (scatter plot).
- Check residuals for systematic bias.
- **Verify parameter quality**: All 7 parameters should ideally have non-zero fitted values. If `a_p = 0` or `b_p = 0`, the profiling data likely lacked sufficient `l_q` variation (check that `l_q` ranges from at least 32 to 2048+). If `a_d = 0`, decode sequences were too short to reveal KV cache length dependency.
- **Verify frequency stability**: Check that each target graphics frequency has a consistent number of batch samples and that the memory clock remained constant throughout. If some graphics frequencies show scattered actual-frequency values (e.g., you locked at 2700 MHz but got data at 2640, 2670, 2700, 2745), that frequency point had unstable clock locking — discard that data.
- If MAPE is too high, consider:
  - Removing outlier batches (first few warm-up batches, or batches where actual frequency deviates >50 MHz from the locked target)
  - Adding higher-order terms
  - Separating by batch size ranges
  - Increasing the number and diversity of profiling workloads

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
- Memory clock (fixed): ... MHz
- Graphics clock frequencies profiled: [list of f values in MHz]
- Graphics clock range profiled: [min MHz, max MHz] (up to ~95% of GPU max)
- Number of distinct graphics frequencies: ...
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
- GPU graphics frequency range: [min MHz, max MHz]
- Fixed memory clock: ... MHz
- Number of distinct graphics frequencies: ...
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
# Profiling script automatically fixes memory clock at max, then sweeps all graphics frequencies and batch configurations
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
4. **GPU clock locking**: Always verify the clock lock took effect by reading back the actual GPU graphics frequency (e.g., via `nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits` or `pynvml`) and the memory frequency (`nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits`). Ensure memory clock stays at the fixed high value. Some GPUs may throttle due to thermal limits — monitor temperature and power.
5. **Clock lock permissions**: `nvidia-smi --lock-gpu-clocks` and `nvidia-smi --lock-memory-clocks` typically require root/sudo. If unavailable, use `nvidia-smi -ac <mem_clock>,<graphics_clock>` (application clocks) as a fallback, though it's less strict.
6. **Frequency unit consistency**: Choose MHz and stick with it everywhere — profiling logs, feature computation, and the fitted model. Document the unit clearly in result.md. Always distinguish between graphics clock and memory clock in logs and documentation.
7. **Chunked prefill must be DISABLED**: Set `enable_chunked_prefill=False` in the engine configuration. Chunked prefill splits long prompts into tiny chunks (often <20 tokens), collapsing the `l_q` range and making it impossible to fit the `a_p * l_q²` quadratic term. The profiling requires observing the full range of `l_q` values (32 to 4096+) in single forward passes.
8. **Memory constraints**: Stay within GPU memory. If the model is large, limit max batch sizes accordingly.
9. **Reproducibility**: Set random seeds, record the exact vLLM commit hash, and note GPU model/driver version.
10. **Restore vLLM**: After profiling, either revert your vLLM modifications or make them toggle-able via an environment variable (e.g., `VLLM_PROFILING=1`).
11. **Reset GPU clocks**: Always run `nvidia-smi --reset-gpu-clocks` and `nvidia-smi --reset-memory-clocks` after profiling to restore default behavior.
12. **Data format**: Use JSON Lines for the profiling data — one JSON object per line, each representing one batch execution. Each record must include `gpu_freq_mhz`. This is easy to parse and append to.
13. **Non-linear fitting**: Since `α` is a non-linear parameter, always validate the fitted α value: it should be in (0, 1). If the optimizer converges to α ≈ 0 or α ≈ 1, this may indicate insufficient graphics frequency variation in the data or other issues.
14. **Prefix caching must be DISABLED**: Set `enable_prefix_caching=False`. Automatic prefix caching causes shared prefixes across requests to be served from cache, making `l_kv > 0` for fresh prefill requests and reducing effective `l_q`. This distorts both the prefill and decode measurements.
15. **Avoid high-frequency GPU clock points**: Do not select profiling graphics frequencies in the top ~5% of the GPU's supported range. At these frequencies, GPU Boost and thermal throttling cause the actual clock to fluctuate unpredictably, producing noisy data with multiple unintended frequency values instead of one stable locked frequency.
16. **Sweep nearly all graphics frequencies**: Do not subsample — include every supported graphics clock frequency from the minimum up to ~95% of the maximum. This typically yields dozens to 100+ frequency points, which greatly improves the accuracy of the fitted α parameter. More frequency diversity directly improves the reliability of the α estimate.
17. **Fix memory clock**: Lock the memory clock at its highest supported value for the entire profiling session. This ensures memory bandwidth is constant, so the only frequency variable is the graphics clock. Always verify the memory clock remains stable by reading it back programmatically.
18. **Reset both clocks**: After profiling, always run both `nvidia-smi --reset-gpu-clocks` and `nvidia-smi --reset-memory-clocks` to restore default behavior.
17. **Prompt diversity**: Use truly randomized prompts (random token sequences or diverse text) rather than repetitive patterns. Repetitive prompts enable prefix caching to kick in even when not explicitly configured, and produce unrealistic compute patterns.
18. **Longer decode sequences**: Set `max_tokens` high enough (e.g., 500-1000) and use `ignore_eos=True` to force long decode sequences. Short decode sequences (e.g., max_tokens < 50) do not accumulate enough KV cache variation to expose the `a_d * l_kv` dependency.
