# Batch Latency Model — Fitted Results

## Model

- **LLM Model**: Qwen3 (from `/home/ubuntu/lqs/L3`)
- **GPU**: NVIDIA GeForce RTX 4090
- **VRAM**: 49140 MiB
- **Driver Version**: 570.153.02
- **Compute Capability**: 8.9
- **vLLM Version**: v0.19.0 (V1 engine)

## Fitted Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a_p | 0.000000e+00 | Prefill: quadratic term coefficient for l_q² |
| b_p | 0.000000e+00 | Prefill: cross-term coefficient for l_q × l_kv |
| c_p | 3.864340e+02 | Prefill: linear term coefficient for l_q |
| a_d | 0.000000e+00 | Decode: linear term coefficient for l_kv |
| b_d | 2.058310e+03 | Decode: constant term per decode request |
| α (alpha) | 0.9900 | Decode: frequency scaling exponent (0 < α < 1) |
| t_c | 15.9795 ms | Batch-level constant overhead |

### Parameter Interpretation

- **c_p = 386.43**: Each prefill token contributes ~386/f ms to latency (compute-bound, scales with 1/f)
- **b_d = 2058.31**: Each decode request contributes ~2058/f^0.99 ms to latency
- **α = 0.99**: Decode latency scales almost linearly with frequency (1/f^0.99 ≈ 1/f), suggesting decode is also compute-bound on this GPU, possibly due to FlashAttention optimizations
- **t_c = 15.98 ms**: Fixed overhead per batch (kernel launch, scheduling, I/O)

## GPU Frequency Information

- **GPU model**: NVIDIA GeForce RTX 4090
- **Supported frequencies profiled**: [210, 495, 930, 1365, 1800, 2235, 2640, 2670, 2745, 2760, 2775] MHz
- **Fitted α value**: 0.99 (decode latency scales as 1/f^0.99, nearly linear with frequency)
- **Number of distinct frequencies**: 11

## Evaluation Metrics

| Metric | Train | Test |
|--------|-------|------|
| MAPE   | 14.57% | 14.57% |
| MAE    | 4.76 ms | 4.77 ms |
| RMSE   | 8.14 ms | 8.16 ms |
| R²     | 0.7797 | 0.7761 |

### Improvement from Decode Data

By using `LLMEngine.step()` instead of the high-level `generate()` API, we captured decode batches and achieved significant improvement:

| Metric | Without Decode | With Decode | Improvement |
|--------|----------------|-------------|-------------|
| Test MAPE | 24.47% | 14.57% | **40% reduction** |
| b_d value | 0 | 2058.31 | Now captures decode |
| Decode requests | 0 | 176,840 | Full decode coverage |

## Data Summary

- **Total batches profiled**: 45,744
- **Train set size**: 36,595 (80%)
- **Test set size**: 9,149 (20%)
- **GPU frequency range**: [210, 2775] MHz
- **Number of distinct frequencies**: 11
- **Prefill requests**: 4,064
- **Decode requests**: 176,840
- **Decode/Prefill ratio**: 43.51
- **Prefill l_q range**: [6, 82] tokens
- **Decode l_kv range (avg)**: [70, 710] tokens
- **Wall time range**: [16.30, 172.43] ms

## Usage

To predict batch execution time at GPU frequency f (MHz):

```
T_pd(f, B) = (1/f) × [a_p × Σl_q² + b_p × Σl_q·l_kv + c_p × Σl_q]
           + (1/f^α) × [a_d × Σl_kv_decode + b_d × num_decode]
           + t_c
```

### Concrete Formula with Fitted Values

```
T_pd(f, B) = (1/f) × [386.43 × Σl_q]
           + (1/f^0.99) × [2058.31 × num_decode]
           + 15.98  (ms)
```

Where:
- `f` is GPU frequency in MHz
- `Σl_q` is sum of query token counts for all prefill requests in the batch
- `num_decode` is the number of decode requests in the batch
- Result is in milliseconds

### Example Calculations

**Example 1: Pure Prefill Batch**

4 prefill requests, each processing 128 tokens, at 2100 MHz:

```
Σl_q = 4 × 128 = 512
num_decode = 0

T = (1/2100) × [386.43 × 512] + (1/2100^0.99) × [2058.31 × 0] + 15.98
  = (1/2100) × 197,852 + 0 + 15.98
  = 94.22 + 15.98
  = 110.20 ms
```

**Example 2: Pure Decode Batch**

8 decode requests at 2100 MHz:

```
Σl_q = 0
num_decode = 8

T = (1/2100) × [386.43 × 0] + (1/2100^0.99) × [2058.31 × 8] + 15.98
  = 0 + (1/1932.7) × 16,466.5 + 15.98
  = 8.52 + 15.98
  = 24.50 ms
```

**Example 3: Mixed Batch**

2 prefill (64 tokens each) + 6 decode requests at 1500 MHz:

```
Σl_q = 2 × 64 = 128
num_decode = 6

T = (1/1500) × [386.43 × 128] + (1/1500^0.99) × [2058.31 × 6] + 15.98
  = (1/1500) × 49,463 + (1/1383.3) × 12,349.9 + 15.98
  = 32.98 + 8.93 + 15.98
  = 57.89 ms
```

## Profiling Methodology

### LLMEngine-based Approach

This profiling used direct `LLMEngine.step()` calls to capture both prefill and decode batches:

1. **Workload Generation**: Mixed workloads with varying prompt lengths (64-1024 tokens) and generation lengths (20-150 tokens)
2. **Step-by-Step Execution**: Each `engine.step()` corresponds to one scheduler batch
3. **Decode Capture**: Multi-token generation ensures decode iterations are captured
4. **Frequency Sweeping**: GPU clock locked at 8 representative frequencies spanning 210-2775 MHz

### Key Differences from High-Level API

| Aspect | generate() API | LLMEngine.step() |
|--------|----------------|------------------|
| Decode visibility | Hidden | Exposed |
| Batch control | Automatic | Manual |
| Decode/Prefill ratio | ~0 | 43:1 |
| MAPE achieved | 24.47% | 14.57% |

## Limitations and Future Work

1. **Quadratic prefill term (a_p = 0)**: The profiling used chunked prefill with max 8192 tokens, limiting exposure to quadratic scaling. Longer prompts would reveal the l_q² term.

2. **KV-dependent decode term (a_d = 0)**: The constant-per-decode model (b_d) dominates. Longer generation sequences would better expose the l_kv dependency.

3. **α ≈ 1**: The high α value suggests decode on RTX 4090 with FlashAttention is more compute-bound than memory-bound. Different GPUs may show different α values.

4. **Prefix caching**: High prefix cache hit rates (95%+) during profiling may affect prefill measurements.

5. **Target MAPE**: Current 14.57% vs target 4.5%. Further improvement possible with:
   - Longer prompts (expose a_p)
   - Longer generations (expose a_d)  
   - More diverse batch sizes
   - Per-frequency model calibration
