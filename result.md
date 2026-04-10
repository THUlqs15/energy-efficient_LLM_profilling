# Batch Latency Model — Fitted Results

## Model
- **LLM**: Qwen3-8B (Qwen3ForCausalLM), bfloat16
- **GPU**: NVIDIA GeForce RTX 4090 (48 GB GDDR6X)
- **Driver**: 570.153.02, CUDA 12.8
- **vLLM**: v0.19.0 (commit `2a69949bd`)
- **Engine config**: `enforce_eager=True`, `enable_chunked_prefill=False`, `enable_prefix_caching=False`

## Fitted Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a_p | 2.1991e-02 | Prefill: quadratic term coefficient for l_q² |
| b_p | ≈ 0 (−1.97e-13) | Prefill: cross-term coefficient for l_q × l_kv (effectively zero — all prefills had l_kv = 0) |
| c_p | 2.1003e+02 | Prefill: linear term coefficient for l_q |
| a_d | 2.7652e-01 | Decode: linear term coefficient for l_kv |
| b_d | 1.7573e+02 | Decode: constant term per decode request |
| α (alpha) | 0.9899 | Decode: frequency scaling exponent (0 < α < 1) |
| t_c | 1.8019e+01 | Batch-level constant overhead (ms) |

## GPU Frequency Information

- **GPU model**: NVIDIA GeForce RTX 4090
- **Supported frequencies profiled** (MHz): 210, 315, 420, 525, 630, 735, 855, 960, 1065, 1170, 1275, 1380, 1500, 1605, 1710, 1815, 1920, 2025, 2145, 2250, 2355, 2460, 2565, 2670, 2790
- **Effective frequency range**: 210–2565 MHz (2670 and 2790 MHz targets experienced thermal throttling and reverted to ~2565 MHz)
- **Fitted α value**: 0.9899 — interpretation: decode latency scales as 1/f^0.99, very close to the prefill scaling of 1/f. On the RTX 4090, the GPU core clock strongly influences both compute-bound (prefill) and nominally memory-bound (decode) operations because the memory controller throughput is partially coupled to the core clock.

## Evaluation Metrics

| Metric | Train | Test |
|--------|-------|------|
| MAPE   | 13.82% | 13.93% |
| MAE    | 5.61 ms | 5.72 ms |
| RMSE   | 11.89 ms | 12.42 ms |
| R²     | 0.9829 | 0.9833 |

## Data Summary

- **Total batches profiled**: 53,065 (51,672 after cleaning)
- **Train set size**: 41,337
- **Test set size**: 10,335
- **GPU frequency range**: 210–2565 MHz
- **Number of distinct frequencies**: 34
- **Prefill l_q range**: 32–2,048
- **Decode l_kv range**: 32–2,048
- **Batch size range**: 1–81
- **Batch composition**: ~2,200 pure prefill, ~49,295 pure decode, ~1,570 mixed

## Usage

To predict batch execution time at GPU frequency f (MHz):

```
T_pd(f, B) = (1/f) * (a_p * Σ l_q² + b_p * Σ l_q·l_kv + c_p * Σ l_q)
           + (1/f^α) * (a_d * Σ l_kv_decode + b_d * num_decode)
           + t_c
```

With fitted values:

```
T_pd(f, B) = (1/f) * (0.02199 * Σ l_q² + 0 * Σ l_q·l_kv + 210.03 * Σ l_q)
           + (1/f^0.9899) * (0.2765 * Σ l_kv_decode + 175.73 * num_decode)
           + 18.02   [ms]
```

where f is in MHz and T is in milliseconds.

### Example predictions

**Single prefill request** (l_q=512, l_kv=0) at 1500 MHz:
```
T = (1/1500) * (0.02199 * 512² + 210.03 * 512) + 18.02
  = (1/1500) * (5763.5 + 107535.4) + 18.02
  = 75.53 + 18.02 = 93.55 ms
```

**8 decode requests** (avg l_kv=500) at 1500 MHz:
```
T = (1/1500^0.9899) * (0.2765 * 4000 + 175.73 * 8) + 18.02
  = (1/1462.9) * (1106.0 + 1405.8) + 18.02
  = 1.72 + 18.02 = 19.74 ms
```
