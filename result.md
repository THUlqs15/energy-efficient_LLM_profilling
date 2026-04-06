# Batch Latency Model — Fitted Results

## Model

**GPU:** NVIDIA A10G (AWS cloud instance, CUDA 13.0, Driver 580.126.09)  
**vLLM version:** 0.19.0 (v1 engine, eager mode)  
**Model profiled:** Qwen3-8B (at `/home/ubuntu/lqs/L3`), bfloat16, max_model_len=2048

## Fitted Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a_p | 6.974e-02 | Prefill: quadratic term for Σ l_q² |
| b_p | 0.000e+00 | Prefill: cross-term for Σ l_q × l_kv |
| c_p | 3.395e+02 | Prefill: linear term for Σ l_q |
| a_d | 2.112e-01 | Decode: linear term for Σ l_kv |
| b_d | 4.822e+01 | Decode: constant per decode request |
| α (alpha) | 0.8167 | Decode frequency scaling exponent |
| t_c | 5.705e+01 | Batch-level constant overhead (ms) |

## GPU Frequency Information

- **GPU model:** NVIDIA A10G
- **Available frequency range:** 210–1710 MHz (101 discrete steps)
- **Effective profiling frequency:** 1710 MHz (only)
  - This AWS cloud instance does not honor `nvidia-smi --lock-gpu-clocks` or `-ac` commands; the GPU remains at its maximum boost clock of 1710 MHz regardless of settings.
  - Clock locking was attempted for 210 MHz and 1710 MHz — both resulted in actual clock = 1710 MHz.
  - Accordingly, `alpha` (the decode frequency scaling exponent) is **not identifiable from single-frequency data**. Its fitted value of **0.817** is consistent with a lightly memory-bound regime (0 = compute-bound, 1 = fully memory-bound) and close to theoretical predictions for decode-phase attention, but carries large uncertainty without multi-frequency data.

## Evaluation Metrics

| Metric | Train | Test |
|--------|-------|------|
| MAPE | 1.92% | 1.42% |
| MAE | 1.97 ms | 1.90 ms |
| RMSE | 6.09 ms | 5.72 ms |
| R² | 0.9974 | 0.9936 |

Train/test split was random (seed=42) to ensure representative distributions.

## Data Summary

- **Total batches recorded:** 596
- **Train/Test split:** 476 / 120
- **GPU frequency range profiled:** [1710, 1710] MHz (single frequency)
- **Wall time range:** 57.1 ms – 1862.7 ms
- **Wall time mean:** 77.6 ms
- **Batches with prefill:** 87 (14.6%)
- **Batches with decode only:** 512 (85.9%)  
- **Mixed batches:** 3 (0.5%)

**Request composition:**
- Total prefill requests: 207 (across all batches)
- Total decode requests: 2800 (across all batches)
- Prefill l_q range: 10–1331 tokens
- Prefill l_kv: all 0 (all first-pass prefills, no cached context)
- Decode num_requests per batch: 1–7
- Decode sum(l_kv) range: 41–2591

**Note on b_p:** The cross-term coefficient b_p = 0 because all prefill requests in this dataset have l_kv = 0 (fresh prefills with no KV cache). In a multi-turn scenario with cached prefill context, b_p would capture the attention cost of attending to prior KV cache during prefill.

## Usage

The batch execution time model (f in MHz, times in ms):

```
T_pd(f, B) = (1/f) * (6.974e-02 * Σ l_q² + 0 * Σ l_q·l_kv + 3.395e+02 * Σ l_q)
           + (1/f^0.8167) * (2.112e-01 * Σ l_kv + 4.822e+01 * num_decode)
           + 5.705e+01
```

At the profiled frequency f = 1710 MHz:

```
T_pd(1710, B) = (5.833e-02 * Σ l_q²/1710 + 1.985e-01 * Σ l_q) 
              + (1.265e-04 * Σ l_kv + 2.892e-02 * num_decode)
              + 57.05
```

The dominant terms are:
- `c_p/f * Σ l_q = 0.1985 * Σ l_q` ms — linear prefill cost (~0.2 ms per token)
- `t_c = 57.05` ms — fixed per-batch overhead (scheduling, memory ops, kernel launch)
- `b_d/f^α * num_decode = 48.22/1710^0.817 * num_decode` — per-request decode overhead

## Caveats

1. **Single frequency**: alpha is not well-determined from single-frequency data. Its value should be validated against multi-frequency experiments on a system with controllable clocks.
2. **No KV-cache prefill data**: b_p is zero because no prefill requests with cached context were observed. Multi-turn scenarios would require additional data collection.
3. **vLLM chunked prefill**: The engine uses chunked prefill with max_num_batched_tokens=8192, so long prompts are split across multiple forward passes. The l_q values logged reflect the actual tokens processed per forward pass, not the original prompt length.
4. **t_c overhead**: The 57ms constant is larger than expected for pure kernel overhead, likely incorporating scheduling latency and IPC overhead from vLLM's v1 engine architecture (which uses a separate EngineCore subprocess).
