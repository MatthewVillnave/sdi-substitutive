# Phase 31AP — Q2_K Encode via llama.cpp Quantize_row_q2_K

## Classification
**`PASS_Q2K_LLAMA_QUANT_LOWK_IMPROVES`**

## Status
Phase 31AP probe complete.

## Source
llama.cpp `quantize_row_q2_K_ref` + `dequantize_row_q2_K` via ctypes from:
`/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so`

## Format Details
- QK_K = 256 (block size in elements)
- Block size = 84 bytes: scales[16] + qs[64] + d[2] + dmin[2]
- Q2_K effective bpe = 2.625
- d = super-block scale for quantized scales (fp16)
- dmin = super-block scale for quantized mins (fp16)

## Byte-Size Validation

| Item | Value |
|------|-------|
| n_elements | 4,358,144 |
| n_blocks | 17,024 |
| expected bytes | 1,430,016 (1396.5 KB) |
| actual bytes | 1,430,016 (1396.5 KB) |
| byte match | True |
| effective bpe | 2.625000 |

## Memory Analysis

| Item | Value |
|------|-------|
| Q4 budget | 2,179,072 (2128.0 KB) |
| Q2_K bytes | 1,430,016 (1396.5 KB) |
| margin | 749,056 (731.5 KB) |
| memory-positive | True |

## Numerical Quality (layer0 ffn_up)

| Metric | Value |
|--------|-------|
| cos(W_ref, W_q2k) | 0.95667356 |
| MAE | 0.00360368 |
| max error | 0.04343033 |
| norm_ref | 31.290634 |
| norm_q2k | 32.171089 |

## Comparison with Prior Phases

| Phase | Format | cos | MAE | Notes |
|-------|--------|-----|-----|-------|
| 31AM | simulated Q2-like | 0.9260 | 0.00904 | block_size=16 simulation |
| 31AN | actual IQ4_NL/Q3_K | 0.996 | 0.001 | memory FAILS |
| **31AP** | **actual Q2_K** | **0.956674** | **0.00360368** | **llama.cpp encode** |

## Low-k Residual Test (layer0 ffn_up)

| k | W_low bytes | res bytes | total | margin | cos_low | MAE_low | delta_cos | improves |
|---|------------|----------|-------|--------|---------|---------|-----------|----------|
| 0 | 1,430,016 | 0 | 1,430,016 | 749,056 | 0.956674 | 0.003604 | +0.000000 | N/A |
| 0.5 | 1,430,016 | 588,348 | 2,018,364 | 160,708 | 0.956674 | 0.003604 | +0.001939 | YES |
| 1 | 1,430,016 | 631,930 | 2,061,946 | 117,126 | 0.956674 | 0.003604 | +0.003258 | YES |
| 2 | 1,430,016 | 719,092 | 2,149,108 | 29,964 | 0.956674 | 0.003604 | +0.005493 | YES |
| 3 | 1,430,016 | 806,256 | 2,236,272 | -57,200 | 0.956674 | 0.003604 | +0.007432 | YES |

## Classification Reasoning

- **byte_match**: True — Q2_K encode produces exact expected byte count
- **no NaN/Inf**: True
- **margin_positive**: True — Q2_K is memory-positive (+731.5 KB)
- **residual_on_improves_any**: True

## Key Finding

True Q2_K (via llama.cpp) with actual decode produces cos=0.9567, MAE=0.003604.
This is significantly better than the 31AM simulated Q2-like (cos≈0.926) and slightly below 31AN's IQ4_NL (cos≈0.996).
Q2_K is memory-positive while 31AN's IQ4_NL was not.

## SOURCE_OF_TRUTH Update

- Section 3: Add 31AP accepted fact
- Section 9: Update to next allowed phase
