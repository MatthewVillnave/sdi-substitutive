# Phase 31AG-R0: Runtime-Consistent Residual Reconciliation

## Header
- **Phase:** 31AG-R0
- **Date:** 2026-05-30
- **OLD_HEAD:** `df3c0d5`
- **Classification:** `PARTIAL_RUNTIME_CONSISTENT_FFN_UP_ONLY`
- **Policy:** k=9.0%, alpha=1.0

## Root Cause

The source-of-truth residual for the substitutive path was computed from raw/ideal W_low
instead of the runtime-decoded W_low. The artifact generation used:

```
R = W_ref - W_low_raw   # WRONG: W_low_raw is ideal float, not runtime-decoded
```

But the actual runtime path decodes the packed W_low before computing the residual
contribution, so the correct residual must be:

```
W_low_decoded = decode(packed W_low)
R_runtime = W_ref - W_low_decoded   # CORRECT
```

This caused a mismatch: the residual was encoded from a different W_low than what the
runtime actually used, leading to suboptimal approximation quality.

## Residual Source-of-Truth Audit

All prior phases used `R = W_ref - W_low` where W_low is the raw/ideal quantized matrix.
This includes:
- `phase31x_manifest_runtime.py` `execute_substitutive_path` (line 142)
- `generate_test_bundle.py` (line 54)
- `phase31y_r_debug.py` (line 31)
- `phase31o_real_artifact.py` (line 75)
- `residual_make.py` (lines 170, 171, 333)
- `artifact_write.py` (line 62)
- All prior 31AA, 31AB, 31AC, 31AD, 31AE, 31AF results

**The correct method** (runtime-consistent) is:
```python
def decode_wlow_matrix(packed, scales, rows, cols):
    scales = np.frombuffer(scales, dtype=np.float16)
    W = np.zeros((rows, cols), dtype=np.float32)
    for b in range((rows * cols) // 32):
        scale = float(scales[b])
        bb = b * 16
        base = b * 32
        for i in range(16):
            byte = packed[bb + i]
            W.flat[base + 2*i] = (float(byte & 0x0F) - 8.0) * scale
            W.flat[base + 2*i + 1] = (float((byte >> 4) & 0x0F) - 8.0) * scale
    return W

W_low_decoded = decode_wlow_matrix(packed, scales_b, rows, cols)
R_runtime = W_ref - W_low_decoded
```

## Corrected Results

### FFN_UP (rows=896, cols=4864, k=9.0%)
| L | cos_sub | cos_low | delta_cos | MAE_sub | MAE_low | nnz |
|---|---------|---------|-----------|---------|---------|-----|
| 0 | 0.995620 | 0.951891 | +0.043729 | 4.8651 | 19.3077 | 392232 |
| 1 | 0.995189 | 0.947079 | +0.048110 | 4.9250 | 19.3264 | 392232 |
| 2 | 0.994917 | 0.945065 | +0.049852 | 4.9204 | 19.4860 | 392232 |
| 3 | 0.995355 | 0.948937 | +0.046417 | 4.9077 | 19.3857 | 392232 |
| 4 | 0.995258 | 0.948241 | +0.047017 | 4.8472 | 19.3580 | 392232 |
| 5 | 0.995394 | 0.950224 | +0.045170 | 4.9178 | 19.4813 | 392232 |

**avg_cos_sub: 0.995289** | avg_cos_low: 0.948573 | avg_MAE_sub: 4.8972 | avg_MAE_low: 19.3908

### FFN_DOWN (rows=4864, cols=896, k=9.0%)
| L | cos_sub | cos_low | delta_cos | MAE_sub | MAE_low | nnz |
|---|---------|---------|-----------|---------|---------|-----|
| 0 | 0.991222 | 0.786604 | +0.204619 | 11.0769 | 74.5810 | 392232 |
| 1 | 0.991731 | 0.792934 | +0.198797 | 10.2091 | 73.9858 | 392232 |
| 2 | 0.990392 | 0.783023 | +0.207369 | 10.9906 | 75.2760 | 392232 |
| 3 | 0.993347 | 0.818501 | +0.174845 | 10.5635 | 75.0677 | 392232 |
| 4 | 0.989829 | 0.756305 | +0.233523 | 10.7993 | 75.7380 | 392232 |
| 5 | 0.992037 | 0.800407 | +0.191629 | 10.7358 | 74.7197 | 392232 |

**avg_cos_sub: 0.991426** | avg_cos_low: 0.789629 | avg_MAE_sub: 10.7292 | avg_MAE_low: 74.8947

## Strict Runtime Counters
| Counter | Value |
|---------|-------|
| W_ref_loaded | 0 |
| W_ref_generated_in_runtime | 0 |
| W_ref_dense_bytes | 0 |
| dense_W_low_materialized | 0 |
| dense_R_materialized | 0 |
| sdiw_loaded | 12 |
| sdir_loaded | 12 |
| fallback_count | 0 |
| error_count | 0 |

## Prior Claim Amendment

All prior phases (31AA, 31AB, 31AC, 31AD, 31AE, 31AF) that reported cosine > 0.995
for ffn_up and > 0.99 for ffn_down using raw W_low residuals **need revision**.

- The residual bug inflates cosine numbers because the residual doesn't match
  what the runtime actually applies.
- With runtime-consistent residual: ffn_up cos=0.9953 (PASSES threshold), ffn_down cos=0.9914
- MAE is materially higher than previously reported because the full approximation
  error from quantization is now visible (not masked by residual mismatch).

## Old vs Corrected Comparison
| Family | Prior Result | Corrected | Delta | Status |
|--------|-------------|-----------|-------|--------|
| ffn_up cos_sub | ~0.9956 (31AA) | 0.9953 | -0.0003 | PRESERVED (marginally weaker) |
| ffn_down cos_sub | ~0.9962 (31AE) | 0.9914 | -0.0048 | REVISED DOWN (still positive) |
| ffn_up MAE | ~0.07 (31AC) | 4.90 | +4.83 | REVISED (materially higher) |
| ffn_down MAE | ~0.15 (31AE) | 10.73 | +10.58 | REVISED (materially higher) |

## Classification
`PARTIAL_RUNTIME_CONSISTENT_FFN_UP_ONLY`

ffn_up with runtime-consistent residual passes the cosine threshold (0.9953 > 0.995).
ffn_down passes cosine (0.9914) but the MAE is high and the delta_cos is very large
(+0.20) suggesting the low-rank approximation is much worse for ffn_down than for ffn_up.

Phase 31AH combined checkpoint is NOT yet unlocked.

## Decision

- Do NOT proceed to 31AH yet.
- The ffn_down architecture is valid (delta_cos positive, no NaN/Inf, strict counters clean).
- But the MAE for ffn_down (10.73) is very high compared to ffn_up (4.90).
- This may indicate the low-rank approximation (U @ V * 0.1) used for ffn_down test weights
  has a very different condition number than ffn_up weights.
- Further investigation needed into why ffn_down MAE is so high.
- All prior claims about approximation quality are revised. No prior result should be
  cited without verification using runtime-consistent residual.

## Claim Boundaries
- **PROVEN:** Runtime-consistent residual is required for accurate approximation reporting.
- **PROVEN:** ffn_up with runtime-consistent residual: cos=0.9953, all deltas positive, strict counters clean.
- **PROVEN:** ffn_down with runtime-consistent residual: cos=0.9914, all deltas positive, strict counters clean.
- **UNKNOWN:** Whether the high MAE for ffn_down is a structural issue or a test artifact.
- **FORBIDDEN:** No model behavior, production readiness, or end-to-end speedup claims.
- **FORBIDDEN:** Prior 31AA/31AE cosine numbers should not be cited as current results.