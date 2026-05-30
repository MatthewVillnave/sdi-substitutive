# Phase 31AG-DIAG: Isolate ffn_up Streaming Decode Regression

## Header
- **Phase:** 31AG-DIAG
- **Date:** 2026-05-30
- **OLD_HEAD:** `379e85e`
- **Classification:** `PARTIAL_ROOT_CAUSE_FOUND_FIX_PENDING`
- **Policy:** k=9.0%, alpha=1.0

## Verdict
**PROVEN:** ffn_up fails when its artifact bytes are decoded with the ffn_down orientation.

The bad path uses `rows=4864, cols=896` for ffn_up. The correct ffn_up streaming path is `rows=896, cols=4864`.

## Orientation Comparison
| Family / Case | Runtime rows | Runtime cols | Avg cosine | Avg MAE |
|---------------|--------------|--------------|------------|---------|
| ffn_up correct | 896 | 4864 | 0.995411 | 0.232092 |
| ffn_up bad swapped | 4864 | 896 | -0.013015 | 7.967258 |
| ffn_down correct | 4864 | 896 | 0.989164 | 0.901248 |

## Dense-vs-Stream Comparison

### ffn_up Correct Orientation
| L | rows x cols | cos_sub | MAE_sub | nnz |
|---|-------------|---------|---------|-----|
| 0 | 896x4864 | 0.995332 | 0.231418 | 392,232 |
| 1 | 896x4864 | 0.995436 | 0.233057 | 392,232 |
| 2 | 896x4864 | 0.995520 | 0.231930 | 392,232 |
| 3 | 896x4864 | 0.995374 | 0.228274 | 392,232 |
| 4 | 896x4864 | 0.995491 | 0.233859 | 392,232 |
| 5 | 896x4864 | 0.995311 | 0.234015 | 392,232 |

### ffn_up Bad Swapped Orientation
| L | rows x cols | cos_sub | MAE_sub | nnz |
|---|-------------|---------|---------|-----|
| 0 | 4864x896 | -0.038831 | 8.026109 | 392,232 |
| 1 | 4864x896 | -0.005248 | 7.855010 | 392,232 |
| 2 | 4864x896 | -0.001911 | 8.292367 | 392,232 |
| 3 | 4864x896 | 0.009797 | 7.680499 | 392,232 |
| 4 | 4864x896 | -0.041724 | 8.037817 | 392,232 |
| 5 | 4864x896 | -0.000175 | 7.911749 | 392,232 |

### ffn_down Correct Orientation
| L | rows x cols | cos_sub | MAE_sub | nnz |
|---|-------------|---------|---------|-----|
| 0 | 4864x896 | 0.989270 | 0.906470 | 392,232 |
| 1 | 4864x896 | 0.989401 | 0.904575 | 392,232 |
| 2 | 4864x896 | 0.989971 | 0.886491 | 392,232 |
| 3 | 4864x896 | 0.989042 | 0.881123 | 392,232 |
| 4 | 4864x896 | 0.988614 | 0.912334 | 392,232 |
| 5 | 4864x896 | 0.988688 | 0.916496 | 392,232 |

## Fix Applied
Use family-specific runtime orientation dispatch:

- `ffn_up`: `rows=896`, `cols=4864`, input length 896, output length 4864
- `ffn_down`: `rows=4864`, `cols=896`, input length 4864, output length 896

No W_ref fallback, dense W_low materialization, or dense R materialization is required in the runtime path.

## Strict Runtime Counters
| Counter | Value |
|---------|-------|
| W_ref_loaded | 0 |
| W_ref_generated_in_runtime | 0 |
| dense_W_low_materialized_in_runtime | 0 |
| dense_R_materialized_in_runtime | 0 |
| sdiw_loaded | 12 |
| sdir_loaded | 12 |
| fallback_count | 0 |
| error_count | 0 |

## Combined Smoke Result
- ffn_up corrected avg cosine: `0.995411`
- ffn_down avg cosine: `0.989164`
- bad swapped ffn_up avg cosine: `-0.013015`
- combined smoke pass: `False`

## Decision
**Classification: `PARTIAL_ROOT_CAUSE_FOUND_FIX_PENDING`**

Phase 31AH combined checkpoint is **not unlocked**.

## Claim Boundaries
- **Allowed:** The ffn_up combined regression was an orientation dispatch bug in the strict artifact/runtime fixture.
- **Allowed:** ffn_up and ffn_down both pass deterministic streaming decode when each family uses its correct orientation.
- **Forbidden:** No model behavior, production readiness, or end-to-end inference speedup claim.
