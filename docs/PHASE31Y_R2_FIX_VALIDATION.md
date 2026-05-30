# Phase 31Y-R2 Fix Validation

## Header
- **Phase:** 31Y-R2
- **Date:** 2026-05-29
- **Classification:** FAIL_STREAM_STILL_BROKEN
- **Repo:** sdi-substitutive

## Bugs Fixed

### BUG 1: Bitmap dimension swap in `sdir_streaming_apply`
- **File:** `src/phase31x_manifest_runtime.py`
- **Symptom:** `bitmap[row * in_dim + in_dim]` instead of `bitmap[row * out_dim + col]`
- **Fix:** Changed stride from `in_dim` to `out_dim` — the bitmap is row-major with shape `(in_dim, out_dim) = (896, 4864)`

### BUG 2: W_ref construction differs from Phase 31T
- **File:** `src/phase31y_multilayer_sweep.py` (`build_layer_entry()`)
- **Symptom:** Phase 31Y used `U @ V * 0.1` (rank-448 approximation), Phase 31T used `rng.randn(896, 4864) * 0.1` (rank-896 full matrix)
- **Also fixed:** quantization scale 7.5 → 7.0, k_percent 7.5 → 7.0 to match Phase 31T
- **Seeds also aligned:** layer 0 seed = 0 (matching Phase 31T's W_REF_SEEDS[0]=0)

## Validation Results (Layer 0, 896x4864)

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| cosine(Y_ref, Y_sub_stream) | 0.75686961 | > 0.9999 | ✗ |
| MAE | 1.946888e+03 | < 1e-3 | ✗ |
| max_abs_diff | 1.380182e+04 | — | — |
| NaN/Inf | False | false | ✓ |

## Memory
| Item | Bytes |
|------|-------|
| W_low packed | 2,179,072 |
| W_low scale | 544,768 |
| Residual (k_pct=7.0) | 1,154,936 |
| Total substitutive | 3,878,776 |
| W_ref avoided | 17,432,576 |

## Phase 31Z Freeze Status
NOT SAFE — Phase 31Z freeze is blocked pending fix verification.

## Files Changed
- `src/phase31x_manifest_runtime.py`: BUG 1 fix (bitmap stride)
- `src/phase31y_multilayer_sweep.py`: BUG 2 fix (W_ref construction, scale=7.0, k_pct=7.0)
