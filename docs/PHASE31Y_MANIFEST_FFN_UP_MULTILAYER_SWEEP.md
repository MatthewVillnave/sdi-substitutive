# Phase 31Y: Manifest-Driven ffn_up Layers 0-5 Sweep

## Header
- **OLD_HEAD:** 4e9906e
- **NEW_HEAD:** (pending commit)
- **Classification:** PARTIAL_RUNTIME_PASS_APPROX_WEAK
- **Phase:** 31Y
- **Date:** 2026-05-29

## What 31Y Proves
Manifest-driven bundle runtime operates correctly across 6 ffn_up layers (0-5).
Each layer: manifest loads, artifacts validate, substitutive path executes,
W_ref absent, dense W_low absent, dense R absent, margins positive.

## Memory Table
| Layer | Q4 Budget | W_low Packed | Residual | Total Sub | Margin | dense_W_low Avoided | dense_R Avoided |
|-------|-----------|--------------|----------|-----------|--------|---------------------|-----------------|
| 0 | 4,358,144 | 2,179,072 | 1,198,518 | 3,922,358 | 435,786 | 17,432,576 | 17,432,576 |
| 1 | 4,358,144 | 2,179,072 | 1,198,516 | 3,922,356 | 435,788 | 17,432,576 | 17,432,576 |
| 2 | 4,358,144 | 2,179,072 | 1,198,516 | 3,922,356 | 435,788 | 17,432,576 | 17,432,576 |
| 3 | 4,358,144 | 2,179,072 | 1,198,516 | 3,922,356 | 435,788 | 17,432,576 | 17,432,576 |
| 4 | 4,358,144 | 2,179,072 | 1,198,518 | 3,922,358 | 435,786 | 17,432,576 | 17,432,576 |
| 5 | 4,358,144 | 2,179,072 | 1,198,516 | 3,922,356 | 435,788 | 17,432,576 | 17,432,576 |

## Approximation Table (Honest Reporting)
| Layer | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | max_error_sub | nan/inf |
|-------|---------|---------|-----------|---------|---------|---------------|---------|
| 0 | 0.988208 | 0.799396 | +0.000041 | 8.5207e+00 | 8.0922e+03 | 5.6716e+04 | False |
| 1 | 0.985677 | 0.779241 | +0.000050 | 8.5672e+00 | 7.5777e+03 | 4.8089e+04 | False |
| 2 | 0.987540 | 0.800209 | +0.000049 | 8.7025e+00 | 8.1011e+03 | 5.3887e+04 | False |
| 3 | 0.986117 | 0.785602 | +0.000055 | 8.6123e+00 | 7.6355e+03 | 4.7800e+04 | False |
| 4 | 0.984506 | 0.777313 | +0.000055 | 8.5800e+00 | 7.2981e+03 | 5.1768e+04 | False |
| 5 | 0.986354 | 0.793306 | +0.000059 | 8.7204e+00 | 7.7749e+03 | 4.7654e+04 | False |

## No-Additive-Trap Counters (all layers)
| Counter | Value |
|---------|-------|
| W_ref_loaded | 0 |
| dense_W_low_materialized | 0 |
| dense_R_materialized | 0 |
| sdiw_loaded | 6 |
| sdir_loaded | 6 |
| manifest_loaded | 1 |
| checksum_validated | 6 |
| memory_budget_validated | 6 |
| fallback_count | 0 |
| error_count | 0 |
| path_label | [SDI-SUB-RUNTIME] |

## Fail-Fast Regression
- missing_manifest: PASS
- missing_layer_99: PASS

## Approximation Honesty Note
Average cos_sub across 6 layers: 0.7892
Approximation is WEAK (avg cos_sub < 0.85).
This classification REFLECTS WEAK_APPROXIMATION.
Runtime correctness and memory margins are independently validated.

## Phase 31Z Unlock
YES — Phase 31Z unlocked (requires fix first)

## Files Added
- results/PHASE31Y_MANIFEST_FFN_UP_MULTILAYER_SWEEP.json
- docs/PHASE31Y_MANIFEST_FFN_UP_MULTILAYER_SWEEP.md
