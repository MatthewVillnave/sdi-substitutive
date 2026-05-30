# Phase 31AF: Combined ffn_up + ffn_down Strict Substitutive Artifact Feasibility

## Header
- **Phase:** 31AF
- **Date:** 2026-05-30
- **Classification:** `PARTIAL_FFN_DOWN_PASS_FFN_UP_WEAK`
- **OLD_HEAD:** `e032964`
- **Repo:** `sdi-substitutive`
- **Official residual policy:** k=9%, alpha=1.0

## Result
**ffn_down passes. ffn_up fails in combined runtime.**

## Layer Results

### ffn_up (BROKEN)
| L | cos_sub | MAE_sub | margin |
|---|---------|---------|--------|
| 0 | -0.0256 | 2.2457 | 305,044 |
| 1 | +0.0337 | 2.3051 | 305,044 |
| 2 | -0.0540 | 2.3842 | 305,044 |
| 3 | +0.0124 | 2.3183 | 305,044 |
| 4 | -0.0424 | 2.3715 | 305,044 |
| 5 | -0.0114 | 2.3461 | 305,044 |

**avg cos_sub: -0.0146 | avg MAE_sub: 2.33** — completely broken, cosine near 0.

### ffn_down (PASSES)
| L | cos_sub | MAE_sub | margin |
|---|---------|---------|--------|
| 0 | 0.9964 | 0.1491 | 305,044 |
| 1 | 0.9963 | 0.1520 | 305,044 |
| 2 | 0.9961 | 0.1590 | 305,044 |
| 3 | 0.9961 | 0.1504 | 305,044 |
| 4 | 0.9961 | 0.1471 | 305,044 |
| 5 | 0.9961 | 0.1506 | 305,044 |

**avg cos_sub: 0.9962 | avg MAE_sub: 0.1514** — passes research gate.

## Critical Finding: ffn_up Regression

ffn_up worked in Phase 31AA (cos=0.9956) but is broken in combined test (cos=-0.0146).
The streaming decode produces near-random output for ffn_up in combined configuration.

**Hypothesis:** The ffn_up sdiw/sdir decode path is orientation-sensitive and fails when both families share the same streaming decode call (rows=4864, cols=896 for both). The ffn_down uses transposed orientation naturally; ffn_up may require different decode orientation that was previously masked or coincidentally working.

## Strict Counters (combined)
- W_ref_loaded=0 ✓
- W_ref_generated=0 ✓
- W_ref_dense_bytes=0 ✓
- dense_W_low_materialized=0 ✓
- dense_R_materialized=0 ✓
- sdiw_loaded=12 ✓
- sdir_loaded=12 ✓
- fallback_count=0 ✓
- error_count=0 ✓

## Memory (combined, k=9%)
- ffn_up margin: 1,830,264 bytes (6 layers × 305,044)
- ffn_down margin: 1,830,264 bytes (6 layers × 305,044)
- Combined margin: 3,660,528 bytes
- Both margins positive ✓

## Decision
**Classification: PARTIAL_FFN_DOWN_PASS_FFN_UP_WEAK**

ffn_down passes all gates. ffn_up fails — streaming decode regression, not a memory or counter issue.

## Next Steps
1. Investigate ffn_up streaming decode orientation failure before expanding
2. ffn_down path is viable and clean
3. ffn_up-only path may be viable as initial target if ffn_up decode issue is isolated

## Claim Boundaries
**Allowed:** "ffn_down substitutive runtime passes; ffn_up has streaming decode regression in combined config."
**Forbidden:** no full FFN claim, no model behavior, no production readiness.
