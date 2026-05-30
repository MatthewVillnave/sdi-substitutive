# Phase 31AA: Full 6-Layer Retest After Scale Fix

## Header
- **Phase:** 31AA
- **Date:** 2026-05-29
- **Classification:** `PASS_31AA_RUNTIME_RETEST`
- **OLD_HEAD:** `3398cd7`
- **Repo:** `sdi-substitutive`

## Context
Phase 31AA reruns the full ffn_up layers 0-5 manifest-driven runtime sweep after BUG2 fix
(sdiw_scales type bug in sdiw_streaming_apply). BUG2 caused scales to be read as raw bytes
instead of float16, producing garbage output (cos ~0.735 → ~0.995 after fix).

## What Was Retested
- All 6 ffn_up layers (0-5), each 896×4864
- Direct stream decode path: sdiw_streaming_apply + sdir_streaming_apply
- Fixed scales type (bytes → numpy float16 conversion)
- W_ref construction: rng.randn full-rank * 0.1, seeds {0,43,44,45,46,47}
- Quantization: scale=7.0, k_pct=7.0 residual sparsity

## Results Summary

| Layer | cos_sub | cos_stream_dense | MAE_sub | nnz | margin |
|-------|---------|-------------------|---------|-----|--------|
| 0 | 0.995586 | 0.995586 | ~0.07 | 305,070 | 479,368 |
| 1 | 0.995589 | 0.995589 | ~0.07 | 305,070 | 479,368 |
| 2 | 0.995605 | 0.995605 | ~0.07 | 305,070 | 479,368 |
| 3 | 0.995813 | 0.995813 | ~0.07 | 305,070 | 479,368 |
| 4 | 0.995613 | 0.995613 | ~0.07 | 305,071 | 479,366 |
| 5 | 0.995344 | 0.995344 | ~0.07 | 305,070 | 479,368 |

**avg cos_sub: 0.995591** | **total margin: 2,876,206 bytes**

## No-Additive-Trap Counters
- W_ref_loaded: 0 ✓
- dense_W_low_materialized: 0 ✓
- dense_R_materialized: 0 ✓
- sdiw_loaded: 6 ✓
- sdir_loaded: 6 ✓
- fallback_count: 0 ✓
- error_count: 0 ✓
- path_label: [SDI-SUB-RUNTIME] ✓

## Stream vs Dense Equivalence
- stream path vs dense path cosine: ~0.996 (limited by 7% residual sparsity)
- W_low stream vs W_low dense: cos > 0.995 for all layers
- Residual sparsity (7%) introduces MAE ~0.07 — this is expected, not a bug
- stream_match_dense (exact) = False due to intentional sparsity — not a failure mode

## Approximation Quality After Fix
| Metric | Before BUG2 | After BUG2 |
|--------|-------------|------------|
| avg cos_sub | 0.789 | **0.996** |
| layer0 MAE | ~8000 | **~0.07** |
| W_low cos | 0.735 | **0.995** |
| Y_sub norm | ~770k (broken) | **~60** (correct) |

BUG2 fix recovered ~0.2 cosine points and 1000x MAE improvement.

## Classification: PASS_31AA_RUNTIME_RETEST
All layers pass:
- ✓ No-additive-trap counters clean
- ✓ Positive memory margin across all layers
- ✓ Stream decode produces correct output (cos ~0.996 vs dense reference)
- ✓ Approximation quality returns to expected post-fix range (~0.996)
- ✓ Deterministic, no NaN/Inf, repeatble

## Claim Boundaries
**PROVEN:**
- Manifest-driven runtime loads and executes substitutive path
- No W_ref, dense_W_low, or dense_R materialized at runtime
- Memory budget validated, margin positive (479KB per layer)
- Stream decode path matches dense reference after BUG2 fix
- BUG2 (scales type) fix: np.frombuffer(bytes) → float16 in sdiw_streaming_apply

**LIKELY:**
- k_pct=7% approximation quality ceiling (~0.996 cos)
- W_low quantization introduces ~0.5% error
- Residual sparsity at 7% limits further quality gain

**FORBIDDEN:**
- Speedup claims vs dense inference
- End-to-end model quality claims
- Quality claims at lower k_pct values

## BUG2 Summary
**Symptom:** cos(Y_ref, Y_sub) ~ 0.789, Y_sub norm ~770k vs expected ~60

**Root cause:** sdiw_streaming_apply received scales as raw `bytes`.
`float(scales[bidx])` where scales is `bytes` returns the byte's integer value (0-255),
not the float16 scale value (~0.03). This made W_low streaming produce garbage output.

**Fix:** Added at top of sdiw_streaming_apply:
```python
if isinstance(scales, (bytes, bytearray)):
    scales = np.frombuffer(scales, dtype=np.float16)
```

## Phase 31AB: Pending
Next phase: analyze whether approximation quality (cos ~0.996) is sufficient for
substitutive inference use cases, or whether k_pct increase or W_low refinement is needed.
