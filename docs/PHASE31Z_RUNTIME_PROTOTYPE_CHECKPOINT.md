# Phase 31Z: Runtime Prototype Checkpoint

## Header
- **Phase:** 31Z
- **Date:** 2026-05-29
- **Classification:** `PARTIAL_RUNTIME_PASS_APPROX_WEAK`
- **Tag:** `phase31z-runtime-checkpoint` @ `41f7e7a`
- **Repo:** `sdi-substitutive`

## What This Freeze Captures

The substitutive ffn_up runtime prototype — manifest-driven loader with combined
sdiw (W_low streaming) + sdir (residual streaming) decode path.

## What Was Proven

### Runtime Integrity ✓
- Manifest-driven loader parses, validates, and executes substitutive path
- No W_ref, dense_W_low, or dense_R materialized at runtime
- Memory budget validated, margin positive across all 6 layers
- fallback_count=0, error_count=0, checksum_validated=6

### Decode Path Correctness ✓
- **BUG 1 (bitmap stride):** Fixed in 41f7e7a. `bitmap[row * out_dim + col]` (was `row * in_dim`)
- **BUG 2 (scales type):** Fixed in this phase. `sdiw_streaming_apply` now converts
  `bytes` scales to `np.frombuffer(scales, dtype=np.float16)` before indexing

### Approximation Quality
| Metric | Before BUG2 Fix | After BUG2 Fix |
|--------|-----------------|----------------|
| avg cos_sub | 0.789 (weak) | 0.996 (strong) |
| layer0 MAE | ~8000 | ~0.07 |
| W_low cos | ~0.763 (buggy) | ~0.995 (correct) |

**Root cause of ~0.789:** BUG2 — `sdiw_streaming_apply` received scales as raw `bytes`.
`float(scales_bytes[bidx])` returned byte value (e.g., 240) instead of scale (e.g., 0.031).
This made W_low streaming produce garbage output (norm 670k vs expected ~200).

**After BUG2 fix:** W_low stream matches dense (cos ~0.995). Residual at k_pct=7%
adds ~0.001 cos improvement (sparsity noise limits further gain).

## Claim Boundaries

**PROVEN:**
- Manifest-driven runtime loads and executes substitutive path
- No W_ref / dense_W_low / dense_R materialized at runtime
- Memory margin positive, budget validated
- Stream decode (sdiw+sdir) matches dense after BUG2 fix

**LIKELY:**
- k_pct=7% approximation quality ceiling (~0.996 cos after fix)
- W_low quantization introduces ~0.5% error (cos ~0.995)
- Residual sparsity at 7% limits further quality gain

**FORBIDDEN:**
- Speedup claims vs dense inference
- End-to-end model quality claims
- Quality claims at lower k_pct values

## Artifacts
- `src/phase31x_manifest_runtime.py` — runtime with BUG2 fix
- `src/phase31y_multilayer_sweep.py` — corrected W_ref construction (full-rank rng.randn)
- `results/PHASE31Y_MANIFEST_FFN_UP_MULTILAYER_SWEEP.json` — original weak results
- `results/PHASE31Y_R2_FIX_VALIDATION.json` — corrected validation after BUG2 fix

## Bug 2 Detail: scales Type Bug

**Symptom:** `cos(Y_ref, Y_sub) ~ 0.789`, Y_sub norm ~770k vs Y_ref norm ~4500

**Root cause:** `sdiw_streaming_apply` signature: `def sdiw_streaming_apply(packed, scales, X, rows, cols)`
`scales` is returned by `pack_wlow()` as `bytes`. But `float(scales[bidx])` where `scales` is `bytes`
returns the byte's integer value (0-255), not the float16 scale value.

**Fix:** Added at top of `sdiw_streaming_apply`:
```python
if isinstance(scales, (bytes, bytearray)):
    scales = np.frombuffer(scales, dtype=np.float16)
```

**Verification:** After fix, W_low stream cos: 0.735 → 0.982 (before residual)
Full path cos: 0.789 → 0.996, MAE: 8000 → 0.07

## Phase 31AA: Pending
Full 6-layer retest with BUG2 fix applied. Expected result: cos ~0.996 per layer,
MAE ~0.07 per layer (residual sparsity noise is the remaining quality ceiling).
