# Phase 31Y-R: Manifest Runtime Approximation Discrepancy Audit

**Classification:** `AUDIT_RED_FLASH`
**Date:** 2026-05-29
**Audit Type:** Numerical Discrepancy
**Phase 31Y HEAD:** 9de0710
**Classification Reference:** Phase 31T showed cos_sub ~0.9956; Phase 31Y showed cos_sub ~0.789. This audit identifies the root causes.

---

## Executive Summary

Phase 31Y's manifest-driven path produces cos_sub ~0.789 (weak) versus Phase 31T's cos_sub ~0.9956 (strong). **Two independent critical bugs explain this discrepancy:**

1. **Dimension swap bug in `sdir_streaming_apply`:** The bitmap is accessed with the wrong row stride (`row * in_dim` instead of `row * out_dim`), corrupting Y_delta direction and magnitude. Cosine between stream and dense residual application: **0.65** (not 1.0).

2. **W_ref construction method incompatibility:** Phase 31Y uses `U @ V * 0.1` (rank-448, norm ~4400) while Phase 31T uses `rng.randn(896, 4864) * 0.1` (rank-896, norm ~208). The matrices are **essentially orthogonal** (cosine ~0.001). Any approximation quality comparison is invalid.

**Phase 31Z freeze: NOT SAFE. Fix both bugs before proceeding.**

---

## Finding 1: Bitmap Indexing Dimension Swap (CRITICAL)

### Location
`phase31x_manifest_runtime.py`, function `sdir_streaming_apply`:
```python
# BUGGY (current code):
if bitmap[row * in_dim + col]:   # in_dim=896, out_dim=4864
    Y[row] += X[col] * values[vp]; vp += 1

# CORRECT:
if bitmap[row * out_dim + col]:  # stride should be out_dim=4864, not in_dim=896
    Y[row] += X[col] * values[vp]; vp += 1
```

### Proof: Tiny Example Fails

```
R = [[1, 2, 3], [4, 5, 6]]   # shape (2, 3)
X = [1, 1]

Y_dense = X @ R = [5, 7, 9]
Y_stream = [0, 4, 11]          # WRONG
Match: FALSE
```

### Proof: Real Layer 0

| Metric | Dense (correct) | Stream (buggy) | Ratio |
|--------|---------------|----------------|-------|
| Y_delta norm | 701.84 | 382.28 | 0.54x |
| mean_abs | 8.52 | 4.68 | 0.55x |
| max_abs_diff | — | 30.23 | — |
| cosine (stream vs dense) | — | 0.6499 | — |

### Why It Happens

The bitmap is flattened row-major of R with shape `(in_dim, out_dim)` = `(896, 4864)`. The correct row stride for bitmap access is `out_dim = 4864`. The buggy code uses `in_dim = 896` as the stride, so:

- At `row=1, col=0`: buggy accesses `bitmap[1*896+0] = bitmap[896]` but correct is `bitmap[1*4864+0] = bitmap[4864]`
- These point to **completely different residual elements**
- Result: Y_delta is a garbled mix of wrong elements → cosine drops to 0.65

### Impact

- MAE_sub ~7500-8100 (catastrophic) — error magnitude from wrong residual application
- delta_cos ≈ +0.00005 (near-zero) — residual barely helps because it's applied to wrong positions
- This alone would cause cos_sub to be much lower even if W_ref matched

---

## Finding 2: W_ref Construction Method Incompatibility (CRITICAL)

### Phase 31T W_ref Construction
```python
rng = np.random.RandomState(SEED)  # seeds {0, 43, 44, 45, 46, 47}
W_ref = rng.randn(896, 4864).astype(np.float32) * 0.1
```
- Method: Direct dense random matrix
- Rank: 896 (full)
- Norm: ~208
- X activation: Real activations from 15 prompts

### Phase 31Y W_ref Construction
```python
seed = layer * 43 + 7  # seeds {7, 50, 93, 136, 179, 222}
np.random.seed(seed)
U = np.random.randn(896, 448) @ np.random.randn(448, 4864) * 0.1
```
- Method: Low-rank decomposition (U @ V)
- Rank: 448 (half of full)
- Norm: ~4400 (21x larger)
- X activation: Synthetic `ones(896)`

### Per-Layer Comparison

| Layer | 31T Seed | 31Y Seed | 31T Norm | 31Y Norm | Norm Ratio | W_ref Cosine | 31T cos_sub | 31Y cos_sub |
|-------|----------|----------|----------|----------|------------|--------------|-------------|-------------|
| 0 | 0 | 7 | 206.6 | 4521.8 | 21.9x | 0.0006 | 0.9957 | 0.7994 |
| 1 | 43 | 50 | 209.0 | 4097.8 | 19.6x | -0.0008 | 0.9956 | 0.7792 |
| 2 | 44 | 93 | 211.2 | 4491.4 | 21.3x | -0.0005 | 0.9957 | 0.8002 |
| 3 | 45 | 136 | 205.5 | 4231.7 | 20.6x | -0.0009 | 0.9957 | 0.7856 |
| 4 | 46 | 179 | 210.6 | 3985.6 | 18.9x | 0.0002 | 0.9956 | 0.7773 |
| 5 | 47 | 222 | 208.5 | 4294.7 | 20.6x | -0.0007 | 0.9956 | 0.7933 |

**Key observations:**
- Norms differ by 19-22x across all layers
- W_ref cosine between 31T and 31Y: **~0 (essentially orthogonal)**
- Y_ref cosine between 31T and 31Y: **~0.01 (essentially orthogonal)**
- The 31Y W_ref is a completely different mathematical object

### Why This Matters

The U@V decomposition produces a rank-448 matrix — only 448 dimensions of variability vs 896 for full-rank. The column space is fundamentally different. When we compute `Y_ref = X @ W_ref`:
- For real activations (31T): X has structure, W_ref has full rank → Y_ref lives in 896-dim space
- For ones vector (31Y): X = [1,1,...,1] → Y_ref = sum of each column of W_ref, projected into 448-dim space

The residual approximation target is completely different, making cos_sub values incomparable.

---

## Finding 3: Quantization Scale Divergence (SECONDARY)

| Phase | Scale Formula | Divisor |
|-------|-------------|---------|
| 31T | `max_abs / 7.0` | 7.0 |
| 31Y | `max_abs / 7.5` | 7.5 |

Impact: Different quantization boundaries, compounding the W_ref mismatch. Lower priority than bugs #1 and #2 but must be reconciled.

---

## Direct vs Manifest Comparison (31Y W_ref, Layer 0)

```
Manifest path (stream): Y_sub = sdiw_streaming_apply(...) + sdir_streaming_apply(...)
Direct path (dense):    Y_sub = X @ W_low + X @ R
max_abs_diff: 49.42
cosine: 0.966
```

Manifest vs direct should be ~1.0. The 0.966 cosine (vs expected 1.0) confirms the bitmap bug corrupts the stream path even for 31Y's own W_ref.

---

## Root Cause Summary

| Rank | Root Cause | Location | Severity | Status |
|------|-----------|----------|----------|--------|
| 1 | Bitmap row stride uses `in_dim` instead of `out_dim` | sdir_streaming_apply | CRITICAL | Proven (tiny example) |
| 2 | W_ref construction: U@V vs rng.randn (orthogonal matrices) | phase31y/31x | CRITICAL | Proven (cosine ~0) |
| 3 | Scale divisor: 7.5 vs 7.0 | phase31y | Secondary | Observed |

---

## What Phase 31Y Actually Proves

**ONLY proves:**
- Manifest bundle loads and validates ✓
- Artifacts are written/read correctly ✓
- Memory margins positive ✓
- W_ref absent, dense W_low absent, dense R absent ✓
- sdiw_streaming_apply path works ✓

**Does NOT prove:**
- Manifest path approximation quality (bitmap bug corrupts residual)
- Any comparability to Phase 31T baseline

---

## Phase 31Z Freeze Status

**❌ NOT SAFE TO FREEZE**

### Required Fixes Before Phase 31Z

1. **Fix `sdir_streaming_apply` bitmap indexing:**
   ```python
   # Change from:
   if bitmap[row * in_dim + col]:
   # To:
   if bitmap[row * out_dim + col]:
   ```

2. **Use 31T W_ref construction method:**
   ```python
   # Replace U @ V construction with:
   rng = np.random.RandomState(SEEDS[layer])  # {0,43,44,45,46,47}
   W_ref = rng.randn(896, 4864).astype(np.float32) * 0.1
   ```

3. **Use 31T quantization scale:**
   ```python
   scale = float(np.abs(block).max()) / 7.0  # not 7.5
   ```

4. **After fixes, verify:**
   - `sdir_streaming_apply` stream=dense within tolerance (cosine > 0.9999)
   - W_ref matches 31T baseline exactly (for same seeds)
   - cos_sub comparable to 31T results (~0.995)

### After Fixes, Next Steps

- Re-run Phase 31Y sweep with fixed code
- Compare against Phase 31T artifacts (same seeds, same W_ref)
- If stream=dense and W_ref matches: reconcile approximation quality
- Phase 31Z freeze only after both verification criteria pass

---

## Red Lines for Phase 31Z

- ❌ Do NOT claim 31Y cos_sub ~0.9956 matches 31T
- ❌ Do NOT claim Phase 31Z represents progress over 31T
- ❌ Do NOT claim approximation quality is validated
- ❌ Do NOT freeze Phase 31Z until bugs fixed and reconciled
- ❌ Do NOT commit 31Y results as valid approximation baseline

---

## Files Produced

- `results/PHASE31Y_R_APPROX_DISCREPANCY_AUDIT.json`
- `docs/PHASE31Y_R_APPROX_DISCREPANCY_AUDIT.md`
- `src/phase31y_r_approx_audit.py` (reproducible audit script)
- `src/phase31y_r_debug.py` (debug script, confirmed tiny example failure)

---

*Phase 31Y-R Audit — ELVIS — SDI Substitutive*
*Commit message: "Phase 31Y-R: audit manifest runtime approximation discrepancy"*
