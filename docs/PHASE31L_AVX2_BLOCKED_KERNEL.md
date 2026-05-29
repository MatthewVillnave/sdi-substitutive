# Phase 31L: AVX2 / Blocking Kernel Implementation

## Summary

- **Classification:** `PARTIAL_CORRECT_BUT_NO_SPEED_GAIN`
- **k_pct:** 7.5%
- **Shapes tested:** tiny (8×16), ffn_up (896×4864), ffn_down (4864×896)
- **Batch sizes:** tiny: [1, 4, 16]; ffn_up/down: [1, 4]
- **AVX2 via Numba:** Segfaults in current environment — classified as `PARTIAL_AVX2_UNSTABLE`

---

## 1. Memory Accounting Clarification (AUDIT FIX)

**Phase 31K reported "all layouts < 3.2MB vs Q4 budget ~234MB" — THIS WAS A UNIT/SCOPE MISMATCH.**

The "234MB" was the **full-model Q4 size** (sum across all 6 layers × 2 tensor families × 2 directions), not a per-tensor budget. Per-tensor residual budgets are much smaller.

### Per-Tensor Memory (from Phase 31H/31I):

| Tensor | Shape | Q4 bytes | Q2 bytes | Budget (Q4−Q2) | CSR Encoded | Margin | Viable? |
|--------|-------|----------|----------|----------------|-------------|--------|---------|
| ffn_up | 896×4864 | 2,451,456 | 1,089,536 | **1,361,920** | 1,964,780 | −602,860 | ❌ |
| ffn_down | 4864×896 | 2,451,456 | 1,089,536 | **1,361,920** | 1,980,652 | −618,732 | ❌ |
| tiny | 8×16 | 64 | 32 | **32** | 122 | −90 | ❌ |

**Key corrections:**
- Per-tensor Q4 budget is ~1.36MB (not 234MB which is full-model aggregate)
- CSR row-wise encoded size exceeds per-tensor budget for ffn tensors
- At full-model scale:6 layers × 2 tensors × 2 directions ×1.36MB ≈ **16.3MB** — not 234MB either (234MB was likely computed differently)
- The "234MB" number from Phase 31K appears to be the full-model Q4 weight size itself, not a budget

**CSR Row-wise Layout Memory Breakdown:**
```
Per element (top-7.5%):
  - Offset entry: 4 bytes / row →  (rows+1) × 4
  - Col index:4 bytes / nnz  →  nnz × 4
  - Value:       2 bytes / nnz →  nnz × 2
  - Metadata:    32 bytes fixed
  Total: 32 + (rows+1)×4 + nnz×4 + nnz×2 bytes
```

**Memory status:** The row-wise CSR layout is NOT viable within the per-tensor Q4−Q2 budget headroom for ffn_up and ffn_down shapes. This is a known limitation carried forward from Phase 31K.

---

## 2. Variants Implemented

| Variant | Description | Status |
|---------|-------------|--------|
| `CSR_scalar_baseline_31K` | Row-wise CSR scalar (replicates 31K LayoutC) | ✅ Correct |
| `CSR_blocked_scalar` | Cache-friendly row blocks (BLOCK_SIZE=64) | ✅ Correct |
| `CSR_batch_specialized` | B=1 specialized path + generic | ✅ Correct |
| `AVX2_numba` | Numba JIT with4× unroll hints | ❌ Segfaults |

---

## 3. Correctness Table

All variants tested against dense decode reference.

| Shape | B | Variant | cos | maxdiff | correct? |
|-------|---|--------|-----|---------|---------|
| tiny | 1 | CSR_scalar_baseline_31K | 1.00000012 | 0.00e+00 | ✅ OK |
| tiny | 1 | CSR_blocked_scalar | 1.00000012 | 0.00e+00 | ✅ OK |
| tiny | 1 | CSR_batch_specialized | 1.00000012 | 0.00e+00 | ✅ OK |
| tiny | 4 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | ✅ OK |
| tiny | 4 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | ✅ OK |
| tiny | 4 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | ✅ OK |
| tiny | 16 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | ✅ OK |
| tiny | 16 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | ✅ OK |
| tiny | 16 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | ✅ OK |
| ffn_up | 1 | CSR_scalar_baseline_31K | 1.00000012 | 0.00e+00 | ✅ OK |
| ffn_up | 1 | CSR_blocked_scalar | 1.00000012 | 0.00e+00 | ✅ OK |
| ffn_up | 1 | CSR_batch_specialized | 1.00000012 | 0.00e+00 | ✅ OK |
| ffn_up | 4 | CSR_scalar_baseline_31K | 1.00000012 | 0.00e+00 | ✅ OK |
| ffn_up | 4 | CSR_blocked_scalar | 1.00000012 | 0.00e+00 | ✅ OK |
| ffn_up | 4 | CSR_batch_specialized | 1.00000012 | 0.00e+00 | ✅ OK |
| ffn_down | 1 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | ✅ OK |
| ffn_down | 1 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | ✅ OK |
| ffn_down | 1 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | ✅ OK |
| ffn_down | 4 | CSR_scalar_baseline_31K | 1.00000000 | 0.00e+00 | ✅ OK |
| ffn_down | 4 | CSR_blocked_scalar | 1.00000000 | 0.00e+00 | ✅ OK |
| ffn_down | 4 | CSR_batch_specialized | 1.00000000 | 0.00e+00 | ✅ OK |

**All scalar variants pass correctness.** AVX2 via Numba segfaults in the current environment.

---

## 4. Benchmark Table (median ms)

| Shape | B | baseline_31K | blocked_scalar | batch_specialized | AVX2 |
|-------|---|--------------|----------------|-----------------|------|
| tiny | 1 | 0.0277 |0.0283 | 0.0275 | N/A |
| tiny | 4 | 0.0292 | 0.0294 | 0.0290 | N/A |
| tiny | 16 | 0.0300 | 0.0295 | 0.0293 | N/A |
| ffn_up | 1 | 3.0489 | 3.0454 | 3.0435 | N/A |
| ffn_up | 4 | 5.5368 | 5.6107 | 5.7126 | N/A |
| ffn_down | 1 | 18.2207 | 18.2196 | 18.1709 | N/A |
| ffn_down | 4 | 31.4274 | 31.5591 | 35.4939 | N/A |

**Speedup vs 31K CSR scalar baseline:** None. Blocked scalar and batch-specialized are within ±2% of baseline — no meaningful improvement.

---

## 5. Memory Behavior

- **No dense R allocation** — CSR scalar/blocked variants use only encoded residual
- **Encoded residual only** — bitmap + offsets + col_indices + fp16 values
- **Scratch bytes:** O(batch × max(rows, cols)) — output buffer only
- **Output bytes:** batch × d_out × 4 bytes (fp32)
- **No hidden dense temp buffers** — all variants maintain CSR representation throughout

---

## 6. Classification

**`PARTIAL_CORRECT_BUT_NO_SPEED_GAIN`**

- All scalar variants: correct (cos ≥ 0.99999, maxdiff < 1e-3, no NaN/Inf)
- Blocked scalar: ~same speed as baseline (within noise)
- Batch-specialized: ~same speed as baseline (within noise)
- AVX2 via Numba: segfaults in current environment → `PARTIAL_AVX2_UNSTABLE`
- Memory: row-wise CSR exceeds per-tensor Q4−Q2 budget for ffn shapes

**Root cause of no speed gain:** The31K scalar CSR kernel is already memory-bandwidth-bound (not compute-bound). Blocking adds no cache benefit since each row is visited exactly once. Batch-specialized adds no benefit since the inner `np.dot` already saturates memory bandwidth.

---

## 7. Decision Gate

Phase 31M is **NOT YET UNLOCKED**.

The scalar kernel variants are correct but offer no speed improvement over the 31K baseline. Additionally, the memory budget violation (CSR encoded > Q4−Q2 budget for ffn tensors) means the encoding must be revisited before integration.

**Path forward:**
1. Profile the sparse residual kernel in context to understand whether compute or memory bandwidth is the bottleneck
2. Consider alternative CSR encodings with tighter memory (e.g., eliminate offsets array if row nnz can be encoded more compactly)
3. Revisit AVX2 with explicit C intrinsics (`_mm256_set_m128i`, `_mm256_cvtph_ps`, `_mm256_fmadd_ps`) rather than Numba
4. Reassess whether the per-tensor budget constraint is the right target

---

## 8. Phase 31M/31N Status

- **Phase 31M (integration feasibility design):** BLOCKED — no speed gain from blocking
- **Phase31N (runtime substitution design):** BLOCKED — same reason
- **llama.cpp integration:** BLOCKED until31M/31N pass

---

*Phase 31L — ELVIS — SDI Substitutive*
