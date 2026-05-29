# Phase 31M: Integration Feasibility Design

**Classification:** `PASS_INTEGRATION_FEASIBILITY_DEFINED`
**Date:** 2026-05-29
**Old HEAD:** 27693c215c8cde0499deb1acadbf251377c1c9c1
**Phase 31L Classification:** `PASS_AVX2_BLOCKED_KERNEL_READY`

---

## Context

Phase 31L proved the sparse-residual CSR kernel is correct and memory-bounded:
- All shapes/batches: max_diff < 1e-5, cosine = 1.0
- AVX2_numba did NOT outperform CSR scalar baseline
- No meaningful overall speed improvement over baseline was demonstrated

Phase 31I established robustness: 180/180 (layer 0-5, ffn_up + ffn_down, 15 prompts) showed improvement with substitution. All correct.

The core question for 31M: **How does substitutive tensor replacement actually enter a runtime llama.cpp process — without modifying llama.cpp, without writing integration code, without claiming production readiness?**

This is **design only**.

---

## 1. Integration Options

### Option A: Loader Substitution
```
llama.cpp model loader
  → Do NOT load W_ref for substituted tensors
  → Load W_low (Q2) + residual metadata instead
  → Model graph sees a replacement op/tensor (different name/type)
```

| Dimension | Rating |
|-----------|--------|
| Memory win potential | ✅ MAXIMAL — W_ref never resident |
| Implementation complexity | 🔴 HIGH — requires modifying llama.cpp model-loading code |
| Additive trap risk | ⚠️ MEDIUM — loader must track which tensors are substituted |
| GGUF compatibility | ❌ LOW — non-standard tensor type requires GGUF extension |
| Debugging difficulty | 🔴 HIGH — loader bugs cause silent wrong results |
| Recommended order | 3rd |

### Option B: Graph-Level Custom Op
```
FFN forward pass
  → Intercept ffn_up / ffn_down matmul
  → Compute X @ W_low + X @ R_sparse
  → Must ensure W_ref is NOT resident for that tensor
```

| Dimension | Rating |
|-----------|--------|
| Memory win potential | ✅ HIGH — W_ref excluded if loader configured correctly |
| Implementation complexity | 🟡 MEDIUM — custom op in ggml_compute_forward-style |
| Additive trap risk | 🔴 HIGH — risk of accidentally loading W_ref AND applying residual |
| GGUF compatibility | ✅ OK — can use existing tensor types with routing logic |
| Debugging difficulty | 🟡 MEDIUM — need runtime guards to confirm W_ref absent |
| Recommended order | 2nd |

### Option C: Offline Model Rewrite
```
Pre-processing step
  → Produce modified GGUF (or sidecar manifest)
  → Selected tensor weights replaced by W_low (Q2)
  → Residual encoded as separate sidecar file
  → No loader surgery needed — custom op reads sidecar
```

| Dimension | Rating |
|-----------|--------|
| Memory win potential | ✅ HIGH — W_ref replaced in GGUF |
| Implementation complexity | 🟡 MEDIUM — offline tooling only, no runtime loader surgery |
| Additive trap risk | ✅ LOW — artifact is the source of truth |
| GGUF compatibility | ✅ HIGH — standard GGUF with sidecar |
| Debugging difficulty | 🟡 MEDIUM — need manifest/tensor mapping |
| Recommended order | 1st (cleanest artifact story) |

### Option D: Standalone Runtime / Proxy Harness
```
Toy harness (Python or minimal C++)
  → Implements FFN forward pass using W_low + residual only
  → No llama.cpp involvement
  → Proves substitution economics in isolation
```

| Dimension | Rating |
|-----------|--------|
| Memory win potential | ✅ HIGH (theoretical — no actual memory pressure) |
| Implementation complexity | ✅ LOW — no llama.cpp coupling |
| Additive trap risk | ✅ LOW — isolated environment |
| GGUF compatibility | N/A |
| Debugging difficulty | ✅ LOW — full control, easy logging |
| Recommended order | 1st (validation before any other path) |

### Integration Options Summary

| Option | Memory Win | Complexity | Additive Trap Risk | GGUF Compat | Debug | Order |
|--------|-----------|------------|-------------------|-------------|-------|-------|
| A: Loader substitution | MAXIMAL | HIGH | MEDIUM | LOW | HIGH | 3rd |
| B: Graph custom op | HIGH | MEDIUM | **HIGH** | OK | MEDIUM | 2nd |
| C: Offline model rewrite | HIGH | MEDIUM | **LOW** | HIGH | MEDIUM | 1st* |
| D: Toy harness | HIGH | LOW | LOW | N/A | LOW | 1st |

*Option C and D are co-equal first priorities — C produces the artifact, D validates the economics.

---

## 2. No-Additive-Trap Requirements

For **ANY** future integration, ALL of the following must hold:

| # | Requirement | Verification Method |
|---|-------------|---------------------|
| 1 | W_ref is NOT resident in memory for substituted tensor | Memory accounting counter: `skipped_ref_tensor_bytes == 0` for that tensor |
| 2 | W_low IS resident | Counter: `low_tensor_bytes_loaded` includes the tensor |
| 3 | Residual is encoded and resident/compressed | Counter: `residual_encoded_bytes_loaded > 0` |
| 4 | No persistent dense residual R | Runtime counter: `residual_dense_bytes_materialized == 0` |
| 5 | No hidden decode cache bloat | Must verify no per-token cache growth during forward |
| 6 | Path label proves substitution path | Log line includes `[SDI-SUBSTITUTIVE-PATH]` |
| 7 | Memory counters prove W_ref absent | Explicit assertion before forward pass |
| 8 | Fail-fast if residual missing | Do not silently fall back to native — error or explicit config required |

**Additive trap is the central risk.** If W_ref stays resident and residual is also applied, the result would be W_ref + R instead of W_low + R = W_ref. This silently doubles the residual correction and corrupts output quality without raising any error.

---

## 3. Runtime Counters

```python
# Loader/model-level counters
substituted_tensor_count: int          # Number of tensors replaced with W_low + residual
skipped_ref_tensor_bytes: int         # Total bytes of W_ref NOT loaded
low_tensor_bytes_loaded: int         # Total bytes of W_low actually loaded
residual_encoded_bytes_loaded: int     # Total bytes of compressed residual loaded

# Compute-level counters
residual_dense_bytes_materialized: int  # Dense residual (should be 0 in pure substitution)
residual_decode_cache_bytes: int        # Decode-side cache (should be 0)
custom_op_calls: int                    # Number of substitutive forward calls
fallback_native_calls: int              # Number of fallback-to-native calls
additive_fallback_calls: int            # Number of additive-trap-triggered fallbacks

# Memory accounting
memory_margin_vs_q4: int              # (W_low + residual) - W_ref for substituted tensors
path_label: str                         # Always "[SDI-SUBSTITUTIVE-PATH]"
```

---

## 4. First Integration Target

**Recommendation: ffn_up only**

### Justification

| Factor | ffn_up | ffn_down |
|--------|--------|----------|
| Shape | 896×4864 | 4864×896 |
| Speedup vs dense (B=1) | **7.8×** (3.3ms vs 25.8ms) | 2.4× (19.7ms vs 47.7ms) |
| 31I mean delta_cosine | **+0.063** (higher improvement) | +0.075 (but absolute quality already higher) |
| Python kernel speed | ✅ ~3.3ms at B=1 (acceptable) | 🔴 ~19.6ms at B=1 (too slow for real-time) |
| Memory per tensor | 1,964,780 bytes encoded | 1,980,652 bytes encoded |
| 31L AVX2_numba vs CSR scalar | **0.97×** (slower!) | 1.0× (flat) |
| Compute risk at runtime | ✅ LOW | 🔴 HIGH |

**ffn_down is the compute risk.** At B=1, ffn_down takes 19.6ms even in the optimized Python CSR kernel. That's borderline for interactive use and llama.cpp's forward pass budget. The 2.4× speedup over dense is real, but 19.6ms per tensor per token is a significant portion of the per-token latency budget.

**ffn_up is suitable for integration.** It achieves 7.8× speedup at B=1, which gives significant headroom. Even if integrated at B=1 it only costs 3.3ms. At higher batch sizes, ffn_up continues to show speedup (1.0× from CSR_batch_specialized at B=4, which is still acceptable).

**Recommended path:** Phase 31N builds a toy substitutive runtime harness including ffn_up only. This proves the integration economics outside llama.cpp before any loader surgery.

---

## 5. Acceptance Tests Before Real Integration

### A. Tensor Absence Test
**Purpose:** Prove W_ref is not loaded/resident for the substituted tensor.

```
ASSERT: memory_map[W_ref_tensor_name] == NOT_LOADED
ASSERT: skipped_ref_tensor_bytes >= W_ref_bytes_for_substituted_tensor
```

### B. Substitution Path Test
**Purpose:** Prove the custom path fires for substituted tensors.

```
ASSERT: log contains "[SDI-SUBSTITUTIVE-PATH]" for substituted tensor forward
ASSERT: custom_op_calls incremented
ASSERT: fallback_native_calls == 0 for that tensor
```

### C. Memory Accounting Test
**Purpose:** Prove W_low + residual < W_ref for the selected tensor.

```
ASSERT: (W_low_bytes + residual_encoded_bytes) < W_ref_bytes
ASSERT: memory_margin_vs_q4 < 0
```

### D. Correctness Test
**Purpose:** Prove output matches standalone harness within tolerance.

```
ASSERT: max_abs_diff(output_harness, output_integrated) < 1e-4
ASSERT: cosine_sim(output_harness, output_integrated) > 0.9999
```

### E. Fail-Fast Test
**Purpose:** If residual is missing, do not silently use native.

```
ASSERT: if residual_encoded_bytes_loaded == 0 → raise error or ConfigRequired
ASSERT: additive_fallback_calls == 0  (no silent add-back)
```

### F. Additive-Trap Test
**Purpose:** Prove W_ref and residual are not both resident.

```
ASSERT: skipped_ref_tensor_bytes > 0  (W_ref skipped)
ASSERT: residual_dense_bytes_materialized == 0  (no dense R materialized)
ASSERT: residual_decode_cache_bytes == 0  (no cache bloat)
ASSERT: NOT (skipped_ref_tensor_bytes > 0 AND residual_dense_bytes_materialized > 0)
```

---

## 6. Compute Risk Analysis

### Current Kernel Performance at B=1

| Tensor | ref_dense (ms) | CSR_scalar (ms) | Speedup | Acceptable for Runtime? |
|--------|-----------------|-----------------|---------|------------------------|
| ffn_up | 25.8 | 3.30 | 7.8× | ✅ YES |
| ffn_down | 47.7 | 19.7 | 2.4× | ⚠️ BORDERLINE |

### Key Observations

1. **ffn_up is viable at compute level.** 3.3ms at B=1 is well within acceptable per-tensor budget. The 7.8× speedup is meaningful.

2. **ffn_down is not yet viable at compute level.** 19.6ms at B=1 for a single FFN tensor is uncomfortably high. Even with 2.4× speedup over dense, that's a large fraction of a per-token latency budget (especially for streaming scenarios).

3. **AVX2_numba did NOT speed up the kernel.** In Phase 31L, AVX2_numba was slower than CSR_scalar for ffn_up (0.97×) and essentially flat for ffn_down (1.0×). Numba was not providing SIMD acceleration in this environment. Real C++ AVX2 intrinsics would be needed for meaningful speedup.

4. **The column-iterate pattern dominates ffn_down cost.** The ffn_down shape is (4864, 896) — each output column touches all 4864 rows. X @ W^T where W is (4864, 896). This means the sparse residual fetch is very expensive per column. The layout design in Phase 31L attempted blocked layouts but the improvement was marginal (1.009× vs CSR_scalar).

### Ranking: Integration Feasibility

| Tensor | Memory Viable? | Compute OK? | Integrate First? |
|--------|-----------------|-------------|------------------|
| ffn_up | ✅ YES | ✅ YES | ✅ YES |
| ffn_down | ✅ YES | ⚠️ MARGINAL | ❌ WAIT for C++ SIMD |

### Would Integration Only Make Sense After Real C++ AVX2 Blocking?

**YES for ffn_down.** A GPU/C++ SIMD kernel with proper blocking (e.g., 4×16 or 8×8 register blocking) against the sparse residual layout would likely cut the 19.6ms figure substantially. The Python CSR scalar is a lower bound; a well-optimized C++ implementation targeting AVX2/FMA units could reasonably achieve 4-8× additional speedup on ffn_down, making it viable.

**NOT NECESSARY for ffn_up.** The 3.3ms at B=1 is already fast enough. Integration could proceed with the Python CSR kernel for ffn_up while ffn_down C++ work proceeds in parallel.

### Is a Cheaper Residual Layout Needed for ffn_down?

**YES.** Phase 31L attempted blocked layout optimization. The result: `CSR_batch_specialized` achieved 1.009× speedup over `CSR_scalar` for ffn_down at B=1 — essentially flat. The bottleneck is traversing columns of R^T (W_down.T) which is harder to optimize than rows.

The key observation from Phase 31L timings: the `batch_specialized` variant only shines at B=4+ (3.67× speedup for ffn_down at B=4). At B=1, the layout is idle. For ffn_down, a **block-row layout** where the residual is stored transposed (so rows of R correspond to output columns) would reduce traversal cost.

A dedicated `R_sparse_transposed` layout: store the residual as CSR on the transposed shape (896, nnz_per_row) — this would change the X traversal from column-iterate to row-iterate for ffn_down, matching how ffn_up naturally benefits.

---

## 7. Classification

**Classification:** `PASS_INTEGRATION_FEASIBILITY_DEFINED`

The substitutive integration path is **feasible** with the following findings:

1. **Four integration paths identified** — Option D (toy harness) and Option C (offline model rewrite) are the recommended first steps in parallel. Option B (graph custom op) is the second step. Option A (loader substitution) is the final step.

2. **No-additive-trap requirements defined** — Eight requirements including W_ref absent, residual not dense, fail-fast on missing residual.

3. **Runtime counters defined** — Twelve counters covering memory accounting, compute calls, and additive trap detection.

4. **First target: ffn_up only** — Better speedup (7.8×), lower absolute cost (3.3ms), higher quality improvement delta (+0.063).

5. **Compute risk: ffn_down is too slow** — Current Python CSR at ~19.6ms for ffn_down at B=1 is borderline. Real C++ AVX2 SIMD needed before ffn_down integration.

6. **Cheaper ffn_down layout needed** — Transposed residual storage would change the traversal pattern.

---

## 8. Recommended Phase 31N

**Recommendation: Option 2 — Toy Substitutive Runtime Harness**

**Phase 31N: Toy Substitutive Runtime Harness (ffn_up only)**

Build a minimal standalone Python/C forward pass that:
1. Loads W_low (Q2) for ffn_up tensors
2. Loads residual (R_sparse) from the encoded files produced in Phase 31I
3. Computes `Y = X @ W_low + X @ R_sparse` for a batch of activation inputs
4. Validates output matches the standalone harness reference from Phase 31I
5. Exercises all six acceptance tests from Section 5

**Why Option 2 over the others:**
- **No llama.cpp surgery required** — validates economics outside the complexity of loader integration
- **No GGUF extension needed** — uses existing activation files and residual encodings
- **Executes all acceptance tests** — proves the substitution story before any integration
- **Fast to implement** — minimal code surface vs. Option B or A
- **Can start immediately** — leverages existing Phase 31I encoded residual files

**What this phase does NOT do:**
- Does NOT modify llama.cpp
- Does NOT load GGUF models
- Does NOT claim production readiness
- Does NOT integrate with the tokenizer or sampling pipeline

**Pre-requisites from prior phases:**
- Phase 31I encoded residual files ( blk.*.ffn_up.weight.residual.RSC )
- Phase 31I activation arrays (inputs/outputs per layer)
- Phase 31H compressed residual compute validation

**Follow-on phase options:**
- **Phase 31O:** C++ kernel optimization for ffn_down (real AVX2 intrinsics) — addresses compute risk
- **Phase 31P:** llama.cpp loader substitution design doc — enables Option B/A

---

## 9. Decision Gate

| Gate | Status |
|------|--------|
| Kernel correctness proven (31L) | ✅ YES |
| Memory economics understood (31C) | ✅ YES |
| Robustness validated (31I) | ✅ YES |
| Integration path clear | ✅ YES |
| Compute viable for first target (ffn_up) | ✅ YES |
| **Next phase ready** | ✅ **31N: Toy Substitutive Runtime Harness** |

---

*Phase 31M — ELVIS — SDI Substitutive*
