# Phase 31B — Substitutive Residual Economics Harness

**Date:** 2026-05-29
**Status:** Design + standalone harness plan. No llama.cpp modification. No benchmarks yet.
**Follows:** Phase 31A (architecture plan)
**Corrects:** Phase 31A's residual economics assumption

---

## Corrected Memory Math

**Critical correction from Matt:**

A dense INT8 residual does NOT beat Q4 residency.

For N weights:

| Representation | Bits/Weight | Verdict |
|----------------|-------------|---------|
| Q4 reference | N × 4 bits | baseline |
| Q2 base | N × 2 bits | |
| Q2 + dense INT8 residual | N × 2 + N × 8 = N × 10 bits | ❌ loses badly |
| Q2 + dense INT6 residual | N × 2 + N × 6 = N × 8 bits | ❌ still loses |
| Q2 + dense INT4 residual | N × 2 + N × 4 = N × 6 bits | ❌ still loses |
| Q2 + dense INT2 residual | N × 2 + N × 2 = N × 4 + metadata | ≈ or > Q4; metadata and decode buffers may erase savings |

**Threshold for viability:**
- Q2 + residual must beat Q4: residual must cost **< 2 bits/weight**
- Q3 + residual must beat Q4: residual must cost **< 1 bit/weight**

**The thesis is still alive** but only if residuals are extremely compressed — sparse, low-rank, or both. Dense residuals at any bit width above 2 are not viable.

---

## Goal

Build a standalone tensor harness that answers:

> How compressed/sparse/low-rank must R be for `W_low + R_sidecar < W_q4` while preserving acceptable tensor approximation?

This is:
- ✅ Tensor-level math + memory economics
- ✅ Not llama.cpp integration
- ✅ Not model quality
- ✅ Not speedup

---

## Target Tensor

**Start with: `attn_out` layer 0** — small (896×896), easy to validate, proof-of-mechanics target.

**State explicitly:** attn_out is the proof-of-mechanics target, not the best memory-win target. ffn_up/ffn_down have more economically meaningful shape ratios (896×4864) and will be tested after mechanics are validated.

---

## Required Residual Representations to Test

| Representation | Expected Memory | Expected Accuracy | Pass Criteria |
|----------------|---------------|-------------------|---------------|
| Dense INT8 residual | ❌ loses | high | baseline accuracy only |
| Dense INT6 residual | ❌ loses | high | baseline accuracy only |
| Dense INT4 residual | ❌ loses | high | baseline accuracy only |
| Dense INT2 residual | ≈ or > Q4 | medium-high | marginal; metadata/decode buffers may erase savings |
| Dense ternary (±1, 0) | ~2 bits/weight | medium | marginal memory |
| Top-k sparse (k% nonzero) | variable | depends on k | must show k vs accuracy tradeoff |
| Top-k sparse (magnitude threshold) | variable | depends on threshold | key test |
| Row/column sparse | depends on sparsity pattern | variable | low-overhead encoding |
| Block sparse (4×4 blocks) | variable | depends on pattern | hardware-friendly |
| Low-rank residual (rank r) | r × (rows + cols) × bits | depends on r | key test |
| Hybrid sparse + low-rank | variable | potentially good | combine above |

**Focus first on:** Top-k sparse and low-rank — these have the best theoretical compression-to-accuracy ratio.

---

## Memory Accounting Formula

```
W_q4_bytes = N × 0.5 bytes         (4 bits/weight)
W_low_bytes = N × 0.25 bytes      (Q2: 2 bits/weight)
W_q3_bytes = N × 0.375 bytes     (Q3: 3 bits/weight)

R_viable_if: W_low_bytes + R_compressed_bytes < W_q4_bytes
R_viable_if: R_compressed_bytes < N × 0.25 bytes   (Q2 case)
R_viable_if: R_compressed_bits_per_weight < 2 bits/weight  (Q2 case)
```

---

## Metrics Per Residual Representation

For each candidate, report:

**Memory:**
- `R_compressed_bytes`
- `effective_bits_per_weight`
- `W_low_bytes + R_compressed_bytes`
- `memory_delta_vs_Q4` (positive = saves memory, negative = loses)
- `memory_viable` (bool: W_low + R < W_q4)

**Accuracy:**
- `cosine(Y_ref, Y_sub)` — primary, dimension-agnostic
- `MAE(Y_ref, Y_sub)` — secondary
- `max_error` — stability indicator
- `top1_token_delta` if logits available (optional for Phase 31B)
- `rank_effective` for low-rank candidates

**Curves to produce:**
- `memory_cost_vs_cosine` — for all candidates on same plot
- `sparsity_vs_approximation` — for sparse candidates
- `rank_vs_approximation` — for low-rank candidates

---

## Test Inputs

1. **Random X:** Use seeded random activation vectors. Reproducible. Fast. Establishes mechanical proof.
2. **Recorded X (if available):** Use activations captured during prior inference runs. More realistic. Only if readily available from prior phases.

---

## Implementation Plan

### Files in `~/sdi-substitutive/`:

```
sdi-substitutive/
├── README.md
├── CLAIMS.md
├── FORBIDDEN_CLAIMS.md
├── .gitignore                    ← ignore data/, *.bin, *.gguf, models/
├── docs/
│   ├── PHASE31A_ARCHITECTURE_PLAN.md
│   └── PHASE31B_RESIDUAL_ECONOMICS_PLAN.md  ← this file
├── src/
│   ├── tensor_extract.py          ← extract W_q4, W_q2, W_q3 from GGUF
│   ├── residual_make.py           ← generate R = W_ref - W_low, quantize
│   ├── residual_probe.py          ← main harness: X @ W_low + X @ R for all R types
│   └── memory_math.py             ← memory accounting helpers
└── results/
    └── PHASE31B_*.json            ← per-candidate results
```

### Implementation steps for 31B harness:

1. **`tensor_extract.py`:** Use `gguf` Python package or `llama.cpp/gguf-py` to read W_ref (Q4) from the GGUF file. Export as numpy `.npy` or raw `.bin`. Small enough for attn_out layer 0 (~3.2 MB F32).

2. **`residual_make.py`:** Compute W_q2 from W_q4 using per-block quantization (same algorithm as llama.cpp Q4_K_M). Compute R_f32 = W_ref - W_q2 in F32. Generate all residual formats: dense INT8/INT6/INT4/ternary, top-k sparse, low-rank via truncated SVD.

3. **`residual_probe.py`:** Load W_q2 (numpy) + each R variant. Generate random X. Compute Y_ref = X @ W_ref_f32 (reference, F32), Y_sub = X @ W_q2 + X @ R (substitutive). Report cosine, MAE, max_error. Compute memory for each component.

4. **`memory_math.py`:** Helper functions for bytes-to-bits conversion, viability check, result aggregation.

---

## Classification

Use ONE:

| Classification | Meaning |
|----------------|---------|
| `PASS_MEMORY_VIABLE_RESIDUAL_FOUND` | At least one residual representation is memory-viable AND produces useful approximation |
| `PARTIAL_MATH_WORKS_MEMORY_FAILS` | Residual approximation works but no representation beats Q4 memory |
| `PARTIAL_MEMORY_WORKS_ACCURACY_POOR` | Some representations beat Q4 memory but approximation is too poor to be useful |
| `BLOCKED_TENSOR_EXTRACTION` | Cannot extract tensors from GGUF |
| `BLOCKED_NUMERICAL_ISSUE` | Numerical issues prevent reliable comparison |

---

## Expected Outcome

| Residual Type | Memory vs Q4 | Accuracy | Classification |
|---------------|-------------|----------|----------------|
| Dense INT8 | ❌ loses | ✅ high | math works, memory fails |
| Dense INT4 | ❌ loses | ✅ high | math works, memory fails |
| Dense ternary | ≈ Q4 | 🔸 medium | marginal |
| Top-10% sparse | 🔸 depends | 🔸 depends | **key test** |
| Top-1% sparse | ✅ saves | ❌ poor | possibly too sparse |
| Low-rank r=8 | 🔸 depends | 🔸 depends | **key test** |
| Low-rank r=2 | ✅ saves | ❌ poor | possibly too aggressive |
| Hybrid sparse+lowrank | 🔸 best case | 🔸 best case | if sparsity + rank combine well |

**If no representation passes memory viability at acceptable accuracy:** The substitutive thesis is blocked at the economics level and llama.cpp integration should not proceed.

**If top-k sparse or low-rank passes:** Move to Phase 31C (extend to ffn_up/ffn_down) and Phase 31D (llama.cpp loader substitution feasibility).

---

## Forbidden Claims

- ❌ No quality recovery
- ❌ No speedup
- ❌ No production readiness
- ❌ No full-model behavior
- ❌ No "SDI solved"
- ❌ No "Q2+sidecar ≈ Q4" unless tensor-level approximation supports it
- ❌ No llama.cpp integration claims until economics pass

---

## Decision Gate After Phase 31B

```
IF any_viable_representation AND cosine > 0.99:
    → Phase 31C: extend to ffn_up/ffn_down
    → Phase 31D: llama.cpp loader substitution feasibility
ELIF any_viable_representation AND cosine < 0.99:
    → Phase 31C: ffn_up only; re-evaluate after larger tensor shape
ELSE:
    → Substitutive thesis blocked at economics level
    → Stop and report before spending more time on integration
```

---

**Key principle:** Substitutive replacement only matters if the replacement is cheaper than the tensor being replaced. Phase 31B is the gate that determines whether that is physically possible at all.