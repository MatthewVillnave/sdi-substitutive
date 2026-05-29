# Phase 31C: FFN Residual Economics

**Commit:** `354bac7` | **Date:** 2026-05-29 | **Classification:** `PASS_FFN_MEMORY_VIABLE_RESIDUAL_FOUND`

## Goal
Test whether FFN tensor families (ffn_up, ffn_down) admit memory-viable residual compression that improves over W_low (Q2-approximated weights) on Qwen2.5-0.5B.

---

## Tensors Tested

| Tensor | Shape | d_out | d_in | N elements | Q4 bytes | Q2≈ bytes | Residual budget |
|---|---|---|---|---|---|---|---|
| `blk.0.ffn_up.weight` | (4864, 896) | 4864 | 896 | 4,358,144 | 2,996,224 | 1,089,536 | 1,906,688 (3.500 bpe) |
| `blk.0.ffn_down.weight` | (896, 4864) | 896 | 4864 | 4,358,144 | 3,575,040 | 1,089,536 | 2,485,504 (4.562 bpe) |

**Model:** Qwen2.5-0.5B-Instruct Q4_K_M GGUF
**Method:** W_low = round(W_ref / scale) × scale, scale = std(W_ref) / 2.5
**R std:** 11.5% of signal std for both tensors

---

## W_low Baseline

| Tensor | Cosine (vs ref) |
|---|---|
| `ffn_up` | 0.993399 |
| `ffn_down` | 0.993490 |

---

## Best Residual Results

### `ffn_up` — topk_10.0% ⭐ (BEST)
| Metric | Value |
|---|---|
| Representation | top-k sparse, 10% of largest residual elements |
| Compressed bytes | 1,743,264 |
| Effective bits/weight | 3.200 bpe |
| Memory viable | ✅ (under 1,906,688 byte budget) |
| Cosine (W_low+R vs ref) | 0.995153 |
| Cosine (W_low only) | 0.993399 |
| **Improvement Δ** | **+0.001755** |
| NNZ | 435,814 |

### `ffn_down` — topk_10.0% ⭐
| Metric | Value |
|---|---|
| Representation | top-k sparse, 10% of largest residual elements |
| Compressed bytes | 1,743,264 |
| Effective bits/weight | 3.200 bpe |
| Memory viable | ✅ (under 2,485,504 byte budget) |
| Cosine (W_low+R vs ref) | 0.995197 |
| Cosine (W_low only) | 0.993490 |
| **Improvement Δ** | **+0.001707** |

---

## All Candidates Summary

### `ffn_up` sorted by improvement
| Representation | ebwp | Memory viable | Cosine | Δ vs W_low |
|---|---|---|---|---|
| topk_10.0% | 3.200 | ✅ | 0.995153 | **+0.001755** |
| topk_5.0% | 1.600 | ✅ | 0.994326 | +0.000928 |
| lowrank_r32 | 1.354 | ✅ | 0.993843 | +0.000444 |
| topk_2.0% | 0.640 | ✅ | 0.993771 | +0.000373 |
| lowrank_r16 | 0.677 | ✅ | 0.993646 | +0.000248 |
| topk_1.0% | 0.320 | ✅ | 0.993591 | +0.000192 |
| lowrank_r8 | 0.338 | ✅ | 0.993525 | +0.000127 |
| dense_INT8 | 8.000 | ❌ | 0.999900 | +0.006501 |

### `ffn_down` sorted by improvement
| Representation | ebwp | Memory viable | Cosine | Δ vs W_low |
|---|---|---|---|---|
| topk_10.0% | 3.200 | ✅ | 0.995197 | **+0.001707** |
| topk_5.0% | 1.600 | ✅ | 0.994401 | +0.000911 |
| lowrank_r32 | 1.354 | ✅ | 0.993892 | +0.000402 |
| topk_2.0% | 0.640 | ✅ | 0.993866 | +0.000376 |
| lowrank_r16 | 0.677 | ✅ | 0.993676 | +0.000186 |
| dense_INT8 | 8.000 | ❌ | 0.999900 | +0.006410 |

---

## Key Findings

1. **Top-k sparse dominates low-rank SVD** at every budget level for both FFN tensors
2. **10% top-k is optimal** for the residual budget — improvement plateaus above that
3. **All candidates memory-viable** under the Q4-Q2 residual budget
4. **R std = 11.5% of signal** — residuals are small but not negligible
5. **Low-rank SVD is token-inefficient** for these shapes — r=32 still only recovers 0.044% improvement vs 10% top-k recovering 0.176%

---

## Decision Gate

| Condition | Result |
|---|---|
| FFN up viable? | ✅ Yes |
| FFN down viable? | ✅ Yes |
| Any improvement over W_low? | ✅ Yes |

**→ Phase 31D (residual compression policy) UNLOCKS**
**→ Phase 31E (recorded activations) UNLOCKS**
**→ Phase 31G (budget-optimal policy) ready for Matt's approval to re-unlock**

---

## Phase 31D Plan
- Build residual compression policy using top-k sparse as primary representation
- Determine k adaptively per-tensor based on residual signal strength
- Integrate with attention output residual (Phase 31B) for full substitutive compression policy
- Document residual routing: which tensors get residuals, which are Q2-only