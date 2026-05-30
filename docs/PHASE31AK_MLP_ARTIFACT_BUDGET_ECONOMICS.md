# Phase 31AK — Full MLP Artifact Budget / Economics Fix

## Classification
**`PARTIAL_MLP_BUDGET_FAILS_CURRENT_ENCODING`**

No memory-positive policy exists under current encoding. Runtime policy changes (which families receive residuals, which k values are used) cannot close the budget gap. Artifact encoding redesign is required.

---

## Core Finding

The combined MLP substitutive artifact budget is structurally infeasible at current encoding:

| Component | Bytes | MB |
|-----------|------:|----:|
| W_low packed (Q4_2) | 44,126,208 | 42.08 |
| Residual (k=9/12%) | 25,511,492 | 24.33 |
| **Total substitutive** | **69,637,700** | **66.41** |
| **Q4 budget** | **39,223,296** | **37.41** |
| **Aggregate margin** | **−30,414,836** | **−29.01** |

**W_low packed alone** (42.08 MB) already **exceeds** the Q4 budget (37.41 MB) by **+4.68 MB**.

Adding residuals adds another 24.33 MB of overrun.

**No subset, family combination, or k-value selection can make this budget-positive under current encoding.**

---

## Scope
- **Layers**: 0–5
- **Families**: ffn_up (k=9%), ffn_gate (k=12%), ffn_down (k=9%)
- **Reference**: 31AJ baseline (PARTIAL_MLP_APPROX_PASS_MEMORY_FAIL)

---

## 31AJ Baseline Reproduction

| Metric | Value |
|--------|-------|
| Full MLP avg cosine | 0.998129 |
| Full MLP avg MAE | 0.023860 |
| Low-only avg cosine | 0.996028 |
| Low-only avg MAE | 0.041963 |
| **Delta cosine** | **+0.002101** |
| **Delta MAE** | **+0.018103** |

Residuals improve approximation on all 6 layers. The approximation math works correctly.

---

## Family Memory Breakdown

### Per-Family (6 layers each)

| Family | W_low (packed+scale) | Residual | Total Sub | Q4 Budget | Margin |
|--------|--------------------:|----------:|----------:|----------:|-------:|
| ffn_up | 14.03 MB | 7.61 MB | 21.64 MB | 12.47 MB | **−9.17 MB** |
| ffn_gate | 14.03 MB | 9.11 MB | 23.14 MB | 12.47 MB | **−10.67 MB** |
| ffn_down | 14.03 MB | 7.61 MB | 21.64 MB | 12.47 MB | **−9.17 MB** |

**Key fact**: Each family W_low alone (14.03 MB) exceeds its per-family Q4 budget (12.47 MB) by **1.56 MB**. This is because `W_low_packed + W_low_scale > W_ref_Q4_budget_bytes`. The scale bytes are the culprit.

---

## Subset Memory Analysis

All 7 family subsets fail:

| Subset | Total Sub | Q4 Budget | Margin | Memory Positive |
|--------|----------:|----------:|-------:|:----------------:|
| up only | 21.64 MB | 12.47 MB | −9.17 MB | ✗ |
| gate only | 23.14 MB | 12.47 MB | −10.67 MB | ✗ |
| down only | 21.64 MB | 12.47 MB | −9.17 MB | ✗ |
| up + gate | 44.78 MB | 24.94 MB | −19.84 MB | ✗ |
| up + down | 43.28 MB | 24.94 MB | −18.34 MB | ✗ |
| gate + down | 44.77 MB | 24.94 MB | −19.83 MB | ✗ |
| **up + gate + down** | **66.41 MB** | **37.41 MB** | **−29.01 MB** | **✗** |

---

## Residual ON/OFF Economics

Turning off a family's residual saves only that residual's bytes — the W_low packed + scale bytes remain.

| Family | Residual Size | Savings if OFF | Margin if OFF |
|--------|-------------:|---------------|-------------:|
| ffn_up | 7.61 MB | 7.61 MB | −1.56 MB |
| ffn_gate | 9.11 MB | 9.11 MB | −1.56 MB |
| ffn_down | 7.61 MB | 7.61 MB | −1.56 MB |

**Even with ALL residuals off**, each family's margin is still **−1.56 MB** because the W_low packed + scale encoding already exceeds the Q4 budget.

---

## Culprit Analysis

**Primary culprit: W_low scale bytes are not accounted for in Q4 budget.**

Each family-layer tensor has:
- `W_low_packed`: 2,179,072 bytes (Q4_2 representation of 4864×896)
- `W_low_scale`: 272,384 bytes (float16 scales)
- `Q4_budget`: 2,179,072 bytes (only the packed data, NOT the scales)

The Q4 budget represents what Q4_K_M would need to store one weight tensor. But the current substitutive artifact stores the packed weight **plus** the scale array — 2× the accounted budget.

**Secondary: residual bytes add ~24 MB on top.**

---

## What This Result Does NOT Claim

- NOT that the substitutive approach is invalid
- NOT that approximation doesn't improve
- NOT that any policy tweak can fix this
- NOT memory-positive for any MLP composition
- NOT quality/speed/integration/production claims

---

## Recommended Next Phase

**Phase 31AL — Artifact encoding redesign, only if explicitly requested.**

Required directions:
1. **Scale compression**: store scales in the same Q4_K_M file format, not as separate float16 arrays
2. **Compact W_low format**: new `.sdiw` variant that stores both packed weights and scales in one coherent format
3. **Residual encoding**: improve residual compression ratio to reduce residual bytes
4. **Budget accounting**: Q4_budget must reflect actual total artifact bytes (packed + scale), not just packed weight bytes

No further runtime policy sweep will change this result.

---

## Script
`src/phase31ak_mlp_budget_economics.py`
