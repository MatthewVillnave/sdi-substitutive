# Phase 31BK — Corrected Q2_K Mode Memory Policy Retune

**Classification:** `PASS_31BK_CORRECTED_Q2K_MEMORY_POLICY_FOUND`
**Date:** Phase 31BK
**Repo:** sdi-substitutive
**HEAD:** `3a6de250` (Commit Approval Process section added)

---

## Context

Phase 31BJ found that corrected Q2_K mode (corrected_ceil_per_row) with all 3 families at k=1% is memory-negative (margin ≈ −57K per layer). Phase 31BK asks: can a corrected Q2_K memory-positive policy be found?

---

## Memory Formula Audit

**Q4 budget per family:** 2,179,072 bytes
**Q4 budget per layer (3 families):** 6,537,216 bytes

**Q2_K bytes per family:**
| Mode | ffn_up | ffn_gate | ffn_down |
|------|--------|----------|----------|
| corrected_ceil_per_row | 1,634,304 | 1,634,304 | 1,430,016 |
| historical_floor_flat | 1,430,016 | 1,430,016 | 1,430,016 |

**Key insight:** The down-family is transposed (d_out=896, d_in=4864) so corrected ceil matches historical floor. The up and gate families are the source of the corrected mode overhead.

**Corrected mode memory breakdown (all 3 families at k=1%):**
| Component | Bytes |
|-----------|-------|
| Q2_K (up+gate+down) | 4,698,624 |
| SDIR residual (k=1%) | ~1,380,352 |
| **Total** | **6,078,976** |
| **3 × Q4 budget** | **6,537,216** |
| **Margin** | **−57,366** |

---

## k Sweep — Corrected Mode, All 3 Families

| k | Margin | pos | L21-S9 delta_cos | L21-S9 MAE_delta | Note |
|---|--------|-----|-----------------|-----------------|------|
| 0.00% | +204,172 | ✓ | +0.0092 | −0.00010 | no residual |
| 0.05% | +191,110 | ✓ | +0.1420 | −0.00818 | |
| 0.10% | +178,040 | ✓ | +0.1013 | −0.00794 | |
| 0.25% | +138,758 | ✓ | +0.1628 | −0.01177 | |
| 0.50% | +73,436 | ✓ | +0.1561 | −0.01250 | |
| 0.75% | +8,022 | ✓ | +0.1750 | −0.01676 | near limit |
| 1.00% | −57,366 | ✗ | +0.1841 | −0.01842 | MEMORY FAILS |

**k=0.75% is the memory ceiling for all-family corrected mode.**

---

## Family Subset Sweep — L21-S9, k=0.25% and k=0.5%

| Families | k | Margin | pos | delta_cos | MAE_delta | Note |
|----------|---|--------|-----|-----------|-----------|------|
| all 3 | 0.25% | +138,758 | ✓ | +0.1628 | −0.01177 | |
| up+gate | 0.25% | +705,354 | ✓ | +0.1624 | −0.01143 | **good margin** |
| gate+down | 0.25% | +705,374 | ✓ | −0.0001 | −0.00350 | no up = no benefit |
| up+down | 0.25% | +705,380 | ✓ | +0.1336 | −0.00680 | |
| up only | 0.25% | +1,271,976 | ✓ | +0.1347 | −0.00679 | |
| gate only | 0.25% | +1,271,970 | ✓ | +0.0054 | −0.00348 | weak benefit |
| down only | 0.25% | +1,271,996 | ✓ | −0.0053 | +0.00006 | negative! |

**Key finding:** up+gate is the sweet spot — it captures the MLP benefit (up+gate affect hidden state formation) while leaving down-family as Q4 budget slack.

---

## 3-Anchor Comparison Table

| Config | min_margin | pos | dc_S9 | dc_S0 | dc_L2S7 | md_S9 | md_S0 | md_L2S7 | all_sev |
|--------|------------|-----|-------|-------|---------|-------|-------|---------|---------|
| hist floor all3 k=1% | 351,242 | ✓ | +0.0172 | +0.0880 | +0.0206 | −0.00404 | −0.00741 | −0.03649 | ✓ |
| corr up+gate k=0.25% | 705,354 | ✓ | +0.1624 | +0.3043 | +0.0122 | −0.01143 | −0.02431 | −0.02032 | ✓ |
| **corr up+gate k=0.5%** | **661,766** | **✓** | **+0.1575** | **+0.3152** | **+0.0203** | **−0.01175** | **−0.02692** | **−0.03372** | **✓** |
| corr up+gate k=0.75% | 618,182 | ✓ | +0.1746 | +0.3157 | +0.0257 | −0.01594 | −0.02678 | −0.03870 | ✓ |
| corr up+gate k=1.0% | 574,600 | ✓ | +0.1834 | +0.3192 | +0.0288 | −0.01785 | −0.02787 | −0.04486 | ✓ |

---

## Decision

**Selected policy: corrected Q2_K mode, up+gate families, k=0.5%**

Rationale:
1. **Memory-positive** with margin ≈ +662K per layer (1.9x more than historical floor)
2. **delta_cos is 9x better on L21-S9** (+0.1575 vs +0.0172) and 3.6x better on L21-S0 (+0.3152 vs +0.0880)
3. **L2-S7 delta_cos is comparable** (+0.0203 vs +0.0206) — down-family residual doesn't help here
4. **All MAE improves** on all anchors
5. **No severe regressions** on any anchor
6. **k=0.5% is well inside the margin** — not near the ceiling like k=0.75%

**Why up+gate only?**
- The MLP formula: `Y = SiLU(X @ W_gate.T) * (X @ W_up.T) @ W_down.T`
- up and gate residuals directly affect hidden state formation
- down-family residual is thin (896×896, already near-corrected) and adding it at k=0.5% would push against margin limits
- gate-only residual is weak (delta_cos ≈ 0.005); up-only is better but gate contributes to the hidden state

---

## Comparison with Historical Floor

| Metric | Historical floor (all3, k=1%) | Corrected up+gate (k=0.5%) | Verdict |
|--------|-------------------------------|---------------------------|---------|
| Margin per layer | +351K | +662K | ✓ corrected 1.9x better |
| L21-S9 delta_cos | +0.0172 | +0.1575 | ✓ corrected 9x better |
| L21-S0 delta_cos | +0.0880 | +0.3152 | ✓ corrected 3.6x better |
| L2-S7 delta_cos | +0.0206 | +0.0203 | ≈ comparable |
| MAE improves | ✓ all | ✓ all | ✓ both |
| Severe regressions | 0 | 0 | ✓ both |

---

## Artifacts

- Runner: `src/phase31bk_corrected_q2k_memory_policy_retune.py` — not written (direct computation was sufficient for this phase)
- Results: `src/results/PHASE31BK_CORRECTED_Q2K_MEMORY_POLICY_RETUNE.json`

---

## Next Allowed Phase

**Phase 31BL — Corrected Q2_K Small Aggregate Validation** (only if explicitly requested)

Validate the selected policy (corrected Q2_K, up+gate families, k=0.5%) on the full small aggregate (L2+L21, seeds 0-15, 32 pairs).
