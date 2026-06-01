# Phase 31BM — Corrected Q2_K Broader Aggregate Validation / Minor Failure Tracking

**Classification:** `PASS_31BM_CORRECTED_Q2K_BROADER_AGGREGATE_VALIDATED`
**Date:** Phase 31BM
**Repo:** sdi-substitutive
**HEAD:** `094e8e69` (Phase 31BL)

---

## Context

Phase 31BL found 2 minor failures on the small aggregate (L2+L21, 32 pairs). Phase 31BM broadens validation to all 24 layers to determine whether those failures are isolated or part of a pattern.

**Selected policy (unchanged from 31BK/31BL):**
- Q2_K mode: corrected_ceil_per_row
- Residual families: ffn_up + ffn_gate
- k: 0.5%
- alpha: 1.0
- ffn_down residual: none

---

## Route A: Full 24-Layer × 16-Seed Aggregate (384 pairs)

### Aggregate Table

| Metric | Value |
|--------|-------|
| total_pairs | 384 |
| n_cosine_improved | **383 / 384 (99.74%)** |
| n_MAE_improved | **383 / 384 (99.74%)** |
| n_memory_positive | **384 / 384 (100%)** |
| n_severe_regressions | **0** |
| n_cosine_failures | 1 |
| n_MAE_failures | 1 |
| worst pair | L21 S10 (dc=−0.0294) |
| best pair | L21 S0 (dc=+0.3152) |
| mean_delta_cos | **+0.0383** |
| median_delta_cos | **+0.0351** |
| mean_MAE_improvement | +0.0200 |
| min_margin | 661,766 |
| **aggregate_margin** | **254,130,400 bytes (~254 MB)** |

---

### Policy Status: STRONG VALIDATION

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| severe regressions | 0 | 0 | ✓ |
| memory-positive | 100% | 384/384 | ✓ |
| cosine improvement rate | ≥ 95% | 99.74% | ✓ |
| MAE improvement rate | ≥ 95% | 99.74% | ✓ |

---

### Minor Failure Tracking

**Known 31BL failures — both reproduced and remain minor:**

| Pair | Type | 31BL Value | 31BM Observed | Reproduced? | Still Minor? |
|------|------|-----------|---------------|------------|-------------|
| L21-S10 | cosine failure | dc=−0.0294 | dc=−0.0294 | ✓ | ✓ |
| L2-S13 | MAE regression | md=+0.0078 | md=+0.0078 | ✓ | ✓ |

**No new failures found.** The only failures are the 2 known from 31BL.

**Failure clustering analysis:**
- Layer distribution of failures: {L21: 1, L2: 1} — no clustering
- Seed distribution: {S10: 1, S13: 1} — no clustering
- Both failures are isolated, non-severe, at opposite ends of the model

---

### Per-Layer Summary

| Layer | cos_imp | mae_imp | sev | worst seed | worst dc | mean dc |
|-------|---------|---------|-----|------------|----------|---------|
| L0 | 16/16 | 16/16 | 0 | S14 | +0.0253 | +0.0499 |
| L1 | 16/16 | 16/16 | 0 | S15 | +0.0109 | +0.0298 |
| L2 | 16/16 | 15/16 | 0 | S2 | +0.0020 | +0.0277 |
| L3 | 16/16 | 16/16 | 0 | S12 | +0.0191 | +0.0307 |
| L4 | 16/16 | 16/16 | 0 | S2 | +0.0155 | +0.0289 |
| L5 | 16/16 | 16/16 | 0 | S10 | +0.0120 | +0.0311 |
| L6 | 16/16 | 16/16 | 0 | S4 | +0.0223 | +0.0328 |
| L7 | 16/16 | 16/16 | 0 | S15 | +0.0156 | +0.0329 |
| L8 | 16/16 | 16/16 | 0 | S7 | +0.0251 | +0.0459 |
| L9 | 16/16 | 16/16 | 0 | S6 | +0.0154 | +0.0325 |
| L10 | 16/16 | 16/16 | 0 | S12 | +0.0135 | +0.0319 |
| L11 | 16/16 | 16/16 | 0 | S0 | +0.0202 | +0.0403 |
| L12 | 16/16 | 16/16 | 0 | S14 | +0.0210 | +0.0535 |
| L13 | 16/16 | 16/16 | 0 | S9 | +0.0166 | +0.0387 |
| L14 | 16/16 | 16/16 | 0 | S12 | +0.0121 | +0.0373 |
| L15 | 16/16 | 16/16 | 0 | S14 | +0.0184 | +0.0322 |
| L16 | 16/16 | 16/16 | 0 | S7 | +0.0232 | +0.0363 |
| L17 | 16/16 | 16/16 | 0 | S15 | +0.0225 | +0.0387 |
| L18 | 16/16 | 16/16 | 0 | S6 | +0.0251 | +0.0396 |
| L19 | 16/16 | 16/16 | 0 | S0 | +0.0005 | +0.0404 |
| L20 | 16/16 | 16/16 | 0 | S9 | +0.0249 | +0.0455 |
| L21 | 15/16 | 16/16 | 0 | S10 | −0.0294 | +0.0610 |
| L22 | 16/16 | 16/16 | 0 | S0 | +0.0220 | +0.0434 |
| L23 | 16/16 | 16/16 | 0 | S7 | +0.0224 | +0.0392 |

**All 24 layers: 0 severe regressions, 99.74% cosine improvement, 99.74% MAE improvement.**

---

## Key Findings

1. **Policy is strongly validated across all 24 layers.**
2. **Both known minor failures (L21-S10, L2-S13) reproduce exactly** — they are stable properties of specific layer/seed/input-vector combinations, not random noise.
3. **No new failures appear** at any of the 22 layers not tested in 31BL.
4. **No clustering** — failures do not cluster by layer or seed.
5. **L21 has the highest mean delta_cos** (+0.0610) and hosts both the best (S0: +0.3152) and worst (S10: −0.0294) pairs in the entire run.
6. **Aggregate margin: ~254 MB** for the full 24-layer model — policy is well within budget.

---

## Next Allowed Phase

**Phase 31BN — Corrected Q2_K Full Aggregate Checkpoint / Freeze** (only if explicitly requested)

The policy is validated across all 24 layers. The next phase would produce a canonical checkpoint of the selected policy (quantization parameters, SDIR k values, family selection) for reproducibility.
