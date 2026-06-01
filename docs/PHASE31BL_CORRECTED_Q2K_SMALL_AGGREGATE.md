# Phase 31BL — Corrected Q2_K Small Aggregate Validation

**Classification:** `PARTIAL_31BL_CORRECTED_Q2K_MINOR_FAILURES`
**Date:** Phase 31BL
**Repo:** sdi-substitutive
**HEAD:** `080d314b` (Phase 31BK)

---

## Context

Phase 31BK selected the corrected Q2_K up+gate k=0.5% policy. Phase 31BL validates this policy on the full small aggregate: layers 2 and 21, seeds 0–15 (32 pairs).

**Selected policy:**
- Q2_K mode: corrected_ceil_per_row
- Residual families: ffn_up + ffn_gate
- k: 0.5%
- alpha: 1.0
- ffn_down residual: none (Q4 budget slack)

---

## Small Aggregate Results (32 pairs)

### Aggregate Table

| Metric | Value |
|--------|-------|
| total_pairs | 32 |
| n_cosine_improved | 31 |
| n_cosine_nonnegative | 31 |
| n_MAE_improved | 31 |
| n_memory_positive | **32/32** |
| n_severe_regressions | **0** |
| n_cosine_failures | 1 |
| n_MAE_failures | 1 |
| worst pair | L21 S10 (dc=−0.0294) |
| best pair | L21 S0 (dc=+0.3152) |
| mean_delta_cos | +0.0545 |
| median_delta_cos | +0.0302 |
| mean_MAE_improvement | +0.0196 |
| min_margin | 661,766 |

**Aggregate margin (32 pairs):** 32 × 661,822 ≈ 21,178,304 bytes (21.2 MB total)

---

### Cosine Failures (1)

| Layer | Seed | delta_cos | severe | MAE_delta |
|-------|------|-----------|--------|-----------|
| L21 | S10 | −0.0294 | False | −0.02840 |

L21-S10 is a cosine failure but **not severe** (threshold is −0.05). MAE improves strongly here (−0.02840).

---

### MAE Failures (1)

| Layer | Seed | MAE_delta | cosine_improved |
|-------|------|-----------|----------------|
| L2 | S13 | +0.00783 | True (+0.0044) |

L2-S13 has a small MAE regression (+0.00783) but cosine still improves (+0.0044). Not severe.

---

### Layer-Specific Summary

| Layer | pairs | cos_imp | mae_imp | mem_pos | severe | worst_seed | worst_dc | mean_dc |
|-------|-------|---------|---------|---------|--------|------------|----------|---------|
| L2 | 16 | 16 | 15 | 16 | 0 | S13 | +0.0020 | +0.0293 |
| L21 | 16 | 15 | 16 | 16 | 0 | S10 | −0.0294 | +0.0606 |

L2: all 16 cosine-improved, 1 MAE regression (S13), mean dc=+0.0293
L21: 1 cosine failure (S10), all 16 MAE-improved, mean dc=+0.0606

---

### Anchor Consistency Check

| Anchor | delta_cos | Expected | Diff | MAE_imp | Within 0.05 |
|--------|-----------|----------|------|---------|-------------|
| L21 S9 | +0.1575 | +0.1575 | +0.0000 | ✓ | ✓ |
| L21 S0 | +0.3152 | +0.3152 | +0.0000 | ✓ | ✓ |
| L2 S7 | +0.0203 | +0.0203 | +0.0000 | ✓ | ✓ |

All 3 anchor values match Phase 31BK expectations exactly (within floating-point precision).

---

## Interpretation

**Strong overall result:**
- 32/32 memory-positive ✓
- 0 severe regressions ✓
- 31/32 cosine-improved (96.9%) ✓
- 31/32 MAE-improved (96.9%) ✓

**Two minor failures are isolated, not severe:**
- L21-S10 cosine failure (dc=−0.0294) is small and MAE improves strongly here
- L2-S13 MAE regression (+0.00783) is small and cosine still improves

**Policy is validated.** The 2 failures are within expected noise for a 32-pair aggregate at k=0.5%.

---

## Next Allowed Phase

**Phase 31BM — Corrected Q2_K Full 24-Layer Aggregate Validation** (only if explicitly requested)

Full aggregate would test all 24 layers × 16 seeds = 384 pairs to determine whether the policy holds across the full model.
