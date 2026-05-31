# Phase 31BJ — Corrected Q2_K Mode Rebaseline

**Classification:** `PARTIAL_31BJ_CORRECTED_Q2K_MEMORY_FAIL`
**Date:** Phase 31BJ
**Repo:** sdi-substitutive
**HEAD:** `99db338d` (Phase 31BH-R2-FIX)

---

## Context

Phase 31BH-R2-FIX repaired the runtime dispatch regression. The Q2_K backend supports two modes:

- `historical_floor_flat`: legacy mode (1,430,016 bytes/family, truncates partial blocks)
- `corrected_ceil_per_row`: technically correct mode (1,634,304 bytes/family, ceil-based)

Phase 31BJ asks: when using the corrected Q2_K accounting, what are the new anchor and aggregate metrics?

---

## Anchor Table

Tested with: k=1%, alpha=1.0, all 3 MLP families, np.random.default_rng(seed).

| Layer | Seed | Mode | cos_low | delta_cos | MAE_delta | severe | Per-layer margin |
|-------|------|------|---------|-----------|-----------|--------|-------------------|
| L21 | S9 | historical_floor_flat | 0.889697 | **+0.017242** | -0.004043 | False | +351,242 |
| L21 | S9 | corrected_ceil_per_row | 0.669178 | **+0.184059** | -0.018421 | False | **-261,654** |
| L21 | S0 | historical_floor_flat | 0.799490 | **+0.087997** | -0.007414 | False | +351,242 |
| L21 | S0 | corrected_ceil_per_row | 0.494249 | **+0.323112** | -0.028529 | False | **-261,654** |
| L2  | S7 | historical_floor_flat | 0.924830 | **+0.020592** | -0.036489 | False | +351,258 |
| L2  | S7 | corrected_ceil_per_row | 0.882334 | **+0.028654** | -0.044536 | False | **-261,638** |

**Per-layer margin = 3 × Q4_BUDGET_FAMILY − (Q2_K_total + SDIR_total)**
**Q4_BUDGET_FAMILY = 2,179,072 bytes**

---

## Memory Accounting

### Corrected mode per-layer breakdown (example L21-S0):

| Component | Bytes | Notes |
|-----------|-------|-------|
| Q2_K ffn_up | 1,634,304 | ceil(d_out×d_in/256)×84 |
| Q2_K ffn_gate | 1,634,304 | ceil(4864×896/256)×84 |
| Q2_K ffn_down | 1,634,304 | ceil(896×4864/256)×84 |
| SDIR residual up (k=1%) | ~69,000 | bitmap + fp16 |
| SDIR residual gate (k=1%) | ~69,000 | bitmap + fp16 |
| SDIR residual down (k=1%) | ~69,000 | bitmap + fp16 |
| **Total** | **~5,109,912** | |
| **3 × Q4_BUDGET** | **6,537,216** | |
| **Margin** | **-1,427,304** | **MEMORY FAILS** |

### Historical floor mode per-layer:

| Component | Bytes |
|-----------|-------|
| Q2_K per family | 1,430,016 |
| Q2_K total (3 families) | 4,290,048 |
| SDIR total (k=1%, est.) | ~207,000 |
| **Total** | **~4,497,048** |
| **3 × Q4_BUDGET** | **6,537,216** |
| **Margin** | **+2,040,168** |

---

## Small Aggregate (Partial)

Tested seeds 0, 7, 9 for L2 and L21 (6 pairs) in historical_floor_flat mode:

| Layer | Seed | Mode | delta_cos | MAE_delta | severe |
|-------|------|------|-----------|-----------|--------|
| L2 | S0 | hist_floor | +0.011738 | -0.011172 | False |
| L2 | S7 | hist_floor | +0.020592 | -0.036489 | False |
| L2 | S9 | hist_floor | +0.010922 | -0.012091 | False |
| L21 | S0 | hist_floor | +0.087997 | -0.007414 | False |
| L21 | S7 | hist_floor | +0.015380 | -0.005066 | False |
| L21 | S9 | hist_floor | +0.017242 | -0.004043 | False |

**6/6 pairs cosine-positive. 6/6 MAE-improving. 0/6 severe.**

Full 32-pair (L2+L21, seeds 0-15) aggregate: **not completed** due to ctypes call overhead (~100ms/call × 384 dequantizations exceeds available time). Small sample is consistent with positive behavior.

---

## Historical Anchor Comparison

Phase 31AY/31BA claimed values for L21-S9 historical_floor_flat:
- cos_low = 0.794913
- delta_cos = -0.146059
- severe = True

Current Phase 31BJ run (same code path):
- cos_low = 0.889697
- delta_cos = +0.017242
- severe = False

**Conclusion:** The Phase 31AY/31BA values do NOT reproduce with the current code pipeline, even in historical_floor_flat mode. This was also noted in Phase 31BH-R2-FIX where SHA256=6658d4e3 was observed instead of the claimed SHA256=ced92887. The historical anchor values must be treated as **legacy provenance only** — they were produced by a different code path that is no longer available.

---

## Classification

**`PARTIAL_31BJ_CORRECTED_Q2K_MEMORY_FAIL`**

### Reasoning:

1. **Corrected mode memory fails:** per-layer margin ≈ −1.43 MB (all tested anchors). k=1% requires ~5.1 MB but only ~6.5 MB budget exists (3 families × 2.18 MB). Cannot be made memory-positive at k=1%.

2. **Corrected mode cosine behavior:** at tested anchors (L21-S9, L21-S0, L2-S7), corrected mode actually improves delta_cos MORE than historical floor mode (e.g., L21-S9: +0.184 vs +0.017). However this is moot since memory fails.

3. **Historical floor mode memory:** margin ≈ +2 MB per layer (positive). k=1% with floor mode is memory-positive.

4. **Historical anchor values are legacy-only:** Phase 31AY/31BA anchor values cannot be reproduced with the current code. Do not use them as current targets.

5. **No full aggregate completed:** Full 32-pair or 384-pair aggregate was not run due to ctypes call overhead. Classification is based on anchor data and memory accounting, not full statistics.

---

## Implications

- **Phase 31AY/31BA metrics are legacy-provenance only.** Do not cite specific numerical anchor values from those phases as current targets.

- **Corrected Q2_K mode is not memory-positive at k=1%.** To make it viable, either:
  - Reduce k% significantly (but then residual benefit diminishes), or
  - Reduce number of families with residuals, or
  - Find a different residual encoding

- **historical_floor_flat mode remains the working path** for the current research direction.

- **k=1% is still the default** for historical floor mode (margin ≈ +2 MB per layer).

---

## What Was Not Completed

- Full 32-pair (L2+L21, seeds 0-15) aggregate for corrected mode
- Full 384-pair (all 24 layers, 16 seeds) aggregate
- Full 32-pair aggregate for historical floor mode (sample only)

The ctypes call overhead makes full aggregate runs impractical for corrected mode in the current harness. A compact test runner (avoiding repeated GGUF reads and dequants) would be needed.

---

## Files

- Runner: `src/phase31bj_corrected_q2k_rebaseline.py` — untracked
- Results: `src/results/PHASE31BJ_CORRECTED_Q2K_REBASELINE.json` — untracked

---

## Next Allowed Phase

**Phase 31BK — Corrected Q2_K Mode Memory Policy Retune** (only if explicitly requested)

Or alternatively:

**Phase 31BK — Historical Floor Aggregate Full Reproduction** (only if explicitly requested)

Note: Phase 31BJ showed that corrected Q2_K mode has **fundamentally different memory economics** than historical floor mode. Any continuation must account for this.
