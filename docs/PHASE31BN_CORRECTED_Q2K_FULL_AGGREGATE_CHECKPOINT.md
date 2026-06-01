# Phase 31BN — Corrected Q2_K Full Aggregate Checkpoint / Freeze

**Classification:** `PASS_31BN_FREEZE_PREPARED_CORRECTED_Q2K_FULL_AGGREGATE`
**Date:** Phase 31BN
**Repo:** sdi-substitutive
**Checkpoint target commit:** `0304590c92d43fdf48d3d28998255d39c9a20c07`
**Proposed tag:** `phase31bn-corrected-q2k-full-aggregate-checkpoint`

---

## Checkpoint Purpose

Phase 31BN is a **checkpoint/freeze phase** — no new science, no new experiments. It creates a canonical documentation record of the selected corrected Q2_K policy and its full 384-pair aggregate validation, suitable for future reproducibility and tag creation.

---

## Selected Policy (Frozen)

| Parameter | Value |
|-----------|-------|
| Q2_K mode | `corrected_ceil_per_row` |
| Residual families | `ffn_up` + `ffn_gate` |
| Residual k | `0.5%` |
| Alpha | `1.0` |
| ffn_down residual | **none** (left as Q4 budget slack) |

---

## Validated Scope

- **Model:** Qwen2.5-0.5B (GGUF Q2_K)
- **Target:** Standalone full-MLP tensor harness only
- **Layers:** All 24 FFN layers (L0 – L23)
- **Seeds:** 0 – 15 (16 seeds per layer)
- **Total pairs:** 384
- **Route:** Route A (full 24×16, completed in ~168s)

---

## Frozen Aggregate Result Table

| Metric | Value |
|--------|-------|
| n_memory_positive | **384 / 384 (100%)** |
| n_cosine_improved | **383 / 384 (99.74%)** |
| n_MAE_improved | **383 / 384 (99.74%)** |
| n_severe_regressions | **0** |
| n_cosine_failures | 1 |
| n_MAE_failures | 1 |
| mean_delta_cos | +0.0383 |
| median_delta_cos | +0.0351 |
| min_margin | 661,766 bytes/layer |
| aggregate_margin | ~254,130,400 bytes (~254 MB) |
| Policy status | **STRONG VALIDATION** |

---

## Known Minor Failures (Documented & Accepted)

| Pair | Type | delta_cos | MAE_delta | Severity |
|------|------|-----------|-----------|----------|
| L21-S10 | cosine failure | −0.0294 | −0.0284 | **Non-severe** (threshold −0.05) |
| L2-S13 | MAE regression | +0.0077 | +0.0078 | **Non-severe** (cosine improves) |

Both failures reproduce exactly across 31BL and 31BM runs. No new failures appeared in the 22 additional layers tested in 31BM.

---

## Per-Layer Summary (from 31BM Route A)

All 24 layers: 16/16 memory-positive, 0 severe regressions.

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
| **L21** | 15/16 | 16/16 | 0 | S10 | **−0.0294** | +0.0610 |
| L22 | 16/16 | 16/16 | 0 | S0 | +0.0220 | +0.0434 |
| L23 | 16/16 | 16/16 | 0 | S7 | +0.0224 | +0.0392 |

---

## Known Limitations

- **Standalone tensor harness only** — no llama.cpp runtime integration validated
- **No generation/inference validation** — only cosine similarity and MAE on random inputs
- **No larger-model validation** — results are specific to Qwen2.5-0.5B
- **Two minor failures** documented and accepted as non-severe
- **Historical anchors (31AY/31BA)** are legacy provenance only — not current canonical metrics for corrected mode
- **historical_floor_flat** is not the selected future-cleaner policy

---

## Superseded / Historical Notes

- Old exact anchor values from Phases 31AY and 31BA are **legacy-provenance only** and do not reproduce with current code — they are not current canonical corrected-mode metrics
- The **historical_floor_flat** policy (all 3 families, k=1%) is not the selected policy for future work — it served as a comparison baseline only

---

## Forbidden Claims (Applies to All Downstream Work)

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference/generation claim
- no larger-model claim
- no runtime-ready output-residual claim
- no claim beyond standalone tensor harness
- no claim that 31AY/31BA exact anchors are current canonical metrics

---

## Proposed Tag Command

```
git tag -a phase31bn-corrected-q2k-full-aggregate-checkpoint 0304590c92d43fdf48d3d28998255d39c9a20c07 -m "Phase 31BN corrected Q2_K full aggregate checkpoint"
```

**Do not run until Matt approves.**

---

## Next Recommended Phase

**Phase 31BO — Corrected Q2_K Artifact/Policy Package Hardening** (only if explicitly requested)

Rationale: After freeze, the next clean step is to package the selected policy and artifact assumptions for reproducibility before larger-model validation or runtime integration.
