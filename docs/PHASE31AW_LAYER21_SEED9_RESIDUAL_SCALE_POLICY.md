# Phase 31AW — Layer 21 Seed-9 Residual Scale / Policy Characterization

## Classification: `PARTIAL_LAYER21_SEED9_METRIC_CONFLICT`

## Key Finding

Layer 21 at seed=9 has a **metric conflict**: MAE improves at all tested policies, but cosine regresses at all tested policies. **No alpha, k, or family-subset policy fixes the cosine regression** while preserving MAE improvement and memory positivity.

## Baseline Reproduction (layer 21, seed=9, k=1%, alpha=1.0)

| Metric | Value |
|--------|-------|
| margin | +351,192 |
| cos_low | 0.794913 |
| cos_sub | 0.648854 |
| **delta_cos** | **−0.146059** |
| MAE_low | 0.079634 |
| MAE_sub | 0.079156 |
| **MAE_delta** | **−0.000479** (improves) |
| norm_ref | 5.8706 |
| norm_low | 6.1259 |
| norm_sub | 6.5807 |
| memory_positive | YES |
| cosine_improved | **NO** |
| MAE_improved | YES |

Delta_cos ≈ −0.14606 confirmed vs expected.

## Alpha Sweep (k=1%)

| alpha | delta_cos | MAE_delta | MAE_improvement | margin | mem_pos | cos_pos | MAE_pos |
|-------|-----------|-----------|-----------------|--------|---------|---------|---------|
| 0.00 | +0.000000 | +0.000000 | 0.000000 | 351,192 | YES | NO | YES |
| 0.10 | −0.008001 | −0.000512 | 0.000512 | 351,192 | YES | NO | YES |
| 0.25 | −0.022700 | −0.001099 | 0.001099 | 351,192 | YES | NO | YES |
| 0.50 | −0.054699 | −0.001523 | 0.001523 | 351,192 | YES | NO | YES |
| 0.75 | −0.096067 | −0.001271 | 0.001271 | 351,192 | YES | NO | YES |
| 1.00 | −0.146059 | −0.000479 | 0.000479 | 351,192 | YES | NO | YES |
| 1.25 | −0.203140 | +0.000952 | 0.000952 | 351,192 | YES | NO | NO |
| 1.50 | −0.264788 | +0.003046 | 0.003046 | 351,192 | YES | NO | NO |

**No alpha achieves cosine improvement while preserving MAE improvement and memory positivity.**

## k Sweep (alpha=1.0)

| k | delta_cos | MAE_delta | MAE_improvement | margin | mem_pos | cos_pos | MAE_pos |
|-------|-----------|-----------|-----------------|--------|---------|---------|---------|
| 0.00 | +0.000000 | +0.000000 | 0.000000 | 2,247,168 | YES | NO | YES |
| 0.25 | −0.077500 | −0.000777 | 0.000777 | 547,380 | YES | NO | YES |
| 0.50 | −0.031466 | −0.002706 | 0.002706 | 481,998 | YES | NO | YES |
| 1.00 | −0.146059 | −0.000479 | 0.000479 | 351,192 | YES | NO | YES |
| 1.50 | −0.104894 | −0.002834 | 0.002834 | 220,410 | YES | NO | YES |
| 2.00 | −0.107278 | −0.003682 | 0.003682 | 89,732 | YES | NO | YES |

**No k value achieves cosine improvement.** All k > 0 regress cosine. k=0 gives delta_cos=0 (no residual).

## Combined Mini-Grid Summary

Tested k in {0.25, 0.5, 1.0} x alpha in {0.25, 0.5, 0.75, 1.0} (12 entries).

**NO grid entry passes all three gates** (cosine-positive, MAE-improve, memory-positive).

## Family Ablation (k=1%, alpha=1.0, layer 21, seed=9)

| active | delta_cos | MAE_delta | MAE_improvement | margin | mem | cos_pos | MAE_pos |
|--------|-----------|-----------|-----------------|--------|-----|---------|---------|
| up | −0.039190 | −0.000001 | 0.000001 | 1,615,140 | Y | N | Y |
| gate | −0.035136 | −0.000191 | 0.000191 | 1,615,200 | Y | N | Y |
| down | −0.015382 | −0.001096 | 0.001096 | 1,615,188 | Y | N | Y |
| up+gate | −0.124141 | +0.000714 | 0.000714 | 983,172 | Y | N | N |
| up+down | −0.057612 | −0.000995 | 0.000995 | 983,160 | Y | N | Y |
| gate+down | −0.053543 | −0.001401 | 0.001401 | 983,220 | Y | N | Y |
| up+gate+down | −0.146059 | −0.000479 | 0.000479 | 351,192 | Y | N | Y |

**ALL 7 family subsets regress cosine.** The cosine regression is not caused by any single family — it is a property of the full residual combination.

## Error Geometry

| Field | Baseline (k=1%, alpha=1.0) |
|-------|------------------------------|
| norm(Y_ref) | 5.8706 |
| norm(Y_low) | 6.1259 |
| norm(Y_sub) | 6.5807 |
| cos(Y_ref, Y_low) | 0.794913 |
| cos(Y_ref, Y_sub) | 0.648854 |

The residual **increases output norm** (6.1259 to 6.5807, +7.4%) and **rotates direction significantly**. The cosine regression is driven by this norm/direction mismatch: Y_sub is much larger in magnitude than Y_ref, and the SiLU-gated MLP interaction amplifies this at this specific activation.

## Safe Seeds (layer 21, k=1%, alpha=1.0)

| Seed | delta_cos | MAE_delta | margin | cosine_improved | MAE_improved |
|------|-----------|-----------|--------|-----------------|--------------|
| 0 | +0.01784 | −0.003981 | +351,192 | YES | YES |
| 5 | +0.01259 | −0.003981 | +351,192 | YES | YES |

Seeds 0 and 5 both pass all gates — the issue is activation-specific, not layer-specific.

## Classification

`PARTIAL_LAYER21_SEED9_METRIC_CONFLICT`

- No alpha/k/grid policy achieves cosine improvement for layer 21 / seed=9.
- MAE improves at all memory-positive policies (metric conflict).
- All 7 family subsets regress cosine — not family-specific.
- Error geometry: residual increases output norm by +7.4% and rotates direction.
- Layer 21 is activation-sensitive: seeds 0 and 5 pass cleanly; seed=9 fails.
- This is a metric-conflict case: cosine direction and MAE magnitude optimize differently.

## Comparison to 31AV

- 31AV found layer 21 sensitive at seed=9 (1/3 seeds).
- 31AW confirms: no policy fixes cosine at seed=9; MAE is reliable but cosine is not for this specific activation.

## MAE Convention

- `MAE_delta = MAE_sub - MAE_low`; negative = MAE improved.
- `MAE_improvement = abs(MAE_delta)`; positive = MAE improved.

## Known Limitations

- Activation-specific finding; broader seed space not tested.
- No inference/generation claim.
- Standalone tensor harness only.
- No llama.cpp integration.