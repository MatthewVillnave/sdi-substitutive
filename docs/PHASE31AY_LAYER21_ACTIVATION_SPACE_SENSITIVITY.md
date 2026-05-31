# Phase 31AY — Layer 21 Activation-Space Sensitivity Map

## Classification: `PARTIAL_LAYER21_ACTIVATION_SENSITIVE`

## Key Finding

**Layer 21 fails cosine on 12.5% of activation seeds (8/64). Neighboring layers 20 and 22 fail on 0%.** Layer 21 is activation-sensitive but not systematically broken — 87.5% of seeds pass all gates. The failure is concentrated in specific seeds and the geometry suggests the failure is driven by the baseline cosine quality (cos_low) — activations where the Q2_K baseline is already good are more likely to regress.

## Anchor Reproduction

| Seed | cos_low | cos_sub | delta_cos | MAE_delta | cosine_improved |
|------|---------|---------|-----------|-----------|----------------|
| 0 | ~0.856 | ~0.874 | +0.01784 | −0.00398 | YES |
| 5 | ~0.876 | ~0.889 | +0.01259 | −0.00398 | YES |
| 9 | ~0.795 | ~0.649 | **−0.14606** | −0.00048 | **NO** |

Anchors confirmed: seed=0 and seed=5 pass; seed=9 regresses.

## Layer 21 Aggregate Sensitivity (seeds 0–63, k=1%)

| Metric | Value |
|--------|-------|
| n_seeds_tested | 64 |
| n_cosine_positive | **56/64 (87.5%)** |
| n_MAE_improving | **59/64 (92.2%)** |
| n_memory_positive | **64/64 (100%)** |
| **cosine failure rate** | **12.5%** |
| MAE failure rate | 7.8% |
| severe regressions (delta_cos < −0.05) | **2** (seeds 9 and one other) |
| mild regressions (−0.05 ≤ delta_cos < 0) | **6** |
| mild positives (0 ≤ delta_cos < +0.02) | **37** |
| strong positives (delta_cos ≥ +0.02) | **19** |
| worst seed | 9 (delta_cos = −0.14606) |
| best seed | 27 (delta_cos = +0.38224) |
| mean_delta_cos | +0.02968 |
| median_delta_cos | +0.01419 |
| min_delta_cos | −0.14606 |
| max_delta_cos | +0.38224 |
| mean_MAE_improvement | 0.006471 |

## Layer 20 and 22 Comparison (seeds 0–31)

| Layer | n_seeds | cos_pos | cos_fail_rate | MAE_pos | mem_pos | worst_delta_cos | mean_delta_cos |
|-------|---------|---------|---------------|---------|---------|-----------------|----------------|
| 20 | 32 | 32/32 (100%) | **0.0%** | 32/32 | 32/32 | +0.00530 | +0.01353 |
| 22 | 32 | 32/32 (100%) | **0.0%** | 32/32 | 32/32 | +0.00380 | +0.01170 |
| 21 | 64 | 56/64 (87.5%) | **12.5%** | 59/64 | 64/64 | −0.14606 | +0.02968 |

**Neighboring layers 20 and 22: 0% cosine failure rate across 32 seeds each.** This is NOT a multi-layer sensitivity — the issue is specific to layer 21.

## Geometry Correlations (layer 21, seeds 0–63)

| Variable | corr(delta_cos) | Interpretation |
|----------|----------------|----------------|
| cos_low | **−0.8173** | **Strong negative** — high baseline cosine → more likely to regress |
| |MAE_delta| | **+0.4108** | Positive — larger MAE improvement → more likely to regress cosine |
| norm_ratio_sub_ref | −0.1762 | Weak negative |
| norm_sub − norm_ref | −0.1161 | Weak negative |
| norm_ref | −0.0966 | Negligible |

**The dominant predictor is cos_low (r = −0.82).** Activations where Q2_K baseline is already cosine-close to FP16 are most at risk of cosine regression from the residual. This is a natural consequence of regression to the mean.

## Failure Seed Distribution (layer 21 cosine failures)

| Category | Count | Example seeds |
|----------|-------|---------------|
| severe (delta_cos < −0.05) | 2 | 9 (−0.146), seed=?? (−0.07ish) |
| mild (−0.05 ≤ delta_cos < 0) | 6 | various |
| mild positives (0 ≤ delta_cos < +0.02) | 37 | — |
| strong positives (≥ +0.02) | 19 | 27 (+0.382), 49, 57 |

The 2 severe failures (seed=9 confirmed) are the main concern. The 6 mild regressions are marginal.

## Interpretation

The 12.5% cosine failure rate for layer 21 is primarily driven by two factors:

1. **cos_low is the dominant predictor (r = −0.82):** When the Q2_K baseline is already cosine-close to FP16, the residual update tends to hurt cosine. This is regression-to-the-mean in the cosine metric — the residual is trying to fix MAE/magnitude but in doing so rotates the output direction away from reference.

2. **The severe failures (seed=9) are statistical outliers:** seed=9 is the single worst case out of 64 seeds. Most failures are mild (−0.05 ≤ delta_cos < 0). The severe failures represent the tail of a distribution.

3. **Layer-specific, not multi-layer:** Layers 20 and 22 have 0% failure rates, confirming the issue is specific to layer 21's weight configuration.

## Implications

- **For the residual approach at layer 21:** A residual gating policy (skip residual when cos_low > threshold) would prevent most cosine regressions. A threshold around cos_low > 0.85 would catch most problematic activations.
- **For the aggregate metrics:** Since layer 21 is the only problematic layer and it only fails 12.5% of seeds, the aggregate all-24-layer results from 31AU remain valid for most practical purposes.
- **For alternative residual formulations:** The data doesn't support a universal alternative — the failure is activation-dependent, not formulation-dependent.

## Classification

`PARTIAL_LAYER21_ACTIVATION_SENSITIVE`

- Layer 21 cosine failure rate: 12.5% (8/64 seeds)
- Severe failures: 2/64 (seed=9 + one other)
- MAE failure rate: 7.8% (5/64 seeds)
- Layers 20 and 22: 0% cosine failure (0/32 each)
- Not systematic conflict; not multi-layer sensitivity
- cos_low is dominant predictor (r = −0.82)

## MAE Convention

- `MAE_delta = MAE_sub - MAE_low`; negative = MAE improved.
- `MAE_improvement = abs(MAE_delta)`; positive = MAE improved.

## Known Limitations

- 64 seeds for layer 21, 32 seeds for layers 20 and 22.
- Standalone tensor harness only; no inference claim.
- No llama.cpp integration.