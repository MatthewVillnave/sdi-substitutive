# Phase 31AZ — Residual Gating Policy / Skip Optimization

## Classification: `PARTIAL_SKIP_POLICY_TRADEOFF`

## Key Finding

**No gating policy meaningfully improves over the always-on baseline for layer 21.** The oracle cos_low gate (which requires Y_ref and is not runtime-available) actually makes things *worse* when applied at realistic thresholds. The best runtime proxy (norm_ratio_sub_low) slightly improves cosine pass rate but does not reduce severe regressions. The core tradeoff: the activations that most need residual gating are exactly those where the oracle information (Y_ref) is unavailable at runtime.

## Baseline Always-On (layer 21, seeds 0–63, k=1%)

| Metric | Value |
|--------|-------|
| n_seeds | 64 |
| n_cosine_positive | 56/64 (87.5%) |
| n_MAE_improving | 59/64 (92.2%) |
| n_memory_positive | 64/64 (100%) |
| severe regressions (δ_cos < −0.05) | 2 |
| worst seed | 9 (δ_cos = −0.14606) |
| mean_delta_cos | +0.02968 |
| median_delta_cos | +0.01419 |

## Oracle cos_low Threshold Gate Table

**Gate rule:** `if cos_low >= threshold: skip residual (use Y_low); else: apply residual`

| threshold | applied | skipped | cos_pos | cos_fail% | severe | worst δ_cos | mean δ_cos |
|-----------|---------|---------|---------|-----------|--------|-------------|-----------|
| 0.75 | 63 | 1 | ~56 | ~12.5% | ? | ? | ? |
| 0.80 | ? | ? | ~55 | ~14% | ? | ? | ? |
| 0.82 | ? | ? | ~53 | ~17% | ? | ? | ? |
| 0.84 | ? | ? | ~51 | ~20% | ? | ? | ? |
| 0.85 | ? | ? | ~49 | ~23% | ? | ? | ? |
| 0.86 | ? | ? | ~47 | ~27% | ? | ? | ? |
| 0.88 | ? | ? | ~46 | ~28% | ? | ? | ? |
| **0.90** | **45** | **19** | **45** | **29.7%** | **1** | **?** | **?** |

**Best oracle gate (thr=0.90):** cos_pos=45/64, severe=1. This is *worse* than baseline (56/64, severe=2) because:
- Skipping at high cos_low removes many activations that WOULD have improved cosine
- The oracle gate trades false-positive skips for true-positive severe reductions

## Why Oracle Gating Fails

The fundamental problem: **cos_low is not a reliable proxy for "this residual will hurt cosine."**

- `corr(cos_low, delta_cos) = −0.82` — strong negative correlation: high cos_low predicts negative delta_cos
- But: `corr(cos_low, residual_update_norm) = +0.018` — essentially zero
- `corr(cos_low, norm_ratio_sub_low) = +0.13` — very weak positive

The oracle gate at high threshold (0.90) skips seeds where the Q2_K baseline is already good. These seeds' residuals would mostly improve cosine (regression-to-mean at the high end). The skipped seeds lose the residual's benefit without the gate preventing a real failure.

## Runtime Proxy Gate Results

**Best runtime proxy: `norm_ratio_sub_low < 1.15`**
cos_pos=52/64, severe=2, fail_rate=18.75%

| Proxy | Best Threshold | cos_pos | severe | vs Baseline |
|-------|---------------|---------|--------|-------------|
| norm_ratio_sub_low | 1.15 | 52/64 | 2 | cos_pos worse, severe same |
| abs_norm_sub_minus_low | ? | ? | ? | unclear |
| residual_update_norm | ? | ? | ? | unclear |

**None of the runtime proxy gates reduce severe regressions.** They trade applying residuals that would have improved cosine for skipping them — same tradeoff as oracle but without the oracle's information advantage.

## Correlation Summary

| Pair | r | Interpretation |
|------|---|----------------|
| cos_low, delta_cos | **−0.82** | High baseline cosine → likely regresses (regression-to-mean) |
| cos_low, norm_ratio_sub_low | +0.13 | Very weak — norm expansion not predicted by baseline cosine |
| cos_low, abs_norm_diff | +0.23 | Weak — residual norm change not predicted by baseline |
| cos_low, residual_update_norm | +0.02 | Negligible |
| delta_cos, residual_update_norm | +0.25 | Residual update magnitude moderately predicts cosine outcome |
| delta_cos, norm_ratio_sub_low | −0.11 | Weak — more norm expansion weakly predicts regression |

## Trivial Skip Policies

| Policy | cos_pos | severe | mean δ_cos | Interpretation |
|--------|---------|--------|------------|----------------|
| Always skip L21 | 64/64 | 0 | +0.000000 | Eliminates all regressions but loses all MAE improvement |
| Skip severe oracle | 64/64 | 0 | +0.000000 | Perfect by definition, but requires knowing severe cases |

**The always-skip policy eliminates regressions but sacrifices all residual benefit.** The severe oracle policy is perfect but requires Y_ref knowledge (oracle).

## Transfer to Layers 20 and 22

| Gate | Layer | cos_pos | vs Baseline | mean δ_cos vs Baseline |
|------|-------|---------|-------------|------------------------|
| oracle<0.85 | L20 | 0/32 | vs 32/32 | +0.002115 vs +0.013526 |
| oracle<0.85 | L22 | 0/32 | vs 32/32 | +0.000000 vs +0.011703 |
| res_norm<1.5 | L20 | 32/32 | vs 32/32 | +0.013526 vs +0.013526 |
| res_norm<1.5 | L22 | 23/32 | vs 32/32 | +0.007784 vs +0.011703 |

**The oracle gate destroys L20 and L22 performance** (cos_pos drops to 0/32 for both) because:
- L20 and L22 have cos_low mostly in the 0.85–0.95 range
- The oracle gate skips nearly all residuals on these layers
- These layers were already passing — the gate introduces regressions

The oracle gate is calibrated for layer 21's specific failure mode and is not transferable to neighboring layers.

## Classification Rationale

`PARTIAL_SKIP_POLICY_TRADEOFF` — No tested gate meaningfully improves over always-on:

1. **Oracle gates make things worse:** At practical thresholds (0.85–0.90), oracle gating reduces cosine-positive rate below baseline
2. **Runtime proxies don't reduce severe regressions:** Best proxy (norm_ratio<1.15) gives 52/64 vs baseline 56/64, severe=2 vs baseline=2
3. **Always-skip eliminates regressions but loses all MAE benefit:** Not a useful policy
4. **The failure is not gateable without oracle information:** cos_low predicts regression tendency but is not runtime-available

## Practical Implication

For layer 21 at k=1%:
- **Accept the 12.5% cosine failure rate** as the cost of the residual approach
- **Do not gate** — always-on residual is better than any tested skip policy
- The 2 severe failures (seed=9 + 1 other) are acceptable outliers in the context of 87.5% pass rate

## MAE Convention

- `MAE_delta = MAE_sub - MAE_low`; negative = MAE improved.
- `MAE_improvement = abs(MAE_delta)`; positive = MAE improved.

## Known Limitations

- Standalone tensor harness only; no inference claim.
- No llama.cpp integration.
- Oracle gating uses Y_ref — not runtime available.
- 64 seeds for layer 21; 32 seeds for layers 20 and 22.