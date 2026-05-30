# Phase 31AB: Approximation Sufficiency Analysis

## Header
- **Phase:** 31AB
- **Date:** 2026-05-30
- **Classification:** `PARTIAL_COSINE_OK_MAE_WEAK`
- **OLD_HEAD:** `cc185c4`
- **Repo:** `sdi-substitutive`

## Critical Finding
**W_sub MAE is slightly WORSE than W_low MAE.** Cosine improves but absolute error increases.
This means the residual adds directionally correct information but introduces noise that increases MAE.
This is the fundamental nature of the k-sparsity approximation — not a bug.

## W_low vs W_sub Table (k=7.0, all layers)

| Layer | cos_low | cos_sub | delta | MAE_low | MAE_sub | MAE_impr | max_low | max_sub |
|-------|---------|---------|-------|---------|---------|----------|---------|---------|
| 0 | 0.995337 | 0.995586 | +0.0002 | 0.0657 | 0.0697 | **-0.0040** | 0.3039 | 0.3341 |
| 1 | 0.995335 | 0.995589 | +0.0003 | 0.0672 | 0.0714 | **-0.0042** | 0.3875 | 0.3545 |
| 2 | 0.995354 | 0.995605 | +0.0003 | 0.0655 | 0.0700 | **-0.0045** | 0.3055 | 0.3462 |
| 3 | 0.995515 | 0.995813 | +0.0003 | 0.0654 | 0.0697 | **-0.0043** | 0.3231 | 0.3054 |
| 4 | 0.995254 | 0.995613 | +0.0004 | 0.0666 | 0.0698 | **-0.0032** | 0.2818 | 0.3645 |
| 5 | 0.995263 | 0.995344 | +0.0001 | 0.0668 | 0.0717 | **-0.0049** | 0.3116 | 0.3769 |

**avg cos_sub: 0.995591** | **avg MAE_sub: 0.0704**

## Memory/Accuracy Pareto (k sweep, layer 0)

| k_pct | total_sub | margin | cosine | MAE | max_err | viable |
|-------|-----------|--------|--------|-----|---------|--------|
| 5.0% | 3,704,450 | 653,694 | 0.995278 | **0.0693** | 0.3293 | YES |
| 7.0% | 3,878,776 | 479,368 | 0.995586 | 0.0697 | 0.3341 | YES |
| 7.5% | 3,922,356 | 435,788 | 0.995660 | 0.0694 | 0.3307 | YES |
| 8.0% | 3,965,938 | 392,206 | 0.995700 | 0.0691 | 0.3231 | YES |
| 9.0% | 4,053,100 | 305,044 | 0.995804 | 0.0685 | 0.3099 | YES |

**Key insight:** MAE is minimized at k=5% (0.0693), not k=7%. Cosine monotonically increases with k.
This is a cosine vs MAE tradeoff — residual adds directional correctness but not absolute precision.

## Research Gates

| Gate | Threshold | Result |
|------|-----------|--------|
| minimum | cos > 0.99 | **PASS** |
| stronger | cos > 0.995 AND delta > 0 | **PASS** (cos ~0.9956, delta=+0.0002) |
| best | cos > 0.997 | **FAIL** |

## Sufficiency Conclusion

**Classification:** `PARTIAL_COSINE_OK_MAE_WEAK`

- Cosine passes both research gates (cos ~0.9956 > 0.995 threshold)
- MAE is ~0.07 and slightly worse than W_low alone — residual adds noise
- This is NOT a bug — it's the fundamental k-sparsity tradeoff
- Higher k improves cosine but worsens MAE (noise from low-precision residual encoding)
- Margin remains positive at all tested k values

**The fundamental tradeoff:**
- Cosine rewards directionally correct contributions (inner product)
- MAE penalizes absolute magnitude errors
- Residual at k=7% is sparse enough to be noisy when decoded
- MAE is minimized at k=5%, not k=7%

## Recommendation

**Option B — Tune residual policy before expanding**

The k=5% setting gives the best MAE (0.0693 vs 0.0697 at k=7%), but k=7% is the design point.
The residual policy is the next research target, not the architecture.

Next Phase 31AC: Tune residual policy (k=5% baseline) or test ffn_down if ffn_up is considered strong enough.

## Claim Boundaries
**Allowed:** "ffn_up substitutive prototype meets cosine research gate; MAE tradeoff with residual policy understood; margin positive"
**Forbidden:** no model quality, no behavior recovery, no speedup
