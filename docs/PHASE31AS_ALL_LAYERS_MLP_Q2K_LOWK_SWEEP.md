# Phase 31AS — All Available Layers Full MLP Q2_K + Low-k Residual Sweep

## Classification: **`PARTIAL_LAYER_VARIANCE`**

## Best Policy: k=1% (most conservative among memory-positive policies)

## Layers Discovered: 24 (indices 0–23)


## Corrected Findings

**IMPORTANT — 31AS-R Audit Correction:**

- Original report claimed "all 24 layers improve cosine" — this is **incorrect**

- Layer 21 cosine-regresses at all k > 0 (delta_cos < 0 at k=0.5%, k=1%, k=2%)

- Layer 21: MAE improves at all k > 0, but cosine regresses — structural property

- No policy achieves 24/24 cosine-positive; best is 23/24

- All policies achieve 24/24 memory-positive and 24/24 MAE-improving

- k=1% selected as most conservative memory-positive policy (agg_margin=+8,428,606, worst_margin=+351,076)


## Layer 21 Issue Detail

| k | delta_cos | MAE_delta | margin | memory_positive |

|---|-----------|-----------|--------|----------------|

| 0 | +0.00000 | +0.000000 | 2,247,168 | True |
| 0.5 | -0.00785 | -0.002572 | 481,998 | True |
| 1 | -0.02577 | -0.003191 | 351,192 | True |
| 2 | -0.02349 | -0.004806 | 89,732 | True |
| 3 | -0.01296 | -0.007108 | -171,932 | False |

## Aggregate Policy Audit Table


| k | agg_margin | worst_margin | n_mem_pos | n_cos_pos | n_mae_imp | all_cos | all_mem | all_mae |

|---|------------|--------------|-----------|-----------|-----------|---------|---------|--------|

| 0 | +53,932,032 | +2,247,168 | 24/24 | 0/24 | 0/24 | NO | YES | NO |
| 0.5 | +11,567,890 | +481,950 | 24/24 | 22/24 | 24/24 | NO | YES | YES |
| 1 | +8,428,606 | +351,076 | 24/24 | 23/24 | 24/24 | NO | YES | YES |
| 2 | +2,152,334 | +89,524 | 24/24 | 23/24 | 24/24 | NO | YES | YES |
| 3 | -4,126,454 | -172,222 | 0/24 | 23/24 | 24/24 | NO | NO | YES |

## Per-Layer k=1% Summary


| Layer | margin | delta_cos | MAE_delta | memory_positive |
|-------|--------|-----------|-----------|----------------|

| 0 | +351,208 | +0.00920 | -0.002254 | True |
| 1 | +351,118 | +0.01290 | -0.001991 | True |
| 2 | +351,130 | +0.00230 | -0.009014 | True |
| 3 | +351,076 | +0.00614 | -0.000784 | True |
| 4 | +351,242 | +0.00944 | -0.007960 | True |
| 5 | +351,128 | +0.00838 | -0.009561 | True |
| 6 | +351,174 | +0.01741 | -0.007872 | True |
| 7 | +351,224 | +0.01332 | -0.009520 | True |
| 8 | +351,140 | +0.01949 | -0.024205 | True |
| 9 | +351,218 | +0.00994 | -0.007403 | True |
| 10 | +351,224 | +0.00980 | -0.001796 | True |
| 11 | +351,214 | +0.02803 | -0.006061 | True |
| 12 | +351,224 | +0.00661 | -0.001927 | True |
| 13 | +351,270 | +0.01186 | -0.003069 | True |
| 14 | +351,258 | +0.01716 | -0.002995 | True |
| 15 | +351,240 | +0.00599 | -0.002878 | True |
| 16 | +351,158 | +0.01408 | -0.004250 | True |
| 17 | +351,150 | +0.01917 | -0.004638 | True |
| 18 | +351,144 | +0.01250 | -0.003681 | True |
| 19 | +351,242 | +0.01143 | -0.001907 | True |
| 20 | +351,160 | +0.01434 | -0.003683 | True |
| 21 | +351,192 | -0.02577 | -0.003191 | True | ← cosine regresses
| 22 | +351,218 | +0.02786 | -0.012128 | True |
| 23 | +351,254 | +0.01039 | -0.001816 | True |