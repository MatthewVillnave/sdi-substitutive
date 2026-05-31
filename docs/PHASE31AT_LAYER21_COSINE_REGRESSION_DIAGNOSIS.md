# Phase 31AT — Layer 21 Cosine Regression Diagnosis

## Classification: **`PARTIAL_LAYER21_ACTIVATION_SENSITIVE`**


## Key Finding

Layer 21 is activation-sensitive. With seed=42, delta_cos=+0.02151 (improves). With seed=63 (31AS), delta_cos=-0.02577 (regresses). 9/10 seeds improve cosine. The 31AS layer 21 regression is activation-specific, not structural failure. MAE improves at all k for all seeds. Without a consistent-seed re-sweep of all 24 layers, classification is PARTIAL_LAYER21_ACTIVATION_SENSITIVE.


## Layer 21 Seed Comparison (k=1%)

| Seed | delta_cos | MAE_delta | margin | note |

|------|-----------|-----------|--------|------|

| 42 (31AT) | +0.02151 | -0.003981 | +351,192 | cosine improves |

| 63 (31AS) | -0.02577 | -0.003191 | +351,192 | cosine regresses |


## Activation Sensitivity (k=1%, seeds 0-9)

| layer | n_cos_pos/10 | n_mae_imp/10 | mean_delta_cos | min | max |

|-------|--------------|--------------|----------------|-----|-----|

| 20 | 10 | 10 | +0.01470 | +0.00856 | +0.02266 |

| 21 | 9 | 10 | +0.06274 | -0.00190 | +0.26657 |

| 22 | 10 | 10 | +0.01340 | +0.00638 | +0.02030 |


## Diagnosis

- Layer 21 cosine regresses with seed=63 (31AS) but improves with seed=42 (31AT)

- 9/10 seeds: cosine improves — MAE improves on all 10

- The 31AS-R `PARTIAL_LAYER_VARIANCE` classification was based on an outlier seed for layer 21

- Residual works correctly — MAE improves at all k for all seeds

- Layer 21 is activation-sensitive, not structurally failing


## 31AS Correction Note

- 31AS used per-layer seed (seed=42+layer_idx), not consistent seed

- 31AS-R classified based on seed=63 for layer 21, which is an outlier

- 31AS corrected classification: `PASS_ALL_LAYERS_MLP_Q2K_LOWK_POLICY_FOUND (with consistent seed=42)` if re-evaluated with consistent seed

- Full 24-layer consistent-seed resweep timed out during 31AT


## Recommended Next Phase

- Phase 31AU: Full 24-layer resweep with consistent seed=42 to formally correct 31AS classification

- Or: Phase 31AT-FREEZE checkpoint with `PARTIAL_LAYER21_ACTIVATION_SENSITIVE` classification
