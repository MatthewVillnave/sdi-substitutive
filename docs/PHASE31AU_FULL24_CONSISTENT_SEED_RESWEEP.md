# Phase 31AU — Full 24-Layer Consistent-Seed Resweep

## Classification: **`PASS_ALL_LAYERS_CONSISTENT_SEED_POLICY_FOUND`**

## Best Policy: k=1%


## Methodology

- Consistent seed=42 for all 24 layers (one X_fixed activation)

- W_low cached in memory per layer (no re-quantization per k)

- Full MLP formula: Y = (SiLU(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T


## Aggregate k=1%

| Metric | Value |

|--------|-------|

| n_memory_positive | 24/24 |

| n_cosine_positive | 24/24 |

| n_MAE_improving | 24/24 |

| aggregate_margin | 8,428,606 |

| worst_margin | layer3: 351,076 |

| worst_cosine | layer4: +0.00500 |

| mean_delta_cos | +0.01199 |

| mean_MAE_improvement | +0.004335 |

| all_pass | mem=True, cos=True, mae=True |


## Per-Layer k=1%

| Layer | margin | delta_cos | MAE_improvement | mem_pos |

|-------|--------|-----------|-----------|--------|

| 0 | +351,208 | +0.00920 | +0.002254 | YES |
| 1 | +351,118 | +0.00797 | +0.000891 | YES |
| 2 | +351,130 | +0.00931 | +0.011314 | YES |
| 3 | +351,076 | +0.00874 | +0.000697 | YES |
| 4 | +351,242 | +0.00500 | +0.003557 | YES |
| 5 | +351,128 | +0.01317 | +0.006436 | YES |
| 6 | +351,174 | +0.00562 | +0.004060 | YES |
| 7 | +351,224 | +0.00645 | +0.004395 | YES |
| 8 | +351,140 | +0.00964 | +0.011869 | YES |
| 9 | +351,218 | +0.01249 | +0.005838 | YES |
| 10 | +351,224 | +0.01226 | +0.003491 | YES |
| 11 | +351,214 | +0.01900 | +0.002992 | YES |
| 12 | +351,224 | +0.01259 | +0.003048 | YES |
| 13 | +351,270 | +0.00824 | +0.003670 | YES |
| 14 | +351,258 | +0.00945 | +0.002249 | YES |
| 15 | +351,240 | +0.01746 | +0.004531 | YES |
| 16 | +351,158 | +0.02614 | +0.005630 | YES |
| 17 | +351,150 | +0.00591 | +0.002988 | YES |
| 18 | +351,144 | +0.02277 | +0.006931 | YES |
| 19 | +351,242 | +0.01488 | +0.004182 | YES |
| 20 | +351,160 | +0.01430 | +0.002807 | YES |
| 21 | +351,192 | +0.02151 | +0.003981 | YES |
| 22 | +351,218 | +0.01030 | +0.005592 | YES |
| 23 | +351,254 | +0.00541 | +0.000630 | YES |