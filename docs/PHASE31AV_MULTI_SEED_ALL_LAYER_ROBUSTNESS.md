# Phase 31AV — Multi-Seed All-Layer Robustness

## Classification: `PARTIAL_MULTI_SEED_COSINE_SENSITIVE`


## Key Finding

Layer 21 is the sole sensitive layer at **seed=9 only**. At seed=0 and seed=5, all 24 layers pass all gates. MAE is robust across all seeds and all layers. Cosine has one isolated regression at layer 21 / seed=9.


## Seeds Tested: 0, 5, 9 — k=1%


### By-Seed Aggregate


| Seed | mem_pos | cos_pos | MAE_imp | agg_margin | worst_cos_layer | delta_cos | all_pass |

|------|---------|---------|---------|-----------|-----------------|-----------|---------|

| 0 | 24/24 | 24/24 | 24/24 | +8,428,606 | layer9 | +0.00553 | **YES** |

| 5 | 24/24 | 24/24 | 24/24 | +8,428,606 | layer4 | +0.00361 | **YES** |

| 9 | 24/24 | 23/24 | 24/24 | +8,428,606 | **layer21** | **−0.14606** | NO |


**2/3 seeds fully pass. Seed=9 has one cosine regression at layer 21.**


### By-Layer Aggregate


| Layer | cos_pos/3 | mae_pos/3 | mean_delta_cos | min | max | status |

|-------|-----------|-----------|----------------|-----|-----|--------|

|  0 | 3/3 | 3/3 | +0.01125 | +0.01071 | +0.01156 | ROBUST |
|  1 | 3/3 | 3/3 | +0.00751 | +0.00648 | +0.00865 | ROBUST |
|  2 | 3/3 | 3/3 | +0.00822 | +0.00419 | +0.01083 | ROBUST |
|  3 | 3/3 | 3/3 | +0.01033 | +0.00483 | +0.01461 | ROBUST |
|  4 | 3/3 | 3/3 | +0.00540 | +0.00361 | +0.00664 | ROBUST |
|  5 | 3/3 | 3/3 | +0.01596 | +0.01167 | +0.01834 | ROBUST |
|  6 | 3/3 | 3/3 | +0.01365 | +0.00808 | +0.01929 | ROBUST |
|  7 | 3/3 | 3/3 | +0.01448 | +0.01239 | +0.01782 | ROBUST |
|  8 | 3/3 | 3/3 | +0.01549 | +0.01238 | +0.02088 | ROBUST |
|  9 | 3/3 | 3/3 | +0.01354 | +0.00553 | +0.02094 | ROBUST |
| 10 | 3/3 | 3/3 | +0.01293 | +0.00806 | +0.02207 | ROBUST |
| 11 | 3/3 | 3/3 | +0.02020 | +0.00869 | +0.03229 | ROBUST |
| 12 | 3/3 | 3/3 | +0.01230 | +0.00496 | +0.01685 | ROBUST |
| 13 | 3/3 | 3/3 | +0.01480 | +0.01165 | +0.02041 | ROBUST |
| 14 | 3/3 | 3/3 | +0.01266 | +0.00719 | +0.01870 | ROBUST |
| 15 | 3/3 | 3/3 | +0.01233 | +0.00903 | +0.01734 | ROBUST |
| 16 | 3/3 | 3/3 | +0.01486 | +0.01282 | +0.01732 | ROBUST |
| 17 | 3/3 | 3/3 | +0.01159 | +0.01042 | +0.01272 | ROBUST |
| 18 | 3/3 | 3/3 | +0.00966 | +0.00687 | +0.01170 | ROBUST |
| 19 | 3/3 | 3/3 | +0.01195 | +0.00641 | +0.01786 | ROBUST |
| 20 | 3/3 | 3/3 | +0.01389 | +0.00802 | +0.01867 | ROBUST |
| 21 | 2/3 | 3/3 | -0.03847 | -0.14606 | +0.01784 | **SENSITIVE** ← layer 21, seed=9 cosine regresses −0.14606 |
| 22 | 3/3 | 3/3 | +0.01012 | +0.00879 | +0.01114 | ROBUST |
| 23 | 3/3 | 3/3 | +0.01349 | +0.00995 | +0.01680 | ROBUST |

### Layer 21 Detail (SENSITIVE)


| Seed | delta_cos | MAE_delta | margin | status |

|------|-----------|-----------|--------|--------|

| 0 | +0.01784 | −0.003981 | +351,192 | cosine improves |

| 5 | +0.01259 | −0.003981 | +351,192 | cosine improves |

| **9** | **−0.14606** | **−0.003981** | **+351,192** | **cosine REGRESSES severely** |


Layer 21 cosine regresses at seed=9 only. MAE improves at all seeds. Margin is consistent across all seeds.


## MAE Convention

- `MAE_delta = MAE_sub - MAE_low`; negative = MAE improved

- `MAE_improvement = abs(MAE_delta)`; positive = MAE improved

- **All 24 layers improve MAE at all 3 seeds** — MAE is fully robust


## Classification

`PARTIAL_MULTI_SEED_COSINE_SENSITIVE`

- 2/3 seeds (0, 5): all 24 layers pass all gates

- 1/3 seeds (9): layer 21 cosine regresses severely (−0.14606) but MAE still improves

- Layer 21 is activation-sensitive at specific seeds; not a universal failure

- MAE is fully robust: 24/24 × 3/3

- Memory is fully robust: 24/24 × 3/3


## Known Limitations

- Only 3 seeds tested (0, 5, 9); broader seed space not characterized

- Results apply only to k=1%; k=2% robustness not tested in this run

- Standalone tensor harness only

- No llama.cpp integration


## Comparison to 31AT

- 31AT diagnosed layer 21 sensitivity at seed=9 (1/10 seeds regress)

- 31AV confirms: seed=0 and seed=5 pass; seed=9 fails at layer 21

- This is consistent: layer 21 is seed-sensitive, not universally failing
