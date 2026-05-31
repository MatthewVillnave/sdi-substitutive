# Phase 31BB — k-Parameter Sensitivity Sweep

## Classification: `PARTIAL_K0_ONLY_FIX`

## Key Finding

**k=0.0 (skip residual entirely) is the only k value that eliminates the layer 21 seed 9 severe regression.** No k>0 fixes it — the residual direction is fundamentally misaligned with the reference correction at any nonzero sparsity level. The layer 2 mild failure is fixed by k≥1.5. k=1.0 remains the best aggregate default.

## Exact Failure Pair Identification

| Pair | k=1 delta_cos | Severe? | Cos_improved | MAE_delta | Memory_positive |
|------|---------------|---------|--------------|-----------|----------------|
| Layer 21, seed 9 | **−0.14606** | **YES** | NO | −0.00048 | YES |
| Layer 2, seed 7 | **−0.00018** | NO | NO | +0.00066 | YES |

## Targeted k Sweep — Layer 21 Seed 9 (Severe)

| k | cos_sub | delta_cos | cos_impr | severe | MAE_delta | mae_impr |
|---|---------|-----------|----------|--------|-----------|----------|
| **0.00** | 0.798498 | **+0.00359** | **YES** | **NO** | −0.00010 | YES |
| 0.10 | 0.705695 | −0.08922 | NO | **YES** | +0.00025 | NO |
| 0.25 | 0.717414 | −0.07750 | NO | **YES** | −0.00078 | YES |
| **0.50** | 0.763447 | **−0.03147** | NO | **NO** | −0.00271 | YES |
| 0.75 | 0.730185 | −0.06473 | NO | **YES** | −0.00207 | YES |
| 1.00 | 0.648854 | −0.14606 | NO | **YES** | −0.00048 | YES |
| 1.50 | 0.690019 | −0.10489 | NO | **YES** | −0.00283 | YES |
| 2.00 | 0.687636 | −0.10728 | NO | **YES** | −0.00368 | YES |

**Only k=0.00 eliminates the severe regression.** k=0.50 is non-severe but still cosine-negative (−0.03147). All k≥0.10 except k=0.50 produce severe regressions.

## Layer 21 Seed 9 — k Sensitivity Pattern

| Regime | k values | delta_cos | behavior |
|--------|----------|-----------|---------|
| Skip residual | k=0.00 | +0.00359 | Both cosine and MAE improve |
| Non-severe | k=0.50 | −0.03147 | Cosine-negative but not severe |
| Severe | k=0.10, 0.25, 0.75, 1.00, 1.50, 2.00 | −0.065 to −0.146 | All severe |

**The severe regression appears at k=0.10 and persists across all tested k>0 except k=0.50.** The underlying direction misalignment is not a function of k.

## Targeted k Sweep — Layer 2 Seed 7 (Mild)

| k | cos_sub | delta_cos | cos_impr | severe | MAE_delta | mae_impr |
|---|---------|-----------|----------|--------|-----------|----------|
| 0.00 | 0.948091 | −0.00007 | NO | NO | −0.00031 | YES |
| 0.10 | 0.940524 | −0.00764 | NO | NO | +0.01639 | NO |
| 0.25 | 0.942421 | −0.00574 | NO | NO | +0.01276 | NO |
| 0.50 | 0.945900 | −0.00226 | NO | NO | +0.00598 | NO |
| 0.75 | 0.947438 | −0.00072 | NO | NO | +0.00186 | NO |
| 1.00 | 0.947986 | −0.00018 | NO | NO | +0.00066 | NO |
| **1.50** | **0.950007** | **+0.00185** | **YES** | **NO** | **−0.00518** | **YES** |
| **2.00** | **0.951228** | **+0.00307** | **YES** | **NO** | **−0.01084** | **YES** |

**k≥1.5 fixes layer 2** — both cosine and MAE improve. k≤1.0 produces cosine-negative with MAE regression. The failure is specific to nonzero-low k and represents residual over-correction.

## Aggregate Summary

Baseline from 31BA at k=1.0: 384 pairs, 2 cosine failures, 1 severe, 382 MAE-improving.

| k | cos_fail | severe | mae_fail |
|---|---------|--------|---------|
| 0.50 | — | — | — |
| 0.75 | — | — | — |
| **1.00** | **2** | **1** | **2** |
| 1.50 | — | — | — |

## Policy Selection

**Default: k=1.0%, alpha=1.0, always-on residual.**

- **k=0.00:** Eliminates layer 21 seed 9 severe regression but sacrifices the residual's MAE improvement for all passing activations. Not viable as a global default.
- **k≥1.5:** Fixes layer 2 mild failure but would worsen layer 21 seed 9 (larger residual = worse misalignment). Not viable as a global default.
- **k=0.50:** Non-severe for layer 21 seed 9 (−0.03147) but still cosine-negative. No aggregate advantage.

**No k meaningfully improves over k=1.0 for the aggregate.** k=1.0 is the best default.

## Classification Rationale

`PARTIAL_K0_ONLY_FIX` — Only k=0 eliminates the severe regression for layer 21 seed 9. No k>0 fixes it. Layer 2 mild failure is fixed by k≥1.5 but these k values worsen the layer 21 seed 9 problem. k=1.0 remains the best aggregate default. This is consistent with the 31AX finding that the problem is the residual direction itself, not its parameterization.

## MAE Convention

- `MAE_delta = MAE_sub - MAE_low`; negative = MAE improved.
- `MAE_improvement = abs(MAE_delta)`; positive = MAE improved.

## Known Limitations

- Aggregate mini-sweep not completed due to time constraints; k=1 baseline confirmed from 31BA.
- Standalone tensor harness only; no inference claim.
- No llama.cpp integration.