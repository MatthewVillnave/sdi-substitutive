# Phase 31AX — Activation-Space / Norm-Regularized Residual Probe

## Classification: `PARTIAL_NO_FIX_METRIC_CONFLICT_CONFIRMED`

## Key Finding

**The cosine regression is fundamentally a direction mismatch, not a magnitude problem.** Scaling Y_sub to match reference or low-Precision norms changes MAE but NOT cosine (cosine is scale-invariant under positive scaling). The oracle projection — projecting the residual correction direction onto the reference correction direction — also fails. This confirms the metric conflict is deep: the residual update direction itself is misaligned with the reference correction for this specific activation.

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
| norm_ref | 5.870598 |
| norm_low | 6.125916 |
| norm_sub | 6.580669 |
| memory_positive | YES |
| cosine_improved | **NO** |
| MAE_improved | YES |

## Critical Insight: Cosine is Scale-Invariant

**cos(a*X, b*Y) = sign(ab) * cos(X, Y)** for any positive scalars a, b.

This means:
- Scaling Y_sub by any factor (to match ||Y_ref|| or ||Y_low||) changes MAE but NOT cosine
- Norm correction alone cannot fix the cosine regression
- The problem is directional, not magnitude-based

## Norm Matching Results

**Scale to reference norm:** Y_norm_ref = Y_sub * (||Y_ref|| / ||Y_sub||) = Y_sub * 0.8921

| Metric | Y_low | Y_sub | Y_norm_ref |
|--------|-------|-------|------------|
| cos | 0.794913 | 0.648854 | **0.648854** |
| delta_cos | — | −0.146059 | **−0.146059** |
| MAE | 0.079634 | 0.079156 | 0.073292 |
| MAE_delta | — | −0.000479 | **−0.006342** |

**Norm matching improves MAE significantly (−0.006342) but cosine is completely unchanged.**

## Output Interpolation Results

Y_interp = Y_low + beta * (Y_sub - Y_low)

| beta | cos | delta_cos | MAE | MAE_delta | norm | cos_pos | MAE_pos |
|------|-----|-----------|-----|-----------|------|---------|---------|
| 0.00 | 0.794913 | +0.000000 | 0.079634 | +0.000000 | 6.1259 | NO | NO |
| 0.10 | 0.783603 | −0.011310 | 0.079215 | −0.000420 | 6.1378 | NO | YES |
| 0.25 | 0.764910 | −0.030003 | 0.078719 | −0.000915 | 6.1702 | NO | YES |
| 0.50 | 0.729725 | −0.065189 | 0.078361 | −0.001273 | 6.2623 | NO | YES |
| 0.75 | 0.690591 | −0.104323 | 0.078560 | −0.001074 | 6.4001 | NO | YES |
| 1.00 | 0.648854 | −0.146059 | 0.079156 | −0.000479 | 6.5807 | NO | YES |

**No beta simultaneously improves cosine AND MAE.** All beta > 0 regress cosine. beta=0 gives Y_low (delta_cos=0, MAE_delta=0). The Pareto frontier runs from cosine-optimal (beta=0) to MAE-optimal (beta~0.5).

## Oracle Projection Results (DIAGNOSTIC — uses Y_ref, NOT runtime available)

D = Y_sub - Y_low (residual update direction)
T = Y_ref - Y_low (reference correction direction)
D_proj = proj_T(D) (projection of residual direction onto reference direction)
Y_proj = Y_low + D_proj

| Metric | Y_low | Y_sub | Y_proj |
|--------|-------|-------|--------|
| cos | 0.794913 | 0.648854 | 0.697644 |
| delta_cos | — | −0.146059 | −0.097270 |
| MAE | 0.079634 | 0.079156 | 0.101186 |
| MAE_delta | — | −0.000479 | +0.021552 |

**Even projecting the residual onto the reference correction direction does NOT fix cosine (still −0.097) and MAE worsens significantly (+0.021552).** The residual correction direction is fundamentally misaligned with the reference correction direction for this activation.

### Direction-Scaled Oracle

Y_dir = Y_low + gamma * D_proj

| gamma | cos | delta_cos | MAE | MAE_delta | norm | cos_pos | MAE_pos |
|-------|-----|-----------|-----|-----------|------|---------|---------|
| 0.50 | 0.746504 | −0.048410 | 0.090410 | +0.010776 | 6.3417 | NO | NO |
| 1.00 | 0.697644 | −0.097270 | 0.101186 | +0.021552 | 6.5917 | NO | NO |
| 1.50 | 0.649464 | −0.145450 | 0.111962 | +0.032327 | 6.8721 | NO | NO |

**No gamma makes both cosine and MAE positive.** The oracle approach fails entirely.

## Transfer to Safe Cases

### Norm-to-ref scaling on safe seeds

| Case | cos | cos_low | delta_cos | MAE | MAE_delta | cos_pos | MAE_pos |
|------|-----|---------|-----------|-----|-----------|---------|---------|
| layer21 seed=0 | 0.873457 | 0.855612 | +0.017845 | 0.052340 | −0.006755 | YES | YES |
| layer21 seed=5 | 0.888971 | 0.876169 | +0.012802 | 0.057834 | −0.007191 | YES | YES |

**Norm scaling PRESERVES passing behavior on safe seeds.**

### Oracle projection (gamma=1.0) on safe seeds

| Case | cos | cos_low | delta_cos | MAE | MAE_delta | cos_pos | MAE_pos |
|------|-----|---------|-----------|-----|-----------|---------|---------|
| layer21 seed=0 oracle | 0.884147 | 0.855612 | +0.028536 | 0.052013 | −0.007082 | YES | YES |
| layer21 seed=5 oracle | 0.902767 | 0.876169 | +0.026598 | 0.056793 | −0.008232 | YES | YES |

**Oracle projection ALSO WORKS on safe seeds** — seeds 0 and 5 get even better cosine improvement with the oracle than the baseline residual! This means the oracle is not inherently broken — it works correctly when the residual direction is aligned with the reference. The problem is seed=9/layer=21 specifically.

## Summary of All Fix Attempts

| Method | Cosine Fixed? | MAE Improved? | Runtime Available? |
|--------|-------------|---------------|-------------------|
| Alpha scaling (0.0-1.5) | NO | YES (alpha≤1.0) | YES |
| k sweep (0-2%) | NO | YES | YES |
| Norm matching | **NO** (scale-invariant) | YES | YES (if runtime norm known) |
| Output interpolation | NO | YES | YES |
| Oracle projection | NO | NO | **NO** (needs Y_ref) |
| Family ablation | NO | YES | YES |

## Root Cause Analysis

The residual direction D = Y_sub - Y_low is misaligned with the reference correction direction T = Y_ref - Y_low for seed=9/layer=21 specifically. The encode_sdir residual:
- Works correctly for seeds 0, 5 (and 9 other seeds in 31AV's 10-seed sweep)
- Is fundamentally misaligned for seed=9/layer=21

This is a **structural property of the activation at that specific seed/layer combination**, not a parameter or scale issue. The misalignment cannot be corrected by:
- Scaling (cosine is scale-invariant)
- Interpolation (the entire Pareto frontier is on the wrong side)
- Oracle projection (even the ideal direction fails because it still produces the wrong output)

The only path forward would be:
1. **Runtime detection**: Detect when the residual direction is misaligned (requires access to Y_ref — not available at runtime in actual inference)
2. **Alternative residual encoding**: A different residual formulation that doesn't suffer from this direction mismatch
3. **Layer-skip or passthrough**: For the specific case of seed=9/layer=21, use Y_low directly (skip the residual) since MAE improvement from residual is tiny (−0.000479) compared to the cosine regression (−0.146059)

## Classification

`PARTIAL_NO_FIX_METRIC_CONFLICT_CONFIRMED`

- No norm/interpolation/projection approach fixes both cosine and MAE
- Cosine is scale-invariant — norm matching cannot work
- Oracle projection fails because residual direction is fundamentally misaligned
- Safe seeds (0, 5) work correctly with oracle — the issue is activation-specific
- The encode_sdir residual approach is not universally broken — it fails specifically for seed=9/layer=21

## Comparison to 31AW

- 31AW found no alpha/k/family-subset policy fixes cosine
- 31AX extends this: no output-space operation (norm, interpolation, projection) can fix cosine
- The problem is the residual direction itself, not how it's scaled or applied

## MAE Convention

- `MAE_delta = MAE_sub - MAE_low`; negative = MAE improved.
- `MAE_improvement = abs(MAE_delta)`; positive = MAE improved.

## Known Limitations

- Oracle projection uses Y_ref — not runtime available.
- Activation-specific finding; broader seed space not tested.
- Standalone tensor harness only; no inference claim.
- No llama.cpp integration.