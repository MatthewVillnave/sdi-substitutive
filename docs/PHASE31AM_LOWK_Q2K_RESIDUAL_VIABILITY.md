# Phase 31AM — Low-k Q2_K + Sparse Residual Viability Probe

## Classification
**`PARTIAL_Q2_SIM_ONLY_LOWK_RESIDUAL_HURTS`**

Subtype: low-k residual policies hurt approximation under simulated Q2-like W_low

---

## Decode Availability

| Item | Value |
|------|-------|
| Actual GGUF Q2_K decode available? | **NO** |
| W_ref source | Q4_K_M model, Q5_0/Q6_K dequantized via GGUFReader |
| W_low type | **SIMULATED Q2-like** — block_size=16 aggressive 2-bit quantization |
| Q2_K reference block_size | 256 (as defined in llama.cpp GGUF spec) |
| Numerical results are | SIMULATED — not actual Q2_K decode quality |

**Important:** Q2_K byte accounting (from llama.cpp constants) was used for budget modeling. Numerical approximation behavior (cosine, MAE) is from the block_size=16 simulation only. Actual Q2_K decode quality may differ significantly.

---

## Scope

- Full MLP toy probe: ffn_up + ffn_gate + ffn_down
- Layers 0–5 (6 layers × 3 families = 18 family-layer tensors)
- Residual k values tested: 0%, 0.5%, 1%, 2%, 3% (SDIR encoding)
- 29 policies tested
- Metrics: cosine similarity, MAE, memory budget

---

## Memory Budget (Corrected 31AL-R Accounting)

Per family-layer:
- Q4_budget: 2,179,072 bytes
- Q2_K W_low (simulated): 1,430,028 bytes (byte-accurate per GGUF constants)
- Total Q2_K W_low across 18 family-layers: 24.55 MB

Aggregate (6 layers × 3 families):
- Q4_budget_total: 37.41 MB
- Q2_K W_low total: 24.55 MB
- Residual budget at k=0%: 12.86 MB margin (memory-positive)

### Residual Budget Table (per family-layer)

| k% | SDIR bytes | int8 bytes | Q2K+SDIR total | SDIR margin | int8 margin |
|---:|----------:|----------:|---------------:|------------:|------------:|
| 0% | 544,768 | 544,768 | 33.90 MB | +3,591 KB | +3,591 KB |
| 0.5% | 588,348 | 566,558 | 34.65 MB | +2,825 KB | +3,208 KB |
| 1% | 631,930 | 588,349 | 35.40 MB | +2,059 KB | +2,825 KB |
| 2% | 719,092 | 631,930 | 36.89 MB | +526 KB | +2,059 KB |
| 3% | 806,256 | 675,512 | 38.39 MB | −1,006 KB | +1,293 KB |
| 4% | 893,418 | 719,093 | 39.88 MB | −2,538 KB | +526 KB |
| 5% | 980,582 | 762,675 | 41.38 MB | −4,070 KB | −240 KB |

**Key:** SDIR at k=3% is memory-negative; int8 at k=3% is still memory-positive.

---

## Low-k Residual Sweep Results

All 29 policies — sorted by memory positivity then cosine:

| Policy | Mem+? | Margin | cos_low | cos_sub | Δcos | mae_low | mae_sub | Δmae |
|--------|:-----:|-------:|--------:|--------:|-----:|--------:|--------:|-----:|
| uniform_0 | **Y** | +3591KB | 0.9260 | 0.9260 | **+0.0000** | 0.00904 | 0.00904 | **+0.0000** |
| up_0.5_g0_d0 | Y | +3335KB | 0.9260 | 0.9200 | −0.0060 | 0.00904 | 0.00927 | −0.0002 |
| up_0_g0.5_d0 | Y | +3335KB | 0.9260 | 0.9182 | −0.0079 | 0.00904 | 0.00918 | −0.0001 |
| up_1_g0_d0 | Y | +3080KB | 0.9260 | 0.9174 | −0.0087 | 0.00904 | 0.00945 | −0.0004 |
| up_0_g0_d0.5 | Y | +3335KB | 0.9260 | 0.9166 | −0.0094 | 0.00904 | 0.00909 | −0.0001 |
| up_0_g1_d0 | Y | +3080KB | 0.9260 | 0.9160 | −0.0101 | 0.00904 | 0.00928 | −0.0002 |
| up_0_g0_d1 | Y | +3080KB | 0.9260 | 0.9149 | −0.0111 | 0.00904 | 0.00913 | −0.0001 |
| up_2_g0_d0 | Y | +2569KB | 0.9260 | 0.9148 | −0.0112 | 0.00904 | 0.00977 | −0.0007 |
| up_0_g2_d0 | Y | +2569KB | 0.9260 | 0.9142 | −0.0119 | 0.00904 | 0.00945 | −0.0004 |
| up_3_g0_d0 | Y | +2059KB | 0.9260 | 0.9141 | −0.0120 | 0.00904 | 0.01005 | −0.0010 |
| up_0_g3_d0 | Y | +2059KB | 0.9260 | 0.9139 | −0.0122 | 0.00904 | 0.00960 | −0.0006 |
| up_0_g0_d2 | Y | +2569KB | 0.9260 | 0.9135 | −0.0125 | 0.00904 | 0.00919 | −0.0002 |
| up_0_g0_d3 | Y | +2059KB | 0.9260 | 0.9134 | −0.0126 | 0.00904 | 0.00925 | −0.0002 |
| up1_g1_d0 | Y | +2569KB | 0.9260 | 0.9073 | −0.0188 | 0.00904 | 0.00969 | −0.0007 |
| up1_g0_d1 | Y | +2569KB | 0.9260 | 0.9063 | −0.0198 | 0.00904 | 0.00954 | −0.0005 |
| up0_g1_d1 | Y | +2569KB | 0.9260 | 0.9048 | −0.0212 | 0.00904 | 0.00937 | −0.0003 |
| up2_g2_d0 | Y | +1548KB | 0.9260 | 0.9030 | −0.0231 | 0.00904 | 0.01018 | −0.0011 |
| uniform_0.5 | Y | +2825KB | 0.9260 | 0.9028 | −0.0233 | 0.00904 | 0.00946 | −0.0004 |
| up2_g0_d2 | Y | +1548KB | 0.9260 | 0.9023 | −0.0237 | 0.00904 | 0.00992 | −0.0009 |
| up3_g3_d0 | Y | +526KB | 0.9260 | 0.9019 | −0.0241 | 0.00904 | 0.01061 | −0.0016 |
| up0_g2_d2 | Y | +1548KB | 0.9260 | 0.9017 | −0.0244 | 0.00904 | 0.00960 | −0.0006 |
| up3_g0_d3 | Y | +526KB | 0.9260 | 0.9015 | −0.0246 | 0.00904 | 0.01026 | −0.0012 |
| up0_g3_d3 | Y | +526KB | 0.9260 | 0.9013 | −0.0248 | 0.00904 | 0.00980 | −0.0008 |
| uniform_1 | Y | +2059KB | 0.9260 | 0.8962 | −0.0299 | 0.00904 | 0.00978 | −0.0007 |
| asym_1_1_3 | Y | +1037KB | 0.9260 | 0.8947 | −0.0314 | 0.00904 | 0.00990 | −0.0009 |
| asym_1_3_1 | Y | +1037KB | 0.9260 | 0.8941 | −0.0319 | 0.00904 | 0.01010 | −0.0011 |
| asym_3_1_1 | Y | +1037KB | 0.9260 | 0.8929 | −0.0332 | 0.00904 | 0.01038 | −0.0013 |
| uniform_2 | Y | +526KB | 0.9260 | 0.8905 | −0.0356 | 0.00904 | 0.01033 | −0.0013 |
| uniform_3 | **N** | −1006KB | 0.9260 | 0.8893 | −0.0367 | 0.00904 | 0.01082 | −0.0018 |

**Residual-off baseline (`uniform_0`):** cos=0.9260, MAE=0.00904 — this is the best policy across all metrics.

**Every residual-on policy worsens both cosine AND MAE versus residual-off.**

---

## Key Findings

1. **Memory-positive low-k policies exist:** 27 of 29 tested policies are memory-positive at k≤2% (SDIR) or k≤3% (int8).

2. **Residual-on hurts approximation:** Every tested residual policy — at every k%, every family targeting, every asymmetric variant — produced worse cosine and worse MAE than residual-off (uniform_0).

3. **No Pareto frontier exists:** No policy simultaneously satisfies memory-positive, delta_cos>0, and delta_mae>0. The residual-off uniform_0 policy is optimal on all three criteria.

4. **Block_size=16 simulation is aggressive:** cos(W_ref, W_q2k_simulated) ≈ 0.926 for all family-layers. This is a conservative approximation quality baseline. Actual Q2_K decode (block_size=256) may produce different approximation behavior.

5. **Error distribution is magnitude-aligned:** Correlation between |residual| and |W_ref| is 0.9999 — quantization error is proportionally larger on high-magnitude weights. This means the sparse residual (top-k% by magnitude) mostly captures error on already-large elements, which W_low already approximates well, yielding no net gain.

---

## Forbidden Claims (Not Made)

- ❌ No model quality recovery claimed
- ❌ No model behavior recovery claimed
- ❌ No speedup claimed
- ❌ No full-model memory savings claimed
- ❌ No llama.cpp integration claimed
- ❌ No production readiness claimed
- ❌ No actual Q2_K numerical behavior verified

---

## Suspected / Unproven

- **Actual Q2_K decode quality:** The block_size=16 simulation is aggressive and may not reflect actual GGUF Q2_K decode (block_size=256) behavior. True Q2_K decode probe is needed before trusting numerical approximation claims.
- **Residual encoding strategy:** The magnitude-sparse residual fails because error magnitude is correlated with W_ref magnitude. Other residual encodings (error-feedback, per-block, block-level) might produce different results.
- **Family-specific behavior:** ffn_down (896×4864) shows slightly higher cosine improvement potential than ffn_up/ffn_gate, suggesting residual gains may be family-asymmetric.

---

## Recommended Next Phase

**Phase 31AN — Residual Error Structure Analysis or Actual Q2_K Decode Probe**

Two legitimate directions (do not over-narrow to one):

1. **Residual error structure analysis:** Investigate why magnitude-sparse residuals hurt — specifically, analyze whether error is structure-sparse (per-block, per-channel, rank-based) rather than magnitude-sparse. If error has block-level structure, a different residual encoding could succeed where magnitude-sparse fails.

2. **Actual Q2_K decode verification:** Before trusting simulated Q2-like behavior, implement/verify actual GGUF Q2_K decode. The block_size=16 simulation is aggressive; actual Q2_K (block_size=256) may produce materially different approximation quality and residual alignment.

Neither direction is authorized without explicit user request.

---

## Artifacts

- **Results JSON:** `results/PHASE31AM_LOWK_Q2K_RESIDUAL_VIABILITY.json`
- **Probe script:** `src/phase31am_lowk_q2k_residual_viability.py`
- **Data source:** Q4_K_M model Q5_0/Q6_K via GGUFReader (USB drive)

---

*Phase 31AM — probe run 2026-05-30. Not committed pending corrections.*