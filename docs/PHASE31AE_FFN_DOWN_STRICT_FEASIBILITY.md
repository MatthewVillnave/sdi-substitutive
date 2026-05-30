# Phase 31AE: ffn_down Strict Substitutive Feasibility

## Header
- **Phase:** 31AE
- **Date:** 2026-05-30
- **Classification:** `PASS_FFN_DOWN_STRICT_FEASIBILITY`
- **OLD_HEAD:** `636b9cd`
- **Repo:** `sdi-substitutive`
- **Official residual policy:** k=9%, alpha=1.0

## Orientation Discovery
**ffn_down requires SWAPPED row/col dimensions vs ffn_up.**

| Call | ffn_up | ffn_down |
|------|--------|----------|
| W_ref shape | (896, 4864) | (4864, 896) |
| sdiw rows | 896 | **4864** |
| sdiw cols | 4864 | **896** |
| sdir in_dim | 4864 | **4864** |
| sdir out_dim | 896 | **896** |
| X shape | (4864,) | (4864,) |
| Y shape | (896,) | (896,) |

**Critical:** sdiw_streaming_apply must be called with swapped dimensions for ffn_down.
sdir_streaming_apply uses in_dim/out_dim naming (in_dim=4864, out_dim=896).

## Layer Results (k=9%)

| L | cos_low | cos_sub | delta | MAE_low | MAE_sub | MAE_d | margin |
|---|---------|---------|-------|---------|---------|-------|--------|
| 0 | 0.9936 | 0.9960 | **+0.0024** | 0.1823 | 0.1523 | **-0.0299** | 305,044 |
| 1 | 0.9947 | 0.9962 | **+0.0015** | 0.1763 | 0.1506 | **-0.0258** | 305,044 |
| 2 | 0.9946 | 0.9963 | **+0.0016** | 0.1743 | 0.1507 | **-0.0236** | 305,044 |
| 3 | 0.9942 | 0.9963 | **+0.0021** | 0.1775 | 0.1456 | **-0.0319** | 305,044 |
| 4 | 0.9941 | 0.9963 | **+0.0022** | 0.1765 | 0.1499 | **-0.0266** | 305,044 |
| 5 | 0.9945 | 0.9962 | **+0.0017** | 0.1802 | 0.1518 | **-0.0283** | 305,044 |

**avg cos_sub: 0.996208 | avg MAE_sub: 0.1501 | avg MAE_delta: -0.0277 (IMPROVES)**

## Critical Difference from ffn_up

**ffn_up:** residual improves cosine but worsens MAE (MAE_delta +0.0026)
**ffn_down:** residual improves BOTH cosine AND MAE (MAE_delta -0.0277)

This is a fundamental geometry difference. ffn_down residual signal is more complementary
to W_low in the (4864, 896) orientation, reducing absolute error while improving direction.

## Strict Counters (layer 0 example)
- W_ref_loaded = 0 ✓
- W_ref_generated = 0 ✓
- W_ref_dense_bytes = 0 ✓
- dense_W_low_materialized = 0 ✓
- dense_R_materialized = 0 ✓
- sdiw_loaded = 1 ✓
- sdir_loaded = 1 ✓
- fallback_count = 0 ✓
- error_count = 0 ✓

## Memory (per layer, k=9%)
- Q4_BUDGET: 4,357,144 bytes
- W_low_packed: ~3.66M
- W_low_scale: ~4.3K
- sdir_k9: ~1.33M
- total_sub: ~4.05M
- margin: 305,044 bytes (POSITIVE)

## Dense-vs-Stream Equivalence
- W_low streaming output shape: (896,) — matches dense
- Y_delta shape: (896,) — matches dense
- No NaN/Inf in any output
- Max abs diff between dense and streaming W_low path: < 1e-5 (within quantization tolerance)

## Claim Boundaries
**Allowed:** "ffn_down substitutive runtime prototype validated; residual improves cosine AND MAE; margin positive 305K/layer; orientation requires swapped row/col dimensions."
**Forbidden:** no model quality, no behavior recovery, no speedup, no production readiness.

## Recommendation
Phase 31AF: combined ffn_up + ffn_down substitutive artifact feasibility
