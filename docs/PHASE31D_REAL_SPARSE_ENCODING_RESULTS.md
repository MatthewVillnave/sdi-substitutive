# Phase 31D: Real Sparse Residual Encoding Policy

## Classification: `PASS_REAL_ENCODING_MEMORY_VIABLE`

---

## Executive Summary

**Phase 31C was corrected.** 31C claimed "memory viable" for topk_10% by only counting residual VALUE bytes (~1.74MB), completely ignoring position/index/metadata overhead. With full storage accounting (values + metadata):

- **fp32 values at 10% nnz: OVER budget** (2,288,024 bytes > 1,906,688 budget for ffn_up)
- **fp16 values at 10% nnz + dense bitmap: VIABLE** (1,416,396 bytes)

**Multiple sparse encodings ARE memory-viable** under real accounting and improve quality over W_low.

**Low-rank SVD is memory-viable but ineffective** — the residual has a flat singular value spectrum (r=32 explains only 6.85% of residual variance), making low-rank decomposition a poor fit.

---

## Key Findings

### 1. Residual Structure: Near-Gaussian Noise

The FFN residual matrices have a **flat singular value spectrum**:

| Rank r | Explained Variance (ffn_up) |
|--------|---------------------------|
| 1      | 0.23%                     |
| 4      | 0.90%                     |
| 8      | 1.78%                     |
| 16     | 3.51%                     |
| 32     | 6.85%                     |

This means the residual is **structurally near-Gaussian noise**. Low-rank approximation captures minimal energy. Sparse coding (capturing individual large elements) is the correct approach.

### 2. Phase 31C Correction

| Aspect | Phase 31C | Phase 31D (corrected) |
|--------|-----------|----------------------|
| ffn_up 10% nnz storage | 1,743,256 bytes (values only) | 2,288,024 bytes (fp32) or 1,416,396 bytes (fp16+bitmap) |
| ffn_up 10% fp32 viable? | Claimed YES | **NO** — exceeds budget |
| ffn_up 10% fp16 viable? | Not assessed | **YES** — 1,416,396 < 1,906,688 |

### 3. W_low Baseline

| Tensor | Cosine vs W_ref | MAE |
|--------|----------------|-----|
| ffn_up | 0.993091 | 0.001474 |
| ffn_down | 0.992919 | 0.001250 |

---

## Viable Sparse Encoding Candidates

### Dense Bitmap + Global Top-K (fp16 values)

Best encoding for ffn_up: **dense bitmap + k=15% fp16**

| k_pct | nnz | Real Bytes | Cosine (ffn_up) | Delta vs W_low | Viable? |
|-------|-----|-----------|----------------|----------------|---------|
| 5% | 217,907 | 980,582 | 0.994072 | +0.000981 | ✅ |
| 7% | 305,070 | 1,154,908 | 0.994435 | +0.001344 | ✅ |
| 10% | 435,814 | 1,416,396 | 0.994945 | +0.001855 | ✅ |
| 15% | 653,721 | 1,852,210 | 0.995714 | +0.002624 | ✅ |
| 20% | 871,628 | 2,288,024 | 0.996396 | +0.003305 | ❌ (over) |

Storage formula: `bitmap (544,768 bytes) + nnz × 2 (fp16 values)`

### Row-wise Top-K + uint16 Indices + fp16 Values

| k_pct | nnz | Real Bytes | Cosine (ffn_up) | Delta vs W_low | Viable? |
|-------|-----|-----------|----------------|----------------|---------|
| 2% | 82,688 | 340,480 | 0.993470 | +0.000379 | ✅ |
| 5% | 214,016 | 865,792 | 0.994042 | +0.000951 | ✅ |
| 8% | 345,344 | 1,391,104 | 0.994578 | +0.001487 | ✅ |
| 10% | 432,896 | 1,741,312 | 0.994823 | +0.001732 | ✅ |

Storage formula: `d_out × 2 + nnz × 2 (indices) + nnz × 2 (fp16 values)`

### Row-wise Top-K + uint16 Indices + int8 Values

| k_pct | nnz | Real Bytes | Cosine (ffn_up) | Delta vs W_low | Viable? |
|-------|-----|-----------|----------------|----------------|---------|
| 10% | 432,896 | 1,318,144 | 0.994862 | +0.001771 | ✅ |

Storage formula: `d_out × 2 + nnz × 2 (indices) + nnz × 1 (int8) + d_out × 2 (row scales)`

### Low-Rank SVD (Memory-Viable but Quality-Poor)

| Rank | Bytes | Explained Var | Cosine Delta | Viable? |
|------|-------|--------------|--------------|---------|
| r=8 | 184,352 | 1.78% | minimal | ✅ |
| r=16 | 368,704 | 3.51% | minimal | ✅ |
| r=32 | 737,408 | 6.85% | minimal | ✅ |

**Low-rank does not meaningfully improve over W_low** — the flat spectrum means decomposing into rank-r captures <7% of residual energy even at r=32.

---

## Best Viable Candidate

**Encoding:** Dense Bitmap + Global Top-15% + fp16 values

| Metric | ffn_up | ffn_down |
|--------|--------|----------|
| Real bytes | 1,852,210 | 1,852,210 |
| Budget | 1,906,688 | 2,485,504 |
| Headroom | 54,478 | 633,294 |
| Final cosine | 0.995714 | 0.995632 |
| Delta vs W_low | **+0.002624** | **+0.002713** |

---

## Decision Gate

```
if any_real_viable_and_improves_over_W_low:
    classification = "PASS_REAL_ENCODING_MEMORY_VIABLE"  ✅
elif any_lowrank_viable_and_improves:
    classification = "PARTIAL_LOWRANK_ONLY"
elif any_sparse_looks_viable_with_metadata_correction:
    classification = "PARTIAL_IDEAL_ONLY_METADATA_FAILS"
else:
    classification = "BLOCKED_BY_STORAGE_ECONOMICS"
```

**Result:** `PASS_REAL_ENCODING_MEMORY_VIABLE` — Multiple sparse encodings are memory-viable and improve quality. Low-rank is viable but ineffective.

---

## Methodology Notes

- **gguf.dequantize** used for exact tensor extraction
- **Cosine similarity** computed row-wise with BATCH=8
- **Storage accounting** includes ALL overhead: bitmaps, indices, scales, value bytes
- **No speed claims** — only storage and quality metrics
- **Quality metric:** cosine similarity of (W_low + encoded_residual) vs W_ref

---

## Phase 31C Provenance

- Phase 31C claimed: `topk_10% memory viable` based on value byte count (1,743,256)
- Actual cost (fp32): `bitmap(544,768) + fp32_values(1,743,256) = 2,288,024` → **OVER budget**
- fp16 makes it viable but requires different quality assessment
