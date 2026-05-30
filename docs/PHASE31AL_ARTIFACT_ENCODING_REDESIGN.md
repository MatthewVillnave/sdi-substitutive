# Phase 31AL — Artifact Encoding Redesign / W_low Budget Fix

## Classification
**`PASS_WLOW_ENCODING_CANDIDATE_FOUND`**

A W_low encoding candidate (Q2_K_M) combined with a redesigned residual encoding (Q2 or int8-sparse) is memory-positive under the Q4 budget.

---

## Scope
- **Layers**: 0–5
- **Families**: ffn_up (k=9%), ffn_gate (k=12%), ffn_down (k=9%)
- **Reference**: Phase 31AK and Phase 31AJ results

---

## Budget Definitions

| Term | Definition | Bytes |
|------|-----------|-------|
| **Q4_budget** | `n_elements × 4 / 8` — raw Q4 nibble storage only | 2,179,072 per family-layer |
| **Q4_K_M** | GGUF Q4_K_M format: block256, 84 bytes/block | 1,430,028 per family-layer |
| **Q2_K_M** | GGUF Q2_K_M format: block256, 108 bytes/block | 1,818,060 per family-layer |
| **SDIR** | Current residual: full bitmap (1 bit/elem) + fp16 per nonzero | 1.33–1.59 MB per family-layer |

**Q4_budget = nibble storage only.** The sdiw format adds scale bytes on top, exceeding the budget.

---

## Current sdiw Byte Breakdown

Per family-layer (e.g., ffn_up layer 0):

| Component | Bytes | Bits/elem |
|-----------|------:|----------:|
| Packed nibbles (Q4_2) | 2,179,072 | 4.00 |
| Scale (fp16, block32) | 272,384 | 0.50 |
| **Total** | **2,451,456** | **4.50** |

**Overhead vs Q4_budget: +272,384 bytes (+12.5%)** per family-layer.

For all 3 families × 6 layers:
- sdiw W_low total: **42.08 MB**
- Q4_budget total: **37.41 MB**
- Overhead: **+4.68 MB** (W_low alone exceeds budget)

---

## Residual Encoding: SDIR Is Inefficient

SDIR stores a full bitmap (1 bit per element = 544,768 bytes per family-layer) regardless of k percentage. This bitmap is the same size whether k=9% or k=12%.

| Family | k% | Bitmap | fp16 values | SDIR total | Bits/nz |
|--------|:--:|-------:|------------:|-----------:|--------:|
| ffn_up | 9% | 544,768 | 784,464 | 1,329,344 | 27.1 |
| ffn_gate | 12% | 544,768 | 1,045,954 | 1,590,782 | 24.3 |
| ffn_down | 9% | 544,768 | 784,464 | 1,329,304 | 27.1 |

**Problem**: The bitmap is 544,768 bytes (always) while the actual data stored is 784K–1M bytes (fp16 values). The bitmap alone is 41–43% of the residual size.

**Dense alternatives** (storing all elements in a lower-precision format):
- Q4 dense: 4 bits/elem = 1,089,536 bytes
- Q3 dense: 3 bits/elem = 816,192 bytes
- Q2 dense: 2 bits/elem = 544,768 bytes
- Int8 sparse (bitmap + int8): 544,768 + nnz × 1 bytes

For k=9% (nnz=392K): int8 sparse = 936,960 bytes vs SDIR 1,329,344 bytes — **1.4× more efficient**.

---

## Candidate Encoding Table

| Candidate | Bits/elem | W_low Total (3×6) | vs Q4_budget | Memory-positive alone? |
|-----------|----------:|------------------:|-------------:|:--------------------:|
| sdiw Q4_2 (current) | 4.50 | 42.08 MB | +4.68 MB | ✗ |
| block64 fp16 | 4.25 | 39.74 MB | +2.34 MB | ✗ |
| block128 fp16 | 4.125 | 38.58 MB | +1.17 MB | ✗ |
| block32 int8 | 4.25 | 39.74 MB | +2.34 MB | ✗ |
| Q4_K_M (GGUF) | 2.625 | 24.55 MB | −12.86 MB | ✓ |
| **Q2_K_M (GGUF)** | **2.0625** | **31.56 MB** | **−5.84 MB** | **✓** |

Q2_K_M W_low fits within Q4_budget (−5.84 MB margin).

---

## Numerical Probe Results

Current sdiw Q4_2 W_low quality (layer 0 ffn_up, vs reference):
- Cosine: 0.995945
- MAE: 0.007523

**Note**: Q2_K_M and Q4_K_M decode quality not measured — requires GGUF dequantization unavailable locally. Quality risk is the primary unknown for Q2_K_M.

---

## Combined W_low + Residual Viability

Only combinations that are memory-positive:

| W_low | Residual | Total | Margin | Status |
|-------|----------|------:|-------:|:------:|
| **Q2_K_M** | **Q2 dense** | **35.07 MB** | **+2,394 KB** | **✓ VIABLE** |
| **Q2_K_M** | **int8 sparse** | **33.20 MB** | **+4,309 KB** | **✓ VIABLE** |

**Winning strategy: W_low=Q2_K_M (GGUF) + Residual=Q2 or int8-sparse**

Key tradeoffs:
- Q2_K_M W_low uses GGUF's existing Q2_K_M format — leverages llama.cpp decode
- Q2 dense residual is the most efficient dense residual format
- int8 sparse residual requires bitmap + int8 values (simpler than SDIR's fp16)
- SDIR residual is NOT viable with any W_low format at current k%

---

## Remaining Unknowns

1. **Q2_K_M W_low quality**: not measured — GGUF dequantize unavailable locally
2. **Q2 dense residual quality**: residual in Q2 format vs fp16 residual — needs verification
3. **Residual accuracy impact**: at k=9%, residual should capture ~9% of elements; compressing residual to Q2/Q4 may partially defeat the purpose
4. **Scale sharing**: up/gate/down could share scale tables since they share d_in=896 and d_out=4864 — not yet explored

---

## Recommended Next Phase

**Phase 31AM — Implement Q2_K_M W_low decode + Q2/int8 residual encoding prototype, only if explicitly requested.**

Steps:
1. Implement GGUF Q2_K_M dequantization for W_low (reuse llama.cpp if possible)
2. Implement Q2 dense residual encoder (2 bits/elem) as replacement for SDIR
3. Implement int8 sparse residual as alternative
4. Verify W_low decode quality numerically against reference
5. Verify combined W_low + residual fits budget
6. Run full MLP substitutive runtime with new artifacts

---

## Forbidden Claims (Maintained)
- no model quality recovery
- no behavior recovery
- no speedup
- no full-model memory savings
- no llama.cpp integration
- no production readiness

---

## Script
`src/phase31al_artifact_encoding_redesign.py`
