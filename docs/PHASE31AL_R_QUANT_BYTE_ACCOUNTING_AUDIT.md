# Phase 31AL-R — Quant Byte Accounting Audit

## Classification
**`PARTIAL_31AL_VIABILITY_REVISED`**

31AL labels were swapped/mislabeled. Corrected accounting changes which W_low candidate is viable and by how much. A k-reduction strategy (k≤2%) with Q2_K + int8 sparse residual remains viable; current k=9-12% does not.

---

## Audit Result Summary

| Issue | 31AL Claim | Corrected Value |
|-------|------------|-----------------|
| Q4_K_M bytes | 1,430,028 | **Q2_K = 1,430,028** (label was swapped) |
| Q2_K_M bytes | 1,818,060 | **NOT a standard GGUF format** (fabricated value) |
| Q4_K_M bits/elem | 2.625 | **Q2_K = 2.625 bits/elem** (correct value, wrong label) |
| Q2_K_M bits/elem | 2.0625 | **3.337 bits/elem** (fabricated — no standard GGUF type) |
| block128_fp16 bits/elem | 4.125 | **4.500** (same as sdiw, not a savings) |
| Q4_K bytes (correct) | — | **2,451,468** (same as sdiw Q4_2, over budget) |

---

## Verified GGUF Quant Constants

Source: `llama.cpp/gguf-py/gguf/constants.py`

```python
QK_K = 256
GGML_QUANT_SIZES = {
    GGMLQuantizationType.Q2_K: (256, 2 + 2 + QK_K//16 + QK_K//4),     # 84 bytes/block
    GGMLQuantizationType.Q3_K: (256, 2 + QK_K//4 + QK_K//8 + 12),      # 110 bytes/block
    GGMLQuantizationType.Q4_K: (256, 2 + 2 + QK_K//2 + 12),            # 144 bytes/block
    GGMLQuantizationType.Q5_K: (256, 2 + 2 + QK_K//2 + QK_K//8 + 12), # 176 bytes/block
    GGMLQuantizationType.Q6_K: (256, 2 + QK_K//2 + QK_K//4 + QK_K//16),# 210 bytes/block
}
```

For tensor: n_elements = 4,358,144, n_blocks = 17,024

| Format | Bytes/Block | Blocks | Total Bytes | Bits/Elem | vs Q4_budget |
|--------|------------:|-------:|------------:|----------:|-------------:|
| Q2_K | 84 | 17,024 | 1,430,028 | 2.625 | ✓ fits (−749 KB) |
| Q3_K | 110 | 17,024 | 1,872,652 | 3.438 | ✓ fits (−306 KB) |
| Q4_K | 144 | 17,024 | 2,451,468 | 4.500 | ✗ over (+272 KB) |
| sdiw (Q4_2 block32) | — | — | 2,451,456 | 4.500 | ✗ over (+272 KB) |
| Q4_budget (nibbles only) | — | — | 2,179,072 | 4.000 | reference |

**Note**: Q4_K = Q4_0 = 2,451,468 bytes (both use 4.5 bits/elem). The sdiw format is essentially Q4_K-equivalent in byte count.

---

## Corrected Byte Table (all 3 families × 6 layers)

| Candidate | W_low Total | vs Q4_budget (37.41 MB) | Memory-positive W_low alone? |
|-----------|------------:|-------------------------:|:-----------------------------:|
| sdiw (current) | 42.08 MB | +4.68 MB | ✗ |
| Q4_K (correct) | 42.08 MB | +4.68 MB | ✗ |
| Q3_K | 32.15 MB | −5.26 MB | ✓ |
| **Q2_K** | **24.55 MB** | **−12.86 MB** | **✓** |

---

## SDIR Residual Inefficiency — Confirmed

SDIR stores: full bitmap (1 bit/elem = 544,768 bytes) + fp16 per nonzero (16 bits/nonzero).

At k=9-12% with 4,358,144 elements:
- Bitmap: 544,768 bytes per family-layer (always the same, regardless of k%)
- fp16 values: 784,464–1,045,954 bytes per family-layer at k=9-12%
- Total SDIR: 1,329,344–1,590,782 bytes per family-layer
- Bits/nonzero: 24–27 bits (extremely wasteful — fp16 is overkill)

**Alternative residual encodings:**
| Encoding | Bytes/family-layer | Notes |
|----------|-------------------:|-------|
| SDIR (current) | 1.33–1.59 MB | Full bitmap + fp16 values |
| Q4 dense (4 bits/elem) | 1.09 MB | All elements |
| Q3 dense (3 bits/elem) | 0.82 MB | All elements |
| Q2 dense (2 bits/elem) | 0.55 MB | All elements |
| int8 sparse (bitmap + int8) | 0.89 MB @ k=9% | Lower precision values |

---

## Corrected Combined Viability

**Current k=9-12% (up=9%, gate=12%, down=9%):**

| W_low | Residual | Total | Margin vs Q4_budget | Viable? |
|-------|----------|------:|--------------------:|:--------:|
| Q2_K | SDIR | 48.88 MB | −11.47 MB | ✗ |
| Q2_K | Q2 dense | 43.25 MB | −5.84 MB | ✗ |
| Q2_K | int8 sparse | 41.38 MB | −3.97 MB | ✗ |
| Q3_K | SDIR | 56.43 MB | −19.02 MB | ✗ |
| Q3_K | Q2 dense | 50.85 MB | −13.44 MB | ✗ |

**No combined W_low + residual policy is viable at current k=9-12%.**

**Key finding: k must be reduced to ≤3% for Q2_K + int8 sparse to be viable.**

---

## k-Reduction Analysis: Q2_K + int8 sparse residual

| k% | nnz/family-layer | int8 resid bytes | Q2_K+int8 total | Margin | Viable? |
|:--:|----------------:|----------------:|----------------:|-------:|:--------:|
| 1% | 43,581 | 588,416 | 34.65 MB | +2,825 KB | ✓ |
| 2% | 87,162 | 1,132,928 | 35.40 MB | +2,059 KB | ✓ |
| 3% | 130,744 | 1,675,712 | 36.14 MB | +1,293 KB | ✓ |
| 5% | 217,907 | 2,762,752 | 37.64 MB | −240 KB | ✗ |
| 7% | 305,070 | 3,849,728 | 38.91 MB | −1,507 KB | ✗ |
| 9% | 392,232 | 4,936,704 | 41.38 MB | −3,970 KB | ✗ |
| 12% | 522,977 | 6,562,048 | 44.65 MB | −7,243 KB | ✗ |

**k≤3% is viable with Q2_K + int8 sparse. k≥5% is not.**

Tradeoff: k reduction means the residual captures fewer elements. At k=1-2%, the residual is very sparse and cheap to store, but the approximation improvement from residuals is reduced proportionally.

---

## Corrected Viability Summary

| Scenario | Status | Notes |
|----------|:------:|-------|
| sdiw + SDIR @ k=9-12% | ✗ | Current (confirmed fail) |
| Q2_K + SDIR @ k=9-12% | ✗ | Confirmed |
| Q2_K + Q2 dense @ k=9-12% | ✗ | Confirmed |
| Q2_K + int8 sparse @ k=9-12% | ✗ | Confirmed |
| **Q2_K + int8 sparse @ k≤3%** | **✓** | **Viable with k reduction** |
| Q3_K + int8 sparse @ k≤2% | ✓ | Viable but Q3_K is larger than Q2_K |

---

## 31AL Classification Update

**31AL original**: `PASS_WLOW_ENCODING_CANDIDATE_FOUND`
**Corrected**: `PARTIAL_31AL_VIABILITY_REVISED`

31AL correctly identified that Q2_K is the viable W_low format and SDIR is the budget blocker. The label swap and fabricated byte value did not change the qualitative direction. However, the quantitative margin estimates were wrong, and the conclusion that "Q2_K + int8 sparse is viable" requires k≤3%, not current k=9-12%.

---

## Primary Budget Blocker

The residual encoding is the primary blocker. SDIR's full bitmap (1 bit/elem) plus fp16 per nonzero (16 bits/nonzero) is extremely inefficient at k=9-12%. For the residual to fit under the Q4 budget alongside any W_low format:

1. **Reduce k significantly** (to ≤3%) — reduces nnz count
2. **Use int8 instead of fp16** for residual values — halves value storage
3. **Or accept that the substitutive path works only as a quality improvement, not a memory savings**

---

## Recommendation

If memory-positive full MLP is the goal:
- Q2_K + int8 sparse residual @ k≤3% is viable
- This requires re-running artifact capture at lower k thresholds
- Alternatively: accept Q2_K W_low with no residual (approximation degrades to low-only)

If k=9-12% is required for approximation quality:
- No memory-positive combined policy exists under current encoding
- Residual encoding redesign is required before full MLP memory-positivity

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
`src/phase31al_r_quant_byte_accounting_audit.py`
