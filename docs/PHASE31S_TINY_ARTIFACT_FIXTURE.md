# Phase 31S: Tiny On-Disk Substitutive Artifact Fixture

**Classification:** `PARTIAL_W_LOW_PACKING_PENDING` ⚠️

---

## What Phase 31S Delivered

**Artifact structure — PASS ✅**
- On-disk fixture created at `data/phase31s_fixture/`
- Manifest: `manifest.json` (schema v0.1.0)
- Tensors: `tensors/blk.0.ffn_up.W_low.bin` + `tensors/blk.0.ffn_up.residual.sdir`
- Metadata: `metadata/checksums.json` (SHA-256)
- Checksums validate ✅
- Loader: `src/artifact_load.py` — parses manifest, verifies checksums, loads artifacts

---

## W_low Format Status — CRITICAL FINDING

| Property | Value |
|----------|-------|
| Format used | `blocked-int8-uint8-storage` |
| Storage type | uint8 (1 byte per element) |
| Actual W_low bytes | 4,630,528 |
| **True Q4_K_M (target)** | **~1,089,536 (0.5 bytes/element)** |
| Toy harness (for comparison) | ~17,432,576 (fp32, 4 bytes/element) |
| Nibble packing | **PENDING** — attempted but dtype issue |

**Honest assessment:** W_low is stored as uint8 (1 byte/element), NOT true Q4_K_M nibble-packed (0.5 bytes/element). The attempted nibble packing had a numpy dtype issue. uint8 was chosen as a middle ground between toy harness (4 bytes/element) and true Q4_K_M (0.5 bytes/element).

---

## Memory Accounting (REAL, based on actual stored bytes)

| Metric | Value | Note |
|--------|-------|------|
| W_ref (fp32) | 17,432,576 | extracted real tensor |
| W_ref Q4 budget | 4,358,144 | 4 bits/element reference |
| **W_low (uint8)** | **4,630,528** | actual artifact bytes |
| Residual encoded | 1,198,506 | bitmap + fp16 + header |
| **Total substitutive** | **5,829,034** | **actual bytes** |
| **Margin vs Q4** | **-1,470,890** | **NEGATIVE ❌** |

**Key finding:** The uint8 W_low storage (4.63MB) is HEAVIER than the Q4 reference budget (4.36MB). Memory margin is **negative**. True Q4_K_M packing (1.09MB) would make the total ~2.29MB with a margin of ~+2.07MB.

---

## What Passed ✅

- Artifact structure loads correctly
- SHA-256 checksums validate for both W_low and residual
- W_ref absent (loader does not load W_ref)
- Dense R absent (0 bytes materialized)
- Residual encoded present and loads correctly
- Fail-fast on missing file works
- Path label `[SDI-SUB-RUNTIME]` confirmed
- Manifest schema v0.1.0 parses correctly

---

## What Is Partial ⚠️

- W_low is NOT true Q4_K_M packed format
- Memory margin is NEGATIVE: -1.47MB (uint8 is heavier than Q4 budget)
- Nibble packing (0.5 bytes/element) was attempted but had a dtype issue
- True Q4_K_M packing needed to achieve positive memory margin

---

## Recommended Next Phase: Fix W_low Nibble Packing

**Before expanding to layers 0–5 (Phase 31T), W_low must achieve true Q4_K_M nibble packing (0.5 bytes/element).**

The current uint8 storage (1 byte/element) produces a **negative memory margin**, which defeats the purpose of the substitutive path.

**Required fix:** Resolve the numpy dtype issue in the nibble packing path of `artifact_write.py` to achieve:
- W_low bytes: ~1,089,536 (0.5 bytes/element)
- Total substitutive: ~2,288,284
- Margin: ~+2,069,860 vs Q4 reference

---
*Phase 31S — ELVIS — SDI Substitutive*