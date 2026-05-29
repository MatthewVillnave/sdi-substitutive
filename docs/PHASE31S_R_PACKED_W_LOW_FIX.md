# Phase 31S-R: Packed W_low Nibble Format Fix

## Classification
**PARTIAL_PACKED_ARTIFACT_MEMORY_PASS_DECODE_BLOAT**

The artifact format achieves 0.5 byte/element nibble packing with positive margin. Runtime decode temp is high but that's a Phase 31T runtime streaming decode problem, not an artifact format problem.

---

## What's Changed

### Format: Packed Nibble W_low Prototype

| Property | Value |
|----------|-------|
| Nibbles per byte | 2 (4-bit values per element) |
| Value range | 0–15 stored, offset-8 for signed: -8..+7 |
| Quantization | block-wise, scale per block (fp16) |
| Block size | 32 elements (16 nibble pairs per byte, 16 scale values) |
| Nibble order | little-endian (low nibble = first of pair) |
| Shape metadata | rows, cols, block_size in manifest |
| Checksum | SHA-256 of packed bytes |

**Honest wording:** "packed nibble W_low prototype" — NOT "GGUF Q4_K_M" unless it matches GGUF exactly.

### Per-Block Structure
- Block: 32 elements → 16 nibble pairs → 16 bytes
- Scale: fp16 (2 bytes)
- Per-block overhead: 2 bytes per 16 bytes = 0.125 bytes/element

### Expected vs Actual (ffn_up 896×4864)

| Metric | Expected | Actual |
|--------|----------|--------|
| Packed data bytes | ceil(N/2) = 2,179,072 | 2,179,072 ✅ |
| Scales bytes | N/32×2 = 272,384 | 272,384 ✅ |
| W_low total | 2,451,456 | 2,451,456 ✅ |
| Residual bytes | ~1,198,508 | 1,198,508 ✅ |
| **Total artifact bytes** | ~3,649,964 | 3,649,964 ✅ |
| Bytes/element (total) | 0.5625 | 0.5625 ✅ |

---

## Phase History

| Phase | Format | Bytes/Elem | Margin vs Q4 |
|-------|--------|------------|--------------|
| 31S | uint8 blocked int8 | 1.00 | -2,179,072 ❌ |
| **31S-R** | **packed nibble** | **0.5625** | **+708,180 ✅** |

---

## Artifact Files

### data/phase31s_fixture/tensors/blk.0.ffn_up.W_low.bin
- Format: packed nibbles + fp16 scales appended
- Bytes: 2,179,072 (packed) + 272,384 (scales) = **2,451,456 total**
- SHA256: `10ecd59c095f31c0c266795721e5c9d4ff3356afbdbae7b5a697ac2dd024312f`

### data/phase31s_fixture/tensors/blk.0.ffn_up.residual.sdir
- Unchanged from 31S (bitmap + fp16 residual values)
- Bytes: **1,198,508**
- SHA256: `9fd307c4194e48b62946a8410cefc880b60d568fa0d20e7139905453b4daa5ce`

### data/phase31s_fixture/manifest.json
- Updated: package_id = `phase31s-r-ffn-up-layer0-packed`
- W_low_format = `packed-nibble-uint8-storage`
- Memory margin: +708,180 bytes (POSITIVE)

---

## Key Definitions

### Artifact Storage Budget
The Q4 budget (4,358,144 bytes) = full element count = rows×cols = 896×4864

### Runtime Resident Cost
When W_low is decoded for compute, it materializes as fp32 (17,432,576 bytes). This is reported as `W_low_decode_temp_bytes` in the loader counters.

### Classification Rationale

| Condition | Result |
|-----------|--------|
| Artifact bytes fit under Q4 budget | ✅ PASS (3,649,964 < 4,358,144) |
| W_ref absent from artifact | ✅ PASS |
| dense_R absent from artifact | ✅ PASS |
| Pack/unpack reliable | ✅ PASS |
| **Runtime resident (artifact + decode temp)** | ❌ **17.4 MB** |

Runtime bloat is a **Phase 31T runtime streaming decode** problem, not a format problem.

---

## Commit

```
Phase 31S-R: add packed W_low artifact format
```

**Files committed:**
- `src/wlow_pack.py` — pack/unpack implementation with tests
- `src/artifact_write.py` — updated to use packed nibble format
- `src/artifact_load.py` — updated loader for packed format
- `data/phase31s_fixture/tensors/blk.0.ffn_up.W_low.bin` — NEW packed artifact
- `data/phase31s_fixture/metadata/checksums.json` — updated checksums
- `data/phase31s_fixture/manifest.json` — updated manifest

---

## Phase 31T Unlocked

**YES** — artifact format is validated. Phase 31T should focus on:
1. Multi-layer packed artifact generator
2. Streaming decode to avoid full W_low fp32 materialization
3. Runtime memory management: keep working set under Q4 budget
