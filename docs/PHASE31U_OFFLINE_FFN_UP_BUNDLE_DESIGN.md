# Phase 31U: Offline ffn_up Substitutive Bundle Design

**Classification:** `PASS_BUNDLE_DESIGN_READY`

---

## Bundle Layout

```
sdi-bundle/
├── manifest.json                    # v0.2 master manifest
├── tensors/
│   ├── blk.{L}.ffn_up.wlow.sdiw    # Packed W_low (streaming decode)
│   └── blk.{L}.ffn_up.residual.sdir # Sparse residual (bitmap+fp16)
└── metadata/
    ├── checksums.json               # SHA-256 per file
    ├── memory_budget.json           # Per-layer bounds
    └── provenance.json               # Generation metadata
```

---

## Manifest Schema v0.2 (vs v0.1 from Phase 31Q)

**Key additions vs v0.1:**
- `decode_temp_bound_bytes: 128` (block32 streaming)
- `W_low_scale_bytes` tracked separately from packed data
- `all_layers_positive` global flag
- `formats.scale_policy: block32_fp16`
- Checksums centralized to `metadata/checksums.json`
- Memory budget separated to `metadata/memory_budget.json`
- Provenance formalized in `metadata/provenance.json`

**Per-layer entry fields:**
tensor_name, layer, family, shape, orientation, W_ref_bytes, W_ref_Q4_budget_bytes, W_low_packed_bytes, W_low_scale_bytes, residual_bytes, total_substitutive_bytes, memory_margin_bytes, decode_temp_bound_bytes, checksums, formats (W_low_format, residual_format, k_pct, value_dtype, mask_encoding, scale_policy), approximation_summary

---

## .sdiw W_low Format (SDI W_low)

**Magic:** `0x53444957` ("SDIW"), version 1

| Section | Size | Format |
|---------|------|--------|
| Header | 16 bytes | magic(4) + version(2) + flags(2) + rows(4) + cols(4) |
| Scale table | n_blocks × 2 bytes | fp16 LE, block size = 32 |
| Packed nibbles | ceil(N/2) bytes | 2 values/byte, LSB-first nibble order |

**Layer 0 example (896×4864):** 16 + 272,384 + 2,179,072 = **2,451,472 bytes**

**Decode semantics:**
- Never materialize full fp32 W_low
- Decode block-by-block during compute
- Peak temp: 128 bytes per block (32 × float32)

---

## .sdir Residual Format (SDI Residual, v0.1)

**Unchanged from Phase 31Q.**

| Section | Size | Format |
|---------|------|--------|
| Header | 16 bytes | in_dim(4) + out_dim(4) + nnz(4) + flags(4) |
| Bitmap | ceil(N/8) bytes | LSB-first row-major |
| Values | nnz × 2 bytes | fp16 LE, row-major for set bits |

**Layer 0 example:** 16 + 544,768 + 653,720 = **1,198,504 bytes**

**Streaming apply semantics:**
- Decode bitmap on load (~544KB)
- Values consumed in row-major order during X @ R_sparse
- Never materialize dense R

---

## Provenance Schema

Required fields: `generation_commit`, `generation_date`, `source_model`, `tensor_family`, `layers[]`, `extraction_method`, `W_low_quantization`, `residual_generation`, `validation_prompts`, `mean_delta_cosine`, `worst_delta_cosine`, `regressions`, `memory_margin_per_layer`, `artifact_bytes_per_layer`, `code_modules_used[]`, `no_model_weights_committed: true`

---

## Validation Suite (20 checks)

**Static (10):**
1. manifest.json parses with schema v0.2
2. All checksums match
3. Every layer memory_margin_bytes > 0
4. All formats fields valid
5. All tensor files exist
6. mean_delta_cosine > 0
7. decode_temp_bound_bytes present
8. schema_version = "0.2.0"
9. layers_included matches per-layer entries
10. bundle_type = "ffn_up_substitutive"

**Behavioral (10):**
1. W_ref_loaded = 0
2. dense_R_materialized = 0
3. dense_W_low_materialized = 0
4. decode_temp_peak_bytes ≤ decode_temp_bound_bytes
5. fallback_count = 0
6. error_count = 0
7. path_label = "[SDI-SUB-RUNTIME]"
8. fail-fast on missing .sdiw
9. fail-fast on missing .sdir
10. fail-fast on checksum mismatch

---

## Runtime Contract

| Counter | Expected |
|---------|---------|
| W_ref_loaded | **0** |
| W_low_loaded | **true** |
| residual_loaded | **true** |
| dense_R_materialized | **0** |
| dense_W_low_materialized | **0** |
| decode_temp_peak_bytes | **≤ 128** |
| fallback_count | **0** |
| error_count | **0** |
| path_label | **`[SDI-SUB-RUNTIME]`** |

---

## Recommended Next Phase

**Phase 31V — Streaming .sdiw Decode Kernel**

Pure C implementation, works with synthetic fixture. Verifies block-by-block decode + peak temp bound. Unblocks Phase 31X (bundle runtime loader).

**Alternative:** Phase 31W — Generate local full ffn_up layers 0–5 bundle with real artifacts.

---

## Classification: PASS_BUNDLE_DESIGN_READY

All design artifacts delivered: bundle layout, manifest schema v0.2, .sdiw format, .sdir format, provenance schema, 20-point validation suite, runtime contract, example manifest.

---
*Phase 31U — ELVIS — SDI Substitutive*
