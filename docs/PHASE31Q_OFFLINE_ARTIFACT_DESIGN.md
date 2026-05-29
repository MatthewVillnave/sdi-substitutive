# Phase 31Q: Offline Substitutive Model Artifact Design

**Classification:** `PASS_ARTIFACT_DESIGN_READY`

---

## 1. Artifact Directory Layout

```
sdi-substitutive-artifact/
├── manifest.json                    # Master manifest
├── tensors/
│   ├── blk.0.ffn_up.W_low.bin        # W_low (Q4_K_M packed GGUF)
│   ├── blk.0.ffn_up.residual.sdir    # Encoded residual
│   ├── blk.1.ffn_up.W_low.bin
│   ├── blk.1.ffn_up.residual.sdir
│   ├── ...
│   ├── blk.5.ffn_up.W_low.bin
│   └── blk.5.ffn_up.residual.sdir
├── metadata/
│   ├── checksums.json               # SHA-256 per file
│   ├── memory_budget.json           # Memory bounds and margins
│   └── tensor_index.json           # Tensor registry
└── examples/
    └── manifest.example.json        # Tiny fixture (no real data)
```

---

## 2. Manifest Schema (key fields)

| Field | Description |
|-------|-------------|
| `schema_version` | semver (current: `0.1.0`) |
| `package_id` | Unique identifier |
| `source_model` | model_name, quantization, gguf_source_path, architecture |
| `substitution_policy.k_percent` | 7.5 |
| `substitution_policy.W_low_format` | `Q4_K_M_packed_GGUF` |
| `substitution_policy.residual_encoding` | `bitmap+fp16-header` |
| `runtime_requirements.W_ref_must_be_absent` | true |
| `runtime_requirements.dense_R_must_not_be_materialized` | true |
| `runtime_requirements.fail_fast_if_residual_missing` | true |
| `runtime_requirements.path_label` | `[SDI-SUB-RUNTIME]` |
| `global_memory.W_ref_total_bytes_avoided` | sum across all tensors |
| `global_memory.memory_margin_bytes` | positive required |
| `layers[].layer` | layer index |
| `layers[].family` | ffn_up / ffn_down |
| `layers[].shape` | rows × cols |
| `layers[].W_ref_f32_bytes` | fp32 size |
| `layers[].W_low_bytes` | actual stored bytes |
| `layers[].residual_encoded_bytes` | bitmap + fp16 + header |
| `layers[].memory_margin_bytes` | W_ref - (W_low + residual) |
| `layers[].hash_W_low` | SHA-256 |
| `layers[].hash_residual` | SHA-256 |

---

## 3. Residual File Format

**Layout:** `[Header: 16 bytes] + [Bitmap: ceil(rows×cols/8) bytes] + [Values: nnz×2 bytes fp16]`

| Section | Size | Format |
|---------|------|--------|
| Header | 16 bytes | 4× uint32 LE: (in_dim, out_dim, nnz, flags) |
| Bitmap | ceil(rows×cols/8) bytes | LSB-first, row-major, 1 bit per element |
| Values | nnz × 2 bytes | IEEE 754 binary16, LE, row-major for set bits |
| Alignment | None required between sections | — |

- **Forward compat:** flags field reserved; unknown non-zero bits = reject
- **Endianness:** little-endian throughout
- **Checksum:** SHA-256 per residual file in `metadata/checksums.json`

---

## 4. W_low Format — Critical Correction

**Important:** The toy runtime (phases 31O/31P) stores W_low as **blocked int8 approximation in fp32 arrays** — this is NOT true Q4_K_M packed format. Memory margin claims from those phases were based on theoretical Q4_K_M sizes, not actual stored bytes.

| Representation | Format | W_low bytes (ffn_up) |
|---|---|---|
| Toy runtime (31O/31P) | blocked int8 → stored in fp32 | ~4.46 MB (estimated) |
| **Design target** | **Q4_K_M_packed_GGUF** | **~1.09 MB** |

**Manifest field `W_low_format`** must distinguish:
- `blocked-int8-fp32` — toy/validation harness only
- `Q4_K_M_packed_GGUF` — production artifact target

**Real Q4_K_M sizes per layer (ffn_up, 896×4864):**
- W_ref (fp32): 17,432,576 bytes
- W_ref Q4 budget: 2,451,456 bytes
- W_low (Q4_K_M): ~1,089,536 bytes
- Residual encoded: ~1,198,748 bytes
- **Total substitutive: ~2,288,284 bytes**
- **Real margin vs Q4: ~163,172 bytes** (using real Q4_K_M, not fp32 W_ref)

---

## 5. No-Additive-Trap Runtime Contract

A conforming SDI-SUB-RUNTIME must:
- **NOT** load W_ref for substituted tensors (`W_ref_loaded = 0`)
- Load W_low from `tensors/blk.{L}.{family}.W_low.bin`
- Load encoded residual from `tensors/blk.{L}.{family}.residual.sdir`
- **NOT** materialize dense residual matrix (`dense_R_materialized = 0`)
- Fail fast if residual missing (no silent fallback)
- Emit path label `[SDI-SUB-RUNTIME]` in all substitutive-path logs

**Required counters:**

| Counter | Expected value |
|---------|---------------|
| `W_ref_loaded` | 0 for substituted tensors |
| `W_low_loaded` | actual bytes |
| `residual_encoded_loaded` | actual bytes |
| `dense_R_materialized` | **0** |
| `memory_margin_vs_ref` | positive |
| `fallback_count` | **0** (no silent fallback) |
| `error_count` | 0 (clean run) |

---

## 6. Validation Checklist

For any artifact package:

1. ✅ `manifest.json` parses without error
2. ✅ `checksums.json` SHA-256 matches each file
3. ✅ `memory_margin_bytes > 0` for every layer
4. ✅ W_ref absent test: reference mode output ≠ substitutive mode (proves W_ref not used)
5. ✅ Missing residual fail-fast: raises clear error, no silent fallback
6. ✅ Output matches toy runtime reference: cosine > 0.9999
7. ✅ `dense_R_materialized = 0` confirmed
8. ✅ Path label `[SDI-SUB-RUNTIME]` in logs

---

## 7. Next Recommended Phase: 31S

**Phase 31S: Tiny on-disk artifact fixture with real Q4_K_M W_low**

**Rationale:**
- Safest next step — synthetic fixture, no GGUF extraction surgery
- Clearest validation — tiny fixture verifiable in isolation
- No llama.cpp surgery required
- Unblocks Phase 31T (full artifact generator) with clear validation gates
- Lowest risk — clear failure signal if format issues arise

**Deliverables:**
- `src/artifact_fixture.cpp` — generates synthetic W_low (Q4_K_M packed format) + residual fixture
- `tests/test_artifact_fixture.cpp` — validation checklist runner
- `examples/manifest.example.json` — minimal manifest fixture

---

## 8. Classification: PASS_ARTIFACT_DESIGN_READY

- Directory layout defined
- Manifest schema specified with all required fields
- Residual format documented (16-byte header + bitmap + fp16)
- W_low format clarified (toy vs production target)
- No-additive-trap runtime contract defined
- Validation checklist defined
- Next phase (31S) recommended

---
*Phase 31Q — ELVIS — SDI Substitutive*
