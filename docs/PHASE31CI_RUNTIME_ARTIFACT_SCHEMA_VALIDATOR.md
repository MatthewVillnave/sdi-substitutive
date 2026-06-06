# Phase 31CI — Runtime Artifact Schema Prototype / Metadata-Only Manifest Validator

> **Implementation phase, but metadata-only. No model load. No Q2_K/SDIR binary generation. No runtime loader implementation. No llama.cpp modification. No raw activation arrays. No inference-quality tests. No generation. No commit, push, or tag without explicit Matt approval.**

---

## 1. Goal and scope

31CI turns the 31CH v1.1.0 runtime artifact-format plan into a working **metadata-only** schema prototype and manifest validator. The validator:

- Validates a v1.1.0 manifest dictionary or JSON file
- Enforces the 31CH v1.1.0 design's policy invariants, runtime safety invariants, legacy sidecar exclusion, manifest hygiene, per-layer/per-family metadata, memory-budget metadata, and provenance metadata
- Produces structured results (passed, error_count, warning_count, errors, warnings, rules_checked, manifest_sha256, classification_suggestion)
- Includes 10 lightweight in-code self-tests (1 valid sample + 9 negative tests + 1 bonus metadata-only placeholder test)
- Does **NOT** load any model file, does **NOT** inspect GGUF tensor payloads, does **NOT** read or write any Q2_K/SDIR binary artifact, does **NOT** read or write any raw activation array, does **NOT** implement a runtime loader, does **NOT** modify or rebuild `~/llama.cpp/`

### 1.1 What 31CI does

- ✓ implements a Python validator module (`src/phase31ci_manifest_validator.py`)
- ✓ implements a metadata-only sample manifest (`src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json`)
- ✓ runs 10 in-code self-tests (all pass)
- ✓ validates the sample manifest against the validator (PASS_31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR_CLEAN)
- ✓ enforces 50 rules in 8 categories
- ✓ provides a CLI entry point for batch validation (`python3 src/phase31ci_manifest_validator.py --self-test` or `--manifest <path>`)

### 1.2 What 31CI does NOT do

- ✗ no model load (no `llama_model_load_from_file`, no `AutoModel.from_pretrained`, no `GGUFReader(...)` for tensor payloads)
- ✗ no Q2_K/SDIR binary artifact generation
- ✗ no raw activation array generation or capture
- ✗ no runtime loader implementation (validation methods only)
- ✗ no `~/llama.cpp/` source modification
- ✗ no `~/llama.cpp/` rebuild
- ✗ no PRT sidecar re-activation
- ✗ no generation, no sampling, no inference
- ✗ no inference-quality test
- ✗ no compiled binary generation
- ✗ no `~/llama.cpp/` build artifacts
- ✗ no tag created or pushed
- ✗ no commit/push/tag without explicit operator approval

### 1.3 Relationship to prior phases

| phase | role | relationship to 31CI |
|---|---|---|
| 31CH | runtime artifact format / loader planning | the design that 31CI implements — v1.1.0 schema, 48 loader rules R1-R48, 4 architecture options |
| 31CD / 31CE | Option A HF-derived real-activation proxy | the source of `cos_low`, `cos_sub`, `delta_cos`, `MAE_low`, `MAE_sub`, `MAE_delta`, `memory_margin_bytes` metadata cross-referenced in the sample manifest's per-layer entries |
| 31CF-S / 31CF-S2 | exact Q4_K_M GGUF runtime activation | the source of the no-modification capture path and the per-layer sha256 + shape metadata |
| 31BZ / 31CA | 1.5B synthetic-X aggregate freeze | the source of the per-family Q4 budget, the per-family memory margin, the per-family SDIR byte counts |
| 31BF / 31AJ | STATIC_ARTIFACT_SCHEMA.md v1.0 | the canonical offline schema that v1.1.0 extends; the v1.0 conventions (canonical_d_out_d_in orientation, etc.) are preserved in 31CI |
| `bundle_manifest.py` | ManifestLoader class | the existing loader that 31CI does NOT replace; 31CI's validator is a sibling module that validates the v1.1.0-specific safety invariants |
| `phase31x_manifest_runtime.py` | standalone tensor harness runtime primitives | the source of the canonical cosine, encode_sdir, decode_sdir helpers; 31CI does NOT modify these |

### 1.4 Out of scope (forbidden by user's prompt)

- ✗ no model load
- ✗ no Q2_K binary artifact generation
- ✗ no SDIR binary artifact generation
- ✗ no raw activation array generation
- ✗ no runtime-loadable blob creation
- ✗ no runtime loader implementation
- ✗ no `~/llama.cpp/` source modification
- ✗ no `~/llama.cpp/` rebuild
- ✗ no generation
- ✗ no inference-quality test
- ✗ no model file commit
- ✗ no compiled binary commit
- ✗ no raw tensor data commit
- ✗ no Q2_K/SDIR blob commit

---

## 2. Planning / implementation inputs reviewed

The following committed evidence and source were reviewed as planning / implementation inputs (per the user's prompt):

### 2.1 Committed evidence

- **`SOURCE_OF_TRUTH.md`** (SOT Section 0 line 11 explicitly lists 31CI as the **primary recommended** next phase after 31CH; the SOT forecast scope — v1.1.0 manifest schema validator + sample metadata-only manifest for 1.5B Qwen2.5-Instruct Q4_K_M that cross-references 31BZ/31CA/31CF-S2 result JSONs, schema-only, no binary generation, no model load, no runtime loader — matches the user's spec exactly)
- **`docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md`** (the v1.1.0 schema design that 31CI implements: 9 runtime safety invariants, 7 legacy-sidecar exclusion glob patterns, 48 loader validation rules, 4 loader architecture options, 9 categories of validation)
- **`docs/STATIC_ARTIFACT_SCHEMA.md`** (v1.0 canonical offline schema, the v1.1.0 extends it; per-family memory budget conventions; canonical_d_out_d_in orientation; metrics conventions)
- **`docs/CORRECTED_Q2K_POLICY_PACKAGE.md`** + `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` (the policy package 31CI validates against)
- **`docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md` Section 6 (48 loader validation rules R1-R48)** — the source-of-truth for the rule IDs that 31CI's validator implements
- **`src/results/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.json`** — the source of the per-family Q4 budget, the per-family memory margin, the per-family SDIR byte counts used in the sample manifest
- **`src/results/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.json`** — confirms the 1.5B evidence tier for `corrected_q2k_policy_v1`
- **`src/results/PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.json`** — the source of the exact Q4_K_M GGUF runtime activation metadata cross-referenced in the sample manifest's per-layer entries

### 2.2 Source reviewed

- **`src/bundle_manifest.py`** — the existing `ManifestLoader` class (which 31CI does NOT replace; 31CI's validator is a sibling module that validates the v1.1.0-specific safety invariants). The existing `SCHEMA_VERSION_ACCEPTED = ("1.0", "0.2.0")` and `Q2_K_FORMAT = "q2_k"` constants are referenced.
- **`src/phase31x_manifest_runtime.py`** — the existing standalone tensor harness runtime primitives (the validator does NOT modify these; 31CI's validator does NOT use them at runtime — only at validation time, the manifest's metadata references the 31BZ/31CA/31CF-S2 per-layer metrics)
- **`src/corrected_q2k_policy.py`** — the canonical `corrected_q2k_policy_v1` policy implementation (the validator's policy invariants match the policy parameters exactly)
- **`src/q2k_backend.py`** — the Q2_K encode/dequantize backends (the validator does NOT use these at runtime; only at validation time, the manifest's metadata references the byte counts that `q2k_backend` would produce)

### 2.3 NOT consulted (per user's prompt)

- ✗ no model binaries inspected (GGUF at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/...` and HF safetensors at `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/` were not opened; SHA-256 hashes from prior committed phases are the only metadata used)
- ✗ no `~/llama.cpp/` source inspection
- ✗ no new tensor artifacts generated
- ✗ no Q2_K or SDIR blob generation
- ✗ no raw activation capture

---

## 3. Validator implementation

The validator is implemented as a single-file Python module at **`src/phase31ci_manifest_validator.py`** (58 KB, lint-clean, dependency-free beyond Python stdlib).

### 3.1 Module structure

```
phase31ci_manifest_validator.py
├── Constants (v1.1.0 schema, policy invariants, excluded keys, hygiene patterns)
├── ValidationResult (structured result data class)
├── _find_excluded_keys (recursive key search for legacy sidecar exclusion)
├── _string_contains_forbidden_path (manifest hygiene check)
├── ManifestValidator (the main validator class)
│   ├── validate_all (entry point)
│   ├── _validate_schema_and_version (R1-R4)
│   ├── _validate_policy_invariants (R5-R9 + R-policy-residual-k-pct, R-policy-residual-alpha, R-policy-tensor-families)
│   ├── _validate_runtime_safety_invariants (R10-R13 + R-consumer-metadata-only, R-loader-integration-false)
│   ├── _validate_legacy_sidecar_exclusion (R14-R16)
│   ├── _validate_manifest_hygiene (R40-R45 + R-inline-payload)
│   ├── _validate_per_layer_per_family (R17-R27 + R-families-limited, R-ffn-down-no-sdir)
│   ├── _validate_memory_budget (R32-R34 + R-memory-numeric-nonneg)
│   └── _validate_provenance (R-provenance-*)
├── validate_manifest_file (convenience wrapper for JSON file)
├── SAMPLE_MANIFEST (the in-code metadata-only sample manifest)
├── _run_self_tests (the 10 in-code self-tests)
└── main (CLI entry point)
```

### 3.2 Implemented rules (50 total)

The validator implements 50 rules in 8 categories. Each rule corresponds to a specific requirement from the user's prompt and/or the 31CH v1.1.0 design.

| category | rule IDs | count | description |
|---|---|---|---|
| **schema_and_version** | R1, R2, R3, R4, R-rule-schema-version | 5 | `schema_version="1.1.0"`, `bundle_type="runtime_loadable_substitutive"`, v1.1.0-specific fields present, `source_model.quantization` in {Q2_K, Q4_K_M} |
| **policy_invariants** | R5, R6, R7, R8, R9, R-policy-invariants, R-policy-residual-k-pct, R-policy-residual-alpha, R-policy-tensor-families | 9 | `policy_name="corrected_q2k_policy_v1"`, `policy_version="1"`, `q2k_mode="corrected_ceil_per_row"`, `residual_families=["ffn_up","ffn_gate"]`, `ffn_down_residual_enabled=false`, `residual_k_pct=0.5`, `residual_alpha=1.0`, `tensor_families=["ffn_up","ffn_gate","ffn_down"]` |
| **runtime_safety_invariants** | R10, R11, R12, R13, R-consumer-metadata-only, R-loader-integration-false | 6 | `no_build_ffn_patch=true`, `no_legacy_prt_sidecar_entries=true`, `no_g_prt_sidecar_root_set_write=true`, `no_activation_capture_artifact_in_runtime_path=true`, `runtime_consumer.consumer_kind in {metadata_only, standalone_tensor_harness}`, `runtime_consumer.runtime_loader_integration=false` |
| **legacy_sidecar_exclusion** | R14, R15, R16 | 3 | recursive key search against 7 glob patterns (`prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*`, `g_apply_layer_*`, `shadow_*`, `pager_*`), `rejection_policy="fail_fast"`, `rejection_error_class="LegacySidecarManifestError"` |
| **manifest_hygiene** | R40, R41, R42, R43, R44, R45, R-inline-payload | 7 | no hardcoded operator paths, no raw activation paths, no model file references, no compiled binary paths, no `~/llama.cpp/` build artifacts, no `/tmp/` paths, no inline payload fields |
| **per_layer_per_family** | R17, R18, R19, R20, R21, R26, R27, R-families-limited, R-ffn-down-no-sdir | 9 | `layer_count` matches, layer ids are integers/zero-padded strings, expected shapes are present, byte counts are non-negative integers, families are limited to ffn_up/ffn_gate/ffn_down, ffn_down may not have sdir, `runtime_loadable` is a bool, sha256 fields are valid 64-char hex OR explicit placeholder, artifact filenames are relative (no absolute, no `..`) |
| **memory_budget** | R32, R33, R34, R-memory-numeric-nonneg | 4 | per-family byte counts are non-negative integers, total expected bytes, memory margin bytes, memory_positive_expected is a bool |
| **provenance** | R-provenance-created_by_phase, R-provenance-derived_from_phases, R-provenance-source_model_quantization, R-provenance-replay_w_ref_source, R-provenance-claim_boundary, R-provenance-forbidden_claims, R-provenance-valid_as_long_as | 7 | `created_by_phase="31CI"`, `derived_from_phases` is a non-empty list (with warnings if 31CH/31BZ are missing), `source_model.quantization="Q4_K_M"`, `replay_w_ref_source="Q4_K_M_GGUF_DEQUANTIZED"`, `claim_boundary` is a dict, `forbidden_claims` is a list, `valid_as_long_as` is a list |

### 3.3 Self-tests (10 total, all pass)

The validator includes 10 lightweight in-code self-tests that run via `python3 src/phase31ci_manifest_validator.py --self-test`:

| # | test | expected | result |
|---|---|---|---|
| 1 | `valid_sample_manifest_passes` | 0 errors | PASS (0 errors) |
| 2 | `ffn_down_residual_enabled_fails` | 1+ errors, including ffn_down_residual_enabled rule | PASS (1 error) |
| 3 | `legacy_prt_key_fails` | 1+ errors, including legacy sidecar key rule | PASS (1 error) |
| 4 | `hardcoded_media_path_fails` | 1+ errors, including hardcoded operator path rule | PASS (1 error) |
| 5 | `tmp_raw_activation_path_fails` | 9+ errors, including /tmp/ and raw activation path rules | PASS (9 errors) |
| 6 | `wrong_q2k_mode_fails` | 1+ errors, including q2k_mode rule | PASS (1 error) |
| 7 | `wrong_policy_version_fails` | 1+ errors, including policy_version rule | PASS (1 error) |
| 8 | `inline_payload_field_fails` | 1+ errors, including inline payload field rule | PASS (1 error) |
| 9 | `missing_required_field_fails` | 1+ errors, including forbidden_claims missing rule | PASS (1 error) |
| 10 | `metadata_only_placeholder_sha256_passes` | 0 errors (placeholder sha256s accepted in metadata_only manifests) | PASS (0 errors) |

**Total: 10/10 pass, 0 fail.** Exit code 0.

### 3.4 CLI usage

```bash
# Run self-tests
python3 src/phase31ci_manifest_validator.py --self-test

# Validate a manifest file
python3 src/phase31ci_manifest_validator.py --manifest path/to/manifest.json

# Validate a non-metadata-only manifest (placeholder sha256s will fail)
python3 src/phase31ci_manifest_validator.py --manifest path/to/manifest.json --no-metadata-only
```

The CLI prints the structured result as JSON to stdout, with keys: `passed`, `error_count`, `warning_count`, `errors`, `warnings`, `rules_checked`, `rules_checked_count`, `manifest_sha256`, `classification_suggestion`.

---

## 4. Sample manifest (metadata-only)

The metadata-only sample manifest is at **`src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json`** (10,732 bytes).

### 4.1 Sample manifest contents

The sample manifest is a **metadata-only** v1.1.0 runtime artifact manifest for Qwen2.5-1.5B-Instruct Q4_K_M, `corrected_q2k_policy_v1`, 3 distinct layer indices (0, 14, 27) × 3 families (ffn_up, ffn_gate, ffn_down) = 9 layer-family entries.

| field | value |
|---|---|
| `schema_version` | `1.1.0` |
| `bundle_type` | `runtime_loadable_substitutive` |
| `package_id` | `phase31ci-sample-qwen2.5-1.5b-instruct-q4_k-m-v1.1.0` |
| `metadata_only` | `true` |
| `source_model.model_name` | `qwen2.5-1.5b-instruct-q4_k_m` |
| `source_model.quantization` | `Q4_K_M` |
| `source_model.source_model_size_bytes` | `1,117,320,736` (matches the 31BS / 31BT / 31BU / 31BV / 31BX / 31BZ / 31CD / 31CE / 31CF-S / 31CF-S2 evidence) |
| `runtime_safety_invariants.policy_name` | `corrected_q2k_policy_v1` |
| `runtime_safety_invariants.policy_version` | `1` |
| `runtime_safety_invariants.q2k_mode` | `corrected_ceil_per_row` |
| `runtime_safety_invariants.residual_families` | `["ffn_up", "ffn_gate"]` |
| `runtime_safety_invariants.ffn_down_residual_enabled` | `false` |
| `runtime_safety_invariants.residual_k_pct` | `0.5` |
| `runtime_safety_invariants.residual_alpha` | `1.0` |
| `runtime_safety_invariants.tensor_families` | `["ffn_up", "ffn_gate", "ffn_down"]` |
| `runtime_safety_invariants.no_build_ffn_patch` | `true` |
| `runtime_safety_invariants.no_legacy_prt_sidecar_entries` | `true` |
| `runtime_safety_invariants.no_g_prt_sidecar_root_set_write` | `true` |
| `runtime_safety_invariants.no_activation_capture_artifact_in_runtime_path` | `true` |
| `runtime_consumer.consumer_kind` | `metadata_only` |
| `runtime_consumer.runtime_loader_integration` | `false` |
| `legacy_sidecar_exclusion.excluded_keys` | 7 glob patterns |
| `legacy_sidecar_exclusion.rejection_policy` | `fail_fast` |
| `legacy_sidecar_exclusion.rejection_error_class` | `LegacySidecarManifestError` |
| `layer_count` | `9` (3 distinct layer indices × 3 families) |
| `unique_layer_indices` | `[0, 14, 27]` |
| `hidden_size` | `1536` |
| `intermediate_size` | `896` |
| `layers[]` | 9 entries, one per (layer, family) pair |
| `memory_budget.expected_q2k_bytes_per_family` | `4,515,840` (matches 31BZ/31CA/31CF-S2 per-family Q4_K_M W_low byte count) |
| `memory_budget.expected_sdir_bytes_per_family_ffn_up` | `1,857,972` (matches 31BZ/31CA/31CF-S2) |
| `memory_budget.expected_sdir_bytes_per_family_ffn_gate` | `1,857,972` (matches 31BZ/31CA/31CF-S2) |
| `memory_budget.expected_sdir_bytes_per_family_ffn_down` | `0` (per policy no-ffn-down-residual) |
| `memory_budget.total_expected_bytes` | `3 × (4,515,840 + 1,857,972 + 1,857,972 + 0)` |
| `memory_budget.memory_margin_bytes` | `3 × (507,468 + 507,466 + 2,365,440)` (matches 31BZ/31CA/31CF-S2 per-family memory-positive margin) |
| `memory_budget.memory_positive_expected` | `true` |
| `artifact_creation_phase` | `31CH (planning) + 31BZ/31CA/31CF-S2 (evidence base)` |
| `consumed_by_phases` | `["31BN", "31BZ", "31CD", "31CE", "31CF-S", "31CF-S2"]` |
| `replay_w_ref_source` | `Q4_K_M_GGUF_DEQUANTIZED` |
| `created_by_phase` | `31CI` |
| `derived_from_phases` | `["31CH", "31BZ", "31CA", "31CF-S2"]` |
| `claim_boundary.allowed` | 3 allowed claims |
| `claim_boundary.forbidden` | 12 forbidden claims |
| `forbidden_claims` | 12 forbidden claims (matches claim_boundary.forbidden) |
| `valid_as_long_as` | 11 valid-as-long-as clauses |
| `replay_artifact` | `null` (no replay artifact referenced in 31CI's sample) |

### 4.2 Per-layer / per-family entries (9 entries)

For each (layer, family) pair, the sample manifest includes:
- `layer` (integer, 0/14/27)
- `family` (string, ffn_up/ffn_gate/ffn_down)
- `shape` (2-element list, canonical_d_out_d_in)
- `orientation` (`canonical_d_out_d_in`)
- `formats.W_low_format` (`q2_k` for all 9)
- `formats.residual_format` (`sdir_v1` for ffn_up/ffn_gate; N/A for ffn_down)
- `q2k_artifact_path` (relative path, e.g. `tensors/blk.0.ffn_up.q2_k.W_low`)
- `sdir_artifact_path` (relative path, e.g. `tensors/blk.0.ffn_up.residual.sdir`; omitted for ffn_down)
- `expected_q2k_bytes` (matches 31BZ/31CA/31CF-S2 per-family W_low byte count)
- `expected_sdir_bytes` (matches 31BZ/31CA/31CF-S2 per-family SDIR byte count; 0 for ffn_down)
- `per_layer_margin_bytes` (matches 31BZ/31CA/31CF-S2 per-family memory-positive margin)
- `q2k_sha256` (`metadata_only_placeholder` — no actual binary, no actual sha256)
- `sdir_sha256` (omitted for ffn_down; `metadata_only_placeholder` for ffn_up/ffn_gate)
- `runtime_loadable` (true for all 9)
- `loader_invocation_count` (1 for all 9)

### 4.3 No forbidden content

- ✗ no Q2_K bytes (only `expected_q2k_bytes` integer metadata)
- ✗ no SDIR bytes (only `expected_sdir_bytes` integer metadata)
- ✗ no raw activations (no `raw_x`, `raw_y`, `raw_r` fields)
- ✗ no model file content (no `.gguf` or `.safetensors` content)
- ✗ no absolute operator paths (all paths are relative; all model locations are in env-var redacted form)
- ✗ no `/tmp` paths
- ✗ no compiled binaries
- ✗ no generated tensor dumps
- ✗ no inline payload fields (no `raw_payload`, `w_low_raw`, `sdir_raw`, etc.)

---

## 5. Manifest validation result

When the sample manifest is validated against the validator, the result is:

```json
{
  "passed": true,
  "error_count": 0,
  "warning_count": 0,
  "rules_checked": 50,
  "classification_suggestion": "PASS_31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR_CLEAN"
}
```

**The sample manifest passes all 50 rules with 0 errors and 0 warnings.**

---

## 6. Pre-existing committed artifacts (UNCHANGED)

- 0.5B 31BN/31BM aggregate freeze (tag `phase31bn-corrected-q2k-full-aggregate-checkpoint` at `0304590c`) ✓
- 1.5B 31BU/31BV/31BX/31BZ/31CA aggregate freeze (tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` at `a433875a`) ✓
- 31CD Option A (`01c20b10`) ✓
- 31CE Option A (`82b1d91c`) ✓
- 31CF (`7f7e4154`) BLOCKED at source-modification implementation level (preserved) ✓
- 31CF-R (`6fdc8357`) PARTIAL design (preserved) ✓
- 31CF-R hotfix (`016bb0e8`) PASS_31CFR_HOTFIX_CLAIM_BOUNDARY_CLEAN (preserved) ✓
- 31CF-S (`16ef1a02`) PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN (preserved) ✓
- 31CF-S2 (`57cff9b8`) PASS_31CFS2_GGUF_RUNTIME_ACTIVATION_MATRIX_CLEAN (preserved) ✓
- 31CH (`c24202fd`) PASS_31CH_RUNTIME_ARTIFACT_FORMAT_PLAN_SELECTED (preserved) ✓
- `STATIC_ARTIFACT_SCHEMA.md` v1.0 (preserved) ✓
- `bundle_manifest.py` (preserved; 31CI does NOT replace it) ✓
- `phase31x_manifest_runtime.py` (preserved) ✓
- `q2k_backend.py` (preserved) ✓
- `corrected_q2k_policy.py` (preserved) ✓
- `corrected_q2k_policy_v1` package (UNCHANGED: parameters + version) ✓

---

## 7. Next allowed phase (per SOT Section 0 line 11, to be updated by 31CI)

After 31CI is committed, the SOT will list:
- **`Phase 31CJ`** — Artifact Writer Dry-Run Planning (recommended next step; planning-only; addresses how a future writer would generate the binary artifacts for the runtime-loadable bundle)
- Alternative: `Phase 31CK` — Loader Integration Planning (planning-only)
- Alternative: `Phase 31CG` — Larger Prompt/Token Sensitivity Planning (planning-only, unrelated to runtime artifact format)

All require explicit operator approval at entry. The agent does NOT proceed to any without a new request.

---

## 8. Files (31CI deliverables, prepared but not committed)

| file | status | role |
|---|---|---|
| `src/phase31ci_manifest_validator.py` | new, 58,841 B | Python validator module (lint-clean, stdlib-only) |
| `docs/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.md` | new, this file | human-readable implementation + design document |
| `src/results/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.json` | new, 10,402 B | metadata-only result summary (no raw arrays, no model references) |
| `src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json` | new, 10,732 B | metadata-only sample manifest (no Q2_K/SDIR bytes, no raw activations, env-var redacted form) |
| `SOURCE_OF_TRUTH.md` | modified | adds 31CI Section 3 entry + advances Section 0 (state label, next-allowed-phase to 31CJ, blockers, artifact status) |
| no Q2_K/SDIR blobs | (none created) | per artifact policy |
| no compiled binaries | (none created) | per artifact policy |
| no `~/llama.cpp/` modifications | (none) | per artifact policy |
| no model file inspection | (none) | per user's prompt |

---

## 9. Wall-clock and scope

- **planning + design review**: ~10 min
- **validator implementation**: ~10 min
- **self-tests**: ~5 min (10 tests, all pass)
- **sample manifest**: ~5 min
- **docs + JSON + SOT**: ~10 min
- **total wall-clock**: ~40 min, no model load, no capture, no generation, no compile, no link, no Q2_K/SDIR blob generation

---

## 10. Allowed claims (31CI PASSED)

1. A metadata-only v1.1.0 runtime artifact manifest validator was implemented for `corrected_q2k_policy_v1`.
2. The validator enforces `corrected_q2k_policy_v1` invariants, runtime safety invariants, legacy sidecar exclusion, manifest hygiene, per-layer/per-family shape metadata, memory-budget metadata, and provenance metadata.
3. A metadata-only sample manifest for Qwen2.5-1.5B `corrected_q2k_policy_v1` (3 layers × 3 families = 9 layer-family entries) was validated successfully.
4. 50 rules are checked by the validator, organized in 8 categories.
5. 10 in-code self-tests pass (1 valid sample + 9 negative tests + 1 bonus metadata-only placeholder test).

## 11. Forbidden claims (all upheld)

- ✗ no runtime loader exists (this is a metadata-only validator, not a loader)
- ✗ no runtime integration exists
- ✗ no artifacts were generated
- ✗ no Q2_K/SDIR blobs were generated
- ✗ no model files were loaded
- ✗ no model files were committed
- ✗ no raw activation arrays were created or committed
- ✗ no generation-quality claim
- ✗ no speedup claim
- ✗ no live-runtime memory savings claim
- ✗ no production readiness claim
- ✗ no claim that 31CI proves future runtime viability
- ✗ no claim that the validator is a full llama.cpp integration
- ✗ no claim that the sample manifest is a deployable artifact
- ✗ no compiled binary committed
- ✗ no `~/llama.cpp/` source modification
- ✗ no `~/llama.cpp/` rebuild
- ✗ no PRT sidecar re-activation
- ✗ no commit/push/tag without explicit operator approval (31CI stops at PRE-COMMIT REPORT)
- ✗ no policy parameter changes (`corrected_q2k_policy_v1` UNCHANGED)
- ✗ no `corrected_q2k_policy_v1` policy version bump (still v1)

---

## 12. How to use the validator

The validator is at `src/phase31ci_manifest_validator.py` and can be used in three ways:

### 12.1 Run the in-code self-tests

```bash
python3 src/phase31ci_manifest_validator.py --self-test
```

Expected output:
```
=== Phase 31CI self-test ===
passed: 10
failed: 0
total:  10
...
exit code: 0
```

### 12.2 Validate a manifest file from the command line

```bash
python3 src/phase31ci_manifest_validator.py --manifest path/to/manifest.json
```

Expected output (for the sample manifest):
```json
{
  "passed": true,
  "error_count": 0,
  "warning_count": 0,
  ...
  "classification_suggestion": "PASS_31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR_CLEAN"
}
```

### 12.3 Use the validator as a Python module

```python
import sys
sys.path.insert(0, "src")
from phase31ci_manifest_validator import ManifestValidator, validate_manifest_file

# From a Python dict
manifest = {...}
v = ManifestValidator(manifest, metadata_only=True)
result = v.validate_all()
print(f"passed={result.passed}, errors={result.error_count}, classification={result.classification_suggestion}")

# From a JSON file
result = validate_manifest_file("path/to/manifest.json", metadata_only=True)
print(f"manifest_sha256={result.manifest_sha256}, passed={result.passed}")
```

The validator returns a `ValidationResult` object with the structured fields specified in §3.4.
