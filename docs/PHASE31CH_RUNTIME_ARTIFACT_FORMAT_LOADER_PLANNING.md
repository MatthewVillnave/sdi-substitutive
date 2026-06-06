# Phase 31CH — Runtime Artifact Format / Loader Planning

> **Planning-only phase. No validation, no generation, no inference, no sampling, no llama.cpp runtime integration, no llama.cpp modification or rebuild, no hook creation, no Q2_K/SDIR artifact generation, no model inference beyond metadata inspection, no quality/performance/runtime/behavior claim, no commit/push/tag without explicit Matt approval.**

---

## 1. Goal and scope

31CH designs the **runtime artifact format** and **loader architecture** needed to turn `corrected_q2k_policy_v1` from a **standalone tensor-harness policy** (current state, per 31BN/31BZ/31CA) into a **future runtime-loadable representation**.

This phase is **planning-only** — no artifacts are generated, no loader is implemented, no model is loaded, no capture runs, no generation runs, no inference runs. Only design documents, a manifest schema, loader architecture options, and a recommended next implementation step are produced.

### 1.1 What 31CH answers

1. **What files would represent Q2_K low weights and SDIR residuals in a future runtime-loadable package?**
2. **What manifest schema would bind model identity, layer index, tensor family, policy version, shapes, hashes, and memory accounting into a single self-describing bundle?**
3. **How would a future loader find and validate artifacts?**
4. **How would a future runtime path consume artifacts without accidentally re-activating the legacy PRT/SDI sidecar machinery that caused the 31CF BLOCKED classification?**
5. **What is the minimal next implementation phase that is safe and well-scoped?**

### 1.2 Relationship to prior phases

| phase | role | relationship to 31CH |
|---|---|---|
| 31AJ | initial source-of-truth schema (v0.2.0) | ancestor; v1.0 supersedes it |
| 31BF | added ffn_gate family, formal v1.0 spec, metric conventions, path rules | the schema 31CH extends; 31CH adds **runtime safety invariants** that v1.0 lacked |
| 31CD / 31CE | Option A HF-derived real-activation proxy replay | consumer of the manifest's X capture (but the manifest can be replayed without X being captured) |
| 31CF / 31CF-R | hook design for exact Q4_K_M GGUF runtime capture | identified the **legacy PRT/SDI sidecar machinery** that 31CH must defend against; 31CH's runtime-safety invariants are derived from this risk |
| 31CF-S / 31CF-S2 | exact Q4_K_M GGUF runtime activation capture (no `~/llama.cpp/` source modification) | the **no-modification alternative path** that 31CH's loader-architecture options must be compatible with (the standalone C++ harness in `src/phase31cfs2_capture.cpp` demonstrates a working no-modification capture path that a runtime loader could potentially reuse) |
| 31S / 31T / 31X / 31Y / 31Z | offline artifact writer + manifest + bundle runtime | the **existing offline artifact format** that 31CH extends with runtime-loadable semantics; 31CH reuses the `bundle_manifest.ManifestLoader` class and the `STATIC_ARTIFACT_SCHEMA.md` v1.0 conventions |
| 31CC | real-activation capture planning | planning-only precedent; 31CH follows the same planning-only discipline (no execution, no commit without explicit approval) |

### 1.3 Out of scope (forbidden by user's prompt)

- ✗ no Q2_K binary artifact generation
- ✗ no SDIR binary artifact generation
- ✗ no `~/llama.cpp/` source modification
- ✗ no `~/llama.cpp/` build
- ✗ no activation capture (no model load, no forward pass, no `llama_decode`)
- ✗ no generation
- ✗ no inference-quality test
- ✗ no runtime loader implementation
- ✗ no runtime integration claim
- ✗ no speedup claim
- ✗ no live-runtime memory savings claim
- ✗ no model file commit
- ✗ no raw tensor data commit
- ✗ no compiled binary commit
- ✗ no generated weight/residual blob commit
- ✗ no commit, push, or tag without explicit Matt approval

---

## 2. Planning inputs reviewed

The following committed evidence and source were reviewed as planning inputs (per the user's prompt):

### 2.1 Committed evidence (no new model loads, no new captures, no new artifacts)

- **`docs/STATIC_ARTIFACT_SCHEMA.md`** (v1.0 canonical schema, 274 lines) — defines `.sdiw` format, `.sdir` format, `manifest.json` schema, metric conventions, memory budget rules, path rules, orientation contract, bundle types, forbidden claims
- **`src/bundle_manifest.py`** — `ManifestLoader` class with `validate_bundle()`, `validate_tensor_entry()`, `select_tensor(family, layer)`, `select_by_name(tensor_name)`, `load_sdiw()`, `load_sdir()`, `load_q2k()`, `Q2_K_FORMAT = "q2_k"` W_low format extension, `SCHEMA_VERSION_ACCEPTED = ("1.0", "0.2.0")`, `ALLOWED_FAMILIES = ("ffn_up", "ffn_gate", "ffn_down")`, `CANONICAL_ORIENTATION = "canonical_d_out_d_in"`
- **`src/phase31x_manifest_runtime.py`** — the standalone tensor harness runtime primitives (cosine, encode_sdir/decode_sdir, pack_wlow/unpack_wlow, etc.)
- **`src/corrected_q2k_policy.py`** — the canonical `corrected_q2k_policy_v1` policy implementation
- **`src/q2k_backend.py`** — Q2_K encode/dequantize backends
- **`docs/CORRECTED_Q2K_POLICY_PACKAGE.md` + `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`** — the policy package documentation
- **31BZ result** (`src/results/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.json` + `docs/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.md`) — 56/56 pairs, 1.5B Q4_K_M dequantized W_ref
- **31CA result** (`src/results/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.json` + `docs/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.md`) — 0.5B + 1.5B evidence tiers for `corrected_q2k_policy_v1`
- **31CD / 31CE** Option A results — HF-derived real-activation proxy replay metrics
- **31CF / 31CF-R** docs — the `DORMANT_SIDECAR_MACHINERY_INTERFERENCE` blocker that motivates 31CH's runtime safety invariants
- **31CF-S / 31CF-S2** results + source — the no-modification capture path that a future runtime could reuse
- **`docs/PHASE31M_INTEGRATION_FEASIBILITY_DESIGN.md`** — earlier integration feasibility design
- **`rescue/stale_phase31bh_31bj/`** — historical context (5 stale 31BH-R2/31BJ files; not consulted for content, only for context)

### 2.2 NOT consulted (per user's prompt)

- ✗ no model binaries inspected (GGUF at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/...` and HF safetensors at `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/` were not opened; their SHA-256 hashes from prior committed phases are the only metadata used)
- ✗ no new tensor artifacts generated
- ✗ no Q2_K or SDIR blob generation or comparison
- ✗ no `~/llama.cpp/` source inspection beyond what is already documented in 31CF-R
- ✗ no activation capture or replay

---

## 3. Artifact families (proposed for runtime-loadable bundles)

A **future runtime-loadable bundle** for `corrected_q2k_policy_v1` would contain the following artifact families. **No artifacts are generated in 31CH** — this section defines only the schema.

### 3.1 W_low artifacts (low-rank weights)

Two formats are supported, matching the existing `bundle_manifest.py` `Q2_K_FORMAT` extension:

| format | magic | description | schema | size formula |
|---|---|---|---|---|
| `sdiw_v1` | `b"SDIW"` (4 bytes) | packed nibble W_low with per-block fp16 scales (per `docs/STATIC_ARTIFACT_SCHEMA.md` v1.0) | header (24 bytes) + fp16 scales + packed nibble bytes | `packed = ceil(n / 2)`, `scales = (n // 32) × 2` |
| `q2_k` | (no magic; raw bytes) | raw llama.cpp Q2_K bytes (per `src/q2k_backend.py`) | raw bytes (no header) | `Q2_K_BYTES = (rows × cols) × (170 / 256)` (rounded to next 256-element superblock) |

For `corrected_q2k_policy_v1`, the canonical W_low format is `q2_k` with mode `corrected_ceil_per_row` (per `src/corrected_q2k_policy.py`). The `sdiw_v1` format is a fallback for environments where llama.cpp Q2_K is not available.

### 3.2 SDIR residual artifacts

One format (matching `docs/STATIC_ARTIFACT_SCHEMA.md` v1.0):

| format | description | schema | size formula |
|---|---|---|---|
| `sdir_v1` | sparse top-k residual, bitmap + fp16 values (per `encode_sdir` in `src/phase31x_manifest_runtime.py`) | block: 4-byte little-endian u32 bitmap + fp16 values for set bits in row-major order | `total_blocks × 4 + sum(n_nnz_per_block × 2)` |

For `corrected_q2k_policy_v1`, the canonical SDIR configuration is `k_percent = 0.5%`, `alpha = 1.0`, families = `[ffn_up, ffn_gate]` (ffn_down has **no** SDIR residual, per the policy).

### 3.3 Manifest JSON

A future runtime-loadable manifest is **schema v1.1.0** (extension of v1.0). It **must** include the runtime safety invariants (see §5) that v1.0 lacked.

The schema is fully specified in §4 below.

### 3.4 Per-layer index files (optional)

For large models (e.g. 1.5B has 28 layers × 3 families = 84 tensor entries; 7B has 32 layers × 3 = 96), a per-layer index file `artifacts/{model}/{policy}/layers/layer_{N:03d}/index.json` can be used to point to the family's three artifacts without scanning the full manifest. This is **optional** — for the 1.5B bundle, the master `manifest.json` is sufficient.

### 3.5 Model-level package manifest (optional)

A `bundle_manifest.json` at the bundle root that references the master `manifest.json` and adds bundle-level metadata (publisher, license, sha256 of master manifest, optional signature). **Optional** — only needed if the bundle will be redistributed as a single distributable artifact.

---

## 4. File naming proposal (deterministic, reproducible)

For a future runtime-loadable bundle, all paths are **relative to the bundle root directory** (per `STATIC_ARTIFACT_SCHEMA.md` v1.0 path rules). **No hardcoded private paths** (e.g. `/home/...`, `/media/...`, `C:\...`).

### 4.1 Proposed layout (for 1.5B Qwen2.5-Instruct, `corrected_q2k_policy_v1`)

```
artifacts/
└── qwen2.5-1.5b-instruct-q4_k_m/
    └── corrected_q2k_policy_v1/
        ├── manifest.json                         # master bundle manifest (schema v1.1.0)
        ├── bundle_manifest.json                  # optional package-level manifest
        ├── checksums.json                         # SHA-256 of every artifact file
        ├── memory_budget.json                    # aggregate memory accounting
        ├── runtime_consumer.json                 # runtime loader path + digest
        ├── tensors/
        │   ├── blk.0.ffn_up.q2_k.W_low           # W_low for layer 0, ffn_up (raw Q2_K)
        │   ├── blk.0.ffn_up.residual.sdir        # SDIR residual for layer 0, ffn_up
        │   ├── blk.0.ffn_gate.q2_k.W_low
        │   ├── blk.0.ffn_gate.residual.sdir
        │   ├── blk.0.ffn_down.q2_k.W_low        # ffn_down has NO .sdir
        │   ├── blk.1.ffn_up.q2_k.W_low
        │   ├── blk.1.ffn_up.residual.sdir
        │   ├── ...                                 # 28 layers × 3 families = 84 entries
        │   └── blk.27.ffn_down.q2_k.W_low
        └── README.md                              # human-readable bundle metadata
```

### 4.2 Naming rules

- `tensors/blk.{N}.{family}.q2_k.W_low` — raw Q2_K W_low for layer N, family = ffn_up/ffn_gate/ffn_down
- `tensors/blk.{N}.{family}.residual.sdir` — SDIR residual (omitted for ffn_down per `corrected_q2k_policy_v1`)
- `manifest.json` — required at the bundle root
- `bundle_manifest.json` — optional at the bundle root (only if the bundle is redistributed as a single distributable)
- `checksums.json` — required at the bundle root (SHA-256 of every other file)
- `memory_budget.json` — required at the bundle root (aggregate memory accounting)
- `runtime_consumer.json` — required at the bundle root (runtime loader path + digest, for safety)
- `README.md` — optional at the bundle root (human-readable metadata)

The naming is **deterministic** — given a model name and policy version, the bundle path is fully predictable. **This is only a proposal** — no files are created in 31CH.

---

## 5. Manifest schema (v1.1.0, runtime-loadable extension of v1.0)

The v1.1.0 manifest extends v1.0 with **runtime safety invariants** that are required for any future runtime-loadable use. v1.0 bundles remain valid for the standalone tensor harness (no migration required).

### 5.1 New top-level fields (v1.1.0 additions)

| field | type | required | description |
|---|---|---|---|
| `runtime_safety_invariants` | object | **yes** | The 9 invariants a future loader must enforce before consuming artifacts. See §5.2. |
| `runtime_consumer` | object | **yes** | Reference to the runtime loader that will consume this bundle. See §5.3. |
| `legacy_sidecar_exclusion` | object | **yes** | Explicit declaration that the bundle does NOT include any legacy PRT/SDI sidecar entries. See §5.4. |
| `replay_artifact` | object (optional) | no | Optional reference to a captured X activation (e.g. SDIX file from 31CF-S) for replay validation. **If present, the file MUST be at a relative path inside the bundle, and the loader MUST validate it is a read-only replay input, not a runtime input.** |
| `artifact_creation_phase` | string | yes | The phase that produced these artifacts (e.g. `31CI` for the future schema prototype, or `31BZ+31CA+31CF-S2` for the existing standalone-tensor evidence base). |
| `consumed_by_phases` | array of strings | yes | The list of phases that have consumed this bundle (e.g. `["31BN", "31BZ", "31CD", "31CE", "31CF-S", "31CF-S2"]`). A new `consumed_by_phases` entry requires a new bundle version. |

### 5.2 `runtime_safety_invariants` (new in v1.1.0)

The 9 invariants a future loader MUST enforce before consuming artifacts. If any invariant fails, the loader MUST refuse the bundle and return a typed error. These are derived from the 31CF BLOCKED `DORMANT_SIDECAR_MACHINERY_INTERFERENCE` blocker + the `corrected_q2k_policy_v1` policy.

```json
{
  "policy_name": "corrected_q2k_policy_v1",
  "policy_version": "1",
  "q2k_mode": "corrected_ceil_per_row",
  "residual_families": ["ffn_up", "ffn_gate"],
  "ffn_down_residual_enabled": false,
  "no_build_ffn_patch": true,
  "no_legacy_prt_sidecar_entries": true,
  "no_g_prt_sidecar_root_set_write": true,
  "no_activation_capture_artifact_in_runtime_path": true
}
```

| invariant | what it forbids |
|---|---|
| `policy_name == "corrected_q2k_policy_v1"` | forbids future drift to other policies; only the canonical 0.5B-validated + 1.5B-validated policy |
| `policy_version == "1"` | forbids future drift to unreviewed policy versions; requires a new bundle version per policy change |
| `q2k_mode == "corrected_ceil_per_row"` | forbids the historical `floor_flat` Q2_K mode (which truncates the final partial superblock) |
| `residual_families == ["ffn_up", "ffn_gate"]` | forbids ffn_down residuals (which would break memory margin +2,365,440 bytes) |
| `ffn_down_residual_enabled == false` | explicit redundant check |
| `no_build_ffn_patch == true` | declares the bundle does NOT include any patch to `llm_graph_context::build_ffn` (the source of the 31CF BLOCKED sidecar issue) |
| `no_legacy_prt_sidecar_entries == true` | declares the bundle does NOT include any `prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*` artifacts or metadata keys |
| `no_g_prt_sidecar_root_set_write == true` | declares the bundle does NOT include any code or linker workaround that writes to `g_prt_sidecar_root_set` |
| `no_activation_capture_artifact_in_runtime_path == true` | declares that any captured X (e.g. SDIX from 31CF-S) is a **replay-validation input only**, not a runtime input — the runtime must not depend on a specific X being present |

### 5.3 `runtime_consumer` (new in v1.1.0)

```json
{
  "consumer_kind": "standalone_tensor_harness",
  "consumer_path": "src/phase31x_manifest_runtime.py",
  "consumer_class": "ManifestLoader",
  "consumer_digest_sha256": "<future SHA256 of src/phase31x_manifest_runtime.py>",
  "consumer_min_version": "1.1.0"
}
```

- `consumer_kind` — what kind of consumer is intended. For 31CH, only `standalone_tensor_harness` is allowed. Future `runtime_loadable_loader` and `llama_cpp_diagnostic_loader` are placeholder values that the v1.1.0 loader MUST reject as not-yet-implemented.
- `consumer_path` — the canonical source path of the consumer.
- `consumer_class` — the canonical class name (e.g. `ManifestLoader`).
- `consumer_digest_sha256` — SHA-256 of the consumer's source file. Allows the future runtime to verify the consumer is the version it expects (defense-in-depth against malicious or out-of-date consumers).
- `consumer_min_version` — minimum consumer schema version required (e.g. `"1.1.0"`).

### 5.4 `legacy_sidecar_exclusion` (new in v1.1.0)

```json
{
  "excluded_keys": [
    "prt_*",
    "sidecar_*",
    "g_prt_*",
    "g_build_ffn_*",
    "g_apply_layer_*",
    "shadow_*",
    "pager_*"
  ],
  "rejection_policy": "fail_fast",
  "rejection_error_class": "LegacySidecarManifestError"
}
```

If a future loader finds any key matching `excluded_keys` in any part of the manifest (top-level, per-layer, per-family, or in the runtime_consumer sub-object), it MUST raise `LegacySidecarManifestError` and refuse the bundle. This is a **defense-in-depth** measure to prevent the dormant sidecar from being silently re-activated by a future bundle.

### 5.5 `per_layer` entry (extended from v1.0)

The v1.0 per-layer schema (per `docs/STATIC_ARTIFACT_SCHEMA.md`) is preserved. v1.1.0 adds:

| new field | type | required | description |
|---|---|---|---|
| `runtime_loadable` | bool | yes | `true` if this layer is intended for runtime loading; `false` if it is replay-validation-only (e.g. from 31BZ/31CA/31CF-S2). The loader MUST reject bundles where ALL layers are `runtime_loadable=false` (which would mean the bundle is replay-only and not a runtime artifact). |
| `loader_invocation_count` | integer | no | The number of times a future loader would invoke this layer's W_low + SDIR. For a runtime substitute, this is the expected number of `ffn_up` invocations per token (typically 1 per token per layer). For replay validation, this is the number of `forward_pass` invocations performed by the consumer. |

### 5.6 `checksums` (extended from v1.0)

The v1.0 checksums sub-object (`wlow`, `residual`) is preserved. v1.1.0 adds:

| new field | type | required | description |
|---|---|---|---|
| `wlow_sha256` | string | yes (if W_low is q2_k) | SHA-256 of the raw Q2_K W_low bytes (replaces `wlow` from v1.0 for clarity) |
| `residual_sha256` | string | yes (if SDIR exists) | SHA-256 of the raw SDIR residual bytes (replaces `residual` from v1.0 for clarity) |
| `manifest_sha256` | string | yes | SHA-256 of the manifest file itself (allows the loader to verify the manifest hasn't been modified) |

### 5.7 `runtime_loadable` bundle_type (new in v1.1.0)

The v1.0 `bundle_type` enum (per `STATIC_ARTIFACT_SCHEMA.md`) is extended with:

| `bundle_type` | description |
|---|---|
| `source_of_truth_regression` | (existing) regression test fixtures only |
| `ffn_up_substitutive` | (existing) FFN up-projection substitutive bundle |
| `ffn_down_substitutive` | (existing) FFN down-projection substitutive bundle |
| `full_mlp_substitutive` | (existing) full MLP (up+gate+down) substitutive bundle |
| `layer_substitutive` | (existing) full layer substitutive bundle |
| **`runtime_loadable_substitutive`** | **(new)** runtime-loadable bundle that meets all v1.1.0 safety invariants. The v1.1.0 loader is the only consumer allowed. |

### 5.8 `per_family` and per-layer validation (extended from v1.0)

The v1.0 per-layer/per-family validation (per `bundle_manifest.py` `validate_tensor_entry`) is preserved. v1.1.0 adds:

- The `formats.W_low_format` field MUST be either `sdiw_v1` or `q2_k` (per v1.0). The loader MUST reject any other format (e.g. `packed_nibble_v0.1` is allowed in v1.0 for backward compatibility but is not recommended for v1.1.0).
- The `orientation` field MUST be `"canonical_d_out_d_in"`. The loader MUST reject any other orientation.
- The `W_ref_Q4_budget_bytes` MUST match the model's per-family Q4 budget. For 1.5B, this is 6,881,280 bytes (per `STATIC_ARTIFACT_SCHEMA.md`); for 0.5B, this is 2,179,072 bytes.

### 5.9 `memory_budget` and aggregate validation

The v1.0 per-family memory margin (e.g. +507,468 bytes for ffn_up) is preserved. v1.1.0 adds aggregate validation:

- `total_substitutive_bytes` across all families per layer MUST be `< 3 × W_ref_Q4_budget_bytes` (i.e. the 3-family Q4 budget is preserved)
- `aggregate_memory_margin_bytes` across all layers MUST be `>= 0` (no bundle is over-budget)
- For the 1.5B bundle: aggregate margin = `(507,468 + 507,466 + 2,365,440) × 28 = 94,570,544 bytes` (memory-positive across the full 28-layer × 3-family tensor grid)

### 5.10 `forbidden_claims` (extended from v1.0)

The v1.0 forbidden-claims list is preserved (per `STATIC_ARTIFACT_SCHEMA.md`). v1.1.0 adds:

- ✗ no runtime integration exists (this is a planning-only phase)
- ✗ no artifacts were generated in 31CH
- ✗ no loader was implemented in 31CH
- ✗ no speedup claim
- ✗ no live-runtime memory savings claim
- ✗ no production readiness claim
- ✗ no claim that the planned loader will be bit-identical to the standalone tensor harness
- ✗ no claim that 31CH proves future runtime viability

### 5.11 `valid_as_long_as` clauses (new in v1.1.0)

The bundle remains valid as long as:

- `corrected_q2k_policy_v1` parameters (q2k_mode, k_percent, alpha, ffn_down_residual_enabled, residual_families) are unchanged. Any change requires a new policy version + new bundle version.
- The source GGUF model (e.g. Qwen2.5-1.5B-Instruct Q4_K_M at the SHA-256 captured in the manifest's `source_model_sha256` field) is unchanged. Any re-quantization or weight change invalidates the bundle.
- The `ManifestLoader` source code (`src/phase31x_manifest_runtime.py` + `src/bundle_manifest.py`) at the SHA-256 captured in `runtime_consumer.consumer_digest_sha256` is unchanged. Any consumer code change requires a new bundle version.
- The `consumer_min_version` is not decreased. Any version downgrade requires a new bundle version with explicit operator approval.
- No future bundle includes any `prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*`, `g_apply_layer_*`, `shadow_*`, or `pager_*` keys (per `legacy_sidecar_exclusion.rejection_policy`).

---

## 6. Loader validation rules (a future loader MUST enforce)

The v1.1.0 loader MUST enforce the following rules before consuming any artifact. **All rules are fail-fast**: any failure MUST raise a typed error and refuse the bundle.

### 6.1 Schema and version rules

| rule | check | failure mode |
|---|---|---|
| **R1** | `manifest.schema_version` ∈ `{"1.0", "1.1.0"}` | `SchemaVersionError` (loader only accepts 1.0 for the existing v1.0 standalone harness; 1.1.0 for runtime-loadable) |
| **R2** | `manifest.bundle_type` ∈ `{source_of_truth_regression, ffn_up_substitutive, ffn_down_substitutive, full_mlp_substitutive, layer_substitutive, runtime_loadable_substitutive}` | `BundleTypeError` |
| **R3** | For `bundle_type == "runtime_loadable_substitutive"`, all v1.1.0-specific fields are present (`runtime_safety_invariants`, `runtime_consumer`, `legacy_sidecar_exclusion`, `artifact_creation_phase`, `consumed_by_phases`) | `RuntimeLoadableFieldError` |
| **R4** | `manifest.source_model.quantization` ∈ `{Q2_K, Q4_K_M}` (per v1.0 + v1.1.0 conventions) | `QuantizationError` |

### 6.2 Policy invariant rules

| rule | check | failure mode |
|---|---|---|
| **R5** | `runtime_safety_invariants.policy_name == "corrected_q2k_policy_v1"` | `PolicyNameError` |
| **R6** | `runtime_safety_invariants.policy_version == "1"` | `PolicyVersionError` |
| **R7** | `runtime_safety_invariants.q2k_mode == "corrected_ceil_per_row"` | `Q2KModeError` (rejects `floor_flat` and other legacy modes) |
| **R8** | `runtime_safety_invariants.residual_families == ["ffn_up", "ffn_gate"]` (exact match) | `ResidualFamiliesError` |
| **R9** | `runtime_safety_invariants.ffn_down_residual_enabled == false` | `FFNDownResidualError` |
| **R10** | `runtime_safety_invariants.no_build_ffn_patch == true` | `BuildFFNPatchError` |
| **R11** | `runtime_safety_invariants.no_legacy_prt_sidecar_entries == true` | `LegacyPRTEntryError` |
| **R12** | `runtime_safety_invariants.no_g_prt_sidecar_root_set_write == true` | `GPRTSidecarWriteError` |
| **R13** | `runtime_safety_invariants.no_activation_capture_artifact_in_runtime_path == true` | `ActivationArtifactInRuntimePathError` |

### 6.3 Legacy sidecar exclusion rules

| rule | check | failure mode |
|---|---|---|
| **R14** | No key in the manifest (top-level, per-layer, per-family, runtime_consumer) matches the `excluded_keys` glob patterns from `legacy_sidecar_exclusion` (`prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*`, `g_apply_layer_*`, `shadow_*`, `pager_*`) | `LegacySidecarManifestError` (fail-fast) |
| **R15** | `legacy_sidecar_exclusion.rejection_policy == "fail_fast"` | `RejectionPolicyError` (must be `fail_fast`; no graceful fallback) |
| **R16** | `legacy_sidecar_exclusion.rejection_error_class == "LegacySidecarManifestError"` | `RejectionErrorClassError` |

### 6.4 Per-layer / per-family rules (extended from v1.0)

| rule | check | failure mode |
|---|---|---|
| **R17** | `layer_count` matches the number of `layers[]` entries | `LayerCountError` |
| **R18** | `tensor_families` ∈ `{"ffn_up", "ffn_gate", "ffn_down"}` (per v1.0 `ALLOWED_FAMILIES`) | `TensorFamilyError` |
| **R19** | Per-layer `shape` matches the GGUF tensor's shape (per `source_model` metadata) | `ShapeMismatchError` |
| **R20** | Per-layer `W_low_packed_bytes` (or `W_low_q2k_bytes` for q2_k format) matches the actual file size on disk | `FileSizeMismatchError` |
| **R21** | Per-layer `residual_bytes` matches the actual file size on disk (omitted for ffn_down) | `FileSizeMismatchError` |
| **R22** | Per-layer `checksums.wlow_sha256` and `checksums.residual_sha256` match the actual file SHA-256 on disk | `ChecksumMismatchError` |
| **R23** | Per-layer `W_ref_Q4_budget_bytes` matches the expected per-family Q4 budget (e.g. 6,881,280 bytes for 1.5B, 2,179,072 bytes for 0.5B) | `Q4BudgetMismatchError` |
| **R24** | Per-layer `orientation == "canonical_d_out_d_in"` | `OrientationError` |
| **R25** | Per-layer `formats.W_low_format` ∈ `{"sdiw_v1", "q2_k"}` | `WLowFormatError` |
| **R26** | Per-layer `runtime_loadable` field is present and is a boolean (v1.1.0 only) | `RuntimeLoadableFieldMissingError` |
| **R27** | At least one layer has `runtime_loadable == true` (otherwise the bundle is not runtime-loadable) | `NoRuntimeLoadableLayersError` |

### 6.5 Model identity rules

| rule | check | failure mode |
|---|---|---|
| **R28** | `source_model.model_name` matches the expected source model (e.g. `qwen2.5-1.5b-instruct-q4_k_m`) | `ModelIdentityError` |
| **R29** | `source_model.architecture` matches the expected architecture (e.g. `qwen2`, `ffn`) | `ModelIdentityError` |
| **R30** | `source_model.source_model_sha256` matches the actual SHA-256 of the source GGUF (if the loader is given a path to the source GGUF) | `ModelIdentityError` |
| **R31** | `source_model.source_model_size_bytes` matches the actual file size of the source GGUF (if the loader is given a path) | `ModelIdentityError` |

### 6.6 Memory budget rules (extended from v1.0)

| rule | check | failure mode |
|---|---|---|
| **R32** | Per-family `memory_margin_bytes >= 0` (per v1.0) | `MemoryMarginError` |
| **R33** | Aggregate `total_substitutive_bytes` across all layers × families is `<= aggregate_W_ref_Q4_budget_bytes` | `AggregateMemoryError` |
| **R34** | Aggregate `aggregate_memory_margin_bytes >= 0` | `AggregateMemoryError` |

### 6.7 Runtime consumer rules (new in v1.1.0)

| rule | check | failure mode |
|---|---|---|
| **R35** | `runtime_consumer.consumer_kind` ∈ `{"standalone_tensor_harness"}` (no `runtime_loadable_loader` until 31CI/31CK/...) | `UnsupportedConsumerKindError` |
| **R36** | `runtime_consumer.consumer_path` matches the expected consumer source path (e.g. `src/phase31x_manifest_runtime.py` or `src/bundle_manifest.py`) | `ConsumerPathError` |
| **R37** | `runtime_consumer.consumer_class` matches the expected consumer class name (e.g. `ManifestLoader`) | `ConsumerClassError` |
| **R38** | `runtime_consumer.consumer_digest_sha256` matches the actual SHA-256 of the consumer source file (defense-in-depth) | `ConsumerDigestError` |
| **R39** | `runtime_consumer.consumer_min_version` ∈ `{"1.0", "1.1.0"}` (no future version downgrades) | `ConsumerVersionError` |

### 6.8 Manifest hygiene rules (new in v1.1.0)

| rule | check | failure mode |
|---|---|---|
| **R40** | Manifest does NOT contain any hardcoded operator paths (e.g. `/home/...`, `/media/...`, `C:\...`, `~/...`) | `HardcodedPathError` (fail-fast) |
| **R41** | Manifest does NOT contain any raw activation file references (`.bin` from `/tmp/...`, `.npz`, `.npy`, `.trit`, `.sdiw` files in `replay_artifact`) | `RawActivationFileError` |
| **R42** | Manifest does NOT contain any model file references (no `model.safetensors` paths, no `.gguf` paths) | `ModelFileError` |
| **R43** | Manifest does NOT contain any compiled binary references (no `.so`, `.o`, `.a`, no `/usr/bin/llama-server` paths) | `CompiledBinaryError` |
| **R44** | Manifest does NOT contain any `~/llama.cpp/` build artifact references (no `build/`, no `llama-server`, no `libllama.so`) | `LlamaCppArtifactError` |
| **R45** | Manifest does NOT contain any `/tmp/` paths (raw activations are written to `/tmp` per artifact policy, but the manifest is committed; if a `/tmp/` path slips in, the manifest is not safe to commit) | `TempPathError` |

### 6.9 Replay artifact rule (new in v1.1.0)

| rule | check | failure mode |
|---|---|---|
| **R46** | If `replay_artifact` is present, its `path` MUST be a relative path inside the bundle. If `path` is absolute or `..`-rooted, the loader MUST reject. | `ReplayArtifactPathError` |
| **R47** | If `replay_artifact` is present, the `consumer_kind` MUST be `standalone_tensor_harness` (replay artifacts are NEVER consumed by a runtime loader) | `ReplayArtifactInRuntimeError` |
| **R48** | If `replay_artifact` is present, the file's `sha256` MUST match the manifest's recorded sha256 | `ReplayArtifactChecksumError` |

### 6.10 Summary

48 rules total (R1-R48). The rules fall into 8 categories: schema/version, policy invariants, legacy sidecar exclusion, per-layer/per-family, model identity, memory budget, runtime consumer, and manifest hygiene. **All 48 rules are fail-fast** — the loader refuses the bundle on the first violation.

The existing `bundle_manifest.py` `validate_tensor_entry` function already implements R17-R25 + R32 (the v1.0 subset). v1.1.0 adds R1-R16, R26-R31, R33-R48. **The future v1.1.0 loader can be implemented as an extension of `ManifestLoader`**, with new validation methods (`validate_safety_invariants`, `validate_legacy_sidecar_exclusion`, `validate_manifest_hygiene`, etc.) added to the existing class.

---

## 7. Loader architecture options (compared)

Four options for the future loader are considered. **None are implemented in 31CH** — only the trade-offs are documented.

### 7.1 Option A — Python artifact writer + standalone replay loader

**Description:** A new Python module (e.g. `src/phase31ch_artifact_loader.py`) that:
- Extends `bundle_manifest.ManifestLoader` with v1.1.0 validation methods
- Loads `.sdiw` / `.sdir` / `.q2_k.W_low` artifacts from a bundle directory
- Decodes them via `src/q2k_backend.dequantize_q2k_bytes_to_f32` and `src/phase31x_manifest_runtime.decode_sdir`
- Performs the standalone tensor harness MLP forward pass via `src/phase31x_manifest_runtime.cosine` and the existing replay pipeline

**Pros:**
- Easiest to implement (Python, reuses existing modules)
- Supports more offline validation (full SCHEMA_VERSION_ACCEPTED + R1-R48)
- No llama.cpp dependency for the loader itself
- Same code path as the existing standalone tensor harness (31BZ/31CA/31CF-S/31CF-S2)
- Preserves the no-modification invariant (no `~/llama.cpp/` source modification)
- Can be tested against the existing 31BZ/31CA/31CF-S2 metrics (delta_cos sign-pattern match)

**Cons:**
- Not a "runtime" in the production sense — it's a standalone tensor harness that runs offline
- No speedup claim (pure-Python, no kernel fusion)
- No live-runtime memory savings claim
- Cannot be used inside a real model serving stack (e.g. llama-server)

**Risk:** LOW. Pure extension of existing code; no new dependencies; no model load.

### 7.2 Option B — llama.cpp-side diagnostic loader

**Description:** A future phase (e.g. 31CF-S's "loader integration" extension) that registers a `cb_eval` callback inside llama.cpp's `llama_decode`, loads artifacts on demand, and uses them in place of (or alongside) the GGUF-loaded weights. This is a **diagnostic** path, not a production path.

**Pros:**
- Useful bridge between the offline harness and a future runtime
- Could capture / replay X activations inside the actual llama.cpp forward pass (similar to 31CF-S's `cb_eval` capture path)
- Could be used for live runtime validation without modifying the model's output

**Cons:**
- **HIGH SIDE-CAR RISK** — the `llm_graph_context::build_ffn` dormant PRT/SDI sidecar machinery could be accidentally re-activated by naive `cb_eval` registration
- Requires careful gating (CMake flag, env-var gates, `cb_func` filter) to prevent accidental capture
- Higher implementation risk than Option A
- Must follow the 31CF-R design's three-level gating pattern

**Risk:** MEDIUM-HIGH. The 31CF BLOCKED classification is in this risk category.

### 7.3 Option C — runtime integration loader

**Description:** A future production path that integrates the loader into a real model serving stack (e.g. llama-server), substituting the GGUF-loaded weights with the corrected Q2_K + SDIR W_low + SDIR at runtime.

**Pros:**
- If it works, could enable real production speedup / memory savings
- Closes the SDI-substitutive → production gap

**Cons:**
- **HIGHEST RISK** — touching a production model serving stack is a major change
- Requires extensive validation against the existing 31BZ/31CA/31CF-S2 metrics
- Requires production-grade testing (correctness, performance, edge cases, fallback behavior)
- Not for immediate implementation
- Would require a separate multi-phase implementation project, not a single phase

**Risk:** VERY HIGH. Not for immediate implementation. The 31CH planning only includes this as a future direction, not as a recommendation.

### 7.4 Option D — legacy sidecar reuse

**Description:** Reuse the existing `g_prt_*` / `prt_*` / `sidecar_*` machinery in `build_ffn` (per the 31CF BLOCKED diagnostic).

**Pros:**
- Could (in theory) leverage existing infrastructure
- Avoids re-implementing sidecar-compatible hooks

**Cons:**
- **DOES NOT SELECT.** The 31CF BLOCKED classification explicitly recommends AGAINST this option. The dormant PRT/SDI sidecar machinery is gated by compile- and runtime-flags (`g_prt_pager_enabled && g_prt_pager && (g_prt_sidecar_apply_enabled || g_prt_sidecar_true_injection_enabled)`) and the operator's `~/llama.cpp/` build has 0 PRT sidecar symbols (per the 31CF-R design's binary-level verification). Reusing this machinery would either:
  - Re-activate the dormant sidecar unintentionally
  - Compete with the sidecar's view-fetching
  - Capture the wrong tensor
- The `legacy_sidecar_exclusion` field (R11, R14-R16) is the explicit policy that 31CH's loader MUST enforce. Selecting Option D would directly violate this policy.
- The `~/llama.cpp/AGENTS.md` content-blocker policy also recommends AGAINST this option.

**Risk:** REJECTED. Do not select.

### 7.5 Selected recommendation: **Option A (with Option B as future extension)**

**Recommended next implementation phase:** `Phase 31CI — Runtime Artifact Schema Prototype / Metadata-Only Manifest Validator` (per the user's prompt).

`Phase 31CI` is **Option A's first concrete step** — it builds the v1.1.0 schema validator + sample metadata-only manifest, without generating any Q2_K or SDIR blobs, without loading any model, and without implementing a runtime loader. It validates the v1.1.0 schema against the existing 31BZ/31CA/31CF-S2 metadata where possible (the manifest's per-layer `cos_low`, `cos_sub`, `delta_cos`, `MAE_low`, `MAE_sub`, `MAE_delta`, `memory_margin_bytes` values can be cross-referenced with the committed result JSONs).

**Why Option A, not Option B, C, or D:**
- Option A is the **lowest-risk** path: no `~/llama.cpp/` modification, no model load, no runtime integration, no live-runtime claim
- Option A's v1.1.0 schema validator can be tested against the existing committed evidence (31BZ/31CA/31CF-S2 metrics) without re-running any capture
- Option A produces a **reusable, lint-clean Python module** that the future Option B (llama.cpp-side diagnostic loader) and Option C (runtime integration loader) can both build on
- Option A's 48 loader rules (R1-R48) provide a **comprehensive safety net** that catches the 31CF BLOCKED sidecar risk before any future runtime could be affected
- Option D is explicitly rejected (per 31CF BLOCKED)

**Why 31CI specifically (not 31CJ or 31CK or 31CG):**
- `Phase 31CI — Runtime Artifact Schema Prototype / Metadata-Only Manifest Validator` is **schema-only, no binary generation, no model load, no runtime loader** — the safest first step
- `Phase 31CJ — Artifact Writer Dry-Run Planning` would be a natural follow-up after 31CI, but is **planning-only** and not the immediate next step
- `Phase 31CK — Loader Integration Planning` would come after 31CJ (if needed) and would address the Option B / C bridge
- `Phase 31CG — Larger Prompt/Token Sensitivity Planning` is unrelated to the runtime artifact format (it's about Option A's prompt/layer sweep, not runtime loading)

**The recommended path is: 31CH (this phase, planning) → 31CI (schema prototype, schema-only) → 31CJ (writer dry-run planning) → 31CK (loader integration planning) → future Option B implementation (with explicit operator approval at each gate).**

---

## 8. Safety / artifact policy (for future phases)

This section defines what future phases may and may not commit, **based on the v1.1.0 manifest's `runtime_safety_invariants` + the 31CH artifact policy**.

### 8.1 Allowed future-phase commits

- ✓ `manifest.json` (the v1.1.0 manifest, schema-validated)
- ✓ `bundle_manifest.json` (optional package-level manifest)
- ✓ `checksums.json` (SHA-256 of every artifact file)
- ✓ `memory_budget.json` (aggregate memory accounting)
- ✓ `runtime_consumer.json` (runtime loader path + digest)
- ✓ `README.md` (human-readable bundle metadata)
- ✓ `tensors/*.q2_k.W_low` and `tensors/*.residual.sdir` (if a future 31CJ-style dry-run generates them, subject to the 31CH artifact policy: written to `/tmp/` or another operator temp path, deleted before PRE-COMMIT REPORT, SHA-256 recorded in the bundle's `checksums.json`)
- ✓ small schema-draft Python files (schema-only, validation-only, no binary generation)
- ✓ small docs files (planning + design documents)
- ✓ source code that generates/validates artifacts (e.g. the future 31CI schema validator, the future 31CJ writer dry-run)

### 8.2 Forbidden future-phase commits (unless explicitly approved)

- ✗ generated Q2_K binary blobs (in `/tmp/...` or in-repo, even with sha256)
- ✗ generated SDIR binary blobs
- ✗ raw activations (X, Y, R arrays)
- ✗ model files (`.gguf`, `.safetensors`, HF cache)
- ✗ compiled binaries
- ✗ `~/llama.cpp/` build artifacts
- ✗ temp tensor dumps
- ✗ any path under `/tmp/` in committed artifacts (raw activations go to `/tmp` but the manifest is committed; `/tmp` paths in the manifest would be unsafe)
- ✗ any hardcoded operator path (e.g. `/home/...`, `/media/...`, `~/...`) in committed artifacts

### 8.3 Optional `/artifacts/` directory (future)

A future phase (e.g. 31CJ dry-run) may create an `artifacts/` directory in the repo for **committed v1.1.0 manifests** (NOT the binary artifacts themselves). The `artifacts/` directory is a registry of bundles, not a registry of binary data. Each bundle's `manifest.json` references binary files at relative paths inside the bundle, but the binary files themselves are NOT committed to the repo (per the artifact policy in 8.2).

The `artifacts/` directory's `manifest.json` MUST itself be v1.1.0-conformant and pass all 48 loader rules (R1-R48). The loader's `validate_manifest_hygiene` method (per R40-R45) MUST be run on every `artifacts/` directory entry before commit.

---

## 9. Recommended next phase

**`Phase 31CI — Runtime Artifact Schema Prototype / Metadata-Only Manifest Validator`**

### 9.1 Scope of 31CI

- Build a v1.1.0 manifest schema validator (Python module, e.g. `src/phase31ch_artifact_schema_validator.py`)
- Write a sample metadata-only manifest (e.g. `src/results/PHASE31CH_SAMPLE_MANIFEST_V1_1_0.json` for 1.5B Qwen2.5-Instruct Q4_K_M) that includes:
  - `runtime_safety_invariants` (all 9 invariants)
  - `runtime_consumer` (pointing to the existing `bundle_manifest.ManifestLoader`)
  - `legacy_sidecar_exclusion` (all 7 excluded_keys)
  - `artifact_creation_phase: "31BZ+31CA+31CF-S2"` (the existing committed evidence base)
  - `consumed_by_phases: ["31BN", "31BZ", "31CD", "31CE", "31CF-S", "31CF-S2"]`
  - Per-layer entries cross-referenced with the 31BZ/31CA/31CF-S2 result JSONs (cos_low, cos_sub, delta_cos, MAE_low, MAE_sub, MAE_delta, memory_margin_bytes)
  - **NO** W_low binary blobs (the W_low paths reference `/dev/null` or are empty files for the schema prototype)
  - **NO** SDIR binary blobs
  - **NO** raw activation references
  - **NO** model file references
- Run the validator against the sample manifest and verify all 48 rules pass
- Update the SOURCE_OF_TRUTH.md to record the 31CI result (classification: `PASS_31CI_V1_1_0_SCHEMA_VALIDATED` if all 48 rules pass; `PARTIAL_31CI_*` if some rules fail)

### 9.2 What 31CI does NOT do

- ✗ no model load
- ✗ no Q2_K binary generation
- ✗ no SDIR binary generation
- ✗ no raw activation capture
- ✗ no runtime loader implementation
- ✗ no llama.cpp modification or rebuild
- ✗ no runtime integration

### 9.3 Alternative next phases

- `Phase 31CJ — Artifact Writer Dry-Run Planning` (planning-only; addresses how a future writer would generate the binary artifacts)
- `Phase 31CK — Loader Integration Planning` (planning-only; addresses Option B / C bridge)
- `Phase 31CG — Larger Prompt/Token Sensitivity Planning` (unrelated to runtime artifact format; addresses Option A's prompt/layer sweep)

### 9.4 Why 31CI (not 31CJ, 31CK, or 31CG)

- 31CI is the **safest first step** — schema-only, no binary generation, no model load
- 31CI **validates the v1.1.0 schema** end-to-end (all 48 rules) without touching any binary artifact or model
- 31CI produces a **reusable Python module** (the validator) that future phases (31CJ, 31CK) can build on
- 31CI's sample manifest is a **metadata-only** reference that can be cross-checked against the existing 31BZ/31CA/31CF-S2 result JSONs (the per-layer `cos_low`, `cos_sub`, `delta_cos`, `MAE_low`, `MAE_sub`, `MAE_delta`, `memory_margin_bytes` fields match the committed metrics)
- 31CI does NOT block any of the other future phases (31CG, 31CH-related extensions) — it's a strictly additive planning step

---

## 10. Allowed claims (31CH PASSED)

The following claims are accepted for 31CH:

1. **A runtime artifact format and loader architecture plan for `corrected_q2k_policy_v1` has been designed**, with:
   - Manifest schema v1.1.0 (extension of v1.0 with runtime safety invariants)
   - File naming proposal (`artifacts/{model}/{policy}/tensors/blk.{N}.{family}.{q2_k.W_low, residual.sdir}`)
   - 48 loader validation rules (R1-R48)
   - 4 loader architecture options compared (A: standalone, B: llama.cpp diagnostic, C: runtime integration, D: legacy sidecar reuse — REJECTED)
   - One recommended next step: `Phase 31CI`
   - Updated SOURCE_OF_TRUTH.md, post-edit regression clean

2. **No runtime integration exists.** This is a planning-only phase.

3. **No artifacts were generated.** This is a planning-only phase.

4. **No loader was implemented.** This is a planning-only phase.

## 11. Forbidden claims (all upheld)

- ✗ no runtime integration exists
- ✗ no artifacts were generated
- ✗ no loader was implemented
- ✗ no speedup claim
- ✗ no live-runtime memory savings claim
- ✗ no generation quality claim
- ✗ no production readiness claim
- ✗ no new model validation claim
- ✗ no raw activation claim
- ✗ no claim that 31CH proves future runtime viability
- ✗ no model files / HF cache / raw activation arrays / build artifacts / Q2_K blobs / SDIR blobs / temp tensor dumps / `llama.cpp` source committed
- ✗ no compiled binary committed
- ✗ no commit/push/tag without explicit operator approval (31CH stops at PRE-COMMIT REPORT)
- ✗ no `~/llama.cpp/` source modification
- ✗ no `~/llama.cpp/` rebuild
- ✗ no PRT sidecar re-activation
- ✗ no policy parameter changes (`corrected_q2k_policy_v1` parameters UNCHANGED)
- ✗ no `corrected_q2k_policy_v1` policy version bump (still v1)

---

## 12. Wall-clock and scope

- **planning-only phase**: ~10 min reading existing schema + planning inputs; ~15 min writing this design document; ~10 min writing the result JSON; ~5 min updating SOT
- **total wall-clock**: ~40 min, no model load, no capture, no generation, no compile, no link, no Q2_K/SDIR blob generation

---

## 13. Files (31CH deliverables, prepared but not committed)

| file | status | role |
|---|---|---|
| `docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md` | new, this file | human-readable planning document |
| `src/results/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.json` | new | machine-readable planning summary (metadata only, no raw arrays, no model references) |
| `SOURCE_OF_TRUTH.md` | modified | adds 31CH Section 3 entry + advances Section 0 (state label, next-allowed-phase, blockers, artifact status) |
| no Q2_K/SDIR blobs | (none created) | per artifact policy |
| no compiled binaries | (none created) | per artifact policy |
| no `~/llama.cpp/` modifications | (none) | per artifact policy |
| no model file inspection | (none) | per user's prompt |

---

## 14. Pre-existing committed artifacts (UNCHANGED)

- 0.5B 31BN/31BM aggregate freeze (tag `phase31bn-corrected-q2k-full-aggregate-checkpoint` at `0304590c`) ✓
- 1.5B 31BU/31BV/31BX/31BZ/31CA aggregate freeze (tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` at `a433875a`) ✓
- 31CD Option A (`01c20b10`) ✓
- 31CE Option A (`82b1d91c`) ✓
- 31CF (`7f7e4154`) BLOCKED at source-modification level (preserved) ✓
- 31CF-R (`6fdc8357`) PARTIAL design (preserved) ✓
- 31CF-R hotfix (`016bb0e8`) PASS_31CFR_HOTFIX_CLAIM_BOUNDARY_CLEAN (preserved) ✓
- 31CF-S (`16ef1a02`) PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN (preserved) ✓
- 31CF-S2 (`57cff9b8`) PASS_31CFS2_GGUF_RUNTIME_ACTIVATION_MATRIX_CLEAN (preserved) ✓
- `STATIC_ARTIFACT_SCHEMA.md` v1.0 (preserved; 31CH extends it via v1.1.0 design) ✓
- `bundle_manifest.py` (preserved; 31CH plans to extend it) ✓
- `corrected_q2k_policy_v1` package (UNCHANGED: parameters + version) ✓
- `phase31x_manifest_runtime.py` (UNCHANGED) ✓
- `q2k_backend.py` (UNCHANGED) ✓

---

## 15. Next allowed phase (per SOT Section 0 line 11, to be updated by 31CH)

After 31CH is committed, the SOT will list:
- **`Phase 31CI`** — Runtime Artifact Schema Prototype / Metadata-Only Manifest Validator (recommended next step, schema-only, no binary generation)
- Alternative: `Phase 31CJ` — Artifact Writer Dry-Run Planning
- Alternative: `Phase 31CK` — Loader Integration Planning
- Alternative: `Phase 31CG` — Larger Prompt/Token Sensitivity Planning (unrelated to runtime artifact format)

All require explicit operator approval at entry. The agent does NOT proceed to any without a new request.
