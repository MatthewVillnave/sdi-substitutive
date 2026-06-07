# Phase 31CJ — Artifact Writer Dry-Run Planning

> **Planning-only phase. No validation, no generation, no inference, no sampling, no llama.cpp runtime integration, no llama.cpp modification or rebuild, no hook creation, no Q2_K/SDIR artifact generation, no model load, no raw activation generation, no runtime loader implementation, no artifact writer implementation, no model file commit, no raw tensor commit, no compiled binary commit, no commit/push/tag without explicit Matt approval.**

---

## 1. Goal and scope

31CJ plans the **first safe artifact writer dry-run path** for `corrected_q2k_policy_v1`. The purpose is to decide **how a future writer would simulate artifact creation without writing real Q2_K / SDIR blobs yet**. This is a *design* of the dry-run writer — not its implementation, not its execution.

This phase is **planning-only** — no artifacts are generated, no model is loaded, no manifest is mutated, no validator is re-run, no regression is touched. Only a design document, a planning JSON, and an SOT sync entry are produced. The actual dry-run writer implementation is reserved for a future phase (recommended: Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype), which would itself produce only a metadata-only JSON (no binary payloads).

### 1.1 What 31CJ answers

1. **What would the artifact writer read?** — the input contract (Section 3)
2. **What would the artifact writer plan to write?** — the output contract (Section 4)
3. **What metadata-only dry-run output should be produced?** — the output JSON shape (Section 4.1–4.2)
4. **How would the dry-run output be validated by the 31CI validator?** — Section 6 (validator integration)
5. **How would the writer avoid accidentally producing binary blobs?** — Section 5 (16 fail-safe rules) + Section 4 (the dry-run writer never imports `q2k_backend` or `phase31x_manifest_runtime`)
6. **What is the smallest safe implementation phase after 31CJ?** — Section 8 (Phase 31CL)

### 1.2 What 31CJ does NOT do (out of scope; forbidden by user's prompt)

- ✗ no model load (no opening Q4_K_M GGUF, no opening HF safetensors)
- ✗ no GGUF tensor payload inspection
- ✗ no Q2_K binary artifact generation
- ✗ no SDIR binary artifact generation
- ✗ no raw activation artifact generation
- ✗ no creation of runtime-loadable blobs
- ✗ no implementation of the actual artifact writer
- ✗ no implementation of a runtime loader
- ✗ no `~/llama.cpp/` modification
- ✗ no `~/llama.cpp/` rebuild
- ✗ no generation, no inference, no inference-quality test
- ✗ no commit of model files / HF cache / GGUF / safetensors / raw activations / build artifacts / Q2_K blobs / SDIR blobs / temp tensor dumps / llama.cpp source
- ✗ no compiled binary committed
- ✗ no tag created or pushed
- ✗ no commit, push, or tag without explicit operator approval (31CJ stops at PRE-COMMIT REPORT)

### 1.3 Relationship to prior phases

| phase | role | relationship to 31CJ |
|---|---|---|
| 31BN / 31BZ / 31CA | 0.5B / 1.5B corrected_q2k_policy_v1 aggregate freezes (synthetic Gaussian replay) | the source of canonical per-layer memory budgets the dry-run writer cross-references |
| 31CD / 31CE | Option A HF-derived real-activation proxy replay (1-pair, 9-pair) | shows that the corrected_q2k_policy_v1 directionally helps on real activations; the manifest's per-layer metrics can be cross-referenced |
| 31CF | exact Q4_K_M GGUF runtime activation capture, BLOCKED (DORMANT_SIDECAR_MACHINERY_INTERFERENCE) | identifies the legacy PRT/SDI sidecar risk; 31CH's 9 runtime safety invariants and 7 legacy-sidecar exclusion glob patterns defend against it; 31CJ inherits these defenses |
| 31CF-R | design-only hook-point recovery (PARTIAL) | the design-only precedent for 31CJ's planning-only discipline |
| 31CF-S / 31CF-S2 | exact Q4_K_M GGUF runtime activation capture (no `~/llama.cpp/` source modification) | the no-modification capture path; the source of per-family memory margins (ffn_up +507,468, ffn_gate +507,466, ffn_down +2,365,440) that the dry-run writer cross-references |
| 31CH | runtime artifact format / loader architecture planning | the v1.1.0 manifest schema, 9 runtime safety invariants, 7 legacy-sidecar exclusion glob patterns, 48 loader validation rules R1-R48, 4 loader architecture options, and file-naming proposal that 31CJ builds on; 31CH explicitly listed 31CJ as the next dry-run planning step after 31CI |
| 31CI | metadata-only runtime artifact manifest validator | the 50-rule validator the dry-run writer depends on (pre-plan and post-plan call sites); the 31CI sample manifest is the dry-run writer's reference input; 31CI's recommended-next-phase is 31CJ (confirmed) |
| 31S / 31T / 31X / 31Y / 31Z | the prior offline artifact writer + manifest + bundle runtime | the existing reference for the writer pattern; 31CJ plans a *separate* metadata-only dry-run writer, NOT a re-implementation of `src/artifact_write.py`; 31S-R is untouched |

### 1.4 Why a *dry-run* writer, not a real writer

The project's discipline has been **schema-first, validation-first, planning-first**. The 31CH planning phase established the v1.1.0 manifest schema before any writer existed. 31CI implemented and validated a *metadata-only* validator before any binary writer existed. 31CJ continues the same pattern: a *metadata-only dry-run* writer is designed before any real writer. The real writer (Option A's eventual successor that would call `q2k_backend.quantize_q2k_f32_to_bytes` and `phase31x_manifest_runtime.encode_sdir`) is gated behind an additional explicit phase (e.g. a future 31CL-RR or 31CM) with operator approval.

This preserves:

- the principle that the *only validated environment* is the standalone tensor harness
- the principle that binary blob generation is gated behind explicit operator approval
- the principle that the corrected_q2k_policy_v1 policy parameters remain UNCHANGED across phases
- the defense in depth provided by the 31CI validator (50 rules) + the 31CJ-designed fail-safes (16 rules) = 66 total rules across two layers

---

## 2. Planning inputs reviewed

The following committed evidence and source were reviewed as planning inputs (per the user's prompt):

### 2.1 Committed evidence (no new model loads, no new captures, no new artifacts)

- **`SOURCE_OF_TRUTH.md`** — Section 0 line 11 lists 31CJ as the primary recommended next phase after 31CI
- **`docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md`** — the v1.1.0 manifest schema design, 9 runtime safety invariants, 7 legacy-sidecar exclusion glob patterns, 48 loader validation rules R1-R48, 4 loader architecture options, file-naming proposal
- **`docs/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.md`** — the 50-rule metadata-only validator implementation, 10 self-tests, `ManifestValidator` class API surface, `main()` CLI
- **`docs/CORRECTED_Q2K_POLICY_PACKAGE.md`** — the `corrected_q2k_policy_v1` package (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR, k=0.5%, alpha=1.0, no ffn_down residual, 0.5B 31BN + 1.5B 31BZ/31CA evidence tiers)
- **`src/results/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.json`** — 1.5B 56-pair aggregate (n_layers=28, n_pairs=56, 28 of 28 layers PASS independently; per-layer memory budget 20,643,840 bytes; per-layer margin range +3,380,312 to +3,380,376 bytes)
- **`src/results/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.json`** — 1.5B aggregate freeze; policy parameters UNCHANGED
- **`src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json`** — 1-pair exact Q4_K_M GGUF runtime activation capture (PASSED; per-family memory margins: ffn_up +507,468 / ffn_gate +507,466 / ffn_down +2,365,440)
- **`src/results/PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.json`** — 9-pair exact Q4_K_M GGUF runtime capture (PASSED; 3 prompts × 3 layers × last prefill token; 9/9 pairs memory-positive, 9/9 delta_cos ≥ 0, 9/9 MAE_delta ≤ 0, 0 severe; mean_delta_cos=+0.001257, mean_MAE_delta=−0.004722)
- **`src/results/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.json`** — the v1.1.0 schema design summary, recommended-next-phase linkage to 31CI
- **`src/results/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.json`** — 50 rules, 10 self-tests, sample_manifest_path, next_allowed_phase.primary = "Phase 31CJ — Artifact Writer Dry-Run Planning"
- **`src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json`** — the 31CI-valid metadata-only sample manifest (3 layers [0, 14, 27] × 3 families [ffn_up, ffn_gate, ffn_down] = 9 layer-family entries; the dry-run writer's reference input; per-family Q2_K = 4,515,840 bytes, per-family SDIR = 1,857,972 bytes for ffn_up+ffn_gate, 0 for ffn_down)
- **`src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`** — canonical `corrected_q2k_policy_v1` parameters and memory budget reference

### 2.2 Committed source (read for API surface; not modified)

- **`src/phase31ci_manifest_validator.py`** — the `ManifestValidator` class, `validate_manifest_file()` function, `main()` CLI (`--manifest PATH`, `--no-metadata-only`, `--self-test`); exports `SCHEMA_VERSION_ACCEPTED='1.1.0'`, `POLICY_NAME_CANONICAL='corrected_q2k_policy_v1'`, `POLICY_VERSION_CANONICAL='1'`, `Q2K_MODE_CANONICAL='corrected_ceil_per_row'`, `RESIDUAL_FAMILIES_CANONICAL=['ffn_up','ffn_gate']`, `TENSOR_FAMILIES_CANONICAL=['ffn_up','ffn_gate','ffn_down']`, `FFN_DOWN_RESIDUAL_ENABLED_CANONICAL=False`, `Q4_BUDGET_FAMILY_REFERENCE_1_5B=688128` (NOTE: this constant name is misleading — it is the per-block Q2_K size, not the per-family Q4 budget; the canonical 1.5B per-family Q2_K size from the 31CI sample manifest is 4,515,840 bytes; the per-family Q4 budget is 6,881,280 per 31CF-S)
- **`src/corrected_q2k_policy.py`** — `POLICY_VERSION='corrected_q2k_policy_v1'` constant (the policy name is encoded in a constant named `POLICY_VERSION`); `package_version` matching logic
- **`src/q2k_backend.py`** — Q2_K encode/dequantize backends; referenced as the quantizer a *future REAL writer* would call; **31CJ does NOT call it**
- **`src/phase31x_manifest_runtime.py`** — standalone tensor-harness SDIR encode/decode; referenced as the encoder a *future REAL writer* would call; **31CJ does NOT call it**
- **`src/artifact_write.py`** — the prior Phase 31S-R OFFLINE artifact writer (loads/generates W_ref, packs nibble W_low, writes `.sdiw` and `.sdir` files to `data/phase31s_fixture/`); **31CJ does NOT extend or re-implement this file**; 31CJ plans a *separate* metadata-only dry-run writer; 31S-R is untouched
- **`src/artifact_load.py`** — the prior Phase 31S-R artifact loader; referenced for consistency only

### 2.3 Inputs NOT consulted (forbidden or not needed)

- model binaries (Qwen2.5-1.5B Q4_K_M GGUF at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/`, HF safetensors at `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/`) — NOT opened; SHA-256 hashes from prior committed phases are the only metadata used
- `~/llama.cpp/` source — NOT modified, NOT rebuilt, NOT inspected this phase
- any new tensor artifacts — none generated
- any `/tmp` scratch artifacts — none created

---

## 3. Four-role distinction

The 31CJ plan explicitly distinguishes four roles that have been conflated in some prior documentation:

| role | purpose | designed in | implemented in | invoked by 31CJ |
|---|---|---|---|---|
| **real artifact writer** | generates real Q2_K W_low + SDIR residual binary artifacts and writes them to disk in a runtime-loadable bundle layout | 31CJ explicitly **does NOT design** this | not yet implemented (would be a future 31CL-RR or 31CM with explicit operator approval) | **NO** |
| **metadata-only dry-run writer** | validates input v1.1.0 manifest, computes PLANNED artifact paths/byte counts/SHA-256 placeholders/memory accounting/validation steps, writes only a metadata-only JSON | **31CJ (this phase)** | not yet implemented (would be Phase 31CL) | **NO** (only designed) |
| **runtime loader** | loads a v1.1.0 manifest, validates it (R1-R48), and consumes artifact files at runtime | 31CH's Option A standalone loader | not yet implemented (Phase 31CK planning is the next planning step) | **NO** |
| **manifest validator** | validates a v1.1.0 manifest against the 50 rules in 8 categories | 31CI | already implemented as `src/phase31ci_manifest_validator.py` | **NO** (only referenced as a library) |

**The 31CJ plan designs ONLY the metadata-only dry-run writer.** It does not design the real writer, the runtime loader, or the manifest validator. The manifest validator is referenced as a library at two call sites; the real writer and the runtime loader are explicitly out of scope.

---

## 4. Dry-run writer input/output contracts

### 4.1 Input contract

A future dry-run writer (Phase 31CL) shall accept:

| input field | type | required | source | validation |
|---|---|---|---|---|
| input manifest path | env-var or relative path | yes | operator CLI / env-var | path must be in a non-model directory; absolute operator paths rejected |
| input manifest content | v1.1.0 metadata-only JSON | yes | a pre-existing 31CI-valid manifest (e.g. `src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json`) | must pass `validate_manifest_file(input_path, metadata_only=True)` with `error_count=0` and `warning_count=0` BEFORE planning; fail-closed if not passed |
| layer selection | list of int | no (default `[0, 14, 27]` to match 31CI sample scope) | operator CLI | must be subset of `[0..27]` for 1.5B (Qwen2.5-1.5B has 28 layers per 31BZ) |
| family selection | list of str | no (default `['ffn_up', 'ffn_gate', 'ffn_down']`) | operator CLI | must be subset of `['ffn_up', 'ffn_gate', 'ffn_down']` |
| artifact root path | env-var or relative path | no (default = env-var if set) | env-var | must NOT be inside the model directory; must NOT be inside `/tmp/` unless explicitly designated as scratch-no-commit |
| model identity metadata | dict | yes (from input manifest) | committed phase result JSONs (env-var redacted form) | must match a known committed phase's recorded identity |

**Canonical policy parameters (asserted by the writer, not configurable by the caller):**

| parameter | canonical value | source |
|---|---|---|
| `policy_name` | `corrected_q2k_policy_v1` | `src/phase31ci_manifest_validator.py:POLICY_NAME_CANONICAL` |
| `policy_version` | `1` | `src/phase31ci_manifest_validator.py:POLICY_VERSION_CANONICAL` |
| `q2k_mode` | `corrected_ceil_per_row` | `src/phase31ci_manifest_validator.py:Q2K_MODE_CANONICAL` |
| `residual_families` | `['ffn_up', 'ffn_gate']` | `src/phase31ci_manifest_validator.py:RESIDUAL_FAMILIES_CANONICAL` |
| `ffn_down_residual_enabled` | `false` | `src/phase31ci_manifest_validator.py:FFN_DOWN_RESIDUAL_ENABLED_CANONICAL` |
| `k_pct` | `0.5` | `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json:selected_policy.k_pct` |
| `alpha` | `1.0` | `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json:selected_policy.alpha` |

**Hard gates (writer hard-codes these; caller cannot override):**

- `dry_run = true` — writer asserts; refuses if false (R_DRY_01)
- `write_binary = false` — writer asserts; refuses if true (R_DRY_02)

### 4.2 Output contract

A future dry-run writer (Phase 31CL) shall produce **exactly one** output file:

**Path:** `src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json` (proposed)

**Format:** single JSON object, metadata-only, no binary payloads, no raw tensors, no model file paths except env-var redacted form

**Top-level fields (mandatory):**

| field | type | description |
|---|---|---|
| `phase` | str | `"31CL"` (set by the future writer) |
| `phase_full_name` | str | `"Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype (future)"` |
| `classification` | str | `"PASS_31CL_ARTIFACT_WRITER_DRY_RUN_CLEAN"` (or BLOCKED/PARTIAL per the writer's runtime check) |
| `dry_run` | bool | `true` (always) |
| `write_binary` | bool | `false` (always) |
| `input_manifest_path` | str | env-var or relative form; never absolute operator path |
| `input_manifest_validator_report` | dict | verbatim `to_dict()` of `validate_manifest_file(input_path, metadata_only=True)` |
| `policy_name` | str | `"corrected_q2k_policy_v1"` |
| `policy_version` | str | `"1"` |
| `q2k_mode` | str | `"corrected_ceil_per_row"` |
| `residual_families` | list | `["ffn_up", "ffn_gate"]` |
| `ffn_down_residual_enabled` | bool | `false` |
| `k_pct` | float | `0.5` |
| `alpha` | float | `1.0` |
| `model_identity` | dict | `source_model_name` / `source_model_quantization` / `source_model_sha256` from the input manifest (committed phase metadata, env-var redacted form) |
| `planned_artifact_package_root` | str | computed planned root, env-var form (e.g. `${SDI_ARTIFACT_DRYRUN_DIR}/qwen2.5-1.5b-instruct-q4_k_m/corrected_q2k_policy_v1`) |
| `planned_manifest_path` | str | planned runtime-loadable v1.1.0 manifest path (`manifest.dryrun.json`) — planned path only, not written |
| `planned_per_layer_directories` | list[str] | e.g. `["layers/layer_000/", "layers/layer_014/", "layers/layer_027/"]` (3 dirs in default scope; up to 28 in full scope) |
| `planned_artifacts_per_layer` | list[dict] | per-layer per-family plan: `layer_index`, `family`, `planned_q2k_filename`, `planned_q2k_byte_count`, `planned_sdir_filename`, `planned_sdir_byte_count`, `planned_q2k_sha256_placeholder`, `planned_sdir_sha256_placeholder` |
| `planned_total_files` | int | integer = `4 * len(layers)` for the default 3-family scope (q2k plan + sdir plan for ffn_up+ffn_gate + q2k plan for ffn_down); for the full 28-layer scope: `4 * 28 = 112` files |
| `planned_total_bytes` | int | integer = sum of `planned_q2k_byte_count + planned_sdir_byte_count` across all layer-family entries (using the 31CI sample manifest canonical per-family Q2_K = 4,515,840 bytes and per-family SDIR = 1,857,972 bytes for ffn_up+ffn_gate, 0 for ffn_down; per-layer Q4 budget = 20,643,840 bytes; per-family margins cross-referenced from 31CF-S: ffn_up +507,468 / ffn_gate +507,466 / ffn_down +2,365,440) |
| `planned_memory_accounting` | dict | the 31BZ per-layer Q4 budget (20,643,840 bytes) and the per-family margins cross-referenced from committed phases |
| `planned_validation_steps` | list[str] | `validate_manifest_file(planned_manifest_path, metadata_only=True)`, `sha256_byte_count_cross_check`, `memory_accounting_cross_check`, `legacy_sidecar_exclusion_walk` |
| `planned_output_manifest_validator_report` | dict | verbatim `to_dict()` of `validate_manifest_file(planned_manifest_path, metadata_only=True)` (post-plan, pre-write) |
| `output_written_path` | str | the absolute path of the written output JSON (env-var-expanded form for transparency; never hardcoded private path) |
| `no_binary_payloads` | bool | `true` |
| `no_raw_tensors` | bool | `true` |
| `no_model_file_paths_except_env_var_redacted_form` | bool | `true` |
| `no_tmp_paths` | bool | `true` |
| `no_hardcoded_operator_paths` | bool | `true` |
| `no_compiled_binaries` | bool | `true` |

**Per-layer per-family canonical values (from 31CI sample manifest, the writer's reference):**

| family | planned_q2k_byte_count | planned_sdir_byte_count | per_layer_margin_bytes |
|---|---|---|---|
| `ffn_up` | 4,515,840 | 1,857,972 | +507,468 |
| `ffn_gate` | 4,515,840 | 1,857,972 | +507,466 |
| `ffn_down` | 4,515,840 | 0 (no SDIR per policy) | +2,365,440 |

**Filename pattern (planned paths only, not created in 31CJ):**

- `blk.{N}.{family}.q2_k.W_low.plan.json` — per-layer per-family W_low plan
- `blk.{N}.{family}.residual.sdir.plan.json` — per-layer per-family SDIR plan (ffn_down: N/A)

---

## 5. Fail-safe rules (16 rules)

The 31CJ-designed dry-run writer is enforced by **16 fail-safe rules** in three enforcement layers (pre-plan, in-plan, post-plan):

| rule | enforcement layer | description |
|---|---|---|
| **R_DRY_01** | pre-plan | refuse if `dry_run` is not `true` (writer hard-coded; cannot be overridden by caller) |
| **R_DRY_02** | pre-plan | refuse if `write_binary` is not `false` (writer hard-coded; cannot be overridden by caller) |
| **R_DRY_03** | pre-plan | refuse if input manifest is not 31CI-valid (`validate_manifest_file(input_path, metadata_only=True)` must pass with `error_count=0` and `warning_count=0`) |
| **R_DRY_04** | pre-plan | refuse if output path is inside the model directory (e.g. under `$SDI_MODEL_DIR`) |
| **R_DRY_05** | pre-plan | refuse if output path contains `/tmp/` UNLESS the path is a dedicated scratch directory explicitly designated for non-committed planning output AND the writer tags the path with a `scratch_no_commit` flag (default deny) |
| **R_DRY_06** | pre-plan + in-plan | refuse if input manifest contains any legacy sidecar key (`prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*`, `g_apply_layer_*`, `shadow_*`, `pager_*`) at any nesting depth (matches 31CI rule category `legacy_sidecar_exclusion` R14-R16) |
| **R_DRY_07** | pre-plan | refuse if `ffn_down_residual_enabled` is `true` in the input manifest |
| **R_DRY_08** | pre-plan | refuse if `q2k_mode` is not `corrected_ceil_per_row` |
| **R_DRY_09** | pre-plan | refuse if `policy_name` is not `corrected_q2k_policy_v1` OR `policy_version` is not `1` |
| **R_DRY_10** | pre-plan | refuse if model identity (`source_model_name`, `source_model_quantization`, `source_model_sha256`) in the input manifest does not match a known committed phase's recorded identity (fail-closed: refuse and require operator override) |
| **R_DRY_11** | in-plan | refuse if any absolute operator path appears anywhere in the planned output (`planned_artifact_package_root`, `planned_manifest_path`, `planned_per_layer_directories`, `planned_artifacts_per_layer`, `output_written_path`) — use env-var or relative form only |
| **R_DRY_12** | in-plan | refuse if any inline payload array field appears in the planned output (no raw tensor bytes, no raw Q2_K bytes, no raw SDIR bytes, no raw activation bytes) — only metadata fields, byte counts, and placeholders |
| **R_DRY_13** | in-plan | refuse if any planned artifact filename has a forbidden extension (`.gguf`, `.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx`, `.npz`, `.npy`, `.raw`, `.x`, `.q2k`, `.sdir`, `.sdiw`) — the writer uses `.plan.json` and `.dryrun.json` extensions only (matches 31CI's `FORBIDDEN_ARTIFACT_EXTENSIONS_IN_PATHS`) |
| **R_DRY_14** | in-plan | refuse if any planned byte count is missing (`planned_q2k_byte_count`, `planned_sdir_byte_count`, `planned_total_bytes` must be present and non-negative integers) |
| **R_DRY_15** | post-plan | refuse if the planned output manifest fails the 31CI post-validation (`validate_manifest_file(planned_manifest_path, metadata_only=True)` must pass with `error_count=0` and `warning_count=0` BEFORE the writer exits; in strict mode, warnings also fail-closed; default mode is strict) |
| **R_DRY_16** | in-plan | refuse if any unknown tensor family appears in the planned output (must be subset of `['ffn_up', 'ffn_gate', 'ffn_down']`) |

**Total defense in depth: 50 rules in the 31CI validator + 16 rules in the 31CJ writer = 66 rules across two layers.**

---

## 6. Relationship to the 31CI validator

The 31CJ-designed dry-run writer uses the 31CI validator as a library at two call sites:

| call site | function call | enforced rule | purpose |
|---|---|---|---|
| pre-plan | `validate_manifest_file(input_path, metadata_only=True)` | R_DRY_03 | input must be 31CI-valid before planning starts |
| post-plan | `validate_manifest_file(planned_manifest_path, metadata_only=True)` | R_DRY_15 | planned output manifest must be 31CI-valid before writer exits |

**Validator reports embedded in output:** the `to_dict()` result of the post-plan `validate_manifest_file()` call is embedded verbatim in the output JSON under `input_manifest_validator_report` AND `planned_output_manifest_validator_report`.

**Fail-closed on warnings in strict mode:** strict mode (default) treats `warning_count > 0` as a fail-closed condition. Lenient mode allows warnings but the writer still records them. Lenient mode is opt-in via a CLI flag and is NOT the default.

**No validator bypass:** the writer MUST call the validator at both call sites; there is no code path that skips the validator. This is a code-level invariant, not a configuration option.

**Library import path:** `from phase31ci_manifest_validator import validate_manifest_file, ManifestValidator`

**Rule count summary:**

- 31CI validator: 50 rules across 8 categories
- 31CJ writer fail-safes: 16 rules
- Total: 66 rules across two layers

---

## 7. Proposed future file layout (NOT created in 31CJ)

This is a proposal for Phase 31CL's scratch directory. 31CJ does NOT create this directory. 31CL (the next implementation phase) would create it as a scratch directory, validate end-to-end, then either delete it before PRE-COMMIT REPORT or persist only the JSON plan files inside `src/results/` (operator decision at 31CL entry gate).

```
artifacts_dryrun/                                       (NOT created in 31CJ)
└── qwen2.5-1.5b-instruct-q4_k_m/                       (NOT created in 31CJ)
    └── corrected_q2k_policy_v1/                        (NOT created in 31CJ)
        ├── manifest.dryrun.json                        (planned v1.1.0 runtime-loadable manifest, metadata-only)
        ├── writer_plan.json                            (the 31CJ output JSON, metadata-only summary)
        └── layers/                                     (NOT created in 31CJ)
            ├── layer_000/                              (NOT created in 31CJ)
            │   ├── blk.0.ffn_up.q2_k.W_low.plan.json   (planned)
            │   ├── blk.0.ffn_up.residual.sdir.plan.json (planned)
            │   ├── blk.0.ffn_gate.q2_k.W_low.plan.json (planned)
            │   ├── blk.0.ffn_gate.residual.sdir.plan.json (planned)
            │   └── blk.0.ffn_down.q2_k.W_low.plan.json (planned; ffn_down has NO SDIR plan)
            ├── layer_014/                              (same 5-file pattern)
            └── layer_027/                              (same 5-file pattern)
```

**Extension policy:** writer emits ONLY `.plan.json` and `.dryrun.json` files. NEVER `.q2k`, `.sdir`, `.sdiw`, `.gguf`, `.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx`, `.npz`, `.npy`, `.raw`, `.x`.

**Directory creation in 31CJ: NOT created.** This is a proposal only.

---

## 8. Writer architecture options (4 options compared)

### 8.1 Option A — Metadata-Only Dry-Run Writer (RECOMMENDED)

**Description:** Validates the input manifest via 31CI validator; computes planned paths, byte counts, SHA-256 placeholders, memory accounting, validation steps; writes only JSON plan files. No binary blobs. Uses `src/phase31ci_manifest_validator.py` as a library. Hard-coded `dry_run=true` and `write_binary=false` gates (cannot be overridden).

**Pros:** Safest. Never touches model files, never invokes `quantize_q2k_f32_to_bytes` or `encode_sdir`, never writes binary. Output is fully diff-able and reviewable in a PR. Aligns with 31CH's Option A loader recommendation. Uses the 31CI validator as defense in depth.

**Cons:** Does not produce a real runtime-loadable bundle; a real writer is still needed later.

**Recommendation:** **RECOMMENDED default for 31CL.**

### 8.2 Option B — No-Write Reporter (ACCEPTABLE)

**Description:** Prints the plan to stdout / result JSON only; writes nothing to disk except the result JSON. The driest possible dry-run.

**Pros:** Safest of all options; nothing to clean up.

**Cons:** Output is less useful for review (no plan files to diff); future real writer still has to be designed separately.

**Recommendation:** **ACCEPTABLE** but less useful than Option A.

### 8.3 Option C — Real Artifact Writer with Dry-Run Flag (NOT RECOMMENDED)

**Description:** Implements the real Q2_K + SDIR binary writer but with a `dry_run` flag that, when set, only PRINTS the plan without writing binary. Closer to the final writer's code shape.

**Pros:** Less code duplication later; the dry-run flag and the real-write flag share most of the writer logic.

**Cons:** **RISKIER**: the binary-writing code path exists in the codebase. A future caller could pass `write_binary=true` (or omit it) and accidentally produce binary blobs. The 31CJ fail-safes (R_DRY_01, R_DRY_02) become the only line of defense, and the real `quantize_q2k_f32_to_bytes` and `encode_sdir` calls are still importable and callable in unit tests.

**Recommendation:** **NOT RECOMMENDED for 31CL.** Defer until after a real writer is explicitly approved and gated by a separate phase (e.g. a future 31CL-RR).

### 8.4 Option D — Runtime-Integrated Writer (REJECTED)

**Description:** Writer that lives inside a llama.cpp diagnostic / runtime integration. Out of scope for 31CJ.

**Pros:** Could produce runtime-ready artifacts in a single phase.

**Cons:** Re-introduces the 31CF BLOCKED risk (legacy PRT/SDI sidecar re-activation); bypasses the standalone tensor harness that is the project's only validated environment; requires `llama.cpp` modification that 31CH explicitly rejects.

**Recommendation:** **REJECTED.**

### 8.5 Selected recommendation

**Option A — Metadata-Only Dry-Run Writer (for Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype)**

---

## 9. Recommended next phase

### 9.1 Primary recommendation: Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype

**Scope:** Implement `src/phase31cj_writer_dry_run.py` (or `src/phase31cl_writer_dry_run.py`) as a Python module that realizes the 31CJ-designed input/output contract, fail-safes, and 31CI validator integration. Produces `src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json` (metadata-only). Uses env-vars / relative paths only.

**Hard scope (no exceptions):**

- NO model load
- NO Q2_K/SDIR blob generation
- NO raw activation generation
- NO runtime loader implementation
- NO llama.cpp modification
- NO compiled binary
- NO HF cache / model file commit
- NO raw tensor commit
- NO tag

**Size estimate:** ~25-35 KB Python module (lint-clean, Python stdlib only); ~10-15 KB output JSON; ~25-30 KB planning doc (31CJ's planning doc is already written at ~33 KB).

### 9.2 Alternatives

- **Phase 31CK** — Loader Integration Planning (planning-only)
- **Phase 31CG** — Larger Prompt/Token Sensitivity Planning (planning-only, unrelated to runtime artifact format)
- **Phase 31CJ-R** — Artifact Writer Dry-Run Planning Repair (only if 31CJ is incomplete or rejected post-merge)

### 9.3 All future phases require explicit operator approval at entry

No automatic progression. Each phase entry gate is operator-controlled per the project's standard discipline.

---

## 10. Claim boundaries

### 10.1 Allowed claims (only if 31CJ passes)

- An artifact writer dry-run architecture was planned for `corrected_q2k_policy_v1`, including a four-role distinction (real writer / metadata-only dry-run writer / runtime loader / manifest validator), an input contract (31CI-valid v1.1.0 manifest + canonical policy parameters + dry_run/write_binary hard gates), an output contract (metadata-only JSON with planned paths/byte counts/SHA-256 placeholders/memory accounting/validation steps), 16 fail-safe rules, 31CI validator integration at pre-plan and post-plan call sites with fail-closed strict mode, a proposed future file layout (`artifacts_dryrun/qwen2.5-1.5b-instruct-q4_k_m/corrected_q2k_policy_v1/`), four writer architecture options compared (A recommended, B acceptable, C not recommended, D rejected), and a recommended next implementation phase (Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype).
- The 31CJ plan is consistent with the 31CH v1.1.0 manifest schema, the 31CI 50-rule validator, the `corrected_q2k_policy_v1` package parameters, and the 31BZ/31CA/31CF-S/31CF-S2 committed per-layer memory budgets.
- The 31CJ plan does NOT design the real artifact writer, does NOT design the runtime loader, does NOT design the manifest validator (already designed in 31CI).

### 10.2 Forbidden claims

- no artifact writer exists (this is a planning-only phase)
- no dry-run writer exists yet (only a plan; implementation is 31CL)
- no real artifacts were generated
- no Q2_K / SDIR blobs were generated
- no model files were loaded
- no runtime loader exists
- no runtime integration exists
- no speedup claim
- no live-runtime memory savings claim
- no generation-quality claim
- no production readiness claim
- no claim that 31CJ proves future runtime viability
- no claim that 31CJ implements the writer (it only plans it)
- no claim that 31CJ replaces the prior 31S-R offline artifact writer (it plans a separate metadata-only dry-run variant; 31S-R is untouched)
- no claim that 31CJ modifies the 31CI validator (the validator is referenced as a library, not modified)
- no claim that 31CJ modifies `corrected_q2k_policy_v1` (the policy parameters are UNCHANGED)
- no claim that 31CJ runs the writer (no execution)
- no claim that 31CJ produces a runtime-loadable bundle (only a plan to produce a bundle)
- no PRT sidecar re-activation
- no llama.cpp source modification
- no llama.cpp rebuild
- no compiled binary committed
- no raw activation array committed
- no model file committed
- no generated Q2_K / SDIR blob committed
- no temp tensor dump committed
- no tag created or pushed
- no commit/push/tag without explicit operator approval (31CJ stops at PRE-COMMIT REPORT)

---

## 11. Success criteria

`PASS_31CJ_ARTIFACT_WRITER_DRY_RUN_PLAN_SELECTED` if **all** of the following are true:

1. SOT was read (this section was written after reading `SOURCE_OF_TRUTH.md`)
2. 31CJ was allowed (SOT Section 0 line 11 explicitly lists 31CJ as the primary recommended next phase after 31CI; user's explicit prompt provided the entry gate)
3. preflight regression passed (`PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true`)
4. planning inputs reviewed (Section 2)
5. dry-run writer input contract defined (Section 4.1)
6. dry-run writer output contract defined (Section 4.2)
7. fail-safe rules defined (Section 5; 16 rules)
8. relationship to 31CI validator defined (Section 6)
9. future package layout proposed (Section 7)
10. architecture options compared (Section 8; 4 options)
11. one recommended next phase selected (Section 9; Phase 31CL)
12. SOT updated (Section 12 in the planning doc, mirrored in the result JSON's `files_to_commit`)
13. post-edit regression passes (`PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`)
14. **and all of the following are true** (process constraints):
    - no model load occurred
    - no Q2_K/SDIR blobs were generated
    - no raw activations were generated
    - no runtime loader was implemented
    - no llama.cpp modifications occurred
    - no forbidden files introduced
    - no compiled binary / build artifact created
    - no hardcoded operator paths
    - no commit, push, or tag

---

## 12. Files to commit (when approved by operator)

1. `docs/PHASE31CJ_ARTIFACT_WRITER_DRY_RUN_PLANNING.md` — this file (~33 KB)
2. `src/results/PHASE31CJ_ARTIFACT_WRITER_DRY_RUN_PLANNING.json` — machine-readable planning summary (~33 KB)
3. `SOURCE_OF_TRUTH.md` — modified (SOT Section 0 state label updated for 31CJ; Section 0 line 11 next-allowed-phase updated; 31CJ entry added; recommended-next-phase updated to 31CL)

**No other files modified. No optional code created (the future 31CL writer is not yet implemented; 31CJ is planning only).**

---

## 13. References (paths in `~/sdi-substitutive/`)

- `SOURCE_OF_TRUTH.md` (Section 0 line 11, Section 0.A forbidden claims)
- `docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md`
- `docs/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.md`
- `docs/CORRECTED_Q2K_POLICY_PACKAGE.md`
- `src/phase31ci_manifest_validator.py` (the validator the writer depends on)
- `src/corrected_q2k_policy.py` (the policy module)
- `src/q2k_backend.py` (Q2_K backend; NOT called by the dry-run writer)
- `src/phase31x_manifest_runtime.py` (SDIR encoder; NOT called by the dry-run writer)
- `src/artifact_write.py` (prior 31S-R OFFLINE writer; NOT extended, NOT re-implemented)
- `src/artifact_load.py` (prior 31S-R loader; referenced for consistency only)
- `src/results/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.json` (per-layer memory budget)
- `src/results/PHASE31CA_1_5B_AGGREGATE_FREEZE_PACKAGE.json` (package freeze)
- `src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json` (per-family memory margins)
- `src/results/PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.json` (9-pair matrix)
- `src/results/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.json` (v1.1.0 schema design summary)
- `src/results/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.json` (50 rules, 10 self-tests)
- `src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json` (the writer's reference input)
- `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` (canonical policy parameters)
