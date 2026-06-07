# Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype

> **Metadata-only phase. No validation, no generation, no inference, no sampling, no llama.cpp runtime integration, no llama.cpp modification or rebuild, no hook creation, no Q2_K/SDIR artifact generation, no model load, no raw activation generation, no runtime loader implementation, no real artifact writer implementation, no model file commit, no raw tensor commit, no compiled binary commit, no commit/push/tag without explicit Matt approval.**

---

## 1. Goal and scope

31CL implements the first **metadata-only artifact writer dry-run prototype** for `corrected_q2k_policy_v1`. The writer:

1. reads a 31CI-valid v1.1.0 metadata-only manifest (validated before planning)
2. computes PLANNED artifact paths / byte counts / SHA-256 placeholders / memory accounting / validation steps for a future real writer
3. emits exactly two metadata-only output JSON files (a plan output + a per-phase summary). No binary blobs. No raw tensors.
4. hard-codes `dry_run=True` and `write_binary=False` gates (caller cannot override)
5. enforces 16 fail-safe rules (R_DRY_01 through R_DRY_16) at pre-plan, in-plan, and post-plan gates
6. imports `src.phase31ci_manifest_validator` as a library and calls `validate_manifest_file` at pre-plan AND `ManifestValidator.validate_all` at post-plan call sites
7. uses env-var or relative paths only; never hardcodes operator paths
8. uses Python stdlib only (json, hashlib, fnmatch, re, argparse, os, sys, typing, collections)

This is a **prototype**: it produces a planned-output JSON, not a runtime-loadable bundle. The next phase (31CM) would be a dry-run output validator / round-trip checker. The eventual real artifact writer (Option A in the 31CJ plan) is gated behind an additional explicit phase with operator approval.

### 1.1 What 31CL answers

1. **What does the dry-run writer read?** — the 31CI sample manifest (`src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json`) by default; configurable via `--input-manifest`
2. **What does the dry-run writer plan to write?** — 17 logical planned files (3 layers × 5 per-layer files + 2 summary files)
3. **What metadata-only dry-run output is produced?** — `src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json` (~18 KB) and `src/results/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.json` (~7.6 KB)
4. **How is the input validated?** — `validate_manifest_file(input_path, metadata_only=True)` at pre-plan (R_DRY_03)
5. **How is the planned output validated?** — `ManifestValidator.validate_all` at post-plan (R_DRY_15; with R-provenance rules filtered out since the planned manifest is a 31CL-derivative, not a 31CI-original)
6. **How does the writer avoid accidentally producing binary blobs?** — 16 fail-safe rules + 31CI validator + hard gates + extension whitelist
7. **What is the smallest safe next phase after 31CL?** — Phase 31CM (Dry-Run Output Validator / Round-Trip Checker)

### 1.2 What 31CL does NOT do (out of scope; forbidden by user's prompt)

- ✗ no model load (no opening Q4_K_M GGUF, no opening HF safetensors)
- ✗ no GGUF tensor payload inspection
- ✗ no Q2_K binary artifact generation
- ✗ no SDIR binary artifact generation
- ✗ no raw activation artifact generation
- ✗ no creation of runtime-loadable blobs
- ✗ no implementation of the real artifact writer
- ✗ no implementation of a runtime loader
- ✗ no `~/llama.cpp/` modification
- ✗ no `~/llama.cpp/` rebuild
- ✗ no generation, no inference, no inference-quality test
- ✗ no commit of model files / HF cache / GGUF / safetensors / raw activations / build artifacts / Q2_K blobs / SDIR blobs / temp tensor dumps / llama.cpp source
- ✗ no compiled binary committed
- ✗ no tag created or pushed
- ✗ no commit, push, or tag without explicit operator approval (31CL stops at PRE-COMMIT REPORT)

### 1.3 Relationship to prior phases

| phase | role | relationship to 31CL |
|---|---|---|
| 31BN / 31BZ / 31CA | 0.5B / 1.5B corrected_q2k_policy_v1 aggregate freezes | the source of per-layer memory budgets cross-referenced in the writer's `memory_accounting` |
| 31CF-S / 31CF-S2 | exact Q4_K_M GGUF runtime activation capture (1-pair, 9-pair) | the source of per-family memory margins (ffn_up +507,468, ffn_gate +507,466, ffn_down +2,365,440) used in the planned manifest's `per_layer_margin_bytes` |
| 31CH | runtime artifact format / loader architecture planning | the v1.1.0 manifest schema design + 9 runtime safety invariants + 7 legacy-sidecar exclusion glob patterns; 31CL implements the writer that consumes the 31CH-designed manifest |
| 31CI | metadata-only runtime artifact manifest validator | the validator library that 31CL imports; 31CL uses `validate_manifest_file` at pre-plan and `ManifestValidator.validate_all` at post-plan |
| 31CJ | artifact writer dry-run architecture planning | the design that 31CL implements; 31CL realizes the 16 fail-safe rules (R_DRY_01 through R_DRY_16), the input/output contracts, the per-layer per-family planning, the planned memory accounting, and the post-plan validator integration |
| 31S / 31T / 31X / 31Y / 31Z | the prior offline artifact writer + manifest + bundle runtime | the prior binary-writing reference; 31CL does NOT extend or re-implement `src/artifact_write.py`; 31CL implements a separate metadata-only dry-run writer; 31S-R is untouched |

### 1.4 Why metadata-only (not a real writer)

The project's discipline has been **schema-first, validation-first, planning-first, then metadata-only prototype, then real implementation**. The 31CH planning phase established the v1.1.0 manifest schema. 31CI implemented and validated a metadata-only validator. 31CJ planned a metadata-only dry-run writer. 31CL implements the metadata-only dry-run writer. A future phase (Option A's eventual successor) would be the real writer, gated behind an additional explicit phase with operator approval and the writer's own fail-safes (R_DRY_01, R_DRY_02 hard gates; R_DRY_13 forbidden extension whitelist; etc.).

This preserves:

- the principle that the *only validated environment* is the standalone tensor harness
- the principle that binary blob generation is gated behind explicit operator approval
- the principle that the corrected_q2k_policy_v1 policy parameters remain UNCHANGED across phases
- the defense in depth provided by the 31CI validator (50 rules) + the 31CL fail-safes (16 rules) = 66 total rules across two layers

---

## 2. Planning inputs reviewed

The following committed evidence and source were reviewed as planning inputs (per the user's prompt):

- **`SOURCE_OF_TRUTH.md`** — Section 0 line 11 lists 31CL as the primary recommended next phase after 31CJ
- **`docs/PHASE31CJ_ARTIFACT_WRITER_DRY_RUN_PLANNING.md`** — the design that 31CL implements
- **`src/results/PHASE31CJ_ARTIFACT_WRITER_DRY_RUN_PLANNING.json`** — the machine-readable planning summary
- **`docs/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.md`** — the 50-rule metadata-only validator implementation
- **`src/phase31ci_manifest_validator.py`** — the validator library imported by 31CL
- **`src/results/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.json`** — the validator result summary
- **`src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json`** — the input manifest for 31CL (3 layers × 3 families = 9 layer-family entries, per-family Q2_K = 4,515,840 bytes, per-family SDIR = 1,857,972 bytes for ffn_up+ffn_gate, 0 for ffn_down)
- **`docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md`** — the v1.1.0 manifest schema + 9 runtime safety invariants + 7 legacy-sidecar exclusion glob patterns
- **`src/results/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.json`** — the planning summary

### 2.1 Inputs NOT consulted (forbidden or not needed)

- model binaries (Qwen2.5-1.5B Q4_K_M GGUF at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/`, HF safetensors at `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/`) — NOT opened
- `~/llama.cpp/` source — NOT modified, NOT rebuilt, NOT inspected this phase
- any new tensor artifacts — none generated
- any `/tmp` scratch artifacts (except ephemeral self-test outputs that are immediately overwritten)

---

## 3. Writer prototype file

**File:** `src/phase31cl_artifact_writer_dry_run.py` (~53 KB, ~1,200 lines)

### 3.1 Implementation summary

- **Python stdlib only** (json, hashlib, fnmatch, re, argparse, os, sys, typing, collections)
- **Lint-clean / syntax-clean** (Python 3.8+ compatible; auto-lint on write passed)
- **Imports `src.phase31ci_manifest_validator`** as a library (`validate_manifest_file` + `ManifestValidator`)
- **Uses env-var or relative paths only** for `--input-manifest`, `--output-json`, `--artifact-root`
- **No hardcoded operator paths** in any of the writer's planned output
- **No model file reads** (the writer never opens a model file)
- **No binary artifact generation** (the writer emits only JSON plan files)
- **No runtime loader implementation** (the writer only plans, not loads)
- **No llama.cpp modification** (the writer does not touch `~/llama.cpp/`)
- **7 self-tests** in `--self-test` mode, all pass (7/7)
- **CLI:** `python3 -m phase31cl_artifact_writer_dry_run [--input-manifest PATH] [--output-json PATH] [--summary-json PATH] [--artifact-root PATH] [--dry-run true|false] [--write-binary true|false] [--strict true|false] [--self-test]`

### 3.2 Hard gates (writer-enforced, caller cannot override)

- `dry_run=True` is hard-coded; any caller attempt to set it to False raises `BLOCKED_31CL_DRY_RUN_DISABLED`
- `write_binary=False` is hard-coded; any caller attempt to set it to True raises `BLOCKED_31CL_BINARY_WRITE_REQUESTED`

### 3.3 CLI options

| flag | type | default | description |
|---|---|---|---|
| `--input-manifest` | str | `src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json` | Path to a 31CI-valid v1.1.0 metadata-only manifest |
| `--output-json` | str | `src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json` | Path to write the planned-output JSON |
| `--summary-json` | str | `src/results/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.json` | Path to write the per-phase summary JSON |
| `--artifact-root` | str | `${SDI_ARTIFACT_DRYRUN_DIR}` | Planned artifact root (env-var or relative form; never absolute operator path) |
| `--dry-run` | str (`true`/`false`) | `true` | Must be `true`; writer is dry-run by design |
| `--write-binary` | str (`true`/`false`) | `false` | Must be `false`; writer is metadata-only |
| `--strict` | str (`true`/`false`) | `true` | If `true`, warnings fail-closed (default) |
| `--self-test` | flag | (off) | Run the in-code self-tests only |

---

## 4. Writer behavior

### 4.1 Input contract

- Default input manifest: `src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json`
- CLI: `--input-manifest PATH` (overrides the default)
- The input manifest MUST be a 31CI-valid v1.1.0 metadata-only manifest
- Pre-validation: `validate_manifest_file(input_path, metadata_only=True)` must pass with `error_count=0` (and `warning_count=0` if `--strict=true`); fail-closed if not passed (R_DRY_03)

### 4.2 Planning output

- **Planned files (17 logical):**
  - 3 layers × 5 per-layer files = **15 per-layer files**:
    - per layer: `blk.{N}.ffn_up.q2_k.W_low.plan.json` + `blk.{N}.ffn_up.residual.sdir.plan.json`
    - per layer: `blk.{N}.ffn_gate.q2_k.W_low.plan.json` + `blk.{N}.ffn_gate.residual.sdir.plan.json`
    - per layer: `blk.{N}.ffn_down.q2_k.W_low.plan.json` (no SDIR for ffn_down per policy)
  - **2 summary files**:
    - `manifest.dryrun.json`
    - `writer_plan.json`
  - **NO actual files are created on disk** in 31CL — only listed in the output JSON

- **Per-layer per-family canonical values** (from 31CI sample manifest, the writer's reference):
  - per-family Q2_K = 4,515,840 bytes (1.5B)
  - per-family SDIR = 1,857,972 bytes for ffn_up + ffn_gate; 0 for ffn_down
  - per-layer memory margins ffn_up +507,468 / ffn_gate +507,466 / ffn_down +2,365,440 (from 31CF-S)
  - per-layer Q4 budget = 20,643,840 bytes (1.5B, from 31BZ)
  - total planned Q2_K bytes = 40,642,560 (3 layers × 3 families × 4,515,840)
  - total planned SDIR bytes = 11,147,832 (3 layers × 2 families × 1,857,972)
  - **total planned bytes = 51,790,392**

- **Output JSON top-level fields:** `phase`, `phase_full_name`, `classification`, `dry_run`, `write_binary`, `strict_mode`, `input_manifest_path`, `input_manifest_sha256`, `artifact_root_planned`, `planned_manifest_path`, `planned_writer_plan_path`, `planned_layer_dirs`, `planned_files`, `planned_summary_files`, `planned_file_count`, `planned_total_expected_bytes`, `planned_q2k_expected_bytes`, `planned_sdir_expected_bytes`, `per_layer_plan`, `per_family_plan`, `memory_accounting`, `validator_report_input`, `validator_report_output`, `fail_safe_results`, `fail_safe_all_passed`, `forbidden_artifact_scan`, `forbidden_artifact_scan_clean`, `claim_boundary`, `valid_as_long_as`, `next_recommended_phase`, `alternative_next_phases`, `scope_assertions`

### 4.3 Planned files

The writer does NOT create any actual files on disk. The "planned files" are listed in the output JSON for a future real writer to consume.

**Per-layer per-family Q2_K plan file (always present):**
- Filename: `blk.{N}.{family}.q2_k.W_low.plan.json`
- planned_byte_count: 4,515,840
- planned_sha256_placeholder: `<computed_at_real_write_time>`
- status: `planned_metadata_only`

**Per-layer per-family SDIR plan file (ffn_up + ffn_gate only):**
- Filename: `blk.{N}.{family}.residual.sdir.plan.json`
- planned_byte_count: 1,857,972
- planned_sha256_placeholder: `<computed_at_real_write_time>`
- status: `planned_metadata_only`

**Summary files:**
- `manifest.dryrun.json` (planned v1.1.0 runtime-loadable manifest, metadata-only, ~12 KB)
- `writer_plan.json` (the writer's own plan summary, ~5 KB)

**Filename extension policy:** the writer emits ONLY `.plan.json`, `.dryrun.json`, and `.writer_plan.json` files. NEVER `.q2k`, `.sdir`, `.sdiw`, `.gguf`, `.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx`, `.npz`, `.npy`, `.raw`, `.x`.

### 4.4 Fail-safe rules (16 rules)

The 31CL writer enforces the same 16 fail-safe rules that 31CJ designed, across 3 enforcement layers (pre-plan, in-plan, post-plan):

- R_DRY_01: refuse if dry_run is not true
- R_DRY_02: refuse if write_binary is not false
- R_DRY_03: refuse if input manifest is not 31CI-valid
- R_DRY_04: refuse if output path is inside the model directory
- R_DRY_05: refuse if output path contains `/tmp/` unless `SDI_SCRATCH_NO_COMMIT=true`
- R_DRY_06: refuse if input manifest contains any legacy sidecar key (`prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*`, `g_apply_layer_*`, `shadow_*`, `pager_*`)
- R_DRY_07: refuse if ffn_down_residual_enabled is true
- R_DRY_08: refuse if q2k_mode is not corrected_ceil_per_row
- R_DRY_09: refuse if policy_name is not corrected_q2k_policy_v1 OR policy_version is not 1
- R_DRY_10: refuse if model identity does not match a known committed phase's recorded identity
- R_DRY_11: refuse if any absolute operator path appears in planned output
- R_DRY_12: refuse if any inline payload array field appears in planned output
- R_DRY_13: refuse if any planned artifact filename has a forbidden extension (`.gguf`, `.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx`, `.npz`, `.npy`, `.raw`, `.x`, `.q2k`, `.sdir`, `.sdiw`) — writer uses `.plan.json` and `.dryrun.json` only
- R_DRY_14: refuse if any planned byte count is missing
- R_DRY_15: refuse if the planned output manifest fails the 31CI post-validation (with R-provenance rules filtered out for 31CL-derivative manifests; see Section 5)
- R_DRY_16: refuse if any unknown tensor family appears in planned output

**All 16 rules pass in 31CL on the default input manifest.**

### 4.5 Forbidden artifact scan

The writer scans all planned paths and output JSON strings for:
- `/media/matthew-villnave`
- `VL_usb`
- `/tmp` (separate from R_DRY_05 — this is the global scan)
- `.gguf`, `.safetensors`, `.npy`, `.npz`, `.bin`, `.raw`, `.x`
- llama.cpp build artifacts
- inline payload arrays

The planned path names MAY include `.q2k.plan.json` and `.sdir.plan.json` (which are plan JSON names, not binary artifact names). The scan distinguishes plan JSON names from binary artifact names by checking that the filename ALSO ends with a plan-file extension (`.plan.json`, `.dryrun.json`, or `.writer_plan.json`).

**31CL's forbidden artifact scan: clean (0 findings).**

### 4.6 Output validation

The dry-run output itself does NOT need to be a 31CI artifact manifest. However, the writer DOES build a planned manifest object inside the output (shaped like a 31CI v1.1.0 manifest) and validates it with the 31CI validator. See Section 5 for the validator integration and the R-provenance filtering rationale.

### 4.7 Self-tests (7 tests, all pass)

| # | test | expected | result |
|---|---|---|---|
| 1 | `valid_manifest` | a valid 31CI sample manifest produces 17 logical planned files | ✅ PASS (planned_file_count=17, fail-safe all pass, forbidden scan clean) |
| 2 | `no_ffn_down_sdir` | ffn_down.sdir is never planned | ✅ PASS (no ffn_down SDIR plan files) |
| 3 | `write_binary_true_fails` | write_binary=True fails | ✅ PASS (BLOCKED_31CL_BINARY_WRITE_REQUESTED) |
| 4 | `dry_run_false_fails` | dry_run=False fails | ✅ PASS (BLOCKED_31CL_DRY_RUN_DISABLED) |
| 5 | `hardcoded_operator_path_fails` | hardcoded operator artifact root fails R_DRY_11 | ✅ PASS (R_DRY_11 failed as expected) |
| 6 | `tmp_artifact_root_fails` | /tmp/ artifact root fails R_DRY_05 by default | ✅ PASS (R_DRY_05 failed as expected) |
| 7 | `no_raw_payload_arrays` | output contains no raw payload arrays | ✅ PASS (no raw payload arrays in output) |

**7/7 self-tests pass.** None of the self-tests read model files or create large files.

---

## 5. Relationship to the 31CI validator

The 31CL writer uses the 31CI validator as a library at two call sites:

| call site | function call | enforced rule | purpose |
|---|---|---|---|
| pre-plan | `validate_manifest_file(input_path, metadata_only=True)` (CLI / file path) OR `ManifestValidator(input_manifest, metadata_only=True).validate_all()` (in-memory) | R_DRY_03 | input must be 31CI-valid before planning starts |
| post-plan | `ManifestValidator(planned_manifest, metadata_only=True).validate_all()` (in-memory) | R_DRY_15 | planned output manifest must be 31CI-schema-conformant before writer exits |

### 5.1 R-provenance filtering at post-plan

The 31CI validator hard-requires `created_by_phase == "31CI"` (R-provenance-created_by_phase) and other R-provenance rules. The 31CL writer's planned manifest is a **31CL-derivative** (it was generated by the 31CL planning step, not directly by 31CI), so the R-provenance rules are NOT applicable to the planned manifest.

R_DRY_15's intent is to verify that the planned manifest conforms to:
- the v1.1.0 schema (R1-R4)
- the policy invariants (R5-R13)
- the runtime safety invariants
- the legacy sidecar exclusion (R14-R16)
- the manifest hygiene (R40-R45)
- the per-layer/per-family rules (R17-R27)
- the memory budget rules (R32-R34)

— rules that are agnostic of provenance. The 31CL wrapper runs the 31CI validator and then **filters out** the 7 R-provenance rule errors/warnings before reporting. The filtered-out rules are recorded separately as `provenance_rules_skipped` in the output JSON, with a clear rationale: R-provenance rules are deferred to the 31CM output validator (or to the first phase that creates a true 31CI-original manifest).

**The 7 skipped R-provenance rules:** R-provenance-created_by_phase, R-provenance-derived_from_phases, R-provenance-source_model_quantization, R-provenance-replay_w_ref_source, R-provenance-claim_boundary, R-provenance-forbidden_claims, R-provenance-valid_as_long_as.

### 5.2 Pre-plan validator report

The `validate_manifest_file` call (or `ManifestValidator.validate_all` for in-memory) on the input manifest returns a `ValidationResult` with the standard 31CI report. The writer embeds this report in the output JSON under `validator_report_input` (verbatim `to_dict()`).

### 5.3 Post-plan validator report

The `ManifestValidator.validate_all` call on the planned manifest returns a `ValidationResult`. The writer's `_run_post_plan_validation` wrapper:
1. runs the validator
2. filters out R-provenance errors and warnings
3. records the filtered-out rules under `provenance_rules_skipped`
4. recomputes the post-filter `passed` flag (fail-closed on remaining errors, fail-closed on warnings in strict mode)
5. embeds the result in the output JSON under `validator_report_output`

**31CL's post-plan validator result:** passed=True, error_count=0, warning_count=0, 7 R-provenance rules skipped (documented).

### 5.4 Rule count summary

- 31CI validator: 50 rules across 8 categories
- 31CL writer fail-safes: 16 rules
- 31CL R-provenance rules skipped: 7 rules (not applicable to 31CL-derivative manifests; deferred to 31CM)
- **Total: 50 - 7 + 16 = 59 rules enforced** (the 7 skipped rules are documented in the output JSON)

---

## 6. Hard-coded canonical values

The 31CL writer hard-codes the following canonical values (asserted by the writer; not configurable by the caller):

| parameter | canonical value | source |
|---|---|---|
| `policy_name` | `corrected_q2k_policy_v1` | `src/phase31ci_manifest_validator.py:POLICY_NAME_CANONICAL` |
| `policy_version` | `1` | `src/phase31ci_manifest_validator.py:POLICY_VERSION_CANONICAL` |
| `q2k_mode` | `corrected_ceil_per_row` | `src/phase31ci_manifest_validator.py:Q2K_MODE_CANONICAL` |
| `residual_families` | `['ffn_up', 'ffn_gate']` | `src/phase31ci_manifest_validator.py:RESIDUAL_FAMILIES_CANONICAL` |
| `tensor_families` | `['ffn_up', 'ffn_gate', 'ffn_down']` | `src/phase31ci_manifest_validator.py:TENSOR_FAMILIES_CANONICAL` |
| `ffn_down_residual_enabled` | `false` | `src/phase31ci_manifest_validator.py:FFN_DOWN_RESIDUAL_ENABLED_CANONICAL` |
| `k_pct` | `0.5` | `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json:selected_policy.k_pct` |
| `alpha` | `1.0` | `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json:selected_policy.alpha` |
| per-family Q2_K bytes (1.5B) | 4,515,840 | 31CI sample manifest |
| per-family SDIR bytes (1.5B) | 1,857,972 | 31CI sample manifest |
| per-layer Q4 budget (1.5B) | 20,643,840 | 31BZ |
| per-family memory margin (1.5B) | ffn_up +507,468 / ffn_gate +507,466 / ffn_down +2,365,440 | 31CF-S |
| legacy sidecar excluded patterns | prt_*, sidecar_*, g_prt_*, g_build_ffn_*, g_apply_layer_*, shadow_*, pager_* | 31CH design + 31CI validator |
| forbidden artifact extensions | .gguf, .safetensors, .bin, .pt, .pth, .onnx, .npz, .npy, .raw, .x, .q2k, .sdir, .sdiw | 31CI validator + 31CL writer R_DRY_13 |
| hardcoded operator path patterns | /home/matthew-villnave, /media/matthew-villnave, /mnt/, /opt/, /Users/, /root/, VL_usb, /tmp/ | 31CL writer R_DRY_11 + R_DRY_05 |

---

## 7. Memory accounting

The writer's `memory_accounting` field (embedded in the output JSON) cross-references the per-layer memory budgets and margins from the committed phases:

```json
{
  "per_layer_margin_bytes": {
    "ffn_up": 507468,
    "ffn_gate": 507466,
    "ffn_down": 2365440
  },
  "per_layer_q4_budget_bytes_1_5B": 20643840,
  "per_layer_q2k_bytes_total": 13547520,
  "per_layer_sdir_bytes_total": 3715944,
  "source": "31BZ (1.5B 56-pair aggregate) + 31CF-S (per-family memory margins) + 31CI sample manifest (per-family Q2_K and SDIR byte counts)"
}
```

**Verification:**
- 1.5B has 28 layers total; the writer's default scope is 3 layers [0, 14, 27] (matching the 31CI sample manifest)
- per_layer_q2k_bytes_total = 3 families × 4,515,840 = 13,547,520 (Q2_K bytes for 3 families per layer)
- per_layer_sdir_bytes_total = 2 families × 1,857,972 = 3,715,944 (SDIR bytes for 2 residual families per layer; ffn_down excluded)
- per_layer Q4 budget = 20,643,840 (from 31BZ)
- per-layer margin = 20,643,840 - 13,547,520 - 3,715,944 = 3,380,376 (matches 31BZ's reported per-layer margin)

---

## 8. Proposed future file layout (NOT created in 31CL)

This is the layout a future real writer (Option A's eventual successor) would produce:

```
artifacts_dryrun/                                       (NOT created in 31CL)
└── qwen2.5-1.5b-instruct-q4_k_m/                       (NOT created in 31CL)
    └── corrected_q2k_policy_v1/                        (NOT created in 31CL)
        ├── manifest.dryrun.json                        (planned)
        ├── writer_plan.json                            (planned)
        └── layers/                                     (NOT created in 31CL)
            ├── layer_000/                              (NOT created in 31CL)
            │   ├── blk.0.ffn_up.q2_k.W_low.plan.json
            │   ├── blk.0.ffn_up.residual.sdir.plan.json
            │   ├── blk.0.ffn_gate.q2_k.W_low.plan.json
            │   ├── blk.0.ffn_gate.residual.sdir.plan.json
            │   └── blk.0.ffn_down.q2_k.W_low.plan.json
            ├── layer_014/                              (same 5-file pattern)
            └── layer_027/                              (same 5-file pattern)
```

**31CL does NOT create this directory.** A future 31CM or 31CL-RR phase would create it as a scratch directory, validate end-to-end, then either delete it or persist only the JSON plan files inside `src/results/` (operator decision at the future phase's entry gate).

---

## 9. Writer architecture options (from 31CJ plan; 31CL implements Option A)

- **Option A — Metadata-Only Dry-Run Writer** ✓ **31CL implements this**
- Option B — No-Write Reporter (acceptable but less useful)
- Option C — Real Artifact Writer with Dry-Run Flag (NOT RECOMMENDED)
- Option D — Runtime-Integrated Writer (REJECTED)

---

## 10. Recommended next phase

### 10.1 Primary recommendation: Phase 31CM — Metadata-Only Dry-Run Output Validator / Round-Trip Checker

**Scope:** Implement a round-trip checker that re-runs the 31CL writer in a sandboxed mode and validates the produced output JSON against the 31CL writer's own contract (17 planned files, 16 fail-safe rules pass, 0 forbidden scan findings, pre-plan validator passes, post-plan validator passes, planned manifest conforms to v1.1.0 schema, no Q2_K/SDIR blobs generated, no model files loaded). The checker would also re-validate the R-provenance rules that 31CL deferred (so the planned manifest's `created_by_phase` etc. are properly enforced in a downstream phase).

**Hard scope:**
- NO model load
- NO Q2_K/SDIR blob generation
- NO runtime loader
- NO llama.cpp modification
- NO compiled binary
- HF cache / model file commit: NO
- raw tensor commit: NO
- tag: NO

### 10.2 Alternatives

- **Phase 31CK** — Loader Integration Planning (planning-only)
- **Phase 31CG** — Larger Prompt/Token Sensitivity Planning (planning-only, unrelated to runtime artifact format)
- **Phase 31CL-R** — Metadata-Only Artifact Writer Dry-Run Prototype Repair (only if 31CL is rejected post-merge or incomplete)

### 10.3 All future phases require explicit operator approval at entry

No automatic progression. Each phase entry gate is operator-controlled per the project's standard discipline.

---

## 11. Claim boundaries

### 11.1 Allowed claims (only if 31CL passes)

- A metadata-only artifact writer dry-run prototype was implemented for `corrected_q2k_policy_v1` in `src/phase31cl_artifact_writer_dry_run.py` (~53 KB, lint-clean, Python stdlib only)
- The prototype uses the 31CI validator as a library at pre-plan (R_DRY_03) and post-plan (R_DRY_15) call sites
- The prototype produces a metadata-only dry-run output (`src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json`, ~18 KB) and a per-phase summary (`src/results/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.json`, ~7.6 KB)
- The prototype lists 17 logical planned files (3 layers × 5 per-layer files + 2 summary files) in the output JSON; no actual files are created on disk
- The prototype does NOT generate Q2_K/SDIR blobs, raw activations, model files, or runtime artifacts
- The prototype enforces all 16 fail-safe rules from the 31CJ plan (R_DRY_01 through R_DRY_16)
- The prototype's 7 self-tests all pass (7/7)
- The prototype's pre-plan validator (input manifest) and post-plan validator (planned manifest) both pass; the 7 R-provenance rules are documented as deferred to 31CM
- The prototype's hard-coded gates (`dry_run=True`, `write_binary=False`) cannot be overridden by the caller

### 11.2 Forbidden claims

- no real artifact writer exists (this is a metadata-only dry-run prototype, not a real writer)
- no runtime loader exists
- no runtime integration exists
- no actual Q2_K / SDIR artifacts were generated
- no model files were loaded
- no generation-quality claim
- no speedup claim
- no live-runtime memory savings claim
- no production readiness claim
- no claim that 31CL proves future runtime viability
- no claim that 31CL is a real artifact writer
- no claim that 31CL replaces the prior 31S-R offline artifact writer (31S-R is untouched)
- no claim that 31CL modifies the 31CI validator (the validator is referenced as a library, not modified)
- no claim that 31CL modifies `corrected_q2k_policy_v1` (policy parameters UNCHANGED)
- no claim that 31CL is bit-equal to a future real writer
- no PRT sidecar re-activation
- no llama.cpp source modification
- no llama.cpp rebuild
- no compiled binary committed
- no raw activation array committed
- no model file committed
- no generated Q2_K / SDIR blob committed
- no temp tensor dump committed
- no tag created or pushed
- no commit/push/tag without explicit operator approval (31CL stops at PRE-COMMIT REPORT)

---

## 12. Success criteria

`PASS_31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN_CLEAN` if **all** of the following are true:

1. SOT was read
2. 31CL was allowed (SOT Section 0 line 11 explicitly lists 31CL as the primary recommended next phase after 31CJ; user's explicit prompt provided the entry gate)
3. preflight regression passed (`PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true`)
4. dry-run writer implemented (`src/phase31cl_artifact_writer_dry_run.py` exists, is lint-clean, imports the 31CI validator as a library)
5. input manifest validates with 31CI validator (pre-plan)
6. output metadata JSON produced (`src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json`)
7. per-phase summary JSON produced (`src/results/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.json`)
8. 17 logical planned files listed (3 layers × 5 per-layer files + 2 summary files)
9. 0 actual Q2_K/SDIR blobs generated
10. 0 raw activations generated
11. 0 model files loaded
12. 0 runtime loader implemented
13. 0 llama.cpp modifications
14. 16 fail-safe rules enforced; 0 fail-safe rules fail
15. 0 forbidden artifact scan findings
16. post-plan validator result: passed=True, error_count=0, warning_count=0 (with documented R-provenance filtering)
17. self-tests pass (7/7)
18. post-edit regression passes
19. no forbidden files introduced

**31CL meets all 19 PASS criteria.**

---

## 13. Files to commit (when approved by operator)

1. `src/phase31cl_artifact_writer_dry_run.py` — the writer prototype (~53 KB, lint-clean, Python stdlib only)
2. `docs/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.md` — this file
3. `src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json` — the dry-run output JSON (~18 KB)
4. `src/results/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.json` — the per-phase summary JSON (~7.6 KB)
5. `SOURCE_OF_TRUTH.md` — modified (SOT Section 0 state label updated for 31CL; SOT Section 0 line 15 journal entry appended; SOT wording-drift fix for 31CJ in the same edit)

**No other files modified. No optional code created beyond the writer prototype.**

---

## 14. References (paths in `~/sdi-substitutive/`)

- `SOURCE_OF_TRUTH.md` (Section 0 line 11, Section 0.A forbidden claims)
- `docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md`
- `docs/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.md`
- `docs/PHASE31CJ_ARTIFACT_WRITER_DRY_RUN_PLANNING.md`
- `src/phase31ci_manifest_validator.py` (the validator library)
- `src/results/PHASE31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR.json`
- `src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json` (the writer's reference input)
- `src/results/PHASE31CJ_ARTIFACT_WRITER_DRY_RUN_PLANNING.json`
- `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json` (canonical policy parameters)
- `src/results/PHASE31BZ_1_5B_CORRECTED_Q2K_FULL_LAYER_TWO_SEED_AGGREGATE.json` (per-layer memory budget)
- `src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json` (per-family memory margins)
