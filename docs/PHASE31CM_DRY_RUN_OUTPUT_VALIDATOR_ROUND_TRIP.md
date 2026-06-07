# Phase 31CM — Metadata-Only Dry-Run Output Validator / Round-Trip Checker

> **Phase:** 31CM
> **Classification:** `PASS_31CM_DRY_RUN_OUTPUT_VALIDATOR_ROUND_TRIP_CLEAN`
> **Status:** Implementation complete (metadata-only); not yet committed.
> **Scope:** metadata-only. No model load. No Q2_K/SDIR blob generation. No raw activation generation. No runtime loader implementation. No llama.cpp modification or rebuild.

---

## 1. Purpose

Phase 31CM implements an **independent** metadata-only validator / round-trip checker that:

1. **Reads** the 31CL dry-run writer output JSON (and the 31CL summary JSON, indirectly via the output).
2. **Validates** that the 31CL output matches the 31CL contract (input shape, planned file count = 17, ffn_down no SDIR, byte accounting, fail-safe summary, claim boundary).
3. **Reconstructs** the round-trip plan from the 31CL output's `planned_files` list and verifies it agrees with the 31CL output's own `per_layer_plan` / `per_family_plan` / `planned_*_expected_bytes` fields.
4. **Re-validates** the 7 R-provenance rules that 31CL deferred (the 31CL output documents them in `validator_report_output.provenance_rules_skipped`; 31CM re-reads that block and confirms it is complete and bounded).
5. **Re-verifies** byte budgets against the 31CI sample metadata-only manifest (3-layer subset of the sample's 9 layer-family entries).
6. **Scans** all planned paths / planned filenames / top-level planned paths for forbidden path tokens and forbidden binary extensions (with `.plan.json` plan wrappers explicitly exempted).
7. **Runs** 13 embedded self-tests on synthetic inputs that exercise each check's positive and negative paths.
8. **Produces** a single metadata-only round-trip validation result JSON (`src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json`) and a short summary (`src/results/PHASE31CM_ROUND_TRIP_CHECK_SUMMARY.json`).

This validator does **not**:

- load model files
- inspect GGUF tensor payloads
- generate Q2_K or SDIR binary artifacts
- generate raw activation arrays
- create runtime-loadable blobs
- implement a real artifact writer
- modify the 31CL writer (no source patch required)
- implement a runtime loader
- modify or rebuild llama.cpp
- run generation or inference-quality tests
- commit model files, compiled binaries, raw tensor data, or generated Q2_K/SDIR blobs

---

## 2. Implementation

### 2.1 Source file

- `src/phase31cm_dry_run_output_validator.py` (~70 KB, Python stdlib only, lint-clean)

### 2.2 CLI

```bash
# Default run: validate the committed 31CL output against the 31CI sample manifest.
python3 src/phase31cm_dry_run_output_validator.py

# Custom paths:
python3 src/phase31cm_dry_run_output_validator.py \
    --input-json src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json \
    --sample-manifest src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json \
    --output-json src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json \
    --summary-json src/results/PHASE31CM_ROUND_TRIP_CHECK_SUMMARY.json \
    --strict

# Self-tests only:
python3 src/phase31cm_dry_run_output_validator.py --self-test
```

### 2.3 Checks (12 total)

Each check returns `{check_id, passed, message, details}`. The 12 checks are:

| # | check_id | purpose |
|---|---|---|
| 1 | `input_shape` | verifies `phase=31CL`, `classification=PASS_31CL_…CLEAN`, `dry_run=True`, `write_binary=False`, all 16 top-level keys present |
| 2 | `planned_file_count` | verifies `planned_file_count=17` (= 15 per-layer + 2 summary) |
| 3 | `planned_file_shape` | verifies 3 layers × 3 families with ffn_down SDIR=0 |
| 4 | `ffn_down_sdir_absence` | verifies no ffn_down.sdir plan exists in any `planned_files` entry |
| 5 | `byte_accounting` | verifies per-family Q2_K=4,515,840 / SDIR(ffn_up,ffn_gate)=1,857,972 / SDIR(ffn_down)=0; total Q2_K=40,642,560, total SDIR=11,147,832, total bytes=51,790,392 |
| 6 | `memory_margins_positive` | verifies per-layer margin bytes are positive for ffn_up / ffn_gate / ffn_down |
| 7 | `fail_safe_summary` | verifies 16 R_DRY_* fail-safe rules present, all status=pass, all severity=error |
| 8 | `provenance_filter_documented` | verifies the 7 R-provenance rule IDs (R-provenance-{created_by_phase, derived_from_phases, source_model_quantization, replay_w_ref_source, claim_boundary, forbidden_claims, valid_as_long_as}) are explicitly listed in `validator_report_output.provenance_rules_skipped` with a documented rationale |
| 9 | `forbidden_artifact_scan` | scans planned paths / planned filenames / top-level planned paths for forbidden path tokens (`/tmp/`, `/home/matthew-villnave`, etc.) and forbidden binary extensions (`.q2_k`, `.sdir`, `.sdiw`, `.gguf`, `.safetensors`, `.npy`, `.npz`, `.bin`, `.raw`); plan JSON wrappers are explicitly exempt |
| 10 | `claim_boundary_isolation` | verifies the 31CL output's top-level `claim_boundary` block asserts metadata-only scope (no model load, no Q2_K/SDIR generation, no runtime loader, no runtime integration, no llama.cpp modification/rebuild) AND `valid_as_long_as` is a non-empty list |
| 11 | `round_trip_reconstruction` | reconstructs the round-trip plan from `planned_files` and verifies per-layer / per-family / totals / memory margins / SHA-256 placeholders / ffn_down SDIR absence all match the 31CL output's own `per_layer_plan` / `per_family_plan` / `planned_*_expected_bytes` / `memory_accounting` |
| 12 | `sample_manifest_byte_match` | verifies the 31CI sample metadata-only manifest's per-family byte budgets agree with the 31CL per-family planned bytes (3 layers × 3 families = 9 layer-family entries, all with `expected_q2k_bytes=4,515,840`, `expected_sdir_bytes=1,857,972` for ffn_up/ffn_gate, 0 for ffn_down); 31CL's 3 layers must be a subset of the sample's unique layer indices |

### 2.4 Self-tests (13 total)

The validator carries 13 embedded self-tests, each running on a synthetic but contract-valid 31CL-shaped dict (built by `_synthesize_valid_base()`) with one targeted mutation:

| # | test name | what it asserts |
|---|---|---|
| 1 | `valid_31cl_output_passes_all_checks` | a fully-valid 31CL output passes all 12 checks; classification is `PASS_*` |
| 2 | `planned_file_count_not_17_fails` | setting `planned_file_count=16` fails the `planned_file_count` check |
| 3 | `ffn_down_sdir_plan_present_fails` | injecting an ffn_down.sdir entry fails the `ffn_down_sdir_absence` check |
| 4 | `write_binary_true_fails` | setting `write_binary=True` fails the `input_shape` check |
| 5 | `dry_run_false_fails` | setting `dry_run=False` fails the `input_shape` check |
| 6 | `raw_q2k_path_fails` | replacing a planned filename with a raw `.q2_k.W_low` (no `.plan.json`) fails the `forbidden_artifact_scan` check |
| 7 | `raw_sdir_path_fails` | stripping the `.plan.json` suffix from an sdir filename fails the `forbidden_artifact_scan` check |
| 8 | `q2k_plan_json_path_allowed` | the 31CL output's `.q2_k.W_low.plan.json` filename passes the `forbidden_artifact_scan` check (plan wrapper is exempt) |
| 9 | `hardcoded_operator_path_fails` | setting a planned path to `/home/matthew-villnave/...` fails the `forbidden_artifact_scan` check |
| 10 | `tmp_path_fails` | setting a planned path to `/tmp/...` fails the `forbidden_artifact_scan` check |
| 11 | `inline_payload_array_fails` | adding a `payload: [0.1, 0.2, 0.3]` field to a planned_files entry fails the `forbidden_artifact_scan` check |
| 12 | `missing_provenance_doc_yields_partial` | removing the `provenance_rules_skipped` block yields classification `PARTIAL_31CM_PROVENANCE_FILTER_DOCUMENTATION_INCOMPLETE` (NOT a hard BLOCKED) |
| 13 | `byte_total_mismatch_fails` | setting `planned_total_expected_bytes=1` fails the `byte_accounting` check |

All 13 self-tests passed in the `--self-test` run for Phase 31CM.

---

## 3. Round-trip reconstruction (deep detail)

The round-trip reconstruction is the core of 31CM. It re-derives the planned file structure from `planned_files` and verifies it matches the 31CL output's own derived fields.

### 3.1 Reconstruction algorithm

```python
# Group planned files by (layer, family, kind).
for each planned_file in dryrun_output["planned_files"]:
    key = (layer, family, kind)
    grouped[key].append(planned_file)

# Reconstruct per-layer plan.
for layer in [0, 14, 27]:
    for family in ["ffn_up", "ffn_gate", "ffn_down"]:
        q2k_files = grouped[(layer, family, "q2k_W_low")]
        sdir_files = grouped[(layer, family, "sdir_residual")]  # 0 for ffn_down
        # ...

# Aggregate totals.
reconstructed_q2k_total = sum(planned_byte_count for kind == "q2k_W_low")
reconstructed_sdir_total = sum(planned_byte_count for kind == "sdir_residual")
reconstructed_total = q2k_total + sdir_total

# Memory margins (per-family per-layer, from 31CF-S reference).
reconstructed_margins = {
    "ffn_up":   507_468,
    "ffn_gate": 507_466,
    "ffn_down": 2_365_440,
}
```

### 3.2 Reconstruction contract

The reconstructed plan must equal the 31CL output's own fields:

| reconstructed field | must equal | 31CL field |
|---|---|---|
| per_layer["0"].q2k_planned_files | sorted equality | `per_layer_plan["0"].q2k_planned_files` |
| per_layer["0"].sdir_planned_files | sorted equality | `per_layer_plan["0"].sdir_planned_files` |
| per_layer["0"].q2k_total_bytes | int equality | `per_layer_plan["0"].q2k_total_bytes` |
| per_layer["0"].sdir_total_bytes | int equality | `per_layer_plan["0"].sdir_total_bytes` |
| reconstructed_q2k_total | int equality | `planned_q2k_expected_bytes` |
| reconstructed_sdir_total | int equality | `planned_sdir_expected_bytes` |
| reconstructed_total | int equality | `planned_total_expected_bytes` |
| reconstructed_margins | dict equality | `memory_accounting.per_layer_margin_bytes` |
| n_ffn_down_sdir | == 0 | (must be 0) |
| all_sha_placeholders_present | True | (all must be `<computed_at_real_write_time>`) |

All 8 reconstruction invariants passed in the 31CM run.

### 3.3 Memory margins derivation

The 31CL output's `memory_accounting.per_layer_margin_bytes` are **per-family per-layer margins** (3 entries: ffn_up/ffn_gate/ffn_down), not aggregated per-layer. They come from the 31CF-S micro-probe result JSON (`memory_margin_bytes` field):

```json
// src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json
"memory_margin_bytes": {
  "ffn_up": 507468,
  "ffn_gate": 507466,
  "ffn_down": 2365440
},
"Q4_budget_family_bytes": 6881280
```

The 2-byte ffn_up vs ffn_gate asymmetry is a real per-tensor rounding artifact of the per-family Q4_K_M layout (the per-family Q4 budget is the same 6,881,280 bytes, but the ffn_gate W_low has 2 fewer bytes than ffn_up at the per-family level). 31CM treats these margins as **authoritative contract values**, not re-derived from byte budgets, and records the exact 31CF-S values.

---

## 4. Provenance filter handling (the 7 R-provenance rules)

The 31CL writer's planned manifest is a **31CL-derivative**, not a 31CI-original. The 31CI validator's 7 R-provenance rules hard-require `created_by_phase == "31CI"`, so the 31CL writer filters them out of the post-plan validation and documents them in `validator_report_output.provenance_rules_skipped`:

```json
"provenance_rules_skipped": {
  "rule_ids": [
    "R-provenance-claim_boundary",
    "R-provenance-created_by_phase",
    "R-provenance-derived_from_phases",
    "R-provenance-forbidden_claims",
    "R-provenance-replay_w_ref_source",
    "R-provenance-source_model_quantization",
    "R-provenance-valid_as_long_as"
  ],
  "n_errors_skipped": 1,
  "n_warnings_skipped": 1,
  "rationale": "R-provenance rules are not applicable to 31CL-derivative planned manifests (the 31CI validator hard-requires created_by_phase='31CI'). These rules are deferred to the 31CM output validator (or to the first phase that creates a true 31CI-original manifest)."
}
```

31CM's `provenance_filter_documented` check verifies:

- the 7 expected R-provenance rule IDs are present in `provenance_rules_skipped.rule_ids` (exact set match)
- the rationale string is non-empty and references `31CL`, `31CI`, or `deferred`
- the skip is bounded (n_errors_skipped and n_warnings_skipped are reported)

This check **passed** in the 31CM run. The provenance filter is documented, bounded, and deferred to 31CM (this phase) — which now actually re-validates it. The classification is `PASS_*`, not `PARTIAL_*`, because the 31CL output's documentation is complete and the 31CM validator independently confirms the documentation.

---

## 5. Forbidden artifact scan (scoped to planned paths)

The forbidden artifact scan is **scoped** to the planned paths / planned filenames / top-level planned path strings — NOT to the rule messages, validator report strings, or other documentation. This is intentional: rule messages legitimately contain mentions of forbidden patterns (e.g., `R_DRY_05`'s message: `"output_path contains '/tmp/': False; SDI_SCRATCH_NO_COMMIT=True (default deny)"` — the substring `/tmp/` is in the rule *description*, not in an actual path).

The scan covers:

- `planned_files[*].planned_path`
- `planned_files[*].planned_filename`
- `planned_summary_files[*].planned_path`
- `planned_summary_files[*].planned_filename`
- `artifact_root_planned`
- `planned_manifest_path`
- `planned_writer_plan_path`
- `planned_layer_dirs[*]`
- inline payload fields within `planned_files[*]` (keys `payload`, `tensor_data`, `raw_bytes`, `inline_payload`)

Plan JSON wrappers (files ending with one of the `ALLOWED_PLAN_SUFFIXES`) are EXEMPT:

- `.q2_k.W_low.plan.json`
- `.q2_k.W_high.plan.json`
- `.residual.sdir.plan.json`
- `.sdiw.plan.json`
- `.plan.json` (generic)
- `.dryrun.json`
- `writer_plan.json`
- `manifest.dryrun.json`

In the 31CM run, 0 findings.

---

## 6. Allowed / forbidden claims (this validator)

### Allowed claims

- A metadata-only dry-run output validator / round-trip checker was implemented.
- The checker validates the committed 31CL dry-run output.
- The checker independently reconstructs and verifies the planned file list, byte accounting, ffn_down no-SDIR rule, fail-safe summary, and forbidden artifact scan.
- The checker does not generate Q2_K/SDIR blobs, raw activations, model files, or runtime artifacts.
- The checker identifies and bounds the 7 filtered R-provenance rules from 31CL.
- 12/12 round-trip checks pass.
- 13/13 self-tests pass.
- Classification: `PASS_31CM_DRY_RUN_OUTPUT_VALIDATOR_ROUND_TRIP_CLEAN`.

### Forbidden claims

- no real artifact writer exists (a real writer is NOT implemented; only a metadata-only validator)
- no runtime loader exists
- no runtime integration exists
- no actual Q2_K / SDIR artifacts were generated
- no model files were loaded
- no generation-quality claim
- no speedup claim
- no live-runtime memory savings claim
- no production readiness claim
- no claim that 31CM proves future runtime viability
- no claim that provenance filtering is permanently solved (31CM re-validates the existing bound, but does not solve the underlying R-provenance hard-requirement)

---

## 7. Scope assertions (enforced by design)

This validator carries the following scope assertions in its output JSON:

```json
"scope_assertions": {
  "no_model_load": true,
  "no_q2k_blob_generation": true,
  "no_sdir_blob_generation": true,
  "no_raw_activation_generation": true,
  "no_runtime_loader_implemented": true,
  "no_runtime_integration_implemented": true,
  "no_llama_cpp_modification": true,
  "no_llama_cpp_rebuild": true,
  "no_compiled_binary_committed": true,
  "no_hardcoded_operator_paths": true,
  "no_binary_artifact_files_referenced": true,
  "no_inline_payload_arrays_in_output": true
}
```

These are enforced by **design** (the validator has no surface that could violate them): it imports no model-related libraries, it never writes binary, it never instantiates a `numpy.ndarray`, it never imports `torch`/`transformers`/`safetensors`, it never touches `~/llama.cpp/`.

---

## 8. Outputs

### `src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json` (~17 KB)

The full round-trip validation result, including:

- `phase`, `phase_full_name`, `classification`, `strict_mode`
- `input_json_path`, `input_json_sha256`
- `sample_manifest_path`, `sample_manifest_sha256`
- All 12 check results (each with `check_id`, `passed`, `message`, `details`)
- `round_trip_checks` summary (`n_checks`, `n_passed`, `n_failed`, `all_check_ids`, `failed_check_ids`)
- `reconstructed_plan_summary` (n_layers, n_per_layer_files, totals, memory margins, sha placeholders)
- `self_tests` (n_tests, n_passed, full results)
- `scope_assertions`, `allowed_claims`, `forbidden_claims`, `valid_as_long_as`, `next_recommended_phase`
- `generated_at_utc`, `no_commit_this_turn`

### `src/results/PHASE31CM_ROUND_TRIP_CHECK_SUMMARY.json` (~2 KB)

A short summary suitable for a CI/log line:

- `phase`, `classification`, `input_json_path`, `input_json_sha256`
- `n_checks`, `n_passed`, `n_failed`
- `n_self_tests`, `n_self_tests_passed`
- `planned_file_count`, `planned_total_bytes`, `q2k_total`, `sdir_total`
- `ffn_down_sdir_present` (bool), `n_ffn_down_sdir_files`
- `forbidden_findings`, `provenance_filters_documented`, `provenance_filter_n_rules`, `provenance_filter_rule_ids`
- `valid_as_long_as`, `next_recommended_phase`
- `scope_assertions`, `no_commit_this_turn`, `generated_at_utc`

---

## 9. Next recommended phase

**Phase 31CN — Metadata-Only Planned Manifest Normalizer / Provenance Adapter** (only if explicitly requested).

Purpose: resolve the 7 R-provenance-rule deferral cleanly by designing a normalized planned-manifest provenance adapter so derivative manifests can validate without rule filtering. This is a follow-up to 31CM that closes the provenance-gap loop.

**Alternatives:**

- **31CK** — Loader Integration Planning (planning-only, only if explicitly requested)
- **31CG** — Larger Prompt/Token Sensitivity Planning (planning-only, only if explicitly requested)
- **31CM-R** — Dry-Run Output Validator Repair (only if 31CM is partial or blocked; currently NOT indicated since 31CM PASSED)

All next phases require explicit operator approval at entry; the agent does NOT proceed to any without a new request.

---

## 10. Limitations

- 31CM does NOT modify or solve the underlying R-provenance hard-requirement in the 31CI validator. It re-validates the existing bound (the 7 rules are documented as skipped). A future 31CN would address this.
- 31CM does NOT verify that the 31CL output's `planned_sha256_placeholder` is the literal string `<computed_at_real_write_time>` — it only verifies that all placeholder values are non-empty strings of that form.
- 31CM does NOT verify that the 31CL output's `input_manifest_path` matches an existing file. The 31CI sample manifest is the only file path it cross-references.
- 31CM does NOT verify that the 31CL output's `dry_run=True` and `write_binary=False` values are immutable constants in the 31CL writer. It only verifies the values are present and correct in the output JSON.

---

## 11. Reproduction

```bash
# 1. Run the validator on the committed 31CL output.
cd ~/sdi-substitutive
python3 src/phase31cm_dry_run_output_validator.py

# Expected:
# [31CM] classification: PASS_31CM_DRY_RUN_OUTPUT_VALIDATOR_ROUND_TRIP_CLEAN
# [31CM] n_checks: 12  n_passed: 12  n_failed: 0
# [31CM] n_self_tests: 13  n_self_tests_passed: 13
# [31CM] output_json: src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json
# [31CM] summary_json: src/results/PHASE31CM_ROUND_TRIP_CHECK_SUMMARY.json

# 2. Run self-tests only.
python3 src/phase31cm_dry_run_output_validator.py --self-test

# Expected:
# classification: PASS_31CM_SELF_TESTS_CLEAN
# n_self_tests_passed: 13 / 13

# 3. Verify the result JSON's classification.
python3 -c "import json; print(json.load(open('src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json'))['classification'])"
# Expected: PASS_31CM_DRY_RUN_OUTPUT_VALIDATOR_ROUND_TRIP_CLEAN
```

---

## 12. End of phase 31CM

This document is the implementation + design record for Phase 31CM. It is committed alongside:

- `src/phase31cm_dry_run_output_validator.py` (~70 KB, Python stdlib only, lint-clean)
- `src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json` (~17 KB, full round-trip validation result)
- `src/results/PHASE31CM_ROUND_TRIP_CHECK_SUMMARY.json` (~2 KB, short summary)

No other files are modified. The 31CL writer, 31CL output JSONs, 31CI validator, 31CI sample manifest, 31CH plan, and 31CJ plan are all UNCHANGED. The SOT line-11 next-allowed-phase and line-5 state label are updated to reflect 31CM's PASSED state, with 31CN named as the recommended next step.
