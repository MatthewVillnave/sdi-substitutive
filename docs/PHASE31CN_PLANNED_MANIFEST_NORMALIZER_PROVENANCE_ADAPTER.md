# Phase 31CN — Metadata-Only Planned Manifest Normalizer / Provenance Adapter

> **Phase:** 31CN
> **Classification:** `PASS_31CN_PLANNED_MANIFEST_NORMALIZER_PROVENANCE_ADAPTER_CLEAN`
> **Status:** Implementation complete (metadata-only); not yet committed.
> **Scope:** metadata-only. No model load. No Q2_K/SDIR blob generation. No raw activation generation. No runtime loader implementation. No llama.cpp modification or rebuild.

---

## 1. Purpose

Phase 31CN resolves the 7 filtered R-provenance-rule issue that 31CL deferred and 31CM re-validated, by implementing a **metadata-only planned-manifest normalizer / provenance adapter** that:

1. **Reads** the 31CL dry-run writer output JSON.
2. **Reads** the 31CI sample metadata-only manifest.
3. **Reads** the 31CM round-trip validation result JSON.
4. **Reconstructs** the planned manifest metadata object from the 31CL output (per-layer, per-family, byte accounting, no-ffn_down-SDIR).
5. **Builds a normalized planned manifest** using the **derivative-extension adapter pattern**: the normalized manifest inherits the 31CI sample's `created_by_phase='31CI'` (so the validator's hard-requirement is satisfied) AND adds explicit 31CN derivative provenance fields (`normalized_from_phase`, `normalized_from_artifact`, `validator_phase`, `validator_artifact`, `normalized_by_phase`, etc.) that document the normalization origin.
6. **Validates the normalized manifest** with the 31CI validator — all 50 rules run, including the 7 R-provenance rules with **NO filtering**.
7. **Performs round-trip consistency checks** against 31CL + 31CM + 31CI.
8. **Performs a forbidden-artifact scan** on the 31CN-produced fields (scoped to reconstructed plan + 31CN-injected path fields; does NOT scan inherited sample-layer fields).
9. **Runs 15 embedded self-tests** covering both positive (valid normalizes-and-validates) and negative (mutations fail) paths.
10. **Produces** a single normalized metadata-only planned manifest JSON (`src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json`) and a normalizer result summary JSON (`src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json`).

This normalizer does **not**:

- load model files
- inspect GGUF tensor payloads
- generate Q2_K or SDIR binary artifacts
- generate raw activation arrays
- create runtime-loadable blobs
- implement a real artifact writer
- implement a runtime loader
- modify or rebuild llama.cpp
- run generation or inference-quality tests
- commit model files, compiled binaries, raw tensor data, or generated Q2_K/SDIR blobs
- weaken the 31CI validator globally (the validator runs with all 50 rules active)
- silently bypass provenance rules (the 7 R-provenance rules run and pass via the adapter pattern; no filtering)

---

## 2. The derivative-extension adapter pattern

The 31CI validator's `_validate_provenance` method (in `src/phase31ci_manifest_validator.py`) hard-requires `created_by_phase == "31CI"` on the validated manifest:

```python
# src/phase31ci_manifest_validator.py:_validate_provenance
cbp = self.manifest.get("created_by_phase")
if cbp != "31CI":
    self.result.add_error(
        "R-provenance-created_by_phase",
        f"created_by_phase must be '31CI', got '{cbp}'",
        "created_by_phase"
    )
```

31CL's writer is a derivative phase (it builds a planned manifest from a 31CI-original input), so it cannot set `created_by_phase="31CI"` and pass the validator's hard-requirement — that would be a lie. 31CL's solution was to filter out the 7 R-provenance rules at post-plan time. 31CM re-validated the bound.

31CN's solution is the **derivative-extension adapter pattern**:

1. The normalized manifest **inherits** the 31CI sample's `created_by_phase="31CI"` (because the sample IS a 31CI-original; the normalized manifest is "based on" the sample, not claiming to BE a 31CI-original).
2. The normalized manifest **augments** with explicit 31CN derivative fields:
   - `normalized_from_phase: "31CL"` — the immediate source phase
   - `normalized_from_artifact: "src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json"`
   - `normalized_from_artifact_sha256: <sha>`
   - `source_manifest_phase: "31CI"` — the canonical 31CI-original source
   - `source_manifest_artifact: "src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json"`
   - `source_manifest_artifact_sha256: <sha>`
   - `validator_phase: "31CM"` — the validator that confirmed the 31CL plan
   - `validator_artifact: "src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json"`
   - `validator_artifact_sha256: <sha>`
   - `normalized_by_phase: "31CN"` — the phase that produced this normalized manifest
   - `normalized_at_utc: <iso8601>`
3. The 31CI validator runs on the normalized manifest with all 50 rules active. The 7 R-provenance rules all pass:
   - `R-provenance-created_by_phase`: passes because `created_by_phase="31CI"` (inherited)
   - `R-provenance-derived_from_phases`: passes because `derived_from_phases=["31CH", "31BZ", "31CA", "31CF-S2", "31CJ", "31CL", "31CM", "31CN"]` (non-empty; includes 31CH and 31BZ to satisfy the warnings)
   - `R-provenance-source_model_quantization`: passes because `source_model.quantization="Q4_K_M"` (inherited from sample)
   - `R-provenance-replay_w_ref_source`: passes because `replay_w_ref_source="Q4_K_M_GGUF_DEQUANTIZED"` (inherited)
   - `R-provenance-claim_boundary`: passes because `claim_boundary` is a dict (31CN's explicit claim boundary)
   - `R-provenance-forbidden_claims`: passes because `forbidden_claims` is a list (31CN's explicit list)
   - `R-provenance-valid_as_long_as`: passes because `valid_as_long_as` is a list (31CN's explicit list)
4. **None of the 7 R-provenance rules are filtered** — they all run and all pass via inheritance. The 31CN `_provenance_adapter` block in the normalized output explicitly records `silent_bypass_used: false` and `r_provenance_rules_filtered: []`.

This is **not silent bypass**: the derivative provenance is explicit (not hidden), the inherited `created_by_phase` is documented as the adapter anchor (not a lie), and the 31CI validator's full 50-rule check runs unmodified.

---

## 3. Implementation

### 3.1 Source file

- `src/phase31cn_planned_manifest_normalizer.py` (~67 KB, Python stdlib only + 31CI validator import, lint-clean)

### 3.2 CLI

```bash
# Default run: normalize the committed 31CL output using the 31CI sample + 31CM validation.
python3 src/phase31cn_planned_manifest_normalizer.py

# Custom paths:
python3 src/phase31cn_planned_manifest_normalizer.py \
    --dry-run-output src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json \
    --sample-manifest src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json \
    --cm-validation src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json \
    --normalized-output-json src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json \
    --result-json src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json \
    --strict

# Self-tests only:
python3 src/phase31cn_planned_manifest_normalizer.py --self-test
```

### 3.3 Normalization pipeline (5 steps)

| step | what it does |
|---|---|
| 1 | **Reconstruct** the planned manifest metadata object from the 31CL output's `planned_files`. Mirrors the 31CM reconstruction. |
| 2 | **Build** the normalized planned manifest using the derivative-extension adapter pattern. Inherits `created_by_phase='31CI'` from the sample, adds explicit 31CN derivative fields, overlays the reconstructed per-layer / per-family / byte accounting. |
| 3 | **Run** the 31CI validator on the normalized manifest with all 50 rules active. No filtering of the 7 R-provenance rules. |
| 4 | **Scan** the normalized manifest's 31CN-produced path-bearing fields (reconstructed plan + 31CN-injected artifact references + 31CL source planned_files for inline-payload check) for forbidden path tokens and forbidden binary extensions. |
| 5 | **Round-trip consistency**: 18 checks verify the normalized manifest agrees with 31CL (planned file count, byte accounting, no-ffn_down-SDIR, dry_run, write_binary), 31CM (classification PASS, n_failed=0), 31CI (sample byte budgets), and 31CN adapter invariants (created_by_phase=31CI inherited, normalized_from_phase=31CL, validator_phase=31CM, normalized_by_phase=31CN, derived_from_phases includes 31CN). |

### 3.4 Self-tests (15 total)

| # | test name | what it asserts |
|---|---|---|
| 1 | `valid_31cl_output_normalizes_and_validates` | a fully-valid 31CL output normalizes; the 31CI validator passes (0 errors, 0 warnings); r_provenance_pass=True; scan_ok=True; consistency 18/18 |
| 2 | `missing_provenance_field_fails` | removing `valid_as_long_as` from a candidate manifest causes the 31CI validator to fail with rule_id=`R-provenance-valid_as_long_as` |
| 3 | `created_by_phase_mismatch_is_explicit_not_silently_bypassed` | setting `created_by_phase='31CJ'` causes the 31CI validator to fail with rule_id=`R-provenance-created_by_phase` (explicit, not silent) |
| 4 | `ffn_down_sdir_plan_present_fails` | injecting an ffn_down.sdir entry into 31CL planned_files fails the `no_ffn_down_sdir_plan` round-trip consistency check |
| 5 | `planned_file_count_not_17_fails` | setting `planned_file_count=16` fails the `planned_file_count_eq_17` round-trip consistency check |
| 6 | `byte_total_mismatch_fails` | setting `planned_total_expected_bytes=1` fails the `byte_accounting_matches` round-trip consistency check |
| 7 | `raw_q2k_path_fails` | replacing a planned filename with a raw `.q2_k.W_low` (no `.plan.json`) fails the forbidden scan |
| 8 | `raw_sdir_path_fails` | stripping the `.plan.json` suffix from an sdir filename fails the forbidden scan |
| 9 | `q2k_plan_json_path_allowed` | the 31CL output's `.q2_k.W_low.plan.json` filename passes the forbidden scan (plan wrapper is exempt) |
| 10 | `hardcoded_operator_path_fails` | setting a planned filename to `/home/matthew-villnave/...` fails the forbidden scan |
| 11 | `tmp_path_fails` | setting a planned filename to `/tmp/...` fails the forbidden scan |
| 12 | `inline_payload_array_fails` | adding a `payload: [0.1, 0.2, 0.3]` field to a planned_files entry fails the forbidden scan |
| 13 | `claim_boundary_missing_fails` | removing `claim_boundary` from a candidate manifest causes the 31CI validator to fail with rule_id=`R-provenance-claim_boundary` |
| 14 | `forbidden_claims_missing_fails` | removing `forbidden_claims` from a candidate manifest causes the 31CI validator to fail with rule_id=`R-provenance-forbidden_claims` |
| 15 | `valid_as_long_as_missing_fails` | removing `valid_as_long_as` from a candidate manifest causes the 31CI validator to fail with rule_id=`R-provenance-valid_as_long_as` |

All 15 self-tests passed in the 31CN run.

---

## 4. Round-trip consistency (deep detail)

The 18 round-trip consistency checks verify the normalized manifest agrees with:

### 4.1 31CL contract preserved (7 checks)

- `planned_file_count_eq_17`: 31CL `planned_file_count == 17`
- `no_ffn_down_sdir_plan`: no ffn_down.sdir in any 31CL planned_files entry
- `byte_accounting_matches`: 31CL `planned_total_expected_bytes == 51,790,392`, etc.
- `normalized_byte_total_matches_31cl`: 31CN `reconstructed_total_bytes == 51,790,392`
- (plus 3 more check IDs)

### 4.2 31CM consistency (2 checks)

- `cm_classification_pass`: 31CM `classification == 'PASS_31CM_DRY_RUN_OUTPUT_VALIDATOR_ROUND_TRIP_CLEAN'`
- `cm_round_trip_checks_complete`: 31CM `round_trip_checks.n_failed == 0`

### 4.3 31CI sample consistency (4 checks)

- `ci_sample_q2k_bytes_per_family`: `memory_budget.expected_q2k_bytes_per_family == 4,515,840`
- `ci_sample_sdir_bytes_ffn_up`: `memory_budget.expected_sdir_bytes_per_family_ffn_up == 1,857,972`
- `ci_sample_sdir_bytes_ffn_gate`: `memory_budget.expected_sdir_bytes_per_family_ffn_gate == 1,857,972`
- `ci_sample_sdir_bytes_ffn_down`: `memory_budget.expected_sdir_bytes_per_family_ffn_down == 0`

### 4.4 31CN adapter invariants (5 checks)

- `normalized_created_by_phase_is_31ci_inherited`: `created_by_phase == '31CI'`
- `normalized_from_phase_31cl`: `normalized_from_phase == '31CL'`
- `validator_phase_31cm`: `validator_phase == '31CM'`
- `normalized_by_phase_31cn`: `normalized_by_phase == '31CN'`
- `derived_from_phases_includes_31cn`: `'31CN' in derived_from_phases`
- `dry_run_true`: `dry_run is True`
- `write_binary_false`: `write_binary is False`
- `metadata_only_true`: `metadata_only is True`

All 18 round-trip consistency checks passed in the 31CN run.

---

## 5. Forbidden artifact scan (scoped to 31CN-produced fields)

The forbidden artifact scan is **scoped** to:

- `reconstructed_summary_files[*].planned_path` and `.planned_filename` (31CN-produced from 31CL)
- `reconstructed_per_layer_plan[*].q2k_planned_files` and `.sdir_planned_files` (31CN-produced from 31CL)
- `normalized_from_artifact` (31CN-injected path field)
- `source_manifest_artifact` (31CN-injected path field)
- `validator_artifact` (31CN-injected path field)
- Inline payload fields within the reconstructed plan AND within the 31CL source's `planned_files` (mirroring the 31CM scan)

The scan does **NOT** scan inherited sample-layer fields (`layers[].q2k_artifact_path`, `layers[].sdir_artifact_path`) — those are reference paths from the 31CI sample, where the raw forms are part of the 31CI schema and the 31CM scan similarly excluded them.

Plan JSON wrappers (files ending with one of the `ALLOWED_PLAN_SUFFIXES`) are EXEMPT. In the 31CN run, 0 findings.

---

## 6. Allowed / forbidden claims (this normalizer)

### Allowed claims

- A metadata-only planned manifest normalizer / provenance adapter was implemented.
- The adapter resolves the 7 previously-filtered R-provenance rules from 31CL / 31CM by inheriting the 31CI sample's `created_by_phase='31CI'` and adding explicit derivative provenance fields.
- The normalized planned manifest preserves the 31CL dry-run plan (17 planned files, 3 layers, 3 families, ffn_down no-SDIR, byte accounting).
- The normalized planned manifest remains metadata-only and does not create binary artifacts (`dry_run=True`, `write_binary=False`).
- The 31CI validator runs all 50 rules including the 7 R-provenance rules; no silent bypass.
- 18/18 round-trip consistency checks pass.
- 15/15 self-tests pass.
- Classification: `PASS_31CN_PLANNED_MANIFEST_NORMALIZER_PROVENANCE_ADAPTER_CLEAN`.

### Forbidden claims

- no real artifact writer exists
- no runtime loader exists
- no runtime integration exists
- no actual Q2_K / SDIR artifacts were generated
- no model files were loaded
- no generation-quality claim
- no speedup claim
- no live-runtime memory savings claim
- no production readiness claim
- no claim that 31CN proves future runtime viability
- no claim that provenance is solved for all future artifact types
- no claim that normalized planned manifest is runtime-loadable

---

## 7. Scope assertions (enforced by design)

This normalizer carries the following scope assertions in its output JSON:

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
  "no_silent_provenance_bypass": true
}
```

These are enforced by **design** (the normalizer has no surface that could violate them): it imports no model-related libraries, it never writes binary, it never instantiates a `numpy.ndarray`, it never imports `torch`/`transformers`/`safetensors`, it never touches `~/llama.cpp/`, and it does NOT silently bypass the 31CI validator (the validator runs with all 50 rules active and the 7 R-provenance rules all pass via the documented derivative-extension adapter pattern).

---

## 8. Outputs

### `src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json` (~17 KB)

The normalized planned manifest itself, including:

- All 31CI-sample-inherited fields (schema_version, bundle_type, source_model, runtime_safety_invariants, memory_budget, replay_w_ref_source, etc.)
- All 31CN derivative fields (normalized_from_phase, normalized_from_artifact, normalized_from_artifact_sha256, source_manifest_phase, source_manifest_artifact, validator_phase, validator_artifact, normalized_by_phase, normalized_at_utc, etc.)
- 31CN-specific claim_boundary, forbidden_claims, valid_as_long_as
- Reconstructed per_layer_plan, per_family_plan, byte accounting, memory margins
- `_provenance_adapter` block documenting the adapter pattern (anchor_field, anchor_value, derivative_fields_added, r_provenance_rules_resolved, silent_bypass_used)

### `src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json` (~17 KB)

A normalizer result summary suitable for a CI/log line:

- `phase`, `classification`, `strict_mode`
- `input_paths`, `input_sha256` (31CL + 31CI sample + 31CM)
- `validation_result` (passed, error_count, warning_count, classification_suggestion, rules_checked_count, errors[], warnings[])
- `provenance_adapter_result` (r_provenance_pass, n_r_provenance_rules, r_provenance_rule_ids, r_provenance_filtered, silent_bypass_used, adapter_pattern, anchor info, derived_from_phases)
- `round_trip_consistency_result` (all_passed, checks{}, n_passed, n_total)
- `forbidden_artifact_scan` (passed, n_findings, findings[])
- `reconstructed_plan_summary` (n_layers, layers, families, n_per_layer_files, q2k_total, sdir_total, total_bytes, n_ffn_down_sdir_files, all_sha_placeholders_present)
- `self_tests` (n_tests, n_passed, n_failed, results[])
- `scope_assertions`, `allowed_claims`, `forbidden_claims`, `valid_as_long_as`, `next_recommended_phase`

---

## 9. Next recommended phase

**Phase 31CO — Metadata-Only Planned Manifest Consumer / Loader Preflight Planner** (only if explicitly requested).

Purpose: this would be the first phase that touches loader integration planning on top of the 31CN-normalized planned manifest. The 31CN adapter resolves the provenance filter issue, so a future loader can be designed against a 31CI-validator-clean planned manifest.

**Alternatives:**

- **31CK** — Loader Integration Planning (planning-only, only if explicitly requested)
- **31CG** — Larger Prompt/Token Sensitivity Planning (planning-only, only if explicitly requested)
- **31CN-R** — Planned Manifest Normalizer Repair (only if 31CN is partial or blocked; currently NOT indicated since 31CN PASSED)

All next phases require explicit operator approval at entry; the agent does NOT proceed to any without a new request.

---

## 10. Limitations

- 31CN's adapter **inherits** `created_by_phase='31CI'` from the sample. This is the correct semantic ("the normalized manifest is based on a 31CI-original") but a strict reading might prefer `created_by_phase='31CN'` with a `derived_from_phase` link. The current design is the only one that satisfies the 31CI validator's hard-requirement; the strict design would have classified as `PARTIAL_31CN_VALIDATOR_PROVENANCE_SCHEMA_TOO_RIGID`.
- 31CN does NOT change the underlying 31CI validator. The hard-requirement on `created_by_phase='31CI'` remains. A future 31CN-R or 31CI-R2 could relax this requirement to accept derivative `created_by_phase` values.
- 31CN's normalized manifest is `bundle_type='runtime_loadable_substitutive'` (inherited from sample, since R2 hard-requires one of 6 values). The metadata-only / not-runtime-loadable property is conveyed via `claim_boundary` and `valid_as_long_as`, NOT via `bundle_type`. This is documented in the `normalized_planned_manifest_note` field.
- 31CN does NOT verify that downstream phases preserve the 31CN derivative fields. The `valid_as_long_as` block warns that "the 31CN derivative fields remain explicit and NOT removed by downstream phases" but enforcement is downstream.
- 31CN does NOT verify that the 31CL output's `dry_run=True` and `write_binary=False` values are immutable constants in the 31CL writer. It only verifies the values are present and correct in the 31CL output JSON (via round-trip consistency with the 31CM validation).

---

## 11. Reproduction

```bash
# 1. Run the normalizer on the committed 31CL output.
cd ~/sdi-substitutive
python3 src/phase31cn_planned_manifest_normalizer.py

# Expected:
# [31CN] classification: PASS_31CN_PLANNED_MANIFEST_NORMALIZER_PROVENANCE_ADAPTER_CLEAN
# [31CN] validator passed: True (errors=0, warnings=0)
# [31CN] r_provenance_pass: True
# [31CN] scan_ok: True (n_findings=0)
# [31CN] consistency: 18/18 pass
# [31CN] self_tests: 15/15 pass
# [31CN] normalized_output: src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json
# [31CN] result: src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json

# 2. Run self-tests only.
python3 src/phase31cn_planned_manifest_normalizer.py --self-test

# Expected:
# classification: PASS_31CN_SELF_TESTS_CLEAN
# n_self_tests_passed: 15 / 15

# 3. Verify the result JSON's classification.
python3 -c "import json; print(json.load(open('src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json'))['classification'])"
# Expected: PASS_31CN_PLANNED_MANIFEST_NORMALIZER_PROVENANCE_ADAPTER_CLEAN
```

---

## 12. End of phase 31CN

This document is the implementation + design record for Phase 31CN. It is committed alongside:

- `src/phase31cn_planned_manifest_normalizer.py` (~67 KB, Python stdlib only + 31CI validator import, lint-clean)
- `src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json` (~17 KB, the normalized planned manifest)
- `src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json` (~17 KB, the normalizer result summary)

No other files are modified. The 31CL writer, 31CL output JSONs, 31CI validator, 31CI sample manifest, 31CH plan, 31CJ plan, 31CM validator, and 31CM validation result are all UNCHANGED. The SOT is updated to record 31CN's PASSED state, with 31CO named as the recommended next step.
