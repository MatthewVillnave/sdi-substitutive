# Phase 31CO — Metadata-Only Planned Manifest Consumer / Loader Preflight Planner

> **Phase:** 31CO
> **Classification:** `PASS_31CO_PLANNED_MANIFEST_CONSUMER_LOADER_PREFLIGHT_PLANNER_CLEAN`
> **Status:** Planning complete (metadata-only); not yet committed.
> **Scope:** metadata-only. No model load. No Q2_K/SDIR blob generation. No raw activation generation. **No runtime loader implementation. No runtime integration. No llama.cpp modification or rebuild. No loader source code. No compiled binary.**
> **Inputs reviewed:** `docs/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_PROVENANCE_ADAPTER.md` (350 lines), `src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json` (621 lines, 22.5 KB), `src/results/PHASE31CN_PLANNED_MANIFEST_NORMALIZER_RESULT.json` (251 lines, 8.9 KB), `docs/PHASE31CM_DRY_RUN_OUTPUT_VALIDATOR_ROUND_TRIP.md` (383 lines), `src/results/PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json` (546 lines, 17 KB), `docs/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.md` (505 lines), `src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json` (18 KB), `docs/PHASE31CJ_ARTIFACT_WRITER_DRY_RUN_PLANNING.md` (468 lines), `docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md` (705 lines), `src/phase31ci_manifest_validator.py` (50 rules in 8 categories), `src/phase31cm_dry_run_output_validator.py` (12 round-trip checks), `src/phase31cn_planned_manifest_normalizer.py` (derivative-extension adapter), `docs/SDI_OPERATOR_SKILL_PACK_PLAN.md` (8-skill pack).

---

## 1. Purpose

Phase 31CO designs the **metadata-only planned manifest consumer / loader preflight planner** that sits on top of the 31CN-normalized planned manifest. This is the **first phase that explicitly names a "loader" in its scope**, but it is **planning-only** — it does **NOT** implement an actual runtime loader. It does **NOT** integrate with llama.cpp. It does **NOT** write source code. It does **NOT** generate any binary artifacts.

The phase answers four questions:

1. **What would a future loader receive?** — the input contract for the planned manifest, anchored to the 31CN-normalized manifest as the canonical input.
2. **In what order would a future loader validate the input?** — the preflight sequence (8 ordered checks) that a future loader would run *before* attempting any actual artifact load.
3. **When is a future loader allowed to proceed?** — the loader-readiness boundary: the set of conditions that, if all true, would allow a future 31CP-style loader to actually attempt a load; if any condition is false, the loader must refuse to start.
4. **When must a future loader refuse to start, even mid-load?** — the fail-fast rules and no-go conditions that a future loader would check at every loading checkpoint, with explicit halt-and-report semantics.

The deliverable is this planning document plus `src/results/PHASE31CO_PLANNED_MANIFEST_CONSUMER_LOADER_PREFLIGHT_PLANNER.json` (a metadata-only planning summary JSON). **No Python implementation is produced in 31CO.** The actual loader would be a future phase (recommended: `31CP`), only with explicit operator approval, and only if the loader-readiness boundary evaluates to PASS for a specific input.

This phase does **not**:

- load any model file (no opening Q4_K_M GGUF, no opening HF safetensors)
- inspect any GGUF tensor payload
- generate Q2_K or SDIR binary artifacts
- generate raw activation arrays
- create runtime-loadable blobs
- implement a runtime loader
- implement a runtime integration
- modify or rebuild llama.cpp
- run generation or inference-quality tests
- write any Python source (`src/phase31co_*.py` is forbidden)
- commit model files, compiled binaries, raw tensor data, or Q2_K/SDIR blobs
- weaken the 31CI validator globally (the validator still runs with all 50 rules active)
- silently bypass provenance rules (no filtering, no silent fallback)
- enable any legacy PRT/SDI sidecar machinery
- claim runtime viability, production readiness, or speedup

---

## 2. Consumer purpose and scope

### 2.1 What is the "consumer"?

The **consumer** is a future metadata-only Python module (not in 31CO; reserved for a future 31CP-style phase) that would:

1. **Read** the 31CN-normalized planned manifest JSON.
2. **Run** the 8-step preflight sequence (§4) on it.
3. **Evaluate** the loader-readiness boundary (§6).
4. **Report** PASS / FAIL / BLOCKED with explicit fail-fast triggers (§7) and no-go conditions (§8).

The consumer is a **reader**, not a writer. It does not modify the manifest. It does not write any artifact. It does not invoke any model. It does not invoke llama.cpp.

### 2.2 What is the "loader preflight planner"?

The **loader preflight planner** is the *planning* artifact (this document + the JSON) that designs the preflight sequence, the loader-readiness boundary, the fail-fast rules, and the no-go conditions. It is the contract that a future loader would implement against. The 31CO phase is the design step; a future phase would be the implementation step (and only with explicit operator approval).

### 2.3 Relationship to the 31C* phase family

| phase | role | relationship to 31CO |
|---|---|---|
| 31CH | runtime artifact format / loader planning | the **upstream** planning phase that designed the v1.1.0 manifest schema, 9 runtime safety invariants, 7 legacy-sidecar exclusion glob patterns, 48 loader validation rules R1-R48, 4 loader architecture options (A: standalone, B: llama.cpp diagnostic, C: runtime integration, D: legacy sidecar reuse — REJECTED); 31CO inherits the v1.1.0 schema and the 31CH safety invariants and the 7 legacy-sidecar exclusion globs |
| 31CI | metadata-only runtime artifact manifest validator | the **50-rule validator** that the 31CO preflight would call at step 1 (preflight check 1) and step 7 (preflight check 7 — re-validation after legacy-sidecar exclusion scan) |
| 31CJ | artifact writer dry-run planning | the **planning-only precedent** for 31CO; 31CO follows the same planning-only discipline (design only, no implementation, no commit without explicit approval) |
| 31CL | metadata-only artifact writer dry-run prototype | the **first** dry-run writer; produces `src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json` (17 planned files); 31CO's preflight would consume this as the *un-normalized* planned manifest input (pre-31CN) |
| 31CM | metadata-only dry-run output validator / round-trip checker | the **12-check round-trip validator**; 31CO's preflight would call 31CM at step 4 (round-trip consistency re-check) to confirm the manifest is internally consistent before considering loader-readiness |
| 31CN | metadata-only planned manifest normalizer / provenance adapter | the **direct upstream phase**; produces `src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json` (the 31CN-normalized planned manifest); 31CO's preflight would consume this as the **canonical input** for the consumer; 31CO inherits 31CN's derivative-extension adapter pattern (the 7 R-provenance rules pass via inheritance + explicit derivative fields, with no silent bypass) |
| 31CO (this) | metadata-only planned manifest consumer / loader preflight planner | **the current phase**; designs the consumer contract, the preflight sequence, the loader-readiness boundary, the fail-fast rules, and the no-go conditions |
| 31CP (recommended next) | metadata-only planned manifest consumer / loader preflight *implementation* (proposed) | **NOT in scope for 31CO**; would be a future phase that actually implements the consumer in Python, only with explicit operator approval, and only if the loader-readiness boundary evaluates to PASS for a specific input |

### 2.4 Out of scope (forbidden by 31CO's prompt)

- ✗ no model load (no opening Q4_K_M GGUF, no opening HF safetensors)
- ✗ no GGUF tensor payload inspection
- ✗ no Q2_K binary artifact generation
- ✗ no SDIR binary artifact generation
- ✗ no raw activation artifact generation
- ✗ no creation of runtime-loadable blobs
- ✗ no implementation of the actual loader
- ✗ no implementation of a runtime integration
- ✗ no `~/llama.cpp/` modification
- ✗ no `~/llama.cpp/` rebuild
- ✗ no generation, no inference, no inference-quality test
- ✗ no Python source file (`src/phase31co_*.py` is forbidden)
- ✗ no compiled binary
- ✗ no commit of model files / HF cache / GGUF / safetensors / raw activations / build artifacts / Q2_K blobs / SDIR blobs / temp tensor dumps / llama.cpp source
- ✗ no tag created or pushed
- ✗ no commit, push, or tag without explicit operator approval (31CO stops at PRE-COMMIT REPORT)

---

## 3. Normalized manifest input contract

The consumer's canonical input is `src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json` (the 31CN-normalized derivative of the 31CL dry-run output + 31CI sample manifest + 31CM validation result). The consumer's input contract is the **shape of that JSON, anchored to the 31CN output's actual fields**, as verified by 31CM round-trip consistency (12/12 checks pass) and 31CN's 31CI-validator pass (50/50 rules, all 7 R-provenance rules pass with no filtering).

### 3.1 Required top-level fields (the consumer would refuse to start if any is missing)

| field | type | source | what it asserts |
|---|---|---|---|
| `metadata_only` | bool = true | 31CI / 31CL / 31CN | the manifest is metadata-only and not runtime-loadable |
| `dry_run` | bool = true | 31CL / 31CN | no binary blobs were written |
| `write_binary` | bool = false | 31CL / 31CN | no binary write was performed |
| `bundle_type` | str = `"runtime_loadable_substitutive"` | 31CI / 31CN | the bundle type the 31CI validator accepts (R2 hard-requirement) |
| `schema_version` | str = `"1.1.0"` | 31CI | the schema version the 31CI validator accepts |
| `created_by_phase` | str = `"31CI"` | 31CI (inherited by 31CN adapter) | the 31CI-original source phase (R-provenance-created_by_phase hard-requirement) |
| `normalized_planned_manifest` | bool = true | 31CN | this is a 31CN-normalized derivative, not a 31CI-original |
| `normalized_from_phase` | str = `"31CL"` | 31CN | the immediate source phase |
| `normalized_from_artifact` | str | 31CN | the immediate source artifact path |
| `normalized_from_artifact_sha256` | str (sha256 hex) | 31CN | the immediate source artifact's SHA-256 |
| `source_manifest_phase` | str = `"31CI"` | 31CN | the canonical 31CI-original source phase |
| `source_manifest_artifact` | str | 31CN | the canonical 31CI-original source artifact path |
| `source_manifest_artifact_sha256` | str (sha256 hex) | 31CN | the canonical 31CI-original source artifact's SHA-256 |
| `validator_phase` | str = `"31CM"` | 31CN | the validator phase that confirmed the 31CL plan |
| `validator_artifact` | str | 31CN | the validator phase's output artifact path |
| `validator_artifact_sha256` | str (sha256 hex) | 31CN | the validator phase's output artifact's SHA-256 |
| `normalized_by_phase` | str = `"31CN"` | 31CN | the phase that produced this normalized manifest |
| `normalized_at_utc` | str (ISO 8601) | 31CN | when the normalization was performed |
| `claim_boundary` | dict | 31CI / 31CN | the top-level claim boundary (must assert no_q2k_blobs_generated, no_sdir_blobs_generated, no_raw_activations_generated, no_model_load, no_runtime_loader, no_runtime_integration) |
| `forbidden_claims` | list[str] | 31CI / 31CN | the top-level forbidden claims (must include the canonical 12 items) |
| `valid_as_long_as` | list[str] | 31CI / 31CN | the validity invariants (must include at least the 7 canonical items) |
| `runtime_safety_invariants` | dict | 31CI / 31CN | the 9 31CH runtime safety invariants (ffn_down_residual_enabled=false, no_activation_capture_artifact_in_runtime_path=true, no_build_ffn_patch=true, no_g_prt_sidecar_root_set_write=true, no_legacy_prt_sidecar_entries=true, normalized_planned_manifest=true, adapter_resolves_provenance_filter=true, derived_provenance_explicit=true, etc.) |
| `legacy_sidecar_exclusion` | dict | 31CI / 31CN | the 7 31CH legacy-sidecar exclusion glob patterns (prt_*, sidecar_*, g_prt_*, g_build_ffn_*, g_apply_layer_*, shadow_*, pager_*) + rejection_policy=fail_fast + rejection_error_class=LegacySidecarManifestError |
| `layers` | list[dict] | 31CI / 31CN | the per-layer per-family planned tensors (3 layers × 3 families = 9 entries) |
| `unique_layer_indices` | list[int] | 31CI / 31CN | the 3 unique layer indices (0, 14, 27 for the 1.5B Qwen2.5 sample) |
| `reconstructed_per_layer_plan` | dict | 31CN | the reconstructed per-layer plan (q2k_planned_files, sdir_planned_files, q2k_total_bytes, sdir_total_bytes for each layer) |
| `reconstructed_summary_files` | list[dict] | 31CN | the 2 summary files (manifest.dryrun.json, writer_plan.json) |
| `reconstructed_q2k_total_bytes` | int = 40,642,560 | 31CN | the reconstructed total Q2_K bytes |
| `reconstructed_sdir_total_bytes` | int = 11,147,832 | 31CN | the reconstructed total SDIR bytes |
| `reconstructed_total_bytes` | int = 51,790,392 | 31CN | the reconstructed total bytes |
| `memory_accounting` | dict | 31CN | the per-layer memory margins (ffn_up=507,468, ffn_gate=507,466, ffn_down=2,365,440 from 31CF-S authoritative reference) |
| `source_model` | dict | 31CI / 31CN | the source model identity (architecture, model_name, quantization, source_model_sha256 placeholder, source_model_size_bytes) |
| `package_id` | str | 31CI | the canonical package ID |
| `policy_name` | str = `"corrected_q2k_policy_v1"` | 31CI | the canonical policy name |

### 3.2 Required SHA-256 placeholders (the consumer would refuse to start if any is missing)

The manifest contains 9 `q2k_sha256` placeholders (one per layer-family entry) and 6 `sdir_sha256` placeholders (one per ffn_up/ffn_gate entry across 3 layers). All 15 placeholders are `metadata_only_placeholder` (the real SHA-256s would be computed at a real-write time, which 31CO does NOT do). The consumer would refuse to start if any of the 15 placeholders is missing or if any is not the string `metadata_only_placeholder` (this guards against a manifest that *claims* a real SHA-256 for a file that was never written).

### 3.3 Required normalized provenance fields (the consumer would refuse to start if any is missing)

The 31CN adapter pattern requires all of the 8 explicit derivative provenance fields (`normalized_from_phase`, `normalized_from_artifact`, `normalized_from_artifact_sha256`, `source_manifest_phase`, `source_manifest_artifact`, `source_manifest_artifact_sha256`, `validator_phase`, `validator_artifact`, `validator_artifact_sha256`, `normalized_by_phase`, `normalized_at_utc`). The consumer would refuse to start if any of these is missing, because the 31CI validator's R-provenance rules hard-require `created_by_phase='31CI'`, and the adapter pattern is the only way to satisfy that hard-requirement without lying about the derivative origin.

---

## 4. Required preflight sequence (8 ordered steps)

The consumer would run these 8 checks **in order**, halting at the first FAIL. Each check returns PASS / FAIL / BLOCKED with explicit fail-fast triggers. The preflight is **fail-fast**: any FAIL means the consumer refuses to start; any BLOCKED means the consumer refuses to start AND escalates to operator review.

### Step 1 — Read the manifest file and parse the JSON

- **Action:** `open(src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json)` + `json.load()`
- **Pass criteria:** file exists, file is valid UTF-8, JSON parses, top-level is a dict
- **Fail-fast on:** file not found; permission denied; JSON parse error; top-level not a dict
- **No-go conditions:** none (a malformed file is a clear FAIL, not a no-go)

### Step 2 — Verify the 28 required top-level fields are present (per §3.1)

- **Action:** iterate the 28 fields in §3.1, check each is present with the expected type
- **Pass criteria:** all 28 fields present, all have the expected type (bool, str, int, list, dict)
- **Fail-fast on:** any required field missing or wrong type
- **No-go conditions:** if `metadata_only != true` or `dry_run != true` or `write_binary != false`, the consumer refuses to start AND escalates (these are *no-go* conditions because they would mean the manifest is not a valid metadata-only dry-run plan; see §8)

### Step 3 — Verify the 31CI-original provenance fields are present (per §3.3)

- **Action:** check `created_by_phase == "31CI"`, `bundle_type == "runtime_loadable_substitutive"`, `schema_version == "1.1.0"`
- **Pass criteria:** all 3 fields present with the expected values
- **Fail-fast on:** any of the 3 fields missing or wrong value
- **No-go conditions:** if `created_by_phase != "31CI"`, the consumer refuses to start AND escalates (this is a *no-go* because the 31CI validator would hard-fail it; a manifest that claims to be a 31CI-original but is actually a derivative is a provenance violation; the 31CN adapter pattern is the only way to pass this check legitimately)

### Step 4 — Verify the 31CN derivative provenance fields are present (per §3.3)

- **Action:** check the 8 explicit derivative fields are all present (`normalized_from_phase`, `normalized_from_artifact`, `normalized_from_artifact_sha256`, `source_manifest_phase`, `source_manifest_artifact`, `source_manifest_artifact_sha256`, `validator_phase`, `validator_artifact`, `validator_artifact_sha256`, `normalized_by_phase`, `normalized_at_utc`)
- **Pass criteria:** all 11 fields present
- **Fail-fast on:** any of the 11 fields missing
- **No-go conditions:** none (a manifest without explicit derivative fields is a *fail*, not a no-go, because it means the derivative-extension adapter pattern was not applied)

### Step 5 — Run the 31CI validator (all 50 rules, no filtering)

- **Action:** call `src.phase31ci_manifest_validator.validate_manifest_file(src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json, metadata_only=True)`
- **Pass criteria:** `validation_result.passed == true`, `error_count == 0`, `warning_count == 0` (or `warning_count == 0` if `--strict` is the default), `rules_checked_count == 50`
- **Fail-fast on:** any error; any warning in strict mode; any rule not run (e.g. validator short-circuited)
- **No-go conditions:** if any of the 7 R-provenance rules fail, the consumer refuses to start AND escalates (this is a *no-go* because a provenance-rule failure means the derivative-extension adapter pattern was bypassed or corrupted)

### Step 6 — Run the 31CM round-trip validator (12 checks)

- **Action:** call `src.phase31cm_dry_run_output_validator` on the 31CN-normalized manifest (as if it were a 31CL output)
- **Pass criteria:** `round_trip_checks.n_passed == 12`, `round_trip_checks.n_failed == 0`, `forbidden_findings == 0`, `provenance_filters_documented == true` (or `provenance_filter_n_rules == 0` if the 31CN adapter pattern eliminated the need to filter)
- **Fail-fast on:** any of the 12 checks fail; any forbidden finding; provenance filters not documented
- **No-go conditions:** if `byte_accounting` fails (planned total bytes don't match expected), the consumer refuses to start AND escalates (this is a *no-go* because byte-accounting failures would mean the manifest is internally inconsistent and a loader would not know how much memory to allocate)

### Step 7 — Re-validate the legacy-sidecar exclusion (7 globs + 9 runtime safety invariants)

- **Action:** scan the manifest for any key matching the 7 31CH legacy-sidecar exclusion glob patterns (`prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*`, `g_apply_layer_*`, `shadow_*`, `pager_*`); verify the 9 31CH runtime safety invariants (per §3.1)
- **Pass criteria:** no key matches any of the 7 globs; all 9 runtime safety invariants present and true
- **Fail-fast on:** any key matches a legacy-sidecar exclusion glob; any runtime safety invariant missing or false
- **No-go conditions:** if any legacy-sidecar glob matches, the consumer refuses to start AND escalates to operator review (this is a *no-go* because legacy-sidecar matches would indicate the dormant Phase 11BB / 24P / 28BR-AT sidecar machinery is being reactivated, which caused the 31CF BLOCKED classification; see §8.2)

### Step 8 — Compute the loader-readiness verdict

- **Action:** evaluate the loader-readiness boundary (per §6)
- **Pass criteria:** all 6 loader-readiness checks PASS
- **Fail-fast on:** any of the 6 checks FAIL
- **No-go conditions:** if any of the 4 no-go conditions (§8) evaluate to TRUE, the consumer refuses to start AND escalates to operator review

If all 8 steps PASS, the consumer would emit a `PASS_LOADER_READINESS` verdict and report which 6 loader-readiness checks PASSed. If any step FAILs or any no-go condition is TRUE, the consumer would emit a `BLOCKED_LOADER_READINESS` verdict and refuse to start.

---

## 5. Relationship to 31CI validator, 31CM dry-run output validator, and 31CN provenance adapter

### 5.1 Relationship to 31CI validator (50 rules in 8 categories)

The 31CI validator (`src/phase31ci_manifest_validator.py`) is the **primary contract enforcer** for the manifest metadata. The 31CO preflight calls the 31CI validator at step 5 and treats its result as a hard precondition. The 31CI validator's 50 rules in 8 categories are inherited by the 31CO consumer as the **minimum-viable rule set** for the manifest to be considered loader-ready:

| 31CI category | rules | what it checks |
|---|---|---|
| schema/version | R1-R2 | `schema_version == "1.1.0"`, `bundle_type` is in the accepted list |
| policy invariants | R3-R6 | `policy_name == "corrected_q2k_policy_v1"`, `policy_version == "1"`, `q2k_mode == "corrected_ceil_per_row"`, residual_families/k_pct/alpha match canonical |
| runtime safety | R7-R15 | the 9 31CH runtime safety invariants (ffn_down_residual_enabled=false, no_build_ffn_patch=true, etc.) |
| legacy sidecar exclusion | R16-R22 | the 7 31CH legacy-sidecar exclusion glob patterns (`prt_*`, `sidecar_*`, etc.) |
| manifest hygiene | R23-R32 | forbidden path fragments, hardcoded operator paths, forbidden binary extensions, forbidden inline payload fields |
| per-layer/per-family | R33-R40 | per-family Q2_K byte budget matches reference, SDIR ffn_up/ffn_gate byte budgets match reference, ffn_down SDIR=0 |
| memory budget | R41-R45 | per-layer memory margins positive, total bytes within budget |
| provenance | R46-R50 (the 7 R-provenance rules) | created_by_phase='31CI', derived_from_phases non-empty, source_model_quantization, replay_w_ref_source, claim_boundary, forbidden_claims, valid_as_long_as |

The 31CO consumer treats the 31CI validator as the **authoritative source of truth** for what a valid manifest looks like. If the 31CI validator says the manifest is invalid, the consumer refuses to start.

### 5.2 Relationship to 31CM dry-run output validator (12 round-trip checks)

The 31CM validator (`src/phase31cm_dry_run_output_validator.py`) is the **round-trip consistency enforcer**. The 31CO preflight calls the 31CM validator at step 6 and treats its result as a second hard precondition. The 31CM validator's 12 checks are inherited by the 31CO consumer as the **round-trip consistency contract**:

| 31CM check | what it confirms |
|---|---|
| input_shape | the manifest's top-level shape matches the 31CL contract |
| planned_file_count | exactly 17 planned files (15 per-layer + 2 summary) |
| planned_file_shape | 3 layers × 3 families with ffn_down SDIR=0 |
| ffn_down_sdir_absence | no ffn_down.sdir plan exists |
| byte_accounting | per-family bytes match expected (Q2_K=4,515,840; SDIR=1,857,972 for ffn_up+ffn_gate; ffn_down=0) |
| memory_margins_positive | per-layer memory margins are positive for all 3 families |
| fail_safe_summary | 16 R_DRY_* rules present and passing |
| provenance_filter_documented | the 7 R-provenance rule IDs are explicitly documented (or, with 31CN's adapter, the filter is empty) |
| forbidden_artifact_scan | no forbidden path tokens, no forbidden binary extensions, no inline payload fields |
| claim_boundary_isolation | top-level claim_boundary asserts metadata-only scope |
| round_trip_reconstruction | reconstructed plan matches the manifest's own per_layer_plan / per_family_plan / planned_*_expected_bytes |
| sample_manifest_byte_match | 31CI sample manifest's 3 layers are a subset of the sample's unique layer indices; per-family byte budgets agree |

### 5.3 Relationship to 31CN provenance adapter (derivative-extension adapter pattern)

The 31CN normalizer's derivative-extension adapter pattern is the **only legitimate way** to satisfy the 31CI validator's R-provenance hard-requirements for a derivative manifest. The 31CO consumer:

1. **Requires** the 8 explicit derivative provenance fields (§3.3) at step 4.
2. **Rejects** any manifest that has `created_by_phase == "31CI"` but is missing the 8 derivative fields (because that would be a derivative manifest *lying* about its 31CI-original origin).
3. **Rejects** any manifest that has all 8 derivative fields but `created_by_phase != "31CI"` (because that would be a derivative manifest *not* using the adapter pattern; the 31CI validator would hard-fail it at step 5).
4. **Rejects** any manifest that uses the 31CL-style filter-the-7-rules approach (`r_provenance_filtered` non-empty) — the 31CN adapter pattern requires `r_provenance_filtered == []` and `silent_bypass_used == false`.

The 31CN normalizer's 5 pipeline steps (reconstruct from 31CL → build normalized → validate with all 50 rules → forbidden scan → round-trip consistency) are inherited by the 31CO consumer as the **5 normalization gates** that the manifest must have already passed before the consumer would consider it.

---

## 6. Loader-readiness checks (the 6 conditions for "may a future loader actually run?")

The consumer would evaluate these 6 checks **after** the 8-step preflight passes. All 6 must be true for the consumer to emit a `PASS_LOADER_READINESS` verdict. If any is false, the consumer emits a `BLOCKED_LOADER_READINESS` verdict and refuses to start.

### Check 1 — Manifest is metadata-only AND dry-run AND no-binary-write

- **Inputs:** `metadata_only == true`, `dry_run == true`, `write_binary == false`
- **Pass criteria:** all 3 booleans have the expected values
- **Fail semantics:** if any of the 3 is wrong, the consumer cannot consider the manifest loader-ready (a manifest that claims to be writable is not loader-ready in the metadata-only sense; it's writer-ready, which is a different contract)

### Check 2 — 31CI validator passes with all 50 rules, 0 errors, 0 warnings, no rule filtering

- **Inputs:** `validation_result.passed == true`, `validation_result.error_count == 0`, `validation_result.warning_count == 0` (or 0 in strict mode), `validation_result.rules_checked_count == 50`, `provenance_adapter_result.silent_bypass_used == false`, `provenance_adapter_result.r_provenance_filtered == []`
- **Pass criteria:** all 6 conditions true
- **Fail semantics:** any rule short-circuited, any error, any warning (in strict), any silent bypass, any provenance rule filter → not loader-ready

### Check 3 — 31CM round-trip consistency passes with 12/12 checks, 0 forbidden findings

- **Inputs:** `round_trip_checks.n_passed == 12`, `round_trip_checks.n_failed == 0`, `forbidden_findings == 0`, `provenance_filters_documented == true` (or `provenance_filter_n_rules == 0` with adapter)
- **Pass criteria:** all 4 conditions true
- **Fail semantics:** any check fails, any forbidden finding, provenance filters not documented → not loader-ready

### Check 4 — Legacy-sidecar exclusion scan finds 0 matches AND all 9 runtime safety invariants are true

- **Inputs:** legacy-sidecar scan finds 0 keys matching the 7 globs; all 9 runtime safety invariants present and true (`ffn_down_residual_enabled == false`, `no_activation_capture_artifact_in_runtime_path == true`, `no_build_ffn_patch == true`, `no_g_prt_sidecar_root_set_write == true`, `no_legacy_prt_sidecar_entries == true`, `normalized_planned_manifest == true`, `adapter_resolves_provenance_filter == true`, `derived_provenance_explicit == true`, plus the policy/range/quantization invariants)
- **Pass criteria:** all conditions true
- **Fail semantics:** any glob match, any invariant false → not loader-ready (and this is a *no-go* per §8.2 — see below)

### Check 5 — Byte accounting matches the 31CI sample manifest's per-family budgets

- **Inputs:** `expected_q2k_bytes == 4,515,840` per family; `expected_sdir_bytes == 1,857,972` for ffn_up/ffn_gate; `expected_sdir_bytes == 0` for ffn_down; `per_layer_q2k_total_bytes == 13,547,520` (3 families); `per_layer_sdir_total_bytes == 3,715,944` (2 residual families); `reconstructed_q2k_total_bytes == 40,642,560`; `reconstructed_sdir_total_bytes == 11,147,832`; `reconstructed_total_bytes == 51,790,392`
- **Pass criteria:** all byte values match expected (exact equality, not just ranges)
- **Fail semantics:** any byte value off by even 1 → not loader-ready (a loader would allocate the wrong memory size)

### Check 6 — All 15 SHA-256 placeholders are present AND are `metadata_only_placeholder`

- **Inputs:** 9 `q2k_sha256` placeholders (one per layer-family entry) + 6 `sdir_sha256` placeholders (one per ffn_up/ffn_gate entry across 3 layers) = 15 placeholders total, all equal to the string `metadata_only_placeholder`
- **Pass criteria:** all 15 placeholders present with the expected value
- **Fail semantics:** any placeholder missing, or any placeholder not equal to `metadata_only_placeholder` → not loader-ready (a manifest that *claims* a real SHA-256 for a file that was never written is a lie; the consumer would refuse to start)

If all 6 loader-readiness checks PASS, the consumer would emit a `PASS_LOADER_READINESS` verdict with a `next_step` recommendation. The next step would be a future 31CP-style phase that actually implements the loader — only with explicit operator approval, and only after this 31CO design is reviewed.

---

## 7. Fail-fast rules (the 12 rules a future loader would check at every loading checkpoint)

These are the **mid-load fail-fast rules** that a future loader would check at every checkpoint. They are *not* preflight checks (those are at §4); they are *runtime* checks that would be evaluated if a future loader actually attempted to load. For 31CO, these are documented as the *contract* a future loader would implement; 31CO does NOT implement the loader.

| rule ID | what it checks | fail-fast trigger | no-go condition? |
|---|---|---|---|
| R_LF_01 | the manifest's `metadata_only` is still `true` at the checkpoint | any other value at the checkpoint | NO (a manifest that flips to `false` mid-load is a writer, not a reader; refuse to load) |
| R_LF_02 | the manifest's `dry_run` is still `true` at the checkpoint | any other value at the checkpoint | NO (a manifest that flips to `false` mid-load is a writer; refuse to load) |
| R_LF_03 | the manifest's `write_binary` is still `false` at the checkpoint | any other value at the checkpoint | NO (a manifest that flips to `true` mid-load is a writer; refuse to load) |
| R_LF_04 | the loaded tensor's actual bytes match the planned byte count | bytes mismatch (planned vs actual) | NO (byte mismatch means the manifest is lying; refuse to load) |
| R_LF_05 | the loaded tensor's orientation matches `canonical_d_out_d_in` | orientation mismatch | NO (orientation mismatch means the tensor was stored in the wrong shape; refuse to load) |
| R_LF_06 | the loaded tensor's family is in `{ffn_up, ffn_gate, ffn_down}` | family mismatch | NO (family mismatch means the tensor is for a different architecture; refuse to load) |
| R_LF_07 | the loaded tensor's layer index is in `unique_layer_indices` | layer mismatch | NO (layer mismatch means the tensor is for a different layer; refuse to load) |
| R_LF_08 | no key in the loaded tensor matches a legacy-sidecar exclusion glob | any match | **YES** (§8.2) — escalate to operator review before refusing |
| R_LF_09 | the loaded tensor's `q2k_sha256` is still `metadata_only_placeholder` | any other value | **YES** (§8.3) — escalate to operator review before refusing (a real SHA-256 means a real file was written; that is out of scope for 31CO) |
| R_LF_10 | the loaded tensor's `sdir_sha256` is still `metadata_only_placeholder` | any other value | **YES** (§8.3) — same as R_LF_09 for SDIR |
| R_LF_11 | the manifest's `runtime_safety_invariants.no_build_ffn_patch` is still `true` | any other value | **YES** (§8.2) — escalate; a `false` value means a `build_ffn` patch was applied, which is the 31CF BLOCKED scenario |
| R_LF_12 | the manifest's `runtime_safety_invariants.no_g_prt_sidecar_root_set_write` is still `true` | any other value | **YES** (§8.2) — escalate; a `false` value means the legacy PRT/SDI sidecar is being reactivated |

The 8 fail-fast rules that are **also no-go conditions** (R_LF_08 through R_LF_12) trigger operator escalation, not just refusal. The consumer would emit a `BLOCKED_LOADER_READINESS_ESCALATE` verdict and present the finding to the operator for explicit review.

---

## 8. No-go conditions (the 4 conditions that always escalate to operator)

These are the conditions that, if TRUE, the consumer would **always escalate to operator review**, not just refuse to start. They are the *safety boundary* of the consumer.

### 8.1 No-go condition 1 — manifest is not metadata-only

- **Trigger:** `metadata_only != true` OR `dry_run != true` OR `write_binary != false`
- **Action:** refuse to start, escalate to operator
- **Why:** a manifest that is not metadata-only dry-run no-binary-write is a *writer* manifest, not a *reader* manifest; a consumer would not be the right tool; an operator should review whether the manifest is correctly classified

### 8.2 No-go condition 2 — legacy-sidecar / 31CF BLOCKED reactivation

- **Trigger:** any of the 7 31CH legacy-sidecar exclusion globs match a key in the manifest OR any of the 9 31CH runtime safety invariants is false (`ffn_down_residual_enabled`, `no_activation_capture_artifact_in_runtime_path`, `no_build_ffn_patch`, `no_g_prt_sidecar_root_set_write`, `no_legacy_prt_sidecar_entries`)
- **Action:** refuse to start, escalate to operator
- **Why:** a match indicates the dormant Phase 11BB / 24P / 28BR-AT sidecar machinery (`g_build_ffn_total`, `prt_get_residual_view`, `prt_shadow_apply`, `prt_shadow_contribution_synthetic`) is being reactivated; this is the EXACT scenario that caused the 31CF BLOCKED classification; an operator must review before any consumer is allowed to proceed

### 8.3 No-go condition 3 — real SHA-256 placeholders (manifest claims a real file was written)

- **Trigger:** any of the 15 `q2k_sha256` or `sdir_sha256` placeholders is not equal to the string `metadata_only_placeholder`
- **Action:** refuse to start, escalate to operator
- **Why:** a manifest that *claims* a real SHA-256 for a file is asserting that a real binary was written; 31CO is metadata-only and does NOT write binaries; a real SHA-256 in a metadata-only manifest is a *lie*; an operator must review whether the manifest is correctly classified (e.g. maybe a different phase was actually run that wrote the file, but the phase wasn't recorded)

### 8.4 No-go condition 4 — provenance rules filtered or silently bypassed

- **Trigger:** `provenance_adapter_result.silent_bypass_used == true` OR `provenance_adapter_result.r_provenance_filtered` is non-empty
- **Action:** refuse to start, escalate to operator
- **Why:** the 31CN adapter pattern requires `silent_bypass_used == false` and `r_provenance_filtered == []`; if the 7 R-provenance rules were filtered or bypassed, the derivative-extension adapter pattern was not used, which means the manifest is either (a) a 31CL-style filtered manifest (the deprecated approach) or (b) a corrupted derivative; an operator must review

---

## 9. Expected future loader inputs

If and when a future 31CP-style phase actually implements the consumer, the inputs would be:

1. **A 31CN-normalized planned manifest JSON** (the canonical input; e.g. `src/results/PHASE31CN_NORMALIZED_PLANNED_MANIFEST.json`)
2. **A path to a future 31CO-implemented consumer module** (e.g. `src/phase31cp_planned_manifest_consumer.py` — *not* in 31CO; reserved for a future phase)
3. **A path to the 31CI validator module** (e.g. `src/phase31ci_manifest_validator.py`)
4. **A path to the 31CM round-trip validator module** (e.g. `src/phase31cm_dry_run_output_validator.py`)
5. **A path to a directory containing future planned-artifact files** (the consumer would *not* read these; it would only check that the manifest's `q2k_artifact_path` / `sdir_artifact_path` fields reference them; the consumer does NOT load the bytes)
6. **A `--strict` flag** (default `true`; if `true`, the consumer treats any warning as a fail-fast)
7. **A `--escalate-cmd` flag** (optional; if set, the consumer would call this command to escalate a no-go condition to the operator; default: print to stderr)

The consumer would NOT take any of these as inputs (these are out of scope for 31CO):

- A model file path (no model load)
- A GGUF file path (no GGUF inspection)
- A Q2_K blob path (no Q2_K loading)
- A SDIR blob path (no SDIR loading)
- A raw activation path (no raw activation loading)
- A llama.cpp path (no llama.cpp integration)
- A compile flag (no compiled binary)
- A write flag (no write)

---

## 10. Expected future loader outputs

If and when a future 31CP-style phase actually implements the consumer, the outputs would be:

1. **A preflight report JSON** (e.g. `src/results/PHASE31CP_PREFLIGHT_REPORT.json`) — the structured output of the 8-step preflight (§4) and the 6 loader-readiness checks (§6), with each step/check having a `passed: bool`, a `message: str`, and a `details: dict`
2. **A loader-readiness verdict** (e.g. `PASS_LOADER_READINESS`, `BLOCKED_LOADER_READINESS`, `BLOCKED_LOADER_READINESS_ESCALATE`) — the top-level verdict emitted at step 8
3. **A no-go findings list** (if any no-go conditions are TRUE) — each finding has a `condition_id`, a `triggered_by` field/value, and an `operator_action_required: true` flag
4. **A next-step recommendation** (if `PASS_LOADER_READINESS`) — the recommended next phase (e.g. "Phase 31CQ — Metadata-Only Loader Pre-Flight Actual Loader Smoke Test" — proposed, *not* in scope for 31CO)
5. **Stdout summary** — a one-line summary (e.g. `31CP preflight: PASS_LOADER_READINESS in 0.42s; 8/8 preflight steps, 6/6 loader-readiness checks, 0 no-go conditions`)

The consumer would NOT produce any of these (out of scope for 31CO):

- A modified manifest (no writes)
- A new manifest (no writes)
- A loaded tensor (no model load)
- A computed SHA-256 (no real writes)
- A binary artifact (no binary writes)
- A compiled consumer module (no compile)

---

## 11. Forbidden runtime behavior

A future loader (e.g. 31CP) implementing this 31CO design would be **forbidden** from:

1. **Loading any model file** — no opening Q4_K_M GGUF, no opening HF safetensors, no opening any binary
2. **Inspecting any GGUF tensor payload** — no reading tensor bytes, no dequantizing, no decoding
3. **Generating any Q2_K or SDIR binary artifact** — no writes to disk, no `os.write`, no `open(path, 'wb')` for binary content
4. **Generating any raw activation array** — no numpy/torch arrays of activations
5. **Creating any runtime-loadable blob** — no `.gguf`, no `.safetensors`, no `.sdiw`, no `.sdir`
6. **Implementing an actual runtime loader** — no consumer code that would *load* bytes, only consumer code that would *read* metadata
7. **Modifying llama.cpp** — no patches to `~/llama.cpp/src/`, no `git -C ~/llama.cpp` operations
8. **Rebuilding llama.cpp** — no `cmake`, no `make`, no `ninja` in `~/llama.cpp/`
9. **Running generation or inference** — no `llama_decode`, no `model.generate`, no `model.forward`
10. **Running inference-quality tests** — no perplexity, no BLEU, no accuracy metrics
11. **Committing any model file, compiled binary, raw tensor, or Q2_K/SDIR blob** — even if the consumer incidentally produces such files (e.g. as a side effect of `python3 -c`), they must be `.gitignore`d or deleted before any commit
12. **Creating or pushing a tag** — the consumer would not tag
13. **Committing, pushing, or tagging without explicit operator approval** — the consumer would stop at PRE-COMMIT REPORT and wait
14. **Silently bypassing provenance rules** — the 31CN adapter pattern requires `silent_bypass_used == false` and `r_provenance_filtered == []`; the consumer would not override these
15. **Enabling the legacy PRT/SDI sidecar** — no enabling `prt_get_residual_view`, no enabling `prt_shadow_apply`, no enabling `prt_shadow_contribution_synthetic`, no enabling `g_build_ffn_total` counters
16. **Re-activating the 31CF BLOCKED scenario** — no patches to `build_ffn`, no patches to `cb(ffn_inp, ...)` that would capture the wrong tensor

---

## 12. Legacy PRT/SDI sidecar exclusion boundary

The 31CN-normalized manifest is required to have:

1. **The 7 31CH legacy-sidecar exclusion glob patterns** declared in `legacy_sidecar_exclusion.excluded_keys`: `prt_*`, `sidecar_*`, `g_prt_*`, `g_build_ffn_*`, `g_apply_layer_*`, `shadow_*`, `pager_*`
2. **The 9 31CH runtime safety invariants** declared in `runtime_safety_invariants` and all set to `true` (except `ffn_down_residual_enabled` which is `false` per the policy):
   - `ffn_down_residual_enabled: false` (corrected_q2k_policy_v1 disables ffn_down residual)
   - `no_activation_capture_artifact_in_runtime_path: true`
   - `no_build_ffn_patch: true`
   - `no_g_prt_sidecar_root_set_write: true`
   - `no_legacy_prt_sidecar_entries: true`
   - `normalized_planned_manifest: true`
   - `adapter_resolves_provenance_filter: true`
   - `derived_provenance_explicit: true`
3. **Zero matches** when the consumer scans the manifest for any key matching the 7 globs
4. **`rejection_policy: "fail_fast"`** and **`rejection_error_class: "LegacySidecarManifestError"`** declared in `legacy_sidecar_exclusion`

If any of the 4 conditions is false, the consumer would emit a `BLOCKED_LOADER_READINESS_ESCALATE` verdict and present the no-go finding to the operator.

The 31CF BLOCKED classification (DORMANT_SIDECAR_MACHINERY_INTERFERENCE) is the **explicit reference scenario** for this boundary. The dormant sidecar machinery (`g_build_ffn_total`, `prt_get_residual_view`, `prt_shadow_apply`, `prt_shadow_contribution_synthetic`) in `~/llama.cpp/src/llama-graph.cpp` `build_ffn` (line 1844) is what 31CH's 9 runtime safety invariants + 7 legacy-sidecar exclusion globs defend against. The 31CO consumer inherits this defense wholesale.

---

## 13. Artifact hygiene rules (the 31CO preflight + loader-readiness rules, when considered as artifact hygiene)

The 31CO consumer's preflight and loader-readiness checks are themselves a form of artifact hygiene: they prevent the consumer from proceeding with a manifest that contains forbidden artifacts. Specifically:

1. **No `.gguf`, `.safetensors`, `.pt`, `.pth`, `.onnx`, `.npy`, `.npz`, `.bin`, `.raw` in any `planned_path` / `q2k_artifact_path` / `sdir_artifact_path` field** — the 31CI validator's R23-R32 manifest hygiene rules check this; the consumer re-checks at step 7
2. **No `/tmp/` paths in any field** — the 31CI validator's R45 checks this
3. **No hardcoded operator paths** (`/home/matthew-villnave`, `/media/matthew-villnave`, `VL_usb`, `~/llama.cpp/build/`) in any field — the 31CI validator's R40-R44 check this
4. **No inline payload fields** (`raw_payload`, `raw_payload_bytes`, `w_low_raw`, `sdir_raw`, `activation_array`, `raw_x`, `raw_y`, `raw_r`) — the 31CI validator's R25 checks this
5. **No `__pycache__` or `.pyc` files** — out of scope for the consumer (the consumer would not write any Python), but the planning doc itself is pure markdown with no Python
6. **No compiled binaries (`.o`, `.so`, `.a`)** — the consumer would not write any compiled binary; the planning doc is pure markdown
7. **No `model.safetensors`, `libllama.so`, `llama-server` references** — the 31CI validator's R41-R44 check this
8. **No `/build/` path fragments** — the 31CI validator's R44 checks this
9. **No `llama.cpp` source/build artifact references** — the consumer would not include any
10. **No raw activation arrays** — the consumer would not generate any
11. **No Q2_K / SDIR blob references that are not `.plan.json` wrappers** — the 31CM round-trip validator's check 9 (forbidden_artifact_scan) checks this with explicit exemption for `.plan.json` plan wrappers

The 31CO planning doc and JSON are themselves artifact-hygiene-clean: they are pure markdown + JSON, with no Python source, no compiled binary, no model file, no Q2_K/SDIR blob, no raw activation.

---

## 14. Claim-boundary rules (the 18 forbidden claims + 10 allowed cautious phrases for 31CO)

The 31CO consumer's preflight report and loader-readiness verdict must use only the following 18 forbidden claims and 10 allowed cautious phrases (per the `sdi-claim-boundary-scan` skill's catalog):

### 14.1 Forbidden claims (the 18 families)

1. **Model quality recovery claim** — the consumer would not say "the model fully recovers its quality"
2. **Behavior recovery claim** — the consumer would not say "the model behavior is fully recovered"
3. **Speedup claim** — the consumer would not say "the loader is faster than X"
4. **Production readiness claim** — the consumer would not say "the loader is production-ready"
5. **Runtime integration claim** — the consumer would not say "the loader is integrated with runtime X" (unless actual runtime integration was implemented, which 31CO does NOT do)
6. **Live runtime memory savings claim** — the consumer would not say "live runtime memory savings of N MB"
7. **Generation / inference quality claim** — the consumer would not say "inference quality is preserved"
8. **All-layer / all-token / all-prompt transfer claim** — the consumer would not say "all layers were validated"
9. **Broader model transfer claim** — the consumer would not say "the result transfers to any model"
10. **"Real activations behave like synthetic Gaussian"** — the consumer would not say this
11. **"HF == GGUF"** — the consumer would not say "HF activations are identical to GGUF activations"
12. **"Same activation values"** — the consumer would not say this
13. **"Identical to 31CD"** — the consumer would not say this
14. **"Bit-equal"** — the consumer would not say this (unless actually proven, which 31CO does NOT do)
15. **"Exact Q4_K_M runtime activation" in Option A contexts** — the consumer would not say this
16. **"Loader implemented"** — the consumer would not say this in 31CO (the consumer is *planned*, not *implemented*; a future 31CP would be implemented, only with operator approval)
17. **"Artifact generated"** — the consumer would not say this in metadata-only contexts
18. **"Proves future runtime viability"** — the consumer would not say this (feasibility study only)

### 14.2 Allowed cautious phrases (the 10 standard phrases)

1. "metadata-only"
2. "planning-only"
3. "directionally consistent by sign pattern"
4. "contextual comparison only"
5. "not bit-equal"
6. "not an equivalence claim"
7. "standalone tensor harness"
8. "no runtime integration"
9. "no generation"
10. "no model quality claim"

### 14.3 Allowed 31CO-specific phrases (additions to the standard 10)

11. "consumer is planning-only; no loader implemented"
12. "loader-readiness verdict is metadata-only; no actual load attempted"
13. "preflight is fail-fast; no go-no-go decision is final until operator review"
14. "31CO does not prove future runtime viability"
15. "31CO does not enable the legacy PRT/SDI sidecar"

The 31CO planning doc and JSON use only the 10 standard + 5 31CO-specific cautious phrases; the 18 forbidden claims are not used.

---

## 15. Recommended next phase

**Phase 31CP (proposed) — Metadata-Only Planned Manifest Consumer / Loader Preflight Implementation.** The 31CO planning doc and JSON describe the contract; a future 31CP would actually implement the consumer in Python (`src/phase31cp_planned_manifest_consumer.py`), calling the 31CI validator at step 5 and the 31CM round-trip validator at step 6. The implementation would:

- Run the 8-step preflight sequence (§4)
- Evaluate the 6 loader-readiness checks (§6)
- Document the 12 fail-fast rules (§7) as in-code assertions
- Document the 4 no-go conditions (§8) as in-code operator-escalation triggers
- Inherit the 7 31CH legacy-sidecar exclusion globs and the 9 31CH runtime safety invariants from the 31CI validator
- Produce a preflight report JSON (similar in shape to the 31CM `PHASE31CM_DRY_RUN_OUTPUT_VALIDATION.json`)
- Stop at PRE-COMMIT REPORT and wait for explicit operator approval before any commit/push/tag

**31CP is NOT in scope for 31CO.** 31CO is a planning-only design phase; 31CP would be a metadata-only implementation phase, only with explicit operator approval, and only if the operator chooses to advance. The operator's 3-strike patch review pattern would apply to 31CP just as it has applied to every prior phase.

Alternative next phases (if 31CP is not requested):

- **31CP-R** (Loader Preflight Implementation Repair, only if 31CP is rejected post-merge or incomplete)
- **31CG** (Larger Prompt/Token Sensitivity Planning, planning-only, only if explicitly requested)
- **31CR** (Loader Pre-Flight Actual Loader Smoke Test, only after 31CP passes, only with operator approval)

All next phases require explicit operator approval at entry; the agent does NOT proceed to any without a new request.

---

## 16. Compliance with the operator's strict non-goals

The 31CO phase complies with every strict non-goal in the operator's prompt:

| non-goal | 31CO compliance |
|---|---|
| no runtime loader implementation | ✓ no Python source file `src/phase31co_*.py` is created |
| no live llama.cpp integration | ✓ no llama.cpp interaction of any kind |
| no llama.cpp source modification | ✓ no `~/llama.cpp/src/` modifications |
| no llama.cpp rebuild | ✓ no `cmake`, no `make`, no `ninja` |
| no model load | ✓ no opening of Q4_K_M GGUF, no opening of HF safetensors |
| no inference | ✓ no `llama_decode`, no `model.generate` |
| no generation | ✓ no token generation, no sampling |
| no Q2_K encoding | ✓ no writing of Q2_K binary artifacts |
| no SDIR encoding | ✓ no writing of SDIR binary artifacts |
| no raw activation capture | ✓ no writing of raw activation arrays |
| no real artifact generation | ✓ no `.gguf`, no `.safetensors`, no `.sdiw`, no `.sdir` writes |
| no binary writer | ✓ no `open(path, 'wb')` for binary content |
| no performance measurement | ✓ no timing, no throughput, no latency |
| no quality evaluation | ✓ no perplexity, no BLEU, no accuracy |
| no production readiness claim | ✓ planning-only; no production claim |
| no commit | ✓ stops at PRE-COMMIT REPORT |
| no push | ✓ no `git push` |
| no tag | ✓ no `git tag` |
| no start of next phase (31CP, 31CG, 31CR) | ✓ next-phase-not-started rule |

---

## 17. End of planning doc

This document is the **31CO planning doc only**. No Python source has been created. No SOT has been modified. No scientific code has been modified. No result JSONs have been modified. The only deliverables are:

1. This planning document: `docs/PHASE31CO_PLANNED_MANIFEST_CONSUMER_LOADER_PREFLIGHT_PLANNER.md`
2. The metadata-only planning summary JSON: `src/results/PHASE31CO_PLANNED_MANIFEST_CONSUMER_LOADER_PREFLIGHT_PLANNER.json`

The next step (when the operator approves) is to commit this 31CO work via the standard 3-strike PRE-COMMIT REPORT cycle, then push to `origin/master`. After 31CO commits, a future 31CP phase (only with explicit operator approval) would implement the consumer in Python.
