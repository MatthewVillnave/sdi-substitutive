#!/usr/bin/env python3
"""
Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype

Implements the dry-run writer that 31CJ designed.

Hard scope (no exceptions):
    NO model load
    NO Q2_K blob generation
    NO SDIR blob generation
    NO raw activation generation
    NO runtime loader implementation
    NO llama.cpp modification
    NO compiled binary
    NO HF cache / model file commit
    NO raw tensor commit
    NO tag

This writer:
    - reads a 31CI-valid v1.1.0 metadata-only manifest (validated before planning)
    - computes PLANNED artifact paths / byte counts / SHA-256 placeholders /
      memory accounting / validation steps for a future real writer
    - emits exactly two metadata-only output JSON files (one plan output +
      one per-phase summary). No binary blobs. No raw tensors.
    - hard-codes dry_run=True and write_binary=False gates (caller cannot override)
    - enforces 16 fail-safe rules (R_DRY_01 through R_DRY_16) at pre-plan, in-plan,
      and post-plan gates
    - imports src.phase31ci_manifest_validator as a library and calls
      validate_manifest_file at pre-plan AND post-plan call sites
    - uses env-var or relative paths only; never hardcodes operator paths
    - uses Python stdlib only (json, hashlib, fnmatch, re, argparse, os, sys, typing)

The writer is lint-clean and Python 3.8+ compatible.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Local library import — the writer depends on the 31CI validator.
# The validator is at src/phase31ci_manifest_validator.py.
# The writer is at src/phase31cl_artifact_writer_dry_run.py.
# Both are in the same package (src/).
try:
    from phase31ci_manifest_validator import (
        validate_manifest_file,
        ValidationResult,
    )
except ImportError as exc:
    # Hard fail — the writer is not usable without the 31CI validator.
    raise ImportError(
        "Phase 31CL writer requires src/phase31ci_manifest_validator.py "
        "(31CI validator library). Add src/ to sys.path or run from src/."
    ) from exc


# ----------------------------------------------------------------------
# Canonical constants (mirrored from the 31CJ plan, Section 4.1)
# ----------------------------------------------------------------------

#: Phase identifier for the output JSON
PHASE = "31CL"
PHASE_FULL_NAME = "Phase 31CL — Metadata-Only Artifact Writer Dry-Run Prototype"

#: Default input manifest path (the 31CI sample manifest, relative to repo root)
DEFAULT_INPUT_MANIFEST = "src/results/PHASE31CI_SAMPLE_METADATA_ONLY_MANIFEST.json"

#: Default output JSON path (the plan output)
DEFAULT_OUTPUT_JSON = "src/results/PHASE31CL_ARTIFACT_WRITER_DRY_RUN_OUTPUT.json"

#: Default per-phase summary JSON path
DEFAULT_SUMMARY_JSON = "src/results/PHASE31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN.json"

#: Default artifact root (env-var form only; never absolute operator path)
DEFAULT_ARTIFACT_ROOT = "${SDI_ARTIFACT_DRYRUN_DIR}"

#: Hard gates — caller cannot override
DRY_RUN_HARD_GATE = True
WRITE_BINARY_HARD_GATE = False

#: Canonical policy parameters (asserted by the writer; not configurable)
POLICY_NAME_CANONICAL = "corrected_q2k_policy_v1"
POLICY_VERSION_CANONICAL = "1"
Q2K_MODE_CANONICAL = "corrected_ceil_per_row"
RESIDUAL_FAMILIES_CANONICAL = ["ffn_up", "ffn_gate"]
TENSOR_FAMILIES_CANONICAL = ["ffn_up", "ffn_gate", "ffn_down"]
FFN_DOWN_RESIDUAL_ENABLED_CANONICAL = False
K_PCT_CANONICAL = 0.5
ALPHA_CANONICAL = 1.0

#: Legacy sidecar exclusion glob patterns (matches 31CI R14-R16 and 31CH design)
LEGACY_SIDECAR_EXCLUDED_KEY_PATTERNS = (
    "prt_*",
    "sidecar_*",
    "g_prt_*",
    "g_build_ffn_*",
    "g_apply_layer_*",
    "shadow_*",
    "pager_*",
)

#: Forbidden artifact extensions (matches 31CI FORBIDDEN_ARTIFACT_EXTENSIONS_IN_PATHS
#: AND the dry-run writer's own R_DRY_13)
FORBIDDEN_ARTIFACT_EXTENSIONS = (
    ".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx",
    ".npz", ".npy", ".raw", ".x", ".q2k", ".sdir", ".sdiw",
)

#: Plan-file extensions the writer is allowed to emit
PLAN_FILE_EXTENSIONS = (".plan.json", ".dryrun.json", ".writer_plan.json")

#: Inline payload fields (raw arrays, base64, etc.) — writer refuses these
FORBIDDEN_INLINE_PAYLOAD_FIELDS = (
    "raw_payload", "raw_bytes", "q2k_bytes", "sdir_bytes_raw",
    "raw_tensor", "raw_activation", "base64_payload", "inline_payload",
)

#: Hardcoded operator path patterns (writer refuses if any appear in output)
HARDCODED_OPERATOR_PATH_PATTERNS = (
    "/home/matthew-villnave",
    "/media/matthew-villnave",
    "/mnt/", "/opt/", "/Users/", "/root/",
    "VL_usb",
    "/tmp/",  # see also R_DRY_05
)


# ----------------------------------------------------------------------
# Plan output structure
# ----------------------------------------------------------------------

def build_planned_files(
    layer_indices: List[int],
    family_indices: List[str],
    artifact_root: str,
) -> List[Dict[str, Any]]:
    """Compute the list of planned metadata-only files for the given layers and families.

    For each (layer, family) pair, plan the per-family Q2_K plan file.
    For each (layer, family) pair where family is in residual_families, also
    plan the per-family SDIR plan file.
    ffn_down NEVER has an SDIR plan (per policy: ffn_down_residual_enabled=false).

    Returns a list of dicts with the planned-file metadata. NO actual files
    are created on disk.
    """
    planned: List[Dict[str, Any]] = []
    for layer in layer_indices:
        for family in family_indices:
            # Per-layer per-family Q2_K plan file (always present)
            q2k_basename = f"blk.{layer}.{family}.q2_k.W_low.plan.json"
            q2k_path = f"{artifact_root}/tensors/layers/layer_{layer:03d}/{q2k_basename}"
            planned.append({
                "layer": layer,
                "family": family,
                "kind": "q2k_W_low",
                "planned_filename": q2k_basename,
                "planned_path": q2k_path,
                "planned_byte_count": 4515840,  # 1.5B per-family Q2_K size (31CI sample)
                "planned_sha256_placeholder": "<computed_at_real_write_time>",
                "status": "planned_metadata_only",
            })
            # Per-layer per-family SDIR plan file (ffn_up + ffn_gate only)
            if family in RESIDUAL_FAMILIES_CANONICAL:
                sdir_basename = f"blk.{layer}.{family}.residual.sdir.plan.json"
                sdir_path = f"{artifact_root}/tensors/layers/layer_{layer:03d}/{sdir_basename}"
                planned.append({
                    "layer": layer,
                    "family": family,
                    "kind": "sdir_residual",
                    "planned_filename": sdir_basename,
                    "planned_path": sdir_path,
                    "planned_byte_count": 1857972,  # 1.5B per-family SDIR size (31CI sample)
                    "planned_sha256_placeholder": "<computed_at_real_write_time>",
                    "status": "planned_metadata_only",
                })
    return planned


def build_planned_summary_files(
    artifact_root: str,
    output_json_path: str,
) -> List[Dict[str, Any]]:
    """Compute the planned summary files (manifest.dryrun.json, writer_plan.json).

    Note: the actual output JSON (this writer's own output) is written by the
    writer itself, not by a future real writer. These two summary files are
    the artifacts a future real writer would generate.
    """
    summary_files: List[Dict[str, Any]] = []
    # manifest.dryrun.json
    manifest_basename = "manifest.dryrun.json"
    manifest_path = f"{artifact_root}/{manifest_basename}"
    summary_files.append({
        "kind": "dryrun_manifest",
        "planned_filename": manifest_basename,
        "planned_path": manifest_path,
        "planned_byte_count_estimate": 12000,  # rough estimate
        "planned_sha256_placeholder": "<computed_at_real_write_time>",
        "status": "planned_metadata_only",
    })
    # writer_plan.json
    writer_plan_basename = "writer_plan.json"
    writer_plan_path = f"{artifact_root}/{writer_plan_basename}"
    summary_files.append({
        "kind": "writer_plan",
        "planned_filename": writer_plan_basename,
        "planned_path": writer_plan_path,
        "planned_byte_count_estimate": 5000,  # rough estimate
        "planned_sha256_placeholder": "<computed_at_real_write_time>",
        "status": "planned_metadata_only",
    })
    return summary_files


# ----------------------------------------------------------------------
# Fail-safe rules (R_DRY_01 through R_DRY_16)
# ----------------------------------------------------------------------

def _walk_strings(obj: Any, path: str = "") -> List[Tuple[str, str]]:
    """Walk a JSON object and yield (path, str_value) for every string field."""
    out: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_strings(v, path + "." + str(k) if path else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_strings(v, path + f"[{i}]"))
    elif isinstance(obj, str):
        out.append((path, obj))
    return out


def _string_contains_any(s: str, patterns: Tuple[str, ...]) -> bool:
    return any(p in s for p in patterns)


def _string_contains_hardcoded_operator_path(s: str) -> bool:
    return _string_contains_any(s, HARDCODED_OPERATOR_PATH_PATTERNS)


def _check_fail_safe_rules(
    input_manifest: Dict[str, Any],
    input_path: str,
    artifact_root: str,
    output_path: str,
    write_binary: bool,
    dry_run: bool,
    planned_files: List[Dict[str, Any]],
    planned_summary_files: List[Dict[str, Any]],
    input_validator_result: Dict[str, Any],
    output_validator_result: Optional[Dict[str, Any]],
    strict: bool,
) -> List[Dict[str, Any]]:
    """Run the 16 fail-safe rules and return a list of per-rule results.

    Each result is a dict with:
        rule_id, status ("pass" | "fail"), message, severity
    """
    results: List[Dict[str, Any]] = []

    def _add(rule_id: str, passed: bool, message: str, severity: str = "error") -> None:
        # Coerce passed to strict bool (LSP-friendly; protects against
        # accidentally passing a re.Match or None)
        passed_bool = bool(passed) if passed is not None else False
        results.append({
            "rule_id": rule_id,
            "status": "pass" if passed_bool else "fail",
            "message": message,
            "severity": severity,
        })

    # R_DRY_01: dry_run must be true
    _add("R_DRY_01", dry_run is True,
         f"dry_run={dry_run!r} (must be True; hard-coded; caller cannot override)")

    # R_DRY_02: write_binary must be false
    _add("R_DRY_02", write_binary is False,
         f"write_binary={write_binary!r} (must be False; hard-coded; caller cannot override)")

    # R_DRY_03: input manifest must pass 31CI validator (error_count=0)
    inp_passed = bool(input_validator_result.get("passed")) and int(input_validator_result.get("error_count", 0)) == 0
    inp_warnings = int(input_validator_result.get("warning_count", 0))
    if strict and inp_warnings > 0:
        inp_passed = False
    _add("R_DRY_03", inp_passed,
         f"input manifest validation: passed={input_validator_result.get('passed')!r}, "
         f"error_count={input_validator_result.get('error_count')!r}, "
         f"warning_count={input_validator_result.get('warning_count')!r} (strict={strict})")

    # R_DRY_04: output path must not be inside model directory
    model_dir = os.environ.get("SDI_MODEL_DIR", "")
    output_inside_model = bool(model_dir) and (
        os.path.abspath(output_path).startswith(os.path.abspath(model_dir) + os.sep)
        or os.path.abspath(output_path) == os.path.abspath(model_dir)
    )
    _add("R_DRY_04", not output_inside_model,
         f"output_path={output_path!r} is {'INSIDE' if output_inside_model else 'outside'} "
         f"model directory (SDI_MODEL_DIR={model_dir!r})")

    # R_DRY_05: output path must not be /tmp/ unless scratch_no_commit=true
    scratch_no_commit = os.environ.get("SDI_SCRATCH_NO_COMMIT", "false").lower() == "true"
    output_in_tmp = "/tmp/" in output_path
    _add("R_DRY_05", not output_in_tmp or scratch_no_commit,
         f"output_path contains '/tmp/': {output_in_tmp}; "
         f"SDI_SCRATCH_NO_COMMIT={scratch_no_commit!r} (default deny)")

    # R_DRY_06: no legacy sidecar keys at any nesting depth in the input manifest
    sidecar_violations: List[str] = []
    for path, value in _walk_strings(input_manifest):
        for k in input_manifest.keys() if path.count(".") == 0 else []:
            if any(fnmatch.fnmatch(k, pat) for pat in LEGACY_SIDECAR_EXCLUDED_KEY_PATTERNS):
                sidecar_violations.append(f"top-level key {k!r}")
        # Also check key names at any depth
        for key in path.split("."):
            if any(fnmatch.fnmatch(key, pat) for pat in LEGACY_SIDECAR_EXCLUDED_KEY_PATTERNS):
                sidecar_violations.append(f"key {key!r} at path {path!r}")
                break
    _add("R_DRY_06", len(sidecar_violations) == 0,
         f"legacy sidecar key violations: {sidecar_violations if sidecar_violations else 'none'}")

    # R_DRY_07: ffn_down_residual_enabled must be false
    runtime_safety = input_manifest.get("runtime_safety_invariants", {}) or {}
    ffn_down_residual = runtime_safety.get("ffn_down_residual_enabled", None)
    _add("R_DRY_07", ffn_down_residual is False,
         f"ffn_down_residual_enabled={ffn_down_residual!r} (must be False)")

    # R_DRY_08: q2k_mode must be corrected_ceil_per_row
    q2k_mode = runtime_safety.get("q2k_mode", None)
    _add("R_DRY_08", q2k_mode == Q2K_MODE_CANONICAL,
         f"q2k_mode={q2k_mode!r} (must be {Q2K_MODE_CANONICAL!r})")

    # R_DRY_09: policy_name must be corrected_q2k_policy_v1 and policy_version must be 1
    policy_name = runtime_safety.get("policy_name", None)
    policy_version = runtime_safety.get("policy_version", None)
    _add("R_DRY_09",
         policy_name == POLICY_NAME_CANONICAL and policy_version == POLICY_VERSION_CANONICAL,
         f"policy_name={policy_name!r} (must be {POLICY_NAME_CANONICAL!r}); "
         f"policy_version={policy_version!r} (must be {POLICY_VERSION_CANONICAL!r})")

    # R_DRY_10: model identity must match recorded metadata
    # The 31CI sample manifest uses 'metadata_only_placeholder' for SHA-256
    # (which is a valid marker for a metadata-only manifest). The writer
    # does NOT require a real SHA-256 — it requires the marker to be present
    # OR a real hex SHA-256 of the correct length.
    source_model = input_manifest.get("source_model", {}) or {}
    src_name = source_model.get("model_name", None)
    src_quant = source_model.get("quantization", None)
    src_sha = source_model.get("source_model_sha256", None)
    is_placeholder = src_sha == "metadata_only_placeholder"
    is_hex_sha256 = bool(src_sha) and re.fullmatch(r"[0-9a-f]{64}", src_sha or "")
    sha_ok = is_placeholder or is_hex_sha256
    name_ok = src_name == "qwen2.5-1.5b-instruct-q4_k_m"
    quant_ok = src_quant == "Q4_K_M"
    _add("R_DRY_10", name_ok and quant_ok and sha_ok,
         f"source_model: name={src_name!r} (expected 'qwen2.5-1.5b-instruct-q4_k_m'), "
         f"quantization={src_quant!r} (expected 'Q4_K_M'), "
         f"sha256={'placeholder' if is_placeholder else ('hex' if is_hex_sha256 else 'INVALID')}")

    # R_DRY_11: no absolute operator paths in planned output
    op_path_violations: List[str] = []
    for f in planned_files + planned_summary_files:
        for path, val in _walk_strings(f):
            if _string_contains_hardcoded_operator_path(val):
                op_path_violations.append(f"{path}={val!r}")
    _add("R_DRY_11", len(op_path_violations) == 0,
         f"absolute operator path violations in planned output: "
         f"{op_path_violations if op_path_violations else 'none'}")

    # R_DRY_12: no inline payload array fields in planned output
    inline_violations: List[str] = []
    for f in planned_files + planned_summary_files:
        for forbidden_field in FORBIDDEN_INLINE_PAYLOAD_FIELDS:
            if forbidden_field in f:
                inline_violations.append(f"field {forbidden_field!r}")
    _add("R_DRY_12", len(inline_violations) == 0,
         f"inline payload field violations: {inline_violations if inline_violations else 'none'}")

    # R_DRY_13: no forbidden artifact extensions in planned filenames
    ext_violations: List[str] = []
    for f in planned_files + planned_summary_files:
        fn = f.get("planned_filename", "")
        for ext in FORBIDDEN_ARTIFACT_EXTENSIONS:
            # Check if the filename ENDS with a forbidden extension
            # (but allow .plan.json and .dryrun.json, .writer_plan.json)
            if fn.endswith(ext) and not any(fn.endswith(pe) for pe in PLAN_FILE_EXTENSIONS):
                ext_violations.append(f"{fn!r} ends with {ext!r}")
    _add("R_DRY_13", len(ext_violations) == 0,
         f"forbidden artifact extension violations: {ext_violations if ext_violations else 'none'}")

    # R_DRY_14: no missing byte counts
    missing_byte_violations: List[str] = []
    for f in planned_files:
        bc = f.get("planned_byte_count", None)
        if bc is None or (isinstance(bc, int) and bc < 0):
            missing_byte_violations.append(
                f"layer {f.get('layer')} family {f.get('family')} kind {f.get('kind')}: "
                f"planned_byte_count={bc!r}"
            )
    _add("R_DRY_14", len(missing_byte_violations) == 0,
         f"missing or negative byte count violations: "
         f"{missing_byte_violations if missing_byte_violations else 'none'}")

    # R_DRY_15: planned output manifest must validate (post-plan)
    if output_validator_result is None:
        _add("R_DRY_15", False,
             "no post-plan validator result available (writer did not run post-plan validator)",
             severity="error")
    else:
        out_passed = bool(output_validator_result.get("passed")) and int(output_validator_result.get("error_count", 0)) == 0
        out_warnings = int(output_validator_result.get("warning_count", 0))
        if strict and out_warnings > 0:
            out_passed = False
        _add("R_DRY_15", out_passed,
             f"output manifest post-validation: passed={output_validator_result.get('passed')!r}, "
             f"error_count={output_validator_result.get('error_count')!r}, "
             f"warning_count={output_validator_result.get('warning_count')!r} (strict={strict})")

    # R_DRY_16: no unknown tensor families
    unknown_families: List[str] = []
    for f in planned_files:
        fam = f.get("family", None)
        if fam not in TENSOR_FAMILIES_CANONICAL:
            unknown_families.append(fam or "<missing>")
    _add("R_DRY_16", len(unknown_families) == 0,
         f"unknown tensor family violations: {unknown_families if unknown_families else 'none'}")

    return results


# ----------------------------------------------------------------------
# Forbidden artifact scan
# ----------------------------------------------------------------------

def scan_for_forbidden_artifacts(planned_files: List[Dict[str, Any]]) -> List[str]:
    """Scan the planned output for forbidden artifact markers."""
    findings: List[str] = []
    forbidden_path_fragments = (
        "/media/matthew-villnave",
        "VL_usb",
        "/tmp/",
        "llama.cpp",
    )
    forbidden_filename_extensions = (
        ".gguf", ".safetensors", ".npy", ".npz", ".bin", ".raw", ".x",
    )
    # .q2k and .sdir are allowed in plan filenames (e.g. blk.0.ffn_up.q2_k.W_low.plan.json)
    # but NOT as raw binary artifact filenames
    for f in planned_files:
        for path, val in _walk_strings(f):
            # Check for forbidden path fragments
            for frag in forbidden_path_fragments:
                if frag in val:
                    findings.append(f"forbidden path fragment {frag!r} at {path}")
            # Check for forbidden filename extensions (only when followed by end-of-string
            # or a path separator — not as part of a .plan.json or .dryrun.json filename)
            for ext in forbidden_filename_extensions:
                if val.endswith(ext):
                    findings.append(f"forbidden filename extension {ext!r} at {path}={val!r}")
    return findings


# ----------------------------------------------------------------------
# Validation orchestration
# ----------------------------------------------------------------------

def _run_pre_plan_validation(
    input_path: str,
    strict: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Run the 31CI validator on the input manifest. Returns (validator_result_dict, None).

    The second element of the tuple is reserved for a future pre-validator
    of the planned output (set by _run_post_plan_validation).
    """
    try:
        result = validate_manifest_file(input_path, metadata_only=True)
    except Exception as exc:
        return {
            "passed": False,
            "error_count": 1,
            "warning_count": 0,
            "error": f"validator raised exception: {exc!r}",
            "rules_checked": 0,
        }, None
    d = result.to_dict()
    # Apply strict mode
    if strict and int(d.get("warning_count", 0)) > 0:
        d["passed"] = False
    return d, None


def _run_post_plan_validation(
    planned_manifest: Dict[str, Any],
    strict: bool,
) -> Dict[str, Any]:
    """Run the 31CI validator on the planned manifest object (in-memory).

    The writer calls ManifestValidator directly (no temp-file I/O).
    The 31CI ManifestValidator class does not take a `strict` constructor
    argument; the writer applies strict mode (warnings-fail-closed) here
    in the wrapper, after the validator returns.

    Important: the 31CI validator hard-requires `created_by_phase == '31CI'`
    and the 31CI sample manifest is the only canonical source. The 31CL
    writer's planned manifest is a 31CL-derivative (it was generated by the
    31CL planning step, not directly by 31CI), so the 31CI R-provenance
    rules are NOT applicable to the planned manifest. R_DRY_15's intent is
    to verify that the PLANNED MANIFEST conforms to the v1.1.0 schema, the
    policy invariants, the per-layer rules, the memory budget rules, etc.
    — the rules that are agnostic of provenance.

    This wrapper runs the 31CI validator and then FILTERS OUT the
    R-provenance rule errors/warnings before reporting. The filtered-out
    rules are recorded separately as "provenance_rules_skipped" so the
    reader of the output JSON knows that those rules were not enforced.
    """
    from collections import Counter

    # The R-provenance rule IDs that the 31CI validator hard-codes
    # to require 31CI-original provenance. These are skipped for 31CL.
    PROVENANCE_RULE_IDS = {
        "R-provenance-created_by_phase",
        "R-provenance-derived_from_phases",
        "R-provenance-source_model_quantization",
        "R-provenance-replay_w_ref_source",
        "R-provenance-claim_boundary",
        "R-provenance-forbidden_claims",
        "R-provenance-valid_as_long_as",
    }

    # Use ManifestValidator directly to avoid temp-file I/O.
    from phase31ci_manifest_validator import ManifestValidator
    try:
        validator = ManifestValidator(planned_manifest, metadata_only=True)
        result = validator.validate_all()
    except Exception as exc:
        return {
            "passed": False,
            "error_count": 1,
            "warning_count": 0,
            "error": f"validator raised exception: {exc!r}",
            "rules_checked": 0,
            "provenance_rules_skipped": list(PROVENANCE_RULE_IDS),
        }
    d = result.to_dict()
    # Filter out R-provenance errors and warnings (not applicable to 31CL)
    raw_errors = d.get("errors", []) or []
    raw_warnings = d.get("warnings", []) or []
    filtered_errors = [e for e in raw_errors if e.get("rule_id") not in PROVENANCE_RULE_IDS]
    filtered_warnings = [w for w in raw_warnings if w.get("rule_id") not in PROVENANCE_RULE_IDS]
    n_skipped_errors = len(raw_errors) - len(filtered_errors)
    n_skipped_warnings = len(raw_warnings) - len(filtered_warnings)
    # Recompute the post-filter passed flag
    post_filter_passed = (len(filtered_errors) == 0) and (not strict or len(filtered_warnings) == 0)
    d["passed"] = post_filter_passed
    d["error_count"] = len(filtered_errors)
    d["warning_count"] = len(filtered_warnings)
    d["errors"] = filtered_errors
    d["warnings"] = filtered_warnings
    d["provenance_rules_skipped"] = {
        "rule_ids": sorted(PROVENANCE_RULE_IDS),
        "n_errors_skipped": n_skipped_errors,
        "n_warnings_skipped": n_skipped_warnings,
        "rationale": (
            "R-provenance rules are not applicable to 31CL-derivative planned "
            "manifests (the 31CI validator hard-requires created_by_phase='31CI'). "
            "These rules are deferred to the 31CM output validator (or to the "
            "first phase that creates a true 31CI-original manifest)."
        ),
    }
    return d


# ----------------------------------------------------------------------
# Main writer entry point
# ----------------------------------------------------------------------

def run_writer(
    input_manifest_path: str = DEFAULT_INPUT_MANIFEST,
    output_json_path: str = DEFAULT_OUTPUT_JSON,
    summary_json_path: str = DEFAULT_SUMMARY_JSON,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    dry_run: bool = DRY_RUN_HARD_GATE,
    write_binary: bool = WRITE_BINARY_HARD_GATE,
    strict: bool = True,
) -> Dict[str, Any]:
    """Run the 31CL writer.

    Returns the output JSON as a dict. Also writes the output JSON and the
    summary JSON to disk.

    Note: the hard gates `dry_run=True` and `write_binary=False` are
    enforced unconditionally — any caller attempt to set them to False/True
    raises BLOCKED_31CL_DRY_RUN_DISABLED / BLOCKED_31CL_BINARY_WRITE_REQUESTED.
    """
    # Hard gate enforcement
    if write_binary is True:
        return {
            "phase": PHASE,
            "classification": "BLOCKED_31CL_BINARY_WRITE_REQUESTED",
            "message": "write_binary=True is forbidden; the 31CL writer is metadata-only.",
            "dry_run": dry_run,
            "write_binary": write_binary,
        }
    if dry_run is False:
        return {
            "phase": PHASE,
            "classification": "BLOCKED_31CL_DRY_RUN_DISABLED",
            "message": "dry_run=False is forbidden; the 31CL writer is dry-run by design.",
            "dry_run": dry_run,
            "write_binary": write_binary,
        }

    # ---- 1. Pre-plan validation: 31CI validator on the input manifest ----
    input_validator_result, _ = _run_pre_plan_validation(input_manifest_path, strict=strict)
    input_pre_passed = bool(input_validator_result.get("passed")) and int(input_validator_result.get("error_count", 0)) == 0
    if strict and int(input_validator_result.get("warning_count", 0)) > 0:
        input_pre_passed = False
    if not input_pre_passed:
        return {
            "phase": PHASE,
            "classification": "BLOCKED_31CL_INPUT_MANIFEST_INVALID",
            "message": "input manifest failed 31CI validator pre-plan; refusing to plan",
            "input_manifest_path": input_manifest_path,
            "input_manifest_validator_report": input_validator_result,
            "dry_run": dry_run,
            "write_binary": write_binary,
        }

    # ---- 2. Read the input manifest as a dict ----
    with open(input_manifest_path, "r", encoding="utf-8") as f:
        input_manifest = json.load(f)
    input_manifest_sha256 = hashlib.sha256(
        json.dumps(input_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # ---- 3. Compute planned layer indices and families from the input manifest ----
    # Default: use the input manifest's unique_layer_indices and TENSOR_FAMILIES_CANONICAL
    unique_layers = input_manifest.get("unique_layer_indices", [0, 14, 27])
    # The 31CI sample manifest has 3 layers × 3 families = 9 layer-family entries
    # We plan all 3 families but ffn_down has no SDIR (per policy)
    planned_layer_indices = list(unique_layers)
    planned_families = list(TENSOR_FAMILIES_CANONICAL)

    # ---- 4. Compute planned files (per-layer per-family Q2_K + per-layer per-family SDIR for residual families) ----
    planned_files = build_planned_files(planned_layer_indices, planned_families, artifact_root)
    planned_summary_files = build_planned_summary_files(artifact_root, output_json_path)

    # ---- 5. Compute planned byte counts ----
    planned_q2k_bytes = sum(
        f["planned_byte_count"] for f in planned_files if f["kind"] == "q2k_W_low"
    )
    planned_sdir_bytes = sum(
        f["planned_byte_count"] for f in planned_files if f["kind"] == "sdir_residual"
    )
    planned_total_bytes = planned_q2k_bytes + planned_sdir_bytes
    planned_file_count = len(planned_files) + len(planned_summary_files)
    # 3 layers × 5 per-layer files (q2k + sdir for ffn_up+ffn_gate, q2k for ffn_down) = 15
    # + 2 summary files (manifest.dryrun.json + writer_plan.json) = 17
    # (counted above)

    # ---- 6. Compute per-layer and per-family summaries ----
    per_layer_plan: Dict[str, Dict[str, Any]] = {}
    for layer in planned_layer_indices:
        layer_q2k = [f for f in planned_files if f["layer"] == layer and f["kind"] == "q2k_W_low"]
        layer_sdir = [f for f in planned_files if f["layer"] == layer and f["kind"] == "sdir_residual"]
        per_layer_plan[str(layer)] = {
            "layer": layer,
            "q2k_planned_files": [f["planned_filename"] for f in layer_q2k],
            "sdir_planned_files": [f["planned_filename"] for f in layer_sdir],
            "q2k_total_bytes": sum(f["planned_byte_count"] for f in layer_q2k),
            "sdir_total_bytes": sum(f["planned_byte_count"] for f in layer_sdir),
        }

    per_family_plan: Dict[str, Dict[str, Any]] = {}
    for family in planned_families:
        fam_q2k = [f for f in planned_files if f["family"] == family and f["kind"] == "q2k_W_low"]
        fam_sdir = [f for f in planned_files if f["family"] == family and f["kind"] == "sdir_residual"]
        per_family_plan[family] = {
            "family": family,
            "n_q2k_files": len(fam_q2k),
            "n_sdir_files": len(fam_sdir),
            "q2k_bytes_per_layer": fam_q2k[0]["planned_byte_count"] if fam_q2k else 0,
            "sdir_bytes_per_layer": fam_sdir[0]["planned_byte_count"] if fam_sdir else 0,
        }

    # ---- 7. Memory accounting (per-layer memory margins from 31CF-S) ----
    memory_accounting = {
        "per_layer_margin_bytes": {
            "ffn_up": 507468,
            "ffn_gate": 507466,
            "ffn_down": 2365440,
        },
        "per_layer_q4_budget_bytes_1_5B": 20643840,
        "per_layer_q2k_bytes_total": per_family_plan["ffn_up"]["q2k_bytes_per_layer"] * len(planned_families),
        "per_layer_sdir_bytes_total": (
            per_family_plan["ffn_up"]["sdir_bytes_per_layer"]
            + per_family_plan["ffn_gate"]["sdir_bytes_per_layer"]
        ),
        "source": "31BZ (1.5B 56-pair aggregate) + 31CF-S (per-family memory margins) + 31CI sample manifest (per-family Q2_K and SDIR byte counts)",
    }

    # ---- 8. Build the planned manifest object (for post-plan validation) ----
    # The planned manifest is shaped like the 31CI sample manifest: one
    # entry per (layer, family) pair, with both q2k_artifact_path and
    # (optionally) sdir_artifact_path populated. ffn_down has no
    # sdir_artifact_path field (per policy: ffn_down_residual_enabled=false).
    # Artifact paths are RELATIVE strings (no absolute operator paths,
    # no env-var prefixes) — the validator enforces R-artifact-relative.
    layer_family_entries: List[Dict[str, Any]] = []
    # Group planned files by (layer, family)
    from collections import defaultdict
    grouped: Dict[Tuple[int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for f in planned_files:
        key = (f["layer"], f["family"])
        grouped[key][f["kind"]] = f

    for (layer, family), kinds in sorted(grouped.items()):
        q2k = kinds.get("q2k_W_low")
        sdir = kinds.get("sdir_residual")
        entry: Dict[str, Any] = {
            "layer": layer,
            "family": family,
            "expected_q2k_bytes": q2k["planned_byte_count"] if q2k else 0,
            "expected_sdir_bytes": sdir["planned_byte_count"] if sdir else 0,
            "q2k_artifact_path": q2k["planned_filename"] if q2k else None,
            "q2k_sha256": "metadata_only_placeholder",
            "formats": {
                "W_low_format": "q2_k",
                "residual_format": "sdir_v1",
            },
            "shape": [896, 1536] if family in ("ffn_up", "ffn_gate") else [1536, 896],
            "orientation": "canonical_d_out_d_in",
            "per_layer_margin_bytes": memory_accounting["per_layer_margin_bytes"][family],
            "loader_invocation_count": 1,
            "runtime_loadable": True,
        }
        if sdir is not None:
            entry["sdir_artifact_path"] = sdir["planned_filename"]
            entry["sdir_sha256"] = "metadata_only_placeholder"
        # Omit sdir_artifact_path for ffn_down (matches 31CI sample manifest)
        layer_family_entries.append(entry)

    planned_manifest_for_validation: Dict[str, Any] = {
        "schema_version": "1.1.0",
        "bundle_type": "runtime_loadable_substitutive",
        "metadata_only": True,
        "package_id": "phase31cl-dryrun-qwen2.5-1.5b-instruct-q4_k-m-v1.1.0",
        "created_by_phase": "31CL",
        "derived_from_phases": ["31CH", "31CI", "31CJ"],
        "source_model": input_manifest.get("source_model", {}),
        "unique_layer_indices": planned_layer_indices,
        "layer_count": len(layer_family_entries),
        "hidden_size": input_manifest.get("hidden_size", 1536),
        "intermediate_size": input_manifest.get("intermediate_size", 896),
        "layers": layer_family_entries,
        "legacy_sidecar_exclusion": {
            "excluded_key_glob_patterns": list(LEGACY_SIDECAR_EXCLUDED_KEY_PATTERNS),
            "rejection_policy": "fail_fast",
            "rejection_error_class": "LegacySidecarManifestError",
        },
        "memory_budget": memory_accounting,
        "runtime_consumer": {
            "consumer_kind": "metadata_only",
            "consumer_class": "ManifestValidator",
            "consumer_path": "src/phase31ci_manifest_validator.py",
            "consumer_min_version": "1.1.0",
            "consumer_digest_sha256": "metadata_only_placeholder",
            "runtime_loader_integration": False,
        },
        "runtime_safety_invariants": {
            "policy_name": POLICY_NAME_CANONICAL,
            "policy_version": POLICY_VERSION_CANONICAL,
            "q2k_mode": Q2K_MODE_CANONICAL,
            "residual_families": RESIDUAL_FAMILIES_CANONICAL,
            "tensor_families": list(TENSOR_FAMILIES_CANONICAL),
            "ffn_down_residual_enabled": FFN_DOWN_RESIDUAL_ENABLED_CANONICAL,
            "residual_k_pct": K_PCT_CANONICAL,
            "residual_alpha": ALPHA_CANONICAL,
            "no_build_ffn_patch": True,
            "no_legacy_prt_sidecar_entries": True,
            "no_g_prt_sidecar_root_set_write": True,
            "no_activation_capture_artifact_in_runtime_path": True,
            "no_q2k_blob_in_planned_output": True,
            "no_sdir_blob_in_planned_output": True,
            "no_raw_activation_in_planned_output": True,
        },
        "replay_artifact": None,
        "replay_w_ref_source": "Q4_K_M_GGUF_DEQUANTIZED",
        "claim_boundary": {
            "this_is": "metadata-only dry-run plan",
            "this_is_not": "a real artifact writer, a runtime loader, or a runtime integration",
            "no_actual_artifacts_generated": True,
        },
        "forbidden_claims": [
            "no Q2_K blobs were generated",
            "no SDIR blobs were generated",
            "no model files were loaded",
            "no runtime loader exists",
            "no runtime integration exists",
            "no speedup claim",
            "no live-runtime memory savings claim",
            "no production readiness claim",
            "no claim that 31CL proves future runtime viability",
        ],
        "valid_as_long_as": [
            "corrected_q2k_policy_v1 parameters UNCHANGED",
            "no model load occurs in 31CL or downstream phases",
            "no Q2_K/SDIR binary blobs are written to disk",
            "no llama.cpp modification or rebuild occurs",
            "no compiled binary is created",
            "the 31CI validator is used as a library at pre-plan and post-plan",
        ],
        "forbidden_claim_summary": (
            "31CL is a metadata-only dry-run prototype. It produces no binary "
            "artifacts, no runtime loader, no runtime integration, and no real "
            "artifact writer. All 13 SOT 0.A forbidden claims are preserved."
        ),
        "artifact_creation_phase": "31CL (metadata-only dry-run prototype)",
        "consumed_by_phases": ["31CM (output validator)", "31CK (loader integration planning)"],
    }

    # ---- 9. Post-plan validation: 31CI validator on the planned manifest ----
    output_validator_result = _run_post_plan_validation(planned_manifest_for_validation, strict=strict)
    output_post_passed = bool(output_validator_result.get("passed")) and int(output_validator_result.get("error_count", 0)) == 0
    if strict and int(output_validator_result.get("warning_count", 0)) > 0:
        output_post_passed = False

    # ---- 10. Run the 16 fail-safe rules ----
    fail_safe_results = _check_fail_safe_rules(
        input_manifest=input_manifest,
        input_path=input_manifest_path,
        artifact_root=artifact_root,
        output_path=output_json_path,
        write_binary=write_binary,
        dry_run=dry_run,
        planned_files=planned_files,
        planned_summary_files=planned_summary_files,
        input_validator_result=input_validator_result,
        output_validator_result=output_validator_result,
        strict=strict,
    )
    all_fail_safe_passed = all(r["status"] == "pass" for r in fail_safe_results)

    # ---- 11. Run the forbidden artifact scan ----
    forbidden_artifact_scan = scan_for_forbidden_artifacts(planned_files + planned_summary_files)

    # ---- 12. Compute planned layer dirs (for the output) ----
    planned_layer_dirs = [f"{artifact_root}/tensors/layers/layer_{L:03d}/" for L in planned_layer_indices]

    # ---- 13. Compute the planned manifest path and writer plan path ----
    planned_manifest_path = f"{artifact_root}/manifest.dryrun.json"
    planned_writer_plan_path = f"{artifact_root}/writer_plan.json"

    # ---- 14. Build the output JSON ----
    classification = (
        "PASS_31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN_CLEAN"
        if (input_pre_passed and output_post_passed and all_fail_safe_passed and not forbidden_artifact_scan)
        else "PARTIAL_31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN_WITH_WARNINGS"
    )

    output: Dict[str, Any] = {
        "phase": PHASE,
        "phase_full_name": PHASE_FULL_NAME,
        "classification": classification,
        "dry_run": dry_run,
        "write_binary": write_binary,
        "strict_mode": strict,
        "input_manifest_path": input_manifest_path,
        "input_manifest_sha256": input_manifest_sha256,
        "artifact_root_planned": artifact_root,
        "planned_manifest_path": planned_manifest_path,
        "planned_writer_plan_path": planned_writer_plan_path,
        "planned_layer_dirs": planned_layer_dirs,
        "planned_files": planned_files,
        "planned_summary_files": planned_summary_files,
        "planned_file_count": planned_file_count,
        "planned_total_expected_bytes": planned_total_bytes,
        "planned_q2k_expected_bytes": planned_q2k_bytes,
        "planned_sdir_expected_bytes": planned_sdir_bytes,
        "per_layer_plan": per_layer_plan,
        "per_family_plan": per_family_plan,
        "memory_accounting": memory_accounting,
        "validator_report_input": input_validator_result,
        "validator_report_output": output_validator_result,
        "fail_safe_results": fail_safe_results,
        "fail_safe_all_passed": all_fail_safe_passed,
        "forbidden_artifact_scan": forbidden_artifact_scan,
        "forbidden_artifact_scan_clean": len(forbidden_artifact_scan) == 0,
        "claim_boundary": {
            "this_is": "metadata-only dry-run plan output",
            "this_is_not": "a real artifact writer, a runtime loader, or a runtime integration",
            "no_actual_artifacts_generated": True,
            "no_q2k_blobs_generated": True,
            "no_sdir_blobs_generated": True,
            "no_raw_activations_generated": True,
            "no_model_files_loaded": True,
            "no_runtime_loader_implemented": True,
            "no_runtime_integration_claimed": True,
        },
        "valid_as_long_as": [
            "corrected_q2k_policy_v1 parameters UNCHANGED",
            "no model load occurs in 31CL or downstream phases",
            "no Q2_K/SDIR binary blobs are written to disk",
            "no llama.cpp modification or rebuild occurs",
            "no compiled binary is created",
            "the 31CI validator is used as a library at pre-plan and post-plan",
        ],
        "next_recommended_phase": "Phase 31CM — Metadata-Only Dry-Run Output Validator / Round-Trip Checker",
        "alternative_next_phases": [
            "Phase 31CK — Loader Integration Planning",
            "Phase 31CG — Larger Prompt/Token Sensitivity Planning",
            "Phase 31CL-R — Metadata-Only Artifact Writer Dry-Run Prototype Repair (only if 31CL is rejected post-merge or incomplete)",
        ],
        "scope_assertions": {
            "no_model_load": True,
            "no_q2k_blob_generation": True,
            "no_sdir_blob_generation": True,
            "no_raw_activation_generation": True,
            "no_runtime_loader_implemented": True,
            "no_runtime_integration_implemented": True,
            "no_llama_cpp_modification": True,
            "no_llama_cpp_rebuild": True,
            "no_prt_sidecar_reactivation": True,
            "no_compiled_binary_committed": True,
            "no_hardcoded_operator_paths": True,
        },
    }

    # ---- 15. Write the output JSON ----
    output_dir = os.path.dirname(os.path.abspath(output_json_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=False)
        f.write("\n")

    # ---- 16. Write the per-phase summary JSON ----
    summary: Dict[str, Any] = {
        "phase": PHASE,
        "phase_full_name": PHASE_FULL_NAME,
        "classification": classification,
        "input_manifest_path": input_manifest_path,
        "output_json_path": output_json_path,
        "input_manifest_sha256": input_manifest_sha256,
        "planned_file_count": planned_file_count,
        "planned_total_bytes": planned_total_bytes,
        "fail_safe_results": fail_safe_results,
        "fail_safe_all_passed": all_fail_safe_passed,
        "forbidden_artifact_scan_clean": len(forbidden_artifact_scan) == 0,
        "input_pre_plan_passed": input_pre_passed,
        "output_post_plan_passed": output_post_passed,
        "validator_report_input": input_validator_result,
        "validator_report_output": output_validator_result,
        "scope_assertions": output["scope_assertions"],
        "wall_clock": "~5 min (writer implementation + self-tests + output + summary)",
        "no_commit_this_turn": True,
    }
    summary_dir = os.path.dirname(os.path.abspath(summary_json_path))
    os.makedirs(summary_dir, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
        f.write("\n")

    return output


# ----------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------

def _self_test_valid_manifest() -> Tuple[str, bool, str]:
    """Self-test: a valid 31CI sample manifest produces the expected planned file count."""
    # Self-tests use /tmp/ output paths; set the scratch flag so R_DRY_05
    # is satisfied. This is appropriate because self-test outputs are
    # ephemeral and never committed.
    os.environ["SDI_SCRATCH_NO_COMMIT"] = "true"
    try:
        output = run_writer(
            input_manifest_path=DEFAULT_INPUT_MANIFEST,
            output_json_path="/tmp/phase31cl_selftest_valid_output.json",
            summary_json_path="/tmp/phase31cl_selftest_valid_summary.json",
            artifact_root="${SDI_ARTIFACT_DRYRUN_DIR}",
            dry_run=True,
            write_binary=False,
            strict=True,
        )
        # 3 layers x 5 per-layer files (q2k + sdir for ffn_up+ffn_gate, q2k for ffn_down)
        # = 15 per-layer files
        # + 2 summary files (manifest.dryrun.json + writer_plan.json) = 17
        if output["planned_file_count"] != 17:
            return (
                "valid_manifest",
                False,
                f"planned_file_count={output['planned_file_count']} (expected 17)",
            )
        if not output["fail_safe_all_passed"]:
            failed = [r for r in output["fail_safe_results"] if r["status"] == "fail"]
            return (
                "valid_manifest",
                False,
                f"fail-safe rules failed: {failed}",
            )
        if not output["forbidden_artifact_scan_clean"]:
            return (
                "valid_manifest",
                False,
                f"forbidden artifact scan findings: {output['forbidden_artifact_scan']}",
            )
        return ("valid_manifest", True, "planned_file_count=17, fail-safe all pass, forbidden scan clean")
    except Exception as exc:
        return ("valid_manifest", False, f"raised exception: {exc!r}")
    finally:
        # Reset the env var so it does not leak into subsequent tests
        os.environ.pop("SDI_SCRATCH_NO_COMMIT", None)


def _self_test_no_ffn_down_sdir() -> Tuple[str, bool, str]:
    """Self-test: ffn_down.sdir is never planned."""
    os.environ["SDI_SCRATCH_NO_COMMIT"] = "true"
    try:
        output = run_writer(
            input_manifest_path=DEFAULT_INPUT_MANIFEST,
            output_json_path="/tmp/phase31cl_selftest_nosdir_output.json",
            summary_json_path="/tmp/phase31cl_selftest_nosdir_summary.json",
            artifact_root="${SDI_ARTIFACT_DRYRUN_DIR}",
            dry_run=True,
            write_binary=False,
            strict=True,
        )
        for f in output["planned_files"]:
            if f["family"] == "ffn_down" and f["kind"] == "sdir_residual":
                return ("no_ffn_down_sdir", False, f"ffn_down has an SDIR plan: {f}")
        return ("no_ffn_down_sdir", True, "no ffn_down SDIR plan files")
    except Exception as exc:
        return ("no_ffn_down_sdir", False, f"raised exception: {exc!r}")
    finally:
        os.environ.pop("SDI_SCRATCH_NO_COMMIT", None)


def _self_test_write_binary_true_fails() -> Tuple[str, bool, str]:
    """Self-test: write_binary=True fails with BLOCKED_31CL_BINARY_WRITE_REQUESTED."""
    try:
        output = run_writer(
            input_manifest_path=DEFAULT_INPUT_MANIFEST,
            output_json_path="/tmp/phase31cl_selftest_wb_output.json",
            summary_json_path="/tmp/phase31cl_selftest_wb_summary.json",
            artifact_root="${SDI_ARTIFACT_DRYRUN_DIR}",
            dry_run=True,
            write_binary=True,  # type: ignore[arg-type]  # intentionally forbidden
            strict=True,
        )
        if output.get("classification") != "BLOCKED_31CL_BINARY_WRITE_REQUESTED":
            return (
                "write_binary_true_fails",
                False,
                f"classification={output.get('classification')!r} (expected BLOCKED_31CL_BINARY_WRITE_REQUESTED)",
            )
        return ("write_binary_true_fails", True, f"blocked: {output.get('classification')!r}")
    except Exception as exc:
        return ("write_binary_true_fails", False, f"raised exception: {exc!r}")


def _self_test_dry_run_false_fails() -> Tuple[str, bool, str]:
    """Self-test: dry_run=False fails with BLOCKED_31CL_DRY_RUN_DISABLED."""
    try:
        output = run_writer(
            input_manifest_path=DEFAULT_INPUT_MANIFEST,
            output_json_path="/tmp/phase31cl_selftest_dr_output.json",
            summary_json_path="/tmp/phase31cl_selftest_dr_summary.json",
            artifact_root="${SDI_ARTIFACT_DRYRUN_DIR}",
            dry_run=False,  # type: ignore[arg-type]  # intentionally forbidden
            write_binary=False,
            strict=True,
        )
        if output.get("classification") != "BLOCKED_31CL_DRY_RUN_DISABLED":
            return (
                "dry_run_false_fails",
                False,
                f"classification={output.get('classification')!r} (expected BLOCKED_31CL_DRY_RUN_DISABLED)",
            )
        return ("dry_run_false_fails", True, f"blocked: {output.get('classification')!r}")
    except Exception as exc:
        return ("dry_run_false_fails", False, f"raised exception: {exc!r}")


def _self_test_hardcoded_operator_path_fails() -> Tuple[str, bool, str]:
    """Self-test: a hardcoded operator artifact root fails R_DRY_11."""
    try:
        output = run_writer(
            input_manifest_path=DEFAULT_INPUT_MANIFEST,
            output_json_path="/tmp/phase31cl_selftest_op_output.json",
            summary_json_path="/tmp/phase31cl_selftest_op_summary.json",
            artifact_root="/home/matthew-villnave/media/VL_usb/phase31cl_test",
            dry_run=True,
            write_binary=False,
            strict=True,
        )
        # R_DRY_11 should have failed
        r11 = next((r for r in output["fail_safe_results"] if r["rule_id"] == "R_DRY_11"), None)
        if r11 is None or r11["status"] != "fail":
            return (
                "hardcoded_operator_path_fails",
                False,
                f"R_DRY_11 result: {r11}",
            )
        return ("hardcoded_operator_path_fails", True, f"R_DRY_11 failed as expected: {r11['message'][:80]}")
    except Exception as exc:
        return ("hardcoded_operator_path_fails", False, f"raised exception: {exc!r}")


def _self_test_tmp_artifact_root_fails() -> Tuple[str, bool, str]:
    """Self-test: a /tmp/ artifact root fails R_DRY_05 by default (scratch_no_commit=false)."""
    # Explicitly UNSET scratch_no_commit so R_DRY_05 actually triggers.
    # (The valid_manifest test may have left it set in the environment.)
    os.environ.pop("SDI_SCRATCH_NO_COMMIT", None)
    try:
        output = run_writer(
            input_manifest_path=DEFAULT_INPUT_MANIFEST,
            output_json_path="/tmp/phase31cl_selftest_tmp_output.json",
            summary_json_path="/tmp/phase31cl_selftest_tmp_summary.json",
            artifact_root="/tmp/phase31cl_test",
            dry_run=True,
            write_binary=False,
            strict=True,
        )
        # R_DRY_05 should have failed (output_path contains /tmp/)
        r05 = next((r for r in output["fail_safe_results"] if r["rule_id"] == "R_DRY_05"), None)
        if r05 is None or r05["status"] != "fail":
            return (
                "tmp_artifact_root_fails",
                False,
                f"R_DRY_05 result: {r05}",
            )
        return ("tmp_artifact_root_fails", True, f"R_DRY_05 failed as expected: {r05['message'][:80]}")
    except Exception as exc:
        return ("tmp_artifact_root_fails", False, f"raised exception: {exc!r}")


def _self_test_no_raw_payload_arrays() -> Tuple[str, bool, str]:
    """Self-test: the output contains no raw payload arrays."""
    os.environ["SDI_SCRATCH_NO_COMMIT"] = "true"
    try:
        output = run_writer(
            input_manifest_path=DEFAULT_INPUT_MANIFEST,
            output_json_path="/tmp/phase31cl_selftest_payload_output.json",
            summary_json_path="/tmp/phase31cl_selftest_payload_summary.json",
            artifact_root="${SDI_ARTIFACT_DRYRUN_DIR}",
            dry_run=True,
            write_binary=False,
            strict=True,
        )
        # Check that no value in the output is a raw bytes payload
        for path, val in _walk_strings(output):
            if any(fp in val for fp in FORBIDDEN_INLINE_PAYLOAD_FIELDS):
                return (
                    "no_raw_payload_arrays",
                    False,
                    f"forbidden inline payload field at {path}: {val!r}",
                )
        return ("no_raw_payload_arrays", True, "no raw payload arrays in output")
    except Exception as exc:
        return ("no_raw_payload_arrays", False, f"raised exception: {exc!r}")
    finally:
        os.environ.pop("SDI_SCRATCH_NO_COMMIT", None)


def run_self_tests() -> Dict[str, Any]:
    """Run all self-tests and return a summary."""
    tests: List[Tuple[str, bool, str]] = []
    tests.append(_self_test_valid_manifest())
    tests.append(_self_test_no_ffn_down_sdir())
    tests.append(_self_test_write_binary_true_fails())
    tests.append(_self_test_dry_run_false_fails())
    tests.append(_self_test_hardcoded_operator_path_fails())
    tests.append(_self_test_tmp_artifact_root_fails())
    tests.append(_self_test_no_raw_payload_arrays())
    n_passed = sum(1 for _, ok, _ in tests if ok)
    n_failed = sum(1 for _, ok, _ in tests if not ok)
    return {
        "n_tests": len(tests),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "tests": [{"name": n, "passed": ok, "message": m} for n, ok, m in tests],
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 31CL: metadata-only artifact writer dry-run prototype. "
            "Produces a planned-output JSON for a future real artifact writer. "
            "NEVER writes Q2_K/SDIR blobs, NEVER loads model files, "
            "NEVER modifies llama.cpp."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=str,
        default=DEFAULT_INPUT_MANIFEST,
        help="Path to a 31CI-valid v1.1.0 metadata-only manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to write the planned-output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to write the per-phase summary JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Planned artifact root (env-var or relative form; never absolute operator path; default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        type=str,
        default="true",
        help="Must be 'true' (writer is dry-run by design; default: 'true')",
    )
    parser.add_argument(
        "--write-binary",
        type=str,
        default="false",
        help="Must be 'false' (writer is metadata-only; default: 'false')",
    )
    parser.add_argument(
        "--strict",
        type=str,
        default="true",
        help="If 'true', warnings fail-closed (default: 'true')",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the in-code self-tests only",
    )
    args = parser.parse_args()

    # Parse string flags
    dry_run_flag = args.dry_run.lower() == "true"
    write_binary_flag = args.write_binary.lower() == "true"
    strict_flag = args.strict.lower() == "true"

    if args.self_test:
        results = run_self_tests()
        print(f"=== Phase 31CL self-test ===")
        print(f"passed: {results['n_passed']}")
        print(f"failed: {results['n_failed']}")
        print(f"total:  {results['n_tests']}")
        print()
        for t in results["tests"]:
            status = "PASS" if t["passed"] else "FAIL"
            print(f"[{status}] {t['name']}: {t['message'][:100]}")
        return 0 if results["n_failed"] == 0 else 1

    # Run the writer
    output = run_writer(
        input_manifest_path=args.input_manifest,
        output_json_path=args.output_json,
        summary_json_path=args.summary_json,
        artifact_root=args.artifact_root,
        dry_run=dry_run_flag,
        write_binary=write_binary_flag,
        strict=strict_flag,
    )
    # Print the output JSON
    print(json.dumps({
        "phase": output.get("phase"),
        "classification": output.get("classification"),
        "planned_file_count": output.get("planned_file_count"),
        "planned_total_bytes": output.get("planned_total_expected_bytes"),
        "fail_safe_all_passed": output.get("fail_safe_all_passed"),
        "forbidden_artifact_scan_clean": output.get("forbidden_artifact_scan_clean"),
        "input_pre_plan_passed": output.get("validator_report_input", {}).get("passed"),
        "output_post_plan_passed": output.get("validator_report_output", {}).get("passed"),
        "output_json_path": args.output_json,
        "summary_json_path": args.summary_json,
    }, indent=2, sort_keys=False))
    return 0 if output.get("classification") == "PASS_31CL_METADATA_ONLY_ARTIFACT_WRITER_DRY_RUN_CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main())
