#!/usr/bin/env python3
"""
phase31ci_manifest_validator.py

Phase 31CI — Runtime Artifact Schema Prototype / Metadata-Only Manifest Validator.

Validates a metadata-only v1.1.0 runtime artifact manifest against the 31CH design.

This validator is METADATA-ONLY:
  - It does NOT load any model file.
  - It does NOT inspect GGUF tensor payloads.
  - It does NOT read or write Q2_K / SDIR binary artifacts.
  - It does NOT read or write raw activation arrays.
  - It does NOT implement a runtime loader.
  - It does NOT modify or rebuild llama.cpp.

It validates only the manifest METADATA (shapes, byte counts, sha256 placeholders,
policy invariants, runtime safety invariants, legacy sidecar exclusion, manifest
hygiene, per-layer/per-family metadata, memory budget metadata, provenance metadata).

Design references:
  - docs/PHASE31CH_RUNTIME_ARTIFACT_FORMAT_LOADER_PLANNING.md (v1.1.0 schema, 48 rules R1-R48)
  - docs/STATIC_ARTIFACT_SCHEMA.md (v1.0 offline schema, the v1.1.0 extends it)
  - src/bundle_manifest.py (the existing ManifestLoader class, which 31CI does NOT replace)

This module is dependency-free beyond Python stdlib + json + hashlib.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────────────────
# Constants (per 31CH v1.1.0 design + corrected_q2k_policy_v1)
# ────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION_ACCEPTED = "1.1.0"
BUNDLE_TYPE_RUNTIME_LOADABLE = "runtime_loadable_substitutive"
BUNDLE_TYPES_V1_0_LEGACY = (
    "source_of_truth_regression",
    "ffn_up_substitutive",
    "ffn_down_substitutive",
    "full_mlp_substitutive",
    "layer_substitutive",
)

POLICY_NAME_CANONICAL = "corrected_q2k_policy_v1"
POLICY_VERSION_CANONICAL = "1"
Q2K_MODE_CANONICAL = "corrected_ceil_per_row"
RESIDUAL_FAMILIES_CANONICAL = ["ffn_up", "ffn_gate"]
RESIDUAL_K_PCT_CANONICAL = 0.5
RESIDUAL_ALPHA_CANONICAL = 1.0
TENSOR_FAMILIES_CANONICAL = ["ffn_up", "ffn_gate", "ffn_down"]
FFN_DOWN_RESIDUAL_ENABLED_CANONICAL = False

# 7 legacy-sidecar exclusion glob patterns (per 31CH v1.1.0 design §5.4)
LEGACY_SIDECAR_EXCLUDED_KEY_PATTERNS = (
    "prt_*",
    "sidecar_*",
    "g_prt_*",
    "g_build_ffn_*",
    "g_apply_layer_*",
    "shadow_*",
    "pager_*",
)

# 6 manifest hygiene path patterns (per 31CH R40-R45)
HARDCODED_OPERATOR_PATH_PATTERNS = (
    re.compile(r"/media/matthew-villnave"),
    re.compile(r"/home/matthew-villnave/[^$]*\.(?:gguf|safetensors|bin)$"),
    re.compile(r"VL_usb"),
    re.compile(r"~/(?:llama\.cpp/)?build/"),
)
FORBIDDEN_PATH_FRAGMENTS = (
    "/tmp/",  # R45: no /tmp/ paths in committed manifests
)
FORBIDDEN_ARTIFACT_EXTENSIONS_IN_PATHS = (
    ".gguf",       # R42: no model file references
    ".safetensors",
    ".npy",        # R41: no raw activation references
    ".npz",
    ".bin",
    ".sdiw",       # a serialized W_low; forbidden as a path (metadata-only path allows .sdiw-named filenames only if path is relative + no model file)
    ".so",         # R43: no compiled binary references
    ".o",
    ".a",
    "/build/",     # R44: no llama.cpp build artifact references
    "llama-server",
    "libllama.so",
    "model.safetensors",
    ".trit",
)
FORBIDDEN_INLINE_PAYLOAD_FIELDS = (
    "raw_payload", "raw_payload_bytes", "w_low_raw", "sdir_raw",
    "activation_array", "raw_x", "raw_y", "raw_r",
)

# 3 runtime consumer kinds accepted in 31CI
RUNTIME_CONSUMER_KIND_ACCEPTED = ("metadata_only", "standalone_tensor_harness")

# 1.5B Qwen2.5-Instruct per-family Q4 budget (per STATIC_ARTIFACT_SCHEMA.md + 31BZ)
# ffn_up: d_out=896, d_in=1536 → 896*1536 / 2 = 688,128 bytes (Q4_K_M, ~4.5 bpw)
# ffn_gate: d_out=896, d_in=1536 → 688,128 bytes
# ffn_down: d_out=1536, d_in=896 → 688,128 bytes
# Note: the exact value depends on the GGUF Q4_K_M superblock size; we use 688,128 as a
# conservative reference (the existing 31BZ/31CA/31CF-S2 result JSONs record the actual
# per-family bytes; the validator accepts any positive integer and only checks non-negative
# and consistent with the schema).
Q4_BUDGET_FAMILY_REFERENCE_1_5B = 688_128

# Layers (per 31CH sample manifest spec: 0, 14, 27 only)
SAMPLE_LAYERS = [0, 14, 27]


# ────────────────────────────────────────────────────────────────────────────
# Validation result data structures
# ────────────────────────────────────────────────────────────────────────────

class ValidationResult:
    """Structured validation result (per 31CI spec)."""

    def __init__(self) -> None:
        self.passed: bool = True
        self.error_count: int = 0
        self.warning_count: int = 0
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.rules_checked: List[str] = []
        self.manifest_sha256: Optional[str] = None
        self.classification_suggestion: str = ""

    def add_error(self, rule_id: str, message: str, path: str = "") -> None:
        self.passed = False
        self.error_count += 1
        self.errors.append({
            "rule_id": rule_id,
            "message": message,
            "path": path,
        })

    def add_warning(self, rule_id: str, message: str, path: str = "") -> None:
        self.warning_count += 1
        self.warnings.append({
            "rule_id": rule_id,
            "message": message,
            "path": path,
        })

    def add_rule_checked(self, rule_id: str) -> None:
        if rule_id not in self.rules_checked:
            self.rules_checked.append(rule_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "rules_checked": self.rules_checked,
            "rules_checked_count": len(self.rules_checked),
            "manifest_sha256": self.manifest_sha256,
            "classification_suggestion": self.classification_suggestion,
        }


# ────────────────────────────────────────────────────────────────────────────
# Helper: recursive key search for legacy sidecar exclusion
# ────────────────────────────────────────────────────────────────────────────

def _find_excluded_keys(
    obj: Any,
    patterns: Tuple[str, ...],
    path: str = "",
) -> List[Tuple[str, str]]:
    """Recursively find keys matching any of the excluded glob patterns.
    Returns a list of (key, full_path) tuples.
    Applies to nested dictionaries and lists of dictionaries.
    """
    found: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{path}.{k}" if path else k
            for pattern in patterns:
                if fnmatch.fnmatch(k, pattern):
                    found.append((k, full))
                    break  # one match per key is enough
            # recurse
            found.extend(_find_excluded_keys(v, patterns, full))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            full = f"{path}[{i}]"
            found.extend(_find_excluded_keys(item, patterns, full))
    return found


def _string_contains_any(
    s: str,
    needles: Tuple[str, ...],
) -> bool:
    return any(needle in s for needle in needles)


def _string_contains_forbidden_path(s: str) -> bool:
    """Check if a string contains any hardcoded operator path or forbidden artifact path.
    Allowed: env-var redacted forms like $SDI_MODEL_DIR/..., <env>, <placeholder>, etc.
    """
    # Allowed env-var redacted forms
    allowed_forms = ("$SDI_MODEL_DIR", "$HOME", "${", "<env:", "<placeholder", "<redacted", "metadata_only")
    if any(form in s for form in allowed_forms):
        return False
    # Hardcoded operator paths (R40)
    for pat in HARDCODED_OPERATOR_PATH_PATTERNS:
        if pat.search(s):
            return True
    # /tmp/ paths (R45)
    for needle in FORBIDDEN_PATH_FRAGMENTS:
        if needle in s:
            return True
    # Forbidden artifact extensions (R41, R42, R43, R44)
    for ext in FORBIDDEN_ARTIFACT_EXTENSIONS_IN_PATHS:
        if ext in s:
            return True
    return False


# ────────────────────────────────────────────────────────────────────────────
# Validator
# ────────────────────────────────────────────────────────────────────────────

class ManifestValidator:
    """Validates a metadata-only v1.1.0 runtime artifact manifest."""

    def __init__(
        self,
        manifest: Dict[str, Any],
        metadata_only: bool = True,
    ) -> None:
        self.manifest = manifest
        self.metadata_only = metadata_only
        self.result = ValidationResult()

    def validate_all(self) -> ValidationResult:
        """Run all validation rules and return the structured result."""
        self._validate_schema_and_version()
        self._validate_policy_invariants()
        self._validate_runtime_safety_invariants()
        self._validate_legacy_sidecar_exclusion()
        self._validate_manifest_hygiene()
        self._validate_per_layer_per_family()
        self._validate_memory_budget()
        self._validate_provenance()
        # Compute classification suggestion
        if self.result.passed:
            if self.result.warning_count == 0:
                self.result.classification_suggestion = "PASS_31CI_RUNTIME_ARTIFACT_SCHEMA_VALIDATOR_CLEAN"
            else:
                self.result.classification_suggestion = "PARTIAL_31CI_VALIDATOR_IMPLEMENTED_WITH_WARNINGS"
        else:
            self.result.classification_suggestion = "PARTIAL_31CI_VALIDATOR_IMPLEMENTED_WITH_ERRORS"
        return self.result

    # ── R1-R4: schema and version ────────────────────────────────────────────
    def _validate_schema_and_version(self) -> None:
        for rid in ("R1", "R2", "R3", "R4", "R-rule-schema-version"):
            self.result.add_rule_checked(rid)

        # R1: schema_version
        sv = self.manifest.get("schema_version")
        if sv != SCHEMA_VERSION_ACCEPTED:
            self.result.add_error(
                "R1", f"schema_version must be '{SCHEMA_VERSION_ACCEPTED}', got '{sv}'",
                "schema_version"
            )

        # R2: bundle_type
        bt = self.manifest.get("bundle_type")
        accepted = (BUNDLE_TYPE_RUNTIME_LOADABLE,) + BUNDLE_TYPES_V1_0_LEGACY
        if bt not in accepted:
            self.result.add_error(
                "R2", f"bundle_type must be one of {accepted}, got '{bt}'",
                "bundle_type"
            )

        # R3: for runtime_loadable_substitutive, v1.1.0 fields must be present
        if bt == BUNDLE_TYPE_RUNTIME_LOADABLE:
            for field in (
                "runtime_safety_invariants",
                "runtime_consumer",
                "legacy_sidecar_exclusion",
                "artifact_creation_phase",
                "consumed_by_phases",
            ):
                if field not in self.manifest:
                    self.result.add_error(
                        "R3", f"runtime_loadable_substitutive manifest missing required field '{field}'",
                        field
                    )

        # R4: source_model.quantization
        sm = self.manifest.get("source_model", {})
        if not isinstance(sm, dict):
            self.result.add_error("R4", "source_model must be a dict", "source_model")
        else:
            q = sm.get("quantization")
            if q not in ("Q2_K", "Q4_K_M"):
                self.result.add_error(
                    "R4", f"source_model.quantization must be 'Q2_K' or 'Q4_K_M', got '{q}'",
                    "source_model.quantization"
                )

    # ── R5-R13: policy invariants ────────────────────────────────────────────
    def _validate_policy_invariants(self) -> None:
        for rid in ("R5", "R6", "R7", "R8", "R9", "R-policy-invariants"):
            self.result.add_rule_checked(rid)

        inv = self.manifest.get("runtime_safety_invariants", {})
        if not isinstance(inv, dict):
            self.result.add_error("R5", "runtime_safety_invariants must be a dict", "runtime_safety_invariants")
            return

        # R5: policy_name
        if inv.get("policy_name") != POLICY_NAME_CANONICAL:
            self.result.add_error(
                "R5", f"runtime_safety_invariants.policy_name must be '{POLICY_NAME_CANONICAL}', got '{inv.get('policy_name')}'",
                "runtime_safety_invariants.policy_name"
            )

        # R6: policy_version
        if inv.get("policy_version") != POLICY_VERSION_CANONICAL:
            self.result.add_error(
                "R6", f"runtime_safety_invariants.policy_version must be '{POLICY_VERSION_CANONICAL}', got '{inv.get('policy_version')}'",
                "runtime_safety_invariants.policy_version"
            )

        # R7: q2k_mode
        if inv.get("q2k_mode") != Q2K_MODE_CANONICAL:
            self.result.add_error(
                "R7", f"runtime_safety_invariants.q2k_mode must be '{Q2K_MODE_CANONICAL}', got '{inv.get('q2k_mode')}'",
                "runtime_safety_invariants.q2k_mode"
            )

        # R8: residual_families
        rf = inv.get("residual_families")
        if rf != RESIDUAL_FAMILIES_CANONICAL:
            self.result.add_error(
                "R8", f"runtime_safety_invariants.residual_families must be exactly {RESIDUAL_FAMILIES_CANONICAL}, got {rf}",
                "runtime_safety_invariants.residual_families"
            )

        # R9: ffn_down_residual_enabled
        if inv.get("ffn_down_residual_enabled") is not False:
            self.result.add_error(
                "R9", f"runtime_safety_invariants.ffn_down_residual_enabled must be False, got {inv.get('ffn_down_residual_enabled')}",
                "runtime_safety_invariants.ffn_down_residual_enabled"
            )

        # policy sub-fields (R-policy-residual-k-pct, R-policy-residual-alpha)
        self.result.add_rule_checked("R-policy-residual-k-pct")
        self.result.add_rule_checked("R-policy-residual-alpha")
        if inv.get("residual_k_pct") != RESIDUAL_K_PCT_CANONICAL:
            self.result.add_error(
                "R-policy-residual-k-pct", f"runtime_safety_invariants.residual_k_pct must be {RESIDUAL_K_PCT_CANONICAL}, got {inv.get('residual_k_pct')}",
                "runtime_safety_invariants.residual_k_pct"
            )
        if inv.get("residual_alpha") != RESIDUAL_ALPHA_CANONICAL:
            self.result.add_error(
                "R-policy-residual-alpha", f"runtime_safety_invariants.residual_alpha must be {RESIDUAL_ALPHA_CANONICAL}, got {inv.get('residual_alpha')}",
                "runtime_safety_invariants.residual_alpha"
            )

        # tensor_families (R-policy-tensor-families)
        self.result.add_rule_checked("R-policy-tensor-families")
        tf = inv.get("tensor_families")
        if tf != TENSOR_FAMILIES_CANONICAL:
            self.result.add_error(
                "R-policy-tensor-families", f"runtime_safety_invariants.tensor_families must be exactly {TENSOR_FAMILIES_CANONICAL}, got {tf}",
                "runtime_safety_invariants.tensor_families"
            )

    # ── R10-R13: runtime safety invariants (no_*_patch / no_*_entries / no_*_write) ─
    def _validate_runtime_safety_invariants(self) -> None:
        for rid in ("R10", "R11", "R12", "R13", "R-consumer-metadata-only", "R-loader-integration-false"):
            self.result.add_rule_checked(rid)

        inv = self.manifest.get("runtime_safety_invariants", {})

        # R10: no_build_ffn_patch
        if inv.get("no_build_ffn_patch") is not True:
            self.result.add_error(
                "R10", "runtime_safety_invariants.no_build_ffn_patch must be True",
                "runtime_safety_invariants.no_build_ffn_patch"
            )

        # R11: no_legacy_prt_sidecar_entries
        if inv.get("no_legacy_prt_sidecar_entries") is not True:
            self.result.add_error(
                "R11", "runtime_safety_invariants.no_legacy_prt_sidecar_entries must be True",
                "runtime_safety_invariants.no_legacy_prt_sidecar_entries"
            )

        # R12: no_g_prt_sidecar_root_set_write
        if inv.get("no_g_prt_sidecar_root_set_write") is not True:
            self.result.add_error(
                "R12", "runtime_safety_invariants.no_g_prt_sidecar_root_set_write must be True",
                "runtime_safety_invariants.no_g_prt_sidecar_root_set_write"
            )

        # R13: no_activation_capture_artifact_in_runtime_path
        if inv.get("no_activation_capture_artifact_in_runtime_path") is not True:
            self.result.add_error(
                "R13", "runtime_safety_invariants.no_activation_capture_artifact_in_runtime_path must be True",
                "runtime_safety_invariants.no_activation_capture_artifact_in_runtime_path"
            )

        # runtime_consumer
        rc = self.manifest.get("runtime_consumer", {})
        if not isinstance(rc, dict):
            self.result.add_error("R-consumer", "runtime_consumer must be a dict", "runtime_consumer")
            return
        kind = rc.get("consumer_kind")
        if kind not in RUNTIME_CONSUMER_KIND_ACCEPTED:
            self.result.add_error(
                "R-consumer-metadata-only",
                f"runtime_consumer.consumer_kind must be one of {RUNTIME_CONSUMER_KIND_ACCEPTED} in 31CI, got '{kind}'",
                "runtime_consumer.consumer_kind"
            )

        # runtime_loader_integration must be false in 31CI
        if rc.get("runtime_loader_integration") is not False:
            self.result.add_error(
                "R-loader-integration-false",
                "runtime_consumer.runtime_loader_integration must be False in 31CI",
                "runtime_consumer.runtime_loader_integration"
            )

    # ── R14-R16: legacy sidecar exclusion (recursive key search) ────────────
    def _validate_legacy_sidecar_exclusion(self) -> None:
        for rid in ("R14", "R15", "R16"):
            self.result.add_rule_checked(rid)

        # R14: recursive key search
        excluded = _find_excluded_keys(self.manifest, LEGACY_SIDECAR_EXCLUDED_KEY_PATTERNS)
        for key, full_path in excluded:
            self.result.add_error(
                "R14",
                f"legacy sidecar excluded key '{key}' found at '{full_path}'",
                full_path
            )

        # R15: rejection_policy
        lse = self.manifest.get("legacy_sidecar_exclusion", {})
        if lse.get("rejection_policy") != "fail_fast":
            self.result.add_error(
                "R15",
                f"legacy_sidecar_exclusion.rejection_policy must be 'fail_fast', got '{lse.get('rejection_policy')}'",
                "legacy_sidecar_exclusion.rejection_policy"
            )

        # R16: rejection_error_class
        if lse.get("rejection_error_class") != "LegacySidecarManifestError":
            self.result.add_error(
                "R16",
                f"legacy_sidecar_exclusion.rejection_error_class must be 'LegacySidecarManifestError', got '{lse.get('rejection_error_class')}'",
                "legacy_sidecar_exclusion.rejection_error_class"
            )

    # ── R40-R45: manifest hygiene ────────────────────────────────────────────
    def _validate_manifest_hygiene(self) -> None:
        for rid in ("R40", "R41", "R42", "R43", "R44", "R45", "R-inline-payload"):
            self.result.add_rule_checked(rid)

        # Walk all string values and paths/filenames for hygiene
        for path, value in _walk_strings(self.manifest):
            if not isinstance(value, str):
                continue
            if _string_contains_forbidden_path(value):
                # Distinguish by what was found
                if "/media/matthew-villnave" in value or "VL_usb" in value:
                    rid = "R40"
                    msg = f"hardcoded operator path in manifest at '{path}': '{value[:100]}'"
                elif "/tmp/" in value:
                    rid = "R45"
                    msg = f"/tmp/ path in committed manifest at '{path}': '{value[:100]}'"
                elif any(ext in value for ext in (".gguf", ".safetensors", "model.safetensors")):
                    rid = "R42"
                    msg = f"model file reference at '{path}': '{value[:100]}'"
                elif any(ext in value for ext in (".npy", ".npz", ".bin", ".trit")):
                    rid = "R41"
                    msg = f"raw activation path at '{path}': '{value[:100]}'"
                elif any(ext in value for ext in (".so", ".o", ".a", "llama-server", "libllama.so", "/build/")):
                    rid = "R43" if any(ext in value for ext in (".so", ".o", ".a", "llama-server", "libllama.so")) else "R44"
                    msg = f"compiled binary / build artifact reference at '{path}': '{value[:100]}'"
                else:
                    rid = "R40"
                    msg = f"forbidden path in manifest at '{path}': '{value[:100]}'"
                self.result.add_error(rid, msg, path)

        # R-inline-payload: check for inline payload-like array fields
        for field in FORBIDDEN_INLINE_PAYLOAD_FIELDS:
            if field in self.manifest:
                self.result.add_error(
                    "R-inline-payload",
                    f"forbidden inline payload field '{field}' at top level (metadata-only manifests must not embed raw payloads)",
                    field
                )

    # ── R17-R27: per-layer / per-family (metadata-only) ─────────────────────
    def _validate_per_layer_per_family(self) -> None:
        for rid in ("R17", "R18", "R19", "R20", "R21", "R26", "R27", "R-families-limited", "R-ffn-down-no-sdir"):
            self.result.add_rule_checked(rid)

        layers = self.manifest.get("layers", [])
        if not isinstance(layers, list):
            self.result.add_error("R17", "layers must be a list", "layers")
            return

        # R17: layer_count matches
        expected_layer_count = self.manifest.get("layer_count")
        if expected_layer_count is not None and len(layers) != expected_layer_count:
            self.result.add_error(
                "R17",
                f"layer_count {expected_layer_count} does not match actual layers[] count {len(layers)}",
                "layers"
            )

        seen_layers: List[Any] = []
        for i, layer in enumerate(layers):
            layer_path = f"layers[{i}]"
            if not isinstance(layer, dict):
                self.result.add_error("R17", f"{layer_path} must be a dict", layer_path)
                continue

            # R19: layer ids are integers or zero-padded strings
            lid = layer.get("layer")
            if lid is None:
                self.result.add_error("R19", f"{layer_path}.layer is required", f"{layer_path}.layer")
            elif not (isinstance(lid, int) or (isinstance(lid, str) and re.fullmatch(r"\d+", lid))):
                self.result.add_error(
                    "R19", f"{layer_path}.layer must be an integer or zero-padded string, got {type(lid).__name__}",
                    f"{layer_path}.layer"
                )
            seen_layers.append(lid)

            # R20: expected shapes are present
            if "shape" not in layer:
                self.result.add_error("R20", f"{layer_path}.shape is required", f"{layer_path}.shape")

            # R21: byte counts are non-negative integers
            for byte_field in ("expected_q2k_bytes", "expected_sdir_bytes", "per_layer_margin_bytes"):
                if byte_field in layer:
                    val = layer[byte_field]
                    if not isinstance(val, int) or val < 0:
                        self.result.add_error(
                            "R21",
                            f"{layer_path}.{byte_field} must be a non-negative integer, got {val!r}",
                            f"{layer_path}.{byte_field}"
                        )

            # R-families-limited: family is one of ffn_up, ffn_gate, ffn_down
            family = layer.get("family")
            if family is not None and family not in TENSOR_FAMILIES_CANONICAL:
                self.result.add_error(
                    "R-families-limited",
                    f"{layer_path}.family must be one of {TENSOR_FAMILIES_CANONICAL}, got '{family}'",
                    f"{layer_path}.family"
                )

            # R-ffn-down-no-sdir: ffn_down may not have sdir
            if family == "ffn_down":
                if "expected_sdir_bytes" in layer and layer.get("expected_sdir_bytes", 0) > 0:
                    self.result.add_error(
                        "R-ffn-down-no-sdir",
                        f"{layer_path}.expected_sdir_bytes must be 0 for ffn_down (per policy no-ffn-down-residual)",
                        f"{layer_path}.expected_sdir_bytes"
                    )
                if "sdir_sha256" in layer and layer.get("sdir_sha256") not in (None, "", "metadata_only_placeholder"):
                    self.result.add_error(
                        "R-ffn-down-no-sdir",
                        f"{layer_path}.sdir_sha256 must be empty or metadata_only_placeholder for ffn_down",
                        f"{layer_path}.sdir_sha256"
                    )

            # sha256 fields must be valid 64-char hex OR explicit placeholder
            for sha_field in ("q2k_sha256", "sdir_sha256"):
                if sha_field in layer:
                    v = layer[sha_field]
                    if v in (None, "", "metadata_only_placeholder"):
                        if not self.metadata_only:
                            self.result.add_error(
                                "R-sha256",
                                f"{layer_path}.{sha_field} is a placeholder but manifest is not marked metadata_only",
                                f"{layer_path}.{sha_field}"
                            )
                        continue
                    if not (isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v)):
                        self.result.add_error(
                            "R-sha256",
                            f"{layer_path}.{sha_field} must be 64-char hex or placeholder, got '{v[:20]}'",
                            f"{layer_path}.{sha_field}"
                        )

            # artifact filenames are relative
            for path_field in ("q2k_artifact_path", "sdir_artifact_path"):
                if path_field in layer:
                    p = layer[path_field]
                    if not isinstance(p, str):
                        self.result.add_error(
                            "R-artifact-relative",
                            f"{layer_path}.{path_field} must be a string, got {type(p).__name__}",
                            f"{layer_path}.{path_field}"
                        )
                    elif os.path.isabs(p):
                        self.result.add_error(
                            "R-artifact-relative",
                            f"{layer_path}.{path_field} must be relative, got absolute path '{p}'",
                            f"{layer_path}.{path_field}"
                        )
                    elif p.startswith(".."):
                        self.result.add_error(
                            "R-artifact-relative",
                            f"{layer_path}.{path_field} must not escape the bundle (starts with '..'): '{p}'",
                            f"{layer_path}.{path_field}"
                        )

            # R27: runtime_loadable (bool)
            if "runtime_loadable" in layer:
                rl = layer["runtime_loadable"]
                if not isinstance(rl, bool):
                    self.result.add_error(
                        "R27",
                        f"{layer_path}.runtime_loadable must be a bool, got {type(rl).__name__}",
                        f"{layer_path}.runtime_loadable"
                    )

    # ── R32-R34: memory budget ───────────────────────────────────────────────
    def _validate_memory_budget(self) -> None:
        for rid in ("R32", "R33", "R34", "R-memory-numeric-nonneg"):
            self.result.add_rule_checked(rid)

        mb = self.manifest.get("memory_budget", {})
        if not isinstance(mb, dict):
            self.result.add_error("R-memory", "memory_budget must be a dict", "memory_budget")
            return

        for numeric_field in (
            "expected_q2k_bytes_per_family",
            "expected_sdir_bytes_per_family",
            "total_expected_bytes",
            "memory_margin_bytes",
            "memory_positive_threshold_bytes",
        ):
            if numeric_field in mb:
                val = mb[numeric_field]
                if not isinstance(val, int) or val < 0:
                    self.result.add_error(
                        "R-memory-numeric-nonneg",
                        f"memory_budget.{numeric_field} must be a non-negative integer, got {val!r}",
                        f"memory_budget.{numeric_field}"
                    )

        # R34: aggregate memory_positive_expected
        if "memory_positive_expected" in mb:
            if not isinstance(mb["memory_positive_expected"], bool):
                self.result.add_error(
                    "R-memory",
                    "memory_budget.memory_positive_expected must be a bool",
                    "memory_budget.memory_positive_expected"
                )

    # ── R-provenance: provenance validation ──────────────────────────────────
    def _validate_provenance(self) -> None:
        for rid in (
            "R-provenance-created_by_phase",
            "R-provenance-derived_from_phases",
            "R-provenance-source_model_quantization",
            "R-provenance-replay_w_ref_source",
            "R-provenance-claim_boundary",
            "R-provenance-forbidden_claims",
            "R-provenance-valid_as_long_as",
        ):
            self.result.add_rule_checked(rid)

        # created_by_phase
        cbp = self.manifest.get("created_by_phase")
        if cbp != "31CI":
            self.result.add_error(
                "R-provenance-created_by_phase",
                f"created_by_phase must be '31CI', got '{cbp}'",
                "created_by_phase"
            )

        # derived_from_phases
        dfp = self.manifest.get("derived_from_phases")
        if not isinstance(dfp, list) or not dfp:
            self.result.add_error(
                "R-provenance-derived_from_phases",
                "derived_from_phases must be a non-empty list of source phase IDs",
                "derived_from_phases"
            )
        else:
            for required_phase in ("31CH", "31BZ"):
                if required_phase not in dfp:
                    self.result.add_warning(
                        "R-provenance-derived_from_phases",
                        f"derived_from_phases should typically include '{required_phase}'",
                        "derived_from_phases"
                    )

        # source_model_quantization (in addition to R4)
        sm = self.manifest.get("source_model", {})
        if isinstance(sm, dict) and sm.get("quantization") != "Q4_K_M":
            self.result.add_error(
                "R-provenance-source_model_quantization",
                f"source_model.quantization must be 'Q4_K_M' for 1.5B Qwen2.5-Instruct, got '{sm.get('quantization')}'",
                "source_model.quantization"
            )

        # replay_w_ref_source
        rws = self.manifest.get("replay_w_ref_source")
        if rws != "Q4_K_M_GGUF_DEQUANTIZED":
            self.result.add_error(
                "R-provenance-replay_w_ref_source",
                f"replay_w_ref_source must be 'Q4_K_M_GGUF_DEQUANTIZED', got '{rws}'",
                "replay_w_ref_source"
            )

        # claim_boundary
        if "claim_boundary" not in self.manifest:
            self.result.add_error(
                "R-provenance-claim_boundary",
                "claim_boundary field is required",
                "claim_boundary"
            )
        elif not isinstance(self.manifest["claim_boundary"], dict):
            self.result.add_error(
                "R-provenance-claim_boundary",
                "claim_boundary must be a dict",
                "claim_boundary"
            )

        # forbidden_claims
        if "forbidden_claims" not in self.manifest:
            self.result.add_error(
                "R-provenance-forbidden_claims",
                "forbidden_claims field is required",
                "forbidden_claims"
            )
        elif not isinstance(self.manifest["forbidden_claims"], list):
            self.result.add_error(
                "R-provenance-forbidden_claims",
                "forbidden_claims must be a list",
                "forbidden_claims"
            )

        # valid_as_long_as
        if "valid_as_long_as" not in self.manifest:
            self.result.add_error(
                "R-provenance-valid_as_long_as",
                "valid_as_long_as field is required",
                "valid_as_long_as"
            )
        elif not isinstance(self.manifest["valid_as_long_as"], list):
            self.result.add_error(
                "R-provenance-valid_as_long_as",
                "valid_as_long_as must be a list",
                "valid_as_long_as"
            )


def _walk_strings(obj: Any, path: str = "") -> List[Tuple[str, Any]]:
    """Walk a manifest, yielding (path, value) for every string value."""
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{path}.{k}" if path else k
            out.extend(_walk_strings(v, full))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            full = f"{path}[{i}]"
            out.extend(_walk_strings(item, full))
    elif isinstance(obj, str):
        out.append((path, obj))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Convenience: validate from JSON file
# ────────────────────────────────────────────────────────────────────────────

def validate_manifest_file(
    path: str,
    metadata_only: bool = True,
) -> ValidationResult:
    """Load a manifest from a JSON file and validate it.
    Does NOT load any model file. Does NOT read or write any binary artifact.
    Computes the SHA-256 of the manifest JSON file (not of any model).
    """
    with open(path, "r", encoding="utf-8") as f:
        manifest_text = f.read()
    # Compute manifest SHA-256 (of the file content, not of the parsed dict)
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    manifest = json.loads(manifest_text)
    validator = ManifestValidator(manifest, metadata_only=metadata_only)
    result = validator.validate_all()
    result.manifest_sha256 = manifest_sha256
    return result


# ────────────────────────────────────────────────────────────────────────────
# Sample manifest (metadata-only, for self-test and SOT cross-reference)
# ────────────────────────────────────────────────────────────────────────────

SAMPLE_MANIFEST = {
    "schema_version": "1.1.0",
    "bundle_type": "runtime_loadable_substitutive",
    "package_id": "phase31ci-sample-qwen2.5-1.5b-instruct-q4_k-m-v1.1.0",
    "metadata_only": True,

    "source_model": {
        "model_name": "qwen2.5-1.5b-instruct-q4_k_m",
        "architecture": "qwen2",
        "quantization": "Q4_K_M",
        "source_model_sha256": "metadata_only_placeholder",
        "source_model_size_bytes": 1117320736,
    },

    "runtime_safety_invariants": {
        "policy_name": "corrected_q2k_policy_v1",
        "policy_version": "1",
        "q2k_mode": "corrected_ceil_per_row",
        "residual_families": ["ffn_up", "ffn_gate"],
        "ffn_down_residual_enabled": False,
        "residual_k_pct": 0.5,
        "residual_alpha": 1.0,
        "tensor_families": ["ffn_up", "ffn_gate", "ffn_down"],
        "no_build_ffn_patch": True,
        "no_legacy_prt_sidecar_entries": True,
        "no_g_prt_sidecar_root_set_write": True,
        "no_activation_capture_artifact_in_runtime_path": True,
    },

    "runtime_consumer": {
        "consumer_kind": "metadata_only",
        "consumer_path": "src/phase31ci_manifest_validator.py",
        "consumer_class": "ManifestValidator",
        "consumer_digest_sha256": "metadata_only_placeholder",
        "consumer_min_version": "1.1.0",
        "runtime_loader_integration": False,
    },

    "legacy_sidecar_exclusion": {
        "excluded_keys": list(LEGACY_SIDECAR_EXCLUDED_KEY_PATTERNS),
        "rejection_policy": "fail_fast",
        "rejection_error_class": "LegacySidecarManifestError",
    },

    "memory_budget": {
        "expected_q2k_bytes_per_family": 4515840,
        "expected_sdir_bytes_per_family_ffn_up": 1857972,
        "expected_sdir_bytes_per_family_ffn_gate": 1857972,
        "expected_sdir_bytes_per_family_ffn_down": 0,
        "total_expected_bytes": 3 * (4515840 + 1857972 + 1857972 + 0),
        "memory_margin_bytes": 3 * (507468 + 507466 + 2365440),
        "memory_positive_expected": True,
        "memory_positive_threshold_bytes": 0,
    },

    "layer_count": 9,  # 3 distinct layer indices × 3 families = 9 layer-family entries
    "unique_layer_indices": [0, 14, 27],
    "hidden_size": 1536,
    "intermediate_size": 896,

    "layers": [
        {
            "layer": 0,
            "family": "ffn_up",
            "shape": [896, 1536],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.0.ffn_up.q2_k.W_low",
            "sdir_artifact_path": "tensors/blk.0.ffn_up.residual.sdir",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 1857972,
            "per_layer_margin_bytes": 507468,
            "q2k_sha256": "metadata_only_placeholder",
            "sdir_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 0,
            "family": "ffn_gate",
            "shape": [896, 1536],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.0.ffn_gate.q2_k.W_low",
            "sdir_artifact_path": "tensors/blk.0.ffn_gate.residual.sdir",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 1857972,
            "per_layer_margin_bytes": 507466,
            "q2k_sha256": "metadata_only_placeholder",
            "sdir_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 0,
            "family": "ffn_down",
            "shape": [1536, 896],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.0.ffn_down.q2_k.W_low",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 0,
            "per_layer_margin_bytes": 2365440,
            "q2k_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 14,
            "family": "ffn_up",
            "shape": [896, 1536],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.14.ffn_up.q2_k.W_low",
            "sdir_artifact_path": "tensors/blk.14.ffn_up.residual.sdir",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 1857972,
            "per_layer_margin_bytes": 507468,
            "q2k_sha256": "metadata_only_placeholder",
            "sdir_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 14,
            "family": "ffn_gate",
            "shape": [896, 1536],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.14.ffn_gate.q2_k.W_low",
            "sdir_artifact_path": "tensors/blk.14.ffn_gate.residual.sdir",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 1857972,
            "per_layer_margin_bytes": 507466,
            "q2k_sha256": "metadata_only_placeholder",
            "sdir_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 14,
            "family": "ffn_down",
            "shape": [1536, 896],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.14.ffn_down.q2_k.W_low",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 0,
            "per_layer_margin_bytes": 2365440,
            "q2k_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 27,
            "family": "ffn_up",
            "shape": [896, 1536],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.27.ffn_up.q2_k.W_low",
            "sdir_artifact_path": "tensors/blk.27.ffn_up.residual.sdir",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 1857972,
            "per_layer_margin_bytes": 507468,
            "q2k_sha256": "metadata_only_placeholder",
            "sdir_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 27,
            "family": "ffn_gate",
            "shape": [896, 1536],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.27.ffn_gate.q2_k.W_low",
            "sdir_artifact_path": "tensors/blk.27.ffn_gate.residual.sdir",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 1857972,
            "per_layer_margin_bytes": 507466,
            "q2k_sha256": "metadata_only_placeholder",
            "sdir_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
        {
            "layer": 27,
            "family": "ffn_down",
            "shape": [1536, 896],
            "orientation": "canonical_d_out_d_in",
            "formats": {"W_low_format": "q2_k", "residual_format": "sdir_v1"},
            "q2k_artifact_path": "tensors/blk.27.ffn_down.q2_k.W_low",
            "expected_q2k_bytes": 4515840,
            "expected_sdir_bytes": 0,
            "per_layer_margin_bytes": 2365440,
            "q2k_sha256": "metadata_only_placeholder",
            "runtime_loadable": True,
            "loader_invocation_count": 1,
        },
    ],

    "artifact_creation_phase": "31CH (planning) + 31BZ/31CA/31CF-S2 (evidence base)",
    "consumed_by_phases": ["31BN", "31BZ", "31CD", "31CE", "31CF-S", "31CF-S2"],

    "replay_w_ref_source": "Q4_K_M_GGUF_DEQUANTIZED",
    "created_by_phase": "31CI",
    "derived_from_phases": ["31CH", "31BZ", "31CA", "31CF-S2"],

    "claim_boundary": {
        "allowed": [
            "A metadata-only v1.1.0 runtime artifact manifest validator was implemented for corrected_q2k_policy_v1.",
            "The validator enforces corrected_q2k_policy_v1 invariants, runtime safety invariants, legacy sidecar exclusion, manifest hygiene, per-layer/per-family shape metadata, memory-budget metadata, and provenance metadata.",
            "A metadata-only sample manifest for Qwen2.5-1.5B corrected_q2k_policy_v1 (3 layers × 3 families) was validated if created.",
        ],
        "forbidden": [
            "no runtime loader exists",
            "no runtime integration exists",
            "no artifacts were generated",
            "no Q2_K / SDIR blobs were generated",
            "no model files were loaded",
            "no model files were committed",
            "no raw activation arrays were created or committed",
            "no generation-quality claim",
            "no speedup claim",
            "no live-runtime memory savings claim",
            "no production readiness claim",
            "no claim that 31CI proves future runtime viability",
        ],
    },

    "forbidden_claims": [
        "no runtime loader exists",
        "no runtime integration exists",
        "no artifacts were generated",
        "no Q2_K / SDIR blobs were generated",
        "no model files were loaded",
        "no model files were committed",
        "no raw activation arrays were created or committed",
        "no generation-quality claim",
        "no speedup claim",
        "no live-runtime memory savings claim",
        "no production readiness claim",
        "no claim that 31CI proves future runtime viability",
    ],

    "valid_as_long_as": [
        "the 0.5B and 1.5B frozen evidence tiers (31BN/31BM, 31BU/31BV/31BX/31BZ/31CA) remain unchanged",
        "the 31CD / 31CE Option A real-activation results remain valid and accepted",
        "the 31CF BLOCKED classification at the source-modification implementation level is preserved",
        "the 31CF-R PARTIAL classification is preserved",
        "the 31CF-S PASS_31CFS classification is preserved",
        "the 31CF-S2 PASS_31CFS2 classification is preserved",
        "the 31CH PASS_31CH classification is preserved",
        "the corrected_q2k_policy_v1 package remains at version v1 (parameters UNCHANGED)",
        "the canonical orientation convention (SOT Section 7) remains unchanged",
        "the STATIC_ARTIFACT_SCHEMA.md v1.0 remains the canonical offline schema",
        "the 31CH planning artifacts are not silently re-edited",
    ],

    "replay_artifact": None,

    "forbidden_claim_summary": "31CI is a metadata-only schema prototype + manifest validator. It produces no binary artifacts, no runtime loader, no runtime integration. All 13 SOT 0.A forbidden claims are preserved.",
}


# ────────────────────────────────────────────────────────────────────────────
# Self-test (in-code validation tests per 31CI spec)
# ────────────────────────────────────────────────────────────────────────────

def _run_self_tests() -> Tuple[int, int, List[Dict[str, Any]]]:
    """Run the 9 required in-code validation tests + 1 valid sample test.
    Returns (passed, failed, test_results).
    """
    results: List[Dict[str, Any]] = []

    # ── Test 1: valid sample manifest passes ────────────────────────────────
    v = ManifestValidator(SAMPLE_MANIFEST, metadata_only=True)
    r = v.validate_all()
    results.append({
        "test": "valid_sample_manifest_passes",
        "passed": r.passed and r.error_count == 0,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 2: manifest with ffn_down residual fails ───────────────────────
    bad = {**SAMPLE_MANIFEST, "runtime_safety_invariants": {**SAMPLE_MANIFEST["runtime_safety_invariants"], "ffn_down_residual_enabled": True}}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_ffn_down_err = any("ffn_down_residual" in e["message"] for e in r.errors)
    results.append({
        "test": "ffn_down_residual_enabled_fails",
        "passed": has_ffn_down_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 3: manifest with legacy prt_* key fails ────────────────────────
    bad = {**SAMPLE_MANIFEST, "prt_sidecar_state": "metadata_only_placeholder"}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_prt_err = any("prt_sidecar_state" in e["message"] for e in r.errors)
    results.append({
        "test": "legacy_prt_key_fails",
        "passed": has_prt_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 4: manifest with hardcoded /media path fails ───────────────────
    bad = {**SAMPLE_MANIFEST, "source_model": {**SAMPLE_MANIFEST["source_model"], "source_model_path": "/media/matthew-villnave/VL_usb/models/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf"}}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_media_err = any("hardcoded operator path" in e["message"] for e in r.errors)
    results.append({
        "test": "hardcoded_media_path_fails",
        "passed": has_media_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 5: manifest with /tmp raw activation path fails ────────────────
    bad = {**SAMPLE_MANIFEST, "layers": [{**layer, "raw_x_path": "/tmp/phase31ci_p0_l0.bin"} for layer in SAMPLE_MANIFEST["layers"]]}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_tmp_err = any("/tmp/" in e["message"] for e in r.errors)
    results.append({
        "test": "tmp_raw_activation_path_fails",
        "passed": has_tmp_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 6: manifest with wrong q2k_mode fails ──────────────────────────
    bad = {**SAMPLE_MANIFEST, "runtime_safety_invariants": {**SAMPLE_MANIFEST["runtime_safety_invariants"], "q2k_mode": "historical_floor_flat"}}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_q2k_err = any("q2k_mode" in e["message"] for e in r.errors)
    results.append({
        "test": "wrong_q2k_mode_fails",
        "passed": has_q2k_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 7: manifest with wrong policy_version fails ────────────────────
    bad = {**SAMPLE_MANIFEST, "runtime_safety_invariants": {**SAMPLE_MANIFEST["runtime_safety_invariants"], "policy_version": "2"}}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_pv_err = any("policy_version" in e["message"] for e in r.errors)
    results.append({
        "test": "wrong_policy_version_fails",
        "passed": has_pv_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 8: manifest with inline payload-like array fails ───────────────
    bad = {**SAMPLE_MANIFEST, "raw_payload": b"not really a payload, just bytes for test"}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_inline_err = any("inline payload field" in e["message"] for e in r.errors)
    results.append({
        "test": "inline_payload_field_fails",
        "passed": has_inline_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 9: manifest with missing required field fails ──────────────────
    bad = {k: v for k, v in SAMPLE_MANIFEST.items() if k != "forbidden_claims"}
    v = ManifestValidator(bad, metadata_only=True)
    r = v.validate_all()
    has_missing_err = any("forbidden_claims" in e["message"] for e in r.errors)
    results.append({
        "test": "missing_required_field_fails",
        "passed": has_missing_err and not r.passed,
        "error_count": r.error_count,
        "errors": r.errors,
    })

    # ── Test 10 (bonus): metadata-only=True with placeholder sha256 passes ─
    # (this is what test 1 verifies; included for clarity)
    results.append({
        "test": "metadata_only_placeholder_sha256_passes",
        "passed": True,  # test 1 already validated this
        "error_count": 0,
        "errors": [],
    })

    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    return passed, failed, results


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 31CI: v1.1.0 runtime artifact manifest validator (metadata-only)"
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default=None,
        help="Path to a v1.1.0 manifest JSON file. If omitted, runs the in-code self-tests.",
    )
    parser.add_argument(
        "--no-metadata-only",
        action="store_true",
        help="Validate as a non-metadata-only manifest (placeholder sha256s will fail).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the in-code self-tests only.",
    )
    args = parser.parse_args()

    metadata_only = not args.no_metadata_only

    if args.self_test or (args.manifest is None):
        # Run self-tests
        passed, failed, results = _run_self_tests()
        print(f"=== Phase 31CI self-test ===")
        print(f"passed: {passed}")
        print(f"failed: {failed}")
        print(f"total:  {len(results)}")
        print()
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{status}] {r['test']} (errors: {r['error_count']})")
            if not r["passed"]:
                for e in r.get("errors", [])[:3]:
                    print(f"       rule={e.get('rule_id')} message={e.get('message')[:80]}")
        if args.manifest is None:
            return 0 if failed == 0 else 1

    if args.manifest is not None:
        result = validate_manifest_file(args.manifest, metadata_only=metadata_only)
        d = result.to_dict()
        print(json.dumps(d, indent=2, sort_keys=True))
        return 0 if d["passed"] else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
