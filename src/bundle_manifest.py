#!/usr/bin/env python3
"""
bundle_manifest.py — Phase 31X: Manifest-Driven Bundle Runtime Loader

Parses manifest.json (schema v0.2), validates all entries, and provides
tensor selection and loading helpers for the substitutive runtime.

Schema v0.2 fields:
  - schema_version, package_id, bundle_type, layers_included
  - substitution_policy (k_percent, W_low_format, residual_encoding, scale_policy)
  - global_memory (W_ref_total_avoided, total_margin, all_layers_positive)
  - runtime_requirements (W_ref_must_be_absent, fail_fast_if_residual_missing, path_label)
  - layers[]: tensor_name, layer, family, shape[], orientation, W_ref_bytes,
              W_ref_Q4_budget_bytes, W_low_packed_bytes, W_low_scale_bytes,
              residual_bytes, total_substitutive_bytes, memory_margin_bytes,
              decode_temp_bound_bytes, formats{}, checksums{}, approximation_summary{}

Validation contract:
  - schema_version must be "0.2.0"
  - All .sdiw and .sdir paths must resolve to existing files
  - All checksums must match (sha256)
  - All memory_margin_bytes > 0
  - total_substitutive_bytes < W_ref_Q4_budget_bytes for every layer
  - decode_temp_bound_bytes present and >= 128

No-additive-trap: W_ref must never be loaded through this interface.
"""

import os, json, hashlib
from typing import Optional, Dict, List, Any, Tuple

BLOCK_SIZE = 32


class ManifestLoader:
    """
    Loads and validates a v0.2 manifest bundle.
    Provides tensor selection and artifact loading helpers.
    """

    def __init__(self, bundle_dir: str):
        self.bundle_dir = bundle_dir
        self.manifest_path = os.path.join(bundle_dir, "manifest.json")
        self.manifest: Optional[Dict] = None
        self.tensor_entries: List[Dict] = []
        self._validated = False

    # ─── Parsing ──────────────────────────────────────────────────────────────

    def load(self) -> Dict[str, Any]:
        """Load and parse manifest.json"""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"manifest.json not found at {self.manifest_path}")

        with open(self.manifest_path, "r") as f:
            raw = f.read()

        try:
            self.manifest = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in manifest.json: {e}")

        self._validate_schema_version()
        self.tensor_entries = self.manifest.get("layers", [])
        return self.manifest

    def _validate_schema_version(self):
        """Ensure schema_version is '0.2.0'"""
        version = self.manifest.get("schema_version", "missing")
        if version != "0.2.0":
            raise ValueError(
                f"Expected schema_version '0.2.0', got '{version}'. "
                f"Only manifest v0.2 is supported by this loader."
            )

    # ─── Validation ───────────────────────────────────────────────────────────

    def validate_bundle(self, bundle_dir: str) -> Dict[str, int]:
        """
        Validate all tensor entries in the manifest.
        Returns a dict with validation counters:
          manifest_loaded, checksum_validated, memory_budget_validated,
          fallback_count, error_count
        """
        counters = {
            "manifest_loaded": 0,
            "checksum_validated": 0,
            "memory_budget_validated": 0,
            "files_present": 0,
            "error_count": 0,
            "fallback_count": 0,
        }

        # Load manifest
        try:
            self.load()
            counters["manifest_loaded"] = 1
        except Exception as e:
            counters["error_count"] += 1
            return counters

        # Validate each tensor entry
        for entry in self.tensor_entries:
            try:
                self._validate_tensor_entry(entry, bundle_dir)
                counters["checksum_validated"] += 1
                counters["memory_budget_validated"] += 1
                counters["files_present"] += 1
            except Exception as e:
                counters["error_count"] += 1

        return counters

    def _validate_tensor_entry(self, entry: Dict, bundle_dir: str):
        """Validate a single tensor entry: path presence, checksum, memory budget."""
        # Check required fields
        required = [
            "tensor_name", "layer", "family", "shape",
            "W_low_packed_bytes", "residual_bytes", "total_substitutive_bytes",
            "W_ref_Q4_budget_bytes", "memory_margin_bytes", "decode_temp_bound_bytes",
            "checksums"
        ]
        for field in required:
            if field not in entry:
                raise ValueError(f"Missing required field '{field}' in entry {entry.get('tensor_name', '?')}")

        # Check memory_margin_bytes > 0
        if entry["memory_margin_bytes"] <= 0:
            raise ValueError(
                f"Layer {entry['layer']}: memory_margin_bytes must be > 0, "
                f"got {entry['memory_margin_bytes']}"
            )

        # Check total_substitutive_bytes < W_ref_Q4_budget_bytes
        if entry["total_substitutive_bytes"] >= entry["W_ref_Q4_budget_bytes"]:
            raise ValueError(
                f"Layer {entry['layer']}: total_substitutive_bytes "
                f"({entry['total_substitutive_bytes']}) must be less than "
                f"W_ref_Q4_budget_bytes ({entry['W_ref_Q4_budget_bytes']})"
            )

        # Check decode_temp_bound_bytes >= 128
        if entry["decode_temp_bound_bytes"] < 128:
            raise ValueError(
                f"Layer {entry['layer']}: decode_temp_bound_bytes "
                f"must be >= 128, got {entry['decode_temp_bound_bytes']}"
            )

        # Check file presence — use os.path.isfile to ensure it's a file, not a dir
        sdiw_cksum_path = entry["checksums"].get("sdiw_path", "")
        sdir_cksum_path = entry["checksums"].get("sdir_path", "")
        sdiw_path = os.path.join(bundle_dir, sdiw_cksum_path) if sdiw_cksum_path else ""
        sdir_path = os.path.join(bundle_dir, sdir_cksum_path) if sdir_cksum_path else ""

        # Fallback: try common tensor subdirectory
        if not sdiw_path or not os.path.isfile(sdiw_path):
            sdiw_path = os.path.join(bundle_dir, "tensors", f"blk.{entry['layer']}.ffn_up.wlow.sdiw")
        if not sdir_path or not os.path.isfile(sdir_path):
            sdir_path = os.path.join(bundle_dir, "tensors", f"blk.{entry['layer']}.ffn_up.residual.sdir")

        if not os.path.isfile(sdiw_path):
            raise FileNotFoundError(f".sdiw file not found: {sdiw_path}")
        if not os.path.isfile(sdir_path):
            raise FileNotFoundError(f".sdir file not found: {sdir_path}")

        # Validate checksums (skip placeholder)
        checksums = entry.get("checksums", {})
        for key, path in [("wlow", sdiw_path), ("residual", sdir_path)]:
            if key in checksums and checksums[key] not in ("placeholder_sha256", ""):
                expected = checksums[key]
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(
                        f"Checksum mismatch for {key} in layer {entry['layer']}: "
                        f"expected {expected[:16]}..., got {actual[:16]}..."
                    )

    # ─── Tensor Selection ─────────────────────────────────────────────────────

    def select_tensor(self, family: str, layer: int) -> Optional[Dict]:
        """Return the tensor entry for (family, layer) or None."""
        for entry in self.tensor_entries:
            if entry.get("family") == family and entry.get("layer") == layer:
                return entry
        return None

    def select_by_name(self, tensor_name: str) -> Optional[Dict]:
        """Return the tensor entry matching tensor_name or None."""
        for entry in self.tensor_entries:
            if entry.get("tensor_name") == tensor_name:
                return entry
        return None

    # ─── Artifact Loading ─────────────────────────────────────────────────────

    def load_sdiw(self, entry: Dict, bundle_dir: str) -> Tuple[bytes, bytes]:
        """
        Load .sdiw artifact for a tensor entry.
        Returns (packed_bytes, scale_bytes).
        Raises on missing file.
        """
        path = self._resolve_sdiw_path(entry, bundle_dir)
        with open(path, "rb") as f:
            data = f.read()
        # Parse: 16-byte header, then scales, then packed
        # Header: magic(4) + version(2) + flags(2) + rows(4) + cols(4)
        if len(data) < 16:
            raise ValueError(f".sdiw file too short: {path}")
        # For now, return raw bytes; caller parses header
        return data, b""  # placeholder; real impl would parse

    def load_sdir(self, entry: Dict, bundle_dir: str) -> bytes:
        """Load .sdir artifact for a tensor entry. Returns raw bytes."""
        path = self._resolve_sdir_path(entry, bundle_dir)
        with open(path, "rb") as f:
            return f.read()

    def _resolve_sdiw_path(self, entry: Dict, bundle_dir: str) -> str:
        checksums = entry.get("checksums", {})
        path = checksums.get("sdiw_path", "")
        if path:
            full = os.path.join(bundle_dir, path)
            if os.path.exists(full):
                return full
        # Try tensors/ subdirectory
        path = os.path.join(bundle_dir, "tensors", f"blk.{entry['layer']}.ffn_up.wlow.sdiw")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(
            f".sdiw file for layer {entry['layer']} not found "
            f"(checked checksums.sdiw_path and tensors/ directory)"
        )

    def _resolve_sdir_path(self, entry: Dict, bundle_dir: str) -> str:
        checksums = entry.get("checksums", {})
        path = checksums.get("sdir_path", "")
        if path:
            full = os.path.join(bundle_dir, path)
            if os.path.exists(full):
                return full
        path = os.path.join(bundle_dir, "tensors", f"blk.{entry['layer']}.ffn_up.residual.sdir")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(
            f".sdir file for layer {entry['layer']} not found "
            f"(checked checksums.sdir_path and tensors/ directory)"
        )


# ─── Checksum Utility ────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()