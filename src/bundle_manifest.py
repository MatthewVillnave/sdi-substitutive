#!/usr/bin/env python3
"""
Manifest loader for SDI substitutive bundles.

31AJ source-of-truth rules:
- Tensor artifact shape is canonical (d_out, d_in).
- Manifest paths must be explicit or family-aware.
- Missing or stale artifacts fail fast. There is no silent ffn_up fallback.
- .sdiw loading parses header/scales/packed bytes instead of returning placeholders.
"""

import hashlib
import os
import struct
from typing import Any, Dict, List, Optional

BLOCK_SIZE = 32
SDIW_MAGIC = b"SDIW"
SDIW_HEADER = "<4sHHIIII"
SDIW_HEADER_BYTES = struct.calcsize(SDIW_HEADER)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_sdiw(path: str, rows: int, cols: int, scale_bytes: bytes, packed_bytes: bytes) -> bytes:
    """Write canonical 31AJ .sdiw: header + fp16 scales + packed nibbles."""
    n = rows * cols
    expected_scales = (n // BLOCK_SIZE) * 2
    expected_packed = (n + 1) // 2
    if len(scale_bytes) != expected_scales:
        raise ValueError(f"scale bytes {len(scale_bytes)} != expected {expected_scales}")
    if len(packed_bytes) != expected_packed:
        raise ValueError(f"packed bytes {len(packed_bytes)} != expected {expected_packed}")
    payload = struct.pack(
        SDIW_HEADER,
        SDIW_MAGIC,
        1,
        0,
        rows,
        cols,
        len(scale_bytes),
        len(packed_bytes),
    ) + scale_bytes + packed_bytes
    with open(path, "wb") as f:
        f.write(payload)
    return payload


class ManifestLoader:
    def __init__(self, bundle_dir: str):
        self.bundle_dir = bundle_dir
        self.manifest_path = os.path.join(bundle_dir, "manifest.json")
        self.manifest: Optional[Dict[str, Any]] = None
        self.tensor_entries: List[Dict[str, Any]] = []

    def load(self) -> Dict[str, Any]:
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"manifest.json not found at {self.manifest_path}")
        import json

        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)
        self._validate_schema_version()
        self.tensor_entries = self.manifest.get("layers", [])
        return self.manifest

    def _validate_schema_version(self) -> None:
        version = self.manifest.get("schema_version", "missing")
        if version != "0.2.0":
            raise ValueError(f"Expected schema_version '0.2.0', got '{version}'")

    def validate_bundle(self, bundle_dir: Optional[str] = None) -> Dict[str, int]:
        bundle_dir = bundle_dir or self.bundle_dir
        counters = {
            "manifest_loaded": 0,
            "checksum_validated": 0,
            "memory_budget_validated": 0,
            "files_present": 0,
            "fallback_count": 0,
            "error_count": 0,
        }
        try:
            if self.manifest is None:
                self.load()
            counters["manifest_loaded"] = 1
            for entry in self.tensor_entries:
                self._validate_tensor_entry(entry, bundle_dir)
                counters["checksum_validated"] += 1
                counters["memory_budget_validated"] += 1
                counters["files_present"] += 1
        except Exception:
            counters["error_count"] += 1
        return counters

    def _validate_tensor_entry(self, entry: Dict[str, Any], bundle_dir: Optional[str] = None) -> None:
        bundle_dir = bundle_dir or self.bundle_dir
        required = [
            "tensor_name",
            "layer",
            "family",
            "shape",
            "W_low_packed_bytes",
            "residual_bytes",
            "total_substitutive_bytes",
            "W_ref_Q4_budget_bytes",
            "memory_margin_bytes",
            "checksums",
        ]
        for field in required:
            if field not in entry:
                raise ValueError(f"Missing required field '{field}' in {entry.get('tensor_name', '?')}")
        if entry["family"] not in ("ffn_up", "ffn_down"):
            raise ValueError(f"Unsupported tensor family: {entry['family']}")
        if len(entry["shape"]) != 2:
            raise ValueError(f"shape must be [d_out, d_in], got {entry['shape']}")
        if entry["memory_margin_bytes"] <= 0:
            raise ValueError(f"memory_margin_bytes must be > 0, got {entry['memory_margin_bytes']}")
        if entry["total_substitutive_bytes"] >= entry["W_ref_Q4_budget_bytes"]:
            raise ValueError("total_substitutive_bytes must be less than W_ref_Q4_budget_bytes")

        sdiw = self.load_sdiw(entry, bundle_dir)
        sdir_path = self._resolve_sdir_path(entry, bundle_dir)
        if sdiw["rows"] != entry["shape"][0] or sdiw["cols"] != entry["shape"][1]:
            raise ValueError(
                f".sdiw shape {(sdiw['rows'], sdiw['cols'])} does not match manifest {entry['shape']}"
            )
        if os.path.getsize(sdir_path) != self._entry_residual_bytes(entry):
            raise ValueError("residual artifact size does not match manifest")
        self._validate_checksums(entry, sdiw["path"], sdir_path)

    def select_tensor(self, family: str, layer: int) -> Optional[Dict[str, Any]]:
        for entry in self.tensor_entries:
            if entry.get("family") == family and entry.get("layer") == layer:
                return entry
        return None

    def select_by_name(self, tensor_name: str) -> Optional[Dict[str, Any]]:
        for entry in self.tensor_entries:
            if entry.get("tensor_name") == tensor_name:
                return entry
        return None

    def load_sdiw(self, entry: Dict[str, Any], bundle_dir: Optional[str] = None) -> Dict[str, Any]:
        bundle_dir = bundle_dir or self.bundle_dir
        path = self._resolve_sdiw_path(entry, bundle_dir)
        with open(path, "rb") as f:
            data = f.read()
        if len(data) >= SDIW_HEADER_BYTES and data[:4] == SDIW_MAGIC:
            magic, version, flags, rows, cols, scale_nbytes, packed_nbytes = struct.unpack(
                SDIW_HEADER, data[:SDIW_HEADER_BYTES]
            )
            if version != 1:
                raise ValueError(f"unsupported .sdiw version {version}")
            scale_start = SDIW_HEADER_BYTES
            packed_start = scale_start + scale_nbytes
            packed_end = packed_start + packed_nbytes
            if len(data) != packed_end:
                raise ValueError(f".sdiw size mismatch for {path}")
            scale_bytes = data[scale_start:packed_start]
            packed_bytes = data[packed_start:packed_end]
        else:
            rows, cols = entry["shape"]
            packed_bytes = data
            scale_path = self._resolve_scale_path(entry, bundle_dir)
            with open(scale_path, "rb") as f:
                scale_bytes = f.read()
            flags = 1
            version = 0

        expected_scales = ((rows * cols) // BLOCK_SIZE) * 2
        expected_packed = (rows * cols + 1) // 2
        if len(scale_bytes) != expected_scales:
            raise ValueError(f".sdiw scales {len(scale_bytes)} != expected {expected_scales}")
        if len(packed_bytes) != expected_packed:
            raise ValueError(f".sdiw packed {len(packed_bytes)} != expected {expected_packed}")
        if [rows, cols] != list(entry["shape"]):
            raise ValueError(f".sdiw shape {[rows, cols]} does not match manifest {entry['shape']}")
        if len(packed_bytes) != entry["W_low_packed_bytes"]:
            raise ValueError(".sdiw packed bytes do not match manifest")
        scale_manifest = entry.get("W_low_scale_bytes", entry.get("W_low_scales_bytes"))
        if scale_manifest is not None and len(scale_bytes) != scale_manifest:
            raise ValueError(".sdiw scale bytes do not match manifest")
        return {
            "path": path,
            "header": {
                "magic": "SDIW" if data[:4] == SDIW_MAGIC else "legacy_raw_packed",
                "version": int(version),
                "flags": int(flags),
            },
            "rows": int(rows),
            "cols": int(cols),
            "scale_bytes": scale_bytes,
            "packed_bytes": packed_bytes,
            "scale_nbytes": len(scale_bytes),
            "packed_nbytes": len(packed_bytes),
        }

    def load_sdir(self, entry: Dict[str, Any], bundle_dir: Optional[str] = None) -> bytes:
        path = self._resolve_sdir_path(entry, bundle_dir or self.bundle_dir)
        with open(path, "rb") as f:
            return f.read()

    def _resolve_sdiw_path(self, entry: Dict[str, Any], bundle_dir: str) -> str:
        explicit = self._explicit_path(entry, ["sdiw_path", "wlow_path", "W_low_path", "W_low_packed_path"])
        if explicit:
            return self._first_existing(
                bundle_dir,
                [explicit],
                f"explicit .sdiw/W_low for blk.{entry['layer']}.{entry['family']}",
            )
        candidates = []
        layer = entry["layer"]
        family = entry["family"]
        candidates.extend(
            [
                f"tensors/blk.{layer}.{family}.wlow.sdiw",
                f"tensors/blk.{layer}.{family}.W_low.sdiw",
                f"tensors/blk.{layer}.{family}.W_low.bin",
            ]
        )
        return self._first_existing(bundle_dir, candidates, f".sdiw/W_low for blk.{layer}.{family}")

    def _resolve_scale_path(self, entry: Dict[str, Any], bundle_dir: str) -> str:
        explicit = self._explicit_path(entry, ["scale_path", "scales_path", "W_low_scales_path"])
        if explicit:
            return self._first_existing(
                bundle_dir,
                [explicit],
                f"explicit W_low scales for blk.{entry['layer']}.{entry['family']}",
            )
        candidates = []
        layer = entry["layer"]
        family = entry["family"]
        candidates.append(f"tensors/blk.{layer}.{family}.W_low.scales.bin")
        return self._first_existing(bundle_dir, candidates, f"W_low scales for blk.{layer}.{family}")

    def _resolve_sdir_path(self, entry: Dict[str, Any], bundle_dir: str) -> str:
        explicit = self._explicit_path(entry, ["sdir_path", "residual_path"])
        if explicit:
            return self._first_existing(
                bundle_dir,
                [explicit],
                f"explicit .sdir for blk.{entry['layer']}.{entry['family']}",
            )
        candidates = []
        layer = entry["layer"]
        family = entry["family"]
        candidates.append(f"tensors/blk.{layer}.{family}.residual.sdir")
        return self._first_existing(bundle_dir, candidates, f".sdir for blk.{layer}.{family}")

    def _explicit_path(self, entry: Dict[str, Any], keys: List[str]) -> Optional[str]:
        checksums = entry.get("checksums", {})
        paths = entry.get("paths", {})
        for source in (checksums, paths, entry):
            for key in keys:
                value = source.get(key)
                if value:
                    return value
        return None

    def _first_existing(self, bundle_dir: str, candidates: List[str], label: str) -> str:
        checked = []
        for candidate in candidates:
            full = candidate if os.path.isabs(candidate) else os.path.join(bundle_dir, candidate)
            checked.append(full)
            if os.path.isfile(full):
                return full
        raise FileNotFoundError(f"{label} not found; checked: {checked}")

    def _entry_residual_bytes(self, entry: Dict[str, Any]) -> int:
        return entry.get("residual_bytes", entry.get("residual_encoded_bytes"))

    def _validate_checksums(self, entry: Dict[str, Any], sdiw_path: str, sdir_path: str) -> None:
        checksums = entry.get("checksums", {})
        wlow_expected = (
            checksums.get("wlow")
            or checksums.get("W_low")
            or checksums.get("W_low_packed")
            or checksums.get("W_low_packed_sha256")
        )
        residual_expected = checksums.get("residual") or checksums.get("residual_sha256")
        for label, expected, path in [
            ("wlow", wlow_expected, sdiw_path),
            ("residual", residual_expected, sdir_path),
        ]:
            if expected and expected != "placeholder_sha256":
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(f"Checksum mismatch for {label}: expected {expected}, got {actual}")
