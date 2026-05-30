#!/usr/bin/env python3
"""
31AJ-clean manifest runtime primitives.

This module intentionally contains no fixture generation in the substitutive
runtime path. Fixture/reference generation belongs in tests or phase scripts.
"""

import os
import struct
import sys
from typing import Any, Dict, Tuple

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from bundle_manifest import ManifestLoader, sha256_bytes, write_sdiw  # noqa: E402

BLOCK_SIZE = 32
RSC_MAGIC = b"RSC\x00"
RSC_HEADER = "<4sHHIIIIHH"
RSC_HEADER_BYTES = struct.calcsize(RSC_HEADER)


def cosine(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def pack_wlow(W: np.ndarray) -> Tuple[bytes, bytes]:
    """Pack canonical W shape (d_out, d_in) to nibble bytes + fp16 scale bytes."""
    d_out, d_in = W.shape
    n = d_out * d_in
    if n % BLOCK_SIZE != 0:
        raise ValueError("W size must be divisible by BLOCK_SIZE")
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    scales = np.zeros(n // BLOCK_SIZE, dtype=np.float16)
    for block_idx in range(n // BLOCK_SIZE):
        block = W.flat[block_idx * BLOCK_SIZE:(block_idx + 1) * BLOCK_SIZE]
        scale = float(np.abs(block).max()) / 7.5
        if scale < 1e-9:
            scale = 1.0
        scales[block_idx] = np.float16(scale)
        q = np.clip(np.round(block / scale), -8.0, 7.0).astype(np.int8)
        q_stored = (q + 8).astype(np.uint8)
        out = block_idx * (BLOCK_SIZE // 2)
        for i in range(BLOCK_SIZE // 2):
            packed[out + i] = (q_stored[2 * i] & 0x0F) | ((q_stored[2 * i + 1] & 0x0F) << 4)
    return packed.tobytes(), scales.tobytes()


def unpack_wlow(packed: bytes, scales: bytes, d_out: int, d_in: int) -> np.ndarray:
    scales_arr = np.frombuffer(scales, dtype=np.float16)
    n = d_out * d_in
    expected_packed = (n + 1) // 2
    expected_scales = (n // BLOCK_SIZE)
    if len(packed) != expected_packed:
        raise ValueError(f"packed bytes {len(packed)} != expected {expected_packed}")
    if len(scales_arr) != expected_scales:
        raise ValueError(f"scale count {len(scales_arr)} != expected {expected_scales}")
    W = np.zeros(n, dtype=np.float32)
    for block_idx in range(n // BLOCK_SIZE):
        scale = float(scales_arr[block_idx])
        in_byte = block_idx * (BLOCK_SIZE // 2)
        out = block_idx * BLOCK_SIZE
        for i in range(BLOCK_SIZE // 2):
            byte = packed[in_byte + i]
            W[out + 2 * i] = (float(byte & 0x0F) - 8.0) * scale
            W[out + 2 * i + 1] = (float((byte >> 4) & 0x0F) - 8.0) * scale
    return W.reshape(d_out, d_in)


def sdiw_streaming_apply(packed: bytes, scales: bytes, X: np.ndarray, d_out: int, d_in: int) -> np.ndarray:
    """Compute Y = X @ W.T from packed W shaped (d_out, d_in), without dense W."""
    if X.shape != (d_in,):
        raise ValueError(f"X shape {X.shape} != expected ({d_in},)")
    scales_arr = np.frombuffer(scales, dtype=np.float16)
    blocks_per_row = d_in // BLOCK_SIZE
    if d_in % BLOCK_SIZE != 0:
        raise ValueError("d_in must be divisible by BLOCK_SIZE")
    Y = np.zeros(d_out, dtype=np.float32)
    scratch = np.zeros(BLOCK_SIZE, dtype=np.float32)
    for row in range(d_out):
        acc = 0.0
        for block_col in range(blocks_per_row):
            block_idx = row * blocks_per_row + block_col
            scale = float(scales_arr[block_idx])
            byte_base = block_idx * (BLOCK_SIZE // 2)
            for i in range(BLOCK_SIZE // 2):
                byte = packed[byte_base + i]
                scratch[2 * i] = (float(byte & 0x0F) - 8.0) * scale
                scratch[2 * i + 1] = (float((byte >> 4) & 0x0F) - 8.0) * scale
            x_base = block_col * BLOCK_SIZE
            acc += float(np.dot(X[x_base:x_base + BLOCK_SIZE], scratch))
        Y[row] = acc
    return Y


def encode_sdir(R: np.ndarray, k_pct: float = 7.5) -> bytes:
    """Encode canonical residual shape (d_out, d_in) as row-major bitmap + fp16 values."""
    d_out, d_in = R.shape
    n = d_out * d_in
    k_nnz = max(1, int(n * k_pct / 100.0))
    threshold = np.partition(np.abs(R).ravel(), -k_nnz)[-k_nnz]
    bitmap = np.zeros(n, dtype=np.uint8)
    values = []
    for idx, value in enumerate(R.ravel()):
        if abs(value) >= threshold:
            bitmap[idx] = 1
            values.append(value)
    values_f16 = np.asarray(values, dtype=np.float16)
    header = struct.pack(
        RSC_HEADER,
        RSC_MAGIC,
        1,
        0,
        d_out,
        d_in,
        int(k_pct * 100),
        len(values_f16),
        0,
        0,
    )
    return header + np.packbits(bitmap).tobytes() + values_f16.tobytes()


def parse_sdir(data: bytes) -> Dict[str, Any]:
    if len(data) < RSC_HEADER_BYTES:
        raise ValueError(".sdir too short")
    magic, version, flags, d_out, d_in, k_pct_int, nnz, r0, r1 = struct.unpack(
        RSC_HEADER, data[:RSC_HEADER_BYTES]
    )
    if magic != RSC_MAGIC:
        raise ValueError(f"bad .sdir magic {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported .sdir version {version}")
    bitmap_nbytes = (d_out * d_in + 7) // 8
    expected = RSC_HEADER_BYTES + bitmap_nbytes + nnz * 2
    if len(data) != expected:
        raise ValueError(f".sdir size {len(data)} != expected {expected}")
    bitmap = np.unpackbits(
        np.frombuffer(data[RSC_HEADER_BYTES:RSC_HEADER_BYTES + bitmap_nbytes], dtype=np.uint8)
    )[:d_out * d_in]
    values = np.frombuffer(data[RSC_HEADER_BYTES + bitmap_nbytes:], dtype=np.float16).astype(np.float32)
    if int(bitmap.sum()) != nnz:
        raise ValueError(f".sdir bitmap nnz {int(bitmap.sum())} != header nnz {nnz}")
    return {
        "d_out": int(d_out),
        "d_in": int(d_in),
        "k_pct": float(k_pct_int) / 100.0,
        "nnz": int(nnz),
        "bitmap": bitmap,
        "values": values,
    }


def decode_sdir(data: bytes) -> np.ndarray:
    parsed = parse_sdir(data)
    R = np.zeros(parsed["d_out"] * parsed["d_in"], dtype=np.float32)
    value_idx = 0
    for idx, bit in enumerate(parsed["bitmap"]):
        if bit:
            R[idx] = parsed["values"][value_idx]
            value_idx += 1
    return R.reshape(parsed["d_out"], parsed["d_in"])


def sdir_streaming_apply(data: bytes, X: np.ndarray, d_out: int, d_in: int) -> Tuple[np.ndarray, int]:
    """Compute Y_delta = X @ R.T from .sdir shaped (d_out, d_in), no dense R."""
    if X.shape != (d_in,):
        raise ValueError(f"X shape {X.shape} != expected ({d_in},)")
    parsed = parse_sdir(data)
    if parsed["d_out"] != d_out or parsed["d_in"] != d_in:
        raise ValueError(
            f".sdir shape {(parsed['d_out'], parsed['d_in'])} != expected {(d_out, d_in)}"
        )
    Y = np.zeros(d_out, dtype=np.float32)
    value_idx = 0
    for row in range(d_out):
        acc = 0.0
        row_base = row * d_in
        for col in range(d_in):
            if parsed["bitmap"][row_base + col]:
                acc += float(X[col]) * float(parsed["values"][value_idx])
                value_idx += 1
        Y[row] = acc
    return Y, value_idx


class ManifestRuntime:
    def __init__(self):
        self.loader = None
        self.bundle_dir = None
        self.counters = {
            "W_ref_loaded": 0,
            "W_ref_generated": 0,
            "dense_W_low_materialized": 0,
            "dense_R_materialized": 0,
            "sdiw_loaded": 0,
            "sdir_loaded": 0,
            "manifest_loaded": 0,
            "checksum_validated": 0,
            "memory_budget_validated": 0,
            "fallback_count": 0,
            "error_count": 0,
            "path_label": "[SDI-SUB-RUNTIME]",
        }

    def load_and_validate_manifest(self, bundle_dir: str) -> Dict[str, Any]:
        self.bundle_dir = bundle_dir
        self.loader = ManifestLoader(bundle_dir)
        try:
            self.loader.load()
            self.counters["manifest_loaded"] = 1
            result = self.loader.validate_bundle(bundle_dir)
            self.counters["checksum_validated"] = result.get("checksum_validated", 0)
            self.counters["memory_budget_validated"] = result.get("memory_budget_validated", 0)
            self.counters["error_count"] += result.get("error_count", 0)
        except Exception:
            self.counters["error_count"] += 1
        return self.get_counters()

    def execute_substitutive_path(self, entry: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Artifact-only substitutive runtime. Does not generate W_ref/W_low/R."""
        if self.loader is None or self.bundle_dir is None:
            raise RuntimeError("manifest must be loaded before execution")
        d_out, d_in = entry["shape"]
        sdiw = self.loader.load_sdiw(entry, self.bundle_dir)
        sdir_bytes = self.loader.load_sdir(entry, self.bundle_dir)
        if sdiw["rows"] != d_out or sdiw["cols"] != d_in:
            raise ValueError("loaded .sdiw shape does not match manifest entry")
        Y_low = sdiw_streaming_apply(sdiw["packed_bytes"], sdiw["scale_bytes"], X, d_out, d_in)
        Y_delta, nnz_seen = sdir_streaming_apply(sdir_bytes, X, d_out, d_in)
        self.counters["sdiw_loaded"] += 1
        self.counters["sdir_loaded"] += 1
        Y_sub = Y_low + Y_delta
        return Y_sub, {
            "layer": entry["layer"],
            "family": entry["family"],
            "d_out": d_out,
            "d_in": d_in,
            "nnz_seen": nnz_seen,
            "nan_inf": bool(np.isnan(Y_sub).any() or np.isinf(Y_sub).any()),
            "Y_sub_norm": float(np.linalg.norm(Y_sub)),
        }

    def get_counters(self) -> Dict[str, Any]:
        return dict(self.counters)
