"""
Q2_K Backend — llama.cpp Q2_K quantization/dequantization via libggml-base.so

Provides a clean interface to llama.cpp's Q2_K quant/dequant for the
SDI-Substitutive static artifact path.

Environment variables (all optional if library is already loaded):
    SDI_LLAMA_CPP_BUILD  — path to llama.cpp build dir
                           (default: $SDI_LLAMA_CPP_ROOT/build)
    SDI_LLAMA_CPP_ROOT  — path to llama.cpp source root
    SDI_LLAMA_CPP_LIB    — explicit path to libggml-base.so
                           (overrides SDI_LLAMA_CPP_BUILD)

If none set and library not yet loaded, raises RuntimeError on first use.

No private paths. All resolution uses env vars or raises with a clear message.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
QK_K = 256          # elements per Q2_K quantization block
Q2_BLOCK_BYTES = 84 # bytes per Q2_K block (QK_K elements → 84 bytes)
DTYPE_F32 = np.float32
DTYPE_U8 = np.uint8

# ─── Library singleton ────────────────────────────────────────────────────────
_lib = None
_lib_path = None


def _resolve_lib_path() -> str:
    """Resolve libggml-base.so path from environment variables."""
    explicit = os.environ.get("SDI_LLAMA_CPP_LIB")
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"SDI_LLAMA_CPP_LIB={explicit} not found")
        return explicit

    build_dir = os.environ.get("SDI_LLAMA_CPP_BUILD")
    if not build_dir:
        root = os.environ.get("SDI_LLAMA_CPP_ROOT", "")
        build_dir = os.path.join(root, "build") if root else ""

    candidate = os.path.join(build_dir, "bin", "libggml-base.so")
    if os.path.isfile(candidate):
        return candidate

    # Search common locations
    common = [
        "/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so",
        "/usr/local/lib/libggml-base.so",
    ]
    for path in common:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "libggml-base.so not found. Set SDI_LLAMA_CPP_LIB, "
        "SDI_LLAMA_CPP_BUILD, or SDI_LLAMA_CPP_ROOT."
    )


def _load_lib():
    """Load and configure libggml-base.so (once per process)."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib

    lib_path = _resolve_lib_path()
    _lib = ctypes.CDLL(lib_path)
    _lib_path = lib_path

    # void quantize_row_q2_K_ref(const float * src, void * dst, int n)
    _lib.quantize_row_q2_K_ref.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib.quantize_row_q2_K_ref.restype = None

    # void dequantize_row_q2_K(const void * src, float * dst, int n)
    _lib.dequantize_row_q2_K.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    _lib.dequantize_row_q2_K.restype = None

    return _lib


def lib():
    """Return the loaded library (load if needed)."""
    return _load_lib()


def lib_path():
    """Return the resolved library path."""
    if _lib_path is None:
        _load_lib()
    return _lib_path


# ─── Public API ───────────────────────────────────────────────────────────────

def q2k_expected_nbytes(d_out: int, d_in: int) -> int:
    """
    Return the Q2_K byte size for ONE ROW of a weight matrix.

    Each row has d_in elements and contributes ceil(d_in / QK_K) blocks.

    Args:
        d_out: number of output rows (d_out dimension of weight matrix)
        d_in: number of input columns per row

    Returns:
        Per-row Q2_K bytes = ceil(d_in / QK_K) * Q2_BLOCK_BYTES
        Total bytes = d_out * this value
    """
    n_blocks_per_row = (d_in + QK_K - 1) // QK_K
    return n_blocks_per_row * Q2_BLOCK_BYTES


def quantize_q2k_f32_to_bytes(
    W: np.ndarray,
    *,
    mode: str = "corrected_ceil_per_row",
) -> bytes:
    """
    Quantize a float32 weight matrix to Q2_K byte format using llama.cpp.

    Two explicit modes:

    mode="corrected_ceil_per_row" (default, technically correct):
        Row-wise quantization with ceil-block accounting.
        Each row gets ceil(d_in/QK_K) blocks.
        Bytes = d_out * ceil(d_in/QK_K) * Q2_BLOCK_BYTES.
        All d_in elements are quantized; partial final block is padded.

    mode="historical_floor_flat" (legacy compatibility):
        Flat (row-major) quantization with floor-block accounting.
        All elements treated as one flat array; only floor(n/QK_K) blocks stored.
        Final partial block is silently DROPPED.
        Bytes = floor(d_out*d_in/QK_K) * Q2_BLOCK_BYTES.
        WARNING: This discards 128 elements per row (last partial Q2_K block).

    Args:
        W: 2D float32 array of shape (d_out, d_in). Each ROW is quantized
           independently with Q2_K.
        mode: "corrected_ceil_per_row" or "historical_floor_flat"

    Returns:
        bytes object of the appropriate length for the chosen mode.

    Raises:
        ValueError: if W.ndim != 2, dtype != float32, or W contains NaN/Inf.
        RuntimeError: if llama.cpp library or symbols unavailable.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2D, got ndim={W.ndim}")
    if W.dtype != DTYPE_F32:
        raise ValueError(f"W must be dtype {DTYPE_F32}, got {W.dtype}")
    if not np.all(np.isfinite(W)):
        raise ValueError("W contains NaN or Inf")

    d_out, d_in = W.shape
    lib = _load_lib()

    if mode == "corrected_ceil_per_row":
        per_row = ((d_in + QK_K - 1) // QK_K) * Q2_BLOCK_BYTES
        out = np.zeros(d_out * per_row, dtype=DTYPE_U8)
        for row_idx in range(d_out):
            row = W[row_idx]
            dst = out[row_idx * per_row:(row_idx + 1) * per_row]
            lib.quantize_row_q2_K_ref(
                row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dst.ctypes.data_as(ctypes.c_void_p),
                d_in,
            )
        return bytes(out)

    elif mode == "historical_floor_flat":
        # Flat quantization: treat the entire matrix as one flat array.
        # Only floor(total_elements / QK_K) blocks are stored.
        # The final partial block (d_in % QK_K = 128 elements) is DROPPED.
        flat = W.flatten()
        n = flat.size
        n_blocks = n // QK_K          # floor — drops final partial block
        buf = np.zeros(n_blocks * Q2_BLOCK_BYTES, dtype=DTYPE_U8)
        lib.quantize_row_q2_K_ref(
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            buf.ctypes.data_as(ctypes.c_void_p),
            n,
        )
        return bytes(buf)

    else:
        raise ValueError(
            f"mode must be 'corrected_ceil_per_row' or 'historical_floor_flat', got {mode!r}"
        )


def dequantize_q2k_bytes_to_f32(
    blob: bytes,
    d_out: int,
    d_in: int,
    *,
    mode: str = "corrected_ceil_per_row",
) -> np.ndarray:
    """
    Dequantize a Q2_K byte blob back to float32 using llama.cpp.

    Two explicit modes (must match the mode used for quantization):

    mode="corrected_ceil_per_row" (default, technically correct):
        Expects d_out * ceil(d_in/QK_K) * 84 bytes.
        Dequantizes each row independently, handling padded final block.

    mode="historical_floor_flat" (legacy compatibility):
        Expects floor(d_out*d_in/QK_K) * 84 bytes.
        Dequantizes as a flat buffer; last partial block elements are ZERO.

    Args:
        blob: Q2_K byte data.
        d_out: number of output rows.
        d_in: number of input columns per row (any value; padded blocks handled).
        mode: "corrected_ceil_per_row" or "historical_floor_flat"

    Returns:
        2D float32 array of shape (d_out, d_in).

    Raises:
        ValueError: if blob size doesn't match expected Q2_K size for the mode.
        RuntimeError: if llama.cpp library or symbols unavailable.
    """
    lib = _load_lib()

    if mode == "corrected_ceil_per_row":
        per_row = ((d_in + QK_K - 1) // QK_K) * Q2_BLOCK_BYTES
        expected = d_out * per_row
        if len(blob) != expected:
            raise ValueError(
                f"blob size {len(blob)} != expected {expected} "
                f"for shape ({d_out}, {d_in}) in mode {mode!r}"
            )
        W = np.zeros(d_out * d_in, dtype=DTYPE_F32)
        for row_idx in range(d_out):
            row_start = row_idx * per_row
            row_end = (row_idx + 1) * per_row
            row_blob = blob[row_start:row_end]
            dst = W[row_idx * d_in:(row_idx + 1) * d_in]
            lib.dequantize_row_q2_K(
                (np.frombuffer(row_blob, dtype=DTYPE_U8)
                   .ctypes.data_as(ctypes.c_void_p)),
                dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                d_in,
            )
        return W.reshape(d_out, d_in)

    elif mode == "historical_floor_flat":
        # Flat buffer: floor(n/QK_K) blocks, each 256 elements
        n_elements = d_out * d_in
        n_blocks = n_elements // QK_K   # floor
        expected = n_blocks * Q2_BLOCK_BYTES
        if len(blob) != expected:
            raise ValueError(
                f"blob size {len(blob)} != expected {expected} "
                f"for shape ({d_out}, {d_in}) in mode {mode!r} "
                f"({n_blocks} blocks)"
            )
        W = np.zeros(n_elements, dtype=DTYPE_F32)
        lib.dequantize_row_q2_K(
            (np.frombuffer(blob, dtype=DTYPE_U8)
               .ctypes.data_as(ctypes.c_void_p)),
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_elements,
        )
        # The last 128 elements of each row are ZERO (not quantized)
        # These zeros contribute nothing to X @ W.T
        return W.reshape(d_out, d_in)

    else:
        raise ValueError(
            f"mode must be 'corrected_ceil_per_row' or 'historical_floor_flat', got {mode!r}"
        )


def is_available() -> bool:
    """Check if llama.cpp Q2_K library is available without raising."""
    try:
        _load_lib()
        return True
    except Exception:
        return False


def availability_report() -> dict:
    """Return a dict describing Q2_K backend availability."""
    try:
        path = _resolve_lib_path()
        _load_lib()
        return {
            "available": True,
            "lib_path": path,
            "symbols": ["quantize_row_q2_K_ref", "dequantize_row_q2_K"],
        }
    except Exception as exc:
        return {
            "available": False,
            "lib_path": None,
            "error": str(exc),
            "symbols": ["quantize_row_q2_K_ref", "dequantize_row_q2_K"],
        }
