#!/usr/bin/env python3
"""
residual_encode.py — Encoder for compressed residual format.

Format (dense_bitmap + top-k% + fp16 values):
  Header (binary, 32 bytes):
    magic: 4 bytes    = b'RSC\x00'
    version: 2 bytes   = 1
    flags: 2 bytes    = reserved (0)
    rows: 4 bytes     = u32
    cols: 4 bytes     = u32
    k_pct: 4 bytes    = u32 (percentage * 100, e.g. 750 = 7.5%)
    nnz: 4 bytes      = u32 (number of set bits)
    value_dtype: 2 bytes = 0 (0=fp16, 1=fp32, 2=int8)
    mask_encoding: 2 bytes = 0 (0=dense_bitmap)
    header_total: 4 bytes = total header size (for forward compat)

  Body:
    bitmap: (rows * cols + 7) // 8 bytes, row-major, LSB first
    values: nnz * 2 bytes, fp16, row-major traversal of set bits

Usage:
  from residual_encode import encode_residual
  enc = encode_residual(R_f32, k_pct=7.5)
  enc.save("tensor_r7p5.bin")
"""

import struct
import io
import numpy as np
from typing import Tuple, Dict, Any


MAGIC = b'RSC\x00'
VERSION = 1
HEADER_SIZE = 32  # fixed 32-byte header
MASK_ENCODING_DENSE_BITMAP = 0
VALUE_DTYPE_FP16 = 0
VALUE_DTYPE_FP32 = 1
VALUE_DTYPE_INT8 = 2


class EncodedResidual:
    """
    In-memory encoded residual: dense bitmap + top-k% + fp16 values.
    Does NOT store full dense R_f32 — only sparse representation.
    """
    def __init__(self, rows: int, cols: int, k_pct: float, bitmap: np.ndarray,
                 values: np.ndarray, nnz: int):
        self.magic = MAGIC
        self.version = VERSION
        self.rows = rows
        self.cols = cols
        self.k_pct = k_pct
        self.bitmap = bitmap  # uint8 array, 1 bit per element, row-major
        self.values = values   # uint16 array (fp16), length = nnz
        self.nnz = nnz
        self.shape = (rows, cols)

    @property
    def n_elements(self) -> int:
        return self.rows * self.cols

    @property
    def bitmap_bytes(self) -> int:
        return len(self.bitmap)

    @property
    def value_bytes(self) -> int:
        return len(self.values) * 2  # fp16 = 2 bytes

    @property
    def total_bytes(self) -> int:
        return HEADER_SIZE + self.bitmap_bytes + self.value_bytes

    @property
    def effective_bits_per_weight(self) -> float:
        return self.total_bytes * 8.0 / self.n_elements

    def save(self, path: str):
        """Write encoded residual to binary file."""
        with open(path, "wb") as f:
            # Header
            k_pct_int = int(round(self.k_pct * 100))  # e.g. 7.5 -> 750
            header = struct.pack(
                "<4sHHIIIHHI",
                MAGIC,          # 4s
                VERSION,        # H
                0,              # flags (H)
                self.rows,      # I
                self.cols,      # I
                k_pct_int,      # I (k_pct * 100)
                self.nnz,       # I
                VALUE_DTYPE_FP16, # H
                MASK_ENCODING_DENSE_BITMAP, # H
                HEADER_SIZE,    # I (header_total)
            )
            assert len(header) == HEADER_SIZE, f"Header size {len(header)} != {HEADER_SIZE}"
            f.write(header)

            # Bitmap
            f.write(self.bitmap.tobytes())

            # Values (fp16, already uint16)
            f.write(self.values.tobytes())

    @classmethod
    def load(cls, path: str) -> "EncodedResidual":
        """Load encoded residual from binary file."""
        with open(path, "rb") as f:
            # Header
            header = f.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                raise ValueError(f"Header too short: {len(header)} < {HEADER_SIZE}")
            (magic, version, flags, rows, cols, k_pct_int,
             nnz, value_dtype, mask_encoding, header_total) = struct.unpack(
                "<4sHHIIIHHI", header)

            if magic != MAGIC:
                raise ValueError(f"Bad magic: {magic!r}")
            if version != VERSION:
                raise ValueError(f"Unsupported version: {version}")

            k_pct = k_pct_int / 100.0

            # Bitmap
            bitmap_nbytes = (rows * cols + 7) // 8
            bitmap_data = f.read(bitmap_nbytes)
            if len(bitmap_data) < bitmap_nbytes:
                raise ValueError(f"Bitmap too short: {len(bitmap_data)} < {bitmap_nbytes}")
            bitmap = np.frombuffer(bitmap_data, dtype=np.uint8).copy()

            # Values
            values_data = f.read(nnz * 2)
            if len(values_data) < nnz * 2:
                raise ValueError(f"Values too short: {len(values_data)} < {nnz * 2}")
            values = np.frombuffer(values_data, dtype=np.uint16).copy()

        return cls(rows, cols, k_pct, bitmap, values, nnz)

    def decode_to_dense(self) -> np.ndarray:
        """
        Decode to full dense R_f32 (for correctness reference only).
        This materializes the full tensor — use only for validation.
        """
        R = np.zeros(self.n_elements, dtype=np.float32)
        flat_bitmap = np.unpackbits(self.bitmap, count=self.n_elements)

        # Extract fp16 values in row-major order
        vals_f16 = self.values.view(np.float16)

        set_indices = np.where(flat_bitmap)[0]
        if len(set_indices) != self.nnz:
            raise ValueError(f"nnz mismatch: bitmap has {len(set_indices)} set, expected {self.nnz}")

        for i, idx in enumerate(set_indices):
            R[idx] = float(vals_f16[i])

        return R.reshape(self.rows, self.cols)

    def streaming_sparse_apply(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Y_delta = X @ R_sparse WITHOUT materializing dense R.
        X: (batch, rows) or (rows,)  — uses cols as feature dim
        R is (rows, cols)
        Returns: (batch, cols) or (cols,) — Y_delta
        """
        # Determine X shape
        if X.ndim == 1:
            X = X[np.newaxis, :]  # (1, rows)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = X.shape[0]
        rows = self.rows
        cols = self.cols

        # Decode fp16 values
        vals_f16 = self.values.view(np.float16)  # (nnz,)

        # Unpack bitmap to find which elements are kept
        flat_bitmap = np.unpackbits(self.bitmap, count=self.n_elements)

        # Get indices of set bits (row-major order)
        set_indices = np.where(flat_bitmap)[0]  # sorted due to row-major packing

        # Compute Y_delta via sparse matmul
        # Each non-zero in R contributes: for col j, row i, value v:
        #   Y_delta[b, j] += X[b, i] * v
        # We iterate row by row to avoid sorting indices
        Y_delta = np.zeros((batch_size, cols), dtype=np.float32)

        for i in range(rows):
            row_start = i * cols
            row_bitmap = flat_bitmap[row_start : row_start + cols]
            row_set = np.where(row_bitmap)[0]  # col indices where bit=1

            if len(row_set) == 0:
                continue

            # Global indices for this row
            global_indices = row_start + row_set

            # Find which of these global indices are in our nnz list
            # We need to map global index -> value position
            # set_indices is sorted row-major, same order as vals_f16
            # Use set membership
            set_set = set(set_indices)
            vals_for_row = []
            cols_for_row = []

            for j, gi in enumerate(global_indices):
                if gi in set_set:
                    # Find position of gi in set_indices (use bisect)
                    # For speed, build a position map
                    pass

            # Faster: build index->value_position map using numpy
            # Create a mask array of size n_elements marking positions in set_indices
            mask_positions = np.zeros(self.n_elements, dtype=np.int32)
            for j, si in enumerate(set_indices):
                mask_positions[si] = j + 1  # 1-indexed, 0 = not in set

            val_positions = mask_positions[global_indices]
            valid_mask = val_positions > 0
            valid_positions = val_positions[valid_mask] - 1  # back to 0-indexed
            valid_cols = row_set[valid_mask]

            if len(valid_positions) > 0:
                v = vals_f16[valid_positions].astype(np.float32)
                x_col = X[:, i]  # (batch,)
                Y_delta[:, valid_cols] += np.outer(x_col, v)

        if squeeze_output:
            return Y_delta[0]
        return Y_delta

    def __repr__(self):
        return (f"EncodedResidual(shape={self.shape}, k_pct={self.k_pct}, "
                f"nnz={self.nnz}, total_bytes={self.total_bytes}, "
                f"ebpw={self.effective_bits_per_weight:.4f})")


def encode_residual(R_f32: np.ndarray, k_pct: float = 7.5,
                    global_topk: bool = True) -> EncodedResidual:
    """
    Encode residual tensor R_f32 as: dense bitmap + top-k% + fp16 values.

    Args:
        R_f32: Residual tensor, shape (rows, cols), float32
        k_pct: Percentage of elements to keep (0-100). Default 7.5.
        global_topk: If True, select top-k% globally across entire tensor.
                     If False, per-row (not implemented here).
    Returns:
        EncodedResidual object (in-memory, not yet saved)
    """
    if R_f32.dtype != np.float32:
        R_f32 = R_f32.astype(np.float32)

    rows, cols = R_f32.shape
    n_elements = rows * cols
    nnz = max(1, int(n_elements * k_pct / 100.0))

    flat = R_f32.flatten()

    if global_topk:
        # Select top-k% globally by magnitude
        abs_flat = np.abs(flat)
        threshold = np.partition(abs_flat, -nnz)[-nnz]
        # Use > for elements strictly above threshold
        mask_flat = abs_flat > threshold
        # Handle ties at threshold: count how many equal threshold, cap excess
        n_above = np.sum(mask_flat)
        if n_above > nnz:
            # Too many elements at exact threshold — need to select exactly nnz
            # Pick first (nnz - n_above) from the tied group
            at_threshold = abs_flat == threshold
            n_excess = n_above - nnz
            if n_excess > 0:
                # Set n_excess of the at-threshold elements to False
                # Use stable ordering: pick first n_excess indices
                threshold_indices = np.where(at_threshold)[0]
                drop_indices = threshold_indices[:n_excess]
                mask_flat[drop_indices] = False
        elif n_above < nnz:
            # Rare: threshold partition picked fewer — include threshold elements to reach nnz
            at_threshold = abs_flat == threshold
            threshold_indices = np.where(at_threshold)[0]
            n_to_add = nnz - n_above
            add_indices = threshold_indices[:n_to_add]
            mask_flat[add_indices] = True
    else:
        raise NotImplementedError("Per-row topk not implemented in this version")

    # Build bitmap (1 bit per element, row-major, LSB first)
    bitmap_bits = mask_flat.astype(np.uint8)  # 0 or 1
    bitmap = np.packbits(bitmap_bits)  # reduces size by 8x

    # Extract values in row-major order corresponding to set bits
    set_indices = np.where(mask_flat)[0]
    values_f16 = flat[set_indices].astype(np.float16).view(np.uint16)

    return EncodedResidual(
        rows=rows,
        cols=cols,
        k_pct=k_pct,
        bitmap=bitmap,
        values=values_f16,
        nnz=nnz,
    )


def encode_residual_to_file(R_f32: np.ndarray, path: str, k_pct: float = 7.5):
    """Encode and save in one call."""
    enc = encode_residual(R_f32, k_pct=k_pct)
    enc.save(path)
    return enc


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Encode residual tensor to compressed format")
    parser.add_argument("--input", "-i", required=True, help="Path to input R_f32 .npy file")
    parser.add_argument("--output", "-o", required=True, help="Output .bin path")
    parser.add_argument("--k-pct", type=float, default=7.5, help="k percentage (default 7.5)")
    args = parser.parse_args()

    R = np.load(args.input)
    print(f"Loaded R: shape={R.shape}, dtype={R.dtype}")
    enc = encode_residual(R, k_pct=args.k_pct)
    enc.save(args.output)
    print(f"Saved: {enc}")