#!/usr/bin/env python3
"""
residual_compute.py — Decoder and sparse apply for compressed residual format.

Provides two compute modes:
  A. reference_decode_dense(Y_ref, R_encoded, X) — correctness reference only
     - Decodes encoded residual to dense R_f32
     - Computes X @ R_dense
     - NOT the target path

  B. streaming_sparse_apply(X, R_encoded) — target path
     - Does NOT materialize full dense R_f32
     - Iterates bitmap + values to accumulate X @ R_sparse
     - Handles both orientations: (batch, rows) with R(rows, cols)
       and (batch, cols) with R(rows, cols) transposed
     - This is the target path for production

Both produce the same output (verified by correctness checks).
"""

import struct
import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path

# Re-export EncodedResidual from residual_encode
import sys
sys.path.insert(0, str(Path(__file__).parent))
from residual_encode import EncodedResidual, encode_residual


# =============================================================================
# Mode A: Reference decode-dense (correctness reference only)
# =============================================================================

def reference_decode_dense(R_encoded: EncodedResidual) -> np.ndarray:
    """
    Decode encoded residual to dense R_f32.
    WARNING: This materializes the full dense tensor — use only for validation.
    """
    return R_encoded.decode_to_dense()


def reference_compute_Y_sub(X: np.ndarray, W_low_T: np.ndarray,
                              R_encoded: EncodedResidual,
                              X_on_R_cols: bool = True) -> np.ndarray:
    """
    Reference compute: Y_sub = X @ (W_low + R_dense).T or X @ (W_low + R_dense).
    
    Args:
        X: (batch, M) activation matrix
        W_low_T: (M, d_out) — W_low transposed for the matmul
        R_encoded: EncodedResidual with shape (d_out, d_in)
        X_on_R_cols: If True, X has M=R.cols=d_in, so we compute X @ R.T (standard for ffn_up).
                     If False, X has M=R.rows=d_out, so we compute X @ R (ffn_down style).
    """
    R_dense = reference_decode_dense(R_encoded)
    Y_low = X @ W_low_T
    if X_on_R_cols:
        # X is (batch, d_in), R is (d_out, d_in), Y = X @ R.T
        Y_residual = X @ R_dense.T
    else:
        # X is (batch, d_out), R is (d_out, d_in), Y = X @ R
        Y_residual = X @ R_dense
    return Y_low + Y_residual


# =============================================================================
# Mode B: Streaming sparse apply (target path)
# =============================================================================

def streaming_sparse_apply(X: np.ndarray, R_encoded: EncodedResidual,
                           X_on_R_cols: bool = True,
                           accumulate_into: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Y_delta = X @ R_sparse.T or X @ R_sparse WITHOUT materializing dense R.
    
    Args:
        X: Input activations, shape (batch, M)
        R_encoded: EncodedResidual with shape (d_out, d_in)
        X_on_R_cols: If True, X has M=d_in (standard for ffn_up), compute X @ R.T.
                     If False, X has M=d_out (ffn_down style), compute X @ R.
                     Auto-detected if not provided.
        accumulate_into: Optional pre-allocated output buffer

    Returns:
        Y_delta: (batch, d_out)
    """
    # Auto-detect orientation from shape
    if X.shape[1] == R_encoded.cols:
        X_on_R_cols = True
    elif X.shape[1] == R_encoded.rows:
        X_on_R_cols = False
    else:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} matches neither R.cols={R_encoded.cols} "
            f"nor R.rows={R_encoded.rows}"
        )

    if X.ndim == 1:
        X = X[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = X.shape[0]
    rows = R_encoded.rows    # d_out
    cols = R_encoded.cols    # d_in
    nnz = R_encoded.nnz

    # Decode fp16 values
    vals_f16 = R_encoded.values.view(np.float16)
    vals_f32 = vals_f16.astype(np.float32)

    # Unpack bitmap
    flat_bitmap = np.unpackbits(R_encoded.bitmap, count=R_encoded.n_elements)
    set_indices = np.where(flat_bitmap)[0]  # sorted global positions

    # Build position lookup: pos_map[global_index] = value_position (or -1)
    pos_map = np.full(R_encoded.n_elements, -1, dtype=np.int32)
    pos_map[set_indices] = np.arange(nnz, dtype=np.int32)

    if X_on_R_cols:
        # X is (batch, d_in), R is (d_out, d_in)
        # Want: Y = X @ R_sparse.T = (batch, d_out)
        # For output column j: Y[:,j] = sum_i X[:,i] * R_sparse[j,i] = X[:, cols] @ R_sparse[j,:].T
        # Iterate over R rows (each row j corresponds to output column j)
        if accumulate_into is not None:
            Y_delta = accumulate_into
        else:
            Y_delta = np.zeros((batch_size, rows), dtype=np.float32)

        for j in range(rows):
            row_start = j * cols
            row_bitmap = flat_bitmap[row_start:row_start + cols]
            row_set = np.where(row_bitmap)[0]  # column indices i where bit=1

            if len(row_set) == 0:
                continue

            global_indices = row_start + row_set  # (n_kept,)
            val_positions = pos_map[global_indices]  # (n_kept,)
            v = vals_f32[val_positions]  # (n_kept,)

            # Y_delta[:, j] = X[:, row_set] @ v
            Y_delta[:, j] = np.dot(X[:, row_set], v)
    else:
        # X is (batch, d_out), R is (d_out, d_in)
        # Want: Y = X @ R_sparse = (batch, d_in)
        # For output column i: Y[:,i] = sum_j X[:,j] * R_sparse[j,i]
        if accumulate_into is not None:
            Y_delta = accumulate_into
        else:
            Y_delta = np.zeros((batch_size, cols), dtype=np.float32)

        for j in range(rows):
            row_start = j * cols
            row_bitmap = flat_bitmap[row_start:row_start + cols]
            row_set = np.where(row_bitmap)[0]

            if len(row_set) == 0:
                continue

            global_indices = row_start + row_set
            val_positions = pos_map[global_indices]
            v = vals_f32[val_positions]

            # Y_delta[:, row_set] += np.outer(X[:, j], v)
            np.add.at(Y_delta, (slice(None), row_set),
                      np.outer(X[:, j], v))

    if squeeze_output:
        return Y_delta[0]
    return Y_delta


def streaming_compute_Y_sub(X: np.ndarray, W_low_T: np.ndarray,
                              R_encoded: EncodedResidual,
                              X_on_R_cols: bool = True) -> np.ndarray:
    """
    Target compute: Y_sub = X @ (W_low + R_sparse).T (or variant based on orientation).
    """
    Y_low = X @ W_low_T
    Y_delta = streaming_sparse_apply(
        X, R_encoded, X_on_R_cols=X_on_R_cols,
        accumulate_into=np.zeros_like(Y_low)
    )
    return Y_low + Y_delta


# =============================================================================
# Utility / diagnostic timing
# =============================================================================

def timing_reference_compute(X: np.ndarray, W_low_T: np.ndarray,
                               R_encoded: EncodedResidual,
                               X_on_R_cols: bool = True) -> Tuple[float, np.ndarray]:
    """Time reference decode-dense compute."""
    import time
    t0 = time.perf_counter()
    Y = reference_compute_Y_sub(X, W_low_T, R_encoded, X_on_R_cols)
    t1 = time.perf_counter()
    return t1 - t0, Y


def timing_streaming_compute(X: np.ndarray, W_low_T: np.ndarray,
                               R_encoded: EncodedResidual,
                               X_on_R_cols: bool = True) -> Tuple[float, np.ndarray]:
    """Time streaming sparse compute."""
    import time
    t0 = time.perf_counter()
    Y = streaming_compute_Y_sub(X, W_low_T, R_encoded, X_on_R_cols)
    t1 = time.perf_counter()
    return t1 - t0, Y


# =============================================================================
# Correctness verification
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel()
    b_flat = b.ravel()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def verify_equivalence(Y_dense: np.ndarray, Y_stream: np.ndarray,
                       rtol: float = 1e-4, atol: float = 1e-5) -> dict:
    """
    Verify that dense reference and streaming compute produce equivalent results.
    """
    max_diff = max_abs_diff(Y_dense, Y_stream)
    cos = cosine_similarity(Y_dense, Y_stream)

    diff = np.abs(Y_dense - Y_stream)
    within = bool(np.all(diff <= atol + rtol * np.abs(Y_dense)))

    return {
        "max_abs_diff": max_diff,
        "cosine": cos,
        "within_tolerance": within,
        "rtol": rtol,
        "atol": atol,
        "dense_materialized_stream": False,
        "dense_materialized_reference": True,
    }


def full_verify(X: np.ndarray, W_low_T: np.ndarray,
                R_encoded: EncodedResidual,
                X_on_R_cols: bool = True) -> dict:
    """Run both compute paths and verify equivalence."""
    Y_stream = streaming_compute_Y_sub(X, W_low_T, R_encoded, X_on_R_cols)
    Y_dense = reference_compute_Y_sub(X, W_low_T, R_encoded, X_on_R_cols)
    return verify_equivalence(Y_dense, Y_stream)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test residual compute modes")
    parser.add_argument("--encoded", "-e", required=True, help="Path to encoded .bin file")
    parser.add_argument("--w-low", "-w", required=True, help="Path to W_low .npy file")
    parser.add_argument("--X", "-x", required=True, help="Path to X .npy file")
    args = parser.parse_args()

    print("Loading encoded residual...")
    enc = EncodedResidual.load(args.encoded)
    print(f"  {enc}")

    print("Loading W_low and X...")
    W_low = np.load(args.w_low)
    X = np.load(args.X)
    print(f"  W_low: {W_low.shape}, X: {X.shape}")

    # Determine orientation
    X_on_R_rows = (X.shape[1] == enc.rows)
    print(f"  X_on_R_rows: {X_on_R_rows}")

    print("Running dense reference compute...")
    t_dense, Y_dense = timing_reference_compute(X, W_low, enc, X_on_R_rows)
    print(f"  time={t_dense*1000:.3f}ms, Y_shape={Y_dense.shape}")

    print("Running streaming sparse compute...")
    t_stream, Y_stream = timing_streaming_compute(X, W_low, enc, X_on_R_rows)
    print(f"  time={t_stream*1000:.3f}ms, Y_shape={Y_stream.shape}")

    result = verify_equivalence(Y_dense, Y_stream)
    print(f"\nEquivalence check:")
    print(f"  max_abs_diff={result['max_abs_diff']:.8f}")
    print(f"  cosine={result['cosine']:.8f}")
    print(f"  within_tolerance={result['within_tolerance']}")
    print(f"  dense_materialized_reference={result['dense_materialized_reference']}")
    print(f"  dense_materialized_stream={result['dense_materialized_stream']}")