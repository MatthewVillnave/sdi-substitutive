#!/usr/bin/env python3
"""
Runtime-consistent residual helpers for SDI Substitutive.

CRITICAL BUG FIX: Prior residual generation used R = W_ref - W_low_raw,
but the runtime actually computes from decode(packed W_low).
The CORRECT residual must be:
    R_runtime = W_ref - W_low_runtime
where:
    W_low_raw    = q4_quantize(W_ref)           # float32 low-approx (before packing)
    W_low_packed = pack_wlow(W_low_raw)         # nibble-packed artifact bytes
    W_low_runtime = unpack_wlow(W_low_packed)   # decoded at runtime
    R_runtime    = W_ref - W_low_runtime        # THIS is the correct residual

Using R_old = W_ref - W_low_raw would encode a residual that doesn't match
the runtime decode path, causing approximation degradation.
"""

import numpy as np

BLOCK_SIZE = 32


# =============================================================================
# Packed-nibble W_low encoding (same as artifact_write, phase31x, etc.)
# =============================================================================

def pack_wlow(W):
    """
    Pack float32 weight matrix to packed-nibble format.
    Returns (packed_bytes, scales_arr).

    Format: per-block BLOCK_SIZE elements:
      - 1 fp16 scale
      - BLOCK_SIZE/2 bytes of nibble pairs
    """
    rows, cols = W.shape
    n = rows * cols
    nb = n // BLOCK_SIZE
    npacked = (n + 1) // 2
    nscales = nb * 2

    packed_arr = np.zeros(npacked, dtype=np.uint8)
    scales_arr = np.zeros(nscales, dtype=np.float16)

    for b in range(nb):
        block = W.flat[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
        scale = float(np.abs(block).max()) / 7.5
        if scale < 1e-9:
            scale = 1.0
        scales_arr[b] = np.float16(scale)

        q = np.clip(np.round(block / scale), -8.0, 7.0).astype(np.int8)
        q_stored = (q + 8).astype(np.uint8)

        bb = b * 16  # BLOCK_SIZE/2 = 16 bytes per block
        for i in range(BLOCK_SIZE // 2):
            lo = q_stored[2*i] & 0x0F
            hi = q_stored[2*i + 1] & 0x0F
            packed_arr[bb + i] = lo | (hi << 4)

    return packed_arr.tobytes(), scales_arr.tobytes()


def unpack_wlow(packed_bytes, scales_bytes, rows, cols, block_size=BLOCK_SIZE):
    """
    Unpack nibble-encoded W_low from packed bytes + scales.
    Returns float32 W_low matrix.

    Arguments:
        packed_bytes: bytes of packed nibble data
        scales_bytes: bytes of fp16 scale data (or np.ndarray of float16)
    """
    if isinstance(scales_bytes, (bytes, bytearray)):
        scales_arr = np.frombuffer(scales_bytes, dtype=np.float16)
    else:
        scales_arr = np.asarray(scales_bytes)

    n = rows * cols
    nb = n // block_size

    W = np.zeros((rows, cols), dtype=np.float32)
    for b in range(nb):
        scale = float(scales_arr[b])
        bb = b * (block_size // 2)
        for i in range(block_size // 2):
            byte = packed_bytes[bb + i]
            lo = (byte & 0x0F)
            hi = (byte >> 4) & 0x0F
            W.flat[b*block_size + 2*i + 0] = (float(lo) - 8.0) * scale
            W.flat[b*block_size + 2*i + 1] = (float(hi) - 8.0) * scale

    return W


# =============================================================================
# CORE HELPER: Runtime-consistent residual
# =============================================================================

def make_runtime_consistent_residual(W_ref, W_low_raw=None, packed_wlow_artifact=None,
                                      scales_artifact=None, rows=None, cols=None):
    """
    Compute the CORRECT runtime-consistent residual.

    The residual must be computed against the DECODED W_low (what runtime uses),
    NOT the pre-packing W_low_raw. This ensures the residual exactly compensates
    for the quantization error that the runtime sees.

    Arguments (mutually exclusive):
        Option A (artifact path):
            packed_wlow_artifact: bytes of packed W_low nibble data
            scales_artifact:      bytes of fp16 scale data
            rows, cols:          weight matrix dimensions

        Option B (raw path):
            W_low_raw:           float32 W_low (pre-quantization approximation)

    Returns:
        {
            "R_runtime": np.ndarray (float32 residual),
            "W_low_runtime": np.ndarray (decoded W_low, same as runtime uses),
            "W_low_raw": np.ndarray or None,
            "packed_bytes": bytes or None,
            "scales_bytes": bytes or None,
            "pack_ratio": bytes/element,
        }

    Usage:
        result = make_runtime_consistent_residual(
            W_ref=W_ref,
            packed_wlow_artifact=packed_bytes,
            scales_artifact=scales,
            rows=rows, cols=cols
        )
        R = result["R_runtime"]  # USE THIS for sdir encoding
    """
    if packed_wlow_artifact is not None:
        # Artifact path: decode to get runtime W_low
        W_low_runtime = unpack_wlow(packed_wlow_artifact, scales_artifact, rows, cols)
    elif W_low_raw is not None:
        # Raw path: pack then unpack to get runtime W_low
        packed_bytes, scales_bytes = pack_wlow(W_low_raw)
        W_low_runtime = unpack_wlow(packed_bytes, scales_bytes,
                                     W_low_raw.shape[0], W_low_raw.shape[1])
        packed_wlow_artifact = packed_bytes
        scales_artifact = scales_bytes
    else:
        raise ValueError("Must provide either packed_wlow_artifact or W_low_raw")

    R_runtime = W_ref - W_low_runtime

    return {
        "R_runtime": R_runtime,
        "W_low_runtime": W_low_runtime,
        "W_low_raw": W_low_raw,
        "packed_bytes": packed_wlow_artifact,
        "scales_bytes": scales_artifact,
        "pack_ratio": len(packed_wlow_artifact) / W_ref.size if W_ref.size > 0 else 0,
    }


# =============================================================================
# Legacy compatibility: given W_ref and W_low_raw, compute R_old for comparison
# =============================================================================

def make_legacy_residual(W_ref, W_low_raw):
    """
    Compute the OLD (buggy) residual: R_old = W_ref - W_low_raw.
    Use only for regression testing (comparing R_old vs R_runtime).
    """
    return W_ref - W_low_raw


# =============================================================================
# Quantization helpers
# =============================================================================

def q4_quantize_blocked(W_ref, block_size=BLOCK_SIZE):
    """
    Quantize W_ref to W_low_raw (float32, same shape as W_ref).
    This is the pre-packing approximation — does NOT yet encode to nibbles.
    """
    rows, cols = W_ref.shape
    W_low = np.zeros_like(W_ref)
    n = rows * cols
    nb = n // block_size

    for b in range(nb):
        s = b * block_size
        e = s + block_size
        block = W_ref.flat[s:e]
        scale = float(np.abs(block).max()) / 7.5
        if scale < 1e-8:
            scale = 1.0
        q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        W_low.flat[s:e] = q * scale

    return W_low


# =============================================================================
# Verification utility
# =============================================================================

def verify_residual_consistency(W_ref, W_low_raw):
    """
    Prove that the runtime-consistent residual is different from the legacy residual.
    Returns comparison dict showing the magnitude of the bug.

    R_old  = W_ref - W_low_raw         (WRONG: ignores pack/unpack rounding)
    R_rt   = W_ref - decode(pack(W_low_raw))  (CORRECT: matches runtime)

    Y_old = X @ W_low_raw + X @ R_old
    Y_rt  = X @ W_low_rt + X @ R_rt

    Y_rt should match Y_ref significantly better than Y_old.
    """
    packed_bytes, scales_bytes = pack_wlow(W_low_raw)
    W_low_rt = unpack_wlow(packed_bytes, scales_bytes,
                            W_low_raw.shape[0], W_low_raw.shape[1])

    R_old = W_ref - W_low_raw
    R_rt = W_ref - W_low_rt

    return {
        "W_low_raw_norm": float(np.linalg.norm(W_low_raw)),
        "W_low_rt_norm": float(np.linalg.norm(W_low_rt)),
        "W_low_rt_vs_raw_max_diff": float(np.abs(W_low_rt - W_low_raw).max()),
        "W_low_rt_vs_raw_mean_diff": float(np.abs(W_low_rt - W_low_raw).mean()),
        "R_old_norm": float(np.linalg.norm(R_old)),
        "R_rt_norm": float(np.linalg.norm(R_rt)),
        "R_rt_vs_R_old_max_diff": float(np.abs(R_rt - R_old).max()),
        "R_rt_vs_R_old_mean_diff": float(np.abs(R_rt - R_old).mean()),
    }


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    W_ref = np.random.randn(896, 4864).astype(np.float32) * 0.1

    W_low_raw = q4_quantize_blocked(W_ref)
    result = make_runtime_consistent_residual(W_ref=W_ref, W_low_raw=W_low_raw)

    print("Runtime-consistent residual helper: SMOKE TEST")
    print(f"  W_ref shape: {W_ref.shape}")
    print(f"  W_low_runtime norm: {np.linalg.norm(result['W_low_runtime']):.4f}")
    print(f"  W_low_raw norm:     {np.linalg.norm(result['W_low_raw']):.4f}")
    print(f"  R_runtime norm:     {np.linalg.norm(result['R_runtime']):.4f}")
    print(f"  R_runtime mean abs: {np.abs(result['R_runtime']).mean():.6f}")
    print(f"  R_runtime max abs:  {np.abs(result['R_runtime']).max():.6f}")
    print(f"  Pack ratio: {result['pack_ratio']:.4f} bytes/element")

    # Prove the bug
    verify = verify_residual_consistency(W_ref, W_low_raw)
    print(f"\nBug magnitude:")
    print(f"  W_low_rt vs W_low_raw max diff: {verify['W_low_rt_vs_raw_max_diff']:.6f}")
    print(f"  R_rt vs R_old max diff:          {verify['R_rt_vs_R_old_max_diff']:.6f}")
    print(f"  R_rt vs R_old mean diff:         {verify['R_rt_vs_R_old_mean_diff']:.6f}")