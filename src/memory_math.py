#!/usr/bin/env python3
"""
memory_math.py — Memory accounting helpers for residual economics.
"""

import numpy as np
from typing import Dict, Any, Tuple


def bytes_to_bits_per_weight(n_bytes: int, n_weights: int) -> float:
    """Convert compressed bytes to effective bits/weight."""
    return n_bytes * 8.0 / n_weights


def memory_viable_gate(
    W_low_bytes: int,
    R_compressed_bytes: int,
    W_q4_bytes: int,
    target: str = "q2"
) -> Dict[str, Any]:
    """
    Check if W_low + R_compressed beats W_q4.
    
    Args:
        W_low_bytes: bytes for low-precision base (e.g., Q2)
        R_compressed_bytes: bytes for compressed residual
        W_q4_bytes: bytes for Q4 reference
        target: "q2" or "q3" — Q2 needs residual < 2 bits/w, Q3 needs residual < 1 bit/w
    
    Returns dict with viability decision and details.
    """
    total_bytes = W_low_bytes + R_compressed_bytes
    N = W_q4_bytes * 2  # Q4 is 0.5 bytes/weight, so N = bytes * 2
    total_bits_per_weight = total_bytes * 8.0 / N
    R_bits_per_weight = R_compressed_bytes * 8.0 / N
    delta_bytes = total_bytes - W_q4_bytes
    delta_pct = delta_bytes / W_q4_bytes * 100
    
    if target == "q2":
        viable = R_bits_per_weight < 2.0 and total_bytes < W_q4_bytes
        threshold_bpw = 2.0
    elif target == "q3":
        viable = R_bits_per_weight < 1.0 and total_bytes < W_q4_bytes
        threshold_bpw = 1.0
    else:
        viable = total_bytes < W_q4_bytes
        threshold_bpw = 0.0
    
    return {
        "viable": viable,
        "target": target,
        "W_low_bytes": W_low_bytes,
        "R_compressed_bytes": R_compressed_bytes,
        "total_bytes": total_bytes,
        "W_q4_bytes": W_q4_bytes,
        "delta_bytes": delta_bytes,
        "delta_pct": delta_pct,
        "R_bits_per_weight": R_bits_per_weight,
        "total_bits_per_weight": total_bits_per_weight,
        "threshold_bits_per_weight": threshold_bpw,
        "memory_saves": total_bytes < W_q4_bytes,
        "meets_residual_threshold": R_bits_per_weight < threshold_bpw,
    }


def memory_table_row(
    name: str,
    rep_type: str,
    compressed_bytes: int,
    W_low_bytes: int,
    W_q4_bytes: int,
    N: int,
) -> Dict[str, Any]:
    """Build a memory table row for a representation."""
    total = W_low_bytes + compressed_bytes
    R_bpw = compressed_bytes * 8.0 / N
    
    return {
        "name": name,
        "type": rep_type,
        "compressed_bytes": compressed_bytes,
        "W_low_bytes": W_low_bytes,
        "total_bytes": total,
        "W_q4_bytes": W_q4_bytes,
        "delta_vs_q4": total - W_q4_bytes,
        "R_bits_per_weight": round(R_bpw, 4),
        "total_bits_per_weight": round((total * 8.0 / N), 4),
        "memory_viable_q2": R_bpw < 2.0 and total < W_q4_bytes,
        "memory_viable_q3": R_bpw < 1.0 and total < W_q4_bytes,
    }


def compute_memory_summary(N: int, W_q4_bytes: int) -> Dict[str, Dict[str, Any]]:
    """
    Compute reference memory values for N weights.
    Q4: 0.5 bytes/weight (4 bits)
    Q3: 0.375 bytes/weight (3 bits)  
    Q2: 0.25 bytes/weight (2 bits)
    F32: 4 bytes/weight (32 bits)
    """
    return {
        "F32": {
            "bytes": N * 4,
            "bits_per_weight": 32.0,
            "label": "float32 reference",
        },
        "Q4": {
            "bytes": N * 0.5,
            "bits_per_weight": 4.0,
            "label": "Q4_K / IQ4_NL (4-bit)",
            "W_ref_bytes": W_q4_bytes,
        },
        "Q3": {
            "bytes": N * 0.375,
            "bits_per_weight": 3.0,
            "label": "Q3_K (3-bit)",
        },
        "Q2": {
            "bytes": N * 0.25,
            "bits_per_weight": 2.0,
            "label": "Q2_K / IQ2_XXS (2-bit)",
        },
    }


def approximate_reconstruction(
    R_flat: np.ndarray,
    rep_type: str,
    **rep_data
) -> np.ndarray:
    """
    Reconstruct full R tensor from compressed representation.
    This is representation-specific.
    """
    if rep_type.startswith("topk") or rep_type.startswith("magnitude"):
        # Sparse: values at specific indices
        vals = rep_data["R"]
        coords = rep_data["coords"]
        R_reconstructed = np.zeros_like(R_flat)
        R_reconstructed[coords] = vals
        return R_reconstructed
    elif rep_type.startswith("lowrank"):
        # Low-rank: U @ diag(S) @ Vt
        U = rep_data["R_U"]
        S = rep_data["R_S"]
        Vt = rep_data["R_Vt"]
        R_reconstructed = U @ np.diag(S) @ Vt
        return R_reconstructed.flatten()
    else:
        # Dense: already full-size
        return R_flat
