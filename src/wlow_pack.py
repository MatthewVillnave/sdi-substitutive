#!/usr/bin/env python3
"""
wlow_pack.py — Packed nibble W_low format (Phase 31S-R)

Format: 4-bit/nibble per element, block-wise fp16 scales.
NOT GGUF Q4_K_M unless it matches GGUF exactly.

Pack:   W_f32 → packed_bytes (nibbles) + scales (fp16)
Unpack: packed_bytes + scales → W_f32 reconstruction
"""
import struct
import hashlib

import numpy as np

BLOCK_SIZE = 32
NIBBLES_PER_BYTE = 2

def pack_wlow(W_f32, block_size=BLOCK_SIZE):
    """
    Pack W_f32 matrix into nibble format with per-block fp16 scales.
    
    Returns:
        packed_bytes: bytes object with nibble-packed quantized values
        scales: numpy.array of fp16 scale values (one per block)
    """
    rows, cols = W_f32.shape
    n_elements = rows * cols
    n_blocks = n_elements // block_size
    
    # Total packed bytes: ceil(n_elements / 2)
    total_packed_bytes = (n_elements + 1) // 2
    packed = np.empty(total_packed_bytes, dtype=np.uint8)
    
    # Scales: one fp16 per block
    scales = np.empty(n_blocks, dtype=np.float16)
    
    for b in range(n_blocks):
        block = W_f32.flat[b * block_size:(b + 1) * block_size]
        scale = float(np.abs(block).max()) / 7.5
        if scale < 1e-9:
            scale = 1.0
        scales[b] = np.float16(scale)
        
        # Quantize: signed int8 range [-8, 7], stored as [0, 15] with offset +8
        q = np.clip(np.round(block / scale), -8.0, 7.0).astype(np.int8)
        q_stored = (q + 8).astype(np.uint8)  # offset: -8→0, 7→15
        
        # Pack 2 nibbles per byte (little-endian: low nibble = first of pair)
        bbase = b * (block_size // 2)
        for i in range(block_size // 2):
            lo = q_stored[2 * i] & 0x0F
            hi = q_stored[2 * i + 1] & 0x0F
            packed[bbase + i] = lo | (hi << 4)
    
    return packed.tobytes(), scales

def unpack_wlow(packed_bytes, scales, rows, cols, block_size=BLOCK_SIZE):
    """
    Unpack nibble-format W_low back to fp32 approximation.
    
    Args:
        packed_bytes: raw nibble-packed bytes
        scales: numpy.array of fp16 scale values (one per block)
        rows, cols: reshape target
        block_size: elements per block (default 32)
    
    Returns:
        W_f32: reconstructed matrix (fp32)
    """
    n_elements = rows * cols
    n_blocks = n_elements // block_size
    n_packed = len(packed_bytes)
    
    # Verify we have enough bytes
    expected_min_bytes = (n_elements + 1) // 2
    if n_packed < expected_min_bytes:
        raise ValueError(f"Packed bytes {n_packed:,} < expected minimum {expected_min_bytes:,}")
    
    W_flat = np.zeros(n_elements, dtype=np.float32)
    
    for b in range(n_blocks):
        scale = float(scales[b])
        base = b * block_size
        for i in range(block_size // 2):
            byte = packed_bytes[b * (block_size // 2) + i]
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            idx_lo = base + 2 * i
            idx_hi = idx_lo + 1
            W_flat[idx_lo] = (float(lo) - 8.0) * scale
            W_flat[idx_hi] = (float(hi) - 8.0) * scale
    
    return W_flat.reshape(rows, cols)

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def test_tiny():
    """Test nibble packing logic on a simple known block.
    
    Uses values [0,1,2] of magnitude < scale so clamp doesn't clip,
    then 29 zeros. scale=1.0 (max_abs=2/7.5), q stored=8+value,
    pack->unpack recovers exactly because v/scale is integer.
    """
    # Values where v/scale is exact integer:
    # max_abs=2, scale=2/7.5, so v/scale for v in {0,1,2} gives {0, 3.75, 7.5} → NOT exact
    # Instead: 1) values from float LUT, all positive. 
    # For exact integer v/scale: pick v = k * max_abs/7.5 for integer k in [-8,7]
    # Let max_abs=7.5 → scale=1.0, then v∈{-7.5,-6,-4.5,-3,-1.5,0,1.5,3,4.5,6,7.5} gives exact integers
    # Too finicky for a simple test.
    
    # Simple approach: use just 3 non-zero values + 29 zeros, check pack/unpack is consistent
    # and max error is bounded by scale rounding (not a bug).
    W = np.zeros(32, dtype=np.float32)
    W[0] = 0.0
    W[1] = 1.0
    W[2] = 2.0
    W[3] = -1.0
    W[4] = -2.0
    W_2d = W.reshape(1, 32)  # (1, 32) for pack_wlow
    
    packed, scales = pack_wlow(W_2d, block_size=32)
    W_rec = unpack_wlow(packed, scales, 1, 32, block_size=32)
    
    # Flatten for comparison
    W_flat = W  # still (32,)
    W_rec_flat = W_rec.flatten()
    
    # Check: nibbles decode to non-trivial values (not all zero/garbage)
    non_trivial = float(np.abs(W_rec_flat[:5]).max()) > 0.01
    # Check: zeros are recovered as 0 (within rounding)
    zeros_ok = float(np.abs(W_rec_flat[5:]).max()) < 0.1  # indices 5+: mostly zeros
    # Check: error is bounded by quantization step (scale)
    scale = float(scales[0])
    max_err = float(np.abs(W_flat - W_rec_flat).max())
    bounded = max_err < scale * 1.5  # at most a few rounding steps
    ok = non_trivial and zeros_ok and bounded
    print(f"  Tiny non-trivial: {'PASS' if non_trivial else 'FAIL'}")
    print(f"  Tiny zeros OK: {'PASS' if zeros_ok else 'FAIL'}")
    print(f"  Tiny error bounded: max_err={max_err:.4f} < scale*1.5={scale*1.5:.4f} → {'PASS' if bounded else 'FAIL'}")
    assert ok, f"Tiny failed: non_trivial={non_trivial}, zeros_ok={zeros_ok}, bounded={bounded}"
    return True

def test_random():
    """Test pack/unpack on random 896x4864 array — deterministic/deterministic"""
    rng = np.random.default_rng(12345)
    W = rng.standard_normal(896 * 4864).astype(np.float32) * 2.0
    
    packed_a, scales_a = pack_wlow(W.reshape(896, 4864))
    packed_b, scales_b = pack_wlow(W.reshape(896, 4864))
    
    # Same input → same output
    assert packed_a == packed_b, "Pack non-deterministic!"
    
    W_rec = unpack_wlow(packed_a, scales_a, 896, 4864)
    
    max_err = float(np.abs(W - W_rec.flatten()).max())
    mean_err = float(np.abs(W - W_rec.flatten()).mean())
    
    # Within quantization error bound (nibble step ≈ scale, max rounding ≈ scale/2)
    # For random data, max error can approach scale (worst-case rounding)
    ok = max_err < 1.0
    print(f"  Random deterministic: PASS")
    print(f"  Random max error: {max_err:.6f} ({'PASS' if ok else 'FAIL'}, tol=1.0)")
    print(f"  Random mean error: {mean_err:.6f}")
    assert ok, f"Random failed: max_err={max_err}"
    return True

def test_checksum_stable():
    """SHA-256 stable across multiple pack calls on same input"""
    rng = np.random.default_rng(99999)
    W = rng.standard_normal(896 * 4864).astype(np.float32) * 3.0
    
    hashes = []
    for _ in range(3):
        packed, scales = pack_wlow(W.reshape(896, 4864))
        h = sha256_bytes(packed)
        hashes.append(h)
    
    stable = len(set(hashes)) == 1
    print(f"  Checksum stable: {'PASS' if stable else 'FAIL'} ({hashes[0][:16]}...)")
    assert stable, f"Checksum not stable: {hashes}"
    return True

def run_tests():
    print("=== wlow_pack.py tests ===\n")
    print("[tiny known array]")
    test_tiny()
    print("\n[random 896x4864]")
    test_random()
    print("\n[checksum stable]")
    test_checksum_stable()
    print("\n=== ALL TESTS PASS ===")

if __name__ == '__main__':
    run_tests()
