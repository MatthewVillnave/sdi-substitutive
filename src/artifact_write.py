#!/usr/bin/env python3
"""
Artifact writer for Phase 31S-R — generates packed nibble W_low + residual artifact fixture.

W_low format: packed 4-bit/nibble (0.5 bytes/element) with per-block fp16 scales.
NOTE: This is NOT GGUF Q4_K_M unless it matches GGUF exactly.
      The per-block structure (fp16 scales + nibble data) is similar in spirit
      but the exact metadata layout is a prototype (not bit-exact GGUF Q4_K_M).
"""
import os, sys, json, struct, hashlib, pathlib

import numpy as np

FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "data" / "phase31s_fixture"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# Import packing utility
from wlow_pack import pack_wlow, unpack_wlow, sha256_bytes, BLOCK_SIZE

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Phase 31S-R Artifact Writer (Packed Nibble W_low) ===\n")
    
    # Load or generate W_ref
    W_ref_path = pathlib.Path("/tmp/ffn_up_W_ref.npy")
    if W_ref_path.exists():
        print(f"Loading real W_ref from {W_ref_path}")
        W_ref = np.load(W_ref_path)
        real_ref = True
    else:
        print("Generating synthetic W_ref (seed=42)")
        np.random.seed(42)
        W_ref = np.random.randn(896, 4864).astype(np.float32) * 0.1
        real_ref = False
    
    rows, cols = W_ref.shape
    print(f"W_ref shape: {rows}x{cols}, dtype: {W_ref.dtype}")
    
    # Pack W_low (nibble format)
    print("\nPacking W_low (packed nibble format)...")
    packed_bytes, scales = pack_wlow(W_ref, block_size=BLOCK_SIZE)
    packed_bytes = bytes(packed_bytes)
    print(f"Packed bytes: {len(packed_bytes):,} ({len(packed_bytes)/W_ref.size:.4f} bytes/element)")
    print(f"Scales: {scales.shape}, dtype: {scales.dtype}")
    print(f"Scales bytes: {scales.nbytes:,}")
    
    W_low_total_bytes = len(packed_bytes) + scales.nbytes
    print(f"W_low total bytes: {W_low_total_bytes:,} (packed: {len(packed_bytes):,} + scales: {scales.nbytes:,})")
    
    # Compute residual and encode (same as 31S)
    print("\nComputing residual R = W_ref - W_low...")
    # For residual, we need to reconstruct W_low to compute residual
    # Since we only store packed format, reconstruct for residual computation
    W_low_reconstructed = unpack_wlow(packed_bytes, scales, rows, cols, block_size=BLOCK_SIZE)
    
    R = W_ref - W_low_reconstructed.astype(np.float32)
    K_PCT = 7.5
    R_flat = R.flatten()
    abs_R = np.abs(R_flat)
    threshold = np.percentile(abs_R, 100 - K_PCT)
    mask = abs_R >= threshold
    nnz = int(mask.sum())
    
    bitmap_packed = np.packbits(mask)
    values_fp16 = R_flat[mask].astype(np.float16)
    
    residual_bytes = 16 + bitmap_packed.nbytes + values_fp16.nbytes
    print(f"Residual: nnz={nnz:,}, bitmap={bitmap_packed.nbytes:,}, values={values_fp16.nbytes:,}, total={residual_bytes:,}")
    
    # Write W_low artifact (packed nibbles + scales appended)
    tensors_dir = FIXTURE_DIR / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    
    w_low_path = tensors_dir / "blk.0.ffn_up.W_low.bin"
    with open(w_low_path, 'wb') as f:
        f.write(packed_bytes)
        f.write(scales.tobytes())
    print(f"\nWrote: {w_low_path}")
    
    # Write residual artifact (16-byte header + bitmap + values)
    residual_path = tensors_dir / "blk.0.ffn_up.residual.sdir"
    with open(residual_path, 'wb') as f:
        f.write(struct.pack('<IIII', rows, cols, nnz, 0))  # header
        f.write(bitmap_packed.tobytes())
        f.write(values_fp16.tobytes())
    print(f"Wrote: {residual_path}")
    
    # Checksums
    w_low_hash = sha256_file(w_low_path)
    residual_hash = sha256_file(residual_path)
    print(f"\nW_low SHA256: {w_low_hash}")
    print(f"Residual SHA256: {residual_hash}")
    
    # Metadata dir
    metadata_dir = FIXTURE_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    checksums = {
        "schema_version": "0.1.0",
        "files": [
            {"path": "tensors/blk.0.ffn_up.W_low.bin", "sha256": w_low_hash},
            {"path": "tensors/blk.0.ffn_up.residual.sdir", "sha256": residual_hash},
        ]
    }
    checksums_path = metadata_dir / "checksums.json"
    with open(checksums_path, 'w') as f:
        json.dump(checksums, f, indent=2)
    print(f"Wrote: {checksums_path}")
    
    # Manifest
    W_ref_fp32 = W_ref.nbytes
    W_ref_Q4 = rows * cols  # Q4 budget = full element count (not compressed size)
    total_sub = W_low_total_bytes + residual_bytes
    margin = W_ref_Q4 - total_sub
    
    manifest = {
        "schema_version": "0.1.0",
        "package_id": "phase31s-r-ffn-up-layer0-packed",
        "source_model": {
            "model_name": "Qwen2.5-0.5B-ffn_up-layer0",
            "quantization": "W_ref_fp32",
            "gguf_source_path": str(W_ref_path) if real_ref else None,
            "architecture": "ffn_up",
            "note": f"Real extracted W_ref from {'/tmp/ffn_up_W_ref.npy' if real_ref else 'synthetic'}"
        },
        "substitution_policy": {
            "k_percent": K_PCT,
            "W_low_format": "packed-nibble-uint8-storage",
            "W_low_format_note": "Packed 4-bit/nibble (0.5 bytes/element) with per-block fp16 scales. NOT bit-exact GGUF Q4_K_M — this is a prototype format (same nibble density, different metadata layout).",
            "W_low_block_size": BLOCK_SIZE,
            "residual_encoding": "bitmap+fp16-header",
            "value_dtype": "fp16",
            "mask_encoding": "dense_bitmap"
        },
        "runtime_requirements": {
            "W_ref_must_be_absent": True,
            "dense_R_must_not_be_materialized": True,
            "fail_fast_if_residual_missing": True,
            "path_label": "[SDI-SUB-RUNTIME]"
        },
        "global_memory": {
            "W_ref_total_bytes_avoided": W_ref_fp32,
            "memory_margin_bytes": margin,
            "note": "Memory margin is REAL based on actual packed W_low artifact bytes"
        },
        "layers": [{
            "layer": 0,
            "family": "ffn_up",
            "shape": [rows, cols],
            "W_ref_f32_bytes": W_ref_fp32,
            "W_ref_Q4_budget_bytes": W_ref_Q4,
            "W_low_packed_bytes": len(packed_bytes),
            "W_low_scales_bytes": scales.nbytes,
            "W_low_total_bytes": W_low_total_bytes,
            "W_low_format": "packed-nibble-uint8-storage",
            "residual_encoded_bytes": residual_bytes,
            "residual_header_bytes": 16,
            "residual_bitmap_bytes": bitmap_packed.nbytes,
            "residual_values_bytes": values_fp16.nbytes,
            "residual_nnz": nnz,
            "memory_margin_bytes": margin,
            "hash_W_low": w_low_hash,
            "hash_residual": residual_hash
        }]
    }
    
    manifest_path = FIXTURE_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote: {manifest_path}")
    
    print(f"\n{'='*50}")
    print(f"W_low format: packed-nibble (0.5 bytes/element)")
    print(f"W_low packed bytes: {len(packed_bytes):,}")
    print(f"W_low scales bytes: {scales.nbytes:,}")
    print(f"W_low total bytes: {W_low_total_bytes:,}")
    print(f"Residual bytes: {residual_bytes:,}")
    print(f"Total substitutive: {total_sub:,}")
    print(f"Q4 budget: {W_ref_Q4:,}")
    print(f"Margin: {margin:,} ({'POSITIVE ✅' if margin > 0 else 'NEGATIVE ❌'})")
    print(f"Bytes/element (W_low only): {len(packed_bytes)/W_ref.size:.4f}")
    print(f"Bytes/element (total incl. scales): {W_low_total_bytes/W_ref.size:.4f}")

if __name__ == '__main__':
    main()
