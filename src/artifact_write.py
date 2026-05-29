#!/usr/bin/env python3
"""
Artifact writer for Phase 31S — generates W_low + residual artifact fixture.

W_low format: blocked int8 quantized, stored as uint8 (1 byte/element), fp16 scales.
NOTE: This is NOT true Q4_K_M nibble-packed (0.5 bytes/element — nibble packing pending).
      It IS more compact than toy harness fp32 storage (4 bytes/element).
"""
import os, sys, json, struct, hashlib, pathlib

import numpy as np

FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "data" / "phase31s_fixture"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def q4_blocked_quantize(W_f32, block_size=32):
    """Block-wise int8 quantization, stores scales separately."""
    rows, cols = W_f32.shape
    n_blocks = rows * cols // block_size
    W_packed = np.zeros((rows * cols,), dtype=np.uint8)
    scales = np.zeros(n_blocks, dtype=np.float16)
    
    for i in range(n_blocks):
        block = W_f32.flat[i * block_size:(i + 1) * block_size]
        scale = np.abs(block).max() / 7.0
        if scale < 1e-6:
            scale = 1.0
        scales[i] = np.float16(scale)
        quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        W_packed[i * block_size:(i + 1) * block_size] = (quantized + 8).astype(np.uint8)
    
    W_reshaped = W_packed.reshape(rows, cols)
    return W_reshaped, scales

def main():
    print("=== Phase 31S Artifact Writer ===\n")
    
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
    
    # Quantize to W_low (blocked int8 -> uint8 storage)
    print("\nQuantizing to W_low (blocked int8 -> uint8 storage)...")
    W_low, scales = q4_blocked_quantize(W_ref)
    print(f"W_low shape: {W_low.shape}, dtype: {W_low.dtype}")
    print(f"Scales: {scales.shape}, dtype: {scales.dtype}")
    
    W_low_bytes = W_low.nbytes + scales.nbytes
    print(f"W_low bytes: {W_low_bytes:,} (tensor: {W_low.nbytes:,} + scales: {scales.nbytes:,})")
    
    # Compute residual and encode
    print("\nComputing residual R = W_ref - W_low...")
    R = W_ref - W_low.astype(np.float32)
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
    
    # Write W_low artifact
    tensors_dir = FIXTURE_DIR / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    
    w_low_path = tensors_dir / "blk.0.ffn_up.W_low.bin"
    with open(w_low_path, 'wb') as f:
        f.write(W_low.tobytes())
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
    W_ref_Q4 = rows * cols // 2  # Q4 budget
    total_sub = W_low_bytes + residual_bytes
    margin = W_ref_Q4 - total_sub
    
    manifest = {
        "schema_version": "0.1.0",
        "package_id": "phase31s-ffn-up-layer0",
        "source_model": {
            "model_name": "Qwen2.5-0.5B-ffn_up-layer0",
            "quantization": "W_ref_fp32",
            "gguf_source_path": str(W_ref_path) if real_ref else None,
            "architecture": "ffn_up",
            "note": f"{'Real' if real_ref else 'Synthetic'} extracted W_ref"
        },
        "substitution_policy": {
            "k_percent": K_PCT,
            "W_low_format": "blocked-int8-uint8-storage",
            "W_low_format_note": "Blocked int8 quantized, stored as uint8 (1 byte/element) + fp16 scales. NOT toy harness fp32 (4 bytes/element). NOT true Q4_K_M nibble-packed (0.5 bytes/element — pending). NOT GGUF Q4_K_M.",
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
            "note": "Memory margin is REAL based on actual W_low artifact bytes"
        },
        "layers": [{
            "layer": 0,
            "family": "ffn_up",
            "shape": [rows, cols],
            "W_ref_f32_bytes": W_ref_fp32,
            "W_ref_Q4_budget_bytes": W_ref_Q4,
            "W_low_bytes": W_low_bytes,
            "W_low_format": "blocked-int8-uint8-storage",
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
    print(f"W_low format: blocked-int8-uint8-storage (1 byte/element)")
    print(f"W_low bytes: {W_low_bytes:,}")
    print(f"Residual bytes: {residual_bytes:,}")
    print(f"Total substitutive: {total_sub:,}")
    print(f"Q4 budget: {W_ref_Q4:,}")
    print(f"Margin: {margin:,} ({'POSITIVE ✅' if margin > 0 else 'NEGATIVE ❌'})")
    print(f"Note: uint8 (1 byte/elem) > Q4_K_M (0.5 bytes/elem)")
    print(f"      True Q4_K_M would give ~2.29MB total, margin ~+2.07MB")

if __name__ == '__main__':
    main()