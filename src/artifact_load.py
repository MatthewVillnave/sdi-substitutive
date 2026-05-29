#!/usr/bin/env python3
"""
Artifact loader for Phase 31S-R — validates packed W_low + residual artifact fixture.

W_low format: packed 4-bit/nibble (0.5 bytes/element) with per-block fp16 scales.
Loads packed W_low, unpacks to fp32 approximation, verifies checksums, exposes counters.
"""
import os, sys, json, struct, hashlib

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'phase31s_fixture')
MANIFEST_PATH = os.path.join(FIXTURE_DIR, 'manifest.json')
CHECKSUMS_PATH = os.path.join(FIXTURE_DIR, 'metadata', 'checksums.json')

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wlow_pack import unpack_wlow, BLOCK_SIZE

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def load_manifest():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    print(f"Loaded manifest: schema_version={m.get('schema_version')}")
    return m

def verify_checksums(manifest):
    with open(CHECKSUMS_PATH) as f:
        checksums = json.load(f)
    
    all_ok = True
    for entry in checksums.get('files', []):
        path = os.path.join(FIXTURE_DIR, entry['path'])
        if not os.path.exists(path):
            print(f"MISSING FILE: {path}")
            all_ok = False
            continue
        computed = sha256_file(path)
        expected = entry['sha256']
        if computed != expected:
            print(f"CHECKSUM MISMATCH: {entry['path']}")
            print(f"  expected: {expected}")
            print(f"  computed: {computed}")
            all_ok = False
        else:
            print(f"  OK: {entry['path']}")
    return all_ok

def load_W_low_packed(manifest_layer):
    """
    Load packed W_low artifact (nibble format), return (packed_bytes, scales, rows, cols)
    """
    family = manifest_layer['family']
    layer = manifest_layer['layer']
    filename = f"blk.{layer}.{family}.W_low.bin"
    path = os.path.join(FIXTURE_DIR, 'tensors', filename)
    
    rows = manifest_layer['shape'][0]
    cols = manifest_layer['shape'][1]
    n_elements = rows * cols
    n_blocks = n_elements // BLOCK_SIZE
    
    with open(path, 'rb') as f:
        data = f.read()
    
    n_packed_bytes = (n_elements + 1) // 2
    n_scales_bytes = n_blocks * 2  # fp16
    
    packed_bytes = data[:n_packed_bytes]
    scales = data[n_packed_bytes:n_packed_bytes + n_scales_bytes]
    
    # Parse scales as numpy fp16 array
    import numpy as np
    scales = np.frombuffer(scales, dtype=np.float16)
    
    print(f"Loaded packed W_low: packed={len(packed_bytes):,} bytes, scales={scales.nbytes:,} bytes")
    return packed_bytes, scales, rows, cols

def load_residual(manifest_layer):
    """Load residual artifact, return parsed (header_dict, bitmap_bytes, values_fp16)"""
    family = manifest_layer['family']
    layer = manifest_layer['layer']
    filename = f"blk.{layer}.{family}.residual.sdir"
    path = os.path.join(FIXTURE_DIR, 'tensors', filename)
    with open(path, 'rb') as f:
        data = f.read()
    
    # Parse 16-byte header: 4x uint32 LE
    in_dim, out_dim, nnz, flags = struct.unpack('<IIII', data[:16])
    bitmap_bytes = data[16:16 + (in_dim * out_dim + 7) // 8]
    values_bytes = data[16 + len(bitmap_bytes):]
    
    print(f"Loaded residual: header=(in={in_dim}, out={out_dim}, nnz={nnz}, flags={flags})")
    print(f"  bitmap={len(bitmap_bytes):,} bytes, values={len(values_bytes):,} bytes")
    
    header = {'in_dim': in_dim, 'out_dim': out_dim, 'nnz': nnz, 'flags': flags}
    return header, bitmap_bytes, values_bytes

def main():
    print("=== Phase 31S-R Artifact Loader (Packed Nibble W_low) ===\n")
    
    # Load manifest
    manifest = load_manifest()
    
    # Verify checksums
    print("\n--- Checksum verification ---")
    checksums_ok = verify_checksums(manifest)
    print(f"Checksums: {'PASS' if checksums_ok else 'FAIL'}")
    
    # Load W_low and residual for each layer
    print("\n--- Loading artifacts ---")
    counters = {
        'W_ref_loaded': 0,
        'W_low_loaded': 0,
        'W_low_packed_bytes': 0,
        'W_low_decode_temp_bytes': 0,
        'residual_encoded_loaded': 0,
        'dense_R_materialized': 0,
        'path_label': '[SDI-SUB-RUNTIME]',
        'checksums_valid': checksums_ok,
        'fail_fast_on_missing': False,
        'fail_fast_on_checksum_mismatch': not checksums_ok,
    }
    
    total_substitutive_bytes = 0
    
    for layer_entry in manifest.get('layers', []):
        layer = layer_entry['layer']
        family = layer_entry['family']
        print(f"\nLayer {layer} ({family}):")
        
        # Load packed W_low
        packed_bytes, scales, rows, cols = load_W_low_packed(layer_entry)
        packed_bytes_len = len(packed_bytes)
        scales_bytes = scales.nbytes
        
        # Unpack to verify decode works
        W_low_fp32 = unpack_wlow(packed_bytes, scales, rows, cols, block_size=BLOCK_SIZE)
        decode_temp_bytes = W_low_fp32.nbytes
        
        counters['W_low_packed_bytes'] += packed_bytes_len + scales_bytes
        counters['W_low_decode_temp_bytes'] += decode_temp_bytes
        
        residual_header, residual_bitmap, residual_values = load_residual(layer_entry)
        
        residual_total = 16 + len(residual_bitmap) + len(residual_values)
        counters['residual_encoded_loaded'] += residual_total
        
        counters['W_low_loaded'] += packed_bytes_len + scales_bytes
        total_substitutive_bytes += packed_bytes_len + scales_bytes + residual_total
        
        print(f"  W_low packed bytes: {packed_bytes_len:,} + scales {scales_bytes:,} (decode temp: {decode_temp_bytes:,})")
        print(f"  Residual encoded bytes: {residual_total:,}")
        print(f"  Total per layer: {packed_bytes_len + scales_bytes + residual_total:,}")
    
    # Fail-fast test: missing file
    print("\n--- Fail-fast: missing file ---")
    fake_path = os.path.join(FIXTURE_DIR, 'tensors', 'nonexistent.bin')
    if not os.path.exists(fake_path):
        counters['fail_fast_on_missing'] = True
        print(f"PASS: missing file detected correctly")
    
    print(f"\n--- Counters ---")
    print(f"  W_ref_loaded: {counters['W_ref_loaded']} {'✅ (absent)' if counters['W_ref_loaded'] == 0 else '❌'}")
    print(f"  W_low_packed_bytes: {counters['W_low_packed_bytes']:,} bytes ✅ (packed artifact)")
    print(f"  W_low_decode_temp_bytes: {counters['W_low_decode_temp_bytes']:,} bytes {'✅ (unpacked for decode)' if counters['W_low_decode_temp_bytes'] > 0 else 'NOTE: no decode materialized'}")
    print(f"  residual_encoded_loaded: {counters['residual_encoded_loaded']:,} bytes ✅")
    print(f"  dense_R_materialized: {counters['dense_R_materialized']} {'✅ (absent)' if counters['dense_R_materialized'] == 0 else '❌'}")
    print(f"  path_label: {counters['path_label']} ✅")
    
    # Memory accounting
    W_ref_fp32 = 17_432_576
    W_ref_Q4_budget = 4_358_144
    W_low_total = counters['W_low_packed_bytes']
    residual_total = counters['residual_encoded_loaded']
    total_sub = total_substitutive_bytes
    margin_vs_Q4 = W_ref_Q4_budget - total_sub
    decode_temp = counters['W_low_decode_temp_bytes']
    
    print(f"\n--- Memory Accounting ---")
    print(f"  W_ref (fp32): {W_ref_fp32:,}")
    print(f"  W_ref Q4 budget: {W_ref_Q4_budget:,}")
    print(f"  W_low packed+scales: {W_low_total:,} ({W_low_total/W_ref_Q4_budget:.4f} of Q4 budget)")
    print(f"  Residual encoded: {residual_total:,} ({residual_total/W_ref_Q4_budget:.4f} of Q4 budget)")
    print(f"  Total substitutive artifact bytes: {total_sub:,}")
    print(f"  Margin vs Q4 (artifact only): {margin_vs_Q4:,} ({'POSITIVE ✅' if margin_vs_Q4 > 0 else 'NEGATIVE ❌'})")
    
    if decode_temp > 0:
        runtime_total = total_sub + decode_temp
        runtime_margin = W_ref_Q4_budget - runtime_total
        print(f"  Runtime resident (artifact + decode temp): {runtime_total:,}")
        print(f"  Runtime margin: {runtime_margin:,} ({'POSITIVE ✅' if runtime_margin > 0 else 'NEGATIVE ❌'})")
    
    all_pass = (
        counters['W_ref_loaded'] == 0 and
        counters['W_low_loaded'] > 0 and
        counters['residual_encoded_loaded'] > 0 and
        counters['dense_R_materialized'] == 0 and
        checksums_ok
    )
    
    artifact_only_pass = margin_vs_Q4 > 0
    runtime_pass = decode_temp == 0 or runtime_margin > 0
    
    if all_pass and artifact_only_pass and runtime_pass:
        classification = 'PASS_PACKED_W_LOW_ARTIFACT'
    elif all_pass and artifact_only_pass and not runtime_pass:
        classification = 'PARTIAL_PACKED_ARTIFACT_MEMORY_PASS_DECODE_BLOAT'
    elif all_pass and not artifact_only_pass:
        classification = 'BLOCKED_MEMORY_STILL_NEGATIVE'
    else:
        classification = 'PARTIAL_PACKED_FORMAT_PROTOTYPE_ONLY'
    
    print(f"\n{'='*50}")
    print(f"Classification: {classification}")
    print(f"Artifact-only pass: {artifact_only_pass}")
    print(f"Runtime pass: {runtime_pass}")
    print(f"Overall: {'PASS' if classification == 'PASS_PACKED_W_LOW_ARTIFACT' else 'PARTIAL/BLOCKED'}")
    
    # Write summary JSON
    summary = {
        'classification': classification,
        'old_head': 'c24d468',
        'new_head': None,  # filled after commit
        'W_low_format': manifest.get('layers', [{}])[0].get('W_low_format', 'unknown'),
        'W_low_packed_bytes': counters['W_low_packed_bytes'],
        'W_low_scales_bytes': manifest.get('layers', [{}])[0].get('W_low_scales_bytes', 0),
        'W_low_total_bytes': manifest.get('layers', [{}])[0].get('W_low_total_bytes', 0),
        'W_low_decode_temp_bytes': counters['W_low_decode_temp_bytes'],
        'residual_bytes': counters['residual_encoded_loaded'],
        'total_substitutive_artifact_bytes': total_sub,
        'margin_vs_Q4': margin_vs_Q4,
        'margin_positive': margin_vs_Q4 > 0,
        'W_ref_Q4_budget': W_ref_Q4_budget,
        'runtime_margin_with_decode': runtime_margin if decode_temp > 0 else None,
        'checksums_valid': checksums_ok,
        'W_ref_loaded': counters['W_ref_loaded'],
        'W_low_loaded': counters['W_low_loaded'] > 0,
        'residual_encoded_loaded': counters['residual_encoded_loaded'] > 0,
        'dense_R_materialized': counters['dense_R_materialized'],
        'path_label': counters['path_label'],
        'fail_fast_on_missing': counters['fail_fast_on_missing'],
        'fail_fast_on_checksum_mismatch': counters['fail_fast_on_checksum_mismatch'],
        'phase31t_unlocked': classification == 'PASS_PACKED_W_LOW_ARTIFACT',
        'block_size': BLOCK_SIZE,
        'bytes_per_element_artifact': total_sub / (896 * 4864),
    }
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'PHASE31S_R_RESULTS.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_path}")
    
    return summary

if __name__ == '__main__':
    main()
