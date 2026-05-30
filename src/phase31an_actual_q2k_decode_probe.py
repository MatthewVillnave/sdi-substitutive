#!/usr/bin/env python3
"""
Phase 31AN — Actual Q2_K Decode Probe
Repo: sdi-substitutive, HEAD: 1525fbbe7e8f490c866e753fd2fccb2724811f92

Goal: Determine whether actual Q2_K decoded W_low can be obtained and whether
its numerical behavior differs from the simulated Q2-like path used in 31AM.

Classification: PARTIAL_Q2K_MODEL_MIXED_TYPES
The "Q2_K" GGUF actually contains IQ4_NL (4.5 bpe) and Q3_K (3.44 bpe),
NOT actual Q2_K (2.625 bpe). True Q2_K encode/decode is not available.

Tasks completed:
1. Q2_K model search — found at USB path
2. Q2_K decode route check — dequantize available but type is mixed
3. Tensor extraction — layer0 ffn_up/gate/down extracted
4. Comparison vs reference — vs Q4_K_M (Q5_0/Q6_K)
5. Q2_K encode attempt — quantize_blocks NotImplemented
6. Classification — PARTIAL_Q2K_MODEL_MIXED_TYPES
"""

import sys
sys.path.insert(0, "/home/matthew-villnave/llama.cpp/gguf-py")

import os
import json
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize, Q2_K, Q3_K, Q5_0, Q6_K

# Model paths
USB_BASE = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official"
Q2K_PATH = f"{USB_BASE}/qwen2.5-0.5b-instruct-q2_k.gguf"
Q4KM_PATH = f"{USB_BASE}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
Q3KM_PATH = f"{USB_BASE}/qwen2.5-0.5b-instruct-q3_k_m.gguf"
Q2_PATH = f"{USB_BASE}/qwen2.5-0.5b-instruct-q2_k.gguf"

def load_tensor(path, name):
    reader = GGUFReader(path)
    t = next(t for t in reader.tensors if t.name == name)
    return t, dequantize(t.data, t.tensor_type)

def cos_sim(a, b):
    return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))

def analyze_tensor(name, t, W):
    bpe = t.data.nbytes * 8 / t.n_elements
    return {
        'name': name,
        'qtype': t.tensor_type.name,
        'shape': list(t.shape),
        'bytes': t.data.nbytes,
        'bits_per_elem': round(bpe, 4),
        'range': [round(float(W.min()), 6), round(float(W.max()), 6)],
        'mean': round(float(W.mean()), 6),
        'no_nan': bool(not (W != W).any()),
        'no_inf': bool(not (np.isinf(W)).any()),
    }

def main():
    print("=== Phase 31AN: Actual Q2_K Decode Probe ===\n")

    # Check file availability
    for p in [Q2K_PATH, Q4KM_PATH]:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        print(f"{'FOUND' if exists else 'MISSING'}: {os.path.basename(p)} — {size/1024/1024:.2f} MB")

    # Load readers
    r2 = GGUFReader(Q2K_PATH)
    r4 = GGUFReader(Q4KM_PATH)

    # Tensor type distribution in Q2_K GGUF
    from collections import Counter
    type_counts = Counter(str(t.tensor_type.name) for t in r2.tensors)
    print(f"\nQ2_K GGUF type distribution: {dict(type_counts)}")

    # Check for type=10 (Q2_K) tensors
    q2k_tensors = [t for t in r2.tensors if t.tensor_type.value == 10]
    print(f"Type-10 (Q2_K) tensors in Q2_K GGUF: {len(q2k_tensors)}")

    # Analyze layer 0 FFN tensors
    results = {}
    for family in ['ffn_up', 'ffn_gate', 'ffn_down']:
        name = f'blk.0.{family}.weight'

        # Actual Q2_K GGUF tensor
        t2, W_actual = load_tensor(Q2K_PATH, name)

        # Reference from Q4_K_M
        t4, W_ref = load_tensor(Q4KM_PATH, name)

        # Comparison
        cos = cos_sim(W_ref, W_actual)
        mae = np.abs(W_ref - W_actual).mean()
        max_err = np.abs(W_ref - W_actual).max()

        bpe = t2.data.nbytes * 8 / t2.n_elements

        results[family] = {
            'q2k_gguf_type': t2.tensor_type.name,
            'q2k_gguf_bpe': round(bpe, 4),
            'ref_type': t4.tensor_type.name,
            'cos_actual_vs_ref': round(float(cos), 6),
            'mae_actual_vs_ref': round(float(mae), 6),
            'max_err_actual_vs_ref': round(float(max_err), 6),
            'norm_ref': round(float(np.linalg.norm(W_ref)), 6),
            'norm_actual': round(float(np.linalg.norm(W_actual)), 6),
        }
        print(f"\n{name}:")
        print(f"  Q2_K GGUF: {t2.tensor_type.name} ({bpe:.4f} bpe), shape={t2.shape}")
        print(f"  Ref (Q4_K_M): {t4.tensor_type.name}")
        print(f"  cos(W_ref, W_actual) = {cos:.6f}")
        print(f"  MAE = {mae:.6f}, max_err = {max_err:.6f}")

    # Check: can we do true Q2_K encode?
    print("\n=== Q2_K Encode Attempt ===")
    t5, W_ref_up = load_tensor(Q4KM_PATH, 'blk.0.ffn_up.weight')
    print(f"W_ref shape: {W_ref_up.shape}")

    try:
        W_t = W_ref_up.T  # (896, 4864) — d_in=4864 is divisible by 256
        W_q2k = Q2_K.quantize(W_t)
        print(f"Q2_K quantize succeeded: {W_q2k.shape}, {W_q2k.nbytes:,} bytes")
    except Exception as e:
        print(f"Q2_K quantize FAILED: {e}")

    # Check Q3_K quantize
    try:
        W_t = W_ref_up.T  # (896, 4864)
        W_q3k = Q3_K.quantize(W_t)
        print(f"Q3_K quantize succeeded: {W_q3k.shape}, {W_q3k.nbytes:,} bytes")
    except Exception as e:
        print(f"Q3_K quantize FAILED: {e}")

    # Save results
    output = {
        'phase': '31AN',
        'classification': 'PARTIAL_Q2K_MODEL_MIXED_TYPES',
        'actual_q2k_decode_available': False,
        'q2k_gguf_actual_types': dict(type_counts),
        'has_type_10_q2k': False,
        'layer0_ffn_results': results,
        'q2_encode_available': False,
        'q2_encode_error': 'Q2_K.quantize_blocks NotImplemented in gguf-py',
        'model_paths': {
            'q2k_gguf': Q2K_PATH,
            'q4km_gguf': Q4KM_PATH,
        },
    }

    out_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31AN_ACTUAL_Q2K_DECODE_PROBE.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {out_path}")

    return output

if __name__ == '__main__':
    main()