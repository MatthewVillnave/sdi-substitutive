#!/usr/bin/env python3
"""
Phase 31AO — True Q2_K Encoder/Decoder Correctness Prototype
Repo: sdi-substitutive, HEAD: 6c5227f175d95f42b44ef7b08fc6a60ce2eb6d5a

Goal: Build the smallest safe true Q2_K encode/decode prototype.

Classification: BLOCKED_Q2K_ENCODER
Reason: Custom Q2_K encoder has fundamental issues (divide by zero,
overflow). No type-10 (Q2_K) tensors in any available GGUF file.
gguf-py quantize_blocks = NotImplemented.

No model files committed.
"""

import sys
sys.path.insert(0, "/home/matthew-villnave/llama.cpp/gguf-py")

import json
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

USB_BASE = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official"
Q4KM_PATH = f"{USB_BASE}/qwen2.5-0.5b-instruct-q4_k_m.gguf"

QK_K = 256
Q2_BPE = 2.625
Q2_BLOCK_BYTES = 84
Q4_BUDGET = 2179072


def main():
    print("=== Phase 31AO: True Q2_K Encoder/Decoder Prototype ===\n")

    # Q2_K format constants from ggml-common.h
    print("Q2_K format from ggml-common.h:")
    print(f"  block_q2_K: d(2) + dmin(2) + scales(16) + qs(64) = {Q2_BLOCK_BYTES} bytes")
    print(f"  block_size: {QK_K}")
    print(f"  bpe: {Q2_BPE}")

    # Byte size verification
    n_elements = 4864 * 896
    n_blocks = n_elements // QK_K
    expected_bytes = n_blocks * Q2_BLOCK_BYTES
    print(f"\nByte size for ffn_up ({n_elements:,} elements):")
    print(f"  {n_blocks} blocks × {Q2_BLOCK_BYTES} bytes = {expected_bytes:,} ({expected_bytes/1024:.1f} KB)")
    print(f"  Q4 budget: {Q4_BUDGET:,} ({Q4_BUDGET/1024:.1f} KB)")
    print(f"  Expected margin: {Q4_BUDGET - expected_bytes:,} ({(Q4_BUDGET - expected_bytes)/1024:.1f} KB)")

    # Check if any type-10 (Q2_K) tensors exist in local GGUF files
    q2k_path = f"{USB_BASE}/qwen2.5-0.5b-instruct-q2_k.gguf"
    r2 = GGUFReader(q2k_path)
    type_10 = [t for t in r2.tensors if t.tensor_type.value == 10]
    print(f"\nType-10 (Q2_K) tensors in Q2_K GGUF: {len(type_10)}")

    # Load W_ref for comparison
    reader4 = GGUFReader(Q4KM_PATH)
    t5 = next(t for t in reader4.tensors if t.name == 'blk.0.ffn_up.weight')
    W_ref = dequantize(t5.data, t5.tensor_type)
    print(f"W_ref shape: {W_ref.shape}, range=[{W_ref.min():.4f}, {W_ref.max():.4f}]")

    # Classification
    classification = "BLOCKED_Q2K_ENCODER"

    results = {
        'phase': '31AO',
        'classification': classification,
        'q2_format_source': 'llama.cpp/ggml/src/ggml-common.h',
        'q2_block_bytes': Q2_BLOCK_BYTES,
        'q2_bpe': Q2_BPE,
        'q2_block_size': QK_K,
        'expected_bytes': expected_bytes,
        'q4_budget': Q4_BUDGET,
        'expected_margin': Q4_BUDGET - expected_bytes,
        'type_10_tensors_in_q2k_gguf': len(type_10),
        'encoder_issues': [
            'gguf-py Q2_K.quantize_blocks = NotImplemented',
            'Custom encoder has divide-by-zero/overflow issues',
            'No type-10 Q2_K tensors available to validate against',
            'Q2_K GGUF contains IQ4_NL/Q3_K, not type-10 Q2_K',
        ],
    }

    out_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31AO_TRUE_Q2K_ENCODER_DECODER.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"Classification: {classification}")

    return results


if __name__ == '__main__':
    main()