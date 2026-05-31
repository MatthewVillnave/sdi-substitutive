#!/usr/bin/env python3
"""
31AW Family Ablation — standalone, clean variable names.
Layer 21, seed=9, k=1%, alpha=1.0.
Tests which family (up/gate/down) causes the cosine regression.
"""

import ctypes, json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

LIB = '/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so'
Q4_BUDGET_FAMILY = 2179072; Q4_BUDGET_LAYER = Q4_BUDGET_FAMILY * 3
QK_K = 256; Q2_BLOCK_BYTES = 84

lib = ctypes.CDLL(LIB)
lib.quantize_row_q2_K_ref.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
lib.quantize_row_q2_K_ref.restype = None
lib.dequantize_row_q2_K.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.dequantize_row_q2_K.restype = None

def q2_encode(flat):
    n = flat.size
    buf = np.zeros(n // QK_K * Q2_BLOCK_BYTES, dtype=np.uint8)
    lib.quantize_row_q2_K_ref(flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               ctypes.cast(buf.ctypes.data, ctypes.c_void_p), n)
    return buf

def q2_decode(buf, n):
    out = np.zeros(n, dtype=np.float32)
    lib.dequantize_row_q2_K(ctypes.cast(buf.ctypes.data, ctypes.c_void_p),
                             out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
    return out

def silu(x): return x / (1.0 + np.exp(-np.clip(x, -709, 709)))
def cos_sim(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print("Loading model...")
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')

# Load layer 21 families
LI = 21
layer_families = {}
for family in ['ffn_up', 'ffn_gate', 'ffn_down']:
    tensor = next(t for t in reader.tensors if t.name == f'blk.{LI}.{family}.weight')
    W_ref = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
    buf = q2_encode(W_ref.flatten())
    W_low = q2_decode(buf, W_ref.size).reshape(W_ref.shape)
    layer_families[family] = {
        'W_ref': W_ref,
        'W_low': W_low,
        'R': W_ref - W_low,
        'base_bytes': buf.nbytes,
    }

# Activation seed=9
rng = np.random.default_rng(9)
X = rng.standard_normal((1, 896)).astype(np.float32)

def compute_Y(WG,WU,WD,X):
    return (silu(X @ WG.T) * (X @ WU.T)) @ WD.T

# Reference and low-Precision baselines
Y_ref = compute_Y(layer_families['ffn_gate']['W_ref'],
                  layer_families['ffn_up']['W_ref'],
                  layer_families['ffn_down']['W_ref'], X)
Y_low = compute_Y(layer_families['ffn_gate']['W_low'],
                  layer_families['ffn_up']['W_low'],
                  layer_families['ffn_down']['W_low'], X)

print(f"\nBaseline:")
print(f"  cos(Y_ref,Y_low)={cos_sim(Y_ref,Y_low):.6f}")
print(f"  MAE(Y_ref,Y_low)={float(np.abs(Y_ref-Y_low).mean()):.6f}")

# Family subsets
# active_families is a list of family names to apply residual to
FAMILY_COMBOS = [
    ('up',),
    ('gate',),
    ('down',),
    ('up', 'gate'),
    ('up', 'down'),
    ('gate', 'down'),
    ('up', 'gate', 'down'),
]

def apply_residual(activations, k_pct, alpha):
    """Compute Y_sub with residual applied to specified families only."""
    results = {}
    for prefix in ['ffn_up', 'ffn_gate', 'ffn_down']:
        results[prefix] = layer_families[prefix]['W_low'].copy()

    base_bytes = sum(layer_families[p]['base_bytes'] for p in ['ffn_up','ffn_gate','ffn_down'])
    res_bytes = 0

    for fam in activations:
        prefix = f'ffn_{fam}'
        R = layer_families[prefix]['R']
        R_scaled = alpha * R
        dec = decode_sdir(encode_sdir(R_scaled, k_pct))
        results[prefix] = layer_families[prefix]['W_low'] + dec
        res_bytes += len(encode_sdir(R, k_pct))

    total = base_bytes + res_bytes
    margin = Q4_BUDGET_LAYER - total

    Y_sub = compute_Y(results['ffn_gate'], results['ffn_up'], results['ffn_down'], X)
    return Y_sub, margin

print(f"\nFamily ablation at k=1%, alpha=1.0, layer 21, seed=9:")
print(f"  {'active':20} | {'delta_cos':>10} | {'MAE_delta':>10} | {'MAE_abs':>9} | {'margin':>10} | {'mem':>4} | {'cos>0':>6} | {'MAE>0':>6}")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}")

best_entry = None
for combo in FAMILY_COMBOS:
    Y_sub, margin = apply_residual(combo, k_pct=1.0, alpha=1.0)
    dc = cos_sim(Y_ref, Y_sub) - cos_sim(Y_ref, Y_low)
    mae_sub = float(np.abs(Y_ref - Y_sub).mean())
    mae_low = float(np.abs(Y_ref - Y_low).mean())
    mae_delta = mae_sub - mae_low
    mem_pos = margin > 0
    cos_pos = dc > 0
    mae_pos = mae_delta < 0  # MAE improved means delta is negative
    label = '+'.join(combo)
    print(f"  {label:>20} | {dc:>+10.6f} | {mae_delta:>+10.6f} | {abs(mae_delta):>9.6f} | {margin:>10,} | {'Y' if mem_pos else 'N':>4} | {'Y' if cos_pos else 'N':>6} | {'Y' if mae_pos else 'N':>6}")
    if mem_pos and mae_pos:
        if best_entry is None or abs(mae_delta) > abs(best_entry[1]):
            best_entry = (label, mae_delta, dc, margin, mem_pos, mae_pos, cos_pos)

print()
if best_entry:
    print(f"  Best MAE-improvement subset: {best_entry[0]} (MAE_delta={best_entry[1]:+.6f}, cosine={'POSITIVE' if best_entry[5] else 'NEGATIVE'})")
else:
    print("  No subset achieves both memory-positive and MAE-improvement.")

print("\nDONE")