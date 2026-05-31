#!/usr/bin/env python3
"""
Phase 31AV — Multi-Seed All-Layer Robustness (Lean)
Seeds: 0, 5, 9 — k=1% only
"""

import ctypes, json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

USB_BASE = '/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official'
Q4KM_PATH = f'{USB_BASE}/qwen2.5-0.5b-instruct-q4_k_m.gguf'
LIB = '/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so'
QK_K = 256; Q2_BLOCK_BYTES = 84; Q4_BUDGET_FAMILY = 2179072; Q4_BUDGET_LAYER = Q4_BUDGET_FAMILY * 3
SEEDS = [0, 5, 9]

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

print("=== Phase 31AV: Multi-Seed Robustness ===\n")

# Load model once
t0 = time.time()
reader = GGUFReader(Q4KM_PATH)
names = [t.name for t in reader.tensors]
up = sorted(set(int(n.split('.')[1]) for n in names if n.endswith('.ffn_up.weight')))
all_layers = sorted(up)
n_layers = len(all_layers)
print(f"Model loaded in {time.time()-t0:.1f}s, {n_layers} layers")

# Per-layer, per-seed results (stored as simple lists)
# layer_stats[li] = {seed: {delta_cos, MAE_improvement, margin, ...}}
layer_stats = [{'layer': li, 'seed_results': {}} for li in all_layers]
seed_agg = {s: {} for s in SEEDS}

for si, seed in enumerate(SEEDS):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1, 896)).astype(np.float32)

    seed_margins = []
    seed_delta_cos = []
    seed_mae_impr = []
    seed_cos_pos = []
    seed_mae_pos = []

    for li in all_layers:
        f = {}
        for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
            t = next(x for x in reader.tensors if x.name == f'blk.{li}.{fam}.weight')
            W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
            buf = q2_encode(W_ref.flatten())
            W_low = q2_decode(buf, W_ref.size).reshape(W_ref.shape)
            f[fam] = {'W_ref': W_ref, 'W_low': W_low, 'base_bytes': buf.nbytes}

        # Y_ref
        Y_ref = (silu(X @ f['ffn_gate']['W_ref'].T) *
                 (X @ f['ffn_up']['W_ref'].T)) @ f['ffn_down']['W_ref'].T
        # Y_low
        Y_low = (silu(X @ f['ffn_gate']['W_low'].T) *
                 (X @ f['ffn_up']['W_low'].T)) @ f['ffn_down']['W_low'].T

        # Y_sub k=1
        dec_gate = decode_sdir(encode_sdir(f['ffn_gate']['W_ref'] - f['ffn_gate']['W_low'], 1.0))
        dec_up   = decode_sdir(encode_sdir(f['ffn_up']['W_ref']   - f['ffn_up']['W_low'],   1.0))
        dec_down = decode_sdir(encode_sdir(f['ffn_down']['W_ref'] - f['ffn_down']['W_low'], 1.0))
        W_gate_sub = f['ffn_gate']['W_low'] + dec_gate
        W_up_sub   = f['ffn_up']['W_low']   + dec_up
        W_down_sub = f['ffn_down']['W_low']  + dec_down
        Y_sub = (silu(X @ W_gate_sub.T) * (X @ W_up_sub.T)) @ W_down_sub.T

        res_bytes = (len(encode_sdir(f['ffn_up']['W_ref']   - f['ffn_up']['W_low'],   1.0)) +
                     len(encode_sdir(f['ffn_gate']['W_ref'] - f['ffn_gate']['W_low'], 1.0)) +
                     len(encode_sdir(f['ffn_down']['W_ref'] - f['ffn_down']['W_low'], 1.0)))
        total_bytes = f['ffn_gate']['base_bytes'] + f['ffn_up']['base_bytes'] + f['ffn_down']['base_bytes'] + res_bytes
        margin = Q4_BUDGET_LAYER - total_bytes

        cos_low = cos_sim(Y_ref, Y_low)
        cos_sub = cos_sim(Y_ref, Y_sub)
        mae_low = float(np.abs(Y_ref - Y_low).mean())
        mae_sub = float(np.abs(Y_ref - Y_sub).mean())
        mae_delta = mae_sub - mae_low

        delta_cos = cos_sub - cos_low
        mae_impr = abs(mae_delta)

        layer_stats[li]['seed_results'][seed] = {
            'margin': int(margin),
            'cos_low': round(cos_low, 6),
            'cos_sub': round(cos_sub, 6),
            'delta_cos': round(delta_cos, 6),
            'MAE_low': round(mae_low, 6),
            'MAE_sub': round(mae_sub, 6),
            'MAE_delta': round(mae_delta, 6),
            'MAE_improvement': round(mae_impr, 6),
            'cosine_improved': bool(delta_cos > 0),
            'MAE_improved': bool(mae_delta < 0),
        }

        seed_margins.append(margin)
        seed_delta_cos.append(delta_cos)
        seed_mae_impr.append(mae_impr)
        seed_cos_pos.append(delta_cos > 0)
        seed_mae_pos.append(mae_delta < 0)

    n_mp = sum(seed_margins[i] > 0 for i in range(n_layers))
    n_cp = sum(seed_cos_pos)
    n_mi = sum(seed_mae_pos)

    seed_agg[seed] = {
        'seed': seed,
        'n_memory_positive': int(n_mp),
        'n_cosine_positive': int(n_cp),
        'n_mae_improving': int(n_mi),
        'aggregate_margin': int(sum(seed_margins)),
        'worst_margin_layer': int(all_layers[seed_margins.index(min(seed_margins))]),
        'worst_margin': int(min(seed_margins)),
        'worst_cosine_layer': int(all_layers[seed_delta_cos.index(min(seed_delta_cos))]),
        'worst_cosine_delta_cos': round(min(seed_delta_cos), 6),
        'avg_delta_cos': round(sum(seed_delta_cos)/n_layers, 6),
        'mean_MAE_improvement': round(sum(seed_mae_impr)/n_layers, 6),
        'all_pass': n_mp == n_layers and n_cp == n_layers and n_mi == n_layers,
    }

    print(f"  seed={seed}: mem={n_mp}/24, cos={n_cp}/24, MAE={n_mi}/24, "
          f"agg_margin={sum(seed_margins):+,}, worst_cos=layer{seed_agg[seed]['worst_cosine_layer']}"
          f"({seed_agg[seed]['worst_cosine_delta_cos']:+.5f}) [{time.time()-t0:.1f}s]")

# By-layer aggregate
print()
layer_by_layer = []
for li_idx, li in enumerate(all_layers):
    deltas = [layer_stats[li_idx]['seed_results'][s]['delta_cos'] for s in SEEDS]
    maes   = [layer_stats[li_idx]['seed_results'][s]['MAE_improvement'] for s in SEEDS]
    cos_pos = sum(1 for d in deltas if d > 0)
    mae_pos = sum(1 for m in maes if m > 0)

    stats = {
        'layer': li,
        'n_cos_pos': int(cos_pos),
        'n_mae_pos': int(mae_pos),
        'n_seeds': len(SEEDS),
        'mean_delta_cos': round(sum(deltas)/len(SEEDS), 6),
        'min_delta_cos': round(min(deltas), 6),
        'max_delta_cos': round(max(deltas), 6),
        'mean_MAE_improvement': round(sum(maes)/len(SEEDS), 6),
        'robust': bool(cos_pos == len(SEEDS) and mae_pos == len(SEEDS)),
        'sensitive': bool(0 < cos_pos < len(SEEDS)),
        'failing': bool(cos_pos == 0),
    }
    layer_by_layer.append(stats)
    flag = "ROBUST" if stats['robust'] else ("SENSITIVE" if stats['sensitive'] else "FAILING")
    print(f"  layer {li:2d}: cos_pos={cos_pos}/3, mae_pos={mae_pos}/3, "
          f"mean_dcos={stats['mean_delta_cos']:+.5f}, range=[{stats['min_delta_cos']:+.5f}, {stats['max_delta_cos']:+.5f}] [{flag}]")

# Classification
robust_seeds = sum(1 for s in SEEDS if seed_agg[s]['all_pass'])
sensitive_layers = sum(1 for ls in layer_by_layer if ls['sensitive'])
failing_layers = sum(1 for ls in layer_by_layer if ls['failing'])
robust_layers = sum(1 for ls in layer_by_layer if ls['robust'])

print(f"\nrobust_seeds={robust_seeds}/{len(SEEDS)}, robust_layers={robust_layers}/24, "
      f"sensitive_layers={sensitive_layers}, failing_layers={failing_layers}")

if robust_seeds == len(SEEDS) and sensitive_layers == 0 and failing_layers == 0:
    classification = "PASS_MULTI_SEED_ALL_LAYER_ROBUST"
elif failing_layers > 0:
    classification = "PARTIAL_MULTI_SEED_LAYER_SENSITIVE"
elif sensitive_layers > 0:
    if robust_seeds == len(SEEDS):
        classification = "PARTIAL_MULTI_SEED_COSINE_SENSITIVE"
    else:
        classification = "PARTIAL_MULTI_SEED_MAE_ONLY"
else:
    classification = "PARTIAL_MULTI_SEED_COSINE_SENSITIVE"

print(f"Classification: {classification}")

# Build JSON
result_out = {
    'phase': '31AV',
    'classification': classification,
    'seeds_tested': SEEDS,
    'k_tested': 1.0,
    'n_layers': n_layers,
    'layer_indices': all_layers,
    'by_seed': [seed_agg[s] for s in SEEDS],
    'by_layer': layer_by_layer,
    'mae_convention': {
        'MAE_delta_formula': 'MAE_sub - MAE_low',
        'MAE_delta_positive_means': 'MAE worsened',
        'MAE_delta_negative_means': 'MAE improved',
        'MAE_improvement': 'abs(MAE_delta); positive means MAE improved',
    },
    'summary': {
        'robust_seeds': robust_seeds,
        'total_seeds': len(SEEDS),
        'robust_layers': robust_layers,
        'sensitive_layers': sensitive_layers,
        'failing_layers': failing_layers,
    },
}

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31AV_MULTI_SEED_ALL_LAYER_ROBUSTNESS.json'
with open(json_path, 'w') as f:
    json.dump(result_out, f, indent=2)
print(f"\nJSON: {json_path} — {os.path.getsize(json_path)/1024:.1f} KB")
print(f"\nDONE. classification={classification}")