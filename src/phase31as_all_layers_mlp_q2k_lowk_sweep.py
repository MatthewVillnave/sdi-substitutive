#!/usr/bin/env python3
"""
Phase 31AS — All Available Layers Full MLP Q2_K + Low-k Residual Sweep
Repo: sdi-substitutive
"""

import ctypes, json, os, sys
import numpy as np

sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/gguf-py')

from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

REPO = '/home/matthew-villnave/sdi-substitutive'
USB_BASE = '/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official'
Q4KM_PATH = f'{USB_BASE}/qwen2.5-0.5b-instruct-q4_k_m.gguf'
LIB = '/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so'

QK_K = 256; Q2_BLOCK_BYTES = 84
Q4_BUDGET_FAMILY = 2179072; Q4_BUDGET_LAYER = Q4_BUDGET_FAMILY * 3

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

def to_native(obj):
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [to_native(x) for x in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return None
    else: return obj

print("=== Phase 31AS: All Layers Full MLP Q2_K + Low-k Sweep ===\n")

reader = GGUFReader(Q4KM_PATH)
names = [t.name for t in reader.tensors]

up   = set(int(n.split('.')[1]) for n in names if n.endswith('.ffn_up.weight'))
gate = set(int(n.split('.')[1]) for n in names if n.endswith('.ffn_gate.weight'))
down = set(int(n.split('.')[1]) for n in names if n.endswith('.ffn_down.weight'))
all_layers = sorted(up | gate | down)
n_layers = len(all_layers)
print(f"Discovered {n_layers} layers: {all_layers[0]}–{all_layers[-1]}")

# Q2 byte verification
print("\n=== Q2_K BYTE VERIFICATION ===")
q2_metrics = {}
for li in all_layers:
    families_data = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        name = f'blk.{li}.{fam}.weight'
        t = next(x for x in reader.tensors if x.name == name)
        W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
        W_flat = W_ref.flatten()
        buf = q2_encode(W_flat)
        W_low = q2_decode(buf, W_flat.size).reshape(W_ref.shape)
        n = W_flat.size; nb = n // QK_K
        exp = nb * Q2_BLOCK_BYTES
        families_data[fam] = {
            'shape': [int(x) for x in W_ref.shape],
            'n_elements': int(n), 'n_blocks': int(nb),
            'expected_bytes': int(exp), 'actual_bytes': int(len(buf)),
            'byte_match': bool(len(buf) == exp),
            'bpe': float(len(buf) * 8 / n),
            'Q4_budget': int(Q4_BUDGET_FAMILY),
            'margin': int(Q4_BUDGET_FAMILY - len(buf)),
            'cos': round(float(cos_sim(W_ref, W_low)), 8),
            'MAE': round(float(np.abs(W_ref - W_low).mean()), 8),
            'max_error': round(float(np.abs(W_ref - W_low).max()), 8),
        }
    q2_metrics[f'layer{li}'] = families_data
    ok = all(families_data[f]['byte_match'] for f in ['ffn_up','ffn_gate','ffn_down'])
    print(f"  layer {li}: all_byte_match={ok}")

# MLP sweep
print(f"\n=== MLP POLICY SWEEP ===")
per_layer_results = {}

for li in all_layers:
    families_data = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        name = f'blk.{li}.{fam}.weight'
        t = next(x for x in reader.tensors if x.name == name)
        W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
        W_flat = W_ref.flatten()
        buf = q2_encode(W_flat)
        W_low = q2_decode(buf, W_flat.size).reshape(W_ref.shape)
        families_data[fam] = {'W_ref': W_ref, 'W_low': W_low}

    base_bytes = sum(q2_metrics[f'layer{li}'][f]['actual_bytes'] for f in ['ffn_up','ffn_gate','ffn_down'])
    np.random.seed(42 + li)
    X = np.random.randn(1, families_data['ffn_up']['W_ref'].shape[1]).astype(np.float32)

    Y_ref = (silu(X @ families_data['ffn_gate']['W_ref'].T) * (X @ families_data['ffn_up']['W_ref'].T)) @ families_data['ffn_down']['W_ref'].T
    Y_q2  = (silu(X @ families_data['ffn_gate']['W_low'].T)  * (X @ families_data['ffn_up']['W_low'].T))  @ families_data['ffn_down']['W_low'].T
    cos_low = cos_sim(Y_ref, Y_q2)
    mae_low = float(np.abs(Y_ref - Y_q2).mean())

    layer_results = []
    for k in [0, 0.5, 1, 2, 3]:
        if k == 0:
            Y_sub = Y_q2; res_bytes = 0
        else:
            R_up   = families_data['ffn_up']['W_ref']   - families_data['ffn_up']['W_low']
            R_gate = families_data['ffn_gate']['W_ref'] - families_data['ffn_gate']['W_low']
            R_down = families_data['ffn_down']['W_ref'] - families_data['ffn_down']['W_low']
            enc_up   = encode_sdir(R_up, k)
            enc_gate = encode_sdir(R_gate, k)
            enc_down = encode_sdir(R_down, k)
            res_bytes = len(enc_up) + len(enc_gate) + len(enc_down)
            dec_up   = decode_sdir(enc_up)
            dec_gate = decode_sdir(enc_gate)
            dec_down = decode_sdir(enc_down)
            W_gate_sub = families_data['ffn_gate']['W_low'] + dec_gate
            W_up_sub = families_data['ffn_up']['W_low'] + dec_up
            W_down_sub = families_data['ffn_down']['W_low'] + dec_down
            Y_sub = (silu(X @ W_gate_sub.T) * (X @ W_up_sub.T)) @ W_down_sub.T

        total_bytes = base_bytes + res_bytes
        margin = Q4_BUDGET_LAYER - total_bytes
        cos_sub = cos_sim(Y_ref, Y_sub)
        mae_sub = float(np.abs(Y_ref - Y_sub).mean())
        mae_delta = mae_sub - mae_low
        delta_cos = cos_sub - cos_low

        layer_results.append({
            'k_pct': k, 'res_bytes': int(res_bytes), 'total_bytes': int(total_bytes),
            'margin': int(margin), 'memory_positive': bool(margin > 0),
            'cos_low': round(float(cos_low), 8), 'cos_sub': round(float(cos_sub), 8),
            'delta_cos': round(float(delta_cos), 8),
            'MAE_low': round(float(mae_low), 8), 'MAE_sub': round(float(mae_sub), 8),
            'MAE_delta': round(float(mae_delta), 8),
            'MAE_improvement': round(float(abs(mae_delta)), 8) if mae_delta != 0 else 0.0,
            'residual_on_improves': bool(delta_cos > 0),
        })

    per_layer_results[li] = layer_results

    if li % 4 == 0:
        r2 = next(x for x in layer_results if x['k_pct'] == 2.0)
        print(f"  layer {li}: k=2% margin={r2['margin']:+,}, delta_cos={r2['delta_cos']:+.5f}")

# Aggregate
print("\n=== AGGREGATE ===")
agg = {}
for k in [0, 0.5, 1, 2, 3]:
    lm  = [next(x for x in per_layer_results[li] if x['k_pct']==k)['margin'] for li in all_layers]
    ldc = [next(x for x in per_layer_results[li] if x['k_pct']==k)['delta_cos'] for li in all_layers]
    lmi = [next(x for x in per_layer_results[li] if x['k_pct']==k)['MAE_improvement'] for li in all_layers]
    lmp = [next(x for x in per_layer_results[li] if x['k_pct']==k)['memory_positive'] for li in all_layers]
    tb  = sum(next(x for x in per_layer_results[li] if x['k_pct']==k)['total_bytes'] for li in all_layers)
    agg[k] = {
        'k_pct': k, 'n_layers': n_layers, 'n_memory_positive': int(sum(lmp)),
        'total_bytes': int(tb), 'agg_budget': int(Q4_BUDGET_LAYER * n_layers),
        'aggregate_margin': int(sum(lm)), 'worst_margin': int(min(lm)),
        'avg_delta_cos': round(sum(ldc)/n_layers, 8),
        'total_mae_improvement': round(sum(lmi), 8),
        'all_memory_positive': all(lmp),
        'layer_margins': [int(x) for x in lm],
        'layer_delta_cos': [float(x) for x in ldc],
        'layer_mae_improvement': [float(x) for x in lmi],
    }
    print(f"  k={k}%: agg_margin={sum(lm):+,}, worst={min(lm):+,}, "
          f"n_mem_pos={sum(lmp)}/{n_layers}, avg_delta_cos={sum(ldc)/n_layers:+.5f}")

# Classification
best_k = None
for k in [2, 1, 0.5]:
    if agg[k]['all_memory_positive']:
        best_k = k
        classification = "PASS_ALL_LAYERS_MLP_Q2K_LOWK_POLICY_FOUND"
        break
if best_k is None:
    for k in [2, 1, 0.5]:
        if agg[k]['aggregate_margin'] > 0:
            best_k = k
            classification = "PARTIAL_LAYER_VARIANCE"
            break
if best_k is None:
    classification = "BLOCKED_ALL_POLICIES_FAIL_MEMORY"

print(f"\nClassification: {classification}, k={best_k}%")

# JSON
result = {
    'phase': '31AS',
    'classification': classification,
    'best_policy_k': float(best_k) if best_k else None,
    'n_layers_discovered': n_layers,
    'layer_indices': all_layers,
    'q2_byte_table': q2_metrics,
    'per_layer_results': {f'layer{li}': [to_native(r) for r in per_layer_results[li]] for li in all_layers},
    'aggregate': {str(k): to_native(agg[k]) for k in [0, 0.5, 1, 2, 3]},
    'mae_convention': {
        'MAE_delta_formula': 'MAE_sub - MAE_low',
        'MAE_delta_positive_means': 'MAE worsened',
        'MAE_delta_negative_means': 'MAE improved',
        'MAE_improvement': 'abs(MAE_delta); positive means improvement',
    },
}
json_path = f'{REPO}/results/PHASE31AS_ALL_LAYERS_MLP_Q2K_LOWK_SWEEP.json'
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)
size = os.path.getsize(json_path)
print(f"\nJSON: {json_path} — {size/1024:.1f} KB")

# MD
md = [
    "# Phase 31AS — All Available Layers Full MLP Q2_K + Low-k Residual Sweep",
    f"## Classification: **`{classification}`**",
    f"## Best Policy: k={best_k}%",
    f"## Layers Discovered: {n_layers} (indices {all_layers[0]}–{all_layers[-1]})",
    "",
    "## Aggregate Policy Table",
    "",
    "| k | agg_margin | worst_margin | n_mem_pos | avg_delta_cos |",
    "|---|------------|--------------|-----------|---------------|",
]
for k in [0, 0.5, 1, 2, 3]:
    r = agg[k]
    md.append(f"| {k} | {r['aggregate_margin']:+,} | {r['worst_margin']:+,} | "
              f"{r['n_memory_positive']}/{n_layers} | {r['avg_delta_cos']:+.5f} |")

md += ["", "## Per-Layer k=2% Summary", "",
       "| Layer | margin | delta_cos | MAE_delta | memory_positive |",
       "|-------|--------|-----------|-----------|----------------|"]
for li in all_layers:
    r = next(x for x in per_layer_results[li] if x['k_pct'] == 2.0)
    md.append(f"| {li} | {r['margin']:+,} | {r['delta_cos']:+.5f} | {r['MAE_delta']:+.6f} | {r['memory_positive']} |")

md_path = f'{REPO}/docs/PHASE31AS_ALL_LAYERS_MLP_Q2K_LOWK_SWEEP.md'
with open(md_path, 'w') as f:
    f.write("\n".join(md))

print(f"MD: {md_path}")
print(f"\nDONE. classification={classification} k={best_k}%")