#!/usr/bin/env python3
"""
Post-process 31BB: targeted k-sweep + aggregate mini-sweep + write JSON.
Fast: only needs needed layers (2, 20, 21, 22) for targeted + all 24 for aggregate.
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
QK_K = 256; Q2_BLOCK_BYTES = 84; HIDDEN = 896
K_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
AGG_K = [0.5, 0.75, 1.0, 1.5]

lib = ctypes.CDLL(LIB)
lib.quantize_row_q2_K_ref.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
lib.quantize_row_q2_K_ref.restype = None
lib.dequantize_row_q2_K.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.dequantize_row_q2_K.restype = None

def q2_encode(flat):
    n = flat.size; buf = np.zeros(n // QK_K * Q2_BLOCK_BYTES, dtype=np.uint8)
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

print("Loading model..."); t0 = time.time()
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')
print(f"Model loaded in {time.time()-t0:.1f}s")

def load_layer(li):
    families = {}
    for fam in ['ffn_up', 'ffn_gate', 'ffn_down']:
        t = next(x for x in reader.tensors if x.name == f'blk.{li}.{fam}.weight')
        W_ref = dequantize(t.data, t.tensor_type).astype(np.float32)
        buf = q2_encode(W_ref.flatten())
        W_low = q2_decode(buf, W_ref.size).reshape(W_ref.shape)
        families[fam] = {'W_ref': W_ref, 'W_low': W_low, 'R': W_ref - W_low, 'base_bytes': buf.nbytes}
    return families, sum(families[f]['base_bytes'] for f in families)

def mlp(X, Wg, Wu, Wd):
    return (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T

def eval_k(families, base_bytes, X, k, alpha=1.0):
    Y_ref = mlp(X, families['ffn_gate']['W_ref'], families['ffn_up']['W_ref'], families['ffn_down']['W_ref'])
    Y_low = mlp(X, families['ffn_gate']['W_low'], families['ffn_up']['W_low'], families['ffn_down']['W_low'])
    dec = {fam: decode_sdir(encode_sdir(alpha * families[fam]['R'], k)) for fam in families}
    Y_sub = mlp(X,
                families['ffn_gate']['W_low'] + dec['ffn_gate'],
                families['ffn_up']['W_low']   + dec['ffn_up'],
                families['ffn_down']['W_low']  + dec['ffn_down'])
    res_bytes = sum(len(encode_sdir(families[f]['R'], k)) for f in families)
    margin = Q4_BUDGET_LAYER - (base_bytes + res_bytes)
    cos_low = cos_sim(Y_ref, Y_low)
    cos_sub = cos_sim(Y_ref, Y_sub)
    mae_low = float(np.abs(Y_ref - Y_low).mean())
    mae_sub = float(np.abs(Y_ref - Y_sub).mean())
    dc = cos_sub - cos_low
    md = mae_sub - mae_low
    return {
        'k': k, 'margin': int(margin), 'residual_bytes': int(res_bytes),
        'cos_low': round(cos_low, 8), 'cos_sub': round(cos_sub, 8),
        'delta_cos': round(dc, 8),
        'cosine_improved': bool(dc > 0), 'cosine_nonnegative': bool(dc >= 0),
        'severe_regression': bool(dc < -0.05),
        'MAE_low': round(mae_low, 8), 'MAE_sub': round(mae_sub, 8),
        'MAE_delta': round(md, 8), 'MAE_improvement': round(abs(md), 8),
        'MAE_improved': bool(md < 0), 'memory_positive': bool(margin > 0),
    }

# TARGETED PAIRS
TARGET_PAIRS = [
    {'layer': 21, 'seed': 9, 'label': 'L21_seed9_SEVERE'},
    {'layer': 2,  'seed': 7, 'label': 'L2_seed7_mild'},
    {'layer': 21, 'seed': 0, 'label': 'L21_seed0_safe'},
    {'layer': 21, 'seed': 5, 'label': 'L21_seed5_safe'},
    {'layer': 21, 'seed': 14,'label': 'L21_seed14_safe'},
    {'layer': 20, 'seed': 9, 'label': 'L20_seed9'},
    {'layer': 22, 'seed': 9, 'label': 'L22_seed9'},
]

needed = list(set(p['layer'] for p in TARGET_PAIRS))
print(f"Loading layers: {sorted(needed)}")
cache = {li: load_layer(li) for li in needed}

print("\n=== TARGETED K SWEEP ===")
t0 = time.time()
results = {}
for pair in TARGET_PAIRS:
    li, s, label = pair['layer'], pair['seed'], pair['label']
    families, base_bytes = cache[li]
    X = np.random.default_rng(s).standard_normal((1, HIDDEN)).astype(np.float32)
    results[label] = [eval_k(families, base_bytes, X, k) for k in K_VALUES]
    print(f"  {label}: done")
print(f"Targeted sweep: {time.time()-t0:.1f}s")

# Print targeted tables
print()
for label, pr in results.items():
    print(f"\n=== {label} ===")
    for r in pr:
        status = []
        if r['severe_regression']: status.append('SEVERE')
        if r['cosine_improved']: status.append('cos_OK')
        if r['MAE_improved']: status.append('mae_OK')
        print(f"  k={r['k']:4.2f} cos_low={r['cos_low']:.6f} cos_sub={r['cos_sub']:.6f} delta_cos={r['delta_cos']:+.6f} cos_impr={str(r['cosine_improved']):5} severe={str(r['severe_regression']):5} MAE_delta={r['MAE_delta']:+.6f} mae_impr={str(r['MAE_improved']):5} mem={str(r['memory_positive'])}")

# AGGREGATE MINI-SWEEP
print("\n=== AGGREGATE MINI-SWEEP ===")
agg_cache = {li: load_layer(li) for li in range(24)}
agg_results = {k: [] for k in AGG_K}
SEEDS = list(range(16))

t0 = time.time()
for k in AGG_K:
    for li in range(24):
        families, base_bytes = agg_cache[li]
        for s in SEEDS:
            X = np.random.default_rng(s).standard_normal((1, HIDDEN)).astype(np.float32)
            r = eval_k(families, base_bytes, X, k)
            r['layer'] = li; r['seed'] = s
            agg_results[k].append(r)
    print(f"  k={k}: done")
print(f"Aggregate done in {time.time()-t0:.1f}s")

# Aggregate summary
print(f"\n{'k':>5} | {'n':>5} | {'cos_impr':>8} | {'cos_fail':>8} | {'severe':>6} | {'mae_impr':>8} | {'mae_fail':>8} | {'mem_pos':>8} | {'mean_dc':>10} | {'mean_mae':>10}")
print(f"{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
agg_summary = {}
for k in AGG_K:
    rs = agg_results[k]
    n = len(rs)
    cos_impr = sum(1 for r in rs if r['cosine_improved'])
    mae_impr = sum(1 for r in rs if r['MAE_improved'])
    mem_pos  = sum(1 for r in rs if r['memory_positive'])
    severe   = sum(1 for r in rs if r['severe_regression'])
    dc_arr = np.array([r['delta_cos'] for r in rs], dtype=np.float64)
    md_arr = np.array([r['MAE_delta'] for r in rs], dtype=np.float64)
    print(f"  {k:>5.2f} | {n:>5} | {cos_impr:>8} | {n-cos_impr:>8} | {severe:>6} | {mae_impr:>8} | {n-mae_impr:>8} | {mem_pos:>8} | {dc_arr.mean():>+10.6f} | {np.mean(np.abs(md_arr)):>10.6f}")
    agg_summary[k] = {
        'n': n, 'cos_impr': cos_impr, 'cos_fail': n-cos_impr,
        'severe': severe, 'mae_impr': mae_impr, 'mae_fail': n-mae_impr,
        'mem_pos': mem_pos, 'mean_delta_cos': round(float(dc_arr.mean()), 6),
        'median_delta_cos': round(float(np.sort(dc_arr)[n//2]), 6),
        'mean_MAE_improvement': round(float(np.mean(np.abs(md_arr))), 6),
        'worst_delta_cos': round(float(dc_arr.min()), 6),
    }

# ANALYSIS
print("\n=== LAYER 21 SEED 9 ANALYSIS ===")
l21s9 = results['L21_seed9_SEVERE']
k1_l21s9 = l21s9[5]  # k=1.0
print(f"  k=1 baseline: delta_cos={k1_l21s9['delta_cos']:+.6f}, severe={k1_l21s9['severe_regression']}, MAE_delta={k1_l21s9['MAE_delta']:+.6f}")
pos_k = [r for r in l21s9 if r['cosine_improved']]
nonsevere_k = [r for r in l21s9 if not r['severe_regression']]
print(f"  k values with cosine_improved: {[r['k'] for r in pos_k]}")
print(f"  k values with non-severe: {[r['k'] for r in nonsevere_k]}")
if pos_k:
    best_pos = max(pos_k, key=lambda r: r['delta_cos'])
    print(f"  Best positive: k={best_pos['k']}, delta_cos={best_pos['delta_cos']:+.6f}")
if nonsevere_k:
    best_ns = max(nonsevere_k, key=lambda r: r['delta_cos'])
    print(f"  Best non-severe: k={best_ns['k']}, delta_cos={best_ns['delta_cos']:+.6f}")

print("\n=== LAYER 2 SEED 7 ANALYSIS ===")
l2s7 = results['L2_seed7_mild']
k1_l2s7 = l2s7[5]
print(f"  k=1: delta_cos={k1_l2s7['delta_cos']:+.6f}, MAE_delta={k1_l2s7['MAE_delta']:+.6f}, cos_impr={k1_l2s7['cosine_improved']}, mae_impr={k1_l2s7['MAE_improved']}")
both_ok = [r for r in l2s7 if r['cosine_improved'] and r['MAE_improved']]
print(f"  k values with both cos_impr and mae_impr: {[r['k'] for r in both_ok]}")

# POLICY SELECTION
print("\n=== POLICY SELECTION ===")
k1_agg = agg_summary[1.0]
print(f"  k=1 baseline: severe={k1_agg['severe']}, cos_fail={k1_agg['cos_fail']}, mae_fail={k1_agg['mae_fail']}")
better = []
for k in AGG_K:
    s = agg_summary[k]
    better_severe = s['severe'] < k1_agg['severe']
    same_or_better_fail = s['cos_fail'] <= k1_agg['cos_fail']
    no_worse_mae = s['mae_fail'] <= k1_agg['mae_fail']
    if better_severe and same_or_better_fail and no_worse_mae:
        better.append((k, s))
        print(f"  k={k}: severe={s['severe']} (< {k1_agg['severe']}), cos_fail={s['cos_fail']} (<= {k1_agg['cos_fail']}), mae_fail={s['mae_fail']} (<= {k1_agg['mae_fail']}) -> BETTER")
    else:
        print(f"  k={k}: severe={s['severe']}, cos_fail={s['cos_fail']}, mae_fail={s['mae_fail']}")

# Classification
print("\n=== CLASSIFICATION ===")
if better:
    classification = "PASS_K_POLICY_IMPROVES_ROBUSTNESS"
elif any(agg_summary[k]['severe'] == 0 for k in AGG_K):
    classification = "PARTIAL_K_TRADEOFF"
else:
    classification = "PARTIAL_K_DOES_NOT_FIX_LAYER21"
print(f"  Classification: {classification}")

# Write JSON
out = {
    'phase': '31BB',
    'classification': classification,
    'k_values_tested': K_VALUES,
    'target_pairs': TARGET_PAIRS,
    'results': results,
    'agg_summary': agg_summary,
}
json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31BB_K_PARAMETER_SENSITIVITY.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nJSON: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"DONE. classification={classification}")