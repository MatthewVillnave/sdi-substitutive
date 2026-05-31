#!/usr/bin/env python3
"""
Phase 31BB — k-Parameter Sensitivity Sweep
Repo: sdi-substitutive
Targeted: layer 21 seed 9, layer 2 seed 7.
Safe: L21 seeds 0,5,14; L20 seed 9; L22 seed 9.
k in {0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}.
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
ALPHA = 1.0

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
    base_bytes = sum(families[f]['base_bytes'] for f in families)
    return families, base_bytes

def mlp(X, Wg, Wu, Wd):
    return (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T

def eval_k(families, base_bytes, X, k, alpha=ALPHA):
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
    delta_cos = cos_sub - cos_low
    mae_delta = mae_sub - mae_low
    return {
        'k': k,
        'margin': int(margin),
        'residual_bytes': int(res_bytes),
        'cos_low': round(cos_low, 8),
        'cos_sub': round(cos_sub, 8),
        'delta_cos': round(delta_cos, 8),
        'cosine_improved': bool(delta_cos > 0),
        'cosine_nonnegative': bool(delta_cos >= 0),
        'severe_regression': bool(delta_cos < -0.05),
        'MAE_low': round(mae_low, 8),
        'MAE_sub': round(mae_sub, 8),
        'MAE_delta': round(mae_delta, 8),
        'MAE_improvement': round(abs(mae_delta), 8),
        'MAE_improved': bool(mae_delta < 0),
        'memory_positive': bool(margin > 0),
    }

# Target pairs
TARGET_PAIRS = [
    {'layer': 21, 'seed': 9, 'label': 'L21_seed9_SEVERE'},
    {'layer': 2,  'seed': 7, 'label': 'L2_seed7_mild'},
    {'layer': 21, 'seed': 0, 'label': 'L21_seed0_safe'},
    {'layer': 21, 'seed': 5, 'label': 'L21_seed5_safe'},
    {'layer': 21, 'seed': 14,'label': 'L21_seed14_safe'},
    {'layer': 20, 'seed': 9, 'label': 'L20_seed9'},
    {'layer': 22, 'seed': 9, 'label': 'L22_seed9'},
]

# Load needed layers
NEEDED_LAYERS = list(set(p['layer'] for p in TARGET_PAIRS))
print(f"Loading layers: {NEEDED_LAYERS}")
layer_cache = {}
for li in NEEDED_LAYERS:
    t1 = time.time()
    families, base_bytes = load_layer(li)
    layer_cache[li] = (families, base_bytes)
    print(f"  Layer {li} loaded in {time.time()-t1:.1f}s")

# Run sweep
print(f"\n=== TARGETED K SWEEP ===")
results = {}
t0 = time.time()
for pair in TARGET_PAIRS:
    li = pair['layer']
    s = pair['seed']
    label = pair['label']
    families, base_bytes = layer_cache[li]
    rng = np.random.default_rng(s)
    X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
    pair_results = []
    for k in K_VALUES:
        r = eval_k(families, base_bytes, X, k)
        pair_results.append(r)
    results[label] = pair_results
    print(f"  {label}: done")

print(f"Sweep done in {time.time()-t0:.1f}s")

# Print tables
print("\n" + "="*100)
for label, pair_results in results.items():
    print(f"\n=== {label} ===")
    print(f"  {'k':>5} | {'margin':>10} | {'cos_low':>9} | {'cos_sub':>9} | {'delta_cos':>+10} | {'cos_impr':>8} | {'severe':>7} | {'MAE_delta':>+10} | {'MAE_impr':>8} | {'mem_pos':>7}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}")
    for r in pair_results:
        print(f"  {r['k']:>5.2f} | {r['margin']:>10} | {r['cos_low']:>9.6f} | {r['cos_sub']:>9.6f} | {r['delta_cos']:>+10.6f} | {str(r['cosine_improved']):>8} | {str(r['severe_regression']):>7} | {r['MAE_delta']:>+10.6f} | {r['MAE_improvement']:>8.6f} | {str(r['memory_positive']):>7}")

# Analysis
print("\n" + "="*100)
print("\n=== ANALYSIS ===")

# Layer 21 seed 9
l21s9 = results['L21_seed9_SEVERE']
print("\n-- Layer 21 seed 9 --")
best_cos_pos = None
best_cos_k = None
for r in l21s9:
    if r['cosine_improved']:
        if best_cos_pos is None or r['k'] > best_cos_k:
            best_cos_k = r['k']
            best_cos_pos = r
if best_cos_pos:
    print(f"  Best cosine-improving k: {best_cos_k} -> delta_cos={best_cos_pos['delta_cos']:+.6f}")
else:
    print(f"  NO k produces positive delta_cos")
    best_nonsevere = None
    for r in l21s9:
        if not r['severe_regression']:
            if best_nonsevere is None or r['delta_cos'] > best_nonsevere['delta_cos']:
                best_nonsevere = r
    if best_nonsevere:
        print(f"  Best non-severe k: {best_nonsevere['k']} -> delta_cos={best_nonsevere['delta_cos']:+.6f}, severe={best_nonsevere['severe_regression']}")
    else:
        print(f"  ALL k produce severe regression for L21 seed 9")

print(f"  k=1 baseline: delta_cos={l21s9[5]['delta_cos']:+.6f}, severe={l21s9[5]['severe_regression']}, MAE_delta={l21s9[5]['MAE_delta']:+.6f}")

# Layer 2 seed 7
l2s7 = results['L2_seed7_mild']
print("\n-- Layer 2 seed 7 --")
print(f"  k=1: delta_cos={l2s7[5]['delta_cos']:+.6f}, MAE_delta={l2s7[5]['MAE_delta']:+.6f}, cosine_improved={l2s7[5]['cosine_improved']}, MAE_improved={l2s7[5]['MAE_improved']}")
for r in l2s7:
    if r['cosine_improved'] and r['MAE_improved']:
        print(f"  k={r['k']}: delta_cos={r['delta_cos']:+.6f}, MAE_delta={r['MAE_delta']:+.6f} -- PASS both")

# Safe cases check
print("\n-- Safe cases at k=1 --")
for label in ['L21_seed0_safe', 'L21_seed5_safe', 'L21_seed14_safe', 'L20_seed9', 'L22_seed9']:
    r_k1 = results[label][5]  # k=1.0
    print(f"  {label}: delta_cos={r_k1['delta_cos']:+.6f}, severe={r_k1['severe_regression']}, MAE_improved={r_k1['MAE_improved']}")

# Aggregate mini-sweep: all 24 layers x seeds 0-15 at k in {0.5, 0.75, 1.0, 1.5}
print("\n" + "="*100)
print("\n=== AGGREGATE MINI-SWEEP ===")
AGG_K_VALUES = [0.5, 0.75, 1.0, 1.5]
agg_results = {k: [] for k in AGG_K_VALUES}

# Load all 24 layers
print("Loading all 24 layers...")
all_layer_cache = {}
for li in range(24):
    families, base_bytes = load_layer(li)
    all_layer_cache[li] = (families, base_bytes)

SEEDS = list(range(16))
print(f"Running {24*16*len(AGG_K_VALUES)} evaluations...")
t0 = time.time()
for k in AGG_K_VALUES:
    count = 0
    for li in range(24):
        families, base_bytes = all_layer_cache[li]
        for s in SEEDS:
            rng = np.random.default_rng(s)
            X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
            r = eval_k(families, base_bytes, X, k)
            r['layer'] = li; r['seed'] = s
            agg_results[k].append(r)
            count += 1
    print(f"  k={k}: {count} evals done")
print(f"Aggregate done in {time.time()-t0:.1f}s")

# Aggregate stats
print(f"\n{'k':>5} | {'n':>5} | {'cos_impr':>8} | {'cos_fail':>8} | {'severe':>6} | {'mae_impr':>8} | {'mae_fail':>8} | {'mem_pos':>8} | {'mean_dc':>10} | {'mean_mae':>10}")
print(f"{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
agg_summary = {}
for k in AGG_K_VALUES:
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
        'mean_MAE_improvement': round(float(np.mean(np.abs(md_arr))), 6),
        'worst_delta_cos': round(float(dc_arr.min()), 6),
    }

# Policy selection
print("\n=== POLICY SELECTION ===")
k1 = agg_summary[1.0]
best_policy = None
best_score = None
for k in AGG_K_VALUES:
    s = agg_summary[k]
    # Score: lower is better; penalize severe regressions heavily
    score = (s['cos_fail'] * 10 + s['severe'] * 100 + s['mae_fail'] * 5 - s['mean_delta_cos'] * 100)
    print(f"  k={k}: cos_fail={s['cos_fail']}, severe={s['severe']}, mae_fail={s['mae_fail']}, score={score:.2f}")
    if best_score is None or score < best_score:
        best_score = score
        best_policy = k
print(f"  Best by score: k={best_policy}")

# Check: does any k eliminate severe regressions?
for k in AGG_K_VALUES:
    if agg_summary[k]['severe'] == 0 and agg_summary[k]['cos_fail'] <= k1['cos_fail']:
        print(f"  k={k}: SEVERE=0, cos_fail={agg_summary[k]['cos_fail']} <= k1 cos_fail={k1['cos_fail']} -> CANDIDATE")

# Classification
print("\n=== CLASSIFICATION ===")
k_baseline = agg_summary[1.0]
severe_at_k1 = k_baseline['severe']
fail_at_k1 = k_baseline['cos_fail']
candidate_k = None
for k in AGG_K_VALUES:
    s = agg_summary[k]
    if s['severe'] < severe_at_k1 or (s['severe'] == 0 and s['cos_fail'] <= fail_at_k1):
        candidate_k = k

if candidate_k and candidate_k > 0:
    classification = "PASS_K_POLICY_IMPROVES_ROBUSTNESS"
elif any(agg_summary[k]['severe'] == 0 for k in AGG_K_VALUES):
    # some k eliminates severe but doesn't beat k1 overall
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
    'aggregate_mini_sweep': {k: agg_results[k] for k in AGG_K_VALUES},
}

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31BB_K_PARAMETER_SENSITIVITY.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nJSON: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"DONE. classification={classification}")