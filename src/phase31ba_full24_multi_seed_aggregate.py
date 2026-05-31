#!/usr/bin/env python3
"""
Phase 31BA — Full 24-Layer Multi-Seed Aggregate Characterization
Repo: sdi-substitutive
24 layers x 16 seeds = 384 seed-layer pairs.
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
def norm_vec(v): return float(np.linalg.norm(v.flatten()))

print("Loading model..."); t_model = time.time()
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')
print(f"Model loaded in {time.time()-t_model:.1f}s")

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

def eval_full(families, base_bytes, X, k=1.0, alpha=1.0):
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
    nr = norm_vec(Y_ref); nl = norm_vec(Y_low); ns = norm_vec(Y_sub)
    res_norm = norm_vec(Y_sub - Y_low)
    delta_cos = cos_sub - cos_low
    mae_delta = mae_sub - mae_low
    return {
        'layer': None,  # set by caller
        'seed': None,
        'margin': int(margin),
        'cos_low': round(cos_low, 8),
        'cos_sub': round(cos_sub, 8),
        'delta_cos': round(delta_cos, 8),
        'MAE_low': round(mae_low, 8),
        'MAE_sub': round(mae_sub, 8),
        'MAE_delta': round(mae_delta, 8),
        'MAE_improvement': round(abs(mae_delta), 8),
        'MAE_improved': bool(mae_delta < 0),
        'cosine_improved': bool(delta_cos > 0),
        'cosine_nonnegative': bool(delta_cos >= 0),
        'severe_regression': bool(delta_cos < -0.05),
        'memory_positive': bool(margin > 0),
        'norm_ref': round(nr, 6),
        'norm_low': round(nl, 6),
        'norm_sub': round(ns, 6),
        'residual_update_norm': round(res_norm, 6),
    }

# Load all 24 layers
t0 = time.time()
print("Loading all 24 layers...")
layer_data = {}
for li in range(24):
    families, base_bytes = load_layer(li)
    layer_data[li] = (families, base_bytes)
    print(f"  Layer {li:2d} loaded")
print(f"All layers loaded in {time.time()-t0:.1f}s")

# Run sweep
SEEDS = list(range(16))  # 0-15
ALL_LAYERS = list(range(24))
TOTAL_PAIRS = len(ALL_LAYERS) * len(SEEDS)
print(f"\nRunning sweep: {len(ALL_LAYERS)} layers x {len(SEEDS)} seeds = {TOTAL_PAIRS} pairs")

t0 = time.time()
results = []
for li in ALL_LAYERS:
    families, base_bytes = layer_data[li]
    layer_times = []
    for s in SEEDS:
        t1 = time.time()
        rng = np.random.default_rng(s)
        X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        r = eval_full(families, base_bytes, X)
        r['layer'] = li; r['seed'] = s
        results.append(r)
        layer_times.append(time.time() - t1)
    mean_t = np.mean(layer_times)
    print(f"  Layer {li:2d}: {len(SEEDS)} seeds done in {sum(layer_times):.1f}s ({mean_t*1000:.0f}ms/seed)")
print(f"Total sweep time: {time.time()-t0:.1f}s ({TOTAL_PAIRS/(time.time()-t0):.1f} pairs/sec)")

# Aggregate
print("\n=== AGGREGATE ===")
dc_arr = np.array([r['delta_cos'] for r in results], dtype=np.float64)
md_arr = np.array([r['MAE_delta'] for r in results], dtype=np.float64)
n = len(results)
cos_impr = sum(1 for r in results if r['cosine_improved'])
cos_nonneg = sum(1 for r in results if r['cosine_nonnegative'])
mae_impr = sum(1 for r in results if r['MAE_improved'])
mem_pos = sum(1 for r in results if r['memory_positive'])
severe = sum(1 for r in results if r['severe_regression'])
worst = min(results, key=lambda r: r['delta_cos'])
best = max(results, key=lambda r: r['delta_cos'])
sorted_dc = np.sort(dc_arr)
sorted_mae_imp = np.sort(np.abs(md_arr))
agg = {
    'total_pairs': n,
    'n_memory_positive': mem_pos,
    'n_cosine_improved': cos_impr,
    'n_cosine_nonnegative': cos_nonneg,
    'n_MAE_improved': mae_impr,
    'n_severe_regressions': severe,
    'cosine_failure_rate': round(100*(n-cos_impr)/n, 2),
    'severe_regression_rate': round(100*severe/n, 2),
    'MAE_failure_rate': round(100*(n-mae_impr)/n, 2),
    'worst_layer': worst['layer'],
    'worst_seed': worst['seed'],
    'worst_delta_cos': worst['delta_cos'],
    'best_layer': best['layer'],
    'best_seed': best['seed'],
    'best_delta_cos': best['delta_cos'],
    'mean_delta_cos': round(float(dc_arr.mean()), 6),
    'median_delta_cos': round(float(sorted_dc[n//2]), 6),
    'min_delta_cos': round(float(dc_arr.min()), 6),
    'max_delta_cos': round(float(dc_arr.max()), 6),
    'mean_MAE_improvement': round(float(np.mean(np.abs(md_arr))), 6),
    'median_MAE_improvement': round(float(sorted_mae_imp[n//2]), 6),
    'total_margin': sum(r['margin'] for r in results),
}
print(f"  total_pairs={n}")
print(f"  cosine_improved={cos_impr}/{n} ({100*cos_impr/n:.1f}%)")
print(f"  cosine_nonnegative={cos_nonneg}/{n} ({100*cos_nonneg/n:.1f}%)")
print(f"  MAE_improved={mae_impr}/{n} ({100*mae_impr/n:.1f}%)")
print(f"  memory_positive={mem_pos}/{n} ({100*mem_pos/n:.1f}%)")
print(f"  severe_regressions={severe} ({100*severe/n:.2f}%)")
print(f"  worst=layer{worst['layer']} seed{worst['seed']} delta_cos={worst['delta_cos']}")
print(f"  mean_delta_cos={agg['mean_delta_cos']:+.6f}, median={agg['median_delta_cos']:+.6f}")
print(f"  mean_MAE_improvement={agg['mean_MAE_improvement']}")

# By-layer aggregate
print("\n=== BY-LAYER ===")
by_layer = {}
for li in ALL_LAYERS:
    lr = [r for r in results if r['layer'] == li]
    dc_l = np.array([r['delta_cos'] for r in lr], dtype=np.float64)
    md_l = np.array([r['MAE_delta'] for r in lr], dtype=np.float64)
    ci = sum(1 for r in lr if r['cosine_improved'])
    cn = sum(1 for r in lr if r['cosine_nonnegative'])
    mi = sum(1 for r in lr if r['MAE_improved'])
    sev = sum(1 for r in lr if r['severe_regression'])
    w = min(lr, key=lambda r: r['delta_cos'])
    b = max(lr, key=lambda r: r['delta_cos'])
    # classify
    if sev == 0 and (len(lr) - ci) == 0:
        cls = 'robust'
    elif sev == 0 and (len(lr) - ci) <= 2:
        cls = 'sensitive'
    else:
        cls = 'problematic'
    by_layer[li] = {
        'n_seeds': len(lr),
        'n_cosine_improved': ci,
        'n_cosine_nonnegative': cn,
        'n_MAE_improved': mi,
        'n_severe_regressions': sev,
        'worst_delta_cos': w['delta_cos'],
        'best_delta_cos': b['delta_cos'],
        'mean_delta_cos': round(float(dc_l.mean()), 6),
        'mean_MAE_improvement': round(float(np.mean(np.abs(md_l))), 6),
        'class': cls,
    }
    print(f"  L{li:2d}: cos_impr={ci}/{len(lr)}, severe={sev}, worst=seed{w['seed']}({w['delta_cos']:+.5f}), mean_dc={by_layer[li]['mean_delta_cos']:+.5f}, class={cls}")

# By-seed aggregate
print("\n=== BY-SEED ===")
by_seed = {}
for s in SEEDS:
    sr = [r for r in results if r['seed'] == s]
    dc_s = np.array([r['delta_cos'] for r in sr], dtype=np.float64)
    md_s = np.array([r['MAE_delta'] for r in sr], dtype=np.float64)
    ci = sum(1 for r in sr if r['cosine_improved'])
    mi = sum(1 for r in sr if r['MAE_improved'])
    sev = sum(1 for r in sr if r['severe_regression'])
    w = min(sr, key=lambda r: r['delta_cos'])
    by_seed[s] = {
        'n_layers_cosine_improved': ci,
        'n_layers_MAE_improved': mi,
        'n_severe_regressions': sev,
        'worst_layer': w['layer'],
        'worst_delta_cos': w['delta_cos'],
        'mean_delta_cos': round(float(dc_s.mean()), 6),
        'mean_MAE_improvement': round(float(np.mean(np.abs(md_s))), 6),
    }
    print(f"  seed{s:2d}: cos_impr={ci}/24, severe={sev}, worst=L{w['layer']}({w['delta_cos']:+.5f}), mean_dc={by_seed[s]['mean_delta_cos']:+.5f}")

# Failure concentration
print("\n=== FAILURE CONCENTRATION ===")
# Layers with cosine failures
layers_with_failures = [(li, by_layer[li]['n_seeds'] - by_layer[li]['n_cosine_improved'])
                          for li in ALL_LAYERS
                          if by_layer[li]['n_seeds'] - by_layer[li]['n_cosine_improved'] > 0]
layers_with_failures.sort(key=lambda x: -x[1])
total_failures = sum(f for _, f in layers_with_failures)
print(f"  Total cosine failure pairs: {total_failures}")
for li, f in layers_with_failures:
    print(f"  Layer {li}: {f} failures ({100*f/by_layer[li]['n_seeds']:.0f}% of layer seeds)")

# Layers with severe regressions
severe_layers = [(li, by_layer[li]['n_severe_regressions']) for li in ALL_LAYERS if by_layer[li]['n_severe_regressions'] > 0]
severe_layers.sort(key=lambda x: -x[1])
print(f"\n  Severe regressions by layer:")
for li, s in severe_layers:
    print(f"  Layer {li}: {s} severe")

# Correlations
print("\n=== CORRELATIONS ===")
cl_arr = np.array([r['cos_low'] for r in results], dtype=np.float64)
dc_arr = np.array([r['delta_cos'] for r in results], dtype=np.float64)
nr_arr = np.array([r['residual_update_norm'] for r in results], dtype=np.float64)
mae_imp_arr = np.array([r['MAE_improvement'] for r in results], dtype=np.float64)

def pearson(x, y):
    xm = x - x.mean(); ym = y - y.mean()
    return float(np.dot(xm, ym) / (np.sqrt(np.sum(xm**2)) * np.sqrt(np.sum(ym**2)) + 1e-12))

r1 = pearson(dc_arr, cl_arr)
r2 = pearson(dc_arr, mae_imp_arr)
r3 = pearson(dc_arr, nr_arr)
print(f"  corr(delta_cos, cos_low) = {r1:+.4f}")
print(f"  corr(delta_cos, MAE_improvement) = {r2:+.4f}")
print(f"  corr(delta_cos, residual_update_norm) = {r3:+.4f}")

# Classification
print("\n=== CLASSIFICATION ===")
n_problematic = sum(1 for li in ALL_LAYERS if by_layer[li]['class'] == 'problematic')
n_sensitive = sum(1 for li in ALL_LAYERS if by_layer[li]['class'] == 'sensitive')
severe_total = agg['n_severe_regressions']
cos_fail_rate = agg['cosine_failure_rate']
mae_fail_rate = agg['MAE_failure_rate']
if severe_total == 0 and cos_fail_rate < 5:
    classification = "PASS_AGGREGATE_MULTI_SEED_ROBUST"
elif n_problematic == 0 and n_sensitive <= 2 and severe_total <= 2:
    classification = "PARTIAL_AGGREGATE_STRONG_WITH_ISOLATED_SENSITIVITY"
elif n_problematic >= 3:
    classification = "PARTIAL_MULTI_LAYER_SENSITIVITY"
elif by_layer[21]['n_severe_regressions'] >= severe_total * 0.8 and n_problematic <= 1:
    classification = "PARTIAL_LAYER21_DOMINANT_SENSITIVITY"
else:
    classification = "PARTIAL_AGGREGATE_STRONG_WITH_ISOLATED_SENSITIVITY"

print(f"  problematic_layers={n_problematic}, sensitive_layers={n_sensitive}, severe_total={severe_total}")
print(f"  cos_fail_rate={cos_fail_rate}%, mae_fail_rate={mae_fail_rate}%")
print(f"  Classification: {classification}")

# Write JSON
out = {
    'phase': '31BA',
    'classification': classification,
    'seeds': SEEDS,
    'n_seeds': len(SEEDS),
    'n_layers': 24,
    'total_pairs': n,
    'aggregate': agg,
    'by_layer': by_layer,
    'by_seed': by_seed,
    'failure_concentration': {
        'layers_with_failures': layers_with_failures,
        'severe_layers': severe_layers,
    },
    'correlations': {
        'corr_delta_cos_cos_low': round(r1, 4),
        'corr_delta_cos_MAE_improvement': round(r2, 4),
        'corr_delta_cos_residual_update_norm': round(r3, 4),
    },
    'results': results,
}

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31BA_FULL24_MULTI_SEED_AGGREGATE.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nJSON: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"DONE. classification={classification}")