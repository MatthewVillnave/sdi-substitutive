#!/usr/bin/env python3
"""
Post-process 31AY: compute correlations and write JSON.
Uses the raw seed data computed in the main sweep.
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
def norm_vec(v): return float(np.linalg.norm(v.flatten()))

print("Loading model...")
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')

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

def eval_layer(families, base_bytes, X, k, alpha):
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
    nr = norm_vec(Y_ref)
    nl = norm_vec(Y_low)
    ns = norm_vec(Y_sub)
    return {
        'margin': int(margin),
        'cos_low': round(cos_low, 8),
        'cos_sub': round(cos_sub, 8),
        'delta_cos': round(cos_sub - cos_low, 8),
        'MAE_low': round(mae_low, 8),
        'MAE_sub': round(mae_sub, 8),
        'MAE_delta': round(mae_sub - mae_low, 8),
        'MAE_improvement': round(abs(mae_sub - mae_low), 8),
        'MAE_improved': bool(mae_sub < mae_low),
        'norm_ref': round(nr, 6),
        'norm_low': round(nl, 6),
        'norm_sub': round(ns, 6),
        'norm_ratio_sub_ref': round(ns / nr, 8),
        'norm_sub_minus_norm_ref': round(ns - nr, 6),
        'cosine_improved': bool(cos_sub > cos_low),
        'memory_positive': bool(margin > 0),
    }

f21, bb21 = load_layer(21)
f20, bb20 = load_layer(20)
f22, bb22 = load_layer(22)

def run_sweep(families, base_bytes, max_seed, label):
    t0 = time.time()
    results = []
    for s in range(max_seed):
        rng = np.random.default_rng(s)
        X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        r = eval_layer(families, base_bytes, X, k=1.0, alpha=1.0)
        r['seed'] = s
        results.append(r)
    print(f"  {label} sweep 0-{max_seed-1} done in {time.time()-t0:.1f}s ({max_seed} seeds)")
    return results

print("\n=== Re-running sweeps (fast: 96 seeds total) ===")
t0 = time.time()
layer21_results = run_sweep(f21, bb21, 64, "L21")
layer20_results = run_sweep(f20, bb20, 32, "L20")
layer22_results = run_sweep(f22, bb22, 32, "L22")
print(f"Total sweep time: {time.time()-t0:.1f}s")

# Anchors
ANCHORS = [0, 5, 9]
anchor_results = {s: layer21_results[s] for s in ANCHORS}
print(f"\nAnchors: seed=0 cos_impr={anchor_results[0]['cosine_improved']}, seed=5 cos_impr={anchor_results[5]['cosine_improved']}, seed=9 cos_impr={anchor_results[9]['cosine_improved']}")

# Aggregate
def aggregate(results, name):
    n = len(results)
    dc = np.array([r['delta_cos'] for r in results])
    cos_pos = sum(1 for r in results if r['cosine_improved'])
    mae_pos = sum(1 for r in results if r['MAE_improved'])
    mem_pos = sum(1 for r in results if r['memory_positive'])
    severe = sum(1 for r in results if r['delta_cos'] < -0.05)
    mild = sum(1 for r in results if -0.05 <= r['delta_cos'] < 0)
    mild_pos = sum(1 for r in results if 0 <= r['delta_cos'] < 0.02)
    strong_pos = sum(1 for r in results if r['delta_cos'] >= 0.02)
    worst = min(results, key=lambda r: r['delta_cos'])
    best = max(results, key=lambda r: r['delta_cos'])
    sorted_dc = np.sort(dc)
    median_dc = sorted_dc[n//2]
    mean_dc = float(dc.mean())
    mean_mae_imp = float(np.mean(np.abs([r['MAE_delta'] for r in results])))
    print(f"  {name}: n={n}, cos_pos={cos_pos}/{n} ({100*cos_pos/n:.1f}%), mae_pos={mae_pos}/{n}, cos_fail={100*(n-cos_pos)/n:.1f}%, worst=seed{worst['seed']}({worst['delta_cos']:+.5f}), best=seed{best['seed']}({best['delta_cos']:+.5f}), mean={mean_dc:+.5f}, median={median_dc:+.5f}")
    return {
        'n_seeds': n,
        'n_cosine_positive': cos_pos,
        'n_MAE_improving': mae_pos,
        'n_memory_positive': mem_pos,
        'cosine_failure_rate': round(100*(n-cos_pos)/n, 2),
        'mae_failure_rate': round(100*(n-mae_pos)/n, 2),
        'severe_regressions': severe,
        'mild_regressions': mild,
        'mild_positive': mild_pos,
        'strong_positive': strong_pos,
        'worst_seed': int(worst['seed']),
        'worst_delta_cos': worst['delta_cos'],
        'best_seed': int(best['seed']),
        'best_delta_cos': best['delta_cos'],
        'mean_delta_cos': round(mean_dc, 6),
        'median_delta_cos': round(float(median_dc), 6),
        'min_delta_cos': round(float(dc.min()), 6),
        'max_delta_cos': round(float(dc.max()), 6),
        'mean_MAE_improvement': round(mean_mae_imp, 6),
    }

agg21 = aggregate(layer21_results, "L21(0-63)")
agg20 = aggregate(layer20_results, "L20(0-31)")
agg22 = aggregate(layer22_results, "L22(0-31)")

# Correlations for layer 21
dc21 = np.array([r['delta_cos'] for r in layer21_results], dtype=np.float64)
nr21 = np.array([r['norm_ratio_sub_ref'] for r in layer21_results], dtype=np.float64)
dnr21 = np.array([r['norm_sub_minus_norm_ref'] for r in layer21_results], dtype=np.float64)
mae_imp21 = np.array([abs(r['MAE_delta']) for r in layer21_results], dtype=np.float64)
cos_low21 = np.array([r['cos_low'] for r in layer21_results], dtype=np.float64)
nr_ref21 = np.array([r['norm_ref'] for r in layer21_results], dtype=np.float64)

def pearson(x, y):
    xm = x - x.mean(); ym = y - y.mean()
    denom = np.sqrt(np.sum(xm**2)) * np.sqrt(np.sum(ym**2))
    return float(np.dot(xm, ym) / (denom + 1e-12))

corr_norm_ratio = pearson(nr21, dc21)
corr_dnr = pearson(dnr21, dc21)
corr_mae_imp = pearson(mae_imp21, dc21)
corr_cos_low = pearson(cos_low21, dc21)
corr_norm_ref = pearson(nr_ref21, dc21)
print(f"\n  Correlations (L21):")
print(f"    corr(delta_cos, norm_ratio_sub_ref) = {corr_norm_ratio:+.4f}")
print(f"    corr(delta_cos, norm_sub-norm_ref) = {corr_dnr:+.4f}")
print(f"    corr(delta_cos, |MAE_delta|) = {corr_mae_imp:+.4f}")
print(f"    corr(delta_cos, cos_low) = {corr_cos_low:+.4f}")
print(f"    corr(delta_cos, norm_ref) = {corr_norm_ref:+.4f}")

# Classification
fr = (len(layer21_results) - agg21['n_cosine_positive']) / len(layer21_results)
neighbor_fail = (agg20['cosine_failure_rate'] >= 10) or (agg22['cosine_failure_rate'] >= 10)
if fr == 0:
    classification = "PASS_LAYER21_ROBUST_EXCEPT_KNOWN_SEED9"
elif fr <= 0.10:
    classification = "PARTIAL_LAYER21_RARE_ACTIVATION_OUTLIER"
elif fr < 0.50:
    classification = "PARTIAL_LAYER21_ACTIVATION_SENSITIVE"
else:
    classification = "PARTIAL_LAYER21_SYSTEMATIC_CONFLICT"
if neighbor_fail and fr < 0.50:
    classification = "PARTIAL_MULTI_LAYER_SENSITIVITY"
print(f"\n  Classification: {classification}")

# Write JSON
out = {
    'phase': '31AY',
    'classification': classification,
    'anchors': {str(s): anchor_results[s] for s in ANCHORS},
    'layer21_sweep': layer21_results,
    'layer20_sweep': layer20_results,
    'layer22_sweep': layer22_results,
    'layer21_aggregate': agg21,
    'layer20_aggregate': agg20,
    'layer22_aggregate': agg22,
    'correlations': {
        'corr_delta_cos_norm_ratio': round(corr_norm_ratio, 4),
        'corr_delta_cos_norm_sub_minus_ref': round(corr_dnr, 4),
        'corr_delta_cos_MAE_improvement': round(corr_mae_imp, 4),
        'corr_delta_cos_cos_low': round(corr_cos_low, 4),
        'corr_delta_cos_norm_ref': round(corr_norm_ref, 4),
    },
    'mae_convention': {
        'MAE_delta_formula': 'MAE_sub - MAE_low; negative = MAE improved',
    },
}

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31AY_LAYER21_ACTIVATION_SPACE_SENSITIVITY.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nJSON written: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"DONE. classification={classification}")