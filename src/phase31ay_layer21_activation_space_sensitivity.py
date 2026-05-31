#!/usr/bin/env python3
"""
Phase 31AY — Layer 21 Activation-Space Sensitivity Map
Repo: sdi-substitutive
64 seeds (0-63) for layer 21 + 32 seeds (0-31) for layers 20 and 22.
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
t0 = time.time()
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')
print(f"Loaded in {time.time()-t0:.1f}s")

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

# Load layer 21
t0 = time.time()
f21, bb21 = load_layer(21)
print(f"Layer 21 loaded in {time.time()-t0:.1f}s")

# ================================================================
# STEP 1: Anchor verification
# ================================================================
print("\n=== STEP 1: Anchor verification (layer 21, k=1%) ===\n")
ANCHORS = [0, 5, 9]
anchor_results = {}
for s in ANCHORS:
    rng = np.random.default_rng(s)
    X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
    r = eval_layer(f21, bb21, X, k=1.0, alpha=1.0)
    r['seed'] = s
    anchor_results[s] = r
    print(f"  seed={s}: cos_low={r['cos_low']:.6f}, cos_sub={r['cos_sub']:.6f}, delta_cos={r['delta_cos']:+.6f}, MAE_delta={r['MAE_delta']:+.6f}, margin={r['margin']:,}, cosine_improved={r['cosine_improved']}")
print(f"  Anchors: seed=0 cosine_improved={anchor_results[0]['cosine_improved']} (expected True), seed=5={anchor_results[5]['cosine_improved']} (expected True), seed=9={anchor_results[9]['cosine_improved']} (expected False)")

# ================================================================
# STEP 2: Layer 21 full seed sweep 0-63
# ================================================================
print("\n=== STEP 2: Layer 21 seed sweep 0-63 ===\n")
t0 = time.time()
SWEEP_MAX = 64
layer21_results = []
for s in range(SWEEP_MAX):
    rng = np.random.default_rng(s)
    X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
    r = eval_layer(f21, bb21, X, k=1.0, alpha=1.0)
    r['seed'] = s
    layer21_results.append(r)
    if s % 16 == 15:
        elapsed = time.time() - t0
        eta = elapsed / (s+1) * (SWEEP_MAX - s - 1)
        print(f"  seed {s}/63 done, elapsed={elapsed:.1f}s, ETA={eta:.1f}s")
print(f"\n  Full sweep 0-63 done in {time.time()-t0:.1f}s")

# ================================================================
# STEP 3: Layer 20 and 22 lighter sweep 0-31
# ================================================================
print("\n=== STEP 3: Layers 20 and 22 sweep 0-31 ===\n")
t0 = time.time()
f20, bb20 = load_layer(20)
f22, bb22 = load_layer(22)
layer20_results = []
layer22_results = []
for s in range(32):
    rng20 = np.random.default_rng(s)
    X20 = rng20.standard_normal((1, HIDDEN)).astype(np.float32)
    r20 = eval_layer(f20, bb20, X20, k=1.0, alpha=1.0)
    r20['seed'] = s
    layer20_results.append(r20)

    rng22 = np.random.default_rng(s)
    X22 = rng22.standard_normal((1, HIDDEN)).astype(np.float32)
    r22 = eval_layer(f22, bb22, X22, k=1.0, alpha=1.0)
    r22['seed'] = s
    layer22_results.append(r22)
    if s % 8 == 7:
        print(f"  seed {s}/31 done")
print(f"  Layers 20+22 sweep done in {time.time()-t0:.1f}s")

# ================================================================
# STEP 4: Aggregate analysis
# ================================================================
print("\n=== STEP 4: Aggregate analysis ===\n")

def aggregate(results, name):
    n = len(results)
    dc = [r['delta_cos'] for r in results]
    md = [r['MAE_delta'] for r in results]
    cos_pos = sum(1 for r in results if r['cosine_improved'])
    mae_pos = sum(1 for r in results if r['MAE_improved'])
    mem_pos = sum(1 for r in results if r['memory_positive'])
    severe = sum(1 for r in results if r['delta_cos'] < -0.05)
    mild = sum(1 for r in results if -0.05 <= r['delta_cos'] < 0)
    mild_pos = sum(1 for r in results if 0 <= r['delta_cos'] < 0.02)
    strong_pos = sum(1 for r in results if r['delta_cos'] >= 0.02)
    worst = min(results, key=lambda r: r['delta_cos'])
    best = max(results, key=lambda r: r['delta_cos'])
    sorted_dc = sorted(dc)
    median_dc = sorted_dc[n//2]
    mean_dc = sum(dc)/n
    mean_mae_imp = sum(abs(r['MAE_delta']) for r in results)/n
    print(f"  {name}:")
    print(f"    n_seeds={n}")
    print(f"    n_cosine_positive={cos_pos}/{n} ({100*cos_pos/n:.1f}%)")
    print(f"    n_MAE_improving={mae_pos}/{n} ({100*mae_pos/n:.1f}%)")
    print(f"    n_memory_positive={mem_pos}/{n} ({100*mem_pos/n:.1f}%)")
    print(f"    cosine failure rate={100*(n-cos_pos)/n:.1f}%")
    print(f"    severe (<-0.05)={severe}, mild=[-0.05,0)={mild}, mild_pos=[0,0.02)={mild_pos}, strong_pos=>=0.02={strong_pos}")
    print(f"    worst seed={worst['seed']} (delta_cos={worst['delta_cos']:+.6f})")
    print(f"    best seed={best['seed']} (delta_cos={best['delta_cos']:+.6f})")
    print(f"    mean_delta_cos={mean_dc:+.6f}, median={median_dc:+.6f}")
    print(f"    mean_MAE_improvement={mean_mae_imp:.6f}")
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
        'worst_seed': worst['seed'],
        'worst_delta_cos': worst['delta_cos'],
        'best_seed': best['seed'],
        'best_delta_cos': best['delta_cos'],
        'mean_delta_cos': round(mean_dc, 6),
        'median_delta_cos': round(median_dc, 6),
        'min_delta_cos': round(min(dc), 6),
        'max_delta_cos': round(max(dc), 6),
        'mean_MAE_improvement': round(mean_mae_imp, 6),
    }

agg21 = aggregate(layer21_results, "Layer 21 (seeds 0-63)")
agg20 = aggregate(layer20_results, "Layer 20 (seeds 0-31)")
agg22 = aggregate(layer22_results, "Layer 22 (seeds 0-31)")

# ================================================================
# STEP 5: Geometry correlation
# ================================================================
print("\n=== STEP 5: Geometry correlations (layer 21, 0-63) ===\n")
nr21 = np.array([r['norm_ratio_sub_ref'] for r in layer21_results])
dnr21 = np.array([r['norm_sub_minus_norm_ref'] for r in layer21_results])
dcos21 = np.array([r['delta_cos'] for r in layer21_results])
mae_imp21 = np.array([abs(r['MAE_delta']) for r in layer21_results])
cos_low21 = np.array([r['cos_low'] for r in layer21_results])
nr_ref21 = np.array([r['norm_ref'] for r in layer21_results])

def pearson(x, y):
    xm = x - x.mean(); ym = y - y.mean()
    denom = np.sqrt((xm*xm).sum()) * np.sqrt((ym*ym).sum)
    return float(np.dot(xm, ym) / (denom + 1e-12))

corr_norm_ratio = pearson(nr21, dcos21)
corr_dnr = pearson(dnr21, dcos21)
corr_mae_imp = pearson(mae_imp21, dcos21)
corr_cos_low = pearson(cos_low21, dcos21)
corr_norm_ref = pearson(nr_ref21, dcos21)
print(f"  corr(delta_cos, norm_ratio_sub_ref) = {corr_norm_ratio:+.4f}")
print(f"  corr(delta_cos, norm_sub - norm_ref) = {corr_dnr:+.4f}")
print(f"  corr(delta_cos, |MAE_delta|) = {corr_mae_imp:+.4f}")
print(f"  corr(delta_cos, cos_low) = {corr_cos_low:+.4f}")
print(f"  corr(delta_cos, norm_ref) = {corr_norm_ref:+.4f}")

# ================================================================
# STEP 6: Classification
# ================================================================
print("\n=== STEP 6: Classification ===\n")
failure_rate = (len(layer21_results) - agg21['n_cosine_positive']) / len(layer21_results)
if failure_rate == 0:
    classification = "PASS_LAYER21_ROBUST_EXCEPT_KNOWN_SEED9"
elif failure_rate <= 0.10:
    classification = "PARTIAL_LAYER21_RARE_ACTIVATION_OUTLIER"
elif failure_rate < 0.50:
    classification = "PARTIAL_LAYER21_ACTIVATION_SENSITIVE"
else:
    classification = "PARTIAL_LAYER21_SYSTEMATIC_CONFLICT"

# Check if neighboring layers also fail
neighbor_fail = (agg20['cosine_failure_rate'] >= 10) or (agg22['cosine_failure_rate'] >= 10)
if neighbor_fail and failure_rate < 0.50:
    classification = "PARTIAL_MULTI_LAYER_SENSITIVITY"

print(f"  layer21 cosine_failure_rate={100*failure_rate:.1f}%")
print(f"  layer20 cosine_failure_rate={agg20['cosine_failure_rate']:.1f}%")
print(f"  layer22 cosine_failure_rate={agg22['cosine_failure_rate']:.1f}%")
print(f"  Classification: {classification}")

# ================================================================
# WRITE JSON
# ================================================================
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
print(f"\nJSON: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"\nDONE. classification={classification}")