#!/usr/bin/env python3
"""
Post-process 31AZ: Oracle gate tables, proxy gates, classification, write JSON.
Uses the layer21 sweep data from phase31ay_write_json.py (already in memory via rerun).
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
        'norm_ref': round(nr, 6), 'norm_low': round(nl, 6), 'norm_sub': round(ns, 6),
        'norm_ratio_sub_low': round(ns / nl, 8) if nl > 1e-12 else 0.0,
        'abs_norm_sub_minus_low': round(abs(ns - nl), 8),
        'residual_update_norm': round(res_norm, 6),
        'cosine_improved': bool(cos_sub > cos_low),
        'memory_positive': bool(margin > 0),
    }

def run_sweep(families, base_bytes, max_seed, label):
    t0 = time.time()
    results = []
    for s in range(max_seed):
        rng = np.random.default_rng(s)
        X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        r = eval_full(families, base_bytes, X)
        r['seed'] = s
        results.append(r)
    print(f"  {label} sweep done in {time.time()-t0:.1f}s")
    return results

def agg(results):
    n = len(results)
    dc = np.array([r['delta_cos'] for r in results], dtype=np.float64)
    md = np.array([r['MAE_delta'] for r in results], dtype=np.float64)
    cos_pos = sum(1 for r in results if r['cosine_improved'])
    mae_pos = sum(1 for r in results if r['MAE_improved'])
    mem_pos = sum(1 for r in results if r['memory_positive'])
    severe = sum(1 for r in results if r['delta_cos'] < -0.05)
    worst = min(results, key=lambda r: r['delta_cos'])
    best = max(results, key=lambda r: r['delta_cos'])
    sorted_dc = np.sort(dc)
    return {
        'n': n, 'n_cos_pos': cos_pos, 'n_mae_pos': mae_pos, 'n_mem_pos': mem_pos,
        'cos_fail_rate': round(100*(n-cos_pos)/n, 2),
        'severe': severe,
        'worst_seed': int(worst['seed']), 'worst_delta_cos': worst['delta_cos'],
        'best_seed': int(best['seed']), 'best_delta_cos': best['delta_cos'],
        'mean_delta_cos': round(float(dc.mean()), 6),
        'median_delta_cos': round(float(sorted_dc[n//2]), 6),
        'mean_MAE_improvement': round(float(np.mean(np.abs(md))), 6),
    }

def gate_eval(results, gate_fn):
    n = len(results)
    cos_pos = 0; mae_pos = 0; severe = 0; skipped = 0; applied = 0
    dc_list = []; md_list = []
    worst_dc = 0.0; worst_seed = 0
    for r in results:
        apply_res = gate_fn(r)
        if not apply_res:
            skipped += 1
            dc = 0.0; md = 0.0; ci = False; mi = False
        else:
            applied += 1
            dc = r['delta_cos']; md = r['MAE_delta']
            ci = r['cosine_improved']; mi = r['MAE_improved']
            if dc < -0.05: severe += 1
            if dc < worst_dc: worst_dc = dc; worst_seed = r['seed']
        dc_list.append(dc); md_list.append(md)
        if ci: cos_pos += 1
        if mi: mae_pos += 1
    dc_arr = np.array(dc_list, dtype=np.float64)
    md_arr = np.array(md_list, dtype=np.float64)
    sorted_dc = np.sort(dc_arr)
    return {
        'n': n, 'n_applied': applied, 'n_skipped': skipped,
        'n_cos_pos': cos_pos, 'n_mae_pos': mae_pos,
        'cos_fail_rate': round(100*(n-cos_pos)/n, 2),
        'severe': severe,
        'worst_seed': int(worst_seed), 'worst_delta_cos': round(worst_dc, 6),
        'mean_delta_cos': round(float(dc_arr.mean()), 6),
        'median_delta_cos': round(float(sorted_dc[n//2]), 6),
        'mean_MAE_improvement': round(float(np.mean(np.abs(md_arr))), 6),
    }

def pearson(x, y):
    xm = x - x.mean(); ym = y - y.mean()
    return float(np.dot(xm, ym) / (np.sqrt(np.sum(xm**2)) * np.sqrt(np.sum(ym**2)) + 1e-12))

# Load + run sweeps
t0 = time.time()
f21, bb21 = load_layer(21)
f20, bb20 = load_layer(20)
f22, bb22 = load_layer(22)
print("Layers loaded")
r21 = run_sweep(f21, bb21, 64, "L21")
r20 = run_sweep(f20, bb20, 32, "L20")
r22 = run_sweep(f22, bb22, 32, "L22")
print(f"Total time: {time.time()-t0:.1f}s")

# Baseline
b21 = agg(r21); b20 = agg(r20); b22 = agg(r22)
print(f"\nBaseline: L21 cos_pos={b21['n_cos_pos']}/64 severe={b21['severe']}; L20 cos_pos={b20['n_cos_pos']}/32; L22 cos_pos={b22['n_cos_pos']}/32")

# Oracle gates
print("\n=== Oracle cos_low gates ===")
print(f"  {'thr':>6} | {'appl':>6} | {'skip':>6} | {'cos_p':>6} | {'fail%':>6} | {'sev':>4} | {'worst_dc':>10} | {'mean_dc':>10}")
print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}")
oracle_gates = {}
for thr in [0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]:
    thr_val = thr  # capture
    d = gate_eval(r21, lambda r, t=thr_val: r['cos_low'] < t)
    oracle_gates[thr] = d
    print(f"  {thr:>6.2f} | {d['n_applied']:>6} | {d['n_skipped']:>6} | {d['n_cos_pos']:>6} | {d['cos_fail_rate']:>6.2f} | {d['severe']:>4} | {d['worst_delta_cos']:>+10.6f} | {d['mean_delta_cos']:>+10.6f}")

# Runtime proxy gates: norm_ratio_sub_low
print("\n=== Runtime proxy: norm_ratio_sub_low ===")
best_nr = None
for thr in [0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15]:
    thr_val = thr
    d = gate_eval(r21, lambda r, t=thr_val: r['norm_ratio_sub_low'] < t)
    if best_nr is None or (d['n_cos_pos'] > best_nr[1]['n_cos_pos']):
        best_nr = (thr, d)
print(f"  Best norm_ratio: thr={best_nr[0]}, cos_pos={best_nr[1]['n_cos_pos']}/64, severe={best_nr[1]['severe']}, fail={best_nr[1]['cos_fail_rate']}%")

# residual_update_norm
print("\n=== Runtime proxy: residual_update_norm ===")
best_res = None
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
    thr_val = thr
    d = gate_eval(r21, lambda r, t=thr_val: r['residual_update_norm'] < t)
    if best_res is None or (d['n_cos_pos'] > best_res[1]['n_cos_pos']):
        best_res = (thr, d)
print(f"  Best res_norm: thr={best_res[0]}, cos_pos={best_res[1]['n_cos_pos']}/64, severe={best_res[1]['severe']}, fail={best_res[1]['cos_fail_rate']}%")

# abs_norm_sub_minus_low
print("\n=== Runtime proxy: abs_norm_sub_minus_low ===")
best_anorm = None
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    thr_val = thr
    d = gate_eval(r21, lambda r, t=thr_val: r['abs_norm_sub_minus_low'] < t)
    if best_anorm is None or (d['n_cos_pos'] > best_anorm[1]['n_cos_pos']):
        best_anorm = (thr, d)
print(f"  Best abs_norm_diff: thr={best_anorm[0]}, cos_pos={best_anorm[1]['n_cos_pos']}/64, severe={best_anorm[1]['severe']}, fail={best_anorm[1]['cos_fail_rate']}%")

# Trivial skip policies
print("\n=== Trivial skip policies ===")
d_always_skip = gate_eval(r21, lambda r: False)
d_skip_severe = gate_eval(r21, lambda r: r['delta_cos'] >= -0.05)
print(f"  Always skip L21: cos_pos={d_always_skip['n_cos_pos']}/64, severe=0, fail={d_always_skip['cos_fail_rate']}%, mean_dc={d_always_skip['mean_delta_cos']:+.6f}")
print(f"  Skip severe oracle: cos_pos={d_skip_severe['n_cos_pos']}/64, severe=0, fail={d_skip_severe['cos_fail_rate']}%, mean_dc={d_skip_severe['mean_delta_cos']:+.6f}")

# Transfer best oracle to L20/L22
print("\n=== Transfer oracle cos_low<0.85 to L20/L22 ===")
thr_oracle = 0.85
thr_val = thr_oracle
t20_o = gate_eval(r20, lambda r, t=thr_val: r['cos_low'] < t)
t22_o = gate_eval(r22, lambda r, t=thr_val: r['cos_low'] < t)
print(f"  L20 oracle<0.85: cos_pos={t20_o['n_cos_pos']}/32 vs baseline={b20['n_cos_pos']}/32, mean_dc={t20_o['mean_delta_cos']:+.6f} vs baseline={b20['mean_delta_cos']:+.6f}")
print(f"  L22 oracle<0.85: cos_pos={t22_o['n_cos_pos']}/32 vs baseline={b22['n_cos_pos']}/32, mean_dc={t22_o['mean_delta_cos']:+.6f} vs baseline={b22['mean_delta_cos']:+.6f}")

# Transfer best runtime proxy to L20/L22
print(f"\n=== Transfer residual_update_norm<{best_res[0]} to L20/L22 ===")
thr_val = best_res[0]
t20_r = gate_eval(r20, lambda r, t=thr_val: r['residual_update_norm'] < t)
t22_r = gate_eval(r22, lambda r, t=thr_val: r['residual_update_norm'] < t)
print(f"  L20 res_norm<{best_res[0]}: cos_pos={t20_r['n_cos_pos']}/32 vs baseline={b20['n_cos_pos']}/32, mean_dc={t20_r['mean_delta_cos']:+.6f} vs baseline={b20['mean_delta_cos']:+.6f}")
print(f"  L22 res_norm<{best_res[0]}: cos_pos={t22_r['n_cos_pos']}/32 vs baseline={b22['n_cos_pos']}/32, mean_dc={t22_r['mean_delta_cos']:+.6f} vs baseline={b22['mean_delta_cos']:+.6f}")

# Correlations
print("\n=== Correlations ===")
cl21 = np.array([r['cos_low'] for r in r21], dtype=np.float64)
dc21 = np.array([r['delta_cos'] for r in r21], dtype=np.float64)
nr21 = np.array([r['norm_ratio_sub_low'] for r in r21], dtype=np.float64)
anorm21 = np.array([r['abs_norm_sub_minus_low'] for r in r21], dtype=np.float64)
resn21 = np.array([r['residual_update_norm'] for r in r21], dtype=np.float64)
print(f"  corr(cos_low, delta_cos) = {pearson(cl21, dc21):+.4f}")
print(f"  corr(cos_low, norm_ratio_sub_low) = {pearson(cl21, nr21):+.4f}")
print(f"  corr(cos_low, abs_norm_diff) = {pearson(cl21, anorm21):+.4f}")
print(f"  corr(cos_low, residual_update_norm) = {pearson(cl21, resn21):+.4f}")
print(f"  corr(delta_cos, norm_ratio_sub_low) = {pearson(dc21, nr21):+.4f}")
print(f"  corr(delta_cos, abs_norm_diff) = {pearson(dc21, anorm21):+.4f}")
print(f"  corr(delta_cos, residual_update_norm) = {pearson(dc21, resn21):+.4f}")

# Classification
best_oracle = max(oracle_gates.items(), key=lambda x: x[1]['n_cos_pos'])
best_proxy = max([best_nr, best_res, best_anorm], key=lambda x: x[1]['n_cos_pos'])
print(f"\nBest oracle: thr={best_oracle[0]}, cos_pos={best_oracle[1]['n_cos_pos']}/64, severe={best_oracle[1]['severe']}")
print(f"Best proxy: {best_proxy[0]}, cos_pos={best_proxy[1]['n_cos_pos']}/64, severe={best_proxy[1]['severe']}")

if best_proxy[1]['cos_fail_rate'] < b21['cos_fail_rate'] and best_proxy[1]['severe'] < b21['severe']:
    classification = "PASS_RUNTIME_PROXY_GATE_FOUND"
elif best_oracle[1]['cos_fail_rate'] == 0.0 and best_oracle[1]['severe'] == 0:
    classification = "PASS_ORACLE_GATE_RECOVERS_LAYER21"
elif best_proxy[1]['n_cos_pos'] >= b21['n_cos_pos']:
    classification = "PARTIAL_ORACLE_ONLY_GATE"
else:
    classification = "PARTIAL_SKIP_POLICY_TRADEOFF"

print(f"Classification: {classification}")

# Write JSON
out = {
    'phase': '31AZ',
    'classification': classification,
    'baseline': {
        'layer21': b21, 'layer20': b20, 'layer22': b22,
    },
    'oracle_gates': {str(k): v for k, v in oracle_gates.items()},
    'runtime_proxy_gates': {
        'norm_ratio_sub_low': {'best_threshold': best_nr[0], 'best_result': best_nr[1]},
        'abs_norm_sub_minus_low': {'best_threshold': best_anorm[0], 'best_result': best_anorm[1]},
        'residual_update_norm': {'best_threshold': best_res[0], 'best_result': best_res[1]},
    },
    'trivial_skip': {
        'always_skip': d_always_skip,
        'skip_severe_oracle': d_skip_severe,
    },
    'transfer': {
        f'oracle_cos_low<{thr_oracle}': {'layer20': t20_o, 'layer22': t22_o},
        f'residual_update_norm<{best_res[0]}': {'layer20': t20_r, 'layer22': t22_r},
    },
    'layer21_sweep': r21,
    'correlations': {
        'corr_cos_low_delta_cos': round(pearson(cl21, dc21), 4),
        'corr_cos_low_norm_ratio': round(pearson(cl21, nr21), 4),
        'corr_cos_low_abs_norm_diff': round(pearson(cl21, anorm21), 4),
        'corr_cos_low_res_norm': round(pearson(cl21, resn21), 4),
        'corr_delta_cos_norm_ratio': round(pearson(dc21, nr21), 4),
        'corr_delta_cos_abs_norm_diff': round(pearson(dc21, anorm21), 4),
        'corr_delta_cos_res_norm': round(pearson(dc21, resn21), 4),
    },
}

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31AZ_RESIDUAL_GATING_POLICY.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nJSON: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"DONE. classification={classification}")