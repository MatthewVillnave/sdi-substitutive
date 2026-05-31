#!/usr/bin/env python3
"""
Phase 31AZ — Residual Gating Policy / Skip Optimization
Repo: sdi-substitutive
Layer 21 seeds 0-63 + layers 20/22 seeds 0-31.
Tests oracle cos_low gates and runtime-plausible proxy gates.
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

def eval_layer_full(families, base_bytes, X, k, alpha):
    """Returns full per-seed data including Y_ref, Y_low, Y_sub for gate computation."""
    Y_ref = mlp(X, families['ffn_gate']['W_ref'], families['ffn_up']['W_ref'], families['ffn_down']['W_ref'])
    Y_low = mlp(X, families['ffn_gate']['W_low'], families['ffn_up']['W_low'], families['ffn_down']['W_low'])
    dec = {fam: decode_sdir(encode_sdir(alpha * families[fam]['R'], k)) for fam in families}
    Y_sub = mlp(X,
                families['ffn_gate']['W_low'] + dec['ffn_gate'],
                families['ffn_up']['W_low']   + dec['ffn_up'],
                families['ffn_down']['W_low']  + dec['ffn_down'])
    res_bytes = sum(len(encode_sdir(families[f]['R'], k)) for f in families)
    margin = Q4_BUDGET_LAYER - (base_bytes + res_bytes)

    # Core metrics
    cos_low = cos_sim(Y_ref, Y_low)
    cos_sub = cos_sim(Y_ref, Y_sub)
    mae_low = float(np.abs(Y_ref - Y_low).mean())
    mae_sub = float(np.abs(Y_ref - Y_sub).mean())
    nr = norm_vec(Y_ref)
    nl = norm_vec(Y_low)
    ns = norm_vec(Y_sub)

    # Runtime proxy features (no Y_ref)
    norm_ratio_sub_low = ns / nl if nl > 1e-12 else 0.0
    abs_norm_sub_minus_low = abs(ns - nl)
    res_update_norm = norm_vec(Y_sub - Y_low)
    rel_update_norm = res_update_norm / nl if nl > 1e-12 else 0.0
    norm_sub_minus_low = ns - nl

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
        'norm_ratio_sub_low': round(norm_ratio_sub_low, 8),
        'abs_norm_sub_minus_low': round(abs_norm_sub_minus_low, 8),
        'residual_update_norm': round(res_update_norm, 6),
        'relative_update_norm': round(rel_update_norm, 8),
        'norm_sub_minus_low': round(norm_sub_minus_low, 6),
        'cosine_improved': bool(cos_sub > cos_low),
        'memory_positive': bool(margin > 0),
    }

def run_sweep(families, base_bytes, max_seed, label):
    t0 = time.time()
    results = []
    for s in range(max_seed):
        rng = np.random.default_rng(s)
        X = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        r = eval_layer_full(families, base_bytes, X, k=1.0, alpha=1.0)
        r['seed'] = s
        results.append(r)
    elapsed = time.time() - t0
    print(f"  {label} sweep 0-{max_seed-1} done in {elapsed:.1f}s ({max_seed} seeds, {elapsed/max_seed*1000:.0f}ms/seed)")
    return results

def agg(results, label=""):
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
    median_dc = sorted_dc[n//2]
    mean_dc = float(dc.mean())
    mean_mae_imp = float(np.mean(np.abs(md)))
    d = {
        'n': n,
        'n_cos_pos': cos_pos,
        'n_mae_pos': mae_pos,
        'n_mem_pos': mem_pos,
        'cos_fail_rate': round(100*(n-cos_pos)/n, 2),
        'severe': severe,
        'worst_seed': int(worst['seed']),
        'worst_delta_cos': worst['delta_cos'],
        'best_seed': int(best['seed']),
        'best_delta_cos': best['delta_cos'],
        'mean_delta_cos': round(mean_dc, 6),
        'median_delta_cos': round(float(median_dc), 6),
        'mean_MAE_improvement': round(mean_mae_imp, 6),
    }
    if label:
        print(f"  {label}: cos_pos={cos_pos}/{n} ({100*cos_pos/n:.1f}%), mae_pos={mae_pos}/{n} ({100*mae_pos/n:.1f}%), severe={severe}, worst=seed{worst['seed']}({worst['delta_cos']:+.5f}), mean={mean_dc:+.5f}")
    return d

def gate_eval(results, gate_fn, label=""):
    """Evaluate a gating function over sweep results.
    gate_fn(seed_data) -> True means APPLY residual, False means SKIP."""
    n = len(results)
    dc_list = []; md_list = []; cos_pos = 0; mae_pos = 0; severe = 0; skipped = 0; applied = 0
    for r in results:
        apply = gate_fn(r)
        if not apply:
            skipped += 1
            dc = 0.0
            md = 0.0
            ci = False
            mi = False
        else:
            applied += 1
            dc = r['delta_cos']
            md = r['MAE_delta']
            ci = r['cosine_improved']
            mi = r['MAE_improved']
            if dc < -0.05: severe += 1
        dc_list.append(dc); md_list.append(md)
        if ci: cos_pos += 1
        if mi: mae_pos += 1
    dc_arr = np.array(dc_list, dtype=np.float64)
    md_arr = np.array(md_list, dtype=np.float64)
    worst_i = int(np.argmin(dc_arr))
    sorted_dc = np.sort(dc_arr)
    median_dc = sorted_dc[n//2]
    mean_dc = float(dc_arr.mean())
    mean_mae_imp = float(np.mean(np.abs(md_arr)))
    d = {
        'n': n,
        'n_applied': applied,
        'n_skipped': skipped,
        'n_cos_pos': cos_pos,
        'n_mae_pos': mae_pos,
        'cos_fail_rate': round(100*(n-cos_pos)/n, 2),
        'severe': severe,
        'worst_seed': int(results[worst_i]['seed']),
        'worst_delta_cos': round(float(dc_arr[worst_i]), 6),
        'mean_delta_cos': round(mean_dc, 6),
        'median_delta_cos': round(float(median_dc), 6),
        'mean_MAE_improvement': round(mean_mae_imp, 6),
    }
    if label:
        print(f"  {label}: applied={applied}, skipped={skipped}, cos_pos={cos_pos}/{n}, severe={severe}, worst=seed{d['worst_seed']}({d['worst_delta_cos']:+.5f}), mean={mean_dc:+.5f}")
    return d

# Load all layers
t0 = time.time()
f21, bb21 = load_layer(21)
f20, bb20 = load_layer(20)
f22, bb22 = load_layer(22)
print(f"Layers loaded in {time.time()-t0:.1f}s")

# Run sweeps
print("\n=== Running sweeps ===")
t0 = time.time()
r21 = run_sweep(f21, bb21, 64, "L21")
r20 = run_sweep(f20, bb20, 32, "L20")
r22 = run_sweep(f22, bb22, 32, "L22")
print(f"Total sweep time: {time.time()-t0:.1f}s")

# Baseline always-on
print("\n=== Baseline always-on ===")
b21 = agg(r21, "L21 baseline")
b20 = agg(r20, "L20 baseline")
b22 = agg(r22, "L22 baseline")

# ================================================================
# STEP 1: Oracle cos_low threshold gates
# ================================================================
print("\n=== STEP 1: Oracle cos_low gates ===")
print(f"  {'threshold':>10} | {'applied':>8} | {'skipped':>8} | {'cos_pos':>8} | {'cos_fail':>8} | {'severe':>7} | {'worst_dc':>10} | {'mean_dc':>10} | {'mean_mae':>10}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
oracle_gates = {}
for thr in [0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]:
    d = gate_eval(r21, lambda r, t=thr: r['cos_low'] < t, label=f"cos_low<{t}")
    oracle_gates[thr] = d

# ================================================================
# STEP 2: Runtime-plausible proxy gates
# ================================================================
print("\n=== STEP 2: Runtime proxy gates ===")

# norm_ratio_sub_low threshold sweep
print("\n  norm_ratio_sub_low threshold gates:")
best_nr_gate = None; best_nr_score = -1
for thr in [0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15]:
    d = gate_eval(r21, lambda r, t=thr: r['norm_ratio_sub_low'] < t,
                  label=f"  norm_ratio<{t}")
    score = d['n_cos_pos'] - 0.5 * d['severe']
    if score > best_nr_score:
        best_nr_score = score; best_nr_gate = (thr, d)
    if thr == 1.10:
        nr_gate_best = d
print(f"  Best norm_ratio gate: threshold={best_nr_gate[0]}, cos_pos={best_nr_gate[1]['n_cos_pos']}, severe={best_nr_gate[1]['severe']}")

# abs_norm_sub_minus_low threshold sweep
print("\n  abs_norm_sub_minus_low threshold gates:")
best_anorm_gate = None; best_anorm_score = -1
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    d = gate_eval(r21, lambda r, t=thr: r['abs_norm_sub_minus_low'] < t,
                  label=f"  abs_norm_diff<{t}")
    score = d['n_cos_pos'] - 0.5 * d['severe']
    if score > best_anorm_score:
        best_anorm_score = score; best_anorm_gate = (thr, d)
print(f"  Best abs_norm gate: threshold={best_anorm_gate[0]}, cos_pos={best_anorm_gate[1]['n_cos_pos']}, severe={best_anorm_gate[1]['severe']}")

# residual_update_norm threshold sweep
print("\n  residual_update_norm threshold gates:")
best_res_gate = None; best_res_score = -1
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
    d = gate_eval(r21, lambda r, t=thr: r['residual_update_norm'] < t,
                  label=f"  res_norm<{t}")
    score = d['n_cos_pos'] - 0.5 * d['severe']
    if score > best_res_score:
        best_res_score = score; best_res_gate = (thr, d)
print(f"  Best residual_update_norm gate: threshold={best_res_gate[0]}, cos_pos={best_res_gate[1]['n_cos_pos']}, severe={best_res_gate[1]['severe']}")

# ================================================================
# STEP 3: Trivial skip policies
# ================================================================
print("\n=== STEP 3: Trivial skip policies ===")

# Policy A: always skip on layer 21
d_always_skip = gate_eval(r21, lambda r: False, label="always_skip_L21")

# Policy B: skip only when delta_cos < -0.05 (oracle-labeled severe)
d_severe_skip = gate_eval(r21, lambda r: r['delta_cos'] >= -0.05, label="skip_severe_oracle")

# Policy C: skip only when residual_update_norm < threshold
best_res_skip = gate_eval(r21, lambda r, t=best_res_gate[0]: r['residual_update_norm'] < t,
                           label=f"skip_res_norm<{best_res_gate[0]}")

# Policy D: skip only when norm_ratio_sub_low < threshold
best_nr_skip = gate_eval(r21, lambda r, t=best_nr_gate[0]: r['norm_ratio_sub_low'] < t,
                          label=f"skip_norm_ratio<{best_nr_gate[0]}")

# ================================================================
# STEP 4: Transfer best oracle gate to layers 20 and 22
# ================================================================
print("\n=== STEP 4: Transfer oracle gate cos_low<0.85 to layers 20 and 22 ===")
thr_best = 0.85
t20_best = gate_eval(r20, lambda r, t=thr_best: r['cos_low'] < t,
                      label=f"L20 cos_low<{t} vs baseline")
t22_best = gate_eval(r22, lambda r, t=thr_best: r['cos_low'] < t,
                      label=f"L22 cos_low<{t} vs baseline")

# Also test best runtime proxy on layers 20 and 22
print("\n=== Transfer best runtime proxy (residual_update_norm) to L20/L22 ===")
thr_res = best_res_gate[0]
t20_res = gate_eval(r20, lambda r, t=thr_res: r['residual_update_norm'] < t,
                     label=f"L20 res_norm<{t} vs baseline")
t22_res = gate_eval(r22, lambda r, t=thr_res: r['residual_update_norm'] < t,
                     label=f"L22 res_norm<{t} vs baseline")

# ================================================================
# STEP 5: Correlation of proxy features with cos_low
# ================================================================
print("\n=== STEP 5: Correlation of proxy features with cos_low ===")
cl = np.array([r['cos_low'] for r in r21], dtype=np.float64)
nr = np.array([r['norm_ratio_sub_low'] for r in r21], dtype=np.float64)
anorm = np.array([r['abs_norm_sub_minus_low'] for r in r21], dtype=np.float64)
resn = np.array([r['residual_update_norm'] for r in r21], dtype=np.float64)
dc = np.array([r['delta_cos'] for r in r21], dtype=np.float64)

def pearson(x, y):
    xm = x - x.mean(); ym = y - y.mean()
    return float(np.dot(xm, ym) / (np.sqrt(np.sum(xm**2)) * np.sqrt(np.sum(ym**2)) + 1e-12))

print(f"  corr(cos_low, norm_ratio_sub_low) = {pearson(cl, nr):+.4f}")
print(f"  corr(cos_low, abs_norm_sub_minus_low) = {pearson(cl, anorm):+.4f}")
print(f"  corr(cos_low, residual_update_norm) = {pearson(cl, resn):+.4f}")
print(f"  corr(cos_low, delta_cos) = {pearson(cl, dc):+.4f}")

# corr of proxies with delta_cos
print(f"  corr(delta_cos, norm_ratio_sub_low) = {pearson(dc, nr):+.4f}")
print(f"  corr(delta_cos, abs_norm_sub_minus_low) = {pearson(dc, anorm):+.4f}")
print(f"  corr(delta_cos, residual_update_norm) = {pearson(dc, resn):+.4f}")

# ================================================================
# STEP 6: Classification
# ================================================================
print("\n=== STEP 6: Classification ===")

# Find best oracle gate
best_oracle = None
for thr, d in oracle_gates.items():
    if best_oracle is None or d['n_cos_pos'] > best_oracle[1]['n_cos_pos']:
        best_oracle = (thr, d)

oracle_recovers = b21['n_cos_pos'] + b21['severe'] - best_oracle[1]['severe']
runtime_proxy_best = max([best_nr_gate, best_res_gate, best_anorm_gate],
                          key=lambda x: x[1]['n_cos_pos'] - 0.5 * x[1]['severe'])

if best_oracle[1]['cos_fail_rate'] == 0.0:
    classification = "PASS_ORACLE_GATE_RECOVERS_LAYER21"
elif best_oracle[1]['cos_fail_rate'] < b21['cos_fail_rate']:
    classification = "PARTIAL_ORACLE_ONLY_GATE"
else:
    classification = "PARTIAL_SKIP_POLICY_TRADEOFF"

if runtime_proxy_best[1]['cos_fail_rate'] < b21['cos_fail_rate']:
    alt_class = "PASS_RUNTIME_PROXY_GATE_FOUND"
else:
    alt_class = classification

# Final classification: prefer the better one
if runtime_proxy_best[1]['n_cos_pos'] > best_oracle[1]['n_cos_pos']:
    classification = alt_class
else:
    classification = classification

print(f"  Best oracle gate: cos_low<{best_oracle[0]}, cos_pos={best_oracle[1]['n_cos_pos']}/64, severe={best_oracle[1]['severe']}, fail_rate={best_oracle[1]['cos_fail_rate']}%")
print(f"  Best runtime proxy gate: {runtime_proxy_best[0]}, cos_pos={runtime_proxy_best[1]['n_cos_pos']}/64, severe={runtime_proxy_best[1]['severe']}, fail_rate={runtime_proxy_best[1]['cos_fail_rate']}%")
print(f"  Baseline: cos_pos={b21['n_cos_pos']}/64, severe={b21['severe']}, fail_rate={b21['cos_fail_rate']}%")
print(f"  Classification: {classification}")

# ================================================================
# WRITE JSON
# ================================================================
out = {
    'phase': '31AZ',
    'classification': classification,
    'baseline': {
        'layer21': b21,
        'layer20': b20,
        'layer22': b22,
    },
    'oracle_gates': {str(k): v for k, v in oracle_gates.items()},
    'runtime_proxy_gates': {
        'norm_ratio_sub_low': {
            'best_threshold': best_nr_gate[0],
            'best_result': best_nr_gate[1],
        },
        'abs_norm_sub_minus_low': {
            'best_threshold': best_anorm_gate[0],
            'best_result': best_anorm_gate[1],
        },
        'residual_update_norm': {
            'best_threshold': best_res_gate[0],
            'best_result': best_res_gate[1],
        },
    },
    'trivial_skip_policies': {
        'always_skip_layer21': d_always_skip,
        'skip_severe_oracle': d_severe_skip,
        f'skip_residual_update_norm<{best_res_gate[0]}': best_res_gate[1],
        f'skip_norm_ratio<{best_nr_gate[0]}': best_nr_gate[1],
    },
    'transfer': {
        f'oracle_cos_low<{thr_best}': {
            'layer20': t20_best,
            'layer22': t22_best,
        },
        f'residual_update_norm<{thr_res}': {
            'layer20': t20_res,
            'layer22': t22_res,
        },
    },
    'layer21_sweep': r21,
    'correlations': {
        'corr_cos_low_norm_ratio': round(pearson(cl, nr), 4),
        'corr_cos_low_abs_norm_diff': round(pearson(cl, anorm), 4),
        'corr_cos_low_res_norm': round(pearson(cl, resn), 4),
        'corr_delta_cos_norm_ratio': round(pearson(dc, nr), 4),
        'corr_delta_cos_abs_norm_diff': round(pearson(dc, anorm), 4),
        'corr_delta_cos_res_norm': round(pearson(dc, resn), 4),
    },
}

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31AZ_RESIDUAL_GATING_POLICY.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nJSON: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")
print(f"\nDONE. classification={classification}")