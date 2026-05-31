#!/usr/bin/env python3
"""31BB: aggregate mini-sweep only."""
import ctypes, json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
from phase31x_manifest_runtime import encode_sdir, decode_sdir

LIB = '/home/matthew-villnave/llama.cpp/build/bin/libggml-base.so'
Q4_BUDGET_LAYER = 2179072 * 3
QK_K = 256; Q2_BLOCK_BYTES = 84; HIDDEN = 896
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

sys.stderr.write("Loading model...\n")
reader = GGUFReader('/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf')
sys.stderr.write("Model loaded\n")

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

def eval_k(families, base_bytes, X, k):
    Y_ref = mlp(X, families['ffn_gate']['W_ref'], families['ffn_up']['W_ref'], families['ffn_down']['W_ref'])
    Y_low = mlp(X, families['ffn_gate']['W_low'], families['ffn_up']['W_low'], families['ffn_down']['W_low'])
    dec = {fam: decode_sdir(encode_sdir(families[fam]['R'], k)) for fam in families}
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
    return {'k': k, 'margin': int(margin), 'cos_low': round(cos_low, 8), 'cos_sub': round(cos_sub, 8),
            'delta_cos': round(dc, 8), 'cosine_improved': bool(dc > 0), 'severe_regression': bool(dc < -0.05),
            'MAE_low': round(mae_low, 8), 'MAE_sub': round(mae_sub, 8), 'MAE_delta': round(md, 8),
            'MAE_improvement': round(abs(md), 8), 'MAE_improved': bool(md < 0), 'memory_positive': bool(margin > 0)}

cache = {li: load_layer(li) for li in range(24)}
agg_results = {k: [] for k in AGG_K}
SEEDS = list(range(16))
t0 = time.time()
for ki, k in enumerate(AGG_K):
    for li in range(24):
        families, base_bytes = cache[li]
        for s in SEEDS:
            X = np.random.default_rng(s).standard_normal((1, HIDDEN)).astype(np.float32)
            r = eval_k(families, base_bytes, X, k)
            r['layer'] = li; r['seed'] = s
            agg_results[k].append(r)
    sys.stderr.write(f"k={k} done ({time.time()-t0:.0f}s)\n")

sys.stderr.write(f"Total: {time.time()-t0:.0f}s\n")
agg_summary = {}
for k in AGG_K:
    rs = agg_results[k]; n = len(rs)
    cos_impr = sum(1 for r in rs if r['cosine_improved'])
    mae_impr = sum(1 for r in rs if r['MAE_improved'])
    mem_pos = sum(1 for r in rs if r['memory_positive'])
    severe = sum(1 for r in rs if r['severe_regression'])
    dc_arr = np.array([r['delta_cos'] for r in rs], dtype=np.float64)
    md_arr = np.array([r['MAE_delta'] for r in rs], dtype=np.float64)
    agg_summary[k] = {
        'n': n, 'cos_impr': cos_impr, 'cos_fail': n-cos_impr, 'severe': severe,
        'mae_impr': mae_impr, 'mae_fail': n-mae_impr, 'mem_pos': mem_pos,
        'mean_delta_cos': round(float(dc_arr.mean()), 6),
        'median_delta_cos': round(float(np.sort(dc_arr)[n//2]), 6),
        'mean_MAE_improvement': round(float(np.mean(np.abs(md_arr))), 6),
        'worst_delta_cos': round(float(dc_arr.min()), 6),
    }
    sys.stderr.write(f"k={k}: cos_fail={n-cos_impr} severe={severe} mae_fail={n-mae_impr} mean_dc={dc_arr.mean():+.5f}\n")

json_path = '/home/matthew-villnave/sdi-substitutive/results/PHASE31BB_K_PARAMETER_SENSITIVITY.json'
with open(json_path) as f:
    existing = json.load(f)
# Add aggregate to existing
existing['agg_summary'] = agg_summary
existing['aggregate_mini_sweep'] = {k: agg_results[k] for k in AGG_K}
with open(json_path, 'w') as f:
    json.dump(existing, f, indent=2)
sys.stderr.write(f"JSON updated: {json_path}\nDONE\n")