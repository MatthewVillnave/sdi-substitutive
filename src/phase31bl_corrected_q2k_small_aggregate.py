#!/usr/bin/env python3
"""
Phase 31BL — Corrected Q2_K Small Aggregate Validation

Validates: corrected_ceil_per_row Q2_K, up+gate residual k=0.5%, no down residual.

Layers: 2, 21
Seeds: 0–15 (16 each = 32 pairs)

Policy:
  - Q2_K mode: corrected_ceil_per_row
  - Residual families: ffn_up + ffn_gate only
  - k: 0.5%
  - alpha: 1.0
  - ffn_down: no residual (Q4 budget slack)

Environment:
  SDI_GGUF_MODEL_PATH   — path to Qwen2.5-0.5B GGUF
  SDI_LLAMA_CPP_ROOT    — path to llama.cpp root
  SDI_LLAMA_CPP_LIB     — path to libggml-base.so

Outputs:
  src/results/PHASE31BL_CORRECTED_Q2K_SMALL_AGGREGATE.json
"""

import os, sys, json, time
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CPP_ROOT = os.environ.get("SDI_LLAMA_CPP_ROOT", "")
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
sys.path.insert(0, os.path.join(LLAMA_CPP_ROOT, "gguf-py"))

from gguf import GGUFReader
from gguf.quants import dequantize
from q2k_backend import quantize_q2k_f32_to_bytes, dequantize_q2k_bytes_to_f32, q2k_expected_nbytes
from phase31x_manifest_runtime import cosine, encode_sdir, decode_sdir

GGUF_PATH = os.environ.get("SDI_GGUF_MODEL_PATH", "")
if not GGUF_PATH or not os.path.exists(GGUF_PATH):
    raise FileNotFoundError(f"SDI_GGUF_MODEL_PATH not set or file not found: {GGUF_PATH!r}")

Q4_BUDGET_FAMILY = 2_179_072
Q4_BUDGET_LAYER  = 3 * Q4_BUDGET_FAMILY
K_PCT = 0.5
ALPHA = 1.0

LAYERS = [2, 21]
SEEDS = list(range(16))

# ─── Helpers ───────────────────────────────────────────────────────────────────

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -709, 709)))

def mlp_full(X, Wg, Wu, Wd):
    H = silu(X @ Wg.T) * (X @ Wu.T)
    return H @ Wd.T

def cosine_sim(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0

def mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())

# ─── Weight cache ──────────────────────────────────────────────────────────────

print("Loading GGUF weights…", flush=True)
reader = GGUFReader(GGUF_PATH)

def load_raw(layer: int, fam: str) -> np.ndarray:
    name = f"blk.{layer}.{fam}.weight"
    t = next(t for t in reader.tensors if t.name == name)
    W = dequantize(t.data, t.tensor_type).astype(np.float32)
    print(f"  loaded {name}: {W.shape}", flush=True)
    return W

# Pre-load all W_ref
wref = {}  # (layer, family) -> np.ndarray
for layer in LAYERS:
    for fam in ["ffn_gate", "ffn_up", "ffn_down"]:
        wref[(layer, fam)] = load_raw(layer, fam)

# ─── Pre-compute Q2_K low-weights and residuals ────────────────────────────────

print("Computing Q2_K corrected dequantized weights…", flush=True)
wlow = {}  # (layer, family) -> np.ndarray
resid = {}  # (layer, family) -> np.ndarray

for key, W in wref.items():
    q2k_bytes = quantize_q2k_f32_to_bytes(W, mode="corrected_ceil_per_row")
    Wlo = dequantize_q2k_bytes_to_f32(q2k_bytes, *W.shape, mode="corrected_ceil_per_row")
    wlow[key] = Wlo
    resid[key] = W - Wlo  # residual for SDIR
    print(f"  {key}: residual RMS={float(np.abs(resid[key]).mean()):.6f}", flush=True)

# ─── Pre-compute SDIR encodings (k=0.5% for gate+up only) ─────────────────────

print("Encoding SDIR residuals (k=0.5%, gate+up only)…", flush=True)
sdir = {}  # (layer, family) -> bytes | None
for layer in LAYERS:
    for fam in ["ffn_gate", "ffn_up"]:
        encoded = encode_sdir(resid[(layer, fam)], k_pct=K_PCT)
        sdir[(layer, fam)] = encoded
        print(f"  sdir[{layer},{fam}]: {len(encoded):,} bytes", flush=True)
    sdir[(layer, "ffn_down")] = None  # no residual for down

# ─── Memory accounting ──────────────────────────────────────────────────────────

def memory_bytes(layer: int) -> dict:
    """Return per-family and total bytes for corrected up+gate k=0.5%."""
    up_q2k  = len(quantize_q2k_f32_to_bytes(wref[(layer,"ffn_up")],   mode="corrected_ceil_per_row"))
    gt_q2k  = len(quantize_q2k_f32_to_bytes(wref[(layer,"ffn_gate")],mode="corrected_ceil_per_row"))
    dn_q2k  = len(quantize_q2k_f32_to_bytes(wref[(layer,"ffn_down")],mode="corrected_ceil_per_row"))
    up_sdir = len(sdir[(layer,"ffn_up")])   if sdir[(layer,"ffn_up")]   is not None else 0
    gt_sdir = len(sdir[(layer,"ffn_gate")]) if sdir[(layer,"ffn_gate")] is not None else 0
    dn_sdir = 0
    total = up_q2k + gt_q2k + dn_q2k + up_sdir + gt_sdir + dn_sdir
    return {
        "q2k_up": up_q2k, "q2k_gate": gt_q2k, "q2k_down": dn_q2k,
        "sdir_up": up_sdir, "sdir_gate": gt_sdir, "sdir_down": dn_sdir,
        "total": total, "margin": Q4_BUDGET_LAYER - total,
        "memory_positive": Q4_BUDGET_LAYER >= total,
    }

# ─── Per-pair computation ──────────────────────────────────────────────────────

print("\nRunning 32-pair aggregate…", flush=True)
t_start = time.time()

results = []

for layer in LAYERS:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((1, 896)).astype(np.float32)

        # Reference output
        Y_ref = mlp_full(X,
                         wref[(layer,"ffn_gate")],
                         wref[(layer,"ffn_up")],
                         wref[(layer,"ffn_down")])

        # Low output (Q2_K corrected, no residual)
        Y_low = mlp_full(X,
                         wlow[(layer,"ffn_gate")],
                         wlow[(layer,"ffn_up")],
                         wlow[(layer,"ffn_down")])

        # Substitute output: apply gate+up SDIR residual
        # W_low_sub = W_low + residual_decoded
        Wg_sub = wlow[(layer,"ffn_gate")] + decode_sdir(sdir[(layer,"ffn_gate")]) \
                 if sdir[(layer,"ffn_gate")] is not None else wlow[(layer,"ffn_gate")]
        Wu_sub = wlow[(layer,"ffn_up")]   + decode_sdir(sdir[(layer,"ffn_up")])   \
                 if sdir[(layer,"ffn_up")]   is not None else wlow[(layer,"ffn_up")]
        Wd_sub = wlow[(layer,"ffn_down")]  # no down residual

        Y_sub = mlp_full(X, Wg_sub, Wu_sub, Wd_sub)

        # Metrics
        cl = cosine_sim(Y_ref.ravel(), Y_low.ravel())
        cs = cosine_sim(Y_ref.ravel(), Y_sub.ravel())
        ml = mae(Y_ref, Y_low)
        ms = mae(Y_ref, Y_sub)

        mem = memory_bytes(layer)

        pair = {
            "layer": layer,
            "seed": seed,
            "cos_low": round(cl, 6),
            "cos_sub": round(cs, 6),
            "delta_cos": round(cs - cl, 6),
            "MAE_low": round(ml, 6),
            "MAE_sub": round(ms, 6),
            "MAE_delta": round(ms - ml, 6),
            "MAE_improvement": round(ml - ms, 6),
            "memory_bytes": mem["total"],
            "margin": mem["margin"],
            "memory_positive": mem["memory_positive"],
            "severe_regression": (cs - cl) < -0.05,
            "cosine_improved": (cs - cl) > 0,
            "cosine_nonnegative": (cs - cl) >= 0,
            "MAE_improved": (ms - ml) < 0,
        }
        results.append(pair)
        tag = "✓" if pair["memory_positive"] and not pair["severe_regression"] else "✗"
        print(f"  L{layer} S{seed:2d}: dc={cs-cl:+.4f} md={ms-ml:+.5f} "
              f"margin={mem['margin']:,} pos={pair['memory_positive']} {tag}", flush=True)

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s ({elapsed/32:.2f}s/pair)", flush=True)

# ─── Aggregate stats ───────────────────────────────────────────────────────────

total_pairs = len(results)
n_cos_imp  = sum(1 for r in results if r["cosine_improved"])
n_cos_nneg = sum(1 for r in results if r["cosine_nonnegative"])
n_mae_imp  = sum(1 for r in results if r["MAE_improved"])
n_mem_pos  = sum(1 for r in results if r["memory_positive"])
n_severe   = sum(1 for r in results if r["severe_regression"])
n_cos_fail = sum(1 for r in results if not r["cosine_improved"])
n_mae_fail = sum(1 for r in results if not r["MAE_improved"])

dcs = [r["delta_cos"] for r in results]
mds = [r["MAE_delta"] for r in results]

worst_pair   = min(results, key=lambda r: r["delta_cos"])
best_pair    = max(results, key=lambda r: r["delta_cos"])

mean_dc  = sum(dcs) / len(dcs)
median_dc = float(np.median(dcs))
mean_mi   = sum(r["MAE_improvement"] for r in results) / len(results)
min_margin = min(r["margin"] for r in results)

# Layer-specific
def layer_stats(layer):
    rl = [r for r in results if r["layer"] == layer]
    dcl = [r["delta_cos"] for r in rl]
    return {
        "layer": layer,
        "n_pairs": len(rl),
        "n_cosine_improved": sum(1 for r in rl if r["cosine_improved"]),
        "n_cosine_nonnegative": sum(1 for r in rl if r["cosine_nonnegative"]),
        "n_MAE_improved": sum(1 for r in rl if r["MAE_improved"]),
        "n_memory_positive": sum(1 for r in rl if r["memory_positive"]),
        "n_severe_regressions": sum(1 for r in rl if r["severe_regression"]),
        "worst_seed": min(rl, key=lambda r: r["delta_cos"])["seed"],
        "worst_delta_cos": min(dcl),
        "best_seed": max(rl, key=lambda r: r["delta_cos"])["seed"],
        "best_delta_cos": max(dcl),
        "mean_delta_cos": sum(dcl) / len(dcl),
        "mean_MAE_improvement": sum(r["MAE_improvement"] for r in rl) / len(rl),
        "min_margin": min(r["margin"] for r in rl),
    }

# Anchor consistency check
anchors = {
    (21, 9): {"expected_delta_cos": 0.1575, "expected_mae_improvement": True},
    (21, 0): {"expected_delta_cos": 0.3152, "expected_mae_improvement": True},
    ( 2, 7): {"expected_delta_cos": 0.0203, "expected_mae_improvement": True},
}
anchor_check = {}
for key, exp in anchors.items():
    layer, seed = key
    r = next(r for r in results if r["layer"] == layer and r["seed"] == seed)
    dc = r["delta_cos"]
    mi = r["MAE_improvement"]
    ok = abs(dc - exp["expected_delta_cos"]) < 0.05 and mi == exp["expected_mae_improvement"]
    anchor_check[f"L{layer}_S{seed}"] = {
        "delta_cos": dc,
        "expected_delta_cos": exp["expected_delta_cos"],
        "delta_diff": round(dc - exp["expected_delta_cos"], 6),
        "MAE_improvement": mi,
        "expected_MAE_improvement": exp["expected_mae_improvement"],
        "within_tolerance": ok,
    }
    print(f"  Anchor L{layer}_S{seed}: dc={dc:+.4f} (exp {exp['expected_delta_cos']:+.4f}) "
          f"diff={dc-exp['expected_delta_cos']:+.4f} OK={ok}", flush=True)

all_anchors_ok = all(v["within_tolerance"] for v in anchor_check.values())

# ─── Classification ─────────────────────────────────────────────────────────────

if all(r["memory_positive"] for r in results) and n_severe == 0 and n_cos_fail == 0 and n_mae_fail == 0:
    classification = "PASS_31BL_CORRECTED_Q2K_SMALL_AGGREGATE_VALIDATED"
elif all(r["memory_positive"] for r in results) and n_severe == 0:
    classification = "PARTIAL_31BL_CORRECTED_Q2K_MINOR_FAILURES"
elif any(not r["memory_positive"] for r in results):
    classification = "PARTIAL_31BL_CORRECTED_Q2K_MEMORY_FAIL"
elif n_severe > 0:
    classification = "PARTIAL_31BL_CORRECTED_Q2K_SEVERE_OUTLIER"
else:
    classification = "PARTIAL_31BL_CORRECTED_Q2K_MINOR_FAILURES"

print(f"\nClassification: {classification}", flush=True)

# ─── Write results JSON ─────────────────────────────────────────────────────────

output = {
    "classification": classification,
    "policy": {
        "q2k_mode": "corrected_ceil_per_row",
        "residual_families": ["ffn_up", "ffn_gate"],
        "down_family_residual": False,
        "k_pct": K_PCT,
        "alpha": ALPHA,
    },
    "layers": LAYERS,
    "seeds": SEEDS,
    "total_pairs": total_pairs,
    "aggregate": {
        "n_cosine_improved": n_cos_imp,
        "n_cosine_nonnegative": n_cos_nneg,
        "n_MAE_improved": n_mae_imp,
        "n_memory_positive": n_mem_pos,
        "n_severe_regressions": n_severe,
        "n_cosine_failures": n_cos_fail,
        "n_MAE_failures": n_mae_fail,
        "worst_pair_layer": worst_pair["layer"],
        "worst_pair_seed": worst_pair["seed"],
        "worst_delta_cos": worst_pair["delta_cos"],
        "best_pair_layer": best_pair["layer"],
        "best_pair_seed": best_pair["seed"],
        "best_delta_cos": best_pair["delta_cos"],
        "mean_delta_cos": round(mean_dc, 6),
        "median_delta_cos": round(median_dc, 6),
        "mean_MAE_improvement": round(mean_mi, 6),
        "min_margin": min_margin,
        "aggregate_margin": sum(r["margin"] for r in results),
    },
    "layer_specific": {
        f"layer_{layer}": layer_stats(layer) for layer in LAYERS
    },
    "anchor_consistency": anchor_check,
    "all_anchors_within_tolerance": all_anchors_ok,
    "elapsed_seconds": round(elapsed, 1),
    "pairs": results,
}

out_path = os.path.join(REPO_DIR, "src", "results", "PHASE31BL_CORRECTED_Q2K_SMALL_AGGREGATE.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults written to {out_path}", flush=True)
