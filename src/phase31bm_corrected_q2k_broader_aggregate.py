#!/usr/bin/env python3
"""
Phase 31BM — Corrected Q2_K Broader Aggregate Validation

Validates: corrected_ceil_per_row Q2_K, up+gate residual k=0.5%, no down residual.

Route A: all 24 layers × seeds 0–15 = 384 pairs
Cache: W_ref, Q2_K W_low, SDIR residuals — all per-layer, reused across seeds.

Environment:
  SDI_GGUF_MODEL_PATH   — path to Qwen2.5-0.5B GGUF
  SDI_LLAMA_CPP_ROOT    — path to llama.cpp root
  SDI_LLAMA_CPP_LIB     — path to libggml-base.so  (optional)

Outputs:
  src/results/PHASE31BM_CORRECTED_Q2K_BROADER_AGGREGATE.json
"""

import os, sys, json, time, re
import numpy as np

REPO_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_ROOT = os.environ.get("SDI_LLAMA_CPP_ROOT", "")
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
sys.path.insert(0, os.path.join(LLAMA_ROOT, "gguf-py"))

from gguf import GGUFReader
from gguf.quants import dequantize
from q2k_backend import quantize_q2k_f32_to_bytes, dequantize_q2k_bytes_to_f32
from phase31x_manifest_runtime import cosine, encode_sdir, decode_sdir

GGUF_PATH = os.environ.get("SDI_GGUF_MODEL_PATH", "")
if not GGUF_PATH or not os.path.exists(GGUF_PATH):
    raise FileNotFoundError(f"SDI_GGUF_MODEL_PATH not set or file not found: {GGUF_PATH!r}")

Q4_BUDGET_FAMILY = 2_179_072
Q4_BUDGET_LAYER  = 3 * Q4_BUDGET_FAMILY
K_PCT = 0.5
ALPHA = 1.0
LAYERS = list(range(24))
SEEDS = list(range(16))

# ─── Helpers ───────────────────────────────────────────────────────────────────

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -709, 709)))

def mlp_full(X, Wg, Wu, Wd):
    return (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T

def cosine_sim(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0

def mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())

# ─── Load all 24 layers ─────────────────────────────────────────────────────────

print("Loading GGUF weights for all 24 layers…", flush=True)
reader = GGUFReader(GGUF_PATH)

def load_raw(layer: int, fam: str) -> np.ndarray:
    t = next(t for t in reader.tensors if t.name == f"blk.{layer}.{fam}.weight")
    return dequantize(t.data, t.tensor_type).astype(np.float32)

wref = {}   # (layer, fam) -> ndarray
for layer in LAYERS:
    for fam in ["ffn_gate", "ffn_up", "ffn_down"]:
        wref[(layer, fam)] = load_raw(layer, fam)
    print(f"  L{layer} loaded", flush=True)

# ─── Pre-compute Q2_K dequantized weights ─────────────────────────────────────

print("Computing Q2_K corrected dequantized weights…", flush=True)
wlow = {}
for key, W in wref.items():
    q2k_bytes = quantize_q2k_f32_to_bytes(W, mode="corrected_ceil_per_row")
    wlow[key] = dequantize_q2k_bytes_to_f32(q2k_bytes, *W.shape,
                                            mode="corrected_ceil_per_row")

# ─── Pre-compute SDIR (up+gate only) ───────────────────────────────────────────

print("Encoding SDIR residuals (k=0.5%, gate+up only)…", flush=True)
resid = {}   # (layer, fam) -> ndarray residual
sdir  = {}   # (layer, fam) -> bytes | None

for layer in LAYERS:
    for fam in ["ffn_gate", "ffn_up"]:
        resid[(layer, fam)] = wref[(layer, fam)] - wlow[(layer, fam)]
        sdir[(layer, fam)]  = encode_sdir(resid[(layer, fam)], k_pct=K_PCT)
        print(f"  sdir[L{layer},{fam}]: {len(sdir[(layer,fam)]):,} bytes", flush=True)
    sdir[(layer, "ffn_down")] = None   # no down residual

# ─── Memory accounting ──────────────────────────────────────────────────────────

def layer_margin(layer: int) -> dict:
    up_q  = len(quantize_q2k_f32_to_bytes(wref[(layer,"ffn_up")],   mode="corrected_ceil_per_row"))
    gt_q  = len(quantize_q2k_f32_to_bytes(wref[(layer,"ffn_gate")],mode="corrected_ceil_per_row"))
    dn_q  = len(quantize_q2k_f32_to_bytes(wref[(layer,"ffn_down")],mode="corrected_ceil_per_row"))
    up_s  = len(sdir[(layer,"ffn_up")])    if sdir[(layer,"ffn_up")]    is not None else 0
    gt_s  = len(sdir[(layer,"ffn_gate")]) if sdir[(layer,"ffn_gate")] is not None else 0
    total = up_q + gt_q + dn_q + up_s + gt_s
    return {"total": total, "margin": Q4_BUDGET_LAYER - total,
            "memory_positive": Q4_BUDGET_LAYER >= total}

# Verify all layers memory-positive before starting
print("\nPre-flight memory check (all 24 layers):", flush=True)
for layer in LAYERS:
    m = layer_margin(layer)
    tag = "✓" if m["memory_positive"] else "✗ FAIL"
    print(f"  L{layer}: total={m['total']:,} margin={m['margin']:,} {tag}", flush=True)
    if not m["memory_positive"]:
        raise RuntimeError(f"L{layer} is memory-negative — aborting")

# ─── Run 384-pair aggregate ────────────────────────────────────────────────────

print(f"\nRunning Route A: 24 layers × 16 seeds = 384 pairs…", flush=True)
t_start = time.time()
results = []

for li, layer in enumerate(LAYERS):
    # Pre-compute Y_ref per seed (reused for all families at this seed)
    layer_mem = layer_margin(layer)
    dc_list = []; md_list = []

    for si, seed in enumerate(SEEDS):
        pair_idx = li * 16 + si + 1

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((1, 896)).astype(np.float32)

        Wg_ref = wref[(layer, "ffn_gate")]
        Wu_ref = wref[(layer, "ffn_up")]
        Wd_ref = wref[(layer, "ffn_down")]
        Wg_lo  = wlow[(layer, "ffn_gate")]
        Wu_lo  = wlow[(layer, "ffn_up")]
        Wd_lo  = wlow[(layer, "ffn_down")]

        # Reference
        Y_ref = mlp_full(X, Wg_ref, Wu_ref, Wd_ref)

        # Low (Q2_K only)
        Y_low = mlp_full(X, Wg_lo, Wu_lo, Wd_lo)

        # Substitute: apply gate+up SDIR residual
        Wg_sub = Wg_lo + decode_sdir(sdir[(layer,"ffn_gate")]) \
                 if sdir[(layer,"ffn_gate")] is not None else Wg_lo
        Wu_sub = Wu_lo + decode_sdir(sdir[(layer,"ffn_up")]) \
                 if sdir[(layer,"ffn_up")]   is not None else Wu_lo
        Wd_sub = Wd_lo
        Y_sub  = mlp_full(X, Wg_sub, Wu_sub, Wd_sub)

        cl = cosine_sim(Y_ref.ravel(), Y_low.ravel())
        cs = cosine_sim(Y_ref.ravel(), Y_sub.ravel())
        ml = mae(Y_ref, Y_low)
        ms = mae(Y_ref, Y_sub)

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
            "memory_bytes": layer_mem["total"],
            "margin": layer_mem["margin"],
            "memory_positive": layer_mem["memory_positive"],
            "severe_regression": (cs - cl) < -0.05,
            "cosine_improved": (cs - cl) > 0,
            "cosine_nonnegative": (cs - cl) >= 0,
            "MAE_improved": (ms - ml) < 0,
        }
        results.append(pair)
        dc_list.append(cs - cl); md_list.append(ms - ml)

        if pair_idx % 50 == 0 or pair_idx == 384:
            print(f"  progress: {pair_idx}/384 ({pair_idx/384*100:.1f}%)", flush=True)

    elapsed = time.time() - t_start
    mean_dc = sum(dc_list) / len(dc_list)
    worst_dc_seed = SEEDS[dc_list.index(min(dc_list))]
    print(f"  L{layer} done: mean_dc={mean_dc:+.4f} worst=S{worst_dc_seed}({min(dc_list):+.4f}) "
          f"[elapsed {elapsed:.0f}s]", flush=True)

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s ({elapsed/384:.2f}s/pair)", flush=True)

# ─── Aggregate stats ────────────────────────────────────────────────────────────

total = len(results)
n_cos_imp  = sum(1 for r in results if r["cosine_improved"])
n_cos_nneg = sum(1 for r in results if r["cosine_nonnegative"])
n_mae_imp  = sum(1 for r in results if r["MAE_improved"])
n_mem_pos  = sum(1 for r in results if r["memory_positive"])
n_severe   = sum(1 for r in results if r["severe_regression"])
n_cos_fail = sum(1 for r in results if not r["cosine_improved"])
n_mae_fail = sum(1 for r in results if not r["MAE_improved"])

dcs = [r["delta_cos"] for r in results]
mds = [r["MAE_delta"]  for r in results]

worst_dc = min(results, key=lambda r: r["delta_cos"])
best_dc  = max(results, key=lambda r: r["delta_cos"])
worst_md = min(results, key=lambda r: r["MAE_delta"])

agg = {
    "route": "A",
    "total_pairs": total,
    "n_cosine_improved": n_cos_imp,
    "n_cosine_nonnegative": n_cos_nneg,
    "n_MAE_improved": n_mae_imp,
    "n_memory_positive": n_mem_pos,
    "n_severe_regressions": n_severe,
    "n_cosine_failures": n_cos_fail,
    "n_MAE_failures": n_mae_fail,
    "worst_pair_layer": worst_dc["layer"],
    "worst_pair_seed": worst_dc["seed"],
    "worst_delta_cos": worst_dc["delta_cos"],
    "worst_MAE_pair_layer": worst_md["layer"],
    "worst_MAE_pair_seed": worst_md["seed"],
    "worst_MAE_delta": worst_md["MAE_delta"],
    "best_pair_layer": best_dc["layer"],
    "best_pair_seed": best_dc["seed"],
    "best_delta_cos": best_dc["delta_cos"],
    "mean_delta_cos": round(sum(dcs)/len(dcs), 6),
    "median_delta_cos": round(float(np.median(dcs)), 6),
    "mean_MAE_improvement": round(sum(r["MAE_improvement"] for r in results)/len(results), 6),
    "min_margin": min(r["margin"] for r in results),
    "aggregate_margin": sum(r["margin"] for r in results),
    "elapsed_seconds": round(elapsed, 1),
}

print(f"\nAggregate: {n_cos_imp}/{total} cos_imp, {n_mae_imp}/{total} mae_imp, "
      f"{n_mem_pos}/{total} mem_pos, {n_severe} severe, "
      f"worst L{worst_dc['layer']}S{worst_dc['seed']} dc={worst_dc['delta_cos']}", flush=True)

# ─── Layer-specific ─────────────────────────────────────────────────────────────

def layer_stats(layer):
    rl = [r for r in results if r["layer"] == layer]
    dcl = [r["delta_cos"] for r in rl]
    mdl = [r["MAE_delta"]  for r in rl]
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
        "worst_MAE_seed": min(rl, key=lambda r: r["MAE_delta"])["seed"],
        "worst_MAE_delta": min(mdl),
        "best_seed": max(rl, key=lambda r: r["delta_cos"])["seed"],
        "best_delta_cos": max(dcl),
        "mean_delta_cos": round(sum(dcl)/len(dcl), 6),
        "mean_MAE_improvement": round(sum(r["MAE_improvement"] for r in rl)/len(rl), 6),
        "min_margin": min(r["margin"] for r in rl),
    }

layer_specific = {f"layer_{layer}": layer_stats(layer) for layer in LAYERS}

# ─── Minor failure tracking ─────────────────────────────────────────────────────

known_failures = {
    "L21_S10": {"type": "cosine_failure", "expected_dc": -0.0294, "expected_severe": False},
    "L2_S13":  {"type": "MAE_regression", "expected_md": +0.00783, "expected_severe": False},
}

_MF_KEY_RE = re.compile(r'^L(?P<layer>\d+)_S(?P<seed>\d+)$')

mf_track = {}
for key, info in known_failures.items():
    m = _MF_KEY_RE.match(key)
    if not m:
        raise ValueError(f"Invalid known failure key format: {key!r}")
    layer = int(m.group('layer'))
    seed  = int(m.group('seed'))
    r = next((x for x in results if x["layer"]==layer and x["seed"]==seed), None)
    if r is None:
        mf_track[key] = {"status": "NOT_FOUND_IN_RUN"}
        continue
    dc = r["delta_cos"]; md = r["MAE_delta"]
    if info["type"] == "cosine_failure":
        ok = (dc < 0) and (not r["severe_regression"]) and (abs(dc - info["expected_dc"]) < 0.01)
    else:
        ok = (md > 0) and (not r["severe_regression"]) and (abs(md - info["expected_md"]) < 0.001)
    mf_track[key] = {
        "type": info["type"],
        "expected_dc": info["expected_dc"] if info["type"]=="cosine_failure" else None,
        "expected_md": info["expected_md"] if info["type"]=="MAE_regression" else None,
        "observed_dc": dc,
        "observed_md": md,
        "reproduced": ok,
        "still_minor": not r["severe_regression"],
        "observed_severe": r["severe_regression"],
    }
    print(f"  MF {key}: dc={dc:+.4f} md={md:+.5f} reproduced={ok}", flush=True)

# New failures
new_cos_fail  = [r for r in results if not r["cosine_improved"]]
new_mae_fail = [r for r in results if not r["MAE_improved"]]
new_severe   = [r for r in results if r["severe_regression"]]
all_failures = new_cos_fail + new_mae_fail

# Cluster analysis
from collections import Counter
layer_dist  = Counter(r["layer"] for r in all_failures)
seed_dist   = Counter(r["seed"]  for r in all_failures)
metric_dist = Counter("cosine" if not r["cosine_improved"] else "MAE" for r in all_failures)

print(f"\nNew failures: {len(new_cos_fail)} cosine, {len(new_mae_fail)} MAE, {len(new_severe)} severe", flush=True)
print(f"  Layer dist: {dict(layer_dist.most_common(5))}", flush=True)
print(f"  Seed dist:  {dict(seed_dist.most_common(5))}", flush=True)

# ─── Policy status ─────────────────────────────────────────────────────────────

rate_cos = n_cos_imp / total
rate_mae = n_mae_imp / total
print(f"\nImprovement rates: cosine={rate_cos:.3f} MAE={rate_mae:.3f}", flush=True)

if n_severe == 0 and n_mem_pos == total and rate_cos >= 0.95 and rate_mae >= 0.95:
    policy_status = "strong"
    print("Policy status: STRONG VALIDATION", flush=True)
elif n_severe == 0 and n_mem_pos == total and rate_cos >= 0.90 and rate_mae >= 0.90:
    policy_status = "partial"
    print("Policy status: PARTIAL VALIDATION", flush=True)
elif n_severe > 0 or n_mem_pos < total:
    policy_status = "concern"
    print("Policy status: CONCERN", flush=True)
else:
    policy_status = "partial"
    print("Policy status: PARTIAL VALIDATION (default)", flush=True)

# ─── Classification ─────────────────────────────────────────────────────────────

if policy_status == "strong":
    classification = "PASS_31BM_CORRECTED_Q2K_BROADER_AGGREGATE_VALIDATED"
elif policy_status == "partial" and len(all_failures) <= 10:
    classification = "PARTIAL_31BM_CORRECTED_Q2K_MINOR_FAILURES_STABLE"
elif policy_status == "concern" or len(all_failures) > 10:
    classification = "PARTIAL_31BM_CORRECTED_Q2K_FAILURE_CLUSTER_FOUND"
elif n_severe > 0:
    classification = "PARTIAL_31BM_CORRECTED_Q2K_SEVERE_OUTLIER"
else:
    classification = "PARTIAL_31BM_CORRECTED_Q2K_MINOR_FAILURES_STABLE"

print(f"\nClassification: {classification}", flush=True)

# ─── Write JSON ────────────────────────────────────────────────────────────────

output = {
    "classification": classification,
    "policy_status": policy_status,
    "route": "A",
    "policy": {
        "q2k_mode": "corrected_ceil_per_row",
        "residual_families": ["ffn_up", "ffn_gate"],
        "down_family_residual": False,
        "k_pct": K_PCT,
        "alpha": ALPHA,
    },
    "layers": LAYERS,
    "seeds": SEEDS,
    "aggregate": agg,
    "layer_specific": layer_specific,
    "minor_failure_tracking": {
        "known_31BL_failures": mf_track,
        "new_cosine_failures": [{"layer":r["layer"],"seed":r["seed"],"delta_cos":r["delta_cos"]} for r in new_cos_fail],
        "new_MAE_failures":    [{"layer":r["layer"],"seed":r["seed"],"MAE_delta":r["MAE_delta"]}  for r in new_mae_fail],
        "severe_regressions":  [{"layer":r["layer"],"seed":r["seed"],"delta_cos":r["delta_cos"]} for r in new_severe],
        "cluster_analysis": {
            "layer_distribution": dict(layer_dist),
            "seed_distribution": dict(seed_dist),
            "metric_distribution": dict(metric_dist),
        }
    },
    "pairs": results,
}

out_path = os.path.join(REPO_DIR, "src", "results",
                         "PHASE31BM_CORRECTED_Q2K_BROADER_AGGREGATE.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults written to {out_path}", flush=True)
