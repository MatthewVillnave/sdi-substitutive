#!/usr/bin/env python3
"""
Phase 31AM — Low-k Q2_K + Sparse Residual Viability Probe
Repo: sdi-substitutive, HEAD: e443756

Goal: Test whether low-k residuals (k <= 3%) that fit inside Q4_budget
      alongside Q2_K W_low can improve full MLP approximation.

Data source: W_ref extracted from Q4_K_M model via GGUFReader + Q5_0.dequantize.
The toy probe used Q4_K_M model as reference (Q5_0 quantized FFN tensors).
Q2_K model uses IQ4_NL/Q3_K mixed — not a pure Q2_K reference.
Using Q4_K_M model Q5_0 dequantized weights as W_ref proxy.
"""
import json, os, sys
from pathlib import Path

import numpy as np

# gguf path must be set before gguf import
sys.path.insert(0, "/home/matthew-villnave/llama.cpp/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize

REPO = Path("/home/matthew-villnave/sdi-substitutive")
MANIFEST_PATH = REPO / "data/phase31aj_mlp_probe/manifest.json"
RESULTS_JSON  = REPO / "results/PHASE31AM_LOWK_Q2K_RESIDUAL_VIABILITY.json"

# ── Constants from 31AL-R ───────────────────────────────────────────────────────
N_ELEMENTS       = 4_358_144  # 4864 * 896
N_BLOCKS         = N_ELEMENTS // 256   # 17,024
Q4_BUDGET_FL     = N_ELEMENTS * 4 // 8   # 2,179,072 per family-layer
Q2_K_BYTES_FL    = 12 + N_BLOCKS * 84   # 1,430,028 per family-layer
Q4_BUDGET_TOTAL  = Q4_BUDGET_FL * 3 * 6  # 37.41 MB total

SDIR_BITMAP_FL   = N_ELEMENTS // 8       # 544,768 bytes (always)


def residual_bytes_fl(k_pct):
    nnz = int(N_ELEMENTS * k_pct / 100)
    return SDIR_BITMAP_FL + nnz * 2  # bitmap + fp16 per nonzero


def int8_residual_bytes_fl(k_pct):
    nnz = int(N_ELEMENTS * k_pct / 100)
    return SDIR_BITMAP_FL + nnz  # bitmap + int8 per nonzero


# ── Load manifest ──────────────────────────────────────────────────────────────
def load_manifest():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    m["bundle_dir"] = str(MANIFEST_PATH.parent)
    return m


def abs_path(manifest, rel):
    return os.path.join(manifest["bundle_dir"], rel)


# ── W_ref extraction from GGUF ─────────────────────────────────────────────────
sys.path.insert(0, "/home/matthew-villnave/llama.cpp/gguf-py")
from gguf import GGUFReader

MODEL_PATH = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Cache for loaded tensors
_WREF_CACHE = {}


def get_W_ref_gguf(layer: int, family: str):
    """Load W_ref from Q4_K_M model via GGUFReader + quant.dequantize."""
    key = (layer, family)
    if key in _WREF_CACHE:
        return _WREF_CACHE[key]

    name_map = {
        "ffn_up":    f"blk.{layer}.ffn_up.weight",
        "ffn_gate":  f"blk.{layer}.ffn_gate.weight",
        "ffn_down":  f"blk.{layer}.ffn_down.weight",
    }
    tensor_name = name_map[family]

    reader = GGUFReader(MODEL_PATH)
    t = next(t_ for t_ in reader.tensors if t_.name == tensor_name)

    raw = np.array(t.data).reshape(-1)
    qt_name = t.tensor_type.name

    qt_map = {
        "Q5_0": ("gguf.quants", "Q5_0"),
        "Q5_1": ("gguf.quants", "Q5_1"),
        "Q6_K": ("gguf.quants", "Q6_K"),
        "Q4_0": ("gguf.quants", "Q4_0"),
        "Q4_1": ("gguf.quants", "Q4_1"),
    }

    if qt_name in qt_map:
        mod_name, cls_name = qt_map[qt_name]
        mod = __import__(mod_name, fromlist=[cls_name])
        qcls = getattr(mod, cls_name)
        deq = qcls.dequantize(raw).astype(np.float32)
    else:
        deq = raw.view(np.float16).astype(np.float32)

    mfest = load_manifest()
    shape = next(l["shape"] for l in mfest["layers"] if l["layer"] == layer and l["family"] == family)
    deq = dequantize(raw.reshape(-1), t.tensor_type)
    W = deq.reshape(shape[::-1])
    if W.shape != (shape[0], shape[1]):
        W = W.T
    _WREF_CACHE[key] = W
    return W


# ── Metrics ─────────────────────────────────────────────────────────────────────
def cosine(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(num / den) if den > 0 else 1.0


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def max_error(a, b):
    return float(np.max(np.abs(a - b)))


# ── Q2-like W_low simulation ──────────────────────────────────────────────────
def simulate_q2k_wlow(W_ref):
    """
    Simulate Q2-like W_low by quantizing W_ref to 2-bit per block.
    Returns (W_low_sim, n_bytes).
    """
    d_out, d_in = W_ref.shape
    block_size = 16
    n_blocks = d_in // block_size
    blocks = W_ref.reshape(d_out, n_blocks, block_size)
    scales = np.max(np.abs(blocks), axis=-1, keepdims=True)
    scales = np.maximum(scales, 1e-8)
    q = np.round(blocks / scales * 1.5)
    q = np.clip(q, -2, 2).astype(np.int8)
    nibbles = d_out * n_blocks * block_size // 2
    fp16_scales = n_blocks * 2
    total_bytes = nibbles + fp16_scales
    W_rec = (q.astype(np.float32) / 1.5 * scales).reshape(d_out, d_in)
    return W_rec, total_bytes


# ── Residual synthesis ──────────────────────────────────────────────────────────
def synthesize_residual(W_ref, k_pct):
    if k_pct == 0:
        return np.zeros_like(W_ref), None
    flat = W_ref.flatten()
    n_nz = int(len(flat) * k_pct / 100)
    abs_flat = np.abs(flat)
    top_idx = np.argpartition(abs_flat, -n_nz)[-n_nz:]
    vals = flat[top_idx]
    # Return as COO (indices, values)
    result = np.zeros_like(flat)
    result[top_idx] = vals
    return result.reshape(W_ref.shape), top_idx


def apply_residual(W, resid, idx=None):
    if idx is None:
        return W + resid
    W_flat = W.flatten()
    W_flat = W_flat.copy()
    W_flat[idx] += resid.flatten()[idx]
    return W_flat.reshape(W.shape)


# ── Main sweep ──────────────────────────────────────────────────────────────────
def run_sweep():
    manifest = load_manifest()

    print("### Data Source")
    print(f"  W_ref: Q4_K_M model Q5_0/Q6_K dequantized via GGUFReader")
    print(f"  Q2-like W_low: SIMULATED (block16 2-bit quantization of W_ref)")
    print(f"  Classification note: PARTIAL_Q2_SIM_ONLY — simulated quality, not actual GGUF Q2_K decode")
    print()

    # Budget table
    print("### Budget Table (per family-layer)")
    print(f"Q4_budget/fl:   {Q4_BUDGET_FL:>12,} bytes")
    print(f"Q2_K W_low/fl: {Q2_K_BYTES_FL:>12,} bytes = {Q2_K_BYTES_FL*3*6/1024/1024:.2f}MB total")
    print()
    print(f"{'k%':>4}  {'SDIR/fl':>10}  {'int8/fl':>10}  {'Q2K+SDIR tot':>13}  {'Q2K+int8 tot':>13}  {'SDIR marg':>10}  {'int8 marg':>10}")
    print("  " + "-" * 78)
    for k in [0, 0.5, 1, 2, 3, 4, 5]:
        sdir_fl = residual_bytes_fl(k)
        int8_fl = int8_residual_bytes_fl(k)
        q2k_sdir_tot = Q2_K_BYTES_FL * 3 * 6 + sdir_fl * 3 * 6
        q2k_int8_tot = Q2_K_BYTES_FL * 3 * 6 + int8_fl * 3 * 6
        m_sdir = Q4_BUDGET_TOTAL - q2k_sdir_tot
        m_int8 = Q4_BUDGET_TOTAL - q2k_int8_tot
        print(f"  {k:>2}%  {sdir_fl:>10,}  {int8_fl:>10,}  "
              f"{q2k_sdir_tot/1024/1024:>11.2f}MB  {q2k_int8_tot/1024/1024:>11.2f}MB  "
              f"{m_sdir/1024:>8.0f}KB  {m_int8/1024:>8.0f}KB")
    print()

    # Policies
    policies = []
    for base_k in [0, 0.5, 1, 2, 3]:
        policies.append((f"uniform_{base_k}", base_k, base_k, base_k))
    for fam_k in [0.5, 1, 2, 3]:
        policies.append((f"up_{fam_k}_g0_d0", fam_k, 0, 0))
        policies.append((f"up_0_g{fam_k}_d0", 0, fam_k, 0))
        policies.append((f"up_0_g0_d{fam_k}", 0, 0, fam_k))
    for k in [1, 2, 3]:
        policies.append((f"up{k}_g{k}_d0", k, k, 0))
        policies.append((f"up{k}_g0_d{k}", k, 0, k))
        policies.append((f"up0_g{k}_d{k}", 0, k, k))
    for a, g, d in [(1, 1, 3), (1, 3, 1), (3, 1, 1)]:
        policies.append((f"asym_{a}_{g}_{d}", a, g, d))

    results = []
    print(f"### Running Low-k Residual Sweep ({len(policies)} policies × 6 layers)")
    print()

    for pol_name, up_k, gate_k, down_k in policies:
        k_pcts = {"ffn_up": up_k, "ffn_gate": gate_k, "ffn_down": down_k}

        resid_sdir_fl = {f: residual_bytes_fl(k) for f, k in k_pcts.items()}
        resid_sdir_tot = sum(v * 6 for v in resid_sdir_fl.values())
        q2k_total = Q2_K_BYTES_FL * 3 * 6
        margin = Q4_BUDGET_TOTAL - q2k_total - resid_sdir_tot
        mem_pos = margin >= 0

        cos_low_acc, cos_sub_acc = 0.0, 0.0
        mae_low_acc, mae_sub_acc = 0.0, 0.0
        mx_low_acc = 0.0
        n = 0

        for layer_idx in range(6):
            W_ref_up = get_W_ref_gguf(layer_idx, "ffn_up")
            W_ref_gate = get_W_ref_gguf(layer_idx, "ffn_gate")
            W_ref_down = get_W_ref_gguf(layer_idx, "ffn_down")

            W_q2k_up, _ = simulate_q2k_wlow(W_ref_up)
            W_q2k_gate, _ = simulate_q2k_wlow(W_ref_gate)
            W_q2k_down, _ = simulate_q2k_wlow(W_ref_down)

            R_up, idx_up = synthesize_residual(W_ref_up, up_k)
            R_gate, idx_gate = synthesize_residual(W_ref_gate, gate_k)
            R_down, idx_down = synthesize_residual(W_ref_down, down_k)

            for W_ref, W_q2k, R, idx in [
                (W_ref_up, W_q2k_up, R_up, idx_up),
                (W_ref_gate, W_q2k_gate, R_gate, idx_gate),
                (W_ref_down, W_q2k_down, R_down, idx_down),
            ]:
                cl = cosine(W_ref, W_q2k)
                cos_low_acc += cl
                if pol_name == "uniform_0":
                    print("DEBUG layer=%d family=%s cl=%.6f" % (layer_idx, str(W_ref.shape), cl))
                mae_low_acc += mae(W_ref, W_q2k)
                mx_low_acc += max_error(W_ref, W_q2k)
                n += 1
                if idx is not None:
                    W_sub = apply_residual(W_q2k, R, idx)
                    cos_sub_acc += cosine(W_ref, W_sub)
                    mae_sub_acc += mae(W_ref, W_sub)
                else:
                    cos_sub_acc += cosine(W_ref, W_q2k)
                    mae_sub_acc += mae(W_ref, W_q2k)

        avg_cos_low = cos_low_acc / n
        avg_cos_sub = cos_sub_acc / n
        avg_mae_low = mae_low_acc / n
        avg_mae_sub = mae_sub_acc / n
        avg_mx_low = mx_low_acc / n
        delta_cos = avg_cos_sub - avg_cos_low
        delta_mae = avg_mae_low - avg_mae_sub  # positive = sub is better

        results.append({
            "policy": pol_name,
            "k_pct": k_pcts,
            "memory_positive": mem_pos,
            "margin_kb": margin / 1024,
            "cos_low": avg_cos_low,
            "cos_sub": avg_cos_sub,
            "delta_cos": delta_cos,
            "mae_low": avg_mae_low,
            "mae_sub": avg_mae_sub,
            "delta_mae": delta_mae,
            "mx_low": avg_mx_low,
        })

    results.sort(key=lambda r: (not r["memory_positive"], -r["delta_cos"]))

    print("  " + "-" * 90)
    print("   Policy                      Mem+   Margin    cos_low   cos_sub       d_cos    mae_low   mae_sub       d_mae")
    print("  " + "-" * 90)
    for r in results:
        dc = r["delta_cos"]
        dm = r["delta_mae"]
        mp = "Y" if r["memory_positive"] else "N"
        print("   %-23s  %5s  %6.0fKB  %9.6f  %9.6f  %+10.6f  %9.6f  %9.6f  %+10.6f" % (
            r["policy"], mp, r["margin_kb"],
            r["cos_low"], r["cos_sub"], dc,
            r["mae_low"], r["mae_sub"], dm))

    # Pareto
    print()
    print("### Pareto Frontier")
    mem_pos_results = [r for r in results if r["memory_positive"]]
    pareto = [r for r in mem_pos_results if r["delta_cos"] > 0 and r["delta_mae"] > 0]
    pareto.sort(key=lambda x: (-x["delta_cos"], -x["delta_mae"]))

    if pareto:
        print("  Memory-positive, delta_cos > 0, delta_mae > 0:")
        for p in pareto:
            print(f"    {p['policy']}: margin={p['margin_kb']:.0f}KB, d_cos={p['delta_cos']:+.6f}, d_mae={p['delta_mae']:+.6f}")
    else:
        print("  No policy satisfies all three criteria.")

    print()
    print("  Memory-positive policies (by delta_cos):")
    if mem_pos_results:
        for p in sorted(mem_pos_results, key=lambda x: -x["delta_cos"]):
            print(f"    {p['policy']}: margin={p['margin_kb']:.0f}KB, d_cos={p['delta_cos']:+.6f}, d_mae={p['delta_mae']:+.6f}")
    else:
        print("    NO memory-positive policies at any k%.")

    # Classification
    print()
    print("### Classification")
    if not mem_pos_results:
        classification = "PARTIAL_MEMORY_STILL_FAILS"
        print(f"  {classification}: Even k<=3% residual policies fail memory check.")
    else:
        best_cos = max(mem_pos_results, key=lambda x: x["delta_cos"])
        best_mae = max(mem_pos_results, key=lambda x: x["delta_mae"])
        print(f"  Best cosine (memory+): {best_cos['policy']} d_cos={best_cos['delta_cos']:+.6f}")
        print(f"  Best MAE (memory+): {best_mae['policy']} d_mae={best_mae['delta_mae']:+.6f}")
        if pareto:
            if pareto[0]["delta_cos"] < 1e-5:
                classification = "PARTIAL_LOWK_GAIN_TOO_SMALL"
                print(f"  {classification}: Gain is too small to justify.")
            else:
                classification = "PASS_LOWK_MEMORY_POSITIVE_POLICY_FOUND"
                print(f"  {classification}: Memory-positive with meaningful improvement.")
        elif any(r["delta_cos"] > 0 for r in mem_pos_results):
            classification = "PARTIAL_LOWK_COS_ONLY"
            print(f"  {classification}: Cosine improves but MAE does not.")
        else:
            classification = "PARTIAL_LOWK_GAIN_TOO_SMALL"
            print(f"  {classification}.")

    print(f"\n  NOTE: Quality claims are SIMULATED (Q2-like approximation).")
    print(f"  PARTIAL_Q2_SIM_ONLY applies.")

    # Save JSON
    output = {
        "phase": "31AM",
        "classification": classification,
        "wref_source": "Q4_K_M model Q5_0/Q6_K dequantized via GGUFReader",
        "q2_decode_note": "SIMULATED Q2-like W_low — actual GGUF Q2_K decode unavailable",
        "budget": {
            "Q4_budget_total_mb": Q4_BUDGET_TOTAL / 1024 / 1024,
            "Q2_K_total_mb": Q2_K_BYTES_FL * 3 * 6 / 1024 / 1024,
        },
        "sweep_results": results,
        "pareto": pareto,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_JSON}")

    return output


if __name__ == "__main__":
    run_sweep()
