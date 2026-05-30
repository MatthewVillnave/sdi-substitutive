#!/usr/bin/env python3
"""
Phase 31AK — Full MLP Artifact Budget / Economics Fix
Scope: Determine whether any full MLP substitutive policy can be memory-positive
       under current encoding. Economics/policy sweep.

Repo: sdi-substitutive
HEAD: 046ddd7
Requires: data/phase31aj_mlp_probe/ (ffn_up k=9%, ffn_down k=9%, ffn_gate k=12%)
          data/PHASE31I_activations.npz
"""
import os, sys, json, time, struct
from typing import Dict, List, Tuple, Optional
import numpy as np

REPO = "/home/matthew-villnave/sdi-substitutive"
DATA = os.path.join(REPO, "data")
BUNDLE_DIR = os.path.join(DATA, "phase31aj_mlp_probe")
MANIFEST_PATH = os.path.join(BUNDLE_DIR, "manifest.json")
ACTIVATION_PATH = os.path.join(DATA, "PHASE31I_activations.npz")
RESULT_PATH = os.path.join(REPO, "results", "PHASE31AK_MLP_ARTIFACT_BUDGET_ECONOMICS.json")
DOC_PATH = os.path.join(REPO, "docs", "PHASE31AK_MLP_ARTIFACT_BUDGET_ECONOMICS.md")
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)

# ── Math helpers ──────────────────────────────────────────────────────────────
def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40, 40))))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def maxabs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))

# ── Artifact parsing ───────────────────────────────────────────────────────────
SDIW_HEADER = '<4sHHIIII'
SDIW_HEADER_BYTES = struct.calcsize(SDIW_HEADER)
SDIR_HEADER = '<4sHHIIIIHH'
SDIR_HEADER_BYTES = struct.calcsize(SDIR_HEADER)
BLOCK_SIZE = 32

def parse_sdiw(packed_bytes: bytes, scale_bytes: bytes,
                X_batch: np.ndarray, d_out: int, d_in: int) -> np.ndarray:
    scales_arr = np.frombuffer(scale_bytes, dtype=np.float16)
    blocks_per_row = d_in // BLOCK_SIZE
    Y = np.zeros((X_batch.shape[0], d_out), dtype=np.float32)
    for row in range(d_out):
        for block in range(blocks_per_row):
            block_idx = row * blocks_per_row + block
            scale = float(scales_arr[block_idx])
            in_byte = block_idx * (BLOCK_SIZE // 2)
            for i in range(BLOCK_SIZE // 2):
                col0 = int(block * BLOCK_SIZE + 2*i)
                col1 = int(block * BLOCK_SIZE + 2*i + 1)
                val0 = (float(packed_bytes[in_byte + i] & 0x0F) - 8.0) * scale
                val1 = (float((packed_bytes[in_byte + i] >> 4) & 0x0F) - 8.0) * scale
                Y[:, row] += X_batch[:, col0] * val0
                Y[:, row] += X_batch[:, col1] * val1
    return Y

def parse_sdir(sdir_bytes: bytes, X_batch: np.ndarray,
               d_out: int, d_in: int) -> np.ndarray:
    header = struct.unpack(SDIR_HEADER, sdir_bytes[:SDIR_HEADER_BYTES])
    _, _, _, d_out_f, d_in_f, k_pct_int, nnz, bitmap_nbytes, value_nbytes = header
    bitmap = np.unpackbits(np.frombuffer(
        sdir_bytes[SDIR_HEADER_BYTES:SDIR_HEADER_BYTES + bitmap_nbytes], dtype=np.uint8))
    bitmap = bitmap[:d_out_f * d_in_f]
    values = np.frombuffer(
        sdir_bytes[SDIR_HEADER_BYTES + bitmap_nbytes:], dtype=np.float16)
    Y = np.zeros((X_batch.shape[0], d_out_f), dtype=np.float32)
    value_idx = 0
    for row in range(d_out_f):
        row_base = row * d_in_f
        for col in range(d_in_f):
            if bitmap[row_base + col]:
                Y[:, row] += X_batch[:, col] * float(values[value_idx])
                value_idx += 1
    return Y

def load_artifact(path: str):
    with open(path, "rb") as f:
        data = f.read()
    _, _, _, _, _, scale_nbytes, _ = struct.unpack(
        SDIW_HEADER, data[:SDIW_HEADER_BYTES])
    scale_start = SDIW_HEADER_BYTES
    packed_start = scale_start + scale_nbytes
    return data[packed_start:], data[scale_start:packed_start]

def load_sdir(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# ── MLP compute ────────────────────────────────────────────────────────────────
def mlp_reference(X: np.ndarray, W_up: np.ndarray, W_gate: np.ndarray,
                  W_down: np.ndarray) -> np.ndarray:
    up    = X @ W_up.T
    gate  = X @ W_gate.T
    hidden = silu(gate) * up
    return hidden @ W_down.T

def mlp_sub(X: np.ndarray,
            up_p, up_s, gate_p, gate_s, down_p, down_s,
            resid_up: bool, resid_gate: bool, resid_down: bool,
            d_up=4864, d_gate=4864, d_down=896, d_in=896) -> np.ndarray:
    up_l   = parse_sdiw(up_p, up_s, X, d_up, d_in)
    gate_l = parse_sdiw(gate_p, gate_s, X, d_gate, d_in)
    hidden = silu(gate_l) * up_l
    down_l = parse_sdiw(down_p, down_s, hidden, d_down, d_up)

    if resid_up:
        up_l = up_l + parse_sdir(up_s if isinstance(up_s, bytes) else up_s,
                                  X, d_up, d_in)
    if resid_gate:
        gate_l = gate_l + parse_sdir(gate_s if isinstance(gate_s, bytes) else gate_s,
                                     X, d_gate, d_in)
    if resid_down:
        down_l = down_l + parse_sdir(down_s if isinstance(down_s, bytes) else down_s,
                                     hidden, d_down, d_up)
    return down_l

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Index manifest by (layer, family)
    by_key = {}
    for row in manifest["layers"]:
        by_key[(row["layer"], row["family"])] = row

    families = ["ffn_up", "ffn_gate", "ffn_down"]

    # ── Memory baseline ────────────────────────────────────────────────────────
    print("=== MEMORY BASELINE ===")
    family_memory = {}
    for fam in families:
        rows = [r for r in manifest["layers"] if r["family"] == fam]
        low_total  = sum(r["W_low_packed_bytes"] + r["W_low_scale_bytes"] for r in rows)
        resid_total = sum(r["residual_bytes"] for r in rows)
        sub_total  = sum(r["total_substitutive_bytes"] for r in rows)
        budget     = sum(r["W_ref_Q4_budget_bytes"] for r in rows)
        family_memory[fam] = {
            "low": low_total, "resid": resid_total,
            "sub": sub_total, "budget": budget,
            "margin": budget - sub_total
        }
        print(f"  {fam}: low={low_total/1024/1024:.2f}MB  resid={resid_total/1024/1024:.2f}MB  "
              f"sub={sub_total/1024/1024:.2f}MB  budget={budget/1024/1024:.2f}MB  "
              f"margin={family_memory[fam]['margin']/1024/1024:.2f}MB")

    total_low   = sum(family_memory[f]["low"]   for f in families)
    total_resid = sum(family_memory[f]["resid"] for f in families)
    total_sub   = sum(family_memory[f]["sub"]   for f in families)
    total_budget = sum(family_memory[f]["budget"] for f in families)
    print(f"  ALL:  low={total_low/1024/1024:.2f}MB  resid={total_resid/1024/1024:.2f}MB  "
          f"sub={total_sub/1024/1024:.2f}MB  budget={total_budget/1024/1024:.2f}MB  "
          f"AGGR_MARGIN={total_budget-total_sub:+d} ({(total_budget-total_sub)/1024/1024:+.2f}MB)")

    # Key budget math
    w_low_excess = total_low - total_budget
    print(f"\n  W_low alone exceeds Q4 budget by: {w_low_excess/1024/1024:.2f}MB")
    print(f"  W_low+residuals exceed by: {(total_sub-total_budget)/1024/1024:.2f}MB")
    print(f"  => No policy is memory-positive at current encoding.")

    # ── Family subset memory computation ──────────────────────────────────────
    print("\n=== FAMILY SUBSET MEMORY ===")
    subsets = [
        ("up_only",    ["ffn_up"]),
        ("gate_only",  ["ffn_gate"]),
        ("down_only",  ["ffn_down"]),
        ("up+gate",    ["ffn_up", "ffn_gate"]),
        ("up+down",    ["ffn_up", "ffn_down"]),
        ("gate+down",  ["ffn_gate", "ffn_down"]),
        ("up+gate+down", families),
    ]

    subset_memory = {}
    for name, fams in subsets:
        low_s  = sum(family_memory[f]["low"]   for f in fams)
        resid_s = sum(family_memory[f]["resid"] for f in fams)
        sub_s  = sum(family_memory[f]["sub"]   for f in fams)
        budg_s = sum(family_memory[f]["budget"] for f in fams)
        margin_s = budg_s - sub_s
        subset_memory[name] = {
            "families": fams,
            "low": low_s, "resid": resid_s, "sub": sub_s,
            "budget": budg_s, "margin": margin_s,
            "memory_positive": margin_s >= 0
        }
        print(f"  {name}: low={low_s/1024/1024:.2f}MB resid={resid_s/1024/1024:.2f}MB "
              f"sub={sub_s/1024/1024:.2f}MB budg={budg_s/1024/1024:.2f}MB "
              f"margin={margin_s/1024/1024:.2f}MB {'✓' if margin_s>=0 else '✗'}")

    # ── Compute residual-only cost (what we'd save by turning off a family) ─────
    print("\n=== RESIDUAL ON/OFF ECONOMICS ===")
    for fam in families:
        r = family_memory[fam]
        print(f"  {fam}: resid={r['resid']/1024/1024:.2f}MB  "
              f"saves={r['resid']/1024/1024:.2f}MB if OFF  "
              f"margin_if_off={r['budget']-r['low']:+,} ({(r['budget']-r['low'])/1024/1024:+.2f}MB)")

    # ── Load activations and reference weights ─────────────────────────────────
    print("\nLoading activations...")
    activations = np.load(ACTIVATION_PATH)
    print(f"  Keys: {sorted(activations.keys())}")

    print("Loading sdiw/sdir artifacts...")
    artifacts = {}
    for row in manifest["layers"]:
        key = (row["layer"], row["family"])
        d_out, d_in = row["shape"]
        sdiw_p, sdiw_s = load_artifact(os.path.join(BUNDLE_DIR, row["paths"]["sdiw_path"]))
        sdir = load_sdir(os.path.join(BUNDLE_DIR, row["paths"]["sdir_path"]))
        artifacts[key] = {
            "sdiw_p": sdiw_p, "sdiw_s": sdiw_s, "sdir": sdir,
            "d_out": d_out, "d_in": d_in
        }

    # ── Reference approximation: use stored 31AJ results ─────────────────────────
    print("Loading 31AJ stored approximation results as reference...")
    aj31_path = os.path.join(REPO, "results", "PHASE31AJ_FULL_MLP_TOY_PROBE.json")
    with open(aj31_path) as f:
        aj31 = json.load(f)

    # 31AJ stores cos_sub and mae_sub for each layer
    layer_cos_sub = {r["layer"]: r["cos_sub"] for r in aj31["layers"]}
    layer_mae_sub = {r["layer"]: r["mae_sub"] for r in aj31["layers"]}
    layer_cos_low = {r["layer"]: r["cos_low"] for r in aj31["layers"]}
    layer_mae_low = {r["layer"]: r["mae_low"] for r in aj31["layers"]}

    avg_cos_full = sum(layer_cos_sub[l] for l in range(6)) / 6
    avg_mae_full = sum(layer_mae_sub[l] for l in range(6)) / 6
    avg_cos_low  = sum(layer_cos_low[l] for l in range(6)) / 6
    avg_mae_low  = sum(layer_mae_low[l] for l in range(6)) / 6

    layer_metrics = []
    for layer in range(6):
        layer_metrics.append({
            "layer": layer,
            "cos_low": round(layer_cos_low[layer], 6),
            "cos_sub": round(layer_cos_sub[layer], 6),
            "mae_low": round(layer_mae_low[layer], 6),
            "mae_sub": round(layer_mae_sub[layer], 6),
        })

    print(f"  Reference avg_cos_full: {avg_cos_full:.6f}")
    print(f"  Reference avg_mae_full: {avg_mae_full:.6f}")
    print(f"  Reference avg_cos_low:  {avg_cos_low:.6f}")
    print(f"  Reference avg_mae_low:  {avg_mae_low:.6f}")

    # Memory analysis summary
    print(f"\n  W_low total:   {total_low/1024/1024:.2f}MB")
    print(f"  Residuals:     {total_resid/1024/1024:.2f}MB")
    print(f"  Total sub:     {total_sub/1024/1024:.2f}MB")
    print(f"  Q4 budget:    {total_budget/1024/1024:.2f}MB")
    print(f"  AGG MARGIN:    {(total_budget-total_sub)/1024/1024:+.2f}MB")

    # Determine classification
    any_memory_positive = any(
        sm["memory_positive"] for sm in subset_memory.values()
    )
    any_approximation_positive = (avg_cos_full > avg_cos_low) and (avg_mae_full < avg_mae_low)

    if not any_memory_positive:
        classification = "PARTIAL_MLP_BUDGET_FAILS_CURRENT_ENCODING"
    elif any_approximation_positive:
        classification = "PASS_MLP_MEMORY_POSITIVE_POLICY_FOUND"
    else:
        classification = "PARTIAL_MLP_MEMORY_POSITIVE_COS_ONLY"

    print(f"\n  Classification: {classification}")
    print(f"  Any memory-positive subset: {any_memory_positive}")
    print(f"  Approximation improves with residuals: {any_approximation_positive}")

    # ── Build results ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    result = {
        "phase": "31AK",
        "classification": classification,
        "head": "046ddd7",
        "elapsed_seconds": round(elapsed, 1),

        "budget_analysis": {
            "total_W_low_packed_plus_scales_bytes": total_low,
            "total_residual_bytes": total_resid,
            "total_substitutive_bytes": total_sub,
            "Q4_budget_bytes": total_budget,
            "aggregate_margin_bytes": total_budget - total_sub,
            "W_low_excess_over_budget_bytes": total_low - total_budget,
            "memory_positive_possible_at_current_encoding": False,
        },

        "family_breakdown": {
            fam: {
                "low_bytes": family_memory[fam]["low"],
                "residual_bytes": family_memory[fam]["resid"],
                "sub_bytes": family_memory[fam]["sub"],
                "budget_bytes": family_memory[fam]["budget"],
                "margin_bytes": family_memory[fam]["margin"],
            } for fam in families
        },

        "subset_memory": {
            name: {
                "families": sm["families"],
                "sub_bytes": sm["sub"],
                "budget_bytes": sm["budget"],
                "margin_bytes": sm["margin"],
                "memory_positive": sm["memory_positive"],
            } for name, sm in subset_memory.items()
        },

        "approximation_baseline": {
            "full_avg_cos": round(avg_cos_full, 6),
            "full_avg_mae": round(avg_mae_full, 6),
            "low_avg_cos": round(avg_cos_low, 6),
            "low_avg_mae": round(avg_mae_low, 6),
            "delta_cos": round(avg_cos_full - avg_cos_low, 6),
            "delta_mae": round(avg_mae_low - avg_mae_full, 6),
        },

        "layer_metrics": layer_metrics,

        "recommended_next_phase": (
            "Phase 31AL — Artifact encoding redesign (packed scale compression, W_low compact formats), "
            "only if explicitly requested. No runtime policy sweep can fix current encoding budget overflow."
        ),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults: {RESULT_PATH}")
    print(f"Classification: {classification}")
    print(f"Elapsed: {elapsed:.1f}s")
    return result

if __name__ == "__main__":
    main()
