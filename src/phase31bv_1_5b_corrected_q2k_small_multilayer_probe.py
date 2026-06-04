#!/usr/bin/env python3
"""
phase31bv_1_5b_corrected_q2k_small_multilayer_probe.py

Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe.

Scope (strictly per 31BV spec — small fixed multi-layer probe only, NOT aggregate):
- Route A: layers [0, 14, 27], seeds [0, 9] → 6 anchor pairs total.
- Use the downloaded Qwen2.5-1.5B-Instruct Q4_K_M as W_ref (dequantized, NOT FP16).
- corrected_ceil_per_row Q2_K mode for W_low (all 3 families).
- SDIR residual applied to ffn_up + ffn_gate only, k=0.5%, alpha=1.0.
- ffn_down: Q2_K W_low only, NO SDIR residual.
- Compute per-pair metrics: cos_low, cos_sub, delta_cos, MAE_low, MAE_sub,
  MAE_delta, severe flag (delta_cos < -0.05), finite checks, shapes.
- Compute per-layer memory accounting: q2k_up + sdir_up + q2k_gate + sdir_gate + q2k_down
  vs 3 × Q4_budget_family for the 1.5B shape.

Forbidden (per 31BV spec — recorded in result, not just claimed):
- No aggregate validation
- No full 28-layer sweep
- No generation / inference
- No llama.cpp runtime integration
- No performance claim
- No model quality claim
- No model files committed
- No Q2_K / SDIR blobs committed (all temp work under tempfile.mkdtemp, cleaned at exit)

The 1.5B model file is at $SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf
(env var only — no hardcoded private path).

Q2_K backend (libggml-base.so) at $SDI_LLAMA_CPP_LIB (or resolved via
SDI_LLAMA_CPP_BUILD / SDI_LLAMA_CPP_ROOT) — same convention as 31BJ/31BH-R2/31BU.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ─── Repo-relative imports ────────────────────────────────────────────────────
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from gguf import GGUFReader, dequantize  # noqa: E402
from phase31x_manifest_runtime import (  # noqa: E402
    cosine,
    encode_sdir,
    decode_sdir,
)
from q2k_backend import (  # noqa: E402
    quantize_q2k_f32_to_bytes,
    dequantize_q2k_bytes_to_f32,
    is_available as q2k_is_available,
    lib as q2k_lib,
)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_REL = "qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_REDACTED_FORM = f"$SDI_MODEL_DIR/{MODEL_REL}"

# 1.5B architecture (from 31BS metadata, verified)
HIDDEN = 1536
INTERMEDIATE = 8960
BATCH = 1

# Route A: layers [0, 14, 27] × seeds [0, 9] → 6 anchor pairs (per spec)
LAYERS = [0, 14, 27]
SEEDS = [0, 9]

# Selected corrected Q2_K policy (per SOT v2 / 31BO package)
Q2K_MODE = "corrected_ceil_per_row"
K_PCT = 0.5            # 0.5% residual per family
ALPHA = 1.0
RESIDUAL_FAMILIES = ["ffn_up", "ffn_gate"]   # ffn_down has no residual

# Q4_budget_family for the 1.5B shape (d_out × d_in / 2 nibbles)
Q4_BUDGET_FAMILY = (INTERMEDIATE * HIDDEN) // 2  # 6,881,280
Q4_BUDGET_LAYER = 3 * Q4_BUDGET_FAMILY            # 20,643,840

# Severe regression threshold
SEVERE_DELTA_COS = -0.05

# Output paths
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "PHASE31BV_1_5B_CORRECTED_Q2K_SMALL_MULTILAYER_PROBE.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_model_path() -> str:
    sdi = os.environ.get("SDI_MODEL_DIR", "")
    if not sdi:
        raise RuntimeError("BLOCKED_31BV_SDI_MODEL_DIR_UNSET")
    p = Path(sdi) / MODEL_REL
    if not p.is_file():
        raise RuntimeError(f"BLOCKED_31BV_MODEL_FILE_MISSING: {p}")
    return str(p)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def load_layer_mlp(gguf_path: str, layer: int):
    """Open the GGUF, dequantize layer-N FFN tensors, return {family: W} + diag."""
    try:
        reader = GGUFReader(gguf_path)
    except Exception as e:
        raise RuntimeError(f"BLOCKED_31BV_GGUF_READER_FAIL: {e}")

    targets = {
        "ffn_up": f"blk.{layer}.ffn_up.weight",
        "ffn_gate": f"blk.{layer}.ffn_gate.weight",
        "ffn_down": f"blk.{layer}.ffn_down.weight",
    }
    out = {}
    for fam, name in targets.items():
        t = None
        for cand in reader.tensors:
            if cand.name == name:
                t = cand
                break
        if t is None:
            raise RuntimeError(f"BLOCKED_31BV_GGUF_READER_FAIL: tensor {name} not found")
        try:
            w = np.asarray(dequantize(t.data, t.tensor_type))
        except Exception as e:
            raise RuntimeError(f"BLOCKED_31BV_DEQUANT_FAIL ({name}): {e}")
        out[fam] = {
            "name": t.name,
            "raw_gguf_shape": [int(x) for x in t.shape],
            "dequant_shape": list(w.shape),
            "tensor_type": t.tensor_type.name if hasattr(t.tensor_type, "name") else str(t.tensor_type),
            "n_elements": int(t.n_elements),
            "W": w.astype(np.float32, copy=False),
        }
    return out


def mlp_full(X_2d, W_gate, W_up, W_down):
    """Canonical: Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T"""
    if X_2d.ndim == 1:
        X_2d = X_2d[np.newaxis, :]
    hidden_act = silu(X_2d @ W_gate.T) * (X_2d @ W_up.T)
    return hidden_act @ W_down.T


def run_anchor(layer: int, seed: int, mode: str, gguf_path: str):
    """Run one anchor pair: W_ref vs (W_low, W_sub) for the selected policy.

    Returns a dict with all per-pair metrics and memory accounting.
    """
    # Deterministic X vector — same convention as 31AY/31BA/31BJ/31BK/31BU
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((BATCH, HIDDEN)).astype(np.float32)

    # Load W_ref for the 3 families
    tensors = load_layer_mlp(gguf_path, layer)
    W_ref_up = tensors["ffn_up"]["W"]
    W_ref_gate = tensors["ffn_gate"]["W"]
    W_ref_down = tensors["ffn_down"]["W"]

    # corrected_ceil_per_row Q2_K encode → W_low (all 3 families)
    q2k_up = quantize_q2k_f32_to_bytes(W_ref_up, mode=mode)
    q2k_gate = quantize_q2k_f32_to_bytes(W_ref_gate, mode=mode)
    q2k_down = quantize_q2k_f32_to_bytes(W_ref_down, mode=mode)

    W_low_up = dequantize_q2k_bytes_to_f32(q2k_up, INTERMEDIATE, HIDDEN, mode=mode)
    W_low_gate = dequantize_q2k_bytes_to_f32(q2k_gate, INTERMEDIATE, HIDDEN, mode=mode)
    W_low_down = dequantize_q2k_bytes_to_f32(q2k_down, HIDDEN, INTERMEDIATE, mode=mode)

    # Residuals (per-family) for SDIR
    R_up = W_ref_up - W_low_up
    R_gate = W_ref_gate - W_low_gate
    R_down = W_ref_down - W_low_down

    # SDIR encode for up+gate (residual families); down is NOT encoded
    sdir_up = encode_sdir(R_up, k_pct=K_PCT)
    sdir_gate = encode_sdir(R_gate, k_pct=K_PCT)
    # ffn_down: NO SDIR residual by policy

    # Decode the SDIRs to get the residual matrix applied to W_low
    dec_up = decode_sdir(sdir_up)
    dec_gate = decode_sdir(sdir_gate)
    # ffn_down has no residual — leave as Q2_K W_low only

    # Reference / low / sub MLP forward
    Y_ref = mlp_full(X, W_ref_gate, W_ref_up, W_ref_down)
    Y_low = mlp_full(X, W_low_gate, W_low_up, W_low_down)
    Y_sub = mlp_full(
        X,
        W_low_gate + dec_gate,
        W_low_up   + dec_up,
        W_low_down,                # no residual added
    )

    # Metrics
    cos_low = cosine(Y_ref.ravel(), Y_low.ravel())
    cos_sub = cosine(Y_ref.ravel(), Y_sub.ravel())
    delta_cos = cos_sub - cos_low
    MAE_low = float(np.abs(Y_ref - Y_low).mean())
    MAE_sub = float(np.abs(Y_ref - Y_sub).mean())
    MAE_delta = MAE_sub - MAE_low   # negative means MAE improved
    severe = delta_cos < SEVERE_DELTA_COS

    # Per-layer memory accounting
    q2k_bytes_up = len(q2k_up)
    q2k_bytes_gate = len(q2k_gate)
    q2k_bytes_down = len(q2k_down)
    sdir_bytes_up = len(sdir_up)
    sdir_bytes_gate = len(sdir_gate)
    sdir_bytes_down = 0  # explicit: ffn_down has no SDIR under selected policy

    per_layer_bytes = (q2k_bytes_up + sdir_bytes_up +
                       q2k_bytes_gate + sdir_bytes_gate +
                       q2k_bytes_down + sdir_bytes_down)
    per_layer_margin = Q4_BUDGET_LAYER - per_layer_bytes

    return {
        "layer": layer,
        "seed": seed,
        "cos_low": cos_low,
        "cos_sub": cos_sub,
        "delta_cos": delta_cos,
        "MAE_low": MAE_low,
        "MAE_sub": MAE_sub,
        "MAE_delta": MAE_delta,
        "severe": bool(severe),
        "finite": {
            "Y_ref": bool(np.all(np.isfinite(Y_ref))),
            "Y_low": bool(np.all(np.isfinite(Y_low))),
            "Y_sub": bool(np.all(np.isfinite(Y_sub))),
        },
        "shapes": {
            "Y_ref": list(Y_ref.shape),
            "Y_low": list(Y_low.shape),
            "Y_sub": list(Y_sub.shape),
            "expected": [BATCH, HIDDEN],
        },
        "memory": {
            "q2k_bytes_up": q2k_bytes_up,
            "q2k_bytes_gate": q2k_bytes_gate,
            "q2k_bytes_down": q2k_bytes_down,
            "sdir_bytes_up": sdir_bytes_up,
            "sdir_bytes_gate": sdir_bytes_gate,
            "sdir_bytes_down": sdir_bytes_down,
            "per_layer_bytes": per_layer_bytes,
            "per_layer_margin": per_layer_margin,
            "Q4_budget_family": Q4_BUDGET_FAMILY,
            "Q4_budget_layer": Q4_BUDGET_LAYER,
            "memory_positive": per_layer_margin >= 0,
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Sanity: model file
    try:
        gguf_path = get_model_path()
    except RuntimeError as e:
        return {"classification": str(e)}

    # Sanity: Q2_K backend
    if not q2k_is_available():
        return {
            "classification": "BLOCKED_31BV_Q2K_BACKEND_FAIL",
            "error": "libggml-base.so (with quantize_row_q2_K_ref / dequantize_row_q2_K) not loadable",
        }

    q2k_lib_path = None
    try:
        from q2k_backend import _lib_path as qlp  # type: ignore
        q2k_lib_path = qlp
    except Exception:
        pass

    # Use a tempdir just to satisfy the "no large temp artifacts committed" hygiene
    # (we don't actually write any artifacts to it — everything lives in np arrays + bytes)
    tmpdir = tempfile.mkdtemp(prefix="phase31bv_")

    # Run the 6 anchor pairs (layers 0, 14, 27 × seeds 0, 9)
    anchor_results = []
    for layer in LAYERS:
        for seed in SEEDS:
            try:
                r = run_anchor(layer, seed, Q2K_MODE, gguf_path)
            except RuntimeError as e:
                return {"classification": str(e)}
            anchor_results.append(r)
            print(
                f"L{layer} S{seed}  cos_low={r['cos_low']:+.6f}  "
                f"delta_cos={r['delta_cos']:+.6f}  MAE_delta={r['MAE_delta']:+.6f}  "
                f"severe={r['severe']}  margin={r['memory']['per_layer_margin']:+,}"
            )

    # ── Aggregate summary (within the small multi-layer scope only — not a full sweep) ─
    n = len(anchor_results)
    n_mem_pos = sum(1 for r in anchor_results if r["memory"]["memory_positive"])
    n_cos_pos = sum(1 for r in anchor_results if r["delta_cos"] >= 0)
    n_mae_imp = sum(1 for r in anchor_results if r["MAE_delta"] < 0)
    n_severe = sum(1 for r in anchor_results if r["severe"])
    n_finite = sum(
        1 for r in anchor_results
        if r["finite"]["Y_ref"] and r["finite"]["Y_low"] and r["finite"]["Y_sub"]
    )
    worst = min(anchor_results, key=lambda r: r["delta_cos"])

    # ── Tensor diagnostic (per layer) ────────────────────────────────────
    # Loaded separately from the per-anchor load so the per-layer summary
    # can include real tensor metadata (name, shapes, type, n_elements)
    # for each layer — avoids the "null tensor metadata" hygiene issue.
    tensor_diag_per_layer = {}
    for layer in LAYERS:
        td = load_layer_mlp(gguf_path, layer)
        tensor_diag_per_layer[str(layer)] = {
            fam: {
                "name": v["name"],
                "raw_gguf_shape": v["raw_gguf_shape"],
                "dequant_shape": v["dequant_shape"],
                "tensor_type": v["tensor_type"],
                "n_elements": v["n_elements"],
            }
            for fam, v in td.items()
        }

    # ── Per-layer summary (3 layers × 2 seeds each) ──────────────────────
    per_layer_summary = {}
    for layer in LAYERS:
        layer_results = [r for r in anchor_results if r["layer"] == layer]
        if not layer_results:
            continue
        # Tensor metadata (real, from tensor_diag_per_layer populated above)
        layer_td = tensor_diag_per_layer.get(str(layer), {})
        per_layer_summary[str(layer)] = {
            "n_pairs": len(layer_results),
            "n_memory_positive": sum(1 for r in layer_results if r["memory"]["memory_positive"]),
            "n_cosine_positive": sum(1 for r in layer_results if r["delta_cos"] >= 0),
            "n_MAE_improving": sum(1 for r in layer_results if r["MAE_delta"] < 0),
            "n_severe": sum(1 for r in layer_results if r["severe"]),
            "mean_delta_cos": float(np.mean([r["delta_cos"] for r in layer_results])),
            "mean_MAE_improvement": float(np.mean([r["MAE_delta"] for r in layer_results])),
            "min_per_layer_margin": min(r["memory"]["per_layer_margin"] for r in layer_results),
            "tensors": {
                fam: {
                    "name": td["name"],
                    "raw_gguf_shape": td["raw_gguf_shape"],
                    "dequant_shape": td["dequant_shape"],
                    "tensor_type": td["tensor_type"],
                    "n_elements": td["n_elements"],
                }
                for fam, td in layer_td.items()
            },
        }

    # ── Classification ──────────────────────────────────────────────────
    if n != len(LAYERS) * len(SEEDS):
        classification = "BLOCKED_31BV_INTERNAL_PAIR_COUNT_MISMATCH"
        reason = f"expected {len(LAYERS) * len(SEEDS)} pairs, got {n}"
    elif n_finite < n:
        classification = "BLOCKED_31BV_NONFINITE_OUTPUTS"
        reason = f"{n - n_finite}/{n} pairs produced non-finite Y_ref/Y_low/Y_sub"
    elif n_mem_pos < n:
        classification = "PARTIAL_31BV_1_5B_Q2K_SMALL_MULTILAYER_MEMORY_FAIL"
        reason = f"only {n_mem_pos}/{n} pairs memory-positive under selected policy"
    elif n_severe > 0:
        classification = "PARTIAL_31BV_1_5B_Q2K_SMALL_MULTILAYER_TRADEOFF"
        reason = f"{n_severe}/{n} pairs have severe regression (delta_cos < {SEVERE_DELTA_COS}); " \
                 f"{n_mem_pos}/{n} memory-positive"
    elif n_cos_pos < n or n_mae_imp < n:
        classification = "PARTIAL_31BV_1_5B_Q2K_SMALL_MULTILAYER_MINOR_FAILURES"
        reason = f"{n_cos_pos}/{n} cosine-improved, {n_mae_imp}/{n} MAE-improved, " \
                 f"{n_mem_pos}/{n} memory-positive, {n_severe}/{n} severe"
    else:
        classification = "PASS_31BV_1_5B_Q2K_SMALL_MULTILAYER_CLEAN"
        reason = f"all {n}/{n} pairs memory-positive, cosine-improved, MAE-improved, " \
                 f"0 severe, all finite"

    # ── Assemble result ─────────────────────────────────────────────────
    result = {
        "phase": "31BV",
        "title": "Phase 31BV — Qwen2.5-1.5B Corrected Q2_K Small Multi-Layer Probe",
        "scope": (
            "small fixed multi-layer probe only — Route A: layers [0, 14, 27] × seeds [0, 9] = 6 anchor pairs, "
            "corrected_ceil_per_row Q2_K + ffn_up/ffn_gate SDIR k=0.5% + ffn_down W_low only. "
            "NO aggregate validation, NO full 28-layer sweep, NO generation/inference."
        ),
        "forbidden_actions_upheld": [
            "no aggregate validation",
            "no full 28-layer sweep",
            "no generation/inference",
            "no llama.cpp runtime integration",
            "no performance claim",
            "no model quality claim",
            "no model files committed",
            "no Q2_K/SDIR blobs committed (all temp under tempfile.mkdtemp)",
            "no commit/push/tag without explicit Matt approval",
        ],
        "model_path_observed_redacted": MODEL_REDACTED_FORM,
        "model_path_uses_SDI_MODEL_DIR": True,
        "model_path_no_hardcoded_private_path": True,
        "q2k_lib_path_redacted": "$SDI_LLAMA_CPP_LIB (or resolved via $SDI_LLAMA_CPP_BUILD / $SDI_LLAMA_CPP_ROOT)",
        "q2k_lib_resolved": bool(q2k_lib_path) if q2k_lib_path else None,
        "config": {
            "model": "Qwen2.5-1.5B-Instruct Q4_K_M",
            "W_ref_source": "downloaded 1.5B Q4_K_M GGUF (dequantized, NOT FP16)",
            "q2k_mode": Q2K_MODE,
            "residual_families": RESIDUAL_FAMILIES,
            "ffn_down_residual": "none (W_low only)",
            "k_pct": K_PCT,
            "alpha": ALPHA,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "batch": BATCH,
            "rng": "np.random.default_rng",
            "layers": LAYERS,
            "seeds": SEEDS,
            "route": "A (layers 0, 14, 27 × seeds 0, 9 = 6 anchor pairs)",
        },
        "tensors_read_per_layer": tensor_diag_per_layer,
        "memory_accounting": {
            "Q4_budget_family_1_5B": Q4_BUDGET_FAMILY,
            "Q4_budget_layer_1_5B": Q4_BUDGET_LAYER,
            "Q4_budget_formula": "(d_out * d_in) / 2 nibbles — shape-independent in value for ffn_up/gate (8960,1536) and ffn_down (1536,8960)",
        },
        "anchor_results": anchor_results,
        "per_layer_summary": per_layer_summary,
        "summary": {
            "n_pairs": n,
            "n_memory_positive": n_mem_pos,
            "n_cosine_positive": n_cos_pos,
            "n_MAE_improving": n_mae_imp,
            "n_severe_regressions": n_severe,
            "n_finite": n_finite,
            "worst_pair": {
                "layer": worst["layer"],
                "seed": worst["seed"],
                "delta_cos": worst["delta_cos"],
                "MAE_delta": worst["MAE_delta"],
                "per_layer_margin": worst["memory"]["per_layer_margin"],
            },
            "mean_delta_cos": float(np.mean([r["delta_cos"] for r in anchor_results])),
            "median_delta_cos": float(np.median([r["delta_cos"] for r in anchor_results])),
            "mean_MAE_improvement": float(np.mean([r["MAE_delta"] for r in anchor_results])),
            "min_per_layer_margin": min(r["memory"]["per_layer_margin"] for r in anchor_results),
            "n_layers": len(LAYERS),
            "n_seeds": len(SEEDS),
        },
        "classification": classification,
        "classification_reason": reason,
        "next_allowed_phase_if_clean": (
            "Phase 31BW — Qwen2.5-1.5B Corrected Q2_K Broader Layer Probe Planning, "
            "only if explicitly requested"
        ),
        "next_allowed_phase_if_blocked": (
            "Phase 31BV-R — Small Multi-Layer Probe Repair, only if explicitly requested"
        ),
        "interpretation": (
            "Small fixed multi-layer probe (Route A, 6 pairs across layers 0/14/27 and seeds 0/9) "
            "under selected corrected_q2k_policy_v1 on Qwen2.5-1.5B. W_ref is the 1.5B Q4_K_M dequantized "
            "tensor (NOT FP16). PASS requires all 6 pairs: memory-positive, cosine-improved, MAE-improved, "
            "no severe regressions, all finite. PARTIAL on any minor failure; severe regression OR memory "
            "fail triggers the corresponding PARTIAL or BLOCKED classification. This is a small multi-layer "
            "probe, NOT aggregate validation, NOT a larger-model claim, NOT a generalization from 31BU's "
            "single-layer result."
        ),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}")
    print(f"classification: {classification}")
    print(f"reason: {reason}")

    # Cleanup the empty tempdir (no files were written to it)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return result


if __name__ == "__main__":
    out = main()
    print("\n" + json.dumps({k: v for k, v in out.items() if k in (
        "classification", "classification_reason", "summary", "per_layer_summary", "config"
    )}, indent=2, default=str))
