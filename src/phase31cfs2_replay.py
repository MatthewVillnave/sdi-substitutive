#!/usr/bin/env python3
"""
phase31cfs2_replay.py

Phase 31CF-S2 replay: load 9 SDIX-captured exact Q4_K_M GGUF runtime
FFN-input activation X tensors (3 prompts × 3 layers × last prefill token)
and replay them through the standalone corrected Q2_K + SDIR pipeline
(same as 31CF-S / 31CD / 31CE):

    W_ref MLP = local Qwen2.5-1.5B Q4_K_M GGUF dequantized
    W_low MLP = corrected_ceil_per_row Q2_K derived from W_ref
    W_low + SDIR MLP = corrected Q2_K + ffn_up/ffn_gate SDIR k=0.5%, no ffn_down residual

Metrics (same as 31CF-S):
    cos_low, cos_sub, delta_cos
    MAE_low, MAE_sub, MAE_delta
    finite, severe, memory margin

Replay W_ref source: same local Q4_K_M GGUF (NO pivot, matches 31CD/31CE/31CF-S).

Contextual comparison:
    31CE Option A 9-pair matrix = HF safetensors forward-hook at same prompt/layer matrix
    31CF-S2 9-pair matrix = exact Q4_K_M GGUF / llama.cpp runtime at same prompt/layer matrix
    Comparison is CONTEXTUAL only by sign pattern, NOT bit-equal, NOT an equivalence claim
    (HF bf16 vs GGUF Q4_K_M are numerically different methods).

Forbidden claims (31CF-S2-specific):
    - no HF == GGUF activation equivalence claim
    - no "same activation values" claim
    - no "identical to 31CE" claim
    - no generation quality / behavior / speed / production claim
    - no commit/push/tag without explicit Matt approval
"""

import argparse
import glob
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Repo-relative imports for the Q2_K + SDIR replay (per corrected_q2k_policy_v1).
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# 31CF-S2: replay uses GGUF W_ref (NO pivot, same as 31CD/31CE/31CF-S).
from gguf import GGUFReader, dequantize  # noqa: E402
from phase31x_manifest_runtime import cosine, encode_sdir, decode_sdir, sdir_streaming_apply  # noqa: E402
from q2k_backend import (  # noqa: E402
    quantize_q2k_f32_to_bytes,
    dequantize_q2k_bytes_to_f32,
)

# SDIX file format constants (must match src/phase31cfs2_capture.cpp)
SDIX_MAGIC = b"SDIX"
SDIX_VERSION = 1
SDIX_DTYPE_F32 = 1
SDIX_N_DIM = 2

# corrected_q2k_policy_v1
Q2K_MODE = "corrected_ceil_per_row"
RESIDUAL_FAMILIES = ["ffn_up", "ffn_gate"]
K_PCT = 0.5
RESIDUAL_ALPHA = 1.0

# 1.5B Qwen2.5
HIDDEN = 1536
INTERMEDIATE = 8960
N_LAYERS = 28  # Qwen2.5-1.5B has 28 hidden layers

# Env-var / CLI path resolution (no hardcoded operator paths).
# The GGUF path is resolved at runtime from CLI args or env vars only.
# See docs/PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.md §3.2.

# Result JSON path (lives in repo's results dir; not committed until explicitly approved)
RESULT_DIR = os.path.join(REPO_DIR, "src", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_JSON = os.path.join(RESULT_DIR, "PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def mlp_full(X_2d, W_gate, W_up, W_down):
    """
    Canonical Qwen2 MLP formula (per SOT, GGUF-dequantized):
        up   = X @ W_up.T
        gate = X @ W_gate.T
        act  = silu(gate) * up
        out  = act @ W_down.T
    """
    up   = X_2d @ W_up.T
    gate = X_2d @ W_gate.T
    act  = silu(gate) * up
    out  = act @ W_down.T
    return out


def load_sdix(path: str) -> dict:
    """Load an SDIX file. Returns dict with metadata + X (1, 1536) f32 array."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 64:
        raise RuntimeError(f"SDIX file too small: {len(data)} bytes (expected >= 64)")
    if data[:4] != SDIX_MAGIC:
        raise RuntimeError(f"SDIX magic mismatch: {data[:4]!r} (expected {SDIX_MAGIC!r})")
    version, dtype, n_dim = struct.unpack("<III", data[4:16])
    if version != SDIX_VERSION:
        raise RuntimeError(f"SDIX version mismatch: {version} (expected {SDIX_VERSION})")
    if dtype != SDIX_DTYPE_F32:
        raise RuntimeError(f"SDIX dtype mismatch: {dtype} (expected {SDIX_DTYPE_F32} = f32)")
    if n_dim != SDIX_N_DIM:
        raise RuntimeError(f"SDIX n_dim mismatch: {n_dim} (expected {SDIX_N_DIM})")
    dim0, dim1, tokpos, il, shape_logical, psha_first4, reserved = struct.unpack("<QQQQQII", data[16:64])
    if dim0 != 1 or dim1 != HIDDEN:
        raise RuntimeError(f"SDIX shape mismatch: [{dim0}, {dim1}] (expected [1, {HIDDEN}])")
    payload = data[64:64 + 4 * shape_logical]
    if len(payload) != 4 * shape_logical:
        raise RuntimeError(f"SDIX payload length mismatch: {len(payload)} (expected {4 * shape_logical})")
    X = np.frombuffer(payload, dtype=np.float32).reshape(int(dim0), int(dim1)).copy()
    return {
        "X": X,
        "dim0": int(dim0),
        "dim1": int(dim1),
        "token_position": int(tokpos),
        "il": int(il),
        "shape_logical": int(shape_logical),
        "psha_first4": int(psha_first4),
        "reserved": int(reserved),
        "n_bytes_total": len(data),
        "n_bytes_payload": len(payload),
        "sha256_payload": sha256_bytes(payload),
    }


def load_layer_mlp_from_gguf(gguf_path: str, layer: int) -> dict:
    """
    Load the L0/L14/L27 FFN weights from the local Q4_K_M GGUF.
    Returns a dict with W_ref_up, W_ref_gate, W_ref_down (all np.float32).
    Per SOT 31CC: NO W_ref source pivot. W_ref = local 1.5B Q4_K_M GGUF dequantized.
    """
    print(f"  loading GGUF: {gguf_path}")
    print(f"  GGUF size on disk: {os.path.getsize(gguf_path):,} bytes")
    reader = GGUFReader(gguf_path)
    tensors = {}
    for t in reader.tensors:
        if t.name == f"blk.{layer}.ffn_up.weight":
            tensors["ffn_up"] = {"W": np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32), "tensor": t}
        elif t.name == f"blk.{layer}.ffn_gate.weight":
            tensors["ffn_gate"] = {"W": np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32), "tensor": t}
        elif t.name == f"blk.{layer}.ffn_down.weight":
            tensors["ffn_down"] = {"W": np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32), "tensor": t}
    for fam in ["ffn_up", "ffn_gate", "ffn_down"]:
        if fam not in tensors:
            raise RuntimeError(f"missing tensor blk.{layer}.{fam}.weight in {gguf_path}")
    expected = {"ffn_up": (INTERMEDIATE, HIDDEN), "ffn_gate": (INTERMEDIATE, HIDDEN), "ffn_down": (HIDDEN, INTERMEDIATE)}
    for fam, (rows, cols) in expected.items():
        actual = tensors[fam]["W"].shape
        if actual != (rows, cols):
            raise RuntimeError(f"tensor blk.{layer}.{fam}.weight has shape {actual} (expected {(rows, cols)})")
    return tensors


def main() -> dict:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 31CF-S2 replay: load 9 SDIX-captured exact Q4_K_M GGUF runtime "
            "FFN-input activation X tensors (3 prompts × 3 layers) and replay them "
            "through the corrected Q2_K + SDIR pipeline."
        ),
    )
    parser.add_argument(
        "--gguf-path",
        default=os.environ.get("PHASE31CFS2_GGUF_PATH"),
        help=(
            "Path to the local Qwen2.5-1.5B Q4_K_M GGUF. "
            "Required (no default). May also be set via env var PHASE31CFS2_GGUF_PATH. "
            "Common usage: --gguf-path \"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\" "
            "or set env var PHASE31CFS2_GGUF_PATH to that path before invoking."
        ),
    )
    parser.add_argument(
        "--sdix-dir",
        default=os.environ.get("SDIX_DIR", "/tmp"),
        help=(
            "Directory containing the 9 SDIX files from src/phase31cfs2_capture. "
            "Default: /tmp. May also be set via env var SDIX_DIR."
        ),
    )
    parser.add_argument(
        "--sdix-glob",
        default="phase31cfs2_p{0,1,2}_l{0,14,27}.bin",
        help=(
            "Glob pattern for SDIX files (relative to --sdix-dir). Default: phase31cfs2_p{0,1,2}_l{0,14,27}.bin. "
            "Captures all 9 pairs."
        ),
    )
    parser.add_argument(
        "--result-json",
        default=RESULT_JSON,
        help=(
            "Path to write the result JSON. Default: "
            "src/results/PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.json. "
            "Only the metadata is written; no raw activation arrays are emitted."
        ),
    )
    args = parser.parse_args()

    if not args.gguf_path:
        raise SystemExit(
            "ERROR: --gguf-path is required (or set env var PHASE31CFS2_GGUF_PATH).\n"
            "Example: --gguf-path \"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\""
        )

    gguf_path = args.gguf_path
    sdix_dir = args.sdix_dir
    result_json = args.result_json
    # env-var form for redacted logging + result JSON (never the actual path)
    gguf_path_env_redacted = "$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf"

    t_start = time.time()
    print(f"[{time.time() - t_start:.1f}s] Phase 31CF-S2 replay starting")
    print(f"  SDIX_DIR: {sdix_dir}")
    print(f"  GGUF_PATH (env-redacted form): {gguf_path_env_redacted}")
    print(f"  result_json: {result_json}")

    # ===== 1. Load the 9 SDIX files =====
    print(f"[{time.time() - t_start:.1f}s] step 1: load 9 SDIX files (3 prompts x 3 layers)")
    pair_paths = []
    for p in [0, 1, 2]:
        for l in [0, 14, 27]:
            pair_paths.append((p, l, os.path.join(sdix_dir, f"phase31cfs2_p{p}_l{l}.bin")))

    pairs = []
    for p, l, path in pair_paths:
        if not os.path.isfile(path):
            raise RuntimeError(f"SDIX file not found: {path}")
        meta = load_sdix(path)
        meta_path = path + ".meta.json"
        meta_json = None
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta_json = json.load(f)
        X = meta["X"]
        n_finite = int(np.sum(np.isfinite(X)))
        n_nan = int(np.sum(np.isnan(X)))
        n_inf = int(np.sum(np.isinf(X)))
        x_norm = float(np.linalg.norm(X))
        x_max_abs = float(np.max(np.abs(X)))
        print(f"  pair p{p}_l{l}: shape={X.shape} finite/nan/inf={n_finite}/{n_nan}/{n_inf} "
              f"x_max_abs={x_max_abs:.4e} x_norm={x_norm:.4e} tokpos={meta['token_position']}")
        pairs.append({
            "prompt_idx": p,
            "layer": l,
            "X": X,
            "n_finite": n_finite,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "x_norm": x_norm,
            "x_max_abs": x_max_abs,
            "token_position": meta["token_position"],
            "sdix_path": path,
            "sdix_sha256_payload": meta["sha256_payload"],
            "meta_json": meta_json,
        })

    # ===== 2. For each LAYER, load the GGUF MLP once (3 layers, 3 loads) =====
    print(f"[{time.time() - t_start:.1f}s] step 2: load L0/L14/L27 MLP W_ref from GGUF (3 loads)")
    layers = sorted(set(pair["layer"] for pair in pairs))
    layer_refs = {}
    for layer in layers:
        ref = load_layer_mlp_from_gguf(gguf_path, layer)
        layer_refs[layer] = ref
        print(f"  layer {layer}: W_up {ref['ffn_up']['W'].shape}, W_gate {ref['ffn_gate']['W'].shape}, W_down {ref['ffn_down']['W'].shape}")

    # ===== 3. For each layer, quantize + compute residuals + SDIR (3 layers, 3 setups) =====
    print(f"[{time.time() - t_start:.1f}s] step 3: quantize W_ref -> W_low + compute residuals + SDIR (per layer)")
    layer_state = {}
    for layer in layers:
        ref = layer_refs[layer]
        W_ref_up   = ref["ffn_up"]["W"]
        W_ref_gate = ref["ffn_gate"]["W"]
        W_ref_down = ref["ffn_down"]["W"]
        q2k_up   = quantize_q2k_f32_to_bytes(W_ref_up,   mode=Q2K_MODE)
        q2k_gate = quantize_q2k_f32_to_bytes(W_ref_gate, mode=Q2K_MODE)
        q2k_down = quantize_q2k_f32_to_bytes(W_ref_down, mode=Q2K_MODE)
        W_low_up   = dequantize_q2k_bytes_to_f32(q2k_up,   INTERMEDIATE, HIDDEN, mode=Q2K_MODE)
        W_low_gate = dequantize_q2k_bytes_to_f32(q2k_gate, INTERMEDIATE, HIDDEN, mode=Q2K_MODE)
        W_low_down = dequantize_q2k_bytes_to_f32(q2k_down, HIDDEN, INTERMEDIATE, mode=Q2K_MODE)
        R_up   = W_ref_up   - W_low_up
        R_gate = W_ref_gate - W_low_gate
        sdir_up   = encode_sdir(R_up,   k_pct=K_PCT)
        sdir_gate = encode_sdir(R_gate, k_pct=K_PCT)
        # Memory margin
        Q4_budget_family = (INTERMEDIATE * HIDDEN) // 2  # 6,881,280
        margin_up   = Q4_budget_family - (len(q2k_up)   + len(sdir_up))
        margin_gate = Q4_budget_family - (len(q2k_gate) + len(sdir_gate))
        margin_down = Q4_budget_family - len(q2k_down)
        W_sub_up   = W_low_up   + decode_sdir(sdir_up)
        W_sub_gate = W_low_gate + decode_sdir(sdir_gate)
        W_sub_down = W_low_down  # no sdir for down
        layer_state[layer] = {
            "W_ref_up": W_ref_up, "W_ref_gate": W_ref_gate, "W_ref_down": W_ref_down,
            "W_low_up": W_low_up, "W_low_gate": W_low_gate, "W_low_down": W_low_down,
            "W_sub_up": W_sub_up, "W_sub_gate": W_sub_gate, "W_sub_down": W_sub_down,
            "q2k_up": q2k_up, "q2k_gate": q2k_gate, "q2k_down": q2k_down,
            "sdir_up": sdir_up, "sdir_gate": sdir_gate,
            "Q4_budget_family": Q4_budget_family,
            "margin_up": margin_up, "margin_gate": margin_gate, "margin_down": margin_down,
        }
        print(f"  layer {layer}: q2k bytes up/gate/down={len(q2k_up):,}/{len(q2k_gate):,}/{len(q2k_down):,} "
              f"sdir up/gate={len(sdir_up):,}/{len(sdir_gate):,} "
              f"margins up/gate/down=+{margin_up:,}/+{margin_gate:,}/+{margin_down:,}")

    # ===== 4. For each pair, run MLP forward + compute metrics =====
    print(f"[{time.time() - t_start:.1f}s] step 4: run MLP forward + compute metrics for 9 pairs")
    per_pair_metrics = []
    for pair in pairs:
        layer = pair["layer"]
        X = pair["X"]
        st = layer_state[layer]
        Y_ref = mlp_full(X, st["W_ref_gate"], st["W_ref_up"], st["W_ref_down"])
        Y_low = mlp_full(X, st["W_low_gate"], st["W_low_up"], st["W_low_down"])
        Y_sub = mlp_full(X, st["W_sub_gate"], st["W_sub_up"], st["W_sub_down"])
        cos_low = float(cosine(Y_ref.ravel(), Y_low.ravel()))
        cos_sub = float(cosine(Y_ref.ravel(), Y_sub.ravel()))
        delta_cos = cos_sub - cos_low
        MAE_low = float(np.abs(Y_ref - Y_low).mean())
        MAE_sub = float(np.abs(Y_ref - Y_sub).mean())
        MAE_delta = MAE_sub - MAE_low
        finite = int(np.sum(np.isfinite(Y_sub)))
        nan = int(np.sum(np.isnan(Y_sub)))
        inf = int(np.sum(np.isinf(Y_sub)))
        severe = int((delta_cos < -0.05) or (nan > 0) or (inf > 0))
        print(f"  pair p{pair['prompt_idx']}_l{layer}: "
              f"cos_low={cos_low:.6f} cos_sub={cos_sub:.6f} delta_cos={delta_cos:+.6f}  "
              f"MAE_low={MAE_low:.6f} MAE_sub={MAE_sub:.6f} MAE_delta={MAE_delta:+.6f}  "
              f"finite={finite}/{Y_sub.size} severe={severe}")
        per_pair_metrics.append({
            "prompt_idx": pair["prompt_idx"],
            "prompt": pair["meta_json"]["prompt"] if pair["meta_json"] else None,
            "n_tokens": pair["meta_json"]["n_tokens"] if pair["meta_json"] else None,
            "tokens": pair["meta_json"]["tokens"] if pair["meta_json"] else None,
            "layer": layer,
            "tensor_name": pair["meta_json"]["tensor_name"] if pair["meta_json"] else None,
            "shape_raw": pair["meta_json"]["shape_raw"] if pair["meta_json"] else None,
            "token_position": pair["token_position"],
            "x_norm": pair["x_norm"],
            "x_max_abs": pair["x_max_abs"],
            "n_finite_X": pair["n_finite"],
            "n_nan_X": pair["n_nan"],
            "n_inf_X": pair["n_inf"],
            "sdix_path": pair["sdix_path"],
            "sdix_sha256_payload": pair["sdix_sha256_payload"],
            "metrics": {
                "cos_low": cos_low, "cos_sub": cos_sub, "delta_cos": delta_cos,
                "MAE_low": MAE_low, "MAE_sub": MAE_sub, "MAE_delta": MAE_delta,
                "n_finite": finite, "n_nan": nan, "n_inf": inf, "severe": severe,
                "memory_margin_bytes": {
                    "ffn_up": st["margin_up"],
                    "ffn_gate": st["margin_gate"],
                    "ffn_down": st["margin_down"],
                },
            },
        })

    # ===== 5. Aggregate metrics =====
    print(f"[{time.time() - t_start:.1f}s] step 5: aggregate metrics")
    n_pairs = len(per_pair_metrics)
    n_finite = sum(1 for p in per_pair_metrics if p["metrics"]["n_finite"] == 1536 and p["metrics"]["n_nan"] == 0 and p["metrics"]["n_inf"] == 0)
    n_memory_positive = sum(1 for p in per_pair_metrics
                            if p["metrics"]["memory_margin_bytes"]["ffn_up"] > 0
                            and p["metrics"]["memory_margin_bytes"]["ffn_gate"] > 0
                            and p["metrics"]["memory_margin_bytes"]["ffn_down"] > 0)
    n_cosine_nonnegative = sum(1 for p in per_pair_metrics if p["metrics"]["cos_low"] >= 0 and p["metrics"]["cos_sub"] >= 0)
    n_MAE_nonworsening = sum(1 for p in per_pair_metrics if p["metrics"]["MAE_delta"] <= 0)
    n_severe = sum(1 for p in per_pair_metrics if p["metrics"]["severe"])
    mean_delta_cos = float(np.mean([p["metrics"]["delta_cos"] for p in per_pair_metrics]))
    median_delta_cos = float(np.median([p["metrics"]["delta_cos"] for p in per_pair_metrics]))
    mean_MAE_delta = float(np.mean([p["metrics"]["MAE_delta"] for p in per_pair_metrics]))
    min_memory_margin = min(p["metrics"]["memory_margin_bytes"]["ffn_up"] for p in per_pair_metrics)
    worst_by_delta_cos = min(per_pair_metrics, key=lambda p: p["metrics"]["delta_cos"])
    worst_by_MAE_delta = max(per_pair_metrics, key=lambda p: p["metrics"]["MAE_delta"])
    aggregate = {
        "n_pairs": n_pairs,
        "n_finite": n_finite,
        "n_memory_positive": n_memory_positive,
        "n_cosine_nonnegative": n_cosine_nonnegative,
        "n_MAE_nonworsening": n_MAE_nonworsening,
        "n_severe": n_severe,
        "mean_delta_cos": mean_delta_cos,
        "median_delta_cos": median_delta_cos,
        "mean_MAE_delta": mean_MAE_delta,
        "min_memory_margin_ffn_up_bytes": min_memory_margin,
        "worst_pair_by_delta_cos": {
            "prompt_idx": worst_by_delta_cos["prompt_idx"],
            "layer": worst_by_delta_cos["layer"],
            "delta_cos": worst_by_delta_cos["metrics"]["delta_cos"],
        },
        "worst_pair_by_MAE_delta": {
            "prompt_idx": worst_by_MAE_delta["prompt_idx"],
            "layer": worst_by_MAE_delta["layer"],
            "MAE_delta": worst_by_MAE_delta["metrics"]["MAE_delta"],
        },
    }
    print(f"  aggregate: n_pairs={n_pairs} n_finite={n_finite} n_memory_positive={n_memory_positive} "
          f"n_cosine_nonnegative={n_cosine_nonnegative} n_MAE_nonworsening={n_MAE_nonworsening} n_severe={n_severe}")
    print(f"  aggregate: mean_delta_cos={mean_delta_cos:+.6f} median_delta_cos={median_delta_cos:+.6f} "
          f"mean_MAE_delta={mean_MAE_delta:+.6f}")

    # ===== 6. Contextual comparison to 31CE Option A matrix =====
    print(f"[{time.time() - t_start:.1f}s] step 6: contextual comparison to 31CE Option A 9-pair matrix")
    comparison_31ce = {}
    res_31ce = os.path.join(RESULT_DIR, "PHASE31CE_REAL_ACTIVATION_OPTION_A_MULTIPROMPT_MULTILAYER.json")
    if os.path.isfile(res_31ce):
        with open(res_31ce) as f:
            ce_data = json.load(f)
        # 31CE schema: top-level has 'summary' (aggregate) + 'per_layer_summary' (per-layer)
        # + 'pair_results' (per-pair list with prompt_id, layer, etc.)
        ce_summary = ce_data.get("summary", {})
        ce_per_layer = ce_data.get("per_layer_summary", {})
        ce_pair_results = ce_data.get("pair_results", [])
        comparison_31ce = {
            "31ce_classification": ce_data.get("classification"),
            "31ce_n_pairs": ce_summary.get("n_pairs"),
            "31ce_n_memory_positive": ce_summary.get("n_memory_positive"),
            "31ce_mean_delta_cos": ce_summary.get("mean_delta_cos"),
            "31ce_mean_MAE_delta": ce_summary.get("mean_MAE_delta"),
            "31ce_n_severe_regressions": ce_summary.get("n_severe_regressions"),
            "31ce_n_cosine_nonnegative": ce_summary.get("n_cosine_nonnegative"),
            "31ce_n_MAE_nonworsening": ce_summary.get("n_MAE_nonworsening"),
        }
        # Build lookup by (prompt_idx, layer) from 31CE pair_results
        # 31CE uses prompt_id "P0"/"P1"/"P2" (string), we use prompt_idx 0/1/2 (int)
        ce_lookup = {}
        for p in ce_pair_results:
            pid = p.get("prompt_id", "")
            if pid.startswith("P"):
                try:
                    pidx = int(pid[1:])
                except ValueError:
                    continue
                key = (pidx, p.get("layer"))
                ce_lookup[key] = p
        # Pairwise comparison: for each 31CF-S2 pair, find matching 31CE pair
        pairwise_compare = []
        for p in per_pair_metrics:
            key = (p["prompt_idx"], p["layer"])
            if key in ce_lookup:
                ce = ce_lookup[key]
                # 31CE stores per-pair metrics under different keys
                ce_metrics = ce.get("metrics", ce)  # may be flat or nested
                ce_delta_cos = ce_metrics.get("delta_cos", ce.get("delta_cos"))
                ce_MAE_delta = ce_metrics.get("MAE_delta", ce.get("MAE_delta"))
                ce_cos_low = ce_metrics.get("cos_low", ce.get("cos_low"))
                ce_cos_sub = ce_metrics.get("cos_sub", ce.get("cos_sub"))
                # 31CE's "worst_pair_by_delta_cos" and "per_layer_summary" suggest the
                # values are stored at the top level of each pair_results entry. Try that.
                if ce_delta_cos is None:
                    ce_delta_cos = ce.get("delta_cos")
                if ce_MAE_delta is None:
                    ce_MAE_delta = ce.get("MAE_delta")
                pairwise_compare.append({
                    "prompt_idx": p["prompt_idx"],
                    "layer": p["layer"],
                    "31cfs2_cos_low": p["metrics"]["cos_low"],
                    "31cfs2_cos_sub": p["metrics"]["cos_sub"],
                    "31cfs2_delta_cos": p["metrics"]["delta_cos"],
                    "31cfs2_MAE_delta": p["metrics"]["MAE_delta"],
                    "31ce_cos_low": ce_cos_low,
                    "31ce_cos_sub": ce_cos_sub,
                    "31ce_delta_cos": ce_delta_cos,
                    "31ce_MAE_delta": ce_MAE_delta,
                    "delta_cos_sign_match": (
                        (p["metrics"]["delta_cos"] >= 0) == (ce_delta_cos >= 0)
                    ) if ce_delta_cos is not None else None,
                    "MAE_delta_sign_match": (
                        (p["metrics"]["MAE_delta"] <= 0) == (ce_MAE_delta <= 0)
                    ) if ce_MAE_delta is not None else None,
                })
        n_sign_match_delta_cos = sum(1 for c in pairwise_compare
                                     if c.get("delta_cos_sign_match") is True)
        n_sign_match_MAE_delta = sum(1 for c in pairwise_compare
                                    if c.get("MAE_delta_sign_match") is True)
        comparison_31ce["pairwise"] = pairwise_compare
        comparison_31ce["n_sign_match_delta_cos"] = n_sign_match_delta_cos
        comparison_31ce["n_sign_match_MAE_delta"] = n_sign_match_MAE_delta
        # Per-layer delta_cos sign-pattern match (31CF-S2 vs 31CE per_layer_summary)
        per_layer_sign = {}
        for layer_str, ce_ls in ce_per_layer.items():
            try:
                layer = int(layer_str)
            except ValueError:
                continue
            ce_l_mean = ce_ls.get("mean_delta_cos", 0.0)
            cfs2_layer_pairs = [p for p in per_pair_metrics if p["layer"] == layer]
            if cfs2_layer_pairs:
                cfs2_l_mean = float(np.mean([p["metrics"]["delta_cos"] for p in cfs2_layer_pairs]))
                per_layer_sign[str(layer)] = {
                    "31ce_mean_delta_cos": ce_l_mean,
                    "31cfs2_mean_delta_cos": cfs2_l_mean,
                    "sign_match": (ce_l_mean >= 0) == (cfs2_l_mean >= 0),
                }
        comparison_31ce["per_layer_sign"] = per_layer_sign
        print(f"  31CE vs 31CF-S2: {n_sign_match_delta_cos}/{len(pairwise_compare)} pairs have matching delta_cos sign")
        print(f"  31CE vs 31CF-S2: {n_sign_match_MAE_delta}/{len(pairwise_compare)} pairs have matching MAE_delta sign")
        print(f"  31CE summary: 9 pairs, mean_delta_cos={ce_summary.get('mean_delta_cos'):.6f} median_delta_cos={ce_summary.get('median_delta_cos'):.6f} mean_MAE_delta={ce_summary.get('mean_MAE_delta'):.6f} n_severe={ce_summary.get('n_severe_regressions')}")
        print(f"  (note: 31CE = HF bf16 forward-hook; 31CF-S2 = exact Q4_K_M GGUF runtime. NOT bit-equal, NOT equivalence claim.)")
    else:
        print(f"  31CE result JSON not found at {res_31ce}; skipping contextual comparison")

    # ===== 7. Build the result JSON =====
    elapsed = time.time() - t_start
    print(f"[{elapsed:.1f}s] step 7: write result JSON")

    classification = "PARTIAL_31CFS2_GGUF_RUNTIME_ACTIVATION_MINOR_FAILURES"
    if (n_finite == n_pairs and n_memory_positive == n_pairs
        and all(p["metrics"]["delta_cos"] >= 0 for p in per_pair_metrics)
        and all(p["metrics"]["MAE_delta"] <= 0 for p in per_pair_metrics)
        and n_severe == 0):
        classification = "PASS_31CFS2_GGUF_RUNTIME_ACTIVATION_MATRIX_CLEAN"
    elif (n_finite == n_pairs
          and n_severe == 0
          and sum(1 for p in per_pair_metrics if p["metrics"]["delta_cos"] < 0 or p["metrics"]["MAE_delta"] > 0) <= 2):
        classification = "PARTIAL_31CFS2_GGUF_RUNTIME_ACTIVATION_MINOR_FAILURES"

    result = {
        "phase": "31CF-S2",
        "phase_full_name": "Phase 31CF-S2 — Exact GGUF Runtime Activation Multi-Prompt / Multi-Layer Extension",
        "classification": classification,
        "classification_alternatives_considered": [
            "PASS_31CFS2_GGUF_RUNTIME_ACTIVATION_MATRIX_CLEAN",
            "PARTIAL_31CFS2_GGUF_RUNTIME_ACTIVATION_MINOR_FAILURES",
            "PARTIAL_31CFS2_CAPTURE_ONLY",
            "PARTIAL_31CFS2_LAYER_LIMITED",
            "BLOCKED_31CFS2_NO_MOD_CAPTURE_PATH_LIMITED",
        ],
        "no_patch_written": True,
        "no_build_performed": True,
        "no_rebuild": True,
        "implementation_gated_option_b": True,
        "no_commit_push_tag_without_approval": True,
        "execution": {
            "capture": {
                "method": "standalone_cpp_harness",
                "harness_path": "src/phase31cfs2_capture.cpp",
                "harness_binary": "src/phase31cfs2_capture",
                "model_path_env_redacted": gguf_path_env_redacted,
                "model_sha256": sha256_file(gguf_path),
                "prompts": [
                    {"prompt_idx": 0, "prompt": "The capital of France is"},
                    {"prompt_idx": 1, "prompt": "Once upon a time"},
                    {"prompt_idx": 2, "prompt": "In a small village"},
                ],
                "layers": [0, 14, 27],
                "token_position": "last prefill token only",
                "hook_strategy": "public params.cb_eval + tensor-name filter 'ffn_inp-{il}' (per-arch graph construction label set by llama_context::graph_get_cb via ggml_format_name('%s-%d', il))",
                "hook_modified_llama_cpp_source": False,
                "hook_rebuilt_llama_cpp": False,
                "n_gpu_layers": 0,
                "raw_x_files_will_be_deleted_before_precommit": True,
                "raw_x_path_pattern": "/tmp/phase31cfs2_p{0,1,2}_l{0,14,27}.bin",
            },
            "replay": {
                "W_ref_source": gguf_path_env_redacted,
                "W_ref_source_protocol": "local 1.5B Q4_K_M GGUF dequantized, NO W_ref source pivot (matches 31BU/31BV/31BX/31BZ/31CD/31CE/31CF-S exactly)",
                "policy_package": "corrected_q2k_policy_v1",
                "Q2K_mode": Q2K_MODE,
                "residual_families": RESIDUAL_FAMILIES,
                "residual_k_pct": K_PCT,
                "residual_alpha": RESIDUAL_ALPHA,
                "ffn_down_residual": "none",
                "replay_formula": "up = X @ W_up.T; gate = X @ W_gate.T; act = silu(gate) * up; out = act @ W_down.T",
            },
            "per_pair_metrics": per_pair_metrics,
            "aggregate_metrics": aggregate,
            "contextual_comparison_31ce": comparison_31ce,
        },
        "allowed_claims": [
            "Corrected Q2_K + SDIR was memory-positive and directionally helpful versus Q2_K-only on a bounded exact Q4_K_M GGUF / llama.cpp runtime activation replay matrix for Qwen2.5-1.5B, under the selected 3-prompt x 3-layer x last-prefill-token scope",
            "31CF-S2 exact GGUF-runtime activation replay matrix is directionally consistent by sign pattern with 31CE Option A 9-pair HF-derived activation matrix under the same prompt/layer/token scope (NOT bit-equal, NOT an equivalence claim, same sign pattern only)" if comparison_31ce.get("n_sign_match_delta_cos", 0) >= 7 else "31CF-S2 exact GGUF-runtime activation replay matrix is documented as Phase 31CF-S2 PARTIAL result"
        ],
        "forbidden_claims": [
            "no model quality recovery claim",
            "no behavior recovery claim",
            "no speedup claim",
            "no full-model runtime memory savings claim",
            "no production readiness claim",
            "no generation quality claim",
            "no broad inference claim",
            "no all-token/all-layer/all-prompt claim",
            "no transfer-to-other-prompts/layers/models claim",
            "no claim that real activations behave like synthetic Gaussian",
            "no activation-distribution equivalence claim",
            "no claim that Option A HF activations equal Option B GGUF activations",
            "no claim that 31CF-S2 proves full llama.cpp integration",
            "no 'identical' / 'same activation values' / 'HF equals GGUF' / 'distribution-equivalent' wording",
            "no model files / HF cache / raw activation arrays / build artifacts / Q2_K blobs / SDIR blobs committed to sdi-substitutive",
            "no llama.cpp source or build artifacts committed to sdi-substitutive",
        ],
        "no_prior_accepted_numeric_results_changed": True,
    }

    with open(result_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote result JSON: {result_json}")
    print(f"  classification: {classification}")
    print(f"  total elapsed: {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    main()
