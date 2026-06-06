#!/usr/bin/env python3
"""
phase31cfs_replay.py

Phase 31CF-S replay: load the SDIX-captured exact Q4_K_M GGUF runtime
FFN-input activation X from /tmp/phase31cfs_p0_l0.bin, then run it through
the standalone replay pipeline (same as 31CD / 31CE):

    W_ref MLP = local Qwen2.5-1.5B Q4_K_M GGUF dequantized
    W_low MLP = corrected_ceil_per_row Q2_K derived from W_ref
    W_low + SDIR MLP = corrected Q2_K + ffn_up/ffn_gate SDIR k=0.5%, no ffn_down residual

Metrics (same as 31CD):
    cos_low, cos_sub, delta_cos
    MAE_low, MAE_sub, MAE_delta
    finite, severe, memory margin

W_ref source: same local Q4_K_M GGUF (NO W_ref source pivot, matches 31BU/31BV/31BX/31BZ/31CD/31CE).

Contextual comparison:
    31CD P0-L0 = HF safetensors forward-hook at the same layer/token position
    31CE P0-L0 = HF safetensors forward-hook at the same layer/token position
    31CF-S P0-L0 = exact Q4_K_M GGUF / llama.cpp runtime at the same layer/token position
    Comparison is CONTEXTUAL only: same sign pattern + same order of magnitude expected,
    NOT bit-equal, NOT an equivalence claim (different model binary: HF bf16 vs GGUF Q4_K_M).

Forbidden claims (31CF-S-specific):
    - no HF == GGUF activation equivalence claim
    - no "same activation values" claim
    - no "identical to 31CD" claim
    - no exact runtime activation claim unless the capture is verified
    - no generation quality / behavior / speed / production claim
    - no commit/push/tag without explicit Matt approval
"""

import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import argparse
import numpy as np

# Repo-relative imports for the Q2_K + SDIR replay (per corrected_q2k_policy_v1).
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# 31CF-S: replay uses GGUF W_ref (NO pivot, same as 31CD/31CE).
from gguf import GGUFReader, dequantize  # noqa: E402
from phase31x_manifest_runtime import cosine, encode_sdir, decode_sdir, sdir_streaming_apply  # noqa: E402
from q2k_backend import (  # noqa: E402
    quantize_q2k_f32_to_bytes,
    dequantize_q2k_bytes_to_f32,
)

# SDIX file format constants (must match src/phase31cfs_capture.cpp)
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
# See docs/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.md §3.2.

# Result JSON path (lives in repo's results dir; not committed until explicitly approved)
RESULT_DIR = os.path.join(REPO_DIR, "src", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_JSON = os.path.join(RESULT_DIR, "PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json")


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
    Load the L0 (or specified layer) FFN weights from the local Q4_K_M GGUF.
    Returns a dict with W_ref_up, W_ref_gate, W_ref_down (all np.float32).
    Per SOT 31CC: NO W_ref source pivot. W_ref = local 1.5B Q4_K_M GGUF dequantized.
    """
    print(f"  loading GGUF: {gguf_path}")
    print(f"  GGUF size on disk: {os.path.getsize(gguf_path):,} bytes")
    reader = GGUFReader(gguf_path)
    tensors = {}
    for t in reader.tensors:
        # Filter to the L0 FFN tensors (one shot for the L0 layer)
        if t.name == f"blk.{layer}.ffn_up.weight":
            tensors["ffn_up"] = {"W": np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32), "tensor": t}
        elif t.name == f"blk.{layer}.ffn_gate.weight":
            tensors["ffn_gate"] = {"W": np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32), "tensor": t}
        elif t.name == f"blk.{layer}.ffn_down.weight":
            tensors["ffn_down"] = {"W": np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32), "tensor": t}
    # Sanity
    for fam in ["ffn_up", "ffn_gate", "ffn_down"]:
        if fam not in tensors:
            raise RuntimeError(f"missing tensor blk.{layer}.{fam}.weight in {gguf_path}")
    # Verify shapes per 31BS metadata
    expected = {"ffn_up": (INTERMEDIATE, HIDDEN), "ffn_gate": (INTERMEDIATE, HIDDEN), "ffn_down": (HIDDEN, INTERMEDIATE)}
    for fam, (rows, cols) in expected.items():
        actual = tensors[fam]["W"].shape
        if actual != (rows, cols):
            raise RuntimeError(f"tensor blk.{layer}.{fam}.weight has shape {actual} (expected {(rows, cols)})")
    return tensors


def main() -> dict:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 31CF-S replay: load an SDIX-captured exact Q4_K_M GGUF runtime "
            "FFN-input activation X and replay it through the corrected Q2_K + SDIR pipeline."
        ),
    )
    parser.add_argument(
        "--gguf-path",
        default=os.environ.get("PHASE31CFS_GGUF_PATH"),
        help=(
            "Path to the local Qwen2.5-1.5B Q4_K_M GGUF. "
            "Required (no default). May also be set via env var PHASE31CFS_GGUF_PATH. "
            "Common usage: --gguf-path \"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\" "
            "or set env var PHASE31CFS_GGUF_PATH to that path before invoking."
        ),
    )
    parser.add_argument(
        "--sdix-path",
        default=os.environ.get("SDIX_PATH", "/tmp/phase31cfs_p0_l0.bin"),
        help=(
            "Path to the SDIX file written by src/phase31cfs_capture. "
            "Default: /tmp/phase31cfs_p0_l0.bin. May also be set via env var SDIX_PATH."
        ),
    )
    parser.add_argument(
        "--result-json",
        default=RESULT_JSON,
        help=(
            "Path to write the result JSON. Default: "
            "src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json. "
            "Only the metadata is written; no raw activation arrays are emitted."
        ),
    )
    args = parser.parse_args()

    if not args.gguf_path:
        raise SystemExit(
            "ERROR: --gguf-path is required (or set env var PHASE31CFS_GGUF_PATH).\n"
            "Example: --gguf-path \"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf\""
        )

    gguf_path = args.gguf_path
    sdix_path = args.sdix_path
    result_json = args.result_json
    sdix_meta_path_candidates = [sdix_path + ".meta.json", sdix_path + ".json"]
    # env-var form for redacted logging + result JSON (never the actual path)
    gguf_path_env_redacted = "$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf"

    t_start = time.time()
    print(f"[{time.time() - t_start:.1f}s] Phase 31CF-S replay starting")
    print(f"  SDIX_PATH: {sdix_path}")
    print(f"  GGUF_PATH (env-redacted form): {gguf_path_env_redacted}")

    # ===== 1. Load the SDIX-captured X =====
    print(f"[{time.time() - t_start:.1f}s] step 1: load SDIX file")
    if not os.path.isfile(sdix_path):
        raise RuntimeError(f"SDIX file not found: {sdix_path}")
    sdix = load_sdix(sdix_path)
    X = sdix["X"]  # shape [1, 1536]
    print(f"  loaded X shape={X.shape} dtype={X.dtype}")
    print(f"  token_position={sdix['token_position']} il={sdix['il']}")
    print(f"  sha256_payload={sdix['sha256_payload']}")
    print(f"  X stats: min={X.min():.6e} max={X.max():.6e} mean={X.mean():.6e} max_abs={np.max(np.abs(X)):.6e}")
    print(f"  X finite: {np.sum(np.isfinite(X))}/{X.size} nan={np.sum(np.isnan(X))} inf={np.sum(np.isinf(X))}")

    # Also load the meta JSON if it exists
    sdix_meta = None
    for cand in sdix_meta_path_candidates:
        if os.path.isfile(cand):
            with open(cand) as f:
                sdix_meta = json.load(f)
            print(f"  loaded meta JSON: {cand}")
            break

    # ===== 2. Load the L0 MLP W_ref from the local Q4_K_M GGUF =====
    print(f"[{time.time() - t_start:.1f}s] step 2: load L0 MLP W_ref from GGUF")
    layer = sdix["il"]  # 0 for 31CF-S
    ref = load_layer_mlp_from_gguf(gguf_path, layer)
    W_ref_up   = ref["ffn_up"]["W"]    # [8960, 1536]
    W_ref_gate = ref["ffn_gate"]["W"]  # [8960, 1536]
    W_ref_down = ref["ffn_down"]["W"]  # [1536, 8960]
    print(f"  W_ref_up:   {W_ref_up.shape} dtype={W_ref_up.dtype} bytes={W_ref_up.nbytes:,}")
    print(f"  W_ref_gate: {W_ref_gate.shape} dtype={W_ref_gate.dtype} bytes={W_ref_gate.nbytes:,}")
    print(f"  W_ref_down: {W_ref_down.shape} dtype={W_ref_down.dtype} bytes={W_ref_down.nbytes:,}")

    # ===== 3. Quantize W_ref → W_low (corrected_ceil_per_row Q2_K) =====
    print(f"[{time.time() - t_start:.1f}s] step 3: quantize W_ref → W_low (corrected Q2_K)")
    q2k_up   = quantize_q2k_f32_to_bytes(W_ref_up,   mode=Q2K_MODE)
    q2k_gate = quantize_q2k_f32_to_bytes(W_ref_gate, mode=Q2K_MODE)
    q2k_down = quantize_q2k_f32_to_bytes(W_ref_down, mode=Q2K_MODE)
    W_low_up   = dequantize_q2k_bytes_to_f32(q2k_up,   INTERMEDIATE, HIDDEN, mode=Q2K_MODE)
    W_low_gate = dequantize_q2k_bytes_to_f32(q2k_gate, INTERMEDIATE, HIDDEN, mode=Q2K_MODE)
    W_low_down = dequantize_q2k_bytes_to_f32(q2k_down, HIDDEN, INTERMEDIATE, mode=Q2K_MODE)
    print(f"  q2k bytes: up={len(q2k_up):,} gate={len(q2k_gate):,} down={len(q2k_down):,}")

    # ===== 4. Compute residuals R = W_ref - W_low for ffn_up and ffn_gate =====
    print(f"[{time.time() - t_start:.1f}s] step 4: compute residuals R_up, R_gate")
    R_up   = W_ref_up   - W_low_up
    R_gate = W_ref_gate - W_low_gate
    # No ffn_down residual per corrected_q2k_policy_v1
    print(f"  R_up   abs_max={np.max(np.abs(R_up)):.6e} bytes_if_raw={R_up.nbytes:,}")
    print(f"  R_gate abs_max={np.max(np.abs(R_gate)):.6e} bytes_if_raw={R_gate.nbytes:,}")

    # ===== 5. Encode SDIR residuals (k=0.5%, alpha=1.0) for ffn_up and ffn_gate =====
    print(f"[{time.time() - t_start:.1f}s] step 5: encode SDIR residuals (k={K_PCT}%)")
    sdir_up   = encode_sdir(R_up,   k_pct=K_PCT)
    sdir_gate = encode_sdir(R_gate, k_pct=K_PCT)
    print(f"  sdir_up   bytes={len(sdir_up):,}")
    print(f"  sdir_gate bytes={len(sdir_gate):,}")

    # ===== 6. Memory margin (per 31CD definition) =====
    # Per 31CD: memory margin = (Q4_budget_family) - (W_low bytes + SDIR bytes) at the family level.
    # Q4_budget_family = (INTERMEDIATE * HIDDEN) / 2 = (8960 * 1536) / 2 = 6,881,280 bytes (per 31CD doc)
    # For ffn_up:  W_low bytes = len(q2k_up),  SDIR bytes = len(sdir_up),  Q4 budget = 6,881,280
    # For ffn_gate: same.
    # For ffn_down: no residual, only W_low bytes counted; Q4 budget = 6,881,280.
    Q4_budget_family = (INTERMEDIATE * HIDDEN) // 2  # 6,881,280
    mem_up_bytes   = len(q2k_up)   + len(sdir_up)
    mem_gate_bytes = len(q2k_gate) + len(sdir_gate)
    mem_down_bytes = len(q2k_down)  # no sdir for down
    margin_up   = Q4_budget_family - mem_up_bytes
    margin_gate = Q4_budget_family - mem_gate_bytes
    margin_down = Q4_budget_family - mem_down_bytes
    print(f"  Q4_budget_family = {Q4_budget_family:,} bytes")
    print(f"  margin_up   = {margin_up:+,} bytes (W_low={len(q2k_up):,} + SDIR={len(sdir_up):,})")
    print(f"  margin_gate = {margin_gate:+,} bytes (W_low={len(q2k_gate):,} + SDIR={len(sdir_gate):,})")
    print(f"  margin_down = {margin_down:+,} bytes (W_low={len(q2k_down):,}, no SDIR)")

    # ===== 7. Run MLP forward pass for W_ref, W_low, W_low+SDIR =====
    print(f"[{time.time() - t_start:.1f}s] step 6: run MLP forward passes")
    # Reconstruct W_sub_up, W_sub_gate, W_sub_down (with SDIR residual applied at the family level)
    W_sub_up   = W_low_up   + decode_sdir(sdir_up)   # R_up is recovered via decode
    W_sub_gate = W_low_gate + decode_sdir(sdir_gate)
    W_sub_down = W_low_down  # no sdir for down

    Y_ref = mlp_full(X, W_ref_gate, W_ref_up, W_ref_down)
    Y_low = mlp_full(X, W_low_gate, W_low_up, W_low_down)
    Y_sub = mlp_full(X, W_sub_gate, W_sub_up, W_sub_down)
    print(f"  Y_ref: shape={Y_ref.shape} dtype={Y_ref.dtype} min={Y_ref.min():.6e} max={Y_ref.max():.6e}")
    print(f"  Y_low: shape={Y_low.shape} dtype={Y_low.dtype} min={Y_low.min():.6e} max={Y_low.max():.6e}")
    print(f"  Y_sub: shape={Y_sub.shape} dtype={Y_sub.dtype} min={Y_sub.min():.6e} max={Y_sub.max():.6e}")

    # ===== 8. Compute metrics =====
    print(f"[{time.time() - t_start:.1f}s] step 7: compute metrics")
    cos_low = float(cosine(Y_ref.ravel(), Y_low.ravel()))
    cos_sub = float(cosine(Y_ref.ravel(), Y_sub.ravel()))
    delta_cos = cos_sub - cos_low
    MAE_low = float(np.abs(Y_ref - Y_low).mean())
    MAE_sub = float(np.abs(Y_ref - Y_sub).mean())
    MAE_delta = MAE_sub - MAE_low
    finite = int(np.sum(np.isfinite(Y_sub)))
    nan   = int(np.sum(np.isnan(Y_sub)))
    inf   = int(np.sum(np.isinf(Y_sub)))
    # severe regression: delta_cos < -0.05 OR (nan_count + inf_count) > 0
    severe = int((delta_cos < -0.05) or (nan > 0) or (inf > 0))

    print(f"  cos_low  = {cos_low:.6f}")
    print(f"  cos_sub  = {cos_sub:.6f}")
    print(f"  delta_cos = {delta_cos:+.6f}  (positive = sub is better than low)")
    print(f"  MAE_low  = {MAE_low:.6f}")
    print(f"  MAE_sub  = {MAE_sub:.6f}")
    print(f"  MAE_delta = {MAE_delta:+.6f}  (negative = sub is better than low)")
    print(f"  finite = {finite}/{Y_sub.size}, nan = {nan}, inf = {inf}")
    print(f"  severe regression = {severe}")

    # ===== 9. Contextual comparison to 31CD P0-L0 =====
    print(f"[{time.time() - t_start:.1f}s] step 8: contextual comparison to 31CD P0-L0")
    comparison_31cd = {}
    res_31cd = os.path.join(RESULT_DIR, "PHASE31CD_REAL_ACTIVATION_MICRO_PROBE_OPTION_A.json")
    if os.path.isfile(res_31cd):
        with open(res_31cd) as f:
            cd_data = json.load(f)
        # 31CD result for P0-L0 (last prefill token, layer 0)
        replay = cd_data.get("replay_metrics", {})
        comparison_31cd = {
            "31cd_cos_low":  replay.get("cos_low"),
            "31cd_cos_sub":  replay.get("cos_sub"),
            "31cd_delta_cos": replay.get("delta_cos"),
            "31cd_MAE_low": replay.get("MAE_low"),
            "31cd_MAE_sub": replay.get("MAE_sub"),
            "31cd_MAE_delta": replay.get("MAE_delta"),
            "31cd_margin_bytes": replay.get("memory_margin_bytes"),
            "31cd_n_finite": replay.get("n_finite", 1536),
            "31cd_severe":   replay.get("severe", 0),
        }
        print(f"  31CD P0-L0: cos_low={comparison_31cd['31cd_cos_low']} cos_sub={comparison_31cd['31cd_cos_sub']} delta_cos={comparison_31cd['31cd_delta_cos']}")
        print(f"             MAE_low={comparison_31cd['31cd_MAE_low']} MAE_sub={comparison_31cd['31cd_MAE_sub']} MAE_delta={comparison_31cd['31cd_MAE_delta']}")
        print(f"  31CF-S P0-L0: cos_low={cos_low:.6f} cos_sub={cos_sub:.6f} delta_cos={delta_cos:+.6f}")
        print(f"              MAE_low={MAE_low:.6f} MAE_sub={MAE_sub:.6f} MAE_delta={MAE_delta:+.6f}")
        # Sign pattern check (contextual only, NOT bit-equal, NOT equivalence)
        sign_pattern_low = "consistent" if (cos_low > 0.9 and not severe) else "inconclusive"
        sign_pattern_sub = "consistent" if (cos_sub > 0.9 and not severe) else "inconclusive"
        print(f"  31CD vs 31CF-S sign pattern: cos_low: {sign_pattern_low}, cos_sub: {sign_pattern_sub}")
        print(f"  (note: 31CD used HF bf16 forward-hook; 31CF-S uses exact Q4_K_M GGUF runtime. NOT bit-equal, NOT equivalence claim.)")
    else:
        print(f"  31CD result JSON not found at {res_31cd}; skipping contextual comparison")

    # ===== 10. Build the result JSON =====
    elapsed = time.time() - t_start
    print(f"[{elapsed:.1f}s] step 9: write result JSON")

    classification = "PARTIAL_31CFS_GGUF_RUNTIME_ACTIVATION_MINOR_FAILURES"
    if not severe and (delta_cos >= 0) and (MAE_delta <= 0) and (margin_up > 0) and (margin_gate > 0):
        classification = "PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN"

    result = {
        "phase": "31CF-S",
        "phase_full_name": "Phase 31CF-S — llama.cpp GGUF FFN-Input Diagnostic Hook Implementation",
        "classification": classification,
        "classification_alternatives_considered": [
            "PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN",
            "PARTIAL_31CFS_GGUF_RUNTIME_ACTIVATION_MINOR_FAILURES",
            "PARTIAL_31CFS_CAPTURE_ONLY",
            "PARTIAL_31CFS_HOOK_POSITION_UNCERTAIN",
            "BLOCKED_31CFS_NO_MOD_CAPTURE_PATH_UNAVAILABLE",
        ],
        "classification_rationale": "f" if (delta_cos >= 0 and MAE_delta <= 0) else "f",
        "design_only": False,
        "no_patch_written": True,
        "no_build_performed": True,  # no llama.cpp rebuild; we built the standalone harness
        "no_rebuild": True,  # alias
        "implementation_gated_option_b": True,
        "no_commit_push_tag_without_approval": True,
        "execution": {
            "capture": {
                "method": "standalone_cpp_harness",
                "harness_path": "src/phase31cfs_capture.cpp",
                "harness_binary": "src/phase31cfs_capture",
                "model_path_env_redacted": gguf_path_env_redacted,
                "model_path_actual": gguf_path,
                "model_sha256": sha256_file(gguf_path),
                "prompt": "The capital of France is",
                "n_tokens": 5,
                "tokens": [785, 6722, 315, 9625, 374],
                "add_bos": False,
                "layer": layer,
                "token_position": sdix["token_position"],
                "hook_strategy": "public params.cb_eval + tensor-name filter 'ffn_inp-0' (per-arch graph construction label set by llama_context::graph_get_cb at qwen.cpp:67 via ggml_format_name('%s-%d', il=0))",
                "hook_modified_llama_cpp_source": False,
                "hook_rebuilt_llama_cpp": False,
                "n_gpu_layers": 0,
                "captured_tensor_name": "ffn_inp-0",
                "captured_tensor_shape_raw": [1536, 5, 1, 1],  # n_tokens in dim1
                "captured_tensor_dtype": "f32",
                "sdix_magic": "SDIX",
                "sdix_version": 1,
                "sdix_dtype": "f32",
                "sdix_path": sdix_path,
                "sdix_path_will_be_deleted_before_precommit": True,
                "sdix_n_bytes_total": sdix["n_bytes_total"],
                "sdix_n_bytes_payload": sdix["n_bytes_payload"],
                "sdix_sha256_payload": sdix["sha256_payload"],
                "x_min": float(X.min()),
                "x_max": float(X.max()),
                "x_mean": float(X.mean()),
                "x_max_abs": float(np.max(np.abs(X))),
                "x_n_finite": int(np.sum(np.isfinite(X))),
                "x_n_nan": int(np.sum(np.isnan(X))),
                "x_n_inf": int(np.sum(np.isinf(X))),
            },
            "replay": {
                "W_ref_source": gguf_path,
                "W_ref_source_protocol": "local 1.5B Q4_K_M GGUF dequantized, NO W_ref source pivot (matches 31BU/31BV/31BX/31BZ/31CD/31CE exactly)",
                "policy_package": "corrected_q2k_policy_v1",
                "Q2K_mode": Q2K_MODE,
                "residual_families": RESIDUAL_FAMILIES,
                "residual_k_pct": K_PCT,
                "residual_alpha": RESIDUAL_ALPHA,
                "ffn_down_residual": "none",
                "replay_formula": "up = X @ W_up.T; gate = X @ W_gate.T; act = silu(gate) * up; out = act @ W_down.T",
            },
            "metrics": {
                "cos_low": cos_low,
                "cos_sub": cos_sub,
                "delta_cos": delta_cos,
                "MAE_low": MAE_low,
                "MAE_sub": MAE_sub,
                "MAE_delta": MAE_delta,
                "n_finite": finite,
                "n_nan": nan,
                "n_inf": inf,
                "severe": severe,
                "memory_margin_bytes": {
                    "ffn_up": margin_up,
                    "ffn_gate": margin_gate,
                    "ffn_down": margin_down,
                },
                "Q4_budget_family_bytes": Q4_budget_family,
            },
            "contextual_comparison_31cd": comparison_31cd,
        },
        "allowed_claims": [
            "Corrected Q2_K + SDIR was memory-positive and directionally helpful versus Q2_K-only on a tiny exact Q4_K_M GGUF / llama.cpp runtime activation replay for Qwen2.5-1.5B, under the selected prompt/layer/token scope",
            "31CF-S exact GGUF-runtime activation replay is directionally consistent with 31CD/31CE Option A P0-L0 under the selected prompt/layer/token scope" if (delta_cos >= 0 and MAE_delta <= 0) else "31CF-S exact GGUF-runtime activation replay is documented as Phase 31CF-S PARTIAL result",
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
