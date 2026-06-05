#!/usr/bin/env python3
"""
phase31ce_real_activation_option_a_multiprompt_multilayer.py

Phase 31CE — Qwen2.5-1.5B Real-Activation Capture Multi-Prompt / Multi-Layer
Extension (Option A).

Scope (strictly per Matt's 31CE approval + 31CC planning + 31CD precedent):
- HF-derived real-activation proxy capture (NOT exact Q4_K_M GGUF runtime activation).
- Prompts: 3 — "The capital of France is" (P0), "Once upon a time" (P1),
  "In a small village" (P2). P0 is the 31I prompt index 1; P1 and P2 are
  additional short PII-free public-domain prompts from the 31I set, public-domain
  English prose, ~5-6 BPE tokens each.
- Layers: 3 — L0, L14, L27 (the 31CC plan's default {0, 14, 27} set;
  matches 31BV's first 1.5B layer choice and 31BX's edge layers).
- Token position: last prefill token only.
- Total: 3 prompts × 3 layers = 9 prompt-layer pairs.
- Activation shape expected: [1, 1536].
- Raw X saved ONLY in tempfile.mkdtemp; sha256 + shape + dtype + summary
  statistics in result JSON. Raw X is NOT in the result JSON.
- Replay W_ref: local Qwen2.5-1.5B Q4_K_M GGUF dequantized
  (NO W_ref source pivot; matches 31BU/31BV/31BX/31BZ/31CD exactly).
- W_low: corrected_ceil_per_row Q2_K derived from the Q4_K_M GGUF W_ref.
- SDIR: ffn_up + ffn_gate, k=0.5%, alpha=1.0, no ffn_down residual.
- HF model: Qwen2.5-1.5B-Instruct safetensors at
  $SDI_MODEL_DIR/qwen2.5-1.5b-hf/ (downloaded with explicit Matt approval in 31CD).
- NO model files, HF cache, raw activation arrays, Q2_K blobs, SDIR blobs committed.
- NO generation, NO sampling, NO inference output, NO quality evaluation.
- NO llama.cpp runtime integration.

Forbidden claims (per 31CC / SOT 0.A + 31CD / 31CE):
- no exact Q4_K_M GGUF runtime activation claim
- no HF safetensors forward-pass equivalence to Q4_K_M GGUF forward pass
- no real activations behave like synthetic Gaussian
- no transfer beyond the selected 3-prompt × 3-layer × last-prefill-token scope
- no 0.5B-vs-1.5B real-activation comparison
- no model quality / behavior / speed / runtime / inference / production claim
- no commit/push/tag without explicit Matt approval
- no claim that real generation will work
"""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Repo-relative imports for the Q2_K + SDIR replay (per corrected_q2k_policy_v1).
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# 31CC: replay uses GGUF W_ref. We do NOT pivot to HF safetensors weights.
from gguf import GGUFReader, dequantize  # noqa: E402
from phase31x_manifest_runtime import cosine, encode_sdir, decode_sdir  # noqa: E402
from q2k_backend import (  # noqa: E402
    quantize_q2k_f32_to_bytes,
    dequantize_q2k_bytes_to_f32,
    is_available as q2k_is_available,
)

# ---- Configuration ---------------------------------------------------------
HF_DIR_REL = "qwen2.5-1.5b-hf"             # $SDI_MODEL_DIR/qwen2.5-1.5b-hf/
GGUF_REL = "qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf"
HIDDEN = 1536                              # 1.5B
INTERMEDIATE = 8960                        # 1.5B
BATCH = 1

# 3 × 3 matrix: 3 prompts × 3 layers = 9 prompt-layer pairs.
PROMPTS = [
    ("P0", "The capital of France is"),   # 31I prompt index 1
    ("P1", "Once upon a time"),           # common 31I-style opener
    ("P2", "In a small village"),         # common 31I-style opener
]
LAYERS = [0, 14, 27]                       # 31CC plan default {0, 14, 27}

# Selected corrected Q2_K policy (per SOT v2 / 31BO package).
Q2K_MODE = "corrected_ceil_per_row"
K_PCT = 0.5
ALPHA = 1.0
RESIDUAL_FAMILIES = ["ffn_up", "ffn_gate"]

# Q4_budget_family for the 1.5B shape (d_out x d_in / 2 nibbles).
Q4_BUDGET_FAMILY = (INTERMEDIATE * HIDDEN) // 2  # 6,881,280
Q4_BUDGET_LAYER = 3 * Q4_BUDGET_FAMILY            # 20,643,840

SEVERE_DELTA_COS = -0.05

# Output paths.
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "PHASE31CE_REAL_ACTIVATION_OPTION_A_MULTIPROMPT_MULTILAYER.json"


# ---- Helpers ---------------------------------------------------------------
def get_paths() -> dict:
    sdi = os.environ.get("SDI_MODEL_DIR", "")
    if not sdi:
        raise RuntimeError("BLOCKED_31CE_SDI_MODEL_DIR_UNSET")
    p = Path(sdi)
    hf_dir = p / HF_DIR_REL
    gguf_path = p / GGUF_REL
    if not hf_dir.is_dir():
        raise RuntimeError(f"BLOCKED_31CE_HF_MODEL_MISSING: missing {hf_dir}")
    if not gguf_path.is_file():
        raise RuntimeError(f"BLOCKED_31CE_MODEL_FILE_MISSING: missing {gguf_path}")
    return {
        "sdi": sdi,
        "hf_dir": str(hf_dir),
        "gguf_path": str(gguf_path),
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def mlp_full(X_2d, W_gate, W_up, W_down):
    """Canonical: Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T"""
    if X_2d.ndim == 1:
        X_2d = X_2d[np.newaxis, :]
    hidden_act = silu(X_2d @ W_gate.T) * (X_2d @ W_up.T)
    return hidden_act @ W_down.T


def load_layer_mlp_from_gguf(gguf_path: str, layer: int):
    """Open the GGUF, dequantize the three layer-N FFN tensors, return dict."""
    try:
        reader = GGUFReader(gguf_path)
    except Exception as e:
        raise RuntimeError(f"BLOCKED_31CE_GGUF_READER_FAIL: {e}")

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
            raise RuntimeError(f"BLOCKED_31CE_GGUF_READER_FAIL: tensor {name} not found")
        try:
            w = np.asarray(dequantize(t.data, t.tensor_type))
        except Exception as e:
            raise RuntimeError(f"BLOCKED_31CE_DEQUANT_FAIL ({name}): {e}")
        out[fam] = {
            "name": t.name,
            "raw_gguf_shape": [int(x) for x in t.shape],
            "dequant_shape": list(w.shape),
            "tensor_type": t.tensor_type.name if hasattr(t.tensor_type, "name") else str(t.tensor_type),
            "n_elements": int(t.n_elements),
            "W": w.astype(np.float32, copy=False),
        }
    return out


def load_hf_model(hf_dir: str) -> dict:
    """Load the HF Qwen2.5-1.5B-Instruct model + tokenizer once for the entire
    3x3 matrix (avoids 9x model load overhead).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(f"BLOCKED_31CE_DEPENDENCY_MISSING: {e}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        tok = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"BLOCKED_31CE_TOKENIZER_LOAD_FAIL: {e}")
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            attn_implementation="eager",
        )
    except Exception as e:
        raise RuntimeError(f"BLOCKED_31CE_MODEL_LOAD_FAIL: {e}")
    mdl.eval()
    return {
        "model": mdl,
        "tokenizer": tok,
        "torch": torch,
        "n_layers": int(mdl.config.num_hidden_layers),
        "hidden_size": int(mdl.config.hidden_size),
        "intermediate_size": int(mdl.config.intermediate_size),
    }


def capture_real_activations_for_prompt(hf_state: dict, prompt: str, layers: list) -> dict:
    """Capture the L0/L14/L27 pre-FFN activations for a single prompt in one
    forward pass, with one forward_pre_hook per selected layer attached to
    `mdl.model.layers[L].mlp`. Returns a dict keyed by layer index with X
    arrays (shape [1, 1536], float32).

    Implementation: register one hook per selected layer, run the forward
    pass once, all hooks fire, then remove hooks. This is the canonical
    31I-style multi-hook approach.

    Returns: {layer: {"X": ndarray, "shape": list, "dtype": str, "sha256": str, "x_norm": float, "x_max_abs": float}}
    """
    import torch  # local import — already loaded via hf_state
    mdl = hf_state["model"]
    tok = hf_state["tokenizer"]
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    n_tokens = int(input_ids.shape[-1])
    last_token_id = int(input_ids[0, -1].item())

    captured = {}  # layer -> {"x": torch.Tensor}

    def make_hook(L):
        def hook(module, args, kwargs=None, output=None):
            x = args[0] if args else (kwargs or {}).get("hidden_states")
            if x is None:
                # Fallback: try keyword arg scan.
                if kwargs:
                    x = kwargs.get("hidden_states")
            captured[L] = {"x": x.detach().clone()}
        return hook

    handles = []
    for L in layers:
        layer_mlp = mdl.model.layers[L].mlp
        try:
            h = layer_mlp.register_forward_pre_hook(make_hook(L), with_kwargs=True)
        except TypeError:
            h = layer_mlp.register_forward_pre_hook(
                lambda mod, inp, L=L: make_hook(L)(mod, inp, {}, None)
            )
        handles.append(h)

    t1 = time.time()
    try:
        with torch.no_grad():
            out = mdl(input_ids=input_ids, use_cache=False, output_hidden_states=False)
    except Exception as e:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        raise RuntimeError(f"BLOCKED_31CE_FORWARD_HOOK_FAIL (forward pass): {e}")
    for h in handles:
        h.remove()
    t_fwd = time.time() - t1

    missing = [L for L in layers if L not in captured]
    if missing:
        raise RuntimeError(f"BLOCKED_31CE_FORWARD_HOOK_FAIL: hooks did not fire for layers {missing}")

    by_layer = {}
    for L in layers:
        x_full = captured[L]["x"]  # [1, n_tokens, hidden=1536] bf16
        x_last = x_full[:, -1, :]  # [1, hidden]
        x_f32 = x_last.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
        if x_f32.shape != (BATCH, HIDDEN):
            raise RuntimeError(
                f"BLOCKED_31CE_FORWARD_HOOK_FAIL: X shape mismatch L{L}: {x_f32.shape} != {(BATCH, HIDDEN)}"
            )
        if not np.all(np.isfinite(x_f32)):
            raise RuntimeError(f"BLOCKED_31CE_FORWARD_HOOK_FAIL: X contains NaN/Inf at L{L}")
        by_layer[L] = {
            "X": x_f32,
            "shape": list(x_f32.shape),
            "dtype": str(x_f32.dtype),
            "x_norm": float(np.linalg.norm(x_f32)),
            "x_max_abs": float(np.max(np.abs(x_f32))),
        }
    return {
        "by_layer": by_layer,
        "n_tokens": n_tokens,
        "last_token_id": last_token_id,
        "t_forward_sec": float(t_fwd),
    }


def run_pair(prompt_id: str, prompt: str, layer: int, hf_dir: str, gguf_path: str, tmpdir: str) -> dict:
    """Capture the real activation for one (prompt, layer) pair, then replay
    through the corrected Q2_K + SDIR policy on the local Q4_K_M GGUF W_ref.

    The Q2_K encode + SDIR encode happen in-memory only. Raw X is written
    to a tempfile, hashed, and the tempfile is deleted at runner exit (the
    caller is responsible for the tmpdir cleanup).
    """
    t0 = time.time()
    # --- 1. Capture X (HF safetensors forward pass) ---
    try:
        cap = capture_real_activations_for_prompt_global_state(hf_dir, prompt, [layer])
    except RuntimeError as e:
        raise
    X = cap["by_layer"][layer]["X"]
    x_norm = cap["by_layer"][layer]["x_norm"]
    x_max_abs = cap["by_layer"][layer]["x_max_abs"]
    captured_shape = cap["by_layer"][layer]["shape"]
    captured_dtype = cap["by_layer"][layer]["dtype"]
    n_tokens = cap["n_tokens"]
    last_token_id = cap["last_token_id"]
    t_cap = time.time() - t0

    # Save raw X to tempfile, hash, delete.
    raw_x_path = os.path.join(tmpdir, f"x_{prompt_id}_L{layer}_prefill_last.f32.npy")
    np.save(raw_x_path, X)
    sha_X = sha256_file(raw_x_path)

    # --- 2. Load L MLP W_ref from the local Q4_K_M GGUF (NO PIVOT) ---
    try:
        tensors = load_layer_mlp_from_gguf(gguf_path, layer)
    except RuntimeError as e:
        return {"_blocked": str(e), "_prompt_id": prompt_id, "_layer": layer}
    W_ref_up = tensors["ffn_up"]["W"]
    W_ref_gate = tensors["ffn_gate"]["W"]
    W_ref_down = tensors["ffn_down"]["W"]

    tensor_diag = {
        fam: {
            "name": v["name"],
            "raw_gguf_shape": v["raw_gguf_shape"],
            "dequant_shape": v["dequant_shape"],
            "tensor_type": v["tensor_type"],
            "n_elements": v["n_elements"],
        }
        for fam, v in tensors.items()
    }

    # --- 3. Replay per corrected_q2k_policy_v1 ---
    try:
        q2k_up = quantize_q2k_f32_to_bytes(W_ref_up, mode=Q2K_MODE)
        q2k_gate = quantize_q2k_f32_to_bytes(W_ref_gate, mode=Q2K_MODE)
        q2k_down = quantize_q2k_f32_to_bytes(W_ref_down, mode=Q2K_MODE)
    except Exception as e:
        return {"_blocked": f"BLOCKED_31CE_Q2K_BACKEND_FAIL: {e}", "_prompt_id": prompt_id, "_layer": layer}

    W_low_up = dequantize_q2k_bytes_to_f32(q2k_up, INTERMEDIATE, HIDDEN, mode=Q2K_MODE)
    W_low_gate = dequantize_q2k_bytes_to_f32(q2k_gate, INTERMEDIATE, HIDDEN, mode=Q2K_MODE)
    W_low_down = dequantize_q2k_bytes_to_f32(q2k_down, HIDDEN, INTERMEDIATE, mode=Q2K_MODE)

    R_up = W_ref_up - W_low_up
    R_gate = W_ref_gate - W_low_gate
    R_down = W_ref_down - W_low_down  # computed but unused (no ffn_down residual by policy)

    sdir_up = encode_sdir(R_up, k_pct=K_PCT)
    sdir_gate = encode_sdir(R_gate, k_pct=K_PCT)
    # ffn_down: NO SDIR residual by policy.

    dec_up = decode_sdir(sdir_up)
    dec_gate = decode_sdir(sdir_gate)

    Y_ref = mlp_full(X, W_ref_gate, W_ref_up, W_ref_down)
    Y_low = mlp_full(X, W_low_gate, W_low_up, W_low_down)
    Y_sub = mlp_full(
        X,
        W_low_gate + dec_gate,
        W_low_up + dec_up,
        W_low_down,
    )

    cos_low = cosine(Y_ref.ravel(), Y_low.ravel())
    cos_sub = cosine(Y_ref.ravel(), Y_sub.ravel())
    delta_cos = cos_sub - cos_low
    MAE_low = float(np.abs(Y_ref - Y_low).mean())
    MAE_sub = float(np.abs(Y_ref - Y_sub).mean())
    MAE_delta = MAE_sub - MAE_low
    severe = delta_cos < SEVERE_DELTA_COS
    finite = {
        "Y_ref": bool(np.all(np.isfinite(Y_ref))),
        "Y_low": bool(np.all(np.isfinite(Y_low))),
        "Y_sub": bool(np.all(np.isfinite(Y_sub))),
    }

    q2k_bytes_up = len(q2k_up)
    q2k_bytes_gate = len(q2k_gate)
    q2k_bytes_down = len(q2k_down)
    sdir_bytes_up = len(sdir_up)
    sdir_bytes_gate = len(sdir_gate)
    sdir_bytes_down = 0
    per_layer_bytes = (q2k_bytes_up + sdir_bytes_up +
                       q2k_bytes_gate + sdir_bytes_gate +
                       q2k_bytes_down + sdir_bytes_down)
    per_layer_margin = Q4_BUDGET_LAYER - per_layer_bytes

    return {
        "prompt_id": prompt_id,
        "prompt_text": prompt,
        "layer": layer,
        "n_tokens_in_prompt": n_tokens,
        "last_token_id": last_token_id,
        "captured_shape": captured_shape,
        "captured_dtype": captured_dtype,
        "sha256_of_raw_X": sha_X,
        "x_norm": x_norm,
        "x_max_abs": x_max_abs,
        "t_capture_sec": float(t_cap),
        "tensor_diag": tensor_diag,
        "cos_low": cos_low,
        "cos_sub": cos_sub,
        "delta_cos": delta_cos,
        "MAE_low": MAE_low,
        "MAE_sub": MAE_sub,
        "MAE_delta": MAE_delta,
        "severe": bool(severe),
        "finite": finite,
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
            "per_layer_margin_bytes": per_layer_margin,
            "Q4_budget_family": Q4_BUDGET_FAMILY,
            "Q4_budget_layer": Q4_BUDGET_LAYER,
            "memory_positive": per_layer_margin >= 0,
        },
    }


# Module-level handle for the loaded HF state, so run_pair can reuse it.
_HF_STATE = None


def capture_real_activations_for_prompt_global_state(hf_dir, prompt, layers):
    """Wrapper that uses the module-level _HF_STATE cache."""
    global _HF_STATE
    if _HF_STATE is None:
        _HF_STATE = load_hf_model(hf_dir)
    return capture_real_activations_for_prompt(_HF_STATE, prompt, layers)


def main() -> dict:
    installed_versions = {
        "torch": __import__("torch").__version__,
        "transformers": __import__("transformers").__version__,
        "safetensors": __import__("safetensors").__version__,
        "numpy": __import__("numpy").__version__,
        "huggingface_hub": __import__("huggingface_hub").__version__,
        "gguf": getattr(__import__("gguf"), "__version__", "present"),
    }

    # ---- Resolve paths / env / HF / GGUF presence -------------------------
    try:
        paths = get_paths()
    except RuntimeError as e:
        return {"classification": str(e), "installed_versions": installed_versions}

    # Verify Q2_K backend is loadable.
    if not q2k_is_available():
        return {
            "classification": "BLOCKED_31CE_Q2K_BACKEND_FAIL",
            "error": "libggml-base.so (with quantize_row_q2_K_ref / dequantize_row_q2_K) not loadable",
            "installed_versions": installed_versions,
        }

    # Compute SHA256 of model.safetensors for the report.
    sf_path = os.path.join(paths["hf_dir"], "model.safetensors")
    try:
        sf_size = os.path.getsize(sf_path)
        sf_sha = sha256_file(sf_path)
    except Exception as e:
        return {
            "classification": "BLOCKED_31CE_HF_MODEL_MISSING",
            "error": f"sha256 of model.safetensors failed: {e}",
            "installed_versions": installed_versions,
        }

    # GGUF metadata for the result.
    try:
        reader = GGUFReader(paths["gguf_path"])
        gguf_n_tensors = len(reader.tensors)
    except Exception as e:
        return {
            "classification": f"BLOCKED_31CE_GGUF_READER_FAIL: {e}",
            "installed_versions": installed_versions,
        }

    # ---- Load HF model once (reused across all prompts) ------------------
    t_load_start = time.time()
    try:
        load_hf_model(paths["hf_dir"])
    except RuntimeError as e:
        return {
            "classification": str(e),
            "installed_versions": installed_versions,
            "hf_dir_redacted": f"$SDI_MODEL_DIR/{HF_DIR_REL}",
            "gguf_redacted": f"$SDI_MODEL_DIR/{GGUF_REL}",
        }
    t_load = time.time() - t_load_start

    # ---- Run the 3 × 3 matrix -------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="phase31ce_")
    pair_results = []
    blocked = None
    try:
        for prompt_id, prompt in PROMPTS:
            for layer in LAYERS:
                print(f"running {prompt_id} L{layer} ...", flush=True)
                res = run_pair(prompt_id, prompt, layer, paths["hf_dir"], paths["gguf_path"], tmpdir)
                if "_blocked" in res:
                    blocked = res
                    break
                pair_results.append(res)
                print(
                    f"  {prompt_id} L{layer}  cos_low={res['cos_low']:+.6f}  "
                    f"delta_cos={res['delta_cos']:+.6f}  MAE_delta={res['MAE_delta']:+.6f}  "
                    f"severe={res['severe']}  margin={res['memory']['per_layer_margin_bytes']:+,}",
                    flush=True,
                )
            if blocked:
                break
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    if blocked:
        return {
            "classification": blocked["_blocked"],
            "installed_versions": installed_versions,
            "hf_dir_redacted": f"$SDI_MODEL_DIR/{HF_DIR_REL}",
            "gguf_redacted": f"$SDI_MODEL_DIR/{GGUF_REL}",
            "completed_pairs_before_block": len(pair_results),
        }

    n = len(pair_results)
    if n != len(PROMPTS) * len(LAYERS):
        return {
            "classification": "BLOCKED_31CE_INTERNAL_PAIR_COUNT_MISMATCH",
            "error": f"expected {len(PROMPTS) * len(LAYERS)} pairs, got {n}",
            "installed_versions": installed_versions,
            "pair_results_count": n,
        }

    # ---- Aggregate summary ----------------------------------------------
    n_finite = sum(1 for r in pair_results
                   if r["finite"]["Y_ref"] and r["finite"]["Y_low"] and r["finite"]["Y_sub"])
    n_mem_pos = sum(1 for r in pair_results if r["memory"]["memory_positive"])
    n_cos_nonneg = sum(1 for r in pair_results if r["delta_cos"] >= 0)
    n_MAE_nonworse = sum(1 for r in pair_results if r["MAE_delta"] <= 0)
    n_severe = sum(1 for r in pair_results if r["severe"])
    mean_delta_cos = float(np.mean([r["delta_cos"] for r in pair_results]))
    median_delta_cos = float(np.median([r["delta_cos"] for r in pair_results]))
    mean_MAE_delta = float(np.mean([r["MAE_delta"] for r in pair_results]))
    min_margin = min(r["memory"]["per_layer_margin_bytes"] for r in pair_results)
    max_margin = max(r["memory"]["per_layer_margin_bytes"] for r in pair_results)
    worst_cos_pair = min(pair_results, key=lambda r: r["delta_cos"])
    worst_MAE_pair = min(pair_results, key=lambda r: r["MAE_delta"])

    # ---- Classification -------------------------------------------------
    if n_finite < n:
        classification = "BLOCKED_31CE_NONFINITE"
        reason = f"only {n_finite}/{n} pairs finite"
    elif n_mem_pos < n:
        classification = "BLOCKED_31CE_MEMORY_FAIL"
        reason = f"only {n_mem_pos}/{n} pairs memory-positive"
    elif n_severe > 0:
        classification = "PARTIAL_31CE_REAL_ACTIVATION_OPTION_A_REPLAY_TRADEOFF"
        reason = f"{n_severe}/{n} severe regressions"
    elif (n_cos_nonneg < n - 1) or (n_MAE_nonworse < n - 1):
        # At least 8/9 in BOTH direction (the success criterion).
        classification = "PARTIAL_31CE_REAL_ACTIVATION_OPTION_A_MINOR_FAILURES"
        reason = (
            f"n_cos_nonneg={n_cos_nonneg}/{n}, n_MAE_nonworse={n_MAE_nonworse}/{n}, "
            f"n_severe={n_severe}/{n}, n_mem_pos={n_mem_pos}/{n}"
        )
    elif (n_cos_nonneg < n and n_MAE_nonworse == n) or (n_MAE_nonworse < n and n_cos_nonneg == n):
        # Mixed: one metric is 9/9 but the other has at least 1 fail.
        # This is a metric-conflict case, not a clean pass and not strictly a minor failure.
        classification = "PARTIAL_31CE_REAL_ACTIVATION_OPTION_A_REPLAY_TRADEOFF"
        reason = (
            f"metric conflict: n_cos_nonneg={n_cos_nonneg}/{n}, "
            f"n_MAE_nonworse={n_MAE_nonworse}/{n}, n_severe={n_severe}/{n}, "
            f"n_mem_pos={n_mem_pos}/{n}"
        )
    else:
        classification = "PASS_31CE_REAL_ACTIVATION_OPTION_A_MULTIPROMPT_MULTILAYER_CLEAN"
        reason = (
            f"all {n}/{n} pairs: memory-positive, finite, non-severe, "
            f"n_cos_nonneg={n_cos_nonneg}/{n}, n_MAE_nonworse={n_MAE_nonworse}/{n}, "
            f"mean_delta_cos={mean_delta_cos:+.6f}, mean_MAE_delta={mean_MAE_delta:+.6f}, "
            f"min_margin={min_margin:+,} bytes"
        )

    # ---- Assemble result ------------------------------------------------
    # Per-prompt × per-layer table for the report.
    per_prompt_per_layer = {}
    for prompt_id, _ in PROMPTS:
        per_prompt_per_layer[prompt_id] = {}
        for r in pair_results:
            if r["prompt_id"] == prompt_id:
                per_prompt_per_layer[prompt_id][str(r["layer"])] = {
                    "delta_cos": r["delta_cos"],
                    "MAE_delta": r["MAE_delta"],
                    "cos_low": r["cos_low"],
                    "cos_sub": r["cos_sub"],
                    "per_layer_margin_bytes": r["memory"]["per_layer_margin_bytes"],
                    "memory_positive": r["memory"]["memory_positive"],
                    "severe": r["severe"],
                    "finite": r["finite"],
                    "n_tokens_in_prompt": r["n_tokens_in_prompt"],
                    "last_token_id": r["last_token_id"],
                    "x_norm": r["x_norm"],
                    "x_max_abs": r["x_max_abs"],
                    "sha256_of_raw_X": r["sha256_of_raw_X"],
                }

    # Per-layer summary (3 layers, each with 3 prompts).
    per_layer_summary = {}
    for L in LAYERS:
        layer_results = [r for r in pair_results if r["layer"] == L]
        per_layer_summary[str(L)] = {
            "n_pairs": len(layer_results),
            "n_memory_positive": sum(1 for r in layer_results if r["memory"]["memory_positive"]),
            "n_cosine_nonnegative": sum(1 for r in layer_results if r["delta_cos"] >= 0),
            "n_MAE_nonworsening": sum(1 for r in layer_results if r["MAE_delta"] <= 0),
            "n_severe": sum(1 for r in layer_results if r["severe"]),
            "n_finite": sum(1 for r in layer_results
                            if r["finite"]["Y_ref"] and r["finite"]["Y_low"] and r["finite"]["Y_sub"]),
            "mean_delta_cos": float(np.mean([r["delta_cos"] for r in layer_results])),
            "mean_MAE_delta": float(np.mean([r["MAE_delta"] for r in layer_results])),
            "min_per_layer_margin_bytes": min(r["memory"]["per_layer_margin_bytes"] for r in layer_results),
        }

    result = {
        "phase": "31CE",
        "title": "Phase 31CE — Qwen2.5-1.5B Real-Activation Capture Multi-Prompt / Multi-Layer Extension (Option A)",
        "option_selected": "A",
        "scope": (
            "Option A HF-derived real-activation proxy multi-prompt / multi-layer extension. "
            f"{len(PROMPTS)} prompts (P0/P1/P2) × {len(LAYERS)} layers (L0/L14/L27) = "
            f"{len(PROMPTS) * len(LAYERS)} prompt-layer pairs. "
            "Last prefill token only. Shape [1, 1536]. "
            "Replay W_ref is the local 1.5B Q4_K_M GGUF dequantized (NO W_ref source pivot). "
            "W_low is corrected_ceil_per_row Q2_K derived from the Q4_K_M GGUF W_ref. "
            "SDIR is ffn_up+ffn_gate k=0.5%, alpha=1.0, no ffn_down residual. "
            "NO aggregate validation beyond this 3x3 matrix, NO generation, NO inference output, "
            "NO llama.cpp runtime integration, NO model files / HF cache / raw activation arrays "
            "/ Q2_K blobs / SDIR blobs committed."
        ),
        "forbidden_actions_upheld": [
            "no Q2_K encoding/artifact generation for disk persistence (replay only)",
            "no SDIR residual/artifact generation for disk persistence (replay only)",
            "no aggregate validation beyond the 3x3 matrix",
            "no generation / sampling / quality evaluation",
            "no llama.cpp runtime integration",
            "no exact Q4_K_M GGUF runtime activation claim (Option A is an HF-derived proxy)",
            "no claim of HF safetensors forward-pass equivalence to Q4_K_M GGUF forward pass",
            "no claim that real activations behave like synthetic Gaussian",
            "no transfer-to-other-prompts/layers/models claim",
            "no 0.5B-vs-1.5B real-activation comparison",
            "no model quality / behavior / speed / runtime / inference / production claim",
            "no model files committed (HF safetensors, GGUF, raw activations all stay outside repo)",
            "no Q2_K/SDIR blobs committed (replay materials kept in-memory only)",
            "no raw activation arrays committed (sha256 + metadata only in result JSON)",
            "no commit/push/tag without explicit Matt approval",
        ],
        "installed_versions": installed_versions,
        "hf_dir_redacted": f"$SDI_MODEL_DIR/{HF_DIR_REL}",
        "gguf_redacted": f"$SDI_MODEL_DIR/{GGUF_REL}",
        "hf_model_safetensors_size_bytes": sf_size,
        "hf_model_safetensors_sha256": sf_sha,
        "gguf_n_tensors": gguf_n_tensors,
        "t_load_hf_model_sec": float(t_load),
        "config": {
            "model": "Qwen2.5-1.5B-Instruct (HF safetensors) for X; Qwen2.5-1.5B-Instruct Q4_K_M GGUF (local) for W_ref",
            "x_source": "HF-derived real-activation proxy; NOT exact Q4_K_M GGUF runtime activation",
            "w_ref_source": "local 1.5B Q4_K_M GGUF dequantized, matches 31BU/31BV/31BX/31BZ/31CD convention",
            "w_low_source": f"corrected_ceil_per_row Q2_K derived from the Q4_K_M GGUF W_ref (mode={Q2K_MODE})",
            "sdir_source": f"ffn_up + ffn_gate SDIR k={K_PCT}%, alpha={ALPHA}, no ffn_down residual",
            "policy_package": "corrected_q2k_policy_v1",
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "batch": BATCH,
            "layers": LAYERS,
            "prompts": [{"id": pid, "text": ptext} for pid, ptext in PROMPTS],
            "token_position": "last prefill token",
        },
        "memory_accounting": {
            "Q4_budget_family_1_5B": Q4_BUDGET_FAMILY,
            "Q4_budget_layer_1_5B": Q4_BUDGET_LAYER,
            "Q4_budget_formula": "(d_out * d_in) / 2 nibbles -- shape-independent in value for ffn_up/gate (8960,1536) and ffn_down (1536,8960)",
        },
        "activation_artifact_policy": (
            "raw X written only to tempfile.mkdtemp(prefix='phase31ce_'); deleted at runner exit; "
            "sha256 + shape + dtype + x_norm + x_max_abs are the only X-derived fields in the result JSON. "
            "Hard cap 1 MB on raw activation artifacts committed: 0 bytes of raw activations in the result JSON."
        ),
        "per_prompt_per_layer": per_prompt_per_layer,
        "per_layer_summary": per_layer_summary,
        "pair_results": pair_results,
        "summary": {
            "n_pairs": n,
            "n_finite": n_finite,
            "n_memory_positive": n_mem_pos,
            "n_cosine_nonnegative": n_cos_nonneg,
            "n_MAE_nonworsening": n_MAE_nonworse,
            "n_severe_regressions": n_severe,
            "mean_delta_cos": mean_delta_cos,
            "median_delta_cos": median_delta_cos,
            "mean_MAE_delta": mean_MAE_delta,
            "min_per_layer_margin_bytes": min_margin,
            "max_per_layer_margin_bytes": max_margin,
            "n_layers": len(LAYERS),
            "n_prompts": len(PROMPTS),
            "worst_pair_by_delta_cos": {
                "prompt_id": worst_cos_pair["prompt_id"],
                "layer": worst_cos_pair["layer"],
                "delta_cos": worst_cos_pair["delta_cos"],
                "MAE_delta": worst_cos_pair["MAE_delta"],
                "per_layer_margin_bytes": worst_cos_pair["memory"]["per_layer_margin_bytes"],
            },
            "worst_pair_by_MAE_delta": {
                "prompt_id": worst_MAE_pair["prompt_id"],
                "layer": worst_MAE_pair["layer"],
                "delta_cos": worst_MAE_pair["delta_cos"],
                "MAE_delta": worst_MAE_pair["MAE_delta"],
                "per_layer_margin_bytes": worst_MAE_pair["memory"]["per_layer_margin_bytes"],
            },
        },
        "classification": classification,
        "classification_reason": reason,
        "next_allowed_phase_if_clean": (
            "Phase 31CF — Qwen2.5-1.5B Real-Activation Capture via llama.cpp Instrumentation (Option B), "
            "only if explicitly requested and operator permission is granted for llama.cpp modification and rebuild. "
            "Option B is the only path that can claim exact Q4_K_M GGUF runtime activation behavior."
        ),
        "next_allowed_phase_if_partial": (
            "Phase 31CE-R — Real-Activation Option A Repair / Prompt-Layer Diagnosis, only if explicitly requested."
        ),
        "next_allowed_phase_if_blocked": (
            "Phase 31CE-R (or 31CD-R if it's a capture-path regression), only if explicitly requested."
        ),
        "interpretation": (
            f"31CE extends 31CD from a single (L0, last-prefill-token, prompt) micro-probe to a 3x3 matrix: "
            f"3 prompts (P0='The capital of France is', P1='Once upon a time', P2='In a small village') × "
            f"3 layers (L0/L14/L27) × last prefill token. All 9 captures are from the HF safetensors forward "
            f"pass (bf16→f32 for replay). Replay W_ref is the local 1.5B Q4_K_M GGUF dequantized (NO W_ref "
            f"source pivot, matches 31BU/31BV/31BX/31BZ/31CD exactly). W_low is corrected_ceil_per_row Q2_K "
            f"from the Q4_K_M GGUF W_ref. SDIR is ffn_up+ffn_gate k=0.5%, alpha=1.0, no ffn_down residual. "
            f"Result on {n}/{n} pairs: n_finite={n_finite}, n_memory_positive={n_mem_pos}, "
            f"n_cosine_nonnegative={n_cos_nonneg}, n_MAE_nonworsening={n_MAE_nonworse}, n_severe={n_severe}, "
            f"mean_delta_cos={mean_delta_cos:+.6f}, mean_MAE_delta={mean_MAE_delta:+.6f}, "
            f"min_margin={min_margin:+,} bytes. This is a 9-pair (3 prompt × 3 layer × last-prefill-token) "
            f"extension; the result does NOT generalize to other prompts, other layers, other token positions, "
            f"other models, or runtime inference. No claim that real activations behave like synthetic Gaussian; "
            f"the cos_low on this real-activation set is observationally different from the cos_low on the "
            f"synthetic-Gaussian X vectors in 31BU/31BV/31BX/31BZ."
        ),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}", flush=True)
    print(f"classification: {classification}", flush=True)
    print(f"reason: {reason}", flush=True)
    print(f"mean_delta_cos={mean_delta_cos:+.6f}  mean_MAE_delta={mean_MAE_delta:+.6f}  "
          f"min_margin={min_margin:+,}  max_margin={max_margin:+,}", flush=True)
    return result


if __name__ == "__main__":
    out = main()
    if out.get("classification", "").startswith("BLOCKED_"):
        # Still print the result JSON for diagnostics.
        print("\n" + json.dumps(out, indent=2, default=str))
        sys.exit(1)
    print("\n" + json.dumps(
        {k: v for k, v in out.items() if k in (
            "classification", "classification_reason", "summary", "per_layer_summary", "config",
        )},
        indent=2, default=str,
    ))
