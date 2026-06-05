#!/usr/bin/env python3
"""
phase31cd_real_activation_micro_probe_option_a.py

Phase 31CD — Qwen2.5-1.5B Real-Activation Capture Micro-Probe (Option A).

Scope (strictly per 31CC planning + Matt's explicit Option A approval):
- HF-derived real-activation proxy capture (NOT exact Q4_K_M GGUF runtime activation).
- Prompt: "The capital of France is" (default 31CD scope, 6 BPE tokens).
- Layer: L0 only for the first execution.
- Token position: last prefill token.
- Activation shape expected: [1, 1536].
- Raw activation arrays saved ONLY in temp storage (tempfile.mkdtemp);
  not committed; result JSON contains only sha256, shape, dtype metadata.
- Replay W_ref: local Qwen2.5-1.5B Q4_K_M GGUF dequantized
  (NO W_ref source pivot; matches 31BU/31BV/31BX/31BZ exactly).
- W_low: corrected_ceil_per_row Q2_K derived from the Q4_K_M GGUF W_ref.
- SDIR: ffn_up + ffn_gate, k=0.5%, alpha=1.0, no ffn_down residual.
- HF model: Qwen2.5-1.5B-Instruct safetensors at
  $SDI_MODEL_DIR/qwen2.5-1.5b-hf/ (downloaded with explicit Matt approval).
- NO model files, HF cache, raw activation arrays, Q2_K blobs, or SDIR blobs committed.
- NO generation, NO sampling, NO inference output, NO quality evaluation.
- NO llama.cpp runtime integration.

Forbidden claims (per 31CC / SOT 0.A + 31CD-specific 20 forbidden):
- no exact Q4_K_M GGUF runtime activation claim
- no HF safetensors forward-pass equivalence to Q4_K_M GGUF claim
- no real activations behave like synthetic Gaussian
- no transfer beyond the selected prompt/layer/token scope
- no 0.5B-vs-1.5B real-activation comparison
- no model quality / behavior / speed / runtime / inference / production claim
- no commit/push/tag without explicit Matt approval
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
PROMPT = "The capital of France is"        # default 31CD scope; 31I prompt index 1
LAYER = 0                                  # L0 only for the first execution
HIDDEN = 1536                              # 1.5B
INTERMEDIATE = 8960                        # 1.5B
BATCH = 1

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
OUT_JSON = OUT_DIR / "PHASE31CD_REAL_ACTIVATION_MICRO_PROBE_OPTION_A.json"


# ---- Helpers ---------------------------------------------------------------
def get_paths() -> dict:
    sdi = os.environ.get("SDI_MODEL_DIR", "")
    if not sdi:
        raise RuntimeError("BLOCKED_31CD_SDI_MODEL_DIR_UNSET")
    p = Path(sdi)
    hf_dir = p / HF_DIR_REL
    gguf_path = p / GGUF_REL
    if not hf_dir.is_dir():
        raise RuntimeError(f"BLOCKED_31CD_HF_DOWNLOAD_NOT_APPROVED: missing {hf_dir}")
    if not gguf_path.is_file():
        raise RuntimeError(f"BLOCKED_31CD_MODEL_FILE_MISSING: missing {gguf_path}")
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
        raise RuntimeError(f"BLOCKED_31CD_GGUF_READER_FAIL: {e}")

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
            raise RuntimeError(f"BLOCKED_31CD_GGUF_READER_FAIL: tensor {name} not found")
        try:
            w = np.asarray(dequantize(t.data, t.tensor_type))
        except Exception as e:
            raise RuntimeError(f"BLOCKED_31CD_DEQUANT_FAIL ({name}): {e}")
        out[fam] = {
            "name": t.name,
            "raw_gguf_shape": [int(x) for x in t.shape],
            "dequant_shape": list(w.shape),
            "tensor_type": t.tensor_type.name if hasattr(t.tensor_type, "name") else str(t.tensor_type),
            "n_elements": int(t.n_elements),
            "W": w.astype(np.float32, copy=False),
        }
    return out


def capture_real_activation_l0(hf_dir: str, prompt: str) -> dict:
    """Use HuggingFace transformers + a forward hook on the L0 MLP input to
    capture the pre-FFN hidden state at the last prefill token position.

    Returns a dict with shape, dtype, sha256, token_id, and the actual array X
    (in float32). The raw X is written to a tempfile and the path is
    returned so the caller can record sha256/clean-up; the X itself is NOT
    written to the result JSON.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(f"BLOCKED_31CD_TOKENIZER_OR_MODEL_LOAD_FAIL: {e}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"BLOCKED_31CD_TOKENIZER_OR_MODEL_LOAD_FAIL (tokenizer): {e}")
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            attn_implementation="eager",
        )
    except Exception as e:
        raise RuntimeError(f"BLOCKED_31CD_TOKENIZER_OR_MODEL_LOAD_FAIL (model): {e}")
    mdl.eval()
    t_load = time.time() - t0

    # Tokenize prompt (no special tokens added — preserve 31I/31CC prompt text).
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    n_tokens = int(input_ids.shape[-1])
    last_token_id = int(input_ids[0, -1].item())

    # Attach a forward hook on the L0 MLP module's input.
    # For Qwen2 in transformers, the L0 block is mdl.model.layers[0],
    # and the MLP is mdl.model.layers[0].mlp. The MLP.forward is called with
    # the post-attention residual stream hidden_states (shape [batch, seq, hidden]).
    # We capture the input to mdl.model.layers[0].mlp.forward.
    captured = {}
    def hook(module, inputs, kwargs, output=None):
        # In recent transformers, forward hook signature is (module, args, kwargs) or
        # (module, inputs, output) depending on version. Normalize.
        x = inputs[0] if inputs else kwargs.get("hidden_states")
        captured["x"] = x.detach().clone()
    layer0_mlp = mdl.model.layers[LAYER].mlp
    try:
        handle = layer0_mlp.register_forward_pre_hook(hook, with_kwargs=True)
    except TypeError:
        # Fallback for older hook signature.
        handle = layer0_mlp.register_forward_pre_hook(
            lambda mod, inp: hook(mod, inp, {}, None)
        )

    t1 = time.time()
    try:
        with torch.no_grad():
            out = mdl(input_ids=input_ids, use_cache=False, output_hidden_states=False)
    except Exception as e:
        handle.remove()
        raise RuntimeError(f"BLOCKED_31CD_FORWARD_HOOK_FAIL (forward pass): {e}")
    handle.remove()
    t_fwd = time.time() - t1

    if "x" not in captured:
        raise RuntimeError("BLOCKED_31CD_FORWARD_HOOK_FAIL: hook did not fire")

    # X shape is [batch=1, n_tokens, hidden=1536]. Take the last prefill token.
    x_full = captured["x"]  # bf16
    x_last = x_full[:, -1, :]  # [1, hidden]
    # Cast to float32 for downstream replay (no information loss; bf16 was just storage).
    x_last_f32 = x_last.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)

    if x_last_f32.shape != (BATCH, HIDDEN):
        raise RuntimeError(
            f"BLOCKED_31CD_FORWARD_HOOK_FAIL: X shape mismatch {x_last_f32.shape} != {(BATCH, HIDDEN)}"
        )
    if not np.all(np.isfinite(x_last_f32)):
        raise RuntimeError("BLOCKED_31CD_FORWARD_HOOK_FAIL: X contains NaN/Inf")

    return {
        "X": x_last_f32,
        "shape": list(x_last_f32.shape),
        "dtype": str(x_last_f32.dtype),
        "n_tokens": n_tokens,
        "last_token_id": last_token_id,
        "t_load_sec": float(t_load),
        "t_forward_sec": float(t_fwd),
        "n_layers_in_hf_model": int(mdl.config.num_hidden_layers),
        "hidden_size_hf": int(mdl.config.hidden_size),
        "intermediate_size_hf": int(mdl.config.intermediate_size),
        "torch_dtype": str(mdl.dtype),
    }


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
            "classification": "BLOCKED_31CD_Q2K_BACKEND_FAIL",
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
            "classification": "BLOCKED_31CD_HF_DOWNLOAD_NOT_APPROVED",
            "error": f"sha256 of model.safetensors failed: {e}",
            "installed_versions": installed_versions,
        }

    # GGUF metadata for the result.
    try:
        reader = GGUFReader(paths["gguf_path"])
        gguf_n_tensors = len(reader.tensors)
    except Exception as e:
        return {
            "classification": f"BLOCKED_31CD_GGUF_READER_FAIL: {e}",
            "installed_versions": installed_versions,
        }

    # ---- Capture real activation (HF safetensors forward pass, L0 MLP input) ----
    tmpdir = tempfile.mkdtemp(prefix="phase31cd_")
    try:
        try:
            cap = capture_real_activation_l0(paths["hf_dir"], PROMPT)
        except RuntimeError as e:
            return {
                "classification": str(e),
                "installed_versions": installed_versions,
                "hf_dir_redacted": f"$SDI_MODEL_DIR/{HF_DIR_REL}",
                "gguf_redacted": f"$SDI_MODEL_DIR/{GGUF_REL}",
            }
        X = cap["X"]
        captured_shape = cap["shape"]
        captured_dtype = cap["dtype"]
        # Save raw X to tempfile and compute sha256; do NOT keep on disk after.
        raw_x_path = os.path.join(tmpdir, f"x_layer{LAYER}_prefill_last.f32.npy")
        np.save(raw_x_path, X)
        sha_X = sha256_file(raw_x_path)
        x_norm = float(np.linalg.norm(X))
        x_max = float(np.max(np.abs(X)))
    finally:
        # Cleanup tempfile immediately; raw X is gone from disk.
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- Load L0 MLP W_ref from the local Q4_K_M GGUF (NO PIVOT) ----
    try:
        tensors = load_layer_mlp_from_gguf(paths["gguf_path"], LAYER)
    except RuntimeError as e:
        return {
            "classification": str(e),
            "installed_versions": installed_versions,
            "captured_shape_per_layer": {str(LAYER): captured_shape},
            "captured_dtype": captured_dtype,
        }
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

    # ---- Replay per corrected_q2k_policy_v1 ------------------------------
    try:
        q2k_up = quantize_q2k_f32_to_bytes(W_ref_up, mode=Q2K_MODE)
        q2k_gate = quantize_q2k_f32_to_bytes(W_ref_gate, mode=Q2K_MODE)
        q2k_down = quantize_q2k_f32_to_bytes(W_ref_down, mode=Q2K_MODE)
    except Exception as e:
        return {
            "classification": f"BLOCKED_31CD_Q2K_BACKEND_FAIL: {e}",
            "installed_versions": installed_versions,
            "captured_shape_per_layer": {str(LAYER): captured_shape},
            "captured_dtype": captured_dtype,
        }

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

    # ---- Classification ---------------------------------------------------
    if not (finite["Y_ref"] and finite["Y_low"] and finite["Y_sub"]):
        classification = "PARTIAL_31CD_REAL_ACTIVATION_CAPTURE_ONLY"
        reason = f"non-finite Y_ref/Y_low/Y_sub: {finite}"
    elif severe:
        classification = "PARTIAL_31CD_REAL_ACTIVATION_MICRO_REPLAY_MINOR_FAILURES"
        reason = f"severe regression: delta_cos={delta_cos:+.6f} < {SEVERE_DELTA_COS}"
    elif per_layer_margin < 0:
        classification = "PARTIAL_31CD_REAL_ACTIVATION_MICRO_REPLAY_MINOR_FAILURES"
        reason = f"per-layer memory-negative: margin={per_layer_margin:+,}"
    elif delta_cos < 0 or MAE_delta > 0:
        classification = "PARTIAL_31CD_REAL_ACTIVATION_MICRO_REPLAY_MINOR_FAILURES"
        reason = (
            f"corrected Q2_K + SDIR did not improve over Q2_K-only on this real activation: "
            f"delta_cos={delta_cos:+.6f}, MAE_delta={MAE_delta:+.6f}"
        )
    else:
        classification = "PASS_31CD_REAL_ACTIVATION_MICRO_REPLAY_CLEAN"
        reason = (
            f"corrected Q2_K + SDIR improved cosine and MAE simultaneously on the HF-derived "
            f"real-activation proxy at L0, last prefill token, prompt='{PROMPT}', "
            f"shape={captured_shape}, dtype={captured_dtype}"
        )

    # ---- Assemble result --------------------------------------------------
    result = {
        "phase": "31CD",
        "title": "Phase 31CD — Qwen2.5-1.5B Real-Activation Capture Micro-Probe (Option A)",
        "option_selected": "A",
        "scope": (
            "Option A HF-derived real-activation proxy micro-probe. L0 only, last prefill token, "
            f"prompt='{PROMPT}' ({cap['n_tokens']} BPE tokens), shape {captured_shape}, "
            "batch=1, dtype=bfloat16 (downcast to float32 for replay). Replay W_ref is the local "
            "1.5B Q4_K_M GGUF dequantized (NO W_ref source pivot). W_low is corrected_ceil_per_row "
            "Q2_K derived from the Q4_K_M GGUF W_ref. SDIR is ffn_up+ffn_gate k=0.5%, alpha=1.0, "
            "no ffn_down residual. NO aggregate validation, NO multi-layer sweep, NO multi-prompt "
            "sweep, NO generation, NO inference output, NO llama.cpp runtime integration, "
            "NO model files / HF cache / raw activation arrays / Q2_K blobs / SDIR blobs committed."
        ),
        "forbidden_actions_upheld": [
            "no Q2_K encoding/artifact generation for disk persistence (replay only)",
            "no SDIR residual/artifact generation for disk persistence (replay only)",
            "no aggregate validation",
            "no multi-layer sweep (L0 only for first execution)",
            "no multi-prompt sweep (single prompt per 31CD scope)",
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
        "config": {
            "model": "Qwen2.5-1.5B-Instruct (HF safetensors) for X; Qwen2.5-1.5B-Instruct Q4_K_M GGUF (local) for W_ref",
            "x_source": "HF-derived real-activation proxy; NOT exact Q4_K_M GGUF runtime activation",
            "w_ref_source": "local 1.5B Q4_K_M GGUF dequantized, matches 31BU/31BV/31BX/31BZ convention",
            "w_low_source": f"corrected_ceil_per_row Q2_K derived from the Q4_K_M GGUF W_ref (mode={Q2K_MODE})",
            "sdir_source": f"ffn_up + ffn_gate SDIR k={K_PCT}%, alpha={ALPHA}, no ffn_down residual",
            "policy_package": "corrected_q2k_policy_v1",
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "batch": BATCH,
            "layer": LAYER,
            "token_position": "last prefill token",
            "prompt": PROMPT,
            "n_tokens_in_prompt": cap["n_tokens"],
            "last_token_id": cap["last_token_id"],
        },
        "tensor_diag_per_layer": {str(LAYER): tensor_diag},
        "memory_accounting": {
            "Q4_budget_family_1_5B": Q4_BUDGET_FAMILY,
            "Q4_budget_layer_1_5B": Q4_BUDGET_LAYER,
            "q2k_bytes_up": q2k_bytes_up,
            "q2k_bytes_gate": q2k_bytes_gate,
            "q2k_bytes_down": q2k_bytes_down,
            "sdir_bytes_up": sdir_bytes_up,
            "sdir_bytes_gate": sdir_bytes_gate,
            "sdir_bytes_down": sdir_bytes_down,
            "per_layer_bytes": per_layer_bytes,
            "per_layer_margin_bytes": per_layer_margin,
            "memory_positive": per_layer_margin >= 0,
        },
        "activation_capture": {
            "captured_shape_per_layer": {str(LAYER): captured_shape},
            "captured_dtype": captured_dtype,
            "sha256_of_raw_X_per_layer": {str(LAYER): sha_X},
            "x_norm": x_norm,
            "x_max_abs": x_max,
            "x_storage_policy": "raw X written only to tempfile.mkdtemp; deleted at runner exit; sha256 + shape + dtype + norm + max_abs are the only X-derived fields in the result JSON",
            "capture_path": "HuggingFace transformers AutoModelForCausalLM + register_forward_pre_hook on mdl.model.layers[0].mlp; input_ids tokenized with add_special_tokens=False; X = hook_input[:, -1, :] (last prefill token), downcast bf16->float32 for replay",
            "t_load_sec": cap["t_load_sec"],
            "t_forward_sec": cap["t_forward_sec"],
        },
        "replay_metrics": {
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
        },
        "summary": {
            "n_layers": 1,
            "n_prompts": 1,
            "n_pairs": 1,
            "delta_cos": delta_cos,
            "MAE_delta": MAE_delta,
            "memory_positive": per_layer_margin >= 0,
            "per_layer_margin_bytes": per_layer_margin,
            "all_finite": bool(finite["Y_ref"] and finite["Y_low"] and finite["Y_sub"]),
        },
        "classification": classification,
        "classification_reason": reason,
        "next_allowed_phase_if_clean": (
            "31CE (Option A multi-prompt / multi-layer extension, only if explicitly requested) "
            "OR 31CF (Option B llama.cpp instrumentation, only if explicitly requested and "
            "operator permission granted) — both require explicit operator approval at entry"
        ),
        "next_allowed_phase_if_blocked": (
            "31CD-R (Real-Activation Capture Repair, only if explicitly requested)"
        ),
        "interpretation": (
            f"31CD Option A captured the real L0 pre-FFN activation at the last prefill token of "
            f"prompt '{PROMPT}' via the HF safetensors forward pass (bf16). The activation is "
            f"labeled an HF-derived real-activation proxy; it is NOT the exact Q4_K_M GGUF runtime "
            f"activation. Replay used the local 1.5B Q4_K_M GGUF dequantized as W_ref "
            f"(NO W_ref source pivot) with corrected_ceil_per_row Q2_K W_low and ffn_up+ffn_gate "
            f"SDIR k=0.5%, alpha=1.0, no ffn_down residual. "
            f"delta_cos={delta_cos:+.6f}, MAE_delta={MAE_delta:+.6f}, "
            f"per_layer_margin={per_layer_margin:+,} bytes, all finite={finite}. "
            f"This is a single-prompt / single-layer / single-token-position micro-probe; "
            f"results do NOT generalize to other prompts, other layers, other token positions, "
            f"other models, or to runtime inference. No claim that the result reproduces or "
            f"contradicts synthetic-Gaussian tensor-harness behavior."
        ),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}", flush=True)
    print(f"classification: {classification}", flush=True)
    print(f"reason: {reason}", flush=True)
    print(f"delta_cos={delta_cos:+.6f}  MAE_delta={MAE_delta:+.6f}  margin={per_layer_margin:+,}", flush=True)
    return result


if __name__ == "__main__":
    out = main()
    if out.get("classification", "").startswith("BLOCKED_"):
        # Still print the result JSON for diagnostics.
        print("\n" + json.dumps(out, indent=2, default=str))
        sys.exit(1)
    print("\n" + json.dumps(
        {k: v for k, v in out.items() if k in (
            "classification", "classification_reason", "config",
            "replay_metrics", "summary", "memory_accounting",
        )},
        indent=2, default=str,
    ))
