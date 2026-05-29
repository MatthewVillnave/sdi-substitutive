#!/usr/bin/env python3
"""
Phase 31E: Recorded Activation Probe
Tests whether 31D sparse residual encoding (bitmap + top-15% + fp16)
improves over W_low under REAL activation distributions.
"""
import os, sys, json, time, pathlib, tempfile
import numpy as np

REPO_DIR = pathlib.Path.home() / "sdi-substitutive"
RESULTS_DIR = REPO_DIR / "results"
DATA_DIR = REPO_DIR / "data"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, str(REPO_DIR / "src"))
import gguf

MODEL_PATH = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
HF_MODEL_PATH = "/home/matthew-villnave/models/hf/qwen2.5/Qwen2.5-0.5B-Instruct"
PROMPTS = ["Hi", "The capital of France is", "2+2=", "def add(a, b):", "Once upon a time"]
SEEDS = [0, 1, 2]

def cosine_batch(X, Y):
    return float(np.sum(X * Y) / (np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel()) + 1e-10))

def mae_batch(Y_ref, Y):
    return float(np.mean(np.abs(Y_ref - Y)))

def maxae_batch(Y_ref, Y):
    return float(np.max(np.abs(Y_ref - Y)))

print("=" * 60, flush=True)
print("Phase 31E: Recorded Activation Probe", flush=True)
print("=" * 60, flush=True)

# ---- Step 1: Load GGUF and extract tensors ----
print("\n[1] Loading GGUF and extracting tensors...", flush=True)
reader = gguf.GGUFReader(MODEL_PATH)
tensors = {t.name: t for t in reader.tensors}

weights = {}
for name, key in [("ffn_up", "blk.0.ffn_up.weight"), ("ffn_down", "blk.0.ffn_down.weight")]:
    t = tensors[key]
    W_ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
    d_out, d_in = W_ref.shape
    scale = float(np.std(W_ref) / 2.5)
    W_low = np.round(W_ref / scale) * scale
    R_f32 = (W_ref - W_low).astype(np.float32)
    weights[name] = {
        "W_ref": W_ref, "W_low": W_low, "R_f32": R_f32,
        "d_out": d_out, "d_in": d_in, "scale": scale,
    }
    print(f"  {name}: W_ref={W_ref.shape}, W_low={W_low.shape}", flush=True)

# ---- Step 2: Capture real activations ----
print("\n[2] Capturing real activations from Qwen2.5-0.5B...", flush=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_path = HF_MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf_path, dtype=torch.float32, trust_remote_code=True)
model = model.to("cpu").eval()

layer0 = model.model.layers[0]
print(f"  Layer 0 MLP children: {[n for n in layer0.mlp.named_children()]}", flush=True)

# Capture hooks: we need the inputs to ffn_up (gate_proj) and ffn_down (input to down_proj)
# Also capture ffn_down output for completeness
# Capture hooks:
#   ffn_up input: the hidden state entering FFN block (layer0's input) → captured by layer0 forward_pre_hook
#   ffn_down input: the SiLU(gate_proj(X)) * up_proj(X) result → captured by down_proj forward_pre_hook
#   ffn_down output: down_proj projection result → for hook on down_proj output

gate_in = [None]
down_in = [None]
down_out = [None]

pre_h0 = layer0.register_forward_pre_hook(
    lambda mod, input: gate_in.__setitem__(0, input[0].detach().clone()), prepend=True
)
pre_h1 = layer0.mlp.down_proj.register_forward_pre_hook(
    lambda mod, input: down_in.__setitem__(0, input[0].detach().clone()), prepend=True
)
post_h2 = layer0.mlp.down_proj.register_forward_hook(
    lambda mod, input, output: down_out.__setitem__(0, output[0].detach().clone())
)

print(f"  Capturing for {len(PROMPTS)} prompts...", flush=True)
t0 = time.time()

recorded_arrays = {"ffn_up": [], "ffn_down": []}
for prompt_idx, prompt in enumerate(PROMPTS):
    gate_in[0] = None
    down_out[0] = None
    
    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        model(**inputs)
    
    # gate_in: hidden_states entering layer0 = FFN input (ffn_up's X) → first token, shape (d_model,)
    if gate_in[0] is not None:
        val_up = gate_in[0][0, 0, :].numpy()
    else:
        val_up = np.zeros(weights["ffn_up"]["d_in"], dtype=np.float32)
    
    # down_in: SiLU(gate_proj(X)) * up_proj(X) result = ffn_down's input → first token, shape (d_ffn,)
    if down_in[0] is not None:
        val_down = down_in[0][0, 0, :].numpy()
    else:
        val_down = np.zeros(weights["ffn_down"]["d_in"], dtype=np.float32)
    
    recorded_arrays["ffn_up"].append(val_up)
    recorded_arrays["ffn_down"].append(val_down)
    
    print(f"    prompt[{prompt_idx}] '{prompt[:40]}': gate_in={val_up.shape}, down_out={val_down.shape}", flush=True)

pre_h0.remove()
pre_h1.remove()
post_h2.remove()

# Stack into matrices (n_prompts, d_in)
for name in ["ffn_up", "ffn_down"]:
    recorded_arrays[name] = np.stack(recorded_arrays[name], axis=0)
    print(f"  Recorded X ({name}): shape={recorded_arrays[name].shape}", flush=True)

print(f"  Activation capture done in {time.time()-t0:.1f}s", flush=True)

# ---- Step 3: Build 31D sparse encodings ----
print("\n[3] Building 31D sparse encodings (bitmap + top-15% + fp16)...", flush=True)

def build_sparse_fp15(residual, k_pct=15):
    d_out, d_in = residual.shape
    n_elements = residual.size
    nnz = max(1, int(n_elements * k_pct / 100))
    bitmap_bytes = int(np.prod(residual.shape) + 7) // 8
    
    R_abs = np.abs(residual).flatten()
    threshold = np.partition(R_abs, -nnz)[-nnz]
    mask_flat = R_abs >= threshold
    mask = mask_flat.reshape(d_out, d_in)
    
    R_sub = np.zeros_like(residual)
    R_sub[mask] = residual[mask]
    residual_values = R_sub[mask].astype(np.float16).view(np.uint16).astype(np.uint16)
    value_bytes = len(residual_values) * 2
    
    return R_sub.astype(np.float32), bitmap_bytes, value_bytes, nnz

for name, w in weights.items():
    R_sub, bitmap_bytes, value_bytes, nnz = build_sparse_fp15(w["R_f32"], k_pct=15)
    total_bytes = bitmap_bytes + value_bytes
    w["R_sub"] = R_sub
    w["nnz"] = nnz
    w["bitmap_bytes"] = bitmap_bytes
    w["value_bytes"] = value_bytes
    w["total_bytes"] = total_bytes
    print(f"  {name}: nnz={nnz:,}, bitmap={bitmap_bytes:,}, values={value_bytes:,}, total={total_bytes:,}", flush=True)

# ---- Step 4: Build control X (seeded random, same as 31D) ----
print("\n[4] Building control X (seeded random, same as 31D)...", flush=True)
control_X = {}
for name, w in weights.items():
    d_in = w["d_in"]
    X_batch_list = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((8, d_in)).astype(np.float32)
        X_batch_list.append(X)
    control_X[name] = np.concatenate(X_batch_list, axis=0)
    print(f"  Control X ({name}): shape={control_X[name].shape}", flush=True)

# ---- Step 5: Compute metrics ----
print("\n[5] Computing metrics...", flush=True)

def compute_metrics(X_batch, W_ref, W_low, R_sub, label):
    """Compute per-sample and aggregate metrics."""
    n_samples = X_batch.shape[0]
    cos_ref_low_list, cos_ref_sub_list = [], []
    mae_low_list, mae_sub_list = [], []
    maxe_low_list, maxe_sub_list = [], []
    
    for i in range(n_samples):
        X = X_batch[i : i + 1]
        Y_ref = X @ W_ref.T
        Y_low = X @ W_low.T
        Y_sub = X @ (W_low + R_sub).T
        
        cos_ref_low_list.append(cosine_batch(Y_ref, Y_low))
        cos_ref_sub_list.append(cosine_batch(Y_ref, Y_sub))
        mae_low_list.append(mae_batch(Y_ref, Y_low))
        mae_sub_list.append(mae_batch(Y_ref, Y_sub))
        maxe_low_list.append(maxae_batch(Y_ref, Y_low))
        maxe_sub_list.append(maxae_batch(Y_ref, Y_sub))
    
    mean_cos_low = float(np.mean(cos_ref_low_list))
    mean_cos_sub = float(np.mean(cos_ref_sub_list))
    mean_mae_low = float(np.mean(mae_low_list))
    mean_mae_sub = float(np.mean(mae_sub_list))
    
    result = {
        "n_samples": n_samples,
        "cosine_ref_low": round(mean_cos_low, 6),
        "cosine_ref_sub": round(mean_cos_sub, 6),
        "improvement_delta": round(mean_cos_sub - mean_cos_low, 6),
        "MAE_low": round(mean_mae_low, 8),
        "MAE_sub": round(mean_mae_sub, 8),
        "MAE_improvement": round(mean_mae_low - mean_mae_sub, 8),
        "max_error_low": round(float(np.max(maxe_low_list)), 8),
        "max_error_sub": round(float(np.max(maxe_sub_list)), 8),
        "max_error_improvement": round(float(np.max(maxe_low_list) - np.max(maxe_sub_list)), 8),
        "per_sample": [
            {
                "sample_idx": i,
                "cosine_ref_low": round(cos_ref_low_list[i], 6),
                "cosine_ref_sub": round(cos_ref_sub_list[i], 6),
                "improvement_delta": round(cos_ref_sub_list[i] - cos_ref_low_list[i], 6),
                "MAE_low": round(mae_low_list[i], 8),
                "MAE_sub": round(mae_sub_list[i], 8),
                "MAE_improvement": round(mae_low_list[i] - mae_sub_list[i], 8),
                "max_error_low": round(maxe_low_list[i], 8),
                "max_error_sub": round(maxe_sub_list[i], 8),
                "max_error_improvement": round(maxe_low_list[i] - maxe_sub_list[i], 8),
            }
            for i in range(n_samples)
        ],
    }
    return result

per_tensor_results = {}
for name, w in weights.items():
    print(f"{'':4}>>> Tensor: {name} <<<", flush=True)
    W_ref, W_low, R_sub = w["W_ref"], w["W_low"], w["R_sub"]
    
    # 5A: Control (random X)
    X_ctrl = control_X[name]
    res_ctrl = compute_metrics(X_ctrl, W_ref, W_low, R_sub, "random_control")
    
    # 5B: Recorded activations
    X_rec = recorded_arrays[name]
    res_rec = compute_metrics(X_rec, W_ref, W_low, R_sub, "recorded_activations")
    
    per_tensor_results[name] = {
        "encoding": "dense_bitmap_fp16_top15pct",
        "nnz": w["nnz"],
        "total_bytes": w["total_bytes"],
        "bitmap_bytes": w["bitmap_bytes"],
        "value_bytes": w["value_bytes"],
        "random_control": res_ctrl,
        "recorded_activations": res_rec,
    }
    
    print(f"{'':8}Control   delta: {res_ctrl['improvement_delta']:+.6f}  (cos_ref_sub={res_ctrl['cosine_ref_sub']:.6f} vs cos_ref_low={res_ctrl['cosine_ref_low']:.6f})")
    print(f"{'':8}Recorded  delta: {res_rec['improvement_delta']:+.6f}  (cos_ref_sub={res_rec['cosine_ref_sub']:.6f} vs cos_ref_low={res_rec['cosine_ref_low']:.6f})")

# ---- Step 6: Per-prompt breakdown ----
print("\n[6] Per-prompt breakdown...", flush=True)

prompt_breakdown = {}
for prompt_idx, prompt in enumerate(PROMPTS):
    key = f"prompt_{prompt_idx}"
    prompt_breakdown[key] = {"prompt": prompt, "tensors": {}}
    
    for name, w in weights.items():
        W_ref, W_low, R_sub = w["W_ref"], w["W_low"], w["R_sub"]
        X_single = recorded_arrays[name][prompt_idx : prompt_idx + 1]
        Y_ref = X_single @ W_ref.T
        Y_low = X_single @ W_low.T
        Y_sub = X_single @ (W_low + R_sub).T
        
        cos_ref_low = cosine_batch(Y_ref, Y_low)
        cos_ref_sub = cosine_batch(Y_ref, Y_sub)
        mae_low_v = mae_batch(Y_ref, Y_low)
        mae_sub_v = mae_batch(Y_ref, Y_sub)
        maxe_low_v = maxae_batch(Y_ref, Y_low)
        maxe_sub_v = maxae_batch(Y_ref, Y_sub)
        
        prompt_breakdown[key]["tensors"][name] = {
            "cosine_ref_low": round(cos_ref_low, 6),
            "cosine_ref_sub": round(cos_ref_sub, 6),
            "improvement_delta": round(cos_ref_sub - cos_ref_low, 6),
            "MAE_low": round(mae_low_v, 8),
            "MAE_sub": round(mae_sub_v, 8),
            "MAE_improvement": round(mae_low_v - mae_sub_v, 8),
            "max_error_low": round(maxe_low_v, 8),
            "max_error_sub": round(maxe_sub_v, 8),
            "max_error_improvement": round(maxe_low_v - maxe_sub_v, 8),
        }
        print(f"  prompt[{prompt_idx}] '{prompt[:40]}', {name}: delta={cos_ref_sub-cos_ref_low:+.6f}")

# ---- Step 7: Classification ----
print("\n[7] Classification...", flush=True)

ffn_up_rec = per_tensor_results["ffn_up"]["recorded_activations"]["improvement_delta"]
ffn_down_rec = per_tensor_results["ffn_down"]["recorded_activations"]["improvement_delta"]
ffn_up_ctrl = per_tensor_results["ffn_up"]["random_control"]["improvement_delta"]
ffn_down_ctrl = per_tensor_results["ffn_down"]["random_control"]["improvement_delta"]

ffn_up_pass = ffn_up_rec > 0
ffn_down_pass = ffn_down_rec > 0

if ffn_up_pass and ffn_down_pass:
    classification = "PASS_RECORDED_ACTIVATION_IMPROVEMENT"
elif ffn_up_pass and not ffn_down_pass:
    classification = "PARTIAL_FFN_UP_ONLY"
elif not ffn_up_pass and ffn_down_pass:
    classification = "PARTIAL_FFN_DOWN_ONLY"
else:
    if ffn_up_ctrl > 0 and ffn_down_ctrl > 0:
        classification = "PARTIAL_RANDOM_ONLY"
    elif ffn_up_ctrl > 0 or ffn_down_ctrl > 0:
        classification = "PARTIAL_RANDOM_ONLY"
    else:
        classification = "BLOCKED_NUMERICAL_ISSUE"

print(f"  ffn_up improvement (recorded):   {ffn_up_rec:+.6f}  {'PASS' if ffn_up_pass else 'FAIL'}")
print(f"  ffn_down improvement (recorded): {ffn_down_rec:+.6f}  {'PASS' if ffn_down_pass else 'FAIL'}")
print(f"  ffn_up improvement (control):    {ffn_up_ctrl:+.6f}")
print(f"  ffn_down improvement (control):  {ffn_down_ctrl:+.6f}")
print(f"  Classification: {classification}", flush=True)

# ---- Step 8: Write JSON ----
json_results = {
    "phase": "31E",
    "classification": classification,
    "old_head": "cd46f2b",
    "new_head": None,
    "encoding": "dense_bitmap_fp16_top15pct",
    "model": MODEL_PATH,
    "tensor_shapes": {
        "ffn_up": list(weights["ffn_up"]["W_ref"].shape),
        "ffn_down": list(weights["ffn_down"]["W_ref"].shape),
    },
    "residual_budgets_from_31D": {
        "ffn_up": 1906688,
        "ffn_down": 2485504,
    },
    "random_seeds": SEEDS,
    "activation_capture": {
        "method": "transformers forward_pre_hook on layer0.mlp.gate_proj + down_proj output",
        "n_predict": 1,
        "cpu_only": True,
        "prompts": PROMPTS,
        "hf_model": HF_MODEL_PATH,
    },
    "prompt_breakdown": prompt_breakdown,
    "per_tensor": per_tensor_results,
    "summary": {
        "ffn_up": {
            "random_control_improvement": round(ffn_up_ctrl, 6),
            "recorded_improvement": round(ffn_up_rec, 6),
            "pass": ffn_up_pass,
        },
        "ffn_down": {
            "random_control_improvement": round(ffn_down_ctrl, 6),
            "recorded_improvement": round(ffn_down_rec, 6),
            "pass": ffn_down_pass,
        },
    },
}

json_path = RESULTS_DIR / "PHASE31E_RECORDED_ACTIVATION_PROBE.json"
with open(json_path, "w") as f:
    json.dump(json_results, f, indent=2)
print(f"\n[Wrote] {json_path}", flush=True)

# ---- Step 9: Write MD ----
md_lines = [
    "# Phase 31E: Recorded Activation Probe Results",
    "",
    f"**Classification:** `{classification}`",
    "",
    "## Executive Summary",
    "",
    "Tests whether Phase 31D sparse residual encoding (bitmap + top-15% + fp16) improves over W_low under **real activation distributions** (not just seeded random X).",
    "",
    "| Tensor | Control Delta (random X) | Recorded Delta (real activations) | PASS? |",
    "|--------|------------------------|-----------------------------------|-------|",
    f"| ffn_up | {ffn_up_ctrl:+.6f} | {ffn_up_rec:+.6f} | {'✅' if ffn_up_pass else '❌'} |",
    f"| ffn_down | {ffn_down_ctrl:+.6f} | {ffn_down_rec:+.6f} | {'✅' if ffn_down_pass else '❌'} |",
    "",
    "## Activation Capture",
    "",
    f"- **Method:** transformers hooks on Qwen2.5-0.5B layer 0 (hf model)",
    f"- **Gate input:** forward_pre_hook on layer0.mlp.gate_proj",
    f"- **Down proj output:** forward_hook on layer0.mlp.down_proj",
    f"- **CPU only:** yes",
    f"- **Prompts:** {PROMPTS}",
    "",
    "## Memory Budget (31D encoding)",
    "",
    f"- **Encoding:** dense_bitmap_fp16 @ k=15%",
    f"- **ffn_up:** nnz={weights['ffn_up']['nnz']:,}, total_bytes={weights['ffn_up']['total_bytes']:,}",
    f"- **ffn_down:** nnz={weights['ffn_down']['nnz']:,}, total_bytes={weights['ffn_down']['total_bytes']:,}",
    "",
    "## Per-Prompt Breakdown",
    "",
    "| Prompt | Tensor | W_low Cosine | W_sub Cosine | Delta | MAE_low | MAE_sub | MAE Imprv |",
    "|--------|--------|-------------|-------------|-------|---------|---------|----------|",
]

for prompt_idx, prompt in enumerate(PROMPTS):
    key = f"prompt_{prompt_idx}"
    for name in ["ffn_up", "ffn_down"]:
        pb = prompt_breakdown[key]["tensors"][name]
        md_lines.append(
            f"| '{prompt[:30]}' | {name} | {pb['cosine_ref_low']:.6f} | {pb['cosine_ref_sub']:.6f} | {pb['improvement_delta']:+.6f} | {pb['MAE_low']:.6f} | {pb['MAE_sub']:.6f} | {pb['MAE_improvement']:+.6f} |"
        )

md_lines += [
    "",
    "## Mean & Worst-Case Across Prompts",
    "",
    f"- **ffn_up random control delta:** {ffn_up_ctrl:+.6f}",
    f"- **ffn_up recorded delta:** {ffn_up_rec:+.6f}",
    f"- **ffn_down random control delta:** {ffn_down_ctrl:+.6f}",
    f"- **ffn_down recorded delta:** {ffn_down_rec:+.6f}",
    f"- **ffn_up MAE improvement (recorded):** {per_tensor_results['ffn_up']['recorded_activations']['MAE_improvement']:+.6f}",
    f"- **ffn_down MAE improvement (recorded):** {per_tensor_results['ffn_down']['recorded_activations']['MAE_improvement']:+.6f}",
    f"- **ffn_up max-error improvement (recorded):** {per_tensor_results['ffn_up']['recorded_activations']['max_error_improvement']:+.6f}",
    f"- **ffn_down max-error improvement (recorded):** {per_tensor_results['ffn_down']['recorded_activations']['max_error_improvement']:+.6f}",
    "",
    "## Decision Gate",
    "",
    "```",
    f"ffn_up improvement (recorded):   {ffn_up_rec:+.6f}  {'> 0 → PASS' if ffn_up_pass else '≤ 0 → FAIL'}",
    f"ffn_down improvement (recorded): {ffn_down_rec:+.6f}  {'> 0 → PASS' if ffn_down_pass else '≤ 0 → FAIL'}",
    f"Classification: {classification}",
    "```",
    "",
    "## Recommendation",
    "",
]

if classification == "PASS_RECORDED_ACTIVATION_IMPROVEMENT":
    md_lines += [
        "✅ **Proceed to Phase 31F:** multi-layer recorded activation sweep (layers 0–2)",
        "✅ **Proceed to Phase 31G:** substitutive compute prototype design",
    ]
elif classification.startswith("PARTIAL"):
    md_lines += [
        "⚠️ **Partial success — requires tuning**",
        "  Consider: adjusting k_pct, value precision, or per-row/per-block policy",
        "  Next step: diagnose which activations fail and why",
    ]
else:
    md_lines += [
        "❌ **Blocked** — residual encoding improves on random X but not on real activations",
        "  Report: residual economics pass but activation-conditioned approximation fails",
    ]

md_path = REPO_DIR / "docs" / "PHASE31E_RECORDED_ACTIVATION_PROBE_RESULTS.md"
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"[Wrote] {md_path}", flush=True)

print("\nDone.", flush=True)
