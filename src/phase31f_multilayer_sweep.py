#!/usr/bin/env python3
"""
Phase 31F: Multi-Layer Recorded Activation Sweep
Tests whether 31D/31E residual encoding (dense_bitmap + global top-15% + fp16)
generalizes across layers 0-2 for both ffn_up and ffn_down tensors.

Same encoding policy as Phase 31D/31E, but extended to layers 0, 1, 2.
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

TARGET_LAYERS = [0, 1, 2]
TENSOR_FAMILIES = ["ffn_up", "ffn_down"]

def cosine_batch(X, Y):
    return float(np.sum(X * Y) / (np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel()) + 1e-10))

def mae_batch(Y_ref, Y):
    return float(np.mean(np.abs(Y_ref - Y)))

def maxae_batch(Y_ref, Y):
    return float(np.max(np.abs(Y_ref - Y)))

print("=" * 60, flush=True)
print("Phase 31F: Multi-Layer Recorded Activation Sweep", flush=True)
print("=" * 60, flush=True)

# ---- Step 1: Load GGUF and extract tensors for all 6 target tensors ----
print("\n[1] Loading GGUF and extracting tensors for layers 0-2...", flush=True)
reader = gguf.GGUFReader(MODEL_PATH)
tensors = {t.name: t for t in reader.tensors}

# Map: (layer, family) -> weight dict
weights = {}
for layer in TARGET_LAYERS:
    for family in TENSOR_FAMILIES:
        key = f"blk.{layer}.{family}.weight"
        t = tensors[key]
        W_ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
        d_out, d_in = W_ref.shape
        
        # Build W_low: round(W_ref / scale) * scale, scale = std/2.5
        scale = float(np.std(W_ref) / 2.5)
        W_low = np.round(W_ref / scale) * scale
        R_f32 = (W_ref - W_low).astype(np.float32)
        
        # Memory budget: Q4 bytes vs Q2 bytes
        # For Q4_K approximation of [d_out, d_in]:
        # Q4 uses 4.5 bits per element = 0.5625 bytes per element
        n_elements = d_out * d_in
        Q4_bytes = int(n_elements * 0.5625)
        # For Q2 (2 bits per element): 0.25 bytes per element
        Q2_bytes = int(n_elements * 0.25)
        residual_budget = Q4_bytes - Q2_bytes
        
        weights[(layer, family)] = {
            "W_ref": W_ref,
            "W_low": W_low,
            "R_f32": R_f32,
            "d_out": d_out,
            "d_in": d_in,
            "scale": scale,
            "Q4_bytes": Q4_bytes,
            "Q2_bytes": Q2_bytes,
            "residual_budget": residual_budget,
            "layer": layer,
            "family": family,
        }
        print(f"  blk.{layer}.{family}: W_ref={W_ref.shape}, Q4={Q4_bytes:,} bytes, budget={residual_budget:,} bytes", flush=True)

# ---- Step 2: Build 31D/31E sparse encodings for all 6 tensors ----
print("\n[2] Building 31D/31E sparse encodings (dense_bitmap + top-15% + fp16)...", flush=True)

def build_sparse_fp15(residual, k_pct=15):
    """dense_bitmap + global top-k% + fp16 values. Returns R_sub, metadata."""
    d_out, d_in = residual.shape
    n_elements = residual.size
    nnz = max(1, int(n_elements * k_pct / 100))
    
    # Dense bitmap: one bit per element, byte-aligned
    bitmap_bytes = int(n_elements + 7) // 8
    
    # Find global top-k% by magnitude
    R_abs = np.abs(residual).flatten()
    threshold = np.partition(R_abs, -nnz)[-nnz]
    mask_flat = R_abs >= threshold
    mask = mask_flat.reshape(d_out, d_in)
    
    # Build sparse residual
    R_sub = np.zeros_like(residual)
    R_sub[mask] = residual[mask]
    residual_values = R_sub[mask].astype(np.float16).view(np.uint16).astype(np.uint16)
    value_bytes = len(residual_values) * 2
    
    total_bytes = bitmap_bytes + value_bytes
    
    return R_sub.astype(np.float32), {
        "nnz": nnz,
        "bitmap_bytes": bitmap_bytes,
        "value_bytes": value_bytes,
        "total_bytes": total_bytes,
    }

for key, w in weights.items():
    R_sub, meta = build_sparse_fp15(w["R_f32"], k_pct=15)
    w["R_sub"] = R_sub
    w["nnz"] = meta["nnz"]
    w["bitmap_bytes"] = meta["bitmap_bytes"]
    w["value_bytes"] = meta["value_bytes"]
    w["total_bytes"] = meta["total_bytes"]
    w["memory_viable"] = w["total_bytes"] <= w["residual_budget"]
    layer, family = key
    print(f"  blk.{layer}.{family}: nnz={meta['nnz']:,}, bitmap={meta['bitmap_bytes']:,}, "
          f"values={meta['value_bytes']:,}, total={meta['total_bytes']:,}, "
          f"budget={w['residual_budget']:,}, viable={w['memory_viable']}", flush=True)

# ---- Step 3: Capture real activations for all layers 0, 1, 2 ----
print("\n[3] Capturing real activations from Qwen2.5-0.5B (layers 0-2)...", flush=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_path = HF_MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf_path, dtype=torch.float32, trust_remote_code=True)
model = model.to("cpu").eval()

# Activation capture per layer per family
# ffn_up input: forward_pre_hook on layerN.mlp.gate_proj (the SiLU gate proj)
# ffn_down input: forward_pre_hook on layerN.mlp.down_proj (input to down_proj)
# These are the X matrices for the matmul tests

# Storage: {(layer, family): [np.array, ...]} per prompt
activation_storage = {(layer, family): [] for layer in TARGET_LAYERS for family in TENSOR_FAMILIES}

# Hook handles storage
hooks = []

for layer_idx in TARGET_LAYERS:
    layer = model.model.layers[layer_idx]
    
    # ffn_up input = gate_proj forward_pre_hook (the MLP up-projection input)
    # The gate_proj receives the hidden state; we capture the first token's vector
    gate_in_list = [None]
    def make_gate_hook(container):
        def hook(mod, input):
            container[0] = input[0].detach().clone()
        return hook
    h_gate = layer.register_forward_pre_hook(make_gate_hook(gate_in_list), prepend=True)
    
    # ffn_down input = down_proj forward_pre_hook (SiLU(gate(X)) * up(X) result)
    down_in_list = [None]
    def make_down_hook(container):
        def hook(mod, input):
            container[0] = input[0].detach().clone()
        return hook
    h_down = layer.mlp.down_proj.register_forward_pre_hook(make_down_hook(down_in_list), prepend=True)
    
    hooks.append((h_gate, h_down))

print(f"  Capturing for {len(PROMPTS)} prompts across {len(TARGET_LAYERS)} layers...", flush=True)
t0 = time.time()

for prompt_idx, prompt in enumerate(PROMPTS):
    # Reset containers
    for h_gate, h_down in hooks:
        pass  # containers reset by closure
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        model(**inputs)
    
    # Extract first token, first batch from each layer's captured activations
    for li, layer_idx in enumerate(TARGET_LAYERS):
        h_gate, h_down = hooks[li]
        gate_in = [None]  # re-bind won't work, but each closure captured separate lists
        
    # Actually the closures share the outer scope list refs... let me re-check.
    # Each hook creates its own closure over a unique list. Let's verify by checking stored values.
    
    for li, layer_idx in enumerate(TARGET_LAYERS):
        # Re-run capture check: each hook's container was set during model(**inputs)
        pass  # placeholder

print(f"  NOTE: Hook capture verification needed - reading stored closures...", flush=True)

# Re-architect: each hook pair stores in layer-specific list refs
activation_storage = {(layer, family): [] for layer in TARGET_LAYERS for family in TENSOR_FAMILIES}

# Create fresh hooks for actual capture
hooks = []
hook_state = {}  # layer_idx -> {'gate': [...], 'down': [...]}

for layer_idx in TARGET_LAYERS:
    layer = model.model.layers[layer_idx]
    
    # Capture list refs for this layer
    gate_container = {'val': None}
    down_container = {'val': None}
    
    h_gate = layer.register_forward_pre_hook(
        lambda mod, input, c=gate_container: c.__setitem__('val', input[0].detach().clone()),
        prepend=True
    )
    h_down = layer.mlp.down_proj.register_forward_pre_hook(
        lambda mod, input, c=down_container: c.__setitem__('val', input[0].detach().clone()),
        prepend=True
    )
    
    hooks.append((h_gate, h_down))
    hook_state[layer_idx] = {'gate': gate_container, 'down': down_container}

print(f"  Running {len(PROMPTS)} prompts...", flush=True)
t0 = time.time()

for prompt_idx, prompt in enumerate(PROMPTS):
    # Reset containers
    for layer_idx in TARGET_LAYERS:
        hook_state[layer_idx]['gate']['val'] = None
        hook_state[layer_idx]['down']['val'] = None
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        model(**inputs)
    
    # Collect first token first batch vectors
    for layer_idx in TARGET_LAYERS:
        gate_val = hook_state[layer_idx]['gate']['val']
        down_val = hook_state[layer_idx]['down']['val']
        
        if gate_val is not None:
            # shape: (batch, seq, d_model) - take first token
            vec_up = gate_val[0, 0, :].numpy()
        else:
            d_in = weights[(layer_idx, 'ffn_up')]['d_in']
            vec_up = np.zeros(d_in, dtype=np.float32)
        
        if down_val is not None:
            # shape: (batch, seq, d_ffn) - take first token
            vec_down = down_val[0, 0, :].numpy()
        else:
            d_in = weights[(layer_idx, 'ffn_down')]['d_in']
            vec_down = np.zeros(d_in, dtype=np.float32)
        
        activation_storage[(layer_idx, 'ffn_up')].append(vec_up)
        activation_storage[(layer_idx, 'ffn_down')].append(vec_down)
    
    print(f"    prompt[{prompt_idx}] '{prompt[:40]}': captured", flush=True)

# Remove hooks
for h_gate, h_down in hooks:
    h_gate.remove()
    h_down.remove()

print(f"  Activation capture done in {time.time()-t0:.1f}s", flush=True)

# Stack into matrices (n_prompts, d_in) for each tensor
for key in activation_storage:
    activation_storage[key] = np.stack(activation_storage[key], axis=0)
    layer, family = key
    print(f"  Activations blk.{layer}.{family}: shape={activation_storage[key].shape}", flush=True)

# ---- Step 4: Compute metrics for all 30 combos (5 prompts × 6 tensors) ----
print("\n[4] Computing metrics for all 30 combos...", flush=True)

# Per-combo metrics
all_results = []  # list of dicts for each prompt×tensor combo
prompt_results = {p: [] for p in range(len(PROMPTS))}  # index by prompt idx
layer_results = {l: [] for l in TARGET_LAYERS}
family_results = {f: [] for f in TENSOR_FAMILIES}

for layer_idx in TARGET_LAYERS:
    for family in TENSOR_FAMILIES:
        w = weights[(layer_idx, family)]
        W_ref = w["W_ref"]
        W_low = w["W_low"]
        R_sub = w["R_sub"]
        
        X_batch = activation_storage[(layer_idx, family)]  # (5, d_in)
        
        for prompt_idx in range(len(PROMPTS)):
            X = X_batch[prompt_idx : prompt_idx + 1]  # (1, d_in)
            
            Y_ref = X @ W_ref.T        # (1, d_out)
            Y_low = X @ W_low.T        # (1, d_out)
            Y_sub = X @ (W_low + R_sub).T  # (1, d_out)
            
            cos_ref_low = cosine_batch(Y_ref, Y_low)
            cos_ref_sub = cosine_batch(Y_ref, Y_sub)
            delta_cosine = cos_ref_sub - cos_ref_low
            
            mae_low_v = mae_batch(Y_ref, Y_low)
            mae_sub_v = mae_batch(Y_ref, Y_sub)
            mae_delta = mae_low_v - mae_sub_v
            
            maxe_low_v = maxae_batch(Y_ref, Y_low)
            maxe_sub_v = maxae_batch(Y_ref, Y_sub)
            maxe_delta = maxe_low_v - maxe_sub_v
            
            result = {
                "layer": layer_idx,
                "family": family,
                "prompt_idx": prompt_idx,
                "prompt": PROMPTS[prompt_idx],
                "cosine_ref_low": round(cos_ref_low, 6),
                "cosine_ref_sub": round(cos_ref_sub, 6),
                "delta_cosine": round(delta_cosine, 6),
                "MAE_low": round(mae_low_v, 8),
                "MAE_sub": round(mae_sub_v, 8),
                "MAE_improvement": round(mae_delta, 8),
                "max_error_low": round(maxe_low_v, 8),
                "max_error_sub": round(maxe_sub_v, 8),
                "max_error_improvement": round(maxe_delta, 8),
            }
            
            all_results.append(result)
            prompt_results[prompt_idx].append(result)
            layer_results[layer_idx].append(result)
            family_results[family].append(result)

# ---- Step 5: Summary tables ----
print("\n[5] Computing summary tables...", flush=True)

# A. Per-prompt table: avg delta_cosine across 6 tensors per prompt
print("\n  [A] Per-prompt table:", flush=True)
prompt_table = []
for prompt_idx in range(len(PROMPTS)):
    results = prompt_results[prompt_idx]
    deltas = [r["delta_cosine"] for r in results]
    mean_delta = round(float(np.mean(deltas)), 6)
    min_delta = round(float(np.min(deltas)), 6)
    max_delta = round(float(np.max(deltas)), 6)
    improving = sum(1 for d in deltas if d > 0)
    prompt_table.append({
        "prompt_idx": prompt_idx,
        "prompt": PROMPTS[prompt_idx],
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "improving_count": improving,
        "total_count": len(deltas),
        "improving_pct": round(improving / len(deltas) * 100, 1),
    })
    print(f"    prompt[{prompt_idx}] '{PROMPTS[prompt_idx][:40]}': mean={mean_delta:+.6f}, min={min_delta:+.6f}, improving={improving}/{len(deltas)}", flush=True)

# B. Per-layer table: avg delta_cosine across prompts and families
print("\n  [B] Per-layer table:", flush=True)
layer_table = []
for layer_idx in TARGET_LAYERS:
    results = layer_results[layer_idx]
    deltas = [r["delta_cosine"] for r in results]
    mean_delta = round(float(np.mean(deltas)), 6)
    min_delta = round(float(np.min(deltas)), 6)
    ffn_up_deltas = [r["delta_cosine"] for r in results if r["family"] == "ffn_up"]
    ffn_down_deltas = [r["delta_cosine"] for r in results if r["family"] == "ffn_down"]
    mean_ffn_up = round(float(np.mean(ffn_up_deltas)), 6)
    mean_ffn_down = round(float(np.mean(ffn_down_deltas)), 6)
    improving = sum(1 for d in deltas if d > 0)
    layer_table.append({
        "layer": layer_idx,
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "mean_ffn_up": mean_ffn_up,
        "mean_ffn_down": mean_ffn_down,
        "improving_count": improving,
        "total_count": len(deltas),
    })
    print(f"    layer{layer_idx}: mean={mean_delta:+.6f}, ffn_up={mean_ffn_up:+.6f}, ffn_down={mean_ffn_down:+.6f}, improving={improving}/{len(deltas)}", flush=True)

# C. Per-tensor-family table: ffn_up across all layers vs ffn_down across all layers
print("\n  [C] Per-family table:", flush=True)
family_table = []
for family in TENSOR_FAMILIES:
    results = family_results[family]
    deltas = [r["delta_cosine"] for r in results]
    mean_delta = round(float(np.mean(deltas)), 6)
    min_delta = round(float(np.min(deltas)), 6)
    max_delta = round(float(np.max(deltas)), 6)
    improving = sum(1 for d in deltas if d > 0)
    family_table.append({
        "family": family,
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "improving_count": improving,
        "total_count": len(deltas),
    })
    print(f"    {family}: mean={mean_delta:+.6f}, min={min_delta:+.6f}, improving={improving}/{len(deltas)}", flush=True)

# D. Worst-case delta
worst = min(all_results, key=lambda r: r["delta_cosine"])
print(f"\n  [D] Worst-case: layer={worst['layer']}, family={worst['family']}, "
      f"prompt='{worst['prompt']}', delta={worst['delta_cosine']:+.6f}", flush=True)

# E. Mean delta
all_deltas = [r["delta_cosine"] for r in all_results]
mean_all = round(float(np.mean(all_deltas)), 6)
print(f"  [E] Mean delta: {mean_all:+.6f} across {len(all_results)} combos", flush=True)

# F. Memory viability table
print("\n  [F] Memory viability table:", flush=True)
memory_table = []
for layer_idx in TARGET_LAYERS:
    for family in TENSOR_FAMILIES:
        w = weights[(layer_idx, family)]
        memory_table.append({
            "tensor": f"blk.{layer_idx}.{family}.weight",
            "shape": [w["d_out"], w["d_in"]],
            "Q4_bytes": w["Q4_bytes"],
            "Q2_bytes": w["Q2_bytes"],
            "residual_budget": w["residual_budget"],
            "encoded_bytes": w["total_bytes"],
            "memory_viable": w["memory_viable"],
        })
        print(f"    blk.{layer_idx}.{family}: Q4={w['Q4_bytes']:,}, budget={w['residual_budget']:,}, "
              f"encoded={w['total_bytes']:,}, viable={w['memory_viable']}", flush=True)

# ---- Step 6: Classification ----
print("\n[6] Classification...", flush=True)

# Check 1: Memory viability for all 6 tensors
all_viable = all(w["memory_viable"] for w in weights.values())

# Check 2: Improvement rate (>80% of combos with delta > 0)
improving_count = sum(1 for r in all_results if r["delta_cosine"] > 0)
improving_pct = improving_count / len(all_results) * 100
more_than_80_pct = improving_pct >= 80.0

# Check 3: No systematic family regression
ffn_up_deltas = [r["delta_cosine"] for r in all_results if r["family"] == "ffn_up"]
ffn_down_deltas = [r["delta_cosine"] for r in all_results if r["family"] == "ffn_down"]
mean_ffn_up = float(np.mean(ffn_up_deltas))
mean_ffn_down = float(np.mean(ffn_down_deltas))
no_systematic_regression = mean_ffn_up > -0.001 and mean_ffn_down > -0.001

# Check 4: Per-layer variance
layer_means = [layer_table[li]["mean_delta"] for li in range(len(TARGET_LAYERS))]
some_improve_some_not = len(set(np.sign(l) for l in layer_means if l != 0)) > 1

if all_viable and more_than_80_pct and no_systematic_regression:
    classification = "PASS_MULTILAYER_RECORDED_IMPROVEMENT"
elif all_viable and improving_pct >= 50:
    if mean_ffn_up > 0 and mean_ffn_down <= 0:
        classification = "PARTIAL_FFN_UP_ONLY"
    elif mean_ffn_down > 0 and mean_ffn_up <= 0:
        classification = "PARTIAL_FFN_DOWN_ONLY"
    elif some_improve_some_not:
        classification = "PARTIAL_LAYER_VARIANCE"
    else:
        classification = "PARTIAL_LAYER_VARIANCE"
elif not all_viable:
    classification = "PARTIAL_MEMORY_FAIL_ON_SOME_LAYERS"
else:
    classification = "BLOCKED_NUMERICAL_ISSUE"

print(f"  Memory viable (all 6): {all_viable}", flush=True)
print(f"  Improving: {improving_count}/{len(all_results)} ({improving_pct:.1f}%)", flush=True)
print(f"  Mean ffn_up delta: {mean_ffn_up:+.6f}", flush=True)
print(f"  Mean ffn_down delta: {mean_ffn_down:+.6f}", flush=True)
print(f"  Classification: {classification}", flush=True)

# ---- Step 7: Write JSON results ----
print("\n[7] Writing results...", flush=True)

json_results = {
    "phase": "31F",
    "classification": classification,
    "old_head": "6f83e4a",
    "new_head": None,
    "encoding": "dense_bitmap_fp16_top15pct",
    "model": MODEL_PATH,
    "layers_tested": TARGET_LAYERS,
    "tensor_families": TENSOR_FAMILIES,
    "prompts": PROMPTS,
    "total_combos": len(all_results),
    "improving_combos": improving_count,
    "improving_pct": round(improving_pct, 2),
    "per_combo": all_results,
    "prompt_table": prompt_table,
    "layer_table": layer_table,
    "family_table": family_table,
    "worst_case": {
        "layer": worst["layer"],
        "family": worst["family"],
        "prompt": worst["prompt"],
        "prompt_idx": worst["prompt_idx"],
        "delta_cosine": worst["delta_cosine"],
        "cosine_ref_low": worst["cosine_ref_low"],
        "cosine_ref_sub": worst["cosine_ref_sub"],
    },
    "mean_delta": mean_all,
    "memory_viability_table": memory_table,
    "summary": {
        "all_memory_viable": all_viable,
        "improving_pct": round(improving_pct, 2),
        "mean_ffn_up_delta": round(mean_ffn_up, 6),
        "mean_ffn_down_delta": round(mean_ffn_down, 6),
    },
}

json_path = RESULTS_DIR / "PHASE31F_MULTILAYER_RECORDED_ACTIVATION_SWEEP.json"
with open(json_path, "w") as f:
    json.dump(json_results, f, indent=2)
print(f"  [Wrote] {json_path}", flush=True)

# ---- Step 8: Write MD summary ----
md_lines = [
    "# Phase 31F: Multi-Layer Recorded Activation Sweep Results",
    "",
    f"**Classification:** `{classification}`",
    "",
    "## Metadata",
    "",
    f"- **Old HEAD:** `6f83e4a`",
    f"- **New HEAD:** `pending`",
    f"- **Layers tested:** {TARGET_LAYERS}",
    f"- **Tensor families:** {TENSOR_FAMILIES}",
    f"- **Prompts:** {PROMPTS}",
    f"- **Total combos:** {len(all_results)} (5 prompts × 6 tensors)",
    "",
    "## Encoding Policy",
    "",
    "Same as Phase 31D/31E: **dense_bitmap + global top-15% + fp16 residual values**",
    "",
    "## Summary Metrics",
    "",
    f"- **Improving combos:** {improving_count}/{len(all_results)} ({improving_pct:.1f}%)",
    f"- **Mean delta cosine:** {mean_all:+.6f}",
    f"- **Worst-case delta:** {worst['delta_cosine']:+.6f} (blk.{worst['layer']}.{worst['family']}, prompt='{worst['prompt'][:30]}')",
    f"- **All memory viable:** {all_viable}",
    "",
    "## F. Memory Viability Table",
    "",
    "| Tensor | Shape | Q4 (bytes) | Q2 (bytes) | Budget (bytes) | Encoded (bytes) | Viable? |",
    "|--------|-------|-----------|-----------|----------------|-----------------|---------|",
]

for row in memory_table:
    md_lines.append(
        f"| blk.{row['tensor'].split('.')[1]}.{row['tensor'].split('.')[2]} | "
        f"{row['shape'][0]}×{row['shape'][1]} | "
        f"{row['Q4_bytes']:,} | {row['Q2_bytes']:,} | {row['residual_budget']:,} | "
        f"{row['encoded_bytes']:,} | {'✅' if row['memory_viable'] else '❌'} |"
    )

md_lines += [
    "",
    "## A. Per-Prompt Table (delta cosine averaged across 6 tensors)",
    "",
    "| Prompt | Mean ΔCos | Min ΔCos | Max ΔCos | Improving |",
    "|--------|----------|---------|---------|----------|",
]

for row in prompt_table:
    md_lines.append(
        f"| '{row['prompt'][:40]}' | {row['mean_delta']:+.6f} | {row['min_delta']:+.6f} | "
        f"{row['max_delta']:+.6f} | {row['improving_count']}/{row['total_count']} ({row['improving_pct']:.0f}%) |"
    )

md_lines += [
    "",
    "## B. Per-Layer Table (delta cosine averaged across prompts & families)",
    "",
    "| Layer | Mean ΔCos | Min ΔCos | ffn_up Δ | ffn_down Δ | Improving |",
    "|-------|----------|---------|----------|------------|----------|",
]

for row in layer_table:
    md_lines.append(
        f"| {row['layer']} | {row['mean_delta']:+.6f} | {row['min_delta']:+.6f} | "
        f"{row['mean_ffn_up']:+.6f} | {row['mean_ffn_down']:+.6f} | {row['improving_count']}/{row['total_count']} |"
    )

md_lines += [
    "",
    "## C. Per-Tensor-Family Table (averaged across all layers & prompts)",
    "",
    "| Family | Mean ΔCos | Min ΔCos | Max ΔCos | Improving |",
    "|---------|----------|---------|---------|----------|",
]

for row in family_table:
    md_lines.append(
        f"| {row['family']} | {row['mean_delta']:+.6f} | {row['min_delta']:+.6f} | "
        f"{row['max_delta']:+.6f} | {row['improving_count']}/{row['total_count']} |"
    )

md_lines += [
    "",
    "## D. Worst-Case Result",
    "",
    f"- **Layer:** {worst['layer']}",
    f"- **Family:** {worst['family']}",
    f"- **Prompt:** '{worst['prompt']}'",
    f"- **delta_cosine:** {worst['delta_cosine']:+.6f}",
    f"- **cosine_ref_low:** {worst['cosine_ref_low']:.6f}",
    f"- **cosine_ref_sub:** {worst['cosine_ref_sub']:.6f}",
    "",
    "## E. Mean Delta",
    "",
    f"- **Mean delta cosine (all 30 combos):** {mean_all:+.6f}",
    "",
    "## Decision Gate",
    "",
    "```",
    f"memory_viable (all 6 tensors): {all_viable}",
    f"improving_pct: {improving_pct:.1f}% (need ≥80%)",
    f"mean_ffn_up_delta: {mean_ffn_up:+.6f}",
    f"mean_ffn_down_delta: {mean_ffn_down:+.6f}",
    f"Classification: {classification}",
    "```",
    "",
]

if classification == "PASS_MULTILAYER_RECORDED_IMPROVEMENT":
    md_lines += [
        "✅ **Proceed to Phase 31G:** substitutive compute prototype design",
        "✅ **Proceed to Phase 31H:** compressed residual compute harness",
        "⚠️ llama.cpp integration still blocked until compute prototype passes",
    ]
elif classification == "PARTIAL_LAYER_VARIANCE":
    md_lines += [
        "⚠️ **Partial success — layer variance detected**",
        "  Recommendation: tune sparsity by layer/family",
        "  Options: top-10%, top-12.5%, top-15%, top-20%, per-layer policy",
    ]
elif classification == "PARTIAL_FFN_UP_ONLY":
    md_lines += [
        "⚠️ **Partial: ffn_up generalizes, ffn_down does not**",
        "  Recommendation: per-family sparsity policy",
    ]
elif classification == "PARTIAL_FFN_DOWN_ONLY":
    md_lines += [
        "⚠️ **Partial: ffn_down generalizes, ffn_up does not**",
        "  Recommendation: per-family sparsity policy",
    ]
elif classification == "PARTIAL_MEMORY_FAIL_ON_SOME_LAYERS":
    md_lines += [
        "❌ **Memory budget exceeded for some layers**",
        "  Recommendation: reduce k_pct or use row-wise sparsity",
    ]
else:
    md_lines += [
        "❌ **Blocked**",
        "  Report: residual encoding does not generalize across layers 0-2",
    ]

md_path = REPO_DIR / "docs" / "PHASE31F_MULTILAYER_RECORDED_ACTIVATION_SWEEP.md"
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  [Wrote] {md_path}", flush=True)

print("\nDone.", flush=True)