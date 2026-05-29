#!/usr/bin/env python3
"""
Phase 31G: Sparse Residual k-Sweep / Memory-Viable Policy Selection
Find the smallest k% that fits the Q4→Q2 residual memory budget
while preserving positive improvement over W_low on recorded activations.
"""
import os, sys, json, time, pathlib, math
import numpy as np

REPO_DIR = pathlib.Path.home() / "sdi-substitutive"
RESULTS_DIR = REPO_DIR / "results"
DATA_DIR = REPO_DIR / "data"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, str(REPO_DIR / "src"))
import gguf

MODEL_PATH = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
HF_MODEL_PATH = "/home/matthew-villnave/models/hf/qwen2.5/Qwen2.5-0.5B-Instruct"

PROMPTS = ["Hi", "The capital of France is", "2+2=", "def add(a, b):", "Once upon a time"]
TARGET_LAYERS = [0, 1, 2]
TENSOR_FAMILIES = ["ffn_up", "ffn_down"]

# k values to sweep (primary + optional)
K_PRIMARY = [5, 7.5, 10, 12.5, 15]
K_OPTIONAL = [8, 9, 11]
ALL_K = sorted(set(K_PRIMARY + K_OPTIONAL))

def cosine_batch(X, Y):
    return float(np.sum(X * Y) / (np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel()) + 1e-10))

def mae_batch(Y_ref, Y):
    return float(np.mean(np.abs(Y_ref - Y)))

def maxae_batch(Y_ref, Y):
    return float(np.max(np.abs(Y_ref - Y)))

print("=" * 70, flush=True)
print("Phase 31G: Sparse Residual k-Sweep / Memory-Viable Policy", flush=True)
print("=" * 70, flush=True)

# ---- Step 1: Load GGUF and extract weights ----
print("\n[1] Loading GGUF and extracting weights for layers 0-2...", flush=True)
reader = gguf.GGUFReader(MODEL_PATH)
tensors = {t.name: t for t in reader.tensors}

weights = {}
for layer in TARGET_LAYERS:
    for family in TENSOR_FAMILIES:
        key = f"blk.{layer}.{family}.weight"
        t = tensors[key]
        W_ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
        d_out, d_in = W_ref.shape

        scale = float(np.std(W_ref) / 2.5)
        W_low = np.round(W_ref / scale) * scale
        R_f32 = (W_ref - W_low).astype(np.float32)

        n_elements = d_out * d_in
        Q4_bytes = int(n_elements * 0.5625)
        Q2_bytes = int(n_elements * 0.25)
        residual_budget = Q4_bytes - Q2_bytes

        weights[(layer, family)] = {
            "W_ref": W_ref, "W_low": W_low, "R_f32": R_f32,
            "d_out": d_out, "d_in": d_in, "scale": scale,
            "Q4_bytes": Q4_bytes, "Q2_bytes": Q2_bytes,
            "residual_budget": residual_budget,
            "layer": layer, "family": family,
        }
        print(f"  blk.{layer}.{family}: shape={d_out}×{d_in}, "
              f"Q4={Q4_bytes:,} bytes, budget={residual_budget:,} bytes", flush=True)

# ---- Step 2: Encode sparse residuals for all k values ----
print(f"\n[2] Encoding sparse residuals for k in {ALL_K}...", flush=True)

def build_sparse_fp16(residual, k_pct):
    """dense_bitmap + global top-k% + fp16 values. Returns R_sub and metadata."""
    d_out, d_in = residual.shape
    n_elements = residual.size
    nnz = max(1, int(n_elements * k_pct / 100))

    bitmap_bytes = int(n_elements + 7) // 8

    R_abs = np.abs(residual).flatten()
    threshold = np.partition(R_abs, -nnz)[-nnz]
    mask_flat = R_abs >= threshold
    mask = mask_flat.reshape(d_out, d_in)

    R_sub = np.zeros_like(residual)
    R_sub[mask] = residual[mask]
    residual_values = R_sub[mask].astype(np.float16).view(np.uint16).astype(np.uint16)
    value_bytes = len(residual_values) * 2

    total_bytes = bitmap_bytes + value_bytes

    return R_sub.astype(np.float32), {
        "nnz": nnz, "bitmap_bytes": bitmap_bytes,
        "value_bytes": value_bytes, "total_bytes": total_bytes,
    }

# Pre-encode all k values for all 6 tensors
encodings = {}  # (layer, family, k) -> encoded dict
for key, w in weights.items():
    for k in ALL_K:
        R_sub, meta = build_sparse_fp16(w["R_f32"], k_pct=k)
        encodings[(*key, k)] = {
            "R_sub": R_sub,
            "nnz": meta["nnz"],
            "bitmap_bytes": meta["bitmap_bytes"],
            "value_bytes": meta["value_bytes"],
            "total_bytes": meta["total_bytes"],
            "memory_viable": meta["total_bytes"] <= w["residual_budget"],
            "memory_margin": w["residual_budget"] - meta["total_bytes"],
        }

print(f"  Encoded {len(encodings)} (tensor, k) combos", flush=True)

# Print memory viability per k per tensor
print("\n  Memory viability by k and tensor:", flush=True)
print(f"  {'k%':>5} | {'Layer.Family':<18} | {'nnz':>8} | {'bitmap':>8} | "
      f"{'values':>8} | {'total':>8} | {'budget':>8} | {'margin':>8} | {'OK?':>4}", flush=True)
print("  " + "-" * 90, flush=True)
for k in ALL_K:
    for key in weights:
        layer, family = key
        e = encodings[(*key, k)]
        w = weights[key]
        print(f"  {k:>5} | blk.{layer}.{family:<14} | {e['nnz']:>8,} | "
              f"{e['bitmap_bytes']:>8,} | {e['value_bytes']:>8,} | "
              f"{e['total_bytes']:>8,} | {w['residual_budget']:>8,} | "
              f"{e['memory_margin']:>8,} | {'✅' if e['memory_viable'] else '❌':>4}", flush=True)

# ---- Step 3: Load activations from Phase 31F data (or re-capture) ----
print("\n[3] Loading/capturing activations...", flush=True)

# Check if 31F activation data is saved
f31f_json = RESULTS_DIR / "PHASE31F_MULTILAYER_RECORDED_ACTIVATION_SWEEP.json"
f31f_npz = DATA_DIR / "PHASE31G_activations.npz"

if f31f_npz.exists():
    print("  Loading cached activations from NPZ...", flush=True)
    data = np.load(f31f_npz)
    activation_storage = {}
    for (layer, family) in [(l, f) for l in TARGET_LAYERS for f in TENSOR_FAMILIES]:
        key = f"layer{l}_{family}"
        activation_storage[(layer, family)] = data[key]
else:
    print("  Capturing fresh activations via HuggingFace...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_path = HF_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_path, dtype=torch.float32, trust_remote_code=True)
    model = model.to("cpu").eval()

    activation_storage = {(l, f): [] for l in TARGET_LAYERS for f in TENSOR_FAMILIES}
    hooks = []
    hook_state = {}

    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
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

    t0 = time.time()
    for prompt_idx, prompt in enumerate(PROMPTS):
        for layer_idx in TARGET_LAYERS:
            hook_state[layer_idx]['gate']['val'] = None
            hook_state[layer_idx]['down']['val'] = None

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)

        for layer_idx in TARGET_LAYERS:
            gate_val = hook_state[layer_idx]['gate']['val']
            down_val = hook_state[layer_idx]['down']['val']

            if gate_val is not None:
                vec_up = gate_val[0, 0, :].numpy()
            else:
                vec_up = np.zeros(weights[(layer_idx, 'ffn_up')]['d_in'], dtype=np.float32)

            if down_val is not None:
                vec_down = down_val[0, 0, :].numpy()
            else:
                vec_down = np.zeros(weights[(layer_idx, 'ffn_down')]['d_in'], dtype=np.float32)

            activation_storage[(layer_idx, 'ffn_up')].append(vec_up)
            activation_storage[(layer_idx, 'ffn_down')].append(vec_down)

        print(f"    prompt[{prompt_idx}] captured", flush=True)

    for h_gate, h_down in hooks:
        h_gate.remove()
        h_down.remove()

    print(f"  Capture done in {time.time()-t0:.1f}s", flush=True)

    # Save to NPZ
    save_dict = {}
    for (layer, family), arr in activation_storage.items():
        save_dict[f"layer{layer}_{family}"] = arr
    np.savez(f31f_npz, **save_dict)
    print(f"  [Saved] {f31f_npz}", flush=True)

# Stack
for key in activation_storage:
    activation_storage[key] = np.stack(activation_storage[key], axis=0)
    layer, family = key
    print(f"  Activations blk.{layer}.{family}: shape={activation_storage[key].shape}", flush=True)

# ---- Step 4: Compute metrics for all k values ----
print("\n[4] Computing metrics for all k values...", flush=True)

# For each k, compute per-combo metrics
k_results = {}  # k -> {per_combo, summary}
prompt_indices = list(range(len(PROMPTS)))

for k in ALL_K:
    print(f"\n  === k={k}% ===", flush=True)
    all_combos = []

    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            w = weights[(layer, family)]
            e = encodings[(layer, family, k)]
            R_sub = e["R_sub"]
            W_ref = w["W_ref"]
            W_low = w["W_low"]
            X_batch = activation_storage[(layer, family)]  # (5, d_in)

            for pi, prompt in enumerate(PROMPTS):
                X = X_batch[pi : pi + 1]  # (1, d_in)

                Y_ref = X @ W_ref.T
                Y_low = X @ W_low.T
                Y_sub = X @ (W_low + R_sub).T

                cos_low = cosine_batch(Y_ref, Y_low)
                cos_sub = cosine_batch(Y_ref, Y_sub)
                delta_cos = cos_sub - cos_low

                mae_low_v = mae_batch(Y_ref, Y_low)
                mae_sub_v = mae_batch(Y_ref, Y_sub)
                mae_delta = mae_low_v - mae_sub_v

                maxe_low_v = maxae_batch(Y_ref, Y_low)
                maxe_sub_v = maxae_batch(Y_ref, Y_sub)
                maxe_delta = maxe_low_v - maxe_sub_v

                all_combos.append({
                    "layer": layer, "family": family,
                    "prompt_idx": pi, "prompt": prompt,
                    "cosine_ref_low": round(cos_low, 6),
                    "cosine_ref_sub": round(cos_sub, 6),
                    "delta_cosine": round(delta_cos, 6),
                    "MAE_low": round(mae_low_v, 8),
                    "MAE_sub": round(mae_sub_v, 8),
                    "MAE_improvement": round(mae_delta, 8),
                    "max_error_low": round(maxe_low_v, 8),
                    "max_error_sub": round(maxe_sub_v, 8),
                    "max_error_improvement": round(maxe_delta, 8),
                    "memory_viable": e["memory_viable"],
                    "memory_margin": e["memory_margin"],
                    "total_bytes": e["total_bytes"],
                    "residual_budget": w["residual_budget"],
                })

    k_results[k] = all_combos

    # Summary
    deltas = [c["delta_cosine"] for c in all_combos]
    improving = sum(1 for d in deltas if d > 0)
    worst_delta = min(deltas)
    best_delta = max(deltas)
    mean_delta = round(float(np.mean(deltas)), 6)
    worst_combo = min(all_combos, key=lambda c: c["delta_cosine"])
    all_viable = all(c["memory_viable"] for c in all_combos)
    any_viable = any(c["memory_viable"] for c in all_combos)

    print(f"    combos={len(all_combos)}, improving={improving}/{len(all_combos)}, "
          f"mean_delta={mean_delta:+.6f}, worst={worst_delta:+.6f}, "
          f"all_viable={all_viable}, any_viable={any_viable}", flush=True)

    k_results[k] = {
        "per_combo": all_combos,
        "summary": {
            "k": k,
            "total_combos": len(all_combos),
            "improving_count": improving,
            "improving_pct": round(improving / len(all_combos) * 100, 1),
            "mean_delta": mean_delta,
            "worst_delta": round(worst_delta, 6),
            "best_delta": round(best_delta, 6),
            "all_memory_viable": all_viable,
            "any_memory_viable": any_viable,
            "worst_layer": worst_combo["layer"],
            "worst_family": worst_combo["family"],
            "worst_prompt": worst_combo["prompt"],
        }
    }

# ---- Step 5: Summary tables ----
print("\n[5] Computing summary tables...", flush=True)

# Table 1: k vs memory viability per tensor
print("\n  [Table 1] k vs memory viability per tensor:", flush=True)
print(f"  {'k%':>5}", end="", flush=True)
for layer in TARGET_LAYERS:
    for family in TENSOR_FAMILIES:
        print(f" | {'L'+str(layer)+'.'+family:>12}", end="", flush=True)
print(f" | {'ALL_VIABLE':>10}", flush=True)
print("  " + "-" * 100, flush=True)

memory_viable_table = []
for k in ALL_K:
    row = {"k": k, "tensors": {}}
    viable_count = 0
    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            e = encodings[(layer, family, k)]
            viable = e["memory_viable"]
            row["tensors"][f"blk.{layer}.{family}"] = viable
            if viable:
                viable_count += 1
            symbol = "✅" if viable else "❌"
            print(f"  {k:>5} | {symbol}", end="", flush=True)
    row["all_viable"] = viable_count == 6
    print(f" | {'✅' if row['all_viable'] else str(viable_count)+'/6':>10}", flush=True)
    memory_viable_table.append(row)

# Table 2: k vs mean delta cosine
print("\n  [Table 2] k vs mean/worst delta cosine:", flush=True)
print(f"  {'k%':>5} | {'mean_delta':>12} | {'worst_delta':>12} | {'improving%':>10} | {'all_viable':>10}", flush=True)
print("  " + "-" * 60, flush=True)

delta_table = []
for k in ALL_K:
    s = k_results[k]["summary"]
    print(f"  {k:>5} | {s['mean_delta']:>+12.6f} | {s['worst_delta']:>+12.6f} | "
          f"{s['improving_pct']:>10.1f} | {'✅' if s['all_memory_viable'] else '❌':>10}", flush=True)
    delta_table.append({
        "k": k,
        "mean_delta": s["mean_delta"],
        "worst_delta": s["worst_delta"],
        "improving_pct": s["improving_pct"],
        "all_memory_viable": s["all_memory_viable"],
    })

# Table 3: Best global k (one k, viable for all 6 tensors, positive delta)
print("\n  [Table 3] Best global k analysis:", flush=True)
global_viable = []
for k in ALL_K:
    s = k_results[k]["summary"]
    if s["all_memory_viable"] and s["mean_delta"] > 0:
        global_viable.append({"k": k, "mean_delta": s["mean_delta"],
                               "worst_delta": s["worst_delta"], "improving_pct": s["improving_pct"]})

if global_viable:
    global_viable.sort(key=lambda x: x["worst_delta"], reverse=True)
    best_global = global_viable[0]
    print(f"  Best global k: {best_global['k']}%", flush=True)
    print(f"    mean_delta={best_global['mean_delta']:+.6f}, "
          f"worst_delta={best_global['worst_delta']:+.6f}, "
          f"improving_pct={best_global['improving_pct']}%", flush=True)
else:
    best_global = None
    print("  No global k works for all 6 tensors.", flush=True)

# Table 4: Adaptive per-family k
print("\n  [Table 4] Adaptive per-family k analysis:", flush=True)
adaptive_results = {}
for family in TENSOR_FAMILIES:
    best_for_family = []
    for k in ALL_K:
        s = k_results[k]["summary"]
        # Check viable for all layers of this family (3 tensors each)
        family_viable = all(
            encodings[(layer, family, k)]["memory_viable"]
            for layer in TARGET_LAYERS
        )
        if family_viable:
            best_for_family.append({"k": k, "mean_delta": s["mean_delta"],
                                     "worst_delta": s["worst_delta"],
                                     "improving_pct": s["improving_pct"]})
    if best_for_family:
        best_for_family.sort(key=lambda x: x["worst_delta"], reverse=True)
        adaptive_results[family] = best_for_family[0]
    print(f"  {family}: viable ks={[x['k'] for x in best_for_family]}, "
          f"best={[x['k'] for x in best_for_family[:3]]}", flush=True)

# Table 5: Per-layer adaptive k
print("\n  [Table 5] Adaptive per-layer k analysis:", flush=True)
layer_adaptive = {}
for layer in TARGET_LAYERS:
    best_for_layer = []
    for k in ALL_K:
        s = k_results[k]["summary"]
        layer_viable = all(
            encodings[(layer, family, k)]["memory_viable"]
            for family in TENSOR_FAMILIES
        )
        if layer_viable:
            best_for_layer.append({"k": k, "mean_delta": s["mean_delta"],
                                    "worst_delta": s["worst_delta"]})
    if best_for_layer:
        best_for_layer.sort(key=lambda x: x["worst_delta"], reverse=True)
        layer_adaptive[layer] = best_for_layer[0]
    print(f"  layer{layer}: viable ks={[x['k'] for x in best_for_layer]}, "
          f"best={best_for_layer[0]['k'] if best_for_layer else 'none'}%", flush=True)

# ---- Step 6: Classification ----
print("\n[6] Classification...", flush=True)

# Rule A: Global policy
if best_global is not None:
    # Verify worst_delta is positive
    s = k_results[best_global["k"]]["summary"]
    if s["worst_delta"] > 0 and s["improving_pct"] >= 80:
        classification = "PASS_GLOBAL_MEMORY_VIABLE_POLICY"
    elif s["worst_delta"] > 0:
        classification = "PASS_GLOBAL_MEMORY_VIABLE_POLICY"
    else:
        classification = "PARTIAL_APPROXIMATION_GOOD_MEMORY_FAILS"
elif adaptive_results:
    # Rule B: Per-family adaptive
    all_families_viable = all(
        encodings[(layer, family, adaptive_results[family]["k"])]["memory_viable"]
        for layer in TARGET_LAYERS
        for family in TENSOR_FAMILIES
    )
    if all_families_viable:
        classification = "PASS_ADAPTIVE_MEMORY_VIABLE_POLICY"
    else:
        classification = "PARTIAL_MEMORY_VIABLE_BUT_WEAK_IMPROVEMENT"
elif global_viable := [k for k in ALL_K if k_results[k]["summary"]["all_memory_viable"] and k_results[k]["summary"]["mean_delta"] > 0]:
    # Some k values are memory viable but have weak/negative worst-case
    classification = "PARTIAL_MEMORY_VIABLE_BUT_WEAK_IMPROVEMENT"
else:
    classification = "PARTIAL_APPROXIMATION_GOOD_MEMORY_FAILS"

print(f"  Classification: {classification}", flush=True)

# ---- Step 7: Build final decision ----
print("\n[7] Final decision...", flush=True)

best_global_k = best_global["k"] if best_global else None
best_global_summary = k_results[best_global_k]["summary"] if best_global else None

adaptive_policy = {}
if classification == "PASS_ADAPTIVE_MEMORY_VIABLE_POLICY":
    for family, info in adaptive_results.items():
        adaptive_policy[family] = {
            "k": info["k"],
            "mean_delta": info["mean_delta"],
            "worst_delta": info["worst_delta"],
        }

print(f"  Best global k: {best_global_k}%", flush=True)
print(f"  Adaptive policy: {adaptive_policy}", flush=True)
print(f"  Classification: {classification}", flush=True)

phase31h_unlocked = classification.startswith("PASS_")

# ---- Step 8: Write results JSON ----
print("\n[8] Writing results JSON...", flush=True)

# Collect per-k summary for all combos
per_k_summaries = {}
for k in ALL_K:
    s = k_results[k]["summary"]
    per_k_summaries[str(k)] = {
        "k": k,
        "mean_delta_cosine": s["mean_delta"],
        "worst_delta_cosine": s["worst_delta"],
        "best_delta_cosine": s["best_delta"],
        "improving_count": s["improving_count"],
        "improving_pct": s["improving_pct"],
        "all_memory_viable": s["all_memory_viable"],
        "any_memory_viable": s["any_memory_viable"],
    }

# Memory viability matrix
memory_matrix = []
for k in ALL_K:
    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            e = encodings[(layer, family, k)]
            w = weights[(layer, family)]
            memory_matrix.append({
                "k_pct": k,
                "tensor": f"blk.{layer}.{family}.weight",
                "shape": [w["d_out"], w["d_in"]],
                "Q4_bytes": w["Q4_bytes"],
                "Q2_bytes": w["Q2_bytes"],
                "residual_budget": w["residual_budget"],
                "total_bytes": e["total_bytes"],
                "memory_margin": e["memory_margin"],
                "memory_viable": e["memory_viable"],
            })

json_results = {
    "phase": "31G",
    "classification": classification,
    "old_head": "d21f814",
    "new_head": None,
    "k_values_tested": ALL_K,
    "layers_tested": TARGET_LAYERS,
    "tensor_families": TENSOR_FAMILIES,
    "total_tensors": 6,
    "prompts": PROMPTS,
    "best_global_k": best_global_k,
    "best_global_summary": {
        "k": best_global_k,
        "mean_delta_cosine": best_global_summary["mean_delta"] if best_global_summary else None,
        "worst_delta_cosine": best_global_summary["worst_delta"] if best_global_summary else None,
        "improving_pct": best_global_summary["improving_pct"] if best_global_summary else None,
        "all_memory_viable": best_global_summary["all_memory_viable"] if best_global_summary else None,
    } if best_global_summary else None,
    "adaptive_policy": adaptive_policy if adaptive_policy else None,
    "phase31h_unlocked": phase31h_unlocked,
    "per_k_summaries": per_k_summaries,
    "memory_matrix": memory_matrix,
    "delta_table": delta_table,
    "memory_viable_table": memory_viable_table,
}

json_path = RESULTS_DIR / "PHASE31G_SPARSE_K_SWEEP_POLICY.json"
with open(json_path, "w") as f:
    json.dump(json_results, f, indent=2)
print(f"  [Wrote] {json_path}", flush=True)

# ---- Step 9: Write MD summary ----
print("\n[9] Writing MD summary...", flush=True)

md_lines = [
    "# Phase 31G: Sparse Residual k-Sweep / Memory-Viable Policy Selection",
    "",
    f"**Classification:** `{classification}`",
    "",
    "## Metadata",
    "",
    f"- **Old HEAD:** `d21f814`",
    f"- **New HEAD:** `pending`",
    f"- **k values tested:** {ALL_K}",
    f"- **Layers tested:** {TARGET_LAYERS}",
    f"- **Tensor families:** {TENSOR_FAMILIES}",
    f"- **Prompts:** {PROMPTS}",
    f"- **Total combos per k:** {5*6} = 30 (5 prompts × 6 tensors)",
    "",
    "## Encoding Policy",
    "",
    "`dense_bitmap + top-k% + fp16 residual values`",
    "",
    "## Summary",
    "",
    f"- **Best global k:** {best_global_k}%",
    f"- **Adaptive policy:** {adaptive_policy if adaptive_policy else 'N/A'}",
    f"- **Phase 31H unlocked:** {phase31h_unlocked}",
    "",
    "## Table 1: k vs Memory Viability (per tensor)",
    "",
    "| k% | " + " | ".join([f"L{l}.{f}" for l in TARGET_LAYERS for f in TENSOR_FAMILIES]) + " | ALL_VIABLE |",
    "|" + "|".join(["---"] * (6 + 1)) + "|",
]

for row in memory_viable_table:
    cells = [f"{row['k']}%"]
    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            key = f"blk.{layer}.{family}"
            cells.append("✅" if row["tensors"][key] else "❌")
    cells.append("✅" if row["all_viable"] else "❌")
    md_lines.append("| " + " | ".join(cells) + " |")

md_lines += [
    "",
    "## Table 2: k vs Approximation Quality (delta cosine)",
    "",
    "| k% | Mean ΔCos | Worst ΔCos | Improving% | All Memory Viable |",
    "|-----|-----------|-----------|-----------|------------------|",
]

for row in delta_table:
    md_lines.append(
        f"| {row['k']}% | {row['mean_delta']:+.6f} | {row['worst_delta']:+.6f} | "
        f"{row['improving_pct']:.1f}% | {'✅' if row['all_memory_viable'] else '❌'} |"
    )

if best_global_k is not None:
    md_lines += [
        "",
        "## Best Global k Policy",
        "",
        f"- **k:** {best_global_k}%",
        f"- **Mean ΔCos:** {best_global_summary['mean_delta']:+.6f}",
        f"- **Worst ΔCos:** {best_global_summary['worst_delta']:+.6f}",
        f"- **Improving:** {best_global_summary['improving_pct']}% of combos",
        f"- **All Memory Viable:** {best_global_summary['all_memory_viable']}",
        "",
    ]

if adaptive_policy:
    md_lines += [
        "## Best Adaptive k Policy (per-family)",
        "",
        "| Family | k% | Mean ΔCos | Worst ΔCos |",
        "|--------|---|-----------|-----------|",
    ]
    for family, info in adaptive_policy.items():
        s = k_results[info["k"]]["summary"]
        md_lines.append(
            f"| {family} | {info['k']}% | {s['mean_delta']:+.6f} | {s['worst_delta']:+.6f} |"
        )
    md_lines.append("")

md_lines += [
    "## Decision Gate",
    "",
    "```",
    f"classification: {classification}",
    f"best_global_k: {best_global_k}",
    f"adaptive_policy: {adaptive_policy}",
    f"phase31h_unlocked: {phase31h_unlocked}",
    "```",
    "",
]

if classification == "PASS_GLOBAL_MEMORY_VIABLE_POLICY":
    md_lines.append("✅ **Phase 31H unlocked.** Proceed to compressed residual compute harness.")
elif classification == "PASS_ADAPTIVE_MEMORY_VIABLE_POLICY":
    md_lines.append("✅ **Phase 31H unlocked with adaptive policy.** Proceed to compressed residual compute harness.")
elif classification == "PARTIAL_MEMORY_VIABLE_BUT_WEAK_IMPROVEMENT":
    md_lines.append("⚠️ Memory-viable but improvements weak. Consider per-layer k tuning.")
elif classification == "PARTIAL_APPROXIMATION_GOOD_MEMORY_FAILS":
    md_lines.append("❌ Approximation good but no k fits memory. Try block sparsity or mixed precision.")
else:
    md_lines.append("❌ Blocked. Numerical issues or no viable policy found.")

md_path = REPO_DIR / "docs" / "PHASE31G_SPARSE_K_SWEEP_POLICY.md"
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"  [Wrote] {md_path}", flush=True)

print("\n✅ Phase 31G complete.", flush=True)