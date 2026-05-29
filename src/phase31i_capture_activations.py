#!/usr/bin/env python3
"""
Capture all 15 prompts for layer/family activation matrices (layers 0-5).
hook input -> up_proj / down_proj => shape (batch, seq, d_hidden_per_fc.weight)
For ffn_up (d_in=d_hidden=896): activation is the hidden state entering FFN
For ffn_down (d_in=d_intermediate=4864): activation is the intermediate (act*up_proj output)
"""

import sys
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_DIR = "/home/matthew-villnave/sdi-substitutive"
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import gguf

PROMPTS = [
    "Hi",
    "The capital of France is",
    "2+2=",
    "def add(a, b):",
    "Once upon a time",
    "What is the largest planet?",
    "x = 5 * 3",
    "class MyClass:",
    "It was a dark and stormy night",
    '{"name": "John", "age":',
    "Sorry, I can't help with that.",
    "apple, banana, cherry,",
    "The reason for this is",
    "Hey there!",
    "🦆",
]

TARGET_LAYERS = [0, 1, 2, 3, 4, 5]
TENSOR_FAMILIES = ["ffn_up", "ffn_down"]


def main():
    print("Loading GGUF for tensor shapes...", flush=True)
    reader = gguf.GGUFReader(
        "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )
    tensors = {t.name: t for t in reader.tensors}

    tensor_shapes = {}   # (layer, family) -> (d_out, d_in)
    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            key = f"blk.{layer}.{family}.weight"
            t = tensors[key]
            W_ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
            tensor_shapes[(layer, family)] = W_ref.shape
            print(f"  {key}: {W_ref.shape}", flush=True)

    # Load model ONCE
    print("\nLoading model...", flush=True)
    tok_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        tok_name, torch_dtype=torch.float32, trust_remote_code=True
    ).cpu().eval()
    print(f"  Model ready", flush=True)

    saved = {}

    def make_hook(layer_idx, family):
        def hook(mod, input_tuple):
            x_in = input_tuple[0].detach().cpu().numpy()
            saved[(layer_idx, family)] = x_in
        return hook

    # Register hooks: one per target layer per family
    handles = []
    for layer_idx in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            mlp = model.model.layers[layer_idx].mlp
            fc = mlp.up_proj if family == "ffn_up" else mlp.down_proj
            handles.append(fc.register_forward_pre_hook(make_hook(layer_idx, family)))

    print(f"  Registered {len(handles)} hooks", flush=True)

    # Run all 15 prompts
    all_activations = {
        layer: {family: [] for family in TENSOR_FAMILIES}
        for layer in TARGET_LAYERS
    }

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cpu()
        saved.clear()
        with torch.no_grad():
            model(input_ids)

        for layer_idx in TARGET_LAYERS:
            for family in TENSOR_FAMILIES:
                d_out, d_in = tensor_shapes[(layer_idx, family)]
                act = saved.get((layer_idx, family))
                if act is None:
                    act = np.zeros((1, d_in), dtype=np.float32)
                else:
                    # act shape: (batch=1, seq_len, d_in)
                    if act.ndim == 3:
                        act = act[0]   # (seq_len, d_in)
                    # Last token
                    act = act[-1]     # (d_in,)
                all_activations[layer_idx][family].append(act)

        print(f"  Prompt {pi:2d}: {prompt[:50]!r} done", flush=True)

    for h in handles:
        h.remove()

    # Assemble and save
    print("\nAssembling and saving...", flush=True)
    OUTPUT_NPZ = os.path.join(REPO_DIR, "data", "PHASE31I_activations.npz")
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)

    to_save = {}
    for layer in TARGET_LAYERS:
        for family in TENSOR_FAMILIES:
            key = f"layer{layer}_{family}"
            arr = np.stack(all_activations[layer][family]).astype(np.float32)
            to_save[key] = arr
            print(f"  {key}: shape={arr.shape}", flush=True)

    np.savez(OUTPUT_NPZ, **to_save)
    print(f"\n[Wrote] {OUTPUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
