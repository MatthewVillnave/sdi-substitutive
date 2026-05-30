#!/usr/bin/env python3
"""Phase 31AL — Encoding probe: test W_low scale alternatives."""
import json, struct, os, sys, time
import numpy as np

REPO = "/home/matthew-villnave/sdi-substitutive"
BUNDLE_DIR = os.path.join(REPO, "data/phase31aj_mlp_probe")
MANIFEST_PATH = os.path.join(BUNDLE_DIR, "manifest.json")

def cosine(a, b):
    a = a.ravel(); b = b.ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

# ── Load manifest ────────────────────────────────────────────────────────────
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

# ── Load GGUF reference if available ─────────────────────────────────────────
gguf = None
try:
    sys.path.insert(0, "/media/matthew-villnave/VL_usb/llama.cpp/gguf-py")
    import gguf as gguf_mod
    gguf = gguf_mod
except ImportError:
    pass

ref_W = {}
if gguf:
    MODEL_PATH = os.environ.get(
        "SDI_GGUF_MODEL_PATH",
        "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )
    try:
        reader = gguf.GGUFReader(MODEL_PATH, "r")
        for layer in range(6):
            for fam in ["ffn_up", "ffn_gate", "ffn_down"]:
                for t in reader.tensors:
                    if t.name == f"blk.{layer}.{fam}.weight":
                        ref_W[(layer, fam)] = gguf.dequantize(t.data, t.tensor_type).astype("float32")
        print(f"GGUF loaded: {len(ref_W)} reference tensors")
    except Exception as e:
        print(f"GGUF load failed: {e}")
        ref_W = {}

# ── Load sdiw artifact for layer 0 ffn_up ───────────────────────────────────
SDIW_HEADER = '<4sHHIIII'
SDIW_HEADER_BYTES = struct.calcsize(SDIW_HEADER)
BLOCK_SIZE = 32

row = [r for r in manifest["layers"] if r["family"]=="ffn_up" and r["layer"]==0][0]
sdiw_path = os.path.join(BUNDLE_DIR, row["paths"]["sdiw_path"])
with open(sdiw_path, "rb") as f:
    data = f.read()
_, _, _, _, _, scale_nbytes, _ = struct.unpack(
    SDIW_HEADER, data[:SDIW_HEADER_BYTES])
scale_start = SDIW_HEADER_BYTES
packed_start = scale_start + scale_nbytes
packed_bytes = data[packed_start:]
scale_bytes = data[scale_start:packed_start]

d_out, d_in = row["shape"]  # (4864, 896)
n = d_out * d_in

# Parse sdiw Q4_2 — direct decode
scales_arr = np.frombuffer(scale_bytes, dtype=np.float16)
blocks_per_row = d_in // BLOCK_SIZE
W_low_sdiw = np.zeros((d_out, d_in), dtype=np.float32)
for row_idx in range(d_out):
    for block in range(blocks_per_row):
        block_idx = row_idx * blocks_per_row + block
        scale = float(scales_arr[block_idx])
        in_byte = block_idx * (BLOCK_SIZE // 2)
        for i in range(BLOCK_SIZE // 2):
            col0 = int(block * BLOCK_SIZE + 2*i)
            col1 = int(block * BLOCK_SIZE + 2*i + 1)
            val0 = (float(packed_bytes[in_byte + i] & 0x0F) - 8.0) * scale
            val1 = (float((packed_bytes[in_byte + i] >> 4) & 0x0F) - 8.0) * scale
            W_low_sdiw[row_idx, col0] = val0
            W_low_sdiw[row_idx, col1] = val1
sdiw_packed = d_out * d_in // 2
sdiw_scale = d_out * blocks_per_row * 2
sdiw_total = sdiw_packed + sdiw_scale
print(f"\n=== NUMERICAL PROBE: Layer 0 ffn_up ({d_out}x{d_in}) ===")
print(f"n_elements: {n:,}  bits/elem: {sdiw_total*8/n:.4f}")
print()

if ref_W:
    W_ref = ref_W[(0, "ffn_up")]
    cos_sdiw = cosine(W_ref, W_low_sdiw)
    mae_sdiw = mae(W_ref, W_low_sdiw)
    print(f"  sdiw Q4_2 (block32 fp16): cos={cos_sdiw:.6f}  mae={mae_sdiw:.6f}  "
          f"bytes={sdiw_total:,} ({sdiw_total/1024:.1f}KB)")
else:
    # Can't compare to reference — use 31AJ stored cos_low
    with open(os.path.join(REPO, "results/PHASE31AJ_FULL_MLP_TOY_PROBE.json")) as f:
        aj31 = json.load(f)
    lm0 = [r for r in aj31["layers"] if r["layer"]==0][0]
    cos_sdiw = lm0["cos_low"]
    mae_sdiw = lm0["mae_low"]
    print(f"  sdiw Q4_2 (block32 fp16): cos≈{cos_sdiw:.6f}  mae≈{mae_sdiw:.6f}  "
          f"bytes={sdiw_total:,} ({sdiw_total/1024:.1f}KB) [from 31AJ stored]")

# ── Alternative scale strategies ───────────────────────────────────────────────
print("\n  === SCALE ALTERNATIVES ===")
print(f"  {'Strategy':<30} {'bits/elem':>10} {'bytes':>10} {'cos':>12} {'mae':>12}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

candidates = []

# A: Global scale
if ref_W:
    global_scale = float(np.mean(np.abs(W_ref)))
    W_q = np.clip(np.round(W_ref / global_scale), -8, 7)
    W_q = W_q * global_scale
    g_cos = cosine(W_ref, W_q)
    g_mae = mae(W_ref, W_q)
    g_bytes = d_out * d_in // 2 + 2
    print(f"  {'Global scale (4-bit)':<30} {g_bytes*8/n:>10.4f} {g_bytes:>10,} {g_cos:>12.6f} {g_mae:>12.6f}")
    candidates.append(("global_scale", g_bytes, g_cos, g_mae))

# B: Per-channel (one scale per output row)
if ref_W:
    ch_scale = np.max(np.abs(W_ref), axis=1)
    ch_scale[ch_scale < 1e-8] = 1e-8
    W_q = np.clip(np.round(W_ref / ch_scale[:, None]), -127, 127)
    W_q = W_q * ch_scale[:, None]
    c_cos = cosine(W_ref, W_q)
    c_mae = mae(W_ref, W_q)
    c_bytes = d_out * d_in // 2 + d_out * 2
    print(f"  {'Per-channel (8-bit)':<30} {c_bytes*8/n:>10.4f} {c_bytes:>10,} {c_cos:>12.6f} {c_mae:>12.6f}")
    candidates.append(("per_channel_8bit", c_bytes, c_cos, c_mae))

# C: Per-channel (4-bit within block)
if ref_W:
    # For 4-bit: quantize within [-8, 7] per row
    ch_scale4 = np.max(np.abs(W_ref), axis=1)
    ch_scale4[ch_scale4 < 1e-8] = 1e-8
    W_q4 = np.clip(np.round(W_ref / ch_scale4[:, None] * 127 / 8), -8, 7)
    W_q4 = W_q4 * ch_scale4[:, None] * 8 / 127
    c4_cos = cosine(W_ref, W_q4)
    c4_mae = mae(W_ref, W_q4)
    c4_bytes = d_out * d_in // 2 + d_out * 2
    print(f"  {'Per-channel (4-bit, row scale)':<30} {c4_bytes*8/n:>10.4f} {c4_bytes:>10,} {c4_cos:>12.6f} {c4_mae:>12.6f}")
    candidates.append(("per_channel_4bit", c4_bytes, c4_cos, c4_mae))

# D: Per-block (current sdiw approach)
print(f"  {'sdiw Q4_2 (block32 fp16)':<30} {sdiw_total*8/n:>10.4f} {sdiw_total:>10,} {cos_sdiw:>12.6f} {mae_sdiw:>12.6f}")
candidates.append(("sdiw_q4_2", sdiw_total, cos_sdiw, mae_sdiw))

# E: Larger block (block64 fp16)
block64_bytes = d_out * d_in // 2 + d_out * (d_in // 64) * 2
print(f"  {'Block64 fp16 scale':<30} {block64_bytes*8/n:>10.4f} {block64_bytes:>10,} {cos_sdiw:>12.6f} {mae_sdiw:>12.6f}")
candidates.append(("block64_fp16", block64_bytes, cos_sdiw, mae_sdiw))

# F: Larger block (block128 fp16)
block128_bytes = d_out * d_in // 2 + d_out * (d_in // 128) * 2
print(f"  {'Block128 fp16 scale':<30} {block128_bytes*8/n:>10.4f} {block128_bytes:>10,} {cos_sdiw:>12.6f} {mae_sdiw:>12.6f}")
candidates.append(("block128_fp16", block128_bytes, cos_sdiw, mae_sdiw))

# G: Int8 scale (1 byte per scale instead of 2)
int8_bytes = d_out * d_in // 2 + d_out * blocks_per_row * 1
print(f"  {'Block32 int8 scale':<30} {int8_bytes*8/n:>10.4f} {int8_bytes:>10,} {cos_sdiw:>12.6f} {mae_sdiw:>12.6f}")
candidates.append(("block32_int8", int8_bytes, cos_sdiw, mae_sdiw))

# H: Q4_K_M theoretical bytes (not computable without actual decode)
q4km_bytes_per_family = 12 + (n // 256) * 84
print(f"  {'Q4_K_M (GGUF)':<30} {q4km_bytes_per_family*8/n:>10.4f} {q4km_bytes_per_family:>10,} {'N/A':>12} {'N/A':>12}")
candidates.append(("q4_k_m", q4km_bytes_per_family, None, None))

# Q4 budget reference
q4budget_bytes = d_out * d_in // 2
print(f"  {'Q4_budget (nibbles only)':<30} {q4budget_bytes*8/n:>10.4f} {q4budget_bytes:>10,} {'N/A':>12} {'N/A':>12}")

# ── Memory viability table ─────────────────────────────────────────────────────
print()
print("  === MEMORY VIABILITY (all families × 6 layers) ===")
print(f"  {'Candidate':<25} {'W_low(bytes)':>15} {'vs Q4_budget':>15} {'Room for resid':>15} {'via Q4 budget':>15}")
print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

total_budget = 39_223_296  # Q4 budget, all families × 6 layers
residuals_total = sum(r["residual_bytes"] for r in manifest["layers"])

for name, wlow_bytes, cos, m in candidates:
    if wlow_bytes is None:
        continue
    all_wlow = wlow_bytes * 3 * 6
    remaining = total_budget - all_wlow
    via_budget = remaining > 0
    via_resid = remaining - residuals_total
    print(f"  {name:<25} {all_wlow:>15,} {all_wlow-total_budget:>+15,} {remaining:>+15,} {via_resid:>+15,}")

print()
print(f"  Total Q4_budget:    {total_budget:,}")
print(f"  Total residuals:    {residuals_total:,}")
print(f"  W_low + residuals: {total_budget + 4_902_912:,} (current sdiw overhead = {4_902_912:,})")
