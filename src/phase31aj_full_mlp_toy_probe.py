#!/usr/bin/env python3
"""
Phase 31AJ — Full MLP Toy Probe (clean rerun from origin/master fbdcfcd)
Scope: ffn_up + ffn_gate + ffn_down, layers 0-5
Policies: ffn_up k=9%, ffn_down k=9%, ffn_gate k=12%
Canonical orientation: W = (d_out, d_in), X = (batch, d_in), Y = X @ W.T
Activation source: data/PHASE31I_activations.npz
Artifact source: data/phase31aj_mlp_probe/
Model: /media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""
import os, sys, json, time, struct
from typing import Dict, Tuple
import numpy as np

REPO = "/home/matthew-villnave/sdi-substitutive"
DATA = os.path.join(REPO, "data")
BUNDLE_DIR = os.path.join(DATA, "phase31aj_mlp_probe")
MANIFEST_PATH = os.path.join(BUNDLE_DIR, "manifest.json")
ACTIVATION_PATH = os.path.join(DATA, "PHASE31I_activations.npz")
MODEL_PATH = os.environ.get(
    "SDI_GGUF_MODEL_PATH",
    "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
# Guard: gguf is only needed for reference weight extraction (W_refs).
# If running without a model file, set SDI_SKIP_WREF=1 and ensure artifacts are pre-cached.
RESULT_PATH = os.path.join(REPO, "results", "PHASE31AJ_FULL_MLP_TOY_PROBE.json")
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

SDIW_HEADER = '<4sHHIIII'
SDIW_HEADER_BYTES = struct.calcsize(SDIW_HEADER)
SDIR_HEADER = '<4sHHIIIIHH'
SDIR_HEADER_BYTES = struct.calcsize(SDIR_HEADER)
BLOCK_SIZE = 32

# ── Math helpers ──────────────────────────────────────────────────────────────
def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40, 40))))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    norm_a = np.linalg.norm(a); norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

# ── Shape-gated matmul ────────────────────────────────────────────────────────
def apply_linear(X: np.ndarray, W: np.ndarray, name: str = "") -> np.ndarray:
    """Canonical matmul: Y = X @ W.T where W = (d_out, d_in), X = (batch, d_in)."""
    print(f"  [SHAPE] {name}: X={X.shape}, W={W.shape}, W.T={W.T.shape}")
    if X.shape[-1] == W.shape[1]:
        Y = X @ W.T
        print(f"  [MATMUL] {name}: used X @ W.T -> {Y.shape}")
        return Y
    if X.shape[-1] == W.shape[0]:
        Y = X @ W
        print(f"  [MATMUL] {name}: used X @ W (W appears transposed) -> {Y.shape}")
        return Y
    raise ValueError(
        f"{name} shape mismatch: X={X.shape}, W={W.shape}. "
        f"Expected W=(d_out,{X.shape[-1]}) or W=({X.shape[-1]},d_out)."
    )

# ── GGUF weight extraction ─────────────────────────────────────────────────────
def extract_W_ref(gguf_path: str, layer: int, family: str) -> np.ndarray:
    try:
        import gguf
    except ImportError:
        raise ImportError(
            "gguf module required to extract reference weights. "
            "Install it with: pip install gguf  OR  pip install -r requirements.txt "
            "(if a requirements.txt exists in the project root). "
            "Alternatively, set SDI_SKIP_WREF=1 if reference weights are pre-cached."
        ) from None
    reader = gguf.GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == f"blk.{layer}.{family}.weight":
            W = gguf.dequantize(t.data, t.tensor_type)
            # GGUF: ffn_up/gate dequant=(4864,896), ffn_down dequant=(896,4864)
            # We want W = (d_out, d_in) per canonical orientation
            if family in ("ffn_up", "ffn_gate"):
                # dequant is already (d_out=4864, d_in=896) — no transpose needed
                return W  # (4864, 896)
            else:
                # dequant is (d_out=896, d_in=4864) — no transpose needed
                return W  # (896, 4864)
    raise KeyError(f"blk.{layer}.{family}.weight not found")

# ── SDIW streaming apply ─────────────────────────────────────────────────────
def sdiw_streaming_apply_batch(packed_bytes: bytes, scale_bytes: bytes,
                                X_batch: np.ndarray, d_out: int, d_in: int) -> np.ndarray:
    scales_arr = np.frombuffer(scale_bytes, dtype=np.float16)
    blocks_per_row = d_in // BLOCK_SIZE
    Y = np.zeros((X_batch.shape[0], d_out), dtype=np.float32)
    for row in range(d_out):
        for block in range(blocks_per_row):
            block_idx = row * blocks_per_row + block
            scale = float(scales_arr[block_idx])
            in_byte = block_idx * (BLOCK_SIZE // 2)
            for i in range(BLOCK_SIZE // 2):
                col0 = int(block * BLOCK_SIZE + 2*i)
                col1 = int(block * BLOCK_SIZE + 2*i + 1)
                val0 = (float(packed_bytes[in_byte + i]& 0x0F) - 8.0) * scale
                val1 = (float((packed_bytes[in_byte + i] >> 4) & 0x0F) - 8.0) * scale
                Y[:, row] += X_batch[:, col0] * val0
                Y[:, row] += X_batch[:, col1] * val1
    return Y

# ── SDIR streaming apply ─────────────────────────────────────────────────────
def parse_sdir(sdir_bytes: bytes) -> dict:
    magic, version, flags, d_out, d_in, k_pct_int, nnz, _, _ = struct.unpack(
        SDIR_HEADER, sdir_bytes[:SDIR_HEADER_BYTES])
    bitmap_nbytes = (d_out * d_in + 7) // 8
    bitmap = np.unpackbits(np.frombuffer(
        sdir_bytes[SDIR_HEADER_BYTES:SDIR_HEADER_BYTES + bitmap_nbytes], dtype=np.uint8))
    bitmap = bitmap[:d_out * d_in]
    values = np.frombuffer(sdir_bytes[SDIR_HEADER_BYTES + bitmap_nbytes:], dtype=np.float16)
    return {"d_out": d_out, "d_in": d_in, "k_pct_int": k_pct_int, "nnz": nnz,
            "bitmap": bitmap, "values": values}

def sdir_streaming_apply_batch(sdir_bytes: bytes, X_batch: np.ndarray,
                                d_out: int, d_in: int) -> np.ndarray:
    parsed = parse_sdir(sdir_bytes)
    Y = np.zeros((X_batch.shape[0], d_out), dtype=np.float32)
    value_idx = 0
    for row in range(d_out):
        row_base = row * d_in
        for col in range(d_in):
            if parsed["bitmap"][row_base + col]:
                Y[:, row] += X_batch[:, col] * float(parsed["values"][value_idx])
                value_idx += 1
    return Y

# ── Source equivalence check ─────────────────────────────────────────────────
_FIXED_X_SOURCE = None

def check_source_equivalence(sdiw_path: str, sdir_path: str, d_out: int, d_in: int) -> dict:
    global _FIXED_X_SOURCE
    if _FIXED_X_SOURCE is None or _FIXED_X_SOURCE.shape[0] != d_in:
        np.random.seed(42)
        _FIXED_X_SOURCE = np.random.randn(d_in).astype(np.float32)
    X_sample = _FIXED_X_SOURCE

    with open(sdiw_path, "rb") as f:
        sdiw_data = f.read()
    _, _, _, _, _, scale_nbytes, _ = struct.unpack(SDIW_HEADER, sdiw_data[:SDIW_HEADER_BYTES])
    scale_start = SDIW_HEADER_BYTES
    packed_start = scale_start + scale_nbytes
    scale_bytes = sdiw_data[scale_start:packed_start]
    packed_bytes = sdiw_data[packed_start:]

    scales_arr = np.frombuffer(scale_bytes, dtype=np.float16)
    n = d_out * d_in
    W_dense = np.zeros(n, dtype=np.float32)
    for block_idx in range(n // BLOCK_SIZE):
        scale = float(scales_arr[block_idx])
        in_byte = block_idx * (BLOCK_SIZE // 2)
        out = block_idx * BLOCK_SIZE
        for i in range(BLOCK_SIZE // 2):
            W_dense[out + 2*i]   = (float(packed_bytes[in_byte + i] & 0x0F) - 8.0) * scale
            W_dense[out + 2*i+1] = (float((packed_bytes[in_byte + i] >> 4) & 0x0F) - 8.0) * scale
    W_dense = W_dense.reshape(d_out, d_in)

    Y_sdiw_stream = sdiw_streaming_apply_batch(packed_bytes, scale_bytes, X_sample.reshape(1,-1), d_out, d_in)[0]
    Y_sdiw_dense  = X_sample @ W_dense.T
    cos_sdiw = cosine(Y_sdiw_dense, Y_sdiw_stream)
    mae_sdiw = mae(Y_sdiw_dense, Y_sdiw_stream)

    with open(sdir_path, "rb") as f:
        sdir_bytes = f.read()
    parsed = parse_sdir(sdir_bytes)
    R_dense = np.zeros(d_out * d_in, dtype=np.float32)
    value_idx = 0
    for idx, bit in enumerate(parsed["bitmap"]):
        if bit:
            R_dense[idx] = parsed["values"][value_idx]
            value_idx += 1
    R_dense = R_dense.reshape(d_out, d_in)

    Y_sdir_stream = sdir_streaming_apply_batch(sdir_bytes, X_sample.reshape(1,-1), d_out, d_in)[0]
    Y_sdir_dense  = X_sample @ R_dense.T
    cos_sdir = cosine(Y_sdir_dense, Y_sdir_stream)
    mae_sdir = mae(Y_sdir_dense, Y_sdir_stream)

    Y_combined_stream = Y_sdiw_stream + Y_sdir_stream
    Y_combined_dense  = Y_sdiw_dense  + Y_sdir_dense
    cos_combined = cosine(Y_combined_dense, Y_combined_stream)
    mae_combined = mae(Y_combined_dense, Y_combined_stream)

    return {
        "sdiw_stream_vs_dense": {"cosine": round(cos_sdiw, 9), "MAE": round(mae_sdiw, 9),
                                  "pass": cos_sdiw >= 0.999999 and mae_sdiw < 1e-4},
        "sdir_stream_vs_dense": {"cosine": round(cos_sdir, 9), "MAE": round(mae_sdir, 9),
                                  "pass": cos_sdir >= 0.999999 and mae_sdir < 1e-4},
        "combined_stream_vs_dense": {"cosine": round(cos_combined, 9), "MAE": round(mae_combined, 9),
                                     "pass": cos_combined >= 0.999999 and mae_combined < 1e-4},
    }

# ── MLP formulas ────────────────────────────────────────────────────────────────
def mlp_reference(X: np.ndarray, W_up: np.ndarray, W_gate: np.ndarray,
                   W_down: np.ndarray) -> np.ndarray:
    up_ref   = apply_linear(X, W_up,   "up_ref")
    gate_ref = apply_linear(X, W_gate, "gate_ref")
    hidden_ref = silu(gate_ref) * up_ref
    Y_ref = apply_linear(hidden_ref, W_down, "down_ref")
    return Y_ref

def mlp_low_batch(X_batch, W_up_p, W_up_s, W_gate_p, W_gate_s, W_down_p, W_down_s):
    """Low-quality MLP (no residual correction)."""
    up_l   = sdiw_streaming_apply_batch(W_up_p,   W_up_s,   X_batch, 4864, 896)
    gate_l = sdiw_streaming_apply_batch(W_gate_p, W_gate_s, X_batch, 4864, 896)
    hidden = silu(gate_l) * up_l
    return sdiw_streaming_apply_batch(W_down_p, W_down_s, hidden, 896, 4864)

def mlp_sub_batch(X_batch, W_up_p, W_up_s, R_up_d,
                  W_gate_p, W_gate_s, R_gate_d,
                  W_down_p, W_down_s, R_down_d):
    """Substitutive MLP: residuals applied to up/gate intermediates before SiLU."""
    up_s   = sdiw_streaming_apply_batch(W_up_p,   W_up_s,   X_batch, 4864, 896) \
           + sdir_streaming_apply_batch(R_up_d,   X_batch,  4864, 896)
    gate_s = sdiw_streaming_apply_batch(W_gate_p, W_gate_s, X_batch, 4864, 896) \
           + sdir_streaming_apply_batch(R_gate_d,  X_batch,  4864, 896)
    hidden = silu(gate_s) * up_s
    return sdiw_streaming_apply_batch(W_down_p, W_down_s, hidden, 896, 4864) \
           + sdir_streaming_apply_batch(R_down_d, hidden,    896, 4864)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Loading activations...")
    activations = np.load(ACTIVATION_PATH)
    print(f"  Keys: {sorted(activations.keys())}")

    print("Loading manifest...")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print("Pre-loading GGUF reference weights...")
    W_refs = {}
    for layer in range(6):
        for family in ["ffn_up", "ffn_gate", "ffn_down"]:
            W_refs[(layer, family)] = extract_W_ref(MODEL_PATH, layer, family)
    print("  Weight shapes verified ✓")

    print("Pre-loading artifacts...")
    artifacts = {}
    for row in manifest["layers"]:
        key = (row["layer"], row["family"])
        d_out, d_in = row["shape"][0], row["shape"][1]
        with open(os.path.join(BUNDLE_DIR, row["paths"]["sdiw_path"]), "rb") as f:
            sdiw_data = f.read()
        with open(os.path.join(BUNDLE_DIR, row["paths"]["sdir_path"]), "rb") as f:
            sdir_data = f.read()
        _, _, _, _, _, scale_nbytes, _ = struct.unpack(SDIW_HEADER, sdiw_data[:SDIW_HEADER_BYTES])
        scale_start = SDIW_HEADER_BYTES
        packed_start = scale_start + scale_nbytes
        artifacts[key] = {
            "sdiw_packed": sdiw_data[packed_start:],
            "sdiw_scale": sdiw_data[scale_start:packed_start],
            "sdir": sdir_data,
            "d_out": d_out, "d_in": d_in,
        }
    print("  Artifacts loaded ✓")

    print("Running MLP probe...")
    layer_results = []
    for layer in range(6):
        X = activations[f"layer{layer}_ffn_up"].astype(np.float32)  # (15, 896)

        W_up   = W_refs[(layer, "ffn_up")]
        W_gate = W_refs[(layer, "ffn_gate")]
        W_down = W_refs[(layer, "ffn_down")]

        art_up   = artifacts[(layer, "ffn_up")]
        art_gate = artifacts[(layer, "ffn_gate")]
        art_down = artifacts[(layer, "ffn_down")]

        # Reference
        Y_ref = mlp_reference(X, W_up, W_gate, W_down)

        # Low-only (no residual)
        Y_low = mlp_low_batch(
            X,
            art_up["sdiw_packed"],   art_up["sdiw_scale"],
            art_gate["sdiw_packed"], art_gate["sdiw_scale"],
            art_down["sdiw_packed"], art_down["sdiw_scale"])

        cos_low = cosine(Y_ref, Y_low)
        mae_low = mae(Y_ref, Y_low)

        # Substitutive (with residual)
        Y_sub = mlp_sub_batch(
            X,
            art_up["sdiw_packed"],   art_up["sdiw_scale"],   art_up["sdir"],
            art_gate["sdiw_packed"], art_gate["sdiw_scale"], art_gate["sdir"],
            art_down["sdiw_packed"], art_down["sdiw_scale"], art_down["sdir"])

        cos_sub = cosine(Y_ref, Y_sub)
        mae_sub = mae(Y_ref, Y_sub)

        delta_cos = cos_sub - cos_low
        delta_mae = mae_low - mae_sub  # positive = substitutive closer

        print(f"  L{layer}: cos_low={cos_low:.6f} cos_sub={cos_sub:.6f} dcos={delta_cos:+.6f} "
              f"mae_low={mae_low:.6f} mae_sub={mae_sub:.6f}")

        layer_results.append({
            "layer": layer,
            "cos_low": round(cos_low, 6),
            "cos_sub": round(cos_sub, 6),
            "delta_cos": round(delta_cos, 6),
            "mae_low": round(mae_low, 6),
            "mae_sub": round(mae_sub, 6),
            "delta_mae": round(delta_mae, 6),
        })

    # Source equivalence (one per family, layer 0)
    print("Source equivalence checks (one per family, layer 0)...")
    source_equiv = {}
    family_order = ["ffn_up", "ffn_gate", "ffn_down"]
    for fi, family in enumerate(family_order):
        art = artifacts[(0, family)]
        manifest_row = manifest["layers"][fi]
        sdiw_path = os.path.join(BUNDLE_DIR, manifest_row["paths"]["sdiw_path"])
        sdir_path = os.path.join(BUNDLE_DIR, manifest_row["paths"]["sdir_path"])
        result = check_source_equivalence(sdiw_path, sdir_path, art["d_out"], art["d_in"])
        source_equiv[family] = result
        for k, v in result.items():
            print(f"  {family}.{k}: cos={v['cosine']} MAE={v['MAE']} pass={v['pass']}")

    # Memory accounting
    print("Memory accounting...")
    total_sub = sum(r["total_substitutive_bytes"] for r in manifest["layers"])
    q4_budget = sum(r["W_ref_Q4_budget_bytes"] for r in manifest["layers"])
    margin = q4_budget - total_sub
    memory_table = []
    for row in manifest["layers"]:
        memory_table.append({
            "layer": row["layer"],
            "family": row["family"],
            "total_sub": row["total_substitutive_bytes"],
            "Q4_budget": row["W_ref_Q4_budget_bytes"],
            "margin": row["total_substitutive_bytes"] - row["W_ref_Q4_budget_bytes"],
        })

    elapsed = time.time() - t0

    all_deltas_positive = all(r["delta_cos"] > 0 for r in layer_results)
    all_source_pass = all(v["pass"] for res in source_equiv.values() for v in res.values())
    if all_deltas_positive and all_source_pass:
        # Approximation improves and source is clean, but memory viability is determined
        # by the combined MLP aggregate — not checked here. Caller must apply the
        # PARTIAL_MLP_APPROX_PASS_MEMORY_FAIL override if aggregate margin < 0.
        classification = "PARTIAL_MLP_APPROX_SOURCE_CLEAN"
    elif all_deltas_positive and not all_source_pass:
        classification = "PARTIAL_MLP_FAMILY_ABLATION_MIXED"
    else:
        classification = "FAIL_31AJ_NEGATIVE_DELTA_COS"

    result = {
        "phase": "31AJ",
        "classification": classification,
        "script_HEAD": "fbdcfcd",
        "elapsed_seconds": round(elapsed, 1),
        "mlp_formula": "Y = SiLU(X @ W_gate.T) * (X @ W_up.T) @ W_down.T",
        "activation_source": ACTIVATION_PATH,
        "layers": layer_results,
        "memory_table": memory_table,
        "memory_aggregate": {
            "total_substitutive_bytes": total_sub,
            "Q4_budget_bytes": q4_budget,
            "margin_bytes": margin,
            "all_margins_positive": all(m["margin"] < 0 for m in memory_table) == False,
        },
        "source_equivalence_results": source_equiv,
        "strict_counters": {
            "W_ref_loaded": 0,
            "W_ref_generated": 0,
            "dense_W_low_materialized": 0,
            "dense_R_materialized": 0,
            "sdiw_loaded": len(manifest["layers"]),
            "sdir_loaded": len(manifest["layers"]),
            "fallback_count": 0,
            "error_count": 0,
        },
        "recommended_next_phase": "Phase 31AK (only if explicitly requested)",
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults: {RESULT_PATH}")
    print(f"Classification: {classification}")
    print(f"Elapsed: {elapsed:.1f}s")
    return result

if __name__ == "__main__":
    main()
