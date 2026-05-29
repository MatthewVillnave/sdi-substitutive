#!/usr/bin/env python3
"""
residual_probe.py — Core harness: X @ W_ref vs X @ W_low + X @ R_compressed.
Tests all residual representations with seeded random X.
"""

import sys
import os
import argparse
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import memory_math as mm


COSINE_OK = 0.99


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def make_seeded_X(rows: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    X = rng.randn(rows, dim).astype(np.float32) * 0.1
    X = np.abs(X) + 0.01
    X = X / (np.abs(X).max() + 1e-8)
    return X


def topk_sparse(R_flat: np.ndarray, k_fraction: float, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    n_keep = max(1, int(len(R_flat) * k_fraction))
    magnitude_scores = np.abs(R_flat) + rng.randn(len(R_flat)) * 1e-6
    indices = np.argpartition(magnitude_scores, -n_keep)[-n_keep:]
    R_sparse = np.zeros_like(R_flat)
    R_sparse[indices] = R_flat[indices]
    return R_sparse, indices


def magnitude_sparse(R_flat: np.ndarray, threshold_fraction: float, seed: int = 42) -> tuple:
    threshold = np.abs(R_flat).max() * threshold_fraction
    mask = np.abs(R_flat) > threshold
    R_sparse = np.zeros_like(R_flat)
    R_sparse[mask] = R_flat[mask]
    return R_sparse, np.where(mask)[0]


def compute_W_low_from_W_ref(W_ref: np.ndarray) -> np.ndarray:
    BLOCK_SIZE = 256
    rows, cols = W_ref.shape
    W_low = np.zeros_like(W_ref)
    for r_start in range(rows):
        for c_start in range(0, cols, BLOCK_SIZE):
            block = W_ref[r_start, c_start:c_start + BLOCK_SIZE]
            if block.size == 0:
                continue
            scale = np.abs(block).max()
            if scale > 0:
                block_q = np.round(block / scale)
                W_low[r_start, c_start:c_start + BLOCK_SIZE] = block_q * scale
            else:
                W_low[r_start, c_start:c_start + BLOCK_SIZE] = block
    return W_low


def main():
    parser = argparse.ArgumentParser(description="Run residual economics harness")
    parser.add_argument("--w-ref", default=None,
                        help="Path to W_ref .npy (Q4 extracted). Falls back to synthetic.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default=os.path.dirname(__file__) + "/../results",
                        help="Output directory")
    parser.add_argument("--X-rows", type=int, default=32, help="X rows for matmul test")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load or create W_ref
    if args.w_ref and os.path.exists(args.w_ref):
        W_ref = np.load(args.w_ref)
        source = "loaded"
    else:
        # Synthetic for fallback
        np.random.seed(args.seed)
        W_ref = np.random.randn(896, 896).astype(np.float32) * 0.1
        # Give it structure (rank-limited product)
        W_ref = W_ref @ W_ref.T / 100
        W_ref = W_ref + np.random.randn(896, 896).astype(np.float32) * 0.001
        source = "synthetic"

    rows, cols = W_ref.shape
    N = rows * cols
    print(f"W_ref: shape={W_ref.shape}, source={source}, N={N}")
    print(f"W_ref F32 bytes: {W_ref.nbytes}")

    # Actual Q4 GGUF bytes for attn_out layer 0
    W_q4_gguf_bytes = 551936
    # Q2: 2 bits/weight
    W_q2_bytes = int(N * 0.25)
    W_q3_bytes = int(N * 0.375)

    print(f"Q4 GGUF bytes (actual): {W_q4_gguf_bytes}")
    print(f"Q2 (2 b/w) bytes: {W_q2_bytes}")
    print(f"Q3 (3 b/w) bytes: {W_q3_bytes}")
    print()

    # Compute W_low (Q2 approx)
    print("Computing W_low (Q2 approximation)...")
    W_low = compute_W_low_from_W_ref(W_ref)
    R = W_ref - W_low

    print(f"R (residual): mean={R.mean():.6f}, std={R.std():.6f}, "
          f"abs_mean={np.abs(R).mean():.6f}, max={R.max():.6f}")
    print()

    # Test inputs
    print("Generating seeded test inputs X...")
    X = make_seeded_X(args.X_rows, cols, seed=args.seed)

    # Reference outputs
    Y_ref = X @ W_ref
    Y_low = X @ W_low

    results = []

    # ============ DENSE REPS ============
    print("Testing dense INT8 residual...")
    scale8 = np.abs(R).max() / 127.0
    R_int8 = np.round(R / scale8) * scale8
    Y_sub = Y_low + X @ R_int8
    comp_bytes = R.nbytes + 4
    ebpw = comp_bytes * 8 / N
    r = {
        "name": "dense_int8", "type": "dense_int8",
        "compressed_bytes": comp_bytes,
        "effective_bits_per_weight": round(ebpw, 4),
        "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
        "memory_delta_vs_Q4": W_q2_bytes + comp_bytes - W_q4_gguf_bytes,
        "memory_viable_q2": False,
        "memory_viable_q3": False,
        "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
        "MAE": round(mae(Y_ref, Y_sub), 6),
        "max_error": round(max_error(Y_ref, Y_sub), 6),
    }
    mem = mm.memory_viable_gate(W_q2_bytes, comp_bytes, W_q4_gguf_bytes, "q2")
    r["memory_viable_q2"] = mem["viable"]
    memq3 = mm.memory_viable_gate(W_q3_bytes, comp_bytes, W_q4_gguf_bytes, "q3")
    r["memory_viable_q3"] = memq3["viable"]
    results.append(r)

    print("Testing dense INT6 residual...")
    scale6 = np.abs(R).max() / 31.0
    R_int6 = np.round(R / scale6) * scale6
    Y_sub = Y_low + X @ R_int6
    comp_bytes = int(N * 0.75) + 4
    ebpw = comp_bytes * 8 / N
    r = {
        "name": "dense_int6", "type": "dense_int6",
        "compressed_bytes": comp_bytes,
        "effective_bits_per_weight": round(ebpw, 4),
        "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
        "memory_delta_vs_Q4": W_q2_bytes + comp_bytes - W_q4_gguf_bytes,
        "memory_viable_q2": False,
        "memory_viable_q3": False,
        "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
        "MAE": round(mae(Y_ref, Y_sub), 6),
        "max_error": round(max_error(Y_ref, Y_sub), 6),
    }
    mem = mm.memory_viable_gate(W_q2_bytes, comp_bytes, W_q4_gguf_bytes, "q2")
    r["memory_viable_q2"] = mem["viable"]
    memq3 = mm.memory_viable_gate(W_q3_bytes, comp_bytes, W_q4_gguf_bytes, "q3")
    r["memory_viable_q3"] = memq3["viable"]
    results.append(r)

    print("Testing dense INT4 residual...")
    scale4 = np.abs(R).max() / 7.0
    R_int4 = np.round(R / scale4) * scale4
    Y_sub = Y_low + X @ R_int4
    comp_bytes = int(N * 0.5) + 4
    ebpw = comp_bytes * 8 / N
    r = {
        "name": "dense_int4", "type": "dense_int4",
        "compressed_bytes": comp_bytes,
        "effective_bits_per_weight": round(ebpw, 4),
        "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
        "memory_delta_vs_Q4": W_q2_bytes + comp_bytes - W_q4_gguf_bytes,
        "memory_viable_q2": False,
        "memory_viable_q3": False,
        "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
        "MAE": round(mae(Y_ref, Y_sub), 6),
        "max_error": round(max_error(Y_ref, Y_sub), 6),
    }
    mem = mm.memory_viable_gate(W_q2_bytes, comp_bytes, W_q4_gguf_bytes, "q2")
    r["memory_viable_q2"] = mem["viable"]
    memq3 = mm.memory_viable_gate(W_q3_bytes, comp_bytes, W_q4_gguf_bytes, "q3")
    r["memory_viable_q3"] = memq3["viable"]
    results.append(r)

    print("Testing dense INT2 residual...")
    scale2 = np.abs(R).max()
    R_int2 = np.clip(np.round(R / scale2), -1, 1) * scale2
    Y_sub = Y_low + X @ R_int2
    comp_bytes = int(N * 0.25) + 4
    ebpw = comp_bytes * 8 / N
    r = {
        "name": "dense_int2", "type": "dense_int2",
        "compressed_bytes": comp_bytes,
        "effective_bits_per_weight": round(ebpw, 4),
        "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
        "memory_delta_vs_Q4": W_q2_bytes + comp_bytes - W_q4_gguf_bytes,
        "memory_viable_q2": False,
        "memory_viable_q3": False,
        "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
        "MAE": round(mae(Y_ref, Y_sub), 6),
        "max_error": round(max_error(Y_ref, Y_sub), 6),
    }
    mem = mm.memory_viable_gate(W_q2_bytes, comp_bytes, W_q4_gguf_bytes, "q2")
    r["memory_viable_q2"] = mem["viable"]
    memq3 = mm.memory_viable_gate(W_q3_bytes, comp_bytes, W_q4_gguf_bytes, "q3")
    r["memory_viable_q3"] = memq3["viable"]
    results.append(r)

    print("Testing dense ternary residual...")
    threshold = np.abs(R).mean()
    R_tern = np.where(R > threshold, threshold,
                      np.where(R < -threshold, -threshold, 0.0))
    Y_sub = Y_low + X @ R_tern
    comp_bytes = int(N * 0.25) + 4
    ebpw = comp_bytes * 8 / N
    r = {
        "name": "dense_ternary", "type": "dense_ternary",
        "compressed_bytes": comp_bytes,
        "effective_bits_per_weight": round(ebpw, 4),
        "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
        "memory_delta_vs_Q4": W_q2_bytes + comp_bytes - W_q4_gguf_bytes,
        "memory_viable_q2": False,
        "memory_viable_q3": False,
        "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
        "MAE": round(mae(Y_ref, Y_sub), 6),
        "max_error": round(max_error(Y_ref, Y_sub), 6),
    }
    mem = mm.memory_viable_gate(W_q2_bytes, comp_bytes, W_q4_gguf_bytes, "q2")
    r["memory_viable_q2"] = mem["viable"]
    memq3 = mm.memory_viable_gate(W_q3_bytes, comp_bytes, W_q4_gguf_bytes, "q3")
    r["memory_viable_q3"] = memq3["viable"]
    results.append(r)

    print("Testing top-k sparse residuals...")
    for k_frac in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        R_sparse_flat, indices = topk_sparse(R.flatten(), k_frac, seed=args.seed + int(k_frac * 100))
        R_sparse = R_sparse_flat.reshape(W_ref.shape)
        Y_sub = Y_low + X @ R_sparse
        n_kept = len(indices)
        comp_bytes = n_kept * 4 + n_kept * 4  # float32 values + int32 indices
        ebpw = comp_bytes * 8 / N
        r = {
            "name": f"topk_sparse_{int(k_frac*100)}",
            "type": f"topk_sparse_k{int(k_frac*100)}",
            "compressed_bytes": comp_bytes,
            "effective_bits_per_weight": round(ebpw, 4),
            "n_kept": n_kept, "k_fraction": k_frac,
            "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
            "memory_delta_vs_Q4": W_q2_bytes + comp_bytes - W_q4_gguf_bytes,
            "memory_viable_q2": False,
            "memory_viable_q3": False,
            "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
            "MAE": round(mae(Y_ref, Y_sub), 6),
            "max_error": round(max_error(Y_ref, Y_sub), 6),
        }
        mem = mm.memory_viable_gate(W_q2_bytes, comp_bytes, W_q4_gguf_bytes, "q2")
        r["memory_viable_q2"] = mem["viable"]
        memq3 = mm.memory_viable_gate(W_q3_bytes, comp_bytes, W_q4_gguf_bytes, "q3")
        r["memory_viable_q3"] = memq3["viable"]
        results.append(r)

    print("Testing magnitude-threshold sparse residuals...")
    for t_frac in [0.01, 0.05, 0.10, 0.20]:
        R_sparse_flat, indices = magnitude_sparse(R.flatten(), t_frac, seed=args.seed)
        R_sparse = R_sparse_flat.reshape(W_ref.shape)
        Y_sub = Y_low + X @ R_sparse
        n_kept = len(indices)
        comp_bytes = n_kept * 4 + n_kept * 4
        ebpw = comp_bytes * 8 / N
        r = {
            "name": f"magnitude_sparse_t{int(t_frac*100)}",
            "type": f"magnitude_sparse_t{int(t_frac*100)}",
            "compressed_bytes": comp_bytes,
            "effective_bits_per_weight": round(ebpw, 4),
            "n_kept": n_kept, "threshold_fraction": t_frac,
            "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
            "memory_delta_vs_Q4": W_q2_bytes + comp_bytes - W_q4_gguf_bytes,
            "memory_viable_q2": False,
            "memory_viable_q3": False,
            "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
            "MAE": round(mae(Y_ref, Y_sub), 6),
            "max_error": round(max_error(Y_ref, Y_sub), 6),
        }
        mem = mm.memory_viable_gate(W_q2_bytes, comp_bytes, W_q4_gguf_bytes, "q2")
        r["memory_viable_q2"] = mem["viable"]
        memq3 = mm.memory_viable_gate(W_q3_bytes, comp_bytes, W_q4_gguf_bytes, "q3")
        r["memory_viable_q3"] = memq3["viable"]
        results.append(r)

    print("Testing low-rank SVD residuals...")
    for rank in [1, 2, 4, 8, 16, 32, 64]:
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        Ur = U[:, :rank]
        Sr = S[:rank]
        Vtr = Vt[:rank, :]
        R_lr = Ur @ np.diag(Sr) @ Vtr
        Y_sub = Y_low + X @ R_lr
        lr_bytes = Ur.nbytes + Sr.nbytes + Vtr.nbytes
        ebpw = lr_bytes * 8 / N
        r = {
            "name": f"lowrank_r{rank}",
            "type": f"lowrank_r{rank}",
            "compressed_bytes": lr_bytes,
            "effective_bits_per_weight": round(ebpw, 4),
            "rank": rank,
            "W_low_bytes": W_q2_bytes, "W_q4_bytes": W_q4_gguf_bytes,
            "memory_delta_vs_Q4": W_q2_bytes + lr_bytes - W_q4_gguf_bytes,
            "memory_viable_q2": False,
            "memory_viable_q3": False,
            "cosine_Y_ref_Y_sub": round(cosine_similarity(Y_ref, Y_sub), 6),
            "MAE": round(mae(Y_ref, Y_sub), 6),
            "max_error": round(max_error(Y_ref, Y_sub), 6),
        }
        mem = mm.memory_viable_gate(W_q2_bytes, lr_bytes, W_q4_gguf_bytes, "q2")
        r["memory_viable_q2"] = mem["viable"]
        memq3 = mm.memory_viable_gate(W_q3_bytes, lr_bytes, W_q4_gguf_bytes, "q3")
        r["memory_viable_q3"] = memq3["viable"]
        results.append(r)

    # ============ CLASSIFICATION ============
    memory_viable_q2 = [x for x in results if x["memory_viable_q2"]]
    memory_viable_q3 = [x for x in results if x["memory_viable_q3"]]
    cosine_ok = [x for x in results if x["cosine_Y_ref_Y_sub"] >= COSINE_OK]
    memory_viable_and_cosine_ok_q2 = [x for x in memory_viable_q2 if x["cosine_Y_ref_Y_sub"] >= COSINE_OK]
    memory_viable_and_cosine_ok_q3 = [x for x in memory_viable_q3 if x["cosine_Y_ref_Y_sub"] >= COSINE_OK]

    if memory_viable_and_cosine_ok_q2 or memory_viable_and_cosine_ok_q3:
        classification = "PASS_MEMORY_VIABLE_RESIDUAL_FOUND"
    elif memory_viable_q2 or memory_viable_q3:
        classification = "PARTIAL_MEMORY_WORKS_ACCURACY_POOR"
    elif cosine_ok:
        classification = "PARTIAL_MATH_WORKS_MEMORY_FAILS"
    else:
        classification = "BLOCKED_NUMERICAL_ISSUE"

    # ============ BUILD RESULT ============
    result = {
        "classification": classification,
        "tensor_extracted": source == "loaded",
        "W_ref_source": source,
        "W_ref_shape": list(W_ref.shape),
        "W_ref_F32_bytes": W_ref.nbytes,
        "N_weights": N,
        "W_q4_gguf_bytes": W_q4_gguf_bytes,
        "W_q2_bytes": W_q2_bytes,
        "W_q3_bytes": W_q3_bytes,
        "random_seed": args.seed,
        "X_shape": list(X.shape),
        "residual_R": {
            "mean": round(float(R.mean()), 6),
            "std": round(float(R.std()), 6),
            "abs_mean": round(float(np.abs(R).mean()), 6),
            "max": round(float(R.max()), 6),
            "min": round(float(R.min()), 6),
        },
        "reference_Y": {
            "cosine_Y_low_vs_Y_ref": round(cosine_similarity(Y_ref, Y_low), 6),
            "cosine_threshold": COSINE_OK,
        },
        "representations": results,
        "summary": {
            "total_representations_tested": len(results),
            "memory_viable_q2": len(memory_viable_q2),
            "memory_viable_q3": len(memory_viable_q3),
            "cosine_ok": len(cosine_ok),
            "memory_viable_and_cosine_ok_q2": len(memory_viable_and_cosine_ok_q2),
            "memory_viable_and_cosine_ok_q3": len(memory_viable_and_cosine_ok_q3),
            "best_cosine": max(x["cosine_Y_ref_Y_sub"] for x in results) if results else 0,
            "best_memory_saver": min((x["memory_delta_vs_Q4"] for x in results), default=0),
        },
        "key_findings": {
            "dense_int8_loses_memory": True,
            "residual_bits_needed_to_beat_Q4": "must be < 2 b/w for Q2+residual to beat Q4",
            "sparse_lowrank_can_beat_Q4": len(memory_viable_q2) > 0 or len(memory_viable_q3) > 0,
            "memory_viable_and_accurate": len(memory_viable_and_cosine_ok_q2) > 0 or len(memory_viable_and_cosine_ok_q3) > 0,
        },
        "recommendation": (
            "Move to ffn_up/ffn_down" if (memory_viable_and_cosine_ok_q2 or memory_viable_and_cosine_ok_q3)
            else "Phase 31B economics gate FAILED: no viable residual found"
        ),
        "memory_viable_q2_reps": [x["name"] for x in memory_viable_q2],
        "memory_viable_q3_reps": [x["name"] for x in memory_viable_q3],
        "cosine_ok_reps": [x["name"] for x in cosine_ok],
        "memory_viable_and_cosine_ok_q2_names": [x["name"] for x in memory_viable_and_cosine_ok_q2],
        "memory_viable_and_cosine_ok_q3_names": [x["name"] for x in memory_viable_and_cosine_ok_q3],
    }

    # Write main result
    result_path = os.path.join(out_dir, "PHASE31B_RESIDUAL_ECONOMICS.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {result_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("MEMORY TABLE")
    print(f"{'Name':<30} {'CompBytes':>10} {'ebpw':>6} {'Delta vs Q4':>12} {'MemViableQ2':>12} {'Cosine':>8}")
    print("-" * 90)
    # Baseline
    print(f"{'W_ref (F32)':.<30} {W_ref.nbytes:>10} {32.0:>6.1f} {'(reference)':>12}")
    print(f"{'Q4 GGUF (actual)':.<30} {W_q4_gguf_bytes:>10} {W_q4_gguf_bytes*8/N:>6.2f} {0:>12}")
    print(f"{'Q2 (2 b/w baseline)':.<30} {W_q2_bytes:>10} {2.0:>6.1f} {W_q2_bytes - W_q4_gguf_bytes:>12}")
    print("-" * 90)
    for row in results:
        mv_q2 = "Y" if row["memory_viable_q2"] else ""
        print(f"{row['name']:<30} {row['compressed_bytes']:>10} {row['effective_bits_per_weight']:>6.3f} "
              f"{row['memory_delta_vs_Q4']:>12} {mv_q2:>12} {row['cosine_Y_ref_Y_sub']:>8.4f}")

    print("\n" + "=" * 90)
    print("APPROXIMATION TABLE")
    print(f"{'Name':<30} {'Cosine':>10} {'MAE':>12} {'MaxErr':>12}")
    print("-" * 90)
    for row in results:
        print(f"{row['name']:<30} {row['cosine_Y_ref_Y_sub']:>10.6f} {row['MAE']:>12.6f} {row['max_error']:>12.6f}")

    print("\n" + "=" * 90)
    print(f"Classification: {classification}")
    print(f"Memory-viable (Q2): {len(memory_viable_q2)} / {len(results)}")
    print(f"Memory-viable (Q3): {len(memory_viable_q3)} / {len(results)}")
    print(f"Cosine ≥ {COSINE_OK}: {len(cosine_ok)} / {len(results)}")
    print(f"Viable AND Accurate (Q2): {len(memory_viable_and_cosine_ok_q2)}")
    print(f"Viable AND Accurate (Q3): {len(memory_viable_and_cosine_ok_q3)}")
    print(f"\nRecommendation: {result['recommendation']}")
    print("=" * 90)

    return result


if __name__ == "__main__":
    main()
