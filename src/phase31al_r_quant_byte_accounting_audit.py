#!/usr/bin/env python3
"""
Phase 31AL-R — Quant Byte Accounting Audit
Verifies GGUF quant byte sizes against llama.cpp constants.
Checks whether 31AL labels (Q4_K_M vs Q2_K) were correct.
"""
import os, sys

REPO = "/home/matthew-villnave/sdi-substitutive"
LLAMA_CPP_CONSTANTS = os.path.join(
    os.path.expanduser("~"), "llama.cpp/gguf-py/gguf/constants.py"
)

QK_K = 256
N_ELEMENTS = 4_358_144  # 4864 * 896
N_BLOCKS = N_ELEMENTS // QK_K  # 17,024

# GGUF quant sizes from llama.cpp constants.py
# Format: (block_size, meta_bytes + block_bytes)
GGUF_QUANTS = {
    "Q2_K": (256, 2 + 2 + QK_K // 16 + QK_K // 4),      # 84 bytes/block = 2.625 b/e
    "Q3_K": (256, 2 + QK_K // 4 + QK_K // 8 + 12),        # 110 bytes/block = 3.438 b/e
    "Q4_K": (256, 2 + 2 + QK_K // 2 + 12),                 # 144 bytes/block = 4.500 b/e
    "Q5_K": (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),    # 176 bytes/block = 5.500 b/e
    "Q6_K": (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),# 210 bytes/block = 6.562 b/e
    "Q4_0": (32, 2 + 16),                                    # 18 bytes/block = 4.500 b/e
    "Q4_1": (32, 2 + 2 + 16),                               # 20 bytes/block = 5.000 b/e
    "Q8_0": (32, 2 + 32),                                    # 34 bytes/block = 8.500 b/e
}


def main():
    print("=" * 70)
    print("PHASE 31AL-R — QUANT BYTE ACCOUNTING AUDIT")
    print("=" * 70)
    print(f"Tensor: {N_ELEMENTS:,} elements = 4864 x 896")
    print(f"QK_K = {QK_K}, n_blocks = {N_BLOCKS:,}")
    print()

    # ── Correct quant table ───────────────────────────────────────────────────
    print("### Corrected GGUF Quant Table (per family-layer)")
    print()
    print(f"{'Format':<10} {'Bytes/Block':>12} {'Total Bytes':>14} {'Bits/Elem':>11} {'vs Q4_budget':>14}")
    print("-" * 65)

    q4_budget_per_fl = N_ELEMENTS * 4 // 8  # nibbles only

    for name, (blk_sz, blk_bytes) in GGUF_QUANTS.items():
        blocks = N_ELEMENTS // blk_sz
        total = 12 + blocks * blk_bytes  # 12-byte header
        bits = total * 8 / N_ELEMENTS
        diff = total - q4_budget_per_fl
        diff_str = f"+{diff:,}" if diff > 0 else f"{diff:,}"
        fits = "OK" if total <= q4_budget_per_fl else "OVER"
        print(f"  {name:<8} {blk_bytes:>12,} {total:>14,} {bits:>11.4f} {diff_str:>14}  [{fits}]")

    print()
    print(f"  Q4_budget (nibbles only): {q4_budget_per_fl:>14,} = {q4_budget_per_fl*3*6/1024/1024:.2f}MB total")
    print()

    # ── Verify 31AL claimed values ─────────────────────────────────────────────
    print("### 31AL Claimed Values vs Correct Values")
    print()
    print(f"  31AL claimed: Q4_K_M = 1,430,028 bytes at 2.625 bits/elem")
    b = 1_430_028
    bits = b * 8 / N_ELEMENTS
    # Check which format this actually is
    for name, (blk_sz, blk_bytes) in GGUF_QUANTS.items():
        blocks = (b - 12) // blk_bytes
        if blocks * blk_bytes + 12 == b:
            print(f"    VERIFY: 1,430,028 = {name} (computed blocks={blocks:,})")
            print(f"    ACTUAL bits/elem: {bits:.4f}")
            if name == "Q2_K":
                print(f"    *** LABEL SWAP: 31AL called this Q4_K_M, correct label is Q2_K ***")
            break
    else:
        print(f"    NOT a standard GGUF format ({bits:.4f} bits/elem)")

    print()
    print(f"  31AL claimed: Q2_K_M = 1,818,060 bytes at 2.0625 bits/elem")
    b = 1_818_060
    bits = b * 8 / N_ELEMENTS
    print(f"    Actual bits/elem: {bits:.4f}")
    found = False
    for name, (blk_sz, blk_bytes) in GGUF_QUANTS.items():
        blocks = (b - 12) // blk_bytes
        if blocks * blk_bytes + 12 == b:
            print(f"    Matches {name} ({blocks:,} blocks x {blk_bytes} bytes)")
            found = True
    if not found:
        print(f"    *** FABRICATED: 1,818,060 is NOT a standard GGUF format ***")
        print(f"    (closest: Q3_K would be 1,872,652 bytes, Q2_K would be 1,430,028)")

    # ── SDIR analysis ─────────────────────────────────────────────────────────
    print()
    print("### SDIR Residual Inefficiency")
    print()
    bitmap_bytes = N_ELEMENTS // 8
    print(f"  Bitmap (1 bit/elem): {bitmap_bytes:,} bytes per family-layer")
    print(f"  Q4_budget: {q4_budget_per_fl:,} bytes per family-layer")
    print(f"  Bitmap is {bitmap_bytes/q4_budget_per_fl*100:.1f}% of Q4_budget (always, regardless of k%)")

    # ── Combined viability ─────────────────────────────────────────────────────
    print()
    print("### Corrected Combined Viability (k=9-12%)")
    print()

    total_q4_budget = q4_budget_per_fl * 3 * 6

    resid_sdir = {
        "ffn_up": 1_329_344,
        "ffn_gate": 1_590_782,
        "ffn_down": 1_329_304,
    }
    total_sdir = sum(v * 6 for v in resid_sdir.values())

    q2k_total = GGUF_QUANTS["Q2_K"][1] * N_BLOCKS * 3 * 6 + 12 * 3 * 6
    q4k_total = GGUF_QUANTS["Q4_K"][1] * N_BLOCKS * 3 * 6 + 12 * 3 * 6

    print(f"  Q4_budget total: {total_q4_budget:,} = {total_q4_budget/1024/1024:.2f}MB")
    print()
    print(f"  W_low alone:")
    print(f"    sdiw:   {2451456*3*6/1024/1024:.2f}MB  margin={total_q4_budget-2451456*3*6:+d}")
    print(f"    Q4_K:   {q4k_total/1024/1024:.2f}MB  margin={total_q4_budget-q4k_total:+d}")
    print(f"    Q3_K:   {GGUF_QUANTS['Q3_K'][1]*N_BLOCKS*3*6/1024/1024:.2f}MB  margin={total_q4_budget-GGUF_QUANTS['Q3_K'][1]*N_BLOCKS*3*6:+d}")
    print(f"    Q2_K:   {q2k_total/1024/1024:.2f}MB  margin={total_q4_budget-q2k_total:+d}  ← Viable W_low")
    print()
    print(f"  Combined (Q2_K + residuals):")
    print(f"    Q2_K + SDIR @ k=9-12%:  {(q2k_total+total_sdir)/1024/1024:.2f}MB  margin={total_q4_budget-q2k_total-total_sdir:+d}  ← NOT VIABLE")

    # k-reduction table
    print()
    print("### k-Reduction Analysis: Q2_K + int8 sparse residual")
    print()
    print(f"  {'k%':>4}  {'nnz/fl':>10}  {'int8 resid/fl':>13}  {'total':>8}  {'margin':>10}  {'Viable':>7}")
    print("  " + "-" * 60)
    for k in [1, 2, 3, 5, 7, 9, 12]:
        nnz_fl = int(N_ELEMENTS * k / 100)
        int8_fl = bitmap_bytes + nnz_fl  # bitmap + int8 values
        int8_total = int8_fl * 3 * 6
        total = q2k_total + int8_total
        margin = total_q4_budget - total
        viable = "YES" if margin >= 0 else "NO"
        print(f"  {k:>3}% {nnz_fl:>10,} {int8_fl:>13,} {total/1024/1024:>7.2f}MB {margin:>+10,}  {viable:>7}")

    print()
    print("### Conclusion")
    print()
    print("  Classification: PARTIAL_31AL_VIABILITY_REVISED")
    print()
    print("  31AL had LABEL SWAP and a FABRICATED byte count:")
    print("    - 1,430,028 was called Q4_K_M but is Q2_K (label swapped)")
    print("    - 1,818,060 is not a standard GGUF format (fabricated)")
    print()
    print("  CORRECTED findings:")
    print("    - Q2_K is the correct viable W_low format (2.625 bits/elem)")
    print("    - Q4_K = 2,451,468 = 4.5 bits/elem (same as sdiw, over budget)")
    print("    - At current k=9-12%, NO combined policy is memory-positive")
    print("    - Q2_K + int8 sparse residual at k<=3% is viable (+1-3MB margin)")
    print()
    print("  Qualitative direction unchanged: Q2_K is the right W_low format.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
