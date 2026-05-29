# Phase 31I: Robustness Sweep — More Layers + More Prompts

## Objective
Validate the compressed residual policy (dense bitmap + global top-7.5% + fp16 + streaming sparse apply) across all 6 layers, both FFN tensor families, and all 15 prompts (180 total combinations).

## Policy
- **Encoding:** dense bitmap + global top-7.5% elements + fp16 values
- **Format magic:** `RSC\x00`, version 1
- **k_pct:** 7.5% (global top-k over entire residual tensor)
- **Compute:** streaming sparse matmul (no dense R materialization)
- **Memory budget:** Q4 bytes − Q2 bytes = residual budget per tensor

## Test Configuration

| Dimension | Value |
|-----------|-------|
| Layers | 0, 1, 2, 3, 4, 5 |
| Tensors | ffn_up (4864×896), ffn_down (896×4864) |
| Prompts | 15 (5 original + 10 new) |
| Total combos | 180 |
| Activation cache | `PHASE31I_activations.npz` (15 prompts × 12 tensors) |

## Classification

**PASS_ROBUSTNESS_SWEEP**

- Regressions: **0 / 180**
- Memory viable: **180 / 180**
- Mean Δcosine: **+0.0633**
- Min Δcosine: **+0.000197** (all positive)

---

## Per-Layer Summary

| Layer | cos_low | cos_sub | ΔCos | minΔ | MAE_low | MAE_sub | Regressions | Memory Viable |
|-------|---------|---------|------|------|---------|---------|-------------|---------------|
| 0 | 0.8036 | 0.8739 | +0.0703 | +0.0447 | 0.1085 | 0.0904 | 0 | 100.0% |
| 1 | 0.7715 | 0.8388 | +0.0673 | +0.0452 | 0.1480 | 0.1259 | 0 | 100.0% |
| 2 | 0.7691 | 0.8360 | +0.0669 | +0.0002 | 1.2197 | 1.0323 | 0 | 100.0% |
| 3 | 0.7930 | 0.8399 | +0.0469 | +0.0002 | 0.7448 | 0.6510 | 0 | 100.0% |
| 4 | 0.7441 | 0.8158 | +0.0718 | +0.0312 | 0.3115 | 0.2668 | 0 | 100.0% |
| 5 | 0.7803 | 0.8367 | +0.0564 | +0.0030 | 0.4865 | 0.4218 | 0 | 100.0% |

**Notes:**
- Layer 2 shows elevated MAE (1.22 vs 0.11–0.49 for other layers) but the compressed residual policy still improves over W_low with positive Δcosine and reduced MAE
- Layer 4 shows the strongest delta improvement (+0.0718), making it the highest-quality layer for residual enhancement
- All layers show meaningful cosine improvement over W_low

---

## Per-Family Summary

| Family | cos_low | cos_sub | ΔCos | MAE_low | MAE_sub | Regressions | Memory Viable |
|--------|---------|---------|------|---------|---------|-------------|---------------|
| ffn_up | 0.7193 | 0.7854 | +0.0661 | 0.8061 | 0.7036 | 0 | 100.0% |
| ffn_down | 0.8345 | 0.8950 | +0.0604 | 0.2003 | 0.1591 | 0 | 100.0% |

**Notes:**
- ffn_down has higher baseline cosine (0.8345 vs 0.7193) — the down projection is a stronger signal that is easier to approximate; the residual still adds value
- ffn_up has larger MAE (0.8061 vs 0.2003) — the up projection sees higher-magnitude intermediate activations; residual compression is more impactful here
- Both families improve uniformly with compressed residuals; no regressions

---

## Per-Prompt Summary

| # | Prompt | ΔCos | Reg | Viable% |
|---|--------|------|-----|---------|
| 0 | Hi | +0.0507 | 0 | 100.0% |
| 1 | The capital of France is | +0.0604 | 0 | 100.0% |
| 2 | 2+2= | +0.0615 | 0 | 100.0% |
| 3 | def add(a, b): | +0.0626 | 0 | 100.0% |
| 4 | Once upon a time | +0.0761 | 0 | 100.0% |
| 5 | What is the largest planet? | +0.0609 | 0 | 100.0% |
| 6 | x = 5 * 3 | +0.0640 | 0 | 100.0% |
| 7 | class MyClass: | +0.0664 | 0 | 100.0% |
| 8 | It was a dark and stormy night | +0.0728 | 0 | 100.0% |
| 9 | {"name": "John", "age": | +0.0642 | 0 | 100.0% |
| 10 | Sorry, I can't help with that. | +0.0613 | 0 | 100.0% |
| 11 | apple, banana, cherry, | +0.0671 | 0 | 100.0% |
| 12 | The reason for this is | +0.0660 | 0 | 100.0% |
| 13 | Hey there! | +0.0633 | 0 | 100.0% |
| 14 | 🦆 | +0.0515 | 0 | 100.0% |

**Notes:**
- Short prompts (Hi, 🦆) show lower but still positive delta — the residual adds less signal for minimal inputs
- Narrative/code prompts (Once upon a time, class MyClass:) show highest deltas — longer sequences benefit most from compressed residual correction
- No regressions across any prompt type, including JSON fragment and special characters

---

## Memory Accounting (selected tensors)

| Tensor | Q4 bytes | Q2 bytes | Encoded bytes | Budget | Margin | Viable |
|--------|----------|----------|---------------|--------|--------|--------|
| blk.0.ffn_up | 2,419,968 | 1,058,048 | 1,198,520 | 1,361,920 | +163,400 | ✅ |
| blk.0.ffn_down | 2,419,968 | 1,058,048 | 1,198,520 | 1,361,920 | +163,400 | ✅ |
| blk.3.ffn_up | 2,419,968 | 1,058,048 | 1,198,520 | 1,361,920 | +163,400 | ✅ |
| blk.5.ffn_down | 2,419,968 | 1,058,048 | 1,198,520 | 1,361,920 | +163,400 | ✅ |

All 12 tensors are memory-viable with consistent margin of ~163,400 bytes (~12% headroom).

---

## Worst-Case / Best-Case

| Case | Layer | Tensor | Prompt | ΔCos |
|------|-------|--------|--------|------|
| **Worst** | 2 | ffn_down | Hi | +0.000197 |
| **Best** | 0 | ffn_down | Once upon a time | +0.143611 |

Even the worst case is **positive** (zero regressions confirmed).

---

## Key Findings

1. **Zero regressions across 180 combinations** — the compressed residual policy improves output cosine over W_low in every single test case
2. **Memory viable for all 12 tensors** — residual encoding fits within the Q4→Q2 budget headroom across every layer
3. **Narrative/code prompts benefit most** — prompts with richer activation patterns (longer sequences, code, storytelling) show the highest delta improvements
4. **Short inputs still benefit** — even "Hi" and 🦆 show positive delta, confirming the residual correction is not amplitude-dependent
5. **Layer 2 elevated MAE is non-blocking** — Layer 2's high MAE (1.22) reflects the specific activation distribution at that layer; the compressed residual still measurably improves cosine similarity despite the noise floor

---

## Classification Rationale

`PASS_ROBUSTNESS_SWEEP` is assigned because:
- `regression_count = 0` — no combination showed degraded cosine similarity
- `memory_viable_count = 180/180` — all combinations fit within memory budget
- `all_correct = True` — streaming sparse compute matches dense reference (Phase 31H established)
- `mean_delta = +0.0633` — clinically and practically meaningful improvement

---

## Phase Progression

| Phase | Focus | Result |
|-------|-------|--------|
| 31A | SDI theory + bitmap encoding | CONCEPTUAL |
| 31B | Qwen2.5-0.5B GGUF weight extraction | COMPLETE |
| 31C | FFN residual analysis | COMPLETE |
| 31D | Encoded residual correctness | COMPLETE |
| 31E | Multi-seed activation capture | COMPLETE |
| 31F | Multi-layer residual sweep | COMPLETE |
| 31G | Sparse k-parameter sweep | COMPLETE |
| 31H | Compressed residual compute (fp16+stream) | COMPLETE |
| **31I** | **Robustness sweep (6 layers × 2 tensors × 15 prompts)** | **PASS** |

---

*Generated: Phase 31I, Robustness Sweep*
*Classification: PASS_ROBUSTNESS_SWEEP*
