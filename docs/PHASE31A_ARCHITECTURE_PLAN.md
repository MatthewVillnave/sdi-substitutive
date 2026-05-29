# Phase 31A — Substitutive Tensor Replacement Architecture Plan

**Date:** 2026-05-29
**Status:** Design only. No code. No repo modification. No benchmarks.

---

## Context: Why a New Repo

The additive sidecar experiment (llama.cpp `experimental/prt-phase19a-alt-sidecar-backed`, tag `SDI_PRT_EXPERIMENTAL_ARCHIVE_CHECKPOINT`) produced a clear negative result:

- Active FFN sidecars added ~+277 MB / +27% regardless of base model
- Q2/Q3 + active sidecars always exceeded Q4 baseline RSS
- The core architecture (additive injection) was the wrong approach

The substitutive direction tests a different thesis: instead of adding a sidecar alongside the resident tensor, replace the resident tensor with a lower-bit version plus a compressed residual that corrects the approximation error.

---

## Core Principle

> Sidecar correction must replace resident cost, not add beside it.

**Old (additive — failed):**
```
W_q4 loaded + sidecar R loaded → additive overhead
```

**New (substitutive — thesis):**
```
W_q4 NOT loaded
W_low loaded + R_sidecar correction
≈ W_q4 behavior at lower resident memory
```

---

## 1. Minimal Proof Target

**Pick: `attn_output` (attention output projection)**

Rationale:
- Smallest shape: 896×896 (Qwen2.5-0.5B layer 0) vs ffn_up at 896×4864
- Phase 30F already showed attn_out is lightweight (+55 MB) — easy to isolate substitutive effect
- ffn_up/ffn_down have better memory economics (larger shape = better compression ratio) but carry more risk
- attn_out has clear semantic role and existing .trit sidecar work from prior phases

**Note:** attn_out is proof-of-mechanics target. ffn_up/ffn_down are economically meaningful targets to test after mechanics are validated.

---

## 2. Replacement Math

**Definition:**
```
W_ref  = full-precision or Q4 reference weight matrix
W_low  = lower-bit approximation (Q2 or Q3)
R      = W_ref - W_low          ← residual

Y_ref  = X @ W_ref              ← ground truth
Y_sub  = X @ W_low + X @ R      ← substitutive computation
```

**Residual storage format:**
- .trit format — same binary format already used for PRT sidecars; packed INT8/INT6 with scale metadata
- Residual stays **compressed** — decoded to F32 only at compute time, not resident
- Residual generated offline: `R = W_ref - W_low` in F32, then quantized to INT8/INT6

**First accuracy metric: cosine similarity**
- Compute on representative activation vectors
- Secondary: per-token logit delta
- Cosine first — dimension-agnostic, fast

**Critical constraint:** For Q2 + residual to beat Q4, residual must average **< 2 bits/weight**. This is the gate Phase 31B tests.

---

## 3. Memory Accounting

**Substitutive path resident set:**
| Component | Size (attn_out layer 0, 896×896) |
|-----------|----------------------------------|
| W_low (Q2) | ~201 KB |
| R_sidecar raw (compressed) | depends on representation |
| Decoded residual cache (F32 temp) | ~3.2 MB transient |
| KV cache | unchanged |

**Key metric:**
```
Savings = W_q4_bytes - (W_low_bytes + R_compressed_bytes + R_decode_working_set)
```

**Key insight from Phase 30F:** FFN sidecar overhead (~277 MB) was dominated by the sidecar being resident in F32. Residual must stay compressed — decompressing to F32 resident recreates the additive trap.

---

## 4. Avoiding the Additive Trap

The failure in Phase 30F was: W_q4 stayed loaded AND the sidecar was loaded = additive.

**Prevention options:**

**A. llama.cpp tensor load interception:**
- Set `tensor->data = nullptr` or replace pointer before first use
- **Risk:** GGML asserts if null tensor is used. Must be atomic before graph finalization.

**B. Replace at GGML graph construction time:**
- At `build_ffn_up()` or `build_attn_out()` call, substitute W_low for W_ref
- **Risk:** Multiple nodes may reference the tensor; replacement must be atomic.

**C. Custom GGML op that takes W_low + R pointers, computes X @ W_low + X @ R:**
- Never materializes W_ref
- Existing PRT infrastructure has pattern for this (`build_prt_*` functions)
- **Recommended for llama.cpp integration phase.**

**D. Standalone harness first (safest):**
- Extract tensors, build residual, measure memory and accuracy outside llama.cpp
- Proves thesis before touching runtime
- **Recommended first move.**

---

## 5. First Implementation Target

**Pick: A. Standalone tensor harness first**

Reason: proves the memory thesis and math thesis outside llama.cpp graph surgery. No risk of corrupting working setup. Fail fast with clear evidence before touching llama.cpp.

**What it does:**
1. Extract W_q4 (attn_out layer 0) from GGUF
2. Generate W_low (Q2) and R = W_q4 - W_low, quantize R
3. Load W_low + R_sidecar
4. Run X @ W_low + X @ R with recorded/random activations
5. Measure memory: W_low + R compressed vs W_q4
6. Measure accuracy: cosine(Y_ref, Y_sub), logit delta

**What it does NOT do:** modify llama.cpp, run inference, touch GGML graph.

---

## 6. Phase Ladder

| Phase | Task | Goal |
|-------|------|------|
| **31A** | Architecture plan | This document — design frozen |
| **31B** | Residual economics harness | Test which residual representations are memory-viable |
| **31C** | Memory accounting for attn_out | Measure W_low + R vs W_q4 with PSS |
| **31D** | Compressed residual compute probe | Decode-on-compute vs pre-decode tradeoffs |
| **31E** | llama.cpp custom op feasibility | Can we build `compute_substitutive()` inside llama.cpp? |
| **31F** | llama.cpp loader substitution | Replace W_ref load with W_low + R at graph construction |

---

## 7. Claim Boundary

**Allowed:**
- Substitutive tensor math prototype runs and produces output
- Memory accounting for one tensor family shows W_low + R_compressed < W_q4
- Residual approximation metrics (cosine, MAE) are reported
- attn_output chosen as minimal proof target

**Forbidden until proven:**
- Quality recovery (behavior matches W_q4 within tolerance)
- Speedup (compute time vs W_q4)
- Production readiness
- Full model behavior
- Generalization to ffn_up/ffn_down
- Portability claims

---

## 8. Top Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Residual cache bloat (decoded F32 R stays resident) | HIGH | Stay compressed; decode at compute time only |
| Not enough memory savings | MEDIUM | Use Q2 or lower; ffn_up has better shape ratio |
| Behavior recovery fails (Y_sub diverges from Y_ref) | MEDIUM | Cosine + logit delta first; threshold before proceeding |
| llama.cpp load interception complexity | HIGH | Standalone harness first; don't touch llama.cpp until thesis proven |
| Quant mismatch (rounding errors accumulate) | LOW | F32 residual before quantization; measure quantization error separately |
| Activation mismatch (X not representative) | MEDIUM | Multiple activation sources; random + recorded |

---

## 9. Final Recommendation

**Start a new repo: YES**

The old experimental repo (`experimental/prt-phase19a-alt-sidecar-backed`) is the archive. It should not be touched for substitutive work — the two architectures are fundamentally different and should not share a codebase until the substitutive path proves viable.

**Lowest-risk first implementation step:**

1. Extract attn_out layer 0 from Qwen2.5-0.5B Q4 GGUF using a Python GGUF reader
2. Generate residual: W_q2 and R = W_q4 - W_q2 in F32, quantize R to INT8
3. Run memory probe: load W_q2 + R_sidecar, run X @ W_q2 + X @ R, measure RSS delta

**Exact next action:**
```bash
mkdir -p ~/sdi-substitutive/{src,data,results,docs}
```

**Critical correction (per Matt):** Dense INT8 residual does NOT beat Q4. Residual must average < 2 bits/weight for Q2 + residual to win. See `docs/PHASE31B_RESIDUAL_ECONOMICS_PLAN.md` for the corrected Phase 31B approach.