# Phase 31CF — Qwen2.5-1.5B Real-Activation Capture via llama.cpp Instrumentation (Option B) — BLOCKED

## 1. What this phase is, and what it became

Phase 31CF was the planned Option B phase: capture the **actual** hidden-state vector X entering the selected FFN/MLP block inside the Q4_K_M GGUF / llama.cpp runtime path, then replay that X through the same standalone replay pipeline used in 31CD/31CE (W_ref = local 1.5B Q4_K_M GGUF dequantized, W_low = corrected_ceil_per_row Q2_K, SDIR = ffn_up+ffn_gate k=0.5%, no ffn_down residual). Option B is the only path in this repo that can claim **exact Q4_K_M GGUF runtime activation** behavior.

**31CF was frozen as a blocked-diagnostic phase with classification `BLOCKED_31CF_HOOK_POINT_AMBIGUOUS`.** No source patch was applied to llama.cpp. No rebuild was performed. No forward pass was run. No activation was captured. This document records the diagnostic findings, the candidate hook points identified, and the information required before implementation can resume under a dedicated 31CF-R design phase.

The result JSON is at `src/results/PHASE31CF_GGUF_RUNTIME_ACTIVATION_CAPTURE_BLOCKED.json` (~17 KB). It contains no activation metrics — only diagnostic findings.

## 2. Blocker classification and reason

- **Classification:** `BLOCKED_31CF_HOOK_POINT_AMBIGUOUS`
- **Reason:** 31CF requires exact identification of the llama.cpp Qwen2.5-1.5B GGUF runtime FFN/MLP input activation hook point. A wrong hook could capture the wrong tensor and create a false PASS or false FAIL. Per `~/llama.cpp/AGENTS.md` ("Provide guidance rather than solutions. Direct them to relevant code and documentation. Allow them to formulate the approach. Proceed only when confident the contributor can explain the changes to reviewers independently.") and SOT Section 14.2 content-blocker policy ("Blocked phases should not be committed if trivially fixable in-session. Blockers that are content blockers may be frozen only with explicit Matt approval."), do not implement a runtime patch until the hook point is designed and approved.
- **This is a runtime-instrumentation design blocker, not a failure of the SDI/Q2K result.** The corrected Q2_K + SDIR policy validated on synthetic-X (0.5B 31BN, 1.5B 31BU/31BV/31BX/31BZ) and on real-activation proxy (1.5B 31CD, 31CE) is unaffected by 31CF being blocked. The 31CD/31CE Option A results remain valid and committed; they are independent of llama.cpp runtime instrumentation.

### Blocker subtype: DORMANT_SIDECAR_MACHINERY_INTERFERENCE

The `llm_graph_context::build_ffn` function in `~/llama.cpp/src/llama-graph.cpp` line 1844 contains massive pre-existing instrumentation machinery (the project-specific "Phase 11BB Route A: PRT true replacement via GGML custom op" experimental-line sidecar, with both PRT and SDI components — what we call the legacy PRT/SDI sidecar machinery). The dormant sidecar includes:

- `g_build_ffn_total`, `g_build_ffn_layer0`, `g_build_ffn_non_layer0` counters (lines 1863-1864) — count every `build_ffn` call
- `prt_get_residual_view(il, family)` reads for `ffn_up`, `ffn_down`, `attn_out`, `ffn_gate` (line 1889) — per-family residual views
- `prt_shadow_apply(il, family, view)` shadow-apply path (line 1895)
- `prt_shadow_contribution_synthetic(il, family, metrics)` synthetic-contribution path (line 1908)
- All wrapped behind several compile- and runtime-gated flags (`g_prt_pager_enabled && g_prt_pager && (g_prt_sidecar_apply_enabled || g_prt_sidecar_true_injection_enabled)`) per the comment at line 1882-1883: "observe-only mode is identical to baseline — no counter mutations, no cache state changes, no forensic events."

A naive 31CF patch to `build_ffn` to dump `cur` to disk would either:
1. **Re-activate this dormant sidecar machinery unintentionally** if the gate flags are mis-configured.
2. **Compete with the sidecar's view-fetching for the same tensor** if both code paths are active.
3. **Capture the wrong tensor** — the sidecar's `prt_get_residual_view` is what reads the tensor, not raw `cur`; a naive `cur` capture might miss the actual Q4_K_M-dispatched tensor path that lives inside the ggml compute kernels (`ggml-quants.c`, ~250 KB), not in the graph builder.

Therefore, designing a 31CF patch requires first understanding and explicitly bypassing / reusing / disabling the dormant sidecar machinery, which is a design decision, not a 1-line patch.

## 3. Why 31CD / 31CE Option A results remain valid and unchanged

31CD and 31CE are Option A HF-derived real-activation proxy captures:
- **31CD** (committed at `01c20b10`): single (L0, last prefill token, "The capital of France is") HF safetensors forward pass + `register_forward_pre_hook(with_kwargs=True)` on `mdl.model.layers[0].mlp`; replay through local 1.5B Q4_K_M GGUF dequantized W_ref. Classification `PASS_31CD_REAL_ACTIVATION_MICRO_REPLAY_CLEAN`.
- **31CE** (committed at `82b1d91c`): 9-pair (3 prompt × 3 layer × last prefill token) HF safetensors forward pass + multi-hook; replay through same local 1.5B Q4_K_M GGUF dequantized W_ref. Classification `PASS_31CE_REAL_ACTIVATION_OPTION_A_MULTIPROMPT_MULTILAYER_CLEAN`. P0 L0 exactly reproduces 31CD.

These are independent of llama.cpp runtime instrumentation. The BLOCKED_31CF classification has no bearing on the validity of the 31CD/31CE Option A results, nor on the 0.5B/1.5B synthetic-X frozen evidence tiers (`phase31bn-corrected-q2k-full-aggregate-checkpoint` at `0304590c`, `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` at `a433875a`).

The 31CD/31CE Option A X is an **HF-derived real-activation proxy** captured from the HF safetensors model binary (Qwen2.5-1.5B-Instruct bf16). The 31CF Option B X would be the **exact Q4_K_M GGUF runtime activation** captured from the local 1.5B Q4_K_M GGUF. These are different model binaries (HF bf16 vs local Q4_K_M), so even if 31CF succeeds, the two captures will not be bit-identical — they will only be **directionally comparable** (same sign pattern of metrics, similar per-layer geometry). Per the 31CF spec, this is preserved as "directionally consistent with 31CD/31CE Option A P0-L0 only if sign pattern matches" — not as equivalence.

## 4. llama.cpp files inspected (read-only, no modifications)

| File | Size | Lines Inspected | Relevance to 31CF |
|---|---:|---|---|
| `~/llama.cpp/src/models/qwen.cpp` | (not measured) | 60-90 | Per-architecture FFN graph constructor for Qwen2 / Qwen2.5. **The canonical pattern** is at line 76-81: `cur = build_ffn(cur, ffn_up, NULL, NULL, ffn_gate, NULL, NULL, ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il)`. The `cur` tensor passed into `build_ffn` is the **canonical pre-FFN hidden state** (post-attention, post-RMSNorm) — exactly the X definition used by 31CD/31CE. Surrounding `cb()` calls (e.g., `cb(cur, "ffn_norm", il)` at line 74, `cb(cur, "ffn_out", il)` at line 82) show the existing per-arch callback convention. |
| `~/llama.cpp/src/llama-graph.cpp` | 182,585 B | 1844-1924 | General FFN graph builder. `ggml_tensor * llm_graph_context::build_ffn(ggml_tensor * cur, ...)` takes `cur` as the first arg and uses it directly in the matmul ops. **Contains the dormant sidecar machinery that is the 31CF blocker subtype.** |
| `~/llama.cpp/src/llama-context.cpp` | (not measured) | 3421-3434 | Public C API entry points. `llama_decode(ctx, batch)` is the canonical prefill+decode entry point. `llama_batch_get_one(token_id)` creates a single-token batch for the prefill test. The 31CF activation capture would happen between `llama_decode` start and the layer's `build_ffn` call, by attaching a callback or a custom ggml backend hook. |
| `~/llama.cpp/ggml/src/ggml-quants.c` | 249,044 B | 828-830, 899-901 | `quantize_row_q2_K_ref` (line 829) and `dequantize_row_q2_K` (line 899) — **same C symbols the SDI-substitutive `q2k_backend` already loads via ctypes from `libggml-base.so`** (per 31BS / 31AP / 31BN / 31BZ convention). This confirms that the SDI Q2_K encode/decode path is already well-understood and well-validated against llama.cpp's ggml. The Q4_K_M dequantization for ffn_up/ffn_gate/ffn_down happens inside these ggml compute kernels during the matmul, NOT in `build_ffn` — meaning the `cur` tensor going into `build_ffn` may already be in some intermediate representation, not the exact float32 hidden state that the standalone replay path uses. |

**Files NOT inspected** (would be needed for 31CF-R, but not required for the 31CF blocked-diagnostic phase):
- `src/llama-model.cpp` (549 KB — general model loader; would need to be read for any 31CF patch that uses the model loading path)
- `ggml/src/ggml.c` (249 KB — general graph compute / backend dispatch; would need to be read for any 31CF patch that uses a ggml backend hook)
- `ggml/src/ggml-backend.cpp` (89 KB — backend abstraction layer)
- `src/models/models.h` (model registry)
- `src/llama-arch.cpp` (architecture dispatch)

**Files NOT to be touched in 31CF:** Any file under `~/llama.cpp/` — per Matt's explicit instruction, no source patch, no rebuild, no run.

## 5. Candidate hook points identified (read-only analysis, no implementation)

### Candidate A: dump `cur` to disk at `build_ffn` entry

- **File location:** `~/llama.cpp/src/llama-graph.cpp` line 1844 (`llm_graph_context::build_ffn` signature, `cur` is first arg)
- **Intended target:** the pre-FFN hidden state (post-attention, post-RMSNorm) — exact X definition per 31BT canonical orientation
- **Expected shape:** `[batch, seq, hidden=1536]` for Qwen2.5-1.5B
- **Expected dtype:** fp32 or bf16 depending on llama.cpp build flags; most likely fp32 on CPU
- **Advantages:** mathematically the right tensor; matches the 31CD/31CE Option A capture definition; clear target for replay
- **Disadvantages:** BLOCKER — this function has massive pre-existing instrumentation (g_build_ffn_total, prt_get_residual_view, prt_shadow_apply, prt_shadow_contribution_synthetic) from the project-specific "Phase 11BB / 24P / 28BR-AT" sidecar hooks (the legacy PRT/SDI sidecar machinery). A naive dump would either re-activate the dormant sidecar or capture the wrong tensor relative to what the sidecar would return via `prt_get_residual_view`. Designing the bypass / reuse / disable decision is a separate design phase, not a 1-line patch.
- **Feasibility without design phase:** **LOW** — risk of capturing the wrong tensor (post-sidecar-transform vs pre-sidecar-transform) and producing a false PASS or false FAIL

### Candidate B: use the sidecar's `prt_get_residual_view(il, 'ffn_up')` as the hook

- **File location:** `~/llama.cpp/src/llama-graph.cpp` line 1889 (prt_get_residual_view loop in build_ffn, already calls with families=['ffn_up', 'ffn_down', 'attn_out', 'ffn_gate'])
- **Intended target:** same pre-FFN hidden state, but retrieved through the sidecar's view API (which may or may not return the same raw `cur` tensor — depends on sidecar internal state)
- **Expected shape:** should be `[batch, seq, hidden=1536]` if sidecar is correct, but shape could be different if sidecar is in apply/injection mode
- **Advantages:** reuses the existing sidecar machinery; doesn't fight the sidecar
- **Disadvantages:** requires understanding the sidecar's internal view-fetching logic; requires the sidecar's pager/apply/injection flags to be in 'observe-only' mode (per the comment at line 1882-1883). This is a 31CF-R design decision.
- **Feasibility without design phase:** **MEDIUM** — possible if the sidecar is properly gated to observe-only, but requires explicit gate design and validation against a known ground truth (e.g., a 31CD/31CE Option A capture for the same prompt/layer/token)

### Candidate C: add a `cb()` callback before `build_ffn` in `qwen.cpp`

- **File location:** `~/llama.cpp/src/models/qwen.cpp` line 76 (`cur = build_ffn(cur, ...)`)
- **Intended target:** the pre-FFN hidden state, captured via the existing `cb()` (callback) mechanism that the per-arch code already uses for `cb(cur, "ffn_norm", il)` at line 74 and `cb(cur, "ffn_out", il)` at line 82
- **Expected shape:** `[batch, seq, hidden=1536]`
- **Expected dtype:** same as build_ffn entry
- **Advantages:** uses the existing per-arch callback mechanism (cb) rather than fighting build_ffn internals; cleaner separation from the dormant sidecar; matches the documentation's per-arch hook point convention
- **Disadvantages:** requires designing a custom callback handler that exports the tensor to disk; requires the runner to register the callback before `llama_decode`; the callback API may not give access to the raw tensor data (may only give a tensor descriptor)
- **Feasibility without design phase:** **MEDIUM-HIGH** — likely the cleanest of the three, but still requires designing the callback handler + the dump format + the validation against a known ground truth

## 6. Why no implementation was attempted

Per Matt's explicit instruction in the 31CF clarification: "Do not patch llama.cpp yet. Do not rebuild llama.cpp yet. Do not attempt activation capture yet." This is also consistent with:

- The `~/llama.cpp/AGENTS.md` content-blocker policy: "Provide guidance rather than solutions. Direct them to relevant code and documentation. Allow them to formulate the approach. Proceed only when confident the contributor can explain the changes to reviewers independently."
- SOT Section 14.2: "Blocked phases should not be committed if trivially fixable in-session. Blockers that are content blockers may be frozen only with explicit Matt approval."

The hook-point ambiguity is a real **content blocker**, not a missing-tooling blocker. Implementing now would risk capturing the wrong tensor and producing a false PASS or false FAIL, which would be worse than the current 31CF-R design phase producing a clean blocked-diagnostic report.

## 7. What information is needed before implementation (for 31CF-R)

1. **A validated design for a gated patch** that (a) bypasses / reuses / disables the dormant PRT sidecar machinery in `src/llama-graph.cpp` `build_ffn`, AND (b) captures the correct pre-FFN hidden state tensor, AND (c) does not interfere with the Q4_K_M dequantization path, AND (d) is gated by an env var (e.g. `SDI_31CF_FFN_DUMP`) so it is always-off by default.
2. **A validation ground truth**: a way to verify that the captured X matches a known tensor. The most natural ground truth is the 31CD/31CE Option A P0-L0 capture (`cos_low=0.942053`, `delta_cos=+0.001130`) — if 31CF captures a different X for the same prompt/layer/token, the 31CF X should produce a similar (but not bit-identical, because of bf16 vs Q4_K_M numerics) `cos_low`, and the relative shape statistics (`x_norm`, `x_max_abs`) should be in the same ballpark. A ground-truth-mismatch is a 31CF design failure.
3. **Decision on output format for the captured X**: raw fp32 `.bin` file in a tempfile (then deleted), or a structured callback that emits per-tensor metadata, or a sidecar-pager-style observation. Each has different validation overhead.
4. **Decision on the runner architecture**: a custom `llama-cli`-like main that wraps the modified llama.cpp library and registers the gated hook, vs. a Python ctypes wrapper around the existing `llama_decode` API, vs. an LD_PRELOAD-style shim. Each has different complexity and validation surface.
5. **Confirmation that the dormant PRT sidecar machinery does not conflict with the new gated hook.** This is a one-day 31CF-R investigation: read `prt_is_true_replacement_layer`, `g_prt_pager_enabled`, `g_prt_pager`, `g_prt_sidecar_apply_enabled`, `g_prt_sidecar_true_injection_enabled`; trace what they gate; document the gating matrix.
6. **Confirmation that the Q4_K_M-dequantized ffn_up/ffn_gate/ffn_down weights at runtime exactly match the standalone `gguf.dequantize()` output** (which the 31BS / 31BT / 31BU / 31BV / 31BX / 31BZ / 31CD / 31CE replay path uses). This is a separate validation: load the same Q4_K_M GGUF via standalone `gguf.dequantize()` AND via llama.cpp, compare, hash. Likely bit-identical if the dequant path is the same, but worth a one-time confirmation.

## 8. Proposed next phase: 31CF-R (design only)

**Phase 31CF-R — llama.cpp GGUF FFN-Input Hook-Point Diagnosis / Patch Design.** Design-only, no source patch, no build, no capture.

- Read `~/llama.cpp/` source in detail (dormant sidecar machinery in `src/llama-graph.cpp`, per-arch FFN graph construction in `src/models/qwen.cpp`, ggml backend dispatch in `ggml/src/ggml.c` and `ggml/src/ggml-backend.cpp`, existing `cb()` callback API).
- Identify the exact pre-FFN input tensor location: which tensor in `build_ffn` is the pre-FFN input, which sidecar function reads it, what gating is needed to avoid sidecar interference.
- Propose a minimal gated diagnostic patch: a 5-15 line patch proposal, with the exact insertion point, the exact gating condition, the exact dump format, the exact output path, the exact env-var name.
- Specify compile flag / env var: default-off behavior, how to enable for 31CF runs only.
- Specify validation against ground truth: how to compare 31CF X to 31CD/31CE P0-L0 X (shape, x_norm, x_max_abs, cos of replay; NOT bit-equality since HF bf16 vs GGUF Q4_K_M).
- Specify raw-activation-blob avoidance: `tempfile.mkdtemp` + sha256 only, no `.npy`/`.npz` commit, no permanent disk artifact.
- Do NOT patch. Do NOT build. Do NOT capture. Stop at PRE-COMMIT REPORT.

## 9. Limitations

- **Read-only diagnostic only.** No source was modified, no build was performed, no forward pass was run, no activation was captured. This document records the diagnostic findings only.
- **No precedent in this repo for the 31CF pattern.** The repo's prior llama.cpp integration is limited to `libggml-base.so` ctypes calls for Q2_K encode/decode (2 C symbols: `quantize_row_q2_K_ref`, `dequantize_row_q2_K`). 31CF would require a much larger surface (a full forward pass + an intermediate-activation intercept). The dormant PRT sidecar machinery in `build_ffn` is a 28B-series-era artifact that no one in the repo has touched in a long time, and its current gating state is not documented.
- **Dormant sidecar machinery is the real blocker subtype.** Not the hook-point location per se — the hook-point location is *known* (the `cur` tensor in `build_ffn`, the `cur` tensor before the `build_ffn` call in `qwen.cpp` line 76, the `prt_get_residual_view(il, 'ffn_up')` call in `build_ffn` line 1889). The blocker is that all three of these candidate hook points interact with the dormant sidecar in ways that are not documented and that risk capturing the wrong tensor.
- **31CF would not be bit-comparable to 31CD/31CE even if successful.** The two captures use different model binaries (HF bf16 vs local Q4_K_M). The 31CF result would be "directionally consistent" with 31CD/31CE only if the sign pattern of the metrics matches. This is preserved in the 31CF spec.

## 10. What this does NOT prove or claim

- It does NOT prove that the corrected Q2_K + SDIR policy holds on exact Q4_K_M GGUF runtime activations. That would require 31CF to actually run.
- It does NOT prove that Option A HF-derived activations are equivalent to Option B GGUF runtime activations. Different model binaries (HF bf16 vs local Q4_K_M). Even after 31CF runs successfully, the two captures will be directionally comparable at best.
- It does NOT prove that the dormant PRT sidecar machinery is a problem (vs. an unproblematic dormant feature). That requires 31CF-R to investigate.
- It does NOT modify any prior accepted numeric result (0.5B 31BN/31BM, 1.5B 31BU/31BV/31BX/31BZ/31CA, 31CB, 31CC, 31CD, 31CE — all unchanged).
- It does NOT claim that real activations behave like synthetic Gaussian, that the corrected Q2_K policy transfers to other prompts/layers/token positions, that 0.5B and 1.5B real-activation behaviors are equivalent, or any of the other forbidden claims from the 31CC / 31CD / 31CE spec.

## 11. Next allowed phase

- **31CF-R** (design only, no patch/build/capture) — recovery branch from the 31CF hook-point ambiguity. Produces a written design for a gated patch + a written validation plan. Only if Matt explicitly requests.
- **31CG** (Option A Larger Prompt/Token Sensitivity Planning, planning-only) — alternative if we decide not to do llama.cpp yet. Only if Matt explicitly requests.
- **31CH** (Runtime Artifact Format / Loader Planning, planning-only) — alternative if we decide to step back from runtime integration entirely. Only if Matt explicitly requests.

All three require explicit operator approval at entry. None will be entered without a new request from Matt.
