# Phase 31CF-R — llama.cpp GGUF FFN-Input Hook-Point Diagnosis / Patch Design

**Phase ID:** 31CF-R
**Classification:** `PARTIAL_31CFR_HOOK_POINT_CANDIDATES_IDENTIFIED` (see JSON; details in §10)
**Type:** Design-only, no source patch, no build, no capture, no commit
**Date:** 2026-06-06
**Author:** agent (Matt-approved phase entry)
**Repo:** `sdi-substitutive`
**Target external repo:** `~/llama.cpp/` (read-only inspection only)
**Allowed by SOT:** yes — see `SOURCE_OF_TRUTH.md` Section 0 line 11, Section 9 (current allowed next phase), Section 0.A (forbidden claims)

---

## 0. Scope (read first)

This phase is **design-only**. It:
- **reads** `~/llama.cpp/` source code in detail
- **identifies** candidate hook points for capturing the exact pre-FFN input activation X for Qwen2.5-1.5B Q4_K_M GGUF
- **rates** each candidate
- **proposes** a minimal gated diagnostic patch design (in prose only — NO actual diff)
- **proposes** a validation plan for a future 31CF-S implementation phase
- **does NOT** patch `~/llama.cpp/`
- **does NOT** rebuild `~/llama.cpp/`
- **does NOT** run `llama-server` / `llama-cli` / `llama_decode` / `llama_encode`
- **does NOT** capture activations
- **does NOT** generate Q2_K / SDIR blobs
- **does NOT** run generation / inference / sampling
- **does NOT** modify external runtime source
- **does NOT** commit patches to `~/llama.cpp/`
- **does NOT** commit raw activation arrays
- **does NOT** commit model files
- **does NOT** commit, push, or tag in this repo without explicit operator approval

The goal of 31CF-R is to **unblock** 31CF (BLOCKED_31CF_HOOK_POINT_AMBIGUOUS) by producing a complete hook design that the operator can review, approve, and have implemented in a later phase (e.g. 31CF-S).

---

## 1. Background

### 1.1 The 31CF blocker

`SOURCE_OF_TRUTH.md` Section 3, 31CF entry: `BLOCKED_31CF_HOOK_POINT_AMBIGUOUS` with subtype `DORMANT_SIDECAR_MACHINERY_INTERFERENCE`. The 31CF diagnostic read `~/llama.cpp/` source (4 files / ~80 lines) and found that `llm_graph_context::build_ffn` (in `~/llama.cpp/src/llama-graph.cpp` line 1844) contains a large body of pre-existing instrumentation machinery (the legacy PRT/SDI sidecar from project-specific Phases 11BB / 24P / 28BR-AT): counters (`g_build_ffn_total`, `g_build_ffn_layer0`, `g_build_ffn_non_layer0` at lines 1863-1864), residual-view fetches (`prt_get_residual_view(il, "ffn_up"|"ffn_down"|"attn_out"|"ffn_gate")` at line 1889), shadow apply (`prt_shadow_apply` at line 1895), and shadow contribution (`prt_shadow_contribution_synthetic` at line 1908), plus the entire `build_prt_true_ffn_{up,gate,down}_injection` family (lines 1342-1770). 31CF did not patch, did not build, did not run — it recorded the blocker and stopped at PRE-COMMIT REPORT.

### 1.2 The 31CF-R recovery branch

31CF-R is the design-only recovery branch listed in SOT Section 0 line 11: *"read `~/llama.cpp/` source in detail, identify the exact pre-FFN input tensor location, propose a minimal gated diagnostic patch, specify compile flag / env var, specify validation against the 31CD/31CE P0-L0 Option A ground truth, do NOT patch, do NOT build, do NOT capture, stop at PRE-COMMIT REPORT"*.

### 1.3 The standalone replay X (what 31CD/31CE captures)

The SDI-substitutive standalone replay pipeline (see `docs/PHASE31CD_REAL_ACTIVATION_MICRO_PROBE_OPTION_A.md` line 46) captures the **input to the L0 MLP** of Qwen2.5-1.5B-Instruct safetensors model via `register_forward_pre_hook(with_kwargs=True)` on `mdl.model.layers[0].mlp`. This is the **post-attention residual stream hidden state at the L0 boundary**, which is the **canonical pre-FFN activation** for L0. For prompt "The capital of France is" (6 BPE tokens, add_special_tokens=False), `X = hook_input[:, -1, :]` with shape `[1, 1536]` (hidden_size=1536 for Qwen2.5-1.5B, last prefill token, batch=1).

In llama.cpp's Qwen2 graph constructor, the equivalent tensor is `ffn_inp` at `~/llama.cpp/src/models/qwen.cpp:66-67`:
```cpp
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);  // post-attention-residual-add
cb(ffn_inp, "ffn_inp", il);                          // per-arch callback convention
```

This `ffn_inp` is then passed through `build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il)` (line 71) to produce the **post-norm** `cur` at `cb(cur, "ffn_norm", il)` (line 74), which is then passed into `build_ffn` (line 76) as the first argument.

**Key fact:** the `ffn_inp` at `qwen.cpp:66-67` is **identical in construction and content** to the `hook_input` captured by the HF forward-hook in 31CD — both are the post-attention-residual-add hidden state at the layer boundary, before the layer's RMSNorm. Shape is `[1, 1536]` for batch=1, last prefill token, Qwen2.5-1.5B.

### 1.4 The dormant-sidecar scope at the binary level

**Important new finding from 31CF-R source inspection (see §4.3 below):** the operator's local `~/llama.cpp/build/bin/llama-server` binary does **NOT** contain any of the PRT sidecar symbols (verified via `strings | grep` returning 0 matches for `prt_get_residual_view`, `prt_shadow_apply`, `prt_shadow_contribution_synthetic`, `build_prt_true_ffn_up_injection`, `g_build_ffn_total`, `g_prt_pager_hook_calls`, `PRT_SIDECAR_PAGER_EXPERIMENTAL`). All PRT sidecar functions in `~/llama.cpp/src/llama-graph.cpp` are guarded by `#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL` (lines 27, 1346, 1455) — a **compile-time gate** — and the operator's binary was built without this define. Therefore, at the **binary level** for the operator's local `llama-server`, the PRT sidecar is **fully dormant**. The 31CF blocker subtype `DORMANT_SIDECAR_MACHINERY_INTERFERENCE` is accurate at the source level but **inactive at the binary level** for the operator's current `llama-server`.

**Implication for the 31CF-R design:** the gating design must include an explicit check that the future 31CF-S build does NOT add `-DPRT_SIDECAR_PAGER_EXPERIMENTAL` to its CMake invocation. The 31CF blocker can be neutralized at the build configuration level, not at the source code level.

---

## 2. AGENTS.md constraint summary

From `~/llama.cpp/AGENTS.md` (read in full, 110 lines):

- llama.cpp does not accept fully or predominantly AI-generated PRs (irrelevant to 31CF-R; 31CF-R produces no PR and is for local use only)
- AI agent guideline: *"Proceed only when confident the contributor can explain the changes to reviewers independently"* (relevant for a future 31CF-S implementation phase; 31CF-R itself is documentation, not code)
- *"Committing or pushing without explicit human approval for each action"* — confirms 31CF-R must stop at PRE-COMMIT REPORT ✓
- *"When uncertain, err toward minimal assistance. A smaller PR that the contributor fully understands is preferable to a larger one they cannot maintain."* — design-only is the minimal possible footprint ✓

The 31CF-R design proposes a **non-invasive** patch: a single opt-in `cb_func` registration (via the existing `llm_graph_context::cb` mechanism) plus a gated read+serialize step at the call site. No new public API, no graph mutation, no upstream-facing change. The patch can live entirely in the operator's local `~/llama.cpp/` checkout and never be sent upstream.

---

## 3. Files / functions inspected (read-only)

| File | Lines inspected | Purpose |
|---|---|---|
| `~/llama.cpp/AGENTS.md` | 1-110 | Contributor + AI agent guidelines |
| `~/llama.cpp/src/models/qwen.cpp` | 1-108 (full file, 108 lines) | Per-arch Qwen2/Qwen2.5 graph constructor |
| `~/llama.cpp/src/llama-graph.cpp` | 1295-1303 (`llm_graph_context::cb`); 1342-1366 (`build_prt_true_attn_out_injection`); 1450-1541 (`build_prt_true_ffn_up_injection`); 1553-1648 (`build_prt_true_ffn_gate_injection`); 1660-1770 (`build_prt_true_ffn_down_injection`); 1844-1933 (`build_ffn` signature + entry + first PRT sidecar block); 27, 1346, 1455 (`#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL` guards); 100-112 (counter globals); 30, 51, 57, 213, 226, 236, 242, 248, 254, 260 (PRT env-var reads) | Per-arch callback mechanism + dormant PRT sidecar |
| `~/llama.cpp/build/bin/llama-server` | binary `strings` (no read) | Verified no PRT sidecar symbols present in operator's local build |

**No** patches, edits, or writes to any file outside this repo's `docs/` and `src/results/` directories.

---

## 4. Candidate hook points

### 4.1 Candidate A — dump `cur` to disk at `build_ffn` entry

- **File / function / line range:** `~/llama.cpp/src/llama-graph.cpp` `llm_graph_context::build_ffn` (line 1844, function entry)
- **What tensor would be captured:** the `cur` first argument passed into `build_ffn` from the per-arch caller (in `qwen.cpp:76`)
- **Whether definitely pre-FFN input X:** **partially.** This `cur` is the **post-attention-RMSNorm** hidden state (after `build_norm(ffn_inp, ffn_norm, ...)` in `qwen.cpp:71`), not the raw `ffn_inp`. **Shape mismatch with 31CD's `[1, 1536]`.** 31CD captures pre-norm `ffn_inp` (shape `[1, 1536]`); `cur` at `build_ffn` entry is post-norm (shape `[1, 1536]` but numerically different values). Cross-validation against 31CD would NOT be a like-for-like comparison.
- **Before or after layer norm:** **AFTER** layer norm (this is post-norm `cur`).
- **Before ffn_up / ffn_gate matmuls:** **YES**, before the matmul (matmuls happen inside `build_ffn`).
- **Risk of interacting with dormant PRT sidecar:** **HIGH.** The function body is where the dormant PRT sidecar machinery lives (counters, residual views, shadow apply, shadow contribution, plus the three `build_prt_true_ffn_{up,gate,down}_injection` callers). A naive dump at entry would create ambiguity about which value is the "true" pre-FFN input. Even with the binary built without `PRT_SIDECAR_PAGER_EXPERIMENTAL`, the source still contains the sidecar code paths, and a future patch that adds the ifdef would silently re-activate them.
- **Can be gated cleanly:** **MEDIUM.** The PRT sidecar gates on env vars (`PRT_V2_SIDECAR_FORMAT`, `PRT_GGML_TEST_LAYER`, `PRT_V2_DECODE_ONLY`, `PRT_V2_USE_NATIVE_MULMAT`, etc.) and counters; a separate `SDI_CAPTURE_*` env-var gate would work but requires careful ordering with the PRT initialization to avoid races.
- **Can write a single layer/token only:** **YES** (gate on `il == SDI_CAPTURE_LAYER`).
- **Expected activation shape:** `[N_tokens, 1536]` for Qwen2.5-1.5B at L0 (post-norm, **not** 31CD's pre-norm shape identity).
- **Implementation risk:** **MEDIUM-HIGH.** The `build_ffn` body is long (~200 lines including the sidecar); a patch must be placed before the sidecar entry block to avoid contamination.
- **Validation risk:** **MEDIUM.** The captured tensor would be post-norm, so direct comparison to 31CD's HF forward-hook (which captures pre-norm `hook_input`) would not be a like-for-like comparison. Would require either a post-norm replay of 31CD or a documented caveat.
- **Recommendation:** **REJECT** for the 31CF-R primary recommendation. The post-norm shape mismatch with 31CD's ground truth is a real problem that would require either a second 31CD-like run for post-norm data or a careful caveat in any 31CF-S claim. Keep `build_ffn` untouched in any future patch.

### 4.2 Candidate B — use the sidecar's `prt_get_residual_view(il, "ffn_up")` as the hook

- **File / function / line range:** `~/llama.cpp/src/llama-graph.cpp` `prt_get_residual_view` (called at line 1889 in `build_ffn`); also `prt_shadow_apply` (line 1895), `prt_shadow_contribution_synthetic` (line 1908)
- **What tensor would be captured:** the **decoded residual view** of the "ffn_up" family sidecar blob, **NOT** the pre-FFN input activation X. This is a *residual R* view, not an *activation X* view.
- **Whether definitely pre-FFN input X:** **NO.** This is a sidecar-residual R tensor, fundamentally different from the activation X that 31CD/31CE captures. The standalone replay pipeline expects X, not R. Using this candidate would require re-purposing the replay pipeline to consume R instead of X, which is **out of scope** for the SDI-substitutive project (R is computed as `R = W_ref - W_low` at offline; X is the runtime activation).
- **Before or after layer norm:** N/A — this is a residual view, not an activation.
- **Before ffn_up / ffn_gate matmuls:** N/A.
- **Risk of interacting with dormant PRT sidecar:** **MAXIMUM.** This candidate **uses** the sidecar machinery directly, so it is completely dependent on it being active and well-behaved. If the sidecar is dormant (as in the operator's current binary), this candidate returns `is_null` views and the hook captures nothing. If the sidecar is active in a future build, the hook is now tightly coupled to PRT/SDI internal state.
- **Can be gated cleanly:** **NO.** Tied to sidecar lifecycle.
- **Can write a single layer/token only:** **YES** in principle (gate on sidecar apply-layer), but the sidecar's own gating is already complex (`g_prt_sidecar_apply_layer`, `g_prt_sidecar_apply_family`).
- **Expected activation shape:** N/A (residual, not activation).
- **Implementation risk:** **HIGH.** Tied to dormant-sidecar behavior.
- **Validation risk:** **HIGH.** Cross-validation against 31CD would be impossible (different tensor type).
- **Recommendation:** **REJECT** strongly. The sidecar machinery is the wrong infrastructure for capturing activation X. Keeping the SDI 1.5B work decoupled from the PRT/SDI sidecar is a hard design constraint.

### 4.3 Candidate C — add a `cb()` callback before `build_ffn` in `qwen.cpp` (at the `cur = build_ffn(...)` call site, line 76)

- **File / function / line range:** `~/llama.cpp/src/models/qwen.cpp` line 76 (between line 74's `cb(cur, "ffn_norm", il)` and the `cur = build_ffn(cur, ...)` at line 76)
- **What tensor would be captured:** the **post-norm** `cur` (same as Candidate A's entry argument)
- **Whether definitely pre-FFN input X:** **partially.** Post-norm, not pre-norm. Same shape-identity mismatch with 31CD as Candidate A.
- **Before or after layer norm:** **AFTER** layer norm.
- **Before ffn_up / ffn_gate matmuls:** **YES.**
- **Risk of interacting with dormant PRT sidecar:** **LOW-MEDIUM.** The call site is in `qwen.cpp`, not inside `build_ffn`. The sidecar machinery operates inside `build_ffn` (and `build_attn` via `build_prt_true_attn_out_injection`). If `SDI_CAPTURE_FFN_INPUT=1` is set, the hook fires before `build_ffn` is called, so the dormant sidecar is never reached (the early-return path doesn't touch the sidecar).
- **Can be gated cleanly:** **YES.** The `cb_func` callback registry is a clean per-graph observer mechanism, not a sidecar-specific path. Gate on `SDI_CAPTURE_FFN_INPUT=1` and `il == SDI_CAPTURE_LAYER`.
- **Can write a single layer/token only:** **YES** (gate on `il == SDI_CAPTURE_LAYER`; token filter on the serialize step).
- **Expected activation shape:** `[N_tokens, 1536]` for Qwen2.5-1.5B at L0 (post-norm).
- **Implementation risk:** **LOW-MEDIUM.** Adding one `cb()` call between lines 74 and 76 of `qwen.cpp` is a 1-line change. The serialize step requires reading `ggml_get_data()` post-execution, which is a standard debug-callback pattern.
- **Validation risk:** **MEDIUM.** Same post-norm shape-identity issue as Candidate A.
- **Recommendation:** **MAYBE / REJECT as primary.** Better than A and B on source-level sidecar interference, but the post-norm shape mismatch with 31CD is a real validation cost. 31CF's own evaluation ranked this as "MEDIUM-HIGH feasibility without design phase" but did not flag the post-norm vs pre-norm issue (their wording at SOT line 982: "the `cur` tensor passed into `build_ffn` is the canonical pre-FFN hidden state (post-attention, post-RMSNorm)" — they correctly identified post-RMSNorm but then mis-stated the relationship to 31CD's pre-norm capture, which is a documentation gap in 31CF, not a source-level ambiguity).

### 4.4 Candidate D — diagnostic callback / named tensor capture after `ffn_norm` and before `ffn_up` / `ffn_gate` matmuls, via ggml graph callback

- **File / function / line range:** hypothetical — would require either a new `cb(cur, "ffn_norm_post", il)` insertion between `qwen.cpp:74` and `qwen.cpp:76`, OR a new ggml-level callback mechanism
- **What tensor would be captured:** the post-norm `cur` (functionally identical to Candidate C)
- **Whether definitely pre-FFN input X:** **NO.** Post-norm.
- **Before or after layer norm:** **AFTER.**
- **Before ffn_up / ffn_gate matmuls:** **YES.**
- **Risk of interacting with dormant PRT sidecar:** **LOW-MEDIUM.** Same as Candidate C.
- **Can be gated cleanly:** **YES.**
- **Can write a single layer/token only:** **YES.**
- **Expected activation shape:** `[N_tokens, 1536]` (post-norm).
- **Implementation risk:** **LOW-MEDIUM** (functionally identical to Candidate C, just at a different name).
- **Validation risk:** **MEDIUM** (same post-norm issue).
- **Recommendation:** **MAYBE / equivalent to Candidate C.** No advantage over C; functionally the same hook at a different name.

### 4.5 Candidate E — use the existing `cb(ffn_inp, "ffn_inp", il)` callback at `qwen.cpp:67` (already in source)

- **File / function / line range:** `~/llama.cpp/src/models/qwen.cpp` line 66-67, **already in source, no patch needed for the callback itself**
  ```cpp
  ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);  // line 66
  cb(ffn_inp, "ffn_inp", il);                          // line 67
  ```
- **What tensor would be captured:** the **pre-norm** `ffn_inp` = the post-attention-residual-add hidden state at the L0 boundary. **This is the same semantic tensor position as 31CD's HF forward-hook capture** (the `hook_input` to `mlp` in HF safetensors terms) — i.e. the pre-FFN input activation X entering the selected layer MLP. It does **NOT** imply identical activation values: 31CD/31CE used HF-derived activation proxies (HF safetensors bf16 forward-hook on `mdl.model.layers[0].mlp`), while a future 31CF-S implementation would capture exact Q4_K_M GGUF runtime activations via llama.cpp instrumentation. The two captures are at the **same semantic tensor position** but are **numerically different methods**; the comparison is **contextual only, not an equivalence claim**.
- **Whether definitely pre-FFN input X:** **YES** (at the same semantic tensor position as 31CD's `hook_input`, i.e. the input to the MLP at the L0 layer boundary). Numerical values will differ because 31CD captured HF safetensors bf16 forward-hook activations and a future 31CF-S would capture exact Q4_K_M GGUF runtime activations; the two are not bit-equal and are not claimed to be equivalent.
- **Before or after layer norm:** **BEFORE** layer norm (the RMSNorm is applied at `qwen.cpp:71`, after the callback at line 67).
- **Before ffn_up / ffn_gate matmuls:** **YES** (matmuls are inside `build_ffn`, which is called at `qwen.cpp:76`).
- **Risk of interacting with dormant PRT sidecar:** **LOW.** The callback fires at `qwen.cpp:67`, which is **outside** `build_ffn` (the sidecar is inside `build_ffn` and inside `build_attn` via `build_prt_true_attn_out_injection`). The sidecar's `attn_out` injection (line 1342) could in principle contaminate `inpSA` (the residual stream before line 66's `ggml_add`), but **only if** the sidecar is active (`#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL` is defined and the build is configured to use it). In the operator's current binary, the sidecar is fully dormant. **To eliminate residual risk:** the gating design must include a check that the future 31CF-S build does NOT add `-DPRT_SIDECAR_PAGER_EXPERIMENTAL` (see §6).
- **Can be gated cleanly:** **YES.** The `cb_func` callback registry is a per-graph observer mechanism. The proposed gating adds an `SDI_CAPTURE_FFN_INPUT=1` env-var check inside a registered `cb_func` that filters on `name == "ffn_inp"` and `il == SDI_CAPTURE_LAYER`. The actual data read+serialize happens post-`llama_decode()` via `ggml_get_data()` on the captured `ggml_tensor *`.
- **Can write a single layer/token only:** **YES** (gate on `il == SDI_CAPTURE_LAYER`; token filter `[-1]` for last prefill token, matches 31CD).
- **Expected activation shape:** `[N_tokens, 1536]` for Qwen2.5-1.5B at L0 with "The capital of France is" (6 tokens) → `[1, 1536]` for batch=1, last prefill token. **Shape matches 31CD's `[1, 1536]` exactly.** Like-for-like comparison is possible (with documented caveat: HF bf16 vs GGUF Q4_K_M, not bit-equal but same sign pattern).
- **Implementation risk:** **LOW.** The callback already exists in source. The patch is:
  1. Register a `cb_func` (operator-side, in a small `.cpp` file compiled into the operator's `llama-server` build) that filters on `name == "ffn_inp"` and `il == SDI_CAPTURE_LAYER` and stores the `ggml_tensor *` pointer to a process-global.
  2. After `llama_decode()` returns, the registered function (or a separate small driver) checks the env-var gates, calls `ggml_get_data()` on the captured tensor, writes a shape/dtype header + raw f32 to `SDI_CAPTURE_OUT`, and exits or continues per the design's "exit or continue" policy.
  3. No edits to `qwen.cpp`, no edits to `llama-graph.cpp`, no edits to `build_ffn`, no new public API.
- **Validation risk:** **LOW.** The captured tensor would be at the **same semantic tensor position as 31CD** (the input to the L0 MLP, shape `[1, 1536]`), but the comparison to 31CD's HF forward-hook capture is **contextual, not bit-equal, and not an equivalence claim**: 31CD captured HF safetensors bf16 forward-hook activations (Option A proxy); a future 31CF-S would capture exact Q4_K_M GGUF runtime activations (Option B). The validation asserts **same sign pattern + same order of magnitude + per-element absolute differences in `[1e-3, 1e-1]`** — these are **contextual indicators of agreement at the same tensor position**, NOT a claim that the activation values are identical or that Option A equals Option B.
- **Recommendation:** **ACCEPT (PRIMARY).** This is the recommended hook for any future 31CF-S implementation.

### 4.6 Comparison summary

| Candidate | Pre-FFN X identity | Shape matches 31CD `[1, 1536]` | Sidecar risk | Gating clean? | Recommendation |
|---|---|---|---|---|---|
| A (`build_ffn` entry) | post-norm, mismatch | shape matches, values differ | HIGH | MEDIUM | REJECT |
| B (`prt_get_residual_view`) | NO (residual, not activation) | N/A | MAXIMUM | NO | REJECT |
| C (cb before `build_ffn` at `qwen.cpp:76`) | post-norm, mismatch | shape matches, values differ | LOW-MEDIUM | YES | MAYBE / REJECT (post-norm issue) |
| D (hypothetical `cb("ffn_norm_post", il)`) | post-norm, mismatch | shape matches, values differ | LOW-MEDIUM | YES | equivalent to C |
| **E (existing `cb("ffn_inp", il)` at `qwen.cpp:67`)** | **YES** (same semantic tensor position as 31CD: pre-FFN input X) | **YES** (shape `[1, 1536]` matches 31CD's capture shape) | **LOW** (with build-gate) | **YES** | **ACCEPT (PRIMARY)** |

**Why E is the winner:** it is the **only candidate** that captures the **same semantic tensor position as 31CD/31CE** — the pre-FFN input activation X entering the selected layer MLP — with **low sidecar risk** (callback fires outside `build_ffn`, and the build-gate eliminates the sidecar), and **clean gating** (existing `cb_func` observer mechanism, no new public API, default-off via env-var). 31CF missed this candidate because they focused on the `cur` at `build_ffn` entry (Candidate A) and the sidecar machinery inside `build_ffn`; they did not evaluate the already-existing `cb(ffn_inp, "ffn_inp", il)` at `qwen.cpp:67`.

---

## 5. Gating design (design only — no implementation)

### 5.1 Compile-time / build-time gates (all default off)

| Gate | Default | Override | Purpose |
|---|---|---|---|
| `-DSDI_CAPTURE_ENABLED=OFF` (CMake option) | OFF | `-DSDI_CAPTURE_ENABLED=ON` | Master switch; off → no SDI capture code compiled in, zero runtime cost |
| `!defined(SDI_CAPTURE_ENABLED)` short-circuit in source | n/a | n/a | If the CMake option is OFF, all SDI capture code paths are `#ifdef`-stripped at compile time |
| `!defined(PRT_SIDECAR_PAGER_EXPERIMENTAL)` build requirement | required | n/a | The build MUST NOT define `PRT_SIDECAR_PAGER_EXPERIMENTAL`. If it does, the SDI capture is auto-disabled (the SDI_CAPTURE_ENABLED block returns early with a one-time stderr warning) |

### 5.2 Runtime gates (all default off, all env-var-controlled)

| Env var | Default | Example | Purpose |
|---|---|---|---|
| `SDI_CAPTURE_FFN_INPUT` | unset | `1` | Master runtime switch for the ffn_inp capture path. Unset → no capture. |
| `SDI_CAPTURE_LAYER` | unset | `0` | Layer id to capture. Unset → no capture. Must match `il` of the targeted layer. |
| `SDI_CAPTURE_TOKEN` | `last` | `0` / `1` / `2` / `last` / `all` | Token position to capture. `last` = last prefill token (matches 31CD). Numeric = specific token index. `all` = entire prefill sequence (default-off behavior; see §5.3). |
| `SDI_CAPTURE_OUT` | unset | `/tmp/phase31cf_x.bin` | Output path. Unset → no capture. Path is opened with `O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW`, written atomically (`write()` + `fsync()` + `rename()`), permissions `0600`. |
| `SDI_CAPTURE_EXIT_AFTER` | unset | `1` | If set, exit the process after the capture write completes. Default unset → continue (for use with `llama-server` for HTTP-driven capture). |

### 5.3 Default-off / no-always-on-dumping guarantees

- All gates default to off. The unset state of any required gate → no capture, no file write, zero I/O.
- The `cb_func` registration is **always** active (it is a passive observer; cost is one pointer-dereference per `cb()` call, negligible). The env-var gates determine whether the observer **records** anything.
- No all-layer dumping by default: `SDI_CAPTURE_LAYER` must be set explicitly, and the observer only records the single layer.
- No all-token dumping by default: `SDI_CAPTURE_TOKEN=last` (or a numeric value) is the default; `SDI_CAPTURE_TOKEN=all` requires explicit opt-in and is intended for diagnostic use only.
- No raw activation arrays in repo: writes go to `SDI_CAPTURE_OUT` (operator-controlled, e.g. `/tmp/...`); no SDI repo path receives raw arrays.

### 5.4 Output format (design only)

`SDI_CAPTURE_OUT` is a binary file with a small header + raw float32 data:

```
Offset  Size  Field
0       4     magic: "SDIX" (0x58494453, little-endian)
4       4     version: 0x00000001
8       4     dtype: 0x00000001 = float32
12      4     n_dim: 2
16      8     dim[0]: batch dim (e.g. 1)
24      8     dim[1]: hidden dim (e.g. 1536 for Qwen2.5-1.5B)
32      8     token_position: index of captured token in prefill (e.g. 5 for last of 6)
40      8     il: layer id
48      8     shape_logical: product of dim[0..n_dim-1] (e.g. 1536)
56      4     prompt_sha256_first4: first 4 bytes of SHA256 of prompt bytes (for traceability; no PII)
60      4     reserved: 0
64      N     raw float32 data, N = shape_logical * 4 bytes
```

The runner that consumes this file (the future 31CF-S Phase 2) reads the header, validates `magic == "SDIX"`, `version == 1`, `dtype == float32`, `n_dim == 2`, then reads `N` float32 values into a `[1, 1536]` array for replay through the standalone pipeline. The replay uses the same 31CD pipeline: W_ref = local 1.5B Q4_K_M GGUF dequantized, W_low = corrected_ceil_per_row Q2_K, SDIR = ffn_up + ffn_gate k=0.5%, no ffn_down residual, metrics `cos_low, cos_sub, delta_cos, MAE_low, MAE_sub, MAE_delta, finite, severe, memory margin`.

### 5.5 How to prove the hook is inactive by default

Three verifications, all part of a future 31CF-S implementation phase (not 31CF-R):

1. **Build-gate verification:** the operator builds `~/llama.cpp/build/bin/llama-server` with `-DSDI_CAPTURE_ENABLED=OFF`. Run `nm` on the binary; the SDI capture symbols are absent.
2. **Env-var-gate verification:** run `llama-server` with no `SDI_*` env vars set. Observe: no `SDIX` files written, no `[SDI-CAPTURE]` log lines, zero I/O on the SDI capture path. Verified by `strace -e trace=write,open,openat` showing no writes to any `SDIX` paths.
3. **Opt-in verification:** set `SDI_CAPTURE_FFN_INPUT=1`, `SDI_CAPTURE_LAYER=0`, `SDI_CAPTURE_OUT=/tmp/test.bin`. Run a single token prefill of "The capital of France is". Observe: a single `/tmp/test.bin` file appears with the SDIX header + 1536 float32 values, SHA256 matches expected, no other files written. The capture is **single-shot** (the `cb_func` clears its recorded pointer after the first matching event, so subsequent layers/tokens are ignored).

---

## 6. Validation plan (for a future 31CF-S implementation phase, not 31CF-R)

### 6.1 Preflight (must run before any capture)

- `python3 -m tests.run_source_of_truth_regression` → must be `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=true`. If regression fails, stop.

### 6.2 Build confirmation (must run before any capture)

- Confirm `~/llama.cpp/build/bin/llama-server` is the operator's local build (not atomic chat's, not system).
- Confirm `nm` shows no PRT sidecar symbols AND no SDI capture symbols (i.e. clean default build).
- Confirm CMake invocation does **not** include `-DPRT_SIDECAR_PAGER_EXPERIMENTAL` and does **not** include `-DSDI_CAPTURE_ENABLED=ON` (this is the default-off build).

### 6.3 Hook disabled by default (must verify before any capture)

- Run `llama-server` with no `SDI_*` env vars. Verify via `strace` that no `SDIX` files are written. Stop if any capture occurs.

### 6.4 Hook enabled only for selected layer/token (must verify)

- Set `SDI_CAPTURE_FFN_INPUT=1`, `SDI_CAPTURE_LAYER=0`, `SDI_CAPTURE_TOKEN=last`, `SDI_CAPTURE_OUT=/tmp/phase31cfs_p0_l0.bin`.
- Run a single prefill of "The capital of France is" (6 BPE tokens) at L0 with batch=1.
- Verify the output file: magic="SDIX", version=1, dtype=float32, n_dim=2, dim[0]=1, dim[1]=1536, token_position=5, il=0, N=1536 float32 values.

### 6.5 Captured shape must be `[1, 1536]`

- Per §6.4 verification.

### 6.6 Captured dtype must be f32 or explicitly convertible to f32

- Per §6.4 verification. The header records `dtype=0x00000001` (float32). If the operator's build ever uses a different dtype (e.g. bfloat16), the design should record `dtype=0x00000002 = bfloat16` and the runner converts to f32 at load time.

### 6.7 Activation finite check

- Read the 1536 float32 values; verify `np.isfinite(X).all() == True`. If any non-finite value (NaN, Inf), stop and report.

### 6.8 SHA256 hash of raw X

- Compute `sha256(X.tobytes())` and record in the result JSON.

### 6.9 Replay through Q4_K_M GGUF W_ref / corrected Q2_K / SDIR pipeline

- Replay the captured X through the 31CD standalone pipeline: W_ref = local 1.5B Q4_K_M GGUF dequantized, W_low = corrected_ceil_per_row Q2_K, SDIR = ffn_up + ffn_gate k=0.5%, no ffn_down residual.
- Compute metrics: `cos_low, cos_sub, delta_cos, MAE_low, MAE_sub, MAE_delta, finite, severe, memory margin`.

### 6.10 Comparison to 31CD/31CE P0-L0 (context only, NOT a pass/fail gate)

- 31CD's HF forward-hook capture (pre-norm, bf16→f32) is the **closest-available ground truth** for the activation X. It is **NOT** bit-equal to a GGUF Q4_K_M capture (HF bf16 vs GGUF Q4_K_M dequantization are numerically different).
- Expected: the captured X should have the **same sign pattern** and **same order of magnitude** as 31CD's `hook_input`, with **per-element absolute differences** in the range of `[1e-3, 1e-1]` (typical bf16 vs Q4_K_M numerical drift).
- The validation is **contextual, not a pass/fail**. Any 31CF-S result will need to explicitly document the cross-validation method and the documented caveat.

### 6.11 No claim of Option A / HF equivalence

- 31CF-R and 31CF-S do **not** claim that Option A (HF bf16 forward-hook) and Option B (GGUF Q4_K_M instrumentation) produce identical activation values. The 31CC 5-strategy table explicitly distinguishes them as different methods. 31CF-S will be the **first** capture via Option B and serves as a **new ground truth**, not a replication of Option A.

### 6.12 No generation quality claim

- 31CF-R and 31CF-S do **not** claim any model generation quality, behavior recovery, or inference quality. The capture is purely an activation tensor for offline replay.

### 6.13 No runtime performance claim

- 31CF-R and 31CF-S do **not** claim any runtime speedup, memory savings at the full-model level, or production readiness.

### 6.14 No raw activation arrays committed

- The capture file is written to `SDI_CAPTURE_OUT` (operator-controlled, e.g. `/tmp/...`). It is **never** committed to the sdi-substitutive repo. The result JSON records `sha256(X.tobytes())`, `captured_shape`, `dtype`, `token_position`, `il`, and the replay metrics — all summary metadata, no raw arrays.

### 6.15 Post-edit regression

- After any future 31CF-S implementation, run `python3 -m tests.run_source_of_truth_regression` again. Must remain `PASS_SOURCE_OF_TRUNTIME_CLEAN`. If it fails, stop and report.

---

## 7. Patch outline (design only, no actual diff)

### 7.1 Files that would be modified in a future 31CF-S phase

| File | Modification |
|---|---|
| `~/llama.cpp/CMakeLists.txt` | Add `option(SDI_CAPTURE_ENABLED "Enable SDI activation capture (default OFF)" OFF)` |
| `~/llama.cpp/src/sdi_capture.h` (new file, small) | Declare `sdi_capture_register_cb_func()`, `sdi_capture_get_recorded_tensor()`, `sdi_capture_serialize_if_ready()`. Guarded by `#ifdef SDI_CAPTURE_ENABLED`. |
| `~/llama.cpp/src/sdi_capture.cpp` (new file, small, ~80 lines) | Implement the registered `cb_func`, the env-var gates, the ggml_get_data read, the SDIX file write. Guarded by `#ifdef SDI_CAPTURE_ENABLED`. Compiled into the `llama-server` binary when `SDI_CAPTURE_ENABLED=ON`. |
| `~/llama.cpp/examples/llama-server/server.cpp` (or equivalent entry point) | Call `sdi_capture_register_cb_func()` once at server startup. Guarded by `#ifdef SDI_CAPTURE_ENABLED`. |

**No edits to `qwen.cpp`, `llama-graph.cpp`, or `build_ffn`.** The existing `cb(ffn_inp, "ffn_inp", il)` at `qwen.cpp:67` is the observation point; the `cb_func` filter is the recording point.

### 7.2 New helper function names (declarations only, not implementations)

```cpp
// sdi_capture.h
#ifdef SDI_CAPTURE_ENABLED
void sdi_capture_register_cb_func(llama_graph_context & gctx);
ggml_tensor * sdi_capture_get_recorded_tensor();
int sdi_capture_serialize_if_ready(const char * prompt_sha256_first4);
#endif
```

### 7.3 Where the gated hook would be inserted

In `sdi_capture.cpp`:
- `sdi_capture_cb_func(ggml_tensor * cur, const char * name, int il)`:
  - if `name == "ffn_inp"` and `il == SDI_CAPTURE_LAYER` (parsed from env) and `g_recorded_tensor == nullptr`: store `cur` to `g_recorded_tensor`; set `g_recorded_token_position = N_tokens - 1` (last prefill token by default)
  - else: ignore
- `sdi_capture_serialize_if_ready()`:
  - if `g_recorded_tensor == nullptr`: return 0 (not ready)
  - if `SDI_CAPTURE_OUT` is unset: return 0 (no output path)
  - read env vars: `SDI_CAPTURE_FFN_INPUT`, `SDI_CAPTURE_LAYER`, `SDI_CAPTURE_TOKEN`, `SDI_CAPTURE_OUT`, `SDI_CAPTURE_EXIT_AFTER`
  - if any required gate is unset/zero: return 0 (gates not satisfied)
  - call `ggml_get_data(g_recorded_tensor)` to get the raw float32 pointer
  - if `ggml_nbytes(g_recorded_tensor) != 4 * shape_logical`: log error, return -1 (size mismatch)
  - open `SDI_CAPTURE_OUT` with `O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW`, mode 0600
  - write the 64-byte SDIX header
  - write the raw float32 data (`N * 4` bytes)
  - `fsync(fd)`; `close(fd)`; `rename(tmp, final)` (atomic)
  - if `SDI_CAPTURE_EXIT_AFTER=1`: `exit(0)`
  - return 1 (capture complete)

### 7.4 How output would be serialized

Per §5.4. Atomic write via `write()` to a temp file + `fsync()` + `rename()` to the final path. Permissions 0600. Path is operator-controlled.

### 7.5 How it would be disabled by default

- `SDI_CAPTURE_ENABLED=OFF` in CMake (default) → the `sdi_capture.cpp` and `sdi_capture.h` files are not compiled into the binary. `nm` shows no SDI capture symbols.
- Even with `SDI_CAPTURE_ENABLED=ON`, the env-var gates are all unset by default. The `sdi_capture_cb_func` registers a pointer but the filter logic short-circuits on missing env vars, recording nothing. The serialize function short-circuits on missing gates, writing nothing.

### 7.6 How the future runner would call it

A small driver script (Python or bash) that:
1. Sets `SDI_CAPTURE_FFN_INPUT=1`, `SDI_CAPTURE_LAYER=0`, `SDI_CAPTURE_TOKEN=last`, `SDI_CAPTURE_OUT=/tmp/phase31cfs_p0_l0.bin`.
2. Spawns `~/llama.cpp/build/bin/llama-server` with the GGUF model loaded and the 31CD/31CE prompt as input.
3. Waits for `/tmp/phase31cfs_p0_l0.bin` to appear (or for a `[SDI-CAPTURE-COMPLETE]` log line, whichever comes first).
4. Sends `SIGTERM` to the server.
5. Reads the SDIX file, validates header, computes SHA256, replays through the 31CD pipeline, records metrics in `src/results/PHASE31CFS_GGUF_RUNTIME_ACTIVATION_CAPTURE.json`.

### 7.7 How future artifacts would be recorded

- `docs/PHASE31CFS_GGUF_RUNTIME_ACTIVATION_CAPTURE.md` (human-readable report)
- `src/phase31cfs_gguf_runtime_activation_capture.py` (the runner script)
- `src/results/PHASE31CFS_GGUF_RUNTIME_ACTIVATION_CAPTURE.json` (metrics + SHA256 + replay results)

All three committed to the sdi-substitutive repo **only** with explicit operator approval. No raw activation arrays in the repo.

---

## 8. Why no implementation was attempted in 31CF-R

Per Matt's explicit instruction in the 31CF clarification (recorded in SOT Section 3, 31CF entry): *"Do not patch llama.cpp yet. Do not rebuild llama.cpp yet. Do not attempt activation capture yet."* Also consistent with `~/llama.cpp/AGENTS.md` content-blocker policy: *"Proceed only when confident the contributor can explain the changes to reviewers independently"*. 31CF-R is the design-only phase that produces the design documentation the operator (and a future AI agent) can review and approve. Implementation belongs in a future 31CF-S phase, only after explicit operator approval of the 31CF-R design.

---

## 9. Allowed claims (for 31CF-R)

- A design for a llama.cpp GGUF FFN-input hook has been produced and reviewed in this document
- The recommended hook (Candidate E, `cb(ffn_inp, "ffn_inp", il)` at `qwen.cpp:67`) is **already in source** and requires no source patch to the existing callback
- The dormant PRT sidecar machinery is **inactive at the binary level** in the operator's current `~/llama.cpp/build/bin/llama-server` (verified via `strings | grep` returning 0 PRT symbols)
- The recommended hook captures the **pre-norm, post-attention-residual-add hidden state** at the L0 boundary, which is the **same semantic tensor position as 31CD/31CE's pre-FFN input X** (the input to the MLP at the selected layer boundary, shape `[1, 1536]` for Qwen2.5-1.5B at L0 with "The capital of France is"). It does **NOT** imply that 31CF-S would capture identical activation values to 31CD/31CE's HF forward-hook: 31CD/31CE used HF-derived activation proxies (HF safetensors bf16 forward-hook on `mlp`), while a future 31CF-S would capture exact Q4_K_M GGUF runtime activations via llama.cpp instrumentation. The two methods target the same semantic tensor position but are numerically different; the comparison is contextual only, not an equivalence claim.
- A future 31CF-S implementation can be gated at three levels: CMake option (default OFF), env vars (all unset by default), and the existing `cb_func` callback filter (which records nothing unless the gates are satisfied)

---

## 10. Forbidden claims (for 31CF-R)

Per the SOT Section 0.A master list (13 items) + 31CF-R specifics:

1. **no exact activation capture claim** — 31CF-R is design-only; no capture has been performed
2. **no replay metric claim** — no replay has been performed
3. **no hook implementation claim** — the design proposes a hook but no implementation exists
4. **no build success claim** — no build was performed or attempted
5. **no runtime integration claim** — no runtime integration exists
6. **no generation quality claim** — out of scope
7. **no performance claim** — out of scope
8. **no production readiness claim** — out of scope
9. **no claim that a candidate hook is correct unless proven by source analysis** — Candidate E's correctness is supported by source analysis (the callback exists, the tensor is pre-norm, the shape matches 31CD) but the analysis is design-stage, not runtime-verified
10. **no claim that Option A HF activations equal Option B GGUF activations** — 31CC explicitly distinguishes them; 31CF-R does not claim equivalence
11. **no model files / raw activation arrays / build artifacts committed** — none have been generated; SDIX file path is operator-controlled, not in the repo
12. **no commit, push, tag without explicit Matt approval** — 31CF-R stops at PRE-COMMIT REPORT
13. **no model quality / behavior / speed / runtime / inference / production claim** — SOT 0.A items 1-10

---

## 11. Limitations

- **The Candidate E correctness is design-stage, not runtime-verified.** A future 31CF-S implementation must build the hook, run a capture, and verify the SDIX header + replay metrics before any claim of "the hook works at runtime" can be made.
- **The cross-validation against 31CD/31CE is contextual, not bit-equal.** HF bf16 forward-hook and GGUF Q4_K_M instrumentation are numerically different methods; same-sign-pattern is the most that can be claimed.
- **The hook captures a single layer/token per process lifetime.** A multi-layer or multi-token sweep would require multiple process invocations, each with a different `SDI_CAPTURE_LAYER` / `SDI_CAPTURE_TOKEN` value. This is by design (single-shot capture prevents accidental bulk dumps).
- **The operator's `~/llama.cpp/` binary must be built without `-DPRT_SIDECAR_PAGER_EXPERIMENTAL` for the design to be valid.** If a future build adds this flag, the SDI capture is auto-disabled (§5.1).
- **The 31CF blocker is NOT reclassified by this design.** 31CF is recorded as `BLOCKED_31CF_HOOK_POINT_AMBIGUOUS`. 31CF-R produces a design that *would* unblock a future 31CF-S implementation, but the BLOCKED classification remains until 31CF-S (or any subsequent implementation phase) actually performs a capture and validates it.

---

## 12. Next allowed phases (after 31CF-R)

Per SOT Section 0 line 11: after 31CF-R, the allowed next phases are:

1. **Phase 31CF-S — llama.cpp GGUF FFN-Input Diagnostic Hook Implementation**, only if explicitly requested and Matt approves:
   - source modification to `~/llama.cpp/`
   - rebuild of `~/llama.cpp/`
   - capture of the FFN/MLP input activation X for Qwen2.5-1.5B Q4_K_M
   - replay through the 31CD/31CE standalone pipeline
   - documentation of the capture + replay in `docs/PHASE31CFS_*` and `src/results/PHASE31CFS_*.json`

2. **Phase 31CF-R2 — additional hook-point diagnosis**, only if the 31CF-R design is rejected or additional candidates need to be evaluated. (Not currently indicated; the Candidate E recommendation is strong.)

3. **Phase 31CG — Option A Larger Prompt/Token Sensitivity Planning**, only if the operator decides to defer GGUF runtime instrumentation and instead expand the Option A 31CD/31CE prompt/layer matrix. (Planning-only.)

4. **Phase 31CH — Runtime Artifact Format / Loader Planning**, only if the operator decides to defer the GGUF runtime instrumentation and instead design the on-disk format for shipping W_low + R at scale. (Planning-only.)

All four require explicit operator approval at entry. 31CF-R produces no automatic progression.

---

## 13. End of design document

This document is the deliverable for Phase 31CF-R. It is **design-only**, **read-only** with respect to `~/llama.cpp/`, and **produces no claim of runtime correctness**. Implementation belongs in a future phase (most likely 31CF-S) with explicit operator approval.

See also:
- `src/results/PHASE31CF_R_HOOK_POINT_DIAGNOSIS_PATCH_DESIGN.json` — machine-readable summary
- `SOURCE_OF_TRUTH.md` Section 0 line 11, Section 3 (31CF-R entry to be added by SOT update), Section 9
- `docs/PHASE31CF_GGUF_RUNTIME_ACTIVATION_CAPTURE_BLOCKED.md` — the BLOCKED 31CF phase this design follows from
- `docs/PHASE31CD_REAL_ACTIVATION_MICRO_PROBE_OPTION_A.md` — the ground-truth replay pipeline
- `docs/PHASE31CE_REAL_ACTIVATION_OPTION_A_MULTIPROMPT_MULTILAYER.md` — the multi-prompt/multi-layer extension
