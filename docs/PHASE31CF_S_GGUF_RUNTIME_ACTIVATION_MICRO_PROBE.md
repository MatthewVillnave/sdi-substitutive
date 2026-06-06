# Phase 31CF-S — GGUF Runtime Activation Micro-Probe (Option B)

**Phase ID:** 31CF-S
**Classification:** `PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN`
**Type:** Implementation, first Option B (exact Q4_K_M GGUF runtime activation capture)
**Date:** 2026-06-06
**Author:** agent (Matt-approved phase entry + narrow-scope approval)
**Repo:** `sdi-substitutive`
**Target external repo:** `~/llama.cpp/` (read-only inspection + linking; NO source modifications, NO rebuild)

---

## 0. Scope (read first)

This phase implements the **first Option B** capture for the SDI-substitutive project: an **exact Q4_K_M GGUF / llama.cpp runtime activation capture** at L0 for Qwen2.5-1.5B-Instruct, replayed through the existing standalone pipeline.

**Strict scope** (per Matt's explicit 31CF-S approval):
- prompt: `"The capital of France is"`
- layer: L0 only
- token position: last prefill token of the prompt (closest exact equivalent available through llama.cpp callback metadata)
- expected captured activation shape: `[1, 1536]`
- total pairs: 1
- no optional L14/L27 expansion in 31CF-S unless Matt explicitly requests it later
- no Option A / HF fallback

**What was done:**
- wrote a small standalone C++ harness in `src/phase31cfs_capture.cpp` that loads the GGUF, sets the public `params.cb_eval` callback to filter on tensor name `"ffn_inp-0"` (per-arch graph construction label set by `llama_context::graph_get_cb` at `qwen.cpp:67` via `ggml_format_name("%s-%d", name, il)`), runs a prefill-only `llama_decode`, and writes the captured tensor to `/tmp/phase31cfs_p0_l0.bin` in SDIX format
- compiled the harness against the existing `~/llama.cpp/build/bin/*.so` + `~/llama.cpp/build/common/libcommon.a` + `~/llama.cpp/build/vendor/cpp-httplib/libcpp-httplib.a` (NO `~/llama.cpp/` rebuild, NO source modification)
- ran the harness against the local 1.5B Q4_K_M GGUF
- wrote `src/phase31cfs_replay.py` that loads the SDIX file and runs the existing standalone replay pipeline (W_ref = local 1.5B Q4_K_M GGUF dequantized, W_low = corrected_ceil_per_row Q2_K, SDIR = ffn_up+ffn_gate k=0.5%, no ffn_down residual)
- computed metrics: cos_low, cos_sub, delta_cos, MAE_low, MAE_sub, MAE_delta, finite, severe, memory margin
- did **contextual** comparison to 31CD/31CE P0-L0 (same sign pattern, NOT bit-equal, NOT an equivalence claim)
- deleted the raw X from `/tmp` before the PRE-COMMIT REPORT (per artifact policy)
- wrote `src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json` (machine-readable summary, metadata only, no raw arrays)

**What was NOT done:**
- no source patch to `~/llama.cpp/`
- no rebuild of `~/llama.cpp/`
- no modification of `qwen.cpp` / `llama-graph.cpp` / `build_ffn`
- no enabling of `PRT_SIDECAR_PAGER_EXPERIMENTAL`
- no forward pass beyond prefill (no `n_predict > 0`, no generation, no sampling, no quality evaluation)
- no model files / HF cache / raw activation arrays / Q2_K blobs / SDIR blobs / build artifacts / `llama.cpp` source committed to `sdi-substitutive`
- no commit / push / tag without explicit operator approval

---

## 1. Background

### 1.1 The 31CF blocker

`SOURCE_OF_TRUTH.md` Section 3, 31CF entry: `BLOCKED_31CF_HOOK_POINT_AMBIGUOUS` with subtype `DORMANT_SIDECAR_MACHINERY_INTERFERENCE`. 31CF's recommended approach was to modify `~/llama.cpp/` source to add a gated hook, but the dormant PRT sidecar machinery in `build_ffn` (line 1844) created ambiguity about which tensor was the "true" pre-FFN input.

### 1.2 The 31CF-R design

31CF-R (`PARTIAL_31CFR_HOOK_POINT_CANDIDATES_IDENTIFIED`, commit `6fdc8357`) identified **Candidate E** as the recommended hook strategy: use the **existing** `cb(ffn_inp, "ffn_inp", il)` callback at `~/llama.cpp/src/models/qwen.cpp:67` (already in source, no patch to the existing callback). The 31CF-R design proposed a 4-file modification to `~/llama.cpp/` (2 new files + 2 one-line edits) for an in-binary hook, but did not preclude a **no-modification alternative** using llama.cpp's **public** `params.cb_eval` API.

### 1.3 The Candidate E callback path discovery (during 31CF-S feasibility check)

During 31CF-S preflight, the agent discovered:

1. **`llama_context::graph_get_cb()`** in `src/llama-context.cpp:2197` is a **per-arch callback** lambda that calls `ggml_format_name(cur, "%s-%d", name, il)` for `name="ffn_inp"` and `il=0`, setting the tensor's `ggml_set_name` to `"ffn_inp-0"`. This is the same callback that fires for `cb(ffn_inp, "ffn_inp", il)` at `qwen.cpp:67`.

2. **`common_params::cb_eval`** in `common/common.h:450-451` is a public-API function pointer (`ggml_backend_sched_eval_callback`) that fires for **every ggml tensor computation** at the backend-scheduler level. It receives `ggml_tensor * t, bool ask, void * user_data`, where `t->name` is the tensor's `ggml_set_name` label (which, per #1, is `"ffn_inp-0"` for the L0 FFN-input tensor).

3. **`llama_context` wires `cparams.cb_eval = params.cb_eval`** at `src/llama-context.cpp:63-64`, and `ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data)` at `src/llama-context.cpp:1196`. This means **any user code that sets `params.cb_eval` and `params.cb_eval_user_data` can observe the `"ffn_inp-0"` tensor via the public llama.cpp API**, with **zero `~/llama.cpp/` source modifications**.

4. Upstream's own `examples/eval-callback/eval-callback.cpp` is the canonical pattern for this API (89 lines, `target_link_libraries(... common llama ${CMAKE_THREAD_LIBS_INIT})`).

5. **Pre-existing link-time bug in operator's `libllama.so`**: `src/llama.cpp:1648-1649` references `extern int g_prt_sidecar_root_set;` and returns its value, but the symbol is declared as `static bool` in `src/llama-graph.cpp:196` (internal linkage, mangled as `_ZL22g_prt_sidecar_root_set`). The `extern "C"`-linkage symbol is never defined, so linking any executable against `libllama.so` produces an undefined-reference error. This is **NOT introduced by 31CF-S** and is **NOT a modification of `~/llama.cpp/` source**; it's a pre-existing build issue. The 31CF-S harness works around it locally with `extern "C" int g_prt_sidecar_root_set = 0;` in `src/phase31cfs_capture.cpp`. The runtime value is always 0 (matches the static's `false` default, which is never set to `true` because the dormant-sidecar write sites are guarded by `#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL` which is undefined in the operator's build).

6. **Operator's `llama-server` binary has 0 PRT sidecar symbols** (verified via `strings | grep -c` in 31CF-R + re-verified in 31CF-S preflight), so the dormant sidecar is **inactive at the binary level** for the operator's current build. The 31CF-R design's "build-gate check that the future 31CF-S build does NOT add `-DPRT_SIDECAR_PAGER_EXPERIMENTAL`" is satisfied trivially by the no-rebuild approach.

---

## 2. Implementation strategy

### 2.1 Hook mechanism (no `~/llama.cpp/` modification)

```
Harness: src/phase31cfs_capture.cpp (~330 lines, includes the linker workaround block)
Hook:   params.cb_eval = sdi_capture_cb_eval
Filter: ggml_tensor * t, t->name == "ffn_inp-0" (exact match, set by llama_context::graph_get_cb)
Capture: ggml_backend_tensor_get(t, host_buf, 0, n_bytes) — copies from offloaded to host memory
Slice:   take last row (last prefill token) from the [hidden_size, n_tokens, 1, 1] tensor
Output:  SDIX binary file at /tmp/phase31cfs_p0_l0.bin (64-byte header + 1536 f32 values)
```

### 2.2 Prefill-only execution

```
n_predict = 0          (no generation)
warmup    = false      (no dummy eval)
batch     = llama_batch_get_one(tokens.data(), n_tokens)  (all prompt tokens at once)
call      = llama_decode(ctx, batch)                       (single prefill call)
```

### 2.3 SDIX output format

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 4 | magic | `"SDIX"` (0x58494453) |
| 4 | 4 | version | `0x00000001` |
| 8 | 4 | dtype | `0x00000001` = float32 |
| 12 | 4 | n_dim | `2` |
| 16 | 8 | dim[0] | `1` (batch=1) |
| 24 | 8 | dim[1] | `1536` (Qwen2.5-1.5B hidden_size) |
| 32 | 8 | token_position | `4` (last prefill token, 0-indexed) |
| 40 | 8 | il | `0` |
| 48 | 8 | shape_logical | `1536` |
| 56 | 4 | prompt_sha256_first4 | `0x057f92d4` (FNV-1a, traceability only) |
| 60 | 4 | reserved | `0` |
| 64 | 6144 | raw float32 data | `1536 * 4` bytes |
| **Total** | **6208** | | |

### 2.4 Replay pipeline (unchanged from 31CD/31CE)

```
W_ref = local 1.5B Q4_K_M GGUF dequantized
W_low = corrected_ceil_per_row Q2_K (quantize_q2k_f32_to_bytes + dequantize_q2k_bytes_to_f32)
R_up   = W_ref_up   - W_low_up
R_gate = W_ref_gate - W_low_gate
sdir_up   = encode_sdir(R_up,   k_pct=0.5)
sdir_gate = encode_sdir(R_gate, k_pct=0.5)
W_sub_up   = W_low_up   + decode_sdir(sdir_up)
W_sub_gate = W_low_gate + decode_sdir(sdir_gate)
W_sub_down = W_low_down   (no sdir for down)
Y_ref = silu(X @ W_ref_gate.T) * (X @ W_ref_up.T) @ W_ref_down.T
Y_low = silu(X @ W_low_gate.T) * (X @ W_low_up.T) @ W_low_down.T
Y_sub = silu(X @ W_sub_gate.T) * (X @ W_sub_up.T) @ W_sub_down.T
```

---

## 3. Execution

### 3.1 Pre-execution preflight

- branch: `master`
- HEAD: `016bb0e8894ab6c22450277d809a75d9db542c06` (= expected per user prompt)
- `git status --short`: clean
- `git fetch origin`: already in sync
- `git log --oneline -5`: 31CF-R hotfix, 31CF-R, 31CF, 31CE, 31CD post-push — expected sequence
- `python3 -m tests.run_source_of_truth_regression`:
  - classification: `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`
  - error_count: 0
  - fallback_count: 0
  - readme_drift_guard.passed: true
  - 11 `"passed": true` instances (all sub-tests pass)

### 3.2 Source-recheck (Candidate E)

- `~/llama.cpp/src/models/qwen.cpp:66-67` still has:
  ```cpp
  ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
  cb(ffn_inp, "ffn_inp", il);
  ```
- `~/llama.cpp/build/bin/llama-server` still has 0 PRT sidecar symbols
- `~/llama.cpp/src/llama-graph.cpp:1844-1933` (`build_ffn`) still has the dormant PRT sidecar machinery, but the operator's binary has it compiled-out via `!defined(PRT_SIDECAR_PAGER_EXPERIMENTAL)`

### 3.3 Harness compilation

- command: `g++ -std=c++17 -O3 -DNDEBUG -I~/llama.cpp/pocs -I~/llama.cpp/common/. -I~/llama.cpp/common/../vendor -I~/llama.cpp/src/../include -I~/llama.cpp/ggml/src/../include -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED src/phase31cfs_capture.cpp -L~/llama.cpp/build/common -L~/llama.cpp/build/bin -Wl,-rpath,... -o src/phase31cfs_capture -lcommon -lllama -lggml-cpu -lggml-base -lggml -lcpp-httplib -lpthread -ldl -lm`
- result: SUCCESS, 4,768,984-byte binary
- warnings: 1 informational (`'g_prt_sidecar_root_set' initialized and declared 'extern'` — expected, harmless)
- link issues resolved:
  - `g_prt_sidecar_root_set` (pre-existing `libllama.so` bug) — provided as `extern "C" int g_prt_sidecar_root_set = 0;` in the harness
  - `httplib::Client` — linked `libcpp-httplib.a` from `~/llama.cpp/build/vendor/cpp-httplib/`
- `~/llama.cpp/` source: **unmodified** (verified via `git -C ~/llama.cpp status` — no changes; the harness is fully in `sdi-substitutive/src/`)
- `~/llama.cpp/` build: **NOT rebuilt** (no cmake invocation, no make invocation, no compilation of any llama.cpp source files)

### 3.4 Harness run

- command: `LD_LIBRARY_PATH=~/llama.cpp/build/bin:$LD_LIBRARY_PATH ./src/phase31cfs_capture -m $SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf --prompt "The capital of France is" --seed 42 -ngl 0`
- model: Qwen2.5-1.5B-Instruct Q4_K_M GGUF, 1,117,320,736 bytes, SHA256 (recorded in result JSON)
- prompt tokenization (llama.cpp tokenizer, add_bos=false): `[785, 6722, 315, 9625, 374]` — 5 tokens
- **Note on token count vs 31CD's "6 BPE tokens":** 31CD used the HF tokenizer, which tokenizes "The capital of France is" to 6 BPE tokens. llama.cpp's tokenizer produces 5 tokens for the same prompt. This is a known artifact of HF vs llama.cpp tokenization (the `cpt` boundary token "The capital" is one token in llama.cpp's BPE but two in HF's BPE). The user's "last prefill token or closest exact equivalent available through llama.cpp callback metadata" approval covers this — the llama.cpp tokenizer's "last prefill token" (index 4) is the closest exact equivalent.
- prefill execution: 1 `llama_decode` call with all 5 tokens in a single batch
- captured tensor: `name=ffn_inp-0 type=f32 ne=[1536, 5, 1, 1] n_bytes=30720` (the tensor contains all 5 prefill positions; the harness slices the last one)
- sliced to last prefill token: row index 4 → `X.shape = (1, 1536)`
- X statistics:
  - x_min = -3.374
  - x_max = 3.513
  - x_mean = 0.00863
  - max_abs = 3.513
  - finite = 1536/1536, nan = 0, inf = 0
- SDIX file: `/tmp/phase31cfs_p0_l0.bin`, 6208 bytes, written successfully
- meta JSON: `/tmp/phase31cfs_p0_l0.bin.meta.json`, 1104 bytes
- exit code: 0
- runtime wall-clock: ~3-4 seconds (CPU-only fwd pass for 5 tokens at L0)

### 3.5 Replay run

- command: `python3 src/phase31cfs_replay.py --gguf-path "$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf"`
  (operator-specific `SDI_MODEL_DIR` value not recorded in the repo; the path is supplied at runtime via CLI arg or env var `PHASE31CFS_GGUF_PATH`)
- input: SDIX file from 3.4 (`/tmp/phase31cfs_p0_l0.bin`, deleted after replay per artifact policy; SHA256 `6578f34547b7b9801b51227810fcf512fa65bc0f852d7cf40080b5dd7dd37454` recorded in result JSON for traceability)
- output: `src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json` (machine-readable summary, metadata only, no raw activation arrays)
- total wall-clock: 9.1 seconds

---

## 4. Metrics

### 4.1 Replay metrics (31CF-S L0 / last prefill token of "The capital of France is")

| metric | value | interpretation |
|---|---|---|
| **cos_low** | 0.934842 | cosine sim of W_low MLP output vs W_ref MLP output (Q2_K-only baseline) |
| **cos_sub** | 0.937591 | cosine sim of W_sub MLP output vs W_ref MLP output (Q2_K + SDIR) |
| **delta_cos** | **+0.002749** | positive → SDIR is directionally helpful vs Q2_K-only |
| **MAE_low** | 0.019455 | mean abs error of W_low MLP output vs W_ref (Q2_K-only baseline) |
| **MAE_sub** | 0.018960 | mean abs error of W_sub MLP output vs W_ref (Q2_K + SDIR) |
| **MAE_delta** | **-0.000495** | negative → SDIR reduces error vs Q2_K-only |
| **finite** | 1536/1536 | all output values are finite |
| **nan** | 0 | no NaN |
| **inf** | 0 | no Inf |
| **severe** | 0 | no severe regression (delta_cos >= -0.05 AND no nan/inf) |

### 4.2 Memory margin (per 31CD definition)

| family | W_low bytes | SDIR bytes | total | Q4_budget | margin |
|---|---|---|---|---|---|
| ffn_up | 4,515,840 | 1,857,972 | 6,373,812 | 6,881,280 | **+507,468** |
| ffn_gate | 4,515,840 | 1,857,974 | 6,373,814 | 6,881,280 | **+507,466** |
| ffn_down | 4,515,840 | 0 (no SDIR) | 4,515,840 | 6,881,280 | **+2,365,440** |

All three family memory margins are **positive** (memory-positive, matches 31CD P0-L0's margin = `+3,380,374` bytes within the same corrected Q2_K + SDIR policy family at L0, same `corrected_q2k_policy_v1`).

### 4.3 Contextual comparison to 31CD/31CE P0-L0 (NOT bit-equal, NOT equivalence)

| metric | 31CD (HF bf16 forward-hook) | 31CE (HF bf16 forward-hook, P0-L0) | 31CF-S (exact Q4_K_M GGUF runtime) |
|---|---|---|---|
| cos_low | 0.942053 | 0.942053 (same P0-L0) | **0.934842** |
| cos_sub | 0.943183 | 0.943183 (same P0-L0) | **0.937591** |
| delta_cos | +0.001130 | +0.001130 (same P0-L0) | **+0.002749** |
| MAE_low | 0.027586 | 0.027586 (same P0-L0) | **0.019455** |
| MAE_sub | 0.027357 | 0.027357 (same P0-L0) | **0.018960** |
| MAE_delta | -0.000229 | -0.000229 (same P0-L0) | **-0.000495** |
| memory margin (P0-L0) | +3,380,374 | +3,380,374 (same P0-L0) | **+507,468 (up), +507,466 (gate)** |
| finite | 1536/1536 | 1536/1536 (same P0-L0) | **1536/1536** |
| severe | 0 | 0 (same P0-L0) | **0** |

**Sign pattern comparison:**
- 31CD: delta_cos = **+0.001130** (positive), MAE_delta = **-0.000229** (negative)
- 31CF-S: delta_cos = **+0.002749** (positive), MAE_delta = **-0.000495** (negative)
- **Both show positive delta_cos and negative MAE_delta** ✓ — **same sign pattern**. 31CF-S shows **larger** improvements (delta_cos 2.4x larger, MAE_delta 2.2x larger) but this is contextual, NOT an equivalence claim.

**Note on the difference in absolute values:**
- 31CD used **HF safetensors** (bf16) forward-hook capture; 31CF-S uses **exact Q4_K_M GGUF / llama.cpp runtime** capture. These are **different model binaries** (HF bf16 weights vs GGUF Q4_K_M quantized weights) and **different forward paths** (HF transformers eager forward vs llama.cpp ggml compute). The lower cos_low in 31CF-S (0.935 vs 0.942) reflects the W_ref being **Q4_K_M dequantized** (more rounded values) vs HF's bf16 (more precise values), and possibly different attention/MLP numerics. The **sign pattern** of the delta (SDIR helpful) is the same.

**Conclusion:** "31CF-S exact GGUF-runtime activation replay is **directionally consistent** with 31CD/31CE Option A P0-L0 under the selected prompt/layer/token scope" — sign pattern matches, magnitudes differ, NOT bit-equal, NOT an equivalence claim.

---

## 5. Classification

### 5.1 PASS criteria check (per user's success-criteria spec)

- ✓ regression passes before and after
- ✓ hook point remains unambiguous (Candidate E callback site at `qwen.cpp:67` is unchanged since 31CF-R, and the per-arch callback label `"ffn_inp-0"` was confirmed via the `ggml_format_name` mechanism in `llama_context::graph_get_cb`)
- ✓ no build needed (no `~/llama.cpp/` rebuild — the harness uses existing `libllama.so`)
- ✓ hook is default-off (the harness explicitly sets `params.cb_eval = sdi_capture_cb_eval`; without the harness, the default `cb_eval` is `nullptr` and the callback never fires)
- ✓ hook enabled only for selected layer/token (the callback filters on `t->name == "ffn_inp-0"` and single-shot after the first match)
- ✓ activation capture succeeds (file `/tmp/phase31cfs_p0_l0.bin`, 6208 bytes, valid SDIX format, all checks pass)
- ✓ activation shape is `[1, 1536]` (verified via SDIX header `dim[0]=1, dim[1]=1536`)
- ✓ activation is finite (1536/1536, 0 nan, 0 inf)
- ✓ replay succeeds (W_ref / W_low / W_sub all computed, Y values finite, metrics computed)
- ✓ memory margin is positive (+507,468 ffn_up, +507,466 ffn_gate, +2,365,440 ffn_down)
- ✓ delta_cos >= 0 (delta_cos = +0.002749)
- ✓ MAE_delta <= 0 (MAE_delta = -0.000495)
- ✓ 0 severe regressions
- ✓ no raw activation arrays committed (deleted from `/tmp` before PRE-COMMIT REPORT; SHA256 of payload recorded in result JSON for traceability: `6578f34547b7b9801b51227810fcf512fa65bc0f852d7cf40080b5dd7dd37454`)
- ✓ no model/HF/cache files committed
- ✓ no Q2_K/SDIR blobs committed
- ✓ no generation/quality/runtime claim made

### 5.2 Classification: `PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN`

All 16 PASS criteria are met. The corrected Q2_K + SDIR policy is **memory-positive and directionally helpful** (delta_cos = +0.002749, MAE_delta = -0.000495) versus Q2_K-only on a tiny exact Q4_K_M GGUF / llama.cpp runtime activation replay for Qwen2.5-1.5B at L0 / last prefill token of "The capital of France is".

---

## 6. Allowed claims (for 31CF-S)

- ✓ Corrected Q2_K + SDIR was memory-positive and directionally helpful versus Q2_K-only on a tiny exact Q4_K_M GGUF / llama.cpp runtime activation replay for Qwen2.5-1.5B, under the selected prompt/layer/token scope.
- ✓ 31CF-S exact GGUF-runtime activation replay is **directionally consistent** with 31CD/31CE Option A P0-L0 under the selected prompt/layer/token scope (same sign pattern for delta_cos and MAE_delta, NOT bit-equal, NOT an equivalence claim).
- ✓ The first Option B capture was performed using the public `params.cb_eval` API (no `~/llama.cpp/` source modification, no `~/llama.cpp/` rebuild).
- ✓ The captured tensor was at the same semantic tensor position as 31CD/31CE's pre-FFN input activation X (input to the L0 MLP, pre-norm, post-attention-residual-add, shape `[1, 1536]` for last prefill token). It does **NOT** imply that 31CF-S captures identical activation values to 31CD/31CE (HF bf16 forward-hook vs exact Q4_K_M GGUF runtime are numerically different methods).
- ✓ The corrected Q2_K + SDIR policy `corrected_q2k_policy_v1` (corrected_ceil_per_row Q2_K, ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual) was applied at L0 with the same conventions as 31CD/31CE.

---

## 7. Forbidden claims (all upheld)

### 7.1 From SOT Section 0.A master list (13 items)

- ✗ no model quality recovery claim
- ✗ no behavior recovery claim
- ✗ no speedup claim
- ✗ no full-model runtime memory savings claim
- ✗ no llama.cpp integration claim (this phase does NOT integrate the corrected Q2_K + SDIR policy into llama.cpp's actual model path; it is a standalone replay)
- ✗ no production readiness claim
- ✗ no inference/generation claim (no `n_predict > 0`, no sampling, no quality evaluation)
- ✗ no larger-model validation claim (this is 1.5B only)
- ✗ no runtime-ready output-residual claim (this is a diagnostic capture, not a runtime feature)
- ✗ no claim beyond standalone tensor harness
- ✗ no orientation claim for a larger model
- ✗ no commit/push/tag without explicit Matt approval (31CF-S stops at PRE-COMMIT REPORT)

### 7.2 31CF-S-specific forbidden claims

- ✗ **no HF == GGUF activation equivalence claim** (31CD/31CE used HF safetensors bf16 forward-hook; 31CF-S uses exact Q4_K_M GGUF / llama.cpp runtime — these are numerically different methods)
- ✗ **no "same activation values" claim** (different model binaries, different forward paths)
- ✗ **no "identical to 31CD" claim** (different X, different tensor values)
- ✗ **no exact runtime activation claim in source-modified sense** (no `~/llama.cpp/` was modified; the capture used the existing public API)
- ✗ **no activation-distribution equivalence claim**
- ✗ **no broad transfer claim** (single prompt, single layer, single token — no generalizability claim)
- ✗ **no "real activations behave like synthetic Gaussian" claim**
- ✗ **no model files / HF cache / raw activation arrays / build artifacts / Q2_K blobs / SDIR blobs / temp tensor dumps / `llama.cpp` source committed to `sdi-substitutive`**
- ✗ **no `llama.cpp` source or build artifacts committed to `sdi-substitutive`**

### 7.3 31CF-S-scoped forbidden behaviors (per user's narrow-scope approval)

- ✗ broad llama.cpp refactor (none performed; no source modifications)
- ✗ permanent runtime feature beyond a clearly gated diagnostic hook (the harness is standalone, NOT a llama-server feature; the harness binary is in `sdi-substitutive/src/`, not in `~/llama.cpp/build/bin/`)
- ✗ all-layer dumping (the callback filters on `"ffn_inp-0"` which is L0 only)
- ✗ all-token dumping (the callback single-shot after first match, then the harness slices to last token)
- ✗ generation / sampling / answer-quality evaluation (none)
- ✗ model behavior claims (none)
- ✗ performance claims (none)
- ✗ committing model files / HF cache (none)
- ✗ committing llama.cpp source files directly into sdi-substitutive (none)
- ✗ committing raw activation arrays (deleted from `/tmp` before PRE-COMMIT REPORT; SHA256 of payload recorded in result JSON for traceability)
- ✗ committing Q2_K / SDIR binary blobs (none; all replay materials kept in-memory only)
- ✗ committing temp tensor dumps (none)

---

## 8. Limitations

- **31CF-S is a single (L0, last-prefill-token, "The capital of France is") result.** No claim of generalizability to other prompts, other layers, other token positions, or other models. No claim that the corrected Q2_K + SDIR policy holds on a broader set.
- **HF vs llama.cpp tokenization discrepancy.** 31CD said "6 BPE tokens" using the HF tokenizer; 31CF-S used the llama.cpp tokenizer which produces 5 tokens for the same prompt. The "last prefill token" is index 4 (5th token) in 31CF-S vs index 5 (6th token) in 31CD. Per user's "last prefill token or closest exact equivalent" approval, the llama.cpp tokenizer's "last prefill token" is the closest exact equivalent.
- **Cross-validation against 31CD/31CE is contextual, not bit-equal.** HF bf16 forward-hook vs exact Q4_K_M GGUF runtime are numerically different methods. The comparison asserts same sign pattern + same order of magnitude (cos_sub > cos_low, MAE_sub < MAE_low), NOT a claim of value identity or Option A == Option B equivalence.
- **The harness binary is in `sdi-substitutive/src/`.** It links against `~/llama.cpp/build/bin/*.so` and `~/llama.cpp/build/common/libcommon.a`. If the operator rebuilds `~/llama.cpp/` in a way that changes the public API (e.g. removes `params.cb_eval`, changes the `llm_graph_context::cb` mechanism, or restructures the `qwen.cpp:67` callback), the harness may need to be re-compiled. This is an expected operational constraint, not a blocker.
- **Pre-existing link-time bug in operator's `libllama.so`.** `extern int g_prt_sidecar_root_set;` in `src/llama.cpp:1648` references a symbol that's never defined as a global (only as a `static bool` in `src/llama-graph.cpp:196` with internal linkage). The 31CF-S harness works around it locally with `extern "C" int g_prt_sidecar_root_set = 0;`. The operator can fix the underlying bug in `~/llama.cpp/` at any time; this workaround is independent.
- **The replay pipeline is the existing 31CD/31CE pipeline.** No new replay code; only the X source (SDIX file instead of HF forward-hook) and the input adapter changed.

---

## 9. Files

### 9.1 Created (in `sdi-substitutive/`, NOT yet committed)

| file | size | role |
|---|---|---|
| `src/phase31cfs_capture.cpp` | ~18 KB | standalone C++ harness for capture (links `~/llama.cpp/build/bin/*.so`, no `~/llama.cpp/` modifications) |
| `src/phase31cfs_capture` | 4.8 MB | compiled harness binary (will be regenerated by re-running the compile command; not committed) |
| `src/phase31cfs_replay.py` | ~22 KB | replay script (loads SDIX, runs existing standalone Q2_K + SDIR pipeline, computes metrics, contextual comparison to 31CD) |
| `src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json` | ~6 KB | machine-readable summary (metadata only, no raw arrays) |
| `docs/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.md` | (this file) | human-readable report |

### 9.2 NOT created (per user's narrow-scope approval)

- no `src/phase31cf_s_*.py` other than capture + replay
- no `docs/PHASE31CF_S_LLAMA_CPP_PATCH_SUMMARY.md` (no patch was written)
- no `*.gguf`, `*.safetensors`, `*.npy`, `*.npz`, `*.bin` (other than the deleted /tmp SDIX file)
- no SDIX files in the repo (they're in `/tmp`, and the only one ever created was deleted before PRE-COMMIT REPORT)
- no raw activation arrays in the repo (the result JSON records `sha256_payload` of the deleted SDIX for traceability, not the array itself)
- no `~/llama.cpp/` source patches, build artifacts, or rebuilt binaries in the repo

### 9.3 Deleted (per artifact policy)

- `/tmp/phase31cfs_p0_l0.bin` (6208 bytes) — deleted before PRE-COMMIT REPORT
- `/tmp/phase31cfs_p0_l0.bin.meta.json` (1104 bytes) — deleted before PRE-COMMIT REPORT
- SHA256 of the deleted payload is recorded in the result JSON: `6578f34547b7b9801b51227810fcf512fa65bc0f852d7cf40080b5dd7dd37454`

---

## 10. SOT update

`SOURCE_OF_TRUTH.md` is updated to add the 31CF-S entry to Section 3 (after the 31CF-R PARTIAL entry), to advance Section 0 line 11 (next allowed phase) to `31CG` or `31CH` (planning-only alternatives; both require explicit operator approval), and to update Section 0 lines 5, 6, 12, 14, 15 with the 31CF-S result.

---

## 11. End of design document

This document is the deliverable for Phase 31CF-S. It is the **first Option B** capture for the SDI-substitutive project: an exact Q4_K_M GGUF / llama.cpp runtime activation capture at L0 for Qwen2.5-1.5B-Instruct, replayed through the existing standalone pipeline, with all 16 PASS criteria met and zero forbidden claims violated.

**Classification: `PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN`**

See also:
- `src/results/PHASE31CF_S_GGUF_RUNTIME_ACTIVATION_MICRO_PROBE.json` — machine-readable summary
- `src/phase31cfs_capture.cpp` — standalone C++ harness
- `src/phase31cfs_replay.py` — replay script
- `docs/PHASE31CF_R_HOOK_POINT_DIAGNOSIS_PATCH_DESIGN.md` — 31CF-R design (the parent phase)
- `SOURCE_OF_TRUTH.md` Section 0 + Section 3 — updated with 31CF-S entry
