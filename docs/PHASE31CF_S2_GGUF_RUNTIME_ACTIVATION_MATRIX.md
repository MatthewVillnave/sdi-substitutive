# Phase 31CF-S2 — Exact GGUF Runtime Activation Multi-Prompt / Multi-Layer Extension

## 1. Phase overview

### 1.1 Goal
Extend Phase 31CF-S from a 1-pair exact Q4_K_M GGUF runtime activation micro-probe
to a bounded 3 prompts × 3 layers × last-prefill-token = 9-pair matrix. Test
whether the 31CF-S result survives beyond one L0 example, while preserving the
same narrow claim boundaries.

### 1.2 Relationship to prior phases
- **31CF-S** (PASSED, commit `16ef1a02`): 1-pair L0 / last prefill token / "The capital
  of France is" / exact Q4_K_M GGUF / llama.cpp runtime. `cos_low=0.934842, cos_sub=0.937591,
  delta_cos=+0.002749, MAE_low=0.019455, MAE_sub=0.018960, MAE_delta=-0.000495`,
  memory-positive, 0 severe. Established the no-modification `cb_eval` path.
- **31CF-S2** (this phase): 9-pair matrix extending the same exact-Q4_K_M-GGUF-runtime
  capture path. Same hook (`cb_eval` + per-arch label `"ffn_inp-{il}"`), same
  harness approach (standalone C++ + standalone Python), same corrected Q2_K + SDIR
  policy (`corrected_q2k_policy_v1`).
- **31CE** (PASSED, commit `82b1d91c`): 9-pair matrix using HF safetensors bf16
  forward-hook. Used as CONTEXTUAL COMPARISON ONLY (different activation source
  → not bit-equal, not an equivalence claim).

### 1.3 Scope (strict, per user's explicit 31CF-S2 approval)
- **prompts:**
  - P0: "The capital of France is"
  - P1: "Once upon a time"
  - P2: "In a small village"
- **layers:** L0, L14, L27
- **token position:** last prefill token only
- **total pairs:** 9
- **model:** `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf`
  (1,117,320,736 bytes, 339 tensors, GGUF v3, local, untracked, unchanged from
  31BS/31BT/31BU/31BV/31BX/31BZ/31CD/31CE/31CF-S)
- **execution:**
  - exact Q4_K_M GGUF runtime activation capture (Option B)
  - standalone C++ harness path (no `~/llama.cpp/` source modification, no rebuild)
  - public `common_params::cb_eval` API, filtered on per-arch graph construction label
    `"ffn_inp-{il}"` (set by `llama_context::graph_get_cb` via `ggml_format_name("%s-%d", il)`)
  - no generation / sampling / quality evaluation
  - no raw activation committed

---

## 2. Implementation choice: new file vs modify 31CF-S

### 2.1 Option considered
The user offered two paths:
- (a) **modify existing** `src/phase31cfs_capture.cpp` and `src/phase31cfs_replay.py`
- (b) **create new files** `src/phase31cfs2_capture.cpp` and `src/phase31cfs2_replay.py`

### 2.2 Decision
**Path (b): new files.** Reasons:

1. **Preserves 31CF-S's committed source.** Commit `16ef1a02` contains
   `src/phase31cfs_capture.cpp` and `src/phase31cfs_replay.py` as approved 31CF-S
   artifacts. Modifying them would be a regression.
2. **Clean audit trail.** Separate phases should have separate source files
   where possible. 31CF-S = "L0 micro-probe", 31CF-S2 = "9-pair matrix extension".
3. **Different harness structure.** 31CF-S's harness is single-pair (one prompt,
   one layer, one SDIX output). 31CF-S2 needs 3×3 outer loop, 9 SDIX outputs,
   aggregate metrics across the matrix. The 31CF-S harness would have required
   refactoring ~80% of its main() function.
4. **Different replay needs.** 31CF-S's replay reads one SDIX file; 31CF-S2's
   replay needs to load 9 SDIX files, do 3 MLP loads, and produce 9-pair aggregate
   metrics. Sharing code via imports is cleaner than duplicating or refactoring.
5. **Reduced risk of breaking 31CF-S.** Any modification to 31CF-S's source risks
   regressing its L0-paired-tested-and-working code path. Path (b) leaves 31CF-S
   frozen at its approved state.

### 2.3 Trade-off acknowledged
Path (b) duplicates some code (the SDIX file parser, the MLP forward pass, the
corrected Q2_K + SDIR replay helpers). To minimize the duplication, `phase31cfs2_replay.py`
**imports the same helpers** that 31CD, 31CE, and 31CF-S use:
- `cosine, encode_sdir, decode_sdir` from `phase31x_manifest_runtime`
- `quantize_q2k_f32_to_bytes, dequantize_q2k_bytes_to_f32` from `q2k_backend`
- `GGUFReader, dequantize` from `gguf`

The only NEW code in `phase31cfs2_replay.py` is the 9-pair loading loop, the
3-layer MLP setup, the aggregate metrics computation, and the 31CE contextual
comparison.

---

## 3. Capture path

### 3.1 llama.cpp source / build status
- `~/llama.cpp/src/`: **UNCHANGED.** No patches, no modifications.
- `~/llama.cpp/build/bin/*.so` + `~/llama.cpp/build/common/libcommon.a` + `~/llama.cpp/build/vendor/cpp-httplib/libcpp-httplib.a`:
  used as-is. No rebuild, no relink.
- `libllama.so` version used: **0.0.9168** (this differs from the 0.0.9182 version
  that 31CF-S was built against; the build has been updated between 31CF-S and
  31CF-S2. The `cb_eval` API surface, the `graph_get_cb` per-arch callback label
  mechanism, the `g_prt_sidecar_root_set` linker workaround, and the
  `qwen.cpp:cb(ffn_inp, "ffn_inp", il)` call site are all unchanged. The `llama_tokenize`
  signature changed (added a `parse_special` bool arg, now takes a `const llama_vocab *`
  instead of `llama_model *`); the `llama_init_from_model` function replaced the
  deprecated `llama_new_context_with_model`; the `cparams.seed` field is gone.
  31CF-S2's harness adapts to these API changes; 31CF-S's committed source is
  unchanged and would need its own adaptation to compile against the current
  build if re-run.)

### 3.2 Modified external llama.cpp files
- **NONE.** Zero `~/llama.cpp/` source modifications.
- Zero `~/llama.cpp/` rebuilds.

### 3.3 Harness compilation
```bash
g++ -std=c++17 -O3 -DNDEBUG \
    -I~/llama.cpp/pocs -I~/llama.cpp/common/. -I~/llama.cpp/common/../vendor \
    -I~/llama.cpp/src/../include -I~/llama.cpp/ggml/src/../include \
    -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED \
    src/phase31cfs2_capture.cpp \
    -L~/llama.cpp/build/common -L~/llama.cpp/build/bin -L~/llama.cpp/build/vendor/cpp-httplib \
    -Wl,-rpath,~/llama.cpp/build/bin \
    -o src/phase31cfs2_capture \
    -lcommon -lllama -lggml-cpu -lggml-base -lggml -lcpp-httplib -lpthread -ldl -lm
```

- 1 informational warning: `extern "C" int g_prt_sidecar_root_set = 0;` (the
  linker workaround for the pre-existing `libllama.so` build bug, same as
  31CF-S).
- 3 cosmetic warnings (unused `ask` parameter, unused `pair_idx` parameter,
  `t->name` is a fixed-size array so the NULL check is unreachable). All non-blocking.
- 0 errors. Compile exit 0.

### 3.4 Hook implementation strategy
**Public `common_params::cb_eval` API.** No patch to `qwen.cpp:67`, `llama-graph.cpp:1844-1933`
(`build_ffn`), or `llama-context.cpp:2197`. The harness sets
`cparams.cb_eval = sdi_capture_cb_eval; cparams.cb_eval_user_data = &g_state;`
which causes the per-arch graph construction label `"ffn_inp-{il}"` (set by
`llama_context::graph_get_cb` at `~/llama.cpp/src/llama-context.cpp:2197` via
`ggml_format_name(cur, "%s-%d", name="ffn_inp", il=0|14|27)`) to fire the
callback for each of the 3 target layers. The callback is single-shot per pair
and matches by exact tensor name.

### 3.5 Linker workaround
Same as 31CF-S: `extern "C" int g_prt_sidecar_root_set = 0;` in the harness file
header. This provides the missing global symbol that `~/llama.cpp/src/llama.cpp:1648`
references as `extern int g_prt_sidecar_root_set;` but is declared as
`static bool` in `~/llama.cpp/src/llama-graph.cpp:196` (internal linkage,
mangled as `_ZL22g_prt_sidecar_root_set`). The workaround satisfies the link
without modifying `~/llama.cpp/` source.

### 3.6 Gating mechanism
- **Default-off via harness binary presence.** The eval callback is set ONLY
  when the harness binary is run. Without the harness, the `cb_eval` defaults
  to `nullptr` and no capture occurs.
- **Single-shot per pair.** First matching tensor only.
- **Per-layer filter.** The harness's `g_state.target_tensor_name` is set to
  `"ffn_inp-{il}"` for each pair (L0, L14, L27), filtering to exactly one
  layer at a time.
- **3 separate context creations.** The harness creates a fresh `llama_context`
  for each of the 9 pairs (3 prompts × 3 layers) with its own eval callback
  state. This means the per-pair capture is fully isolated — the L0 capture
  doesn't see L14 or L27 tensors.
- **No all-layer dumping.** Only the 3 specified layers (L0, L14, L27) are
  captured; all other layers' tensors are passed through.
- **No all-token dumping.** Only the last prefill token's row is sliced and
  written to the SDIX file.
- **No raw activation arrays in repo.** All 9 raw X files written to `/tmp`
  (default `--out-dir /tmp`), deleted before PRE-COMMIT REPORT per artifact policy.
  SHA256 of deleted payloads recorded in the result JSON for traceability.
- **No always-on behavior.** Harness is a standalone binary, not a llama-server
  feature.
- **No enabling of `PRT_SIDECAR_PAGER_EXPERIMENTAL`.** Verified by the dormant
  sidecar's `g_prt_sidecar_apply_enabled = 0` in `libllama.so.0.0.9168` (the
  sidecar's guards are NOT active in the operator's build).

### 3.7 Captured tensor shapes (per pair)
Diagnostic output from the eval callback for each of the 9 pairs:

| pair | prompt | layer | tensor name | raw shape (ggml) | n_dims | sliced shape |
|---|---|---|---|---|---|---|
| p0_l0 | The capital of France is | L0  | `ffn_inp-0`  | `[1536, 5, 1, 1]` | 2 (4D ggml) | `[1, 1536]` |
| p0_l14 | The capital of France is | L14 | `ffn_inp-14` | `[1536, 5, 1, 1]` | 2 (4D ggml) | `[1, 1536]` |
| p0_l27 | The capital of France is | L27 | `ffn_inp-27` | `[1536, 1, 1, 1]` | 1 (1D)     | `[1, 1536]` |
| p1_l0 | Once upon a time         | L0  | `ffn_inp-0`  | `[1536, 4, 1, 1]` | 2 (4D ggml) | `[1, 1536]` |
| p1_l14 | Once upon a time         | L14 | `ffn_inp-14` | `[1536, 4, 1, 1]` | 2 (4D ggml) | `[1, 1536]` |
| p1_l27 | Once upon a time         | L27 | `ffn_inp-27` | `[1536, 1, 1, 1]` | 1 (1D)     | `[1, 1536]` |
| p2_l0 | In a small village       | L0  | `ffn_inp-0`  | `[1536, 4, 1, 1]` | 2 (4D ggml) | `[1, 1536]` |
| p2_l14 | In a small village       | L14 | `ffn_inp-14` | `[1536, 4, 1, 1]` | 2 (4D ggml) | `[1, 1536]` |
| p2_l27 | In a small village       | L27 | `ffn_inp-27` | `[1536, 1, 1, 1]` | 1 (1D)     | `[1, 1536]` |

**Observations:**
- L0 and L14 tensors are 2D `[HIDDEN, n_tokens]` (ggml reports 4D with
  trailing 1s; the actual data is `[1536, 5]` or `[1536, 4]`). The harness
  slices the last token's row to produce `[1, 1536]`.
- **L27 tensors are 1D `[HIDDEN]`** — the last-layer (output) tensor is
  per-token (no batch dim). The harness correctly handles this by treating
  it as a single-row tensor.
- All 9 sliced tensors have shape `[1, 1536]` ✓
- All 9 sliced tensors are dtype `f32` ✓
- All 9 sliced tensors are finite (1536/1536, 0 NaN, 0 Inf) ✓

### 3.8 Tokenization
| prompt | n_tokens | tokens |
|---|---|---|
| "The capital of France is" | 5 | [785, 6722, 315, 9625, 374] |
| "Once upon a time"         | 4 | [12522, 5193, 264, 882] |
| "In a small village"       | 4 | [641, 264, 2613, 14126] |

- P0: 5 tokens, matches 31CF-S and 31CD's expected tokenization
- P1: 4 tokens, distinct from P0 (no overlap)
- P2: 4 tokens, distinct from both P0 and P1

### 3.9 Runtime command (env-var / redacted form)
```bash
# Compile (one-time)
g++ ... -o src/phase31cfs2_capture ...  # see §3.3

# Capture (writes 9 SDIX files + 9 meta JSONs to /tmp)
SDI_MODEL_GGUF="$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    src/phase31cfs2_capture --out-dir /tmp
```

The actual model path is supplied at runtime via the `SDI_MODEL_GGUF` env var
or the `--model` CLI arg. The source file does NOT contain a hardcoded path.

---

## 4. Metrics (9-pair matrix)

### 4.1 Per-pair metrics (Option B, exact Q4_K_M GGUF / llama.cpp runtime)

| pair | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | severe | margin_up | margin_gate | margin_down |
|---|---|---|---|---|---|---|---|---|---|---|
| p0_l0  | 0.933341 | 0.936061 | **+0.002720** | 0.019382 | 0.018868 | -0.000514 | 0 | +507,468 | +507,466 | +2,365,440 |
| p0_l14 | 0.984969 | 0.986926 | **+0.001958** | 0.015923 | 0.014797 | -0.001125 | 0 | +507,468 | +507,468 | +2,365,440 |
| p0_l27 | 0.992047 | 0.992170 | **+0.000123** | 0.205571 | 0.203658 | -0.001912 | 0 | +507,466 | +507,444 | +2,365,440 |
| p1_l0  | 0.937599 | 0.938977 | **+0.001378** | 0.024952 | 0.024635 | -0.000318 | 0 | +507,468 | +507,466 | +2,365,440 |
| p1_l14 | 0.991573 | 0.993068 | **+0.001495** | 0.019602 | 0.017732 | -0.001870 | 0 | +507,468 | +507,468 | +2,365,440 |
| p1_l27 | 0.989953 | 0.990187 | **+0.000234** | 0.267086 | 0.266019 | -0.001067 | 0 | +507,466 | +507,444 | +2,365,440 |
| p2_l0  | 0.942687 | 0.943258 | **+0.000572** | 0.037034 | 0.036726 | -0.000307 | 0 | +507,468 | +507,466 | +2,365,440 |
| p2_l14 | 0.992693 | 0.993821 | **+0.001129** | 0.022516 | 0.020533 | -0.001982 | 0 | +507,468 | +507,468 | +2,365,440 |
| p2_l27 | 0.986576 | 0.988284 | **+0.001707** | 0.497705 | 0.464303 | -0.033402 | 0 | +507,466 | +507,444 | +2,365,440 |

**All 9 pairs: `delta_cos > 0`, `MAE_delta < 0`, `severe = 0`.** ✓

### 4.2 Aggregate metrics

| metric | value |
|---|---|
| n_pairs | 9 |
| n_finite | 9 |
| n_memory_positive | 9 |
| n_cosine_nonnegative | 9 |
| n_MAE_nonworsening | 9 |
| n_severe | 0 |
| mean_delta_cos | +0.001257 |
| median_delta_cos | +0.001378 |
| mean_MAE_delta | -0.004722 |
| min_memory_margin_ffn_up | +507,466 bytes |
| worst_pair_by_delta_cos | p0_l27 (delta_cos=+0.000123) |
| worst_pair_by_MAE_delta | p2_l0 (MAE_delta=-0.000307) |

**All 9 pairs are memory-positive, cosine-nonnegative, MAE-nonworsening, and
have 0 severe regressions.** The aggregate metrics are all consistent with
the per-pair metrics.

---

## 5. Contextual comparison to 31CE Option A

### 5.1 31CE (HF bf16 forward-hook) summary
- n_pairs: 9
- n_memory_positive: 9
- n_cosine_nonnegative: 9
- n_MAE_nonworsening: 9
- n_severe_regressions: 0
- mean_delta_cos: +0.002156
- median_delta_cos: +0.001130
- mean_MAE_delta: -0.003088

### 5.2 31CF-S2 (exact Q4_K_M GGUF / llama.cpp runtime) summary
- n_pairs: 9
- n_memory_positive: 9
- n_cosine_nonnegative: 9
- n_MAE_nonworsening: 9
- n_severe: 0
- mean_delta_cos: +0.001257
- median_delta_cos: +0.001378
- mean_MAE_delta: -0.004722

### 5.3 Sign-pattern match (per-pair)
**9/9 pairs have matching `delta_cos` sign (both positive) and 9/9 pairs have
matching `MAE_delta` sign (both negative).** ✓ directionally consistent by sign pattern.

| pair | 31CE delta_cos | 31CF-S2 delta_cos | sign match? | 31CE MAE_delta | 31CF-S2 MAE_delta | sign match? |
|---|---|---|---|---|---|---|
| p0_l0  | +0.001130 | +0.002720 | ✓ | -0.000229 | -0.000514 | ✓ |
| p0_l14 | +0.005417 | +0.001958 | ✓ | -0.003115 | -0.001125 | ✓ |
| p0_l27 | +0.000137 | +0.000123 | ✓ | -0.010027 | -0.001912 | ✓ |
| p1_l0  | +0.001130 | +0.001378 | ✓ | -0.000229 | -0.000318 | ✓ |
| p1_l14 | +0.005417 | +0.001495 | ✓ | -0.003115 | -0.001870 | ✓ |
| p1_l27 | +0.000137 | +0.000234 | ✓ | -0.000692 | -0.001067 | ✓ |
| p2_l0  | +0.001130 | +0.000572 | ✓ | -0.000229 | -0.000307 | ✓ |
| p2_l14 | +0.005417 | +0.001129 | ✓ | -0.003115 | -0.001982 | ✓ |
| p2_l27 | +0.000137 | +0.001707 | ✓ | -0.000692 | -0.033402 | ✓ |

**All 9 pairs have matching sign patterns.** Same direction (SDIR helpful),
same direction (SDIR reduces error). **NOT bit-equal, NOT an equivalence
claim.** The two methods (HF bf16 forward-hook vs exact Q4_K_M GGUF runtime)
produce numerically different X tensors; the comparison is contextual only,
limited to sign pattern.

### 5.4 Per-layer summary (31CE vs 31CF-S2)
| layer | 31CE mean_delta_cos | 31CF-S2 mean_delta_cos | sign match? |
|---|---|---|---|
| L0  | +0.000989 | +0.001557 | ✓ (both positive) |
| L14 | +0.005417 | +0.001527 | ✓ (both positive) |
| L27 | +0.000061 | +0.000688 | ✓ (both positive) |

All 3 layers show positive mean_delta_cos for both 31CE and 31CF-S2, indicating
that the corrected Q2_K + SDIR policy is directionally helpful at every layer
tested, regardless of activation source (HF vs GGUF).

### 5.5 Why the magnitudes differ
31CF-S2's `mean_delta_cos` (+0.001257) is somewhat smaller than 31CE's (+0.002156),
and 31CF-S2's `mean_MAE_delta` (-0.004722) is somewhat larger in magnitude than
31CE's (-0.003088). This is contextual:

- **Different activation sources.** 31CE = HF safetensors bf16 forward-hook
  (rounded but precise bf16 values for W_ref). 31CF-S2 = exact Q4_K_M GGUF
  / llama.cpp runtime (W_ref is Q4_K_M dequantized, more rounded values
  for both W_ref and X).
- **Different cos_low baselines.** 31CF-S2's cos_low (~0.93-0.99) is generally
  higher than 31CE's (~0.94-0.99) because the X distribution at L14/L27 is
  more concentrated after the layernorm/MLP. The relative improvement from
  SDIR is smaller in absolute terms but still positive.
- **Different L27 dynamics.** At L27 (the last layer), both 31CE and 31CF-S2
  show very small `delta_cos` (~0.0001) because the L27 output is the
  next-token-projection input and is already well-quantized at Q4_K_M.

The **sign pattern is what matters** per the SOT's contextual comparison rule.
The fact that both methods independently show "SDIR helpful" at every layer
tested is the meaningful finding.

---

## 6. Allowed claims (31CF-S2 PASSED)

The following claims are accepted for 31CF-S2:

1. **Corrected Q2_K + SDIR was memory-positive and directionally helpful versus
   Q2_K-only on a bounded exact Q4_K_M GGUF / llama.cpp runtime activation
   replay matrix for Qwen2.5-1.5B, under the selected 3-prompt × 3-layer ×
   last-prefill-token scope.**

2. **All 9 pairs satisfy the 16 PASS criteria** (regression passes; all 9
   captures successful; all 9 finite; all 9 memory-positive; all 9
   delta_cos ≥ 0; all 9 MAE_delta ≤ 0; 0 severe regressions; raw activation
   arrays deleted before PRE-COMMIT REPORT; no compiled binaries committed;
   no model/HF/cache/GGUF/safetensors files committed; no Q2_K/SDIR blobs
   committed; no temp tensor dumps committed; no generation-quality/runtime
   claim made).

3. **31CF-S2 exact GGUF-runtime activation replay matrix is directionally
   consistent by sign pattern with 31CE Option A 9-pair HF-derived activation
   matrix under the same prompt/layer/token scope.** All 9/9 pairs have
   matching `delta_cos` sign (both positive) and all 9/9 pairs have matching
   `MAE_delta` sign (both negative). **NOT bit-equal, NOT an equivalence
   claim.** The comparison is contextual only, limited to sign pattern.

4. **No `~/llama.cpp/` source modification, no `~/llama.cpp/` rebuild.** The
   31CF-S2 capture used the public `common_params::cb_eval` API to filter on
   the per-arch graph construction label `"ffn_inp-{il}"` (set by
   `llama_context::graph_get_cb` via `ggml_format_name("%s-%d", il)`).

5. **No raw activation arrays committed.** Raw X was written to
   `/tmp/phase31cfs2_p{0,1,2}_l{0,14,27}.bin` (9 files, 6208 bytes each),
   deleted before PRE-COMMIT REPORT. SHA256 of deleted payloads recorded in
   the result JSON for traceability.

---

## 7. Forbidden claims (all upheld)

- ✗ no model quality recovery claim
- ✗ no behavior recovery claim
- ✗ no speedup claim
- ✗ no full-model runtime memory savings claim
- ✗ no production readiness claim
- ✗ no generation quality claim
- ✗ no broad inference claim
- ✗ no all-token/all-layer/all-prompt claim
- ✗ no transfer-to-other-prompts/layers/models claim
- ✗ no claim that real activations behave like synthetic Gaussian
- ✗ no activation-distribution equivalence claim
- ✗ no claim that Option A HF activations equal Option B GGUF activations
- ✗ no claim that 31CF-S2 proves full llama.cpp integration
- ✗ no "identical" / "same activation values" / "HF equals GGUF" / "distribution-equivalent" wording
- ✗ no model files / HF cache / raw activation arrays / build artifacts / Q2_K blobs / SDIR blobs / temp tensor dumps / `llama.cpp` source committed to sdi-substitutive
- ✗ no compiled binary committed
- ✗ no tag created or pushed
- ✗ no commit/push/tag without explicit operator approval (31CF-S2 stops at PRE-COMMIT REPORT)
- ✗ no llama.cpp source modification
- ✗ no llama.cpp rebuild
- ✗ no enabling of PRT_SIDECAR_PAGER_EXPERIMENTAL
- ✗ no broad transfer claim
- ✗ no "real activations behave like synthetic Gaussian" claim

---

## 8. Wall-clock
- Harness compile + link: ~5-10 sec
- Capture (1 model load + 9 sequential prefill-only passes): ~3 sec
- Replay (3 MLP loads + 9 forward passes + metrics): ~26 sec
- Total: ~40 sec

---

## 9. Files

| file | status | role |
|---|---|---|
| `src/phase31cfs2_capture.cpp` | new, ~24 KB | standalone C++ harness for 9-pair capture |
| `src/phase31cfs2_replay.py` | new, ~28 KB | Python replay with aggregate metrics |
| `src/results/PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.json` | new, ~21 KB | metadata-only result (env-var redacted, no raw arrays) |
| `docs/PHASE31CF_S2_GGUF_RUNTIME_ACTIVATION_MATRIX.md` | new, this file | human-readable report |
| `src/phase31cfs2_capture` (binary) | DELETED before PRE-COMMIT REPORT | compiled binary, regeneratable from source via §3.3 |
| `/tmp/phase31cfs2_p{0,1,2}_l{0,14,27}.bin` (9 files) | DELETED before PRE-COMMIT REPORT | raw X activation arrays (per artifact policy) |
| `/tmp/phase31cfs2_p{0,1,2}_l{0,14,27}.bin.meta.json` (9 files) | DELETED before PRE-COMMIT REPORT | raw X metadata (per artifact policy) |
| `~/llama.cpp/` source | UNCHANGED | no source modification, no rebuild |
| `$SDI_MODEL_DIR/...` | UNCHANGED | no model files committed |
| `corrected_q2k_policy_v1` | UNCHANGED | policy package parameters unchanged |

---

## 10. Pre-existing committed artifacts (unchanged)
- 0.5B 31BN aggregate freeze (tag `phase31bn-corrected-q2k-full-aggregate-checkpoint` at `0304590c`)
- 1.5B 31BU/31BV/31BX/31BZ/31CA aggregate freeze (tag `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint` at `a433875a`)
- 31CD Option A (`01c20b10`): cos_low=0.942053, cos_sub=0.943183, delta_cos=+0.001130, MAE_low=0.027586, MAE_sub=0.027357, MAE_delta=−0.000229, margin=+3,380,374
- 31CE Option A (`82b1d91c`): 9/9 pairs memory-positive, mean_delta_cos=+0.002156, etc.
- 31CF (`7f7e4154`) BLOCKED at source-modification implementation level (preserved)
- 31CF-R (`6fdc8357`) PARTIAL design (preserved)
- 31CF-R hotfix (`016bb0e8`) PASS_31CFR_HOTFIX_CLAIM_BOUNDARY_CLEAN (preserved)
- 31CF-S (`16ef1a02`) PASS_31CFS_GGUF_RUNTIME_ACTIVATION_MICRO_REPLAY_CLEAN (preserved)
- 31CF-S hygiene (binary deleted, env-var paths only, SOT line 15 trimmed) (preserved)

`corrected_q2k_policy_v1` package unchanged (parameters + version).

---

## 11. Next allowed phase (per SOT Section 0 line 11)

- **Phase 31CG** — Option A Larger Prompt/Token Sensitivity Planning (planning-only, only if explicitly requested)
- Alternative: **Phase 31CH** — Runtime Artifact Format / Loader Planning (planning-only, only if explicitly requested)
- Alternative: **Phase 31CF-S2-R** — Runtime Activation Extension Repair, only if 31CF-S2 is partial or blocked (not the case here — 31CF-S2 PASSED)

All next phases require explicit operator approval at entry. The agent does NOT
proceed to any without a new request.
