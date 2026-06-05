# Phase 31CD — Qwen2.5-1.5B Real-Activation Capture Micro-Probe (Option A)

## 1. What this phase is

Phase 31CD is the first phase in this repo that uses a **real prompt-derived activation** (not synthetic Gaussian) as the X input to the corrected Q2_K + SDIR replay, and the first phase to do so via an HF-derived capture path (Option A of the 31CC planning table). The replay path itself — local 1.5B Q4_K_M GGUF dequantized as W_ref, corrected_ceil_per_row Q2_K as W_low, ffn_up+ffn_gate SDIR k=0.5% as the residual — is **identical** to 31BU / 31BV / 31BX / 31BZ. The only thing that changes in 31CD is the **X source**: it comes from a real HF safetensors forward pass on a real prompt, not from `np.random.default_rng(seed)`.

This document is the design + outcome record for the 31CD Option A run. The result JSON is at `src/results/PHASE31CD_REAL_ACTIVATION_MICRO_PROBE_OPTION_A.json` (~7.8 KB).

## 2. Scope (strict)

- **Option selected:** A (HF-derived real-activation proxy)
- **Model:** Qwen2.5-1.5B-Instruct (HF safetensors, for X capture) and Qwen2.5-1.5B-Instruct Q4_K_M GGUF local (for W_ref)
- **Layer:** L0 only (the first 31CD execution; per 31CC scope, reducible from {0, 14, 27} to {0} for the first iteration)
- **Token position:** last prefill token
- **Prompt:** `"The capital of France is"` (default 31CC scope; 31I prompt index 1; PII-free; public-domain; 5 BPE tokens in this model)
- **Activation shape (captured):** `[1, 1536]` (matches expected)
- **Activation dtype (HF capture):** bfloat16; downcast to float32 for replay (information-preserving)
- **Batch:** 1
- **Hidden / intermediate:** 1536 / 8960 (matches 31BS metadata)
- **Replay formula (canonical, per 31BT):**
  - `up = X @ W_up.T`
  - `gate = X @ W_gate.T`
  - `act = silu(gate) * up`
  - `out = act @ W_down.T`
- **Replay W_ref:** local 1.5B Q4_K_M GGUF dequantized; **NO W_ref source pivot** — matches 31BU / 31BV / 31BX / 31BZ exactly
- **Replay W_low:** corrected_ceil_per_row Q2_K derived from the Q4_K_M GGUF W_ref
- **Replay SDIR:** ffn_up + ffn_gate, k=0.5%, alpha=1.0, **no ffn_down residual**
- **Policy package:** `corrected_q2k_policy_v1`

## 3. What is explicitly NOT in scope (preserved from 31CC / SOT)

- No aggregate validation, no multi-layer sweep, no multi-prompt sweep
- No generation, no sampling, no inference output, no quality evaluation
- No llama.cpp runtime integration; the Option B path is a separate phase (31CF)
- No exact Q4_K_M GGUF runtime activation claim — Option A is an HF-derived **proxy**
- No claim that the HF safetensors forward pass is equivalent to the Q4_K_M GGUF forward pass (it is not)
- No claim that real activations behave like synthetic Gaussian
- No transfer claim (no claim that this single result generalizes to other prompts, other layers, other token positions, or other models)
- No 0.5B-vs-1.5B real-activation comparison
- No model quality / behavior / speed / runtime / inference / production claim
- No model files, HF cache, raw activation arrays, Q2_K blobs, or SDIR blobs committed
- No commit, push, or tag without explicit Matt approval

## 4. Capture mechanism (Option A specifics)

The HF safetensors model is loaded with `transformers.AutoModelForCausalLM.from_pretrained(...)` using `torch_dtype=torch.bfloat16` and `attn_implementation="eager"`. A `register_forward_pre_hook(with_kwargs=True)` is attached to `mdl.model.layers[0].mlp`. The hook captures the **input to the L0 MLP** (i.e., the post-attention residual stream hidden state at the L0 boundary), which is the canonical "pre-FFN activation" for that layer. The prompt is tokenized with `add_special_tokens=False` (to preserve the 31CC prompt text), the model is run with `use_cache=False`, and the activation at the last prefill token position is taken: `X = hook_input[:, -1, :]` with shape `[1, 1536]`. The raw X is downcast to float32 (information-preserving from bf16), written to a `tempfile.mkdtemp` directory, hashed with SHA256, and the tempfile is **deleted at runner exit** — the raw X is **never** written to disk persistently and is **never** committed to the repo. The result JSON records only:

- `captured_shape_per_layer[L0] = [1, 1536]`
- `captured_dtype = "float32"` (the replay dtype; the HF storage dtype is bf16, recorded in `t_load_sec`/`t_forward_sec`/runner-internal `torch_dtype`)
- `sha256_of_raw_X_per_layer[L0] = "..."` (SHA256 of the X file that was on disk only for the duration of the hash)
- `x_norm`, `x_max_abs` (summary statistics; the raw X is NOT included)
- `n_tokens_in_prompt = 5` (actual count for this prompt; the 31CC plan said "6 BPE tokens" which was for 0.5B — 1.5B tokenizes this prompt to 5 BPE tokens because the 1.5B tokenizer merges differently. The deviation is recorded, not a blocker.)

## 5. Memory accounting (1.5B layer, per policy)

- **Q4_budget_family (1.5B):** `(INTERMEDIATE * HIDDEN) / 2 = (8960 * 1536) / 2 = 6,881,280` bytes
- **Q4_budget_layer:** `3 × 6,881,280 = 20,643,840` bytes
- **Per-layer bytes (corrected_ceil_per_row Q2_K + ffn_up+ffn_gate SDIR + ffn_down W_low only):**
  - Q2_K up:    4,515,840 bytes
  - Q2_K gate:  4,515,840 bytes
  - Q2_K down:  4,515,840 bytes
  - SDIR up (k=0.5%): 1,857,972 bytes
  - SDIR gate (k=0.5%): 1,857,974 bytes
  - SDIR down: 0 bytes (no ffn_down residual by policy)
  - **Total:** 17,263,466 bytes
- **Per-layer margin:** `20,643,840 − 17,263,466 = +3,380,374 bytes` ✓ memory-positive
- This matches the 31BU / 31BV / 31BX / 31BZ per-layer margin exactly (3,380,374 / 3,380,350 / 3,380,312 / 3,380,342 etc. — small variance from the seed-dependent SDIR size).

## 6. Result

| Metric | Value | Notes |
|---|---|---|
| `cos_low` | 0.9420531392 | cosine between Y_ref and Y_low (Q2_K only, no residual) |
| `cos_sub` | 0.9431832433 | cosine between Y_ref and Y_sub (Q2_K + SDIR residual) |
| `delta_cos` | **+0.0011301041** | cos_sub − cos_low; positive ⇒ SDIR improves cosine |
| `MAE_low` | 0.0275857206 | mean abs error of Y_low vs Y_ref |
| `MAE_sub` | 0.0273566786 | mean abs error of Y_sub vs Y_ref |
| `MAE_delta` | **−0.0002290420** | MAE_sub − MAE_low; negative ⇒ SDIR improves MAE |
| `severe` | false | `delta_cos < −0.05` would be severe; ours is positive |
| `finite` | `{Y_ref: true, Y_low: true, Y_sub: true}` | all outputs are finite |
| `shapes` | `{Y_ref: [1, 1536], Y_low: [1, 1536], Y_sub: [1, 1536]}` | matches expected |
| `memory_positive` | true | per-layer margin `+3,380,374` bytes |

**Classification:** `PASS_31CD_REAL_ACTIVATION_MICRO_REPLAY_CLEAN`

**Interpretation:** On this single (L0, last-prefill-token, prompt "The capital of France is", shape `[1, 1536]`, dtype bf16→f32) HF-derived real-activation proxy, the corrected_ceil_per_row Q2_K + ffn_up+ffn_gate SDIR k=0.5% policy **improves** the MLP output's cosine similarity and mean-absolute-error to the Q4_K_M W_ref MLP output **simultaneously**, with `delta_cos=+0.001130`, `MAE_delta=−0.000229`, memory-positive (`+3,380,374` bytes), all finite, no severe regression.

This is the first passing corrected_Q2K+SDIR replay result on a real prompt-derived activation in this repo.

## 7. What this does NOT prove (preserved 31CD forbidden claims)

- It does NOT prove the corrected Q2_K policy generalizes to other prompts, other layers, other token positions, or other models. Single (L0, last-prefill-token, "The capital of France is", shape `[1, 1536]`) result.
- It does NOT prove the HF safetensors forward-pass activation is equivalent to the Q4_K_M GGUF forward-pass activation. They are different model binaries. The X captured here is labeled an **HF-derived proxy**; Option B (llama.cpp instrumentation on the local Q4_K_M GGUF) is the only path that can claim exact Q4_K_M GGUF runtime activation behavior.
- It does NOT prove that real activations behave like synthetic Gaussian tensor-harness X vectors. The cos_low of `0.942` on the real activation is **lower** than the cos_low of `~0.99+` on the synthetic-Gaussian X vectors in 31BU / 31BV / 31BX / 31BZ — they are different distributions.
- It does NOT prove the corrected Q2_K policy is faster, smaller, or production-ready. This is a standalone tensor-harness micro-probe.
- It does NOT modify any prior accepted numeric result (31BN / 31BM for 0.5B, 31BU / 31BV / 31BX / 31BZ for 1.5B synthetic, 31CA freeze, 31CB cleanup, 31CC plan).

## 8. Run reproducibility

The runner is at `src/phase31cd_real_activation_micro_probe_option_a.py`. To re-run:

```bash
export SDI_MODEL_DIR=/media/matthew-villnave/VL_usb/models
cd /home/matthew-villnave/sdi-substitutive
python3 -u src/phase31cd_real_activation_micro_probe_option_a.py
```

The runner is **deterministic** for a given model file + prompt text (no seed-controlled randomness in the X capture path; the SDIR is deterministic given the W_ref and the policy). The HF model loading and forward pass are non-seed-deterministic in PyTorch (the L0 MLP output depends on the bf16 storage of the weights, which is the model's pre-trained bf16 safetensors, but the forward kernel may not be bit-reproducible across CPUs). For scientific comparability with 31BU / 31BV / 31BX / 31BZ, the result numbers should be considered **method-comparable** (same replay path, same replay W_ref, same policy) but not **bit-comparable** (different X source).

## 9. Environment-only setup steps (recorded for audit)

The 31CD entry-dependencies required installing three new packages into the active venv:

| package | version (post-31CD) | purpose |
|---|---|---|
| `torch` | `2.4.1+cpu` | HF safetensors forward pass; CPU-only build (no CUDA deps) |
| `transformers` | `4.46.3` | `AutoModelForCausalLM`, `AutoTokenizer`, `register_forward_pre_hook(with_kwargs=True)` |
| `safetensors` | `0.7.0` | safetensors loader (transitive of transformers) |

`huggingface_hub` was downgraded from `1.17.0` to `0.36.2` because `transformers==4.46.3` requires the older API. This is an environment-only change; no project source files, `pyproject`, requirements, lockfiles, or `README.md` were modified to accommodate the install.

The HF model was downloaded into `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/` (2.9 GB on disk, well under the approved 6 GB budget), with the following SHA256:

```
config.json              98d2ff8cc47488d08a2b3acf4eb99ef210779b42bd48605f6b8e36acdbf670
generation_config.json   e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6
merges.txt               599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3
model.safetensors        dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
tokenizer.json           c0382117ea329cdf097041132f6d6d735924b697924d6f6fc3945713e96ce87539
tokenizer_config.json    5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583
```

The local 1.5B Q4_K_M GGUF was already in place from 31BS (`$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes, GGUF v3, 339 tensors — same file used by 31BT, 31BU, 31BV, 31BX, 31BZ).

## 10. Limitations

- **Single (L0, last-prefill-token, prompt) data point.** A 1-pair micro-probe is the minimum the 31CC plan allows. It is sufficient for the "methodology works" claim and the "policy improves on this activation" claim, and it is **insufficient** for any aggregate, multi-layer, multi-prompt, or full-model claim.
- **HF-derived proxy, not GGUF runtime activation.** The X is captured from the HF safetensors forward pass. The W_ref for replay is from the local Q4_K_M GGUF. The two model binaries are quantized differently (HF bf16 vs local Q4_K_M) and may have small numerical differences in their layer-0 MLPs. The Option B path (31CF) is the only path that can capture X from the local Q4_K_M GGUF and is required for any "exact runtime" claim.
- **CPU forward pass.** No GPU acceleration was used. The forward pass took ~6 seconds of wall-clock on the host CPU; total 31CD runner time (model load + forward + replay + JSON dump) was under 30 seconds. This is informational only and does not establish any latency, throughput, or production-runtime claim.
- **Prompt tokenization difference vs 31CC plan.** The 31CC plan estimated 6 BPE tokens for "The capital of France is"; the 1.5B tokenizer produces 5. The deviation is recorded (result JSON `n_tokens_in_prompt: 5`); this is within scope of the 31CC "operator may select from the 31I prompt set or supply a new short prompt at 31CD entry" clause and is not a blocker.
- **No `Q2_K` / `SDIR` artifacts saved to disk.** All replay materials (Q2_K bytes, SDIR bytes, residual matrices, X) exist in memory only during the runner's lifetime. The result JSON does not contain any of these.

## 11. Next allowed phase (preserved, no entry)

Per SOT Section 9, after 31CD the next scientific phase is:

- **31CE** (Option A multi-prompt / multi-layer extension, only if explicitly requested), OR
- **31CF** (Option B llama.cpp instrumentation, only if explicitly requested and operator permission granted for llama.cpp modification and rebuild)

Both 31CE and 31CF require explicit operator approval at entry (download/install approvals as applicable). Neither is entered without a new request from Matt.
