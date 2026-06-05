# Phase 31CE — Qwen2.5-1.5B Real-Activation Capture Multi-Prompt / Multi-Layer Extension (Option A)

## 1. What this phase is

Phase 31CE extends the 31CD Option A real-activation micro-probe from a single (L0, last-prefill-token, "The capital of France is") data point to a small bounded 3×3 matrix of prompts × layers, with the same corrected Q2_K + SDIR replay pipeline and the same Option A HF-derived real-activation proxy capture mechanism. The replay W_ref stays the local 1.5B Q4_K_M GGUF dequantized; **no W_ref source pivot**.

This document is the design + outcome record for the 31CE Option A run. The result JSON is at `src/results/PHASE31CE_REAL_ACTIVATION_OPTION_A_MULTIPROMPT_MULTILAYER.json` (~35 KB).

## 2. Scope (strict)

- **Option selected:** A (HF-derived real-activation proxy)
- **Prompts (3):**
  - **P0:** `"The capital of France is"` — 31I prompt index 1 (5 BPE tokens, last token id 374)
  - **P1:** `"Once upon a time"` — common 31I-style opener (4 BPE tokens, last token id 882)
  - **P2:** `"In a small village"` — common 31I-style opener (4 BPE tokens, last token id 14126)
- **Layers (3):** L0, L14, L27 (the 31CC plan's default {0, 14, 27}; matches 31BV's first 1.5B layer choice and 31BX's edge layers)
- **Token position:** last prefill token only
- **Total:** 3 prompts × 3 layers = **9 prompt-layer pairs**
- **Activation shape (captured):** `[1, 1536]` (matches expected)
- **Activation dtype (HF capture):** bfloat16; downcast to float32 for replay (information-preserving)
- **Batch:** 1
- **Hidden / intermediate:** 1536 / 8960 (matches 31BS metadata, 31CD, 31BU/31BV/31BX/31BZ)
- **Replay formula (canonical, per 31BT):**
  - `up = X @ W_up.T`
  - `gate = X @ W_gate.T`
  - `act = silu(gate) * up`
  - `out = act @ W_down.T`
- **Replay W_ref:** local 1.5B Q4_K_M GGUF dequantized; **NO W_ref source pivot** — matches 31BU / 31BV / 31BX / 31BZ / 31CD exactly
- **Replay W_low:** corrected_ceil_per_row Q2_K derived from the Q4_K_M GGUF W_ref
- **Replay SDIR:** ffn_up + ffn_gate, k=0.5%, alpha=1.0, **no ffn_down residual**
- **Policy package:** `corrected_q2k_policy_v1`

## 3. What is explicitly NOT in scope (preserved from 31CC / SOT)

- No aggregate validation beyond the 3×3 matrix
- No generation, no sampling, no inference output, no quality evaluation
- No llama.cpp runtime integration; the Option B path is a separate phase (31CF)
- No exact Q4_K_M GGUF runtime activation claim — Option A is an HF-derived **proxy**
- No claim that the HF safetensors forward pass is equivalent to the Q4_K_M GGUF forward pass (it is not)
- No claim that real activations behave like synthetic Gaussian
- No transfer claim (no claim that this 9-pair result generalizes to other prompts, other layers, other token positions, or other models)
- No 0.5B-vs-1.5B real-activation comparison
- No model quality / behavior / speed / runtime / inference / production claim
- No model files, HF cache, raw activation arrays, Q2_K blobs, or SDIR blobs committed
- No commit, push, or tag without explicit Matt approval

## 4. Capture mechanism (Option A specifics, 31CE extension)

The HF safetensors model is loaded once (`AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, attn_implementation="eager")`) and reused across all 9 captures — no per-prompt reload. For each prompt, one `register_forward_pre_hook(with_kwargs=True)` is attached to **each** of the 3 selected `mdl.model.layers[L].mlp` modules (L0, L14, L27), so a single forward pass produces all 3 layer-level activations for that prompt. After the forward pass, all 3 hooks fire and are removed. Each hook captures the input to the L MLP (i.e., the post-attention residual stream hidden state at the L boundary), which is the canonical "pre-FFN activation" for that layer. The prompt is tokenized with `add_special_tokens=False` (to preserve the prompt text), the model is run with `use_cache=False`, and the activation at the last prefill token position is taken: `X = hook_input[:, -1, :]` with shape `[1, 1536]`. The raw X is downcast to float32 (information-preserving from bf16), written to a per-pair file in a `tempfile.mkdtemp(prefix="phase31ce_")` directory, hashed with SHA256, and the tempfile is **deleted at runner exit** — the raw X is **never** written to disk persistently and is **never** committed to the repo. The result JSON records only:

- `per_prompt_per_layer[prompt_id][L]` block: `delta_cos`, `MAE_delta`, `cos_low`, `cos_sub`, `per_layer_margin_bytes`, `memory_positive`, `severe`, `finite`, `n_tokens_in_prompt`, `last_token_id`, `x_norm`, `x_max_abs`, `sha256_of_raw_X`
- `pair_results[].sha256_of_raw_X`, `pair_results[].x_norm`, `pair_results[].x_max_abs`, `pair_results[].n_tokens_in_prompt`, `pair_results[].last_token_id`, `pair_results[].captured_shape`, `pair_results[].captured_dtype`
- **No raw activation array** in any field

## 5. Memory accounting (1.5B layer, per policy, identical to 31BU/31BV/31BX/31BZ/31CD)

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
- **Per-layer margin:** `20,643,840 − 17,263,466 = +3,380,374 bytes` (L0, L14) / `+3,380,376` (L14) / `+3,380,350` (L27) — matches 31BU/31BV/31BX/31BZ/31CD per-layer margin exactly (variance 26 bytes is the seed-dependent SDIR size variance).
- This is the **same memory accounting** as 31BU / 31BV / 31BX / 31BZ / 31CD, repeated 9 times for 9 (prompt × layer) pairs.

## 6. Per-pair metrics (9 prompt-layer pairs)

| Prompt | Layer | n_tokens | last_tok_id | cos_low | cos_sub | delta_cos | MAE_low | MAE_sub | MAE_delta | margin (B) | severe | finite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| P0 | L0  | 5 | 374   | 0.942053 | 0.943183 | **+0.001130** | 0.027586 | 0.027357 | **−0.000229** | +3,380,374 | false | yes |
| P0 | L14 | 5 | 374   | 0.920508 | 0.924695 | **+0.004187** | 0.077379 | 0.075201 | **−0.002178** | +3,380,376 | false | yes |
| P0 | L27 | 5 | 374   | 0.996658 | 0.996794 | **+0.000137** | 0.445803 | 0.435777 | **−0.010027** | +3,380,350 | false | yes |
| P1 | L0  | 4 | 882   | 0.939424 | 0.940607 | **+0.001183** | 0.036584 | 0.036224 | **−0.000360** | +3,380,374 | false | yes |
| P1 | L14 | 4 | 882   | 0.912128 | 0.920127 | **+0.007999** | 0.069443 | 0.065866 | **−0.003577** | +3,380,376 | false | yes |
| P1 | L27 | 4 | 882   | 0.997085 | 0.997088 | **+0.000004** | 0.474501 | 0.473809 | **−0.000692** | +3,380,350 | false | yes |
| P2 | L0  | 4 | 14126 | 0.946322 | 0.946975 | **+0.000653** | 0.034458 | 0.034280 | **−0.000178** | +3,380,374 | false | yes |
| P2 | L14 | 4 | 14126 | 0.943306 | 0.947372 | **+0.004065** | 0.070052 | 0.066464 | **−0.003588** | +3,380,376 | false | yes |
| P2 | L27 | 4 | 14126 | 0.997968 | 0.998009 | **+0.000041** | 0.365466 | 0.358507 | **−0.006959** | +3,380,350 | false | yes |

**P0 L0 exactly reproduces 31CD** (cos_low=+0.942053, delta_cos=+0.001130, MAE_delta=−0.000229) — confirms cross-runner reproducibility between 31CD (single-pair runner) and 31CE (multi-pair runner).

**Per-layer summary:**

| Layer | mean_delta_cos | mean_MAE_delta | min_margin (B) | n_cos_nonneg | n_MAE_nonworse | n_severe |
|---|---:|---:|---:|:---:|:---:|:---:|
| L0  | +0.000989 | −0.000256 | +3,380,374 | 3/3 | 3/3 | 0/3 |
| L14 | +0.005417 | −0.003115 | +3,380,376 | 3/3 | 3/3 | 0/3 |
| L27 | +0.0000606 | −0.005893 | +3,380,350 | 3/3 | 3/3 | 0/3 |

**Layer-structure observation (informational, not a claim):** L14 is the layer where SDIR helps the most in absolute terms (mean delta_cos=+0.0054), L0 is intermediate (+0.0010), and L27 is the layer where real activations are already nearly at the Q2_K-only cosine ceiling (cos_low 0.997-0.998, delta_cos essentially 0 but still positive). The MAE improvements are largest at L27 (mean MAE_delta=−0.0059), intermediate at L14 (−0.0031), and smallest at L0 (−0.000256). This is a structural property of the per-layer real-activation geometry, not a behavioral claim about the model.

## 7. Aggregate summary (within the 9-pair scope — not aggregate validation)

| metric | value |
|---|---:|
| n_pairs | 9 |
| n_finite | 9/9 |
| n_memory_positive | 9/9 |
| n_cosine_nonnegative | **9/9** |
| n_MAE_nonworsening | **9/9** |
| n_severe_regressions | 0 |
| mean_delta_cos | **+0.002156** |
| median_delta_cos | +0.001130 |
| mean_MAE_delta | **−0.003088** |
| min_per_layer_margin_bytes | +3,380,350 |
| max_per_layer_margin_bytes | +3,380,376 |
| worst_pair_by_delta_cos | P1 L27 (delta_cos=+0.0000036, MAE_delta=−0.000692) |
| worst_pair_by_MAE_delta | P0 L27 (delta_cos=+0.000137, MAE_delta=−0.010027) |

**Classification:** `PASS_31CE_REAL_ACTIVATION_OPTION_A_MULTIPROMPT_MULTILAYER_CLEAN`

**Interpretation:** On this 9-pair (3 prompt × 3 layer × last-prefill-token) HF-derived real-activation proxy matrix, the corrected_ceil_per_row Q2_K + ffn_up+ffn_gate SDIR k=0.5% policy **improved** the MLP output's cosine similarity and mean-absolute-error to the Q4_K_M W_ref MLP output **simultaneously on every single pair** (9/9 cosine non-negative, 9/9 MAE non-worsening, 0 severe, all finite, all memory-positive). Mean delta_cos=+0.002156, mean MAE_delta=−0.003088, all per-layer margins in [+3,380,350, +3,380,376] bytes.

## 8. What this does NOT prove (preserved 31CE forbidden claims)

- It does NOT prove the corrected Q2_K policy generalizes to other prompts, other layers, other token positions, or other models. Single (3 prompt × 3 layer × last-prefill-token, 9-pair) micro-probe.
- It does NOT prove the HF safetensors forward-pass activation is equivalent to the Q4_K_M GGUF forward-pass activation. They are different model binaries. The X captured here is labeled an **HF-derived proxy**; Option B (llama.cpp instrumentation on the local Q4_K_M GGUF) is the only path that can claim exact Q4_K_M GGUF runtime activation behavior.
- It does NOT prove that real activations behave like synthetic Gaussian tensor-harness X vectors. The cos_low of 0.91-0.95 on the real activations at L0/L14 is **lower** than the cos_low of ~0.99+ on the synthetic-Gaussian X vectors in 31BU / 31BV / 31BX / 31BZ at those layers — they are different distributions. (Note: 31BU/31BV/31BX/31BZ used synthetic Gaussian X; their cos_low values are not directly comparable to 31CE's because the X source is different. The "real-activation cos_low is lower than synthetic-Gaussian cos_low" observation is recorded as a finding, not a claim of behavioral equivalence.)
- It does NOT prove the corrected Q2_K policy is faster, smaller, or production-ready. This is a standalone tensor-harness micro-probe.
- It does NOT modify any prior accepted numeric result (31BN / 31BM for 0.5B synthetic, 31BU / 31BV / 31BX / 31BZ for 1.5B synthetic, 31CA freeze, 31CB cleanup, 31CC plan, 31CD Option A real-activation micro-probe).

## 9. Run reproducibility

The runner is at `src/phase31ce_real_activation_option_a_multiprompt_multilayer.py`. To re-run:

```bash
export SDI_MODEL_DIR=/media/matthew-villnave/VL_usb/models
cd /home/matthew-villnave/sdi-substitutive
python3 -u src/phase31ce_real_activation_option_a_multiprompt_multilayer.py
```

The runner loads the HF model once (~5-10 sec on CPU), then runs 3 forward passes (one per prompt, each with 3 hooks attached). Total wall-clock for the 9-pair matrix: under 30 seconds on CPU. The P0 L0 pair exactly reproduces 31CD (cos_low=+0.942053, delta_cos=+0.001130, MAE_delta=−0.000229), so the P0 L0 numbers in this JSON are also a cross-runner reproducibility check against 31CD.

The runner is **deterministic for a given (model file, prompt text) pair** — no seed-controlled randomness in the X capture path; the SDIR is deterministic given the W_ref and the policy. The HF model loading and forward pass are non-seed-deterministic in PyTorch (the L MLP output depends on the bf16 storage of the weights, which is the model's pre-trained bf16 safetensors, but the forward kernel may not be bit-reproducible across CPUs). For scientific comparability with 31CD, the result numbers should be considered **method-comparable** (same replay path, same replay W_ref, same policy, same X source) but not **bit-comparable** to 31BU / 31BV / 31BX / 31BZ (which used synthetic-Gaussian X).

## 10. Environment setup (preserved from 31CD; no new installs)

The 31CE entry-dependencies were all satisfied by 31CD's environment setup (no new installs required for 31CE):
- `torch==2.4.1+cpu`, `transformers==4.46.3`, `safetensors==0.7.0`, `numpy==2.4.3`, `huggingface_hub==0.36.2`, `gguf` present — same as 31CD.
- HF model at `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/` (2.9 GB on disk, well under the 6 GB budget approved in 31CD), SHA256 of `model.safetensors` = `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` (3,087,467,144 bytes). All 6 files in the HF dir are unchanged from 31CD.
- Local 1.5B Q4_K_M GGUF at `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` (1,117,320,736 bytes, 339 tensors, GGUF v3) — unchanged from 31BS / 31BT / 31BU / 31BV / 31BX / 31BZ / 31CD.

## 11. Limitations

- **3-prompt × 3-layer × last-prefill-token = 9-pair micro-probe.** A 9-pair matrix is still a micro-probe, not aggregate validation. It is sufficient for the "policy helps on multiple prompts and multiple layers, not just L0 with one prompt" claim, and is **insufficient** for any aggregate, multi-prompt-multi-seed, full-28-layer, or full-model claim.
- **3 prompts are all short, PII-free, public-domain, and English.** A broader prompt set (longer prompts, multilingual prompts, prompts with different token-count characteristics) would test the SDIR residual's behavior more thoroughly. Not in scope.
- **3 layers (L0/L14/L27) are the 31CC plan's default {0, 14, 27} — top/mid/end sampling, 3 of 28 layers = 11% layer coverage.** Not a 28-layer aggregate. 31BZ (the synthetic-X 28-layer aggregate) is the prior accepted aggregate; 31CE is a real-activation extension at lower layer coverage.
- **Last-prefill-token only.** Mid-prompt and last-token-of-generation positions are not tested.
- **HF-derived proxy, not GGUF runtime activation.** Same caveat as 31CD.
- **CPU forward pass.** No GPU acceleration. Total 31CE runner time under 30 seconds on CPU. Informational only.
- **No Q2_K / SDIR artifacts saved to disk.** All replay materials exist in memory only during the runner's lifetime. The result JSON does not contain any of these.
- **P0/P1/P2 tokenization:** P0 tokenizes to 5 BPE tokens; P1 and P2 tokenize to 4 BPE tokens each. The 31CC plan said "6 BPE tokens" based on 0.5B tokenization; the 1.5B tokenizer produces 5 (P0) or 4 (P1, P2) BPE tokens. The deviations are recorded (per-pair `n_tokens_in_prompt` and `last_token_id` fields) and are within the 31CC "operator may select from the 31I prompt set or supply a new short prompt at 31CD/31CE entry" clause.

## 12. Next allowed phase (preserved, no entry)

Per SOT Section 9, after 31CE the next scientific phase is:

- **31CF** (Option B llama.cpp instrumentation, only if explicitly requested and operator permission granted for llama.cpp modification and rebuild) — Option B is the only path that can claim exact Q4_K_M GGUF runtime activation behavior. **This is the natural next step** if the goal is exact Q4_K_M GGUF runtime activation capture.
- **31CE-R** (Real-Activation Option A Repair / Prompt-Layer Diagnosis, only if explicitly requested) — recovery branch if 31CE is rejected or needs per-pair diagnosis.
- **31CG** (Option A Larger Prompt/Token Sensitivity Planning, only if explicitly requested) — alternative if we decide not to do llama.cpp yet.

All three require explicit operator approval at entry. None will be entered without a new request from Matt.
