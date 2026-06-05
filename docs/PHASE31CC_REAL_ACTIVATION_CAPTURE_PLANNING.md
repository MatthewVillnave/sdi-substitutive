# Phase 31CC — Qwen2.5-1.5B Real-Activation Capture Planning

> **Planning-only phase. No activation capture runs. No generation. No inference. No sampling. No llama.cpp runtime integration. No llama.cpp modification. No hook creation. No Q2_K/SDIR artifact generation. No model inference beyond metadata inspection. No quality / performance / runtime / behavior claim. No commit, push, or tag without explicit Matt approval.**

---

## 1. Goal

After the 31CB stale-file provenance cleanup, the next scientific gap in the project is **real prompt-derived activation behavior**. Every accepted numeric result so far (0.5B 31BN/31BM and 1.5B 31BU/31BV/31BX/31BZ/31CA) used `np.random.default_rng(seed).standard_normal((1, hidden))` as the "activation" — pure Gaussian, not prompt-derived. The corrected Q2_K policy `corrected_q2k_policy_v1` is therefore **only proven on synthetic Gaussian activations**; the real-activation question is open.

31CC is **planning-only**:
- Define what "real activation" means for this project (precisely enough to design a micro-probe).
- Compare the candidate capture strategies and pick the safest next executable phase.
- Define the exact 31CD micro-probe scope (prompt, layers, token positions, batch, shape, allowed artifacts, privacy).
- Define 31CD stop conditions, classification list, allowed / forbidden claims.
- Define a hard 31CD-entry decision point (Option A / B / C) — Matt must explicitly choose one before 31CD can begin.
- Do NOT run capture, do NOT modify llama.cpp, do NOT download anything, do NOT generate Q2_K/SDIR artifacts, do NOT run generation, do NOT run inference beyond metadata inspection.

The historical precedent for real-activation capture in this repo is **Phase 31I** (0.5B, untracked, not committed). 31I used HuggingFace `transformers` `forward_hooks` on a `AutoModelForCausalLM` instance to capture `[15 prompts × 6 layers × 2 families]` of pre-FFN activations (shape `(15, 896)` for ffn_up, `(15, 4864)` for ffn_down). 31I used the **pre-31AL** residual policy at `k=7.5%`, not the current `corrected_q2k_policy_v1` k=0.5%; its activations were consumed by 31P (also historical, untracked, not committed). The 31I capture runner script (`phase31i_capture_activations.py`) is **not on disk**; only the log, the shell wrapper, and the `PHASE31I_activations.npz` artifact survive. 31CC acknowledges 31I as historical methodology evidence only.

---

## 2. Definition of "real activation" for this project

A real activation, for the purposes of any future real-activation replay / micro-probe phase, is:

> The actual hidden-state vector X entering a selected FFN/MLP block during a **real model forward pass** on **tokenized prompt text**, captured **before** `ffn_up` and `ffn_gate` are applied. X has shape `[tokens_or_batch, hidden]` where `hidden` is the model's hidden size (1.5B → 1536, 0.5B → 896).

Properties this definition enforces:

- **Real model forward pass** — not a synthetic Gaussian; not a randomly sampled vector. The forward pass may be prefill-only (no autoregressive generation) or include a small fixed number of decode steps, but **no sampling** (no `do_sample=True`, no temperature > 0, no nucleus / top-k / top-p filtering). For the first 31CD micro-probe, **prefill-only** is the recommended setting.
- **Tokenized prompt text** — the model must be invoked on a real text string that is tokenized by the model's own tokenizer (Qwen2.5 BPE / `gpt2` tokenizer for Qwen2.5-1.5B-Instruct). The tokenization and the resulting input_ids are reproducible from the prompt string.
- **Captured before `ffn_up` / `ffn_gate` are applied** — the capture point is the **residual stream input** to the FFN block of layer L (also called the MLP input / pre-FFN hidden state). For Qwen2.5, this is the tensor output by the `blk.L.attn_norm` (post-attention residual-add) that flows into `blk.L.ffn_up` and `blk.L.ffn_gate`.
- **Captured X is a single concrete vector** — the harness picks one token position (or one batch of token positions) per layer, with a deterministic method for choosing the position. The recommended default is the **last token of the input_ids** (the standard "prefill final-position" hidden state used in causal-LM activation probing). The harness must document exactly which position(s) it captured.
- **Replay uses the same standalone MLP formula from 31BT** — after capture, the captured X is replayed through:
  - W_ref MLP (dequantized Q4_K_M, the **frozen convention from 31BU/31BV/31BX/31BZ**)
  - corrected Q2_K-only MLP
  - corrected Q2_K + SDIR MLP
  using the canonical formula `Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T` (per 31BT).
- **Policy, W_ref source, residual construction are unchanged** — `corrected_q2k_policy_v1` applies. W_ref is **always** Q4_K_M dequantized (the frozen 1.5B convention from 31BU/31BV/31BX/31BZ). Residual is ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual.
- **The X source may differ from the W_ref source** — Option A captures X via HuggingFace `transformers` (a different model binary from the local 1.5B Q4_K_M GGUF) but replays through the Q4_K_M GGUF W_ref; this is an **HF-derived real-activation proxy**, NOT exact GGUF-runtime activation capture. Option B captures X via llama.cpp instrumentation against the local 1.5B Q4_K_M GGUF, making it the only path that can claim exact Q4_K_M GGUF runtime activation capture.

This definition deliberately excludes: sampling, generation, attention-pattern capture, key/value cache capture, residual-stream output capture (post-FFN), and any per-token multi-step autoregressive state. Any future phase that wants to expand the definition must do so explicitly in a new planning phase, not by extending 31CD's scope.

---

## 3. Capture strategies evaluated

Four candidate strategies for the first real-activation micro-probe (31CD). All four are evaluated for safety, cost, faithfulness to "what a real prompt-derived activation looks like", and **which W_ref source the replay uses**. **Two are kept as alternatives (A and B); one is rejected; one is out of scope.** A hard 31CD-entry decision point is defined in Section 4.

### 3.1 Option A — HuggingFace `transformers` forward-hook capture (HF-derived real-activation proxy)

**Mechanism.** Load the model with `transformers.AutoModelForCausalLM.from_pretrained(...)` for Qwen2.5-1.5B-Instruct in **safetensors** format, register a `forward_hook` on the target FFN block to capture the pre-FFN residual-stream input, run a prefill-only forward pass on a single tokenized prompt, save the captured X.

**Provenance.** This is exactly the mechanism Phase 31I used for 0.5B. The 31I capture log shows the hook-based methodology was proven end-to-end on this host: 12 hooks registered, 15 prompts processed, output saved to `data/PHASE31I_activations.npz`. The methodology is **not new** for this project; only the model is new.

**W_ref source for replay — UNCHANGED.** Despite X being captured from a different model binary (HF safetensors), the replay still uses the **existing local 1.5B Q4_K_M GGUF dequantized W_ref**, matching 31BU/31BV/31BX/31BZ exactly. The W_low (corrected_ceil_per_row Q2_K) and the SDIR residual (ffn_up+ffn_gate, k=0.5%, alpha=1.0, no ffn_down residual) are also derived from the Q4_K_M GGUF W_ref — unchanged from the frozen 1.5B convention. **There is NO W_ref source pivot in Option A.** The only thing that differs from 31BZ is the X distribution: real prompt-derived instead of synthetic Gaussian.

**Why this is a "proxy" not "real runtime activation".** Option A captures X from a **different model binary** (the HF safetensors) than the one that produced the deployed artifacts (the local Q4_K_M GGUF). Even though both binaries are nominally the same Qwen2.5-1.5B-Instruct weights, they differ in:

- **Quantization**: HF safetensors are BF16/FP16; the local artifact is Q4_K_M. The two binaries do not contain bit-identical weights.
- **Architecture version**: GGUF may bundle a slightly different metadata-only revision; safetensors may be from a different snapshot.
- **Loading path**: GGUF goes through `gguf.dequantize()`; safetensors go through `transformers.AutoModelForCausalLM`. The numerics at the residual-stream point may differ by tiny amounts due to these loading differences.

Therefore the X captured under Option A is **not** the X that the Q4_K_M GGUF would have produced. Calling Option A a "real runtime activation capture" would be a claim that is not supported by the artifacts. The accurate label is **HF-derived real-activation proxy**: the activation comes from a real prompt-derived forward pass, but the binary is the HF safetensors, not the local Q4_K_M GGUF. Only Option B can claim exact Q4_K_M GGUF runtime activation capture.

**Pros.**

- Proven methodology on this host (31I succeeded for 0.5B).
- Self-contained Python; no llama.cpp modification; no GGUF instrumentation.
- HuggingFace `transformers` is the standard, well-supported Python forward path; the 1.5B safetensors model is publicly available and small.
- W_ref source for replay is **unchanged** from the frozen 1.5B convention — Option A does NOT introduce a W_ref source pivot.
- Output is a single numpy array per (layer, prompt, token-position) — easy to inspect, hash, summarize, and re-feed into the existing 31BX/31BZ harness via a one-line change in X generation.

**Cons.**

- Requires a new model download (HF safetensors, NOT the existing local Q4_K_M GGUF). The 31BS download protocol requires explicit Matt approval. **A new download is required.**
- Adds a new dependency surface (HuggingFace `transformers` + `safetensors` + the Qwen2.5-1.5B safetensors model file). The current repo has HF libs installed in the venv (the 31BS log mentions active venv installs: `numpy==2.4.6`, `huggingface-hub==1.17.0`, `gguf==0.19.0`). `transformers` and `torch` may or may not be installed; must be checked at 31CD entry.
- HF safetensors model size is ~3.0 GB (BF16/fp16) — well within the 7 GB budget that 31BR planned for 1.5B, but still requires disk + bandwidth.
- The X captured under Option A is a **proxy**, not exact Q4_K_M GGUF runtime activation. Option A is NOT a path to "claim GGUF-runtime activation behavior".

**Cost estimate (planning only, NOT a runtime guarantee).**

- Download: ~3 GB, ~5-15 min on a normal connection.
- Model load (HF transformers, CPU): ~20-40 sec for 1.5B on this host (extrapolated from 31I's 0.5B load time of a few seconds; 1.5B is ~3x larger).
- Hook registration + prefill forward for a 5-token prompt: ~2-5 sec per prompt.
- Total for 31CD's small scope (1 prompt × 3 layers × 1 token position = 3 captures): well under 5 min wall clock.

**Approval dependencies.** 31CD's Option A is **conditional on explicit Matt approval** of the HF safetensors download AND on Matt's explicit choice of Option A at 31CD entry. The approval phrase template (per 31BR/31BS) is:

> "I approve Phase 31CD (Option A) to download Qwen2.5-1.5B-Instruct safetensors (BF16 or FP16) into `$SDI_MODEL_DIR/qwen2.5-1.5b-hf/`, max 4 GB budget, huggingface-cli or hf CLI download, forward-hook capture only, no Q2_K/SDIR artifact generation, no generation/inference, no runtime integration, no FP16 W_ref claim, W_ref for replay stays as Q4_K_M GGUF dequantized."

If Matt does NOT approve Option A or does not choose Option A at 31CD entry, **31CD falls back to Option B (llama.cpp instrumentation)** only with Matt's explicit choice of Option B. If Matt chooses neither A nor B, 31CD classifies as `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE` and the next permissible phase is **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested** (re-plan).

### 3.2 Option B — llama.cpp instrumentation capture (the only path that can claim exact Q4_K_M GGUF runtime activation capture)

**Mechanism.** Add a minimal hook / debug flag inside llama.cpp to dump the FFN input activation (pre-`ffn_up`/`ffn_gate`) for a specified layer and token position to a file or stdout, then invoke llama.cpp on a tokenized prompt and capture the dump. The captured X is then replayed through the existing 31BX/31BZ standalone harness using the **frozen Q4_K_M GGUF dequantized W_ref** (unchanged from 31BU/31BV/31BX/31BZ).

**Why Option B is preferred for exact runtime capture.** Option B captures X from the **same model binary** that produced the deployed artifacts (the local 1.5B Q4_K_M GGUF), via the **actual llama.cpp forward path**. The X is therefore an exact Q4_K_M GGUF runtime activation, not a proxy. Option B is the **only path that can claim exact Q4_K_M GGUF runtime activation capture** for the 1.5B.

**Pros.**

- Uses the **existing local 1.5B Q4_K_M GGUF** — no new download required.
- W_ref source for replay is **unchanged** from the frozen 1.5B convention (Q4_K_M dequantized).
- Closest to the actual llama.cpp inference path; X is produced by the actual runtime, not by a different Python model.
- Avoids any "proxy" framing — the X is exact Q4_K_M GGUF runtime activation.

**Cons.**

- **No existing precedent in this repo for llama.cpp Python integration** beyond the `libggml-base.so` ctypes calls for `quantize_row_q2_K_ref` / `dequantize_row_q2_K` (Q2_K encode/decode). There is no existing `libllama.so` Python binding in the repo.
- **Invasive**: requires modifying llama.cpp source, compiling, testing. The repo has no documentation of how llama.cpp is built on this host.
- **Compile risk**: rebuilding llama.cpp with a custom hook takes minutes-to-hours and can break on upstream API changes. 31CC cannot predict the cost.
- **Runtime complexity**: the dump format, hook placement, and token-position selection must all be implemented and tested.
- **llama.cpp modification rule**: 31CC is planning-only and is NOT supposed to modify llama.cpp. Even if Option B is selected, the actual llama.cpp modification would happen in 31CD (or a sub-phase), not in 31CC.
- **Forbids claim "llama.cpp integration"**: any successful Option B capture cannot be framed as a llama.cpp integration; it is a **diagnostic hook** for activation capture only. This is consistent with the existing 31AM/31AQ/31BC diagnostic-only framing.

**Cost estimate (planning only).**

- llama.cpp build: ~5-30 min depending on host (CPU build, no CUDA). One-time cost.
- Hook + dump wiring: ~30-60 min coding.
- Capture run: seconds.
- Total for 31CD: ~1-2 hours of engineering, mostly in 31CD not 31CC.

**Approval dependencies.** Option B does NOT require a new download. It DOES require Matt's **explicit choice of Option B at 31CD entry** AND explicit permission to modify and rebuild llama.cpp on this host (the `~/llama.cpp` directory is operator-owned). The 31CC plan does not require this approval now; the approval is required at 31CD entry.

### 3.3 Option C — Stop and re-plan via 31CC-R

**Mechanism.** Matt chooses not to proceed with either Option A or Option B at 31CD entry. 31CD classifies as `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE`. The next permissible phase becomes **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested**, which re-opens 31CC's planning decisions (e.g. considers a different capture path, a different scope, or defers real-activation capture to a later lane entirely).

**Pros.**

- Preserves the option to re-plan with a different approach.
- No forced commitment to either Option A or Option B.

**Cons.**

- Delays the largest open scientific question: do real prompt-derived activations show the same corrected-Q2_K sign pattern (memory-positive, finite, delta_cos non-negative or positive, MAE non-worsening or improving) as the prior synthetic-Gaussian tensor-harness results, on the selected 31CD prompt/layer/token scope?
- Requires a future 31CC-R phase to revisit the design.

**Verdict.** Option C is a valid choice at 31CD entry. It is the explicit "stop and re-plan" branch in the 31CD-entry decision point.

### 3.4 Option D — Minimal local forward-proxy (rejected)

**Mechanism.** Reconstruct enough of one layer's upstream (embedding + attention + residual-add) using small-numpy primitives to produce an "activation-like" vector, then replay it through the FFN block. The activation is derived from token IDs but the upstream layers are not a real model.

**Pros.**

- Self-contained.
- No download, no model load.
- Pure Python / numpy.

**Cons.**

- **Almost certainly incorrect**: reproducing even one layer's upstream (RMSNorm + 12-head attention + residual + RMSNorm) faithfully without the real model is highly error-prone. The "real activation" it produces is **not real**.
- **High risk of becoming an accidental runtime implementation** — once you reconstruct the upstream, you are implementing a tiny inference engine. This crosses into territory that 31CC is forbidden to enter.
- **Cannot be validated without running the real model for comparison**, which defeats the purpose.
- **No value beyond a deterministic synthetic input** — and the project already uses `np.random.default_rng` for that.

**Verdict: rejected.** Option D is excluded from 31CC's recommendation set. If a future phase wants to explore partial forward-proxies, that must be a new planning phase.

### 3.5 Option E — Stop and plan runtime / loader format first (out of scope)

**Mechanism.** Skip real-activation capture for now; instead design the runtime artifact format and loader for the static Q2_K + SDIR path, which is a prerequisite for any future llama.cpp integration claim.

**Pros.**

- Prepares the runtime path.
- Removes a prerequisite for llama.cpp integration (which is a separate lane from real-activation claims).

**Cons.**

- Delays the largest open scientific question: do real prompt-derived activations show the same corrected-Q2_K sign pattern (memory-positive, finite, delta_cos non-negative or positive, MAE non-worsening or improving) as the prior synthetic-Gaussian tensor-harness results, on the selected 31CD prompt/layer/token scope?
- The 31BE roadmap already deferred runtime-integration to a later lane; the artifact format hardening (31BF) is also a separate lane.
- 31CC is the recommended lane per 31CB's reasoning. Selecting Option E would mean deferring 31CC's purpose entirely.

**Verdict: out of scope for 31CC.** Option E is a legitimate next phase (already a permitted alternative in SOT Section 9: "31CC Runtime Artifact Format / Loader Planning"). It is not selected as 31CC's recommendation because the 31CB reasoning specifically identified real-activation capture as the largest remaining scientific gap.

---

## 4. 31CD-entry decision point (HARD GATE)

**31CD must not begin until Matt explicitly chooses one of A, B, or C at 31CD entry.** The default 31CD plan documents both A and B; the choice is Matt's. Until Matt chooses, 31CD is in the **`BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE`** state.

The three options at 31CD entry, in full:

### 4.1 Decision option A — HF-derived real-activation proxy

> Matt chooses Option A. Approves the HF safetensors download (per the 31BS approval phrase template). 31CD proceeds with the **HF-derived real-activation proxy** path.

- **X source**: HuggingFace `transformers` `forward_hook` on the FFN block of each selected layer, capturing the pre-FFN residual stream input. X is a real prompt-derived activation, but it comes from the **HF safetensors model binary**, not the local 1.5B Q4_K_M GGUF.
- **W_ref for replay**: the **frozen local 1.5B Q4_K_M GGUF dequantized W_ref**, matching 31BU/31BV/31BX/31BZ exactly. **No W_ref source pivot.**
- **W_low for replay**: corrected_ceil_per_row Q2_K W_low derived from the Q4_K_M GGUF W_ref (unchanged from 31BU/31BV/31BX/31BZ).
- **SDIR residual for replay**: ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual (unchanged from 31BU/31BV/31BX/31BZ).
- **Labeling**: any 31CD result under Option A must label the result as an **HF-derived real-activation proxy** and must NOT claim exact Q4_K_M GGUF runtime activation behavior.
- **Forbidden claims under Option A** (in addition to the standard 31CD forbidden-claims list):
  - ❌ No claim of exact Q4_K_M GGUF runtime activation behavior.
  - ❌ No claim of real llama.cpp runtime integration.
  - ❌ No claim of generation / inference quality.
  - ❌ No claim that the HF safetensors forward pass is equivalent to the Q4_K_M GGUF forward pass.

### 4.2 Decision option B — llama.cpp/GGUF activation capture instrumentation

> Matt chooses Option B. Approves the llama.cpp source modification and rebuild. 31CD proceeds with the **llama.cpp instrumentation** path.

- **X source**: minimal diagnostic hook inside llama.cpp to dump the FFN input activation (pre-`ffn_up`/`ffn_gate`) for a specified layer and token position. X is the **exact Q4_K_M GGUF runtime activation**.
- **W_ref for replay**: the **frozen local 1.5B Q4_K_M GGUF dequantized W_ref** (unchanged from 31BU/31BV/31BX/31BZ).
- **W_low for replay**: corrected_ceil_per_row Q2_K W_low derived from the Q4_K_M GGUF W_ref (unchanged).
- **SDIR residual for replay**: ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual (unchanged).
- **Labeling**: any 31CD result under Option B may claim **exact Q4_K_M GGUF runtime activation capture** (this is the only path that can make this claim).
- **Forbidden claims under Option B** (in addition to the standard 31CD forbidden-claims list):
  - ❌ No claim of real llama.cpp runtime integration (the hook is diagnostic, not an integration).
  - ❌ No claim of generation / inference quality.
- **Preferred if the goal is exact Q4_K_M GGUF runtime activation capture.** Operator may also choose this even if the goal is just a real-activation micro-probe, since it is the more faithful path.

### 4.3 Decision option C — Stop and re-plan

> Matt chooses neither A nor B. 31CD classifies as `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE`. The next permissible phase becomes **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested**.

- 31CC-R re-opens 31CC's planning decisions: it may consider a different capture path (e.g. a custom llama.cpp build script in-repo, a HuggingFace `transformers` model cache move, a small forward-proxy that is scoped to a clearly-validated subset of layers), or it may defer real-activation capture to a later lane entirely.
- No 31CD work runs in the C branch.

---

## 5. Default 31CD plan (applies to both Option A and Option B at entry)

The plan below is the default 31CD plan. It applies whether Matt chooses Option A or Option B at entry. Differences between Option A and Option B are limited to the **X source** and the **labeling / forbidden-claim additions**; the W_ref, W_low, SDIR, prompt, layers, token position, metrics, and artifact policy are identical.

### 5.1 31CD scope

- **Prompt set.** A small operator-selectable prompt set, defaulting to a single short prompt (operator may choose from the 31I prompt list or supply a new short factual / arithmetic / code prompt). All prompts in 31CD must be short (≤ 32 tokens), operator-authored or public-domain, with no PII. Prompts longer than 64 characters are redacted in the 31CD result JSON to `sha256[:16] + "..." + len_chars + "(redacted)"`. Recommended default prompt: `"The capital of France is"` (matches 31I prompt index 1; 6 tokens after BPE tokenization). Final prompt selection is an operator decision at 31CD entry.
- **Layers.** Default: `{0, 14, 27}` (mirrors 31BV's first 1.5B layer choice — top / middle / bottom of the 28-layer stack). Operator may reduce to `{0}` only for the first iteration if timing or scope is a concern.
- **Token position.** Default: **last token of the prefill input_ids** (the standard "prefill final-position" hidden state used in causal-LM activation probing). This is the most reproducible and well-defined position. Alternative positions (first token, middle token) are not in 31CD's default scope but may be added in a later phase.
- **Activation shape.** Per layer: `[1, 1536]` (batch=1, hidden=1536, fp32 stored on disk from the bf16/fp16 capture or from the float32 llama.cpp dump). For Option A with a longer prompt (e.g. 6 tokens), the operator may also capture all token positions and report per-position summary metrics, but the **default single-position replay is last-token-only**.
- **W_ref (UNCHANGED from 31BU/31BV/31BX/31BZ).** The existing local 1.5B Q4_K_M GGUF dequantized W_ref, applied uniformly for the Y_ref / Y_low / Y_sub replay. **No W_ref source pivot in either Option A or Option B.** Both options use the same W_ref.
- **W_low (UNCHANGED).** Corrected_ceil_per_row Q2_K W_low derived from the Q4_K_M GGUF W_ref (the actual deployed W_low). Identical to 31BU/31BV/31BX/31BZ.
- **SDIR residual (UNCHANGED).** ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual. Identical to 31BU/31BV/31BX/31BZ.
- **Replay (UNCHANGED).** For each (layer, token position, prompt) captured X, replay through:
  - W_ref MLP (Q4_K_M dequantized from the local GGUF)
  - corrected Q2_K-only MLP (corrected_ceil_per_row Q2_K for all 3 families)
  - corrected Q2_K + SDIR MLP (per `corrected_q2k_policy_v1`: ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0; ffn_down W_low only, no residual)
  using the canonical 31BT formula `Y = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T`.
- **Allowed metrics.** Per pair (layer, prompt, token-position):
  - `cos_low = cosine(Y_ref, Y_low)`, `cos_sub = cosine(Y_ref, Y_sub)`, `delta_cos = cos_sub - cos_low`.
  - `MAE_low = mean(|Y_ref - Y_low|)`, `MAE_sub = mean(|Y_ref - Y_sub|)`, `MAE_delta = MAE_sub - MAE_low`.
  - `finite`: bool for Y_ref / Y_low / Y_sub.
  - `severe`: `delta_cos < -0.05` (per 31BX/31BZ threshold).
  - `memory_margin`: per-layer margin under Q4 budget for the corrected Q2_K + SDIR path (informational; not 31CD's primary metric).
- **Disallowed metrics.** No inference latency, no per-pair wall-clock claim, no throughput claim, no generation-quality claim, no behavior-recovery claim, no production-readiness claim, no llama.cpp-integration claim.
- **31CD result JSON mandatory fields** (regardless of option):
  - `option_selected`: `"A"` or `"B"` (Matt's choice at 31CD entry).
  - `w_ref_source`: `"$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf (Q4_K_M dequantized; matches 31BU/31BV/31BX/31BZ convention)"` — for both Option A and Option B.
  - `w_low_source`: `"corrected_ceil_per_row Q2_K from Q4_K_M GGUF W_ref (unchanged from 31BU/31BV/31BX/31BZ)"`.
  - `sdir_source`: `"ffn_up+ffn_gate SDIR k=0.5%, alpha=1.0, no ffn_down residual (unchanged from 31BU/31BV/31BX/31BZ)"`.
  - `x_source` (Option A): `"HuggingFace transformers forward-hook on Qwen2.5-1.5B-Instruct safetensors; HF-derived real-activation proxy; NOT exact Q4_K_M GGUF runtime activation"`.
  - `x_source` (Option B): `"llama.cpp instrumentation hook on Qwen2.5-1.5B-Instruct Q4_K_M GGUF; exact Q4_K_M GGUF runtime activation"`.
  - `model_path_observed_redacted`: env-var form, never the operator-specific absolute path.
  - `sha256_of_raw_X_per_layer`: dict per layer.
  - `captured_shape_per_layer`: dict per layer (default `[1, 1536]`).
  - `captured_dtype`: `"float32"` (dequantized from bf16/fp16 on capture or from float32 llama.cpp dump).

### 5.2 31CD activation artifact policy

**Conservative default (recommended):**

- Raw activation arrays (`X[1, 1536]` per (layer, prompt, token-position)) are written to a **temp directory** (e.g. `tempfile.mkdtemp(prefix="phase31cd_")`) and are **NOT** committed to the repo.
- The 31CD result JSON contains only **summary metrics and metadata** (see Section 5.1 for the full mandatory-field list).
- A `gitignore` recommendation: add `/data/phase31cd_*` to `.gitignore` (if not already covered) so any accidentally-created activation fixture is not staged.
- **Activation data counts as generated artifact** under the 31CC scope: it is a side-effect of the model forward pass, similar to a Q2_K or SDIR blob. It is **not** committed by default.

**Exception (requires explicit Matt approval at 31CD time):** if the operator wishes to commit a small activation fixture for reproducibility (e.g. `data/phase31cd_real_activations.npz` with shape `[3, 1, 1536]` for 3 layers × 1 prompt × 1 position × 1536 hidden), the fixture must be:

- Tiny (target < 100 KB, hard cap 1 MB; 3 × 1 × 1536 × 4 bytes = ~18 KB, well under both).
- Documented in 31CD's SOT entry with a `committed_activation_fixture: true` field and a `committed_fixture_path: data/phase31cd_real_activations.npz` field.
- Covered by a `data/phase31cd_*.gitkeep` or equivalent `.gitignore` exemption so the fixture is the only thing committed under that path.
- Stored under `data/` with a `phase31cd_` prefix and a `.npz` extension (consistent with the existing `data/PHASE31I_activations.npz` precedent).

31CC does NOT pre-approve this exception. The 31CD entry approval includes the activation artifact disposition decision.

### 5.3 31CD prompt set (planning)

The default 31CD prompt set is a **single prompt** for the first micro-probe iteration. The operator may extend to a small set (≤ 5 prompts) in subsequent iterations. The recommended default prompt is:

> `"The capital of France is"` (6 BPE tokens, public-domain factual, no PII, matches 31I prompt index 1)

Other candidates from the 31I prompt set that are short, public-domain, and PII-free (operator may select at 31CD entry):

- `"Hi"` (1 token)
- `"2+2="` (4 tokens)
- `"The reason for this is"` (6 tokens)
- `"Hey there!"` (3 tokens)

The operator may also supply a custom prompt at 31CD entry. Any prompt longer than 64 characters is redacted in the 31CD result JSON.

### 5.4 31CD stop conditions

- Regression fails (per SOT Section 8): `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN` is the required state.
- `SDI_MODEL_DIR` unset.
- Model file missing (HF safetensors for Option A; GGUF for Option B).
- **No option chosen at 31CD entry** (A / B / C). 31CD is `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE` until Matt explicitly chooses one.
- Capture path unavailable (HuggingFace `transformers` not importable for Option A; llama.cpp hook not built for Option B).
- Activation capture requires an unapproved dependency (e.g. `torch` not in the active venv and not approved for install).
- Activation capture requires invasive runtime patch beyond approved scope (for Option B: the hook touches llama.cpp source; the scope of the touch must be approved at 31CD entry).
- Captured X has shape != `[1, 1536]` (or != `[tokens, 1536]` if multi-position is approved at 31CD entry).
- Captured X contains NaN or Inf.
- Replay formula mismatch (Y_ref / Y_low / Y_sub shapes inconsistent with the 31BT canonical formula output shape).
- Q2_K encode backend fails.
- SDIR encode/decode fails.
- Any raw activation artifact exceeds 1 MB (the hard cap for committed fixtures; uncommitted raw arrays are not capped).
- Any model file, HF cache, or raw activation blob would be committed accidentally (operator's responsibility to verify; the runner must use a clearly non-canonical path field name like `model_path_observed_redacted`).
- Any inference/generation/quality/behavior/speed/runtime claim would be added to the 31CD result.
- Any claim of exact Q4_K_M GGUF runtime activation behavior would be added to an Option A result.
- Any claim of real llama.cpp runtime integration would be added to either an Option A or an Option B result.

### 5.5 31CD classification list

- `PASS_31CD_REAL_ACTIVATION_MICRO_REPLAY_CLEAN` — capture succeeds, replay succeeds, all metrics finite, memory-positive (under the corrected Q2_K + SDIR path), delta_cos >= 0 on all pairs, MAE_sub < MAE_low on all pairs, 0 severe regressions. No forbidden artifacts / claims. The headline: corrected Q2_K + SDIR path is non-degrading on the captured real-activation micro-probe.
- `PARTIAL_31CD_REAL_ACTIVATION_MICRO_REPLAY_MINOR_FAILURES` — capture and replay succeed; memory-positive on all pairs; all finite; but 1+ pair has delta_cos < 0 (non-severe) or MAE_sub >= MAE_low (non-severe). The headline: real-activation micro-probe is broadly consistent with the random-Gaussian micro-probe (31BU/31BV) at the same layers, with minor non-severe deviations.
- `PARTIAL_31CD_REAL_ACTIVATION_CAPTURE_ONLY` — activation capture succeeds for all planned (layer, prompt, token-position) tuples, but replay is not completed (Q2_K backend fail, SDIR fail, regression fail, etc.). The headline: real-activation capture methodology is viable on 1.5B; the replay step is blocked.
- `PARTIAL_31CD_REPLAY_FORMULA_MISMATCH` — capture succeeds; replay produces output that is not consistent with the 31BT canonical MLP formula shape (unexpected Y_ref / Y_low / Y_sub shapes; e.g. `Y_sub` shape is `[1, 1536]` but `cos_low` is degenerate).
- `BLOCKED_31CD_NO_OPTION_SELECTED_AT_ENTRY` — Matt has not yet chosen between Option A, Option B, or Option C at 31CD entry. 31CD is in the waiting-for-decision state. (Distinct from `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE` which is the post-decision state if both A and B are declined.)
- `BLOCKED_31CD_CAPTURE_PATH_UNAVAILABLE` — Matt has explicitly chosen Option C (or has declined both Option A and Option B). No safe capture path exists. The next permissible phase is 31CC-R (re-plan).
- `BLOCKED_31CD_HF_DOWNLOAD_NOT_APPROVED` — Option A is the chosen mechanism but the operator has not approved the HF safetensors download. 31CD may switch to Option B (fallback) only with operator's explicit switch at 31CD entry.
- `BLOCKED_31CD_LLAMA_CPP_MODIFICATION_NOT_APPROVED` — Option B is the chosen mechanism but the operator has not approved the llama.cpp source modification. 31CD may switch to Option A (default) only with operator's explicit approval of the HF safetensors download at 31CD entry.
- `BLOCKED_31CD_SDI_MODEL_DIR_UNSET` — `SDI_MODEL_DIR` env var is not set on the operator's host.
- `BLOCKED_31CD_MODEL_FILE_MISSING` — the model file (HF safetensors for Option A; GGUF for Option B) is not present at the expected path.
- `BLOCKED_31CD_Q2K_BACKEND_FAIL` — `q2k_is_available()` returns false; `libggml-base.so` cannot be loaded.
- `BLOCKED_31CD_SDIR_FAIL` — `encode_sdir` / `decode_sdir` raises or returns malformed bytes.
- `BLOCKED_31CD_TOKENIZER_OR_MODEL_LOAD_FAIL` — HuggingFace `transformers.from_pretrained()` fails (Option A) or the GGUFReader fails to load the model (Option B).
- `BLOCKED_31CD_FORWARD_HOOK_FAIL` — the registered hook does not fire, fires on the wrong tensor, fires with a NaN/Inf tensor, or fires with a non-`[*, 1536]` tensor (Option A).
- `BLOCKED_31CD_LLAMA_CPP_BUILD_FAIL` — the llama.cpp rebuild with the Option B hook fails to compile (Option B).
- `BLOCKED_31CD_REGRESSION_FAIL` — pre-flight or post-edit regression fails.
- `BLOCKED_31CD_ALLOWED_PHASE_CONFLICT` — SOT Section 9 does not list 31CD as the current allowed next phase (i.e. a future phase has been completed that supersedes 31CC, or a future allowed-phase change has been recorded).

### 5.6 31CD allowed claims (if PASS or PARTIAL)

These claims are allowed for both Option A and Option B unless explicitly qualified:

- "Corrected Q2_K + SDIR (per `corrected_q2k_policy_v1`) did not degrade MLP output similarity over Q2_K-only on a tiny real prompt-derived activation micro-probe on Qwen2.5-1.5B at the selected layer(s) and token position(s)." (Allowed when delta_cos >= 0 on all pairs.)
- "The corrected Q2_K policy remains memory-positive on the real-activation micro-probe (per-layer margin under Q4 budget is non-negative)." (Allowed when memory-positive on all pairs.)
- "Real-activation capture on Qwen2.5-1.5B is viable via [HuggingFace `transformers` forward hooks | llama.cpp instrumentation]." (Allowed when the corresponding capture path succeeded; methodology-only, not a runtime claim.)

**Option-B-only additional allowed claim:**

- "Exact Q4_K_M GGUF runtime activation capture on Qwen2.5-1.5B is viable via llama.cpp instrumentation." (Allowed for Option B only; not allowed for Option A because Option A is a proxy.)

### 5.7 31CD forbidden claims (still forbidden, even if PASS)

- No generation quality claim.
- No model behavior claim.
- No inference latency / throughput / wall-clock claim beyond informational wall-clock note (no per-token, no per-prompt, no end-to-end).
- No runtime integration claim.
- No production readiness claim.
- No broader activation-distribution claim (no claim that the micro-probe is representative of all real prompts, all token positions, all layers, all attention patterns).
- No all-token / all-layer / all-prompt real-activation claim.
- No claim that real generation will work.
- No claim that the micro-probe result transfers to other prompts, layers, token positions, or models.
- No 0.5B-vs-1.5B real-activation comparison claim.
- **No "real activations behave like synthetic Gaussian" claim.** The 31CD result text MAY use **permitted description language** that says real-activation replay is **directionally consistent** with prior synthetic-Gaussian tensor-harness results only within the selected 31CD prompt/layer/token scope, and only if the real-activation metrics show the same sign pattern: memory-positive, finite, delta_cos non-negative or positive, and MAE non-worsening or improving — but this is **not** a scientific claim, only descriptive language.
- **Explicit claim-language reinforcements** (these are not separate items; they reinforce the above):
  - **(1) Do not claim real activations behave like synthetic Gaussian.**
  - **(2) Do not claim activation-distribution equivalence.**
  - **(3) Do not claim transfer beyond the selected prompt/layer/token scope.**
- No model files committed; no HF cache files committed; no raw activation arrays committed (without explicit operator approval per Section 5.2).

**Option-A-only additional forbidden claims:**

- No claim of exact Q4_K_M GGUF runtime activation behavior (Option A is an HF-derived proxy, not a GGUF runtime capture).
- No claim that the HF safetensors forward pass is equivalent to the Q4_K_M GGUF forward pass.
- No claim of real llama.cpp runtime integration (Option A does not touch llama.cpp at all).
- No claim of generation / inference quality (this is universally forbidden, restated for Option A emphasis).

**Option-B-only additional forbidden claims:**

- No claim of real llama.cpp runtime integration (the Option B hook is a diagnostic capture, not an integration).
- No claim of generation / inference quality (universally forbidden, restated for Option B emphasis).

---

## 6. What 31CC does NOT do (upheld)

- ❌ Does not run activation capture.
- ❌ Does not run model inference beyond metadata inspection (the 31BS-style metadata probe is already complete; 31CC does not re-run it).
- ❌ Does not run generation / sampling / autoregressive decoding.
- ❌ Does not modify llama.cpp source.
- ❌ Does not create forward hooks (anywhere; in transformers, in llama.cpp, or in any other framework).
- ❌ Does not download any model file (no HF safetensors, no GGUF, no anything).
- ❌ Does not generate Q2_K or SDIR artifacts.
- ❌ Does not run validation / aggregate / generation / runtime integration.
- ❌ Does not make quality / performance / runtime / behavior claim.
- ❌ Does not commit, push, tag, or download without explicit Matt approval.
- ❌ Does not modify scientific results from prior phases (0.5B 31BN/31BM, 1.5B 31BZ/31CA).
- ❌ Does not modify the `corrected_q2k_policy_v1` package.
- ❌ Does not modify the existing 1.5B Q4_K_M W_ref convention. (Earlier 31CC draft had a "W_ref source pivot" idea for Option A; this is **explicitly retracted** in this revised planning doc. Option A is an HF-derived real-activation **proxy** that replays through the frozen Q4_K_M GGUF W_ref — there is NO W_ref source pivot.)

---

## 7. What 31CC DOES do

- ✅ Reads SOT, policy package, existing 1.5B runners, prior 31I capture, manifest runtime, regression suite.
- ✅ Reviews the four candidate capture strategies (A, B, C, D, E).
- ✅ Selects Option A (HF forward hooks, **HF-derived real-activation proxy**) as one of two valid 31CD options, with Option B (llama.cpp instrumentation, exact Q4_K_M GGUF runtime activation capture) as the other.
- ✅ Defines the exact 31CD micro-probe scope (prompt, layers, token positions, batch, shape, metrics, artifact policy, stop conditions, classifications, allowed claims, forbidden claims) — identical across Option A and Option B except for X source and labeling.
- ✅ Documents the W_ref / W_low / SDIR convention is **unchanged** across 31CD (Q4_K_M GGUF, matching 31BU/31BV/31BX/31BZ exactly) — explicitly retracting any earlier W_ref source pivot idea.
- ✅ Defines a hard 31CD-entry decision point (A / B / C) with explicit approval requirements for each.
- ✅ Writes this planning doc + planning JSON.
- ✅ Updates SOT Section 0 (current next phase), Section 3 (new 31CC entry), Section 9 (advance to 31CD).
- ✅ Runs pre-flight regression before SOT/doc/result edits; runs post-edit regression; both must pass with `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=True`.
- ✅ Produces a PRE-COMMIT REPORT; **stops and waits for Matt approval** before any commit, push, or tag.

---

## 8. Selected next phase and updated SOT pointers

- **Current allowed next phase (after 31CC):** **Phase 31CD — Qwen2.5-1.5B Real-Activation Capture Micro-Probe, only if explicitly requested.** 31CD has not yet begun — Matt must explicitly choose between Option A (HF-derived real-activation proxy), Option B (llama.cpp/GGUF exact runtime activation capture), or Option C (stop and re-plan via 31CC-R) at 31CD entry.
- **Alternative 31CC option** (recorded in SOT Section 9 as a permissible alternative, NOT selected by 31CC): 31CC Runtime Artifact Format / Loader Planning (i.e. Option E above). The 31CC selected lane is real-activation capture planning per 31CB's reasoning.
- **If 31CD is blocked at entry** (e.g. operator declines both Option A and Option B), the next permissible phase becomes **Phase 31CC-R — Real-Activation Capture Planning Repair, only if explicitly requested** (the canonical "re-plan" phase when a planning phase is blocked at execution).
- **Frozen state preserved through 31CC:** 0.5B `corrected_q2k_policy_v1` evidence tier (`PASS_31BN` / `PASS_31BM` / `PASS_31BO`); 1.5B `corrected_q2k_policy_v1` evidence tier (`PASS_31BZ` / `PASS_31CA`). All accepted numeric results are unchanged. The `corrected_q2k_policy_v1` package is unchanged. The 0.5B and 1.5B checkpoints (tags `phase31bn-corrected-q2k-full-aggregate-checkpoint`, `phase31ca-1_5b-corrected-q2k-aggregate-checkpoint`) are unchanged. No new tag is created in 31CC. No new commit is created in 31CC.

---

## 9. Open question for the operator at 31CD entry (NOT a 31CC question)

The 31CC plan defers the following decisions to 31CD entry (after explicit 31CC approval and before 31CD execution):

- **Hard decision: Option A / B / C** (Section 4). Matt must choose. This is the 31CD-entry decision point.
- **If Option A is chosen:** HF safetensors download approval (per the 31BS approval phrase template).
- **If Option B is chosen:** llama.cpp modification and rebuild approval.
- **Final prompt selection** (31CD-specific, not 31CC): operator picks the prompt(s) from the planning set in Section 5.3 or supplies a new short prompt.
- **Final layer selection** (31CD-specific, not 31CC): operator confirms `{0, 14, 27}` or reduces to `{0}` for the first iteration.
- **Activation artifact disposition** (31CD-specific, not 31CC): operator confirms the conservative no-commit default (Section 5.2) or explicitly approves a small committed fixture with a target size ≤ 1 MB.

These are recorded as 31CD-entry questions, not 31CC questions. 31CC does not block on them.

---

## 10. Audit trail

- 31CC reads: SOT Section 0, SOT Section 0.A, SOT Section 3 (31BT, 31BU, 31BV, 31BW, 31BX, 31BY, 31BZ, 31CA, 31CB entries), SOT Section 9, SOT Sections 7-8, SOT Section 14, `docs/CORRECTED_Q2K_POLICY_PACKAGE.md`, `src/results/CORRECTED_Q2K_POLICY_PACKAGE.json`, `src/corrected_q2k_policy.py`, `src/phase31x_manifest_runtime.py`, `src/phase31bt_1_5b_orientation_parity_micro_probe.py`, `src/phase31bx_1_5b_corrected_q2k_stratified_layer_probe.py`, `src/phase31bz_1_5b_corrected_q2k_full_layer_two_seed_aggregate.py`, `data/phase31i_capture.log`, `data/phase31i_sweep.log`, `data/phase31i_capture.sh`, `data/PHASE31I_activations.npz` (size + keys via `np.load` metadata only; no payload extraction), `tests/run_source_of_truth_regression.py`, `README.md`.
- 31CC writes: this document (`docs/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.md`), `src/results/PHASE31CC_REAL_ACTIVATION_CAPTURE_PLANNING.json`, `README.md` (conservative patch: advance "current scientific next phase" from 31CC to 31CD; still defers to SOT), and SOT edits (Section 0, Section 3 31CC entry, Section 9 advance).
- 31CC runs: pre-flight `python3 -m tests.run_source_of_truth_regression` (before SOT/doc/result edits), post-edit `python3 -m tests.run_source_of_truth_regression` (after SOT/doc/result edits). Both must report `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0`, `readme_drift_guard.passed=True`.
- 31CC does NOT run: any model forward pass, any generation, any sampling, any llama.cpp modification, any activation capture, any Q2_K/SDIR artifact generation, any model file download, any model file commit, any push, any tag creation, any 31CD execution.

---

## 11. README drift guard impact

The README is patched by 31CC (this revision). The README's "Current scientific next phase" line and the "Status / Drift Guard" section are updated to point to **Phase 31CD** (not 31CC). The README remains a high-level orientation page and still defers to SOT Section 0 and Section 9 for the full audit trail. No expanded report, no claim-by-claim summary, no policy-package deep-dive — the README is intentionally brief and only names the current scientific next phase.

The 31CC planning doc + planning JSON are the full planning deliverables; the README is a one-paragraph pointer.
