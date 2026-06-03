# Phase 31BS — Approved Larger-Model Download / Metadata Verification

**Classification:** `PASS_31BS_1_5B_DOWNLOAD_METADATA_VERIFIED`
**Repo:** sdi-substitutive
**Branch:** `master`
**HEAD at preflight:** `b0bd2d2f74508e8decdc32bc19550fd33f169cf6`
**Approval:** Explicit (received from Matt for Qwen2.5-1.5B-Instruct Q4_K_M only, into `$SDI_MODEL_DIR/qwen2.5-1.5b-official/`, max 7 GB budget, metadata-probe only, no tensor validation)

> Note: an earlier 31BS preflight run in this session was classified `BLOCKED_31BS_SDI_MODEL_DIR_UNSET` because `SDI_MODEL_DIR` was unset in the agent's process. The blocker was a pure environment-preflight miss, NOT frozen into `SOURCE_OF_TRUTH.md`. This document **replaces/overwrites** that earlier blocker report with the successful rerun outcome, per Matt's instruction "Do not keep the blocker docs as the final 31BS artifacts if the rerun succeeds."

---

## Goal

Download the approved Qwen2.5-1.5B-Instruct Q4_K_M GGUF into the approved destination and verify metadata only. This is not a validation phase.

---

## Environment Preflight

The local operator exported `SDI_MODEL_DIR` in the same process/session as the rerun preflight (e.g.):

```bash
export SDI_MODEL_DIR=/media/matthew-villnave/VL_usb/models   # operator-specific, not committed
```

Verification:

| Check | Result |
|-------|--------|
| `SDI_MODEL_DIR` set | yes |
| `SDI_MODEL_DIR` exists | yes |
| `df -h "$SDI_MODEL_DIR"` | `/dev/sda1` 115G, used 74G, **avail 42G** (well over 7 GB budget) |
| Destination filesystem free before download | 42 GB |
| Destination filesystem free after download | 41 GB |
| Operator-exported in same process as preflight | yes |

Candidate filesystems with ≥7 GB free (from earlier disk observations):
- root NVMe (`/`) — 96 GB free
- VL_usb (`/media/matthew-villnave/VL_usb`) — 42 GB free

The operator selected VL_usb. Destination root: `$SDI_MODEL_DIR/qwen2.5-1.5b-official/`. No filesystem was auto-selected by the agent.

**Active venv adjustments** (env-only; not code changes):
- `numpy==2.4.6` installed via `uv pip install --python .../venv/bin/python3 numpy` (regression imports numpy).
- `huggingface-hub==1.17.0` installed via `uv` (provides the download toolchain).
- `gguf==0.19.0` installed via `uv` (provides `GGUFReader`).

These are env-level installations into the active venv; no project source files were modified by these installs.

---

## Download

The 31BS spec requested `huggingface-cli download ...`. The installed `huggingface-cli` printed:

> Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead. `hf` is already installed!

`hf` is the non-deprecated successor with the same arguments. The agent used the new CLI (same arguments) and recorded the substitution explicitly.

### Exact filename identification

```python
from huggingface_hub import list_repo_files
for f in list_repo_files("Qwen/Qwen2.5-1.5B-Instruct-GGUF"):
    if "Q4_K_M" in f or "q4_k_m" in f.lower():
        print(f)
```

Output (single match, no sharding):
```
qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### Command used

```
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir "$SDI_MODEL_DIR/qwen2.5-1.5b-official/"
```

(The spec's exact text was `huggingface-cli`; the new `hf` CLI uses identical arguments. Same upstream toolchain.)

### Download result

| Field | Value |
|-------|-------|
| **Downloaded file path** | `$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| **Downloaded file size** | **1,117,320,736 bytes (1.04 GiB)** — well under 7 GB budget |
| **GGUF magic** | `GGUF` (0x47475546) at offset 0, version field = 3 (GGUF v3) ✓ |
| **Extra model files downloaded** | none — only the Q4_K_M file |
| **Sharding** | none — single-file model |
| **HF download cache** | `.cache/huggingface/` created by `hf` (cache, not a model file) — outside repo, will not be committed |

Disk usage moved from 74 GB used / 42 GB free to 75 GB used / 41 GB free (i.e. ~1 GB consumed — consistent with the 1.04 GiB model file).

**Warning emitted:** `You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.` This is a non-issue for this phase — the file was downloaded successfully and verified.

---

## GGUFReader Probe

| Field | Value |
|-------|-------|
| Opened successfully | yes |
| GGUF format version | 3 |
| KV count | 26 |
| Tensor count | 339 |

---

## Metadata Table

| Key | Value | Notes |
|-----|-------|-------|
| `GGUF.version` | 3 | GGUF v3 |
| `GGUF.kv_count` | 26 | |
| `GGUF.tensor_count` | 339 | |
| `general.architecture` | `qwen2` | Qwen2.5 is a Qwen2-arch model |
| `general.name` | `qwen2.5-1.5b-instruct` | |
| `general.file_type` | 15 | `LlamaFileType.MOSTLY_Q4_K_M = 15` (per GGUF enum) — Q4_K_M confirmed by per-tensor types below |
| `general.finetune` | `qwen2.5-1.5b-instruct` | |
| `general.quantization_version` | 2 | |
| `general.size_label` | `1.8B` | |
| `general.type` | `model` | |
| `general.version` | `v0.1` | |
| `qwen2.attention.head_count` | 12 | Q heads |
| `qwen2.attention.head_count_kv` | 2 | KV heads (GQA, 6:1 ratio) |
| `qwen2.attention.layer_norm_rms_epsilon` | 9.999999974752427e-07 | (≈ 1e-6) |
| `qwen2.block_count` | 28 | n_layers |
| `qwen2.context_length` | 32768 | |
| `qwen2.embedding_length` | 1536 | hidden_size |
| `qwen2.feed_forward_length` | 8960 | intermediate_size |
| `qwen2.rope.freq_base` | 1000000.0 | |
| `tokenizer.ggml.add_bos_token` | false | |
| `tokenizer.ggml.bos_token_id` | 151643 | |
| `tokenizer.ggml.eos_token_id` | 151645 | |
| `tokenizer.ggml.model` | `gpt2` | BPE tokenizer type |
| `tokenizer.ggml.padding_token_id` | 151643 | |
| `tokenizer.ggml.pre` | `qwen2` | |
| `tokenizer.chat_template` | (2509 chars; full dump skipped) | chat template present |
| `tokenizer.ggml.tokens` | 151,937 entries (full dump skipped) | |
| `tokenizer.ggml.merges` | 151,388 entries (full dump skipped) | |
| `tokenizer.ggml.token_type` | 75,969 entries (full dump skipped) | |

---

## Raw Tensor Shape Table — Layer 0 (FFN families of interest)

| Tensor | Raw GGUFReader shape | n_elements | tensor_type | elements per family |
|--------|----------------------|-----------:|-------------|---------------------:|
| `blk.0.ffn_up.weight` | `[1536, 8960]` | 13,762,560 | `Q4_K` | 13,762,560 |
| `blk.0.ffn_gate.weight` | `[1536, 8960]` | 13,762,560 | `Q4_K` | 13,762,560 |
| `blk.0.ffn_down.weight` | `[8960, 1536]` | 13,762,560 | `Q6_K` | 13,762,560 |

### Other layer-0 tensors (context only — not validated)

| Tensor | Raw shape | tensor_type |
|--------|-----------|-------------|
| `blk.0.attn_norm.weight` | `[1536]` | F32 |
| `blk.0.attn_q.weight` | `[1536, 1536]` | Q4_K |
| `blk.0.attn_q.bias` | `[1536]` | F32 |
| `blk.0.attn_k.weight` | `[1536, 256]` | Q4_K |
| `blk.0.attn_k.bias` | `[256]` | F32 |
| `blk.0.attn_v.weight` | `[1536, 256]` | Q6_K |
| `blk.0.attn_v.bias` | `[256]` | F32 |
| `blk.0.attn_output.weight` | `[1536, 1536]` | Q4_K |
| `blk.0.ffn_norm.weight` | `[1536]` | F32 |

---

## Orientation Caveat (Unresolved by 31BS)

Raw GGUFReader tensor shape display is reader/storage-specific and is **not** canonical. Per `SOURCE_OF_TRUTH.md` Section 3 (31BQ entry) and the 31BR plan, canonical MLP orientation is:

| Family | Role | Canonical artifact orientation |
|--------|------|-------------------------------|
| `ffn_up` | hidden → intermediate | (d_out=intermediate=8960, d_in=hidden=1536) |
| `ffn_gate` | hidden → intermediate | (d_out=intermediate=8960, d_in=hidden=1536) |
| `ffn_down` | intermediate → hidden | (d_out=hidden=1536, d_in=intermediate=8960) |

Comparing raw GGUFReader layer-0 shapes against the canonical orientation:

| Family | Raw display | Canonical (d_out, d_in) | Raw-vs-canonical observation |
|--------|-------------|-------------------------|-------------------------------|
| `ffn_up` | `[1536, 8960]` | `(8960, 1536)` | Raw looks like the **transposed** form (i.e. storage ordering may be `[d_in, d_out]`) |
| `ffn_gate` | `[1536, 8960]` | `(8960, 1536)` | Same as ffn_up |
| `ffn_down` | `[8960, 1536]` | `(1536, 8960)` | Raw looks like the **transposed** form (storage `[d_in, d_out]`) |

**Conclusion:** 31BS does **not** resolve orientation. The raw shapes suggest the GGUF storage ordering is `[d_in, d_out]` for these weights, but this is a hypothesis that must be parity-tested (e.g. by comparing raw vs transposed against a model-derived W_ref in a future orientation parity micro-probe) before any tensor validation can run. **No orientation claim is made by 31BS.**

A future **Phase 31BT — Qwen2.5-1.5B Orientation Parity Micro-Probe** (layer 0 only, tiny seed, orientation formula only, no aggregate validation) would resolve this — only if explicitly requested.

---

## Expectations vs Observations

| Check | Result |
|-------|--------|
| Is Qwen2/Qwen2.5 architecture? | ✓ yes (`qwen2`) |
| `n_layers` (28) plausible? | ✓ yes (consistent with public Qwen2.5-1.5B spec) |
| `hidden_size` (1536) plausible? | ✓ yes (consistent with public spec) |
| `intermediate_size` (8960) plausible? | ✓ yes (consistent with public spec) |
| `context_length` (32768) plausible? | ✓ yes (consistent with public spec) |
| Single file or sharded? | single file, no sharding |
| Metadata sufficient for next phase? | yes |
| Safe to proceed to orientation parity micro-probe later? | yes |
| GQA configuration | 12 Q heads, 2 KV heads (6:1 ratio) |

---

## File Location (env-var form, not committed as canonical)

```
$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Concrete operator-set value (operator-specific, not committed):
`/media/matthew-villnave/VL_usb/models/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf`

The destination is **outside the repo** (`git ls-files "$SDI_MODEL_DIR/..."` errors with "is outside repository"). Git status shows no qwen file. The model is untracked and outside the working tree.

---

## Actions NOT Taken (Upheld)

- no tensor harness validation executed
- no full tensor dequantization for validation (metadata/shape listing only)
- no Q2_K artifacts generated
- no SDIR artifacts generated
- no orientation parity probe (deferred to 31BT, only if explicitly requested)
- no anchor probe (deferred, only if explicitly requested)
- no aggregate validation (deferred, only if explicitly requested)
- no model files committed to the repo (model lives under `$SDI_MODEL_DIR`, outside repo)
- no commit/push/tag without explicit Matt approval (this is a PRE-COMMIT REPORT, not a commit)

---

## Prior Accepted Numeric Results — Unchanged

31BS is a download + metadata probe only. No tensor validation was executed, no Q2_K/SDIR artifacts were generated, no 0.5B reference metrics were re-derived. The 0.5B Q2_K and Q4_K_M reference metrics in accepted prior phases (31AY / 31BA / 31BM / 31BN / 31BO) remain unchanged.

---

## Next Allowed Phase After 31BS

**Phase 31BT — Qwen2.5-1.5B Orientation Parity Micro-Probe**, only if explicitly requested.
- Layer 0 only.
- Tiny seed/sample.
- Orientation formula only.
- No aggregate validation.

---

## Stale Untracked Files (not committed, not staged)

- `docs/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.md`
- `src/phase31bh_q2k_clean_reproduction.py`
- `src/results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json`
- `src/results/PHASE31BH_R2_Q2K_ANCHOR_REPRO_FIX.json`
- `src/results/PHASE31BJ_ANCHOR_ONLY.json`

---

## Forbidden Claims (Upheld)

- no model quality recovery claim
- no behavior recovery claim
- no speedup claim
- no full-model runtime memory savings claim
- no llama.cpp integration claim
- no production readiness claim
- no inference/generation claim
- no larger-model validation/result claim (31BS explicitly does not validate the 1.5B model; only metadata)
- no runtime-ready output-residual claim
- no claim beyond metadata verification — 31BS explicitly states that orientation is unresolved and deferred to 31BT
