# Phase 31BT — Qwen2.5-1.5B Orientation Parity Micro-Probe

**Classification:** `PASS_31BT_1_5B_ORIENTATION_PARITY_CONFIRMED`
**Repo:** sdi-substitutive
**Branch:** `master`
**HEAD at preflight:** `963d4a893dd6d613731d9f42de8068eab40eb5c0`
**Scope:** orientation parity micro-probe only — layer 0 MLP, tiny deterministic input, **no anchor/aggregate/generation, no Q2_K/SDIR artifacts**.

---

## Goal

Determine the correct orientation interpretation for Qwen2.5-1.5B layer-0 MLP tensors before any future tensor validation. This is an orientation-only micro-probe.

---

## Environment Preflight

```bash
export SDI_MODEL_DIR=/media/matthew-villnave/VL_usb/models   # operator-specific, not committed
```

| Check | Result |
|-------|--------|
| `SDI_MODEL_DIR` set | yes (operator-exported in same process) |
| `SDI_MODEL_DIR` exists | yes |
| Model file exists | yes |
| Model file size | 1.1 GiB (1,117,320,736 bytes) |
| Destination filesystem free | 41 GB at `/dev/sda1` (VL_usb) — well above 7 GB budget |
| Preflight regression | `PASS_SOURCE_OF_TRUTH_RUNTIME_CLEAN`, `error_count=0`, `fallback_count=0` |

---

## Probe Configuration

| Field | Value |
|-------|-------|
| Random seed | 42 (`np.random.default_rng`) |
| Batch size | 1 |
| Hidden | 1536 |
| Intermediate | 8960 |
| Layer read | 0 only (blk.0.ffn_*.weight) |
| Tensors read | `blk.0.ffn_up.weight`, `blk.0.ffn_gate.weight`, `blk.0.ffn_down.weight` |
| Tooling | `huggingface-hub==1.17.0`, `gguf==0.19.0`, `numpy==2.4.6` (env-only, no project source files modified) |

---

## Raw Shape Table

| Tensor | Raw GGUFReader shape | Dequantized shape | Tensor type | n_elements |
|--------|----------------------|-------------------|-------------|-----------:|
| `blk.0.ffn_up.weight` | `[1536, 8960]` | `(8960, 1536)` | Q4_K | 13,762,560 |
| `blk.0.ffn_gate.weight` | `[1536, 8960]` | `(8960, 1536)` | Q4_K | 13,762,560 |
| `blk.0.ffn_down.weight` | `[8960, 1536]` | `(1536, 8960)` | Q6_K | 13,762,560 |

### Important empirical finding (discovered by the probe)

`gguf.dequantize()` returns the tensor in **canonical (d_out, d_in) layout**:
- ffn_up / ffn_gate dequantized shape = `(intermediate, hidden)` = `(8960, 1536)`
- ffn_down dequantized shape = `(hidden, intermediate)` = `(1536, 8960)`

The raw GGUFReader shape display is storage-order-specific (likely `[d_in, d_out]`) and the dequantize step performs the orientation-correcting reshape. This is consistent with `SOURCE_OF_TRUTH.md` `CANONICAL_ORIENTATION = "canonical_d_out_d_in"` (Section 13 / `metric_convention_sanity`).

---

## Candidate Orientation Results

### ffn_up (X shape `[1, 1536]`, target output `[1, 8960]`)

| Candidate | Formulation | Shape result | Expected | Match? | Finite? | Norm |
|-----------|-------------|--------------|----------|--------|---------|------|
| **A (canonical)** | `Y = X @ W.T` (W interpreted as (intermediate, hidden)) | `[1, 8960]` | `[1, 8960]` | ✓ | ✓ | 64.12 |
| B (raw) | `Y = X @ W` (treat W as raw [hidden, intermediate]) | shape-fail (`size 8960 ≠ 1536`) | — | recorded | — | — |

### ffn_gate (X shape `[1, 1536]`, target output `[1, 8960]`)

| Candidate | Formulation | Shape result | Expected | Match? | Finite? | Norm |
|-----------|-------------|--------------|----------|--------|---------|------|
| **A (canonical)** | `Y = X @ W.T` | `[1, 8960]` | `[1, 8960]` | ✓ | ✓ | 64.12 |
| B (raw) | `Y = X @ W` | shape-fail | — | recorded | — | — |

### ffn_down (H shape `[1, 8960]`, target output `[1, 1536]`)

| Candidate | Formulation | Shape result | Expected | Match? | Finite? | Norm |
|-----------|-------------|--------------|----------|--------|---------|------|
| **D (canonical)** | `Y = H @ W.T` (W interpreted as (hidden, intermediate)) | `[1, 1536]` | `[1, 1536]` | ✓ | ✓ | 51.20 |
| E (raw) | `Y = H @ W` (treat W as raw [intermediate, hidden]) | shape-fail (`size 1536 ≠ 8960`) | — | recorded | — | — |

**The shape-failures of the raw formulations (B and E) are EXPECTED and RECORDED, not errors.** They confirm that the dequantized arrays are NOT in the raw GGUFReader storage layout — they are in the canonical (d_out, d_in) layout. A clean matmul under the raw formulations would have indicated the opposite.

---

## Full MLP Formula Parity

Standard layer-0 MLP computation:

```
up   = X @ W_up.T
gate = X @ W_gate.T
act  = silu(gate) * up
out  = act @ W_down.T
```

with `X ∈ R^{1×1536}`, `W_up, W_gate ∈ R^{8960×1536}` (canonical), `W_down ∈ R^{1536×8960}` (canonical).

| Quantity | Value |
|----------|-------|
| `up` shape | `[1, 8960]` ✓ |
| `gate` shape | `[1, 8960]` ✓ |
| `act` shape | `[1, 8960]` ✓ |
| `out` shape | `[1, 1536]` ✓ (expected `[batch, hidden]`) |
| `out` finite | ✓ |
| `out` norm | 21.21 |
| Parity between Formulation 1 and Formulation 2 (both canonical `.T`) | `max_abs_diff = 0.0`, `cosine = 1.0` |

Formulations 1 and 2 are identical by construction (both use the canonical `X @ W.T` pattern), so the parity test confirms numerical reproducibility of the canonical formulation on this 1.5B model. This is an ORIENTATION/EQUIVALENCE check; it is **not** a model quality validation.

---

## Finite / Norm Sanity

- All per-tensor matmul outputs are finite (`np.all(np.isfinite(Y))` is `True` for A and D).
- All MLP intermediate tensors (`up`, `gate`, `act`, `out`) are finite.
- Norms are in plausible ranges (tens of units for activations of random inputs at this scale).
- No NaN, no Inf, no shape mismatches in the canonical formulations.

---

## Actions NOT Taken (Upheld)

- no Q2_K encoding/artifact
- no SDIR residual/artifact
- no anchor probe
- no aggregate validation
- no multi-layer sweep
- no generation/inference
- no model files committed
- no quality/performance claim

31BT is strictly an orientation parity micro-probe.

---

## Prior Accepted Numeric Results — Unchanged

31BT did not run any anchor probe, aggregate validation, Q2_K encoding, or SDIR residual. The 0.5B Q2_K and Q4_K_M reference metrics in accepted prior phases (31AY / 31BA / 31BM / 31BN / 31BO) remain unchanged. The 31BS metadata values remain unchanged.

---

## File Location (env-var form, not committed as canonical)

```
$SDI_MODEL_DIR/qwen2.5-1.5b-official/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

The downloaded model is at the operator-specific path (operator-specific, not committed). The probe runner reads only layer-0 FFN tensors and writes only the probe's own JSON results to `src/results/`. No model files are in the repo, no Q2_K/SDIR artifacts are generated.

---

## Next Allowed Phase After 31BT

**Phase 31BU — Qwen2.5-1.5B Corrected Q2_K Anchor Probe**, only if explicitly requested.
- Must use the downloaded 1.5B Q4_K_M as the W_ref.
- Must use the corrected Q2_K policy (`corrected_ceil_per_row`, ffn_up + ffn_gate, k=0.5%, alpha=1.0, no ffn_down residual).
- No aggregate validation, no multi-layer sweep without explicit approval.
- Orientation is now settled (canonical `X @ W.T` after dequantize) and should not need to be re-derived.

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
- no larger-model validation/result claim
- no runtime-ready output-residual claim
- no Q2_K approximation claim
- no residual improvement claim
- no cosine vs FP/reference claim between quant modes
- 31BT explicitly does NOT validate the 1.5B model; it only verifies orientation equivalence for layer 0
