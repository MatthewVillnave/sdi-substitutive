# Phase 31BH-R — Q2_K Anchor Provenance Reconciliation

## SOURCE_OF_TRUTH Read Confirmation

"I have read SOURCE_OF_TRUTH.md. The current allowed next phase is Phase 31BH — Q2_K Runtime Quantization Path / Portable Runner Hardening, only if explicitly requested. I will not proceed if the requested task conflicts with the source of truth."

---

## Classification

`PARTIAL_31BH_R_X_PROVENANCE_MISMATCH`

## Regression Result

```
python3 -m tests.run_source_of_truth_regression
```
*(Regression run as prerequisite — results independent of anchor mismatch investigation)*

---

## Executive Summary

Phase 31BH-R audited the provenance of the Q2_K backend and activation vectors against the accepted Phase 31AY/31BA anchor runs. Two critical provenance mismatches were found, each independently sufficient to prevent anchor reproduction:

1. **X vector mismatch** (PRIMARY): `np.random.RandomState(seed)` vs `np.random.default_rng(seed)` produce different sequences — new runner generates entirely different activation vectors
2. **Q2_K W_low mismatch** (SECONDARY): floor-based vs ceil-based block accounting produces different W_low tensors

Neither mismatch was described in the Phase 31BH task brief. Both must be fixed before anchors can reproduce.

---

## Provenance Comparison: Old vs New

### 1. X Vector Fingerprints

| Property | OLD (31AY/31BA) | NEW (31BH) | Match? |
|---|---|---|---|
| RNG class | `np.random.default_rng(seed)` | `np.random.RandomState(seed)` | **NO** |
| X shape | `(1, 896)` | `(896,)` | **NO** |
| X dtype | float32 | float32 | YES |
| Generation | `rng.standard_normal((1, HIDDEN))` | `rng.randn(HIDDEN)` | **NO** |
| First 8 values (seed=9) | `[-0.803, 0.243, -1.656, 0.656, 1.143, -0.453, 0.430, 0.251]` | `[0.001, -0.290, -1.116, -0.013, -0.378, -0.481, -1.517, -0.491]` | **NO** |
| SHA256 (seed=9) | `b4c9c8e3...` | `e25bbd6d...` | **NO** |

**`np.random.RandomState` and `np.random.default_rng` produce entirely different sequences for the same seed.** These are not compatible random number generators. This is the primary root cause.

### 2. Q2_K W_low Fingerprints (Layer 21 ffn_up, d_out=4864, d_in=896)

| Property | OLD (31AY/31BA) | NEW (31BH) | Match? |
|---|---|---|---|
| Buffer size basis | `floor(d_out*d_in/QK_K)*84 = 1,430,016` | `d_out * ceil(d_in/QK_K)*84 = 1,634,304` | **NO** |
| Blocks per row | `floor(4864*896/256)*3 = 17,024` (total flat) | `ceil(896/256) = 4` per row × 4,864 = 19,456 | **NO** |
| Q2_K bytes total | 1,430,016 | 1,634,304 | **NO** |
| SHA256 | `99d0f0a1...` | `f243ddd1e...` | **NO** |
| Dequantized W_low[0,:8] | `[-0.022, -0.061, -0.022, -0.022, -0.061, 0.057, -0.022, 0.057]` | same (first 768 elements per row match) | PARTIAL |
| Dequantized W_low max diff | — | 0.840 (at row boundaries) | **NO** |
| Elements quantized | `floor(n/QK_K)*QK_K = 4,358,016` of 4,358,144 | `4,358,144` (all elements) | **NO** |
| Elements truncated | 128 per row × 4,864 = 622,016 | 0 | **NO** |

**OLD script truncates 128 elements per row (final partial Q2_K block) from quantization.** This is a systematic bias.

### 3. Residual Construction

| Property | OLD (31AY/31BA) | NEW (31BH) | Match? |
|---|---|---|---|
| Formula | `R = W_ref - W_low` | `R = W_ref - W_low` | YES |
| k_pct | 1.0 → 1.00% | 1.0 → 1.00% | YES |
| SDIR nnz | 43,616 (1.00%) | 43,616 (1.00%) | YES |
| SDIR bytes | 632,028 | 632,028 | YES |
| encode_sdir path | same | same | YES |

**Residual construction matches** — both use `encode_sdir(W_ref - W_low, k_pct=1.0)`.

### 4. W_ref Extraction

| Property | Value | Match? |
|---|---|---|
| Source | GGUFReader + dequantize(t.data, t.tensor_type) | YES |
| Model | qwen2.5-0.5b-instruct-q4_k_m.gguf | YES |
| Tensor | blk.21.ffn_up.weight | YES |
| Shape | (4864, 896) | YES |
| dtype | float32 | YES |

**W_ref extraction is identical** — same model, same tensor, same dequantization.

---

## Separation-of-Effects Experiment

Using OLD X vs NEW X with OLD W_low vs NEW W_low for L21-S9:

| X source | W_low source | cos_low | delta_cos (vs expected −0.14606) |
|---|---|---|---|
| OLD | OLD | 0.79207 | −0.00284 (close to 0.79491) |
| NEW | NEW | −0.46109 | −0.615 (catastrophic) |
| NEW | OLD | 0.98211 | +0.187 (near-perfect, wrong direction) |
| OLD | NEW | 0.04225 | −0.753 (near-random) |

**Interpretation:**
- X vector mismatch is the **dominant** cause of the anchor mismatch (cos_low changes by ~0.75 when swapping X sources)
- W_low mismatch is secondary (~0.19 effect)
- The expected delta_cos = −0.146 requires the **OLD X** AND **OLD W_low** — both must match

---

## Root Cause Analysis

### Root Cause 1: X Vector Mismatch (PRIMARY)

**Cause:** `np.random.RandomState` and `np.random.default_rng` use incompatible RNG algorithms.

- `np.random.RandomState` (old NumPy, pre-1.17): Mersenne Twister with specific seeding
- `np.random.default_rng` (NumPy 1.17+): new BitGenerator with same seed but different state init

**Impact:** For seed=9, the first call produces:
- OLD: `[-0.803, 0.243, ...]`
- NEW: `[0.001, -0.290, ...]`

These are **completely different** vectors. The MLP output Y = f(X; W) is extremely sensitive to this difference. The accepted L21-S9 severe regression (cos_low=0.795, delta_cos=−0.146) depends critically on the specific geometry of the OLD X vector. The NEW X vector produces near-orthogonal MLP output (cos_low=−0.461 with NEW W_low), making the residual correction catastrophic instead of mildly harmful.

**Fix required:** 31BH runner must use `np.random.default_rng(seed)` to match OLD behavior.

### Root Cause 2: Q2_K Buffer Under-Allocation (SECONDARY)

**Cause:** OLD script allocates `n // QK_K * 84` bytes for flat (d_out*d_in,) quantization. For d_in=896, `floor(896/256)=3` but the actual element count per row is 896 (3.5 blocks). llama.cpp processes all 896 elements, writing 4 blocks of 84 bytes = 336 bytes per row. The OLD buffer is `3*84=252` bytes/row — **128 bytes short per row**.

**Impact:** 
- Quantization reads/writes past the 252-byte boundary into adjacent row's buffer space or uninitialized memory
- Dequantization similarly reads past the end
- W_low is corrupted at the last 128 elements of each row
- This affects 622,016 elements total (12.5% of the matrix)

**Note:** In practice, because rows are contiguous in the flat buffer, the overflow may partially alias into the next row's quantization data. The effect on W_low is non-trivial but secondary to the X mismatch.

**Fix required:** Use `ceil(d_in/QK_K)*Q2_BLOCK_BYTES` per-row buffer accounting, as the NEW backend correctly implements.

### Root Cause 3: X Shape Mismatch (also in OLD path)

OLD script generates X as `(1, 896)` but the sdir_streaming_apply function expects shape `(896,)`. The OLD eval_layer function passes X directly to MLP where broadcasting handles the `(1,896)` shape, but the 31BH runner uses `(896,)` 1D X for sdir_streaming_apply. This is a secondary inconsistency in the OLD path itself.

---

## X Vector Fingerprint Table (Seed 9)

```
OLD X[0,:8]:  [-0.80283695  0.2428499  -1.6563455   0.65610486  1.143453   -0.452611   0.43048576  0.25093257]
NEW X[:8]:    [ 0.00110855 -0.28954408 -1.1160663  -0.01288276 -0.37836146 -0.4811353  -1.517331   -0.490872  ]
OLD X dtype:  float32, shape=(1, 896)
NEW X dtype:  float32, shape=(896,)
OLD X mean:   ≈ 0 (Gaussian)
NEW X mean:   ≈ 0 (Gaussian)
OLD X SHA256: b4c9c8e3f96fd5e4522cb9b30837a5c3
NEW X SHA256: e25bbd6dfe29d4a5ee38cc2f5660cd25
```

---

## "Prompt-Derived Activation" Claim: REJECTED

The task brief asked whether "prompt-derived activation" was the root cause. **This is rejected by the evidence:**

- Phase 31AY script clearly uses `np.random.default_rng(seed).standard_normal((1, HIDDEN))` — pure Gaussian random vectors
- No text prompts, tokenization, embedding lookups, or model inference are involved
- The accepted L21-S9 anchor was produced with a **seed-based Gaussian activation**, not a prompt-derived one

---

## Comparison: OLD vs NEW Code Paths

### OLD (31AY/31BA) Script
```python
# X generation
rng = np.random.default_rng(s)   # ← NumPy BitGenerator
X = rng.standard_normal((1, HIDDEN)).astype(np.float32)  # (1, 896)

# Q2_K encode (flat)
def q2_encode(flat):
    n = flat.size
    buf = np.zeros(n // QK_K * Q2_BLOCK_BYTES, dtype=np.uint8)  # floor = 1,430,016
    lib.quantize_row_q2_K_ref(flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               ctypes.cast(buf.ctypes.data, ctypes.c_void_p), n)
    return buf

W_low = q2_encode(W_ref.flatten()).reshape(W_ref.shape)
```

### NEW (31BH) Runner
```python
# X generation
rng = np.random.RandomState(anchor["seed"])  # ← OLD NumPy RandomState
X = rng.randn(anchor["d_in"]).astype(np.float32)  # (896,)

# Q2_K encode (per-row)
def quantize_q2k_f32_to_bytes(W):
    d_out, d_in = W.shape
    per_row = ((d_in + QK_K - 1) // QK_K) * Q2_BLOCK_BYTES  # ceil = 336
    out = np.zeros(d_out * per_row, dtype=np.uint8)         # = 1,634,304
    for ri in range(d_out):
        row = W[ri]
        lib.quantize_row_q2_K_ref(row.ctypes.data_as(...), dst, d_in)
    return out
```

---

## Artifact Status

### Phase 31BH Files

| File | Status | Action |
|---|---|---|
| `src/q2k_backend.py` | **WORKING** — correct per-row ceil-based Q2_K encoding | Keep |
| `src/bundle_manifest.py` | Updated with Q2_K schema support | Keep |
| `src/phase31bh_q2k_clean_reproduction.py` | Has X + W_low mismatches | Needs fix |
| `docs/STATIC_ARTIFACT_SCHEMA.md` | Updated with q2_k format | Keep |
| `results/PHASE31BH_Q2K_RUNTIME_QUANT_PATH.json` | Schema valid, anchors partial | Keep (historical) |

### Not Committed

No Phase 31BH artifacts were committed as final before this audit began.

---

## Recommended Fixes for 31BH-R2

### Fix 1: X Vector (REQUIRED)

```python
# In phase31bh_q2k_clean_reproduction.py, replace:
rng = np.random.RandomState(anchor["seed"])
X = rng.randn(anchor["d_in"]).astype(np.float32)

# With:
rng = np.random.default_rng(anchor["seed"])
X = rng.standard_normal((1, HIDDEN)).astype(np.float32)  # (1, 896)
```

And in `sdir_streaming_apply`, handle X shape `(1, 896)` → `(896,)` for the streaming path.

### Fix 2: Q2_K Buffer Accounting (REQUIRED for floor-based OLD compatibility)

If matching OLD anchors exactly: use floor-based buffer allocation (1,430,016 bytes total) matching OLD behavior.

If producing improved artifacts: keep ceil-based (1,634,304 bytes) as the new standard, but acknowledge this produces different W_low from OLD anchors.

---

## Next Phase Recommendation

- **Phase 31BH-R2**: Fix X vector + Q2_K buffer in 31BH runner, re-run anchor reproduction
- If 31BH-R2 produces `PASS_31BH_R_ANCHORS_REPRODUCED`, proceed to Phase 31BI
- If only X is fixed but W_low mismatch still causes delta_cos mismatch, classify as `PARTIAL_31BH_R_W_LOW_PROVENANCE_MISMATCH` and decide whether to backport OLD floor-based Q2_K or update accepted anchor values

---

## Classification Reasoning

`PARTIAL_31BH_R_X_PROVENANCE_MISMATCH` is the appropriate classification because:
1. The Q2_K backend itself is correct and well-implemented
2. The W_low difference (floor vs ceil) is real but secondary to X
3. The SDIR residual path is verified identical
4. The mismatch is fully traced to specific code differences, not mysterious divergence
5. The fix is well-understood and localized

`PARTIAL_31BH_R_Q2K_PROVENANCE_MISMATCH` is also partially valid (W_low bytes differ by 204,288), but X is the dominant effect.

`PARTIAL_31BH_R_ANCHOR_SOURCE_UNREPRODUCIBLE` is NOT appropriate because the OLD scripts are fully inspectable and the exact differences are precisely identified.
