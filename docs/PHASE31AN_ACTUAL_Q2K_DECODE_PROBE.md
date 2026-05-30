# Phase 31AN — Actual Q2_K Decode Probe

## Classification
**`PARTIAL_ACTUAL_DECODED_RESIDUAL_IMPROVES_MEMORY_FAILS`**

Subtype: Actual IQ4_NL/Q3_K decoded W_low — residuals improve approximation but W_low exceeds Q4 budget

---

## Q2_K Model Availability

| Item | Value |
|------|-------|
| Q2_K GGUF path | `/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q2_k.gguf` |
| File size | 395.95 MB |
| Safe to use locally | ✓ (USB drive path, excluded from commits via .gitignore) |
| Actual GGUF Q2_K (type=10) tensors | **0 — none present** |
| Actual decoded types | IQ4_NL (4.5 bpe) for ffn_up/ffn_gate; Q3_K (3.44 bpe) for ffn_down |
| Actual Q2_K encode available | ✗ — `Q2_K.quantize_blocks` is `NotImplemented` in gguf-py |

**Critical finding:** The file named `qwen2.5-0.5b-instruct-q2_k.gguf` does NOT contain actual Q2_K (type=10) encoded tensors. It contains IQ4_NL (4.5 bpe) and Q3_K (3.44 bpe). This is a quantization configuration used during export, not a literal Q2_K type encoding. The naming is misleading.

---

## Q2_K GGUF Tensor Type Distribution

| Type | Count | Description |
|------|------:|-------------|
| IQ4_NL | 121 | 4-bit non-linear, ~4.5 bpe |
| F32 | 121 | Full precision (biases, norms, etc.) |
| Q3_K | 24 | Q3_K type, ~3.44 bpe |
| Q5_0 | 24 | Q5_0 type, ~5.5 bpe |
| Q8_0 | 1 | Q8_0 for output.weight only |

**FFN tensor encoding in Q2_K GGUF:**

| Family | Layer | Type | bpe | Bytes | Margin vs Q4 |
|--------|-------|------|----:|------:|------------:|
| ffn_up | 0 | IQ4_NL | 4.5000 | 2,451,456 | −266 KB |
| ffn_gate | 0 | IQ4_NL | 4.5000 | 2,451,456 | −266 KB |
| ffn_down | 0 | Q3_K | 3.4375 | 1,872,640 | +299 KB |

All 6 layers have identical type assignment per family.

---

## Decode Route

| Item | Value |
|------|-------|
| Decode route | B. GGUFReader + gguf-py dequantize |
| Source of Q2_K block layout | llama.cpp/gguf-py/gguf/quants.py |
| Q2_K dequantize available | ✗ (no type=10 tensors in Q2_K GGUF) |
| IQ4_NL dequantize available | ✓ |
| Q3_K dequantize available | ✓ |
| Validation method | No NaN, no Inf in all dequantized tensors |

---

## Tensor Extraction — Layer 0

| Tensor | Type | Shape | Bytes | bpe | cos(W_ref, W_low) | MAE | no_nan | no_inf |
|--------|------|-------|------:|----:|------------------:|----:|:------:|:------:|
| blk.0.ffn_up.weight | IQ4_NL | (896, 4864) | 2,451,456 | 4.5000 | 0.996254 | 0.001050 | ✓ | ✓ |
| blk.0.ffn_gate.weight | IQ4_NL | (896, 4864) | 2,451,456 | 4.5000 | 0.995915 | 0.000927 | ✓ | ✓ |
| blk.0.ffn_down.weight | Q3_K | (4864, 896) | 1,872,640 | 3.4375 | 0.987369 | 0.001589 | ✓ | ✓ |

Reference: Q4_K_M GGUF (ffn_up/ffn_gate: Q5_0, ffn_down: Q6_K)

---

## Actual vs Simulated Q2-like (31AM) Comparison

| Family | Actual Type | Actual bpe | Actual cos | Sim cos (31AM) | Actual MAE | Sim MAE (31AM) |
|--------|-------------|-----------:|-----------:|---------------:|-----------:|---------------:|
| ffn_up | IQ4_NL | 4.5000 | 0.996254 | 0.926044 | 0.001050 | 0.009039 |
| ffn_gate | IQ4_NL | 4.5000 | 0.995915 | 0.926044 | 0.000927 | 0.009039 |
| ffn_down | Q3_K | 3.4375 | 0.987369 | 0.926044 | 0.001589 | 0.009039 |

**Key finding:** Actual IQ4_NL/Q3_K decode produces cos ≈ 0.996 vs 31AM simulated cos ≈ 0.926. The 31AM block_size=16 simulation was far more aggressive than actual low-bit encoding quality. **31AM's numerical conclusions are NOT transferable to actual decode paths.**

---

## Residual Sweep — Layer 0 Individual

| Family | k% | cos_low | cos_sub | Δcos | mae_low | mae_sub | Δmae |
|--------|---:|--------:|--------:|-----:|--------:|--------:|-----:|
| ffn_up | 0% | 0.996254 | 0.996254 | +0.0000 | 0.001050 | 0.001050 | +0.0000 |
| ffn_up | 0.5% | 0.996254 | 0.996457 | **+0.0002** | 0.001050 | 0.001029 | **−0.0000** |
| ffn_up | 1% | 0.996254 | 0.996582 | **+0.0003** | 0.001050 | 0.001012 | **−0.0000** |
| ffn_up | 2% | 0.996254 | 0.996787 | **+0.0005** | 0.001050 | 0.000982 | **−0.0000** |
| ffn_up | 3% | 0.996254 | 0.996961 | **+0.0007** | 0.001050 | 0.000954 | **−0.0000** |
| ffn_gate | 0% | 0.995915 | 0.995915 | +0.0000 | 0.000927 | 0.000927 | +0.0000 |
| ffn_gate | 0.5% | 0.995915 | 0.996365 | **+0.0005** | 0.000927 | 0.000914 | **−0.0000** |
| ffn_gate | 1% | 0.995915 | 0.996555 | **+0.0006** | 0.000927 | 0.000903 | **−0.0000** |
| ffn_gate | 2% | 0.995915 | 0.996829 | **+0.0009** | 0.000927 | 0.000888 | **−0.0000** |
| ffn_gate | 3% | 0.995915 | 0.997041 | **+0.0011** | 0.000927 | 0.000876 | **−0.0000** |
| ffn_down | 0% | 0.987369 | 0.987369 | +0.0000 | 0.001589 | 0.001589 | +0.0000 |
| ffn_down | 0.5% | 0.987369 | 0.988651 | **+0.0013** | 0.001589 | 0.001571 | **−0.0000** |
| ffn_down | 1% | 0.987369 | 0.989163 | **+0.0018** | 0.001589 | 0.001557 | **−0.0000** |
| ffn_down | 2% | 0.987369 | 0.989886 | **+0.0025** | 0.001589 | 0.001539 | **−0.0000** |
| ffn_down | 3% | 0.987369 | 0.990451 | **+0.0031** | 0.001589 | 0.001525 | **−0.0000** |

**Every residual-on policy IMPROVES cosine for ALL families at ALL k%.** This is the opposite of 31AM's simulated finding.

---

## Residual Sweep — Layers 0–5 Average (18 family-layers)

| k% | avg_cos_low | avg_cos_sub | Δcos | Residual improves? |
|---:|------------:|------------:|-----:|:-----------------:|
| 0% | 0.993128 | 0.993128 | +0.0000 | baseline |
| 0.5% | 0.993128 | 0.993622 | **+0.0005** | ✓ |
| 1% | 0.993128 | 0.993872 | **+0.0007** | ✓ |
| 2% | 0.993128 | 0.994263 | **+0.0011** | ✓ |
| 3% | 0.993128 | 0.994585 | **+0.0015** | ✓ |

---

## Memory Budget — All 18 Family-Layers

| Family | Type | Per tensor margin | Count | Total margin |
|--------|------|------------------:|------:|-------------:|
| ffn_up (×6) | IQ4_NL | −266 KB | 6 | −1,596 KB |
| ffn_gate (×6) | IQ4_NL | −266 KB | 6 | −1,596 KB |
| ffn_down (×6) | Q3_K | +299 KB | 6 | +1,796 KB |
| **Total** | | | | **−1,396 KB** |

**Total across 18 tensors: −1,396.5 KB (exceeds Q4 budget)**

IQ4_NL (ffn_up/ffn_gate) exceeds Q4_budget by 272,384 bytes per tensor. Q3_K (ffn_down) is under Q4_budget by 306,432 bytes per tensor. Net: −1,430,016 bytes.

**True Q2_K (2.625 bpe) would fit:** 1,430,028 bytes per ffn_up/ffn_gate vs Q4_budget 2,179,072 bytes → +749 KB margin each. But actual Q2_K encode is NotImplemented.

---

## Classification Rationale

`PARTIAL_ACTUAL_DECODED_RESIDUAL_IMPROVES_MEMORY_FAILS` — three-part classification:

1. **ACTUAL DECODED** — We obtained real GGUF-dequantized low-bit tensors (IQ4_NL/Q3_K), not simulated
2. **RESIDUAL IMPROVES** — Residual-on improves cosine at every k% for every family, opposite of 31AM
3. **MEMORY FAILS** — IQ4_NL exceeds Q4_budget; only Q3_K (ffn_down) is memory-positive; net −1.4 MB

---

## Forbidden Claims (Not Made)

- ❌ No model quality recovery claimed
- ❌ No behavior recovery claimed  
- ❌ No speedup claimed
- ❌ No full-model memory savings claimed
- ❌ No llama.cpp integration claimed
- ❌ No production readiness claimed
- ❌ No actual Q2_K (type=10) behavior verified
- ❌ No claim that IQ4_NL/Q3_K results generalize to true Q2_K

---

## Key Findings vs 31AM

| Property | 31AM (Simulated) | 31AN (Actual) |
|----------|-----------------:|-------------:|
| W_low type | block_size=16 aggressive | IQ4_NL (4.5 bpe) / Q3_K (3.44 bpe) |
| cos(W_ref, W_low) | ~0.926 | ~0.996 |
| MAE | ~0.009 | ~0.001 |
| Residual-on effect | HURTS cosine | IMPROVES cosine |
| Memory vs Q4 | positive | NEGATIVE (−1.4 MB net) |
| Pareto frontier | empty | not tested (memory-fails first) |

**31AM's simulated Q2-like behavior is not representative of actual low-bit decode quality.** The block_size=16 simulation was far more aggressive than actual IQ4_NL encoding.

---

## Suspected / Unproven

- Actual Q2_K (type=10) decode behavior remains unknown
- IQ4_NL/Q3_K residual improvement may be specific to this quantization scheme, not generalizable to Q2_K
- True Q2_K encoding with actual block_size=256 decode could behave differently from IQ4_NL
- Memory failure is due to IQ4_NL's 4.5 bpe — true Q2_K (2.625 bpe) would be memory-positive

---

## Recommended Next Phase

**Phase 31AO — True Q2_K Encoder Implementation, only if explicitly requested.**

Two findings from 31AN make true Q2_K encoding a high-value next step:

1. **Memory:** True Q2_K (2.625 bpe) would be memory-positive (+731 KB margin per ffn_up/ffn_gate)
2. **Residual effectiveness:** Actual decoded residuals IMPROVE approximation under IQ4_NL — the same or stronger effect likely holds for Q2_K

Required: Implement Q2_K encode (since `gguf-py` dequantize is available but quantize_blocks is NotImplemented), or use a separate encoding approach.

Do not proceed without explicit user request.

---

## Artifacts

- **Results JSON:** `results/PHASE31AN_ACTUAL_Q2K_DECODE_PROBE.json`
- **Probe script:** `src/phase31an_actual_q2k_decode_probe.py`
- **Data source:** USB drive Q2_K/Q4_K_M GGUF files

---

*Phase 31AN — probe run 2026-05-30. Not committed pending corrections.*