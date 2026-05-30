#!/usr/bin/env python3
"""
Deep debug: sdir_streaming_apply vs dense residual apply
Root cause investigation for the 0.65 cosine between stream and dense.
"""
import os, sys, struct, numpy as np
sys.path.insert(0, '/home/matthew-villnave/sdi-substitutive/src')

from phase31x_manifest_runtime import encode_sdir, sdir_streaming_apply, RSC_MAGIC, BLOCK_SIZE

ROWS, COLS = 896, 4864
layer = 0

# Build W_ref in 31Y style
np.random.seed(layer * 43 + 7)
U = np.random.randn(ROWS, min(ROWS, COLS)//2).astype(np.float32)
V = np.random.randn(min(ROWS, COLS)//2, COLS).astype(np.float32)
W_ref = U @ V * 0.1

# Quantize W_ref in 31Y style
W_low = np.zeros((ROWS, COLS), dtype=np.float32)
n = ROWS * COLS
nb = n // BLOCK_SIZE
for b in range(nb):
    block = W_ref.flat[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
    scale = float(np.abs(block).max()) / 7.5
    if scale < 1e-8: scale = 1.0
    q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
    W_low.flat[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE] = q * scale

R = W_ref - W_low
X = np.ones(ROWS, dtype=np.float32)

# Dense residual apply
Y_delta_dense = X @ R

# Encode and decode
sdir_bytes = encode_sdir(R, k_pct=7.5)
Y_delta_stream, vp = sdir_streaming_apply(sdir_bytes, X, ROWS, COLS)

print(f"=== sdir_streaming_apply Debug (Layer {layer}) ===")
print(f"Matrix: {ROWS}x{COLS} = {ROWS*COLS:,} elements")
print(f"R norm: {float(np.linalg.norm(R)):.4f}, mean abs: {float(np.mean(np.abs(R))):.6f}")
print(f"nnz (>1e-9): {int(np.sum(np.abs(R) > 1e-9)):,}")
print(f"k_pct=7.5%: expected nnz = {int(ROWS*COLS*7.5/100):,}")
print()
print(f"sdir_bytes total: {len(sdir_bytes):,} bytes")
print(f"Header: 28 bytes, magic={sdir_bytes[:4]}")
print(f"RSC_MAGIC={RSC_MAGIC}, match={sdir_bytes[:4]==RSC_MAGIC}")

# Parse header
hdr = struct.unpack("<4sHHIIIIHH", sdir_bytes[:28])
print(f"Header parsed: magic={hdr[0]}, version={hdr[1]}, flags={hdr[2]}")
print(f"  rows={hdr[3]}, cols={hdr[4]}, k_pct_int={hdr[5]}, nnz={hdr[6]}")
print(f"  value_dtype={hdr[7]}, mask_encoding={hdr[8]}")
k_pct_header = hdr[5] / 100.0
print(f"  k_pct from header: {k_pct_header:.2f}%")

# Compute bitmap size
bitmap_bytes_computed = (ROWS * COLS + 7) // 8
print(f"\nbitmap_bytes computed: {bitmap_bytes_computed:,}")
bitmap_data = sdir_bytes[28:28+bitmap_bytes_computed]
print(f"bitmap_data actual: {len(bitmap_data):,} bytes")

# Values
values_data = sdir_bytes[28+bitmap_bytes_computed:]
values_count = len(values_data) // 2
print(f"values_count: {values_count:,}")
print(f"nnz from header: {hdr[6]:,}")

# Check: do they match?
print(f"\nValues count matches nnz: {values_count == hdr[6]}")

# Unpack bitmap
flat_bitmap = np.unpackbits(np.frombuffer(bitmap_data, dtype=np.uint8))
nnz_actual = int(np.sum(flat_bitmap))
print(f"Actual nnz from bitmap: {nnz_actual:,}")

# Y_delta comparison
print(f"\nY_delta dense:  norm={float(np.linalg.norm(Y_delta_dense)):.4f}, mean_abs={float(np.mean(np.abs(Y_delta_dense))):.6f}")
print(f"Y_delta stream: norm={float(np.linalg.norm(Y_delta_stream)):.4f}, mean_abs={float(np.mean(np.abs(Y_delta_stream))):.6f}")
print(f"max_abs_diff: {float(np.max(np.abs(Y_delta_dense - Y_delta_stream))):.4f}")

# Top elements comparison
top_n = 20
top_dense_idx = np.argsort(np.abs(Y_delta_dense))[-top_n:][::-1]
top_stream_idx = np.argsort(np.abs(Y_delta_stream))[-top_n:][::-1]
print(f"\nTop {top_n} abs(Y_delta_dense):")
for idx in top_dense_idx[:10]:
    stream_val = Y_delta_stream[idx]
    print(f"  [{idx}]: dense={Y_delta_dense[idx]:.4f}, stream={stream_val:.4f}, diff={Y_delta_dense[idx]-stream_val:.4f}")

# Check: is the issue with how values are being looked up?
# Direct decode using EncodedResidual
from residual_encode import EncodedResidual

# Manually construct an EncodedResidual from sdir_bytes
off = 28
bitmap_bytes = (ROWS * COLS + 7) // 8
bitmap = np.frombuffer(sdir_bytes[off:off+bitmap_bytes], dtype=np.uint8).copy()
values_raw = np.frombuffer(sdir_bytes[off+bitmap_bytes:], dtype=np.uint16).copy()
nnz_hdr = hdr[6]
enc = EncodedResidual(ROWS, COLS, k_pct_header, bitmap, values_raw, nnz_hdr)

# Use EncodedResidual.streaming_sparse_apply
# But this expects X as (batch, rows) where R is (rows, cols) and X_on_R_cols=True
# For X as (batch, rows) with X_on_R_cols: Y_delta[b, col_j] += X[b, row_i] * R[row_i, col_j]
X_batch = X[np.newaxis, :]  # (1, 896)
Y_delta_encoded = enc.streaming_sparse_apply(X_batch)  # (1, 4864)
print(f"\nEncodedResidual.streaming_sparse_apply Y_delta: norm={float(np.linalg.norm(Y_delta_encoded)):.4f}")

# What about the residual_encode.py streaming method?
# EncodedResidual is (rows, cols) = (896, 4864)
# X_batch is (1, 896) with X_on_R_cols: Y_delta[b, j] += X[b, i] * R[i, j]
# This should give Y_delta of shape (1, 4864)
print(f"Y_delta_encoded vs Y_delta_dense: cos={float(np.dot(Y_delta_encoded.ravel(), Y_delta_dense) / (np.linalg.norm(Y_delta_encoded)*np.linalg.norm(Y_delta_dense))):.6f}")

# Now check: what is the correct bitmap layout?
# The bitmap was created by encode_residual's row-major iteration
# For each (i,j) in row-major order: if |R[i,j]| >= threshold: bitmap[i*cols+j] = 1, values.append(R[i,j])
# So values[t] corresponds to R[i,j] where t = i*cols + j (and bitmap[t] == 1)

# Decode to dense for verification
R_decoded = enc.decode_to_dense()
print(f"\nR_decoded vs R: max_diff={float(np.max(np.abs(R - R_decoded))):.2e}")
print(f"R_decoded norm: {float(np.linalg.norm(R_decoded)):.4f}, R norm: {float(np.linalg.norm(R)):.4f}")

# Now the key question: when sdir_streaming_apply iterates,
# does it access the same (i,j) positions that were set in the bitmap?

# In sdir_streaming_apply:
# for row in range(out_dim):  # 0..4863
#   for col in range(in_dim):  # 0..895
#     if bitmap[row * in_dim + col]:  # accessing bitmap[row*896+col]
#       Y[row] += X[col] * values[vp]; vp += 1

# But the bitmap was created for R of shape (in_dim=896, out_dim=4864)
# bitmap[i*out_dim + j] = 1 if |R[i,j]| >= threshold
# Wait, no. The bitmap is (rows*cols,) flattened.
# In encode: for idx, v in enumerate(R.flat): bitmap[idx] = ...
# R.flat goes row by row: R[0,0], R[0,1], ..., R[0,4863], R[1,0], ...
# So bitmap[t] corresponds to R.flat[t] = R[t//cols, t%cols] = R[i,j] where i=t//cols, j=t%cols
# So bitmap[i*cols + j] = 1 if |R[i,j]| >= threshold ✓

# In sdir_streaming_apply:
# bitmap[row * in_dim + col] where row ranges [0, out_dim) and col ranges [0, in_dim)
# With out_dim=4864, in_dim=896: bitmap[0*896 + 0] = bitmap[0] = R[0,0] ✓
# bitmap[0*896 + 1] = bitmap[1] = R[0,1] ✓
# ...
# bitmap[4863*896 + 895] = bitmap[4358129] = R[895, 4863] ✓

# So the bitmap indexing IS correct for the matrix dimensions.

# The issue might be with the values array ordering.
# When we read values[t] in decode (at position t in the nnz list),
# this should correspond to the t-th element that was appended.
# The t-th appended element is at index t in values[].
# And values[t] corresponds to R[i,j] where (i,j) is the t-th row-major element.

# So bitmap[global_idx]==1 means the next value to read is the one at
# position in values[] corresponding to that element.

# Actually, let me trace through a specific example.
# Suppose values were appended in row-major order.
# values[0] = R[0,0], values[1] = R[0,1], ...
# In decode, at row=0, col=0: bitmap[0]==1 → read values[0]=R[0,0] ✓
# At row=0, col=1: bitmap[1]==1 → read values[1]=R[0,1] ✓
# ...
# At row=1, col=0: bitmap[896]==1 → read values[?] = ?

# If bitmap[896] corresponds to position 896 in row-major order:
# position 896 = 1*4864 + 0 → R[1, 0]
# If the 2nd element in values[] is R[1,0], that's correct.

# So the ordering should be correct.

# Let me just verify: manually compute Y_delta from the EncodedResidual
# and compare to dense.
Y_delta_manual = X_batch @ enc.decode_to_dense()  # (1, 4864)
print(f"\nY_delta_dense: norm={float(np.linalg.norm(Y_delta_dense)):.4f}")
print(f"Y_delta_manual (via decode): norm={float(np.linalg.norm(Y_delta_manual)):.4f}")

# Now check sdir_streaming_apply issue with in_dim/out_dim swap
# What if the bitmap is stored with dimensions swapped?
# R has shape (896, 4864) = (in_dim, out_dim)
# bitmap is (in_dim * out_dim,) = (896 * 4864,)
# If bitmap is accessed as (out_dim, in_dim) in decode, that's correct.

# But what if in decode, out_dim is actually interpreted as in_dim and vice versa?
# sdir_streaming_apply(data, X, in_dim, out_dim) with in_dim=896, out_dim=4864
# bitmap_bytes = (in_dim * out_dim + 7) // 8 → (896*4864+7)//8 = 544768 ✓
# bitmap = np.unpackbits(data[28:28+544768]) → 4358144 bits ✓
# flat_bitmap[0] = bit 0 of byte 0 = R[0,0] ✓
# flat_bitmap[896] = bit 0 of byte 112 = R[1,0] ✓

# The decode loop:
# for row in range(out_dim):  # 0..4863
#   for col in range(in_dim):  # 0..895
#     if bitmap[row * in_dim + col]:
#       Y[row] += X[col] * values[vp]; vp += 1

# At row=0, col=0: bitmap[0]=1 → read values[0]=R[0,0]; Y[0] += X[0]*R[0,0]
# At row=0, col=1: bitmap[1]=1 → read values[1]=R[0,1]; Y[0] += X[1]*R[0,1]
# ...
# Y[0] = sum over col where bitmap[0*896+col]=1 of X[col] * values[idx]
# This is correct for Y = X @ R, row 0.

# I can't find the bug from code inspection alone.
# Let me just verify with a tiny example.
print("\n=== TINY EXAMPLE ===")
R_tiny = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # (2,3)
X_tiny = np.array([1.0, 1.0], dtype=np.float32)  # (2,)
Y_dense_tiny = X_tiny @ R_tiny  # (3,) = [1+4, 2+5, 3+6] = [5, 7, 9]
print(f"Tiny R: {R_tiny.shape}, X: {X_tiny.shape}, Y_dense: {Y_dense_tiny}")

sdir_tiny = encode_sdir(R_tiny, k_pct=50.0)  # 50% to get top elements
Y_stream_tiny, vp_tiny = sdir_streaming_apply(sdir_tiny, X_tiny, 2, 3)
print(f"Tiny Y_stream: {Y_stream_tiny}, vp={vp_tiny}")
print(f"Y_dense_tiny: {Y_dense_tiny}")
print(f"Match: {np.allclose(Y_dense_tiny, Y_stream_tiny, atol=1e-4)}")
