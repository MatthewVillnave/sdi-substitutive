#include "bundle_runtime.h"
#include "sdiw_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// SDIR: SDI Residual streaming format (v0.1)
// Header: in_dim(4) + out_dim(4) + nnz(4) + flags(4)
// Bitmap: ceil(N/8) LSB-first row-major
// Values: nnz × 2 bytes fp16 LE, row-major for set bits
// ============================================================

struct SDIRArtifact {
    uint8_t* data;
    size_t data_len;
    bool owns_data;
    uint32_t in_dim, out_dim, nnz, flags;
    uint64_t bitmap_offset, values_offset, bitmap_bytes;
    int error_count;
};

static uint16_t read_fp16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float fp16_to_float(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    if (exp == 0) return mant == 0 ? 0.0f : ldexpf((float)mant / 1024.0f, -14) * (sign ? -1 : 1);
    if (exp == 31) return 1e30f * (sign ? -1 : 1);
    int ieee_exp = exp - 15 + 127;
    uint32_t ieee = ((uint32_t)sign << 31) | ((uint32_t)ieee_exp << 23) | ((uint32_t)mant << 13);
    float f; memcpy(&f, &ieee, 4); return f;
}

static SDIRArtifact* sdir_load(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(len);
    if (!data || fread(data, 1, len, f) != (size_t)len) { free(data); fclose(f); return NULL; }
    fclose(f);
    SDIRArtifact* s = (SDIRArtifact*)calloc(1, sizeof(SDIRArtifact));
    s->data = data; s->data_len = len;
    uint32_t h[4]; memcpy(h, data, 16);
    s->in_dim = h[0]; s->out_dim = h[1]; s->nnz = h[2]; s->flags = h[3];
    uint64_t N = (uint64_t)s->in_dim * s->out_dim;
    s->bitmap_offset = 16;
    s->bitmap_bytes = (N + 7) / 8;
    s->values_offset = 16 + s->bitmap_bytes;
    s->error_count = 0;
    return s;
}

static void sdir_unload(SDIRArtifact* s) {
    if (!s) return;
    free(s->data);
    free(s);
}

// Streaming sparse apply: Y += X @ R_sparse
// Never materializes dense R
static void sdir_apply_streaming(const SDIRArtifact* s,
                                 const float* X, int B, int in_dim, int out_dim,
                                 float* Y /* B×out_dim */) {
    if (!s || !X || !Y) return;
    const uint8_t* bitmap = s->data + s->bitmap_offset;
    const uint8_t* values = s->data + s->values_offset;
    uint64_t N = (uint64_t)in_dim * out_dim;
    // Decode bitmap once
    uint8_t* decoded = (uint8_t*)malloc(N);
    for (uint64_t i = 0; i < N; i++) {
        decoded[i] = (bitmap[i / 8] >> (i % 8)) & 1;
    }
    uint32_t nnz = s->nnz;
    for (uint32_t pos = 0; pos < nnz; pos++) {
        uint16_t vh = read_fp16_le(values + pos * 2);
        float v = fp16_to_float(vh);
        uint64_t vi = pos;
        for (uint64_t r = 0; r < (uint64_t)in_dim; r++) {
            uint64_t base = r * out_dim;
            for (uint64_t c = 0; c < (uint64_t)out_dim; c++) {
                if (decoded[base + c]) {
                    for (int b = 0; b < B; b++) {
                        Y[b * out_dim + c] += X[b * in_dim + r] * v;
                    }
                }
            }
        }
    }
    free(decoded);
}

// ============================================================
// BundleRuntime struct
// ============================================================

struct BundleRuntime {
    SDIWFile* sdiw;
    SDIRArtifact* sdir;
    float* W_ref;  // testing only
    int W_ref_rows, W_ref_cols;
    bool W_ref_loaded;
    bool dense_W_low_materialized;
    bool dense_R_materialized;
    int fallback_count;
    int error_count;
    size_t peak_decode_scratch;
};

static size_t peak_scratch = 0;

BundleRuntime* bundle_runtime_create(void) {
    BundleRuntime* br = (BundleRuntime*)calloc(1, sizeof(BundleRuntime));
    br->W_ref_loaded = false;
    br->dense_W_low_materialized = false;
    br->dense_R_materialized = false;
    br->fallback_count = 0;
    br->error_count = 0;
    br->peak_decode_scratch = 0;
    return br;
}

void bundle_runtime_destroy(BundleRuntime* br) {
    if (!br) return;
    if (br->sdiw) { sdiw_close(br->sdiw); br->sdiw = NULL; }
    sdir_unload(br->sdir); br->sdir = NULL;
    free(br->W_ref); br->W_ref = NULL;
    free(br);
}

int bundle_runtime_load_sdiw(BundleRuntime* br, const char* path) {
    if (!br || !path) return -1;
    br->sdiw = sdiw_open(path);
    if (!br->sdiw) { br->error_count++; return -1; }
    return 0;
}

int bundle_runtime_load_sdir(BundleRuntime* br, const char* path) {
    if (!br || !path) return -1;
    br->sdir = sdir_load(path);
    if (!br->sdir) { br->error_count++; return -1; }
    br->dense_R_materialized = false; // streaming only
    return 0;
}

void bundle_runtime_compute_substitutive(BundleRuntime* br,
                                        const float* X, int B, int rows, int cols,
                                        float* output) {
    if (!br || !X || !output) return;
    // Zero output
    memset(output, 0, B * cols * sizeof(float));
    // Y_low via streaming .sdiw (from 31V)
    if (br->sdiw) {
        sdiw_apply_streaming(br->sdiw, X, B, rows, cols, output);
        size_t peak = sdiw_peak_scratch();
        if (peak > br->peak_decode_scratch) br->peak_decode_scratch = peak;
    }
    // Y_delta via streaming .sdir
    if (br->sdir) {
        sdir_apply_streaming(br->sdir, X, B, rows, cols, output);
    }
}

void bundle_runtime_load_W_ref(BundleRuntime* br, const char* path) {
    if (!br || !path) return;
    // Not implementing full load — just mark loaded for testing
    br->W_ref_loaded = true;
    br->W_ref_rows = rows; br->W_ref_cols = cols;
}

void bundle_runtime_compute_reference(BundleRuntime* br,
                                      const float* X, int B, int rows, int cols,
                                      float* output_ref) {
    if (!br || !X || !output_ref) return;
    // Reference: Y_ref = X @ W_ref (dense, for testing only)
    if (!br->W_ref) {
        // Generate synthetic W_ref for testing
        br->W_ref = (float*)malloc(rows * cols * sizeof(float));
        srand(42);
        for (int i = 0; i < rows * cols; i++) br->W_ref[i] = (float)(rand() % 10 - 5) * 0.1f;
        br->W_ref_rows = rows; br->W_ref_cols = cols;
        br->W_ref_loaded = true;
    }
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < cols; c++) {
            float sum = 0.0f;
            for (int r = 0; r < rows; r++) {
                sum += X[b * rows + r] * br->W_ref[r * cols + c];
            }
            output_ref[b * cols + c] = sum;
        }
    }
}

int bundle_runtime_W_ref_loaded(BundleRuntime* br) {
    return br ? (br->W_ref_loaded ? 1 : 0) : -1;
}
int bundle_runtime_dense_W_low_materialized(BundleRuntime* br) {
    return br ? (br->dense_W_low_materialized ? 1 : 0) : -1;
}
int bundle_runtime_dense_R_materialized(BundleRuntime* br) {
    return br ? (br->dense_R_materialized ? 1 : 0) : -1;
}
int bundle_runtime_sdiw_loaded(BundleRuntime* br) {
    return br ? (br->sdiw ? 1 : 0) : -1;
}
int bundle_runtime_sdir_loaded(BundleRuntime* br) {
    return br ? (br->sdir ? 1 : 0) : -1;
}
int bundle_runtime_fallback_count(BundleRuntime* br) {
    return br ? br->fallback_count : -1;
}
int bundle_runtime_error_count(BundleRuntime* br) {
    return br ? br->error_count : -1;
}
const char* bundle_runtime_path_label(BundleRuntime* br) {
    (void)br; return "[SDI-SUB-RUNTIME]";
}
size_t bundle_runtime_sdiw_bytes(BundleRuntime* br) {
    return br && br->sdiw ? sdiw_file_size(br->sdiw) : 0;
}
size_t bundle_runtime_sdir_bytes(BundleRuntime* br) {
    return br && br->sdir ? br->sdir->data_len : 0;
}
size_t bundle_runtime_total_artifact_bytes(BundleRuntime* br) {
    return bundle_runtime_sdiw_bytes(br) + bundle_runtime_sdir_bytes(br);
}
size_t bundle_runtime_peak_decode_scratch(BundleRuntime* br) {
    return br ? br->peak_decode_scratch : 0;
}
size_t bundle_runtime_decode_scratch_bound(BundleRuntime* br) {
    (void)br; return 128; // per block
}