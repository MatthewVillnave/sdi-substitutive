#ifndef BUNDLE_RUNTIME_H
#define BUNDLE_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// BundleRuntime
//
// Combined .sdiw (streaming W_low) + .sdir (streaming residual)
// Substitutive path: no W_ref, no dense W_low, no dense R
// ============================================================

typedef struct BundleRuntime BundleRuntime;

// Lifecycle
BundleRuntime* bundle_runtime_create(void);
void bundle_runtime_destroy(BundleRuntime* br);

// Load artifacts (substitutive path — no W_ref)
int bundle_runtime_load_sdiw(BundleRuntime* br, const char* path);
int bundle_runtime_load_sdir(BundleRuntime* br, const char* path);

// Compute paths
// X: B×in_dim (row-major), output: B×out_dim (row-major)
void bundle_runtime_compute_substitutive(BundleRuntime* br,
                                         const float* X, int B,
                                         int rows, int cols,
                                         float* output);

// Reference mode (for testing only — W_ref loaded)
void bundle_runtime_load_W_ref(BundleRuntime* br, const char* path);
void bundle_runtime_compute_reference(BundleRuntime* br,
                                      const float* X, int B,
                                      int rows, int cols,
                                      float* output_ref);

// Counters
int bundle_runtime_W_ref_loaded(BundleRuntime* br);
int bundle_runtime_dense_W_low_materialized(BundleRuntime* br);
int bundle_runtime_dense_R_materialized(BundleRuntime* br);
int bundle_runtime_sdiw_loaded(BundleRuntime* br);
int bundle_runtime_sdir_loaded(BundleRuntime* br);
int bundle_runtime_fallback_count(BundleRuntime* br);
int bundle_runtime_error_count(BundleRuntime* br);
const char* bundle_runtime_path_label(BundleRuntime* br);

// Memory info
size_t bundle_runtime_sdiw_bytes(BundleRuntime* br);
size_t bundle_runtime_sdir_bytes(BundleRuntime* br);
size_t bundle_runtime_total_artifact_bytes(BundleRuntime* br);
size_t bundle_runtime_peak_decode_scratch(BundleRuntime* br);
size_t bundle_runtime_decode_scratch_bound(BundleRuntime* br);

#ifdef __cplusplus
}
#endif

#endif // BUNDLE_RUNTIME_H