#!/usr/bin/env python3
"""
test_toy_runtime.py — Phase 31N tests for toy substitutive runtime.

Run directly: python3 tests/test_toy_runtime.py
Run via pytest: python3 -m pytest tests/test_toy_runtime.py -v
"""

import sys, os, gc, tempfile, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from residual_encode import encode_residual, EncodedResidual
from residual_compute import streaming_sparse_apply


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def q4_quantize_dequantize(W: np.ndarray) -> np.ndarray:
    flat = W.flatten()
    n = len(flat)
    block_size = 32
    out = np.zeros(n, dtype=np.float32)
    for b in range((n + block_size - 1) // block_size):
        s, e = b * block_size, min(b * block_size + block_size, n)
        block = flat[s:e]
        scale = max(float(np.abs(block).max()) / 7.0, 1e-8)
        q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        out[s:e] = q * scale
    return out[:n].reshape(W.shape)


class TestToySubstitutiveRuntime:
    """Test suite for Phase 31N: Toy Substitutive Runtime Harness."""

    @classmethod
    def setup_class(cls):
        """Generate shared synthetic weights and inputs."""
        np.random.seed(42)
        cls.rows, cls.cols = 896, 4864
        cls.W_ref = np.random.randn(cls.rows, cls.cols).astype(np.float32) * 0.1
        cls.W_low = q4_quantize_dequantize(cls.W_ref)
        cls.R = cls.W_ref - cls.W_low
        cls.enc = encode_residual(cls.R, k_pct=7.5)
        np.random.seed(99)
        cls.X = np.random.randn(1, cls.rows).astype(np.float32)

    def test_weight_shapes(self):
        assert self.W_ref.shape == (self.rows, self.cols)
        assert self.W_low.shape == (self.rows, self.cols)

    def test_residual_encoding(self):
        enc = self.enc
        assert enc.rows == self.rows
        assert enc.cols == self.cols
        assert enc.k_pct == 7.5
        expected_nnz = int(self.rows * self.cols * 7.5 / 100)
        assert abs(enc.nnz - expected_nnz) < int(expected_nnz * 0.1), \
            f"nnz={enc.nnz} far from expected ~{expected_nnz}"
        assert enc.total_bytes > 0

    def test_residual_binary_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            self.enc.save(path)
            enc2 = EncodedResidual.load(path)
            assert enc2.rows == self.enc.rows
            assert enc2.cols == self.enc.cols
            assert enc2.nnz == self.enc.nnz
            assert enc2.k_pct == self.enc.k_pct
        finally:
            os.unlink(path)

    def test_reference_mode_loads_W_ref(self):
        from toy_runtime import ModeReference
        m = ModeReference(self.W_ref)
        Y = m.compute(self.X)
        assert m.W_ref_loaded == True
        assert Y.shape == (1, self.cols)

    def test_low_only_W_ref_not_loaded(self):
        """Low-only mode must NOT load W_ref."""
        from toy_runtime import ModeLowOnly
        m = ModeLowOnly(self.W_low)
        assert m.W_ref_loaded == False
        Y = m.compute(self.X)
        assert Y.shape == (1, self.cols)
        np.testing.assert_allclose(Y, self.X @ self.W_low, rtol=1e-5)

    def test_substitutive_W_ref_not_loaded(self):
        """Substitutive mode must NOT load W_ref."""
        from toy_runtime import ModeSubstitutive
        m = ModeSubstitutive(self.W_low, self.enc, self.enc.total_bytes)
        assert m.W_ref_loaded == False

    def test_substitutive_dense_residual_not_materialized(self):
        """Substitutive mode must materialize 0 bytes of dense residual."""
        from toy_runtime import ModeSubstitutive
        m = ModeSubstitutive(self.W_low, self.enc, self.enc.total_bytes)
        assert m.residual_dense_bytes == 0

    def test_substitutive_path_label(self):
        from toy_runtime import ModeSubstitutive
        m = ModeSubstitutive(self.W_low, self.enc, self.enc.total_bytes)
        assert m.path_label == "[SDI-SUB-RUNTIME]"

    def test_streaming_sparse_apply_matches_dense(self):
        """streaming_sparse_apply must match decode-to-dense reference."""
        R_dense = self.enc.decode_to_dense()
        # X_on_R_cols=True: Y_dense = X @ R_dense (1,896) @ (896,4864) → (1,4864)
        Y_dense = self.X @ R_dense
        Y_sparse = streaming_sparse_apply(self.X, self.enc, X_on_R_cols=True)
        max_diff = float(np.abs(Y_dense - Y_sparse).max())
        cos = cosine(Y_dense, Y_sparse)
        assert max_diff < 1e-3, f"max_diff={max_diff}"
        assert cos > 0.9999, f"cos={cos}"

    def test_substitutive_approximation_quality(self):
        """Substitutive must be closer to reference than low-only (delta_cosine > 0)."""
        from toy_runtime import ModeReference, ModeLowOnly, ModeSubstitutive
        m_ref = ModeReference(self.W_ref)
        m_low = ModeLowOnly(self.W_low)
        m_sub = ModeSubstitutive(self.W_low, self.enc, self.enc.total_bytes)

        Y_ref = m_ref.compute(self.X)
        Y_low = m_low.compute(self.X)
        Y_sub = m_sub.compute(self.X)

        cos_low = cosine(Y_ref, Y_low)
        cos_sub = cosine(Y_ref, Y_sub)
        delta_cos = cos_sub - cos_low

        assert delta_cos > 0, \
            f"delta_cosine={delta_cos:.6f} not positive (sub not better than low)"
        assert cos_sub > cos_low, \
            f"cos_sub={cos_sub:.6f} should exceed cos_low={cos_low:.6f}"

    def test_substitutive_reduces_mae(self):
        """Substitutive must have lower MAE vs reference than low-only."""
        from toy_runtime import ModeReference, ModeLowOnly, ModeSubstitutive
        m_ref = ModeReference(self.W_ref)
        m_low = ModeLowOnly(self.W_low)
        m_sub = ModeSubstitutive(self.W_low, self.enc, self.enc.total_bytes)

        Y_ref = m_ref.compute(self.X)
        Y_low = m_low.compute(self.X)
        Y_sub = m_sub.compute(self.X)

        mae_low = float(np.abs(Y_ref - Y_low).mean())
        mae_sub = float(np.abs(Y_ref - Y_sub).mean())
        assert mae_sub < mae_low, \
            f"MAE_sub={mae_sub:.6f} should be < MAE_low={mae_low:.6f}"

    def test_fail_fast_missing_residual(self):
        """Loading missing residual must raise FileNotFoundError."""
        try:
            EncodedResidual.load("/tmp/THIS_PATH_DOES_NOT_EXIST_31N.bin")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_results_json_smoke(self):
        """results/ JSON must exist and have PASS classification."""
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            "PHASE31N_TOY_SUBSTITUTIVE_RUNTIME.json"
        )
        if not os.path.exists(json_path):
            import subprocess
            subprocess.run(
                ["python3", os.path.join(os.path.dirname(__file__), "..", "src", "toy_runtime.py")],
                check=True, capture_output=True
            )
        with open(json_path) as f:
            data = json.load(f)
        assert data["classification"] == "PASS_TOY_SUBSTITUTIVE_RUNTIME"
        assert data["checks"]["W_ref_absent_in_substitutive"] == True
        assert data["checks"]["residual_dense_bytes_zero"] == True
        assert data["checks"]["delta_cosine_positive"] == True
        assert data["checks"]["fail_fast_on_missing_residual"] == True
        assert data["metrics"]["delta_cosine"] > 0

    def test_docs_md_smoke(self):
        """docs/ MD must exist and contain key strings."""
        md_path = os.path.join(
            os.path.dirname(__file__), "..", "docs",
            "PHASE31N_TOY_SUBSTITUTIVE_RUNTIME.md"
        )
        if not os.path.exists(md_path):
            import subprocess
            subprocess.run(
                ["python3", os.path.join(os.path.dirname(__file__), "..", "src", "toy_runtime.py")],
                check=True, capture_output=True
            )
        with open(md_path) as f:
            content = f.read()
        assert "PASS_TOY_SUBSTITUTIVE_RUNTIME" in content
        assert "SDI-SUB-RUNTIME" in content
        assert "delta_cosine" in content


if __name__ == "__main__":
    suite = TestToySubstitutiveRuntime()
    suite.setup_class()

    tests = [t for t in dir(suite) if t.startswith("test_")]
    passed = failed = 0

    for name in tests:
        try:
            getattr(suite, name)()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} passed")
    if failed:
        raise SystemExit(1)
