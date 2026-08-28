"""Python-binding coverage for the Izzo Lambert solvers (``lambert_izzo``).

First-ever Python-level test coverage for this binding. Pins three
contracts at the binding boundary:

1. The scalar ``lambert_izzo`` overload NaN-poisons ``V1``/``V2`` and the
   binding translates that into ``RuntimeError`` — including the
   "NaN-with-success" collinear-geometry case, where the underlying
   Newton iteration reports convergence (``exint == 0``) but the
   transfer-plane normal is undefined (see the warning on
   ``lambert_izzo_impl`` in ``include/tycho/detail/astro/kepler/
   lambert_solvers.h``). The scalar overload already guards this via an
   unconditional ``allFinite()`` check, independent of ``exint``.
2. The vectorized batch overload validates ``axis`` and the shapes of
   all six array arguments (``R1s``, ``R2s``, ``dts``, ``longways``,
   ``V1s``, ``V2s``) before indexing, raising ``ValueError`` (nanobind's
   translation of ``std::invalid_argument``) on mismatch.
3. The batch overload's per-problem exit codes now flag collinear-
   geometry NaN results too (previously such a problem silently reported
   exit code 0 with NaN velocities), so callers can "trust the exit
   codes" instead of separately checking ``V1s``/``V2s`` for
   finiteness.

These tests do not re-derive Izzo's algorithm — that is covered by the
underlying C++ implementation. The gap they close is binding-layer
input validation and error/exit-code translation, plus a basic
batch-matches-scalar cross-check on the vectorized code path.
"""

import math
import unittest

import numpy as np

import tychopy as typy


class TestLambertScalar(unittest.TestCase):
    MU = 1.0

    def test_known_transfer_roundtrip(self):
        # Quarter-circle transfer on a unit-radius circular orbit (mu=1):
        # departing at R1=[1,0,0] and arriving at R2=[0,1,0] after a
        # quarter period (dt = pi/2) is a *known* circular-orbit solution
        # (speed = sqrt(mu/r) = 1, transverse only). Verify by propagating
        # the returned V1 forward with `propagate_cartesian` and checking
        # it reaches R2 — this exercises the binding end-to-end rather
        # than pinning brittle literal-constant expectations.
        R1 = np.array([1.0, 0.0, 0.0])
        R2 = np.array([0.0, 1.0, 0.0])
        dt = math.pi / 2.0

        v1, v2 = typy.astro.lambert_izzo(R1, R2, dt, self.MU, False)
        self.assertTrue(np.all(np.isfinite(v1)))
        self.assertTrue(np.all(np.isfinite(v2)))

        rv0 = np.concatenate([R1, v1])
        rvf = typy.astro.propagate_cartesian(rv0, dt, self.MU)
        np.testing.assert_allclose(rvf[:3], R2, atol=1e-9)

    def test_collinear_raises_runtime_error(self):
        # cross(R1, R2) == 0 -> transfer-plane normal undefined -> NaN
        # V1/V2 despite exint == 0 ("NaN-with-success"); the scalar
        # binding's allFinite() guard must still catch this.
        R1 = np.array([1.0, 0.0, 0.0])
        R2 = np.array([2.0, 0.0, 0.0])
        with self.assertRaises(RuntimeError):
            typy.astro.lambert_izzo(R1, R2, 1.0, self.MU, True)


class TestLambertBatchValidation(unittest.TestCase):
    MU = 1.0

    def _mk(self, n):
        R1s = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, n))
        R2s = np.tile(np.array([[0.0], [1.0], [0.0]]), (1, n))
        dts = np.full(n, 1.0)
        lws = [True] * n
        V1s = np.zeros((3, n))
        V2s = np.zeros((3, n))
        return R1s, R2s, dts, lws, V1s, V2s

    def test_mismatched_dts_raises(self):
        R1s, R2s, dts, lws, V1s, V2s = self._mk(4)
        with self.assertRaises(ValueError):
            typy.astro.lambert_izzo(R1s, R2s, dts[:2], self.MU, lws, V1s, V2s, 0, False)

    def test_mismatched_longways_raises(self):
        R1s, R2s, dts, lws, V1s, V2s = self._mk(4)
        with self.assertRaises(ValueError):
            typy.astro.lambert_izzo(R1s, R2s, dts, self.MU, lws[:2], V1s, V2s, 0, False)

    def test_mismatched_output_shape_raises(self):
        R1s, R2s, dts, lws, V1s, V2s = self._mk(4)
        bad_v1s = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            typy.astro.lambert_izzo(R1s, R2s, dts, self.MU, lws, bad_v1s, V2s, 0, False)

    def test_bad_axis_raises(self):
        R1s, R2s, dts, lws, V1s, V2s = self._mk(4)
        with self.assertRaises(ValueError):
            typy.astro.lambert_izzo(R1s, R2s, dts, self.MU, lws, V1s, V2s, 2, False)

    def test_collinear_pair_gets_nonzero_exitcode(self):
        R1s, R2s, dts, lws, V1s, V2s = self._mk(4)
        R2s[:, 1] = [2.0, 0.0, 0.0]  # make problem 1 collinear with R1
        codes = typy.astro.lambert_izzo(R1s, R2s, dts, self.MU, lws, V1s, V2s, 0, False)
        self.assertEqual(codes[0], 0)
        self.assertNotEqual(codes[1], 0)
        self.assertEqual(codes[2], 0)
        self.assertEqual(codes[3], 0)
        # The exit code is trustworthy on its own now: no finite problem's
        # output should be NaN, and the collinear one's output should be.
        self.assertTrue(np.all(np.isfinite(V1s[:, [0, 2, 3]])))
        self.assertTrue(np.all(np.isfinite(V2s[:, [0, 2, 3]])))
        self.assertTrue(np.all(np.isnan(V1s[:, 1])))

    def test_batch_matches_scalar(self):
        R1s, R2s, dts, lws, V1s, V2s = self._mk(3)
        codes = typy.astro.lambert_izzo(R1s, R2s, dts, self.MU, lws, V1s, V2s, 0, False)
        np.testing.assert_array_equal(codes, np.zeros(3, dtype=codes.dtype))
        for i in range(3):
            v1, v2 = typy.astro.lambert_izzo(
                R1s[:, i], R2s[:, i], dts[i], self.MU, lws[i]
            )
            np.testing.assert_allclose(V1s[:, i], v1, rtol=1e-12)
            np.testing.assert_allclose(V2s[:, i], v2, rtol=1e-12)

    def test_batch_matches_scalar_axis1(self):
        # Same problems, transposed (N, 3) layout via axis=1.
        R1s, R2s, dts, lws, V1s, V2s = self._mk(3)
        R1s_t = np.ascontiguousarray(R1s.T)
        R2s_t = np.ascontiguousarray(R2s.T)
        V1s_t = np.zeros((3, 3))
        V2s_t = np.zeros((3, 3))
        typy.astro.lambert_izzo(R1s_t, R2s_t, dts, self.MU, lws, V1s_t, V2s_t, 1, False)
        for i in range(3):
            v1, v2 = typy.astro.lambert_izzo(
                R1s_t[i, :], R2s_t[i, :], dts[i], self.MU, lws[i]
            )
            np.testing.assert_allclose(V1s_t[i, :], v1, rtol=1e-12)
            np.testing.assert_allclose(V2s_t[i, :], v2, rtol=1e-12)


class TestLambertBatchVectorized(unittest.TestCase):
    """Coverage for the ``vectorize=True`` SuperScalar (vsize=8) packed loop.

    ``_mk`` above builds ``N`` *identical* problems, which cannot expose a
    lane-mapping bug (every lane produces the same answer regardless of
    mapping). The tests here use ``N=10`` *distinct* well-posed problems —
    one 8-wide SIMD pack plus a 2-wide scalar remainder — so a mis-mapped
    SIMD lane would show up as a mismatch against the corresponding scalar
    call.
    """

    MU = 1.0

    def _mk_varied(self, n):
        # N distinct, non-collinear short-way transfers on the unit circle:
        # R1 fixed, R2 swept through n transfer angles strictly between 0
        # and 180 degrees (so none are collinear / well-posed for all n).
        thetas = np.deg2rad(np.linspace(10.0, 150.0, n))
        R1s = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, n))
        R2s = np.vstack([np.cos(thetas), np.sin(thetas), np.zeros(n)])
        dts = np.full(n, 1.0)
        lws = [False] * n
        V1s = np.zeros((3, n))
        V2s = np.zeros((3, n))
        return R1s, R2s, dts, lws, V1s, V2s

    def test_batch_vectorized_matches_scalar(self):
        # N=10 covers one full 8-wide SuperScalar pack plus a 2-wide scalar
        # remainder; vectorize=True routes the first 8 problems through the
        # packed SIMD loop. Confirmed passing on the current .so.
        n = 10
        R1s, R2s, dts, lws, V1s, V2s = self._mk_varied(n)
        codes = typy.astro.lambert_izzo(R1s, R2s, dts, self.MU, lws, V1s, V2s, 0, True)
        np.testing.assert_array_equal(codes, np.zeros(n, dtype=codes.dtype))
        for i in range(n):
            v1, v2 = typy.astro.lambert_izzo(
                R1s[:, i], R2s[:, i], dts[i], self.MU, lws[i]
            )
            np.testing.assert_allclose(V1s[:, i], v1, rtol=1e-12)
            np.testing.assert_allclose(V2s[:, i], v2, rtol=1e-12)

    def test_batch_vectorized_axis1_matches_scalar(self):
        # Same problems and packing as above, transposed (N, 3) layout.
        # Confirmed passing on the current .so.
        n = 10
        R1s, R2s, dts, lws, _, _ = self._mk_varied(n)
        R1s_t = np.ascontiguousarray(R1s.T)
        R2s_t = np.ascontiguousarray(R2s.T)
        V1s_t = np.zeros((n, 3))
        V2s_t = np.zeros((n, 3))
        codes = typy.astro.lambert_izzo(
            R1s_t, R2s_t, dts, self.MU, lws, V1s_t, V2s_t, 1, True
        )
        np.testing.assert_array_equal(codes, np.zeros(n, dtype=codes.dtype))
        for i in range(n):
            v1, v2 = typy.astro.lambert_izzo(
                R1s_t[i, :], R2s_t[i, :], dts[i], self.MU, lws[i]
            )
            np.testing.assert_allclose(V1s_t[i, :], v1, rtol=1e-12)
            np.testing.assert_allclose(V2s_t[i, :], v2, rtol=1e-12)

    def test_vectorized_collinear_exitcodes(self):
        # NEW BEHAVIOR: passes only after rebuilding against the
        # updated lambert_solvers_bind.cpp. On the pre-Task-10 .so, the
        # vectorized packed loop does *not* promote the "NaN-with-success"
        # collinear exit code (verified: codes stayed all-zero even though
        # V1s/V2s were NaN at both collinear indices below), so this test
        # is expected to fail until the branch's binding changes are built.
        #
        # One collinear problem inside the packed SIMD lane (index 3, the
        # 4th of the first 8-wide pack) and one in the scalar remainder
        # (index 9, the last of the 2-wide tail).
        n = 10
        R1s, R2s, dts, lws, V1s, V2s = self._mk_varied(n)
        R2s[:, 3] = [2.0, 0.0, 0.0]  # collinear with R1s[:, 3] == [1,0,0]
        R2s[:, 9] = [3.0, 0.0, 0.0]  # collinear with R1s[:, 9] == [1,0,0]

        codes = typy.astro.lambert_izzo(R1s, R2s, dts, self.MU, lws, V1s, V2s, 0, True)

        collinear = {3, 9}
        for i in range(n):
            if i in collinear:
                self.assertNotEqual(
                    codes[i], 0, msg=f"index {i} should be flagged collinear"
                )
            else:
                self.assertEqual(
                    codes[i], 0, msg=f"index {i} should have converged cleanly"
                )

        ok_idx = [i for i in range(n) if i not in collinear]
        self.assertTrue(np.all(np.isfinite(V1s[:, ok_idx])))
        self.assertTrue(np.all(np.isfinite(V2s[:, ok_idx])))


if __name__ == "__main__":
    unittest.main()
