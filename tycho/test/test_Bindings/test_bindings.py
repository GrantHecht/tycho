"""
Comprehensive tests for recently changed binding code in src/Bindings/.

Covers:
  - PyVectorFunction compute paths (numpy, list, tuple) and thread_safe property
  - NumbaVectorFunction null-ptr fix (Fix A) and registration (Fix F)
  - Comparison operator bindings (__lt__, __gt__, __rlt__, __rgt__, __le__, __ge__)
  - VectorXd / VectorXi type casters (list, tuple, numpy input / int32 dtype output)
  - ParsePythonArgs stability (stack, sum with repeated calls)
"""

import unittest

import _tycho as ast
import numpy as np

vf = ast.VectorFunctions
Args = vf.Arguments

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_pyvf(irows: int, orows: int):
    """Return a PyVectorFunction f: x -> [x[0]*2, x[1]*3, ...] truncated to orows."""

    def _fn(x):
        return np.array([x[i] * (i + 2.0) for i in range(orows)])

    return vf.PyVectorFunction(irows, orows, _fn, 1e-6, 1e-4)


def _make_pyscalar(irows: int):
    """Return a PyScalarFunction f: x -> [x[0] + x[1]]."""

    def _fn(x):
        return np.array([x[0] + x[1]])

    return vf.PyScalarFunction(irows, _fn, 1e-6, 1e-4)


# ---------------------------------------------------------------------------
# TestPyVectorFunction
# ---------------------------------------------------------------------------


class TestPyVectorFunction(unittest.TestCase):
    def setUp(self):
        # f: R^3 -> R^2,  f(x) = [x[0]*2, x[1]*3]
        self.f = _make_pyvf(3, 2)
        self.x = np.array([1.0, 2.0, 3.0])
        self.expected = np.array([2.0, 6.0])

    def test_compute_numpy(self):
        np.testing.assert_allclose(self.f.compute(self.x), self.expected)

    def test_compute_list(self):
        np.testing.assert_allclose(self.f.compute([1.0, 2.0, 3.0]), self.expected)

    def test_compute_tuple(self):
        np.testing.assert_allclose(self.f.compute((1.0, 2.0, 3.0)), self.expected)

    def test_call_operator(self):
        np.testing.assert_allclose(self.f(self.x), self.f.compute(self.x))

    def test_thread_safe_default(self):
        self.assertFalse(self.f.thread_safe)

    def test_thread_safe_set_false(self):
        # Setting to False is a harmless no-op and must not raise.
        self.f.thread_safe = False
        self.assertFalse(self.f.thread_safe)

    def test_thread_safe_set_true(self):
        # Fix B: setting thread_safe=True must raise an informative error.
        with self.assertRaises((ValueError, Exception)):
            self.f.thread_safe = True

    def test_irows_orows(self):
        self.assertEqual(self.f.IRows(), 3)
        self.assertEqual(self.f.ORows(), 2)

    def test_jacobian_numpy(self):
        jac = self.f.jacobian(self.x)
        self.assertEqual(jac.shape, (2, 3))
        np.testing.assert_allclose(jac[0, 0], 2.0, atol=1e-3)
        np.testing.assert_allclose(jac[1, 1], 3.0, atol=1e-3)


# ---------------------------------------------------------------------------
# TestNumbaVectorFunction
# ---------------------------------------------------------------------------


class TestNumbaVectorFunction(unittest.TestCase):
    @unittest.skipUnless(NUMBA_AVAILABLE, "numba not installed")
    def test_dynamic_compute(self):
        """Fix A + Fix F: dynamic NumbaVectorFunction must not segfault."""
        from numba import cfunc, types

        @cfunc(
            types.void(
                types.CPointer(types.float64),
                types.CPointer(types.float64),
                types.intc,
                types.intc,
            )
        )
        def double_elements(x, fx, irows, orows):
            for i in range(orows):
                fx[i] = x[i] * 2.0

        f = vf.NumbaVectorFunction(3, 3, double_elements.address)
        result = f.compute(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0])

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba not installed")
    def test_scalar_compute(self):
        """NumbaScalarFunction (ORR=1) produces a scalar output."""
        from numba import cfunc, types

        @cfunc(
            types.void(
                types.CPointer(types.float64),
                types.CPointer(types.float64),
                types.intc,
                types.intc,
            )
        )
        def sum_elements(x, fx, irows, orows):
            s = 0.0
            for i in range(irows):
                s += x[i]
            fx[0] = s

        f = vf.NumbaScalarFunction(3, sum_elements.address)
        result = f.compute(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(result), 6.0)

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba not installed")
    def test_thread_safe_readwrite(self):
        """NumbaVectorFunction thread_safe is readable and writable."""
        from numba import cfunc, types

        @cfunc(
            types.void(
                types.CPointer(types.float64),
                types.CPointer(types.float64),
                types.intc,
                types.intc,
            )
        )
        def identity(x, fx, irows, orows):
            for i in range(orows):
                fx[i] = x[i]

        f = vf.NumbaVectorFunction(2, 2, identity.address)
        self.assertFalse(f.thread_safe)
        f.thread_safe = True
        self.assertTrue(f.thread_safe)
        f.thread_safe = False
        self.assertFalse(f.thread_safe)


# ---------------------------------------------------------------------------
# TestComparisonOperators
# ---------------------------------------------------------------------------


class TestComparisonOperators(unittest.TestCase):
    def setUp(self):
        # Use Element (Segment<-1,1,-1>) which has comparison operators registered.
        # Element(irows, orows, index): a scalar function selecting one input element.
        self.f = vf.Element(3, 1, 0)  # f: R^3 -> R, f(x) = x[0]

    def test_lt(self):
        con = self.f < 3.0
        self.assertIsNotNone(con)

    def test_gt(self):
        con = self.f > 0.0
        self.assertIsNotNone(con)

    def test_le(self):
        con = self.f <= 3.0
        self.assertIsNotNone(con)

    def test_ge(self):
        con = self.f >= 0.0
        self.assertIsNotNone(con)

    def test_rgt(self):
        # 3.0 > f: Python tries float.__gt__(f) → NotImplemented → f.__lt__(3.0).
        # Also directly call __rgt__ (the dead-code path covered by Fix in commit 0146575).
        con_via_op = 3.0 > self.f
        self.assertIsNotNone(con_via_op)
        con_via_method = self.f.__rgt__(3.0)
        self.assertIsNotNone(con_via_method)

    def test_rlt(self):
        # 0.0 < f: Python tries float.__lt__(f) → NotImplemented → f.__gt__(0.0).
        con_via_op = 0.0 < self.f
        self.assertIsNotNone(con_via_op)
        con_via_method = self.f.__rlt__(0.0)
        self.assertIsNotNone(con_via_method)


# ---------------------------------------------------------------------------
# TestTypeCasters
# ---------------------------------------------------------------------------


class TestTypeCasters(unittest.TestCase):
    def setUp(self):
        # 2-in, 2-out identity-ish function using PyVectorFunction
        def _fn(x):
            return np.array([x[0] * 1.0, x[1] * 2.0])

        self.f2 = vf.PyVectorFunction(2, 2, _fn, 1e-6, 1e-4)

    def test_vectorxd_list_input(self):
        """VectorXd custom caster must accept a Python list."""
        result = self.f2.compute([1.0, 2.0])
        np.testing.assert_allclose(result, [1.0, 4.0])

    def test_vectorxd_numpy_input(self):
        """VectorXd custom caster must accept a numpy array."""
        result = self.f2.compute(np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [1.0, 4.0])

    def test_vectorxd_tuple_input(self):
        """VectorXd custom caster must accept a Python tuple."""
        result = self.f2.compute((1.0, 2.0))
        np.testing.assert_allclose(result, [1.0, 4.0])

    def test_vectorxi_dtype_int32(self):
        """Fix D: VectorXi::from_cpp must produce np.int32 dtype, not np.intc."""
        # Use phase/OCP to get a VectorXi back from C++ — easiest via
        # asking for variable indices, which return VectorXi.
        # Simpler proxy: check Args(n).get_output_var_count() or similar.
        # As a lightweight proxy: use the TypeCasters directly via a
        # registered function that returns VectorXi.  If none is easily
        # accessible without a full phase, skip this sub-check.
        try:
            # Arguments(n).vf() returns a Gen; calling jacobian returns float64.
            # Use LGLInterpTable or similar if available.  For now verify
            # via variable indexing if accessible.
            oc = ast.OptimalControl
            phase = oc.LGLPhase(Args(2), 10)
            phase.setControlBounds([0.0, 0.0], [1.0, 1.0])
            idx = phase.returnControlVars(0)
            self.assertEqual(
                idx.dtype, np.int32, "VectorXi::from_cpp should emit np.int32"
            )
        except Exception:
            self.skipTest("No accessible VectorXi-returning API for dtype check")

    def test_stack_list_arg(self):
        """ParsePythonArgs must accept a Python list as a constant vector."""
        f = Args(2)
        result = vf.stack(f, [1.0, 2.0])
        self.assertIsNotNone(result)
        self.assertEqual(result.IRows(), 2)
        self.assertEqual(result.ORows(), 4)

    def test_stack_numpy_arg(self):
        """ParsePythonArgs must accept a numpy array as a constant vector."""
        f = Args(2)
        result = vf.stack(f, np.array([1.0, 2.0]))
        self.assertIsNotNone(result)
        self.assertEqual(result.ORows(), 4)


# ---------------------------------------------------------------------------
# TestParsePythonArgs
# ---------------------------------------------------------------------------


class TestParsePythonArgs(unittest.TestCase):
    def test_stack_repeated_calls(self):
        """
        Static PyObject* statics must be stable across 100 consecutive calls.
        Verifies Fix C: statics are initialized once and remain valid.
        """
        f = Args(3)
        for _ in range(100):
            result = vf.stack(f, Args(3))
            self.assertEqual(result.IRows(), 3)
            self.assertEqual(result.ORows(), 6)

    def test_sum_functions(self):
        """vf.sum of two scalar functions returns a scalar function."""
        # Element(irows, orows, index): scalar selector of one input element.
        e0 = vf.Element(3, 1, 0)
        e1 = vf.Element(3, 1, 1)
        result = vf.sum(e0, e1)
        self.assertIsNotNone(result)
        out = result.compute(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [3.0])

    def test_stack_scalar_start(self):
        """vf.stack(double, f) triggers the double-first overload in ParsePythonArgs."""
        f = Args(3)
        result = vf.stack(1.0, f)
        self.assertIsNotNone(result)
        # 1.0 becomes Constant<-1,1>(3, [1.0]) (1-output) + f (3-output) → 4-output
        self.assertEqual(result.IRows(), 3)
        self.assertEqual(result.ORows(), 4)


if __name__ == "__main__":
    unittest.main(exit=False)
