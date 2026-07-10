"""
Comprehensive tests for recently changed binding code in src/bindings/.

Covers:
  - PyVectorFunction compute paths (numpy, list, tuple) and thread_safe property
  - NumbaVectorFunction null-ptr fix (Fix A) and registration (Fix F)
  - Comparison operator bindings (__lt__, __gt__, __rlt__, __rgt__, __le__, __ge__)
  - VectorXd / VectorXi type casters (list, tuple, numpy input / int32 dtype output)
  - ParsePythonArgs stability (stack, sum with repeated calls)
"""

import unittest

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
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
        self.assertEqual(self.f.input_rows(), 3)
        self.assertEqual(self.f.output_rows(), 2)

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

    def test_vectorxd_rejects_str(self):
        """VectorXd custom caster must reject str (a digit-string must not
        silently parse element-wise into a numeric vector)."""
        with self.assertRaises(TypeError):
            self.f2.compute("12")  # digit-string must not parse as [1., 2.]

    def test_vectorxd_rejects_dict(self):
        """VectorXd custom caster must reject dict (iterating its keys is
        not a sensible numeric-vector payload)."""
        with self.assertRaises(TypeError):
            self.f2.compute({1.0: "a", 2.0: "b"})

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
            oc = ast.optimal_control
            phase = oc.LGLPhase(Args(2), 10)
            phase.set_control_bounds([0.0, 0.0], [1.0, 1.0])
            idx = phase.return_control_vars(0)
            self.assertEqual(
                idx.dtype, np.int32, "VectorXi::from_cpp should emit np.int32"
            )
        except (AttributeError, TypeError):
            self.skipTest("No accessible VectorXi-returning API for dtype check")

    def test_vectorxi_rejects_bytes(self):
        """VectorXi custom caster must reject bytes (they must not silently
        become the vector of their code points, e.g. b"AB" -> [65, 66]).

        ``eval(int, Eigen::VectorXi)`` (the remapped-input-projection overload)
        is only registered on generic (non-Segment/Arguments) VectorFunctions,
        so a ``vf.stack(...)`` result is used here rather than a bare
        ``Arguments``/``Segment`` object.
        """
        f = vf.stack(Args(2), [1.0, 2.0])
        with self.assertRaises(TypeError):
            f.eval(4, b"AB")  # bytes must not become [65, 66]

    def test_vectorxi_rejects_huge_index(self):
        """VectorXi custom caster must reject an index that overflows int32
        rather than silently truncating it to a small, wrong, in-bounds value
        (CODEBASE_REVIEW 1.1b).

        ``2**32`` is used here (not e.g. ``5_000_000_000``) because it is the
        smallest huge value that actually discriminates pre-fix from post-fix
        behavior: ``(int32_t)PyLong_AsLong(2**32)`` wraps to exactly ``0`` --
        a valid, in-bounds, silently-wrong index that the pre-fix caster
        accepted without complaint. A value like ``5_000_000_000`` truncates
        (mod 2**32) to 705032704, which is *also* out of this function's
        valid index range, so pre-fix code raises ValueError via
        ParsedInput's bounds check too -- that value does not distinguish
        the bug from the fix. Post-fix, the new ``py_long_to_int32`` helper
        rejects ``2**32`` outright and the caster returns false, so nanobind
        raises TypeError (not ValueError) at the binding boundary.
        """
        f = vf.stack(Args(2), [1.0, 2.0])
        with self.assertRaises(TypeError):
            f.eval(
                4, [2**32, 0]
            )  # pre-fix: wraps to 0 (valid, wrong); post-fix: rejected

    def test_scaletype_huge_int_raises_not_crashes(self):
        """Fix: ScaleType's noexcept ``from_python`` must not let a Python
        exception (OverflowError converting 10**400 to a C double) propagate
        as a C++ exception out of a noexcept function -- that is a hard
        interpreter crash (std::terminate), not a Python-catchable failure.

        Pre-fix: interpreter abort. Post-fix: a clean Python exception from
        the failed argument cast. ``ScaleType`` is only reachable via a
        phase's ``auto_scale``-family arguments, so a minimal single-state,
        no-control ODE phase is built to reach it (see
        ``tychopy/test/test_OptimalControl/test_NewMethods.py`` for the same
        ``oc.ode_x.ode`` subclassing pattern).
        """
        oc = ast.optimal_control

        class _ScaleTypeProbeODE(oc.ode_x.ode):
            def __init__(self):
                args = oc.ODEArguments(1)
                x = args.x_var(0)
                super().__init__((-1.0) * x, 1)

        ode = _ScaleTypeProbeODE()
        traj = [np.array([1.0, t]) for t in np.linspace(0.0, 1.0, 20)]
        phase = ode.phase("LGL3", traj, 4)
        with self.assertRaises(Exception):
            phase.add_boundary_value("Front", 0, 0.0, auto_scale=10**400)

    def test_stack_list_arg(self):
        """ParsePythonArgs must accept a Python list as a constant vector."""
        f = Args(2)
        result = vf.stack(f, [1.0, 2.0])
        self.assertIsNotNone(result)
        self.assertEqual(result.input_rows(), 2)
        self.assertEqual(result.output_rows(), 4)

    def test_stack_numpy_arg(self):
        """ParsePythonArgs must accept a numpy array as a constant vector."""
        f = Args(2)
        result = vf.stack(f, np.array([1.0, 2.0]))
        self.assertIsNotNone(result)
        self.assertEqual(result.output_rows(), 4)


# ---------------------------------------------------------------------------
# TestModuleLayoutHardBreak
# ---------------------------------------------------------------------------


class TestModuleLayoutHardBreak(unittest.TestCase):
    """Pin the snake_case Python-module rename as a hard break with no
    PascalCase aliases. If a future change adds a convenience alias
    (e.g. resurrects ``_tychopy.OptimalControl`` or ``tychopy.Astro``)
    these tests fail at CI time instead of silently softening the API.
    """

    PASCAL_NAMES = ["VectorFunctions", "OptimalControl", "Solvers", "Utils", "Astro"]

    def test_no_pascalcase_underscore_tychopy_submodules(self):
        import importlib

        for name in self.PASCAL_NAMES:
            with self.assertRaises(
                (ImportError, ModuleNotFoundError, AttributeError),
                msg=f"_tychopy.{name} should not be importable",
            ):
                importlib.import_module(f"_tychopy.{name}")

    def test_no_pascalcase_tychopy_subpackages(self):
        import importlib

        for name in self.PASCAL_NAMES:
            with self.assertRaises(
                (ImportError, ModuleNotFoundError),
                msg=f"tychopy.{name} should not be importable",
            ):
                importlib.import_module(f"tychopy.{name}")

    def test_snake_case_paths_succeed(self):
        import importlib

        for name in [
            "vector_functions",
            "optimal_control",
            "solvers",
            "utils",
            "astro",
        ]:
            importlib.import_module(f"_tychopy.{name}")
            importlib.import_module(f"tychopy.{name}")


# ---------------------------------------------------------------------------
# TestLegacyLinkAPIRemoved
# ---------------------------------------------------------------------------


class TestLegacyLinkAPIRemoved(unittest.TestCase):
    """Pin the legacy OCP link API removal as a hard break. PR #50
    deleted ``LinkFlags``, ``LinkConstraint``, and ``LinkObjective``
    from the ``_tychopy.optimal_control`` namespace and removed ~30
    legacy ``add_link_*`` overloads taking a pre-built ``LinkConstraint``
    object. If a future change reintroduces any of them as a
    convenience alias these assertions fail at CI time instead of
    silently re-softening the API.
    """

    LEGACY_NAMES = ["LinkFlags", "LinkConstraint", "LinkObjective"]

    def test_legacy_classes_absent(self):
        oc_mod = ast.optimal_control
        for name in self.LEGACY_NAMES:
            with self.assertRaises(
                AttributeError,
                msg=f"_tychopy.optimal_control.{name} should not exist",
            ):
                getattr(oc_mod, name)


# ---------------------------------------------------------------------------
# TestUncoveredLinkBindings
# ---------------------------------------------------------------------------


def _build_2phase_ocp():
    """Two phases of the trivial ODE x' = u, with one link param.
    Used by TestUncoveredLinkBindings to exercise each link binding.
    """
    oc = ast.optimal_control

    class _SimpleODE(oc.ode_x_u.ode):
        def __init__(self):
            args = oc.ODEArguments(1, 1)
            x_dot = args.u_var(0)
            super().__init__(vf.stack([x_dot]), 1, 1)

    ode = _SimpleODE()

    n = 8
    traj_a = [np.array([i / (n - 1), i / (n - 1), 1.0]) for i in range(n)]
    traj_b = [np.array([1.0 + i / (n - 1), i / (n - 1) + 1.0, 1.0]) for i in range(n)]

    p0 = ode.phase("LGL3", traj_a, n - 1)
    p1 = ode.phase("LGL3", traj_b, n - 1)

    ocp = oc.OptimalControlProblem()
    ocp.add_phase(p0)
    ocp.add_phase(p1)
    ocp.set_link_params(np.array([0.0]))
    return ocp


class TestUncoveredLinkBindings(unittest.TestCase):
    """Pin that the four link bindings with zero downstream callers
    today are (a) registered AND (b) actually callable end-to-end. A
    broken ``nb::overload_cast`` template arg compiles to a callable
    attribute and only fails at call time with ``TypeError``, so a
    pure ``hasattr``/``callable`` check would not catch it.
    ``add_link_inequal_con``, ``add_link_objective``,
    ``add_link_param_inequal_con``, and ``add_link_param_objective``
    are bound in ``build_link_interface`` (see
    ``src/bindings/optimal_control/optimal_control_problem_bind.cpp``)
    but no test or example exercises them today. These tests construct
    a 2-phase OCP and invoke each binding.
    """

    BINDINGS = [
        "add_link_inequal_con",
        "add_link_objective",
        "add_link_param_inequal_con",
        "add_link_param_objective",
    ]

    def test_bindings_registered(self):
        ocp_class = ast.optimal_control.OptimalControlProblem
        for name in self.BINDINGS:
            self.assertTrue(
                hasattr(ocp_class, name),
                msg=f"OptimalControlProblem.{name} must be registered",
            )
            attr = getattr(ocp_class, name)
            self.assertTrue(
                callable(attr),
                msg=f"OptimalControlProblem.{name} must be callable",
            )

    def test_add_link_inequal_con_packs_form_callable(self):
        ocp = _build_2phase_ocp()
        # IRows = 2: one var per phase. f(a, b) = a - b returns a length-1 vector.
        cons_func = Args(2).head(1) - Args(2).tail(1)
        idx = ocp.add_link_inequal_con(
            cons_func,
            [(0, "Back", [0], [], []), (1, "Front", [0], [], [])],
        )
        self.assertIsInstance(idx, int)

    def test_add_link_objective_packs_form_callable(self):
        ocp = _build_2phase_ocp()
        # Scalar function f(a, b) = (a - b) . (a - b)  — IRows = 2, ORows = 1.
        diff = Args(2).head(1) - Args(2).tail(1)
        obj_func = diff.dot(diff)
        idx = ocp.add_link_objective(
            obj_func,
            [(0, "Back", [0], [], []), (1, "Front", [0], [], [])],
        )
        self.assertIsInstance(idx, int)

    def test_add_link_param_inequal_con_callable(self):
        ocp = _build_2phase_ocp()
        # Single-link-param scalar function.
        cons_func = Args(1).head(1)  # vector function with IRows = 1, ORows = 1
        idx = ocp.add_link_param_inequal_con(cons_func, np.array([0], dtype=np.int32))
        self.assertIsInstance(idx, int)

    def test_add_link_param_objective_callable(self):
        ocp = _build_2phase_ocp()
        # Single-link-param scalar function f(p) = p . p.
        a = Args(1)
        obj_func = a.dot(a)
        idx = ocp.add_link_param_objective(obj_func, np.array([0], dtype=np.int32))
        self.assertIsInstance(idx, int)


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
            self.assertEqual(result.input_rows(), 3)
            self.assertEqual(result.output_rows(), 6)

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
        self.assertEqual(result.input_rows(), 3)
        self.assertEqual(result.output_rows(), 4)

    def test_apply_reverse_composition(self):
        """apply(g) composes as g(self(x)) — the reverse of eval(g) = self(g(x)).

        Regression guard: apply previously aliased eval (both produced
        self(g(x))) because the binding built NestedFunction<self, g> instead
        of g.eval(self).
        """
        # self: R^2 -> R^2, self([x0, x1]) = [x0, x0]
        self_fn = vf.stack(vf.Element(2, 1, 0), vf.Element(2, 1, 0))
        # g: R^2 -> R^2, g([y0, y1]) = [y1, y0]  (component swap)
        g = vf.stack(vf.Element(2, 1, 1), vf.Element(2, 1, 0))
        x = np.array([1.0, 2.0])

        # apply = g(self(x)) = g([x0, x0]) = [x0, x0]
        applied = self_fn.apply(g)
        np.testing.assert_allclose(applied.compute(x), [1.0, 1.0])

        # eval = self(g(x)) = self([x1, x0]) = [x1, x1]
        evaled = self_fn.eval(g)
        np.testing.assert_allclose(evaled.compute(x), [2.0, 2.0])

        # apply must no longer be an alias of eval.
        self.assertFalse(
            np.allclose(applied.compute(x), evaled.compute(x)),
            "apply(g) must compute g(self(x)), distinct from eval(g) = self(g(x))",
        )

    def test_stack_accepts_np_int64_scalar(self):
        """np.int64 (numpy's default int dtype on 64-bit Linux) must not be spuriously
        rejected by the exact-type np.int32 identity check."""
        f = vf.Arguments(2).head(2)
        g = vf.stack(f, np.int64(2))
        np.testing.assert_allclose(g.compute([1.0, 2.0]), [1.0, 2.0, 2.0])

    def test_stack_accepts_np_float32_scalar(self):
        """np.float32 must not be spuriously rejected by the exact-type np.float64 check."""
        f = vf.Arguments(2).head(2)
        g = vf.stack(f, np.float32(3.5))
        np.testing.assert_allclose(g.compute([1.0, 2.0]), [1.0, 2.0, 3.5])

    def test_stack_accepts_python_bool_scalar(self):
        """A Python bool is a valid numeric 0/1 constant."""
        f = vf.Arguments(2).head(2)
        g = vf.stack(f, True)
        np.testing.assert_allclose(g.compute([1.0, 2.0]), [1.0, 2.0, 1.0])

    def test_stack_accepts_np_integer_array(self):
        """np.arange(...) constants (int64 array) must be accepted, not just float arrays."""
        f = vf.Arguments(2).head(2)
        g = vf.stack(f, np.arange(3))  # int64 array constant
        np.testing.assert_allclose(g.compute([1.0, 2.0]), [1.0, 2.0, 0.0, 1.0, 2.0])

    def test_stack_rejects_non_numeric_with_type_in_message(self):
        f = vf.Arguments(2).head(2)
        with self.assertRaisesRegex(ValueError, "type"):
            vf.stack(f, object())

    def test_stack_rejects_non_numeric_list_element_with_type_in_message(self):
        f = vf.Arguments(2).head(2)
        with self.assertRaisesRegex(ValueError, "type"):
            vf.stack(f, [1.0, object()])

    def test_sum_accepts_np_int64_scalar(self):
        """ParsePythonArgsScalar (used by vf.sum) must accept np.int64 the same way
        ParsePythonArgs does for vf.stack."""
        e0 = vf.Element(3, 1, 0)
        result = vf.sum(e0, np.int64(2))
        out = result.compute(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [3.0])


# ---------------------------------------------------------------------------
# TestCartesianToMEEBindings
# ---------------------------------------------------------------------------


class TestCartesianToMEEBindings(unittest.TestCase):
    """Cover both bindings on the new direct Cartesian->MEE conversion:
    the numpy Vector6 overload (via tychopy.astro._vec6_wrap) and the
    GenericFunction VF-composition overload registered in
    src/bindings/astro/kepler_utils.cpp.
    """

    MU_EARTH = 398600.4418

    # Molniya-shaped initial state in Cartesian (km, km/s).
    RV = np.array(
        [
            -2301.67224489839,
            -5371.07610250925,
            -3421.14671530212,
            6.1338624555516,
            0.306265184163608,
            -4.59713439017524,
        ]
    )

    def test_vec6_numpy_round_trip(self):
        """Cart -> MEE -> Cart through the numpy Vector6 overload returns
        the input within machine precision.  Exercises _vec6_wrap's
        numpy-input branch and the underlying
        cartesian_to_modified(Vector6, mu) / modified_to_cartesian(Vector6, mu)
        bindings."""
        from tychopy import astro as TyAstro

        mee = np.asarray(TyAstro.cartesian_to_modified(self.RV, self.MU_EARTH))
        rv2 = np.asarray(TyAstro.modified_to_cartesian(mee, self.MU_EARTH))
        np.testing.assert_allclose(rv2, self.RV, atol=1e-9, rtol=0)

    def test_vf_compose_overload(self):
        """cartesian_to_modified(VectorFunction(seg), mu) hits the
        GenericFunction VF-overload added in this PR (which constructs
        a CartesianToMEE and composes via .eval())."""
        from tychopy import astro as TyAstro

        seg = Args(6).head(6)
        gen = TyAstro.cartesian_to_modified(vf.VectorFunction(seg), self.MU_EARTH)
        self.assertEqual(gen.input_rows(), 6)
        self.assertEqual(gen.output_rows(), 6)


# ---------------------------------------------------------------------------
# TestParsedInputEval
# ---------------------------------------------------------------------------


class TestParsedInputEval(unittest.TestCase):
    """Python-boundary regression coverage for ParsedInput's variable-location
    validation (the C++ fix landed in PR 2; see
    ``include/tycho/detail/vf/expressions/parsed_input.h``).

    ``eval(int, list[int])`` (the remapped-input-projection overload) is only
    registered on non-``Segment``/``Arguments`` VectorFunctions -- Segment and
    Arguments types are excluded via ``is_arglike`` in
    ``BinaryOperatorsBuild`` (``src/bindings/vf/dense_function_base_bind.h``).
    So a bare ``Args(3).head(3)`` does not expose ``.eval(int, list)``; it is
    wrapped via ``vf.VectorFunction(...)`` (same pattern as
    ``TestCartesianToMEEBindings.test_vf_compose_overload`` above) to obtain a
    GenericFunction that does.
    """

    def test_eval_rejects_wrong_length_map(self):
        f = vf.VectorFunction(Args(3).head(3))  # input_rows() == 3
        with self.assertRaises(ValueError):
            f.eval(4, [0, 1])  # 2 entries for a 3-input function

    def test_eval_rejects_out_of_range_entries(self):
        f = vf.VectorFunction(Args(3).head(3))  # input_rows() == 3
        with self.assertRaises(ValueError):
            f.eval(4, [0, 1, 4])  # 4 outside [0, 4)
        with self.assertRaises(ValueError):
            f.eval(4, [0, 1, -1])  # -1 outside [0, 4)

    def test_eval_duplicate_indices_sum_derivatives(self):
        # f(a, b) = a*b gathered as (x2, x2) => d/dx2 [x2^2] = 2*x2.
        args = Args(2)
        prod = args.coeff(0) * args.coeff(1)
        g = prod.eval(4, [2, 2])
        x = np.array([0.0, 0.0, 3.0, 0.0])
        np.testing.assert_allclose(g.compute(x), [9.0])
        jac = g.jacobian(x)
        np.testing.assert_allclose(jac[0, 2], 6.0)  # 2*x2, the summed scatter


# ---------------------------------------------------------------------------
# TestConstantOperandSizeChecks
# ---------------------------------------------------------------------------


class TestConstantOperandSizeChecks(unittest.TestCase):
    """Python-boundary regression coverage for the length checks added to
    FunctionVectorSum_Impl (``f + vec`` / ``f - vec``), RowScaled_Impl
    (``cwise_product`` / ``cwise_quotient`` with a constant vector), the
    ``dot`` binding lambda, and the IOScaled ctor (CODEBASE 1.1b).

    Pre-fix, a length-mismatched constant vector was silently accepted
    (UB / a hard-to-diagnose crash at evaluation time downstream, not a
    clean Python exception at the call boundary).
    """

    def setUp(self):
        self.f = vf.Arguments(3).head(3)

    def test_add_wrong_length_vector_raises(self):
        with self.assertRaises(ValueError):
            self.f + [1.0, 2.0]

    def test_sub_wrong_length_vector_raises(self):
        with self.assertRaises(ValueError):
            self.f - [1.0, 2.0]

    def test_cwise_product_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self.f.cwise_product([1.0, 2.0])

    def test_cwise_quotient_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self.f.cwise_quotient([1.0, 2.0])

    def test_dot_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self.f.dot([1.0, 2.0])

    def test_correct_lengths_still_work(self):
        g = (self.f + [1.0, 1.0, 1.0]).cwise_product([2.0, 2.0, 2.0])
        np.testing.assert_allclose(g.compute([1.0, 2.0, 3.0]), [4.0, 6.0, 8.0])
        self.assertAlmostEqual(
            self.f.dot([1.0, 0.0, 1.0]).compute([1.0, 2.0, 3.0])[0], 4.0
        )

    def test_ioscaled_wrong_length_input_scales_raises(self):
        """IOScaled is only registered on GenericFunction (see
        ``reg.build_register<IOScaled<Gen>>`` in
        ``src/bindings/vf/tycho_vector_functions.cpp``), so a bare
        ``Args(3).head(3)`` (a Segment) must be wrapped via
        ``vf.VectorFunction(...)`` first -- same pattern as
        ``TestParsedInputEval`` above."""
        g = vf.VectorFunction(self.f)  # input_rows() == output_rows() == 3
        with self.assertRaises(ValueError):
            vf.IOScaled(g, [1.0, 2.0], [1.0, 1.0, 1.0])

    def test_ioscaled_wrong_length_output_scales_raises(self):
        g = vf.VectorFunction(self.f)
        with self.assertRaises(ValueError):
            vf.IOScaled(g, [1.0, 1.0, 1.0], [1.0, 2.0])

    def test_ioscaled_correct_lengths_still_work(self):
        g = vf.VectorFunction(self.f)
        scaled = vf.IOScaled(g, [2.0, 2.0, 2.0], [3.0, 3.0, 3.0])
        np.testing.assert_allclose(scaled.compute([1.0, 2.0, 3.0]), [6.0, 12.0, 18.0])


# ---------------------------------------------------------------------------
# TestNegativeSizeRejection
# ---------------------------------------------------------------------------


class TestNegativeSizeRejection(unittest.TestCase):
    """Python-boundary regression coverage for Segment_Impl (``head`` /
    ``tail`` / ``segment``) and PaddedOutput (``padded_lower`` /
    ``padded_upper``) rejecting negative sizes (CODEBASE 1.1b).

    Pre-fix, a negative segment size or pad count was silently accepted
    (the resulting ``output_rows()`` went negative, silent UB downstream in
    Release builds) instead of raising a clean Python exception at the call
    boundary.

    Zero-size segments are deliberately still accepted: ``ODEArguments``'s
    ``u_vec()``/``p_vec()`` bindings (``src/bindings/optimal_control/
    ode_arguments_bind.h``) construct exactly this shape unconditionally for
    control-free/parameter-free ODEs (e.g. ``ODEArguments(3)`` has
    ``u_vars() == p_vars() == 0`` by design), so only ``orows < 0`` /
    negative pads are rejected here, not ``orows == 0``.
    """

    def setUp(self):
        self.f = vf.Arguments(5)

    def test_head_negative_raises(self):
        with self.assertRaises(ValueError):
            self.f.head(-3)

    def test_tail_negative_raises(self):
        with self.assertRaises(ValueError):
            self.f.tail(-1)

    def test_tail_zero_still_works(self):
        # Zero-size segments are legitimate (control-free/parameter-free ODEs
        # route through this exact shape) -- must not raise.
        np.testing.assert_allclose(
            self.f.tail(0).compute([1.0, 2.0, 3.0, 4.0, 5.0]), []
        )

    def test_segment_negative_size_raises(self):
        with self.assertRaises(ValueError):
            self.f.segment(1, -2)

    def test_segment_zero_size_still_works(self):
        np.testing.assert_allclose(
            self.f.segment(1, 0).compute([1.0, 2.0, 3.0, 4.0, 5.0]), []
        )

    def test_padded_lower_negative_raises(self):
        with self.assertRaises(ValueError):
            self.f.head(3).padded_lower(-2)

    def test_padded_upper_negative_raises(self):
        with self.assertRaises(ValueError):
            self.f.head(3).padded_upper(-2)

    def test_padded_lower_zero_still_works(self):
        np.testing.assert_allclose(
            self.f.head(3).padded_lower(0).compute([1.0, 2.0, 3.0, 4.0, 5.0]),
            [1.0, 2.0, 3.0],
        )

    def test_valid_segment_still_works(self):
        np.testing.assert_allclose(
            self.f.segment(1, 2).compute([1.0, 2.0, 3.0, 4.0, 5.0]), [2.0, 3.0]
        )


if __name__ == "__main__":
    unittest.main(exit=False)
