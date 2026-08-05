"""
Regression tests: the T6 empty-`what()` sweep folded printed diagnostics into
the exception messages at every ``throw std::invalid_argument("")`` site in
``ode_phase_base.cpp``/``.h`` and ``optimization_problem.cpp``/``.h``. Before
that fix, Python users only ever saw a bare ``ValueError:`` with no text.

The minimal ODE/phase setup mirrors test_SetUnitsDtypes.py's SHO_Ode pattern
(``oc.ode_x.ode`` subclass, no controls) -- the cheapest way to obtain a real
``PhaseInterface``.
"""

import unittest

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
oc = ast.optimal_control


class SHO_Ode(oc.ode_x.ode):
    """Simple harmonic oscillator: x'' = -x. State (x, v), so xdot = (v, -x)."""

    def __init__(self):
        args = oc.ODEArguments(2)
        x = args.x_var(0)
        v = args.x_var(1)
        xdot = v
        vdot = (-1.0) * x
        ode = vf.stack([xdot, vdot])
        super().__init__(ode, 2)
        self.add_idx("x", [0])
        self.add_idx("v", [1])
        self.add_idx("t", [2])
        self.add_idx("xv", [0, 1])


class TestErrorMessagesNotEmpty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ode = SHO_Ode()
        traj_ig = [
            np.array([np.cos(t), -np.sin(t), t]) for t in np.linspace(0.0, 1.0, 20)
        ]
        cls.phase = ode.phase("LGL3", traj_ig, 4)

    def test_lb_greater_than_ub_message(self):
        """Pattern A: lowerbound > upperbound must raise with both values named."""
        with self.assertRaisesRegex(
            ValueError, r"Lower-bound.*greater than Upper-bound"
        ):
            self.phase.add_lu_var_bound("Path", 0, 2.0, 1.0)  # lb > ub

    def test_set_traj_dimension_mismatch_message(self):
        """Pattern D: mesh node width inconsistent with the ODE's xtu dimension."""
        with self.assertRaisesRegex(ValueError, r"[Dd]imension"):
            self.phase.set_traj([np.zeros(2)], 10)  # wrong state width (expects 3)

    def test_lbscale_nonpositive_message(self):
        """Pattern B: a non-positive bound scale must raise with the value named."""
        with self.assertRaisesRegex(ValueError, r"scale.*strictly positive"):
            self.phase.add_lower_delta_var_bound(0, 1.0, -1.0)  # negative scale

    def test_check_function_size_message(self):
        """Pattern E: a boundary-value constraint built over more args than the
        function's input rows must raise with both sizes named."""
        with self.assertRaisesRegex(ValueError, r"Input size of .* does not match"):
            # A 1-input scalar function bound over indices [0, 1] (size 2) at
            # "Front" is inconsistent -- check_function_size requires
            # func.input_rows() == len(indices) for a Front/Path-style region.
            func = vf.Arguments(1).sum()
            self.phase.add_equal_con("Front", func, [0, 1])


class TestOptimizationProblemBoundsOffByOne(unittest.TestCase):
    """``OptimizationProblem::transcribe`` bounds-checks each constraint/
    objective's variable index vector against ``numVars`` (the size of
    ``active_variables_``). Valid indices are ``[0, numVars)``, so the check
    must reject an index exactly equal to ``numVars`` (the boundary), not
    just indices greater than it. Pre-fix, ``maxCoeff() > numVars`` let
    ``index == numVars`` through -- an out-of-bounds read one past the end of
    ``active_variables_``.

    The same check also rejects negative indices (``minCoeff() < 0``): a
    negative index passes the VectorXi caster (which only rejects int32
    overflow, not negative values) and would otherwise reach unchecked
    matrix indexing.
    """

    def test_equality_constraint_index_at_numvars_raises(self):
        solvs = ast.solvers
        prob = solvs.OptimizationProblem()
        prob.set_vars([0.0, 0.0])  # numVars == 2; valid indices are {0, 1}
        con = vf.Arguments(2)  # identity: 2 in, 2 out
        # Index 2 == numVars is the exact off-by-one boundary (not just > numVars).
        prob.add_equal_con(con, [0, 2])
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            prob.solve()

    def test_equality_constraint_negative_index_raises(self):
        """A negative variable index (e.g. -1) is not caught by the VectorXi
        caster (which only rejects int32 overflow), so it must be rejected by
        the ``minCoeff() < 0`` guard in ``transcribe()`` -- otherwise it sails
        past the upper-bound check into unchecked matrix indexing."""
        solvs = ast.solvers
        prob = solvs.OptimizationProblem()
        prob.set_vars([0.0, 0.0])  # numVars == 2; valid indices are {0, 1}
        con = vf.Arguments(2)  # identity: 2 in, 2 out
        prob.add_equal_con(con, [0, -1])
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            prob.solve()

    def test_inequality_constraint_valid_indices_still_work(self):
        """Sanity check of the construction path with in-bounds indices --
        must not raise at transcribe time."""
        solvs = ast.solvers
        prob = solvs.OptimizationProblem()
        prob.set_vars([0.0, 0.0])
        con = vf.Arguments(2).squared_norm() - 2.0
        prob.add_inequal_con(con, [0, 1])  # both indices < numVars == 2
        prob.optimizer.print_level = 0
        prob.solve()  # must not raise


if __name__ == "__main__":
    unittest.main()
