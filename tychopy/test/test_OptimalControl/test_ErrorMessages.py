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
            self.phase.add_lu_var_bound("Path", 0, 0.0, 1.0, -1.0)  # negative scale

    def test_check_function_size_message(self):
        """Pattern E: a boundary-value constraint built over more args than the
        function's input rows must raise with both sizes named."""
        with self.assertRaisesRegex(ValueError, r"Input size of .* does not match"):
            # A 1-input scalar function bound over indices [0, 1] (size 2) at
            # "Front" is inconsistent -- check_function_size requires
            # func.input_rows() == len(indices) for a Front/Path-style region.
            func = vf.Arguments(1).sum()
            self.phase.add_equal_con("Front", func, [0, 1])


if __name__ == "__main__":
    unittest.main()
