"""
Regression tests: phase.set_units(**kwargs) must accept any numpy.number
scalar/array dtype, not just the exact np.int32 / np.float64 identity match.

The minimal ODE/phase setup mirrors test_NewMethods.py's SHO_Ode pattern
(``oc.ode_x.ode`` subclass, no controls) -- the cheapest way to obtain a real
``PhaseInterface`` -- plus named index groups (via ``add_idx``, mirroring
``ode_base_class.ODEBase.add_Vgroups``) since ``set_units(**kwargs)`` requires
named variable groups to resolve its keyword names.
"""

import unittest

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
oc = ast.optimal_control


class SHO_Ode(oc.ode_x.ode):
    """Simple harmonic oscillator: x'' = -x. State (x, v), so xdot = (v, -x).

    Registers named index groups "x", "v" (single-index) and "t" (the packed
    input's time slot, index == x_vars()) so that set_units(**kwargs) has
    something to resolve.
    """

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


class TestSetUnitsDtypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ode = SHO_Ode()
        traj_ig = [
            np.array([np.cos(t), -np.sin(t), t]) for t in np.linspace(0.0, 1.0, 20)
        ]
        cls.phase = ode.phase("LGL3", traj_ig, 4)

    def test_set_units_accepts_numpy_scalars(self):
        """np.float32/np.int64 scalars must not be spuriously rejected by the
        exact-type np.int32/np.float64 identity check."""
        self.phase.set_units(t=np.float32(2.0))  # must not raise
        self.phase.set_units(t=np.int64(2))  # must not raise

    def test_set_units_accepts_python_bool_scalar(self):
        """A Python bool is a valid numeric 0/1 unit value."""
        self.phase.set_units(t=True)  # must not raise

    def test_set_units_accepts_numpy_array_elements(self):
        """An index-group-sized array of np.float32 elements must be accepted."""
        self.phase.set_units(
            xv=np.array([1.5, 2.5], dtype=np.float32)
        )  # must not raise

    def test_set_units_accepts_numpy_integer_array_elements(self):
        """An index-group-sized array of np.int64 elements (numpy's default int
        dtype on 64-bit Linux) must be accepted."""
        self.phase.set_units(xv=np.array([1, 2], dtype=np.int64))  # must not raise

    def test_set_units_rejects_non_numeric_with_type_in_message(self):
        with self.assertRaisesRegex(ValueError, "type"):
            self.phase.set_units(t=object())

    def test_set_units_rejects_non_numeric_list_element_with_type_in_message(self):
        with self.assertRaisesRegex(ValueError, "type"):
            self.phase.set_units(xv=[1.0, object()])


if __name__ == "__main__":
    unittest.main()
