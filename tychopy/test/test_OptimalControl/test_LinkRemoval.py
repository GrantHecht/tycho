"""
Python-boundary coverage for OC review §1.9 (remove_phase/phase bounds
checking + link-index guard).

OptimalControlProblem.remove_phase()/phase() are bound directly to the C++
OptimalControlProblemBase methods (see optimal_control_problem_bind.cpp);
this file pins that the exceptions thrown by the C++ layer (out-of-range
index, phase still referenced by a link function) surface as Python
exceptions rather than crashing or being silently swallowed.
"""

import unittest

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
oc = ast.optimal_control


class SHO_Ode(oc.ode_x.ode):
    """Simple harmonic oscillator: x'' = -x. State (x, v)."""

    def __init__(self):
        args = oc.ODEArguments(2)
        x = args.x_var(0)
        v = args.x_var(1)
        xdot = v
        vdot = (-1.0) * x
        ode = vf.stack([xdot, vdot])
        super().__init__(ode, 2)


def _make_linear_guess(x0, xf, n=5):
    """[x, v, t] node guess -- a trivial linear ramp, just enough to build a
    phase; these tests never solve/optimize the OCP."""
    traj = []
    for i in range(n):
        s = i / (n - 1)
        traj.append([x0 + (xf - x0) * s, xf - x0, s])
    return traj


def _make_phase():
    ode = SHO_Ode()
    return ode.phase("LGL3", _make_linear_guess(0.0, 1.0), 2)


class LinkRemoval(unittest.TestCase):
    def _make_two_phase_ocp(self):
        ocp = oc.OptimalControlProblem()
        ocp.add_phase(_make_phase())
        ocp.add_phase(_make_phase())
        return ocp

    def _make_two_phase_linked_ocp(self):
        ocp = self._make_two_phase_ocp()
        ocp.add_direct_link_equal_con(0, "Back", [0], 1, "Front", [0], auto_scale="auto")
        return ocp

    def test_remove_phase_out_of_range_raises(self):
        ocp = self._make_two_phase_ocp()
        with self.assertRaises(Exception):
            ocp.remove_phase(10)
        with self.assertRaises(Exception):
            ocp.phase(10)

    def test_remove_referenced_phase_raises(self):
        ocp = self._make_two_phase_linked_ocp()
        with self.assertRaises(Exception):
            ocp.remove_phase(0)


if __name__ == "__main__":
    unittest.main()
