"""Regression tests for MEE two-body frame construction (CODEBASE 1.2).

Two live defects fixed here:

1. ``MEETwoBodyFrame.MEETwoBodyEOMs`` indexed ``otherAccs[0]`` unconditionally
   when ``len(otherAccs) <= 1``, so the documented ballistic default
   ``otherAccs=[]`` raised ``IndexError`` instead of adding zero perturbing
   acceleration.
2. ``MEETwoBody_LT.__init__`` called ``self.MEETwoBodyFrame(...)`` (the mixin
   *class* name, shadowing nothing since it's never assigned as an attribute)
   instead of ``self.MEETwoBodyEOMs(...)``, so the class was entirely
   unconstructable (``AttributeError``).
"""

import unittest

import numpy as np

import tychopy as typy
import tychopy.astro.astro_models as am
from tychopy.astro.Extensions.MEETwoBodyFrame import MEETwoBodyFrame

vf = typy.vector_functions
Args = vf.Arguments

# Sane non-dimensional MEE state: [p, f, g, h, k, L]. p > 0, f/g/h/k small
# (near-circular, near-equatorial orbit), L an arbitrary true-longitude angle.
_SANE_MEE_STATE = np.array([1.0, 0.01, 0.02, 0.005, -0.003, 0.7])


class TestMEETwoBodyFrameBallistic(unittest.TestCase):
    def test_meetwobody_frame_ballistic_constructs(self):
        """otherAccs=[] (the documented ballistic default) must not raise."""
        f = MEETwoBodyFrame(1.0, 1.0)
        ode = f.MEETwoBodyEOMs(Args(6), otherAccs=[], otherEOMs=[])
        self.assertEqual(ode.input_rows(), 6)
        self.assertEqual(ode.output_rows(), 6)

    def test_meetwobody_frame_ballistic_evaluates_finite(self):
        """The zero-acceleration EOM VF must evaluate to finite values on a
        sane MEE state (not merely construct)."""
        f = MEETwoBodyFrame(1.0, 1.0)
        ode = f.MEETwoBodyEOMs(Args(6), otherAccs=[], otherEOMs=[])
        out = ode.compute(_SANE_MEE_STATE)
        self.assertEqual(out.shape, (6,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_meetwobody_frame_single_acc_still_works(self):
        """len(otherAccs) == 1 path (passthrough) must be unaffected."""
        f = MEETwoBodyFrame(1.0, 1.0)
        X = Args(9)
        acc = X.tail3()
        ode = f.MEETwoBodyEOMs(X.head(6), otherAccs=[acc], otherEOMs=[])
        state = np.zeros(9)
        state[0:6] = _SANE_MEE_STATE
        out = ode.compute(state)
        self.assertEqual(out.shape, (6,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_meetwobody_frame_two_accs_still_summed(self):
        """len(otherAccs) > 1 path (vf.sum) must be unaffected."""
        f = MEETwoBodyFrame(1.0, 1.0)
        X = Args(12)
        acc1 = X.segment3(6)
        acc2 = X.segment3(9)
        ode = f.MEETwoBodyEOMs(X.head(6), otherAccs=[acc1, acc2], otherEOMs=[])
        state = np.zeros(12)
        state[0:6] = _SANE_MEE_STATE
        out = ode.compute(state)
        self.assertEqual(out.shape, (6,))
        self.assertTrue(np.all(np.isfinite(out)))


class TestMEETwoBodyLT(unittest.TestCase):
    def test_meetwobody_lt_constructs(self):
        """MEETwoBody_LT.__init__ referenced the mixin class instead of the
        MEETwoBodyEOMs method — construction must not raise."""
        ode = am.MEETwoBody_LT(1.0, 1.0)
        self.assertEqual(ode.x_vars(), 6)

    def test_meetwobody_lt_constructs_and_integrates(self):
        # Default zero-arg thruster=LowThrustAcc(): ThrustExpr(u, astar) = u
        # (non-dimensional passthrough) — pinned so a future Task 8 signature
        # change to LowThrustAcc must preserve this zero-arg behavior.
        ode = am.MEETwoBody_LT(1.0, 1.0)
        integ = ode.integrator(0.01)

        # Layout is [x(6), t, u(3)] per ODEArguments(6, 3).
        X0 = np.zeros(10)
        X0[0:6] = _SANE_MEE_STATE
        X0[6] = 0.0
        X0[7:10] = np.array([1e-4, 0.0, 0.0])

        Xf = integ.integrate(X0, 1e-3)

        self.assertEqual(len(Xf), 10)
        self.assertTrue(np.all(np.isfinite(Xf)))


class TestMEETwoBodyCSI(unittest.TestCase):
    def test_meetwobody_csi_constructs(self):
        """MEETwoBody_CSI already calls MEETwoBodyEOMs correctly; cheap
        construction smoke test for insurance."""
        from tychopy.astro.Extensions.ThrusterModels import CSIThruster

        thruster = CSIThruster(F=0.32, Isp=3000, M=4000)
        ode = am.MEETwoBody_CSI(1.0, 1.0, thruster)
        self.assertEqual(ode.x_vars(), 7)


if __name__ == "__main__":
    unittest.main()
