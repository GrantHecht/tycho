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

Two more live defects fixed in Task 8 (see below):

3. ``ThrusterModels.LowThrustAcc.__init__`` used ``== False`` sentinel checks
   on ``LTacc``/``NonDim_LTacc`` to detect "not specified". Since
   ``0.0 == False`` is ``True`` in Python, an explicit dimensional
   ``LTacc=0.0`` was silently misread as "unset", and ``ThrustExpr`` then
   took the non-dimensional branch with the *default* ``NonDim_LTacc=True``
   (numerically 1.0) instead of the requested zero-dimensional-thrust
   branch — live-verified: ``LowThrustAcc(LTacc=0.0).ThrustExpr(3.0, 4.0)``
   returned ``3.0`` instead of ``0.0``.
4. ``CR3BPFrame.CalcSubPoint`` ran an uncapped ``while`` Newton loop checking
   the FULL residual ``func.compute(X)`` while the update only ever adjusts
   ``X[0:2]`` via a 2x2 sub-Jacobian of rows ``[3:5]`` — residual components
   outside that slice the update cannot influence. No iteration cap means any
   IG/func combination that cannot drive the full residual below ``1e-14``
   hangs forever.
"""

import unittest

import numpy as np
import pytest

import tychopy as typy
import tychopy.astro.astro_models as am
from tychopy.astro.Extensions.CR3BPFrame import CR3BPFrame
from tychopy.astro.Extensions.MEETwoBodyFrame import MEETwoBodyFrame
from tychopy.astro.Extensions.ThrusterModels import LowThrustAcc

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


class TestLowThrustAcc:
    """LowThrustAcc None-sentinel fix (Task 8, defect 3)."""

    def test_lowthrustacc_accepts_explicit_zero_dimensional(self):
        """Explicit dimensional LTacc=0.0 must not be misread as "unset" —
        was: spurious silent fallback to the non-dim default (bug), not even
        a ValueError; ThrustExpr must route to the DIMENSIONAL branch."""
        t = LowThrustAcc(LTacc=0.0)
        assert t.LTacc == 0.0
        assert t.NDLTacc is None
        # Dimensional branch: u * (LTacc / astar) == 3.0 * (0.0 / 4.0) == 0.0
        assert t.ThrustExpr(3.0, 4.0) == 0.0

    def test_lowthrustacc_default_still_nondim_unity(self):
        """Zero-arg default must pin today's non-dim accel == 1.0 passthrough
        behavior (NonDim_LTacc defaulted to True, numerically 1.0)."""
        t = LowThrustAcc()
        assert t.LTacc is None
        assert t.NDLTacc == 1.0
        # Non-dim branch: u * NDLTacc == u * 1.0 == u
        assert t.ThrustExpr(3.0, 4.0) == 3.0

    def test_lowthrustacc_both_provided_raises(self):
        with pytest.raises(ValueError):
            LowThrustAcc(NonDim_LTacc=0.5, LTacc=1e-4)

    def test_lowthrustacc_explicit_nondim_still_works(self):
        """Explicit non-dim path (LTacc left at its new None default) must
        still route to the non-dim branch."""
        t = LowThrustAcc(NonDim_LTacc=0.5)
        assert t.LTacc is None
        assert t.NDLTacc == 0.5
        assert t.ThrustExpr(2.0, 100.0) == 1.0


class TestCalcSubPoint:
    """CalcSubPoint Newton iteration cap (Task 8, defect 4)."""

    @pytest.mark.timeout(30)
    def test_calcsubpoint_converges_bounded(self):
        """The real CR3BP_SolarSail.__init__ path calls CalcSubPoints (which
        calls CalcSubPoint for L1-L5); construction must return quickly and
        the resulting sub-Lagrange points must be finite."""
        frame = am.CR3BP_SolarSail()
        for name in ("SubL1", "SubL2", "SubL3", "SubL4", "SubL5"):
            pt = getattr(frame, name)
            assert pt.shape == (3,)
            assert np.all(np.isfinite(pt))

    @pytest.mark.timeout(30)
    def test_calcsubpoint_nonconvergent_raises_not_hangs(self):
        """A func whose driven residual (rows [3:5]) can never be reduced
        below tol must raise RuntimeError within max_iters, not hang. This
        exercises exactly the code path that looped forever pre-fix (no
        iteration cap) while remaining safe to run post-fix (the cap
        guarantees termination)."""

        class _NeverConvergesFunc:
            """Stand-in VectorFunction: residual rows [3:5] are a constant
            nonzero vector regardless of X, with an invertible (identity)
            2x2 sub-Jacobian, so Newton always takes a nonzero step but the
            residual magnitude never drops below tol."""

            def input_rows(self):
                return 6

            def compute(self, X):
                out = np.zeros(6)
                out[3] = 1.0
                out[4] = 1.0
                return out

            def jacobian(self, X):
                J = np.zeros((6, 6))
                J[3, 0] = 1.0
                J[4, 1] = 1.0
                return J

        frame = CR3BPFrame(1.0, 1.0, 1.0)
        with pytest.raises(RuntimeError):
            frame.CalcSubPoint(_NeverConvergesFunc(), np.array([0.1, 0.1, 0.0]))


if __name__ == "__main__":
    unittest.main()
