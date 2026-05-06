"""
Method selection and basic correctness via Python bindings.

Verifies Tsit5, BS3, BS5, Vern7, Vern8, Vern9 are selectable (both by
enum and by string) and produce accurate results on a LEO circular orbit
over a quarter period.
"""

import unittest

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
oc = ast.optimal_control
IVPAlg = oc.IVPAlg


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


class TestNewMethods(unittest.TestCase):
    """Verify each method integrates a SHO over tf=1 to matching accuracy."""

    def _integrate_sho(self, method, atol=1e-10, rtol=1e-10):
        ode = SHO_Ode()
        integ = ode.integrator(method, 1e-3)
        integ.set_abs_tol(atol)
        integ.set_rel_tol(rtol)
        x0 = np.array([1.0, 0.0, 0.0])  # x(0)=1, v(0)=0, t=0
        xf = integ.integrate(x0, 1.0)
        # Exact: x(1) = cos(1) ≈ 0.5403, v(1) = -sin(1) ≈ -0.8415
        return xf

    def test_Tsit5(self):
        xf = self._integrate_sho(IVPAlg.Tsit5)
        self.assertAlmostEqual(xf[0], np.cos(1.0), places=8)
        self.assertAlmostEqual(xf[1], -np.sin(1.0), places=8)

    def test_BS3(self):
        xf = self._integrate_sho(IVPAlg.BS3, atol=1e-8, rtol=1e-8)
        self.assertAlmostEqual(xf[0], np.cos(1.0), places=6)
        self.assertAlmostEqual(xf[1], -np.sin(1.0), places=6)

    def test_BS5(self):
        xf = self._integrate_sho(IVPAlg.BS5)
        self.assertAlmostEqual(xf[0], np.cos(1.0), places=8)
        self.assertAlmostEqual(xf[1], -np.sin(1.0), places=8)

    def test_Vern7(self):
        xf = self._integrate_sho(IVPAlg.Vern7, atol=1e-12, rtol=1e-13)
        self.assertAlmostEqual(xf[0], np.cos(1.0), places=10)
        self.assertAlmostEqual(xf[1], -np.sin(1.0), places=10)

    def test_Vern8(self):
        xf = self._integrate_sho(IVPAlg.Vern8, atol=1e-12, rtol=1e-13)
        self.assertAlmostEqual(xf[0], np.cos(1.0), places=10)
        self.assertAlmostEqual(xf[1], -np.sin(1.0), places=10)

    def test_Vern9(self):
        xf = self._integrate_sho(IVPAlg.Vern9, atol=1e-12, rtol=1e-13)
        self.assertAlmostEqual(xf[0], np.cos(1.0), places=10)
        self.assertAlmostEqual(xf[1], -np.sin(1.0), places=10)

    def test_StringDispatch(self):
        """Each method is also selectable by string name."""
        for name in ["Tsit5", "BS3", "BS5", "Vern7", "Vern8", "Vern9"]:
            with self.subTest(method=name):
                ode = SHO_Ode()
                integ = ode.integrator(name, 1e-3)
                integ.set_abs_tol(1e-8)
                integ.set_rel_tol(1e-8)
                xf = integ.integrate(np.array([1.0, 0.0, 0.0]), 1.0)
                # Loose tolerance — we're verifying the string parser hands
                # off to the right RK tableau, not method accuracy.
                self.assertAlmostEqual(xf[0], np.cos(1.0), places=4)

    def test_InternalTagsNotExposed(self):
        """Internal template-dispatch tags are not exposed in the Python enum."""
        internal_names = [
            "Tsit5Trans",
            "BS3Trans",
            "BS5Trans",
            "Vern7Trans",
            "Vern8Trans",
            "Vern9Trans",
            "DOPRI5",
            "RK4Classic",
        ]
        for name in internal_names:
            with self.subTest(tag=name):
                self.assertFalse(
                    hasattr(IVPAlg, name),
                    f"IVPAlg.{name} should not be exposed to Python",
                )


if __name__ == "__main__":
    unittest.main()
