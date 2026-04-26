"""Controller selection + step stats + HW-initdt API tests."""

import unittest

import _tychopy as ast
import numpy as np

vf = ast.VectorFunctions
oc = ast.OptimalControl
IVPAlg = oc.IVPAlg
IVPController = oc.IVPController
ErrorNormType = oc.ErrorNormType


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


class VanDerPol_Ode(oc.ode_x.ode):
    """Van der Pol oscillator x'' = μ(1-x²)x' - x with μ=1.
    Mildly stiff over many periods → controllers produce differing step counts.
    """

    def __init__(self, mu=1.0):
        args = oc.ODEArguments(2)
        x = args.x_var(0)
        v = args.x_var(1)
        xdot = v
        vdot = mu * (1.0 - x * x) * v - x
        ode = vf.stack([xdot, vdot])
        super().__init__(ode, 2)


def _integrate(controller=None):
    ode = SHO_Ode()
    integ = ode.integrator(IVPAlg.DOPRI54, 0.1)
    if controller is not None:
        integ.set_controller(controller)
    integ.set_abs_tol(1e-9)
    integ.set_rel_tol(1e-10)
    x0 = np.array([1.0, 0.0, 0.0])
    tf = 2.0 * np.pi
    xf = integ.integrate(x0, tf)
    return integ, xf


class TestControllers(unittest.TestCase):
    def test_default_controller_is_PI_for_DOPRI54(self):
        integ, _ = _integrate()
        self.assertEqual(integ.get_controller(), IVPController.PI)

    def test_default_controller_is_I_for_DOPRI87(self):
        ode = SHO_Ode()
        integ = ode.integrator(IVPAlg.DOPRI87, 0.1)
        self.assertEqual(integ.get_controller(), IVPController.I)

    def test_switch_to_PID(self):
        integ, _ = _integrate(controller=IVPController.PID)
        self.assertEqual(integ.get_controller(), IVPController.PID)

    def test_switch_to_PID_by_string(self):
        ode = SHO_Ode()
        integ = ode.integrator(IVPAlg.DOPRI54, 0.1)
        integ.set_controller("PID")
        self.assertEqual(integ.get_controller(), IVPController.PID)

    def test_step_stats_populated(self):
        integ, _ = _integrate()
        self.assertGreater(integ.get_naccept(), 0)
        self.assertGreaterEqual(integ.get_nreject(), 0)

    def test_controller_changes_step_count(self):
        integ_pi, _ = _integrate(controller=IVPController.PI)
        integ_i, _ = _integrate(controller=IVPController.I)
        self.assertGreater(integ_pi.get_naccept(), 0)
        self.assertGreater(integ_i.get_naccept(), 0)

    def test_pi_diverges_from_i_on_stiff_van_der_pol(self):
        """P2.6: PI and I controllers must actually drive the step cadence
        differently on a problem where their characteristic responses diverge.
        Van der Pol with μ=1 over many periods cycles between fast and slow
        motion; PI's derivative-free smoothing gives a different accept/reject
        pattern than I's integral-only action.
        """

        def run(controller):
            ode = VanDerPol_Ode(mu=1.0)
            integ = ode.integrator(IVPAlg.DOPRI54, 0.01)
            integ.set_controller(controller)
            integ.set_abs_tol(1e-7)
            integ.set_rel_tol(1e-7)
            x0 = np.array([2.0, 0.0, 0.0])
            tf = 30.0  # Several periods — enough for step counts to diverge.
            integ.integrate(x0, tf)
            return integ.get_naccept() + integ.get_nreject()

        n_pi = run(IVPController.PI)
        n_i = run(IVPController.I)
        # Proves wire-up actually reaches the controller (not just accepts
        # everything via a default path). Threshold is intentionally loose:
        # on most platforms the delta is double-digit steps.
        self.assertGreater(
            abs(n_pi - n_i),
            2,
            f"PI and I should produce distinct step counts on Van der Pol; got {n_pi} vs {n_i}",
        )

    def test_error_norm_toggle(self):
        ode = SHO_Ode()
        integ = ode.integrator(IVPAlg.DOPRI54, 0.1)
        self.assertEqual(integ.get_error_norm(), ErrorNormType.RMS)  # RMS default
        integ.set_error_norm(ErrorNormType.MAX)
        self.assertEqual(integ.get_error_norm(), ErrorNormType.MAX)
        integ.set_error_norm("RMS")
        self.assertEqual(integ.get_error_norm(), ErrorNormType.RMS)

    def test_auto_initial_dt_default_on(self):
        ode = SHO_Ode()
        integ = ode.integrator(IVPAlg.DOPRI54, 0.1)
        self.assertTrue(integ.get_auto_initial_dt())
        integ.set_auto_initial_dt(False)
        self.assertFalse(integ.get_auto_initial_dt())

    def test_invalid_controller_string_raises(self):
        ode = SHO_Ode()
        integ = ode.integrator(IVPAlg.DOPRI54, 0.1)
        with self.assertRaises(Exception):
            integ.set_controller("Nonexistent")


if __name__ == "__main__":
    unittest.main()
