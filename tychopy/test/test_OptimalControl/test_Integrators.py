import time
import unittest

import _tychopy as ast
import matplotlib.pyplot as plt
import numpy as np

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


class LorenzODE(oc.ode_x.ode):
    def __init__(self, sigma, rho, beta):

        x, y, z = oc.ODEArguments(3).x_vec().tolist()

        ode = vf.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

        super().__init__(ode, 3)


class CauchyEulerODE(oc.ode_x.ode):
    def __init__(self, a, b):

        self.a = a
        self.b = b

        args = oc.ODEArguments(2)
        x = args.x_var(0)
        xdot = args.x_var(1)
        t = args.t_var()

        xddot = -a * xdot / t - b * x / t**2

        ode = vf.stack([xdot, xddot])

        super().__init__(ode, 2)

    def analytic(self, x0, xdot0, t0, tf, n=1000):

        r1, r2 = np.roots([1.0, self.a - 1, self.b])

        ts = np.linspace(t0, tf, n)

        if np.isreal(r1) and np.isreal(r2):
            M = np.array([[t0**r1, t0**r2], [r1 * t0 ** (r1 - 1), r2 * t0 ** (r2 - 1)]])

            c1, c2 = np.dot(np.linalg.inv(M), np.array([x0, xdot0]))

            xs = c1 * ts**r1 + c2 * ts**r2
            xdots = r1 * c1 * ts ** (r1 - 1) + c2 * r2 * ts ** (r2 - 1)

        else:
            alpha = np.real(r1)
            beta = np.imag(r1)

            M = np.array(
                [
                    [
                        (t0**alpha) * np.cos(beta * np.log(t0)),
                        (t0**alpha) * np.sin(beta * np.log(t0)),
                    ],
                    [
                        (t0 ** (alpha - 1))
                        * (
                            alpha * np.cos(beta * np.log(t0))
                            - beta * np.sin(beta * np.log(t0))
                        ),
                        (t0 ** (alpha - 1))
                        * (
                            alpha * np.sin(beta * np.log(t0))
                            + beta * np.cos(beta * np.log(t0))
                        ),
                    ],
                ]
            )

            c1, c2 = np.dot(np.linalg.inv(M), np.array([x0, xdot0]))

            xs = c1 * (ts**alpha) * np.cos(beta * np.log(ts)) + c2 * (
                ts**alpha
            ) * np.sin(beta * np.log(ts))

            xdots = c1 * (ts ** (alpha - 1)) * (
                alpha * np.cos(beta * np.log(ts)) - beta * np.sin(beta * np.log(ts))
            ) + c2 * (ts ** (alpha - 1)) * (
                alpha * np.sin(beta * np.log(ts)) + beta * np.cos(beta * np.log(ts))
            )

        return xs, xdots, ts


class QuatModel(oc.ode_7_3.ode):
    def __init__(self, I):
        Xvars = 7
        Uvars = 3
        Ivars = Xvars + 1 + Uvars

        self.Ivec = I
        ############################################################
        args = vf.Arguments(Ivars)

        q = args.head(4)
        w = args.segment3(4)
        T = args.tail3()

        qdot = vf.quat_product(q, w.padded_lower(1)) / 2.0

        L = w.cwise_product(I)
        wdot = (L.cross(w) + T).cwise_quotient(I)
        ode = vf.stack(qdot, wdot)

        ##############################################################
        super().__init__(ode, Xvars, Uvars)

    def DetumbleLaw(self):
        w = Args(3).head3()
        Lhat = w.cwise_product(self.Ivec).normalized()
        return -Lhat


class test_Integrators(unittest.TestCase):
    def test_Lorenz(self):
        """
        Tests: Adaptive Integration of Autonomous ODEs
        """
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0

        abstol = 1.0e-13
        defstepsize = 0.001
        minstepsize = 0.000000001
        errtol = 1.0e-6
        n = 2000

        tf = 20
        X0 = np.array([1, 1, 1, 0])
        ode = LorenzODE(sigma, rho, beta)

        integ = ode.integrator(defstepsize)
        integ.set_abs_tol(abstol)
        integ.adaptive = True

        Traj = integ.integrate_dense(X0, tf, n)

        XF = np.array([13.79319963, 12.95180398, 34.90160871, 20])

        Err = np.linalg.norm(XF - Traj[-1])

        self.assertLess(Err, errtol, "Integration Error exceeds expected maximum")

        if __name__ == "__main__":
            Traj = np.array(Traj)
            fig = plt.figure()
            ax = plt.subplot(projection="3d")
            ax.plot(Traj[:, 0], Traj[:, 1], Traj[:, 2])
            plt.show()

    def test_CauchyEuler(self):
        """
        Tests: Adaptive Integration of non-autonomous ODEs
        """
        a = -0.5
        b = 16

        x0 = 1
        xdot0 = 0.25
        t0 = 0.1
        tf = 10

        abstol = 1.0e-13

        defstepsize = 0.01
        minstepsize = 0.0000000001

        errtol = 1.0e-11

        ode = CauchyEulerODE(a, b)

        integ = ode.integrator("DOPRI54", defstepsize)
        integ.set_abs_tol(abstol)
        integ.adaptive = True

        n = 100

        xs, xdots, ts = ode.analytic(x0, xdot0, t0, tf, n)

        Traj = integ.integrate_dense([x0, xdot0, t0], tf, n)
        T = np.array(Traj).T

        xerr = max(abs(xs - T[0]))
        xdoterr = max(abs(xdots - T[1]))

        maxerr = max(xerr, xdoterr)

        self.assertLess(maxerr, errtol, "Integration Error exceeds expected maximum")

        if __name__ == "__main__":
            plt.plot(ts, abs(xs - T[0]))
            plt.plot(ts, abs(xdots - T[1]))
            plt.yscale("log")
            plt.show()

            plt.plot(T[2], T[0])
            plt.plot(T[2], T[1])

            plt.show()

    def test_Detumble(self):
        """
        Tests: Adaptive Integration of state dependent control law, Quaternions, Cross Products
        """

        QErrtol = 1.0e-10
        WErrtol = 1.0e-6

        Ivec = np.array([1.0, 2.0, 3.0])
        W0 = np.array([3.0, 9.0, 3.0])

        X0 = np.zeros((11))
        X0[3] = 1
        X0[4:7] = W0

        ode = QuatModel(Ivec)
        integ = ode.integrator(0.01, ode.DetumbleLaw(), range(4, 7))
        integ.adaptive = True
        integ.set_abs_tol(1.0e-14)
        tf = np.linalg.norm(W0 * Ivec)

        n = 1000

        Traj = integ.integrate_dense(X0, tf)

        QF = np.array([-0.893804752502, 0.125508984388, 0.070813040593, 0.424671723248])

        QErr = np.linalg.norm(Traj[-1][0:4] - QF)
        WErr = np.linalg.norm(Traj[-1][4:7])

        self.assertLess(
            QErr, QErrtol, "Quaternion Integration Error exceeds expected maximum"
        )
        self.assertLess(
            WErr, WErrtol, "Angular Velocity Integration Error exceeds expected maximum"
        )

        if __name__ == "__main__":
            T = np.array(Traj).T

            plt.plot(T[7], T[4])
            plt.plot(T[7], T[5])
            plt.plot(T[7], T[6])
            plt.show()

    def test_TwoBodySTM(self):
        """
        Tests: Serial and parallel STM computation
        """
        ode = ast.Astro.Kepler.ode(1)
        kprop = ast.Astro.Kepler.KeplerPropagator(1.0)  # Ground Truth

        integ = ode.integrator(0.01)
        integ.set_abs_tol(1.0e-13)
        integ.adaptive = True

        Xtol = 1.0e-10
        Jtol = 1.0e-9

        x0 = 1
        vy0 = 1.35
        vz0 = 0.1
        tf = 10

        X0 = np.zeros((7))
        X0[0] = x0
        X0[4] = vy0
        X0[5] = vz0

        KX = np.copy(X0)
        KX[6] = tf

        fx, jx = kprop.vf().computeall(KX, np.ones((6)))[0:2]

        Xf, STM = integ.integrate_stm(X0, tf)

        Xerr = np.linalg.norm(Xf[0:6] - fx)
        Jerr = abs(STM[0:6, 0:6] - jx[0:6, 0:6]).max()

        with self.subTest("Serial"):
            self.assertLess(
                Xerr, Xtol, "State Integration Error exceeds expected maximum"
            )
            self.assertLess(
                Jerr, Jtol, "STM Integration Error exceeds expected maximum"
            )

        Xf, STM = integ.integrate_stm_parallel(X0, tf, 8)

        Xerr = np.linalg.norm(Xf[0:6] - fx)
        Jerr = abs(STM[0:6, 0:6] - jx[0:6, 0:6]).max()

        with self.subTest("Parallel"):
            self.assertLess(
                Xerr, Xtol, "State Integration Error exceeds expected maximum"
            )
            self.assertLess(
                Jerr, Jtol, "STM Integration Error exceeds expected maximum"
            )

    def test_EventDetection(self):

        r = 1.0
        v = 1.1
        t0 = 0.0
        tf = 200.0

        X0t0 = np.zeros((7))
        X0t0[0] = r
        X0t0[4] = v
        X0t0[6] = t0

        def ApseFunc():
            R, V = Args(7).tolist([(0, 3), (3, 3)])
            return R.dot(V)

        direction = -1
        stopcode = False
        ApoApseEvent = (ApseFunc(), direction, stopcode)

        direction = 1
        stopcode = False
        PeriApseEvent = (ApseFunc(), direction, stopcode)

        direction = 0
        stopcode = 10
        AllApseEvent = (ApseFunc(), direction, stopcode)

        Events = [ApoApseEvent, PeriApseEvent, AllApseEvent]

        ode = ast.Astro.Kepler.ode(1)

        integ = ode.integrator(0.01)
        integ.set_abs_tol(1.0e-13)
        integ.adaptive = True

        integ.event_tol = 1.0e-10
        integ.max_event_iters = 12

        Xf, EventLocs1 = integ.integrate(X0t0, tf, Events)
        Xf, EventLocs2 = integ.integrate(X0t0, tf, Events)

        self.assertTrue(len(EventLocs1) == len(EventLocs2))

        self.assertTrue(len(EventLocs1[0]) == 5)
        self.assertTrue(len(EventLocs1[1]) == 5)
        self.assertTrue(len(EventLocs1[2]) == 10)

        afunc = ApseFunc()

        for i in range(0, len(EventLocs1)):
            self.assertTrue(len(EventLocs1[i]) == len(EventLocs2[i]))

            for j in range(0, len(EventLocs1[i])):
                Xerr = np.linalg.norm(EventLocs1[i][j][0:6] - EventLocs2[i][j][0:6])
                Fxerr = abs(afunc(EventLocs1[i][j])[0])

                self.assertLess(
                    Xerr,
                    1.0e-10,
                    "Forward time and backward time event states are different",
                )

                self.assertLess(
                    Fxerr, integ.event_tol, "Event root error exceeds tolerance"
                )

    def test_BatchCalls1(self):
        a = -0.5
        b = 16

        x0 = 1
        xdot0 = 0.25
        t0 = 0.1
        tf = 10

        abstol = 1.0e-13

        defstepsize = 0.01
        minstepsize = 0.0000000001

        errtol = 1.0e-11

        ode = CauchyEulerODE(a, b)

        integ = ode.integrator("DOPRI87", defstepsize)
        integ.set_abs_tol(abstol)
        integ.set_initial_step_size(defstepsize)
        integ.vectorize_batch_calls = True

        batchsizes = [1, 3, 4, 15, 100, 1003]
        X0 = [x0, xdot0, t0]

        for batchsize in batchsizes:
            tfs = np.linspace(tf, tf * 2, batchsize)

            X0s = [X0] * batchsize

            Xfs = integ.integrate(X0s, tfs)

            for tfi, Xf in zip(tfs, Xfs):
                xs, xdots, ts = ode.analytic(x0, xdot0, t0, tfi, 5)

                xerr = abs(xs[-1] - Xf[0])
                xdoterr = abs(xdots[-1] - Xf[1])

                maxerr = max(xerr, xdoterr)

                self.assertLess(
                    maxerr, errtol, "Integration Error exceeds expected maximum"
                )

    def test_StopFuncExceptionPropagates(self):
        """Pin the contract that a Python `stop_func` raising an exception
        propagates to the caller as a Python exception (NOT a silent
        truthy stop). The PR review flagged this as a possible silent-
        failure in the integrator_bind.h `pyfunc(x).ptr()` path; this
        test confirms nanobind's nb::python_error propagation makes the
        concern moot. If a future binding rewrite breaks propagation,
        this test will catch it.
        """
        ode = ast.Astro.Kepler.ode(1.0)
        integ = ode.integrator(0.01)
        integ.set_abs_tol(1.0e-10)

        X0 = np.zeros(7)
        X0[0] = 1.0
        X0[4] = 1.0

        sentinel = "user-injected stop_func failure"

        def bad_stop(_x):
            raise ValueError(sentinel)

        with self.assertRaises(ValueError) as ctx:
            integ.integrate_dense(X0, 5.0, 100, bad_stop)
        self.assertIn(sentinel, str(ctx.exception))

    def test_BatchCalls2(self):

        kprop = ast.Astro.Kepler.KeplerPropagator(1.0)  # Ground Truth
        ode1 = ast.Astro.Kepler.ode(1)
        ode2 = ast.OptimalControl.ode_6.ode(ode1.vf(), 6)
        ode3 = ast.OptimalControl.ode_x.ode(ode1.vf(), 6)
        ode4 = ast.OptimalControl.ode_x_u.ode(ode1.vf(), 6, 0)
        ode5 = ast.OptimalControl.ode_x_u_p.ode(ode1.vf(), 6, 0, 0)

        for ode in [ode1, ode2, ode3, ode4, ode5]:
            integ = ode.integrator(0.001)
            integ.set_abs_tol(1.0e-13)
            integ.vectorize_batch_calls = True

            Xtol = 1.0e-11
            Jtol = 1.0e-10
            Htol = 1.0e-9

            x0 = 1
            vy0 = 1.35
            vz0 = 0.1
            tf = 10

            X0 = np.zeros((7))
            X0[0] = x0
            X0[4] = vy0
            X0[5] = vz0

            batchsizes = [1, 3, 4, 5, 15, 42, 69, 103]

            for batchsize in batchsizes:
                tfs = np.linspace(tf * 0.5, tf * 1.5, batchsize)

                X0s = [X0] * batchsize

                LF = np.ones((7))
                LF[6] = 0
                LFs = [LF] * batchsize

                Result1 = integ.integrate(X0s, tfs)
                Result2 = integ.integrate_stm(X0s, tfs)
                Result3 = integ.integrate_stm2(X0s, tfs, LFs)

                for tfi, Res1, Res2, Res3 in zip(tfs, Result1, Result2, Result3):
                    Xf1 = Res1
                    Xf2, J2 = Res2
                    Xf3, J3, H3 = Res3

                    KX = np.copy(X0)
                    KX[6] = tfi
                    fx, jx, gx, hx = kprop.vf().computeall(KX, np.ones((6)))

                    for Xf in [Xf1, Xf2, Xf3]:
                        Xerr = np.linalg.norm(Xf[0:6] - fx)

                        self.assertLess(
                            Xerr,
                            Xtol,
                            "State Integration Error exceeds expected maximum",
                        )
                    for J in [J2, J3]:
                        Jerr = abs((J[0:6, 0:6] - jx[0:6, 0:6])).max()

                        self.assertLess(
                            Jerr, Jtol, "STM Integration Error exceeds expected maximum"
                        )

                    Herr = abs((H3[0:6, 0:6] - hx[0:6, 0:6]) / hx[0:6, 0:6]).max()

                    self.assertLess(
                        Herr, Htol, "Hessian Integration Error exceeds expected maximum"
                    )


class TestBindingValidators(unittest.TestCase):
    """Pins the Phase 2 Python-side validator contract: setters that route
    through C++ set_* now raise ValueError on invalid input instead of
    silently storing NaN / non-physical values that would later blow up
    inside the adaptive loop. A regression bypassing the bind-side wrapper
    would accept the bad value and this class would catch it."""

    def _make(self):
        ode = LorenzODE(10.0, 28.0, 8.0 / 3.0)
        return ode.integrator(0.01)

    def test_event_tol_rejects_nonpositive(self):
        integ = self._make()
        with self.assertRaises(ValueError):
            integ.event_tol = -1.0
        with self.assertRaises(ValueError):
            integ.event_tol = 0.0
        with self.assertRaises(ValueError):
            integ.event_tol = float("nan")
        with self.assertRaises(ValueError):
            integ.event_tol = float("inf")
        # Positive round-trips cleanly.
        integ.event_tol = 1e-9
        self.assertEqual(integ.event_tol, 1e-9)

    def test_max_event_iters_rejects_below_one(self):
        integ = self._make()
        with self.assertRaises(ValueError):
            integ.max_event_iters = 0
        with self.assertRaises(ValueError):
            integ.max_event_iters = -5
        integ.max_event_iters = 7
        self.assertEqual(integ.max_event_iters, 7)

    def test_set_abs_tol_rejects_negative_and_nonfinite(self):
        integ = self._make()
        with self.assertRaises(ValueError):
            integ.set_abs_tol(-1e-8)
        with self.assertRaises(ValueError):
            integ.set_abs_tol(float("nan"))
        with self.assertRaises(ValueError):
            integ.set_abs_tol(float("inf"))
        integ.set_abs_tol(1e-10)  # sanity: zero allowed, positive works

    def test_set_rel_tol_rejects_negative_and_nonfinite(self):
        integ = self._make()
        with self.assertRaises(ValueError):
            integ.set_rel_tol(-1e-8)
        with self.assertRaises(ValueError):
            integ.set_rel_tol(float("nan"))
        with self.assertRaises(ValueError):
            integ.set_rel_tol(float("inf"))
        integ.set_rel_tol(1e-10)

    def test_def_step_size_rejects_nonpositive(self):
        integ = self._make()
        with self.assertRaises(ValueError):
            integ.def_step_size = -0.01
        with self.assertRaises(ValueError):
            integ.def_step_size = 0.0
        integ.def_step_size = 0.05
        self.assertEqual(integ.def_step_size, 0.05)

    def test_max_step_change_rejects_invalid(self):
        integ = self._make()
        with self.assertRaises(ValueError):
            integ.max_step_change = 0.9  # must be > 1
        with self.assertRaises(ValueError):
            integ.max_step_change = 1.0
        with self.assertRaises(ValueError):
            integ.max_step_change = float("inf")
        integ.max_step_change = 5.0

    def test_max_steps_round_trip(self):
        integ = self._make()
        integ.set_max_steps(123)
        self.assertEqual(integ.get_max_steps(), 123)

    def test_max_steps_rejects_non_positive(self):
        integ = self._make()
        with self.assertRaises((ValueError, RuntimeError)):
            integ.set_max_steps(0)
        with self.assertRaises((ValueError, RuntimeError)):
            integ.set_max_steps(-1)

    def test_max_steps_enforces_cap(self):
        # Force a runaway by demanding an impossible tolerance, then cap at
        # a tiny value. The loop must trip the cap rather than spin forever.
        # Assert the message names "max_steps" so a different RuntimeError
        # (e.g., the non-finite-state guard) would not silently pass.
        integ = self._make()
        integ.set_abs_tol(1.0e-30)
        integ.set_rel_tol(1.0e-30)
        integ.set_max_steps(5)
        # LorenzODE has 3 state vars + 1 t_var.
        X0 = np.array([1.0, 1.0, 1.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "max_steps"):
            integ.integrate(X0, 10.0)

    def test_max_steps_default_value(self):
        # Default cap is 1_000_000 (see Integrator::max_steps_ default). A
        # regression that changed the default would surface here as an
        # unexpected value without breaking any integration in the examples.
        # A distinctive round-trip covers the binding layout in the other
        # direction — set/get against a non-default value.
        integ = self._make()
        self.assertEqual(integ.get_max_steps(), 1_000_000)
        integ.set_max_steps(12345)
        self.assertEqual(integ.get_max_steps(), 12345)
        integ.set_max_steps(1_000_000)

    def test_integrate_stm2_exception_path(self):
        # Regression: the nanobind function-pointer cast for integrate_stm2
        # previously hard-coded Eigen::MatrixXd for Jacobian/Hessian. For
        # statically-sized DODEs (Kepler's 7-state/8-input) those resolve
        # to fixed-size inline-storage matrices, so the cast overruns the
        # return slot and aligned_free runs on garbage pointers during
        # unwind. A regression would surface as a segfault or aligned_free
        # crash — not as a Python-visible RuntimeError. By forcing a throw
        # mid-call on the Kepler static-sized path, we exercise the
        # exception-propagation pathway of the binding return slot;
        # surviving this call without SIGSEGV is the pass condition.
        kepler_ode = ast.Astro.Kepler.ode(1)  # mu=1 (dimensionless units)
        integ = kepler_ode.integrator(0.001)
        integ.set_abs_tol(1.0e-30)
        integ.set_rel_tol(1.0e-30)
        integ.set_max_steps(5)

        X0 = np.zeros(7)
        X0[0] = 1.0
        X0[4] = 1.35
        LF = np.ones(7)
        LF[6] = 0.0

        with self.assertRaises(RuntimeError):
            integ.integrate_stm2([X0], np.array([10.0]), [LF])


from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    unittest.main(exit=False)
