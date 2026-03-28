import unittest

import _tychopy as ast
import numpy as np

vf = ast.VectorFunctions
oc = ast.OptimalControl
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Args = vf.Arguments


class CartPole(oc.ode_x_u.ode):
    def __init__(self, l, m1, m2, g):

        args = oc.ODEArguments(4, 1)

        q1 = args.XVar(0)
        q2 = args.XVar(1)
        q1d = args.XVar(2)
        q2d = args.XVar(3)

        q1, q2, q1d, q2d = args.x_vec().tolist()

        u = args.UVar(0)

        q1dd = (
            l * m2 * vf.sin(q2) * (q2d**2) + u + m2 * g * vf.cos(q2) * vf.sin(q2)
        ) / (m1 + m2 * (1 - vf.cos(q2) ** 2))
        q2dd = (
            -1
            * (
                l * m2 * vf.cos(q2) * vf.sin(q2) * (q2d**2)
                + u * vf.cos(q2)
                + (m1 * g + m2 * g) * vf.sin(q2)
            )
            / (l * m1 + l * m2 * (1 - vf.cos(q2) ** 2))
        )

        ode = vf.stack([q1d, q2d, q1dd, q2dd])
        super().__init__(ode, 4, 1)


###############################################################################
class test_CartPole(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.FinalObj = 58.83219229674185  ## Sensitive to Segments
        self.MaxObjError = 0.1
        self.MaximumIters = 20

    def problem_impl(self, tmode, cmode, nsegs, errest):
        m1 = 1
        m2 = 0.3
        l = 0.5
        g = 9.81

        umax = 20
        dmax = 2

        tf = 2
        d = 1

        ts = np.linspace(0, tf, 100)
        IG = [[d * t / tf, np.pi * t / tf, 0, 0, t, 0.00] for t in ts]
        ode = CartPole(l, m1, m2, g)

        phase = ode.phase(tmode, IG, nsegs)
        phase.set_control_mode(cmode)
        phase.add_boundary_value("Front", range(0, 5), [0, 0, 0, 0, 0])
        phase.add_boundary_value("Back", range(0, 5), [d, np.pi, 0, 0, tf])
        phase.add_lu_var_bound("Path", 5, -umax, umax, 1.0)
        phase.add_lu_var_bound("Path", 0, -dmax, dmax, 1.0)
        phase.add_integral_objective(Args(1)[0] ** 2, [5])
        phase.optimizer.print_level = 3
        phase.optimizer.eq_con_tol = 1.0e-8
        phase.adaptive_mesh = True
        phase.set_num_partitions(1, 1)
        phase.mesh_error_estimator = errest
        phase.print_mesh_info = False

        phase.mesh_err_factor = 20

        Flag = phase.optimize()

        Obj = phase.optimizer.last_obj_val
        ObjError = abs(Obj - self.FinalObj)

        self.assertTrue(phase.mesh_converged, "Problem Meshs did not converge converge")

        self.assertLess(
            phase.optimizer.last_iter_num,
            self.MaximumIters,
            "Optimizer iterations exceeded expected maximum",
        )
        self.assertEqual(
            Flag, ast.Solvers.ConvergenceFlags.CONVERGED, "Problem did not converge"
        )
        self.assertLess(
            ObjError,
            self.MaxObjError,
            "Final objective significantly differs from known answer",
        )

    def test_FullProblem(self):

        tmodes = [
            "LGL3",
            "LGL5",
            "LGL7",
        ]
        nsegs = [32, 16, 10]

        for tmode, nseg in zip(tmodes, nsegs):
            with self.subTest(TranscriptionMode=tmode):
                with self.subTest(errorest="deboor"):
                    self.problem_impl(
                        tmode,
                        "HighestOrderSpline",
                        nseg,
                        oc.MeshErrorEstimators.DEBOOR,
                    )
                with self.subTest(errorest="integrator"):
                    self.problem_impl(
                        tmode,
                        "HighestOrderSpline",
                        nseg,
                        oc.MeshErrorEstimators.INTEGRATOR,
                    )


##############################################################################

if __name__ == "__main__":
    unittest.main(exit=False)

    ###########################################################################
