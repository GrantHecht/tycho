import unittest

import matplotlib.pyplot as plt
import numpy as np

import tychopy as ast
import tychopy.OptimalControl as oc
import tychopy.VectorFunctions as vf
from tychopy.VectorFunctions import Arguments as Args


class ODE(oc.ODEBase):
    def __init__(self):

        XVars = 1
        UVars = 1

        args = oc.ODEArguments(XVars, UVars)
        x = args.x_var(0)
        u = args.u_var(0)
        xdot = 0.5 * x + u
        super().__init__(xdot, XVars, UVars)

    class obj(vf.ScalarFunction):
        def __init__(self):
            x, u = Args(2).tolist()
            obj = u * u + x * u + 1.25 * x**2
            super().__init__(obj)


###############################################################################
class test_ObjScaling(unittest.TestCase):
    """
    Testing to see whether we get same solution with and without autoscaling
    """

    @classmethod
    def setUpClass(self):

        self.FinalState = 0.3185865574270634
        self.MaxStateError = 0.001

    def problem_impl(self, tmode, cmode, mtol):

        ode = ODE()

        nsegs = 20

        iscale = np.pi
        vscale = np.e

        xstar = 2.11
        tstar = 3.3
        ustar = 5.1
        pstar = 8.2

        x0 = 1.0
        t0 = 0.0
        tf = 1.0
        u0 = 0.4

        TrajIG = [[x0, t, u0] for t in np.linspace(t0, tf, 100)]

        #### Ground Truth, No AutoScaling #################
        phase = ode.phase(tmode, TrajIG, nsegs)
        phase.set_adaptive_mesh(True)
        phase.set_control_mode(cmode)
        phase.set_mesh_tol(mtol)

        phase.add_boundary_value("Front", [0, 1], [x0, t0])
        phase.add_boundary_value("Back", [1], [tf])
        phase.add_integral_objective(ODE.obj() * iscale, [0, 2])
        phase.add_value_objective("Back", 0, vscale)

        phase.set_num_partitions(1, 1)
        phase.optimizer.cnr_mode = True
        phase.optimizer.set_qp_ordering_mode("MINDEG")

        if __name__ != "__main__":
            phase.optimizer.print_level = 3
            phase.print_mesh_info = False

        phase.optimize()

        xf1 = phase.return_traj()[-1][0]
        ###################################################

        #### AutoScaling, Value + Integral ###############
        phase = ode.phase(tmode, TrajIG, nsegs)
        phase.set_adaptive_mesh(True)
        phase.set_mesh_tol(mtol)

        phase.set_units([xstar, tstar, ustar])
        phase.add_boundary_value("Front", [0, 1], [x0, t0])
        phase.add_boundary_value("Back", [1], [tf])
        phase.add_integral_objective(ODE.obj() * iscale, [0, 2])
        phase.add_value_objective("Back", 0, vscale)

        phase.set_auto_scaling(True)
        phase.set_num_partitions(1, 1)
        phase.optimizer.cnr_mode = True
        phase.optimizer.set_qp_ordering_mode("MINDEG")

        if __name__ != "__main__":
            phase.optimizer.print_level = 3
            phase.print_mesh_info = False

        phase.optimize()
        xf2 = phase.return_traj()[-1][0]
        ###################################################

        #### AutoScaling, Value + IntegralParam ###############
        phase = ode.phase(tmode, TrajIG, nsegs)
        phase.set_adaptive_mesh(True)
        phase.set_mesh_tol(mtol)

        phase.set_units([xstar, tstar, ustar])
        phase.set_static_params([0.0], [pstar])

        phase.add_boundary_value("Front", [0, 1], [x0, t0])
        phase.add_boundary_value("Back", [1], [tf])
        phase.add_integral_param_function(ODE.obj(), [0, 2], 0)
        phase.add_value_objective("Back", 0, vscale)
        phase.add_value_objective("StaticParams", 0, iscale)

        phase.set_auto_scaling(True)

        phase.set_num_partitions(1, 1)
        phase.optimizer.cnr_mode = True
        phase.optimizer.set_qp_ordering_mode("MINDEG")

        if __name__ != "__main__":
            phase.optimizer.print_level = 3
            phase.print_mesh_info = False

        phase.optimize()
        xf3 = phase.return_traj()[-1][0]
        ###################################################

        #### AutoScaling, Value + IntegralParam style 2 ###############
        phase = ode.phase(tmode, TrajIG, nsegs)
        phase.set_adaptive_mesh(True)
        phase.set_mesh_tol(mtol)

        phase.set_units([xstar, tstar, ustar])
        phase.set_static_params([0.0], [pstar])

        phase.add_boundary_value("Front", [0, 1], [x0, t0])
        phase.add_boundary_value("Back", [1], [tf])
        phase.add_integral_param_function(ODE.obj(), [0, 2], 0)

        phase.add_state_objective("Back", Args(2).dot([vscale, iscale]), [0], [], [0])

        phase.set_auto_scaling(True)

        phase.set_num_partitions(1, 1)
        phase.optimizer.cnr_mode = True
        phase.optimizer.set_qp_ordering_mode("MINDEG")

        if __name__ != "__main__":
            phase.optimizer.print_level = 3
            phase.print_mesh_info = False

        phase.optimize()
        xf4 = phase.return_traj()[-1][0]
        ###################################################

        #### AutoScaling, Multi Phase Value + Integral ###############
        phase1 = ode.phase(tmode, TrajIG[0 : int(nsegs / 2)], nsegs)
        phase1.set_adaptive_mesh(True)
        phase1.set_mesh_tol(mtol)

        phase1.set_units([xstar, tstar, ustar])
        phase1.add_boundary_value("Front", [0, 1], [x0, t0])
        phase1.add_integral_objective(ODE.obj() * iscale, [0, 2])
        phase1.add_delta_time_equal_con(tf / 2.0)

        phase2 = ode.phase(tmode, TrajIG[int(nsegs / 2) :], nsegs)
        phase2.set_adaptive_mesh(True)
        phase2.set_mesh_tol(mtol)

        phase2.set_units([xstar * 0.99, tstar * 0.783, ustar * 1.1])
        phase2.set_adaptive_mesh(True)
        phase2.set_mesh_tol(mtol)

        phase2.add_integral_objective(ODE.obj() * iscale, [0, 2])
        phase2.add_boundary_value("Back", [1], [tf])
        phase2.add_value_objective("Back", 0, vscale)

        ocp = oc.OptimalControlProblem()

        ocp.add_phase(phase1)
        ocp.add_phase(phase2)

        ocp.add_forward_link_equal_con(phase1, phase2, range(0, 3))

        ocp.set_auto_scaling(True, True)
        ocp.set_num_partitions(1, 1)
        ocp.optimizer.cnr_mode = True
        ocp.optimizer.set_qp_ordering_mode("MINDEG")

        if __name__ != "__main__":
            ocp.optimizer.print_level = 3
            ocp.print_mesh_info = False

        ocp.optimize()
        xf5 = phase2.return_traj()[-1][0]
        ###################################################

        xferr1 = abs(xf1 - self.FinalState)
        self.assertLess(
            xferr1,
            self.MaxStateError,
            "Final state significantly differs from known answer",
        )

        xferr2 = abs(xf2 - self.FinalState)
        self.assertLess(
            xferr2,
            self.MaxStateError,
            "Final state significantly differs from known answer",
        )

        xferr3 = abs(xf3 - self.FinalState)
        self.assertLess(
            xferr3,
            self.MaxStateError,
            "Final state significantly differs from known answer",
        )

        xferr4 = abs(xf4 - self.FinalState)
        self.assertLess(
            xferr4,
            self.MaxStateError,
            "Final state significantly differs from known answer",
        )

        xferr5 = abs(xf5 - self.FinalState)
        self.assertLess(
            xferr5,
            self.MaxStateError,
            "Final state significantly differs from known answer",
        )

    def test_FullProblem(self):

        for tmode in ["LGL3", "LGL5", "LGL7", "Trapezoidal"]:
            with self.subTest(TranscriptionMode=tmode):
                mtol = 1.0e-5 if tmode == "Trapezoidal" else 1.0e-7
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode, "HighestOrderSpline", mtol)
                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode, "BlockConstant", mtol)


if __name__ == "__main__":
    unittest.main(exit=False)
