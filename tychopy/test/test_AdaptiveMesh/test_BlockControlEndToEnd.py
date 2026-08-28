import unittest

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
oc = ast.optimal_control
Args = vf.Arguments


class CartPole(oc.ode_x_u.ode):
    def __init__(self, l, m1, m2, g):

        args = oc.ODEArguments(4, 1)

        q1, q2, q1d, q2d = args.x_vec().tolist()

        u = args.u_var(0)

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
class test_BlockControlEndToEnd(unittest.TestCase):
    """Regression test for the per-block control refresh defect.

    ``calc_global_error`` re-propagates the trajectory block by block to
    estimate the ENDTOEND mesh error, but must re-seed the control segment
    of the re-integrator from the trajectory at the start of each block when
    the phase uses ``BlockConstant`` control parameterization (the
    ``BlockConstant`` re-integrator carries no control interpolant, so it
    otherwise keeps replaying whatever control happened to be loaded before
    the loop started -- block 0's -- for every later block). With
    ``num_tran_card_states_ > 2`` (LGL5/LGL7), a block spans interior
    transcription nodes whose control differs from the block-start control,
    so the bug produces a badly wrong ENDTOEND error and either fails to
    converge or "converges" against nonsense.
    """

    @classmethod
    def setUpClass(cls):
        cls.m1 = 1
        cls.m2 = 0.3
        cls.l = 0.5
        cls.g = 9.81

        cls.umax = 20
        cls.dmax = 2

        cls.tf = 2
        cls.d = 1

    def _make_cartpole_phase(self, tmode, nsegs):
        ts = np.linspace(0, self.tf, 100)
        IG = [[self.d * t / self.tf, np.pi * t / self.tf, 0, 0, t, 0.00] for t in ts]
        ode = CartPole(self.l, self.m1, self.m2, self.g)

        phase = ode.phase(tmode, IG, nsegs)
        phase.add_boundary_value("Front", range(0, 5), [0, 0, 0, 0, 0])
        phase.add_boundary_value("Back", range(0, 5), [self.d, np.pi, 0, 0, self.tf])
        phase.add_lu_var_bound("Path", 5, -self.umax, self.umax)
        phase.add_lu_var_bound("Path", 0, -self.dmax, self.dmax)
        phase.add_integral_objective(Args(1)[0] ** 2, [5])
        phase.print_mesh_info = False
        return phase

    def _make_engine(self, **kwargs):
        ipm = ast.solvers.InteriorPointSolver(**kwargs)
        ipm.print_level = 0
        ipm.eq_con_tol = 1.0e-8
        return ipm

    def test_blockconstant_endtoend_converges(self):
        # Reference objective: a fixed, densely-segmented BlockConstant solve
        # (no adaptive mesh involved) for this control parameterization.
        ref_phase = self._make_cartpole_phase("LGL5", 48)
        ref_phase.set_control_mode("BlockConstant")
        ref_ipm = self._make_engine()

        ref_result = ref_phase.solve(ref_ipm)
        self.assertEqual(
            ref_result.flag,
            ast.solvers.ConvergenceFlags.CONVERGED,
            "Reference problem did not converge",
        )
        ref_obj = ref_ipm.last_obj_val

        # Solve under test: BlockConstant control (u_vars > 0), adaptive mesh
        # with ENDTOEND mesh-error criteria -> exercises the calc_global_error
        # per-block control refresh defect fixed alongside this test.
        phase = self._make_cartpole_phase("LGL5", 12)
        phase.set_control_mode("BlockConstant")
        phase.adaptive_mesh = True
        phase.set_num_partitions(1)
        ipm = self._make_engine()
        ipm.qp_threads = 1
        phase.mesh_error_criteria = oc.MeshErrorAggregation.ENDTOEND
        phase.mesh_error_distributor = oc.MeshErrorAggregation.AVG
        phase.mesh_error_estimator = oc.MeshErrorEstimators.DEBOOR
        phase.set_mesh_tol(1e-6)
        phase.mesh_err_factor = 20

        result = phase.solve(ipm)

        self.assertEqual(
            result.flag,
            ast.solvers.ConvergenceFlags.CONVERGED,
            "Problem did not converge",
        )
        self.assertTrue(
            phase.mesh_converged, "Mesh did not converge under ENDTOEND criteria"
        )

        # `MeshIterateInfo.global_error` is not exposed to Python, so verify
        # correctness indirectly: pre-fix, the ENDTOEND error is computed
        # against block-0's stale control for every later block, which either
        # prevents convergence outright or lets the mesh "converge" against a
        # trajectory whose objective has drifted from the true optimum.
        obj = ipm.last_obj_val
        self.assertLess(
            abs(obj - ref_obj),
            0.1,
            "Adaptively-refined BlockConstant objective differs from the reference solve; "
            "the ENDTOEND global error is likely still being computed against a stale "
            "per-block control.",
        )


##############################################################################

if __name__ == "__main__":
    unittest.main(exit=False)

    ###########################################################################
