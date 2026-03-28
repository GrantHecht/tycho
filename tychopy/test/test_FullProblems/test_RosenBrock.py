import random as rand
import time
import unittest

import _tychopy as ast
import matplotlib.pyplot as plt
import numpy as np

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
solvs = ast.Solvers


def RosenBrockObj(xy=Args(2)):
    x = xy[0]
    y = xy[1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def DiskCon():
    return Args(2).squared_norm() - 2.0


class test_RosenBrock(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.InitPoint = np.array([-1, -1])
        self.KnownSol = np.array([1, 1])
        self.MaximumIters = 25

    def problem_impl(self, con, lsmode):
        Ipoint = [-1, -1]

        prob = solvs.OptimizationProblem()
        prob.set_vars(Ipoint)
        prob.add_objective(RosenBrockObj(), [0, 1])
        prob.add_inequal_con(con, [0, 1])
        prob.optimizer.set_OptLSMode(lsmode)
        prob.optimizer.PrintLevel = 3
        Flag = prob.optimize()
        Fpoint = prob.return_vars()

        SolError = np.linalg.norm(Fpoint - self.KnownSol)

        self.assertEqual(
            Flag, ast.Solvers.ConvergenceFlags.CONVERGED, "Problem did not converge"
        )

        self.assertLess(
            prob.optimizer.LastIterNum,
            self.MaximumIters,
            "Optimizer iterations exceeded expected maximum",
        )

        self.assertLess(
            SolError,
            1.0e-5,
            "Solution variables significantly differ from known answer",
        )

    def test_FullProblem(self):
        lsmodes = ["NOLS", "AUGLANG", "L1"]

        for lsmode in lsmodes:
            with self.subTest(LineSearchMode=lsmode):
                self.problem_impl(DiskCon(), lsmode)


if __name__ == "__main__":
    unittest.main(exit=False)
