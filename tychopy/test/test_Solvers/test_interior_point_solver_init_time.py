import unittest

import _tychopy as ast

vf = ast.vector_functions
solvs = ast.solvers
Args = vf.Arguments


def RosenBrockObj(xy=Args(2)):
    x = xy[0]
    y = xy[1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def DiskCon():
    return Args(2).squared_norm() - 2.0


class test_InteriorPointSolverInitTime(unittest.TestCase):
    """Regression test for InteriorPointSolver's solver-init one-shot contract
    (CODEBASE_REVIEW.md Sec 1.3, item P7).

    ``ensure_solver_initialized()`` (src/solvers/solver_init.cpp) must report
    a nonzero init time only for the process-wide first call that actually
    runs the math-runtime initializer; every subsequent call must report
    ``0.0``, per its documented contract ("Returns ... 0.0 if already
    initialized"). ``InteriorPointSolver::ensure_solver_initialized`` mirrors that value
    into ``opt.last_solver_init_time`` on every ``solve()``/``optimize()``
    call (src/solvers/interior_point_solver.cpp:786-795).

    The underlying ``std::once_flag`` is process-global, so this test does
    not assume it is the first solver initialization to happen anywhere in
    the pytest process (another test module may run first and consume it).
    It only asserts the part of the contract that is robust to call order:
    solving the SAME problem instance's optimizer a second time in this
    process must report exactly ``0.0``.
    """

    def _make_problem(self):
        prob = solvs.OptimizationProblem()
        prob.set_vars([-1, -1])
        prob.add_objective(RosenBrockObj(), [0, 1])
        prob.add_inequal_con(DiskCon(), [0, 1])
        return prob

    def test_init_time_zero_on_second_solve(self):
        prob = self._make_problem()
        ipm = solvs.InteriorPointSolver()
        ipm.print_level = 0

        flag1 = prob.solve(ipm)
        self.assertEqual(flag1.flag, ast.solvers.ConvergenceFlags.CONVERGED)
        first_init_time = ipm.last_solver_init_time
        # Allowed to be > 0 (this is the process-wide first solve) or 0.0
        # (an earlier test in this pytest session already initialized the
        # math runtime) -- both are valid under the one-shot contract.
        self.assertGreaterEqual(first_init_time, 0.0)

        flag2 = prob.solve(ipm)
        self.assertEqual(flag2.flag, ast.solvers.ConvergenceFlags.CONVERGED)
        second_init_time = ipm.last_solver_init_time
        # Never the process-wide first call to ensure_solver_initialized()
        # (already consumed by flag1's solve, if not earlier) -> must be
        # exactly 0.0 per the documented contract.
        self.assertEqual(second_init_time, 0.0)


if __name__ == "__main__":
    unittest.main(exit=False)
