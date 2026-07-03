"""Direct PSIOPT.solve/optimize and single-trajectory integrate* must
release the GIL (CODEBASE_REVIEW 1.1).

Technique: crank sys.setswitchinterval so the interpreter never
preempts; a parked worker thread can then only run if the measured call
releases the GIL itself. Deterministic — no timing thresholds.
"""

import sys
import threading
import unittest

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
oc = ast.optimal_control
solvs = ast.solvers
IVPAlg = oc.IVPAlg
Args = vf.Arguments


class SHO_Ode(oc.ode_x.ode):
    def __init__(self):
        args = oc.ODEArguments(2)
        x = args.x_var(0)
        v = args.x_var(1)
        super().__init__(vf.stack([v, (-1.0) * x]), 2)


def runs_concurrently(fn):
    """True iff a parked worker thread gets to run while fn() executes."""
    flag = threading.Event()
    parked = threading.Event()
    go = threading.Event()

    def worker():
        parked.set()
        go.wait()  # parked here, GIL released by wait()
        flag.set()  # only reachable once this thread re-acquires the GIL

    t = threading.Thread(target=worker)
    t.start()
    parked.wait()
    old = sys.getswitchinterval()
    sys.setswitchinterval(300.0)  # no involuntary GIL switches from here on
    try:
        go.set()  # wakes worker at the C level; it still needs the GIL
        fn()  # if fn holds the GIL throughout, worker cannot run
        released = flag.is_set()
    finally:
        sys.setswitchinterval(old)
        t.join()
    return released


class TestGilRelease(unittest.TestCase):
    def test_single_trajectory_integrate_releases_gil(self):
        ode = SHO_Ode()
        integ = ode.integrator(IVPAlg.DOPRI87, 1e-3)
        integ.set_abs_tol(1e-13)
        integ.set_rel_tol(1e-13)
        x0 = np.array([1.0, 0.0, 0.0])
        # ~50 calls guarantees the worker wakes during at least one release.
        self.assertTrue(
            runs_concurrently(lambda: [integ.integrate(x0, 50.0) for _ in range(50)])
        )

    def test_direct_psiopt_solve_releases_gil(self):
        def rosen_obj(xy=Args(2)):
            x, y = xy[0], xy[1]
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        prob = solvs.OptimizationProblem()
        prob.set_vars([-1.0, -1.0])
        prob.add_objective(rosen_obj(), [0, 1])
        prob.add_inequal_con(Args(2).squared_norm() - 2.0, [0, 1])
        prob.optimizer.print_level = 3
        prob.optimize()  # transcribe + first solve (already guarded path)
        x0 = np.array([-1.0, -1.0])
        self.assertTrue(
            runs_concurrently(lambda: [prob.optimizer.solve(x0) for _ in range(20)])
        )


if __name__ == "__main__":
    unittest.main(exit=False)
