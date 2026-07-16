"""tests/corpus/problems/stub_fails.py — Task 1 throwaway stub.

Exists only to exercise the harness (``scripts/run_corpus.py``) and the
problem-module contract end-to-end before any real corpus problem exists.
Deleted in Task 5 once the real degenerate/hard/literature tiers land.

Same double-integrator setup as ``stub_converges.py``, but with two
CONFLICTING terminal equality constraints on x(1): one pinning it to 1.0,
a second (independent ``add_boundary_value`` call) pinning it to 2.0. The
resulting NLP is infeasible by construction — this probes that the harness
honestly records non-convergence (failed/diverged/error) rather than
silently reporting success.

Note: an earlier revision used a much smaller conflict gap (1.0 vs 1.001).
That residual (~5e-4 per constraint) landed inside PSIOPT's default
acceptable-equality-constraint tolerance, so the solver reported
ACCEPTABLE rather than a true failure — observed on defaults 2026-07-16.
The 1.0 vs 2.0 gap here is comfortably outside that tolerance.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "degenerate"
TIMEOUT = 30


class _DoubleIntegrator(oc.ODEBase):
    def __init__(self):
        XVars = 2
        UVars = 1
        args = oc.ODEArguments(XVars, UVars)
        v = args.x_vec()[1]
        u = args.u_vec()[0]
        ode = vf.stack([v, u])
        super().__init__(ode, XVars, UVars)


def _initial_guess(n=9):
    ts = np.linspace(0.0, 1.0, n)
    xs = np.linspace(0.0, 1.0, n)
    return [[x, 0.0, t, 0.0] for x, t in zip(xs, ts)]


def build_and_solve(configure) -> dict:
    """Construct, configure, and solve the (infeasible) stub problem.

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 8)
    phase.add_boundary_value("Front", range(0, 2), [0.0, 0.0])
    phase.add_boundary_value("Back", [1], [0.0])
    # Two independent equality constraints on the same terminal state
    # (x(1)) with conflicting target values: infeasible by construction.
    phase.add_boundary_value("Back", [0], [1.0])
    phase.add_boundary_value("Back", [0], [2.0])
    phase.add_integral_objective((Args(1)[0] ** 2) / 2, [3])

    configure(phase.optimizer)
    flag = phase.optimize()

    optimizer = phase.optimizer
    try:
        objective = float(optimizer.last_obj_val)
    except AttributeError:
        objective = None
    try:
        iterations = int(optimizer.last_iter_num)
    except AttributeError:
        iterations = None

    return {
        "flag": flag.name,
        "objective": objective,
        "iterations": iterations,
        "notes": "",
    }
