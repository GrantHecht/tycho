"""tests/corpus/problems/stub_converges.py — Task 1 throwaway stub.

Exists only to exercise the harness (``scripts/run_corpus.py``) and the
problem-module contract end-to-end before any real corpus problem exists.
Deleted in Task 5 once the real degenerate/hard/literature tiers land.

Problem: a double-integrator rest-to-rest maneuver.

    states:   x, v
    control:  u
    dynamics: xdot = v, vdot = u
    t in [0, 1], LGL3 collocation, 8 segments

    boundary conditions: x(0) = 0, v(0) = 0, x(1) = 1, v(1) = 0
    objective: minimize 0.5 * integral(u^2) dt

This is a small, well-posed, well-scaled control problem with a linear
system and a convex quadratic objective — PSIOPT converges in a handful of
iterations on defaults.
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
    """Construct, configure, and solve the stub problem.

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 8)
    phase.add_boundary_value("Front", range(0, 2), [0.0, 0.0])
    phase.add_boundary_value("Back", range(0, 2), [1.0, 0.0])
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
