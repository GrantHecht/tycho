"""tests/corpus/problems/deg_zero_objective.py — E2 G0 degenerate tier.

Baseline problem: a double-integrator rest-to-rest maneuver.

    states:   x, v
    control:  u
    dynamics: xdot = v, vdot = u
    t in [0, 1], LGL3 collocation, 32 segments

    boundary conditions: x(0) = 0, v(0) = 0, x(1) = 1, v(1) = 0, t(1) = 1

Note on time: the phase's node "t" (index 2 of [x, v, t, u]) is itself a
free decision variable in this collocation formulation, not automatically
tied to [0, 1] just because the initial guess spans that range, so t is
pinned explicitly at Front (t = 0) and Back (t = 1) to match the brief's
stated "t in [0, 1]" fixed-duration setup.

Perturbation: NO objective is added at all — no
``add_integral_objective``, ``add_value_objective``, or any other
objective term. This is pure feasibility: PSIOPT must only satisfy the
dynamics defects and the boundary conditions, with nothing to minimize.

Any (x, v, u) trajectory satisfying ẋ = v, v̇ = u and the four boundary
values is a valid solution — the feasible set is a genuine (non-singleton)
continuum, since u(t) is essentially free subject only to reaching
x(1) = 1, v(1) = 0 from rest. With no objective, the Lagrangian's
objective Hessian block is identically zero (only the constraint-Hessian
contribution to the KKT system survives), which is the "zero objective
Hessian" pathology named in the Task 2 brief: it exercises PSIOPT's
perturbation ladder (the mechanism that adds regularization when the
reduced KKT system would otherwise be singular/indefinite purely from a
zero cost Hessian) rather than any constraint-side degeneracy.

Observed on defaults 2026-07-16: CONVERGED, 3 iterations, objective 0
(as expected — there is nothing to minimize). This is a FINDING TO
REPORT: the zero-objective-Hessian pathology named in the Task 2 brief
does NOT genuinely manifest as any difficulty here — PSIOPT's
perturbation ladder handles a fully-zero cost Hessian trivially,
converging in the same handful of iterations as the well-posed baseline
(feasibility alone is an easy problem for this system). A harder
zero-objective probe would likely need a feasible set with a more
complex/nonconvex shape than this linear double integrator's; that is
out of scope for the degenerate tier (see the "hard"/"literature" tiers
in later G0 tasks).
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control

TIER = "degenerate"
TIMEOUT = 30
SOLVE_MODE = "optimize"


class _DoubleIntegrator(oc.ODEBase):
    def __init__(self):
        XVars = 2
        UVars = 1
        args = oc.ODEArguments(XVars, UVars)
        v = args.x_vec()[1]
        u = args.u_vec()[0]
        ode = vf.stack([v, u])
        super().__init__(ode, XVars, UVars)


def _initial_guess(n=100):
    ts = np.linspace(0.0, 1.0, n)
    xs = np.linspace(0.0, 1.0, n)
    return [[x, 0.0, t, 0.0] for x, t in zip(xs, ts)]


def build():
    """Construct the pure-feasibility (no-objective) problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 32)
    # Indices are [x, v, t, u]; t is pinned at both ends so the phase is a
    # genuinely fixed-duration (T=1) transfer, matching the brief's "t in
    # [0, 1]" setup (an un-pinned Back time is itself a free decision
    # variable in this collocation formulation).
    phase.add_boundary_value("Front", range(0, 3), [0.0, 0.0, 0.0])
    phase.add_boundary_value("Back", range(0, 3), [1.0, 0.0, 1.0])
    # Deliberately no objective term of any kind.

    return phase
