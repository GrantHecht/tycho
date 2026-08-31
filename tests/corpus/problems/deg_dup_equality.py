"""tests/corpus/problems/deg_dup_equality.py — E2 G0 degenerate tier.

Baseline problem: a double-integrator rest-to-rest maneuver.

    states:   x, v
    control:  u
    dynamics: xdot = v, vdot = u
    t in [0, 1], LGL3 collocation, 32 segments

    boundary conditions: x(0) = 0, v(0) = 0, x(1) = 1, v(1) = 0
    objective: minimize 0.5 * integral(u^2) dt

Note on time: the phase's node "t" (index 2 of [x, v, t, u]) is itself a
free decision variable in this collocation formulation, not automatically
tied to [0, 1] just because the initial guess spans that range — an
un-pinned Back time lets the interior-point solver silently rescale the effective transfer
duration (verified empirically: without a Back constraint on t, this
model converges to a near-zero objective by stretching the mesh's
effective duration far beyond 1, not the T=1 minimum-energy optimum). So
Front and Back both pin t explicitly here (t(0) = 0, t(1) = 1), matching
the brief's stated "t in [0, 1]" fixed-duration setup.

Perturbation: the terminal (``"Back"``) boundary condition is added TWICE,
identically — two separate ``add_boundary_value`` calls with the exact same
region, indices, and values. Each call appends its own rows to the
constraint Jacobian, so the two calls produce three exactly-duplicate row
pairs (x(1) = 1, v(1) = 0, t(1) = 1): the constraint Jacobian is exactly
(not just numerically nearly) rank-deficient by 3, entirely at the
boundary. This is the simplest possible degenerate perturbation of the
baseline and is the counterpart to ``deg_redundant_defects`` (which forces
the same kind of exact row duplication but interior to the path rather than
at a single boundary node).

Because the two constraints are identical (not conflicting), the
*feasible set* is unchanged from the well-posed baseline — only the
Jacobian's rank is affected. This probes whether the interior-point solver's KKT
factorization (Pardiso) tolerates an exactly rank-deficient equality block
without perturbation/regularization failures, as distinct from
``deg_conflicting_equality`` (same terminal state, incompatible target
values, genuinely infeasible) or ``deg_near_infeasible`` (a genuinely
infeasible but numerically well-conditioned reachability bound).

Observed on defaults 2026-07-16: CONVERGED, 3 iterations, objective
6.0115 (matching the analytic minimum-energy optimum, ~6.0, to 3 sig
figs). This is a FINDING TO REPORT PROMINENTLY, not hidden: the
pathology does NOT genuinely manifest as non-convergence for this
perturbation — the interior-point solver's KKT factorization evidently tolerates an exactly
duplicated boundary equality block without difficulty, converging in as
few iterations as the well-posed baseline. Useful negative result for
future E2 work: a plain duplicate-row degeneracy at the boundary is
already handled robustly by the interior-point solver's defaults.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "degenerate"
TIMEOUT = 30
SOLVE_CALL = {"mode": "optimal"}


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
    """Construct the duplicated-terminal-BC problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 32)
    # Indices are [x, v, t, u]; t is pinned at both ends so the phase is a
    # genuinely fixed-duration (T=1) transfer, matching the brief's "t in
    # [0, 1]" setup (the collocation mesh's node "t" is itself a free
    # decision variable unless explicitly bounded, so an un-pinned Back
    # time lets the interior-point solver silently rescale the effective transfer duration).
    phase.add_boundary_value("Front", range(0, 3), [0.0, 0.0, 0.0])
    phase.add_boundary_value("Back", range(0, 3), [1.0, 0.0, 1.0])
    # Identical second copy of the terminal boundary condition: three exact
    # duplicate row pairs (x, v, t) in the constraint Jacobian, not a
    # conflict.
    phase.add_boundary_value("Back", range(0, 3), [1.0, 0.0, 1.0])
    phase.add_integral_objective((Args(1)[0] ** 2) / 2, [3])

    return phase
