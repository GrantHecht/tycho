"""tests/corpus/problems/deg_redundant_defects.py — E2 G0 degenerate tier.

Baseline problem: a double-integrator rest-to-rest maneuver.

    states:   x, v
    control:  u
    dynamics: xdot = v, vdot = u
    t in [0, 1], LGL3 collocation, 32 segments

    boundary conditions: x(0) = 0, v(0) = 0, x(1) = 1, v(1) = 0, t(1) = 1
    objective: minimize 0.5 * integral(u^2) dt

Note on time: the phase's node "t" (index 2 of [x, v, t, u]) is itself a
free decision variable in this collocation formulation, not automatically
tied to [0, 1] just because the initial guess spans that range, so t is
pinned explicitly at Front (t = 0) and Back (t = 1) to match the brief's
stated "t in [0, 1]" fixed-duration setup — this matters a great deal
here specifically, since the u*(t) = 6 - 12t derivation below assumes a
genuinely fixed T = 1; verified empirically that an un-pinned Back time
lets the objective collapse toward 0 by silently stretching the effective
transfer duration, invalidating the derivation.

Perturbation: an algebraic "Path" equality constraint (applied at every
collocation node, not just a boundary) is added TWICE, identically,
producing exact duplicate Jacobian rows *interior* to the problem rather
than at a single boundary node (contrast with ``deg_dup_equality``, whose
duplication is a single boundary-node row pair).

Why this particular path constraint (not the trivial ``v - v = 0``, which
the Task 2 brief explicitly rules out because it is an all-zero Jacobian
row, a different and uninteresting degeneracy): the constraint pins the
control exactly to this problem's own closed-form minimum-energy optimal
profile, which is genuinely a *linear* function of (u, t) — i.e. a linear
combination of quantities already fully determined by the linear dynamics
defects plus boundary conditions plus the quadratic objective's first-order
optimality condition. Deriving it (Pontryagin minimum principle):

    H = u^2/2 + lambda_x * v + lambda_v * u
    dH/du = 0        => u = -lambda_v
    lambda_x' = -dH/dx = 0             => lambda_x(t) = c1            (const)
    lambda_v' = -dH/dv = -lambda_x     => lambda_v(t) = c2 - c1*t
    u(t) = -lambda_v(t) = c1*t - c2

Integrating twice with x(0) = v(0) = 0:
    v(t) = c1*t^2/2 - c2*t
    x(t) = c1*t^3/6 - c2*t^2/2

Applying v(1) = 0 and x(1) = 1:
    v(1) = 0:  c1/2 - c2 = 0           => c2 = c1/2
    x(1) = 1:  c1/6 - c2/2 = 1         => c1*(1/6 - 1/4) = 1 => c1 = -12
    => c2 = -6

So the (unique) unconstrained optimum's control is exactly

    u*(t) = c1*t - c2 = -12*t + 6 = 6 - 12*t

Since this is an affine (degree-1) function of t, it is exactly
representable regardless of the polynomial degree LGL3 uses to interpolate
u within a segment — pinning u(t_k) = 6 - 12*t_k at every node via a "Path"
equality constraint does not change the feasible optimum (it is already
satisfied by the true solution) or the achievable objective (analytically,
integral(u^2)/2 dt = integral_0^1 (6-12t)^2/2 dt = 6.0). It purely adds one
new, exactly-linear, genuinely-implied-by-the-rest-of-the-problem equality
row per node. Adding the SAME ``add_equal_con`` call a second time,
verbatim, then produces an exact duplicate of that new row at every one of
the ~65 LGL3 nodes (32 segments): the constraint Jacobian is exactly
rank-deficient by one row per node, entirely interior to the path (not at
a single boundary node as in ``deg_dup_equality``).

Observed on defaults 2026-07-16: CONVERGED, 3 iterations, objective
6.01172 (matching the analytic 6.0 optimum, confirming the derivation
above is correctly implemented). This is a FINDING TO REPORT
PROMINENTLY, not hidden: like ``deg_dup_equality``, this exact-duplicate-
row pathology does NOT genuinely manifest as non-convergence — PSIOPT's
KKT factorization tolerates the interior rank deficiency (one duplicate
row per one of the ~65 LGL3 nodes across 32 segments) just as readily as
the single-boundary-row duplication in ``deg_dup_equality``, converging
in the same handful of iterations as the well-posed baseline. Taken
together, both duplicate-row degenerate modules suggest PSIOPT's default
Pardiso pivoting/regularization path is already robust to plain exact
row duplication, whether at the boundary or replicated across the entire
interior path.
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


def _initial_guess(n=100):
    ts = np.linspace(0.0, 1.0, n)
    xs = np.linspace(0.0, 1.0, n)
    return [[x, 0.0, t, 0.0] for x, t in zip(xs, ts)]


def build_and_solve(configure) -> dict:
    """Construct, configure, and solve the redundant-interior-row problem.

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 32)
    # Indices are [x, v, t, u]; t is pinned at both ends so the phase is a
    # genuinely fixed-duration (T=1) transfer — required for the u*(t) =
    # 6 - 12t closed-form derivation above to actually apply (an un-pinned
    # Back time is itself a free decision variable in this collocation
    # formulation and lets PSIOPT silently rescale the effective transfer
    # duration, which was verified empirically to break the derivation).
    phase.add_boundary_value("Front", range(0, 3), [0.0, 0.0, 0.0])
    phase.add_boundary_value("Back", range(0, 3), [1.0, 0.0, 1.0])
    phase.add_integral_objective((Args(1)[0] ** 2) / 2, [3])

    # u - (6 - 12*t) == 0, i.e. u == closed-form min-energy optimal
    # control. Genuinely implied by the rest of the problem (see
    # derivation above) — added, then duplicated verbatim, to force an
    # exact duplicate Jacobian row at every node interior to the path.
    node_indices = [0, 1, 2, 3]  # [x, v, t, u]
    redundant_row = Args(4)[3] - 6.0 + 12.0 * Args(4)[2]
    phase.add_equal_con("Path", redundant_row, node_indices)
    phase.add_equal_con("Path", redundant_row, node_indices)

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
