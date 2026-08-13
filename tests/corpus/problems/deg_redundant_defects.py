"""tests/corpus/problems/deg_redundant_defects.py — E2 G0 degenerate tier.

Baseline problem: a double-integrator rest-to-rest maneuver, extended with
one inert "shadow" state that mirrors the velocity exactly.

    states:   x, v, w
    control:  u
    dynamics: xdot = v, vdot = u, wdot = u   (w integrates the SAME control
                                               input as v)
    t in [0, 1], LGL3 collocation, 32 segments

    boundary conditions: x(0) = 0, v(0) = 0, w(0) = 0, t(0) = 0,
                          x(1) = 1, v(1) = 0, t(1) = 1
                          (w is deliberately left UNconstrained at Back —
                          see the redundancy argument below)
    objective: minimize 0.5 * integral(u^2) dt

Note on time: the phase's node "t" (index 3 of [x, v, w, t, u]) is itself a
free decision variable in this collocation formulation, not automatically
tied to [0, 1] just because the initial guess spans that range, so t is
pinned explicitly at Front (t = 0) and Back (t = 1) to match the brief's
stated "t in [0, 1]" fixed-duration setup (an un-pinned Back time was
verified, in the sibling module ``deg_dup_equality``, to let the interior-point solver
silently rescale the effective transfer duration instead).

Perturbation, and why this replaces an earlier construction: an earlier
version of this module added the path constraint ``u - (6 - 12*t) = 0``,
i.e. this problem's own closed-form minimum-energy optimal control,
derived via Pontryagin's minimum principle from the *objective's*
stationarity condition. On review that was rejected as the wrong kind of
degeneracy: that row is satisfied by exactly ONE trajectory (the
optimum), not by every feasible trajectory, so it is not "structurally
redundant given the dynamics" — it is redundant only at the solution,
which made it (a) misleading as a "redundant defects" case and (b) a
near-duplicate of ``deg_dup_equality``'s exact-row-duplication test in
everything but name once the optimizer converged.

This version instead adds a genuinely dynamics-implied redundant row via
a shadow state. Extend the ODE with a third state w whose dynamics are
IDENTICAL to v's (wdot = u = vdot), and pin w(0) = v(0) = 0 at Front.
Then for ANY feasible trajectory (any u(t) satisfying the dynamics
defects and these initial conditions, not just the optimal one):

    d/dt (w - v) = wdot - vdot = u - u = 0   for every t,

so w(t) - v(t) is constant, and w(0) - v(0) = 0 fixes that constant at
zero: w(t) = v(t) identically, for every feasible control, everywhere on
the path. Concretely, in the NLP: the "Path" equality constraint

    w - v = 0

added at every collocation node is implied purely by (i) the two
defect-constraint blocks for v and w, which are literally the same
integration of the same control input, and (ii) the shared w(0) = v(0) =
0 initial condition — with no reference to the objective at all. This is
exactly the "redundant interior rows" pathology the brief asks for at
scale: rows that are never binding and never informative for the entire
feasible manifold, not just at one distinguished (optimal) point.

At 32 LGL3 segments the path region is evaluated at every node — LGL3 has
2 collocation sub-nodes per segment sharing segment endpoints, giving
2*32 + 1 = 65 nodes — so this adds 65 new equality rows, and (unlike the
old construction, where duplicating one already-meaningful row left the
Jacobian rank-deficient by one row per node but that one row was still
individually informative pre-duplication) ALL 65 of these rows are
individually redundant: each is a linear combination of rows already
present (defects for v and w, plus the Front IC block), contributing zero
new information to the KKT system at any feasible point.

Contrast with ``deg_dup_equality``: that module's degeneracy is a
byte-for-byte VERBATIM duplicate of a single existing row set — the same
``add_boundary_value("Back", ...)`` call issued twice, producing 3 exact
duplicate row pairs confined to the boundary. This module's degeneracy is
a single NEW, distinct constraint (w - v, referencing a variable that
appears nowhere else in the Jacobian) that is nonetheless analytically
implied by the rest of the problem (dynamics + initial conditions) at
every one of 65 interior/path nodes — a structurally different failure
mode (implied-by-dynamics row, not literal duplication) and two orders of
magnitude more redundant rows (65 vs. 3).

Expected solution: x, v, u, and the objective are unaffected by w, which
is entirely inert (it does not appear in the objective, the boundary
conditions besides its own Front pin, or any other constraint) — the
well-posed baseline solution (min 0.5 * integral(u^2), analytic optimum
6.0) should be recovered exactly as before.

Observed on defaults 2026-07-16: CONVERGED, 3 iterations, objective
6.01150 (matching the analytic 6.0 minimum-energy optimum to 3 sig
figs, confirming w's inertness and that the shadow-state redundant rows
did not change the achievable objective). This is a FINDING TO REPORT
PROMINENTLY, not hidden: like ``deg_dup_equality``, this dynamics-implied
redundant-row pathology does NOT genuinely manifest as non-convergence —
the interior-point solver's KKT factorization tolerates 65 structurally-redundant interior
rows (an exact rank deficiency of 65 in the constraint Jacobian, at every
LGL3 node across 32 segments) about as readily as the well-posed
baseline, converging in the same handful of iterations (3, matching both
the sibling boundary-duplication module and the well-posed baseline).
Useful negative result
for future E2 work: the interior-point solver's default Pardiso pivoting/regularization path
appears robust not only to verbatim duplicate rows (``deg_dup_equality``)
but also to a much larger block of rows that are merely *implied* by
other constraints rather than copied from them.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "degenerate"
TIMEOUT = 30
SOLVE_MODE = "optimize"


class _DoubleIntegrator(oc.ODEBase):
    def __init__(self):
        XVars = 3
        UVars = 1
        args = oc.ODEArguments(XVars, UVars)
        v = args.x_vec()[1]
        u = args.u_vec()[0]
        # wdot = u = vdot: w is a shadow state that integrates the exact
        # same control input as v, so w(t) = v(t) identically once w(0) =
        # v(0) is pinned (see module docstring for the full argument).
        ode = vf.stack([v, u, u])
        super().__init__(ode, XVars, UVars)


def _initial_guess(n=100):
    ts = np.linspace(0.0, 1.0, n)
    xs = np.linspace(0.0, 1.0, n)
    return [[x, 0.0, 0.0, t, 0.0] for x, t in zip(xs, ts)]


def build():
    """Construct the redundant-interior-row problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 32)
    # Indices are [x, v, w, t, u]; t is pinned at both ends so the phase is
    # a genuinely fixed-duration (T=1) transfer (the collocation mesh's
    # node "t" is itself a free decision variable unless explicitly
    # bounded). w(0) is pinned to 0 alongside x(0), v(0) — this is the
    # initial-condition half of the w(t) = v(t) redundancy argument above.
    # w is deliberately left unconstrained at Back: its terminal value is
    # implied by the dynamics, not asserted.
    phase.add_boundary_value("Front", range(0, 4), [0.0, 0.0, 0.0, 0.0])
    phase.add_boundary_value("Back", [0, 1, 3], [1.0, 0.0, 1.0])
    phase.add_integral_objective((Args(1)[0] ** 2) / 2, [4])

    # w - v == 0 at every node. Genuinely implied by the v/w defect blocks
    # plus the shared Front initial condition (see derivation above) — not
    # asserted as a duplicate of any existing row, and true for every
    # feasible trajectory, not just the optimal one.
    node_indices = [1, 2]  # [v, w]
    redundant_row = Args(2)[1] - Args(2)[0]  # w - v
    phase.add_equal_con("Path", redundant_row, node_indices)

    return phase
