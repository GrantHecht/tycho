"""tests/corpus/problems/deg_conflicting_equality.py — E2 G0 degenerate tier.

Baseline problem: a double-integrator rest-to-rest maneuver.

    states:   x, v
    control:  u
    dynamics: xdot = v, vdot = u
    t in [0, 1], LGL3 collocation, 32 segments

    boundary conditions: x(0) = 0, v(0) = 0, v(1) = 0, t(1) = 1
    objective: minimize 0.5 * integral(u^2) dt

Note on time: the phase's node "t" (index 2 of [x, v, t, u]) is itself a
free decision variable in this collocation formulation, not automatically
tied to [0, 1] just because the initial guess spans that range, so t is
pinned explicitly at Front (t = 0) and Back (t = 1) to match the brief's
stated "t in [0, 1]" fixed-duration setup.

Perturbation: the terminal state x(1) is constrained by TWO independent
``add_boundary_value`` calls with conflicting targets: x(1) = 1.0 and
x(1) = CONFLICT_VALUE. This is infeasible by construction (a scalar
variable cannot equal two different values) and probes whether the interior-point solver
honestly reports non-convergence rather than silently reporting an
"acceptable" solution that actually straddles the conflict.

Gap sizing (HARD-LEARNED LESSON from Task 1's ``stub_fails``, applied
here): the brief specifies CONFLICT_VALUE = 1.001 (a 1e-3 absolute gap).
That exact gap was tried FIRST, via a temporary edit run through
``conda run -n tycho python scripts/run_corpus.py --filter deg_conflicting``.
Observed on defaults 2026-07-16 at gap 1.001: ACCEPTABLE, 54 iterations,
objective 6.0235 — the residual from the conflicting pair lands inside
the interior-point solver's default acceptable-equality-constraint tolerance, exactly the
same failure-mode-that-isn't-a-failure documented for ``stub_fails`` in
tests/corpus/README.md (that module's own earlier revision used the same
1.001 gap and hit the same ACCEPTABLE band before being widened). The gap
was therefore widened to CONFLICT_VALUE = 2.0 (matching the widened
``stub_fails`` gap), empirically confirmed below to read as a genuine
failure, to keep this problem's pathology unambiguous. Both the original
spec value (1.001) and the adjustment (2.0) are recorded here per the
Task 2 brief's instructions.

Observed on defaults 2026-07-16 at gap 2.0: NOTCONVERGED (harness status
"failed"), 500 iterations (hits max_iters), objective 24.046. A genuine,
unambiguous failure.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "degenerate"
TIMEOUT = 30
SOLVE_MODE = "optimize"

# Original brief spec value was 1.001 (a 1e-3 gap against the x(1) = 1.0
# target); that fell inside the interior-point solver's acceptable-equality tolerance and read
# as ACCEPTABLE rather than a genuine failure (see docstring above), so the
# gap was widened here, mirroring the same adjustment Task 1 made to
# stub_fails for the identical reason.
_CONFLICT_VALUE = 2.0


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
    """Construct the conflicting-terminal-BC problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 32)
    # Indices are [x, v, t, u]; t is pinned at both ends so the phase is a
    # genuinely fixed-duration (T=1) transfer, matching the brief's "t in
    # [0, 1]" setup (an un-pinned Back time is itself a free decision
    # variable in this collocation formulation and lets the interior-point solver silently
    # rescale the effective transfer duration).
    phase.add_boundary_value("Front", range(0, 3), [0.0, 0.0, 0.0])
    phase.add_boundary_value("Back", [1, 2], [0.0, 1.0])
    # Two independent equality constraints on the same terminal state
    # (x(1)) with conflicting target values: infeasible by construction.
    phase.add_boundary_value("Back", [0], [1.0])
    phase.add_boundary_value("Back", [0], [_CONFLICT_VALUE])
    phase.add_integral_objective((Args(1)[0] ** 2) / 2, [3])

    return phase
