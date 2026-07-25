"""tests/corpus/problems/deg_near_infeasible.py — E2 G0 degenerate tier.

Baseline problem: a double-integrator rest-to-rest maneuver.

    states:   x, v
    control:  u
    dynamics: xdot = v, vdot = u
    t in [0, 1], LGL3 collocation, 32 segments

    boundary conditions: x(0) = 0, v(0) = 0, v(1) = 0, t(1) = 1
    bound:    |u| <= 1  (a "Path" upper/lower bound, not an equality)
    objective: minimize 0.5 * integral(u^2) dt

Note on time: the phase's node "t" (index 2 of [x, v, t, u]) is itself a
free decision variable in this collocation formulation, not automatically
tied to [0, 1] just because the initial guess spans that range, so t is
pinned explicitly at Front (t = 0) and Back (t = 1). This is essential
here specifically: the 0.25 reachability bound derived below assumes a
genuinely fixed T = 1 — an un-pinned Back time would let PSIOPT stretch
the effective transfer duration and make the target reachable after all,
which would defeat the whole point of this problem (verified empirically
that this is exactly what happens without the t(1) pin).

Perturbation: x(1) is pinned to a value that is marginally BEYOND the
maximum reachable displacement under |u| <= 1 in T=1 with the system at
rest at both ends. The problem is infeasible by construction; it probes
"near-infeasible" behavior — a target just barely outside the reachable
set, rather than the grossly-conflicting-constraint case exercised by
``deg_conflicting_equality``.

Reachability bound derivation (bang-bang, as specified in the Task 2
brief): with v(0) = v(1) = 0 and |u(t)| <= 1 on t in [0, 1], the control
that MAXIMIZES x(1) is bang-bang: full acceleration for the first half of
the interval, then full deceleration for the second half (by symmetry —
any other switching time either fails to return v to 0 at t=1, or reaches
a strictly smaller x(1); this is the textbook time-optimal/reach-optimal
double-integrator solution).

    u(t) = +1,  t in [0, 0.5)
    u(t) = -1,  t in [0.5, 1]

Phase 1 (t in [0, 0.5], v(0) = 0):
    v(t) = t
    x(t) = t^2 / 2
    => v(0.5) = 0.5,  x(0.5) = 0.125

Phase 2 (t in [0.5, 1], continuing from v(0.5) = 0.5, x(0.5) = 0.125):
    v(t) = 0.5 - (t - 0.5) = 1 - t
    => v(1) = 0                                            (checks out)
    x(t) = x(0.5) + integral_{0.5}^{t} v(tau) dtau
         = 0.125 + [tau - tau^2/2]_{0.5}^{t}
    => x(1) = 0.125 + (1 - 0.5) - (0.5 - 0.125) = 0.125 + 0.5 - 0.375 = 0.25

So the maximum reachable |x(1)| under this rest-to-rest, T=1, |u| <= 1
setup is exactly 0.25 — matching the Task 2 brief's stated bound.

Margin sizing (same HARD-LEARNED-LESSON principle applied to
``deg_conflicting_equality``'s conflict gap): the brief's concrete spec is
x(1) = 0.2505 (a 0.2% relative margin beyond the 0.25 bound). That exact
value was tried FIRST, via
``conda run -n tycho python scripts/run_corpus.py --filter deg_near_infeasible``.
Observed on defaults 2026-07-16 at margin 0.2505: NOTCONVERGED (harness
status "failed"), 500 iterations (hits max_iters), objective 0.4844 —
this lands as a genuine, unambiguous failure straight away (unlike
``deg_conflicting_equality``'s 1.001 gap, no widening was needed here; the
0.2% margin is apparently large enough, at this problem's scale/control-
bound structure, to clear PSIOPT's acceptable-equality-constraint
tolerance rather than falling inside it). X_TARGET is therefore left at
the brief's original spec value, 0.2505 — no adjustment was necessary.

Observed on defaults 2026-07-16 (final margin, 0.2505 - unchanged from
spec): NOTCONVERGED (harness status "failed"), 500 iterations, objective
0.4844.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "degenerate"
TIMEOUT = 30

# Exact bang-bang reachability bound derived in the module docstring.
_MAX_REACH = 0.25

# Brief spec value: 0.2505 = 0.25 * 1.002 (a 0.2% relative margin beyond
# the reachability bound). Adjusted here if empirically it lands in
# PSIOPT's acceptable-tolerance band rather than reading as a genuine
# failure — see docstring for the observed result and any adjustment.
_X_TARGET = 0.2505


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
    xs = np.linspace(0.0, _MAX_REACH, n)
    return [[x, 0.0, t, 0.0] for x, t in zip(xs, ts)]


def build_and_solve(configure) -> dict:
    """Construct, configure, and solve the near-infeasible-reach problem.

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _DoubleIntegrator()
    ig = _initial_guess()

    phase = ode.phase("LGL3", ig, 32)
    # Indices are [x, v, t, u]; t is pinned at both ends so the phase is a
    # genuinely fixed-duration (T=1) transfer — required for the 0.25
    # bang-bang reachability bound above to actually apply (an un-pinned
    # Back time is itself a free decision variable in this collocation
    # formulation and would let a longer effective duration make the
    # target reachable after all, defeating the whole point of this
    # problem).
    phase.add_boundary_value("Front", range(0, 3), [0.0, 0.0, 0.0])
    phase.add_lu_var_bound("Path", 3, -1.0, 1.0)
    phase.add_boundary_value("Back", range(0, 3), [_X_TARGET, 0.0, 1.0])
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
