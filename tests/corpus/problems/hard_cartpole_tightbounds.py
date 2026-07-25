# =============================================================================
# ODE/setup adapted from examples/python_examples/CartPole.py, originally
# from ASSET (AlabamaASRL/asset_asrl). Copyright 2020-present The
# University of Alabama-Astrodynamics and Space Research Lab. Licensed
# under the Apache License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_cartpole_tightbounds.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/CartPole.py`` — Kelly (2017)'s
minimum-effort cart-pole swing-up (LGL5, 64 segments, ``F`` bounded in
``[-Fmax, Fmax]`` with the parent's ``Fmax = 20`` N). Solver setup
(dynamics via mass-matrix inversion, boundary values, x path bound,
squared-force integral objective, ``set_num_partitions(8, 8)``) is copied
verbatim from the parent, with only ``Fmax`` changed.

Perturbation: ``Fmax`` tightened toward the minimum feasible value, per
the brief's instruction to bisect manually and set the bound just inside
where defaults start straining. Bisection (probed manually, not part of
this module): ``Fmax=20`` (parent) converges in 14 iterations; ``Fmax=10``
-> 33; ``Fmax=8`` -> 93 (still converges); ``Fmax=7`` -> 95 (still
converges); ``Fmax=6.5`` -> DIVERGING at 289 iterations (the swing-up
becomes actuator-infeasible-adjacent right around here); ``Fmax=6`` and
below -> NOTCONVERGED, hits ``max_iters=500``. ``Fmax=7`` N was chosen: it
sits just inside the straining-but-still-converging regime (deliberately
short of the ``Fmax=6.5`` cliff into DIVERGING, to avoid picking an
unstable/flaky point right at the edge), giving reproducible, deep strain
while still returning a clean CONVERGED flag.

Observed on defaults 2026-07-16: parent (Fmax=20): CONVERGED, 14
iterations, objective ≈ 58.808. Tight-bounds variant (Fmax=7): CONVERGED,
95 iterations, objective ≈ 78.546 — 95 / 14 ≈ 6.8x the parent's iteration
count, comfortably clearing the brief's ≥2x strain rule while still
converging.

Determinism note: iteration count is exactly reproducible (95 every run
observed). The objective value showed LSB-level float noise across
repeats (e.g. 78.54562203020079 vs 78.545622030201 vs 78.54562203020076 —
differing only at the ~13th significant digit), consistent with the
parent's own ``set_num_partitions(8, 8)`` threaded evaluation order; this
is present in the unperturbed parent too, not introduced by the
perturbation. It is far smaller than the order-sensitivity seen in
SimpleLowThrust/MountainCar/Zermelo (which changes iteration counts and
objectives at a macroscopically visible level) and does not affect the
harness's reported ``iterations``/``flag``, only the least-significant
digits of ``objective``.

Deviation from parent: ``phase.optimizer.set_print_level(1)`` omitted
(cosmetic; harness iteration instrument unaffected at print_level<2).
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "hard"
TIMEOUT = 60
SOLVE_MODE = "optimize"

# Deviation from the parent's Fmax=20: see module docstring for the
# manual bisection that led to this value.
_FMAX = 7.0


class _CartPole(oc.ODEBase):
    def __init__(self, l, m1, m2, g):
        Xvars = 4
        Uvars = 1
        XtU = oc.ODEArguments(Xvars, Uvars)
        x, theta, xdot, thetadot = XtU.x_vec().tolist()
        F = XtU.u_var(0)
        Q = vf.stack([-g * vf.sin(theta), F + m2 * l * vf.sin(theta) * thetadot**2])
        Mvec_rm = vf.stack(vf.cos(theta), l, m1 + m2, m2 * l * vf.cos(theta))
        M = vf.RowMatrix(Mvec_rm, 2, 2)
        xddot_thetaddot = M.inverse() * Q
        ode = vf.stack([xdot, thetadot, xddot_thetaddot])
        super().__init__(ode, Xvars, Uvars)


def build():
    """Construct the tight-bounds CartPole problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    m1 = 1
    m2 = 0.3
    l = 0.5
    g = 9.81
    xmax = 2
    tf = 2
    xf = 1
    Fmax = _FMAX

    ts = np.linspace(0, tf, 100)
    IG = [[xf * t / tf, np.pi * t / tf, 0, 0, t, 0.00] for t in ts]

    ode = _CartPole(l, m1, m2, g)

    phase = ode.phase("LGL5", IG, 64)
    phase.add_boundary_value("First", range(0, 5), [0, 0, 0, 0, 0])
    phase.add_boundary_value("Last", range(0, 5), [xf, np.pi, 0, 0, tf])
    phase.add_lu_var_bound("Path", 5, -Fmax, Fmax)
    phase.add_lu_var_bound("Path", 0, -xmax, xmax)
    phase.add_integral_objective(Args(1)[0] ** 2, [5])

    phase.set_num_partitions(8, 8)

    return phase
