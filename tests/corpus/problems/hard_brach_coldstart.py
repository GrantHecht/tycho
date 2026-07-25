# =============================================================================
# ODE/setup adapted from examples/python_examples/Brachistochrone.py,
# originally from ASSET (AlabamaASRL/asset_asrl). Copyright 2020-present
# The University of Alabama-Astrodynamics and Space Research Lab. Licensed
# under the Apache License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_brach_coldstart.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/Brachistochrone.py`` — the
classic Brachistochrone problem (minimum-time slide between two points
under gravity). Solver setup (LGL3, 32 segments, boundary values, theta
path bound, delta-time objective) is copied verbatim from the parent.

Perturbation: the initial guess is replaced by a constant ("all-midpoint")
trajectory instead of the parent's linear interpolation between the front
and back boundary states. Concretely: x and y are held at the midpoint of
their respective boundary values for every guess node (instead of
interpolating linearly from (x0, y0) to (xf, yf)), v is held at 0 for every
node (instead of the parent's v(t) = g*t*cos(theta0) guess consistent with
constant-theta free-fall), and theta is still held at the parent's
theta0 = 1.0 constant. Time nodes still span [0, tf] linearly, matching the
parent (the brief says to keep the parent's own time handling — only the
state/control initial guess is degraded).

Observed on defaults 2026-07-16: parent (linear-interpolation guess):
CONVERGED, 7 iterations, objective 1.8012954671855554. Coldstart (constant-
midpoint guess): CONVERGED, 24 iterations, objective 1.8012954658420313 —
same physical solution (objective matches to 9 significant figures), but
24 / 7 ≈ 3.4x the parent's iteration count, satisfying the brief's ≥2x
strain rule while still converging cleanly. Verified byte-identical
(except wall_s) across repeats.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control

TIER = "hard"
TIMEOUT = 60


class _Brachistochrone(oc.ODEBase):
    def __init__(self, g):
        XVars = 3
        UVars = 1
        XtU = oc.ODEArguments(XVars, UVars)
        x, y, v = XtU.x_vec().tolist()
        theta = XtU.u_var(0)
        xdot = vf.sin(theta) * v
        ydot = -1.0 * vf.cos(theta) * v
        zdot = g * vf.cos(theta)
        ode = vf.stack([xdot, ydot, zdot])
        super().__init__(ode, XVars, UVars)


def build_and_solve(configure) -> dict:
    """Construct, configure, and solve the cold-started Brachistochrone problem.

    See tests/corpus/README.md for the full problem-module contract.
    """
    g = 9.81
    ode = _Brachistochrone(g)

    x0, y0, v0, theta0 = 0, 10, 0, 1.0
    xf, yf = 10, 5
    tf = 1

    ts = np.linspace(0, tf, 100)

    # Perturbation: constant ("all-midpoint") state/control guess instead of
    # the parent's linear interpolation.
    xm = 0.5 * (x0 + xf)
    ym = 0.5 * (y0 + yf)
    Xs = []
    for t in ts:
        X = np.zeros(5)
        X[0] = xm
        X[1] = ym
        X[2] = 0.0
        X[3] = t
        X[4] = theta0
        Xs.append(X)

    phase = ode.phase("LGL3", Xs, 32)
    phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
    phase.add_lu_var_bound("Path", 4, -0.1, 2.00)
    phase.add_boundary_value("Back", [0, 1], [xf, yf])
    phase.add_delta_time_objective(1.0)

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
