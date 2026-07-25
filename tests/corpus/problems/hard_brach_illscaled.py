# =============================================================================
# ODE/setup adapted from examples/python_examples/Brachistochrone.py,
# originally from ASSET (AlabamaASRL/asset_asrl). Copyright 2020-present
# The University of Alabama-Astrodynamics and Space Research Lab. Licensed
# under the Apache License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_brach_illscaled.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/Brachistochrone.py``. Solver
setup (LGL3, 32 segments, boundary/path constraints, linear-interpolation
initial guess, delta-time objective) is copied verbatim from the parent.

Perturbation: all lengths (x0, y0, xf, yf, and the guess's x/y nodes) are
scaled down by a factor ``s``, and gravity ``g`` is scaled by the same
factor ``s``, so the NLP is expressed in a badly-scaled unit system while
describing the *same physical curve*. This is consistent scaling, not a
different problem: with x' = s*x, y' = s*y, g' = s*g (time and theta
untouched), the dynamics ``xdot = sin(theta)*v``, ``ydot =
-cos(theta)*v``, ``vdot = g*cos(theta)`` are invariant under this
substitution (v' = s*v is implied), so the optimal transfer TIME is
unchanged — only the position/velocity/gravity numbers shrink, becoming
numerically tiny relative to the theta path bound and the O(1) time
variable, which is exactly the "badly scaled NLP, same physical solution"
perturbation the brief calls for.

Deviation from the brief's literal spec: the brief suggests ``s = 1e-6``
(lengths "1e6× smaller"). That value was tried FIRST. Observed at s=1e-6:
CONVERGED, 11 iterations vs the parent's 7 — only a 1.57x strain, short of
the brief's binding ≥2x rule, and still a clean, easy convergence. Per the
brief's own escape hatch ("otherwise perturb harder before committing"),
the scale was pushed further. The transition turns out to be a sharp
cliff: s=1e-7 already fails outright (NOTCONVERGED, hits max_iters=500),
while s=1e-6 converges in 11. s=1e-7 was adopted as the final scale
because it produces an unambiguous, deterministic failure rather than a
borderline strain.

Observed on defaults 2026-07-16 (final scale, s=1e-7): NOTCONVERGED
(harness status "failed"), 500 iterations (hits max_iters), objective
1.4059823628797314 — far from the true value (~1.8013), confirming this
is a genuine failure to converge, not a lucky wrong-but-plausible
objective. Verified byte-identical (except wall_s) across repeats.

For reference, s=1e-8 through s=1e-12 were also probed and remain
NOTCONVERGED/DIVERGING (never recovering), so s=1e-7 is the mildest scale
in the failing regime, kept for minimal deviation from the brief's spirit.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control

TIER = "hard"
TIMEOUT = 60
SOLVE_MODE = "optimize"

# Deviation from the brief's literal 1e-6: see module docstring — 1e-6 only
# strains 1.57x (short of the ≥2x rule) while 1e-7 already fails cleanly.
_SCALE = 1e-7


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


def build():
    """Construct the ill-scaled Brachistochrone problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    s = _SCALE
    g = 9.81 * s
    ode = _Brachistochrone(g)

    x0, y0, v0, theta0 = 0.0 * s, 10.0 * s, 0.0 * s, 1.0
    xf, yf = 10.0 * s, 5.0 * s
    tf = 1

    ts = np.linspace(0, tf, 100)

    Xs = []
    for t in ts:
        X = np.zeros(5)
        X[0] = x0 + (xf - x0) * t / tf
        X[1] = y0 + (yf - y0) * t / tf
        X[2] = g * t * np.cos(theta0)
        X[3] = t
        X[4] = theta0
        Xs.append(X)

    phase = ode.phase("LGL3", Xs, 32)
    phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
    phase.add_lu_var_bound("Path", 4, -0.1, 2.00)
    phase.add_boundary_value("Back", [0, 1], [xf, yf])
    phase.add_delta_time_objective(1.0)

    return phase
