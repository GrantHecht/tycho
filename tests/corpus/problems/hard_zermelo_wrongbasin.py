# =============================================================================
# ODE/setup adapted from examples/python_examples/Zermelo.py, originally
# from ASSET (AlabamaASRL/asset_asrl). Copyright 2020-present The
# University of Alabama-Astrodynamics and Space Research Lab. Licensed
# under the Apache License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_zermelo_wrongbasin.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/Zermelo.py`` — Zermelo's
navigation problem (minimum-time steering under a wind field), solved via
``navigate(A, B, vM, wF)``. The parent example's ``main()`` drives many
``navigate()`` calls across two comparison sweeps (``compareWind()``,
``compareSpeed()``); this module is SELF-CONTAINED and isolates exactly
ONE representative ``navigate``-equivalent call — the parent's
``navigate(A, B, vM=vM, wF=lambda xyt: uniformWind(xyt, vel=0.5))`` case
from ``compareWind()``, with ``A = [0, -1]``, ``B = [1, 1]``,
``vM = 1.25`` — rather than replaying the full multi-solve sweep (which
would be neither self-contained in spirit nor fast). The ODE class, the
``uniformWind`` wind function, the straight-line-initial-guess
construction, ``num_partitions``, path bound, objective, and optimizer
tolerances (``eq_con_tol``/``kkt_tol`` = 1e-12) are all copied verbatim
from the parent's ``Zermelo`` class and ``navigate()`` function.

Perturbation: the initial guess heading is rotated 180° (``ang + pi``)
relative to the parent's straight-line-towards-B heading, while the
straight-line position guess itself (A -> B) is left unperturbed — i.e.
the boat's guessed position path still runs from A to B, but every guessed
heading points the boat backwards away from B at every node.

Observed on defaults 2026-07-16: parent (straight-line heading):
CONVERGED, 12 iterations, objective ≈ 1.701. Wrong-basin (heading rotated
180°): DIVERGING (harness status "diverged"), iterations in the 800-830
range across repeated runs (≈ 67-69x the parent's 12) — an unambiguous,
large-margin failure, satisfying the brief's rule by a wide margin even
accounting for run-to-run noise (see below).

Determinism note — Zermelo is ORDER-SENSITIVE (one of the corpus's
order-sensitive parents, alongside SimpleLowThrust and MountainCar; the
brief's own SimpleLowThrust caveat generalizes to these two as well,
discovered empirically while building this tier): repeated runs of this
exact wrong-basin variant were NOT byte-identical — iteration counts
observed across probing were 799, 805, 810, 813, 831 and the objective
varied at the ~1e-3 relative level, even with ``phase.num_partitions``
forced down to 1 (single-threaded). The DIVERGING trajectory appears to be
genuinely numerically chaotic near the point of failure (tiny floating-
point differences in constraint/Jacobian evaluation order compound before
PSIOPT gives up), not an artifact of parallelism alone. The flag
(DIVERGING) and the overall magnitude of the failure are stable; the exact
iteration count and objective are not. This is documented here rather than
fought, per the brief's guidance for the (separately) noisy
SimpleLowThrust parent.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control

tModes = oc.TranscriptionModes

TIER = "hard"
TIMEOUT = 90

_NSEG = 250
_TOL = 1e-12


class _Zermelo(oc.ODEBase):
    def __init__(self, vMax, wFunc):
        Xvars = 2
        Uvars = 1
        args = vf.Arguments(Xvars + 1 + Uvars)
        xyt = args.head_3()
        th = args[3]
        wx, wy = wFunc(xyt)
        xD = vMax * vf.cos(th) + wx
        yD = vMax * vf.sin(th) + wy
        ode = vf.stack([xD, yD])
        super().__init__(ode, Xvars, Uvars)


def _uniform_wind(xyt, ang=135 * np.pi / 180, vel=2):
    return vel * np.cos(ang), vel * np.sin(ang)


def build_and_solve(configure) -> dict:
    """Construct, configure, and solve the wrong-basin-heading Zermelo problem.

    See tests/corpus/README.md for the full problem-module contract.
    """
    A = np.array([0.0, -1.0])
    B = np.array([1.0, 1.0])
    vM = 1.25

    def wF(xyt):
        return _uniform_wind(xyt, vel=0.5)

    dist = np.linalg.norm(B - A)
    t0 = dist / vM
    d = (B - A) / dist
    # Perturbation: heading guess rotated 180 degrees away from B.
    ang = np.arctan2(d[1], d[0]) + np.pi

    trajG = [
        np.array(list(A + d * x) + [t0 * x, ang]) for x in np.linspace(0, 1, num=_NSEG)
    ]

    phase = _Zermelo(vM, wF).phase(tModes.LGL3)
    phase.num_partitions = 10
    phase.set_traj(trajG, _NSEG)

    phase.add_boundary_value("Front", [0, 1], A)
    phase.add_boundary_value("Front", [2], [0.0])
    phase.add_boundary_value("Back", [0, 1], B)
    phase.add_lu_var_bound("Path", 3, -np.pi, np.pi, 1)
    phase.add_delta_time_objective(1.0)

    phase.optimizer.set_eq_con_tol(_TOL)
    phase.optimizer.set_kkt_tol(_TOL)

    configure(phase.optimizer)
    flag = phase.solve_optimize()

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
        "notes": "Order-sensitive: iteration count and objective vary "
        "run-to-run near this DIVERGING failure; flag and failure "
        "magnitude are stable. See module docstring.",
    }
