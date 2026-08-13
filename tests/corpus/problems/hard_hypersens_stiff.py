# =============================================================================
# ODE/setup adapted from examples/python_examples/HyperSens.py, originally
# from ASSET (AlabamaASRL/asset_asrl). Copyright 2020-present The
# University of Alabama-Astrodynamics and Space Research Lab. Licensed
# under the Apache License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_hypersens_stiff.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/HyperSens.py`` — the classic
Rao/Betts hyper-sensitive mesh-refinement benchmark (xdot = -x + u, t in
[0, 10000], sharp boundary-layer transients near t=0 and t=tf). The
parent's dynamics, boundary values, objective, ``opt_ls_mode``,
``soe_ls_mode``, ``qp_ordering_mode="MINDEG"`` ("Necessary for this
problem" per the parent's own comment), path bounds, and adaptive-mesh
machinery (``set_adaptive_mesh``/``set_mesh_tol``/``set_max_mesh_iters``)
are all copied verbatim; only the starting mesh coarseness, the adaptive-
mesh on/off switch, and the equality/KKT tolerances are perturbed (see
below for why more than "just nsegs" was needed).

Deviation #1 (print_level): the parent sets ``phase.optimizer.print_level
= 2``, which per the corpus README's own "Iteration counting" section
SUPPRESSES the "Iterations : N" console line the harness's instrument
depends on (that line is only emitted when print_level < 2). Reproduced
here: running the parent verbatim through this harness's iteration-count
regex yields no matches at all -- this is exactly the "HyperSens
0-visible" case flagged in the Task 3 brief. This module sets
``print_level = 1`` instead (one level below the parent's), which is
otherwise cosmetic (console verbosity only, does not change convergence
behavior) and restores the instrument.

Deviation #2 (perturbation strategy): the brief calls for "a coarser mesh
than the example uses (boundary-layer stiffness under-resolved)." A
literal single-lever change -- shrinking the parent's starting ``nsegs``
from 10 down to 2, keeping the parent's adaptive-mesh refinement and its
``mesh_tol=1e-6``/``max_mesh_iters=10`` untouched -- was tried FIRST.
Observed: the adaptive-mesh loop simply refines its way out of the
coarser start (mesh_converged=True, CONVERGED, only 44 total iterations
vs the parent's own 30 at nsegs=10) -- barely 1.47x, well short of the
brief's ≥2x rule, and not really "under-resolved" at all: that is
precisely the point of adaptive refinement, which exists to rescue a
coarse starting mesh. To make the "boundary-layer stiffness under-
resolved" failure mode the brief names actually manifest, adaptive
refinement itself has to be constrained, not just the starting mesh:
this module additionally turns adaptive meshing OFF
(``set_adaptive_mesh(False)``, so ``nsegs=2`` is the mesh the interior-point solver actually
solves on, permanently) and tightens both ``eq_con_tol`` (from the
parent's 1e-7) and ``kkt_tol`` (the parent never sets ``kkt_tol``, so it
runs at the interior-point solver's default of 1e-6) down to 1e-12 (this second change was
also empirically necessary -- at the parent's own eq_con_tol=1e-7/
kkt_tol=1e-6 defaults, the 2-segment fixed mesh "converges" trivially in
3-4 iterations to a badly wrong objective, e.g. 969.8 vs the true ~1.669,
which is an interesting silent-wrong-answer finding in its own right, see
below, but is not a failure BY THE HARNESS'S OWN FLAG-BASED STATUS -- it
still reports CONVERGED -- and does not strain by iteration count either,
so it would not satisfy the brief's binding rule on its own).

Observed on defaults 2026-07-16: parent (nsegs=10, adaptive mesh on,
eq_con_tol=1e-7, kkt_tol at the interior-point solver's default 1e-6, print_level=1 for
instrumentation): CONVERGED, mesh converged, 30 total iterations (summed
across mesh-refinement rounds), objective ≈ 1.669 (matching the problem's
known solution).
Stiff variant (nsegs=2, adaptive mesh OFF, eq_con_tol/kkt_tol=1e-12):
ACCEPTABLE (harness status "acceptable" — not a clean CONVERGED), 103
total iterations (52 + 51 across the two internal interior-point solver calls issued by
``phase.optimize_solve()``) -- 103 / 30 ≈ 3.4x the parent, clearing the
≥2x rule -- AND ``phase.mesh_converged`` is False, AND the objective
(≈ 969.8) is wildly wrong relative to the true ≈ 1.669. This is recorded
in ``notes`` since the harness's flag-derived status does not capture the
mesh-convergence signal at all: a corpus consumer reading only
flag/status would see "acceptable" and could easily miss that the
underlying answer is badly wrong. Verified byte-identical (except wall_s)
across repeats.
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "hard"
TIMEOUT = 60
SOLVE_MODE = "optimize_solve"

# Deviation from the parent's nsegs=10 / adaptive-mesh-on / tol=1e-7 — see
# module docstring for why a single-lever nsegs change was insufficient.
_NSEGS = 2
_TOL = 1.0e-12


class _HyperSens(oc.ODEBase):
    def __init__(self):
        XtU = oc.ODEArguments(1, 1)
        x = XtU.x_var(0)
        u = XtU.u_var(0)
        xdot = -(x) + u
        super().__init__(xdot, 1, 1)


def build():
    """Construct the under-resolved HyperSens problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    xt0 = 1.5
    xtf = 1.0
    tf = 10000.0

    ode = _HyperSens()
    TrajIG = [
        [xt0 * (1 - t / tf) + xtf * (t / tf), t, 0] for t in np.linspace(0, tf, 1000)
    ]

    phase = ode.phase("LGL7", TrajIG, _NSEGS)
    phase.add_boundary_value("First", [0, 1], [xt0, 0])
    phase.add_boundary_value("Last", [0, 1], [xtf, tf])
    phase.add_integral_objective(Args(2).squared_norm() / 2, [0, 2])
    phase.add_lu_var_bound("Path", 0, -50, 50)
    phase.add_lu_var_bound("Path", 2, -50, 50)
    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_soe_ls_mode("L1")
    phase.optimizer.set_qp_ordering_mode("MINDEG")
    # One level below the parent's print_level=2 -- see docstring Deviation #1.
    phase.optimizer.print_level = 1
    phase.set_num_partitions(1, 1)

    # Perturbation: adaptive mesh refinement disabled, so nsegs=2 is the
    # permanent (under-resolved) mesh -- see docstring Deviation #2.
    phase.set_adaptive_mesh(False)
    phase.optimizer.set_eq_con_tol(_TOL)
    phase.optimizer.set_kkt_tol(_TOL)

    return phase


def POST_SOLVE(prob) -> str:
    """Post-solve notes: flag alone hides the under-resolved-mesh outcome.

    See tests/corpus/README.md for the full problem-module contract and
    module docstring above for why this check is needed.
    """
    if bool(prob.mesh_converged):
        return ""
    return (
        "mesh_converged=False despite a non-DIVERGING flag: the "
        "2-segment fixed mesh under-resolves the boundary layer, so "
        "the reported objective is far from the true converged value "
        "(~1.669) even though the flag alone would not show this."
    )
