# =============================================================================
# ODE/setup adapted from examples/python_examples/MountainCar.py,
# originally from ASSET (AlabamaASRL/asset_asrl). Copyright 2020-present
# The University of Alabama-Astrodynamics and Space Research Lab. Licensed
# under the Apache License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_mountaincar_badguess.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/MountainCar.py`` (Dymos-style
mountain-car minimum-time escape problem, already iteration-heavy on
defaults per the Task 3 brief: ~124-131 iterations observed across runs on
this toolchain -- see the determinism note below, this range itself
reflects order-sensitivity, not measurement error). Solver setup (LGL3,
128 segments, boundary values, path bounds with their scale factors,
delta-time objective, ``opt_ls_mode="L1"``) is copied verbatim from the
parent.

Perturbation: the brief calls for "a zero-control, stationary initial
guess." The most literal reading -- park the car at its true initial
position x0 = -0.5, v = 0, u = 0, for every guess node, with only time
advancing linearly -- was tried FIRST. Observed: CONVERGED, 115 iterations
total (13 + 102 across the two ``solve_optimize`` stages) -- FEWER
iterations than the parent's own linear-interpolation guess (~118-128),
i.e. this literal reading does not strain the solver at all; if anything
it is a slightly easier starting point for PSIOPT than the parent's guess
(which drives a sinusoidal control profile that itself needs partially
undone). Per the brief's "otherwise perturb harder before committing"
escape hatch, the stationary point was moved to the FAR/target position
instead: x = xf = 0.52 (still v = 0, u = 0, constant across every guess
node -- still a bona fide "zero-control, stationary initial guess," just
parked at the wrong end of the domain instead of the right one). This
produces a genuine, unambiguous failure (see below).

Observed on defaults 2026-07-16 (final variant, stationary at xf):
NOTCONVERGED (harness status "failed"), 1000 iterations total (500 + 500,
hitting ``max_iters`` on both stages of ``solve_optimize``) -- roughly
8x the parent's ~124-131 baseline, and a genuine failure flag (not merely
a strain-but-converges case), so the brief's ≥2x rule is satisfied by a
wide margin regardless of which reading of "parent" iteration count is
used.

Determinism note — MountainCar is ORDER-SENSITIVE (one of the corpus's
noisy parents, alongside SimpleLowThrust and Zermelo; discovered
empirically while building this tier, generalizing the brief's
SimpleLowThrust caveat). This holds for the PARENT example itself, not
just this perturbation: repeated runs of the verbatim parent setup
(linear-interpolation guess) gave iteration counts of 118, 119, 121, 123,
124, 128 across probing runs, and this persisted even with
``phase.set_num_partitions(1)`` (single partition) and
``OMP_NUM_THREADS=1``/``MKL_NUM_THREADS=1`` forced in the environment --
i.e. the nondeterminism is not solely a threading artifact of this
problem's own defaults. For the stationary-at-xf failure variant used
here, the two-stage 500+500 iteration counts were reproduced across
several manual probing runs (hitting ``max_iters`` deterministically on
both stages), but the objective value at that point is NOT guaranteed
byte-identical run to run. This is documented here rather than fought, per
the brief's guidance for the (separately) noisy SimpleLowThrust parent.

Deviation from parent: ``phase.optimizer.set_print_level(1)`` omitted
(cosmetic; harness iteration instrument unaffected at print_level<2).
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control

TIER = "hard"
TIMEOUT = 120
SOLVE_MODE = "solve_optimize"
NOTES = (
    "Order-sensitive (also true of the unperturbed parent); "
    "iteration count at the two solve_optimize stages' max_iters cap "
    "is stable, objective is not guaranteed byte-identical. See "
    "module docstring."
)


class _MountainCar(oc.ODEBase):
    def __init__(self):
        args = oc.ODEArguments(2, 1)
        x = args.x_var(0)
        v = args.x_var(1)
        u = args.u_var(0)
        xdot = v
        vdot = 0.001 * u - 0.0025 * vf.cos(3 * x)
        ode = vf.stack([xdot, vdot])
        super().__init__(ode, 2, 1)


def build():
    """Construct the bad-guess MountainCar problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _MountainCar()

    x0 = -0.5
    v0 = 0
    xf = 0.52
    tf = 500

    # Perturbation: zero-control, stationary initial guess parked at the
    # FAR/target position (see docstring for why not x0).
    IG = [[xf, 0.0, t, 0.0] for t in np.linspace(0, tf, 100)]

    phase = ode.phase("LGL3", IG, 128)
    phase.add_boundary_value("First", [0, 1, 2], [x0, v0, 0])
    phase.add_boundary_value("Last", [0], [xf])
    phase.add_lower_var_bound("Back", 1, 0.0, 1.0)
    phase.add_lu_var_bound("Path", 0, -1.2, 0.55, 1.0)
    phase.add_lu_var_bound("Path", 1, -0.07, 0.07, 100.0)
    phase.add_lu_var_bound("Path", 3, -1, 1, 1.0)
    phase.add_delta_time_objective(0.01)

    phase.optimizer.set_opt_ls_mode("L1")

    return phase
