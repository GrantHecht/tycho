# =============================================================================
# ODE/setup adapted from examples/python_examples/SimpleLowThrust.py,
# originally from ASSET (AlabamaASRL/asset_asrl). Copyright 2020-present
# The University of Alabama-Astrodynamics and Space Research Lab. Licensed
# under the Apache License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_lowthrust_badguess.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/SimpleLowThrust.py`` — a
circle-to-circle low-thrust orbit transfer. This module isolates just the
FIRST solve in the parent's three-solve sequence (time-optimal transfer,
before the power-optimal/mass-optimal continuation calls), which is the
self-contained, single-``optimize()`` scenario the corpus contract wants.
The ``LTModel`` ODE class, boundary values, control-norm path bound, and
optimizer knobs (``bound_fraction``, ``opt_ls_mode``, ``max_ls_iters``,
``delta_h``) are copied verbatim from the parent.

Perturbation: the parent's initial guess is an actual integrated spiral
trajectory (``ode.integrator(...).integrate_dense(...)`` propagating the
initial circular orbit under a fixed steering law for 6.4*pi time units).
This module instead DEGRADES the guess to the initial circular-orbit state
copied unchanged across every guess node (no integration, no spiral shape
at all) with only the time coordinate advancing linearly over the same
span — i.e. the interior-point solver is handed a "trajectory" that just sits at the starting
orbit forever, giving it no useful information about how to reach the
target orbit.

Observed on defaults 2026-07-16: parent (integrated-spiral guess):
CONVERGED, 26 iterations, objective ≈ 18.263 (this single time-optimal
solve; the brief's ~374-ish baseline refers to the parent's FULL
three-solve sequence including the power-optimal and mass-optimal
continuation solves, which this module does not replay — see contract
note above). Bad-guess (initial-orbit-copied guess): DIVERGING (harness
status "diverged"), 1 iteration, objective 20.106192982974676 — an
immediate, maximally fast failure (the interior-point solver's very first iteration already
diverges), which needs no iteration-count comparison to qualify as a
genuine failure under the brief's rule (the ≥2x threshold only binds for
cases that "still converge easily"; this one does not converge at all).
Verified byte-identical (except wall_s) across repeats — unlike the
parent SimpleLowThrust problem itself (flagged in the brief as one of the
corpus's three order-sensitive/noisy examples), THIS degraded-guess
variant is fully deterministic in practice: it fails on iteration 1,
before any of the numerically sensitive accumulated-solve-path behavior
that makes the parent's own multi-solve sequence noisy has a chance to
manifest.

Deviation from parent: ``phase.optimizer.set_print_level(1)`` omitted
(cosmetic; harness iteration instrument unaffected at print_level<2).
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "hard"
TIMEOUT = 120
SOLVE_MODE = "optimize"


class _LTModel(oc.ODEBase):
    def __init__(self, mu, ltacc):
        Xvars = 6
        Uvars = 3
        args = oc.ODEArguments(Xvars, Uvars)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3() * (-mu)
        thrust = u * ltacc
        acc = g + thrust
        ode = vf.stack([v, acc])
        super().__init__(ode, Xvars, Uvars)


def build():
    """Construct the bad-guess SimpleLowThrust problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    mu = 1
    acc = 0.02
    ode = _LTModel(mu, acc)

    r0 = 1.0
    v0 = np.sqrt(mu / r0)
    rf = 2.0
    vF = np.sqrt(mu / rf)

    X0 = np.zeros(7)
    X0[0] = r0
    X0[4] = v0

    Xf = np.zeros(6)
    Xf[0] = rf
    Xf[4] = vF

    # Perturbation: initial orbit state copied across all guess nodes
    # (no integrated spiral shape at all); t linear over the parent's span.
    tf = 6.4 * np.pi
    TrajIG = []
    for t in np.linspace(0, tf, 100):
        X = np.zeros(10)
        X[0:7] = X0
        X[6] = t
        TrajIG.append(X)

    phase = ode.phase("LGL3", TrajIG, 256)
    phase.add_boundary_value("Front", range(0, 7), X0)
    phase.add_lu_norm_bound("Path", [7, 8, 9], 0.001, 1, 1.0)
    phase.add_boundary_value("Back", range(0, 6), Xf[0:6])

    phase.optimizer.set_bound_fraction(0.995)
    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_max_ls_iters(2)
    phase.optimizer.set_delta_h(1.0e-6)
    phase.add_delta_time_objective(1.0)

    return phase
