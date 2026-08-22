# =============================================================================
# ODE/setup adapted verbatim (modulo plotting) from
# examples/python_examples/VanDerPol.py, originally from ASSET
# (AlabamaASRL/asset_asrl). Copyright 2020-present The University of
# Alabama-Astrodynamics and Space Research Lab. Licensed under the Apache
# License, Version 2.0. License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# =============================================================================
"""tests/corpus/problems/hard_vanderpol.py — E2 G0 in-domain hard tier.

Parent example: ``examples/python_examples/VanDerPol.py`` (Van der Pol
oscillator minimum-energy control problem, taken from the Dymos example
gallery: https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html).

Perturbation: NONE — this module is the parent example verbatim, modulo
stripping the plotting block at the end. There is no synthetic perturbation
here because the parent example already fails on this toolchain out of the
box (a live, in-tree divergence), which is the single highest-value corpus
member the brief calls for: a real failure that needs no manufacturing.

Per ``project_vanderpol_diverges`` (recorded project memory): VanDerPol.py
diverges (KKT = nan at iteration 0) on the current clang22/MKL2026
fast-math toolchain and was dropped from the docs example gallery for this
reason. The divergence is TOOLCHAIN-DEPENDENT, not a property of the
problem itself or of the interior-point solver's algorithm in general — a different
clang/MKL/fast-math combination could plausibly not reproduce it. It is
tracked here specifically because it is a real, currently-reproducible
failure mode worth having permanent regression coverage for via the
corpus, regardless of root cause.

Observed on defaults 2026-07-16: DIVERGING (harness status "diverged"),
1 iteration, objective 0.0, KKT Inf = nan at iteration 0 — i.e. the very
first KKT residual evaluation is already NaN, so the interior-point solver's iteration loop
exits immediately. This is not a "strain" case in the iteration-count
sense (the strain rule does not apply — the failure is instantaneous by
construction, not something a slow convergence trace would show), it is
an outright, maximally fast, and perfectly deterministic failure. Verified
byte-identical (except wall_s) across repeats: flag, iterations (1), and
objective (0.0) are unchanged run to run.

No parent iteration count is meaningful here for the ≥2x strain comparison
(this problem never converges on this toolchain, defaults or otherwise;
the "parent" and the "variant" are the same code).
"""

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

TIER = "hard"
TIMEOUT = 60
SOLVE_MODE = "optimize"
NOTES = (
    "Historically diverged (KKT=nan at iteration 0) because the squared_norm "
    "derivative was 0/0 at the norm's centre and the all-zero initial guess sat "
    "exactly there; converges since that derivative was fixed. Kept in the hard "
    "tier as a regression sentinel for the centre-of-norm start."
)


class _VanderPol(oc.ODEBase):
    def __init__(self):
        args = oc.ODEArguments(2, 1)
        x0 = args[0]
        x1 = args[1]
        u = args[3]
        x0dot = (1.0 - x1 * x1) * x0 - x1 + u
        x1dot = x0
        ode = vf.stack(x0dot, x1dot)
        super().__init__(ode, 2, 1)


def build():
    """Construct the (verbatim) Van der Pol problem (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    ode = _VanderPol()
    tf = 10.0
    TrajIG = [[0, 0, t, 0] for t in np.linspace(0, tf, 100)]

    phase = ode.phase("LGL3", TrajIG, 256)
    phase.integrator.set_initial_step_size(0.25)
    phase.set_control_mode("BlockConstant")
    phase.add_boundary_value("Front", range(0, 3), [0, 1, 0])
    phase.add_lu_var_bound("Path", 3, -0.75, 1.0)
    phase.add_integral_objective(Args(3).squared_norm(), [0, 1, 3])
    phase.add_boundary_value("Back", [0, 1, 2], [0.0, 0.0, tf])
    phase.set_num_partitions(8)
    phase.optimizer.qp_threads = 8
    phase.optimizer.set_tols(1.0e-8, 1.0e-8, 1.0e-8)

    return phase
