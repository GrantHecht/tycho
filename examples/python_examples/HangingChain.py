# =============================================================================
# Originally from ASSET (AlabamaASRL/asset_asrl)
# Copyright 2020-present The University of Alabama-Astrodynamics and Space
#   Research Lab. Licensed under the Apache License, Version 2.0
# License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# Original Developer: James B. Pezent
#
# Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
#   Apache 2.0 — see LICENSE.txt):
#   - Updated imports: import asset_asrl -> import tychopy
#   - Module usage updated to _tychopy (nanobind) bindings
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tychopy as typy

norm = np.linalg.norm
vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
solvs = typy.Solvers


class Chain(oc.ODEBase):
    def __init__(self):
        args = oc.ODEArguments(1, 1)
        super().__init__(args[2], 1, 1)


def Energy():
    x, u = Args(2).tolist()
    return x * vf.sqrt(1 + u**2)


def Length():
    (u,) = Args(1).tolist()
    return vf.sqrt(1 + u**2)


def GetIG(a, b, ts):
    IG = []

    for t in ts:
        if b > a:
            tm = 0.25
        else:
            tm = 0.75
        x = 2 * abs(b - a) * t * (t - 2 * tm) + a
        u = 2 * abs(b - a) * (t * 2.0 - 2 * tm)
        IG.append([x, t, u])

    return IG


def Job(a, b, n, L):

    ts = np.linspace(0, 1, n)
    IG = GetIG(a, b, ts)

    ode = Chain()

    phase = ode.phase("LGL5", IG, n)
    phase.set_static_params([L])

    phase.add_boundary_value("Front", [0, 1], [a, 0])
    phase.add_boundary_value("Back", [0, 1], [b, 1])
    phase.add_boundary_value("StaticParams", [0], [L])

    phase.add_upper_var_bound("Path", 0, max(a, b) + 0.001)

    phase.add_integral_objective(Energy(), [0, 2])
    phase.add_integral_param_function(Length(), [2], 0)

    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_max_ls_iters(2)
    phase.optimizer.print_level = 1
    phase.jet_job_mode = typy.Solvers.JetJobModes.SolveOptimize

    return phase


a = 1
b = 3
L = 4
n = 500


Ls = np.linspace(2.25, 8, 100)
cols = sns.color_palette("plasma", len(Ls))


JArgs = [(a, b, n, L) for L in Ls]

Res = solvs.Jet.map(Job, JArgs, True)

for i, res in enumerate(Res):
    Traj = res.return_traj()

    TT = np.array(Traj).T

    plt.plot(TT[1], TT[0], color=cols[i])


plt.grid(True)
plt.show()
