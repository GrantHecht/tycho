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
#   - Updated imports: import asset_asrl -> import tycho
#   - Module usage updated to _tycho (nanobind) bindings
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

import tycho as ast

################################################################################
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

###############################################################################

"""
Classic Bryson Denham problem, can find it anywhere
"""


class Model(oc.ODEBase):
    def __init__(self):
        Xvars = 2
        Uvars = 1
        ############################################################
        args = oc.ODEArguments(Xvars, Uvars)
        x = args.XVec()[0]
        v = args.XVec()[1]
        u = args.UVec()[0]
        ode = vf.stack([v, u])
        #############################################################
        super().__init__(ode, Xvars, Uvars)


if __name__ == "__main__":
    ###############################################################################
    n = 100
    ts = np.linspace(0, 1, n)
    vs = np.linspace(1, -1, n)

    IG = [[0.0, v, t, 0] for t, v in zip(ts, vs)]

    ode = Model()

    phase = ode.phase("LGL5", IG, 32)
    phase.addBoundaryValue("Front", range(0, 3), [0, 1, 0])
    phase.addUpperVarBound("Path", 0, 1 / 9)
    phase.addIntegralObjective((Args(1)[0] ** 2) / 2, [3])
    phase.addBoundaryValue("Back", range(0, 3), [0, -1, 1])
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_KKTtol(1.0e-10)
    phase.optimizer.set_PrintLevel(0)
    phase.setThreads(1, 1)
    phase.optimize()

    Traj = phase.returnTraj()

    #############################################################################

    TT = np.array(phase.returnTraj()).T

    fig, axs = plt.subplots(3, 1)

    axs[0].plot(TT[2], TT[0])
    axs[1].plot(TT[2], TT[1])
    axs[2].plot(TT[2], TT[3])

    axs[0].set_ylabel(r"$x$")
    axs[1].set_ylabel(r"$v$")
    axs[2].set_ylabel(r"$u$")
    axs[2].set_xlabel(r"$t$")

    for i in range(0, 3):
        axs[i].grid(True)
    plt.show()

    #############################################################################
