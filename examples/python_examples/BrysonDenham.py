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

import tychopy as typy

################################################################################
vf = typy.vector_functions
oc = typy.optimal_control
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
        x = args.x_vec()[0]
        v = args.x_vec()[1]
        u = args.u_vec()[0]
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
    phase.add_boundary_value("Front", range(0, 3), [0, 1, 0])
    phase.add_upper_var_bound("Path", 0, 1 / 9)
    phase.add_integral_objective((Args(1)[0] ** 2) / 2, [3])
    phase.add_boundary_value("Back", range(0, 3), [0, -1, 1])
    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_kkt_tol(1.0e-10)
    phase.optimizer.set_print_level(0)
    phase.set_num_partitions(1, 1)
    phase.optimize()

    Traj = phase.return_traj()

    #############################################################################

    TT = np.array(phase.return_traj()).T

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
