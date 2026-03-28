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
from tychopy.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments


"""
Hyper-Sensitive Problem
Classic hypersensitive mesh refinement benchmark problem from Rao and company
https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.2114

See MeshRefinement Folder for more in depth version

"""


class HyperSens(oc.ODEBase):
    def __init__(self):
        ################################
        XtU = oc.ODEArguments(1, 1)
        x = XtU.x_var(0)
        u = XtU.u_var(0)
        xdot = -(x) + u
        ################################
        super().__init__(xdot, 1, 1)


if __name__ == "__main__":
    xt0 = 1.5
    xtf = 1.0
    tf = 10000.0

    ode = HyperSens()
    ## Lerp boundary conditions
    TrajIG = [
        [xt0 * (1 - t / tf) + xtf * (t / tf), t, 0] for t in np.linspace(0, tf, 1000)
    ]

    nsegs = 10
    phase = ode.phase("LGL7", TrajIG, nsegs)
    # Boundary Conditions
    phase.add_boundary_value("First", [0, 1], [xt0, 0])
    phase.add_boundary_value("Last", [0, 1], [xtf, tf])
    # Objective
    phase.add_integral_objective(Args(2).squared_norm() / 2, [0, 2])
    # Some loose bounds on variables
    phase.add_lu_var_bound("Path", 0, -50, 50)
    phase.add_lu_var_bound("Path", 2, -50, 50)
    # Enable line searches
    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_soe_ls_mode("L1")
    ## Neccessary for this problem
    phase.optimizer.set_qp_ordering_mode("MINDEG")
    phase.optimizer.print_level = 2
    phase.set_num_partitions(1, 1)

    # Enable Adaptive Mesh
    phase.set_adaptive_mesh(True)
    ## Set Error tolerance on mesh:
    phase.set_mesh_tol(1.0e-6)  # default = 1.0e-6
    ## Set Max number of mesh iterations:
    phase.set_max_mesh_iters(10)  # default = 10
    ## Make sure to set optimizer Econtol to be the same as or smaller than MeshTol
    phase.optimizer.set_eq_con_tol(1.0e-7)

    flag = phase.optimize_solve()  # Recommended to run with post solve enabled

    if phase.mesh_converged and flag == 0:
        print("Success")
    else:
        print("Failure")

    #######################################################
    TT = np.array(phase.return_traj()).T

    fig = plt.figure()

    ax0 = plt.subplot(211)
    ax1 = plt.subplot(223)
    ax2 = plt.subplot(224)

    axs = [ax0, ax1, ax2]

    for ax in axs:
        ax.grid(True)
        ax.plot(TT[1], TT[0], label="x", marker="o")
        ax.plot(TT[1], TT[2], label="u", marker="o")
        ax.set_xlabel("t")

    axs[0].legend()
    axs[1].set_xlim([-0.5, 12])
    axs[2].set_xlim([tf - 12, tf + 0.5])

    plt.show()
    ###############################################################
    PhaseMeshErrorPlot(phase, show=True)
