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

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


"""
Vanderpol Osscilator Optimization Problem Taken From 
https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
"""


class VanderPol(oc.ODEBase):
    def __init__(self):
        ############################################################
        args = oc.ODEArguments(2, 1)
        x0 = args[0]
        x1 = args[1]
        u = args[3]

        x0dot = (1.0 - x1 * x1) * x0 - x1 + u
        x1dot = x0
        ode = vf.stack(x0dot, x1dot)
        ##############################################################
        super().__init__(ode, 2, 1)


if __name__ == "__main__":
    ode = VanderPol()

    tf = 10.0

    TrajIG = [[0, 0, t, 0] for t in np.linspace(0, tf, 100)]

    integ = ode.integrator(0.001)

    phase = ode.phase("LGL3", TrajIG, 256)
    phase.integrator.setStepSizes(0.25, 0.001, 3)
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front", range(0, 3), [0, 1, 0])
    phase.addLUVarBound("Path", 3, -0.75, 1.0, 1.0)

    phase.addIntegralObjective(Args(3).squared_norm(), [0, 1, 3])
    phase.addBoundaryValue("Back", [0, 1, 2], [0.0, 0.0, tf])
    phase.optimizer.PrintLevel = 0
    phase.setNumPartitions(8, 8)
    phase.optimizer.set_tols(1.0e-8, 1.0e-8, 1.0e-8)

    phase.optimize()
    Traj = phase.returnTraj()
    T = np.array(Traj).T
    plt.plot(T[2], T[0], label=r"$x_0$")
    plt.plot(T[2], T[1], label=r"$x_1$")
    plt.plot(T[2], T[3], label=r"$u$")
    plt.grid(True)
    plt.legend()
    plt.xlabel(r"$t$")
    plt.show()
