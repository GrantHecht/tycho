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

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments
ODEArgs = oc.ODEArguments

"""
Classic Brachistochrone problem, can find it anywhere
"""


class Brachistochrone(oc.ODEBase):
    def __init__(self, g):

        XVars = 3
        UVars = 1

        XtU = oc.ODEArguments(XVars, UVars)

        x, y, v = XtU.x_vec().tolist()
        theta = XtU.u_var(0)

        xdot = vf.sin(theta) * v
        ydot = -1.0 * vf.cos(theta) * v
        zdot = g * vf.cos(theta)
        ode = vf.stack([xdot, ydot, zdot])

        super().__init__(ode, XVars, UVars)


if __name__ == "__main__":
    g = 9.81
    ode = Brachistochrone(g)

    x0 = 0
    y0 = 10
    v0 = 0
    theta0 = 1.0

    xf = 10
    yf = 5

    tf = 1

    ts = np.linspace(0, tf, 100)

    Xs = []
    for t in ts:
        X = np.zeros((5))
        X[0] = x0 + (xf - x0) * t / tf
        X[1] = y0 + (yf - y0) * t / tf
        X[2] = g * t * np.cos(theta0)
        X[3] = t
        X[4] = theta0
        Xs.append(X)

    phase = ode.phase("LGL3", Xs, 32)
    phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
    phase.add_lu_var_bound("Path", 4, -0.1, 2.00)
    phase.add_boundary_value("Back", [0, 1], [xf, yf])
    phase.add_delta_time_objective(1.0)
    phase.optimize()

    Traj = phase.return_traj()

    TT = np.array(Traj).T

    plt.plot(TT[0], TT[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    plt.plot(TT[3], TT[4])
    plt.xlabel("t")
    plt.ylabel(r"$\theta$")
    plt.grid(True)
    plt.show()
