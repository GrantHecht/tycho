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

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

"""
Source for problem formulation
https://www.hindawi.com/journals/aaa/2014/851720/
"""


class ODE(oc.ODEBase):
    def __init__(self):

        XVars = 1
        UVars = 1

        args = oc.ODEArguments(XVars, UVars)
        x = args.x_var(0)
        u = args.u_var(0)
        xdot = 0.5 * x + u
        super().__init__(xdot, XVars, UVars)

    class obj(vf.ScalarFunction):
        def __init__(self):
            args = Args(2)
            x = args[0]
            u = args[1]
            obj = u * u + x * u + 1.25 * x**2
            super().__init__(obj)


##############################################

if __name__ == "__main__":
    ode = ODE()

    x0 = 1.0
    t0 = 0.0

    tf = 1.0
    u0 = 0.00

    nsegs = 20

    TrajIG = [[x0, t, u0] for t in np.linspace(t0, tf, 100)]

    phase = ode.phase("LGL5", TrajIG, nsegs)
    phase.add_boundary_value("Front", [0, 1], [x0, t0])
    phase.add_boundary_value("Back", [1], [tf])
    phase.add_integral_objective(ODE.obj(), [0, 2])
    phase.optimize()

    Traj = phase.return_traj()
    CTraj = phase.return_costate_traj()

    ###########################################
    T = np.array(Traj).T
    CT = np.array(CTraj).T

    X = T[0]
    t = T[1]

    # Collocation control
    U = T[2]
    ## Collocation Costates
    L = CT[0]

    ### Analytic costates
    Lstar = 2 * np.cosh(1 - t) * np.tanh(1 - t) / np.cosh(1)
    ### Analytic control
    Ustar = -(np.tanh(1 - t) + 0.5) * np.cosh(1 - t) / np.cosh(1)

    plt.plot(t, L, label=r"$L$" + "-Collocation", marker="o")
    plt.plot(t, Lstar, label=r"$L$" + "-Analytic")

    plt.plot(t, U, label=r"$U$" + "-Collocation", marker="o")
    plt.plot(t, Ustar, label=r"$U$" + "-Analytic")
    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$L,U$")
    plt.show()
    #################################
