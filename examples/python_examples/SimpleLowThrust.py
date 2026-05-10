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
sol = typy.solvers

Args = vf.Arguments


def Plot(Traj, name, col, ax=plt, linestyle="-"):
    TT = np.array(Traj).T
    ax.plot(TT[0], TT[1], label=name, color=col, linestyle=linestyle)


def Scatter(State, name, col, ax=plt):
    ax.scatter(State[0], State[1], label=name, c=col)


def ThrottlePlot(Traj, name, col, ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[6], (TT[7] ** 2 + TT[8] ** 2 + TT[9] ** 2) ** 0.5, label=name, color=col)


def ThrottlePlot2(Traj, name, col, ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[6], TT[7], label=name + " x", color=col)
    ax.plot(TT[6], TT[8], label=name + " y", color=col)
    ax.plot(TT[6], TT[9], label=name + " z", color=col)
    ax.plot(
        TT[6],
        (TT[7] ** 2 + TT[8] ** 2 + TT[9] ** 2) ** 0.5,
        label=name + " |mag|",
        color=col,
    )


class LTModel(oc.ODEBase):
    def __init__(self, mu, ltacc):
        Xvars = 6
        Uvars = 3
        ############################################################
        args = oc.ODEArguments(Xvars, Uvars)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3() * (-mu)
        thrust = u * ltacc
        acc = g + thrust
        ode = vf.stack([v, acc])
        #############################################################
        super().__init__(ode, Xvars, Uvars)

    class massobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(3)
            super().__init__(u.norm() * scale)

    class powerobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(3)
            super().__init__(u.norm().squared() * scale)


if __name__ == "__main__":
    mu = 1
    acc = 0.02
    ode = LTModel(mu, acc)

    r0 = 1.0
    v0 = np.sqrt(mu / r0)

    rf = 2.0
    vF = np.sqrt(mu / rf)

    X0 = np.zeros((7))
    X0[0] = r0
    X0[4] = v0

    Xf = np.zeros((6))
    Xf[0] = rf
    Xf[4] = vF

    XIG = np.zeros((10))
    XIG[0:7] = X0

    integ = ode.integrator(0.01, Args(3).normalized() * 0.8, [3, 4, 5])
    TrajIG = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

    phase = ode.phase("LGL3", TrajIG, 256)
    phase.add_boundary_value("Front", range(0, 7), X0)
    phase.add_lu_norm_bound("Path", [7, 8, 9], 0.001, 1, 1.0)
    phase.add_boundary_value("Back", range(0, 6), Xf[0:6])

    phase.optimizer.set_print_level(1)
    phase.optimizer.set_bound_fraction(0.995)
    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_max_ls_iters(2)
    phase.optimizer.set_delta_h(1.0e-6)
    ########################################
    phase.add_delta_time_objective(1.0)

    phase.optimize()
    phase.optimizer.max_ls_iters = 0

    TimeOptimal = phase.return_traj()
    TimeCostates = phase.return_costate_traj()

    phase.remove_state_objective(-1)
    ########################################
    phase.add_integral_objective(LTModel.powerobj(0.5), [7, 8, 9])
    phase.optimize()

    PowerOptimal = phase.return_traj()
    PowerCostates = phase.return_costate_traj()
    phase.remove_integral_objective(-1)
    #######################################

    phase.add_integral_objective(LTModel.massobj(1.0), [7, 8, 9])
    phase.optimize()

    MassOptimal = phase.return_traj()
    MassCostates = phase.return_costate_traj()

    ###########################################
    Plot(TrajIG, "Initial Guess", "blue")
    Scatter(X0, "X0", "black")
    Scatter(Xf, "XF", "red")
    plt.grid(True)
    plt.axis("Equal")
    plt.show()
    ###########################################

    ###########################################
    Plot(TimeOptimal, "TimeOptimal", "blue")
    Plot(MassOptimal, "MassOptimal", "green")
    Plot(PowerOptimal, "PowerOptimal", "red")
    plt.legend()
    Scatter(X0, "X0", "black")
    Scatter(Xf, "XF", "gold")
    plt.grid(True)
    plt.axis("Equal")
    plt.show()

    ThrottlePlot(TimeOptimal, "TimeOptimal", "blue")
    ThrottlePlot(MassOptimal, "MassOptimal", "green")
    ThrottlePlot(PowerOptimal, "MassOptimal", "red")
    plt.grid(True)
    plt.show()
    ###########################################

    MT = np.array(MassOptimal).T
    CT = np.array(MassCostates).T
    U = (MT[7] ** 2 + MT[8] ** 2 + MT[9] ** 2) ** 0.5

    plt.plot(MT[6], MT[7] / U, color="red", label="Ux(t)")
    plt.plot(MT[6], MT[8] / U, color="blue", label="Uy(t)")

    # Controls should be alligned oppiste the normalized velocity costates
    N = (CT[3] ** 2 + CT[4] ** 2 + CT[5] ** 2) ** 0.5
    plt.plot(CT[6], -CT[3] / N, color="red", label="-Px(t)", linestyle="--", marker=".")
    plt.plot(
        CT[6], -CT[4] / N, color="blue", label="-Py(t)", linestyle="--", marker="."
    )

    ## Switches should be at zeros of switching function
    plt.plot(CT[6], (N * acc - 1) * 10, color="green", label="S(t)")
    plt.plot(MT[6], U, color="black", label="|U(t)|", zorder=10)

    plt.legend()
    plt.xlabel("t (ND)")
    plt.grid(True)
    plt.show()
    ###########################################

    ###########################################

    MT = np.array(PowerOptimal).T
    CT = np.array(PowerCostates).T
    U = (MT[7] ** 2 + MT[8] ** 2 + MT[9] ** 2) ** 0.5

    plt.plot(MT[6], MT[7], color="red", label="Ux(t)")
    plt.plot(MT[6], MT[8], color="blue", label="Uy(t)")

    # Negative velocity costates
    plt.plot(
        CT[6], -CT[3] * acc, color="red", label="-Px(t)", linestyle="--", marker="."
    )
    plt.plot(
        CT[6], -CT[4] * acc, color="blue", label="-Py(t)", linestyle="--", marker="."
    )

    plt.plot(MT[6], U, color="black", label="|U(t)|", zorder=10)

    plt.legend()
    plt.xlabel("t (ND)")
    plt.grid(True)
    plt.show()
    ###########################################
