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
import tychopy.Astro.Constants as c
from tychopy.Astro.AstroModels import MEETwoBody_CSI
from tychopy.Astro.Extensions.ThrusterModels import CSIThruster
from tychopy.Astro.FramePlot import TBPlot, colpal

##############################################################################
vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments

"""
This is a widely used low-thrust optimization test problem of computing
a mass optimal transfer from Earth to Dionysus.

Im basing the initial and terminal conditons and non-dimensional scaling
off of the Junkins Paper below.

https://arc.aiaa.org/doi/abs/10.2514/1.G003686


"""


if __name__ == "__main__":
    Isp_dim = 3000  # S
    Tmag_dim = 0.32  # N
    tf_dim = 3534 * c.day  # s
    mass_dim = 4000  # kg

    typy.SoftwareInfo()
    thruster = CSIThruster(Tmag_dim, Isp_dim, mass_dim)
    ode = MEETwoBody_CSI(c.MuSun, c.AU, thruster)

    tf = tf_dim / ode.tstar
    integ = ode.integrator(0.1)

    ## Already Non Dimesonalized in the correct units from junkins paper
    X0 = np.array([0.99969, -0.00376, 0.01628, -7.702e-6, 6.188e-7, 14.161])
    XF = np.array([1.5536, 0.15303, -0.51994, 0.01618, 0.11814, 46.3302])

    Istate = np.zeros((11))
    Istate[0:6] = X0
    Istate[6] = 1  # Full mass is non-dimensionalized to one
    Istate[9] = 0.5

    ts = np.linspace(0, tf, 500)

    TrajIG = []

    # Lerp initial guess
    for t in ts:
        State = np.zeros((11))
        Xi = X0 + (XF - X0) * t / tf
        State[0:6] = Xi
        State[6] = 1
        State[7] = t
        State[9] = 0.5
        TrajIG.append(State)

    phase = ode.phase("LGL5", TrajIG, 160)
    phase.set_control_mode("BlockConstant")  # This problem only likes this

    phase.add_boundary_value("Front", range(0, 8), Istate[0:8])
    phase.add_lu_norm_bound(
        "Path", range(8, 11), 0.000001, 1, 1
    )  # Lowbound is leakmass
    phase.add_boundary_value("Back", [7], [tf])
    phase.add_boundary_value("Back", range(0, 6), XF[0:6])
    phase.add_value_objective("Back", 6, -1.0)

    phase.set_num_partitions(8, 8)
    phase.optimizer.set_opt_ls_mode("AUGLANG")
    phase.optimizer.set_max_ls_iters(2)
    phase.optimizer.set_max_acc_iters(200)
    phase.optimizer.set_bound_fraction(0.997)
    phase.optimizer.set_print_level(1)
    phase.optimizer.set_delta_h(1.0e-6)
    phase.optimizer.set_eq_con_tol(1.0e-9)
    phase.optimize()

    ConvTraj = phase.return_traj()

    FinalMass = ConvTraj[-1][6] * mass_dim

    print("Final Mass   :", FinalMass, " kg")
    print("Mass Expended:", mass_dim - FinalMass, " kg")

    Tab = phase.return_traj_table()

    integ = ode.integrator(0.1, Tab)
    integ.set_abs_tol(1.0e-13)

    ## Do this for non-blockconstant control or if you dont care about exact accuracy
    ReintTraj1 = integ.integrate_dense(ConvTraj[0], ConvTraj[-1][7])

    ## This is to be preffered if control is blockconstant
    ReintTraj2 = [ConvTraj[0]]
    for i in range(0, len(ConvTraj) - 1):
        Next = integ.integrate_dense(ReintTraj2[-1], ConvTraj[i + 1][7])[1::]
        ReintTraj2 += Next

    print(ReintTraj1[-1] - ConvTraj[-1])  # Less Accurate but still fine
    print(ReintTraj2[-1] - ConvTraj[-1])  # More Accurate

    #########################################################################

    fig, axs = plt.subplots(1, 2)

    TT = np.array(ReintTraj1).T
    axs[0].plot(TT[7], TT[8], label=r"$U_r$")
    axs[0].plot(TT[7], TT[9], label=r"$U_t$")
    axs[0].plot(TT[7], TT[10], label=r"$U_n$")
    axs[0].plot(TT[7], (TT[8] ** 2 + TT[9] ** 2 + TT[10] ** 2) ** 0.5, label=r"$|U|$")
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlabel(r"$t$")

    Earth = []
    Dion = []
    for ang in np.linspace(0, 2 * np.pi, 1000):
        XE = np.zeros((6))
        XE[0:5] = X0[0:5]
        XE[5] = ang
        Earth.append(typy.Astro.modified_to_cartesian(XE, 1))

        XD = np.zeros((6))
        XD[0:5] = XF[0:5]
        XD[5] = ang
        Dion.append(typy.Astro.modified_to_cartesian(XD, 1))

    TrajCart = ode.MEEToCartesian(ReintTraj1)
    TrajIGCart = ode.MEEToCartesian(TrajIG)

    plot = TBPlot(ode)
    plot.addTraj(TrajCart, "Converged", color="b")
    plot.addTraj(TrajIGCart, "Lerped Initial Guess", color="r")

    plot.addTraj(Earth, "Earth", color="g", linestyle="--")
    plot.addTraj(Dion, "Dionysus", color="k", linestyle="--")

    plot.Plot2dAx(axs[1], legend=True)
    axs[1].axis("Equal")
    axs[1].grid(True)

    plt.show()
