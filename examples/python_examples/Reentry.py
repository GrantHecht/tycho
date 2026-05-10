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
import tychopy.optimal_control as oc
import tychopy.vector_functions as vf
from tychopy.vector_functions import Arguments as Args

"""
Space Shuttle Reentry
Betts, J.T. Practical methods for Optimal Control and Estimation Using Nonlinear Programming, Cambridge University Press, 2009
"""

################### Non Dimensionalize ##################################
g0 = 32.2
W = 203000

Lstar = 100000.0  ## feet
Tstar = 60.0  ## sec
Vstar = Lstar / Tstar


tmax = 2500
Re = 20902900
S = 2690.0
m = W / g0
mu = 0.140765e17
rho0 = 0.002378
h_ref = 23800

a0 = -0.20704
a1 = 0.029244

b0 = 0.07854
b1 = -0.61592e-2
b2 = 0.621408e-3

c0 = 1.0672181
c1 = -0.19213774e-1
c2 = 0.21286289e-3
c3 = -0.10117e-5

Qlimit = 70.0


##############################################################################
class ShuttleReentry(oc.ODEBase):
    def __init__(self):

        Xvars = 5
        Uvars = 2

        ############################################################
        XtU = oc.ODEArguments(Xvars, Uvars)

        h, theta, v, gamma, psi = XtU.x_vec().tolist()

        alpha, beta = XtU.u_vec().tolist()

        alphadeg = (180.0 / np.pi) * alpha

        CL = a0 + a1 * alphadeg
        CD = b0 + b1 * alphadeg + b2 * (alphadeg**2)
        rho = rho0 * vf.exp(-h / h_ref)
        r = h + Re

        L = 0.5 * CL * S * rho * (v**2)
        D = 0.5 * CD * S * rho * (v**2)
        g = mu / (r**2)

        sgam = vf.sin(gamma)
        cgam = vf.cos(gamma)

        sbet = vf.sin(beta)
        cbet = vf.cos(beta)

        spsi = vf.sin(psi)
        cpsi = vf.cos(psi)
        tantheta = vf.tan(theta)

        hdot = v * sgam
        thetadot = (v / r) * cgam * cpsi
        vdot = -D / m - g * sgam
        gammadot = (L / (m * v)) * cbet + cgam * (v / r - g / v)
        psidot = L * sbet / (m * v * cgam) + (v / (r)) * cgam * spsi * tantheta

        ode = vf.stack([hdot, thetadot, vdot, gammadot, psidot])

        ## Create dict of aliases for different variables in the ODE
        ## We can use these later in our constraints instead of integer indices
        Vgroups = {}
        Vgroups[("h", "altitude")] = h
        Vgroups[("v", "velocity")] = v
        Vgroups["theta"] = theta
        Vgroups["gamma"] = gamma
        Vgroups["psi"] = psi
        Vgroups[("alpha", "AoA")] = alpha
        Vgroups["beta"] = beta
        Vgroups[("t", "time")] = XtU.t_var()

        ##############################################################
        super().__init__(ode, Xvars, Uvars, Vgroups=Vgroups)  ## Pass to CTor


def QFunc():
    h, v, alpha = Args(3).tolist()
    alphadeg = (180.0 / np.pi) * alpha
    rhodim = rho0 * vf.exp(-h / h_ref)
    vdim = v
    qr = 17700 * vf.sqrt(rhodim) * ((0.0001 * vdim) ** 3.07)
    qa = c0 + c1 * alphadeg + c2 * (alphadeg**2) + c3 * (alphadeg**3)

    return qa * qr


#############################################################################


def Plot(Traj1, Traj2):
    TT1 = np.array(Traj1).T
    TT2 = np.array(Traj2).T

    fig, axs = plt.subplots(4, 1)

    axs[0].plot(TT1[5] / 60.0, TT1[0] / 5280, label="No Q limit", color="r")
    axs[0].plot(TT2[5] / 60.0, TT2[0] / 5280, label="Q limited", color="b")
    axs[0].set_ylabel("Altitude (Miles)")

    axs[1].plot(TT1[5] / 60.0, TT1[2], label="No Q limit", color="r")
    axs[1].plot(TT2[5] / 60.0, TT2[2], label="Q limited", color="b")

    axs[1].set_ylabel(r"Velocity $\frac{ft}{s}$")

    axs[2].plot(
        TT1[5] / 60.0, np.rad2deg(TT1[6]), label="Angle of Attack No Q limit", color="r"
    )
    axs[2].plot(
        TT1[5] / 60.0,
        np.rad2deg(TT1[7]),
        label="Bank Angle No Q limit",
        color="r",
        linestyle="dotted",
    )
    axs[2].plot(
        TT2[5] / 60.0, np.rad2deg(TT2[6]), label="Angle of Attack  Q limited", color="b"
    )
    axs[2].plot(
        TT2[5] / 60.0,
        np.rad2deg(TT2[7]),
        label="Bank Angle  Q limited",
        color="b",
        linestyle="dotted",
    )

    axs[2].set_ylabel("Angle(deg)")

    qfunc = QFunc().eval(8, [0, 2, 6])

    qs1 = [qfunc(T)[0] for T in Traj1]
    qs2 = [qfunc(T)[0] for T in Traj2]

    axs[3].plot(TT1[5] / 60.0, qs1, label="No Q limit", color="r")
    axs[3].plot(TT2[5] / 60.0, qs2, label="Q limited", color="b")
    axs[3].set_ylabel(r"Q $(\frac{BTU}{ft^2 *s})$")

    axs[3].plot(
        TT2[5] / 60.0,
        np.ones_like(TT2[5]) * 70,
        label=r"Q = 70 $\frac{BTU}{ft^2 *s}$",
        color="k",
        linestyle="dashed",
    )

    for i in range(0, 4):
        axs[i].grid(True)
        axs[i].set_xlabel("Time (min)")
        axs[i].legend()

    fig.set_size_inches(8.0, 11.0, forward=True)
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    ##########################################################################

    ode = ShuttleReentry()

    tf = 2000

    ht0 = 260000
    htf = 80000
    vt0 = 25600
    vtf = 2500

    gammat0 = np.deg2rad(-1.0)
    gammatf = np.deg2rad(-5.0)
    psit0 = np.deg2rad(90.0)

    ## Construct state,time,control vector using names with ode.make_input
    X0 = ode.make_input(h=ht0, v=vt0, gamma=gammat0, psi=psit0)
    Xf = ode.make_input(h=htf, v=vtf, gamma=gammatf, psi=psit0, t=tf)
    # Lerp IG
    TrajIG = [X0 * (1.0 - t) + Xf * t for t in np.linspace(0.0, 1.0, 200)]

    ################################################################

    phase = ode.phase("LGL5", TrajIG, 40)

    ############################
    phase.set_auto_scaling(True)
    ## Declare canonical units for state time, control, and parm variables
    ## All default to 1, which is fine for all of our angular variables
    ## Only need to scale altitude, velocity and time.
    phase.set_units(h=Lstar, v=Vstar, t=Tstar)
    ###########################

    phase.add_boundary_value("Front", range(0, 6), TrajIG[0][0:6])

    phase.add_lu_var_bound("Path", "theta", np.deg2rad(-89.0), np.deg2rad(89.0))
    phase.add_lu_var_bound("Path", "gamma", np.deg2rad(-89.0), np.deg2rad(89.0))

    phase.add_lu_var_bound("Path", "AoA", np.deg2rad(-90.0), np.deg2rad(90.0))
    phase.add_lu_var_bound("Path", "beta", np.deg2rad(-90.0), np.deg2rad(1.0))

    phase.add_upper_delta_time_bound(tmax, 1.0)

    phase.add_boundary_value("Back", ["h", "v", "gamma"], [htf, vtf, gammatf])

    phase.add_delta_var_objective("theta", -1.0)

    phase.set_num_partitions(8, 8)

    phase.optimizer.set_soe_ls_mode("L1")
    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_print_level(1)

    ## All error estimates and tolerances are in reference to the scaled ODE system
    phase.set_adaptive_mesh(True)
    phase.set_mesh_tol(1.0e-7)
    phase.set_mesh_error_estimator(oc.MeshErrorEstimators.INTEGRATOR)

    phase.solve_optimize()

    # Returns unscaled trajectory original units
    Traj1 = phase.return_traj()

    ## Add in Heating Rate Constraint
    phase.add_upper_func_bound("Path", QFunc(), ["h", "v", "alpha"], Qlimit)

    phase.optimize()

    Traj2 = phase.return_traj()

    print(
        "Final Time:",
        Traj1[-1][5],
        "(s) , Final Cross Range:",
        Traj1[-1][1] * 180 / np.pi,
        " deg",
    )
    print(
        "Final Time:",
        Traj2[-1][5],
        "(s) , Final Cross Range:",
        Traj2[-1][1] * 180 / np.pi,
        " deg",
    )

    Plot(Traj1, Traj2)
