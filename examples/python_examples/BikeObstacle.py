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
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes


"""
This example is derived from the paper linked below. The objective is
too steer a bicycle around a circular obstacle and reach a specified desitination
in minimum time.

#https://arxiv.org/pdf/2003.00142.pdf

"""


class BikeODE(oc.ODEBase):
    def __init__(self, la, lb):
        args = oc.ODEArguments(4, 2)

        # x,y are position of bike
        # psi is direction of velocity
        # v in velocity of bike
        # acc is accelleration of bike
        # alpha is steering angle

        x, y, psi, v = args.x_vec().tolist()

        acc, alpha = args.u_vec().tolist()

        beta = vf.arctan((la / (la + lb)) * vf.tan(alpha))

        xdot = v * vf.cos(psi + beta)
        ydot = v * vf.sin(psi + beta)
        psidot = v * vf.sin(beta) / lb
        vdot = acc

        ode = vf.stack(xdot, ydot, psidot, vdot)
        super().__init__(ode, 4, 2)


def ObstacleConstraint(xobs, yobs, obsrad, m):
    # Path constraint
    x, y = Args(2).tolist()

    denom = obsrad + m
    ellips = ((x - xobs) / denom) ** 2 + ((y - yobs) / denom) ** 2

    return 1.0 - ellips


obsrad = 5  # obstacale radius m
m = 2.5  # extra margin for avioding obstacle m

xobs = 0  # obstacle x coordianate m
yobs = 50  # obstacle y coordinate m

la = 1.58  # front axle distance m
lb = 1.72  # back axle distance m


# initial conditions
x0 = 0  # m
y0 = 0  # m
psi0 = np.pi / 2  # rad
v0 = 15  # m/s
t0 = 0  # s
acc0 = 0
alpha0 = 0  # rad

## Bounds
accbound = 2  # max accelleration m/s^2
vlbound = 5  # lower bound on velocity m/s
vubound = 29  # upper bound on velocity m/s^


# terminal conditions
xf = 0
yf = 100

# estimate of final time
tfIG = yf / v0


TrajIG = []

for t in np.linspace(0, tfIG, 100):
    X = np.zeros(7)
    X[0] = x0 + obsrad + m + 1.0  # Bias IG to one side of obstacle
    X[1] = yf * t / tfIG
    X[2] = psi0
    X[3] = v0
    X[4] = t
    X[5] = acc0
    X[6] = alpha0
    TrajIG.append(X)


ode = BikeODE(la, lb)

phase = ode.phase("LGL3", TrajIG, 128)
phase.add_boundary_value("Front", [0, 1, 2, 3, 4], [x0, y0, psi0, v0, t0])
phase.add_lu_var_bound("Path", 3, vlbound, vubound)
phase.add_lu_var_bound("Path", 5, -accbound, accbound)
phase.add_lu_var_bound("Path", 6, -np.pi / 6, np.pi / 6)
phase.add_inequal_con("Path", ObstacleConstraint(xobs, yobs, obsrad, m), [0, 1])
phase.add_boundary_value("Back", [0, 1], [xf, yf])
phase.add_delta_time_objective(1.0)
phase.optimizer.set_tols(1.0e-9, 1.0e-9, 1.0e-9)
phase.optimize()


TrajF = phase.return_traj()

#########################################
TT = np.array(TrajF).T

ax0 = plt.subplot(421)
ax1 = plt.subplot(423)
ax2 = plt.subplot(425)
ax3 = plt.subplot(427)

idxs = [2, 3, 5, 6]
axs = [ax0, ax1, ax2, ax3]
ylabs = [r"$\psi$", r"$v$", r"$a$", r"$\alpha$"]

for i, ax in enumerate(axs):
    ax.plot(TT[4], TT[idxs[i]])
    ax.set_ylabel(ylabs[i])
    ax.grid(True)

ax4 = plt.subplot(122)

ax4.plot(TT[0], TT[1], label="Bike Path")
ax4.scatter(TT[0][0], TT[1][0], label="Start", color="k", marker="o")
ax4.scatter(TT[0][-1], TT[1][-1], label="Finish", color="k", marker="*")


ax4.set_xlabel(r"$x$")
ax4.set_ylabel(r"$y$")

angs = np.linspace(0, 2 * np.pi, 1000)

Xs = np.cos(angs)
Ys = np.sin(angs)

ax4.plot(Xs * obsrad, Ys * obsrad + yobs, label="Obstacle", color="r")
ax4.plot(
    Xs * (obsrad + m),
    Ys * (obsrad + m) + yobs,
    label="Obstacle+margin",
    color="k",
    linestyle="--",
)

ax4.grid(True)
ax4.legend(loc=3)
ax4.axis("Equal")
plt.show()
