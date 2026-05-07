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

## Aerodynamic Tables generated in seperate script
from MinimumTimeToClimbTables import (
    CD0Tab,
    ClalphaTab,
    ThrustTab,
    etaTab,
    rhoTab,
    sosTab,
)

import tychopy as typy
import tychopy.optimal_control as oc
import tychopy.vector_functions as vf
from tychopy.optimal_control.mesh_error_plots import PhaseMeshErrorPlot, sns
from tychopy.vector_functions import Arguments as Args

"""
Minumum time to climb for supersonic aircraft. 

Original problem data comes from:

A. E. Bryson, M. N. Desai, and W. C. Hoffman, Energy-State Approximation 
in Performance Optimization of Supersonic Aircraft, 
Journal of Aircraft, Vol. 6, No. 6, November-December, 1969, pp. 481-488.

Using version reformulated in SI units by ICLOS2 
http://www.ee.ic.ac.uk/ICLOCS/ExampleMinFuelClimb.html

"""

############################################################################

g0 = 9.80665

Lstar = 10000
Tstar = 250.0
Mstar = 19050.864
Vstar = Lstar / Tstar


#############################################################################

mu = 3.986012e14
Re = 6378145
S = 49.2386
vexhaust = 1600 * g0


#############################################################################
class AirPlane(oc.ODEBase):
    def __init__(self):

        XtU = oc.ODEArguments(4, 1)

        # Altitude,velocity,flight path angle, mass
        h, v, fpa, mass = XtU.x_vec().tolist()
        # Angle of attack
        alpha = XtU.u_var(0)

        rho = rhoTab(h)
        sos = sosTab(h)

        Mach = v / sos
        CD0 = CD0Tab(Mach)
        Clalpha = ClalphaTab(Mach)
        eta = etaTab(Mach)

        Thrust = ThrustTab(Mach, h)

        CD = CD0 + eta * Clalpha * (alpha**2)
        CL = Clalpha * alpha
        q = 0.5 * rho * (v**2)
        D = q * S * CD
        L = q * S * CL
        r = h + Re

        hdot = v * vf.sin(fpa)
        vdot = (Thrust * vf.cos(alpha) - D) / mass - mu * vf.sin(fpa) / (r**2)
        fpadot = (Thrust * vf.sin(alpha) + L) / (mass * v) + vf.cos(fpa) * (
            v / r - mu / (v * (r**2))
        )
        mdot = -Thrust / vexhaust

        ode = vf.stack([hdot, vdot, fpadot, mdot])
        ########################################
        Vgroups = {}
        Vgroups[("h", "altitude")] = h
        Vgroups[("v", "velocity")] = v
        Vgroups["mass"] = mass
        Vgroups["fpa"] = fpa
        Vgroups[("alpha", "AoA")] = alpha
        Vgroups[("t", "time")] = XtU.t_var()

        super().__init__(ode, 4, 1, Vgroups=Vgroups)


#############################################################################
def Plot(Traj):

    sns.set_context("paper")

    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    ax4 = plt.subplot(122)

    T1 = np.array(Traj).T

    ax0.plot(T1[4], T1[0])
    ax1.plot(T1[4], T1[1])
    ax2.plot(T1[4], T1[2] * 180 / np.pi)
    ax3.plot(T1[4], T1[5] * 180 / np.pi)

    v = np.linspace(0, max(T1[1]) * 1.1, 30)
    h = np.linspace(0, max(T1[0]) * 1.05, 30)
    V, H = np.meshgrid(v, h)

    ## Energy height contours from bryson's paper
    def EnergyHeight(v, h):
        g = mu / (h + Re) ** 2
        return (0.5 * v**2) / g + h

    E = EnergyHeight(V, H)
    ax4.plot(T1[1], T1[0])
    cs = ax4.contour(V, H, E, colors="k", levels=7, linestyles="dotted")

    ax4.scatter(T1[1][0], T1[0][0], color="k")
    ax4.scatter(T1[1][-1], T1[0][-1], color="k", marker="*")

    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax0.set_ylabel(r"$h\;(m)$")
    ax1.set_ylabel(r"$v\;(\frac{m}{s})$")
    ax2.set_ylabel(r"$\gamma\;(^\circ)$")
    ax3.set_ylabel(r"$\alpha\;(^\circ)$")
    ax3.set_xlabel(r"$t\;(s)$")
    ax4.set_ylabel(r"$h\;(m)$")
    ax4.set_xlabel(r"$v\;(\frac{m}{s})$")

    fig.set_size_inches(10.0, 6.0)
    fig.tight_layout()
    plt.show()


#############################################################################
if __name__ == "__main__":
    #########################
    ht0 = 0.010  # Dont make exactly zero so it doesnt conflict with path constraint
    htf = 19994.88
    vt0 = 129.314
    vtf = 295.092
    fpat0 = 0
    fpatf = 0
    mass0 = 19050.864
    #########################
    hmin = 0
    hmax = 21000.0
    vmin = 5
    vmax = 600
    fpamin = -20 * np.pi / 180
    fpamax = 40 * np.pi / 180
    massmin = 16500
    alphamin = -np.pi / 4
    alphamax = np.pi / 4
    tfig = 200
    #########################
    ode = AirPlane()

    XtU0 = ode.make_input(h=ht0, v=vt0, fpa=fpat0, mass=mass0)
    XtUf = ode.make_input(h=htf, v=vtf, fpa=fpatf, mass=mass0, t=tfig)

    Traj = [XtU0 * (1 - t) + XtUf * t for t in np.linspace(0, 1, 100)]
    #########################

    phase = ode.phase("LGL3", Traj, 50)
    phase.set_auto_scaling(True)
    phase.set_units(h=Lstar, v=Vstar, mass=Mstar, t=Tstar)

    phase.set_control_mode("HighestOrderSpline")
    phase.add_boundary_value("First", range(0, 5), [ht0, vt0, fpat0, mass0, 0])

    phase.add_lu_var_bound("Path", "h", hmin, hmax)
    phase.add_lu_var_bound("Path", "v", vmin, vmax)
    phase.add_lu_var_bound("Path", "fpa", fpamin, fpamax)
    phase.add_lower_var_bound("Last", "mass", massmin)
    phase.add_lu_var_bound("Path", "alpha", alphamin, alphamax)
    phase.add_boundary_value("Last", ["h", "v", "fpa"], [htf, vtf, fpatf])
    phase.add_delta_time_objective(1.0)

    phase.optimizer.print_level = 1
    phase.set_num_partitions(8, 8)

    ## All error estimates and tolerances are in reference to the scaled ODE system
    phase.set_adaptive_mesh(True)
    phase.set_mesh_error_estimator(oc.MeshErrorEstimators.INTEGRATOR)
    phase.set_mesh_tol(1.0e-7)
    phase.optimize()

    Traj = phase.return_traj()

    print("Minimum Time to Climb:{0:.2f}s".format(Traj[-1][4]))

    PhaseMeshErrorPlot(phase, False)
    Plot(Traj)
