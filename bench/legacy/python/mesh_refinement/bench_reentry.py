"""
Space Shuttle Reentry — Python Benchmark

Minimum cross-range reentry (Betts reference). Solves, refines mesh, and
re-optimizes. Heating-rate constraint pass omitted.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl

################### Non-dimensionalization constants ############################
g0 = 32.2
W = 203000
Lstar = 100000.0
Tstar = 60.0
Mstar = W / g0
Vstar = Lstar / Tstar
Fstar = Mstar * Lstar / (Tstar**2)
Astar = Lstar / (Tstar**2)
Rhostar = Mstar / (Lstar**3)
BTUstar = 778.0 * Lstar * Fstar
Mustar = (Lstar**3) / (Tstar**2)

tmax = 2500 / Tstar
Re = 20902900 / Lstar
S = 2690.0 / (Lstar**2)
m = (W / g0) / Mstar
mu = (0.140765e17) / Mustar
rho0 = 0.002378 / Rhostar
h_ref = 23800 / Lstar

a0 = -0.20704
a1 = 0.029244
b0 = 0.07854
b1 = -0.61592e-2
b2 = 0.621408e-3


class ShuttleReentry(oc.ODEBase):
    def __init__(self):
        Xvars = 5
        Uvars = 2
        XtU = oc.ODEArguments(Xvars, Uvars)
        h, theta, v, gamma, psi = XtU.XVec().tolist()
        alpha, beta = XtU.UVec().tolist()
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
        super().__init__(ode, Xvars, Uvars)


if __name__ == "__main__":
    tf = 1000 / Tstar
    ht0 = 260000 / Lstar
    htf = 80000 / Lstar
    vt0 = 25600 / Vstar
    vtf = 2500 / Vstar
    gammat0 = np.deg2rad(-1.0)
    gammatf = np.deg2rad(-5.0)
    psit0 = np.deg2rad(90.0)

    ts = np.linspace(0, tf, 200)
    TrajIG = []
    for t in ts:
        X = np.zeros(8)
        X[0] = ht0 * (1 - t / tf) + htf * t / tf
        X[1] = 0
        X[2] = vt0 * (1 - t / tf) + vtf * t / tf
        X[3] = gammat0 * (1 - t / tf) + gammatf * t / tf
        X[4] = psit0
        X[5] = t
        X[6] = 0.00
        X[7] = 0.00
        TrajIG.append(np.copy(X))

    ode = ShuttleReentry()
    phase = ode.phase("LGL3", TrajIG, 40)
    phase.addBoundaryValue("Front", range(0, 6), TrajIG[0][0:6])
    phase.addLUVarBounds("Path", [1, 3], np.deg2rad(-89.0), np.deg2rad(89.0), 1.0)
    phase.addLUVarBound("Path", 6, np.deg2rad(-90.0), np.deg2rad(90.0), 1.0)
    phase.addLUVarBound("Path", 7, np.deg2rad(-90.0), np.deg2rad(1.0), 1.0)
    phase.addUpperDeltaTimeBound(tmax, 1.0)
    phase.addBoundaryValue("Back", [0, 2, 3], [htf, vtf, gammatf])
    phase.addDeltaVarObjective(1, -1.0)
    phase.setNumPartitions(8, 8)
    phase.optimizer.set_SoeLSMode("L1")
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.PrintLevel = 4

    start = time.perf_counter()
    phase.solve_optimize()
    phase.refineTrajManual(300)
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
