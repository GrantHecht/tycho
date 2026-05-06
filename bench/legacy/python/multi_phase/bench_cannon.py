"""
Multi-Phase Cannonball — Python Benchmark

Optimal cannonball radius for maximum range (Dymos reference).
Uses ascent and descent phases linked through an OCP.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

g0 = 9.81
Lstar = 1000
Tstar = 60.0
Mstar = 10
Astar = Lstar / Tstar**2
Vstar = Lstar / Tstar
Rhostar = Mstar / Lstar**3
Estar = Mstar * (Vstar**2)

CD = 0.5
RhoAir = 1.225 / Rhostar
RhoIron = 7870 / Rhostar
h_scale = 8.44e3 / Lstar
E0 = 400000 / Estar
g = g0 / Astar


def MFunc(rad, RhoIron):
    return (4 / 3) * (np.pi * RhoIron) * (rad**3)


def SFunc(rad):
    return np.pi * (rad**2)


class Cannon(oc.ODEBase):
    def __init__(self, CD, RhoAir, RhoIron, h_scale, g):
        args = oc.ODEArguments(4, 0, 1)
        v = args.XVar(0)
        gamma = args.XVar(1)
        h = args.XVar(2)
        r = args.XVar(3)
        rad = args.PVar(0)
        S = SFunc(rad)
        M = MFunc(rad, RhoIron)
        rho = RhoAir * vf.exp(-h / h_scale)
        D = (0.5 * CD) * rho * (v**2) * S
        vdot = -D / M - g * vf.sin(gamma)
        gammadot = -g * vf.cos(gamma) / v
        hdot = v * vf.sin(gamma)
        rdot = v * vf.cos(gamma)
        ode = vf.stack([vdot, gammadot, hdot, rdot])
        super().__init__(ode, 4, 0, 1)


def EFunc():
    v, rad = Args(2).tolist()
    M = MFunc(rad, RhoIron)
    E = 0.5 * M * (v**2)
    return E - E0


if __name__ == "__main__":
    rad0 = 0.1 / Lstar
    h0 = 100 / Lstar
    r0 = 0
    m0 = MFunc(rad0, RhoIron)
    gamma0 = np.deg2rad(45)
    v0 = np.sqrt(2 * E0 / m0) * 0.99

    ode = Cannon(CD, RhoAir, RhoIron, h_scale, g)
    integ = ode.integrator(0.01)
    integ.setAbsTol(1.0e-14)

    IG = np.zeros(6)
    IG[0] = v0
    IG[1] = gamma0
    IG[2] = h0
    IG[3] = r0
    IG[5] = rad0

    def AscentEvent():
        args = oc.ODEArguments(4, 0, 1)
        return args[0] * vf.sin(args[1])

    AscentIG = integ.integrate_dense(IG, 60 / Tstar, [(AscentEvent(), 0, 1)])[0]
    DescentIG = integ.integrate_dense(
        AscentIG[-1],
        AscentIG[-1][4] + 1000 / Tstar,
        [(oc.ODEArguments(4, 0, 1)[2], 0, 1)],
    )[0]

    tmode = "LGL5"
    nsegs = 128

    aphase = ode.phase(tmode, AscentIG, nsegs)
    aphase.addLowerVarBound("ODEParams", 0, 0.0, 1)
    aphase.addLowerVarBound("Front", 1, 0.0, 1.0)
    aphase.addBoundaryValue("Front", [2, 3, 4], [h0, r0, 0])
    aphase.addInequalCon("Front", EFunc() * 0.01, [0], [0], [])
    aphase.addBoundaryValue("Back", [1], [0.0])

    dphase = ode.phase(tmode, DescentIG, nsegs)
    dphase.addBoundaryValue("Back", [2], [0.0])
    dphase.addValueObjective("Back", 3, -1.0)

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(aphase)
    ocp.addPhase(dphase)
    ocp.addForwardLinkEqualCon(aphase, dphase, [0, 1, 2, 3, 4])
    ocp.addDirectLinkEqualCon(0, "ODEParams", [0], 1, "ODEParams", [0])
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.PrintLevel = 4

    start = time.perf_counter()
    ocp.optimize()
    end = time.perf_counter()

    print(end - start)
