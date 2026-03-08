"""
Delta-3 Launch Vehicle — Python Benchmark

Four-phase launch trajectory optimization maximizing payload mass.
Uses 3D Cartesian dynamics with drag and rotating Earth.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time
import numpy as np
import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

g0 = 9.80665
Lstar = 6378145
Tstar = 961.0
Mstar = 301454.0

Astar = Lstar / Tstar**2
Vstar = Lstar / Tstar
Rhostar = Mstar / Lstar**3
Mustar = (Lstar**3) / (Tstar**2)
Fstar = Astar * Mstar

mu = 3.986012e14 / Mustar
Re = 6378145 / Lstar
We = 7.29211585e-5 * Tstar

RhoAir = 1.225 / Rhostar
h_scale = 7200 / Lstar
g = g0 / Astar

CD = 0.5
S = 4 * np.pi / Lstar**2

TS = 628500 / Fstar
T1 = 1083100 / Fstar
T2 = 110094 / Fstar

IS = 283.33364 / Tstar
I1 = 301.68 / Tstar
I2 = 467.21 / Tstar

tS = 75.2 / Tstar
t1 = 261 / Tstar
t2 = 700 / Tstar

TMS = 19290 / Mstar
TM1 = 104380 / Mstar
TM2 = 19300 / Mstar
TMPay = 4164 / Mstar

PMS = 17010 / Mstar
PM1 = 95550 / Mstar
PM2 = 16820 / Mstar

SMS = TMS - PMS
SM1 = TM1 - PM1
SM2 = TM2 - PM2

T_phase1 = 6 * TS + T1
T_phase2 = 3 * TS + T1
T_phase3 = T1
T_phase4 = T2

mdot_phase1 = (6 * TS / IS + T1 / I1) / g
mdot_phase2 = (3 * TS / IS + T1 / I1) / g
mdot_phase3 = T1 / (g * I1)
mdot_phase4 = T2 / (g * I2)

tf_phase1 = tS
tf_phase2 = 2 * tS
tf_phase3 = t1
tf_phase4 = t1 + t2

m0_phase1 = 9 * TMS + TM1 + TM2 + TMPay
mf_phase1 = m0_phase1 - 6 * PMS - (tS / t1) * PM1

m0_phase2 = mf_phase1 - 6 * SMS
mf_phase2 = m0_phase2 - 3 * PMS - (tS / t1) * PM1

m0_phase3 = mf_phase2 - 3 * SMS
mf_phase3 = m0_phase3 - (1 - 2 * tS / t1) * PM1

m0_phase4 = mf_phase3 - SM1
mf_phase4 = m0_phase4 - PM2


class RocketODE(oc.ODEBase):
    def __init__(self, T, mdot):
        XtU = oc.ODEArguments(7, 3)
        R = XtU.XVec().head3()
        V = XtU.XVec().segment3(3)
        m = XtU.XVar(6)
        u = XtU.UVec().normalized()
        h = R.norm() - Re
        rho = RhoAir * vf.exp(-h / h_scale)
        Vr = V + R.cross(np.array([0, 0, We]))
        D = (-0.5 * CD * S) * rho * (Vr * Vr.norm())
        Rdot = V
        Vdot = (-mu) * R.normalized_power3() + (T * u + D) / m
        ode = vf.stack(Rdot, Vdot, -mdot)
        super().__init__(ode, 7, 3)


def TargetOrbit(at, et, it, Ot, Wt):
    R, V = Args(6).tolist([(0, 3), (3, 3)])
    r = R.norm()
    v = V.norm()
    hvec = R.cross(V)
    nvec = vf.cross([0, 0, 1], hvec)
    eps = 0.5 * (v**2) - mu / r
    a = -0.5 * mu / eps
    evec = V.cross(hvec) / mu - R.normalized()
    e = evec.norm()
    i = vf.arccos(hvec.normalized()[2])
    O = vf.arccos(nvec.normalized()[0])
    O = vf.ifelse(nvec[1] > 0, O, 2 * np.pi - O)
    W = vf.arccos(nvec.normalized().dot(evec.normalized()))
    W = vf.ifelse(evec[2] > 0, W, 2 * np.pi - W)
    return vf.stack([a, e, i, O, W]) - np.array([at, et, it, Ot, Wt])


if __name__ == "__main__":
    at = 24361140 / Lstar
    et = 0.7308
    Ot = np.deg2rad(269.8)
    Wt = np.deg2rad(130.5)
    istart = np.deg2rad(28.5)

    y0 = np.zeros((6))
    y0[0:3] = np.array([np.cos(istart), 0, np.sin(istart)]) * Re
    y0[3:6] = -np.cross(y0[0:3], np.array([0, 0, We]))
    y0[3] += 0.00001 / Vstar

    MF = -0.05
    OEF = [at, et, istart, Ot, Wt, MF]
    yf = ast.Astro.classic_to_cartesian(OEF, mu)

    ts = np.linspace(0, tf_phase4, 1000)

    IG1 = []
    IG2 = []
    IG3 = []
    IG4 = []

    for t in ts:
        X = np.zeros((11))
        X[0:6] = y0 + (yf - y0) * (t / ts[-1])
        X[7] = t
        X[8:11] = np.array([0, 1, 0])
        if t < tf_phase1:
            m = m0_phase1 + (mf_phase1 - m0_phase1) * (t / tf_phase1)
            X[6] = m
            IG1.append(X)
        elif t < tf_phase2:
            m = m0_phase2 + (mf_phase2 - m0_phase2) * (
                (t - tf_phase1) / (tf_phase2 - tf_phase1)
            )
            X[6] = m
            IG2.append(X)
        elif t < tf_phase3:
            m = m0_phase3 + (mf_phase3 - m0_phase3) * (
                (t - tf_phase2) / (tf_phase3 - tf_phase2)
            )
            X[6] = m
            IG3.append(X)
        elif t < tf_phase4:
            m = m0_phase4 + (mf_phase4 - m0_phase4) * (
                (t - tf_phase3) / (tf_phase4 - tf_phase3)
            )
            X[6] = m
            IG4.append(X)

    ode1 = RocketODE(T_phase1, mdot_phase1)
    ode2 = RocketODE(T_phase2, mdot_phase2)
    ode3 = RocketODE(T_phase3, mdot_phase3)
    ode4 = RocketODE(T_phase4, mdot_phase4)

    tmode = "LGL3"
    cmode = "HighestOrderSpline"

    phase1 = ode1.phase(tmode, IG1, 40)
    phase1.setControlMode(cmode)
    phase1.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase1.addBoundaryValue("Front", range(0, 8), IG1[0][0:8])
    phase1.addLowerNormBound("Path", [0, 1, 2], Re * 0.999999)
    phase1.addBoundaryValue("Back", [7], [tf_phase1])

    phase2 = ode2.phase(tmode, IG2, 40)
    phase2.setControlMode(cmode)
    phase2.addLowerNormBound("Path", [0, 1, 2], Re)
    phase2.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase2.addBoundaryValue("Front", [6], [m0_phase2])
    phase2.addBoundaryValue("Back", [7], [tf_phase2])

    phase3 = ode3.phase(tmode, IG3, 40)
    phase3.setControlMode(cmode)
    phase3.addLowerNormBound("Path", [0, 1, 2], Re)
    phase3.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase3.addBoundaryValue("Front", [6], [m0_phase3])
    phase3.addBoundaryValue("Back", [7], [tf_phase3])

    phase4 = ode4.phase(tmode, IG4, 40)
    phase4.setControlMode(cmode)
    phase4.addLowerNormBound("Path", [0, 1, 2], Re)
    phase4.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase4.addBoundaryValue("Front", [6], [m0_phase4])
    phase4.addUpperVarBound("Back", 7, tf_phase4, 1.0)
    phase4.addEqualCon("Back", TargetOrbit(at, et, istart, Ot, Wt), range(0, 6))
    phase4.addValueObjective("Back", 6, -1.0)

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)
    ocp.addForwardLinkEqualCon(phase1, phase4, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.set_SoeLSMode("L1")
    ocp.optimizer.set_MaxLSIters(2)
    ocp.optimizer.PrintLevel = 4

    start = time.perf_counter()
    ocp.solve_optimize()
    end = time.perf_counter()

    print(end - start)
