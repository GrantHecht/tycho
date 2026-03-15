"""
Multi-Phase Zermelo — Python Benchmark

Multi-phase ship routing through waypoints with variable-direction wind.
Compares four wind models (4 separate OCP solves).
All plotting code removed. Output suppressed via PrintLevel.
"""

import time
import numpy as np
import tycho as ast

oc = ast.OptimalControl
vf = ast.VectorFunctions

nSeg = 150
tol = 1e-12


class Zermelo(oc.ODEBase):
    def __init__(self, vMax, wFunc):
        Xvars = 2
        Uvars = 1
        args = vf.Arguments(Xvars + 1 + Uvars)
        xyt = args.head_3()
        th = args[3]
        wx, wy = wFunc(xyt)
        xD = vMax * vf.cos(th) + wx
        yD = vMax * vf.sin(th) + wy
        ode = vf.Stack([xD, yD])
        super().__init__(ode, Xvars, Uvars)


def noWind(xyt):
    return 0, 0


def uniformWind(xyt, ang=135 * np.pi / 180, vel=2):
    return vel * np.cos(ang), vel * np.sin(ang)


def constantDirWind(xyt, ang=45 * np.pi / 180):
    vel = vf.cos(xyt.head2().norm())
    return vel * np.cos(ang), vel * np.sin(ang)


def variableDirWind(xyt):
    vel = vf.sin(xyt.head2().norm())
    ang = 2 * (xyt[0] + xyt[1])
    return vel * vf.cos(ang), vel * vf.sin(ang)


def navigate(Points, vM=1, wF=uniformWind):
    numphase = len(Points) - 1
    trajG = []
    for i in range(numphase):
        A = Points[i]
        B = Points[i + 1]
        dist = np.linalg.norm(B - A)
        t0 = dist / vM
        d = (B - A) / dist
        ang = np.arctan2(d[1], d[0])
        trajG.append([
            np.array(list(A + d * x) + [t0 * x, ang])
            for x in np.linspace(0, 1, num=nSeg)
        ])

    ocp = oc.OptimalControlProblem()
    for i in range(numphase):
        A = Points[i]
        B = Points[i + 1]
        phase = Zermelo(vM, wF).phase(oc.TranscriptionModes.LGL3)
        phase.Threads = 8
        phase.setTraj(trajG[i], nSeg)
        if i == 0:
            phase.addBoundaryValue("Front", [0, 1], A)
            phase.addBoundaryValue("Front", [2], [0.0])
            phase.addBoundaryValue("Back", [0, 1], B)
        else:
            phase.addBoundaryValue("Back", [0, 1], B)
        phase.addLUVarBound("Path", 3, -np.pi, np.pi, 1)
        phase.addDeltaTimeObjective(1.0)
        phase.addLowerDeltaTimeBound(0)
        phase.optimizer.set_EContol(tol)
        phase.optimizer.set_KKTtol(tol)
        phase.optimizer.PrintLevel = 4
        ocp.addPhase(phase)
    ocp.addForwardLinkEqualCon(0, -1, [0, 1, 2])
    ocp.optimizer.PrintLevel = 4
    ocp.solve_optimize()


if __name__ == "__main__":
    A = np.array([0, -1])
    B = np.array([1, 1])
    C = np.array([4, 0])
    D = np.array([0, -1])
    vM = 1.5

    start = time.perf_counter()

    navigate([A, B, C, D], vM=1, wF=noWind)
    navigate([A, B, C, D], vM=vM, wF=lambda xyt: uniformWind(xyt, vel=0.5))
    navigate([A, B, C, D], vM=vM, wF=constantDirWind)
    navigate([A, B, C, D], vM=vM, wF=variableDirWind)

    end = time.perf_counter()

    print(end - start)
