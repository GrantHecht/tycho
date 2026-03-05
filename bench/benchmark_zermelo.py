"""
Zermelo's Problem — Python Benchmark

Time-dependent wind dynamics with varying wind model.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time
import numpy as np
import tycho as ast

oc = ast.OptimalControl
vf = ast.VectorFunctions

nSeg = 250
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


def variableDirWind(xyt):
    vel = vf.sin(xyt.head2().norm())
    ang = 2 * (xyt[0] + xyt[1])
    return vel * vf.cos(ang), vel * vf.sin(ang)


def navigate(A, B, vM=1, wF=variableDirWind):
    dist = np.linalg.norm(B - A)
    t0 = dist / vM
    d = (B - A) / dist
    ang = np.arctan2(d[1], d[0])
    trajG = [
        np.array(list(A + d * x) + [t0 * x, ang]) for x in np.linspace(0, 1, num=nSeg)
    ]

    phase = Zermelo(vM, wF).phase(oc.TranscriptionModes.LGL3)
    phase.Threads = 10

    phase.setTraj(trajG, nSeg)

    phase.addBoundaryValue("Front", [0, 1], A)
    phase.addBoundaryValue("Front", [2], [0.0])
    phase.addBoundaryValue("Back", [0, 1], B)

    phase.addLUVarBound("Path", 3, -np.pi, np.pi, 1)

    phase.addDeltaTimeObjective(1.0)

    phase.optimizer.set_EContol(tol)
    phase.optimizer.set_KKTtol(tol)
    phase.optimizer.PrintLevel = 3

    start = time.perf_counter()
    phase.solve_optimize()
    end = time.perf_counter()

    return end - start


if __name__ == "__main__":
    A = np.array([0, -1])
    B = np.array([1, 1])
    vM = 1.25

    duration = navigate(A, B, vM=vM, wF=variableDirWind)
    print(duration)