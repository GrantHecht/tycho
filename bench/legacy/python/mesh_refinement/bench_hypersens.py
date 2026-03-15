"""
Hyper-Sensitive Problem — Python Benchmark

Classic hypersensitive mesh-refinement benchmark with adaptive mesh.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time
import numpy as np
import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class HyperSens(oc.ODEBase):
    def __init__(self):
        XtU = oc.ODEArguments(1, 1)
        x = XtU.XVar(0)
        u = XtU.UVar(0)
        xdot = -(x) + u
        super().__init__(xdot, 1, 1)


if __name__ == "__main__":
    xt0 = 1.5
    xtf = 1.0
    tf = 10000.0

    ode = HyperSens()
    TrajIG = [
        [xt0 * (1 - t / tf) + xtf * (t / tf), t, 0] for t in np.linspace(0, tf, 1000)
    ]

    nsegs = 10
    phase = ode.phase("LGL7", TrajIG, nsegs)
    phase.addBoundaryValue("First", [0, 1], [xt0, 0])
    phase.addBoundaryValue("Last", [0, 1], [xtf, tf])
    phase.addIntegralObjective(Args(2).squared_norm() / 2, [0, 2])
    phase.addLUVarBound("Path", 0, -50, 50)
    phase.addLUVarBound("Path", 2, -50, 50)
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_SoeLSMode("L1")
    phase.optimizer.set_QPOrderingMode("MINDEG")
    phase.optimizer.PrintLevel = 4
    phase.PrintMeshInfo = False
    phase.setThreads(1, 1)
    phase.setAdaptiveMesh(True)
    phase.setMeshTol(1.0e-6)
    phase.setMaxMeshIters(10)
    phase.optimizer.set_EContol(1.0e-7)

    start = time.perf_counter()
    phase.optimize_solve()
    end = time.perf_counter()

    print(end - start)
