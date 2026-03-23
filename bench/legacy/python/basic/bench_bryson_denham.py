"""
Bryson Denham — Python Benchmark

Classic integral-objective optimal control problem.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tychopy as typy

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments


class Model(oc.ODEBase):
    def __init__(self):
        Xvars = 2
        Uvars = 1
        args = oc.ODEArguments(Xvars, Uvars)
        x = args.XVec()[0]
        v = args.XVec()[1]
        u = args.UVec()[0]
        ode = vf.stack([v, u])
        super().__init__(ode, Xvars, Uvars)


if __name__ == "__main__":
    n = 100
    ts = np.linspace(0, 1, n)
    vs = np.linspace(1, -1, n)

    IG = [[0.0, v, t, 0] for t, v in zip(ts, vs)]

    ode = Model()

    phase = ode.phase("LGL5", IG, 32)
    phase.addBoundaryValue("Front", range(0, 3), [0, 1, 0])
    phase.addUpperVarBound("Path", 0, 1 / 9)
    phase.addIntegralObjective((Args(1)[0] ** 2) / 2, [3])
    phase.addBoundaryValue("Back", range(0, 3), [0, -1, 1])
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_KKTtol(1.0e-10)
    phase.optimizer.PrintLevel = 4
    phase.setNumPartitions(1, 1)

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
