"""
Mountain Car — Python Benchmark

Minimum-time escape from a steep valley; engine too weak to climb directly.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl


class MountainCar(oc.ODEBase):
    def __init__(self):
        args = oc.ODEArguments(2, 1)
        x = args.XVar(0)
        v = args.XVar(1)
        u = args.UVar(0)
        xdot = v
        vdot = 0.001 * u - 0.0025 * vf.cos(3 * x)
        ode = vf.stack([xdot, vdot])
        super().__init__(ode, 2, 1)


if __name__ == "__main__":
    x0 = -0.5
    v0 = 0
    xf = 0.52
    tf = 500

    IG = [
        [x0 + (xf - x0) * t / tf, t / tf, t, np.sin(t / tf)]
        for t in np.linspace(0, tf, 100)
    ]

    ode = MountainCar()
    phase = ode.phase("LGL3", IG, 128)
    phase.addBoundaryValue("First", [0, 1, 2], [x0, v0, 0])
    phase.addBoundaryValue("Last", [0], [xf])
    phase.addLowerVarBound("Back", 1, 0.0, 1.0)
    phase.addLUVarBound("Path", 0, -1.2, 0.55, 1.0)
    phase.addLUVarBound("Path", 1, -0.07, 0.07, 100.0)
    phase.addLUVarBound("Path", 3, -1, 1, 1.0)
    phase.addDeltaTimeObjective(0.01)
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.PrintLevel = 4

    start = time.perf_counter()
    phase.solve_optimize()
    end = time.perf_counter()

    print(end - start)
