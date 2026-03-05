"""
Van der Pol Oscillator — Python Benchmark

Complex ODE with high node count and threading.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time
import numpy as np
import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class VanderPol(oc.ODEBase):
    def __init__(self):
        args = oc.ODEArguments(2, 1)
        x0 = args[0]
        x1 = args[1]
        u = args[3]

        x0dot = (1.0 - x1 * x1) * x0 - x1 + u
        x1dot = x0
        ode = vf.stack(x0dot, x1dot)
        super().__init__(ode, 2, 1)


if __name__ == "__main__":
    ode = VanderPol()

    tf = 10.0

    TrajIG = [[0, 0, t, 0] for t in np.linspace(0, tf, 100)]

    phase = ode.phase("LGL3", TrajIG, 256)
    phase.integrator.setStepSizes(0.25, 0.001, 3)
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front", range(0, 3), [0, 1, 0])
    phase.addLUVarBound("Path", 3, -0.75, 1.0, 1.0)

    phase.addIntegralObjective(Args(3).squared_norm(), [0, 1, 3])
    phase.addBoundaryValue("Back", [0, 1, 2], [0.0, 0.0, tf])
    phase.optimizer.PrintLevel = 3
    phase.setThreads(8, 8)
    phase.optimizer.set_tols(1.0e-8, 1.0e-8, 1.0e-8)

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)