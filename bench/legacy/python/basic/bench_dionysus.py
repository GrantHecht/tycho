"""
Dionysus Low-Thrust Trajectory — Python Benchmark

Earth to Dionysus low-thrust orbit transfer using MEE dynamics.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tycho as ast
import tycho.Astro.Constants as c
from tycho.Astro.AstroModels import MEETwoBody_CSI
from tycho.Astro.Extensions.ThrusterModels import CSIThruster

vf = ast.VectorFunctions
oc = ast.OptimalControl


if __name__ == "__main__":
    Isp_dim = 3000
    Tmag_dim = 0.32
    tf_dim = 3534 * c.day
    mass_dim = 4000

    thruster = CSIThruster(Tmag_dim, Isp_dim, mass_dim)
    ode = MEETwoBody_CSI(c.MuSun, c.AU, thruster)

    tf = tf_dim / ode.tstar

    X0 = np.array([0.99969, -0.00376, 0.01628, -7.702e-6, 6.188e-7, 14.161])
    XF = np.array([1.5536, 0.15303, -0.51994, 0.01618, 0.11814, 46.3302])

    Istate = np.zeros(11)
    Istate[0:6] = X0
    Istate[6] = 1
    Istate[9] = 0.5

    ts = np.linspace(0, tf, 500)
    TrajIG = []
    for t in ts:
        State = np.zeros(11)
        Xi = X0 + (XF - X0) * t / tf
        State[0:6] = Xi
        State[6] = 1
        State[7] = t
        State[9] = 0.5
        TrajIG.append(State)

    phase = ode.phase("LGL5", TrajIG, 160)
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front", range(0, 8), Istate[0:8])
    phase.addLUNormBound("Path", range(8, 11), 0.000001, 1, 1)
    phase.addBoundaryValue("Back", [7], [tf])
    phase.addBoundaryValue("Back", range(0, 6), XF[0:6])
    phase.addValueObjective("Back", 6, -1.0)
    phase.setNumPartitions(8, 8)
    phase.optimizer.set_OptLSMode("AUGLANG")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.set_MaxAccIters(200)
    phase.optimizer.set_BoundFraction(0.997)
    phase.optimizer.PrintLevel = 4
    phase.optimizer.set_deltaH(1.0e-6)
    phase.optimizer.set_EContol(1.0e-9)

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
