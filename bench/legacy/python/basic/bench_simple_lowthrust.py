"""
Simple Low-Thrust Transfer — Python Benchmark

Time-optimal Cartesian 3D low-thrust orbit transfer.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tychopy as typy

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments


class LTModel(oc.ODEBase):
    def __init__(self, mu, ltacc):
        Xvars = 6
        Uvars = 3
        args = oc.ODEArguments(Xvars, Uvars)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3() * (-mu)
        thrust = u * ltacc
        acc = g + thrust
        ode = vf.stack([v, acc])
        super().__init__(ode, Xvars, Uvars)


if __name__ == "__main__":
    mu = 1
    acc = 0.02
    ode = LTModel(mu, acc)

    r0 = 1.0
    v0 = np.sqrt(mu / r0)
    rf = 2.0
    vF = np.sqrt(mu / rf)

    X0 = np.zeros(7)
    X0[0] = r0
    X0[4] = v0

    Xf = np.zeros(6)
    Xf[0] = rf
    Xf[4] = vF

    XIG = np.zeros(10)
    XIG[0:7] = X0

    integ = ode.integrator(0.01, Args(3).normalized() * 0.8, [3, 4, 5])
    TrajIG = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

    phase = ode.phase("LGL3", TrajIG, 256)
    phase.addBoundaryValue("Front", range(0, 7), X0)
    phase.addLUNormBound("Path", [7, 8, 9], 0.001, 1, 1.0)
    phase.addBoundaryValue("Back", range(0, 6), Xf[0:6])
    phase.optimizer.PrintLevel = 4
    phase.optimizer.set_BoundFraction(0.995)
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.set_deltaH(1.0e-6)
    phase.addDeltaTimeObjective(1.0)

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
