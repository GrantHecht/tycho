"""
Cart-Pole — Python Benchmark

Minimum-effort swing-up of a cart-pole system via direct collocation.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class CartPole(oc.ODEBase):
    def __init__(self, l, m1, m2, g):
        Xvars = 4
        Uvars = 1
        XtU = oc.ODEArguments(Xvars, Uvars)
        x, theta, xdot, thetadot = XtU.XVec().tolist()
        F = XtU.UVar(0)
        Q = vf.stack([-g * vf.sin(theta), F + m2 * l * vf.sin(theta) * thetadot**2])
        Mvec_rm = vf.stack(vf.cos(theta), l, m1 + m2, m2 * l * vf.cos(theta))
        M = vf.RowMatrix(Mvec_rm, 2, 2)
        xddot_thetaddot = M.inverse() * Q
        ode = vf.stack([xdot, thetadot, xddot_thetaddot])
        super().__init__(ode, Xvars, Uvars)


if __name__ == "__main__":
    m1 = 1.0
    m2 = 0.3
    l = 0.5
    g = 9.81
    Fmax = 20.0
    xmax = 2.0
    tf = 2.0
    xf = 1.0

    ts = np.linspace(0, tf, 100)
    IG = [[xf * t / tf, np.pi * t / tf, 0, 0, t, 0.00] for t in ts]

    ode = CartPole(l, m1, m2, g)
    phase = ode.phase("LGL5", IG, 64)
    phase.addBoundaryValue("First", range(0, 5), [0, 0, 0, 0, 0])
    phase.addBoundaryValue("Last", range(0, 5), [xf, np.pi, 0, 0, tf])
    phase.addLUVarBound("Path", 5, -Fmax, Fmax)
    phase.addLUVarBound("Path", 0, -xmax, xmax)
    phase.addIntegralObjective(Args(1)[0] ** 2, [5])
    phase.setThreads(8, 8)
    phase.optimizer.PrintLevel = 4

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
