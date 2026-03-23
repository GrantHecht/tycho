"""
Free-Flying Robot — Python Benchmark

Minimum-fuel maneuver of a free-flying robot between two configurations.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tychopy as typy

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments


class FreeFlyingRobotODE(oc.ODEBase):
    def __init__(self, alpha, beta):
        args = oc.ODEArguments(6, 4)
        xydot = args.XVec().segment2(2)
        theta = args.XVar(4)
        omega = args.XVar(5)
        u = args.UVec()
        vscale = u[0] - u[1] + u[2] - u[3]
        vxydot = vf.stack([vf.cos(theta), vf.sin(theta)]) * vscale
        theta_dot = omega
        omega_dot = (u[0] - u[1]) * alpha + (u[3] - u[2]) * beta
        ode = vf.stack([xydot, vxydot, theta_dot, omega_dot])
        super().__init__(ode, 6, 4)


if __name__ == "__main__":
    ode = FreeFlyingRobotODE(0.2, 0.2)

    t0 = 0
    tf = 12
    X0 = np.array([-10, -10, 0, 0, np.pi / 2.0, 0, 0])
    XF = np.array([0, 0, 0, 0, 0, 0, tf])

    IG = []
    for t in np.linspace(0, tf, 100):
        T = np.zeros(11)
        T[0:7] = X0[0:7] + ((t - t0) / (tf - t0)) * (XF[0:7] - X0[0:7])
        T[7:11] = np.ones(4) * 0.50
        IG.append(T)

    phase = ode.phase("LGL5", IG, 128)
    phase.addBoundaryValue("Front", range(0, 7), X0)
    phase.addBoundaryValue("Back", range(0, 7), XF)
    phase.addLUVarBounds("Path", range(7, 11), 0.0, 1.0, 1)
    phase.addIntegralObjective(Args(4).sum(), range(7, 11))
    phase.optimizer.PrintLevel = 4
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.set_tols(1.0e-9, 1.0e-9, 1.0e-9)

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
