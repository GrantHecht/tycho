"""
Topputto Low-Thrust Transfer — Python Benchmark

Minimum-time and minimum-fuel polar low-thrust orbit transfers
(Topputto 2014 formulation). Only the time-optimal solve is timed.
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class LTModel(oc.ODEBase):
    def __init__(self, amax):
        Xvars = 4
        Uvars = 2
        XtU = oc.ODEArguments(Xvars, Uvars)
        r, theta, vr, vt = XtU.XVec().tolist()
        u, alpha = XtU.UVec().tolist()
        rdot = vr
        thetadot = vt / r
        vrdot = (vt**2) / r - 1 / (r**2) + amax * u * vf.sin(alpha)
        vtdot = -(vt * vr) / r + amax * u * vf.cos(alpha)
        ode = vf.stack([rdot, thetadot, vrdot, vtdot])
        super().__init__(ode, Xvars, Uvars)


if __name__ == "__main__":
    amax = 0.01
    ode = LTModel(amax)
    integ = ode.integrator(0.01)

    RF = 4.0
    VF = np.sqrt(1 / RF)

    IState = np.zeros(7)
    IState[0] = 1
    IState[3] = 1
    IState[5] = 0.99
    IState[6] = 0

    def RFunc(x):
        return x[0] > RF

    ToptIG = integ.integrate_dense(IState, 130, 1000, RFunc)

    phase = ode.phase("LGL3", ToptIG, 400)
    phase.addBoundaryValue("Front", range(0, 5), IState[0:5])
    phase.addLUVarBound("Path", 5, 0.0001, 1, 100.0)
    phase.addLUVarBound("Path", 6, -2 * np.pi, 2 * np.pi, 1.0)
    phase.addBoundaryValue("Back", [0, 2, 3], [RF, 0, VF])
    phase.optimizer.PrintLevel = 4
    phase.optimizer.set_MaxAccIters(500)
    phase.optimizer.set_MaxIters(1000)
    phase.optimizer.set_BoundFraction(0.995)
    phase.optimizer.deltaH = 1.0e-5
    phase.addDeltaTimeObjective(1 / 100)

    start = time.perf_counter()
    phase.solve_optimize_solve()
    end = time.perf_counter()

    print(end - start)
