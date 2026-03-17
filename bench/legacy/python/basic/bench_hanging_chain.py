"""
Hanging Chain — Python Benchmark

Single chain shape optimization (catenary energy minimization).
All plotting code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class Chain(oc.ODEBase):
    def __init__(self):
        args = oc.ODEArguments(1, 1)
        super().__init__(args[2], 1, 1)


def Energy():
    x, u = Args(2).tolist()
    return x * vf.sqrt(1 + u**2)


def Length():
    (u,) = Args(1).tolist()
    return vf.sqrt(1 + u**2)


if __name__ == "__main__":
    a, b, L, n = 1, 3, 4, 500

    ts = np.linspace(0, 1, n)
    IG = []
    for t in ts:
        tm = 0.25 if b > a else 0.75
        x = 2 * abs(b - a) * t * (t - 2 * tm) + a
        u = 2 * abs(b - a) * (t * 2.0 - 2 * tm)
        IG.append([x, t, u])

    ode = Chain()
    phase = ode.phase("LGL5", IG, n)
    phase.setStaticParams([L])
    phase.addBoundaryValue("Front", [0, 1], [a, 0])
    phase.addBoundaryValue("Back", [0, 1], [b, 1])
    phase.addBoundaryValue("StaticParams", [0], [L])
    phase.addUpperVarBound("Path", 0, max(a, b) + 0.001)
    phase.addIntegralObjective(Energy(), [0, 2])
    phase.addIntegralParamFunction(Length(), [2], 0)
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.PrintLevel = 4

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
