"""
CR3BP Orbit Continuation — Python Benchmark

Solves a single L1 Lyapunov periodic orbit in the Earth-Moon CR3BP,
then continues it through 10 steps. Exercises the integrator, STM,
and repeated OCP solves.
All plotting code removed. Output suppressed via PrintLevel.
"""

import copy
import time

import numpy as np

import tychopy as typy
import tychopy.Astro.Constants as c
from tychopy.Astro.AstroModels import CR3BP

oc = typy.OptimalControl
vf = typy.VectorFunctions

muE = c.MuEarth
muM = c.MuMoon
lstar = c.LD
dt = 3.1415 / 10000


def solvePeriodic(ig, tf, ode, odeItg, fixInit=[0, 1, 2]):
    steps = 1000
    trajGuess = odeItg.integrate_dense(ig, tf, steps)
    odePhase = ode.phase("LGL3")
    odePhase.NumPartitions = 8
    nSeg = 150
    odePhase.setTraj(trajGuess, nSeg)
    for idx in fixInit:
        odePhase.addBoundaryValue("Front", [idx], [ig[idx]])
    odePhase.addBoundaryValue("Front", [3, 6], [0.0, 0.0])
    odePhase.addBoundaryValue("Back", [1, 3, 5], [0.0, 0.0, 0.0])
    tol = 1e-12
    odePhase.optimizer.set_EContol(tol)
    odePhase.optimizer.PrintLevel = 4
    odePhase.solve()
    return odePhase.returnTraj()


if __name__ == "__main__":
    ode = CR3BP(muE, muM, lstar)
    odeItg = ode.integrator(dt)

    # L1 Lyapunov orbit initial guess
    ig = np.zeros(7)
    ig[0] = 0.8234
    ig[4] = 0.1263
    tf = 1.3

    start = time.perf_counter()

    # Solve initial orbit
    tj = solvePeriodic(ig, tf, ode, odeItg)

    # Continue 10 steps
    n_steps = 10
    dx = -0.001
    cIdx = 0
    lim = 0.77

    for _ in range(n_steps):
        g = np.copy(tj[0])
        t = np.copy(tj[-1][6])
        if np.sign(g[cIdx] - lim) != np.sign(tj[0][cIdx] - lim):
            break
        g[cIdx] += dx
        tj = solvePeriodic(g, t, ode, odeItg)

    end = time.perf_counter()

    print(end - start)
