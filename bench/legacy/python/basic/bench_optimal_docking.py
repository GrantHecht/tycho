"""
Optimal Docking (Form 2) — Python Benchmark

Minimum-time rendezvous with an uncontrolled tumbling target.
Uses an LGLInterpTable for the target trajectory (eliminates target
states from the ODE, yielding a 13-state problem).
All plotting/animation code removed. Output suppressed via PrintLevel.
"""

import time

import numpy as np

import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

Lstar = 10.0
Tstar = 30.0
Mstar = 10.0

Astar = Lstar / Tstar**2
Vstar = Lstar / Tstar
Mustar = (Lstar**3) / (Tstar**2)
Fstar = Astar * Mstar

a = 7071000 / Lstar
mu = 3.986e14 / Mustar
n = np.sqrt(mu / a**3)
m = 100 / Mstar

MaxThrust = 0.1 / Fstar
MaxTorque = 1 / (Fstar * Lstar)

Srad = 1 / Lstar
Udvec = np.array([0, 1.01, 0]) / Lstar

Ivec = np.array([1000, 2000, 1000]) / (Mstar * Lstar * Lstar)


class RelDynModel2(oc.ODEBase):
    def __init__(self, I1, n, m):
        Xvars = 13
        Uvars = 6
        args = oc.ODEArguments(Xvars, Uvars)
        X = args.XVec().head3()
        V = args.XVec().segment3(3)
        q = args.XVec().segment(6, 4).normalized()
        w = args.XVec().segment3(10)
        Thrust = args.UVec().head3()
        Torque = args.UVec().tail3()
        Xdot = V
        Vdoto = vf.stack(
            [2 * n * V[1] + (3 * n**2) * X[0], -2 * n * V[0], -(n**2) * X[2]]
        )
        Thrust_global = vf.quatRotate(q, Thrust)
        Vdot = Vdoto + Thrust_global / m
        qdot = vf.quatProduct(q, w.padded_lower(1)) / 2.0
        L1 = w.cwiseProduct(I1)
        wdot = (L1.cross(w) + Torque).cwiseQuotient(I1)
        ode = vf.stack([Xdot, Vdot, qdot, wdot])
        super().__init__(ode, Xvars, Uvars)


class TorqueFree(oc.ODEBase):
    def __init__(self, I2):
        Xvars = 7
        Uvars = 0
        args = oc.ODEArguments(Xvars, Uvars)
        p = args.XVec().head(4).normalized()
        phi = args.XVec().segment3(4)
        pdot = vf.quatProduct(p, phi.padded_lower(1)) / 2.0
        L2 = phi.cwiseProduct(I2)
        phidot = (L2.cross(phi)).cwiseQuotient(I2)
        ode = vf.stack([pdot, phidot])
        super().__init__(ode, Xvars, Uvars)


def RendCon(ud):
    X, V, q, w, p, phi = Args(20).tolist(
        [(0, 3), (3, 3), (6, 4), (10, 3), (13, 4), (17, 3)]
    )
    q = q.normalized()
    p = p.normalized()
    Xdq = vf.quatRotate(q, ud)
    vdq = vf.quatRotate(q, w)
    Vdq = -Xdq.cross(vdq)
    Xdp = vf.quatRotate(p, ud)
    vdp = vf.quatRotate(p, phi)
    Vdp = -Xdp.cross(vdp)
    return vf.stack([X + Xdq - Xdp, V + Vdq - Vdp])


def RendCon2(ud, tab):
    func = oc.InterpFunction(tab, range(0, 7)).vf()
    X, V, q, w, t = Args(14).tolist([(0, 3), (3, 3), (6, 4), (10, 3), (13, 1)])
    return RendCon(ud)(X, V, q, w, func(t))


if __name__ == "__main__":
    ode_torquefree = TorqueFree(Ivec)
    integ_torquefree = ode_torquefree.integrator(0.01)
    integ_torquefree.Adaptive = True

    SimTime = 600 / Tstar

    TargetTraj_IS = np.zeros((8))
    TargetTraj_IS[0] = 0.05
    TargetTraj_IS[3] = np.sqrt(1 - TargetTraj_IS[0] ** 2)
    TargetTraj_IS[5] = 0.0349 * Tstar
    TargetTraj_IS[6] = 0.017453 * Tstar

    TargetTraj = integ_torquefree.integrate_dense(TargetTraj_IS, SimTime, 2000)
    TargetTab = oc.LGLInterpTable(ode_torquefree.vf(), 7, 0, TargetTraj)

    X0 = np.zeros((20))
    X0[1] = -10.0 / Lstar
    X0[9] = 1
    X0[14] = -MaxThrust
    X0[15] = MaxThrust
    X0[19] = -MaxTorque / 4

    ode = RelDynModel2(Ivec, n, m)
    integ = ode.integrator(0.01)
    integ.Adaptive = True

    Traj = integ.integrate_dense(X0, 200 / Tstar, 1000)

    phase = ode.phase("LGL3", Traj, 384)
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front", range(0, 14), X0[0:14])
    phase.addLUVarBounds("Path", [14, 15, 16], -MaxThrust, MaxThrust, 0.1)
    phase.addLUVarBounds("Path", [17, 18, 19], -MaxTorque, MaxTorque, 1)
    phase.addLowerNormBound("Path", [0, 1, 2], 2 * Srad, 1.0)
    phase.addEqualCon("Last", RendCon2(Udvec, TargetTab), range(0, 14))
    phase.addUpperDeltaTimeBound(SimTime)
    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.set_BoundFraction(0.995)
    phase.optimizer.PrintLevel = 4

    start = time.perf_counter()
    phase.optimize()
    end = time.perf_counter()

    print(end - start)
