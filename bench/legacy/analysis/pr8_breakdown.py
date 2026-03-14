#!/usr/bin/env python3
"""
Break down transcription vs solve time for representative PR 8 workloads.

Run with a built `_tycho` on `PYTHONPATH`, for example:

  conda run -n tycho env \
    PYTHONPATH=/path/to/build/src/Bindings:/path/to/repo \
    python bench/analysis/pr8_breakdown.py --case brachistochrone --runs 5
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


def build_brachistochrone():
    class Brachistochrone(oc.ODEBase):
        def __init__(self, g):
            xtu = oc.ODEArguments(3, 1)
            _, _, v = xtu.XVec().tolist()
            theta = xtu.UVar(0)
            ode = vf.stack([vf.sin(theta) * v, -1.0 * vf.cos(theta) * v, g * vf.cos(theta)])
            super().__init__(ode, 3, 1)

    g = 9.81
    ode = Brachistochrone(g)
    x0, y0, v0 = 0, 10, 0
    xf, yf = 10, 5
    theta0 = 1.0
    tf = 1.0

    ts = np.linspace(0, tf, 100)
    traj = []
    for t in ts:
        x = np.zeros(5)
        x[0] = x0 + (xf - x0) * t / tf
        x[1] = y0 + (yf - y0) * t / tf
        x[2] = g * t * np.cos(theta0)
        x[3] = t
        x[4] = theta0
        traj.append(x)

    phase = ode.phase("LGL3", traj, 32)
    phase.addBoundaryValue("Front", range(0, 4), [x0, y0, v0, 0])
    phase.addLUVarBound("Path", 4, -0.1, 2.0)
    phase.addBoundaryValue("Back", [0, 1], [xf, yf])
    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.PrintLevel = 4
    return phase


def build_delta3():
    g0 = 9.80665
    lstar = 6378145
    tstar = 961.0
    mstar = 301454.0

    astar = lstar / tstar**2
    vstar = lstar / tstar
    rhostar = mstar / lstar**3
    mustar = (lstar**3) / (tstar**2)
    fstar = astar * mstar

    mu = 3.986012e14 / mustar
    re = 6378145 / lstar
    we = 7.29211585e-5 * tstar

    rho_air = 1.225 / rhostar
    h_scale = 7200 / lstar
    g = g0 / astar

    cd = 0.5
    s = 4 * np.pi / lstar**2

    ts = 628500 / fstar
    t1 = 1083100 / fstar
    t2 = 110094 / fstar

    isp_s = 283.33364 / tstar
    isp_1 = 301.68 / tstar
    isp_2 = 467.21 / tstar

    burn_s = 75.2 / tstar
    burn_1 = 261 / tstar
    burn_2 = 700 / tstar

    tms = 19290 / mstar
    tm1 = 104380 / mstar
    tm2 = 19300 / mstar
    tmpay = 4164 / mstar

    pms = 17010 / mstar
    pm1 = 95550 / mstar
    pm2 = 16820 / mstar

    sms = tms - pms
    sm1 = tm1 - pm1

    thrust_phase1 = 6 * ts + t1
    thrust_phase2 = 3 * ts + t1
    thrust_phase3 = t1
    thrust_phase4 = t2

    mdot_phase1 = (6 * ts / isp_s + t1 / isp_1) / g
    mdot_phase2 = (3 * ts / isp_s + t1 / isp_1) / g
    mdot_phase3 = t1 / (g * isp_1)
    mdot_phase4 = t2 / (g * isp_2)

    tf_phase1 = burn_s
    tf_phase2 = 2 * burn_s
    tf_phase3 = burn_1
    tf_phase4 = burn_1 + burn_2

    m0_phase1 = 9 * tms + tm1 + tm2 + tmpay
    mf_phase1 = m0_phase1 - 6 * pms - (burn_s / burn_1) * pm1

    m0_phase2 = mf_phase1 - 6 * sms
    mf_phase2 = m0_phase2 - 3 * pms - (burn_s / burn_1) * pm1

    m0_phase3 = mf_phase2 - 3 * sms
    mf_phase3 = m0_phase3 - (1 - 2 * burn_s / burn_1) * pm1

    m0_phase4 = mf_phase3 - sm1
    mf_phase4 = m0_phase4 - pm2

    class RocketODE(oc.ODEBase):
        def __init__(self, thrust, mdot):
            xtu = oc.ODEArguments(7, 3)
            r = xtu.XVec().head3()
            v = xtu.XVec().segment3(3)
            mass = xtu.XVar(6)
            u = xtu.UVec().normalized()
            h = r.norm() - re
            rho = rho_air * vf.exp(-h / h_scale)
            vr = v + r.cross(np.array([0, 0, we]))
            drag = (-0.5 * cd * s) * rho * (vr * vr.norm())
            rdot = v
            vdot = (-mu) * r.normalized_power3() + (thrust * u + drag) / mass
            ode = vf.stack(rdot, vdot, -mdot)
            super().__init__(ode, 7, 3)

    def target_orbit(at, et, inc, om, argp):
        r, v = Args(6).tolist([(0, 3), (3, 3)])
        radius = r.norm()
        speed = v.norm()
        hvec = r.cross(v)
        nvec = vf.cross([0, 0, 1], hvec)
        eps = 0.5 * (speed**2) - mu / radius
        sma = -0.5 * mu / eps
        evec = v.cross(hvec) / mu - r.normalized()
        ecc = evec.norm()
        inc_val = vf.arccos(hvec.normalized()[2])
        raan = vf.arccos(nvec.normalized()[0])
        raan = vf.ifelse(nvec[1] > 0, raan, 2 * np.pi - raan)
        aop = vf.arccos(nvec.normalized().dot(evec.normalized()))
        aop = vf.ifelse(evec[2] > 0, aop, 2 * np.pi - aop)
        return vf.stack([sma, ecc, inc_val, raan, aop]) - np.array([at, et, inc, om, argp])

    at = 24361140 / lstar
    et = 0.7308
    om = np.deg2rad(269.8)
    argp = np.deg2rad(130.5)
    inc = np.deg2rad(28.5)

    y0 = np.zeros(6)
    y0[0:3] = np.array([np.cos(inc), 0, np.sin(inc)]) * re
    y0[3:6] = -np.cross(y0[0:3], np.array([0, 0, we]))
    y0[3] += 0.00001 / vstar

    oef = [at, et, inc, om, argp, -0.05]
    yf = ast.Astro.classic_to_cartesian(oef, mu)

    times = np.linspace(0, tf_phase4, 1000)
    ig1, ig2, ig3, ig4 = [], [], [], []
    for t in times:
        x = np.zeros(11)
        x[0:6] = y0 + (yf - y0) * (t / times[-1])
        x[7] = t
        x[8:11] = np.array([0, 1, 0])
        if t < tf_phase1:
            x[6] = m0_phase1 + (mf_phase1 - m0_phase1) * (t / tf_phase1)
            ig1.append(x.copy())
        elif t < tf_phase2:
            x[6] = m0_phase2 + (mf_phase2 - m0_phase2) * ((t - tf_phase1) / (tf_phase2 - tf_phase1))
            ig2.append(x.copy())
        elif t < tf_phase3:
            x[6] = m0_phase3 + (mf_phase3 - m0_phase3) * ((t - tf_phase2) / (tf_phase3 - tf_phase2))
            ig3.append(x.copy())
        elif t < tf_phase4:
            x[6] = m0_phase4 + (mf_phase4 - m0_phase4) * ((t - tf_phase3) / (tf_phase4 - tf_phase3))
            ig4.append(x.copy())

    ode1 = RocketODE(thrust_phase1, mdot_phase1)
    ode2 = RocketODE(thrust_phase2, mdot_phase2)
    ode3 = RocketODE(thrust_phase3, mdot_phase3)
    ode4 = RocketODE(thrust_phase4, mdot_phase4)

    phase1 = ode1.phase("LGL3", ig1, 40)
    phase1.setControlMode("HighestOrderSpline")
    phase1.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase1.addBoundaryValue("Front", range(0, 8), ig1[0][0:8])
    phase1.addLowerNormBound("Path", [0, 1, 2], re * 0.999999)
    phase1.addBoundaryValue("Back", [7], [tf_phase1])

    phase2 = ode2.phase("LGL3", ig2, 40)
    phase2.setControlMode("HighestOrderSpline")
    phase2.addLowerNormBound("Path", [0, 1, 2], re)
    phase2.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase2.addBoundaryValue("Front", [6], [m0_phase2])
    phase2.addBoundaryValue("Back", [7], [tf_phase2])

    phase3 = ode3.phase("LGL3", ig3, 40)
    phase3.setControlMode("HighestOrderSpline")
    phase3.addLowerNormBound("Path", [0, 1, 2], re)
    phase3.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase3.addBoundaryValue("Front", [6], [m0_phase3])
    phase3.addBoundaryValue("Back", [7], [tf_phase3])

    phase4 = ode4.phase("LGL3", ig4, 40)
    phase4.setControlMode("HighestOrderSpline")
    phase4.addLowerNormBound("Path", [0, 1, 2], re)
    phase4.addLUNormBound("Path", [8, 9, 10], 0.5, 1.5)
    phase4.addBoundaryValue("Front", [6], [m0_phase4])
    phase4.addUpperVarBound("Back", 7, tf_phase4, 1.0)
    phase4.addEqualCon("Back", target_orbit(at, et, inc, om, argp), range(0, 6))
    phase4.addValueObjective("Back", 6, -1.0)

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)
    ocp.addForwardLinkEqualCon(phase1, phase4, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.set_SoeLSMode("L1")
    ocp.optimizer.set_MaxLSIters(2)
    ocp.optimizer.PrintLevel = 4
    return ocp


def build_cannon():
    g0 = 9.81
    lstar = 1000
    tstar = 60.0
    mstar = 10
    astar = lstar / tstar**2
    vstar = lstar / tstar
    rhostar = mstar / lstar**3
    estar = mstar * (vstar**2)

    cd = 0.5
    rho_air = 1.225 / rhostar
    rho_iron = 7870 / rhostar
    h_scale = 8.44e3 / lstar
    e0 = 400000 / estar
    g = g0 / astar

    def mass_func(rad):
        return (4 / 3) * (np.pi * rho_iron) * (rad**3)

    def area_func(rad):
        return np.pi * (rad**2)

    class Cannon(oc.ODEBase):
        def __init__(self):
            args = oc.ODEArguments(4, 0, 1)
            v = args.XVar(0)
            gamma = args.XVar(1)
            h = args.XVar(2)
            r = args.XVar(3)
            rad = args.PVar(0)
            area = area_func(rad)
            mass = mass_func(rad)
            rho = rho_air * vf.exp(-h / h_scale)
            drag = (0.5 * cd) * rho * (v**2) * area
            vdot = -drag / mass - g * vf.sin(gamma)
            gammadot = -g * vf.cos(gamma) / v
            hdot = v * vf.sin(gamma)
            rdot = v * vf.cos(gamma)
            ode = vf.stack([vdot, gammadot, hdot, rdot])
            super().__init__(ode, 4, 0, 1)

    def energy_constraint():
        v, rad = Args(2).tolist()
        mass = mass_func(rad)
        energy = 0.5 * mass * (v**2)
        return energy - e0

    rad0 = 0.1 / lstar
    h0 = 100 / lstar
    r0 = 0
    m0 = mass_func(rad0)
    gamma0 = np.deg2rad(45)
    v0 = np.sqrt(2 * e0 / m0) * 0.99

    ode = Cannon()
    integ = ode.integrator(0.01)
    integ.setAbsTol(1.0e-14)

    ig = np.zeros(6)
    ig[0] = v0
    ig[1] = gamma0
    ig[2] = h0
    ig[3] = r0
    ig[5] = rad0

    def ascent_event():
        args = oc.ODEArguments(4, 0, 1)
        return args[0] * vf.sin(args[1])

    ascent_ig = integ.integrate_dense(ig, 60 / tstar, [(ascent_event(), 0, 1)])[0]
    descent_ig = integ.integrate_dense(
        ascent_ig[-1],
        ascent_ig[-1][4] + 1000 / tstar,
        [(oc.ODEArguments(4, 0, 1)[2], 0, 1)],
    )[0]

    aphase = ode.phase("LGL5", ascent_ig, 128)
    aphase.addLowerVarBound("ODEParams", 0, 0.0, 1)
    aphase.addLowerVarBound("Front", 1, 0.0, 1.0)
    aphase.addBoundaryValue("Front", [2, 3, 4], [h0, r0, 0])
    aphase.addInequalCon("Front", energy_constraint() * 0.01, [0], [0], [])
    aphase.addBoundaryValue("Back", [1], [0.0])

    dphase = ode.phase("LGL5", descent_ig, 128)
    dphase.addBoundaryValue("Back", [2], [0.0])
    dphase.addValueObjective("Back", 3, -1.0)

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(aphase)
    ocp.addPhase(dphase)
    ocp.addForwardLinkEqualCon(aphase, dphase, [0, 1, 2, 3, 4])
    ocp.addDirectLinkEqualCon(0, "ODEParams", [0], 1, "ODEParams", [0])
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.PrintLevel = 4
    return ocp


def build_optimal_docking():
    lstar = 10.0
    tstar = 30.0
    mstar = 10.0

    astar = lstar / tstar**2
    vstar = lstar / tstar
    mustar = (lstar**3) / (tstar**2)
    fstar = astar * mstar

    a = 7071000 / lstar
    mu = 3.986e14 / mustar
    n = np.sqrt(mu / a**3)
    mass = 100 / mstar

    max_thrust = 0.1 / fstar
    max_torque = 1 / (fstar * lstar)

    srad = 1 / lstar
    udvec = np.array([0, 1.01, 0]) / lstar

    ivec = np.array([1000, 2000, 1000]) / (mstar * lstar * lstar)

    class RelDynModel2(oc.ODEBase):
        def __init__(self):
            args = oc.ODEArguments(13, 6)
            x = args.XVec().head3()
            v = args.XVec().segment3(3)
            q = args.XVec().segment(6, 4).normalized()
            w = args.XVec().segment3(10)
            thrust = args.UVec().head3()
            torque = args.UVec().tail3()
            xdot = v
            vdoto = vf.stack([2 * n * v[1] + (3 * n**2) * x[0], -2 * n * v[0], -(n**2) * x[2]])
            thrust_global = vf.quatRotate(q, thrust)
            vdot = vdoto + thrust_global / mass
            qdot = vf.quatProduct(q, w.padded_lower(1)) / 2.0
            l1 = w.cwiseProduct(ivec)
            wdot = (l1.cross(w) + torque).cwiseQuotient(ivec)
            ode = vf.stack([xdot, vdot, qdot, wdot])
            super().__init__(ode, 13, 6)

    class TorqueFree(oc.ODEBase):
        def __init__(self):
            args = oc.ODEArguments(7, 0)
            p = args.XVec().head(4).normalized()
            phi = args.XVec().segment3(4)
            pdot = vf.quatProduct(p, phi.padded_lower(1)) / 2.0
            l2 = phi.cwiseProduct(ivec)
            phidot = (l2.cross(phi)).cwiseQuotient(ivec)
            ode = vf.stack([pdot, phidot])
            super().__init__(ode, 7, 0)

    def rend_con(ud):
        x, v, q, w, p, phi = Args(20).tolist([(0, 3), (3, 3), (6, 4), (10, 3), (13, 4), (17, 3)])
        q = q.normalized()
        p = p.normalized()
        xdq = vf.quatRotate(q, ud)
        vdq = vf.quatRotate(q, w)
        vdq = -xdq.cross(vdq)
        xdp = vf.quatRotate(p, ud)
        vdp = vf.quatRotate(p, phi)
        vdp = -xdp.cross(vdp)
        return vf.stack([x + xdq - xdp, v + vdq - vdp])

    def rend_con2(ud, tab):
        func = oc.InterpFunction(tab, range(0, 7)).vf()
        x, v, q, w, t = Args(14).tolist([(0, 3), (3, 3), (6, 4), (10, 3), (13, 1)])
        return rend_con(ud)(x, v, q, w, func(t))

    ode_torquefree = TorqueFree()
    integ_torquefree = ode_torquefree.integrator(0.01)
    integ_torquefree.Adaptive = True

    sim_time = 600 / tstar
    target_traj_is = np.zeros(8)
    target_traj_is[0] = 0.05
    target_traj_is[3] = np.sqrt(1 - target_traj_is[0] ** 2)
    target_traj_is[5] = 0.0349 * tstar
    target_traj_is[6] = 0.017453 * tstar

    target_traj = integ_torquefree.integrate_dense(target_traj_is, sim_time, 2000)
    target_tab = oc.LGLInterpTable(ode_torquefree.vf(), 7, 0, target_traj)

    x0 = np.zeros(20)
    x0[1] = -10.0 / lstar
    x0[9] = 1
    x0[14] = -max_thrust
    x0[15] = max_thrust
    x0[19] = -max_torque / 4

    ode = RelDynModel2()
    integ = ode.integrator(0.01)
    integ.Adaptive = True
    traj = integ.integrate_dense(x0, 200 / tstar, 1000)

    phase = ode.phase("LGL3", traj, 384)
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front", range(0, 14), x0[0:14])
    phase.addLUVarBounds("Path", [14, 15, 16], -max_thrust, max_thrust, 0.1)
    phase.addLUVarBounds("Path", [17, 18, 19], -max_torque, max_torque, 1)
    phase.addLowerNormBound("Path", [0, 1, 2], 2 * srad, 1.0)
    phase.addEqualCon("Last", rend_con2(udvec, target_tab), range(0, 14))
    phase.addUpperDeltaTimeBound(sim_time)
    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.set_BoundFraction(0.995)
    phase.optimizer.PrintLevel = 4
    return phase


BUILDERS = {
    "brachistochrone": build_brachistochrone,
    "cannon": build_cannon,
    "delta3": build_delta3,
    "optimal_docking": build_optimal_docking,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    build = BUILDERS[args.case]
    runs = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        prob = build()
        t1 = time.perf_counter()
        prob.transcribe(False, False)
        t2 = time.perf_counter()
        if args.case in {"delta3"}:
            flag = prob.solve_optimize()
        else:
            flag = prob.optimize()
        t3 = time.perf_counter()
        runs.append(
            {
                "setup": t1 - t0,
                "transcribe": t2 - t1,
                "solve": t3 - t2,
                "total": t3 - t0,
                "flag": int(flag),
            }
        )

    print(json.dumps({"case": args.case, "runs": runs}))


if __name__ == "__main__":
    main()
