# =============================================================================
# Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
#
# CentralShooting variant of BettsLowThrust.py.
#
# The companion `BettsLowThrust.py` solves the same problem with LGL5
# collocation; this file uses the CentralShooting transcription, which
# enforces inter-knot defect equality by *propagating* the dynamics through
# the modernized Integrator (RK adaptive stepper) between consecutive knots
# and matching the propagated end-state to the next knot. CentralShooting
# is the natural fit for problems where the dynamics integrate cleanly and
# the control profile is well-modelled as piecewise-constant (BlockConstant
# is forced as the control mode by the transcription's default).
#
# Differences from the LGL5 sibling:
#   - phase("CentralShooting", IG, n_seg) — integrator-driven defects.
#   - set_control_mode("BlockConstant") — required for CentralShooting.
#   - Higher segment count (CentralShooting expresses the trajectory at
#     knots only; resolution comes from segment density, not collocation
#     polynomial order).
#   - Adaptive mesh refinement disabled — refinement is collocation-mesh
#     machinery; CentralShooting integrates and does not benefit from it.
#   - No basemap dependency: plotting is line-chart only so the example
#     runs in the standard `scripts/run_examples.py` sweep.
#
# Source problem: Betts, J.T. *Practical Methods for Optimal Control and
# Estimation Using Nonlinear Programming*, Cambridge University Press, 2009,
# example 6 page 265 (LEO -> MEO low-thrust orbit transfer with MEE
# dynamics + zonal-harmonics gravity).
# =============================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

import tychopy as typy

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments

# ----------------------------- Non-dimensionalization -----------------------
g0 = 32.174
W = 1  # lb
mu_e = 1.407645794e16

Lstar = 20925662.73  # feet
Tstar = Lstar / np.sqrt(mu_e / Lstar)
Mstar = W / g0  # slugs

Vstar = Lstar / Tstar
Fstar = Mstar * Lstar / (Tstar**2)
Astar = Lstar / (Tstar**2)
Mustar = (Lstar**3) / (Tstar**2)

Re = 20925662.73 / Lstar
mdot = (4.446618e-3 / (450 * g0)) / (Mstar / Tstar)
mu = (mu_e) / Mustar
Thrust = 4.446618e-3 / Fstar
Isp = 450 / Tstar
gs = g0 / Astar

J2 = 1082.639e-6
J3 = -2.565e-6
J4 = -1.608e-6

pt0 = 21837080.052835 / Lstar
ptf = 40007346.015232 / Lstar


# ----------------------------- VF helpers -----------------------------------
# (Identical to BettsLowThrust.py — kept inline so the example stands alone.)


def RTNBasisFunc():
    R, V = Args(6).tolist([(0, 3), (3, 3)])
    Rhat = R.normalized()
    Nhat = R.cross(V).normalized()
    That = Nhat.cross(R).normalized()
    return vf.stack(Rhat, That, Nhat)


def MEECartFunc(mu):
    X = Args(6)
    p, f, g, h, k, L = X.tolist()

    sinL = vf.sin(L)
    cosL = vf.cos(L)
    sqp = vf.sqrt(mu / p)

    w = 1 + f * cosL + g * sinL
    s2 = 1 + h**2 + k**2
    a2 = h**2 - k**2
    r = p / w
    r_s2 = r / s2
    subs2 = 1.0 / s2

    R = r_s2 * vf.stack(
        [
            cosL + a2 * cosL + 2.0 * h * k * sinL,
            sinL - a2 * sinL + 2.0 * h * k * cosL,
            2.0 * (h * sinL - k * cosL),
        ]
    )

    V = (
        -subs2
        * sqp
        * vf.stack(
            [
                sinL + a2 * sinL - 2.0 * h * k * cosL + g - 2.0 * f * h * k + a2 * g,
                -cosL + a2 * cosL + 2.0 * h * k * sinL - f + 2.0 * g * h * k + a2 * f,
                -2.0 * (h * cosL + k * sinL + f * h + g * k),
            ]
        )
    )

    return vf.stack([R, V])


def RadFunc(mu):
    X = Args(6)
    p, f, g, h, k, L = X.tolist()
    sinL = vf.sin(L)
    cosL = vf.cos(L)
    w = 1.0 + f * cosL + g * sinL
    return p / w


def ZonalGrav(mu, Re, J2, J3, J4):
    X = Args(6)
    R, V = X.tolist([(0, 3), (3, 3)])
    r = R.norm()

    Ir = R.normalized()
    North = np.array([0, 0, 1.0])
    In = (North - Ir * (Ir.dot(North))).normalized()

    sphi = Ir[2]
    cphi = vf.sqrt(1 - sphi**2)

    P2 = 0.5 * (3.0 * (sphi**2) - 1.0)
    P3 = 0.5 * (5.0 * (sphi**3) - 3 * sphi)
    P4 = (35 / 8) * (sphi**4) - (30 / 8) * (sphi**2) + 3 / 8

    D2 = 3 * sphi
    D3 = 0.5 * (15.0 * (sphi**2) - 3.0)
    D4 = (35 / 2) * (sphi**3) - (30 / 4) * (sphi)

    Js = [J2, J3, J4]
    Ps = [P2, P3, P4]
    Ds = [D2, D3, D4]

    grs = []
    gns = []
    for k in range(2, 5):
        gns.append(Ds[k - 2] * Js[k - 2] * ((Re / r) ** k))
        grs.append(((k + 1) * Ps[k - 2] * Js[k - 2]) * ((Re / r) ** k))

    gn = vf.sum(gns) * cphi
    gr = vf.sum(grs)
    Gcart = (gn * In - gr * Ir) * (-mu / R.squared_norm())

    RTNBasis = RTNBasisFunc()
    M = vf.RowMatrix(RTNBasis, 3, 3)
    return M * Gcart


def MEEDynamics2(mu):
    """Sub-expression-eliminated MEE dynamics in RTN-frame perturbing accels."""
    X = Args(9)
    p, f, g, h, k, L, ur, ut, un = X.tolist()
    sinL = vf.sin(L)
    cosL = vf.cos(L)
    w = 1.0 + f * cosL + g * sinL
    Xtmp = vf.stack(X, sinL, cosL, w)

    X2 = Args(12)
    p, f, g, h, k, L, ur, ut, un, sinL, cosL, w = X2.tolist()
    hk = X2.segment2(3)

    sqp = vf.sqrt(p) / np.sqrt(mu)
    s2 = 1.0 + hk.squared_norm()

    pdot = 2.0 * (p / w) * ut
    fdot = vf.sum(
        [
            ur * sinL,
            ((w + 1) * cosL + f) * (ut / w),
            -(h * sinL - k * cosL) * (g * un / w),
        ]
    )
    gdot = vf.sum(
        [
            -ur * cosL,
            ((w + 1) * sinL + g) * (ut / w),
            (h * sinL - k * cosL) * (f * un / w),
        ]
    )
    hkdot = vf.stack([cosL, sinL]) * ((s2 * un / w) / 2.0)
    Ldot = mu * (w / p) * (w / p) + (1.0 / w) * (h * sinL - k * cosL) * un

    return (vf.stack([pdot, fdot, gdot, hkdot, Ldot]) * sqp)(Xtmp)


class LTModel(oc.ODEBase):
    def __init__(self, mu, T, gs, Isp, Re, J2=False):
        XtUP = oc.ODEArguments(7, 3, 1)

        MEEs = XtUP.x_vec().head(6)
        ww = XtUP.x_var(6)
        p, f, g, h, k, L = MEEs.tolist()

        U = XtUP.u_vec().head3().normalized()
        tau = XtUP.p_var(0)
        wwdot = -T * (1 + 0.01 * tau) / (Isp)
        acc_T = gs * T * (1 + 0.01 * tau) * U / ww

        acc_J2 = ZonalGrav(mu, Re, J2, J3, J4)(MEECartFunc(mu))(MEEs)

        acc = acc_T + acc_J2
        Xdot = MEEDynamics2(mu).eval(vf.stack(MEEs, acc))

        ode = vf.stack([Xdot, wwdot])
        super().__init__(ode, 7, 3, 1)


def EqBCon():
    X = Args(6)
    p, f, g, h, k, L = X.tolist()
    eq1 = p - ptf
    eq2 = X.segment2(1).squared_norm() - 0.73550320568829**2
    eq3 = X.segment2(3).squared_norm() - 0.61761258786099**2
    eq4 = f * h + g * k
    return vf.stack([eq1, eq2, eq3, eq4])


def IqBCon():
    X = Args(6)
    p, f, g, h, k, L = X.tolist()
    return g * h - k * f


# ----------------------------- Plotting -------------------------------------
# Line plots only — no basemap dependency, so this example runs in the
# standard examples sweep.


def plot_results(Traj):
    T = np.array(Traj).T
    t_hr = T[7] * Tstar / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    axes[0].plot(t_hr, T[0] * Lstar * 0.3048 / 1000.0, label="p (km)")
    axes[0].set_xlabel("t (hr)")
    axes[0].set_ylabel("Semi-latus rectum p")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title(
        "Betts low-thrust transfer — CentralShooting transcription"
    )

    axes[1].plot(t_hr, T[8], label="u_r")
    axes[1].plot(t_hr, T[9], label="u_t")
    axes[1].plot(t_hr, T[10], label="u_n")
    axes[1].set_xlabel("t (hr)")
    axes[1].set_ylabel("Control direction (RTN)")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


# ----------------------------- Main -----------------------------------------
if __name__ == "__main__":
    X0 = np.zeros((12))
    X0[0] = pt0
    X0[3] = -0.25396764647494
    X0[5] = np.pi
    X0[6] = 1 / Fstar
    X0[8:11] = [0, 1, 0]
    X0[11] = -25

    ode = LTModel(mu, Thrust, gs, Isp, Re, J2)
    tfig = 90000 / Tstar

    def Prograde():
        RV = MEECartFunc(mu)
        RTNBasis = RTNBasisFunc()(RV)
        U = vf.RowMatrix(RTNBasis, 3, 3) * RV.tail(3).normalized()
        return U

    integ = ode.integrator(0.1, Prograde(), range(0, 6))
    IG = integ.integrate_dense(X0, tfig)

    # ---- CentralShooting transcription -------------------------------------
    # 64 segments: a sweet spot — fewer than the LGL5 sibling's effective
    # collocation count after mesh refinement, but enough for the integrator
    # to resolve the spiral-out trajectory between knots.
    n_seg = 64
    phase = ode.phase("CentralShooting", IG, n_seg)
    # CentralShooting forces BlockConstant control internally (see
    # ode_phase.h:236); set explicitly for clarity at the call site.
    phase.set_control_mode("BlockConstant")
    phase.integrator.set_initial_step_size(0.1)

    phase.add_boundary_value("Front", range(0, 8), X0[0:8])
    phase.add_equal_con("Path", Args(3).norm() - 1, [8, 9, 10])
    phase.add_lu_func_bound(
        "Path", RadFunc(mu), range(0, 6), Re, 10 * Re
    )
    phase.add_equal_con("Back", EqBCon(), range(0, 6))
    phase.add_inequal_con("Back", IqBCon(), range(0, 6))
    phase.add_lu_var_bound("ODEParams", 0, -50, 0)
    phase.add_lower_var_bound("Back", 6, 0.05)
    phase.add_value_objective("Back", 6, -1.0)

    phase.set_num_partitions(8, 8)
    phase.optimizer.print_level = 1
    phase.optimizer.set_eq_con_tol(1.0e-9)

    # NOTE: no `set_adaptive_mesh(True)` — mesh refinement targets collocation
    # transcriptions; CentralShooting resolves accuracy via the embedded RK
    # integrator and segment density, not by polynomial-order subdivision.

    t0 = time.perf_counter()
    flag = phase.optimize_solve()
    elapsed = time.perf_counter() - t0

    Traj = phase.return_traj()

    FinalWeight = Traj[-1][6] * Fstar
    FinalTime = Traj[-1][7] * Tstar
    ThrottleParam = Traj[-1][-1]

    print(f"CentralShooting solve flag : {flag}")
    print(f"Wall-clock                  : {elapsed:.2f} s")
    print(f"Final Weight                : {FinalWeight:.6f} lb")
    print(f"Final Time                  : {FinalTime:.6f} s")
    print(f"Throttle Parameter          : {ThrottleParam:.6f}")

    plot_results(Traj)
