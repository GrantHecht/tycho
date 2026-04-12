#!/usr/bin/env python3
"""Single-phase regression test: minimum-time two-body low-thrust.

If this diverges to tf ~ 0 in tycho but works in asset, the bug is in
per-phase evaluation (objective/constraint Jacobian). If it works, the
bug is in link-constraint machinery.
"""
import sys
import numpy as np

import tychopy as typy

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments


class TwoBody(oc.ode_x_u.ode):
    def __init__(self, P1mu, ltacc):
        args = oc.ODEArguments(6, 3)
        r = args.head3()
        v = args.segment3(3)
        g = r.normalized_power3() * (-P1mu)
        acc = g + args.tail3() * ltacc
        super().__init__(vf.stack([v, acc]), 6, 3)


def circ_state(r, td):
    v = np.sqrt(1.0 / r)
    t = np.deg2rad(td)
    X = np.zeros(7)
    X[0] = np.cos(t) * r
    X[1] = np.sin(t) * r
    X[3] = -np.sin(t) * v
    X[4] = np.cos(t) * v
    return X


def make_traj(r0, td0, r1, td1, tf, n):
    # Linear interp between two circular states, plus small control guess.
    X0 = circ_state(r0, td0)
    X1 = circ_state(r1, td1)
    out = []
    for i in range(n):
        s = i / (n - 1)
        X = (1 - s) * X0 + s * X1
        X[6] = s * tf
        pt = np.concatenate([X, np.ones(3) * 0.01])
        out.append(pt)
    return out


def main():
    LTacc = 0.01
    NSegs = 75
    ode = TwoBody(1.0, LTacc)

    # Start at theta=0 on unit circle, end at theta=90 deg on unit circle,
    # free tf (min-time). Guess tf=2*pi (one period).
    IG = make_traj(1.0, 0.0, 1.0, 90.0, 2.0 * np.pi, 300)
    X0 = circ_state(1.0, 0.0)
    X1 = circ_state(1.0, 90.0)

    phase = ode.phase("LGL5")
    phase.set_traj(IG, NSegs)
    phase.set_control_mode("BlockConstant")
    phase.add_value_lock("Front", range(0, 7))
    phase.sub_variables("Front", range(0, 7), X0[0:7])
    # Pin terminal position to target circle point.
    phase.add_boundary_value("Back", range(0, 6), X1[0:6])
    phase.add_lu_norm_bound("Path", [7, 8, 9], 0.01, 1.0, 1)
    phase.add_delta_time_objective(1.0)

    phase.optimizer.set_opt_ls_mode("L1")
    phase.optimizer.set_delta_h(5.0e-8)
    phase.optimizer.set_kkt_tol(1.0e-9)
    phase.optimizer.set_bound_fraction(0.997)
    phase.optimizer.print_level = 1
    phase.optimizer.set_max_ls_iters(1)

    phase.solve()
    flag = phase.optimize()
    traj = phase.return_traj()
    tf_nd = traj[-1][6]
    tf_periods = tf_nd / (2 * np.pi)
    back_pos = np.array(traj[-1][0:3])
    back_err = float(np.linalg.norm(back_pos - X1[0:3]))
    print(f"flag={int(flag)} tf={tf_periods:.6f}p tf_nd={tf_nd:.4f} "
          f"back_err={back_err:.6f}")
    if tf_nd > 0.5 and back_err < 1e-3:
        print("PASS")
        sys.exit(0)
    print("FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
