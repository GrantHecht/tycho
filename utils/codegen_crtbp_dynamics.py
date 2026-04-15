"""Generate CRTBPDynamics VectorFunction with analytic derivatives.

Usage:
    cd utils && conda run -n tycho python codegen_crtbp_dynamics.py

Output:
    include/tycho/detail/astro/crtbp_dynamics.h
"""

import os

import sympy as sp
from CodeGen import TychoHeaderGen


def CRTBPDynamics():
    """Generate CR3BP rotating-frame dynamics VectorFunction.

    Input:  [x, y, z, vx, vy, vz, ax, ay, az] — 6 rotating-frame state
            (position + velocity) plus 3 extra accelerations.
    Output: [xdot, ydot, zdot, vxdot, vydot, vzdot]
    Parameter: mu — CR3BP mass ratio (0 < mu < 1)

    Dynamics: classical CR3BP with Coriolis and centrifugal terms plus
    attractions from both primaries, with a user-supplied perturbation
    added to the accelerations.
    """
    Xs = sp.symbols("x:9")
    x, y, z, vx, vy, vz, ax, ay, az = Xs
    mu = sp.symbols("mu")

    one_minus_mu = 1 - mu

    dx1 = x + mu
    dx2 = x - one_minus_mu
    d1 = sp.sqrt(dx1**2 + y**2 + z**2)
    d2 = sp.sqrt(dx2**2 + y**2 + z**2)
    inv_d1_3 = 1 / d1**3
    inv_d2_3 = 1 / d2**3

    g1x = -one_minus_mu * dx1 * inv_d1_3
    g1y = -one_minus_mu * y * inv_d1_3
    g1z = -one_minus_mu * z * inv_d1_3

    g2x = -mu * dx2 * inv_d2_3
    g2y = -mu * y * inv_d2_3
    g2z = -mu * z * inv_d2_3

    xdot = vx
    ydot = vy
    zdot = vz
    vxdot = 2 * vy + x + g1x + g2x + ax
    vydot = -2 * vx + y + g1y + g2y + ay
    vzdot = g1z + g2z + az

    Eq = sp.Matrix([xdot, ydot, zdot, vxdot, vydot, vzdot])

    header = TychoHeaderGen(
        "CRTBPDynamics",
        Eq,
        sp.Matrix(Xs),
        [(mu, "CR3BP Mass Ratio", "mu > 0.0 && mu < 1.0")],
        docstr="CR3BP rotating-frame dynamics with extra acceleration",
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_crtbp_dynamics.py",
    )
    print(f"Generated crtbp_dynamics.h in {output_dir}")


if __name__ == "__main__":
    CRTBPDynamics()
