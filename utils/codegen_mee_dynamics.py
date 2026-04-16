"""Generate MEEDynamics VectorFunction with analytic derivatives.

Usage:
    cd utils && conda run -n tycho python codegen_mee_dynamics.py

Output:
    include/tycho/detail/astro/MEEDynamics.h
"""

import os

import sympy as sp
from CodeGen import TychoHeaderGen


def MEEDynamics():
    """Generate MEE dynamics (Gauss variational equations) VectorFunction.

    Input:  [p, f, g, h, k, L, ur, ut, un] — 6 MEE + 3 RTN accelerations
    Output: [pdot, fdot, gdot, hdot, kdot, Ldot] — MEE rates
    Parameter: mu — gravitational parameter

    Formulas: standard Gauss variational equations for MEE.
    """
    Xs = sp.symbols("x:9")
    p, f, g, h, k, L, ur, ut, un = Xs
    mu = sp.symbols("mu")

    sinL = sp.sin(L)
    cosL = sp.cos(L)
    sqp = sp.sqrt(p) / sp.sqrt(mu)

    w = 1 + f * cosL + g * sinL
    s2 = 1 + h**2 + k**2

    pdot = 2 * (p / w) * ut
    fdot = sum(
        [
            ur * sinL,
            ((w + 1) * cosL + f) * (ut / w),
            -(h * sinL - k * cosL) * (g * un / w),
        ]
    )
    gdot = sum(
        [
            -ur * cosL,
            ((w + 1) * sinL + g) * (ut / w),
            (h * sinL - k * cosL) * (f * un / w),
        ]
    )
    hdot = cosL * ((s2 * un / w) / 2)
    kdot = sinL * ((s2 * un / w) / 2)
    Ldot = mu * (w / p) * (w / p) + (1 / w) * (h * sinL - k * cosL) * un

    Eq = sp.Matrix([pdot, fdot, gdot, hdot, kdot, Ldot]) * sqp

    header = TychoHeaderGen(
        "MEEDynamics",
        Eq,
        sp.Matrix(Xs),
        [(mu, "Gravitational Parameter", "mu > 0.0")],
        docstr="Modified Equinoctial Element Dynamics",
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_mee_dynamics.py",
    )
    print(f"Generated MEEDynamics.h in {output_dir}")


if __name__ == "__main__":
    MEEDynamics()
