"""Generate MEEToCartesian VectorFunction with analytic derivatives.

Usage:
    cd utils && conda run -n tycho python codegen_mee_to_cartesian.py

Output:
    include/tycho/detail/astro/MEEToCartesian.h
"""

import os

import sympy as sp
from CodeGen import TychoHeaderGen


def MEEToCartesian():
    """Generate MEE-to-Cartesian state conversion VectorFunction.

    Input:  [p, f, g, h, k, L] — Modified Equinoctial Elements
    Output: [x, y, z, vx, vy, vz] — Cartesian state
    Parameter: mu — gravitational parameter

    Formulas match kepler_utils.h modified_to_cartesian().
    """
    Xs = sp.symbols("x:6")
    p, f, g, h, k, L = Xs
    mu = sp.symbols("mu")

    sinL = sp.sin(L)
    cosL = sp.cos(L)

    a2 = h**2 - k**2
    s2 = 1 + h**2 + k**2
    w = 1 + f * cosL + g * sinL
    rr = p / w

    Xscale = rr / s2
    Vscale = sp.sqrt(mu / p) / s2

    # Position
    x_pos = Xscale * (cosL + a2 * cosL + 2 * h * k * sinL)
    y_pos = Xscale * (sinL - a2 * sinL + 2 * h * k * cosL)
    z_pos = 2 * Xscale * (h * sinL - k * cosL)

    # Velocity
    vx = -Vscale * (sinL + a2 * sinL - 2 * h * k * cosL + g - 2 * f * h * k + a2 * g)
    vy = -Vscale * (-cosL + a2 * cosL + 2 * h * k * sinL - f + 2 * g * h * k + a2 * f)
    vz = 2 * Vscale * (h * cosL + k * sinL + f * h + g * k)

    Eq = sp.Matrix([x_pos, y_pos, z_pos, vx, vy, vz])

    header = TychoHeaderGen(
        "MEEToCartesian",
        Eq,
        sp.Matrix(Xs),
        [(mu, "Gravitational Parameter", "mu > 0.0", 1.0)],
        docstr="MEE to Cartesian state conversion",
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro", "conversions"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_mee_to_cartesian.py",
    )
    print(f"Generated MEEToCartesian.h in {output_dir}")


if __name__ == "__main__":
    MEEToCartesian()
