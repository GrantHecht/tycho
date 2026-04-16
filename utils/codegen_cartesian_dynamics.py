"""Generate CartesianDynamics VectorFunction with analytic derivatives.

Usage:
    cd utils && conda run -n tycho python codegen_cartesian_dynamics.py

Output:
    include/tycho/detail/astro/cartesian_dynamics.h
"""

import os

import sympy as sp
from CodeGen import TychoHeaderGen


def CartesianDynamics():
    """Generate Cartesian two-body dynamics VectorFunction.

    Input:  [x, y, z, vx, vy, vz, ax, ay, az] — 6 state + 3 extra accelerations
    Output: [xdot, ydot, zdot, vxdot, vydot, vzdot]
    Parameter: mu — gravitational parameter

    Dynamics: point-mass two-body gravity plus user-supplied perturbation.
    """
    Xs = sp.symbols("x:9")
    x, y, z, vx, vy, vz, ax, ay, az = Xs
    mu = sp.symbols("mu")

    r2 = x**2 + y**2 + z**2
    r = sp.sqrt(r2)
    r3 = r**3
    inv_r3 = 1 / r3

    xdot = vx
    ydot = vy
    zdot = vz
    vxdot = -mu * x * inv_r3 + ax
    vydot = -mu * y * inv_r3 + ay
    vzdot = -mu * z * inv_r3 + az

    Eq = sp.Matrix([xdot, ydot, zdot, vxdot, vydot, vzdot])

    header = TychoHeaderGen(
        "CartesianDynamics",
        Eq,
        sp.Matrix(Xs),
        [(mu, "Gravitational Parameter", "mu > 0.0")],
        docstr="Cartesian two-body dynamics with extra acceleration",
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_cartesian_dynamics.py",
    )
    print(f"Generated cartesian_dynamics.h in {output_dir}")


if __name__ == "__main__":
    CartesianDynamics()
