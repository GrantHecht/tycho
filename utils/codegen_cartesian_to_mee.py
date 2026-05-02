"""Generate CartesianToMEE VectorFunction with analytic derivatives.

Direct posigrade Cartesian -> Modified Equinoctial Elements conversion that
does NOT detour through classical orbital elements. Mirrors
jacobwilliams/Fortran-Astrodynamics-Toolkit
modified_equinoctial_module.f90::cartesian_to_equinoctial.

Usage:
    cd utils && conda run -n tycho python codegen_cartesian_to_mee.py

Output:
    include/tycho/detail/astro/cartesian_to_mee.h
"""

import os

import sympy as sp
from CodeGen import TychoHeaderGen


def CartesianToMEE():
    """Generate Cartesian-to-MEE state conversion VectorFunction.

    Input:  [rx, ry, rz, vx, vy, vz] - Cartesian state
    Output: [p, f, g, h, k, L]       - Modified Equinoctial Elements
    Parameter: mu                    - gravitational parameter
    """
    Xs = sp.symbols("x:6")
    rx, ry, rz, vx, vy, vz = Xs
    mu = sp.symbols("mu")

    R = sp.Matrix([rx, ry, rz])
    V = sp.Matrix([vx, vy, vz])

    rdv = R.dot(V)
    r2 = R.dot(R)
    rmag = sp.sqrt(r2)
    rhat = R / rmag

    hvec = R.cross(V)
    hmag_sq = hvec.dot(hvec)
    hmag = sp.sqrt(hmag_sq)
    hhat = hvec / hmag

    vhat = (rmag * V - rdv * rhat) / hmag

    p = hmag_sq / mu
    one_plus_hz = 1 + hhat[2]
    k_eq = hhat[0] / one_plus_hz
    h_eq = -hhat[1] / one_plus_hz
    kk = k_eq * k_eq
    hh = h_eq * h_eq
    s2 = 1 + hh + kk
    tkh = 2 * k_eq * h_eq

    fhat = sp.Matrix(
        [
            (1 - kk + hh) / s2,
            tkh / s2,
            -2 * k_eq / s2,
        ]
    )
    ghat = sp.Matrix(
        [
            tkh / s2,
            (1 + kk - hh) / s2,
            2 * h_eq / s2,
        ]
    )

    ecc = V.cross(hvec) / mu - rhat
    f_eq = ecc.dot(fhat)
    g_eq = ecc.dot(ghat)
    L = sp.atan2(rhat[1] - vhat[0], rhat[0] + vhat[1])

    Eq = sp.Matrix([p, f_eq, g_eq, h_eq, k_eq, L])

    # SuperScalar primal (is_vectorizable=true, default) works because
    # Eigen 5 ships a free Eigen::atan2(ArrayBase, ArrayBase) overload
    # backed by scalar_atan2_op (true SIMD via the packet backend);
    # ADL on Eigen::Array<double, W, 1> finds it.
    #
    # precompute_params left default (True): the post-diff
    # _identify_precomputed_post_diff sp.cse pass on the full
    # {Func, Jac, Grad, Hess} set is wall-clock heavy (sympy CSE over a
    # Hessian whose terms compose atan2 with sqrt and division of all 6
    # inputs is intrinsically expensive), but the resulting 1/mu_ member
    # precompute is worth the codegen wait - this script is run-once,
    # not per build.  In practice pre-diff catches 1/mu_ on Func alone
    # and post-diff finds nothing new for this VF.
    header = TychoHeaderGen(
        "CartesianToMEE",
        Eq,
        sp.Matrix(Xs),
        [(mu, "Gravitational Parameter", "mu > 0.0")],
        docstr=(
            "Cartesian state to Modified Equinoctial Elements (posigrade, "
            "direct - no classical-element detour)"
        ),
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_cartesian_to_mee.py",
    )
    print(f"Generated cartesian_to_mee.h in {output_dir}")


if __name__ == "__main__":
    CartesianToMEE()
