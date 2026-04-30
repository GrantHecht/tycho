"""Generate CartesianToModified VectorFunction with analytic derivatives.

Direct posigrade Cartesian -> Modified Equinoctial Elements conversion that
does NOT detour through classical orbital elements. Mirrors
jacobwilliams/Fortran-Astrodynamics-Toolkit
modified_equinoctial_module.f90::cartesian_to_equinoctial.

Usage:
    cd utils && conda run -n tycho python codegen_cartesian_to_modified.py

Output:
    include/tycho/detail/astro/cartesian_to_modified.h
"""

import os

import sympy as sp
from CodeGen import TychoHeaderGen


def CartesianToModified():
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

    # precompute_params=False: the only parameter is mu, and the
    # identify_precomputed step's full {Func, Jac, Grad, Hess} sp.cse
    # blew past 30 min on this conversion just to surface the trivial
    # 1/mu substitution. The per-method CSE in gen_compute_all still
    # delivers analytic derivatives; we only forfeit the small win of
    # caching 1/mu as a constructor-time member (negligible vs the
    # cost of computing it once per call).
    #
    # SuperScalar primal works because tycho/detail/utils/simd_math.h
    # adds the missing Eigen::atan2(Array<double, N, 1>, Array<double, N, 1>)
    # free overload (Eigen 3.4 ships only the unary cwise free ops via
    # EIGEN_ARRAY_DECLARE_GLOBAL_UNARY; binary atan2 stays a member).
    header = TychoHeaderGen(
        "CartesianToModified",
        Eq,
        sp.Matrix(Xs),
        [(mu, "Gravitational Parameter", "mu > 0.0")],
        docstr=(
            "Cartesian state to Modified Equinoctial Elements (posigrade, "
            "direct - no classical-element detour)"
        ),
        precompute_params=False,
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_cartesian_to_modified.py",
    )
    print(f"Generated cartesian_to_modified.h in {output_dir}")


if __name__ == "__main__":
    CartesianToModified()
