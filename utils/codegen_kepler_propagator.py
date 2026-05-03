"""Generate KeplerPrimal_VF and KeplerResidual_VF closed-form helpers.

These VFs are *internal* helpers consumed by the IFT layer of the Kepler
propagator (Task 3 of the LCD plan); they are not registered as
Python-visible VFs.

- ``KeplerPrimal_VF``: post-converged primal map
  ``S(R0, V0, dt, X*, U0, U1, U2) -> (rf, vf)`` via Goodyear ``aF, aG,
  aFt, aGt`` formulas.  ``mu`` is a member parameter (constructor arg).
  11 inputs -> 6 outputs.
- ``KeplerResidual_VF``: universal-variable residual
  ``F = r0*U1 + sigma0*U2 + U3 - sqrt(mu)*dt`` (note: F has *no*
  explicit ``X`` dependence).  ``mu`` is a member parameter.
  10 inputs -> 1 output.

Both VFs are emitted as separate snake_case headers
(``kepler_primal_vf.h`` and ``kepler_residual_vf.h``) under
``include/tycho/detail/astro/``; the IFT layer includes both directly.

Usage:
    cd utils && conda run -n tycho python codegen_kepler_propagator.py
"""

import os

import sympy as sp
from CodeGen import TychoHeaderGen


def _common_symbols():
    R0 = sp.symbols("R0_:3")
    V0 = sp.symbols("V0_:3")
    dt, X = sp.symbols("dt X")
    U0, U1, U2, U3 = sp.symbols("U0 U1 U2 U3")
    mu = sp.symbols("mu")
    return R0, V0, dt, X, U0, U1, U2, U3, mu


def _kepler_primal():
    """S(R0, V0, dt, X, U0, U1, U2) -> (rf, vf). 11 inputs, 6 outputs.

    Note: U3 is *not* a primal input — the post-converged primal map
    only depends on (U0, U1, U2) plus (R0, V0, dt, X).  ``dt`` is
    carried as an input (rather than derived) so the IFT chain rule has
    the same symbol set as the residual VF.
    """
    R0, V0, dt, X, U0, U1, U2, _U3, mu = _common_symbols()
    r0 = sp.sqrt(R0[0] ** 2 + R0[1] ** 2 + R0[2] ** 2)
    sigma0 = (R0[0] * V0[0] + R0[1] * V0[1] + R0[2] * V0[2]) / sp.sqrt(mu)
    r = r0 * U0 + sigma0 * U1 + U2

    aF = 1 - U2 / r0
    aG = (r0 * U1 + sigma0 * U2) / sp.sqrt(mu)
    aFt = -sp.sqrt(mu) / (r0 * r) * U1
    aGt = 1 - U2 / r

    rf = sp.Matrix([aF * R0[i] + aG * V0[i] for i in range(3)])
    vf = sp.Matrix([aFt * R0[i] + aGt * V0[i] for i in range(3)])

    Xs = sp.Matrix(list(R0) + list(V0) + [dt, X, U0, U1, U2])
    Func = sp.Matrix.vstack(rf, vf)
    return TychoHeaderGen(
        "KeplerPrimal_VF",
        Func,
        Xs,
        [(mu, "Gravitational parameter", "mu > 0.0")],
        docstr=("Kepler primal map post-converged: (R0,V0,dt,X*,U0..U2) -> (rf,vf)"),
        gen_build_method=False,
        is_vectorizable=True,
    )


def _kepler_residual():
    """F(R0, V0, dt, U1, U2, U3) -> [F]. 10 inputs, 1 output.

    Note: X is *not* a residual input — F has no explicit X dependence
    (the X-dependence enters only through the U_n via the universal
    functions).  F is linear in (U1, U2, U3).
    """
    R0, V0, dt, _X, _U0, U1, U2, U3, mu = _common_symbols()
    r0 = sp.sqrt(R0[0] ** 2 + R0[1] ** 2 + R0[2] ** 2)
    sigma0 = (R0[0] * V0[0] + R0[1] * V0[1] + R0[2] * V0[2]) / sp.sqrt(mu)
    F = r0 * U1 + sigma0 * U2 + U3 - sp.sqrt(mu) * dt

    Xs = sp.Matrix(list(R0) + list(V0) + [dt, U1, U2, U3])
    return TychoHeaderGen(
        "KeplerResidual_VF",
        sp.Matrix([F]),
        Xs,
        [(mu, "Gravitational parameter", "mu > 0.0")],
        docstr=(
            "Kepler universal-variable residual "
            "F = r0*U1 + sigma0*U2 + U3 - sqrt(mu)*dt"
        ),
        gen_build_method=False,
        is_vectorizable=True,
    )


def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )

    primal = _kepler_primal()
    primal.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_kepler_propagator.py",
    )

    residual = _kepler_residual()
    residual.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_kepler_propagator.py",
    )

    print(f"Generated kepler_primal_vf.h and kepler_residual_vf.h in {output_dir}")


if __name__ == "__main__":
    main()
