// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// Implicit-function-theorem composition layer for the LCD Kepler propagator.
//
// Given an iteration kernel that returns the universal anomaly X* and the
// derived universal functions U_n(α, X*) along with derived state quantities,
// this header composes the structural Jacobians of the codegen primal-state
// VF (S: 11 → 6) and the codegen Kepler-equation residual VF (F: 10 → 1)
// into the total Jacobian d(rf, vf) / d(R0, V0, dt) by chain rule, treating
// X* and U_n as implicit functions of (R0, V0, dt) via:
//
//   F(R0, V0, dt; X*, U_1, U_2, U_3) = 0   (Kepler's eqn in universal vars)
//
// Step 1: get X* by Newton-Raphson on F (handled by kepler_lcd_iterate).
// Step 2: ∂U_n/∂X = U_{n-1}; ∂U_n/∂α follows the recursion
//                 ∂U_n/∂α = (X·U_{n-1} - n·U_n) / (2α) for n ≥ 1.
//         Near α = 0, the recursion has a 1/α singularity that cancels with
//         the numerator; we Taylor-fall back to leading order to avoid the
//         catastrophic cancellation: ∂U_n/∂α |_{α→0} ≈ -X^{n+2} / (n+2)!
// Step 3: dα/dy comes from α = 2/r0 - v0²/μ.
// Step 4: dX*/dy = -dF/dy_total / dF/dX* by IFT, where
//         dF/dy_total = F_struct_y + Σ_{n=1,2,3} F_{Un} · ∂U_n/∂α · ∂α/∂y
//         (F has no explicit X dependence; total dF/dX = r from the kernel).
// Step 5: dS/dy_total = S_struct_y + S_struct_X · dX*/dy
//                     + Σ_{n=0,1,2} S_{Un} · (∂U_n/∂X · dX*/dy + ∂U_n/∂α · ∂α/∂y)
//
// =============================================================================
#pragma once

#include "tycho/detail/astro/kepler_lcd_iterate.h"
#include "tycho/detail/astro/kepler_primal_vf.h"
#include "tycho/detail/astro/kepler_residual_vf.h"
#include "tycho/detail/typedefs/eigen_types.h"

#include <cmath>
#include <cstddef>

namespace tycho::astro::detail {

// -------------------------------------------------------------------------
// Forward declarations
// -------------------------------------------------------------------------

template <class Scalar>
inline void kepler_propagate(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0, Scalar dt,
                             double mu, Vector6<Scalar> &xf);

template <class Scalar>
inline void kepler_propagate_jacobian(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0,
                                      Scalar dt, double mu, Vector6<Scalar> &xf,
                                      Eigen::Matrix<Scalar, 6, 7> &jac);

template <class Scalar>
inline void
kepler_propagate_adjoint_hessian(const Vector3<Scalar> & /*R0*/, const Vector3<Scalar> & /*V0*/,
                                 Scalar /*dt*/, double /*mu*/, const Vector6<Scalar> & /*adjvars*/,
                                 Vector6<Scalar> & /*xf*/, Eigen::Matrix<Scalar, 6, 7> & /*jac*/,
                                 Vector7<Scalar> & /*adjgrad*/,
                                 Eigen::Matrix<Scalar, 7, 7> & /*adjhess*/) {
    static_assert(sizeof(Scalar) == 0,
                  "kepler_propagate_adjoint_hessian: implemented in Task 4 of the LCD plan");
}

// -------------------------------------------------------------------------
// U-derivative recursion helpers
// -------------------------------------------------------------------------

// ∂U_n/∂X = U_{n-1} for n ≥ 1, and ∂U_0/∂X = -α · U_1 (since U_0 = 1 - α U_2
// and ∂U_2/∂X = U_1; equivalently U_0' = -α U_2' = -α U_1).
template <class Scalar>
inline void U_partials_X(Scalar alpha, const KeplerLCDResult<Scalar> &k, Scalar &dU0_dX,
                         Scalar &dU1_dX, Scalar &dU2_dX, Scalar &dU3_dX) {
    dU0_dX = -alpha * k.U1;
    dU1_dX = k.U0;
    dU2_dX = k.U1;
    dU3_dX = k.U2;
}

// ∂U_n/∂α for n ≥ 1 via the recursion (X · U_{n-1} - n · U_n) / (2α).
// ∂U_0/∂α = -X · U_1 / 2 (limit form, no 1/α singularity).
//
// Near |α| < α_tol the recursion has a removable 1/α singularity; substitute
// the Taylor leading order  U_n = X^n/n! - α X^{n+2}/(n+2)! + O(α²)
//   ⇒ ∂U_n/∂α |_{α→0} = -X^{n+2} / (n+2)!
//
// (Scalar = double overload.)
template <class Scalar>
inline void U_partials_alpha(Scalar alpha, Scalar X, const KeplerLCDResult<Scalar> &k,
                             double alpha_tol, Scalar &dU0_da, Scalar &dU1_da, Scalar &dU2_da,
                             Scalar &dU3_da) {
    using std::abs;
    dU0_da = Scalar(-0.5) * X * k.U1;
    if (abs(alpha) > Scalar(alpha_tol)) {
        const Scalar inv2a = Scalar(0.5) / alpha;
        dU1_da = inv2a * (X * k.U0 - Scalar(1) * k.U1);
        dU2_da = inv2a * (X * k.U1 - Scalar(2) * k.U2);
        dU3_da = inv2a * (X * k.U2 - Scalar(3) * k.U3);
    } else {
        const Scalar X2 = X * X;
        const Scalar X3 = X2 * X;
        const Scalar X4 = X2 * X2;
        const Scalar X5 = X3 * X2;
        dU1_da = -X3 * Scalar(1.0 / 6.0);
        dU2_da = -X4 * Scalar(1.0 / 24.0);
        dU3_da = -X5 * Scalar(1.0 / 120.0);
    }
}

// SuperScalar overload for U_partials_alpha.  Per-lane Taylor blending via
// Eigen::Array::select avoids div-by-zero in lanes where |α| < α_tol while
// using the recursion in well-conditioned lanes.
template <int W>
inline void
U_partials_alpha(const Eigen::Array<double, W, 1> &alpha, const Eigen::Array<double, W, 1> &X,
                 const KeplerLCDResult<Eigen::Array<double, W, 1>> &k, double alpha_tol,
                 Eigen::Array<double, W, 1> &dU0_da, Eigen::Array<double, W, 1> &dU1_da,
                 Eigen::Array<double, W, 1> &dU2_da, Eigen::Array<double, W, 1> &dU3_da) {
    using SS = Eigen::Array<double, W, 1>;
    const SS abs_a = alpha.abs();
    const auto is_para = (abs_a <= SS::Constant(alpha_tol));
    // Clamp α to ±α_tol in lanes where Taylor wins to avoid div-by-zero in the
    // recursion branch (the select() picks the Taylor result for those lanes).
    const SS safe_a = is_para.select(SS::Constant(alpha_tol), alpha);
    const SS inv2a = SS::Constant(0.5) / safe_a;

    dU0_da = SS::Constant(-0.5) * X * k.U1;
    const SS rec1 = inv2a * (X * k.U0 - k.U1);
    const SS rec2 = inv2a * (X * k.U1 - SS::Constant(2.0) * k.U2);
    const SS rec3 = inv2a * (X * k.U2 - SS::Constant(3.0) * k.U3);

    const SS X2 = X * X;
    const SS X3 = X2 * X;
    const SS X4 = X3 * X;
    const SS X5 = X4 * X;
    const SS tay1 = SS::Constant(-1.0 / 6.0) * X3;
    const SS tay2 = SS::Constant(-1.0 / 24.0) * X4;
    const SS tay3 = SS::Constant(-1.0 / 120.0) * X5;

    dU1_da = is_para.select(tay1, rec1);
    dU2_da = is_para.select(tay2, rec2);
    dU3_da = is_para.select(tay3, rec3);
}

// -------------------------------------------------------------------------
// Primal-only propagation
// -------------------------------------------------------------------------

template <class Scalar>
inline void kepler_propagate(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0, Scalar dt,
                             double mu, Vector6<Scalar> &xf) {
    using std::sqrt;
    auto k = kepler_lcd_iterate<Scalar>(R0, V0, dt, mu);
    const Scalar sqmu = sqrt(Scalar(mu));
    const Scalar aF = Scalar(1) - k.U2 / k.r0;
    const Scalar aG = (k.r0 * k.U1 + k.sigma0 * k.U2) / sqmu;
    const Scalar aFt = -sqmu / (k.r0 * k.r) * k.U1;
    const Scalar aGt = Scalar(1) - k.U2 / k.r;
    xf.template head<3>() = aF * R0 + aG * V0;
    xf.template tail<3>() = aFt * R0 + aGt * V0;
}

// -------------------------------------------------------------------------
// IFT-composed Jacobian
// -------------------------------------------------------------------------

template <class Scalar>
inline void kepler_propagate_jacobian(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0,
                                      Scalar dt, double mu, Vector6<Scalar> &xf,
                                      Eigen::Matrix<Scalar, 6, 7> &jac) {
    KeplerLCDOptions opts;
    auto k = kepler_lcd_iterate<Scalar>(R0, V0, dt, mu, opts);

    // -- Pack S input vector: (R0(3), V0(3), dt, X, U0, U1, U2)  -> 11 scalars
    Eigen::Matrix<Scalar, 11, 1> sin_;
    sin_.template head<3>() = R0;
    sin_.template segment<3>(3) = V0;
    sin_[6] = dt;
    sin_[7] = k.X;
    sin_[8] = k.U0;
    sin_[9] = k.U1;
    sin_[10] = k.U2;

    // -- Compute structural Jacobian of S (6 × 11)
    ::tycho::astro::KeplerPrimal_VF primal{mu};
    Eigen::Matrix<Scalar, 6, 1> sout;
    Eigen::Matrix<Scalar, 6, 11> S_jac;
    primal.compute_jacobian_impl(sin_, sout, S_jac);
    xf = sout;

    // -- Pack F input vector: (R0(3), V0(3), dt, U1, U2, U3) -> 10 scalars
    Eigen::Matrix<Scalar, 10, 1> fin;
    fin.template head<3>() = R0;
    fin.template segment<3>(3) = V0;
    fin[6] = dt;
    fin[7] = k.U1;
    fin[8] = k.U2;
    fin[9] = k.U3;

    // -- Compute structural Jacobian of F (1 × 10)
    ::tycho::astro::KeplerResidual_VF residual{mu};
    Eigen::Matrix<Scalar, 1, 1> F_val;
    Eigen::Matrix<Scalar, 1, 10> F_jac;
    residual.compute_jacobian_impl(fin, F_val, F_jac);

    // -- U-recursions w.r.t. X and α
    Scalar dU0_dX, dU1_dX, dU2_dX, dU3_dX;
    U_partials_X<Scalar>(k.alpha, k, dU0_dX, dU1_dX, dU2_dX, dU3_dX);
    Scalar dU0_da, dU1_da, dU2_da, dU3_da;
    U_partials_alpha(k.alpha, k.X, k, opts.alpha_tol, dU0_da, dU1_da, dU2_da, dU3_da);

    // -- dα/dy where α = 2/r0 - v0²/μ and y = (R0_x, R0_y, R0_z, V0_x, V0_y, V0_z, dt)
    //    ∂α/∂R0_i = -2 · R0_i / r0³
    //    ∂α/∂V0_i = -2 · V0_i / μ
    //    ∂α/∂dt   =  0
    Vector7<Scalar> dalpha_dy = Vector7<Scalar>::Zero();
    const Scalar r0_3 = k.r0 * k.r0 * k.r0;
    for (int i = 0; i < 3; ++i) {
        dalpha_dy[i] = -Scalar(2) * R0[i] / r0_3;
        dalpha_dy[i + 3] = -Scalar(2) * V0[i] / Scalar(mu);
    }
    // dalpha_dy[6] = 0  (kept from Zero())

    // -- Total dF/dy (F has no explicit X dependence in its structural form; the
    //    only X coupling comes through the (U1, U2, U3) packed inputs).
    //    F input layout: F_jac(0, 0..5)  = ∂F/∂(R0,V0)
    //                    F_jac(0, 6)     = ∂F/∂dt
    //                    F_jac(0, 7..9)  = ∂F/∂(U1, U2, U3)
    Vector7<Scalar> dF_dy_total;
    for (int i = 0; i < 7; ++i) {
        const Scalar dF_struct_y = F_jac(0, i);
        const Scalar dF_chain = F_jac(0, 7) * dU1_da * dalpha_dy[i] +
                                F_jac(0, 8) * dU2_da * dalpha_dy[i] +
                                F_jac(0, 9) * dU3_da * dalpha_dy[i];
        dF_dy_total[i] = dF_struct_y + dF_chain;
    }
    // -- Total dF/dX: F has no explicit X column in its structural Jacobian;
    //    the implicit X-derivative folds through (U1, U2, U3) via ∂U_n/∂X = U_{n-1}.
    //    Substituting and collecting terms gives dF/dX = r (closed-form from the
    //    kernel — this is precisely the F1 = r used as Newton's first-derivative
    //    term during the iteration loop).
    const Scalar F_X_total = k.r;

    // -- IFT: dX*/dy = -dF/dy_total / dF/dX
    Vector7<Scalar> dX_dy;
    for (int i = 0; i < 7; ++i)
        dX_dy[i] = -dF_dy_total[i] / F_X_total;

    // -- Total dS/dy by chain rule.  S input layout:
    //    S_jac(:, 0..5) = ∂S/∂(R0,V0)
    //    S_jac(:, 6)    = ∂S/∂dt
    //    S_jac(:, 7)    = ∂S/∂X
    //    S_jac(:, 8..10)= ∂S/∂(U0, U1, U2)
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
            const Scalar dS_struct_y = S_jac(row, col);
            const Scalar dS_X_path = S_jac(row, 7) * dX_dy[col];
            const Scalar dS_U0_path =
                S_jac(row, 8) * (dU0_dX * dX_dy[col] + dU0_da * dalpha_dy[col]);
            const Scalar dS_U1_path =
                S_jac(row, 9) * (dU1_dX * dX_dy[col] + dU1_da * dalpha_dy[col]);
            const Scalar dS_U2_path =
                S_jac(row, 10) * (dU2_dX * dX_dy[col] + dU2_da * dalpha_dy[col]);
            jac(row, col) = dS_struct_y + dS_X_path + dS_U0_path + dS_U1_path + dS_U2_path;
        }
    }
}

} // namespace tycho::astro::detail
