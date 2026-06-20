// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// IFT composition for the LCD Kepler propagator: takes the structural Jacobians
// of the codegen primal VF (S: 11 → 6) and Kepler residual VF (F: 10 → 1) and
// composes them into the total d(rf, vf) / d(R0, V0, dt), treating X* and the
// universal functions U_n(α, X*) as implicit functions of (R0, V0, dt) via
// F(R0, V0, dt; X*, U_n) = 0.  The per-section comments below cover the
// derivation step-by-step, including the Taylor fallback for ∂U_n/∂α near α=0.
// =============================================================================
#pragma once

#include "tycho/detail/astro/kepler/kepler_lcd_iterate.h"
#include "tycho/detail/astro/kepler/kepler_primal.h"
#include "tycho/detail/astro/kepler/kepler_residual.h"
#include "tycho/detail/typedefs/eigen_types.h"

#include <cassert>
#include <cmath>
#include <type_traits>

namespace tycho::astro::detail {

// -------------------------------------------------------------------------
// Internal helpers
// -------------------------------------------------------------------------

// Scalar-polymorphic finite-check for the IFT outputs.  For Scalar=double we
// can call Eigen::Matrix::allFinite() directly; for SS Scalar
// (Eigen::Array<double, W, 1>) the matrix entries are themselves Eigen
// arrays, so allFinite() is not available — iterate entries and AND-reduce
// per-lane finiteness.  Used by kepler_propagate_jacobian /
// kepler_propagate_adjoint_hessian to detect degenerate IFT composition
// (e.g. r → 0 producing inf in 1/F_X_total) on lanes where the LCD kernel
// itself converged.
/// @internal
/// @brief Returns true when every entry of a dense matrix is finite.
///
/// For SuperScalar matrix entries (Eigen::Array<double,W,1>), reduces per-lane
/// finiteness across all entries.  Used to detect degenerate IFT composition.
/// @tparam Derived Eigen dense base type.
/// @param[in] m Matrix to check.
/// @return True if all entries (and all lanes) are finite.
/// @endinternal
template <class Derived> inline bool kepler_block_all_finite(const Eigen::DenseBase<Derived> &m) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, double>) {
        return m.derived().allFinite();
    } else {
        // SS Scalar — iterate entries, AND-reduce lane-finite checks.
        for (Eigen::Index i = 0; i < m.rows(); ++i)
            for (Eigen::Index j = 0; j < m.cols(); ++j)
                if (!m(i, j).isFinite().all())
                    return false;
        return true;
    }
}

// -------------------------------------------------------------------------
// Forward declarations
// -------------------------------------------------------------------------

/// @internal @brief Forward declaration of kepler_propagate (primal only). @endinternal
template <class Scalar>
inline void kepler_propagate(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0, Scalar dt,
                             double mu, Vector6<Scalar> &xf);

/// @internal @brief Forward declaration of kepler_propagate_jacobian (IFT Jacobian). @endinternal
template <class Scalar>
inline void kepler_propagate_jacobian(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0,
                                      Scalar dt, const ::tycho::astro::detail::KeplerPrimal &primal,
                                      const ::tycho::astro::detail::KeplerResidual &residual,
                                      Vector6<Scalar> &xf, Eigen::Matrix<Scalar, 6, 7> &jac);

/// @internal @brief Forward declaration of kepler_propagate_adjoint_hessian (IFT adjoint Hessian). @endinternal
template <class Scalar>
inline void
kepler_propagate_adjoint_hessian(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0, Scalar dt,
                                 const ::tycho::astro::detail::KeplerPrimal &primal,
                                 const ::tycho::astro::detail::KeplerResidual &residual,
                                 const Vector6<Scalar> &adjvars, Vector6<Scalar> &xf,
                                 Eigen::Matrix<Scalar, 6, 7> &jac, Vector7<Scalar> &adjgrad,
                                 Eigen::Matrix<Scalar, 7, 7> &adjhess);

// -------------------------------------------------------------------------
// U-derivative recursion helpers
// -------------------------------------------------------------------------

// Recursion-vs-Taylor cutoff for the IFT-layer α partials.  Distinct from
// the kernel's orbit-type dispatch threshold KeplerLCDOptions::alpha_tol():
// the kernel uses alpha_tol() to choose ellipse / parabola / hyperbola at
// |α| ~ 1e-12 (essentially "is it exactly parabolic?"), but the IFT layer's
// 1/(2α) factors in U_partials_alpha lose precision much earlier — the band
// roughly 1e-12 < |α| < 1e-9 is well-conditioned for the kernel but suspect
// for the IFT recursion form.  Set to match the kernel's Stumpff y→0
// threshold so the Taylor branch covers the whole suspect band.
/// @internal
/// @brief Taylor-fallback threshold for IFT-layer alpha partials near the parabolic branch.
/// @endinternal
inline constexpr double kIFTAlphaTaylorEps = 1.0e-8;

// ∂U_n/∂X = U_{n-1} for n ≥ 1, and ∂U_0/∂X = -α · U_1 (since U_0 = 1 - α U_2
// and ∂U_2/∂X = U_1; equivalently U_0' = -α U_2' = -α U_1).
/// @internal
/// @brief Compute first-order Stumpff partials w.r.t. the universal anomaly X.
/// @tparam Scalar Numeric scalar type.
/// @param[in] alpha Vis-viva energy parameter.
/// @param[in] k     LCD result containing converged U values.
/// @return Stumpff bundle { dU0/dX, dU1/dX, dU2/dX, dU3/dX }.
/// @endinternal
template <class Scalar>
inline Stumpff<Scalar> U_partials_X(Scalar alpha, const KeplerLCDResult<Scalar> &k) {
    return {-alpha * k.U.U1, k.U.U0, k.U.U1, k.U.U2};
}

// ∂U_n/∂α for n ≥ 1 via the recursion (X · U_{n-1} - n · U_n) / (2α).
// ∂U_0/∂α = -X · U_1 / 2 (limit form, no 1/α singularity).
//
// Near |α| < α_tol the recursion has a removable 1/α singularity; substitute
// the Taylor leading order  U_n = X^n/n! - α X^{n+2}/(n+2)! + O(α²)
//   ⇒ ∂U_n/∂α |_{α→0} = -X^{n+2} / (n+2)!
//
/// @internal
/// @brief Compute first-order Stumpff partials w.r.t. the energy parameter alpha (scalar path).
///
/// Near |alpha| < alpha_tol uses Taylor leading-order values to avoid the removable 1/alpha
/// singularity in the recursion form.
/// @tparam Scalar Numeric scalar type (double).
/// @param[in] alpha     Vis-viva energy parameter.
/// @param[in] X         Converged universal anomaly.
/// @param[in] k         LCD result containing converged U values.
/// @param[in] alpha_tol Threshold for Taylor fallback.
/// @return Stumpff bundle { dU0/dalpha, dU1/dalpha, dU2/dalpha, dU3/dalpha }.
/// @endinternal
template <class Scalar>
inline Stumpff<Scalar> U_partials_alpha(Scalar alpha, Scalar X, const KeplerLCDResult<Scalar> &k,
                                        double alpha_tol) {
    using std::abs;
    Stumpff<Scalar> out;
    out.U0 = Scalar(-0.5) * X * k.U.U1;
    if (abs(alpha) > Scalar(alpha_tol)) {
        const Scalar inv2a = Scalar(0.5) / alpha;
        out.U1 = inv2a * (X * k.U.U0 - Scalar(1) * k.U.U1);
        out.U2 = inv2a * (X * k.U.U1 - Scalar(2) * k.U.U2);
        out.U3 = inv2a * (X * k.U.U2 - Scalar(3) * k.U.U3);
    } else {
        const Scalar X2 = X * X;
        const Scalar X3 = X2 * X;
        const Scalar X4 = X2 * X2;
        const Scalar X5 = X3 * X2;
        out.U1 = -X3 * Scalar(1.0 / 6.0);
        out.U2 = -X4 * Scalar(1.0 / 24.0);
        out.U3 = -X5 * Scalar(1.0 / 120.0);
    }
    return out;
}

/// @internal
/// @brief SuperScalar overload of U_partials_alpha with per-lane Taylor blending.
/// @tparam W SIMD width.
/// @param[in] alpha     Vis-viva energy parameter (W-lane).
/// @param[in] X         Converged universal anomaly (W-lane).
/// @param[in] k         LCD result containing converged U values (W-lane).
/// @param[in] alpha_tol Threshold for Taylor fallback.
/// @return W-lane Stumpff bundle { dU0/dalpha, dU1/dalpha, dU2/dalpha, dU3/dalpha }.
/// @endinternal
template <int W>
inline Stumpff<Eigen::Array<double, W, 1>>
U_partials_alpha(const Eigen::Array<double, W, 1> &alpha, const Eigen::Array<double, W, 1> &X,
                 const KeplerLCDResult<Eigen::Array<double, W, 1>> &k, double alpha_tol) {
    using SS = Eigen::Array<double, W, 1>;
    const SS abs_a = alpha.abs();
    const auto is_para = (abs_a <= SS::Constant(alpha_tol));
    // Clamp α to ±α_tol in lanes where Taylor wins to avoid div-by-zero in the
    // recursion branch (the select() picks the Taylor result for those lanes).
    const SS safe_a = is_para.select(SS::Constant(alpha_tol), alpha);
    const SS inv2a = SS::Constant(0.5) / safe_a;

    Stumpff<SS> out;
    out.U0 = SS::Constant(-0.5) * X * k.U.U1;
    const SS rec1 = inv2a * (X * k.U.U0 - k.U.U1);
    const SS rec2 = inv2a * (X * k.U.U1 - SS::Constant(2.0) * k.U.U2);
    const SS rec3 = inv2a * (X * k.U.U2 - SS::Constant(3.0) * k.U.U3);

    const SS X2 = X * X;
    const SS X3 = X2 * X;
    const SS X4 = X3 * X;
    const SS X5 = X4 * X;
    const SS tay1 = SS::Constant(-1.0 / 6.0) * X3;
    const SS tay2 = SS::Constant(-1.0 / 24.0) * X4;
    const SS tay3 = SS::Constant(-1.0 / 120.0) * X5;

    out.U1 = is_para.select(tay1, rec1);
    out.U2 = is_para.select(tay2, rec2);
    out.U3 = is_para.select(tay3, rec3);
    return out;
}

// -------------------------------------------------------------------------
// Second-order U-partials (∂²U_n/∂X², ∂²U_n/∂X∂α, ∂²U_n/∂α²)
// -------------------------------------------------------------------------
//
// All three blocks follow by differentiating the first-order recursions one
// more time.  X-X is straightforward (∂U_n/∂X = U_{n-1}, so ∂²U_n/∂X² =
// ∂U_{n-1}/∂X).  X-α and α-α both produce 1/α factors that share the same
// removable singularity as the first-order α-recursion: near α=0 we Taylor-
// fall back to the leading series term to avoid catastrophic cancellation.
// 4×3 second-partials table: rows index U_n (n = 0..3), columns index the
// partial kind (X², X·α, α²).  Replaces the loose 12-field layout — the
// named accessors expose the natural row-by-Stumpff-index, column-by-
// partial-kind structure that downstream consumers (the second-order IFT
// chain rule) read in groups.  data_ is private + accessed only via the
// named accessors so the row/column convention can't be bypassed by a
// later refactor.
/// @internal
/// @brief Named 4×3 table of second-order Stumpff partials (∂²U_n/∂X², ∂²U_n/∂X∂α, ∂²U_n/∂α²).
///
/// Rows index U_n (n=0..3); columns index the partial kind (dX2, dXda, da2).
/// @tparam Scalar Numeric scalar type.
/// @endinternal
template <class Scalar> class U_second_partials {
  public:
    /// @internal @brief Column index enum for the three second-partial kinds. @endinternal
    enum Kind : int { dX2 = 0, dXda = 1, da2 = 2 };

    /// @internal @brief Mutable accessor for ∂²U_n/∂X². @param[in] n Stumpff index (0–3). @return Reference to the entry. @endinternal
    [[nodiscard]] Scalar &d2U_dX2(int n) {
        assert(0 <= n && n < 4);
        return data_(n, dX2);
    }
    /// @internal @brief Mutable accessor for ∂²U_n/∂X∂α. @param[in] n Stumpff index (0–3). @return Reference to the entry. @endinternal
    [[nodiscard]] Scalar &d2U_dXda(int n) {
        assert(0 <= n && n < 4);
        return data_(n, dXda);
    }
    /// @internal @brief Mutable accessor for ∂²U_n/∂α². @param[in] n Stumpff index (0–3). @return Reference to the entry. @endinternal
    [[nodiscard]] Scalar &d2U_da2(int n) {
        assert(0 <= n && n < 4);
        return data_(n, da2);
    }
    /// @internal @brief Const accessor for ∂²U_n/∂X². @param[in] n Stumpff index (0–3). @return Const reference to the entry. @endinternal
    [[nodiscard]] const Scalar &d2U_dX2(int n) const {
        assert(0 <= n && n < 4);
        return data_(n, dX2);
    }
    /// @internal @brief Const accessor for ∂²U_n/∂X∂α. @param[in] n Stumpff index (0–3). @return Const reference to the entry. @endinternal
    [[nodiscard]] const Scalar &d2U_dXda(int n) const {
        assert(0 <= n && n < 4);
        return data_(n, dXda);
    }
    /// @internal @brief Const accessor for ∂²U_n/∂α². @param[in] n Stumpff index (0–3). @return Const reference to the entry. @endinternal
    [[nodiscard]] const Scalar &d2U_da2(int n) const {
        assert(0 <= n && n < 4);
        return data_(n, da2);
    }

  private:
    Eigen::Matrix<Scalar, 4, 3> data_;
};

/// @internal
/// @brief Compute the 4×3 table of second-order Stumpff partials (scalar path).
///
/// Near |alpha| < alpha_tol uses Taylor leading-order values for X-alpha and alpha-alpha
/// partials to avoid the removable 1/alpha singularity.
/// @tparam Scalar Numeric scalar type (double).
/// @param[in] alpha     Vis-viva energy parameter.
/// @param[in] X         Converged universal anomaly.
/// @param[in] k         LCD result containing converged U values.
/// @param[in] alpha_tol Threshold for Taylor fallback.
/// @param[in] dU_dX     First-order X-partials from U_partials_X.
/// @param[in] dU_da     First-order alpha-partials from U_partials_alpha.
/// @return Second-order partial table.
/// @endinternal
template <class Scalar>
inline U_second_partials<Scalar>
compute_U_second_partials(Scalar alpha, Scalar X, const KeplerLCDResult<Scalar> &k,
                          double alpha_tol, const Stumpff<Scalar> &dU_dX,
                          const Stumpff<Scalar> &dU_da) {
    using std::abs;
    U_second_partials<Scalar> p;

    // ∂²U_n/∂X² = ∂(∂U_{n-1}/∂X)/∂X.  Recursing one level on the X-recursion:
    //   U_0' = -α U_1  ⇒  U_0'' = -α U_0
    //   U_1' = U_0     ⇒  U_1'' = -α U_1
    //   U_2' = U_1     ⇒  U_2'' =  U_0
    //   U_3' = U_2     ⇒  U_3'' =  U_1
    p.d2U_dX2(0) = -alpha * k.U.U0;
    p.d2U_dX2(1) = -alpha * k.U.U1;
    p.d2U_dX2(2) = k.U.U0;
    p.d2U_dX2(3) = k.U.U1;

    // ∂²U_n/∂X∂α: differentiate dU_n/dα w.r.t. X.
    //   For n = 0: dU_0/dα = -X U_1 / 2  ⇒  ∂/∂X = -U_1/2 - X (∂U_1/∂X)/2
    //   For n ≥ 1: dU_n/dα = (X U_{n-1} - n U_n) / (2α)
    //              ⇒ ∂/∂X = (U_{n-1} + X (∂U_{n-1}/∂X) - n (∂U_n/∂X)) / (2α)
    p.d2U_dXda(0) = Scalar(-0.5) * (k.U.U1 + X * dU_dX.U1);
    if (abs(alpha) > Scalar(alpha_tol)) {
        const Scalar inv2a = Scalar(0.5) / alpha;
        p.d2U_dXda(1) = inv2a * (k.U.U0 + X * dU_dX.U0 - Scalar(1) * dU_dX.U1);
        p.d2U_dXda(2) = inv2a * (k.U.U1 + X * dU_dX.U1 - Scalar(2) * dU_dX.U2);
        p.d2U_dXda(3) = inv2a * (k.U.U2 + X * dU_dX.U2 - Scalar(3) * dU_dX.U3);

        // ∂²U_n/∂α²: differentiate dU_n/dα w.r.t. α (n ≥ 1).
        //   dU_n/dα = (X U_{n-1} - n U_n) / (2α)
        //   ⇒ ∂/∂α = (X (∂U_{n-1}/∂α) - n (∂U_n/∂α)) / (2α) - dU_n/dα / α
        // For n = 0: dU_0/dα = -X U_1 / 2  ⇒  ∂/∂α = -X (∂U_1/∂α) / 2.
        p.d2U_da2(0) = Scalar(-0.5) * X * dU_da.U1;
        p.d2U_da2(1) = inv2a * (X * dU_da.U0 - Scalar(1) * dU_da.U1) - dU_da.U1 / alpha;
        p.d2U_da2(2) = inv2a * (X * dU_da.U1 - Scalar(2) * dU_da.U2) - dU_da.U2 / alpha;
        p.d2U_da2(3) = inv2a * (X * dU_da.U2 - Scalar(3) * dU_da.U3) - dU_da.U3 / alpha;
    } else {
        // Taylor leading terms.  From U_n = X^n/n! - α X^{n+2}/(n+2)! + α² X^{n+4}/(n+4)! - ...,
        //   ∂U_n/∂α |_{α→0}      = -X^{n+2}/(n+2)!
        //   ∂²U_n/∂α² |_{α→0}    =  2 X^{n+4}/(n+4)!
        //   ∂²U_n/∂X∂α |_{α→0}   = -∂_X(X^{n+2}/(n+2)!) = -X^{n+1}/(n+1)!
        const Scalar X2 = X * X;
        const Scalar X3 = X2 * X;
        const Scalar X4 = X2 * X2;
        const Scalar X5 = X3 * X2;
        const Scalar X6 = X3 * X3;
        const Scalar X7 = X4 * X3;
        p.d2U_dXda(1) = Scalar(-1.0 / 2.0) * X2;
        p.d2U_dXda(2) = Scalar(-1.0 / 6.0) * X3;
        p.d2U_dXda(3) = Scalar(-1.0 / 24.0) * X4;
        p.d2U_da2(0) = Scalar(2.0 / 24.0) * X4;
        p.d2U_da2(1) = Scalar(2.0 / 120.0) * X5;
        p.d2U_da2(2) = Scalar(2.0 / 720.0) * X6;
        p.d2U_da2(3) = Scalar(2.0 / 5040.0) * X7;
    }
    return p;
}

/// @internal
/// @brief SuperScalar overload of compute_U_second_partials with per-lane Taylor blending.
/// @tparam W SIMD width.
/// @param[in] alpha     Vis-viva energy parameter (W-lane).
/// @param[in] X         Converged universal anomaly (W-lane).
/// @param[in] k         LCD result containing converged U values (W-lane).
/// @param[in] alpha_tol Threshold for Taylor fallback.
/// @param[in] dU_dX     First-order X-partials (W-lane).
/// @param[in] dU_da     First-order alpha-partials (W-lane).
/// @return W-lane second-order partial table.
/// @endinternal
template <int W>
inline U_second_partials<Eigen::Array<double, W, 1>>
compute_U_second_partials(const Eigen::Array<double, W, 1> &alpha,
                          const Eigen::Array<double, W, 1> &X,
                          const KeplerLCDResult<Eigen::Array<double, W, 1>> &k, double alpha_tol,
                          const Stumpff<Eigen::Array<double, W, 1>> &dU_dX,
                          const Stumpff<Eigen::Array<double, W, 1>> &dU_da) {
    using SS = Eigen::Array<double, W, 1>;
    U_second_partials<SS> p;

    // ∂²U_n/∂X² (no singularity).
    p.d2U_dX2(0) = -alpha * k.U.U0;
    p.d2U_dX2(1) = -alpha * k.U.U1;
    p.d2U_dX2(2) = k.U.U0;
    p.d2U_dX2(3) = k.U.U1;

    // Per-lane Taylor blend mask + safe inverse.
    const SS abs_a = alpha.abs();
    const auto is_para = (abs_a <= SS::Constant(alpha_tol));
    const SS safe_a = is_para.select(SS::Constant(alpha_tol), alpha);
    const SS inv2a = SS::Constant(0.5) / safe_a;
    const SS inv_a = SS::Constant(1.0) / safe_a;

    // Recursion-branch values (X-α partials, n ≥ 1).
    const SS rec_dU1_dXda = inv2a * (k.U.U0 + X * dU_dX.U0 - dU_dX.U1);
    const SS rec_dU2_dXda = inv2a * (k.U.U1 + X * dU_dX.U1 - SS::Constant(2.0) * dU_dX.U2);
    const SS rec_dU3_dXda = inv2a * (k.U.U2 + X * dU_dX.U2 - SS::Constant(3.0) * dU_dX.U3);

    // Recursion-branch values (α-α partials, n ≥ 1).
    const SS rec_dU1_da2 = inv2a * (X * dU_da.U0 - dU_da.U1) - dU_da.U1 * inv_a;
    const SS rec_dU2_da2 = inv2a * (X * dU_da.U1 - SS::Constant(2.0) * dU_da.U2) - dU_da.U2 * inv_a;
    const SS rec_dU3_da2 = inv2a * (X * dU_da.U2 - SS::Constant(3.0) * dU_da.U3) - dU_da.U3 * inv_a;

    // Taylor-leading-term branch values.
    const SS X2 = X * X;
    const SS X3 = X2 * X;
    const SS X4 = X3 * X;
    const SS X5 = X4 * X;
    const SS X6 = X5 * X;
    const SS X7 = X6 * X;
    const SS tay_dU1_dXda = SS::Constant(-1.0 / 2.0) * X2;
    const SS tay_dU2_dXda = SS::Constant(-1.0 / 6.0) * X3;
    const SS tay_dU3_dXda = SS::Constant(-1.0 / 24.0) * X4;
    const SS tay_dU0_da2 = SS::Constant(2.0 / 24.0) * X4;
    const SS tay_dU1_da2 = SS::Constant(2.0 / 120.0) * X5;
    const SS tay_dU2_da2 = SS::Constant(2.0 / 720.0) * X6;
    const SS tay_dU3_da2 = SS::Constant(2.0 / 5040.0) * X7;

    // n = 0 partials (no singularity, common for both branches).
    p.d2U_dXda(0) = SS::Constant(-0.5) * (k.U.U1 + X * dU_dX.U1);
    // For α-α: dU_0/dα = -X U_1 / 2  ⇒  ∂/∂α = -X (∂U_1/∂α) / 2 (recursion side);
    // Taylor side gives 2 X^4 / 24 (from leading series).  Blend per-lane.
    const SS rec_dU0_da2 = SS::Constant(-0.5) * X * dU_da.U1;
    p.d2U_da2(0) = is_para.select(tay_dU0_da2, rec_dU0_da2);

    p.d2U_dXda(1) = is_para.select(tay_dU1_dXda, rec_dU1_dXda);
    p.d2U_dXda(2) = is_para.select(tay_dU2_dXda, rec_dU2_dXda);
    p.d2U_dXda(3) = is_para.select(tay_dU3_dXda, rec_dU3_dXda);

    p.d2U_da2(1) = is_para.select(tay_dU1_da2, rec_dU1_da2);
    p.d2U_da2(2) = is_para.select(tay_dU2_da2, rec_dU2_da2);
    p.d2U_da2(3) = is_para.select(tay_dU3_da2, rec_dU3_da2);

    return p;
}

// -------------------------------------------------------------------------
// Primal-only propagation
// -------------------------------------------------------------------------

/// @internal
/// @brief Propagate Cartesian state using the LCD kernel (primal only, no derivatives).
///
/// On any non-convergence NaN-poisons xf entirely (all-or-nothing contract).
/// @tparam Scalar Numeric scalar type (double or SuperScalar).
/// @param[in]  R0  Initial position vector.
/// @param[in]  V0  Initial velocity vector.
/// @param[in]  dt  Time-of-flight.
/// @param[in]  mu  Gravitational parameter.
/// @param[out] xf  Propagated 6-vector [rx, ry, rz, vx, vy, vz].
/// @endinternal
template <class Scalar>
inline void kepler_propagate(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0, Scalar dt,
                             double mu, Vector6<Scalar> &xf) {
    using std::sqrt;
    auto k = kepler_lcd_iterate(R0, V0, dt, mu);
    // SS reduction is all-or-nothing: when any lane fails to converge,
    // every lane's output is NaN-poisoned, not just the failing lane.
    // This matches PSIOPT's step-rejection semantics — the whole micro-
    // batch is rejected — and avoids the implementation complexity of
    // mixing converged-lane outputs with NaN-poisoned ones.  The same
    // pattern is repeated in kepler_propagate_jacobian and
    // kepler_propagate_adjoint_hessian below.
    if (!all_converged(k.converged)) {
        xf.setConstant(kepler_nan_value<Scalar>());
        return;
    }
    const Scalar sqmu = sqrt(Scalar(mu));
    const Scalar aF = Scalar(1) - k.U.U2 / k.r0;
    const Scalar aG = (k.r0 * k.U.U1 + k.sigma0 * k.U.U2) / sqmu;
    const Scalar aFt = -sqmu / (k.r0 * k.r) * k.U.U1;
    const Scalar aGt = Scalar(1) - k.U.U2 / k.r;
    xf.template head<3>() = aF * R0 + aG * V0;
    xf.template tail<3>() = aFt * R0 + aGt * V0;
}

// -------------------------------------------------------------------------
// IFT-composed Jacobian
// -------------------------------------------------------------------------

/// @internal
/// @brief Propagate Cartesian state and compute its 6×7 Jacobian via IFT composition.
///
/// On any non-convergence or post-IFT non-finite result, NaN-poisons both xf and jac.
/// @tparam Scalar Numeric scalar type (double or SuperScalar).
/// @param[in]  R0       Initial position vector.
/// @param[in]  V0       Initial velocity vector.
/// @param[in]  dt       Time-of-flight.
/// @param[in]  primal   KeplerPrimal VF (carries precomputed sqrt(mu) cache).
/// @param[in]  residual KeplerResidual VF (carries precomputed sqrt(mu) cache).
/// @param[out] xf       Propagated 6-vector [rx, ry, rz, vx, vy, vz].
/// @param[out] jac      6×7 Jacobian d(rf,vf)/d(R0,V0,dt).
/// @endinternal
template <class Scalar>
inline void kepler_propagate_jacobian(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0,
                                      Scalar dt, const ::tycho::astro::detail::KeplerPrimal &primal,
                                      const ::tycho::astro::detail::KeplerResidual &residual,
                                      Vector6<Scalar> &xf, Eigen::Matrix<Scalar, 6, 7> &jac) {
    KeplerLCDOptions opts;
    const double mu = primal.mu();
    auto k = kepler_lcd_iterate(R0, V0, dt, mu, opts);
    // SS all-or-nothing NaN-poisoning — see kepler_propagate header above.
    if (!all_converged(k.converged)) {
        xf.setConstant(kepler_nan_value<Scalar>());
        jac.setConstant(kepler_nan_value<Scalar>());
        return;
    }

    // -- Pack S input vector: (R0(3), V0(3), dt, X, U0, U1, U2)  -> 11 scalars
    Eigen::Matrix<Scalar, 11, 1> sin_;
    sin_.template head<3>() = R0;
    sin_.template segment<3>(3) = V0;
    sin_[6] = dt;
    sin_[7] = k.X;
    sin_[8] = k.U.U0;
    sin_[9] = k.U.U1;
    sin_[10] = k.U.U2;

    // -- Compute structural Jacobian of S (6 × 11) via the wrapper-owned VF.
    Eigen::Matrix<Scalar, 6, 1> sout;
    Eigen::Matrix<Scalar, 6, 11> S_jac;
    primal.compute_jacobian_impl(sin_, sout, S_jac);
    xf = sout;

    // -- Pack F input vector: (R0(3), V0(3), dt, U1, U2, U3) -> 10 scalars
    Eigen::Matrix<Scalar, 10, 1> fin;
    fin.template head<3>() = R0;
    fin.template segment<3>(3) = V0;
    fin[6] = dt;
    fin[7] = k.U.U1;
    fin[8] = k.U.U2;
    fin[9] = k.U.U3;

    // -- Compute structural Jacobian of F (1 × 10) via the wrapper-owned VF.
    Eigen::Matrix<Scalar, 1, 1> F_val;
    Eigen::Matrix<Scalar, 1, 10> F_jac;
    residual.compute_jacobian_impl(fin, F_val, F_jac);

    // -- U-recursions w.r.t. X and α
    const Stumpff<Scalar> dU_dX = U_partials_X<Scalar>(k.alpha, k);
    const Stumpff<Scalar> dU_da = U_partials_alpha(k.alpha, k.X, k, kIFTAlphaTaylorEps);

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

    // -- Total dF/dy (F has no explicit X dependence in its structural form; the
    //    only X coupling comes through the (U1, U2, U3) packed inputs).
    //    F input layout: F_jac(0, 0..5)  = ∂F/∂(R0,V0)
    //                    F_jac(0, 6)     = ∂F/∂dt
    //                    F_jac(0, 7..9)  = ∂F/∂(U1, U2, U3)
    Vector7<Scalar> dF_dy_total;
    for (int i = 0; i < 7; ++i) {
        const Scalar dF_struct_y = F_jac(0, i);
        const Scalar dF_chain = F_jac(0, 7) * dU_da.U1 * dalpha_dy[i] +
                                F_jac(0, 8) * dU_da.U2 * dalpha_dy[i] +
                                F_jac(0, 9) * dU_da.U3 * dalpha_dy[i];
        dF_dy_total[i] = dF_struct_y + dF_chain;
    }
    // -- Total dF/dX: F has no explicit X column in its structural Jacobian;
    //    the implicit X-derivative folds through (U1, U2, U3) via ∂U_n/∂X = U_{n-1}.
    //    Substituting and collecting terms gives dF/dX = r (closed-form from the
    //    kernel — this is the same F1 = r factor the LCD update uses as f'(X)
    //    inside the iteration loop).
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
                S_jac(row, 8) * (dU_dX.U0 * dX_dy[col] + dU_da.U0 * dalpha_dy[col]);
            const Scalar dS_U1_path =
                S_jac(row, 9) * (dU_dX.U1 * dX_dy[col] + dU_da.U1 * dalpha_dy[col]);
            const Scalar dS_U2_path =
                S_jac(row, 10) * (dU_dX.U2 * dX_dy[col] + dU_da.U2 * dalpha_dy[col]);
            jac(row, col) = dS_struct_y + dS_X_path + dS_U0_path + dS_U1_path + dS_U2_path;
        }
    }

    // Post-composition finite-guard.  The LCD kernel can converge to a
    // numerically-degenerate state (e.g. r → 0 producing inf in
    // 1/F_X_total = 1/r) where the IFT-derived xf / jac are non-finite
    // even though k.converged was true.  PSIOPT step-rejection requires
    // an observable NaN; mirror the all-or-nothing contract above.
    if (!kepler_block_all_finite(xf) || !kepler_block_all_finite(jac)) [[unlikely]] {
        xf.setConstant(kepler_nan_value<Scalar>());
        jac.setConstant(kepler_nan_value<Scalar>());
    }
}

// -------------------------------------------------------------------------
// IFT-composed adjoint Hessian
// -------------------------------------------------------------------------
//
// Returns the total state, the total 6×7 Jacobian, the adjoint gradient
// J^T·λ, and the 7×7 adjoint Hessian H = ∇²(λᵀ·xf).
//
// Derivation: extend the first-order chain rule of `kepler_propagate_jacobian`
// to second order.  The implicit-function theorem (IFT) determines X*(y) by
// F(y, X*(y), U_n(X*, α(y))) = 0, where the U_n's are themselves implicit in
// X and α (which is itself a function of y).  The structural codegen gives us
// the second-order partials of S and F in (y, X, U_n) space.  We compose:
//
//   (1) total dα/dy and d²α/dy² (closed form from α = 2/r0 - v0²/μ).
//   (2) total dU_n/dy and d²U_n/dy² (chain rule through X and α).
//   (3) total dX*/dy via first-order IFT (= -F_y_total / F_X_total).
//   (4) total d²X*/dy² via second-order IFT (uses F_yy_total etc.).
//   (5) S''_{ij} via second-order chain rule, then contract with adjvars.
//
// The codegen call returns S_grad and S_hess already contracted with adjvars
// (so S_hess(i,j) = Σ_row adjvars[row]·∂²S_row/(∂x_i ∂x_j) in the 11-dim
// structural input space).  F is single-output, so passing F_lm[0] = 1 makes
// F_grad and F_hess equal to the raw structural gradient/Hessian of F.
/// @internal
/// @brief Propagate Cartesian state and compute its Jacobian, adjoint gradient, and adjoint Hessian.
///
/// Implements the full second-order IFT chain rule.  On any non-convergence or
/// post-IFT non-finite result, NaN-poisons all four output buffers.
/// @tparam Scalar Numeric scalar type (double or SuperScalar).
/// @param[in]  R0       Initial position vector.
/// @param[in]  V0       Initial velocity vector.
/// @param[in]  dt       Time-of-flight.
/// @param[in]  primal   KeplerPrimal VF.
/// @param[in]  residual KeplerResidual VF.
/// @param[in]  adjvars  Adjoint variables lambda (6-vector).
/// @param[out] xf       Propagated 6-vector.
/// @param[out] jac      6×7 Jacobian.
/// @param[out] adjgrad  Adjoint gradient J^T*lambda (7-vector).
/// @param[out] adjhess  7×7 adjoint Hessian.
/// @endinternal
template <class Scalar>
inline void
kepler_propagate_adjoint_hessian(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0, Scalar dt,
                                 const ::tycho::astro::detail::KeplerPrimal &primal,
                                 const ::tycho::astro::detail::KeplerResidual &residual,
                                 const Vector6<Scalar> &adjvars, Vector6<Scalar> &xf,
                                 Eigen::Matrix<Scalar, 6, 7> &jac, Vector7<Scalar> &adjgrad,
                                 Eigen::Matrix<Scalar, 7, 7> &adjhess) {
    using std::abs;
    KeplerLCDOptions opts;
    const double mu = primal.mu();
    auto k = kepler_lcd_iterate(R0, V0, dt, mu, opts);
    // SS all-or-nothing NaN-poisoning — see kepler_propagate header above.
    if (!all_converged(k.converged)) {
        xf.setConstant(kepler_nan_value<Scalar>());
        jac.setConstant(kepler_nan_value<Scalar>());
        adjgrad.setConstant(kepler_nan_value<Scalar>());
        adjhess.setConstant(kepler_nan_value<Scalar>());
        return;
    }

    // --- Step A: codegen call for S (primal) — 11-dim structural input,
    //     6-dim output, adjvar-contracted gradient and Hessian.  The
    //     wrapper-owned VF carries the precomputed sqrt(mu) cache so we
    //     avoid the per-call construction cost.
    Eigen::Matrix<Scalar, 11, 1> sin_;
    sin_.template head<3>() = R0;
    sin_.template segment<3>(3) = V0;
    sin_[6] = dt;
    sin_[7] = k.X;
    sin_[8] = k.U.U0;
    sin_[9] = k.U.U1;
    sin_[10] = k.U.U2;

    Vector6<Scalar> sout;
    Eigen::Matrix<Scalar, 6, 11> S_jac;
    Vector<Scalar, 11> S_grad;
    Eigen::Matrix<Scalar, 11, 11> S_hess;
    primal.compute_jacobian_adjointgradient_adjointhessian_impl(sin_, sout, S_jac, S_grad, S_hess,
                                                                adjvars);
    xf = sout;

    // --- Step B: codegen call for F (residual) — 10-dim structural input,
    //     1-dim output.  F_lm[0]=1 makes F_grad / F_hess the raw gradient
    //     and Hessian of F.
    Eigen::Matrix<Scalar, 10, 1> fin;
    fin.template head<3>() = R0;
    fin.template segment<3>(3) = V0;
    fin[6] = dt;
    fin[7] = k.U.U1;
    fin[8] = k.U.U2;
    fin[9] = k.U.U3;

    Eigen::Matrix<Scalar, 1, 1> F_val;
    Eigen::Matrix<Scalar, 1, 10> F_jac;
    Vector<Scalar, 10> F_grad;
    Eigen::Matrix<Scalar, 10, 10> F_hess;
    Vector<Scalar, 1> F_lm;
    F_lm[0] = Scalar(1);
    residual.compute_jacobian_adjointgradient_adjointhessian_impl(fin, F_val, F_jac, F_grad, F_hess,
                                                                  F_lm);

    // --- Step C: U partials (1st and 2nd order).
    const Stumpff<Scalar> dU_dX = U_partials_X<Scalar>(k.alpha, k);
    const Stumpff<Scalar> dU_da = U_partials_alpha(k.alpha, k.X, k, kIFTAlphaTaylorEps);
    auto p2 = compute_U_second_partials(k.alpha, k.X, k, kIFTAlphaTaylorEps, dU_dX, dU_da);

    // --- Step D: α partials (1st and 2nd order).
    //   α = 2/r0 - v0²/μ.
    //   ∂α/∂R0_i = -2 R0_i / r0³
    //   ∂α/∂V0_i = -2 V0_i / μ
    //   ∂²α/∂R0_i∂R0_j = -2(δ_ij r0² - 3 R0_i R0_j) / r0⁵
    //   ∂²α/∂V0_i∂V0_j = -2 δ_ij / μ
    //   All other 2nd-order α partials vanish.
    const Scalar r0 = k.r0;
    const Scalar r0_2 = r0 * r0;
    const Scalar r0_3 = r0_2 * r0;
    const Scalar r0_5 = r0_3 * r0_2;
    Vector7<Scalar> dalpha_dy = Vector7<Scalar>::Zero();
    for (int i = 0; i < 3; ++i) {
        dalpha_dy[i] = -Scalar(2) * R0[i] / r0_3;
        dalpha_dy[i + 3] = -Scalar(2) * V0[i] / Scalar(mu);
    }
    Eigen::Matrix<Scalar, 7, 7> d2alpha_dy2 = Eigen::Matrix<Scalar, 7, 7>::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const Scalar delta_ij = (i == j ? Scalar(1) : Scalar(0));
            d2alpha_dy2(i, j) = -Scalar(2) * (delta_ij * r0_2 - Scalar(3) * R0[i] * R0[j]) / r0_5;
        }
    }
    for (int i = 0; i < 3; ++i)
        d2alpha_dy2(i + 3, i + 3) = -Scalar(2) / Scalar(mu);

    // --- Step E: first-order IFT.
    Vector7<Scalar> dF_dy_total;
    for (int i = 0; i < 7; ++i) {
        const Scalar dF_struct_y = F_jac(0, i);
        const Scalar dF_chain = F_jac(0, 7) * dU_da.U1 * dalpha_dy[i] +
                                F_jac(0, 8) * dU_da.U2 * dalpha_dy[i] +
                                F_jac(0, 9) * dU_da.U3 * dalpha_dy[i];
        dF_dy_total[i] = dF_struct_y + dF_chain;
    }
    const Scalar F_X_total = k.r;
    Vector7<Scalar> dX_dy;
    for (int i = 0; i < 7; ++i)
        dX_dy[i] = -dF_dy_total[i] / F_X_total;

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
            const Scalar dS_struct_y = S_jac(row, col);
            const Scalar dS_X_path = S_jac(row, 7) * dX_dy[col];
            const Scalar dS_U0_path =
                S_jac(row, 8) * (dU_dX.U0 * dX_dy[col] + dU_da.U0 * dalpha_dy[col]);
            const Scalar dS_U1_path =
                S_jac(row, 9) * (dU_dX.U1 * dX_dy[col] + dU_da.U1 * dalpha_dy[col]);
            const Scalar dS_U2_path =
                S_jac(row, 10) * (dU_dX.U2 * dX_dy[col] + dU_da.U2 * dalpha_dy[col]);
            jac(row, col) = dS_struct_y + dS_X_path + dS_U0_path + dS_U1_path + dS_U2_path;
        }
    }
    adjgrad = jac.transpose() * adjvars;

    // --- Step F: second-order IFT — adjhess.

    // (F.1) Total first derivatives of U_n (n ∈ 0..3) w.r.t. y_i.
    Eigen::Matrix<Scalar, 4, 7> dU;
    {
        const Scalar dUdX[4] = {dU_dX.U0, dU_dX.U1, dU_dX.U2, dU_dX.U3};
        const Scalar dUda[4] = {dU_da.U0, dU_da.U1, dU_da.U2, dU_da.U3};
        for (int n = 0; n < 4; ++n)
            for (int i = 0; i < 7; ++i)
                dU(n, i) = dUdX[n] * dX_dy[i] + dUda[n] * dalpha_dy[i];
    }

    // The IFT formula for X''_{ij} requires partials of F̃(y, X) at FIXED X
    // (only y varies) — NOT total derivatives.  Since F̃(y, X) =
    // F_struct(y, U_n(X, α(y))) and F_struct is linear in (U_1, U_2, U_3),
    // these partials simplify substantially.
    //
    // (F.3.a.i) F̃_XX|_fixed_X = Σ_n F_struct_{Un} · ∂²U_n/∂X² = σ
    //          (closed-form from the kernel; equals the LCD iteration's
    //          f''(X) = σ).
    const Scalar F_XX_total = k.sigma;

    // (F.3.a.ii) F̃_yX(i)|_fixed_X = ∂²F̃/(∂y_i ∂X).
    //   F̃_X = Σ_n F_struct_{Un} · ∂U_n/∂X.
    //   ∂/∂y_i at fixed X: only α and explicit y-dependence vary, so
    //     F̃_yX(i)|_X = Σ_n [F_hess(U+n, i) · ∂U_n/∂X
    //                      + F_struct_{Un} · ∂²U_n/(∂X∂α) · α'_i]
    //   The U-U cross piece F_hess(U+n, U+m)·(∂U_m/∂α·α'_i)·∂U_n/∂X is 0
    //   because F is linear in U_n.  Crucially, no X² term appears here:
    //   this is a *partial* at fixed X.
    Eigen::Matrix<Scalar, 1, 7> F_yX_total;
    {
        const int U_start = 7;
        const Scalar dUdX_n[3] = {dU_dX.U1, dU_dX.U2, dU_dX.U3};
        const Scalar d2U_dXda_n[3] = {p2.d2U_dXda(1), p2.d2U_dXda(2), p2.d2U_dXda(3)};
        for (int i = 0; i < 7; ++i) {
            Scalar acc = Scalar(0);
            for (int n = 0; n < 3; ++n) {
                acc += F_hess(U_start + n, i) * dUdX_n[n] +
                       F_jac(0, U_start + n) * d2U_dXda_n[n] * dalpha_dy[i];
            }
            F_yX_total(0, i) = acc;
        }
    }

    // (F.3.a.iii) F̃_yy(i, j)|_fixed_X = ∂²F̃/(∂y_i ∂y_j).
    //   F̃_y_i = F_struct_{y_i} + Σ_n F_struct_{Un} · ∂U_n/∂α · α'_i
    //   ∂/∂y_j at fixed X expands to (using F linear in U for the U-U cross):
    //     F_hess(i, j)
    //   + Σ_n [F_hess(i, U+n)·∂U_n/∂α·α'_j + F_hess(j, U+n)·∂U_n/∂α·α'_i]   (y-U cross)
    //   + Σ_n F_struct_{Un} · ∂²U_n/∂α² · α'_i · α'_j                       (α-α through U)
    //   + Σ_n F_struct_{Un} · ∂U_n/∂α · α''_{ij}                            (closed-form α'')
    Eigen::Matrix<Scalar, 7, 7> F_yy_total;
    {
        const int U_start = 7;
        const Scalar dUda_n[3] = {dU_da.U1, dU_da.U2, dU_da.U3};
        const Scalar d2Uda2_n[3] = {p2.d2U_da2(1), p2.d2U_da2(2), p2.d2U_da2(3)};
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j <= i; ++j) {
                Scalar val = F_hess(i, j);
                for (int n = 0; n < 3; ++n) {
                    val += F_hess(i, U_start + n) * dUda_n[n] * dalpha_dy[j] +
                           F_hess(j, U_start + n) * dUda_n[n] * dalpha_dy[i];
                    val += F_jac(0, U_start + n) * d2Uda2_n[n] * dalpha_dy[i] * dalpha_dy[j];
                    val += F_jac(0, U_start + n) * dUda_n[n] * d2alpha_dy2(i, j);
                }
                F_yy_total(i, j) = val;
                F_yy_total(j, i) = val;
            }
        }
    }

    // (F.3.b) X''_{ij} from second-order IFT.  Differentiating
    //   F̃(y, X*(y)) ≡ 0 twice:
    //     0 = F̃_yy + F̃_yX·X'_j + F̃_yX·X'_i + F̃_XX·X'_i·X'_j + F̃_X·X''_{ij}
    //   ⇒ X''_{ij} = -(F̃_yy + F̃_yX·X' (sym) + F̃_XX·X'·X') / F̃_X
    //   where all F̃ partials are at fixed X.
    Eigen::Matrix<Scalar, 7, 7> X_pp;
    const Scalar F_X_total_inv = Scalar(1) / F_X_total;
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j <= i; ++j) {
            const Scalar v = F_yy_total(i, j) + F_yX_total(0, i) * dX_dy[j] +
                             F_yX_total(0, j) * dX_dy[i] + F_XX_total * dX_dy[i] * dX_dy[j];
            X_pp(i, j) = -v * F_X_total_inv;
            X_pp(j, i) = X_pp(i, j);
        }
    }

    // (F.3.c) Total second derivatives of U_n w.r.t. (y_i, y_j) using X''.
    //   d²U_n/(dy_i dy_j) = ∂²U_n/∂X² · X'_i X'_j
    //                     + ∂²U_n/∂X∂α · (X'_i α'_j + X'_j α'_i)
    //                     + ∂²U_n/∂α² · α'_i α'_j
    //                     + ∂U_n/∂X · X''_{ij}
    //                     + ∂U_n/∂α · α''_{ij}
    Eigen::Matrix<Scalar, 4, 49> d2U_flat;
    auto d2U_idx = [](int i, int j) { return i * 7 + j; };
    {
        const Scalar d2UdX2_n[4] = {p2.d2U_dX2(0), p2.d2U_dX2(1), p2.d2U_dX2(2), p2.d2U_dX2(3)};
        const Scalar d2UdXda_n[4] = {p2.d2U_dXda(0), p2.d2U_dXda(1), p2.d2U_dXda(2),
                                     p2.d2U_dXda(3)};
        const Scalar d2Uda2_n[4] = {p2.d2U_da2(0), p2.d2U_da2(1), p2.d2U_da2(2), p2.d2U_da2(3)};
        const Scalar dUdX_n[4] = {dU_dX.U0, dU_dX.U1, dU_dX.U2, dU_dX.U3};
        const Scalar dUda_n[4] = {dU_da.U0, dU_da.U1, dU_da.U2, dU_da.U3};
        for (int n = 0; n < 4; ++n) {
            for (int i = 0; i < 7; ++i) {
                for (int j = 0; j <= i; ++j) {
                    const Scalar v =
                        d2UdX2_n[n] * dX_dy[i] * dX_dy[j] +
                        d2UdXda_n[n] * (dX_dy[i] * dalpha_dy[j] + dX_dy[j] * dalpha_dy[i]) +
                        d2Uda2_n[n] * dalpha_dy[i] * dalpha_dy[j] + dUdX_n[n] * X_pp(i, j) +
                        dUda_n[n] * d2alpha_dy2(i, j);
                    d2U_flat(n, d2U_idx(i, j)) = v;
                    d2U_flat(n, d2U_idx(j, i)) = v;
                }
            }
        }
    }

    // (F.3.d) Adjoint Hessian.  S_hess (and S_grad) are returned ALREADY
    // contracted with adjvars by the codegen, so:
    //   S_hess(i, j) = Σ_row adjvars[row] · ∂²S_row/(∂x_i ∂x_j)   (in 11-D x).
    // We need to compose total ∂²/(∂y_i ∂y_j) of (Σ_row adjvars[row]·S_row).
    // Expanding ∂(λᵀS)/∂y_i = (λᵀS)_y_i + (λᵀS)_X·X'_i + Σ_n (λᵀS)_{Un}·dU(n,i)
    // and differentiating again:
    //
    //   adjhess(i,j) = S_hess(i,j)                                  // y-y
    //                + S_hess(7,i)·X'_j + S_hess(7,j)·X'_i          // y-X
    //                + S_hess(7,7)·X'_i·X'_j                        // X-X
    //                + Σ_n [ S_hess(8+n,i)·dU(n,j) + S_hess(8+n,j)·dU(n,i)   // y-U
    //                      + S_hess(7,8+n)·(X'_i·dU(n,j)+X'_j·dU(n,i)) ]    // X-U
    //                + Σ_{n,m} S_hess(8+n,8+m)·dU(n,i)·dU(m,j)      // U-U
    //                + adj_S_X · X''_{ij}                           // 1st-order chain on X
    //                + Σ_n adj_S_U[n] · d²U_n/(dy_i dy_j)           // 1st-order chain on U
    //
    // where adj_S_X = Σ_row adjvars[row]·S_jac(row,7)
    //   and adj_S_U[n] = Σ_row adjvars[row]·S_jac(row, 8+n).
    Scalar adj_S_X = Scalar(0);
    for (int row = 0; row < 6; ++row)
        adj_S_X += adjvars[row] * S_jac(row, 7);
    Scalar adj_S_U[3];
    for (int n = 0; n < 3; ++n) {
        adj_S_U[n] = Scalar(0);
        for (int row = 0; row < 6; ++row)
            adj_S_U[n] += adjvars[row] * S_jac(row, 8 + n);
    }

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar v = S_hess(i, j) + S_hess(7, i) * dX_dy[j] + S_hess(7, j) * dX_dy[i] +
                       S_hess(7, 7) * dX_dy[i] * dX_dy[j] + adj_S_X * X_pp(i, j);
            for (int n = 0; n < 3; ++n) {
                v += S_hess(8 + n, i) * dU(n, j) + S_hess(8 + n, j) * dU(n, i) +
                     S_hess(7, 8 + n) * (dX_dy[i] * dU(n, j) + dX_dy[j] * dU(n, i)) +
                     adj_S_U[n] * d2U_flat(n, d2U_idx(i, j));
            }
            for (int n = 0; n < 3; ++n)
                for (int m = 0; m < 3; ++m)
                    v += S_hess(8 + n, 8 + m) * dU(n, i) * dU(m, j);
            adjhess(i, j) = v;
            adjhess(j, i) = v;
        }
    }

    // Post-composition finite-guard — see kepler_propagate_jacobian's
    // analogous block.  The Hessian path adds 1/(2α) factors in the U-α
    // recursion and dot-product contractions that can amplify finite-but-
    // ill-conditioned IFT inputs to non-finite outputs.  PSIOPT step-
    // rejection requires an observable NaN on every output of the call.
    if (!kepler_block_all_finite(xf) || !kepler_block_all_finite(jac) ||
        !kepler_block_all_finite(adjgrad) || !kepler_block_all_finite(adjhess)) [[unlikely]] {
        xf.setConstant(kepler_nan_value<Scalar>());
        jac.setConstant(kepler_nan_value<Scalar>());
        adjgrad.setConstant(kepler_nan_value<Scalar>());
        adjhess.setConstant(kepler_nan_value<Scalar>());
    }
}

} // namespace tycho::astro::detail
