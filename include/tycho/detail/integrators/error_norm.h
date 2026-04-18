// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <Eigen/Core>
#include <cmath>

namespace tycho::integrators {

/// Error norm kind applied to scaled residuals.
///
///   RMS — sqrt(sum(res_i²) / n) — Julia's ODE_DEFAULT_NORM (default).
///   MAX — max_i |res_i| — Tycho's pre-SP3 default.
///
/// Julia reference: ~/.julia/packages/DiffEqBase/.../common_defaults.jl
/// (`ODE_DEFAULT_NORM` for arrays is RMS).
enum class ErrorNormType { RMS, MAX };

/// Compute element-wise scaled residuals per Julia's convention:
///
///   res_i = ũ_i / (α_i + max(|u₀_i|, |u₁_i|) · ρ_i)
///
/// where α is abs_tols, ρ is rel_tols, u₀ is the pre-step state, u₁ the post-step
/// state. Pre-SP3 Tycho used `|u₁|` alone in the denominator; aligning with Julia
/// is the SP3 change.
///
/// Julia reference: ~/.julia/packages/DiffEqBase/.../calculate_residuals.jl:9-14
template <class Derived1, class Derived2, class Derived3, class Derived4, class Derived5>
inline auto scaled_residuals(const Eigen::MatrixBase<Derived1> &utilde,
                             const Eigen::MatrixBase<Derived2> &u0,
                             const Eigen::MatrixBase<Derived3> &u1,
                             const Eigen::MatrixBase<Derived4> &abs_tols,
                             const Eigen::MatrixBase<Derived5> &rel_tols) {
    return utilde.cwiseQuotient(
        abs_tols + u0.cwiseAbs().cwiseMax(u1.cwiseAbs()).cwiseProduct(rel_tols));
}

/// Reduce a residual vector to a scalar error norm (RMS or MAX).
template <class Derived>
inline double error_norm(const Eigen::MatrixBase<Derived> &res, ErrorNormType type) {
    if (type == ErrorNormType::RMS) {
        const auto n = res.size();
        if (n == 0) return 0.0;
        return std::sqrt(res.squaredNorm() / static_cast<double>(n));
    }
    // MAX
    return res.cwiseAbs().maxCoeff();
}

} // namespace tycho::integrators
