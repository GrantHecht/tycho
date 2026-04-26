// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tycho::integrators {

/// Throw a descriptive runtime_error if any element of `v` is non-finite.
///
/// `site` names the production site (e.g. "AdaptiveDriver::stepper.step"); `t` is the
/// current independent-variable value; `h` is the step size in flight.
/// `trajectory_idx` should be set to the lane/trajectory index in batch
/// paths, or left at -1 for the scalar path.
///
/// The check is unconditional: there is no opt-out. Cost is one fused
/// vectorized scan (~size * 1ns); for typical state sizes this is
/// well below 0.1% of step time. The check exists to convert silent
/// NaN-driven step rejection (which would otherwise march to the
/// max_steps cap with a generic "stuck" error) into an immediate,
/// localized diagnostic naming the offending state component.
template <class Derived>
inline void check_state_finite_or_throw(const Eigen::MatrixBase<Derived> &v, double t, double h,
                                        const char *site, int trajectory_idx = -1) {
    if (v.allFinite())
        return;
    Eigen::Index bad = -1;
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (!std::isfinite(static_cast<double>(v[i]))) {
            bad = i;
            break;
        }
    }
    std::string msg = "Non-finite state produced by ";
    msg += site;
    msg += " at t=" + std::to_string(t) + " (h=" + std::to_string(h) + ")";
    if (trajectory_idx >= 0)
        msg += " trajectory=" + std::to_string(trajectory_idx);
    msg += "; first non-finite component index = " + std::to_string(bad);
    msg += ". This usually indicates the ODE produced NaN/Inf in its derivative; "
           "check the dynamics at this state for divisions by zero, sqrt of negatives, "
           "or other ill-defined operations.";
    throw std::runtime_error(msg);
}

/// Error norm kind applied to scaled residuals.
///
///   RMS — sqrt(sum(res_i²) / n) — Julia's ODE_DEFAULT_NORM (default).
///   MAX — max_i |res_i|.
enum class ErrorNormType { RMS, MAX };

/// Compute element-wise scaled residuals per Julia's convention:
///
///   res_i = ũ_i / (α_i + max(|u₀_i|, |u₁_i|) · ρ_i)
///
/// where α is abs_tols, ρ is rel_tols, u₀ is the pre-step state, u₁ the post-step
/// state (matches Julia DiffEqBase `calculate_residuals`).
template <class Derived1, class Derived2, class Derived3, class Derived4, class Derived5>
inline auto
scaled_residuals(const Eigen::MatrixBase<Derived1> &utilde, const Eigen::MatrixBase<Derived2> &u0,
                 const Eigen::MatrixBase<Derived3> &u1, const Eigen::MatrixBase<Derived4> &abs_tols,
                 const Eigen::MatrixBase<Derived5> &rel_tols) {
    return utilde.cwiseQuotient(abs_tols +
                                u0.cwiseAbs().cwiseMax(u1.cwiseAbs()).cwiseProduct(rel_tols));
}

/// Reduce a residual vector to a scalar error norm (RMS or MAX).
template <class Derived>
inline double error_norm(const Eigen::MatrixBase<Derived> &res, ErrorNormType type) {
    if (type == ErrorNormType::RMS) {
        const auto n = res.size();
        if (n == 0)
            return 0.0;
        return std::sqrt(res.squaredNorm() / static_cast<double>(n));
    }
    // MAX
    return res.cwiseAbs().maxCoeff();
}

} // namespace tycho::integrators
