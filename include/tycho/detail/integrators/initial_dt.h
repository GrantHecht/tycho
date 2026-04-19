// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/error_norm.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace tycho::integrators {

/// Hairer-Wanner initial-step estimate.
///
/// Two-stage algorithm from Hairer, Nørsett & Wanner (1993, Section II.4),
/// implemented by OrdinaryDiffEq.jl's `_ode_initdt_oop`/`_ode_initdt_iip`.
///
///   sk_i = atol_i + |u0_i| · rtol_i
///   d₀ = norm(u0 ./ sk)
///   d₁ = norm(f₀ ./ sk)
///   if d₀ < 1e-5 or d₁ < 1e-5: dt₀ = 1e-6
///   else:                      dt₀ = 0.01 · (d₀/d₁)
///   u₁ = u0 + tdir·dt₀ · f₀
///   f₁ = f(u₁, t+tdir·dt₀)
///   d₂ = norm((f₁-f₀) ./ sk) / dt₀
///   if max(d₁,d₂) ≤ 1e-15: dt₁ = max(smalldt, 1e-3 · dt₀)
///   else:                  dt₁ = (0.01 / max(d₁,d₂))^(1/(order+1))
///   dt  = tdir · min(100·dt₀, dt₁)
///
/// Direction: sign(tf - t0). Atol/Rtol are per-component.
///
/// OOP scalar variant — no diffusion/g term (Julia's `g==nothing` branch),
/// matching Tycho's deterministic ODE subsystem.
template <class DODE, class InputVec, class TolVec>
double estimate_initial_dt(const DODE &ode, const InputVec &x0, double tf, const TolVec &abs_tols,
                           const TolVec &rel_tols, int order, ErrorNormType norm_type) {
    const double t0 = x0[ode.t_var()];
    const double tdir = (tf >= t0) ? 1.0 : -1.0;
    const int n = ode.x_vars();
    constexpr double smalldt = 1.0e-6;

    Eigen::VectorXd sk(n);
    for (int i = 0; i < n; ++i) {
        sk[i] = abs_tols[i] + std::abs(x0[i]) * rel_tols[i];
    }

    typename DODE::template Output<double> f0(ode.output_rows());
    f0.setZero();
    ode.compute(x0, f0);
    check_state_finite_or_throw(f0.head(n), t0, 0.0,
                                "ode.compute (Hairer-Wanner initial-dt: f(x0))");

    Eigen::VectorXd scaled0(n), scaled1(n);
    for (int i = 0; i < n; ++i) {
        scaled0[i] = x0[i] / sk[i];
        scaled1[i] = f0[i] / sk[i];
    }
    double d0 = error_norm(scaled0, norm_type);
    double d1 = error_norm(scaled1, norm_type);

    double dt0;
    if (d0 < 1e-5 || d1 < 1e-5) {
        dt0 = smalldt;
    } else {
        dt0 = 0.01 * d0 / d1;
    }

    typename DODE::template Input<double> x1 = x0;
    for (int i = 0; i < n; ++i) {
        x1[i] = x0[i] + tdir * dt0 * f0[i];
    }
    x1[ode.t_var()] = t0 + tdir * dt0;

    typename DODE::template Output<double> f1(ode.output_rows());
    f1.setZero();
    ode.compute(x1, f1);
    check_state_finite_or_throw(f1.head(n), t0 + tdir * dt0, dt0,
                                "ode.compute (Hairer-Wanner initial-dt: f(x1))");

    for (int i = 0; i < n; ++i) {
        scaled1[i] = (f1[i] - f0[i]) / sk[i];
    }
    double d2 = error_norm(scaled1, norm_type) / dt0;

    double max_d1d2 = std::max(d1, d2);
    double dt1;
    if (max_d1d2 <= 1e-15) {
        dt1 = std::max(smalldt, dt0 * 1e-3);
    } else {
        dt1 = std::pow(0.01 / max_d1d2, 1.0 / (static_cast<double>(order) + 1.0));
    }

    double dt = std::min(100.0 * dt0, dt1);
    return tdir * dt;
}

} // namespace tycho::integrators
