// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/error_norm.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>

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
///   dt₀ = min(dt₀, dtmax)                        [clamp — see dtmax/dtmin note]
///   u₁ = u0 + tdir·dt₀ · f₀
///   f₁ = f(u₁, t+tdir·dt₀)
///   d₂ = norm((f₁-f₀) ./ sk) / dt₀
///   if max(d₁,d₂) ≤ 1e-15: dt₁ = max(smalldt, 1e-3 · dt₀)
///   else:                  dt₁ = (0.01 / max(d₁,d₂))^(1/(order+1))
///   dt  = tdir · max(dtmin, min(100·dt₀, dt₁, dtmax))   [clamp — see note below]
///
/// Direction: sign(tf - t0). Atol/Rtol are per-component.
///
/// @note Exponent convention — matches OrdinaryDiffEq.jl, NOT the Hairer
///   textbook. The `order` argument passed by the drivers is the *embedded*
///   error-estimator order (ErrorOrder = p − 1 with p the method order), so the
///   exponent `1/(order+1)` equals `1/p`. OrdinaryDiffEq's `ode_determine_initdt`
///   uses `dt1 = (0.01/max(d1,d2))^(1/order)` with `order =
///   get_current_alg_order(alg) = alg_order = p`, i.e. also `1/p`. So this
///   implementation is exponent-identical to OrdinaryDiffEq (DOPRI54 → 1/5).
///   Hairer, Nørsett & Wanner (II.4.14) print `1/(p+1)`; OrdinaryDiffEq
///   deliberately deviates to `1/p`, and Tycho follows OrdinaryDiffEq. Do NOT
///   "correct" this to `1/(order+2)`: it enlarges the first step and drives
///   stiff/bang-bang problems (e.g. GoddardRocket) into a non-recovering
///   rejection loop.
///
/// @note dtmax/dtmin clamping — matches OrdinaryDiffEq's `ode_determine_initdt`:
///   `dt₀ = min(dt₀, dtmax)`, and the returned magnitude is
///   `max(dtmin, min(100·dt₀, dt₁, dtmax))`. `dtmax = |tf − t0|` (OrdinaryDiffEq's
///   default: the integration span) so the first step is capped at the interval
///   width; `dtmin = max(eps(t0), eps(tf))` (the ULP at the endpoints) floors it
///   against sub-representable steps. The dtmin floor wins over the dtmax cap
///   only in the extreme regime dtmin > dtmax (huge |t| with a sub-ULP span) —
///   matching OrdinaryDiffEq, and harmless because the driver's per-step
///   overshoot clamp trims the first step to tf regardless. Both are computed in
///   positive-magnitude space (like the rest of this routine); `tdir` is applied
///   at return. `tf == t0` (dtmax == 0) returns a zero step (a no-op).
template <class DODE, class InputVec, class TolVec>
double estimate_initial_dt(const DODE &ode, const InputVec &x0, double tf, const TolVec &abs_tols,
                           const TolVec &rel_tols, int order, ErrorNormType norm_type) {
    const double t0 = x0[ode.t_var()];
    const double tdir = (tf >= t0) ? 1.0 : -1.0;
    const int n = ode.x_vars();
    constexpr double smalldt = 1.0e-6;

    // dtmax/dtmin clamps (OrdinaryDiffEq parity, positive-magnitude space):
    // dtmax = |tf - t0| (the integration span — OrdinaryDiffEq's default dtmax);
    // dtmin = max(eps(t0), eps(tf)) (ULP at the endpoints, == Julia's
    // prob2dtmin with use_end_time). tdir is applied to the final magnitude.
    const double dtmax = std::abs(tf - t0);
    const auto ulp = [](double t) {
        const double a = std::abs(t);
        return std::nextafter(a, std::numeric_limits<double>::infinity()) - a;
    };
    const double dtmin = std::max(ulp(t0), ulp(tf));
    // Zero-duration span (tf == t0) is a no-op: there is no step to take, so
    // return a zero step (no integration) rather than computing one. This also
    // avoids the 0/0 that would otherwise arise below (dt0 clamped to
    // dtmax == 0 -> d2 = norm/dt0 = NaN, which std::max/std::min silently drop by
    // argument order). The drivers short-circuit H == 0 and return the initial
    // state before ever reaching here; a direct caller simply gets dt == 0.
    if (dtmax == 0.0) {
        return 0.0;
    }

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
    // Clamp dt0 to dtmax before the trial step (matches OrdinaryDiffEq, which
    // clamps prior to evaluating f at the trial point x1).
    dt0 = std::min(dt0, dtmax);

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

    // Raw Hairer-Wanner estimate (positive magnitude) BEFORE the dtmin floor.
    // The finite/nonzero guard below must see this raw value, not the floored
    // one: a degenerate per-component scaling — sk[i] = atol + |x0|*rtol
    // underflowing to 0 on some component — drives the raw estimate to 0 or
    // NaN (e.g. under MAX norm, maxCoeff drops a NaN that isn't the first
    // element, leaving a finite dt0 but an Inf d2 -> dt1 == 0 -> raw == 0).
    // Applying std::max(dtmin, …) first would launder that into a silent
    // ~1-ULP step, masking the exact tolerance misconfiguration this
    // diagnostic names. Guard first, floor second.
    double dt_raw = std::min({100.0 * dt0, dt1, dtmax});
    double signed_dt = tdir * dt_raw;
    if (!std::isfinite(signed_dt) || signed_dt == 0.0) {
        // Non-finite or exactly-zero estimate is almost always the user setting
        // both abs_tol and rel_tol to zero on a component where x0 is zero —
        // the scaling vector sk[i] = atol + |x0|*rtol goes to zero, and the
        // scaled residuals become 0/0 = NaN or a/0 = Inf. The adaptive driver
        // rejects abs_tol + rel_tol <= 0 upstream, but that check passes when
        // rel_tol > 0 and x0[i] == 0 (sk[i] still underflows to 0); this guards
        // that remaining FP edge.
        throw std::runtime_error(
            "Hairer-Wanner initial-dt estimator produced a non-finite or zero step: dt=" +
            std::to_string(signed_dt) + " (d0=" + std::to_string(d0) +
            ", d1=" + std::to_string(d1) + ", d2=" + std::to_string(d2) +
            "). Check per-component abs_tol / rel_tol and the initial state.");
    }
    // Apply the OrdinaryDiffEq dtmin floor only to a legitimately finite,
    // positive estimate (a genuinely-stiff sub-dtmin step).
    return tdir * std::max(dtmin, dt_raw);
}

} // namespace tycho::integrators
