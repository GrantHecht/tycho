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
///   if f₁ == f₀: return tdir · max(dtmin, 100·dt₀)   [const-deriv early-out]
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
///   width; `dtmin = nextfloat(max(eps(t0), eps(tf)))` (one ULP above the ULP at
///   the endpoints — OrdinaryDiffEq's nextfloat(max(opts.dtmin, eps(t))) floor)
///   guards against sub-representable steps. The dtmin floor wins over the dtmax
///   cap only in the extreme regime dtmin > dtmax (huge |t| with a sub-ULP span) —
///   matching OrdinaryDiffEq, and harmless because the driver's per-step
///   overshoot clamp trims the first step to tf regardless. Both are computed in
///   positive-magnitude space (like the rest of this routine); `tdir` is applied
///   at return. `tf == t0` (dtmax == 0) returns a zero step (a no-op).
///
/// @note Degenerate scaling is non-throwing (multiple-shooting robustness). If a
///   degenerate per-component scaling (sk[i] → 0) drives the raw estimate to 0 or
///   a non-finite value, this routine returns the finite dtmin floor rather than
///   throwing — an intermediate PSIOPT / multiple-shooting iterate must be able to
///   recover. Only genuinely-unrecoverable *dynamics* (NaN/Inf from f(x0)/f(x1))
///   throw, matching OrdinaryDiffEq's NaN-in-f₀ exit.
template <class DODE, class InputVec, class TolVec>
double estimate_initial_dt(const DODE &ode, const InputVec &x0, double tf, const TolVec &abs_tols,
                           const TolVec &rel_tols, int order, ErrorNormType norm_type) {
    const double t0 = x0[ode.t_var()];
    const double tdir = (tf >= t0) ? 1.0 : -1.0;
    const int n = ode.x_vars();

    // dtmax/dtmin clamps (OrdinaryDiffEq parity, positive-magnitude space):
    // dtmax = |tf - t0| (the integration span — OrdinaryDiffEq's default dtmax);
    // dtmin = nextfloat(max(eps(t0), eps(tf))). OrdinaryDiffEq computes
    // `nextfloat(max(opts.dtmin, eps(t)))` with opts.dtmin defaulting to
    // `prob2dtmin(use_end_time) = max(eps(t0), eps(tf))`, so the effective floor
    // is nextfloat(max(eps(t0), eps(tf))) — the nextfloat bumps it one ULP, which
    // we replicate. tdir is applied to the final magnitude.
    const double dtmax = std::abs(tf - t0);
    const auto ulp = [](double t) {
        const double a = std::abs(t);
        return std::nextafter(a, std::numeric_limits<double>::infinity()) - a;
    };
    const double dtmin =
        std::nextafter(std::max(ulp(t0), ulp(tf)), std::numeric_limits<double>::infinity());
    // smalldt = max(dtmin, 1e-6) (OrdinaryDiffEq parity): the 1e-6 fallback for
    // the d0/d1 < 1e-5 and tiny-derivative branches, floored by dtmin so it never
    // underflows the representable step spacing at large |t|.
    const double smalldt = std::max(dtmin, 1.0e-6);
    // Zero-duration span (tf == t0) is a no-op: there is no step to take, so
    // return a zero step (no integration) rather than computing one. The drivers
    // short-circuit H == 0 and return the initial state before ever reaching
    // here; a direct caller simply gets dt == 0.
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

    // OrdinaryDiffEq parity: if the trial step reproduced the initial derivative
    // exactly (a constant/zero-derivative region — e.g. f ≡ 0 at a fixed point),
    // the second-derivative estimate d2 = ||f1 - f0|| / dt0 is 0/dt0 and carries
    // no curvature information. OrdinaryDiffEq short-circuits with
    // `f₀ == f₁ && return tdir * max(dtmin, 100dt₀)` (note: not clamped to dtmax —
    // the driver's per-step overshoot clamp trims the first step to tf if needed).
    // We replicate exactly.
    if (f0 == f1) {
        return tdir * std::max(dtmin, 100.0 * dt0);
    }

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

    // OrdinaryDiffEq parity + multiple-shooting robustness: return a finite,
    // dtmin-floored step here — do NOT throw. An intermediate PSIOPT /
    // multiple-shooting iterate can transiently produce a degenerate per-component
    // scaling (sk[i] = atol + |x0|*rtol underflowing to 0 when atol[i] == 0 and
    // x0[i] == 0), which drives the raw estimate to 0 or a non-finite value.
    // Aborting would kill a solve PSIOPT could otherwise step away from, so we
    // coerce such an estimate to the dtmin floor — a tiny but finite step the
    // integrator can take and the optimizer can move past — exactly as
    // OrdinaryDiffEq returns tdir*max(dtmin, min(100dt0, dt1, dtmax)). Genuinely
    // unrecoverable *dynamics* (NaN/Inf in f(x0) or f(x1)) still throw above via
    // check_state_finite_or_throw, matching OrdinaryDiffEq's NaN-in-f0 exit —
    // PSIOPT cannot recover from NaN dynamics regardless. The coercion is explicit
    // rather than a reliance on C++ std::max(x, NaN) == x (which is order-dependent
    // and differs from Julia's NaN-propagating max), so the result is
    // deterministic.
    double dt_raw = std::min({100.0 * dt0, dt1, dtmax});
    if (!std::isfinite(dt_raw) || dt_raw <= 0.0) {
        dt_raw = dtmin;
    }
    return tdir * std::max(dtmin, dt_raw);
}

} // namespace tycho::integrators
