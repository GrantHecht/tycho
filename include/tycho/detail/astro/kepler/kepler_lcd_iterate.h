#pragma once
#include "tycho/detail/typedefs/eigen_types.h"
#include <Eigen/Geometry>
#include <cmath>
#include <numbers>

namespace tycho::astro::detail {

struct KeplerLCDOptions {
    double Xtol = 1.0e-12;
    double alpha_tol = 1.0e-12;
    int max_order = 10;
    int iters_per_order = 10;
    double hyp_guard = 30.0;
};

template <class Scalar> struct KeplerLCDResult {
    Scalar X;
    Scalar alpha;
    Scalar r0;
    Scalar sigma0;
    Scalar r;
    Scalar sigma;
    Scalar U0, U1, U2, U3;
    bool converged;
};

// compute_universal_functions: orbit-type branch lives here.
// alpha > alpha_tol  -> ellipse via Stumpff
// alpha < -alpha_tol -> hyperbola via cosh/sinh
// |alpha| <= alpha_tol -> parabola
template <class Scalar>
inline void compute_universal_functions(Scalar alpha, Scalar X, double alpha_tol, Scalar &U0,
                                        Scalar &U1, Scalar &U2, Scalar &U3);

template <class Scalar>
KeplerLCDResult<Scalar> kepler_lcd_iterate(const Vector3<Scalar> &R0, const Vector3<Scalar> &V0,
                                           Scalar dt, double mu, const KeplerLCDOptions &opts = {});

// Why the orbit-type branch: the universal-variable Stumpff functions C(y) and
// S(y) develop a 0/0 indeterminacy as alpha (and hence y = alpha*X^2) crosses
// zero, so we dispatch on the sign of alpha and use the closed-form cosh/sinh
// expressions on the hyperbolic branch and the limit values on the parabolic
// branch.
template <>
inline void compute_universal_functions<double>(double alpha, double X, double alpha_tol,
                                                double &U0, double &U1, double &U2, double &U3) {
    using std::cos;
    using std::cosh;
    using std::sin;
    using std::sinh;
    using std::sqrt;
    if (alpha > alpha_tol) { // ellipse
        const double y = alpha * X * X;
        const double sq = sqrt(y);
        const double C = (1.0 - cos(sq)) / y;
        const double S = (sq - sin(sq)) / (sq * y);
        U1 = X * (1.0 - y * S);
        U2 = X * X * C;
        U3 = X * X * X * S;
        U0 = 1.0 - alpha * U2;
    } else if (alpha < -alpha_tol) { // hyperbola
        const double sqma = sqrt(-alpha);
        const double sqmaX = sqma * X;
        U0 = cosh(sqmaX);
        U1 = sinh(sqmaX) / sqma;
        U2 = (1.0 - U0) / alpha;
        U3 = (X - U1) / alpha;
    } else { // parabola
        U0 = 1.0;
        U1 = X;
        U2 = U1 * X / 2.0;
        U3 = U2 * X / 3.0;
    }
}

template <>
inline KeplerLCDResult<double> kepler_lcd_iterate<double>(const Vector3<double> &R0,
                                                          const Vector3<double> &V0, double dt,
                                                          double mu, const KeplerLCDOptions &opts) {
    using std::fabs;
    using std::log;
    using std::sqrt;
    KeplerLCDResult<double> r;
    r.converged = true;
    const double sqmu = sqrt(mu);
    const double r0 = R0.norm();
    const double v2 = V0.squaredNorm();
    const double sigma0 = R0.dot(V0) / sqmu;
    const double alpha = 2.0 / r0 - v2 / mu;
    r.r0 = r0;
    r.sigma0 = sigma0;
    r.alpha = alpha;

    if (dt == 0.0) {
        r.X = 0.0;
        // U_n(X=0) is the same on every branch: U0=1, U1=U2=U3=0.  Set directly
        // to avoid the 0/0 indeterminacy in the ellipse Stumpff form at y=0.
        r.U0 = 1.0;
        r.U1 = 0.0;
        r.U2 = 0.0;
        r.U3 = 0.0;
        r.r = r0;
        r.sigma = sigma0;
        return r;
    }

    double X;
    if (alpha > opts.alpha_tol) {
        X = sqmu * dt * alpha;
    } else if (alpha < -opts.alpha_tol) {
        const double a = 1.0 / alpha;
        const double sgn_dt = dt >= 0.0 ? 1.0 : -1.0;
        X = sgn_dt * sqrt(-a) *
            log(fabs((-2.0 * mu * alpha * dt) /
                     (R0.dot(V0) + sgn_dt * sqrt(-mu * a) * (1.0 - r0 * alpha))));
    } else {
        const Vector3<double> h = R0.cross(V0);
        const double hmag = h.norm();
        const double p = hmag * hmag / mu;
        const double s =
            (std::numbers::pi / 2.0 - std::atan(3.0 * sqrt(mu / (p * p * p)) * dt)) / 2.0;
        const double w = std::atan(std::cbrt(std::tan(s)));
        X = sqrt(p) * 2.0 / std::tan(2.0 * w);
    }

    double X_new = X;
    int N = 1;
    int iters_this_N = 0;
    double dX = 1e+100;
    double U0 = 0, U1 = 0, U2 = 0, U3 = 0;
    double rad = r0, sig = sigma0;
    double F = 0;

    bool guard_tripped = false;
    while (fabs(dX) > opts.Xtol && N <= opts.max_order) {
        X = X_new;

        // Why the asymptote guard: on a hyperbolic orbit the universal anomaly
        // grows logarithmically with time, but cosh/sinh blow up exponentially
        // in sqrt(-alpha)*X.  If the iterate ever overshoots past the guard,
        // U0..U3 lose precision before the iteration can recover, so we bail.
        if (alpha < -opts.alpha_tol) {
            const double sqma = sqrt(-alpha);
            if (fabs(sqma * X) > opts.hyp_guard) {
                guard_tripped = true;
                break;
            }
        }

        compute_universal_functions<double>(alpha, X, opts.alpha_tol, U0, U1, U2, U3);
        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (1.0 - alpha * r0) * U1;
        F = r0 * U1 + sigma0 * U2 + U3 - sqmu * dt;
        const double F1 = rad;
        const double F2 = sig;

        const double sgn = F1 >= 0 ? 1.0 : -1.0;
        const double disc = (N - 1) * (N - 1) * F1 * F1 - N * (N - 1) * F * F2;
        const double denom = fabs(disc);
        if (denom > 0.0)
            dX = N * F / (F1 + sgn * sqrt(denom));
        else
            dX = F / F1;
        X_new = X - dX;

        // Counter bookkeeping done after the work so each order N actually
        // gets opts.iters_per_order updates before promotion.  The order-cap
        // condition is re-checked by the while header on the next pass.
        ++iters_this_N;
        if (iters_this_N >= opts.iters_per_order) {
            ++N;
            iters_this_N = 0;
        }
    }
    if (guard_tripped || fabs(dX) > opts.Xtol)
        r.converged = false;

    // Only refresh the final U/r/sigma at X_new on a clean convergence exit.
    // On guard- or order-cap bailout, X_new may be in the regime that tripped
    // the guard, so reusing the last in-loop values (computed at the most
    // recent stable X) is the safer reporting choice.
    if (r.converged) {
        X = X_new;
        compute_universal_functions<double>(alpha, X, opts.alpha_tol, U0, U1, U2, U3);
        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (1.0 - alpha * r0) * U1;
    }

    r.X = X;
    r.U0 = U0;
    r.U1 = U1;
    r.U2 = U2;
    r.U3 = U3;
    r.r = rad;
    r.sigma = sig;
    return r;
}

template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>>
kepler_lcd_iterate_per_lane(const Vector3<Eigen::Array<double, W, 1>> &R0,
                            const Vector3<Eigen::Array<double, W, 1>> &V0,
                            const Eigen::Array<double, W, 1> &dt, double mu,
                            const KeplerLCDOptions &opts) {
    using SS = Eigen::Array<double, W, 1>;
    KeplerLCDResult<SS> out;
    out.converged = true;
    for (int lane = 0; lane < W; ++lane) {
        Vector3<double> R0_l, V0_l;
        for (int i = 0; i < 3; ++i) {
            R0_l[i] = R0[i][lane];
            V0_l[i] = V0[i][lane];
        }
        auto k = kepler_lcd_iterate<double>(R0_l, V0_l, dt[lane], mu, opts);
        out.X[lane] = k.X;
        out.alpha[lane] = k.alpha;
        out.r0[lane] = k.r0;
        out.sigma0[lane] = k.sigma0;
        out.r[lane] = k.r;
        out.sigma[lane] = k.sigma;
        out.U0[lane] = k.U0;
        out.U1[lane] = k.U1;
        out.U2[lane] = k.U2;
        out.U3[lane] = k.U3;
        out.converged = out.converged && k.converged;
    }
    return out;
}

// True-SIMD Laguerre-Conway-Der iteration for the case where all W lanes are
// elliptic (alpha[i] > opts.alpha_tol) and all have nonzero dt.  The caller is
// responsible for verifying the precondition; this routine does not re-check.
//
// The iteration body is fully packed across lanes — no per-lane scalar
// dispatch.  Lanes that converge first are kept in an "active" mask: they
// continue to ride through the SIMD body but no longer update X_new, so the
// SIMD lane keeps its converged value.  The shared order N is promoted whenever
// the budget for the current order is exhausted.
template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>>
kepler_lcd_iterate_simd_ellipse(const Vector3<Eigen::Array<double, W, 1>> &R0,
                                const Vector3<Eigen::Array<double, W, 1>> &V0,
                                const Eigen::Array<double, W, 1> &dt, double mu,
                                const KeplerLCDOptions &opts) {
    using SS = Eigen::Array<double, W, 1>;
    using Mask = Eigen::Array<bool, W, 1>;
    using std::sqrt;

    KeplerLCDResult<SS> r;
    r.converged = true;

    const SS sqmu = SS::Constant(sqrt(mu));
    const SS r0 = (R0[0].square() + R0[1].square() + R0[2].square()).sqrt();
    const SS v2 = V0[0].square() + V0[1].square() + V0[2].square();
    const SS sigma0 = (R0[0] * V0[0] + R0[1] * V0[1] + R0[2] * V0[2]) / sqmu;
    const SS alpha = SS::Constant(2.0) / r0 - v2 / SS::Constant(mu);
    r.r0 = r0;
    r.sigma0 = sigma0;
    r.alpha = alpha;

    SS X_new = sqmu * dt * alpha;
    SS X = X_new;
    SS dX = SS::Constant(1.0e+100);
    SS U0 = SS::Constant(1.0);
    SS U1 = SS::Zero();
    SS U2 = SS::Zero();
    SS U3 = SS::Zero();
    SS rad = r0;
    SS sig = sigma0;

    Mask active = Mask::Constant(true);
    int N = 1;
    int iters_this_N = 0;

    while (active.any() && N <= opts.max_order) {
        // Refresh working X from latest X_new.  Inactive lanes keep their
        // converged X_new, so this is safe (and avoids a select).
        X = X_new;

        const SS y = alpha * X * X;
        const SS sq = y.sqrt();
        const SS C = (SS::Constant(1.0) - sq.cos()) / y;
        const SS S = (sq - sq.sin()) / (sq * y);
        U1 = X * (SS::Constant(1.0) - y * S);
        U2 = X * X * C;
        U3 = X * X * X * S;
        U0 = SS::Constant(1.0) - alpha * U2;

        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (SS::Constant(1.0) - alpha * r0) * U1;
        const SS F = r0 * U1 + sigma0 * U2 + U3 - sqmu * dt;
        const SS F1 = rad;
        const SS F2 = sig;

        const SS sgn =
            (F1 >= SS::Zero()).select(SS::Constant(1.0), SS::Constant(-1.0));
        const double Nm = static_cast<double>(N);
        const double Nm1 = Nm - 1.0;
        const SS disc =
            SS::Constant(Nm1 * Nm1) * F1 * F1 - SS::Constant(Nm * Nm1) * F * F2;
        const SS denom = disc.abs();
        const SS dX_step = (denom > SS::Zero())
                               .select(SS::Constant(Nm) * F /
                                           (F1 + sgn * denom.sqrt()),
                                       F / F1);

        // Only active lanes advance.  Inactive lanes keep their X_new fixed at
        // the converged value from the iteration where they deactivated.
        X_new = active.select(X - dX_step, X_new);
        dX = active.select(dX_step, SS::Zero());

        active = active && (dX.abs() > SS::Constant(opts.Xtol));

        ++iters_this_N;
        if (iters_this_N >= opts.iters_per_order) {
            ++N;
            iters_this_N = 0;
        }
    }
    if (active.any())
        r.converged = false;

    // Final SIMD refresh at X = X_new for every lane (mirrors the scalar's
    // post-loop refresh: U/r/sigma should reflect the converged X*).  For
    // unconverged lanes this is a best-effort evaluation; r.converged is
    // already false.
    X = X_new;
    {
        const SS y = alpha * X * X;
        const SS sq = y.sqrt();
        const SS C = (SS::Constant(1.0) - sq.cos()) / y;
        const SS S = (sq - sq.sin()) / (sq * y);
        U1 = X * (SS::Constant(1.0) - y * S);
        U2 = X * X * C;
        U3 = X * X * X * S;
        U0 = SS::Constant(1.0) - alpha * U2;
        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (SS::Constant(1.0) - alpha * r0) * U1;
    }

    r.X = X;
    r.U0 = U0;
    r.U1 = U1;
    r.U2 = U2;
    r.U3 = U3;
    r.r = rad;
    r.sigma = sig;
    return r;
}

template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>>
kepler_lcd_iterate(const Vector3<Eigen::Array<double, W, 1>> &R0,
                   const Vector3<Eigen::Array<double, W, 1>> &V0,
                   const Eigen::Array<double, W, 1> &dt, double mu,
                   const KeplerLCDOptions &opts = {}) {
    using SS = Eigen::Array<double, W, 1>;

    // Dispatch to true-SIMD path only when every lane is elliptic with nonzero
    // dt — the common case for PSIOPT collocation across an elliptic phase.
    // Mixed orbit types and dt==0 lanes fall through to the per-lane fallback,
    // which exercises the full scalar dispatch (incl. orbit-type branches and
    // the dt==0 fast path).
    const SS r0 = (R0[0].square() + R0[1].square() + R0[2].square()).sqrt();
    const SS v2 = V0[0].square() + V0[1].square() + V0[2].square();
    const SS alpha = SS::Constant(2.0) / r0 - v2 / SS::Constant(mu);

    const bool all_elliptic = (alpha > SS::Constant(opts.alpha_tol)).all();
    const bool all_nonzero_dt = (dt.abs() > SS::Zero()).all();
    if (all_elliptic && all_nonzero_dt)
        return kepler_lcd_iterate_simd_ellipse<W>(R0, V0, dt, mu, opts);

    return kepler_lcd_iterate_per_lane<W>(R0, V0, dt, mu, opts);
}

} // namespace tycho::astro::detail
