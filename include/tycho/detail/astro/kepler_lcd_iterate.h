#pragma once
#include "tycho/detail/typedefs/eigen_types.h"
#include <Eigen/Geometry>
#include <cmath>

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
        const double s = (M_PI / 2.0 - std::atan(3.0 * sqrt(mu / (p * p * p)) * dt)) / 2.0;
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

    while (fabs(dX) > opts.Xtol && N <= opts.max_order) {
        ++iters_this_N;
        if (iters_this_N >= opts.iters_per_order) {
            ++N;
            iters_this_N = 0;
            if (N > opts.max_order) {
                r.converged = false;
                break;
            }
        }
        X = X_new;

        // Why the asymptote guard: on a hyperbolic orbit the universal anomaly
        // grows logarithmically with time, but cosh/sinh blow up exponentially
        // in sqrt(-alpha)*X.  If the iterate ever overshoots past the guard,
        // U0..U3 lose precision before the iteration can recover, so we bail.
        if (alpha < -opts.alpha_tol) {
            const double sqma = sqrt(-alpha);
            if (fabs(sqma * X) > opts.hyp_guard) {
                r.converged = false;
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
    }

    X = X_new;
    compute_universal_functions<double>(alpha, X, opts.alpha_tol, U0, U1, U2, U3);
    rad = r0 * U0 + sigma0 * U1 + U2;
    sig = sigma0 * U0 + (1.0 - alpha * r0) * U1;

    r.X = X;
    r.U0 = U0;
    r.U1 = U1;
    r.U2 = U2;
    r.U3 = U3;
    r.r = rad;
    r.sigma = sig;
    return r;
}

} // namespace tycho::astro::detail
