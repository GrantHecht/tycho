#pragma once
#include "tycho/detail/astro/kepler/stumpff.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/compiler_macros.h"
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace tycho::astro::detail {

// Threshold below which the Stumpff C(y) and S(y) functions switch from the
// recursion form
//   C(y) = (1 - cos(sqrt(y))) / y,    S(y) = (sqrt(y) - sin(sqrt(y))) / (y*sqrt(y))
// to the leading-order Taylor expansions
//   C(y) = 1/2 - y/24 + O(y^2),       S(y) = 1/6 - y/120 + O(y^2).
// The recursion forms are 0/0 indeterminate at y == 0 and lose precision via
// catastrophic cancellation as y -> 0.  The dispatcher guarantees alpha is
// bounded away from 0 on the elliptic branch, but X (and therefore
// y = alpha*X^2) can still pass through zero during the LCD iteration on
// very small dt or near-period seeds.
inline constexpr double kStumpffTaylorEps = 1.0e-8;

// Stumpff helper shared by the scalar elliptic branch in
// compute_universal_functions<double> and the true-SIMD elliptic kernel
// kepler_lcd_iterate_simd_ellipse<W>.  Writes C(y) and S(y) on the
// elliptic branch, switching between the recursion form and the
// leading-order Taylor expansion at kStumpffTaylorEps.  Out-params
// (rather than a returned pair) match the surrounding compute_*
// idiom and keep clang's instruction selection bit-for-bit identical
// to the inlined-by-hand form across both the scalar and SS paths.
template <class Scalar>
TYCHO_ALWAYS_INLINE void stumpff_C_S(const Scalar &y, Scalar &C, Scalar &S) {
    using std::cos;
    using std::sin;
    using std::sqrt;
    if constexpr (std::is_same_v<Scalar, double>) {
        if (y <= kStumpffTaylorEps) {
            C = 0.5 - y / 24.0;
            S = 1.0 / 6.0 - y / 120.0;
        } else {
            const double sq = sqrt(y);
            C = (1.0 - cos(sq)) / y;
            S = (sq - sin(sq)) / (sq * y);
        }
    } else {
        const Scalar sq = y.sqrt();
        const auto small = (y <= Scalar::Constant(kStumpffTaylorEps));
        const Scalar C_rec = (Scalar::Constant(1.0) - sq.cos()) / y;
        const Scalar S_rec = (sq - sq.sin()) / (sq * y);
        const Scalar C_tay = Scalar::Constant(0.5) - y * Scalar::Constant(1.0 / 24.0);
        const Scalar S_tay = Scalar::Constant(1.0 / 6.0) - y * Scalar::Constant(1.0 / 120.0);
        C = small.select(C_tay, C_rec);
        S = small.select(S_tay, S_rec);
    }
}

// Iteration knobs for kepler_lcd_iterate.  Invariants are enforced at
// construction — every reachable instance is well-formed.
//
// alpha_tol must be strictly positive (not just non-negative): the
// IFT-layer Taylor-blend mask `abs(alpha) <= alpha_tol` (used in
// U_partials_alpha and compute_U_second_partials) is true only at
// exactly-zero alpha when alpha_tol == 0, letting numerically-near-
// parabolic alphas slip through to the 1/(2*alpha) recursion branch
// that the Taylor branch is meant to short-circuit.
class KeplerLCDOptions {
  public:
    constexpr KeplerLCDOptions() noexcept = default;

    // Validating ctor.  Throws std::invalid_argument on bad inputs.
    // The negated comparisons (!(x > 0)) reject NaN inputs as well as
    // zero/negative values, since any comparison against NaN is unordered.
    KeplerLCDOptions(double Xtol, double alpha_tol, int max_order, int iters_per_order,
                     double hyp_guard)
        : Xtol_(Xtol), alpha_tol_(alpha_tol), max_order_(max_order),
          iters_per_order_(iters_per_order), hyp_guard_(hyp_guard) {
        if (!(Xtol_ > 0.0))
            throw std::invalid_argument("KeplerLCDOptions: Xtol must be > 0");
        if (!(alpha_tol_ > 0.0))
            throw std::invalid_argument("KeplerLCDOptions: alpha_tol must be > 0");
        if (max_order_ < 1)
            throw std::invalid_argument("KeplerLCDOptions: max_order must be >= 1");
        if (iters_per_order_ < 1)
            throw std::invalid_argument("KeplerLCDOptions: iters_per_order must be >= 1");
        if (!(hyp_guard_ > 0.0))
            throw std::invalid_argument("KeplerLCDOptions: hyp_guard must be > 0");
    }

    [[nodiscard]] constexpr double Xtol() const noexcept { return Xtol_; }
    [[nodiscard]] constexpr double alpha_tol() const noexcept { return alpha_tol_; }
    [[nodiscard]] constexpr int max_order() const noexcept { return max_order_; }
    [[nodiscard]] constexpr int iters_per_order() const noexcept { return iters_per_order_; }
    [[nodiscard]] constexpr double hyp_guard() const noexcept { return hyp_guard_; }

  private:
    double Xtol_ = 1.0e-12;
    double alpha_tol_ = 1.0e-12;
    int max_order_ = 10;
    int iters_per_order_ = 10;
    double hyp_guard_ = 30.0;
};

// Per-lane convergence type: scalar bool for Scalar=double, Eigen lane mask for
// SuperScalar.  PSIOPT consumers use all_converged() below to reduce to a single
// bool early-exit gate.
template <class Scalar> struct KeplerLCDConvergedFlag {
    using type = bool;
};

template <int W> struct KeplerLCDConvergedFlag<Eigen::Array<double, W, 1>> {
    using type = Eigen::Array<bool, W, 1>;
};

inline bool all_converged(bool b) { return b; }

template <int W> inline bool all_converged(const Eigen::Array<bool, W, 1> &m) { return m.all(); }

// Polymorphic NaN scalar: yields a plain double NaN for Scalar=double, and an
// SS-broadcast NaN for Scalar=Eigen::Array<double,W,1>.  Used by the IFT layer
// and propagate_cartesian to NaN-poison outputs when the LCD kernel fails to
// converge.
template <class Scalar> inline Scalar kepler_nan_value() {
    if constexpr (std::is_same_v<Scalar, double>) {
        return std::numeric_limits<double>::quiet_NaN();
    } else {
        return Scalar::Constant(std::numeric_limits<double>::quiet_NaN());
    }
}

template <class Scalar> struct KeplerLCDResult {
    Scalar X;
    Scalar alpha;
    Scalar r0;
    Scalar sigma0;
    Scalar r;
    Scalar sigma;
    Stumpff<Scalar> U;
    typename KeplerLCDConvergedFlag<Scalar>::type converged;
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
        double C, S;
        stumpff_C_S<double>(y, C, S);
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
    if (!(mu > 0.0)) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate: mu must satisfy mu > 0");
    if (!std::isfinite(dt)) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate: dt must be finite");
    if (!V0.allFinite()) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate: V0 must be finite");
    KeplerLCDResult<double> r;
    r.converged = true;
    const double sqmu = sqrt(mu);
    const double r0 = R0.norm();
    if (!(r0 > 0.0)) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate: r0 must satisfy r0 > 0");
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
        r.U.U0 = 1.0;
        r.U.U1 = 0.0;
        r.U.U2 = 0.0;
        r.U.U3 = 0.0;
        r.r = r0;
        r.sigma = sigma0;
        return r;
    }

    double X;
    if (alpha > opts.alpha_tol()) {
        X = sqmu * dt * alpha;
    } else if (alpha < -opts.alpha_tol()) {
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
    while (fabs(dX) > opts.Xtol() && N <= opts.max_order()) {
        X = X_new;

        // Why the asymptote guard: on a hyperbolic orbit the universal anomaly
        // grows logarithmically with time, but cosh/sinh blow up exponentially
        // in sqrt(-alpha)*X.  If the iterate ever overshoots past the guard,
        // U0..U3 lose precision before the iteration can recover, so we bail.
        if (alpha < -opts.alpha_tol()) {
            const double sqma = sqrt(-alpha);
            if (fabs(sqma * X) > opts.hyp_guard()) {
                guard_tripped = true;
                break;
            }
        }

        compute_universal_functions<double>(alpha, X, opts.alpha_tol(), U0, U1, U2, U3);
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
        // gets opts.iters_per_order() updates before promotion.
        ++iters_this_N;
        if (iters_this_N >= opts.iters_per_order()) {
            ++N;
            iters_this_N = 0;
        }
    }
    if (guard_tripped || fabs(dX) > opts.Xtol())
        r.converged = false;

    // Only refresh the final U/r/sigma at X_new on a clean convergence exit.
    // On guard- or order-cap bailout, X_new may be in the regime that tripped
    // the guard, so reusing the last in-loop values (computed at the most
    // recent stable X) is the safer reporting choice.
    if (r.converged) {
        X = X_new;
        compute_universal_functions<double>(alpha, X, opts.alpha_tol(), U0, U1, U2, U3);
        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (1.0 - alpha * r0) * U1;
    }

    r.X = X;
    r.U.U0 = U0;
    r.U.U1 = U1;
    r.U.U2 = U2;
    r.U.U3 = U3;
    r.r = rad;
    r.sigma = sig;
    // NaN-poisoned outputs must not report converged=true: IEEE-754 makes the
    // loop's dX > Xtol comparison false on NaN, so a poisoned loop can exit
    // silently with the converged flag still set.
    if (!(std::isfinite(r.X) && std::isfinite(r.r) && std::isfinite(r.sigma) &&
          std::isfinite(r.U.U0) && std::isfinite(r.U.U1) && std::isfinite(r.U.U2) &&
          std::isfinite(r.U.U3))) [[unlikely]]
        r.converged = false;
    return r;
}

template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>> kepler_lcd_iterate_per_lane(
    const Vector3<Eigen::Array<double, W, 1>> &R0, const Vector3<Eigen::Array<double, W, 1>> &V0,
    const Eigen::Array<double, W, 1> &dt, double mu, const KeplerLCDOptions &opts) {
    using SS = Eigen::Array<double, W, 1>;
    KeplerLCDResult<SS> out;
    out.converged = Eigen::Array<bool, W, 1>::Constant(true);
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
        out.U.U0[lane] = k.U.U0;
        out.U.U1[lane] = k.U.U1;
        out.U.U2[lane] = k.U.U2;
        out.U.U3[lane] = k.U.U3;
        out.converged[lane] = k.converged;
    }
    return out;
}

// True-SIMD Laguerre-Conway-Der iteration for the case where all W lanes are
// elliptic (alpha[i] > opts.alpha_tol()) and all have nonzero dt.  The caller is
// responsible for verifying the precondition; this routine does not re-check.
//
// The iteration body is fully packed across lanes — no per-lane scalar
// dispatch.  Lanes that converge first are kept in an "active" mask: they
// continue to ride through the SIMD body but no longer update X_new, so the
// SIMD lane keeps its converged value.  The shared order N is promoted whenever
// the budget for the current order is exhausted.
template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>> kepler_lcd_iterate_simd_ellipse(
    const Vector3<Eigen::Array<double, W, 1>> &R0, const Vector3<Eigen::Array<double, W, 1>> &V0,
    const Eigen::Array<double, W, 1> &dt, double mu, const KeplerLCDOptions &opts) {
    using SS = Eigen::Array<double, W, 1>;
    using Mask = Eigen::Array<bool, W, 1>;
    using std::sqrt;

    KeplerLCDResult<SS> r;
    r.converged = Mask::Constant(true);

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

    while (active.any() && N <= opts.max_order()) {
        // Refresh working X from latest X_new.  Inactive lanes keep their
        // converged X_new, so this is safe (and avoids a select).
        X = X_new;

        const SS y = alpha * X * X;
        SS C, S;
        stumpff_C_S<SS>(y, C, S);
        U1 = X * (SS::Constant(1.0) - y * S);
        U2 = X * X * C;
        U3 = X * X * X * S;
        U0 = SS::Constant(1.0) - alpha * U2;

        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (SS::Constant(1.0) - alpha * r0) * U1;
        const SS F = r0 * U1 + sigma0 * U2 + U3 - sqmu * dt;
        const SS F1 = rad;
        const SS F2 = sig;

        const SS sgn = (F1 >= SS::Zero()).select(SS::Constant(1.0), SS::Constant(-1.0));
        const double Nm = static_cast<double>(N);
        const double Nm1 = Nm - 1.0;
        const SS disc = SS::Constant(Nm1 * Nm1) * F1 * F1 - SS::Constant(Nm * Nm1) * F * F2;
        const SS denom = disc.abs();
        const SS dX_step =
            (denom > SS::Zero()).select(SS::Constant(Nm) * F / (F1 + sgn * denom.sqrt()), F / F1);

        // Only active lanes advance.  Inactive lanes keep their X_new fixed at
        // the converged value from the iteration where they deactivated.
        X_new = active.select(X - dX_step, X_new);
        dX = active.select(dX_step, SS::Zero());

        // Include dX.isFinite() in the active mask: a NaN dX evaluates the
        // tolerance comparison as unordered (false), which would otherwise
        // mask the lane out of `active` and let it masquerade as converged.
        // The isFinite() term forces NaN-poisoned lanes to deactivate via
        // the false branch; the post-loop reduction `(!active) && finite`
        // (line below where converged is computed) then sees `finite` as
        // false on those lanes and reports converged = false.  Both terms
        // are needed: dropping isFinite() costs iteration budget (a NaN
        // lane would keep iterating until the order cap exhausts; the
        // post-loop finite mask still catches it for correctness, but at
        // wasted iterations); dropping the tolerance term loops forever
        // on a finite-but-non-converging dX.
        active = active && (dX.abs() > SS::Constant(opts.Xtol())) && dX.isFinite();

        ++iters_this_N;
        if (iters_this_N >= opts.iters_per_order()) {
            ++N;
            iters_this_N = 0;
        }
    }
    // Snapshot the last in-loop values before the post-loop refresh (these are
    // computed at the committed X via "X = X_new" at the top of the loop body).
    // Unconverged (still-active) lanes will keep this snapshot — the scalar
    // path applies the same strategy via a `if (r.converged)` guard above.
    const SS X_in = X;
    const SS U0_in = U0, U1_in = U1, U2_in = U2, U3_in = U3;
    const SS rad_in = rad, sig_in = sig;

    // Refresh U/r/sigma at the final X = X_new (matches scalar's "refresh on
    // convergence" — for converged lanes X_new equals their committed X to
    // within Xtol, so this is essentially idempotent and just polishes them
    // to the latest iterate; for unconverged lanes we'll discard the result
    // via the active-gated select below).
    X = X_new;
    {
        const SS y = alpha * X * X;
        SS C, S;
        stumpff_C_S<SS>(y, C, S);
        U1 = X * (SS::Constant(1.0) - y * S);
        U2 = X * X * C;
        U3 = X * X * X * S;
        U0 = SS::Constant(1.0) - alpha * U2;
        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (SS::Constant(1.0) - alpha * r0) * U1;
    }

    // Restore in-loop snapshot on unconverged lanes only.
    X = active.select(X_in, X);
    U0 = active.select(U0_in, U0);
    U1 = active.select(U1_in, U1);
    U2 = active.select(U2_in, U2);
    U3 = active.select(U3_in, U3);
    rad = active.select(rad_in, rad);
    sig = active.select(sig_in, sig);

    // Per-lane converged: lanes that left `active` (|dX| <= Xtol) AND whose
    // current values are finite.
    const Mask finite = X.isFinite() && rad.isFinite() && sig.isFinite() && U0.isFinite() &&
                        U1.isFinite() && U2.isFinite() && U3.isFinite();
    r.converged = (!active) && finite;

    r.X = X;
    r.U.U0 = U0;
    r.U.U1 = U1;
    r.U.U2 = U2;
    r.U.U3 = U3;
    r.r = rad;
    r.sigma = sig;
    return r;
}

template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>> kepler_lcd_iterate(
    const Vector3<Eigen::Array<double, W, 1>> &R0, const Vector3<Eigen::Array<double, W, 1>> &V0,
    const Eigen::Array<double, W, 1> &dt, double mu, const KeplerLCDOptions &opts = {}) {
    using SS = Eigen::Array<double, W, 1>;
    if (!(mu > 0.0)) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate (SS): mu must satisfy mu > 0");
    if (!dt.isFinite().all()) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate (SS): every lane's dt must be finite");
    if (!(V0[0].isFinite() && V0[1].isFinite() && V0[2].isFinite()).all()) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate (SS): every lane's V0 must be finite");

    // Dispatch to true-SIMD path only when every lane is elliptic with nonzero
    // dt — the common case for PSIOPT collocation across an elliptic phase.
    // Mixed orbit types and dt==0 lanes fall through to the per-lane fallback,
    // which exercises the full scalar dispatch (incl. orbit-type branches and
    // the dt==0 fast path).
    const SS r0 = (R0[0].square() + R0[1].square() + R0[2].square()).sqrt();
    // Match the scalar kernel's r0 > 0 precondition uniformly across all
    // SS lanes — the SIMD-ellipse path silently NaN-poisons on r0 == 0
    // because it computes `2 / r0` unconditionally; the per-lane fallback
    // would catch it via the inner scalar throw, so checking here gives
    // both paths the same diagnostic.
    if (!(r0 > SS::Zero()).all()) [[unlikely]]
        throw std::invalid_argument("kepler_lcd_iterate (SS): every lane's r0 must satisfy r0 > 0");
    const SS v2 = V0[0].square() + V0[1].square() + V0[2].square();
    const SS alpha = SS::Constant(2.0) / r0 - v2 / SS::Constant(mu);

    const bool all_elliptic = (alpha > SS::Constant(opts.alpha_tol())).all();
    const bool all_nonzero_dt = (dt.abs() > SS::Zero()).all();
    if (all_elliptic && all_nonzero_dt)
        return kepler_lcd_iterate_simd_ellipse<W>(R0, V0, dt, mu, opts);

    return kepler_lcd_iterate_per_lane<W>(R0, V0, dt, mu, opts);
}

} // namespace tycho::astro::detail
