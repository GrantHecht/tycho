#pragma once
#include "tycho/detail/astro/conversions/kepler_utils.h"
#include "tycho/detail/astro/kepler/kepler_lcd_iterate.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include <cmath>
#include <stdexcept>

namespace tycho::astro {

/// @brief Propagate a Cartesian state forward under two-body gravity.
///
/// Uses the universal-variable LCD iteration followed by the Goodyear f-g map.
/// On LCD non-convergence returns a NaN-poisoned vector.
/// @tparam Scalar Numeric scalar type (double or Eigen::Array SuperScalar).
/// @param[in] RV  Initial Cartesian state [rx, ry, rz, vx, vy, vz].
/// @param[in] dt  Time-of-flight.
/// @param[in] mu  Gravitational parameter (must be > 0); cast to double internally
///                regardless of Scalar type.
/// @return Propagated Cartesian state [rx', ry', rz', vx', vy', vz'].
template <class Scalar>
Vector6<Scalar> propagate_cartesian(const Vector6<Scalar> &RV, Scalar dt, Scalar mu) {
    using std::sqrt;
    if (dt == Scalar(0))
        return RV;

    // Materialize R0/V0 into concrete Vector3<Scalar> values: the segment
    // expressions returned by .head<3>() / .tail<3>() are VectorBlock types,
    // which prevent template-argument deduction from picking the right
    // kepler_lcd_iterate overload (the deducible SS overload requires a
    // concrete Vector3<Eigen::Array<...>>).
    const Vector3<Scalar> R0 = RV.template head<3>();
    const Vector3<Scalar> V0 = RV.template tail<3>();
    auto k = ::tycho::astro::detail::kepler_lcd_iterate(R0, V0, dt, double(mu));
    if (!::tycho::astro::detail::all_converged(k.converged)) [[unlikely]]
        return Vector6<Scalar>::Constant(::tycho::astro::detail::kepler_nan_value<Scalar>());

    Scalar sqmu = sqrt(mu);
    Scalar aF = Scalar(1) - k.U.U2 / k.r0;
    Scalar aG = (k.r0 * k.U.U1 + k.sigma0 * k.U.U2) / sqmu;
    Scalar aFt = -sqmu / (k.r0 * k.r) * k.U.U1;
    Scalar aGt = Scalar(1) - k.U.U2 / k.r;

    Vector6<Scalar> fx;
    fx.template head<3>() = aF * RV.template head<3>() + aG * RV.template tail<3>();
    fx.template tail<3>() = aFt * RV.template head<3>() + aGt * RV.template tail<3>();
    return fx;
}

/// @brief Propagate classic orbital elements forward by dt.
///
/// Increments the mean anomaly: M_new = M + n*dt, n = sqrt(mu/|a|^3).
/// All angles are in radians; M is the mean anomaly (oelems[5]).
/// @tparam Scalar Numeric scalar type.
/// @param[in] oelems Classic elements [a, e, i, Omega, w, M].
/// @param[in] dt     Time-of-flight.
/// @param[in] mu     Gravitational parameter (must be > 0).
/// @return Propagated classic elements (only M is updated).
template <class Scalar>
Vector6<Scalar> propagate_classic(const Vector6<Scalar> &oelems, Scalar dt, Scalar mu) {
    using std::abs;
    using std::sqrt;
    if constexpr (std::is_same_v<Scalar, double>) {
        if (!(mu > 0.0))
            throw std::invalid_argument("propagate_classic: mu must satisfy mu > 0");
        if (!(std::isfinite(oelems[0]) && oelems[0] != 0.0))
            throw std::invalid_argument(
                "propagate_classic: semi-major axis a (oelems[0]) must be finite and non-zero");
    }
    Scalar a = oelems[0];
    Scalar n = sqrt(mu / abs(a * a * a));
    Vector6<Scalar> noelems = oelems;
    noelems[5] += n * dt;
    return noelems;
}

/// @brief Propagate Modified Equinoctial Elements forward by dt.
///
/// Round-trips through classical elements for the analytic mean-anomaly update:
/// MEE → classic → propagate_classic → classic → MEE.
/// @tparam Scalar Numeric scalar type.
/// @param[in] meelems MEE state [p, f, g, h, k, L] with L the true longitude.
/// @param[in] dt      Time-of-flight.
/// @param[in] mu      Gravitational parameter (must be > 0).
/// @return Propagated MEE state.
template <class Scalar>
Vector6<Scalar> propagate_modified(const Vector6<Scalar> &meelems, Scalar dt, Scalar mu) {
    if constexpr (std::is_same_v<Scalar, double>) {
        if (!(mu > 0.0))
            throw std::invalid_argument("propagate_modified: mu must satisfy mu > 0");
    }
    Vector6<Scalar> oelems = modified_to_classic(meelems, mu);
    Vector6<Scalar> noelems = propagate_classic(oelems, dt, mu);
    return classic_to_modified(noelems, mu);
}

} // namespace tycho::astro
