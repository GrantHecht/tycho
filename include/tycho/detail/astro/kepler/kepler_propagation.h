#pragma once
#include "tycho/detail/astro/conversions/kepler_utils.h"
#include "tycho/detail/astro/kepler/kepler_lcd_iterate.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include <cmath>

namespace tycho::astro {

// Cartesian-state two-body propagation via the universal-variable
// Laguerre-Conway-Der iteration in detail::kepler_lcd_iterate, followed by the
// Goodyear closed-form f-g map.
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

    Scalar sqmu = sqrt(mu);
    Scalar aF = Scalar(1) - k.U2 / k.r0;
    Scalar aG = (k.r0 * k.U1 + k.sigma0 * k.U2) / sqmu;
    Scalar aFt = -sqmu / (k.r0 * k.r) * k.U1;
    Scalar aGt = Scalar(1) - k.U2 / k.r;

    Vector6<Scalar> fx;
    fx.template head<3>() = aF * RV.template head<3>() + aG * RV.template tail<3>();
    fx.template tail<3>() = aFt * RV.template head<3>() + aGt * RV.template tail<3>();
    return fx;
}

// Classical-element two-body propagation: mean anomaly increments by n*dt,
// where n = sqrt(mu / |a|^3).
template <class Scalar>
Vector6<Scalar> propagate_classic(const Vector6<Scalar> &oelems, Scalar dt, Scalar mu) {
    using std::abs;
    using std::sqrt;
    Scalar a = oelems[0];
    Scalar n = sqrt(mu / abs(a * a * a));
    Vector6<Scalar> noelems = oelems;
    noelems[5] += n * dt;
    return noelems;
}

// Modified-equinoctial-element propagation: round-trip through classical for
// the analytic mean-anomaly update.
template <class Scalar>
Vector6<Scalar> propagate_modified(const Vector6<Scalar> &meelems, Scalar dt, Scalar mu) {
    Vector6<Scalar> oelems = modified_to_classic(meelems, mu);
    Vector6<Scalar> noelems = propagate_classic(oelems, dt, mu);
    return classic_to_modified(noelems, mu);
}

} // namespace tycho::astro
