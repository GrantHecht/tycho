// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include "tycho/vector_functions.h"

namespace tycho::astro {

// Import cross-namespace types from vf and utils.
using utils::SZ_SUM;
using vf::Arguments;
using vf::CMatRef;
using vf::Constant;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::IfElseFunction;
using vf::MatRef;
using vf::VecRef;
using vf::VectorExpression;
using vf::VectorFunction;

////////////////////////////////////////////////////////////////////////////////////////
////////////////////              Conversions                  /////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/// @brief Convert classical orbital elements to Cartesian state.
///
/// Solves Kepler's equation via Newton iteration (elliptic) or the hyperbolic
/// analogue to find eccentric/hyperbolic anomaly, then rotates to the inertial
/// frame via the Euler-angle sequence (Omega, i, omega).
///
/// @tparam Scalar Floating-point scalar type (double or equivalent).
/// @param oelems Six classical elements [a, e, i, Omega, omega, M] (semi-major axis,
///               eccentricity, inclination, RAAN, argument of perigee, mean anomaly).
///               For hyperbolic orbits (e > 1), a < 0 by convention, and M is the
///               hyperbolic mean anomaly (M = e·sinh(H) − H).
/// @param mu     Gravitational parameter (km³/s² or consistent units).
/// @return Six Cartesian state [rx, ry, rz, vx, vy, vz].
template <class Scalar>
Vector6<Scalar> classic_to_cartesian(const Vector6<Scalar> &oelems, Scalar mu) {

    const int MAXITERS = 15;
    const double TOL = 1.0e-12;
    const double PI = 3.14159265358979;

    Scalar a = oelems[0];
    Scalar e = oelems[1];
    Scalar i = oelems[2];
    Scalar Omega = oelems[3];
    Scalar w = oelems[4];
    Scalar M = oelems[5];

    // Calc Eccentric anomally

    Scalar x, y, vx, vy;

    if (e < 1.0) { // Elliptic
        Scalar E = M;
        Scalar sinE;
        Scalar cosE;
        Scalar fE;
        Scalar jE;
        for (int i = 0; i < MAXITERS; i++) {
            sinE = sin(E);
            cosE = cos(E);
            fE = E - e * sinE - M;
            if (abs(fE) < TOL)
                break;
            E = E - (fE) / (1 - e * cosE);
        }
        Scalar v = 2.0 * atan2(sqrt(1. + e) * sin(E / 2.0), sqrt(1. - e) * cos(E / 2.0));
        Scalar rc = a * (1. - e * cos(E));
        Scalar vc = sqrt(mu * a) / rc;

        x = rc * cos(v);
        y = rc * sin(v);
        vx = -vc * sinE;
        vy = vc * sqrt(1. - e * e) * cosE;

    } else { // Hyperbolic
        Scalar H = M;
        Scalar sinhH;
        Scalar coshH;
        Scalar fH;
        Scalar jH;

        for (int i = 0; i < MAXITERS; i++) {
            sinhH = sinh(H);
            coshH = cosh(H);
            fH = e * sinhH - H - M;
            if (abs(fH) < TOL)
                break;
            H = H - (fH) / (e * coshH - 1);
        }
        Scalar rc = a * (1 - e * coshH);

        Scalar v = 2.0 * atan2(sqrt(1. + e) * sinh(H / 2.0), sqrt(e - 1) * cosh(H / 2.0));
        Scalar vc = sqrt(-mu * a) / rc;

        x = rc * cos(v);
        y = rc * sin(v);
        vx = -vc * sinhH;
        vy = vc * sqrt(e * e - 1) * coshH;
    }
    /////////////////////////

    Vector6<Scalar> XV;

    Scalar ci = cos(i);
    Scalar si = sin(i);

    Scalar cw = cos(w);
    Scalar sw = sin(w);

    Scalar cO = cos(Omega);
    Scalar sO = sin(Omega);

    XV[0] = x * (cw * cO - sw * ci * sO) - y * (sw * cO + cw * ci * sO);
    XV[1] = x * (cw * sO + sw * ci * cO) + y * (cw * ci * cO - sw * sO);
    XV[2] = x * (sw * si) + y * (cw * si);

    XV[3] = vx * (cw * cO - sw * ci * sO) - vy * (sw * cO + cw * ci * sO);
    XV[4] = vx * (cw * sO + sw * ci * cO) + vy * (cw * ci * cO - sw * sO);
    XV[5] = vx * (sw * si) + vy * (cw * si);

    return XV;
}

/// @brief Convert Cartesian state to classical orbital elements (mean anomaly).
///
/// Computes orbital elements via angular momentum and eccentricity vectors.
/// Mean anomaly is computed from the eccentric (elliptic) or hyperbolic anomaly.
///
/// @tparam Scalar Floating-point scalar type (double or equivalent).
/// @param XV Six Cartesian state [rx, ry, rz, vx, vy, vz].
/// @param mu Gravitational parameter (km³/s² or consistent units).
/// @return Six classical elements [a, e, i, Omega, omega, M], where M is the
///         elliptic mean anomaly (M = E − e·sin(E)) for e < 1 and the hyperbolic
///         mean anomaly (M = e·sinh(H) − H) for e > 1.
template <class Scalar> Vector6<Scalar> cartesian_to_classic(const Vector6<Scalar> &XV, Scalar mu) {

    const double PI = 3.14159265358979;

    Vector3<Scalar> R = XV.template head<3>();
    Vector3<Scalar> V = XV.template tail<3>();

    Vector3<Scalar> hvec = R.cross(V);
    Vector3<Scalar> evec = V.cross(hvec) / mu - R.normalized();

    Vector3<Scalar> nvec;
    nvec[0] = -hvec[1];
    nvec[1] = hvec[0];

    Scalar e = evec.norm();

    Scalar drv = R.dot(V);
    Scalar v = acos(evec.normalized().dot(R.normalized()));
    if (drv < 0)
        v = 2.0 * PI - v;

    Scalar M;

    if (e < 1) { // Elliptic
        Scalar E = 2. * atan(tan(v / 2.0) / (sqrt((1.0 + e) / (1.0 - e))));
        M = E - e * sin(E);
    } else { // Hyperbolic
        Scalar H = 2. * atanh(tan(v / 2.0) / (sqrt((1.0 + e) / (e - 1))));
        M = e * sinh(H) - H;
    }

    Scalar Omega = acos(nvec[0] / nvec.norm());
    if (nvec[1] < 0)
        Omega = 2.0 * PI - Omega;

    Scalar w = acos(evec.normalized().dot(nvec.normalized()));
    if (evec[2] < 0)
        w = 2.0 * PI - w;

    Scalar i = acos(hvec[2] / hvec.norm());

    Scalar a = 1.0 / (2.0 / R.norm() - V.squaredNorm() / mu);

    Vector6<Scalar> oelems;

    oelems[0] = a;
    oelems[1] = e;
    oelems[2] = i;
    oelems[3] = Omega;
    oelems[4] = w;
    oelems[5] = M;
    return oelems;
}

/// @brief Convert Modified Equinoctial Elements (MEE) to Cartesian state.
///
/// Direct closed-form conversion using the standard MEE frame vectors.
/// Elements are [p, f, g, h, k, L] where L is the true longitude.
///
/// @tparam Scalar Floating-point scalar type (double or equivalent).
/// @param meelems Six MEE [p, f, g, h, k, L] (semi-latus rectum, equinoctial
///                eccentricity components, inclination components, true longitude).
/// @param mu      Gravitational parameter (km³/s² or consistent units).
/// @return Six Cartesian state [rx, ry, rz, vx, vy, vz].
template <class Scalar>
Vector6<Scalar> modified_to_cartesian(const Vector6<Scalar> &meelems, Scalar mu) {

    Scalar p = meelems[0];
    Scalar f = meelems[1];
    Scalar g = meelems[2];
    Scalar h = meelems[3];
    Scalar k = meelems[4];
    Scalar L = meelems[5];

    Scalar cosL = cos(L);
    Scalar sinL = sin(L);

    Scalar a2 = h * h - k * k;
    Scalar s2 = 1 + h * h + k * k;
    Scalar w = 1 + f * cosL + g * sinL;
    Scalar rr = p / w;

    Scalar Xscale = rr / s2;
    Scalar Vscale = sqrt(mu / p) / s2;

    Vector6<Scalar> XV;

    XV[0] = Xscale * (cosL + a2 * cosL + 2 * h * k * sinL);
    XV[1] = Xscale * (sinL - a2 * sinL + 2 * h * k * cosL);
    XV[2] = 2 * Xscale * (h * sinL - k * cosL);

    XV[3] = -Vscale * (sinL + a2 * sinL - 2 * h * k * cosL + g - 2 * f * h * k + a2 * g);
    XV[4] = -Vscale * (-cosL + a2 * cosL + 2 * h * k * sinL - f + 2 * g * h * k + a2 * f);
    XV[5] = 2 * Vscale * (h * cosL + k * sinL + f * h + g * k);

    return XV;
}

/// @brief Convert Modified Equinoctial Elements to classical orbital elements (mean anomaly).
///
/// Converts MEE [p, f, g, h, k, L] to classical elements [a, e, i, Omega, omega, M]
/// using algebraic relations plus Kepler-equation solution for mean anomaly.
///
/// @tparam Scalar Floating-point scalar type (double or equivalent).
/// @param meelems Six MEE [p, f, g, h, k, L].
/// @param mu      Gravitational parameter (km³/s² or consistent units); accepted for
///                API symmetry but is not used in the algebraic conversion.
/// @return Six classical elements [a, e, i, Omega, omega, M], where M is the
///         elliptic mean anomaly (M = E − e·sin(E)) for e < 1 and the hyperbolic
///         mean anomaly (M = e·sinh(H) − H) for e > 1.
template <class Scalar>
Vector6<Scalar> modified_to_classic(const Vector6<Scalar> &meelems, Scalar mu) {

    Scalar p = meelems[0];
    Scalar f = meelems[1];
    Scalar g = meelems[2];
    Scalar h = meelems[3];
    Scalar k = meelems[4];
    Scalar L = meelems[5];

    Scalar a = p / (1 - f * f - g * g);
    Scalar e = sqrt(f * f + g * g);
    Scalar i = atan2(2 * sqrt(h * h + k * k), 1 - h * h - k * k);
    Scalar Omega = atan2(k, h);
    Scalar w = atan2(g * h - f * k, f * h + g * k);
    Scalar v = L - (Omega + w);

    Scalar M;

    if (e < 1) { // Elliptic
        Scalar E = 2. * atan(tan(v / 2.0) / (sqrt((1.0 + e) / (1.0 - e))));
        M = E - e * sin(E);
    } else { // Hyperbolic
        Scalar H = 2. * atanh(tan(v / 2.0) / (sqrt((1.0 + e) / (e - 1))));
        M = e * sinh(H) - H;
    }

    Vector6<Scalar> oelems;

    oelems[0] = a;
    oelems[1] = e;
    oelems[2] = i;
    oelems[3] = Omega;
    oelems[4] = w;
    oelems[5] = M;
    return oelems;
}

/// @brief Convert classical orbital elements to Modified Equinoctial Elements.
///
/// Solves for the true anomaly from the mean anomaly via Newton iteration, then
/// computes MEE [p, f, g, h, k, L] algebraically from the classical elements.
///
/// @tparam Scalar Floating-point scalar type (double or equivalent).
/// @param oelems Six classical elements [a, e, i, Omega, omega, M].
/// @param mu     Gravitational parameter (km³/s² or consistent units); accepted for
///               API symmetry but is not used in the algebraic conversion.
/// @return Six MEE [p, f, g, h, k, L].
template <class Scalar>
Vector6<Scalar> classic_to_modified(const Vector6<Scalar> &oelems, Scalar mu) {

    const int MAXITERS = 15;
    const double TOL = 1.0e-12;
    const double PI = 3.14159265358979;

    Scalar a = oelems[0];
    Scalar e = oelems[1];
    Scalar i = oelems[2];
    Scalar Omega = oelems[3];
    Scalar w = oelems[4];
    Scalar M = oelems[5];

    // Calc True anomally
    Scalar v;
    if (e < 1.0) { // Elliptic
        Scalar E = M;
        Scalar sinE;
        Scalar cosE;
        Scalar fE;
        Scalar jE;
        for (int i = 0; i < MAXITERS; i++) {
            sinE = sin(E);
            cosE = cos(E);
            fE = E - e * sinE - M;
            if (abs(fE) < TOL)
                break;
            E = E - (fE) / (1 - e * cosE);
        }
        v = 2.0 * atan2(sqrt(1. + e) * sin(E / 2.0), sqrt(1. - e) * cos(E / 2.0));
    } else { // Hyperbolic
        Scalar H = M;
        Scalar sinhH;
        Scalar coshH;
        Scalar fH;
        Scalar jH;

        for (int i = 0; i < MAXITERS; i++) {
            sinhH = sinh(H);
            coshH = cosh(H);
            fH = e * sinhH - H - M;
            if (abs(fH) < TOL)
                break;
            H = H - (fH) / (e * coshH - 1);
        }
        v = 2.0 * atan2(sqrt(1. + e) * sinh(H / 2.0), sqrt(e - 1) * cosh(H / 2.0));
    }
    /////////////////////////

    Vector6<Scalar> meelems;

    meelems[0] = a * (1 - e * e);         // p
    meelems[1] = e * cos(w + Omega);      // f
    meelems[2] = e * sin(w + Omega);      // g
    meelems[3] = tan(i / 2) * cos(Omega); // h
    meelems[4] = tan(i / 2) * sin(Omega); // k
    meelems[5] = w + Omega + v;           // L

    return meelems;
}

/// @brief Convert Cartesian state to Modified Equinoctial Elements.
///
/// Direct conversion with no Kepler-equation iteration. Mirrors the
/// Fortran-Astrodynamics-Toolkit modified_equinoctial_module::cartesian_to_equinoctial.
/// Valid for all inclinations except exactly i = 180° (retrograde equatorial), where
/// the representation is singular; retrograde non-equatorial orbits are supported.
///
/// @tparam Scalar Floating-point scalar type (double or equivalent).
/// @param XV Six Cartesian state [rx, ry, rz, vx, vy, vz].
/// @param mu Gravitational parameter (km³/s² or consistent units).
/// @return Six MEE [p, f, g, h, k, L].
template <class Scalar>
Vector6<Scalar> cartesian_to_modified(const Vector6<Scalar> &XV, Scalar mu) {
    using std::atan2;
    using std::sqrt;

    Vector3<Scalar> R = XV.template head<3>();
    Vector3<Scalar> V = XV.template tail<3>();

    Scalar rdv = R.dot(V);
    Scalar rmag = R.norm();
    Vector3<Scalar> rhat = R / rmag;

    Vector3<Scalar> hvec = R.cross(V);
    Scalar hmag_sq = hvec.squaredNorm();
    Scalar hmag = sqrt(hmag_sq);
    Vector3<Scalar> hhat = hvec / hmag;

    Vector3<Scalar> vhat = (rmag * V - rdv * rhat) / hmag;

    Scalar p = hmag_sq / mu;
    Scalar inv_one_plus_hz = Scalar(1.0) / (Scalar(1.0) + hhat[2]);
    Scalar k = hhat[0] * inv_one_plus_hz;
    Scalar h = -hhat[1] * inv_one_plus_hz;
    Scalar kk = k * k;
    Scalar hh = h * h;
    Scalar s2 = Scalar(1.0) + hh + kk;
    Scalar inv_s2 = Scalar(1.0) / s2;
    Scalar tkh = Scalar(2.0) * k * h;

    Vector3<Scalar> fhat;
    fhat[0] = (Scalar(1.0) - kk + hh) * inv_s2;
    fhat[1] = tkh * inv_s2;
    fhat[2] = Scalar(-2.0) * k * inv_s2;

    Vector3<Scalar> ghat;
    ghat[0] = tkh * inv_s2;
    ghat[1] = (Scalar(1.0) + kk - hh) * inv_s2;
    ghat[2] = Scalar(2.0) * h * inv_s2;

    Vector3<Scalar> ecc = V.cross(hvec) / mu - rhat;
    Scalar f = ecc.dot(fhat);
    Scalar g = ecc.dot(ghat);
    Scalar L = atan2(rhat[1] - vhat[0], rhat[0] + vhat[1]);

    Vector6<Scalar> meelems;
    meelems[0] = p;
    meelems[1] = f;
    meelems[2] = g;
    meelems[3] = h;
    meelems[4] = k;
    meelems[5] = L;
    return meelems;
}

/// @brief Convert Cartesian state to classical orbital elements (true anomaly).
///
/// Like cartesian_to_classic() but returns true anomaly instead of mean anomaly
/// as the sixth element. Useful when the true anomaly is needed directly.
///
/// @tparam Scalar Floating-point scalar type (double or equivalent).
/// @param XV Six Cartesian state [rx, ry, rz, vx, vy, vz].
/// @param mu Gravitational parameter (km³/s² or consistent units).
/// @return Six classical elements [a, e, i, Omega, omega, nu] (nu = true anomaly).
template <class Scalar>
Vector6<Scalar> cartesian_to_classic_true(const Vector6<Scalar> &XV, Scalar mu) {

    const double PI = 3.14159265358979;

    Vector3<Scalar> R = XV.template head<3>();
    Vector3<Scalar> V = XV.template tail<3>();

    Vector3<Scalar> hvec = R.cross(V);
    Vector3<Scalar> evec = V.cross(hvec) / mu - R.normalized();

    Vector3<Scalar> nvec;
    nvec[0] = -hvec[1];
    nvec[1] = hvec[0];

    Scalar e = evec.norm();

    Scalar drv = R.dot(V);
    Scalar v = acos(evec.normalized().dot(R.normalized()));
    if (drv < 0)
        v = 2.0 * PI - v;

    Scalar Omega = acos(nvec[0] / nvec.norm());
    if (nvec[1] < 0)
        Omega = 2.0 * PI - Omega;

    Scalar w = acos(evec.normalized().dot(nvec.normalized()));
    if (evec[2] < 0)
        w = 2.0 * PI - w;

    Scalar i = acos(hvec[2] / hvec.norm());

    Scalar a = 1.0 / (2.0 / R.norm() - V.squaredNorm() / mu);

    Vector6<Scalar> oelems;

    oelems[0] = a;
    oelems[1] = e;
    oelems[2] = i;
    oelems[3] = Omega;
    oelems[4] = w;
    oelems[5] = v;
    return oelems;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////    Conversions as Tycho VectorFunctions   /////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Implementation body for the MEE → Cartesian VectorFunction.
///
/// Expression-builder implementation; the public type ModifiedToCartesian is created
/// from this via BUILD_FROM_EXPRESSION.
/// @endinternal
struct ModifiedToCartesian_Impl {
    /// @internal @brief Build the MEE → Cartesian conversion expression. @endinternal
    static auto Definition(double mu) {

        auto meelems = Arguments<6>();

        auto p = meelems.coeff<0>();
        auto f = meelems.coeff<1>();
        auto g = meelems.coeff<2>();
        auto h = meelems.coeff<3>();
        auto k = meelems.coeff<4>();
        auto L = meelems.coeff<5>();

        auto cosL = cos(L);
        auto sinL = sin(L);

        auto a2 = h * h - k * k;
        auto s2 = 1.0 + h * h + k * k;
        auto w = 1.0 + f * cosL + g * sinL;
        auto rr = p / w;

        auto Xscale = rr / s2;
        auto Vscale = sqrt(mu / p) / s2;

        auto x = (cosL + a2 * cosL + 2.0 * h * k * sinL);
        auto y = (sinL - a2 * sinL + 2.0 * h * k * cosL);
        auto z = 2.0 * (h * sinL - k * cosL);

        auto R = stack(x, y, z) * Xscale;

        auto vx = -1.0 * (sinL + a2 * sinL - 2 * h * k * cosL + g - 2 * f * h * k + a2 * g);
        auto vy = -1.0 * (-1.0 * cosL + a2 * cosL + 2 * h * k * sinL - f + 2 * g * h * k + a2 * f);
        auto vz = 2 * (h * cosL + k * sinL + f * h + g * k);

        auto V = stack(vx, vy, vz) * Vscale;

        return stack(R, V);
    }
};

/// @brief MEE → Cartesian VectorFunction (IR=6, OR=6).
///
/// Accepts MEE [p, f, g, h, k, L] and returns Cartesian state [rx, ry, rz, vx, vy, vz].
/// Constructed via BUILD_FROM_EXPRESSION from ModifiedToCartesian_Impl.
/// Constructed with a gravitational parameter argument, e.g. `ModifiedToCartesian vf(mu);`.
BUILD_FROM_EXPRESSION(ModifiedToCartesian, ModifiedToCartesian_Impl, double);

/// @internal
/// @brief Implementation body for the Cartesian → classical orbital elements VectorFunction.
///
/// Expression-builder implementation using IfElseFunction for the elliptic/hyperbolic branch.
/// The public type CartesianToClassic is created from this via BUILD_FROM_EXPRESSION.
/// @endinternal
struct CartesianToClassic_Impl {
    /// @internal @brief Build the Cartesian → classical elements conversion expression. @endinternal
    static auto Definition(double mu) {
        const double PI = 3.14159265358979;

        auto RV = Arguments<6>();

        auto ZVec = Constant<6, 3>(6, Eigen::Vector3d::UnitZ());

        auto R = RV.head<3>();
        auto V = RV.tail<3>();
        auto hvec = R.cross(V);
        auto nvec = ZVec.cross(hvec);

        auto r = R.norm();
        auto v2 = V.squared_norm();
        auto eps = v2 / 2.0 - mu / r;

        auto a = (-0.5 * mu) / eps;
        auto evec = V.cross(hvec) / mu - R.normalized();
        auto e = evec.norm();

        auto drv = R.dot(V);

        auto vtmp = acos(evec.normalized().dot(R.normalized()));

        // True-anomaly quadrant: when R·V < 0 (past periapsis, descending), the
        // true anomaly is 2*PI - vtmp. This matches the scalar cartesian_to_classic
        // and the Omega/w branches below (the original had these two branches
        // swapped, producing the wrong anomaly quadrant for descending states).
        auto v = IfElseFunction{drv < 0, 2 * PI - vtmp, vtmp};

        auto M =
            [mu]() {
                auto ev = Arguments<2>();
                auto e = ev.coeff<0>();
                auto v = ev.coeff<1>();

                auto E = 2. * atan(tan(v / 2.0) / (sqrt((1.0 + e) / (1.0 - e))));
                auto ME = E - e * sin(E);

                auto H = 2. * atanh(tan(v / 2.0) / (sqrt((1.0 + e) / (e - 1.0))));
                auto MH = e * sinh(H) - H;

                auto M = IfElseFunction{e < 1.0, ME, MH};
                return M;
            }()
                .eval(stack(e, v));

        auto Omegatmp = acos(nvec.coeff<0>() / nvec.norm());
        auto Omega = IfElseFunction{nvec.coeff<1>() < 0, 2.0 * PI - Omegatmp, Omegatmp};
        auto wtmp = acos(evec.normalized().dot(nvec.normalized()));
        auto w = IfElseFunction{evec.coeff<2>() < 0, 2.0 * PI - wtmp, wtmp};
        auto i = acos(hvec.coeff<2>() / hvec.norm());

        return stack(a, e, i, Omega, w, M);
    }
};

/// @brief Cartesian → classical orbital elements VectorFunction (IR=6, OR=6).
///
/// Accepts Cartesian state [rx, ry, rz, vx, vy, vz] and computes classical orbital
/// elements [a, e, i, Omega, omega, M], where M is the elliptic mean anomaly
/// (M = E − e·sin(E)) for e < 1 and the hyperbolic mean anomaly (M = e·sinh(H) − H)
/// for e > 1. Constructed via BUILD_FROM_EXPRESSION from CartesianToClassic_Impl.
/// Constructed with a gravitational parameter argument, e.g. `CartesianToClassic vf(mu);`.
BUILD_FROM_EXPRESSION(CartesianToClassic, CartesianToClassic_Impl, double)

// Two-body propagation entry points (propagate_cartesian / propagate_classic /
// propagate_modified) live in tycho/detail/astro/kepler/kepler_propagation.h.

} // namespace tycho::astro
