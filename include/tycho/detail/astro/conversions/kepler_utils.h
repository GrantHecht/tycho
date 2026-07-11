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
#include "tycho/detail/astro/kepler/kepler_lcd_iterate.h"
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

// kepler_nan_value<Scalar>() is the LCD/IFT NaN-poison primitive (see
// kepler_lcd_iterate.h) — reused here so the hyperbolic Newton solves below
// poison their output via the exact same mechanism/constant as the rest of
// the Kepler subsystem on non-convergence.
using detail::kepler_nan_value;

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
/// @warning Elliptic branch (e < 1): the Newton iteration on E runs at most
///          MAXITERS_ELLIPTIC=17 steps from the E = M seed and tests convergence
///          on the Newton step (|dE| < TOL, 1e-12) — the same step-size
///          convention as the hyperbolic branch below and kepler_lcd_iterate.
///          On non-convergence the whole output is NaN-poisoned via
///          kepler_nan_value<Scalar>(), rather than silently returning an
///          under-converged state (OC review §1.14). The budget is 17 rather
///          than the hyperbolic branch's 15: a grid probe found a thin
///          knife-edge band of near-parabolic (e → 1⁻), small-|M| inputs whose
///          Newton step falls below TOL on iteration 16 (i.e. the 15-iteration
///          budget was one or two steps short, poisoning inputs that were in
///          fact converging), while genuinely divergent near-parabolic inputs
///          (e.g. e = 1 − 1e-9, M = 1e-8) still exhaust the larger budget and
///          poison. Convergence can still fail for near-parabolic orbits
///          sampled near periapsis (small |M|), where the update's denominator
///          `1 - e·cosE` shrinks near E ≈ 0 and the E = M seed leaves the
///          Newton basin; such inputs poison, they no longer return a
///          finite-but-wrong state with no diagnostic.
/// @warning Hyperbolic branch (e > 1): converges via an asinh(M/e) seed and a
///          step-size (|dH| < TOL) test over MAXITERS=15 steps, NaN-poisoning
///          the whole output on non-convergence (incl. sinh/cosh overflow).
///          Genuinely non-convergent near-parabolic orbits (e → 1⁺) can still
///          diverge and NaN-poison; a Barker-style seed for the near-parabolic
///          regime is a potential future enhancement, not currently
///          implemented.
template <class Scalar>
Vector6<Scalar> classic_to_cartesian(const Vector6<Scalar> &oelems, Scalar mu) {
    using std::abs;

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
        // Elliptic branch gets a larger iteration budget than the hyperbolic
        // branch below (MAXITERS=15): a thin knife-edge band of near-parabolic
        // (e -> 1-) small-|M| inputs converges to an accurate E within ~16
        // iterations under the stricter step-size test but not within 15 (see
        // the @warning above and the probe referenced there). Scoped to this
        // branch only -- the hyperbolic branch's basin/step-size behavior is
        // unrelated and is left at MAXITERS.
        const int MAXITERS_ELLIPTIC = 17;
        Scalar E = M;
        Scalar sinE;
        Scalar cosE;
        bool converged = false;
        for (int i = 0; i < MAXITERS_ELLIPTIC; i++) {
            sinE = sin(E);
            cosE = cos(E);
            Scalar fE = E - e * sinE - M;
            Scalar dE = fE / (1 - e * cosE);
            E = E - dE;
            // Step-size convergence test, matching kepler_lcd_iterate's
            // |dX| <= Xtol convention (and the hyperbolic branch below).
            // Scale-invariant near a well-conditioned root, unlike the former
            // raw-residual break (|fE| < TOL *before* stepping), which accepted
            // an under-converged E near periapsis of near-parabolic orbits: a
            // tiny residual fE need not imply a tiny step there because the
            // 1 - e*cosE denominator shrinks.
            if (abs(dE) < TOL) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            // Non-convergence (e.g. e -> 1- near periapsis, where the
            // 1 - e*cosE denominator shrinks and the E = M seed leaves the
            // Newton basin) must not silently return a finite-but-wrong state
            // — poison the whole output, mirroring the hyperbolic branch below
            // and the LCD/IFT Kepler paths.
            return Vector6<Scalar>::Constant(kepler_nan_value<Scalar>());
        }
        // Refresh sin/cos at the accepted E: the loop breaks after taking the
        // final sub-tolerance step, so the in-loop sinE/cosE are one step stale
        // and vx/vy below consume them.
        sinE = sin(E);
        cosE = cos(E);
        Scalar v = 2.0 * atan2(sqrt(1. + e) * sin(E / 2.0), sqrt(1. - e) * cos(E / 2.0));
        Scalar rc = a * (1. - e * cos(E));
        Scalar vc = sqrt(mu * a) / rc;

        x = rc * cos(v);
        y = rc * sin(v);
        vx = -vc * sinE;
        vy = vc * sqrt(1. - e * e) * cosE;

    } else { // Hyperbolic
        // Gooding-class initial guess: asinh(M/e) tracks the root far more
        // closely than H = M for moderate/large |M|, where the H = M seed
        // can leave the Newton iterate outside the basin of convergence.
        Scalar H = asinh(M / e);
        bool converged = false;

        for (int i = 0; i < MAXITERS; i++) {
            Scalar sinhHi = sinh(H);
            Scalar coshHi = cosh(H);
            Scalar fH = e * sinhHi - H - M;
            Scalar dH = fH / (e * coshHi - 1);
            H = H - dH;
            // Step-size convergence test, matching kepler_lcd_iterate's
            // |dX| <= Xtol convention (same 1e-12 magnitude as LCD's default
            // Xtol): scale-invariant near a well-conditioned root, unlike a
            // raw-residual test whose FP noise floor grows as O(eps*M) (three
            // O(M) terms cancel in fH) and would falsely reject valid states
            // for |M| beyond a few thousand.
            if (abs(dH) < TOL) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            // Non-convergence (incl. NaN steps from sinh/cosh overflow, which
            // fail the |dH| < TOL comparison every iteration) must not
            // silently propagate a finite-but-wrong state — poison the whole
            // output, mirroring the LCD/IFT Kepler paths.
            return Vector6<Scalar>::Constant(kepler_nan_value<Scalar>());
        }
        // Evaluate sinh/cosh at the accepted H (the loop breaks after taking
        // the final sub-tolerance step, so in-loop values are one step stale).
        Scalar sinhH = sinh(H);
        Scalar coshH = cosh(H);
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
/// @warning `v`, `Omega`, `w`, and `i` are each computed via `acos()` of a
///          normalized-vector dot product. Mathematically these arguments lie
///          in [-1, 1], but floating-point round-off (e.g. near-collinear
///          `evec`/`R`, near-zero eccentricity, or near-zero/near-180°
///          inclination) can push an argument marginally outside [-1, 1],
///          making `acos()` return NaN with no clamp/guard.
/// @warning `M` is derived from `v` via `atan(tan(v/2)/sqrt((1+e)/(1-e)))`
///          (elliptic) or `atanh(tan(v/2)/sqrt((1+e)/(e-1)))` (hyperbolic).
///          `atanh`'s argument must lie strictly in (-1, 1); near-parabolic
///          orbits (e → 1) or `v → π` push the argument toward ±1
///          (`atanh` → ±∞) or beyond (`atanh` → NaN), again unguarded.
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
/// @warning `M` is derived from `v` via `atan(tan(v/2)/sqrt((1+e)/(1-e)))`
///          (elliptic) or `atanh(tan(v/2)/sqrt((1+e)/(e-1)))` (hyperbolic),
///          the same domain-sensitive formula used by cartesian_to_classic().
///          `atanh`'s argument must lie strictly in (-1, 1); near-parabolic
///          orbits (e → 1) or `v → π` push the argument toward ±1
///          (`atanh` → ±∞) or beyond (`atanh` → NaN), unguarded.
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
/// @warning Elliptic branch (e < 1): same MAXITERS_ELLIPTIC=17 Newton loop on E
///          as classic_to_cartesian(), converging on the step size (|dE| <
///          TOL) and NaN-poisoning the whole output on non-convergence — see
///          the warning there (including the knife-edge-band rationale for
///          the 17-iteration budget). Near-parabolic orbits (e → 1⁻) sampled
///          near periapsis can exhaust MAXITERS_ELLIPTIC (the `1 - e·cosE`
///          denominator shrinks) and are poisoned rather than silently
///          under-converged.
/// @warning Hyperbolic branch (e > 1): NaN-poisons on non-convergence over
///          MAXITERS=15 steps (see classic_to_cartesian()); near-parabolic
///          orbits (e → 1⁺) can still genuinely diverge and NaN-poison.
template <class Scalar>
Vector6<Scalar> classic_to_modified(const Vector6<Scalar> &oelems, Scalar mu) {
    using std::abs;

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
        // See classic_to_cartesian() for why the elliptic branch gets a
        // larger budget (17) than the hyperbolic branch below (MAXITERS=15):
        // a knife-edge band of near-parabolic small-|M| inputs needs ~16
        // iterations to fall below the step-size tolerance.
        const int MAXITERS_ELLIPTIC = 17;
        Scalar E = M;
        bool converged = false;
        for (int i = 0; i < MAXITERS_ELLIPTIC; i++) {
            Scalar sinE = sin(E);
            Scalar cosE = cos(E);
            Scalar fE = E - e * sinE - M;
            Scalar dE = fE / (1 - e * cosE);
            E = E - dE;
            // Step-size convergence test, matching kepler_lcd_iterate's
            // |dX| <= Xtol convention (and the hyperbolic branch below); see
            // classic_to_cartesian() for the near-parabolic rationale.
            if (abs(dE) < TOL) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            // Non-convergence near e -> 1- (periapsis) must not silently
            // propagate a finite-but-wrong state — poison, mirroring the
            // hyperbolic branch and classic_to_cartesian().
            return Vector6<Scalar>::Constant(kepler_nan_value<Scalar>());
        }
        v = 2.0 * atan2(sqrt(1. + e) * sin(E / 2.0), sqrt(1. - e) * cos(E / 2.0));
    } else { // Hyperbolic
        // Gooding-class initial guess: asinh(M/e) tracks the root far more
        // closely than H = M for moderate/large |M|, where the H = M seed
        // can leave the Newton iterate outside the basin of convergence.
        Scalar H = asinh(M / e);
        bool converged = false;

        for (int i = 0; i < MAXITERS; i++) {
            Scalar sinhHi = sinh(H);
            Scalar coshHi = cosh(H);
            Scalar fH = e * sinhHi - H - M;
            Scalar dH = fH / (e * coshHi - 1);
            H = H - dH;
            // Step-size convergence test, matching kepler_lcd_iterate's
            // |dX| <= Xtol convention (same 1e-12 magnitude as LCD's default
            // Xtol): scale-invariant near a well-conditioned root, unlike a
            // raw-residual test whose FP noise floor grows as O(eps*M) (three
            // O(M) terms cancel in fH) and would falsely reject valid states
            // for |M| beyond a few thousand.
            if (abs(dH) < TOL) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            // Non-convergence (incl. NaN steps from sinh/cosh overflow, which
            // fail the |dH| < TOL comparison every iteration) must not
            // silently propagate a finite-but-wrong state — poison the whole
            // output, mirroring the LCD/IFT Kepler paths.
            return Vector6<Scalar>::Constant(kepler_nan_value<Scalar>());
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
/// @warning `nu` (v), `Omega`, `w`, and `i` are each computed via `acos()` of a
///          normalized-vector dot product — see the identical warning on
///          cartesian_to_classic(). Round-off can push an argument marginally
///          outside [-1, 1] (near-collinear vectors, near-zero eccentricity or
///          inclination), making `acos()` return NaN with no clamp/guard.
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
    /// @internal @brief Build the Cartesian → classical elements conversion expression.
    /// @endinternal
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

        // NOTE (OC §3.9): vtmp/Omegatmp/wtmp/i below are each acos() of a
        // normalized-vector dot product; the argument is mathematically in
        // [-1, 1] but round-off can push it marginally outside, and acos()
        // returns NaN unguarded — same caveat as the scalar
        // cartesian_to_classic() this expression mirrors.
        auto vtmp = acos(evec.normalized().dot(R.normalized()));

        // True-anomaly quadrant: when R·V < 0 (past periapsis, descending), the
        // true anomaly is 2*PI - vtmp. This matches the scalar cartesian_to_classic
        // and the Omega/w branches below (the original had these two branches
        // swapped, producing the wrong anomaly quadrant for descending states).
        auto v = IfElseFunction{drv < 0, 2 * PI - vtmp, vtmp};

        // NOTE (OC §3.9): the atanh() branch below (MH) requires its argument
        // strictly in (-1, 1); near-parabolic e -> 1 or v -> pi push it toward
        // +-1 (atanh -> +-inf) or beyond (atanh -> NaN), unguarded — same
        // caveat as the scalar cartesian_to_classic()/modified_to_classic().
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
