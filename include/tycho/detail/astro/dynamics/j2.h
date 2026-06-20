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
/// @internal
/// @brief Implementation body for the J2 Cartesian perturbation acceleration VectorFunction.
///
/// Takes [r(3), north_pole(3)] and returns the J2 gravitational acceleration
/// contribution. The north_pole vector need not be pre-normalized; it is normalized
/// internally. The public type J2Cartesian is created from this via BUILD_FROM_EXPRESSION.
/// @endinternal
struct J2Cartesian_Impl {
    /// @internal @brief Build the J2 Cartesian acceleration expression. @endinternal
    static auto Definition(double mu, double J2, double Rb) {

        auto rp = Arguments<6>();

        auto r = rp.head<3>();
        auto p = rp.tail<3>().normalized();

        auto rn5 = r.normalized_power<5>();

        double Scale = 0.5 * (mu)*J2 * Rb * Rb;

        auto dotterm = r.normalized().dot(p).square();

        auto term1 = (15.0 * dotterm - 3.0) * rn5;

        auto term2 = -6.0 * rn5.dot(p) * p;

        auto acc = Scale * (term1 + term2);

        return acc;
    }
};

/// @brief J2 perturbation acceleration in Cartesian coordinates.
///
/// Input: [rx, ry, rz, north_px, north_py, north_pz] (IR=6). Output: J2 acceleration (3-vector).
/// The north-pole direction [north_px, north_py, north_pz] need not be pre-normalized;
/// it is normalized internally.
/// Constructed via BUILD_FROM_EXPRESSION from J2Cartesian_Impl.
BUILD_FROM_EXPRESSION(J2Cartesian, J2Cartesian_Impl, double, double, double);

/// @internal
/// @brief Incomplete implementation body for J2 perturbation in MEE coordinates (stub only).
/// @endinternal
struct J2Modified_Impl {
    /// @internal @brief Placeholder Definition — not yet implemented, do not use. @endinternal
    static auto Definition(double mu, double J2, double Rb) {

        auto args = Arguments<6>();

        auto p = args.coeff<0>();
        auto f = args.coeff<1>();
        auto g = args.coeff<2>();
        auto h = args.coeff<3>();
        auto k = args.coeff<4>();
        auto L = args.coeff<5>();

        auto sinL = sin(L);
        auto cosL = cos(L);

        // auto acc = Scale * (term1 + term2);

        // return acc;
    }
};

// BUILD_FROM_EXPRESSION(J2Modified, J2Modified_Impl, double, double,double );

} // namespace tycho::astro
