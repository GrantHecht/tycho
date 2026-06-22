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
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::VectorExpression;
using vf::VectorFunction;

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Implementation body for the ideal solar-sail acceleration VectorFunction.
///
/// Input: [rx, ry, rz, nx, ny, nz] (position and sail normal). Output: 3-vector acceleration.
/// @endinternal
struct IdealSolarSail_Impl {
    /// @internal @brief Build the ideal solar-sail acceleration expression. @endinternal
    static auto Definition(double mu, double beta) {

        double scale = mu * beta;
        auto RN = Arguments<6>();
        auto r = RN.head<3>();
        auto n = RN.tail<3>();

        auto ndr = r.dot(n);
        auto ndr2 = ndr * ndr;
        auto acc = scale * (ndr2 * r.inverse_norm_power<4>() * n.normalized_power<3>());

        return acc;
    }
};

/// @brief Ideal solar-sail thrust acceleration VectorFunction.
///
/// Input: [rx, ry, rz, nx, ny, nz] (IR=6). Output: sail acceleration (3-vector).
/// Constructed via BUILD_FROM_EXPRESSION from IdealSolarSail_Impl.
BUILD_FROM_EXPRESSION(IdealSolarSail, IdealSolarSail_Impl, double, double);

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Implementation body for the non-ideal solar-sail acceleration VectorFunction.
///
/// Input: [rx, ry, rz, nx, ny, nz] (position and sail normal). Output: 3-vector acceleration.
/// Parameters: mu, beta (lightness number), n1 (normal force efficiency), n2 (absorption efficiency),
/// t1 (tangential force efficiency).
/// @endinternal
struct NonIdealSolarSail_Impl {
    /// @internal @brief Build the non-ideal solar-sail acceleration expression. @endinternal
    static auto Definition(double mu, double beta, double n1, double n2, double t1) {
        double scale = mu * beta / 2.0;

        auto RN = Arguments<6>();
        auto r = RN.head<3>();
        auto n = RN.tail<3>();

        auto ndr = r.dot(n);
        auto rn = r.norm() * n.norm();
        auto ncrn = n.cross(r).cross(n);
        auto n3dr4 = n.normalized_power<3>().dot(r.normalized_power<4>());

        auto acc = n3dr4 * (((n1 * scale) * ndr + (n2 * scale) * rn) * n + (t1 * scale) * ncrn);
        return acc;
    }
};

/// @brief Non-ideal solar-sail thrust acceleration VectorFunction.
///
/// Input: [rx, ry, rz, nx, ny, nz] (IR=6). Output: sail acceleration (3-vector).
/// Constructed via BUILD_FROM_EXPRESSION from NonIdealSolarSail_Impl.
BUILD_FROM_EXPRESSION(NonIdealSolarSail, NonIdealSolarSail_Impl, double, double, double, double,
                      double);

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

} // namespace tycho::astro
