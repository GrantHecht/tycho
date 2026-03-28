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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once
#include "tycho/vector_functions.h"

namespace tycho::astro {

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

struct IdealSolarSail_Impl {
    /// Tycho VectorFunction to compute thrust output of an Ideal Solar Sail.
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

BUILD_FROM_EXPRESSION(IdealSolarSail, IdealSolarSail_Impl, double, double);

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

struct NonIdealSolarSail_Impl {
    /// Tycho VectorFunction to compute thrust output of a Non-Ideal Solar Sail.
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

BUILD_FROM_EXPRESSION(NonIdealSolarSail, NonIdealSolarSail_Impl, double, double, double, double,
                      double);

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

} // namespace tycho::astro
