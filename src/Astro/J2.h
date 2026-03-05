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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once
#include "VectorFunctions/Tycho_VectorFunctions.h"

namespace Tycho {
struct J2Cartesian_Impl {
    /// <summary>
    /// Computes J2 effect given position vector relative to body and the north pole vector of the
    /// body
    /// </summary>
    /// <param name="mu"></param>
    /// <param name="J2"></param>
    /// <param name="Rb"></param>
    /// <returns></returns>
    static auto Definition(double mu, double J2, double Rb) {

        auto rp = Arguments<6>();

        auto r = rp.head<3>();
        auto p = rp.tail<3>().normalized();

        auto rn5 = r.normalized_power<5>();

        double Scale = 0.5 * (mu)*J2 * Rb * Rb;

        auto dotterm = r.normalized().dot(p).Square();

        auto term1 = (15.0 * dotterm - 3.0) * rn5;

        auto term2 = -6.0 * rn5.dot(p) * p;

        auto acc = Scale * (term1 + term2);

        return acc;
    }
};

BUILD_FROM_EXPRESSION(J2Cartesian, J2Cartesian_Impl, double, double, double);

struct J2Modified_Impl {
    /// <summary>
    /// Computes J2 effect given position vector relative to body and the north pole vector of the
    /// body
    /// </summary>
    /// <param name="mu"></param>
    /// <param name="J2"></param>
    /// <param name="Rb"></param>
    /// <returns></returns>
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

} // namespace Tycho