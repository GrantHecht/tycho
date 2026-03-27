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
#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/vector_functions.h"

namespace Tycho {

struct CR3BP_Impl : ODESize<6, 0, 0> {
    static auto Definition(double mu) {
        auto args = Arguments<7>();

        auto X = args.head<3>();

        auto V = args.segment<3, 3>();

        Vector3<double> p1loc;
        p1loc[0] = -mu;

        Vector3<double> p2loc;
        p2loc[0] = 1.0 - mu;

        auto dvec = X - p1loc;
        auto rvec = X - p2loc;

        auto x = X.coeff<0>();

        auto y = X.coeff<1>();

        auto xdot = V.coeff<0>();
        auto ydot = V.coeff<1>();

        auto rotterms = StackedOutputs{2.0 * ydot + x, (-2.0) * xdot + y};

        auto acc = rotterms.padded_lower<1>() - (1.0 - mu) * dvec.normalized_power<3>() -
                   mu * rvec.normalized_power<3>();

        auto ode = StackedOutputs{V, acc};

        return ode;
    }
};

BUILD_ODE_FROM_EXPRESSION(CR3BP, CR3BP_Impl, double);

} // namespace Tycho
