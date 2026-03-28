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
#include "tycho/detail/astro/kepler_propagator.h"
#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/vector_functions.h"

namespace tycho::astro {

// Import cross-namespace types from vf and oc.
using vf::Arguments;
using vf::StackedOutputs;
using oc::ODEPhase;
using oc::ODESize;
using oc::ODE_Expression;

struct Kepler_Impl : ODESize<6, 0, 0> {
    static auto Definition(double mu) {
        auto args = Arguments<7>();
        auto X = args.head<3>();
        auto V = args.segment<3, 3>();
        auto rvec = X;
        auto acc = (-mu) * rvec.normalized_power<3>();
        auto ode = StackedOutputs{V, acc};
        return ode;
    }
};
struct KeplerPhase;

struct Kepler : ODE_Expression<Kepler, Kepler_Impl, double> {
    using Base = ODE_Expression<Kepler, Kepler_Impl, double>;
    using Base::Base;
    double mu = 1.0;
    Kepler(double mu) : Base(mu) { this->mu = mu; }
};

struct KeplerPhase : ODEPhase<Kepler> {
    using Base = ODEPhase<Kepler>;
    using Base::Base;
    bool UseKeplerPropagator = true;

    tycho::solvers::ConstraintInterface make_shooter() {
        if (UseKeplerPropagator) {
            auto kprop = KeplerPropagator(this->ode.mu);
            auto Args = Arguments<14>();
            auto X1 = Args.head<6>();
            auto X2 = Args.segment<6, 7>();
            auto t1 = Args.coeff<6>();
            auto t2 = Args.coeff<13>();

            auto Xk1 = StackedOutputs{X1, (t2 - t1) / 2.0};
            auto Xk2 = StackedOutputs{X2, (t1 - t2) / 2.0};

            auto shooter = kprop.eval(Xk1) - kprop.eval(Xk2);

            return tycho::solvers::ConstraintInterface(shooter);
        } else {
            return Base::make_shooter();
        }
    }
};

} // namespace tycho::astro
