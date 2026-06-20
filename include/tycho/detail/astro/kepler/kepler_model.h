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
#include "tycho/detail/astro/kepler/kepler_propagator.h"
#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/vector_functions.h"

namespace tycho::astro {

// Import cross-namespace types from vf and oc.
using oc::ODEPhase;
using oc::ODESize;
using oc::StaticODE_Expression;
using vf::Arguments;
using vf::StackedOutputs;

/// @internal
/// @brief ODE implementation body for the Kepler two-body equations of motion.
///
/// Defines the 6-state [rx, ry, rz, vx, vy, vz] dynamics dX/dt = [V, -mu*R/|R|^3].
/// @endinternal
struct Kepler_Impl : ODESize<6, 0, 0> {
    /// @internal @brief Build the ODE expression for the given gravitational parameter. @endinternal
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

/// @brief 6-state Keplerian two-body ODE (no perturbations) for use with ODEPhase.
///
/// State vector: [rx, ry, rz, vx, vy, vz]. Dynamics: dR/dt = V, dV/dt = -mu*R/|R|^3.
struct Kepler : StaticODE_Expression<Kepler, Kepler_Impl, double> {
    using Base = StaticODE_Expression<Kepler, Kepler_Impl, double>; ///< @internal CRTP base alias. @endinternal
    using Base::Base;
    double mu_ = 1.0; ///< Gravitational parameter.
    /// @brief Construct with the given gravitational parameter.
    /// @param[in] mu Gravitational parameter.
    Kepler(double mu) : Base(mu) { this->mu_ = mu; }
};

/// @brief ODEPhase for Keplerian two-body propagation with an optional KeplerPropagator shooting backbone.
///
/// When `use_kepler_propagator_` is true (the default), shooting segments use the
/// high-accuracy LCD Kepler propagator instead of numerical integration.
/// The shooting constraint uses a midpoint (central) shooting defect: each boundary
/// state is propagated by the LCD Kepler propagator to the interval midpoint and the
/// two results are differenced (the analytic analogue of CentralShootingDefect),
/// NOT a simple forward-propagation defect.
struct KeplerPhase : ODEPhase<Kepler> {
    using Base = ODEPhase<Kepler>; ///< @internal CRTP base alias. @endinternal
    using Base::Base;
    bool use_kepler_propagator_ = true; ///< Use the analytic KeplerPropagator for shooting segments.

    /// @brief Build the shooting constraint, optionally using KeplerPropagator.
    /// @return ConstraintInterface for the shooting defect.
    tycho::solvers::ConstraintInterface make_shooter() {
        if (use_kepler_propagator_) {
            auto kprop = KeplerPropagator(this->ode_.mu_);
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
