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
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "astro/tycho_astro.h"

#include "tycho/detail/vf/operators/root_finder.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;

template <> struct TychoBind<ModifiedToCartesian> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<ModifiedToCartesian>(m, name).def(nb::init<double>());
        bind::DenseBaseBuild<ModifiedToCartesian>(obj);
    }
};

template <> struct TychoBind<CartesianToClassic> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<CartesianToClassic>(m, name).def(nb::init<double>());
        bind::DenseBaseBuild<CartesianToClassic>(obj);
    }
};

template <> struct TychoBind<CartesianToMEE> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<CartesianToMEE>(m, name).def(nb::init<double>());
        bind::DenseBaseBuild<CartesianToMEE>(obj);
    }
};

void kepler_utils_build(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::kepler_utils_build(FunctionRegistry &reg, nb::module_ &m) {

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////              Conversions                  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    m.def("cartesian_to_classic",
          [](const Vector6<double> &XV, double mu) { return cartesian_to_classic(XV, mu); });
    m.def("cartesian_to_modified",
          [](const Vector6<double> &XV, double mu) { return cartesian_to_modified(XV, mu); });
    m.def("classic_to_cartesian", [](const Vector6<double> &oelems, double mu) {
        return classic_to_cartesian(oelems, mu);
    });
    m.def("classic_to_modified",
          [](const Vector6<double> &oelems, double mu) { return classic_to_modified(oelems, mu); });

    m.def("modified_to_cartesian", [](const Vector6<double> &meelems, double mu) {
        return modified_to_cartesian(meelems, mu);
    });
    m.def("modified_to_classic", [](const Vector6<double> &meelems, double mu) {
        return modified_to_classic(meelems, mu);
    });
    m.def("cartesian_to_classic_true",
          [](const Vector6<double> &XV, double mu) { return cartesian_to_classic_true(XV, mu); });

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////       Conversions as VectorFunctions      /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    m.def("modified_to_cartesian", [](const GenericFunction<-1, -1> &meelems, double mu) {
        return GenericFunction<-1, -1>(ModifiedToCartesian(mu).eval(meelems));
    });

    reg.build_register<ModifiedToCartesian>(m, "ModifiedToCartesian");

    m.def("cartesian_to_classic", [](const GenericFunction<-1, -1> &RV, double mu) {
        return GenericFunction<-1, -1>(CartesianToClassic(mu).eval(RV));
    });

    reg.build_register<CartesianToClassic>(m, "CartesianToClassic");

    m.def("cartesian_to_modified", [](const GenericFunction<-1, -1> &RV, double mu) {
        return GenericFunction<-1, -1>(CartesianToMEE(mu).eval(RV));
    });

    reg.build_register<CartesianToMEE>(m, "CartesianToMEE");

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////              Propagators                  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    // Python contract: propagate_* raises RuntimeError on non-convergence
    // or NaN-poisoned output.  The C++ kernels return a NaN-filled Vector6
    // (PSIOPT step-rejection semantics), but Python users expect exceptions;
    // translate at the boundary.  Detection via allFinite() is cheap (6
    // doubles) and avoids exposing the internal KeplerLCDResult::converged
    // flag.  A structured KeplerLCDStatus enum that distinguishes the
    // bailout causes (max_iters / hyp_guard / NaN) is the more principled
    // diagnostic surface and is intentionally deferred to a follow-up; the
    // enriched messages below list the typical causes so users can debug
    // without it.
    m.def("propagate_cartesian", [](const Vector6<double> &RV, double dt, double mu) {
        const auto rv = propagate_cartesian(RV, dt, mu);
        if (!rv.allFinite())
            throw std::runtime_error(
                "propagate_cartesian: LCD iteration did not converge "
                "(check dt magnitude, hyperbolic asymptote at sqrt(-alpha)*X = 30, "
                "or non-finite inputs)");
        return rv;
    });
    // propagate_classic / propagate_modified do not run an LCD iteration —
    // input validation in the C++ overloads (see kepler_propagation.h) raises
    // std::invalid_argument for mu <= 0 or non-finite/zero semi-major axis,
    // which nanobind translates to ValueError.  This RuntimeError fires only
    // if the analytic path itself produces non-finite output despite valid
    // inputs (e.g. a numerically extreme intermediate via modified_to_classic).
    m.def("propagate_classic", [](const Vector6<double> &oelems, double dt, double mu) {
        const auto out = propagate_classic(oelems, dt, mu);
        if (!out.allFinite())
            throw std::runtime_error("propagate_classic: produced non-finite output");
        return out;
    });

    m.def("propagate_modified", [](const Vector6<double> &meelems, double dt, double mu) {
        const auto out = propagate_modified(meelems, dt, mu);
        if (!out.allFinite())
            throw std::runtime_error("propagate_modified: produced non-finite output");
        return out;
    });
}
