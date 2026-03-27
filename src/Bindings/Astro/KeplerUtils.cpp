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
//   - Namespace: Tycho
// =============================================================================

#include "Astro/tycho_astro.h"

#include "tycho/detail/vf/operators/root_finder.h"

namespace Tycho {

template <> struct TychoBind<ModifiedToCartesian> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<ModifiedToCartesian>(m, name).def(nb::init<double>());
        Bind::DenseBaseBuild<ModifiedToCartesian>(obj);
    }
};

template <> struct TychoBind<CartesianToClassic> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<CartesianToClassic>(m, name).def(nb::init<double>());
        Bind::DenseBaseBuild<CartesianToClassic>(obj);
    }
};

void KeplerUtilsBuild(FunctionRegistry &reg, nb::module_ &m);
} // namespace Tycho

void Tycho::KeplerUtilsBuild(FunctionRegistry &reg, nb::module_ &m) {

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
    ////////////////////         Conversions as ASSET Functions    /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    m.def("modified_to_cartesian", [](const GenericFunction<-1, -1> &meelems, double mu) {
        return GenericFunction<-1, -1>(ModifiedToCartesian(mu).eval(meelems));
    });

    reg.Build_Register<ModifiedToCartesian>(m, "ModifiedToCartesian");

    m.def("cartesian_to_classic", [](const GenericFunction<-1, -1> &RV, double mu) {
        return GenericFunction<-1, -1>(CartesianToClassic(mu).eval(RV));
    });

    reg.Build_Register<CartesianToClassic>(m, "CartesianToClassic");

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////              Propagators                  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    m.def("propagate_cartesian", [](const Vector6<double> &RV, double dt, double mu) {
        return propagate_cartesian(RV, dt, mu);
    });
    m.def("propagate_classic", [](const Vector6<double> &oelems, double dt, double mu) {
        return propagate_classic(oelems, dt, mu);
    });

    m.def("propagate_modified", [](const Vector6<double> &meelems, double dt, double mu) {
        return propagate_modified(meelems, dt, mu);
    });
}
