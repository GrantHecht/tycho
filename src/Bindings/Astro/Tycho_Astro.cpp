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
#include "tycho/detail/astro/cr3bp_model.h"

namespace Tycho {
void AstroBuild(FunctionRegistry &reg, nb::module_ &m);
void BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m);
void KeplerUtilsBuild(FunctionRegistry &reg, nb::module_ &m);
void LambertSolversBuild(FunctionRegistry &reg, nb::module_ &m);
} // namespace Tycho

void Tycho::AstroBuild(FunctionRegistry &reg, nb::module_ &m) {
    auto mod = m.def_submodule("Astro");

    BuildKeplerMod(reg, mod);
    KeplerUtilsBuild(reg, mod);
    LambertSolversBuild(reg, mod);

    /////////////////////////////////////////////////////////////
    //////////// Binding Misc CPP Functions here for now ////////
    /////////////////////////////////////////////////////////////

    mod.def("ModifiedDynamics", [](double mu) { return GenericFunction<-1, -1>(MEEDynamics(mu)); });

    mod.def("J2Cartesian", [](double mu, double J2, double Rb) {
        return GenericFunction<-1, -1>(J2Cartesian_Impl::Definition(mu, J2, Rb));
    });

    mod.def("NonIdealSolarSail", [](double mu, double beta, double n1, double n2, double t1) {
        return GenericFunction<-1, -1>(NonIdealSolarSail_Impl::Definition(mu, beta, n1, n2, t1));
    });
}
