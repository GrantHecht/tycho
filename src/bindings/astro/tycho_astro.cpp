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

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
void astro_build(FunctionRegistry &reg, nb::module_ &m);
void BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m);
void BuildKeplerIntegrator(FunctionRegistry &reg, nb::module_ &m);
void kepler_utils_build(FunctionRegistry &reg, nb::module_ &m);
void lambert_solvers_build(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::astro_build(FunctionRegistry &reg, nb::module_ &m) {
    auto mod = m.def_submodule("astro");

    BuildKeplerMod(reg, mod);
    BuildKeplerIntegrator(reg, mod);
    kepler_utils_build(reg, mod);
    lambert_solvers_build(reg, mod);

    /////////////////////////////////////////////////////////////
    //////////// Binding Misc CPP Functions here for now ////////
    /////////////////////////////////////////////////////////////

    mod.def("modified_dynamics",
            [](double mu) { return GenericFunction<-1, -1>(MEEDynamics(mu)); });

    mod.def("cartesian_dynamics",
            [](double mu) { return GenericFunction<-1, -1>(CartesianDynamics(mu)); });

    mod.def("crtbp_dynamics", [](double mu) { return GenericFunction<-1, -1>(CRTBPDynamics(mu)); });

    mod.def("j2_cartesian", [](double mu, double J2, double Rb) {
        return GenericFunction<-1, -1>(J2Cartesian_Impl::Definition(mu, J2, Rb));
    });

    mod.def("non_ideal_solar_sail", [](double mu, double beta, double n1, double n2, double t1) {
        return GenericFunction<-1, -1>(NonIdealSolarSail_Impl::Definition(mu, beta, n1, n2, t1));
    });
}
