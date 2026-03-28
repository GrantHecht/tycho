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
//   - pybind11 Build() methods moved to bindings layer (PR 2)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Binding code extracted from Tycho_Extensions.cpp into this file,
//     separating it from pure C++ ODE model definitions (PR 4)
// =============================================================================

#include "Tycho_Extensions.h"

namespace tycho {

void ExtensionsBuild(FunctionRegistry &reg, nb::module_ &extmod);

template <> struct TychoBind<CR3BPAD> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<CR3BPAD>(m, name).def(nb::init<double>());
        bind::DenseBaseBuild<CR3BPAD>(obj);
    }
};

template <> struct TychoBind<ModifiedDynamicsAD> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<ModifiedDynamicsAD>(m, name).def(nb::init<double>());
        bind::DenseBaseBuild<ModifiedDynamicsAD>(obj);
    }
};

} // namespace tycho

void tycho::ExtensionsBuild(FunctionRegistry &reg, nb::module_ &extmod) {

    extmod.def("cpp_cr3bp", [](double mu) {
        // Example of how to write CR3BP dynamics as C++ vector function and bind it to python
        // After compilation can import in python w/ tycho.Extensions.cpp_cr3bp(mu)
        // Docs on CPP vector function interface forthcoming but in general it mimics python

        auto args = Arguments<7>();

        auto X = args.head<3>();

        auto V = args.segment<3, 3>(); //.segment<size,start>()

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

        auto rotterms = stack(2.0 * ydot + x, (-2.0) * xdot + y);

        auto acc = rotterms.padded_lower<1>() - (1.0 - mu) * dvec.normalized_power<3>() -
                   mu * rvec.normalized_power<3>();

        auto ode = stack(V, acc);

        return GenericFunction<-1, -1>(ode); // Wrap as dynamic sized generic vector function
    });

    reg.Build_Register<CR3BPAD>(extmod, "cpp_cr3bp_ad");
    reg.Build_Register<ModifiedDynamicsAD>(extmod, "ModifiedDynamicsAD");
}
