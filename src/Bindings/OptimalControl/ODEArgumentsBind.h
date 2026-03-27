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

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "DenseFunctionBaseBind.h"
#include "FunctionRegistry.h"
#include "tycho/detail/optimal_control/core/ode_arguments.h"

namespace Tycho {

template <int _XV, int _UV, int _PV> struct TychoBind<ODEArguments<_XV, _UV, _PV>> {
    using Derived = ODEArguments<_XV, _UV, _PV>;
    using Base = typename Derived::Base;
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived, Base>(m, name);
        obj.def(nb::init<int, int, int>());
        obj.def(nb::init<int, int>());
        obj.def(nb::init<int>());

        Bind::DenseBaseBuild<Derived>(obj);

        obj.def("XVec", [](const Derived &a) { return a.segment(0, a.XVars()); });
        obj.def("XVar", [](const Derived &a, int i) { return a.segment(0, a.XVars()).coeff(i); });
        obj.def("XtVec", [](const Derived &a) { return a.segment(0, a.XtVars()); });
        obj.def("TVar", [](const Derived &a) { return a.coeff(a.TVar()); });
        obj.def("UVec", [](const Derived &a) { return a.segment(a.XtVars(), a.UVars()); });
        obj.def("UVar",
                [](const Derived &a, int i) { return a.segment(a.XtVars(), a.UVars()).coeff(i); });
        obj.def("PVec", [](const Derived &a) { return a.segment(a.XtUVars(), a.PVars()); });
        obj.def("PVar",
                [](const Derived &a, int i) { return a.segment(a.XtUVars(), a.PVars()).coeff(i); });
    }
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
