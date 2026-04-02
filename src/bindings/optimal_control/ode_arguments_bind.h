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

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "dense_function_base_bind.h"
#include "function_registry.h"
#include "tycho/detail/optimal_control/core/ode_arguments.h"

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

template <int _XV, int _UV, int _PV> struct TychoBind<ODEArguments<_XV, _UV, _PV>> {
    using Derived = ODEArguments<_XV, _UV, _PV>;
    using Base = typename Derived::Base;
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived, Base>(m, name);
        obj.def(nb::init<int, int, int>());
        obj.def(nb::init<int, int>());
        obj.def(nb::init<int>());

        bind::DenseBaseBuild<Derived>(obj);

        obj.def("x_vec", [](const Derived &a) { return a.segment(0, a.x_vars()); });
        obj.def("x_var", [](const Derived &a, int i) { return a.segment(0, a.x_vars()).coeff(i); });
        obj.def("xt_vec", [](const Derived &a) { return a.segment(0, a.xt_vars()); });
        obj.def("t_var", [](const Derived &a) { return a.coeff(a.t_var()); });
        obj.def("u_vec", [](const Derived &a) { return a.segment(a.xt_vars(), a.u_vars()); });
        obj.def("u_var", [](const Derived &a, int i) {
            return a.segment(a.xt_vars(), a.u_vars()).coeff(i);
        });
        obj.def("p_vec", [](const Derived &a) { return a.segment(a.xtu_vars(), a.p_vars()); });
        obj.def("p_var", [](const Derived &a, int i) {
            return a.segment(a.xtu_vars(), a.p_vars()).coeff(i);
        });
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
