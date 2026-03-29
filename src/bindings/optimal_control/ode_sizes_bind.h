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

// Free-function binding helper for ODESize member definitions.
// Used by ODEBind.h and any other code that needs to expose ODE size members.

namespace tycho::bind {

using namespace tycho::utils;

template <int XV, int UV, int PV, class Derived, class Obj> void ODESizeBuild(Obj &obj) {
    obj.def("x_vars", &Derived::x_vars);
    obj.def("u_vars", &Derived::u_vars);
    obj.def("p_vars", &Derived::p_vars);
    obj.def("t_var", &Derived::t_var);

    obj.def("xt_vars", &Derived::xt_vars);
    obj.def("xtu_vars", &Derived::xtu_vars);
    obj.def("xtup_vars", &Derived::xtu_p_vars);

    obj.def("x_idxs", nb::overload_cast<>(&Derived::x_idxs, nb::const_));
    obj.def("x_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::x_idxs, nb::const_));

    obj.def("xt_idxs", nb::overload_cast<>(&Derived::xt_idxs, nb::const_));
    obj.def("xt_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::xt_idxs, nb::const_));

    obj.def("xtu_idxs", nb::overload_cast<>(&Derived::xtu_idxs, nb::const_));
    obj.def("xtu_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::xtu_idxs, nb::const_));

    obj.def("u_idxs", nb::overload_cast<>(&Derived::u_idxs, nb::const_));
    obj.def("u_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::u_idxs, nb::const_));

    obj.def("p_idxs", nb::overload_cast<>(&Derived::p_idxs, nb::const_));
    obj.def("p_idxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::p_idxs, nb::const_));

    obj.def("add_idx",
            nb::overload_cast<const std::string &, const Eigen::VectorXi &>(&Derived::add_idx));
    obj.def("get_idxs", [](const Derived &self) {
        nb::dict result;
        for (const auto &[k, v] : self.get_idxs()) {
            result[nb::cast(k)] = nb::cast(v);
        }
        return result;
    });
    obj.def("set_idxs", [](Derived &self, nb::dict d) {
        FlatMap<std::string, Eigen::VectorXi> fm;
        for (auto [k, v] : d) {
            fm[nb::cast<std::string>(k)] = nb::cast<Eigen::VectorXi>(v);
        }
        self.set_idxs(fm);
    });
    obj.def("idx", &Derived::idx);
}

} // namespace tycho::bind

#endif // TYCHO_PYTHON_BINDINGS
