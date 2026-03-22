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

// Free-function binding helper for ODESize member definitions.
// Used by ODEBind.h and any other code that needs to expose ODE size members.

namespace Tycho::Bind {

template <int XV, int UV, int PV, class Derived, class Obj> void ODESizeBuild(Obj &obj) {
    obj.def("XVars", &Derived::XVars);
    obj.def("UVars", &Derived::UVars);
    obj.def("PVars", &Derived::PVars);
    obj.def("TVar", &Derived::TVar);
    obj.def("tVar", &Derived::TVar); // Capital is inconsistent in hindsight

    obj.def("XtVars", &Derived::XtVars);
    obj.def("XtUVars", &Derived::XtUVars);
    obj.def("XtUPVars", &Derived::XtUPVars);

    obj.def("Xidxs", nb::overload_cast<>(&Derived::Xidxs, nb::const_));
    obj.def("Xidxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::Xidxs, nb::const_));

    obj.def("Xtidxs", nb::overload_cast<>(&Derived::Xtidxs, nb::const_));
    obj.def("Xtidxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::Xtidxs, nb::const_));

    obj.def("XtUidxs", nb::overload_cast<>(&Derived::XtUidxs, nb::const_));
    obj.def("XtUidxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::XtUidxs, nb::const_));

    obj.def("Uidxs", nb::overload_cast<>(&Derived::Uidxs, nb::const_));
    obj.def("Uidxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::Uidxs, nb::const_));

    obj.def("Pidxs", nb::overload_cast<>(&Derived::Pidxs, nb::const_));
    obj.def("Pidxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::Pidxs, nb::const_));

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

} // namespace Tycho::Bind

#endif // TYCHO_PYTHON_BINDINGS
