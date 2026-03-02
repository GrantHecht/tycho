#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Free-function binding helper for ODESize member definitions.
// Used by ODEBind.h and any other code that needs to expose ODE size members.

namespace Tycho::Bind {

template <int XV, int UV, int PV, class Derived, class Obj>
void ODESizeBuild(Obj &obj) {
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
    obj.def("XtUidxs",
            nb::overload_cast<const Eigen::VectorXi &>(&Derived::XtUidxs, nb::const_));

    obj.def("Uidxs", nb::overload_cast<>(&Derived::Uidxs, nb::const_));
    obj.def("Uidxs", nb::overload_cast<const Eigen::VectorXi &>(&Derived::Uidxs, nb::const_));

    obj.def("add_idx",
            nb::overload_cast<const std::string &, const Eigen::VectorXi &>(&Derived::add_idx));
    obj.def("get_idxs", &Derived::get_idxs);
    obj.def("set_idxs", &Derived::set_idxs);
    obj.def("idx", &Derived::idx);
}

} // namespace Tycho::Bind

#endif // TYCHO_PYTHON_BINDINGS
