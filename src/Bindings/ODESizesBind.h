#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class definitions of ODESize binding helper methods.
// Included from ODESizes.h under TYCHO_PYTHON_BINDINGS.

namespace Tycho {

template <int XV, int UV, int PV>
template <class Obj, class Derived>
void ODESize<XV, UV, PV>::BuildODESizeMembers(Obj &obj) {
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

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
