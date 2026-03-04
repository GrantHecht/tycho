#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "DenseFunctionBaseBind.h"
#include "FunctionRegistry.h"
#include "OptimalControl/ODEArguments.h"

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
