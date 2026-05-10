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

// Free-function binding helpers for StaticODE_Expression, ODEBase, and GenericODE.

#include "dense_function_base_bind.h"
#include "ode_sizes_bind.h"

namespace tycho::bind {

using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

template <class Derived, class ExprImpl, class... Ts>
void StaticODE_ExpressionBuild(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Derived>(m, name).def(nb::init<Ts...>());
    DenseBaseBuild<Derived>(obj);
    obj.def("phase", [](const Derived &od, TranscriptionModes Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
    });
    IntegratorBuildConstructors<Derived>(obj);
}

template <class BaseType, class Derived, int _XV, int _UV, int _PV>
void BuildODEModule(const char *name, nb::module_ &mod, FunctionRegistry &reg) {
    auto odemod = mod.def_submodule(name);
    reg.template build_register<Derived>(odemod, "ode");
    reg.template build_register<Integrator<Derived>>(odemod, "integrator");
    TychoBind<ODEPhase<Derived>>::build(odemod);
}

template <class BaseType, class Derived, int _XV, int _UV, int _PV>
void BuildODEModule(const char *name, FunctionRegistry &reg) {
    BuildODEModule<BaseType, Derived, _XV, _UV, _PV>(name, reg.mod, reg);
}

template <class BaseType, int _XV, int _UV, int _PV>
void BuildGenODEModule(const char *name, nb::module_ &mod, FunctionRegistry &reg) {
    using Derived = GenericODE<BaseType, _XV, _UV, _PV>;
    auto odemod = mod.def_submodule(name);

    auto obj = nb::class_<Derived>(odemod, "ode");
    obj.def(nb::init<BaseType, int, int, int>());
    obj.def(nb::init<BaseType, int, int>());
    obj.def(nb::init<BaseType, int>());
    obj.def(nb::init<BaseType>());
    TychoBind<ODEPhase<Derived>>::build(odemod);
    obj.def("phase", [](const Derived &od, TranscriptionModes Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
    });
    obj.def("phase", [](const Derived &od, TranscriptionModes Tmode,
                        const std::vector<Eigen::VectorXd> &Traj, int numdef) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef);
    });

    obj.def("phase", [](const Derived &od, std::string Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
    });

    obj.def("phase",
            [](const Derived &od, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj) {
                return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj);
            });

    obj.def("phase", [](const Derived &od, std::string Tmode,
                        const std::vector<Eigen::VectorXd> &Traj, int numdef) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef);
    });
    obj.def("phase", [](const Derived &od, std::string Tmode,
                        const std::vector<Eigen::VectorXd> &Traj, int numdef, bool LerpIG) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef, LerpIG);
    });

    nb::implicitly_convertible<Derived, GenericFunction<-1, -1>>();
    reg.vfuncx.def("__init__", [](GenericFunction<-1, -1> *self, const Derived &ode) {
        new (self) GenericFunction<-1, -1>(ode.func_);
    });

    IntegratorBuildConstructors<Derived>(obj);

    ODESizeBuild<_XV, _UV, _PV, Derived>(obj);

    obj.def("vf", [](const Derived &od) { return od.func_; });

    obj.def("shooting_defect", [](const Derived &ode, const Integrator<Derived> &integ) {
        auto shooter = CentralShootingDefect<Derived, Integrator<Derived>>(ode, integ);
        return GenericFunction<-1, -1>(shooter);
    });
}

template <class BaseType, int _XV, int _UV, int _PV>
void BuildGenODEIntegrator(const char *name, nb::module_ &mod, FunctionRegistry &reg) {
    using Derived = GenericODE<BaseType, _XV, _UV, _PV>;
    auto odemod = nb::borrow<nb::module_>(mod.attr(name));
    reg.template build_register<Integrator<Derived>>(odemod, "integrator");
}

} // namespace tycho::bind

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

template <int OR> struct TychoBind<InterpFunction<OR>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<InterpFunction<OR>>(m, name);
        if constexpr (OR == -1) {
            obj.def(nb::init<std::shared_ptr<LGLInterpTable>, Eigen::VectorXi>());
        } else {
            obj.def(nb::init<std::shared_ptr<LGLInterpTable>>());
        }
        bind::DenseBaseBuild<InterpFunction<OR>>(obj);
        obj.def("__call__", [](const InterpFunction<OR> &self, const GenericFunction<-1, 1> &t) {
            return GenericFunction<-1, -1>(self.eval(t));
        });
        obj.def("__call__", [](const InterpFunction<OR> &self, const Segment<-1, 1, -1> &t) {
            return GenericFunction<-1, -1>(self.eval(t));
        });
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
