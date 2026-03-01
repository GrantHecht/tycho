#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class definitions of ODE_Expression::Build(), ODEBase::BuildODEModule(),
// and GenericODE::BuildGenODEModule() binding methods.
// Included from ODE.h under TYCHO_PYTHON_BINDINGS.

namespace Tycho {

template <class Derived, class ExprImpl, class... Ts>
void ODE_Expression<Derived, ExprImpl, Ts...>::Build(nb::module_ &m, const char *name) {
    using Base = typename ODE_Expression<Derived, ExprImpl, Ts...>::Base;
    auto obj = nb::class_<Derived>(m, name).def(nb::init<Ts...>());
    Base::DenseBaseBuild(obj);
    obj.def("phase", [](const Derived &od, TranscriptionModes Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
    });
    Integrator<Derived>::BuildConstructors(obj);
}

template <class BaseType, class Derived, int _XV, int _UV, int _PV>
void ODEBase<BaseType, Derived, _XV, _UV, _PV>::BuildODEModule(const char *name, nb::module_ &mod,
                                                                FunctionRegistry &reg) {
    auto odemod = mod.def_submodule(name);
    reg.template Build_Register<Derived>(odemod, "ode");
    reg.template Build_Register<Integrator<Derived>>(odemod, "integrator");
    ODEPhase<Derived>::Build(odemod);
}

template <class BaseType, class Derived, int _XV, int _UV, int _PV>
void ODEBase<BaseType, Derived, _XV, _UV, _PV>::BuildODEModule(const char *name,
                                                                FunctionRegistry &reg) {
    BuildODEModule(name, reg.mod, reg);
}

template <class BaseType, int _XV, int _UV, int _PV>
void GenericODE<BaseType, _XV, _UV, _PV>::BuildGenODEModule(const char *name, nb::module_ &mod,
                                                              FunctionRegistry &reg) {
    using Derived = GenericODE<BaseType, _XV, _UV, _PV>;
    auto odemod = mod.def_submodule(name);

    auto obj = nb::class_<Derived>(odemod, "ode");
    obj.def(nb::init<BaseType, int, int, int>());
    obj.def(nb::init<BaseType, int, int>());
    obj.def(nb::init<BaseType, int>());
    obj.def(nb::init<BaseType>());
    ODEPhase<Derived>::Build(odemod);
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
        new (self) GenericFunction<-1, -1>(ode.func);
    });

    reg.template Build_Register<Integrator<Derived>>(odemod, "integrator");

    Integrator<Derived>::BuildConstructors(obj);

    ODESize<_XV, _UV, _PV>::template BuildODESizeMembers<decltype(obj), Derived>(obj);

    obj.def("vf", [](const Derived &od) { return od.func; });

    obj.def("shooting_defect", [](const Derived &ode, const Integrator<Derived> &integ) {
        auto shooter = CentralShootingDefect<Derived, Integrator<Derived>>(ode, integ);
        return GenericFunction<-1, -1>(shooter);
    });
}

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
