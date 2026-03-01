#pragma once
#include "KeplerPropagator.h"
#include "OptimalControl/ODE.h"
#include "OptimalControl/ODEPhase.h"
#include "VectorFunctions/Tycho_VectorFunctions.h"

namespace Tycho {

struct Kepler_Impl : ODESize<6, 0, 0> {
    static auto Definition(double mu) {
        auto args = Arguments<7>();
        auto X = args.head<3>();
        auto V = args.segment<3, 3>();
        auto rvec = X;
        auto acc = (-mu) * rvec.normalized_power<3>();
        auto ode = StackedOutputs{V, acc};
        return ode;
    }
};
struct KeplerPhase;

struct Kepler : ODE_Expression<Kepler, Kepler_Impl, double> {
    using Base = ODE_Expression<Kepler, Kepler_Impl, double>;
    using Base::Base;
    double mu = 1.0;
    Kepler(double mu) : Base(mu) { this->mu = mu; }

#ifdef TYCHO_PYTHON_BINDINGS
    static void Build(py::module &m, const char *name);
#endif // TYCHO_PYTHON_BINDINGS
};

struct KeplerPhase : ODEPhase<Kepler> {
    using Base = ODEPhase<Kepler>;
    using Base::Base;
    bool UseKeplerPropagator = true;

    Tycho::ConstraintInterface make_shooter() {
        if (UseKeplerPropagator) {
            auto kprop = KeplerPropagator(this->ode.mu);
            auto Args = Arguments<14>();
            auto X1 = Args.head<6>();
            auto X2 = Args.segment<6, 7>();
            auto t1 = Args.coeff<6>();
            auto t2 = Args.coeff<13>();

            auto Xk1 = StackedOutputs{X1, (t2 - t1) / 2.0};
            auto Xk2 = StackedOutputs{X2, (t1 - t2) / 2.0};

            auto shooter = kprop.eval(Xk1) - kprop.eval(Xk2);

            return Tycho::ConstraintInterface(shooter);
        } else {
            return Base::make_shooter();
        }
    }

#ifdef TYCHO_PYTHON_BINDINGS
    static void Build(py::module &m) {
        auto phase =
            py::class_<KeplerPhase, std::shared_ptr<KeplerPhase>, ODEPhaseBase>(m, "phase");
        BuildImpl(phase);
        phase.def_readwrite("integrator", &KeplerPhase::integrator);
        phase.def_readwrite("UseKeplerPropagator", &KeplerPhase::UseKeplerPropagator);
    }
#endif // TYCHO_PYTHON_BINDINGS
};

#ifdef TYCHO_PYTHON_BINDINGS
void BuildKeplerMod(FunctionRegistry &reg, py::module &m);
#endif // TYCHO_PYTHON_BINDINGS

} // namespace Tycho