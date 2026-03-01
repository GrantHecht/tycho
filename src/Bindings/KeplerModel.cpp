#include "Astro/KeplerModel.h"

void Tycho::KeplerPhase::Build(nb::module_ &m) {
    auto phase = nb::class_<KeplerPhase, ODEPhaseBase>(m, "phase");
    BuildImpl(phase);
    phase.def_rw("integrator", &KeplerPhase::integrator);
    phase.def_rw("UseKeplerPropagator", &KeplerPhase::UseKeplerPropagator);
}

void Tycho::BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m) {
    auto odemod = m.def_submodule("Kepler");
    reg.template Build_Register<Kepler>(odemod, "ode");
    reg.template Build_Register<Integrator<Kepler>>(odemod, "integrator");
    reg.Build_Register<KeplerPropagator>(odemod, "KeplerPropagator");
    KeplerPhase::Build(odemod);
}

void Tycho::Kepler::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Kepler>(m, name).def(nb::init<double>());
    Base::DenseBaseBuild(obj);
    obj.def("phase", [](const Kepler &od, TranscriptionModes Tmode) {
        return std::make_shared<KeplerPhase>(od, Tmode);
    });
    obj.def("phase",
            [](const Kepler &od, TranscriptionModes Tmode, const std::vector<Eigen::VectorXd> &Traj,
               int numdef) { return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef); });

    obj.def("phase", [](const Kepler &od, std::string Tmode) {
        return std::make_shared<KeplerPhase>(od, Tmode);
    });
    obj.def("phase",
            [](const Kepler &od, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj,
               int numdef) { return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef); });

    Integrator<Kepler>::BuildConstructors(obj);
}
