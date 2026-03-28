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

#include "astro/tycho_astro.h"
#include "optimal_control/tycho_optimal_control.h"

// TychoBind<KeplerPropagator> — defined here since KeplerPropagator.h's inline Build was removed.
namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
template <> struct TychoBind<KeplerPropagator> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<KeplerPropagator>(m, name);
        obj.def(nb::init<double>());
        bind::DenseBaseBuild<KeplerPropagator>(obj);
    }
};
} // namespace tycho

// TychoBind<KeplerPhase> — KeplerPhase is a subclass of ODEPhase<Kepler> with extra members.
namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
template <> struct TychoBind<KeplerPhase> {
    static void Build(nb::module_ &m) {
        auto phase = nb::class_<KeplerPhase, ODEPhaseBase>(m, "phase");
        bind::ODEPhaseBuildImpl<Kepler>(phase);
        phase.def_rw("integrator", &KeplerPhase::integrator);
        phase.def_rw("use_kepler_propagator", &KeplerPhase::UseKeplerPropagator);
    }
};
} // namespace tycho

// TychoBind<Kepler> — the Kepler ODE type.
namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
template <> struct TychoBind<Kepler> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Kepler>(m, name).def(nb::init<double>());
        bind::DenseBaseBuild<Kepler>(obj);
        obj.def("phase", [](const Kepler &od, TranscriptionModes Tmode) {
            return std::make_shared<KeplerPhase>(od, Tmode);
        });
        obj.def("phase", [](const Kepler &od, TranscriptionModes Tmode,
                            const std::vector<Eigen::VectorXd> &Traj, int numdef) {
            return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef);
        });

        obj.def("phase", [](const Kepler &od, std::string Tmode) {
            return std::make_shared<KeplerPhase>(od, Tmode);
        });
        obj.def("phase",
                [](const Kepler &od, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj,
                   int numdef) { return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef); });

        bind::IntegratorBuildConstructors<Kepler>(obj);
    }
};
void BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m) {
    auto odemod = m.def_submodule("Kepler");
    reg.template Build_Register<Kepler>(odemod, "ode");
    reg.template Build_Register<Integrator<Kepler>>(odemod, "integrator");
    reg.Build_Register<KeplerPropagator>(odemod, "KeplerPropagator");
    TychoBind<KeplerPhase>::Build(odemod);
}
