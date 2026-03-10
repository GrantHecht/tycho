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

#include "Astro/Tycho_Astro.h"
#include "OptimalControl/Tycho_OptimalControl.h"

// TychoBind<KeplerPropagator> — defined here since KeplerPropagator.h's inline Build was removed.
namespace Tycho {
template <> struct TychoBind<KeplerPropagator> {
    using BuildTag = void;

    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<KeplerPropagator>(m, name);
        obj.def(nb::init<double>());
        Bind::DenseBaseBuild<KeplerPropagator>(obj);
    }
};
} // namespace Tycho

// TychoBind<KeplerPhase> — KeplerPhase is a subclass of ODEPhase<Kepler> with extra members.
namespace Tycho {
template <> struct TychoBind<KeplerPhase> {
    static void Build(nb::module_ &m) {
        auto phase = nb::class_<KeplerPhase, ODEPhaseBase>(m, "phase");
        Bind::ODEPhaseBuildImpl<Kepler>(phase);
        phase.def_rw("integrator", &KeplerPhase::integrator);
        phase.def_rw("UseKeplerPropagator", &KeplerPhase::UseKeplerPropagator);
    }
};
} // namespace Tycho

// TychoBind<Kepler> — the Kepler ODE type.
namespace Tycho {
template <> struct TychoBind<Kepler> {
    using BuildTag = void;

    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Kepler>(m, name).def(nb::init<double>());
        Bind::DenseBaseBuild<Kepler>(obj);
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

        Bind::IntegratorBuildConstructors<Kepler>(obj);
    }
};
void BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m);
} // namespace Tycho

void Tycho::BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m) {
    auto odemod = m.def_submodule("Kepler");
    reg.template Build_Register<Kepler>(odemod, "ode");
    reg.template Build_Register<Integrator<Kepler>>(odemod, "integrator");
    reg.Build_Register<KeplerPropagator>(odemod, "KeplerPropagator");
    TychoBind<KeplerPhase>::Build(odemod);
}
