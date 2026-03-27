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

// TychoBind specializations for ODEPhase<DODE>, StateFunction<T>, LinkFunction<T>,
// FDDerivArbitrary<T>, ODEPhaseBase, LGLInterpTable, and OptimalControlProblem.

#include "tycho/detail/vf/derivatives/fd_deriv_arbitrary.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
#include "tycho/detail/optimal_control/core/link_function.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
#include "tycho/detail/optimal_control/phase/optimal_control_problem.h"
#include "tycho/detail/optimal_control/core/state_function.h"

namespace Tycho {

namespace Bind {

template <class DODE, class PyClass> void ODEPhaseBuildImpl(PyClass &phase) {
    phase.def(nb::init<DODE, TranscriptionModes>());
    phase.def(nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(
        nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int, bool>());

    phase.def(nb::init<DODE, std::string>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &>());

    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int, bool>());
}

} // namespace Bind

template <class DODE> struct TychoBind<ODEPhase<DODE>> {
    static void Build(nb::module_ &m) {
        auto phase = nb::class_<ODEPhase<DODE>, ODEPhaseBase>(m, "phase");

        Bind::ODEPhaseBuildImpl<DODE>(phase);
        phase.def_rw("integrator", &ODEPhase<DODE>::integrator);
        phase.def_rw("EnableHessianSparsity", &ODEPhase<DODE>::EnableHessianSparsity);
        phase.def_rw("OldShootingDefect", &ODEPhase<DODE>::OldShootingDefect);

        phase.def("get_input_scale", &ODEPhase<DODE>::get_input_scale);
        phase.def("get_defect", &ODEPhase<DODE>::get_defect);
    }
};

template <class FuncType> struct TychoBind<StateFunction<FuncType>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<StateFunction<FuncType>>(m, name);
        obj.def(nb::init<FuncType, PhaseRegionFlags, Eigen::VectorXi, Eigen::VectorXi,
                         Eigen::VectorXi>());
        obj.def(nb::init<FuncType, PhaseRegionFlags, Eigen::VectorXi, PhaseRegionFlags,
                         Eigen::VectorXi>());
        obj.def(nb::init<FuncType, PhaseRegionFlags, Eigen::VectorXi>());
    }
};

template <class FuncType> struct TychoBind<LinkFunction<FuncType>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<LinkFunction<FuncType>>(m, name);
        obj.def(nb::init<FuncType, LinkFlags, std::vector<Eigen::VectorXi>, Eigen::VectorXi>());
        obj.def(
            nb::init<FuncType, Eigen::Matrix<PhaseRegionFlags, -1, 1>, std::vector<Eigen::VectorXi>,
                     std::vector<Eigen::VectorXi>, std::vector<Eigen::VectorXi>,
                     std::vector<Eigen::VectorXi>, std::vector<Eigen::VectorXi>>());
        obj.def(nb::init<FuncType, LinkFlags, std::vector<Eigen::VectorXi>,
                         std::vector<Eigen::VectorXi>, std::vector<Eigen::VectorXi>,
                         std::vector<Eigen::VectorXi>, std::vector<Eigen::VectorXi>>());
    }
};

template <class DType> struct TychoBind<FDDerivArbitrary<DType>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<FDDerivArbitrary<DType>>(m, name);
        obj.def(nb::init<int, const std::vector<DType> &>());
        obj.def("all_derivs", &FDDerivArbitrary<DType>::template allderiv_python<DType>);
        obj.def("deriv", &FDDerivArbitrary<DType>::template ithderiv_python<DType>);
    }
};

// Forward declarations for out-of-line TychoBind implementations
template <> struct TychoBind<ODEPhaseBase> {
    static void Build(nb::module_ &m);
};

template <> struct TychoBind<LGLInterpTable> {
    static void Build(nb::module_ &m);
};

template <> struct TychoBind<OptimalControlProblem> {
    static void Build(nb::module_ &m);
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
