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

#include "ODEPhaseBind.h"

using namespace Tycho;
using VectorFunctionalX = GenericFunction<-1, -1>;

void TychoBind<LGLInterpTable>::Build(nb::module_ &m) {
    auto obj = nb::class_<LGLInterpTable>(m, "LGLInterpTable");
    obj.def(nb::init<VectorFunctionalX, int, int, TranscriptionModes,
                     const std::vector<Eigen::VectorXd> &, int>());

    obj.def(nb::init<VectorFunctionalX, int, int, std::string, const std::vector<Eigen::VectorXd> &,
                     int>());

    obj.def(nb::init<VectorFunctionalX, int, int, int, std::string,
                     const std::vector<Eigen::VectorXd> &, int>());

    obj.def(nb::init<VectorFunctionalX, int, int, const std::vector<Eigen::VectorXd> &>());
    obj.def(nb::init<VectorFunctionalX, int, int, int, const std::vector<Eigen::VectorXd> &>());

    obj.def(nb::init<int, const std::vector<Eigen::VectorXd> &, int>());
    obj.def(nb::init<const std::vector<Eigen::VectorXd> &>());

    obj.def(nb::init<VectorFunctionalX, int, int, TranscriptionModes>());

    obj.def(nb::init<int, int, TranscriptionModes>());
    obj.def("loadEvenData", &LGLInterpTable::loadEvenData);
    obj.def("getTablePtr", &LGLInterpTable::getTablePtr);
    obj.def("loadUnevenData", &LGLInterpTable::loadUnevenData);
    obj.def("Interpolate", &LGLInterpTable::Interpolate<double>);

    obj.def("NewErrorIntegral", &LGLInterpTable::NewErrorIntegral);

    obj.def("__call__", nb::overload_cast<double>(&LGLInterpTable::Interpolate<double>, nb::const_),
            nb::is_operator());

    obj.def("InterpolateDeriv", &LGLInterpTable::InterpolateDeriv<double>);
    obj.def("makePeriodic", &LGLInterpTable::makePeriodic);

    obj.def("InterpRange", &LGLInterpTable::InterpRange);
    obj.def("InterpWholeRange", &LGLInterpTable::InterpWholeRange);
    obj.def("ErrorIntegral", &LGLInterpTable::ErrorIntegral);

    obj.def_ro("T0", &LGLInterpTable::T0);
    obj.def_ro("TF", &LGLInterpTable::TF);

    obj.def("InterpNonDim", nb::overload_cast<int, double, double>(&LGLInterpTable::NDequidist));
}
