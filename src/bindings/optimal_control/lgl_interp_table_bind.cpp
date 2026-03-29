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

#include "ode_phase_bind.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;
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
    obj.def("load_even_data", &LGLInterpTable::load_even_data);
    obj.def("get_table_ptr", &LGLInterpTable::getTablePtr);
    obj.def("load_uneven_data", &LGLInterpTable::load_uneven_data);
    obj.def("interpolate", &LGLInterpTable::Interpolate<double>);

    obj.def("new_error_integral", &LGLInterpTable::NewErrorIntegral);

    obj.def("__call__", nb::overload_cast<double>(&LGLInterpTable::Interpolate<double>, nb::const_),
            nb::is_operator());

    obj.def("interpolate_deriv", &LGLInterpTable::InterpolateDeriv<double>);
    obj.def("make_periodic", &LGLInterpTable::make_periodic);

    obj.def("interp_range", &LGLInterpTable::InterpRange);
    obj.def("interp_whole_range", &LGLInterpTable::InterpWholeRange);
    obj.def("error_integral", &LGLInterpTable::ErrorIntegral);

    obj.def_ro("t0", &LGLInterpTable::T0);
    obj.def_ro("tf", &LGLInterpTable::TF);

    obj.def("interp_non_dim", nb::overload_cast<int, double, double>(&LGLInterpTable::NDequidist));
}
