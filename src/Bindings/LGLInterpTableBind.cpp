#include "LGLInterpTable.h"

void Tycho::LGLInterpTable::Build(nb::module_ &m) {
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
