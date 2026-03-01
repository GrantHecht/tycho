#include "LGLInterpTable.h"

void Tycho::LGLInterpTable::Build(py::module &m) {
    auto obj = py::class_<LGLInterpTable, std::shared_ptr<LGLInterpTable>>(m, "LGLInterpTable");
    obj.def(py::init<VectorFunctionalX, int, int, TranscriptionModes,
                     const std::vector<Eigen::VectorXd> &, int>());

    obj.def(py::init<VectorFunctionalX, int, int, std::string, const std::vector<Eigen::VectorXd> &,
                     int>());

    obj.def(py::init<VectorFunctionalX, int, int, int, std::string,
                     const std::vector<Eigen::VectorXd> &, int>());

    obj.def(py::init<VectorFunctionalX, int, int, const std::vector<Eigen::VectorXd> &>());
    obj.def(py::init<VectorFunctionalX, int, int, int, const std::vector<Eigen::VectorXd> &>());

    obj.def(py::init<int, const std::vector<Eigen::VectorXd> &, int>());
    obj.def(py::init<const std::vector<Eigen::VectorXd> &>());

    obj.def(py::init<VectorFunctionalX, int, int, TranscriptionModes>());

    obj.def(py::init<int, int, TranscriptionModes>());
    obj.def("loadEvenData", &LGLInterpTable::loadEvenData);
    obj.def("getTablePtr", &LGLInterpTable::getTablePtr);
    obj.def("loadUnevenData", &LGLInterpTable::loadUnevenData);
    obj.def("Interpolate", &LGLInterpTable::Interpolate<double>);

    obj.def("NewErrorIntegral", &LGLInterpTable::NewErrorIntegral);

    obj.def("__call__", py::overload_cast<double>(&LGLInterpTable::Interpolate<double>, py::const_),
            py::is_operator());

    obj.def("InterpolateDeriv", &LGLInterpTable::InterpolateDeriv<double>);
    obj.def("makePeriodic", &LGLInterpTable::makePeriodic);

    obj.def("InterpRange", &LGLInterpTable::InterpRange);
    obj.def("InterpWholeRange", &LGLInterpTable::InterpWholeRange);
    obj.def("ErrorIntegral", &LGLInterpTable::ErrorIntegral);

    obj.def_readonly("T0", &LGLInterpTable::T0);
    obj.def_readonly("TF", &LGLInterpTable::TF);

    obj.def("InterpNonDim", py::overload_cast<int, double, double>(&LGLInterpTable::NDequidist));
}
