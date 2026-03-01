#include "OptimizationProblem.h"

void Tycho::OptimizationProblem::Build(py::module &m) {

    auto obj = py::class_<OptimizationProblem, std::shared_ptr<OptimizationProblem>,
                          OptimizationProblemBase>(m, "OptimizationProblem");

    obj.def(py::init<>());

    obj.def("setVars", &OptimizationProblem::setVars);
    obj.def("returnVars", &OptimizationProblem::returnVars);

    obj.def("addEqualCon", py::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                               &OptimizationProblem::addEqualCon));

    obj.def("addEqualCon",
            py::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addEqualCon));

    obj.def("addInequalCon", py::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                                 &OptimizationProblem::addInequalCon));

    obj.def("addInequalCon",
            py::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addInequalCon));

    obj.def("addObjective", py::overload_cast<ScalarFunctionalX, const std::vector<VectorXi> &>(
                                &OptimizationProblem::addObjective));

    obj.def("addObjective",
            py::overload_cast<ScalarFunctionalX, VectorXi>(&OptimizationProblem::addObjective));
}
