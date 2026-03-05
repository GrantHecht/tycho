#include "OptimizationProblemBind.h"
#include "OptimizationProblem.h"

using namespace Tycho;

void TychoBind<OptimizationProblem>::Build(nb::module_ &m) {
    using VectorFunctionalX = OptimizationProblem::VectorFunctionalX;
    using ScalarFunctionalX = OptimizationProblem::ScalarFunctionalX;
    using VectorXi = OptimizationProblem::VectorXi;

    auto obj = nb::class_<OptimizationProblem, OptimizationProblemBase>(m, "OptimizationProblem");

    obj.def(nb::init<>());

    obj.def("setVars", &OptimizationProblem::setVars);
    obj.def("returnVars", &OptimizationProblem::returnVars);

    obj.def("addEqualCon", nb::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                               &OptimizationProblem::addEqualCon));

    obj.def("addEqualCon",
            nb::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addEqualCon));

    obj.def("addInequalCon", nb::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                                 &OptimizationProblem::addInequalCon));

    obj.def("addInequalCon",
            nb::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addInequalCon));

    obj.def("addObjective", nb::overload_cast<ScalarFunctionalX, const std::vector<VectorXi> &>(
                                &OptimizationProblem::addObjective));

    obj.def("addObjective",
            nb::overload_cast<ScalarFunctionalX, VectorXi>(&OptimizationProblem::addObjective));
}
