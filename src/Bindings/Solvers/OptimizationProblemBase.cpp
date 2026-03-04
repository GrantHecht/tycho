#include "OptimizationProblemBase.h"
#include "OptimizationProblemBind.h"

using namespace Tycho;

void TychoBind<OptimizationProblemBase>::Build(nb::module_ &m) {
    using JetJobModes = OptimizationProblemBase::JetJobModes;
    auto obj = nb::class_<OptimizationProblemBase>(m, "OptimizationProblemBase");
    // JetJobMode is stored as int internally; expose it with the enum type so that
    // Python code can assign JetJobModes values (matching pybind11 behaviour).
    obj.def_prop_rw(
        "JetJobMode",
        [](const OptimizationProblemBase &self) { return (JetJobModes)self.JetJobMode; },
        [](OptimizationProblemBase &self, JetJobModes v) { self.JetJobMode = (int)v; });
    obj.def_rw("Threads", &OptimizationProblemBase::Threads);
    obj.def_ro("optimizer", &OptimizationProblemBase::optimizer);

    obj.def("setThreads", nb::overload_cast<int, int>(&OptimizationProblemBase::setThreads),
            nb::arg("FuncThreads"), nb::arg("KKTThreads"));

    obj.def("setThreads", nb::overload_cast<int>(&OptimizationProblemBase::setThreads));

    obj.def("setJetJobMode",
            nb::overload_cast<JetJobModes>(&OptimizationProblemBase::setJetJobMode));
    obj.def("setJetJobMode",
            nb::overload_cast<const std::string &>(&OptimizationProblemBase::setJetJobMode));

    obj.def("solve", &OptimizationProblemBase::solve, nb::call_guard<nb::gil_scoped_release>());
    obj.def("optimize", &OptimizationProblemBase::optimize,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("solve_optimize", &OptimizationProblemBase::solve_optimize,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("solve_optimize_solve", &OptimizationProblemBase::solve_optimize_solve,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("optimize_solve", &OptimizationProblemBase::optimize_solve,
            nb::call_guard<nb::gil_scoped_release>());

    /// <summary>
    /// Probably need to move these enums somewhere else
    /// </summary>
    /// <param name="m"></param>
    nb::enum_<JetJobModes>(m, "JetJobModes")
        .value("DoNothing", JetJobModes::DoNothing)
        .value("NotSet", JetJobModes::NotSet)
        .value("Solve", JetJobModes::Solve)
        .value("Optimize", JetJobModes::Optimize)
        .value("SolveOptimize", JetJobModes::SolveOptimize)
        .value("SolveOptimizeSolve", JetJobModes::SolveOptimizeSolve)
        .value("OptimizeSolve", JetJobModes::OptimizeSolve);
}