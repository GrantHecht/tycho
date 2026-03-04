#include "JetBind.h"
#include "Solvers/Jet.h"

using namespace Tycho;

void TychoBind<Jet>::Build(nb::module_ &m) {
    auto obj = nb::class_<Jet>(m, "Jet");

    obj.def_static(
        "map",
        [](const std::vector<std::shared_ptr<OptimizationProblemBase>> &optprobs, int nt) {
            return Jet::map(optprobs, nt, true);
        },
        nb::call_guard<nb::gil_scoped_release>());

    obj.def_static(
        "map",
        [](std::function<std::shared_ptr<OptimizationProblemBase>(nb::detail::args_proxy)> genfun,
           const std::vector<nb::args> &args, int nt) { return Jet::map(genfun, args, nt, true); },
        nb::call_guard<nb::gil_scoped_release>());

    obj.def_static(
        "map",
        [](const std::vector<std::shared_ptr<OptimizationProblemBase>> &optprobs, int nt, bool v) {
            return Jet::map(optprobs, nt, v);
        },
        nb::call_guard<nb::gil_scoped_release>());

    obj.def_static(
        "map",
        [](std::function<std::shared_ptr<OptimizationProblemBase>(nb::detail::args_proxy)> genfun,
           const std::vector<nb::args> &args, int nt,
           bool v) { return Jet::map(genfun, args, nt, v); },
        nb::call_guard<nb::gil_scoped_release>());
}
