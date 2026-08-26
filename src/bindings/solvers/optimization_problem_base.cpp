// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "optimization_problem_bind.h"
#include "tycho/detail/hven_namespaces.h"
#include <hven/drivers/optimization_problem_base.h>
#include <hven/warmstart/warm_start_data.h>

#include <stdexcept>
#include <string>

#include <nanobind/stl/map.h>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

namespace {

// engine=: an InteriorPointSolver, SqpSolver, or IpoptSolver instance,
// resolved to the non-owning EngineRef the pipeline dispatches on.
EngineRef engine_ref_from_pyobject(nb::handle h, const char *argname) {
    InteriorPointSolver *ipm = nullptr;
    if (nb::try_cast<InteriorPointSolver *>(h, ipm, false)) {
        return EngineRef{ipm};
    }
    SqpSolver *sqp = nullptr;
    if (nb::try_cast<SqpSolver *>(h, sqp, false)) {
        return EngineRef{sqp};
    }
    IpoptSolver *ipopt = nullptr;
    if (nb::try_cast<IpoptSolver *>(h, ipopt, false)) {
        return EngineRef{ipopt};
    }
    throw std::invalid_argument(
        fmt::format("{}: expected an InteriorPointSolver, SqpSolver, or IpoptSolver instance, "
                    "got {}",
                    argname, nb::cast<std::string>(nb::str(h.type()))));
}

// mode=: a Mode enum value, or a case-insensitive "optimal"/"feasible" string
// (mode_from_string itself names both legal values on an unrecognized one).
Mode mode_from_pyobject(nb::handle h) {
    Mode mode;
    if (nb::try_cast<Mode>(h, mode, false)) {
        return mode;
    }
    if (nb::isinstance<nb::str>(h)) {
        return mode_from_string(nb::cast<std::string>(h));
    }
    throw std::invalid_argument(
        fmt::format("mode: expected a Mode enum or 'optimal'/'feasible' string, got {}",
                    nb::cast<std::string>(nb::str(h.type()))));
}

} // namespace

void TychoBind<BackendProblemBase>::build(nb::module_ &m) {
    using JetJobModes = BackendProblemBase::JetJobModes;
    auto obj = nb::class_<BackendProblemBase>(m, "OptimizationProblemBase");
    obj.def_rw("jet_job_mode", &BackendProblemBase::jet_job_mode_);
    obj.def_prop_rw(
        "num_partitions", [](const BackendProblemBase &self) { return self.num_partitions_; },
        [](BackendProblemBase &self, int n) { self.set_num_partitions(n); },
        R"doc(Number of NLP matrix partitions.

Assignment routes through :meth:`set_num_partitions` and raises
``ValueError`` for values < 1. The QP thread count is a separate setting:
assign ``optimizer.qp_threads``.)doc");
    obj.def_ro("optimizer", &BackendProblemBase::optimizer_);

    obj.def_rw("nlp_solver", &BackendProblemBase::nlp_solver_,
               R"doc(NLP solver backend for the solve/optimize entry points.

NLPSolvers.interior_point (default) is the built-in solver, byte-identical to
previous behavior. NLPSolvers.ipopt runs the identical transcribed NLP
through a linked Ipopt installation; requires a build configured with
ENABLE_IPOPT (raises RuntimeError otherwise). The ipopt backend always
performs a single NLP solve of the full objective-bearing problem: the
feasibility-then-optimize staging modes have no Ipopt analog. In
particular ``solve()`` -- which under the built-in solver runs the
feasibility-only stage -- minimizes the objective like ``optimize()``
when this backend is selected; there is no feasibility-only analog.

Jet batch runs reject this backend: Ipopt is not reliably re-entrant, so
a Jet job whose problem selects it raises ValueError before that job's
solve begins. Run the ipopt backend one solve at a time.

The built-in solver's own diagnostics (``optimizer.last_obj_val``,
``optimizer.last_iter_num``, and every other result()-backed property on
``optimizer``) reflect only the most recent InteriorPointSolver run and are left
untouched by an ipopt-backend run -- use ``last_ipopt_result`` as the
source of truth for diagnostics of the most recent ipopt-backend
solve.)doc");
    obj.def_rw("ipopt_options", &BackendProblemBase::ipopt_options_,
               R"doc(String key/value options forwarded verbatim to Ipopt (e.g.
{"linear_solver": "pardisomkl"}). Applied after the matched-tolerance
baseline, so entries here win. Ignored by the interior-point backend.

Reading this attribute returns a *copy* of the stored map, so in-place
mutation (``prob.ipopt_options["linear_solver"] = "ma57"``) silently has
no effect. Assign a whole dict instead, or read-modify-write:
``opts = prob.ipopt_options; opts["linear_solver"] = "ma57";
prob.ipopt_options = opts``.)doc");
    obj.def_ro("last_ipopt_result", &BackendProblemBase::last_ipopt_result_,
               R"doc(Diagnostics of the most recent ipopt-backend run on this problem
(sentinel values with ran == False before any such run).)doc");

    obj.def("set_num_partitions", &BackendProblemBase::set_num_partitions,
            nb::arg("num_partitions"),
            R"doc(Set the number of NLP matrix partitions (must be >= 1).

The QP thread count is a separate setting: assign ``optimizer.qp_threads``.)doc");

    obj.def("set_jet_job_mode",
            nb::overload_cast<JetJobModes>(&BackendProblemBase::set_jet_job_mode));
    obj.def("set_jet_job_mode",
            nb::overload_cast<const std::string &>(&BackendProblemBase::set_jet_job_mode));

    // BackendProblemBase::solve now has a second overload (the engine-
    // driven staged solve, EngineRef + SolveOptions -- not yet exposed to
    // Python), so the member-pointer expression needs disambiguating the
    // same way set_jet_job_mode's two overloads are just below.
    obj.def("solve", nb::overload_cast<>(&BackendProblemBase::solve),
            nb::call_guard<nb::gil_scoped_release>());

    // The engine-driven staged solve() (M5 solve-API). Argument conversion
    // (engine/mode/presolve/polish/warm) runs with the GIL held; the run
    // itself releases it. presolve_engine_storage/polish_engine_storage are
    // stack locals whose addresses SolveOptions borrows for the duration of
    // this call only -- never returned, never stored past the call.
    obj.def(
        "solve",
        [](BackendProblemBase &self, nb::object engine, nb::object mode, nb::object presolve,
           nb::object polish, nb::object warm) -> SolveResult {
            EngineRef main_engine = engine_ref_from_pyobject(engine, "engine");

            SolveOptions opts;
            opts.mode = mode_from_pyobject(mode);

            EngineRef presolve_engine_storage{};
            if (nb::isinstance<nb::bool_>(presolve)) {
                opts.presolve = nb::cast<bool>(presolve);
            } else {
                presolve_engine_storage = engine_ref_from_pyobject(presolve, "presolve");
                opts.presolve = true;
                opts.presolve_engine = &presolve_engine_storage;
            }

            EngineRef polish_engine_storage{};
            if (!polish.is_none()) {
                polish_engine_storage = engine_ref_from_pyobject(polish, "polish");
                opts.polish = &polish_engine_storage;
            }

            const hven::solvers::WarmStartData *warm_ptr = nullptr;
            if (!warm.is_none()) {
                SolveResult *sr = nullptr;
                hven::solvers::WarmStartData *wsd = nullptr;
                if (nb::try_cast<SolveResult *>(warm, sr, false)) {
                    warm_ptr = &sr->warm_;
                } else if (nb::try_cast<hven::solvers::WarmStartData *>(warm, wsd, false)) {
                    warm_ptr = wsd;
                } else {
                    throw std::invalid_argument(
                        fmt::format("warm: expected a SolveResult, WarmStartData, or None, got {}",
                                    nb::cast<std::string>(nb::str(warm.type()))));
                }
            }
            opts.warm = warm_ptr;

            SolveResult result;
            {
                nb::gil_scoped_release release;
                result = self.solve(main_engine, opts);
            }
            return result;
        },
        nb::arg("engine"), nb::arg("mode") = "optimal", nb::arg("presolve") = false,
        nb::arg("polish") = nb::none(), nb::arg("warm") = nb::none(),
        R"doc(Run the engine-driven staged solve: an optional Feasible presolve
stage, the main stage (``mode``), and an optional Optimal polish stage,
in that order.

Parameters
----------
engine : InteriorPointSolver | SqpSolver | IpoptSolver
    The main-stage engine.
mode : Mode | str, optional
    ``Mode.Optimal``/``"optimal"`` (default) or ``Mode.Feasible``/
    ``"feasible"``.
presolve : bool | InteriorPointSolver | SqpSolver | IpoptSolver, optional
    ``False`` (default): no presolve stage. ``True``: run a Feasible
    presolve stage on ``engine`` itself. An engine instance: run the
    presolve stage on that engine instead (implies presolve).
polish : InteriorPointSolver | SqpSolver | IpoptSolver | None, optional
    When given, run an Optimal polish stage on this engine after the
    main stage.
warm : SolveResult | WarmStartData | None, optional
    Declared-space warm-start currency seeding the first stage that
    runs. Its declaration-identity stamp must match the current
    transcription's, or the call raises ValueError naming both.

Returns
-------
SolveResult
    The deciding convergence flag, every stage that ran, every OCP
    phase's slice, and the warm-start currency from the final deciding
    stage.

Raises
------
ValueError
    Per the refusal matrix (mode/presolve/polish combinations, a stale
    warm stamp, an engine already solving), or whatever the dispatched
    engine itself raises for a malformed problem.
TypeError
    If ``engine``/``presolve``/``polish``/``warm`` is not one of the
    types listed above.
)doc");

    obj.def("optimize", &BackendProblemBase::optimize, nb::call_guard<nb::gil_scoped_release>());
    obj.def("solve_optimize", &BackendProblemBase::solve_optimize,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("solve_optimize_solve", &BackendProblemBase::solve_optimize_solve,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("optimize_solve", &BackendProblemBase::optimize_solve,
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