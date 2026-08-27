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
#include <hven/warmstart/warm_start_data.h>

#include <stdexcept>
#include <string>

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
    auto obj = nb::class_<BackendProblemBase>(m, "OptimizationProblemBase");
    obj.def_prop_rw(
        "num_partitions", [](const BackendProblemBase &self) { return self.num_partitions_; },
        [](BackendProblemBase &self, int n) { self.set_num_partitions(n); },
        R"doc(Number of NLP matrix partitions.

Assignment routes through :meth:`set_num_partitions` and raises
``ValueError`` for values < 1. The QP thread count is a separate setting,
set on whichever engine is passed to :meth:`solve`.)doc");

    obj.def("set_num_partitions", &BackendProblemBase::set_num_partitions,
            nb::arg("num_partitions"),
            R"doc(Set the number of NLP matrix partitions (must be >= 1).

The QP thread count is a separate setting, set on whichever engine is
passed to :meth:`solve`.)doc");

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
            if (presolve.is_none()) {
                // R-11: None aliases False, symmetric with polish=None.
                opts.presolve = false;
            } else if (nb::isinstance<nb::bool_>(presolve)) {
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
        nb::arg("engine"), nb::arg("mode") = "optimal", nb::arg("presolve").none() = false,
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
presolve : bool | InteriorPointSolver | SqpSolver | IpoptSolver | None, optional
    ``False`` (default) or ``None``: no presolve stage (``None`` is the
    same as ``False``). ``True``: run a Feasible presolve stage on
    ``engine`` itself. An engine instance: run the presolve stage on
    that engine instead (implies presolve).
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
    warm stamp, an engine already solving), if ``engine``/``presolve``/
    ``polish``/``mode``/``warm`` is not one of the types listed above,
    or whatever the dispatched engine itself raises for a malformed
    problem.
)doc");
}