// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Selection of the NLP solver backend a problem is dispatched to.
//
// The solver library tycho builds on carries no interface to a foreign solver:
// its problem base runs the built-in interior-point solver and nothing else.
// The selector, the options forwarded to a foreign solver, the diagnostics of
// the last foreign run, and the dispatch that routes a solve to it therefore
// live on tycho's side of the boundary — here.
//
// This header deliberately contains no Ipopt includes and no Ipopt types, so it
// is safe to include in every build regardless of how the project was
// configured. The Ipopt dependency is confined to the translation unit that
// implements the backend.

#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#include <hven/detail/drivers/interior_point_solver_fwd.h>
#include <hven/drivers/optimization_problem_base.h>

#include "tycho/detail/hven_namespaces.h"

namespace tycho::solvers {

/// The transcribed problem published as a claim-stream-bearing provider. Named
/// here and defined in solvers_vf/transcribed_aggregate.h, so that a problem
/// can hold one without every consumer of this header seeing the layout it
/// publishes.
class TranscribedAggregate;

/// NLP solver backend selector (BackendProblemBase::nlp_solver_).
/// interior_point is the built-in interior-point solver (default). ipopt hands the
/// identical transcribed NLP to a linked Ipopt installation for reference
/// comparison; dispatching it requires a build configured with Ipopt support.
enum class NLPSolvers { interior_point = 0, ipopt = 1 };

/// Outcome of one Ipopt run on the transcribed NLP. Sentinel values (-1 /
/// empty / ran_ == false) mean no Ipopt solve has run on this problem object.
struct IpoptRunInfo {
    bool ran_ = false;
    std::string status_ = "";     ///< Raw Ipopt ApplicationReturnStatus name.
    std::string normalized_ = ""; ///< converged/acceptable/infeasible/failed/diverged.
    ConvergenceFlags converge_flag_ = ConvergenceFlags::NOTCONVERGED;
    int iterations_ = -1;
    double objective_ = 0.0;
    double constraint_violation_ = -1.0;
    double wall_time_s_ = -1.0;
};

/// Every tycho optimization problem — OptimizationProblem, ODEPhaseBase and
/// OptimalControlProblemBase alike — derives from this rather than from the
/// solver library's OptimizationProblemBase directly, so that backend
/// selection is available wherever a problem is. It is the class the Python
/// binding exposes under the name "OptimizationProblemBase"; the distinct C++
/// name keeps the two apart at every unqualified use site.
struct BackendProblemBase : hven::solvers::OptimizationProblemBase {
    using Base = hven::solvers::OptimizationProblemBase;

    /// NLP solver backend for the solve/optimize entry points. interior_point (default)
    /// is the built-in path, byte-identical to previous behavior. ipopt runs
    /// the identical transcribed NLP through a linked Ipopt installation
    /// (requires a build with Ipopt support; throws std::runtime_error
    /// otherwise). Not usable in a Jet batch run — see the guard in jet_run().
    NLPSolvers nlp_solver_ = NLPSolvers::interior_point;

    /// String key/value options forwarded verbatim to Ipopt (e.g.
    /// {"linear_solver", "pardisomkl"}). Applied after the matched-tolerance
    /// baseline, so user entries win. Ignored by the interior-point backend.
    std::map<std::string, std::string> ipopt_options_;

    /// Diagnostics of the most recent ipopt-backend run on this problem.
    IpoptRunInfo last_ipopt_result_;

    /// The transcribed problem published as a claim-stream-bearing provider:
    /// the declaration this problem was laid from, and the per-slot coordinates
    /// a consumer needs in order to lay a destination of its own. Replaced by
    /// every transcription, and null until the first one runs.
    std::shared_ptr<TranscribedAggregate> provider_;

    /// Concurrency guard, checked before jet_initialize() mutates this problem
    /// and before any solve work begins. Jet::map builds each problem inside
    /// its worker job, so this entry point is the first place the selected
    /// backend is observable; rejecting here means no batch element ever
    /// reaches an Ipopt solve. Ipopt is not reliably re-entrant, so running
    /// several of its solves concurrently is unsupported rather than merely
    /// untested.
    ConvergenceFlags jet_run() override {
        if (this->nlp_solver_ == NLPSolvers::ipopt) {
            throw std::invalid_argument(
                "nlp_solver=ipopt cannot be used in a Jet batch run: Ipopt is not reliably "
                "re-entrant, so concurrent solves through it are unsupported. Run the ipopt "
                "backend one solve at a time (solve/optimize on a single problem), or set "
                "nlp_solver=interior_point for the batch.");
        }
        return Base::jet_run();
    }

    /// Single backend dispatch point for the five solve modes. The interior-point
    /// branch is the base's own dispatch, unchanged; the ipopt branch always
    /// runs a single NLP solve from the given input (the
    /// feasibility-then-optimize staging has no Ipopt analog).
    ///
    /// FRAGILITY, STATED PLAINLY: the base's member of this name is NOT
    /// virtual, so this one hides it rather than overriding it, and which of
    /// the two runs is decided by the STATIC type of the object the call is
    /// made on -- not by what the object actually is.
    ///
    /// That is sound for every path tycho owns. ODEPhaseBase::interior_point_call_impl
    /// and OptimalControlProblemBase::interior_point_call_impl are the only call sites
    /// on this side, and in both `this` is statically a class derived from
    /// BackendProblemBase, so both reach this one. The Jet batch path arrives
    /// the same way: the base's jet_run() dispatches through the VIRTUAL
    /// solve()/optimize()/... entry points, which land in those same tycho
    /// overrides, so a batch element also resolves to this member -- and the
    /// jet_run() override above has already rejected the Ipopt backend before
    /// any of that begins, so the batch path only ever takes the interior-point branch
    /// below anyway.
    ///
    /// The base's own member does still run, on objects that are not tycho
    /// problems: the solver library's NLPSolver derives from the library's base
    /// directly, never from this one, and its run() calls the base member. That
    /// is correct -- that surface has no backend selection to make -- and it is
    /// also why NLPSolver is not reachable through tycho's solver namespace
    /// (see detail/hven_namespaces.h). The rule to keep: anything that wants
    /// this dispatch must derive from BackendProblemBase and be seen as such at
    /// the call site.
    ///
    /// Defined out of line below, after the ipopt_backend::solve declaration it
    /// calls (a member function body defined in the class cannot name a
    /// namespace-scope function declared later in the file).
    NlpSolveOutput run_nlp_solver(JetJobModes mode, const Eigen::VectorXd &input);
};

namespace ipopt_backend {

/// True when this build was configured with Ipopt support linked in.
bool available();

/// Run Ipopt on the problem's transcribed NLP. A real implementation is linked
/// only in builds configured with Ipopt support; the stub throws
/// std::runtime_error.
BackendProblemBase::NlpSolveOutput
solve(BackendProblemBase &prob, BackendProblemBase::JetJobModes mode, const Eigen::VectorXd &input);

} // namespace ipopt_backend

inline BackendProblemBase::NlpSolveOutput
BackendProblemBase::run_nlp_solver(JetJobModes mode, const Eigen::VectorXd &input) {
    // This site branches on `== NLPSolvers::ipopt`, while the call-impl sites
    // in ode_phase_base.cpp / optimal_control_problem.cpp branch on
    // `== NLPSolvers::interior_point` (or its negation) to make the same backend
    // choice. Both predicates are equivalent today (only two backends exist),
    // but if a third backend enumerator is ever added, all of these sites
    // should be unified on `!= NLPSolvers::interior_point` ("anything that isn't interior_point
    // uses the generic NLP-solver path") rather than each needing its own added
    // `== <new-backend>` branch.
    if (this->nlp_solver_ == NLPSolvers::ipopt) {
        return ipopt_backend::solve(*this, mode, input);
    }
    return Base::run_nlp_solver(mode, input);
}

} // namespace tycho::solvers
