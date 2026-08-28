// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// The engine handle layer: thin tycho-side handles for the three solve
// engines a `solve()` call can dispatch to (the built-in interior-point
// solver, the SQP engine, and Ipopt as a peer engine), the non-owning
// `EngineRef` the pipeline dispatches on, and `run_engine_stage` -- the one
// seam that runs a single solver stage against a transcribed NLP, whichever
// engine it names, and reports back in the pipeline's own internal currency
// (`StageOutput`).
//
// `InteriorPointSolver` is not wrapped here: it is already a tycho-visible
// hven type (bridged in unchanged by hven_namespaces.h) with its own settings/
// result surface, so `EngineRef` names it directly. `SqpSolver` and
// `IpoptSolver` exist because neither the SQP driver nor Ipopt carries a
// tycho-shaped settings/result pair of its own -- SqpDriver is stateless per
// solve (a driver is constructed fresh from its options for every call) and
// Ipopt's whole surface today is a string/string option map plus an
// availability guard.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <variant>

#include <Eigen/Core>

#include <hven/drivers/interior_point_solver.h>
#include <hven/drivers/sqp_driver.h>
#include <hven/drivers/sqp_types.h>
#include <hven/model/non_linear_program.h>
#include <hven/warmstart/warm_start_data.h>

#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers/solve_types.h"

// Deliberately NOT including nlp_backend.h: BackendProblemBase's new solve()
// pipeline needs EngineRef/StageOutput from THIS header, so the
// dependency runs nlp_backend.h -> engines.h. Including it back here would
// make the pair circular; nothing declared in this header actually needs
// BackendProblemBase (a stray include from before that pipeline existed), so
// dropping it here is what breaks the cycle. Translation units that need both
// (e.g. engines.cpp's IpoptShimProblem) include nlp_backend.h directly.

namespace tycho::solvers {

/// Thin tycho-side handle owning SQP options; the hven driver is constructed
/// per solve from these options (hot reuse stays engine-internal, later).
class SqpSolver {
  public:
    SqpSolver() = default;
    explicit SqpSolver(const hven::solvers::SqpOptions &opts) : options_(opts) {}
    hven::solvers::SqpOptions &options() { return options_; }
    const hven::solvers::SqpOptions &options() const { return options_; }
    static const char *name() { return "SqpSolver"; }

  private:
    hven::solvers::SqpOptions options_{};
};

/// Ipopt as a peer engine handle (replaces the problem-owned NLPSolvers enum +
/// ipopt_options_ map). Constructible only when the backend is compiled in.
class IpoptSolver {
  public:
    /// @throws std::runtime_error naming ENABLE_IPOPT when
    ///         ipopt_backend::available() is false.
    IpoptSolver();
    std::map<std::string, std::string> &options() { return options_; }
    const std::map<std::string, std::string> &options() const { return options_; }
    static const char *name() { return "Ipopt"; }

  private:
    std::map<std::string, std::string> options_;
};

/// Non-owning engine reference the pipeline dispatches on.
using EngineRef = std::variant<InteriorPointSolver *, SqpSolver *, IpoptSolver *>;

/// @brief The engine's display name -- "InteriorPointSolver" / "SqpSolver" /
///        "Ipopt" -- read from `SqpSolver::name()`/`IpoptSolver::name()`
///        (both static; the pointer held is never dereferenced for this) and
///        hardcoded for InteriorPointSolver, which carries no such method of
///        its own.
const char *engine_name(EngineRef e);

/// What one stage hands back to the pipeline (internal currency).
struct StageOutput {
    tycho::ConvergenceFlags flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
    Eigen::VectorXd primal_;                               ///< caller's full space
    Eigen::VectorXd eq_lmults_, iq_lmults_, bound_lmults_; ///< declared space
    Eigen::VectorXd eq_cons_, iq_cons_; ///< residuals (adaptive mesh consumes these)
    StageResult report_;                ///< engine_name, flag, iterations, objective,
                                        ///< kkt/eq/iq residuals, wall_time, annex
    hven::solvers::WarmStartData warm_; ///< engine export after the stage
};

/// Run one stage. `mode` maps: Optimal -> the engine's optimality run
/// (IPM: optimize(x)); Feasible -> the feasibility run (IPM: solve(x)).
/// `warm` (may be null) is staged into the engine before the run (R5:
/// non-consuming; the engine one-shots it).
///
/// SQP + Feasible refuses: the SQP engine has no feasibility-only mode in
/// this milestone (M6+ on-demand); the refusal is checked first, before `nlp`
/// or `x0` is touched.
///
/// @throws std::invalid_argument if `engine` holds a SqpSolver and `mode ==
///         Mode::Feasible`, naming exactly why (see engines.cpp); whatever
///         the bound engine itself throws for a malformed problem, a stale
///         reduced-space NLP, a stamp/size mismatch on `warm`, or an
///         unavailable Ipopt build.
StageOutput run_engine_stage(EngineRef engine, Mode mode,
                             const std::shared_ptr<NonLinearProgram> &nlp,
                             const Eigen::VectorXd &x0, const hven::solvers::WarmStartData *warm);

/// @brief Prototype clone for jets: options copied, no runtime state.
///
/// InteriorPointSolver's overload returns by unique_ptr rather than by value
/// (a deliberate, documented departure from an earlier by-value draft of
/// this signature): InteriorPointSolver deletes all four of its copy/move
/// special members (kkt_sol_'s factorization and the globalization
/// components have no defined transfer semantics), and returning it by
/// value from a function that must apply settings() after construction is
/// not just slow but ill-formed -- named return value optimization does not
/// exempt a function from needing an accessible move (or copy) constructor
/// for the "as-if" call the standard requires, elided or not. heap
/// allocation is already how every long-lived owner in this codebase holds
/// one (e.g. a `shared_ptr<InteriorPointSolver>` engine handle a caller
/// keeps around across solve() calls), so this keeps the same ownership
/// shape rather than fighting it.
std::unique_ptr<InteriorPointSolver> clone_prototype(const InteriorPointSolver &e);
SqpSolver clone_prototype(const SqpSolver &e);
IpoptSolver clone_prototype(const IpoptSolver &e);

} // namespace tycho::solvers
