// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// The tycho-side problem base every optimization problem type derives from,
// plus the Ipopt backend entry points the engine seam (engines.h/engines.cpp)
// dispatches a Mode::Optimal stage to when the caller's engine is an
// IpoptSolver.
//
// Backend selection is no longer a property stored on the problem: it is
// simply which engine (InteriorPointSolver, SqpSolver, or IpoptSolver) the
// caller passes to solve() (BackendProblemBase::solve(EngineRef,
// SolveOptions), declared below; the engine-level dispatch itself lives in
// engines.h/engines.cpp). This header carries only what every problem type
// needs regardless of backend: the transcribed NLP handle, the partitioned-
// evaluation settings, and the solve()/last_result() surface.
//
// This header deliberately contains no Ipopt includes and no Ipopt types
// beyond the plain-data IpoptRunInfo/IpoptSolveOutput below, so it is safe to
// include in every build regardless of how the project was configured. The
// Ipopt dependency itself is confined to the translation unit that implements
// ipopt_backend::solve.
//
// BackendProblemBase owns the NLP handle and the partitioned-evaluation
// settings directly rather than inheriting them from hven's
// OptimizationProblemBase: hven's own base
// (dep/hven/include/hven/drivers/optimization_problem_base.h) is unchanged
// and still used as-is by anything on the solver-library side of the
// boundary (e.g. NLPSolver), but it also owns an InteriorPointSolver instance
// and a non-virtual run_nlp_solver() that dispatches through it -- neither of
// which fits a problem type that may solve through any of several engines,
// none of which it owns.

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#include <fmt/format.h>

#include <hven/detail/drivers/interior_point_solver_fwd.h>
#include <hven/detail/interior/utils/thread_pool.h>
#include <hven/drivers/interior_point_solver.h>
#include <hven/model/non_linear_program.h>

#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers/engines.h"

namespace tycho::solvers {

/// The transcribed problem published as a claim-stream-bearing provider. Named
/// here and defined in solvers_vf/transcribed_aggregate.h, so that a problem
/// can hold one without every consumer of this header seeing the layout it
/// publishes.
class TranscribedAggregate;

/// Outcome of one Ipopt run on a transcribed NLP. Sentinel values (-1 /
/// empty / ran_ == false) mean the run never completed (an early abort or an
/// exception before finalize_solution).
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

/// Everything one ipopt_backend::solve() call reports: the primal/dual point
/// Ipopt returned, and the diagnostics of that run.
struct IpoptSolveOutput {
    Eigen::VectorXd variables_;
    Eigen::VectorXd eq_lmults_;
    Eigen::VectorXd iq_lmults_;
    ConvergenceFlags flag_ = ConvergenceFlags::NOTCONVERGED;
    IpoptRunInfo info_;
};

/// @brief Options for the engine-driven staged solve(): which mode the
///        main stage runs, whether a feasibility presolve stage precedes it
///        and on which engine, whether an optimality polish stage follows it
///        and on which engine, and the warm-start currency (if any) that
///        seeds the first stage that runs.
///
/// Verbatim per the solve-API spec: every field name and default here is
/// binding.
struct SolveOptions {
    Mode mode = Mode::Optimal;
    bool presolve = false; ///< true: run a Feasible stage first with the main engine.
    EngineRef *presolve_engine =
        nullptr;                 ///< overrides the presolve stage's engine (implies presolve).
    EngineRef *polish = nullptr; ///< second engine after the main stage.
    const hven::solvers::WarmStartData *warm = nullptr; ///< seeds the first stage.
};

/// Every tycho optimization problem — OptimizationProblem, ODEPhaseBase and
/// OptimalControlProblemBase alike — derives from this rather than from the
/// solver library's OptimizationProblemBase directly, so that backend
/// selection is available wherever a problem is. It is the class the Python
/// binding exposes under the name "OptimizationProblemBase"; the distinct C++
/// name keeps the two apart at every unqualified use site.
struct BackendProblemBase {

    /// @brief Number of evaluation partitions the NLP is split over.
    int num_partitions_ = 1;

    /// @brief The problem's program (built by the derived class's make_nlp()).
    std::shared_ptr<NonLinearProgram> nlp_;

    /// The transcribed problem published as a claim-stream-bearing provider:
    /// the declaration this problem was laid from, and the per-slot coordinates
    /// a consumer needs in order to lay a destination of its own. Replaced by
    /// every transcription, and null until the first one runs.
    std::shared_ptr<TranscribedAggregate> provider_;

    virtual ~BackendProblemBase() = default;

    /// @brief Constructs the problem and applies the default partitioning.
    BackendProblemBase() { this->init_partitions(); }

    // -------------------------------------------------------------------
    // The engine-driven staged solve(). Implemented once, in
    // solve_pipeline.cpp, dispatching through the protected hooks below;
    // each concrete problem type (OptimizationProblem, ODEPhaseBase,
    // OptimalControlProblemBase) supplies its own override set. A derived
    // class that declares its own `solve()` hides this overload of that
    // name unless it re-exposes it with `using BackendProblemBase::solve;`
    // -- none of the three currently declares a 0-arg `solve()` of its own,
    // so none needs to.
    // -------------------------------------------------------------------

    /// @brief Runs a staged solve against `engine`: an optional Feasible
    ///        presolve stage, the main stage (`opts.mode`), and an optional
    ///        Optimal polish stage, in that order. Every stage that runs
    ///        appends its StageResult to the returned SolveResult in run
    ///        order and runs to completion regardless of an earlier stage's
    ///        flag -- a non-convergent stage is a value in that stage's
    ///        report, not a thrown exception.
    ///
    /// Refused before any stage runs (`std::invalid_argument`, in this
    /// order): `opts.polish` paired with `opts.mode == Mode::Feasible`;
    /// `opts.presolve`/`opts.presolve_engine` paired with
    /// `opts.mode == Mode::Feasible`; `opts.presolve_engine` set with
    /// `engine`, `opts.presolve_engine` (if set) or `opts.polish` (if set)
    /// already inside another solve() call (a per-engine concurrency latch --
    /// engines serve solves sequentially, never concurrently). A
    /// `opts.presolve_engine` with `opts.presolve == false` is not a
    /// refusal: it implies `presolve = true`.
    ///
    /// `opts.warm`, when set, seeds the primal/dual state of whichever stage
    /// runs FIRST (presolve if requested, else main) after a stamp check:
    /// its `DeclarationKey` must match the current transcription's, or the
    /// call refuses naming both keys' digests. Every stage after the first is
    /// a uniform value chain: a presolve stage's own `StageOutput::warm_`
    /// export seeds the main stage that follows it, and the main stage's own
    /// export seeds a `Mode::Optimal` polish stage that follows that --
    /// independent of `opts.warm`, which only ever seeds the first stage. A
    /// predecessor's export seeds the next stage only when it is non-empty;
    /// an empty export (the exporting stage captured nothing to hand
    /// forward) leaves the next stage unseeded rather than failing on a
    /// block-size mismatch.
    ///
    /// `prepare_solve()` (transcribe-if-needed) runs before the stamp check,
    /// so the check compares against the transcription this call will
    /// actually solve.
    ///
    /// @throws std::invalid_argument per the refusal matrix above, or
    ///         whatever `run_engine_stage` itself throws for a malformed
    ///         problem, a stale reduced-space NLP, or an engine-specific
    ///         mode refusal (e.g. SQP/Ipopt + Mode::Feasible).
    SolveResult solve(EngineRef engine, const SolveOptions &opts = SolveOptions{});

    /// @brief Convenience overloads: the same call, taking a concrete engine
    ///        by lvalue reference instead of an already-formed EngineRef, so
    ///        a caller can write `problem.solve(ipm, opts)` directly. Each
    ///        forwards to the EngineRef overload above via `&e`; EngineRef
    ///        itself is unchanged (still a non-owning pointer variant) --
    ///        these exist purely so the caller does not have to spell that
    ///        `&` out at every call site.
    SolveResult solve(InteriorPointSolver &e, const SolveOptions &opts = SolveOptions{}) {
        return this->solve(EngineRef{&e}, opts);
    }
    SolveResult solve(SqpSolver &e, const SolveOptions &opts = SolveOptions{}) {
        return this->solve(EngineRef{&e}, opts);
    }
    SolveResult solve(IpoptSolver &e, const SolveOptions &opts = SolveOptions{}) {
        return this->solve(EngineRef{&e}, opts);
    }

    /// @brief The most recently completed solve()'s result, as a value
    ///        copy -- reading it twice, or after the problem has since been
    ///        re-transcribed or re-solved differently, still returns the
    ///        snapshot taken at that solve() call.
    /// @throws std::logic_error if no solve() has completed on this
    ///         instance yet.
    const SolveResult &last_result() const;

  protected:
    /// @brief Transcribes the problem if needed and wires nlp_/provider_/
    ///        (and any per-type solver state) for the solve() call about to
    ///        run. One override per concrete problem type.
    virtual void prepare_solve() = 0;

    /// @brief The full decision-variable vector solve() seeds the first
    ///        stage's starting point with -- the active trajectory/static
    ///        parameters flattened (Phase/OCP) or the active variables
    ///        vector (a bare VF problem), read AFTER prepare_solve() has run.
    virtual Eigen::VectorXd initial_primal() const = 0;

    /// @brief Writes one finished stage's primal/dual output back onto the
    ///        problem (and its post-optimality constraint info, when the
    ///        engine reported residuals), so the next stage's
    ///        initial_primal() reflects it.
    virtual void accept_stage(const StageOutput &out) = 0;

    /// @brief Slices `r.phases_` from the just-finished solve; `r` already
    ///        carries `r.stages_`/`r.flag_`/`r.warm_`/`r.structure_key_`.
    ///        Base: no-op, `r.phases_` stays empty -- a bare VF problem
    ///        (OptimizationProblem) has no phases and keeps this no-op;
    ///        ODEPhaseBase and OptimalControlProblemBase each override it
    ///        with their own real per-phase slicing.
    virtual void fill_phase_results(SolveResult &r) const { (void)r; }

    /// @brief Whether this problem type runs an adaptive-mesh loop inside
    ///        the main stage. Base: false (a bare VF problem never does).
    virtual bool adaptive_mesh_enabled() const { return false; }

    /// @brief Runs the main stage as an adaptive-mesh loop: `active_
    ///        solve_options_->presolve` (if set) runs a Feasible stage
    ///        before mesh iteration 0 only, unless the override's own
    ///        `solve_only_first_`-shaped knob says to repeat it every
    ///        iteration; every iteration's optimality run (`mode`) is one
    ///        `run_engine_stage` call. The override appends one StageResult
    ///        per internal engine call to `r.stages_`, in run order, and
    ///        sets `r.warm_` to the LAST main-mode stage's export (the seed
    ///        solve() hands a polish stage). Only called when
    ///        `adaptive_mesh_enabled()` is true; Phase and OCP each reshape
    ///        their own mesh loop by calling run_amr_loop() below with their
    ///        mesh-specific callbacks. The base body is unreachable through
    ///        solve() (it never calls this when `adaptive_mesh_enabled()` is
    ///        false) and throws if reached some other way.
    /// @throws std::invalid_argument if the MAIN stage's engine reports no
    ///         constraint residuals -- mesh error estimation has nothing to
    ///         measure from -- naming the engine. The presolve stage (if
    ///         any) is not held to this: it is never the stage mesh error
    ///         estimation reads from, so an engine without residual
    ///         reporting is a legal presolve even under adaptive mesh.
    virtual tycho::ConvergenceFlags run_adaptive_mesh(EngineRef engine, Mode mode, SolveResult &r);

    /// @brief The mesh-representation-specific callbacks run_amr_loop()
    ///        needs from a concrete adaptive-mesh problem type. Everything
    ///        that is NOT mesh-representation-specific -- stage bookkeeping,
    ///        the re-transcription gate, the residual-report check, warm
    ///        seeding, the banner/timer -- lives once in run_amr_loop()
    ///        itself, so the two overrides (Phase, OCP) cannot drift apart
    ///        on those parts.
    struct AmrLoopHooks {
        std::function<void()> init_mesh;               ///< init_mesh_refinement() / init_meshs().
        std::function<bool(int iter)> mesh_converged;  ///< check_mesh()-shaped test for `iter`.
        std::function<void()> update_mesh;             ///< update_mesh() / update_meshs(...).
        std::function<void(int iter)> print_iteration; ///< per-iteration mesh-state print.
        const char *converged_message = "";            ///< e.g. "Mesh Converged".
        const char *not_converged_message = "";        ///< e.g. "Mesh Not Converged".
    };

    /// @brief Shared adaptive-mesh loop body: presolve (optional, before
    ///        mesh iteration 0, repeated every iteration when
    ///        `solve_only_first` is false) then one main-mode
    ///        `run_engine_stage` call per mesh iteration, re-transcribing
    ///        (`prepare_solve()`) before every stage -- a mesh iteration
    ///        invalidates the prior transcription (`reset_transcription()`),
    ///        exactly as the old per-type `interior_point_call_impl` did at
    ///        its own entry on every call. Only the MAIN stage is required
    ///        to report residuals; a misuse refusal is raised before
    ///        `accept_stage()` runs, so it never leaves the host
    ///        half-mutated. `hooks` supplies the mesh-representation-specific
    ///        pieces; `mesh_abort_flag`/`max_mesh_iters`/`solve_only_first`/
    ///        `print_mesh_info` are the calling type's own settings, read by
    ///        the override and passed through unchanged.
    /// @return The deciding stage's convergence flag -- always
    ///         `r.stages_.back().flag_` on return.
    /// @throws std::invalid_argument if the main stage's engine reports no
    ///         constraint residuals, naming the engine.
    tycho::ConvergenceFlags run_amr_loop(EngineRef engine, Mode mode, SolveResult &r,
                                         tycho::ConvergenceFlags mesh_abort_flag,
                                         int max_mesh_iters, bool solve_only_first,
                                         bool print_mesh_info, const AmrLoopHooks &hooks);

    /// @brief The options of the solve() call presently dispatching on this
    ///        instance, or null outside of one. Set for the duration of the
    ///        run_adaptive_mesh() call above so an override can read
    ///        presolve/presolve_engine/warm without widening that method's
    ///        own fixed signature; never persists past the solve() call that
    ///        set it.
    const SolveOptions *active_solve_options_ = nullptr;

  private:
    /// @brief Value cache backing last_result(); unset until the first
    ///        solve() call completes.
    std::optional<SolveResult> last_result_cache_;

  public:
    /// Compute default partition count from the global thread budget.
    /// Over-partitions by 4x so the work-stealing pool can smooth out
    /// unequal partition costs.
    ///
    /// @return 1 on a single-thread budget, else 4x the thread count.
    static int default_num_partitions() {
        int nt = hven::utils::get_num_threads();
        if (nt <= 1)
            return 1;
        return nt * 4;
    }

    /// Applies the default partitioning: num_partitions_ from
    /// default_num_partitions(). QP thread count is a separate, per-engine
    /// setting now (there is no problem-owned optimizer to apply it to) --
    /// set it on whichever InteriorPointSolver engine is passed to solve().
    virtual void init_partitions() { this->num_partitions_ = default_num_partitions(); }

    /// @brief Sets the number of evaluation partitions the problem is split over.
    /// @param num_partitions Partition count; must be positive.
    ///
    /// Partition count and QP thread count are independent settings: the
    /// latter is set on whichever InteriorPointSolver engine is passed to
    /// solve() (`engine.set_qp_threads(n)`).
    ///
    /// @throws std::invalid_argument if `num_partitions < 1`.
    virtual void set_num_partitions(int num_partitions) {
        if (num_partitions < 1) {
            throw std::invalid_argument("Number of partitions must be positive");
        }
        this->num_partitions_ = num_partitions;
    }

    /// Prepares the problem for inline (non-partitioned) evaluation inside
    /// a batched (Jet) solve; must leave num_partitions_ == 1.
    virtual void jet_initialize() = 0;

    /// @brief Releases whatever jet_initialize() acquired.
    virtual void jet_release() = 0;

    /// @brief Interim placeholder for the batched (Jet) solve entry point.
    ///
    /// The mode-sequence surface jet_run() used to dispatch through
    /// (JetJobModes/jet_job_mode_) is retired along with the five mode
    /// methods; the batched entry point is staged through set_jet_job
    /// instead, landing in a follow-up task. Until then this always throws,
    /// naming what replaces it, so Jet::map keeps compiling against this
    /// entry point without silently running a stale job description.
    ///
    /// @throws std::logic_error unconditionally.
    ConvergenceFlags jet_run() {
        throw std::logic_error(
            "jet_run(): the retired mode-sequence surface (jet_job_mode_/JetJobModes) that "
            "used to drive this is gone; jets are staged via set_jet_job(), not yet landed on "
            "this branch");
    }
};

namespace ipopt_backend {

/// True when this build was configured with Ipopt support linked in.
bool available();

/// Run Ipopt on `nlp` from starting point `x0`. `ipopt_options` are string
/// key/value options forwarded verbatim to Ipopt (e.g.
/// {"linear_solver", "pardisomkl"}), applied after the matched-tolerance
/// baseline so they win. `tolerance_baseline` supplies that baseline (tol,
/// constr_viol_tol, acceptable_tol, acceptable_constr_viol_tol, max_iter,
/// obj_scale) -- an IpoptSolver engine carries no tolerance settings of its
/// own, so the caller decides what baseline to match (the engine seam in
/// engines.cpp matches a default-constructed InteriorPointSolver::Settings).
///
/// A real implementation is linked only in builds configured with Ipopt
/// support; the stub throws std::runtime_error.
///
/// @throws std::runtime_error if `nlp` is null, or (stub build) unconditionally.
IpoptSolveOutput solve(const std::shared_ptr<NonLinearProgram> &nlp, const Eigen::VectorXd &x0,
                       const std::map<std::string, std::string> &ipopt_options,
                       const InteriorPointSolver::Settings &tolerance_baseline);

} // namespace ipopt_backend

} // namespace tycho::solvers
