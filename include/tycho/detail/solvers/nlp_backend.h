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
#include <variant>

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
struct SolveOptions {
    Mode mode = Mode::Optimal;
    bool presolve = false; ///< true: run a Feasible stage first with the main engine.
    EngineRef *presolve_engine =
        nullptr;                 ///< overrides the presolve stage's engine (implies presolve).
    EngineRef *polish = nullptr; ///< second engine after the main stage.
    const hven::solvers::WarmStartData *warm = nullptr; ///< seeds the first stage.
};

/// @brief Refuses, by name, an engine asked to run the presolve stage that
///        has no feasibility-only mode of its own.
///
/// A presolve stage always runs `Mode::Feasible`, and only the interior-point
/// engine implements that mode -- `SqpSolver` and `IpoptSolver` each refuse it
/// from inside `run_engine_stage`. Checking it here instead means the
/// combination is refused as part of the refusal matrix, before the engine or
/// the NLP is touched, rather than from the middle of a call that has already
/// transcribed and latched.
///
/// @throws std::invalid_argument naming both halves: that `presolve=` runs a
///         feasibility stage, and which engine cannot run one.
inline void refuse_presolve_engine_without_feasible_mode(EngineRef presolve_engine) {
    if (std::holds_alternative<InteriorPointSolver *>(presolve_engine)) {
        return;
    }
    throw std::invalid_argument(
        fmt::format("presolve= runs a feasibility stage, and the {0} engine has no "
                    "feasibility-only mode; run the presolve stage on the interior-point "
                    "engine instead",
                    engine_name(presolve_engine)));
}

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
    /// `opts.mode == Mode::Feasible`; a presolve stage asked of an engine
    /// with no feasibility-only mode (SqpSolver/IpoptSolver, whether named by
    /// `opts.presolve_engine` or inherited from `engine` by
    /// `opts.presolve == true`); `engine`, `opts.presolve_engine` (if set) or
    /// `opts.polish` (if set) already inside another solve() call (a
    /// per-engine concurrency latch -- engines serve solves sequentially,
    /// never concurrently). A `opts.presolve_engine` with
    /// `opts.presolve == false` is not a refusal: it implies
    /// `presolve = true`.
    ///
    /// `opts.warm`, when set, seeds the primal/dual state of whichever stage
    /// runs FIRST (presolve if requested, else main). A payload that is
    /// EMPTY, or that carries a non-finite value in any block, is not a
    /// refusal: that stage simply runs cold, and the reason is recorded in
    /// the first stage's `engine_notes_["warm"]`. This is what makes the
    /// documented retry idiom -- solve, and on a non-convergent result solve
    /// again with `warm=` that result -- work in the case it exists for: a
    /// diverged stage's export is exactly the payload most likely to be
    /// non-finite. A payload that is usable but was taken under a DIFFERENT
    /// declaration is still refused, naming both keys' digests. Every stage
    /// after the first is a uniform value chain: a presolve stage's own
    /// `StageOutput::warm_` export seeds the main stage that follows it, and
    /// the main stage's own export seeds a `Mode::Optimal` polish stage that
    /// follows that -- independent of `opts.warm`, which only ever seeds the
    /// first stage. A predecessor's export seeds the next stage under the
    /// same two conditions: non-empty, and finite throughout.
    ///
    /// A stage whose engine is a different CLASS from the previous stage's
    /// (in this call, or in an earlier call on this same problem) is preceded
    /// by a re-transcription, so that whatever layout the previous engine
    /// left the NLP on never reaches the next engine's adapter --
    /// see require_declared_layout_for().
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

    /// @brief The most recently completed solve()'s result: a reference to
    ///        the snapshot this problem cached when that call returned.
    ///        Nothing in the problem writes into that cached value
    ///        afterwards, so reading it twice -- or after the problem has
    ///        since been re-transcribed, or re-solved differently -- still
    ///        reports the solve it was taken from, until the next solve()
    ///        replaces it wholesale.
    /// @throws std::logic_error if no solve() has completed on this
    ///         instance yet.
    const SolveResult &last_result() const;

  protected:
    /// @brief Transcribes the problem if needed and wires nlp_/provider_/
    ///        (and any per-type solver state) for the solve() call about to
    ///        run. One override per concrete problem type.
    virtual void prepare_solve() = 0;

    /// @brief Marks the current transcription stale, so that the next
    ///        prepare_solve() lays the problem out from its declaration
    ///        again. Every concrete problem type already owns exactly this
    ///        switch (`reset_transcription()`); this is the base-visible
    ///        name the staged pipeline reaches it through.
    virtual void invalidate_transcription() = 0;

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

    /// @brief Called before every stage, with that stage's engine: when the
    ///        engine is a different CLASS from the one that ran the previous
    ///        stage -- in this solve() call or in an earlier one on this same
    ///        problem instance -- the current transcription is marked stale,
    ///        so the prepare_solve() that follows lays the DECLARED layout
    ///        again before the stage runs.
    ///
    ///        This exists because an engine may legitimately leave the
    ///        transcribed NLP on a layout of its own: the interior-point
    ///        engine's default fixed-variable treatment eliminates every
    ///        fixed variable, leaving the NLP on a reduced variable space
    ///        after the solve, and its MakeConstraint treatment appends one
    ///        internal equality row per fixed variable. Neither shape is
    ///        something the SQP or Ipopt adapters accept -- they apply
    ///        variable bounds directly and refuse both by name. Rather than
    ///        let a caller meet that refusal halfway through a crossover the
    ///        API advertises (`solve(ipm, polish=sqp)`, or a `solve(ipm)`
    ///        then `solve(sqp, warm=...)` chain on one problem), the
    ///        pipeline restores the declared layout at the hand-off itself.
    ///        The cost is one transcription, paid only when the engine class
    ///        actually changes; the primal has already been written back by
    ///        accept_stage(), and a warm payload's stamp is keyed on the
    ///        DECLARATION, so it survives the re-transcription unchanged.
    void require_declared_layout_for(EngineRef engine);

    /// @brief The options of the solve() call presently dispatching on this
    ///        instance, or null outside of one. Set for the duration of the
    ///        run_adaptive_mesh() call above so an override can read
    ///        presolve/presolve_engine/warm without widening that method's
    ///        own fixed signature; never persists past the solve() call that
    ///        set it.
    const SolveOptions *active_solve_options_ = nullptr;

  private:
    /// @brief Which EngineRef alternative ran the most recent stage on this
    ///        problem (`EngineRef::index()`), or -1 before any stage has run.
    ///        Deliberately persists ACROSS solve() calls: the layout an
    ///        engine leaves behind lives on the transcribed NLP, not on the
    ///        call, so a two-call cross-engine chain needs the same hand-off
    ///        handling a single staged call does. See
    ///        require_declared_layout_for().
    int last_stage_engine_class_ = -1;

    /// @brief Value cache backing last_result(); unset until the first
    ///        solve() call completes.
    std::optional<SolveResult> last_result_cache_;

    /// @brief Non-owning prototype engine staged by set_jet_job(); unset
    ///        (nullopt) until that call runs. jet_run() clones this on every
    ///        call rather than dispatching through it directly -- see
    ///        set_jet_job()'s doc comment for the lifetime contract this
    ///        implies.
    std::optional<EngineRef> jet_prototype_;

    /// @brief Solve options set_jet_job() stages for jet_run() to pass the
    ///        per-call clone; copied by value from the caller's argument.
    ///        `jet_solve_options_->presolve_engine`/`->polish`, when set,
    ///        point into jet_presolve_engine_storage_/jet_polish_engine_storage_
    ///        below rather than into whatever storage the caller's own
    ///        SolveOptions argument used -- see set_jet_job()'s doc comment.
    std::optional<SolveOptions> jet_solve_options_;

    /// @brief Owns the EngineRef VALUE (a small pointer variant, not the
    ///        engine it names) that jet_solve_options_->presolve_engine
    ///        points at, when set_jet_job() was called with a presolve
    ///        engine. SolveOptions::presolve_engine is a raw, non-owning
    ///        EngineRef* -- fine for solve(), where the caller's own local
    ///        naturally outlives one synchronous call, but a staged
    ///        set_jet_job() call can run much later (inside a Jet::map on a
    ///        pool worker thread), so set_jet_job() copies the pointee HERE
    ///        instead of trusting a caller-side temporary to still be alive
    ///        then. The referenced engine ITSELF is still the caller's
    ///        responsibility to keep alive -- this only owns the pointer
    ///        variant, not the object it points to.
    std::optional<EngineRef> jet_presolve_engine_storage_;

    /// @brief Same as jet_presolve_engine_storage_ above, for
    ///        jet_solve_options_->polish.
    std::optional<EngineRef> jet_polish_engine_storage_;

    /// @brief Owns the WarmStartData VALUE (a value-semantic struct, unlike
    ///        the two above) that jet_solve_options_->warm points at, when
    ///        set_jet_job() was called with warm= set. Same rationale as
    ///        jet_presolve_engine_storage_/jet_polish_engine_storage_: a
    ///        staged call can run long after whatever the caller's own
    ///        SolveOptions.warm pointed at (e.g. a stack-local SolveResult)
    ///        has gone out of scope, so the payload is copied here instead
    ///        of trusted to still be alive later.
    std::optional<hven::solvers::WarmStartData> jet_warm_storage_;

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

    /// @brief Stages a batched (Jet) solve on this problem: jet_run() clones
    ///        `prototype` -- and, if staged, `opts.presolve_engine`/
    ///        `opts.polish` too -- via clone_prototype(), once per jet_run()
    ///        call, and calls this->solve() on the clones with `opts`.
    ///        `prototype` (and any staged auxiliary engine) is never run
    ///        itself, only copied from. This is what lets one prototype (and
    ///        one presolve/polish engine) be shared as the job description
    ///        across every problem in a Jet::map batch: each pool-worker's
    ///        jet_run() call gets its own independent set of clones, so none
    ///        of them ever contends on the shared engines' own per-engine
    ///        concurrency latch (solve()'s "this engine instance is already
    ///        inside a solve" guard), and every shared engine is left
    ///        exactly as cold as it started once the whole batch finishes.
    ///
    ///        Refusal-matrix predicates that do not depend on engine/NLP
    ///        state (`polish=` with `mode=Feasible`, `presolve=`/
    ///        `presolve_engine=` with `mode=Feasible`, and a presolve stage
    ///        asked of an engine with no feasibility-only mode) are checked
    ///        here too, eagerly -- the same checks solve() runs, but surfaced
    ///        at the call that staged the bad combination rather than from
    ///        inside a Jet::map pool worker.
    ///
    ///        jet_run()'s clones are NOT byte-identical to what solve() would
    ///        hand the same engine object: every InteriorPointSolver clone
    ///        gets `qp_threads_` pinned to 1 (the shared thread pool already
    ///        parallelizes across jobs; hven's own per-worker MKL pin does
    ///        not survive into the engine call, since InteriorPointSolver
    ///        re-applies `qp_threads_` at every solve entry) and
    ///        `print_level_` forced to a silent value UNLESS the source
    ///        engine had already moved it off the class default of 0 -- so a
    ///        batched run of many jobs is quiet by default, and a caller
    ///        opts into per-job console output by setting `print_level`
    ///        explicitly on whichever engine it stages.
    ///
    /// @param prototype Non-owning reference. The CALLER must keep the
    ///        referenced engine alive for as long as jet_run() can still be
    ///        called on this problem -- i.e. for the duration of any
    ///        Jet::map call this problem participates in -- the same
    ///        lifetime contract solve()'s own EngineRef parameter already
    ///        carries, just held open longer because this stages the call
    ///        rather than running it immediately. clear_jet_job() is the
    ///        supported way to end that requirement early.
    /// @param opts Copied by value onto this problem. When `opts.presolve_engine`,
    ///        `opts.polish`, or `opts.warm` are set, the VALUE each points at
    ///        is also copied -- into jet_presolve_engine_storage_/
    ///        jet_polish_engine_storage_/jet_warm_storage_, which this
    ///        problem instance owns -- so the caller's own SolveOptions
    ///        argument (and whatever temporary held those pointees) may go
    ///        out of scope immediately after this call returns. The
    ///        referenced ENGINE objects named by `presolve_engine`/`polish`
    ///        are a different matter (an engine is not copyable/clonable in
    ///        place): those are only cloned later, inside jet_run(), so the
    ///        CALLER must still keep them alive for as long as jet_run() can
    ///        run -- exactly the same contract `prototype` carries above.
    ///        `opts.warm`'s pointee (WarmStartData) IS a value type, so it is
    ///        fully copied here and carries no such requirement.
    /// @throws std::invalid_argument per the refusal-matrix predicates above.
    void set_jet_job(EngineRef prototype, SolveOptions opts) {
        if (opts.polish != nullptr && opts.mode == Mode::Feasible) {
            throw std::invalid_argument(
                "polish= is an optimality refinement; it cannot follow mode=Feasible");
        }
        if ((opts.presolve || opts.presolve_engine != nullptr) && opts.mode == Mode::Feasible) {
            throw std::invalid_argument(
                "presolve= runs a feasibility stage; mode=Feasible already is one");
        }
        if (opts.presolve_engine != nullptr) {
            // Not a refusal: an explicit override engine implies presolve --
            // mirrors solve()'s own normalization.
            opts.presolve = true;
        }
        if (opts.presolve) {
            refuse_presolve_engine_without_feasible_mode(
                opts.presolve_engine != nullptr ? *opts.presolve_engine : prototype);
        }

        this->jet_prototype_ = prototype;

        if (opts.presolve_engine != nullptr) {
            this->jet_presolve_engine_storage_ = *opts.presolve_engine;
            opts.presolve_engine = &*this->jet_presolve_engine_storage_;
        } else {
            this->jet_presolve_engine_storage_.reset();
        }

        if (opts.polish != nullptr) {
            this->jet_polish_engine_storage_ = *opts.polish;
            opts.polish = &*this->jet_polish_engine_storage_;
        } else {
            this->jet_polish_engine_storage_.reset();
        }

        if (opts.warm != nullptr) {
            this->jet_warm_storage_ = *opts.warm;
            opts.warm = &*this->jet_warm_storage_;
        } else {
            this->jet_warm_storage_.reset();
        }

        this->jet_solve_options_ = std::move(opts);
    }

    /// @brief Convenience overloads: the same call, taking a concrete engine
    ///        by lvalue reference instead of an already-formed EngineRef --
    ///        mirrors solve()'s own convenience overloads above.
    void set_jet_job(InteriorPointSolver &e, SolveOptions opts) {
        this->set_jet_job(EngineRef{&e}, std::move(opts));
    }
    void set_jet_job(SqpSolver &e, SolveOptions opts) {
        this->set_jet_job(EngineRef{&e}, std::move(opts));
    }
    void set_jet_job(IpoptSolver &e, SolveOptions opts) {
        this->set_jet_job(EngineRef{&e}, std::move(opts));
    }

    /// @brief Un-stages whatever set_jet_job() staged: detaches the
    ///        prototype and clears every piece of owned storage (the
    ///        auxiliary-engine references, the warm-start payload, and the
    ///        options themselves). This is the supported way to let a
    ///        staged engine go out of scope -- without calling this first,
    ///        the problem retains a non-owning pointer to it indefinitely,
    ///        and any later jet_run() (or Jet::map call that includes this
    ///        problem) dereferences whatever is left there.
    ///
    ///        jet_release() deliberately does NOT do this: it runs at the
    ///        end of every jet_run() call, successful or not, and a batch
    ///        must stay re-runnable afterward -- clearing the staged job on
    ///        every call would defeat the point of staging it once for many
    ///        jet_run() calls.
    void clear_jet_job() {
        this->jet_prototype_.reset();
        this->jet_solve_options_.reset();
        this->jet_presolve_engine_storage_.reset();
        this->jet_polish_engine_storage_.reset();
        this->jet_warm_storage_.reset();
    }

    /// @brief Runs the job set_jet_job() staged: jet_initialize(), clone the
    ///        staged prototype engine (and, if staged, the presolve/polish
    ///        auxiliary engines), this->solve() the clones with the staged
    ///        options, jet_release(), and return the deciding stage's
    ///        convergence flag. last_result() reflects this call afterward,
    ///        same as any other solve() call. Called by Jet::map on pool
    ///        worker threads (see the concurrency note on set_jet_job()
    ///        above for why that is safe against shared engines).
    ///
    /// jet_release() still runs when this->solve() throws, so a failed
    /// jet_run() leaves the problem in the same released state a successful
    /// one would.
    ///
    /// @throws std::logic_error if set_jet_job() was never called on this
    ///         instance (or clear_jet_job() ran since). Otherwise, whatever
    ///         this->solve() itself throws for a malformed problem, a stale
    ///         reduced-space NLP, or an engine-specific mode refusal.
    ConvergenceFlags jet_run();
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
