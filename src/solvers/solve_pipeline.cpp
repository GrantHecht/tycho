// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// The staged solve() pipeline on BackendProblemBase: the refusal matrix,
// the per-engine concurrency latch, the warm-start stamp pre-check, and the
// presolve/main/polish stage sequence -- coded once here, for every concrete
// problem type (OptimizationProblem, ODEPhaseBase, OptimalControlProblemBase),
// which supply only the five hooks declared in nlp_backend.h
// (prepare_solve/initial_primal/accept_stage/fill_phase_results/
// adaptive_mesh_enabled/run_adaptive_mesh).
//
// STAGE SEQUENCE. presolve (Mode::Feasible, on presolve_engine if given, else
// the main engine) -> main (opts.mode; an adaptive-mesh problem runs its own
// mesh loop here, via run_adaptive_mesh) -> polish (Mode::Optimal, on
// opts.polish, warm-seeded from the main stage's own export). Every stage
// that runs appends its StageResult, in run order, and runs to completion
// regardless of an earlier stage's flag: a non-convergent stage is a value in
// that stage's report, never a thrown exception.
//
// WARM SEEDING is a uniform chain: the caller's opts.warm, when set, seeds
// only whichever stage runs FIRST (presolve if requested, else main) -- a
// one-shot value, exactly as run_engine_stage/stage_warm_start already
// document. EVERY stage after the first is seeded by its immediate
// predecessor's own StageOutput::warm_ export -- a presolve stage's export
// seeds the main stage that follows it, and the main stage's export seeds a
// polish stage that follows that, independent of opts.warm. This applies
// inside the adaptive-mesh loop too: the pre-loop presolve's export seeds
// mesh iteration 0's main stage, and (when solve_only_first is false and the
// presolve repeats every iteration) each iteration's own presolve export
// seeds that same iteration's main stage. The adaptive-mesh override still
// sets SolveResult::warm_ itself from whichever main-mode stage ran LAST,
// since that is the one a polish stage after the loop should seed from.

#include "tycho/detail/solvers/nlp_backend.h"

// nlp_backend.h only forward-declares TranscribedAggregate (provider_'s
// pointee); this TU calls provider_->declaration(), which needs the
// complete type.
#include "tycho/detail/solvers_vf/transcribed_aggregate.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <hven/detail/interior/utils/timer.h>
#include <hven/model/structure_identity.h>

namespace tycho::solvers {

namespace {

/// @brief The address a concurrency latch keys on for whichever concrete
///        engine type an EngineRef holds -- type-erased, so two EngineRef
///        values naming the SAME underlying engine object key identically
///        regardless of which alternative the variant happens to hold at
///        each call site.
const void *engine_identity(EngineRef e) {
    return std::visit([](auto *p) -> const void * { return p; }, e);
}

/// @brief Process-wide set of engine identities presently inside a solve()
///        call. InteriorPointSolver is an hven type this library cannot add
///        a latch member to, and SqpSolver/IpoptSolver are tycho's own but
///        gain nothing from carrying one of their own -- EngineRef already
///        erases which alternative is live at a given call site -- so one
///        side table, keyed by the type-erased pointer, serves all three
///        uniformly.
std::mutex &engine_latch_mutex() {
    static std::mutex m;
    return m;
}
std::unordered_set<const void *> &engines_in_solve() {
    static std::unordered_set<const void *> s;
    return s;
}

/// @brief RAII latch on one engine identity: acquired in the constructor,
///        released in the destructor, refusing a re-entrant acquire.
class EngineLatch {
  public:
    explicit EngineLatch(const void *key) : key_(key) {
        std::lock_guard<std::mutex> lock(engine_latch_mutex());
        if (!engines_in_solve().insert(key_).second) {
            throw std::invalid_argument(
                "this engine instance is already inside a solve; engines serve solves "
                "sequentially");
        }
    }
    EngineLatch(const EngineLatch &) = delete;
    EngineLatch &operator=(const EngineLatch &) = delete;
    EngineLatch(EngineLatch &&other) noexcept : key_(other.key_) { other.key_ = nullptr; }
    EngineLatch &operator=(EngineLatch &&) = delete;
    ~EngineLatch() {
        if (key_ == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(engine_latch_mutex());
        engines_in_solve().erase(key_);
    }

  private:
    const void *key_;
};

/// @brief Latches every DISTINCT engine identity this solve() call names --
///        `engine` itself, plus `opts.presolve_engine`/`opts.polish` when set
///        and not already latched -- for the call's duration. Using the same
///        engine object in two roles within ONE call (e.g. polish == the
///        main engine) latches it once, not twice: that is sequential reuse
///        within one call, not concurrency across two.
std::vector<EngineLatch> latch_solve_engines(EngineRef engine, const SolveOptions &opts) {
    std::vector<const void *> keys;
    keys.push_back(engine_identity(engine));
    if (opts.presolve_engine != nullptr) {
        keys.push_back(engine_identity(*opts.presolve_engine));
    }
    if (opts.polish != nullptr) {
        keys.push_back(engine_identity(*opts.polish));
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    std::vector<EngineLatch> latches;
    latches.reserve(keys.size());
    for (const void *k : keys) {
        latches.emplace_back(k);
    }
    return latches;
}

/// @brief Appends one stage's report to `r.stages_`, tagging it with `role`.
void append_stage(SolveResult &r, const StageOutput &out, const char *role) {
    StageResult report = out.report_;
    report.role_ = role;
    r.stages_.push_back(std::move(report));
}

} // namespace

SolveResult BackendProblemBase::solve(EngineRef engine, const SolveOptions &opts_in) {
    SolveOptions opts = opts_in;

    // --- Refusal matrix, checked before touching the engine or the NLP ---
    if (opts.polish != nullptr && opts.mode == Mode::Feasible) {
        throw std::invalid_argument(
            "polish= is an optimality refinement; it cannot follow mode=Feasible");
    }
    if ((opts.presolve || opts.presolve_engine != nullptr) && opts.mode == Mode::Feasible) {
        throw std::invalid_argument(
            "presolve= runs a feasibility stage; mode=Feasible already is one");
    }
    if (opts.presolve_engine != nullptr) {
        // Not a refusal: an explicit override engine implies presolve.
        opts.presolve = true;
    }

    // Throws std::invalid_argument, naming the reason, if any named engine is
    // already inside another solve() call. Held for the rest of this call.
    std::vector<EngineLatch> latches = latch_solve_engines(engine, opts);

    // --- Transcription / wiring ---
    this->prepare_solve();

    // --- Warm-start stamp pre-check ---
    // The engine's own check at solve entry is the authoritative one (it
    // runs regardless of this pre-check); this exists only so the refusal
    // surfaces before any stage work runs.
    if (opts.warm != nullptr) {
        const hven::solvers::DeclarationKey current_key =
            hven::solvers::declaration_key(this->provider_->declaration());
        if (!(current_key == opts.warm->structure_key_)) {
            throw std::invalid_argument(fmt::format(
                "solve: the warm-start payload's declaration key (digest={0}) does not match "
                "the current transcription's declaration key (digest={1}); the payload was "
                "taken from a different declared problem",
                opts.warm->structure_key_.digest(), current_key.digest()));
        }
    }

    SolveResult result;
    const hven::solvers::WarmStartData *first_stage_warm = opts.warm;

    if (this->adaptive_mesh_enabled()) {
        this->active_solve_options_ = &opts;
        tycho::ConvergenceFlags amr_flag;
        try {
            amr_flag = this->run_adaptive_mesh(engine, opts.mode, result);
        } catch (...) {
            this->active_solve_options_ = nullptr;
            throw;
        }
        this->active_solve_options_ = nullptr;
        // run_adaptive_mesh is documented as returning the deciding stage's
        // flag -- the one it just appended to result.stages_. Cross-check
        // that agreement rather than silently trusting one of the two: a
        // future override that lets them drift apart should fail loudly,
        // not silently pick whichever this call happened to read.
        if (result.stages_.empty() || amr_flag != result.stages_.back().flag_) {
            throw std::logic_error(
                "run_adaptive_mesh: returned flag disagrees with the last appended stage's "
                "flag_ (or appended no stage at all) -- these must stay in agreement");
        }
    } else {
        // Holds the presolve stage's own warm export alive long enough to
        // seed the main stage below -- part of the uniform value chain
        // (top-of-file WARM SEEDING note): each stage after the first is
        // seeded by its immediate predecessor's export, not by opts.warm
        // again.
        hven::solvers::WarmStartData presolve_warm;
        if (opts.presolve) {
            EngineRef presolve_engine =
                opts.presolve_engine != nullptr ? *opts.presolve_engine : engine;
            StageOutput out = run_engine_stage(presolve_engine, Mode::Feasible, this->nlp_,
                                               this->initial_primal(), first_stage_warm);
            this->accept_stage(out);
            append_stage(result, out, "presolve");
            presolve_warm = std::move(out.warm_);
            first_stage_warm = &presolve_warm;
        }

        StageOutput main_out = run_engine_stage(engine, opts.mode, this->nlp_,
                                                this->initial_primal(), first_stage_warm);
        this->accept_stage(main_out);
        append_stage(result, main_out, "main");
        result.warm_ = main_out.warm_;
    }

    if (opts.polish != nullptr) {
        StageOutput polish_out = run_engine_stage(*opts.polish, Mode::Optimal, this->nlp_,
                                                  this->initial_primal(), &result.warm_);
        this->accept_stage(polish_out);
        append_stage(result, polish_out, "polish");
        result.warm_ = polish_out.warm_;
    }

    if (result.stages_.empty()) {
        // Unreachable through the paths above (the main stage always runs
        // and always appends), but the failure mode if it somehow happened
        // -- SolveResult::final_stage()/flag_ read out of an empty stage
        // list -- is worse than a clear refusal here.
        throw std::logic_error("solve: no stage ran (internal pipeline error)");
    }
    result.flag_ = result.stages_.back().flag_;
    result.structure_key_ = hven::solvers::declaration_key(this->provider_->declaration());
    this->fill_phase_results(result);

    this->last_result_cache_ = result;
    return result;
}

const SolveResult &BackendProblemBase::last_result() const {
    if (!this->last_result_cache_.has_value()) {
        throw std::logic_error("last_result: no solve() has completed on this instance yet");
    }
    return *this->last_result_cache_;
}

tycho::ConvergenceFlags BackendProblemBase::run_adaptive_mesh(EngineRef, Mode, SolveResult &) {
    throw std::logic_error(
        "run_adaptive_mesh: unreachable on a problem type with adaptive_mesh_enabled() == "
        "false");
}

tycho::ConvergenceFlags BackendProblemBase::run_amr_loop(
    EngineRef engine, Mode mode, SolveResult &r, tycho::ConvergenceFlags mesh_abort_flag,
    int max_mesh_iters, bool solve_only_first, bool print_mesh_info, const AmrLoopHooks &hooks) {
    const SolveOptions *opts = this->active_solve_options_;
    const bool do_presolve = opts != nullptr && opts->presolve;
    EngineRef presolve_engine =
        (opts != nullptr && opts->presolve_engine != nullptr) ? *opts->presolve_engine : engine;
    const hven::solvers::WarmStartData *first_stage_warm = opts != nullptr ? opts->warm : nullptr;

    // Every engine call re-transcribes first: a mesh update
    // (update_mesh()/update_meshs()) invalidates the current transcription
    // via reset_transcription(), exactly as the old per-type
    // interior_point_call_impl's own two-statement transcription gate did on
    // every mesh-loop call -- prepare_solve() IS that gate, reused verbatim.
    auto run_one = [&](EngineRef eng, Mode m, const char *role,
                       const hven::solvers::WarmStartData *warm,
                       bool require_residual_report) -> StageOutput {
        this->prepare_solve();
        StageOutput out = run_engine_stage(eng, m, this->nlp_, this->initial_primal(), warm);
        if (require_residual_report && out.eq_cons_.size() == 0 && out.iq_cons_.size() == 0) {
            // Refused before accept_stage() runs, so a misuse path never
            // leaves the host half-mutated.
            throw std::invalid_argument(fmt::format(
                "adaptive mesh refinement requires an engine that reports constraint residuals; "
                "{0} does not report them",
                engine_name(eng)));
        }
        this->accept_stage(out);
        append_stage(r, out, role);
        return out;
    };

    if (print_mesh_info) {
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
        fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
        fmt::print("\n");
    }

    tycho::utils::Timer amr_timer;
    amr_timer.start();

    // Holds the most recent presolve stage's warm export -- reused as the
    // backing storage for both the pre-loop presolve below and each
    // per-iteration presolve inside the loop (do_presolve && !solve_only_first),
    // since only one of those exports is ever pending at a time. Part of the
    // same uniform value chain described at the top of this file: a
    // presolve's export seeds the main stage that immediately follows it.
    hven::solvers::WarmStartData presolve_warm;
    if (do_presolve) {
        // The presolve stage is never the one mesh error estimation reads
        // from (the main stage always runs after it and owns the post-opt
        // info), so it is not held to the residual-report requirement -- an
        // SQP/Ipopt presolve feeding an interior-point main stage under
        // adaptive mesh is a legal composition.
        StageOutput presolve_out =
            run_one(presolve_engine, Mode::Feasible, "presolve", first_stage_warm, false);
        presolve_warm = std::move(presolve_out.warm_);
        first_stage_warm = &presolve_warm;
    }

    StageOutput main_out = run_one(engine, mode, "main", first_stage_warm, true);
    tycho::ConvergenceFlags flag = main_out.flag_;
    r.warm_ = main_out.warm_;

    if (flag >= mesh_abort_flag) {
        if (print_mesh_info) {
            fmt::print(fmt::fg(fmt::color::red), "Mesh Iteration 0 Failed to Solve: Aborting\n");
        }
    } else {
        hooks.init_mesh();
        for (int i = 0; i < max_mesh_iters; i++) {
            if (hooks.mesh_converged(i)) {
                if (print_mesh_info) {
                    hooks.print_iteration(i);
                    fmt::print(fmt::fg(fmt::color::lime_green), "{}\n", hooks.converged_message);
                }
                break;
            } else if (i == max_mesh_iters - 1) {
                if (print_mesh_info) {
                    hooks.print_iteration(i);
                    fmt::print(fmt::fg(fmt::color::red), "{}\n", hooks.not_converged_message);
                }
                break;
            } else {
                hooks.update_mesh();
                if (print_mesh_info) {
                    hooks.print_iteration(i);
                }
            }

            // solve_only_first keeps its old name and its false-meaning: when
            // false, the presolve stage repeats on every mesh iteration
            // rather than running only before iteration 0. Each repeat's own
            // export re-seeds THIS iteration's main stage -- the same
            // predecessor-seeds-successor chain as the pre-loop presolve
            // above, not the stale export from a previous iteration.
            const hven::solvers::WarmStartData *iter_main_warm = nullptr;
            if (do_presolve && !solve_only_first) {
                StageOutput presolve_out =
                    run_one(presolve_engine, Mode::Feasible, "presolve", nullptr, false);
                presolve_warm = std::move(presolve_out.warm_);
                iter_main_warm = &presolve_warm;
            }

            main_out = run_one(engine, mode, "main", iter_main_warm, true);
            flag = main_out.flag_;
            r.warm_ = main_out.warm_;

            if (flag >= mesh_abort_flag) {
                if (print_mesh_info) {
                    fmt::print(fmt::fg(fmt::color::red),
                               "Mesh Iteration {0:} Failed to Solve: Aborting\n", i + 1);
                }
                break;
            }
        }
    }

    if (print_mesh_info) {
        amr_timer.stop();
        double tseconds = double(amr_timer.count<std::chrono::microseconds>()) / 1000000;
        fmt::print("Total Time:");
        if (tseconds > 0.5) {
            fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.4f} s\n", tseconds);
        } else {
            fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.2f} ms\n", tseconds * 1000);
        }

        fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
        fmt::print("\n");
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
    }

    return flag;
}

} // namespace tycho::solvers
