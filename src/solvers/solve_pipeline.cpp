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
// WARM SEEDING chains a stage to its predecessor, but only where the
// multipliers mean the same thing on both sides. The caller's opts.warm, when
// set, seeds only whichever stage runs FIRST (presolve if requested, else
// main) -- a one-shot value, exactly as run_engine_stage/stage_warm_start
// already document. After that, a stage is seeded by its immediate
// predecessor's own StageOutput::warm_ export WHEN BOTH RAN THE SAME MODE:
// the main stage's export seeds a polish stage that follows it, multipliers
// and all, which is the whole point of the polish stage.
//
// A Mode::Feasible presolve stage is the exception, and the reason the rule
// is stated in terms of modes at all. Its multipliers are duals of the
// feasibility measure IT minimized, not of the objective the main stage
// minimizes, and seeding an optimality stage with them measurably costs that
// stage iterations. So the presolve hands the main stage its PRIMAL only --
// which needs no payload: accept_stage() has already written the presolve's
// point onto the problem, so initial_primal() for the main stage IS the
// presolve's solution, and the engine derives its own multipliers from
// there, exactly as it does on a cold optimality run. The same holds inside
// the adaptive-mesh loop, where a presolve stage precedes mesh iteration 0's
// main stage (and, when solve_only_first is false, every iteration's). The
// adaptive-mesh override still sets SolveResult::warm_ itself from whichever
// main-mode stage ran LAST, since that is the one a polish stage after the
// loop should seed from.
//
// The CALLER's own opts.warm is held to the same two conditions (empty, or
// non-finite anywhere): such a payload costs the seeding and nothing more --
// the first stage runs cold and records why in its
// engine_notes_["warm_payload"] -- rather than raising. A usable payload
// taken under a different declaration still refuses, by name. This is what
// keeps the documented retry idiom (solve; on a non-convergent result, solve
// again with warm= that result) working in the case it exists for: a diverged
// stage's export is exactly the payload most likely to be non-finite. And it
// is held to the mode rule too: a payload a caller took from a solve whose
// deciding stage ran Mode::Feasible (SolveOptions::set_warm() records that;
// a bare WarmStartData carries no such record and is taken as given) seeds an
// optimality first stage with its primal alone, and says so in the same note.
//
// ENGINE HAND-OFF. Before any stage whose engine is a different CLASS from
// the previous stage's -- in this call or an earlier one on the same problem
// -- the pipeline restores the declared layout (require_declared_layout_for()
// + prepare_solve()). An engine may leave the NLP on a layout of its own (the
// interior-point engine's default fixed-variable treatment leaves it on a
// reduced variable space), and the SQP/Ipopt adapters refuse such a layout by
// name. Restoring it at the hand-off is what makes the crossover the API
// advertises -- solve(ipm, polish=sqp), or solve(ipm) then solve(sqp, warm=)
// -- work on a problem with a fixed variable.
//
// Main -> polish is the one hand-off that passes a payload from stage to
// stage (the presolve seam passes only a point, through the problem), and
// that payload is screened before it seeds anything: it must be non-empty AND
// finite in every block (warm_or_null() below). An empty export means the
// main stage's own engine call captured nothing to hand forward (e.g. a
// defensive internal check inside the engine skipped the capture; engines.cpp's
// fill_ipm_stage documents this as non-fatal), and a non-finite value means
// that stage genuinely diverged rather than exporting a usable point. Either
// way the polish stage runs unseeded, from the point the main stage left on
// the problem, and its annex says so -- instead of the run failing on a
// block-size mismatch or a downstream non-finite-value check.

#include "tycho/detail/solvers/nlp_backend.h"

// nlp_backend.h only forward-declares TranscribedAggregate (provider_'s
// pointee); this TU calls provider_->declaration(), which needs the
// complete type.
#include "tycho/detail/solvers_vf/transcribed_aggregate.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
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

/// @brief The two shapes a warm payload can carry that make it unusable as a
///        seed. Stated once here because two callers need them: warm_or_null()
///        turns them into "run this stage unseeded", and
///        warm_unusable_reason() turns them into the sentence the degraded
///        stage's annex carries. The reasons they are not refusals are on
///        those two functions.
bool warm_is_empty(const hven::solvers::WarmStartData &w) { return w.primal_.size() == 0; }
bool warm_has_non_finite(const hven::solvers::WarmStartData &w) {
    return !(w.primal_.allFinite() && w.eq_lmults_.allFinite() && w.iq_lmults_.allFinite() &&
             w.bound_lmults_.allFinite());
}

/// @brief `w` as a warm-start pointer to seed the next stage with, or null
///        when `w` is empty or carries a non-finite value in any block. An
///        empty `primal_` means the stage that produced `w` exported
///        nothing -- engines.cpp's `fill_ipm_stage` documents this as a
///        defensive, non-fatal outcome (e.g. the engine's own
///        internal-consistency check skipped the capture) -- and seeding an
///        empty payload unconditionally would turn that into a hard
///        `std::invalid_argument` out of the next stage's own block-size
///        check instead of the "run unseeded" outcome the pipeline intends.
///        Mirrors the size guard `fill_ipopt_stage` already applies to a
///        warm payload before using it as a starting point.
///
///        The finite check exists because a stage that ran to a genuine
///        DIVERGING result can still export a non-empty payload -- e.g. a
///        main stage that diverges under `polish=` can leave non-finite
///        multipliers in its own `StageOutput::warm_` -- and staging that
///        payload as the next stage's seed reaches
///        `InteriorPointSolver::stage_warm_start`'s own block validation,
///        which throws `std::invalid_argument` rather than reporting a
///        flag. That contradicts the pipeline's own contract (top-of-file:
///        "runs to completion regardless of an earlier stage's flag; a
///        non-convergent stage is a value ... never a thrown exception"),
///        so a payload with any non-finite entry is treated exactly like an
///        empty one: the next stage runs unseeded, from whatever primal the
///        problem itself already holds, instead of raising at the hand-off.
///
///        The caller-supplied `opts.warm` is held to the SAME two conditions,
///        through warm_unusable_reason() below: the payload a caller is most
///        likely to hand back is the one a non-convergent solve just returned
///        (the documented retry idiom), which is precisely the payload most
///        likely to be empty or non-finite. Refusing it there would make the
///        retry raise in the one case it exists for.
const hven::solvers::WarmStartData *warm_or_null(const hven::solvers::WarmStartData &w) {
    return (warm_is_empty(w) || warm_has_non_finite(w)) ? nullptr : &w;
}

/// @brief Empty when `w` can seed a stage; otherwise a sentence naming why it
///        cannot, for the annex note the degraded stage carries.
///
/// The two unusable shapes are the two warm_or_null() above screens every
/// payload for -- an empty payload (the stage it came from exported nothing)
/// and a non-finite one (that stage diverged rather than exporting a usable
/// point). Neither is a caller error, so neither refuses; each simply costs
/// the seeding.
///
/// @param subject names the payload the sentence is about, since both places
///        a payload is screened write one of these notes: the caller's own
///        `opts.warm` on the first stage, and the main stage's export on a
///        polish stage that follows it.
std::string warm_unusable_reason(const hven::solvers::WarmStartData &w, const char *subject) {
    if (warm_is_empty(w)) {
        return fmt::format("{0} was empty (the stage it came from exported no warm start), so "
                           "this stage ran cold, from the problem's own current point.",
                           subject);
    }
    if (warm_has_non_finite(w)) {
        return fmt::format("{0} carried a non-finite value (the stage it came from diverged "
                           "rather than exporting a usable point), so this stage ran cold, from "
                           "the problem's own current point.",
                           subject);
    }
    return {};
}

/// @brief What warm_unusable_reason() calls the two payloads it is asked
///        about, so the sentence names the one the reader has to look at.
const char *kCallerPayload = "the warm= payload";
const char *kPredecessorExport = "the preceding stage's own export";

/// @brief The mode the FIRST stage of a solve() call runs: a presolve stage,
///        when one was asked for, always runs Mode::Feasible; otherwise the
///        main stage runs first, in the mode the call asked for. Read by both
///        solve() and run_amr_loop(), which each seed that first stage
///        themselves.
Mode first_stage_mode(const SolveOptions &opts) {
    return opts.presolve ? Mode::Feasible : opts.mode;
}

/// @brief Whether a usable caller payload's MULTIPLIERS may seed the first
///        stage. They may not when they were produced by a feasibility-only
///        stage and the stage about to run pursues optimality: those prices
///        belong to the feasibility measure, not to the objective. The
///        payload's primal still seeds that stage (solve() passes it as the
///        starting point); only the duals are dropped.
bool caller_warm_duals_travel(const SolveOptions &opts) {
    return !(opts.warm_duals_from_feasible_stage() && first_stage_mode(opts) == Mode::Optimal);
}

/// @brief Refuses, naming both sizes, a caller payload whose primal cannot BE
///        this stage's starting point because it does not have the problem's
///        own number of variables.
///
/// Unreachable through the declared path: the declaration key is compared
/// first, and the primal variable count is one of the dimensions that key is
/// taken over -- a payload of a different width keys differently and is
/// refused there, naming both digests. It is written as a refusal anyway
/// rather than as a silent skip because the alternative is worse than either:
/// skipping the seed leaves the stage's annex saying the payload's primal was
/// the starting point when it was not.
void refuse_primal_seed_size_mismatch(Eigen::Index payload_size, Eigen::Index problem_size) {
    if (payload_size != problem_size) {
        throw std::invalid_argument(fmt::format(
            "solve: the warm-start payload's primal holds {0} entries but this problem's "
            "current point holds {1}; a payload seeding a stage's starting point must be "
            "stated over the same declared variables",
            payload_size, problem_size));
    }
}

/// @brief The sentence the first stage's annex carries when a usable caller
///        payload seeded that stage's PRIMAL but not its multipliers -- the
///        payload came from a feasibility-only stage, whose prices belong to
///        a different objective (top-of-file WARM SEEDING note).
const char *kWarmPrimalOnlyNote =
    "the warm= payload came from a feasibility solve, so this optimality stage took its primal "
    "as the starting point and derived its own multipliers: a feasibility stage's multipliers "
    "price the feasibility measure it minimized, not this stage's objective.";

/// @brief Owns one per-jet-call engine clone (clone_prototype()) and exposes
///        an EngineRef into it. jet_run() constructs one of these for the
///        prototype and, if staged, one each for opts.presolve_engine/
///        opts.polish -- so every engine a batched call names gets its own
///        independent clone rather than sharing one across pool workers.
///
///        Every InteriorPointSolver clone made this way gets two jet-
///        specific defaults applied on top of whatever clone_prototype()
///        copied from the source: `qp_threads_` pinned to 1 (the shared
///        thread pool already parallelizes across jobs; hven's own
///        per-worker MKL pin -- jet.h's MklLocalPinGuard -- does not survive
///        into the engine call, since InteriorPointSolver re-applies
///        `qp_threads_` at every solve entry) and `print_level_` forced to a
///        silent value UNLESS the source engine had already moved it off
///        the class default of 0 -- so a batched call is quiet by default
///        while still letting a caller opt into per-job console output by
///        setting `print_level` explicitly before staging.
class ClonedEngine {
  public:
    explicit ClonedEngine(EngineRef source) {
        std::visit(
            [this](auto *p) {
                using T = std::decay_t<decltype(*p)>;
                if constexpr (std::is_same_v<T, InteriorPointSolver>) {
                    auto clone = clone_prototype(*p);
                    clone->set_qp_threads(1);
                    if (clone->settings().print_level_ == 0) {
                        clone->set_print_level(10);
                    }
                    this->storage_ = std::move(clone);
                } else {
                    this->storage_ = clone_prototype(*p);
                }
            },
            source);
    }

    /// @return An EngineRef into this clone. Valid for as long as this
    ///         ClonedEngine instance is alive.
    EngineRef ref() {
        return std::visit(
            [](auto &v) -> EngineRef {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, std::unique_ptr<InteriorPointSolver>>) {
                    return EngineRef{v.get()};
                } else {
                    return EngineRef{&v};
                }
            },
            this->storage_);
    }

  private:
    std::variant<std::unique_ptr<InteriorPointSolver>, SqpSolver, IpoptSolver> storage_;
};

} // namespace

void BackendProblemBase::require_declared_layout_for(EngineRef engine) {
    const int engine_class = static_cast<int>(engine.index());
    if (this->last_stage_engine_class_ >= 0 && this->last_stage_engine_class_ != engine_class) {
        // The previous stage's engine may have left the NLP on a layout of
        // its own (a reduced variable space, or appended internal fixing
        // rows). Marking the transcription stale here means the
        // prepare_solve() that follows re-lays the declared layout, so the
        // next engine's adapter sees the problem as declared. See the
        // declaration's own comment for why this is the pipeline's job.
        this->invalidate_transcription();
    }
    this->last_stage_engine_class_ = engine_class;
}

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
    if (opts.presolve) {
        // A presolve stage runs Mode::Feasible, which only the interior-point
        // engine implements. Refused here rather than from inside the stage,
        // so the combination never gets as far as the latch or a
        // transcription.
        refuse_presolve_engine_without_feasible_mode(
            opts.presolve_engine != nullptr ? *opts.presolve_engine : engine);
    }

    // Throws std::invalid_argument, naming the reason, if any named engine is
    // already inside another solve() call. Held for the rest of this call.
    std::vector<EngineLatch> latches = latch_solve_engines(engine, opts);

    // --- Transcription / wiring ---
    // Whichever engine runs the FIRST stage decides whether the layout the
    // last stage (possibly from an earlier solve() call on this problem) left
    // behind has to be undone first.
    this->require_declared_layout_for(
        opts.presolve && opts.presolve_engine != nullptr ? *opts.presolve_engine : engine);
    this->prepare_solve();

    // --- Warm-start pre-check ---
    // Three outcomes: an unusable payload (empty, or non-finite anywhere)
    // costs the seeding and nothing else -- the first stage runs cold and
    // says so in its annex; a usable payload whose multipliers came from a
    // feasibility stage seeds this stage's primal only, and says that in the
    // same annex note; a usable payload taken under a different
    // declaration refuses, naming both digests. The emptiness/finiteness test
    // runs FIRST, so an empty payload is never diagnosed as a stamp mismatch
    // (a default-constructed payload carries a default stamp, which would
    // otherwise be reported as "taken from a different declared problem" --
    // true of the bytes, wrong about the cause).
    //
    // The stamp check duplicates the engine's own check at solve entry (which
    // runs regardless); it exists here only so the refusal surfaces before any
    // stage work does.
    SolveResult result;
    const hven::solvers::WarmStartData *first_stage_warm = nullptr;
    // Non-empty only on the primal-only path below: the caller's payload
    // seeds the first stage's starting point directly, since the problem's
    // own current point is not that payload's (a caller may hand back a
    // result taken on another instance of the same declared problem).
    Eigen::VectorXd first_stage_primal_seed;
    std::string warm_annex_note;
    if (opts.warm != nullptr) {
        warm_annex_note = warm_unusable_reason(*opts.warm, kCallerPayload);
        if (warm_annex_note.empty()) {
            const hven::solvers::DeclarationKey current_key =
                hven::solvers::declaration_key(this->provider_->declaration());
            if (!(current_key == opts.warm->structure_key_)) {
                throw std::invalid_argument(fmt::format(
                    "solve: the warm-start payload's declaration key (digest={0}) does not match "
                    "the current transcription's declaration key (digest={1}); the payload was "
                    "taken from a different declared problem",
                    opts.warm->structure_key_.digest(), current_key.digest()));
            }
            if (caller_warm_duals_travel(opts)) {
                first_stage_warm = opts.warm;
            } else {
                // The duals are dropped, the primal is not: it is handed to
                // the stage as its starting point, in place of the problem's
                // own current point.
                refuse_primal_seed_size_mismatch(opts.warm->primal_.size(),
                                                 this->initial_primal().size());
                first_stage_primal_seed = opts.warm->primal_;
                warm_annex_note = kWarmPrimalOnlyNote;
            }
        }
    }

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
        if (opts.presolve) {
            EngineRef presolve_engine =
                opts.presolve_engine != nullptr ? *opts.presolve_engine : engine;
            StageOutput out = run_engine_stage(presolve_engine, Mode::Feasible, this->nlp_,
                                               this->initial_primal(), first_stage_warm);
            this->accept_stage(out);
            append_stage(result, out, "presolve");
            // The main stage that follows is seeded by this stage's PRIMAL
            // and nothing else -- carried by the accept_stage() above, which
            // is what initial_primal() below reads. The presolve's own
            // multipliers do not travel across the mode change (top-of-file
            // WARM SEEDING note), so its export is not staged onto the main
            // stage's engine at all.
            first_stage_warm = nullptr;
        }

        this->require_declared_layout_for(engine);
        this->prepare_solve();
        StageOutput main_out = run_engine_stage(
            engine, opts.mode, this->nlp_,
            first_stage_primal_seed.size() != 0 ? first_stage_primal_seed : this->initial_primal(),
            first_stage_warm);
        this->accept_stage(main_out);
        append_stage(result, main_out, "main");
        result.warm_ = main_out.warm_;
    }

    if (opts.polish != nullptr) {
        this->require_declared_layout_for(*opts.polish);
        this->prepare_solve();
        // The one stage-to-stage hand-off that passes a payload: the main
        // stage's export seeds the polish stage, multipliers and all. When
        // that export is unusable (the main stage diverged, or captured
        // nothing), the polish stage runs cold from the point the main stage
        // left on the problem -- and says so in its own annex, under a key
        // that names the hand-off rather than the caller's payload. No engine
        // writes this key: an engine's own warm-start note is "warm_export".
        const std::string handoff_note = warm_unusable_reason(result.warm_, kPredecessorExport);
        StageOutput polish_out =
            run_engine_stage(*opts.polish, Mode::Optimal, this->nlp_, this->initial_primal(),
                             warm_or_null(result.warm_));
        this->accept_stage(polish_out);
        append_stage(result, polish_out, "polish");
        if (!handoff_note.empty()) {
            result.stages_.back().engine_notes_["warm_handoff"] = handoff_note;
        }
        result.warm_ = polish_out.warm_;
    }

    if (result.stages_.empty()) {
        // Unreachable through the paths above (the main stage always runs
        // and always appends), but the failure mode if it somehow happened
        // -- SolveResult::final_stage()/flag_ read out of an empty stage
        // list -- is worse than a clear refusal here.
        throw std::logic_error("solve: no stage ran (internal pipeline error)");
    }
    if (!warm_annex_note.empty()) {
        // The first stage is the one opts.warm would have seeded, whichever
        // role it took (presolve, main, or -- under adaptive mesh -- mesh
        // iteration 0's own first stage). The key names the caller's payload
        // specifically, and is one no engine writes: an engine's own
        // warm-start note is "warm_export", so this cannot overwrite one.
        result.stages_.front().engine_notes_["warm_payload"] = warm_annex_note;
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

ConvergenceFlags BackendProblemBase::jet_run() {
    if (!this->jet_prototype_.has_value()) {
        throw std::logic_error(
            "jet_run(): no jet job staged on this problem; call set_jet_job(prototype, opts) "
            "before Jet::map runs it");
    }

    this->jet_initialize();

    SolveResult result;
    try {
        // Clone every engine this call names -- the prototype, and (if
        // staged) the presolve/polish auxiliary engines -- into per-call
        // instances, so N pool workers sharing one prototype/presolve/polish
        // engine across a Jet::map batch never contend on any of their
        // concurrency latches. ClonedEngine additionally applies jet-
        // specific defaults (qp_threads pin, forced-silent print_level) to
        // every InteriorPointSolver clone it makes -- see its own doc
        // comment.
        ClonedEngine main_clone(*this->jet_prototype_);

        SolveOptions opts = *this->jet_solve_options_;

        std::optional<ClonedEngine> presolve_clone;
        EngineRef presolve_ref{};
        if (opts.presolve_engine != nullptr) {
            presolve_clone.emplace(*opts.presolve_engine);
            presolve_ref = presolve_clone->ref();
            opts.presolve_engine = &presolve_ref;
        }

        std::optional<ClonedEngine> polish_clone;
        EngineRef polish_ref{};
        if (opts.polish != nullptr) {
            polish_clone.emplace(*opts.polish);
            polish_ref = polish_clone->ref();
            opts.polish = &polish_ref;
        }

        result = this->solve(main_clone.ref(), opts);
    } catch (...) {
        this->jet_release();
        throw;
    }

    this->jet_release();
    return result.flag_;
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
    // Held to the same conditions the rest of the chain is: solve() records
    // in the first stage's annex what became of a caller's payload, but the
    // screening itself has to happen here too, since this loop reads
    // opts->warm directly rather than taking solve()'s already-screened
    // pointer. Unusable (empty, or non-finite anywhere) drops it; usable but
    // carrying a feasibility stage's multipliers into an optimality first
    // stage keeps only its primal, handed to that stage as its starting point
    // below.
    const hven::solvers::WarmStartData *first_stage_warm =
        (opts != nullptr && opts->warm != nullptr) ? warm_or_null(*opts->warm) : nullptr;
    Eigen::VectorXd first_stage_primal_seed;
    if (first_stage_warm != nullptr && !caller_warm_duals_travel(*opts)) {
        refuse_primal_seed_size_mismatch(first_stage_warm->primal_.size(),
                                         this->initial_primal().size());
        first_stage_primal_seed = first_stage_warm->primal_;
        first_stage_warm = nullptr;
    }

    // Every engine call re-transcribes first: a mesh update
    // (update_mesh()/update_meshs()) invalidates the current transcription
    // via reset_transcription(), exactly as the old per-type
    // interior_point_call_impl's own two-statement transcription gate did on
    // every mesh-loop call -- prepare_solve() IS that gate, reused verbatim.
    // `primal_seed`, when non-empty, replaces the problem's own current point
    // as this stage's starting point -- the primal-only path above, used by
    // the first stage and no other.
    auto run_one = [&](EngineRef eng, Mode m, const char *role,
                       const hven::solvers::WarmStartData *warm, bool require_residual_report,
                       const Eigen::VectorXd &primal_seed = Eigen::VectorXd()) -> StageOutput {
        // Marks the transcription stale when this stage's engine class
        // differs from the previous stage's, so the prepare_solve() below
        // re-lays the declared layout for it (solve()'s own hand-off rule,
        // applied to every stage the mesh loop runs).
        this->require_declared_layout_for(eng);
        this->prepare_solve();
        StageOutput out =
            run_engine_stage(eng, m, this->nlp_,
                             primal_seed.size() != 0 ? primal_seed : this->initial_primal(), warm);
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

    if (do_presolve) {
        // The presolve stage is never the one mesh error estimation reads
        // from (the main stage always runs after it and owns the post-opt
        // info), so it is not held to the residual-report requirement -- only
        // the main stage is. (Which engines may run a presolve stage at all
        // is a separate question, settled by solve()'s refusal matrix: the
        // stage runs Mode::Feasible, so the engine must implement it.)
        run_one(presolve_engine, Mode::Feasible, "presolve", first_stage_warm, false);
        // Mesh iteration 0's main stage inherits the presolve's point through
        // run_one's own accept_stage(), and nothing else: a feasibility
        // stage's multipliers do not seed an optimality stage (top-of-file
        // WARM SEEDING note).
        first_stage_warm = nullptr;
    }

    StageOutput main_out =
        run_one(engine, mode, "main", first_stage_warm, true, first_stage_primal_seed);
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
            // rather than running only before iteration 0. Each repeat hands
            // THIS iteration's main stage its point, through the same
            // write-back the pre-loop presolve uses, and none of its
            // multipliers.
            if (do_presolve && !solve_only_first) {
                run_one(presolve_engine, Mode::Feasible, "presolve", nullptr, false);
            }

            main_out = run_one(engine, mode, "main", nullptr, true);
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
