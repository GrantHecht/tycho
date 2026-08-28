// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// The engine handle layer's implementation: IpoptSolver's availability guard,
// engine_name/clone_prototype, and run_engine_stage -- the one seam that runs
// a single solver stage against a transcribed NLP against whichever engine
// EngineRef names, reporting back in the pipeline's internal currency
// (StageOutput).
//
// The SQP branch needs a model bridge: SqpDriver consumes hven's NlpModel/
// NlpModelAggregate contract, not tycho's NonLinearProgram directly. Local to
// this translation unit, SqpModelAdapter wraps a NonLinearProgram the same
// way src/solvers/ipopt_tnlp_adapter.cpp's TychoTNLP wraps one for Ipopt's
// TNLP contract: one shared KKT assembly buffer (NonLinearProgram::
// analyze_sparsity), classified once into Hessian/equality-Jacobian/
// inequality-Jacobian slot maps, gathered fresh on every evaluation. It
// differs from TychoTNLP in the shape it hands back -- separate eq/ineq
// Jacobians and an upper-triangle Hessian, matching NlpModel's contract,
// rather than one combined lower-triangle Ipopt view -- and it evaluates the
// NLP directly (no fixed-variable treatment applied): the SQP engine treats
// variable bounds as ordinary box constraints, so this adapter hands it the
// declared bounds verbatim through lower()/upper(), exactly as TychoTNLP
// hands Ipopt the same declared bounds through get_bounds_info.

#include "tycho/detail/solvers/engines.h"

// engines.h no longer includes this (see its own comment on the point): this
// translation unit needs IpoptRunInfo/IpoptSolveOutput/ipopt_backend::solve
// from it directly.
#include "tycho/detail/solvers/nlp_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include <fmt/format.h>

#include <hven/model/structure_identity.h>

namespace tycho::solvers {

namespace {

// -----------------------------------------------------------------------------
// SqpModelAdapter: NonLinearProgram, viewed as an hven::solvers::NlpModel.
// -----------------------------------------------------------------------------

/// One classified nonzero: its logical (row, col) in the destination matrix,
/// and the slot in the shared KKT buffer its value is read from.
struct AdapterCoord {
    int row_;
    int col_;
    std::size_t kkt_slot_;
};

/// Packs (row, col) into one 64-bit key for the slot lookup map below. Row/col
/// are always non-negative and well within 32 bits (declared NLP dimensions),
/// so this is a lossless, collision-free encoding.
std::uint64_t coord_key(int row, int col) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32) |
           static_cast<std::uint32_t>(col);
}

/// Builds one compressed sparse pattern from a classified coordinate list, and
/// the parallel slot array a later evaluation gathers through: slots_[i] is
/// the shared KKT buffer index whose value belongs at the i-th entry of the
/// FINAL compressed matrix's own value array (not the i-th input coordinate --
/// setFromTriplets is free to reorder, so the slot array is built by walking
/// the matrix's own compressed structure after construction, looking each
/// (row, col) back up in a map built from the input list).
struct SparsePattern {
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_;
    std::vector<std::size_t> slots_;
};

SparsePattern build_sparse_pattern(const std::vector<AdapterCoord> &coords, int rows, int cols) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(coords.size());
    std::unordered_map<std::uint64_t, std::size_t> slot_by_coord;
    slot_by_coord.reserve(coords.size() * 2);
    for (const auto &c : coords) {
        triplets.emplace_back(c.row_, c.col_, 0.0);
        slot_by_coord[coord_key(c.row_, c.col_)] = c.kkt_slot_;
    }

    SparsePattern out;
    out.mat_.resize(rows, cols);
    out.mat_.setFromTriplets(triplets.begin(), triplets.end());
    out.mat_.makeCompressed();

    out.slots_.assign(static_cast<std::size_t>(out.mat_.nonZeros()), 0);
    const auto *outer = out.mat_.outerIndexPtr();
    const auto *inner = out.mat_.innerIndexPtr();
    for (int r = 0; r < out.mat_.outerSize(); ++r) {
        for (int k = outer[r]; k < outer[r + 1]; ++k) {
            const int c = static_cast<int>(inner[k]);
            auto it = slot_by_coord.find(coord_key(r, c));
            if (it == slot_by_coord.end()) {
                throw std::logic_error(
                    "SqpModelAdapter: internal sparsity-pattern mismatch building a compressed "
                    "matrix from its own classified coordinates");
            }
            out.slots_[static_cast<std::size_t>(k)] = it->second;
        }
    }
    return out;
}

class SqpModelAdapter final : public hven::solvers::NlpModel {
  public:
    SqpModelAdapter(std::shared_ptr<NonLinearProgram> nlp, Eigen::VectorXd start_point);

    // analyze_sparsity() rebinds NLP-global state (kkt_locations_/
    // analyzed_kkt_values_/analyzed_kkt_matrix_ on the SHARED NonLinearProgram)
    // to point at this adapter's own kkt_ buffer, which is about to be
    // destroyed. The destructor restores exactly what the constructor
    // captured before rebinding it, so a caller reusing nlp_ with another
    // consumer afterward (an already-configured InteriorPointSolver that
    // skips its own set_nlp(), or a second SqpModelAdapter) sees the state it
    // would have seen had this adapter never run, rather than a pointer
    // comparison that can never match again. See the constructor's own note
    // for why restoring is chosen over bumping the structure epoch.
    ~SqpModelAdapter() override { restore_nlp_kkt_binding(); }

    Eigen::Index n() const override { return static_cast<Eigen::Index>(primal_vars_); }
    Eigen::Index me() const override { return static_cast<Eigen::Index>(equal_cons_); }
    Eigen::Index mi() const override { return static_cast<Eigen::Index>(inequal_cons_); }

    double eval_f(const Eigen::VectorXd &x) const override {
        refresh_point(x);
        return f_cache_;
    }
    Eigen::VectorXd eval_grad(const Eigen::VectorXd &x) const override {
        refresh_point(x);
        return grad_cache_;
    }
    Eigen::VectorXd eval_ce(const Eigen::VectorXd &x) const override {
        refresh_point(x);
        return ce_cache_;
    }
    Eigen::VectorXd eval_ci(const Eigen::VectorXd &x) const override {
        refresh_point(x);
        return ci_cache_;
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor>
    eval_jac_e(const Eigen::VectorXd &x) const override {
        refresh_jac(x);
        return jac_e_.mat_;
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor>
    eval_jac_i(const Eigen::VectorXd &x) const override {
        refresh_jac(x);
        return jac_i_.mat_;
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor>
    eval_hess(const Eigen::VectorXd &x, double obj_scale, const Eigen::VectorXd &lambda_e,
              const Eigen::VectorXd &lambda_i) const override;

    const Eigen::VectorXd &lower() const override { return lower_; }
    const Eigen::VectorXd &upper() const override { return upper_; }
    Eigen::VectorXd start_point() const override { return start_point_; }

  private:
    void refresh_point(const Eigen::VectorXd &x) const;
    void refresh_jac(const Eigen::VectorXd &x) const;
    void prepare_kkt_assembly() const;
    void restore_nlp_kkt_binding() const;

    std::shared_ptr<NonLinearProgram> nlp_;
    Eigen::VectorXd start_point_;
    Eigen::VectorXd lower_, upper_;
    int primal_vars_ = 0;
    int slack_vars_ = 0;
    int equal_cons_ = 0;
    int inequal_cons_ = 0;

    // NLP-global KKT-location state as it stood before this adapter's
    // constructor called analyze_sparsity() -- captured so it can be put
    // back. See restore_nlp_kkt_binding() and the destructor above.
    Eigen::VectorXi saved_kkt_locations_;
    const double *saved_analyzed_kkt_values_ = nullptr;
    Eigen::SparseMatrix<double, Eigen::RowMajor> *saved_analyzed_kkt_matrix_ = nullptr;
    bool kkt_binding_saved_ = false;

    mutable Eigen::SparseMatrix<double, Eigen::RowMajor> kkt_;
    // The values in these are refreshed from kkt_ inside const methods
    // (eval_hess/eval_jac_e/eval_jac_i), so the patterns holding them are
    // mutable -- the same rationale as kkt_ itself just above.
    mutable SparsePattern hess_;
    mutable SparsePattern jac_e_;
    mutable SparsePattern jac_i_;

    // Point-keyed cache for the value/gradient/constraint quadruple: SqpDriver
    // reads eval_grad/eval_ce/eval_ci at the same iterate in short succession,
    // and NonLinearProgram::eval_ogc computes all four in one pass, so caching
    // by point turns the common case back into one evaluation.
    mutable bool point_valid_ = false;
    mutable Eigen::VectorXd x_cache_;
    mutable double f_cache_ = 0.0;
    mutable Eigen::VectorXd grad_cache_;
    mutable Eigen::VectorXd ce_cache_;
    mutable Eigen::VectorXd ci_cache_;

    // Same idea for the Jacobian pair: one eval_soe call fills both blocks.
    mutable bool jac_valid_ = false;
    mutable Eigen::VectorXd jac_x_cache_;

    // Scratch destinations eval_ogc/eval_soe/eval_kkt always write, even when
    // this adapter has no use for the value (the Lagrangian gradient AGX in
    // particular).
    mutable Eigen::VectorXd pgx_scratch_, agx_scratch_, fxe_scratch_, fxi_scratch_;
    Eigen::VectorXd zero_le_, zero_li_;
};

SqpModelAdapter::SqpModelAdapter(std::shared_ptr<NonLinearProgram> nlp, Eigen::VectorXd start_point)
    : nlp_(std::move(nlp)), start_point_(std::move(start_point)) {
    if (!nlp_) {
        throw std::invalid_argument("SqpModelAdapter: NonLinearProgram pointer must not be null");
    }
    if (nlp_->is_reduced()) {
        throw std::invalid_argument(
            "SqpModelAdapter: this NLP was left on its reduced variable space by a prior "
            "InteriorPointSolver solve with fixed_variable_treatment=MakeParameter. Solve with "
            "the SQP engine on a freshly transcribed problem, or re-transcribe before switching "
            "engines; the SQP engine applies variable bounds directly, never through a "
            "fixed-variable treatment.");
    }
    if (nlp_->internal_fixed_constraints() > 0) {
        // A prior InteriorPointSolver solve with fixed_variable_treatment=
        // MakeConstraint installed one internal equality row per fixed
        // variable, appended at the tail of the user's own equality rows
        // (non_linear_program.h's internal-fixing-row note). Reading
        // nlp_->equal_cons_ raw, as this adapter does, would hand those rows
        // to SQP as ordinary user equalities -- redundant with the same
        // variables' bounds this adapter ALSO hands through lower()/upper()
        // (a LICQ hazard for the QP subproblem), and it would widen
        // out.eq_lmults_ past the user-row width every declared-space
        // consumer (PhaseResult, the IPM's own export) expects.
        throw std::invalid_argument(
            "SqpModelAdapter: this NLP carries internal fixing rows from a prior "
            "InteriorPointSolver solve with fixed_variable_treatment=MakeConstraint "
            "(NonLinearProgram::internal_fixed_constraints() > 0). Solve with the SQP engine "
            "on a freshly transcribed problem, or re-transcribe before switching engines; the "
            "SQP engine applies variable bounds directly and needs no internal fixing rows.");
    }

    primal_vars_ = nlp_->primal_vars_;
    slack_vars_ = nlp_->slack_vars_;
    equal_cons_ = nlp_->equal_cons_;
    inequal_cons_ = nlp_->inequal_cons_;

    if (primal_vars_ < 1) {
        throw std::invalid_argument(
            fmt::format("SqpModelAdapter: NLP has {0} primal variables; at least one is required",
                        primal_vars_));
    }
    if (start_point_.size() != primal_vars_) {
        throw std::invalid_argument(
            fmt::format("SqpModelAdapter: starting point has {0} entries, expected {1} "
                        "(primal_vars)",
                        start_point_.size(), primal_vars_));
    }
    if (nlp_->x_lower_.size() != primal_vars_ || nlp_->x_upper_.size() != primal_vars_) {
        throw std::logic_error(fmt::format(
            "SqpModelAdapter: NLP's materialized bounds have {0}/{1} entries, expected {2} "
            "(primal_vars) -- make_nlp must run before this adapter is built",
            nlp_->x_lower_.size(), nlp_->x_upper_.size(), primal_vars_));
    }
    lower_ = nlp_->x_lower_;
    upper_ = nlp_->x_upper_;

    // analyze_sparsity() is not a read: it rebinds kkt_locations_/
    // analyzed_kkt_values_/analyzed_kkt_matrix_ on the SHARED nlp_ to this
    // adapter's own kkt_ buffer (non_linear_program.cpp's analyze_sparsity),
    // without bumping structure_epoch() -- so another consumer of this same
    // nlp_ (an InteriorPointSolver that skips set_nlp(), or a second
    // SqpModelAdapter) would otherwise see stale state once this adapter is
    // destroyed and kkt_ goes with it. Capture what was there first, so the
    // destructor (and the catch below, for a throw during construction after
    // this point -- a constructor's own destructor never runs on a throw from
    // its own body) can put it back.
    //
    // Bumping the structure epoch instead -- the other contract-shaped fix --
    // is not available here: NlpAggregate::bump_structure_epoch() is
    // `protected`, reachable only from NonLinearProgram's own member
    // functions, not from an external adapter like this one. Restoring the
    // saved fields (all three are public data members) is therefore the only
    // externally reachable way to undo the rebind, and it is exact: it
    // reproduces whatever state nlp_ carried before THIS adapter touched it,
    // whatever that was.
    saved_kkt_locations_ = nlp_->kkt_locations_;
    saved_analyzed_kkt_values_ = nlp_->analyzed_kkt_values_;
    saved_analyzed_kkt_matrix_ = nlp_->analyzed_kkt_matrix_;
    kkt_binding_saved_ = true;

    // Inside the try from here on: analyze_sparsity() is the call that
    // actually performs the rebind, so a throw out of it (or out of the
    // resize that sizes its destination) is exactly the case the restore
    // below exists for -- leaving it outside the try would leave the shared
    // nlp_ half-rebound to a kkt_ buffer that dies with this half-built
    // adapter.
    try {
        kkt_.resize(nlp_->kkt_dim_, nlp_->kkt_dim_);
        nlp_->analyze_sparsity(kkt_);

        const int cons_start = primal_vars_ + slack_vars_;
        std::vector<AdapterCoord> hess_coords, jac_e_coords, jac_i_coords;

        const auto *outer = kkt_.outerIndexPtr();
        const auto *inner = kkt_.innerIndexPtr();
        for (int r = 0; r < kkt_.outerSize(); ++r) {
            for (int k = outer[r]; k < outer[r + 1]; ++k) {
                const int c = static_cast<int>(inner[k]);
                const int lo = std::min(r, c);
                const int hi = std::max(r, c);

                if (hi < primal_vars_) {
                    // Hessian of the Lagrangian, both endpoints primal. lo <=
                    // hi always, so (row=lo, col=hi) is the upper triangle
                    // NlpModel wants -- no reorientation needed, unlike
                    // TychoTNLP's lower triangle for Ipopt.
                    hess_coords.push_back({lo, hi, static_cast<std::size_t>(k)});
                } else if (lo < primal_vars_ && hi >= cons_start) {
                    // Constraint Jacobian: one primal endpoint, one
                    // constraint-row endpoint. Constraint rows run
                    // [equalities; inequalities] immediately after the slack
                    // block.
                    const int cons_idx = hi - cons_start;
                    if (cons_idx < equal_cons_) {
                        jac_e_coords.push_back({cons_idx, lo, static_cast<std::size_t>(k)});
                    } else {
                        jac_i_coords.push_back(
                            {cons_idx - equal_cons_, lo, static_cast<std::size_t>(k)});
                    }
                }
                // Everything else is solver bookkeeping (slack Jacobian,
                // slack Hessian diagonal, constraint-row pivots) with no
                // counterpart in the NlpModel contract.
            }
        }

        hess_ = build_sparse_pattern(hess_coords, primal_vars_, primal_vars_);
        jac_e_ = build_sparse_pattern(jac_e_coords, equal_cons_, primal_vars_);
        jac_i_ = build_sparse_pattern(jac_i_coords, inequal_cons_, primal_vars_);

        zero_le_ = Eigen::VectorXd::Zero(equal_cons_);
        zero_li_ = Eigen::VectorXd::Zero(inequal_cons_);
        pgx_scratch_.setZero(primal_vars_);
        agx_scratch_.setZero(primal_vars_);
        fxe_scratch_.setZero(equal_cons_);
        fxi_scratch_.setZero(inequal_cons_);
    } catch (...) {
        // A throw here unwinds out of the constructor, so this object is
        // never considered constructed and its destructor never runs -- the
        // restore has to happen here instead, on this one path, before
        // rethrowing.
        restore_nlp_kkt_binding();
        throw;
    }
}

void SqpModelAdapter::restore_nlp_kkt_binding() const {
    if (!kkt_binding_saved_ || !nlp_) {
        return;
    }
    nlp_->kkt_locations_ = saved_kkt_locations_;
    nlp_->analyzed_kkt_values_ = saved_analyzed_kkt_values_;
    nlp_->analyzed_kkt_matrix_ = saved_analyzed_kkt_matrix_;
}

void SqpModelAdapter::prepare_kkt_assembly() const {
    std::fill_n(kkt_.valuePtr(), kkt_.nonZeros(), 0.0);
    // fill_rhs accumulates (target[loc] += source[i]; non_linear_program.h),
    // it does not assign -- so these four must be zeroed on every assembly,
    // not only once at construction, or they drift unboundedly across a
    // solve's repeated eval_soe/eval_kkt calls. TychoTNLP::prepare_kkt_assembly
    // does the same four lines for the same reason.
    pgx_scratch_.setZero();
    agx_scratch_.setZero();
    fxe_scratch_.setZero();
    fxi_scratch_.setZero();
    nlp_->set_primal_diags(0.0);
    nlp_->set_e_pivots(0.0);
    nlp_->set_i_pivots(0.0);
}

void SqpModelAdapter::refresh_point(const Eigen::VectorXd &x) const {
    if (x.size() != primal_vars_) {
        throw std::invalid_argument(fmt::format(
            "SqpModelAdapter: x has {0} entries, expected {1} (n)", x.size(), primal_vars_));
    }
    if (point_valid_ && x_cache_.size() == x.size() && x_cache_ == x) {
        return;
    }
    // Cleared BEFORE the evaluation, not after: if eval_ogc throws (an
    // un-evaluable trial point is a live path), this adapter must not be left
    // looking valid at the new x with zeroed/partial caches -- a later call at
    // that same x would then return the garbage without re-evaluating.
    point_valid_ = false;
    x_cache_ = x;
    grad_cache_.setZero(primal_vars_);
    ce_cache_.setZero(equal_cons_);
    ci_cache_.setZero(inequal_cons_);
    double val = 0.0;
    // ObjScale = 1.0: eval_f/eval_grad are the UNSCALED objective and its
    // gradient by NlpModel's own contract (obj_scale is only a parameter of
    // eval_hess, for driver-side Lagrangian scaling).
    nlp_->eval_ogc(1.0, x_cache_, val, grad_cache_, ce_cache_, ci_cache_);
    f_cache_ = val;
    point_valid_ = true;
}

void SqpModelAdapter::refresh_jac(const Eigen::VectorXd &x) const {
    if (x.size() != primal_vars_) {
        throw std::invalid_argument(fmt::format(
            "SqpModelAdapter: x has {0} entries, expected {1} (n)", x.size(), primal_vars_));
    }
    if (jac_valid_ && jac_x_cache_.size() == x.size() && jac_x_cache_ == x) {
        return;
    }
    // Same cache-validity discipline as refresh_point: cleared before the
    // evaluation that can throw, set only after it returns.
    jac_valid_ = false;
    jac_x_cache_ = x;
    prepare_kkt_assembly();
    double val = 0.0;
    // eval_soe fills only the constraint Jacobian into kkt_ (no Hessian
    // block); LE/LI are unused by that pass (the Jacobian does not depend on
    // the multipliers) but must be correctly sized.
    nlp_->eval_soe(1.0, x, zero_le_, zero_li_, val, pgx_scratch_, agx_scratch_, fxe_scratch_,
                   fxi_scratch_, kkt_);

    const double *kkt_values = kkt_.valuePtr();
    double *jac_e_values = jac_e_.mat_.valuePtr();
    for (std::size_t i = 0; i < jac_e_.slots_.size(); ++i) {
        jac_e_values[i] = kkt_values[jac_e_.slots_[i]];
    }
    double *jac_i_values = jac_i_.mat_.valuePtr();
    for (std::size_t i = 0; i < jac_i_.slots_.size(); ++i) {
        jac_i_values[i] = kkt_values[jac_i_.slots_[i]];
    }
    jac_valid_ = true;
}

Eigen::SparseMatrix<double, Eigen::RowMajor>
SqpModelAdapter::eval_hess(const Eigen::VectorXd &x, double obj_scale,
                           const Eigen::VectorXd &lambda_e, const Eigen::VectorXd &lambda_i) const {
    if (x.size() != primal_vars_) {
        throw std::invalid_argument(
            fmt::format("SqpModelAdapter::eval_hess: x has {0} entries, expected {1} (n)", x.size(),
                        primal_vars_));
    }
    if (lambda_e.size() != equal_cons_) {
        throw std::invalid_argument(
            fmt::format("SqpModelAdapter::eval_hess: lambda_e has {0} entries, expected {1} (me)",
                        lambda_e.size(), equal_cons_));
    }
    if (lambda_i.size() != inequal_cons_) {
        throw std::invalid_argument(
            fmt::format("SqpModelAdapter::eval_hess: lambda_i has {0} entries, expected {1} (mi)",
                        lambda_i.size(), inequal_cons_));
    }

    prepare_kkt_assembly();
    double val = 0.0;
    // eval_kkt fills both the Hessian and the Jacobian blocks; only the
    // Hessian slots are gathered here (the Jacobian is this adapter's own
    // separate, point-cached responsibility via eval_soe).
    nlp_->eval_kkt(obj_scale, x, lambda_e, lambda_i, val, pgx_scratch_, agx_scratch_, fxe_scratch_,
                   fxi_scratch_, kkt_);

    Eigen::SparseMatrix<double, Eigen::RowMajor> hess = hess_.mat_;
    const double *kkt_values = kkt_.valuePtr();
    double *hess_values = hess.valuePtr();
    for (std::size_t i = 0; i < hess_.slots_.size(); ++i) {
        hess_values[i] = kkt_values[hess_.slots_[i]];
    }
    return hess;
}

// -----------------------------------------------------------------------------
// Per-engine stage runners.
// -----------------------------------------------------------------------------

void fill_ipm_stage(StageOutput &out, InteriorPointSolver &engine, Mode mode,
                    const std::shared_ptr<NonLinearProgram> &nlp, const Eigen::VectorXd &x0,
                    const hven::solvers::WarmStartData *warm) {
    if (!nlp) {
        throw std::invalid_argument("run_engine_stage: NonLinearProgram pointer must not be null");
    }

    engine.set_nlp(nlp);
    if (warm != nullptr) {
        engine.stage_warm_start(*warm);
    }

    if (mode == Mode::Optimal) {
        out.primal_ = engine.optimize(x0);
    } else {
        out.primal_ = engine.solve(x0);
    }

    const InteriorPointSolver::SolveResult &result = engine.result();
    out.flag_ = result.converge_flag_;
    out.eq_lmults_ = result.eq_lmults_;
    out.iq_lmults_ = result.iq_lmults_;
    out.bound_lmults_ = result.bound_lmults_;
    out.eq_cons_ = result.eq_cons_;
    out.iq_cons_ = result.iq_cons_;

    StageResult &report = out.report_;
    report.engine_name_ = "InteriorPointSolver";
    report.flag_ = result.converge_flag_;
    report.iterations_ = result.iter_num_;
    report.objective_ = result.obj_val_;
    report.wall_time_s_ = result.total_time_;

    // kkt_inf_ is the classical "KKT residual" (dual/stationarity
    // infeasibility); econ_inf_/icon_inf_ are the max-norm constraint
    // residuals StageResult's own fields are documented as. barr_inf_
    // (complementarity) has no dedicated StageResult field and rides the
    // annex instead.
    report.kkt_residual_ = result.kkt_inf_;
    report.eq_violation_ = result.econ_inf_;
    report.iq_violation_ = result.icon_inf_;
    report.engine_notes_["kkt_residual_scale"] =
        "kkt_residual_ and engine_details_[\"barr_inf\"] are on the solver's scaled objective "
        "(Settings::obj_scale_); eq_violation_/iq_violation_ are unscaled constraint residuals "
        "(SolveResult's own scale caveat).";

    report.engine_details_["barr_inf"] = result.barr_inf_;
    report.engine_details_["factor_mem"] = static_cast<double>(result.factor_mem_);
    report.engine_details_["factor_flops"] = static_cast<double>(result.factor_flops_);
    report.engine_details_["soc_steps_taken"] = static_cast<double>(result.soc_steps_taken_);
    report.engine_details_["watchdog_activations"] =
        static_cast<double>(result.watchdog_activations_);
    report.engine_details_["last_funnel_width"] = result.last_funnel_width_;
    report.engine_details_["last_filter_size"] = static_cast<double>(result.last_filter_size_);
    report.engine_details_["last_filter_resets"] = static_cast<double>(result.last_filter_resets_);
    report.engine_details_["last_monotone_switches"] =
        static_cast<double>(result.last_monotone_switches_);
    report.engine_details_["last_monotone_iters"] =
        static_cast<double>(result.last_monotone_iters_);
    report.engine_details_["last_feas_rest_entries"] =
        static_cast<double>(result.last_feas_rest_entries_);
    report.engine_details_["last_feas_rest_iters"] =
        static_cast<double>(result.last_feas_rest_iters_);
    report.engine_details_["last_prox_reg_primal"] = result.last_prox_reg_primal_;
    report.engine_details_["last_prox_reg_dual"] = result.last_prox_reg_dual_;

    if (!result.last_eval_exception_.empty()) {
        report.engine_notes_["last_eval_exception"] = result.last_eval_exception_;
    }

    const bool restoration_active =
        engine.settings().restoration_mode_ != RestorationModes::off &&
        (result.converge_flag_ == tycho::ConvergenceFlags::NOTCONVERGED ||
         result.converge_flag_ == tycho::ConvergenceFlags::DIVERGING);
    if (restoration_active) {
        report.engine_notes_["residuals"] =
            "restoration-active exit: the four terminal KKT residuals describe the restoration "
            "subproblem, not the original NLP; obj_val_ remains the true objective.";
    }

    // export_warm_start() throws std::logic_error when no solve has completed
    // on this instance -- and the engine's own capture is documented as
    // defensive but not fatal: a failed internal-consistency check SKIPS the
    // capture and clears the marker precisely so a solve can still return
    // (interior_point_solver.cpp's capture_completed_warm_start). Calling this
    // unconditionally would re-introduce exactly the failure hven avoids: a
    // defect in this side product would throw away an otherwise-good
    // StageOutput the caller was about to receive.
    try {
        out.warm_ = engine.export_warm_start();
    } catch (const std::logic_error &) {
        report.engine_notes_["warm_export"] =
            "no warm start captured: export_warm_start() reports nothing completed on this "
            "instance (either no solve returned, or the engine's own defensive "
            "internal-consistency check skipped the capture); warm_ is left at its default "
            "(empty) value.";
    }
}

void fill_sqp_stage(StageOutput &out, SqpSolver &engine, Mode mode,
                    const std::shared_ptr<NonLinearProgram> &nlp, const Eigen::VectorXd &x0,
                    const hven::solvers::WarmStartData *warm) {
    if (mode == Mode::Feasible) {
        throw std::invalid_argument(
            "the SQP engine has no feasibility-only mode; use the interior-point engine for "
            "mode=Feasible");
    }
    if (!nlp) {
        throw std::invalid_argument("run_engine_stage: NonLinearProgram pointer must not be null");
    }

    auto model = std::make_shared<SqpModelAdapter>(nlp, x0);
    hven::solvers::NlpModelAggregate bridge(model);

    SqpDriver driver(engine.options());
    if (warm != nullptr) {
        driver.stage_warm_start(*warm);
    }

    SqpSolution sol = driver.solve(bridge, x0);

    StageResult &report = out.report_;
    report.engine_name_ = SqpSolver::name();
    report.iterations_ = static_cast<int>(sol.counters.major_iters);
    report.objective_ = sol.f;
    report.wall_time_s_ = sol.wall_seconds;

    // kkt_residual_ maps to stationarity, not the driver's own combined
    // sol.kkt_residual (= max(stationarity, feasibility), sqp_types.h): the
    // IPM writes kkt_inf_ into this same field, and kkt_inf_ is documented as
    // stationarity ALONE (inf-norm dual infeasibility), not a combined
    // quantity. Writing the combined scalar here would make one field name
    // two different things depending on which engine filled it -- a later
    // comparison against a stationarity tolerance would silently read a
    // feasibility term on one engine and not the other. The combined scalar
    // is still available, under its own name.
    report.kkt_residual_ = sol.stationarity;
    report.engine_details_["kkt_residual_combined"] = sol.kkt_residual;

    // hven's cross-engine contract: NaN means UNMEASURED, never "zero
    // residual" (interior_point_solver.h's terminal-KKT-residual note, which
    // states this is the same convention SqpSolution's four columns carry).
    // SQP reports one combined feasibility scalar rather than a
    // declared-space eq/iq split, so eq_violation_/iq_violation_ have no SQP
    // measurement to report at all -- NaN, not the 0.0 default, which would
    // read as a converged-to-machine-precision residual.
    report.eq_violation_ = std::numeric_limits<double>::quiet_NaN();
    report.iq_violation_ = std::numeric_limits<double>::quiet_NaN();

    report.engine_details_["feasibility"] = sol.feasibility;
    report.engine_details_["complementarity"] = sol.complementarity;
    report.engine_notes_["residuals"] =
        "SQP reports one combined feasibility scalar (engine_details_[\"feasibility\"]), not "
        "separate eq_violation_/iq_violation_ or eq_cons_/iq_cons_ vectors -- those two fields "
        "are set to NaN (unmeasured) rather than left at their 0.0 default.";

    if (sol.counters.restoration_iters > 0) {
        report.engine_details_["restoration_iters"] =
            static_cast<double>(sol.counters.restoration_iters);
    }

    bool fill_multipliers = false;
    switch (sol.status) {
    case SqpStatus::kOptimal:
        out.flag_ = tycho::ConvergenceFlags::CONVERGED;
        fill_multipliers = true;
        break;
    case SqpStatus::kMaxIter:
        out.flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
        fill_multipliers = true;
        break;
    case SqpStatus::kInfeasible:
        out.flag_ = sol.infeasibility_certified ? tycho::ConvergenceFlags::DIVERGING
                                                : tycho::ConvergenceFlags::NOTCONVERGED;
        report.engine_notes_["multipliers"] =
            sol.infeasibility_certified
                ? "certified-infeasible exit: lambda_e/lambda_i/z are the restoration problem's "
                  "subgradient certificate, not NLP prices; left empty here."
                : "kInfeasible (not certified): the driver makes no claim; lambda_e/lambda_i/z "
                  "carry no reliable NLP prices and are left empty here.";
        break;
    case SqpStatus::kNumericalError:
        out.flag_ = tycho::ConvergenceFlags::SINGULAR_KKT;
        report.engine_notes_["multipliers"] =
            "kNumericalError exit: lambda_e/lambda_i/z are zeroed \"no evidence\"; left empty "
            "here.";
        break;
    case SqpStatus::kBudgetExhausted:
        out.flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
        report.engine_details_["budget_exhausted"] = 1.0;
        fill_multipliers = true;
        break;
    default:
        throw std::logic_error("run_engine_stage: unrecognized SqpStatus");
    }
    report.flag_ = out.flag_;

    out.primal_ = sol.x;
    if (fill_multipliers) {
        out.eq_lmults_ = sol.lambda_e;
        out.iq_lmults_ = sol.lambda_i;
        out.bound_lmults_ = sol.z;
        // Same defensive-but-not-fatal capture contract as the IPM's
        // export_warm_start() (sqp_driver.h): a failed internal-consistency
        // check skips the capture rather than throwing through a converged
        // solve, but only up to the point where export_warm_start() itself
        // still refuses "nothing to export" via std::logic_error. Guard it
        // here so that refusal cannot destroy an otherwise-good StageOutput.
        try {
            out.warm_ = driver.export_warm_start();
        } catch (const std::logic_error &) {
            report.engine_notes_["warm_export"] =
                "no warm start captured: export_warm_start() reports nothing completed on "
                "this instance; warm_ is left at its default (empty) value.";
        }
    }
}

void fill_ipopt_stage(StageOutput &out, IpoptSolver &engine, Mode mode,
                      const std::shared_ptr<NonLinearProgram> &nlp, const Eigen::VectorXd &x0,
                      const hven::solvers::WarmStartData *warm) {
    if (mode == Mode::Feasible) {
        // ipopt_backend::solve ignores its mode parameter outright -- every
        // mode runs the one NLP solve -- so silently mapping Mode::Feasible
        // onto it would hand back a full optimality solve labelled as a
        // feasibility stage, with nothing in flag_/the annex/the notes saying
        // so. Refuse by name instead, the same shape as the SQP engine's
        // refusal, checked first, before nlp/x0 are touched.
        throw std::invalid_argument(
            "the Ipopt backend has no feasibility-only mode; use the interior-point engine "
            "for mode=Feasible");
    }
    if (!nlp) {
        throw std::invalid_argument("run_engine_stage: NonLinearProgram pointer must not be null");
    }

    // Core-only warm: the Ipopt path has no staging surface of its own, so a
    // warm currency's primal block seeds the starting point directly, exactly
    // as an explicit x0 would. Sizes are trusted here; a mismatch surfaces as
    // TychoTNLP's own starting-point-size refusal inside ipopt_backend::solve.
    Eigen::VectorXd start =
        (warm != nullptr && warm->primal_.size() == x0.size()) ? warm->primal_ : x0;

    // The IpoptSolver engine handle carries only string options, no
    // termination tolerances of its own (unlike InteriorPointSolver), so a
    // default-constructed InteriorPointSolver::Settings supplies the
    // matched-tolerance baseline ipopt_backend::solve applies before
    // engine.options() (which can override it).
    IpoptSolveOutput result =
        ipopt_backend::solve(nlp, start, engine.options(), InteriorPointSolver::Settings{});

    out.flag_ = result.flag_;
    out.primal_ = result.variables_;
    out.eq_lmults_ = result.eq_lmults_;
    out.iq_lmults_ = result.iq_lmults_;
    // bound_lmults_, eq_cons_, iq_cons_ intentionally left empty: this adapter
    // does not surface bound multipliers or residual vectors from Ipopt.

    const IpoptRunInfo &info = result.info_;
    StageResult &report = out.report_;
    report.engine_name_ = IpoptSolver::name();
    report.flag_ = result.flag_;
    report.iterations_ = info.iterations_;
    report.objective_ = info.objective_;
    report.wall_time_s_ = info.wall_time_s_;
    // hven's cross-engine contract: NaN means UNMEASURED, never "zero
    // residual" (interior_point_solver.h's terminal-KKT-residual note). This
    // adapter has no stationarity measurement at all, and only one combined
    // constraint_violation rather than a declared-space eq/iq split, so all
    // three ride NaN rather than the StageResult default of 0.0, which would
    // read as a converged-to-machine-precision residual.
    report.kkt_residual_ = std::numeric_limits<double>::quiet_NaN();
    report.eq_violation_ = std::numeric_limits<double>::quiet_NaN();
    report.iq_violation_ = std::numeric_limits<double>::quiet_NaN();
    report.engine_details_["constraint_violation"] = info.constraint_violation_;
    report.engine_notes_["status"] = info.status_;
    report.engine_notes_["normalized"] = info.normalized_;
    report.engine_notes_["residuals"] =
        "Ipopt reports one combined constraint_violation (engine_details_) and no "
        "stationarity measurement at all, so kkt_residual_/eq_violation_/iq_violation_ are "
        "set to NaN (unmeasured) rather than left at their 0.0 default; eq_cons_/iq_cons_ and "
        "bound multipliers are also unavailable and left empty.";
    report.engine_notes_["tolerance_baseline"] =
        "Ipopt's matched-tolerance baseline (tol, constr_viol_tol, acceptable_tol, "
        "acceptable_constr_viol_tol, max_iter) is read from a default-constructed "
        "InteriorPointSolver::Settings, since IpoptSolver carries no tolerance settings of its "
        "own; override via engine.options().";

    // No polish extension: the Ipopt backend produces no such value. The core
    // is filled from what this adapter has -- primal/eq/iq multipliers -- and
    // bound_lmults_ is zero-filled at declared width (not a measured value:
    // this adapter surfaces no bound duals) so the block still validates at a
    // later staging call.
    out.warm_.primal_ = result.variables_;
    out.warm_.eq_lmults_ = result.eq_lmults_;
    out.warm_.iq_lmults_ = result.iq_lmults_;
    out.warm_.bound_lmults_ = Eigen::VectorXd::Zero(nlp->primal_vars_);
    out.warm_.structure_key_ = hven::solvers::declaration_key(nlp->declaration());
    report.engine_notes_["warm_export"] =
        "Ipopt's warm export carries no bound multipliers; bound_lmults_ is zero-filled at "
        "declared width rather than a measured value, and no polish extension is attached.";
}

} // namespace

IpoptSolver::IpoptSolver() {
    if (!ipopt_backend::available()) {
        throw std::runtime_error(
            "IpoptSolver: this build was not configured with Ipopt support; configure with "
            "-DENABLE_IPOPT=ON (requires an installed Ipopt discoverable via pkg-config) to "
            "construct this engine handle");
    }
}

const char *engine_name(EngineRef e) {
    if (std::holds_alternative<InteriorPointSolver *>(e)) {
        return "InteriorPointSolver";
    }
    if (std::holds_alternative<SqpSolver *>(e)) {
        return SqpSolver::name();
    }
    return IpoptSolver::name();
}

StageOutput run_engine_stage(EngineRef engine, Mode mode,
                             const std::shared_ptr<NonLinearProgram> &nlp,
                             const Eigen::VectorXd &x0, const hven::solvers::WarmStartData *warm) {
    StageOutput out;
    std::visit(
        [&](auto *e) {
            using EngineType = std::decay_t<decltype(*e)>;
            if constexpr (std::is_same_v<EngineType, InteriorPointSolver>) {
                fill_ipm_stage(out, *e, mode, nlp, x0, warm);
            } else if constexpr (std::is_same_v<EngineType, SqpSolver>) {
                fill_sqp_stage(out, *e, mode, nlp, x0, warm);
            } else {
                static_assert(std::is_same_v<EngineType, IpoptSolver>);
                fill_ipopt_stage(out, *e, mode, nlp, x0, warm);
            }
        },
        engine);
    return out;
}

std::unique_ptr<InteriorPointSolver> clone_prototype(const InteriorPointSolver &e) {
    auto clone = std::make_unique<InteriorPointSolver>();
    clone->settings() = e.settings();
    return clone;
}

SqpSolver clone_prototype(const SqpSolver &e) { return SqpSolver(e.options()); }

IpoptSolver clone_prototype(const IpoptSolver &e) {
    IpoptSolver clone;
    clone.options() = e.options();
    return clone;
}

} // namespace tycho::solvers
