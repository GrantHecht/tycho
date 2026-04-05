// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#include "tycho/detail/solvers/psiopt.h"

#include "tycho/detail/solvers/solver_init.h"
#include "tycho/detail/utils/timer.h"

#ifndef USE_ACCELERATE_SPARSE
#include <mkl.h>
#endif

// =============================================================================
// Static string-to-enum converters
// =============================================================================

auto tycho::solvers::PSIOPT::strto_OrderingMode(const std::string &str) -> QPOrderingModes {
    if (str.compare("MINDEG") == 0)
        return QPOrderingModes::MINDEG;
    else if (str.compare("METIS") == 0)
        return QPOrderingModes::METIS;
    else if (str.compare("PARMETIS") == 0 || str.compare("MTMETIS") == 0)
        return QPOrderingModes::PARMETIS;
    else {
        auto msg =
            fmt::format("Unrecognized QPOrderingMode: {0}\n"
                        "Valid Options Are: MINDEG, METIS, PARMETIS (or MTMETIS on Accelerate)",
                        str);
        throw std::invalid_argument(msg);
    }
}

auto tycho::solvers::PSIOPT::strto_LineSearchMode(const std::string &str) -> LineSearchModes {
    if (str.compare("L1") == 0)
        return LineSearchModes::L1;
    else if (str.compare("NOLS") == 0)
        return LineSearchModes::NOLS;
    else if (str.compare("LANG") == 0)
        return LineSearchModes::LANG;
    else if (str.compare("AUGLANG") == 0)
        return LineSearchModes::AUGLANG;
    else {
        auto msg = fmt::format("Unrecognized LineSearchMode: {0}\n"
                               "Valid Options Are: AUGLANG, LANG, L1, NOLS ",
                               str);
        throw std::invalid_argument(msg);
    }
}

auto tycho::solvers::PSIOPT::strto_BarrierMode(const std::string &str) -> BarrierModes {
    if (str.compare("PROBE") == 0)
        return BarrierModes::PROBE;
    else if (str.compare("LOQO") == 0)
        return BarrierModes::LOQO;
    else {
        auto msg = fmt::format("Unrecognized BarrierMode: {0}\n"
                               "Valid Options Are: LOQO, PROBE ",
                               str);
        throw std::invalid_argument(msg);
    }
}

auto tycho::solvers::PSIOPT::strto_BestCriteriaMode(const std::string &str) -> BestCriteriaModes {
    if (str == "ECons" || str == "ECon")
        return BestCriteriaModes::ECONS;
    else if (str == "ICons" || str == "ICon")
        return BestCriteriaModes::ICONS;
    else if (str == "KKT")
        return BestCriteriaModes::KKT;
    else if (str == "Obj" || str == "Prim Obj")
        return BestCriteriaModes::OBJ;
    else {
        throw std::invalid_argument(fmt::format("Unrecognized BestCriteriaMode: {0}", str));
    }
}

// =============================================================================
// Validated setter methods
// =============================================================================

void tycho::solvers::PSIOPT::set_max_iters(int max_iters) {
    if (max_iters < 1) {
        throw std::invalid_argument("max_iters must be greater than 0.");
    }
    settings_.max_iters_ = max_iters;
}

void tycho::solvers::PSIOPT::set_max_acc_iters(int max_acc_iters) {
    if (max_acc_iters < 1) {
        throw std::invalid_argument("max_acc_iters must be greater than 0.");
    }
    settings_.max_acc_iters_ = max_acc_iters;
}

void tycho::solvers::PSIOPT::set_max_ls_iters(int max_ls_iters) {
    if (max_ls_iters < 0) {
        throw std::invalid_argument("max_ls_iters must be non-negative (>= 0).");
    }
    settings_.max_ls_iters_ = max_ls_iters;
}

void tycho::solvers::PSIOPT::set_all_max_iters(int m1, int m2) {
    set_max_iters(m1);
    set_max_acc_iters(m2);
}

void tycho::solvers::PSIOPT::set_kkt_tol(double kkt_tol) {
    settings_.kkt_tol_ = std::abs(kkt_tol);
}

void tycho::solvers::PSIOPT::set_bar_tol(double bar_tol) {
    settings_.bar_tol_ = std::abs(bar_tol);
}

void tycho::solvers::PSIOPT::set_econ_tol(double econ_tol) {
    settings_.econ_tol_ = std::abs(econ_tol);
}

void tycho::solvers::PSIOPT::set_icon_tol(double icon_tol) {
    settings_.icon_tol_ = std::abs(icon_tol);
}

void tycho::solvers::PSIOPT::set_tols(double kkt_tol, double econ_tol, double icon_tol,
                                      double bar_tol) {
    this->set_kkt_tol(kkt_tol);
    this->set_econ_tol(econ_tol);
    this->set_icon_tol(icon_tol);
    this->set_bar_tol(bar_tol);
}

void tycho::solvers::PSIOPT::set_acc_kkt_tol(double acc_kkt_tol) {
    settings_.acc_kkt_tol_ = std::abs(acc_kkt_tol);
}

void tycho::solvers::PSIOPT::set_acc_bar_tol(double acc_bar_tol) {
    settings_.acc_bar_tol_ = std::abs(acc_bar_tol);
}

void tycho::solvers::PSIOPT::set_acc_econ_tol(double acc_econ_tol) {
    settings_.acc_econ_tol_ = std::abs(acc_econ_tol);
}

void tycho::solvers::PSIOPT::set_acc_icon_tol(double acc_icon_tol) {
    settings_.acc_icon_tol_ = std::abs(acc_icon_tol);
}

void tycho::solvers::PSIOPT::set_acc_tols(double acc_kkt_tol, double acc_econ_tol,
                                          double acc_icon_tol, double acc_bar_tol) {
    this->set_acc_kkt_tol(acc_kkt_tol);
    this->set_acc_econ_tol(acc_econ_tol);
    this->set_acc_icon_tol(acc_icon_tol);
    this->set_acc_bar_tol(acc_bar_tol);
}

void tycho::solvers::PSIOPT::set_unacc_tols(double kktol, double etol, double itol,
                                             double bartol) {
    settings_.unacc_kkt_tol_ = kktol;
    settings_.unacc_bar_tol_ = bartol;
    settings_.unacc_econ_tol_ = etol;
    settings_.unacc_icon_tol_ = itol;
}

void tycho::solvers::PSIOPT::set_div_kkt_tol(double div_kkt_tol) {
    settings_.div_kkt_tol_ = std::abs(div_kkt_tol);
}

void tycho::solvers::PSIOPT::set_div_bar_tol(double div_bar_tol) {
    settings_.div_bar_tol_ = std::abs(div_bar_tol);
}

void tycho::solvers::PSIOPT::set_div_econ_tol(double div_econ_tol) {
    settings_.div_econ_tol_ = std::abs(div_econ_tol);
}

void tycho::solvers::PSIOPT::set_div_icon_tol(double div_icon_tol) {
    settings_.div_icon_tol_ = std::abs(div_icon_tol);
}

void tycho::solvers::PSIOPT::set_div_tols(double div_kkt_tol, double div_econ_tol,
                                          double div_icon_tol, double div_bar_tol) {
    this->set_div_kkt_tol(div_kkt_tol);
    this->set_div_econ_tol(div_econ_tol);
    this->set_div_icon_tol(div_icon_tol);
    this->set_div_bar_tol(div_bar_tol);
}

void tycho::solvers::PSIOPT::set_bound_fraction(double bound_fraction) {
    if (bound_fraction >= 1.0 || bound_fraction <= 0.0) {
        throw std::invalid_argument("bound_fraction must be between 0 and 1.");
    }
    settings_.bound_fraction_ = bound_fraction;
}

void tycho::solvers::PSIOPT::set_bound_push(double bound_push) {
    if (bound_push <= 0.0) {
        throw std::invalid_argument("bound_push must be greater than 0.");
    }
    settings_.bound_push_ = bound_push;
}

void tycho::solvers::PSIOPT::set_alpha_red(double ared) {
    if (ared <= 1.0) {
        throw std::invalid_argument("alpha_red must be greater than 1.0");
    }
    settings_.alpha_red_ = ared;
}

void tycho::solvers::PSIOPT::set_delta_h(double delta_h) {
    if (delta_h <= 0.0) {
        throw std::invalid_argument("delta_h must be greater than 0.");
    }
    settings_.delta_h_ = delta_h;
}

void tycho::solvers::PSIOPT::set_incr_h(double incr_h) {
    if (incr_h <= 1.0) {
        throw std::invalid_argument("incr_h must be greater than 1.0.");
    }
    settings_.incr_h_ = incr_h;
}

void tycho::solvers::PSIOPT::set_decr_h(double decr_h) {
    if (decr_h >= 1.0 || decr_h <= 0) {
        throw std::invalid_argument("decr_h must be between 0 and 1.");
    }
    settings_.decr_h_ = decr_h;
}

void tycho::solvers::PSIOPT::set_hpert_params(double delta_h, double incr_h, double decr_h) {
    this->set_delta_h(delta_h);
    this->set_incr_h(incr_h);
    this->set_decr_h(decr_h);
}

void tycho::solvers::PSIOPT::set_print_level(int plevel) { settings_.print_level_ = plevel; }

void tycho::solvers::PSIOPT::set_qp_ordering_mode(QPOrderingModes mode) {
    settings_.qp_ord_ = mode;
}

void tycho::solvers::PSIOPT::set_qp_ordering_mode(const std::string &str) {
    settings_.qp_ord_ = strto_OrderingMode(str);
}

void tycho::solvers::PSIOPT::set_opt_bar_mode(BarrierModes mode) {
    settings_.opt_bar_mode_ = mode;
}

void tycho::solvers::PSIOPT::set_opt_bar_mode(const std::string &str) {
    settings_.opt_bar_mode_ = strto_BarrierMode(str);
}

void tycho::solvers::PSIOPT::set_soe_bar_mode(BarrierModes mode) {
    settings_.soe_bar_mode_ = mode;
}

void tycho::solvers::PSIOPT::set_soe_bar_mode(const std::string &str) {
    settings_.soe_bar_mode_ = strto_BarrierMode(str);
}

void tycho::solvers::PSIOPT::set_opt_ls_mode(LineSearchModes mode) {
    settings_.opt_ls_mode_ = mode;
}

void tycho::solvers::PSIOPT::set_opt_ls_mode(const std::string &str) {
    settings_.opt_ls_mode_ = strto_LineSearchMode(str);
}

void tycho::solvers::PSIOPT::set_soe_ls_mode(LineSearchModes mode) {
    settings_.soe_ls_mode_ = mode;
}

void tycho::solvers::PSIOPT::set_soe_ls_mode(const std::string &str) {
    settings_.soe_ls_mode_ = strto_LineSearchMode(str);
}

void tycho::solvers::PSIOPT::set_best_criteria(BestCriteriaModes mode) {
    settings_.best_criteria_ = mode;
}

void tycho::solvers::PSIOPT::set_best_criteria(const std::string &str) {
    settings_.best_criteria_ = strto_BestCriteriaMode(str);
}

#ifdef USE_ACCELERATE_SPARSE
void tycho::solvers::PSIOPT::set_accel_pivot_tolerance(double tol) {
    settings_.accel_pivot_tolerance_ = tol;
}

void tycho::solvers::PSIOPT::set_accel_zero_tolerance(double tol) {
    settings_.accel_zero_tolerance_ = tol;
}
#endif

// =============================================================================
// QP parameter setup
// =============================================================================

void tycho::solvers::PSIOPT::set_qp_params() {
#ifdef USE_ACCELERATE_SPARSE
    // Accelerate interface uses different configuration methods
    switch (settings_.qp_ord_) {
    case QPOrderingModes::MINDEG:
        this->kkt_sol_.set_order(SparseOrderAMD);
        break;
    case QPOrderingModes::METIS:
        // Serial METIS: faster than MT-METIS at all tested scales (up to
        // ~5400 primal variables) due to per-call thread coordination overhead.
        this->kkt_sol_.set_order(SparseOrderMetis);
        break;
    case QPOrderingModes::PARMETIS:
        // MT-METIS (macOS 26+): currently slower than serial METIS at tested
        // scales. Retained for tracking Apple's improvements across releases.
#ifdef TYCHO_HAS_MTMETIS
        this->kkt_sol_.set_order(SparseOrderMTMetis);
#else
        this->kkt_sol_.set_order(SparseOrderMetis);
#endif
        break;
    }
    this->kkt_sol_.set_num_threads(settings_.qp_threads_);
    this->kkt_sol_.set_iterative_refinement(settings_.qp_ref_steps_ > 0);
    this->kkt_sol_.set_iterative_refinement_iterations(settings_.qp_ref_steps_);
    this->kkt_sol_.set_pivot_tolerance(settings_.accel_pivot_tolerance_);
    this->kkt_sol_.set_zero_tolerance(settings_.accel_zero_tolerance_);
#else
    this->kkt_sol_.ord_ = static_cast<int>(settings_.qp_ord_);
    this->kkt_sol_.pivotstrat_ = static_cast<int>(settings_.qp_pivot_strategy_);
    this->kkt_sol_.pivotpert_ = settings_.qp_pivot_perturb_;
    this->kkt_sol_.matching_ = settings_.qp_matching_;
    this->kkt_sol_.scaling_ = settings_.qp_scaling_;
    this->kkt_sol_.iterref_ = settings_.qp_ref_steps_;
    this->kkt_sol_.alg_ = static_cast<int>(settings_.qp_alg_);
    this->kkt_sol_.msglvl_ = settings_.qp_print_;

    if (settings_.cnr_mode_)
        this->kkt_sol_.threads_ = settings_.qp_threads_;
    this->kkt_sol_.parsolve_ = settings_.qp_par_solve_;
    this->kkt_sol_.set_params();
#endif
}

// =============================================================================
// KKT matrix analysis
// =============================================================================

bool tycho::solvers::PSIOPT::analyze_kkt_matrix() {
    bool docompute = true;
    if (this->qp_analyzed_ && !(settings_.force_qp_analysis_)) {
        docompute = false;
    } else {
        this->qp_analyzed_ = true;
        docompute = true;
    }
    return docompute;
}

// =============================================================================
// Release
// =============================================================================

void tycho::solvers::PSIOPT::release() {
    this->kkt_sol_.release();
    this->qp_analyzed_ = false;
    this->nlp_ = std::shared_ptr<NonLinearProgram>();
    result_.eq_lmults_.resize(0);
    result_.iq_lmults_.resize(0);
}

// =============================================================================
// Barrier math helpers
// =============================================================================

void tycho::solvers::PSIOPT::apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S,
                                                 Eigen::Ref<Eigen::VectorXd> FXI) const {
    for (int i = 0; i < this->slack_vars_; i++) {
        double fxi = FXI[i];
        double si = S[i];
        if (si < settings_.neg_slack_reset_) {
            si = settings_.neg_slack_reset_;
        }

        if (fxi < 0.0) {
            FXI[i] = 0.0;
            S[i] = std::max(std::abs(fxi), settings_.neg_slack_reset_);
        } else {
            FXI[i] += si;
        }
    }
}

double tycho::solvers::PSIOPT::max_step_to_boundary(Eigen::Ref<Eigen::VectorXd> SLI,
                                                     Eigen::Ref<Eigen::VectorXd> dSLI,
                                                     double bfrac) const {
    double alpha = 1.0;
    for (int i = 0; i < this->inequal_cons_; i++) {
        if (dSLI[i] < -bfrac * SLI[i]) {
            double an = -bfrac * SLI[i] / dSLI[i];
            if (an < alpha)
                alpha = an;
        }
    }
    return alpha;
}

void tycho::solvers::PSIOPT::complementarity(Eigen::Ref<Eigen::VectorXd> S,
                                              Eigen::Ref<Eigen::VectorXd> LI, double &avgcomp,
                                              double &mincomp, double &maxcomp) const {
    Eigen::VectorXd StLI = S.cwiseProduct(LI);
    mincomp = StLI.minCoeff();
    maxcomp = StLI.maxCoeff();
    avgcomp = StLI.sum() / double(StLI.size());
}

double tycho::solvers::PSIOPT::barrier_objective(Eigen::Ref<Eigen::VectorXd> S,
                                                  double mu) const {
    double psi = 0;
    for (int i = 0; i < this->inequal_cons_; i++) {
        psi += -mu * std::log(S[i]);
    }
    return psi;
}

void tycho::solvers::PSIOPT::barrier_gradient(Eigen::Ref<Eigen::VectorXd> S,
                                               Eigen::Ref<Eigen::VectorXd> LI, double mu,
                                               Eigen::Ref<Eigen::VectorXd> AGS) const {
    AGS = LI - mu * (S.cwiseInverse());
}

void tycho::solvers::PSIOPT::barrier_gradient(Eigen::Ref<Eigen::VectorXd> LI,
                                               Eigen::Ref<Eigen::VectorXd> AGS) const {
    AGS = LI;
}

void tycho::solvers::PSIOPT::barrier_hessian(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                              Eigen::Ref<Eigen::VectorXd> S,
                                              Eigen::Ref<Eigen::VectorXd> LI, double mu) {
    Eigen::VectorXd hp = LI.cwiseQuotient(S);
    for (int i = 0; i < this->inequal_cons_; i++) {
        if (hp[i] < 0.0) {
            hp[i] = mu / (S[i] * S[i]);
        }
    }
    this->nlp_->assign_kkt_slack_hessian(hp, KKTmat);
}

double tycho::solvers::PSIOPT::loqo_mu(Eigen::Ref<Eigen::VectorXd> S,
                                        Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                                        double mincomp) const {
    double eta = mincomp / avgcomp;
    double sigmat = .1 * std::pow(0.05 * (1.0 - eta) / eta, 3);
    double sigma = std::min(0.8, abs(sigmat));
    return sigma * avgcomp;
}

double tycho::solvers::PSIOPT::mpc_mu(Eigen::Ref<Eigen::VectorXd> S,
                                       Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                                       double mincomp) const {
    double navgcomp = 0;
    double nmincomp = 0;
    double nmaxcomp = 0;
    this->complementarity(S, LI, navgcomp, nmincomp, nmaxcomp);
    return std::pow(navgcomp / avgcomp, 3) * avgcomp;
}

// =============================================================================
// NLP eval dispatch methods
// =============================================================================

void tycho::solvers::PSIOPT::eval_kkt(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                       EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                       Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_kkt(
        obj_scale, XSL.head(primal_vars_),
        XSL.segment(primal_vars_ + slack_vars_, equal_cons_), XSL.tail(inequal_cons_), val,
        GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_),
        AGXS_FX.tail(inequal_cons_), KKTmat);
}

void tycho::solvers::PSIOPT::eval_kkt_no(double obj_scale, ConstEigenRef<VectorXd> XSL,
                                          double &val, EigenRef<VectorXd> GX,
                                          EigenRef<VectorXd> AGXS_FX,
                                          Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_kkt_no(
        obj_scale, XSL.head(primal_vars_),
        XSL.segment(primal_vars_ + slack_vars_, equal_cons_), XSL.tail(inequal_cons_), val,
        GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_),
        AGXS_FX.tail(inequal_cons_), KKTmat);
}

void tycho::solvers::PSIOPT::eval_aug(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                       EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                       Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_aug(
        obj_scale, XSL.head(primal_vars_),
        XSL.segment(primal_vars_ + slack_vars_, equal_cons_), XSL.tail(inequal_cons_), val,
        GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_),
        AGXS_FX.tail(inequal_cons_), KKTmat);
}

void tycho::solvers::PSIOPT::eval_soe(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                       EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                       Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_soe(
        obj_scale, XSL.head(primal_vars_),
        XSL.segment(primal_vars_ + slack_vars_, equal_cons_), XSL.tail(inequal_cons_), val,
        GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_),
        AGXS_FX.tail(inequal_cons_), KKTmat);
}

void tycho::solvers::PSIOPT::eval_rhs(double obj_scale,
                                       const Eigen::Ref<const Eigen::VectorXd> &XSL, double &val,
                                       Eigen::Ref<Eigen::VectorXd> GX,
                                       Eigen::Ref<Eigen::VectorXd> AGXS_FX) {
    this->nlp_->eval_rhs(
        obj_scale, XSL.head(primal_vars_),
        XSL.segment(primal_vars_ + slack_vars_, equal_cons_), XSL.tail(inequal_cons_), val,
        GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_),
        AGXS_FX.tail(inequal_cons_));
}

// =============================================================================
// Existing implementations
// =============================================================================

void tycho::solvers::PSIOPT::ensure_solver_initialized() {
    double initMs = ::tycho::solvers::ensure_solver_initialized();
    if (initMs > 0.0) {
        this->result_.solver_init_time_ = initMs / 1000.0;
        // Suppress the init line when init was trivially fast (< 0.5 ms),
        // which also covers subsequent calls (return 0.0).
        constexpr double kSolverInitPrintThresholdMs = 0.5;
        if (initMs > kSolverInitPrintThresholdMs && settings_.print_level_ < 2) {
            fmt::print(" Solver Initialization : ");
            fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", initMs);
        }
    }
}

void tycho::solvers::PSIOPT::set_nlp(std::shared_ptr<NonLinearProgram> np) {
    this->nlp_ = np;
    this->primal_vars_ = this->nlp_->primal_vars_;
    this->equal_cons_ = this->nlp_->equal_cons_;
    this->inequal_cons_ = this->nlp_->inequal_cons_;
    this->slack_vars_ = this->nlp_->slack_vars_;
    this->kkt_dim_ = this->nlp_->kkt_dim_;
    this->set_qp_params();
#ifdef USE_ACCELERATE_SPARSE
    accelerate_set_num_threads(settings_.qp_threads_);
#else
    mkl_set_num_threads(settings_.qp_threads_);
#endif

    this->nlp_->analyze_sparsity(this->kkt_sol_.get_matrix());
#ifdef USE_ACCELERATE_SPARSE
    // we need to call this to update the internal AccelSparseMatrix since
    // we changed the sparsity pattern via the reference returned from get_matrix.
    this->kkt_sol_.reinitialize_internal_matrix_representation();
#endif
    this->qp_analyzed_ = false;
}

void tycho::solvers::PSIOPT::max_primal_dual_step(KKTVector &xsl, KKTVector &dxsl, double bfrac,
                                                  double &alphap, double &alphad) {
    double Smax = this->max_step_to_boundary(xsl.slacks(), dxsl.slacks(), bfrac);
    double Lmax = this->max_step_to_boundary(xsl.iq_lmults(), dxsl.iq_lmults(), bfrac);

    double primstep = Smax;
    double slackstep = Smax;
    double eqmultstep = Smax;
    double iqmultstep = Lmax;

    if (settings_.pd_step_strategy_ == PDStepStrategies::PrimSlackEq_Iq) {
    } else if (settings_.pd_step_strategy_ == PDStepStrategies::AllMinimum) {
        double step = std::min(Smax, Lmax);
        primstep = step;
        slackstep = step;
        eqmultstep = step;
        iqmultstep = step;
    } else if (settings_.pd_step_strategy_ == PDStepStrategies::PrimSlack_EqIq) {
        eqmultstep = Lmax;
    } else if (settings_.pd_step_strategy_ == PDStepStrategies::MaxEq) {
        double step = std::max(Smax, Lmax);
        eqmultstep = step;
    }
    dxsl.primals() *= primstep;
    if (inequal_cons_ > 0)
        dxsl.slacks() *= slackstep;
    if (equal_cons_ > 0)
        dxsl.eq_lmults() *= eqmultstep;
    if (inequal_cons_ > 0)
        dxsl.iq_lmults() *= iqmultstep;

    alphap = Smax;
    alphad = Lmax;
}

void tycho::solvers::PSIOPT::fill_iter_info(KKTVector &xsl, KKTVector &rhs, double pobj,
                                            double bobj, double mu, IterateInfo &iter) const {

    iter.prim_obj_ = pobj;
    iter.barr_obj_ = bobj;
    iter.mu_ = mu;
    iter.p_pivots_ = this->kkt_sol_.ppivs();
    iter.kkt_inf_ = rhs.prim_grad().lpNorm<Eigen::Infinity>();

    double avgcomp = 0;
    double mincomp = 0;
    double maxcomp = 0;
    if (inequal_cons_ > 0) {
        iter.icon_inf_ = rhs.iq_cons().lpNorm<Eigen::Infinity>();
        iter.icon_norm_err_ = rhs.iq_cons().norm();
        iter.max_i_mult_ = xsl.iq_lmults().lpNorm<Eigen::Infinity>();
        this->complementarity(xsl.slacks(), xsl.iq_lmults(), avgcomp, mincomp, maxcomp);

        iter.barr_inf_ = maxcomp;
        iter.barr_norm_err_ = avgcomp;
    }
    if (equal_cons_ > 0) {
        iter.econ_inf_ = rhs.eq_cons().lpNorm<Eigen::Infinity>();
        iter.econ_norm_err_ = rhs.eq_cons().norm();
        iter.max_e_mult_ = xsl.eq_lmults().lpNorm<Eigen::Infinity>();
    }

    iter.kkt_norm_err_ = rhs.prim_grad().norm();

    if (equal_cons_ > 0 || inequal_cons_ > 0)
        iter.all_con_norm_err_ = rhs.all_cons().norm();
}

void tycho::solvers::PSIOPT::eval_nlp(AlgorithmModes algmode, double obj_scale,
                                      ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    std::fill_n(KKTmat.valuePtr(), KKTmat.nonZeros(), 0.0);

    switch (algmode) {
    case AlgorithmModes::OPT:
        eval_kkt(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
        break;
    case AlgorithmModes::OPTNO:
        eval_kkt_no(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);

        break;
    case AlgorithmModes::INIT:
        eval_aug(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
        break;
    case AlgorithmModes::SOE:
        this->nlp_->set_primal_diags(1.0);
        eval_soe(0.0, XSL, val, GX, AGXS_FX, KKTmat);
        this->nlp_->set_primal_diags(0.0);
        GX.head(primal_vars_).setZero();
        AGXS_FX.head(primal_vars_).setZero();
        break;
    default:
        throw std::invalid_argument("Unknown AlgorithmMode");
    }
}

tycho::ConvergenceFlags tycho::solvers::PSIOPT::converge_check(std::vector<IterateInfo> &iters) {
    ConvergenceFlags Flag = ConvergenceFlags::CONVERGED;
    IterateInfo last = iters.back();
    bool KKTFeas = (last.kkt_inf_ < settings_.kkt_tol_);
    bool EConFeas = (last.econ_inf_ < settings_.econ_tol_);
    bool IConFeas = (last.icon_inf_ < settings_.icon_tol_);
    bool BarFeas = (last.barr_inf_ < settings_.bar_tol_);

    bool KKTDiv = (last.kkt_inf_ > settings_.div_kkt_tol_) || !std::isfinite(last.kkt_inf_);
    bool EConDiv = (last.econ_inf_ > settings_.div_econ_tol_) || !std::isfinite(last.econ_inf_);
    bool IConDiv = (last.icon_inf_ > settings_.div_icon_tol_) || !std::isfinite(last.icon_inf_);
    bool BarDiv = (last.barr_inf_ > settings_.div_bar_tol_) || !std::isfinite(last.barr_inf_);

    if (KKTDiv || EConDiv || IConDiv || BarDiv) {
        Flag = ConvergenceFlags::DIVERGING;
        return Flag;
    } else if (KKTFeas && EConFeas && IConFeas && BarFeas) {
        Flag = ConvergenceFlags::CONVERGED;
        return Flag;
    } else if (int(iters.size()) > settings_.max_acc_iters_) {
        int nfeas = 0;
        for (int i = 0; i < settings_.max_acc_iters_; i++) {
            last = iters[int(iters.size()) - i - 1];
            KKTFeas = (last.kkt_inf_ < settings_.acc_kkt_tol_);
            EConFeas = (last.econ_inf_ < settings_.acc_econ_tol_);
            IConFeas = (last.icon_inf_ < settings_.acc_icon_tol_);
            BarFeas = (last.barr_inf_ < settings_.acc_bar_tol_);
            if (KKTFeas && EConFeas && IConFeas && BarFeas)
                nfeas++;
            else
                break;
        }
        if (nfeas == settings_.max_acc_iters_) {
            Flag = ConvergenceFlags::ACCEPTABLE;
            return Flag;
        }
    }
    Flag = ConvergenceFlags::NOTCONVERGED;
    return Flag;
}

int tycho::solvers::PSIOPT::factor_impl(bool docompute, bool Zfac, double ipurt, double incpurt0,
                                        double incpurt, double &finalpert) {
    auto Inertia = [&]() {
        return this->kkt_sol_.neigs() - (this->equal_cons_ + this->inequal_cons_);
    };
    auto RankDef = [&]() {
        if ((this->kkt_sol_.neigs() + this->kkt_sol_.peigs() - this->kkt_dim_) != 0) {
            std::cout << "Potential Rank Deficiency Detected!!!" << std::endl;
        }
    };
    auto Perturb = [&](double p) {
        this->nlp_->perturb_kkt_p_diags(p, this->kkt_sol_.get_matrix());
    };
    auto Refactor = [&]() { this->kkt_sol_.refactorize_internal(); };
    auto Compute = [&]() { this->kkt_sol_.compute_internal(); };
    int IncEigs;

    if (Zfac || docompute) {
        if (!docompute)
            Refactor();
        else
            Compute();
        RankDef();
        IncEigs = Inertia();
        finalpert = 0.0;
        if (IncEigs <= 0)
            return 0;
    }
    double p = ipurt;

    for (int i = 0; i < settings_.max_refac_; i++) {
        Perturb(p);
        Refactor();
        RankDef();
        IncEigs = Inertia();
        finalpert = p;

        if (IncEigs <= 0)
            return i + 1;
        if (i == 0)
            p *= incpurt0;
        else
            p *= incpurt;
        p -= finalpert;
    }
    return settings_.max_refac_;
}

Eigen::VectorXd tycho::solvers::PSIOPT::alg_impl(AlgorithmModes algmode, BarrierModes barmode,
                                                 LineSearchModes lsmode, double obj_scale,
                                                 double MuI, Eigen::Ref<Eigen::VectorXd> xsl) {
    Eigen::VectorXd XSL = xsl;
    Eigen::VectorXd RHS(this->kkt_dim_);
    Eigen::VectorXd DXSL(this->kkt_dim_);
    Eigen::VectorXd RHS2(this->kkt_dim_);
    Eigen::VectorXd PGX(this->primal_vars_);

    Eigen::VectorXd Temp(this->kkt_dim_);
    Eigen::VectorXd Err;

    Eigen::VectorXd BestXSL;
    Eigen::VectorXd BestRHS;
    double BestCriteriaVal = 1.0e10;
    int BestIter = 0;

    double mu = MuI;

    // Create KKTVector views over the working vectors
    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_rhs = kkt_view(RHS);
    KKTVector v_dxsl = kkt_view(DXSL);
    KKTVector v_temp = kkt_view(Temp);

    tycho::utils::Timer Runtimer;
    tycho::utils::Timer Funtimer;
    tycho::utils::Timer LStimer;
    tycho::utils::Timer QPtimer;
    tycho::utils::Timer
        CBtimer; // Callback time is included in result_.misc_time_ (not separately reported)
    tycho::utils::Timer Printtimer;

    double Hpert0 = settings_.delta_h_;
    std::vector<IterateInfo> iters;
    iters.reserve(settings_.max_iters_);
    ConvergenceFlags ExitCode;
    bool FirstPert = true;

    Runtimer.start();
    for (int i = 0; i < settings_.max_iters_; i++) {
        IterateInfo Citer;
        Citer.iter = i;

        double avgcomp = 0;
        double mincomp = 0;
        double maxcomp = 0;
        double alpha = 1.0;
        double alphap = 1.0;
        double alphad = 1.0;

        RHS.setZero();
        PGX.setZero();
        double prim_obj = 0;
        double barr_obj = 0;

        Funtimer.start();
        /////////////////////////////////////////////////////////////

        this->eval_nlp(algmode, obj_scale, XSL, prim_obj, PGX, RHS, this->kkt_sol_.get_matrix());

        if (this->inequal_cons_ > 0) {
            this->apply_reset_slacks(v_xsl.slacks(), v_rhs.iq_cons());
            this->barrier_hessian(this->kkt_sol_.get_matrix(), v_xsl.slacks(), v_xsl.iq_lmults(),
                                  mu);
            this->complementarity(v_xsl.slacks(), v_xsl.iq_lmults(), avgcomp, mincomp, maxcomp);
        }

        ///////////////////////////////////////////////////////////////
        Funtimer.stop();
        if (this->early_callback_enabled_) {
            CBtimer.start();
            this->early_callback_(i, obj_scale, XSL, prim_obj, PGX, RHS,
                                  this->kkt_sol_.get_matrix());
            CBtimer.stop();
        }
        QPtimer.start();
        ////////////////////////////////////////////////////////////////
        v_rhs.prim_grad() += PGX;

        ////////////////////////////////////////////////////////////////
        double nhpert = 0;
        double Incr = settings_.incr_h_;
        double Incr2 = settings_.incr_h_;
        if (FirstPert)
            Incr2 *= settings_.incr_h_;
        bool Zfac = true;
        if (settings_.fast_factor_alg_ && i > 6 && ((i * 3) % 4) != 0) {
            bool cycling = true;
            for (int j = 0; j < 4; j++) {
                int ns = iters[iters.size() - 1 - j].h_facs_;
                if (ns == 0) {
                    cycling = false;
                    break;
                }
            }
            Zfac = !cycling;
        }

        Citer.h_facs_ = this->factor_impl(false, Zfac, Hpert0, Incr, Incr2, nhpert);

        if (Citer.h_facs_ > 0) {
            Hpert0 = std::max(settings_.delta_h_, nhpert * settings_.decr_h_);
            FirstPert = false;
        }
        Citer.h_pert_ = nhpert;
        ///////////////////////////////////////////////////////////////////

        if (this->inequal_cons_ > 0) {
            switch (barmode) {
            case BarrierModes::PROBE:
                this->barrier_gradient(v_xsl.iq_lmults(), v_rhs.dual_grad());
                DXSL = -this->kkt_sol_.solve(RHS);
                this->max_primal_dual_step(v_xsl, v_dxsl, settings_.bound_fraction_, alphap,
                                           alphad);
                Temp = XSL + DXSL;
                mu = this->mpc_mu(v_temp.slacks(), v_temp.iq_lmults(), avgcomp, mincomp);

                break;
            case BarrierModes::LOQO:
                mu = this->loqo_mu(v_xsl.slacks(), v_xsl.iq_lmults(), avgcomp, mincomp);
                break;
            default:
                break;
            }

            mu = std::max(mu, settings_.min_mu_);
            mu = std::min(mu, settings_.max_mu_);
            barr_obj = this->barrier_objective(v_xsl.slacks(), mu);
            this->barrier_gradient(v_xsl.slacks(), v_xsl.iq_lmults(), mu, v_rhs.dual_grad());
        }

        DXSL = -this->kkt_sol_.solve(RHS);
        bool GoodStep = std::isfinite(DXSL.squaredNorm());
        if (this->inequal_cons_ > 0)
            this->max_primal_dual_step(v_xsl, v_dxsl, settings_.bound_fraction_, alphap, alphad);
        /////////////////////////////////////////////////////////////////////
        QPtimer.stop();

        Funtimer.start();
        //////////////////////////////////////////////////////////////////////

        if (GoodStep) {
            double lsobjscale =
                algmode == AlgorithmModes::SOE || algmode == AlgorithmModes::OPTNO ? 0.0 : 1.0;
            alpha = ls_impl(lsmode, obj_scale * lsobjscale, mu, prim_obj, barr_obj, XSL, DXSL, Temp,
                            RHS, RHS2, Citer, iters);

        } else {
            Citer.h_facs_ = -1;
        }

        //////////////////////////////////////////////////////////////////////
        Funtimer.stop();

        Citer.alpha_p_ = alphap;
        Citer.alpha_d_ = alphad;
        Citer.alpha_t_ = alpha;

        this->fill_iter_info(v_xsl, v_rhs, prim_obj, barr_obj, mu, Citer);
        iters.push_back(Citer);

        if (settings_.return_best_) {
            double critval;
            switch (settings_.best_criteria_) {
            case BestCriteriaModes::ECONS:
                critval = iters.back().econ_inf_;
                break;
            case BestCriteriaModes::ICONS:
                critval = iters.back().icon_inf_;
                break;
            case BestCriteriaModes::KKT:
                critval = iters.back().kkt_inf_;
                break;
            case BestCriteriaModes::OBJ:
                critval = iters.back().prim_obj_;
                break;
            default:
                throw std::invalid_argument("Unknown BestCriteriaModes");
            }
            if (critval <= BestCriteriaVal || i == 0) {
                BestCriteriaVal = critval;
                BestXSL = XSL;
                BestRHS = RHS;
                BestIter = i;
            }
        }

        if (this->late_callback_enabled_) {
            CBtimer.start();
            this->late_callback_(iters.back(), XSL, RHS);
            CBtimer.stop();
        }

        ExitCode = this->converge_check(iters);
        if (!GoodStep)
            ExitCode = ConvergenceFlags::DIVERGING;

        if (settings_.print_level_ == 0) {
            Printtimer.start();
            this->print_last_iterate(iters);
            Printtimer.stop();
        }

        if (ExitCode == ConvergenceFlags::CONVERGED || ExitCode == ConvergenceFlags::ACCEPTABLE ||
            ExitCode == ConvergenceFlags::DIVERGING || i == (settings_.max_iters_ - 1)) {

            if (ExitCode != ConvergenceFlags::CONVERGED && settings_.return_best_) {
                XSL = BestXSL;
                RHS = BestRHS;
            }

            this->result_.converge_flag_ = ExitCode;
            break;
        }

        /////////Very Important/////////
        XSL += alpha * DXSL;
        ///////////////////////////////
    }

    if (algmode == AlgorithmModes::OPT) {
        this->result_.obj_val_ = iters.back().prim_obj_;
    } else {
        Funtimer.start();
        this->result_.obj_val_ = 0;
        this->nlp_->eval_obj(obj_scale, v_xsl.primals(), this->result_.obj_val_);
        Funtimer.stop();
    }

    if (this->equal_cons_ > 0) {
        this->result_.eq_cons_ = v_rhs.eq_cons();
        this->result_.eq_lmults_ = v_xsl.eq_lmults();
    }
    if (this->inequal_cons_ > 0) {
        this->result_.iq_cons_ = v_rhs.iq_cons() - v_xsl.slacks();
        this->result_.iq_lmults_ = v_xsl.iq_lmults();
    }

    Runtimer.stop();
    this->result_.iter_num_ += iters.size();
    double qptime = double(QPtimer.count<std::chrono::microseconds>()) / 1000000.0;
    double nlptime = double(Funtimer.count<std::chrono::microseconds>()) / 1000000.0;
    double tottime = double(Runtimer.count<std::chrono::microseconds>()) / 1000000.0;

    this->result_.func_time_ += nlptime;
    this->result_.kkt_time_ += qptime;
    double printtime = double(Printtimer.count<std::chrono::microseconds>()) / 1000000.0;
    this->result_.print_time_ += printtime;

    ///////////////////////////////////////////////////////////////////////////

    int retiter = (settings_.return_best_ ? BestIter : iters.size() - 1);
    print_exit_stats(ExitCode, iters[retiter], iters.size(), tottime * 1000, nlptime * 1000,
                     qptime * 1000, printtime * 1000);
    ////////////////////////////////////////////////////////////////////////////

    return XSL;
}

Eigen::VectorXd tycho::solvers::PSIOPT::init_impl(const Eigen::VectorXd &x, double mu,
                                                  bool docompute) {

    tycho::utils::Timer kktt;
    kktt.start();

    Eigen::VectorXd XSL(this->kkt_dim_);
    XSL.setZero();
    XSL.head(this->primal_vars_) = x;

    Eigen::VectorXd RHS(this->kkt_dim_);
    RHS.setZero();
    double val = 0;
    this->nlp_->set_primal_diags(1.0);
    if (this->inequal_cons_ > 0) {
        this->nlp_->set_slacks_ones();
    }
    this->eval_nlp(AlgorithmModes::INIT, settings_.obj_scale_, XSL, val, RHS.head(this->primal_vars_),
                   RHS, this->kkt_sol_.get_matrix());

    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_rhs = kkt_view(RHS);

    Eigen::VectorXd hp(this->slack_vars_);

    for (int i = 0; i < this->slack_vars_; i++) {
        double fxi = v_rhs.iq_cons()[i];
        if (fxi < -settings_.bound_push_) {
            v_xsl.slacks()[i] = abs(fxi);
        } else {
            v_xsl.slacks()[i] = settings_.bound_push_;
        }
        hp[i] = 1.0;
        v_xsl.iq_lmults()[i] = mu / v_xsl.slacks()[i];
    }

    RHS.tail(this->equal_cons_ + this->inequal_cons_).setZero();

    if (this->inequal_cons_ > 0)
        this->nlp_->assign_kkt_slack_hessian(hp, this->kkt_sol_.get_matrix());
    if (settings_.print_level_ < 2) {
        print_beginning("KKT-Matrix Analysis ");
    }

    if (docompute)
        this->kkt_sol_.compute_internal();
    else
        this->kkt_sol_.refactorize_internal();
    kktt.stop();

    double pretime = double(kktt.count<std::chrono::microseconds>()) / 1000000.0;
    this->result_.pre_time_ += pretime;

    this->result_.factor_flops_ = this->kkt_sol_.flops_;
    this->result_.factor_mem_ = this->kkt_sol_.mem_;

    if (settings_.print_level_ < 2) {
        auto cyan = fmt::fg(fmt::color::cyan);
        if (docompute) {
            fmt::print(" LDLT Factor Size      : ");
            fmt::print(cyan, "{0:<10}\n", this->result_.factor_mem_);
            if (this->result_.factor_flops_ > 0) {
                fmt::print(" LDLT Factor FLOPs     : ");
                fmt::print(cyan, "{0} MFLOPs\n", this->result_.factor_flops_);
            }
        }
        fmt::print(" Analysis/Reorder Time : ");
        fmt::print(cyan, "{0:.3f} ms\n", pretime * 1000);
        print_finished("KKT-Matrix Analysis ");
    }

    Eigen::VectorXd dx = -this->kkt_sol_.solve(RHS);
    KKTVector v_dx = kkt_view(dx);

    if (equal_cons_ > 0)
        v_xsl.eq_lmults() = v_dx.eq_lmults();
    if (this->inequal_cons_ > 0)
        this->nlp_->set_slack_diags(0.0);
    this->nlp_->set_primal_diags(0.0);

    return XSL;
}

// ============================================================================
// Line search — shared helpers
// ============================================================================

void tycho::solvers::PSIOPT::eval_trial_point_occ(double obj_scale, double mu, double alpha,
                                                   KKTVector &xsl, KKTVector &dxsl,
                                                   KKTVector &xsl2, KKTVector &rhs2,
                                                   double &ptest, double &btest) {
    xsl2.data() = xsl.data() + alpha * dxsl.data();
    rhs2.data().setZero();
    this->nlp_->eval_occ(obj_scale, xsl2.primals(), ptest, rhs2.eq_cons(), rhs2.iq_cons());
    this->apply_reset_slacks(xsl2.slacks(), rhs2.iq_cons());
    btest = this->barrier_objective(xsl2.slacks(), mu);
}

auto tycho::solvers::PSIOPT::compute_penalties(KKTVector &xsl, KKTVector &rhs) const
    -> PenaltyTerms {
    return {xsl.lmults().cwiseAbs().dot(rhs.all_cons().cwiseAbs()),
            rhs.all_cons().squaredNorm(),
            rhs.all_cons().template lpNorm<Eigen::Infinity>()};
}

bool tycho::solvers::PSIOPT::secondary_accept(double ptest, double prim_obj,
                                               const PenaltyTerms &test,
                                               const PenaltyTerms &init) const {
    return (ptest < prim_obj && test.l2_ < init.l2_) ||
           (ptest < prim_obj && test.linf_ < init.linf_);
}

// ============================================================================
// Line search — variant implementations
// ============================================================================

double tycho::solvers::PSIOPT::ls_lang(double obj_scale, double mu, double prim_obj,
                                       double barr_obj, KKTVector &xsl, KKTVector &dxsl,
                                       KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                                       IterateInfo &citer) {
    double alpha = 1.0;
    double LangInit = prim_obj + barr_obj + xsl.lmults().dot(rhs.all_cons());

    for (int j = 0; j < settings_.max_ls_iters_; j++) {
        double ptest = 0;
        double btest = 0;
        xsl2.data() = xsl.data() + alpha * dxsl.data();
        rhs2.data().setZero();
        this->eval_rhs(obj_scale, xsl2.data(), ptest, rhs2.data(), rhs2.data());
        this->apply_reset_slacks(xsl2.slacks(), rhs2.iq_cons());
        btest = this->barrier_objective(xsl2.slacks(), mu);
        this->barrier_gradient(xsl2.slacks(), xsl2.iq_lmults(), mu, rhs2.dual_grad());
        double LangTest = ptest + btest + xsl2.lmults().dot(rhs2.all_cons());
        citer.ls_iters_ = j;
        if (LangTest < LangInit) {
            break;
        } else {
            alpha = alpha / settings_.alpha_red_;
        }
    }
    return alpha;
}

double tycho::solvers::PSIOPT::ls_l1(double obj_scale, double mu, double prim_obj,
                                     double barr_obj, KKTVector &xsl, KKTVector &dxsl,
                                     KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                                     IterateInfo &citer) {
    double alpha = 1.0;
    double vv = rhs.prim_dual_grad().dot(dxsl.primals_slacks());
    double cv = dxsl.lmults().dot(rhs.all_cons());

    PenaltyTerms init = compute_penalties(xsl, rhs);

    double sc = .1 + std::abs(vv - cv) / init.l2_;
    if (init.l2_ == 0.0)
        sc = 1.0;

    double LangInit = prim_obj + barr_obj + init.l1_ + init.l2_ * sc;

    for (int j = 0; j < settings_.max_ls_iters_; j++) {
        double ptest = 0;
        double btest = 0;
        eval_trial_point_occ(obj_scale, mu, alpha, xsl, dxsl, xsl2, rhs2, ptest, btest);

        double LangTest = ptest + btest;
        PenaltyTerms test = compute_penalties(xsl, rhs2);
        LangTest += test.l1_ + test.l2_ * sc;

        citer.merit_val_ = LangTest;
        if (LangTest < LangInit || secondary_accept(ptest, prim_obj, test, init)) {
            citer.ls_iters_ = j;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / settings_.alpha_red_;
        }
    }
    return alpha;
}

double tycho::solvers::PSIOPT::ls_auglang(double obj_scale, double mu, double prim_obj,
                                           double barr_obj, KKTVector &xsl, KKTVector &dxsl,
                                           KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                                           IterateInfo &citer) {
    double alpha = 1.0;
    double vv = rhs.prim_dual_grad().dot(dxsl.primals_slacks());
    double cv = dxsl.lmults().dot(rhs.all_cons());

    PenaltyTerms init = compute_penalties(xsl, rhs);

    double sc = .01 + std::abs(vv - cv) / init.l2_;
    if (init.l2_ == 0.0)
        sc = 1.0;

    double LangInit = prim_obj + barr_obj + init.l1_ + init.l2_ * sc;

    for (int j = 0; j < settings_.max_ls_iters_; j++) {
        double ptest = 0;
        double btest = 0;
        eval_trial_point_occ(obj_scale, mu, alpha, xsl, dxsl, xsl2, rhs2, ptest, btest);

        double LangTest = ptest + btest;

        // Tolerance-filtered L1 penalty
        double TestL1Pen = 0;
        for (int i = 0; i < this->equal_cons_; i++) {
            double eqerr = abs(rhs2.eq_cons()[i]);
            double eqmul = abs(xsl.eq_lmults()[i]);
            if (eqerr > settings_.econ_tol_ * 10) {
                TestL1Pen += eqerr * eqmul;
            }
        }
        for (int i = 0; i < this->inequal_cons_; i++) {
            double iqerr = abs(rhs2.iq_cons()[i]);
            double iqmul = abs(xsl.iq_lmults()[i]);
            if (iqerr > settings_.icon_tol_ * 10) {
                TestL1Pen += iqerr * iqmul;
            }
        }

        double TestL2Pen = rhs2.all_cons().squaredNorm();
        double TestLinfPenalty = rhs2.all_cons().template lpNorm<Eigen::Infinity>();

        // Zero L2 when within tolerance threshold
        if (TestL2Pen < settings_.econ_tol_ * settings_.econ_tol_ * equal_cons_ +
                            settings_.icon_tol_ * settings_.icon_tol_ * inequal_cons_) {
            TestL2Pen = 0;
        }

        LangTest += TestL1Pen + TestL2Pen * sc;

        PenaltyTerms test{TestL1Pen, TestL2Pen, TestLinfPenalty};
        citer.merit_val_ = LangTest;
        if (LangTest < LangInit || secondary_accept(ptest, prim_obj, test, init)) {
            citer.ls_iters_ = j;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / settings_.alpha_red_;
        }
    }
    return alpha;
}

// ============================================================================
// Line search — dispatcher
// ============================================================================

double tycho::solvers::PSIOPT::ls_impl(LineSearchModes lsmode, double obj_scale, double mu,
                                       double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                                       Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                                       Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2,
                                       IterateInfo &Citer, const std::vector<IterateInfo> &iters) {

    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_dxsl = kkt_view(DXSL);
    KKTVector v_xsl2 = kkt_view(XSL2);
    KKTVector v_rhs = kkt_view(RHS);
    KKTVector v_rhs2 = kkt_view(RHS2);

    switch (lsmode) {
    case LineSearchModes::LANG:
        return ls_lang(obj_scale, mu, prim_obj, barr_obj, v_xsl, v_dxsl, v_xsl2, v_rhs, v_rhs2,
                       Citer);
    case LineSearchModes::L1:
        return ls_l1(obj_scale, mu, prim_obj, barr_obj, v_xsl, v_dxsl, v_xsl2, v_rhs, v_rhs2,
                     Citer);
    case LineSearchModes::AUGLANG:
        return ls_auglang(obj_scale, mu, prim_obj, barr_obj, v_xsl, v_dxsl, v_xsl2, v_rhs, v_rhs2,
                          Citer);
    case LineSearchModes::NOLS:
        Citer.ls_iters_ = 0;
        return 1.0;
    default:
        throw std::invalid_argument("Unknown LineSearchMode");
    }
}

Eigen::VectorXd tycho::solvers::PSIOPT::run_phase_sequence(
    const Eigen::VectorXd &x, std::initializer_list<PhaseStep> steps) {

    this->result_.zero_timing();

    if (settings_.print_level_ == 0)
        print_stats();
    if (settings_.print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();

    tycho::utils::Timer t;
    t.start();

    bool docompute = analyze_kkt_matrix();
    Eigen::VectorXd XSL = this->init_impl(x, settings_.init_mu_, docompute);
    Eigen::VectorXd XSLans(this->kkt_dim_);
    XSLans.setZero();

    auto it = steps.begin();
    auto end = steps.end();
    while (it != end) {
        const auto &step = *it;
        ++it;
        bool is_last = (it == end);

        // Conditional steps only run if the previous phase didn't converge
        if (step.conditional_ && this->result_.converge_flag_ == ConvergenceFlags::CONVERGED)
            continue;

        if (settings_.print_level_ < 2)
            print_beginning(step.label_);

        XSLans = this->alg_impl(step.alg_mode_, step.bar_mode_, step.ls_mode_,
                                settings_.obj_scale_, settings_.init_mu_, XSL);

        if (settings_.print_level_ < 2)
            print_finished(step.label_);

        // Re-init for the next phase (extract primals, re-run init_impl)
        if (!is_last) {
            Eigen::VectorXd Xt = XSLans.head(primal_vars_);
            XSL = this->init_impl(Xt, settings_.init_mu_, false);
        }
    }

    t.stop();
    double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
    this->result_.total_time_ = tottime / 1000.0;
    this->result_.misc_time_ = this->result_.total_time_ - this->result_.pre_time_ -
                               this->result_.kkt_time_ - this->result_.func_time_ -
                               this->result_.print_time_;

    if (settings_.print_level_ < 2) {
        print_timing_summary();
        fmt::print(" PSIOPT Total Time            : ");
        fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.3f} ms\n", tottime);
        print_finished("PSIOPT ");
        print_header();
    }

    return XSLans.head(primal_vars_);
}

Eigen::VectorXd tycho::solvers::PSIOPT::optimize(const Eigen::VectorXd &x) {
    return run_phase_sequence(
        x, {{AlgorithmModes::OPT, settings_.opt_bar_mode_, settings_.opt_ls_mode_,
             "Optimization Algorithm "}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve(const Eigen::VectorXd &x) {
    return run_phase_sequence(
        x, {{settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_,
             "Solve Algorithm "}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve_optimize(const Eigen::VectorXd &x) {
    return run_phase_sequence(
        x, {{settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_,
             "Solve Algorithm "},
            {AlgorithmModes::OPT, settings_.opt_bar_mode_, settings_.opt_ls_mode_,
             "Optimization Algorithm "}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::optimize_solve(const Eigen::VectorXd &x) {
    return run_phase_sequence(
        x, {{AlgorithmModes::OPT, settings_.opt_bar_mode_, settings_.opt_ls_mode_,
             "Optimization Algorithm "},
            {settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_,
             "Solve Algorithm ", /*conditional_=*/true}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve_optimize_solve(const Eigen::VectorXd &x) {
    return run_phase_sequence(
        x, {{settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_,
             "Solve Algorithm "},
            {AlgorithmModes::OPT, settings_.opt_bar_mode_, settings_.opt_ls_mode_,
             "Optimization Algorithm "},
            {settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_,
             "Solve Algorithm ", /*conditional_=*/true}});
}
