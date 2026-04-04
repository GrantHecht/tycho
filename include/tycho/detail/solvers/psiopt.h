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

#pragma once
#include <algorithm>
#include <cmath>
#include <compare>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "tycho/detail/solvers/iterate_info.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/get_core_count.h"

#ifdef USE_ACCELERATE_SPARSE
#include "tycho/detail/solvers/linear/accelerate_interface.h"
#else
#include "tycho/detail/solvers/linear/pardiso_interface.h"
#endif

namespace tycho {

/// Optimizer convergence status — extracted to root namespace so that callers
/// outside tycho::solvers can reference it without a full PSIOPT qualification.
enum class ConvergenceFlags {
    CONVERGED,
    ACCEPTABLE,
    NOTCONVERGED,
    DIVERGING,
};

// Severity ordering: CONVERGED < ACCEPTABLE < NOTCONVERGED < DIVERGING
constexpr auto operator<=>(ConvergenceFlags a, ConvergenceFlags b) {
    return static_cast<int>(a) <=> static_cast<int>(b);
}

} // namespace tycho

namespace tycho::solvers {

// Pull root-namespace Eigen type aliases into tycho::solvers so that PSIOPT
// member declarations (EigenRef<VectorXd>, ConstEigenRef<VectorXd>, …) resolve
// without full qualification inside this namespace.
using tycho::ConstEigenRef;
using tycho::EigenRef;

struct IterateInfo;

struct PSIOPT {

    enum class BarrierModes { PROBE, LOQO };
    enum class LineSearchModes { AUGLANG, LANG, L1, NOLS };
    enum class AlgorithmModes { OPT, OPTNO, SOE, INIT };

    /// Alias so existing callers using PSIOPT::ConvergenceFlags continue to work.
    using ConvergenceFlags = tycho::ConvergenceFlags;

    enum class QPAlgModes {
        Classic = 0,
        TwoLevel = 1,
    };

    enum class QPOrderingModes { MINDEG = 0, METIS = 2, PARMETIS = 3 };
    enum class BestCriteriaModes { ECONS, ICONS, KKT, OBJ };

    enum class QPPivotModes {
        OneByOne = 0,
        TwoByTwo = 1,
        E4 = 4,
        E6 = 6,
        E8 = 8,
        E13 = 13,
    };
    enum class PDStepStrategies { PrimSlackEq_Iq, AllMinimum, PrimSlack_EqIq, MaxEq };

    static QPOrderingModes strto_OrderingMode(const std::string &str) {

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
    static LineSearchModes strto_LineSearchMode(const std::string &str) {

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
    static BarrierModes strto_BarrierMode(const std::string &str) {

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
    static BestCriteriaModes strto_BestCriteriaMode(const std::string &str) {

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

    using VectorXd = Eigen::VectorXd;
    std::shared_ptr<NonLinearProgram> nlp_;

    int primal_vars_ = 0;
    int slack_vars_ = 0;
    int equal_cons_ = 0;
    int inequal_cons_ = 0;
    int kkt_dim_ = 0;

#ifdef USE_ACCELERATE_SPARSE
    Eigen::AccelerateLDLTTPP<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> kkt_sol_;
#else
    Eigen::PardisoLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> kkt_sol_;
#endif

    int qp_threads_ = TYCHO_DEFAULT_QP_THREADS;
    QPAlgModes qp_alg_ = QPAlgModes::Classic;
    QPOrderingModes qp_ord_ = QPOrderingModes::METIS;
    QPPivotModes qp_pivot_strategy_ = QPPivotModes::TwoByTwo;

    void set_qp_ordering_mode(QPOrderingModes mode) { this->qp_ord_ = mode; }
    void set_qp_ordering_mode(const std::string &str) { this->qp_ord_ = strto_OrderingMode(str); }

    int qp_matching_ = 1;
    int qp_scaling_ = 0;
    int qp_pivot_perturb_ = 8;
    int qp_ref_steps_ = 0;
#ifdef USE_ACCELERATE_SPARSE
    double accel_pivot_tolerance_ = 0.01;
    double accel_zero_tolerance_ = 1e-4 * std::numeric_limits<double>::epsilon();

    void set_accel_pivot_tolerance(double tol) { this->accel_pivot_tolerance_ = tol; }
    void set_accel_zero_tolerance(double tol) { this->accel_zero_tolerance_ = tol; }
#endif
    bool qp_print_ = false;
    bool qp_analyzed_ = false;
    bool force_qp_analysis_ = false;
    int qp_par_solve_ = 0;

    /////////////////////////////////////////////////////////////////////
    int max_iters_ = 500;
    int max_ls_iters_ = 2;
    int max_acc_iters_ = 50;

    void set_max_iters(int max_iters) {
        if (max_iters < 1) {
            throw std::invalid_argument("max_iters must be greater than 0.");
        }
        this->max_iters_ = max_iters;
    }
    void set_max_acc_iters(int max_acc_iters) {
        if (max_acc_iters < 1) {
            throw std::invalid_argument("max_acc_iters must be greater than 0.");
        }
        this->max_acc_iters_ = max_acc_iters;
    }
    void set_max_ls_iters(int max_ls_iters) {
        if (max_ls_iters < 0) {
            throw std::invalid_argument("max_ls_iters must be non-negative (>= 0).");
        }
        this->max_ls_iters_ = max_ls_iters;
    }
    void set_all_max_iters(int m1, int m2) {
        set_max_iters(m1);
        set_max_acc_iters(m2);
    }

    int max_refac_ = 15;

    int max_soc_ = 1;
    int max_feas_rest_ = 2;

    AlgorithmModes soe_mode_ = AlgorithmModes::SOE;

    BarrierModes opt_bar_mode_ = BarrierModes::LOQO;
    BarrierModes soe_bar_mode_ = BarrierModes::LOQO;

    void set_opt_bar_mode(BarrierModes mode) { this->opt_bar_mode_ = mode; }
    void set_opt_bar_mode(const std::string &str) { this->opt_bar_mode_ = strto_BarrierMode(str); }
    void set_soe_bar_mode(BarrierModes mode) { this->soe_bar_mode_ = mode; }
    void set_soe_bar_mode(const std::string &str) { this->soe_bar_mode_ = strto_BarrierMode(str); }

    LineSearchModes opt_ls_mode_ = LineSearchModes::AUGLANG;
    LineSearchModes soe_ls_mode_ = LineSearchModes::NOLS;

    void set_opt_ls_mode(LineSearchModes mode) { this->opt_ls_mode_ = mode; }
    void set_opt_ls_mode(const std::string &str) { this->opt_ls_mode_ = strto_LineSearchMode(str); }
    void set_soe_ls_mode(LineSearchModes mode) { this->soe_ls_mode_ = mode; }
    void set_soe_ls_mode(const std::string &str) { this->soe_ls_mode_ = strto_LineSearchMode(str); }

    double obj_scale_ = 1.0;

    /////////////////////////////////////////////////////////////////////////
    double kkt_tol_ = 1.0e-6;
    double econ_tol_ = 1.0e-6;
    double icon_tol_ = 1.0e-6;
    double bar_tol_ = 1.0e-6;

    void set_kkt_tol(double kkt_tol) { this->kkt_tol_ = std::abs(kkt_tol); }
    void set_bar_tol(double bar_tol) { this->bar_tol_ = std::abs(bar_tol); }
    void set_econ_tol(double econ_tol) { this->econ_tol_ = std::abs(econ_tol); }
    void set_icon_tol(double icon_tol) { this->icon_tol_ = std::abs(icon_tol); }
    void set_tols(double kkt_tol, double econ_tol, double icon_tol, double bar_tol) {
        this->set_kkt_tol(kkt_tol);
        this->set_econ_tol(econ_tol);
        this->set_icon_tol(icon_tol);
        this->set_bar_tol(bar_tol);
    }

    double acc_kkt_tol_ = 1.0e-2;
    double acc_econ_tol_ = 1.0e-3;
    double acc_icon_tol_ = 1.0e-3;
    double acc_bar_tol_ = 1.0e-3;

    void set_acc_kkt_tol(double acc_kkt_tol) { this->acc_kkt_tol_ = std::abs(acc_kkt_tol); }
    void set_acc_bar_tol(double acc_bar_tol) { this->acc_bar_tol_ = std::abs(acc_bar_tol); }
    void set_acc_econ_tol(double acc_econ_tol) { this->acc_econ_tol_ = std::abs(acc_econ_tol); }
    void set_acc_icon_tol(double acc_icon_tol) { this->acc_icon_tol_ = std::abs(acc_icon_tol); }
    void set_acc_tols(double acc_kkt_tol, double acc_econ_tol, double acc_icon_tol,
                      double acc_bar_tol) {
        this->set_acc_kkt_tol(acc_kkt_tol);
        this->set_acc_econ_tol(acc_econ_tol);
        this->set_acc_icon_tol(acc_icon_tol);
        this->set_acc_bar_tol(acc_bar_tol);
    }

    double unacc_kkt_tol_ = 10;
    double unacc_econ_tol_ = 2;
    double unacc_icon_tol_ = 2;
    double unacc_bar_tol_ = 2;

    void set_unacc_tols(double kktol, double etol, double itol, double bartol) {
        this->unacc_kkt_tol_ = kktol;
        this->unacc_bar_tol_ = bartol;
        this->unacc_econ_tol_ = etol;
        this->unacc_icon_tol_ = itol;
    }

    double div_kkt_tol_ = 1.0e15;
    double div_econ_tol_ = 1.0e15;
    double div_icon_tol_ = 1.0e15;
    double div_bar_tol_ = 1.0e15;

    void set_div_kkt_tol(double div_kkt_tol) { this->div_kkt_tol_ = std::abs(div_kkt_tol); }
    void set_div_bar_tol(double div_bar_tol) { this->div_bar_tol_ = std::abs(div_bar_tol); }
    void set_div_econ_tol(double div_econ_tol) { this->div_econ_tol_ = std::abs(div_econ_tol); }
    void set_div_icon_tol(double div_icon_tol) { this->div_icon_tol_ = std::abs(div_icon_tol); }
    void set_div_tols(double div_kkt_tol, double div_econ_tol, double div_icon_tol,
                      double div_bar_tol) {
        this->set_div_kkt_tol(div_kkt_tol);
        this->set_div_econ_tol(div_econ_tol);
        this->set_div_icon_tol(div_icon_tol);
        this->set_div_bar_tol(div_bar_tol);
    }

    /////////////////////////////////////////////////////////////////////////

    double bound_fraction_ = 0.99;

    void set_bound_fraction(double bound_fraction) {
        if (bound_fraction >= 1.0 || bound_fraction <= 0.0) {
            throw std::invalid_argument("bound_fraction must be between 0 and 1.");
        }
        this->bound_fraction_ = bound_fraction;
    }

    double bound_push_ = 1.0e-3;
    void set_bound_push(double bound_push) {
        if (bound_push <= 0.0) {
            throw std::invalid_argument("bound_push must be greater than 0.");
        }
        this->bound_push_ = bound_push;
    }

    double neg_slack_reset_ = 1.0e-12;

    double soe_bound_relax_ = 1.0e-8;
    double alpha_red_ = 2.0;

    void set_alpha_red(double ared) {
        if (ared <= 1.0) {
            throw std::invalid_argument("alpha_red must be greater than 1.0");
        }
        this->alpha_red_ = ared;
    }

    /////////////////////////////////////////////////////////////////////////
    double delta_h_ = 1.0e-5;
    double incr_h_ = 8.00;
    double decr_h_ = 0.333333;

    void set_delta_h(double delta_h) {
        if (delta_h <= 0.0) {
            throw std::invalid_argument("delta_h must be greater than 0.");
        }
        this->delta_h_ = delta_h;
    }
    void set_incr_h(double incr_h) {
        if (incr_h <= 1.0) {
            throw std::invalid_argument("incr_h must be greater than 1.0.");
        }
        this->incr_h_ = incr_h;
    }
    void set_decr_h(double decr_h) {
        if (decr_h >= 1.0 || decr_h <= 0) {
            throw std::invalid_argument("decr_h must be between 0 and 1.");
        }
        this->decr_h_ = decr_h;
    }
    void set_hpert_params(double delta_h, double incr_h, double decr_h) {
        this->set_delta_h(delta_h);
        this->set_incr_h(incr_h);
        this->set_decr_h(decr_h);
    }
    /////////////////////////////////////////////////////////////////////////
    ConvergenceFlags converge_flag_ = ConvergenceFlags::NOTCONVERGED;
    ConvergenceFlags get_convergence_flag() const { return this->converge_flag_; }

    double init_mu_ = 0.001;
    double max_mu_ = 100.0;
    double min_mu_ = 1.0e-12;

    bool cnr_mode_ = false;
    int print_level_ = 0;
    void set_print_level(int plevel) { this->print_level_ = plevel; }

    PDStepStrategies pd_step_strategy_ = PDStepStrategies::PrimSlackEq_Iq;
    double last_obj_val_ = 0.0;
    bool fast_factor_alg_ = true;

    double last_total_time_ = 0;
    double last_pre_time_ = 0;
    double last_misc_time_ = 0;
    double last_func_time_ = 0;
    double last_kkt_time_ = 0;
    double last_print_time_ = 0;
    double last_solver_init_time_ = 0;
    int last_iter_num_ = 0;

    void zero_timing_stats() {
        this->last_total_time_ = 0;
        this->last_pre_time_ = 0;
        this->last_misc_time_ = 0;
        this->last_func_time_ = 0;
        this->last_kkt_time_ = 0;
        this->last_print_time_ = 0;
        this->last_solver_init_time_ = 0;
        this->last_iter_num_ = 0;
    }

    bool wide_console_ = false;
    int factor_mem_ = 0;
    int factor_flops_ = 0;
    Eigen::VectorXd last_eq_lmults_;
    Eigen::VectorXd last_iq_lmults_;

    Eigen::VectorXd last_eq_cons_;
    Eigen::VectorXd last_iq_cons_;

    bool return_best_ = false;
    BestCriteriaModes best_criteria_ = BestCriteriaModes::ECONS;

    void set_best_criteria(BestCriteriaModes mode) { this->best_criteria_ = mode; }
    void set_best_criteria(const std::string &str) {
        this->best_criteria_ = strto_BestCriteriaMode(str);
    }

    /////////////////////////////////////////////////////////////////////

    using EarlyCallBackType =
        std::function<int(int, double, EigenRef<VectorXd>, double, EigenRef<VectorXd>,
                          EigenRef<VectorXd>, Eigen::SparseMatrix<double, Eigen::RowMajor> &)>;

    using LateCallBackType =
        std::function<int(const IterateInfo &, ConstEigenRef<VectorXd>, ConstEigenRef<VectorXd>)>;

    EarlyCallBackType early_callback_; // = [](int i, EigenRef<VectorXd> XSL, EigenRef<VectorXd>
                                       // GX, EigenRef<VectorXd> AGXFX) {return 0; };
    bool early_callback_enabled_ = false;
    LateCallBackType late_callback_; // = [](const IterateInfo& i, EigenRef<VectorXd> XSL,
                                     // EigenRef<VectorXd> AGXFX) {return 0; };
    bool late_callback_enabled_ = false;

    ////////////////////////////////////////////////////////////////////

    PSIOPT() {
        this->qp_threads_ = std::min(TYCHO_DEFAULT_QP_THREADS, tycho::utils::get_core_count());
    }
    PSIOPT(std::shared_ptr<NonLinearProgram> np) {
        this->qp_threads_ = std::min(TYCHO_DEFAULT_QP_THREADS, tycho::utils::get_core_count());
        this->set_nlp(np);
    }

    void release() {
        this->kkt_sol_.release();
        this->qp_analyzed_ = false;
        this->nlp_ = std::shared_ptr<NonLinearProgram>();
        this->last_eq_lmults_.resize(0);
        this->last_iq_lmults_.resize(0);
    }

    void set_nlp(std::shared_ptr<NonLinearProgram> np);

    void set_qp_params() {
#ifdef USE_ACCELERATE_SPARSE
        // Accelerate interface uses different configuration methods
        switch (qp_ord_) {
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
        this->kkt_sol_.set_num_threads(qp_threads_);
        this->kkt_sol_.set_iterative_refinement(qp_ref_steps_ > 0);
        this->kkt_sol_.set_iterative_refinement_iterations(qp_ref_steps_);
        this->kkt_sol_.set_pivot_tolerance(accel_pivot_tolerance_);
        this->kkt_sol_.set_zero_tolerance(accel_zero_tolerance_);
#else
        this->kkt_sol_.ord_ = static_cast<int>(qp_ord_);
        this->kkt_sol_.pivotstrat_ = static_cast<int>(qp_pivot_strategy_);
        this->kkt_sol_.pivotpert_ = qp_pivot_perturb_;
        this->kkt_sol_.matching_ = qp_matching_;
        this->kkt_sol_.scaling_ = qp_scaling_;
        this->kkt_sol_.iterref_ = qp_ref_steps_;
        this->kkt_sol_.alg_ = static_cast<int>(qp_alg_);
        this->kkt_sol_.msglvl_ = qp_print_;

        if (this->cnr_mode_)
            this->kkt_sol_.threads_ = this->qp_threads_;
        this->kkt_sol_.parsolve_ = this->qp_par_solve_;
        this->kkt_sol_.set_params();
#endif
    }

    void set_early_callback(const EarlyCallBackType &f) {
        this->early_callback_enabled_ = true;
        this->early_callback_ = f;
    }
    void disable_early_callback() { this->early_callback_enabled_ = false; }
    void set_late_callback(const LateCallBackType &f) {
        this->late_callback_enabled_ = true;
        this->late_callback_ = f;
    }
    void disable_late_callback() { this->late_callback_enabled_ = false; }
    /////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> get_primals(EigenRef<VectorXd> XSL) const {
        return XSL.head(this->primal_vars_);
    }
    EigenRef<VectorXd> get_slacks(EigenRef<VectorXd> XSL) const {
        return XSL.segment(this->primal_vars_, this->slack_vars_);
    }
    EigenRef<VectorXd> get_primals_slacks(EigenRef<VectorXd> XSL) const {
        return XSL.head(this->primal_vars_ + this->slack_vars_);
    }

    EigenRef<VectorXd> get_lmults(EigenRef<VectorXd> XSL) const {
        return XSL.tail(this->equal_cons_ + this->inequal_cons_);
    }

    EigenRef<VectorXd> get_eq_lmults(EigenRef<VectorXd> XSL) const {
        return XSL.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_);
    }
    EigenRef<VectorXd> get_iq_lmults(EigenRef<VectorXd> XSL) const {
        return XSL.tail(this->inequal_cons_);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> get_prim_grad(EigenRef<VectorXd> GX_or_AGX_FX) const {
        return GX_or_AGX_FX.head(this->primal_vars_);
    }
    EigenRef<VectorXd> get_dual_grad(EigenRef<VectorXd> AGX_FX) const {
        return AGX_FX.segment(this->primal_vars_, this->slack_vars_);
    }
    EigenRef<VectorXd> get_prim_dual_grad(EigenRef<VectorXd> AGX_FX) const {
        return AGX_FX.head(this->primal_vars_ + this->slack_vars_);
    }
    EigenRef<VectorXd> get_eq_cons(EigenRef<VectorXd> AGX_FX) const {
        return AGX_FX.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_);
    }
    EigenRef<VectorXd> get_iq_cons(EigenRef<VectorXd> AGX_FX) const {
        return AGX_FX.tail(this->inequal_cons_);
    }
    EigenRef<VectorXd> get_all_cons(EigenRef<VectorXd> AGX_FX) const {
        return AGX_FX.tail(this->inequal_cons_ + this->equal_cons_);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    void apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> FXI) const {
        for (int i = 0; i < this->slack_vars_; i++) {
            double fxi = FXI[i];
            double si = S[i];
            if (si < neg_slack_reset_) {
                si = neg_slack_reset_;
            }

            if (fxi < 0.0) {
                FXI[i] = 0.0;
                S[i] = std::max(std::abs(fxi), neg_slack_reset_);
            } else {
                FXI[i] += si;
            }
        }
    }
    double max_step_to_boundary(Eigen::Ref<Eigen::VectorXd> SLI, Eigen::Ref<Eigen::VectorXd> dSLI,
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
    void complementarity(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI,
                         double &avgcomp, double &mincomp, double &maxcomp) const {
        Eigen::VectorXd StLI = S.cwiseProduct(LI);
        mincomp = StLI.minCoeff();
        maxcomp = StLI.maxCoeff();
        avgcomp = StLI.sum() / double(StLI.size());
    }

    double barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu) const {
        double psi = 0;
        for (int i = 0; i < this->inequal_cons_; i++) {
            psi += -mu * std::log(S[i]);
        }
        return psi;
    }
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu,
                          Eigen::Ref<Eigen::VectorXd> AGS) const {
        AGS = LI - mu * (S.cwiseInverse());
    }
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> LI, Eigen::Ref<Eigen::VectorXd> AGS) const {
        AGS = LI;
    }

    void barrier_hessian(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                         Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu) {
        Eigen::VectorXd hp = LI.cwiseQuotient(S);
        for (int i = 0; i < this->inequal_cons_; i++) {
            if (hp[i] < 0.0) {
                hp[i] = mu / (S[i] * S[i]);
            }
        }
        this->nlp_->assign_kkt_slack_hessian(hp, KKTmat);
    }

    double loqo_mu(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                   double mincomp) const {
        double eta = mincomp / avgcomp;
        double sigmat = .1 * std::pow(0.05 * (1.0 - eta) / eta, 3);
        double sigma = std::min(0.8, abs(sigmat));
        return sigma * avgcomp;
    }
    double mpc_mu(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                  double mincomp) const {
        double navgcomp = 0;
        double nmincomp = 0;
        double nmaxcomp = 0;
        this->complementarity(S, LI, navgcomp, nmincomp, nmaxcomp);
        return std::pow(navgcomp / avgcomp, 3) * avgcomp;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    void eval_kkt(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
        this->nlp_->eval_kkt(obj_scale, XSL.head(this->primal_vars_),
                             XSL.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_),
                             XSL.tail(this->inequal_cons_), val, this->get_prim_grad(GX),
                             this->get_prim_grad(AGXS_FX), this->get_eq_cons(AGXS_FX),
                             this->get_iq_cons(AGXS_FX), KKTmat);
    }

    void eval_kkt_no(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                     EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                     Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
        this->nlp_->eval_kkt_no(
            obj_scale, XSL.head(this->primal_vars_),
            XSL.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_),
            XSL.tail(this->inequal_cons_), val, this->get_prim_grad(GX),
            this->get_prim_grad(AGXS_FX), this->get_eq_cons(AGXS_FX), this->get_iq_cons(AGXS_FX),
            KKTmat);
    }

    void eval_aug(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
        this->nlp_->eval_aug(obj_scale, XSL.head(this->primal_vars_),
                             XSL.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_),
                             XSL.tail(this->inequal_cons_), val, this->get_prim_grad(GX),
                             this->get_prim_grad(AGXS_FX), this->get_eq_cons(AGXS_FX),
                             this->get_iq_cons(AGXS_FX), KKTmat);
    }

    void eval_soe(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
        this->nlp_->eval_soe(obj_scale, XSL.head(this->primal_vars_),
                             XSL.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_),
                             XSL.tail(this->inequal_cons_), val, this->get_prim_grad(GX),
                             this->get_prim_grad(AGXS_FX), this->get_eq_cons(AGXS_FX),
                             this->get_iq_cons(AGXS_FX), KKTmat);
    }

    void eval_rhs(double obj_scale, const Eigen::Ref<const Eigen::VectorXd> &XSL, double &val,
                  Eigen::Ref<Eigen::VectorXd> GX, Eigen::Ref<Eigen::VectorXd> AGXS_FX) {
        this->nlp_->eval_rhs(obj_scale, XSL.head(this->primal_vars_),
                             XSL.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_),
                             XSL.tail(this->inequal_cons_), val, this->get_prim_grad(GX),
                             this->get_prim_grad(AGXS_FX), this->get_eq_cons(AGXS_FX),
                             this->get_iq_cons(AGXS_FX));
    }

    void max_primal_dual_step(Eigen::Ref<Eigen::VectorXd> XSL, Eigen::Ref<Eigen::VectorXd> DXSL,
                              double bfrac, double &alphap, double &alphad);

    void fill_iter_info(Eigen::Ref<Eigen::VectorXd> XSL, Eigen::Ref<Eigen::VectorXd> RHS,
                        double pobj, double bobj, double mu, IterateInfo &iter) const;

    void eval_nlp(AlgorithmModes algmode, double obj_scale, ConstEigenRef<VectorXd> XSL,
                  double &val, EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    ConvergenceFlags converge_check(std::vector<IterateInfo> &iters);

    static void print_psiopt();

    void print_settings();
    void print_stats();
    void print_last_iterate(const std::vector<IterateInfo> &iters);

    static void print_header() { fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65); }
    void print_beginning(std::string msg) const;
    void print_finished(std::string msg) const;
    void print_exit_stats(ConvergenceFlags ExitCode, const IterateInfo &last, int iternum,
                          double tottime, double nlptime, double qptime, double printtime);

    fmt::text_style calculate_color(double val, double targ, double acc);

    int factor_impl(bool docompute, bool ZFac, double ipurt, double incpurt0, double incpurt,
                    double &finalpert);

    bool analyze_kkt_matrix() {
        bool docompute = true;
        if (this->qp_analyzed_ && !(this->force_qp_analysis_)) {
            docompute = false;
        } else {
            this->qp_analyzed_ = true;
            docompute = true;
        }
        return docompute;
    }

    Eigen::VectorXd alg_impl(AlgorithmModes algmode, BarrierModes barmode, LineSearchModes lsmode,
                             double obj_scale, double MuI, Eigen::Ref<Eigen::VectorXd> xsl);

    Eigen::VectorXd init_impl(const Eigen::VectorXd &x, double Mu, bool docompute);

    double ls_impl(LineSearchModes lsmode, double obj_scale, double Mu, double prim_obj,
                   double barr_obj, EigenRef<VectorXd> XSL, EigenRef<VectorXd> DXSL,
                   EigenRef<VectorXd> XSL2, EigenRef<VectorXd> RHS, EigenRef<VectorXd> RHS2,
                   IterateInfo &Citer, const std::vector<IterateInfo> &iters);

    void ensure_solver_initialized();
    void print_timing_summary();

    Eigen::VectorXd optimize(const Eigen::VectorXd &x);

    Eigen::VectorXd solve_optimize(const Eigen::VectorXd &x);

    Eigen::VectorXd solve_optimize_solve(const Eigen::VectorXd &x);

    Eigen::VectorXd optimize_solve(const Eigen::VectorXd &x);

    Eigen::VectorXd solve(const Eigen::VectorXd &x);
};

} // namespace tycho::solvers
