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
//   - Configuration fields grouped into Settings struct
//   - Converted from struct to class with public/private access sections
// =============================================================================

#pragma once
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <fmt/color.h>
#include <fmt/core.h>

#include "tycho/detail/solvers/iterate_info.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/psiopt_fwd.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/get_core_count.h"

#ifdef USE_ACCELERATE_SPARSE
#include <limits>
#include "tycho/detail/solvers/linear/accelerate_interface.h"
#else
#include "tycho/detail/solvers/linear/pardiso_interface.h"
#endif

namespace tycho::solvers {

// Pull root-namespace Eigen type aliases into tycho::solvers so that PSIOPT
// member declarations (EigenRef<VectorXd>, ConstEigenRef<VectorXd>, …) resolve
// without full qualification inside this namespace.
using tycho::ConstEigenRef;
using tycho::EigenRef;

struct IterateInfo;

class PSIOPT {
  public:
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

    // --- Static string-to-enum converters (defined in psiopt.cpp) ---
    static QPOrderingModes strto_OrderingMode(const std::string &str);
    static LineSearchModes strto_LineSearchMode(const std::string &str);
    static BarrierModes strto_BarrierMode(const std::string &str);
    static BestCriteriaModes strto_BestCriteriaMode(const std::string &str);

    // =========================================================================
    // Settings — all user-configurable parameters grouped in one place
    // =========================================================================
    struct Settings {
        // --- Iteration limits ---
        int max_iters_ = 500;
        int max_ls_iters_ = 2;
        int max_acc_iters_ = 50;
        int max_refac_ = 15;
        int max_soc_ = 1;
        int max_feas_rest_ = 2;

        // --- Convergence tolerances ---
        double kkt_tol_ = 1.0e-6;
        double econ_tol_ = 1.0e-6;
        double icon_tol_ = 1.0e-6;
        double bar_tol_ = 1.0e-6;

        // --- Acceptable tolerances ---
        double acc_kkt_tol_ = 1.0e-2;
        double acc_econ_tol_ = 1.0e-3;
        double acc_icon_tol_ = 1.0e-3;
        double acc_bar_tol_ = 1.0e-3;

        // --- Unacceptable tolerances ---
        double unacc_kkt_tol_ = 10;
        double unacc_econ_tol_ = 2;
        double unacc_icon_tol_ = 2;
        double unacc_bar_tol_ = 2;

        // --- Divergence tolerances ---
        double div_kkt_tol_ = 1.0e15;
        double div_econ_tol_ = 1.0e15;
        double div_icon_tol_ = 1.0e15;
        double div_bar_tol_ = 1.0e15;

        // --- Algorithm modes ---
        AlgorithmModes soe_mode_ = AlgorithmModes::SOE;
        BarrierModes opt_bar_mode_ = BarrierModes::LOQO;
        BarrierModes soe_bar_mode_ = BarrierModes::LOQO;
        LineSearchModes opt_ls_mode_ = LineSearchModes::AUGLANG;
        LineSearchModes soe_ls_mode_ = LineSearchModes::NOLS;
        PDStepStrategies pd_step_strategy_ = PDStepStrategies::PrimSlackEq_Iq;

        // --- Barrier parameters ---
        double init_mu_ = 0.001;
        double max_mu_ = 100.0;
        double min_mu_ = 1.0e-12;

        // --- Step parameters ---
        double bound_fraction_ = 0.99;
        double bound_push_ = 1.0e-3;
        double neg_slack_reset_ = 1.0e-12;
        double soe_bound_relax_ = 1.0e-8;
        double alpha_red_ = 2.0;

        // --- Hessian perturbation ---
        double delta_h_ = 1.0e-5;
        double incr_h_ = 8.0;
        double decr_h_ = 0.333333;

        // --- QP solver ---
        int qp_threads_ = TYCHO_DEFAULT_QP_THREADS;
        QPAlgModes qp_alg_ = QPAlgModes::Classic;
        QPOrderingModes qp_ord_ = QPOrderingModes::METIS;
        QPPivotModes qp_pivot_strategy_ = QPPivotModes::TwoByTwo;
        int qp_matching_ = 1;
        int qp_scaling_ = 0;
        int qp_pivot_perturb_ = 8;
        int qp_ref_steps_ = 0;
        int qp_par_solve_ = 0;
        bool qp_print_ = false;
#ifdef USE_ACCELERATE_SPARSE
        double accel_pivot_tolerance_ = 0.01;
        double accel_zero_tolerance_ = 1e-4 * std::numeric_limits<double>::epsilon();
#endif

        // --- Objective ---
        double obj_scale_ = 1.0;

        // --- Output/behavior ---
        int print_level_ = 0;
        bool wide_console_ = false;
        bool cnr_mode_ = false;
        bool fast_factor_alg_ = true;
        bool force_qp_analysis_ = false;
        bool return_best_ = false;
        BestCriteriaModes best_criteria_ = BestCriteriaModes::ECONS;
    };

    // =========================================================================
    // SolveResult — all output fields produced by a solve/optimize call
    // =========================================================================
    struct SolveResult {
        // --- Solve outcome ---
        int iter_num_ = 0;
        double obj_val_ = 0;
        ConvergenceFlags converge_flag_ = ConvergenceFlags::NOTCONVERGED;

        // --- Solution ---
        Eigen::VectorXd primals_;

        // --- Multipliers and constraints ---
        Eigen::VectorXd eq_lmults_;
        Eigen::VectorXd iq_lmults_;
        Eigen::VectorXd eq_cons_;
        Eigen::VectorXd iq_cons_;

        // --- Timing (seconds) ---
        double total_time_ = 0;
        double pre_time_ = 0;
        double func_time_ = 0;
        double kkt_time_ = 0;
        double print_time_ = 0;
        double misc_time_ = 0;
        double solver_init_time_ = 0;

        // --- Factorization stats ---
        int factor_mem_ = 0;
        int factor_flops_ = 0;

        void zero_timing() {
            total_time_ = 0;
            pre_time_ = 0;
            func_time_ = 0;
            kkt_time_ = 0;
            print_time_ = 0;
            misc_time_ = 0;
            solver_init_time_ = 0;
            iter_num_ = 0;
        }
    };

    // =========================================================================
    // KKTVector — lightweight non-owning view over compound KKT layout
    //   [primals | slacks | eq_lmults | iq_lmults]
    // =========================================================================
    class KKTVector {
      public:
        KKTVector(Eigen::VectorXd &data, int pv, int sv, int ec, int ic)
            : data_(data), pv_(pv), sv_(sv), ec_(ec), ic_(ic) {}

        // --- Primal/slack segments ---
        auto primals() { return data_.head(pv_); }
        auto slacks() { return data_.segment(pv_, sv_); }
        auto primals_slacks() { return data_.head(pv_ + sv_); }

        // --- Multiplier segments ---
        auto eq_lmults() { return data_.segment(pv_ + sv_, ec_); }
        auto iq_lmults() { return data_.tail(ic_); }
        auto lmults() { return data_.tail(ec_ + ic_); }

        // --- Gradient/constraint segments (same layout, different semantics) ---
        auto prim_grad() { return data_.head(pv_); }
        auto dual_grad() { return data_.segment(pv_, sv_); }
        auto prim_dual_grad() { return data_.head(pv_ + sv_); }
        auto eq_cons() { return data_.segment(pv_ + sv_, ec_); }
        auto iq_cons() { return data_.tail(ic_); }
        auto all_cons() { return data_.tail(ec_ + ic_); }

        // --- Full vector access ---
        Eigen::VectorXd &data() { return data_; }
        const Eigen::VectorXd &data() const { return data_; }

      private:
        Eigen::VectorXd &data_;
        int pv_, sv_, ec_, ic_;
    };

    using VectorXd = Eigen::VectorXd;

    using EarlyCallBackType =
        std::function<int(int, double, EigenRef<VectorXd>, double, EigenRef<VectorXd>,
                          EigenRef<VectorXd>, Eigen::SparseMatrix<double, Eigen::RowMajor> &)>;

    using LateCallBackType =
        std::function<int(const IterateInfo &, ConstEigenRef<VectorXd>, ConstEigenRef<VectorXd>)>;

    // --- Constructors ---
    PSIOPT() {
        settings_.qp_threads_ = std::min(TYCHO_DEFAULT_QP_THREADS, tycho::utils::get_core_count());
    }
    PSIOPT(std::shared_ptr<NonLinearProgram> np) {
        settings_.qp_threads_ = std::min(TYCHO_DEFAULT_QP_THREADS, tycho::utils::get_core_count());
        this->set_nlp(np);
    }

    // --- Accessors ---
    Settings &settings() { return settings_; }
    const Settings &settings() const { return settings_; }
    const SolveResult &result() const { return result_; }

    // --- NLP management ---
    void set_nlp(std::shared_ptr<NonLinearProgram> np);
    void release();

    // --- Entry points ---
    Eigen::VectorXd optimize(const Eigen::VectorXd &x);
    Eigen::VectorXd solve(const Eigen::VectorXd &x);
    Eigen::VectorXd solve_optimize(const Eigen::VectorXd &x);
    Eigen::VectorXd optimize_solve(const Eigen::VectorXd &x);
    Eigen::VectorXd solve_optimize_solve(const Eigen::VectorXd &x);

    // --- Validated setter methods (defined in psiopt.cpp) ---
    void set_max_iters(int max_iters);
    void set_max_acc_iters(int max_acc_iters);
    void set_max_ls_iters(int max_ls_iters);
    void set_all_max_iters(int m1, int m2);

    void set_kkt_tol(double kkt_tol);
    void set_bar_tol(double bar_tol);
    void set_econ_tol(double econ_tol);
    void set_icon_tol(double icon_tol);
    void set_tols(double kkt_tol, double econ_tol, double icon_tol, double bar_tol);

    void set_acc_kkt_tol(double acc_kkt_tol);
    void set_acc_bar_tol(double acc_bar_tol);
    void set_acc_econ_tol(double acc_econ_tol);
    void set_acc_icon_tol(double acc_icon_tol);
    void set_acc_tols(double acc_kkt_tol, double acc_econ_tol, double acc_icon_tol,
                      double acc_bar_tol);

    void set_unacc_tols(double kktol, double etol, double itol, double bartol);

    void set_div_kkt_tol(double div_kkt_tol);
    void set_div_bar_tol(double div_bar_tol);
    void set_div_econ_tol(double div_econ_tol);
    void set_div_icon_tol(double div_icon_tol);
    void set_div_tols(double div_kkt_tol, double div_econ_tol, double div_icon_tol,
                      double div_bar_tol);

    void set_bound_fraction(double bound_fraction);
    void set_bound_push(double bound_push);
    void set_alpha_red(double ared);

    void set_delta_h(double delta_h);
    void set_incr_h(double incr_h);
    void set_decr_h(double decr_h);
    void set_hpert_params(double delta_h, double incr_h, double decr_h);

    void set_print_level(int plevel);

    void set_qp_ordering_mode(QPOrderingModes mode);
    void set_qp_ordering_mode(const std::string &str);

    void set_opt_bar_mode(BarrierModes mode);
    void set_opt_bar_mode(const std::string &str);
    void set_soe_bar_mode(BarrierModes mode);
    void set_soe_bar_mode(const std::string &str);

    void set_opt_ls_mode(LineSearchModes mode);
    void set_opt_ls_mode(const std::string &str);
    void set_soe_ls_mode(LineSearchModes mode);
    void set_soe_ls_mode(const std::string &str);

    void set_best_criteria(BestCriteriaModes mode);
    void set_best_criteria(const std::string &str);

#ifdef USE_ACCELERATE_SPARSE
    void set_accel_pivot_tolerance(double tol);
    void set_accel_zero_tolerance(double tol);
#endif

    // --- Callback methods ---
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

    // --- Query methods ---
    ConvergenceFlags get_convergence_flag() const { return result_.converge_flag_; }

    // --- QP parameter setup (defined in psiopt.cpp) ---
    void set_qp_params();

    // --- Printing ---
    static void print_header() { fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65); }

  private:
    Settings settings_;
    SolveResult result_;
    std::shared_ptr<NonLinearProgram> nlp_;

    // --- Problem dimensions ---
    int primal_vars_ = 0;
    int slack_vars_ = 0;
    int equal_cons_ = 0;
    int inequal_cons_ = 0;
    int kkt_dim_ = 0;

    // --- KKT solver ---
#ifdef USE_ACCELERATE_SPARSE
    Eigen::AccelerateLDLTTPP<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> kkt_sol_;
#else
    Eigen::PardisoLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> kkt_sol_;
#endif
    bool qp_analyzed_ = false;

    // --- Callbacks ---
    EarlyCallBackType early_callback_;
    bool early_callback_enabled_ = false;
    LateCallBackType late_callback_;
    bool late_callback_enabled_ = false;

    /// Create a KKTVector view over a VectorXd using this solver's dimensions.
    KKTVector kkt_view(Eigen::VectorXd &v) {
        return KKTVector(v, primal_vars_, slack_vars_, equal_cons_, inequal_cons_);
    }

    // --- Phase sequence ---
    struct PhaseStep {
        AlgorithmModes alg_mode_;
        BarrierModes bar_mode_;
        LineSearchModes ls_mode_;
        const char *label_;
        bool conditional_ = false; // only run if previous phase didn't converge
    };

    Eigen::VectorXd run_phase_sequence(const Eigen::VectorXd &x,
                                       std::initializer_list<PhaseStep> steps);

    // --- Core algorithm (defined in psiopt.cpp) ---
    Eigen::VectorXd alg_impl(AlgorithmModes algmode, BarrierModes barmode, LineSearchModes lsmode,
                             double obj_scale, double MuI, Eigen::Ref<Eigen::VectorXd> xsl);

    Eigen::VectorXd init_impl(const Eigen::VectorXd &x, double Mu, bool docompute);

    // --- Line search (defined in psiopt.cpp) ---
    double ls_impl(LineSearchModes lsmode, double obj_scale, double Mu, double prim_obj,
                   double barr_obj, Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL,
                   Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2,
                   IterateInfo &Citer, const std::vector<IterateInfo> &iters);

    double ls_lang(double obj_scale, double mu, double prim_obj, double barr_obj, KKTVector &xsl,
                   KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                   IterateInfo &citer);

    double ls_l1(double obj_scale, double mu, double prim_obj, double barr_obj, KKTVector &xsl,
                 KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                 IterateInfo &citer);

    double ls_auglang(double obj_scale, double mu, double prim_obj, double barr_obj, KKTVector &xsl,
                      KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                      IterateInfo &citer);

    // --- Line search shared helpers ---
    struct PenaltyTerms {
        double l1_, l2_, linf_;
    };

    void eval_trial_point_occ(double obj_scale, double mu, double alpha, KKTVector &xsl,
                              KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs2, double &ptest,
                              double &btest);

    PenaltyTerms compute_penalties(KKTVector &xsl, KKTVector &rhs) const;

    bool secondary_accept(double ptest, double prim_obj, const PenaltyTerms &test,
                          const PenaltyTerms &init) const;

    // --- KKT factorization (defined in psiopt.cpp) ---
    int factor_impl(bool docompute, bool ZFac, double ipurt, double incpurt0, double incpurt,
                    double &finalpert);

    bool analyze_kkt_matrix();

    void ensure_solver_initialized();

    // --- Barrier math helpers (defined in psiopt.cpp) ---
    void apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> FXI) const;
    double max_step_to_boundary(Eigen::Ref<Eigen::VectorXd> SLI, Eigen::Ref<Eigen::VectorXd> dSLI,
                                double bfrac) const;
    void complementarity(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI,
                         double &avgcomp, double &mincomp, double &maxcomp) const;
    double barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu) const;
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu,
                          Eigen::Ref<Eigen::VectorXd> AGS) const;
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> LI, Eigen::Ref<Eigen::VectorXd> AGS) const;
    void barrier_hessian(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                         Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu);
    double loqo_mu(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                   double mincomp) const;
    double mpc_mu(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                  double mincomp) const;

    // --- NLP eval dispatch methods (defined in psiopt.cpp) ---
    void eval_kkt(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void eval_kkt_no(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                     EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                     Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void eval_aug(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void eval_soe(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void eval_rhs(double obj_scale, const Eigen::Ref<const Eigen::VectorXd> &XSL, double &val,
                  Eigen::Ref<Eigen::VectorXd> GX, Eigen::Ref<Eigen::VectorXd> AGXS_FX);

    void eval_nlp(AlgorithmModes algmode, double obj_scale, ConstEigenRef<VectorXd> XSL,
                  double &val, EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    // --- Convergence and stepping ---
    void fill_iter_info(KKTVector &xsl, KKTVector &rhs, double pobj, double bobj, double mu,
                        IterateInfo &iter) const;
    ConvergenceFlags converge_check(std::vector<IterateInfo> &iters);
    void max_primal_dual_step(KKTVector &xsl, KKTVector &dxsl, double bfrac, double &alphap,
                              double &alphad);

    // --- Printing methods ---
    static void print_psiopt();
    void print_settings();
    void print_stats();
    void print_last_iterate(const std::vector<IterateInfo> &iters);
    void print_beginning(std::string_view msg) const;
    void print_finished(std::string_view msg) const;
    void print_exit_stats(ConvergenceFlags ExitCode, const IterateInfo &last, int iternum,
                          double tottime, double nlptime, double qptime, double printtime);
    void print_timing_summary();
    static fmt::text_style calculate_color(double val, double targ, double acc);
};

} // namespace tycho::solvers
