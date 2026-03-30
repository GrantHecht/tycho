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

void tycho::solvers::PSIOPT::ensure_solver_initialized() {
    double initMs = ::tycho::solvers::ensure_solver_initialized();
    if (initMs > 0.0) {
        this->last_solver_init_time_ = initMs / 1000.0;
        // Suppress the init line when init was trivially fast (< 0.5 ms),
        // which also covers subsequent calls (return 0.0).
        constexpr double kSolverInitPrintThresholdMs = 0.5;
        if (initMs > kSolverInitPrintThresholdMs && this->print_level_ < 2) {
            fmt::print(" Solver Initialization : ");
            fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", initMs);
        }
    }
}

void tycho::solvers::PSIOPT::print_timing_summary() {
    auto cyan = fmt::fg(fmt::color::cyan);
    fmt::print(" KKT Analysis/Init Time       : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->last_pre_time_ * 1000.0);
    fmt::print(" NLP Function Evaluation Time : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->last_func_time_ * 1000.0);
    fmt::print(" KKT Factor/Solve Time        : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->last_kkt_time_ * 1000.0);
    fmt::print(" Console Print Time           : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->last_print_time_ * 1000.0);
    fmt::print(" Misc Time                    : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->last_misc_time_ * 1000.0);
}

void tycho::solvers::PSIOPT::set_nlp(std::shared_ptr<NonLinearProgram> np) {
    this->nlp = np;
    this->primal_vars_ = this->nlp->primal_vars_;
    this->equal_cons_ = this->nlp->equal_cons_;
    this->inequal_cons_ = this->nlp->inequal_cons_;
    this->slack_vars_ = this->nlp->slack_vars_;
    this->kkt_dim_ = this->nlp->kkt_dim_;
    this->set_qp_params();
#ifdef USE_ACCELERATE_SPARSE
    accelerate_set_num_threads(qp_threads_);
#else
    mkl_set_num_threads(qp_threads_);
#endif

    this->nlp->analyze_sparsity(this->kkt_sol_.getMatrix());
#ifdef USE_ACCELERATE_SPARSE
    // we need to call this to update the internal AccelSparseMatrix since
    // we changed the sparsity pattern via the reference returned from getMatrix.
    this->kkt_sol_.reinitializeInternalMatrixRepresentation();
#endif
    if (store_sp_mat_)
        spmat = this->kkt_sol_.getMatrix();
    this->qp_analyzed_ = false;
}

void tycho::solvers::PSIOPT::max_primal_dual_step(Eigen::Ref<Eigen::VectorXd> XSL,
                                                  Eigen::Ref<Eigen::VectorXd> DXSL, double bfrac,
                                                  double &alphap, double &alphad) {
    double Smax = this->max_step_to_boundary(this->get_slacks(XSL), this->get_slacks(DXSL), bfrac);
    double Lmax =
        this->max_step_to_boundary(this->get_iq_lmults(XSL), this->get_iq_lmults(DXSL), bfrac);

    double primstep = Smax;
    double slackstep = Smax;
    double eqmultstep = Smax;
    double iqmultstep = Lmax;

    if (this->pd_step_strategy_ == PDStepStrategies::PrimSlackEq_Iq) {
    } else if (this->pd_step_strategy_ == PDStepStrategies::AllMinimum) {
        double step = std::min(Smax, Lmax);
        primstep = step;
        slackstep = step;
        eqmultstep = step;
        iqmultstep = step;
    } else if (this->pd_step_strategy_ == PDStepStrategies::PrimSlack_EqIq) {
        eqmultstep = Lmax;
    } else if (this->pd_step_strategy_ == PDStepStrategies::MaxEq) {
        double step = std::max(Smax, Lmax);
        eqmultstep = step;
    }
    this->get_primals(DXSL) *= primstep;
    if (inequal_cons_ > 0)
        this->get_slacks(DXSL) *= slackstep;
    if (equal_cons_ > 0)
        this->get_eq_lmults(DXSL) *= eqmultstep;
    if (inequal_cons_ > 0)
        this->get_iq_lmults(DXSL) *= iqmultstep;

    alphap = Smax;
    alphad = Lmax;
}

void tycho::solvers::PSIOPT::fill_iter_info(Eigen::Ref<Eigen::VectorXd> XSL,
                                            Eigen::Ref<Eigen::VectorXd> RHS, double pobj,
                                            double bobj, double mu, IterateInfo &iter) const {

    iter.PrimObj = pobj;
    iter.BarrObj = bobj;
    iter.Mu = mu;
    iter.PPivots = this->kkt_sol_.ppivs();
    iter.KKTInf = this->get_prim_grad(RHS).lpNorm<Eigen::Infinity>();

    double avgcomp = 0;
    double mincomp = 0;
    double maxcomp = 0;
    if (inequal_cons_ > 0) {
        iter.IConInf = this->get_iq_cons(RHS).lpNorm<Eigen::Infinity>();
        iter.IConNormErr = this->get_iq_cons(RHS).norm();
        iter.MaxIMult = this->get_iq_lmults(XSL).lpNorm<Eigen::Infinity>();
        this->complementarity(this->get_slacks(XSL), this->get_iq_lmults(XSL), avgcomp, mincomp,
                              maxcomp);

        iter.BarrInf = maxcomp;
        iter.BarrNormErr = avgcomp;
    }
    if (equal_cons_ > 0) {
        iter.EConInf = this->get_eq_cons(RHS).lpNorm<Eigen::Infinity>();
        iter.EConNormErr = this->get_eq_cons(RHS).norm();
        iter.MaxEMult = this->get_eq_lmults(XSL).lpNorm<Eigen::Infinity>();
    }

    iter.KKTNormErr = this->get_prim_grad(RHS).norm();

    if (equal_cons_ > 0 || inequal_cons_ > 0)
        iter.AllConNormErr = this->get_all_cons(RHS).norm();
}

void tycho::solvers::PSIOPT::eval_nlp(AlgorithmModes algmode, double obj_scale_,
                                      ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    std::fill_n(KKTmat.valuePtr(), KKTmat.nonZeros(), 0.0);

    switch (algmode) {
    case AlgorithmModes::OPT:
        eval_kkt(obj_scale_, XSL, val, GX, AGXS_FX, KKTmat);
        break;
    case AlgorithmModes::OPTNO:
        eval_kkt_no(obj_scale_, XSL, val, GX, AGXS_FX, KKTmat);

        break;
    case AlgorithmModes::INIT:
        eval_aug(obj_scale_, XSL, val, GX, AGXS_FX, KKTmat);
        break;
    case AlgorithmModes::SOE:
        this->nlp->set_primal_diags(1.0);
        eval_soe(0.0, XSL, val, GX, AGXS_FX, KKTmat);
        this->nlp->set_primal_diags(0.0);
        this->get_prim_grad(GX).setZero();
        this->get_prim_grad(AGXS_FX).setZero();
        break;
    default:
        throw std::invalid_argument("Unknown AlgorithmMode");
    }
}

tycho::ConvergenceFlags tycho::solvers::PSIOPT::converge_check(std::vector<IterateInfo> &iters) {
    ConvergenceFlags Flag = ConvergenceFlags::CONVERGED;
    IterateInfo last = iters.back();
    bool KKTFeas = (last.KKTInf < this->kkt_tol_);
    bool EConFeas = (last.EConInf < this->econ_tol_);
    bool IConFeas = (last.IConInf < this->icon_tol_);
    bool BarFeas = (last.BarrInf < this->bar_tol_);

    bool KKTDiv = (last.KKTInf > this->div_kkt_tol_) || !std::isfinite(last.KKTInf);
    bool EConDiv = (last.EConInf > this->div_econ_tol_) || !std::isfinite(last.EConInf);
    bool IConDiv = (last.IConInf > this->div_icon_tol_) || !std::isfinite(last.IConInf);
    bool BarDiv = (last.BarrInf > this->div_bar_tol_) || !std::isfinite(last.BarrInf);

    if (KKTDiv || EConDiv || IConDiv || BarDiv) {
        Flag = ConvergenceFlags::DIVERGING;
        return Flag;
    } else if (KKTFeas && EConFeas && IConFeas && BarFeas) {
        Flag = ConvergenceFlags::CONVERGED;
        return Flag;
    } else if (int(iters.size()) > this->max_acc_iters_) {
        int nfeas = 0;
        for (int i = 0; i < this->max_acc_iters_; i++) {
            last = iters[int(iters.size()) - i - 1];
            KKTFeas = (last.KKTInf < this->acc_kkt_tol_);
            EConFeas = (last.EConInf < this->acc_econ_tol_);
            IConFeas = (last.IConInf < this->acc_icon_tol_);
            BarFeas = (last.BarrInf < this->acc_bar_tol_);
            if (KKTFeas && EConFeas && IConFeas && BarFeas)
                nfeas++;
            else
                break;
        }
        if (nfeas == this->max_acc_iters_) {
            Flag = ConvergenceFlags::ACCEPTABLE;
            return Flag;
        }
    }
    Flag = ConvergenceFlags::NOTCONVERGED;
    return Flag;
}

void tycho::solvers::PSIOPT::print_psiopt() {

    constexpr const char *PsioptStr =
        "       ____    _____    ____          ____     ____   ______\n"
        "      / __ \\  / ___/   /  _/         / __ \\   / __ \\ /_  __/\n"
        "     / /_/ /  \\__ \\    / /   ______ / / / /  / /_/ /  / /   \n"
        "    / ____/  ___/ /  _/ /   /_____// /_/ /  / ____/  / /    \n"
        "   /_/      /____/  /___/          \\____/  /_/      /_/    \n";
    print_header();
    fmt::print(fmt::fg(fmt::color::crimson), PsioptStr);
    fmt::print(fmt::fg(fmt::color::crimson),
               " \n       Parallel Sparse Interior-point Optimizer\n");
    print_header();
}

void tycho::solvers::PSIOPT::print_settings() {
    using std::cout;
    using std::endl;

    auto cyan = fmt::fg(fmt::color::cyan);
    auto magenta = fmt::fg(fmt::color::magenta);

    fmt::print(magenta, "Convergence Criteria\n\n");

    fmt::print("{0:_^{1}}\n", "", 39);
    fmt::print("|------|   tol   | Acctol  | Divtol  |\n");
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "KKT", this->kkt_tol_,
               this->acc_kkt_tol_, this->div_kkt_tol_);
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "Bar", this->bar_tol_,
               this->acc_bar_tol_, this->div_bar_tol_);
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "ECons", this->econ_tol_,
               this->acc_econ_tol_, this->div_econ_tol_);
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "ICons", this->icon_tol_,
               this->acc_icon_tol_, this->div_icon_tol_);
}

void tycho::solvers::PSIOPT::print_matrixinfo() {}

void tycho::solvers::PSIOPT::print_stats() {
    print_psiopt();

    auto cyan = fmt::fg(fmt::color::cyan);
    auto magenta = fmt::fg(fmt::color::magenta);

    fmt::print(magenta, "Problem Statistics\n\n");

    fmt::print(" Primal Variables         : ");
    fmt::print(cyan, "{:<10}\n", this->primal_vars_);
    fmt::print(" Equality Constraints     : ");
    fmt::print(cyan, "{:<10}\n", this->equal_cons_);
    fmt::print(" Inequality Constraints   : ");
    fmt::print(cyan, "{:<10}\n", this->inequal_cons_);
    fmt::print("\n");
    fmt::print(" KKT-Matrix DIM (P+E+2*I) : ");
    fmt::print(cyan, "{:<10}\n", this->kkt_dim_);
    fmt::print(" KKT-Matrix NNZs          : ");
    fmt::print(cyan, "{:<10}\n", this->kkt_sol_.getMatrix().nonZeros());
    fmt::print(" KKT-Matrix NNZ%          : ");
    fmt::print(cyan, "{:.6f}%\n",
               100.0 * double(this->kkt_sol_.getMatrix().nonZeros()) /
                   (double(this->kkt_dim_) * double(this->kkt_dim_)));
    fmt::print("\n");
}

void tycho::solvers::PSIOPT::print_last_iterate(const std::vector<IterateInfo> &iters) {
    auto last = iters.back();
    bool wide = false;

    if (last.iter % 10 == 0) {
        if (wide_console_) {
            fmt::print("{0:=^{1}}\n", "", 159);
            fmt::print(
                "|Iter| Mu Val | Prim Obj |  Bar Obj |  KKT Inf |  Bar Inf | ECons Inf| ICons "
                "Inf|Max "
                "EMult|Max IMult| AlphaP | AlphaD | AlphaT | Merit Val|LSI|PPS|HFI| HPert |\n");
        } else {
            fmt::print("{0:=^{1}}\n", "", 119);
            fmt::print("|Iter| Mu Val | Prim Obj |  Bar Obj |  KKT Inf |  Bar Inf | ECons Inf| "
                       "ICons Inf| AlphaP | "
                       "AlphaD |LS| PPS |HF| HPert |\n");
        }
        auto tst =
            "|Iter|Mu Val | Prim Obj |KKT Inf |ECon Inf|ICon Inf|AlphaM |LS|PPS|HF| HPert |\n";
    }

    fmt::text_style PHashcol = fmt::text_style();
    fmt::text_style EHashcol = fmt::text_style();
    fmt::text_style IHashcol = fmt::text_style();
    fmt::text_style KHashcol = fmt::text_style();
    fmt::text_style BHashcol = fmt::text_style();
    fmt::text_style BOHashcol = fmt::text_style();

    fmt::text_style Kcol = calculate_color(last.KKTInf, this->kkt_tol_, this->acc_kkt_tol_);
    fmt::text_style Bcol = calculate_color(last.BarrInf, this->bar_tol_, this->acc_bar_tol_);
    fmt::text_style Ecol = calculate_color(last.EConInf, this->econ_tol_, this->acc_econ_tol_);
    fmt::text_style Icol = calculate_color(last.IConInf, this->icon_tol_, this->acc_icon_tol_);

    if (iters.size() > 1) {

        auto GCol = fmt::fg(fmt::color::lime_green);
        auto BCol = fmt::fg(fmt::color::red);

        PHashcol = (iters.back().PrimObj <= iters[iters.size() - 2].PrimObj) ? GCol : BCol;
        BOHashcol = (iters.back().BarrObj <= iters[iters.size() - 2].BarrObj) ? GCol : BCol;
        BHashcol = (iters.back().BarrInf <= iters[iters.size() - 2].BarrInf) ? GCol : BCol;
        EHashcol = (iters.back().EConInf <= iters[iters.size() - 2].EConInf) ? GCol : BCol;
        IHashcol = (iters.back().IConInf <= iters[iters.size() - 2].IConInf) ? GCol : BCol;
        KHashcol = (iters.back().KKTInf <= iters[iters.size() - 2].KKTInf) ? GCol : BCol;
    }

    auto hash = []() { fmt::print("|"); };
    auto chash = [](fmt::text_style c) { fmt::print(c, "|"); };

    hash();
    fmt::print("{:<4}", last.iter);
    hash();
    fmt::print("{:.2e}", last.Mu);
    hash();
    fmt::print("{:>10.3e}", last.PrimObj);
    chash(PHashcol);
    fmt::print("{:>10.3e}", last.BarrObj);
    chash(BOHashcol);
    fmt::print(Kcol, "{:>10.4e}", last.KKTInf);
    chash(KHashcol);
    fmt::print(Bcol, "{:>10.4e}", last.BarrInf);
    chash(BHashcol);
    fmt::print(Ecol, "{:>10.4e}", last.EConInf);
    chash(EHashcol);
    fmt::print(Icol, "{:>10.4e}", last.IConInf);
    chash(IHashcol);

    if (wide_console_) {
        fmt::print(
            "{:>9.3e}|{:>9.3e}|{:>8.2e}|{:>8.2e}|{:>8.2e}|{:>10.3e}|{:>3}|{:>3}|{:>3}|{:>6.1e}|\n",
            last.MaxEMult, last.MaxIMult, last.alphaP, last.alphaD, last.alphaT, last.MeritVal,
            last.LSiters, last.PPivots, last.Hfacs, last.Hpert);
    } else {
        fmt::print("{:>8.2e}|{:>8.2e}|{:>2}|{:>5}|{:>2}|{:>6.1e}|\n", last.alphaT * last.alphaP,
                   last.alphaT * last.alphaD, last.LSiters, last.PPivots, last.Hfacs, last.Hpert);
    }
}

void tycho::solvers::PSIOPT::print_beginning(std::string msg) const {
    fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "{}", msg);
    fmt::print("\n");
}

void tycho::solvers::PSIOPT::print_finished(std::string msg) const {

    fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "{}", msg);
    fmt::print("\n");
}

void tycho::solvers::PSIOPT::print_exit_stats(ConvergenceFlags ExitCode, const IterateInfo &last,
                                              int iternum, double tottime, double nlptime,
                                              double qptime, double printtime) {
    fmt::text_style Kcol = calculate_color(last.KKTInf, this->kkt_tol_, this->acc_kkt_tol_);
    fmt::text_style Bcol = calculate_color(last.BarrInf, this->bar_tol_, this->acc_bar_tol_);
    fmt::text_style Ecol = calculate_color(last.EConInf, this->econ_tol_, this->acc_econ_tol_);
    fmt::text_style Icol = calculate_color(last.IConInf, this->icon_tol_, this->acc_icon_tol_);

    auto TColor = fmt::fg(fmt::color::cyan);
    auto Printtime = [&](const char *msg, double t1) {
        fmt::print("{}", msg);
        fmt::print(TColor, "{0:>10.3f} ms {1:>10.3f} ms/iter\n", t1, double(t1 / iternum));
    };

    if (this->print_level_ < 3) {
        if (ExitCode == ConvergenceFlags::CONVERGED) {
            fmt::print(fmt::fg(fmt::color::lime_green), "\nOptimal Solution Found\n");
        } else if (ExitCode == ConvergenceFlags::ACCEPTABLE) {
            fmt::print(fmt::fg(fmt::color::yellow), "\nAcceptable Solution Found\n");
        } else if (ExitCode == ConvergenceFlags::DIVERGING) {
            fmt::print(fmt::fg(fmt::color::dark_red), "\nSolution Diverging\n");
        } else if (ExitCode == ConvergenceFlags::NOTCONVERGED) {
            fmt::print(fmt::fg(fmt::color::red), "\nNo Solution Found\n");
        }
    }

    if (this->print_level_ < 2) {

        fmt::print(" Iterations : ");
        fmt::print("{:<5}\n", iternum);
        fmt::print(" Prim Obj   : ");
        fmt::print("{:<15.8e}\n", last.PrimObj);
        fmt::print(" KKT Inf    : ");
        fmt::print(Kcol, "{:<15.8e}\n", last.KKTInf);
        fmt::print(" Bar Inf    : ");
        fmt::print(Bcol, "{:<15.8e}\n", last.BarrInf);
        fmt::print(" ECons Inf  : ");
        fmt::print(Ecol, "{:<15.8e}\n", last.EConInf);
        fmt::print(" ICons Inf  : ");
        fmt::print(Icol, "{:<15.8e}\n", last.IConInf);

        fmt::print("\n");

        Printtime(" NLP Function Evaluation Time : ", nlptime);
        Printtime(" KKT Matrix Factor/Solve Time : ", qptime);
        Printtime(" Console Print Time           : ", printtime);
        Printtime(" Total Time                   : ", tottime);

        fmt::print("\n");
    }
}

fmt::text_style tycho::solvers::PSIOPT::calculate_color(double val, double targ, double acc) {
    auto level1 = std::log(targ);
    auto level3 = std::log(acc);
    auto level5 = std::log(acc * 1000.0);
    auto level2 = (level1 + level3) / 2.0;
    auto level4 = (level3 + level5) / 2.0;

    auto logval = std::log(val);
    fmt::color c;

    if (logval < level1)
        c = fmt::color::lime_green;
    else if (logval < level2)
        c = fmt::color::yellow;
    else if (logval < level3)
        c = fmt::color::orange;
    else if (logval < level4)
        c = fmt::color::red;
    else
        c = fmt::color::dark_red;
    return fmt::fg(c);
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
    auto Perturb = [&](double p) { this->nlp->perturb_kkt_p_diags(p, this->kkt_sol_.getMatrix()); };
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

    for (int i = 0; i < this->max_refac_; i++) {
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
    return this->max_refac_;
}

Eigen::VectorXd tycho::solvers::PSIOPT::alg_impl(AlgorithmModes algmode, BarrierModes barmode,
                                                 LineSearchModes lsmode, double obj_scale_,
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

    double Mu = MuI;

    tycho::utils::Timer Runtimer;
    tycho::utils::Timer Funtimer;
    tycho::utils::Timer LStimer;
    tycho::utils::Timer QPtimer;
    tycho::utils::Timer
        CBtimer; // Callback time is included in last_misc_time_ (not separately reported)
    tycho::utils::Timer Printtimer;

    double Hpert0 = this->delta_h_;
    std::vector<IterateInfo> iters;
    iters.reserve(this->max_iters_);
    ConvergenceFlags ExitCode;
    bool FirstPert = true;

    Runtimer.start();
    for (int i = 0; i < this->max_iters_; i++) {
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
        double PrimObj = 0;
        double BarrObj = 0;

        Funtimer.start();
        /////////////////////////////////////////////////////////////

        this->eval_nlp(algmode, obj_scale_, XSL, PrimObj, PGX, RHS, this->kkt_sol_.getMatrix());

        if (this->inequal_cons_ > 0) {
            this->apply_reset_slacks(this->get_slacks(XSL), this->get_iq_cons(RHS));
            this->barrier_hessian(this->kkt_sol_.getMatrix(), this->get_slacks(XSL),
                                  this->get_iq_lmults(XSL), Mu);
            this->complementarity(this->get_slacks(XSL), this->get_iq_lmults(XSL), avgcomp, mincomp,
                                  maxcomp);
        }

        ///////////////////////////////////////////////////////////////
        Funtimer.stop();
        if (this->early_callback_enabled_) {
            CBtimer.start();
            this->early_callback_(i, obj_scale_, XSL, PrimObj, PGX, RHS,
                                  this->kkt_sol_.getMatrix());
            CBtimer.stop();
        }
        QPtimer.start();
        ////////////////////////////////////////////////////////////////
        RHS.head(this->primal_vars_) += PGX;

        ////////////////////////////////////////////////////////////////
        double nhpert = 0;
        double Incr = this->incr_h_;
        double Incr2 = this->incr_h_;
        if (FirstPert)
            Incr2 *= this->incr_h_;
        bool Zfac = true;
        if (this->fast_factor_alg_ && i > 6 && ((i * 3) % 4) != 0) {
            bool cycling = true;
            for (int j = 0; j < 4; j++) {
                int ns = iters[iters.size() - 1 - j].Hfacs;
                if (ns == 0) {
                    cycling = false;
                    break;
                }
            }
            Zfac = !cycling;
        }

        Citer.Hfacs = this->factor_impl(false, Zfac, Hpert0, Incr, Incr2, nhpert);

        if (Citer.Hfacs > 0) {
            Hpert0 = std::max(this->delta_h_, nhpert * decr_h_);
            FirstPert = false;
        }
        Citer.Hpert = nhpert;
        ///////////////////////////////////////////////////////////////////

        if (this->inequal_cons_ > 0) {
            switch (barmode) {
            case BarrierModes::PROBE:
                this->barrier_gradient(this->get_iq_lmults(XSL), this->get_dual_grad(RHS));
                DXSL = -this->kkt_sol_.solve(RHS);
                this->max_primal_dual_step(XSL, DXSL, this->bound_fraction_, alphap, alphad);
                Temp = XSL + DXSL;
                Mu = this->mpc_mu(this->get_slacks(Temp), this->get_iq_lmults(Temp), avgcomp,
                                  mincomp);

                break;
            case BarrierModes::LOQO:
                Mu = this->loqo_mu(this->get_slacks(XSL), this->get_iq_lmults(XSL), avgcomp,
                                   mincomp);
                break;
            case BarrierModes::FIACCO:
                break;
            default:
                break;
            }

            Mu = std::max(Mu, this->min_mu_);
            Mu = std::min(Mu, this->max_mu_);
            BarrObj = this->barrier_objective(this->get_slacks(XSL), Mu);
            this->barrier_gradient(this->get_slacks(XSL), this->get_iq_lmults(XSL), Mu,
                                   this->get_dual_grad(RHS));
        }

        DXSL = -this->kkt_sol_.solve(RHS);
        bool GoodStep = std::isfinite(DXSL.squaredNorm());
        if (this->inequal_cons_ > 0)
            this->max_primal_dual_step(XSL, DXSL, this->bound_fraction_, alphap, alphad);
        /////////////////////////////////////////////////////////////////////
        if (diagnostic_) {
        }
        QPtimer.stop();

        Funtimer.start();
        //////////////////////////////////////////////////////////////////////

        if (GoodStep) {
            double lsobjscale =
                algmode == AlgorithmModes::SOE || algmode == AlgorithmModes::OPTNO ? 0.0 : 1.0;
            alpha = ls_impl(lsmode, obj_scale_ * lsobjscale, Mu, PrimObj, BarrObj, XSL, DXSL, Temp,
                            RHS, RHS2, Citer, iters);

        } else {
            Citer.Hfacs = -1;
        }

        //////////////////////////////////////////////////////////////////////
        Funtimer.stop();

        Citer.alphaP = alphap;
        Citer.alphaD = alphad;
        Citer.alphaT = alpha;

        this->fill_iter_info(XSL, RHS, PrimObj, BarrObj, Mu, Citer);
        iters.push_back(Citer);

        if (this->return_best_) {
            double critval;
            switch (this->best_criteria_) {
            case BestCriteriaModes::ECONS:
                critval = iters.back().EConInf;
                break;
            case BestCriteriaModes::ICONS:
                critval = iters.back().IConInf;
                break;
            case BestCriteriaModes::KKT:
                critval = iters.back().KKTInf;
                break;
            case BestCriteriaModes::OBJ:
                critval = iters.back().PrimObj;
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

        if (this->print_level_ == 0) {
            Printtimer.start();
            this->print_last_iterate(iters);
            Printtimer.stop();
        }

        if (ExitCode == ConvergenceFlags::CONVERGED || ExitCode == ConvergenceFlags::ACCEPTABLE ||
            ExitCode == ConvergenceFlags::DIVERGING || i == (this->max_iters_ - 1)) {

            if (ExitCode != ConvergenceFlags::CONVERGED && this->return_best_) {
                XSL = BestXSL;
                RHS = BestRHS;
            }

            this->converge_flag_ = ExitCode;
            break;
        }

        /////////Very Important/////////
        XSL += alpha * DXSL;
        ///////////////////////////////
    }

    if (algmode == AlgorithmModes::OPT) {
        this->last_obj_val_ = iters.back().PrimObj;
    } else {
        Funtimer.start();
        this->last_obj_val_ = 0;
        this->nlp->eval_obj(obj_scale_, XSL.head(this->primal_vars_), this->last_obj_val_);
        Funtimer.stop();
    }

    if (this->equal_cons_ > 0) {
        this->last_eq_cons_ = this->get_eq_cons(RHS);
        this->last_eq_lmults_ = this->get_eq_lmults(XSL);
    }
    if (this->inequal_cons_ > 0) {
        this->last_iq_cons_ = this->get_iq_cons(RHS) - this->get_slacks(XSL);
        this->last_iq_lmults_ = this->get_iq_lmults(XSL);
    }

    Runtimer.stop();
    this->last_iter_num_ += iters.size();
    double qptime = double(QPtimer.count<std::chrono::microseconds>()) / 1000000.0;
    double nlptime = double(Funtimer.count<std::chrono::microseconds>()) / 1000000.0;
    double tottime = double(Runtimer.count<std::chrono::microseconds>()) / 1000000.0;

    this->last_func_time_ += nlptime;
    this->last_kkt_time_ += qptime;
    double printtime = double(Printtimer.count<std::chrono::microseconds>()) / 1000000.0;
    this->last_print_time_ += printtime;

    ///////////////////////////////////////////////////////////////////////////

    int retiter = (this->return_best_ ? BestIter : iters.size() - 1);
    print_exit_stats(ExitCode, iters[retiter], iters.size(), tottime * 1000, nlptime * 1000,
                     qptime * 1000, printtime * 1000);
    ////////////////////////////////////////////////////////////////////////////

    return XSL;
}

Eigen::VectorXd tycho::solvers::PSIOPT::init_impl(const Eigen::VectorXd &x, double Mu,
                                                  bool docompute) {

    tycho::utils::Timer kktt;
    kktt.start();

    Eigen::VectorXd XSL(this->kkt_dim_);
    XSL.setZero();
    XSL.head(this->primal_vars_) = x;

    Eigen::VectorXd RHS(this->kkt_dim_);
    RHS.setZero();
    double val = 0;
    this->nlp->set_primal_diags(1.0);
    if (this->inequal_cons_ > 0) {
        this->nlp->set_slacks_ones();
    }
    this->eval_nlp(AlgorithmModes::INIT, this->obj_scale_, XSL, val, RHS.head(this->primal_vars_),
                   RHS, this->kkt_sol_.getMatrix());

    Eigen::VectorXd hp(this->slack_vars_);

    for (int i = 0; i < this->slack_vars_; i++) {
        double fxi = this->get_iq_cons(RHS)[i];
        if (fxi < -this->bound_push_) {
            this->get_slacks(XSL)[i] = abs(fxi);
        } else {
            this->get_slacks(XSL)[i] = this->bound_push_;
        }
        hp[i] = 1.0;
        this->get_iq_lmults(XSL)[i] = Mu / this->get_slacks(XSL)[i];
    }

    RHS.tail(this->equal_cons_ + this->inequal_cons_).setZero();

    if (this->inequal_cons_ > 0)
        this->nlp->assign_kkt_slack_hessian(hp, this->kkt_sol_.getMatrix());
    if (this->print_level_ < 2) {
        print_beginning("KKT-Matrix Analysis ");
    }

    if (docompute)
        this->kkt_sol_.compute_internal();
    else
        this->kkt_sol_.refactorize_internal();
    kktt.stop();

    double pretime = double(kktt.count<std::chrono::microseconds>()) / 1000000.0;
    this->last_pre_time_ += pretime;

    this->factor_flops_ = this->kkt_sol_.m_flops;
    this->factor_mem_ = this->kkt_sol_.m_mem;

    if (this->print_level_ < 2) {
        auto cyan = fmt::fg(fmt::color::cyan);
        if (docompute) {
            fmt::print(" LDLT Factor Size      : ");
            fmt::print(cyan, "{0:<10}\n", this->factor_mem_);
            if (this->factor_flops_ > 0) {
                fmt::print(" LDLT Factor FLOPs     : ");
                fmt::print(cyan, "{0} MFLOPs\n", this->factor_flops_);
            }
        }
        fmt::print(" Analysis/Reorder Time : ");
        fmt::print(cyan, "{0:.3f} ms\n", pretime * 1000);
        print_finished("KKT-Matrix Analysis ");
    }

    Eigen::VectorXd dx = -this->kkt_sol_.solve(RHS);

    if (equal_cons_ > 0)
        this->get_eq_lmults(XSL) = this->get_eq_lmults(dx);
    if (this->inequal_cons_ > 0)
        this->nlp->set_slack_diags(0.0);
    this->nlp->set_primal_diags(0.0);

    return XSL;
}

double tycho::solvers::PSIOPT::ls_impl(LineSearchModes lsmode, double obj_scale_, double Mu,
                                       double PrimObj, double BarrObj, EigenRef<VectorXd> XSL,
                                       EigenRef<VectorXd> DXSL, EigenRef<VectorXd> XSL2,
                                       EigenRef<VectorXd> RHS, EigenRef<VectorXd> RHS2,
                                       IterateInfo &Citer, const std::vector<IterateInfo> &iters) {

    // Do not modify RHS,XSL,DXSL. EigenRef<VectorXd> doesnt like to be explicitly const in this
    // instance, I will fix this later in refactor

    double alpha = 1.0;

    if (lsmode == LineSearchModes::LANG) {
        double LangInit = PrimObj + BarrObj + this->get_lmults(XSL).dot(this->get_all_cons(RHS));

        for (int j = 0; j < this->max_ls_iters_; j++) {
            double ptest = 0;
            double btest = 0;
            XSL2 = XSL + alpha * DXSL;
            RHS2.setZero();
            this->eval_rhs(obj_scale_, XSL2, ptest, RHS2, RHS2);
            this->apply_reset_slacks(this->get_slacks(XSL2), this->get_iq_cons(RHS2));
            btest = this->barrier_objective(this->get_slacks(XSL2), Mu);
            this->barrier_gradient(this->get_slacks(XSL2), this->get_iq_lmults(XSL2), Mu,
                                   this->get_dual_grad(RHS2));
            double LangTest = ptest + btest + this->get_lmults(XSL2).dot(this->get_all_cons(RHS2));
            Citer.LSiters = j;
            if (LangTest < LangInit) {
                break;
            } else {
                alpha = alpha / this->alpha_red_;
            }
        }
    } else if (lsmode == LineSearchModes::L1) {
        double vv = RHS.head(this->primal_vars_ + this->slack_vars_)
                        .dot(DXSL.head(this->primal_vars_ + this->slack_vars_));
        double cv = this->get_lmults(DXSL).dot(this->get_all_cons(RHS));

        double LangInit = PrimObj + BarrObj;
        double InitL1Pen = this->get_lmults(XSL).cwiseAbs().dot(this->get_all_cons(RHS).cwiseAbs());
        double InitL2Pen = this->get_all_cons(RHS).squaredNorm();
        double InitLinfPenalty = this->get_all_cons(RHS).lpNorm<Eigen::Infinity>();

        double sc = .1 + std::abs(vv - cv) / InitL2Pen;
        if (InitL2Pen == 0.0)
            sc = 1.0;

        LangInit += InitL1Pen + InitL2Pen * sc;

        for (int j = 0; j < this->max_ls_iters_; j++) {
            double ptest = 0;
            double btest = 0;
            XSL2 = XSL + alpha * DXSL;
            RHS2.setZero();
            this->nlp->eval_occ(
                obj_scale_, XSL2.head(this->primal_vars_), ptest,
                RHS2.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_),
                RHS2.tail(this->inequal_cons_));

            this->apply_reset_slacks(this->get_slacks(XSL2), this->get_iq_cons(RHS2));
            btest = this->barrier_objective(this->get_slacks(XSL2), Mu);

            double LangTest = ptest + btest;
            double TestL1Pen =
                this->get_lmults(XSL).cwiseAbs().dot(this->get_all_cons(RHS2).cwiseAbs());
            double TestL2Pen = this->get_all_cons(RHS2).squaredNorm();
            double TestLinfPenalty = this->get_all_cons(RHS2).lpNorm<Eigen::Infinity>();

            LangTest += TestL1Pen + TestL2Pen * sc;

            Citer.MeritVal = LangTest;
            if (LangTest < LangInit || (ptest < PrimObj) && (TestL2Pen < InitL2Pen) ||
                (ptest < PrimObj) && (TestLinfPenalty < InitLinfPenalty)) {
                Citer.LSiters = j;
                break;
            } else {
                Citer.LSiters = j + 1;
                alpha = alpha / this->alpha_red_;
            }
        }
    } else if (lsmode == LineSearchModes::AUGLANG) {

        double vv = this->get_prim_dual_grad(RHS).dot(this->get_primals_slacks(DXSL));
        double cv = this->get_lmults(DXSL).dot(this->get_all_cons(RHS));

        double LangInit = PrimObj + BarrObj;
        double InitL1Pen = this->get_lmults(XSL).cwiseAbs().dot(this->get_all_cons(RHS).cwiseAbs());
        double InitL2Pen = this->get_all_cons(RHS).squaredNorm();
        double InitLinfPenalty = this->get_all_cons(RHS).lpNorm<Eigen::Infinity>();

        double sc = .01 + std::abs(vv - cv) / InitL2Pen;
        if (InitL2Pen == 0.0)
            sc = 1.0;

        LangInit += InitL1Pen + InitL2Pen * sc;

        for (int j = 0; j < this->max_ls_iters_; j++) {
            double ptest = 0;
            double btest = 0;
            XSL2 = XSL + alpha * DXSL;
            RHS2.setZero();
            this->nlp->eval_occ(
                obj_scale_, XSL2.head(this->primal_vars_), ptest,
                RHS2.segment(this->primal_vars_ + this->slack_vars_, this->equal_cons_),
                RHS2.tail(this->inequal_cons_));

            this->apply_reset_slacks(this->get_slacks(XSL2), this->get_iq_cons(RHS2));
            btest = this->barrier_objective(this->get_slacks(XSL2), Mu);

            double LangTest = ptest + btest;

            double TestL1Pen =
                this->get_lmults(XSL).cwiseAbs().dot(this->get_all_cons(RHS2).cwiseAbs());

            TestL1Pen = 0;

            for (int i = 0; i < this->equal_cons_; i++) {
                double eqerr = abs(this->get_eq_cons(RHS2)[i]);
                double eqmul = abs(this->get_eq_lmults(XSL)[i]);
                if (eqerr > this->econ_tol_ * 10) {
                    TestL1Pen += eqerr * eqmul;
                }
            }

            for (int i = 0; i < this->inequal_cons_; i++) {
                double iqerr = abs(this->get_iq_cons(RHS2)[i]);
                double iqmul = abs(this->get_iq_lmults(XSL)[i]);
                if (iqerr > this->icon_tol_ * 10) {
                    TestL1Pen += iqerr * iqmul;
                }
            }

            double TestL2Pen = this->get_all_cons(RHS2).squaredNorm();
            double TestLinfPenalty = this->get_all_cons(RHS2).lpNorm<Eigen::Infinity>();

            if (TestL2Pen <
                econ_tol_ * econ_tol_ * equal_cons_ + icon_tol_ * icon_tol_ * inequal_cons_) {
                TestL2Pen = 0;
            }

            LangTest += TestL1Pen + TestL2Pen * sc;

            Citer.MeritVal = LangTest;
            if (LangTest < LangInit || (ptest < PrimObj) && (TestL2Pen < InitL2Pen) ||
                (ptest < PrimObj) && (TestLinfPenalty < InitLinfPenalty)) {
                Citer.LSiters = j;
                break;
            } else {
                Citer.LSiters = j + 1;
                alpha = alpha / this->alpha_red_;
            }
        }
    } else
        Citer.LSiters = 0;

    return alpha;
}

Eigen::VectorXd tycho::solvers::PSIOPT::optimize(const Eigen::VectorXd &x) {

    this->zero_timing_stats();

    if (this->print_level_ == 0)
        print_stats();
    if (this->print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();
    tycho::utils::Timer t;
    t.start();

    bool docompute = analyze_kkt_matrix();

    Eigen::VectorXd XSL = this->init_impl(x, this->init_mu_, docompute);

    Eigen::VectorXd XSLans(this->kkt_dim_);
    XSLans.setZero();
    if (this->print_level_ < 2) {
        print_beginning("Optimization Algorithm ");
    }
    XSLans = this->alg_impl(AlgorithmModes::OPT, this->opt_bar_mode_, this->opt_ls_mode_,
                            this->obj_scale_, this->init_mu_, XSL);
    if (this->print_level_ < 2) {
        print_finished("Optimization Algorithm ");
    }

    t.stop();
    double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
    this->last_total_time_ = tottime / 1000.0;
    this->last_misc_time_ = this->last_total_time_ - this->last_pre_time_ - this->last_kkt_time_ -
                            this->last_func_time_ - this->last_print_time_;

    if (this->print_level_ < 2) {
        print_timing_summary();
        fmt::print(" PSIOPT Total Time            : ");
        fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.3f} ms\n", tottime);
        print_finished("PSIOPT ");
        print_header();
    }
    return this->get_primals(XSLans);
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve_optimize(const Eigen::VectorXd &x) {

    this->zero_timing_stats();
    if (this->print_level_ == 0)
        print_stats();
    if (this->print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();
    tycho::utils::Timer t;
    t.start();

    bool docompute = analyze_kkt_matrix();

    Eigen::VectorXd XSL = this->init_impl(x, this->init_mu_, docompute);
    Eigen::VectorXd XSLans(this->kkt_dim_);
    XSLans.setZero();

    if (this->print_level_ < 2) {
        print_beginning("Solve Algorithm ");
    }

    XSLans = this->alg_impl(this->soe_mode_, this->soe_bar_mode_, this->soe_ls_mode_,
                            this->obj_scale_, this->init_mu_, XSL);
    if (this->print_level_ < 2) {
        print_finished("Solve Algorithm ");
    }
    Eigen::VectorXd Xt = this->get_primals(XSLans);
    XSL = this->init_impl(Xt, this->init_mu_, false);

    if (this->print_level_ < 2) {
        print_beginning("Optimization Algorithm ");
    }
    XSLans = this->alg_impl(AlgorithmModes::OPT, this->opt_bar_mode_, this->opt_ls_mode_,
                            this->obj_scale_, this->init_mu_, XSL);

    t.stop();
    double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
    this->last_total_time_ = tottime / 1000.0;
    this->last_misc_time_ = this->last_total_time_ - this->last_pre_time_ - this->last_kkt_time_ -
                            this->last_func_time_ - this->last_print_time_;

    if (this->print_level_ < 2) {
        print_finished("Optimization Algorithm ");
        print_timing_summary();
        fmt::print(" PSIOPT Total Time            : ");
        fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.3f} ms\n", tottime);
        print_finished("PSIOPT ");
        print_header();
    }

    return this->get_primals(XSLans);
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve_optimize_solve(const Eigen::VectorXd &x) {
    this->zero_timing_stats();
    if (this->print_level_ == 0)
        print_stats();
    if (this->print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();
    tycho::utils::Timer t;
    t.start();

    bool docompute = analyze_kkt_matrix();

    Eigen::VectorXd XSL = this->init_impl(x, this->init_mu_, docompute);
    Eigen::VectorXd XSLans(this->kkt_dim_);
    XSLans.setZero();

    if (this->print_level_ < 2) {
        print_beginning("Solve Algorithm ");
    }

    XSLans = this->alg_impl(this->soe_mode_, this->soe_bar_mode_, this->soe_ls_mode_,
                            this->obj_scale_, this->init_mu_, XSL);
    if (this->print_level_ < 2) {
        print_finished("Solve Algorithm ");
    }
    Eigen::VectorXd Xt = this->get_primals(XSLans);
    XSL = this->init_impl(Xt, this->init_mu_, false);

    if (this->print_level_ < 2) {
        print_beginning("Optimization Algorithm ");
    }
    XSLans = this->alg_impl(AlgorithmModes::OPT, this->opt_bar_mode_, this->opt_ls_mode_,
                            this->obj_scale_, this->init_mu_, XSL);
    if (this->print_level_ < 2) {
        print_finished("Optimization Algorithm ");
    }
    if (this->converge_flag_ == ConvergenceFlags::CONVERGED) {

    } else {
        Xt = this->get_primals(XSLans);
        XSL = this->init_impl(Xt, this->init_mu_, false);

        if (this->print_level_ < 2) {
            print_beginning("Solve Algorithm ");
        }
        XSLans = this->alg_impl(this->soe_mode_, this->soe_bar_mode_, this->soe_ls_mode_,
                                this->obj_scale_, this->init_mu_, XSL);

        if (this->print_level_ < 2) {
            print_finished("Solve Algorithm ");
        }
    }
    t.stop();
    double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
    this->last_total_time_ = tottime / 1000.0;
    this->last_misc_time_ = this->last_total_time_ - this->last_pre_time_ - this->last_kkt_time_ -
                            this->last_func_time_ - this->last_print_time_;

    if (this->print_level_ < 2) {
        print_timing_summary();
        fmt::print(" PSIOPT Total Time            : ");
        fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.3f} ms\n", tottime);
        print_finished("PSIOPT ");
        print_header();
    }

    return this->get_primals(XSLans);
}

Eigen::VectorXd tycho::solvers::PSIOPT::optimize_solve(const Eigen::VectorXd &x) {
    this->zero_timing_stats();
    if (this->print_level_ == 0)
        print_stats();
    if (this->print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();
    tycho::utils::Timer t;
    t.start();

    bool docompute = analyze_kkt_matrix();

    Eigen::VectorXd XSL = this->init_impl(x, this->init_mu_, docompute);
    Eigen::VectorXd XSLans(this->kkt_dim_);

    if (this->print_level_ < 2) {
        print_beginning("Optimization Algorithm ");
    }

    XSLans = this->alg_impl(AlgorithmModes::OPT, this->opt_bar_mode_, this->opt_ls_mode_,
                            this->obj_scale_, this->init_mu_, XSL);

    if (this->print_level_ < 2) {
        print_finished("Optimization Algorithm ");
    }

    if (this->converge_flag_ == ConvergenceFlags::CONVERGED) {

    } else {
        Eigen::VectorXd Xt = this->get_primals(XSLans);
        XSL = this->init_impl(Xt, this->init_mu_, false);

        if (this->print_level_ < 2) {
            print_beginning("Solve Algorithm ");
        }
        XSLans = this->alg_impl(this->soe_mode_, this->soe_bar_mode_, this->soe_ls_mode_,
                                this->obj_scale_, this->init_mu_, XSL);

        if (this->print_level_ < 2) {
            print_finished("Solve Algorithm ");
        }
    }
    t.stop();
    double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
    this->last_total_time_ = tottime / 1000.0;
    this->last_misc_time_ = this->last_total_time_ - this->last_pre_time_ - this->last_kkt_time_ -
                            this->last_func_time_ - this->last_print_time_;

    if (this->print_level_ < 2) {
        print_timing_summary();
        fmt::print(" PSIOPT Total Time            : ");
        fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.3f} ms\n", tottime);
        print_finished("PSIOPT ");
        print_header();
    }

    return this->get_primals(XSLans);
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve(const Eigen::VectorXd &x) {

    this->zero_timing_stats();
    if (this->print_level_ == 0)
        print_stats();
    if (this->print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();
    tycho::utils::Timer t;
    t.start();
    bool docompute = analyze_kkt_matrix();

    Eigen::VectorXd XSL = this->init_impl(x, this->init_mu_, docompute);
    Eigen::VectorXd XSLans(this->kkt_dim_);
    XSLans.setZero();
    if (this->print_level_ < 2) {
        print_beginning("Solve Algorithm ");
    }
    XSLans = this->alg_impl(this->soe_mode_, this->soe_bar_mode_, this->soe_ls_mode_,
                            this->obj_scale_, this->init_mu_, XSL);

    t.stop();
    double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
    this->last_total_time_ = tottime / 1000.0;
    this->last_misc_time_ = this->last_total_time_ - this->last_pre_time_ - this->last_kkt_time_ -
                            this->last_func_time_ - this->last_print_time_;

    if (this->print_level_ < 2) {
        print_finished("Solve Algorithm ");
        print_timing_summary();
        fmt::print(" PSIOPT Total Time            : ");
        fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.3f} ms\n", tottime);
        print_finished("PSIOPT ");
        print_header();
    }

    return this->get_primals(XSLans);
}
