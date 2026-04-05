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
//   - Printing methods extracted from psiopt.cpp for build-organization clarity
// =============================================================================

#include "tycho/detail/solvers/psiopt.h"

#include <iostream>

void tycho::solvers::PSIOPT::print_timing_summary() {
    auto cyan = fmt::fg(fmt::color::cyan);
    fmt::print(" KKT Analysis/Init Time       : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->result_.pre_time_ * 1000.0);
    fmt::print(" NLP Function Evaluation Time : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->result_.func_time_ * 1000.0);
    fmt::print(" KKT Factor/Solve Time        : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->result_.kkt_time_ * 1000.0);
    fmt::print(" Console Print Time           : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->result_.print_time_ * 1000.0);
    fmt::print(" Misc Time                    : ");
    fmt::print(cyan, "{0:>10.3f} ms\n", this->result_.misc_time_ * 1000.0);
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
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "KKT", settings_.kkt_tol_,
               settings_.acc_kkt_tol_, settings_.div_kkt_tol_);
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "Bar", settings_.bar_tol_,
               settings_.acc_bar_tol_, settings_.div_bar_tol_);
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "ECons", settings_.econ_tol_,
               settings_.acc_econ_tol_, settings_.div_econ_tol_);
    fmt::print("|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "ICons", settings_.icon_tol_,
               settings_.acc_icon_tol_, settings_.div_icon_tol_);
}

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
    fmt::print(cyan, "{:<10}\n", this->kkt_sol_.get_matrix().nonZeros());
    fmt::print(" KKT-Matrix NNZ%          : ");
    fmt::print(cyan, "{:.6f}%\n",
               100.0 * double(this->kkt_sol_.get_matrix().nonZeros()) /
                   (double(this->kkt_dim_) * double(this->kkt_dim_)));
    fmt::print("\n");
}

void tycho::solvers::PSIOPT::print_last_iterate(const std::vector<IterateInfo> &iters) {
    auto last = iters.back();

    if (last.iter % 10 == 0) {
        if (settings_.wide_console_) {
            fmt::print("{0:=^{1}}\n", "", 159);
            fmt::print(
                "|Iter| mu Val | Prim Obj |  Bar Obj |  KKT Inf |  Bar Inf | ECons Inf| ICons "
                "Inf|Max "
                "EMult|Max IMult| AlphaP | AlphaD | AlphaT | Merit Val|LSI|PPS|HFI| HPert |\n");
        } else {
            fmt::print("{0:=^{1}}\n", "", 119);
            fmt::print("|Iter| mu Val | Prim Obj |  Bar Obj |  KKT Inf |  Bar Inf | ECons Inf| "
                       "ICons Inf| AlphaP | "
                       "AlphaD |LS| PPS |HF| HPert |\n");
        }
    }

    fmt::text_style PHashcol = fmt::text_style();
    fmt::text_style EHashcol = fmt::text_style();
    fmt::text_style IHashcol = fmt::text_style();
    fmt::text_style KHashcol = fmt::text_style();
    fmt::text_style BHashcol = fmt::text_style();
    fmt::text_style BOHashcol = fmt::text_style();

    fmt::text_style Kcol =
        calculate_color(last.kkt_inf_, settings_.kkt_tol_, settings_.acc_kkt_tol_);
    fmt::text_style Bcol =
        calculate_color(last.barr_inf_, settings_.bar_tol_, settings_.acc_bar_tol_);
    fmt::text_style Ecol =
        calculate_color(last.econ_inf_, settings_.econ_tol_, settings_.acc_econ_tol_);
    fmt::text_style Icol =
        calculate_color(last.icon_inf_, settings_.icon_tol_, settings_.acc_icon_tol_);

    if (iters.size() > 1) {

        auto GCol = fmt::fg(fmt::color::lime_green);
        auto BCol = fmt::fg(fmt::color::red);

        PHashcol = (iters.back().prim_obj_ <= iters[iters.size() - 2].prim_obj_) ? GCol : BCol;
        BOHashcol = (iters.back().barr_obj_ <= iters[iters.size() - 2].barr_obj_) ? GCol : BCol;
        BHashcol = (iters.back().barr_inf_ <= iters[iters.size() - 2].barr_inf_) ? GCol : BCol;
        EHashcol = (iters.back().econ_inf_ <= iters[iters.size() - 2].econ_inf_) ? GCol : BCol;
        IHashcol = (iters.back().icon_inf_ <= iters[iters.size() - 2].icon_inf_) ? GCol : BCol;
        KHashcol = (iters.back().kkt_inf_ <= iters[iters.size() - 2].kkt_inf_) ? GCol : BCol;
    }

    auto hash = []() { fmt::print("|"); };
    auto chash = [](fmt::text_style c) { fmt::print(c, "|"); };

    hash();
    fmt::print("{:<4}", last.iter);
    hash();
    fmt::print("{:.2e}", last.mu_);
    hash();
    fmt::print("{:>10.3e}", last.prim_obj_);
    chash(PHashcol);
    fmt::print("{:>10.3e}", last.barr_obj_);
    chash(BOHashcol);
    fmt::print(Kcol, "{:>10.4e}", last.kkt_inf_);
    chash(KHashcol);
    fmt::print(Bcol, "{:>10.4e}", last.barr_inf_);
    chash(BHashcol);
    fmt::print(Ecol, "{:>10.4e}", last.econ_inf_);
    chash(EHashcol);
    fmt::print(Icol, "{:>10.4e}", last.icon_inf_);
    chash(IHashcol);

    if (settings_.wide_console_) {
        fmt::print(
            "{:>9.3e}|{:>9.3e}|{:>8.2e}|{:>8.2e}|{:>8.2e}|{:>10.3e}|{:>3}|{:>3}|{:>3}|{:>6.1e}|\n",
            last.max_e_mult_, last.max_i_mult_, last.alpha_p_, last.alpha_d_, last.alpha_t_,
            last.merit_val_, last.ls_iters_, last.p_pivots_, last.h_facs_, last.h_pert_);
    } else {
        fmt::print("{:>8.2e}|{:>8.2e}|{:>2}|{:>5}|{:>2}|{:>6.1e}|\n", last.alpha_t_ * last.alpha_p_,
                   last.alpha_t_ * last.alpha_d_, last.ls_iters_, last.p_pivots_, last.h_facs_,
                   last.h_pert_);
    }
}

void tycho::solvers::PSIOPT::print_beginning(std::string_view msg) const {
    fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "{}", msg);
    fmt::print("\n");
}

void tycho::solvers::PSIOPT::print_finished(std::string_view msg) const {

    fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "{}", msg);
    fmt::print("\n");
}

void tycho::solvers::PSIOPT::print_exit_stats(ConvergenceFlags ExitCode, const IterateInfo &last,
                                              int iternum, double tottime, double nlptime,
                                              double qptime, double printtime) {
    fmt::text_style Kcol =
        calculate_color(last.kkt_inf_, settings_.kkt_tol_, settings_.acc_kkt_tol_);
    fmt::text_style Bcol =
        calculate_color(last.barr_inf_, settings_.bar_tol_, settings_.acc_bar_tol_);
    fmt::text_style Ecol =
        calculate_color(last.econ_inf_, settings_.econ_tol_, settings_.acc_econ_tol_);
    fmt::text_style Icol =
        calculate_color(last.icon_inf_, settings_.icon_tol_, settings_.acc_icon_tol_);

    auto TColor = fmt::fg(fmt::color::cyan);
    auto Printtime = [&](const char *msg, double t1) {
        fmt::print("{}", msg);
        fmt::print(TColor, "{0:>10.3f} ms {1:>10.3f} ms/iter\n", t1, double(t1 / iternum));
    };

    if (settings_.print_level_ < 3) {
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

    if (settings_.print_level_ < 2) {

        fmt::print(" Iterations : ");
        fmt::print("{:<5}\n", iternum);
        fmt::print(" Prim Obj   : ");
        fmt::print("{:<15.8e}\n", last.prim_obj_);
        fmt::print(" KKT Inf    : ");
        fmt::print(Kcol, "{:<15.8e}\n", last.kkt_inf_);
        fmt::print(" Bar Inf    : ");
        fmt::print(Bcol, "{:<15.8e}\n", last.barr_inf_);
        fmt::print(" ECons Inf  : ");
        fmt::print(Ecol, "{:<15.8e}\n", last.econ_inf_);
        fmt::print(" ICons Inf  : ");
        fmt::print(Icol, "{:<15.8e}\n", last.icon_inf_);

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
