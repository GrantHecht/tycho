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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/optimization_problem_base.h"
#include "tycho/detail/solvers/psiopt.h"
#include "tycho/vector_functions.h"

namespace tycho::solvers {

// Import cross-namespace types used by OptimizationProblem.
using vf::GenericFunction;

struct OptimizationProblem : OptimizationProblemBase {

    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    template <class Func> struct FuncIndexHolder {
        Func func_;
        std::vector<VectorXi> indices_;
        FuncIndexHolder() {}
        FuncIndexHolder(Func func, const std::vector<VectorXi> &indices)
            : func_(func), indices_(indices) {}
    };

    bool do_transcription_ = true;
    void reset_transcription() { this->do_transcription_ = true; };
    bool enable_vectorization_ = true;

    VectorXd active_variables_;
    bool multipliers_loaded_ = false;

    VectorXd active_eq_lmults_;
    VectorXd active_iq_lmults_;

    std::vector<FuncIndexHolder<ConstraintInterface>> user_equalities_;
    std::vector<FuncIndexHolder<ConstraintInterface>> user_inequalities_;
    std::vector<FuncIndexHolder<ObjectiveInterface>> user_objectives_;

    OptimizationProblem() { this->set_num_partitions(1, 1); }
    virtual ~OptimizationProblem() = default;

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    template <class T> static void check_function_size(const T &func, std::string ftype) {
        int irows = func.func_.input_rows();
        for (auto &index : func.indices_) {
            int isize = index.size();
            if (irows != isize) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Input size of {0:} (IRows = {1:}) does not match that implied by "
                           "indexing parameters "
                           "(IRows = {2:}).\n",
                           ftype, irows, isize);
                throw std::invalid_argument("");
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    void set_vars(const VectorXd &v) { this->active_variables_ = v; }
    VectorXd return_vars() const { return this->active_variables_; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    int add_equal_con(VectorFunctionalX fun, const std::vector<VectorXi> &indices) {
        this->reset_transcription();
        int index = int(this->user_equalities_.size());
        this->user_equalities_.emplace_back(FuncIndexHolder<ConstraintInterface>(fun, indices));
        check_function_size(this->user_equalities_.back(), "Equality Constraint");
        return index;
    }
    int add_equal_con(VectorFunctionalX fun, VectorXi index) {
        std::vector<VectorXi> indices = {index};
        return this->add_equal_con(fun, indices);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    int add_inequal_con(VectorFunctionalX fun, const std::vector<VectorXi> &indices) {
        this->reset_transcription();
        int index = int(this->user_inequalities_.size());
        this->user_inequalities_.emplace_back(FuncIndexHolder<ConstraintInterface>(fun, indices));
        check_function_size(this->user_inequalities_.back(), "Inequality Constraint");
        return index;
    }

    int add_inequal_con(VectorFunctionalX fun, VectorXi index) {
        std::vector<VectorXi> indices = {index};
        return this->add_inequal_con(fun, indices);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    int add_objective(ScalarFunctionalX fun, const std::vector<VectorXi> &indices) {
        this->reset_transcription();
        int index = int(this->user_objectives_.size());
        this->user_objectives_.emplace_back(FuncIndexHolder<ObjectiveInterface>(fun, indices));
        check_function_size(this->user_objectives_.back(), "Objective");

        return index;
    }
    int add_objective(ScalarFunctionalX fun, VectorXi index) {
        std::vector<VectorXi> indices = {index};
        return this->add_objective(fun, indices);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    void transcribe();

    void jet_initialize() {
        this->set_num_partitions(1, 1);
        this->optimizer_->set_print_level(10);
        this->transcribe();
    }
    void jet_release() {
        this->optimizer_->release();
        this->set_num_partitions(1, 1);
        this->optimizer_->set_print_level(0);
        this->nlp_ = std::shared_ptr<NonLinearProgram>();
        this->reset_transcription();
    }

    PSIOPT::ConvergenceFlags solve() {
        if (this->do_transcription_)
            this->transcribe();
        this->active_variables_ = this->optimizer_->solve(this->active_variables_);
        this->active_eq_lmults_ = this->optimizer_->result().eq_lmults_;
        this->active_iq_lmults_ = this->optimizer_->result().iq_lmults_;
        return this->optimizer_->result().converge_flag_;
    }

    PSIOPT::ConvergenceFlags optimize() {
        if (this->do_transcription_)
            this->transcribe();
        this->active_variables_ = this->optimizer_->optimize(this->active_variables_);
        this->active_eq_lmults_ = this->optimizer_->result().eq_lmults_;
        this->active_iq_lmults_ = this->optimizer_->result().iq_lmults_;
        return this->optimizer_->result().converge_flag_;
    }

    PSIOPT::ConvergenceFlags solve_optimize() {
        if (this->do_transcription_)
            this->transcribe();
        this->active_variables_ = this->optimizer_->solve_optimize(this->active_variables_);
        this->active_eq_lmults_ = this->optimizer_->result().eq_lmults_;
        this->active_iq_lmults_ = this->optimizer_->result().iq_lmults_;
        return this->optimizer_->result().converge_flag_;
    }

    PSIOPT::ConvergenceFlags solve_optimize_solve() {
        if (this->do_transcription_)
            this->transcribe();
        this->active_variables_ = this->optimizer_->solve_optimize_solve(this->active_variables_);
        this->active_eq_lmults_ = this->optimizer_->result().eq_lmults_;
        this->active_iq_lmults_ = this->optimizer_->result().iq_lmults_;
        return this->optimizer_->result().converge_flag_;
    }

    PSIOPT::ConvergenceFlags optimize_solve() {
        if (this->do_transcription_)
            this->transcribe();
        this->active_variables_ = this->optimizer_->optimize_solve(this->active_variables_);
        this->active_eq_lmults_ = this->optimizer_->result().eq_lmults_;
        this->active_iq_lmults_ = this->optimizer_->result().iq_lmults_;
        return this->optimizer_->result().converge_flag_;
    }
};

} // namespace tycho::solvers
