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
        Func func;
        std::vector<VectorXi> indices;
        FuncIndexHolder() {}
        FuncIndexHolder(Func func, const std::vector<VectorXi> &indices)
            : func(func), indices(indices) {}
    };

    bool doTranscription = true;
    void resetTranscription() { this->doTranscription = true; };
    bool EnableVectorization = true;

    VectorXd ActiveVariables;
    bool MultipliersLoaded = false;

    VectorXd ActiveEqLmults;
    VectorXd ActiveIqLmults;

    std::vector<FuncIndexHolder<ConstraintInterface>> userEqualities;
    std::vector<FuncIndexHolder<ConstraintInterface>> userInequalities;
    std::vector<FuncIndexHolder<ObjectiveInterface>> userObjectives;

    OptimizationProblem() { this->setNumPartitions(1, 1); }
    virtual ~OptimizationProblem() = default;

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    template <class T> static void check_function_size(const T &func, std::string ftype) {
        int irows = func.func.input_rows();
        for (auto &index : func.indices) {
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

    void setVars(const VectorXd &v) { this->ActiveVariables = v; }
    VectorXd returnVars() const { return this->ActiveVariables; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    int addEqualCon(VectorFunctionalX fun, const std::vector<VectorXi> &indices) {
        this->resetTranscription();
        int index = int(this->userEqualities.size());
        this->userEqualities.emplace_back(FuncIndexHolder<ConstraintInterface>(fun, indices));
        check_function_size(this->userEqualities.back(), "Equality Constraint");
        return index;
    }
    int addEqualCon(VectorFunctionalX fun, VectorXi index) {
        std::vector<VectorXi> indices = {index};
        return this->addEqualCon(fun, indices);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    int addInequalCon(VectorFunctionalX fun, const std::vector<VectorXi> &indices) {
        this->resetTranscription();
        int index = int(this->userInequalities.size());
        this->userInequalities.emplace_back(FuncIndexHolder<ConstraintInterface>(fun, indices));
        check_function_size(this->userInequalities.back(), "Inequality Constraint");
        return index;
    }

    int addInequalCon(VectorFunctionalX fun, VectorXi index) {
        std::vector<VectorXi> indices = {index};
        return this->addInequalCon(fun, indices);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    int addObjective(ScalarFunctionalX fun, const std::vector<VectorXi> &indices) {
        this->resetTranscription();
        int index = int(this->userObjectives.size());
        this->userObjectives.emplace_back(FuncIndexHolder<ObjectiveInterface>(fun, indices));
        check_function_size(this->userObjectives.back(), "Objective");

        return index;
    }
    int addObjective(ScalarFunctionalX fun, VectorXi index) {
        std::vector<VectorXi> indices = {index};
        return this->addObjective(fun, indices);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    void transcribe();

    void jet_initialize() {
        this->setNumPartitions(1, 1);
        this->optimizer->print_level_ = 10;
        this->transcribe();
    }
    void jet_release() {
        this->optimizer->release();
        this->setNumPartitions(1, 1);
        this->optimizer->print_level_ = 0;
        this->nlp = std::shared_ptr<NonLinearProgram>();
        this->resetTranscription();
    }

    PSIOPT::ConvergenceFlags solve() {
        if (this->doTranscription)
            this->transcribe();
        this->ActiveVariables = this->optimizer->solve(this->ActiveVariables);
        this->ActiveEqLmults = this->optimizer->last_eq_lmults_;
        this->ActiveIqLmults = this->optimizer->last_iq_lmults_;
        return this->optimizer->converge_flag_;
    }

    PSIOPT::ConvergenceFlags optimize() {
        if (this->doTranscription)
            this->transcribe();
        this->ActiveVariables = this->optimizer->optimize(this->ActiveVariables);
        this->ActiveEqLmults = this->optimizer->last_eq_lmults_;
        this->ActiveIqLmults = this->optimizer->last_iq_lmults_;
        return this->optimizer->converge_flag_;
    }

    PSIOPT::ConvergenceFlags solve_optimize() {
        if (this->doTranscription)
            this->transcribe();
        this->ActiveVariables = this->optimizer->solve_optimize(this->ActiveVariables);
        this->ActiveEqLmults = this->optimizer->last_eq_lmults_;
        this->ActiveIqLmults = this->optimizer->last_iq_lmults_;
        return this->optimizer->converge_flag_;
    }

    PSIOPT::ConvergenceFlags solve_optimize_solve() {
        if (this->doTranscription)
            this->transcribe();
        this->ActiveVariables = this->optimizer->solve_optimize_solve(this->ActiveVariables);
        this->ActiveEqLmults = this->optimizer->last_eq_lmults_;
        this->ActiveIqLmults = this->optimizer->last_iq_lmults_;
        return this->optimizer->converge_flag_;
    }

    PSIOPT::ConvergenceFlags optimize_solve() {
        if (this->doTranscription)
            this->transcribe();
        this->ActiveVariables = this->optimizer->optimize_solve(this->ActiveVariables);
        this->ActiveEqLmults = this->optimizer->last_eq_lmults_;
        this->ActiveIqLmults = this->optimizer->last_iq_lmults_;
        return this->optimizer->converge_flag_;
    }
};

} // namespace tycho::solvers
