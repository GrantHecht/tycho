// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers/nlp_backend.h"
#include "tycho/detail/solvers_vf/transcribed_aggregate.h"
#include "tycho/vector_functions.h"
#include <hven/drivers/interior_point_solver.h>
#include <hven/model/non_linear_program.h>

namespace tycho::solvers {

// Import cross-namespace types used by OptimizationProblem.
using vf::GenericFunction;

struct OptimizationProblem : BackendProblemBase {

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

    /// Declared bounds on primal variables, verbatim and in declaration order.
    /// Repeated records on one index are intersected tightest-wins when the
    /// problem is laid out.
    std::vector<hven::solvers::VariableBound> user_var_bounds_;

    // Defaults (partitions, QP threads) come from the base ctor's
    // init_partitions(), matching Phase/OCP behavior.
    OptimizationProblem() = default;
    virtual ~OptimizationProblem() = default;

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    template <class T> static void check_function_size(const T &func, std::string ftype) {
        int irows = func.func_.input_rows();
        for (auto &index : func.indices_) {
            int isize = index.size();
            if (irows != isize) {
                throw std::invalid_argument(
                    fmt::format("Input size of {0:} (IRows = {1:}) does not match that implied by "
                                "indexing parameters (IRows = {2:}).",
                                ftype, irows, isize));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Declares a bound on one primal variable.
    ///
    /// The bound is part of the declared problem, so it travels with the
    /// transcription rather than being written onto the laid-out program.
    /// Repeated declarations on one index are intersected tightest-wins; a
    /// record that bounds neither side narrows nothing and is dropped.
    ///
    /// @param index the primal variable the bound applies to.
    /// @param lower the lower side.
    /// @param upper the upper side.
    /// @throws std::invalid_argument if @p index is negative, if either side is
    ///         NaN, or if the two finite sides are inverted.
    void add_variable_bound(int index, double lower, double upper) {
        if (index < 0) {
            throw std::invalid_argument(
                fmt::format("add_variable_bound: variable index {0} is negative", index));
        }
        if (std::isnan(lower) || std::isnan(upper)) {
            throw std::invalid_argument(
                fmt::format("add_variable_bound: the bound on variable {0} is not a number "
                            "(lower={1}, upper={2})",
                            index, lower, upper));
        }
        constexpr double kInf = std::numeric_limits<double>::infinity();
        const bool lower_finite = lower > -kInf && lower < kInf;
        const bool upper_finite = upper > -kInf && upper < kInf;
        if (lower_finite && upper_finite && lower > upper) {
            throw std::invalid_argument(
                fmt::format("add_variable_bound: the bound on variable {0} is inverted "
                            "(lower={1} is above upper={2})",
                            index, lower, upper));
        }
        this->reset_transcription();
        this->user_var_bounds_.push_back(hven::solvers::VariableBound{index, lower, upper});
    }

    /// @brief Drops every declared variable bound.
    void clear_variable_bounds() {
        this->reset_transcription();
        this->user_var_bounds_.clear();
    }

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
        // Single-partition evaluation on the calling thread, and a single QP
        // thread: two independent settings, set independently. set_qp_threads
        // goes first because it validates and can throw.
        this->optimizer_->set_qp_threads(1);
        this->set_num_partitions(1);
        this->optimizer_->set_print_level(10);
        this->transcribe();
    }
    void jet_release() {
        this->optimizer_->release();
        // Same ordering rationale as jet_initialize() above.
        this->optimizer_->set_qp_threads(1);
        this->set_num_partitions(1);
        this->optimizer_->set_print_level(0);
        this->nlp_ = std::shared_ptr<NonLinearProgram>();
        this->provider_ = std::shared_ptr<TranscribedAggregate>();
        this->reset_transcription();
    }

    tycho::ConvergenceFlags solve() {
        if (this->do_transcription_)
            this->transcribe();
        auto out = this->run_nlp_solver(JetJobModes::Solve, this->active_variables_);
        this->active_variables_ = out.variables_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
        return out.flag_;
    }

    tycho::ConvergenceFlags optimize() {
        if (this->do_transcription_)
            this->transcribe();
        auto out = this->run_nlp_solver(JetJobModes::Optimize, this->active_variables_);
        this->active_variables_ = out.variables_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
        return out.flag_;
    }

    tycho::ConvergenceFlags solve_optimize() {
        if (this->do_transcription_)
            this->transcribe();
        auto out = this->run_nlp_solver(JetJobModes::SolveOptimize, this->active_variables_);
        this->active_variables_ = out.variables_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
        return out.flag_;
    }

    tycho::ConvergenceFlags solve_optimize_solve() {
        if (this->do_transcription_)
            this->transcribe();
        auto out = this->run_nlp_solver(JetJobModes::SolveOptimizeSolve, this->active_variables_);
        this->active_variables_ = out.variables_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
        return out.flag_;
    }

    tycho::ConvergenceFlags optimize_solve() {
        if (this->do_transcription_)
            this->transcribe();
        auto out = this->run_nlp_solver(JetJobModes::OptimizeSolve, this->active_variables_);
        this->active_variables_ = out.variables_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
        return out.flag_;
    }

    // BackendProblemBase's new solve(EngineRef, SolveOptions) overload (M5)
    // is otherwise hidden by the 0-arg solve() override above.
    using BackendProblemBase::solve;

  protected:
    /// @brief M5 solve() hook: transcribe if needed (make_nlp + set_nlp
    ///        wiring already lives inside transcribe()).
    void prepare_solve() override {
        if (this->do_transcription_)
            this->transcribe();
    }

    /// @brief M5 solve() hook: the active variables vector, as
    ///        solve()/optimize() etc. above already seed run_nlp_solver with.
    Eigen::VectorXd initial_primal() const override { return this->active_variables_; }

    /// @brief M5 solve() hook: write the stage's primal/multipliers back,
    ///        the same fields solve()/optimize() etc. above write.
    void accept_stage(const StageOutput &out) override {
        this->active_variables_ = out.primal_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
    }
};

} // namespace tycho::solvers
