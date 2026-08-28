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

    /// @internal
    /// @brief Prepare the problem for a "jet" (batched) solve.
    ///
    /// Single-partition evaluation on the calling thread. The engine-level
    /// settings (QP thread count, print level) that jet_initialize() used to
    /// force onto the problem's own owned optimizer no longer apply here --
    /// there is no owned optimizer; the caller configures whichever engine
    /// it hands to solve() directly.
    /// @endinternal
    void jet_initialize() override {
        this->set_num_partitions(1);
        this->transcribe();
    }
    /// @internal
    /// @brief Tear down jet-solve state and restore the default configuration.
    /// @endinternal
    void jet_release() override {
        this->set_num_partitions(1);
        this->nlp_ = std::shared_ptr<NonLinearProgram>();
        this->provider_ = std::shared_ptr<TranscribedAggregate>();
        this->reset_transcription();
    }

    // BackendProblemBase's new solve(EngineRef, SolveOptions) overload is
    // the only solve() this class has -- OptimizationProblem declares no
    // 0-arg override of its own, so nothing here hides it and no
    // using-declaration is needed.

  protected:
    /// @brief solve() hook: transcribe if needed (make_nlp + set_nlp
    ///        wiring already lives inside transcribe()).
    void prepare_solve() override {
        if (this->do_transcription_)
            this->transcribe();
    }

    /// @brief solve() hook: mark this problem as needing transcription again.
    void invalidate_transcription() override { this->reset_transcription(); }

    /// @brief solve() hook: the active variables vector.
    Eigen::VectorXd initial_primal() const override { return this->active_variables_; }

    /// @brief solve() hook: write the stage's primal/multipliers back.
    void accept_stage(const StageOutput &out) override {
        this->active_variables_ = out.primal_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
    }
};

} // namespace tycho::solvers
