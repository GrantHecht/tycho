// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// This file defines the default composite non-linear program class
// for interfacing with PSIOPT. This class is responsible for combining many different
// dense or sparse objective or constraints into a single optimization problem and
// manages all memory allocation, sparsity pattern computation, work partitioning, and function
// evaluation.
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <fmt/format.h>

#include "tycho/detail/solvers/bound_set.h"
#include "tycho/detail/solvers/constraint_function.h"
#include "tycho/detail/solvers/objective_function.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/thread_pool.h"

namespace tycho::solvers {

/// <summary>
/// How a primal variable whose declared lower and upper bounds are equal is
/// handed to the solver.
///
/// MakeParameter removes the variable from the optimization entirely: it is
/// pinned at its bound value for every evaluation and the Newton system the
/// solver factorizes is the system of the REMAINING variables. MakeConstraint
/// would instead keep the variable free and add an equality constraint fixing
/// it; RelaxBounds would keep it as an ordinary two-sided bounded variable with
/// its bounds pushed apart. Only MakeParameter is implemented —
/// NonLinearProgram::configure_variable_treatment rejects the other two with a
/// clear message.
/// </summary>
enum class FixedVariableTreatments { MakeParameter, MakeConstraint, RelaxBounds };

/// Human-readable name for a treatment, for diagnostics and error messages.
inline const char *fixed_variable_treatment_name(FixedVariableTreatments treatment) {
    switch (treatment) {
    case FixedVariableTreatments::MakeParameter:
        return "make_parameter";
    case FixedVariableTreatments::MakeConstraint:
        return "make_constraint";
    case FixedVariableTreatments::RelaxBounds:
        return "relax_bounds";
    }
    return "unknown";
}

/// Default widening applied to every finite, non-fixing variable bound before
/// it is recorded in the BoundSet: b is moved outward by this factor times
/// max(1, |b|). Matches Ipopt's bound_relax_factor default.
inline constexpr double kDefaultBoundRelaxFactor = 1.0e-8;

struct NonLinearProgram {
    using VectorXi = Eigen::VectorXi;
    using VectorXd = Eigen::VectorXd;
    using MatrixXi = Eigen::MatrixXi;

    int num_partitions_ = 1;

    /// <summary>
    /// Master List of Objective functions that will be partitioned across work partitions
    /// (part_obj_)
    /// </summary>
    std::vector<ObjectiveFunction> objectives_;

    /// <summary>
    /// Master List of Equality Constraint functions that will be partitioned across work partitions
    /// (part_eq_)
    /// </summary>
    std::vector<ConstraintFunction> equality_constraints_;

    /// <summary>
    /// Master List of Inequality Constraint functions that will be partitioned across work
    /// partitions (part_iq_)
    /// </summary>
    std::vector<ConstraintFunction> inequality_constraints_;

    /// <summary>
    /// Vector with each element being the list of ObjectiveFunctions
    /// assigned to the corresponding partition.
    /// </summary>
    std::vector<std::vector<ObjectiveFunction>> part_obj_;

    /// <summary>
    /// Vector with each element being the list of equality_constraints_
    /// assigned to the corresponding partition.
    /// </summary>
    std::vector<std::vector<ConstraintFunction>> part_eq_;

    /// <summary>
    /// Vector with each element being the list of inequality_constraints_
    /// assigned to the corresponding partition.
    /// </summary>
    std::vector<std::vector<ConstraintFunction>> part_iq_;

    int primal_vars_ = 0; // Number of design variables
    int slack_vars_ = 0;  // Number of slack variables appended to problem. One for every inequalcon
    int equal_cons_ = 0;  // Number of equality constraints,
    int inequal_cons_ = 0; // Number of inequality constraints
    int kkt_dim_ = 0; // Edge dimension of KKT matrix: = primal_vars_ + slack_vars_ + equal_cons_ +
                      // inequal_cons_

    VectorXi kkt_coeff_rows_; // matched row indices
    VectorXi kkt_coeff_cols_; // matched col indices
    VectorXi kkt_coeff_part_ids_;
    VectorXi kkt_locations_;

    int num_user_kkt_elems_ = 0;
    int num_solver_kkt_elems_ = 0;
    int num_kkt_elems_ = 0;

    VectorXd solver_coeffs_;
    int slack_jac_data_start_;    //// Solver supplied slack jacobian data, usually just
                                  /// a vector of ones
    int primal_diags_data_start_; //// Solver supplied diaganol elements for inertia
                                  /// modification or least norm solving
    int slack_diag_data_start_;   //// Solver suppled diaganols for slack elements in
                                  /// the hessian, used for interior point methods,
                                  /// zeros for SQP
    int e_pivot_data_start_;      //// Solver suppled Equality pivots
    int i_pivot_data_start_;      //// Solver suppled Inequality pivots

    std::vector<std::mutex> kkt_locks_;
    int num_kkt_clashes_ = 0;

    //// [i] = -1 if no fill clash, [i] = mutex lock index otherwise
    VectorXi kkt_clashes_;

    VectorXd rhs_coeffs_;
    VectorXi rhs_coeff_rows_;

    int num_pgx_elems_ = 0;
    int num_agx_elems_ = 0;
    int num_icon_elems_ = 0;
    int num_econ_elems_ = 0;
    int num_rhs_elems_ = 0;

    int pgx_data_start_ = 0;
    int agx_data_start_ = 0;
    int econ_data_start_ = 0;
    int icon_data_start_ = 0;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    NonLinearProgram(int NumParts) { this->num_partitions_ = std::max(NumParts, 1); }
    NonLinearProgram(int PV, int EQ, int IQ, std::vector<ObjectiveFunction> &obj,
                     std::vector<ConstraintFunction> &eq, std::vector<ConstraintFunction> &ineq,
                     int NumParts) {
        this->num_partitions_ = std::max(NumParts, 1);

        this->objectives_ = obj;
        this->equality_constraints_ = eq;
        this->inequality_constraints_ = ineq;
        this->make_nlp(PV, EQ, IQ);
    }

    void make_nlp(int PV, int EQ, int IQ);

    /// <summary>
    /// One staged variable-bound declaration, as handed to set_variable_bound.
    /// Recorded verbatim (no merging at declaration time) so that repeated
    /// make_nlp calls re-derive the same tightest-wins result from the same
    /// history.
    /// </summary>
    struct VariableBoundStage {
        int index_;
        double lower_;
        double upper_;
    };

    /// Staged variable-bound declarations, applied by make_nlp. Declaration
    /// may precede sizing (primal_vars_ is only known once make_nlp runs), so
    /// index-range validation and the tightest-wins merge both happen at
    /// materialization time, not here. Cleared only by clear_variable_bounds().
    std::vector<VariableBoundStage> staged_variable_bounds_;

    /// Dense per-primal-variable bounds materialized from
    /// staged_variable_bounds_ by make_nlp. Length primal_vars_ after
    /// make_nlp; -inf/+inf where no staged bound narrows the variable. Empty
    /// (size 0) before the first make_nlp call or after clear_variable_bounds().
    VectorXd x_lower_;
    VectorXd x_upper_;

    /// <summary>
    /// Stages a bound declaration for primal variable global_index. Repeated
    /// declarations on the same index are intersected (tightest wins) when
    /// make_nlp materializes x_lower_/x_upper_: l = max(l_prev, l_new),
    /// u = min(u_prev, u_new). A declaration with both bounds infinite leaves
    /// the variable unbounded and is a no-op. NaN bounds are rejected
    /// immediately, since they cannot participate in the max/min merge.
    /// </summary>
    void set_variable_bound(int global_index, double lower, double upper) {
        if (std::isnan(lower) || std::isnan(upper)) {
            throw std::invalid_argument(
                fmt::format("set_variable_bound: bound for index {0} is NaN (lower={1}, "
                            "upper={2})",
                            global_index, lower, upper));
        }
        constexpr double kInf = std::numeric_limits<double>::infinity();
        if (lower == -kInf && upper == kInf) {
            return;
        }
        this->staged_variable_bounds_.push_back({global_index, lower, upper});
    }

    /// Drops every staged declaration and the materialized x_lower_/x_upper_
    /// vectors. The next make_nlp call starts from an unbounded problem.
    void clear_variable_bounds() {
        this->staged_variable_bounds_.clear();
        this->x_lower_.resize(0);
        this->x_upper_.resize(0);
        this->bounds_revision_++;
    }

    /// True iff any primal variable has a finite lower or upper bound after
    /// materialization. False before the first make_nlp call.
    bool has_variable_bounds() const {
        if (this->x_lower_.size() == 0) {
            return false;
        }
        constexpr double kInf = std::numeric_limits<double>::infinity();
        return (this->x_lower_.array() > -kInf).any() || (this->x_upper_.array() < kInf).any();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Bound-fixed variable treatment
    //
    // A variable declared with lower == upper carries no degree of freedom. Under
    // the MakeParameter treatment it is ELIMINATED: pinned at its value for every
    // evaluation, and removed from the Newton system the solver factorizes.
    //
    // How the elimination is realized. The transcription's objective and
    // constraint functions address primal variables by their global index, both
    // when they gather their inputs out of x and when they scatter their
    // Jacobian/Hessian entries into the KKT matrix. Renumbering the primal space
    // underneath them is therefore not a local change -- it would have to reach
    // into every function's indexing data and into the KKT scatter itself. So the
    // elimination is applied to the ASSEMBLED system instead, once per assembly
    // and in O(nnz of the eliminated columns):
    //
    //   * the eliminated variable keeps its coordinate, but every KKT entry in
    //     its row and column is zeroed after assembly and its diagonal is set to
    //     one, leaving a decoupled unit row;
    //   * its stationarity row in the Newton right-hand side is cleared (the
    //     solver-side seam, PSIOPT::eval_nlp);
    //   * its value is pinned into the iterate once, at solve entry.
    //
    // The factorized matrix is then exactly the reduced problem's KKT matrix
    // bordered by an identity block, so the step is the reduced problem's step
    // with a structural zero in the eliminated coordinate, the inertia count is
    // unchanged (each pinned coordinate contributes exactly the one positive
    // eigenvalue the primal count already expects), and the eliminated variable's
    // residual never enters any convergence norm. Eliminated variables'
    // contributions to constraint values and to the remaining variables'
    // derivatives happen automatically: the functions still evaluate at the full
    // x with the pinned values in place, and never learn anything happened.
    //
    // Identity fast path. With no fixed variables, is_reduced() is false, every
    // site above short-circuits on that single cached bool, and NOTHING on the
    // evaluation path changes -- no extra arithmetic, no extra indirection, no
    // change to the assembled values.
    ////////////////////////////////////////////////////////////////////////////

    /// <summary>
    /// Classifies every primal variable against the materialized
    /// x_lower_/x_upper_ (free / lower-only / upper-only / two-sided / fixed),
    /// records the non-fixed finite bounds in variable_bound_set_, and -- under
    /// MakeParameter, when at least one variable is fixed -- builds the
    /// full<->reduced index maps and the KKT slot lists the per-assembly pin
    /// walks.
    ///
    /// Called once at solve setup. Idempotent and re-entrant: a call that
    /// repeats the same treatment, the same relax factor, and the same bound
    /// state returns immediately; a call after the bounds were re-materialized
    /// (make_nlp / clear_variable_bounds) or after the sparsity pattern was
    /// recomputed (analyze_sparsity) re-classifies from scratch, so one solver
    /// instance solving twice against changed bounds picks the change up.
    ///
    /// Requires analyze_sparsity() to have run when any variable is fixed: the
    /// per-assembly pin addresses KKT value slots by the offsets that routine
    /// computes. Throws std::invalid_argument for a treatment that is not yet
    /// available, for a negative or non-finite relax factor, for an infinite
    /// fixing value, and when the sparsity pattern is missing.
    /// </summary>
    void configure_variable_treatment(FixedVariableTreatments treatment, double bound_relax_factor);

    /// Number of primal variables the solve actually optimizes over: primal_vars_
    /// minus the eliminated ones. Equal to primal_vars_ on the identity path.
    /// Note that the KKT system stays primal_vars_ wide either way (see the
    /// reduction note above); this is the problem's degree-of-freedom count, and
    /// the size gather_reduced_x/scatter_full_x compact to and expand from.
    int reduced_primal_vars() const { return this->reduced_primal_vars_count_; }

    /// True iff at least one variable was eliminated by the configured
    /// treatment. THE guard for every piece of reduction work on the evaluation
    /// path: when it is false the solver runs exactly the code it ran before the
    /// treatment existed.
    bool is_reduced() const { return this->fixed_reduction_active_; }

    /// The finite bounds that survived classification (eliminated variables
    /// excluded), relaxed by the configured factor. Empty before
    /// configure_variable_treatment has run.
    const BoundSet &variable_bound_set() const { return this->variable_bound_set_; }

    /// Indices of the eliminated variables, ascending. Empty on the identity path.
    const VectorXi &fixed_variable_indices() const { return this->fixed_idx_; }

    /// Values the eliminated variables are pinned at, parallel to
    /// fixed_variable_indices().
    const VectorXd &fixed_variable_values() const { return this->fixed_vals_; }

    /// Full-space index -> compacted index, -1 for an eliminated variable. Empty
    /// on the identity path (where the map is the identity).
    const VectorXi &full_to_reduced() const { return this->full_to_reduced_; }

    /// Compacted index -> full-space index. Empty on the identity path.
    const VectorXi &reduced_to_full() const { return this->reduced_to_full_; }

    /// <summary>
    /// Compacts a full-space primal vector into reduced_primal_vars() entries,
    /// dropping the eliminated coordinates. Pass-through copy on the identity
    /// path. Throws std::invalid_argument on a size mismatch.
    /// </summary>
    void gather_reduced_x(ConstEigenRef<VectorXd> x_full, EigenRef<VectorXd> x_reduced) const;

    /// <summary>
    /// Expands a reduced primal vector back to primal_vars_ entries, writing each
    /// eliminated coordinate's pinned value. Pass-through copy on the identity
    /// path. Throws std::invalid_argument on a size mismatch.
    /// </summary>
    void scatter_full_x(ConstEigenRef<VectorXd> x_reduced, EigenRef<VectorXd> x_full) const;

    /// <summary>
    /// Writes the pinned value of every eliminated variable into a full-space
    /// primal vector, leaving the remaining entries alone. This is the ONE
    /// reinsertion seam: applied to the iterate at solve entry, it makes every
    /// downstream primal exposure -- the returned solution, the late callback's
    /// IterateInfo primals, every primal-exposing diagnostic -- read the
    /// full-space form with the fixed values already in it, with no second
    /// reinsertion anywhere.
    ///
    /// A caller that starts from an arbitrary guess gets the guess's value for a
    /// fixed variable silently replaced by the declared one; that is the point.
    /// No-op on the identity path. Throws std::invalid_argument on a size
    /// mismatch.
    /// </summary>
    void pin_fixed_variables(EigenRef<VectorXd> x_full) const;

    /// <summary>
    /// Zeroes the eliminated variables' entries of a full-space primal-row
    /// vector (an objective gradient or an adjoint gradient).
    ///
    /// An eliminated variable's stationarity row holds the bound multiplier that
    /// holds it at its bound, which is not part of the reduced problem's
    /// optimality residual. Clearing it makes the Newton right-hand side zero on
    /// that row -- matching the pinned unit KKT row, so the step in that
    /// coordinate is exactly zero -- and keeps it out of every primal residual
    /// norm. The multiplier itself is consequently NOT reported: the value that
    /// was cleared is the reduced-gradient residual at the solution, and
    /// recovering it belongs to the bound-multiplier work, not here.
    ///
    /// No-op on the identity path. Throws std::invalid_argument on a size
    /// mismatch.
    /// </summary>
    void clear_fixed_variable_rows(EigenRef<VectorXd> primal_rows) const;

    /// <summary>
    /// Post-assembly pin: zeroes every KKT value slot in an eliminated variable's
    /// row or column, then writes one onto each eliminated diagonal. Both slot
    /// lists are built once by configure_variable_treatment, so this is a flat
    /// walk of precomputed offsets with no per-element search.
    ///
    /// Runs at the END of every KKT assembly (see the eval_kkt family), after
    /// fill_solver_coeffs has added the solver's own primal diagonals, so the
    /// unit diagonal is the final value. A later `+=` onto the primal diagonals
    /// (perturb_kkt_p_diags, the inertia ladder) only makes it more positive and
    /// cannot couple the row back in.
    ///
    /// Only ever called under is_reduced(); the caller holds the guard.
    /// </summary>
    void pin_fixed_variable_kkt_rows(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) const {
        double *vals = mat.valuePtr();
        const int num_slots = static_cast<int>(this->fixed_kkt_slots_.size());
        for (int i = 0; i < num_slots; i++) {
            vals[this->fixed_kkt_slots_[i]] = 0.0;
        }
        const int num_diags = static_cast<int>(this->fixed_diag_slots_.size());
        for (int i = 0; i < num_diags; i++) {
            vals[this->fixed_diag_slots_[i]] = 1.0;
        }
    }

    /// <summary>
    /// Drops every piece of classification state that addresses the current KKT
    /// layout, and with it the is_reduced() guard, so no stale storage offset
    /// can survive into an assembly. Called wherever the KKT arrays are
    /// reallocated (set_mat_dimensions) or their offsets recomputed
    /// (analyze_sparsity); the next configure_variable_treatment call rebuilds
    /// from the live bounds and the live pattern.
    /// </summary>
    void invalidate_variable_treatment() {
        this->fixed_treatment_valid_ = false;
        this->fixed_reduction_active_ = false;
        this->fixed_kkt_slots_.resize(0);
        this->fixed_diag_slots_.resize(0);
    }

    /// Selected treatment, as last configured.
    FixedVariableTreatments variable_treatment_ = FixedVariableTreatments::MakeParameter;
    /// Relax factor, as last configured.
    double bound_relax_factor_ = 0.0;

    /// Bumped whenever x_lower_/x_upper_ are (re)materialized or dropped, so
    /// configure_variable_treatment can tell a repeat call from a real change.
    /// Staging a declaration does NOT bump it: staged declarations only reach
    /// x_lower_/x_upper_ through make_nlp, which does.
    long bounds_revision_ = 0;

    /// True once a classification pass has produced state consistent with the
    /// current bounds AND the current sparsity pattern. analyze_sparsity clears
    /// it (the KKT slot offsets it recomputes invalidate the pin lists).
    bool fixed_treatment_valid_ = false;
    long configured_bounds_revision_ = -1;

    /// True once analyze_sparsity has computed kkt_locations_.
    bool sparsity_analyzed_ = false;

    /// Cached is_reduced() answer -- the single bool every evaluation-path guard
    /// reads.
    bool fixed_reduction_active_ = false;
    int reduced_primal_vars_count_ = 0;

    VectorXi full_to_reduced_; ///< primal_vars_ entries, -1 where eliminated.
    VectorXi reduced_to_full_; ///< reduced_primal_vars_count_ entries.
    VectorXi fixed_idx_;       ///< Eliminated variable indices, ascending.
    VectorXd fixed_vals_;      ///< Their pinned values.

    /// KKT value-array offsets of every entry in an eliminated variable's row or
    /// column, sorted and deduplicated.
    VectorXi fixed_kkt_slots_;
    /// KKT value-array offsets of the eliminated variables' diagonals.
    VectorXi fixed_diag_slots_;

    BoundSet variable_bound_set_;

    void print_data() {
        for (int i = 0; i < this->num_partitions_; i++) {
            std::cout << "Partition: " << i << std::endl << std::endl;
            std::cout << "---------------objectives_---------------" << std::endl << std::endl;

            for (auto &obj : this->part_obj_[i]) {
                obj.print_data();
            }

            std::cout << "---------------Equalities---------------" << std::endl << std::endl;

            for (auto &eq : this->part_eq_[i]) {
                eq.print_data();
            }
            std::cout << "--------------Inequalities--------------" << std::endl << std::endl;

            for (auto &ineq : this->part_iq_[i]) {
                ineq.print_data();
            }
        }
    }

    void count_elems();

    void materialize_variable_bounds();

    void analyze_partitioning();

    void get_mat_space();

    void get_rhs_space();

    void set_mat_dimensions();

    void set_rhs_dimensions();

    void finalize_data();

    void analyze_sparsity(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void make_compressed() {
        this->kkt_coeff_part_ids_.resize(0);
        this->kkt_coeff_rows_.resize(0);
        this->kkt_coeff_cols_.resize(0);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> slack_coeffs() {
        return this->solver_coeffs_.segment(this->slack_jac_data_start_, this->slack_vars_);
    }
    EigenRef<VectorXd> primal_diag_coeffs() {
        return this->solver_coeffs_.segment(this->primal_diags_data_start_, this->primal_vars_);
    }
    EigenRef<VectorXd> slack_diag_coeffs() {
        return this->solver_coeffs_.segment(this->slack_diag_data_start_, this->slack_vars_);
    }
    EigenRef<VectorXd> e_pivot_coeffs() {
        return this->solver_coeffs_.segment(this->e_pivot_data_start_, this->equal_cons_);
    }
    EigenRef<VectorXd> i_pivot_coeffs() {
        return this->solver_coeffs_.segment(this->i_pivot_data_start_, this->inequal_cons_);
    }

    EigenRef<VectorXi> slack_coeff_cols() {
        return this->kkt_coeff_cols_.segment(
            this->slack_jac_data_start_ + this->num_user_kkt_elems_, this->slack_vars_);
    }
    EigenRef<VectorXi> primal_diag_coeff_cols() {
        return this->kkt_coeff_cols_.segment(
            this->primal_diags_data_start_ + this->num_user_kkt_elems_, this->primal_vars_);
    }
    EigenRef<VectorXi> slack_diag_coeff_cols() {
        return this->kkt_coeff_cols_.segment(
            this->slack_diag_data_start_ + this->num_user_kkt_elems_, this->slack_vars_);
    }
    EigenRef<VectorXi> e_pivot_coeff_cols() {
        return this->kkt_coeff_cols_.segment(this->e_pivot_data_start_ + this->num_user_kkt_elems_,
                                             this->equal_cons_);
    }
    EigenRef<VectorXi> i_pivot_coeff_cols() {
        return this->kkt_coeff_cols_.segment(this->i_pivot_data_start_ + this->num_user_kkt_elems_,
                                             this->inequal_cons_);
    }

    EigenRef<VectorXi> slack_coeff_rows() {
        return this->kkt_coeff_rows_.segment(
            this->slack_jac_data_start_ + this->num_user_kkt_elems_, this->slack_vars_);
    }
    EigenRef<VectorXi> primal_diag_coeff_rows() {
        return this->kkt_coeff_rows_.segment(
            this->primal_diags_data_start_ + this->num_user_kkt_elems_, this->primal_vars_);
    }
    EigenRef<VectorXi> slack_diag_coeff_rows() {
        return this->kkt_coeff_rows_.segment(
            this->slack_diag_data_start_ + this->num_user_kkt_elems_, this->slack_vars_);
    }
    EigenRef<VectorXi> e_pivot_coeff_rows() {
        return this->kkt_coeff_rows_.segment(this->e_pivot_data_start_ + this->num_user_kkt_elems_,
                                             this->equal_cons_);
    }
    EigenRef<VectorXi> i_pivot_coeff_rows() {
        return this->kkt_coeff_rows_.segment(this->i_pivot_data_start_ + this->num_user_kkt_elems_,
                                             this->inequal_cons_);
    }

    void set_primal_diags(const Eigen::VectorXd &pdiags) { this->primal_diag_coeffs() = pdiags; }
    void set_primal_diags(double val) { this->primal_diag_coeffs().setConstant(val); }
    void set_slack_diags(const Eigen::VectorXd &sdiags) { this->slack_diag_coeffs() = sdiags; }
    void set_slack_diags(double val) { this->slack_diag_coeffs().setConstant(val); }
    void set_e_pivots(const Eigen::VectorXd &epivs) { this->e_pivot_coeffs() = epivs; }
    void set_e_pivots(double val) { this->e_pivot_coeffs().setConstant(val); }
    void set_i_pivots(const Eigen::VectorXd &ipivs) { this->i_pivot_coeffs() = ipivs; }
    void set_i_pivots(double val) { this->i_pivot_coeffs().setConstant(val); }
    void set_slacks_ones() { this->slack_coeffs().setConstant(1.0); }

    void fill_solver_coeffs(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
        auto FillOp = [&](int start, int stop) {
            for (int i = start; i < stop; i++) {
                mat.valuePtr()[this->kkt_locations_.tail(this->num_solver_kkt_elems_)[i]] +=
                    this->solver_coeffs_[i];
            }
        };

        tycho::utils::parallel_blocks(this->num_solver_kkt_elems_, FillOp, this->num_partitions_);
    }

    void assign_kkt_slack_hessian(const Eigen::Ref<const Eigen::VectorXd> &slhs,
                                  Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
        int ofs = this->slack_diag_data_start_ + this->num_user_kkt_elems_;
        for (int i = 0; i < this->slack_vars_; i++) {
            mat.valuePtr()[this->kkt_locations_[ofs + i]] = slhs[i];
        }
    }
    void perturb_kkt_p_diags(double pert, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
        int ofs = this->primal_diags_data_start_ + this->num_user_kkt_elems_;
        for (int i = 0; i < this->primal_vars_; i++) {
            mat.valuePtr()[this->kkt_locations_[ofs + i]] += pert;
        }
    }
    // Post-assembly `+=` onto every constraint-row diagonal slot (the equality
    // and inequality pivot ranges), the mirror of perturb_kkt_p_diags over the
    // constraint block. Used by the proximal primal-dual regularization mode to
    // apply the dual shift (−δ_c) as part of the base matrix, after the KKT
    // assembly and before the first factorization; on the default path these
    // slots are 0.0 and this helper is never called.
    void perturb_kkt_c_diags(double pert, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
        int eofs = this->e_pivot_data_start_ + this->num_user_kkt_elems_;
        for (int i = 0; i < this->equal_cons_; i++) {
            mat.valuePtr()[this->kkt_locations_[eofs + i]] += pert;
        }
        int iofs = this->i_pivot_data_start_ + this->num_user_kkt_elems_;
        for (int i = 0; i < this->inequal_cons_; i++) {
            mat.valuePtr()[this->kkt_locations_[iofs + i]] += pert;
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> pgx_coeffs() {
        return this->rhs_coeffs_.segment(this->pgx_data_start_, this->num_pgx_elems_);
    }
    EigenRef<VectorXd> agx_coeffs() {
        return this->rhs_coeffs_.segment(this->agx_data_start_, this->num_agx_elems_);
    }
    EigenRef<VectorXd> econ_coeffs() {
        return this->rhs_coeffs_.segment(this->econ_data_start_, this->num_econ_elems_);
    }
    EigenRef<VectorXd> icon_coeffs() {
        return this->rhs_coeffs_.segment(this->icon_data_start_, this->num_icon_elems_);
    }

    EigenRef<VectorXi> pgx_coeff_rows() {
        return this->rhs_coeff_rows_.segment(this->pgx_data_start_, this->num_pgx_elems_);
    }
    EigenRef<VectorXi> agx_coeff_rows() {
        return this->rhs_coeff_rows_.segment(this->agx_data_start_, this->num_agx_elems_);
    }
    EigenRef<VectorXi> econ_coeff_rows() {
        return this->rhs_coeff_rows_.segment(this->econ_data_start_, this->num_econ_elems_);
    }
    EigenRef<VectorXi> icon_coeff_rows() {
        return this->rhs_coeff_rows_.segment(this->icon_data_start_, this->num_icon_elems_);
    }

    EigenRef<VectorXi> get_kkt_locations() {
        return this->kkt_locations_.head(this->num_user_kkt_elems_);
    }
    EigenRef<VectorXi> get_kkt_clashes() { return this->kkt_clashes_.head(this->primal_vars_); }

    void set_con_coeffs_zero() {
        this->econ_coeffs().setZero();
        this->icon_coeffs().setZero();
    }
    void set_pgx_coeffs_zero() { this->pgx_coeffs().setZero(); }
    void set_agx_coeffs_zero() { this->agx_coeffs().setZero(); }
    void set_rhs_coeffs_zero() {
        this->set_pgx_coeffs_zero();
        this->set_agx_coeffs_zero();
        this->set_con_coeffs_zero();
    }

    void fill_pgx(EigenRef<VectorXd> PGX) {
        this->rhs_fill_op(PGX, this->pgx_coeffs(), this->pgx_coeff_rows());
    }
    void fill_agx(EigenRef<VectorXd> AGX) {
        this->rhs_fill_op(AGX, this->agx_coeffs(), this->agx_coeff_rows());
    }
    void fill_fxe(EigenRef<VectorXd> FXE) {
        this->rhs_fill_op(FXE, this->econ_coeffs(), this->econ_coeff_rows());
    }
    void fill_fxi(EigenRef<VectorXd> FXI) {
        this->rhs_fill_op(FXI, this->icon_coeffs(), this->icon_coeff_rows());
    }
    void fill_rhs(EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE,
                  EigenRef<VectorXd> FXI) {
        this->fill_pgx(PGX);
        this->fill_agx(AGX);
        this->fill_fxe(FXE);
        this->fill_fxi(FXI);
    }

    static void rhs_fill_op(EigenRef<VectorXd> target, EigenRef<VectorXd> source,
                            EigenRef<VectorXi> sourcelocs) {
        for (int i = 0; i < source.size(); i++) {
            target[sourcelocs[i]] += source[i];
        }
    }

    /// <summary>
    /// Per-partition objective/value accumulator scratch, shared across
    /// eval_rhs/eval_ogc/eval_occ/eval_obj/eval_kkt/eval_aug. Each of those
    /// entry points is only ever invoked serially on this NLP instance -- a
    /// single NLP is inside at most one alg_impl call at a time (PSIOPT's
    /// outer control loop is single-threaded; the only concurrency is the
    /// parallel_sequence dispatch *within* one call, which writes disjoint
    /// vals_scratch_[thrnum] entries, exactly as the old per-call local
    /// `Vals` vector did). Resized in place (assign() re-zeros without a
    /// realloc once sized to num_partitions_).
    /// </summary>
    std::vector<double> vals_scratch_;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void eval_rhs(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                  ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                  EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI);

    void eval_ogc(double ObjScale, ConstEigenRef<VectorXd> X, double &val, EigenRef<VectorXd> PGX,
                  EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI);

    void eval_occ(double ObjScale, ConstEigenRef<VectorXd> X, double &val, EigenRef<VectorXd> FXE,
                  EigenRef<VectorXd> FXI);

    void eval_obj(double ObjScale, ConstEigenRef<VectorXd> X, double &val);

    void eval_kkt(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                  ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                  EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    void eval_kkt_no(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                     ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                     EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                     Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    void eval_soe(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                  ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                  EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    void eval_aug(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                  ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                  EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    static void nlp_test(const Eigen::VectorXd &x, int n, std::shared_ptr<NonLinearProgram> nlp1,
                         std::shared_ptr<NonLinearProgram> nlp2);
};

} // namespace tycho::solvers
