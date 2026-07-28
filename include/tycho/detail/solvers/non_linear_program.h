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
    // A variable declared with lower == upper carries no degree of freedom.
    // Under the MakeParameter treatment it is ELIMINATED: it does not appear in
    // the solver's variable space at all, and the KKT system the solver
    // factorizes is the system of the remaining variables -- narrower by exactly
    // one row and column per eliminated variable.
    //
    // How the elimination is realized. Each objective and constraint addresses
    // primal variables by index, and does so for two different purposes: to
    // GATHER its arguments out of the primal vector, and to say where its
    // outputs (KKT columns, gradient rows) belong. Those two roles are split
    // here. The input map is left alone -- a function still reads exactly the
    // variables it was declared over -- and every evaluation hands it a
    // full-space buffer built from the reduced iterate plus the pinned values,
    // so the functions never learn anything happened and their contributions to
    // constraint values and to the surviving variables' derivatives are correct
    // for free. Only the OUTPUT map is rewritten, at configuration time, from
    // the pristine input map:
    //
    //   * retained variables' outputs are renumbered into the reduced space;
    //   * eliminated variables' outputs are marked -1;
    //   * the KKT/RHS location tables, the sparsity pattern, the partition clash
    //     marks and the solver-coefficient ranges are all rebuilt over that
    //     rewritten map.
    //
    // Element CLAIMS stay exactly as they were -- same count, same contiguous
    // per-application ranges -- because the scatters walk their claims in
    // lockstep with the function's own loop bounds. A -1 element keeps its claim
    // and simply names no matrix entry: analyze_sparsity leaves it out of the
    // pattern, so the eliminated row and column never exist, and the scatter
    // steps over it instead of writing.
    //
    // Identity fast path. With no fixed variables nothing is rewritten at all:
    // no output map is installed, so every scatter takes its original loop,
    // every table is the pristine one, no expansion buffer is built, and the
    // assembled system is what it was before this feature existed.
    ////////////////////////////////////////////////////////////////////////////

    /// <summary>
    /// Classifies every primal variable against the materialized
    /// x_lower_/x_upper_ (free / lower-only / upper-only / two-sided / fixed),
    /// records the non-fixed finite bounds in variable_bound_set_, and -- under
    /// MakeParameter, when at least one variable is fixed -- rewrites every
    /// function's output map into the reduced space and rebuilds the KKT/RHS
    /// structures over it.
    ///
    /// Called once at solve setup. Idempotent and re-entrant: a call that
    /// repeats the same treatment, the same relax factor, and the same bound
    /// state returns immediately having changed nothing; a call after the bounds
    /// were re-materialized (make_nlp / clear_variable_bounds) reclassifies from
    /// scratch. Every rewritten output map is regenerated from the pristine
    /// input map, which is never edited, so repeated configuration cannot
    /// compound -- including the return to the identity path when the last fixed
    /// bound is dropped.
    ///
    /// Throws std::invalid_argument for a treatment that is not yet available,
    /// for a negative or non-finite relax factor, and for an infinite fixing
    /// value.
    ///
    /// @return true iff this call rebuilt the KKT/RHS structures, in which case
    /// the caller must re-read the dimensions and recompute the sparsity pattern
    /// (PSIOPT does both at its solve entry). false means nothing changed.
    /// </summary>
    bool configure_variable_treatment(FixedVariableTreatments treatment, double bound_relax_factor);

    /// Width of the primal block of the KKT system the solver factorizes:
    /// primal_vars_ minus the eliminated variables. This is the size of every
    /// primal vector the solver works with, and the size gather_reduced_x
    /// compacts to and scatter_full_x expands from. Equal to primal_vars_ on the
    /// identity path.
    int reduced_primal_vars() const { return this->reduced_primal_vars_count_; }

    /// True iff at least one variable was eliminated by the configured
    /// treatment. The guard for every piece of reduction work outside the
    /// scatters (which carry their own, per-function): when it is false the
    /// solver runs exactly the code it ran before the treatment existed.
    bool is_reduced() const { return this->fixed_reduction_active_; }

    /// The finite bounds that survived classification (eliminated variables
    /// excluded), in REDUCED indices -- the same space every primal vector the
    /// solver holds is in -- and relaxed by the configured factor. Empty before
    /// configure_variable_treatment has run.
    const BoundSet &variable_bound_set() const { return this->variable_bound_set_; }

    /// Indices of the eliminated variables in the FULL problem space, ascending.
    /// Empty on the identity path.
    const VectorXi &fixed_variable_indices() const { return this->fixed_idx_; }

    /// Values the eliminated variables are pinned at, parallel to
    /// fixed_variable_indices().
    const VectorXd &fixed_variable_values() const { return this->fixed_vals_; }

    /// Full-space index -> reduced index, -1 for an eliminated variable. Empty
    /// on the identity path (where the map is the identity).
    const VectorXi &full_to_reduced() const { return this->full_to_reduced_; }

    /// Reduced index -> full-space index. Empty on the identity path.
    const VectorXi &reduced_to_full() const { return this->reduced_to_full_; }

    /// <summary>
    /// Compacts a full-space primal vector into reduced_primal_vars() entries,
    /// dropping the eliminated coordinates. Used at the solve entry to turn a
    /// caller's initial guess into the solver's iterate. Pass-through copy on
    /// the identity path. Throws std::invalid_argument on a size mismatch.
    /// </summary>
    void gather_reduced_x(ConstEigenRef<VectorXd> x_full, EigenRef<VectorXd> x_reduced) const;

    /// <summary>
    /// Expands a reduced primal vector back to primal_vars_ entries, writing
    /// each eliminated coordinate's pinned value.
    ///
    /// Two callers, and they are the whole reinsertion story. Every evaluation
    /// uses it to build the full-space buffer the functions read, which is why
    /// they never learn a variable was eliminated. And the solver uses it once
    /// at its return boundary to hand the solution back in the caller's own
    /// variable space -- the single seam where the reduced space stops being an
    /// internal detail.
    ///
    /// Pass-through copy on the identity path. Throws std::invalid_argument on a
    /// size mismatch.
    /// </summary>
    void scatter_full_x(ConstEigenRef<VectorXd> x_reduced, EigenRef<VectorXd> x_full) const;

    /// <summary>
    /// The full-space primal vector the objective and constraint functions must
    /// read, given the solver's reduced iterate: the reduced values put back in
    /// their own coordinates, with the eliminated ones pinned at their declared
    /// values.
    ///
    /// Returns a view of @p x itself on the identity path -- no buffer, no copy.
    /// Otherwise returns a view of an internal buffer, valid until the next
    /// call. Like vals_scratch_, that buffer is safe to share because a single
    /// NLP is inside at most one evaluation at a time; the parallelism inside an
    /// evaluation only reads it.
    /// </summary>
    Eigen::Ref<const VectorXd> primal_view(ConstEigenRef<VectorXd> x) {
        if (!this->fixed_reduction_active_) {
            return x;
        }
        this->full_x_scratch_.resize(this->primal_vars_);
        this->scatter_full_x(x, this->full_x_scratch_);
        return this->full_x_scratch_;
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

    /// True once a classification pass has produced structures consistent with
    /// the current bounds. make_nlp clears it: it rebuilds the whole layout from
    /// the pristine full space, so whatever the previous configuration derived
    /// is gone.
    bool fixed_treatment_valid_ = false;
    long configured_bounds_revision_ = -1;

    /// Cached is_reduced() answer -- the bool the evaluation-path guards outside
    /// the scatters read.
    bool fixed_reduction_active_ = false;
    int reduced_primal_vars_count_ = 0;

    VectorXi full_to_reduced_; ///< primal_vars_ entries, -1 where eliminated.
    VectorXi reduced_to_full_; ///< reduced_primal_vars_count_ entries.
    VectorXi fixed_idx_;       ///< Eliminated variable indices, ascending.
    VectorXd fixed_vals_;      ///< Their pinned values.

    /// Full-space buffer the objective and constraint functions read while the
    /// solver iterates in the reduced space -- reduced values in their own
    /// coordinates plus the pinned ones. Empty and untouched on the identity
    /// path. See primal_view().
    VectorXd full_x_scratch_;

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

    /// The set_mat_dimensions -> finalize_data chain: everything whose layout
    /// depends on the width of the solver's primal block. Run once by make_nlp
    /// over the full space, and again by configure_variable_treatment whenever
    /// that width changes.
    void rebuild_structures();

    /// Rewrites every partitioned function's output map into the reduced space
    /// from its pristine input map (-1 where the variable is eliminated).
    void install_function_output_maps();

    /// Puts every partitioned function back on its pristine input map.
    void clear_function_output_maps();

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
        return this->solver_coeffs_.segment(this->primal_diags_data_start_,
                                            this->reduced_primal_vars_count_);
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
        return this->kkt_coeff_cols_.segment(this->primal_diags_data_start_ +
                                                 this->num_user_kkt_elems_,
                                             this->reduced_primal_vars_count_);
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
        return this->kkt_coeff_rows_.segment(this->primal_diags_data_start_ +
                                                 this->num_user_kkt_elems_,
                                             this->reduced_primal_vars_count_);
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
        for (int i = 0; i < this->reduced_primal_vars_count_; i++) {
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
    EigenRef<VectorXi> get_kkt_clashes() {
        return this->kkt_clashes_.head(this->reduced_primal_vars_count_);
    }

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

    // Gradient rows for eliminated variables are marked -1 by get_gradient_space
    // (the claim keeps its size -- the function still writes a gradient value
    // per argument -- but that value has no row to be summed into). The skip is
    // hoisted to the call, so the unreduced fill loop is the original one and
    // the whole cost of the feature here is one branch per fill.
    void fill_pgx(EigenRef<VectorXd> PGX) {
        if (this->fixed_reduction_active_) {
            rhs_fill_op_reduced(PGX, this->pgx_coeffs(), this->pgx_coeff_rows());
        } else {
            rhs_fill_op(PGX, this->pgx_coeffs(), this->pgx_coeff_rows());
        }
    }
    void fill_agx(EigenRef<VectorXd> AGX) {
        if (this->fixed_reduction_active_) {
            rhs_fill_op_reduced(AGX, this->agx_coeffs(), this->agx_coeff_rows());
        } else {
            rhs_fill_op(AGX, this->agx_coeffs(), this->agx_coeff_rows());
        }
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

    /// rhs_fill_op over a location table that may carry -1 for the gradient rows
    /// of eliminated variables. Those rows are dropped: an eliminated variable's
    /// stationarity row is not part of the reduced problem's residual, and its
    /// value at a solution is the bound multiplier that holds the variable at
    /// its bound. Reached only under is_reduced().
    static void rhs_fill_op_reduced(EigenRef<VectorXd> target, EigenRef<VectorXd> source,
                                    EigenRef<VectorXi> sourcelocs) {
        for (int i = 0; i < source.size(); i++) {
            const int row = sourcelocs[i];
            if (row >= 0) {
                target[row] += source[i];
            }
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
