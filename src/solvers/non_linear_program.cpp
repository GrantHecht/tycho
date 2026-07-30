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

#include <fmt/format.h>

#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/utils/timer.h"

void tycho::solvers::NonLinearProgram::make_nlp(int PV, int EQ, int IQ) {
    // A previous configuration's internal fixing rows describe the bounds as they
    // were then, and this call re-materializes those bounds from scratch -- so the
    // rows go first, and the next configure_variable_treatment call derives its
    // own. Dropped before equal_cons_ is set from EQ, which is the USER's own row
    // count and knows nothing about them.
    this->discard_fixed_variable_rows();

    // The one invariant that would catch an internal-row bookkeeping slip before
    // it can corrupt a scatter: after the discard nothing internal may remain, and
    // the equality row space must be back at the count the last make_nlp was
    // handed. A logic_error rather than an assert because asserts are compiled out
    // in every shipped configuration, and this is not a user-facing rejection --
    // no argument can trip it, only a defect in the install/discard pair.
    if (this->internal_fixed_cons_ != 0 || this->equal_cons_ != this->user_equal_cons_) {
        throw std::logic_error(
            fmt::format("make_nlp: internal fixing rows were not fully discarded (internal {0}, "
                        "equal_cons {1}, user_equal_cons {2})",
                        this->internal_fixed_cons_, this->equal_cons_, this->user_equal_cons_));
    }

    this->primal_vars_ = PV;
    this->user_equal_cons_ = EQ;
    this->equal_cons_ = EQ;
    this->inequal_cons_ = IQ;
    this->slack_vars_ = IQ;

    // Everything below is laid out over the full variable space. A previous
    // configuration's reduction does not survive a re-transcription: its output
    // maps are dropped and its structures are about to be rebuilt from scratch,
    // so the next configure_variable_treatment call starts from the pristine
    // state and re-derives whatever the new bounds call for.
    this->reduced_primal_vars_count_ = PV;
    this->fixed_reduction_active_ = false;
    this->fixed_treatment_valid_ = false;
    this->clear_function_output_maps();

    this->materialize_variable_bounds();

    this->count_elems();

    // Cap partitions so each has enough work to offset dispatch overhead.
    // num_user_kkt_elems_ counts Jacobian + Hessian NNZ across all functions —
    // proportional to per-partition compute in eval_kkt/eval_aug. Below ~1000
    // NNZ per partition, dispatch overhead dominates actual work.
    // Threshold empirically chosen via solver benchmarks (bench_all);
    // re-evaluate with bench/bench_track.sh if dispatch overhead changes.
    if (this->num_partitions_ > 1) {
        static constexpr int MIN_NNZ_PER_PARTITION = 1000;
        int max_parts = std::max(1, this->num_user_kkt_elems_ / MIN_NNZ_PER_PARTITION);
        this->num_partitions_ = std::min(this->num_partitions_, max_parts);
    }

    this->analyze_partitioning();
    this->rebuild_structures();
}

void tycho::solvers::NonLinearProgram::rebuild_structures() {
    this->set_mat_dimensions();
    this->set_rhs_dimensions();

    this->get_mat_space();
    this->get_rhs_space();
    this->finalize_data();
}

void tycho::solvers::NonLinearProgram::refresh_function_partitions() {
    this->count_elems();
    this->analyze_partitioning();
}

void tycho::solvers::NonLinearProgram::install_fixed_variable_rows() {
    const int count = static_cast<int>(this->fixed_idx_.size());

    // Built into a local first, spliced onto the master list only once every row
    // exists. All-or-nothing on purpose: a throw part-way through the
    // construction would otherwise leave rows on the master list that
    // internal_fixed_cons_ does not yet count, and a count that does not see them
    // is a count nothing can repair -- the restore's discard would skip them, the
    // element counts and the partitioning would not include them, and the next
    // successful install would append on top, giving two functions the same
    // constraint row to scatter into. There is no such throw today (a non-finite
    // fixing value is rejected during classification, before this runs, and the
    // factory's own validation is defence in depth), and this is what keeps that
    // from being load-bearing.
    std::vector<ConstraintFunction> rows;
    rows.reserve(count);
    for (int k = 0; k < count; k++) {
        // Numbered after every row the transcription declared, so the user's own
        // constraint rows keep the indices they were given and the multipliers
        // that come back are still theirs.
        rows.push_back(make_fixed_variable_row(this->fixed_idx_[k], this->fixed_vals_[k],
                                               this->user_equal_cons_ + k));
    }

    const std::size_t before = this->equality_constraints_.size();
    try {
        this->equality_constraints_.reserve(before + rows.size());
        for (auto &row : rows) {
            this->equality_constraints_.push_back(std::move(row));
        }
    } catch (...) {
        // Truncating back is a no-op when nothing landed, and cannot throw. What
        // it buys is that the list and internal_fixed_cons_ (still 0 here) agree
        // on every path out of this function.
        this->equality_constraints_.resize(before);
        throw;
    }
    this->internal_fixed_cons_ = count;
    this->equal_cons_ = this->user_equal_cons_ + count;
}

void tycho::solvers::NonLinearProgram::discard_fixed_variable_rows() {
    if (this->internal_fixed_cons_ <= 0) {
        return;
    }
    const int installed = this->internal_fixed_cons_;
    const int total = static_cast<int>(this->equality_constraints_.size());
    if (total < installed) {
        throw std::logic_error(
            fmt::format("discard_fixed_variable_rows: {0} internal fixing rows are recorded but "
                        "the equality list holds only {1} functions",
                        installed, total));
    }
    this->equality_constraints_.resize(total - installed);
    this->internal_fixed_cons_ = 0;
    this->equal_cons_ = this->user_equal_cons_;
}

void tycho::solvers::NonLinearProgram::clear_function_output_maps() {
    auto clear_all = [](auto &funcs) {
        for (auto &func : funcs) {
            func.index_data_.clear_output_v_index();
        }
    };
    for (int i = 0; i < static_cast<int>(this->part_obj_.size()); i++) {
        clear_all(this->part_obj_[i]);
    }
    for (int i = 0; i < static_cast<int>(this->part_eq_.size()); i++) {
        clear_all(this->part_eq_[i]);
    }
    for (int i = 0; i < static_cast<int>(this->part_iq_.size()); i++) {
        clear_all(this->part_iq_[i]);
    }
}

void tycho::solvers::NonLinearProgram::install_function_output_maps() {
    // Always regenerated from the pristine input map, never edited in place, so
    // repeated configuration cannot compound a previous remapping.
    auto install = [&](auto &funcs) {
        for (auto &func : funcs) {
            const Eigen::MatrixXi &v_in = func.index_data_.get_v_index();
            Eigen::MatrixXi v_out(v_in.rows(), v_in.cols());
            for (int col = 0; col < v_in.cols(); col++) {
                for (int row = 0; row < v_in.rows(); row++) {
                    const int global = v_in(row, col);
                    if (global < 0 || global >= this->primal_vars_) {
                        throw std::logic_error(fmt::format(
                            "configure_variable_treatment: function variable index {0} is "
                            "outside [0, {1})",
                            global, this->primal_vars_));
                    }
                    v_out(row, col) = this->full_to_reduced_[global];
                }
            }
            func.index_data_.set_output_v_index(v_out);
        }
    };
    for (int i = 0; i < static_cast<int>(this->part_obj_.size()); i++) {
        install(this->part_obj_[i]);
    }
    for (int i = 0; i < static_cast<int>(this->part_eq_.size()); i++) {
        install(this->part_eq_[i]);
    }
    for (int i = 0; i < static_cast<int>(this->part_iq_.size()); i++) {
        install(this->part_iq_[i]);
    }
}

void tycho::solvers::NonLinearProgram::materialize_variable_bounds() {
    constexpr double kInf = std::numeric_limits<double>::infinity();

    // Any re-materialization invalidates a previous classification, whether the
    // declared bounds changed or only primal_vars_ did.
    this->bounds_revision_++;

    this->x_lower_ = Eigen::VectorXd::Constant(this->primal_vars_, -kInf);
    this->x_upper_ = Eigen::VectorXd::Constant(this->primal_vars_, kInf);

    for (const auto &stage : this->staged_variable_bounds_) {
        if (stage.index_ < 0 || stage.index_ >= this->primal_vars_) {
            throw std::invalid_argument(
                fmt::format("set_variable_bound: index {0} out of range [0, {1})", stage.index_,
                            this->primal_vars_));
        }

        double &lower = this->x_lower_[stage.index_];
        double &upper = this->x_upper_[stage.index_];
        lower = std::max(lower, stage.lower_);
        upper = std::min(upper, stage.upper_);

        if (lower > upper) {
            throw std::invalid_argument(
                fmt::format("set_variable_bound: conflicting bounds for index {0}: "
                            "lower={1}, upper={2}",
                            stage.index_, lower, upper));
        }
    }
}

void tycho::solvers::NonLinearProgram::count_elems() {
    int nkkt = 0;

    int npgx = 0;
    int nagx = 0;
    int nec = 0;
    int nic = 0;

    for (auto &obj : this->objectives_) {
        nkkt += obj.num_kkt_elements(false, true);
        npgx += obj.num_grad_eles();
    }
    for (auto &eq : this->equality_constraints_) {
        nkkt += eq.num_kkt_elements(true, true);
        nagx += eq.num_grad_eles();
        nec += eq.num_con_eles();
    }
    for (auto &ineq : this->inequality_constraints_) {
        nkkt += ineq.num_kkt_elements(true, true);
        nagx += ineq.num_grad_eles();
        nic += ineq.num_con_eles();
    }

    this->num_user_kkt_elems_ = nkkt;
    this->num_pgx_elems_ = npgx;
    this->num_agx_elems_ = nagx;
    this->num_icon_elems_ = nic;
    this->num_econ_elems_ = nec;
}

void tycho::solvers::NonLinearProgram::analyze_partitioning() {
    /*
    This function loops over the master list of objectives and constraints and assigns them
    to num_partitions_ work partitions. Each partition's work is dispatched as a single task
    to the global thread pool.
    */
    this->part_obj_.clear();
    this->part_eq_.clear();
    this->part_iq_.clear();

    this->part_obj_.resize(this->num_partitions_);
    this->part_eq_.resize(this->num_partitions_);
    this->part_iq_.resize(this->num_partitions_);

    int rrPart = 0;

    auto analyzeOP = [&](auto &SourceFuncs, auto &TargetPartFuncs) {
        for (auto &func : SourceFuncs) {
            if (func.get_thread_mode() ==
                ThreadingFlags::MainThread) { // Force to last partition — parallel_sequence runs
                                              // the last index inline on the calling thread, so
                                              // MainThread functions stay safe.
                TargetPartFuncs.back().push_back(func);
            } else if (func.get_thread_mode() == ThreadingFlags::RoundRobin) {
                TargetPartFuncs[rrPart].push_back(func);
                rrPart++;
                if (rrPart > (this->num_partitions_ - 1))
                    rrPart = 0;
            } else if (static_cast<int>(func.get_thread_mode()) >=
                       0) { // Specific Partition Assignment
                int part =
                    std::min(static_cast<int>(func.get_thread_mode()), this->num_partitions_ - 1);
                TargetPartFuncs[part].push_back(func);
            } else { // By application
                auto TempPartFuncs = func.thread_split(this->num_partitions_);
                for (int i = 0; i < TempPartFuncs.size(); i++) {
                    TargetPartFuncs[i].push_back(TempPartFuncs[i]);
                }
            }
        }
    };

    analyzeOP(this->objectives_, this->part_obj_);
    analyzeOP(this->equality_constraints_, this->part_eq_);
    analyzeOP(this->inequality_constraints_, this->part_iq_);
}

void tycho::solvers::NonLinearProgram::get_mat_space() {
    /*
     * Loops over all constraints and objectives on each partition and has each claim its
     * own portion of kkt_coeff_cols_,kkt_coeff_rows_. Tags each element with the partition that
     * will be operating on it, then from this info calculates which columns/rows of the KKT matrix
     * need to be locked when multiple partitions are scattering into the KKT matrix. Allocates
     * kkt_locks_ mutexes based on this info.
     *
     * Canonical-column locking protocol: every KKT scatter site (kkt_fill_all and
     * kkt_fill_hess in dense_function_base.h) locks each element's mutex on the slot's
     * canonical column -- tycho::solvers::kkt_canonical_lock_col(row, col), the smaller
     * endpoint, which is the same endpoint analyze_sparsity stores the physical slot
     * under. The clash detection below marks contested columns with that SAME shared
     * keying function, so cross-partition claimants of one physical slot agree on the
     * mutex BY CONSTRUCTION -- there is no per-site convention that can drift, and hence
     * no runtime check (a claims-map assertion that once lived here verified min == min
     * and was deleted as tautological; see kkt_canonical_lock_col's doc comment for the
     * structural argument). Writes within one partition need no mutual exclusion -- each
     * partition's scatter runs serially on a single thread (parallel_sequence dispatches
     * one task per partition), and single-partition problems take no locks at all (no
     * column can be claimed by more than one partition, so kkt_clashes_ is all -1).
     */

    int KKTfreeloc = 0;

    int eqoffset = this->reduced_primal_vars_count_ + this->slack_vars_;
    int iqoffset = this->reduced_primal_vars_count_ + this->slack_vars_ + this->equal_cons_;
    for (int i = 0; i < this->num_partitions_; i++) {
        int kkstart = KKTfreeloc;

        for (auto &obj : this->part_obj_[i])
            obj.get_kkt_space(this->kkt_coeff_rows_.head(this->num_user_kkt_elems_),
                              this->kkt_coeff_cols_.head(this->num_user_kkt_elems_), KKTfreeloc, 0,
                              false, true);
        for (auto &eq : this->part_eq_[i])
            eq.get_kkt_space(this->kkt_coeff_rows_.head(this->num_user_kkt_elems_),
                             this->kkt_coeff_cols_.head(this->num_user_kkt_elems_), KKTfreeloc,
                             eqoffset, true, true);
        for (auto &ineq : this->part_iq_[i])
            ineq.get_kkt_space(this->kkt_coeff_rows_.head(this->num_user_kkt_elems_),
                               this->kkt_coeff_cols_.head(this->num_user_kkt_elems_), KKTfreeloc,
                               iqoffset, true, true);

        int kklen = KKTfreeloc - kkstart;

        this->kkt_coeff_part_ids_.segment(kkstart, kklen).setConstant(i);
    }

    // Mark a KKT column contested iff >= 2 partitions write a slot whose CANONICAL column
    // (kkt_canonical_lock_col(row, col), the smaller endpoint) is that column -- the same
    // shared keying function every scatter site locks with, so a contested slot's writers
    // all map to the mutex allocated here. (Historically this keyed on the raw recorded
    // column kkt_coeff_cols_[i], i.e. the outer-loop variable, which mis-attributed
    // mirror-order Hessian writes and left genuinely shared slots unlocked -- the race
    // this keying closes.)
    Eigen::MatrixXi KKTclash(this->num_partitions_, this->kkt_dim_);
    KKTclash.setZero();
    for (int i = 0; i < this->num_user_kkt_elems_; i++) {
        // An element belonging to an eliminated variable names no matrix entry
        // (get_kkt_space recorded it as (-1, -1)) and is never written, so it
        // must not mark a column contested either. Its own column does not exist
        // in this space at all.
        if (this->kkt_coeff_rows_[i] < 0 || this->kkt_coeff_cols_[i] < 0) {
            continue;
        }
        int lockcol = kkt_canonical_lock_col(this->kkt_coeff_rows_[i], this->kkt_coeff_cols_[i]);
        int thrid = this->kkt_coeff_part_ids_[i];
        KKTclash(thrid, lockcol) = 1;
    }

    this->kkt_clashes_.resize(this->kkt_dim_);
    this->num_kkt_clashes_ = 0;

    for (int i = 0; i < this->kkt_dim_; i++) {
        if (KKTclash.col(i).sum() > 1) {
            this->kkt_clashes_[i] = num_kkt_clashes_;
            num_kkt_clashes_++;
        } else {
            this->kkt_clashes_[i] = -1;
        }
    }
    std::vector<std::mutex> kktemp(this->num_kkt_clashes_);

    this->kkt_locks_.swap(kktemp);
}

void tycho::solvers::NonLinearProgram::get_rhs_space() {
    int PGXfreeloc = 0;
    int AGXfreeloc = 0;
    int FXEfreeloc = 0;
    int FXIfreeloc = 0;

    for (int i = 0; i < this->num_partitions_; i++) {
        for (auto &obj : this->part_obj_[i]) {
            obj.get_gradient_space(this->pgx_coeff_rows(), PGXfreeloc);
        }
        for (auto &eq : this->part_eq_[i]) {
            eq.get_gradient_space(this->agx_coeff_rows(), AGXfreeloc);
            eq.get_constraint_space(this->econ_coeff_rows(), FXEfreeloc);
        }
        for (auto &ineq : this->part_iq_[i]) {
            ineq.get_gradient_space(this->agx_coeff_rows(), AGXfreeloc);
            ineq.get_constraint_space(this->icon_coeff_rows(), FXIfreeloc);
        }
    }
}

void tycho::solvers::NonLinearProgram::set_mat_dimensions() {
    // Sized over the space the solver actually factorizes: the full variable
    // space until a configuration eliminates bound-fixed variables from it.
    this->kkt_dim_ = this->reduced_primal_vars_count_ + this->slack_vars_ + this->equal_cons_ +
                     this->inequal_cons_;

    ////////////////// This is the storage order of Solver data/////////////////
    ////////////////////////////////////////////////////////////////////////////
    this->num_solver_kkt_elems_ = this->slack_vars_                  // solver ijac slack ones
                                  + this->reduced_primal_vars_count_ // solver primal hess diags
                                  + this->slack_vars_                // solver slack hessian diags
                                  + this->equal_cons_                // solver equal pivots
                                  + this->inequal_cons_;             // solver inequal pivots
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    this->slack_jac_data_start_ = 0;
    this->primal_diags_data_start_ = this->slack_jac_data_start_ + this->slack_vars_;
    this->slack_diag_data_start_ =
        this->primal_diags_data_start_ + this->reduced_primal_vars_count_;
    this->e_pivot_data_start_ = this->slack_diag_data_start_ + this->slack_vars_;
    this->i_pivot_data_start_ = this->e_pivot_data_start_ + this->equal_cons_;

    this->solver_coeffs_ = Eigen::VectorXd::Zero(this->num_solver_kkt_elems_);
    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////

    this->num_kkt_elems_ = this->num_user_kkt_elems_ + this->num_solver_kkt_elems_;

    this->kkt_coeff_rows_ = Eigen::VectorXi::Constant(this->num_kkt_elems_, -1);
    this->kkt_coeff_cols_ = Eigen::VectorXi::Constant(this->num_kkt_elems_, -1);
    this->kkt_coeff_part_ids_ = Eigen::VectorXi::Constant(this->num_kkt_elems_, 0);
    this->kkt_locations_ = Eigen::VectorXi::Constant(this->num_kkt_elems_, -1);

    this->solver_coeffs_ = Eigen::VectorXd::Constant(this->num_solver_kkt_elems_, 0);
    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
}

void tycho::solvers::NonLinearProgram::set_rhs_dimensions() {
    this->num_rhs_elems_ =
        this->num_pgx_elems_ + this->num_agx_elems_ + this->num_econ_elems_ + this->num_icon_elems_;

    this->pgx_data_start_ = 0;
    this->agx_data_start_ = this->num_pgx_elems_;
    this->econ_data_start_ = this->agx_data_start_ + this->num_agx_elems_;
    this->icon_data_start_ = this->econ_data_start_ + this->num_econ_elems_;

    this->rhs_coeffs_ = Eigen::VectorXd::Zero(this->num_rhs_elems_);
    this->rhs_coeff_rows_ = Eigen::VectorXi::Constant(this->num_rhs_elems_, -1);
}

void tycho::solvers::NonLinearProgram::finalize_data() {
    // Solver-owned elements sit at fixed offsets in the space being factorized,
    // so every offset here counts from the solver's primal width.
    const int pv = this->reduced_primal_vars_count_;

    for (int i = 0; i < pv; i++) {
        this->primal_diag_coeff_cols()[i] = i;
        this->primal_diag_coeff_rows()[i] = i;
    }

    for (int i = 0; i < this->equal_cons_; i++) {
        this->e_pivot_coeff_cols()[i] = pv + this->slack_vars_ + i;
        this->e_pivot_coeff_rows()[i] = pv + this->slack_vars_ + i;
    }

    for (int i = 0; i < this->inequal_cons_; i++) {
        this->slack_coeff_cols()[i] = pv + i;
        this->slack_coeff_rows()[i] = pv + this->slack_vars_ + this->equal_cons_ + i;

        this->slack_diag_coeff_cols()[i] = pv + i;
        this->slack_diag_coeff_rows()[i] = pv + i;

        this->i_pivot_coeff_cols()[i] = pv + this->slack_vars_ + this->equal_cons_ + i;
        this->i_pivot_coeff_rows()[i] = pv + this->slack_vars_ + this->equal_cons_ + i;
    }
}

void tycho::solvers::NonLinearProgram::analyze_sparsity(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    /*
    Calculates Sparsity Pattern of NLP. PSIOPT requires that only the upper triangular part of a CSR
    matrix be filled. get_mat_space calculates the non-zeros of the lower triangular part. Therefore
    in this routine we transpose the the row-column indices when making the triplet vector that
    Eigen uses to calculate the compressed sparsity pattern of the upper triangular CSR matrix. Once
    this routine clculates the sparsity pattern of the KKT matrix it back calculates where every
    element specified by kkt_coeff_rows_[i],kkt_coeff_cols_[i], should be summed into the KKT
    matrix. This info is stored in kkt_locations_, and is passed back to all functions so that they
    know where to scatter their outputs.

    */
    KKTmat.resize(this->kkt_dim_, this->kkt_dim_);
    std::vector<Eigen::Triplet<double>> kktvec(this->num_kkt_elems_,
                                               Eigen::Triplet<double>(0, 0, 0.0));

    auto TripFillOP = [&](int start, int stop) {
        for (int i = start; i < stop; i++) {
            int row = this->kkt_coeff_rows_[i];
            int col = this->kkt_coeff_cols_[i];
            if (row < 0 || col < 0) {
                // Element of an eliminated variable. It keeps its claim so the
                // scatter's cursor still walks the claims in lockstep, but it
                // names no matrix entry: this placeholder adds 0.0 to a slot
                // that always exists (index 0's own diagonal, which finalize_data
                // lays down for every solver primal column -- and there is always
                // at least one, since configure_variable_treatment refuses a
                // configuration that would eliminate every variable) and its
                // location is left at -1, which
                // the scatter reads as "step over". This is why the eliminated
                // variable's row and column are simply absent from the pattern.
                kktvec[i] = Eigen::Triplet<double>(0, 0, 0.0);
                continue;
            }
            if (col <= row) { //// only accept lower triangular part
                kktvec[i] = Eigen::Triplet<double>(col, row, 1.0);
            } else {
                this->kkt_coeff_rows_[i] = col;
                this->kkt_coeff_cols_[i] = row;
                kktvec[i] = Eigen::Triplet<double>(row, col, 1.0);
            }
        }
    };
    tycho::utils::parallel_blocks(this->num_kkt_elems_, TripFillOP, this->num_partitions_);

    KKTmat.setFromTriplets(kktvec.begin(), kktvec.end());
    KKTmat.makeCompressed();

    /////////////////////////////////////////////////////////////
    Eigen::VectorXi innerKKTNNZ(this->kkt_dim_);

    for (int i = 0; i < this->kkt_dim_; i++) {
        innerKKTNNZ[i] = KKTmat.row(i).nonZeros();
    }

    auto FindOP = [&](int start, int stop) {
        for (int i = start; i < stop; i++) {
            int row = this->kkt_coeff_rows_(i);
            int col = this->kkt_coeff_cols_(i);
            if (row < 0 || col < 0) {
                continue; // eliminated element: its location stays -1
            }
            if (col <= row) { //// only accept lower triangular part
                for (int k = 0; k < innerKKTNNZ[col]; k++) {
                    int trow = KKTmat.innerIndexPtr()[KKTmat.outerIndexPtr()[col] + k];
                    if (trow == row) {
                        this->kkt_locations_[i] = KKTmat.outerIndexPtr()[col] + k;
                        break;
                    }
                }
            }
        }
    };

    tycho::utils::parallel_blocks(this->num_kkt_elems_, FindOP, this->num_partitions_);
    /////////////////////////////////////////////////////////////
}

bool tycho::solvers::NonLinearProgram::configure_variable_treatment(
    FixedVariableTreatments treatment, double bound_relax_factor) {

    // Written as a switch with no default label, like
    // fixed_variable_treatment_name, so that adding a treatment to the enum makes
    // the compiler point here rather than letting a new value fall through as
    // "unrecognized".
    bool treatment_known = false;
    switch (treatment) {
    case FixedVariableTreatments::MakeParameter:
    case FixedVariableTreatments::MakeConstraint:
    case FixedVariableTreatments::RelaxBounds:
        treatment_known = true;
        break;
    }
    if (!treatment_known) {
        throw std::invalid_argument(fmt::format(
            "configure_variable_treatment: unrecognized fixed-variable treatment ({0}); "
            "expected one of '{1}', '{2}', '{3}'",
            static_cast<int>(treatment),
            fixed_variable_treatment_name(FixedVariableTreatments::MakeParameter),
            fixed_variable_treatment_name(FixedVariableTreatments::MakeConstraint),
            fixed_variable_treatment_name(FixedVariableTreatments::RelaxBounds)));
    }
    if (!std::isfinite(bound_relax_factor) || bound_relax_factor < 0.0) {
        throw std::invalid_argument(
            fmt::format("configure_variable_treatment: bound_relax_factor must be finite and "
                        ">= 0 (got {0})",
                        bound_relax_factor));
    }

    // Idempotence: same treatment, same relax factor, same bound state -> the
    // structures already on hand are the answer, and nothing was rebuilt.
    if (this->fixed_treatment_valid_ && this->variable_treatment_ == treatment &&
        this->bound_relax_factor_ == bound_relax_factor &&
        this->configured_bounds_revision_ == this->bounds_revision_) {
        return false;
    }

    this->variable_treatment_ = treatment;
    this->bound_relax_factor_ = bound_relax_factor;

    const int num_vars = this->primal_vars_;
    const bool was_reduced = this->fixed_reduction_active_;
    constexpr double kInf = std::numeric_limits<double>::infinity();

    // What this treatment does with a variable whose bounds are equal, decided
    // once here and read wherever the three paths differ: eliminate it from the
    // solver's variable space, hand it to an internal equality row, or keep it as
    // a two-sided bounded variable with its bounds pushed apart.
    const bool eliminate_fixed = (treatment == FixedVariableTreatments::MakeParameter);
    const bool constrain_fixed = (treatment == FixedVariableTreatments::MakeConstraint);
    const bool relax_fixed = (treatment == FixedVariableTreatments::RelaxBounds);

    // Set immediately before either of the two statements below that re-lay the
    // KKT structures, and read by the restore alongside was_reduced.
    // rebuild_structures() leaves kkt_locations_ reset to -1 (only
    // analyze_sparsity ever fills it), and the restore rethrows rather than
    // reporting a rebuild -- so a restore that calls it unconditionally hands
    // the caller an NLP whose scatter targets are all -1, with no signal that
    // the pattern has to be re-analyzed. A classification-stage rejection
    // arriving at an NLP that was NOT reduced touches no layout, so for that
    // case the rest of the restore suffices and the analyzed pattern survives.
    bool layout_touched = false;

    // Set when this call has changed the SET of functions the problem is made of
    // -- the MakeConstraint treatment installing or dropping its internal fixing
    // rows, and nothing else. The element counts and the work partitioning are
    // derived from that set, so re-laying the layout alone would not describe the
    // problem the NLP now holds; every exit that sees this flag set, the restore
    // included, re-derives all three.
    bool functions_touched = false;

    // Everything from here to the successful exits is inside the restore. The
    // first statement below already rewrites derived state, so any throw past
    // this point -- classification rejecting a bound, the all-fixed rejection,
    // a failed map install -- would otherwise leave the maps describing one
    // problem while is_reduced() and the reduced width still describe another.
    // On an NLP that was already reduced that combination reads off the end of
    // the maps, silently, since Eigen's bounds checks are compiled out.
    try {
        this->variable_bound_set_.clear();
        this->full_to_reduced_.resize(0);
        this->reduced_to_full_.resize(0);
        this->fixed_idx_.resize(0);
        this->fixed_vals_.resize(0);

        // The internal fixing rows a previous configuration installed belong to
        // the classification that produced them, so they go before this one
        // starts. Dropping them up front -- rather than at the point the new
        // treatment would install its own -- is what makes a treatment SWITCH
        // clean: every path out of here, the restore included, then starts from
        // the user's own row space and cannot leave a stale row behind.
        if (this->internal_fixed_cons_ > 0) {
            this->discard_fixed_variable_rows();
            functions_touched = true;
        }

        const bool bounds_materialized = (num_vars > 0 && this->x_lower_.size() == num_vars &&
                                          this->x_upper_.size() == num_vars);

        // --- Classification -------------------------------------------------
        int num_fixed = 0;
        if (bounds_materialized) {
            for (int i = 0; i < num_vars; i++) {
                if (this->x_lower_[i] == this->x_upper_[i]) {
                    if (!std::isfinite(this->x_lower_[i])) {
                        throw std::invalid_argument(fmt::format(
                            "configure_variable_treatment: variable {0} has equal but non-finite "
                            "bounds ({1}); a fixed variable needs a finite value",
                            i, this->x_lower_[i]));
                    }
                    num_fixed++;
                }
            }
        }

        // Under RelaxBounds the widening is the ONLY thing separating a declared
        // [c, c] pair, so a zero factor leaves a "two-sided" variable whose
        // interval has zero width: the interior push has no interior to land in
        // and the barrier takes the log of zero. Caught here rather than as a
        // non-finite objective a few evaluations later. The reference does not
        // guard this case; rejecting it is a deliberate improvement.
        if (relax_fixed && num_fixed > 0 && !(bound_relax_factor > 0.0)) {
            throw std::invalid_argument(fmt::format(
                "configure_variable_treatment: the '{0}' treatment relies on the "
                "bound_relax_factor * max(1, |bound|) widening to separate the bounds of the "
                "{1} variable(s) whose bounds are equal, and a factor of {2} separates them "
                "not at all. Pass a positive factor, or use the '{3}' treatment.",
                fixed_variable_treatment_name(FixedVariableTreatments::RelaxBounds), num_fixed,
                bound_relax_factor,
                fixed_variable_treatment_name(FixedVariableTreatments::MakeParameter)));
        }

        // The fixed variables themselves are recorded whatever the treatment:
        // MakeConstraint builds its rows out of them, and under the other two they
        // are what fixed_variable_indices()/_values() report. The index MAPS are
        // narrower than that -- they exist only where something is actually
        // eliminated, and on every other path they stay empty and every consumer
        // treats the mapping as the identity.
        if (num_fixed > 0) {
            this->fixed_idx_.resize(num_fixed);
            this->fixed_vals_.resize(num_fixed);
        }
        if (eliminate_fixed && num_fixed > 0) {
            this->full_to_reduced_ = Eigen::VectorXi::Constant(num_vars, -1);
            this->reduced_to_full_.resize(num_vars - num_fixed);
        }

        std::vector<int> lower_idx;
        std::vector<double> lower_val;
        std::vector<double> lower_damp;
        std::vector<int> upper_idx;
        std::vector<double> upper_val;
        std::vector<double> upper_damp;

        int next_reduced = 0;
        int next_fixed = 0;
        for (int i = 0; i < (bounds_materialized ? num_vars : 0); i++) {
            const double lower = this->x_lower_[i];
            const double upper = this->x_upper_[i];
            const bool is_fixed = (lower == upper);

            if (is_fixed) {
                this->fixed_idx_[next_fixed] = i;
                this->fixed_vals_[next_fixed] = lower;
                next_fixed++;

                if (eliminate_fixed) {
                    // Gone from the solver's variable space: no column, no bound,
                    // and next_reduced does not advance past it.
                    continue;
                }
                // RelaxBounds needs nothing of its own here, and that IS the
                // treatment: the variable falls through to the shared two-sided
                // path below, which moves each finite bound out by the relax factor
                // times max(1, |bound|) -- ONE such move per side, by exactly the
                // rule every other declared bound gets, so a pair declared at
                // [c, c] reaches the solver 2 * factor * max(1, |c|) wide. The
                // reference applies the widening once as well: its adapter enters a
                // relaxed variable in both bound maps at [c, c] and does not count
                // it among the fixed ones, so the adapter's own separation loop
                // (which iterates the fixed count) never runs on this path and the
                // widening comes solely from the universal relax factor.
                //
                // MakeConstraint records nothing here either, for a different
                // reason: the internal equality row installed below is what holds
                // the variable, and the variable itself reaches the solver free,
                // keeping its column and carrying no bound for the barrier.
            }

            if (eliminate_fixed && num_fixed > 0) {
                this->full_to_reduced_[i] = next_reduced;
                this->reduced_to_full_[next_reduced] = i;
            }

            // Recorded in the space the solver iterates in, and widened by the relax
            // factor so consumers never have to re-apply it. The damping
            // indicator is recorded alongside because this loop is the only place
            // that still sees both endpoints of a variable at once -- once the
            // two lists are separate, "is this bound the only one on its
            // variable" is no longer a local question.
            const bool record_bounds = !is_fixed || relax_fixed;
            if (record_bounds && lower > -kInf) {
                lower_idx.push_back(next_reduced);
                lower_val.push_back(lower - bound_relax_factor * std::max(1.0, std::abs(lower)));
                lower_damp.push_back(upper < kInf ? 0.0 : 1.0);
            }
            if (record_bounds && upper < kInf) {
                upper_idx.push_back(next_reduced);
                upper_val.push_back(upper + bound_relax_factor * std::max(1.0, std::abs(upper)));
                upper_damp.push_back(lower > -kInf ? 0.0 : 1.0);
            }
            next_reduced++;
        }

        auto fill_bound_list = [](const std::vector<int> &idx, const std::vector<double> &val,
                                  const std::vector<double> &damp, Eigen::VectorXi &idx_out,
                                  Eigen::VectorXd &val_out, Eigen::VectorXd &damp_out) {
            const int count = static_cast<int>(idx.size());
            idx_out.resize(count);
            val_out.resize(count);
            damp_out.resize(count);
            for (int i = 0; i < count; i++) {
                idx_out[i] = idx[i];
                val_out[i] = val[i];
                damp_out[i] = damp[i];
            }
        };
        fill_bound_list(lower_idx, lower_val, lower_damp, this->variable_bound_set_.lower_idx_,
                        this->variable_bound_set_.lower_val_,
                        this->variable_bound_set_.lower_damp_);
        fill_bound_list(upper_idx, upper_val, upper_damp, this->variable_bound_set_.upper_idx_,
                        this->variable_bound_set_.upper_val_,
                        this->variable_bound_set_.upper_damp_);

        // --- Treatment ------------------------------------------------------
        //
        // Commit-on-success: fixed_treatment_valid_ and configured_bounds_revision_
        // are stamped ONLY at the three successful exits below. Everything from here
        // on can throw, and a configuration that threw must not be remembered as
        // done -- otherwise the next call would take the idempotence shortcut and
        // the solve would proceed on the unreduced problem with every fixing bound
        // silently ignored. Leaving the flag clear makes the next call re-attempt,
        // and fail the same way, until the bounds are corrected.
        if (eliminate_fixed && num_fixed > 0 && num_fixed == num_vars) {
            // Nothing would be left to solve for, and a zero-width primal block is
            // not a system this solver can lay out. Rejected here rather than as a
            // degenerate factorization later. The classification above has already
            // rewritten the maps by this point, which is why this throw is inside
            // the restore.
            //
            // Only this treatment can reach it: the other two keep every variable,
            // so an all-fixed problem is a square system of internal rows under
            // MakeConstraint and a fully boxed one under RelaxBounds.
            throw std::invalid_argument(fmt::format(
                "configure_variable_treatment: all {0} primal variables are fixed by "
                "their bounds, leaving no variable to solve for. The '{1}' and '{2}' "
                "treatments keep every variable and can solve this problem.",
                num_vars, fixed_variable_treatment_name(FixedVariableTreatments::MakeConstraint),
                fixed_variable_treatment_name(FixedVariableTreatments::RelaxBounds)));
        }

        if (constrain_fixed && num_fixed > 0) {
            // One internal equality row per fixed variable, appended to the master
            // list before the counts and the partitioning below are re-derived --
            // so the rows are sized, partitioned and laid out exactly as a user's
            // own constraint would have been.
            this->install_fixed_variable_rows();
            functions_touched = true;
        }

        const bool reduce_now = (eliminate_fixed && num_fixed > 0);
        this->reduced_primal_vars_count_ = reduce_now ? num_vars - num_fixed : num_vars;
        this->fixed_reduction_active_ = reduce_now;

        if (!reduce_now) {
            if (!was_reduced && !functions_touched) {
                // Identity path, and it already was: nothing to rebuild. Both
                // treatments that keep every variable land here whenever they
                // changed no row either -- RelaxBounds always, MakeConstraint when
                // nothing is fixed.
                this->fixed_treatment_valid_ = true;
                this->configured_bounds_revision_ = this->bounds_revision_;
                return false;
            }
            // Either an elimination is no longer in force (the last fixing bound
            // was dropped, or the treatment changed to one that keeps the
            // variable), or the equality row space changed under MakeConstraint.
            // Put every function back on its pristine input map and lay the
            // structures out over the full variable space.
            layout_touched = true;
            if (functions_touched) {
                this->refresh_function_partitions();
            }
            this->clear_function_output_maps();
            this->rebuild_structures();
            this->fixed_treatment_valid_ = true;
            this->configured_bounds_revision_ = this->bounds_revision_;
            return true;
        }

        layout_touched = true;
        // Before the map install, not after: re-partitioning hands out fresh copies
        // of the master functions, and those copies carry the pristine input map.
        if (functions_touched) {
            this->refresh_function_partitions();
        }
        this->install_function_output_maps();
        this->rebuild_structures();

        if (this->kkt_dim_ != this->reduced_primal_vars_count_ + this->slack_vars_ +
                                  this->equal_cons_ + this->inequal_cons_) {
            throw std::logic_error(fmt::format(
                "configure_variable_treatment: kkt_dim ({0}) != reduced primal_vars ({1}) + "
                "slack_vars ({2}) + equal_cons ({3}) + inequal_cons ({4})",
                this->kkt_dim_, this->reduced_primal_vars_count_, this->slack_vars_,
                this->equal_cons_, this->inequal_cons_));
        }
        if (this->num_kkt_elems_ != this->num_user_kkt_elems_ + this->num_solver_kkt_elems_) {
            throw std::logic_error(fmt::format(
                "configure_variable_treatment: KKT element bookkeeping is inconsistent "
                "(num_kkt_elems {0}, user {1}, solver {2})",
                this->num_kkt_elems_, this->num_user_kkt_elems_, this->num_solver_kkt_elems_));
        }
    } catch (...) {
        // Put the whole problem back on the full variable space before letting the
        // exception out, so a caller that catches it holds a coherent (if
        // unconfigured) NLP rather than one describing two problems at once. Two
        // ways it could: a partial map install leaves some functions on a reduced
        // output map and others on the pristine one; and a throw during
        // classification leaves maps sized for a reduction that is not in effect,
        // which -- on an NLP that WAS reduced -- would be read off the end by the
        // next expansion, silently, since Eigen's bounds checks are compiled out.
        // Clearing the maps and the reduction flag closes both.
        //
        // Sequenced deliberately: the allocation-free work first, the rebuild
        // last, so even if the rebuild itself fails only the KKT layout is left
        // stale and no index space is left mixed.
        //
        // The rebuild runs when this call re-laid the layout itself, OR when the
        // NLP arrived here ALREADY REDUCED. The second condition is the one a
        // narrower guard gets wrong: the two lines below declare the problem
        // full-width again, but the structures on hand were laid out for the
        // reduced width, so skipping the rebuild leaves kkt_dim_ and every
        // block offset describing the reduced problem while the width says
        // otherwise -- exactly the two-problems-at-once state this restore
        // exists to prevent. Only a classification-stage rejection at an NLP
        // that was NOT reduced leaves the layout genuinely untouched, and that
        // is the one case worth sparing an analyzed sparsity pattern for.
        //
        // The internal fixing rows get the same treatment for the same reason. A
        // rejection reaching here with rows installed -- or, just as importantly,
        // with a previous configuration's rows already dropped -- leaves the
        // equality row space describing something other than what the counts and
        // the partitioning were derived from, so both are re-derived before the
        // layout is, and the row space itself goes back to the user's own.
        // The validity stamp goes too, and it has to: commit-on-success only ever
        // SETS it, so a rejection arriving at an NLP that a previous call had
        // configured would otherwise leave it true -- while the treatment and the
        // factor above were already overwritten with the rejected call's own, and
        // the bound revision is unchanged. The next call with those same arguments
        // would then match the idempotence shortcut exactly and report "no change"
        // on a problem this restore has just put back to unconfigured, so a solve
        // would proceed with every fixing bound silently ignored. Clearing it is
        // what makes the rejection re-attempt (and fail the same way) instead.
        this->fixed_treatment_valid_ = false;
        this->configured_bounds_revision_ = -1;
        this->reduced_primal_vars_count_ = num_vars;
        this->fixed_reduction_active_ = false;
        this->full_to_reduced_.resize(0);
        this->reduced_to_full_.resize(0);
        this->fixed_idx_.resize(0);
        this->fixed_vals_.resize(0);
        if (this->internal_fixed_cons_ > 0) {
            this->discard_fixed_variable_rows();
            functions_touched = true;
        }
        if (functions_touched)
            this->refresh_function_partitions();
        this->clear_function_output_maps();
        if (was_reduced || layout_touched || functions_touched)
            this->rebuild_structures();
        throw;
    }

    this->fixed_treatment_valid_ = true;
    this->configured_bounds_revision_ = this->bounds_revision_;
    return true;
}

void tycho::solvers::NonLinearProgram::gather_reduced_x(ConstEigenRef<VectorXd> x_full,
                                                        EigenRef<VectorXd> x_reduced) const {
    if (x_full.size() != this->primal_vars_ ||
        x_reduced.size() != this->reduced_primal_vars_count_) {
        throw std::invalid_argument(fmt::format(
            "gather_reduced_x: expected a {0}-element full vector and a "
            "{1}-element reduced vector (got {2} and {3})",
            this->primal_vars_, this->reduced_primal_vars_count_, x_full.size(), x_reduced.size()));
    }
    if (!this->fixed_reduction_active_) {
        x_reduced = x_full;
        return;
    }
    for (int i = 0; i < this->reduced_primal_vars_count_; i++) {
        x_reduced[i] = x_full[this->reduced_to_full_[i]];
    }
}

void tycho::solvers::NonLinearProgram::scatter_full_x(ConstEigenRef<VectorXd> x_reduced,
                                                      EigenRef<VectorXd> x_full) const {
    if (x_full.size() != this->primal_vars_ ||
        x_reduced.size() != this->reduced_primal_vars_count_) {
        throw std::invalid_argument(fmt::format(
            "scatter_full_x: expected a {0}-element full vector and a "
            "{1}-element reduced vector (got {2} and {3})",
            this->primal_vars_, this->reduced_primal_vars_count_, x_full.size(), x_reduced.size()));
    }
    if (!this->fixed_reduction_active_) {
        x_full = x_reduced;
        return;
    }
    for (int i = 0; i < this->reduced_primal_vars_count_; i++) {
        x_full[this->reduced_to_full_[i]] = x_reduced[i];
    }
    for (int j = 0; j < this->fixed_idx_.size(); j++) {
        x_full[this->fixed_idx_[j]] = this->fixed_vals_[j];
    }
}

void tycho::solvers::NonLinearProgram::eval_rhs(double ObjScale, ConstEigenRef<VectorXd> X,
                                                ConstEigenRef<VectorXd> LE,
                                                ConstEigenRef<VectorXd> LI, double &val,
                                                EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                                EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto RHSevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, Xf, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_adjointgradient(Xf, LE, this->econ_coeffs(), this->agx_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_adjointgradient(Xf, LI, this->icon_coeffs(), this->agx_coeffs());
        this->vals_scratch_[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, RHSevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += this->vals_scratch_[i];

    this->fill_rhs(PGX, AGX, FXE, FXI);
}

void tycho::solvers::NonLinearProgram::eval_ogc(double ObjScale, ConstEigenRef<VectorXd> X,
                                                double &val, EigenRef<VectorXd> PGX,
                                                EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, Xf, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints(Xf, this->econ_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints(Xf, this->icon_coeffs());
        this->vals_scratch_[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, OGCevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += this->vals_scratch_[i];

    this->fill_pgx(PGX);
    this->fill_fxe(FXE);
    this->fill_fxi(FXI);
}

void tycho::solvers::NonLinearProgram::eval_occ(double ObjScale, ConstEigenRef<VectorXd> X,
                                                double &val, EigenRef<VectorXd> FXE,
                                                EigenRef<VectorXd> FXI) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_con_coeffs_zero();
    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective(ObjScale, Xf, localVal);
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints(Xf, this->econ_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints(Xf, this->icon_coeffs());
        this->vals_scratch_[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, OGCevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += this->vals_scratch_[i];

    this->fill_fxe(FXE);
    this->fill_fxi(FXI);
}

void tycho::solvers::NonLinearProgram::eval_obj(double ObjScale, ConstEigenRef<VectorXd> X,
                                                double &val) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    this->vals_scratch_.assign(this->num_partitions_, 0.0);

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective(ObjScale, Xf, localVal);
        this->vals_scratch_[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, OGCevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += this->vals_scratch_[i];
}

void tycho::solvers::NonLinearProgram::eval_kkt(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    this->vals_scratch_.assign(this->num_partitions_, 0.0);

    this->set_rhs_coeffs_zero();

    auto KKTevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient_hessian(ObjScale, Xf, localVal, this->pgx_coeffs(), KKTmat,
                                           this->kkt_locations_, this->kkt_clashes_,
                                           this->kkt_locks_);
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                Xf, LE, this->econ_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                Xf, LI, this->icon_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
        this->vals_scratch_[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, KKTevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += this->vals_scratch_[i];

    // NOTE: fill_solver_coeffs internally calls parallel_blocks, creating a nested
    // dispatch from the inline arm. Safe because: (1) the calling thread is the main
    // thread (not a pool worker), so the pool absorbs all tasks without deadlock, and
    // (2) fill_rhs and fill_solver_coeffs operate on disjoint data (RHS vectors vs. KKT
    // matrix entries), so concurrent execution requires no synchronization.
    tycho::utils::parallel_task(
        this->num_partitions_, [&] { this->fill_rhs(PGX, AGX, FXE, FXI); },
        [&] { this->fill_solver_coeffs(KKTmat); });
}

void tycho::solvers::NonLinearProgram::eval_kkt_no(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    // No-objective mode: ObjScale and val are unused but kept in the signature
    // for API consistency with eval_kkt/eval_aug (polymorphic dispatch via evalNLP).
    (void)ObjScale;
    (void)val;

    this->set_rhs_coeffs_zero();

    auto KKTevalOP = [&](int thrnum) {
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                Xf, LE, this->econ_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                Xf, LI, this->icon_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
    };

    tycho::utils::parallel_sequence(this->num_partitions_, KKTevalOP);

    // NOTE: nested dispatch from inline arm — see comment in eval_kkt.
    tycho::utils::parallel_task(
        this->num_partitions_, [&] { this->fill_rhs(PGX, AGX, FXE, FXI); },
        [&] { this->fill_solver_coeffs(KKTmat); });
}
void tycho::solvers::NonLinearProgram::eval_soe(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    // Constraint-only mode: ObjScale and val are unused but kept in the signature
    // for API consistency with eval_kkt/eval_aug (polymorphic dispatch via evalNLP).
    (void)ObjScale;
    (void)val;

    this->set_rhs_coeffs_zero();

    auto SOEevalOP = [&](int thrnum) {
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian(Xf, this->econ_coeffs(), KKTmat, this->kkt_locations_,
                                     this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian(Xf, this->icon_coeffs(), KKTmat, this->kkt_locations_,
                                     this->kkt_clashes_, this->kkt_locks_);
    };

    tycho::utils::parallel_sequence(this->num_partitions_, SOEevalOP);

    // NOTE: nested dispatch from inline arm — see comment in eval_kkt.
    tycho::utils::parallel_task(
        this->num_partitions_, [&] { this->fill_rhs(PGX, AGX, FXE, FXI); },
        [&] { this->fill_solver_coeffs(KKTmat); });
}
void tycho::solvers::NonLinearProgram::eval_aug(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // The functions address the problem's own variable space. primal_view hands
    // them a buffer in it: on the identity path a view of X itself, and once
    // variables are eliminated the reduced iterate expanded back into its own
    // coordinates with the pinned values in place. Nothing downstream of here
    // can tell the difference, which is why eliminated variables' contributions
    // to constraint values and to the surviving variables' derivatives need no
    // handling of their own.
    const Eigen::Ref<const VectorXd> Xf = this->primal_view(X);

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto SOEevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, Xf, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian_adjointgradient(
                Xf, LE, this->econ_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian_adjointgradient(
                Xf, LI, this->icon_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
        this->vals_scratch_[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, SOEevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += this->vals_scratch_[i];

    // NOTE: nested dispatch from inline arm — see comment in eval_kkt.
    tycho::utils::parallel_task(
        this->num_partitions_, [&] { this->fill_rhs(PGX, AGX, FXE, FXI); },
        [&] { this->fill_solver_coeffs(KKTmat); });
}

void tycho::solvers::NonLinearProgram::nlp_test(const Eigen::VectorXd &x, int n,
                                                std::shared_ptr<NonLinearProgram> nlp1,
                                                std::shared_ptr<NonLinearProgram> nlp2) {
    using std::cout;
    using std::endl;

    Eigen::SparseMatrix<double, Eigen::RowMajor> KKTmat1(nlp1->kkt_dim_, nlp1->kkt_dim_);
    Eigen::SparseMatrix<double, Eigen::RowMajor> KKTmat2(nlp1->kkt_dim_, nlp1->kkt_dim_);

    nlp1->analyze_sparsity(KKTmat1);
    nlp2->analyze_sparsity(KKTmat2);

    Eigen::VectorXd X = x;

    std::cout << X.size() << endl;

    Eigen::VectorXd FXE1(nlp1->equal_cons_);
    Eigen::VectorXd FXE2(nlp1->equal_cons_);
    FXE1.setZero();
    FXE2.setZero();

    Eigen::VectorXd LE(nlp1->equal_cons_);
    LE.setRandom();
    LE *= 100;

    Eigen::VectorXd FXI1(nlp1->inequal_cons_);
    Eigen::VectorXd FXI2(nlp1->inequal_cons_);
    FXI1.setZero();
    FXI2.setZero();

    Eigen::VectorXd LI(nlp1->inequal_cons_);
    LI.setRandom();
    LI *= 100;
    Eigen::VectorXd PGX1(nlp1->primal_vars_);
    Eigen::VectorXd AGX1(nlp1->primal_vars_);
    PGX1.setZero();
    AGX1.setZero();

    Eigen::VectorXd PGX2(nlp1->primal_vars_);
    Eigen::VectorXd AGX2(nlp1->primal_vars_);
    PGX2.setZero();
    AGX2.setZero();

    double v1 = 0;
    double v2 = 0;

    tycho::utils::Timer t1;
    tycho::utils::Timer t2;

    tycho::utils::Timer t3;
    tycho::utils::Timer t4;

    cout << nlp1->kkt_locations_.minCoeff() << endl;
    // nlp2->kkt_clashes_.setConstant(-1);

    for (int i = 0; i < n; i++) {
        std::fill_n(KKTmat1.valuePtr(), KKTmat1.nonZeros(), 0.0);
        std::fill_n(KKTmat2.valuePtr(), KKTmat2.nonZeros(), 0.0);

        t1.start();
        nlp1->eval_kkt(1.0, X, LE, LI, v1, PGX1, AGX1, FXE1, FXI1, KKTmat1);
        t1.stop();

        t2.start();
        nlp2->eval_kkt(1.0, X, LE, LI, v2, PGX2, AGX2, FXE2, FXI2, KKTmat2);
        t2.stop();

        if (i % 10 == 0) {
            double maxval = 0;
            double maxrow = 0;
            double maxcol = 0;
            Eigen::SparseMatrix<double, Eigen::RowMajor> mat = (KKTmat1 - KKTmat2).cwiseAbs();
            for (int k = 0; k < mat.outerSize(); ++k)
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat, k); it;
                     ++it) {
                    it.value();
                    if (it.value() > maxval) {
                        maxval = it.value();
                        maxrow = it.row();
                        maxcol = it.col();
                    }
                }

            int e_err_idx = 0;
            double FXErr = (FXE1 - FXE2).cwiseAbs().maxCoeff(&e_err_idx);
            int i_err_idx = 0;
            double FXIrr = (FXI1 - FXI2).cwiseAbs().maxCoeff(&i_err_idx);
            int gx_err_idx = 0;
            double GXIrr = (PGX1 - PGX2).cwiseAbs().maxCoeff(&gx_err_idx);
            int agx_err_idx = 0;
            double AGXIrr = (AGX1 - AGX2).cwiseAbs().maxCoeff(&agx_err_idx);

            std::cout << "KKTmat Diff:" << maxval << " row: " << maxrow << "  col:" << maxcol
                      << endl;
            std::cout << "FXE Diff:" << FXErr << " row: " << e_err_idx << endl;
            std::cout << "FXI Diff:" << FXIrr << " row: " << i_err_idx << endl;
            std::cout << "PGX Diff:" << GXIrr << " row: " << gx_err_idx << endl;
            std::cout << "AGX Diff:" << AGXIrr << " row: " << agx_err_idx << endl;
        }

        t3.start();
        nlp1->eval_occ(1.0, X, v1, FXE1, FXI1);
        t3.stop();

        t4.start();
        nlp2->eval_occ(1.0, X, v2, FXE2, FXI2);
        t4.stop();

        FXE1.setZero();
        FXI1.setZero();
        PGX1.setZero();
        AGX1.setZero();

        FXE2.setZero();
        FXI2.setZero();
        PGX2.setZero();
        AGX2.setZero();
        LI.setRandom();
        LI *= 100;
        LE.setRandom();
        LE *= 100;
    }

    double t1t = double(t1.count<std::chrono::microseconds>()) / 1000.0;
    double t2t = double(t2.count<std::chrono::microseconds>()) / 1000.0;
    double t3t = double(t3.count<std::chrono::microseconds>()) / 1000.0;
    double t4t = double(t4.count<std::chrono::microseconds>()) / 1000.0;

    cout << t1t / double(n) << " ms" << endl;
    cout << t2t / double(n) << " ms" << endl;

    cout << t3t / double(n) << " ms" << endl;
    cout << t4t / double(n) << " ms" << endl;
}
