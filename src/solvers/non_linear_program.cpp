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
    this->primal_vars_ = PV;
    this->equal_cons_ = EQ;
    this->inequal_cons_ = IQ;
    this->slack_vars_ = IQ;

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
    this->set_mat_dimensions();
    this->set_rhs_dimensions();

    this->get_mat_space();
    this->get_rhs_space();
    this->finalize_data();
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

    int eqoffset = this->primal_vars_ + this->slack_vars_;
    int iqoffset = this->primal_vars_ + this->slack_vars_ + this->equal_cons_;
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
    this->kkt_dim_ =
        this->primal_vars_ + this->slack_vars_ + this->equal_cons_ + this->inequal_cons_;

    ////////////////// This is the storage order of Solver data/////////////////
    ////////////////////////////////////////////////////////////////////////////
    this->num_solver_kkt_elems_ = this->slack_vars_      // solver ijac slack ones
                                  + this->primal_vars_   // solver primal hessian diags
                                  + this->slack_vars_    // solver slack hessian diags
                                  + this->equal_cons_    // solver equal pivots
                                  + this->inequal_cons_; // solver inequal pivots
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    this->slack_jac_data_start_ = 0;
    this->primal_diags_data_start_ = this->slack_jac_data_start_ + this->slack_vars_;
    this->slack_diag_data_start_ = this->primal_diags_data_start_ + this->primal_vars_;
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

    // kkt_locations_ has just been discarded, so there is no analyzed pattern
    // any more and no fixed-variable pin list can still be addressing it.
    this->sparsity_analyzed_ = false;
    this->invalidate_variable_treatment();

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
    for (int i = 0; i < this->primal_vars_; i++) {
        this->primal_diag_coeff_cols()[i] = i;
        this->primal_diag_coeff_rows()[i] = i;
    }

    for (int i = 0; i < this->equal_cons_; i++) {
        this->e_pivot_coeff_cols()[i] = this->primal_vars_ + this->slack_vars_ + i;
        this->e_pivot_coeff_rows()[i] = this->primal_vars_ + this->slack_vars_ + i;
    }

    for (int i = 0; i < this->inequal_cons_; i++) {
        this->slack_coeff_cols()[i] = this->primal_vars_ + i;
        this->slack_coeff_rows()[i] =
            this->primal_vars_ + this->slack_vars_ + this->equal_cons_ + i;

        this->slack_diag_coeff_cols()[i] = this->primal_vars_ + i;
        this->slack_diag_coeff_rows()[i] = this->primal_vars_ + i;

        this->i_pivot_coeff_cols()[i] =
            this->primal_vars_ + this->slack_vars_ + this->equal_cons_ + i;
        this->i_pivot_coeff_rows()[i] =
            this->primal_vars_ + this->slack_vars_ + this->equal_cons_ + i;
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

    // kkt_locations_ has just been recomputed, so any fixed-variable pin lists
    // built against the previous pattern now address the wrong slots. Force the
    // next configure_variable_treatment call to rebuild them.
    this->sparsity_analyzed_ = true;
    this->invalidate_variable_treatment();
}

void tycho::solvers::NonLinearProgram::configure_variable_treatment(
    FixedVariableTreatments treatment, double bound_relax_factor) {

    if (treatment != FixedVariableTreatments::MakeParameter) {
        throw std::invalid_argument(fmt::format(
            "configure_variable_treatment: fixed-variable treatment '{0}' is not available "
            "yet. Only '{1}' (eliminate every variable whose lower and upper bounds are "
            "equal) is implemented.",
            fixed_variable_treatment_name(treatment),
            fixed_variable_treatment_name(FixedVariableTreatments::MakeParameter)));
    }
    if (!std::isfinite(bound_relax_factor) || bound_relax_factor < 0.0) {
        throw std::invalid_argument(
            fmt::format("configure_variable_treatment: bound_relax_factor must be finite and "
                        ">= 0 (got {0})",
                        bound_relax_factor));
    }

    // Idempotence: same treatment, same relax factor, same bound state, same
    // sparsity pattern -> the state already on hand is the answer.
    if (this->fixed_treatment_valid_ && this->variable_treatment_ == treatment &&
        this->bound_relax_factor_ == bound_relax_factor &&
        this->configured_bounds_revision_ == this->bounds_revision_) {
        return;
    }

    this->variable_treatment_ = treatment;
    this->bound_relax_factor_ = bound_relax_factor;

    const int num_vars = this->primal_vars_;
    constexpr double kInf = std::numeric_limits<double>::infinity();

    this->variable_bound_set_.clear();
    this->full_to_reduced_.resize(0);
    this->reduced_to_full_.resize(0);
    this->fixed_idx_.resize(0);
    this->fixed_vals_.resize(0);
    this->fixed_kkt_slots_.resize(0);
    this->fixed_diag_slots_.resize(0);
    this->fixed_reduction_active_ = false;
    this->reduced_primal_vars_count_ = num_vars;

    const bool bounds_materialized =
        (num_vars > 0 && this->x_lower_.size() == num_vars && this->x_upper_.size() == num_vars);

    if (!bounds_materialized) {
        this->fixed_treatment_valid_ = true;
        this->configured_bounds_revision_ = this->bounds_revision_;
        return;
    }

    // --- Classification -------------------------------------------------
    int num_fixed = 0;
    for (int i = 0; i < num_vars; i++) {
        if (this->x_lower_[i] == this->x_upper_[i]) {
            if (!std::isfinite(this->x_lower_[i])) {
                throw std::invalid_argument(
                    fmt::format("configure_variable_treatment: variable {0} has equal but "
                                "non-finite bounds ({1}); a fixed variable needs a finite value",
                                i, this->x_lower_[i]));
            }
            num_fixed++;
        }
    }

    std::vector<int> lower_idx;
    std::vector<double> lower_val;
    std::vector<int> upper_idx;
    std::vector<double> upper_val;

    if (num_fixed > 0) {
        this->fixed_idx_.resize(num_fixed);
        this->fixed_vals_.resize(num_fixed);
        this->full_to_reduced_ = Eigen::VectorXi::Constant(num_vars, -1);
        this->reduced_to_full_.resize(num_vars - num_fixed);
    }

    int next_reduced = 0;
    int next_fixed = 0;
    for (int i = 0; i < num_vars; i++) {
        const double lower = this->x_lower_[i];
        const double upper = this->x_upper_[i];

        if (lower == upper) {
            this->fixed_idx_[next_fixed] = i;
            this->fixed_vals_[next_fixed] = lower;
            next_fixed++;
            continue;
        }

        if (num_fixed > 0) {
            this->full_to_reduced_[i] = next_reduced;
            this->reduced_to_full_[next_reduced] = i;
        }
        next_reduced++;

        // Record the surviving finite bounds, widened by the relax factor so
        // consumers never have to re-apply it.
        if (lower > -kInf) {
            lower_idx.push_back(i);
            lower_val.push_back(lower - bound_relax_factor * std::max(1.0, std::abs(lower)));
        }
        if (upper < kInf) {
            upper_idx.push_back(i);
            upper_val.push_back(upper + bound_relax_factor * std::max(1.0, std::abs(upper)));
        }
    }

    this->reduced_primal_vars_count_ = next_reduced;

    auto fill_bound_list = [](const std::vector<int> &idx, const std::vector<double> &val,
                              Eigen::VectorXi &idx_out, Eigen::VectorXd &val_out) {
        const int count = static_cast<int>(idx.size());
        idx_out.resize(count);
        val_out.resize(count);
        for (int i = 0; i < count; i++) {
            idx_out[i] = idx[i];
            val_out[i] = val[i];
        }
    };
    fill_bound_list(lower_idx, lower_val, this->variable_bound_set_.lower_idx_,
                    this->variable_bound_set_.lower_val_);
    fill_bound_list(upper_idx, upper_val, this->variable_bound_set_.upper_idx_,
                    this->variable_bound_set_.upper_val_);

    if (num_fixed == 0) {
        // Identity path: no maps, no pin lists, no per-assembly work.
        this->fixed_treatment_valid_ = true;
        this->configured_bounds_revision_ = this->bounds_revision_;
        return;
    }

    // --- Reduction: KKT slot lists for the eliminated rows/columns -------
    if (!this->sparsity_analyzed_) {
        throw std::invalid_argument(fmt::format(
            "configure_variable_treatment: {0} variable(s) are fixed by their bounds, but the "
            "KKT sparsity pattern has not been computed yet. Call analyze_sparsity() (PSIOPT "
            "does this in set_nlp) before configuring the fixed-variable treatment.",
            num_fixed));
    }
    if (this->kkt_dim_ !=
        this->primal_vars_ + this->slack_vars_ + this->equal_cons_ + this->inequal_cons_) {
        throw std::logic_error(fmt::format(
            "configure_variable_treatment: kkt_dim ({0}) != primal_vars ({1}) + slack_vars "
            "({2}) + equal_cons ({3}) + inequal_cons ({4})",
            this->kkt_dim_, this->primal_vars_, this->slack_vars_, this->equal_cons_,
            this->inequal_cons_));
    }
    if (this->num_kkt_elems_ != this->num_user_kkt_elems_ + this->num_solver_kkt_elems_ ||
        this->kkt_locations_.size() != this->num_kkt_elems_) {
        throw std::logic_error(
            fmt::format("configure_variable_treatment: KKT element bookkeeping is inconsistent "
                        "(num_kkt_elems {0}, user {1}, solver {2}, locations {3})",
                        this->num_kkt_elems_, this->num_user_kkt_elems_,
                        this->num_solver_kkt_elems_, this->kkt_locations_.size()));
    }

    std::vector<char> is_fixed(num_vars, 0);
    for (int j = 0; j < num_fixed; j++) {
        is_fixed[this->fixed_idx_[j]] = 1;
    }

    // Every user element whose row OR column is an eliminated variable. The
    // stored pattern is one triangle of a symmetric matrix, so testing both
    // endpoints covers the eliminated variable's row and its column.
    std::vector<int> slots;
    for (int e = 0; e < this->num_user_kkt_elems_; e++) {
        const int row = this->kkt_coeff_rows_[e];
        const int col = this->kkt_coeff_cols_[e];
        const bool touches_fixed = (row >= 0 && row < num_vars && is_fixed[row]) ||
                                   (col >= 0 && col < num_vars && is_fixed[col]);
        if (!touches_fixed) {
            continue;
        }
        const int loc = this->kkt_locations_[e];
        if (loc < 0) {
            throw std::logic_error(
                fmt::format("configure_variable_treatment: KKT element {0} (row {1}, col {2}) "
                            "has no assigned storage location",
                            e, row, col));
        }
        slots.push_back(loc);
    }
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());

    this->fixed_kkt_slots_.resize(static_cast<int>(slots.size()));
    for (int i = 0; i < static_cast<int>(slots.size()); i++) {
        this->fixed_kkt_slots_[i] = slots[i];
    }

    // The eliminated diagonals come from the solver-owned primal-diagonal block,
    // which finalize_data laid out at (i, i) for every primal variable.
    const int pdiag_base = this->primal_diags_data_start_ + this->num_user_kkt_elems_;
    this->fixed_diag_slots_.resize(num_fixed);
    for (int j = 0; j < num_fixed; j++) {
        const int loc = this->kkt_locations_[pdiag_base + this->fixed_idx_[j]];
        if (loc < 0) {
            throw std::logic_error(
                fmt::format("configure_variable_treatment: primal diagonal of variable {0} has "
                            "no assigned storage location",
                            this->fixed_idx_[j]));
        }
        this->fixed_diag_slots_[j] = loc;
    }

    this->fixed_reduction_active_ = true;
    this->fixed_treatment_valid_ = true;
    this->configured_bounds_revision_ = this->bounds_revision_;
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

void tycho::solvers::NonLinearProgram::pin_fixed_variables(EigenRef<VectorXd> x_full) const {
    if (!this->fixed_reduction_active_) {
        return;
    }
    if (x_full.size() != this->primal_vars_) {
        throw std::invalid_argument(
            fmt::format("pin_fixed_variables: expected a {0}-element primal vector (got {1})",
                        this->primal_vars_, x_full.size()));
    }
    for (int j = 0; j < this->fixed_idx_.size(); j++) {
        x_full[this->fixed_idx_[j]] = this->fixed_vals_[j];
    }
}

void tycho::solvers::NonLinearProgram::clear_fixed_variable_rows(
    EigenRef<VectorXd> primal_rows) const {
    if (!this->fixed_reduction_active_) {
        return;
    }
    if (primal_rows.size() != this->primal_vars_) {
        throw std::invalid_argument(fmt::format(
            "clear_fixed_variable_rows: expected a {0}-element primal-row vector (got {1})",
            this->primal_vars_, primal_rows.size()));
    }
    for (int j = 0; j < this->fixed_idx_.size(); j++) {
        primal_rows[this->fixed_idx_[j]] = 0.0;
    }
}

void tycho::solvers::NonLinearProgram::eval_rhs(double ObjScale, ConstEigenRef<VectorXd> X,
                                                ConstEigenRef<VectorXd> LE,
                                                ConstEigenRef<VectorXd> LI, double &val,
                                                EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                                EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto RHSevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_adjointgradient(X, LE, this->econ_coeffs(), this->agx_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_adjointgradient(X, LI, this->icon_coeffs(), this->agx_coeffs());
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

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints(X, this->econ_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints(X, this->icon_coeffs());
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

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_con_coeffs_zero();
    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective(ObjScale, X, localVal);
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints(X, this->econ_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints(X, this->icon_coeffs());
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

    this->vals_scratch_.assign(this->num_partitions_, 0.0);

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective(ObjScale, X, localVal);
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

    this->vals_scratch_.assign(this->num_partitions_, 0.0);

    this->set_rhs_coeffs_zero();

    auto KKTevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient_hessian(ObjScale, X, localVal, this->pgx_coeffs(), KKTmat,
                                           this->kkt_locations_, this->kkt_clashes_,
                                           this->kkt_locks_);
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LE, this->econ_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LI, this->icon_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
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

    // Fixed-variable pin, the LAST write of the assembly: whatever the user
    // functions and fill_solver_coeffs scattered into an eliminated variable's
    // row/column, it ends up decoupled with a unit diagonal. Guarded on the one
    // cached bool -- with no fixed variables this branch is not taken and the
    // assembled matrix is bit-identical to what it was before the treatment
    // existed.
    if (this->fixed_reduction_active_) {
        this->pin_fixed_variable_kkt_rows(KKTmat);
    }
}

void tycho::solvers::NonLinearProgram::eval_kkt_no(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // No-objective mode: ObjScale and val are unused but kept in the signature
    // for API consistency with eval_kkt/eval_aug (polymorphic dispatch via evalNLP).
    (void)ObjScale;
    (void)val;

    this->set_rhs_coeffs_zero();

    auto KKTevalOP = [&](int thrnum) {
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LE, this->econ_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LI, this->icon_coeffs(), this->agx_coeffs(), KKTmat, this->kkt_locations_,
                this->kkt_clashes_, this->kkt_locks_);
    };

    tycho::utils::parallel_sequence(this->num_partitions_, KKTevalOP);

    // NOTE: nested dispatch from inline arm — see comment in eval_kkt.
    tycho::utils::parallel_task(
        this->num_partitions_, [&] { this->fill_rhs(PGX, AGX, FXE, FXI); },
        [&] { this->fill_solver_coeffs(KKTmat); });

    // Fixed-variable pin — see the comment in eval_kkt.
    if (this->fixed_reduction_active_) {
        this->pin_fixed_variable_kkt_rows(KKTmat);
    }
}
void tycho::solvers::NonLinearProgram::eval_soe(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // Constraint-only mode: ObjScale and val are unused but kept in the signature
    // for API consistency with eval_kkt/eval_aug (polymorphic dispatch via evalNLP).
    (void)ObjScale;
    (void)val;

    this->set_rhs_coeffs_zero();

    auto SOEevalOP = [&](int thrnum) {
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian(X, this->econ_coeffs(), KKTmat, this->kkt_locations_,
                                     this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian(X, this->icon_coeffs(), KKTmat, this->kkt_locations_,
                                     this->kkt_clashes_, this->kkt_locks_);
    };

    tycho::utils::parallel_sequence(this->num_partitions_, SOEevalOP);

    // NOTE: nested dispatch from inline arm — see comment in eval_kkt.
    tycho::utils::parallel_task(
        this->num_partitions_, [&] { this->fill_rhs(PGX, AGX, FXE, FXI); },
        [&] { this->fill_solver_coeffs(KKTmat); });

    // Fixed-variable pin — see the comment in eval_kkt.
    if (this->fixed_reduction_active_) {
        this->pin_fixed_variable_kkt_rows(KKTmat);
    }
}
void tycho::solvers::NonLinearProgram::eval_aug(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    this->vals_scratch_.assign(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto SOEevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_jacobian_adjointgradient(X, LE, this->econ_coeffs(), this->agx_coeffs(),
                                                     KKTmat, this->kkt_locations_,
                                                     this->kkt_clashes_, this->kkt_locks_);
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_jacobian_adjointgradient(X, LI, this->icon_coeffs(), this->agx_coeffs(),
                                                     KKTmat, this->kkt_locations_,
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

    // Fixed-variable pin — see the comment in eval_kkt.
    if (this->fixed_reduction_active_) {
        this->pin_fixed_variable_kkt_rows(KKTmat);
    }
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
