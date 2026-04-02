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

#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/utils/timer.h"

void tycho::solvers::NonLinearProgram::make_nlp(int PV, int EQ, int IQ) {
    this->primal_vars_ = PV;
    this->equal_cons_ = EQ;
    this->inequal_cons_ = IQ;
    this->slack_vars_ = IQ;

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

    Eigen::MatrixXi KKTclash(this->num_partitions_, this->kkt_dim_);
    KKTclash.setZero();
    for (int i = 0; i < this->num_user_kkt_elems_; i++) {
        int col = this->kkt_coeff_cols_[i];
        int thrid = this->kkt_coeff_part_ids_[i];
        KKTclash(thrid, col) = 1;
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
}

void tycho::solvers::NonLinearProgram::eval_rhs(double ObjScale, ConstEigenRef<VectorXd> X,
                                                ConstEigenRef<VectorXd> LE,
                                                ConstEigenRef<VectorXd> LI, double &val,
                                                EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                                EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto RHSevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints_adjointgradient(X, LE, this->econ_coeffs(), this->agx_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints_adjointgradient(X, LI, this->icon_coeffs(), this->agx_coeffs());
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, RHSevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += Vals[i];

    this->fill_rhs(PGX, AGX, FXE, FXI);
}

void tycho::solvers::NonLinearProgram::eval_ogc(double ObjScale, ConstEigenRef<VectorXd> X,
                                                double &val, EigenRef<VectorXd> PGX,
                                                EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->num_partitions_, 0.0);
    this->set_rhs_coeffs_zero();

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->pgx_coeffs());
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints(X, this->econ_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints(X, this->icon_coeffs());
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, OGCevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += Vals[i];

    this->fill_pgx(PGX);
    this->fill_fxe(FXE);
    this->fill_fxi(FXI);
}

void tycho::solvers::NonLinearProgram::eval_occ(double ObjScale, ConstEigenRef<VectorXd> X,
                                                double &val, EigenRef<VectorXd> FXE,
                                                EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->num_partitions_, 0.0);
    this->set_con_coeffs_zero();
    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective(ObjScale, X, localVal);
        for (auto &Con : this->part_eq_[thrnum])
            Con.constraints(X, this->econ_coeffs());
        for (auto &Con : this->part_iq_[thrnum])
            Con.constraints(X, this->icon_coeffs());
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, OGCevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += Vals[i];

    this->fill_fxe(FXE);
    this->fill_fxi(FXI);
}

void tycho::solvers::NonLinearProgram::eval_obj(double ObjScale, ConstEigenRef<VectorXd> X,
                                                double &val) {

    std::vector<double> Vals(this->num_partitions_, 0.0);

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->part_obj_[thrnum])
            Obj.objective(ObjScale, X, localVal);
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, OGCevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += Vals[i];
}

void tycho::solvers::NonLinearProgram::eval_kkt(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->num_partitions_, 0.0);

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
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, KKTevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += Vals[i];

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
}
void tycho::solvers::NonLinearProgram::eval_aug(
    double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
    ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
    EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->num_partitions_, 0.0);
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
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->num_partitions_, SOEevalOP);
    for (int i = 0; i < this->num_partitions_; i++)
        val += Vals[i];

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
