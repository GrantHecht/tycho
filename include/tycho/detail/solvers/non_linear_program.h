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
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "tycho/detail/solvers/constraint_function.h"
#include "tycho/detail/solvers/objective_function.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/thread_pool.h"

namespace tycho::solvers {

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
    void set_e_pivots(double val) { this->e_pivot_coeffs().setConstant(val); }
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
