// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Implements the ConstraintFunction class.
// Holds an ConstraintInterface type erasure class and SolverIndexingData struct.
// Interfaces directly with NonLinearProgram and PSIOPT.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "tycho/detail/solvers/solver_function_base.h"
#include "tycho/detail/typedefs/eigen_types.h"

namespace tycho::solvers {

struct ConstraintFunction : SolverFunctionBase<ConstraintInterface> {
    using Base = SolverFunctionBase<ConstraintInterface>;
    using Base::function_;
    using Base::index_data_;
    using MatrixXi = Eigen::MatrixXi;
    using VectorXi = Eigen::VectorXi;

    ConstraintFunction() {}

    ConstraintFunction(const ConstraintInterface &f, const MatrixXi &vindex,
                       const MatrixXi &cindex) {
        this->function_ = f;
        this->index_data_ = SolverIndexingData(f.input_rows(), f.output_rows(), vindex, cindex);
    }

    ConstraintFunction(const ConstraintInterface &f, const SolverIndexingData &data) {
        this->function_ = f;
        this->index_data_ = data;
    }

    /*
    Partitions multiple calls to this function into seperate ConstraintFunction instances that
    will be called on multiple threads.
    */
    std::vector<ConstraintFunction> thread_split(int Thr) const {
        std::vector<SolverIndexingData> idat = this->index_data_.thread_split(Thr);
        std::vector<ConstraintFunction> split(idat.size());
        for (int i = 0; i < idat.size(); i++) {
            split[i] = ConstraintFunction(this->function_, idat[i]);
        }
        return split;
    }

    /*
    Interface for calling the underlying type erased function's .constraints method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void constraints(ConstEigenRef<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> FX) const {
        this->function_.constraints(X, FX, this->index_data_);
    }

    /*
    Interface for calling the underlying type erased function's .constraints_adjointgradient method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX) const {
        this->function_.constraints_adjointgradient(X, L, FX, AGX, this->index_data_);
    }

    /*
    Interface for calling the underlying type erased function's .constraints_jacobian method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              EigenRef<Eigen::VectorXi> KKTLocations,
                              EigenRef<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks) const {
        this->function_.constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks,
                                             this->index_data_);
    }

    /*
    Interface for calling the underlying type erased function's
    .constraints_jacobian_adjointgradient method. Passes the arguments from PSIOPT and
    NonLinearProgram as well as the indexing data struct to the underlying vector function.
    */
    void constraints_jacobian_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                              ConstEigenRef<Eigen::VectorXd> L,
                                              EigenRef<Eigen::VectorXd> FX,
                                              EigenRef<Eigen::VectorXd> AGX,
                                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                              EigenRef<Eigen::VectorXi> KKTLocations,
                                              EigenRef<Eigen::VectorXi> KKTClashes,
                                              std::vector<std::mutex> &KKTLocks) const {
        this->function_.constraints_jacobian_adjointgradient(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, this->index_data_);
    }

    /*
    Interface for calling the underlying type erased function's
    .constraints_jacobian_adjointgradient_adjointhessian method. Passes the arguments from PSIOPT
    and NonLinearProgram as well as the indexing data struct to the underlying vector function.
    */
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks) const {
        this->function_.constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, this->index_data_);
    }
};

} // namespace tycho::solvers
