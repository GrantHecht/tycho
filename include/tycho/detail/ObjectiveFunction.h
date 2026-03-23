// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Implements the ObjectiveFunction class.
// Holds an ObjectiveInterface type erasure class and SolverIndexingData struct.
// Interfaces directly with NonLinearProgram and PSIOPT.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "tycho/detail/eigen_types.h"
#include "tycho/detail/SolverFunctionBase.h"

namespace Tycho {

struct ObjectiveFunction : SolverFunctionBase<ObjectiveInterface> {
    ObjectiveFunction() {}

    ObjectiveFunction(const ObjectiveInterface &f, const MatrixXi &vindex) {
        this->function = f;
        this->index_data = SolverIndexingData(f.IRows(), vindex);
    }
    ObjectiveFunction(const ObjectiveInterface &f, const SolverIndexingData &data) {
        this->function = f;
        this->index_data = data;
    }

    /*
    Partitions multiple calls to this function into seperate ObjectiveFunction instances that
    will be called on multiple threads.
    */
    std::vector<ObjectiveFunction> thread_split(int Thr) const {
        std::vector<SolverIndexingData> idat = this->index_data.thread_split(Thr);
        std::vector<ObjectiveFunction> split(idat.size());
        for (int i = 0; i < idat.size(); i++) {
            split[i] = ObjectiveFunction(this->function, idat[i]);
        }
        return split;
    }

    /*
    Interface for calling the underlying type erased function's .objective method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void objective(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val) const {
        this->function.objective(ObjScale, X, Val, this->index_data);
    }

    /*
    Interface for calling the underlying type erased function's .objective_gradient method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void objective_gradient(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                            EigenRef<Eigen::VectorXd> GX) const {
        this->function.objective_gradient(ObjScale, X, Val, GX, this->index_data);
    }

    /*
    Interface for calling the underlying type erased function's .objective_gradient_hessian method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void objective_gradient_hessian(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                                    EigenRef<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    EigenRef<Eigen::VectorXi> KKTLocations,
                                    EigenRef<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks) {
        this->function.objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat, KKTLocations,
                                                  KKTClashes, KKTLocks, this->index_data);
    }
};

} // namespace Tycho
