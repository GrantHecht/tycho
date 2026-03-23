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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
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

#include "tycho/detail/eigen_types.h"
#include "tycho/detail/thread_pool.h"
#include "tycho/detail/ConstraintFunction.h"
#include "tycho/detail/ObjectiveFunction.h"

namespace Tycho {

struct NonLinearProgram {
    using VectorXi = Eigen::VectorXi;
    using VectorXd = Eigen::VectorXd;
    using MatrixXi = Eigen::MatrixXi;

    int NumPartitions = 1;

    /// <summary>
    /// Master List of Objective functions that will be partitioned across work partitions (PartObj)
    /// </summary>
    std::vector<ObjectiveFunction> Objectives;

    /// <summary>
    /// Master List of Equality Constraint functions that will be partitioned across work partitions
    /// (PartEq)
    /// </summary>
    std::vector<ConstraintFunction> EqualityConstraints;

    /// <summary>
    /// Master List of Inequality Constraint functions that will be partitioned across work
    /// partitions (PartIq)
    /// </summary>
    std::vector<ConstraintFunction> InequalityConstraints;

    /// <summary>
    /// Vector with each element being the list of ObjectiveFunctions
    /// assigned to the corresponding partition.
    /// </summary>
    std::vector<std::vector<ObjectiveFunction>> PartObj;

    /// <summary>
    /// Vector with each element being the list of EqualityConstraints
    /// assigned to the corresponding partition.
    /// </summary>
    std::vector<std::vector<ConstraintFunction>> PartEq;

    /// <summary>
    /// Vector with each element being the list of InequalityConstraints
    /// assigned to the corresponding partition.
    /// </summary>
    std::vector<std::vector<ConstraintFunction>> PartIq;

    int PrimalVars = 0;  // Number of design variables
    int SlackVars = 0;   // Number of slack variables appended to problem. One for every inequalcon
    int EqualCons = 0;   // Number of equality constraints,
    int InequalCons = 0; // Number of inequality constraints
    int KKTdim =
        0; // Edge dimension of KKT matrix: = PrimalVars + SlackVars + EqualCons + InequalCons

    VectorXi KKTcoeffRows; // matched row indices
    VectorXi KKTcoeffCols; // matched col indices
    VectorXi KKTcoeffPartIds;
    VectorXi KKTLocations;

    int numUserKKTElems = 0;
    int numSolverKKTElems = 0;
    int numKKTElems = 0;

    VectorXd SolverCoeffs;
    int SlackJacDataStart;    //// Solver supplied slack jacobian data, usually just
                              /// a vector of ones
    int PrimalDiagsDataStart; //// Solver supplied diaganol elements for inertia
                              /// modification or least norm solving
    int SlackDiagDataStart;   //// Solver suppled diaganols for slack elements in
                              /// the hessian, used for interior point methods,
                              /// zeros for SQP
    int EPivotDataStart;      //// Solver suppled Equality pivots
    int IPivotDataStart;      //// Solver suppled Inequality pivots

    std::vector<std::mutex> KKTLocks;
    int numKKTClashes = 0;

    //// [i] = -1 if no fill clash, [i] = mutex lock index otherwise
    VectorXi KKTClashes;

    VectorXd RHScoeffs;
    VectorXi RHScoeffRows;

    int numPGXElems = 0;
    int numAGXElems = 0;
    int numIConElems = 0;
    int numEConElems = 0;
    int numRHSElems = 0;

    int PGXDataStart = 0;
    int AGXDataStart = 0;
    int EConDataStart = 0;
    int IConDataStart = 0;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    NonLinearProgram(int NumParts) { this->NumPartitions = std::max(NumParts, 1); }
    NonLinearProgram(int PV, int EQ, int IQ, std::vector<ObjectiveFunction> &obj,
                     std::vector<ConstraintFunction> &eq, std::vector<ConstraintFunction> &ineq,
                     int NumParts) {
        this->NumPartitions = std::max(NumParts, 1);

        this->Objectives = obj;
        this->EqualityConstraints = eq;
        this->InequalityConstraints = ineq;
        this->make_NLP(PV, EQ, IQ);
    }

    void make_NLP(int PV, int EQ, int IQ);

    void print_data() {
        for (int i = 0; i < this->NumPartitions; i++) {
            std::cout << "Partition: " << i << std::endl << std::endl;
            std::cout << "---------------Objectives---------------" << std::endl << std::endl;

            for (auto &obj : this->PartObj[i]) {
                obj.print_data();
            }

            std::cout << "---------------Equalities---------------" << std::endl << std::endl;

            for (auto &eq : this->PartEq[i]) {
                eq.print_data();
            }
            std::cout << "--------------Inequalities--------------" << std::endl << std::endl;

            for (auto &ineq : this->PartIq[i]) {
                ineq.print_data();
            }
        }
    }

    void countElems();

    void analyzePartitioning();

    void getMATSpace();

    void getRHSSpace();

    void setMATDimensions();

    void setRHSDimensions();

    void finalizeData();

    void analyzeSparsity(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void make_compressed() {
        this->KKTcoeffPartIds.resize(0);
        this->KKTcoeffRows.resize(0);
        this->KKTcoeffCols.resize(0);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> SlackCoeffs() {
        return this->SolverCoeffs.segment(this->SlackJacDataStart, this->SlackVars);
    }
    EigenRef<VectorXd> PrimalDiagCoeffs() {
        return this->SolverCoeffs.segment(this->PrimalDiagsDataStart, this->PrimalVars);
    }
    EigenRef<VectorXd> SlackDiagCoeffs() {
        return this->SolverCoeffs.segment(this->SlackDiagDataStart, this->SlackVars);
    }
    EigenRef<VectorXd> EPivotCoeffs() {
        return this->SolverCoeffs.segment(this->EPivotDataStart, this->EqualCons);
    }
    EigenRef<VectorXd> IPivotCoeffs() {
        return this->SolverCoeffs.segment(this->IPivotDataStart, this->InequalCons);
    }

    EigenRef<VectorXi> SlackCoeffCols() {
        return this->KKTcoeffCols.segment(this->SlackJacDataStart + this->numUserKKTElems,
                                          this->SlackVars);
    }
    EigenRef<VectorXi> PrimalDiagCoeffCols() {
        return this->KKTcoeffCols.segment(this->PrimalDiagsDataStart + this->numUserKKTElems,
                                          this->PrimalVars);
    }
    EigenRef<VectorXi> SlackDiagCoeffCols() {
        return this->KKTcoeffCols.segment(this->SlackDiagDataStart + this->numUserKKTElems,
                                          this->SlackVars);
    }
    EigenRef<VectorXi> EPivotCoeffCols() {
        return this->KKTcoeffCols.segment(this->EPivotDataStart + this->numUserKKTElems,
                                          this->EqualCons);
    }
    EigenRef<VectorXi> IPivotCoeffCols() {
        return this->KKTcoeffCols.segment(this->IPivotDataStart + this->numUserKKTElems,
                                          this->InequalCons);
    }

    EigenRef<VectorXi> SlackCoeffRows() {
        return this->KKTcoeffRows.segment(this->SlackJacDataStart + this->numUserKKTElems,
                                          this->SlackVars);
    }
    EigenRef<VectorXi> PrimalDiagCoeffRows() {
        return this->KKTcoeffRows.segment(this->PrimalDiagsDataStart + this->numUserKKTElems,
                                          this->PrimalVars);
    }
    EigenRef<VectorXi> SlackDiagCoeffRows() {
        return this->KKTcoeffRows.segment(this->SlackDiagDataStart + this->numUserKKTElems,
                                          this->SlackVars);
    }
    EigenRef<VectorXi> EPivotCoeffRows() {
        return this->KKTcoeffRows.segment(this->EPivotDataStart + this->numUserKKTElems,
                                          this->EqualCons);
    }
    EigenRef<VectorXi> IPivotCoeffRows() {
        return this->KKTcoeffRows.segment(this->IPivotDataStart + this->numUserKKTElems,
                                          this->InequalCons);
    }

    void setPrimalDiags(const Eigen::VectorXd &pdiags) { this->PrimalDiagCoeffs() = pdiags; }
    void setPrimalDiags(double val) { this->PrimalDiagCoeffs().setConstant(val); }
    void setSlackDiags(const Eigen::VectorXd &sdiags) { this->SlackDiagCoeffs() = sdiags; }
    void setSlackDiags(double val) { this->SlackDiagCoeffs().setConstant(val); }
    void setEPivots(double val) { this->EPivotCoeffs().setConstant(val); }
    void setIPivots(double val) { this->IPivotCoeffs().setConstant(val); }
    void setSlacksOnes() { this->SlackCoeffs().setConstant(1.0); }

    void fillSolverCoeffs(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
        auto FillOp = [&](int start, int stop) {
            for (int i = start; i < stop; i++) {
                mat.valuePtr()[this->KKTLocations.tail(this->numSolverKKTElems)[i]] +=
                    this->SolverCoeffs[i];
            }
        };

        Tycho::parallel_blocks(this->numSolverKKTElems, FillOp, this->NumPartitions);
    }

    void assignKKTSlackHessian(const Eigen::Ref<const Eigen::VectorXd> &slhs,
                               Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
        int ofs = this->SlackDiagDataStart + this->numUserKKTElems;
        for (int i = 0; i < this->SlackVars; i++) {
            mat.valuePtr()[this->KKTLocations[ofs + i]] = slhs[i];
        }
    }
    void perturbKKTPDiags(double pert, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
        int ofs = this->PrimalDiagsDataStart + this->numUserKKTElems;
        for (int i = 0; i < this->PrimalVars; i++) {
            mat.valuePtr()[this->KKTLocations[ofs + i]] += pert;
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> PGXCoeffs() {
        return this->RHScoeffs.segment(this->PGXDataStart, this->numPGXElems);
    }
    EigenRef<VectorXd> AGXCoeffs() {
        return this->RHScoeffs.segment(this->AGXDataStart, this->numAGXElems);
    }
    EigenRef<VectorXd> EConCoeffs() {
        return this->RHScoeffs.segment(this->EConDataStart, this->numEConElems);
    }
    EigenRef<VectorXd> IConCoeffs() {
        return this->RHScoeffs.segment(this->IConDataStart, this->numIConElems);
    }

    EigenRef<VectorXi> PGXCoeffRows() {
        return this->RHScoeffRows.segment(this->PGXDataStart, this->numPGXElems);
    }
    EigenRef<VectorXi> AGXCoeffRows() {
        return this->RHScoeffRows.segment(this->AGXDataStart, this->numAGXElems);
    }
    EigenRef<VectorXi> EConCoeffRows() {
        return this->RHScoeffRows.segment(this->EConDataStart, this->numEConElems);
    }
    EigenRef<VectorXi> IConCoeffRows() {
        return this->RHScoeffRows.segment(this->IConDataStart, this->numIConElems);
    }

    EigenRef<VectorXi> getKKTLocations() { return this->KKTLocations.head(this->numUserKKTElems); }
    EigenRef<VectorXi> getKKTClashes() { return this->KKTClashes.head(this->PrimalVars); }

    void setConCoeffsZero() {
        this->EConCoeffs().setZero();
        this->IConCoeffs().setZero();
    }
    void setPGXCoeffsZero() { this->PGXCoeffs().setZero(); }
    void setAGXCoeffsZero() { this->AGXCoeffs().setZero(); }
    void setRHSCoeffsZero() {
        this->setPGXCoeffsZero();
        this->setAGXCoeffsZero();
        this->setConCoeffsZero();
    }

    void fillPGX(EigenRef<VectorXd> PGX) {
        this->RHSFillOP(PGX, this->PGXCoeffs(), this->PGXCoeffRows());
    }
    void fillAGX(EigenRef<VectorXd> AGX) {
        this->RHSFillOP(AGX, this->AGXCoeffs(), this->AGXCoeffRows());
    }
    void fillFXE(EigenRef<VectorXd> FXE) {
        this->RHSFillOP(FXE, this->EConCoeffs(), this->EConCoeffRows());
    }
    void fillFXI(EigenRef<VectorXd> FXI) {
        this->RHSFillOP(FXI, this->IConCoeffs(), this->IConCoeffRows());
    }
    void fillRHS(EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE,
                 EigenRef<VectorXd> FXI) {
        this->fillPGX(PGX);
        this->fillAGX(AGX);
        this->fillFXE(FXE);
        this->fillFXI(FXI);
    }

    static void RHSFillOP(EigenRef<VectorXd> target, EigenRef<VectorXd> source,
                          EigenRef<VectorXi> sourcelocs) {
        for (int i = 0; i < source.size(); i++) {
            target[sourcelocs[i]] += source[i];
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void evalRHS(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                 ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                 EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI);

    void evalOGC(double ObjScale, ConstEigenRef<VectorXd> X, double &val, EigenRef<VectorXd> PGX,
                 EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI);

    void evalOCC(double ObjScale, ConstEigenRef<VectorXd> X, double &val, EigenRef<VectorXd> FXE,
                 EigenRef<VectorXd> FXI);

    void evalOBJ(double ObjScale, ConstEigenRef<VectorXd> X, double &val);

    void evalKKT(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                 ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                 EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                 Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    void evalKKTNO(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                   ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                   EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                   Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    void evalSOE(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                 ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                 EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                 Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    void evalAUG(double ObjScale, ConstEigenRef<VectorXd> X, ConstEigenRef<VectorXd> LE,
                 ConstEigenRef<VectorXd> LI, double &val, EigenRef<VectorXd> PGX,
                 EigenRef<VectorXd> AGX, EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                 Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    static void NLPTest(const Eigen::VectorXd &x, int n, std::shared_ptr<NonLinearProgram> nlp1,
                        std::shared_ptr<NonLinearProgram> nlp2);
};

} // namespace Tycho
