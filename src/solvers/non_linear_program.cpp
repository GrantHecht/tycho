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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/utils/timer.h"

void tycho::solvers::NonLinearProgram::make_NLP(int PV, int EQ, int IQ) {
    this->PrimalVars = PV;
    this->EqualCons = EQ;
    this->InequalCons = IQ;
    this->SlackVars = IQ;

    this->countElems();

    // Cap partitions so each has enough work to offset dispatch overhead.
    // numUserKKTElems counts Jacobian + Hessian NNZ across all functions —
    // proportional to per-partition compute in evalKKT/evalAUG. Below ~1000
    // NNZ per partition, dispatch overhead dominates actual work.
    // Threshold empirically chosen via solver benchmarks (bench_all);
    // re-evaluate with bench/bench_track.sh if dispatch overhead changes.
    if (this->NumPartitions > 1) {
        static constexpr int MIN_NNZ_PER_PARTITION = 1000;
        int max_parts = std::max(1, this->numUserKKTElems / MIN_NNZ_PER_PARTITION);
        this->NumPartitions = std::min(this->NumPartitions, max_parts);
    }

    this->analyzePartitioning();
    this->setMATDimensions();
    this->setRHSDimensions();

    this->getMATSpace();
    this->getRHSSpace();
    this->finalizeData();
}

void tycho::solvers::NonLinearProgram::countElems() {
    int nkkt = 0;

    int npgx = 0;
    int nagx = 0;
    int nec = 0;
    int nic = 0;

    for (auto &obj : this->Objectives) {
        nkkt += obj.numKKTEles(false, true);
        npgx += obj.numGradEles();
    }
    for (auto &eq : this->EqualityConstraints) {
        nkkt += eq.numKKTEles(true, true);
        nagx += eq.numGradEles();
        nec += eq.numConEles();
    }
    for (auto &ineq : this->InequalityConstraints) {
        nkkt += ineq.numKKTEles(true, true);
        nagx += ineq.numGradEles();
        nic += ineq.numConEles();
    }

    this->numUserKKTElems = nkkt;
    this->numPGXElems = npgx;
    this->numAGXElems = nagx;
    this->numIConElems = nic;
    this->numEConElems = nec;
}

void tycho::solvers::NonLinearProgram::analyzePartitioning() {
    /*
    This function loops over the master list of objectives and constraints and assigns them
    to NumPartitions work partitions. Each partition's work is dispatched as a single task
    to the global thread pool.
    */
    this->PartObj.clear();
    this->PartEq.clear();
    this->PartIq.clear();

    this->PartObj.resize(this->NumPartitions);
    this->PartEq.resize(this->NumPartitions);
    this->PartIq.resize(this->NumPartitions);

    int rrPart = 0;

    auto analyzeOP = [&](auto &SourceFuncs, auto &TargetPartFuncs) {
        for (auto &func : SourceFuncs) {
            if (func.getThreadMode() ==
                ThreadingFlags::MainThread) { // Force to last partition — parallel_sequence runs
                                              // the last index inline on the calling thread, so
                                              // MainThread functions stay safe.
                TargetPartFuncs.back().push_back(func);
            } else if (func.getThreadMode() == ThreadingFlags::RoundRobin) {
                TargetPartFuncs[rrPart].push_back(func);
                rrPart++;
                if (rrPart > (this->NumPartitions - 1))
                    rrPart = 0;
            } else if (static_cast<int>(func.getThreadMode()) >=
                       0) { // Specific Partition Assignment
                int part =
                    std::min(static_cast<int>(func.getThreadMode()), this->NumPartitions - 1);
                TargetPartFuncs[part].push_back(func);
            } else { // By application
                auto TempPartFuncs = func.thread_split(this->NumPartitions);
                for (int i = 0; i < TempPartFuncs.size(); i++) {
                    TargetPartFuncs[i].push_back(TempPartFuncs[i]);
                }
            }
        }
    };

    analyzeOP(this->Objectives, this->PartObj);
    analyzeOP(this->EqualityConstraints, this->PartEq);
    analyzeOP(this->InequalityConstraints, this->PartIq);
}

void tycho::solvers::NonLinearProgram::getMATSpace() {
    /*
     * Loops over all constraints and objectives on each partition and has each claim its
     * own portion of KKTcoeffCols,KKTcoeffRows. Tags each element with the partition that will be
     * operating on it, then from this info calculates which columns/rows of the KKT matrix need
     * to be locked when multiple partitions are scattering into the KKT matrix. Allocates KKTLocks
     * mutexes based on this info.
     */

    int KKTfreeloc = 0;

    int eqoffset = this->PrimalVars + this->SlackVars;
    int iqoffset = this->PrimalVars + this->SlackVars + this->EqualCons;
    for (int i = 0; i < this->NumPartitions; i++) {
        int kkstart = KKTfreeloc;

        for (auto &obj : this->PartObj[i])
            obj.getKKTSpace(this->KKTcoeffRows.head(this->numUserKKTElems),
                            this->KKTcoeffCols.head(this->numUserKKTElems), KKTfreeloc, 0, false,
                            true);
        for (auto &eq : this->PartEq[i])
            eq.getKKTSpace(this->KKTcoeffRows.head(this->numUserKKTElems),
                           this->KKTcoeffCols.head(this->numUserKKTElems), KKTfreeloc, eqoffset,
                           true, true);
        for (auto &ineq : this->PartIq[i])
            ineq.getKKTSpace(this->KKTcoeffRows.head(this->numUserKKTElems),
                             this->KKTcoeffCols.head(this->numUserKKTElems), KKTfreeloc, iqoffset,
                             true, true);

        int kklen = KKTfreeloc - kkstart;

        this->KKTcoeffPartIds.segment(kkstart, kklen).setConstant(i);
    }

    Eigen::MatrixXi KKTclash(this->NumPartitions, this->KKTdim);
    KKTclash.setZero();
    for (int i = 0; i < this->numUserKKTElems; i++) {
        int col = this->KKTcoeffCols[i];
        int thrid = this->KKTcoeffPartIds[i];
        KKTclash(thrid, col) = 1;
    }

    this->KKTClashes.resize(this->KKTdim);
    this->numKKTClashes = 0;

    for (int i = 0; i < this->KKTdim; i++) {
        if (KKTclash.col(i).sum() > 1) {
            this->KKTClashes[i] = numKKTClashes;
            numKKTClashes++;
        } else {
            this->KKTClashes[i] = -1;
        }
    }
    std::vector<std::mutex> kktemp(this->numKKTClashes);

    this->KKTLocks.swap(kktemp);
}

void tycho::solvers::NonLinearProgram::getRHSSpace() {
    int PGXfreeloc = 0;
    int AGXfreeloc = 0;
    int FXEfreeloc = 0;
    int FXIfreeloc = 0;

    for (int i = 0; i < this->NumPartitions; i++) {
        for (auto &obj : this->PartObj[i]) {
            obj.getGradientSpace(this->PGXCoeffRows(), PGXfreeloc);
        }
        for (auto &eq : this->PartEq[i]) {
            eq.getGradientSpace(this->AGXCoeffRows(), AGXfreeloc);
            eq.getConstraintSpace(this->EConCoeffRows(), FXEfreeloc);
        }
        for (auto &ineq : this->PartIq[i]) {
            ineq.getGradientSpace(this->AGXCoeffRows(), AGXfreeloc);
            ineq.getConstraintSpace(this->IConCoeffRows(), FXIfreeloc);
        }
    }
}

void tycho::solvers::NonLinearProgram::setMATDimensions() {
    this->KKTdim = this->PrimalVars + this->SlackVars + this->EqualCons + this->InequalCons;

    ////////////////// This is the storage order of Solver data/////////////////
    ////////////////////////////////////////////////////////////////////////////
    this->numSolverKKTElems = this->SlackVars      // solver ijac slack ones
                              + this->PrimalVars   // solver primal hessian diags
                              + this->SlackVars    // solver slack hessian diags
                              + this->EqualCons    // solver equal pivots
                              + this->InequalCons; // solver inequal pivots
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    this->SlackJacDataStart = 0;
    this->PrimalDiagsDataStart = this->SlackJacDataStart + this->SlackVars;
    this->SlackDiagDataStart = this->PrimalDiagsDataStart + this->PrimalVars;
    this->EPivotDataStart = this->SlackDiagDataStart + this->SlackVars;
    this->IPivotDataStart = this->EPivotDataStart + this->EqualCons;

    this->SolverCoeffs = Eigen::VectorXd::Zero(this->numSolverKKTElems);
    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////

    this->numKKTElems = this->numUserKKTElems + this->numSolverKKTElems;

    this->KKTcoeffRows = Eigen::VectorXi::Constant(this->numKKTElems, -1);
    this->KKTcoeffCols = Eigen::VectorXi::Constant(this->numKKTElems, -1);
    this->KKTcoeffPartIds = Eigen::VectorXi::Constant(this->numKKTElems, 0);
    this->KKTLocations = Eigen::VectorXi::Constant(this->numKKTElems, -1);
    this->SolverCoeffs = Eigen::VectorXd::Constant(this->numSolverKKTElems, 0);
    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
}

void tycho::solvers::NonLinearProgram::setRHSDimensions() {
    this->numRHSElems =
        this->numPGXElems + this->numAGXElems + this->numEConElems + this->numIConElems;

    this->PGXDataStart = 0;
    this->AGXDataStart = this->numPGXElems;
    this->EConDataStart = this->AGXDataStart + this->numAGXElems;
    this->IConDataStart = this->EConDataStart + this->numEConElems;

    this->RHScoeffs = Eigen::VectorXd::Zero(this->numRHSElems);
    this->RHScoeffRows = Eigen::VectorXi::Constant(this->numRHSElems, -1);
}

void tycho::solvers::NonLinearProgram::finalizeData() {
    for (int i = 0; i < this->PrimalVars; i++) {
        this->PrimalDiagCoeffCols()[i] = i;
        this->PrimalDiagCoeffRows()[i] = i;
    }

    for (int i = 0; i < this->EqualCons; i++) {
        this->EPivotCoeffCols()[i] = this->PrimalVars + this->SlackVars + i;
        this->EPivotCoeffRows()[i] = this->PrimalVars + this->SlackVars + i;
    }

    for (int i = 0; i < this->InequalCons; i++) {
        this->SlackCoeffCols()[i] = this->PrimalVars + i;
        this->SlackCoeffRows()[i] = this->PrimalVars + this->SlackVars + this->EqualCons + i;

        this->SlackDiagCoeffCols()[i] = this->PrimalVars + i;
        this->SlackDiagCoeffRows()[i] = this->PrimalVars + i;

        this->IPivotCoeffCols()[i] = this->PrimalVars + this->SlackVars + this->EqualCons + i;
        this->IPivotCoeffRows()[i] = this->PrimalVars + this->SlackVars + this->EqualCons + i;
    }
}

void tycho::solvers::NonLinearProgram::analyzeSparsity(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    /*
    Calculates Sparsity Pattern of NLP. PSIOPT requires that only the upper triangular part of a CSR
    matrix be filled. getMATSpace calculates the non-zeros of the lower triangular part. Therefore
    in this routine we transpose the the row-column indices when making the triplet vector that
    Eigen uses to calculate the compressed sparsity pattern of the upper triangular CSR matrix. Once
    this routine clculates the sparsity pattern of the KKT matrix it back calculates where every
    element specified by KKTcoeffRows[i],KKTcoeffCols[i], should be summed into the KKT matrix. This
    info is stored in KKTLocations, and is passed back to all functions so that they know where to
    scatter their outputs.

    */
    KKTmat.resize(this->KKTdim, this->KKTdim);
    std::vector<Eigen::Triplet<double>> kktvec(this->numKKTElems,
                                               Eigen::Triplet<double>(0, 0, 0.0));

    auto TripFillOP = [&](int start, int stop) {
        for (int i = start; i < stop; i++) {
            int row = this->KKTcoeffRows[i];
            int col = this->KKTcoeffCols[i];
            if (col <= row) { //// only accept lower triangular part
                kktvec[i] = Eigen::Triplet<double>(col, row, 1.0);
            } else {
                this->KKTcoeffRows[i] = col;
                this->KKTcoeffCols[i] = row;
                kktvec[i] = Eigen::Triplet<double>(row, col, 1.0);
            }
        }
    };
    tycho::utils::parallel_blocks(this->numKKTElems, TripFillOP, this->NumPartitions);

    KKTmat.setFromTriplets(kktvec.begin(), kktvec.end());
    KKTmat.makeCompressed();

    /////////////////////////////////////////////////////////////
    Eigen::VectorXi innerKKTNNZ(this->KKTdim);

    for (int i = 0; i < this->KKTdim; i++) {
        innerKKTNNZ[i] = KKTmat.row(i).nonZeros();
    }

    auto FindOP = [&](int start, int stop) {
        for (int i = start; i < stop; i++) {
            int row = this->KKTcoeffRows(i);
            int col = this->KKTcoeffCols(i);
            if (col <= row) { //// only accept lower triangular part
                for (int k = 0; k < innerKKTNNZ[col]; k++) {
                    int trow = KKTmat.innerIndexPtr()[KKTmat.outerIndexPtr()[col] + k];
                    if (trow == row) {
                        this->KKTLocations[i] = KKTmat.outerIndexPtr()[col] + k;
                        break;
                    }
                }
            }
        }
    };

    tycho::utils::parallel_blocks(this->numKKTElems, FindOP, this->NumPartitions);
    /////////////////////////////////////////////////////////////
}

void tycho::solvers::NonLinearProgram::evalRHS(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto RHSevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->PartObj[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->PGXCoeffs());
        for (auto &Con : this->PartEq[thrnum])
            Con.constraints_adjointgradient(X, LE, this->EConCoeffs(), this->AGXCoeffs());
        for (auto &Con : this->PartIq[thrnum])
            Con.constraints_adjointgradient(X, LI, this->IConCoeffs(), this->AGXCoeffs());
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->NumPartitions, RHSevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    this->fillRHS(PGX, AGX, FXE, FXI);
}

void tycho::solvers::NonLinearProgram::evalOGC(double ObjScale, ConstEigenRef<VectorXd> X, double &val,
                                      EigenRef<VectorXd> PGX, EigenRef<VectorXd> FXE,
                                      EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->PartObj[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->PGXCoeffs());
        for (auto &Con : this->PartEq[thrnum])
            Con.constraints(X, this->EConCoeffs());
        for (auto &Con : this->PartIq[thrnum])
            Con.constraints(X, this->IConCoeffs());
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->NumPartitions, OGCevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    this->fillPGX(PGX);
    this->fillFXE(FXE);
    this->fillFXI(FXI);
}

void tycho::solvers::NonLinearProgram::evalOCC(double ObjScale, ConstEigenRef<VectorXd> X, double &val,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setConCoeffsZero();
    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->PartObj[thrnum])
            Obj.objective(ObjScale, X, localVal);
        for (auto &Con : this->PartEq[thrnum])
            Con.constraints(X, this->EConCoeffs());
        for (auto &Con : this->PartIq[thrnum])
            Con.constraints(X, this->IConCoeffs());
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->NumPartitions, OGCevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    this->fillFXE(FXE);
    this->fillFXI(FXI);
}

void tycho::solvers::NonLinearProgram::evalOBJ(double ObjScale, ConstEigenRef<VectorXd> X, double &val) {

    std::vector<double> Vals(this->NumPartitions, 0.0);

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->PartObj[thrnum])
            Obj.objective(ObjScale, X, localVal);
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->NumPartitions, OGCevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];
}

void tycho::solvers::NonLinearProgram::evalKKT(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->NumPartitions, 0.0);

    this->setRHSCoeffsZero();

    auto KKTevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->PartObj[thrnum])
            Obj.objective_gradient_hessian(ObjScale, X, localVal, this->PGXCoeffs(), KKTmat,
                                           this->KKTLocations, this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->PartEq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LE, this->EConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->PartIq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LI, this->IConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->NumPartitions, KKTevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    // NOTE: fillSolverCoeffs internally calls parallel_blocks, creating a nested
    // dispatch from the inline arm. Safe because: (1) the calling thread is the main
    // thread (not a pool worker), so the pool absorbs all tasks without deadlock, and
    // (2) fillRHS and fillSolverCoeffs operate on disjoint data (RHS vectors vs. KKT
    // matrix entries), so concurrent execution requires no synchronization.
    tycho::utils::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}

void tycho::solvers::NonLinearProgram::evalKKTNO(double ObjScale, ConstEigenRef<VectorXd> X,
                                        ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                        double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                        EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // No-objective mode: ObjScale and val are unused but kept in the signature
    // for API consistency with evalKKT/evalAUG (polymorphic dispatch via evalNLP).
    (void)ObjScale;
    (void)val;

    this->setRHSCoeffsZero();

    auto KKTevalOP = [&](int thrnum) {
        for (auto &Con : this->PartEq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LE, this->EConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->PartIq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LI, this->IConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
    };

    tycho::utils::parallel_sequence(this->NumPartitions, KKTevalOP);

    // NOTE: nested dispatch from inline arm — see comment in evalKKT.
    tycho::utils::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}
void tycho::solvers::NonLinearProgram::evalSOE(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    // Constraint-only mode: ObjScale and val are unused but kept in the signature
    // for API consistency with evalKKT/evalAUG (polymorphic dispatch via evalNLP).
    (void)ObjScale;
    (void)val;

    this->setRHSCoeffsZero();

    auto SOEevalOP = [&](int thrnum) {
        for (auto &Con : this->PartEq[thrnum])
            Con.constraints_jacobian(X, this->EConCoeffs(), KKTmat, this->KKTLocations,
                                     this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->PartIq[thrnum])
            Con.constraints_jacobian(X, this->IConCoeffs(), KKTmat, this->KKTLocations,
                                     this->KKTClashes, this->KKTLocks);
    };

    tycho::utils::parallel_sequence(this->NumPartitions, SOEevalOP);

    // NOTE: nested dispatch from inline arm — see comment in evalKKT.
    tycho::utils::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}
void tycho::solvers::NonLinearProgram::evalAUG(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto SOEevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->PartObj[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->PGXCoeffs());
        for (auto &Con : this->PartEq[thrnum])
            Con.constraints_jacobian_adjointgradient(X, LE, this->EConCoeffs(), this->AGXCoeffs(),
                                                     KKTmat, this->KKTLocations, this->KKTClashes,
                                                     this->KKTLocks);
        for (auto &Con : this->PartIq[thrnum])
            Con.constraints_jacobian_adjointgradient(X, LI, this->IConCoeffs(), this->AGXCoeffs(),
                                                     KKTmat, this->KKTLocations, this->KKTClashes,
                                                     this->KKTLocks);
        Vals[thrnum] = localVal;
    };

    tycho::utils::parallel_sequence(this->NumPartitions, SOEevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    // NOTE: nested dispatch from inline arm — see comment in evalKKT.
    tycho::utils::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}

void tycho::solvers::NonLinearProgram::NLPTest(const Eigen::VectorXd &x, int n,
                                      std::shared_ptr<NonLinearProgram> nlp1,
                                      std::shared_ptr<NonLinearProgram> nlp2) {
    using std::cout;
    using std::endl;

    Eigen::SparseMatrix<double, Eigen::RowMajor> KKTmat1(nlp1->KKTdim, nlp1->KKTdim);
    Eigen::SparseMatrix<double, Eigen::RowMajor> KKTmat2(nlp1->KKTdim, nlp1->KKTdim);

    nlp1->analyzeSparsity(KKTmat1);
    nlp2->analyzeSparsity(KKTmat2);

    Eigen::VectorXd X = x;

    std::cout << X.size() << endl;

    Eigen::VectorXd FXE1(nlp1->EqualCons);
    Eigen::VectorXd FXE2(nlp1->EqualCons);
    FXE1.setZero();
    FXE2.setZero();

    Eigen::VectorXd LE(nlp1->EqualCons);
    LE.setRandom();
    LE *= 100;

    Eigen::VectorXd FXI1(nlp1->InequalCons);
    Eigen::VectorXd FXI2(nlp1->InequalCons);
    FXI1.setZero();
    FXI2.setZero();

    Eigen::VectorXd LI(nlp1->InequalCons);
    LI.setRandom();
    LI *= 100;
    Eigen::VectorXd PGX1(nlp1->PrimalVars);
    Eigen::VectorXd AGX1(nlp1->PrimalVars);
    PGX1.setZero();
    AGX1.setZero();

    Eigen::VectorXd PGX2(nlp1->PrimalVars);
    Eigen::VectorXd AGX2(nlp1->PrimalVars);
    PGX2.setZero();
    AGX2.setZero();

    double v1 = 0;
    double v2 = 0;

    tycho::utils::Timer t1;
    tycho::utils::Timer t2;

    tycho::utils::Timer t3;
    tycho::utils::Timer t4;

    cout << nlp1->KKTLocations.minCoeff() << endl;
    // nlp2->KKTClashes.setConstant(-1);

    for (int i = 0; i < n; i++) {
        std::fill_n(KKTmat1.valuePtr(), KKTmat1.nonZeros(), 0.0);
        std::fill_n(KKTmat2.valuePtr(), KKTmat2.nonZeros(), 0.0);

        t1.start();
        nlp1->evalKKT(1.0, X, LE, LI, v1, PGX1, AGX1, FXE1, FXI1, KKTmat1);
        t1.stop();

        t2.start();
        nlp2->evalKKT(1.0, X, LE, LI, v2, PGX2, AGX2, FXE2, FXI2, KKTmat2);
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
        nlp1->evalOCC(1.0, X, v1, FXE1, FXI1);
        t3.stop();

        t4.start();
        nlp2->evalOCC(1.0, X, v2, FXE2, FXI2);
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
