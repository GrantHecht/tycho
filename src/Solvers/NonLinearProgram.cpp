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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#include "NonLinearProgram.h"

void Tycho::NonLinearProgram::make_NLP(int PV, int EQ, int IQ) {
    this->PrimalVars = PV;
    this->EqualCons = EQ;
    this->InequalCons = IQ;
    this->SlackVars = IQ;

    this->countElems();
    this->analyzeThreading();
    this->setMATDimensions();
    this->setRHSDimensions();

    this->getMATSpace();
    this->getRHSSpace();
    this->finalizeData();
}

void Tycho::NonLinearProgram::countElems() {
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

void Tycho::NonLinearProgram::analyzeThreading() {
    /*
    This function loops over the Master list of objective and constraints and partitions them onto
    the different threads allocated for function evaluation.
    */
    this->ThrObj.clear();
    this->ThrEq.clear();
    this->ThrIq.clear();

    this->ThrObj.resize(this->NumPartitions);
    this->ThrEq.resize(this->NumPartitions);
    this->ThrIq.resize(this->NumPartitions);

    int RRThr = 0;

    auto analyzeOP = [&](auto &SourceFuncs, auto &TargetThrFuncs) {
        for (auto &func : SourceFuncs) {
            if (func.getThreadMode() == ThreadingFlags::MainThread) { // Force to main thread
                TargetThrFuncs.back().push_back(func);
            } else if (func.getThreadMode() == ThreadingFlags::RoundRobin) {
                TargetThrFuncs[RRThr].push_back(func);
                RRThr++;
                if (RRThr > (this->NumPartitions - 1))
                    RRThr = 0;
            } else if (static_cast<int>(func.getThreadMode()) >= 0) { // Specific Thread Assignment
                int thr = std::min(static_cast<int>(func.getThreadMode()), this->NumPartitions - 1);
                TargetThrFuncs[thr].push_back(func);
            } else { // By application
                auto TempThrFuncs = func.thread_split(this->NumPartitions);
                for (int i = 0; i < TempThrFuncs.size(); i++) {
                    TargetThrFuncs[i].push_back(TempThrFuncs[i]);
                }
            }
        }
    };

    analyzeOP(this->Objectives, this->ThrObj);
    analyzeOP(this->EqualityConstraints, this->ThrEq);
    analyzeOP(this->InequalityConstraints, this->ThrIq);
}

void Tycho::NonLinearProgram::getMATSpace() {
    /*
     * Loops over all constraints and objectives on each thread and has each claim its
     * own portion of KKTcoeffCols,KKTcoeffRows. Tags each element with thread that will be
     * operating on it then from this info and calculates which columns/rows of the KKT matrix need
     * to be locked when multiple threads are scattering into KKT matrix. Allocates KKTLocks mutexs
     * based on this info.
     */

    int KKTfreeloc = 0;

    int eqoffset = this->PrimalVars + this->SlackVars;
    int iqoffset = this->PrimalVars + this->SlackVars + this->EqualCons;
    for (int i = 0; i < this->NumPartitions; i++) {
        int kkstart = KKTfreeloc;

        for (auto &obj : this->ThrObj[i])
            obj.getKKTSpace(this->KKTcoeffRows.head(this->numUserKKTElems),
                            this->KKTcoeffCols.head(this->numUserKKTElems), KKTfreeloc, 0, false,
                            true);
        for (auto &eq : this->ThrEq[i])
            eq.getKKTSpace(this->KKTcoeffRows.head(this->numUserKKTElems),
                           this->KKTcoeffCols.head(this->numUserKKTElems), KKTfreeloc, eqoffset,
                           true, true);
        for (auto &ineq : this->ThrIq[i])
            ineq.getKKTSpace(this->KKTcoeffRows.head(this->numUserKKTElems),
                             this->KKTcoeffCols.head(this->numUserKKTElems), KKTfreeloc, iqoffset,
                             true, true);

        int kklen = KKTfreeloc - kkstart;

        this->KKTcoeffThrIds.segment(kkstart, kklen).setConstant(i);
    }

    Eigen::MatrixXi KKTclash(this->NumPartitions, this->KKTdim);
    KKTclash.setZero();
    for (int i = 0; i < this->numUserKKTElems; i++) {
        int col = this->KKTcoeffCols[i];
        int thrid = this->KKTcoeffThrIds[i];
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

void Tycho::NonLinearProgram::getRHSSpace() {
    int PGXfreeloc = 0;
    int AGXfreeloc = 0;
    int FXEfreeloc = 0;
    int FXIfreeloc = 0;

    for (int i = 0; i < this->NumPartitions; i++) {
        for (auto &obj : this->ThrObj[i]) {
            obj.getGradientSpace(this->PGXCoeffRows(), PGXfreeloc);
        }
        for (auto &eq : this->ThrEq[i]) {
            eq.getGradientSpace(this->AGXCoeffRows(), AGXfreeloc);
            eq.getConstraintSpace(this->EConCoeffRows(), FXEfreeloc);
        }
        for (auto &ineq : this->ThrIq[i]) {
            ineq.getGradientSpace(this->AGXCoeffRows(), AGXfreeloc);
            ineq.getConstraintSpace(this->IConCoeffRows(), FXIfreeloc);
        }
    }
}

void Tycho::NonLinearProgram::setMATDimensions() {
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
    this->KKTcoeffThrIds = Eigen::VectorXi::Constant(this->numKKTElems, 0);
    this->KKTLocations = Eigen::VectorXi::Constant(this->numKKTElems, -1);
    this->SolverCoeffs = Eigen::VectorXd::Constant(this->numSolverKKTElems, 0);
    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
}

void Tycho::NonLinearProgram::setRHSDimensions() {
    this->numRHSElems =
        this->numPGXElems + this->numAGXElems + this->numEConElems + this->numIConElems;

    this->PGXDataStart = 0;
    this->AGXDataStart = this->numPGXElems;
    this->EConDataStart = this->AGXDataStart + this->numAGXElems;
    this->IConDataStart = this->EConDataStart + this->numEConElems;

    this->RHScoeffs = Eigen::VectorXd::Zero(this->numRHSElems);
    this->RHScoeffRows = Eigen::VectorXi::Constant(this->numRHSElems, -1);
}

void Tycho::NonLinearProgram::finalizeData() {
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

void Tycho::NonLinearProgram::analyzeSparsity(
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
    Tycho::parallel_blocks(this->numKKTElems, TripFillOP, this->NumPartitions);

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

    Tycho::parallel_blocks(this->numKKTElems, FindOP, this->NumPartitions);
    // this->make_compressed();
    /////////////////////////////////////////////////////////////
}

void Tycho::NonLinearProgram::evalRHS(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto RHSevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->ThrObj[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->PGXCoeffs());
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints_adjointgradient(X, LE, this->EConCoeffs(), this->AGXCoeffs());
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints_adjointgradient(X, LI, this->IConCoeffs(), this->AGXCoeffs());
        Vals[thrnum] = localVal;
    };

    Tycho::parallel_sequence(this->NumPartitions, RHSevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    this->fillRHS(PGX, AGX, FXE, FXI);
}

void Tycho::NonLinearProgram::evalOGC(double ObjScale, ConstEigenRef<VectorXd> X, double &val,
                                      EigenRef<VectorXd> PGX, EigenRef<VectorXd> FXE,
                                      EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->ThrObj[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->PGXCoeffs());
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints(X, this->EConCoeffs());
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints(X, this->IConCoeffs());
        Vals[thrnum] = localVal;
    };

    Tycho::parallel_sequence(this->NumPartitions, OGCevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    this->fillPGX(PGX);
    this->fillFXE(FXE);
    this->fillFXI(FXI);
}

void Tycho::NonLinearProgram::evalOCC(double ObjScale, ConstEigenRef<VectorXd> X, double &val,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    // this->setRHSCoeffsZero();
    this->setConCoeffsZero();
    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->ThrObj[thrnum])
            Obj.objective(ObjScale, X, localVal);
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints(X, this->EConCoeffs());
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints(X, this->IConCoeffs());
        Vals[thrnum] = localVal;
    };

    Tycho::parallel_sequence(this->NumPartitions, OGCevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    this->fillFXE(FXE);
    this->fillFXI(FXI);
}

void Tycho::NonLinearProgram::evalOBJ(double ObjScale, ConstEigenRef<VectorXd> X, double &val) {

    std::vector<double> Vals(this->NumPartitions, 0.0);

    auto OGCevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->ThrObj[thrnum])
            Obj.objective(ObjScale, X, localVal);
        Vals[thrnum] = localVal;
    };

    Tycho::parallel_sequence(this->NumPartitions, OGCevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];
}

void Tycho::NonLinearProgram::evalKKT(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->NumPartitions, 0.0);

    this->setRHSCoeffsZero();

    auto KKTevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->ThrObj[thrnum])
            Obj.objective_gradient_hessian(ObjScale, X, localVal, this->PGXCoeffs(), KKTmat,
                                           this->KKTLocations, this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LE, this->EConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LI, this->IConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
        Vals[thrnum] = localVal;
    };

    Tycho::parallel_sequence(this->NumPartitions, KKTevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    Tycho::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}

void Tycho::NonLinearProgram::evalKKTNO(double ObjScale, ConstEigenRef<VectorXd> X,
                                        ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                        double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                        EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto KKTevalOP = [&](int thrnum) {
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LE, this->EConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LI, this->IConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
    };

    Tycho::parallel_sequence(this->NumPartitions, KKTevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    this->fillSolverCoeffs(KKTmat);

    this->fillRHS(PGX, AGX, FXE, FXI);
}
void Tycho::NonLinearProgram::evalSOE(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto SOEevalOP = [&](int thrnum) {
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints_jacobian(X, this->EConCoeffs(), KKTmat, this->KKTLocations,
                                     this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints_jacobian(X, this->IConCoeffs(), KKTmat, this->KKTLocations,
                                     this->KKTClashes, this->KKTLocks);
    };

    Tycho::parallel_sequence(this->NumPartitions, SOEevalOP);

    Tycho::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}
void Tycho::NonLinearProgram::evalAUG(double ObjScale, ConstEigenRef<VectorXd> X,
                                      ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                      double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                      EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    std::vector<double> Vals(this->NumPartitions, 0.0);
    this->setRHSCoeffsZero();

    auto SOEevalOP = [&](int thrnum) {
        double localVal = 0.0;
        for (auto &Obj : this->ThrObj[thrnum])
            Obj.objective_gradient(ObjScale, X, localVal, this->PGXCoeffs());
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints_jacobian_adjointgradient(X, LE, this->EConCoeffs(), this->AGXCoeffs(),
                                                     KKTmat, this->KKTLocations, this->KKTClashes,
                                                     this->KKTLocks);
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints_jacobian_adjointgradient(X, LI, this->IConCoeffs(), this->AGXCoeffs(),
                                                     KKTmat, this->KKTLocations, this->KKTClashes,
                                                     this->KKTLocks);
        Vals[thrnum] = localVal;
    };

    Tycho::parallel_sequence(this->NumPartitions, SOEevalOP);
    for (int i = 0; i < this->NumPartitions; i++)
        val += Vals[i];

    Tycho::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}

void Tycho::NonLinearProgram::NLPTest(const Eigen::VectorXd &x, int n,
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

    Utils::Timer t1;
    Utils::Timer t2;

    Utils::Timer t3;
    Utils::Timer t4;

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
