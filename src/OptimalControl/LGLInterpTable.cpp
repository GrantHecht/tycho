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

#include "LGLInterpTable.h"

void Tycho::LGLInterpTable::setMethod(TranscriptionModes m) {
    this->Method = m;
    switch (this->Method) {

    case TranscriptionModes::CentralShooting:
    case TranscriptionModes::Trapezoidal:
    case TranscriptionModes::LGL3: {
        this->BlockSize = 2;
        this->Tspacing.resize(2);
        this->Tspacing[0] = 0;
        this->Tspacing[1] = 1;

        this->Xweights.resize(4, 2);
        this->DXweights.resize(4, 2);
        this->Uweights.resize(2, 2);
        this->Order = 3.0;
        this->ErrorWeight = LGLCoeffs<2>::ErrorWeight;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                this->Xweights(j, i) = LGLCoeffs<2>::Cardinal_XPower_Weights[i][3 - j];
                this->DXweights(j, i) = LGLCoeffs<2>::Cardinal_DXPower_Weights[i][3 - j];
            }
            for (int j = 0; j < 2; j++) {
                this->Uweights(j, i) = LGLCoeffs<2>::Cardinal_UPolyPower_Weights[i][1 - j];
            }
        }

        break;
    }
    case TranscriptionModes::LGL5: {
        this->BlockSize = 3;
        this->Tspacing.resize(3);
        this->Tspacing[0] = 0;
        this->Tspacing[1] = 0.5;
        this->Tspacing[2] = 1;

        this->Xweights.resize(6, 3);
        this->DXweights.resize(6, 3);
        this->Uweights.resize(3, 3);
        this->Order = 5.0;
        this->ErrorWeight = LGLCoeffs<3>::ErrorWeight;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 6; j++) {
                this->Xweights(j, i) = LGLCoeffs<3>::Cardinal_XPower_Weights[i][5 - j];
                this->DXweights(j, i) = LGLCoeffs<3>::Cardinal_DXPower_Weights[i][5 - j];
            }
            for (int j = 0; j < 3; j++) {
                this->Uweights(j, i) = LGLCoeffs<3>::Cardinal_UPolyPower_Weights[i][2 - j];
            }
        }

        break;
    }

    case TranscriptionModes::LGL7: {
        this->BlockSize = 4;
        this->Tspacing.resize(4);
        this->Tspacing[0] = 0;
        this->Tspacing[1] = 2.65575603264643e-1;
        this->Tspacing[2] = 1.0 - 2.65575603264643e-1;
        this->Tspacing[3] = 1.0;
        this->Xweights.resize(8, 4);
        this->DXweights.resize(8, 4);
        this->Uweights.resize(4, 4);
        this->Order = 7.0;
        this->ErrorWeight = LGLCoeffs<4>::ErrorWeight;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                this->Xweights(j, i) = LGLCoeffs<4>::Cardinal_XPower_Weights[i][7 - j];
                this->DXweights(j, i) = LGLCoeffs<4>::Cardinal_DXPower_Weights[i][7 - j];
            }
            for (int j = 0; j < 4; j++) {
                this->Uweights(j, i) = LGLCoeffs<4>::Cardinal_UPolyPower_Weights[i][3 - j];
            }
        }
        break;
    }
    }
}

std::vector<Eigen::VectorXd> Tycho::LGLInterpTable::NewErrorIntegral() {

    auto factorial = [](int n) {
        double fact = 1;
        for (int i = 1; i <= n; i++)
            fact = fact * i;
        return fact;
    };

    double PolyFact = factorial(this->Order);

    Eigen::VectorXd XerrWeights = this->Xweights.bottomRows(1).transpose() * PolyFact;
    Eigen::VectorXd DXerrWeights = this->DXweights.bottomRows(1).transpose() * PolyFact;

    std::vector<Eigen::VectorXd> yvecs(this->NumBlocks);
    Eigen::VectorXd hs(this->NumBlocks);
    Eigen::VectorXd tsnd(this->NumBlocks + 1);
    Eigen::VectorXd errs(this->NumBlocks + 1);
    Eigen::VectorXd errs2(this->NumBlocks + 1);

    for (int i = 0; i < this->NumBlocks; i++) {
        int start = (this->BlockSize - 1) * i;

        hs[i] = this->XtUData.col(start + (this->BlockSize - 1))[this->XVars] -
                this->XtUData.col(start)[this->XVars];

        tsnd[i] = (this->XtUData.col(start)[this->XVars] - this->T0) / this->TotalT;

        Eigen::VectorXd yvec(this->XVars);
        yvec.setZero();
        double powh = std::pow(hs[i], this->Order);

        for (int j = 0; j < this->BlockSize; j++) {
            yvec += this->XtUData.col(start + j).head(this->XVars) * XerrWeights[j] / powh;
            yvec +=
                this->XdotData.col(start + j).head(this->XVars) * DXerrWeights[j] * hs[i] / powh;
        }

        yvecs[i] = yvec;
    }

    tsnd[this->NumBlocks] = 1.0;

    for (int i = 0; i < this->NumBlocks; i++) {

        double err;
        if (i > 0 && i < (this->NumBlocks - 1)) {
            err = (yvecs[i] - yvecs[i - 1]).lpNorm<Eigen::Infinity>() / (hs[i] + hs[i - 1]) +
                  (yvecs[i] - yvecs[i + 1]).lpNorm<Eigen::Infinity>() / (hs[i] + hs[i + 1]);

            err = ((yvecs[i] - yvecs[i - 1]) / (hs[i] + hs[i - 1]) +
                   (yvecs[i] - yvecs[i + 1]) / (hs[i] + hs[i + 1]))
                      .lpNorm<Eigen::Infinity>();
        } else if (i == 0) {
            err = 2 * (yvecs[i] - yvecs[i + 1]).lpNorm<Eigen::Infinity>() / (hs[i] + hs[i + 1]);
        } else {
            err = 2 * (yvecs[i] - yvecs[i - 1]).lpNorm<Eigen::Infinity>() / (hs[i] + hs[i - 1]);
        }

        errs[i] = std::pow(std::abs(err), 1 / (this->Order + 1));
        errs2[i] = std::abs(err) * std::pow(hs[i], this->Order + 1) * this->ErrorWeight;
    }
    errs[this->NumBlocks] = errs[this->NumBlocks - 1];
    errs2[this->NumBlocks] = errs2[this->NumBlocks - 1];

    Eigen::VectorXd errint(this->NumBlocks + 1);
    errint[0] = 0;

    for (int i = 0; i < this->NumBlocks; i++) {
        errint[i + 1] = errint[i] + (errs[i]) * (tsnd[i + 1] - tsnd[i]);
    }
    return std::vector{tsnd, errs2, errint};
}

void Tycho::LGLInterpTable::DeboorMeshError(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                                            Eigen::MatrixXd &mesh_dist) const {

    auto factorial = [](int n) {
        double fact = 1;
        for (int i = 1; i <= n; i++)
            fact = fact * i;
        return fact;
    };

    double PolyFact = factorial(this->Order);

    Eigen::VectorXd XerrWeights = this->Xweights.bottomRows(1).transpose() * PolyFact;
    Eigen::VectorXd DXerrWeights = this->DXweights.bottomRows(1).transpose() * PolyFact;

    std::vector<Eigen::VectorXd> yvecs(this->NumBlocks);
    Eigen::VectorXd hs(this->NumBlocks);

    tsnd.resize(this->NumBlocks + 1);

    for (int i = 0; i < this->NumBlocks; i++) {
        int start = (this->BlockSize - 1) * i;

        hs[i] = this->XtUData.col(start + (this->BlockSize - 1))[this->XVars] -
                this->XtUData.col(start)[this->XVars];

        tsnd[i] = (this->XtUData.col(start)[this->XVars] - this->T0) / this->TotalT;

        Eigen::VectorXd yvec(this->XVars);
        yvec.setZero();
        double powh = std::pow(hs[i], this->Order);

        for (int j = 0; j < this->BlockSize; j++) {
            yvec += this->XtUData.col(start + j).head(this->XVars) * XerrWeights[j] / powh;
            yvec +=
                this->XdotData.col(start + j).head(this->XVars) * DXerrWeights[j] * hs[i] / powh;
        }

        yvecs[i] = yvec;
    }

    tsnd[this->NumBlocks] = 1.0;

    mesh_errors.resize(this->XVars, this->NumBlocks + 1);
    mesh_dist.resize(this->XVars, this->NumBlocks + 1);

    for (int i = 0; i < this->NumBlocks; i++) {

        Eigen::VectorXd err_tmp(this->XVars);

        double err;
        if (i > 0 && i < (this->NumBlocks - 1)) {

            err_tmp = ((yvecs[i] - yvecs[i - 1]) / (hs[i] + hs[i - 1]) +
                       (yvecs[i] - yvecs[i + 1]) / (hs[i] + hs[i + 1]))
                          .cwiseAbs();
        } else if (i == 0) {
            err_tmp = 2 * (yvecs[i] - yvecs[i + 1]).cwiseAbs() / (hs[i] + hs[i + 1]);
        } else {
            err_tmp = 2 * (yvecs[i] - yvecs[i - 1]).cwiseAbs() / (hs[i] + hs[i - 1]);
        }

        mesh_dist.col(i) = err_tmp.array().pow(1 / (this->Order + 1));
        mesh_errors.col(i) =
            err_tmp * std::pow(std::abs(hs[i]), this->Order + 1) * this->ErrorWeight;
    }

    mesh_dist.col(this->NumBlocks) = mesh_dist.col(this->NumBlocks - 1);
    mesh_errors.col(this->NumBlocks) = mesh_errors.col(this->NumBlocks - 1);
}
