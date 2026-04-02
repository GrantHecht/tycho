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

#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"

void tycho::oc::LGLInterpTable::set_method(TranscriptionModes m) {
    this->method_ = m;
    switch (this->method_) {

    case TranscriptionModes::CentralShooting:
    case TranscriptionModes::Trapezoidal:
    case TranscriptionModes::LGL3: {
        this->block_size_ = 2;
        this->t_spacing_.resize(2);
        this->t_spacing_[0] = 0;
        this->t_spacing_[1] = 1;

        this->x_weights_.resize(4, 2);
        this->dx_weights_.resize(4, 2);
        this->u_weights_.resize(2, 2);
        this->order_ = 3.0;
        this->error_weight_ = LGLCoeffs<2>::error_weight_;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                this->x_weights_(j, i) = LGLCoeffs<2>::Cardinal_XPower_Weights[i][3 - j];
                this->dx_weights_(j, i) = LGLCoeffs<2>::Cardinal_DXPower_Weights[i][3 - j];
            }
            for (int j = 0; j < 2; j++) {
                this->u_weights_(j, i) = LGLCoeffs<2>::Cardinal_UPolyPower_Weights[i][1 - j];
            }
        }

        break;
    }
    case TranscriptionModes::LGL5: {
        this->block_size_ = 3;
        this->t_spacing_.resize(3);
        this->t_spacing_[0] = 0;
        this->t_spacing_[1] = 0.5;
        this->t_spacing_[2] = 1;

        this->x_weights_.resize(6, 3);
        this->dx_weights_.resize(6, 3);
        this->u_weights_.resize(3, 3);
        this->order_ = 5.0;
        this->error_weight_ = LGLCoeffs<3>::error_weight_;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 6; j++) {
                this->x_weights_(j, i) = LGLCoeffs<3>::Cardinal_XPower_Weights[i][5 - j];
                this->dx_weights_(j, i) = LGLCoeffs<3>::Cardinal_DXPower_Weights[i][5 - j];
            }
            for (int j = 0; j < 3; j++) {
                this->u_weights_(j, i) = LGLCoeffs<3>::Cardinal_UPolyPower_Weights[i][2 - j];
            }
        }

        break;
    }

    case TranscriptionModes::LGL7: {
        this->block_size_ = 4;
        this->t_spacing_.resize(4);
        this->t_spacing_[0] = 0;
        this->t_spacing_[1] = 2.65575603264643e-1;
        this->t_spacing_[2] = 1.0 - 2.65575603264643e-1;
        this->t_spacing_[3] = 1.0;
        this->x_weights_.resize(8, 4);
        this->dx_weights_.resize(8, 4);
        this->u_weights_.resize(4, 4);
        this->order_ = 7.0;
        this->error_weight_ = LGLCoeffs<4>::error_weight_;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                this->x_weights_(j, i) = LGLCoeffs<4>::Cardinal_XPower_Weights[i][7 - j];
                this->dx_weights_(j, i) = LGLCoeffs<4>::Cardinal_DXPower_Weights[i][7 - j];
            }
            for (int j = 0; j < 4; j++) {
                this->u_weights_(j, i) = LGLCoeffs<4>::Cardinal_UPolyPower_Weights[i][3 - j];
            }
        }
        break;
    }
    }
}

std::vector<Eigen::VectorXd> tycho::oc::LGLInterpTable::new_error_integral() {

    auto factorial = [](int n) {
        double fact = 1;
        for (int i = 1; i <= n; i++)
            fact = fact * i;
        return fact;
    };

    double PolyFact = factorial(this->order_);

    Eigen::VectorXd XerrWeights = this->x_weights_.bottomRows(1).transpose() * PolyFact;
    Eigen::VectorXd DXerrWeights = this->dx_weights_.bottomRows(1).transpose() * PolyFact;

    Eigen::MatrixXd yvecs(this->x_vars_, this->num_blocks_);
    Eigen::VectorXd hs(this->num_blocks_);
    Eigen::VectorXd tsnd(this->num_blocks_ + 1);
    Eigen::VectorXd errs(this->num_blocks_ + 1);
    Eigen::VectorXd errs2(this->num_blocks_ + 1);

    Eigen::VectorXd yvec(this->x_vars_);
    for (int i = 0; i < this->num_blocks_; i++) {
        int start = (this->block_size_ - 1) * i;

        hs[i] = this->xtu_data_.col(start + (this->block_size_ - 1))[this->x_vars_] -
                this->xtu_data_.col(start)[this->x_vars_];

        tsnd[i] = (this->xtu_data_.col(start)[this->x_vars_] - this->t0_) / this->total_t_;

        yvec.setZero();
        double powh = std::pow(hs[i], this->order_);

        for (int j = 0; j < this->block_size_; j++) {
            yvec += this->xtu_data_.col(start + j).head(this->x_vars_) * XerrWeights[j] / powh;
            yvec += this->xdot_data_.col(start + j).head(this->x_vars_) * DXerrWeights[j] * hs[i] /
                    powh;
        }

        yvecs.col(i) = yvec;
    }

    tsnd[this->num_blocks_] = 1.0;

    for (int i = 0; i < this->num_blocks_; i++) {

        double err;
        if (i > 0 && i < (this->num_blocks_ - 1)) {
            err =
                (yvecs.col(i) - yvecs.col(i - 1)).lpNorm<Eigen::Infinity>() / (hs[i] + hs[i - 1]) +
                (yvecs.col(i) - yvecs.col(i + 1)).lpNorm<Eigen::Infinity>() / (hs[i] + hs[i + 1]);

            err = ((yvecs.col(i) - yvecs.col(i - 1)) / (hs[i] + hs[i - 1]) +
                   (yvecs.col(i) - yvecs.col(i + 1)) / (hs[i] + hs[i + 1]))
                      .lpNorm<Eigen::Infinity>();
        } else if (i == 0) {
            err = 2 * (yvecs.col(i) - yvecs.col(i + 1)).lpNorm<Eigen::Infinity>() /
                  (hs[i] + hs[i + 1]);
        } else {
            err = 2 * (yvecs.col(i) - yvecs.col(i - 1)).lpNorm<Eigen::Infinity>() /
                  (hs[i] + hs[i - 1]);
        }

        errs[i] = std::pow(std::abs(err), 1 / (this->order_ + 1));
        errs2[i] = std::abs(err) * std::pow(hs[i], this->order_ + 1) * this->error_weight_;
    }
    errs[this->num_blocks_] = errs[this->num_blocks_ - 1];
    errs2[this->num_blocks_] = errs2[this->num_blocks_ - 1];

    Eigen::VectorXd errint(this->num_blocks_ + 1);
    errint[0] = 0;

    for (int i = 0; i < this->num_blocks_; i++) {
        errint[i + 1] = errint[i] + (errs[i]) * (tsnd[i + 1] - tsnd[i]);
    }
    return std::vector{tsnd, errs2, errint};
}

void tycho::oc::LGLInterpTable::deboor_mesh_error(Eigen::VectorXd &tsnd,
                                                  Eigen::MatrixXd &mesh_errors,
                                                  Eigen::MatrixXd &mesh_dist) const {

    auto factorial = [](int n) {
        double fact = 1;
        for (int i = 1; i <= n; i++)
            fact = fact * i;
        return fact;
    };

    double PolyFact = factorial(this->order_);

    Eigen::VectorXd XerrWeights = this->x_weights_.bottomRows(1).transpose() * PolyFact;
    Eigen::VectorXd DXerrWeights = this->dx_weights_.bottomRows(1).transpose() * PolyFact;

    Eigen::MatrixXd yvecs(this->x_vars_, this->num_blocks_);
    Eigen::VectorXd hs(this->num_blocks_);

    tsnd.resize(this->num_blocks_ + 1);

    Eigen::VectorXd yvec(this->x_vars_);
    for (int i = 0; i < this->num_blocks_; i++) {
        int start = (this->block_size_ - 1) * i;

        hs[i] = this->xtu_data_.col(start + (this->block_size_ - 1))[this->x_vars_] -
                this->xtu_data_.col(start)[this->x_vars_];

        tsnd[i] = (this->xtu_data_.col(start)[this->x_vars_] - this->t0_) / this->total_t_;

        yvec.setZero();
        double powh = std::pow(hs[i], this->order_);

        for (int j = 0; j < this->block_size_; j++) {
            yvec += this->xtu_data_.col(start + j).head(this->x_vars_) * XerrWeights[j] / powh;
            yvec += this->xdot_data_.col(start + j).head(this->x_vars_) * DXerrWeights[j] * hs[i] /
                    powh;
        }

        yvecs.col(i) = yvec;
    }

    tsnd[this->num_blocks_] = 1.0;

    mesh_errors.resize(this->x_vars_, this->num_blocks_ + 1);
    mesh_dist.resize(this->x_vars_, this->num_blocks_ + 1);

    Eigen::VectorXd err_tmp(this->x_vars_);
    for (int i = 0; i < this->num_blocks_; i++) {

        double err;
        if (i > 0 && i < (this->num_blocks_ - 1)) {

            err_tmp = ((yvecs.col(i) - yvecs.col(i - 1)) / (hs[i] + hs[i - 1]) +
                       (yvecs.col(i) - yvecs.col(i + 1)) / (hs[i] + hs[i + 1]))
                          .cwiseAbs();
        } else if (i == 0) {
            err_tmp = 2 * (yvecs.col(i) - yvecs.col(i + 1)).cwiseAbs() / (hs[i] + hs[i + 1]);
        } else {
            err_tmp = 2 * (yvecs.col(i) - yvecs.col(i - 1)).cwiseAbs() / (hs[i] + hs[i - 1]);
        }

        mesh_dist.col(i) = err_tmp.array().pow(1 / (this->order_ + 1));
        mesh_errors.col(i) =
            err_tmp * std::pow(std::abs(hs[i]), this->order_ + 1) * this->error_weight_;
    }

    mesh_dist.col(this->num_blocks_) = mesh_dist.col(this->num_blocks_ - 1);
    mesh_errors.col(this->num_blocks_) = mesh_errors.col(this->num_blocks_ - 1);
}
