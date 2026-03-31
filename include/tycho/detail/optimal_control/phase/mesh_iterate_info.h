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

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <fmt/color.h>
#include <fmt/format.h>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/type_name.h"

namespace tycho::oc {

struct MeshIterateInfo {

    bool converged_ = false;
    int numsegs_;
    int up_numsegs_;
    double tol_;

    double avg_error_;
    double gmean_error_;
    double max_error_;
    double global_error_ = -1;

    Eigen::VectorXd times_;
    Eigen::VectorXd error_;
    Eigen::VectorXd distribution_;
    Eigen::VectorXd distintegral_;

    MeshIterateInfo() {}

    MeshIterateInfo(int numsegs, double tol, const Eigen::VectorXd &times,
                    const Eigen::VectorXd &error, const Eigen::VectorXd &distribution)
        : numsegs_(numsegs), up_numsegs_(numsegs), tol_(tol), times_(times), error_(error),
          distribution_(distribution) {

        int n = times.size();
        ////////////////////////////////////////////////////

        Eigen::VectorXd hs = this->times_.tail(n - 1) - this->times_.head(n - 1);
        this->max_error_ = this->error_.maxCoeff();
        this->avg_error_ = (this->error_.head(n - 1).cwiseProduct(hs)).sum();

        this->gmean_error_ = std::exp((std::log(this->max_error_) + std::log(this->avg_error_)) / 2.0);

        // this->gmean_error_ = std::exp((this->error_.head(n - 1).array().log()*hs.array()).sum());

        ////////////////////////////////////////////////////
        this->distintegral_.resize(this->times_.size());
        this->distintegral_[0] = 0;

        for (int i = 0; i < n - 1; i++) {
            this->distintegral_[i + 1] =
                this->distintegral_[i] +
                (this->distribution_[i]) * (this->times_[i + 1] - this->times_[i]);
        }

        this->distintegral_ = this->distintegral_ / this->distintegral_[n - 1];
        /////////////////////////////////////////////////////
    }

    Eigen::VectorXd calc_bins(int nbins) {

        Eigen::VectorXd bins;
        bins.setLinSpaced(nbins + 1, 0.0, 1.0);
        int elem = 0;
        for (int i = 1; i < nbins; i++) {
            double di = double(i) / double(nbins);
            auto it =
                std::upper_bound(this->distintegral_.cbegin() + elem, this->distintegral_.cend(), di);
            elem = int(it - this->distintegral_.cbegin()) - 1;

            double t0 = this->times_[elem];
            double t1 = this->times_[elem + 1];
            double d0 = this->distintegral_[elem];
            double d1 = this->distintegral_[elem + 1];
            double slope = (d1 - d0) / (t1 - t0);
            bins[i] = (di - d0) / slope + t0;
        }
        return bins;
    }

    static void print_header(int iter) {

        fmt::print("{0:=^{1}}\n", "", 52);
        fmt::print("Mesh Iteration: {0:}\n", iter);
        fmt::print("{0:=^{1}}\n", "", 52);
        fmt::print("|Phase|#Segs| Max Err | Avg Err | EtE Err |Up #Segs|\n");
    }

    void print(int phasenum) {

        auto level1 = std::log(tol_);
        auto level3 = std::log(tol_ * 1000);
        auto level5 = std::log(tol_ * 100000.0);
        auto level2 = (level1 + level3) / 2.0;
        auto level4 = (level3 + level5) / 2.0;

        auto calccolor = [&](double val) {
            auto logval = std::log(val);
            fmt::color c;

            if (logval < level1)
                c = fmt::color::lime_green;
            else if (logval < level2)
                c = fmt::color::yellow;
            else if (logval < level3)
                c = fmt::color::orange;
            else if (logval < level4)
                c = fmt::color::red;
            else
                c = fmt::color::dark_red;
            return fmt::fg(c);
        };

        fmt::print("|{:<5}|", phasenum);
        fmt::print("{:<5}|", this->numsegs_);
        fmt::print(calccolor(this->max_error_), "{:>9.3e}", this->max_error_);
        fmt::print("|");
        fmt::print(calccolor(this->avg_error_), "{:>9.3e}", this->avg_error_);
        fmt::print("|");
        if (this->global_error_ > 0) {
            fmt::print(calccolor(this->global_error_), "{:>9.3e}", this->global_error_);
            fmt::print("|");

        } else {
            fmt::print("   N/A   |");
        }
        fmt::print("{:<8}|\n", this->up_numsegs_);
    }
};

} // namespace tycho::oc
