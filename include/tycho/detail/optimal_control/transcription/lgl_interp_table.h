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

#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/transcription/lgl_coeffs.h"
#include "tycho/detail/vf/derivatives/fd_deriv_arbitrary.h"
#include "tycho/vector_functions.h"
#include <algorithm>
#include <array>
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

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::oc {

using vf::FDDerivArbitrary;
using vf::GenericFunction;

struct LGLInterpTable {
    using VectorFunctionalX = GenericFunction<-1, -1>;

    Eigen::MatrixXd xtu_data_;
    Eigen::MatrixXd xdot_data_;

    int x_vars_ = 0;
    int u_vars_ = 0;
    int xtu_vars_ = 0;
    int axis_ = 0;

    double delta_t_ = 0.0;
    double total_t_ = 0.0;
    double t0_ = 0;
    double tf_ = 0;
    double eps_ = 1.0e-4;

    TranscriptionModes method_ = TranscriptionModes::LGL3;
    bool blocked_controls_ = false;
    bool has_ode_ = false;
    double order_ = 3.0;
    double error_weight_;

    int block_size_ = 0;
    int num_blocks_ = 0;
    int num_states_ = 0;
    mutable int last_block_accessed_ = 0;

    bool periodic_ = false;
    bool even_data_ = false;

    bool warn_out_of_bounds_ = true;
    bool throw_out_of_bounds_ = false;

    Eigen::VectorXd t_spacing_;
    Eigen::MatrixXd x_weights_;
    Eigen::MatrixXd dx_weights_;
    Eigen::MatrixXd u_weights_;
    VectorFunctionalX ode_;
    LGLInterpTable(VectorFunctionalX od, int xv, int uv, TranscriptionModes m) {
        this->x_vars_ = xv;
        this->u_vars_ = uv;
        this->axis_ = xv;
        this->xtu_vars_ = xv + uv + 1;
        this->ode_ = od;
        this->has_ode_ = true;
        this->set_method(m);
    }
    LGLInterpTable(VectorFunctionalX od, int xv, int uv, TranscriptionModes m,
                   const std::vector<Eigen::VectorXd> &xtudat, int dnum) {
        this->x_vars_ = xv;
        this->u_vars_ = uv;
        this->axis_ = xv;
        this->xtu_vars_ = xv + uv + 1;
        this->ode_ = od;
        this->has_ode_ = true;
        this->set_method(m);
        this->load_uneven_data(dnum, xtudat);
    }

    LGLInterpTable(VectorFunctionalX od, int xv, int uv, int pv, std::string m,
                   const std::vector<Eigen::VectorXd> &xtudat, int dnum) {
        this->x_vars_ = xv;
        this->u_vars_ = uv + pv;
        this->axis_ = xv;
        this->xtu_vars_ = xv + uv + 1 + pv;
        this->ode_ = od;
        this->has_ode_ = true;
        this->set_method(strto_TranscriptionMode(m));
        this->load_uneven_data(dnum, xtudat);
    }
    LGLInterpTable(VectorFunctionalX od, int xv, int uv, int pv,
                   const std::vector<Eigen::VectorXd> &xtudat)
        : LGLInterpTable(od, xv, uv, pv, "LGL3", xtudat, xtudat.size() - 1) {}
    LGLInterpTable(VectorFunctionalX od, int xv, int uv, const std::vector<Eigen::VectorXd> &xtudat)
        : LGLInterpTable(od, xv, uv, 0, "LGL3", xtudat, xtudat.size() - 1) {}

    LGLInterpTable(VectorFunctionalX od, int xv, int uv, std::string m,
                   const std::vector<Eigen::VectorXd> &xtudat, int dnum)
        : LGLInterpTable(od, xv, uv, 0, m, xtudat, dnum) {}

    LGLInterpTable(int xv, int uv, TranscriptionModes m, const std::vector<Eigen::VectorXd> &xtudat,
                   int dnum) {
        this->x_vars_ = xv;
        this->u_vars_ = uv;
        this->axis_ = xv;
        this->xtu_vars_ = xv + uv + 1;
        this->set_method(m);
        this->load_uneven_data(dnum, xtudat);
    }

    LGLInterpTable(int xv, const std::vector<Eigen::VectorXd> &xtudat, int dnum) {
        this->x_vars_ = xv;
        this->u_vars_ = 0;
        this->axis_ = xv;
        this->xtu_vars_ = xv + 0 + 1;
        this->set_method(TranscriptionModes::LGL3);
        this->load_uneven_data(dnum, xtudat);
    }
    LGLInterpTable(const std::vector<Eigen::VectorXd> &xtudat) {
        this->x_vars_ = xtudat[0].size() - 1;
        this->u_vars_ = 0;
        this->axis_ = xtudat[0].size() - 1;
        this->xtu_vars_ = xtudat[0].size();
        this->set_method(TranscriptionModes::LGL3);
        this->load_uneven_data(xtudat.size() - 1, xtudat);
    }
    LGLInterpTable(int xv, int uv, TranscriptionModes m) {
        this->x_vars_ = xv;
        this->u_vars_ = uv;
        this->axis_ = xv;
        this->xtu_vars_ = xv + uv + 1;
        this->set_method(m);
    }
    LGLInterpTable() {}
    std::shared_ptr<LGLInterpTable> get_table_ptr() {
        return std::shared_ptr<LGLInterpTable>(this);
    }
    void set_method(TranscriptionModes m);

    template <class VecType> void check_input(const std::vector<VecType> &xtudat) {

        double t0 = xtudat.front()[this->axis_];
        double tf = xtudat.back()[this->axis_];

        for (int i = 0; i < xtudat.size() - 1; i++) {
            double ti = xtudat[i][this->axis_];
            double tip1 = xtudat[i + 1][this->axis_];
            if (tip1 == ti) {
                throw std::invalid_argument(
                    fmt::format("Duplicate time coordinates in LGLInterpTable."));
            }

            if (((tip1 < ti) && tf > t0) || (((tip1 > ti) && tf < t0))) {
                throw std::invalid_argument(
                    fmt::format("Non monotonic time coordinates in LGLInterpTable."));
            }
        }
    }

    void make_periodic() {
        this->periodic_ = true;

        Eigen::VectorXd x0 = xtu_data_.col(0);
        Eigen::VectorXd xd0 = xdot_data_.col(0);
        Eigen::VectorXd xf = xtu_data_.col(xtu_data_.cols() - 1);

        double diff = (x0.head(this->x_vars_) - xf.head(this->x_vars_)).cwiseAbs().maxCoeff();

        if (diff > 1.0e-8) {
            std::cout << "Warning: Calling make_periodic on non-periodic_ table data" << std::endl;
        }
        xtu_data_.col(xtu_data_.cols() - 1).head(this->x_vars_) = x0.head(this->x_vars_);
        xdot_data_.col(xdot_data_.cols() - 1).head(this->x_vars_) = xd0.head(this->x_vars_);
    }
    void load_even_data(const std::vector<Eigen::VectorXd> &xtudat) {
        int msize = xtudat[0].size();
        if (msize != this->xtu_vars_) {
            std::cout << "User Input Error in supplying data to LGLInterpTable" << std::endl;
            std::cout << " Dimension of Input States(" << msize
                      << ") does not match expected dimensions of the Table(" << this->xtu_vars_
                      << ")" << std::endl;
            exit(1);
        }
        this->xtu_data_.resize(this->xtu_vars_, xtudat.size());
        this->xdot_data_.resize(this->x_vars_, xtudat.size());
        this->t0_ = xtudat[0][axis_];
        this->tf_ = xtudat.back()[axis_];
        this->total_t_ = xtudat.back()[axis_] - xtudat[0][axis_];
        this->num_states_ = xtudat.size();
        this->num_blocks_ = (this->num_states_ - 1) / (this->block_size_ - 1);
        this->delta_t_ = this->total_t_ / double(this->num_blocks_);
        this->even_data_ = true;
        this->last_block_accessed_ = 0;
        Eigen::VectorXd temp(this->x_vars_);

        for (int i = 0; i < this->num_states_; i++) {
            this->xtu_data_.col(i) = xtudat[i];
            if (this->has_ode_) {
                temp.setZero();
                this->ode_.compute(xtudat[i], temp);
                this->xdot_data_.col(i) = temp;
            }
        }

        if (!this->has_ode_) {
            FDDerivArbitrary<Eigen::VectorXd> dterp;
            dterp.set_axis(this->axis_);
            dterp.set_data(xtudat);
            std::vector<Eigen::VectorXd> datatmp;
            if (xtudat.size() > 8) {
                datatmp = dterp.deriv<Eigen::VectorXd>(1, 4);
            } else {
                datatmp = dterp.deriv<Eigen::VectorXd>(1, 1);
            }

            for (int i = 0; i < this->num_states_; i++) {
                this->xdot_data_.col(i) = datatmp[i].head(this->x_vars_);
            }
        }
    }

    void load_even_data2(const std::vector<Eigen::VectorXd> &xtudat,
                         const std::vector<Eigen::VectorXd> &xdotdat) {
        int msize = xtudat[0].size();
        if (msize != this->xtu_vars_) {
            std::cout << "User Input Error in supplying data to LGLInterpTable" << std::endl;
            std::cout << " Dimension of Input States(" << msize
                      << ") does not match expected dimensions of the Table(" << this->xtu_vars_
                      << ")" << std::endl;
            exit(1);
        }

        check_input(xtudat);

        this->xtu_data_.resize(this->xtu_vars_, xtudat.size());
        this->xdot_data_.resize(this->x_vars_, xtudat.size());
        this->t0_ = xtudat[0][axis_];
        this->tf_ = xtudat.back()[axis_];

        this->total_t_ = xtudat.back()[axis_] - xtudat[0][axis_];
        this->num_states_ = xtudat.size();
        this->num_blocks_ = (this->num_states_ - 1) / (this->block_size_ - 1);
        this->delta_t_ = this->total_t_ / double(this->num_blocks_);
        this->even_data_ = true;
        this->last_block_accessed_ = 0;
        Eigen::VectorXd temp(this->x_vars_);

        for (int i = 0; i < this->num_states_; i++) {
            this->xtu_data_.col(i) = xtudat[i];
            this->xdot_data_.col(i) = xdotdat[i];
        }
    }

    void load_uneven_data(int dnum, const std::vector<Eigen::VectorXd> &xtudat) {
        int msize = xtudat[0].size();
        if (msize != this->xtu_vars_) {
            std::cout << "User Input Error in supplying data to LGLInterpTable" << std::endl;
            std::cout << " Dimension of Input States(" << msize
                      << ") does not match expected dimensions of the Table(" << this->xtu_vars_
                      << ")" << std::endl;
            exit(1);
        }
        check_input(xtudat);

        Eigen::VectorXd myspace = this->t_spacing_;
        TranscriptionModes mymeth = this->method_;
        this->set_method(TranscriptionModes::LGL3);

        this->xtu_data_.resize(this->xtu_vars_, xtudat.size());
        this->xtu_data_.setZero();
        this->xdot_data_.resize(this->x_vars_, xtudat.size());
        this->xdot_data_.setZero();
        this->t0_ = xtudat[0][axis_];
        this->tf_ = xtudat.back()[axis_];

        this->total_t_ = xtudat.back()[axis_] - xtudat[0][axis_];
        this->num_states_ = xtudat.size();
        this->num_blocks_ = (this->num_states_ - 1) / (this->block_size_ - 1);
        this->even_data_ = false;
        this->last_block_accessed_ = 0;
        Eigen::VectorXd temp(this->x_vars_);

        for (int i = 0; i < this->num_states_; i++) {
            this->xtu_data_.col(i) = xtudat[i];
            if (this->has_ode_) {
                temp.setZero();
                this->ode_.compute(xtudat[i], temp);
                this->xdot_data_.col(i) = temp;
            }
        }

        if (!this->has_ode_) {
            FDDerivArbitrary<Eigen::VectorXd> dterp;
            dterp.set_axis(this->axis_);
            dterp.set_data(xtudat);
            std::vector<Eigen::VectorXd> datatmp;
            if (xtudat.size() > 8) {
                dterp.deriv<Eigen::VectorXd>(datatmp, 1, 4);
            } else {
                dterp.deriv<Eigen::VectorXd>(datatmp, 1, 1);
            }

            for (int i = 0; i < this->num_states_; i++) {
                this->xdot_data_.col(i) = datatmp[i].head(this->x_vars_);
            }
        }

        std::vector<Eigen::VectorXd> nxs = this->nd_equidist(myspace, dnum, 0.0, 1.0);
        std::vector<Eigen::VectorXd> ndxs(nxs.size());

        this->set_method(mymeth);
        this->load_even_data(nxs);
    }

    void load_regular_data(int dnum, const std::vector<Eigen::VectorXd> &xtudat) {

        check_input(xtudat);

        this->xtu_data_.resize(this->xtu_vars_, xtudat.size());
        this->xtu_data_.setZero();
        this->xdot_data_.resize(this->x_vars_, xtudat.size());
        this->xdot_data_.setZero();
        this->t0_ = xtudat[0][axis_];
        this->tf_ = xtudat.back()[axis_];

        this->total_t_ = xtudat.back()[axis_] - xtudat[0][axis_];
        this->num_states_ = xtudat.size();
        this->num_blocks_ = (this->num_states_ - 1) / (this->block_size_ - 1);
        this->even_data_ = false;
        this->last_block_accessed_ = 0;
        Eigen::VectorXd temp(this->x_vars_);
        for (int i = 0; i < this->num_states_; i++) {
            temp.setZero();
            this->ode_.compute(xtudat[i], temp);
            this->xtu_data_.col(i) = xtudat[i];
            this->xdot_data_.col(i) = temp;
        }
        std::vector<Eigen::VectorXd> nxs = this->nd_equidist(dnum, 0.0, 1.0);
        this->load_even_data(nxs);
    }
    void load_exact_data(const std::vector<Eigen::VectorXd> &xtudat) {

        check_input(xtudat);

        this->xtu_data_.resize(this->xtu_vars_, xtudat.size());
        this->xtu_data_.setZero();
        this->xdot_data_.resize(this->x_vars_, xtudat.size());
        this->xdot_data_.setZero();
        this->t0_ = xtudat[0][axis_];
        this->tf_ = xtudat.back()[axis_];

        this->total_t_ = xtudat.back()[axis_] - xtudat[0][axis_];
        this->num_states_ = xtudat.size();
        this->num_blocks_ = (this->num_states_ - 1) / (this->block_size_ - 1);
        this->even_data_ = false;
        this->last_block_accessed_ = 0;
        Eigen::VectorXd temp(this->x_vars_);
        for (int i = 0; i < this->num_states_; i++) {
            temp.setZero();
            this->ode_.compute(xtudat[i], temp);
            this->xtu_data_.col(i) = xtudat[i];
            this->xdot_data_.col(i) = temp;
        }
    }
    template <class V1, class V2>
    void load_exact_data(const std::vector<V1> &xtudat, const std::vector<V2> &xdotdat) {
        check_input(xtudat);
        this->xtu_data_.resize(this->xtu_vars_, xtudat.size());
        this->xtu_data_.setZero();
        this->xdot_data_.resize(this->x_vars_, xtudat.size());
        this->xdot_data_.setZero();
        this->t0_ = xtudat[0][axis_];
        this->tf_ = xtudat.back()[axis_];

        this->total_t_ = xtudat.back()[axis_] - xtudat[0][axis_];
        this->num_states_ = xtudat.size();
        this->num_blocks_ = (this->num_states_ - 1) / (this->block_size_ - 1);
        this->even_data_ = false;
        this->last_block_accessed_ = 0;
        for (int i = 0; i < this->num_states_; i++) {
            this->xtu_data_.col(i) = xtudat[i];
            this->xdot_data_.col(i) = xdotdat[i];
        }
    }

    std::vector<Eigen::VectorXd> nd_equidist(Eigen::VectorXd spacing, int dnum, double low,
                                             double high) { // 0 to 1;
        Eigen::VectorXd node_space;
        node_space.setLinSpaced(dnum + 1, low, high);
        Eigen::VectorXd all_space(dnum * (spacing.size() - 1) + 1);
        for (int i = 0; i < dnum; i++) {
            all_space.segment(i * (spacing.size() - 1), spacing.size()) =
                Eigen::VectorXd::Constant(spacing.size(), node_space[i]) +
                spacing * (node_space[i + 1] - node_space[i]);
        }
        all_space *= this->total_t_;
        all_space += Eigen::VectorXd::Constant(all_space.size(), this->t0_);
        std::vector<Eigen::VectorXd> mesh(all_space.size());
        for (int i = 0; i < all_space.size(); i++) {
            mesh[i] = this->interpolate(all_space[i]);
        }
        return mesh;
    }
    std::vector<Eigen::VectorXd> nd_equidist(int dnum, double low, double high) {
        return this->nd_equidist(this->t_spacing_, dnum, low, high);
    }
    std::vector<Eigen::VectorXd> nd_distribute(Eigen::VectorXd ndspacing, Eigen::VectorXd ndtimes,
                                               Eigen::VectorXi defper) { // 0-1
        std::vector<Eigen::VectorXd> mesh;
        for (int i = 0; i < defper.size(); i++) {
            std::vector<Eigen::VectorXd> submesh =
                this->nd_equidist(ndspacing, defper[i], ndtimes[i], ndtimes[i + 1]);
            int jstart = 1;
            if (i == 0)
                jstart = 0;
            for (int j = jstart; j < (submesh.size()); j++) {
                mesh.push_back(submesh[j]);
            }
        }
        return mesh;
    }
    std::vector<Eigen::VectorXd> nd_distribute(Eigen::VectorXd ndtimes, Eigen::VectorXi defper) {
        return this->nd_distribute(this->t_spacing_, ndtimes, defper);
    }

    std::vector<Eigen::VectorXd> interp_range(int dnum, double tlow, double thig) {
        double frac1 = (tlow - this->t0_) / this->total_t_;
        double frac2 = (thig - this->t0_) / this->total_t_;
        return this->nd_equidist(dnum, frac1, frac2);
    }
    std::vector<Eigen::VectorXd> interp_whole_range(int dnum) {
        return this->nd_equidist(dnum, 0.0, 1.0);
    }

    std::vector<Eigen::VectorXd> error_integral(int num_samps) {
        Eigen::VectorXd ts;
        ts.setLinSpaced(num_samps, this->t0_, this->tf_);
        std::vector<Eigen::VectorXd> errint(ts.size(), Eigen::VectorXd(2));
        errint[0][0] = 0.0;
        errint[0][1] = ts[0];

        Eigen::Matrix<double, -1, 2> XXd;
        Eigen::VectorXd temp(this->x_vars_);
        Eigen::VectorXd temp2(this->x_vars_);

        double h = ts[1] - ts[0];
        for (int i = 1; i < ts.size(); i++) {
            XXd = this->interpolate_deriv(ts[i]);
            temp.setZero();
            temp2 = XXd.col(0);
            this->ode_.compute(temp2, temp);
            errint[i][0] =
                errint[i - 1][0] +
                std::pow((temp - XXd.col(1).head(this->x_vars_)).norm(), 1.0 / (this->order_ + 1)) *
                    h;
            errint[i][1] = ts[i];
        }
        return errint;
    }

    std::vector<Eigen::VectorXd> new_error_integral();

    void deboor_mesh_error(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                           Eigen::MatrixXd &mesh_dist) const;

    template <class Scalar> VectorX<Scalar> interpolate(Scalar tglobal) const {
        VectorX<Scalar> fx(this->xtu_vars_);
        fx.setZero();
        this->interpolate_ref(tglobal, fx);
        return fx;
    }
    template <class Scalar> Eigen::Matrix<Scalar, -1, 2> interpolate_deriv(Scalar tglobal) const {
        Eigen::Matrix<Scalar, -1, 2> fx(this->xtu_vars_, 2);
        fx.setZero();
        this->interpolate_deriv_ref(tglobal, fx);
        return fx;
    }
    template <class Scalar> void find_block(Scalar tglobal, Scalar &tnd, int &element) const {
        if (this->even_data_) {
            Scalar tlocal = tglobal - this->t0_;

            if (this->periodic_) {

                Scalar frac = tlocal / this->total_t_;
                if (frac > 1.0) {
                    tlocal -= int(frac) * this->total_t_;
                } else if (frac < 0.0) {
                    tlocal += (int(std::abs(frac)) + 1) * this->total_t_;
                }
            }

            element = int((tlocal / this->delta_t_));
            if (element < 0)
                element = 0;
            element = std::min(element, this->num_blocks_ - 1);
            double root = double(element) * (this->delta_t_);
            Scalar remainder = tlocal - root;
            tnd = remainder / this->delta_t_;
            return;
        }
        element = this->last_block_accessed_;
        int sd = 0;
        do {
            sd = this->check_ith_block(tglobal, element);

            element += sd;
            if (element < 0) {
                element = 0;
                break;
            } else if (element == this->num_blocks_) {
                element = this->num_blocks_ - 1;
                break;
            }
        } while (sd != 0);
        this->last_block_accessed_ = element;
        double tb0 = this->xtu_data_.middleCols((this->block_size_ - 1) * element,
                                                this->block_size_)(axis_, 0);
        double tbf = this->xtu_data_.middleCols((this->block_size_ - 1) * element,
                                                this->block_size_)(axis_, this->block_size_ - 1);
        tnd = (tglobal - tb0) / (tbf - tb0);
    }

    template <class Scalar, class OutType>
    void interpolate_ref(Scalar tglobal, const Eigen::MatrixBase<OutType> &fx) const {
        int element = 0;
        Scalar tnd = 0;
        this->find_block(tglobal, tnd, element);

        return interp_ith_block(tnd, fx, element);
    }
    template <class Scalar, class OutType>
    void interpolate_deriv_ref(Scalar tglobal, const Eigen::MatrixBase<OutType> &fx) const {
        int element = 0;
        Scalar tnd = 0;
        this->find_block(tglobal, tnd, element);
        return interp_ith_block_deriv(tnd, fx, element);
    }
    template <class Scalar, class OutType>
    void interpolate_2nd_deriv_ref(Scalar tglobal, const Eigen::MatrixBase<OutType> &fx) const {
        int element = 0;
        Scalar tnd = 0;
        this->find_block(tglobal, tnd, element);
        return interp_ith_block_2nd_deriv(tnd, fx, element);
    }

    template <class Scalar, class OutType>
    void interp_ith_block(Scalar t, const Eigen::MatrixBase<OutType> &fx, int i) const {
        return this->interp_block(
            t, fx, this->xtu_data_.middleCols((this->block_size_ - 1) * i, this->block_size_),
            this->xdot_data_.middleCols((this->block_size_ - 1) * i, this->block_size_));
    }
    template <class Scalar, class OutType>
    void interp_ith_block_deriv(Scalar t, const Eigen::MatrixBase<OutType> &fx, int i) const {
        return this->interp_block_deriv(
            t, fx, this->xtu_data_.middleCols((this->block_size_ - 1) * i, this->block_size_),
            this->xdot_data_.middleCols((this->block_size_ - 1) * i, this->block_size_));
    }
    template <class Scalar, class OutType>
    void interp_ith_block_2nd_deriv(Scalar t, const Eigen::MatrixBase<OutType> &fx, int i) const {
        return this->interp_block_2nd_deriv(
            t, fx, this->xtu_data_.middleCols((this->block_size_ - 1) * i, this->block_size_),
            this->xdot_data_.middleCols((this->block_size_ - 1) * i, this->block_size_));
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_gen(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                          const Eigen::MatrixBase<XtUBlockType> &xtublk,
                          const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        VectorX<Scalar> tpow(x_weights_.rows());
        tpow[0] = Scalar(1.0);
        for (int i = 1; i < x_weights_.rows(); i++) {
            tpow[i] = tpow[i - 1] * t;
        }

        Eigen::MatrixBase<OutType> &xtu_n = fx.const_cast_derived();
        Scalar t0 = xtublk(axis_, 0);
        Scalar tf = xtublk(axis_, this->block_size_ - 1);
        Scalar h = tf - t0;

        for (int i = 0; i < this->block_size_; i++) {
            Scalar xsc = tpow.dot(this->x_weights_.col(i).template cast<Scalar>());
            Scalar dxsc = tpow.dot(this->dx_weights_.col(i).template cast<Scalar>());
            xtu_n.head(this->x_vars_) +=
                xtublk.col(i).head(this->x_vars_).template cast<Scalar>() * xsc +
                dxblk.col(i).head(this->x_vars_).template cast<Scalar>() * dxsc * h;
        }
        xtu_n[axis_] = t0 + h * t;

        if (this->u_vars_ > 0) {
            if (this->blocked_controls_) {
                xtu_n.tail(this->u_vars_) =
                    xtublk.col(0).tail(this->u_vars_).template cast<Scalar>();
            } else {
                VectorX<Scalar> utpow = tpow.head(u_weights_.rows());
                for (int i = 0; i < this->block_size_; i++) {
                    Scalar usc = utpow.dot(this->u_weights_.col(i).template cast<Scalar>());
                    xtu_n.tail(this->u_vars_) +=
                        xtublk.col(i).tail(this->u_vars_).template cast<Scalar>() * usc;
                }
            }
        }
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_deriv_gen(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                                const Eigen::MatrixBase<XtUBlockType> &xtublk,
                                const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        VectorX<Scalar> tpow(x_weights_.rows());
        tpow[0] = Scalar(1.0);
        VectorX<Scalar> tpow2(x_weights_.rows());
        tpow2[0] = Scalar(0.0);
        tpow2[1] = Scalar(1.0);
        for (int i = 1; i < x_weights_.rows(); i++) {
            tpow[i] = tpow[i - 1] * t;
        }
        for (int i = 2; i < x_weights_.rows(); i++) {
            tpow2[i] = tpow[i - 1] * Scalar(i);
        }

        Eigen::MatrixBase<OutType> &xtu_n = fx.const_cast_derived();
        Scalar t0 = xtublk(axis_, 0);
        Scalar tf = xtublk(axis_, this->block_size_ - 1);
        Scalar h = tf - t0;

        for (int i = 0; i < this->block_size_; i++) {
            Scalar xsc = tpow.dot(this->x_weights_.col(i).template cast<Scalar>());
            Scalar dxsc = tpow.dot(this->dx_weights_.col(i).template cast<Scalar>());
            Scalar xsc_dt = tpow2.dot(this->x_weights_.col(i).template cast<Scalar>());
            Scalar dxsc_dt = tpow2.dot(this->dx_weights_.col(i).template cast<Scalar>());
            xtu_n.col(0).head(this->x_vars_) +=
                xtublk.col(i).head(this->x_vars_).template cast<Scalar>() * xsc +
                dxblk.col(i).head(this->x_vars_).template cast<Scalar>() * dxsc * h;
            xtu_n.col(1).head(this->x_vars_) +=
                xtublk.col(i).head(this->x_vars_).template cast<Scalar>() * xsc_dt / h +
                dxblk.col(i).head(this->x_vars_).template cast<Scalar>() * dxsc_dt;
        }
        xtu_n.col(0)[axis_] = t0 + h * t;
        xtu_n.col(1)[axis_] = 1.0;

        if (this->u_vars_ > 0) {
            VectorX<Scalar> utpow = tpow.head(u_weights_.rows());
            VectorX<Scalar> utpow2 = tpow2.head(u_weights_.rows());

            for (int i = 0; i < this->block_size_; i++) {
                Scalar usc = utpow.dot(this->u_weights_.col(i).template cast<Scalar>());
                Scalar usc_dt = utpow2.dot(this->u_weights_.col(i).template cast<Scalar>());

                xtu_n.col(0).tail(this->u_vars_) +=
                    xtublk.col(i).tail(this->u_vars_).template cast<Scalar>() * usc;
                xtu_n.col(1).tail(this->u_vars_) +=
                    xtublk.col(i).tail(this->u_vars_).template cast<Scalar>() * usc_dt / h;
            }
        }
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_deriv2_gen(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                                 const Eigen::MatrixBase<XtUBlockType> &xtublk,
                                 const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        VectorX<Scalar> tpow(x_weights_.rows());
        tpow[0] = Scalar(1.0);
        VectorX<Scalar> tpow2(x_weights_.rows());
        tpow2[0] = Scalar(0.0);
        tpow2[1] = Scalar(1.0);
        for (int i = 1; i < x_weights_.rows(); i++) {
            tpow[i] = tpow[i - 1] * t;
        }
        for (int i = 2; i < x_weights_.rows(); i++) {
            tpow2[i] = tpow[i - 1] * Scalar(i);
        }

        Eigen::MatrixBase<OutType> &xtu_n = fx.const_cast_derived();
        Scalar t0 = xtublk(axis_, 0);
        Scalar tf = xtublk(axis_, this->block_size_ - 1);
        Scalar h = tf - t0;

        for (int i = 0; i < this->block_size_; i++) {
            Scalar xsc = tpow.dot(this->x_weights_.col(i).template cast<Scalar>());
            Scalar dxsc = tpow.dot(this->dx_weights_.col(i).template cast<Scalar>());
            Scalar xsc_dt = tpow2.dot(this->x_weights_.col(i).template cast<Scalar>());
            Scalar dxsc_dt = tpow2.dot(this->dx_weights_.col(i).template cast<Scalar>());
            xtu_n.col(0).head(this->x_vars_) += xtublk.col(i).head(this->x_vars_) * xsc +
                                                dxblk.col(i).head(this->x_vars_) * dxsc * h;
            xtu_n.col(1).head(this->x_vars_) += xtublk.col(i).head(this->x_vars_) * xsc_dt / h +
                                                dxblk.col(i).head(this->x_vars_) * dxsc_dt;
        }
        xtu_n.col(0)[axis_] = t0 + h * t;
        xtu_n.col(1)[axis_] = 1.0;

        if (this->u_vars_ > 0) {
            VectorX<Scalar> utpow = tpow.head(u_weights_.rows());
            VectorX<Scalar> utpow2 = tpow2.head(u_weights_.rows());

            for (int i = 0; i < this->block_size_; i++) {
                Scalar usc = utpow.dot(this->u_weights_.col(i).template cast<Scalar>());
                Scalar usc_dt = utpow2.dot(this->u_weights_.col(i).template cast<Scalar>());

                xtu_n.col(0).tail(this->u_vars_) +=
                    xtublk.col(i).tail(this->u_vars_).template cast<Scalar>() * usc;
                xtu_n.col(1).tail(this->u_vars_) +=
                    xtublk.col(i).tail(this->u_vars_).template cast<Scalar>() * usc_dt / h;
            }
        }
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_lgl3(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                           const Eigen::MatrixBase<XtUBlockType> &xtublk,
                           const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        Eigen::MatrixBase<OutType> &xtu_n = fx.const_cast_derived();
        Scalar t0 = xtublk(axis_, 0);
        Scalar tf = xtublk(axis_, this->block_size_ - 1);
        Scalar h = tf - t0;
        Scalar t2 = t * t;
        Scalar t3 = t2 * t;
        Scalar xsc0 = (2.0 * t3 - 3.0 * t2 + 1.0);
        Scalar dxsc0 = (t3 - 2.0 * t2 + t) * h;
        Scalar xsc1 = (-2.0 * t3 + 3.0 * t2);
        Scalar dxsc1 = (t3 - t2) * h;

        xtu_n.head(this->x_vars_) =
            xtublk.col(0).head(this->x_vars_).template cast<Scalar>() * xsc0 +
            dxblk.col(0).head(this->x_vars_).template cast<Scalar>() * dxsc0 +
            xtublk.col(1).head(this->x_vars_).template cast<Scalar>() * xsc1 +
            dxblk.col(1).head(this->x_vars_).template cast<Scalar>() * dxsc1;
        xtu_n[axis_] = t0 + h * t;

        if (this->u_vars_ > 0) {

            if (this->blocked_controls_) {
                xtu_n.tail(this->u_vars_) =
                    xtublk.col(0).tail(this->u_vars_).template cast<Scalar>();
            } else {
                Scalar usc0 = 1.0 - t;
                Scalar usc1 = t;
                xtu_n.tail(this->u_vars_) =
                    xtublk.col(0).tail(this->u_vars_).template cast<Scalar>() * usc0 +
                    xtublk.col(1).tail(this->u_vars_).template cast<Scalar>() * usc1;
            }
        }
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_deriv_lgl3(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                                 const Eigen::MatrixBase<XtUBlockType> &xtublk,
                                 const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        Eigen::MatrixBase<OutType> &xtu_n = fx.const_cast_derived();
        Scalar t0 = xtublk(axis_, 0);
        Scalar tf = xtublk(axis_, this->block_size_ - 1);
        Scalar h = tf - t0;
        Scalar t2 = t * t;
        Scalar t3 = t2 * t;

        Scalar xsc0 = (2.0 * t3 - 3.0 * t2 + 1.0);
        Scalar dxsc0 = (t3 - 2.0 * t2 + t) * h;
        Scalar xsc1 = (-2.0 * t3 + 3.0 * t2);
        Scalar dxsc1 = (t3 - t2) * h;

        Scalar xsc0_dt = (6.0 * t2 - 6.0 * t) / h;
        Scalar dxsc0_dt = (3.0 * t2 - 4.0 * t + 1.0);
        Scalar xsc1_dt = (-6.0 * t2 + 6.0 * t) / h;
        Scalar dxsc1_dt = (3.0 * t2 - 2.0 * t);

        if constexpr (std::is_same<Scalar, double>::value) {
            xtu_n.col(0).head(this->x_vars_) = xtublk.col(0).head(this->x_vars_) * xsc0 +
                                               dxblk.col(0).head(this->x_vars_) * dxsc0 +
                                               xtublk.col(1).head(this->x_vars_) * xsc1 +
                                               dxblk.col(1).head(this->x_vars_) * dxsc1;
            xtu_n.col(1).head(this->x_vars_) = xtublk.col(0).head(this->x_vars_) * xsc0_dt +
                                               dxblk.col(0).head(this->x_vars_) * dxsc0_dt +
                                               xtublk.col(1).head(this->x_vars_) * xsc1_dt +
                                               dxblk.col(1).head(this->x_vars_) * dxsc1_dt;
        } else {
            xtu_n.col(0).head(this->x_vars_) =
                xtublk.col(0).head(this->x_vars_).template cast<Scalar>() * xsc0 +
                dxblk.col(0).head(this->x_vars_).template cast<Scalar>() * dxsc0 +
                xtublk.col(1).head(this->x_vars_).template cast<Scalar>() * xsc1 +
                dxblk.col(1).head(this->x_vars_).template cast<Scalar>() * dxsc1;
            xtu_n.col(1).head(this->x_vars_) =
                xtublk.col(0).head(this->x_vars_).template cast<Scalar>() * xsc0_dt +
                dxblk.col(0).head(this->x_vars_).template cast<Scalar>() * dxsc0_dt +
                xtublk.col(1).head(this->x_vars_).template cast<Scalar>() * xsc1_dt +
                dxblk.col(1).head(this->x_vars_).template cast<Scalar>() * dxsc1_dt;
        }

        xtu_n.col(0)[axis_] = t0 + h * t;
        xtu_n.col(1)[axis_] = 1;

        if (this->u_vars_ > 0) {
            Scalar usc0 = 1.0 - t;
            Scalar usc1 = t;
            Scalar usc0_dt = -1.0 / h;
            Scalar usc1_dt = 1.0 / h;
            xtu_n.col(0).tail(this->u_vars_) =
                xtublk.col(0).tail(this->u_vars_).template cast<Scalar>() * usc0 +
                xtublk.col(1).tail(this->u_vars_).template cast<Scalar>() * usc1;
            xtu_n.col(1).tail(this->u_vars_) =
                xtublk.col(0).tail(this->u_vars_).template cast<Scalar>() * usc0_dt +
                xtublk.col(1).tail(this->u_vars_).template cast<Scalar>() * usc1_dt;
        }
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_deriv2_lgl3(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                                  const Eigen::MatrixBase<XtUBlockType> &xtublk,
                                  const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        Eigen::MatrixBase<OutType> &xtu_n = fx.const_cast_derived();
        Scalar t0 = xtublk(axis_, 0);
        Scalar tf = xtublk(axis_, this->block_size_ - 1);
        Scalar h = tf - t0;
        Scalar t2 = t * t;
        Scalar t3 = t2 * t;

        Scalar xsc0 = (2.0 * t3 - 3.0 * t2 + 1.0);
        Scalar dxsc0 = (t3 - 2.0 * t2 + t) * h;
        Scalar xsc1 = (-2.0 * t3 + 3.0 * t2);
        Scalar dxsc1 = (t3 - t2) * h;

        Scalar xsc0_dt = (6.0 * t2 - 6.0 * t) / h;
        Scalar dxsc0_dt = (3.0 * t2 - 4.0 * t + 1.0);
        Scalar xsc1_dt = (-6.0 * t2 + 6.0 * t) / h;
        Scalar dxsc1_dt = (3.0 * t2 - 2.0 * t);

        Scalar xsc0_dt2 = (12.0 * t - 6.0) / (h * h);
        Scalar dxsc0_dt2 = (6.0 * t - 4.0) / h;
        Scalar xsc1_dt2 = (-12.0 * t + 6.0) / (h * h);
        Scalar dxsc1_dt2 = (6.0 * t - 2.0) / h;

        xtu_n.col(0).head(this->x_vars_) =
            xtublk.col(0).head(this->x_vars_) * xsc0 + dxblk.col(0).head(this->x_vars_) * dxsc0 +
            xtublk.col(1).head(this->x_vars_) * xsc1 + dxblk.col(1).head(this->x_vars_) * dxsc1;
        xtu_n.col(1).head(this->x_vars_) = xtublk.col(0).head(this->x_vars_) * xsc0_dt +
                                           dxblk.col(0).head(this->x_vars_) * dxsc0_dt +
                                           xtublk.col(1).head(this->x_vars_) * xsc1_dt +
                                           dxblk.col(1).head(this->x_vars_) * dxsc1_dt;
        xtu_n.col(2).head(this->x_vars_) = xtublk.col(0).head(this->x_vars_) * xsc0_dt2 +
                                           dxblk.col(0).head(this->x_vars_) * dxsc0_dt2 +
                                           xtublk.col(1).head(this->x_vars_) * xsc1_dt2 +
                                           dxblk.col(1).head(this->x_vars_) * dxsc1_dt2;

        xtu_n.col(0)[axis_] = t0 + h * t;
        xtu_n.col(1)[axis_] = 1;
        xtu_n.col(2)[axis_] = 0;

        if (this->u_vars_ > 0) {
            Scalar usc0 = 1.0 - t;
            Scalar usc1 = t;
            Scalar usc0_dt = -1.0 / h;
            Scalar usc1_dt = 1.0 / h;
            xtu_n.col(0).tail(this->u_vars_) =
                xtublk.col(0).tail(this->u_vars_) * usc0 + xtublk.col(1).tail(this->u_vars_) * usc1;
            xtu_n.col(1).tail(this->u_vars_) = xtublk.col(0).tail(this->u_vars_) * usc0_dt +
                                               xtublk.col(1).tail(this->u_vars_) * usc1_dt;
            xtu_n.col(2).tail(this->u_vars_).setZero();
        }
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                      const Eigen::MatrixBase<XtUBlockType> &xtublk,
                      const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        if (this->method_ == TranscriptionModes::LGL3 ||
            this->method_ == TranscriptionModes::Trapezoidal) {
            return this->interp_block_lgl3(t, fx, xtublk, dxblk);
        }
        return this->interp_block_gen(t, fx, xtublk, dxblk);
    }
    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_deriv(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                            const Eigen::MatrixBase<XtUBlockType> &xtublk,
                            const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        if (this->method_ == TranscriptionModes::LGL3 ||
            this->method_ == TranscriptionModes::Trapezoidal) {
            return this->interp_block_deriv_lgl3(t, fx, xtublk, dxblk);
        }
        return this->interp_block_deriv_gen(t, fx, xtublk, dxblk);
    }

    template <class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void interp_block_2nd_deriv(Scalar t, const Eigen::MatrixBase<OutType> &fx,
                                const Eigen::MatrixBase<XtUBlockType> &xtublk,
                                const Eigen::MatrixBase<DXBlockType> &dxblk) const {
        if (this->method_ == TranscriptionModes::LGL3) {
            return this->interp_block_deriv2_lgl3(t, fx, xtublk, dxblk);
        } else {
            throw std::invalid_argument("Implement LGL Table 2ndDerives");
        }
    }

    template <class Scalar> int check_ith_block(Scalar tglob, int i) const {
        Scalar t0 =
            this->xtu_data_.middleCols((this->block_size_ - 1) * i, this->block_size_)(axis_, 0);
        Scalar tf = this->xtu_data_.middleCols((this->block_size_ - 1) * i,
                                               this->block_size_)(axis_, this->block_size_ - 1);

        int sd = 0;

        if (tf > t0) {
            if (t0 <= tglob && tf >= tglob)
                sd = 0;
            else if (tglob > tf)
                sd = 1;
            else if (tglob < t0)
                sd = -1;
        } else {
            if (t0 >= tglob && tf <= tglob)
                sd = 0;
            else if (tglob < tf)
                sd = 1;
            else if (tglob > t0)
                sd = -1;
        }

        return sd;
    }
};

} // namespace tycho::oc
