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
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

struct InterpTable1D {

    enum class InterpType { cubic_interp, linear_interp };

    using MatType = Eigen::Matrix<double, -1, -1>;

    Eigen::VectorXd ts_;
    MatType vs_;
    MatType dvs_dts_;

    InterpType interp_kind_ = InterpType::cubic_interp;
    bool teven_ = true;
    int axis_ = 0;
    int tsize_;
    double ttotal_;
    int vlen_;
    bool warn_out_of_bounds_ = true;
    bool throw_out_of_bounds_ = false;

    InterpTable1D() {}

    InterpTable1D(const Eigen::VectorXd &Ts, const MatType &Vs, int axis, std::string kind) {
        set_data(Ts, Vs, axis, kind);
    }
    InterpTable1D(const Eigen::VectorXd &Ts, const Eigen::VectorXd &Vs, int axis,
                  std::string kind) {
        MatType Vstmp = Vs.transpose();
        set_data(Ts, Vstmp, 1, kind);
    }
    InterpTable1D(const std::vector<Eigen::VectorXd> &Vts, int tvar, std::string kind) {

        if (Vts.size() == 0) {
            throw std::invalid_argument("Input is empty");
        }
        if (Vts[0].size() < 2) {
            throw std::invalid_argument("Invalid sized value-time data.");
        }

        if (tvar < 0) {
            tvar = Vts[0].size() + tvar;
        }
        if (tvar > Vts[0].size() - 1 || tvar < 0) {
            throw std::invalid_argument("Invalid time variable index");
        }

        Eigen::VectorXd Ts(Vts.size());

        Eigen::MatrixXd Vs(Vts[0].size() - 1, Vts.size());

        for (int i = 0; i < Vts.size(); i++) {
            int isize = Vts[i].size();
            if (isize != Vts[0].size()) {
                throw std::invalid_argument("All value-time vectors must have same size");
            }
            int shift = 0;
            for (int j = 0; j < isize; j++) {
                if (j == tvar) {
                    Ts[i] = Vts[i][j];
                    shift = 1;
                } else {
                    Vs.col(i)[j - shift] = Vts[i][j];
                }
            }
        }
        set_data(Ts, Vs, 1, kind);
    }
    void set_data(const Eigen::VectorXd &Ts, const MatType &Vs, int axis, std::string kind) {

        this->ts_ = Ts;

        if (axis == 1) {
            this->axis_ = 1;
            this->vs_ = Vs;
        } else if (axis == 0) {
            this->axis_ = 0;
            this->vs_ = Vs.transpose();
        } else {
            throw std::invalid_argument("Interpolation axis must be 0 or 1");
        }

        if (kind == "cubic" || kind == "Cubic") {
            this->interp_kind_ = InterpType::cubic_interp;
        } else if (kind == "linear" || kind == "Linear") {
            this->interp_kind_ = InterpType::linear_interp;
        } else {
            throw std::invalid_argument("Unrecognized interpolation type");
        }

        tsize_ = ts_.size();
        vlen_ = vs_.rows();
        ttotal_ = ts_[tsize_ - 1] - ts_[0];

        if (tsize_ < 5) {
            throw std::invalid_argument("t coordinates must be larger than 4");
        }
        if (tsize_ != vs_.cols()) {
            throw std::invalid_argument(
                "Length of t coordinates must match length of interpolation axis");
        }
        for (int i = 0; i < tsize_ - 1; i++) {
            if (ts_[i + 1] < ts_[i]) {
                throw std::invalid_argument("t Coordinates must be in ascending order");
            }
        }

        Eigen::VectorXd testt;
        testt.setLinSpaced(ts_.size(), ts_[0], ts_[tsize_ - 1]);

        double terr = (ts_ - testt).lpNorm<Eigen::Infinity>();

        if (terr > abs(ttotal_) * 1.0e-12) {
            this->teven_ = false;
        }

        if (this->interp_kind_ == InterpType::cubic_interp)
            calc_derivs();
    }

    void calc_derivs() {

        this->dvs_dts_.resize(this->vlen_, this->tsize_);

        Eigen::Matrix<double, 5, 5> stens;
        stens.row(0).setOnes();
        Eigen::Matrix<double, 5, 1> rhs;
        rhs << 0, 1, 0, 0, 0;
        Eigen::Matrix<double, 5, 1> times;
        Eigen::Matrix<double, 5, 1> coeffs;

        bool hitcent = false;
        for (int i = 0; i < this->tsize_; i++) {
            int start = 0;
            bool recalc = true;
            if (i + 2 <= this->tsize_ - 1 && i - 2 >= 0) {
                // central difference
                if (this->teven_ && hitcent) {
                    recalc = false;
                }
                hitcent = true;
                start = i - 2;
            } else if (i < this->tsize_ - 1 - i) {
                // forward difference
                start = 0;
            } else {
                // backward difference
                start = this->tsize_ - 5;
            }
            int stepdir = (i < this->tsize_ - 1) ? 1 : -1;
            double ti = this->ts_[i];
            double tstep = std::abs(this->ts_[i + stepdir] - ti);
            if (recalc) {
                times = this->ts_.segment(start, 5);
                times -= Eigen::Matrix<double, 5, 1>::Constant(ti);
                times /= tstep;
                stens.row(1) = times.transpose();
                stens.row(2) = stens.row(1).cwiseProduct(times.transpose());
                stens.row(3) = stens.row(2).cwiseProduct(times.transpose());
                stens.row(4) = stens.row(3).cwiseProduct(times.transpose());
                coeffs = stens.inverse() * rhs;
            }

            dvs_dts_.col(i) = this->vs_.middleCols(start, 5) * (coeffs / tstep);
        }
    }

    int get_telem(double t) const {
        int telem;
        if (this->teven_) {
            double tlocal = t - ts_[0];
            double tstep = ts_[1] - ts_[0];
            telem = std::min(int(tlocal / tstep), this->tsize_ - 2);
        } else {
            int center = int(ts_.size() / 2);
            int shift = (ts_[center] > t) ? 0 : center;
            auto it = std::upper_bound(ts_.cbegin(), ts_.cend(), t);
            telem = int(it - ts_.begin()) - 1;
        }

        telem = std::min(telem, this->tsize_ - 2);
        telem = std::max(telem, 0);
        return telem;
    }

    template <class VType>
    void interp_impl(double t, int deriv, VType &v, VType &dv_dt, VType &dv2_dt2) const {

        if (warn_out_of_bounds_ || throw_out_of_bounds_) {
            double eps = std::numeric_limits<double>::epsilon() * ttotal_;
            if (t < (ts_[0] - eps) || t > (ts_[ts_.size() - 1] + eps)) {
                fmt::print(fmt::fg(fmt::color::red),
                           "WARNING: t= {0:} falls outside of InterpTable1D time range. Data is "
                           "being extrapolated!!\n",
                           t);
                if (throw_out_of_bounds_) {
                    throw std::invalid_argument("");
                }
            }
        }

        double telem = this->get_telem(t);
        double tstep = ts_[telem + 1] - ts_[telem];
        double tnd = (t - ts_[telem]) / tstep;

        if (this->interp_kind_ == InterpType::cubic_interp) {

            double tnd2 = tnd * tnd;
            double tnd3 = tnd2 * tnd;

            double p0 = (2.0 * tnd3 - 3.0 * tnd2 + 1.0);
            double m0 = (tnd3 - 2.0 * tnd2 + tnd) * tstep;
            double p1 = (-2.0 * tnd3 + 3.0 * tnd2);
            double m1 = (tnd3 - tnd2) * tstep;

            v = vs_.col(telem) * p0 + vs_.col(telem + 1) * p1 + dvs_dts_.col(telem) * m0 +
                dvs_dts_.col(telem + 1) * m1;

            if (deriv > 0) {

                double p0_dt = (6.0 * tnd2 - 6.0 * tnd) / tstep;
                double m0_dt = (3.0 * tnd2 - 4.0 * tnd + 1.0);
                double p1_dt = (-6.0 * tnd2 + 6.0 * tnd) / tstep;
                double m1_dt = (3.0 * tnd2 - 2.0 * tnd);

                dv_dt = vs_.col(telem) * p0_dt + vs_.col(telem + 1) * p1_dt +
                        dvs_dts_.col(telem) * m0_dt + dvs_dts_.col(telem + 1) * m1_dt;

                if (deriv > 1) {

                    double p0_dt2 = (12.0 * tnd - 6.0) / (tstep * tstep);
                    double m0_dt2 = (6.0 * tnd - 4.0) / tstep;
                    double p1_dt2 = (-12.0 * tnd + 6.0) / (tstep * tstep);
                    double m1_dt2 = (6.0 * tnd - 2.0) / tstep;

                    dv2_dt2 = vs_.col(telem) * p0_dt2 + vs_.col(telem + 1) * p1_dt2 +
                              dvs_dts_.col(telem) * m0_dt2 + dvs_dts_.col(telem + 1) * m1_dt2;
                }
            }

        } else {
            v = vs_.col(telem) * (1 - tnd) + vs_.col(telem + 1) * tnd;
            if (deriv > 0) {
                dv_dt = (vs_.col(telem + 1) - vs_.col(telem)) / tstep;
                if (deriv > 1) {
                    // Zero
                }
            }
        }
    }

    Eigen::VectorXd interp(double t) const {

        Eigen::VectorXd v;
        v.resize(vlen_);
        interp_impl(t, 0, v, v, v);
        return v;
    }

    Eigen::MatrixXd interp(const Eigen::VectorXd &t_vals) const {

        Eigen::MatrixXd v_out;
        v_out.resize(vlen_, t_vals.size());
        Eigen::VectorXd v;
        v.resize(vlen_);

        for (int i = 0; i < t_vals.size(); i++) {
            interp_impl(t_vals[i], 0, v, v, v);
            v_out.col(i) = v;
            v.setZero();
        }

        return v_out;
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXd> interp_deriv1(double t) const {

        Eigen::VectorXd v;
        v.resize(vlen_);
        Eigen::VectorXd dv_dt;
        dv_dt.resize(vlen_);

        interp_impl(t, 1, v, dv_dt, dv_dt);

        return std::tuple{v, dv_dt};
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> interp_deriv2(double t) const {

        Eigen::VectorXd v;
        v.resize(vlen_);
        Eigen::VectorXd dv_dt;
        dv_dt.resize(vlen_);
        Eigen::VectorXd dv2_dt2;
        dv2_dt2.resize(vlen_);

        interp_impl(t, 2, v, dv_dt, dv2_dt2);

        return std::tuple{v, dv_dt, dv2_dt2};
    }
};

template <int ORR>
struct InterpFunction1D
    : VectorFunction<InterpFunction1D<ORR>, 1, ORR, DenseDerivativeMode::Analytic,
                     DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<InterpFunction1D<ORR>, 1, ORR, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    std::shared_ptr<InterpTable1D> tab;

    InterpFunction1D() {}
    InterpFunction1D(std::shared_ptr<InterpTable1D> tab) : tab(tab) {
        this->set_io_rows(1, tab->vlen_);
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &v) {
            this->tab->interp_impl(x[0], 0, v, v, v);
            fx = v;
        };

        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &v, auto &dv_dt) {
            this->tab->interp_impl(x[0], 1, v, dv_dt, v);
            fx = v;
            jx = dv_dt;
        };

        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1),
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        auto Impl = [&](auto &v, auto &dv_dt, auto &dv2_dt2) {
            this->tab->interp_impl(x[0], 2, v, dv_dt, dv2_dt2);
            fx = v;
            jx = dv_dt;
            adjgrad[0] = dv_dt.dot(adjvars);
            adjhess(0, 0) = dv2_dt2.dot(adjvars);
        };

        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1),
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1),
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
};

} // namespace tycho::oc
