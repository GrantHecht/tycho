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
#include "tycho/detail/optimal_control/interp/interp_type.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using tycho::InterpType;
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::MatRef;
using vf::ThreadingFlags;
using vf::VecRef;
using vf::VectorExpression;
using vf::VectorFunction;

/// @ingroup optimal_control
/// @brief 1-D cubic/linear interpolation table over sampled vector data.
///
/// Stores vector-valued samples @f$v(t)@f$ on a 1-D grid of times and
/// interpolates them (and their first and second time derivatives) with either
/// cubic-Hermite or linear interpolation. Used to inject tabulated data into
/// VectorFunction expressions via @ref InterpFunction1D.
struct InterpTable1D {

    using MatType = Eigen::Matrix<double, -1, -1>; ///< @brief Dense double matrix type.

  private:
    // Cached derivative state: mutating any of these without rerunning
    // calc_derivs() corrupts cubic evaluation. Access only via set_data().
    Eigen::VectorXd ts_; ///< Sample times (ascending).
    MatType vs_;         ///< Sample values (columns are time samples).
    MatType dvs_dts_;    ///< Cached per-sample time derivatives (cubic mode).
    InterpType interp_kind_ = InterpType::Cubic; ///< Interpolation kind.

  public:
    bool teven_ = true; ///< Whether the time grid is evenly spaced.
    int axis_ = 0;      ///< Axis along which values are stored (0 or 1).
    int tsize_;         ///< Number of time samples.
    double ttotal_;     ///< Total time span.
    int vlen_;          ///< Length of each value vector.

    /// @brief Default constructor; produces an empty table (call @ref set_data first).
    InterpTable1D() {}

    /// @brief Construct from a time grid and a value matrix.
    /// @param Ts    Sample times (ascending, length >= 5).
    /// @param Vs    Sample values.
    /// @param axis  Axis of @p Vs that indexes time (0 or 1).
    /// @param kind  Interpolation kind.
    InterpTable1D(const Eigen::VectorXd &Ts, const MatType &Vs, int axis, InterpType kind) {
        set_data(Ts, Vs, axis, kind);
    }
    /// @brief Construct a scalar-valued table from a time grid and a value vector.
    /// @param Ts    Sample times (ascending, length >= 5).
    /// @param Vs    Sample values (one per time).
    /// @param kind  Interpolation kind.
    InterpTable1D(const Eigen::VectorXd &Ts, const Eigen::VectorXd &Vs, InterpType kind) {
        MatType Vstmp = Vs.transpose();
        set_data(Ts, Vstmp, 1, kind);
    }
    /// @brief Construct from a list of value-plus-time vectors.
    /// @param Vts   Each vector packs the values and a time coordinate.
    /// @param tvar  Index of the time coordinate within each vector (negative counts from end).
    /// @param kind  Interpolation kind.
    /// @throws std::invalid_argument if the input is empty, malformed, or @p tvar is out of range.
    InterpTable1D(const std::vector<Eigen::VectorXd> &Vts, int tvar, InterpType kind) {

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

    /// @brief Set the table data and (for cubic mode) precompute derivatives.
    /// @param Ts    Sample times (ascending, length >= 5).
    /// @param Vs    Sample values.
    /// @param axis  Axis of @p Vs that indexes time (0 or 1).
    /// @param kind  Interpolation kind.
    /// @throws std::invalid_argument if @p axis is invalid, fewer than 5 times are given,
    ///         sizes mismatch, or times are not ascending.
    void set_data(const Eigen::VectorXd &Ts, const MatType &Vs, int axis, InterpType kind) {

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

        this->interp_kind_ = kind;

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

        if (this->interp_kind_ == InterpType::Cubic)
            calc_derivs();
    }

    /// @internal
    /// @brief Precompute the per-sample time derivatives used by cubic interpolation.
    /// @endinternal
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

    /// @internal
    /// @brief Locate the grid interval index containing time @p t.
    /// @param t  Query time.
    /// @return Index of the lower grid node of the containing interval.
    /// @endinternal
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

    /// @internal
    /// @brief Core interpolation kernel writing the value and requested derivatives.
    /// @tparam VType  The value-vector type.
    /// @param t        Query time.
    /// @param deriv    Highest derivative order to compute (0, 1, or 2).
    /// @param v        Output interpolated value.
    /// @param dv_dt    Output first time derivative (if @p deriv > 0).
    /// @param dv2_dt2  Output second time derivative (if @p deriv > 1).
    /// @throws std::invalid_argument if @p t is outside the table range.
    /// @endinternal
    template <class VType>
    void interp_impl(double t, int deriv, VType &v, VType &dv_dt, VType &dv2_dt2) const {

        double eps = std::numeric_limits<double>::epsilon() * ttotal_;
        if (t < (ts_[0] - eps) || t > (ts_[ts_.size() - 1] + eps)) {
            throw std::invalid_argument(
                fmt::format("InterpTable1D: query t={} is outside table range [{}, {}]", t, ts_[0],
                            ts_[ts_.size() - 1]));
        }

        double telem = this->get_telem(t);
        double tstep = ts_[telem + 1] - ts_[telem];
        double tnd = (t - ts_[telem]) / tstep;

        if (this->interp_kind_ == InterpType::Cubic) {

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

    /// @brief Interpolate the value at a single time.
    /// @param t  Query time.
    /// @return The interpolated value vector.
    Eigen::VectorXd interp(double t) const {

        Eigen::VectorXd v;
        v.resize(vlen_);
        interp_impl(t, 0, v, v, v);
        return v;
    }

    /// @brief Interpolate the value at several times.
    /// @param t_vals  Query times.
    /// @return Matrix of interpolated values (one column per query time).
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

    /// @brief Interpolate the value and its first time derivative at a time.
    /// @param t  Query time.
    /// @return Tuple of {value, d/dt}.
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> interp_deriv1(double t) const {

        Eigen::VectorXd v;
        v.resize(vlen_);
        Eigen::VectorXd dv_dt;
        dv_dt.resize(vlen_);

        interp_impl(t, 1, v, dv_dt, dv_dt);

        return std::tuple{v, dv_dt};
    }

    /// @brief Interpolate the value and its first and second time derivatives at a time.
    /// @param t  Query time.
    /// @return Tuple of {value, d/dt, d2/dt2}.
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

/// @ingroup optimal_control
/// @brief VectorFunction wrapping an @ref InterpTable1D as a function of time.
///
/// Maps a scalar time input to the interpolated table value, with analytic first
/// and second derivatives, so tabulated data can be composed into VectorFunction
/// expressions.
/// @tparam ORR  Compile-time output dimension (table value length).
template <int ORR>
struct InterpFunction1D
    : VectorFunction<InterpFunction1D<ORR>, 1, ORR, DenseDerivativeMode::Analytic,
                     DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<InterpFunction1D<ORR>, 1, ORR, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    std::shared_ptr<InterpTable1D> tab; ///< The interpolation table being wrapped.

    /// @brief Default constructor; leaves the table unset.
    InterpFunction1D() {}
    /// @brief Construct from an interpolation table.
    /// @param tab  The table to wrap.
    InterpFunction1D(std::shared_ptr<InterpTable1D> tab) : tab(tab) {
        this->set_io_rows(1, tab->vlen_);
    }

    /// @internal
    /// @brief Evaluate the interpolated value at the input time.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input time (1-vector).
    /// @param fx_  Output interpolated-value vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &v) {
            this->tab->interp_impl(x[0], 0, v, v, v);
            fx = v;
        };

        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
    /// @internal
    /// @brief Evaluate the interpolated value and its time derivative (Jacobian).
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input time (1-vector).
    /// @param fx_  Output interpolated-value vector to write.
    /// @param jx_  Output Jacobian (d/dt) to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &v, auto &dv_dt) {
            this->tab->interp_impl(x[0], 1, v, dv_dt, v);
            fx = v;
            jx = dv_dt;
        };

        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1),
                                                  TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
    /// @internal
    /// @brief Evaluate the interpolated value, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input time (1-vector).
    /// @param fx_      Output interpolated-value vector to write.
    /// @param jx_      Output Jacobian (d/dt) to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjhess_ Output adjoint Hessian to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

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
