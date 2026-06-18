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
#include "tycho/detail/utils/timer.h"
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
/// @brief 2-D cubic/linear interpolation table over a scalar field @f$z(x,y)@f$.
///
/// Stores a scalar field sampled on a rectilinear @f$(x,y)@f$ grid and
/// interpolates it (and its gradient and Hessian) with bicubic or bilinear
/// interpolation. Exposed to expressions via @ref InterpFunction2D.
struct InterpTable2D {

    using MatType =
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor>; ///< @brief Row-major double matrix.

  private:
    // Cached derivative state: mutating any of these without rerunning
    // calc_derivs() corrupts cubic evaluation. Access only via set_data().
    Eigen::VectorXd xs_; ///< X grid coordinates (ascending).
    Eigen::VectorXd ys_; ///< Y grid coordinates (ascending).
    MatType zs_;         ///< Field values @f$z(x,y)@f$.
    MatType dzxs_;       ///< Cached @f$\partial z/\partial x@f$ at grid nodes.
    MatType dzys_;       ///< Cached @f$\partial z/\partial y@f$ at grid nodes.
    MatType dzys_dxs_;   ///< Cached @f$\partial^2 z/\partial x\partial y@f$ at grid nodes.
    Eigen::Matrix<Eigen::Array4d, -1, -1, Eigen::RowMajor>
        all_dat_;                                ///< Per-cell coefficient cache.
    InterpType interp_kind_ = InterpType::Cubic; ///< Interpolation kind.

  public:
    bool xeven_ = true; ///< Whether the X grid is evenly spaced.
    bool yeven_ = true; ///< Whether the Y grid is evenly spaced.
    int xsize_;         ///< Number of X grid nodes.
    double xtotal_;     ///< Total X span.
    int ysize_;         ///< Number of Y grid nodes.
    double ytotal_;     ///< Total Y span.

    /// @brief Default constructor; produces an empty table (call @ref set_data first).
    InterpTable2D() {}

    /// @brief Construct from grid coordinates and a field matrix.
    /// @param Xs    X grid coordinates (ascending, length >= 5).
    /// @param Ys    Y grid coordinates (ascending, length >= 5).
    /// @param Zs    Field values (rows index Y, cols index X).
    /// @param kind  Interpolation kind.
    InterpTable2D(const Eigen::VectorXd &Xs, const Eigen::VectorXd &Ys, const MatType &Zs,
                  InterpType kind) {
        set_data(Xs, Ys, Zs, kind);
    }

    /// @brief Set the table data and (for cubic mode) precompute derivatives.
    /// @param Xs    X grid coordinates (ascending, length >= 5).
    /// @param Ys    Y grid coordinates (ascending, length >= 5).
    /// @param Zs    Field values (rows index Y, cols index X).
    /// @param kind  Interpolation kind.
    /// @throws std::invalid_argument on too-few nodes, size mismatch, or non-ascending grids.
    void set_data(const Eigen::VectorXd &Xs, const Eigen::VectorXd &Ys, const MatType &Zs,
                  InterpType kind) {

        this->interp_kind_ = kind;

        this->xs_ = Xs;
        this->ys_ = Ys;
        this->zs_ = Zs;

        xsize_ = xs_.size();
        ysize_ = ys_.size();

        if (xsize_ < 5) {
            throw std::invalid_argument("X coordinates must be larger than 4");
        }
        if (ysize_ < 5) {
            throw std::invalid_argument("Y  coordinates must be larger than 4");
        }

        if (xsize_ != zs_.cols()) {
            throw std::invalid_argument("X coordinates must match cols in Z matrix");
        }
        if (ysize_ != zs_.rows()) {
            throw std::invalid_argument("Y coordinates must match rows in Z matrix");
        }

        for (int i = 0; i < xs_.size() - 1; i++) {
            if (xs_[i + 1] < xs_[i]) {
                throw std::invalid_argument("X Coordinates must be in ascending order");
            }
        }
        for (int i = 0; i < ys_.size() - 1; i++) {
            if (ys_[i + 1] < ys_[i]) {
                throw std::invalid_argument("Y Coordinates must be in ascending order");
            }
        }

        xtotal_ = xs_[xsize_ - 1] - xs_[0];
        ytotal_ = ys_[ysize_ - 1] - ys_[0];

        Eigen::VectorXd testx;
        testx.setLinSpaced(xs_.size(), xs_[0], xs_[xs_.size() - 1]);
        Eigen::VectorXd testy;
        testy.setLinSpaced(ys_.size(), ys_[0], ys_[ys_.size() - 1]);

        double xerr = (xs_ - testx).lpNorm<Eigen::Infinity>();
        double yerr = (ys_ - testy).lpNorm<Eigen::Infinity>();

        if (xerr > abs(xtotal_) * 1.0e-12) {
            this->xeven_ = false;
        }
        if (yerr > abs(ytotal_) * 1.0e-12) {
            this->yeven_ = false;
        }

        if (this->interp_kind_ == InterpType::Cubic)
            calc_derivs();
    }

    /// @internal
    /// @brief Precompute the grid-node partial derivatives used by bicubic interpolation.
    /// @endinternal
    void calc_derivs() {
        dzxs_.resize(ysize_, xsize_);
        dzys_.resize(ysize_, xsize_);
        dzys_dxs_.resize(ysize_, xsize_);
        all_dat_.resize(ysize_, xsize_);

        Eigen::Matrix<double, 5, 5> stens;
        stens.row(0).setOnes();
        Eigen::Matrix<double, 5, 1> rhs;
        rhs << 0, 1, 0, 0, 0;
        Eigen::Matrix<double, 5, 1> times;
        Eigen::Matrix<double, 5, 1> coeffs;

        bool hitcent = false;
        for (int i = 0; i < this->ysize_; i++) {
            int start = 0;
            bool recalc = true;
            if (i + 2 <= this->ysize_ - 1 && i - 2 >= 0) {
                // central difference
                if (this->yeven_ && hitcent) {
                    recalc = false;
                }
                hitcent = true;
                start = i - 2;
            } else if (i < this->ysize_ - 1 - i) {
                // forward difference
                start = 0;
            } else {
                // backward difference
                start = this->ysize_ - 5;
            }
            int stepdir = (i < this->ysize_ - 1) ? 1 : -1;
            double yi = this->ys_[i];
            double ystep = std::abs(this->ys_[i + stepdir] - yi);
            if (recalc) {
                times = this->ys_.segment(start, 5);
                times -= Eigen::Matrix<double, 5, 1>::Constant(yi);
                times /= ystep;
                stens.row(1) = times.transpose();
                stens.row(2) = stens.row(1).cwiseProduct(times.transpose());
                stens.row(3) = stens.row(2).cwiseProduct(times.transpose());
                stens.row(4) = stens.row(3).cwiseProduct(times.transpose());
                coeffs = stens.inverse() * rhs;
            }
            dzys_.row(i) = (coeffs / ystep).transpose() * this->zs_.middleRows(start, 5);
        }

        hitcent = false;
        for (int i = 0; i < this->xsize_; i++) {
            int start = 0;
            bool recalc = true;
            if (i + 2 <= this->xsize_ - 1 && i - 2 >= 0) {
                // central difference
                if (this->xeven_ && hitcent) {
                    recalc = false;
                }
                hitcent = true;
                start = i - 2;
            } else if (i < this->xsize_ - 1 - i) {
                // forward difference
                start = 0;
            } else {
                // backward difference
                start = this->xsize_ - 5;
            }
            int stepdir = (i < this->xsize_ - 1) ? 1 : -1;
            double xi = this->xs_[i];
            double xstep = std::abs(this->xs_[i + stepdir] - xi);
            if (recalc) {
                times = this->xs_.segment(start, 5);
                times -= Eigen::Matrix<double, 5, 1>::Constant(xi);
                times /= xstep;
                stens.row(1) = times.transpose();
                stens.row(2) = stens.row(1).cwiseProduct(times.transpose());
                stens.row(3) = stens.row(2).cwiseProduct(times.transpose());
                stens.row(4) = stens.row(3).cwiseProduct(times.transpose());
                coeffs = stens.inverse() * rhs;
            }

            dzxs_.col(i) = this->zs_.middleCols(start, 5) * (coeffs / xstep);
            dzys_dxs_.col(i) = this->dzys_.middleCols(start, 5) * (coeffs / xstep);
        }
    }

    /// @internal
    /// @brief Find the grid interval index containing a value (uneven-grid search).
    /// @param vals  Ascending grid coordinates.
    /// @param v     Query value.
    /// @return Index of the lower node of the containing interval.
    /// @endinternal
    int find_elem(const Eigen::VectorXd &vals, double v) const {
        int center = int(vals.size() / 2);
        int shift = (vals[center] > v) ? 0 : center;
        auto it = std::upper_bound(vals.begin() + shift, vals.end(), v);
        int elem = int(it - vals.begin()) - 1;
        return elem;
    }

    /// @internal
    /// @brief Locate the grid cell containing @f$(x,y)@f$.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @return Tuple of {x-interval index, y-interval index}.
    /// @endinternal
    std::tuple<int, int> get_xyelems(double x, double y) const {
        int xelem, yelem;

        if (this->xeven_) {
            double xlocal = x - this->xs_[0];
            double xstep = this->xs_[1] - this->xs_[0];
            xelem = std::min(int(xlocal / xstep), this->xsize_ - 2);
        } else {
            xelem = this->find_elem(this->xs_, x);
        }

        if (this->yeven_) {
            double ylocal = y - this->ys_[0];
            double ystep = this->ys_[1] - this->ys_[0];
            yelem = std::min(int(ylocal / ystep), this->ysize_ - 2);
        } else {
            yelem = this->find_elem(this->ys_, y);
        }

        xelem = std::min(xelem, this->xsize_ - 2);
        yelem = std::min(yelem, this->ysize_ - 2);

        xelem = std::max(xelem, 0);
        yelem = std::max(yelem, 0);

        return std::tuple{xelem, yelem};
    }

    /// @internal
    /// @brief Build the 4x4 bicubic coefficient matrix for one grid cell.
    /// @param xelem  X-interval index.
    /// @param yelem  Y-interval index.
    /// @return The bicubic coefficient matrix for the cell.
    /// @endinternal
    Eigen::Matrix4<double> get_amatrix(int xelem, int yelem) const {

        double xstep = xs_[xelem + 1] - xs_[xelem];
        double ystep = ys_[yelem + 1] - ys_[yelem];

        Eigen::Matrix4<double> a;
        Eigen::Matrix4<double> L;
        L << 1, 0, 0, 0, 0, 0, 1, 0, -3, 3, -2, -1, 2, -2, 1, 1;
        Eigen::Matrix4<double> R;

        R << 1, 0, -3, 2, 0, 0, 3, -2, 0, 1, -2, 1, 0, 0, -1, 1;

        Eigen::Matrix4<double> Z;

        double z00 = zs_(yelem, xelem);
        double z10 = zs_(yelem, xelem + 1);
        double z01 = zs_(yelem + 1, xelem);
        double z11 = zs_(yelem + 1, xelem + 1);

        double dz00_x = dzxs_(yelem, xelem) * xstep;
        double dz10_x = dzxs_(yelem, xelem + 1) * xstep;
        double dz01_x = dzxs_(yelem + 1, xelem) * xstep;
        double dz11_x = dzxs_(yelem + 1, xelem + 1) * xstep;

        double dz00_y = dzys_(yelem, xelem) * ystep;
        double dz10_y = dzys_(yelem, xelem + 1) * ystep;
        double dz01_y = dzys_(yelem + 1, xelem) * ystep;
        double dz11_y = dzys_(yelem + 1, xelem + 1) * ystep;

        double dz00_xy = dzys_dxs_(yelem, xelem) * xstep * ystep;
        double dz10_xy = dzys_dxs_(yelem, xelem + 1) * xstep * ystep;
        double dz01_xy = dzys_dxs_(yelem + 1, xelem) * xstep * ystep;
        double dz11_xy = dzys_dxs_(yelem + 1, xelem + 1) * xstep * ystep;

        Z << z00, z01, dz00_y, dz01_y, z10, z11, dz10_y, dz11_y, dz00_x, dz01_x, dz00_xy, dz01_xy,
            dz10_x, dz11_x, dz10_xy, dz11_xy;

        a = L * Z * R;
        return a;
    }

    /// @internal
    /// @brief Core interpolation kernel writing the value and requested derivatives.
    /// @param x       Query x coordinate.
    /// @param y       Query y coordinate.
    /// @param deriv   Highest derivative order to compute (0, 1, or 2).
    /// @param z       Output interpolated value.
    /// @param dzxy    Output gradient @f$[\partial_x z, \partial_y z]@f$ (if @p deriv > 0).
    /// @param d2zxy   Output Hessian (if @p deriv > 1).
    /// @throws std::invalid_argument if @p x or @p y is outside the table range.
    /// @endinternal
    void interp_impl(double x, double y, int deriv, double &z, Eigen::Vector2<double> &dzxy,
                     Eigen::Matrix2<double> &d2zxy) const {

        double xeps = std::numeric_limits<double>::epsilon() * xtotal_;
        if (x < (xs_[0] - xeps) || x > (xs_[xs_.size() - 1] + xeps)) {
            throw std::invalid_argument(
                fmt::format("InterpTable2D: query x={} is outside table x range [{}, {}]", x,
                            xs_[0], xs_[xs_.size() - 1]));
        }
        double yeps = std::numeric_limits<double>::epsilon() * ytotal_;
        if (y < (ys_[0] - yeps) || y > (ys_[ys_.size() - 1]) + yeps) {
            throw std::invalid_argument(
                fmt::format("InterpTable2D: query y={} is outside table y range [{}, {}]", y,
                            ys_[0], ys_[ys_.size() - 1]));
        }

        auto [xelem, yelem] = get_xyelems(x, y);

        double xstep = xs_[xelem + 1] - xs_[xelem];
        double ystep = ys_[yelem + 1] - ys_[yelem];

        double xf = (x - xs_[xelem]) / xstep;
        double yf = (y - ys_[yelem]) / ystep;

        if (this->interp_kind_ == InterpType::Cubic) {

            double yf2 = yf * yf;
            double yf3 = yf2 * yf;
            double xf2 = xf * xf;
            double xf3 = xf2 * xf;

            Eigen::Matrix4<double> amat = get_amatrix(xelem, yelem);
            Vector4<double> xvec;
            xvec << 1, xf, xf2, xf3;

            Vector4<double> yvec;
            yvec << 1, yf, yf2, yf3;

            z = xvec.transpose() * amat * yvec;

            if (deriv > 0) {

                Vector4<double> dxvec;
                dxvec << 0, 1 / xstep, 2 * xf / xstep, 3 * xf2 / xstep;

                Vector4<double> dyvec;
                dyvec << 0, 1 / ystep, 2 * yf / ystep, 3 * yf2 / ystep;

                dzxy[0] = dxvec.transpose() * amat * yvec;
                dzxy[1] = xvec.transpose() * amat * dyvec;

                if (deriv > 1) {

                    Vector4<double> d2xvec;
                    d2xvec << 0, 0, 2 / (xstep * xstep), 6 * xf / (xstep * xstep);

                    Vector4<double> d2yvec;
                    d2yvec << 0, 0, 2 / (ystep * ystep), 6 * yf / (ystep * ystep);

                    d2zxy(0, 0) = d2xvec.transpose() * amat * yvec;
                    d2zxy(1, 0) = dxvec.transpose() * amat * dyvec;
                    d2zxy(0, 1) = d2zxy(1, 0);
                    d2zxy(1, 1) = xvec.transpose() * amat * d2yvec;
                }
            }

        } else {
            // Linear
            double zx0y0 = zs_(yelem, xelem);
            double zx1y0 = zs_(yelem, xelem + 1);
            double zy0m = zx0y0 * (1 - xf) + zx1y0 * xf;

            double zx0y1 = zs_(yelem + 1, xelem);
            double zx1y1 = zs_(yelem + 1, xelem + 1);
            double zy1m = zx0y1 * (1 - xf) + zx1y1 * xf;

            z = zy0m * (1 - yf) + zy1m * (yf);

            if (deriv > 0) {
                dzxy[0] = ((zx1y0 - zx0y0) * (1 - yf) + (zx1y1 - zx0y1) * (yf)) / xstep;
                dzxy[1] = (zy1m - zy0m) / ystep;
                if (deriv > 1) {
                    d2zxy.setZero();
                    d2zxy(1, 0) =
                        ((zx1y0 - zx0y0) * (-1) + (zx1y1 - zx0y1) * (1)) / (xstep * ystep);
                    d2zxy(0, 1) = d2zxy(1, 0);
                }
            }
        }
    }

    /// @brief Interpolate the field value at a point.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @return The interpolated value.
    double interp(double x, double y) const {

        double z;
        Eigen::Vector2<double> dzxy;
        Eigen::Matrix2<double> d2zxy;
        interp_impl(x, y, 0, z, dzxy, d2zxy);

        return z;
    }

    /// @brief Interpolate the field value at a grid of points.
    /// @param x_vals  X coordinates of the query points.
    /// @param y_vals  Y coordinates of the query points (same shape as @p x_vals).
    /// @return Matrix of interpolated values.
    MatType interp(const MatType &x_vals, const MatType &y_vals) const {
        MatType z_out(x_vals.rows(), x_vals.cols());

        for (int i = 0; i < x_vals.rows(); i++) {
            for (int j = 0; j < x_vals.cols(); j++) {
                z_out(i, j) = interp(x_vals(i, j), y_vals(i, j));
            }
        }
        return z_out;
    }

    /// @brief Interpolate the value and gradient at a point.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @return Tuple of {value, gradient}.
    std::tuple<double, Eigen::Vector2<double>> interp_deriv1(double x, double y) const {
        double z;
        Eigen::Vector2<double> dzxy;
        Eigen::Matrix2<double> d2zxy;

        interp_impl(x, y, 1, z, dzxy, d2zxy);

        return std::tuple{z, dzxy}; // intellisense is confused pls ignore
    }

    /// @brief Interpolate the value, gradient, and Hessian at a point.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @return Tuple of {value, gradient, Hessian}.
    std::tuple<double, Eigen::Vector2<double>, Eigen::Matrix2<double>>
    interp_deriv2(double x, double y) const {
        double z;
        Eigen::Vector2<double> dzxy;
        Eigen::Matrix2<double> d2zxy;

        interp_impl(x, y, 2, z, dzxy, d2zxy);

        return std::tuple{z, dzxy, d2zxy}; // intellisense is confused pls ignore
    }
};

/// @ingroup optimal_control
/// @brief VectorFunction wrapping an @ref InterpTable2D as a function of @f$(x,y)@f$.
///
/// Maps a 2-vector input to the interpolated scalar field value, with analytic
/// gradient and Hessian, for composition into VectorFunction expressions.
struct InterpFunction2D : VectorFunction<InterpFunction2D, 2, 1, DenseDerivativeMode::Analytic,
                                         DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<InterpFunction2D, 2, 1, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    std::shared_ptr<InterpTable2D> tab; ///< The interpolation table being wrapped.

    /// @brief Default constructor; leaves the table unset.
    InterpFunction2D() {}
    /// @brief Construct from an interpolation table.
    /// @param tab  The table to wrap.
    InterpFunction2D(std::shared_ptr<InterpTable2D> tab) : tab(tab) { this->set_io_rows(2, 1); }

    /// @internal
    /// @brief Evaluate the interpolated value at the input point.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input @c [x, y].
    /// @param fx_  Output value (1-vector) to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx[0] = this->tab->interp(x[0], x[1]);
    }
    /// @internal
    /// @brief Evaluate the interpolated value and gradient (Jacobian).
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input @c [x, y].
    /// @param fx_  Output value (1-vector) to write.
    /// @param jx_  Output Jacobian (gradient) to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto [z, dzdx] = this->tab->interp_deriv1(x[0], x[1]);
        fx[0] = z;
        jx = dzdx.transpose();
    }
    /// @internal
    /// @brief Evaluate the value, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input @c [x, y].
    /// @param fx_      Output value (1-vector) to write.
    /// @param jx_      Output Jacobian to write.
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

        auto [z, dzdx, d2zdx] = this->tab->interp_deriv2(x[0], x[1]);
        fx[0] = z;
        jx = dzdx.transpose();
        adjgrad = adjvars[0] * dzdx;
        adjhess = adjvars[0] * d2zdx;
    }
};

} // namespace tycho::oc