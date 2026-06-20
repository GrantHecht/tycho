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
#include <unsupported/Eigen/CXX11/Tensor>

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
/// @brief 3-D tricubic/trilinear interpolation table over a scalar field @f$f(x,y,z)@f$.
///
/// Stores a scalar field sampled on a rectilinear @f$(x,y,z)@f$ grid (as an
/// Eigen tensor) and interpolates it (and its gradient and Hessian) with
/// tricubic or trilinear interpolation, optionally caching per-cell coefficients.
/// Exposed to expressions via @ref InterpFunction3D.
struct InterpTable3D {

  private:
    // Cached derivative state: mutating any of these without rerunning
    // calc_derivs() corrupts cubic evaluation. Access only via set_data().
    Eigen::VectorXd xs_;
    Eigen::VectorXd ys_;
    Eigen::VectorXd zs_;

    // numpy meshgrid ij format (x,y,z). ColMajor (Eigen's tensor default) is
    // intentional: the calc_alphavec inner loop accesses (i, j, k) and
    // (i+1, j, k) consecutively, which is contiguous in ColMajor (X-axis
    // stride 1). Layout conversion from numpy's row-major data happens in the
    // type_caster<Eigen::Tensor> at construction time; see
    // src/bindings/type_casters.h for the swap_layout()+shuffle() conversion.
    Eigen::Tensor<double, 3> fs_;

    Eigen::Tensor<double, 3> fs_dx_;
    Eigen::Tensor<double, 3> fs_dy_;
    Eigen::Tensor<double, 3> fs_dz_;

    Eigen::Tensor<double, 3> fs_dxdy_;
    Eigen::Tensor<double, 3> fs_dxdz_;
    Eigen::Tensor<double, 3> fs_dydz_;

    Eigen::Tensor<double, 3> fs_dxdydz_;

    Eigen::Tensor<Eigen::Matrix<double, 64, 1>, 3> alphavecs_;

    InterpType interp_kind_ = InterpType::Linear;

  public:
    bool xeven_ = true; ///< Whether the X grid is evenly spaced.
    bool yeven_ = true; ///< Whether the Y grid is evenly spaced.
    bool zeven_ = true; ///< Whether the Z grid is evenly spaced.

    int xsize_;                ///< Number of X grid nodes.
    double xtotal_;            ///< Total X span.
    int ysize_;                ///< Number of Y grid nodes.
    double ytotal_;            ///< Total Y span.
    int zsize_;                ///< Number of Z grid nodes.
    double ztotal_;            ///< Total Z span.
    bool cache_alpha_ = false; ///< Whether per-cell tricubic coefficients are precomputed.
    int cache_threads_ = 1;    ///< Threads used to populate the coefficient cache.

    /// @brief Default constructor; produces an empty table.
    InterpTable3D() {}

    /// @brief Construct from grid coordinates and a value tensor.
    /// @param Xs     X grid coordinates (ascending, length >= 5).
    /// @param Ys     Y grid coordinates (ascending, length >= 5).
    /// @param Zs     Z grid coordinates (ascending, length >= 5).
    /// @param Fs     Field values as a rank-3 tensor (dims index X, Y, Z).
    /// @param kind   Interpolation kind.
    /// @param cache  Whether to precompute per-cell tricubic coefficients.
    /// @throws std::invalid_argument on too-few nodes, dimension mismatch, or non-ascending grids.
    InterpTable3D(const Eigen::VectorXd &Xs, const Eigen::VectorXd &Ys, const Eigen::VectorXd &Zs,
                  const Eigen::Tensor<double, 3> &Fs, InterpType kind, bool cache) {

        this->xs_ = Xs;
        this->ys_ = Ys;
        this->zs_ = Zs;
        this->fs_ = Fs;
        this->cache_alpha_ = cache;
        this->interp_kind_ = kind;

        xsize_ = xs_.size();
        ysize_ = ys_.size();
        zsize_ = zs_.size();

        if (xsize_ < 5) {
            throw std::invalid_argument("X coordinates must be larger than 4");
        }
        if (ysize_ < 5) {
            throw std::invalid_argument("Y  coordinates must be larger than 4");
        }
        if (zsize_ < 5) {
            throw std::invalid_argument("Z  coordinates must be larger than 4");
        }

        if (xsize_ != fs_.dimension(0)) {
            throw std::invalid_argument("X coordinates must be first dimension of value tensor");
        }
        if (ysize_ != fs_.dimension(1)) {
            throw std::invalid_argument("Y coordinates must be second dimension of value tensor");
        }
        if (zsize_ != fs_.dimension(2)) {
            throw std::invalid_argument("Z coordinates must be third dimension of value tensor");
        }

        for (int i = 0; i < xsize_ - 1; i++) {
            if (xs_[i + 1] < xs_[i]) {
                throw std::invalid_argument("X coordinates must be in ascending order");
            }
        }
        for (int i = 0; i < ysize_ - 1; i++) {
            if (ys_[i + 1] < ys_[i]) {
                throw std::invalid_argument("Y coordinates must be in ascending order");
            }
        }
        for (int i = 0; i < zsize_ - 1; i++) {
            if (zs_[i + 1] < zs_[i]) {
                throw std::invalid_argument("Z coordinates must be in ascending order");
            }
        }

        xtotal_ = xs_[xsize_ - 1] - xs_[0];
        ytotal_ = ys_[ysize_ - 1] - ys_[0];
        ztotal_ = zs_[zsize_ - 1] - zs_[0];

        Eigen::VectorXd testx;
        testx.setLinSpaced(xsize_, xs_[0], xs_[xsize_ - 1]);
        Eigen::VectorXd testy;
        testy.setLinSpaced(ysize_, ys_[0], ys_[ysize_ - 1]);
        Eigen::VectorXd testz;
        testz.setLinSpaced(zsize_, zs_[0], zs_[zsize_ - 1]);

        double xerr = (xs_ - testx).lpNorm<Eigen::Infinity>();
        double yerr = (ys_ - testy).lpNorm<Eigen::Infinity>();
        double zerr = (zs_ - testz).lpNorm<Eigen::Infinity>();

        if (xerr > abs(xtotal_) * 1.0e-12) {
            this->xeven_ = false;
        }
        if (yerr > abs(ytotal_) * 1.0e-12) {
            this->yeven_ = false;
        }
        if (zerr > abs(ztotal_) * 1.0e-12) {
            this->zeven_ = false;
        }

        if (this->interp_kind_ == InterpType::Cubic) {
            this->calc_derivs();
            if (this->cache_alpha_) {
                this->cache_alphavecs();
            }
        }
    }

    /// @internal
    /// @brief Precompute the grid-node partial derivatives used by tricubic interpolation.
    /// @endinternal
    void calc_derivs() {

        fs_dx_.resize(fs_.dimension(0), fs_.dimension(1), fs_.dimension(2));
        fs_dy_.resize(fs_.dimension(0), fs_.dimension(1), fs_.dimension(2));
        fs_dz_.resize(fs_.dimension(0), fs_.dimension(1), fs_.dimension(2));
        fs_dxdy_.resize(fs_.dimension(0), fs_.dimension(1), fs_.dimension(2));
        fs_dxdz_.resize(fs_.dimension(0), fs_.dimension(1), fs_.dimension(2));
        fs_dydz_.resize(fs_.dimension(0), fs_.dimension(1), fs_.dimension(2));
        fs_dxdydz_.resize(fs_.dimension(0), fs_.dimension(1), fs_.dimension(2));

        auto fdiffimpl = [&](int dir, bool even, const auto &ts, const auto &src, auto &dest) {
            Eigen::Matrix<double, 5, 5> stens;
            stens.row(0).setOnes();
            Eigen::Matrix<double, 5, 1> rhs;
            rhs << 0, 1, 0, 0, 0;
            Eigen::Matrix<double, 5, 1> times;
            Eigen::Matrix<double, 5, 1> coeffs;

            int tsize = ts.size();
            bool hitcent = false;
            for (int i = 0; i < tsize; i++) {
                int start = 0;
                bool recalc = true;
                if (i + 2 <= tsize - 1 && i - 2 >= 0) {
                    // central difference
                    if (even && hitcent) {
                        recalc = false;
                    }
                    hitcent = true;
                    start = i - 2;
                } else if (i < tsize - 1 - i) {
                    // forward difference
                    start = 0;
                } else {
                    // backward difference
                    start = tsize - 5;
                }
                int stepdir = (i < tsize - 1) ? 1 : -1;
                double ti = ts[i];
                double tstep = std::abs(ts[i + stepdir] - ti);
                if (recalc) {
                    times = ts.segment(start, 5);
                    times -= Eigen::Matrix<double, 5, 1>::Constant(ti);
                    times /= tstep;
                    stens.row(1) = times.transpose();
                    stens.row(2) = (stens.row(1).cwiseProduct(times.transpose())).eval();
                    stens.row(3) = (stens.row(2).cwiseProduct(times.transpose())).eval();
                    stens.row(4) = (stens.row(3).cwiseProduct(times.transpose())).eval();

                    coeffs = stens.inverse() * rhs;
                }
                dest.chip(i, dir) = src.chip(start, dir) * (coeffs[0] / tstep) +
                                    src.chip(start + 1, dir) * (coeffs[1] / tstep) +
                                    src.chip(start + 2, dir) * (coeffs[2] / tstep) +
                                    src.chip(start + 3, dir) * (coeffs[3] / tstep) +
                                    src.chip(start + 4, dir) * (coeffs[4] / tstep);
            }
        };

        fdiffimpl(0, this->xeven_, this->xs_, this->fs_, this->fs_dx_);
        fdiffimpl(1, this->yeven_, this->ys_, this->fs_, this->fs_dy_);
        fdiffimpl(2, this->zeven_, this->zs_, this->fs_, this->fs_dz_);

        fdiffimpl(1, this->yeven_, this->ys_, this->fs_dx_, this->fs_dxdy_);
        fdiffimpl(2, this->zeven_, this->zs_, this->fs_dx_, this->fs_dxdz_);
        fdiffimpl(2, this->zeven_, this->zs_, this->fs_dy_, this->fs_dydz_);

        fdiffimpl(2, this->zeven_, this->zs_, this->fs_dxdy_, this->fs_dxdydz_);
    }

    /// @internal
    /// @brief Precompute and cache the tricubic coefficient vector of every grid cell.
    /// @endinternal
    void cache_alphavecs() {
        this->alphavecs_.resize(fs_.dimension(0) - 1, fs_.dimension(1) - 1, fs_.dimension(2) - 1);
        for (int i = 0; i < zsize_ - 1; i++) {
            for (int j = 0; j < ysize_ - 1; j++) {
                for (int k = 0; k < xsize_ - 1; k++) {
                    this->alphavecs_(k, j, i) = this->calc_alphavec(k, j, i);
                }
            }
        }
    }

    /// @internal
    /// @brief Find the grid interval index containing a value along one axis.
    /// @param ts     Ascending grid coordinates for the axis.
    /// @param teven  Whether the axis is evenly spaced.
    /// @param t      Query value.
    /// @return Index of the lower node of the containing interval.
    /// @endinternal
    static int find_elem(const Eigen::VectorXd &ts, bool teven, double t) {
        int elem;
        if (teven) {
            double tlocal = t - ts[0];
            double tstep = ts[1] - ts[0];
            elem = int(tlocal / tstep);
        } else {
            int center = int(ts.size() / 2);
            int shift = (ts[center] > t) ? 0 : center;
            auto it = std::upper_bound(ts.begin() + shift, ts.end(), t);
            elem = int(it - ts.begin()) - 1;
        }
        elem = std::min(elem, int(ts.size() - 2));
        elem = std::max(elem, 0);
        return elem;
    }

    /// @internal
    /// @brief Locate the grid cell containing @f$(x,y,z)@f$.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @param z  Query z coordinate.
    /// @return Tuple of {x-, y-, z-interval indices}.
    /// @endinternal
    std::tuple<int, int, int> get_xyzelems(double x, double y, double z) const {

        int xelem = this->find_elem(this->xs_, this->xeven_, x);
        int yelem = this->find_elem(this->ys_, this->yeven_, y);
        int zelem = this->find_elem(this->zs_, this->zeven_, z);

        return std::tuple{xelem, yelem, zelem};
    }

    /// @internal
    /// @brief Compute the 64-element tricubic coefficient vector for one grid cell.
    /// @param xelem  X-interval index.
    /// @param yelem  Y-interval index.
    /// @param zelem  Z-interval index.
    /// @return The tricubic coefficient vector for the cell.
    /// @endinternal
    Eigen::Matrix<double, 64, 1> calc_alphavec(int xelem, int yelem, int zelem) const {

        double xstep = xs_[xelem + 1] - xs_[xelem];
        double ystep = ys_[yelem + 1] - ys_[yelem];
        double zstep = zs_[zelem + 1] - zs_[zelem];

        Eigen::Matrix<double, 64, 1> bvec;
        Eigen::Matrix<double, 64, 1> alphavec;

        auto fillop = [&](auto start, const auto &src) {
            bvec[start] = src(xelem, yelem, zelem);
            bvec[start + 1] = src(xelem + 1, yelem, zelem);
            bvec[start + 2] = src(xelem, yelem + 1, zelem);
            bvec[start + 3] = src(xelem + 1, yelem + 1, zelem);

            bvec[start + 4] = src(xelem, yelem, zelem + 1);
            bvec[start + 5] = src(xelem + 1, yelem, zelem + 1);
            bvec[start + 6] = src(xelem, yelem + 1, zelem + 1);
            bvec[start + 7] = src(xelem + 1, yelem + 1, zelem + 1);
        };

        fillop(0, this->fs_);
        fillop(8, this->fs_dx_);
        fillop(16, this->fs_dy_);
        fillop(24, this->fs_dz_);
        fillop(32, this->fs_dxdy_);
        fillop(40, this->fs_dxdz_);
        fillop(48, this->fs_dydz_);
        fillop(56, this->fs_dxdydz_);

        bvec.segment(8, 8) *= (xstep);
        bvec.segment(16, 8) *= (ystep);
        bvec.segment(24, 8) *= (zstep);
        bvec.segment(32, 8) *= (xstep * ystep);
        bvec.segment(40, 8) *= (xstep * zstep);
        bvec.segment(48, 8) *= (ystep * zstep);
        bvec.segment(56, 8) *= (xstep * ystep * zstep);

        return this->apply_coeefs(bvec);
    }

    /// @internal
    /// @brief Apply the fixed tricubic interpolation matrix to a sampled-derivative vector.
    /// @param bvec  The 64-element vector of cell corner values and derivatives.
    /// @return The 64-element tricubic coefficient vector.
    /// @endinternal
    Eigen::Matrix<double, 64, 1> apply_coeefs(const Eigen::Matrix<double, 64, 1> &bvec) const {

        Eigen::Matrix<double, 64, 1> alphavec;

        alphavec[0] = +1 * bvec[0];
        alphavec[1] = +1 * bvec[8];
        alphavec[2] = -3 * bvec[0] + 3 * bvec[1] - 2 * bvec[8] - 1 * bvec[9];
        alphavec[3] = +2 * bvec[0] - 2 * bvec[1] + 1 * bvec[8] + 1 * bvec[9];
        alphavec[4] = +1 * bvec[16];
        alphavec[5] = +1 * bvec[32];
        alphavec[6] = -3 * bvec[16] + 3 * bvec[17] - 2 * bvec[32] - 1 * bvec[33];
        alphavec[7] = +2 * bvec[16] - 2 * bvec[17] + 1 * bvec[32] + 1 * bvec[33];
        alphavec[8] = -3 * bvec[0] + 3 * bvec[2] - 2 * bvec[16] - 1 * bvec[18];
        alphavec[9] = -3 * bvec[8] + 3 * bvec[10] - 2 * bvec[32] - 1 * bvec[34];
        alphavec[10] = +9 * bvec[0] - 9 * bvec[1] - 9 * bvec[2] + 9 * bvec[3] + 6 * bvec[8] +
                       3 * bvec[9] - 6 * bvec[10] - 3 * bvec[11] + 6 * bvec[16] - 6 * bvec[17] +
                       3 * bvec[18] - 3 * bvec[19] + 4 * bvec[32] + 2 * bvec[33] + 2 * bvec[34] +
                       1 * bvec[35];
        alphavec[11] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[2] - 6 * bvec[3] - 3 * bvec[8] -
                       3 * bvec[9] + 3 * bvec[10] + 3 * bvec[11] - 4 * bvec[16] + 4 * bvec[17] -
                       2 * bvec[18] + 2 * bvec[19] - 2 * bvec[32] - 2 * bvec[33] - 1 * bvec[34] -
                       1 * bvec[35];
        alphavec[12] = +2 * bvec[0] - 2 * bvec[2] + 1 * bvec[16] + 1 * bvec[18];
        alphavec[13] = +2 * bvec[8] - 2 * bvec[10] + 1 * bvec[32] + 1 * bvec[34];
        alphavec[14] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[2] - 6 * bvec[3] - 4 * bvec[8] -
                       2 * bvec[9] + 4 * bvec[10] + 2 * bvec[11] - 3 * bvec[16] + 3 * bvec[17] -
                       3 * bvec[18] + 3 * bvec[19] - 2 * bvec[32] - 1 * bvec[33] - 2 * bvec[34] -
                       1 * bvec[35];
        alphavec[15] = +4 * bvec[0] - 4 * bvec[1] - 4 * bvec[2] + 4 * bvec[3] + 2 * bvec[8] +
                       2 * bvec[9] - 2 * bvec[10] - 2 * bvec[11] + 2 * bvec[16] - 2 * bvec[17] +
                       2 * bvec[18] - 2 * bvec[19] + 1 * bvec[32] + 1 * bvec[33] + 1 * bvec[34] +
                       1 * bvec[35];
        alphavec[16] = +1 * bvec[24];
        alphavec[17] = +1 * bvec[40];
        alphavec[18] = -3 * bvec[24] + 3 * bvec[25] - 2 * bvec[40] - 1 * bvec[41];
        alphavec[19] = +2 * bvec[24] - 2 * bvec[25] + 1 * bvec[40] + 1 * bvec[41];
        alphavec[20] = +1 * bvec[48];
        alphavec[21] = +1 * bvec[56];
        alphavec[22] = -3 * bvec[48] + 3 * bvec[49] - 2 * bvec[56] - 1 * bvec[57];
        alphavec[23] = +2 * bvec[48] - 2 * bvec[49] + 1 * bvec[56] + 1 * bvec[57];
        alphavec[24] = -3 * bvec[24] + 3 * bvec[26] - 2 * bvec[48] - 1 * bvec[50];
        alphavec[25] = -3 * bvec[40] + 3 * bvec[42] - 2 * bvec[56] - 1 * bvec[58];
        alphavec[26] = +9 * bvec[24] - 9 * bvec[25] - 9 * bvec[26] + 9 * bvec[27] + 6 * bvec[40] +
                       3 * bvec[41] - 6 * bvec[42] - 3 * bvec[43] + 6 * bvec[48] - 6 * bvec[49] +
                       3 * bvec[50] - 3 * bvec[51] + 4 * bvec[56] + 2 * bvec[57] + 2 * bvec[58] +
                       1 * bvec[59];
        alphavec[27] = -6 * bvec[24] + 6 * bvec[25] + 6 * bvec[26] - 6 * bvec[27] - 3 * bvec[40] -
                       3 * bvec[41] + 3 * bvec[42] + 3 * bvec[43] - 4 * bvec[48] + 4 * bvec[49] -
                       2 * bvec[50] + 2 * bvec[51] - 2 * bvec[56] - 2 * bvec[57] - 1 * bvec[58] -
                       1 * bvec[59];
        alphavec[28] = +2 * bvec[24] - 2 * bvec[26] + 1 * bvec[48] + 1 * bvec[50];
        alphavec[29] = +2 * bvec[40] - 2 * bvec[42] + 1 * bvec[56] + 1 * bvec[58];
        alphavec[30] = -6 * bvec[24] + 6 * bvec[25] + 6 * bvec[26] - 6 * bvec[27] - 4 * bvec[40] -
                       2 * bvec[41] + 4 * bvec[42] + 2 * bvec[43] - 3 * bvec[48] + 3 * bvec[49] -
                       3 * bvec[50] + 3 * bvec[51] - 2 * bvec[56] - 1 * bvec[57] - 2 * bvec[58] -
                       1 * bvec[59];
        alphavec[31] = +4 * bvec[24] - 4 * bvec[25] - 4 * bvec[26] + 4 * bvec[27] + 2 * bvec[40] +
                       2 * bvec[41] - 2 * bvec[42] - 2 * bvec[43] + 2 * bvec[48] - 2 * bvec[49] +
                       2 * bvec[50] - 2 * bvec[51] + 1 * bvec[56] + 1 * bvec[57] + 1 * bvec[58] +
                       1 * bvec[59];
        alphavec[32] = -3 * bvec[0] + 3 * bvec[4] - 2 * bvec[24] - 1 * bvec[28];
        alphavec[33] = -3 * bvec[8] + 3 * bvec[12] - 2 * bvec[40] - 1 * bvec[44];
        alphavec[34] = +9 * bvec[0] - 9 * bvec[1] - 9 * bvec[4] + 9 * bvec[5] + 6 * bvec[8] +
                       3 * bvec[9] - 6 * bvec[12] - 3 * bvec[13] + 6 * bvec[24] - 6 * bvec[25] +
                       3 * bvec[28] - 3 * bvec[29] + 4 * bvec[40] + 2 * bvec[41] + 2 * bvec[44] +
                       1 * bvec[45];
        alphavec[35] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[4] - 6 * bvec[5] - 3 * bvec[8] -
                       3 * bvec[9] + 3 * bvec[12] + 3 * bvec[13] - 4 * bvec[24] + 4 * bvec[25] -
                       2 * bvec[28] + 2 * bvec[29] - 2 * bvec[40] - 2 * bvec[41] - 1 * bvec[44] -
                       1 * bvec[45];
        alphavec[36] = -3 * bvec[16] + 3 * bvec[20] - 2 * bvec[48] - 1 * bvec[52];
        alphavec[37] = -3 * bvec[32] + 3 * bvec[36] - 2 * bvec[56] - 1 * bvec[60];
        alphavec[38] = +9 * bvec[16] - 9 * bvec[17] - 9 * bvec[20] + 9 * bvec[21] + 6 * bvec[32] +
                       3 * bvec[33] - 6 * bvec[36] - 3 * bvec[37] + 6 * bvec[48] - 6 * bvec[49] +
                       3 * bvec[52] - 3 * bvec[53] + 4 * bvec[56] + 2 * bvec[57] + 2 * bvec[60] +
                       1 * bvec[61];
        alphavec[39] = -6 * bvec[16] + 6 * bvec[17] + 6 * bvec[20] - 6 * bvec[21] - 3 * bvec[32] -
                       3 * bvec[33] + 3 * bvec[36] + 3 * bvec[37] - 4 * bvec[48] + 4 * bvec[49] -
                       2 * bvec[52] + 2 * bvec[53] - 2 * bvec[56] - 2 * bvec[57] - 1 * bvec[60] -
                       1 * bvec[61];
        alphavec[40] = +9 * bvec[0] - 9 * bvec[2] - 9 * bvec[4] + 9 * bvec[6] + 6 * bvec[16] +
                       3 * bvec[18] - 6 * bvec[20] - 3 * bvec[22] + 6 * bvec[24] - 6 * bvec[26] +
                       3 * bvec[28] - 3 * bvec[30] + 4 * bvec[48] + 2 * bvec[50] + 2 * bvec[52] +
                       1 * bvec[54];
        alphavec[41] = +9 * bvec[8] - 9 * bvec[10] - 9 * bvec[12] + 9 * bvec[14] + 6 * bvec[32] +
                       3 * bvec[34] - 6 * bvec[36] - 3 * bvec[38] + 6 * bvec[40] - 6 * bvec[42] +
                       3 * bvec[44] - 3 * bvec[46] + 4 * bvec[56] + 2 * bvec[58] + 2 * bvec[60] +
                       1 * bvec[62];
        alphavec[42] = -27 * bvec[0] + 27 * bvec[1] + 27 * bvec[2] - 27 * bvec[3] + 27 * bvec[4] -
                       27 * bvec[5] - 27 * bvec[6] + 27 * bvec[7] - 18 * bvec[8] - 9 * bvec[9] +
                       18 * bvec[10] + 9 * bvec[11] + 18 * bvec[12] + 9 * bvec[13] - 18 * bvec[14] -
                       9 * bvec[15] - 18 * bvec[16] + 18 * bvec[17] - 9 * bvec[18] + 9 * bvec[19] +
                       18 * bvec[20] - 18 * bvec[21] + 9 * bvec[22] - 9 * bvec[23] - 18 * bvec[24] +
                       18 * bvec[25] + 18 * bvec[26] - 18 * bvec[27] - 9 * bvec[28] + 9 * bvec[29] +
                       9 * bvec[30] - 9 * bvec[31] - 12 * bvec[32] - 6 * bvec[33] - 6 * bvec[34] -
                       3 * bvec[35] + 12 * bvec[36] + 6 * bvec[37] + 6 * bvec[38] + 3 * bvec[39] -
                       12 * bvec[40] - 6 * bvec[41] + 12 * bvec[42] + 6 * bvec[43] - 6 * bvec[44] -
                       3 * bvec[45] + 6 * bvec[46] + 3 * bvec[47] - 12 * bvec[48] + 12 * bvec[49] -
                       6 * bvec[50] + 6 * bvec[51] - 6 * bvec[52] + 6 * bvec[53] - 3 * bvec[54] +
                       3 * bvec[55] - 8 * bvec[56] - 4 * bvec[57] - 4 * bvec[58] - 2 * bvec[59] -
                       4 * bvec[60] - 2 * bvec[61] - 2 * bvec[62] - 1 * bvec[63];
        alphavec[43] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[4] +
                       18 * bvec[5] + 18 * bvec[6] - 18 * bvec[7] + 9 * bvec[8] + 9 * bvec[9] -
                       9 * bvec[10] - 9 * bvec[11] - 9 * bvec[12] - 9 * bvec[13] + 9 * bvec[14] +
                       9 * bvec[15] + 12 * bvec[16] - 12 * bvec[17] + 6 * bvec[18] - 6 * bvec[19] -
                       12 * bvec[20] + 12 * bvec[21] - 6 * bvec[22] + 6 * bvec[23] + 12 * bvec[24] -
                       12 * bvec[25] - 12 * bvec[26] + 12 * bvec[27] + 6 * bvec[28] - 6 * bvec[29] -
                       6 * bvec[30] + 6 * bvec[31] + 6 * bvec[32] + 6 * bvec[33] + 3 * bvec[34] +
                       3 * bvec[35] - 6 * bvec[36] - 6 * bvec[37] - 3 * bvec[38] - 3 * bvec[39] +
                       6 * bvec[40] + 6 * bvec[41] - 6 * bvec[42] - 6 * bvec[43] + 3 * bvec[44] +
                       3 * bvec[45] - 3 * bvec[46] - 3 * bvec[47] + 8 * bvec[48] - 8 * bvec[49] +
                       4 * bvec[50] - 4 * bvec[51] + 4 * bvec[52] - 4 * bvec[53] + 2 * bvec[54] -
                       2 * bvec[55] + 4 * bvec[56] + 4 * bvec[57] + 2 * bvec[58] + 2 * bvec[59] +
                       2 * bvec[60] + 2 * bvec[61] + 1 * bvec[62] + 1 * bvec[63];
        alphavec[44] = -6 * bvec[0] + 6 * bvec[2] + 6 * bvec[4] - 6 * bvec[6] - 3 * bvec[16] -
                       3 * bvec[18] + 3 * bvec[20] + 3 * bvec[22] - 4 * bvec[24] + 4 * bvec[26] -
                       2 * bvec[28] + 2 * bvec[30] - 2 * bvec[48] - 2 * bvec[50] - 1 * bvec[52] -
                       1 * bvec[54];
        alphavec[45] = -6 * bvec[8] + 6 * bvec[10] + 6 * bvec[12] - 6 * bvec[14] - 3 * bvec[32] -
                       3 * bvec[34] + 3 * bvec[36] + 3 * bvec[38] - 4 * bvec[40] + 4 * bvec[42] -
                       2 * bvec[44] + 2 * bvec[46] - 2 * bvec[56] - 2 * bvec[58] - 1 * bvec[60] -
                       1 * bvec[62];
        alphavec[46] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[4] +
                       18 * bvec[5] + 18 * bvec[6] - 18 * bvec[7] + 12 * bvec[8] + 6 * bvec[9] -
                       12 * bvec[10] - 6 * bvec[11] - 12 * bvec[12] - 6 * bvec[13] + 12 * bvec[14] +
                       6 * bvec[15] + 9 * bvec[16] - 9 * bvec[17] + 9 * bvec[18] - 9 * bvec[19] -
                       9 * bvec[20] + 9 * bvec[21] - 9 * bvec[22] + 9 * bvec[23] + 12 * bvec[24] -
                       12 * bvec[25] - 12 * bvec[26] + 12 * bvec[27] + 6 * bvec[28] - 6 * bvec[29] -
                       6 * bvec[30] + 6 * bvec[31] + 6 * bvec[32] + 3 * bvec[33] + 6 * bvec[34] +
                       3 * bvec[35] - 6 * bvec[36] - 3 * bvec[37] - 6 * bvec[38] - 3 * bvec[39] +
                       8 * bvec[40] + 4 * bvec[41] - 8 * bvec[42] - 4 * bvec[43] + 4 * bvec[44] +
                       2 * bvec[45] - 4 * bvec[46] - 2 * bvec[47] + 6 * bvec[48] - 6 * bvec[49] +
                       6 * bvec[50] - 6 * bvec[51] + 3 * bvec[52] - 3 * bvec[53] + 3 * bvec[54] -
                       3 * bvec[55] + 4 * bvec[56] + 2 * bvec[57] + 4 * bvec[58] + 2 * bvec[59] +
                       2 * bvec[60] + 1 * bvec[61] + 2 * bvec[62] + 1 * bvec[63];
        alphavec[47] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[4] -
                       12 * bvec[5] - 12 * bvec[6] + 12 * bvec[7] - 6 * bvec[8] - 6 * bvec[9] +
                       6 * bvec[10] + 6 * bvec[11] + 6 * bvec[12] + 6 * bvec[13] - 6 * bvec[14] -
                       6 * bvec[15] - 6 * bvec[16] + 6 * bvec[17] - 6 * bvec[18] + 6 * bvec[19] +
                       6 * bvec[20] - 6 * bvec[21] + 6 * bvec[22] - 6 * bvec[23] - 8 * bvec[24] +
                       8 * bvec[25] + 8 * bvec[26] - 8 * bvec[27] - 4 * bvec[28] + 4 * bvec[29] +
                       4 * bvec[30] - 4 * bvec[31] - 3 * bvec[32] - 3 * bvec[33] - 3 * bvec[34] -
                       3 * bvec[35] + 3 * bvec[36] + 3 * bvec[37] + 3 * bvec[38] + 3 * bvec[39] -
                       4 * bvec[40] - 4 * bvec[41] + 4 * bvec[42] + 4 * bvec[43] - 2 * bvec[44] -
                       2 * bvec[45] + 2 * bvec[46] + 2 * bvec[47] - 4 * bvec[48] + 4 * bvec[49] -
                       4 * bvec[50] + 4 * bvec[51] - 2 * bvec[52] + 2 * bvec[53] - 2 * bvec[54] +
                       2 * bvec[55] - 2 * bvec[56] - 2 * bvec[57] - 2 * bvec[58] - 2 * bvec[59] -
                       1 * bvec[60] - 1 * bvec[61] - 1 * bvec[62] - 1 * bvec[63];
        alphavec[48] = +2 * bvec[0] - 2 * bvec[4] + 1 * bvec[24] + 1 * bvec[28];
        alphavec[49] = +2 * bvec[8] - 2 * bvec[12] + 1 * bvec[40] + 1 * bvec[44];
        alphavec[50] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[4] - 6 * bvec[5] - 4 * bvec[8] -
                       2 * bvec[9] + 4 * bvec[12] + 2 * bvec[13] - 3 * bvec[24] + 3 * bvec[25] -
                       3 * bvec[28] + 3 * bvec[29] - 2 * bvec[40] - 1 * bvec[41] - 2 * bvec[44] -
                       1 * bvec[45];
        alphavec[51] = +4 * bvec[0] - 4 * bvec[1] - 4 * bvec[4] + 4 * bvec[5] + 2 * bvec[8] +
                       2 * bvec[9] - 2 * bvec[12] - 2 * bvec[13] + 2 * bvec[24] - 2 * bvec[25] +
                       2 * bvec[28] - 2 * bvec[29] + 1 * bvec[40] + 1 * bvec[41] + 1 * bvec[44] +
                       1 * bvec[45];
        alphavec[52] = +2 * bvec[16] - 2 * bvec[20] + 1 * bvec[48] + 1 * bvec[52];
        alphavec[53] = +2 * bvec[32] - 2 * bvec[36] + 1 * bvec[56] + 1 * bvec[60];
        alphavec[54] = -6 * bvec[16] + 6 * bvec[17] + 6 * bvec[20] - 6 * bvec[21] - 4 * bvec[32] -
                       2 * bvec[33] + 4 * bvec[36] + 2 * bvec[37] - 3 * bvec[48] + 3 * bvec[49] -
                       3 * bvec[52] + 3 * bvec[53] - 2 * bvec[56] - 1 * bvec[57] - 2 * bvec[60] -
                       1 * bvec[61];
        alphavec[55] = +4 * bvec[16] - 4 * bvec[17] - 4 * bvec[20] + 4 * bvec[21] + 2 * bvec[32] +
                       2 * bvec[33] - 2 * bvec[36] - 2 * bvec[37] + 2 * bvec[48] - 2 * bvec[49] +
                       2 * bvec[52] - 2 * bvec[53] + 1 * bvec[56] + 1 * bvec[57] + 1 * bvec[60] +
                       1 * bvec[61];
        alphavec[56] = -6 * bvec[0] + 6 * bvec[2] + 6 * bvec[4] - 6 * bvec[6] - 4 * bvec[16] -
                       2 * bvec[18] + 4 * bvec[20] + 2 * bvec[22] - 3 * bvec[24] + 3 * bvec[26] -
                       3 * bvec[28] + 3 * bvec[30] - 2 * bvec[48] - 1 * bvec[50] - 2 * bvec[52] -
                       1 * bvec[54];
        alphavec[57] = -6 * bvec[8] + 6 * bvec[10] + 6 * bvec[12] - 6 * bvec[14] - 4 * bvec[32] -
                       2 * bvec[34] + 4 * bvec[36] + 2 * bvec[38] - 3 * bvec[40] + 3 * bvec[42] -
                       3 * bvec[44] + 3 * bvec[46] - 2 * bvec[56] - 1 * bvec[58] - 2 * bvec[60] -
                       1 * bvec[62];
        alphavec[58] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[4] +
                       18 * bvec[5] + 18 * bvec[6] - 18 * bvec[7] + 12 * bvec[8] + 6 * bvec[9] -
                       12 * bvec[10] - 6 * bvec[11] - 12 * bvec[12] - 6 * bvec[13] + 12 * bvec[14] +
                       6 * bvec[15] + 12 * bvec[16] - 12 * bvec[17] + 6 * bvec[18] - 6 * bvec[19] -
                       12 * bvec[20] + 12 * bvec[21] - 6 * bvec[22] + 6 * bvec[23] + 9 * bvec[24] -
                       9 * bvec[25] - 9 * bvec[26] + 9 * bvec[27] + 9 * bvec[28] - 9 * bvec[29] -
                       9 * bvec[30] + 9 * bvec[31] + 8 * bvec[32] + 4 * bvec[33] + 4 * bvec[34] +
                       2 * bvec[35] - 8 * bvec[36] - 4 * bvec[37] - 4 * bvec[38] - 2 * bvec[39] +
                       6 * bvec[40] + 3 * bvec[41] - 6 * bvec[42] - 3 * bvec[43] + 6 * bvec[44] +
                       3 * bvec[45] - 6 * bvec[46] - 3 * bvec[47] + 6 * bvec[48] - 6 * bvec[49] +
                       3 * bvec[50] - 3 * bvec[51] + 6 * bvec[52] - 6 * bvec[53] + 3 * bvec[54] -
                       3 * bvec[55] + 4 * bvec[56] + 2 * bvec[57] + 2 * bvec[58] + 1 * bvec[59] +
                       4 * bvec[60] + 2 * bvec[61] + 2 * bvec[62] + 1 * bvec[63];
        alphavec[59] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[4] -
                       12 * bvec[5] - 12 * bvec[6] + 12 * bvec[7] - 6 * bvec[8] - 6 * bvec[9] +
                       6 * bvec[10] + 6 * bvec[11] + 6 * bvec[12] + 6 * bvec[13] - 6 * bvec[14] -
                       6 * bvec[15] - 8 * bvec[16] + 8 * bvec[17] - 4 * bvec[18] + 4 * bvec[19] +
                       8 * bvec[20] - 8 * bvec[21] + 4 * bvec[22] - 4 * bvec[23] - 6 * bvec[24] +
                       6 * bvec[25] + 6 * bvec[26] - 6 * bvec[27] - 6 * bvec[28] + 6 * bvec[29] +
                       6 * bvec[30] - 6 * bvec[31] - 4 * bvec[32] - 4 * bvec[33] - 2 * bvec[34] -
                       2 * bvec[35] + 4 * bvec[36] + 4 * bvec[37] + 2 * bvec[38] + 2 * bvec[39] -
                       3 * bvec[40] - 3 * bvec[41] + 3 * bvec[42] + 3 * bvec[43] - 3 * bvec[44] -
                       3 * bvec[45] + 3 * bvec[46] + 3 * bvec[47] - 4 * bvec[48] + 4 * bvec[49] -
                       2 * bvec[50] + 2 * bvec[51] - 4 * bvec[52] + 4 * bvec[53] - 2 * bvec[54] +
                       2 * bvec[55] - 2 * bvec[56] - 2 * bvec[57] - 1 * bvec[58] - 1 * bvec[59] -
                       2 * bvec[60] - 2 * bvec[61] - 1 * bvec[62] - 1 * bvec[63];
        alphavec[60] = +4 * bvec[0] - 4 * bvec[2] - 4 * bvec[4] + 4 * bvec[6] + 2 * bvec[16] +
                       2 * bvec[18] - 2 * bvec[20] - 2 * bvec[22] + 2 * bvec[24] - 2 * bvec[26] +
                       2 * bvec[28] - 2 * bvec[30] + 1 * bvec[48] + 1 * bvec[50] + 1 * bvec[52] +
                       1 * bvec[54];
        alphavec[61] = +4 * bvec[8] - 4 * bvec[10] - 4 * bvec[12] + 4 * bvec[14] + 2 * bvec[32] +
                       2 * bvec[34] - 2 * bvec[36] - 2 * bvec[38] + 2 * bvec[40] - 2 * bvec[42] +
                       2 * bvec[44] - 2 * bvec[46] + 1 * bvec[56] + 1 * bvec[58] + 1 * bvec[60] +
                       1 * bvec[62];
        alphavec[62] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[4] -
                       12 * bvec[5] - 12 * bvec[6] + 12 * bvec[7] - 8 * bvec[8] - 4 * bvec[9] +
                       8 * bvec[10] + 4 * bvec[11] + 8 * bvec[12] + 4 * bvec[13] - 8 * bvec[14] -
                       4 * bvec[15] - 6 * bvec[16] + 6 * bvec[17] - 6 * bvec[18] + 6 * bvec[19] +
                       6 * bvec[20] - 6 * bvec[21] + 6 * bvec[22] - 6 * bvec[23] - 6 * bvec[24] +
                       6 * bvec[25] + 6 * bvec[26] - 6 * bvec[27] - 6 * bvec[28] + 6 * bvec[29] +
                       6 * bvec[30] - 6 * bvec[31] - 4 * bvec[32] - 2 * bvec[33] - 4 * bvec[34] -
                       2 * bvec[35] + 4 * bvec[36] + 2 * bvec[37] + 4 * bvec[38] + 2 * bvec[39] -
                       4 * bvec[40] - 2 * bvec[41] + 4 * bvec[42] + 2 * bvec[43] - 4 * bvec[44] -
                       2 * bvec[45] + 4 * bvec[46] + 2 * bvec[47] - 3 * bvec[48] + 3 * bvec[49] -
                       3 * bvec[50] + 3 * bvec[51] - 3 * bvec[52] + 3 * bvec[53] - 3 * bvec[54] +
                       3 * bvec[55] - 2 * bvec[56] - 1 * bvec[57] - 2 * bvec[58] - 1 * bvec[59] -
                       2 * bvec[60] - 1 * bvec[61] - 2 * bvec[62] - 1 * bvec[63];
        alphavec[63] = +8 * bvec[0] - 8 * bvec[1] - 8 * bvec[2] + 8 * bvec[3] - 8 * bvec[4] +
                       8 * bvec[5] + 8 * bvec[6] - 8 * bvec[7] + 4 * bvec[8] + 4 * bvec[9] -
                       4 * bvec[10] - 4 * bvec[11] - 4 * bvec[12] - 4 * bvec[13] + 4 * bvec[14] +
                       4 * bvec[15] + 4 * bvec[16] - 4 * bvec[17] + 4 * bvec[18] - 4 * bvec[19] -
                       4 * bvec[20] + 4 * bvec[21] - 4 * bvec[22] + 4 * bvec[23] + 4 * bvec[24] -
                       4 * bvec[25] - 4 * bvec[26] + 4 * bvec[27] + 4 * bvec[28] - 4 * bvec[29] -
                       4 * bvec[30] + 4 * bvec[31] + 2 * bvec[32] + 2 * bvec[33] + 2 * bvec[34] +
                       2 * bvec[35] - 2 * bvec[36] - 2 * bvec[37] - 2 * bvec[38] - 2 * bvec[39] +
                       2 * bvec[40] + 2 * bvec[41] - 2 * bvec[42] - 2 * bvec[43] + 2 * bvec[44] +
                       2 * bvec[45] - 2 * bvec[46] - 2 * bvec[47] + 2 * bvec[48] - 2 * bvec[49] +
                       2 * bvec[50] - 2 * bvec[51] + 2 * bvec[52] - 2 * bvec[53] + 2 * bvec[54] -
                       2 * bvec[55] + 1 * bvec[56] + 1 * bvec[57] + 1 * bvec[58] + 1 * bvec[59] +
                       1 * bvec[60] + 1 * bvec[61] + 1 * bvec[62] + 1 * bvec[63];

        return alphavec;
    }

    /// @internal
    /// @brief Return a cell's tricubic coefficient vector (cached or freshly computed).
    /// @param xelem  X-interval index.
    /// @param yelem  Y-interval index.
    /// @param zelem  Z-interval index.
    /// @return The tricubic coefficient vector for the cell.
    /// @endinternal
    Eigen::Matrix<double, 64, 1> get_alphavec(int xelem, int yelem, int zelem) const {
        if (this->cache_alpha_) {
            return this->alphavecs_(xelem, yelem, zelem);
        } else {
            return this->calc_alphavec(xelem, yelem, zelem);
        }
    }

    /// @internal
    /// @brief Core interpolation kernel writing the value and requested derivatives.
    /// @param x       Query x coordinate.
    /// @param y       Query y coordinate.
    /// @param z       Query z coordinate.
    /// @param deriv   Highest derivative order to compute (0, 1, or 2).
    /// @param fval    Output interpolated value.
    /// @param dfxyz   Output gradient (if @p deriv > 0).
    /// @param d2fxyz  Output Hessian (if @p deriv > 1).
    /// @throws std::invalid_argument if @p x, @p y, or @p z is outside the table range.
    /// @endinternal
    void interp_impl(double x, double y, double z, int deriv, double &fval,
                     Eigen::Vector3<double> &dfxyz, Eigen::Matrix3<double> &d2fxyz) const {

        double xeps = std::numeric_limits<double>::epsilon() * xtotal_;
        if (x < (xs_[0] - xeps) || x > (xs_[xs_.size() - 1] + xeps)) {
            throw std::invalid_argument(
                fmt::format("InterpTable3D: query x={} is outside table x range [{}, {}]", x,
                            xs_[0], xs_[xs_.size() - 1]));
        }
        double yeps = std::numeric_limits<double>::epsilon() * ytotal_;
        if (y < (ys_[0] - yeps) || y > (ys_[ys_.size() - 1]) + yeps) {
            throw std::invalid_argument(
                fmt::format("InterpTable3D: query y={} is outside table y range [{}, {}]", y,
                            ys_[0], ys_[ys_.size() - 1]));
        }
        double zeps = std::numeric_limits<double>::epsilon() * ztotal_;
        if (z < (zs_[0] - zeps) || z > (zs_[zs_.size() - 1]) + zeps) {
            throw std::invalid_argument(
                fmt::format("InterpTable3D: query z={} is outside table z range [{}, {}]", z,
                            zs_[0], zs_[zs_.size() - 1]));
        }

        auto [xelem, yelem, zelem] = get_xyzelems(x, y, z);

        double xstep = xs_[xelem + 1] - xs_[xelem];
        double ystep = ys_[yelem + 1] - ys_[yelem];
        double zstep = zs_[zelem + 1] - zs_[zelem];

        double xf = (x - xs_[xelem]) / xstep;
        double yf = (y - ys_[yelem]) / ystep;
        double zf = (z - zs_[zelem]) / zstep;

        if (this->interp_kind_ == InterpType::Cubic) {
            Eigen::Matrix<double, 64, 1> alphavec = this->get_alphavec(xelem, yelem, zelem);

            double xf2 = xf * xf;
            double xf3 = xf2 * xf;

            double yf2 = yf * yf;
            double yf3 = yf2 * yf;

            double zf2 = zf * zf;
            double zf3 = zf2 * zf;

            Eigen::Vector4d yfs{1, yf, yf2, yf3};
            Eigen::Vector4d zfs{1, zf, zf2, zf3};

            fval = 0;

            for (int i = 0, start = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++, start += 4) {
                    fval +=
                        (yfs[j] * zfs[i]) * (alphavec[start] + xf * alphavec[start + 1] +
                                             xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3]);
                }
            }

            if (deriv > 0) {

                dfxyz[0] = 0;
                dfxyz[1] = 0;
                dfxyz[2] = 0;

                for (int i = 0, start = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++, start += 4) {
                        double xterm = alphavec[start] + xf * alphavec[start + 1] +
                                       xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3];
                        double dxterm = alphavec[start + 1] + 2 * xf * alphavec[start + 2] +
                                        3 * xf2 * alphavec[start + 3];
                        dfxyz[0] += (yfs[j] * zfs[i]) * (dxterm);
                        if (j > 0) {
                            dfxyz[1] += (j * yfs[j - 1] * zfs[i]) * (xterm);
                        }
                        if (i > 0) {
                            dfxyz[2] += (yfs[j] * i * zfs[i - 1]) * (xterm);
                        }
                    }
                }

                dfxyz[0] /= xstep;
                dfxyz[1] /= ystep;
                dfxyz[2] /= zstep;

                if (deriv > 1) {
                    Eigen::DiagonalMatrix<double, 3> dmat(1.0 / xstep, 1.0 / ystep, 1.0 / zstep);

                    d2fxyz.setZero();

                    for (int i = 0, start = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++, start += 4) {

                            double xterm = alphavec[start] + xf * alphavec[start + 1] +
                                           xf2 * alphavec[start + 2] + xf3 * alphavec[start + 3];
                            double dxterm = alphavec[start + 1] + 2 * xf * alphavec[start + 2] +
                                            3 * xf2 * alphavec[start + 3];

                            // Hit first row of hessian, diffing this term
                            // dfxyz[0] += (yfs[j] * zfs[i]) * (dxterm);
                            // wrt.x
                            d2fxyz(0, 0) += (yfs[j] * zfs[i]) * (2 * alphavec[start + 2] +
                                                                 6 * xf * alphavec[start + 3]);
                            // wrt.y
                            if (j > 0) {
                                d2fxyz(0, 1) += (j * yfs[j - 1] * zfs[i]) * (dxterm);
                            }
                            // wrt.z
                            if (i > 0) {
                                d2fxyz(0, 2) += (yfs[j] * i * zfs[i - 1]) * (dxterm);
                            }

                            if (j > 0) {
                                // Hit second row of hessian, diffing this term
                                // dfxyz[1] += (j * yfs[j - 1] * zfs[i]) * (xterm);
                                // wrt.y
                                if (j > 1) {
                                    d2fxyz(1, 1) += (j * (j - 1) * yfs[j - 2] * zfs[i]) * (xterm);
                                }
                                // wrt.z
                                if (i > 0) {
                                    d2fxyz(1, 2) += (j * yfs[j - 1] * i * zfs[i - 1]) * (xterm);
                                }
                            }
                            if (i > 0) {
                                // Hit third row of hessian, diffing this term
                                // dfxyz[2] += (yfs[j] * i * zfs[i - 1]) * (xterm);
                                if (i > 1) {
                                    d2fxyz(2, 2) += (yfs[j] * i * (i - 1) * zfs[i - 2]) * (xterm);
                                }
                            }
                        }
                    }

                    // symmetric
                    d2fxyz(1, 0) = d2fxyz(0, 1);
                    d2fxyz(2, 0) = d2fxyz(0, 2);
                    d2fxyz(2, 1) = d2fxyz(1, 2);

                    d2fxyz = (dmat * d2fxyz * dmat).eval();
                }
            }

        } else {
            double c000 = this->fs_(xelem, yelem, zelem);
            double c100 = this->fs_(xelem + 1, yelem, zelem);

            double c001 = this->fs_(xelem, yelem, zelem + 1);
            double c101 = this->fs_(xelem + 1, yelem, zelem + 1);

            double c010 = this->fs_(xelem, yelem + 1, zelem);
            double c110 = this->fs_(xelem + 1, yelem + 1, zelem);

            double c011 = this->fs_(xelem, yelem + 1, zelem + 1);
            double c111 = this->fs_(xelem + 1, yelem + 1, zelem + 1);

            double x0 = this->xs_[xelem];
            double x1 = this->xs_[xelem + 1];

            double y0 = this->ys_[yelem];
            double y1 = this->ys_[yelem + 1];

            double z0 = this->zs_[zelem];
            double z1 = this->zs_[zelem + 1];

            double scale = -1.0 / (xstep * ystep * zstep);

            double a0 = (-c000 * x1 * y1 * z1 + c001 * x1 * y1 * z0 + c010 * x1 * y0 * z1 -
                         c011 * x1 * y0 * z0 + c100 * x0 * y1 * z1 - c101 * x0 * y1 * z0 -
                         c110 * x0 * y0 * z1 + c111 * x0 * y0 * z0) *
                        scale;

            double a1 = (c000 * y1 * z1 - c001 * y1 * z0 - c010 * y0 * z1 + c011 * y0 * z0 -
                         c100 * y1 * z1 + c101 * y1 * z0 + c110 * y0 * z1 - c111 * y0 * z0) *
                        scale;

            double a2 = (c000 * x1 * z1 - c001 * x1 * z0 - c010 * x1 * z1 + c011 * x1 * z0 -
                         c100 * x0 * z1 + c101 * x0 * z0 + c110 * x0 * z1 - c111 * x0 * z0) *
                        scale;

            double a3 = (c000 * x1 * y1 - c001 * x1 * y1 - c010 * x1 * y0 + c011 * x1 * y0 -
                         c100 * x0 * y1 + c101 * x0 * y1 + c110 * x0 * y0 - c111 * x0 * y0) *
                        scale;

            double a4 = (-c000 * z1 + c001 * z0 + c010 * z1 - c011 * z0 + c100 * z1 - c101 * z0 -
                         c110 * z1 + c111 * z0) *
                        scale;
            double a5 = (-c000 * y1 + c001 * y1 + c010 * y0 - c011 * y0 + c100 * y1 - c101 * y1 -
                         c110 * y0 + c111 * y0) *
                        scale;
            double a6 = (-c000 * x1 + c001 * x1 + c010 * x1 - c011 * x1 + c100 * x0 - c101 * x0 -
                         c110 * x0 + c111 * x0) *
                        scale;

            double a7 = (c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111) * scale;

            fval = a0 + a1 * x + a2 * y + a3 * z + a4 * x * y + a5 * x * z + a6 * y * z +
                   a7 * x * y * z;

            if (deriv > 0) {
                dfxyz[0] = a1 + a4 * y + a5 * z + a7 * y * z;
                dfxyz[1] = a2 + a4 * x + a6 * z + a7 * x * z;
                dfxyz[2] = a3 + a5 * x + a6 * y + a7 * x * y;

                if (deriv > 1) {
                    d2fxyz.setZero();
                    d2fxyz(1, 0) = a4 + a7 * z;
                    d2fxyz(2, 0) = a5 + a7 * y;
                    d2fxyz(2, 1) = a6 + a7 * x;
                    d2fxyz(0, 1) = d2fxyz(1, 0);
                    d2fxyz(0, 2) = d2fxyz(2, 0);
                    d2fxyz(1, 2) = d2fxyz(2, 1);
                }
            }
        }
    }

    /// @brief Interpolate the field value at a point.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @param z  Query z coordinate.
    /// @return The interpolated value.
    double interp(double x, double y, double z) const {
        double f;
        Eigen::Vector3<double> dfxyz;
        Eigen::Matrix3<double> d2fxyz;
        interp_impl(x, y, z, 0, f, dfxyz, d2fxyz);
        return f;
    }

    /// @brief Interpolate the value and gradient at a point.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @param z  Query z coordinate.
    /// @return Tuple of {value, gradient}.
    std::tuple<double, Eigen::Vector3<double>> interp_deriv1(double x, double y, double z) const {
        double f;
        Eigen::Vector3<double> dfxyz;
        Eigen::Matrix3<double> d2fxyz;
        interp_impl(x, y, z, 1, f, dfxyz, d2fxyz);
        return std::tuple{f, dfxyz};
    }
    /// @brief Interpolate the value, gradient, and Hessian at a point.
    /// @param x  Query x coordinate.
    /// @param y  Query y coordinate.
    /// @param z  Query z coordinate.
    /// @return Tuple of {value, gradient, Hessian}.
    std::tuple<double, Eigen::Vector3<double>, Eigen::Matrix3<double>>
    interp_deriv2(double x, double y, double z) const {
        double f;
        Eigen::Vector3<double> dfxyz;
        Eigen::Matrix3<double> d2fxyz;
        interp_impl(x, y, z, 2, f, dfxyz, d2fxyz);
        return std::tuple{f, dfxyz, d2fxyz};
    }
};

/// @ingroup optimal_control
/// @brief VectorFunction wrapping an @ref InterpTable3D as a function of @f$(x,y,z)@f$.
///
/// Maps a 3-vector input to the interpolated scalar field value, with analytic
/// gradient and Hessian, for composition into VectorFunction expressions.
struct InterpFunction3D : VectorFunction<InterpFunction3D, 3, 1, DenseDerivativeMode::Analytic,
                                         DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<InterpFunction3D, 3, 1, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    std::shared_ptr<InterpTable3D> tab; ///< The interpolation table being wrapped.

    /// @brief Default constructor; leaves the table unset.
    InterpFunction3D() {}
    /// @brief Construct from an interpolation table.
    /// @param tab  The table to wrap.
    InterpFunction3D(std::shared_ptr<InterpTable3D> tab) : tab(tab) { this->set_io_rows(3, 1); }

    /// @internal
    /// @brief Evaluate the interpolated value at the input point.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input @c [x, y, z].
    /// @param fx_  Output value (1-vector) to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx[0] = this->tab->interp(x[0], x[1], x[2]);
    }
    /// @internal
    /// @brief Evaluate the interpolated value and gradient (Jacobian).
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input @c [x, y, z].
    /// @param fx_  Output value (1-vector) to write.
    /// @param jx_  Output Jacobian (gradient) to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto [z, dzdx] = this->tab->interp_deriv1(x[0], x[1], x[2]);
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
    /// @param x        Input @c [x, y, z].
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

        auto [z, dzdx, d2zdx] = this->tab->interp_deriv2(x[0], x[1], x[2]);
        fx[0] = z;
        jx = dzdx.transpose();
        adjgrad = adjvars[0] * dzdx;
        adjhess = adjvars[0] * d2zdx;
    }
};

} // namespace tycho::oc