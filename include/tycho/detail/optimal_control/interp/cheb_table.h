// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/vector_function.h"
#include <pocketfft_hdronly.h>
#include <Eigen/Dense>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace tycho::oc {

using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::MatRef;
using vf::VecRef;
using vf::VectorFunction;

/// @ingroup optimal_control
/// @brief Rank-generic Chebyshev (2nd-kind / DCT-I) interpolant data class.
///
/// Stores Chebyshev-T coefficients per output channel in a flat row-major
/// tensor layout.  For dimension @c dim_==1, the class reproduces the exact
/// Phase-1 1-D behaviour: @ref from_values (olen×(n+1) overload), @ref eval(double),
/// @ref eval_deriv1, @ref eval_deriv2, and @ref coeff_tail_norm all behave
/// identically to the Phase-1 implementation.
///
/// **Storage layout (generic):**
/// Coefficient data is stored as flat `Eigen::MatrixXd` of shape `(tsize, olen)`
/// where `tsize = ∏(orderᵢ+1)` and rows are row-major multi-indices over the
/// grid axes (last axis fastest).  `value_` holds the value coefficients;
/// `grad_[j]` holds the first-derivative coefficients along axis @c j (zero-
/// extended to the same tsize); `hess_[hess_index(j,k,dim_)]` holds the
/// mixed-derivative coefficients (upper triangle, `j≤k`), also zero-extended.
///
/// Derivative-series coefficients are precomputed once in @ref from_values
/// so that @ref eval_impl performs no per-call allocation.
struct ChebTable {
    using MatType = Eigen::Matrix<double, -1, -1>;

  private:
    // -------------------------------------------------------------------------
    // Rank-generic members (Task 0)
    // -------------------------------------------------------------------------
    int dim_ = 1;                      ///< Number of input dimensions.
    std::vector<int> orders_;          ///< Polynomial order per axis n_d.
    Eigen::VectorXd lb_, ub_;          ///< Domain bounds, length dim_.
    std::vector<int> shape_;           ///< Grid shape: shape_[d] = orders_[d] + 1.
    std::vector<long> strides_;        ///< Row-major strides over shape_.
    long tsize_ = 1;                   ///< Total grid size = prod(shape_).
    int olen_ = 0;                     ///< Output dimension.

    MatType value_;                    ///< tsize_ × olen_: value Chebyshev coefficients.
    std::vector<MatType> grad_;        ///< dim_ tensors (G_j = D_j value_), each tsize_ × olen_.
    std::vector<MatType> hess_;        ///< dim_*(dim_+1)/2 tensors (H_jk, j≤k), tsize_ × olen_.

    // -------------------------------------------------------------------------
    // 1-D domain helpers (unchanged from Phase 1; used only for dim_==1 path).
    // A per-axis N-D analog of this mapping is added in a later task, so these
    // 1-D helpers coexist with the N-D path rather than being replaced.
    // -------------------------------------------------------------------------

    // Map physical t -> xi in [-1,1]. Queries outside [lb_, ub_] are CLAMPED to
    // the nearest endpoint (no extrapolation): a global Chebyshev polynomial
    // diverges catastrophically outside its domain, so clamping is deliberate.
    // This differs from InterpTable1D, which extrapolates. Also guards end round-off.
    double to_xi(double t) const {
        double xi = (2.0 * t - lb_[0] - ub_[0]) / (ub_[0] - lb_[0]);
        return std::clamp(xi, -1.0, 1.0);
    }
    double half_span() const { return 0.5 * (ub_[0] - lb_[0]); }

    // -------------------------------------------------------------------------
    // Static helpers (reused by both 1-D and N-D paths)
    // -------------------------------------------------------------------------

    // Clenshaw eval of sum_{k=0}^{order} c[k] T_k(xi), c = coeffs column slice.
    static double clenshaw(const Eigen::Ref<const Eigen::VectorXd> &c, int order, double xi) {
        double b1 = 0.0, b2 = 0.0;
        for (int k = order; k >= 1; --k) {
            double bk = 2.0 * xi * b1 - b2 + c[k];
            b2 = b1;
            b1 = bk;
        }
        return c[0] + xi * b1 - b2;
    }

    // Derivative-series coefficients g[0..order-1] for d/dxi of c[0..order].
    // Returns a vector of length max(order, 1).
    static Eigen::VectorXd deriv_coeffs_from(const Eigen::Ref<const Eigen::VectorXd> &c,
                                              int order) {
        Eigen::VectorXd g = Eigen::VectorXd::Zero(std::max(order, 1));
        Eigen::VectorXd gp = Eigen::VectorXd::Zero(order + 2);
        for (int k = order; k >= 1; --k) gp[k - 1] = gp[k + 1] + 2.0 * double(k) * c[k];
        for (int k = 0; k < order; ++k) g[k] = gp[k];
        if (order >= 1) g[0] *= 0.5;  // convert prime-convention T_0 term
        return g;
    }

  public:
    ChebTable() {}

    // -------------------------------------------------------------------------
    // Public static index helpers (Task 0)
    // -------------------------------------------------------------------------

    /// @brief Compute row-major strides for the given shape vector.
    /// Stride for axis d = product of shape[d+1..D-1]; last axis stride = 1.
    static std::vector<long> row_major_strides(const std::vector<int> &shape) {
        std::vector<long> s(shape.size());
        long acc = 1;
        for (int d = int(shape.size()) - 1; d >= 0; --d) {
            s[d] = acc;
            acc *= shape[d];
        }
        return s;
    }

    /// @brief Compute flat row-major index from a multi-index and precomputed strides.
    static long flat_index(const std::vector<int> &mi, const std::vector<long> &strides) {
        long f = 0;
        for (size_t d = 0; d < mi.size(); ++d) f += long(mi[d]) * strides[d];
        return f;
    }

    /// @brief Recover the multi-index from a flat row-major index.
    static void unflatten(long flat, const std::vector<int> &shape,
                          const std::vector<long> &strides, std::vector<int> &mi) {
        mi.assign(shape.size(), 0);
        for (size_t d = 0; d < shape.size(); ++d) {
            mi[d] = int(flat / strides[d]);
            flat %= strides[d];
        }
    }

    /// @brief Index into the packed upper-triangle Hessian tensor array (j≤k).
    /// For j > k the arguments are swapped (symmetric). Returns the 0-based index.
    static int hess_index(int j, int k, int dim) {
        if (j > k) std::swap(j, k);
        return j * dim - (j * (j - 1)) / 2 + (k - j);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    int input_dim() const { return dim_; }
    int output_dim() const { return olen_; }
    int order() const { return orders_.empty() ? 0 : orders_[0]; }
    const std::vector<int> &orders() const { return orders_; }

    // -------------------------------------------------------------------------
    // N-D private building blocks (Task 1)
    // -------------------------------------------------------------------------

    /// @brief Apply the 1-D DCT-I coefficient transform along one axis of a
    /// flat @c (tsize×olen) tensor, in place per output channel.
    ///
    /// Normalization is identical to the 1-D @ref from_values path:
    ///   - divide each element by @c n (= order along the axis),
    ///   - halve the first and last elements of each line.
    ///
    /// A "line" is a contiguous sequence of @c (n+1) elements along axis @c ax.
    /// Its start in the flat buffer is any flat index @c start such that
    /// the coordinate of @c start along @c ax is zero, i.e.
    ///   @code (start / stride) % shape[ax] == 0 @endcode.
    static void coeffs_along_axis(MatType &tens, const std::vector<int> &shape,
                                  const std::vector<long> &strides, int ax, int nthreads) {
        const int n = shape[ax] - 1;                // polynomial order along this axis
        const long stride = strides[ax];
        const long tsize = tens.rows();
        const int olen = int(tens.cols());
        const size_t N1 = size_t(n) + 1;
        const pocketfft::shape_t pshape{N1};
        const pocketfft::stride_t pstride{ptrdiff_t(sizeof(double))};
        const pocketfft::shape_t axes{0};
        std::vector<double> in(N1), yk(N1);
        // Iterate over all lines along `ax`.  A flat index is the start of a
        // line iff its `ax`-coordinate is 0: (start / stride) % shape[ax] == 0.
        for (long start = 0; start < tsize; ++start) {
            if ((start / stride) % long(shape[ax]) != 0) continue;  // not a line start
            for (int c = 0; c < olen; ++c) {
                for (int k = 0; k <= n; ++k) in[size_t(k)] = tens(start + long(k) * stride, c);
                pocketfft::dct(pshape, pstride, pstride, axes, /*type=*/1, in.data(), yk.data(),
                               /*fct=*/1.0, /*ortho=*/false, size_t(nthreads));
                for (int k = 0; k <= n; ++k) tens(start + long(k) * stride, c) = yk[size_t(k)] / double(n);
                tens(start, c) *= 0.5;
                tens(start + long(n) * stride, c) *= 0.5;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 1-D node generation (unchanged Phase-1 API)
    // -------------------------------------------------------------------------

    /// @brief 2nd-kind node coordinates on [lb,ub], j=0..n (t_0=ub .. t_n=lb).
    static Eigen::VectorXd cheb_points(int n, double lb, double ub) {
        if (n < 1) throw std::invalid_argument("ChebTable: order must be >= 1");
        const double m = 0.5 * (lb + ub), h = 0.5 * (ub - lb);
        Eigen::VectorXd t(n + 1);
        for (int j = 0; j <= n; ++j)
            t[j] = m + h * std::cos(double(j) * std::numbers::pi / double(n));
        return t;
    }

    // -------------------------------------------------------------------------
    // 1-D from_values (Phase-1 API, re-homed onto generic members)
    // -------------------------------------------------------------------------

    /// @brief Build coefficients from grid values (olen × (n+1)) via DCT-I.
    ///
    /// Derivative-series coefficients (first and second order) are precomputed
    /// here and stored in @c grad_[0] and @c hess_[0] (zero-extended to tsize
    /// rows), so that @ref eval_impl incurs no per-call allocation during
    /// solve-time evaluation.
    ///
    /// @param grid_values  Row-major matrix of shape (olen, order+1).
    /// @param lb           Lower bound of the physical domain.
    /// @param ub           Upper bound of the physical domain.
    /// @param order        Polynomial order n; the table uses n+1 nodes.
    /// @param nthreads     Thread count forwarded to pocketfft for the DCT-I.
    ///                     Defaults to 1; 0 = auto (all cores). Threaded
    ///                     construction is safe because pocketfft threads run
    ///                     only during interpolant construction, never during
    ///                     evaluation or solve-time hot paths.
    static ChebTable from_values(const MatType &grid_values, double lb, double ub, int order,
                                 int nthreads = 1) {
        if (order < 1) throw std::invalid_argument("ChebTable: order must be >= 1");
        if (grid_values.cols() != order + 1)
            throw std::invalid_argument("ChebTable: grid_values must have order+1 columns");
        if (ub <= lb) throw std::invalid_argument("ChebTable: require ub > lb");
        if (nthreads < 0)
            throw std::invalid_argument("ChebTable: nthreads must be >= 0 (0 = auto)");

        ChebTable out;

        // Set rank-generic members for dim==1
        out.dim_ = 1;
        out.orders_ = {order};
        out.lb_.resize(1);
        out.lb_[0] = lb;
        out.ub_.resize(1);
        out.ub_[0] = ub;
        out.shape_ = {order + 1};
        out.strides_ = row_major_strides(out.shape_);
        out.tsize_ = order + 1;
        out.olen_ = int(grid_values.rows());

        // Build value_ (tsize × olen) via DCT-I.
        // Phase-1 stored coeffs_ as (order+1) × olen_; value_ has the same layout.
        out.value_.resize(out.tsize_, out.olen_);

        const size_t N1 = size_t(order) + 1;
        const pocketfft::shape_t shape{N1};
        const pocketfft::stride_t stride{ptrdiff_t(sizeof(double))};
        const pocketfft::shape_t axes{0};

        std::vector<double> in(N1), yk(N1);
        for (int c = 0; c < out.olen_; ++c) {
            for (size_t j = 0; j < N1; ++j) in[j] = grid_values(c, ptrdiff_t(j));
            pocketfft::dct(shape, stride, stride, axes, /*type=*/1, in.data(), yk.data(),
                           /*fct=*/1.0, /*ortho=*/false, size_t(nthreads));
            for (int k = 0; k <= order; ++k) out.value_(k, c) = yk[size_t(k)] / double(order);
            out.value_(0, c) *= 0.5;
            out.value_(order, c) *= 0.5;
        }

        // Precompute derivative-series coefficients (zero-extended to tsize rows).
        // grad_[0]: tsize rows (= order+1), first-derivative series in rows [0..order-1],
        //           row [order] is zero.
        // hess_[0]: tsize rows, second-derivative series in rows [0..order-2],
        //           rows [order-1..order] are zero.
        out.grad_.resize(1);
        out.grad_[0] = MatType::Zero(out.tsize_, out.olen_);
        for (int c = 0; c < out.olen_; ++c) {
            Eigen::VectorXd g1 = deriv_coeffs_from(out.value_.col(c), order);
            // g1 has length max(order,1) = order; fill rows [0..order-1]
            for (int k = 0; k < int(g1.size()); ++k) out.grad_[0](k, c) = g1[k];
        }

        out.hess_.resize(1);
        out.hess_[0] = MatType::Zero(out.tsize_, out.olen_);
        if (order >= 2) {
            for (int c = 0; c < out.olen_; ++c) {
                // Differentiate the first-derivative series again. deriv_coeffs_from
                // reads only indices 1..(order-1) of its input, so the zero-extension
                // row at index `order` in grad_[0] is never touched — no slicing needed.
                Eigen::VectorXd g2 = deriv_coeffs_from(out.grad_[0].col(c), order - 1);
                // g2 has length max(order-1,1) = order-1; fill rows [0..order-2]
                for (int k = 0; k < int(g2.size()); ++k) out.hess_[0](k, c) = g2[k];
            }
        }
        // order == 1: second derivative is identically zero — hess_[0] stays all-zero.

        return out;
    }

    // -------------------------------------------------------------------------
    // N-D node generation (Task 1)
    // -------------------------------------------------------------------------

    /// @brief Per-axis Chebyshev node vectors for a tensor-product grid.
    ///
    /// Returns a vector of length @c orders.size(), where element @c d is the
    /// @c (orders[d]+1)-point 2nd-kind node vector on @c [lb[d], ub[d]].
    /// The per-axis nodes are produced by the scalar @ref cheb_points overload.
    static std::vector<Eigen::VectorXd> cheb_points(const std::vector<int> &orders,
                                                    const Eigen::VectorXd &lb,
                                                    const Eigen::VectorXd &ub) {
        std::vector<Eigen::VectorXd> out(orders.size());
        for (size_t d = 0; d < orders.size(); ++d) out[d] = cheb_points(orders[d], lb[d], ub[d]);
        return out;
    }

    // -------------------------------------------------------------------------
    // N-D construction (Task 1)
    // -------------------------------------------------------------------------

    /// @brief Build an N-D interpolant from a flat grid-value matrix via
    /// separable DCT-I.
    ///
    /// @param grid_values_flat  Flat value matrix of shape @c (tsize, olen),
    ///   where @c tsize = ∏(orders[d]+1) and rows are row-major multi-indices
    ///   over the grid axes (last axis fastest; same ordering as nested loops
    ///   with the first axis in the outermost loop).
    /// @param lb      Lower bounds per axis (length D).
    /// @param ub      Upper bounds per axis (length D).
    /// @param orders  Polynomial order per axis (length D); each entry ≥ 1.
    /// @param nthreads  Thread count for pocketfft DCT calls; 0 = auto.
    ///
    /// Applies the 1-D DCT-I coefficient transform separably along each axis
    /// using @ref coeffs_along_axis.  @c grad_ and @c hess_ are left empty
    /// (Task 2 / Task 3 fill them).
    static ChebTable from_values(const MatType &grid_values_flat, const Eigen::VectorXd &lb,
                                 const Eigen::VectorXd &ub, const std::vector<int> &orders,
                                 int nthreads = 1) {
        const int D = int(orders.size());
        if (D < 1) throw std::invalid_argument("ChebTable: need >=1 axis");
        if (lb.size() != D || ub.size() != D)
            throw std::invalid_argument("ChebTable: lb/ub length must match orders");
        if (nthreads < 0) throw std::invalid_argument("ChebTable: nthreads >= 0 (0=auto)");
        ChebTable out;
        out.dim_ = D;
        out.orders_ = orders;
        out.lb_ = lb;
        out.ub_ = ub;
        out.shape_.resize(D);
        long tsize = 1;
        for (int d = 0; d < D; ++d) {
            if (orders[d] < 1) throw std::invalid_argument("ChebTable: each order >= 1");
            if (ub[d] <= lb[d]) throw std::invalid_argument("ChebTable: require ub > lb per axis");
            out.shape_[d] = orders[d] + 1;
            tsize *= out.shape_[d];
        }
        out.strides_ = row_major_strides(out.shape_);
        out.tsize_ = tsize;
        if (grid_values_flat.rows() != tsize)
            throw std::invalid_argument("ChebTable: grid_values rows must equal prod(order+1)");
        out.olen_ = int(grid_values_flat.cols());
        out.value_ = grid_values_flat;  // copy; DCT applied in place below
        for (int ax = 0; ax < D; ++ax)
            coeffs_along_axis(out.value_, out.shape_, out.strides_, ax, nthreads);
        // grad_ and hess_ are left empty; Tasks 2/3 populate them.
        return out;
    }

    // -------------------------------------------------------------------------
    // N-D evaluation helpers (Task 1)
    // -------------------------------------------------------------------------

    /// @brief Tensor Clenshaw: evaluate a flat coefficient tensor at a point @c xi.
    ///
    /// Contracts axes one at a time (always contracts the current axis 0 first),
    /// reusing the 1-D Clenshaw recurrence.  After contracting all D axes the
    /// result is a length-@c olen vector.
    ///
    /// @param coef   Flat coefficient matrix of shape @c (tsize, olen).
    /// @param shape  Grid shape (size D, shape[d] = order_d + 1).
    /// @param xi     Normalized evaluation point in [-1,1]^D (length D).
    /// @param olen   Number of output channels (= @c coef.cols()).
    static Eigen::VectorXd clenshaw_nd(const MatType &coef, const std::vector<int> &shape,
                                       const Eigen::VectorXd &xi, int olen) {
        const int D = int(shape.size());
        std::vector<int> shp = shape;
        // Working buffer per channel: indexed [channel][flat_in_remaining_tensor].
        std::vector<std::vector<double>> cur(olen);
        for (int c = 0; c < olen; ++c) {
            cur[c].resize(coef.rows());
            for (long r = 0; r < coef.rows(); ++r) cur[c][r] = coef(r, c);
        }
        for (int ax = 0; ax < D; ++ax) {
            // Contract the current axis 0 (order = shp[0]-1).
            const int n = shp[0] - 1;
            // Strides for the *current* (partially contracted) tensor.
            std::vector<long> strides = row_major_strides(shp);
            // stride0 = product of remaining axes sizes = number of lines.
            const long stride0 = strides[0];
            const long outer = stride0;  // number of lines == stride[0]
            const double xi_ax = xi[ax];
            for (int c = 0; c < olen; ++c) {
                std::vector<double> next(outer);
                for (long line = 0; line < outer; ++line) {
                    // Clenshaw along this line (k = n down to 1, then combine with k=0).
                    double b1 = 0.0, b2 = 0.0;
                    for (int k = n; k >= 1; --k) {
                        double bk = 2.0 * xi_ax * b1 - b2 + cur[c][line + long(k) * stride0];
                        b2 = b1;
                        b1 = bk;
                    }
                    next[line] = cur[c][line] + xi_ax * b1 - b2;
                }
                cur[c].swap(next);
            }
            shp.erase(shp.begin());  // drop the contracted axis
        }
        Eigen::VectorXd v(olen);
        for (int c = 0; c < olen; ++c) v[c] = cur[c][0];
        return v;
    }

    // -------------------------------------------------------------------------
    // Evaluation (1-D path re-expressed to read value_/grad_[0]/hess_[0])
    // -------------------------------------------------------------------------

    /// @brief Per-output norm of the trailing-half coefficients (adaptive convergence loop).
    Eigen::VectorXd coeff_tail_norm() const {
        Eigen::VectorXd tn(olen_);
        const int n = orders_[0];
        const int lo = (n + 1) / 2;
        for (int c = 0; c < olen_; ++c)
            tn[c] = value_.col(c).segment(lo, n + 1 - lo).norm();
        return tn;
    }

    /// @internal
    /// Unified writer: deriv=0 value only; 1 adds dv; 2 adds d2v. Pure double.
    /// Reads value_/grad_[0]/hess_[0] — no per-call allocation.
    /// @endinternal
    template <class VType>
    void eval_impl(double t, int deriv, VType &v, VType &dv, VType &d2v) const {
        const double xi = to_xi(t), h = half_span();
        const int order = orders_[0];
        for (int c = 0; c < olen_; ++c) {
            v[c] = clenshaw(value_.col(c), order, xi);
            if (deriv >= 1) {
                dv[c] = (order >= 1 ? clenshaw(grad_[0].col(c), order - 1, xi) : 0.0) / h;
                if (deriv >= 2) {
                    d2v[c] =
                        (order >= 2 ? clenshaw(hess_[0].col(c), order - 2, xi) : 0.0) / (h * h);
                }
            }
        }
    }

    /// @brief Evaluate all output channels at physical coordinate t (1-D).
    Eigen::VectorXd eval(double t) const {
        Eigen::VectorXd v(olen_);
        eval_impl(t, 0, v, v, v);
        return v;
    }

    /// @brief Evaluate all output channels at a physical point @c x (N-D).
    ///
    /// Each coordinate @c x[d] is clamped to @c [lb_[d], ub_[d]] before
    /// mapping to the normalised domain (no extrapolation).  Uses
    /// @ref clenshaw_nd internally.
    Eigen::VectorXd eval(const Eigen::VectorXd &x) const {
        Eigen::VectorXd xi(dim_);
        for (int d = 0; d < dim_; ++d) {
            double t = (2.0 * x[d] - lb_[d] - ub_[d]) / (ub_[d] - lb_[d]);
            xi[d] = std::clamp(t, -1.0, 1.0);
        }
        return clenshaw_nd(value_, shape_, xi, olen_);
    }

    /// @brief Evaluate value and first derivative at t. Returns (value, deriv1).
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> eval_deriv1(double t) const {
        Eigen::VectorXd v(olen_), dv(olen_);
        eval_impl(t, 1, v, dv, dv);
        return {v, dv};
    }

    /// @brief Evaluate value, first, and second derivative at t.
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> eval_deriv2(double t) const {
        Eigen::VectorXd v(olen_), dv(olen_), d2v(olen_);
        eval_impl(t, 2, v, dv, d2v);
        return {v, dv, d2v};
    }
};

/// @ingroup optimal_control
/// @brief VectorFunction wrapper over a 1-D ChebTable (analytic derivatives).
///
/// Wraps a shared @ref ChebTable as a CRTP VectorFunction with scalar input
/// (the independent variable) and vector output (one component per channel).
/// Analytic Jacobian and Hessian are provided via the precomputed Chebyshev
/// derivative-series stored in the @ref ChebTable; see @ref ChebTable::eval_impl.
///
/// @tparam IR  Compile-time input dimension. Currently only @c IR==-1 (dynamic,
///             scalar input) is ever instantiated; the parameter exists to
///             anticipate a future N-D generalization. NOTE: unlike
///             @ref InterpFunction1D whose template parameter is the @e output
///             dimension, @c ChebFunction's parameter is the @e input dimension
///             (output rows are always dynamic, -1).
template <int IR>
struct ChebFunction
    : VectorFunction<ChebFunction<IR>, IR, -1, DenseDerivativeMode::Analytic,
                     DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<ChebFunction<IR>, IR, -1, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    std::shared_ptr<ChebTable> tab; ///< Shared interpolant data (precomputed coefficients).

    ChebFunction() {}
    explicit ChebFunction(std::shared_ptr<ChebTable> tab_) : tab(tab_) {
        this->set_io_rows(1, tab->output_dim());
    }

    /// @internal
    /// Evaluate value; delegates to ChebTable::eval_impl(deriv=0).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        auto Impl = [&](auto &v) {
            this->tab->eval_impl(double(x[0]), 0, v, v, v);
            fx = v;
        };
        tycho::utils::BumpAllocator::allocate_run(
            Impl, TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }

    /// @internal
    /// Evaluate value and Jacobian; delegates to ChebTable::eval_impl(deriv=1).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        auto Impl = [&](auto &v, auto &dv) {
            this->tab->eval_impl(double(x[0]), 1, v, dv, v);
            fx = v;
            jx = dv;
        };
        tycho::utils::BumpAllocator::allocate_run(
            Impl, TempSpec<Output<Scalar>>(this->output_rows(), 1),
            TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }

    /// @internal
    /// Evaluate value, Jacobian, adjoint gradient, and adjoint Hessian;
    /// delegates to ChebTable::eval_impl(deriv=2).
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
        auto Impl = [&](auto &v, auto &dv, auto &d2v) {
            this->tab->eval_impl(double(x[0]), 2, v, dv, d2v);
            fx = v;
            jx = dv;
            adjgrad[0] = dv.dot(adjvars);
            adjhess(0, 0) = d2v.dot(adjvars);
        };
        tycho::utils::BumpAllocator::allocate_run(
            Impl, TempSpec<Output<Scalar>>(this->output_rows(), 1),
            TempSpec<Output<Scalar>>(this->output_rows(), 1),
            TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
};

}  // namespace tycho::oc
