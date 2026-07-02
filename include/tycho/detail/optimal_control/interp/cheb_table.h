// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/vector_function.h"
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <numbers>
#include <pocketfft_hdronly.h>
#include <stdexcept>
#include <string>

namespace tycho::oc {

using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::MatRef;
using vf::VecRef;
using vf::VectorFunction;

template <int IR> struct ChebFunction; // forward decl (befriended by ChebTable)

/// @ingroup optimal_control
/// @brief Rank-generic Chebyshev (2nd-kind / DCT-I) interpolant data class.
///
/// Holds Chebyshev-T coefficients and evaluates them numerically; it is not
/// itself a VectorFunction — wrap it in @ref ChebFunction (or call the binding's
/// `.vf()`) to obtain one.  Coefficients are stored per output channel in a flat
/// row-major tensor layout.  The @c dim_==1 case exposes the scalar-input API
/// (@ref from_values(olen×(n+1) overload), @ref eval(double), @ref eval_deriv1,
/// @ref eval_deriv2, @ref coeff_tail_norm) and is API- and result-compatible with
/// the original 1-D-only implementation.
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

    // ChebFunction drives the zero-allocation solve-time eval path and needs
    // access to the private coefficient tensors + raw eval helpers.
    template <int IR> friend struct ChebFunction;

    /// @brief Per-axis behaviour for queries that fall outside @c [lb_[d], ub_[d]].
    /// @c Clamp (default) snaps the query to the nearest endpoint (a global
    /// Chebyshev polynomial diverges outside its domain, so extrapolation is never
    /// desired). @c Periodic wraps the query modulo the axis span back into the
    /// domain — intended for angle-like / invariant-manifold axes; the sampled
    /// function must satisfy @c f(lb)≈f(ub) on that axis for the interpolant to be
    /// continuous across the seam.
    enum class OutOfDomain { Clamp, Periodic };

  private:
    // -------------------------------------------------------------------------
    // Rank-generic members
    // -------------------------------------------------------------------------
    int dim_ = 1;                  ///< Number of input dimensions.
    std::vector<int> orders_;      ///< Polynomial order per axis n_d.
    Eigen::VectorXd lb_, ub_;      ///< Domain bounds, length dim_.
    std::vector<OutOfDomain> oob_; ///< Per-axis out-of-domain policy, length dim_.
    std::vector<int> shape_;       ///< Grid shape: shape_[d] = orders_[d] + 1.
    std::vector<long> strides_;    ///< Row-major strides over shape_.
    long tsize_ = 1;               ///< Total grid size = prod(shape_).
    int olen_ = 0;                 ///< Output dimension.

    MatType value_;             ///< tsize_ × olen_: value Chebyshev coefficients.
    std::vector<MatType> grad_; ///< dim_ tensors (G_j = D_j value_), each tsize_ × olen_.
    std::vector<MatType> hess_; ///< dim_*(dim_+1)/2 tensors (H_jk, j≤k), tsize_ × olen_.

    // -------------------------------------------------------------------------
    // 1-D domain helpers (unchanged from Phase 1; used only for dim_==1 path).
    // A per-axis N-D analog of this mapping is added in a later task, so these
    // 1-D helpers coexist with the N-D path rather than being replaced.
    // -------------------------------------------------------------------------

    // Map physical x -> xi in [-1,1] for axis d, honoring the per-axis
    // out-of-domain policy. Clamp (default): queries outside [lb_[d], ub_[d]]
    // snap to the nearest endpoint (no extrapolation) — a global Chebyshev
    // polynomial diverges catastrophically outside its domain, so clamping is
    // deliberate. This differs from InterpTable1D, which extrapolates. Periodic:
    // wrap the query modulo the axis span back into the domain before mapping.
    // The trailing clamp also guards endpoint round-off in both cases.
    // Note: a non-finite (NaN/Inf) query is not rejected here — it propagates to
    // a NaN output. Construction rejects non-finite samples; eval-time callers
    // that need a guard should use contains() first.
    double axis_xi(int d, double x) const {
        const double span = ub_[d] - lb_[d];
        double t = x;
        if (oob_[d] == OutOfDomain::Periodic) {
            double u = std::fmod(x - lb_[d], span);
            if (u < 0.0)
                u += span; // fmod keeps the sign of the dividend
            t = lb_[d] + u;
        }
        double xi = (2.0 * t - lb_[d] - ub_[d]) / span;
        return std::clamp(xi, -1.0, 1.0);
    }
    // 1-D convenience wrapper (axis 0).
    double to_xi(double t) const { return axis_xi(0, t); }
    double half_span() const { return 0.5 * (ub_[0] - lb_[0]); }

    // Derivative scale factor for axis d at physical coordinate x: 1.0 when the
    // query is in-domain (or the axis is Periodic — Periodic wraps in-domain, so
    // derivatives are always valid), 0.0 when the query is strictly outside
    // [lb_[d], ub_[d]] under Clamp. Outside the domain the Clamp value is a flat
    // constant, so its derivative is zero; this keeps the analytic Jacobian /
    // Hessian consistent with eval() there. The boundary itself counts as
    // in-domain (the one-sided endpoint slope is retained).
    double axis_deriv_scale(int d, double x) const {
        if (oob_[d] == OutOfDomain::Periodic)
            return 1.0;
        return (x < lb_[d] || x > ub_[d]) ? 0.0 : 1.0;
    }

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
        for (int k = order; k >= 1; --k)
            gp[k - 1] = gp[k + 1] + 2.0 * double(k) * c[k];
        for (int k = 0; k < order; ++k)
            g[k] = gp[k];
        if (order >= 1)
            g[0] *= 0.5; // convert prime-convention T_0 term
        return g;
    }

  public:
    ChebTable() {}

    // -------------------------------------------------------------------------
    // Public static index helpers (row-major flat tensor layout)
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
        for (size_t d = 0; d < mi.size(); ++d)
            f += long(mi[d]) * strides[d];
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
        if (j > k)
            std::swap(j, k);
        return j * dim - (j * (j - 1)) / 2 + (k - j);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    int input_dim() const { return dim_; }
    int output_dim() const { return olen_; }
    /// @brief Polynomial order of a 1-D table. Throws on N-D tables (query
    /// per-axis orders via @ref orders instead), mirroring @ref coeff_tail_norm.
    int order() const {
        require_scalar();
        return orders_[0];
    }
    const std::vector<int> &orders() const { return orders_; }
    /// @brief Per-axis lower domain bounds (length dim_).
    const Eigen::VectorXd &lb() const { return lb_; }
    /// @brief Per-axis upper domain bounds (length dim_).
    const Eigen::VectorXd &ub() const { return ub_; }
    /// @brief Per-axis out-of-domain policy (length dim_).
    const std::vector<OutOfDomain> &out_of_domain() const { return oob_; }

    /// @brief True iff every coordinate of @c x lies within @c [lb_[d], ub_[d]].
    /// Useful for callers that want to guard against out-of-domain queries
    /// (which otherwise clamp, or wrap on Periodic axes) before evaluating.
    bool contains(const Eigen::VectorXd &x) const {
        require_dim(int(x.size()));
        for (int d = 0; d < dim_; ++d)
            if (x[d] < lb_[d] || x[d] > ub_[d])
                return false;
        return true;
    }
    /// @brief 1-D convenience overload of @ref contains.
    bool contains(double t) const {
        require_scalar();
        return t >= lb_[0] && t <= ub_[0];
    }

  private:
    // -------------------------------------------------------------------------
    // Query-shape guards (release builds disable Eigen bounds asserts, so a
    // wrong-rank query would otherwise read out of bounds and return garbage).
    // -------------------------------------------------------------------------
    void require_populated() const {
        if (orders_.empty())
            throw std::invalid_argument(
                "ChebTable: table is empty (default-constructed); build one via from_values");
    }
    void require_dim(int qsize) const {
        require_populated();
        if (qsize != dim_)
            throw std::invalid_argument("ChebTable: query dimension " + std::to_string(qsize) +
                                        " does not match table input dimension " +
                                        std::to_string(dim_));
    }
    void require_scalar() const {
        require_populated();
        if (dim_ != 1)
            throw std::invalid_argument(
                "ChebTable: scalar eval requires a 1-D table; this table is " +
                std::to_string(dim_) + "-dimensional — pass a length-" + std::to_string(dim_) +
                " array");
    }

    // Establish the grid-layout invariant (dim_/orders_/shape_/strides_/tsize_)
    // from per-axis orders in one place, so these interdependent fields are
    // derived from a single source of truth and cannot desync between the 1-D
    // and N-D from_values paths. Does not validate orders (callers check).
    void set_grid_layout(const std::vector<int> &orders) {
        dim_ = int(orders.size());
        orders_ = orders;
        shape_.resize(dim_);
        long ts = 1;
        for (int d = 0; d < dim_; ++d) {
            shape_[d] = orders[d] + 1;
            ts *= shape_[d];
        }
        strides_ = row_major_strides(shape_);
        tsize_ = ts;
    }

    // -------------------------------------------------------------------------
    // N-D building blocks (private — not part of the public API)
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
        const int n = shape[ax] - 1; // polynomial order along this axis
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
            if ((start / stride) % long(shape[ax]) != 0)
                continue; // not a line start
            for (int c = 0; c < olen; ++c) {
                for (int k = 0; k <= n; ++k)
                    in[size_t(k)] = tens(start + long(k) * stride, c);
                pocketfft::dct(pshape, pstride, pstride, axes, /*type=*/1, in.data(), yk.data(),
                               /*fct=*/1.0, /*ortho=*/false, size_t(nthreads));
                for (int k = 0; k <= n; ++k)
                    tens(start + long(k) * stride, c) = yk[size_t(k)] / double(n);
                tens(start, c) *= 0.5;
                tens(start + long(n) * stride, c) *= 0.5;
            }
        }
    }

    /// @brief Return a copy of @c coef differentiated along axis @c ax.
    ///
    /// Applies @ref deriv_coeffs_from to each line along @c ax (same line-start
    /// iteration as @ref coeffs_along_axis) and zero-extends the result to the
    /// same shape as the input (the derivative T-series is shorter; the trailing
    /// entries are left at zero).
    static MatType deriv_along_axis(const MatType &coef, const std::vector<int> &shape,
                                    const std::vector<long> &strides, int ax) {
        MatType out = MatType::Zero(coef.rows(), coef.cols());
        const int n = shape[ax] - 1; // order along this axis
        const long stride = strides[ax];
        const int olen = int(coef.cols());
        Eigen::VectorXd line(n + 1);
        for (long start = 0; start < coef.rows(); ++start) {
            if ((start / stride) % long(shape[ax]) != 0)
                continue; // not a line start
            for (int c = 0; c < olen; ++c) {
                for (int k = 0; k <= n; ++k)
                    line[k] = coef(start + long(k) * stride, c);
                Eigen::VectorXd g = deriv_coeffs_from(line, n); // length max(n,1)
                for (int k = 0; k < int(g.size()); ++k)
                    out(start + long(k) * stride, c) = g[k];
                // entries k=g.size()..n stay zero (zero-extension)
            }
        }
        return out;
    }

    /// @brief Compute per-axis normalised coordinates @c xi and inverse half-spans
    /// @c hinv from a physical point @c x.
    ///
    /// For axis @c d, @c xi[d] is produced by @ref axis_xi (which applies the
    /// per-axis @ref OutOfDomain policy), and @c hinv[d] = 2 / (ub_[d] - lb_[d])
    /// is the physical→normalised chain-rule factor, multiplied by the
    /// @ref axis_deriv_scale mask (0 outside the Clamp domain so derivatives stay
    /// consistent with the flat clamped value; 1 in-domain and for Periodic axes,
    /// whose wrap is a pure translation that does not change @c hinv).
    ///
    /// Used by @ref eval(const Eigen::VectorXd&), @ref eval_jacobian, and
    /// @ref eval_hessian to avoid duplicating the per-axis mapping logic.
    ///
    /// The @c _raw overload writes into caller-provided length-@c dim_ buffers so
    /// the zero-allocation @ref ChebFunction solve-time path can supply arena
    /// scratch instead of allocating Eigen vectors.
    void compute_xi_hinv_raw(const double *x, double *xi, double *hinv) const {
        for (int d = 0; d < dim_; ++d) {
            // Fold the out-of-domain derivative mask into hinv: since the Jacobian
            // uses hinv[j] linearly and the Hessian uses hinv[j]*hinv[k], zeroing
            // hinv[d] for a clamped axis zeros every derivative involving axis d,
            // matching the flat clamped value. The value path uses only xi.
            hinv[d] = (2.0 / (ub_[d] - lb_[d])) * axis_deriv_scale(d, x[d]);
            xi[d] = axis_xi(d, x[d]);
        }
    }
    void compute_xi_hinv(const Eigen::VectorXd &x, Eigen::VectorXd &xi,
                         Eigen::VectorXd &hinv) const {
        xi.resize(dim_);
        hinv.resize(dim_);
        compute_xi_hinv_raw(x.data(), xi.data(), hinv.data());
    }

    /// @brief Tensor Clenshaw: evaluate a flat coefficient tensor at @c xi into
    /// caller-provided scratch/output, allocation-free.
    ///
    /// Contracts axes one at a time (always the current axis 0 first), reusing the
    /// 1-D Clenshaw recurrence.  Because every coefficient tensor shares this
    /// table's @c shape_/@c strides_, and @c strides_[ax] equals the stride of the
    /// first remaining axis at contraction step @c ax (both are ∏shape_[ax+1..]),
    /// no per-axis stride recomputation or shape copy is needed.
    ///
    /// Each output channel keeps its data packed at the front of its own
    /// @c tsize_-sized slot in the ping-pong buffers (@c cur/@c next), so both
    /// buffers must have length @c olen_*tsize_.
    ///
    /// @param coef  Flat coefficient matrix of shape @c (tsize_, olen_).
    /// @param xi    Normalized evaluation point in [-1,1]^D (length @c dim_).
    /// @param cur   Scratch buffer, length @c olen_*tsize_ (overwritten).
    /// @param next  Scratch buffer, length @c olen_*tsize_ (overwritten).
    /// @param out   Output buffer, length @c olen_.
    void eval_tensor_into(const MatType &coef, const double *xi, double *cur, double *next,
                          double *out) const {
        for (int c = 0; c < olen_; ++c)
            for (long r = 0; r < tsize_; ++r)
                cur[long(c) * tsize_ + r] = coef(r, c);
        double *a = cur, *b = next;
        for (int ax = 0; ax < dim_; ++ax) {
            const int n = shape_[ax] - 1;
            const long stride0 = strides_[ax]; // == number of lines (outer)
            const double xi_ax = xi[ax];
            for (int c = 0; c < olen_; ++c) {
                const double *cc = a + long(c) * tsize_;
                double *nc = b + long(c) * tsize_;
                for (long line = 0; line < stride0; ++line) {
                    double b1 = 0.0, b2 = 0.0;
                    for (int k = n; k >= 1; --k) {
                        double bk = 2.0 * xi_ax * b1 - b2 + cc[line + long(k) * stride0];
                        b2 = b1;
                        b1 = bk;
                    }
                    nc[line] = cc[line] + xi_ax * b1 - b2;
                }
            }
            std::swap(a, b);
        }
        for (int c = 0; c < olen_; ++c)
            out[c] = a[long(c) * tsize_];
    }

    /// @brief Allocating convenience wrapper around @ref eval_tensor_into for the
    /// public (non-solve-time) eval methods.  Allocates its own scratch + result.
    Eigen::VectorXd clenshaw_nd(const MatType &coef, const Eigen::VectorXd &xi) const {
        std::vector<double> cur(size_t(olen_) * size_t(tsize_));
        std::vector<double> next(size_t(olen_) * size_t(tsize_));
        Eigen::VectorXd out(olen_);
        eval_tensor_into(coef, xi.data(), cur.data(), next.data(), out.data());
        return out;
    }

  public:
    // -------------------------------------------------------------------------
    // 1-D node generation (unchanged Phase-1 API)
    // -------------------------------------------------------------------------

    /// @brief 2nd-kind node coordinates on [lb,ub], j=0..n (t_0=ub .. t_n=lb).
    static Eigen::VectorXd cheb_points(int n, double lb, double ub) {
        if (n < 1)
            throw std::invalid_argument("ChebTable: order must be >= 1");
        const double m = 0.5 * (lb + ub), h = 0.5 * (ub - lb);
        Eigen::VectorXd t(n + 1);
        for (int j = 0; j <= n; ++j)
            t[j] = m + h * std::cos(double(j) * std::numbers::pi / double(n));
        return t;
    }

    // -------------------------------------------------------------------------
    // 1-D from_values (Phase-1 API, re-homed onto generic members)
    // -------------------------------------------------------------------------

    /// @brief Build coefficients from grid values ((n+1) × olen) via DCT-I.
    ///
    /// Derivative-series coefficients (first and second order) are precomputed
    /// here and stored in @c grad_[0] and @c hess_[0] (zero-extended to tsize
    /// rows), so that @ref eval_impl incurs no per-call allocation during
    /// solve-time evaluation.
    ///
    /// @param grid_values  Matrix of shape (order+1, olen): row @c j is the value
    ///   at node @c j (all output channels), column @c c is output channel @c c.
    ///   This matches the N-D overload's @c (tsize, olen) convention (here
    ///   @c tsize = order+1) so both ranks share one node-major orientation.
    /// @param lb           Lower bound of the physical domain.
    /// @param ub           Upper bound of the physical domain.
    /// @param order        Polynomial order n; the table uses n+1 nodes.
    /// @param nthreads     Thread count forwarded to pocketfft for the DCT-I.
    ///                     Defaults to 1; 0 = auto (all cores). Threaded
    ///                     construction is safe because pocketfft threads run
    ///                     only during interpolant construction, never during
    ///                     evaluation or solve-time hot paths.
    /// @param oob          Out-of-domain policy for the single axis (default Clamp).
    static ChebTable from_values(const MatType &grid_values, double lb, double ub, int order,
                                 int nthreads = 1, OutOfDomain oob = OutOfDomain::Clamp) {
        if (order < 1)
            throw std::invalid_argument("ChebTable: order must be >= 1");
        if (grid_values.rows() != order + 1)
            throw std::invalid_argument("ChebTable: grid_values must have order+1 rows");
        if (ub <= lb)
            throw std::invalid_argument("ChebTable: require ub > lb");
        if (nthreads < 0)
            throw std::invalid_argument("ChebTable: nthreads must be >= 0 (0 = auto)");

        ChebTable out;

        // Set rank-generic layout members (dim_/orders_/shape_/strides_/tsize_).
        out.set_grid_layout({order});
        out.lb_.resize(1);
        out.lb_[0] = lb;
        out.ub_.resize(1);
        out.ub_[0] = ub;
        out.oob_ = {oob};
        out.olen_ = int(grid_values.cols());

        // Build value_ (tsize × olen) via DCT-I; grid_values is already (tsize, olen).
        out.value_.resize(out.tsize_, out.olen_);

        const size_t N1 = size_t(order) + 1;
        const pocketfft::shape_t shape{N1};
        const pocketfft::stride_t stride{ptrdiff_t(sizeof(double))};
        const pocketfft::shape_t axes{0};

        std::vector<double> in(N1), yk(N1);
        for (int c = 0; c < out.olen_; ++c) {
            for (size_t j = 0; j < N1; ++j)
                in[j] = grid_values(ptrdiff_t(j), c);
            pocketfft::dct(shape, stride, stride, axes, /*type=*/1, in.data(), yk.data(),
                           /*fct=*/1.0, /*ortho=*/false, size_t(nthreads));
            for (int k = 0; k <= order; ++k)
                out.value_(k, c) = yk[size_t(k)] / double(order);
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
            for (int k = 0; k < int(g1.size()); ++k)
                out.grad_[0](k, c) = g1[k];
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
                for (int k = 0; k < int(g2.size()); ++k)
                    out.hess_[0](k, c) = g2[k];
            }
        }
        // order == 1: second derivative is identically zero — hess_[0] stays all-zero.

        return out;
    }

    // -------------------------------------------------------------------------
    // N-D node generation
    // -------------------------------------------------------------------------

    /// @brief Per-axis Chebyshev node vectors for a tensor-product grid.
    ///
    /// Returns a vector of length @c orders.size(), where element @c d is the
    /// @c (orders[d]+1)-point 2nd-kind node vector on @c [lb[d], ub[d]].
    /// The per-axis nodes are produced by the scalar @ref cheb_points overload.
    static std::vector<Eigen::VectorXd> cheb_points(const std::vector<int> &orders,
                                                    const Eigen::VectorXd &lb,
                                                    const Eigen::VectorXd &ub) {
        if (lb.size() != Eigen::Index(orders.size()) || ub.size() != Eigen::Index(orders.size()))
            throw std::invalid_argument("ChebTable::cheb_points: lb/ub length must match orders");
        std::vector<Eigen::VectorXd> out(orders.size());
        for (size_t d = 0; d < orders.size(); ++d)
            out[d] = cheb_points(orders[d], lb[d], ub[d]);
        return out;
    }

    // -------------------------------------------------------------------------
    // N-D construction
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
    /// @param oob     Per-axis out-of-domain policy (length D, or empty for
    ///                all-Clamp default).
    ///
    /// Applies the 1-D DCT-I coefficient transform separably along each axis
    /// using @ref coeffs_along_axis.  All three precomputed tensors are filled:
    /// @c value_ (DCT-I coefficients), @c grad_[j] = D_j value_ (gradient tensors),
    /// and @c hess_[hess_index(j,k,D)] = D_k D_j value_ (upper-triangle Hessian tensors).
    static ChebTable from_values(const MatType &grid_values_flat, const Eigen::VectorXd &lb,
                                 const Eigen::VectorXd &ub, const std::vector<int> &orders,
                                 int nthreads = 1, std::vector<OutOfDomain> oob = {}) {
        const int D = int(orders.size());
        if (D < 1)
            throw std::invalid_argument("ChebTable: need >=1 axis");
        if (lb.size() != D || ub.size() != D)
            throw std::invalid_argument("ChebTable: lb/ub length must match orders");
        if (nthreads < 0)
            throw std::invalid_argument("ChebTable: nthreads must be >= 0 (0 = auto)");
        if (oob.empty())
            oob.assign(D, OutOfDomain::Clamp);
        else if (int(oob.size()) != D)
            throw std::invalid_argument("ChebTable: out_of_domain length must match orders");
        for (int d = 0; d < D; ++d) {
            if (orders[d] < 1)
                throw std::invalid_argument("ChebTable: each order >= 1");
            if (ub[d] <= lb[d])
                throw std::invalid_argument("ChebTable: require ub > lb per axis");
        }
        ChebTable out;
        out.set_grid_layout(orders); // dim_/orders_/shape_/strides_/tsize_
        out.lb_ = lb;
        out.ub_ = ub;
        out.oob_ = std::move(oob);
        if (grid_values_flat.rows() != out.tsize_)
            throw std::invalid_argument("ChebTable: grid_values rows must equal prod(order+1)");
        out.olen_ = int(grid_values_flat.cols());
        // Guard the solve-time scratch-buffer index type: ChebFunction's arena
        // (TempSpec) sizes are int, and eval scratch is olen*tsize doubles. Reject
        // tables whose product would overflow int rather than silently truncate.
        if (long(out.olen_) * out.tsize_ > long(std::numeric_limits<int>::max()))
            throw std::invalid_argument(
                "ChebTable: table too large — olen*prod(order+1) exceeds the int "
                "scratch-buffer limit");
        out.value_ = grid_values_flat; // copy; DCT applied in place below
        for (int ax = 0; ax < D; ++ax)
            coeffs_along_axis(out.value_, out.shape_, out.strides_, ax, nthreads);
        // Precompute gradient tensors G_j = D_j value_ (zero-extended, same shape).
        out.grad_.resize(D);
        for (int j = 0; j < D; ++j)
            out.grad_[j] = deriv_along_axis(out.value_, out.shape_, out.strides_, j);
        // Precompute Hessian tensors H_jk = D_k G_j (upper triangle, j<=k).
        // Differentiating grad_[j] along axis k gives D_k D_j value_ = H_jk.
        out.hess_.resize(D * (D + 1) / 2);
        for (int j = 0; j < D; ++j)
            for (int k = j; k < D; ++k)
                out.hess_[hess_index(j, k, D)] =
                    deriv_along_axis(out.grad_[j], out.shape_, out.strides_, k);
        return out;
    }

    // -------------------------------------------------------------------------
    // Evaluation (1-D path re-expressed to read value_/grad_[0]/hess_[0])
    // -------------------------------------------------------------------------

    /// @brief Per-output norm of the trailing-half Chebyshev coefficients along the
    /// single axis (1-D adaptive-convergence indicator).
    /// @pre 1-D table only. N-D adaptivity refines each axis via marginal 1-D tables
    /// (see the Python `cheb_adaptive` N-D path); calling this on an N-D table throws
    /// rather than silently returning an axis-0-only value.
    Eigen::VectorXd coeff_tail_norm() const {
        if (dim_ != 1)
            throw std::invalid_argument(
                "ChebTable::coeff_tail_norm is 1-D only; N-D adaptivity uses per-axis marginals");
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
        // Zero the derivative outside the domain (Clamp value is flat there); a
        // 0/1 factor, so applying it once also zeroes the second derivative.
        const double dscale = (deriv >= 1) ? axis_deriv_scale(0, t) : 1.0;
        for (int c = 0; c < olen_; ++c) {
            v[c] = clenshaw(value_.col(c), order, xi);
            if (deriv >= 1) {
                dv[c] = dscale * (order >= 1 ? clenshaw(grad_[0].col(c), order - 1, xi) : 0.0) / h;
                if (deriv >= 2) {
                    d2v[c] = dscale *
                             (order >= 2 ? clenshaw(hess_[0].col(c), order - 2, xi) : 0.0) /
                             (h * h);
                }
            }
        }
    }

    /// @brief Evaluate all output channels at physical coordinate t (1-D).
    Eigen::VectorXd eval(double t) const {
        require_scalar();
        Eigen::VectorXd v(olen_);
        eval_impl(t, 0, v, v, v);
        return v;
    }

    /// @brief Evaluate all output channels at a physical point @c x (N-D).
    ///
    /// Each coordinate @c x[d] is mapped to the normalised domain per the axis's
    /// @ref OutOfDomain policy (Clamp by default, Periodic wraps) before
    /// evaluation.  Uses @ref clenshaw_nd internally.
    Eigen::VectorXd eval(const Eigen::VectorXd &x) const {
        require_dim(int(x.size()));
        Eigen::VectorXd xi, hinv;
        compute_xi_hinv(x, xi, hinv);
        return clenshaw_nd(value_, xi);
    }

    /// @brief Evaluate the analytic Jacobian at a physical point @c x (N-D).
    ///
    /// Returns an @c (olen × dim_) matrix where column @c j is the partial
    /// derivative of all output channels with respect to input @c x[j].
    /// Reads the precomputed gradient tensors @c grad_[j] (populated by the
    /// N-D @ref from_values overload); the chain-rule factor @c hinv[j] = 2/(ub_[j]-lb_[j])
    /// accounts for the change of variables from physical to normalised coordinates.
    ///
    /// @pre @c grad_ must be non-empty (constructed via the N-D @ref from_values overload).
    Eigen::MatrixXd eval_jacobian(const Eigen::VectorXd &x) const {
        require_dim(int(x.size()));
        Eigen::VectorXd xi, hinv;
        compute_xi_hinv(x, xi, hinv);
        Eigen::MatrixXd J(olen_, dim_);
        for (int j = 0; j < dim_; ++j)
            J.col(j) = clenshaw_nd(grad_[j], xi) * hinv[j];
        return J;
    }

    /// @brief Evaluate the analytic Hessian at a physical point @c x (N-D).
    ///
    /// Returns a vector of @c olen symmetric @c (dim_×dim_) Hessian matrices,
    /// one per output channel.  Entry @c H[c](j,k) = ∂²f_c/∂x_j∂x_k.
    ///
    /// Reads the precomputed Hessian tensors @c hess_[hess_index(j,k,dim_)]
    /// (upper triangle, populated by the N-D @ref from_values overload) and
    /// applies the chain-rule factors @c hinv[j]*hinv[k] = 4/((ub_[j]-lb_[j])*(ub_[k]-lb_[k])).
    ///
    /// @pre @c hess_ must be non-empty (constructed via the N-D @ref from_values overload).
    std::vector<Eigen::MatrixXd> eval_hessian(const Eigen::VectorXd &x) const {
        require_dim(int(x.size()));
        Eigen::VectorXd xi, hinv;
        compute_xi_hinv(x, xi, hinv);
        std::vector<Eigen::MatrixXd> H(olen_, Eigen::MatrixXd::Zero(dim_, dim_));
        for (int j = 0; j < dim_; ++j) {
            for (int k = j; k < dim_; ++k) {
                Eigen::VectorXd hv =
                    clenshaw_nd(hess_[hess_index(j, k, dim_)], xi) * (hinv[j] * hinv[k]);
                for (int c = 0; c < olen_; ++c) {
                    H[c](j, k) = hv[c];
                    H[c](k, j) = hv[c];
                }
            }
        }
        return H;
    }

    /// @brief Evaluate value and first derivative at t. Returns (value, deriv1).
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> eval_deriv1(double t) const {
        require_scalar();
        Eigen::VectorXd v(olen_), dv(olen_);
        eval_impl(t, 1, v, dv, dv);
        return {v, dv};
    }

    /// @brief Evaluate value, first, and second derivative at t.
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> eval_deriv2(double t) const {
        require_scalar();
        Eigen::VectorXd v(olen_), dv(olen_), d2v(olen_);
        eval_impl(t, 2, v, dv, d2v);
        return {v, dv, d2v};
    }
};

/// @ingroup optimal_control
/// @brief VectorFunction wrapper over a ChebTable (analytic value, Jacobian, and Hessian).
///
/// Wraps a shared @ref ChebTable as a CRTP VectorFunction.  For 1-D input
/// (@c IR==1 or @c IR==-1 with a 1-D table) the wrapper delegates to
/// @ref ChebTable::eval_impl for all three compute methods (allocation-free).
/// For N-D input (N≥2) it delegates to
/// @ref ChebTable::eval, @ref ChebTable::eval_jacobian, and
/// @ref ChebTable::eval_hessian using the precomputed gradient / Hessian tensors.
///
/// @tparam IR  Compile-time input dimension.  May be any positive integer or
///             @c -1 (fully dynamic).  The output dimension is always dynamic
///             (@c -1) — it is determined at runtime from @c tab->output_dim().
///             NOTE: unlike @ref InterpFunction1D whose template parameter is the
///             @e output dimension, @c ChebFunction's parameter is the
///             @e input dimension.
template <int IR>
struct ChebFunction : VectorFunction<ChebFunction<IR>, IR, -1, DenseDerivativeMode::Analytic,
                                     DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<ChebFunction<IR>, IR, -1, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    std::shared_ptr<ChebTable> tab; ///< Shared interpolant data (precomputed coefficients).

    ChebFunction() {}
    explicit ChebFunction(std::shared_ptr<ChebTable> tab_) : tab(tab_) {
        if (!tab_)
            throw std::invalid_argument("ChebFunction: null ChebTable");
        // set_input_rows is a compile-time no-op when IR >= 0 (fixed input dim),
        // so a fixed IR that disagrees with the table's runtime input dim would
        // otherwise be silently accepted. Reject it here.
        if constexpr (IR >= 0) {
            if (tab_->input_dim() != IR)
                throw std::invalid_argument("ChebFunction<IR>: compile-time IR (" +
                                            std::to_string(IR) +
                                            ") does not match table input dimension (" +
                                            std::to_string(tab_->input_dim()) + ")");
        }
        this->set_io_rows(tab_->input_dim(), tab_->output_dim());
    }

    /// @internal
    /// Evaluate value.
    /// 1-D path: delegates to ChebTable::eval_impl(deriv=0) (no allocation).
    /// N-D path: delegates to ChebTable::eval(VectorXd).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        if (this->tab->input_dim() == 1) {
            typedef typename InType::Scalar Scalar;
            auto Impl = [&](auto &v) {
                this->tab->eval_impl(double(x[0]), 0, v, v, v);
                fx = v;
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, TempSpec<Output<Scalar>>(this->output_rows(), 1));
        } else {
            // N-D path: allocation-free tensor-Clenshaw over arena scratch.
            const int D = this->tab->dim_;
            const int olen = this->tab->olen_;
            const int buf = int(long(olen) * this->tab->tsize_);
            auto Impl = [&](auto &xd, auto &xi, auto &hinv, auto &cur, auto &next, auto &out) {
                for (int i = 0; i < D; ++i)
                    xd[i] = double(x[i]);
                this->tab->compute_xi_hinv_raw(xd.data(), xi.data(), hinv.data());
                this->tab->eval_tensor_into(this->tab->value_, xi.data(), cur.data(), next.data(),
                                            out.data());
                fx = out;
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(buf, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(buf, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(olen, 1));
        }
    }

    /// @internal
    /// Evaluate value and Jacobian.
    /// 1-D path: delegates to ChebTable::eval_impl(deriv=1).
    /// N-D path: delegates to ChebTable::eval + ChebTable::eval_jacobian.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        if (this->tab->input_dim() == 1) {
            typedef typename InType::Scalar Scalar;
            auto Impl = [&](auto &v, auto &dv) {
                this->tab->eval_impl(double(x[0]), 1, v, dv, v);
                fx = v;
                jx = dv;
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, TempSpec<Output<Scalar>>(this->output_rows(), 1),
                TempSpec<Output<Scalar>>(this->output_rows(), 1));
        } else {
            // N-D path: value + analytic Jacobian, allocation-free over arena scratch.
            const int D = this->tab->dim_;
            const int olen = this->tab->olen_;
            const int buf = int(long(olen) * this->tab->tsize_);
            auto Impl = [&](auto &xd, auto &xi, auto &hinv, auto &cur, auto &next, auto &col) {
                for (int i = 0; i < D; ++i)
                    xd[i] = double(x[i]);
                this->tab->compute_xi_hinv_raw(xd.data(), xi.data(), hinv.data());
                this->tab->eval_tensor_into(this->tab->value_, xi.data(), cur.data(), next.data(),
                                            col.data());
                fx = col;
                for (int j = 0; j < D; ++j) {
                    this->tab->eval_tensor_into(this->tab->grad_[j], xi.data(), cur.data(),
                                                next.data(), col.data());
                    for (int c = 0; c < olen; ++c)
                        jx(c, j) = col[c] * hinv[j];
                }
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(buf, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(buf, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(olen, 1));
        }
    }

    /// @internal
    /// Evaluate value, Jacobian, adjoint gradient, and adjoint Hessian.
    /// 1-D path: delegates to ChebTable::eval_impl(deriv=2).
    /// N-D path: delegates to ChebTable::eval_jacobian + ChebTable::eval_hessian.
    ///   adjgrad[j] = J^T[:,j] . adjvars
    ///   adjhess(j,k) = sum_c adjvars[c] * H[c](j,k)
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
        if (this->tab->input_dim() == 1) {
            typedef typename InType::Scalar Scalar;
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
        } else {
            // N-D path: value, Jacobian, adjoint gradient + adjoint Hessian, all
            // allocation-free by contracting the precomputed tensors over arena scratch.
            const int D = this->tab->dim_;
            const int olen = this->tab->olen_;
            const int buf = int(long(olen) * this->tab->tsize_);
            auto Impl = [&](auto &xd, auto &xi, auto &hinv, auto &cur, auto &next, auto &col) {
                for (int i = 0; i < D; ++i)
                    xd[i] = double(x[i]);
                this->tab->compute_xi_hinv_raw(xd.data(), xi.data(), hinv.data());
                // Value.
                this->tab->eval_tensor_into(this->tab->value_, xi.data(), cur.data(), next.data(),
                                            col.data());
                fx = col;
                // Jacobian column j = hinv[j] * D_j value; adjgrad[j] = J[:,j] . adjvars.
                for (int j = 0; j < D; ++j) {
                    this->tab->eval_tensor_into(this->tab->grad_[j], xi.data(), cur.data(),
                                                next.data(), col.data());
                    double g = 0.0;
                    for (int c = 0; c < olen; ++c) {
                        double v = col[c] * hinv[j];
                        jx(c, j) = v;
                        g += v * adjvars[c];
                    }
                    adjgrad[j] = g;
                }
                // Adjoint Hessian(j,k) = sum_c adjvars[c] * hinv[j]*hinv[k] * D_k D_j value_c.
                adjhess.setZero();
                for (int j = 0; j < D; ++j) {
                    for (int k = j; k < D; ++k) {
                        this->tab->eval_tensor_into(
                            this->tab->hess_[ChebTable::hess_index(j, k, D)], xi.data(), cur.data(),
                            next.data(), col.data());
                        const double f = hinv[j] * hinv[k];
                        double h = 0.0;
                        for (int c = 0; c < olen; ++c)
                            h += adjvars[c] * col[c] * f;
                        adjhess(j, k) = h;
                        adjhess(k, j) = h;
                    }
                }
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(D, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(buf, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(buf, 1),
                tycho::utils::TempSpec<Eigen::VectorXd>(olen, 1));
        }
    }
};

} // namespace tycho::oc
