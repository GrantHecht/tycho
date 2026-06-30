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
/// @brief 1-D Chebyshev (2nd-kind / DCT-I) interpolant data class.
///
/// Stores Chebyshev-T coefficients per output channel and evaluates via
/// Clenshaw recurrence with analytic first/second derivatives.  Built from
/// values sampled at the 2nd-kind nodes returned by @ref cheb_points.
/// Derivative-series coefficients are precomputed once in @ref from_values
/// so that @ref eval_impl performs no per-call allocation.
/// Structured for a future N-D generalization; this phase is 1-D only.
struct ChebTable {
    using MatType = Eigen::Matrix<double, -1, -1>;

  private:
    int order_ = 0;               ///< Polynomial order n (node count = n+1).
    double lb_ = -1.0, ub_ = 1.0; ///< Domain.
    MatType coeffs_;               ///< (order_+1) x olen_ Chebyshev coefficients.
    MatType dcoeffs_;              ///< order_ x olen_ first-derivative series coefficients.
    MatType d2coeffs_;             ///< max(order_-1, 1) x olen_ second-derivative series
                                   ///< coefficients (1 row of zeros when order_==1).
    int olen_ = 0;                 ///< Output dimension.

    // Map physical t -> xi in [-1,1]; clamp to guard round-off at the ends.
    double to_xi(double t) const {
        double xi = (2.0 * t - lb_ - ub_) / (ub_ - lb_);
        return std::clamp(xi, -1.0, 1.0);
    }
    double half_span() const { return 0.5 * (ub_ - lb_); }

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

    int input_dim() const { return 1; }
    int output_dim() const { return olen_; }
    int order() const { return order_; }

    /// @brief 2nd-kind node coordinates on [lb,ub], j=0..n (t_0=ub .. t_n=lb).
    static Eigen::VectorXd cheb_points(int n, double lb, double ub) {
        if (n < 1) throw std::invalid_argument("ChebTable: order must be >= 1");
        const double m = 0.5 * (lb + ub), h = 0.5 * (ub - lb);
        Eigen::VectorXd t(n + 1);
        for (int j = 0; j <= n; ++j)
            t[j] = m + h * std::cos(double(j) * std::numbers::pi / double(n));
        return t;
    }

    /// @brief Build coefficients from grid values (olen x (n+1)) via DCT-I.
    ///
    /// Derivative-series coefficients (first and second order) are precomputed
    /// here and stored as members, so @ref eval_impl incurs no per-call
    /// allocation or recurrence work during solve-time evaluation.
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
        out.order_ = order;
        out.lb_ = lb;
        out.ub_ = ub;
        out.olen_ = int(grid_values.rows());
        out.coeffs_.resize(order + 1, out.olen_);

        const size_t N1 = size_t(order) + 1;
        const pocketfft::shape_t shape{N1};
        const pocketfft::stride_t stride{ptrdiff_t(sizeof(double))};
        const pocketfft::shape_t axes{0};

        std::vector<double> in(N1), yk(N1);
        for (int c = 0; c < out.olen_; ++c) {
            for (size_t j = 0; j < N1; ++j) in[j] = grid_values(c, ptrdiff_t(j));
            pocketfft::dct(shape, stride, stride, axes, /*type=*/1, in.data(), yk.data(),
                           /*fct=*/1.0, /*ortho=*/false, size_t(nthreads));
            for (int k = 0; k <= order; ++k) out.coeffs_(k, c) = yk[size_t(k)] / double(order);
            out.coeffs_(0, c) *= 0.5;      // halve endpoint coefficients
            out.coeffs_(order, c) *= 0.5;
        }

        // Precompute derivative-series coefficients once — eliminates per-call
        // recurrence and allocation in eval_impl.
        // dcoeffs_:  order rows × olen columns  (d/dxi series; order >= 1 guaranteed above)
        // d2coeffs_: max(order-1, 1) rows × olen columns  (d²/dxi² series;
        //            1 row of zeros when order == 1)
        out.dcoeffs_.resize(std::max(order, 1), out.olen_);
        for (int c = 0; c < out.olen_; ++c) {
            Eigen::VectorXd g1 = deriv_coeffs_from(out.coeffs_.col(c), order);
            out.dcoeffs_.col(c) = g1;
        }

        const int d2rows = std::max(order - 1, 1);
        out.d2coeffs_.resize(d2rows, out.olen_);
        if (order >= 2) {
            for (int c = 0; c < out.olen_; ++c) {
                Eigen::VectorXd g2 = deriv_coeffs_from(out.dcoeffs_.col(c), order - 1);
                out.d2coeffs_.col(c) = g2;
            }
        } else {
            // order == 1: second derivative is identically zero
            out.d2coeffs_.setZero();
        }

        return out;
    }

    /// @brief Per-output norm of the trailing-half coefficients (adaptive convergence loop).
    Eigen::VectorXd coeff_tail_norm() const {
        Eigen::VectorXd tn(olen_);
        const int lo = (order_ + 1) / 2;
        for (int c = 0; c < olen_; ++c)
            tn[c] = coeffs_.col(c).segment(lo, order_ + 1 - lo).norm();
        return tn;
    }

    /// @internal
    /// Unified writer: deriv=0 value only; 1 adds dv; 2 adds d2v. Pure double.
    /// Derivative-series columns are stored members — no per-call allocation.
    /// @endinternal
    template <class VType>
    void eval_impl(double t, int deriv, VType &v, VType &dv, VType &d2v) const {
        const double xi = to_xi(t), h = half_span();
        for (int c = 0; c < olen_; ++c) {
            v[c] = clenshaw(coeffs_.col(c), order_, xi);
            if (deriv >= 1) {
                dv[c] = (order_ >= 1 ? clenshaw(dcoeffs_.col(c), order_ - 1, xi) : 0.0) / h;
                if (deriv >= 2) {
                    d2v[c] =
                        (order_ >= 2 ? clenshaw(d2coeffs_.col(c), order_ - 2, xi) : 0.0) / (h * h);
                }
            }
        }
    }

    /// @brief Evaluate all output channels at physical coordinate t.
    Eigen::VectorXd eval(double t) const {
        Eigen::VectorXd v(olen_);
        eval_impl(t, 0, v, v, v);
        return v;
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
