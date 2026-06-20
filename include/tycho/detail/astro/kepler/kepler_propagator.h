// =============================================================================
// Tycho fork: Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt.
//
// VF wrapper over the LCD kernel + IFT composition.  See kepler_lcd_iterate.h
// and kepler_propagator_ift.h for the implementation.
// =============================================================================
#pragma once
#include "tycho/detail/astro/kepler/kepler_primal.h"
#include "tycho/detail/astro/kepler/kepler_propagator_ift.h"
#include "tycho/detail/astro/kepler/kepler_residual.h"
#include "tycho/vector_functions.h"

#include <limits>
#include <stdexcept>

namespace tycho::astro {

using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::MatRef;
using vf::VecRef;
using vf::VectorFunction;

/// @brief Universal-variable Kepler propagator VectorFunction (IR=7, OR=6).
///
/// Maps [rx, ry, rz, vx, vy, vz, dt] to the propagated Cartesian state
/// [rx', ry', rz', vx', vy', vz'] using the LCD kernel with IFT-composed
/// analytic Jacobian and Hessian.
class KeplerPropagator
    : public VectorFunction<KeplerPropagator, 7, 6, DenseDerivativeMode::Analytic,
                            DenseDerivativeMode::Analytic> {
  public:
    using Base = VectorFunction<KeplerPropagator, 7, 6, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>; ///< @internal CRTP base alias. @endinternal
    VF_TYPE_ALIASES(Base);

    static constexpr bool is_vectorizable = true; ///< @internal Enable SuperScalar dispatch. @endinternal

    /// @brief Default constructor; sets mu=1.
    KeplerPropagator() : KeplerPropagator(1.0) {}
    /// @brief Construct with the given gravitational parameter.
    /// @param[in] mu Gravitational parameter (must be > 0).
    explicit KeplerPropagator(double mu) {
        set_mu(mu);
        this->set_io_rows(7, 6);
    }

    /// @brief Set the gravitational parameter (propagates to internal codegen VFs).
    /// @param[in] mu Gravitational parameter (must be > 0).
    void set_mu(double mu) {
        if (!(mu > 0.0))
            throw std::invalid_argument("KeplerPropagator: mu must satisfy mu > 0");
        mu_ = mu;
        primal_.set_mu(mu);
        residual_.set_mu(mu);
    }
    /// @brief Return the current gravitational parameter.
    [[nodiscard]] double mu() const noexcept { return mu_; }

    /// @internal @brief Compute propagated Cartesian state (primal only). @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        detail::kepler_propagate<Scalar>(R0, V0, dt, mu_, out);
        fx = out;
    }

    /// @internal @brief Compute propagated state and 6×7 Jacobian via IFT composition. @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        Eigen::Matrix<Scalar, 6, 7> jac;
        detail::kepler_propagate_jacobian<Scalar>(R0, V0, dt, primal_, residual_, out, jac);
        fx = out;
        jx = jac;
    }

    /// @internal @brief Compute propagated state, Jacobian, adjoint gradient, and adjoint Hessian. @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> ag = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> ah = adjhess_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        Eigen::Matrix<Scalar, 6, 7> jac;
        Vector7<Scalar> grad;
        Eigen::Matrix<Scalar, 7, 7> hess;
        Vector6<Scalar> lm = adjvars;
        detail::kepler_propagate_adjoint_hessian<Scalar>(R0, V0, dt, primal_, residual_, lm, out,
                                                         jac, grad, hess);
        fx = out;
        jx = jac;
        ag = grad;
        ah = hess;
    }

  private:
    double mu_ = std::numeric_limits<double>::quiet_NaN();
    detail::KeplerPrimal primal_;
    detail::KeplerResidual residual_;
};

} // namespace tycho::astro
