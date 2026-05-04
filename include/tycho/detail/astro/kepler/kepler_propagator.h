// =============================================================================
// Tycho fork: Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt.
//
// VF wrapper over the LCD kernel + IFT composition; no algorithmic descent
// from ASSET's KeplerPropagator (which used a GenericFunction + ScalarRoot-
// Finder for differentiation through Newton iteration).  See
// kepler_lcd_iterate.h and kepler_propagator_ift.h for the implementation.
// =============================================================================
#pragma once
#include "tycho/detail/astro/kepler/kepler_primal_vf.h"
#include "tycho/detail/astro/kepler/kepler_propagator_ift.h"
#include "tycho/detail/astro/kepler/kepler_residual_vf.h"
#include "tycho/vector_functions.h"

#include <stdexcept>

namespace tycho::astro {

using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::MatRef;
using vf::VecRef;
using vf::VectorFunction;

class KeplerPropagator
    : public VectorFunction<KeplerPropagator, 7, 6,
                            DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic> {
public:
    using Base = VectorFunction<KeplerPropagator, 7, 6,
                                DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    static constexpr bool is_vectorizable = true;

    KeplerPropagator() : KeplerPropagator(1.0) {}
    explicit KeplerPropagator(double mu) {
        // primal_ and residual_ default-construct with the canonical
        // mu = 1.0, then set_mu propagates the user-supplied mu through
        // the wrapper-level validation (which fires before primal_'s own
        // throw, giving the user a KeplerPropagator-flavored message).
        set_mu(mu);
        this->set_io_rows(7, 6);
    }

    // Mutating mu propagates to the owned codegen VFs so their precomputed
    // sqrt(mu) / 1/sqrt(mu) caches stay coherent — every reachable state of
    // KeplerPropagator has all three mu storage sites in sync.
    void set_mu(double mu) {
        if (!(mu > 0.0))
            throw std::invalid_argument("KeplerPropagator: mu must satisfy mu > 0");
        mu_ = mu;
        primal_.set_mu(mu);
        residual_.set_mu(mu);
    }
    [[nodiscard]] double mu() const noexcept { return mu_; }

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

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x,
                                      CVecRef<OutType> fx_,
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
        detail::kepler_propagate_adjoint_hessian<Scalar>(
            R0, V0, dt, primal_, residual_, lm, out, jac, grad, hess);
        fx = out;
        jx = jac;
        ag = grad;
        ah = hess;
    }

private:
    double mu_;
    KeplerPrimal_VF primal_;
    KeplerResidual_VF residual_;
};

} // namespace tycho::astro
