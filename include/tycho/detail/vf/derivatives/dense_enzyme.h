// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// EnzymeAD specializations for DenseFirstDerivatives / DenseSecondDerivatives.
// Self-guarded on TYCHO_HAS_ENZYME_AD; safe to include unconditionally.
// =============================================================================
#pragma once

#include "tycho/detail/vf/derivatives/dense_derivatives.h"

namespace tycho::vf {

#ifndef TYCHO_HAS_ENZYME_AD

namespace detail {
template <class> struct enzyme_dependent_false : std::false_type {};
} // namespace detail

template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::EnzymeAD>
    : DenseFunctionBase<Derived, IR, OR> {
    static_assert(detail::enzyme_dependent_false<Derived>::value,
        "DenseDerivativeMode::EnzymeAD requires CMake option ENABLE_ENZYME_AD=ON "
        "and the Enzyme Clang plugin installed. See CLAUDE.md for the override "
        "invocation.");
};

template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeMode::EnzymeAD>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    static_assert(detail::enzyme_dependent_false<Derived>::value,
        "DenseDerivativeMode::EnzymeAD Hessian requires CMake option ENABLE_ENZYME_AD=ON "
        "and the Enzyme Clang plugin installed. Note: EnzymeAD Hessian also requires "
        "Phase 2 of the Enzyme rollout to have landed; before Phase 2, pair "
        "<EnzymeAD, AutodiffFwd> instead.");
};

#else // TYCHO_HAS_ENZYME_AD

// Enzyme magic externs.  These are sentinel symbols Enzyme recognises at the
// IR level; they are never linked.  Declaring them as extern "C" int avoids
// C++ name mangling so the plugin sees the canonical names.
extern "C" {
extern int enzyme_dup;
extern int enzyme_const;
extern int enzyme_dupnoneed;
extern int enzyme_width;
}

template <typename RT, typename... Args> RT __enzyme_fwddiff(Args...);
template <typename RT, typename... Args> RT __enzyme_autodiff(Args...);

namespace detail {

// Free-function wrapper for Enzyme to differentiate.  Enzyme operates on plain
// function pointers and looks at the LLVM IR of the callee, so we expose the
// user's `compute_impl<double>` body via a flat C-style entry point.  The Eigen
// Maps are constructed inside so the body is fully visible to LLVM by the time
// Enzyme runs the derivative pass.
//
// The Map element type matches the EnzymeAD path's primal type (`double`) and
// the Maps use `Eigen::Dynamic` size template parameters because this wrapper
// is a free function — it cannot see the surrounding struct's compile-time IR
// / OR template parameters.  Performance is unaffected: the runtime n_in /
// n_out arguments are always the actual sizes for the active VF.
template <class Derived>
inline void enzyme_compute_wrapper(const Derived* self,
                                   const double* x_data, double* fx_data,
                                   int n_in, int n_out) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x(x_data, n_in);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> fx(fx_data, n_out);
    self->compute_impl(x, fx);
}

} // namespace detail

//! First derivatives via Enzyme forward-mode AD.
/*!
  Per-column dispatch: for each input dimension i in [0, IR), seed the tangent
  vector with e_i and call __enzyme_fwddiff once.  Reads back column i of the
  Jacobian and (on i==0) the primal output.  Mirrors AutodiffFwd's structure
  exactly so the two paths are interchangeable.

  Phase 1 only supports scalar dispatch.  SuperScalar/Vectorizable specializations
  are deferred to Phase 5; until then, attempting EnzymeAD on a Vectorizable
  VectorFunction triggers a static_assert with a guiding message.
*/
template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::EnzymeAD>
    : DenseFunctionBase<Derived, IR, OR> {
    using Base = DenseFunctionBase<Derived, IR, OR>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        using Scalar = typename InType::Scalar;
        static_assert(!IsSuperScalar<Scalar>,
            "DenseDerivativeMode::EnzymeAD does not yet support SuperScalar/"
            "Vectorizable dispatch. Vectorized EnzymeAD is planned for Phase 5; "
            "until then, do not mark EnzymeAD VectorFunctions as "
            "Vectorizable<Derived>=true.");

        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        const int ir = this->input_rows();
        const int or_ = this->output_rows();

        // Contiguous double-precision scratch.  Using IR / OR template parameters
        // gives a fixed-size Eigen::Matrix when IR/OR are known at compile time
        // and a dynamic one when they are -1; the runtime ir / or_ resize args
        // apply only in the dynamic case.
        Eigen::Matrix<double, IR, 1> x_local = x;
        Eigen::Matrix<double, IR, 1> dx_local(ir);
        Eigen::Matrix<double, OR, 1> fx_local(or_);
        Eigen::Matrix<double, OR, 1> dfx_local(or_);

        const Derived* self = static_cast<const Derived*>(this);

        for (int i = 0; i < ir; i++) {
            dx_local.setZero();
            dx_local[i] = 1.0;
            fx_local.setZero();
            dfx_local.setZero();

            // enzyme_const on `self` -> stored constants (mu_, g_, ...) are NOT
            // differentiated through.  enzyme_dup pairs each primal pointer
            // with its tangent.  enzyme_const on the int sizes prevents Enzyme
            // from synthesizing tangents for them.
            __enzyme_fwddiff<void>(
                reinterpret_cast<void*>(&detail::enzyme_compute_wrapper<Derived>),
                enzyme_const, self,
                enzyme_dup,   x_local.data(),  dx_local.data(),
                enzyme_dup,   fx_local.data(), dfx_local.data(),
                enzyme_const, ir,
                enzyme_const, or_);

            for (int j = 0; j < or_; j++) {
                jx(j, i) = dfx_local[j];
            }
            if (i == 0) {
                for (int j = 0; j < or_; j++) {
                    fx[j] = fx_local[j];
                }
            }
        }
    }
};

#endif // TYCHO_HAS_ENZYME_AD

} // namespace tycho::vf
