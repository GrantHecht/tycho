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
        "and the Enzyme Clang plugin installed. See CLAUDE.md for the override "
        "invocation.");
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

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)

// FoR outer wrapper: computes g(x) = J(x)^T λ via Enzyme reverse mode.
//
// The trick: reverse-mode differentiation of a vector-output function with
// the OUTPUT shadow seeded to v computes v^T J in the input shadow.  Setting
// v = lam gives us the adjoint gradient g = J^T lam, written into g_data.
// We differentiate the existing compute_wrapper directly — no extra scalar
// wrapper s(x) = lam^T f(x) is needed.
//
// Scratch buffers are passed in (allocated by the caller in
// compute_adjoint_hessian_for_).  Keeping the wrapper allocation-free is
// important: Enzyme's IR analysis pulls in any Eigen::Matrix<…, Dynamic, …>
// resize/allocate machinery that lives inside the wrapper, and the resulting
// alignment-mask arithmetic emits i128 shifts that Enzyme cannot
// differentiate.
//
// Pre-conditions (caller responsibility):
//   - fx_scratch zero-initialised, length n_out.
//   - lam_scratch initialised to lam (cotangent seed), length n_out.
//   - g_data zero-initialised, length n_in.
template <class Derived>
inline void enzyme_for_outer_wrapper(const Derived* self,
                                     const double* x_data, double* g_data,
                                     double* fx_scratch, double* lam_scratch,
                                     int n_in, int n_out) {
    __enzyme_autodiff<void>(
        reinterpret_cast<void*>(&enzyme_compute_wrapper<Derived>),
        enzyme_const, self,
        enzyme_dup,   x_data,     g_data,
        enzyme_dup,   fx_scratch, lam_scratch,
        enzyme_const, n_in,
        enzyme_const, n_out);
}

#endif // TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)

// FoF inner wrapper: computes f(x) AND J(x)·dx_inner via Enzyme forward mode.
//
// Outputs:
//   fx_data        = f(x)              (primal)
//   dfx_inner_data = J(x) · dx_inner   (forward tangent)
//
// dx_inner is the fixed inner-seed (will be e_j when called from the FoF
// caller).  Same allocation-discipline as the FoR wrapper: nothing inside
// the wrapper allocates so Enzyme's analysis cannot encounter heap-resize
// alignment-mask arithmetic that produces i128 shifts.
//
// Future-work note: when this wrapper is wired through the outer
// __enzyme_fwddiff in compute_adjoint_hessian_fof_, the OUTPUT SHADOW on
// fx_data is "J(x) · e_outer" — i.e. the corresponding column of the
// Jacobian falls out of the same outer pass that computes the Hessian
// element.  The current FoF caller still recomputes the full Jacobian
// up-front via DenseFirstDerivatives::compute_jacobian_impl; a planned
// optimisation reads the column out of the outer call's fx-shadow
// instead, eliminating IR redundant Phase-1 forward sweeps per Hessian
// computation.  This combined-J-and-H property — and the corresponding
// W²-block under Phase 3 enzyme_width batching — is why FoF is retained
// as research scaffolding even though it currently loses to FoR head-to-
// head.
template <class Derived>
inline void enzyme_fof_inner_wrapper(const Derived* self,
                                     const double* x_data, double* fx_data,
                                     double* dx_inner, double* dfx_inner_data,
                                     int n_in, int n_out) {
    __enzyme_fwddiff<void>(
        reinterpret_cast<void*>(&enzyme_compute_wrapper<Derived>),
        enzyme_const, self,
        enzyme_dup,   x_data,         dx_inner,
        enzyme_dup,   fx_data,        dfx_inner_data,
        enzyme_const, n_in,
        enzyme_const, n_out);
}

#endif // TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward

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
        if constexpr (IsSuperScalar<Scalar>) {
            // Phase 5a: scalarize-per-lane dispatch.  For each lane in the
            // SuperScalar Eigen::Array, extract a double-typed input vector,
            // run the scalar EnzymeAD Jacobian, and pack the lane outputs
            // back into the SuperScalar fx / jx.
            constexpr int vsize = Scalar::SizeAtCompileTime;
            const int ir = this->input_rows();
            const int or_ = this->output_rows();

            Eigen::Matrix<double, IR, 1> x_lane(ir);
            Eigen::Matrix<double, OR, 1> fx_lane(or_);
            Eigen::Matrix<double, OR, IR> jac_lane(or_, ir);

            VecRef<OutType> fx = fx_.const_cast_derived();
            MatRef<JacType> jx = jx_.const_cast_derived();

            for (int lane = 0; lane < vsize; ++lane) {
                for (int j = 0; j < ir; ++j) x_lane[j] = x[j][lane];
                fx_lane.setZero();
                jac_lane.setZero();

                this->scalar_compute_jacobian_impl(x_lane, fx_lane, jac_lane);

                for (int j = 0; j < or_; ++j) fx[j][lane] = fx_lane[j];
                for (int i = 0; i < ir; ++i)
                    for (int j = 0; j < or_; ++j)
                        jx(j, i)[lane] = jac_lane(j, i);
            }
        } else {
            this->scalar_compute_jacobian_impl(x, fx_, jx_);
        }
    }

    // Phase 1 scalar Jacobian body. Templated on the local types so the
    // SuperScalar dispatch can pass plain double-typed Eigen::Matrix locals.
    // Inputs are read; fx / jx outputs are written via const_cast_derived.
    template <class XLocal, class FxLocal, class JacLocal>
    inline void scalar_compute_jacobian_impl(const Eigen::MatrixBase<XLocal>& x,
                                             const Eigen::MatrixBase<FxLocal>& fx_,
                                             const Eigen::MatrixBase<JacLocal>& jx_) const {
        FxLocal& fx = fx_.const_cast_derived();
        JacLocal& jx = jx_.const_cast_derived();

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

//! Second derivatives via Enzyme nested AD (Phase 2).
/*!
  Pipeline:
  1. fx and jx come from the JMode-driven first-derivative path
     (DenseFirstDerivatives<..., JMode>::compute_jacobian_impl).
  2. The adjoint gradient gx = J^T lam is computed from jx by direct
     matrix multiply — Enzyme already produced jx, no need to recompute.
  3. The adjoint Hessian hx is the column-by-column derivative of
     g(x) = J(x)^T lam with respect to x, dispatched at configure time
     via TYCHO_ENZYME_HESSIAN_STRATEGY:

       ForwardOverReverse (default):  __enzyme_fwddiff over a wrapper
                                      that calls __enzyme_autodiff.
                                      O(IR) outer Enzyme calls.
       ForwardOverForward:            __enzyme_fwddiff over a wrapper
                                      that calls __enzyme_fwddiff.
                                      O(IR^2) outer Enzyme calls.

  Phase 2's head-to-head benchmark picks FoR as the production default
  (4–9× faster on every Hessian micro-benchmark; tied on full-solve TTS).
  FoF is retained as internal research scaffolding — its outer fwddiff
  pass naturally produces J(x)·e_i alongside the Hessian element, and a
  future commit can reimplement compute_adjoint_hessian_fof_ to compute
  Jacobian and Hessian simultaneously instead of calling
  compute_jacobian_impl up-front.  Combined with Phase 3's enzyme_width
  batching, that path could yield a W-column Jacobian slab and a W²
  Hessian block per outer call.

  Strategy is a private (cmake-time) selector — DenseDerivativeMode does
  not surface a public FoF mode.  Promoting FoF to a public mode is a
  future option once the combined-J+H optimisation is implemented and
  benchmarked.
*/
template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeMode::EnzymeAD>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, JMode>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        using Scalar = typename InType::Scalar;

        if constexpr (IsSuperScalar<Scalar>) {
            // Phase 5a: scalarize-per-lane Hessian. For each lane, extract
            // double-typed locals, run the scalar Phase 2 Hessian, pack
            // results back into the SuperScalar fx/jx/gx/hx.
            constexpr int vsize = Scalar::SizeAtCompileTime;
            const int ir = this->input_rows();
            const int or_ = this->output_rows();

            VecRef<OutType> fx = fx_.const_cast_derived();
            MatRef<JacType> jx = jx_.const_cast_derived();
            VecRef<AdjGradType> gx = adjgrad_.const_cast_derived();
            MatRef<AdjHessType> hx = adjhess_.const_cast_derived();

            Eigen::Matrix<double, IR, 1> x_lane(ir);
            Eigen::Matrix<double, OR, 1> fx_lane(or_);
            Eigen::Matrix<double, OR, IR> jac_lane(or_, ir);
            Eigen::Matrix<double, IR, 1> g_lane(ir);
            Eigen::Matrix<double, IR, IR> h_lane(ir, ir);
            Eigen::Matrix<double, OR, 1> lam_lane(or_);

            for (int lane = 0; lane < vsize; ++lane) {
                for (int j = 0; j < ir; ++j) x_lane[j] = x[j][lane];
                for (int k = 0; k < or_; ++k) lam_lane[k] = adjvars[k][lane];
                fx_lane.setZero();
                jac_lane.setZero();
                g_lane.setZero();
                h_lane.setZero();

                this->scalar_compute_jacobian_adjointgradient_adjointhessian_impl(
                    x_lane, fx_lane, jac_lane, g_lane, h_lane, lam_lane);

                for (int j = 0; j < or_; ++j) fx[j][lane] = fx_lane[j];
                for (int i = 0; i < ir; ++i) {
                    gx[i][lane] = g_lane[i];
                    for (int j = 0; j < or_; ++j)
                        jx(j, i)[lane] = jac_lane(j, i);
                    for (int j = 0; j < ir; ++j)
                        hx(j, i)[lane] = h_lane(j, i);
                }
            }
        } else {
            this->scalar_compute_jacobian_adjointgradient_adjointhessian_impl(
                x, fx_, jx_, adjgrad_, adjhess_, adjvars);
        }
    }

    // Phase 1/2 scalar body, factored out so the SuperScalar dispatch can
    // call it per lane with already-double-typed Eigen::Matrix locals.
    template <class XLocal, class FxLocal, class JacLocal, class AdjGradLocal,
              class AdjHessLocal, class AdjVarLocal>
    inline void scalar_compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<XLocal>& x, const Eigen::MatrixBase<FxLocal>& fx_,
        const Eigen::MatrixBase<JacLocal>& jx_,
        const Eigen::MatrixBase<AdjGradLocal>& adjgrad_,
        const Eigen::MatrixBase<AdjHessLocal>& adjhess_,
        const Eigen::MatrixBase<AdjVarLocal>& adjvars) const {

        FxLocal& fx = fx_.const_cast_derived();
        JacLocal& jx = jx_.const_cast_derived();
        AdjGradLocal& gx = adjgrad_.const_cast_derived();
        AdjHessLocal& hx = adjhess_.const_cast_derived();

        const int ir = this->input_rows();
        const int or_ = this->output_rows();

        // Step 1: fx and jx via the JMode first-derivative path.
        this->compute_jacobian_impl(x, fx_, jx_);

        // Step 2: gx = J^T lam (closed-form from the Jacobian).
        for (int i = 0; i < ir; ++i) {
            double acc = 0.0;
            for (int k = 0; k < or_; ++k) acc += jx(k, i) * adjvars[k];
            gx[i] = acc;
        }

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)
        compute_adjoint_hessian_for_(x, hx, adjvars, ir, or_);
#elif defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)
        compute_adjoint_hessian_fof_(x, hx, adjvars, ir, or_);
#else
#  error "TYCHO_ENZYME_HESSIAN_STRATEGY_<Strategy> must be defined."
#endif
    }

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)
    template <class InType, class AdjHessType, class AdjVarType>
    inline void compute_adjoint_hessian_for_(
        CVecRef<InType> x,
        MatRef<AdjHessType> hx,
        CVecRef<AdjVarType> adjvars,
        int ir, int or_) const {

        Eigen::Matrix<double, IR, 1> x_local = x;
        Eigen::Matrix<double, IR, 1> dx_local(ir);
        Eigen::Matrix<double, IR, 1> g_local(ir);
        Eigen::Matrix<double, IR, 1> dg_local(ir);
        // Scratch buffers passed into the FoR wrapper.  Allocated here so the
        // wrapper itself stays allocation-free — see the Enzyme i128-shift
        // hazard note in the wrapper docstring.
        Eigen::Matrix<double, OR, 1> fx_scratch(or_);
        Eigen::Matrix<double, OR, 1> lam_scratch(or_);

        const Derived* self = static_cast<const Derived*>(this);

        for (int i = 0; i < ir; ++i) {
            dx_local.setZero();
            dx_local[i] = 1.0;
            g_local.setZero();
            dg_local.setZero();
            // Re-prime the scratch each iteration: the inner reverse pass
            // consumes fx_scratch and may modify lam_scratch.
            for (int k = 0; k < or_; ++k) {
                fx_scratch[k] = 0.0;
                lam_scratch[k] = adjvars[k];
            }

            __enzyme_fwddiff<void>(
                reinterpret_cast<void*>(&detail::enzyme_for_outer_wrapper<Derived>),
                enzyme_const, self,
                enzyme_dup,   x_local.data(),    dx_local.data(),
                enzyme_dup,   g_local.data(),    dg_local.data(),
                enzyme_const, fx_scratch.data(),
                enzyme_const, lam_scratch.data(),
                enzyme_const, ir,
                enzyme_const, or_);

            for (int j = 0; j < ir; ++j) hx(j, i) = dg_local[j];
        }

        // Symmetrize. Mathematically the Hessian is symmetric; numerically
        // each column may differ from its row-counterpart by ~1e-13.
        for (int i = 0; i < ir; ++i) {
            for (int j = i + 1; j < ir; ++j) {
                double avg = 0.5 * (hx(i, j) + hx(j, i));
                hx(i, j) = avg;
                hx(j, i) = avg;
            }
        }
    }
#endif // TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)
    template <class InType, class AdjHessType, class AdjVarType>
    inline void compute_adjoint_hessian_fof_(
        CVecRef<InType> x,
        MatRef<AdjHessType> hx,
        CVecRef<AdjVarType> adjvars,
        int ir, int or_) const {

        Eigen::Matrix<double, IR, 1> x_local = x;
        Eigen::Matrix<double, IR, 1> dx_outer(ir);
        Eigen::Matrix<double, IR, 1> dx_inner(ir);
        Eigen::Matrix<double, OR, 1> fx_primal(or_);
        Eigen::Matrix<double, OR, 1> dfx_outer_shadow(or_);     // d/dx of fx, ignored
        Eigen::Matrix<double, OR, 1> dfx_inner_primal(or_);     // J · e_j primal
        Eigen::Matrix<double, OR, 1> ddfx_outer_shadow(or_);    // ∂(J·e_j)/∂x · e_i

        const Derived* self = static_cast<const Derived*>(this);

        // O(IR^2) outer fwddiff calls; each yields ∂²f_k/∂x_i∂x_j for all k.
        for (int i = 0; i < ir; ++i) {
            dx_outer.setZero();
            dx_outer[i] = 1.0;

            for (int j = 0; j < ir; ++j) {
                dx_inner.setZero();
                dx_inner[j] = 1.0;
                fx_primal.setZero();
                dfx_outer_shadow.setZero();
                dfx_inner_primal.setZero();
                ddfx_outer_shadow.setZero();

                __enzyme_fwddiff<void>(
                    reinterpret_cast<void*>(&detail::enzyme_fof_inner_wrapper<Derived>),
                    enzyme_const, self,
                    enzyme_dup,   x_local.data(),         dx_outer.data(),
                    enzyme_dup,   fx_primal.data(),       dfx_outer_shadow.data(),
                    enzyme_const, dx_inner.data(),
                    enzyme_dup,   dfx_inner_primal.data(),
                                  ddfx_outer_shadow.data(),
                    enzyme_const, ir,
                    enzyme_const, or_);

                // ddfx_outer_shadow[k] = ∂²f_k/∂x_i∂x_j; sum with adjvars.
                double acc = 0.0;
                for (int k = 0; k < or_; ++k) acc += adjvars[k] * ddfx_outer_shadow[k];
                hx(i, j) = acc;
            }
        }

        // Symmetrize. Mathematically the Hessian is symmetric; numerically each
        // (i,j) and (j,i) pair may differ by ~1e-13.
        for (int i = 0; i < ir; ++i) {
            for (int j = i + 1; j < ir; ++j) {
                double avg = 0.5 * (hx(i, j) + hx(j, i));
                hx(i, j) = avg;
                hx(j, i) = avg;
            }
        }
    }
#endif // TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward
};

#endif // TYCHO_HAS_ENZYME_AD

} // namespace tycho::vf
