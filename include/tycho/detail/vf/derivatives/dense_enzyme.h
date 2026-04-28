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
//
// enzyme_dup    — scalar duplicate (primal_ptr, shadow_ptr) pair.
// enzyme_dupv   — vectorized duplicate; used WITH enzyme_width.  Arg form is
//                 (enzyme_dupv, byte_stride, primal_ptr, shadow_base_ptr).
//                 Shadow buffer holds W lanes laid out SoA-by-lane: lane k
//                 starts at shadow_base_ptr + k*byte_stride.
// enzyme_const  — argument is not differentiated.
// enzyme_width  — followed by the integer batch width W; placed immediately
//                 after the function pointer.
extern "C" {
extern int enzyme_dup;
extern int enzyme_dupv;
extern int enzyme_const;
extern int enzyme_dupnoneed;
extern int enzyme_width;
}

template <typename RT, typename... Args> RT __enzyme_fwddiff(Args...);
template <typename RT, typename... Args> RT __enzyme_autodiff(Args...);

// Opt-out marker for Phase 6 SIMD Hessian.  Any Vectorizable+EnzymeAD VF is
// eligible by default; VFs whose body is too dense for Enzyme's SS
// reverse-mode IR analysis (e.g. composite trig+sqrt+division bodies like
// MEE) can disable per-VF by declaring:
//
//   static constexpr bool enzyme_simd_hessian_supported = false;
//
// Such VFs fall through to the Phase 5a scalarize-per-lane fallback.
template <class T>
constexpr bool enzyme_simd_hessian_eligible() {
    if constexpr (requires { T::enzyme_simd_hessian_supported; })
        return static_cast<bool>(T::enzyme_simd_hessian_supported);
    else
        return true;
}

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

// Phase 5b SIMD wrapper.  Same structure as enzyme_compute_wrapper but the
// scalar type is the user's SuperScalar Eigen::Array<double, W, 1>.  Enzyme
// differentiates through the SuperScalar ops directly (vector LLVM types),
// producing W lane-local tangents per single __enzyme_fwddiff call — true
// SIMD differentiation, not scalarize-per-lane.  Only enabled when the
// VectorFunction opts in via Vectorizable<Derived>=true.
template <class Derived, class SSType>
inline void enzyme_compute_wrapper_simd(const Derived* self,
                                        const SSType* x_data, SSType* fx_data,
                                        int n_in, int n_out) {
    Eigen::Map<const Eigen::Matrix<SSType, Eigen::Dynamic, 1>> x(x_data, n_in);
    Eigen::Map<Eigen::Matrix<SSType, Eigen::Dynamic, 1>> fx(fx_data, n_out);
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

// Phase 6 SIMD twin of enzyme_for_outer_wrapper: computes g(x) = J(x)^T λ
// under SuperScalar arithmetic, so the outer __enzyme_fwddiff in
// compute_adjoint_hessian_for_simd_ propagates W lane-local Hessian columns
// per call.  Differentiates enzyme_compute_wrapper_simd<Derived, SSType>
// reused unchanged from the Phase 5b Jacobian path.
//
// Allocation discipline (CRITICAL): no Eigen::Map construction, no resize,
// no heap touches inside this function.  Enzyme's IR analysis pulls in any
// heap-resize alignment-mask arithmetic that lives in the differentiated
// callee, and the resulting i128 shifts cannot be lowered.  Caller (the
// SIMD Hessian helper) is responsible for sizing fx_scratch / lam_scratch
// to n_out at compile time.
template <class Derived, class SSType>
inline void enzyme_for_outer_wrapper_simd(const Derived* self,
                                          const SSType* x_data, SSType* g_data,
                                          SSType* fx_scratch, SSType* lam_scratch,
                                          int n_in, int n_out) {
    __enzyme_autodiff<void>(
        reinterpret_cast<void*>(&enzyme_compute_wrapper_simd<Derived, SSType>),
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

// Phase 7 SIMD twin of enzyme_fof_inner_wrapper.  All forward mode (no
// SuperScalar reverse-mode tape — the inner is fwddiff, the outer is
// also fwddiff).  Reuses Phase 5b's SS primal entry point.
//
// Allocation discipline (CRITICAL): no Eigen::Map construction, no
// resize, no heap touches inside this function.  Same i128-shift hazard
// as the Phase 6 SIMD wrappers; caller sizes all buffers at compile-time.
template <class Derived, class SSType>
inline void enzyme_fof_inner_wrapper_simd(const Derived* self,
                                          const SSType* x_data, SSType* fx_data,
                                          SSType* dx_inner, SSType* dfx_inner_data,
                                          int n_in, int n_out) {
    __enzyme_fwddiff<void>(
        reinterpret_cast<void*>(&enzyme_compute_wrapper_simd<Derived, SSType>),
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
  Jacobian and (on i==0) the primal output.

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
            if constexpr (Vectorizable<Derived>) {
                // Phase 5b: direct-SIMD Enzyme.  The user's compute_impl runs
                // on Eigen::Array<double, W, 1> SIMD ops; Enzyme differentiates
                // through them producing W per-lane tangents per call.
                //
                // Benchmarks (see /tmp/enzyme_phase5b_findings.md): Phase 5b
                // wins ~25-37% per-lane on small bodies (Brach 5->3, CR3BP
                // 7->6) vs scalarize-per-lane.  Loses ~1.7-6x on bodies
                // with many transcendentals (MEE 9->6 with sqrt/sin/cos)
                // because the SIMD math units don't keep up with Enzyme's
                // analysis through Eigen's per-Scalar function dispatch.
                // Net: Phase 5b is the right default for Vectorizable+SS;
                // MEE-class users can fall back to scalar_compute_jacobian_impl
                // via Vectorizable=false (base class will scalarize).
                simd_compute_jacobian_impl(x, fx_, jx_);
            } else {
                // Phase 5a fallback path.  Reachable only if compute_jacobian_impl
                // is called directly with SS Scalar on a non-Vectorizable VF
                // (the base-class compute_jacobian dispatch normally scalarizes
                // before reaching this point for Vectorizable=false).
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
            }
        } else {
            this->scalar_compute_jacobian_impl(x, fx_, jx_);
        }
    }

    // Phase 5b SIMD Jacobian: __enzyme_fwddiff over a SuperScalar-typed
    // wrapper.  Phase 3 batching (enzyme_width=BW) is layered on top so a
    // single call propagates BW tangents simultaneously with each lane in W
    // SIMD lanes — total BW*W parallelism per Enzyme invocation.
    //
    // Tail handles the (ir % BW != 0) remainder via single-tangent SIMD
    // calls (BW=1 path).  Requires Vectorizable<Derived> (templated
    // compute_impl that accepts SuperScalar Eigen::Matrix inputs).
    template <class InType, class OutType, class JacType>
    inline void simd_compute_jacobian_impl(CVecRef<InType> x,
                                           CVecRef<OutType> fx_,
                                           CMatRef<JacType> jx_) const {
        using Scalar = typename InType::Scalar;
        constexpr int BW = TYCHO_ENZYME_BATCH_WIDTH;

        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        const int ir = this->input_rows();
        const int or_ = this->output_rows();

        Eigen::Matrix<Scalar, IR, 1> x_local = x;
        Eigen::Matrix<Scalar, IR, 1> dx_local(ir);
        Eigen::Matrix<Scalar, OR, 1> fx_local(or_);
        Eigen::Matrix<Scalar, OR, 1> dfx_local(or_);

        const Derived* self = static_cast<const Derived*>(this);

        int i = 0;

        if constexpr (BW > 1) {
            // Batched Phase 3 + Phase 5b SIMD: enzyme_width=BW with
            // enzyme_dupv shadows of SuperScalar elements.  Each shadow
            // column is one tangent vector (Scalar = SS); BW columns =
            // BW tangents stepped at byte stride IR*sizeof(SS).
            Eigen::Matrix<Scalar, IR, BW> dx_shadow(ir, BW);
            Eigen::Matrix<Scalar, OR, BW> dfx_shadow(or_, BW);
            const long stride_x = static_cast<long>(ir) * static_cast<long>(sizeof(Scalar));
            const long stride_y = static_cast<long>(or_) * static_cast<long>(sizeof(Scalar));

            const int ir_chunked = (ir / BW) * BW;
            for (; i < ir_chunked; i += BW) {
                dx_shadow.setZero();
                for (int b = 0; b < BW; ++b)
                    dx_shadow(i + b, b).setConstant(1.0);
                dfx_shadow.setZero();
                fx_local.setZero();

                __enzyme_fwddiff<void>(
                    reinterpret_cast<void*>(
                        &detail::enzyme_compute_wrapper_simd<Derived, Scalar>),
                    enzyme_width, BW,
                    enzyme_const, self,
                    enzyme_dupv, stride_x,
                        x_local.data(), dx_shadow.data(),
                    enzyme_dupv, stride_y,
                        fx_local.data(), dfx_shadow.data(),
                    enzyme_const, ir,
                    enzyme_const, or_);

                for (int b = 0; b < BW; ++b)
                    for (int k = 0; k < or_; ++k)
                        jx(k, i + b) = dfx_shadow(k, b);
                if (i == 0)
                    for (int k = 0; k < or_; ++k) fx[k] = fx_local[k];
            }
        }

        // Tail (or full when BW=1): single-tangent SIMD fwddiff.
        for (; i < ir; ++i) {
            dx_local.setZero();
            dx_local[i].setConstant(1.0);  // unit tangent for ALL lanes
            fx_local.setZero();
            dfx_local.setZero();

            __enzyme_fwddiff<void>(
                reinterpret_cast<void*>(
                    &detail::enzyme_compute_wrapper_simd<Derived, Scalar>),
                enzyme_const, self,
                enzyme_dup,   x_local.data(),  dx_local.data(),
                enzyme_dup,   fx_local.data(), dfx_local.data(),
                enzyme_const, ir,
                enzyme_const, or_);

            for (int k = 0; k < or_; ++k) jx(k, i) = dfx_local[k];
            if (i == 0)
                for (int k = 0; k < or_; ++k) fx[k] = fx_local[k];
        }
    }

    // Phase 1 / Phase 3 scalar Jacobian body.  Templated on the local types
    // so the SuperScalar dispatch can pass plain double-typed Eigen::Matrix
    // locals.  Inputs are read; fx / jx outputs are written via
    // const_cast_derived.
    //
    // Phase 3 batching: when TYCHO_ENZYME_BATCH_WIDTH > 1, the input-dim
    // loop is chunked W at a time.  Each batched call uses
    // __enzyme_fwddiff(..., enzyme_width, W, enzyme_dupv, ...) to propagate
    // W tangents simultaneously.  Shadow buffers are SoA-by-lane: lane k's
    // tangent vector occupies shadow[k*ir .. k*ir+ir-1].  The tail
    // (ir % W != 0) falls back to single-tangent calls (W=1 path).
    template <class XLocal, class FxLocal, class JacLocal>
    inline void scalar_compute_jacobian_impl(const Eigen::MatrixBase<XLocal>& x,
                                             const Eigen::MatrixBase<FxLocal>& fx_,
                                             const Eigen::MatrixBase<JacLocal>& jx_) const {
        constexpr int W = TYCHO_ENZYME_BATCH_WIDTH;
        static_assert(W == 1 || W == 4 || W == 8,
            "TYCHO_ENZYME_BATCH_WIDTH must be 1, 4, or 8.");

        FxLocal& fx = fx_.const_cast_derived();
        JacLocal& jx = jx_.const_cast_derived();

        const int ir = this->input_rows();
        const int or_ = this->output_rows();

        // Contiguous double-precision scratch.  IR/OR template params give a
        // fixed-size Eigen::Matrix when known at compile time, runtime-sized
        // otherwise.  Per-lane shadows allocate dynamically when W > 1
        // (size = ir*W and or_*W).
        Eigen::Matrix<double, IR, 1> x_local = x;
        Eigen::Matrix<double, IR, 1> dx_local(ir);
        Eigen::Matrix<double, OR, 1> fx_local(or_);
        Eigen::Matrix<double, OR, 1> dfx_local(or_);

        const Derived* self = static_cast<const Derived*>(this);

        int i = 0;

        if constexpr (W > 1) {
            // Batched fwddiff over W input-dim columns at a time.
            //
            // Shadow buffers are Eigen::Matrix<double, IR, W> (column-major)
            // and Eigen::Matrix<double, OR, W>.  Column w == lane w's tangent
            // vector; column-major layout means consecutive lanes are
            // adjacent in memory at stride IR*sizeof(double) — exactly the
            // SoA-by-lane layout enzyme_dupv expects.  Using Eigen::Matrix
            // (vs std::vector) keeps the shadow stack-allocated when IR/OR
            // are compile-time constants, avoiding per-call heap-alloc cost
            // that dominated for small IR (Brach was 4× slower than W=1
            // before this).
            Eigen::Matrix<double, IR, W> dx_shadow(ir, W);
            Eigen::Matrix<double, OR, W> dfx_shadow(or_, W);
            const long stride_x = static_cast<long>(ir) * static_cast<long>(sizeof(double));
            const long stride_y = static_cast<long>(or_) * static_cast<long>(sizeof(double));

            const int ir_chunked = (ir / W) * W;
            for (; i < ir_chunked; i += W) {
                // Re-prime tangent vectors: lane w gets the unit tangent e_{i+w}.
                dx_shadow.setZero();
                for (int w = 0; w < W; ++w) dx_shadow(i + w, w) = 1.0;
                dfx_shadow.setZero();
                fx_local.setZero();

                __enzyme_fwddiff<void>(
                    reinterpret_cast<void*>(&detail::enzyme_compute_wrapper<Derived>),
                    enzyme_width, W,
                    enzyme_const, self,
                    enzyme_dupv, stride_x,
                        x_local.data(), dx_shadow.data(),
                    enzyme_dupv, stride_y,
                        fx_local.data(), dfx_shadow.data(),
                    enzyme_const, ir,
                    enzyme_const, or_);

                // Lane w (= column w of dfx_shadow) → column i+w of the Jacobian.
                for (int w = 0; w < W; ++w)
                    for (int k = 0; k < or_; ++k)
                        jx(k, i + w) = dfx_shadow(k, w);

                if (i == 0)
                    for (int k = 0; k < or_; ++k) fx[k] = fx_local[k];
            }
        }

        // Tail (or full loop when W=1): single-tangent __enzyme_fwddiff.
        for (; i < ir; ++i) {
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
#if defined(TYCHO_ENZYME_SIMD_HESSIAN_ENABLED) \
    && defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)
            if constexpr (Vectorizable<Derived>
                          && JMode == DenseDerivativeMode::EnzymeAD
                          && enzyme_simd_hessian_eligible<Derived>()) {
                // Phase 6 direct-SIMD FoR Hessian.  Routes through Phase 5b
                // for fx + jx and the SIMD FoR helper for hx.  Falls through
                // to Phase 5a otherwise.
                simd_compute_jacobian_adjointgradient_adjointhessian_impl(
                    x, fx_, jx_, adjgrad_, adjhess_, adjvars);
                return;
            }
#elif defined(TYCHO_ENZYME_SIMD_HESSIAN_ENABLED) \
    && defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)
            if constexpr (Vectorizable<Derived>
                          && JMode == DenseDerivativeMode::EnzymeAD) {
                // Phase 7 direct-SIMD FoF combined J+H Hessian.  No
                // enzyme_simd_hessian_eligible<> gate: FoF avoids the SS
                // reverse-mode tape that forced MEE-class VFs to opt out
                // of the FoR SIMD path.
                simd_compute_jacobian_adjointgradient_adjointhessian_impl(
                    x, fx_, jx_, adjgrad_, adjhess_, adjvars);
                return;
            }
#endif
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

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)
        // Step 1: fx and jx via the JMode first-derivative path.
        this->compute_jacobian_impl(x, fx_, jx_);

        // Step 2: gx = J^T lam (closed-form from the Jacobian).
        for (int i = 0; i < ir; ++i) {
            double acc = 0.0;
            for (int k = 0; k < or_; ++k) acc += jx(k, i) * adjvars[k];
            gx[i] = acc;
        }

        // Step 3: hx via FoR.
        compute_adjoint_hessian_for_(x, hx, adjvars, ir, or_);
#elif defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)
        // Phase 7: combined J+H — the FoF helper writes fx, jx, AND hx in
        // one set of outer fwddiff calls (column i of J is read from the
        // outer fx-shadow that was previously discarded).  Saves the
        // up-front compute_jacobian_impl call.
        compute_jacobian_adjoint_hessian_fof_(x, fx_, jx_, hx, adjvars, ir, or_);

        // Step 2: gx = J^T lam (still a separate pass — no Enzyme call).
        for (int i = 0; i < ir; ++i) {
            double acc = 0.0;
            for (int k = 0; k < or_; ++k) acc += jx(k, i) * adjvars[k];
            gx[i] = acc;
        }
#else
#  error "TYCHO_ENZYME_HESSIAN_STRATEGY_<Strategy> must be defined."
#endif
    }

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)
    // Phase 6 SIMD direct path: the SuperScalar twin of
    // scalar_compute_jacobian_adjointgradient_adjointhessian_impl.  Public so
    // tests can call it directly to A/B against the Phase 5a fallback.
    //
    //   1. fx + jx via Phase 5b SIMD Jacobian (simd_compute_jacobian_impl).
    //   2. gx = J^T λ on SSType.
    //   3. hx via direct-SIMD FoR (compute_adjoint_hessian_for_simd_).
    //
    // Requires Vectorizable<Derived> and JMode == EnzymeAD (the parent
    // simd_compute_jacobian_impl is only defined for the EnzymeAD JMode
    // specialisation of DenseFirstDerivatives).
    template <class InType, class OutType, class JacType, class AdjGradType,
              class AdjHessType, class AdjVarType>
    inline void simd_compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        using Scalar = typename InType::Scalar;

        const int ir = this->input_rows();
        const int or_ = this->output_rows();

        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> gx = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> hx = adjhess_.const_cast_derived();

        // Step 1: fx + jx via Phase 5b SIMD Jacobian.
        this->simd_compute_jacobian_impl(x, fx_, jx_);

        // Step 2: gx = J^T λ on SSType.
        for (int i = 0; i < ir; ++i) {
            Scalar acc;
            acc.setZero();
            for (int k = 0; k < or_; ++k) acc = acc + jx(k, i) * adjvars[k];
            gx[i] = acc;
        }

        // Step 3: hx via direct-SIMD FoR.
        compute_adjoint_hessian_for_simd_(x, hx, adjvars, ir, or_);
    }
#endif // TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse

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

    // Phase 6 SIMD FoR Hessian.  Direct-SIMD twin of compute_adjoint_hessian_for_:
    // every local is SSType (Eigen::Array<double, W, 1>); the outer
    // __enzyme_fwddiff over enzyme_for_outer_wrapper_simd propagates W
    // lane-local Hessian columns per call.  Phase 3 batching (enzyme_width=BW)
    // is layered on top so each call also propagates BW outer tangents.
    //
    // The inner __enzyme_autodiff (inside enzyme_for_outer_wrapper_simd) runs
    // on SuperScalar arithmetic — the unproven combo flagged in the Phase 6
    // plan.  fx_scratch / lam_scratch are passed enzyme_const at the outer
    // fwddiff so they are not BW-replicated; the inner reverse pass mutates
    // lam_scratch per its standard convention.
    template <class InType, class AdjHessType, class AdjVarType>
    inline void compute_adjoint_hessian_for_simd_(
        CVecRef<InType> x,
        MatRef<AdjHessType> hx,
        CVecRef<AdjVarType> adjvars,
        int ir, int or_) const {
        using SSType = typename InType::Scalar;
        constexpr int BW = TYCHO_ENZYME_BATCH_WIDTH;

        Eigen::Matrix<SSType, IR, 1> x_local = x;
        Eigen::Matrix<SSType, IR, 1> dx_local(ir);
        Eigen::Matrix<SSType, IR, 1> g_local(ir);
        Eigen::Matrix<SSType, IR, 1> dg_local(ir);
        Eigen::Matrix<SSType, OR, 1> fx_scratch(or_);
        Eigen::Matrix<SSType, OR, 1> lam_scratch(or_);

        const Derived* self = static_cast<const Derived*>(this);

        int i = 0;

        if constexpr (BW > 1) {
            // Batched outer fwddiff: BW Hessian columns per call.  Shadow
            // buffers are SSType-valued and column-major (Eigen default), so
            // BW columns at byte stride ir*sizeof(SSType) match the
            // SoA-by-lane layout enzyme_dupv expects.
            Eigen::Matrix<SSType, IR, BW> dx_shadow(ir, BW);
            Eigen::Matrix<SSType, IR, BW> dg_shadow(ir, BW);
            const long stride_x =
                static_cast<long>(ir) * static_cast<long>(sizeof(SSType));

            const int ir_chunked = (ir / BW) * BW;
            for (; i < ir_chunked; i += BW) {
                dx_shadow.setZero();
                for (int b = 0; b < BW; ++b)
                    dx_shadow(i + b, b).setConstant(1.0);
                dg_shadow.setZero();
                g_local.setZero();
                fx_scratch.setZero();
                for (int k = 0; k < or_; ++k) lam_scratch[k] = adjvars[k];

                __enzyme_fwddiff<void>(
                    reinterpret_cast<void*>(
                        &detail::enzyme_for_outer_wrapper_simd<Derived, SSType>),
                    enzyme_width, BW,
                    enzyme_const, self,
                    enzyme_dupv, stride_x,
                        x_local.data(), dx_shadow.data(),
                    enzyme_dupv, stride_x,
                        g_local.data(), dg_shadow.data(),
                    enzyme_const, fx_scratch.data(),
                    enzyme_const, lam_scratch.data(),
                    enzyme_const, ir,
                    enzyme_const, or_);

                for (int b = 0; b < BW; ++b)
                    for (int j = 0; j < ir; ++j)
                        hx(j, i + b) = dg_shadow(j, b);
            }
        }

        // Tail (or full loop when BW=1): single-tangent SIMD fwddiff.
        // Identical shape to the Phase 5b Jacobian tail at lines 316-335.
        for (; i < ir; ++i) {
            dx_local.setZero();
            dx_local[i].setConstant(1.0);
            g_local.setZero();
            dg_local.setZero();
            fx_scratch.setZero();
            for (int k = 0; k < or_; ++k) lam_scratch[k] = adjvars[k];

            __enzyme_fwddiff<void>(
                reinterpret_cast<void*>(
                    &detail::enzyme_for_outer_wrapper_simd<Derived, SSType>),
                enzyme_const, self,
                enzyme_dup,   x_local.data(),  dx_local.data(),
                enzyme_dup,   g_local.data(),  dg_local.data(),
                enzyme_const, fx_scratch.data(),
                enzyme_const, lam_scratch.data(),
                enzyme_const, ir,
                enzyme_const, or_);

            for (int j = 0; j < ir; ++j) hx(j, i) = dg_local[j];
        }

        // Symmetrize on SSType — each lane is independently symmetric.
        for (int i2 = 0; i2 < ir; ++i2) {
            for (int j = i2 + 1; j < ir; ++j) {
                SSType avg = 0.5 * (hx(i2, j) + hx(j, i2));
                hx(i2, j) = avg;
                hx(j, i2) = avg;
            }
        }
    }
#endif // TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse

#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)
    // Phase 7: combined J+H FoF helper.  The outer __enzyme_fwddiff
    // naturally produces J(x)·e_i in dfx_outer_shadow as a free byproduct
    // (it only depends on the outer dx_outer = e_i, not on dx_inner = e_j);
    // we read column i of J on the FIRST inner-j iteration and ignore the
    // recomputation on subsequent j.  Replaces compute_adjoint_hessian_fof_
    // and saves one compute_jacobian_impl call per Hessian invocation.
    template <class InType, class FxType, class JacType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjoint_hessian_fof_(
        CVecRef<InType> x,
        const Eigen::MatrixBase<FxType>& fx_,
        const Eigen::MatrixBase<JacType>& jx_,
        MatRef<AdjHessType> hx,
        CVecRef<AdjVarType> adjvars,
        int ir, int or_) const {

        FxType& fx = fx_.const_cast_derived();
        JacType& jx = jx_.const_cast_derived();

        Eigen::Matrix<double, IR, 1> x_local = x;
        Eigen::Matrix<double, IR, 1> dx_outer(ir);
        Eigen::Matrix<double, IR, 1> dx_inner(ir);
        Eigen::Matrix<double, OR, 1> fx_primal(or_);
        Eigen::Matrix<double, OR, 1> dfx_outer_shadow(or_);  // = J(x) · e_i (= column i of J)
        Eigen::Matrix<double, OR, 1> dfx_inner_primal(or_);  // = J · e_j
        Eigen::Matrix<double, OR, 1> ddfx_outer_shadow(or_); // = ∂(J·e_j)/∂x · e_i

        const Derived* self = static_cast<const Derived*>(this);

        // O(IR^2) outer fwddiff calls.  Each (i, j) yields ∂²f_k/∂x_i∂x_j
        // for all k AND, when j == 0, column i of J = J(x)·e_i.
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

                // J extraction: read column i of J on the first inner pass
                // (the value depends only on dx_outer = e_i, so subsequent
                // j iterations recompute the same column — ignore them).
                if (j == 0) {
                    for (int k = 0; k < or_; ++k) jx(k, i) = dfx_outer_shadow[k];
                    // fx is the same across all (i, j); copy it on the very
                    // first call so subsequent calls don't re-stamp it.
                    if (i == 0) {
                        for (int k = 0; k < or_; ++k) fx[k] = fx_primal[k];
                    }
                }

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

    // Phase 7 SIMD FoF combined J+H helper.  All-forward-mode (fwddiff over
    // fwddiff over SuperScalar arithmetic) — no SS reverse-mode tape, so
    // composite trig+sqrt+division bodies (MEE-class) work without opt-out.
    //
    // Per batched outer call (BW different e_i × one fixed e_j):
    //   - dfx_outer_shadow_b[b][k] = J(x)·e_{i+b} → pack into jx(k, i+b)
    //     (only on the j == 0 inner pass; J doesn't depend on dx_inner)
    //   - ddfx_outer_shadow_b[b][k] = ∂²f_k/∂x_{i+b}∂x_j → reduce with
    //     adjvars into hx(i+b, j)
    template <class InType, class FxType, class JacType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjoint_hessian_fof_simd_(
        CVecRef<InType> x,
        const Eigen::MatrixBase<FxType>& fx_,
        const Eigen::MatrixBase<JacType>& jx_,
        MatRef<AdjHessType> hx,
        CVecRef<AdjVarType> adjvars,
        int ir, int or_) const {
        using SSType = typename InType::Scalar;
        constexpr int BW = TYCHO_ENZYME_BATCH_WIDTH;

        FxType& fx = fx_.const_cast_derived();
        JacType& jx = jx_.const_cast_derived();

        Eigen::Matrix<SSType, IR, 1> x_local = x;
        Eigen::Matrix<SSType, IR, 1> dx_outer(ir);
        Eigen::Matrix<SSType, IR, 1> dx_inner(ir);
        Eigen::Matrix<SSType, OR, 1> fx_primal(or_);
        Eigen::Matrix<SSType, OR, 1> dfx_outer_shadow(or_);  // BW=1 tail: J·e_i
        Eigen::Matrix<SSType, OR, 1> dfx_inner_primal(or_);
        Eigen::Matrix<SSType, OR, 1> ddfx_outer_shadow(or_); // BW=1 tail: H column

        const Derived* self = static_cast<const Derived*>(this);

        int i = 0;

        if constexpr (BW > 1) {
            Eigen::Matrix<SSType, IR, BW> dx_outer_shadow(ir, BW);
            Eigen::Matrix<SSType, OR, BW> dfx_outer_shadow_b(or_, BW);
            Eigen::Matrix<SSType, OR, BW> ddfx_outer_shadow_b(or_, BW);
            const long stride_x =
                static_cast<long>(ir) * static_cast<long>(sizeof(SSType));
            const long stride_y =
                static_cast<long>(or_) * static_cast<long>(sizeof(SSType));

            const int ir_chunked = (ir / BW) * BW;
            for (; i < ir_chunked; i += BW) {
                // Prime BW outer tangents: e_{i+0} .. e_{i+BW-1}.
                dx_outer_shadow.setZero();
                for (int b = 0; b < BW; ++b)
                    dx_outer_shadow(i + b, b).setConstant(1.0);

                for (int j = 0; j < ir; ++j) {
                    dx_inner.setZero();
                    dx_inner[j].setConstant(1.0);
                    fx_primal.setZero();
                    dfx_outer_shadow_b.setZero();
                    dfx_inner_primal.setZero();
                    ddfx_outer_shadow_b.setZero();

                    __enzyme_fwddiff<void>(
                        reinterpret_cast<void*>(
                            &detail::enzyme_fof_inner_wrapper_simd<Derived, SSType>),
                        enzyme_width, BW,
                        enzyme_const, self,
                        enzyme_dupv, stride_x,
                            x_local.data(), dx_outer_shadow.data(),
                        enzyme_dupv, stride_y,
                            fx_primal.data(), dfx_outer_shadow_b.data(),
                        enzyme_const, dx_inner.data(),
                        enzyme_dupv, stride_y,
                            dfx_inner_primal.data(), ddfx_outer_shadow_b.data(),
                        enzyme_const, ir,
                        enzyme_const, or_);

                    if (j == 0) {
                        for (int b = 0; b < BW; ++b)
                            for (int k = 0; k < or_; ++k)
                                jx(k, i + b) = dfx_outer_shadow_b(k, b);
                        if (i == 0) {
                            for (int k = 0; k < or_; ++k) fx[k] = fx_primal[k];
                        }
                    }

                    for (int b = 0; b < BW; ++b) {
                        SSType acc;
                        acc.setZero();
                        for (int k = 0; k < or_; ++k)
                            acc = acc + adjvars[k] * ddfx_outer_shadow_b(k, b);
                        hx(i + b, j) = acc;
                    }
                }
            }
        }

        // Tail (or full when BW=1): single-tangent outer fwddiff.
        for (; i < ir; ++i) {
            dx_outer.setZero();
            dx_outer[i].setConstant(1.0);

            for (int j = 0; j < ir; ++j) {
                dx_inner.setZero();
                dx_inner[j].setConstant(1.0);
                fx_primal.setZero();
                dfx_outer_shadow.setZero();
                dfx_inner_primal.setZero();
                ddfx_outer_shadow.setZero();

                __enzyme_fwddiff<void>(
                    reinterpret_cast<void*>(
                        &detail::enzyme_fof_inner_wrapper_simd<Derived, SSType>),
                    enzyme_const, self,
                    enzyme_dup,   x_local.data(),         dx_outer.data(),
                    enzyme_dup,   fx_primal.data(),       dfx_outer_shadow.data(),
                    enzyme_const, dx_inner.data(),
                    enzyme_dup,   dfx_inner_primal.data(),
                                  ddfx_outer_shadow.data(),
                    enzyme_const, ir,
                    enzyme_const, or_);

                if (j == 0) {
                    for (int k = 0; k < or_; ++k) jx(k, i) = dfx_outer_shadow[k];
                    if (i == 0) {
                        for (int k = 0; k < or_; ++k) fx[k] = fx_primal[k];
                    }
                }

                SSType acc;
                acc.setZero();
                for (int k = 0; k < or_; ++k)
                    acc = acc + adjvars[k] * ddfx_outer_shadow[k];
                hx(i, j) = acc;
            }
        }

        // Symmetrize on SSType — each lane independently symmetric.
        for (int i2 = 0; i2 < ir; ++i2) {
            for (int j = i2 + 1; j < ir; ++j) {
                SSType avg = 0.5 * (hx(i2, j) + hx(j, i2));
                hx(i2, j) = avg;
                hx(j, i2) = avg;
            }
        }
    }

    // Phase 7 SIMD FoF public method.  Twin of the FoR variant under
    // strategy=ForwardOverReverse.  Both produce SuperScalar fx + jx + g + h
    // for Vectorizable+EnzymeAD VFs; this variant uses combined J+H FoF
    // (one helper) where the FoR variant stitches separate Phase 5b + FoR.
    template <class InType, class OutType, class JacType, class AdjGradType,
              class AdjHessType, class AdjVarType>
    inline void simd_compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        using Scalar = typename InType::Scalar;

        const int ir = this->input_rows();
        const int or_ = this->output_rows();

        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> gx = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> hx = adjhess_.const_cast_derived();

        // Combined J+H: SIMD helper writes fx, jx, hx in one pass.
        compute_jacobian_adjoint_hessian_fof_simd_(
            x, fx_, jx_, hx, adjvars, ir, or_);

        // gx = J^T λ on SSType.
        for (int i = 0; i < ir; ++i) {
            Scalar acc;
            acc.setZero();
            for (int k = 0; k < or_; ++k) acc = acc + jx(k, i) * adjvars[k];
            gx[i] = acc;
        }
    }
#endif // TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward
};

#endif // TYCHO_HAS_ENZYME_AD

} // namespace tycho::vf
