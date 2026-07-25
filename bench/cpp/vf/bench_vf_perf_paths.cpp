///////////////////////////////////////////////////////////////////////////////
// VF perf-paths benchmarks — PR 8 coverage foundation
//
// Targets four specific VF-subsystem code paths flagged in the review
// remediation series' fact dossier (docs/superpowers/specs/
// 2026-07-02-review-remediation-design.md PR 8 section) as having NO
// pre-existing benchmark coverage anywhere in bench/cpp/:
//   1. ComparativeFunction (min/max) double-eval / O(2^N) variadic blowup
//      (VF_REVIEW §2.1, type_erasure/comparative.h + conditional.h).
//   2. Per-call heap allocations in dynamic-size (OutputIsDynamic) hot paths
//      that never fire for any compile-time-sized existing bench
//      (VF_REVIEW §2.2 bullets 1-3: cwise_operators.h, cwise_sum.h,
//      parsed_input.h).
//   3. Redundant FD primal evaluations under FDiffFwd mode
//      (VF_REVIEW §2.3, derivatives/dense_fdiff_fwd.h).
//   4. Scaling-wrapper `cast<Scalar>()` copies and IOScaled's unfused
//      per-element loops (VF_REVIEW §2.6, scaling/scaled.h + io_scaled.h).
//
// Every construction below was verified against the actual header source at
// HEAD (line numbers cited in each benchmark's comment), not inferred from
// the review docs alone.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/vf/scaling/io_scaled.h"
#include <benchmark/benchmark.h>
#include <tycho/vector_functions.h>

#include <cmath>

using namespace tycho::vf;

///////////////////////////////////////////////////////////////////////////////
// 1. ComparativeFunction (min/max) double-eval — VF §2.1 (dossier item 1, ★)
//
// comparative.h:73-93 (2-operand specialization) and :40-65 (3+ operand
// variadic form) each construct the tail operand(s) TWICE — once inside the
// `BaseCond` (ConditionalStatement) predicate and once as the `IfElseFunction`
// false branch — so `IfElseFunction::compute*` (conditional.h) evaluates the
// predicate (both operands' values) and then re-evaluates the selected
// branch: 3 value evals for N=2, ~2^N for general N. "Static" benches below
// use concrete (non-type-erased) VF operands; "Generic" benches wrap each
// operand in GenericFunction<-1,1> — the exact shape the Python binding path
// uses (src/bindings/vf/generic_function_bind.h:229-230,
// `ComparativeFunction<GenS, GenS>(ComparativeFlags::MinFlag, f1, f2)`),
// where the dossier notes each redundant eval is a virtual dispatch, not just
// a template call.
///////////////////////////////////////////////////////////////////////////////

static void BM_VF_MinMax_Static(benchmark::State &state) {
    // 2-operand concrete min: comparative.h:88-93 double-store (`first`
    // stored once in BaseCond, once as the IfElse true branch; `second`
    // likewise inside BaseCond and as the false branch).
    auto args = Arguments<4>();
    auto f = args.coeff<0>();
    auto g = args.coeff<1>();
    using F = decltype(f);
    using G = decltype(g);
    ComparativeFunction<F, G> minfg(ComparativeFlags::MinFlag, f, g);

    Eigen::VectorXd x(4);
    x << 1.5, -2.5, 3.0, 4.0;
    Eigen::VectorXd fx(1);
    for (auto _ : state) {
        fx.setZero();
        minfg.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_MinMax_Static);

static void BM_VF_MinMax_Static_N4(benchmark::State &state) {
    // 4-operand variadic min, concrete operands: comparative.h:58-63 —
    // each recursion level constructs `Second(type, rest...)` (the tail
    // ComparativeFunction over the remaining operands) TWICE, so N=4 nests
    // one extra recursion level beyond the 2-operand case above, exposing
    // the O(2^N) growth the dossier flags ("~2^N for N operands").
    auto args = Arguments<4>();
    auto f = args.coeff<0>();
    auto g = args.coeff<1>();
    auto h = args.coeff<2>();
    auto k = args.coeff<3>();
    using F = decltype(f);
    using G = decltype(g);
    using H = decltype(h);
    using K = decltype(k);
    ComparativeFunction<F, G, H, K> minall(ComparativeFlags::MinFlag, f, g, h, k);

    Eigen::VectorXd x(4);
    x << 1.5, -2.5, 3.0, 4.0;
    Eigen::VectorXd fx(1);
    for (auto _ : state) {
        fx.setZero();
        minall.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_MinMax_Static_N4);

static void BM_VF_MinMax_Generic(benchmark::State &state) {
    // 2-operand min over type-erased GenericFunction<-1,1> operands — the
    // PSIOPT/Python-binding-relevant shape (generic_function_bind.h:229-230).
    // Each of the 3 value evals per call chases a GFConcept vtable dispatch
    // rather than a plain inlined template call.
    using GenS = GenericFunction<-1, 1>;
    auto args = Arguments<4>();
    GenS f(args.coeff<0>());
    GenS g(args.coeff<1>());
    ComparativeFunction<GenS, GenS> minfg(ComparativeFlags::MinFlag, f, g);

    Eigen::VectorXd x(4);
    x << 1.5, -2.5, 3.0, 4.0;
    Eigen::VectorXd fx(1);
    for (auto _ : state) {
        fx.setZero();
        minfg.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_MinMax_Generic);

static void BM_VF_MinMax_Generic_N4(benchmark::State &state) {
    // 4-operand variadic min over type-erased operands: combines the O(2^N)
    // storage/eval blowup (comparative.h:58-63) with the virtual-dispatch
    // cost of GenericFunction — the worst case for item 1's value-only path.
    using GenS = GenericFunction<-1, 1>;
    auto args = Arguments<4>();
    GenS f(args.coeff<0>());
    GenS g(args.coeff<1>());
    GenS h(args.coeff<2>());
    GenS k(args.coeff<3>());
    ComparativeFunction<GenS, GenS, GenS, GenS> minall(ComparativeFlags::MinFlag, f, g, h, k);

    Eigen::VectorXd x(4);
    x << 1.5, -2.5, 3.0, 4.0;
    Eigen::VectorXd fx(1);
    for (auto _ : state) {
        fx.setZero();
        minall.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_MinMax_Generic_N4);

static void BM_VF_MinMax_Generic_JGH(benchmark::State &state) {
    // Second-order (KKT-path) variant of the worst case above: the same
    // 4-operand variadic min over type-erased operands, but timing the full
    // `compute_jacobian_adjointgradient_adjointhessian` entry point
    // (conditional.h's IfElseFunction JGH hook, which first evaluates BOTH
    // operands' values via the predicate's ConditionalStatement, then
    // re-evaluates the selected branch's full Jacobian/gradient/Hessian).
    // Added beyond the brief's two literal names because the dossier
    // repeatedly flags this exact combination — type-erased operands, full
    // JGH — as "the waste is worst in the KKT path": PSIOPT's interior-point
    // loop calls exactly this entry point every iteration whenever min/max
    // appears in a constraint or objective. Cheap to add (same operands as
    // BM_VF_MinMax_Generic_N4, one extra timed call), so no reason to defer.
    using GenS = GenericFunction<-1, 1>;
    auto args = Arguments<4>();
    GenS f(args.coeff<0>());
    GenS g(args.coeff<1>());
    GenS h(args.coeff<2>());
    GenS k(args.coeff<3>());
    ComparativeFunction<GenS, GenS, GenS, GenS> minall(ComparativeFlags::MinFlag, f, g, h, k);

    Eigen::VectorXd x(4);
    x << 1.5, -2.5, 3.0, 4.0;
    Eigen::VectorXd lm(1);
    lm << 1.0;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, 4);
    Eigen::VectorXd adjgrad(4);
    Eigen::MatrixXd adjhess(4, 4);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();
        adjhess.setZero();
        minall.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, lm);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::DoNotOptimize(adjgrad);
        benchmark::DoNotOptimize(adjhess);
    }
}
BENCHMARK(BM_VF_MinMax_Generic_JGH);

///////////////////////////////////////////////////////////////////////////////
// 2. Dynamic-size (OutputIsDynamic) cwise / CwiseSum / ParsedInput heap
//    locals — VF §2.2 bullets 1-3 (dossier item 2)
//
// Builds a single dynamic-size (IR=-1) composite expression that fires all
// three flagged per-call heap-allocation paths at once, verified by reading
// each header directly (not inferred):
//   - cwise_operators.h:1408-1419 / 1429-1451 / 1469-1510
//     (`CwiseFunctionOperator`'s `fxt`/`jxdiag`/`hxdiag`/`adjtemp` plain-
//     local-plus-resize temps, guarded by `if constexpr (Func::OutputIsDynamic)`)
//     — fired by `.sin()`/`.cos()` on a *dynamic* (ORC=-1) Segment.
//     `OutputIsDynamic` is literally `(OR < 0)` (computable_base.h:179-180),
//     and the runtime (non-template) `.segment(start, size)` overload
//     (dense_function_base.h:172) yields `Segment<-1,-1,-1>` (ORC=-1) — see
//     test_mixed_size_ops.cpp:80's own comment for the identical pattern.
//   - cwise_sum.h:111-117 (`CwiseSum_Impl::compute_impl`'s non-segment
//     branch, `typename Func::template Output<Scalar> fxv;` + `.resize()`)
//     — fired here because `combo` (the elementwise sum of two dynamic-ORC
//     cwise results) is not a `Segment`, so `IsSegmentOp` is false and the
//     plain-local branch — not the `.segment<...>().sum()` fast path — runs.
//   - parsed_input.h:102-106 (`compute_impl`'s non-contiguous branch,
//     `xin`) / :127-138 (`compute_jacobian_impl`'s `xin`/`jxin`) / :179-187
//     (JGH hook's `xin`/`jxin`/`gxin`/`hxin`) — fired by constructing
//     `ParsedInput` with a deliberately non-contiguous, non-monotonic index
//     map (`vlocs = {6, 0, 3}`); the ctor's own contiguity check
//     (`if (delta != 1) contiguous_ = false;`) clears the fast-path flag on
//     the very first pair, so every call runs the gather/scatter loop.
//
// None of these branches has ever fired in bench/cpp/vector_functions/
// bench_vector_functions.cpp — every existing BM_VF_* there builds ODEs via
// ODESize<...> (fixed IR/OR), so `Func::OutputIsDynamic` is always false.
///////////////////////////////////////////////////////////////////////////////

namespace {

constexpr int kDynODERows = 8; // outer state: [r0, r1, r2, v0, v1, v2, m, t]

auto build_dynamic_ode_expr() {
    auto args = Arguments<-1>(kDynODERows);

    auto r = args.segment(0, 3); // Segment<-1,-1,-1>, ORC=-1 -> OutputIsDynamic
    auto v = args.segment(3, 3); // Segment<-1,-1,-1>, ORC=-1 -> OutputIsDynamic

    auto sin_r = r.sin();       // CwiseSin<Segment<-1,-1,-1>>  -- cwise_operators.h locals
    auto cos_v = v.cos();       // CwiseCos<Segment<-1,-1,-1>>  -- cwise_operators.h locals
    auto combo = sin_r + cos_v; // dynamic elementwise sum, ORC=-1, not a Segment
    auto energy = combo.sum();  // CwiseSum non-segment branch -- cwise_sum.h `fxv` local

    // Non-contiguous ParsedInput: inner function reads outer indices
    // {6, 0, 3} (m, r0, v0) in gather order -- contiguous_ = false from the
    // very first pair (delta = 0 - 6 = -6 != 1).
    auto a3 = Arguments<3>();
    auto inner_expr = a3.coeff<0>() * a3.coeff<1>() + a3.coeff<2>().sin();
    GenericFunction<-1, -1> inner(inner_expr);

    Eigen::VectorXi vlocs(3);
    vlocs << 6, 0, 3;
    ParsedInput<GenericFunction<-1, -1>, -1, -1> parsed(inner, vlocs, kDynODERows);

    // Mixed static(ORC=1)/dynamic(ORC=-1) operator+ -- the relaxed-operator
    // path exercised by test_mixed_size_ops.cpp's "operator+ with mixed ORC"
    // suite (operator_overloads.h:448-455's static_asserts allow OR1==OR2
    // OR either side < 0).
    return energy + parsed;
}

} // namespace

static void BM_VF_DynamicODE_Compute(benchmark::State &state) {
    auto expr = build_dynamic_ode_expr();
    Eigen::VectorXd x(kDynODERows);
    x << 0.3, 0.6, 0.9, 1.2, 0.4, -0.7, 2.0, 5.0;
    Eigen::VectorXd fx(1);
    for (auto _ : state) {
        fx.setZero();
        expr.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_DynamicODE_Compute);

static void BM_VF_DynamicODE_Jacobian(benchmark::State &state) {
    auto expr = build_dynamic_ode_expr();
    Eigen::VectorXd x(kDynODERows);
    x << 0.3, 0.6, 0.9, 1.2, 0.4, -0.7, 2.0, 5.0;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, kDynODERows);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        expr.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
    }
}
BENCHMARK(BM_VF_DynamicODE_Jacobian);

static void BM_VF_DynamicODE_JGH(benchmark::State &state) {
    // Second-order (KKT-path) variant of the two benches above: same dynamic
    // composite expression, but timing the full
    // `compute_jacobian_adjointgradient_adjointhessian` entry point (mirrors
    // the call shape of `BM_VF_MinMax_Generic_JGH` above). This is the only
    // call shape that reaches the JGH-only heap locals flagged by
    // VF_REVIEW/dossier item 2 but left uncovered by `_Compute`/`_Jacobian`
    // above -- verified by reading both headers at HEAD:
    //   - cwise_operators.h:1467-1510
    //     (`CwiseFunctionOperator::compute_jacobian_adjointgradient_adjointhessian_impl`):
    //     `hxdiag` is heap-resized under the same `if constexpr
    //     (Func::OutputIsDynamic)` guard as `fxt`/`jxdiag` (lines 1478-1486),
    //     but `hxdiag` itself is a local unique to this JGH hook --
    //     `compute_impl`/`compute_jacobian_impl` never declare it.
    //     `adjtemp` (line 1492, `jxdiag.cwiseProduct(adjvars)`) is likewise
    //     only ever constructed inside this JGH function body, as an
    //     unguarded plain `Output<Scalar>` local -- it heap-allocates
    //     whenever `Output<Scalar>` is dynamic-sized, i.e. under the
    //     identical `Func::OutputIsDynamic` condition that gates the other
    //     locals in this same function. `compute`/`compute_jacobian` (timed
    //     by `_Compute`/`_Jacobian` above) never call this function, so
    //     `hxdiag`/`adjtemp` never fire for them.
    //   - parsed_input.h:157-206
    //     (`ParsedInput::compute_jacobian_adjointgradient_adjointhessian_impl`,
    //     non-contiguous branch lines 178-205): `gxin` (line 182) and `hxin`
    //     (line 183) are locals unique to this JGH hook --
    //     `compute_impl`/`compute_jacobian_impl` have no gradient/Hessian
    //     locals at all, only `xin`/`jxin`. The bench's non-contiguous
    //     `vlocs = {6, 0, 3}` keeps `contiguous_ = false` for the whole call
    //     (same ctor reasoning as `_Compute`/`_Jacobian`: `delta = 0 - 6 =
    //     -6 != 1` on the first pair), so this branch -- and both locals --
    //     fire on every call.
    auto expr = build_dynamic_ode_expr();
    Eigen::VectorXd x(kDynODERows);
    x << 0.3, 0.6, 0.9, 1.2, 0.4, -0.7, 2.0, 5.0;
    Eigen::VectorXd lm(1);
    lm << 1.0;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, kDynODERows);
    Eigen::VectorXd adjgrad(kDynODERows);
    Eigen::MatrixXd adjhess(kDynODERows, kDynODERows);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();
        adjhess.setZero();
        expr.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, lm);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::DoNotOptimize(adjgrad);
        benchmark::DoNotOptimize(adjhess);
    }
}
BENCHMARK(BM_VF_DynamicODE_JGH);

///////////////////////////////////////////////////////////////////////////////
// 3. Redundant FD primal evaluations under FDiffFwd mode — VF §2.3
//    (dossier item 4)
//
// SmallNonlinearFD opts into FDiffFwd for both the Jacobian and Hessian
// slots, mirroring tests/cpp/vector_functions/test_vf_fd_modes.cpp's
// SquareFDStatic/CubeFDStatic pattern (VectorFunction<Derived, IR, OR,
// DenseDerivativeMode::FDiffFwd, DenseDerivativeMode::FDiffFwd>), but with
// 3 inputs / 2 outputs and a genuinely nonlinear (sin/cos-coupled) body so
// the IR+1 extra primal evals are not a trivial no-op.
///////////////////////////////////////////////////////////////////////////////

namespace {

struct SmallNonlinearFD : VectorFunction<SmallNonlinearFD, 3, 2, DenseDerivativeMode::FDiffFwd,
                                         DenseDerivativeMode::FDiffFwd> {
    using Base = VectorFunction<SmallNonlinearFD, 3, 2, DenseDerivativeMode::FDiffFwd,
                                DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx[0] = std::sin(x[0]) * x[1] + x[2] * x[2];
        fx[1] = std::cos(x[1]) * x[0] - x[2];
    }
};

} // namespace

static void BM_VF_FDiffFwd_Jacobian(benchmark::State &state) {
    // dense_fdiff_fwd.h:71-103 (`DenseFirstDerivatives<...,FDiffFwd>::
    // compute_jacobian_impl`): one baseline `derived().compute()` (:96) plus
    // IR forward-difference `derived().compute()` calls in the loop
    // (:91-102) -- IR+1 = 4 primal evals per Jacobian call for this 3-input
    // function.
    SmallNonlinearFD f;
    Eigen::VectorXd x(3);
    x << 0.7, -1.3, 2.1;
    Eigen::VectorXd fx(2);
    Eigen::MatrixXd jx(2, 3);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        f.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
    }
}
BENCHMARK(BM_VF_FDiffFwd_Jacobian);

static void BM_VF_FDiffFwd_AdjointHessian(benchmark::State &state) {
    // VF_REVIEW §2.3 / dossier item 4: the *combined* entry point
    // (`compute_jacobian_adjointgradient_adjointhessian_impl`,
    // dense_fdiff_fwd.h:224-232) calls `compute_jacobian_adjointgradient`
    // (which already computes `adjgrad_`, line 230) and THEN calls
    // `adjointhessian` (:180-205), which independently recomputes the SAME
    // base-point gradient at line 190
    // (`this->adjointgradient(x, ag, adjvars)`) before forward-differencing
    // it IR times (:192-203) -- IR+1 = 4 wasted extra primal evals per
    // Hessian call that passing the already-computed gradient through would
    // eliminate. Timing the *combined* entry point (not the standalone
    // public `adjointhessian()`) is required to observe this cost: calling
    // `adjointhessian()` alone only pays for the one gradient computation it
    // actually needs.
    SmallNonlinearFD f;
    Eigen::VectorXd x(3);
    x << 0.7, -1.3, 2.1;
    Eigen::VectorXd lm(2);
    lm << 1.3, -0.4;
    Eigen::VectorXd fx(2);
    Eigen::MatrixXd jx(2, 3);
    Eigen::VectorXd adjgrad(3);
    Eigen::MatrixXd adjhess(3, 3);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();
        adjhess.setZero();
        f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, lm);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::DoNotOptimize(adjgrad);
        benchmark::DoNotOptimize(adjhess);
    }
}
BENCHMARK(BM_VF_FDiffFwd_AdjointHessian);

///////////////////////////////////////////////////////////////////////////////
// 4. Scaling-wrapper copies — VF §2.6 (dossier item 6)
///////////////////////////////////////////////////////////////////////////////

namespace {

constexpr int kScaledN = 10;

// MatrixScaled_Impl::compute_impl (scaled.h:846-871, both the `NoTemp` and
// non-`NoTemp` branches) does `mattmp = this->mat.template cast<Scalar>();`
// every call -- a full matrix copy -- even when Scalar == double; confirmed
// present at HEAD with no `if constexpr (std::is_same_v<Scalar, double>)`
// guard anywhere in the file. Uses a 10x10 constant matrix (square, so
// `NoTemp` is true) wrapping a dynamic-output cwise operand, per the
// dossier's own sketch ("visible at matrix sizes >= 10x10").
auto build_matrix_scaled_expr() {
    auto args = Arguments<-1>(kScaledN);
    auto expr = args.sin(); // CwiseSin<Arguments<-1>>, ORC=-1, output_rows()=10

    Eigen::MatrixXd A(kScaledN, kScaledN);
    for (int i = 0; i < kScaledN; i++) {
        for (int j = 0; j < kScaledN; j++) {
            A(i, j) = 0.1 * (i + 1) + 0.05 * (j + 1);
        }
    }

    return MatrixScaled<decltype(expr), -1>(expr, A);
}

} // namespace

static void BM_VF_Scaled_Compute(benchmark::State &state) {
    auto ms = build_matrix_scaled_expr();
    Eigen::VectorXd x(kScaledN);
    for (int i = 0; i < kScaledN; i++) {
        x[i] = 0.1 * (i + 1);
    }
    Eigen::VectorXd fx(kScaledN);
    for (auto _ : state) {
        fx.setZero();
        ms.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_Scaled_Compute);

namespace {

// IOScaled (io_scaled.h) has no product/accumulate-forwarding overrides
// (falls back to the dense base paths) and per-element scaling loops
// (io_scaled.h:100-102 input-scale loop inside `compute_impl`'s Impl lambda,
// :135-137/141-147 in `compute_jacobian_impl`) that could be fused into
// `cwiseProduct()`s -- confirmed present, no forwarding overrides anywhere
// in the 215-line file. Wraps a BrachODE-shaped (IR=5, OR=3) nonlinear VF —
// the same dynamics as bench/cpp/vector_functions/bench_vector_functions.cpp's
// BM_VF_ComputeJacobian — so the scaling-wrapper overhead is directly
// comparable against that existing unscaled baseline.
auto build_io_scaled_expr() {
    auto args = Arguments<5>();
    auto v = args.coeff<2>();
    auto theta = args.coeff<4>();
    auto xdot = sin(theta) * v;
    auto ydot = cos(theta) * v * (-1.0);
    auto vdot = 9.81 * cos(theta);
    auto stacked = StackedOutputs{xdot, ydot, vdot}; // IR=5, OR=3 (static)

    Eigen::VectorXd in_scales(5);
    in_scales << 1.0, 1.0, 2.0, 0.5, 3.0;
    Eigen::VectorXd out_scales(3);
    out_scales << 10.0, 10.0, 5.0;

    return IOScaled<decltype(stacked)>(stacked, in_scales, out_scales);
}

} // namespace

static void BM_VF_IOScaled_Jacobian(benchmark::State &state) {
    auto io = build_io_scaled_expr();
    Eigen::VectorXd x(5);
    x << 1.0, 2.0, 3.0, 0.5, 1.0;
    Eigen::VectorXd fx(3);
    Eigen::MatrixXd jx(3, 5);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        io.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
    }
}
BENCHMARK(BM_VF_IOScaled_Jacobian);
