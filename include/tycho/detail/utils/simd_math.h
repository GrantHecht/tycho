// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
//
// SIMD-friendly math wrappers for Enzyme-differentiated VectorFunctions.
//
// Eigen 5 ships vectorised pcos/psin/etc for Packet4d (and friends), but
// the lowered IR uses x86 round intrinsics + packed bitwise sign-bit
// tricks that Enzyme cannot currently differentiate (upstream issues
// [#689](https://github.com/EnzymeAD/Enzyme/issues/689),
// [#2679](https://github.com/EnzymeAD/Enzyme/issues/2679),
// [#2794](https://github.com/EnzymeAD/Enzyme/issues/2794)).
//
// Solution: implement `tycho::math::cos / sin` for the SuperScalar
// `Eigen::Array<double, W, 1>` type by manually unrolling W *scalar*
// `std::cos(double)` / `std::sin(double)` calls instead of one vectorised
// `pcos<Packet4d>` / `psin<Packet4d>`.  Each scalar libm call lowers to
// `@llvm.cos.f64` / `@llvm.sin.f64`, which Enzyme's built-in handler
// differentiates cleanly — including under `enzyme_width > 1` batching.
//
// Costs:
//   * Per primal call: W scalar libm trig invocations instead of one SIMD
//     pcos.  On modern Linux glibc, scalar libm cos+sin are tightly
//     pipelined; in practice this is comparable to or faster than the
//     SIMD polynomial when amortised across BW tangents.
// Wins:
//   * Phase 3 batched tangents (`enzyme_width = BW`) and Phase 5b SIMD
//     primal (Scalar = `Eigen::Array<double, W, 1>`) compose: one fwddiff
//     call computes one primal block + BW × W independent gradients.
//   * No custom Enzyme derivative — no `fixderivative_*` IR-verification
//     bug to dance around, no per-VF batching opt-out flag needed.
//
// USAGE (only needed for code paths that Enzyme will differentiate):
//   tycho::math::cos(x)   // x: double or Eigen::Array<double, W, 1>
//   tycho::math::sin(x)
//
// Code paths that are NOT differentiated by Enzyme should keep using
// Eigen directly — `mee_dynamics.h`'s analytic-integration path benefits
// from Eigen 5 SIMD trig without going through this layer.
// =============================================================================
#pragma once

#include <Eigen/Core>
#include <cmath>

#include "tycho/detail/utils/compiler_macros.h"

namespace tycho::math {

// Scalar overloads — forward to libm; Enzyme handles natively.
inline double cos(double x) { return std::cos(x); }
inline double sin(double x) { return std::sin(x); }

// SuperScalar overloads.  Scalar libm calls so Enzyme sees per-lane
// @llvm.cos.f64 / @llvm.sin.f64 — bithack-free, composes with
// enzyme_width > 1.
//
// `TYCHO_NOINLINE` keeps the function-call boundary stable so Enzyme's
// activity analysis doesn't have to walk through Eigen's expression
// templates.  Each call is small (W libm invocations) so the overhead is
// trivial.
inline TYCHO_NOINLINE Eigen::Array<double, 2, 1>
cos(const Eigen::Array<double, 2, 1>& x) {
    Eigen::Array<double, 2, 1> y;
    y[0] = std::cos(x[0]);
    y[1] = std::cos(x[1]);
    return y;
}

inline TYCHO_NOINLINE Eigen::Array<double, 2, 1>
sin(const Eigen::Array<double, 2, 1>& x) {
    Eigen::Array<double, 2, 1> y;
    y[0] = std::sin(x[0]);
    y[1] = std::sin(x[1]);
    return y;
}

inline TYCHO_NOINLINE Eigen::Array<double, 4, 1>
cos(const Eigen::Array<double, 4, 1>& x) {
    Eigen::Array<double, 4, 1> y;
    y[0] = std::cos(x[0]);
    y[1] = std::cos(x[1]);
    y[2] = std::cos(x[2]);
    y[3] = std::cos(x[3]);
    return y;
}

inline TYCHO_NOINLINE Eigen::Array<double, 4, 1>
sin(const Eigen::Array<double, 4, 1>& x) {
    Eigen::Array<double, 4, 1> y;
    y[0] = std::sin(x[0]);
    y[1] = std::sin(x[1]);
    y[2] = std::sin(x[2]);
    y[3] = std::sin(x[3]);
    return y;
}

inline TYCHO_NOINLINE Eigen::Array<double, 8, 1>
cos(const Eigen::Array<double, 8, 1>& x) {
    Eigen::Array<double, 8, 1> y;
    y[0] = std::cos(x[0]);
    y[1] = std::cos(x[1]);
    y[2] = std::cos(x[2]);
    y[3] = std::cos(x[3]);
    y[4] = std::cos(x[4]);
    y[5] = std::cos(x[5]);
    y[6] = std::cos(x[6]);
    y[7] = std::cos(x[7]);
    return y;
}

inline TYCHO_NOINLINE Eigen::Array<double, 8, 1>
sin(const Eigen::Array<double, 8, 1>& x) {
    Eigen::Array<double, 8, 1> y;
    y[0] = std::sin(x[0]);
    y[1] = std::sin(x[1]);
    y[2] = std::sin(x[2]);
    y[3] = std::sin(x[3]);
    y[4] = std::sin(x[4]);
    y[5] = std::sin(x[5]);
    y[6] = std::sin(x[6]);
    y[7] = std::sin(x[7]);
    return y;
}

} // namespace tycho::math
