#pragma once

#include <cassert>
#include <memory>

#include "tycho/detail/optimal_control/interp/interp_table_1d.h"
#include "tycho/detail/optimal_control/interp/interp_table_2d.h"
#include "tycho/detail/optimal_control/interp/interp_table_3d.h"
#include "tycho/detail/optimal_control/interp/interp_table_4d.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include "tycho/detail/vf/expressions/stacked.h"

namespace tycho::vf {

// ── 1D: single VF input (general, multi-output) ───────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable1D> &table,
            const DenseFunctionBase<Func, IR, OR> &input) {
    assert(table && "interp: table pointer must not be null");
    if (table->vlen_ == 1) {
        return GenericFunction<-1, -1>(
            oc::InterpFunction1D<1>(table).eval(input.derived()));
    }
    return GenericFunction<-1, -1>(
        oc::InterpFunction1D<-1>(table).eval(input.derived()));
}

// ── 1D scalar: single VF input, scalar-valued table ───────────────
// Use this overload when the table returns a single value and the
// result must participate in VF operator/ or operator*(scalar, VF).
template <class Func, int IR, int OR>
auto interp_scalar(const std::shared_ptr<oc::InterpTable1D> &table,
                   const DenseFunctionBase<Func, IR, OR> &input) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, 1>(
        oc::InterpFunction1D<1>(table).eval(input.derived()));
}

// ── 2D: two scalar VF inputs ───────────────────────────────────────
template <class F1, int IR1, int OR1, class F2, int IR2, int OR2>
auto interp(const std::shared_ptr<oc::InterpTable2D> &table,
            const DenseFunctionBase<F1, IR1, OR1> &x,
            const DenseFunctionBase<F2, IR2, OR2> &y) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, 1>(
        oc::InterpFunction2D(table).eval(stack(x.derived(), y.derived())));
}

// ── 2D: single 2-element VF input ──────────────────────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable2D> &table,
            const DenseFunctionBase<Func, IR, OR> &xy) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, 1>(
        oc::InterpFunction2D(table).eval(xy.derived()));
}

// ── 3D: three scalar VF inputs ─────────────────────────────────────
template <class F1, int IR1, int OR1,
          class F2, int IR2, int OR2,
          class F3, int IR3, int OR3>
auto interp(const std::shared_ptr<oc::InterpTable3D> &table,
            const DenseFunctionBase<F1, IR1, OR1> &x,
            const DenseFunctionBase<F2, IR2, OR2> &y,
            const DenseFunctionBase<F3, IR3, OR3> &z) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, 1>(
        oc::InterpFunction3D(table).eval(
            stack(x.derived(), y.derived(), z.derived())));
}

// ── 3D: single 3-element VF input ──────────────────────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable3D> &table,
            const DenseFunctionBase<Func, IR, OR> &xyz) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, 1>(
        oc::InterpFunction3D(table).eval(xyz.derived()));
}

// ── 4D: four scalar VF inputs ──────────────────────────────────────
template <class F1, int IR1, int OR1,
          class F2, int IR2, int OR2,
          class F3, int IR3, int OR3,
          class F4, int IR4, int OR4>
auto interp(const std::shared_ptr<oc::InterpTable4D> &table,
            const DenseFunctionBase<F1, IR1, OR1> &x,
            const DenseFunctionBase<F2, IR2, OR2> &y,
            const DenseFunctionBase<F3, IR3, OR3> &z,
            const DenseFunctionBase<F4, IR4, OR4> &w) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, 1>(
        oc::InterpFunction4D(table).eval(
            stack(x.derived(), y.derived(), z.derived(), w.derived())));
}

// ── 4D: single 4-element VF input ──────────────────────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable4D> &table,
            const DenseFunctionBase<Func, IR, OR> &xyzw) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, 1>(
        oc::InterpFunction4D(table).eval(xyzw.derived()));
}

// ── LGL trajectory interpolation ───────────────────────────────────

/// Return a GenericFunction wrapping an LGLInterpTable (not yet composed
/// with a time expression). Call .eval(t_expr) on the result to compose.
inline auto lgl_interp(const std::shared_ptr<oc::LGLInterpTable> &table,
                       const Eigen::VectorXi &vars) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, -1>(oc::InterpFunction<-1>(table, vars));
}

/// Compose an LGLInterpTable with a VF time expression directly.
template <class Func, int IR, int OR>
auto lgl_interp(const std::shared_ptr<oc::LGLInterpTable> &table,
                const Eigen::VectorXi &vars,
                const DenseFunctionBase<Func, IR, OR> &t_expr) {
    assert(table && "interp: table pointer must not be null");
    return GenericFunction<-1, -1>(
        oc::InterpFunction<-1>(table, vars).eval(t_expr.derived()));
}

} // namespace tycho::vf
