// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/vf/common/common_functions.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"

namespace tycho::vf {

/////////////////////// Scalar Multiplication and
/// Division//////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

template <class Derived, int IR, int OR>
decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR> &func, double s) {
    return Scaled<Derived>(func.derived(), s);
}

template <int IR, int OR> auto operator*(const GenericFunction<IR, OR> &func, double s) {
    return Scaled<GenericFunction<IR, OR>>(func, s);
}

template <class Derived, int IR, int OR, class VALUE>
decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR> &func, StaticScaleBase<VALUE> s) {
    return StaticScaled<Derived, VALUE>(func.derived());
}

template <class Derived, int IR, int OR>
decltype(auto) operator*(double s, const DenseFunctionBase<Derived, IR, OR> &func) {
    return Scaled<Derived>(func.derived(), s);
}

template <int IR, int OR> auto operator*(double s, const GenericFunction<IR, OR> &func) {
    return Scaled<GenericFunction<IR, OR>>(func, s);
}

template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return RowScaled<Derived>(func.derived(), s);
}
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator*(const Eigen::MatrixBase<OutType> &s,
                         const DenseFunctionBase<Derived, IR, OR> &func) {
    return RowScaled<Derived>(func.derived(), s);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
template <class Derived> decltype(auto) operator*(const Scaled<Derived> &func, double s) {
    return Scaled<Derived>(func.func_, func.scale_value_ * s);
}
template <class Derived> decltype(auto) operator*(double s, const Scaled<Derived> &func) {
    return Scaled<Derived>(func.func_, func.scale_value_ * s);
}
template <class Derived> decltype(auto) operator*(const RowScaled<Derived> &func, double s) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_ * s);
}
template <class Derived> decltype(auto) operator*(double s, const RowScaled<Derived> &func) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_ * s);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Derived, class OutType>
decltype(auto) operator*(const RowScaled<Derived> &func, const Eigen::MatrixBase<OutType> &s) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_.cwiseProduct(s));
}
template <class Derived, class OutType>
decltype(auto) operator*(const Eigen::MatrixBase<OutType> &s, const RowScaled<Derived> &func) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_.cwiseProduct(s));
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Derived, int IR, int OR>
decltype(auto) operator/(const DenseFunctionBase<Derived, IR, OR> &func, double s) {
    return Scaled<Derived>(func.derived(), 1.0 / s);
}
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator/(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return RowScaled<Derived>(func.derived(), s.cwiseInverse());
}
template <class Derived> decltype(auto) operator/(const Scaled<Derived> &func, double s) {
    return Scaled<Derived>(func.func_, func.scale_value_ / s);
}
template <class Derived> decltype(auto) operator/(const RowScaled<Derived> &func, double s) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_ / s);
}

template <class Derived, int IR>
decltype(auto) operator/(double s, const DenseFunctionBase<Derived, IR, 1> &func) {
    return Scaled<CwiseInverse<Derived>>(CwiseInverse<Derived>(func.derived()), s);
}

/////////////////////// Scalar Addition
/// Subtraction///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator+(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<Derived>(func.derived(), s);
}

template <int IR, int OR, class OutType>
auto operator+(const GenericFunction<IR, OR> &func, const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, s);
}

template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator+(const Eigen::MatrixBase<OutType> &s,
                         const DenseFunctionBase<Derived, IR, OR> &func) {
    return FunctionPlusVector<Derived>(func.derived(), s);
}

template <int IR, int OR, class OutType>
auto operator+(const Eigen::MatrixBase<OutType> &s, const GenericFunction<IR, OR> &func) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, s);
}

template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator-(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<Derived>(func.derived(), (-s));
}

template <int IR, int OR, class OutType>
auto operator-(const GenericFunction<IR, OR> &func, const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, (-s));
}

template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator-(const Eigen::MatrixBase<OutType> &s,
                         const DenseFunctionBase<Derived, IR, OR> &func) {
    return FunctionPlusVector<Scaled<Derived>>(Scaled<Derived>(func.derived(), -1.0), s);
}

template <class Derived, int IR>
decltype(auto) operator+(const DenseFunctionBase<Derived, IR, 1> &func, double s) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), st);
}
template <class Derived, int IR>
decltype(auto) operator+(double s, const DenseFunctionBase<Derived, IR, 1> &func) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), st);
}
template <class Derived, int IR>
decltype(auto) operator-(const DenseFunctionBase<Derived, IR, 1> &func, double s) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), (-st));
}
template <class Derived, int IR>
decltype(auto) operator-(double s, const DenseFunctionBase<Derived, IR, 1> &func) {
    Vector1<double> st;
    st[0] = s;

    return FunctionPlusVector<Scaled<Derived>>(Scaled<Derived>(func.derived(), -1.0), st);
}

/////////////////////// Function Scalar Multiplication And Division
///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<VFunc, IR1, OR> &vf,
                         const DenseFunctionBase<SFunc, IR2, 1> &sf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between vector and scalar functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR2, 1> &sf,
                         const DenseFunctionBase<VFunc, IR1, OR> &vf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between scalar and vector functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
template <class VFunc, int IR1, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR2, 1> &sf,
                         const DenseFunctionBase<VFunc, IR1, 1> &vf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between scalar functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator/(const DenseFunctionBase<VFunc, IR1, OR> &vf,
                         const DenseFunctionBase<SFunc, IR2, 1> &sf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator/: input size mismatch between vector and scalar functions");
    return VectorScalarFunctionDivision<VFunc, SFunc>(vf.derived(), sf.derived());
}

/////////////////////// Function Addition
/// Subtraction/////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
decltype(auto) operator+(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                         const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator+: input size mismatch — both functions have "
                  "static sizes that don't match");
    static_assert(OR1 == OR2 || OR1 < 0 || OR2 < 0,
                  "VF operator+: output size mismatch — both functions have "
                  "static sizes that don't match");
    return TwoFunctionSum<Func1, Func2>(f1.derived(), f2.derived());
}

template <class Func1, int IR, int OR, class Func2, class Func3>
decltype(auto) operator+(const DenseFunctionBase<Func3, IR, OR> &f2,
                         const TwoFunctionSum<Func1, Func2> &f1) {
    static_assert(IR == TwoFunctionSum<Func1, Func2>::IRC || IR < 0 ||
                      TwoFunctionSum<Func1, Func2>::IRC < 0,
                  "VF operator+: input size mismatch with sum operand");
    static_assert(OR == TwoFunctionSum<Func1, Func2>::ORC || OR < 0 ||
                      TwoFunctionSum<Func1, Func2>::ORC < 0,
                  "VF operator+: output size mismatch with sum operand");
    return MultiFunctionSum<Func1, Func2, Func3>(f1.func1, f1.func2, f2.derived());
}
template <class Func1, int IR, int OR, class Func2, class Func3, class Func4>
decltype(auto) operator+(const DenseFunctionBase<Func4, IR, OR> &f2,
                         const MultiFunctionSum<Func1, Func2, Func3> &f1) {
    static_assert(IR == MultiFunctionSum<Func1, Func2, Func3>::IRC || IR < 0 ||
                      MultiFunctionSum<Func1, Func2, Func3>::IRC < 0,
                  "VF operator+: input size mismatch with sum operand");
    static_assert(OR == MultiFunctionSum<Func1, Func2, Func3>::ORC || OR < 0 ||
                      MultiFunctionSum<Func1, Func2, Func3>::ORC < 0,
                  "VF operator+: output size mismatch with sum operand");
    return MultiFunctionSum<Func1, Func2, Func3, Func4>(f1.func1, f1.func2, std::get<0>(f1.funcs),
                                                        f2.derived());
}

template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
decltype(auto) operator-(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                         const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator-: input size mismatch — both functions have "
                  "static sizes that don't match");
    static_assert(OR1 == OR2 || OR1 < 0 || OR2 < 0,
                  "VF operator-: output size mismatch — both functions have "
                  "static sizes that don't match");
    return FunctionDifference<Func1, Func2>(f1.derived(), f2.derived());
}

////////////Comparisons///////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

template <class Func1, int IR1, class Func2, int IR2>
auto operator<(const DenseFunctionBase<Func1, IR1, 1> &lhs,
               const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::LessThanFlag,
                                              rhs.derived());
}
template <class Func1, int IR1, class Func2, int IR2>
auto operator<=(const DenseFunctionBase<Func1, IR1, 1> &lhs,
                const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::LessThanEqualToFlag,
                                              rhs.derived());
}
template <class Func1, int IR1, class Func2, int IR2>
auto operator>(const DenseFunctionBase<Func1, IR1, 1> &lhs,
               const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::GreaterThanFlag,
                                              rhs.derived());
}
template <class Func1, int IR1, class Func2, int IR2>
auto operator>=(const DenseFunctionBase<Func1, IR1, 1> &lhs,
                const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(
        lhs.derived(), ConditionalFlags::GreaterThanEqualToFlag, rhs.derived());
}
template <class Func1, int IR1, class Func2, int IR2>
auto operator==(const DenseFunctionBase<Func1, IR1, 1> &lhs,
                const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::EqualToFlag,
                                              rhs.derived());
}

template <class Func1, int IR>
auto operator<(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::LessThanFlag, rhs.derived());
}
template <class Func1, int IR>
auto operator<=(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::LessThanEqualToFlag, rhs.derived());
}
template <class Func1, int IR>
auto operator>(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::GreaterThanFlag, rhs.derived());
}
template <class Func1, int IR>
auto operator>=(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::GreaterThanEqualToFlag, rhs.derived());
}
template <class Func1, int IR>
auto operator==(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::EqualToFlag, rhs.derived());
}

template <class Func1, int IR>
auto operator<(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs > lhs);
}
template <class Func1, int IR>
auto operator<=(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs >= lhs);
}
template <class Func1, int IR>
auto operator>(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs < lhs);
}
template <class Func1, int IR>
auto operator>=(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs <= lhs);
}
template <class Func1, int IR>
auto operator==(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs == lhs);
}

////////////////////// Matrix Products /////////////////////////////////

template <class M1, class M2, int M1Rows, int M1Cols_M2Rows, int M2Cols, int M1Major, int M2Major>
decltype(auto) operator*(const MatrixFunctionView<M1, M1Rows, M1Cols_M2Rows, M1Major> &m1,
                         const MatrixFunctionView<M2, M1Cols_M2Rows, M2Cols, M2Major> &m2) {

    using MatFunc1 = MatrixFunctionView<M1, M1Rows, M1Cols_M2Rows, M1Major>;
    using MatFunc2 = MatrixFunctionView<M2, M1Cols_M2Rows, M2Cols, M2Major>;

    return MatrixFunctionProduct<MatFunc1, MatFunc2>(m1, m2);
}

/// Matrix-vector product: MatrixFunctionView * VF expression.
/// Wraps the VF in a column-vector MatrixFunctionView, then delegates
/// to the existing MatrixFunctionProduct.
template <class M1, int M1Rows, int M1Cols, int M1Major, class Func, int IR, int OR>
decltype(auto) operator*(const MatrixFunctionView<M1, M1Rows, M1Cols, M1Major> &m,
                         const DenseFunctionBase<Func, IR, OR> &v) {
    auto col_vec =
        MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>(v.derived(), m.matrix_cols_, 1);

    using MatFunc1 = MatrixFunctionView<M1, M1Rows, M1Cols, M1Major>;
    using MatFunc2 = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>;

    return MatrixFunctionProduct<MatFunc1, MatFunc2>(m, col_vec);
}
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////// pack — type-erase to GenericFunction ///////////////////

template <class Derived, int IR, int OR>
GenericFunction<IR, OR> DenseFunctionBase<Derived, IR, OR>::pack() const {
    return GenericFunction<IR, OR>(this->derived());
}

/////////////////////// MatrixFunctionView::inverse() /////////////////////

template <class Func, int MRows, int MCols, int MMajor>
auto MatrixFunctionView<Func, MRows, MCols, MMajor>::inverse() const {
    if (this->matrix_rows_ != this->matrix_cols_) {
        throw std::invalid_argument(
            "MatrixFunctionView::inverse: matrix must be square");
    }
    int size = this->matrix_rows_;
    GenericFunction<-1, -1> inv_func;
    if (size == 2) {
        inv_func = MatrixInverse<2, MMajor>(size);
    } else if (size == 3) {
        inv_func = MatrixInverse<3, MMajor>(size);
    } else {
        inv_func = MatrixInverse<-1, MMajor>(size);
    }
    GenericFunction<-1, -1> composed(inv_func.eval(static_cast<const Func &>(*this)));
    return MatrixFunctionView<GenericFunction<-1, -1>, -1, -1, MMajor>(composed, size, size);
}

/////////////////////// quat_rotate ////////////////////////////////////////

/// Rotate a 3-element vector by a 4-element quaternion (scalar-last).
/// Computes: q * (v, 0) * q_inv, then extracts the 3-element vector part.
template <class QFunc, int QIR, int QOR, class VFunc, int VIR, int VOR>
auto quat_rotate(const DenseFunctionBase<QFunc, QIR, QOR> &q,
                 const DenseFunctionBase<VFunc, VIR, VOR> &v) {
    auto qinv = StackedOutputs{q.derived().template head<3>() * (-1.0),
                               q.derived().template coeff<3>()};
    auto vq = v.derived().template padded_lower<1>();
    auto qvq = FunctionQuatProduct<QFunc, decltype(vq)>{q.derived(), vq};
    return FunctionQuatProduct<decltype(qvq), decltype(qinv)>{qvq, qinv}
        .template head<3>();
}

/// Overload accepting Eigen::Vector3d for the vector argument.
template <class QFunc, int QIR, int QOR>
auto quat_rotate(const DenseFunctionBase<QFunc, QIR, QOR> &q,
                 const Eigen::Vector3d &v) {
    auto v_const = Constant<-1, 3>(q.derived().input_rows(), v);
    return quat_rotate(q, v_const);
}

/////////////////////// cwise_product Eigen overloads //////////////////////

/// Element-wise product: VF * Eigen::VectorXd.
/// Uses RowScaled internally (same as Python binding).
template <class Func, int IR, int OR>
auto cwise_product(const DenseFunctionBase<Func, IR, OR> &f,
                   const Eigen::VectorXd &v) {
    return RowScaled<Func>(f.derived(), v);
}

/// Element-wise product: Eigen::VectorXd * VF (commutative).
template <class Func, int IR, int OR>
auto cwise_product(const Eigen::VectorXd &v,
                   const DenseFunctionBase<Func, IR, OR> &f) {
    return RowScaled<Func>(f.derived(), v);
}

} // namespace tycho::vf
