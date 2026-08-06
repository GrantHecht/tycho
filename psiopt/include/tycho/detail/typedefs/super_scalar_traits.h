// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// SuperScalar (Eigen::Array<double, W, 1>) trait registrations needed for
// Eigen 5 to accept it as a Matrix Scalar.
//
// Eigen 3.4 was lenient about non-arithmetic Scalar types; Eigen 5 enforces
// stricter contracts. When an Eigen::Matrix<Eigen::Array<double, W, 1>, ...>
// is evaluated, Eigen 5 reaches code paths that call:
//   - numext::conj(scalar)
//   - numext::imag(scalar) / numext::real(scalar)
//   - internal::is_exactly_zero(scalar)
//   - typename NumTraits<Scalar>::Literal{0} construction
//   - negate_impl<Scalar>::run(scalar) returning Scalar
//
// For built-in Scalars these are trivially defined; for our SuperScalar pattern
// the default overloads end up returning ArrayBase<...> expression templates
// with protected destructors, which fail to compile in stricter contexts.
//
// This header registers minimal real-typed overloads that route the relevant
// numext functions back to the concrete Array<double, W, 1> type. Conj/imag
// for a real type are identity / zero respectively.
//
// The header is included from eigen_types.h after <Eigen/Core>, so by the
// time tycho code instantiates an Eigen::Matrix<Array<...>, ...>, these
// overloads are already in scope.
// =============================================================================
#pragma once

#include <Eigen/Core>

namespace Eigen {

namespace internal {

// Mark Array<double, W, 1> as "arithmetic" so Eigen 5's binary-op overloads
// (e.g. ArrayBase::pow(scalar_exponent)) accept it as a scalar exponent.
// Required for tycho::vf::CwisePow::cwise_compute(x, fx) where the SuperScalar
// path constructs `Scalar(power)` from a double and passes it to .pow().
template <>
struct is_arithmetic<Array<double, 2, 1>> {
    enum { value = true };
};
template <>
struct is_arithmetic<Array<double, 4, 1>> {
    enum { value = true };
};
template <>
struct is_arithmetic<Array<double, 8, 1>> {
    enum { value = true };
};

// Override Eigen's "math-function filtering" for Array<double, W, 1> so that
// numext::conj/imag/real/abs are dispatched as scalar operations (returning
// Array<double, W, 1>) rather than ArrayBase coefficient-wise expressions.
//
// Without this override, ArrayBase's Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl
// member typedef routes the math impls through ArrayBase, producing
// CwiseUnaryOp expression templates with an ArrayBase return type. In Eigen 5,
// ArrayBase's destructor is protected, so those temporaries fail to compile
// at the inner-product / negate / is_exactly_zero call sites.
//
// Setting type = Array<...> short-circuits the filter and makes the scalar
// path (return x;) the active dispatch.
//
// Full specializations are needed (not partial-on-W) because Eigen's existing
// partial specialization in MathFunctions.h:59 also matches Array<...>, and
// two partial specializations would be ambiguous.
template <>
struct global_math_functions_filtering_base<Array<double, 2, 1>, void> {
    typedef Array<double, 2, 1> type;
};
template <>
struct global_math_functions_filtering_base<Array<double, 4, 1>, void> {
    typedef Array<double, 4, 1> type;
};
template <>
struct global_math_functions_filtering_base<Array<double, 8, 1>, void> {
    typedef Array<double, 8, 1> type;
};

} // namespace internal

// Provide a Literal that is constructible from int{0}/int{1} so
// is_exactly_zero / is_exactly_one (which call NumTraits<T>::Literal{0}) work.
// We forward to the existing NumTraits<Array<...>> default but override
// only the Literal alias.
template <int W>
struct NumTraits<Array<double, W, 1>> : NumTraits<double> {
    using Real = Array<double, W, 1>;
    using NonInteger = Array<double, W, 1>;
    using Nested = Array<double, W, 1>;
    using Literal = Array<double, W, 1>;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };
};

namespace numext {

// numext::conj returning the same Array (real-typed).
template <int W>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Array<double, W, 1>
conj(const Array<double, W, 1>& x) {
    return x;
}

// numext::real returning the same Array.
template <int W>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Array<double, W, 1>
real(const Array<double, W, 1>& x) {
    return x;
}

// numext::imag returning a zero-Array.
template <int W>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Array<double, W, 1>
imag(const Array<double, W, 1>& /*x*/) {
    return Array<double, W, 1>::Zero();
}

// numext::abs returning the absolute-value Array.
template <int W>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Array<double, W, 1>
abs(const Array<double, W, 1>& x) {
    return x.abs();
}

// numext::abs2 returning the squared-magnitude Array.
template <int W>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Array<double, W, 1>
abs2(const Array<double, W, 1>& x) {
    return x.abs2();
}

// equal_strict for Array Scalar comparison — true iff all lanes equal.
// Lives in Eigen::numext alongside the primary equal_strict_impl.
// Full specializations (one per W) avoid ambiguity with Eigen's mixed-
// signedness partial specialisations.
template <>
struct equal_strict_impl<Array<double, 2, 1>, Array<double, 2, 1>, false, true, false, true> {
    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool
    run(const Array<double, 2, 1>& x, const Array<double, 2, 1>& y) {
        return (x == y).all();
    }
};
template <>
struct equal_strict_impl<Array<double, 4, 1>, Array<double, 4, 1>, false, true, false, true> {
    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool
    run(const Array<double, 4, 1>& x, const Array<double, 4, 1>& y) {
        return (x == y).all();
    }
};
template <>
struct equal_strict_impl<Array<double, 8, 1>, Array<double, 8, 1>, false, true, false, true> {
    static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool
    run(const Array<double, 8, 1>& x, const Array<double, 8, 1>& y) {
        return (x == y).all();
    }
};

} // namespace numext

namespace internal {

// is_exactly_zero for the Array Scalar — true iff all lanes are exactly zero.
template <int W>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool
is_exactly_zero(const Array<double, W, 1>& x) {
    return (x == Array<double, W, 1>::Zero()).all();
}

// is_exactly_one — true iff all lanes are exactly one.
template <int W>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool
is_exactly_one(const Array<double, W, 1>& x) {
    return (x == Array<double, W, 1>::Ones()).all();
}


// Specialize negate_impl for Array Scalar so the return type stays Array
// instead of CwiseUnaryOp<scalar_opposite_op, ...>.
template <int W>
struct negate_impl<Array<double, W, 1>, /*IsInteger=*/false> {
    static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    Array<double, W, 1> run(const Array<double, W, 1>& a) {
        return -a.matrix();  // routes through Matrix's negation, returns Array via assignment
    }
};

} // namespace internal

} // namespace Eigen
