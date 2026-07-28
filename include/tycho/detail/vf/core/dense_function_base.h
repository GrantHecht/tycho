// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Implements the Base class for all dense vector functions in Tycho.
// Forwards the Derived class and compile time input(IR) and output(OR) rows down the CRTP
// inheritance chain. Also inherits from domain holder so dynamic sized vector functions can
// hold an array containing their true input domain that is computed at run-time. Also defines the
// default compile time INPUT_DOMAIN of both constant and dynamic size functions. Composite derived
// classes will override this typedef by design, in order to explicitly participate in the input
// domain tracking system. The default input domain simply says that all inputs are used. This input
// domain info is used to implement the default set of functions for manipulating the
// jacobian,hessian,and gradient of Derived function such as right_jacobian_product. Specialized
// derived classes should perform simple overloads of these methods.
//
// This class also defines most of the vector function indexing (.segment) and unary(.normalize)
// and binary(.dot) mathematical operations that are used when writing expressions.
//
// Additionally, This class also defines the .compute_jacobian etc. methods in terms of the
// compute_jacobian_impl methods implemented in derived classes. These methods are then used to
// implement the dense vector functions interface with psiopt through the constraints_jacobian etc.
// methods.
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/assignment_types.h"
#include "tycho/detail/vf/core/computable_base.h"
#include "tycho/detail/vf/core/dense_function_operations.h"
#include "tycho/detail/vf/core/expression_fwd_declarations.h"
#include "tycho/detail/vf/core/function_domains.h"
#include "tycho/detail/vf/core/function_type_def_macros.h"
#include "tycho/detail/vf/core/var_tags.h"
#include "tycho/detail/vf/operators/binary_math.h"

namespace tycho::vf {

/// @brief CRTP base for all dense VectorFunctions in Tycho.
///
/// Extends @ref ComputableBase with everything specific to *dense* functions:
/// the Jacobian/Hessian type aliases, the input-domain tracking machinery (which
/// records which input columns a function actually depends on), the full
/// expression-builder DSL (`segment`, `head`, `tail`, `dot`, `cross`, `norm`,
/// `padded`, the coefficient-wise math operators, etc.), the
/// `compute_jacobian` / `adjointhessian` family of differentiation entry points,
/// and the `constraints_jacobian` / objective interfaces that connect a function
/// to the PSIOPT optimizer.
///
/// The default @ref INPUT_DOMAIN declares that every input is used; composite
/// derived classes override it to participate in sparsity-aware Jacobian/Hessian
/// products such as @ref right_jacobian_product.
///
/// @tparam Derived  CRTP derived (user) function type.
/// @tparam IR       Input rows at compile time (`-1` for dynamic).
/// @tparam OR       Output rows at compile time (`-1` for dynamic).
/// @ingroup vf
// Canonical CRTP hook inventory: see computable_base.h ("Canonical CRTP hook inventory").
template <class Derived, int IR, int OR>
struct DenseFunctionBase : ComputableBase<Derived, IR, OR>, DomainHolder<IR> {
    /// @brief Immediate base class supplying the evaluation interface.
    using Base = ComputableBase<Derived, IR, OR>;

    /////////////////////////////////////////////////////////////
    /// @brief Output column-vector type (inherited from @ref ComputableBase).
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    /// @brief Input column-vector type (inherited from @ref ComputableBase).
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    /// @brief Gradient column-vector type (inherited from @ref ComputableBase).
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;
    /// @brief Jacobian matrix type, shape (OR, IR).
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
    /// @brief Hessian matrix type, shape (IR, IR).
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Hessian = Eigen::Matrix<Scalar, IR, IR>;
    /////////////////////////////////////////////////////////////

    /// @brief Default input domain: a single block spanning all @c IR inputs.
    using INPUT_DOMAIN = SingleDomain<IR, 0, IR>;

    /// @brief Rebinds this function as a base class of a new derived type.
    /// @tparam NewDerived  The new CRTP derived type to host this function.
    template <class NewDerived> using AsBaseClass = FunctionHolder<NewDerived, Derived, IR, OR>;

    /// @brief Self type alias for the derived function.
    using NAME = Derived;

    /// @brief Selector for composing this function as the outer operand of a nested call.
    /// @tparam Func  Inner function type.
    template <class Func> using EVALOP = NestedFunctionSelector<Derived, Func>;
    /// @brief Selector for composing this function as the inner operand of a nested call.
    /// @tparam Func  Outer function type.
    template <class Func> using FWDOP = NestedFunctionSelector<Func, Derived>;
    /// @brief Selector for composing a @ref Segment view over this function's output.
    /// @tparam SZ  Segment size.
    /// @tparam ST  Segment start index.
    template <int SZ, int ST>
    using SEGMENTOP = NestedFunctionSelector<Segment<OR, SZ, ST>, Derived>;

    /// @brief Set both input and output row counts and update the input domain.
    /// @param inputrows   Number of input rows.
    /// @param outputrows  Number of output rows.
    void set_io_rows(int inputrows, int outputrows) {
        Base::set_io_rows(inputrows, outputrows);
        if constexpr (IR < 0) {
            DomainMatrix dmn(2, 1);
            dmn(0, 0) = 0;
            dmn(1, 0) = inputrows;
            this->set_input_domain(inputrows, {dmn});
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Compose this function with an inner function: `this(f(.))`.
    /// @tparam Func     Inner function type.
    /// @tparam FuncIRC  Inner function input rows.
    /// @param f  Inner function whose output feeds this function's input.
    /// @return The nested composition expression.
    template <class Func, int FuncIRC>
    auto operator()(const DenseFunctionBase<Func, FuncIRC, IR> &f) const {
        return EVALOP<Func>::make_nested(this->derived(), f.derived());
    }

    /// @brief Compose this function with an inner function: `this(f(.))`.
    /// @tparam Func     Inner function type.
    /// @tparam FuncIRC  Inner function input rows.
    /// @tparam FuncORC  Inner function output rows.
    /// @param f  Inner function whose output feeds this function's input.
    /// @return The nested composition expression.
    template <class Func, int FuncIRC, int FuncORC>
        requires Composable<Func, Derived>
    auto eval(const DenseFunctionBase<Func, FuncIRC, FuncORC> &f) const {
        return EVALOP<Func>::make_nested(this->derived(), f.derived());
    }

    /// @brief Output sub-vector with compile-time size and start.
    /// @tparam SZ  Segment size.
    /// @tparam ST  Segment start index.
    /// @return Expression viewing `SZ` outputs starting at `ST`.
    template <int SZ, int ST> auto segment() const {
        return SEGMENTOP<SZ, ST>::make_nested(Segment<OR, SZ, ST>(this->output_rows(), SZ, ST),
                                              this->derived());
    }

    /// @brief Output sub-vector with compile-time hints and runtime start/size.
    /// @tparam SZ  Segment size hint.
    /// @tparam ST  Segment start hint.
    /// @param start  Runtime start index.
    /// @param size   Runtime segment size.
    /// @return Expression viewing @p size outputs starting at @p start.
    template <int SZ, int ST> auto segment(int start, int size) const {
        return SEGMENTOP<SZ, ST>::make_nested(Segment<OR, SZ, ST>(this->output_rows(), size, start),
                                              this->derived());
    }

    /// @brief Output sub-vector with fully dynamic start and size.
    /// @param start  Runtime start index.
    /// @param size   Runtime segment size.
    /// @return Expression viewing @p size outputs starting at @p start.
    auto segment(int start, int size) const {
        return SEGMENTOP<-1, -1>::make_nested(Segment<OR, -1, -1>(this->output_rows(), size, start),
                                              this->derived());
    }

    /// @brief Output sub-vector with compile-time size and runtime start.
    /// @tparam SZ  Segment size.
    /// @param start  Runtime start index.
    /// @return Expression viewing `SZ` outputs starting at @p start.
    template <int SZ> auto segment(int start) const {
        return SEGMENTOP<SZ, -1>::make_nested(Segment<OR, SZ, -1>(this->output_rows(), SZ, start),
                                              this->derived());
    }

    /// @brief First `SZ` outputs (compile-time size).
    /// @tparam SZ  Number of leading outputs.
    /// @return Expression viewing the first `SZ` outputs.
    template <int SZ> auto head() const {
        return SEGMENTOP<SZ, 0>::make_nested(Segment<OR, SZ, 0>(this->output_rows(), SZ, 0),
                                             this->derived());
    }

    /// @brief First @p sz outputs (compile-time hint, runtime size).
    /// @tparam SZ  Size hint.
    /// @param sz  Runtime number of leading outputs.
    /// @return Expression viewing the first @p sz outputs.
    template <int SZ> auto head(int sz) const {
        return SEGMENTOP<SZ, 0>::make_nested(Segment<OR, SZ, 0>(this->output_rows(), sz, 0),
                                             this->derived());
    }

    /// @brief First @p sz outputs (fully dynamic).
    /// @param sz  Runtime number of leading outputs.
    /// @return Expression viewing the first @p sz outputs.
    auto head(int sz) const {
        return SEGMENTOP<-1, -1>::make_nested(Segment<OR, -1, -1>(this->output_rows(), sz, 0),
                                              this->derived());
    }

    /// @brief Last `SZ` outputs (compile-time size).
    /// @tparam SZ  Number of trailing outputs.
    /// @return Expression viewing the last `SZ` outputs.
    template <int SZ> auto tail() const {
        return SEGMENTOP<SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>::make_nested(
            Segment<OR, SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>(this->output_rows(), SZ,
                                                                  this->output_rows() - SZ),
            this->derived());
    }

    /// @brief Last @p sz outputs (compile-time hint, runtime size).
    /// @tparam SZ  Size hint.
    /// @param sz  Runtime number of trailing outputs.
    /// @return Expression viewing the last @p sz outputs.
    template <int SZ> decltype(auto) tail(int sz) const {
        return SEGMENTOP<SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>::make_nested(
            Segment<OR, SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>(this->output_rows(), sz,
                                                                  this->output_rows() - sz),
            this->derived());
    }

    /// @brief Last @p sz outputs (fully dynamic).
    /// @param sz  Runtime number of trailing outputs.
    /// @return Expression viewing the last @p sz outputs.
    decltype(auto) tail(int sz) const {
        return SEGMENTOP<-1, -1>::make_nested(
            Segment<OR, -1, -1>(this->output_rows(), sz, this->output_rows() - sz),
            this->derived());
    }

    /// @brief Single output coefficient at a compile-time index.
    /// @tparam ELE  Element index.
    /// @return Scalar expression viewing output element `ELE`.
    template <int ELE> decltype(auto) coeff() const {
        return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->output_rows(), 1, ELE),
                                              this->derived());
    }

    /// @brief Single output coefficient (compile-time hint, runtime index).
    /// @tparam ELE  Index hint.
    /// @param ele  Runtime element index.
    /// @return Scalar expression viewing output element @p ele.
    template <int ELE> decltype(auto) coeff(int ele) const {
        return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->output_rows(), 1, ele),
                                              this->derived());
    }

    /// @brief Single output coefficient at a runtime index.
    /// @param ele  Runtime element index.
    /// @return Scalar expression viewing output element @p ele.
    auto coeff(int ele) const {
        return SEGMENTOP<1, -1>::make_nested(Segment<OR, 1, -1>(this->output_rows(), 1, ele),
                                             this->derived());
    }

    /// @brief Subscript operator selecting a single output coefficient.
    /// @tparam ELE  Element index.
    /// @param ele  Compile-time index tag.
    /// @return Scalar expression viewing output element `ELE`.
    template <int ELE> decltype(auto) operator[](std::integral_constant<int, ELE> ele) const {
        return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->output_rows(), 1, ELE),
                                              this->derived());
    }

    /// @brief Select an arbitrary set of output elements by compile-time index.
    /// @tparam EL1  First element index.
    /// @tparam ELS  Remaining element indices.
    /// @return Expression gathering the selected output elements.
    template <int EL1, int... ELS> decltype(auto) elements() const {
        return FWDOP<Elements<OR, EL1, ELS...>>::make_nested(
            Elements<OR, EL1, ELS...>(this->output_rows()), this->derived());
    }

    /// @brief Normalize the output to unit Euclidean norm: `f / ||f||`.
    /// @return Expression of the normalized output.
    decltype(auto) normalized() const {
        return FWDOP<Normalized<OR>>::make_nested(Normalized<OR>(this->output_rows()),
                                                  this->derived());
    }

    /// @brief Output normalized then raised to an integer power of its norm.
    /// @tparam PW  Norm power exponent.
    /// @return Expression `f / ||f||^PW`.
    template <int PW> decltype(auto) normalized_power() const {
        return FWDOP<NormalizedPower<OR, PW>>::make_nested(
            NormalizedPower<OR, PW>(this->output_rows()), this->derived());
    }

    /// @brief Euclidean norm of the output: `||f||`.
    /// @return Scalar expression of the norm.
    decltype(auto) norm() const {
        return FWDOP<Norm<OR>>::make_nested(Norm<OR>(this->output_rows()), this->derived());
    }

    /// @brief Squared Euclidean norm of the output: `||f||^2`.
    /// @return Scalar expression of the squared norm.
    decltype(auto) squared_norm() const {
        return FWDOP<SquaredNorm<OR>>::make_nested(SquaredNorm<OR>(this->output_rows()),
                                                   this->derived());
    }

    /// @brief Reciprocal Euclidean norm of the output: `1 / ||f||`.
    /// @return Scalar expression of the inverse norm.
    decltype(auto) inverse_norm() const {
        return FWDOP<InverseNorm<OR>>::make_nested(InverseNorm<OR>(this->output_rows()),
                                                   this->derived());
    }

    /// @brief Reciprocal squared Euclidean norm of the output: `1 / ||f||^2`.
    /// @return Scalar expression of the inverse squared norm.
    decltype(auto) inverse_squared_norm() const {
        return FWDOP<InverseSquaredNorm<OR>>::make_nested(
            InverseSquaredNorm<OR>(this->output_rows()), this->derived());
    }

    /// @brief Integer power of the output norm: `||f||^PW`.
    /// @tparam PW  Norm power exponent.
    /// @return Scalar expression of the norm raised to `PW`.
    template <int PW> decltype(auto) norm_power() const {
        return FWDOP<NormPower<OR, PW>>::make_nested(NormPower<OR, PW>(this->output_rows()),
                                                     this->derived());
    }

    /// @brief Reciprocal integer power of the output norm: `1 / ||f||^PW`.
    /// @tparam PW  Norm power exponent.
    /// @return Scalar expression of the inverse norm raised to `PW`.
    template <int PW> decltype(auto) inverse_norm_power() const {
        return FWDOP<InverseNormPower<OR, PW>>::make_nested(
            InverseNormPower<OR, PW>(this->output_rows()), this->derived());
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Reinterpret the output as a matrix of the given shape.
    /// @tparam MRows  Compile-time row count.
    /// @tparam MCols  Compile-time column count.
    /// @tparam MMajor Storage order (defaults to column-major).
    /// @param rows  Runtime row count.
    /// @param cols  Runtime column count.
    /// @return A matrix view over the output.
    template <int MRows, int MCols, int MMajor = Eigen::ColMajor>
    auto matrix(int rows, int cols) const {
        return MatrixFunctionView<Derived, MRows, MCols, MMajor>(this->derived(), rows, cols);
    }

    /// @brief Reinterpret the output as a column-major dynamic matrix.
    /// @param rows  Runtime row count.
    /// @param cols  Runtime column count.
    /// @return A column-major matrix view over the output.
    auto colmatrix(int rows, int cols) const {
        return MatrixFunctionView<Derived, -1, -1, Eigen::ColMajor>(this->derived(), rows, cols);
    }

    /// @brief Reinterpret the output as a row-major dynamic matrix.
    /// @param rows  Runtime row count.
    /// @param cols  Runtime column count.
    /// @return A row-major matrix view over the output.
    auto rowmatrix(int rows, int cols) const {
        return MatrixFunctionView<Derived, -1, -1, Eigen::RowMajor>(this->derived(), rows, cols);
    }

    /// @brief Reinterpret the output as a matrix of compile-time shape.
    /// @tparam MRows  Compile-time row count.
    /// @tparam MCols  Compile-time column count.
    /// @tparam MMajor Storage order (defaults to column-major).
    /// @return A matrix view over the output.
    template <int MRows, int MCols, int MMajor = Eigen::ColMajor> auto matrix() const {
        return MatrixFunctionView<Derived, MRows, MCols, MMajor>(this->derived(), MRows, MCols);
    }

    /// @brief Reinterpret the output as a single-column matrix.
    /// @return A column-vector matrix view over the output.
    auto colvector() const {
        return MatrixFunctionView<Derived, OR, 1, Eigen::ColMajor>(this->derived(),
                                                                   this->output_rows(), 1);
    }

    /// @brief Reinterpret the output as a single-row matrix.
    /// @return A row-vector matrix view over the output.
    auto rowvector() const {
        return MatrixFunctionView<Derived, 1, OR, Eigen::ColMajor>(this->derived(), 1,
                                                                   this->output_rows());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Coefficient-wise sine of the output.
    /// @return Expression `sin(f)`.
    auto sin() const { return CwiseSin<Derived>(this->derived()); }
    /// @brief Coefficient-wise cosine of the output.
    /// @return Expression `cos(f)`.
    auto cos() const { return CwiseCos<Derived>(this->derived()); }
    /// @brief Coefficient-wise arcsine of the output.
    /// @return Expression `asin(f)`.
    auto arc_sin() const { return CwiseArcSin<Derived>(this->derived()); }
    /// @brief Coefficient-wise arccosine of the output.
    /// @return Expression `acos(f)`.
    auto arc_cos() const { return CwiseArcCos<Derived>(this->derived()); }
    /// @brief Coefficient-wise tangent of the output.
    /// @return Expression `tan(f)`.
    auto tan() const { return CwiseTan<Derived>(this->derived()); }
    /// @brief Coefficient-wise square of the output.
    /// @return Expression `f^2` (element-wise).
    auto square() const { return CwiseSquare<Derived>(this->derived()); }
    /// @brief Coefficient-wise square root of the output.
    /// @return Expression `sqrt(f)` (element-wise).
    auto sqrt() const { return CwiseSqrt<Derived>(this->derived()); }
    /// @brief Coefficient-wise exponential of the output.
    /// @return Expression `exp(f)` (element-wise).
    auto exp() const { return CwiseExp<Derived>(this->derived()); }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Append `LP` zero rows below the output (compile-time count).
    /// @tparam LP  Number of trailing zero rows to add.
    /// @return Expression with `LP` zeros appended after the output.
    template <int LP> auto padded_lower() const {
        return PaddedOutput<Derived, 0, LP>(this->derived(), 0, LP);
    }
    /// @brief Prepend `UP` zero rows above the output (compile-time count).
    /// @tparam UP  Number of leading zero rows to add.
    /// @return Expression with `UP` zeros prepended before the output.
    template <int UP> auto padded_upper() const {
        return PaddedOutput<Derived, UP, 0>(this->derived(), UP, 0);
    }
    /// @brief Pad the output with zeros above and below (compile-time counts).
    /// @tparam UP  Number of leading zero rows.
    /// @tparam LP  Number of trailing zero rows.
    /// @return Expression with `UP` leading and `LP` trailing zeros.
    template <int UP, int LP> auto padded() const {
        return PaddedOutput<Derived, UP, LP>(this->derived(), UP, LP);
    }

    /// @brief Append @p LP zero rows below the output (runtime count).
    /// @param LP  Number of trailing zero rows to add.
    /// @return Expression with @p LP zeros appended after the output.
    auto padded_lower(int LP) const { return PaddedOutput<Derived, 0, -1>(this->derived(), 0, LP); }
    /// @brief Prepend @p UP zero rows above the output (runtime count).
    /// @param UP  Number of leading zero rows to add.
    /// @return Expression with @p UP zeros prepended before the output.
    auto padded_upper(int UP) const { return PaddedOutput<Derived, -1, 0>(this->derived(), UP, 0); }
    /// @brief Pad the output with zeros above and below (runtime counts).
    /// @param UP  Number of leading zero rows.
    /// @param LP  Number of trailing zero rows.
    /// @return Expression with @p UP leading and @p LP trailing zeros.
    auto padded(int UP, int LP) const {
        return PaddedOutput<Derived, -1, -1>(this->derived(), UP, LP);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Coefficient-wise product with another function: `this .* f`.
    /// @tparam Func  Type of the other function (same input/output size).
    /// @param f  The other function.
    /// @return Expression of the element-wise product.
    template <class Func>
    decltype(auto) cwise_product(const DenseFunctionBase<Func, IR, OR> &f) const {
        return CwiseFunctionProduct<Derived, Func>(this->derived(), f.derived());
    }
    /// @brief Coefficient-wise reciprocal of the output: `1 / f` (element-wise).
    /// @return Expression of the element-wise inverse.
    decltype(auto) cwise_inverse() const { return CwiseInverse<Derived>(this->derived()); }
    /// @brief Sum of all output coefficients into a scalar.
    /// @return Scalar expression of the sum (a copy of the function when `OR == 1`).
    decltype(auto) sum() const {
        if constexpr (OR == 1)
            return Derived(this->derived());
        else
            return CwiseSum<Derived>(this->derived());
    }

    /// @brief Vector cross product with another 3-vector function: `this x f`.
    /// @tparam Func  Type of the other 3-output function.
    /// @param f  The other function (output size 3).
    /// @return Expression of the cross product.
    template <class Func> decltype(auto) cross(const DenseFunctionBase<Func, IR, 3> &f) const {
        return FunctionCrossProduct<Derived, Func>(this->derived(), f.derived());
    }

    /// @brief Dot product with another function: `this . f`.
    /// @tparam Func  Type of the other function (same input/output size).
    /// @param f  The other function.
    /// @return Scalar expression of the dot product.
    template <class Func> decltype(auto) dot(const DenseFunctionBase<Func, IR, OR> &f) const {
        return FunctionDotProduct<Derived, Func>(this->derived(), f.derived());
    }
    /// @brief Scale this vector function by a scalar-valued function: `f * this`.
    /// @tparam ScalFunc  Type of the scalar (1-output) scaling function.
    /// @param f  The scalar-valued scaling function.
    /// @return Expression of the scaled vector.
    template <class ScalFunc>
    decltype(auto) scale(const DenseFunctionBase<ScalFunc, IR, 1> &f) const {
        return VectorScalarFunctionProduct<Derived, ScalFunc>(this->derived(), f.derived());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Evaluate the function and its Jacobian: `fx = f(x)`, `jx = J(x)`.
    ///
    /// CRTP entry point dispatching to `Derived::compute_jacobian_impl`. When
    /// called with a SuperScalar pack but the derived function is not
    /// vectorizable, this scalarizes the call lane-by-lane.
    ///
    /// @tparam InType   Eigen type of the input @p x.
    /// @tparam OutType  Eigen type of the output @p fx_.
    /// @tparam JacType  Eigen matrix type of the Jacobian @p jx_.
    /// @param x    Input vector, shape (IR,).
    /// @param fx_  Output vector, shape (OR,); overwritten with `f(x)`.
    /// @param jx_  Output Jacobian, shape (OR, IR); overwritten with `J(x)`.
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                 CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        if constexpr (!Vectorizable<Derived>) {
            if constexpr (Is_SuperScalar<Scalar>::value) {
                VecRef<OutType> fx = fx_.const_cast_derived();
                VecRef<JacType> jx = jx_.const_cast_derived();

                typedef typename Scalar::Scalar RealScalar;

                Input<RealScalar> x_r;
                Output<RealScalar> fx_r;
                Jacobian<RealScalar> jx_r;

                const int IRR = this->input_rows();
                const int ORR = this->output_rows();

                if constexpr (Base::InputIsDynamic)
                    x_r.resize(IRR);
                if constexpr (Base::OutputIsDynamic)
                    fx_r.resize(ORR);
                if constexpr (Base::JacobianIsDynamic)
                    jx_r.resize(ORR, IRR);

                for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                    for (int j = 0; j < IRR; j++)
                        x_r[j] = x[j][i];
                    this->derived().compute_jacobian_impl(x_r, fx_r, jx_r);
                    for (int j = 0; j < ORR; j++)
                        fx[j][i] = fx_r[j];

                    for (int j = 0; j < IRR; j++)
                        for (int k = 0; k < ORR; k++)
                            jx(k, j)[i] = jx_r(k, j);

                    // See computable_base.h::compute's identical comment: only
                    // the NEXT lane needs fx_r/jx_r re-zeroed; skip on the
                    // final lane where they are dead.
                    if (i + 1 < Scalar::SizeAtCompileTime) {
                        fx_r.setZero();
                        jx_r.setZero();
                    }
                }

            } else {
                this->derived().compute_jacobian_impl(x, fx_, jx_);
            }
        } else {
            this->derived().compute_jacobian_impl(x, fx_, jx_);
        }
    }

    /// @brief Compute the Jacobian (function value discarded).
    /// @tparam InType   Eigen type of the input @p x.
    /// @tparam JacType  Eigen matrix type of the Jacobian @p jx_.
    /// @param x    Input vector, shape (IR,).
    /// @param jx_  Output Jacobian, shape (OR, IR); overwritten with `J(x)`.
    template <class InType, class JacType>
    inline void jacobian(CVecRef<InType> x, CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx_(this->output_rows());
        fx_.setZero();
        this->derived().compute_jacobian(x, fx_, jx_);
    }

    /// @brief Compute and return the Jacobian by value.
    /// @tparam InType  Eigen type of the input @p x.
    /// @param x  Input vector, shape (IR,).
    /// @return The Jacobian `J(x)`, shape (OR, IR).
    template <class InType>
    inline Jacobian<typename InType::Scalar> jacobian(CVecRef<InType> x) const {
        typedef typename InType::Scalar Scalar;
        Jacobian<Scalar> jx(this->output_rows(), this->input_rows());
        jx.setZero();
        this->derived().jacobian(x, jx);
        return jx;
    }

    /// @brief Contract a Jacobian with an adjoint vector: `adjgrad = J^T * adjvars`.
    /// @tparam JacType      Eigen matrix type of the Jacobian @p jx_.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient @p adjgrad_.
    /// @tparam AdjVarType   Eigen type of the adjoint vector @p adjvars.
    /// @param jx_      Jacobian, shape (OR, IR).
    /// @param adjgrad_ Output adjoint gradient, shape (IR,).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    template <class JacType, class AdjGradType, class AdjVarType>
    inline void jacobian_x_adjoint(CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
                                   CVecRef<AdjVarType> adjvars) const {
        typedef typename JacType::Scalar JScalar;
        typedef typename AdjVarType::Scalar AScalar;
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();

        if constexpr (std::is_same<JScalar, AScalar>::value) {
            adjgrad.noalias() = (adjvars.transpose() * jx_).transpose();
        } else {
            adjgrad.setZero();
            for (int i = 0; i < this->input_rows(); i++) {
                for (int j = 0; j < this->output_rows(); j++) {
                    adjgrad[i] += adjvars[j] * jx_(j, i);
                }
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Evaluate the function and the adjoint gradient via the Jacobian.
    ///
    /// Default dense implementation: forms `J(x)` then contracts it as
    /// `adjgrad = J^T * adjvars`.
    ///
    /// @tparam InType       Eigen type of the input @p x.
    /// @tparam OutType      Eigen type of the output @p fx_.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient @p adjgrad_.
    /// @tparam AdjVarType   Eigen type of the adjoint vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param fx_      Output vector, shape (OR,); overwritten with `f(x)`.
    /// @param adjgrad_ Output adjoint gradient, shape (IR,).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    template <class InType, class OutType, class AdjGradType, class AdjVarType>
    inline void compute_adjointgradient(CVecRef<InType> x, CVecRef<OutType> fx_,
                                        CVecRef<AdjGradType> adjgrad_,
                                        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Jacobian<Scalar> jx(this->output_rows(), this->input_rows());
        jx.setZero();
        this->derived().compute_jacobian(x, fx_, jx);
        this->jacobian_x_adjoint(jx, adjgrad_, adjvars);
    }

    /// @brief Evaluate the function, its Jacobian, and the adjoint gradient.
    ///
    /// Computes `f(x)`, `J(x)`, and `adjgrad = J^T * adjvars` in one call.
    ///
    /// @tparam InType       Eigen type of the input @p x.
    /// @tparam OutType      Eigen type of the output @p fx_.
    /// @tparam JacType      Eigen matrix type of the Jacobian @p jx_.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient @p adjgrad_.
    /// @tparam AdjVarType   Eigen type of the adjoint vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param fx_      Output vector, shape (OR,); overwritten with `f(x)`.
    /// @param jx_      Output Jacobian, shape (OR, IR); overwritten with `J(x)`.
    /// @param adjgrad_ Output adjoint gradient, shape (IR,).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(CVecRef<InType> x, CVecRef<OutType> fx_,
                                                 CMatRef<JacType> jx_,
                                                 CVecRef<AdjGradType> adjgrad_,
                                                 CVecRef<AdjVarType> adjvars) const {
        this->derived().compute_jacobian(x, fx_, jx_);
        this->jacobian_x_adjoint(jx_, adjgrad_, adjvars);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Evaluate value, Jacobian, adjoint gradient, and adjoint Hessian.
    ///
    /// The full second-order entry point. Computes `f(x)`, `J(x)`,
    /// `adjgrad = J^T * adjvars`, and the adjoint Hessian
    /// @f$ \sum_k \lambda_k \nabla^2 f_k(x) @f$. Dispatches to
    /// `Derived::compute_jacobian_adjointgradient_adjointhessian_impl`,
    /// scalarizing lane-by-lane for non-vectorizable SuperScalar calls.
    ///
    /// @tparam InType       Eigen type of the input @p x.
    /// @tparam OutType      Eigen type of the output @p fx_.
    /// @tparam JacType      Eigen matrix type of the Jacobian @p jx_.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient @p adjgrad_.
    /// @tparam AdjHessType  Eigen matrix type of the adjoint Hessian @p adjhess_.
    /// @tparam AdjVarType   Eigen type of the adjoint vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param fx_      Output vector, shape (OR,); overwritten with `f(x)`.
    /// @param jx_      Output Jacobian, shape (OR, IR).
    /// @param adjgrad_ Output adjoint gradient, shape (IR,).
    /// @param adjhess_ Output adjoint Hessian, shape (IR, IR).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian(CVecRef<InType> x,
                                                                CVecRef<OutType> fx_,
                                                                CMatRef<JacType> jx_,
                                                                CVecRef<AdjGradType> adjgrad_,
                                                                CMatRef<AdjHessType> adjhess_,
                                                                CVecRef<AdjVarType> adjvars) const {

        typedef typename InType::Scalar Scalar;
        if constexpr (!Vectorizable<Derived>) {

            if constexpr (Is_SuperScalar<Scalar>::value) {
                VecRef<OutType> fx = fx_.const_cast_derived();
                MatRef<JacType> jx = jx_.const_cast_derived();
                Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
                Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

                typedef typename Scalar::Scalar RealScalar;

                Input<RealScalar> x_r;
                Output<RealScalar> fx_r;
                Jacobian<RealScalar> jx_r;
                Gradient<RealScalar> gx_r;
                Hessian<RealScalar> hx_r;
                Output<RealScalar> l_r;

                const int IRR = this->input_rows();
                const int ORR = this->output_rows();

                if constexpr (Base::InputIsDynamic)
                    x_r.resize(IRR);
                if constexpr (Base::InputIsDynamic)
                    gx_r.resize(IRR);
                if constexpr (Base::InputIsDynamic)
                    hx_r.resize(IRR, IRR);

                if constexpr (Base::OutputIsDynamic)
                    fx_r.resize(ORR);
                if constexpr (Base::OutputIsDynamic)
                    l_r.resize(ORR);

                if constexpr (Base::JacobianIsDynamic)
                    jx_r.resize(ORR, IRR);

                for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                    for (int j = 0; j < IRR; j++)
                        x_r[j] = x[j][i];
                    for (int j = 0; j < ORR; j++)
                        l_r[j] = adjvars[j][i];

                    this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
                        x_r, fx_r, jx_r, gx_r, hx_r, l_r);
                    for (int j = 0; j < ORR; j++)
                        fx[j][i] = fx_r[j];
                    for (int j = 0; j < IRR; j++)
                        adjgrad[j][i] = gx_r[j];

                    for (int j = 0; j < IRR; j++)
                        for (int k = 0; k < ORR; k++)
                            jx(k, j)[i] = jx_r(k, j);

                    for (int j = 0; j < IRR; j++)
                        for (int k = 0; k < IRR; k++)
                            adjhess(k, j)[i] = hx_r(k, j);

                    // See computable_base.h::compute's identical comment: only
                    // the NEXT lane needs these re-zeroed; skip on the final
                    // lane where they are dead.
                    if (i + 1 < Scalar::SizeAtCompileTime) {
                        fx_r.setZero();
                        jx_r.setZero();
                        gx_r.setZero();
                        hx_r.setZero();
                    }
                }

            } else {
                this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
                    x, fx_, jx_, adjgrad_, adjhess_, adjvars);
            }
        } else {
            this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
                x, fx_, jx_, adjgrad_, adjhess_, adjvars);
        }
    }

    /// @brief Compute the adjoint Hessian (value, Jacobian, gradient discarded).
    /// @tparam InType       Eigen type of the input @p x.
    /// @tparam AdjHessType  Eigen matrix type of the adjoint Hessian @p adjhess_.
    /// @tparam AdjVarType   Eigen type of the adjoint vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param adjhess_ Output adjoint Hessian, shape (IR, IR).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    template <class InType, class AdjHessType, class AdjVarType>
    inline void adjointhessian(CVecRef<InType> x, CMatRef<AdjHessType> adjhess_,
                               CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx(this->output_rows());
        Input<Scalar> adjgrad(this->input_rows());
        Jacobian<Scalar> jx(this->output_rows(), this->input_rows());
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();

        this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad,
                                                                        adjhess_, adjvars);
    }

    /// @brief Compute and return the adjoint Hessian by value.
    /// @tparam InType      Eigen type of the input @p x.
    /// @tparam AdjVarType  Eigen type of the adjoint vector @p adjvars.
    /// @param x        Input vector, shape (IR,).
    /// @param adjvars  Adjoint (Lagrange-multiplier) weights, shape (OR,).
    /// @return The adjoint Hessian, shape (IR, IR).
    template <class InType, class AdjVarType>
    inline Hessian<typename InType::Scalar> adjointhessian(CVecRef<InType> x,
                                                           CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Hessian<Scalar> adjhess(this->input_rows(), this->input_rows());
        adjhess.setZero();
        this->derived().adjointhessian(x, adjhess, adjvars);
        return adjhess;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Type-erase this expression into an arbitrary generic wrapper type.
    /// @tparam GenType  The target generic/type-erased function type.
    /// @return A @p GenType wrapping this function.
    template <class GenType> GenType make_generic() const { return GenType(this->derived()); }

    /// @brief Type-erase this expression into a @ref GenericFunction.
    ///
    /// Similar in spirit to Eigen's `.eval()` in that it breaks expression
    /// template chains, but does not evaluate the function -- it wraps it for
    /// deferred evaluation with reduced compile-time cost. Preserves
    /// compile-time input/output sizes. Defined in `OperatorOverloads.h`
    /// (requires a complete @ref GenericFunction).
    ///
    /// @return A `GenericFunction<IR, OR>` wrapping this function.
    GenericFunction<IR, OR> pack() const;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Default sparsity-aware right-multiply by a Jacobian-structured matrix.
    ///
    /// Multiplies an arbitrarily structured matrix @p left on the right by
    /// @p right, where @p right is assumed to share the non-zero structure of
    /// this function's Jacobian, and assigns the product into @p target_ per the
    /// @p assign policy. Routes to the constant-, single-, or dynamic-domain
    /// product implementation in `DenseFunctionOperations.h` depending on whether
    /// the input domain is known at compile time, skipping explicit zero columns.
    /// Derived classes may override this to exploit the exact Jacobian sparsity
    /// pattern.
    ///
    /// @tparam Target      Eigen matrix type of the destination @p target_.
    /// @tparam Left        Eigen type of the left operand @p left.
    /// @tparam Right       Eigen type of the Jacobian-structured right operand @p right.
    /// @tparam Assignment  Assignment policy type.
    /// @tparam Aliased     `true` if any of the operands share memory.
    /// @param target_  Destination matrix (assigned into).
    /// @param left     Left operand.
    /// @param right    Right operand with the Jacobian sparsity structure.
    /// @param assign   Assignment policy (set/add/etc.).
    /// @param aliased  Compile-time aliasing flag.
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {

        // Run Time Input Domain
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {

            if constexpr (Base::InputIsDynamic) {
                right_jacobian_product_dynamic_impl(this->sub_domains, target_, left, right, assign,
                                                    aliased);
            } else {
                right_jacobian_product_impl(target_, left, right, assign, aliased);
            }

        } else {

            // Compile Time Input Domain
            using DMN = typename Derived::INPUT_DOMAIN;
            right_jacobian_product_constant_impl(DMN(), target_, left, right, assign, aliased);
        }
    }

    /// @brief Right-multiply by a matrix carrying this function's input-domain structure.
    ///
    /// Behaves like @ref right_jacobian_product but @p right is assumed to carry
    /// the input-domain (rather than full Jacobian) structure. This is the only
    /// implementation and forwards to @ref right_jacobian_product; derived
    /// classes should **not** override it.
    ///
    /// @tparam Target      Eigen matrix type of the destination @p target_.
    /// @tparam Left        Eigen type of the left operand @p left.
    /// @tparam Right       Eigen type of the domain-structured right operand @p right.
    /// @tparam Assignment  Assignment policy type.
    /// @tparam Aliased     `true` if any of the operands share memory.
    /// @param target_  Destination matrix (assigned into).
    /// @param left     Left operand.
    /// @param right    Right operand with the input-domain structure.
    /// @param assign   Assignment policy.
    /// @param aliased  Compile-time aliasing flag.
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_domain_product(CMatRef<Target> target_, CEigRef<Left> left,
                                              CEigRef<Right> right, Assignment assign,
                                              std::bool_constant<Aliased> aliased) const {
        this->right_jacobian_product(target_, left, right, assign, aliased);
    }

    /// @brief Sparsity-aware symmetric Jacobian product.
    ///
    /// Like @ref right_jacobian_product but for a symmetric (Hessian-structured)
    /// product; routes to the appropriate domain-specialized implementation.
    ///
    /// @tparam Target      Eigen matrix type of the destination @p target_.
    /// @tparam Left        Eigen type of the left operand @p left.
    /// @tparam Right       Eigen type of the right operand @p right.
    /// @tparam Assignment  Assignment policy type.
    /// @tparam Aliased     `true` if any of the operands share memory.
    /// @param target_  Destination matrix (assigned into).
    /// @param left     Left operand.
    /// @param right    Right operand.
    /// @param assign   Assignment policy.
    /// @param aliased  Compile-time aliasing flag.
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                symetric_jacobian_product_dynamic_impl(this->sub_domains, target_, left, right,
                                                       assign, aliased);
            } else {
                symetric_jacobian_product_impl(target_, left, right, assign, aliased);
            }

        } else {
            using DMN = typename Derived::INPUT_DOMAIN;
            symetric_jacobian_product_constant_impl(DMN(), target_, left, right, assign, aliased);
        }
    }

    /// @brief Accumulate a Jacobian-structured matrix into @p target_.
    ///
    /// Adds (per @p assign) a matrix with the same non-zero structure as this
    /// function's Jacobian into @p target_, touching only the domain columns.
    ///
    /// @tparam Target      Eigen matrix type of the destination @p target_.
    /// @tparam JacType     Eigen matrix type of the source @p right.
    /// @tparam Assignment  Assignment policy type.
    /// @param target_  Destination matrix (accumulated into).
    /// @param right    Source matrix with the Jacobian sparsity structure.
    /// @param assign   Assignment policy.
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_matrix_dynamic_domain_impl(this->sub_domains, target_, right, assign);

            } else {
                accumulate_impl(target_, right, assign);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            CMatRef<JacType> right_ref(right.derived());
            MatRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    accumulate_impl(target_ref.template middleCols<size>(start, size),
                                    right_ref.template middleCols<size>(start, size), assign);
                });
        }
    }
    /// @brief Accumulate a gradient-structured vector into @p target_.
    ///
    /// Adds (per @p assign) a vector with the non-zero structure of this
    /// function's gradient into @p target_, touching only the domain entries.
    ///
    /// @tparam Target      Eigen type of the destination @p target_.
    /// @tparam JacType     Eigen type of the source @p right.
    /// @tparam Assignment  Assignment policy type.
    /// @param target_  Destination vector (accumulated into).
    /// @param right    Source vector with the gradient sparsity structure.
    /// @param assign   Assignment policy.
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_vector_dynamic_domain_impl(this->sub_domains, target_, right, assign);
            } else {
                accumulate_impl(target_, right, assign);
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            CMatRef<JacType> right_ref(right.derived());
            MatRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    accumulate_impl(target_ref.template segment<size>(start, size),
                                    right_ref.template segment<size>(start, size), assign);
                });
        }
    }

    /// @brief Accumulate a Hessian-structured matrix into @p target_.
    ///
    /// Adds (per @p assign) a matrix with the same non-zero structure as this
    /// function's Hessian into @p target_, touching only the domain blocks. Does
    /// nothing if the function is known to be linear at compile time.
    ///
    /// @tparam Target      Eigen matrix type of the destination @p target_.
    /// @tparam JacType     Eigen matrix type of the source @p right.
    /// @tparam Assignment  Assignment policy type.
    /// @param target_  Destination matrix (accumulated into).
    /// @param right    Source matrix with the Hessian sparsity structure.
    /// @param assign   Assignment policy.
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        if constexpr (LinearVF<Derived>) {
        } else if constexpr (Base::InputIsDynamic ||
                             std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_symetric_matrix_dynamic_domain_impl(this->sub_domains, target_, right,
                                                               assign);
            } else {
                accumulate_impl(target_, right, assign);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            CMatRef<JacType> right_ref(right.derived());
            MatRef<Target> target_ref(target_.const_cast_derived());

            if constexpr (sds == 1) {
                constexpr int start = Derived::INPUT_DOMAIN::sub_domains[0][0];
                constexpr int size = Derived::INPUT_DOMAIN::sub_domains[0][1];

                accumulate_impl(target_ref.template block<size, size>(start, start, size, size),
                                right_ref.template block<size, size>(start, start, size, size),
                                assign);

            } else if constexpr (sds == 2) {
                constexpr int Start1 = Derived::INPUT_DOMAIN::sub_domains[0][0];
                constexpr int Size1 = Derived::INPUT_DOMAIN::sub_domains[0][1];

                constexpr int Start2 = Derived::INPUT_DOMAIN::sub_domains[1][0];
                constexpr int Size2 = Derived::INPUT_DOMAIN::sub_domains[1][1];

                accumulate_impl(
                    target_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1),
                    right_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1), assign);

                accumulate_impl(
                    target_ref.template block<Size2, Size2>(Start2, Start2, Size2, Size2),
                    right_ref.template block<Size2, Size2>(Start2, Start2, Size2, Size2), assign);

                accumulate_impl(
                    target_ref.template block<Size1, Size2>(Start1, Start2, Size1, Size2),
                    right_ref.template block<Size1, Size2>(Start1, Start2, Size1, Size2), assign);

                accumulate_impl(
                    target_ref.template block<Size2, Size1>(Start2, Start1, Size2, Size1),
                    right_ref.template block<Size2, Size1>(Start2, Start1, Size2, Size1), assign);

            } else {
                tycho::utils::constexpr_for_loop(
                    std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                    [&](auto i) {
                        constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                        constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                        accumulate_impl(target_ref.template middleCols<size>(start, size),
                                        right_ref.template middleCols<size>(start, size), assign);
                    });
            }
        }
    }

    /// @brief Accumulate a matrix carrying this function's input-domain structure.
    ///
    /// Adds (per @p assign) the domain-structured columns of @p right into
    /// @p target_.
    ///
    /// @tparam Target      Eigen matrix type of the destination @p target_.
    /// @tparam JacType     Eigen matrix type of the source @p right.
    /// @tparam Assignment  Assignment policy type.
    /// @param target_  Destination matrix (accumulated into).
    /// @param right    Source matrix with the input-domain structure.
    /// @param assign   Assignment policy.
    template <class Target, class JacType, class Assignment>
    inline void accumulate_matrix_domain(CMatRef<Target> target_, CMatRef<JacType> right,
                                         Assignment assign) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_matrix_dynamic_domain_impl(this->sub_domains, target_, right, assign);
            } else {
                accumulate_impl(target_, right, assign);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            CMatRef<JacType> right_ref(right.derived());
            MatRef<Target> target_ref(target_.const_cast_derived());

            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    accumulate_impl(target_ref.template middleCols<size>(start, size),
                                    right_ref.template middleCols<size>(start, size), assign);
                });
        }
    }

    /// @brief Symmetrize and accumulate a Hessian-product matrix into @p target_.
    ///
    /// Adds `right + right^T` (restricted to the input-domain blocks) into
    /// @p target_, as arises when forming the Hessian of a composed function.
    ///
    /// @tparam Target   Eigen matrix type of the destination @p target_.
    /// @tparam JacType  Eigen matrix type of the source @p right.
    /// @param target_  Destination matrix (accumulated into).
    /// @param right    Source product matrix to symmetrize and add.
    template <class Target, class JacType>
    inline void accumulate_product_hessian(CMatRef<Target> target_, CMatRef<JacType> right) const {
        MatRef<Target> target_ref(target_.const_cast_derived());

        if constexpr (Derived::InputIsDynamic) {
            const int sds = this->sub_domains.cols();
            if (sds == 0) {
                target_ref += right + right.transpose();
            } else {
                for (int i = 0; i < sds; i++) {
                    int start = this->sub_domains(0, i);
                    int size = this->sub_domains(1, i);
                    target_ref.middleCols(start, size) += right.middleCols(start, size);
                    target_ref.middleRows(start, size) += right.middleCols(start, size).transpose();
                }
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    target_ref.template middleCols<size>(start, size) +=
                        right.template middleCols<size>(start, size);
                    target_ref.template middleRows<size>(start, size) +=
                        right.template middleCols<size>(start, size).transpose();
                });
        }
    }

    /// @brief In-place scale the whole @p target_ by scalar @p s.
    /// @internal
    /// @tparam Target  Eigen matrix type of @p target_.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param target_  Matrix to scale in place.
    /// @param s        Scalar multiplier.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_impl(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        target *= s;
    }

    /// @brief Scale only the Jacobian-domain columns of @p target_ by @p s.
    /// @tparam Target  Eigen matrix type of @p target_.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param target_  Matrix to scale in place (domain columns only).
    /// @param s        Scalar multiplier.
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                scale_matrix_dynamic_domain_impl(this->sub_domains, target_, s);

            } else {
                this->scale_impl(target_, s);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            MatRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    this->scale_impl(target_ref.template middleCols<size>(start, size), s);
                });
        }
    }
    /// @brief Scale only the gradient-domain entries of @p target_ by @p s.
    /// @tparam Target  Eigen type of @p target_.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param target_  Vector to scale in place (domain entries only).
    /// @param s        Scalar multiplier.
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                scale_vector_dynamic_domain_impl(this->sub_domains, target_, s);

            } else {
                this->scale_impl(target_, s);
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            MatRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    this->scale_impl(target_ref.template segment<size>(start, size), s);
                });
        }
    }
    /// @brief Scale only the Hessian-domain blocks of @p target_ by @p s.
    ///
    /// No-op if the function is known to be linear at compile time.
    ///
    /// @tparam Target  Eigen matrix type of @p target_.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param target_  Matrix to scale in place (domain blocks only).
    /// @param s        Scalar multiplier.
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        if constexpr (LinearVF<Derived>) {
        } else if constexpr (Base::InputIsDynamic ||
                             std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                scale_matrix_dynamic_domain_impl(this->sub_domains, target_, s);

            } else {
                this->scale_impl(target_, s);
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            MatRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    this->scale_impl(target_ref.template middleCols<size>(start, size), s);
                });
        }
    }

    /// @brief Zero only the input-domain columns of @p target_.
    /// @tparam Target  Eigen matrix type of @p target_.
    /// @param target_  Matrix whose domain columns are set to zero.
    template <class Target> inline void zero_matrix_domain(CMatRef<Target> target_) const {

        MatRef<Target> target_ref(target_.const_cast_derived());

        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                const int sds = this->sub_domains.cols();
                if (sds == 0) {
                    target_ref.setZero();
                } else {
                    for (int i = 0; i < sds; i++) {
                        int start = this->sub_domains(0, i);
                        int size = this->sub_domains(1, i);
                        target_ref.middleCols(start, size).setZero();
                    }
                }
            } else {
                target_ref.setZero();
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    target_ref.template middleCols<size>(start, size).setZero();
                });
        }
    }

    /// @brief Count the non-zero KKT elements contributed per function application.
    ///
    /// Sums the non-zeros in the lower triangle of the Hessian and in the
    /// Jacobian that will be scattered into the optimization problem's KKT matrix
    /// on each call. Dense functions normally assume every element is non-zero,
    /// but explicit non-zeros are honored when the derived class overrides
    /// `jacobian_elem_is_nonzero` / `hessian_elem_is_nonzero`. Used by the
    /// `ConstraintFunction` / `ObjectiveFunction` optimizer wrappers.
    ///
    /// @param dojac   Include Jacobian elements in the count.
    /// @param dohess  Include Hessian elements in the count.
    /// @return Total number of KKT elements scattered per application.
    int num_kkt_elements(bool dojac, bool dohess) const {
        int hesselems = 0;
        int jacelems = 0;
        for (int i = 0; i < this->input_rows(); i++) {
            if (dohess) {
                for (int j = i; j < this->input_rows(); j++) {
                    if (this->derived().hessian_elem_is_nonzero(j, i))
                        hesselems++;
                }
            }
            if (dojac) {
                for (int j = 0; j < this->output_rows(); j++) {
                    if (this->derived().jacobian_elem_is_nonzero(j, i))
                        jacelems++;
                }
            }
        }
        return ((hesselems + jacelems));
    }

    /// @brief Emit the (row, col) KKT-matrix coordinates for this function's elements.
    ///
    /// Fills @p KKTrows / @p KKTcols with the target locations of this function's
    /// Jacobian and Hessian elements within the optimization problem's KKT matrix
    /// when used as a constraint, advancing @p freeloc as it goes. Variable and
    /// constraint-row locations for each application come from @p data.
    ///
    /// @param KKTrows    Output array of KKT row indices (written into).
    /// @param KKTcols    Output array of KKT column indices (written into).
    /// @param freeloc    Running write cursor into the KKT arrays (advanced).
    /// @param conoffset  Row offset applied to constraint (Jacobian) rows.
    /// @param dojac      Emit Jacobian element locations.
    /// @param dohess     Emit Hessian element locations.
    /// @param data       Solver indexing metadata (also stores per-application starts).
    void get_kkt_space(EigenRef<Eigen::VectorXi> KKTrows, EigenRef<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       tycho::solvers::SolverIndexingData &data) {
        data.inner_kkt_starts_.resize(data.num_appl());

        // Indices come from the OUTPUT map (data.v_scatter_loc), which equals the
        // input map unless the solver has eliminated variables. An element that
        // touches an eliminated variable is recorded as (-1, -1): it keeps its
        // claim -- the claims stay contiguous and the scatter's cursor walks them
        // in lockstep exactly as before -- but it names no matrix entry, so
        // NonLinearProgram::analyze_sparsity leaves it out of the pattern and
        // out of the clash bookkeeping, and the scatter never writes it. That is
        // what makes the eliminated variable's row and column simply not exist
        // in the assembled matrix.
        for (int V = 0; V < data.num_appl(); V++) {
            data.inner_kkt_starts_[V] = freeloc;
            for (int i = 0; i < this->input_rows(); i++) {
                const int col = data.v_scatter_loc(i, V);
                if (dohess) {
                    for (int j = i; j < this->input_rows(); j++) {
                        if (this->derived().hessian_elem_is_nonzero(j, i)) {
                            const int row = data.v_scatter_loc(j, V);
                            const bool eliminated = (row < 0) || (col < 0);
                            KKTrows[freeloc] = eliminated ? -1 : row;
                            KKTcols[freeloc] = eliminated ? -1 : col;
                            freeloc++;
                        }
                    }
                }
                if (dojac) {
                    for (int j = 0; j < this->output_rows(); j++) {
                        if (this->derived().jacobian_elem_is_nonzero(j, i)) {
                            KKTrows[freeloc] = (col < 0) ? -1 : data.c_loc(j, V) + conoffset;
                            KKTcols[freeloc] = col;
                            freeloc++;
                        }
                    }
                }
            }
        }
    }

    ////
    /// @brief PSIOPT interface: evaluate constraint values and scatter their Jacobian.
    ///
    /// Called during the solve loop. For each application, evaluates `f(x)` and
    /// `J(x)`, writes values into @p FX, and sums the Jacobian into the global
    /// KKT matrix @p KKTmat at the precomputed @p KKTLocations. @p KKTClashes /
    /// @p KKTLocks coordinate concurrent column writes across threads (a clash
    /// flag of `-1` means uncontested; otherwise it indexes a mutex to lock).
    /// Supports SuperScalar packed evaluation when enabled.
    ///
    /// @param X            Full optimization variable vector.
    /// @param FX           Full constraint vector (written into).
    /// @param KKTmat       Global sparse KKT matrix (accumulated into).
    /// @param KKTLocations Value-array offsets for each scattered element.
    /// @param KKTClashes   Per-variable clash flags (or mutex indices).
    /// @param KKTLocks     Mutexes guarding contested KKT columns.
    /// @param data         Solver indexing metadata for each application.
    /// @note Do not override unless you fully understand the KKT-fill protocol.
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &jx) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());

            auto ScalarImpl = [&](int start, int stop) {
                for (int V = start; V < stop; V++) {
                    this->gather_input(X, x, V, data);

                    new (&fx) Eigen::Map<Output<double>>(
                        FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                    fx.setZero();
                    jx.setZero();
                    this->derived().compute_jacobian(x, fx, jx);

                    this->derived().kkt_fill_jac(V, jx, KKTmat, KKTLocations, KKTClashes, KKTLocks,
                                                 data);
                }
            };
            const int IRR = this->input_rows();
            const int ORR = this->output_rows();
            auto VectorImpl = [&]() {
                using SuperScalar = tycho::DefaultSuperScalar;
                constexpr int vsize = SuperScalar::SizeAtCompileTime;
                int Packs = data.num_appl() / vsize;

                Input<SuperScalar> x_vect(this->input_rows());
                Output<SuperScalar> fx_vect(this->output_rows());
                Jacobian<SuperScalar> jx_vect(this->output_rows(), this->input_rows());

                for (int i = 0; i < Packs; i++) {
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        this->gather_input(X, x, V, data);
                        for (int k = 0; k < IRR; k++) {
                            x_vect[k][j] = x[k];
                        }
                    }
                    fx_vect.setZero();
                    jx_vect.setZero();
                    this->derived().compute_jacobian(x_vect, fx_vect, jx_vect);

                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        for (int k = 0; k < IRR; k++) {
                            for (int l = 0; l < ORR; l++) {
                                jx(l, k) = jx_vect(l, k)[j];
                            }
                        }
                        this->derived().kkt_fill_jac(V, jx, KKTmat, KKTLocations, KKTClashes,
                                                     KKTLocks, data);
                    }
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        new (&fx) Eigen::Map<Output<double>>(
                            FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                        for (int l = 0; l < ORR; l++) {
                            fx[l] = fx_vect[l][j];
                        }
                    }
                }

                ScalarImpl(Packs * vsize, data.num_appl());
            };

            if constexpr (Vectorizable<Derived>) {
                if (this->derived().enable_vectorization_) {
                    VectorImpl();
                } else {
                    ScalarImpl(0, data.num_appl());
                }
            } else {
                ScalarImpl(0, data.num_appl());
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Jacobian<double>>(this->output_rows(), this->input_rows()));
    }

    /// @brief PSIOPT interface: constraint values, adjoint gradients, and Jacobian scatter.
    ///
    /// Like @ref constraints_jacobian, but also gathers the multipliers from
    /// @p L and writes the adjoint gradient into @p AGX for each application.
    ///
    /// @param X            Full optimization variable vector.
    /// @param L            Full Lagrange-multiplier vector.
    /// @param FX           Full constraint vector (written into).
    /// @param AGX          Full adjoint-gradient vector (written into).
    /// @param KKTmat       Global sparse KKT matrix (accumulated into).
    /// @param KKTLocations Value-array offsets for each scattered element.
    /// @param KKTClashes   Per-variable clash flags (or mutex indices).
    /// @param KKTLocks     Mutexes guarding contested KKT columns.
    /// @param data         Solver indexing metadata for each application.
    void constraints_jacobian_adjointgradient(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &l, auto &jx) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());
            Eigen::Map<Input<double>> agx(NULL, this->input_rows());

            for (int V = 0; V < data.num_appl(); V++) {
                this->gather_input(X, x, V, data);
                this->gather_mult(L, l, V, data);

                new (&fx) Eigen::Map<Output<double>>(FX.data() + data.inner_constraint_starts_[V],
                                                     this->output_rows());
                new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.inner_gradient_starts_[V],
                                                     this->input_rows());

                fx.setZero();
                agx.setZero();
                jx.setZero();

                this->derived().compute_jacobian_adjointgradient(x, fx, jx, agx, l);

                this->derived().kkt_fill_jac(V, jx, KKTmat, KKTLocations, KKTClashes, KKTLocks,
                                             data);
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Output<double>>(this->output_rows(), 1),
            tycho::utils::TempSpec<Jacobian<double>>(this->output_rows(), this->input_rows()));
    }

    /// @brief PSIOPT interface: values, Jacobian, adjoint gradient, and Hessian scatter.
    ///
    /// Called during the optimize loop. For each application, evaluates `f(x)`,
    /// `J(x)`, the adjoint gradient, and the adjoint Hessian, then scatters both
    /// the Jacobian and the Hessian into the global KKT matrix @p KKTmat using
    /// the same threaded clash/lock protocol as @ref constraints_jacobian.
    /// Supports SuperScalar packed evaluation when enabled.
    ///
    /// @param X            Full optimization variable vector.
    /// @param L            Full Lagrange-multiplier vector.
    /// @param FX           Full constraint vector (written into).
    /// @param AGX          Full adjoint-gradient vector (written into).
    /// @param KKTmat       Global sparse KKT matrix (accumulated into).
    /// @param KKTLocations Value-array offsets for each scattered element.
    /// @param KKTClashes   Per-variable clash flags (or mutex indices).
    /// @param KKTLocks     Mutexes guarding contested KKT columns.
    /// @param data         Solver indexing metadata for each application.
    /// @note Do not override unless you fully understand the KKT-fill protocol.
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &l, auto &jx, auto &hx) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());
            Eigen::Map<Input<double>> agx(NULL, this->input_rows());

            auto ScalarImpl = [&](int start, int stop) {
                for (int V = start; V < stop; V++) {
                    this->gather_input(X, x, V, data);
                    this->gather_mult(L, l, V, data);

                    new (&fx) Eigen::Map<Output<double>>(
                        FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                    new (&agx) Eigen::Map<Input<double>>(
                        AGX.data() + data.inner_gradient_starts_[V], this->input_rows());

                    fx.setZero();
                    agx.setZero();
                    jx.setZero();
                    hx.setZero();

                    this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, agx,
                                                                                    hx, l);

                    this->derived().kkt_fill_all(V, jx, hx, KKTmat, KKTLocations, KKTClashes,
                                                 KKTLocks, data);
                }
            };

            const int IRR = this->input_rows();
            const int ORR = this->output_rows();
            auto VectorImpl = [&]() {
                using SuperScalar = tycho::DefaultSuperScalar;
                constexpr int vsize = SuperScalar::SizeAtCompileTime;
                int Packs = data.num_appl() / vsize;

                Input<SuperScalar> x_vect(this->input_rows());
                Output<SuperScalar> fx_vect(this->output_rows());
                Jacobian<SuperScalar> jx_vect(this->output_rows(), this->input_rows());
                Gradient<SuperScalar> agx_vect(this->input_rows());
                Hessian<SuperScalar> hx_vect(this->input_rows(), this->input_rows());
                Output<SuperScalar> l_vect(this->output_rows());

                for (int i = 0; i < Packs; i++) {
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        this->gather_input(X, x, V, data);
                        this->gather_mult(L, l, V, data);

                        for (int k = 0; k < IRR; k++) {
                            x_vect[k][j] = x[k];
                        }
                        for (int k = 0; k < ORR; k++) {
                            l_vect[k][j] = l[k];
                        }
                    }
                    fx_vect.setZero();
                    jx_vect.setZero();
                    hx_vect.setZero();
                    agx_vect.setZero();

                    this->derived().compute_jacobian_adjointgradient_adjointhessian(
                        x_vect, fx_vect, jx_vect, agx_vect, hx_vect, l_vect);

                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        for (int k = 0; k < IRR; k++) {
                            for (int l = 0; l < ORR; l++) {
                                jx(l, k) = jx_vect(l, k)[j];
                            }
                        }
                        for (int k = 0; k < IRR; k++) {
                            for (int l = k; l < IRR; l++) {
                                hx(l, k) = hx_vect(l, k)[j];
                            }
                        }
                        this->derived().kkt_fill_all(V, jx, hx, KKTmat, KKTLocations, KKTClashes,
                                                     KKTLocks, data);
                    }
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        new (&fx) Eigen::Map<Output<double>>(
                            FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                        for (int l = 0; l < ORR; l++) {
                            fx[l] = fx_vect[l][j];
                        }
                        new (&agx) Eigen::Map<Input<double>>(
                            AGX.data() + data.inner_gradient_starts_[V], this->input_rows());
                        for (int l = 0; l < IRR; l++) {
                            agx[l] = agx_vect[l][j];
                        }
                    }
                }

                ScalarImpl(Packs * vsize, data.num_appl());
            };

            if constexpr (Vectorizable<Derived>) {
                if (this->derived().enable_vectorization_) {
                    VectorImpl();
                } else {
                    ScalarImpl(0, data.num_appl());
                }
            } else {
                ScalarImpl(0, data.num_appl());
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Output<double>>(this->output_rows(), 1),
            tycho::utils::TempSpec<Jacobian<double>>(this->output_rows(), this->input_rows()),
            tycho::utils::TempSpec<Hessian<double>>(this->input_rows(), this->input_rows()));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // ---- Scalar objective interface (OR == 1 only) ----
    /// @brief PSIOPT interface: accumulate this scalar function's objective value.
    ///
    /// Only available when `OR == 1`. Sums `ObjScale * f(x)` over all
    /// applications into @p Val.
    ///
    /// @param ObjScale  Scale factor applied to the objective contribution.
    /// @param X         Full optimization variable vector.
    /// @param Val       Running objective accumulator (added into).
    /// @param data      Solver indexing metadata for each application.
    void objective(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                   const tycho::solvers::SolverIndexingData &data) const
        requires(OR == 1)
    {
        Input<double> x(this->input_rows());
        Output<double> fx(1);

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            fx.setZero();
            this->derived().compute(x, fx);
            Val += fx[0] * ObjScale;
        }
    }

    /// @brief PSIOPT interface: accumulate the objective value and its gradient.
    ///
    /// Only available when `OR == 1`. Sums `ObjScale * f(x)` into @p Val and
    /// scatters `ObjScale * J(x)^T` into the gradient vector @p GX.
    ///
    /// @param ObjScale  Scale factor applied to the objective and gradient.
    /// @param X         Full optimization variable vector.
    /// @param Val       Running objective accumulator (added into).
    /// @param GX        Full gradient vector (written into).
    /// @param data      Solver indexing metadata for each application.
    void objective_gradient(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                            EigenRef<Eigen::VectorXd> GX,
                            const tycho::solvers::SolverIndexingData &data) const
        requires(OR == 1)
    {
        Input<double> x(this->input_rows());
        Output<double> fx(1);
        Jacobian<double> jx(1, this->input_rows());
        Eigen::Map<Input<double>> gx(NULL, this->input_rows());

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            new (&gx) Eigen::Map<Input<double>>(GX.data() + data.inner_gradient_starts_[V],
                                                this->input_rows());
            fx.setZero();
            jx.setZero();
            gx.setZero();

            this->derived().compute_jacobian(x, fx, jx);
            Val += fx[0] * ObjScale;
            gx = jx.transpose() * ObjScale;
        }
    }

    /// @brief PSIOPT interface: accumulate objective value, gradient, and Hessian.
    ///
    /// Only available when `OR == 1`. Sums `ObjScale * f(x)` into @p Val,
    /// scatters the gradient into @p GX, and scatters the (objective-scaled)
    /// Hessian into the global KKT matrix @p KKTmat.
    ///
    /// @param ObjScale     Scale factor applied to the objective derivatives.
    /// @param X            Full optimization variable vector.
    /// @param Val          Running objective accumulator (added into).
    /// @param GX           Full gradient vector (written into).
    /// @param KKTmat       Global sparse KKT matrix (accumulated into).
    /// @param KKTLocations Value-array offsets for each scattered element.
    /// @param KKTClashes   Per-variable clash flags (or mutex indices).
    /// @param KKTLocks     Mutexes guarding contested KKT columns.
    /// @param data         Solver indexing metadata for each application.
    void objective_gradient_hessian(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                                    EigenRef<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    EigenRef<Eigen::VectorXi> KKTLocations,
                                    EigenRef<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const tycho::solvers::SolverIndexingData &data) const
        requires(OR == 1)
    {
        Input<double> x(this->input_rows());
        Output<double> fx(1);
        Jacobian<double> jx(1, this->input_rows());
        Eigen::Map<Input<double>> gx(NULL, this->input_rows());
        Hessian<double> hx(this->input_rows(), this->input_rows());
        Output<double> lm(1);
        lm[0] = ObjScale;

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            new (&gx) Eigen::Map<Input<double>>(GX.data() + data.inner_gradient_starts_[V],
                                                this->input_rows());

            fx.setZero();
            jx.setZero();
            gx.setZero();
            hx.setZero();

            this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

            Val += fx[0] * ObjScale;
            this->kkt_fill_hess(V, hx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
    }

  protected:
    /// @brief Scatter a scalar function's Hessian into the KKT matrix (objective path).
    /// @internal
    /// Accumulates the lower triangle of @p hx into the KKT value array under the
    /// canonical-column lock protocol: each element (j, i) locks the mutex of
    /// `tycho::solvers::kkt_canonical_lock_col(v_loc(j), v_loc(i))` -- the same
    /// keying used by `kkt_fill_all` and by `get_mat_space`'s clash detection --
    /// via a single-held-lock hand-off (see the comment in `kkt_fill_all`).
    /// Objective and constraint scatters for different partitions run concurrently
    /// into the same KKT matrix (`eval_kkt`), so an objective Hessian term and a
    /// constraint Hessian term coupling the same primal pair must resolve to the
    /// same mutex; the shared keying makes that structural. `OR == 1` only.
    ///
    /// @param Apl         Application index.
    /// @param hx          Local Hessian for this application.
    /// @param KKTmat      Global sparse KKT matrix (accumulated into).
    /// @param KKTLocs     Value-array offsets for each scattered element.
    /// @param VarClashes  Per-variable clash flags (or mutex indices).
    /// @param ClashLocks  Mutexes guarding contested columns.
    /// @param data        Solver indexing metadata.
    /// @endinternal
    void kkt_fill_hess(int Apl, const Hessian<double> &hx,
                       Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                       EigenRef<Eigen::VectorXi> KKTLocs, EigenRef<Eigen::VectorXi> VarClashes,
                       std::vector<std::mutex> &ClashLocks,
                       const tycho::solvers::SolverIndexingData &data) const
        requires(OR == 1)
    {
        int freeloc = data.inner_kkt_starts_[Apl];
        double *mpt = KKTmat.valuePtr();
        const int *lpt = KKTLocs.data();
        int ActiveVar;

        auto Lock = [&](int var) {
            if (VarClashes[var] == -1) {
            } else {
                ClashLocks[VarClashes[var]].lock();
            }
        };
        auto UnLock = [&](int var) {
            if (VarClashes[var] == -1) {
            } else {
                ClashLocks[VarClashes[var]].unlock();
            }
        };
        // Same single-held-lock hand-off as kkt_fill_all: release before acquire, at most
        // one lock held at any instant (no hold-and-wait -> deadlock-free).
        auto Relock = [&](int cur, int want) {
            if (want != cur) {
                if (cur != -1)
                    UnLock(cur);
                Lock(want);
            }
            return want;
        };

        const int IRR = (IR > 0) ? IR : this->input_rows();

        // Reduced variant hoisted out of the loop -- see the argument in
        // kkt_fill_all. Without bound-fixed variables the loop below is the
        // original one and costs one predicated branch per application.
        if (!data.v_out_reduced_) {
            for (int i = 0; i < IRR; i++) {
                ActiveVar = data.v_loc(i, Apl);
                int held = -1;
                if constexpr (!LinearVF<Derived>) {
                    for (int j = i; j < IRR; j++) {
                        const int lockcol =
                            tycho::solvers::kkt_canonical_lock_col(data.v_loc(j, Apl), ActiveVar);
                        held = Relock(held, lockcol);
                        this->derived().add_hessian_elem(hx(j, i), j, i, mpt, lpt, freeloc);
                    }
                }
                if (held != -1)
                    UnLock(held);
            }
            return;
        }

        for (int i = 0; i < IRR; i++) {
            const int active_out = data.v_out_loc(i, Apl);
            int held = -1;
            if constexpr (!LinearVF<Derived>) {
                for (int j = i; j < IRR; j++) {
                    const int row_out = data.v_out_loc(j, Apl);
                    if (row_out < 0 || active_out < 0) {
                        if (this->derived().hessian_elem_is_nonzero(j, i))
                            freeloc++;
                        continue;
                    }
                    const int lockcol = tycho::solvers::kkt_canonical_lock_col(row_out, active_out);
                    held = Relock(held, lockcol);
                    this->derived().add_hessian_elem(hx(j, i), j, i, mpt, lpt, freeloc);
                }
            }
            if (held != -1)
                UnLock(held);
        }
    }

    /// @brief Whether Jacobian element (@p row, @p col) is structurally non-zero.
    /// @internal
    /// @param row  Output (constraint) row.
    /// @param col  Input (variable) column.
    /// @return `true` by default; derived classes may override for sparsity.
    /// @endinternal
    inline constexpr bool jacobian_elem_is_nonzero(int row, int col) const { return true; }
    /// @brief Whether Hessian element (@p row, @p col) is structurally non-zero.
    /// @internal
    /// @param row  Hessian row.
    /// @param col  Hessian column.
    /// @return `false` if the function is linear, else `true` by default.
    /// @endinternal
    inline constexpr bool hessian_elem_is_nonzero(int row, int col) const {
        return !LinearVF<Derived>;
    }
    /// @brief Add one Hessian value into the KKT value array (no-op if linear).
    /// @internal
    /// @param v        Value to add.
    /// @param row      Hessian row (unused; documentary).
    /// @param col      Hessian column (unused; documentary).
    /// @param mpt      KKT value array base pointer.
    /// @param lpt      KKT location array base pointer.
    /// @param freeloc  Running write cursor (advanced).
    /// @endinternal
    inline void add_hessian_elem(double v, int row, int col, double *mpt, const int *lpt,
                                 int &freeloc) const {
        if constexpr (!LinearVF<Derived>) {
            mpt[lpt[freeloc]] += v;
            freeloc++;
        }
    }
    /// @brief Add one Jacobian value into the KKT value array.
    /// @internal
    /// @param v        Value to add.
    /// @param row      Jacobian row (unused; documentary).
    /// @param col      Jacobian column (unused; documentary).
    /// @param mpt      KKT value array base pointer.
    /// @param lpt      KKT location array base pointer.
    /// @param freeloc  Running write cursor (advanced).
    /// @endinternal
    inline void add_jacobian_elem(double v, int row, int col, double *mpt, const int *lpt,
                                  int &freeloc) const {
        mpt[lpt[freeloc]] += v;
        freeloc++;
    }

    /// @brief Scatter both Jacobian and Hessian of one application into the KKT matrix.
    /// @internal
    /// Per input column, symmetrically inserts the Hessian column then the
    /// Jacobian column under the canonical-column lock protocol (each Hessian
    /// element keyed on `tycho::solvers::kkt_canonical_lock_col`; see the comment
    /// in the body); the Jacobian column is unlocked when the constraint rows are
    /// unique to this application.
    ///
    /// @tparam JacType   Eigen matrix type of the local Jacobian @p jx.
    /// @tparam HessType  Eigen matrix type of the local Hessian @p hx.
    /// @param Apl         Application index.
    /// @param jx          Local Jacobian for this application.
    /// @param hx          Local Hessian for this application.
    /// @param KKTmat      Global sparse KKT matrix (accumulated into).
    /// @param KKTLocs     Value-array offsets for each scattered element.
    /// @param VarClashes  Per-variable clash flags (or mutex indices).
    /// @param ClashLocks  Mutexes guarding contested columns.
    /// @param data        Solver indexing metadata.
    /// @endinternal
    template <class JacType, class HessType>
    void kkt_fill_all(int Apl, const JacType &jx, const HessType &hx,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                      EigenRef<Eigen::VectorXi> KKTLocs, EigenRef<Eigen::VectorXi> VarClashes,
                      std::vector<std::mutex> &ClashLocks,
                      const tycho::solvers::SolverIndexingData &data) const {
        int freeloc = data.inner_kkt_starts_[Apl];
        double *mpt = KKTmat.valuePtr();
        const int *lpt = KKTLocs.data();
        int ActiveVar;
        const int IRR = (Base::IRC > 0) ? Base::IRC : this->input_rows();
        const int ORR = (Base::ORC > 0) ? Base::ORC : this->output_rows();

        // Canonical-column lock protocol (load-bearing; see kkt_canonical_lock_col in
        // indexing_data.h and NonLinearProgram::get_mat_space). A Hessian element couples the
        // two *global* variables (v_loc(j), v_loc(i)); the sparsity routine stores the
        // physical KKT slot it writes under the SMALLER of the two endpoints, min(row, col)
        // (analyze_sparsity keeps the lower triangle: col <= row, value offset
        // outerIndexPtr()[col]). Locking the outer variable v_loc(i) alone -- as this scatter
        // historically did -- is unsafe: when v_loc(j) < v_loc(i) the symmetric partner in
        // another partition iterates the pair in the opposite local order and would lock the
        // *other* endpoint, so the two claimants of one physical double take different mutexes
        // and their non-atomic += race. The fix: every element locks its slot's canonical
        // column via the ONE shared keying function tycho::solvers::kkt_canonical_lock_col,
        // used identically by kkt_fill_hess (the objective Hessian scatter, which runs
        // concurrently with this one in eval_kkt) and by get_mat_space's clash detection --
        // so all claimants of a slot agree on the mutex BY CONSTRUCTION; there is no per-site
        // convention that can drift. For the Jacobian block the row is a constraint row,
        // which sorts above every variable column, so its canonical column is always
        // ActiveVar == v_loc(i) -- unchanged from before.
        //
        // Writes within one partition need no mutual exclusion (each partition's scatter runs
        // serially on one thread; single-partition problems take no locks at all -- `Lock` is
        // a no-op for uncontested columns), so two elements of the SAME partition may still
        // touch a slot under different canonical columns; that is serialized by the partition
        // thread itself. At most ONE lock is held at any instant (a mirror element hands the
        // lock from ActiveVar over to the smaller column and back), so the protocol is
        // deadlock-free. Consecutive writes that share a canonical column reuse the held lock,
        // so the common already-ordered case still costs exactly one lock per column.
        auto Lock = [&](int var) {
            if (VarClashes[var] == -1) {
                //// uncontested
            } else {
                /// contested lock mutex
                ClashLocks[VarClashes[var]].lock();
            }
        };
        auto UnLock = [&](int var) {
            if (VarClashes[var] == -1) {
                //// uncontested
            } else {
                /// contested unlock mutex
                ClashLocks[VarClashes[var]].unlock();
            }
        };

        const bool uniquecon = data.unique_constraints_;

        // Hand the single held lock from column `cur` to column `want`, coalescing runs of
        // writes that share a canonical column. Returns the now-held column. `cur == -1`
        // means no lock is held; the held lock is released BEFORE the next is acquired, so a
        // thread never holds two locks at once (no hold-and-wait -> deadlock-free).
        auto Relock = [&](int cur, int want) {
            if (want != cur) {
                if (cur != -1)
                    UnLock(cur);
                Lock(want);
            }
            return want;
        };

        // Fixed-variable elimination: hoisted out of the loops on purpose. A
        // function only carries a solver-space output map when the solver has
        // eliminated bound-fixed variables, so on every problem without them the
        // loop below is the original one, instruction for instruction, and the
        // only cost of the feature existing is this one predicated branch per
        // application. The reduced variant is a separate loop nest further down,
        // so nothing on the common path pays a per-element test.
        if (!data.v_out_reduced_) {
            for (int i = 0; i < IRR; i++) {
                ActiveVar = data.v_loc(i, Apl);
                int held = -1;
                ///// insert hessian column symetrically -- each element under its canonical
                ///// (min-endpoint) lock column min(v_loc(j), v_loc(i))
                if constexpr (!LinearVF<Derived>) {
                    for (int j = i; j < IRR; j++) {
                        const int lockcol =
                            tycho::solvers::kkt_canonical_lock_col(data.v_loc(j, Apl), ActiveVar);
                        held = Relock(held, lockcol);
                        this->derived().add_hessian_elem(hx(j, i), j, i, mpt, lpt, freeloc);
                    }
                }
                ///// insert jacobian column -- canonical lock column is always ActiveVar
                ///// (constraint rows sort above every variable column); unlocked when the
                ///// constraint rows are unique to this application (no cross-partition clash)
                if (uniquecon) {
                    if (held != -1)
                        UnLock(held);
                    held = -1;
                } else {
                    held = Relock(held, ActiveVar);
                }
                for (int j = 0; j < ORR; j++) {
                    this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
                }
                ///////////////////////////////////////////////////////////////////////////////
                if (held != -1)
                    UnLock(held);
            }
            return;
        }

        // Reduced variant. Two differences from the loop above, and no others:
        // an element whose row or column names an eliminated variable is stepped
        // over instead of written, and every index -- lock key included -- reads
        // the solver-space output map rather than the problem-space input map.
        // The second is required for the same structural reason the original
        // keying is: every claimant of one physical slot must derive the same
        // lock column, and once the space is renumbered the slot is determined
        // by the SOLVER-space pair. The clash table is sized and marked in that
        // same space, so the two agree by construction.
        //
        // Stepping over an element must advance the cursor by exactly what
        // add_*_elem would have. That is expressible without a new hook because
        // the two agree by construction: add_*_elem advances iff
        // *_elem_is_nonzero(row, col) -- for this base (where the predicate is
        // !LinearVF and the add is guarded by the same trait) and for the
        // sparsity-aware overrides in the transcription defects (where both test
        // the same nz_locs_ entry). It is also the same predicate get_kkt_space
        // claims by, and the same idiom kkt_fill_jac already uses to step over
        // the Hessian block.
        for (int i = 0; i < IRR; i++) {
            const int active_out = data.v_out_loc(i, Apl);
            int held = -1;
            if constexpr (!LinearVF<Derived>) {
                for (int j = i; j < IRR; j++) {
                    const int row_out = data.v_out_loc(j, Apl);
                    if (row_out < 0 || active_out < 0) {
                        if (this->derived().hessian_elem_is_nonzero(j, i))
                            freeloc++;
                        continue;
                    }
                    const int lockcol = tycho::solvers::kkt_canonical_lock_col(row_out, active_out);
                    held = Relock(held, lockcol);
                    this->derived().add_hessian_elem(hx(j, i), j, i, mpt, lpt, freeloc);
                }
            }
            if (active_out < 0) {
                // The whole Jacobian column belongs to an eliminated variable.
                if (held != -1)
                    UnLock(held);
                for (int j = 0; j < ORR; j++) {
                    if (this->derived().jacobian_elem_is_nonzero(j, i))
                        freeloc++;
                }
                continue;
            }
            if (uniquecon) {
                if (held != -1)
                    UnLock(held);
                held = -1;
            } else {
                held = Relock(held, active_out);
            }
            for (int j = 0; j < ORR; j++) {
                this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
            }
            if (held != -1)
                UnLock(held);
        }
    }

    /// @brief Scatter only the Jacobian of one application into the KKT matrix.
    /// @internal
    /// Skips over the Hessian element slots (advancing @p freeloc) and inserts
    /// the Jacobian column, applying the per-variable lock protocol when the
    /// constraints are not unique.
    ///
    /// @tparam JacType  Eigen matrix type of the local Jacobian @p jx.
    /// @param Apl         Application index.
    /// @param jx          Local Jacobian for this application.
    /// @param KKTmat      Global sparse KKT matrix (accumulated into).
    /// @param KKTLocs     Value-array offsets for each scattered element.
    /// @param VarClashes  Per-variable clash flags (or mutex indices).
    /// @param ClashLocks  Mutexes guarding contested columns.
    /// @param data        Solver indexing metadata.
    /// @endinternal
    template <class JacType>
    void kkt_fill_jac(int Apl, const JacType &jx,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                      Eigen::Ref<Eigen::VectorXi> KKTLocs, Eigen::Ref<Eigen::VectorXi> VarClashes,
                      std::vector<std::mutex> &ClashLocks,
                      const tycho::solvers::SolverIndexingData &data) const {
        int freeloc = data.inner_kkt_starts_[Apl];
        double *mpt = KKTmat.valuePtr();
        const int *lpt = KKTLocs.data();
        int ActiveVar;

        // Reduced variant, hoisted for the same reason as in kkt_fill_all: on a
        // problem with no bound-fixed variables the loops below are the original
        // ones. Simpler here than there -- this routine only ever steps over the
        // Hessian block, so elimination matters for the Jacobian column alone,
        // and a column whose variable was eliminated is stepped over whole (one
        // test per column, not per element).
        if (data.v_out_reduced_) {
            for (int i = 0; i < this->input_rows(); i++) {
                const int active_out = data.v_out_loc(i, Apl);
                for (int j = i; j < this->input_rows(); j++) {
                    if (this->derived().hessian_elem_is_nonzero(j, i))
                        freeloc++;
                }
                if (active_out < 0) {
                    for (int j = 0; j < this->output_rows(); j++) {
                        if (this->derived().jacobian_elem_is_nonzero(j, i))
                            freeloc++;
                    }
                    continue;
                }
                const bool lock_column =
                    !data.unique_constraints_ && (VarClashes[active_out] != -1);
                if (lock_column)
                    ClashLocks[VarClashes[active_out]].lock();
                for (int j = 0; j < this->output_rows(); j++) {
                    this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
                }
                if (lock_column)
                    ClashLocks[VarClashes[active_out]].unlock();
            }
            return;
        }

        if (data.unique_constraints_) {
            for (int i = 0; i < this->input_rows(); i++) {
                ActiveVar = data.v_loc(i, Apl);
                for (int j = i; j < this->input_rows(); j++) {
                    if (this->derived().hessian_elem_is_nonzero(j, i))
                        freeloc++;
                }
                for (int j = 0; j < this->output_rows(); j++) {
                    this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
                }
            }
        } else {
            for (int i = 0; i < this->input_rows(); i++) {
                ActiveVar = data.v_loc(i, Apl);
                if (VarClashes[ActiveVar] == -1) {
                    //// uncontested
                } else {
                    /// contested lock mutex
                    ClashLocks[VarClashes[ActiveVar]].lock();
                }

                //////////////////////////Mutex
                /// Locked////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////
                ///// insert hessian column symetrically
                for (int j = i; j < this->input_rows(); j++) {
                    if (this->derived().hessian_elem_is_nonzero(j, i))
                        freeloc++;
                }
                ///// insert jacobian column

                for (int j = 0; j < this->output_rows(); j++) {
                    this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
                }
                ///////////////////////////////////////////////////////////////////////////////
                if (VarClashes[ActiveVar] == -1) {
                    //// uncontested
                } else {
                    /// contested unlock mutex
                    ClashLocks[VarClashes[ActiveVar]].unlock();
                }
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
