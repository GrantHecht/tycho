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

#include <stdexcept>
#include <string>

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @cond INTERNAL
template <int MRows, int MCols> struct MatrixRowsCols;
/// @endcond

/// @ingroup vf
/// @brief Reshapes a vector-valued function's output into a matrix view.
///
/// Wraps a backing VectorFunction @p Func and reinterprets its flat output as a
/// @p MRows × @p MCols matrix in @p MMajor storage order. Used by the
/// matrix-product and matrix-inverse operators.
/// @tparam Func    The backing VectorFunction whose output is reshaped.
/// @tparam MRows   Compile-time matrix row count (-1 for runtime-sized).
/// @tparam MCols   Compile-time matrix column count (-1 for runtime-sized).
/// @tparam MMajor  Eigen storage order (Eigen::RowMajor or Eigen::ColMajor).
template <class Func, int MRows, int MCols, int MMajor>
struct MatrixFunctionView : Func, MatrixRowsCols<MRows, MCols> {
    static constexpr int Major = MMajor; ///< Eigen storage order of the matrix view.
    static constexpr int MROWS = MRows;  ///< Compile-time matrix row count.
    static constexpr int MCOLS = MCols;  ///< Compile-time matrix column count.

    /// @brief Construct the matrix view, validating that @p rows * @p cols matches the output.
    /// @param f     The backing function whose output is reshaped.
    /// @param rows  Runtime matrix row count.
    /// @param cols  Runtime matrix column count.
    MatrixFunctionView(Func f, int rows, int cols)
        : Func(f), MatrixRowsCols<MRows, MCols>(rows, cols) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument(
                "MatrixFunctionView: rows and cols must be positive (got rows=" +
                std::to_string(rows) + ", cols=" + std::to_string(cols) + ")");
        }
        // output_rows() is always >= 0 at runtime: it returns the compile-time
        // OR for non-dynamic functions, and otherwise the runtime output_rows_val
        // which is zero-initialized and set by the backing expression during
        // construction. A zero result here indicates a function that never had
        // its output size established — catch it as a shape mismatch.
        const int out = f.output_rows();
        if (rows * cols != out) {
            throw std::invalid_argument(
                "MatrixFunctionView: rows * cols must match function output size (got rows=" +
                std::to_string(rows) + ", cols=" + std::to_string(cols) +
                ", output_rows=" + std::to_string(out) + ")");
        }
    }

    /// @brief Return the matrix inverse as a new MatrixFunctionView.
    ///
    /// Dispatches to @c MatrixInverse<2,Major>, @c <3,Major>, or @c <-1,Major>.
    /// Throws if the matrix is not square. Defined out-of-line in
    /// @c operator_overloads.h (needs GenericFunction + MatrixInverse).
    /// @return A MatrixFunctionView representing the inverse.
    auto inverse() const;
};

/// @brief Construct a row-major MatrixFunctionView from any VF expression.
/// @tparam Func  The backing VectorFunction type.
/// @tparam IR    Compile-time input row count of @p f.
/// @tparam OR    Compile-time output row count of @p f.
/// @param f     The source function expression.
/// @param rows  Runtime matrix row count.
/// @param cols  Runtime matrix column count.
/// @return A row-major MatrixFunctionView over @p f.
template <class Func, int IR, int OR>
auto row_matrix(const DenseFunctionBase<Func, IR, OR> &f, int rows, int cols) {
    return MatrixFunctionView<Func, -1, -1, Eigen::RowMajor>(f.derived(), rows, cols);
}

/// @ingroup vf
/// @brief Column-major matrix assembled by stacking function outputs as columns.
/// @tparam Func1  First column function (sets the matrix row count).
/// @tparam Funcs  Remaining column functions.
template <class Func1, class... Funcs>
struct ColMajorMatrix : MatrixFunctionView<StackedOutputs<Func1, Funcs...>, Func1::ORC,
                                           1 + sizeof...(Funcs), Eigen::ColMajor> {
    /// @brief Convenience alias for the underlying matrix-view base.
    using Base = MatrixFunctionView<StackedOutputs<Func1, Funcs...>, Func1::ORC,
                                    1 + sizeof...(Funcs), Eigen::ColMajor>;
    /// @brief Construct by stacking the column functions.
    /// @param f1  First column function.
    /// @param fs  Remaining column functions.
    ColMajorMatrix(Func1 f1, Funcs... fs)
        : Base(StackedOutputs{f1, fs...}, f1.output_rows(), 1 + sizeof...(Funcs)) {}
};

/// @ingroup vf
/// @brief Row-major matrix assembled by stacking function outputs as rows.
/// @tparam Func1  First row function (sets the matrix column count).
/// @tparam Funcs  Remaining row functions.
template <class Func1, class... Funcs>
struct RowMajorMatrix : MatrixFunctionView<StackedOutputs<Func1, Funcs...>, 1 + sizeof...(Funcs),
                                           Func1::ORC, Eigen::RowMajor> {
    /// @brief Convenience alias for the underlying matrix-view base.
    using Base = MatrixFunctionView<StackedOutputs<Func1, Funcs...>, 1 + sizeof...(Funcs),
                                    Func1::ORC, Eigen::RowMajor>;
    /// @brief Construct by stacking the row functions.
    /// @param f1  First row function.
    /// @param fs  Remaining row functions.
    RowMajorMatrix(Func1 f1, Funcs... fs)
        : Base(StackedOutputs{f1, fs...}, 1 + sizeof...(Funcs), f1.output_rows()) {}
};

/// @internal
/// @brief Holds the matrix shape (rows × cols) of a @ref MatrixFunctionView.
///
/// Primary template stores both dimensions at compile time; partial
/// specialisations carry runtime dimensions where -1 (dynamic) is requested.
/// @tparam MRows  Compile-time row count.
/// @tparam MCols  Compile-time column count.
/// @endinternal
template <int MRows, int MCols> struct MatrixRowsCols {
    static constexpr int matrix_rows_ = MRows; ///< Compile-time row count.
    static constexpr int matrix_cols_ = MCols; ///< Compile-time column count.

    /// @brief Construct; runtime arguments are ignored for the fully-static shape.
    /// @param rows  Ignored runtime row count.
    /// @param cols  Ignored runtime column count.
    MatrixRowsCols(int rows, int cols) {}
};

/// @internal
/// @brief @ref MatrixRowsCols specialisation with both dimensions dynamic.
/// @endinternal
template <> struct MatrixRowsCols<-1, -1> {
    int matrix_rows_ = 0; ///< Runtime row count.
    int matrix_cols_ = 0; ///< Runtime column count.

    /// @brief Construct from runtime dimensions.
    /// @param rows  Runtime row count.
    /// @param cols  Runtime column count.
    MatrixRowsCols(int rows, int cols) : matrix_rows_(rows), matrix_cols_(cols) {}
};

/// @internal
/// @brief @ref MatrixRowsCols specialisation with dynamic rows and static columns.
/// @tparam MCols  Compile-time column count.
/// @endinternal
template <int MCols> struct MatrixRowsCols<-1, MCols> {
    int matrix_rows_ = 0;                      ///< Runtime row count.
    static constexpr int matrix_cols_ = MCols; ///< Compile-time column count.
    /// @brief Construct from a runtime row count.
    /// @param rows  Runtime row count.
    /// @param cols  Ignored runtime column count.
    MatrixRowsCols(int rows, int cols) : matrix_rows_(rows) {}
};

/// @internal
/// @brief @ref MatrixRowsCols specialisation with static rows and dynamic columns.
/// @tparam MRows  Compile-time row count.
/// @endinternal
template <int MRows> struct MatrixRowsCols<MRows, -1> {
    static constexpr int matrix_rows_ = MRows; ///< Compile-time row count.
    int matrix_cols_ = 0;                      ///< Runtime column count.
    /// @brief Construct from a runtime column count.
    /// @param rows  Ignored runtime row count.
    /// @param cols  Runtime column count.
    MatrixRowsCols(int rows, int cols) : matrix_cols_(cols) {}
};

} // namespace tycho::vf
