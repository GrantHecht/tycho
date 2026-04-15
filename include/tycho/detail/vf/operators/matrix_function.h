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

template <int MRows, int MCols> struct MatrixRowsCols;

template <class Func, int MRows, int MCols, int MMajor>
struct MatrixFunctionView : Func, MatrixRowsCols<MRows, MCols> {
    static constexpr int Major = MMajor;
    static constexpr int MROWS = MRows;
    static constexpr int MCOLS = MCols;

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

    /// Return the matrix inverse as a new MatrixFunctionView.
    /// Dispatches to MatrixInverse<2,Major>, <3,Major>, or <-1,Major>.
    /// Throws if matrix is not square.
    /// Defined out-of-line in operator_overloads.h (needs GenericFunction + MatrixInverse).
    auto inverse() const;
};

/// Construct a row-major MatrixFunctionView from any VF expression.
template <class Func, int IR, int OR>
auto row_matrix(const DenseFunctionBase<Func, IR, OR> &f, int rows, int cols) {
    return MatrixFunctionView<Func, -1, -1, Eigen::RowMajor>(f.derived(), rows, cols);
}

template <class Func1, class... Funcs>
struct ColMajorMatrix : MatrixFunctionView<StackedOutputs<Func1, Funcs...>, Func1::ORC,
                                           1 + sizeof...(Funcs), Eigen::ColMajor> {
    using Base = MatrixFunctionView<StackedOutputs<Func1, Funcs...>, Func1::ORC,
                                    1 + sizeof...(Funcs), Eigen::ColMajor>;
    ColMajorMatrix(Func1 f1, Funcs... fs)
        : Base(StackedOutputs{f1, fs...}, f1.output_rows(), 1 + sizeof...(Funcs)) {}
};

template <class Func1, class... Funcs>
struct RowMajorMatrix : MatrixFunctionView<StackedOutputs<Func1, Funcs...>, 1 + sizeof...(Funcs),
                                           Func1::ORC, Eigen::RowMajor> {
    using Base = MatrixFunctionView<StackedOutputs<Func1, Funcs...>, 1 + sizeof...(Funcs),
                                    Func1::ORC, Eigen::RowMajor>;
    RowMajorMatrix(Func1 f1, Funcs... fs)
        : Base(StackedOutputs{f1, fs...}, 1 + sizeof...(Funcs), f1.output_rows()) {}
};

template <int MRows, int MCols> struct MatrixRowsCols {
    static constexpr int matrix_rows_ = MRows;
    static constexpr int matrix_cols_ = MCols;

    MatrixRowsCols(int rows, int cols) {}
};

template <> struct MatrixRowsCols<-1, -1> {
    int matrix_rows_ = 0;
    int matrix_cols_ = 0;

    MatrixRowsCols(int rows, int cols) : matrix_rows_(rows), matrix_cols_(cols) {}
};

template <int MCols> struct MatrixRowsCols<-1, MCols> {
    int matrix_rows_ = 0;
    static constexpr int matrix_cols_ = MCols;
    MatrixRowsCols(int rows, int cols) : matrix_rows_(rows) {}
};

template <int MRows> struct MatrixRowsCols<MRows, -1> {
    static constexpr int matrix_rows_ = MRows;
    int matrix_cols_ = 0;
    MatrixRowsCols(int rows, int cols) : matrix_cols_(cols) {}
};

} // namespace tycho::vf
