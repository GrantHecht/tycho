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

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

template <int MRows, int MCols> struct MatrixRowsCols;

template <class Func, int MRows, int MCols, int MMajor>
struct MatrixFunctionView : Func, MatrixRowsCols<MRows, MCols> {
    static const int Major = MMajor;
    static const int MROWS = MRows;
    static const int MCOLS = MCols;

    MatrixFunctionView(Func f, int rows, int cols)
        : Func(f), MatrixRowsCols<MRows, MCols>(rows, cols) {
        // make sure row col consistent with orows
    }
};

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
    static const int matrix_rows_ = MRows;
    static const int matrix_cols_ = MCols;

    MatrixRowsCols(int rows, int cols) {}
};

template <> struct MatrixRowsCols<-1, -1> {
    int matrix_rows_ = 0;
    int matrix_cols_ = 0;

    MatrixRowsCols(int rows, int cols) : matrix_rows_(rows), matrix_cols_(cols) {}
};

template <int MCols> struct MatrixRowsCols<-1, MCols> {
    int matrix_rows_ = 0;
    static const int matrix_cols_ = MCols;
    MatrixRowsCols(int rows, int cols) : matrix_rows_(rows) {}
};

template <int MRows> struct MatrixRowsCols<MRows, -1> {
    static const int matrix_rows_ = MRows;
    int matrix_cols_ = 0;
    MatrixRowsCols(int rows, int cols) : matrix_cols_(cols) {}
};

} // namespace tycho::vf
