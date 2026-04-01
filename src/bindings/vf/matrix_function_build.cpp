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
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "tycho_vector_functions.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

void MatrixFunctionBuild(nb::module_ &m) {

    using Gen = GenericFunction<-1, -1>;
    using Func = GenericFunction<-1, -1>;

    using colmattype = MatrixFunctionView<Func, -1, -1, Eigen::ColMajor>;
    using colvectype = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>;
    using rowmattype = MatrixFunctionView<Func, -1, -1, Eigen::RowMajor>;
    using rowvectype = MatrixFunctionView<Func, 1, -1, Eigen::RowMajor>;

    auto ColMat = nb::class_<colmattype>(m, "ColMatrix");

    ColMat.def(nb::init<Func, int, int>());
    ColMat.def("__init__", [](colmattype *self, const std::vector<Func> &colfuns) {
        int cols = colfuns.size();
        if (cols == 0) {
            throw std::invalid_argument("List must contain at least one function.");
        }
        int rows = colfuns[0].output_rows();
        for (auto &fun : colfuns)
            if (fun.output_rows() != rows)
                throw std::invalid_argument("Column Functions must have same output size");
        auto tmp = DynamicStack(colfuns);
        new (self) colmattype(tmp, rows, cols);
    });

    ColMat.def("__mul__", [](const colmattype &m1, const colmattype &m2) {
        auto tmp = MatrixFunctionProduct<colmattype, colmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    ColMat.def("__mul__", [](const colmattype &m1, const rowmattype &m2) {
        auto tmp = MatrixFunctionProduct<colmattype, rowmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    ColMat.def("__mul__", [](const colmattype &m1, double scale) {
        return colmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    });
    ColMat.def("__rmul__", [](const colmattype &m1, double scale) {
        return colmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    });
    ColMat.def("__add__", [](const colmattype &m1, const Eigen::MatrixXd &mshift) {
        if (m1.matrix_rows_ != mshift.rows() || m1.matrix_cols_ != mshift.cols()) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }
        Eigen::VectorXd v = mshift.reshaped(mshift.rows() * mshift.cols(), 1);
        return colmattype(m1 + v, m1.matrix_rows_, m1.matrix_cols_);
    });
    ColMat.def("__radd__", [](const colmattype &m1, const Eigen::MatrixXd &mshift) {
        if (m1.matrix_rows_ != mshift.rows() || m1.matrix_cols_ != mshift.cols()) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }
        Eigen::VectorXd v = mshift.reshaped(mshift.rows() * mshift.cols(), 1);
        return colmattype(m1 + v, m1.matrix_rows_, m1.matrix_cols_);
    });

    ColMat.attr("__array_ufunc__") = nb::none();

    ColMat.def("__add__", [](const colmattype &m1, const colmattype &m2) {
        if (m1.matrix_rows_ != m2.matrix_rows_ || m1.matrix_cols_ != m2.matrix_cols_) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }

        return colmattype(m1 + m2, m1.matrix_rows_, m1.matrix_cols_);
    });

    ColMat.def("inverse", [](const colmattype &m1) {
        if (m1.matrix_rows_ != m1.matrix_cols_) {
            throw std::invalid_argument("Matrix must be square to be invertible");
        }

        int size = m1.matrix_rows_;

        GenericFunction<-1, -1> invfunc;

        if (size == 2) {
            invfunc = MatrixInverse<2, Eigen::ColMajor>(size);
        } else if (size == 3) {
            invfunc = MatrixInverse<3, Eigen::ColMajor>(size);
        } else {
            invfunc = MatrixInverse<-1, Eigen::ColMajor>(size);
        }

        GenericFunction<-1, -1> minv(invfunc.eval(m1));

        return colmattype(minv, size, size);
    });

    ColMat.def("transpose",
               [](const colmattype &m1) { return rowmattype(m1, m1.matrix_cols_, m1.matrix_rows_); });

    ColMat.def("vf", [](const colmattype &m) { return GenericFunction<-1, -1>(m); });

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    auto RowMat = nb::class_<rowmattype>(m, "RowMatrix");

    RowMat.def(nb::init<Func, int, int>());
    RowMat.def("__init__", [](rowmattype *self, const std::vector<Func> &rowfuns) {
        int rows = rowfuns.size();
        if (rows == 0) {
            throw std::invalid_argument("List must contain at least one function.");
        }
        int cols = rowfuns[0].output_rows();
        for (auto &fun : rowfuns)
            if (fun.output_rows() != cols)
                throw std::invalid_argument("Row Functions must have same output size");
        auto tmp = DynamicStack(rowfuns);
        new (self) rowmattype(tmp, rows, cols);
    });

    RowMat.def("__mul__", [](const rowmattype &m1, const colmattype &m2) {
        auto tmp = MatrixFunctionProduct<rowmattype, colmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    RowMat.def("__mul__", [](const rowmattype &m1, const rowmattype &m2) {
        auto tmp = MatrixFunctionProduct<rowmattype, rowmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    RowMat.def("__mul__", [](const rowmattype &m1, double scale) {
        return rowmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    });
    RowMat.def("__rmul__", [](const rowmattype &m1, double scale) {
        return rowmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    });

    RowMat.def("__add__", [](const rowmattype &m1, const rowmattype &m2) {
        if (m1.matrix_rows_ != m2.matrix_rows_ || m1.matrix_cols_ != m2.matrix_cols_) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }

        return rowmattype(m1 + m2, m1.matrix_rows_, m1.matrix_cols_);
    });

    RowMat.attr("__array_ufunc__") = nb::none();

    RowMat.def("__add__", [](const rowmattype &m1, const Eigen::MatrixXd &mshift) {
        if (m1.matrix_rows_ != mshift.rows() || m1.matrix_cols_ != mshift.cols()) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }
        Eigen::MatrixXd tmp = mshift.transpose();
        Eigen::VectorXd v = tmp.reshaped(mshift.rows() * mshift.cols(), 1);
        return rowmattype(m1 + v, m1.matrix_rows_, m1.matrix_cols_);
    });
    RowMat.def("__radd__", [](const rowmattype &m1, const Eigen::MatrixXd &mshift) {
        if (m1.matrix_rows_ != mshift.rows() || m1.matrix_cols_ != mshift.cols()) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }
        Eigen::MatrixXd tmp = mshift.transpose();
        Eigen::VectorXd v = tmp.reshaped(mshift.rows() * mshift.cols(), 1);
        return rowmattype(m1 + v, m1.matrix_rows_, m1.matrix_cols_);
    });

    RowMat.def("inverse", [](const rowmattype &m1) {
        if (m1.matrix_rows_ != m1.matrix_cols_) {
            throw std::invalid_argument("Matrix must be square to be invertible.");
        }

        int size = m1.matrix_rows_;

        GenericFunction<-1, -1> invfunc;

        if (size == 2) {
            invfunc = MatrixInverse<2, Eigen::RowMajor>(size);
        } else if (size == 3) {
            invfunc = MatrixInverse<3, Eigen::RowMajor>(size);
        } else {
            invfunc = MatrixInverse<-1, Eigen::RowMajor>(size);
        }

        GenericFunction<-1, -1> minv(invfunc.eval(m1));

        return rowmattype(minv, size, size);
    });

    RowMat.def("transpose",
               [](const rowmattype &m1) { return colmattype(m1, m1.matrix_cols_, m1.matrix_rows_); });

    RowMat.def("vf", [](const rowmattype &m) { return GenericFunction<-1, -1>(m); });

    /*
    These two must come last.
    */

    ColMat.def("__mul__", [](const colmattype &m1, const Func &m2) {
        auto tmp =
            MatrixFunctionProduct<colmattype, colvectype>(m1, colvectype(m2, m2.output_rows(), 1));
        return GenericFunction<-1, -1>(tmp); // Result must be vector
    });

    RowMat.def("__mul__", [](const rowmattype &m1, const Func &m2) {
        auto tmp =
            MatrixFunctionProduct<rowmattype, colvectype>(m1, colvectype(m2, m2.output_rows(), 1));
        return GenericFunction<-1, -1>(tmp);
    });

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////

    m.def("matmul", [](const colmattype &m1, const colmattype &m2) {
        auto tmp = MatrixFunctionProduct<colmattype, colmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    m.def("matmul", [](const colmattype &m1, const rowmattype &m2) {
        auto tmp = MatrixFunctionProduct<colmattype, rowmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    m.def("matmul", [](const colmattype &m1, const Func &m2) {
        auto tmp =
            MatrixFunctionProduct<colmattype, colvectype>(m1, colvectype(m2, m2.output_rows(), 1));
        return GenericFunction<-1, -1>(tmp);
    });

    m.def("matmul", [](const colmattype &m1, const Eigen::VectorXd &v) {
        auto m2 = Constant<-1, -1>(m1.input_rows(), v);
        auto tmp =
            MatrixFunctionProduct<colmattype, colvectype>(m1, colvectype(m2, m2.output_rows(), 1));
        return GenericFunction<-1, -1>(tmp);
    });

    m.def("matmul", [](const rowmattype &m1, const colmattype &m2) {
        auto tmp = MatrixFunctionProduct<rowmattype, colmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    m.def("matmul", [](const rowmattype &m1, const rowmattype &m2) {
        auto tmp = MatrixFunctionProduct<rowmattype, rowmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    m.def("matmul", [](const rowmattype &m1, const Func &m2) {
        auto tmp =
            MatrixFunctionProduct<rowmattype, colvectype>(m1, colvectype(m2, m2.output_rows(), 1));
        return GenericFunction<-1, -1>(tmp);
    });

    m.def("matmul", [](const rowmattype &m1, const Eigen::VectorXd &v) {
        auto m2 = Constant<-1, -1>(m1.input_rows(), v);
        auto tmp =
            MatrixFunctionProduct<rowmattype, colvectype>(m1, colvectype(m2, m2.output_rows(), 1));
        return GenericFunction<-1, -1>(tmp);
    });

    m.def("matmul", [](const Eigen::Matrix<double, 2, 2> &mat, const Gen &vec) {
        return Gen(MatrixScaled<Gen, 2>(vec, mat));
    });

    m.def("matmul", [](const Eigen::Matrix<double, 3, 3> &mat, const Gen &vec) {
        return Gen(MatrixScaled<Gen, 3>(vec, mat));
    });

    m.def("matmul", [](const Eigen::MatrixXd &mat, const Gen &vec) {
        return Gen(MatrixScaled<Gen, -1>(vec, mat));
    });
}

} // namespace tycho