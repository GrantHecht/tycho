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

    ColMat.def(nb::init<Func, int, int>(),
               R"doc(Construct a ColMatrix by reshaping a VectorFunction output into a column-major matrix.

Parameters
----------
f : VectorFunction
    Backing VectorFunction whose flat output (length ``rows * cols``) is reinterpreted
    as a column-major matrix.
rows : int
    Number of rows in the matrix view.  Must satisfy ``rows * cols == f.output_rows()``.
cols : int
    Number of columns in the matrix view.

Raises
------
ValueError
    If ``rows`` or ``cols`` is non-positive, or if ``rows * cols`` does not equal
    ``f.output_rows()``.
)doc");
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
    },
               R"doc(Matrix product of two column-major matrix VectorFunctions (``self @ other``).

Computes ``M1(x) @ M2(x)`` where both factors are evaluated at the same input
``x``.  The inner dimension of ``M1`` must equal the outer dimension of ``M2``.

Parameters
----------
other : ColMatrix or RowMatrix or VectorFunction or float
    Right-hand factor.  When ``other`` is a plain VectorFunction it is treated
    as a column vector; when it is a ``float`` every element is scaled by that
    constant.

Returns
-------
ColMatrix or VectorFunction
    Column-major matrix (or flat VectorFunction when ``other`` is a column
    vector) representing the product.

Raises
------
ValueError
    If the inner dimensions of the two matrix functions do not match.
)doc");

    ColMat.def("__mul__", [](const colmattype &m1, const rowmattype &m2) {
        auto tmp = MatrixFunctionProduct<colmattype, rowmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    ColMat.def("__mul__", [](const colmattype &m1, double scale) {
        return colmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    });
    ColMat.def("__rmul__", [](const colmattype &m1, double scale) {
        return colmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    },
               R"doc(Scalar multiplication with the scalar on the left (``scale * self``).

Parameters
----------
scale : float
    Scalar factor applied to every element of the matrix function.

Returns
-------
ColMatrix
    A new ColMatrix scaled by ``scale``.
)doc");
    ColMat.def("__add__", [](const colmattype &m1, const Eigen::MatrixXd &mshift) {
        if (m1.matrix_rows_ != mshift.rows() || m1.matrix_cols_ != mshift.cols()) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }
        Eigen::VectorXd v = mshift.reshaped(mshift.rows() * mshift.cols(), 1);
        return colmattype(m1 + v, m1.matrix_rows_, m1.matrix_cols_);
    },
               R"doc(Add a constant matrix offset to a ColMatrix VectorFunction (``self + mshift``).

Adds the constant numpy matrix ``mshift`` element-wise to the matrix produced
by ``self`` at every evaluation point.  The dimensions must match exactly.

Parameters
----------
mshift : array_like, shape (rows, cols)
    Constant matrix offset with the same shape as ``self``.

Returns
-------
ColMatrix
    A new ColMatrix evaluating ``M(x) + mshift``.

Raises
------
ValueError
    If ``mshift`` does not have the same number of rows and columns as ``self``.
)doc");
    ColMat.def("__radd__", [](const colmattype &m1, const Eigen::MatrixXd &mshift) {
        if (m1.matrix_rows_ != mshift.rows() || m1.matrix_cols_ != mshift.cols()) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }
        Eigen::VectorXd v = mshift.reshaped(mshift.rows() * mshift.cols(), 1);
        return colmattype(m1 + v, m1.matrix_rows_, m1.matrix_cols_);
    },
               R"doc(Add a constant matrix offset with the constant on the left (``mshift + self``).

Commutative companion to ``__add__``; produces the same result as
``self + mshift``.

Parameters
----------
mshift : array_like, shape (rows, cols)
    Constant matrix offset with the same shape as ``self``.

Returns
-------
ColMatrix
    A new ColMatrix evaluating ``mshift + M(x)``.

Raises
------
ValueError
    If ``mshift`` does not have the same number of rows and columns as ``self``.
)doc");

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
    },
               R"doc(Compute the symbolic matrix inverse of this square ColMatrix VectorFunction.

Returns a new ColMatrix VectorFunction that, for any input ``x``, evaluates
to the inverse of the matrix ``M(x)``.  Dispatches to analytically-derived
2×2 or 3×3 inverse kernels for small matrices and to a general LU-based
kernel otherwise.

Returns
-------
ColMatrix
    A ColMatrix representing ``M(x)^{-1}``.

Raises
------
ValueError
    If the matrix is not square (``rows != cols``).
)doc");

    ColMat.def("transpose", [](const colmattype &m1) {
        return rowmattype(m1, m1.matrix_cols_, m1.matrix_rows_);
    },
               R"doc(Transpose this ColMatrix VectorFunction.

Returns a RowMatrix VectorFunction whose (i, j) entry equals the (j, i) entry
of ``self``.  The transposed matrix shares the same backing VectorFunction;
no copy of the output data is made at evaluation time.

Returns
-------
RowMatrix
    A row-major matrix view of the transposed function.
)doc");

    ColMat.def("vf", [](const colmattype &m) { return GenericFunction<-1, -1>(m); },
               R"doc(Extract the underlying flat VectorFunction from this ColMatrix.

Returns the backing VectorFunction whose output is the column-major
flattening of the matrix (column-by-column, length ``rows * cols``).
Use this to pass a matrix VectorFunction to APIs that expect a plain
VectorFunction.

Returns
-------
VectorFunction
    The flat VectorFunction underlying this ColMatrix.
)doc");

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    auto RowMat = nb::class_<rowmattype>(m, "RowMatrix");

    RowMat.def(nb::init<Func, int, int>(),
               R"doc(Construct a RowMatrix by reshaping a VectorFunction output into a row-major matrix.

Parameters
----------
f : VectorFunction
    Backing VectorFunction whose flat output (length ``rows * cols``) is reinterpreted
    as a row-major matrix.
rows : int
    Number of rows in the matrix view.  Must satisfy ``rows * cols == f.output_rows()``.
cols : int
    Number of columns in the matrix view.

Raises
------
ValueError
    If ``rows`` or ``cols`` is non-positive, or if ``rows * cols`` does not equal
    ``f.output_rows()``.
)doc");
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
    },
               R"doc(Matrix product of two row-major matrix VectorFunctions (``self @ other``).

Computes ``M1(x) @ M2(x)`` where both factors are evaluated at the same input
``x``.  The inner dimension of ``M1`` must equal the outer dimension of ``M2``.

Parameters
----------
other : ColMatrix or RowMatrix or VectorFunction or float
    Right-hand factor.  When ``other`` is a plain VectorFunction it is treated
    as a column vector; when it is a ``float`` every element is scaled by that
    constant.

Returns
-------
ColMatrix or RowMatrix
    Column-major matrix (or row-major when both operands are RowMatrix)
    representing the product.

Raises
------
ValueError
    If the inner dimensions of the two matrix functions do not match.
)doc");

    RowMat.def("__mul__", [](const rowmattype &m1, const rowmattype &m2) {
        auto tmp = MatrixFunctionProduct<rowmattype, rowmattype>(m1, m2);
        return colmattype(tmp, m1.matrix_rows_, m2.matrix_cols_);
    });

    RowMat.def("__mul__", [](const rowmattype &m1, double scale) {
        return rowmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    });
    RowMat.def("__rmul__", [](const rowmattype &m1, double scale) {
        return rowmattype(m1 * scale, m1.matrix_rows_, m1.matrix_cols_);
    },
               R"doc(Scalar multiplication with the scalar on the left (``scale * self``).

Parameters
----------
scale : float
    Scalar factor applied to every element of the matrix function.

Returns
-------
RowMatrix
    A new RowMatrix scaled by ``scale``.
)doc");

    RowMat.def("__add__", [](const rowmattype &m1, const rowmattype &m2) {
        if (m1.matrix_rows_ != m2.matrix_rows_ || m1.matrix_cols_ != m2.matrix_cols_) {
            throw std::invalid_argument("Matrices must have the same dimensions to be added.");
        }

        return rowmattype(m1 + m2, m1.matrix_rows_, m1.matrix_cols_);
    },
               R"doc(Element-wise sum of two RowMatrix VectorFunctions (``self + other``).

Computes ``M1(x) + M2(x)`` pointwise.  Both matrix functions must have the
same shape.

Parameters
----------
other : RowMatrix or array_like, shape (rows, cols)
    Right-hand summand.  When ``other`` is a constant numpy matrix it is
    broadcast as a fixed offset.

Returns
-------
RowMatrix
    A new RowMatrix evaluating the element-wise sum.

Raises
------
ValueError
    If the two matrix functions do not have the same shape.
)doc");

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
    },
               R"doc(Add a constant matrix offset with the constant on the left (``mshift + self``).

Commutative companion to ``__add__``; produces the same result as
``self + mshift``.

Parameters
----------
mshift : array_like, shape (rows, cols)
    Constant matrix offset with the same shape as ``self``.

Returns
-------
RowMatrix
    A new RowMatrix evaluating ``mshift + M(x)``.

Raises
------
ValueError
    If ``mshift`` does not have the same number of rows and columns as ``self``.
)doc");

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
    },
               R"doc(Compute the symbolic matrix inverse of this square RowMatrix VectorFunction.

Returns a new RowMatrix VectorFunction that, for any input ``x``, evaluates
to the inverse of the matrix ``M(x)``.  Dispatches to analytically-derived
2×2 or 3×3 inverse kernels for small matrices and to a general LU-based
kernel otherwise.

Returns
-------
RowMatrix
    A RowMatrix representing ``M(x)^{-1}``.

Raises
------
ValueError
    If the matrix is not square (``rows != cols``).
)doc");

    RowMat.def("transpose", [](const rowmattype &m1) {
        return colmattype(m1, m1.matrix_cols_, m1.matrix_rows_);
    },
               R"doc(Transpose this RowMatrix VectorFunction.

Returns a ColMatrix VectorFunction whose (i, j) entry equals the (j, i) entry
of ``self``.  The transposed matrix shares the same backing VectorFunction;
no copy of the output data is made at evaluation time.

Returns
-------
ColMatrix
    A column-major matrix view of the transposed function.
)doc");

    RowMat.def("vf", [](const rowmattype &m) { return GenericFunction<-1, -1>(m); },
               R"doc(Extract the underlying flat VectorFunction from this RowMatrix.

Returns the backing VectorFunction whose output is the row-major
flattening of the matrix (row-by-row, length ``rows * cols``).
Use this to pass a matrix VectorFunction to APIs that expect a plain
VectorFunction.

Returns
-------
VectorFunction
    The flat VectorFunction underlying this RowMatrix.
)doc");

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
    },
          R"doc(Matrix product of two matrix-valued VectorFunctions or a matrix and a vector/constant.

Computes ``M1(x) @ M2(x)`` (or ``M(x) @ v`` when the right operand is a
VectorFunction column vector or a constant numpy vector).  All matrix and
vector operands are evaluated at the same input ``x``.  Multiple overloads
accept ``ColMatrix``, ``RowMatrix``, ``VectorFunction``, ``numpy.ndarray``
(constant vector), and small constant matrices (2×2, 3×3, or general
``numpy.ndarray``).

Parameters
----------
m1 : ColMatrix or RowMatrix or array_like
    Left-hand factor.  May be a matrix VectorFunction or a constant numpy
    matrix (2×2, 3×3, or general ``ndarray``).
m2 : ColMatrix or RowMatrix or VectorFunction or array_like
    Right-hand factor.  May be a matrix VectorFunction, a plain VectorFunction
    (treated as a column vector), or a constant numpy vector.

Returns
-------
ColMatrix or VectorFunction
    Column-major matrix VectorFunction, or a flat VectorFunction when the
    right operand is a column vector.

Raises
------
ValueError
    If the inner dimensions of the two factors do not match.
)doc");

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