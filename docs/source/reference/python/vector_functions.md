(vectorfunctions-python)=
# VectorFunction

The Python API for Tycho's VectorFunction subsystem, exposed through the
`tychopy.vector_functions` module. Every symbol below is a thin re-export of
a nanobind-bound C++ type or function; C++ users should consult the
{doc}`C++ reference </reference/cpp/vector_functions>` for the underlying
templates.

```{eval-rst}
.. currentmodule:: tychopy.vector_functions
```

A VectorFunction is a symbolic, differentiable map
$f : \mathbb{R}^{n} \to \mathbb{R}^{m}$. You build one by composing the
{py:class}`~tychopy.vector_functions.Arguments` placeholder with the elementwise math functions,
vector operations, and composition helpers documented here; the resulting
object evaluates its primal, Jacobian, and Hessian on demand. For the
conceptual model see the {doc}`Explanation </explanation/index>` section; for
a hands-on introduction see the {doc}`Tutorials </tutorials/basics/index>`.

## Argument construction

The entry point for every VectorFunction:
{py:class}`~tychopy.vector_functions.Arguments` declares the symbolic input
vector. Indexing and slicing an `Arguments` object yields
{py:class}`~tychopy.vector_functions.Element` (scalar) and
{py:class}`~tychopy.vector_functions.Segment` (sub-vector) views.

```{eval-rst}
.. autoclass:: Arguments
   :members:
   :special-members: __getitem__, __call__

.. autoclass:: Element
   :members:

.. autoclass:: Segment
   :members:

.. autoclass:: Segment2
   :members:

.. autoclass:: Segment3
   :members:
```

## Core function types

The base wrapper types returned by composition.
{py:class}`~tychopy.vector_functions.VectorFunction` is the vector-valued
base; {py:class}`~tychopy.vector_functions.ScalarFunction` is its
scalar-output counterpart.

```{eval-rst}
.. autoclass:: VectorFunction
   :members:

.. autoclass:: ScalarFunction
   :members:

.. autoclass:: Comparative
   :members:

.. autoclass:: Conditional
   :members:
```

## Constants

```{eval-rst}
.. autoclass:: ConstantScalar
   :members:

.. autoclass:: ConstantVector
   :members:
```

## Elementwise math

Scalar elementwise functions. Each accepts a scalar-valued VectorFunction
(output dimension 1) and returns a new scalar VectorFunction.

```{eval-rst}
.. autofunction:: sin
.. autofunction:: cos
.. autofunction:: tan
.. autofunction:: sinh
.. autofunction:: cosh
.. autofunction:: tanh
.. autofunction:: arcsin
.. autofunction:: arccos
.. autofunction:: arctan
.. autofunction:: arcsinh
.. autofunction:: arccosh
.. autofunction:: arctanh
.. autofunction:: arctan2
.. autofunction:: exp
.. autofunction:: log
.. autofunction:: sqrt
.. autofunction:: squared
.. autofunction:: pow
.. autofunction:: abs
.. autofunction:: sign
```

## Vector operations

Operations over vector-valued VectorFunctions: norms, normalization, and
the inner/outer/cross products.

```{eval-rst}
.. autofunction:: norm
.. autofunction:: squared_norm
.. autofunction:: cubed_norm
.. autofunction:: inverse_norm
.. autofunction:: inverse_squared_norm
.. autofunction:: inverse_cubed_norm
.. autofunction:: inverse_four_norm
.. autofunction:: normalize
.. autofunction:: normalized
.. autofunction:: normalized_power2
.. autofunction:: normalized_power3
.. autofunction:: normalized_power4
.. autofunction:: normalized_power5
.. autofunction:: dot
.. autofunction:: cross
.. autofunction:: doublecross
.. autofunction:: cwise_product
.. autofunction:: cwise_quotient
.. autofunction:: matmul
.. autofunction:: quat_product
.. autofunction:: quat_rotate
```

## Composition and reduction

Helpers that combine several VectorFunctions into one, or reduce a
VectorFunction's outputs.

```{eval-rst}
.. autofunction:: stack
.. autofunction:: stack_scalar
.. autofunction:: sum
.. autofunction:: sum_scalar
.. autofunction:: sum_elems
.. autofunction:: ifelse
```

## Root finding

Build a scalar VectorFunction defined implicitly by the root of another
scalar function.

```{eval-rst}
.. autofunction:: scalar_root_finder
```

## Scaling

Input/output scaling wrappers used to non-dimensionalize a VectorFunction.

```{eval-rst}
.. autoclass:: IOScaled
   :members:
```

## Matrix views

```{eval-rst}
.. autoclass:: ColMatrix
   :members:

.. autoclass:: RowMatrix
   :members:
```

## Interpolation tables

Lookup-table VectorFunctions that interpolate tabulated data of one to four
dimensions.

```{eval-rst}
.. autoclass:: InterpType
   :members:

.. autoclass:: InterpTable1D
   :members:

.. autoclass:: InterpTable2D
   :members:

.. autoclass:: InterpTable3D
   :members:

.. autoclass:: InterpTable4D
   :members:
```

## Python-authored functions

Wrappers for defining a VectorFunction from a Python (or Numba-compiled)
callable, plus the finite-difference derivative checker.

```{eval-rst}
.. autoclass:: PyVectorFunction
   :members:

.. autoclass:: PyScalarFunction
   :members:

.. autoclass:: NumbaVectorFunction
   :members:

.. autoclass:: NumbaScalarFunction
   :members:

.. autofunction:: FDDerivChecker
```
