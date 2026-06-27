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
conceptual model see {doc}`The VectorFunction model </explanation/vector_function>`;
for a hands-on introduction see the
{doc}`first-VectorFunction tutorial </tutorials/basics/your_first_vectorfunction>`.

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
{py:class}`~tychopy.vector_functions.Comparative` and
{py:class}`~tychopy.vector_functions.Conditional` are the type-erased wrappers
produced by the min/max helpers and by `ifelse`, respectively.

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
(output dimension 1) and returns a new scalar VectorFunction, chaining the inner
Jacobian through the known derivative of the operation. In the table, $f$ denotes
the scalar input function.

| Python name | Computes |
| --- | --- |
| `sin` | $\sin f$ |
| `cos` | $\cos f$ |
| `tan` | $\tan f$ |
| `sinh` | $\sinh f$ |
| `cosh` | $\cosh f$ |
| `tanh` | $\tanh f$ |
| `arcsin` | $\arcsin f$ |
| `arccos` | $\arccos f$ |
| `arctan` | $\arctan f$ |
| `arcsinh` | $\operatorname{arcsinh} f$ |
| `arccosh` | $\operatorname{arccosh} f$ |
| `arctanh` | $\operatorname{arctanh} f$ |
| `arctan2` | Two-argument arctangent $\operatorname{atan2}(y, x)$, matching `numpy.arctan2` |
| `exp` | $e^{f}$ |
| `log` | Natural logarithm $\ln f$ |
| `sqrt` | $\sqrt{f}$ |
| `squared` | $f^2$ |
| `pow` | $f^{p}$ for a constant exponent $p$ |
| `abs` | $\lvert f\rvert$ |
| `sign` | $\operatorname{sign}(f)$ (locally constant; derivative is zero) |

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
the inner/outer/cross products. `normalize` is an alias of `normalized` (both
return $x/\lVert x\rVert$); use whichever reads better. In the table, $r$, $a$,
$b$, $c$ denote vector-valued operands.

| Python name | Computes |
| --- | --- |
| `norm` | $\lVert r\rVert$ |
| `squared_norm` | $\lVert r\rVert^2$ |
| `cubed_norm` | $\lVert r\rVert^3$ |
| `inverse_norm` | $1/\lVert r\rVert$ |
| `inverse_squared_norm` | $1/\lVert r\rVert^2$ |
| `inverse_cubed_norm` | $1/\lVert r\rVert^3$ |
| `inverse_four_norm` | $1/\lVert r\rVert^4$ |
| `normalize` | $r/\lVert r\rVert$ (alias of `normalized`) |
| `normalized` | $r/\lVert r\rVert$ |
| `normalized_power2` | $r/\lVert r\rVert^2$ |
| `normalized_power3` | $r/\lVert r\rVert^3$ (the two-body gravity term) |
| `normalized_power4` | $r/\lVert r\rVert^4$ |
| `normalized_power5` | $r/\lVert r\rVert^5$ |
| `dot` | Inner product $a \cdot b$ |
| `cross` | 3-vector cross product $a \times b$ |
| `doublecross` | Iterated cross product $(a \times b) \times c$ |
| `cwise_product` | Elementwise product $a \odot b$ |
| `cwise_quotient` | Elementwise quotient $a / b$ |
| `matmul` | Matrix product $M_1\, M_2$ (or $M\,v$) of matrix-valued functions |
| `quat_product` | Hamilton product $q_1 \otimes q_2$ of two scalar-last unit quaternions (4-vectors) |
| `quat_rotate` | Rotate a 3-vector by a scalar-last unit quaternion, $q \otimes (v,0) \otimes q^{-1}$ |

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

| Python name | Meaning |
| --- | --- |
| `stack` | Concatenate the outputs of several functions vertically into one longer vector |
| `stack_scalar` | `stack` restricted to scalar-output (dimension-1) functions |
| `sum` | Elementwise sum of several equal-output functions |
| `sum_scalar` | `sum` restricted to scalar-output functions |
| `sum_elems` | Sum several scalar segment functions into a single scalar (optionally weighted) |
| `ifelse` | Select between two branches by a predicate; differentiates the active branch (piecewise differentiable) |

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
