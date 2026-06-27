(vectorfunctions-cpp)=
# VectorFunction

The C++ API for Tycho's VectorFunction subsystem, declared in the
`tycho::vf` namespace and exposed through the
`include/tycho/vector_functions.h` umbrella header. Python users should
consult the {doc}`Python reference </reference/python/vector_functions>`.

A VectorFunction is a CRTP expression template representing a
differentiable map $f : \mathbb{R}^{n} \to \mathbb{R}^{m}$. Concrete
functions are built by composing {cpp:struct}`tycho::vf::Arguments` with the
operator DSL and the expression types below; the resulting type evaluates
its primal, Jacobian, and Hessian according to its derivative mode. For the
conceptual model see {doc}`The VectorFunction model </user_guide/vectorfunctions>`.

## Core types

The CRTP base classes every VectorFunction shares. `VectorFunction` is the base
users inherit from to author a custom function; lower in the inheritance chain,
`DenseFunctionBase` supplies the operator DSL (`+`, `*`, `.dot()`, `.norm()`,
`.segment()`, …) that every expression inherits.

```{eval-rst}
.. doxygenstruct:: tycho::vf::VectorFunction
   :project: tycho
   :members:

.. doxygenstruct:: tycho::vf::DenseFunctionBase
   :project: tycho
   :members:

.. doxygenstruct:: tycho::vf::SparseFunctionBase
   :project: tycho

.. doxygenstruct:: tycho::vf::VectorExpression
   :project: tycho
```

## Type-erased VectorFunctions

Runtime-polymorphic wrappers that erase the concrete expression type behind
a stable interface. These are the types stored by the optimal-control layer
and exposed to Python.

```{eval-rst}
.. doxygenstruct:: tycho::vf::GenericFunction
   :project: tycho
   :members:

.. doxygenstruct:: tycho::vf::GenericConditional
   :project: tycho
   :members:

.. doxygenstruct:: tycho::vf::GenericComparative
   :project: tycho
   :members:
```

## Derivative modes

Compile-time tags selecting how a function's Jacobian and Hessian are
computed, plus the finite-difference stencil configuration. (Authored
directly here because the enumerators are grouped under the `vf` Doxygen
module; their definitions live in
`detail/vf/derivatives/dense_derivatives.h` and
`detail/vf/derivatives/fd_coeffs.h`.)

```{eval-rst}
.. cpp:enum-class:: tycho::vf::DenseDerivativeMode

   Selects the Jacobian/Hessian evaluation strategy mixed into a dense
   VectorFunction.

   .. cpp:enumerator:: Analytic

      Hand-written analytic derivatives supplied by the function itself.

   .. cpp:enumerator:: FDiffFwd

      Forward (first-order) finite-difference Jacobian.

   .. cpp:enumerator:: FDiffCentArray

      Central finite-difference Jacobian via packed SuperScalar arrays.

   .. cpp:enumerator:: EnzymeAD

      Enzyme automatic differentiation (compiler-plugin AD).

.. cpp:enum-class:: tycho::vf::FDCoeffType

   Finite-difference stencil orientation.

   .. cpp:enumerator:: Backwards

      Backward (left-sided) stencil.

   .. cpp:enumerator:: Central

      Centered (symmetric) stencil.

   .. cpp:enumerator:: Forwards

      Forward (right-sided) stencil.
```

## Arguments and selectors

The symbolic input vector and the views that select scalar elements or
contiguous sub-vectors from it.

```{eval-rst}
.. doxygenstruct:: tycho::vf::Arguments
   :project: tycho

.. doxygenstruct:: tycho::vf::Segment
   :project: tycho

.. doxygenstruct:: tycho::vf::Elements
   :project: tycho

.. doxygenstruct:: tycho::vf::Constant
   :project: tycho

.. doxygenstruct:: tycho::vf::Value
   :project: tycho
```

## Elementwise math expressions

One expression type per scalar elementwise operation, produced by the free
functions `sin`, `cos`, `exp`, ….

```{eval-rst}
.. doxygenstruct:: tycho::vf::CwiseSin
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseCos
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseTan
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseSinH
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseCosH
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseTanH
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseArcSin
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseArcCos
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseArcTan
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseArcSinH
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseArcCosH
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseArcTanH
   :project: tycho
.. doxygenstruct:: tycho::vf::ArcTan2Op
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseExp
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseLog
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseSqrt
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseSquare
   :project: tycho
.. doxygenstruct:: tycho::vf::CwisePow
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseAbs
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseInverse
   :project: tycho
.. doxygenstruct:: tycho::vf::SignFunction
   :project: tycho
```

## Norms and vector products

Vector-valued reductions and products: Euclidean norms and their inverse
powers, normalization, and the cross / dot / quaternion products.

```{eval-rst}
.. doxygenstruct:: tycho::vf::Norm
   :project: tycho
.. doxygenstruct:: tycho::vf::SquaredNorm
   :project: tycho
.. doxygenstruct:: tycho::vf::NormPower
   :project: tycho
.. doxygenstruct:: tycho::vf::InverseNorm
   :project: tycho
.. doxygenstruct:: tycho::vf::InverseSquaredNorm
   :project: tycho
.. doxygenstruct:: tycho::vf::InverseNormPower
   :project: tycho
.. doxygenstruct:: tycho::vf::Normalized
   :project: tycho
.. doxygenstruct:: tycho::vf::NormalizedPower
   :project: tycho
.. doxygenstruct:: tycho::vf::CrossProduct
   :project: tycho
.. doxygenstruct:: tycho::vf::FunctionCrossProduct
   :project: tycho
.. doxygenstruct:: tycho::vf::FunctionDotProduct
   :project: tycho
.. doxygenstruct:: tycho::vf::FunctionQuatProduct
   :project: tycho
.. doxygenstruct:: tycho::vf::FunctionImagProduct
   :project: tycho
```

## Composition and reduction

Expression types that combine several functions: vertical stacking,
summation, differences, and function nesting (substituting one function
into another's arguments).

```{eval-rst}
.. doxygenstruct:: tycho::vf::StackedOutputs
   :project: tycho
.. doxygenstruct:: tycho::vf::DynamicStackedOutputs
   :project: tycho
.. doxygenstruct:: tycho::vf::TwoFunctionSum
   :project: tycho
.. doxygenstruct:: tycho::vf::MultiFunctionSum
   :project: tycho
.. doxygenstruct:: tycho::vf::FunctionDifference
   :project: tycho
.. doxygenstruct:: tycho::vf::FunctionPlusVector
   :project: tycho
.. doxygenstruct:: tycho::vf::NestedFunction
   :project: tycho
.. doxygenstruct:: tycho::vf::FunctionHolder
   :project: tycho
```

## Products and matrix functions

Function-by-function and function-by-matrix products, and the matrix-valued
function views.

```{eval-rst}
.. doxygenstruct:: tycho::vf::CwiseFunctionProduct
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseFunctionOperator
   :project: tycho
.. doxygenstruct:: tycho::vf::CwiseOperator
   :project: tycho
.. doxygenstruct:: tycho::vf::MatrixFunctionProduct
   :project: tycho
.. doxygenstruct:: tycho::vf::MatrixFunctionView
   :project: tycho
.. doxygenstruct:: tycho::vf::MatrixInverse
   :project: tycho
.. doxygenstruct:: tycho::vf::VectorScalarFunctionProduct
   :project: tycho
.. doxygenstruct:: tycho::vf::VectorScalarFunctionDivision
   :project: tycho
.. doxygenstruct:: tycho::vf::ColMajorMatrix
   :project: tycho
.. doxygenstruct:: tycho::vf::RowMajorMatrix
   :project: tycho
```

## Conditionals and comparatives

Branching expressions: `if`/`else` selection and the boolean comparatives
that drive them. The flag enumerators are authored here for the same
Doxygen-grouping reason as the derivative modes above; their definitions
live in `detail/vf/type_erasure/comparative.h` and
`detail/vf/type_erasure/conditional.h`.

```{eval-rst}
.. cpp:enum-class:: tycho::vf::ComparativeFlags

   Selects the reduction taken by a :cpp:struct:`tycho::vf::ComparativeFunction`.

   .. cpp:enumerator:: MinFlag

      Take the scalar minimum across the operand functions.

   .. cpp:enumerator:: MaxFlag

      Take the scalar maximum across the operand functions.

.. cpp:enum-class:: tycho::vf::ConditionalFlags

   The comparison or boolean connective evaluated by a
   :cpp:struct:`tycho::vf::ConditionalStatement`.

   .. cpp:enumerator:: LessThanFlag

      Less-than comparison (``<``).

   .. cpp:enumerator:: GreaterThanFlag

      Greater-than comparison (``>``).

   .. cpp:enumerator:: LessThanEqualToFlag

      Less-than-or-equal comparison (``<=``).

   .. cpp:enumerator:: GreaterThanEqualToFlag

      Greater-than-or-equal comparison (``>=``).

   .. cpp:enumerator:: EqualToFlag

      Equality comparison (``==``).

   .. cpp:enumerator:: ANDFlag

      Logical AND of two sub-conditions.

   .. cpp:enumerator:: ORFlag

      Logical OR of two sub-conditions.

.. doxygenstruct:: tycho::vf::IfElseFunction
   :project: tycho
.. doxygenstruct:: tycho::vf::ConditionalStatement
   :project: tycho
.. doxygenstruct:: tycho::vf::ComparativeFunction
   :project: tycho
.. doxygenstruct:: tycho::vf::ConstantConditional
   :project: tycho
```

## Scaling

Wrappers that non-dimensionalize a function by scaling its inputs and/or
outputs, with static (compile-time) and runtime variants.

```{eval-rst}
.. doxygenstruct:: tycho::vf::Scaled
   :project: tycho
.. doxygenstruct:: tycho::vf::StaticScaled
   :project: tycho
.. doxygenstruct:: tycho::vf::RowScaled
   :project: tycho
.. doxygenstruct:: tycho::vf::MatrixScaled
   :project: tycho
.. doxygenstruct:: tycho::vf::IOScaled
   :project: tycho
.. doxygenstruct:: tycho::vf::PaddedOutput
   :project: tycho
```

## Lambda and root-finding functions

Wrap an arbitrary C++ callable as a VectorFunction, or build a function
defined implicitly by a scalar root.

```{eval-rst}
.. doxygenstruct:: tycho::vf::LambdaFunction
   :project: tycho
.. doxygenstruct:: tycho::vf::LambdaFunction2
   :project: tycho
.. doxygenstruct:: tycho::vf::ScalarRootFinder
   :project: tycho
```

## Free functions

Factory helpers and the operator DSL. The arithmetic operators (`+`, `-`,
`*`, `/`) and comparison operators (`<`, `<=`, `>`, `>=`, `==`) are
overloaded for every combination of functions, scalars, and matrices; see
their declarations in `operators/operator_overloads.h`. The
`make_dynamic_stack` / `make_dynamic_sum` factories build runtime-sized
stacks and sums; `stack` and `sum` are the fixed-size counterparts.

```{eval-rst}
.. doxygenfunction:: tycho::vf::cross_product
   :project: tycho
.. doxygenfunction:: tycho::vf::dot_product
   :project: tycho
.. doxygenfunction:: tycho::vf::quat_product
   :project: tycho
.. doxygenfunction:: tycho::vf::quat_rotate(const DenseFunctionBase<QFunc, QIR, QOR>&, const DenseFunctionBase<VFunc, VIR, VOR>&)
   :project: tycho
```
