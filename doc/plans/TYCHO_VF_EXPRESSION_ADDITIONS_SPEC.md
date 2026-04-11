# Spec: VF Expression Additions (Subsystem 3)

> Add `RowMatrix`/`.inverse()`, `quat_rotate`, and `cwise_product` Eigen
> overloads to the C++ VectorFunction expression DSL, exposing existing
> internal machinery as convenient free functions.

## Motivation

Three VF expression operations are available in Python but not in C++:

1. `vf.RowMatrix(vec, rows, cols).inverse() * Q` — runtime matrix
   construction, inversion, and matrix-function product. Used by CartPole
   (mass matrix) and BettsLowThrust (RTN rotation).
2. `vf.quat_rotate(q, v)` — quaternion rotation of a vector. Used by
   OptimalDocking for attitude dynamics.
3. `vf.cwise_product(vf_expr, eigen_vec)` — element-wise product accepting
   Eigen vectors directly. Currently requires wrapping in `Constant<>`.

All three have fully functional internal C++ types (`MatrixFunctionView`,
`MatrixInverse`, `MatrixFunctionProduct`, `FunctionQuatProduct`,
`RowScaled`). The gap is purely missing convenience API — no new algorithmic
work is needed.

## Scope

This spec covers only the three VF expression additions. It does NOT cover:
- InterpTable VF composition — Subsystem 4
- Advanced OC features (LGLInterpTable, STM, Jet) — Subsystem 5
- Domain models (MEE factories) — Subsystem 6

## Changes

### 1. RowMatrix Free Function + .inverse() + operator*

Add a free function that constructs a `MatrixFunctionView` from any VF
expression:

```cpp
// In tycho::vf namespace
template <class Func>
auto RowMatrix(Func&& f, int rows, int cols)
    -> MatrixFunctionView<std::decay_t<Func>, -1, -1, Eigen::RowMajor>;
```

The returned `MatrixFunctionView` supports:

**`.inverse()` member** — returns a new `MatrixFunctionView` wrapping
`MatrixInverse` composed with self. Dispatches to `MatrixInverse<2>`,
`MatrixInverse<3>`, or `MatrixInverse<-1>` based on runtime matrix size.
Throws if matrix is not square. Matches the Python binding implementation
in `matrix_function_build.cpp` lines 180-200.

**`operator*` for matrix-function product** — `MatrixFunctionView * VF`
returns a VF expression (the matrix applied to a vector). Uses
`MatrixFunctionProduct` internally. Matches the Python `__mul__` binding.

Usage:
```cpp
auto Mvec = stack(cos(theta), l, m1 + m2, m2 * l * cos(theta));
auto M = RowMatrix(Mvec, 2, 2);
auto accel = M.inverse() * Q;  // VF expression, not a matrix
```

### 2. quat_rotate Free Function

Add a free function that rotates a 3-element vector by a 4-element
quaternion (scalar-last convention):

```cpp
// In tycho::vf namespace
template <class QFunc, class VFunc>
auto quat_rotate(QFunc&& q, VFunc&& v);
```

- `q`: VF expression with 4-element output (quaternion, scalar-last)
- `v`: VF expression with 3-element output (vector to rotate)
- Returns: VF expression with 3-element output (rotated vector)

Internally matches the Python binding implementation exactly — uses
`FunctionQuatProduct` directly and `StackedOutputs` for the conjugate:

```cpp
auto qinv = StackedOutputs{q.head<3>() * (-1.0), q.coeff<3>()};
auto vq = v.padded_lower<1>();
auto qvq = FunctionQuatProduct{q, vq};
return FunctionQuatProduct{qvq, qinv}.head<3>();
```

Also add an overload accepting `Eigen::Vector3d` for the vector argument
(wraps in `Constant<>` internally).

### 3. cwise_product Eigen Overloads

Add overloads to the `cwise_product` free function in `binary_math.h` that
accept `Eigen::VectorXd`:

```cpp
// VF x Eigen
template <class Func, int IR, int OR>
auto cwise_product(const DenseFunctionBase<Func, IR, OR>& f,
                   const Eigen::VectorXd& v);

// Eigen x VF
template <class Func, int IR, int OR>
auto cwise_product(const Eigen::VectorXd& v,
                   const DenseFunctionBase<Func, IR, OR>& f);
```

Internally wraps the Eigen vector using `RowScaled<Func>` — the same
approach the Python bindings use in `dense_function_base_bind.h`.

## Verification

Update C++ examples that use workarounds:

- **CartPole** — replace analytic 2x2 inverse with
  `RowMatrix(Mvec, 2, 2).inverse() * Q`
- **OptimalDocking** — replace workaround with `quat_rotate(q, v)`
- **BettsLowThrust** — use `RowMatrix` for RTN rotation if the simplified
  version can now be made more faithful

**Success criteria per example:**
1. Uses the new VF functions matching the Python pattern
2. Same convergence behavior as before
3. No analytic inverse workarounds remaining in CartPole

**No regressions:**
- All C++ unit tests pass
- All C++ example CTests pass
- All Python examples pass

## Files Modified

**New/modified VF headers:**
- `include/tycho/detail/vf/operators/binary_math.h` — `cwise_product` Eigen
  overloads, `quat_rotate` free function
- `include/tycho/detail/vf/operators/matrix_function.h` — `RowMatrix` free
  function, `.inverse()` member on `MatrixFunctionView`, `operator*` for
  matrix-function product (if not already present as member)

**Examples:**
- `examples/cpp_examples/builder/cart_pole/main.cpp`
- `examples/cpp_examples/builder/optimal_docking/main.cpp`
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp`

**Umbrella headers (if new functions need re-export):**
- `include/tycho/vector_functions.h` — may need to include new headers
