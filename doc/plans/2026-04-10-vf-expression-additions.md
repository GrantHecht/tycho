# VF Expression Additions — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `RowMatrix`/`.inverse()`, `quat_rotate`, and `cwise_product` Eigen overloads to the C++ VF expression DSL, exposing existing internal machinery as convenient free functions.

**Architecture:** Each addition composes existing internal types (`MatrixFunctionView`, `MatrixInverse`, `MatrixFunctionProduct`, `FunctionQuatProduct`, `RowScaled`) through new free functions and member methods. No new algorithmic or derivative code is needed — all Jacobian/adjoint machinery is already in the underlying types.

**Tech Stack:** C++20, Eigen, existing VF expression template machinery

**Spec:** `doc/plans/TYCHO_VF_EXPRESSION_ADDITIONS_SPEC.md`

---

## File Structure

**Modified files:**
- `include/tycho/detail/vf/operators/matrix_function.h` — `RowMatrix` free function, `.inverse()` member on `MatrixFunctionView`
- `include/tycho/detail/vf/operators/operator_overloads.h` — `operator*(MatrixFunctionView, DenseFunctionBase)` for matrix-vector product
- `include/tycho/detail/vf/operators/binary_math.h` — `quat_rotate` free function, `cwise_product` Eigen overloads (+ new includes)
- `examples/cpp_examples/builder/cart_pole/main.cpp` — use `RowMatrix().inverse() * Q`
- `examples/cpp_examples/builder/optimal_docking/main.cpp` — use `quat_rotate(q, v)`
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp` — use `RowMatrix` for RTN rotation (if applicable)

---

## Task 0: RowMatrix Free Function + .inverse() Member

**Files:**
- Modify: `include/tycho/detail/vf/operators/matrix_function.h`

- [ ] **Step 1: Read the current file**

Read `include/tycho/detail/vf/operators/matrix_function.h` to understand the `MatrixFunctionView` struct and `MatrixRowsCols` base.

- [ ] **Step 2: Add RowMatrix free function**

Add after the `MatrixFunctionView` struct definition (after line ~33):

```cpp
/// Construct a row-major MatrixFunctionView from any VF expression.
template <class Func>
auto RowMatrix(const DenseFunctionBase<Func, -1, -1> &f, int rows, int cols) {
    return MatrixFunctionView<Func, -1, -1, Eigen::RowMajor>(f.derived(), rows, cols);
}

// Also support derived types with known sizes
template <class Func, int IR, int OR>
auto RowMatrix(const DenseFunctionBase<Func, IR, OR> &f, int rows, int cols) {
    return MatrixFunctionView<Func, -1, -1, Eigen::RowMajor>(f.derived(), rows, cols);
}
```

- [ ] **Step 3: Add .inverse() member to MatrixFunctionView**

Add inside the `MatrixFunctionView` struct, after the constructor:

```cpp
/// Return the matrix inverse as a new MatrixFunctionView.
/// Dispatches to MatrixInverse<2,Major>, <3,Major>, or <-1,Major>.
/// Throws if matrix is not square.
auto inverse() const {
    if (this->matrix_rows_ != this->matrix_cols_) {
        throw std::invalid_argument(
            "MatrixFunctionView::inverse: matrix must be square");
    }
    int size = this->matrix_rows_;
    GenericFunction<-1, -1> inv_func;
    if (size == 2) {
        inv_func = MatrixInverse<2, MMajor>(size);
    } else if (size == 3) {
        inv_func = MatrixInverse<3, MMajor>(size);
    } else {
        inv_func = MatrixInverse<-1, MMajor>(size);
    }
    GenericFunction<-1, -1> composed(inv_func.eval(static_cast<const Func &>(*this)));
    return MatrixFunctionView<GenericFunction<-1, -1>, -1, -1, MMajor>(
        composed, size, size);
}
```

This matches the Python binding implementation exactly (matrix_function_build.cpp lines 180-200). The `MMajor` template parameter ensures the inverse uses the same major order as the original matrix.

**Note:** This requires includes for `MatrixInverse` and `GenericFunction`. Add at the top of matrix_function.h:
```cpp
#include "tycho/detail/vf/operators/matrix_inverse.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
```

- [ ] **Step 4: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/vf/operators/matrix_function.h
git commit -m "feat: add RowMatrix free function and .inverse() member

RowMatrix(func, rows, cols) constructs a row-major MatrixFunctionView.
.inverse() dispatches to MatrixInverse<2/3/-1, Major> and returns a
new MatrixFunctionView. Matches Python vf.RowMatrix().inverse() API."
```

---

## Task 1: operator*(MatrixFunctionView, DenseFunctionBase) for Matrix-Vector Product

**Files:**
- Modify: `include/tycho/detail/vf/operators/operator_overloads.h`

- [ ] **Step 1: Read the current file**

Read `include/tycho/detail/vf/operators/operator_overloads.h` to find the existing matrix `operator*` (lines 368-376).

- [ ] **Step 2: Add matrix-vector operator***

Add after the existing matrix-matrix `operator*` (after line ~376):

```cpp
/// Matrix-vector product: MatrixFunctionView * VF expression.
/// Wraps the VF in a column-vector MatrixFunctionView, then delegates
/// to the existing matrix-matrix operator*.
template <class M1, int M1Rows, int M1Cols, int M1Major,
          class Func, int IR, int OR>
decltype(auto) operator*(const MatrixFunctionView<M1, M1Rows, M1Cols, M1Major> &m,
                         const DenseFunctionBase<Func, IR, OR> &v) {
    // Wrap VF as column vector: N×1, column-major
    auto col_vec = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>(
        v.derived(), m.matrix_cols_, 1);

    using MatFunc1 = MatrixFunctionView<M1, M1Rows, M1Cols, M1Major>;
    using MatFunc2 = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>;

    return MatrixFunctionProduct<MatFunc1, MatFunc2>(m, col_vec);
}
```

- [ ] **Step 3: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/vf/operators/operator_overloads.h
git commit -m "feat: add operator*(MatrixFunctionView, VF) for matrix-vector product

Wraps the VF expression as a column-vector MatrixFunctionView then
delegates to existing matrix-matrix operator*."
```

---

## Task 2: quat_rotate Free Function

**Files:**
- Modify: `include/tycho/detail/vf/operators/binary_math.h`

- [ ] **Step 1: Read the current file**

Read `include/tycho/detail/vf/operators/binary_math.h` to understand existing includes and free functions.

- [ ] **Step 2: Add required includes**

Add at the top of binary_math.h (after the existing include):

```cpp
#include "tycho/detail/vf/operators/vector_products.h"
#include "tycho/detail/vf/expressions/stacked.h"
```

- [ ] **Step 3: Add quat_rotate free function**

Add after the existing `quat_product` function (after line ~30):

```cpp
/// Rotate a 3-element vector by a 4-element quaternion (scalar-last).
/// Computes: q * (v, 0) * q_inv, then extracts the 3-element vector part.
template <class QFunc, int QIR, int QOR, class VFunc, int VIR, int VOR>
auto quat_rotate(const DenseFunctionBase<QFunc, QIR, QOR> &q,
                 const DenseFunctionBase<VFunc, VIR, VOR> &v) {
    auto qinv = StackedOutputs{q.derived().template head<3>() * (-1.0),
                               q.derived().template coeff<3>()};
    auto vq = v.derived().template padded_lower<1>();
    auto qvq = FunctionQuatProduct<QFunc, decltype(vq)>{q.derived(), vq};
    return FunctionQuatProduct<decltype(qvq), decltype(qinv)>{qvq, qinv}
        .template head<3>();
}

/// Overload accepting Eigen::Vector3d for the vector argument.
template <class QFunc, int QIR, int QOR>
auto quat_rotate(const DenseFunctionBase<QFunc, QIR, QOR> &q,
                 const Eigen::Vector3d &v) {
    auto v_const = Constant<-1, 3>(q.derived().input_rows(), v);
    return quat_rotate(q, v_const);
}
```

- [ ] **Step 4: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/vf/operators/binary_math.h
git commit -m "feat: add quat_rotate free function to VF DSL

quat_rotate(q, v) rotates a 3-element vector by a 4-element quaternion
(scalar-last convention). Composes FunctionQuatProduct calls matching
the Python binding implementation. Includes Eigen::Vector3d overload."
```

---

## Task 3: cwise_product Eigen Overloads

**Files:**
- Modify: `include/tycho/detail/vf/operators/binary_math.h`

- [ ] **Step 1: Add cwise_product Eigen overloads**

Add after the existing `cwise_product` function (after line ~42):

```cpp
/// Element-wise product: VF * Eigen::VectorXd.
/// Uses RowScaled internally (same as Python binding).
template <class Func, int IR, int OR>
auto cwise_product(const DenseFunctionBase<Func, IR, OR> &f,
                   const Eigen::VectorXd &v) {
    return RowScaled<Func>(f.derived(), v);
}

/// Element-wise product: Eigen::VectorXd * VF (commutative).
template <class Func, int IR, int OR>
auto cwise_product(const Eigen::VectorXd &v,
                   const DenseFunctionBase<Func, IR, OR> &f) {
    return RowScaled<Func>(f.derived(), v);
}
```

**Note:** This requires including the scaled header. Add at the top of binary_math.h:
```cpp
#include "tycho/detail/vf/scaling/scaled.h"
```

- [ ] **Step 2: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 3: Commit**

```bash
git add include/tycho/detail/vf/operators/binary_math.h
git commit -m "feat: add cwise_product Eigen overloads to VF DSL

cwise_product(VF, VectorXd) and cwise_product(VectorXd, VF) use
RowScaled internally, matching the Python binding approach."
```

---

## Task 4: Update CartPole Example

**Files:**
- Modify: `examples/cpp_examples/builder/cart_pole/main.cpp`

- [ ] **Step 1: Read current CartPole implementation**

Read `examples/cpp_examples/builder/cart_pole/main.cpp` and `examples/python_examples/CartPole.py` to understand the current analytic inverse workaround and the target Python pattern.

- [ ] **Step 2: Replace analytic inverse with RowMatrix().inverse() * Q**

The Python pattern is:
```python
Mvec = vf.stack(vf.cos(theta), l, m1 + m2, m2 * l * vf.cos(theta))
M = vf.RowMatrix(Mvec, 2, 2)
xddot_thetaddot = M.inverse() * Q
```

Replace the current ODE definition to use:
```cpp
auto Mvec = stack(cos(theta), l, m1 + m2, m2 * l * cos(theta));
auto M = RowMatrix(Mvec, 2, 2);
auto accel = M.inverse() * Q;
```

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 cart_pole_builder
./examples/cpp_examples/builder/cart_pole/cart_pole_builder
```
Expected: "PASSED", same convergence as before.

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/cart_pole/main.cpp
git commit -m "refactor: CartPole uses RowMatrix().inverse() * Q instead of analytic inverse"
```

---

## Task 5: Update OptimalDocking Example

**Files:**
- Modify: `examples/cpp_examples/builder/optimal_docking/main.cpp`

- [ ] **Step 1: Read current OptimalDocking implementation**

Read the current C++ example and the Python `examples/python_examples/OptimalDocking.py` to understand how `quat_rotate` is used in the attitude dynamics.

- [ ] **Step 2: Replace workaround with quat_rotate(q, v)**

Replace any manual quaternion rotation construction with `quat_rotate(q, v)`.

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 optimal_docking_builder
./examples/cpp_examples/builder/optimal_docking/optimal_docking_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/optimal_docking/main.cpp
git commit -m "refactor: OptimalDocking uses quat_rotate() for attitude dynamics"
```

---

## Task 6: Update BettsLowThrust Example (if applicable)

**Files:**
- Modify: `examples/cpp_examples/builder/betts_low_thrust/main.cpp`

- [ ] **Step 1: Read current BettsLowThrust implementation**

Read the current C++ example and check if `RowMatrix` can replace any workarounds for RTN rotation matrix construction. The example was simplified (no J2/J3/J4) — check if `RowMatrix` enables a more faithful port.

- [ ] **Step 2: Update if applicable**

If the current implementation has matrix construction workarounds that `RowMatrix` can replace, update them. If the example is already as faithful as it can be without InterpTable (Subsystem 4), skip this task.

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 betts_low_thrust_builder
./examples/cpp_examples/builder/betts_low_thrust/betts_low_thrust_builder
```

- [ ] **Step 4: Commit if changed**

---

## Task 7: Full Verification

- [ ] **Step 1: Build all examples**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j2 cpp_examples_all
```

- [ ] **Step 2: Run CTest**

```bash
ctest -R cpp_example --output-on-failure -j1
```
Expected: all pass.

- [ ] **Step 3: Run C++ unit tests**

```bash
ctest --output-on-failure -j1
```

- [ ] **Step 4: Commit any remaining fixes**
