# Dynamic VF DSL Operator Completeness — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable dynamic VF segments to compose with all VF DSL operators (cross, normalized_power, etc.), add `pack()` for compile-time optimization, and add template segment accessors to ODEArgsProxy.

**Architecture:** Relax 13 operator template signatures in OperatorOverloads.h to accept independent IR/OR parameters with static_assert guards; add `pack()` method on DenseFunctionBase for opt-in type erasure; add template XVec/UVec/PVec overloads to ODEArgsProxy; demonstrate with updated delta3_launch example.

**Tech Stack:** C++20, Eigen, Google Test. Build: CMake preset `linux-clang-release`, `-j6` for test targets, `-j2` for full builds.

**Spec:** `docs/superpowers/specs/2026-03-25-dynamic-vf-operator-completeness-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `include/tycho/detail/DenseFunctionBase.h` | Modify | Declare `pack()` method |
| `include/tycho/detail/OperatorOverloads.h` | Modify | Relax 13 operator signatures + define `pack()` |
| `include/tycho/detail/ode_builder.h` | Modify | Add template XVec/UVec/PVec to ODEArgsProxy |
| `tests/cpp/vector_functions/test_mixed_size_ops.cpp` | Create | Tests for mixed-size operators and `pack()` |
| `tests/cpp/optimal_control/test_ode_builder.cpp` | Modify | Add tests for template ODEArgsProxy overloads |
| `tests/cpp/CMakeLists.txt` | Modify | Register new test file |
| `examples/cpp_examples/builder/delta3_launch/main.cpp` | Modify | Use ODEArguments<-1,-1,-1> with template accessors |
| `examples/cpp_examples/builder/PAIN_POINTS.md` | Modify | Update pain point 1 status |

---

### Task 1: Add `pack()` method to DenseFunctionBase

**Files:**
- Modify: `include/tycho/detail/DenseFunctionBase.h` (declare method, ~line 160)
- Modify: `include/tycho/detail/OperatorOverloads.h` (define method, bottom of file before closing `}`)

`pack()` type-erases a VF expression into `GenericFunction<IR, OR>`, analogous to Eigen's `.eval()`. Because `DenseFunctionBase.h` cannot include `GenericFunction.h` (circular dependency), we declare the method in `DenseFunctionBase.h` (where `GenericFunction` is forward-declared via `ExpressionFwdDeclarations.h:53`) and define it in `OperatorOverloads.h` (which already includes `GenericFunction.h`).

- [ ] **Step 1: Declare `pack()` in DenseFunctionBase.h**

Add near the existing `MakeGeneric()` method (around line 502), or before the closing `};` of the class body (line 1355):

```cpp
    /// Type-erase this expression into a GenericFunction, analogous to
    /// Eigen's .eval().  Use at intermediate points to reduce compile-time
    /// cost of deeply nested expression templates.  Preserves compile-time
    /// input/output sizes.
    GenericFunction<IR, OR> pack() const;
```

- [ ] **Step 2: Define `pack()` in OperatorOverloads.h**

Add just before the closing `} // namespace Tycho` at the bottom of `OperatorOverloads.h`:

```cpp
/////////////////////// pack — type-erase to GenericFunction ///////////////////

template <class Derived, int IR, int OR>
GenericFunction<IR, OR> DenseFunctionBase<Derived, IR, OR>::pack() const {
    return GenericFunction<IR, OR>(this->derived());
}
```

- [ ] **Step 3: Commit**

```bash
git add include/tycho/detail/DenseFunctionBase.h include/tycho/detail/OperatorOverloads.h
git commit -m "feat: add pack() method to DenseFunctionBase for opt-in type erasure"
```

---

### Task 2: Relax arithmetic operator signatures (operators 1–4)

**Files:**
- Modify: `include/tycho/detail/OperatorOverloads.h:202-224`

These are the critical operators that block the pain point. Each gets independent IR/OR template parameters with `static_assert` guards to preserve compile-time safety when both sides are fully static.

- [ ] **Step 1: Relax `operator+` (TwoFunctionSum) at line 202**

Replace:
```cpp
template <class Func1, int IR, int OR, class Func2>
decltype(auto) operator+(const DenseFunctionBase<Func1, IR, OR> &f1,
                         const DenseFunctionBase<Func2, IR, OR> &f2) {
    return TwoFunctionSum<Func1, Func2>(f1.derived(), f2.derived());
}
```

With:
```cpp
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
decltype(auto) operator+(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                         const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator+: input size mismatch — both functions have "
                  "static sizes that don't match");
    static_assert(OR1 == OR2 || OR1 < 0 || OR2 < 0,
                  "VF operator+: output size mismatch — both functions have "
                  "static sizes that don't match");
    return TwoFunctionSum<Func1, Func2>(f1.derived(), f2.derived());
}
```

- [ ] **Step 2: Relax `operator+` (MultiFunctionSum 3-arg) at line 208**

Replace:
```cpp
template <class Func1, int IR, int OR, class Func2, class Func3>
decltype(auto) operator+(const DenseFunctionBase<Func3, IR, OR> &f2,
                         const TwoFunctionSum<Func1, Func2> &f1) {
    return MultiFunctionSum<Func1, Func2, Func3>(f1.func1, f1.func2, f2.derived());
}
```

With:
```cpp
template <class Func1, int IR, int OR, class Func2, class Func3>
decltype(auto) operator+(const DenseFunctionBase<Func3, IR, OR> &f2,
                         const TwoFunctionSum<Func1, Func2> &f1) {
    static_assert(IR == TwoFunctionSum<Func1, Func2>::IRC ||
                      IR < 0 || TwoFunctionSum<Func1, Func2>::IRC < 0,
                  "VF operator+: input size mismatch with sum operand");
    static_assert(OR == TwoFunctionSum<Func1, Func2>::ORC ||
                      OR < 0 || TwoFunctionSum<Func1, Func2>::ORC < 0,
                  "VF operator+: output size mismatch with sum operand");
    return MultiFunctionSum<Func1, Func2, Func3>(f1.func1, f1.func2, f2.derived());
}
```

Note: `TwoFunctionSum<Func1, Func2>` already inherits from `DenseFunctionBase` with `IRC = SZ_MAX<Func1::IRC, Func2::IRC>` and `ORC = SZ_MAX<Func1::ORC, Func2::ORC>`, so the template deduction for the second argument already uses those values. The IR/OR from `Func3` just need to be compatible.

- [ ] **Step 3: Relax `operator+` (MultiFunctionSum 4-arg) at line 213**

Replace:
```cpp
template <class Func1, int IR, int OR, class Func2, class Func3, class Func4>
decltype(auto) operator+(const DenseFunctionBase<Func4, IR, OR> &f2,
                         const MultiFunctionSum<Func1, Func2, Func3> &f1) {
    return MultiFunctionSum<Func1, Func2, Func3, Func4>(f1.func1, f1.func2, std::get<0>(f1.funcs),
                                                        f2.derived());
}
```

With:
```cpp
template <class Func1, int IR, int OR, class Func2, class Func3, class Func4>
decltype(auto) operator+(const DenseFunctionBase<Func4, IR, OR> &f2,
                         const MultiFunctionSum<Func1, Func2, Func3> &f1) {
    static_assert(IR == MultiFunctionSum<Func1, Func2, Func3>::IRC ||
                      IR < 0 || MultiFunctionSum<Func1, Func2, Func3>::IRC < 0,
                  "VF operator+: input size mismatch with sum operand");
    static_assert(OR == MultiFunctionSum<Func1, Func2, Func3>::ORC ||
                      OR < 0 || MultiFunctionSum<Func1, Func2, Func3>::ORC < 0,
                  "VF operator+: output size mismatch with sum operand");
    return MultiFunctionSum<Func1, Func2, Func3, Func4>(f1.func1, f1.func2, std::get<0>(f1.funcs),
                                                        f2.derived());
}
```

- [ ] **Step 4: Relax `operator-` (FunctionDifference) at line 220**

Replace:
```cpp
template <class Func1, int IR, int OR, class Func2>
decltype(auto) operator-(const DenseFunctionBase<Func1, IR, OR> &f1,
                         const DenseFunctionBase<Func2, IR, OR> &f2) {
    return FunctionDifference<Func1, Func2>(f1.derived(), f2.derived());
}
```

With:
```cpp
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
decltype(auto) operator-(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                         const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator-: input size mismatch — both functions have "
                  "static sizes that don't match");
    static_assert(OR1 == OR2 || OR1 < 0 || OR2 < 0,
                  "VF operator-: output size mismatch — both functions have "
                  "static sizes that don't match");
    return FunctionDifference<Func1, Func2>(f1.derived(), f2.derived());
}
```

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/OperatorOverloads.h
git commit -m "feat: relax operator+/- signatures to allow mixed static/dynamic sizes"
```

---

### Task 3: Relax vector*scalar, division, and comparison operators (operators 5–13)

**Files:**
- Modify: `include/tycho/detail/OperatorOverloads.h:177-258`

These operators only need IR relaxation (OR values are already independent or both fixed at 1).

- [ ] **Step 1: Relax `operator*` VFunc * SFunc at line 177**

Replace:
```cpp
template <class VFunc, int IR, int OR, class SFunc>
decltype(auto) operator*(const DenseFunctionBase<VFunc, IR, OR> &vf,
                         const DenseFunctionBase<SFunc, IR, 1> &sf) {
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

With:
```cpp
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<VFunc, IR1, OR> &vf,
                         const DenseFunctionBase<SFunc, IR2, 1> &sf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between vector and scalar functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

- [ ] **Step 2: Relax `operator*` SFunc * VFunc at line 182**

Replace:
```cpp
template <class VFunc, int IR, int OR, class SFunc>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR, 1> &sf,
                         const DenseFunctionBase<VFunc, IR, OR> &vf) {
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

With:
```cpp
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR2, 1> &sf,
                         const DenseFunctionBase<VFunc, IR1, OR> &vf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between scalar and vector functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

- [ ] **Step 3: Relax `operator*` SFunc * SFunc (both OR=1) at line 187**

Replace:
```cpp
template <class VFunc, int IR, class SFunc>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR, 1> &sf,
                         const DenseFunctionBase<VFunc, IR, 1> &vf) {
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

With:
```cpp
template <class VFunc, int IR1, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR2, 1> &sf,
                         const DenseFunctionBase<VFunc, IR1, 1> &vf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between scalar functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

- [ ] **Step 4: Relax `operator/` VFunc / SFunc at line 192**

Replace:
```cpp
template <class VFunc, int IR, int OR, class SFunc>
decltype(auto) operator/(const DenseFunctionBase<VFunc, IR, OR> &vf,
                         const DenseFunctionBase<SFunc, IR, 1> &sf) {
    return VectorScalarFunctionDivision<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

With:
```cpp
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator/(const DenseFunctionBase<VFunc, IR1, OR> &vf,
                         const DenseFunctionBase<SFunc, IR2, 1> &sf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator/: input size mismatch between vector and scalar functions");
    return VectorScalarFunctionDivision<VFunc, SFunc>(vf.derived(), sf.derived());
}
```

- [ ] **Step 5: Relax 5 comparison operators at lines 229–258**

Replace each of the 5 comparison operators (`<`, `<=`, `>`, `>=`, `==`) from the pattern:
```cpp
template <class Func1, int IR, class Func2>
auto operator<(const DenseFunctionBase<Func1, IR, 1> &lhs,
               const DenseFunctionBase<Func2, IR, 1> &rhs) {
```

To:
```cpp
template <class Func1, int IR1, class Func2, int IR2>
auto operator<(const DenseFunctionBase<Func1, IR1, 1> &lhs,
               const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
```

Apply this pattern to all 5: `operator<` (line 229), `operator<=` (line 235), `operator>` (line 241), `operator>=` (line 247), `operator==` (line 253). The VF-vs-double overloads (lines 260+) are unaffected.

- [ ] **Step 6: Commit**

```bash
git add include/tycho/detail/OperatorOverloads.h
git commit -m "feat: relax operator*/÷ and comparison signatures for mixed static/dynamic IR"
```

---

### Task 4: Write tests for mixed-size operators and pack()

**Files:**
- Create: `tests/cpp/vector_functions/test_mixed_size_ops.cpp`
- Modify: `tests/cpp/CMakeLists.txt` (add new file to `TYCHO_TEST_SOURCES`)

- [ ] **Step 1: Create test file**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Mixed static/dynamic size operator tests and pack() tests
//
// Validates that VF DSL operators compose correctly when one operand has
// dynamic sizes (IRC/ORC = -1) and the other has static sizes.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

class MixedSizeOpsTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// operator+ with mixed ORC (the core pain point)
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, DynamicSegmentPlusStaticCrossProduct) {
    // This is the exact scenario from PAIN_POINTS.md #1:
    // ODEArguments<-1,-1,-1> segment (ORC=-1) + CrossProduct (ORC=3)
    auto args = ODEArguments<-1, -1, -1>(7, 3, 0);
    auto R = args.segment(0, 3);  // Segment<-1,-1,-1>, ORC=-1
    auto V = args.segment(3, 3);  // Segment<-1,-1,-1>, ORC=-1

    Eigen::Vector3d omega_val(0.0, 0.0, 1.0);
    auto omega = Constant<-1, 3>(11, omega_val);  // ORC=3

    // This is the line that previously failed to compile
    auto Vr = V + R.cross(omega);

    EXPECT_EQ(Vr.IRows(), 11);
    EXPECT_EQ(Vr.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(11, 100, -5.0, 5.0);
    verify_jacobian_fd(Vr, x);
}

TEST_F(MixedSizeOpsTest, StaticSegmentPlusStaticCrossProduct) {
    // Template segment accessors: ORC=3 + ORC=3 (should still work)
    auto args = ODEArguments<-1, -1, -1>(7, 3, 0);
    auto R = args.head<3>();       // ORC=3
    auto V = args.segment<3>(3);   // ORC=3

    Eigen::Vector3d omega_val(0.0, 0.0, 1.0);
    auto omega = Constant<-1, 3>(11, omega_val);

    auto Vr = V + R.cross(omega);
    EXPECT_EQ(Vr.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(11, 101, -5.0, 5.0);
    verify_jacobian_fd(Vr, x);
}

TEST_F(MixedSizeOpsTest, DynamicSegmentMinusStaticCrossProduct) {
    auto args = ODEArguments<-1, -1, -1>(7, 3, 0);
    auto R = args.segment(0, 3);
    auto V = args.segment(3, 3);

    Eigen::Vector3d omega_val(0.0, 0.0, 1.0);
    auto omega = Constant<-1, 3>(11, omega_val);

    auto Vr = V - R.cross(omega);
    EXPECT_EQ(Vr.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(11, 102, -5.0, 5.0);
    verify_jacobian_fd(Vr, x);
}

///////////////////////////////////////////////////////////////////////////////
// operator* and operator/ with mixed IRC
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, DynamicVectorTimesStaticScalar) {
    // Vector (IRC=-1) * Scalar (IRC=3) — mixed IRC
    auto dyn_args = Arguments<-1>(6);
    auto vec = dyn_args.segment(0, 3);   // IRC=-1, ORC=-1 (3 at runtime)

    auto stat_args = Arguments<6>();
    auto scl = stat_args.coeff<5>();      // IRC=6, ORC=1

    // Both have the same runtime IRows (6) but different compile-time IRC
    auto result = vec * scl;
    EXPECT_EQ(result.IRows(), 6);
    EXPECT_EQ(result.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 103, 0.5, 5.0);
    verify_jacobian_fd(result, x);
}

TEST_F(MixedSizeOpsTest, DynamicVectorDividedByStaticScalar) {
    auto dyn_args = Arguments<-1>(6);
    auto vec = dyn_args.segment(0, 3);

    auto stat_args = Arguments<6>();
    auto scl = stat_args.coeff<5>();

    auto result = vec / scl;
    EXPECT_EQ(result.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 104, 0.5, 5.0);
    verify_jacobian_fd(result, x);
}

///////////////////////////////////////////////////////////////////////////////
// Full ODE-like expression with dynamic args + vector operations
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, FullDynamicODEExpression) {
    // Simulate a simplified rocket ODE with dynamic args
    auto args = ODEArguments<-1, -1, -1>(7, 3, 0);

    // Use dynamic segments (ORC=-1) — the relaxed operators allow this
    auto R = args.segment(0, 3);
    auto V = args.segment(3, 3);
    auto m = args.coeff(6);

    Eigen::Vector3d omega_val(0.0, 0.0, 0.1);
    auto omega = Constant<-1, 3>(11, omega_val);

    auto Vr = V + R.cross(omega);         // dynamic + cross → mixed ORC
    auto grav = (-1.0) * R.normalized();   // normalized preserves dynamic ORC
    auto Rdot = V;
    auto Vdot = grav + Vr / m;            // mixed operations

    auto ode = StackedOutputs{Rdot, Vdot};
    EXPECT_EQ(ode.IRows(), 11);
    EXPECT_EQ(ode.ORows(), 6);

    Eigen::VectorXd x = deterministic_random_vector(11, 105, 1.0, 10.0);
    // Ensure position vector norm > 0 for normalized()
    x[0] = 5.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[6] = 1.0;  // mass > 0
    verify_jacobian_fd(ode, x, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// pack() tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, PackPreservesValue) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto cross_expr = a.cross(b);

    auto packed = cross_expr.pack();  // GenericFunction<6, 3>
    EXPECT_EQ(packed.IRows(), 6);
    EXPECT_EQ(packed.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 106, -5.0, 5.0);
    Eigen::VectorXd fx_orig(3), fx_packed(3);
    fx_orig.setZero();
    fx_packed.setZero();
    cross_expr.compute(x, fx_orig);
    packed.compute(x, fx_packed);

    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(fx_orig[i], fx_packed[i]);
}

TEST_F(MixedSizeOpsTest, PackPreservesJacobian) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto cross_expr = a.cross(b);

    auto packed = cross_expr.pack();

    Eigen::VectorXd x = deterministic_random_vector(6, 107, -5.0, 5.0);
    verify_jacobian_fd(packed, x);
}

TEST_F(MixedSizeOpsTest, PackedExpressionComposable) {
    // pack() result can be composed further
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();

    auto cross_packed = a.cross(b).pack();    // GenericFunction<6, 3>
    auto result = cross_packed + a;           // should compile: both ORC=3

    EXPECT_EQ(result.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 108, -5.0, 5.0);
    verify_jacobian_fd(result, x);
}

TEST_F(MixedSizeOpsTest, PackDynamicSizes) {
    // pack() on a dynamic expression preserves dynamic IR/OR
    auto args = Arguments<-1>(6);
    auto seg = args.segment(0, 3);
    auto packed = seg.pack();  // GenericFunction<-1, -1>
    EXPECT_EQ(packed.IRows(), 6);
    EXPECT_EQ(packed.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 109, -5.0, 5.0);
    Eigen::VectorXd fx(3);
    fx.setZero();
    packed.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], x[0]);
    EXPECT_DOUBLE_EQ(fx[1], x[1]);
    EXPECT_DOUBLE_EQ(fx[2], x[2]);
}

TEST_F(MixedSizeOpsTest, PackAdjointConsistency) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto packed = (a.cross(b) + 2.0 * a).pack();

    Eigen::VectorXd x = deterministic_random_vector(6, 110, -5.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 111, -1.0, 1.0);
    verify_adjoint_consistency(packed, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Regression: static-static mismatch still caught at compile time
// (these are compile-time checks — we just verify the valid cases work)
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, StaticMatchStillWorks) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto sum = a + b;  // ORC=3 + ORC=3 — was and still is valid
    EXPECT_EQ(sum.ORows(), 3);
}

///////////////////////////////////////////////////////////////////////////////
// Comparison operators with mixed IRC
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, ComparisonMixedIR) {
    // Dynamic scalar (IRC=-1) compared with static scalar (IRC=6)
    auto dyn_args = Arguments<-1>(6);
    auto dyn_scl = dyn_args.coeff(0);         // IRC=-1, ORC=1

    auto stat_args = Arguments<6>();
    auto stat_scl = stat_args.coeff<5>();      // IRC=6, ORC=1

    auto cond = dyn_scl > stat_scl;
    EXPECT_EQ(cond.IRows(), 6);

    Eigen::VectorXd x = deterministic_random_vector(6, 112, -5.0, 5.0);
    Eigen::VectorXd fx(1);
    fx.setZero();
    cond.compute(x, fx);
    // Result should be 1.0 if x[0] > x[5], else 0.0
    double expected = (x[0] > x[5]) ? 1.0 : 0.0;
    EXPECT_DOUBLE_EQ(fx[0], expected);
}
```

- [ ] **Step 2: Register test in CMakeLists.txt**

Add `vector_functions/test_mixed_size_ops.cpp` to the `TYCHO_TEST_SOURCES` list in `tests/cpp/CMakeLists.txt`, after the existing `test_cwise_jacobian_stress.cpp` entry:

```
    vector_functions/test_cwise_jacobian_stress.cpp
    vector_functions/test_mixed_size_ops.cpp
```

- [ ] **Step 3: Commit**

```bash
git add tests/cpp/vector_functions/test_mixed_size_ops.cpp tests/cpp/CMakeLists.txt
git commit -m "test: add mixed-size operator and pack() tests"
```

---

### Task 5: Build and run tests for Tasks 1–4

- [ ] **Step 1: Reconfigure with tests enabled**

```bash
cmake --preset linux-clang-release -DBUILD_CPP_TESTS=ON
```

- [ ] **Step 2: Build test executable**

```bash
cd build && ninja -j6 tycho_tests
```

Expected: Clean compile, no errors. If there are template ambiguity issues from the relaxed operators, they will show up here.

- [ ] **Step 3: Run all tests**

```bash
ctest --output-on-failure
```

Expected: All tests pass, including the new `MixedSizeOpsTest.*` tests and all existing tests (no regressions).

- [ ] **Step 4: If any failures, fix and rebuild**

If the relaxed operators cause template ambiguity with existing code, diagnose and add SFINAE constraints. The most likely issue is with the MultiFunctionSum overloads where the TwoFunctionSum argument's IRC/ORC must be cross-checked with the new DenseFunctionBase argument.

---

### Task 6: Add template overloads to ODEArgsProxy

**Files:**
- Modify: `include/tycho/detail/ode_builder.h:28-119` (ODEArgsProxy class)
- Modify: `tests/cpp/optimal_control/test_ode_builder.cpp` (add tests)

- [ ] **Step 1: Add template XVec/UVec/PVec to ODEArgsProxy**

Add after the existing `PVec(int start, int count)` method (around line 111), before the size accessors:

```cpp
    /// Sub-segment of the state vector with compile-time size (0-based within X).
    template <int SZ>
    auto XVec(int start = 0) const {
        if (start < 0 || start + SZ > args_.XVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::XVec<{}>: range [{}, {}) out of X range [0, {})", SZ,
                            start, start + SZ, args_.XVars()));
        return args_.segment<SZ>(start);
    }

    /// Sub-segment of the control vector with compile-time size (0-based within U).
    template <int SZ>
    auto UVec(int start = 0) const {
        if (start < 0 || start + SZ > args_.UVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::UVec<{}>: range [{}, {}) out of U range [0, {})", SZ,
                            start, start + SZ, args_.UVars()));
        return args_.segment<SZ>(args_.XtVars() + start);
    }

    /// Sub-segment of the parameter vector with compile-time size (0-based within P).
    template <int SZ>
    auto PVec(int start = 0) const {
        if (start < 0 || start + SZ > args_.PVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::PVec<{}>: range [{}, {}) out of P range [0, {})", SZ,
                            start, start + SZ, args_.PVars()));
        return args_.segment<SZ>(args_.XtUVars() + start);
    }
```

- [ ] **Step 2: Add tests to test_ode_builder.cpp**

Append at the end of the file:

```cpp
TEST_F(ODEBuilderTest, TemplateXVecCrossProduct) {
    // Template XVec<3> produces ORC=3, enabling cross products in define()
    auto ode = ODEBuilder(7, 3)
                   .define([](auto &args) {
                       auto R = args.template XVec<3>(0);
                       auto V = args.template XVec<3>(3);
                       auto u = args.template UVec<3>();

                       Eigen::Vector3d omega_val(0.0, 0.0, 0.1);
                       auto omega = Constant<-1, 3>(11, omega_val);
                       auto Vr = V + R.cross(omega);

                       auto Rdot = V;
                       auto Vdot = Vr + u;
                       auto mdot_val = Eigen::Matrix<double, 1, 1>::Constant(-0.1);
                       auto mdot = Constant<-1, 1>(11, mdot_val);
                       return StackedOutputs{Rdot, Vdot, mdot};
                   })
                   .build();

    EXPECT_EQ(ode.xvars(), 7);
    EXPECT_EQ(ode.uvars(), 3);
    EXPECT_EQ(ode.function().IRows(), 11);
    EXPECT_EQ(ode.function().ORows(), 7);
}

TEST_F(ODEBuilderTest, TemplateXVecBoundsCheck) {
    ODEArgsProxy proxy(3, 1, 0);
    EXPECT_THROW(proxy.XVec<4>(0), std::invalid_argument);   // 4 > xvars=3
    EXPECT_THROW(proxy.XVec<2>(2), std::invalid_argument);   // 2+2 > 3
    EXPECT_NO_THROW(proxy.XVec<3>(0));                        // exact fit
    EXPECT_NO_THROW(proxy.XVec<2>(1));                        // 1+2 = 3, ok
}

TEST_F(ODEBuilderTest, TemplateUVecBoundsCheck) {
    ODEArgsProxy proxy(3, 2, 0);
    EXPECT_THROW(proxy.UVec<3>(0), std::invalid_argument);   // 3 > uvars=2
    EXPECT_NO_THROW(proxy.UVec<2>(0));                        // exact fit
}

TEST_F(ODEBuilderTest, TemplatePVecBoundsCheck) {
    ODEArgsProxy proxy(3, 1, 2);
    EXPECT_THROW(proxy.PVec<3>(0), std::invalid_argument);   // 3 > pvars=2
    EXPECT_NO_THROW(proxy.PVec<2>(0));                        // exact fit
}
```

- [ ] **Step 3: Build and run**

```bash
cd build && ninja -j6 tycho_tests && ctest --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/ode_builder.h tests/cpp/optimal_control/test_ode_builder.cpp
git commit -m "feat: add template XVec/UVec/PVec overloads to ODEArgsProxy"
```

---

### Task 7: Update delta3_launch example

**Files:**
- Modify: `examples/cpp_examples/builder/delta3_launch/main.cpp`

Replace the `make_rocket_ode()` function to use `ODEArguments<-1,-1,-1>` with template segment accessors instead of `Arguments<11>()`.

- [ ] **Step 1: Replace make_rocket_ode()**

Replace the function body (lines 111–152) with:

```cpp
RuntimeODE make_rocket_ode(double T, double mdot) {
    // Dynamic args with template segment accessors: dynamic shell, static core.
    // head<3>() / segment<3>() produce ORC=3, enabling cross/normalized_power.
    auto args = ODEArguments<-1, -1, -1>(7, 3, 0);
    auto R = args.head<3>();
    auto V = args.segment<3>(3);
    auto m = args.coeff(6);
    auto u = args.tail<3>().normalized();

    auto h = R.norm() - Re;

    // Atmospheric density
    auto exp_neg_h = exp(h * (-1.0 / h_scale));

    // Relative velocity: Vr = V + R x [0, 0, We]
    Eigen::Vector3d omega_val(0.0, 0.0, We);
    auto omega = Constant<-1, 3>(11, omega_val);
    auto Vr = V + R.cross(omega);

    // Drag
    const double drag_rho = -0.5 * CD * S * RhoAir;
    auto D = (drag_rho * exp_neg_h) * (Vr * Vr.norm());

    auto Rdot = V;
    auto Vdot = (-mu) * R.normalized_power<3>() + (T * u + D) / m;

    // Constant mass flow rate
    Eigen::Matrix<double, 1, 1> neg_mdot_val;
    neg_mdot_val[0] = -mdot;
    auto neg_mdot_vf = Constant<-1, 1>(11, neg_mdot_val);

    auto ode_expr = StackedOutputs{Rdot, Vdot, neg_mdot_vf};

    return ODEBuilder(7, 3)
        .from(ode_expr)
        .var_names({{"m", 6}, {"t", 7}})
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_group("u", 8, 3)
        .build();
}
```

Also remove the "PAIN POINT" comment block that was at lines 112–114.

- [ ] **Step 2: Build and verify convergence**

```bash
cd build && ninja -j6 delta3_launch_builder
./examples/cpp_examples/builder/delta3_launch/delta3_launch_builder
```

Expected: "Optimal Solution Found" from PSIOPT, final mass near the static version's result.

- [ ] **Step 3: Commit**

```bash
git add examples/cpp_examples/builder/delta3_launch/main.cpp
git commit -m "feat: delta3_launch uses ODEArguments<-1,-1,-1> with template accessors"
```

---

### Task 8: Update PAIN_POINTS.md and full validation

**Files:**
- Modify: `examples/cpp_examples/builder/PAIN_POINTS.md`

- [ ] **Step 1: Update pain point 1 status**

Replace the entire Pain Point 1 section (lines 6–33) with:

```markdown
## Pain Point 1: Dynamic VF DSL operators are incomplete — RESOLVED

**Status: Fixed** — Three complementary fixes address this:

1. **Template segment accessors** (`head<3>()`, `segment<3>(start)`, `tail<3>()`)
   on dynamic `ODEArguments<-1,-1,-1>` produce segments with static output sizes
   (ORC=3), enabling composition with `cross()`, `normalized_power<N>()`, etc.

2. **Relaxed operator signatures** — `operator+`, `operator-`, `operator*`,
   `operator/`, and comparison operators now accept operands with mixed
   static/dynamic compile-time sizes. `static_assert` guards preserve
   compile-time error detection when both operands are fully static.

3. **`pack()` method** — `DenseFunctionBase::pack()` type-erases an expression
   into `GenericFunction<IR, OR>` (like Eigen's `.eval()`), giving users
   opt-in control over compile-time complexity for deeply nested expressions.

Template overloads `XVec<SZ>()`, `UVec<SZ>()`, `PVec<SZ>()` were also added
to `ODEArgsProxy` for use in the `define()` lambda path.

The Delta3Launch example now uses `ODEArguments<-1,-1,-1>` with template
segment accessors:

```cpp
auto args = ODEArguments<-1, -1, -1>(7, 3, 0);
auto R = args.head<3>();           // ORC=3
auto V = args.segment<3>(3);      // ORC=3
auto Vr = V + R.cross(omega);     // compiles — both ORC=3
```
```

- [ ] **Step 2: Run full C++ test suite**

```bash
cd build && ninja -j6 tycho_tests && ctest --output-on-failure
```

- [ ] **Step 3: Run all builder examples**

```bash
cd build && ninja -j6 brachistochrone_builder hypersens_builder zermelo_builder delta3_launch_builder
./examples/cpp_examples/builder/brachistochrone/brachistochrone_builder
./examples/cpp_examples/builder/hypersens/hypersens_builder
./examples/cpp_examples/builder/zermelo/zermelo_builder
./examples/cpp_examples/builder/delta3_launch/delta3_launch_builder
```

Expected: All converge successfully.

- [ ] **Step 4: Run static C++ brachistochrone example**

```bash
./examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found" with objective near 1.8013 s.

- [ ] **Step 5: Run Python examples**

```bash
conda activate tycho
cd build && ninja -j2 all
python ../scripts/run_examples.py
```

Expected: Exit code 0, all 38 examples pass.

- [ ] **Step 6: Format changed files**

```bash
cd build && ninja clang-format
```

- [ ] **Step 7: Commit and verify clean status**

```bash
git add examples/cpp_examples/builder/PAIN_POINTS.md
git commit -m "docs: update PAIN_POINTS.md — mark pain point 1 as resolved"
```
