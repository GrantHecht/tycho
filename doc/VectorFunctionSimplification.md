# VectorFunction CRTP Simplification Plan

This document outlines three independent directions for simplifying the
VectorFunction CRTP hierarchy. They can be done in any order but there is a
natural sequence: concepts first, macro cleanup second, chain flattening last.

---

## Current Hierarchy (8 levels)

```
CRTPBase<Derived>                                          // derived(), name(), cast(), copy()
  → ComputableBase<Derived, IR, OR>                        // IO sizes, static flags, compute(), constraints()
    → Computable<Derived, IR, OR>                          // forwards 3 type aliases — nothing else
      → DenseFunctionBase<Derived, IR, OR>                 // Jacobian/Hessian, expression ops, domain products
        → DenseFunction<Derived, IR, OR>                   // routes OR==1 → DenseScalarFunctionBase
          → DenseFirstDerivatives<Derived, IR, OR, JMode>  // specialization injection point
            → DenseSecondDerivatives<..., JMode, HMode>    // specialization injection point
              → DenseDerivatives<..., JMode, HMode>        // VF_TYPE_ALIASES only
                → VectorFunction<..., JMode, HMode>        // VF_TYPE_ALIASES only
```

### What each level contributes

| Level | Genuine content? | Could be eliminated? |
|-------|-----------------|---------------------|
| `CRTPBase` | 7 trivial utility methods | Yes — inline into ComputableBase |
| `ComputableBase` | **Core** — IO, flags, compute, constraints | No |
| `Computable` | **Nothing** — 3 forwarded type aliases | Yes — identical to Base |
| `DenseFunctionBase` | **Core** — Jacobian/Hessian, expression DSL | No |
| `DenseFunction` | Routes OR==1 → DenseScalarFunctionBase | Yes — use `requires` instead |
| `DenseScalarFunctionBase` | objective(), objective_gradient(), objective_gradient_hessian() | Merge into DenseFunctionBase with constraints |
| `DenseFirstDerivatives` | Specialization point for Jacobian mode | No |
| `DenseSecondDerivatives` | Specialization point for Hessian mode | No |
| `DenseDerivatives` | Only `DENSE_FUNCTION_BASE_TYPES(Base)` | Yes — merge into VectorFunction |
| `VectorFunction` | Only `DENSE_FUNCTION_BASE_TYPES(Base)` | No — this is the public name |

---

## Direction 1: Flatten the CRTP Chain (8 → 5 levels)

### Target hierarchy

```
ComputableBase<Derived, IR, OR>                              // absorbs CRTPBase + Computable
  → DenseFunctionBase<Derived, IR, OR>                       // absorbs DenseScalarFunctionBase via requires
    → DenseFirstDerivatives<Derived, IR, OR, JMode>          // unchanged
      → DenseSecondDerivatives<Derived, IR, OR, JMode, HMode> // unchanged
        → VectorFunction<Derived, IR, OR, JMode, HMode>       // absorbs DenseDerivatives
```

Every remaining level has genuine content. The derivative specialization mechanism
is preserved exactly as-is.

### 1a. Inline `CRTPBase` into `ComputableBase`

```cpp
// Before: computable_base.h
template <class Derived, int IR, int OR>
struct ComputableBase : tycho::utils::CRTPBase<Derived>, InputOutputSize<IR, OR> {
    // ...
};

// After: computable_base.h (no CRTPBase dependency)
template <class Derived, int IR, int OR>
struct ComputableBase : InputOutputSize<IR, OR> {
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
    std::string name() const { return type_name<Derived>(); }
    template <class T> T cast() const { return T(this->derived()); }
    template <class T> T copy() const { return T(this->derived()); }

    // ... rest of ComputableBase unchanged ...
};
```

The rarely-used `deep_copy_into`, `find`, `copy_if_same`, `forward_if_same` can be
dropped or moved to a utility — they are not called from the base class chain.

`CRTPMixin` is still needed for the mixin pattern (used by `DomainHolder`), so keep
it as a standalone in `crtp_base.h`.

Risk: **Very low.** Pure mechanical inlining.

### 1b. Eliminate `Computable`

The primary template and the `OR==1` specialization both just forward 3 type
aliases that are already on `ComputableBase`. Change `DenseFunctionBase` to inherit
directly:

```cpp
// Before: dense_function_base.h
template <class Derived, int IR, int OR>
struct DenseFunctionBase : Computable<Derived, IR, OR>, DomainHolder<IR> {

// After: dense_function_base.h
template <class Derived, int IR, int OR>
struct DenseFunctionBase : ComputableBase<Derived, IR, OR>, DomainHolder<IR> {
```

Delete `computable.h`.

Risk: **Very low.** `Computable` had no content.

### 1c. Eliminate `DenseFunction` + `DenseScalarFunctionBase` — use `requires`

The entire purpose of `DenseFunction<Derived, IR, OR>` is to route `OR==1` to
`DenseScalarFunctionBase`, which adds three methods: `objective()`,
`objective_gradient()`, `objective_gradient_hessian()`.

With C++20 constraints, these can live directly on `DenseFunctionBase`:

```cpp
// In dense_function_base.h, inside the DenseFunctionBase class body:

    void objective(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double& Val,
                   const tycho::solvers::SolverIndexingData& data) const
        requires (OR == 1)
    {
        Input<double> x(this->input_rows());
        Output<double> fx(1);
        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            fx.setZero();
            this->derived().compute(x, fx);
            Val += fx[0] * ObjScale;
        }
    }

    void objective_gradient(/* ... */) const requires (OR == 1) { /* ... */ }
    void objective_gradient_hessian(/* ... */) const requires (OR == 1) { /* ... */ }
```

Then `DenseFirstDerivatives` inherits from `DenseFunctionBase` directly:

```cpp
// Before: dense_derivatives.h
template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseFirstDerivatives : DenseFunction<Derived, IR, OR> {

// After:
template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseFirstDerivatives : DenseFunctionBase<Derived, IR, OR> {
```

Delete `dense_function.h` and `dense_scalar_function_base.h`.

Risk: **Low-medium.** The `requires` approach is semantically identical to the
partial-specialization routing. Any code explicitly naming `DenseFunction<X,Y,Z>`
or `DenseScalarFunctionBase<X,Y>` as a type would break — but these are not part
of the public API; only `VectorFunction` is.

### 1d. Eliminate `DenseDerivatives` — merge into `VectorFunction`

```cpp
// Before: vector_function.h
template <class Derived, int IR, int OR, DenseDerivativeMode Jm, DenseDerivativeMode Hm>
struct VectorFunction : DenseDerivatives<Derived, IR, OR, Jm, Hm> {
    using Base = DenseDerivatives<Derived, IR, OR, Jm, Hm>;
    DENSE_FUNCTION_BASE_TYPES(Base)
};

// After: vector_function.h
template <class Derived, int IR, int OR,
          DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct VectorFunction : DenseSecondDerivatives<Derived, IR, OR, Jm, Hm> {
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jm, Hm>;
    DENSE_FUNCTION_BASE_TYPES(Base)
};
```

Delete the `DenseDerivatives` class from `dense_derivatives.h` (keep the enum and
the First/Second templates).

Risk: **Very low.** `DenseDerivatives` had no content.

### Files affected

| What | Files touched | Risk |
|------|--------------|------|
| Inline CRTPBase | `computable_base.h`, `crtp_base.h` (keep for CRTPMixin) | Very low |
| Delete Computable | `computable.h` (delete), `dense_function_base.h` (re-parent) | Very low |
| Merge DenseScalarFunctionBase | `dense_function_base.h`, delete `dense_function.h` + `dense_scalar_function_base.h` | Low-medium |
| Merge DenseDerivatives | `vector_function.h`, `dense_derivatives.h` | Very low |

---

## Direction 2: Replace `DENSE_FUNCTION_BASE_TYPES` Macro

### What the macro does

The macro creates 13 template aliases in every class that uses it. These fall into
two categories:

**Category A — Fixed aliases (8).** Same everywhere, no dependency on IR/OR/Base:

```cpp
template <class T> using ConstVectorBaseRef   = const Eigen::MatrixBase<T>&;
template <class T> using VectorBaseRef        = Eigen::MatrixBase<T>&;
template <class T> using ConstMatrixBaseRef   = const Eigen::MatrixBase<T>&;
template <class T> using MatrixBaseRef        = Eigen::MatrixBase<T>&;
template <class T> using ConstEigenBaseRef    = const Eigen::EigenBase<T>&;
template <class T> using EigenBaseRef         = Eigen::EigenBase<T>&;
template <class T> using ConstDiagonalBaseRef = const Eigen::DiagonalBase<T>&;
template <class T> using DiagonalBaseRef      = Eigen::DiagonalBase<T>&;
```

**Category B — Dependent aliases (5).** Depend on IR/OR, must be propagated:

```cpp
template <class Scalar> using Output   = Eigen::Matrix<Scalar, OR, 1>;
template <class Scalar> using Input    = Eigen::Matrix<Scalar, IR, 1>;
template <class Scalar> using Gradient = Eigen::Matrix<Scalar, IR, 1>;
template <class Scalar> using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
template <class Scalar> using Hessian  = Eigen::Matrix<Scalar, IR, IR>;
```

### 2a. Move fixed aliases to namespace scope

```cpp
// New file: include/tycho/detail/vf/core/eigen_ref_aliases.h
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace tycho::vf {

template <class T> using CVecRef  = const Eigen::MatrixBase<T>&;
template <class T> using VecRef   = Eigen::MatrixBase<T>&;
template <class T> using CMatRef  = const Eigen::MatrixBase<T>&;
template <class T> using MatRef   = Eigen::MatrixBase<T>&;
template <class T> using CEigRef  = const Eigen::EigenBase<T>&;
template <class T> using EigRef   = Eigen::EigenBase<T>&;
template <class T> using CDiagRef = const Eigen::DiagonalBase<T>&;
template <class T> using DiagRef  = Eigen::DiagonalBase<T>&;

} // namespace tycho::vf
```

These carry no semantic meaning tied to a particular VectorFunction's dimensions.
Having them at namespace scope makes them available everywhere without propagation.

### 2b. Reduce the macro to only the 5 dependent aliases

```cpp
// function_type_def_macros.h
#define VF_TYPE_ALIASES(Base)                                                      \
    template <class Scalar> using Output   = typename Base::template Output<Scalar>; \
    template <class Scalar> using Input    = typename Base::template Input<Scalar>;  \
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>; \
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>; \
    template <class Scalar> using Hessian  = typename Base::template Hessian<Scalar>;
```

### Why not eliminate the 5-alias macro entirely?

The fundamental C++ limitation: template aliases from a dependent base class cannot
be inherited via `using`. There is no way to write `using Base::template Output;`.
P1945 would fix this, but it is not in any standard yet.

A 5-line macro named `VF_TYPE_ALIASES` is reasonable — it is clear, grep-able, and
the alternative is writing `typename Base::template Output<Scalar>` dozens of times
per file.

### 2c. Update all call sites

In every class and method body:

```cpp
// Before
template <class InType, class OutType>
inline void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {

// After
template <class InType, class OutType>
inline void compute(CVecRef<InType> x, CVecRef<OutType> fx_) const {
```

### 2d. Eliminate `SUB_FUNCTION_IO_TYPES` entirely

This macro generates aliases like `Func1_Output<Scalar>`, `Func1_jacobian<Scalar>`.
Replace with direct qualified access:

```cpp
// Before (in NestedFunction_Impl)
SUB_FUNCTION_IO_TYPES(OuterFunc)
SUB_FUNCTION_IO_TYPES(InnerFunc)
// ...
OuterFunc_Output<Scalar> fx_outer;
InnerFunc_Output<Scalar> fx_inner;

// After — direct qualified access
typename OuterFunc::template Output<Scalar> fx_outer;
typename InnerFunc::template Output<Scalar> fx_inner;

// Or with local aliases inside each method body:
template <class InType, class OutType>
void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
    using Scalar = typename InType::Scalar;
    using OuterOut = typename OuterFunc::template Output<Scalar>;
    using InnerOut = typename InnerFunc::template Output<Scalar>;
    // ...
}
```

More verbose per-use, but eliminates the macro and its confusing `Func##_`
concatenation.

### Summary

| Change | Impact |
|--------|--------|
| 8 fixed ref-type aliases → namespace scope | Never need propagation again |
| Rename macro `DENSE_FUNCTION_BASE_TYPES` → `VF_TYPE_ALIASES` | 5 lines instead of 13 |
| Eliminate `SUB_FUNCTION_IO_TYPES` | Remove macro entirely, use qualified types |
| Update all method signatures | `ConstVectorBaseRef<T>` → `CVecRef<T>` |

Risk: **Low.** Purely mechanical rename/move.

---

## Direction 3: Concepts for Leaf Type Constraints

### Current state

Flags are defined as `static const bool` in `ComputableBase`, overridden in leaf
types, and queried in two inconsistent ways:

```cpp
// Way 1: Direct access (in dense_function_base.h)
if constexpr (Derived::is_linear_function) { ... }

// Way 2: Via concepts (in computable_base.h)
if constexpr (!Vectorizable<Derived>) { ... }
```

The concepts already exist in `vector_function_concepts.h` but are used
inconsistently.

### 3a. `static const bool` → `static constexpr bool`

Trivial modernization. `static const bool` in a class template is a C++03 pattern.
`static constexpr bool` is the C++11+ equivalent with stronger guarantees.

```cpp
// Before (computable_base.h)
static const bool is_vectorizable = false;
static const bool is_linear_function = false;

// After
static constexpr bool is_vectorizable = false;
static constexpr bool is_linear_function = false;
```

Same in all leaf types that override these.

### 3b. Add missing concepts, use them consistently

Add concepts for the flags that currently lack them:

```cpp
// vector_function_concepts.h — add:
template <typename T>
concept ConditionalVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_conditional); };

template <typename T>
concept CwiseVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_cwise_operator); };

template <typename T>
concept HasDiagonalHess =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::has_diagonal_hessian); };
```

Then unify all `if constexpr` branches:

```cpp
// Before (dense_function_base.h)
if constexpr (!Derived::is_vectorizable) {
if constexpr (Derived::is_linear_function) {

// After
if constexpr (!Vectorizable<Derived>) {
if constexpr (LinearVF<Derived>) {
```

Same codegen, better consistency.

### 3c. Add a `DenseVectorFunction` concept for type erasure

```cpp
template <typename T>
concept DenseVectorFunction = requires(const T& t) {
    // Size interface
    { T::IRC } -> std::convertible_to<int>;
    { T::ORC } -> std::convertible_to<int>;
    { t.input_rows() } -> std::same_as<int>;
    { t.output_rows() } -> std::same_as<int>;

    // Trait flags
    { T::is_vectorizable } -> std::convertible_to<bool>;
    { T::is_linear_function } -> std::convertible_to<bool>;

    // Type aliases exist
    typename T::template Output<double>;
    typename T::template Input<double>;
    typename T::template Jacobian<double>;
    typename T::INPUT_DOMAIN;
};
```

Use it to constrain the type-erasure entry point:

```cpp
// gf_type_erasure.h
template <int IR, int OR, DenseVectorFunction T>
    requires (T::IRC == IR || IR == -1) && (T::ORC == OR || OR == -1)
struct GFModel final : GFModelCommon<IR, OR, T> { /* ... */ };
```

And on the `GenericFunction` constructor:

```cpp
template <int IR, int OR>
struct GenericFunction : VectorFunction<GenericFunction<IR, OR>, IR, OR> {
    template <DenseVectorFunction T>
        requires (T::IRC == IR || IR == -1) && (T::ORC == OR || OR == -1)
    GenericFunction(T func) { /* ... */ }
};
```

This gives dramatically better error messages when users pass the wrong type.

### What NOT to do: external traits class

Extracting all flags into an external `VFTraits<T>` specialization point was
considered but rejected. The current pattern where expression types compute their
flags locally:

```cpp
static constexpr bool is_linear_function =
    OuterFunc::is_linear_function && InnerFunc::is_linear_function;
```

...is already clean, local, and obvious. An external traits class would scatter
this logic across specializations with no benefit.

### Summary

| Change | Lines affected | Risk |
|--------|---------------|------|
| `static const` → `static constexpr` | ~50 declarations across leaf types | None |
| Unify `if constexpr` to use concepts | ~30 branches in dense_function_base.h | None |
| Add missing concepts | 3 new concepts in vector_function_concepts.h | None |
| Add `DenseVectorFunction` concept | 1 new concept, constrain GFModel + GenericFunction | Low |

---

## Recommended Sequence

1. **Direction 3** (concepts) — smallest diff, zero risk, immediate consistency win
2. **Direction 2** (macro cleanup) — medium diff, mechanical, no semantic change
3. **Direction 1** (flatten chain) — largest diff, benefits from cleanup in 1 and 2

### End state

```cpp
// 5-level chain (down from 8)
ComputableBase<Derived, IR, OR>         // IO, flags, compute(), constraints()
  → DenseFunctionBase<Derived, IR, OR>  // Jacobian/Hessian, DSL, objective() requires OR==1
    → DenseFirstDerivatives<...>        // Jacobian mode specialization
      → DenseSecondDerivatives<...>     // Hessian mode specialization
        → VectorFunction<...>           // Public name, VF_TYPE_ALIASES

// A typical leaf type (unchanged user-facing API):
struct MyODE : VectorFunction<MyODE, 7, 6, DenseDerivativeMode::AutodiffFwd> {
    static constexpr bool is_vectorizable = true;

    void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // ...
    }
};
```

---

## Why Not Deducing This (C++23)?

The `Derived` CRTP template parameter serves three roles simultaneously:

| Role | Example | Deducing this helps? |
|------|---------|---------------------|
| Runtime dispatch | `this->derived().compute_impl(...)` | Yes |
| Compile-time traits | `Derived::is_vectorizable`, `Derived::is_linear_function` | **No** |
| Expression type construction | `EVALOP<Func>::make_nested(this->derived(), ...)` | **No** |

Because `Derived` is used for type-level computation (trait queries, expression
template construction via `EVALOP`, `FWDOP`, `SEGMENTOP` aliases), it cannot be
removed from the class template parameter list. Deducing this would only simplify
the dispatch calls (`self.compute_impl()` vs `this->derived().compute_impl()`),
which is a marginal improvement that does not justify a C++23 requirement.

## Why Not C++20 Modules?

Eigen has zero module support and no timeline for adding it. Every Tycho header
touches Eigen types. Until Eigen ships module interfaces, modules are not viable.
The current clean umbrella-header structure already mirrors the module boundaries
that would be used in a future migration — no prep work needed.
