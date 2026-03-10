# VectorFunction Developer Guide

This document provides a comprehensive, bottom-up explanation of Tycho's VectorFunction system -- the core abstraction for defining dynamics, constraints, and objectives throughout the library. After reading this guide, you should be able to understand, use, and extend VectorFunctions at every level: from calling the Python API, to implementing new C++ function types, to understanding how the optimizer consumes them.

## Table of Contents

1. [What Is a VectorFunction?](#1-what-is-a-vectorfunction)
2. [The CRTP Inheritance Hierarchy](#2-the-crtp-inheritance-hierarchy)
3. [Template Parameters](#3-template-parameters)
4. [Implementing a VectorFunction (C++)](#4-implementing-a-vectorfunction-c)
5. [Input Domains](#5-input-domains)
6. [Derivative Computation](#6-derivative-computation)
7. [DefaultSuperScalar and Vectorized Evaluation](#7-defaultsuperscalar-and-vectorized-evaluation)
8. [Built-In Function Types](#8-built-in-function-types)
9. [Expression Composition (Operator Overloads)](#9-expression-composition-operator-overloads)
10. [Type Erasure: GenericFunction](#10-type-erasure-genericfunction)
11. [The PSIOPT Solver Interface](#11-the-psiopt-solver-interface)
12. [Python API](#12-python-api)
13. [End-to-End Examples](#13-end-to-end-examples)
14. [Reference: Complete Class Hierarchy Diagram](#14-reference-complete-class-hierarchy-diagram)

---

## 1. What Is a VectorFunction?

A **VectorFunction** is a differentiable mapping from an input vector of dimension `IR` (Input Rows) to an output vector of dimension `OR` (Output Rows):

```
f : R^IR --> R^OR
```

Everything in Tycho's problem definition layer is expressed as a VectorFunction:
- ODE right-hand sides (dynamics)
- Equality and inequality constraints
- Objective functions (scalar: OR=1)
- Intermediate expressions (norms, cross products, etc.)

VectorFunctions carry their own **derivative machinery**. They can compute:
- `f(x)` -- the function value
- `J(x)` -- the Jacobian matrix (OR x IR)
- `g(x, lambda)` -- the adjoint gradient `J^T * lambda` (IR x 1)
- `H(x, lambda)` -- the adjoint Hessian `sum_i lambda_i * H_i(x)` (IR x IR)

These derivatives can be computed analytically (hand-derived), via automatic differentiation (the `autodiff` library), or via finite differences.

**Key design principle:** VectorFunctions are compiled C++ expression templates. When you write `vf.sin(theta) * v` in Python, no computation happens -- you're building a C++ expression tree that will be evaluated later at full native speed by the optimizer.

---

## 2. The CRTP Inheritance Hierarchy

The entire class hierarchy uses the **Curiously Recurring Template Pattern (CRTP)**, which means each base class is parameterized by its derived type. This enables static (compile-time) polymorphism with zero virtual function overhead.

```
InputOutputSize<IR, OR>                     // Stores input/output sizes
    |
CRTPBase<Derived>                           // Provides derived() cast
    |
    v
ComputableBase<Derived, IR, OR>             // compute() dispatch + solver interface
    |
    v
Computable<Derived, IR, OR>                 // Type aliases (specialization for OR=1)
    |
    v
DomainHolder<IR>                            // Input domain tracking
    +
DenseFunctionBase<Derived, IR, OR>          // Jacobian ops, indexing, math methods
    |                                       // (OR=1 specialization: DenseScalarFunctionBase)
    v
DenseFunction<Derived, IR, OR>              // Routing layer (OR=1 -> scalar base)
    |
    v
DenseFirstDerivatives<Derived, IR, OR, Jm> // 1st derivative strategy
    |
    v
DenseSecondDerivatives<Derived, IR, OR, Jm, Hm> // 2nd derivative strategy
    |
    v
DenseDerivatives<Derived, IR, OR, Jm, Hm>  // Aggregates derivative layers
    |
    v
VectorFunction<Derived, IR, OR, Jm, Hm>    // ** Top-level base for user types **
```

Every concrete function (e.g., `Segment`, `Scaled`, `Norm`) inherits from `VectorFunction` and provides its own `compute_impl`, `compute_jacobian_impl`, etc.

### The DenseFunction OR=1 Specialization

When a function has scalar output (OR=1), `DenseFunction` routes inheritance through `DenseScalarFunctionBase` instead of `DenseFunctionBase`. This adds objective-specific methods (`objective()`, `objective_gradient()`, `objective_gradient_hessian()`) that the PSIOPT optimizer calls directly.

```cpp
// OR > 1: vector output -> full DenseFunctionBase
template <class Derived, int IR, int OR>
struct DenseFunction : DenseFunctionBase<Derived, IR, OR> {};

// OR == 1: scalar output -> DenseScalarFunctionBase (adds objective interface)
template <class Derived, int IR>
struct DenseFunction<Derived, IR, 1> : DenseScalarFunctionBase<Derived, IR> {};
```

### Why CRTP? Design Rationale and Alternatives

The CRTP hierarchy is an **implementation choice**, not a mathematical or architectural necessity. Understanding *why* it was chosen (and what the alternatives are) helps when reading or extending the code.

#### The Core Constraint: Templated Evaluation

VectorFunctions must be callable with **multiple scalar types** -- `double`, `dual<double>` (for autodiff Jacobians), `dual<dual<double>>` (for Hessians), and `DefaultSuperScalar` (for batched evaluation). The `compute_impl`, `compute_jacobian_impl`, etc. are all **function templates** parameterized on the scalar type:

```cpp
template <class InType, class OutType>
inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const;
```

Virtual functions cannot be templates in C++. So the two main architectural options are:

1. **CRTP (static polymorphism)** -- zero-overhead dispatch via `this->derived().compute_impl(...)`, resolved at compile time.
2. **Virtual functions with type erasure** -- define a fixed set of virtual methods for each scalar type, or use `std::any`/`void*` to erase the scalar type at runtime.

#### Why CRTP Was Chosen

- **Zero overhead:** No vtable indirection on the hot path. The compiler inlines `compute_impl` through the entire expression tree.
- **Template-compatible:** Each function naturally works with any `Scalar` type (double, dual, SuperScalar) through the same code path.
- **Composability:** Expression types like `NestedFunction<Outer, Inner>` carry the exact types of their operands, so the compiler can see through the entire expression tree and optimize globally.

#### What CRTP Costs

- **Compile times:** Each unique expression tree is a unique C++ type. Deep expressions produce deeply nested template types, which are slow to compile.
- **No runtime polymorphism:** You can't put a `Scaled<Segment<6,3,0>>` and a `Norm<3>` into the same `std::vector` without type erasure.
- **Code complexity:** The inheritance chain (`ComputableBase` -> `Computable` -> `DenseFunctionBase` -> `DenseFunction` -> `DenseFirstDerivatives` -> `DenseSecondDerivatives` -> `DenseDerivatives` -> `VectorFunction`) is deep and hard to navigate.

#### The Hybrid Approach Tycho Actually Uses

Tycho doesn't rely on CRTP alone. It uses a **two-layer architecture**:

1. **CRTP layer** -- for building and evaluating expressions at full speed during the hot loop (solver iterations).
2. **Type erasure layer** -- `GenericFunction<IR, OR>` wraps any CRTP expression into a uniform runtime type using `rubber_types::TypeErasure`, with virtual dispatch (see [Section 10](#10-type-erasure-genericfunction)).

The type erasure boundary is where CRTP expressions enter the solver. When you call `phase.addEqualCon(...)` or `vf.stack(...)`, the concrete expression type is erased into a `GenericFunction`. From that point on, the solver calls through virtual dispatch -- but the virtual call is at the *function* level, not the *element* level. Inside each virtual call, the full CRTP expression tree is inlined.

#### Alternative Designs

Other valid architectures that could achieve the same goals:

- **Virtual functions with a fixed scalar type set:** Define `virtual void compute_double(...)`, `virtual void compute_dual(...)`, etc. Eliminates CRTP but requires listing every scalar type explicitly and prevents adding new ones without modifying the base class.
- **Type erasure everywhere (no CRTP):** Use `GenericFunction` as the only representation. Simpler code, but every sub-expression incurs virtual dispatch overhead, and the compiler can never inline across expression boundaries.
- **Code generation / JIT:** Generate specialized evaluation code at problem setup time (like JAX or CasADi). Avoids both CRTP complexity and virtual overhead, but adds a compilation step.

The CRTP approach is a pragmatic choice for a C++ library where **hot-path performance** matters (solver iterations run millions of function evaluations) and **compile-time type safety** catches dimension mismatches early. The `GenericFunction` layer provides the runtime flexibility where needed.

#### Potential Future Improvement: C++23 Deducing This

C++23 introduces "deducing this" (explicit object parameters, [P0847](https://wg21.link/P0847)), which can replace the CRTP dispatch mechanism. This is a potential improvement worth experimenting with in a future release.

**What it replaces.** Instead of threading `Derived` through every base class and calling `this->derived()`, the derived type is deduced at the call site via an explicit `this` parameter:

```cpp
// Current CRTP pattern:
template <class Derived, int IR, int OR>
struct ComputableBase : CRTPBase<Derived> {
    template <class InType, class OutType>
    void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
        if constexpr (!Derived::IsVectorizable) { /* ... */ }
        this->derived().compute_impl(x, fx_);
    }
};

// With deducing this (C++23):
template <int IR, int OR>
struct ComputableBase {
    template <class Self, class InType, class OutType>
    void compute(this Self const& self, ConstVectorBaseRef<InType> x,
                 ConstVectorBaseRef<OutType> fx_) {
        if constexpr (!Self::IsVectorizable) { /* ... */ }
        self.compute_impl(x, fx_);
    }
};
```

Concrete types would simplify from `VectorFunction<Norm<IR>, IR, 1>` to `VectorFunction<IR, 1>` -- no need to pass the type to its own base class.

**What it does NOT replace.** Deducing this only eliminates the CRTP dispatch mechanism. Several aspects of the design are orthogonal and would remain unchanged:

- **Expression type explosion.** Types like `NestedFunction<CwiseSin<Segment<5,1,4>>, Arguments<5>>` exist because each operator creates a new type parameterized on its operands (the expression template pattern). Deducing this doesn't change this.
- **The deep inheritance chain.** The layered functionality (compute dispatch, domain tracking, Jacobian operations, derivative strategies) is separated across base classes by concern. Deducing this removes `Derived` from each layer but doesn't flatten the hierarchy.
- **GenericFunction type erasure.** Heterogeneous VectorFunctions still can't be stored in containers or passed to the solver without runtime type erasure.
- **The `IR`/`OR`/`Jm`/`Hm` template parameters.** These are compile-time parameterization for sizing and derivative strategy, unrelated to CRTP.

**Complications.** Some current patterns rely on `Derived` being available as a type-level parameter, not just inside methods:

- **Class-scope type aliases** like `EVALOP`, `FWDOP`, and `SEGMENTOP` in `DenseFunctionBase` use `Derived` to build return types. With deducing this, `Self` is only available inside method bodies, so these aliases would move to local computation:
  ```cpp
  // Current: class-scope alias using Derived
  template <class Func>
  using EVALOP = NestedFunctionSelector<Derived, Func>;

  // With deducing this: computed locally in each method
  template <int SZ, int ST>
  auto segment(this auto const& self) {
      using Self = std::remove_cvref_t<decltype(self)>;
      using SEGOP = NestedFunctionSelector<Segment<OR,SZ,ST>, Self>;
      return SEGOP::make_nested(Segment<OR,SZ,ST>(...), self);
  }
  ```
- The **`DENSE_FUNCTION_BASE_TYPES` macro** propagates type aliases using `Derived` and would need rewriting (though it would get simpler).

**Summary of pros and cons:**

|                      | Pros                                                              | Cons                                                                                                |
| -------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Readability**      | Simpler class declarations; no `CRTPBase`/`derived()` boilerplate | Class-scope type aliases must become local computations                                             |
| **Compilation**      | Marginally fewer template parameters per class                    | Expression type nesting still dominates compile time                                                |
| **Refactor scope**   | N/A                                                               | Massive refactor touching every file in `src/VectorFunctions/` and `src/Bindings/`                  |
| **Compatibility**    | N/A                                                               | Requires C++23; must verify Clang/LLVM, GCC, and MSVC support                                       |
| **Performance**      | Identical runtime performance (same inlining guarantees)          | N/A                                                                                                 |
| **Correctness risk** | N/A                                                               | Subtle behavioral differences in overload resolution between CRTP hiding and deducing this dispatch |

**Recommendation:** This is a worthwhile experiment for a future modernization pass, but not a high-priority change. The readability improvement is real but the refactor scope is large, and the hardest parts of the codebase (expression type nesting, type erasure, SuperScalar dispatch, KKT filling) are all orthogonal to CRTP. A proof-of-concept branch converting one base class (e.g., `ComputableBase`) would clarify the practical impact before committing to a full migration.

---

## 3. Template Parameters

Every VectorFunction carries five compile-time template parameters:

```cpp
template <class Derived, int IR, int OR,
          DenseDerivativeMode Jm = Analytic,
          DenseDerivativeMode Hm = Analytic>
struct VectorFunction;
```

| Parameter | Meaning                           | Values                                                  |
| --------- | --------------------------------- | ------------------------------------------------------- |
| `Derived` | The concrete derived class (CRTP) | Any class inheriting VectorFunction                     |
| `IR`      | Input dimension                   | Positive integer, or `-1` for dynamic                   |
| `OR`      | Output dimension                  | Positive integer, or `-1` for dynamic                   |
| `Jm`      | Jacobian computation mode         | `Analytic`, `FDiffFwd`, `FDiffCentArray`, `AutodiffFwd` |
| `Hm`      | Hessian computation mode          | Same as Jm                                              |

### Static vs. Dynamic Sizing

When `IR` and `OR` are known at compile time (e.g., `IR=6, OR=3`), Eigen can use fixed-size matrices, enabling stack allocation and SIMD optimization. When set to `-1`, sizes are stored as runtime integers and Eigen uses dynamically-allocated vectors/matrices.

```cpp
// Static sizing (preferred for performance when dimensions are known):
template <> struct InputOutputSize<6, 3> {
    static const int InputRows = 6;
    static const int OutputRows = 3;
};

// Dynamic sizing (required when dimensions vary at runtime):
template <> struct InputOutputSize<-1, -1> {
    int InputRows = 0;  // Mutable, set at construction
    int OutputRows = 0;
};
```

**Rule of thumb:** Use fixed sizes when the dimensions are inherent to the function. Use dynamic sizes (`-1`) when the function wraps arbitrary user input (like `GenericFunction`).

---

## 4. Implementing a VectorFunction (C++)

To implement a new VectorFunction, you inherit from `VectorFunction<Derived, IR, OR, Jm, Hm>` and provide `_impl` methods. Here's the contract:

### Required Methods

| Method         | Signature  | Purpose                                 |
| -------------- | ---------- | --------------------------------------- |
| `compute_impl` | `(x, fx_)` | Compute `f(x)`, storing result in `fx_` |

### Optional Methods (for Analytic Derivatives)

| Method                                                 | Signature                                    | Purpose                                                                 |
| ------------------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------- |
| `compute_jacobian_impl`                                | `(x, fx_, jx_)`                              | Compute `f(x)` and Jacobian `J(x)` simultaneously                       |
| `compute_jacobian_adjointgradient_adjointhessian_impl` | `(x, fx_, jx_, adjgrad_, adjhess_, adjvars)` | Compute `f(x)`, `J(x)`, `g = J^T * adjvars`, `H = sum(adjvars_i * H_i)` |

If you set `Jm = AutodiffFwd`, you only need `compute_impl` -- the base class generates Jacobians by running your compute with dual numbers. If you set `Jm = Analytic`, you must provide `compute_jacobian_impl` yourself (the default implementation would call the compute dispatch which may not give correct Jacobians). Similarly for `Hm`.

### Compile-Time Flags

You can override these `static const bool` flags to enable optimizations:

| Flag                  | Default | Meaning                                                                                                     |
| --------------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `IsVectorizable`      | `false` | Function supports SuperScalar evaluation (see [Section 7](#7-defaultsuperscalar-and-vectorized-evaluation)) |
| `IsLinearFunction`    | `false` | Jacobian is constant (independent of x). Hessian is identically zero.                                       |
| `HasDiagonalJacobian` | `false` | Jacobian is diagonal (enables optimized products)                                                           |
| `IsGenericFunction`   | `false` | Set only by `GenericFunction`                                                                               |

### Minimal Example: A Power Function

Here's a complete C++ VectorFunction that computes `f(x) = [x_0^2, x_1^2, ..., x_{n-1}^2]`:

```cpp
#include "VectorFunction.h"

namespace Tycho {

// Square each element of the input vector
template <int IR>
struct CwiseSquareExample : VectorFunction<CwiseSquareExample<IR>, IR, IR> {
    using Base = VectorFunction<CwiseSquareExample<IR>, IR, IR>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    // Enable batch evaluation with DefaultSuperScalar
    static const bool IsVectorizable = true;

    CwiseSquareExample() {}
    CwiseSquareExample(int ir) { this->setIORows(ir, ir); }

    // --- Required: compute f(x) ---
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x,
                             ConstVectorBaseRef<OutType> fx_) const {
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().square();
    }

    // --- Analytic Jacobian: diag(2*x) ---
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        fx = x.array().square();
        jx.diagonal() = x * Scalar(2.0);
    }

    // --- Analytic Hessian: H_i = 2*e_i*e_i^T  (diagonal) ---
    template <class InType, class OutType, class JacType,
              class AdjGradType, class AdjHessType, class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const {

        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        fx = x.array().square();
        jx.diagonal() = x * Scalar(2.0);
        adjgrad = jx.transpose() * adjvars;
        adjhess.diagonal() = adjvars * Scalar(2.0);
    }
};

} // namespace Tycho
```

### Using AutodiffFwd Instead of Manual Derivatives

If you only want to provide `compute_impl` and let the library handle derivatives:

```cpp
template <int IR>
struct CwiseSquareAD : VectorFunction<CwiseSquareAD<IR>, IR, IR,
                                       AutodiffFwd, AutodiffFwd> {
    using Base = VectorFunction<CwiseSquareAD<IR>, IR, IR, AutodiffFwd, AutodiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    static const bool IsVectorizable = true;

    CwiseSquareAD() {}
    CwiseSquareAD(int ir) { this->setIORows(ir, ir); }

    // Only compute_impl is needed -- autodiff generates Jacobians and Hessians
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x,
                             ConstVectorBaseRef<OutType> fx_) const {
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().square();
    }
};
```

### The BUILD_ODE_FROM_EXPRESSION Macro (C++)

For C++ ODE definitions, the `VectorExpression` utility and `BUILD_ODE_FROM_EXPRESSION` macro let you define ODEs declaratively:

```cpp
struct Brachistochrone_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args  = Arguments<5>();  // [x, y, v, t, theta]
        auto v     = args.coeff<2>();
        auto theta = args.coeff<4>();

        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);

        return StackedOutputs{xdot, ydot, vdot};
    }
};

BUILD_ODE_FROM_EXPRESSION(Brachistochrone, Brachistochrone_Impl, double);
```

The `Definition` static method returns a VectorFunction expression, and the macro wraps it in a proper ODE type that can be used with `ODEPhase`.

---

## 5. Input Domains

**Input domains** track which elements of the input vector a function actually uses. This is critical for performance -- during Jacobian accumulation, the system skips columns of zeros in the Jacobian that correspond to unused inputs.

### Compile-Time Domains

For statically-sized functions, domains are `constexpr`:

```cpp
// SingleDomain: the function uses a single contiguous block of inputs
template <int IR, int Start, int Size>
struct SingleDomain {
    static constexpr std::array<std::array<int, 2>, 1> SubDomains = {
        std::array<int, 2>{Start, Size}
    };
};

// Example: A Segment<8, 3, 2> uses inputs [2, 3, 4] out of 8 total inputs
// INPUT_DOMAIN = SingleDomain<8, 2, 3>  ->  SubDomains = {{2, 3}}
```

For composite functions, `CompositeDomain` merges the sub-domains of its operands:

```cpp
// CompositeDomain: merges multiple sub-domains, detects contiguous regions
template <int IR, class Domain1, class Domain2, ...>
struct CompositeDomain;
// e.g. f uses [0,3) and g uses [5,8) out of 10
//   -> SubDomains = {{0,3}, {5,3}}
```

### Runtime Domains

When `IR = -1` (dynamic sizing), domains are stored in a `DomainMatrix` (2 x N matrix where each column is `[start, size]`):

```cpp
template <> struct DomainHolder<-1> {
    DomainMatrix SubDomains;  // Eigen::Matrix<int, 2, -1>
    // SubDomains.col(0) = [start_0, size_0]
    // SubDomains.col(1) = [start_1, size_1]
    // ...
};
```

### Why Domains Matter

Consider a function `f` that maps from `R^100` but only reads inputs `[3..8]`. Its Jacobian has 94 zero columns. Without domain tracking, every Jacobian accumulation would touch all 100 columns. With domain tracking, only columns `[3..8]` are processed:

```cpp
// In DenseFunctionBase::right_jacobian_product:
if constexpr (std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
    // No special domain -- process all columns
    right_jacobian_product_impl(target_, left, right, assign, aliased);
} else {
    // Use compile-time domain to skip zero columns
    using DMN = typename Derived::INPUT_DOMAIN;
    right_jacobian_product_constant_impl(DMN(), target_, left, right, assign, aliased);
}
```

---

## 6. Derivative Computation

### The Three Strategies

The `Jm` and `Hm` template parameters select the derivative strategy via partial specialization of `DenseFirstDerivatives` and `DenseSecondDerivatives`:

#### 1. Analytic (Default)

The derived class provides hand-coded derivatives in `compute_jacobian_impl` and `compute_jacobian_adjointgradient_adjointhessian_impl`. This is the fastest option but requires manual derivation.

#### 2. Forward Automatic Differentiation (`AutodiffFwd`)

Uses the `autodiff` library's dual numbers. For the Jacobian, each input variable is "seeded" one at a time:

```cpp
// From DenseAutodiffFwd.h -- Jacobian via forward AD
template <class InType, class OutType, class JacType>
void compute_jacobian_impl(x, fx_, jx_) {
    Input<dual<Scalar>> xdual = x.cast<dual<Scalar>>();
    Output<dual<Scalar>> fdual(ORows());

    for (int i = 0; i < IRows(); i++) {
        xdual[i].grad = 1.0;          // Seed input i
        compute(xdual, fdual);          // Evaluate with dual numbers
        for (int j = 0; j < ORows(); j++)
            jx(j, i) = fdual[j].grad;  // Extract partial derivative
        xdual[i].grad = 0.0;           // Reset seed
        fdual.setZero();
    }
}
```

For the Hessian, second-order duals (`dual<dual<Scalar>>`) are used.

**Cost:** `IR` evaluations of `compute` for the Jacobian, `IR*(IR+1)/2` for the Hessian. Derivatives are machine-precision accurate.

#### 3. Forward Finite Differences (`FDiffFwd`)

Standard numerical differentiation: `df/dx_i ≈ (f(x + h*e_i) - f(x)) / h`. Less accurate than autodiff, but works for any function including ones that call external libraries.

#### 4. Central Finite Differences (`FDiffCentArray`)

Higher-accuracy numerical differentiation: `df/dx_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)`. More function evaluations but better accuracy than forward differences.

### What the Optimizer Actually Calls

During optimization, PSIOPT calls the "all-in-one" method:

```
compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, adjvars)
```

This simultaneously computes:
- `fx = f(x)`
- `jx = J(x)` (Jacobian)
- `adjgrad = J(x)^T * adjvars` (adjoint gradient)
- `adjhess = sum_i adjvars_i * H_i(x)` (adjoint Hessian)

For constraints, `adjvars` are the Lagrange multipliers. For objectives, `adjvars = [ObjScale]`.

---

## 7. DefaultSuperScalar and Vectorized Evaluation

### What Is a SuperScalar?

A **SuperScalar** is an Eigen fixed-size array that packs multiple independent scalar values into a single "slot":

```cpp
// From src/TypeDefs/EigenTypes.h:
template <class Scalar, int Sz>
using SuperScalarType = Eigen::Array<Scalar, Sz, 1>;

#ifdef __ARM_NEON
using DefaultSuperScalar = SuperScalarType<double, 2>;  // ARM: 2 doubles
#else
using DefaultSuperScalar = SuperScalarType<double, 4>;  // x86: 4 doubles
#endif
```

On ARM (Apple Silicon), `DefaultSuperScalar` is `Eigen::Array<double, 2, 1>` -- a 2-element array. On x86, it's 4 elements. The size is chosen to match the platform's SIMD register width.

### How It Enables Batched Evaluation

During optimization, a constraint function is applied at many different points along the trajectory (once per collocation node). Rather than evaluating the function one point at a time, Tycho packs multiple inputs into a single SuperScalar vector and evaluates them all at once:

```
                    Scalar evaluation (1 at a time)
                    ================================
                    compute(x_1, fx_1)
                    compute(x_2, fx_2)
                    compute(x_3, fx_3)
                    compute(x_4, fx_4)

                    SuperScalar evaluation (4 at a time on x86)
                    ============================================
                    x_packed[j] = Array<double,4>{x_1[j], x_2[j], x_3[j], x_4[j]}
                    compute(x_packed, fx_packed)  // single call, 4 evaluations
                    fx_1[j] = fx_packed[j][0]     // unpack results
                    fx_2[j] = fx_packed[j][1]
                    ...
```

### The Dispatch Logic in ComputableBase

The `compute()` method in `ComputableBase` handles the dispatch. There are three cases:

```cpp
template <class InType, class OutType>
inline void compute(x, fx_) const {
    typedef typename InType::Scalar Scalar;

    if constexpr (!Derived::IsVectorizable) {
        // Case 1: Function does NOT support vectorization
        if constexpr (Is_SuperScalar<Scalar>::value) {
            // But caller passed SuperScalar input! Must unpack and call one at a time.
            for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                for (int j = 0; j < IRows(); j++)
                    x_r[j] = x[j][i];           // Extract the i-th scalar from each input
                compute_impl(x_r, fx_r);          // Scalar evaluation
                for (int j = 0; j < ORows(); j++)
                    fx[j][i] = fx_r[j];           // Pack scalar result back
            }
        } else {
            // Case 2: Normal scalar call. Direct dispatch.
            compute_impl(x, fx_);
        }
    } else {
        // Case 3: Function supports vectorization. Regardless of whether input is
        // SuperScalar or not, just call compute_impl directly -- the function handles both.
        compute_impl(x, fx_);
    }
}
```

**Key insight:** When `IsVectorizable = true`, the function's `compute_impl` is written to work correctly with *both* scalar types (`double`) and SuperScalar types (`Eigen::Array<double, N>`). Eigen's expression templates make this work automatically -- operations like `+`, `*`, `.sin()`, `.cos()` are overloaded for both scalars and arrays.

When `IsVectorizable = false` but the solver calls with a SuperScalar, the base class automatically unpacks the array, calls `compute_impl` N times with plain `double`, and repacks the results. This is correct but slower.

### The Solver-Level Batching

The `constraints()` method in `ComputableBase` implements the batch loop:

```cpp
void constraints(X, FX, data) const {
    // ...
    auto VectorImpl = [&]() {
        using SuperScalar = Tycho::DefaultSuperScalar;
        constexpr int vsize = SuperScalar::SizeAtCompileTime;  // 2 on ARM, 4 on x86
        int Packs = data.NumAppl() / vsize;

        Input<SuperScalar> x_vect(IRows());
        Output<SuperScalar> fx_vect(ORows());

        for (int i = 0; i < Packs; i++) {
            // Pack vsize independent inputs into x_vect
            for (int j = 0; j < vsize; j++) {
                gatherInput(X, x, i * vsize + j, data);
                for (int k = 0; k < IRows(); k++)
                    x_vect[k][j] = x[k];
            }
            // Single batched evaluation
            fx_vect.setZero();
            compute(x_vect, fx_vect);
            // Unpack results
            for (int j = 0; j < vsize; j++) {
                // ... scatter fx_vect[l][j] into FX ...
            }
        }
        // Handle remainder (< vsize applications) with scalar fallback
        ScalarImpl(Packs * vsize, data.NumAppl());
    };

    if constexpr (Derived::IsVectorizable) {
        if (EnableVectorization)
            VectorImpl();
        else
            ScalarImpl(0, data.NumAppl());
    } else {
        ScalarImpl(0, data.NumAppl());
    }
}
```

### Guidelines for IsVectorizable

Set `IsVectorizable = true` when your `compute_impl` (and `compute_jacobian_impl`, etc.) works correctly with `Eigen::Array<double, N, 1>` as the scalar type. This is automatic if you:
- Use only Eigen array/matrix operations (`.sin()`, `.cos()`, `+`, `*`, etc.)
- Avoid branches that depend on the scalar value (no `if (x[0] > 0)`)
- Avoid calling external functions that expect `double`

**If your function calls Python code or external C libraries, it is NOT vectorizable.**

For composite functions (`NestedFunction`, `StackedOutputs`, etc.), `IsVectorizable` is the logical AND of all sub-functions' vectorizability.

---

## 8. Built-In Function Types

Tycho provides a rich library of pre-built VectorFunction types. All live under `src/VectorFunctions/CommonFunctions/`.

### Fundamental Building Blocks

#### `Segment<IR, OR, ST>` and `Arguments<IR>`

`Segment` extracts a contiguous slice from the input vector:

```
Segment<8, 3, 2>:  R^8 --> R^3
                    x  |-->  [x[2], x[3], x[4]]
```

Its Jacobian is a selector matrix (1s on the diagonal in columns `[ST..ST+OR)`, zeros elsewhere). It is `IsLinearFunction = true` and `IsVectorizable = true`.

`Arguments<IR>` is a special case: a Segment that passes the entire input through unchanged.

```cpp
// Arguments<N> ≡ Segment<N, N, 0>  (identity function)
template <int IR_OR>
struct Arguments : Segment_Impl<Arguments<IR_OR>, IR_OR, IR_OR, 0> { ... };
```

#### `Constant<IR, OR>` and `ConstantScalar<IR>`

Returns a fixed vector regardless of input. `IsLinearFunction = true`, Jacobian is all zeros.

#### `Scaled<Func>`, `RowScaled<Func>`, `StaticScaled<Func, Value>`

Scale the outputs of another function:
- `Scaled<Func>`: Multiply all outputs by a single scalar. `f(x) = s * g(x)`
- `RowScaled<Func>`: Multiply each output element by a corresponding scale vector. `f(x)_i = s_i * g(x)_i`
- `StaticScaled<Func, Value>`: Compile-time constant scaling factor.

Jacobian: `J_f = s * J_g` (or `diag(s) * J_g` for RowScaled).

#### `NestedFunction<OuterFunc, InnerFunc>` (Function Composition)

Represents `f(x) = Outer(Inner(x))`:

```
NestedFunction<Outer, Inner>:
    IRows = Inner.IRows
    ORows = Outer.ORows
    Jacobian: J_f = J_outer * J_inner  (chain rule)
```

There is a special partial specialization when the inner function is a `Segment` -- in that case, the Segment is "absorbed" and the outer function operates directly on a sub-range of the input, avoiding an intermediate allocation.

#### `StackedOutputs` (Output Concatenation)

Stacks the outputs of multiple functions into a single output vector:

```
StackedOutputs{f1, f2, f3}:
    ORows = f1.ORows + f2.ORows + f3.ORows
    f(x) = [f1(x); f2(x); f3(x)]
    J(x) = [J1(x); J2(x); J3(x)]    // vertical concatenation
```

All sub-functions must share the same input dimension `IR`.

### Unary Math Operations (CwiseOperators)

Each wraps a function and applies an element-wise operation. All provide analytic derivatives.

| Type             | Operation    | Jacobian Diagonal        |
| ---------------- | ------------ | ------------------------ |
| `CwiseSin<F>`    | `sin(f(x))`  | `cos(f(x)) * J_f`        |
| `CwiseCos<F>`    | `cos(f(x))`  | `-sin(f(x)) * J_f`       |
| `CwiseTan<F>`    | `tan(f(x))`  | `sec^2(f(x)) * J_f`      |
| `CwiseExp<F>`    | `exp(f(x))`  | `exp(f(x)) * J_f`        |
| `CwiseSqrt<F>`   | `sqrt(f(x))` | `1/(2*sqrt(f(x))) * J_f` |
| `CwiseSquare<F>` | `f(x)^2`     | `2*f(x) * J_f`           |
| `CwiseArcSin<F>` | `asin(f(x))` | `1/sqrt(1-f(x)^2) * J_f` |

These are all implemented via the `CwiseFunctionOperator<Derived, Func>` base, which chains derivatives using the inner function's Jacobian:

```cpp
// Simplified structure of CwiseSin
template <class Func>
struct CwiseSin : CwiseFunctionOperator<CwiseSin<Func>, Func> {
    // Provides cwise_compute, cwise_compute_jacobian, cwise_compute_jacobian_hessian
    // which operate on the inner function's output

    static void cwise_compute(x, fx_) {
        fx = x.array().sin();   // x here is the inner function's output
    }
    static void cwise_compute_jacobian(x, fx_, jx_) {
        fx = x.array().sin();
        jx = x.array().cos();   // Diagonal Jacobian of sin
    }
};
```

The `CwiseFunctionOperator` base handles the chain rule: it calls the inner function to get `g(x)` and `J_g(x)`, then applies the cwise operation and multiplies the diagonal cwise Jacobian by `J_g(x)`.

### Norm Functions

```
Norm<IR>:             f(x) = ||x||              (OR = 1)
SquaredNorm<IR>:      f(x) = ||x||^2            (OR = 1)
InverseNorm<IR>:      f(x) = 1/||x||            (OR = 1)
InverseSquaredNorm:   f(x) = 1/||x||^2          (OR = 1)
NormPower<IR, PW>:    f(x) = ||x||^PW           (OR = 1)
```

All have `OR = 1` (scalar output) and provide analytic derivatives. They all support vectorization.

### Normalized and NormalizedPower

```
Normalized<IR>:       f(x) = x / ||x||           (OR = IR)
NormalizedPower<IR,P>: f(x) = x / ||x||^P        (OR = IR)
```

### Binary Operations

| Type                                  | Operation        | Notes                            |
| ------------------------------------- | ---------------- | -------------------------------- |
| `FunctionDotProduct<F1, F2>`          | `f1(x) . f2(x)`  | Scalar output                    |
| `FunctionCrossProduct<F1, F2>`        | `f1(x) x f2(x)`  | 3D vectors only                  |
| `CwiseFunctionProduct<F1, F2>`        | `f1(x) .* f2(x)` | Element-wise multiplication      |
| `VectorScalarFunctionProduct<VF, SF>` | `vf(x) * sf(x)`  | Vector scaled by scalar function |

---

## 9. Expression Composition (Operator Overloads)

### Arithmetic Operators

All operate on objects inheriting from `DenseFunctionBase` and return new VectorFunction expression objects:

```cpp
// Scalar multiplication (from OperatorOverloads.h)
f * 2.0       -->  Scaled<Func>(f, 2.0)
2.0 * f       -->  Scaled<Func>(f, 2.0)

// Chained scaling is optimized:
(f * 2.0) * 3.0  -->  Scaled<Func>(f, 6.0)  // NOT Scaled<Scaled<Func>>

// Element-wise scaling
f * vec       -->  RowScaled<Func>(f, vec)   // vec is an Eigen vector

// Division
f / 2.0       -->  Scaled<Func>(f, 0.5)

// Addition with a constant vector
f + vec       -->  FunctionPlusVector<Func>(f, vec)

// Negation
-f            -->  Scaled<Func>(f, -1.0)
```

### Free-Function Math (C++)

For scalar functions (OR=1), free functions are defined in `MathOverloads.h`:

```cpp
sin(f)    -->  CwiseSin<Func>(f)
cos(f)    -->  CwiseCos<Func>(f)
sqrt(f)   -->  CwiseSqrt<Func>(f)
exp(f)    -->  CwiseExp<Func>(f)
```

### Function Composition

The `operator()` on `DenseFunctionBase` creates a `NestedFunction`:

```cpp
auto composed = outer(inner);  // NestedFunction<OuterFunc, InnerFunc>
// Requirement: inner.ORows() == outer.IRows()
```

### Python DSL

In Python, the same operators and functions are available:

```python
import tycho as ast
vf = ast.VectorFunctions
Args = vf.Arguments

args = Args(3)
x, y, z = args.tolist()

# Build expressions -- no computation happens here, only expression construction
f = vf.sin(x) * y + z**2            # scalar function of 3 inputs
g = vf.stack([x**2, y + z, x * y])  # 3-output function of 3 inputs
h = g.norm()                         # scalar: ||g(x)||

# Evaluate
print(f.compute([1.0, 2.0, 3.0]))      # sin(1)*2 + 9 = 10.6829...
print(g.compute([1.0, 2.0, 3.0]))      # [1.0, 5.0, 2.0]
print(g.jacobian([1.0, 2.0, 3.0]))     # 3x3 Jacobian matrix
```

---

## 10. Type Erasure: GenericFunction

### The Problem

VectorFunctions are compile-time polymorphic (CRTP). A `Scaled<Segment<6, 3, 0>>` and a `NestedFunction<Norm<3>, Segment<6, 3, 0>>` are entirely different C++ types. You can't store them in the same container, pass them to the same function, or hold them in a class member.

### The Solution: GenericFunction<IR, OR>

`GenericFunction` wraps any compatible VectorFunction in a type-erased container, providing runtime polymorphism:

```cpp
template <int IR, int OR>
struct GenericFunction : VectorFunction<GenericFunction<IR, OR>, IR, OR> {
    TE func;  // Type-erased storage (rubber_types::TypeErasure)

    // Construct from ANY VectorFunction with matching dimensions
    template <class T>
    GenericFunction(const T& t) : func(t) { cachedata(); }

    // Forwarding implementations
    template <class InType, class OutType>
    void compute_impl(x, fx_) const {
        func.compute(x, fx_);  // Virtual dispatch to stored function
    }
    // ... same for compute_jacobian_impl, etc.
};
```

### Key GenericFunction Aliases

| Type                      | Meaning                          |
| ------------------------- | -------------------------------- |
| `GenericFunction<-1, -1>` | Fully dynamic vector function    |
| `GenericFunction<-1, 1>`  | Dynamic-input scalar function    |
| `GenericFunction<6, 3>`   | Fixed 6-input, 3-output function |

### Deep Copy Between GenericFunction Variants

A `GenericFunction<6, 3>` can be deep-copied into a `GenericFunction<-1, -1>`:

```cpp
GenericFunction<6, 3> specific_func(some_expression);
GenericFunction<-1, -1> generic_func(specific_func);  // Deep copy, dynamic sizing
```

### When Type Erasure Happens

In the Python API, all expressions are ultimately converted to `GenericFunction<-1, -1>` (or `<-1, 1>` for scalars) before being stored by the optimal control system. This is the `.vf()` and `.sf()` methods:

```python
f = vf.sin(args[0]) * args[1]  # Still a compile-time type in C++
g = f.vf()                       # Now a GenericFunction<-1, -1> -- type erased
```

The `vf.stack()`, `vf.sum()`, etc. functions also return `GenericFunction`.

---

## 11. The PSIOPT Solver Interface

### How VectorFunctions Become Constraints

When you call `phase.addEqualCon(...)`, the phase creates a `ConstraintFunction` wrapper that pairs the VectorFunction with a `SolverIndexingData` struct describing:

1. **Which variables** from the full NLP decision vector to pass as inputs (for each collocation node)
2. **Where to write** the constraint outputs in the full constraint vector
3. **Where to scatter** Jacobian/Hessian elements in the KKT matrix

### SolverIndexingData

```cpp
struct SolverIndexingData {
    MatrixXi Vindex;  // Columns = function applications, rows = input variable indices
    MatrixXi Cindex;  // Columns = function applications, rows = constraint indices

    // Pre-computed metadata
    VectorXi InnerConstraintStarts;  // Where each constraint block starts in FX
    VectorXi InnerGradientStarts;    // Where each gradient block starts in AGX
    VectorXi InnerKKTStarts;         // Where each KKT block starts in the sparse matrix

    int NumAppl();   // Number of times this function is applied (e.g., # collocation nodes)
    int VLoc(i, V);  // Input variable index i for application V
    int CLoc(j, V);  // Constraint index j for application V
};
```

### The gatherInput Pattern

Before each function evaluation, the solver gathers the relevant variables from the full NLP vector `X`:

```cpp
void gatherInput(X, xt, V, data) {
    if (data.VindexContinuity[V] == Contiguous) {
        // Variables are contiguous in X -- fast memcpy-like operation
        xt = X.segment<IR>(data.VLoc(0, V), IRows());
    } else {
        // Variables are scattered -- gather one by one
        for (int i = 0; i < IRows(); i++)
            xt(i) = X(data.VLoc(i, V));
    }
}
```

### KKT Matrix Population

After computing the Jacobian and Hessian, elements must be scattered into the global KKT matrix (which is sparse). The `KKTFillJac` and `KKTFillAll` methods handle this, including thread-safety via column-level mutexes:

```
KKTFillAll(V, jx, hx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data):
    for each Hessian element (i,j):
        lock column if contested
        KKTmat[KKTLocations[...]] += hx(j,i)
        unlock
    for each Jacobian element (j,i):
        KKTmat[KKTLocations[...]] += jx(j,i)
```

### The Full Evaluation Chain

During a single PSIOPT iteration:

1. PSIOPT calls `constraints_jacobian_adjointgradient_adjointhessian(X, L, FX, AGX, KKTmat, ...)`
2. For each of `data.NumAppl()` applications:
   a. `gatherInput(X, x, V, data)` -- extract local inputs from global vector
   b. `gatherMult(L, l, V, data)` -- extract local multipliers
   c. `compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, agx, hx, l)` -- evaluate
   d. `KKTFillAll(V, jx, hx, ...)` -- scatter into KKT matrix
3. If `IsVectorizable && EnableVectorization`, step 2 is batched using `DefaultSuperScalar`

---

## 12. Python API

### Import Convention

```python
import tycho as ast
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
```

### Creating Input Variables

```python
# General input variable (for constraints, objectives, standalone functions)
args = Args(n)          # n-dimensional input

# ODE-specific input variable (separates state, control, parameters)
ode_args = oc.ODEArguments(XVars=3, UVars=1, PVars=0)
```

### Indexing and Slicing

```python
args = Args(6)

x = args[0]              # Single element (scalar function)
xy = args[0:2]           # Slice (2-element vector function)
r = args.head(3)         # First 3 elements
v = args.tail(3)         # Last 3 elements
mid = args.segment(2, 3) # 3 elements starting at index 2
r, v = args.tolist([(0,3), (3,3)])  # Unpack into named segments
```

### Arithmetic and Math

```python
x, y = Args(2).tolist()

# Arithmetic
f = x + y        # Addition of scalar functions
g = x * 2.0      # Scalar multiplication
h = x * y        # Product of scalar functions
k = x / y        # Division
m = -x           # Negation
n = x ** 3       # Power

# Math functions
s = vf.sin(x)
c = vf.cos(x)
e = vf.exp(x)
l = vf.log(x)
q = vf.sqrt(x)
a = vf.arctan2(y, x)

# Vector operations
args6 = Args(6)
r = args6.head(3)
v = args6.tail(3)

n = r.norm()             # ||r|| (scalar)
d = r.dot(v)             # r . v (scalar)
cr = r.cross(v)          # r x v (3-vector, both must be 3D)
rn = r.normalized()      # r / ||r||
rp3 = r.normalized_power3()  # r / ||r||^3 (common in gravity models)
```

### Stacking

```python
# Concatenate scalar/vector functions into a vector
ode = vf.stack([xdot, ydot, zdot])     # Stack scalars -> 3-vector
full = vf.stack([position, velocity])   # Stack 3-vecs -> 6-vector
```

### Function Composition

```python
# Evaluate f at the output of g: f(g(x))
composed = f(g)       # or f.eval(g)

# This also works with raw indices:
r = args6.head(3)     # Segment: extracts [0,1,2] from 6-dim input
gravity = r.normalized_power3() * (-mu)  # Nested: normalized_power3(Segment(x))
```

### Evaluation (For Debugging/Plotting)

```python
f = vf.sin(Args(1)[0])
val = f.compute([0.5])         # Returns numpy array: [sin(0.5)]
jac = f.jacobian([0.5])        # Returns numpy matrix: [[cos(0.5)]]
fx, jx = f.compute_jacobian([0.5])  # Both at once

# Performance testing
f.rpt([1.0, 2.0, 3.0], 100000)  # Run 100k evaluations and print timing
```

### Type Conversion

```python
f = vf.sin(Args(1)[0]) * Args(1)[0]  # Compile-time expression type

gf = f.vf()   # Convert to GenericFunction<-1, -1> (vector)
sf = f.sf()   # Convert to GenericFunction<-1, 1> (scalar, only if ORows==1)
```

### Defining ODEs in Python

```python
class MyODE(oc.ODEBase):
    def __init__(self, g):
        XVars = 3
        UVars = 1
        XtU = oc.ODEArguments(XVars, UVars)

        x, y, v = XtU.XVec().tolist()   # State variables
        theta = XtU.UVar(0)              # Control variable

        # Build symbolic dynamics
        xdot = vf.sin(theta) * v
        ydot = -1.0 * vf.cos(theta) * v
        vdot = g * vf.cos(theta)
        ode = vf.stack([xdot, ydot, vdot])

        # Register with base class
        super().__init__(ode, XVars, UVars)
```

The `ODEBase.__init__` method:
1. Takes the ODE VectorFunction and dimension info
2. Selects the appropriate C++ ODE template (e.g., `ode_3_1` for XVars=3, UVars=1)
3. Passes the type-erased function to the C++ layer

### Using Custom Constraints/Objectives

```python
# Equality constraint: a VectorFunction that must equal zero
def orbit_constraint(a_target, e_target):
    r, v = Args(6).tolist([(0,3), (3,3)])
    a = ...  # compute semi-major axis from r, v
    e = ...  # compute eccentricity from r, v
    return vf.stack([a - a_target, e - e_target])

phase.addEqualCon("Back", orbit_constraint(a_t, e_t), range(0, 6))
#                  ^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^
#                  where   VectorFunction (must=0)     which variables to pass

# Integral objective: a ScalarFunction integrated over the trajectory
class ControlEffort(vf.ScalarFunction):
    def __init__(self):
        u = Args(3)
        super().__init__(u.squared_norm())

phase.addIntegralObjective(ControlEffort(), [7, 8, 9])
#                                            ^^^^^^^^^
#                                            indices into the phase vector
```

---

## 13. End-to-End Examples

### Example 1: Brachistochrone (Python)

The classic minimum-time brachistochrone: find the wire shape that lets a bead slide between two points under gravity in minimum time.

```python
import numpy as np
import tycho as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl

class Brachistochrone(oc.ODEBase):
    def __init__(self, g):
        XVars = 3   # State: [x, y, v]
        UVars = 1   # Control: [theta]

        XtU = oc.ODEArguments(XVars, UVars)
        x, y, v = XtU.XVec().tolist()
        theta = XtU.UVar(0)

        # Dynamics
        xdot = vf.sin(theta) * v
        ydot = -1.0 * vf.cos(theta) * v
        vdot = g * vf.cos(theta)
        ode = vf.stack([xdot, ydot, vdot])

        super().__init__(ode, XVars, UVars)

# Problem setup
g = 9.81
ode = Brachistochrone(g)

# Initial guess: straight line, constant angle
tf = 1.0
ts = np.linspace(0, tf, 100)
Xs = []
for t in ts:
    X = np.zeros(5)  # [x, y, v, t, theta]
    X[0] = 10.0 * t / tf           # x: 0 -> 10
    X[1] = 10.0 - 5.0 * t / tf     # y: 10 -> 5
    X[2] = g * t * np.cos(1.0)     # v: linear ramp
    X[3] = t                        # t: 0 -> tf
    X[4] = 1.0                      # theta: constant guess
    Xs.append(X)

# Create phase and add constraints
phase = ode.phase("LGL3", Xs, 32)
phase.addBoundaryValue("Front", range(0, 4), [0, 10, 0, 0])
phase.addBoundaryValue("Back", [0, 1], [10, 5])
phase.addLUVarBound("Path", 4, -0.1, 2.0)
phase.addDeltaTimeObjective(1.0)

# Solve
phase.optimize()
Traj = phase.returnTraj()
print(f"Optimal time: {Traj[-1][3]:.4f} s")  # ~1.8013 s
```

### Example 2: Low-Thrust Spacecraft (Python)

```python
class LowThrust(oc.ODEBase):
    def __init__(self, mu, accel):
        XVars = 6   # State: [rx, ry, rz, vx, vy, vz]
        UVars = 3   # Control: [ux, uy, uz] (thrust direction)

        args = oc.ODEArguments(XVars, UVars)
        r = args.head3()         # Position vector
        v = args.segment3(3)     # Velocity vector
        u = args.tail3()         # Control (thrust unit vector)

        # Gravitational acceleration: -mu * r / |r|^3
        gravity = r.normalized_power3() * (-mu)

        # Thrust acceleration
        thrust = u * accel

        # Full dynamics: rdot = v, vdot = gravity + thrust
        ode = vf.stack([v, gravity + thrust])

        super().__init__(ode, XVars, UVars)

    class effort_obj(vf.ScalarFunction):
        """Integral objective: minimize ||u||^2"""
        def __init__(self):
            u = vf.Arguments(3)
            super().__init__(u.squared_norm())

# Usage
ode = LowThrust(mu=1.0, accel=0.01)
phase = ode.phase("LGL5", initial_guess, 64)
phase.addIntegralObjective(LowThrust.effort_obj(), [7, 8, 9])
```

### Example 3: Brachistochrone (C++)

```cpp
#include "Tycho.h"
using namespace Tycho;

// Define ODE dynamics as a VectorFunction expression
struct Brach_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args  = Arguments<5>();     // [x, y, v, t, theta]
        auto v     = args.coeff<2>();    // Speed
        auto theta = args.coeff<4>();    // Control angle

        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);

        return StackedOutputs{xdot, ydot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(BrachODE, Brach_Impl, double);

int main() {
    BrachODE ode(9.81);

    // Build initial guess trajectory
    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 100; ++i) {
        double s = double(i) / 99.0;
        Eigen::VectorXd pt(5);
        pt << 10.0*s, 10.0 - 5.0*s, 9.81*s*std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    // Create phase
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, "LGL3", traj, 32);

    // Boundary conditions
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << 0, 10, 0, 0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, std::string("auto"));

    Eigen::VectorXi back_idx(2); back_idx << 0, 1;
    Eigen::VectorXd back_val(2); back_val << 10, 5;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, std::string("auto"));

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, std::string("auto"));

    phase->solve_optimize();
    return 0;
}
```

### Example 4: Understanding Expression Types

To build intuition about what the expression system produces at the C++ type level:

```python
args = Args(5)       # Arguments<-1> (dynamic, from Python)
v = args[2]          # NestedFunction<Segment<-1,1,-1>, Arguments<-1>>
theta = args[4]      # NestedFunction<Segment<-1,1,-1>, Arguments<-1>>

s = vf.sin(theta)    # NestedFunction<CwiseSin<Segment<-1,1,-1>>, Arguments<-1>>
p = s * v            # VectorScalarFunctionProduct<..., ...>
                     #   (scalar function * scalar function = scalar function)

ode = vf.stack([p, ...])  # GenericFunction<-1, -1> (type erased by stack)
```

In C++ with compile-time sizes, the types are more specific:

```cpp
auto args = Arguments<5>();
auto v = args.coeff<2>();           // NestedFunction<Segment<5,1,2>, Arguments<5>>
auto theta = args.coeff<4>();       // NestedFunction<Segment<5,1,4>, Arguments<5>>
auto s = sin(theta);                // NestedFunction<CwiseSin<Segment<5,1,4>>, Arguments<5>>
auto p = s * v;                     // VectorScalarFunctionProduct<NestedFunction<...>, NestedFunction<...>>
auto ode = StackedOutputs{p, ...};  // StackedOutputs<VectorScalarFunctionProduct<...>, ...>
```

---

## 14. Reference: Complete Class Hierarchy Diagram

```
InputOutputSize<IR,OR>
 + CRTPBase<Derived>
    |
    v
ComputableBase<Derived,IR,OR>          CORE: compute(), constraints(), gatherInput()
    |
    v
Computable<Derived,IR,OR>              Aliases (OR=1 specialization exists)
 + DomainHolder<IR>
    |
    v
DenseFunctionBase<Derived,IR,OR>       CORE: jacobian ops, indexing, math, KKT fill
    |                                    OR=1 --> DenseScalarFunctionBase (objective interface)
    v
DenseFunction<Derived,IR,OR>           Routing: OR=1 -> DenseScalarFunctionBase
    |
    v
DenseFirstDerivatives<Derived,IR,OR,Jm>   Jacobian strategy (Analytic/AutodiffFwd/FDiff)
    |
    v
DenseSecondDerivatives<Derived,IR,OR,Jm,Hm>  Hessian strategy
    |
    v
DenseDerivatives<Derived,IR,OR,Jm,Hm>       Aggregate
    |
    v
VectorFunction<Derived,IR,OR,Jm,Hm>         ** BASE FOR ALL USER TYPES **
    |
    |--- Segment_Impl<D,IR,OR,ST>  (+ SegStartHolder<ST>)
    |       |--- Segment<IR,OR,ST>
    |       |--- Arguments<IR>  (= Segment<IR,IR,0>)
    |
    |--- Scaled_Impl<D,Func>
    |       |--- Scaled<Func>
    |
    |--- RowScaled_Impl<D,Func>
    |       |--- RowScaled<Func>
    |
    |--- NestedFunction_Impl<D,OuterFunc,InnerFunc>
    |       |--- NestedFunction<OuterFunc,InnerFunc>
    |       |--- (Partial spec for InnerFunc = Segment: no temp allocation)
    |
    |--- CwiseFunctionOperator<D,Func>
    |       |--- CwiseSin, CwiseCos, CwiseTan, CwiseExp, CwiseSqrt, ...
    |
    |--- IntegralNorm_Impl<D,USZ,Power>
    |       |--- Norm, SquaredNorm, InverseNorm, NormPower, ...
    |
    |--- FunctionDotProduct<F1,F2>
    |--- FunctionCrossProduct<F1,F2>
    |--- CwiseFunctionProduct<F1,F2>
    |--- VectorScalarFunctionProduct<VF,SF>
    |--- FunctionPlusVector<Func>
    |--- PaddedOutput<Func,UP,LP>
    |--- MatrixFunctionView<Func,MR,MC,Major>
    |--- Conditional<Func>
    |
    |--- GenericFunction<IR,OR>  (TYPE ERASURE via rubber_types)
```

### File Organization

```
src/VectorFunctions/
    VectorFunction.h              Top-level struct + VectorExpression + BUILD_* macros
    ComputableBase.h              compute() dispatch, constraints() solver interface
    Computable.h                  Minimal wrapper, OR=1 specialization
    DenseFunctionBase.h           ~1200 lines: Jacobian ops, indexing, math, KKT fill
    DenseScalarFunctionBase.h     objective/gradient/hessian interface (OR=1)
    DenseFunction.h               DenseFunction -> DenseScalarFunctionBase routing
    DenseDerivatives.h            DenseDerivativeMode enum, derivative layer structs
    InputOuputSize.h              Static/dynamic size storage
    FunctionDomains.h             SingleDomain, CompositeDomain, DomainHolder
    IndexingData.h                SolverIndexingData (Vindex, Cindex, KKT locations)
    FunctionalFlags.h             ParsedIOFlags, VarTypes enums
    AssigmentTypes.h              DirectAssignment, PlusEqualsAssignment, etc.
    DetectSuperScalar.h           Is_SuperScalar<T> trait
    OperatorOverloads.h           +, -, *, / on VectorFunctions
    MathOverloads.h               sin, cos, sqrt, etc. on scalar functions
    BinaryMath.h                  crossProduct, dotProduct, cwiseProduct
    DenseFunctionOperations.h     right_jacobian_product implementations

    DenseDifferentiation/
        DenseAutodiffFwd.h        Autodiff Jacobian/Hessian via dual numbers
        DenseFDiffFwd.h           Forward finite-difference derivatives
        DenseFDiffCentArray.h     Central finite-difference derivatives

    CommonFunctions/
        Segment.h                 Segment, Arguments
        Scaled.h                  Scaled, RowScaled, StaticScaled, MatrixScaled
        NestedFunction.h          Function composition (chain rule)
        Stacked.h                 StackedOutputs (vertical concatenation)
        CwiseOperators.h          CwiseSin, CwiseCos, CwiseTan, etc.
        Norms.h                   Norm, SquaredNorm, InverseNorm, etc.
        Normalized.h              Normalized, NormalizedPower
        Constant.h                Constant value functions
        Elements.h                Extract non-contiguous elements
        CwiseSum.h                Sum all outputs to a scalar
        CwiseProduct.h            Element-wise function product
        Conditional.h             Conditional/ifelse logic

    VectorFunctionTypeErasure/
        GenericFunction.h         GenericFunction<IR,OR> (type erasure wrapper)
        DenseFunctionSpecs.h      rubber_types concept for compute/jacobian
        SizingSpecs.h             rubber_types concept for IRows/ORows
        SolverInterfaceSpecs.h    rubber_types concept for constraints/objective
        DeepCopySpecs.h           rubber_types deep copy between GenericFunction variants

src/TypeDefs/
    EigenTypes.h                  SuperScalarType, DefaultSuperScalar definitions
```
