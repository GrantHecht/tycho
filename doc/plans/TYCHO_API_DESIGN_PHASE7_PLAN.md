# Phase 7: Static DSL Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Address the five known pain points in the static VF DSL: template memory explosion, single-parameter ODE macro, no named variables, `Scaled<Scaled>` nesting bugs, and verbose definition ceremony.

**Architecture:** Each pain point is addressed independently. The firebreak mechanism and FD/AD mode switches add new user-facing options to the existing DSL without breaking backward compatibility. The ODE definition syntax exploration produces two prototype approaches evaluated against real examples before committing to one.

**Tech Stack:** C++20, template metaprogramming, `GenericFunction` type erasure

---

### Task 1: Create feature branch

```bash
git checkout -b feat/static-dsl-improvements main
```

---

### Task 2: Fix `Scaled<Scaled<...>>` nesting (Pain Point 4)

**Files:**
- Modify: `include/tycho/detail/operator_overloads.h` (or `src/VectorFunctions/OperatorOverloads.h`)
- Test: `tests/cpp/vector_functions/test_scaled_nesting.cpp`

This is the simplest fix — pure implementation, no API change.

**Step 1: Write failing test**

```cpp
TEST(ScaledNesting, DoubleScaleCollapse) {
    auto args = Tycho::Arguments<3>();
    auto x = args.coeff<0>();

    // This pattern triggers Scaled<Scaled<...>> today
    double a = 2.0;
    double b = 3.0;
    auto expr = a * (b * x);  // Should be Scaled<..., 6.0>, not Scaled<Scaled<...>>

    Eigen::Vector3d input(1.0, 2.0, 3.0);
    Eigen::VectorXd output(1);
    expr.compute(input, output);
    EXPECT_DOUBLE_EQ(output(0), 6.0);  // 2 * 3 * 1.0
}

TEST(ScaledNesting, DivisionPattern) {
    auto args = Tycho::Arguments<3>();
    auto x = args.coeff<0>();

    // Pattern from TODO.md: scalar * (1.0 + 0.01 * vf) / scalar
    double s = 5.0;
    auto expr = s * x / s;  // Should simplify to just x

    Eigen::Vector3d input(7.0, 0.0, 0.0);
    Eigen::VectorXd output(1);
    expr.compute(input, output);
    EXPECT_DOUBLE_EQ(output(0), 7.0);
}
```

**Step 2: Fix `OperatorOverloads.h`**

Add specialization: when `operator*` or `operator/` produces `Scaled<Scaled<...>>`,
collapse to a single `Scaled` with the combined coefficient:

```cpp
// Detect Scaled<Func> and collapse
template <typename Func, typename = void>
struct is_scaled : std::false_type {};

template <typename Func>
struct is_scaled<Scaled<Func>> : std::true_type {};

// When multiplying scalar * Scaled<Func>, produce Scaled<Func> with combined coeff
template <typename Func>
auto operator*(double scalar, const Scaled<Func>& s) {
    return Scaled<Func>(s.inner(), scalar * s.scale());
}
```

**Step 3: Build and run tests**

```bash
cd build && ninja -j2 tycho_tests && ctest --output-on-failure -R ScaledNesting
```

**Step 4: Commit**

```bash
git commit -m "fix: collapse Scaled<Scaled<...>> nesting to single Scaled"
```

---

### Task 3: Fix `BUILD_ODE_FROM_EXPRESSION` multi-parameter support (Pain Point 2)

**Files:**
- Modify: `include/tycho/detail/ode.h` (or `src/OptimalControl/ODE.h`)
- Test: `tests/cpp/optimal_control/test_multi_param_ode.cpp`

**Step 1: Write failing test**

```cpp
struct MultiParamODE_Impl : Tycho::ODESize<3, 1, 0> {
    static auto Definition(double g, double drag) {
        auto args = Tycho::Arguments<5>();
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();
        return Tycho::StackedOutputs{
            Tycho::sin(theta) * v,
            -1.0 * Tycho::cos(theta) * v,
            g * Tycho::cos(theta) - drag * v
        };
    }
};

// This should work with multiple parameters
BUILD_ODE_FROM_EXPRESSION(MultiParamODE, MultiParamODE_Impl, double, double);

TEST(MultiParamODE, Constructs) {
    MultiParamODE ode(9.81, 0.1);
    EXPECT_EQ(ode.IRows(), 5);
    EXPECT_EQ(ode.ORows(), 3);
}
```

**Step 2: Fix macro**

Update `BUILD_ODE_FROM_EXPRESSION` (or create `TYCHO_BUILD_ODE`) to accept
variadic parameters:

```cpp
#define TYCHO_BUILD_ODE(Name, Impl, ...)                                     \
    struct Name : Tycho::VectorExpression<Name, Impl, __VA_ARGS__> {         \
        using Base = Tycho::VectorExpression<Name, Impl, __VA_ARGS__>;       \
        Name(__VA_ARGS__) : Base(/* forward args */) {}                      \
    };
```

The challenge is correctly forwarding the variadic constructor parameters.
May need a helper that generates the constructor signature from the type list.

**Alternative approach:** If the macro is too fragile, provide a function-based
construction:

```cpp
auto ode = Tycho::make_static_ode<MultiParamODE_Impl>(9.81, 0.1);
```

Where `make_static_ode` is a factory function template.

**Step 3: Verify `Input<Scalar>` / `Output<Scalar>` aliases work**

The key issue from TODO.md is that struct parameters break these aliases.
Test that `Integrator<MultiParamODE>` compiles.

**Step 4: Build, test, commit**

---

### Task 4: Implement firebreak mechanism (Pain Point 1)

**Files:**
- Create: `include/tycho/detail/barrier.h`
- Test: `tests/cpp/vector_functions/test_barrier.cpp`

**Step 1: Write test**

```cpp
TEST(Barrier, CutsTypeTree) {
    auto args = Tycho::Arguments<5>();
    auto x = args.coeff<0>();
    auto y = args.coeff<1>();

    // Build a complex subexpression
    auto complex_expr = Tycho::sin(x) * Tycho::cos(y) + Tycho::exp(x * y);

    // Insert a barrier — type-erases to GenericFunction
    auto barrier_expr = Tycho::barrier(complex_expr);

    // barrier_expr should be a GenericFunction, not a deeply nested template type
    static_assert(std::is_same_v<
        std::decay_t<decltype(barrier_expr)>,
        Tycho::GenericFunction<-1, -1>
    > || /* some other erased type */);

    // Should still compute correctly
    Eigen::VectorXd input(5);
    input << 1.0, 2.0, 0, 0, 0;
    Eigen::VectorXd output(1);
    barrier_expr.compute(input, output);

    double expected = std::sin(1.0) * std::cos(2.0) + std::exp(1.0 * 2.0);
    EXPECT_NEAR(output(0), expected, 1e-12);
}
```

**Step 2: Implement `Tycho::barrier()`**

```cpp
namespace Tycho {

// Type-erase a VectorFunction expression to cut template depth.
// The resulting GenericFunction uses virtual dispatch for evaluation.
template <typename Func>
GenericFunction<-1, -1> barrier(const Func& f) {
    return GenericFunction<-1, -1>(f);
}

// Sized variant that preserves compile-time size info
template <int IR, int OR, typename Func>
GenericFunction<IR, OR> barrier(const Func& f) {
    return GenericFunction<IR, OR>(f);
}

} // namespace Tycho
```

This is surprisingly simple — `GenericFunction`'s constructor already type-erases
any VectorFunction. The `barrier()` function is just a named wrapper that makes
the intent explicit in ODE definitions.

**Step 3: Verify it reduces template depth**

Create a moderately complex expression and verify compilation with and without
barriers:

```cpp
// Without barrier: deep template nesting, high memory
auto deep = sin(cos(sin(cos(sin(x)))));

// With barrier: intermediate results type-erased
auto shallow = sin(Tycho::barrier(cos(sin(Tycho::barrier(cos(sin(x)))))));
```

**Step 4: Build, test, commit**

---

### Task 5: Surface FD and forward-mode AD as first-class options (Pain Point 1b)

**Files:**
- Modify: `include/tycho/detail/ode.h` — expose derivative mode in ODE definition
- Test: `tests/cpp/optimal_control/test_ode_fd_mode.cpp`

**Step 1: Write test**

```cpp
// ODE with finite-difference Jacobian
struct FD_ODE_Impl : Tycho::ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Tycho::Arguments<5>();
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();
        return Tycho::StackedOutputs{
            Tycho::sin(theta) * v,
            -1.0 * Tycho::cos(theta) * v,
            g * Tycho::cos(theta)
        };
    }
};

// Use FD for Jacobian (avoids template explosion)
BUILD_ODE_FROM_EXPRESSION_FD(FD_ODE, FD_ODE_Impl, double);

TEST(FDODE, ConvergesWithFD) {
    FD_ODE ode(9.81);
    auto phase = std::make_shared<Tycho::ODEPhase<FD_ODE>>(
        ode, Tycho::TranscriptionModes::LGL3, traj, 32);
    // ... add constraints ...
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, Tycho::PSIOPT::ConvergenceFlags::CONVERGED);
}
```

**Step 2: Create `BUILD_ODE_FROM_EXPRESSION_FD` macro**

This macro is identical to `BUILD_ODE_FROM_EXPRESSION` but specifies
`DenseDerivativeMode::ForwardFiniteDiff` for the Jacobian mode:

```cpp
#define BUILD_ODE_FROM_EXPRESSION_FD(Name, Impl, ...)                       \
    /* Same as BUILD_ODE_FROM_EXPRESSION but with FD Jacobian mode */
```

The VectorFunction infrastructure already supports FD via the `Jm` template
parameter. The key is threading this parameter through the `ODE_Expression`
and `VectorExpression` base classes.

**Step 3: Also provide a forward-mode AD variant**

```cpp
#define BUILD_ODE_FROM_EXPRESSION_FWAD(Name, Impl, ...)
```

**Step 4: Build, test, commit**

---

### Task 6: Implement compile-time named variables (Pain Point 3)

**Files:**
- Create: `include/tycho/detail/static_var_tags.h`
- Test: `tests/cpp/vector_functions/test_static_var_tags.cpp`

**Step 1: Write test**

```cpp
TEST(StaticVarTags, IndexAccess) {
    constexpr auto x = Tycho::XVar<0>;
    constexpr auto y = Tycho::XVar<1>;
    constexpr auto theta = Tycho::UVar<0>;

    auto args = Tycho::Arguments<4>();
    auto x_expr = args[x];   // Same as args.coeff<0>()
    auto y_expr = args[y];   // Same as args.coeff<1>()
    auto th_expr = args[theta]; // Maps to correct index based on ODE sizing

    Eigen::Vector4d input(1.0, 2.0, 3.0, 0.5);
    Eigen::VectorXd output(1);
    x_expr.compute(input, output);
    EXPECT_DOUBLE_EQ(output(0), 1.0);
}
```

**Step 2: Implement var tags**

```cpp
namespace Tycho {

// Compile-time variable index tags
template <int I>
struct XVarTag { static constexpr int index = I; };

template <int I>
struct UVarTag { static constexpr int index = I; };

template <int I>
struct PVarTag { static constexpr int index = I; };

// Convenience constexpr instances
template <int I> inline constexpr XVarTag<I> XVar{};
template <int I> inline constexpr UVarTag<I> UVar{};
template <int I> inline constexpr PVarTag<I> PVar{};

// Arguments::operator[] overloads for var tags
// In the Arguments<N> class or via free function:
template <int N, int I>
auto operator[](const Arguments<N>& args, XVarTag<I>) {
    return args.template coeff<I>();
}

} // namespace Tycho
```

The mapping from `UVar<I>` to the correct index in the full XtUP vector
depends on the ODE sizing. For `Arguments<N>` (standalone, no ODE context),
the user provides the raw index. For `ODEArguments<XV, UV, PV>`, the tag
knows the offset:

```cpp
// ODEArguments provides sized access
template <int XV, int UV, int PV>
struct StaticODEArguments {
    static constexpr int xtup_size = XV + 1 + UV + PV;

    template <int I>
    auto operator[](XVarTag<I>) const {
        static_assert(I < XV, "XVar index out of range");
        return Arguments<xtup_size>().template coeff<I>();
    }

    template <int I>
    auto operator[](UVarTag<I>) const {
        static_assert(I < UV, "UVar index out of range");
        return Arguments<xtup_size>().template coeff<XV + 1 + I>();
    }
};
```

**Step 3: Build, test, commit**

---

### Task 7: Prototype ODE definition syntax options (Pain Point 5)

**Files:**
- Create: `examples/cpp_examples/static/brachistochrone/main_option_a.cpp`
- Create: `examples/cpp_examples/static/brachistochrone/main_option_b.cpp`

This task produces TWO alternative implementations for evaluation, not a final
choice.

**Step 1: Implement Option A — Single-struct, macro-free**

```cpp
struct Brachistochrone : Tycho::StaticODE<3, 1, 0> {
    double g;
    Brachistochrone(double g) : g(g) {}

    auto ode() const {
        constexpr auto v = XVar<2>;
        constexpr auto theta = UVar<0>;

        auto args = this->args();
        return Tycho::stack(
            Tycho::sin(args[theta]) * args[v],
            -1.0 * Tycho::cos(args[theta]) * args[v],
            g * Tycho::cos(args[theta])
        );
    }
};
```

Requires implementing `Tycho::StaticODE<XV, UV, PV>` base class.

**Step 2: Implement Option B — Refined macro**

```cpp
TYCHO_DEFINE_ODE(Brachistochrone, XVars=3, UVars=1, Params=(double g)) {
    constexpr auto v = XVar<2>;
    constexpr auto theta = UVar<0>;

    return Tycho::stack(
        Tycho::sin(args[theta]) * args[v],
        -1.0 * Tycho::cos(args[theta]) * args[v],
        g * Tycho::cos(args[theta])
    );
}
```

Requires designing the `TYCHO_DEFINE_ODE` macro.

**Step 3: Compile both and compare**

Document for each option:
- Lines of code
- Compile time
- Peak memory
- Readability assessment
- Integration with `ODEPhase` and `Integrator`
- How well it handles multi-parameter ODEs

**Step 4: Commit both prototypes**

```bash
git commit -m "feat: prototype two ODE definition syntax options (A and B)"
```

The final choice between A and B will be made based on evaluation. Both
prototypes are committed for reference.

---

### Task 8: Full validation

**Step 1: Run all C++ tests**

```bash
cd build && ninja -j2 tycho_tests && ctest --output-on-failure
```

**Step 2: Run all examples (static DSL, builder, Python)**

```bash
# Static DSL examples
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
./examples/cpp_examples/static/zermelo/zermelo_cpp
# ... etc ...

# Python examples
python scripts/run_examples.py
```

**Step 3: Verify backward compatibility**

The existing `BUILD_ODE_FROM_EXPRESSION` macro must still work exactly as before.
All existing code using the old macro should compile without changes.

---

## Validation Checklist

- [ ] `Scaled<Scaled<...>>` collapses to single `Scaled`
- [ ] `BUILD_ODE_FROM_EXPRESSION` works with multiple parameters
- [ ] `Tycho::barrier()` type-erases expressions to `GenericFunction`
- [ ] FD Jacobian mode available via `BUILD_ODE_FROM_EXPRESSION_FD`
- [ ] Forward-mode AD available via `BUILD_ODE_FROM_EXPRESSION_FWAD`
- [ ] `XVar<I>`, `UVar<I>`, `PVar<I>` compile-time tags work
- [ ] Both ODE definition syntax options (A and B) compile and produce correct results
- [ ] All existing code still compiles and runs (backward compatibility)
- [ ] All C++ tests pass
- [ ] All Python examples pass
