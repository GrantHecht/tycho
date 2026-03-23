# Phase 5: Builder API (Runtime ODE Interface) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Create a C++ equivalent of Python's `ODEBase` — an ergonomic runtime ODE definition interface with string-named variables, `make_input()`, `make_units()`, and phase construction helpers. Zero overhead when names are not used.

**Architecture:** Three new public types: `Tycho::ODEBuilder` (fluent builder), `Tycho::RuntimeODE` (built ODE + variable registry), and `Tycho::Phase` (wrapper around `ODEPhaseBase` that adds named-variable overloads). The builder produces a `GenericODE` internally. The `Phase` wrapper delegates to index-based methods after resolving names via a map lookup.

**Tech Stack:** C++20, `std::unordered_map<std::string, ...>`, `GenericFunction<-1,-1>`, `GenericODE`

---

### Task 1: Create feature branch

```bash
git checkout -b feat/builder-api main
```

---

### Task 2: Design and implement `VarRegistry`

**Files:**
- Create: `include/tycho/detail/var_registry.h`
- Test: `tests/cpp/utils/test_var_registry.cpp`

The variable registry is the core data structure shared by `RuntimeODE` and `Phase`.

**Step 1: Write test**

```cpp
TEST(VarRegistry, SingleVarLookup) {
    Tycho::VarRegistry reg(/*XVars=*/3, /*UVars=*/1, /*PVars=*/0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);
    reg.add_name("v", 2);
    reg.add_name("theta", 3);  // index in XtUP space (X=0-2, t=3... wait)

    auto idx = reg.resolve("x");
    EXPECT_EQ(idx.size(), 1);
    EXPECT_EQ(idx[0], 0);
}

TEST(VarRegistry, GroupLookup) {
    Tycho::VarRegistry reg(7, 3, 1);
    reg.add_group("MEEs", 0, 5);   // indices 0-5
    reg.add_name("w", 6);
    reg.add_name("t", 7);          // time index in XtUP space

    auto idx = reg.resolve({"MEEs", "w", "t"});
    EXPECT_EQ(idx.size(), 8);
    // MEEs: 0,1,2,3,4,5; w: 6; t: 7
}

TEST(VarRegistry, MakeInput) {
    Tycho::VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);
    reg.add_name("v", 2);
    reg.add_name("theta", 3);

    auto vec = reg.make_input({{"x", 1.0}, {"y", 2.0}, {"v", 0.0}, {"theta", 0.5}});
    EXPECT_EQ(vec.size(), 5);  // XtUP size = 3 + 1 + 1 + 0 = 5
    EXPECT_DOUBLE_EQ(vec[0], 1.0);
    EXPECT_DOUBLE_EQ(vec[1], 2.0);
    EXPECT_DOUBLE_EQ(vec[3], 0.0);  // time defaults to 0
    EXPECT_DOUBLE_EQ(vec[4], 0.5);  // theta is UVar, maps to index 4
}
```

**Step 2: Implement `VarRegistry`**

```cpp
namespace Tycho {

class VarRegistry {
public:
    VarRegistry(int xvars, int uvars, int pvars);

    // Add a single named variable (maps name -> single index in XtUP space)
    void add_name(const std::string& name, int xtup_index);

    // Add a named group spanning a contiguous range [start, end] inclusive
    void add_group(const std::string& name, int start, int end);

    // Add a named group from a list of individual names
    void add_group(const std::string& name,
                   std::initializer_list<std::string> members);

    // Resolve a single name -> indices
    Eigen::VectorXi resolve(const std::string& name) const;

    // Resolve multiple names -> concatenated indices
    Eigen::VectorXi resolve(std::initializer_list<std::string> names) const;

    // Build an input vector from name-value pairs
    Eigen::VectorXd make_input(
        std::initializer_list<std::pair<std::string, double>> scalar_vals) const;

    // Build a units vector from name-value pairs (unset entries = 1.0)
    Eigen::VectorXd make_units(
        std::initializer_list<std::pair<std::string, double>> unit_vals) const;

    // Extract named values from a state vector
    std::vector<double> get_vars(
        std::initializer_list<std::string> names,
        const Eigen::VectorXd& state) const;

    int xvars() const;
    int uvars() const;
    int pvars() const;
    int xtup_size() const;  // XV + 1 + UV + PV

private:
    int xvars_, uvars_, pvars_;
    std::unordered_map<std::string, Eigen::VectorXi> name_to_indices_;
};

} // namespace Tycho
```

**Step 3: Build and run tests**

```bash
cd build && ninja -j2 tycho_tests && ctest --output-on-failure -R VarRegistry
```

**Step 4: Commit**

```bash
git add include/tycho/detail/var_registry.h src/Utils/VarRegistry.cpp tests/cpp/utils/test_var_registry.cpp
git commit -m "feat: implement VarRegistry for named variable resolution"
```

---

### Task 3: Implement `RuntimeODE`

**Files:**
- Create: `include/tycho/detail/runtime_ode.h`
- Test: `tests/cpp/optimal_control/test_runtime_ode.cpp`

**Step 1: Write test**

```cpp
TEST(RuntimeODE, BrachistochroneConstruction) {
    auto args = Tycho::ODEArguments(3, 1);
    auto XtU = args;
    auto x = XtU.XVec().coeff(0);
    auto y = XtU.XVec().coeff(1);
    auto v = XtU.XVec().coeff(2);
    auto theta = XtU.UVar(0);

    auto xdot = Tycho::sin(theta) * v;
    auto ydot = -1.0 * Tycho::cos(theta) * v;
    auto vdot = 9.81 * Tycho::cos(theta);
    auto ode_expr = Tycho::stack(xdot, ydot, vdot);

    Tycho::RuntimeODE ode(ode_expr, 3, 1, 0);
    EXPECT_EQ(ode.xvars(), 3);
    EXPECT_EQ(ode.uvars(), 1);
    EXPECT_EQ(ode.pvars(), 0);
}
```

**Step 2: Implement `RuntimeODE`**

```cpp
namespace Tycho {

class RuntimeODE {
public:
    // Construct from a pre-built VectorFunction expression
    template <typename Func>
    RuntimeODE(const Func& ode_func, int xvars, int uvars, int pvars = 0);

    // Named variable registration
    RuntimeODE& var_names(
        std::initializer_list<std::pair<std::string, int>> names);
    RuntimeODE& var_group(const std::string& name, int start, int end);
    RuntimeODE& var_group(const std::string& name,
                          std::initializer_list<std::string> members);

    // Phase construction (returns Tycho::Phase if names registered, raw phase otherwise)
    Phase phase(TranscriptionModes mode,
                const std::vector<Eigen::VectorXd>& traj,
                int num_segments) const;

    // Convenience methods (delegate to VarRegistry)
    Eigen::VectorXd make_input(
        std::initializer_list<std::pair<std::string, double>> vals) const;
    Eigen::VectorXd make_units(
        std::initializer_list<std::pair<std::string, double>> vals) const;
    std::vector<double> get_vars(
        std::initializer_list<std::string> names,
        const Eigen::VectorXd& state) const;

    // Integrator construction
    // ... (with control law support)

    // Access underlying GenericODE for advanced use
    const auto& generic_ode() const;

    int xvars() const;
    int uvars() const;
    int pvars() const;

private:
    // Internal GenericODE (type-erased)
    // The exact GenericODE<...> type depends on XV, UV, PV
    // We need a type-erased wrapper since these are runtime values
    struct Impl;
    std::shared_ptr<Impl> impl_;
    VarRegistry registry_;
};

} // namespace Tycho
```

**Key design decision:** Since XV, UV, PV are runtime values, we can't use
`GenericODE<GenericFunction<-1,-1>, XV, UV, PV>` as a concrete type (the template
parameters must be compile-time). Options:

1. Use `GenericODE<GenericFunction<-1,-1>, -1, -1, -1>` if it exists (fully dynamic)
2. Use a jump table to dispatch to the correct `GenericODE<..., XV, UV, PV>` for
   common sizes
3. Store the ODE as a `GenericFunction<-1,-1>` directly and reconstruct the
   `GenericODE` when creating phases

Option 3 is simplest. The `RuntimeODE` stores the function + size metadata, and
constructs the appropriately-typed `GenericODE` on demand (when creating a phase).

**Step 3: Build and test**

**Step 4: Commit**

---

### Task 4: Implement `ODEBuilder`

**Files:**
- Create: `include/tycho/detail/ode_builder.h`
- Test: `tests/cpp/optimal_control/test_ode_builder.cpp`

**Step 1: Write test**

```cpp
TEST(ODEBuilder, LambdaDefinition) {
    auto ode = Tycho::ODEBuilder(3, 1)
        .define([](auto& args) {
            auto x = args.XVar(0);
            auto y = args.XVar(1);
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return Tycho::stack(
                Tycho::sin(theta) * v,
                -1.0 * Tycho::cos(theta) * v,
                9.81 * Tycho::cos(theta)
            );
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 3}})
        .build();

    EXPECT_EQ(ode.xvars(), 3);
    EXPECT_EQ(ode.uvars(), 1);

    auto input = ode.make_input({{"x", 0}, {"y", 10}, {"v", 0}, {"theta", 1.0}});
    EXPECT_EQ(input.size(), 5);
}

TEST(ODEBuilder, FromExpression) {
    auto XtU = Tycho::ODEArguments(3, 1);
    auto theta = XtU.UVar(0);
    auto v = XtU.XVar(2);
    auto expr = Tycho::stack(
        Tycho::sin(theta) * v,
        -1.0 * Tycho::cos(theta) * v,
        9.81 * Tycho::cos(theta)
    );

    auto ode = Tycho::ODEBuilder(3, 1)
        .from(expr)
        .build();

    EXPECT_EQ(ode.xvars(), 3);
}
```

**Step 2: Implement `ODEBuilder`**

```cpp
namespace Tycho {

class ODEBuilder {
public:
    ODEBuilder(int xvars, int uvars, int pvars = 0);

    // Define ODE via lambda (receives ODEArguments-like object)
    template <typename F>
    ODEBuilder& define(F&& func);

    // Define ODE from pre-built expression
    template <typename Func>
    ODEBuilder& from(const Func& ode_expr);

    // Named variables
    ODEBuilder& var_names(
        std::initializer_list<std::pair<std::string, int>> names);
    ODEBuilder& var_group(const std::string& name, int start, int end);

    // Build the RuntimeODE
    RuntimeODE build();

private:
    int xvars_, uvars_, pvars_;
    GenericFunction<-1, -1> ode_func_;
    bool func_set_ = false;
    std::vector<std::pair<std::string, int>> pending_names_;
    std::vector<std::tuple<std::string, int, int>> pending_groups_;
};

} // namespace Tycho
```

The `define()` lambda receives an `ODEArguments`-like proxy object that provides
`XVar()`, `UVar()`, `PVar()`, `TVar()`, `XVec()`, `UVec()` accessors. Internally
it creates the expression using `Tycho::ODEArguments` and type-erases the result.

**Step 3: Build and test**

**Step 4: Commit**

---

### Task 5: Implement `Tycho::Phase` wrapper

**Files:**
- Create: `include/tycho/detail/phase_wrapper.h`
- Test: `tests/cpp/optimal_control/test_phase_wrapper.cpp`

**Step 1: Write test**

```cpp
TEST(PhaseWrapper, NamedBoundaryValue) {
    // ... create RuntimeODE with names ...
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Named overload
    phase.addBoundaryValue(PhaseRegionFlags::Front,
        {"x", "y", "v", "t"}, Eigen::Vector4d(0, 10, 0, 0));

    // Index overload (passthrough, no lookup)
    Eigen::VectorXi idx(2); idx << 0, 1;
    phase.addBoundaryValue(PhaseRegionFlags::Back, idx,
        Eigen::Vector2d(10, 5));
}
```

**Step 2: Implement `Tycho::Phase`**

```cpp
namespace Tycho {

class Phase {
public:
    Phase(std::shared_ptr<ODEPhaseBase> phase, VarRegistry registry);

    // === Named-variable overloads ===

    void addBoundaryValue(PhaseRegionFlags flag,
        std::initializer_list<std::string> var_names,
        const Eigen::VectorXd& values,
        const ScaleType& scale = "auto");

    void addLUVarBound(PhaseRegionFlags flag,
        const std::string& var_name,
        double lower, double upper, double scale = 1.0);

    void addEqualCon(PhaseRegionFlags flag,
        const GenericFunction<-1,-1>& func,
        const std::string& var_group);

    void addLUFuncBound(PhaseRegionFlags flag,
        const GenericFunction<-1,-1>& func,
        const std::string& var_group,
        double lower, double upper);

    void addValueObjective(PhaseRegionFlags flag,
        const std::string& var_name, double scale);

    void addIntegralObjective(const GenericFunction<-1,-1>& func,
        std::initializer_list<std::string> var_names);

    // ... other constraint/objective methods with name overloads ...

    // === Index-based overloads (direct passthrough) ===

    void addBoundaryValue(PhaseRegionFlags flag,
        const Eigen::VectorXi& indices,
        const Eigen::VectorXd& values,
        const ScaleType& scale = "auto");

    void addLUVarBound(PhaseRegionFlags flag,
        int index, double lower, double upper, double scale = 1.0);

    // ... all other ODEPhaseBase methods forwarded ...

    // === Phase configuration (direct delegation) ===

    void setAutoScaling(bool enable);
    void setAdaptiveMesh(bool enable);
    void setControlMode(ControlModes mode);
    void setMeshTol(double tol);
    void setMaxMeshIters(int iters);
    void setMeshErrorEstimator(MeshErrorEstimators est);
    void setMeshErrorCriteria(MeshErrorAggregation crit);
    void setMeshIncFactor(double factor);
    void setMeshRedFactor(double factor);
    void setMeshErrFactor(double factor);
    void setNumPartitions(int nparts);
    void setNumPartitions(int nparts, int qp_threads);
    void setUnits(const Eigen::VectorXd& units);

    // Solve
    PSIOPT::ConvergenceFlags optimize();
    PSIOPT::ConvergenceFlags solve_optimize();
    PSIOPT::ConvergenceFlags optimize_solve();

    // Results
    std::vector<Eigen::VectorXd> returnTraj();
    bool MeshConverged() const;

    // Access underlying phase and optimizer
    ODEPhaseBase& base();
    const ODEPhaseBase& base() const;
    std::shared_ptr<ODEPhaseBase> base_ptr();
    PSIOPT& optimizer();

    // Delta-time objective
    void addDeltaTimeObjective(double scale);

private:
    std::shared_ptr<ODEPhaseBase> phase_;
    VarRegistry registry_;
};

} // namespace Tycho
```

**Step 3: Build and test**

**Step 4: Commit**

---

### Task 6: Integration test — Brachistochrone via Builder API

**Files:**
- Test: `tests/cpp/optimal_control/test_builder_brachistochrone.cpp`

**Step 1: Write end-to-end test**

```cpp
TEST(BuilderAPI, BrachistochroneConverges) {
    auto ode = Tycho::ODEBuilder(3, 1)
        .define([](auto& args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return Tycho::stack(
                Tycho::sin(theta) * v,
                -1.0 * Tycho::cos(theta) * v,
                9.81 * Tycho::cos(theta)
            );
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 3}})
        .build();

    // Build initial guess
    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 100; ++i) {
        double s = static_cast<double>(i) / 99.0;
        Eigen::VectorXd pt(5);
        pt << 0 + 10 * s, 10 + (5 - 10) * s, 9.81 * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase = ode.phase(Tycho::TranscriptionModes::LGL3, traj, 32);
    phase.addBoundaryValue(Tycho::PhaseRegionFlags::Front,
        {"x", "y", "v", "t"}, Eigen::Vector4d(0, 10, 0, 0));
    phase.addBoundaryValue(Tycho::PhaseRegionFlags::Back,
        {"x", "y"}, Eigen::Vector2d(10, 5));
    phase.addLUVarBound(Tycho::PhaseRegionFlags::Path, "theta", -0.1, 2.0);
    phase.addDeltaTimeObjective(1.0);

    auto status = phase.optimize();
    EXPECT_EQ(status, Tycho::PSIOPT::ConvergenceFlags::CONVERGED);

    auto result = phase.returnTraj();
    double tf = result.back()(3);
    EXPECT_NEAR(tf, 1.8013, 0.01);
}
```

**Step 2: Build and run**

```bash
cd build && ninja -j2 tycho_tests && ctest --output-on-failure -R BuilderAPI
```

**Step 3: Commit**

```bash
git add tests/cpp/optimal_control/test_builder_brachistochrone.cpp
git commit -m "test: end-to-end Brachistochrone via Builder API"
```

---

### Task 7: Add public header entries and update umbrella

**Files:**
- Modify: `include/tycho/optimal_control.h` — add builder API headers
- Modify: `include/tycho/tycho.h` — ensure everything is accessible

**Step 1: Add to public headers**

```cpp
// In include/tycho/optimal_control.h, add:
#include "tycho/detail/var_registry.h"
#include "tycho/detail/runtime_ode.h"
#include "tycho/detail/ode_builder.h"
#include "tycho/detail/phase_wrapper.h"
```

**Step 2: Build and verify**

```bash
cd build && ninja -j2 all
```

**Step 3: Commit**

---

### Task 8: Full validation

**Step 1: Run all C++ tests**

```bash
cd build && ninja -j2 tycho_tests && ctest --output-on-failure
```

**Step 2: Run Python examples**

```bash
python scripts/run_examples.py
```

(Python should be unaffected by C++-only additions.)

**Step 3: Run C++ brachistochrone**

```bash
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
```

**Step 4: Verify builder test passes**

```bash
ctest --output-on-failure -R BuilderAPI
```

---

## Validation Checklist

Before marking Phase 5 complete:

- [ ] `VarRegistry` resolves single names, groups, and multiple names
- [ ] `VarRegistry::make_input()` and `make_units()` work correctly
- [ ] `ODEBuilder` accepts both lambda and pre-built expression forms
- [ ] `RuntimeODE` correctly constructs and stores GenericODE
- [ ] `Tycho::Phase` named overloads resolve to correct index-based calls
- [ ] `Tycho::Phase` index overloads are direct passthroughs
- [ ] End-to-end Brachistochrone via builder converges to correct solution
- [ ] All existing C++ tests still pass
- [ ] All 38 Python examples still pass

## Design Notes

**`make_input` with vector-valued groups:** For variable groups like `"U"` that
map to multiple indices, `make_input` needs to accept vector values:
```cpp
ode.make_input({{"x", 0.0}, {"y", 10.0}}, {{"U", vec3}});
```
This may need a separate overload or a variant type. Design during implementation.

**Thread safety:** `VarRegistry` is immutable after construction (all `add_*` calls
happen during building). The `Phase` wrapper's name resolution is therefore
thread-safe for read operations.

**`RuntimeODE` internal dispatch:** The `phase()` method needs to construct the
right `GenericODE<..., XV, UV, PV>` type. Since common sizes are finite
(XV from 1-10, UV from 0-5, PV from 0-3), a jump table or switch statement
over common combinations works. Uncommon sizes can fall back to a fully dynamic
path if one exists in the codebase.
