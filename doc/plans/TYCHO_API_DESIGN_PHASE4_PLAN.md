# Phase 4: C++ Examples (Static DSL) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Implement three new C++ examples (Zermelo, Delta3Launch, HyperSensLong) using the current static VF DSL, mirroring their Python counterparts. These validate the C++ API end-to-end and surface pain points for phases 5-7.

**Architecture:** Each example is a standalone CMake project that consumes Tycho via `find_package(tycho)`. Examples live under `examples/cpp_examples/static/`. The existing brachistochrone moves into this structure. For Tycho's own CI, a top-level CMake option builds them in-tree.

**Tech Stack:** C++20, CMake, Tycho static VF DSL

**Threading model note:** Tycho now uses a global work-stealing thread pool controlled by `Tycho::set_num_threads(n)`. Per-phase work distribution is configured via `phase->setNumPartitions(n)` (not the old `setThreads()`). Examples should demonstrate this two-level model where appropriate: `set_num_threads()` for the global pool size, `setNumPartitions()` for per-phase partitioning.

---

### Task 1: Create feature branch and reorganize example structure

**Files:**
- Move: `examples/cpp_examples/brachistochrone/` -> `examples/cpp_examples/static/brachistochrone/`
- Create: `examples/cpp_examples/static/` directory structure

**Step 1: Create branch**

```bash
git checkout -b feat/cpp-examples main
```

**Step 2: Reorganize**

```bash
mkdir -p examples/cpp_examples/static
git mv examples/cpp_examples/brachistochrone examples/cpp_examples/static/brachistochrone
```

**Step 3: Update CMakeLists.txt references**

Update the root or parent `CMakeLists.txt` that references the brachistochrone
subdirectory to point to the new location.

Update `examples/cpp_examples/static/brachistochrone/CMakeLists.txt` to be a
standalone project:

```cmake
cmake_minimum_required(VERSION 3.20)
project(brachistochrone CXX)

find_package(tycho REQUIRED)

add_executable(brachistochrone_cpp main.cpp)
target_link_libraries(brachistochrone_cpp PRIVATE tycho::tycho)
```

**Step 4: Update brachistochrone includes and namespace**

Update `main.cpp`:
```cpp
#include <tycho/tycho.h>
using namespace Tycho;
```

(These changes may already be done from Phase 2.)

**Step 5: Build and verify brachistochrone still works**

```bash
cd build && ninja -j2 all
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found", objective ~1.8013 s.

**Step 6: Commit**

```bash
git add examples/
git commit -m "refactor: move brachistochrone to examples/cpp_examples/static/"
```

---

### Task 2: Implement Zermelo C++ example

**Files:**
- Create: `examples/cpp_examples/static/zermelo/CMakeLists.txt`
- Create: `examples/cpp_examples/static/zermelo/main.cpp`

**Step 1: Create standalone CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(zermelo CXX)

find_package(tycho REQUIRED)

add_executable(zermelo_cpp main.cpp)
target_link_libraries(zermelo_cpp PRIVATE tycho::tycho)
```

**Step 2: Implement Zermelo ODE**

The Python Zermelo has 2 states (x, y), 1 control (theta), and composable wind
functions. The C++ static DSL equivalent:

```cpp
#include <tycho/tycho.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace Tycho;

// Wind function type: takes Arguments<3> (x, y, t), returns 2 outputs (wx, wy)
// Uniform wind model
struct UniformWind_Impl : ODESize<2, 0, 0> {
    static auto Definition(double vel, double ang) {
        auto args = Arguments<3>();
        // Wind is constant — doesn't depend on position/time
        // But we still need a VectorFunction that takes 3 inputs
        return StackedOutputs{
            Value<double, 3>(vel * std::cos(ang)),
            Value<double, 3>(vel * std::sin(ang))
        };
    }
};

// Variable-direction wind model
struct VariableDirWind_Impl {
    static auto Definition() {
        auto args = Arguments<3>();
        auto x = args.coeff<0>();
        auto y = args.coeff<1>();

        auto pos = args.head<2>();
        auto vel = sin(pos.norm());
        auto ang = (x + y) * 2.0;

        return StackedOutputs{vel * cos(ang), vel * sin(ang)};
    }
};

// Zermelo ODE: xdot = vMax*cos(theta) + wx, ydot = vMax*sin(theta) + wy
// This is time-dependent, so we use Arguments directly rather than ODEArguments
template <typename WindFunc>
struct ZermeloODE_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax, WindFunc wFunc) {
        // Input: [x, y, t, theta]
        auto args = Arguments<4>();
        auto xyt = args.head<3>();  // [x, y, t]
        auto theta = args.coeff<3>();

        // Get wind components
        auto wind = wFunc(xyt);  // Returns [wx, wy]
        // ... compose with boat velocity ...
    }
};
```

**Key C++ challenges to document as pain points:**
- Composing wind functions with the ODE requires careful template management
- The `BUILD_ODE_FROM_EXPRESSION` macro may need workarounds for template wind types
- Manual index tracking: `args.coeff<0>()` vs Python's `XtU.XVar(0)`
- Time-dependent models need `Arguments<N>` directly, not `ODEArguments`

**Step 3: Implement phase setup and solve**

Mirror the Python `navigate()` function:
- Linear interpolation initial guess
- LGL3 collocation, boundary values, control bounds
- Delta-time objective
- Solve and extract trajectory

Implement at least the `uniformWind` and `variableDirWind` cases.

**Step 4: Add convergence verification**

```cpp
auto status = phase->solve_optimize();
if (status == PSIOPT::ConvergenceFlags::CONVERGED) {
    std::cout << "Zermelo: optimal solution found" << std::endl;
    auto traj = phase->returnTraj();
    // Print final time
    std::cout << "Final time: " << traj.back()(2) << std::endl;
} else {
    std::cerr << "Zermelo: FAILED to converge" << std::endl;
    return 1;
}
```

**Step 5: Document pain points as comments**

At the top of the file, add a comment block documenting difficulties encountered:
```cpp
// === PAIN POINTS (for Phase 7 static DSL improvements) ===
// 1. ...
// 2. ...
```

**Step 6: Build and verify**

```bash
cd build && ninja -j2 all
./examples/cpp_examples/static/zermelo/zermelo_cpp
```

Compare solution quality against the Python version.

**Step 7: Commit**

```bash
git add examples/cpp_examples/static/zermelo/
git commit -m "feat: add Zermelo C++ example (static DSL)"
```

---

### Task 3: Implement HyperSensLong C++ example

**Files:**
- Create: `examples/cpp_examples/static/hypersens/CMakeLists.txt`
- Create: `examples/cpp_examples/static/hypersens/main.cpp`

**Step 1: Create standalone CMakeLists.txt**

Same pattern as Zermelo.

**Step 2: Implement HyperSens ODE**

This is a trivially simple ODE (1 state, 1 control):

```cpp
// xdot = -x + u (linear version)
// xdot = -x^3 + u (cubed version)
struct HyperSens_Impl : ODESize<1, 1, 0> {
    static auto Definition() {
        auto args = Arguments<3>();  // [x, t, u]
        auto x = args.coeff<0>();
        auto u = args.coeff<2>();
        return -x + u;  // scalar output, no StackedOutputs needed
    }
};
BUILD_ODE_FROM_EXPRESSION(HyperSens, HyperSens_Impl);
```

**Step 3: Implement phase setup with mesh refinement**

This example exercises the full mesh refinement API:

```cpp
auto phase = std::make_shared<ODEPhase<HyperSens>>(
    ode, TranscriptionModes::LGL7, traj_ig, 50);

phase->setControlMode(ControlModes::NoSpline);

// Boundary conditions
Eigen::VectorXi front_idx(2); front_idx << 0, 1;
Eigen::VectorXd front_val(2); front_val << xt0, 0.0;
phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val);

Eigen::VectorXi back_idx(2); back_idx << 0, 1;
Eigen::VectorXd back_val(2); back_val << xtf, tf;
phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val);

// Integral objective: minimize integral of (x^2 + u^2)/2
auto obj_func = Arguments<2>();  // receives [x, u]
auto obj = obj_func.squared_norm() / 2.0;
Eigen::VectorXi obj_vars(2); obj_vars << 0, 2;
phase->addIntegralObjective(GenericFunction<-1, -1>(obj), obj_vars);

// Variable bounds
phase->addLUVarBound(PhaseRegionFlags::Path, 0, -50.0, 50.0);
phase->addLUVarBound(PhaseRegionFlags::Path, 2, -50.0, 50.0);

// Solver configuration
phase->optimizer.set_OptLSMode("L1");
phase->optimizer.set_SoeLSMode("L1");
phase->optimizer.set_MaxLSIters(2);
phase->optimizer.PrintLevel = 2;
phase->setNumPartitions(1, 1);
phase->optimizer.set_QPOrderingMode("MINDEG");

// Adaptive mesh refinement
phase->setAdaptiveMesh(true);
phase->setMeshTol(1e-7);
phase->optimizer.set_EContol(1e-7);
phase->setMaxMeshIters(10);
phase->setMeshErrorEstimator(MeshErrorEstimators::DEBOOR);
phase->setMeshErrorCriteria(MeshErrorAggregation::MAX);
phase->setMeshIncFactor(5.0);
phase->setMeshRedFactor(0.5);
phase->setMeshErrFactor(10.0);
```

**Step 4: Check convergence and mesh status**

```cpp
auto flag = phase->optimize_solve();
if (phase->MeshConverged) {
    std::cout << "HyperSens: mesh converged" << std::endl;
} else {
    std::cerr << "HyperSens: mesh did NOT converge" << std::endl;
}
```

**Step 5: Document pain points**

**Step 6: Build and verify**

```bash
cd build && ninja -j2 all
./examples/cpp_examples/static/hypersens/hypersens_cpp
```

**Step 7: Commit**

```bash
git add examples/cpp_examples/static/hypersens/
git commit -m "feat: add HyperSensLong C++ example (static DSL)"
```

---

### Task 4: Implement Delta3Launch C++ example

**Files:**
- Create: `examples/cpp_examples/static/delta3_launch/CMakeLists.txt`
- Create: `examples/cpp_examples/static/delta3_launch/main.cpp`

**This is the most complex example.** 7 states + 3 controls, 4 phases, multi-phase
linking, custom VectorFunction constraints, adaptive mesh.

**Step 1: Create standalone CMakeLists.txt**

Same pattern. Note this may need reduced parallelism for compilation due to
template cost (~8-10 GB for `Arguments<11>`):

```cmake
cmake_minimum_required(VERSION 3.20)
project(delta3_launch CXX)

find_package(tycho REQUIRED)

add_executable(delta3_launch_cpp main.cpp)
target_link_libraries(delta3_launch_cpp PRIVATE tycho::tycho)

# This example uses Arguments<11> which is expensive to compile
# Recommend -j1 if memory is limited
```

**Step 2: Implement RocketODE**

```cpp
struct RocketODE_Impl : ODESize<7, 3, 0> {
    static auto Definition(double T, double mdot) {
        auto args = Arguments<11>(); // [R(3), V(3), m, t, u(3)]
        auto R = args.head<3>();
        auto V = args.segment<3, 3>();
        auto m = args.coeff<6>();
        auto u = args.tail<3>().normalized();

        auto h = R.norm() - Re;
        auto rho = RhoAir * exp(-h / h_scale);
        auto Vr = V + R.cross(Eigen::Vector3d(0, 0, We));
        auto D = (-0.5 * CD * S) * rho * (Vr * Vr.norm());

        auto Rdot = V;
        auto Vdot = (-mu) * R.normalized_power3() + (T * u + D) / m;

        return StackedOutputs{Rdot, Vdot, Value<double, 11>(-mdot)};
    }
};
BUILD_ODE_FROM_EXPRESSION(RocketODE, RocketODE_Impl, double, double);
```

**WARNING:** The `BUILD_ODE_FROM_EXPRESSION` macro currently only supports single
parameters. The two-parameter `(T, mdot)` case will likely hit the limitation
documented in TODO.md. This is a key pain point to document.

**Workaround options:**
- Struct wrapper: `struct RocketParams { double T, mdot; };`
- Hardcode parameters and create 4 separate ODE types
- Use GenericODE with runtime function construction

**Step 3: Implement TargetOrbit constraint function**

```cpp
// Custom VectorFunction for orbital element targeting
auto TargetOrbit(double at, double et, double it, double Ot, double Wt) {
    auto args = Arguments<6>();
    auto R = args.head<3>();
    auto V = args.tail<3>();
    // ... orbital element computation ...
    // Returns 5-element vector: [a-at, e-et, i-it, O-Ot, W-Wt]
}
```

**Step 4: Set up 4 phases with linking**

```cpp
auto ode1 = RocketODE(T_phase1, mdot_phase1);
// ... ode2, ode3, ode4 ...

auto phase1 = std::make_shared<ODEPhase<RocketODE>>(
    ode1, TranscriptionModes::LGL3, IG1, nsegs1);
// ... configure boundary values, bounds ...

// Multi-phase problem
auto ocp = OptimalControlProblem();
ocp.addPhase(phase1);
ocp.addPhase(phase2);
ocp.addPhase(phase3);
ocp.addPhase(phase4);

// Link phases: continuous in everything except mass (index 6)
Eigen::VectorXi link_vars(10);
link_vars << 0, 1, 2, 3, 4, 5, 7, 8, 9, 10;
ocp.addForwardLinkEqualCon(phase1, phase4, link_vars);
```

**Step 5: Document ALL pain points extensively**

This example will surface the most issues. Expected pain points:
1. `BUILD_ODE_FROM_EXPRESSION` single-parameter limitation
2. Manual index vectors for boundary conditions and link constraints
3. Template memory usage for `Arguments<11>`
4. Verbose phase construction (initial guess generation is manual)
5. No named variable groups
6. Astro utility functions (classic_to_cartesian) may need special handling

**Step 6: Build and verify**

```bash
cd build && ninja -j1 delta3_launch_cpp  # -j1 due to template memory
./examples/cpp_examples/static/delta3_launch/delta3_launch_cpp
```

Compare final mass against Python version.

**Step 7: Commit**

```bash
git add examples/cpp_examples/static/delta3_launch/
git commit -m "feat: add Delta3Launch C++ example (static DSL)"
```

---

### Task 5: Update top-level CMake for in-tree example builds

**Files:**
- Modify: `CMakeLists.txt` (root) or `examples/cpp_examples/CMakeLists.txt`

**Step 1: Add subdirectories for new examples**

When `BUILD_CPP_EXAMPLES` is ON and building in-tree, add all static examples:

```cmake
if(BUILD_CPP_EXAMPLES)
    add_subdirectory(examples/cpp_examples/static/brachistochrone)
    add_subdirectory(examples/cpp_examples/static/zermelo)
    add_subdirectory(examples/cpp_examples/static/hypersens)
    add_subdirectory(examples/cpp_examples/static/delta3_launch)
endif()
```

When building in-tree, the examples use the `tycho::tycho` alias target directly
(no `find_package` needed). The standalone CMakeLists.txt should handle both modes:

```cmake
cmake_minimum_required(VERSION 3.20)
project(zermelo CXX)

# If building standalone, find tycho. If in-tree, target already exists.
if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(zermelo_cpp main.cpp)
target_link_libraries(zermelo_cpp PRIVATE tycho::tycho)
```

**Step 2: Build and verify all examples**

```bash
cd build && ninja -j2 all
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
./examples/cpp_examples/static/zermelo/zermelo_cpp
./examples/cpp_examples/static/hypersens/hypersens_cpp
./examples/cpp_examples/static/delta3_launch/delta3_launch_cpp
```

**Step 3: Commit**

```bash
git add CMakeLists.txt examples/
git commit -m "feat: integrate static C++ examples into build system"
```

---

### Task 6: Record compile times and pain points summary

**Files:**
- Create: `examples/cpp_examples/static/PAIN_POINTS.md`

**Step 1: Record compile metrics**

For each example, record:
- Compile time (wall clock)
- Peak memory usage during compilation
- Number of template instantiations (if measurable via `-ftime-trace`)

**Step 2: Compile pain points document**

Aggregate all pain points documented in the example source files into a summary
that will inform phases 5-7:

```markdown
# C++ Static DSL Pain Points

Documented during Phase 4 C++ example implementation.

## Pain Point 1: Manual index tracking
...

## Pain Point 2: BUILD_ODE_FROM_EXPRESSION single-parameter limitation
...

## Pain Point 3: Template memory explosion
...

## Pain Point N: ...
```

**Step 3: Commit**

```bash
git add examples/cpp_examples/static/PAIN_POINTS.md
git commit -m "docs: document C++ static DSL pain points from example implementation"
```

---

## Validation Checklist

Before marking Phase 4 complete:

- [ ] Brachistochrone moved to `static/` and still converges
- [ ] Zermelo converges with at least 2 wind models
- [ ] HyperSensLong converges with adaptive mesh
- [ ] Delta3Launch converges (4-phase problem)
- [ ] All examples are standalone `find_package(tycho)` projects
- [ ] All examples also build in-tree via `BUILD_CPP_EXAMPLES`
- [ ] Pain points documented in source files and summary document
- [ ] Compile times and memory usage recorded
- [ ] All 38 Python examples still pass
