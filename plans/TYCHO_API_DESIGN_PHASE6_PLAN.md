# Phase 6: Builder API Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Re-implement the four C++ examples (Brachistochrone, Zermelo, Delta3Launch, HyperSensLong) using the Builder API. Validates the runtime API is functionally complete and ergonomic.

**Architecture:** Each example lives under `examples/cpp_examples/builder/` as a standalone `find_package(tycho)` project. Each mirrors its `static/` counterpart and must converge to the same solution. API gaps discovered during implementation feed back into the Builder API.

**Tech Stack:** C++20, CMake, Tycho Builder API

---

### Task 1: Create feature branch

```bash
git checkout -b feat/builder-examples main
```

---

### Task 2: Implement Brachistochrone builder example

**Files:**
- Create: `examples/cpp_examples/builder/brachistochrone/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/brachistochrone/main.cpp`

**Step 1: Create standalone CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(brachistochrone_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(brachistochrone_builder main.cpp)
target_link_libraries(brachistochrone_builder PRIVATE tycho::tycho)
```

**Step 2: Implement using Builder API**

```cpp
#include <tycho/tycho.h>
#include <iostream>

int main() {
    double g = 9.81;

    auto ode = tycho::ODEBuilder(3, 1)
        .define([g](auto& args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return tycho::stack(
                tycho::sin(theta) * v,
                -1.0 * tycho::cos(theta) * v,
                g * tycho::cos(theta)
            );
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 3}})
        .build();

    // Initial guess
    std::vector<Eigen::VectorXd> traj;
    double x0=0, y0=10, xf=10, yf=5, v0=0, tf=1.0, theta0=1.0;
    for (int i = 0; i < 100; ++i) {
        double t = static_cast<double>(i) / 99.0;
        Eigen::VectorXd pt(5);
        pt << x0 + (xf-x0)*t, y0 + (yf-y0)*t, g*t*tf*std::cos(theta0), t*tf, theta0;
        traj.push_back(pt);
    }

    auto phase = ode.phase(tycho::TranscriptionModes::LGL3, traj, 32);
    phase.addBoundaryValue(tycho::PhaseRegionFlags::Front,
        {"x","y","v","t"}, Eigen::Vector4d(x0, y0, v0, 0));
    phase.addBoundaryValue(tycho::PhaseRegionFlags::Back,
        {"x","y"}, Eigen::Vector2d(xf, yf));
    phase.addLUVarBound(tycho::PhaseRegionFlags::Path, "theta", -0.1, 2.0);
    phase.addDeltaTimeObjective(1.0);

    phase.optimize();

    auto result = phase.returnTraj();
    double final_time = result.back()(3);
    std::cout << "Brachistochrone (builder): optimal time = "
              << final_time << " s" << std::endl;
    return 0;
}
```

**Step 3: Build and verify**

```bash
cd build && ninja -j2 brachistochrone_builder
./examples/cpp_examples/builder/brachistochrone/brachistochrone_builder
```

Compare solution against static DSL and Python versions.

**Step 4: Commit**

```bash
git add examples/cpp_examples/builder/brachistochrone/
git commit -m "feat: add Brachistochrone builder API example"
```

---

### Task 3: Implement Zermelo builder example

**Files:**
- Create: `examples/cpp_examples/builder/zermelo/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/zermelo/main.cpp`

**Key differences from static DSL version:**
- Wind functions composed at runtime using `GenericFunction<-1,-1>`
- ODE constructed via `ODEBuilder::from()` with pre-built expression
- Named variables for all state/control variables

**Step 1: Implement with composable wind functions**

```cpp
// Wind as a GenericFunction
auto uniform_wind = [](double vel, double ang) {
    auto args = tycho::Arguments<3>();
    return tycho::stack(
        tycho::Value<double, 3>(vel * std::cos(ang)),
        tycho::Value<double, 3>(vel * std::sin(ang))
    );
};

auto make_zermelo_ode = [](double vMax, auto wind_func) {
    auto args = tycho::Arguments<4>(); // [x, y, t, theta]
    auto xyt = args.head<3>();
    auto theta = args.coeff<3>();
    auto wind = wind_func(xyt);
    auto xdot = vMax * tycho::cos(theta) + /* wx from wind */;
    auto ydot = vMax * tycho::sin(theta) + /* wy from wind */;

    return tycho::ODEBuilder(2, 1)
        .from(tycho::stack(xdot, ydot))
        .var_names({{"x", 0}, {"y", 1}, {"theta", 2}})
        .build();
};
```

**Step 2: Implement multiple navigation cases**

Mirror the Python `compareWind()` function — solve with multiple wind models.

**Step 3: Build, verify, commit**

---

### Task 4: Implement Delta3Launch builder example

**Files:**
- Create: `examples/cpp_examples/builder/delta3_launch/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/delta3_launch/main.cpp`

**Key features exercised:**
- `var_group()` for named groups (`"R"`, `"V"`, `"u"`)
- `make_input()` for initial conditions
- `make_units()` for auto-scaling
- `OptimalControlProblem` with multiple builder-constructed phases
- Custom VectorFunction constraints with named variable selection

**Step 1: Implement RocketODE with named variables**

```cpp
auto make_rocket_ode = [](double T, double mdot) {
    auto XtU = tycho::ODEArguments(7, 3);
    auto R = XtU.XVec().head(3);
    auto V = XtU.XVec().segment(3, 3);
    auto m = XtU.XVar(6);
    auto u = XtU.UVec().normalized();

    // ... dynamics ...

    return tycho::ODEBuilder(7, 3)
        .from(ode_expr)
        .var_group("R", 0, 2)
        .var_group("V", 3, 5)
        .var_names({{"m", 6}, {"t", 7}})
        .var_group("u", 8, 10)
        .build();
};
```

**Step 2: Set up 4 phases with named variables**

```cpp
auto phase1 = ode1.phase(tycho::TranscriptionModes::LGL3, IG1, nsegs1);
phase1.addBoundaryValue(tycho::PhaseRegionFlags::Front,
    {"R", "V", "m", "t"}, IG1[0].head(8));
phase1.addLUNormBound(tycho::PhaseRegionFlags::Path, "u", 0.5, 1.5);
```

**Step 3: Compare ergonomics against static DSL version**

Document the differences in code size, readability, and any API gaps.

**Step 4: Build, verify, commit**

---

### Task 5: Implement HyperSensLong builder example

**Files:**
- Create: `examples/cpp_examples/builder/hypersens/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/hypersens/main.cpp`

**Step 1: Implement**

```cpp
auto ode = tycho::ODEBuilder(1, 1)
    .define([](auto& args) {
        auto x = args.XVar(0);
        auto u = args.UVar(0);
        return -x + u;
    })
    .var_names({{"x", 0}, {"u", 1}})
    .build();

auto phase = ode.phase(tycho::TranscriptionModes::LGL7, traj_ig, 50);
phase.addBoundaryValue(tycho::PhaseRegionFlags::Front, {"x", "t"}, ...);
phase.addIntegralObjective(obj_func, {"x", "u"});
```

**Step 2: Exercise full mesh refinement API**

**Step 3: Build, verify, commit**

---

### Task 6: Update top-level CMake and full validation

**Files:**
- Modify: `CMakeLists.txt` — add builder example subdirectories

**Step 1: Add builder examples to in-tree build**

```cmake
if(BUILD_CPP_EXAMPLES)
    # Static DSL examples
    add_subdirectory(examples/cpp_examples/static/brachistochrone)
    # ... etc ...

    # Builder API examples
    add_subdirectory(examples/cpp_examples/builder/brachistochrone)
    add_subdirectory(examples/cpp_examples/builder/zermelo)
    add_subdirectory(examples/cpp_examples/builder/delta3_launch)
    add_subdirectory(examples/cpp_examples/builder/hypersens)
endif()
```

**Step 2: Build and run all examples**

```bash
cd build && ninja -j2 all
# Run all builder examples and compare against static versions
```

**Step 3: Run full test suite**

```bash
ctest --output-on-failure
python scripts/run_examples.py
```

**Step 4: Commit**

---

## Validation Checklist

- [ ] All 4 builder examples converge to same solutions as static/Python versions
- [ ] Named variables used throughout (no raw index vectors)
- [ ] `make_input()` and `make_units()` used where applicable
- [ ] `var_group()` used for multi-variable groups
- [ ] API gaps documented and fed back to Phase 5 if fixes needed
- [ ] All existing tests and Python examples still pass
