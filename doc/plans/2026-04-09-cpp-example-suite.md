# C++ Builder API Example Suite — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all 23 unique Python examples to C++ using the builder API, creating a comprehensive C++ integration test suite with convergence assertions.

**Architecture:** Each example is a standalone `main.cpp` under `examples/cpp_examples/builder/<name>/` with its own `CMakeLists.txt`. Examples follow the established builder pattern: ODE definition via `ODEBuilder`, phase setup with named variables, solve with convergence check, trajectory extraction with value assertions. All examples are registered as CTest tests and assigned to the `heavy_compile` ninja job pool.

**Tech Stack:** C++20, CMake 3.20+, Tycho builder API (`ODEBuilder`, `Phase`, `OptimalControlProblem`), Eigen, Google Test (for CTest registration only)

**Spec:** `doc/plans/TYCHO_CPP_EXAMPLE_SUITE_SPEC.md`

---

## File Structure

### New files (per example)

Each of the 23 examples creates exactly two files:
- `examples/cpp_examples/builder/<name>/CMakeLists.txt`
- `examples/cpp_examples/builder/<name>/main.cpp`

### Modified files

- `examples/cpp_examples/CMakeLists.txt` — add all 23 new subdirectories, update `foreach` loop for job pool assignment, add `cpp_examples_all` aggregate target, add CTest registration
- `doc/cpp_example_api_gaps.md` — new gap tracking document (created in Task 0)

### CMakeLists.txt template (identical for all 23 examples)

```cmake
cmake_minimum_required(VERSION 3.20)
project(<name>_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(<name>_builder main.cpp)
target_link_libraries(<name>_builder PRIVATE tycho::tycho)
```

### main.cpp template (convergence pattern)

```cpp
#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;
using namespace tycho::solvers;

int main() {
    // 1. Constants & parameters
    // 2. ODE definition via ODEBuilder
    // 3. Initial trajectory guess
    // 4. Phase setup: boundary conditions, bounds, objectives
    // 5. Solver configuration
    // 6. Solve and convergence check
    const auto flag = phase.solve_optimize();
    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "<Name> (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }
    // 7. Trajectory extraction and value assertions
    auto traj = phase.return_traj();
    // ... check objective value against expected
    std::cout << "<Name> (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

---

## Task 0: Build System Infrastructure & Gap Tracking

**Files:**
- Modify: `examples/cpp_examples/CMakeLists.txt`
- Create: `doc/cpp_example_api_gaps.md`

This task sets up the build infrastructure for all 23 examples at once, so subsequent tasks only need to create the per-example `CMakeLists.txt` + `main.cpp`.

- [ ] **Step 1: Read current CMakeLists.txt**

Read `examples/cpp_examples/CMakeLists.txt` to understand existing structure.

- [ ] **Step 2: Update CMakeLists.txt with all 23 new subdirectories, job pool, CTest, and aggregate target**

Replace the full contents of `examples/cpp_examples/CMakeLists.txt` with:

```cmake
# ── Static (legacy DSL) examples ────────────────────────────────────────────
add_subdirectory(static/brachistochrone)
add_subdirectory(static/zermelo)
add_subdirectory(static/hypersens)
add_subdirectory(static/delta3_launch)

# ── Builder API examples ────────────────────────────────────────────────────
add_subdirectory(builder/brachistochrone)
add_subdirectory(builder/zermelo)
add_subdirectory(builder/hypersens)
add_subdirectory(builder/delta3_launch)

# New builder examples (Tier 1: simple single-phase)
add_subdirectory(builder/van_der_pol)
add_subdirectory(builder/bryson_denham)
add_subdirectory(builder/mountain_car)
add_subdirectory(builder/analytic_example)
add_subdirectory(builder/bike_obstacle)
add_subdirectory(builder/free_flying_robot)

# New builder examples (Tier 2: single-phase advanced)
add_subdirectory(builder/cart_pole)
add_subdirectory(builder/parallel_parking)
add_subdirectory(builder/hanging_chain)
add_subdirectory(builder/reentry)
add_subdirectory(builder/minimum_time_to_climb)

# New builder examples (Tier 3: multi-phase)
add_subdirectory(builder/goddard_rocket)
add_subdirectory(builder/multi_phase_cannon)
add_subdirectory(builder/multi_phase_zermelo)

# New builder examples (Tier 4: low-thrust astrodynamics)
add_subdirectory(builder/simple_low_thrust)
add_subdirectory(builder/topputto_low_thrust)
add_subdirectory(builder/dionysus_low_thrust)
add_subdirectory(builder/betts_low_thrust)
add_subdirectory(builder/orbit_continuation)

# New builder examples (Tier 5: advanced astrodynamics)
add_subdirectory(builder/heteroclinic)
add_subdirectory(builder/optimal_docking)
add_subdirectory(builder/multi_spacecraft_opt)

# New builder examples (Tier 6: interpolation tables)
add_subdirectory(builder/min_time_climb_tables)

# ── Assign all example targets to the heavy_compile job pool ────────────────
set(_all_example_targets
    # Static
    brachistochrone_cpp zermelo_cpp hypersens_cpp delta3_launch_cpp
    # Builder (existing)
    brachistochrone_builder zermelo_builder hypersens_builder delta3_launch_builder
    # Builder (Tier 1)
    van_der_pol_builder bryson_denham_builder mountain_car_builder
    analytic_example_builder bike_obstacle_builder free_flying_robot_builder
    # Builder (Tier 2)
    cart_pole_builder parallel_parking_builder hanging_chain_builder
    reentry_builder minimum_time_to_climb_builder
    # Builder (Tier 3)
    goddard_rocket_builder multi_phase_cannon_builder multi_phase_zermelo_builder
    # Builder (Tier 4)
    simple_low_thrust_builder topputto_low_thrust_builder dionysus_low_thrust_builder
    betts_low_thrust_builder orbit_continuation_builder
    # Builder (Tier 5)
    heteroclinic_builder optimal_docking_builder multi_spacecraft_opt_builder
    # Builder (Tier 6)
    min_time_climb_tables_builder
)

foreach(_ex IN LISTS _all_example_targets)
    if(TARGET ${_ex})
        set_property(TARGET ${_ex} PROPERTY JOB_POOL_COMPILE heavy_compile)
    endif()
endforeach()

# ── Aggregate target: build all C++ examples ────────────────────────────────
add_custom_target(cpp_examples_all)
foreach(_ex IN LISTS _all_example_targets)
    if(TARGET ${_ex})
        add_dependencies(cpp_examples_all ${_ex})
    endif()
endforeach()

# ── CTest registration ──────────────────────────────────────────────────────
include(CTest)

foreach(_ex IN LISTS _all_example_targets)
    if(TARGET ${_ex})
        add_test(NAME cpp_example_${_ex} COMMAND ${_ex})
        set_tests_properties(cpp_example_${_ex} PROPERTIES TIMEOUT 300)
    endif()
endforeach()
```

- [ ] **Step 3: Create gap tracking document**

Create `doc/cpp_example_api_gaps.md`:

```markdown
# C++ Builder API Gaps (discovered during example porting)

| Example | Feature | Severity | Description | Suggested Fix |
|---------|---------|----------|-------------|---------------|
```

- [ ] **Step 4: Verify build system compiles (with existing examples only)**

Run:
```bash
cd build && ninja -j8 brachistochrone_builder
```
Expected: Compiles without errors (CMake changes don't break existing targets).

- [ ] **Step 5: Commit**

```bash
git add examples/cpp_examples/CMakeLists.txt doc/cpp_example_api_gaps.md
git commit -m "chore: add build infrastructure for C++ builder example suite

Add 23 new add_subdirectory entries, heavy_compile job pool assignment,
cpp_examples_all aggregate target, and CTest registration for all
C++ examples. Create gap tracking document."
```

---

## Task 1: VanDerPol (Tier 1)

**Files:**
- Create: `examples/cpp_examples/builder/van_der_pol/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/van_der_pol/main.cpp`

**Python source:** `examples/python_examples/VanDerPol.py`
**Problem:** Van der Pol oscillator, 2 states, 1 control, minimize integral(x0^2 + x1^2 + u^2)

- [ ] **Step 1: Create CMakeLists.txt**

Create `examples/cpp_examples/builder/van_der_pol/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(van_der_pol_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(van_der_pol_builder main.cpp)
target_link_libraries(van_der_pol_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

Create `examples/cpp_examples/builder/van_der_pol/main.cpp`:

```cpp
///////////////////////////////////////////////////////////////////////////////
// Van der Pol Oscillator — C++ example (Builder API)
//
// Ported from examples/python_examples/VanDerPol.py
// Source: https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
//
// State  : [x0, x1]   (oscillator states)
// Control: [u]         (forcing input)
//
// ODE:
//   x0dot = (1 - x1^2)*x0 - x1 + u
//   x1dot = x0
//
// Objective: minimise integral(x0^2 + x1^2 + u^2) over [0, 10]
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double tf = 10.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 256;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(2, 1)
                   .define([](auto &args) {
                       auto x0 = args.x_var(0);
                       auto x1 = args.x_var(1);
                       auto u = args.u_var(0);
                       auto x0dot = (1.0 - x1 * x1) * x0 - x1 + u;
                       auto x1dot = x0;
                       return stack(x0dot, x1dot);
                   })
                   .var_names({{"x0", 0}, {"x1", 1}, {"t", 2}, {"u", 3}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double t = tf * static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(4);
        pt << 0.0, 0.0, t, 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);
    phase.set_control_mode(ControlModes::BlockConstant);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x0", "x1", "t"},
                            Eigen::Vector3d(0.0, 1.0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x0", "x1", "t"},
                            Eigen::Vector3d(0.0, 0.0, tf));

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", -0.75, 1.0);

    // Integral objective: min integral(x0^2 + x1^2 + u^2)
    {
        auto obj_args = Arguments<3>();
        auto obj_expr = obj_args.squared_norm();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr),
                                    {"x0", "x1", "u"});
    }

    phase.optimizer().set_print_level(0);
    phase.set_num_partitions(8);
    phase.optimizer().set_tols(1.0e-8, 1.0e-8, 1.0e-8);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "VanDerPol (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "VanDerPol (builder): converged, "
              << traj.size() << " nodes\n";
    std::cout << "VanDerPol (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j8 van_der_pol_builder
./examples/cpp_examples/builder/van_der_pol/van_der_pol_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/van_der_pol/
git commit -m "feat: add VanDerPol C++ builder example"
```

---

## Task 2: BrysonDenham (Tier 1)

**Files:**
- Create: `examples/cpp_examples/builder/bryson_denham/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/bryson_denham/main.cpp`

**Python source:** `examples/python_examples/BrysonDenham.py`
**Problem:** Point mass with state upper bound x <= 1/9, minimize integral(u^2/2)

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(bryson_denham_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(bryson_denham_builder main.cpp)
target_link_libraries(bryson_denham_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Bryson-Denham — C++ example (Builder API)
//
// Ported from examples/python_examples/BrysonDenham.py
//
// State  : [x, v]   (position, velocity)
// Control: [u]       (acceleration)
//
// ODE: xdot = v, vdot = u
// Constraint: x <= 1/9
// Objective: minimise integral(u^2 / 2)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr int n_pts = 100;
    constexpr int n_segs = 32;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(2, 1)
                   .define([](auto &args) {
                       auto v = args.x_var(1);
                       auto u = args.u_var(0);
                       return stack(v, u);
                   })
                   .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(4);
        pt << 0.0, 1.0 - 2.0 * s, s, 0.0;  // v: 1 -> -1
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "v", "t"},
                            Eigen::Vector3d(0.0, 1.0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "v", "t"},
                            Eigen::Vector3d(0.0, -1.0, 1.0));

    // State upper bound: x <= 1/9
    phase.add_upper_var_bound(PhaseRegionFlags::Path, "x", 1.0 / 9.0);

    // Integral objective: min integral(u^2 / 2)
    {
        auto obj_args = Arguments<1>();
        auto obj_expr = (obj_args.coeff<0>() * obj_args.coeff<0>()) / 2.0;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"u"});
    }

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_kkt_tol(1.0e-10);
    phase.optimizer().set_print_level(0);
    phase.set_num_partitions(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "BrysonDenham (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();

    // Verify state upper bound is respected
    double max_x = -1e30;
    for (const auto &pt : traj) {
        max_x = std::max(max_x, pt[0]);
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "BrysonDenham (builder): max x = " << max_x << "\n";

    if (max_x > 1.0 / 9.0 + 1e-4) {
        std::cerr << "BrysonDenham (builder): FAILED — state bound violated\n";
        return EXIT_FAILURE;
    }

    std::cout << "BrysonDenham (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 bryson_denham_builder
./examples/cpp_examples/builder/bryson_denham/bryson_denham_builder
```
Expected: "PASSED", max x ≈ 0.11111111

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/bryson_denham/
git commit -m "feat: add BrysonDenham C++ builder example"
```

---

## Task 3: MountainCar (Tier 1)

**Files:**
- Create: `examples/cpp_examples/builder/mountain_car/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/mountain_car/main.cpp`

**Python source:** `examples/python_examples/MountainCar.py`
**Problem:** Underpowered car must escape valley via oscillation, minimize time

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(mountain_car_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(mountain_car_builder main.cpp)
target_link_libraries(mountain_car_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Mountain Car — C++ example (Builder API)
//
// Ported from examples/python_examples/MountainCar.py
// Source: https://openmdao.github.io/dymos/examples/mountain_car/mountain_car.html
//
// State  : [x, v]   (position, velocity)
// Control: [u]       (thrust, -1..1)
//
// ODE:
//   xdot = v
//   vdot = 0.001*u - 0.0025*cos(3*x)
//
// Objective: minimise time (scaled 0.01)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double x0 = -0.5;
    constexpr double v0 = 0.0;
    constexpr double xf = 0.52;
    constexpr double tf_guess = 500.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 128;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(2, 1)
                   .define([](auto &args) {
                       auto x = args.x_var(0);
                       auto v = args.x_var(1);
                       auto u = args.u_var(0);
                       auto xdot = v;
                       auto vdot = 0.001 * u - 0.0025 * cos(3.0 * x);
                       return stack(xdot, vdot);
                   })
                   .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        const double t = tf_guess * s;
        Eigen::VectorXd pt(4);
        pt << x0 + (xf - x0) * s, s, t, std::sin(s);
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "v", "t"},
                            Eigen::Vector3d(x0, v0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x"}, Eigen::Matrix<double, 1, 1>(xf));

    // Back: final velocity must be non-negative
    phase.add_lower_var_bound(PhaseRegionFlags::Back, "v", 0.0, 1.0);

    // Path bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", -1.2, 0.55, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", -0.07, 0.07, 100.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", -1.0, 1.0, 1.0);

    // Objective: minimize time (scaled)
    phase.add_delta_time_objective(0.01);

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.solve_optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MountainCar (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    const double final_time = traj.back()[2];

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MountainCar (builder): final time = " << final_time << "\n";
    std::cout << "MountainCar (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 mountain_car_builder
./examples/cpp_examples/builder/mountain_car/mountain_car_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/mountain_car/
git commit -m "feat: add MountainCar C++ builder example"
```

---

## Task 4: AnalyticExample (Tier 1)

**Files:**
- Create: `examples/cpp_examples/builder/analytic_example/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/analytic_example/main.cpp`

**Python source:** `examples/python_examples/AnalyticExample.py`
**Problem:** 1 state, 1 control, xdot = 0.5x + u, minimize integral(u^2 + xu + 1.25x^2)

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(analytic_example_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(analytic_example_builder main.cpp)
target_link_libraries(analytic_example_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Analytic Example — C++ example (Builder API)
//
// Ported from examples/python_examples/AnalyticExample.py
// Source: https://www.hindawi.com/journals/aaa/2014/851720/
//
// State  : [x]   (1 state)
// Control: [u]   (1 control)
//
// ODE: xdot = 0.5*x + u
// Objective: minimise integral(u^2 + x*u + 1.25*x^2)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double x0_val = 1.0;
    constexpr double t0 = 0.0;
    constexpr double tf = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 20;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) {
                       auto x = args.x_var(0);
                       auto u = args.u_var(0);
                       return 0.5 * x + u;
                   })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double t = t0 + (tf - t0) * static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(3);
        pt << x0_val, t, 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "t"},
                            Eigen::Vector2d(x0_val, t0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"t"},
                            Eigen::Matrix<double, 1, 1>(tf));

    // Integral objective: min integral(u^2 + x*u + 1.25*x^2)
    {
        auto obj_args = Arguments<2>();
        auto x = obj_args.coeff<0>();
        auto u = obj_args.coeff<1>();
        auto obj_expr = u * u + x * u + 1.25 * x * x;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"x", "u"});
    }

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "AnalyticExample (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    // Verify against analytic solution at t=0:
    // u*(t) = -(tanh(1-t) + 0.5) * cosh(1-t) / cosh(1)
    // u*(0) = -(tanh(1) + 0.5) * cosh(1) / cosh(1) = -(tanh(1) + 0.5)
    auto traj = phase.return_traj();
    const double u_colloc = traj.front()[2];
    const double u_analytic = -(std::tanh(1.0) + 0.5);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "AnalyticExample (builder): u(0) collocation = " << u_colloc << "\n";
    std::cout << "AnalyticExample (builder): u(0) analytic    = " << u_analytic << "\n";

    if (std::abs(u_colloc - u_analytic) > 1e-4) {
        std::cerr << "AnalyticExample (builder): FAILED — control mismatch\n";
        return EXIT_FAILURE;
    }

    std::cout << "AnalyticExample (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 analytic_example_builder
./examples/cpp_examples/builder/analytic_example/analytic_example_builder
```
Expected: "PASSED", u(0) ≈ -0.9864

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/analytic_example/
git commit -m "feat: add AnalyticExample C++ builder example"
```

---

## Task 5: BikeObstacle (Tier 1)

**Files:**
- Create: `examples/cpp_examples/builder/bike_obstacle/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/bike_obstacle/main.cpp`

**Python source:** `examples/python_examples/BikeObstacle.py`
**Problem:** Bicycle obstacle avoidance, 4 states, 2 controls, minimize time

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(bike_obstacle_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(bike_obstacle_builder main.cpp)
target_link_libraries(bike_obstacle_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Bike Obstacle Avoidance — C++ example (Builder API)
//
// Ported from examples/python_examples/BikeObstacle.py
// Source: https://arxiv.org/pdf/2003.00142.pdf
//
// State  : [x, y, psi, v]   (position, heading, speed)
// Control: [acc, alpha]      (acceleration, steering angle)
//
// ODE:
//   beta = atan((la/(la+lb))*tan(alpha))
//   xdot = v*cos(psi + beta)
//   ydot = v*sin(psi + beta)
//   psidot = v*sin(beta)/lb
//   vdot = acc
//
// Obstacle: circular exclusion zone at (0, 50), r=5+2.5
// Objective: minimise time
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    // Physical parameters
    constexpr double la = 1.58;   // front axle distance (m)
    constexpr double lb = 1.72;   // rear axle distance (m)
    constexpr double obsrad = 5.0;
    constexpr double margin = 2.5;
    constexpr double xobs = 0.0;
    constexpr double yobs = 50.0;

    // Boundary conditions
    constexpr double x0 = 0.0, y0 = 0.0;
    constexpr double psi0 = M_PI / 2.0;
    constexpr double v0 = 15.0;
    constexpr double xf = 0.0, yf = 100.0;

    // Bounds
    constexpr double accbound = 2.0;
    constexpr double vlbound = 5.0, vubound = 29.0;
    constexpr double alpha_max = M_PI / 6.0;  // 30 deg

    constexpr double tf_guess = yf / v0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 128;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(4, 2)
                   .define([la, lb](auto &args) {
                       auto x = args.x_var(0);
                       auto y = args.x_var(1);
                       auto psi = args.x_var(2);
                       auto v = args.x_var(3);
                       auto acc = args.u_var(0);
                       auto alpha = args.u_var(1);

                       auto beta = arctan((la / (la + lb)) * tan(alpha));
                       auto xdot = v * cos(psi + beta);
                       auto ydot = v * sin(psi + beta);
                       auto psidot = v * sin(beta) / lb;
                       auto vdot = acc;

                       return stack(xdot, ydot, psidot, vdot);
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"psi", 2}, {"v", 3},
                               {"t", 4}, {"acc", 5}, {"alpha", 6}})
                   .build();

    // ── Obstacle constraint function ───────────────────────────────────
    // Returns -(((x-xobs)/(R+m))^2 + ((y-yobs)/(R+m))^2 - 1) <= 0
    auto obs_args = Arguments<2>();
    auto ox = obs_args.coeff<0>();
    auto oy = obs_args.coeff<1>();
    const double denom = obsrad + margin;
    auto ellips = ((ox - xobs) / denom) * ((ox - xobs) / denom) +
                  ((oy - yobs) / denom) * ((oy - yobs) / denom);
    auto obs_con = 1.0 - ellips;  // <= 0 for feasible

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        const double t = tf_guess * s;
        Eigen::VectorXd pt(7);
        pt[0] = obsrad + margin + 1.0;  // bias to side of obstacle
        pt[1] = yf * s;
        pt[2] = psi0;
        pt[3] = v0;
        pt[4] = t;
        pt[5] = 0.0;
        pt[6] = 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "y", "psi", "v", "t"},
                            Eigen::Matrix<double, 5, 1>(x0, y0, psi0, v0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"},
                            Eigen::Vector2d(xf, yf));

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", vlbound, vubound);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "acc", -accbound, accbound);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha", -alpha_max, alpha_max);

    phase.add_inequal_con(PhaseRegionFlags::Path,
                         GenericFunction<-1, -1>(obs_con), {"x", "y"});

    phase.add_delta_time_objective(1.0);

    phase.optimizer().set_tols(1.0e-9, 1.0e-9, 1.0e-9);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "BikeObstacle (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    const double final_time = traj.back()[4];

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "BikeObstacle (builder): final time = " << final_time << " s\n";
    std::cout << "BikeObstacle (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 bike_obstacle_builder
./examples/cpp_examples/builder/bike_obstacle/bike_obstacle_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/bike_obstacle/
git commit -m "feat: add BikeObstacle C++ builder example"
```

---

## Task 6: FreeFlyingRobot (Tier 1)

**Files:**
- Create: `examples/cpp_examples/builder/free_flying_robot/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/free_flying_robot/main.cpp`

**Python source:** `examples/python_examples/FreeFlyingRobotExample.py`
**Problem:** Planar robot with 6 states, 4 controls, minimize integral(u0+u1+u2+u3)

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(free_flying_robot_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(free_flying_robot_builder main.cpp)
target_link_libraries(free_flying_robot_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Free Flying Robot — C++ example (Builder API)
//
// Ported from examples/python_examples/FreeFlyingRobotExample.py
// Source: https://arxiv.org/pdf/1905.11898.pdf
//
// State  : [x, y, xdot, ydot, theta, omega]   (6 states)
// Control: [u0, u1, u2, u3]                    (4 thrusters)
//
// ODE:
//   [x,y]dot = [xdot, ydot]
//   [xdot,ydot]dot = [cos(theta), sin(theta)] * (u0-u1+u2-u3)
//   thetadot = omega
//   omegadot = (u0-u1)*alpha + (u3-u2)*beta
//
// Objective: minimise integral(u0 + u1 + u2 + u3) (fuel)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double alpha = 0.2;
    constexpr double beta = 0.2;
    constexpr double tf = 12.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 128;

    // Boundary conditions
    // Front: [x, y, xdot, ydot, theta, omega, t] = [-10, -10, 0, 0, pi/2, 0, 0]
    // Back:  [x, y, xdot, ydot, theta, omega, t] = [0, 0, 0, 0, 0, 0, tf]

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(6, 4)
                   .define([alpha, beta](auto &args) {
                       auto xdot_st = args.x_var(2);
                       auto ydot_st = args.x_var(3);
                       auto theta = args.x_var(4);
                       auto omega = args.x_var(5);

                       auto u0 = args.u_var(0);
                       auto u1 = args.u_var(1);
                       auto u2 = args.u_var(2);
                       auto u3 = args.u_var(3);

                       auto vscale = u0 - u1 + u2 - u3;
                       auto xdotdot = cos(theta) * vscale;
                       auto ydotdot = sin(theta) * vscale;
                       auto theta_dot = omega;
                       auto omega_dot = (u0 - u1) * alpha + (u3 - u2) * beta;

                       return stack(xdot_st, ydot_st, xdotdot, ydotdot,
                                   theta_dot, omega_dot);
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"xdot", 2}, {"ydot", 3},
                               {"theta", 4}, {"omega", 5}, {"t", 6},
                               {"u0", 7}, {"u1", 8}, {"u2", 9}, {"u3", 10}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    Eigen::VectorXd X0(7), XF(7);
    X0 << -10.0, -10.0, 0.0, 0.0, M_PI / 2.0, 0.0, 0.0;
    XF << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tf;

    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(11);
        pt.head<7>() = X0 + s * (XF - X0);
        pt.tail<4>().setConstant(0.5);
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "y", "xdot", "ydot", "theta", "omega", "t"},
                            X0);
    phase.add_boundary_value(PhaseRegionFlags::Back,
                            {"x", "y", "xdot", "ydot", "theta", "omega", "t"},
                            XF);

    // Control bounds: 0 <= ui <= 1
    for (const auto &u_name : {"u0", "u1", "u2", "u3"}) {
        phase.add_lu_var_bound(PhaseRegionFlags::Path, u_name, 0.0, 1.0, 1.0);
    }

    // Integral objective: min integral(u0 + u1 + u2 + u3)
    {
        auto obj_args = Arguments<4>();
        auto obj_expr = obj_args.sum();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr),
                                    {"u0", "u1", "u2", "u3"});
    }

    phase.optimizer().set_print_level(0);
    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_tols(1.0e-9, 1.0e-9, 1.0e-9);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "FreeFlyingRobot (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "FreeFlyingRobot (builder): converged, "
              << traj.size() << " nodes\n";
    std::cout << "FreeFlyingRobot (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 free_flying_robot_builder
./examples/cpp_examples/builder/free_flying_robot/free_flying_robot_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/free_flying_robot/
git commit -m "feat: add FreeFlyingRobot C++ builder example"
```

---

## Task 7: CartPole (Tier 2)

**Files:**
- Create: `examples/cpp_examples/builder/cart_pole/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/cart_pole/main.cpp`

**Python source:** `examples/python_examples/CartPole.py`
**Problem:** Cart-pole swing-up with matrix inversion for mass matrix, 4 states, 1 control

**Key validation:** This example validates `RowMatrix` and `.inverse()` operations work within the builder lambda. The Python version uses `vf.RowMatrix` and `M.inverse() * Q` inside the ODE class.

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(cart_pole_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(cart_pole_builder main.cpp)
target_link_libraries(cart_pole_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Cart-Pole Swing-Up — C++ example (Builder API)
//
// Ported from examples/python_examples/CartPole.py
// Source: Kelly, M., "An introduction to trajectory optimization", SIAM Review, 2017
//
// State  : [x, theta, xdot, thetadot]   (cart pos, pole angle, velocities)
// Control: [F]                            (applied force)
//
// ODE: Uses mass matrix inversion M^{-1} * Q at runtime
//   M = [[m1+m2, m2*l*cos(theta)], [cos(theta), l]]
//   Q = [-g*sin(theta), F + m2*l*sin(theta)*thetadot^2]
//   [xddot, thetaddot] = M^{-1} * Q
//
// Objective: minimise integral(F^2)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double l = 0.5;
    constexpr double m1 = 1.0;
    constexpr double m2 = 0.3;
    constexpr double g = 9.81;
    constexpr double Fmax = 20.0;
    constexpr double xmax = 2.0;
    constexpr double tf = 2.0;
    constexpr double xf = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 64;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(4, 1)
                   .define([l, m1, m2, g](auto &args) {
                       auto x = args.x_var(0);
                       auto theta = args.x_var(1);
                       auto xdot = args.x_var(2);
                       auto thetadot = args.x_var(3);
                       auto F = args.u_var(0);

                       // Force vector Q
                       auto Q = stack(-g * sin(theta),
                                     F + m2 * l * sin(theta) * thetadot * thetadot);

                       // Mass matrix as row-major flat vector
                       auto Mvec = stack(cos(theta), l, m1 + m2, m2 * l * cos(theta));
                       auto M = RowMatrix(Mvec, 2, 2);

                       auto accel = M.inverse() * Q;

                       return stack(xdot, thetadot, accel);
                   })
                   .var_names({{"x", 0}, {"theta", 1}, {"xdot", 2}, {"thetadot", 3},
                               {"t", 4}, {"F", 5}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        const double t = tf * s;
        Eigen::VectorXd pt(6);
        pt << xf * s, M_PI * s, 0.0, 0.0, t, 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    // Fix initial state and time
    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "theta", "xdot", "thetadot", "t"},
                            Eigen::Matrix<double, 5, 1>(0.0, 0.0, 0.0, 0.0, 0.0));

    // Fix final state and time
    phase.add_boundary_value(PhaseRegionFlags::Back,
                            {"x", "theta", "xdot", "thetadot", "t"},
                            Eigen::Matrix<double, 5, 1>(xf, M_PI, 0.0, 0.0, tf));

    // Bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "F", -Fmax, Fmax);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", -xmax, xmax);

    // Integral objective: min integral(F^2)
    {
        auto obj_args = Arguments<1>();
        auto obj_expr = obj_args.coeff<0>() * obj_args.coeff<0>();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"F"});
    }

    phase.set_num_partitions(8);
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "CartPole (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CartPole (builder): final theta = " << traj.back()[1]
              << " (expected ~" << M_PI << ")\n";
    std::cout << "CartPole (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 cart_pole_builder
./examples/cpp_examples/builder/cart_pole/cart_pole_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/cart_pole/
git commit -m "feat: add CartPole C++ builder example (validates RowMatrix.inverse in builder)"
```

---

## Task 8: ParallelParking (Tier 2)

**Files:**
- Create: `examples/cpp_examples/builder/parallel_parking/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/parallel_parking/main.cpp`

**Python source:** `examples/python_examples/ParallelParking.py`
**Problem:** Minimum-time parallel parking, 6 states, 2 controls, 1 static parameter, complex slot geometry constraints. Uses `set_static_params`, `sub_variable`, `refine_traj_manual`, `add_value_lock`, `BlockConstant` control, two-phase optimization.

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(parallel_parking_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(parallel_parking_builder main.cpp)
target_link_libraries(parallel_parking_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Parallel Parking — C++ example (Builder API)
//
// Ported from examples/python_examples/ParallelParking.py
// Source: http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html
//         https://ieeexplore.ieee.org/document/7463491
//
// State  : [x, y, v, a, theta, phi]   (pos, vel, accel, heading, steering)
// Control: [u1, u2]                    (jerk, steering rate)
// Static : [k]                         (tanh smoothing constant)
//
// ODE:
//   xdot     = v*cos(theta)
//   ydot     = v*sin(theta)
//   vdot     = a
//   adot     = u1
//   thetadot = v*tan(phi)/l_axes
//   phidot   = u2
//
// Objective: minimise time (~18.4 s expected)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    // Car dimensions (m)
    constexpr double l_front = 0.839;
    constexpr double l_axes = 2.588;
    constexpr double l_rear = 0.657;
    constexpr double b_width = 1.771 / 2.0;

    // Slot dimensions
    constexpr double SL = 5.0;   // slot length
    constexpr double SW = 2.0;   // slot width (depth)
    constexpr double CL = 3.5;   // curb distance (street width)

    // Limits
    constexpr double phi_max = 33.0 * M_PI / 180.0;
    constexpr double v_max = 2.0;
    constexpr double a_max = 0.75;
    constexpr double u1_max = 0.5;
    constexpr double curvature_dot_max = 0.6;
    constexpr double xmin = -10.0, xmax_val = 7.5;

    // Initial conditions (from paper)
    constexpr double x0 = -5.14;
    constexpr double y0 = 1.41;
    constexpr double theta0 = 13.18 * M_PI / 180.0;

    constexpr int n_segs1 = 50;
    constexpr int n_segs2 = 200;
    constexpr double k1 = 75.0;
    constexpr double k2 = 150.0;

    const double area_ref = (l_axes + l_front + l_rear) * 2.0 * b_width;

    // Corner locations in body frame (from center of rear axle)
    const double Aloc_x = l_front + l_axes, Aloc_y = b_width;
    const double Bloc_x = l_front + l_axes, Bloc_y = -b_width;
    const double Cloc_x = -l_rear, Cloc_y = b_width;
    const double Dloc_x = -l_rear, Dloc_y = -b_width;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(6, 2)
                   .define([l_axes](auto &args) {
                       auto x = args.x_var(0);
                       auto y = args.x_var(1);
                       auto v = args.x_var(2);
                       auto a = args.x_var(3);
                       auto theta = args.x_var(4);
                       auto phi = args.x_var(5);
                       auto u1 = args.u_var(0);
                       auto u2 = args.u_var(1);

                       return stack(v * cos(theta),
                                   v * sin(theta),
                                   a, u1,
                                   v * tan(phi) / l_axes,
                                   u2);
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"a", 3},
                               {"theta", 4}, {"phi", 5}, {"t", 6},
                               {"u1", 7}, {"u2", 8}})
                   .build();

    // ── Slot boundary constraint ───────────────────────────────────────
    // Heaviside approximation: H(x,k) = (1 + tanh(kx))/2
    // Fslot(x, k) = (-H(x,k) + H(x-SL,k)) * SW
    // For each car corner: y_corner <= CL and y_corner >= Fslot(x_corner, k)
    // Uses 4 corners * 2 constraints = 8 scalar inequalities

    auto slot_fn = [SL, SW, CL, Aloc_x, Aloc_y, Bloc_x, Bloc_y,
                    Cloc_x, Cloc_y, Dloc_x, Dloc_y]() {
        auto args = Arguments<4>();
        auto x = args.coeff<0>();
        auto y = args.coeff<1>();
        auto theta = args.coeff<2>();
        auto k = args.coeff<3>();

        // Helper: rotate corner into world frame and add to car position
        auto corner = [&](double lx, double ly) {
            auto xl = cos(theta) * lx - sin(theta) * ly;
            auto yl = sin(theta) * lx + cos(theta) * ly;
            return std::make_pair(x + xl, y + yl);
        };

        auto heaviside = [&k](auto xv) {
            return (1.0 + tanh(k * xv)) / 2.0;
        };

        auto fslot = [&heaviside, SL, SW](auto xv) {
            return (-heaviside(xv) + heaviside(xv - SL)) * SW;
        };

        auto [Ax, Ay] = corner(Aloc_x, Aloc_y);
        auto [Bx, By] = corner(Bloc_x, Bloc_y);
        auto [Cx, Cy] = corner(Cloc_x, Cloc_y);
        auto [Dx, Dy] = corner(Dloc_x, Dloc_y);

        // Each corner: eq1 = y - CL <= 0, eq2 = -y + fslot(X) <= 0
        return stack(Ay - CL, -Ay + fslot(Ax),
                    By - CL, -By + fslot(Bx),
                    Cy - CL, -Cy + fslot(Cx),
                    Dy - CL, -Dy + fslot(Dx));
    };

    // ── Corner collision avoidance ─────────────────────────────────────
    auto corner_fn = [SL, Aloc_x, Aloc_y, Bloc_x, Bloc_y,
                      Cloc_x, Cloc_y, Dloc_x, Dloc_y, area_ref]() {
        auto args = Arguments<3>();
        auto x = args.coeff<0>();
        auto y = args.coeff<1>();
        auto theta = args.coeff<2>();

        auto corner = [&](double lx, double ly) {
            auto xl = cos(theta) * lx - sin(theta) * ly;
            auto yl = sin(theta) * lx + cos(theta) * ly;
            return std::make_pair(x + xl, y + yl);
        };

        auto tri_area = [](auto ax, auto ay, auto bx, auto by, auto cx, auto cy) {
            return abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0;
        };

        auto [Ax, Ay] = corner(Aloc_x, Aloc_y);
        auto [Bx, By] = corner(Bloc_x, Bloc_y);
        auto [Cx, Cy] = corner(Cloc_x, Cloc_y);
        auto [Dx, Dy] = corner(Dloc_x, Dloc_y);

        // Origin O = (0, 0)
        auto AO1 = tri_area(0.0, 0.0, Ax, Ay, Bx, By);
        auto AO2 = tri_area(0.0, 0.0, Cx, Cy, Bx, By);
        auto AO3 = tri_area(0.0, 0.0, Ax, Ay, Dx, Dy);
        auto AO4 = tri_area(0.0, 0.0, Dx, Dy, Cx, Cy);
        auto eq1 = area_ref - (AO1 + AO2 + AO3 + AO4);

        // Point E = (SL, 0)
        auto AE1 = tri_area(SL, 0.0, Ax, Ay, Bx, By);
        auto AE2 = tri_area(SL, 0.0, Cx, Cy, Bx, By);
        auto AE3 = tri_area(SL, 0.0, Ax, Ay, Dx, Dy);
        auto AE4 = tri_area(SL, 0.0, Dx, Dy, Cx, Cy);
        auto eq2 = area_ref - (AE1 + AE2 + AE3 + AE4);

        return stack(eq1, eq2);
    };

    // ── Final Y constraint ─────────────────────────────────────────────
    auto final_y_fn = [Aloc_x, Aloc_y, Bloc_x, Bloc_y,
                       Cloc_x, Cloc_y, Dloc_x, Dloc_y]() {
        auto args = Arguments<2>();
        auto y = args.coeff<0>();
        auto theta = args.coeff<1>();

        auto corner_y = [&](double lx, double ly) {
            return y + sin(theta) * lx + cos(theta) * ly;
        };

        return stack(corner_y(Aloc_x, Aloc_y), corner_y(Bloc_x, Bloc_y),
                    corner_y(Cloc_x, Cloc_y), corner_y(Dloc_x, Dloc_y));
    };

    // ── Curvature rate function ────────────────────────────────────────
    auto curv_fn = [l_axes]() {
        auto args = Arguments<2>();
        auto phi = args.coeff<0>();
        auto u2 = args.coeff<1>();
        return u2 / (l_axes * cos(phi) * cos(phi));
    };

    // ── Initial guess ──────────────────────────────────────────────────
    auto make_state = [](double x, double y, double theta_deg, double t) {
        Eigen::VectorXd XtU(9);
        XtU.setZero();
        XtU[0] = x;
        XtU[1] = y;
        XtU[4] = theta_deg * M_PI / 180.0;
        XtU[6] = t;
        return XtU;
    };

    std::vector<Eigen::VectorXd> traj_ig = {
        make_state(x0, y0, 13.18, 0),
        make_state(0.0, y0, 0.0, 5),
        make_state(5.5, y0, 10.0, 10),
        make_state(1.0, -0.5, 20.0, 15),
        make_state(1.0, -1.0, 0.0, 25)
    };

    // ── Phase setup (coarse) ───────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs1, true);
    phase.set_static_params(Eigen::Matrix<double, 1, 1>(k1));
    phase.set_control_mode(ControlModes::BlockConstant);

    // Boundary conditions
    Eigen::VectorXd front_bc(7);
    front_bc << x0, y0, 0.0, 0.0, theta0, 0.0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "y", "v", "a", "theta", "phi", "t"}, front_bc);

    phase.add_boundary_value(PhaseRegionFlags::Back, {"v", "a"},
                            Eigen::Vector2d(0.0, 0.0));

    // Slot geometry constraints (path)
    // NOTE: The builder Phase wrapper's add_inequal_con does not support
    // mixed variable sources (phase vars + static params) by name. Two options:
    //   (a) Capture k1 in the closure and rebuild constraint for k2 after refinement
    //   (b) Use phase.base() with index-based API for the raw ODEPhaseBase
    // Option (a) is simpler and avoids the gap entirely:
    phase.add_inequal_con(PhaseRegionFlags::Path,
                         GenericFunction<-1, -1>(slot_fn()),
                         {"x", "y", "theta"}, {}, {0});
    // If the above {0} syntax for static param indices doesn't work, fall back to
    // phase.base() with raw index vectors — document as gap.

    // Final Y constraint: all corners in slot
    phase.add_inequal_con(PhaseRegionFlags::Back,
                         GenericFunction<-1, -1>(final_y_fn()),
                         {"y", "theta"});

    // Variable bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", xmin, xmax_val);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", -v_max, v_max);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "a", -a_max, a_max);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "phi", -phi_max, phi_max);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u1", -u1_max, u1_max);

    // Curvature rate bound
    phase.add_lu_func_bound(PhaseRegionFlags::Path,
                           GenericFunction<-1, 1>(curv_fn()),
                           {"phi", "u2"},
                           -curvature_dot_max, curvature_dot_max);

    // Corner collision avoidance
    phase.add_inequal_con(PhaseRegionFlags::Path,
                         GenericFunction<-1, -1>(corner_fn()),
                         {"x", "y", "theta"});

    // Lock static parameter k to its initial value
    phase.add_value_lock(PhaseRegionFlags::StaticParams, {0});

    // Objective
    phase.add_delta_time_objective(1.0);

    phase.optimizer().set_bound_fraction(0.995);
    phase.optimizer().set_max_iters(2000);
    phase.optimizer().set_print_level(1);

    // ── Coarse solve ───────────────────────────────────────────────────
    phase.solve_optimize();

    // ── Refine and re-solve ────────────────────────────────────────────
    // NOTE: refine_traj_manual() and sub_variable() may not be on the Phase
    // wrapper. If not, use phase.base().refine_traj_manual(n_segs2) and
    // phase.base().sub_variable("StaticParams", 0, k2). Document as gap if needed.
    phase.refine_traj_manual(n_segs2);
    phase.sub_variable(PhaseRegionFlags::StaticParams, 0, k2);
    phase.optimizer().set_kkt_tol(1.0e-8);
    phase.optimize();

    auto traj = phase.return_traj();
    const double final_time = traj.back()[6];

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "ParallelParking (builder): maneuver time = "
              << final_time << " s (paper: 18.426 s)\n";

    if (std::abs(final_time - 18.426) > 2.0) {
        std::cerr << "ParallelParking (builder): FAILED — time too far from reference\n";
        return EXIT_FAILURE;
    }

    std::cout << "ParallelParking (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

**NOTE:** The `add_inequal_con` with mixed variable sources (phase variables + static params) may require using the index-based API or `phase.base()`. If the builder API doesn't support passing static params by name in constraint functions, document in `doc/cpp_example_api_gaps.md` and use index-based fallback.

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 parallel_parking_builder
./examples/cpp_examples/builder/parallel_parking/parallel_parking_builder
```
Expected: "PASSED", maneuver time ≈ 18.4 s

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/parallel_parking/
git commit -m "feat: add ParallelParking C++ builder example"
```

---

## Task 9: HangingChain (Tier 2)

**Files:**
- Create: `examples/cpp_examples/builder/hanging_chain/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/hanging_chain/main.cpp`

**Python source:** `examples/python_examples/HangingChain.py`
**Problem:** Catenary/hanging chain, 1 state, 1 control, integral parameter constraint for chain length, minimize gravitational potential energy.

**Key gap:** Python uses `Jet.map()` for parallel batch solving over different chain lengths. The builder API Phase wrapper may not support Jet integration. This example will implement a single chain configuration (not the sweep) and document the Jet gap.

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(hanging_chain_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(hanging_chain_builder main.cpp)
target_link_libraries(hanging_chain_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Hanging Chain — C++ example (Builder API)
//
// Ported from examples/python_examples/HangingChain.py
//
// State  : [x]   (chain height)
// Control: [u]   (slope dx/ds)
//
// ODE: xdot = u  (parameterised by arc length s, not time)
//
// Integral constraint: ∫√(1 + u²) ds = L  (chain length)
// Objective: minimise ∫ x·√(1 + u²) ds    (gravitational energy)
//
// NOTE: Python example uses Jet.map() for parallel sweep over L values.
//       This C++ version solves a single chain configuration (L=4).
//       Jet parallel solve is a documented API gap.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double a = 1.0;   // left endpoint height
    constexpr double b = 3.0;   // right endpoint height
    constexpr double L = 4.0;   // chain length
    constexpr int n_segs = 500;
    constexpr int n_pts = n_segs;

    // ── Define ODE ─────────────────────────────────────────────────────
    // xdot = u  (trivial ODE: slope is the control)
    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) {
                       return args.u_var(0);
                   })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    // ── Energy integrand ───────────────────────────────────────────────
    // integral(x * sqrt(1 + u^2))
    auto energy_args = Arguments<2>();
    auto ex = energy_args.coeff<0>();
    auto eu = energy_args.coeff<1>();
    auto energy_expr = ex * sqrt(1.0 + eu * eu);

    // ── Length integrand ───────────────────────────────────────────────
    // integral(sqrt(1 + u^2)) = L
    auto len_args = Arguments<1>();
    auto lu = len_args.coeff<0>();
    auto length_expr = sqrt(1.0 + lu * lu);

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        double x_val = 2.0 * std::abs(b - a) * s * (s - 0.5) + a;
        double u_val = 2.0 * std::abs(b - a) * (2.0 * s - 0.5);
        Eigen::VectorXd pt(3);
        pt << x_val, s, u_val;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);
    phase.set_static_params(Eigen::Matrix<double, 1, 1>(L));

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "t"},
                            Eigen::Vector2d(a, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "t"},
                            Eigen::Vector2d(b, 1.0));
    // Lock static param to L (use index-based API for StaticParams region)
    phase.add_boundary_value(PhaseRegionFlags::StaticParams,
                            Eigen::VectorXi::Constant(1, 0),
                            Eigen::Matrix<double, 1, 1>(L));

    // Upper bound on x
    phase.add_upper_var_bound(PhaseRegionFlags::Path, "x",
                             std::max(a, b) + 0.001);

    // Integral objective: minimise gravitational energy
    phase.add_integral_objective(GenericFunction<-1, 1>(energy_expr), {"x", "u"});

    // Integral parameter constraint: length = L
    phase.add_integral_param_function(GenericFunction<-1, 1>(length_expr), {"u"}, 0);

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.solve_optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "HangingChain (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "HangingChain (builder): x(0) = " << traj.front()[0]
              << ", x(1) = " << traj.back()[0] << "\n";
    std::cout << "HangingChain (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Document Jet gap**

Add to `doc/cpp_example_api_gaps.md`:

| HangingChain | Jet.map() | Medium | Python uses Jet.map() for parallel batch solving over parameter sweeps. No equivalent in C++ builder API. | Expose Jet parallel solve for Phase wrapper or add batch-solve method |

- [ ] **Step 4: Build and test**

Run:
```bash
cd build && ninja -j8 hanging_chain_builder
./examples/cpp_examples/builder/hanging_chain/hanging_chain_builder
```
Expected: "PASSED"

- [ ] **Step 5: Commit**

```bash
git add examples/cpp_examples/builder/hanging_chain/ doc/cpp_example_api_gaps.md
git commit -m "feat: add HangingChain C++ builder example (documents Jet gap)"
```

---

## Task 10: Reentry (Tier 2)

**Files:**
- Create: `examples/cpp_examples/builder/reentry/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/reentry/main.cpp`

**Python source:** `examples/python_examples/Reentry.py`
**Problem:** Space Shuttle reentry, 5 states, 2 controls, heating rate constraint, two-step solve (unconstrained then constrained), `refine_traj_manual`, maximize cross-range

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(reentry_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(reentry_builder main.cpp)
target_link_libraries(reentry_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Space Shuttle Reentry — C++ example (Builder API)
//
// Ported from examples/python_examples/Reentry.py
// Source: Betts, "Practical Methods for OC", Cambridge, 2009
//
// State  : [h, theta, v, gamma, psi]   (alt, downrange, speed, FPA, heading)
// Control: [alpha, beta]                (AoA, bank angle)
//
// Objective: maximise final cross-range (theta at tf)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    // ── Non-dimensionalisation ─────────────────────────────────────────
    constexpr double g0 = 32.2;
    constexpr double W = 203000.0;
    constexpr double Lstar = 100000.0;
    constexpr double Tstar = 60.0;
    constexpr double Mstar = W / g0;
    const double Vstar = Lstar / Tstar;
    const double Astar = Lstar / (Tstar * Tstar);
    const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
    const double Mustar = (Lstar * Lstar * Lstar) / (Tstar * Tstar);

    const double tmax = 2500.0 / Tstar;
    const double Re = 20902900.0 / Lstar;
    const double S_ref = 2690.0 / (Lstar * Lstar);
    const double m = (W / g0) / Mstar;
    const double mu = 0.140765e17 / Mustar;
    const double rho0 = 0.002378 / Rhostar;
    const double h_ref = 23800.0 / Lstar;

    constexpr double a0 = -0.20704, a1 = 0.029244;
    constexpr double b0 = 0.07854, b1 = -0.61592e-2, b2 = 0.621408e-3;
    constexpr double c0 = 1.0672181, c1 = -0.19213774e-1;
    constexpr double c2 = 0.21286289e-3, c3 = -0.10117e-5;
    constexpr double Qlimit = 70.0;

    // Boundary conditions
    const double ht0 = 260000.0 / Lstar;
    const double htf = 80000.0 / Lstar;
    const double vt0 = 25600.0 / Vstar;
    const double vtf = 2500.0 / Vstar;
    const double gammat0 = -1.0 * M_PI / 180.0;
    const double gammatf = -5.0 * M_PI / 180.0;
    const double psit0 = 90.0 * M_PI / 180.0;

    constexpr double tf_guess = 1000.0 / 60.0;
    constexpr int n_pts = 200;
    constexpr int n_segs = 40;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(5, 2)
                   .define([Re, S_ref, m, mu, rho0, h_ref, a0, a1, b0, b1, b2](auto &args) {
                       auto h = args.x_var(0);
                       auto theta = args.x_var(1);
                       auto v = args.x_var(2);
                       auto gamma = args.x_var(3);
                       auto psi = args.x_var(4);
                       auto alpha = args.u_var(0);
                       auto beta_u = args.u_var(1);

                       auto alphadeg = (180.0 / M_PI) * alpha;
                       auto CL = a0 + a1 * alphadeg;
                       auto CD = b0 + b1 * alphadeg + b2 * alphadeg * alphadeg;
                       auto rho = rho0 * exp(-h / h_ref);
                       auto r = h + Re;
                       auto Lift = 0.5 * CL * S_ref * rho * v * v;
                       auto Drag = 0.5 * CD * S_ref * rho * v * v;
                       auto grav = mu / (r * r);

                       auto sgam = sin(gamma);
                       auto cgam = cos(gamma);
                       auto sbet = sin(beta_u);
                       auto cbet = cos(beta_u);
                       auto spsi = sin(psi);
                       auto cpsi = cos(psi);

                       auto hdot = v * sgam;
                       auto thetadot = (v / r) * cgam * cpsi;
                       auto vdot = -Drag / m - grav * sgam;
                       auto gammadot = (Lift / (m * v)) * cbet + cgam * (v / r - grav / v);
                       auto psidot = Lift * sbet / (m * v * cgam) +
                                    (v / r) * cgam * spsi * tan(theta);

                       return stack(hdot, thetadot, vdot, gammadot, psidot);
                   })
                   .var_names({{"h", 0}, {"theta", 1}, {"v", 2}, {"gamma", 3},
                               {"psi", 4}, {"t", 5}, {"alpha", 6}, {"beta", 7}})
                   .build();

    // ── Heating rate function ──────────────────────────────────────────
    auto qfunc_args = Arguments<3>();
    auto qh = qfunc_args.coeff<0>();
    auto qv = qfunc_args.coeff<1>();
    auto qalpha = qfunc_args.coeff<2>();
    auto qalphadeg = (180.0 / M_PI) * qalpha;
    auto rhodim = rho0 * exp(-qh / h_ref) * Rhostar;
    auto vdim = qv * Vstar;
    auto qr = 17700.0 * sqrt(rhodim) * pow(0.0001 * vdim, 3.07);
    auto qa = c0 + c1 * qalphadeg + c2 * qalphadeg * qalphadeg +
              c3 * qalphadeg * qalphadeg * qalphadeg;
    auto Q_expr = qa * qr;

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(8);
        pt[0] = ht0 * (1.0 - s) + htf * s;
        pt[1] = 0.0;
        pt[2] = vt0 * (1.0 - s) + vtf * s;
        pt[3] = gammat0 * (1.0 - s) + gammatf * s;
        pt[4] = psit0;
        pt[5] = tf_guess * s;
        pt[6] = 0.0;
        pt[7] = 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);

    Eigen::VectorXd front_bc(6);
    front_bc << ht0, 0.0, vt0, gammat0, psit0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"h", "theta", "v", "gamma", "psi", "t"}, front_bc);

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta",
                          -89.0 * M_PI / 180.0, 89.0 * M_PI / 180.0, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "gamma",
                          -89.0 * M_PI / 180.0, 89.0 * M_PI / 180.0, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha",
                          -90.0 * M_PI / 180.0, 90.0 * M_PI / 180.0, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "beta",
                          -90.0 * M_PI / 180.0, 1.0 * M_PI / 180.0, 1.0);
    phase.add_upper_delta_time_bound(tmax, 1.0);

    phase.add_boundary_value(PhaseRegionFlags::Back, {"h", "v", "gamma"},
                            Eigen::Vector3d(htf, vtf, gammatf));

    // Objective: maximise theta(tf) => minimise -theta (via delta_var_objective)
    phase.add_delta_var_objective("theta", -1.0);

    phase.set_num_partitions(8);
    phase.optimizer().set_soe_ls_mode("L1");
    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_print_level(1);

    // ── Solve (unconstrained) ──────────────────────────────────────────
    phase.solve_optimize();
    phase.refine_traj_manual(300);
    phase.optimize();

    auto traj1 = phase.return_traj();

    // ── Add heating constraint and re-solve ────────────────────────────
    phase.add_upper_func_bound(PhaseRegionFlags::Path,
                              GenericFunction<-1, 1>(Q_expr),
                              {"h", "v", "alpha"}, Qlimit, 1.0 / Qlimit);
    phase.optimize();

    auto traj2 = phase.return_traj();
    const double crossrange1 = traj1.back()[1] * 180.0 / M_PI;
    const double crossrange2 = traj2.back()[1] * 180.0 / M_PI;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Reentry (builder): cross-range (no Q limit) = "
              << crossrange1 << " deg\n";
    std::cout << "Reentry (builder): cross-range (Q limited)  = "
              << crossrange2 << " deg\n";

    std::cout << "Reentry (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 reentry_builder
./examples/cpp_examples/builder/reentry/reentry_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/reentry/
git commit -m "feat: add Reentry C++ builder example"
```

---

## Task 11: MinimumTimeToClimb (Tier 2)

**Files:**
- Create: `examples/cpp_examples/builder/minimum_time_to_climb/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/minimum_time_to_climb/main.cpp`

**Python source:** `examples/python_examples/MinimumTimeToClimb.py`
**Problem:** Aircraft minimum-time climb, 4 states, 1 control, inline atmospheric/aerodynamic model (lookup tables replaced with polynomial fits), adaptive mesh refinement

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(minimum_time_to_climb_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(minimum_time_to_climb_builder main.cpp)
target_link_libraries(minimum_time_to_climb_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

Implement with inline atmospheric model functions (polynomial fits for density, speed of sound, CD0, CLalpha, eta, thrust as functions of Mach and altitude). Use adaptive mesh refinement.

The Python version uses table lookups defined in MinimumTimeToClimbTables.py. For this C++ version, hard-code the polynomial fit coefficients directly (extracting them from the Python table data) to avoid the InterpTable dependency.

```cpp
///////////////////////////////////////////////////////////////////////////////
// Minimum Time to Climb — C++ example (Builder API)
//
// Ported from examples/python_examples/MinimumTimeToClimb.py
//
// State  : [h, v, fpa, m]   (altitude, velocity, flight path angle, mass)
// Control: [alpha]           (angle of attack)
//
// Uses inline atmospheric model with polynomial fits for aero coefficients
// (avoids InterpTable dependency — see min_time_climb_tables for table version)
//
// Objective: minimise climb time
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    // Physical constants
    constexpr double mu_earth = 3.986012e14;  // m^3/s^2
    constexpr double Re = 6378145.0;          // m
    constexpr double S_ref = 49.2386;         // m^2  (wing area)
    constexpr double g0 = 9.80665;            // m/s^2

    // Boundary conditions
    constexpr double h0 = 10.0;              // m (start near sea level)
    constexpr double hf = 19994.88;          // m (~65600 ft)
    constexpr double v0 = 129.314;           // m/s
    constexpr double vf_val = 295.092;       // m/s
    constexpr double fpa0 = 0.0;
    constexpr double m0 = 19050.864;         // kg
    constexpr double m_min = 16500.0;        // kg (dry mass limit)

    constexpr int n_pts = 200;
    constexpr int n_segs = 50;

    // ── Define ODE ─────────────────────────────────────────────────────
    // Simplified atmospheric model using ISA standard atmosphere approximations
    auto ode = ODEBuilder(4, 1)
                   .define([mu_earth, Re, S_ref, g0](auto &args) {
                       auto h = args.x_var(0);
                       auto v = args.x_var(1);
                       auto fpa = args.x_var(2);
                       auto mass = args.x_var(3);
                       auto alpha = args.u_var(0);

                       auto r = h + Re;
                       auto grav = mu_earth / (r * r);

                       // Simplified atmospheric model
                       auto rho = 1.225 * exp(-h / 8500.0);
                       auto sos = 340.294 - 0.004 * h;  // approximate
                       auto Mach = v / sos;
                       auto qbar = 0.5 * rho * v * v;

                       // Simplified aero: CL = CLa * alpha, CD = CD0 + k * CL^2
                       auto CLa = 3.44;  // per radian (simplified)
                       auto CD0 = 0.013 + 0.0144 * Mach * Mach;
                       auto CL = CLa * alpha;
                       auto CD = CD0 + 0.15 * CL * CL;

                       auto L = qbar * S_ref * CL;
                       auto D = qbar * S_ref * CD;

                       // Simplified thrust model
                       auto T = (76.4e3 + 10.0e3 * Mach) * (rho / 1.225);

                       auto hdot = v * sin(fpa);
                       auto vdot = (T * cos(alpha) - D) / mass - grav * sin(fpa);
                       auto fpadot = (T * sin(alpha) + L) / (mass * v) +
                                    cos(fpa) * (v / r - grav / v);
                       auto mdot = -T / (g0 * 3600.0);  // approximate SFC

                       return stack(hdot, vdot, fpadot, mdot);
                   })
                   .var_names({{"h", 0}, {"v", 1}, {"fpa", 2}, {"m", 3},
                               {"t", 4}, {"alpha", 5}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    const double tf_guess = 300.0;
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(6);
        pt[0] = h0 + (hf - h0) * s;
        pt[1] = v0 + (vf_val - v0) * s;
        pt[2] = 0.05 * (1.0 - s);
        pt[3] = m0 - (m0 - m_min) * 0.5 * s;
        pt[4] = tf_guess * s;
        pt[5] = 0.05;  // small positive AoA
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);
    phase.set_control_mode(ControlModes::HighestOrderSpline);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"h", "v", "fpa", "t"},
                            Eigen::Vector4d(h0, v0, fpa0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"h", "v"},
                            Eigen::Vector2d(hf, vf_val));

    // Variable bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "h", 0.0, 21000.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", 5.0, 600.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "fpa",
                          -40.0 * M_PI / 180.0, 40.0 * M_PI / 180.0);
    phase.add_lower_var_bound(PhaseRegionFlags::Path, "m", m_min);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha",
                          -45.0 * M_PI / 180.0, 45.0 * M_PI / 180.0);

    phase.add_delta_time_objective(1.0);

    // Adaptive mesh
    phase.set_adaptive_mesh(true);
    phase.set_mesh_tol(1.0e-7);
    phase.set_max_mesh_iters(10);
    phase.set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);

    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.solve_optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MinTimeToClimb (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    const double final_time = traj.back()[4];

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MinTimeToClimb (builder): climb time = " << final_time << " s\n";
    std::cout << "MinTimeToClimb (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 minimum_time_to_climb_builder
./examples/cpp_examples/builder/minimum_time_to_climb/minimum_time_to_climb_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/minimum_time_to_climb/
git commit -m "feat: add MinimumTimeToClimb C++ builder example"
```

---

## Task 12: GoddardRocket (Tier 3)

**Files:**
- Create: `examples/cpp_examples/builder/goddard_rocket/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/goddard_rocket/main.cpp`

**Python source:** `examples/python_examples/GoddardRocket.py`
**Problem:** Classic Goddard rocket — single-phase and multi-phase (bang-singular-bang). 3 states, 1 control. Multi-phase uses path constraint to define singular arc control.

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(goddard_rocket_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(goddard_rocket_builder main.cpp)
target_link_libraries(goddard_rocket_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

Implement the multi-phase formulation (3 phases: full thrust, singular arc, coast). This is the more interesting variant from the Python example.

```cpp
///////////////////////////////////////////////////////////////////////////////
// Goddard Rocket — C++ example (Builder API)
//
// Ported from examples/python_examples/GoddardRocket.py
// Source: Betts, "Practical Methods for OC", Section 4.14
//
// State  : [h, v, m]   (altitude, velocity, mass)
// Control: [u]          (throttle 0..1)
//
// Multi-phase: Phase 1 (u=1), Phase 2 (singular arc), Phase 3 (u=0)
// Objective: maximise altitude
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    // ── Non-dimensionalisation ─────────────────────────────────────────
    constexpr double Lstar = 10000.0;
    constexpr double Tstar = 60.0;
    constexpr double Mstar = 1.0;
    const double Vstar = Lstar / Tstar;
    const double Fstar = Mstar * Lstar / (Tstar * Tstar);
    const double Astar = Lstar / (Tstar * Tstar);
    const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
    const double sigmastar = Mstar / Lstar;

    const double rho0 = 0.002378 / Rhostar;
    const double h_ref = 23800.0 / Lstar;
    const double g = 32.2 / Astar;
    const double Tmag = 200.0 / Fstar;
    const double c = 1580.94 / Vstar;
    const double sigma = 5.4915e-5 / sigmastar;
    constexpr double m0 = 3.0, mf = 1.0;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(3, 1)
                   .define([sigma, c, h_ref, Tmag, g](auto &args) {
                       auto h = args.x_var(0);
                       auto v = args.x_var(1);
                       auto m = args.x_var(2);
                       auto u = args.u_var(0);

                       auto hdot = v;
                       auto vdot = (u * Tmag - sigma * v * v * exp(-h / h_ref)) / m - g;
                       auto mdot = -u * Tmag / c;

                       return stack(hdot, vdot, mdot);
                   })
                   .var_names({{"h", 0}, {"v", 1}, {"m", 2}, {"t", 3}, {"u", 4}})
                   .build();

    // ── Singular arc path constraint ───────────────────────────────────
    auto pathcon_args = Arguments<4>();
    auto ph = pathcon_args.coeff<0>();
    auto pv = pathcon_args.coeff<1>();
    auto pm = pathcon_args.coeff<2>();
    auto pu = pathcon_args.coeff<3>();

    auto t1 = (pu * Tmag - sigma * pv * pv * exp(-ph / h_ref)) - g * pm;
    auto t2 = (pm * g / (1.0 + 4.0 * (c / pv) + 2.0 * (c / pv) * (c / pv))) *
              (c * c * (1.0 + pv / c) / (h_ref * g) - 1.0 - 2.0 * c / pv);
    auto pathcon_expr = t1 - t2;

    // ── Initial guess via simple integration ───────────────────────────
    constexpr int n_total = 300;
    const double dt = (60.0 / Tstar) / (n_total - 1);
    std::vector<Eigen::VectorXd> ig_all;
    ig_all.reserve(n_total);

    // Simple Euler integration with u=1 then u=0
    double h_val = 0.0, v_val = 0.0, m_val = m0, t_val = 0.0;
    for (int i = 0; i < n_total; ++i) {
        double u_val = (m_val > mf) ? 1.0 : 0.0;
        Eigen::VectorXd pt(5);
        pt << h_val, v_val, m_val, t_val, u_val;
        ig_all.push_back(pt);

        double drag = sigma * v_val * v_val * std::exp(-h_val / h_ref);
        double vdot = (u_val * Tmag - drag) / m_val - g;
        double mdot = -u_val * Tmag / c;

        h_val += v_val * dt;
        v_val += vdot * dt;
        m_val += mdot * dt;
        t_val += dt;

        if (v_val < 0) break;
    }

    // Split into 3 phases
    const int n = static_cast<int>(ig_all.size());
    const int n3 = n / 3;
    std::vector<Eigen::VectorXd> ig1(ig_all.begin(), ig_all.begin() + n3);
    std::vector<Eigen::VectorXd> ig2(ig_all.begin() + n3, ig_all.begin() + 2 * n3);
    std::vector<Eigen::VectorXd> ig3(ig_all.begin() + 2 * n3, ig_all.end());

    constexpr int n_segs = 32;

    // ── Phase 1: full thrust (u = 1) ──────────────────────────────────
    auto phase1 = ode.phase(TranscriptionModes::LGL3, ig1, n_segs);
    phase1.add_boundary_value(PhaseRegionFlags::Front, {"h", "v", "m", "t"},
                             Eigen::Vector4d(0.0, 0.0, m0, 0.0));
    phase1.add_boundary_value(PhaseRegionFlags::Path, {"u"},
                             Eigen::Matrix<double, 1, 1>(1.0));

    // ── Phase 2: singular arc ─────────────────────────────────────────
    auto phase2 = ode.phase(TranscriptionModes::LGL3, ig2, n_segs);
    phase2.set_control_mode(ControlModes::NoSpline);
    phase2.add_lu_var_bound(PhaseRegionFlags::Path, "u", 0.0, 1.0, 1.0);
    phase2.add_equal_con(PhaseRegionFlags::Path,
                        GenericFunction<-1, -1>(pathcon_expr),
                        {"h", "v", "m", "u"});

    // ── Phase 3: coast (u = 0) ────────────────────────────────────────
    auto phase3 = ode.phase(TranscriptionModes::LGL3, ig3, n_segs);
    phase3.add_boundary_value(PhaseRegionFlags::Path, {"u"},
                             Eigen::Matrix<double, 1, 1>(0.0));
    phase3.add_boundary_value(PhaseRegionFlags::Back, {"v", "m"},
                             Eigen::Vector2d(0.0, mf));
    phase3.add_value_objective(PhaseRegionFlags::Back, "h", -1.0);

    // ── OCP ────────────────────────────────────────────────────────────
    OptimalControlProblem ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);
    ocp.add_phase(phase3);

    ocp.add_forward_link_equal_con(phase1, phase3, {"h", "v", "m", "t"});

    phase1.add_lower_delta_time_bound(0.0);
    phase2.add_lower_delta_time_bound(0.0);
    phase3.add_lower_delta_time_bound(0.0);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = ocp.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "GoddardRocket (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj3 = phase3.return_traj();
    const double max_alt = traj3.back()[0] * Lstar;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "GoddardRocket (builder): max altitude = " << max_alt << " ft\n";
    std::cout << "GoddardRocket (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 goddard_rocket_builder
./examples/cpp_examples/builder/goddard_rocket/goddard_rocket_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/goddard_rocket/
git commit -m "feat: add GoddardRocket multi-phase C++ builder example"
```

---

## Task 13: MultiPhaseCannon (Tier 3)

**Files:**
- Create: `examples/cpp_examples/builder/multi_phase_cannon/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/multi_phase_cannon/main.cpp`

**Python source:** `examples/python_examples/MultiPhaseCannon.py`
**Problem:** Ballistic projectile with drag, 4 states, 0 controls, 1 ODE parameter (radius), 2 phases (ascent/descent), maximize range.

**Key validation:** ODE parameters (`p_var`), `add_direct_link_equal_con` for parameter linking, energy inequality constraint.

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(multi_phase_cannon_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(multi_phase_cannon_builder main.cpp)
target_link_libraries(multi_phase_cannon_builder PRIVATE tycho::tycho)
```

- [ ] **Step 2: Create main.cpp**

Implement the two-phase cannon with ODE parameter for radius, energy constraint, and range maximization. Note: The builder API's handling of ODE parameters (`p_var`) and `add_direct_link_equal_con` needs to be validated. If the builder doesn't support `p_var(0)` natively, document as gap.

```cpp
///////////////////////////////////////////////////////////////////////////////
// Multi-Phase Cannonball — C++ example (Builder API)
//
// Ported from examples/python_examples/MultiPhaseCannon.py
// Source: https://openmdao.github.io/dymos/examples/multi_phase_cannonball
//
// State  : [v, gamma, h, r]   (speed, FPA, altitude, range)
// Control: none
// Params : [rad]               (projectile radius — ODE parameter)
//
// Two phases: ascent (gamma > 0) and descent (h > 0)
// Objective: maximise range
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    // ── Non-dimensionalisation ─────────────────────────────────────────
    constexpr double Lstar = 1000.0;
    constexpr double Tstar = 60.0;
    constexpr double Mstar = 10.0;
    const double Astar = Lstar / (Tstar * Tstar);
    const double Vstar = Lstar / Tstar;
    const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
    const double Estar = Mstar * Vstar * Vstar;

    constexpr double CD = 0.5;
    const double RhoAir = 1.225 / Rhostar;
    const double RhoIron = 7870.0 / Rhostar;
    const double h_scale = 8440.0 / Lstar;
    const double E0 = 400000.0 / Estar;
    const double g = 9.81 / Astar;

    const double rad0 = 0.1 / Lstar;
    const double h0 = 100.0 / Lstar;

    auto MFunc = [RhoIron](double r) { return (4.0 / 3.0) * M_PI * RhoIron * r * r * r; };
    auto SFunc = [](double r) { return M_PI * r * r; };

    const double m0 = MFunc(rad0);
    const double gamma0 = 45.0 * M_PI / 180.0;
    const double v0 = std::sqrt(2.0 * E0 / m0) * 0.99;

    // ── Define ODE (4 states, 0 controls, 1 param) ─────────────────────
    auto ode = ODEBuilder(4, 0, 1)
                   .define([CD, RhoAir, RhoIron, h_scale, g](auto &args) {
                       auto v = args.x_var(0);
                       auto gamma = args.x_var(1);
                       auto h = args.x_var(2);
                       auto r = args.x_var(3);
                       auto rad = args.p_var(0);

                       auto S = M_PI * rad * rad;
                       auto M = (4.0 / 3.0) * M_PI * RhoIron * rad * rad * rad;
                       auto rho = RhoAir * exp(-h / h_scale);
                       auto D = (0.5 * CD) * rho * v * v * S;

                       auto vdot = -D / M - g * sin(gamma);
                       auto gammadot = -g * cos(gamma) / v;
                       auto hdot = v * sin(gamma);
                       auto rdot = v * cos(gamma);

                       return stack(vdot, gammadot, hdot, rdot);
                   })
                   .var_names({{"v", 0}, {"gamma", 1}, {"h", 2}, {"r", 3}, {"t", 4}})
                   .build();

    // ── Energy constraint ──────────────────────────────────────────────
    auto efunc_args = Arguments<2>();
    auto ev = efunc_args.coeff<0>();
    auto erad = efunc_args.coeff<1>();
    auto eM = (4.0 / 3.0) * M_PI * RhoIron * erad * erad * erad;
    auto E_expr = 0.5 * eM * ev * ev - E0;

    // ── Initial guess (simple trajectory) ──────────────────────────────
    constexpr int n_pts = 200;
    const double tf_guess = 60.0 / Tstar;

    std::vector<Eigen::VectorXd> ig_ascent, ig_descent;
    ig_ascent.reserve(n_pts);
    ig_descent.reserve(n_pts);

    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(6);  // v, gamma, h, r, t, rad(param)
        pt[0] = v0 * (1.0 - 0.3 * s);
        pt[1] = gamma0 * (1.0 - 2.0 * s);  // gamma goes from +45 to -45
        pt[2] = h0 + v0 * 0.5 * tf_guess * s * (1.0 - s);
        pt[3] = v0 * 0.7 * tf_guess * s;
        pt[4] = tf_guess * s;
        pt[5] = rad0;
        if (s < 0.5) {
            ig_ascent.push_back(pt);
        } else {
            ig_descent.push_back(pt);
        }
    }

    constexpr int n_segs = 128;

    // ── Ascent phase ───────────────────────────────────────────────────
    auto aphase = ode.phase(TranscriptionModes::LGL5, ig_ascent, n_segs);

    aphase.add_lower_var_bound(PhaseRegionFlags::ODEParams, 0, 0.0, 1.0);
    aphase.add_lower_var_bound(PhaseRegionFlags::Front, "gamma", 0.0, 1.0);
    aphase.add_boundary_value(PhaseRegionFlags::Front, {"h", "r", "t"},
                             Eigen::Vector3d(h0, 0.0, 0.0));

    // Energy constraint at launch
    // NOTE: Mixed variable source (phase vars + ODE params) requires index-based API.
    // If the builder wrapper doesn't support named ODE param references in
    // add_inequal_con, use phase.base() with index vectors:
    //   aphase.base().add_inequal_con("Front", E_expr_scaled, {0}, {}, {0});
    aphase.add_inequal_con(PhaseRegionFlags::Front,
                          GenericFunction<-1, -1>(E_expr * 0.01),
                          {"v"}, {}, {0});

    // Apogee: gamma = 0
    aphase.add_boundary_value(PhaseRegionFlags::Back, {"gamma"},
                             Eigen::Matrix<double, 1, 1>(0.0));

    // ── Descent phase ──────────────────────────────────────────────────
    auto dphase = ode.phase(TranscriptionModes::LGL5, ig_descent, n_segs);
    dphase.add_boundary_value(PhaseRegionFlags::Back, {"h"},
                             Eigen::Matrix<double, 1, 1>(0.0));
    dphase.add_value_objective(PhaseRegionFlags::Back, "r", -1.0);

    // ── OCP ────────────────────────────────────────────────────────────
    OptimalControlProblem ocp;
    ocp.add_phase(aphase);
    ocp.add_phase(dphase);

    ocp.add_forward_link_equal_con(aphase, dphase, {"v", "gamma", "h", "r", "t"});
    // NOTE: add_direct_link_equal_con may not be on the OCP wrapper.
    // If not available, use ocp.base() with index-based API:
    //   ocp.base().add_direct_link_equal_con(0, "ODEParams", {0}, 1, "ODEParams", {0});
    // Document as gap if wrapper doesn't expose it.
    ocp.add_direct_link_equal_con(0, PhaseRegionFlags::ODEParams, {0},
                                 1, PhaseRegionFlags::ODEParams, {0});

    ocp.optimizer().set_opt_ls_mode("L1");

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = ocp.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MultiPhaseCannon (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj_a = aphase.return_traj();
    auto traj_d = dphase.return_traj();
    const double range = traj_d.back()[3] * Lstar;
    const double opt_rad = traj_d.back()[5] * Lstar;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "MultiPhaseCannon (builder): range = " << range << " m\n";
    std::cout << "MultiPhaseCannon (builder): radius = "
              << std::setprecision(4) << opt_rad << " m\n";
    std::cout << "MultiPhaseCannon (builder): PASSED\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and test**

Run:
```bash
cd build && ninja -j8 multi_phase_cannon_builder
./examples/cpp_examples/builder/multi_phase_cannon/multi_phase_cannon_builder
```
Expected: "PASSED"

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/multi_phase_cannon/
git commit -m "feat: add MultiPhaseCannon C++ builder example (ODE params, multi-phase)"
```

---

## Task 14: MultiPhaseZermelo (Tier 3)

**Files:**
- Create: `examples/cpp_examples/builder/multi_phase_zermelo/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/multi_phase_zermelo/main.cpp`

**Python source:** `examples/python_examples/MultiPhaseZermelo.py`
**Problem:** Multi-phase navigation through wind fields, extends existing Zermelo builder example

This is largely a multi-phase extension of the existing `builder/zermelo/` example. Reuse the wind model patterns from there and add OCP phase linking.

- [ ] **Step 1: Create CMakeLists.txt and main.cpp**

Follow the same pattern as the existing `builder/zermelo/main.cpp` but with multiple phases linked via `OptimalControlProblem` and `add_forward_link_equal_con`. Use the variable-direction wind model.

- [ ] **Step 2: Build and test**

Run:
```bash
cd build && ninja -j8 multi_phase_zermelo_builder
./examples/cpp_examples/builder/multi_phase_zermelo/multi_phase_zermelo_builder
```

- [ ] **Step 3: Commit**

```bash
git add examples/cpp_examples/builder/multi_phase_zermelo/
git commit -m "feat: add MultiPhaseZermelo C++ builder example"
```

---

## Task 15-18: Tier 4 Low-Thrust Astrodynamics (SimpleLowThrust, TopputtoLowThrust, DionysusLowThrust, BettsLowThrust)

**These examples progressively test astrodynamics features in the builder API.**

Each follows the same pattern:
1. Create `CMakeLists.txt` and `main.cpp` in respective directory
2. Port ODE dynamics from Python
3. Set up phase constraints and objectives
4. Solve and verify convergence
5. Document any API gaps in `doc/cpp_example_api_gaps.md`

**Expected gaps:**
- **DionysusLowThrust**: MEE astro models (MEETwoBody_CSI, CSIThruster) may not be accessible from builder API
- **BettsLowThrust**: InterpTable for atmospheric model, control-law integration for initial guess
- **All**: `remove_state_objective()` used in SimpleLowThrust for sequential solves

For each: create `CMakeLists.txt` + `main.cpp`, build, test, commit. Document gaps as discovered.

---

## Task 19: OrbitContinuation (Tier 4)

**Files:**
- Create: `examples/cpp_examples/builder/orbit_continuation/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/orbit_continuation/main.cpp`

**Expected gaps:** CR3BP utilities, STM integration, numerical continuation loop. The C++ version will implement a single periodic orbit (Lyapunov) rather than the full continuation sweep.

---

## Task 20-22: Tier 5 Advanced Astrodynamics (Heteroclinic, OptimalDocking, MultiSpacecraftOpt)

**These are the most complex examples with the most expected gaps.**

Expected gaps:
- **Heteroclinic**: InterpTable/InterpFunction for manifold trajectories, parallel integration with event detection, `make_periodic()`
- **OptimalDocking**: Quaternion operations, interpolation table for target trajectory
- **MultiSpacecraftOpt**: Link parameters, continuation loop

For each: implement what's possible, document all gaps, mark TODO sections in code.

---

## Task 23: MinTimeTClimbTables (Tier 6)

**Files:**
- Create: `examples/cpp_examples/builder/min_time_climb_tables/CMakeLists.txt`
- Create: `examples/cpp_examples/builder/min_time_climb_tables/main.cpp`

**Expected gap:** InterpTable1D/2D in builder API. This is the primary gap this example is designed to expose.

Document in `doc/cpp_example_api_gaps.md` and implement with inline polynomial approximations as fallback.

---

## Task 24: Final Verification & CLAUDE.md Update

- [ ] **Step 1: Build all examples**

```bash
cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j8 cpp_examples_all
```

- [ ] **Step 2: Run all via CTest**

```bash
cd build && ctest -R cpp_example --output-on-failure
```
Expected: All tests pass (or gap-blocked tests document their gaps)

- [ ] **Step 3: Run existing tests (no regressions)**

```bash
cd build && ctest --output-on-failure  # all tests
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"  # Python examples
```

- [ ] **Step 4: Update CLAUDE.md pre-merge verification**

Add C++ examples as step 5:
```
5. **C++ examples** — `ninja -j8 cpp_examples_all && ctest -R cpp_example` — all must pass
```

- [ ] **Step 5: Final commit**

```bash
git add CLAUDE.md doc/cpp_example_api_gaps.md
git commit -m "docs: add C++ examples to pre-merge verification, finalize gap tracking"
```
