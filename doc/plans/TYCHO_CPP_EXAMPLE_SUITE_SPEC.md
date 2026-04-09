# Spec: C++ Builder API Example Suite

> Port all unique Python examples to C++ using the builder API.
> Serves as both a C++ integration test suite and an API deficiency catalog.

## Motivation

The Python example suite (38 scripts) is the primary acceptance gate for Tycho.
There is no equivalent C++ coverage — only 4 builder examples exist. Porting all
unique Python examples to C++ will:

1. Validate the builder API against real problems at scale
2. Expose remaining API deficiencies (missing features, ergonomic issues)
3. Create a C++ integration test suite with convergence assertions
4. Provide reference code for C++ users

## Prerequisites

- **Builder type renames** (see `TYCHO_BUILDER_TYPE_RENAMES_SPEC.md`) must land first.
  All new examples use the renamed types (`ODE`, `OptimalControlProblem`, etc.).

## Example Inventory

23 examples after deduplication (removing UpdatedInterface duplicates and 4 already
ported). Organized into tiers by complexity and expected API gaps.

### Tier 1: Simple Single-Phase (no expected gaps)

| # | Example | States | Ctrls | Dynamics | Objective | Key Features |
|---|---------|--------|-------|----------|-----------|--------------|
| 1 | CartPole | 4 | 1 | Swingup, matrix inverse | min control effort | Matrix VF operations |
| 2 | VanDerPol | 2 | 1 | Oscillator | min integral(x0^2+x1^2+u^2) | BlockConstant control |
| 3 | BrysonDenham | 2 | 1 | Point mass | min integral(u^2/2) | State upper bound (x<=1/9) |
| 4 | MountainCar | 2 | 1 | Terrain oscillator | min time | State bounds, scaling |
| 5 | ParallelParking | 6 | 2+1p | Bicycle kinematics | min time | Path constraints (slot), BlockConstant, manual mesh (two k values) |
| 6 | AnalyticExample | 1 | 1 | Linear | min custom integral | Custom objective function |

### Tier 2: Single-Phase with Special Features

| # | Example | States | Ctrls | Dynamics | Objective | Key Features |
|---|---------|--------|-------|----------|-----------|--------------|
| 7 | BikeObstacle | 4 | 2 | Bicycle kinematics | min time | Obstacle avoidance inequality |
| 8 | FreeFlyingRobot | 6 | 4 | Planar 6-DOF | min integral(u_sum) | Large control vector |
| 9 | HangingChain | 1 | 1+1p | Chain statics | min potential energy | Integral parameter constraint, Jet parallel solve |
| 10 | Reentry | 5 | 2 | Shuttle reentry | max cross-range | Heating constraint, adaptive mesh, two-step solve |
| 11 | MinimumTimeToClimb | 4 | 1 | Aircraft climb | min time | Inline atmospheric model, adaptive mesh |

### Tier 3: Multi-Phase

| # | Example | States | Ctrls | Phases | Dynamics | Objective | Key Features |
|---|---------|--------|-------|--------|----------|-----------|--------------|
| 12 | GoddardRocket | 3 | 1 | 3 | Rocket ascent | max altitude | Singular arc, stage separation |
| 13 | MultiPhaseCannon | 4 | 0 | 2 | Ballistic trajectory | max range | ODE parameter opt, phase-dependent controls |
| 14 | MultiPhaseZermelo | 2 | 1 | N | Point navigation | min time | Region-dependent wind, continuation |

### Tier 4: Low-Thrust Astrodynamics

| # | Example | States | Ctrls | Dynamics | Objective | Expected Gaps |
|---|---------|--------|-------|----------|-----------|---------------|
| 15 | SimpleLowThrust | 6 | 3 | Two-body + thrust | min time/power/mass | Likely none (Cartesian dynamics) |
| 16 | TopputtoLowThrust | 4 | 2 | Polar orbit transfer | min time, min mass | Control-law integration for initial guess |
| 17 | DionysusLowThrust | 7 | 3 | MEE + CSI thruster | max final mass | **MEE astro models from C++** |
| 18 | BettsLowThrust | 7 | 3+1p | MEE + J2/J3/J4 | max final mass | **Interp tables, MEE, control-law integration** |
| 19 | OrbitContinuation | 7 | 0 | CR3BP | Periodic orbits | **CR3BP utils, STM integration, continuation** |

### Tier 5: Advanced Astrodynamics

| # | Example | States | Ctrls | Phases | Dynamics | Expected Gaps |
|---|---------|--------|-------|--------|----------|---------------|
| 20 | Heteroclinic | 7 | 0 | 2 | CR3BP manifolds | **CR3BP, manifold generation, event detection** |
| 21 | OptimalDocking | 13-20 | 6 | 1 | Relative quaternion | **Quaternion ops, interp table for target traj** |
| 22 | MultiSpacecraftOpt | 6*N | 3*N | N | Two-body multi-agent | **Link parameters, continuation loop** |

### Tier 6: Interpolation Tables

| # | Example | States | Ctrls | Dynamics | Expected Gaps |
|---|---------|--------|-------|----------|---------------|
| 23 | MinTimeTClimbTables | 4 | 1 | Aircraft + table aero | **InterpTable1D/2D in builder API** |

## Directory Structure

All new examples under `examples/cpp_examples/builder/<snake_case_name>/`:

```
examples/cpp_examples/builder/
  brachistochrone/          # (existing)
  zermelo/                  # (existing)
  hypersens/                # (existing)
  delta3_launch/            # (existing)
  cart_pole/                # new
  van_der_pol/              # new
  bryson_denham/            # new
  mountain_car/             # new
  parallel_parking/         # new
  analytic_example/         # new
  bike_obstacle/            # new
  free_flying_robot/        # new
  hanging_chain/            # new
  reentry/                  # new
  minimum_time_to_climb/    # new
  goddard_rocket/           # new
  multi_phase_cannon/       # new
  multi_phase_zermelo/      # new
  simple_low_thrust/        # new
  topputto_low_thrust/      # new
  dionysus_low_thrust/      # new
  betts_low_thrust/         # new
  orbit_continuation/       # new
  heteroclinic/             # new
  optimal_docking/          # new
  multi_spacecraft_opt/     # new
  min_time_climb_tables/    # new
```

Each directory contains `CMakeLists.txt` + `main.cpp`.

## Example Structure (template)

```cpp
#include "tycho/tycho.h"
#include <cmath>
#include <cstdlib>
#include <fmt/core.h>

using namespace tycho;

int main() {
    // 1. Define ODE
    auto ode = oc::ODEBuilder(nstates, ncontrols)
        .var_names({...})
        .define([](auto& args) { ... })
        .build();

    // 2. Configure phase
    auto phase = ode.phase(transcription, segments);
    phase.add_boundary_value(...);
    phase.add_delta_time_objective(1.0);
    // ... constraints, bounds, objectives

    // 3. Solve
    phase.optimizer().set_...();
    auto flag = phase.solve_optimize();

    // 4. Extract and verify results
    auto traj = phase.return_traj();
    double obj = phase.return_objective();
    fmt::print("Objective: {:.6f}\n", obj);

    // 5. Convergence assertion
    if (flag != oc::ConvergenceFlags::CONVERGED) {
        fmt::print(stderr, "FAILED: solver did not converge\n");
        return EXIT_FAILURE;
    }
    if (std::abs(obj - EXPECTED_OBJ) > TOLERANCE) {
        fmt::print(stderr, "FAILED: objective {:.6f} != expected {:.6f}\n", obj, EXPECTED_OBJ);
        return EXIT_FAILURE;
    }
    fmt::print("PASSED\n");
    return EXIT_SUCCESS;
}
```

## Build System

- Each example gets a `CMakeLists.txt` that adds an executable and links `tycho`
- `examples/cpp_examples/CMakeLists.txt` conditionally adds all subdirectories
  when `BUILD_CPP_EXAMPLES=ON`
- All example executables placed in `heavy_compile` job pool
- A top-level CMake target `cpp_examples_all` builds all examples

## Gap Tracking

A new file `doc/cpp_example_api_gaps.md` will track every feature gap discovered:

```markdown
# C++ Builder API Gaps (discovered during example porting)

| Example | Feature | Severity | Description | Suggested Fix |
|---------|---------|----------|-------------|---------------|
| BettsLT | InterpTable | High | No InterpTable1D/2D in builder API | Expose via ODE lambda or add builder method |
| ... | ... | ... | ... | ... |
```

Examples blocked by gaps will compile but include `// TODO: <description>` markers
at the blocked sections, with the gap tracked in the above document.

## Verification

1. `cmake --preset <preset> -DBUILD_CPP_EXAMPLES=ON`
2. `ninja -j8 cpp_examples_all` — all examples compile
3. Each example runs and prints `PASSED` (or `TODO: <gap>` for blocked features)
4. `ctest -R cpp_example` — integrated into test suite
5. No regressions in existing tests or Python examples

## Implementation Order

1. Tier 1 (6 examples) — validates core builder API
2. Tier 2 (5 examples) — mesh refinement, special constraints, Jet parallel
3. Tier 3 (3 examples) — multi-phase, OCP linking
4. Tier 4 (5 examples) — low-thrust, astro features (gap discovery)
5. Tier 5 (3 examples) — advanced astro (most gaps expected)
6. Tier 6 (1 example) — interpolation tables

Each tier is independently valuable and can be merged separately.
