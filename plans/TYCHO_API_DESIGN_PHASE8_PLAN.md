# Phase 8: Refine Static DSL Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Update the Phase 4 static DSL examples to use the improvements from Phase 7 (compile-time named variables, firebreaks, improved ODE syntax, multi-parameter macros). Validates that the improvements make the code cleaner and measures compile-time improvements.

**Architecture:** Each static DSL example is rewritten using the settled-upon ODE definition syntax. Compile times and memory usage are measured before and after.

**Tech Stack:** C++20, Tycho improved static DSL

---

### Task 1: Create feature branch

```bash
git checkout -b feat/refine-static-examples main
```

---

### Task 2: Record baseline compile metrics

**Files:**
- Update: `examples/cpp_examples/static/PAIN_POINTS.md`

**Step 1: Measure current compile times**

For each static example, record:
```bash
# Clean build of single example with timing
time ninja -j1 brachistochrone_cpp
time ninja -j1 zermelo_cpp
time ninja -j1 hypersens_cpp
time ninja -j1 delta3_launch_cpp
```

Also record peak memory (use `/usr/bin/time -l` on macOS).

**Step 2: Save baseline**

Document in `PAIN_POINTS.md` under a "Baseline Metrics" section.

**Step 3: Commit**

---

### Task 3: Refine Brachistochrone example

**Files:**
- Modify: `examples/cpp_examples/static/brachistochrone/main.cpp`

**Step 1: Apply chosen ODE syntax (Option A or B)**

Replace the existing `BUILD_ODE_FROM_EXPRESSION` definition with the settled
syntax from Phase 7.

**Step 2: Add compile-time named variables**

```cpp
constexpr auto x_var = tycho::XVar<0>;
constexpr auto y_var = tycho::XVar<1>;
constexpr auto v_var = tycho::XVar<2>;
constexpr auto theta_var = tycho::UVar<0>;
```

Use these for boundary conditions:
```cpp
// Before (manual indices):
Eigen::VectorXi idx(4); idx << 0, 1, 2, 3;
phase->addBoundaryValue(PhaseRegionFlags::Front, idx, vals);

// After (named, if the static API supports it):
// Use tag-based indexing for clarity even if not runtime-resolved
```

**Step 3: Build, verify convergence, measure compile time**

**Step 4: Commit**

---

### Task 4: Refine Zermelo example

**Files:**
- Modify: `examples/cpp_examples/static/zermelo/main.cpp`

**Step 1: Apply improved syntax**

**Step 2: Use firebreaks if wind functions cause template bloat**

If the composable wind function pattern creates deep template trees, insert
`tycho::barrier()` at the wind function boundary.

**Step 3: Build, verify, measure, commit**

---

### Task 5: Refine HyperSensLong example

**Files:**
- Modify: `examples/cpp_examples/static/hypersens/main.cpp`

Simple ODE — improvements here are primarily the cleaner definition syntax.

**Step 1: Apply improved syntax, build, verify, commit**

---

### Task 6: Refine Delta3Launch example

**Files:**
- Modify: `examples/cpp_examples/static/delta3_launch/main.cpp`

This is the key example. Expected improvements:
- Multi-parameter ODE macro works for `(double T, double mdot)`
- Firebreaks reduce template memory for `Arguments<11>`
- Named variable tags improve readability of index references

**Step 1: Apply multi-parameter ODE fix**

```cpp
// Before (workaround):
struct RocketParams { double T; double mdot; };
BUILD_ODE_FROM_EXPRESSION(RocketODE, RocketODE_Impl, RocketParams);
// ... or hardcoded constants

// After:
TYCHO_BUILD_ODE(RocketODE, RocketODE_Impl, double, double);
// or single-struct pattern with multiple constructor args
```

**Step 2: Add firebreaks if needed**

If template memory is still problematic, insert barriers in the dynamics:
```cpp
auto drag = tycho::barrier(drag_expr);
auto gravity = tycho::barrier(gravity_expr);
auto Vdot = gravity + (T * u + drag) / m;
```

**Step 3: Add named variable tags**

```cpp
constexpr auto R = tycho::XVar<0>;  // (or use a group concept)
// ...
```

**Step 4: Build with memory tracking**

```bash
/usr/bin/time -l ninja -j1 delta3_launch_cpp
```

Compare against baseline.

**Step 5: Commit**

---

### Task 7: Document improvements and final metrics

**Files:**
- Update: `examples/cpp_examples/static/PAIN_POINTS.md`

**Step 1: Record post-improvement metrics**

| Example | Before (time/memory) | After (time/memory) | Improvement |
|---------|---------------------|---------------------|-------------|
| Brachistochrone | ... | ... | ... |
| Zermelo | ... | ... | ... |
| HyperSens | ... | ... | ... |
| Delta3Launch | ... | ... | ... |

**Step 2: Document remaining pain points**

Any issues that persist feed into the Phase 9 backlog.

**Step 3: Commit**

---

### Task 8: Full validation

**Step 1: All C++ examples converge**

```bash
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
./examples/cpp_examples/static/zermelo/zermelo_cpp
./examples/cpp_examples/static/hypersens/hypersens_cpp
./examples/cpp_examples/static/delta3_launch/delta3_launch_cpp
```

**Step 2: Solutions match pre-improvement results**

**Step 3: All tests pass**

```bash
ctest --output-on-failure
python scripts/run_examples.py
```

---

## Validation Checklist

- [ ] All 4 static examples use the improved ODE definition syntax
- [ ] Compile-time named variables used where applicable
- [ ] Firebreaks applied to Delta3Launch if needed
- [ ] Multi-parameter ODE macro used for RocketODE
- [ ] All examples converge to same solutions as before improvements
- [ ] Compile time and memory improvements documented
- [ ] Remaining pain points captured for Phase 9 backlog
