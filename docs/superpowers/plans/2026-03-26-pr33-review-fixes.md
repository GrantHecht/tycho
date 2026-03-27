# PR #33 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all critical and important findings from the PR #33 comprehensive review — harden public API error paths, close test coverage gaps, and fix comment inaccuracies.

**Architecture:** Pure additive fixes — no structural refactoring. Each task targets a specific review finding in a single file (or file + test pair). All changes are in `include/tycho/detail/` headers, `tests/cpp/` test files, and example comments.

**Tech Stack:** C++20, Google Test, CMake (ninja build)

---

### Task 1: Throw on empty input in `make_dynamic_sum`

**Review finding:** Critical #1 — `make_dynamic_sum` has `if (size == 0) {}` that silently returns a default-constructed `GenericFunction` with IRows=0/ORows=0.

**Files:**
- Modify: `include/tycho/detail/Summation.h:85-86`
- Test: `tests/cpp/vector_functions/test_mixed_size_ops.cpp`

- [ ] **Step 1: Write the failing test**

Add to `test_mixed_size_ops.cpp` at the end of the file:

```cpp
///////////////////////////////////////////////////////////////////////////////
// Runtime error detection
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, EmptyDynamicSumThrows) {
    std::vector<GenericFunction<-1, -1>> empty;
    EXPECT_THROW(make_dynamic_sum<GenericFunction<-1, -1>>(empty),
                 std::invalid_argument);
}
```

- [ ] **Step 2: Build and verify the test fails**

Run: `cd build && ninja -j6 tycho_tests && ./tests/cpp/tycho_tests --gtest_filter="MixedSizeOpsTest.EmptyDynamicSumThrows"`

Expected: FAIL — no exception thrown (returns default-constructed object)

- [ ] **Step 3: Fix `make_dynamic_sum` to throw on empty input**

In `include/tycho/detail/Summation.h`, replace the empty `if (size == 0) {}` block:

```cpp
    if (size == 0) {
        throw std::invalid_argument(
            "make_dynamic_sum: cannot construct a sum from an empty vector of functions");
    } else if (size == 1) {
```

- [ ] **Step 4: Build and verify the test passes**

Run: `cd build && ninja -j6 tycho_tests && ./tests/cpp/tycho_tests --gtest_filter="MixedSizeOpsTest.EmptyDynamicSumThrows"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/Summation.h tests/cpp/vector_functions/test_mixed_size_ops.cpp
git commit -m "fix: throw on empty input in make_dynamic_sum"
```

---

### Task 2: Require both phases to have registries in named-variable `addForwardLinkEqualCon`

**Review finding:** Critical #2 — when `p2.registry().empty()`, the method silently uses p1's indices for p2. If the phases have different variable layouts, this produces wrong link constraints.

**Design decision:** If the user is calling the named-variable overload, they expect name-based resolution for both phases. An empty registry on either side is an error — they should use the index-based overload instead.

**Files:**
- Modify: `include/tycho/detail/ocp_wrapper.h:38-51`
- Test: `tests/cpp/optimal_control/test_builder_brachistochrone.cpp`

- [ ] **Step 1: Write the failing tests**

Add to `test_builder_brachistochrone.cpp`:

```cpp
TEST_F(BuilderAPITest, OCPLinkThrowsWhenP2HasNoRegistry) {
    constexpr double g = 9.81;

    // Phase 1: has registry
    auto ode1 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
        .build();

    // Phase 2: no registry
    auto ode2 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .build();

    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 50; ++i) {
        double s = static_cast<double>(i) / 49.0;
        Eigen::VectorXd pt(5);
        pt << 10.0 * s, 10.0 - 5.0 * s, g * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase1 = ode1.phase(TranscriptionModes::LGL3, traj, 16);
    auto phase2 = ode2.phase(TranscriptionModes::LGL3, traj, 16);

    OCP ocp;
    ocp.addPhase(phase1);
    ocp.addPhase(phase2);

    EXPECT_THROW(
        ocp.addForwardLinkEqualCon(phase1, phase2, {"x", "y", "v", "t"}),
        std::invalid_argument);
}

TEST_F(BuilderAPITest, OCPLinkThrowsOnMismatchedRegistries) {
    constexpr double g = 9.81;

    // Phase 1: x=0, y=1, v=2
    auto ode1 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
        .build();

    // Phase 2: same names but x=1, y=0 (swapped)
    auto ode2 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .var_names({{"x", 1}, {"y", 0}, {"v", 2}, {"t", 3}, {"theta", 4}})
        .build();

    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 50; ++i) {
        double s = static_cast<double>(i) / 49.0;
        Eigen::VectorXd pt(5);
        pt << 10.0 * s, 10.0 - 5.0 * s, g * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase1 = ode1.phase(TranscriptionModes::LGL3, traj, 16);
    auto phase2 = ode2.phase(TranscriptionModes::LGL3, traj, 16);

    OCP ocp;
    ocp.addPhase(phase1);
    ocp.addPhase(phase2);

    EXPECT_THROW(
        ocp.addForwardLinkEqualCon(phase1, phase2, {"x", "y"}),
        std::invalid_argument);
}
```

- [ ] **Step 2: Build and verify the tests fail**

Run: `cd build && ninja -j6 tycho_tests && ./tests/cpp/tycho_tests --gtest_filter="BuilderAPITest.OCPLink*"`

Expected: `OCPLinkThrowsWhenP2HasNoRegistry` FAILS (no exception), `OCPLinkThrowsOnMismatchedRegistries` PASSES (exception already fires)

- [ ] **Step 3: Fix OCP to throw when either registry is empty**

In `include/tycho/detail/ocp_wrapper.h`, replace the `addForwardLinkEqualCon` string overload body:

```cpp
    int addForwardLinkEqualCon(Phase &p1, Phase &p2,
                               std::initializer_list<std::string> var_names) {
        if (p1.registry().empty() || p2.registry().empty())
            throw std::invalid_argument(
                "OCP::addForwardLinkEqualCon: both phases must have variable "
                "names registered when using the named-variable overload — "
                "register names via ODEBuilder::var_names() or use the "
                "index-based overload");
        auto idx1 = p1.registry().resolve(var_names);
        auto idx2 = p2.registry().resolve(var_names);
        if (idx1.size() != idx2.size() ||
            (idx1.array() != idx2.array()).any()) {
            throw std::invalid_argument(
                "OCP::addForwardLinkEqualCon: variable names resolve to "
                "different indices in p1 vs p2 — use index-based overload "
                "for heterogeneous phase layouts");
        }
        return ocp_.addForwardLinkEqualCon(p1.base_ptr(), p2.base_ptr(), idx1);
    }
```

- [ ] **Step 4: Update the doc comment**

Replace the doc comment above `addForwardLinkEqualCon`:

```cpp
    /// Link phases with named variables.
    /// Creates equality constraints between consecutive phase pairs from p1
    /// through p2 (using phase ordering within the OCP).  Both phases must
    /// have registries; throws if either is empty or if names resolve to
    /// different indices.
```

- [ ] **Step 5: Build and verify both tests pass**

Run: `cd build && ninja -j6 tycho_tests && ./tests/cpp/tycho_tests --gtest_filter="BuilderAPITest.OCPLink*"`

Expected: Both PASS

- [ ] **Step 6: Commit**

```bash
git add include/tycho/detail/ocp_wrapper.h tests/cpp/optimal_control/test_builder_brachistochrone.cpp
git commit -m "fix: require both phases to have registries in named-variable link constraints"
```

---

### Task 3: Add zero-count guards to `XVec()`/`UVec()`/`PVec()` no-arg overloads

**Review finding:** Important #6 — scalar accessors (`XVar`, `UVar`, `PVar`) throw for zero counts, but the vector accessors silently return zero-length segments.

**Files:**
- Modify: `include/tycho/detail/ode_builder.h:44,73,99`
- Test: `tests/cpp/optimal_control/test_ode_builder.cpp`

- [ ] **Step 1: Write the failing tests**

Add to `test_ode_builder.cpp`:

```cpp
TEST_F(ODEBuilderTest, UVecThrowsWhenNoControls) {
    ODEArgsProxy proxy(3, 0, 0);
    EXPECT_THROW(proxy.UVec(), std::invalid_argument);
}

TEST_F(ODEBuilderTest, PVecThrowsWhenNoParams) {
    ODEArgsProxy proxy(3, 1, 0);
    EXPECT_THROW(proxy.PVec(), std::invalid_argument);
}
```

Note: `XVec()` with xvars=0 cannot happen via `ODEBuilder` (which enforces `xvars > 0`). `UVec()` and `PVec()` with zero counts are the practical cases.

- [ ] **Step 2: Build and verify the tests fail**

Run: `cd build && ninja -j6 tycho_tests && ./tests/cpp/tycho_tests --gtest_filter="ODEBuilderTest.*ThrowsWhenNo*"`

Expected: FAIL — no exception thrown

- [ ] **Step 3: Add zero-count guards**

In `include/tycho/detail/ode_builder.h`, modify the three no-arg accessors:

```cpp
    /// Full state vector segment.
    auto XVec() const {
        if (args_.XVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::XVec: no state variables declared (xvars=0)");
        return args_.segment(0, args_.XVars());
    }
```

```cpp
    /// Full control vector segment.
    auto UVec() const {
        if (args_.UVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::UVec: no control variables declared (uvars=0)");
        return args_.segment(args_.XtVars(), args_.UVars());
    }
```

```cpp
    /// Full parameter vector segment.
    auto PVec() const {
        if (args_.PVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::PVec: no parameter variables declared (pvars=0)");
        return args_.segment(args_.XtUVars(), args_.PVars());
    }
```

- [ ] **Step 4: Build and verify the tests pass**

Run: `cd build && ninja -j6 tycho_tests && ./tests/cpp/tycho_tests --gtest_filter="ODEBuilderTest.*ThrowsWhenNo*"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/ode_builder.h tests/cpp/optimal_control/test_ode_builder.cpp
git commit -m "fix: throw when accessing empty variable groups via XVec()/UVec()/PVec()"
```

---

### Task 4: Add missing OCP and builder test coverage

**Review findings:** Important #3 (check_has_phases untested), Important #4 (define/from after build untested), Important #5 (runtime mismatch for relaxed operators untested)

**Files:**
- Test: `tests/cpp/optimal_control/test_builder_brachistochrone.cpp`
- Test: `tests/cpp/optimal_control/test_ode_builder.cpp`
- Test: `tests/cpp/vector_functions/test_mixed_size_ops.cpp`

- [ ] **Step 1: Add OCP check_has_phases test**

Add to `test_builder_brachistochrone.cpp`:

```cpp
TEST_F(BuilderAPITest, OCPSolveThrowsWithNoPhases) {
    OCP ocp;
    EXPECT_THROW(ocp.solve(), std::invalid_argument);
    EXPECT_THROW(ocp.optimize(), std::invalid_argument);
    EXPECT_THROW(ocp.solve_optimize(), std::invalid_argument);
    EXPECT_THROW(ocp.optimize_solve(), std::invalid_argument);
}
```

- [ ] **Step 2: Add define/from-after-build tests**

Add to `test_ode_builder.cpp`:

```cpp
TEST_F(ODEBuilderTest, DefineAfterBuildThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        return stack(args.XVar(0), args.XVar(1), args.XVar(2));
    });
    builder.build();
    EXPECT_THROW(
        builder.define([](auto &args) { return args.XVar(0); }),
        std::invalid_argument);
}

TEST_F(ODEBuilderTest, FromAfterBuildThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        return stack(args.XVar(0), args.XVar(1), args.XVar(2));
    });
    builder.build();
    auto args = ODEArguments(3, 1, 0);
    auto expr = stack(args.segment(0, 3).coeff(0),
                      args.segment(0, 3).coeff(1),
                      args.segment(0, 3).coeff(2));
    EXPECT_THROW(builder.from(expr), std::invalid_argument);
}
```

- [ ] **Step 3: Add runtime mismatch test for relaxed operators**

Add to `test_mixed_size_ops.cpp`:

```cpp
TEST_F(MixedSizeOpsTest, RuntimeMismatchThrows) {
    // Two dynamic functions with different runtime IRows
    auto args6 = Arguments<-1>(6);
    auto seg6 = args6.segment(0, 3);   // IRC=-1, IRows=6, ORows=3

    auto args8 = Arguments<-1>(8);
    auto seg8 = args8.segment(0, 3);   // IRC=-1, IRows=8, ORows=3

    // Same ORC, different IRows — should throw at construction time
    EXPECT_THROW(seg6 + seg8, std::invalid_argument);
    EXPECT_THROW(seg6 - seg8, std::invalid_argument);
}

TEST_F(MixedSizeOpsTest, RuntimeORowsMismatchThrows) {
    // Two dynamic functions with different runtime ORows
    auto args = Arguments<-1>(6);
    auto seg2 = args.segment(0, 2);   // IRC=-1, ORows=2
    auto seg3 = args.segment(0, 3);   // IRC=-1, ORows=3

    EXPECT_THROW(seg2 + seg3, std::invalid_argument);
}
```

- [ ] **Step 4: Build and verify all new tests pass**

Run: `cd build && ninja -j6 tycho_tests && ./tests/cpp/tycho_tests --gtest_filter="BuilderAPITest.OCPSolve*:ODEBuilderTest.*AfterBuild*:MixedSizeOpsTest.Runtime*"`

Expected: All PASS (these test existing guards that are already implemented)

- [ ] **Step 5: Commit**

```bash
git add tests/cpp/optimal_control/test_builder_brachistochrone.cpp \
        tests/cpp/optimal_control/test_ode_builder.cpp \
        tests/cpp/vector_functions/test_mixed_size_ops.cpp
git commit -m "test: add coverage for OCP guards, post-build mutation, and runtime mismatch"
```

---

### Task 5: Improve `check_has_phases` error message

**Review finding:** Suggestion #6 — message says what went wrong but not what to do.

**Files:**
- Modify: `include/tycho/detail/ocp_wrapper.h:98-102`

- [ ] **Step 1: Update the error message**

In `ocp_wrapper.h`, replace the `check_has_phases` method body:

```cpp
    void check_has_phases(const char *method) const {
        if (ocp_.phases.empty())
            throw std::invalid_argument(
                fmt::format("OCP::{}: no phases added — call addPhase() before solving", method));
    }
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd build && ninja -j6 tycho_tests`

Expected: Compiles cleanly

- [ ] **Step 3: Commit**

```bash
git add include/tycho/detail/ocp_wrapper.h
git commit -m "fix: make check_has_phases error message actionable"
```

---

### Task 6: Fix comment inaccuracies

**Review findings:** Comment issues #8-12 from the review.

**Files:**
- Modify: `examples/cpp_examples/builder/delta3_launch/main.cpp:122`
- Modify: `examples/cpp_examples/builder/delta3_launch/main.cpp:334`
- Modify: `tests/cpp/vector_functions/test_mixed_size_ops.cpp:78`
- Modify: `examples/cpp_examples/builder/zermelo/main.cpp:109-111`
- Modify: `examples/cpp_examples/builder/PAIN_POINTS.md:42`

- [ ] **Step 1: Fix "Atmospheric density" comment in delta3_launch**

`delta3_launch/main.cpp:122` — change:
```
    // Atmospheric density
```
to:
```
    // Exponential atmosphere model: density decay factor exp(-h / h_scale)
```

- [ ] **Step 2: Fix consecutive-pairs comment in delta3_launch**

`delta3_launch/main.cpp:334-335` — change:
```
    // Continuity in everything except mass — named variables
    ocp.addForwardLinkEqualCon(phase1, phase4, {"R", "V", "t", "u"});
```
to:
```
    // Continuity in R, V, t, u across consecutive phase pairs (1->2, 2->3, 3->4);
    // mass is discontinuous at staging events.
    ocp.addForwardLinkEqualCon(phase1, phase4, {"R", "V", "t", "u"});
```

- [ ] **Step 3: Fix IRC comment in test_mixed_size_ops**

`test_mixed_size_ops.cpp:78` — change:
```
    // Vector (IRC=-1) * Scalar (IRC=3) — mixed IRC
```
to:
```
    // Vector (IRC=-1, ORC=3) * Scalar (IRC=6, ORC=1) — mixed IRC
```

- [ ] **Step 4: Fix misleading Zermelo comment**

`zermelo/main.cpp:109-111` — replace the three comment lines:
```
    // For position-dependent wind, we need the VF DSL norm.
    // ODEArgsProxy gives scalar XVar(0), XVar(1); we need sqrt(x^2 + y^2).
    // Alternative: use from() with ODEArguments for vector operations.
```
with:
```
    // Position-dependent wind requires the full-vector norm of pos,
    // so we use ODEArguments + from() to build the expression directly.
```

- [ ] **Step 5: Fix Segment type notation in PAIN_POINTS.md**

`PAIN_POINTS.md:42` — change:
```
Note: sub-segments are dynamic-sized (`Segment<-1,-1,-1>`), producing dynamic
```
to:
```
Note: sub-segments are dynamic-sized (with dynamic ORC), producing dynamic
```

- [ ] **Step 6: Build to verify nothing is broken**

Run: `cd build && ninja -j6 tycho_tests`

Expected: Compiles cleanly

- [ ] **Step 7: Commit**

```bash
git add examples/cpp_examples/builder/delta3_launch/main.cpp \
        examples/cpp_examples/builder/zermelo/main.cpp \
        examples/cpp_examples/builder/PAIN_POINTS.md \
        tests/cpp/vector_functions/test_mixed_size_ops.cpp
git commit -m "docs: fix comment inaccuracies flagged in PR review"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full C++ test suite**

Run: `cd build && ninja -j6 tycho_tests && ctest --output-on-failure`

Expected: All tests pass (including all new tests)

- [ ] **Step 2: Run C++ brachistochrone example**

Run: `./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp`

Expected: "Optimal Solution Found", objective near 1.8013 s

- [ ] **Step 3: Run Python example suite**

Run: `cd /home/ghecht/Projects/tycho && conda run -n tycho python scripts/run_examples.py`

Expected: All examples pass (exit 0)
