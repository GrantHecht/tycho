# Apple Accelerate Interface Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn PR #88's four isolated Accelerate fixes into a scoped overhaul of the Apple Accelerate sparse-solver backend, establishing six invariants and giving the interface its first test coverage.

**Architecture:** All changes are confined to two headers under `include/tycho/detail/solvers/linear/` plus one new test file. `src/solvers/psiopt.cpp` is deliberately untouched: platform-capability policy moves *into* the solver interface (`set_order()`) so PSIOPT keeps passing its preference unchanged. Each invariant is one commit; each is independently revertible.

**Tech Stack:** C++20, Eigen 5 (bundled), Apple Accelerate (`Sparse/Solve.h`), Google Test, CMake + Ninja, clang 22 from Homebrew LLVM.

**Spec:** `docs/dev/plans/2026-07-24-accelerate-interface-overhaul-design.md` — read §2 (decisions) and §3 (invariants) before starting. Deferred work is tracked in issue #105.

## Global Constraints

- **Platform:** every change is macOS-only code (`USE_ACCELERATE_SPARSE`, an Apple-only global define at `CMakeLists.txt:710`). It cannot be compiled on Linux/Windows. Do not touch `pardiso_interface.h` — that is issue #105's scope.
- **Build parallelism: `-j4`** on this 16 GB machine. **Never start a second build while one is running** — two concurrent builds OOM the system. `TYCHO_HEAVY_COMPILE_JOBS=1`.
- **Build command:** `cmake --preset macos-llvm-release -DBUILD_CPP_TESTS=ON`, then `cd build && ninja -j4 <target>`. Requires `conda activate tycho` first (use `eval "$(conda shell.bash hook)" && conda activate tycho` under fish).
- **Licence:** `accelerate_interface.h` is derived from Eigen's `AccelerateSupport` and is **MPL-2.0**. Preserve its existing header comment block verbatim. Never modify or delete anything in `notices/`.
- **Formatting:** do **not** run the global `ninja clang-format` target (version churn reformats unrelated files). Format only the files you touched:
  `/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h`
- **Naming:** member variables `snake_case_` with trailing underscore; compile-time constants `kPascalCase`; free functions `snake_case`. Note this file predates the convention and uses Eigen's `m_` prefix in places — match the surrounding code, do not rename existing members.
- **Commit prefixes:** `fix:`, `refactor:`, `test:`, `docs:`, `chore:`. End every commit message with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- **Human review:** CLAUDE.md requires Grant's explicit review for Apple Accelerate integration changes. Do not merge.
- **Probes:** verify non-obvious behavior with a standalone probe rather than by reading. Compiles in ~3 s:
  ```bash
  /opt/homebrew/opt/llvm/bin/clang++ -std=c++20 -O2 -DNDEBUG -DFMT_HEADER_ONLY \
    -I include -I dep/eigen -I dep/fmt/include probe.cpp -framework Accelerate -o probe
  ```
  Put probes in the session scratchpad, never in the repo.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `include/tycho/detail/solvers/linear/accelerate_interface.h` | The `AccelerateImpl` solver template — all six invariants | Modify (896 lines) |
| `include/tycho/detail/solvers/linear/accelerate_utils.h` | Process-wide Accelerate init, thread control, warmup | Modify (107 lines) |
| `tests/cpp/solvers/test_accelerate_interface.cpp` | Regression coverage for every invariant | **Create** |
| `tests/cpp/CMakeLists.txt` | Register the new test in the light target | Modify (one line at `:55`) |
| `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification.md` | macOS verification handoff | Modify |

Everything lands in the light test target: `tycho_add_test_executable(tycho_tests_light ...)` at `tests/cpp/CMakeLists.txt:214` already links `-framework Accelerate` (verified in `build/build.ninja:2110`) and `USE_ACCELERATE_SPARSE` is global, so **no build-system change beyond the one source line is needed.**

---

## Task 1: Test scaffold and build wiring

Enabling task: proves the leaf-TU premise inside the real build before any invariant depends on it.

**Files:**
- Create: `tests/cpp/solvers/test_accelerate_interface.cpp`
- Modify: `tests/cpp/CMakeLists.txt:55` (add to `TYCHO_TEST_LIGHT_SOURCES`)

**Interfaces:**
- Consumes: nothing.
- Produces: test fixtures used by every later task — `SpMat`, `LDLT`, `LLT`, `spd_both_triangles()`, `spd_full_upper()`, `indefinite_full_upper()`, `dense_symmetric()`. Later tasks add `TEST`s to this same file.

- [ ] **Step 1: Create the test file with fixtures and one wiring test**

```cpp
///////////////////////////////////////////////////////////////////////////////
// Apple Accelerate sparse-solver interface tests.
//
// LEAF-HEADER TU: includes only accelerate_interface.h + Eigen + gtest, never
// tycho/tycho.h or test_utils.h, so this stays in TYCHO_TEST_LIGHT_SOURCES
// (~200 MB, seconds to compile) rather than the 4-7 GB heavy target. Do not
// add includes that pull in the tycho umbrella.
//
// macOS-only: the interface under test is Apple-only, so off-platform this
// compiles to an empty TU -- the inverse of solvers/test_jet_mkl_guard.cpp.
//
// Design: docs/dev/plans/2026-07-24-accelerate-interface-overhaul-design.md
///////////////////////////////////////////////////////////////////////////////

#ifdef USE_ACCELERATE_SPARSE

#include "tycho/detail/solvers/linear/accelerate_interface.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using LDLT = Eigen::AccelerateLDLTTPP<SpMat, Eigen::Upper>;
using LLT = Eigen::AccelerateLLT<SpMat, Eigen::Upper>;

// A = [[4,1,0],[1,3,1],[0,1,2]] -- SPD, with BOTH triangles populated so the
// triangle-canonicalization path (PR 88's P1 fix) is exercised.
SpMat spd_both_triangles() {
    SpMat A(3, 3);
    A.insert(0, 0) = 4.0;
    A.insert(0, 1) = 1.0;
    A.insert(1, 0) = 1.0;
    A.insert(1, 1) = 3.0;
    A.insert(1, 2) = 1.0;
    A.insert(2, 1) = 1.0;
    A.insert(2, 2) = 2.0;
    A.makeCompressed();
    return A;
}

// Dense upper triangle, SPD. Shares its nonzero pattern with
// indefinite_full_upper() so one can be refactored into the other.
SpMat spd_full_upper() {
    SpMat A(3, 3);
    A.insert(0, 0) = 4.0;
    A.insert(0, 1) = 1.0;
    A.insert(0, 2) = 1.0;
    A.insert(1, 1) = 4.0;
    A.insert(1, 2) = 1.0;
    A.insert(2, 2) = 4.0;
    A.makeCompressed();
    return A;
}

// Same pattern as spd_full_upper(), but indefinite: Cholesky MUST fail on it.
// (LDLT^TPP will not -- it handles indefinite matrices by design, and probing
// showed it reports SparseStatusOK even for a zero matrix. Failure tests must
// use LLT.)
SpMat indefinite_full_upper() {
    SpMat A(3, 3);
    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = 8.0;
    A.insert(0, 2) = 8.0;
    A.insert(1, 1) = 1.0;
    A.insert(1, 2) = 8.0;
    A.insert(2, 2) = 1.0;
    A.makeCompressed();
    return A;
}

// Expand a triangle-stored sparse matrix to its full dense symmetric form,
// for residual checks.
Eigen::MatrixXd dense_symmetric(const SpMat &A) {
    Eigen::MatrixXd d = Eigen::MatrixXd(A);
    Eigen::MatrixXd full = d;
    full.triangularView<Eigen::Lower>() = d.transpose();
    return full;
}

TEST(AccelerateInterface, SolvesAnSpdSystem) {
    LDLT s;
    const SpMat A = spd_both_triangles();
    s.compute(A);
    ASSERT_EQ(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    const Eigen::VectorXd x = s.solve(b);

    EXPECT_LT((Eigen::MatrixXd(A) * x - b).cwiseAbs().maxCoeff(), 1e-12);
}

} // namespace

#endif // USE_ACCELERATE_SPARSE
```

- [ ] **Step 2: Register the test in the light target**

In `tests/cpp/CMakeLists.txt`, inside `set(TYCHO_TEST_LIGHT_SOURCES ...)`, immediately after the line `solvers/test_kkt_canonical_lock.cpp` (currently line 55), add:

```cmake
    solvers/test_accelerate_interface.cpp
```

- [ ] **Step 3: Configure and build the light target only**

```bash
eval "$(conda shell.bash hook)" && conda activate tycho
cmake --preset macos-llvm-release -DBUILD_CPP_TESTS=ON
cd build && ninja -j4 tycho_tests_light
```

Expected: compiles and links. If it fails to find `Accelerate` symbols, STOP — the leaf-TU premise is broken and the test must move to `TYCHO_TEST_HEAVY_SOURCES` instead (see spec §8).

- [ ] **Step 4: Run the new test**

```bash
cd build && ctest -R AccelerateInterface --output-on-failure
```

Expected: `1 test from 1 test suite ran. [  PASSED  ] 1 test.`

- [ ] **Step 5: Commit**

```bash
git add tests/cpp/solvers/test_accelerate_interface.cpp tests/cpp/CMakeLists.txt
git commit -m "test(solvers): Accelerate interface test scaffold

First test coverage for the Apple Accelerate sparse-solver interface.
Leaf-header TU in the light target: ~200 MB and seconds to compile,
versus 4-7 GB for the heavy target that pulls tycho.h.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: I1 — no indeterminate state at construction or after `release()`

**Files:**
- Modify: `include/tycho/detail/solvers/linear/accelerate_interface.h` (`:161`, `:241-275`, `:279`, `:595`, `:864-892`)
- Test: `tests/cpp/solvers/test_accelerate_interface.cpp`

**Interfaces:**
- Consumes: Task 1's fixtures.
- Produces: `void resetInertia()` — new **private** member of `AccelerateImpl`, zeroes `peigs_`/`neigs_`/`zeigs_`. Task 4 adds more call sites.

- [ ] **Step 1: Write the failing tests**

Add inside the anonymous namespace, after `TEST(AccelerateInterface, SolvesAnSpdSystem)`:

```cpp
TEST(AccelerateInterface, DefaultConstructedDimsAreZero) {
    LDLT s;
    EXPECT_EQ(s.rows(), 0);
    EXPECT_EQ(s.cols(), 0);
}

#ifdef NDEBUG
// info() guards with eigen_assert(m_isInitialized), which only vanishes under
// NDEBUG -- so this test is Release-only by construction. Pre-fix, info_ is
// uninitialized and this reads indeterminate memory (it may pass by luck; it
// is a regression guard, not a demonstration).
TEST(AccelerateInterface, DefaultConstructedInfoIsInitialized) {
    LDLT s;
    EXPECT_EQ(s.info(), Eigen::Success);
}
#endif

TEST(AccelerateInterface, ReleaseRestoresDefaultConstructedState) {
    LDLT s;
    s.compute(spd_both_triangles());
    ASSERT_EQ(s.info(), Eigen::Success);
    ASSERT_EQ(s.rows(), 3);

    s.release();

    EXPECT_EQ(s.rows(), 0);
    EXPECT_EQ(s.cols(), 0);
    EXPECT_EQ(s.peigs(), 0);
    EXPECT_EQ(s.neigs(), 0);
    EXPECT_EQ(s.zeigs(), 0);

    // ...and the solver is still reusable afterwards.
    s.compute(spd_both_triangles());
    EXPECT_EQ(s.info(), Eigen::Success);
}
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: `ReleaseRestoresDefaultConstructedState` FAILS with `rows()` returning 3 (not 0) and `peigs()` returning 3 (not 0). The other two should pass already (P4 fixed dims; `info_` reads 0 by luck).

- [ ] **Step 3: Initialize `info_` and move constructor defaults in-class**

At `:595`, change:
```cpp
    mutable ComputationInfo info_;
```
to:
```cpp
    mutable ComputationInfo info_ = Success;
```

In the member block, give the three simple settings in-class initializers (replacing the constructor's body assignments):
```cpp
    SparseOrder_t order_ = SparseOrderMetis;
    bool do_iterative_refinement_ = false;
    int iterative_refinement_iterations_ = 2;
```

In the constructor (`:241-275`), delete these three now-redundant lines, keeping only the `UpLo_` dispatch that assigns `m_sparseKind`/`m_triType`:
```cpp
        order_ = SparseOrderMetis;
        do_iterative_refinement_ = false;
        iterative_refinement_iterations_ = 2;
```

- [ ] **Step 4: Default the destructor and drop the deleter's dead store**

At `:279`, change `~AccelerateImpl() {}` to:
```cpp
    // Defaulted, not removed: a user-declared destructor suppresses the
    // implicit move constructor, and this type MUST stay non-movable --
    // accel_matrix_ caches raw pointers into matrix_'s buffers, which a move
    // would silently invalidate.
    ~AccelerateImpl() = default;
```

At `:161`, in `AccelFactorizationDeleter::operator()`, delete the dead store to the local parameter:
```cpp
    void operator()(T *sym) {
        if (sym) {
            SparseCleanup(*sym);
            delete sym;
        }
    }
```

- [ ] **Step 5: Add `resetInertia()` and restore the `release()` invariant**

Add to the `private:` section, next to `cacheInertia()` (`:411`):
```cpp
    void resetInertia() {
        peigs_ = 0;
        neigs_ = 0;
        zeigs_ = 0;
    }
```

In `release()` (`:864-892`), after `permutation_.clear();`, add:
```cpp
    // Restore the default-constructed invariant. rows()/cols() must not report
    // dimensions for an emptied solver, and accel_matrix_ must not retain
    // pointers into matrix_'s just-freed buffers.
    n_rows_ = 0;
    n_cols_ = 0;
    accel_matrix_ = AccelSparseMatrix{};
    m_columnStarts.clear();
    resetInertia();
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Format and commit**

```bash
/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h
git add include/tycho/detail/solvers/linear/accelerate_interface.h tests/cpp/solvers/test_accelerate_interface.cpp
git commit -m "fix(solvers): Accelerate — no indeterminate state at construction or release (I1)

info_ was uninitialized and never assigned by the constructor; info()'s
eigen_assert guard vanishes under NDEBUG, so a release-build caller read
indeterminate memory. Same bug class as PR 88's P4, one member below it.

release() left n_rows_/n_cols_ at their old values (undoing P4 for an
emptied solver), left inertia stale, and left accel_matrix_ pointing into
matrix_'s freed buffers. It now restores the default-constructed state.

Also: three constructor body assignments become in-class initializers
(that style is how info_ was missed), the destructor is defaulted with a
comment recording why the type must stay non-movable, and a dead store in
AccelFactorizationDeleter is dropped.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: I2 — diagnostics do not leak across translation units

Also carries the `:332` deprecation fix, which the spec files under I5. Reassigned deliberately: both are compile-time diagnostic concerns sharing one verification step (a warning-free probe compile), so they form one reviewer gate.

**Files:**
- Modify: `include/tycho/detail/solvers/linear/accelerate_interface.h` (`:332`, `:896`)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Reproduce the leaked-suppression and deprecation warnings**

Write `/tmp/accel_warn_probe.cpp` in the scratchpad:

```cpp
// Does accelerate_interface.h leave warnings suppressed for the rest of the TU,
// and does it emit its own deprecation warning?
#include "tycho/detail/solvers/linear/accelerate_interface.h"

#include <Eigen/Sparse>

// If the Disable/Reenable pragma pair were balanced, this would warn under
// -Wimplicit-int-float-conversion. If suppression leaked, it stays silent.
double leaked_suppression_canary(int i) { return i; }

int main() {
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    Eigen::AccelerateLDLTTPP<SpMat, Eigen::Upper> s;
    SpMat A(2, 2);
    A.insert(0, 0) = 2.0;
    A.insert(0, 1) = -1.0;
    A.insert(1, 1) = 2.0;
    A.makeCompressed();
    s.compute(A);
    return s.info() == Eigen::Success ? 0 : 1;
}
```

Run:
```bash
/opt/homebrew/opt/llvm/bin/clang++ -std=c++20 -O2 -DNDEBUG -DFMT_HEADER_ONLY \
  -Wimplicit-int-float-conversion -I include -I dep/eigen -I dep/fmt/include \
  -c /tmp/accel_warn_probe.cpp -o /tmp/accel_warn_probe.o
```

Expected before the fix: **no** warning for `leaked_suppression_canary` (proving suppression leaked past the header), and **two** `-Wdeprecated-anon-enum-enum-conversion` warnings pointing at `accelerate_interface.h:332`.

- [ ] **Step 2: Balance the warning pragma**

At the very end of `accelerate_interface.h`, between `} // end namespace Eigen` and the final `#endif`, add:

```cpp
} // end namespace Eigen

#include <Eigen/src/Core/util/ReenableStupidWarnings.h>

#endif // EIGEN_ACCELERATESUPPORT_H
```

- [ ] **Step 3: Fix the deprecated enum-mixing at `:332`**

`get_matrix`'s symmetric overload is `template <int U = UpLo>`, but its body uses the anonymous enum member `UpLo` in a bitwise op with `Eigen::UpLoType`. Its sibling at `:356` already uses `U`. Change `:332` from:
```cpp
        constexpr int TriangleType = (UpLo & Lower) ? Lower : Upper;
```
to:
```cpp
        constexpr int TriangleType = (U & Lower) ? Lower : Upper;
```

- [ ] **Step 4: Re-run the probe to verify both are fixed**

```bash
/opt/homebrew/opt/llvm/bin/clang++ -std=c++20 -O2 -DNDEBUG -DFMT_HEADER_ONLY \
  -Wimplicit-int-float-conversion -I include -I dep/eigen -I dep/fmt/include \
  -c /tmp/accel_warn_probe.cpp -o /tmp/accel_warn_probe.o
```

Expected after: **one** warning, on `leaked_suppression_canary` (suppression no longer leaks), and **zero** warnings from `accelerate_interface.h`.

- [ ] **Step 5: Rebuild the light target and check for newly-exposed warnings**

```bash
cd build && ninja -j4 tycho_tests_light 2>&1 | grep -i warning | head -20
```

Any warnings that appear here were previously masked by the leaked suppression. Triage each: fix it if it is in a file this PR already touches, otherwise record it in the PR body as follow-up. Do **not** silence it by reverting this change.

- [ ] **Step 6: Run tests**

```bash
cd build && ctest -R AccelerateInterface --output-on-failure
```

Expected: all PASS (no behavior change).

- [ ] **Step 7: Format and commit**

```bash
/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h
git add include/tycho/detail/solvers/linear/accelerate_interface.h
git commit -m "fix(solvers): Accelerate — balance Eigen warning pragma, fix deprecated enum mix (I2)

accelerate_interface.h included DisableStupidWarnings.h (which issues
#pragma clang diagnostic push) but never the matching Reenable, so
-Wconstant-logical-operand and -Wimplicit-int-float-conversion stayed
suppressed for the remainder of every TU that includes it -- which is
most of the Accelerate-enabled TUs plus the PCH, via psiopt.h,
solver_context.h and tycho_solvers.h. pardiso_interface.h balances it
correctly. Notably the leaked suppression covers the warning class that
would have flagged PR 88's P3/P4 narrowing bugs.

Also fixes the file's own -Wdeprecated-anon-enum-enum-conversion at :332,
where get_matrix used the anonymous enum member UpLo instead of its int
template parameter U; the sibling at :356 already used U.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: I3 — cached derived state is never stale

**Files:**
- Modify: `include/tycho/detail/solvers/linear/accelerate_interface.h` (`:428-434`, `:544`, `:567`, `:617-636`, `:783-805`)
- Test: `tests/cpp/solvers/test_accelerate_interface.cpp`

**Interfaces:**
- Consumes: `resetInertia()` from Task 2; Task 1's fixtures.
- Produces: nothing new.

- [ ] **Step 1: Probe to pin the expected inertia values**

Do not guess what Accelerate reports. Write `/tmp/inertia_probe.cpp`:

```cpp
#include "tycho/detail/solvers/linear/accelerate_interface.h"

#include <Eigen/Sparse>
#include <cstdio>

int main() {
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    SpMat A(3, 3);
    A.insert(0, 0) = 1.0;
    A.insert(1, 1) = 1.0;
    A.insert(2, 2) = -1.0;
    A.makeCompressed();

    Eigen::AccelerateLDLTTPP<SpMat, Eigen::Upper> s;
    s.compute(A);
    std::printf("info=%d peigs=%lld neigs=%lld zeigs=%lld\n", (int)s.info(),
                (long long)s.peigs(), (long long)s.neigs(), (long long)s.zeigs());
    return 0;
}
```

Compile and run it. Expected `info=0 peigs=2 neigs=1 zeigs=0` for `diag(1, 1, -1)`. **Use whatever it actually prints** in Step 2; if it differs, the matrix or expectation needs adjusting before writing the test.

- [ ] **Step 2: Write the failing tests**

```cpp
TEST(AccelerateInterface, InertiaOnKnownIndefiniteMatrix) {
    SpMat A(3, 3);
    A.insert(0, 0) = 1.0;
    A.insert(1, 1) = 1.0;
    A.insert(2, 2) = -1.0;
    A.makeCompressed();

    LDLT s;
    s.compute(A);
    ASSERT_EQ(s.info(), Eigen::Success);
    EXPECT_EQ(s.peigs(), 2);
    EXPECT_EQ(s.neigs(), 1);
    EXPECT_EQ(s.zeigs(), 0);
}

// PSIOPT's inertia-correction loop reads neigs()/peigs() right after
// Compute()/Refactor() without consulting info(), so cached inertia must never
// outlive the factorization it describes.
TEST(AccelerateInterface, SetMatrixClearsStaleInertia) {
    SpMat indefinite(3, 3);
    indefinite.insert(0, 0) = 1.0;
    indefinite.insert(1, 1) = 1.0;
    indefinite.insert(2, 2) = -1.0;
    indefinite.makeCompressed();

    LDLT s;
    s.compute(indefinite);
    ASSERT_EQ(s.info(), Eigen::Success);
    ASSERT_GT(s.peigs() + s.neigs(), 0);

    s.set_matrix(spd_both_triangles());

    EXPECT_EQ(s.peigs(), 0);
    EXPECT_EQ(s.neigs(), 0);
    EXPECT_EQ(s.zeigs(), 0);
}

TEST(AccelerateInterface, ReinitializeClearsStaleInertia) {
    SpMat indefinite(3, 3);
    indefinite.insert(0, 0) = 1.0;
    indefinite.insert(1, 1) = 1.0;
    indefinite.insert(2, 2) = -1.0;
    indefinite.makeCompressed();

    LDLT s;
    s.compute(indefinite);
    ASSERT_GT(s.peigs() + s.neigs(), 0);

    s.reinitialize_internal_matrix_representation();

    EXPECT_EQ(s.peigs(), 0);
    EXPECT_EQ(s.neigs(), 0);
    EXPECT_EQ(s.zeigs(), 0);
}
```

- [ ] **Step 3: Run to verify the two clearing tests fail**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: `InertiaOnKnownIndefiniteMatrix` PASSES; `SetMatrixClearsStaleInertia` and `ReinitializeClearsStaleInertia` FAIL with `peigs()` still 2.

- [ ] **Step 4: Reset inertia and metrics on every invalidating path**

In `doFactorization()` (`:544`), extend the failure branch:
```cpp
            if (status != SparseStatusOK) {
                m_numericFactorization.reset(nullptr);
                resetInertia();
                flops_ = 0;
                mem_ = 0;
            } else
                cacheInertia();
```

In `doRefactorization()` (`:567`), add the else branch:
```cpp
        if (status == SparseStatusOK) {
            cacheInertia();
            updatePerformanceMetrics();
        } else {
            resetInertia();
            flops_ = 0;
            mem_ = 0;
        }
```

In `set_matrix()` (`:617-636`), alongside the existing invalidation, add:
```cpp
    resetInertia();
```

In `reinitialize_internal_matrix_representation()` (`:783-805`), alongside the existing factorization resets, add:
```cpp
    resetInertia();
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: all PASS.

- [ ] **Step 6: Format and commit**

```bash
/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h
git add include/tycho/detail/solvers/linear/accelerate_interface.h tests/cpp/solvers/test_accelerate_interface.cpp
git commit -m "fix(solvers): Accelerate — inertia and metrics never outlive their factorization (I3)

cacheInertia() ran only on success, so peigs_/neigs_/zeigs_ kept the
PREVIOUS factorization's values after a failure, and set_matrix() /
reinitialize_internal_matrix_representation() reset info_ and both
factorizations while leaving inertia untouched.

This matters because PSIOPT's Inertia() and RankDef() read neigs()/peigs()
immediately after Compute()/Refactor() without consulting info() --
CheckInfo() is explicitly observational -- so stale inertia steers the
perturbation loop. cacheInertia() already zeroed on its own
SparseGetInertia-failure path; this extends the same policy to every
invalidating transition.

Stakes: probing shows LDLT^TPP reports SparseStatusOK even for an all-zero
matrix, and for NaN/Inf input (counting the NaN pivot as negative), so
inertia -- not status -- is this backend's real singularity signal.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: I4a — failure is reported, never silently substituted

The behavioral core of the PR. Fixes an active-corruption path, not just staleness.

**Files:**
- Modify: `include/tycho/detail/solvers/linear/accelerate_interface.h` (`:499-502`, `:693-775`)
- Test: `tests/cpp/solvers/test_accelerate_interface.cpp`

**Interfaces:**
- Consumes: Task 1's fixtures.
- Produces: nothing new. Task 6 changes the workspace helpers this function calls.

- [ ] **Step 1: Write the failing tests**

```cpp
// Drives doRefactorization() to failure the way PSIOPT does: mutate values in
// the internal matrix in place (pattern unchanged), then refactor.
void break_factorization_in_place(LLT &s) {
    s.get_matrix().coeffRef(0, 1) = 8.0;
    s.get_matrix().coeffRef(0, 2) = 8.0;
    s.refactorize_internal();
}

TEST(AccelerateInterface, FailedFactorizationIsReported) {
    LLT s;
    s.analyze_pattern(spd_full_upper());
    ASSERT_EQ(s.info(), Eigen::Success);

    s.factorize(indefinite_full_upper());

    EXPECT_NE(s.info(), Eigen::Success);
}

// Contract, matching PardisoImpl::_solve_impl: a solve on a bad factorization
// reports through info() and leaves the destination alone. Zeroing it instead
// is deferred to issue #105 (both backends together).
TEST(AccelerateInterface, SolveAfterFailedRefactorReportsAndLeavesDestination) {
    LLT s;
    s.compute(spd_full_upper());
    ASSERT_EQ(s.info(), Eigen::Success);

    break_factorization_in_place(s);
    ASSERT_NE(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    Eigen::VectorXd x(3);
    x.setConstant(-12345.0);

    x = s.solve(b);

    EXPECT_NE(s.info(), Eigen::Success);
    EXPECT_DOUBLE_EQ(x[0], -12345.0);
    EXPECT_DOUBLE_EQ(x[1], -12345.0);
    EXPECT_DOUBLE_EQ(x[2], -12345.0);
}

// Pre-fix this is worse than stale: the main SparseSolve no-ops, then the
// refinement loop computes r = A*x - b (SparseMultiplyAdd takes the MATRIX, so
// it works fine), the inner SparseSolve also no-ops leaving r as the raw
// residual rather than a correction, and x -= r actively corrupts x -- once per
// refinement iteration.
TEST(AccelerateInterface, SolveAfterFailedRefactorDoesNotCorruptUnderRefinement) {
    LLT s;
    s.set_iterative_refinement(true);
    s.set_iterative_refinement_iterations(2);
    s.compute(spd_full_upper());
    ASSERT_EQ(s.info(), Eigen::Success);

    break_factorization_in_place(s);
    ASSERT_NE(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    Eigen::VectorXd x(3);
    x.setConstant(-12345.0);

    x = s.solve(b);

    EXPECT_NE(s.info(), Eigen::Success);
    EXPECT_DOUBLE_EQ(x[0], -12345.0);
    EXPECT_DOUBLE_EQ(x[1], -12345.0);
    EXPECT_DOUBLE_EQ(x[2], -12345.0);
}

// The reason doRefactorization deliberately does NOT reset the numeric
// factorization on failure: Apple documents the failed object as refactorable,
// and PSIOPT's perturb-and-retry loop depends on that cheap recovery.
TEST(AccelerateInterface, RefactorRecoversAfterAFailedRefactor) {
    LLT s;
    const SpMat A = spd_full_upper();
    s.compute(A);
    ASSERT_EQ(s.info(), Eigen::Success);

    const double saved01 = s.get_matrix().coeff(0, 1);
    const double saved02 = s.get_matrix().coeff(0, 2);

    break_factorization_in_place(s);
    ASSERT_NE(s.info(), Eigen::Success);

    s.get_matrix().coeffRef(0, 1) = saved01;
    s.get_matrix().coeffRef(0, 2) = saved02;
    s.refactorize_internal();
    ASSERT_EQ(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    const Eigen::VectorXd x = s.solve(b);
    EXPECT_LT((dense_symmetric(A) * x - b).cwiseAbs().maxCoeff(), 1e-10);
}
```

- [ ] **Step 2: Run to verify the failure tests fail**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: `SolveAfterFailedRefactorDoesNotCorruptUnderRefinement` FAILS (x corrupted, no longer -12345). The others may already pass — `SolveAfterFailedRefactorReportsAndLeavesDestination` passes only accidentally, because Accelerate's own parameter check no-ops the solve; the guard makes it intentional. Note both failing solves print a red Accelerate diagnostic to stderr; that is expected.

- [ ] **Step 3: Add the status guard to `_solve_impl`**

Replace the existing guard at the top of `_solve_impl` (`:697-700`):
```cpp
    if (!m_numericFactorization) {
        info_ = InvalidInput;
        return;
    }
```
with:
```cpp
    // A factorization that is absent or not in a good state cannot be solved
    // with. Accelerate would accept the call, no-op it via its own parameter
    // check, and return -- leaving x holding whatever it held before, which in
    // PSIOPT is the previous iteration's step. Under iterative refinement it is
    // worse: the refinement loop would then apply x -= (A*x - b) per iteration.
    //
    // x is deliberately left untouched rather than zeroed, matching
    // PardisoImpl::_solve_impl. Unifying both backends on a zero-x contract is
    // tracked in issue #105 -- do not "fix" this in isolation.
    if (!m_numericFactorization || m_numericFactorization->status != SparseStatusOK) {
        if (m_numericFactorization)
            updateInfoStatus(m_numericFactorization->status);
        else if (info_ == Success)
            // Never computed. A failed factorize() already recorded a more
            // specific status, so do not overwrite it.
            info_ = InvalidInput;
        return;
    }
```

- [ ] **Step 4: Make the `reportError` label phase-neutral**

Accelerate stores this callback in the symbolic factor and reuses it for numeric, refactor **and solve-time** diagnostics, so the current label misattributes every later failure. At `:499-502`, change the message from `"Accelerate Sparse Symbolic Factorization Error: {}\n"` to:

```cpp
        fopts.reportError = [](const char *msg) {
            // Accelerate stores this callback in the symbolic factorization and
            // reuses it for numeric-factor, refactor and solve-time errors too,
            // so the label must not name a phase.
            fmt::print(fmt::fg(fmt::color::red), "Accelerate Sparse Solver Error: {}\n", msg);
        };
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: all PASS.

- [ ] **Step 6: Format and commit**

```bash
/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h
git add include/tycho/detail/solvers/linear/accelerate_interface.h tests/cpp/solvers/test_accelerate_interface.cpp
git commit -m "fix(solvers): Accelerate — guard solves against a failed factorization (I4)

_solve_impl checked only that its unique_ptr was non-null, not that the
factorization was usable. doRefactorization leaves the pointer non-null
with a bad status on failure, so SparseSolve was called on it; Accelerate
parameter-checks the factorization, prints through the reportError
callback, and returns WITHOUT writing x. In PSIOPT that means DXSL --
declared once and reused every iteration -- keeps the previous step, which
is then negated.

Worse under iterative refinement (qp_ref_steps_ > 0, off by default): the
refinement loop still runs, SparseMultiplyAdd takes the matrix so the
residual computes normally, the inner in-place SparseSolve also no-ops
leaving the raw residual in place of a correction, and x -= residual then
actively corrupts x once per iteration. The early return removes this.

x is left untouched rather than zeroed, matching PardisoImpl; the
null-pointer branch preserves an existing non-Success classification
instead of overwriting NumericalIssue with the vaguer InvalidInput.
Unifying both backends on a zero-x contract is issue #105.

Also relabels the reportError diagnostic, which named the symbolic phase
but is reused by Accelerate for numeric, refactor and solve errors.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: I4b — workspace sizing and alignment

**Files:**
- Modify: `include/tycho/detail/solvers/linear/accelerate_interface.h` (`:193-209`, `:404-409`, `:430`, `:561`, `:600`, `:723-734`)
- Test: `tests/cpp/solvers/test_accelerate_interface.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: `Eigen::internal::aligned_subbuffer(std::vector<uint8_t> &storage)` → `void *`; returns a 16-byte-aligned pointer into `storage`, or `nullptr` if `storage` is too small to contain one.

- [ ] **Step 1: Write the failing test**

```cpp
// The alignment arithmetic lives in Eigen::internal so it can be tested
// directly: its only in-class caller is private, and the size==0 path is
// unreachable through the public API.
TEST(AccelerateInterface, AlignedSubbufferRejectsUndersizedStorage) {
    std::vector<uint8_t> empty(0), small(8), exact(16), ample(64);

    EXPECT_EQ(Eigen::internal::aligned_subbuffer(empty), nullptr);
    EXPECT_EQ(Eigen::internal::aligned_subbuffer(small), nullptr);

    void *p_exact = Eigen::internal::aligned_subbuffer(exact);
    void *p_ample = Eigen::internal::aligned_subbuffer(ample);
    ASSERT_NE(p_exact, nullptr);
    ASSERT_NE(p_ample, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p_exact) % 16u, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p_ample) % 16u, 0u);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd build && ninja -j4 tycho_tests_light 2>&1 | tail -5
```

Expected: FAILS TO COMPILE with `no member named 'aligned_subbuffer' in namespace 'Eigen::internal'`.

- [ ] **Step 3: Add the helper**

In `namespace internal`, immediately after `resizeForAccelerateAlignment` (`:209`):

```cpp
// Returns a 16-byte-aligned pointer into an ALREADY-SIZED buffer, or nullptr if
// the buffer cannot contain one. Callers must treat nullptr as a failure:
// Accelerate's workspace parameters are _Nonnull.
//
// Split out of AccelerateImpl::getAlignedPointer, whose `space - kAlign`
// underflowed for buffers smaller than the alignment, making std::align return
// nullptr behind an assert that NDEBUG deletes.
inline void *aligned_subbuffer(std::vector<uint8_t> &storage) {
    constexpr size_t kAccelerateRequiredAlignment = 16;
    if (storage.size() < kAccelerateRequiredAlignment)
        return nullptr;
    void *ptr = static_cast<void *>(storage.data());
    size_t space = storage.size();
    return std::align(kAccelerateRequiredAlignment,
                      storage.size() - kAccelerateRequiredAlignment, ptr, space);
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: all PASS.

- [ ] **Step 5: Route the private helper through it and make the guards real**

Replace `getAlignedPointer` (`:404-409`) with:
```cpp
    void *getAlignedPointer(std::vector<uint8_t> &storage) const {
        return internal::aligned_subbuffer(storage);
    }
```

In `doRefactorization()` (`:561`), which has no guard at all today:
```cpp
        void *ws = getAlignedPointer(workspace_);
        if (!ws) {
            // Accelerate's workspace parameter is _Nonnull; a null here would
            // be a dereference, not a diagnosable failure.
            updateInfoStatus(SparseInternalError);
            return;
        }
        SparseRefactor(accel_matrix_, m_numericFactorization.get(), ws);
```

In `_solve_impl` (`:734`), replace the `assert` — which `NDEBUG` deletes — with a real check:
```cpp
    if (!ws) {
        info_ = InvalidInput;
        return;
    }
```

- [ ] **Step 6: Widen the workspace size types**

At `:600`, change `mutable int cached_solve_workspace_size_ = 0;` to:
```cpp
    // Allocated size, not requested size: the buffer is a high-water mark.
    mutable size_t cached_solve_workspace_size_ = 0;
```

In `_solve_impl` (`:723-730`), the SDK declares both fields as `size_t` (`Solve.h:1557-1558`):
```cpp
    const size_t workspaceSize =
        m_numericFactorization->solveWorkspaceRequiredStatic +
        static_cast<size_t>(nrhs) * m_numericFactorization->solveWorkspaceRequiredPerRHS;

    // Grow only: a shrinking nrhs must not trigger a resize round-trip.
    void *ws;
    if (workspaceSize > cached_solve_workspace_size_) {
        ws = internal::resizeForAccelerateAlignment(workspaceSize, &solve_workspace_);
        cached_solve_workspace_size_ = workspaceSize;
    } else {
        ws = getAlignedPointer(solve_workspace_);
    }
```

- [ ] **Step 7: Saturate the `mem_` conversion**

At `:430`, `factorSize_Double` is a `size_t` byte count assigned into an `int`. `result_.factor_mem_` is `int` (`psiopt.h:413`) and is display-only, so saturate rather than widen:
```cpp
    void updatePerformanceMetrics() {
        if (m_symbolicFactorization) {
            // NOTE: this is the factor size in BYTES, whereas Pardiso's mem_ is
            // the NUMBER OF NONZEROS in the factor (iparm_[17]). Both surface as
            // result_.factor_mem_, so the reported number means different things
            // per platform. Unifying is tracked in issue #105.
            const size_t bytes = std::is_same<Scalar, double>::value
                                     ? m_symbolicFactorization->factorSize_Double
                                     : m_symbolicFactorization->factorSize_Float;
            mem_ = static_cast<int>(
                std::min<size_t>(bytes, static_cast<size_t>(std::numeric_limits<int>::max())));
        }
        flops_ = 0; // Accelerate exposes no FLOP count; psiopt guards its print with > 0.
    }
```

Add `#include <algorithm>` to the header's include block if not already present.

- [ ] **Step 8: Run tests**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: all PASS.

- [ ] **Step 9: Format and commit**

```bash
/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h
git add include/tycho/detail/solvers/linear/accelerate_interface.h tests/cpp/solvers/test_accelerate_interface.cpp
git commit -m "fix(solvers): Accelerate — workspace sizing and alignment (I4)

getAlignedPointer computed \`space - kAlign\` on a size_t, which wrapped for
buffers smaller than the alignment and made std::align return nullptr --
behind a raw assert that NDEBUG deletes, into an Accelerate parameter
declared _Nonnull. The arithmetic moves to Eigen::internal::aligned_subbuffer
(directly unit-testable; the size==0 path is unreachable through the public
API) and both call sites get real checks. doRefactorization had no guard at
all.

_solve_impl's workspaceSize and cached_solve_workspace_size_ widen int ->
size_t, matching the SDK's solveWorkspaceRequiredStatic/PerRHS, and the
cache comparison becomes grow-only so a shrinking nrhs no longer triggers a
resize round-trip. This completes PR 88's recorded P3 follow-up.

mem_'s size_t -> int assignment is saturated: Accelerate reports the factor
size in bytes, so >2 GB factors are reachable and the truncation was a
plausible display bug at scale.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: I5 — preconditions are compile-time or asserted, not assumed

**Files:**
- Modify: `include/tycho/detail/solvers/linear/accelerate_interface.h` (`:224-239`, `:450-451`, `:469-470`, `:486`, `:702-703`, `:783-805`)
- Test: `tests/cpp/solvers/test_accelerate_interface.cpp`

**Interfaces:**
- Consumes: Task 1's fixtures.
- Produces: nothing new.

- [ ] **Step 1: Probe whether the aliasing guard is warranted**

The spec leaves this open: an aliased solve currently returns correct results for a 3×3, so the guard may be unnecessary. Decide with evidence. Write `/tmp/alias_probe.cpp`:

```cpp
#include "tycho/detail/solvers/linear/accelerate_interface.h"

#include <Eigen/Sparse>
#include <cstdio>

int main() {
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    // Tridiagonal SPD, n large enough that any internal buffering would show.
    const int n = 200;
    SpMat A(n, n);
    for (int i = 0; i < n; ++i) {
        A.insert(i, i) = 4.0;
        if (i + 1 < n)
            A.insert(i, i + 1) = -1.0;
    }
    A.makeCompressed();

    Eigen::AccelerateLDLTTPP<SpMat, Eigen::Upper> s;
    s.compute(A);

    Eigen::MatrixXd full = Eigen::MatrixXd(A);
    full.triangularView<Eigen::Lower>() = full.transpose();

    for (int nrhs : {1, 3}) {
        Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, nrhs);
        Eigen::MatrixXd X_ref = s.solve(B);           // distinct buffers
        Eigen::MatrixXd XB = B;
        XB = s.solve(XB);                             // aliased
        std::printf("nrhs=%d aliased-vs-reference diff = %.3e\n", nrhs,
                    (XB - X_ref).cwiseAbs().maxCoeff());
    }
    return 0;
}
```

Compile and run. **Decision rule:** if both diffs are at solver tolerance (< 1e-10), aliasing is safe in practice — implement Step 4 as a documenting comment only and skip the temp-copy. If either diff is large, implement the temp-copy guard. Record which branch you took in the commit message.

- [ ] **Step 2: Write the failing test for the compressed-matrix precondition**

```cpp
// PSIOPT assembles the KKT matrix in place through get_matrix() and then calls
// reinitialize_internal_matrix_representation(). buildAccelSparseMatrix reads
// outerIndexPtr/innerIndexPtr/valuePtr raw, which describe nothing coherent
// unless the matrix is compressed -- safe in-tree today only because
// analyze_sparsity happens to end with makeCompressed().
TEST(AccelerateInterface, ReinitializeToleratesUncompressedMatrix) {
    LDLT s;
    s.compute(spd_both_triangles());
    ASSERT_EQ(s.info(), Eigen::Success);

    // Force the internal matrix into uncompressed mode, as an in-place
    // assembler that inserts a new entry would.
    s.get_matrix().uncompress();
    ASSERT_FALSE(s.get_matrix().isCompressed());

    s.reinitialize_internal_matrix_representation();
    EXPECT_TRUE(s.get_matrix().isCompressed());

    s.compute_internal();
    ASSERT_EQ(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    const Eigen::VectorXd x = s.solve(b);
    EXPECT_LT((Eigen::MatrixXd(spd_both_triangles()) * x - b).cwiseAbs().maxCoeff(), 1e-10);
}
```

- [ ] **Step 3: Run to verify it fails**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: FAILS — either on `EXPECT_TRUE(isCompressed())`, or on the residual, or by crashing. A crash still counts as red; note which occurred.

- [ ] **Step 4: Add the preconditions**

**(a)** After the typedef block (`:239`), add:
```cpp
    // Accelerate's SparseMatrixStructure::rowIndices and
    // SparseSymbolicFactorOptions::order are both int*, and this class hands
    // matrix_.innerIndexPtr() and permutation_.data() to them directly. A
    // different StorageIndex is already ill-formed (const_cast cannot change a
    // pointee type); this turns that into one readable message.
    static_assert(std::is_same<StorageIndex, int>::value,
                  "AccelerateImpl requires StorageIndex == int: Accelerate's sparse "
                  "structure and ordering fields are int*.");
```

**(b)** In `reinitialize_internal_matrix_representation()` (`:783-805`), before `buildAccelSparseMatrix()`:
```cpp
    // buildAccelSparseMatrix reads outerIndexPtr/innerIndexPtr/valuePtr raw,
    // which only describe a valid CSR/CSC layout when the matrix is compressed.
    // No-op when it already is.
    matrix_.makeCompressed();
```

**(c)** In `buildAccelSparseMatrix()`, guard the `Index` → `int` narrowing. Before the `if constexpr` at `:443`:
```cpp
        eigen_assert(matrix_.rows() <= std::numeric_limits<int>::max() &&
                     matrix_.cols() <= std::numeric_limits<int>::max() &&
                     "Accelerate's sparse structure uses int dimensions");
```

**(d)** In `doAnalysis()` (`:486`), size the permutation for the largest dimension:
```cpp
        // fopts.order is a symmetric row-and-column permutation for the
        // symmetric and QR cases (Solve.h:1310-1319), so size it for the larger
        // dimension. No-op for square matrices, which is every type
        // instantiated in-tree -- this hardens AccelerateQR/AccelerateCholeskyAtA,
        // which are declared EnforceSquare_ = false but never instantiated.
        const Index perm_size = std::max(n_rows_, n_cols_);
        if (permutation_.size() != static_cast<size_t>(perm_size)) {
            permutation_.resize(perm_size);
        }
```

**(e)** In `_solve_impl`, after the existing asserts (`:702-703`), add Pardiso's row-major asserts (`pardiso_interface.h:377-380`). These are `eigen_assert`s, so like Pardiso's they protect debug builds only:
```cpp
    eigen_assert(((MatrixBase<Rhs>::Flags & RowMajorBit) == 0 || b.cols() == 1) &&
                 "Row-major right hand sides are not supported");
    eigen_assert(((MatrixBase<Dest>::Flags & RowMajorBit) == 0 || x.cols() == 1) &&
                 "Row-major matrices of unknowns are not supported");
```

**(f)** Aliasing — per Step 1's decision. If the probe said "safe", add only a comment above the `SparseSolve` call:
```cpp
    // b and x may alias (Eigen's Solve assignment forwards both straight here).
    // Accelerate does not document the out-of-place overload as alias-safe, but
    // probing at n=200 with nrhs in {1,3} showed correct results; revisit if a
    // caller ever depends on it. Pardiso, by contrast, must copy (see
    // PardisoImpl::_solve_impl).
```
If the probe said "unsafe", instead copy `b` before solving, mirroring Pardiso:
```cpp
    Matrix<Scalar, Dynamic, Dynamic, ColMajor> b_tmp;
    if (b_ptr == x_ptr) {
        b_tmp = b;
        b_ptr = b_tmp.data();
        bmat.data = b_ptr;
    }
```
placed after `bmat` is populated. Do **not** use Accelerate's in-place overload: it destroys `b`, which the refinement block still needs as its residual reference.

Add `#include <limits>` and `#include <type_traits>` to the include block if not already present.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: all PASS.

- [ ] **Step 6: Format and commit**

```bash
/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h
git add include/tycho/detail/solvers/linear/accelerate_interface.h tests/cpp/solvers/test_accelerate_interface.cpp
git commit -m "fix(solvers): Accelerate — enforce the interface's implicit preconditions (I5)

- static_assert StorageIndex == int. Already ill-formed otherwise, but the
  diagnostic was a template error cascade.
- reinitialize_internal_matrix_representation() now calls makeCompressed().
  buildAccelSparseMatrix reads the index/value pointers raw, which is only
  valid for a compressed matrix; this is the entry point PSIOPT uses for
  in-place KKT assembly, and it was safe only because analyze_sparsity
  happens to end with makeCompressed().
- assert that matrix dimensions fit int before buildAccelSparseMatrix
  narrows them -- the same width class as PR 88's P3, at the 2^31 boundary.
- doAnalysis sizes permutation_ by max(rows, cols): fopts.order is a
  symmetric row-and-column permutation, so a wide matrix could have
  Accelerate write past the buffer. Hardens AccelerateQR/CholeskyAtA, which
  are declared EnforceSquare_ = false but never instantiated in-tree.
- add Pardiso's row-major RHS asserts; _solve_impl hardcodes
  columnStride = rowCount and would otherwise be silently wrong. Debug-only,
  like Pardiso's.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: I6 — platform capability checked at runtime, not at build time

**Files:**
- Modify: `include/tycho/detail/solvers/linear/accelerate_utils.h` (`:42-78`, `:97-106`)
- Modify: `include/tycho/detail/solvers/linear/accelerate_interface.h` (`:301`)
- Test: `tests/cpp/solvers/test_accelerate_interface.cpp`

**Interfaces:**
- Consumes: Task 1's fixtures.
- Produces: `accelerate_supported_order(SparseOrder_t)` → `SparseOrder_t` (free function in `accelerate_utils.h`); `AccelerateImpl::effective_order()` → `SparseOrder_t`.

- [ ] **Step 1: Write the failing test**

```cpp
#ifdef TYCHO_HAS_MTMETIS
// MT-METIS is macOS 26+. The SDK-version macro only says the enum exists, not
// that the host supports it, so set_order downgrades at runtime. A silent
// downgrade of an explicit request must be observable.
TEST(AccelerateInterface, MtMetisOrderIsDowngradedWhenUnavailable) {
    LDLT s;
    s.set_order(SparseOrderMTMetis);

    if (__builtin_available(macOS 26.0, *)) {
        EXPECT_EQ(s.effective_order(), SparseOrderMTMetis);
    } else {
        EXPECT_EQ(s.effective_order(), SparseOrderMetis);
    }

    s.compute(spd_both_triangles());
    EXPECT_EQ(s.info(), Eigen::Success);
}
#endif

TEST(AccelerateInterface, SupportedOrdersPassThroughUnchanged) {
    LDLT s;
    s.set_order(SparseOrderAMD);
    EXPECT_EQ(s.effective_order(), SparseOrderAMD);
    s.set_order(SparseOrderMetis);
    EXPECT_EQ(s.effective_order(), SparseOrderMetis);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd build && ninja -j4 tycho_tests_light 2>&1 | tail -5
```

Expected: FAILS TO COMPILE with `no member named 'effective_order'`.

- [ ] **Step 3: Add the capability helper to `accelerate_utils.h`**

After the `TYCHO_HAS_MTMETIS` define block (`:37`):

```cpp
// Maps a requested ordering onto one this HOST supports.
//
// TYCHO_HAS_MTMETIS is an SDK-version macro: it says the enum constant exists
// at compile time, not that the running OS implements it. Passing
// SparseOrderMTMetis on macOS < 26 yields SparseParameterError and a dead
// solver, so downgrade at runtime instead.
inline SparseOrder_t accelerate_supported_order(SparseOrder_t order) {
#ifdef TYCHO_HAS_MTMETIS
    if (order == SparseOrderMTMetis && !__builtin_available(macOS 26.0, *))
        return SparseOrderMetis;
#endif
    return order;
}
```

- [ ] **Step 4: Route `set_order` through it and expose the effective order**

In `accelerate_interface.h` at `:301`:
```cpp
    /** Sets the ordering algorithm to use.
     *
     * An ordering the running OS does not support is downgraded (see
     * accelerate_supported_order); query effective_order() for what was kept.
     */
    void set_order(SparseOrder_t order) { order_ = accelerate_supported_order(order); }

    /** The ordering actually in use, after any runtime capability downgrade. */
    SparseOrder_t effective_order() const { return order_; }
```

- [ ] **Step 5: Add the runtime availability check to the thread setter**

In `accelerate_utils.h`, replace `accelerate_set_num_threads` (`:97-106`):
```cpp
inline void accelerate_set_num_threads(int num_threads) {
#ifdef TYCHO_HAS_BLAS_SET_THREADING
    // The #ifdef only guarantees the DECLARATION exists in this SDK.
    // BLASSetThreading is API_AVAILABLE(macos(15.0)) and therefore weak-linked,
    // so a binary built against a newer SDK with an older deployment target
    // would null-call it. The runtime check is the actual guard.
    if (__builtin_available(macOS 15.0, *)) {
        if (num_threads <= 1)
            BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        else
            BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
        return;
    }
#endif
    setenv("VECLIB_MAXIMUM_THREADS", std::to_string(num_threads).c_str(), 1);
}
```

- [ ] **Step 6: Fix the warmup leaks and trap risk**

Replace the body of `warmup_sparse_solver()` (`:42-78`) from the `SparseSymbolicFactorOptions` declaration onward:
```cpp
    SparseSymbolicFactorOptions fopts{};
    fopts.control = SparseDefaultControl;
    fopts.orderMethod = accelerate_supported_order(
#ifdef TYCHO_HAS_MTMETIS
        SparseOrderMTMetis
#else
        SparseOrderMetis
#endif
    );
    fopts.malloc = malloc;
    fopts.free = free;
    // Without a reportError callback, an Accelerate parameter-check failure
    // takes its null-callback branch: os_log_error followed by _SparseTrap(),
    // i.e. an abort on the startup path.
    fopts.reportError = [](const char *) {};

    auto sym = SparseFactor(SparseFactorizationLDLTTPP, structure, fopts);
    if (sym.status == SparseStatusOK) {
        auto num = SparseFactor(sym, A);
        // Apple requires SparseCleanup even for a FAILED factorization.
        SparseCleanup(num);
    }
    SparseCleanup(sym);
}
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
```

Expected: all PASS. On this macOS 26.5 host the MT-METIS test takes the pass-through branch; the downgrade branch is unreachable here and is inspection-only (spec §5 gap 4).

- [ ] **Step 8: Format and commit**

```bash
/opt/homebrew/opt/llvm/bin/clang-format -i include/tycho/detail/solvers/linear/accelerate_interface.h include/tycho/detail/solvers/linear/accelerate_utils.h
git add include/tycho/detail/solvers/linear/accelerate_interface.h include/tycho/detail/solvers/linear/accelerate_utils.h tests/cpp/solvers/test_accelerate_interface.cpp
git commit -m "fix(solvers): Accelerate — check platform capability at runtime, not build time (I6)

Both gates tested __MAC_OS_X_VERSION_MAX_ALLOWED, which is the SDK version
and says nothing about the running OS or the deployment target:

- BLASSetThreading is API_AVAILABLE(macos(15.0)) and weak-linked, so a
  wheel built on a newer SDK with a lower deployment target could
  null-call it. Now guarded by __builtin_available, falling back to
  VECLIB_MAXIMUM_THREADS.
- SparseOrderMTMetis is API_AVAILABLE(macos(26.0)); passing it on an older
  host gives SparseParameterError and a dead solver. set_order() now
  downgrades via accelerate_supported_order(), and effective_order()
  exposes the result so a silent downgrade of an explicit request is
  observable. Keeping the policy in the interface is what lets psiopt.cpp
  stay untouched -- it keeps passing its QPOrderingModes preference.

Also fixes warmup_sparse_solver(): SparseCleanup was skipped for a failed
symbolic or numeric factorization (Apple requires it either way), and no
reportError callback was set, so a parameter-check failure would take
Accelerate's null-callback branch into _SparseTrap() -- an abort on the
startup path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Documentation, full gate, and PR

**Files:**
- Modify: `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification.md`
- Create: `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification-RESULTS.md`

**Interfaces:**
- Consumes: everything.
- Produces: the merge-ready PR.

- [ ] **Step 1: Add the P1 both-triangle reuse test the handoff asked for**

The handoff's check 1 wants an analyze→factorize reuse probe; make it a permanent test instead:

```cpp
// Handoff check 1 / PR 88 P1: factorize() must apply the same triangle
// canonicalization analyze_pattern() did, or the numeric structure will not
// match the symbolic one for both-triangle symmetric input.
TEST(AccelerateInterface, BothTrianglePatternSurvivesAnalyzeThenFactorize) {
    const SpMat A1 = spd_both_triangles();
    SpMat A2 = A1;
    for (Eigen::Index k = 0; k < A2.nonZeros(); ++k)
        A2.valuePtr()[k] *= 1.5; // same pattern, different values, still symmetric

    LDLT s;
    s.analyze_pattern(A1);
    ASSERT_EQ(s.info(), Eigen::Success);
    s.factorize(A2);
    ASSERT_EQ(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    const Eigen::VectorXd x = s.solve(b);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ref;
    const Eigen::SparseMatrix<double> A2_colmajor = A2;
    ref.compute(A2_colmajor);
    ASSERT_EQ(ref.info(), Eigen::Success);
    const Eigen::VectorXd x_ref = ref.solve(b);

    EXPECT_LT((x - x_ref).cwiseAbs().maxCoeff(), 1e-10);
}

// Regression guard: single-triangle input must stay correct.
TEST(AccelerateInterface, SingleTrianglePatternSurvivesAnalyzeThenFactorize) {
    const SpMat A1 = spd_full_upper();
    SpMat A2 = A1;
    for (Eigen::Index k = 0; k < A2.nonZeros(); ++k)
        A2.valuePtr()[k] *= 1.5;

    LDLT s;
    s.analyze_pattern(A1);
    s.factorize(A2);
    ASSERT_EQ(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    const Eigen::VectorXd x = s.solve(b);
    EXPECT_LT((dense_symmetric(A2) * x - b).cwiseAbs().maxCoeff(), 1e-10);
}
```

Build, run, commit:
```bash
cd build && ninja -j4 tycho_tests_light && ctest -R AccelerateInterface --output-on-failure
cd .. && git add tests/cpp/solvers/test_accelerate_interface.cpp
git commit -m "test(solvers): pin P1 triangle canonicalization across analyze/factorize

Converts the macOS handoff's check-1 probe into permanent coverage.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Update the handoff doc**

Edit `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification.md`:
1. Retitle to the overhaul and note it now covers I1–I6, not just P1–P4.
2. Mark **check 2c PASS**, resolved by reading the SDK's inline source: every `SparseSolve` overload takes `Factored` **by value** and passes `&Factored` to `_SparseSolveOpaque` (`SolveImplementationTyped.h:754-761`), so the caller's status is unreachable from a solve. No rework of P2 is needed.
3. Mark checks 1, 2a and 2b as superseded by `tests/cpp/solvers/test_accelerate_interface.cpp`.
4. In step 0, before `cmake --preset`, add the submodule step that `main` introduced:
   ```bash
   git submodule update --init dep/pocketfft
   ```

- [ ] **Step 3: Run the full pre-merge gate**

**One build at a time.** This is the only heavy build in the plan; touching `accelerate_interface.h` invalidates every `psiopt.h` consumer, so expect 20–40 minutes.

```bash
eval "$(conda shell.bash hook)" && conda activate tycho
cd build && ninja -j4 all
ctest --output-on-failure
cd .. && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Then the C++ example and benchmarks:
```bash
cmake --preset macos-llvm-release -DBUILD_CPP_EXAMPLES=ON -DBUILD_CPP_TESTS=ON
cd build && ninja -j4 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
cd .. && bench/bench_track.sh compare
```

Expected: all ctest pass; all 32 examples exit 0; brachistochrone prints "Optimal Solution Found" with objective ≈ 1.8013 s; no unexplained benchmark regressions.

- [ ] **Step 4: Record the results file**

Create `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification-RESULTS.md` with the machine and toolchain versions, `git rev-parse HEAD`, verbatim results for each gate step above, the outcome of Task 7 Step 1's aliasing decision, and any warnings newly exposed by Task 3. Commit with prefix `docs(handoff):`.

- [ ] **Step 5: Retitle PR #88 and rewrite its body**

```bash
gh pr edit 88 --title "fix(solvers): Apple Accelerate interface overhaul (review series, split from PR 7)"
```

The body must cover: the six invariants and what each fixes; that check 2c is resolved with the SDK citation; the new test suite and its four honest coverage gaps; the deliberate decision to match Pardiso's failure contract with issue #105 tracking the unification; the deferred divergences (`set_matrix` semantics, `mem_` units, `flops_`/`ppivs()`); gate results; and that **CLAUDE.md requires Grant's explicit review for Accelerate integration changes**.

- [ ] **Step 6: Push**

```bash
git push origin fix/review-accelerate
```

Do not merge. Await Grant's review.

---

## Self-Review

**Spec coverage:** I1 → Task 2. I2 → Task 3. I3 → Task 4. I4 → Tasks 5 (contract) and 6 (workspace). I5 → Tasks 7 and 3 (the `:332` deprecation, reassigned from I5 to I2 as noted in Task 3's header). I6 → Task 8. Test suite (spec §5) → Tasks 1, 2, 4, 5, 6, 7, 8, 9 incrementally; all nine spec test rows are covered, including both P1 variants (Task 9) and the refinement-enabled failure case (Task 5). Probe requirements (spec §6.1) → Task 4 Step 1 (inertia), Task 5 Step 2 (refinement failure), Task 7 Step 1 (aliasing), Task 8 Step 7 (downgrade). Verification plan (§6.2) → Task 9 Step 3. PR mechanics (§7) → Task 9 Steps 2, 5, 6. Issue #105 → referenced in Task 5's in-code comment and Task 6's `mem_` comment.

**Gap accepted deliberately:** clearing inertia on *factorization failure* has no test, because the only type exposing `neigs()`/`peigs()` is LDLT^TPP and probing showed it will not fail numerically even on NaN input. Task 4 covers the `set_matrix`/`reinitialize` paths instead; the failure paths are inspection-only. This matches spec §5 gap 1.

**Type consistency:** `resetInertia()` (Task 2, used again in Task 4), `aligned_subbuffer()` (Task 6, one definition and one caller), `accelerate_supported_order()` (Task 8, used by `set_order` and by warmup), `effective_order()` (Task 8), and the fixtures `spd_both_triangles()` / `spd_full_upper()` / `indefinite_full_upper()` / `dense_symmetric()` / `break_factorization_in_place()` (Task 1 and Task 5) are each defined once and referenced by those exact names throughout.
