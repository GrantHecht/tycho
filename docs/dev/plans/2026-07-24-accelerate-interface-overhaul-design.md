# Apple Accelerate interface overhaul — design

**Date:** 2026-07-24
**Branch:** `fix/review-accelerate` (PR #88, to be retitled)
**Scope owner:** Grant Hecht — CLAUDE.md requires explicit human review for Apple Accelerate
integration changes regardless of review depth.

## 1. Context

PR #88 was split out of review-series PR 7 to carry four inspection-only fixes to
`include/tycho/detail/solvers/linear/accelerate_interface.h` that could not be compiled on the
Linux dev machine (`USE_ACCELERATE_SPARSE` is Apple-only):

- **P1** `factorize(a)` routed through `get_matrix(a)` instead of `matrix_ = a`, restoring the
  triangle canonicalization that `analyze_pattern` applies.
- **P2** `_solve_impl` reads the real `m_numericFactorization->status` instead of reporting a
  hardcoded `SparseStatusOK`.
- **P3** `doFactorization()`'s `factorSize`/`workspaceSize` locals widened `int` → `size_t`.
- **P4** `Index n_rows_ = 0, n_cols_ = 0;` in-class initializers.

A full-file review on macOS (2026-07-24) confirmed all four hunks and surfaced sixteen further
findings, several of them the same bug class as P3/P4 sitting on adjacent lines. This document
turns PR #88 into a scoped overhaul of the Accelerate backend.

### 1.1 PR #88's blocking unknown is resolved

The handoff doc (`docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification.md`, check 2c)
gated the merge on an empirical question: *can `SparseSolve` clobber
`m_numericFactorization->status` on a successful solve?* If it could, P2 would be unsound.

**Answer: it cannot, by construction.** Every `SparseSolve` overload takes `Factored`
**by value** and passes `&Factored` — a pointer to its own stack copy — to `_SparseSolveOpaque`
(`SolveImplementationTyped.h:754-761`, shipped inline in the macOS SDK). The caller's status field
is unreachable from a solve. Confirmed empirically: status identical across a good solve,
`||Ax-b||_inf = 4.4e-16`.

Corollary: P2's in-code comment is exactly right that solve-time-specific failures remain
unobservable through this API, and P2 cannot regress any path. Check 2c is marked **PASS**, resolved
by reading the SDK's own inline implementation and confirmed by probe — so the handoff's empirical
step is satisfied rather than merely waived.

## 2. Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Failed-solve contract | **Match Pardiso's current behavior**: guard, report via `info()`, leave `x` untouched | Parity now; leveling both backends up to a zero-`x` contract is deferred to a tracked issue |
| `SparseRefactor` failure | **Keep** the numeric factorization object — i.e. decline to add a reset; this is today's behavior | Apple documents the failed-numeric object as refactorable (state 3), and the user-workspace `SparseRefactor` overload validates only the **symbolic** status (`SolveImplementationTyped.h:865-872`). Probe-confirmed end to end: SPD factor → indefinite refactor (`status -1`, pointer unchanged) → SPD refactor on the *same object via the same overload* → `status 0`, residual 8.9e-16; and `info(): 0 → 1 → 0` through the public `refactorize_internal()`. PSIOPT's perturb-and-retry loop is exactly this pattern, so a reset would forfeit the documented cheap recovery. Because `doRefactorization` already keeps the object, this decision **cannot regress anything** — I4's guard is the only behavior change on that path |
| Structure | Single PR, one commit per invariant | Findings are independent few-line fixes; invariant grouping keeps each commit reviewable and revertible, and one macOS gate run covers the lot |
| Tests | Light gtest suite | The interface has zero coverage today; a leaf-header TU costs ~200 MB and seconds, versus 4–7 GB for the heavy path |
| `psiopt.cpp` | **Untouched** | Capability policy moves into `set_order()` so PSIOPT keeps passing its preference unchanged |

## 3. Invariants

Each becomes one commit. Line references are against the post-merge working tree.

### I1 — No indeterminate state at construction or after `release()`

- `mutable ComputationInfo info_ = Success;` (`:595`). Currently uninitialized and never assigned
  by the constructor. `info()` guards with `eigen_assert(m_isInitialized)`, which `NDEBUG` deletes,
  so a release-build caller reads indeterminate memory. Same bug class as P4, one member below it.
- `order_`, `do_iterative_refinement_`, `iterative_refinement_iterations_` move to in-class
  initializers; the constructor (`:241-275`) keeps only the `UpLo_ →` kind/triangle dispatch.
  The body-assignment style is how `info_` was missed.
- `~AccelerateImpl()` (`:279`) becomes `= default`. The class stays non-copyable and non-movable,
  which is required: `accel_matrix_` caches pointers into `matrix_`'s buffers. A comment records
  this so nobody "helpfully" adds a move constructor.
- `AccelFactorizationDeleter` (`:161`) drops its `sym = nullptr;` dead store on a local parameter.
- `release()` (`:864-892`) restores the default-constructed invariant: `n_rows_ = n_cols_ = 0`
  (today `rows()`/`cols()` report stale dims for an emptied solver, undoing P4), inertia cleared,
  and `accel_matrix_ = {}` plus `m_columnStarts.clear()` — after `matrix_.resize(0,0)` and
  `matrix_.data().squeeze()`, the cached `.data` and `.structure.rowIndices` dangle into freed
  buffers. Latent only because every in-tree path rebuilds via
  `reinitialize_internal_matrix_representation()` first.

### I2 — Diagnostics do not leak across translation units

Add `#include <Eigen/src/Core/util/ReenableStupidWarnings.h>` before the final `#endif` (`:896`).
The file includes `DisableStupidWarnings.h` at `:19`, which issues
`#pragma clang diagnostic push` (`dep/eigen/Eigen/src/Core/util/DisableStupidWarnings.h:46`), and
never includes the matching `pop` (`ReenableStupidWarnings.h:19`). `pardiso_interface.h:657`
balances it correctly.

Impact: `accelerate_interface.h` is pulled in by `psiopt.h:42`, `solver_context.h:50`, and
`src/solvers/tycho_solvers.h:19` — i.e. most of the ~155 Accelerate-enabled TUs (count estimated
from `compile_commands.json`, not verified per-TU) plus the PCH. Those
TUs compile the *rest* of their content with `-Wconstant-logical-operand` and
**`-Wimplicit-int-float-conversion`** suppressed. The second is the warning class that would have
flagged P3/P4.

### I3 — Cached derived state is never stale

- New private `resetInertia()` zeroing `peigs_`/`neigs_`/`zeigs_`, called on `doFactorization`
  failure (`:544`), `doRefactorization` failure (`:567`), and `release()`. `cacheInertia()`
  (`:411-426`) already zeroes in its `SparseGetInertia`-failure branch; this extends the same
  policy to factorization failure. It matters because PSIOPT's `Inertia()` and `RankDef()`
  (`src/solvers/psiopt.cpp:1641-1649`) read `neigs()`/`peigs()` immediately after
  `Compute()`/`Refactor()` without consulting `info()` — `CheckInfo()` (`:1662`) is explicitly
  observational — so stale inertia steers the perturbation loop.
- `flops_`/`mem_` reset on the same failure paths. Display-only, same class.
- `set_matrix()` (`:617-636`) and `reinitialize_internal_matrix_representation()` (`:783-805`) reset
  `info_` and both factorizations but leave inertia stale — unreachable through PSIOPT's read
  pattern, but the same class this invariant exists to eliminate, so both call `resetInertia()`.

Verified supporting fact: `SparseGetInertia` returns `int`, documented "0 on success, non-zero on
error" (`Solve.h:4497-4498`), so the existing `== 0` success test is correct.

Context that raises the stakes: LDLT^TPP applied to a fully singular (all-zero) matrix returns
`SparseStatusOK` and hands back `x = 0`. Status is a weak singularity signal on this backend;
inertia is the real one. `zeigs()` is never consumed anywhere, but that is not information loss —
`RankDef()`'s `neigs + peigs - kkt_dim != 0` recovers the zero-pivot count by identity. It does
mean that check is only as good as the freshness of those values.

Probing sharpened this further: LDLT^TPP fed a **NaN/Inf diagonal** also returns `info = 0`, with the
NaN pivot counted as a *negative* eigenvalue (`peigs=2, neigs=1`). So PSIOPT's inertia loop can
perturb against NaN-poisoned data while `info() == Success` — which is why I3's freshness guarantee
matters more than the status channel it sits beside.

### I4 — Failure is reported, never silently substituted

- `_solve_impl` (`:697`) gains a status guard:

  ```cpp
  if (!m_numericFactorization || m_numericFactorization->status != SparseStatusOK) {
      updateInfoStatus(m_numericFactorization ? m_numericFactorization->status
                                              : SparseStatusReleased);
      return;   // x deliberately unchanged — matches PardisoImpl::_solve_impl
  }
  ```

  Today only the pointer is checked. After a failed `SparseRefactor` the pointer is non-null with
  a bad status, so `SparseSolve` is called and Accelerate's own
  `SPARSE_CHECK_VALID_NUMERIC_FACTOR` (`SolveImplementation.h:137-146`) no-ops the solve after
  printing through tycho's `reportError` callback — leaving `x` at its previous contents. In PSIOPT
  that means `DXSL` (declared once at `psiopt.cpp:1729`, reused every iteration) keeps the previous
  iteration's step and is then negated at `:2324`. Empirically confirmed with an indefinite
  Cholesky refactor: status `SparseFactorizationFailed`, pointer non-NULL, `x` untouched.

  The retained hazard is deliberate parity. An in-code comment says so and points at the issue in
  §4, so a future reader does not "fix" it unaware.

  Precise severity: `EIGEN_INITIALIZE_MATRICES_BY_ZERO` is defined globally
  (`CMakeLists.txt:562`), so a failure on the **first** iteration leaves `DXSL` zeroed — a benign
  no-move step. The hazard is iterations ≥ 2, where `DXSL` holds the previous step and
  `psiopt.cpp:2324` negates it. Note the guard does **not** make every bad solve
  impossible: solve-time parameter failures (dimension or stride mismatch, whose `eigen_assert`s
  vanish under `NDEBUG`) still no-op while the factorization status the guard reads is healthy, so
  `info()` stays `Success`. That residue is inherent to the API, as §1.1 concedes.

  **The guard also fixes an active-corruption path, not just staleness.** With iterative refinement
  enabled (`do_iterative_refinement_`, driven by `qp_ref_steps_ > 0`; default `0` at
  `psiopt.h:346`), a failed factorization today produces *worse* than a stale `x`. The main
  `SparseSolve` no-ops, then the refinement loop (`:743-774`) runs anyway:
  `SparseMultiplyAdd` takes the **matrix**, not the factorization, so it computes
  `ref = -b + A·x_stale` normally; the inner in-place `SparseSolve` (`:765`) no-ops through the same
  `SPARSE_CHECK_VALID_NUMERIC_FACTOR`, leaving `ref` holding the raw residual rather than a
  correction; and `vDSP_vsubD` then applies `x -= ref`, i.e. `x = x_stale - (A·x_stale - b)` — once
  per refinement iteration. The early return removes this entirely. §5 adds a refinement-enabled
  failure test to pin it.

- **`info_` classification on the null branch.** A failed `factorize()` resets the pointer to null
  *and* leaves `info_ = NumericalIssue`. Recomputing from `SparseStatusReleased` would overwrite
  that with the less informative `InvalidInput`. So the guard preserves an existing non-`Success`
  classification and only assigns `InvalidInput` when the pointer is null and `info_` is currently
  `Success` (the never-computed case, where `InvalidInput` is right).
- `doAnalysis`'s `reportError` callback (`:499-502`) is labelled "Accelerate Sparse **Symbolic
  Factorization** Error", but Accelerate stores it in the symbolic factor and reuses it for numeric,
  refactor, **and solve-time** diagnostics — a probed solve failure printed
  "Symbolic Factorization Error: Factored does not hold a completed matrix factorization." The
  message becomes phase-neutral, since this PR is precisely about making failure triage legible.
- `getAlignedPointer` (`:404-409`) stops underflowing: `space - kAlign` wraps for
  `storage.size() < 16`, making `std::align` return `nullptr` (verified for sizes 0 and 8).
  The alignment arithmetic is extracted to `internal::aligned_subbuffer()`, beside
  `resizeForAccelerateAlignment` (`:193`), so it is unit-testable.
- Both call sites get a real check that sets `info_ = InvalidInput` and returns, replacing the raw
  `assert` at `:734` that `NDEBUG` deletes. `doRefactorization`'s call (`:561`) has no guard at all
  today. Accelerate's workspace parameter is `_Nonnull`, so this is the difference between a
  diagnosable failure and a null dereference.
- `_solve_impl`'s `int workspaceSize` (`:723`) and `cached_solve_workspace_size_` (`:600`) widen to
  `size_t`; the SDK declares `solveWorkspaceRequiredStatic`/`PerRHS` as `size_t`
  (`Solve.h:1557-1558`). The `!=` comparison (`:728`) becomes `>` so the buffer is a high-water
  mark rather than thrashing when `nrhs` shrinks.
- `mem_`'s narrowing assignment (`:430`) gets an explicit saturating conversion. `result_.mem_`
  stays `int` (`psiopt.h:413-414`) — see §4 for the unit divergence.

### I5 — Preconditions are compile-time or asserted, not assumed

- `static_assert(std::is_same_v<StorageIndex, int>)`. The requirement is real —
  `const_cast<int *>(innerIndexPtr())` (`:454`, `:472`) and `fopts.order = permutation_.data()`
  both hard-code `int` — but it is **already enforced**: `const_cast` cannot change a pointee type,
  so a non-`int` `StorageIndex` is ill-formed today. This is a diagnostic-quality improvement
  (one clear message instead of a template error cascade), not new enforcement. Pardiso asserts its
  index width in its constructor (`pardiso_interface.h:120`).
- `doAnalysis()` sizes `permutation_` by `std::max(n_rows_, n_cols_)` (`:486`). Per the SDK
  (`Solve.h:1310-1319`), `fopts.order` is a *symmetric row and column* permutation for the symmetric
  and QR cases and only splits row/column counts for LU — so `max` is the safe size for every type
  this template can instantiate. Caveat for the commit message: only `AccelerateLDLTTPP` is
  instantiated in-tree (`psiopt.h:856`), so this hardens **dead template code** rather than a
  reachable path.
- `buildAccelSparseMatrix` narrows `Index` dims to `int` (`:450-451`, `:469-470`) with no overflow
  check — the same width bug-class as P3, unaddressed at the 2^31 boundary. Add an assert that the
  dims fit `int`, completing the class.
- `reinitialize_internal_matrix_representation()` (`:783-805`) assumes `matrix_` is compressed but
  neither enforces nor documents it; `buildAccelSparseMatrix` reads `outerIndexPtr`/`innerIndexPtr`/
  `valuePtr` raw, which describe nothing coherent in uncompressed mode. This is safe in-tree **only
  by accident**: `analyze_sparsity` happens to end with `KKTmat.makeCompressed()`
  (`src/solvers/non_linear_program.cpp:331`). Since this is the entry point PSIOPT uses for in-place
  assembly (`psiopt.cpp:976-980`), add `matrix_.makeCompressed()` (a no-op when already compressed)
  so the contract holds by construction.
- Add the two row-major RHS asserts Pardiso has (`pardiso_interface.h:377-380`); `_solve_impl`
  hardcodes `columnStride = rowCount` (`:712`, `:719`) and would otherwise return wrong answers
  silently. Note these are `eigen_assert`s, so like Pardiso's they protect debug builds only —
  the project's Release default compiles them out.
- `get_matrix`'s `constexpr int TriangleType = (UpLo & Lower) ? Lower : Upper;` (`:332`) uses the
  anonymous enum member `UpLo` (`:228`) in a bitwise operation with `Eigen::UpLoType`, which clang
  reports as `-Wdeprecated-anon-enum-enum-conversion` (surfaced by the leaf-TU probe, §6.1). Its
  sibling in `get_matrix_twisted` (`:356`) already uses the `int` template parameter `U`; `:332`
  becomes `(U & Lower)` to match, fixing the deprecation and the inconsistency in one change.
- Aliasing: when `b.derived().data() == x.derived().data()`, copy `b` into a local temp and use the
  out-of-place solve — the same shape as Pardiso's `tmp` (`:393-399`), no new member. Deliberately
  *not* Accelerate's in-place overload, which destroys `b` while the iterative-refinement block
  (`:743-774`) still needs it as the residual reference.

  **Severity note:** probing shows an aliased solve (`xb = solver.solve(xb)`, 3×3, single RHS)
  currently returns the *correct* answer (`||Ax-b||_inf = 4.4e-16`). Accelerate does not document
  the out-of-place overload as alias-safe, and no in-tree caller aliases today
  (`DXSL = kkt_sol_.solve(RHS)`), so this is defensive hardening against unspecified behavior —
  **not** a live bug, contrary to the initial review write-up. A plan-stage probe characterizes the
  larger and multi-RHS cases before the guard is added; if aliasing proves reliably safe at scale,
  this item reduces to a documenting comment.

### I6 — Platform capability is checked at runtime, not at build time

Both gates in `accelerate_utils.h:29-37` test `__MAC_OS_X_VERSION_MAX_ALLOWED`, the **SDK**
version, which says nothing about the runtime OS or the deployment target.

- `BLASSetThreading` is `API_AVAILABLE(macos(15.0))` (`thread_api.h:34-35`) and therefore
  weak-linked: a wheel built on the 26 SDK with a lower deployment target can null-call it. The
  `#if` stays to guard the declaration; `if (__builtin_available(macOS 15.0, *))` layers on top,
  falling through to the `VECLIB_MAXIMUM_THREADS` path otherwise.
- `SparseOrderMTMetis` is `API_AVAILABLE(macos(26.0))` (`Solve.h:1262`) but only an enum constant —
  no link hazard, just `SparseParameterError` and a dead solver if passed on macOS < 26. The policy
  goes in `set_order()` (`:301`) via `internal::accelerate_supported_order()`, which downgrades to
  `SparseOrderMetis` at runtime. **This is what keeps `psiopt.cpp` out of the diff**: PSIOPT keeps
  passing its `QPOrderingModes::PARMETIS` preference unchanged and the interface owns its own
  capability policy. The layering is in fact *forced*: `psiopt.cpp:710-714`'s
  `#ifdef TYCHO_HAS_MTMETIS` must stay regardless, because the enum symbol does not exist on
  SDK < 26 — so build-time symbol policy in `psiopt.cpp` plus runtime OS policy in the interface is
  coherent rather than hidden.

  A silent downgrade of an explicitly requested ordering is user-visible behavior, so it must be
  observable: the interface exposes the effective order (a getter), which both makes the downgrade
  inspectable and closes §5's coverage gap 4.
- `warmup_sparse_solver()` sets `fopts.reportError`, and calls `SparseCleanup` on both the symbolic
  and numeric factors unconditionally rather than only on success. Without a `reportError`
  callback, any parameter-check failure takes the null-callback branch → `os_log_error` then
  **`_SparseTrap()`**, an abort (`SolveImplementation.h:81-94`). The null callback here is
  warmup's **own** zero-initialized `fopts`, not `_SparseDefaultSymbolicFactorOptions` (that object
  covers the pre-verification default and the 2-arg overloads); the conclusion is identical either
  way. Low severity — the warmup matrix is a hardcoded 2×2 SPD — but it is a startup-path abort risk
  plus a leak.

`CMAKE_OSX_DEPLOYMENT_TARGET` is currently empty, so clang defaults to the host and these gates are
*accidentally* correct for local builds. I6 is a wheel-shipping fix.

## 4. Explicitly out of scope

| Deferred | Destination |
|---|---|
| Unifying the failure contract (`info() != Success` **and** `x` zeroed) across both backends | New GitHub issue |
| Pardiso's uninitialized `analysis_is_ok_`/`factorization_is_ok_` (`:223`); dead `factorization_is_ok_`; success flags set unconditionally ignoring the error code (`:258-259`, `:281-282`); the `iparm_[0] == 0` guard (`:368`) that never fires because PSIOPT always calls `set_params()` (`psiopt.cpp:737`) | Same issue |
| The PSIOPT stale-step hazard (`DXSL` reused then negated) | Same issue |
| `set_matrix()` semantic divergence — Accelerate canonicalizes, rebuilds, and invalidates; `PardisoLDLT::set_matrix()` (`:605`) is a bare `matrix_ = mat` | Documented in both headers; issue tracks alignment |
| `mem_` unit divergence — Accelerate reports factor size in **bytes**, Pardiso reports **nonzeros** in the factor (`iparm_[17]`), both surfaced as `result_.factor_mem_` (`psiopt.cpp:2770-2771`) | Documented; the narrowing *is* fixed here, on real grounds: bytes means 2 GB factors are reachable, so the `int` truncation is a plausible display bug at scale rather than hygiene |
| `flops_` hardwired to 0 (`:433`) and `ppivs()` hardwired to 0 (`:379-383`) — Accelerate exposes neither | Documented in the same divergence table. PSIOPT guards the FLOPs print with `> 0` (`psiopt.cpp:2778`) so the line is simply absent rather than wrong |
| PSIOPT reacting to `info()` | Same issue; PSIOPT internals are CLAUDE.md-flagged and would need an E2 G0 corpus re-run |
| Rewriting the interface (unify phase methods, `constexpr` kind/triangle, single workspace owner) | Rejected: discards the deliberate shape-tracking with Eigen's `AccelerateSupport` that the MPL2 header calls out, and loses side-by-side diffability with `pardiso_interface.h` |

## 5. Test design

New `tests/cpp/solvers/test_accelerate_interface.cpp`, registered in `TYCHO_TEST_LIGHT_SOURCES`
(`tests/cpp/CMakeLists.txt:36-56`), body wrapped in `#ifdef USE_ACCELERATE_SPARSE` — the inverse of
`test_jet_mkl_guard.cpp:10`, so it compiles to an empty TU off-platform. Includes only
`accelerate_interface.h`, `<Eigen/Sparse>`, and gtest; explicitly **not** `test_utils.h`, which
would drag in `tycho.h` and make it a 4–7 GB TU.

| Test | Covers |
|---|---|
| Default-ctor `rows()`/`cols()`/`info()` | I1 |
| Both-triangle `analyze_pattern` → `factorize` reuse, residual vs `SimplicialLDLT`; single-triangle regression variant | P1 |
| Failed factorization surfaces via `info()` — `AccelerateLLT` + indefinite matrix | P2, I4 |
| Solve after failure: `info() != Success` and `x` unchanged | I4 contract |
| Solve after failure **with iterative refinement enabled** — `x` unchanged, not corrupted | I4 corruption path |
| Refactor recovery: good → failed → good on the same object, final solve correct | §2 keep-the-object decision |
| `release()` → dims 0, inertia 0, then a fresh `compute()` succeeds | I1, I3 |
| Inertia correctness on known-indefinite input | I3 |
| `internal::aligned_subbuffer` on 0/8/16/64-byte buffers | I4 |
| `b = solver.solve(b)` aliased solve | I5 |
| `set_order(SparseOrderMTMetis)` then solve succeeds | I6 |

The failure tests use `AccelerateLLT`, not the in-tree LDLT^TPP: probing showed LDLT^TPP returns
`SparseStatusOK` even for a zero matrix, whereas Cholesky of an indefinite matrix reliably fails.

**Coverage gaps, stated rather than papered over:**

1. Clearing inertia specifically on *factorization failure* is not directly observable — the only
   type exposing `neigs()`/`peigs()` is LDLT^TPP (SFINAE-gated), and that is the type that resists
   being forced into failure. Covered via `release()` plus inspection.
2. The `_solve_impl` workspace guard is reachable only when `workspaceSize == 0`, so it is covered
   at the `internal::aligned_subbuffer` level, not end-to-end.
3. The default-ctor `info()` test is a regression guard, not a demonstration: pre-fix it reads
   indeterminate memory and can pass by luck.
4. ~~`set_order` downgrade cannot assert which ordering ran.~~ **Closed** by I6's effective-order
   getter. What remains untestable on this host is the *downgrade branch itself* — on macOS 26.5
   `accelerate_supported_order(MTMetis)` returns MTMetis unchanged, so the test exercises the
   pass-through branch. The `< 26` branch is inspection-only.
5. The `:332` deprecation fix (I5) is a compile-time concern, so no runtime test covers it. The
   leaf-TU probe compile is the check: it must emit no `accelerate_interface.h` warnings afterwards.
6. The default-ctor `info()` test calls `info()` before initialization, violating its own
   `eigen_assert(m_isInitialized)` precondition — it passes only because the light suite builds
   Release/`NDEBUG`. It is written with an `#ifdef NDEBUG` guard so a Debug test build does not abort.

## 6. Verification plan

### 6.1 Probe-driven design and implementation

Every non-obvious premise in this document is settled by a standalone probe rather than by reading,
because the full build is 20–40 minutes and this backend cannot be exercised by CI at all. Probes
compile against the repo include tree in ~3 s:

```bash
clang++ -std=c++20 -O2 -DNDEBUG -DFMT_HEADER_ONLY \
  -I include -I dep/eigen -I dep/fmt/include probe.cpp -framework Accelerate -o probe
```

**Probes already run (2026-07-24):**

| Probe | Settled |
|---|---|
| `accel_probe.cpp` | Check 2c: no status clobber across a good solve; `SparseSolve` workspace sizes; `std::align` underflow returns `nullptr` for sizes 0 and 8 |
| `accel_probe2.cpp` | Failed `SparseRefactor` → pointer non-NULL, status `SparseFactorizationFailed`, `SparseSolve` a silent no-op leaving `x` untouched |
| `leaf_tu_probe.cpp` | `accelerate_interface.h` is self-contained as a leaf TU (3.3 s compile) — retires the §8 risk; `release()` leaves `rows()=3`, `peigs()=3` (confirms I1 and I3 empirically); aliased solve currently correct; surfaced the `:332` deprecation warning |
| `probe_state3.cpp` (review pass) | The keep-the-object decision: `SparseRefactor`'s user-workspace overload validates only the symbolic status, and a failed-numeric object recovers — `status -1` → `status 0`, residual 8.9e-16 — both through the raw C API and through `refactorize_internal()` (`info(): 0 → 1 → 0`) |
| `probe_ldlttpp_fail.cpp` (review pass) | LDLT^TPP will not report numeric failure even on NaN/Inf input (`info = 0`, NaN counted as a negative pivot), confirming coverage gap 1 is honest and that the failure tests must use `AccelerateLLT` |

**Probes required during implementation**, before the corresponding invariant is committed:

- **I3** — inertia on a known-indefinite matrix, to pin expected `peigs`/`neigs` values for the test
  rather than guessing them. Note the probe above returned `peigs=3` for an SPD matrix as expected.
- **I4** — the refinement-enabled failure sequence, to confirm the corruption path exists before the
  guard and is gone after it. The bare failed-refactor-then-solve case is already settled by
  `accel_probe2.cpp` and `probe_state3.cpp`.
- **I5** — characterize aliasing at larger `n` and with multiple RHS, to decide whether the guard is
  needed or a comment suffices.
- **I6** — confirm `internal::accelerate_supported_order()` downgrades without breaking a solve, and
  that `__builtin_available` compiles under the project's flags. The negative branch (macOS < 15/26)
  cannot be exercised on this host and is inspection-only.

Probes stay in the scratchpad and are **not** committed; whatever they establish is either encoded
in `test_accelerate_interface.cpp` or recorded in the handoff RESULTS file.

### 6.2 Build sequence

Respecting CLAUDE.md's 16 GB / one-build-at-a-time rule (`-j4`, `TYCHO_HEAVY_COMPILE_JOBS=1`):

1. **Leaf-TU compile check** of `accelerate_interface.h` standalone — **done**, 3.3 s (§6.1). The
   light-TU premise §5 depends on is confirmed.
2. **Fast loop:** `ninja -j4 tycho_tests_light` + `ctest -R Accelerate`.
3. **Full gate, once, at the end:** `ninja -j4 all` (touching this header invalidates every
   `psiopt.h` consumer), `ctest --output-on-failure`, all 32 Python examples via
   `scripts/run_examples.py`, `brachistochrone_cpp` ≈ 1.8013 s, and `bench/bench_track.sh compare`.
   The `/pre-merge` skill can drive this.

Note for anyone repeating step 3 from a fresh clone: `main` added a `dep/pocketfft` submodule, so
`git submodule update --init dep/pocketfft` is required before configure succeeds. This is added to
the handoff doc, whose step 0 predates it.

## 7. PR mechanics

- Retitle PR #88 and rewrite the body; the four-fix description no longer describes the change.
- Keep `5b581ae` (P1–P4) and the merge `f19ed05` as history so the existing Opus review of the
  original hunks remains meaningful. Add one commit per invariant (I1–I6), plus a tests commit and
  a docs commit.
- Update `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification.md`: retarget at the
  overhaul, mark check 2c **PASS** with the `SolveImplementationTyped.h:754-761` citation, add the
  `dep/pocketfft` step.
- Open the parity issue from §4 and reference it from both the in-code comment in `_solve_impl` and
  the PR body.
- State in the PR body that CLAUDE.md requires Grant's explicit review for Accelerate integration
  changes.

## 8. Risks

| Risk | Mitigation |
|---|---|
| ~~`accelerate_interface.h` is not self-contained as a leaf TU~~ | **Retired** — verified by probe, 3.3 s standalone compile (§6.1) |
| The aliasing guard (I5) may be solving a non-problem | Probe showed the current aliased path returns correct results; a plan-stage probe characterizes larger/multi-RHS cases, and the item reduces to a comment if aliasing proves safe |
| Keeping the failed numeric factorization diverges from `doFactorization`'s reset-on-failure | Intentional and documented; Apple's state-3 semantics make it the supported path (cite the *double*-variant doc block — the float variant's identical list says "pass this object to `SparseRefactor_Double`", an upstream copy-paste, so citing it is misleading), and the `_solve_impl` status guard closes the hazard the reset was protecting against |
| Behavior change on the failure path could perturb PSIOPT convergence | The failure path is unreachable on converging problems; the full gate (32 examples + benchmarks) is the check |
| Runtime `__builtin_available` gates cannot be tested on this machine (macOS 26.5 host) | Inspection-only, and stated as such. The `set_order` test exercises the **pass-through** branch, not the downgrade — on this host `accelerate_supported_order(MTMetis)` returns MTMetis unchanged. Neither the `macOS < 15` nor the `< 26` branch can be reached here |
| ~~Keeping the failed numeric factorization may not actually be supported by `SparseRefactor`~~ | **Retired** — probe-confirmed recovery through both the C API and `refactorize_internal()` (§6.1) |
| Suppressed-warning removal (I2) may expose pre-existing warnings elsewhere | Expected and desirable; any that appear are triaged in this PR or recorded |
