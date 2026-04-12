# PSIOPT Regression — `multi_spacecraft_opt` Convergence Failure

**Status:** Identified — root cause narrowed, fix not yet implemented.

**TL;DR:** Asset_asrl converges this problem to a physically meaningful solution (`tf=0.947 periods`, positions on the unit circle). Tycho diverges to a degenerate local minimum (`tf≈0.002` or negative, positions scattered). The bug is **not** in the PSIOPT algorithmic code (tycho has all the relevant asset fixes). It is most likely in tycho's post-fork refactor of the VectorFunction type erasure system, which would corrupt gradient/Hessian evaluation in subtle ways that manifest as optimizer divergence.

---

## Reproducer

```python
import tychopy as typy
import numpy as np

vf = typy.VectorFunctions
oc = typy.OptimalControl
Args = vf.Arguments

class TwoBody(oc.ode_x_u.ode):
    def __init__(self, P1mu, ltacc=False):
        Xvars = 6
        Uvars = 3 if ltacc else 0
        args = oc.ODEArguments(Xvars, Uvars)
        r = args.head3()
        v = args.segment3(3)
        g = r.normalized_power3() * (-P1mu)
        acc = (g + args.tail3() * ltacc) if ltacc else g
        super().__init__(vf.stack([v, acc]), Xvars, Uvars)

# ... (see /tmp/psiopt_regression_test.py for full reproducer)
```

Or run the example directly:
```bash
./build/examples/cpp_examples/builder/multi_spacecraft_opt/multi_spacecraft_opt_builder
```

**Expected (asset_asrl):** `tf ≈ 0.947 periods`, rendezvous on unit circle, `flag = 0` (OPTIMAL)

**Actual (tycho):** `tf ≈ 0.002 periods` (or negative), rendezvous scattered, `flag = 3` (DIVERGING). Result is non-deterministic across runs but always degenerate.

---

## Investigation timeline

### Step 1 — Confirmed asset_asrl pip 0.5.1 works
Ran the exact example code under `asset_asrl 0.5.1` (pip-installed in `asset_test` conda env).
Result: `flag=0 tf=0.946862p tf_nd=5.9493 pos_norm=0.9754 phase0_back_r=0.9754` ✓

### Step 2 — Confirmed asset_asrl HEAD (built from source) works
Built `/home/ghecht/Projects/asset_asrl` at HEAD (`3a3ba6c`, dated 2024-09-30).
Result: identical to pip — `flag=0 tf=0.946862p ...` ✓

### Step 3 — Tested all reachable tycho commits, all fail
| Commit | Description | Result |
|--------|-------------|--------|
| `3b228ed` (main HEAD) | promote builder API types | FAIL |
| `61e1266` | static DSL examples — Phase 8 (#38) | FAIL |
| `6819977` | Phase 2 identifier migration (#35) | FAIL |
| `9915ee9` | project reorganization (#34) | FAIL |

The regression exists from at least PR #34 — earlier than all tycho-specific PSIOPT refactoring.

### Step 4 — Asset_asrl at tycho's fork point won't compile on modern GCC
Tried to checkout asset_asrl at `bec673e` (2022-11-15, the tycho fork point) but it fails to build on GCC 15.2.1 due to template-body strictness errors in `Integrator.h`. Also `-Wdivision-by-zero` flag is GCC-incompatible.

### Step 5 — Found the asset_asrl PSIOPT fix commit
Searched asset_asrl's git log for PSIOPT/phase changes between `bec673e..HEAD` and found:

**`96c3ceb` (2023-08-07) "Modified PSIOPT defaults"**

This commit contains 5 critical fixes:

1. **`OptLSMode` default: `NOLS` → `AUGLANG`** (line search enabled by default)
2. **`BoundFraction` default: `0.98` → `0.99`** (tighter)
3. **`GoodStep` detection** in `alg_impl`:
   ```cpp
   bool GoodStep = std::isfinite(DXSL.squaredNorm());
   if (GoodStep) {
       alpha = ls_impl(...);
   } else {
       Citer.Hfacs = -1;
   }
   if (!GoodStep) ExitCode = ConvergenceFlags::DIVERGING;
   ```
   Without this, when KKT solve produces NaN, the old code blindly takes the bad step and corrupts further iterations.
4. **Less restrictive line search acceptance**: accept step when objective decreases AND penalty (L2 or L∞) decreases, even if Lagrangian merit function doesn't strictly decrease:
   ```cpp
   if (LangTest < LangInit
       || (ptest < PrimObj) && (TestL2Pen < InitL2Pen)
       || (ptest < PrimObj) && (TestLinfPenalty < InitLinfPenalty))
   ```
5. **Hessian perturbation tweak**:
   ```cpp
   finalpert = p;  // was: finalpert += p
   if (i == 0) p *= incpurt0;
   else p *= incpurt;
   p -= finalpert;  // NEW
   ```

### Step 6 — All five fixes are present in tycho
| Fix | Asset commit | Tycho location |
|-----|--------------|----------------|
| `OptLSMode = AUGLANG` default | `96c3ceb` | `psiopt.h` Settings — VERIFIED at default `LineSearchModes.AUGLANG` |
| `BoundFraction = 0.99` default | `96c3ceb` | Settings — VERIFIED at default `0.99` |
| `GoodStep` detection | `96c3ceb` | `src/solvers/psiopt.cpp:1166-1225` ✓ |
| `secondary_accept` in line search | `96c3ceb` | `src/solvers/psiopt.cpp:1394-1399` ✓ used in `ls_l1` and `ls_auglang` |
| Hessian perturbation `p -= finalpert` | `96c3ceb` | `src/solvers/psiopt.cpp:998-1017` ✓ |

**Conclusion:** Tycho has the asset fix, so the bug isn't in the PSIOPT algorithmic code.

---

## Where the bug likely is

Since:
1. Asset HEAD works
2. Tycho has all PSIOPT fixes from asset
3. Tycho diverges in a way that suggests bad gradient/Hessian evaluation

The most likely culprit is **tycho's VectorFunction type erasure refactor**:

- **Commit:** `ac92cae feat: type erasure refactor (flat vtable + SBO) and C++20 bump (PR 6)`
- **Why suspicious:**
  - Type erasure bugs cause subtle UB (uninitialized memory, wrong vtable dispatch, alignment issues)
  - This refactor changed how functions are stored, dispatched, and evaluated
  - PSIOPT relies on accurate gradient/Hessian evaluation through the VF system
  - Subtle errors propagate as numerical noise that the optimizer sees as inconsistency
- **Was NOT tested:** I didn't bisect through PRs #1–#10 because each requires a full tycho rebuild (~3 minutes).

Other candidates:
- `ff31abb PR #4 add-provenance-headers`
- `bd1142d PR #2 decouple-cpp-python-bindings`
- `0e38b2c PR #12 typed enum state` (could affect PSIOPT enum dispatch)

---

## Update 2026-04-12 — Single-phase is pathological; multi-phase regression confirmed

Built a minimal **single-phase** min-time reproducer
(`doc/plans/single_phase_regression_test.py`). It **diverges in both tycho and
asset_asrl**, so the single-phase min-time formulation is inherently ill-posed
(degenerate local min at `tf ≈ 0`) — it is **not** a regression signal.

Re-verified the original multi-phase reproducer:
- `conda run -n asset_test python /tmp/psiopt_regression_test_camel.py` → **PASS**
- `conda run -n tycho      python /tmp/psiopt_regression_test.py      ` → **FAIL**

So the regression only appears in the multi-phase OCP with link constraints, which
points the finger back at **link-constraint machinery** (`add_link_equal_con`,
`add_link_param_equal_con`, or the OCP indexer handling of link-param variables
in the KKT matrix). Per-phase evaluation is not the bug on its own.

### Diagnostic results (2026-04-12)

- **N=2 reproducer behaves the same as N=10**, so link-constraint cardinality is
  not the issue. `doc/plans/n2_regression_tycho.py` diverges, `n2_regression_asset.py`
  converges in 81 iters to `tf=5.9493, pos_norm=0.9754`.
- **Iteration-0 comparison** at `print_level=3`:
  - Asset: `ECons Inf = 3.42e-01`, `AlphaP = 1.32e-1`, `AlphaD = 2.49e-1`
  - Tycho: `ECons Inf = 1.00e+00`, `AlphaP = 8.76e-3`, `AlphaD = 5.22e-3`
  Same IG, same problem, but different initial constraint residual in tycho.
- **Tycho is non-deterministic across runs**: 5 consecutive invocations of the
  N=2 script produced `tf_nd ∈ {−0.11, −129805, −3.47, 674, 2.37}` with equally
  varied `pos_norm`. Asset produces identical results every run.
  - Non-determinism persists with `OMP_NUM_THREADS=1` and
    `typy.Utils.set_num_threads(1)` (not a threading race).
  - Non-determinism persists when the bump-allocator capacity is pre-grown to
    20M elements (not an overflow-path bug).
  - Non-determinism occurs even across calls to `solve()` within a single
    Python process — so it's not process-startup state.
- **Narrowing:**
  - **No link constraints** (pure per-phase OCP): **deterministic** across runs
    (every run: `lp6 = 6.283185`, same flag). Still ends at a degenerate local
    min, but reproducibly.
  - **With `add_link_equal_con`** only (no link-param equal con): **non-
    deterministic**, same class of garbage results.
  - **With `add_forward_link_equal_con`**: also non-deterministic.
- **At the VF level**: `GenericFunction<-1,-1>` built from `Args(14).head(7) - Args(14).tail(7)`
  returns identical `compute` / `jacobian` values across repeated calls — the
  bug is not in the constraint function itself.

### Conclusion

The link-constraint evaluation path reads uninitialized memory **after** the
constraint function is stored in the NLP. Candidates, in order of suspicion:

1. **`OptimalControlProblemBase::transcribe_links` assembly into `nlp_`** — the
   `Func = IOScaled<decltype(Func)>(Eq.func_, ...)` wrap builds a local scaled
   function that is then stored via `ConstraintFunction(Func, VC[0], VC[1])`.
   If `ConstraintFunction` captures by reference rather than by value somewhere
   (or if `GFStorage::emplace` moves from a temporary that is subsequently
   destroyed), the constraint evaluates against freed memory. **Not** the
   auto-scaling branch — the reproducer leaves `auto_scaling_ = false`, so the
   wrap is skipped and `Func = Eq.func_` goes straight into ConstraintFunction.
2. **`LinkFunction` storage in `std::map<int, LinkConstraint>`** — the map
   values are copied when the map rehashes/rebalances. If `LinkFunction`'s
   copy constructor silently drops some member (e.g., `output_scales_`
   default-constructed in the copy but not in the source), a subsequent
   `ConstraintFunction(Eq.func_, ...)` sees a zeroed or garbage scale.
3. **`ConstraintFunction` constructor sizing the Jacobian workspace** — if it
   allocates a scratch `VectorXd` / `MatrixXd` and does not zero-init it
   before the first `compute_jacobian`, a sparse accumulate-into pattern
   could pick up stale stack values.

### Update 2026-04-12 — ASan/UBSan results

Wired AddressSanitizer + UBSan into CMake via `TYCHO_ENABLE_SANITIZERS=ON`
(adds `-fsanitize=address,undefined` + `-shared-libsan` on clang so the
runtime can be LD_PRELOADed into Python). Built the N=2 reproducer under
clang 21 + Fedora `compiler-rt` at `STRICT` FP mode, preloading
`/usr/lib/clang/21/lib/x86_64-redhat-linux-gnu/libclang_rt.asan.so`.

**ASan/UBSan findings:**
- One UBSan hit at `src/bindings/type_casters.h:193` — `memcpy(dst, nullptr, 0)`
  from `Eigen::VectorXi` with `size()==0`. Fixed separately (guard the memcpy
  and allocate at least one element).
- After the type_casters fix: **no ASan or UBSan diagnostics** on the N=2
  multi-phase reproducer across 3 consecutive runs. The reproducer still
  diverges and still reports different `tf_nd` every run.
- Non-determinism narrowed further:
  - `MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_CBWR=COMPATIBLE PYTHONHASHSEED=0`:
    still non-deterministic, but only in trailing digits
    (`tf_nd ≈ -82000 ± 4000` across runs). This is the footprint of
    ill-conditioned numerics being amplified, not wildly different
    iteration paths — the iterate is diverging toward `-∞ · t` where
    small perturbations explode.
  - Disabling ASLR (`setarch -R`) did not help.
- All fast-path instrumentation is clean, which means the source of
  perturbation is **either in a non-instrumented library** (statically linked
  MKL Pardiso or libiomp5) or is a **legitimate uninitialized-read that ASan
  cannot see** (ASan catches heap/stack OOB and UAF, not uninitialized
  reads — that would need MemorySanitizer, which conflicts with ASan and
  needs a fully-instrumented libstdc++).

### Update 2026-04-12 — NLP state after transcribe is fully deterministic

Dumped the full post-`transcribe(True,True)` state (indexer sizes, `v_index_`,
`c_index_`, input/output rows, variable counts, link-param locations, etc.)
across three consecutive runs of the N=2 reproducer. All 3097 lines of
output are **bit-identical**.

**Implication:** link-constraint indexing setup is fully deterministic. The
corruption is downstream of `transcribe_links()` — it must happen *during*
the numerical path of `solve()`, i.e., in one of:

- `ConstraintFunction::constraints_jacobian` (or siblings) reading a scratch
  buffer that is only partially written
- `NonLinearProgram::kkt_fill_*` accumulating into a pre-existing sparse
  matrix whose prior values are not zeroed per iteration
- PSIOPT's own internal workspace reusing a `VectorXd` / `MatrixXd` from a
  prior iteration without fully overwriting it

This moves hypothesis #1 (indexing corruption) to the bottom of the list.

### Hypotheses ranked after ASan + transcribe-dump results

1. **Numerical scratch buffer reused without full re-init per iteration.**
   Most likely inside `NonLinearProgram` or PSIOPT's KKT assembly — a
   `VectorXd` or sparse matrix is resized once, then accumulated into each
   iteration via `+=`, but a rare code path (link constraints only) writes
   to a different subset of indices than the previous iteration filled,
   leaving stale values in the unwritten slots.
2. **MKL Pardiso receives a slightly different `nnz` / `csr` layout each
   run** due to a hash-unstable assembly order. Unlikely — asset uses the
   exact same Pardiso path and is fully deterministic.
3. **Wild pointer dereference landing inside memory that happens to be
   "live" ASan-wise** — e.g., a stale `shared_ptr` that was upgraded from
   weak. Less likely given the shared-ptr refactor audit.

### Next diagnostic steps

1. Dump `nlp_->equality_constraints_[link_idx].index_data_` after
   `transcribe_links()` — compare `v_index_`, `c_index_`, and the inner
   starts across runs. If any of those differ, the corruption is in
   indexing setup, not constraint evaluation.
2. Rebuild with MemorySanitizer (even with false positives, the first few
   flagged reads in link-constraint territory will be meaningful).
3. Add a debug print at the top of `ConstraintFunction::constraints_jacobian`
   that dumps `X.segment(v_index_)` for the link constraint — if that differs
   run-to-run, the variable gather is reading from uninitialized state.
4. **Print the raw constraint output** at iteration 0 — add a debug print
   in `NonLinearProgram::compute_equalities` or equivalent to dump the link
   constraint's Fx value. If it varies across runs for the same input, the
   corruption is in the constraint eval itself.
2. **Enable AddressSanitizer** on a debug build and run the N=2 reproducer.
   This should flag the uninitialized read directly with a stack trace.
3. **Audit `ConstraintFunction` copy/move semantics** in tycho vs asset. The
   type-erased function may be stored by value in one and by reference in the
   other.

---

## How to continue the investigation

1. **Bisect through tycho commits 2a4cc72..ac92cae** to find the exact commit that introduces the divergence. Each iteration takes ~3 minutes (build + 10s test) so the full bisect of ~6 PRs is ~30 minutes.

2. **Compare ASSET::PSIOPT output to tycho::solvers::PSIOPT output** at iteration 0. Run both with `print_level=2` and a tiny test problem and diff the iterate-by-iterate values. The first divergence point will reveal which evaluation (objective, gradient, Hessian, KKT solve) differs.

3. **Reproduce on a single-phase problem** to rule out OCP linking machinery. If a single-phase minimum-time problem with the same dynamics also diverges in tycho but works in asset, the bug is in the per-phase evaluation, not the link constraints.

4. **Check tycho's `eval_*` for the constraint Jacobian** specifically — this is the most likely place for a type erasure bug to corrupt numerical results.

---

## Test fixture

Files used for testing during this investigation (kept under `/tmp` for easy access):

- `/tmp/psiopt_regression_test.py` — snake_case version for current tycho
- `/tmp/psiopt_regression_test_camel.py` — camelCase version for asset/old-tycho

Asset_asrl source repo: `/home/ghecht/Projects/asset_asrl` (built from source at HEAD)
Asset_asrl pip env: conda env `asset_test` (Python 3.12)

To rerun the regression test against current tycho:
```bash
cd /home/ghecht/Projects/tycho && conda run -n tycho python /tmp/psiopt_regression_test.py
```

To rerun against asset_asrl:
```bash
conda run -n asset_test python /tmp/psiopt_regression_test_camel.py
```

---

## Workarounds for the failing test

Until the regression is fixed, the `cpp_example_multi_spacecraft_opt_builder` ctest will fail because my Task 8 fix in the PR review correctly detects the divergence (which the original code silently swallowed). Options:

1. **Mark as `WILL_FAIL` in CMake**: documents the known failure in-tree
2. **Revert the multi_spacecraft_opt portion of Task 8**: silent-success behavior, lose detection
3. **Leave failing**: forces fix before merge

The author of this investigation recommends option 1 with a reference to this document.
