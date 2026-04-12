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

### Next diagnostic steps

1. **Two-phase minimum reproducer** — shrink from N=10 to N=2 phases and verify the
   regression still triggers. Then the diff surface is tiny enough to step through.
2. **Diff tycho vs asset on the link-constraint implementation**:
   - `src/optimal_control/optimal_control_problem.cpp` (tycho)
   - `src/OptimalControl/OptimalControlProblemBase.cpp` (asset)
   Look for how `LinkEqualCon` / `LinkParamEqualCon` compute sparsity, scale, and
   feed into the NLP Jacobian. The type-erasure refactor (`ac92cae`) could have
   corrupted the link-function storage or its argument binding.
3. **Run the multi-phase reproducer with `print_level=3`** in both tycho and
   asset_asrl, and compare the initial-iterate KKT inf, primal residuals, and
   the link-parameter values.

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
