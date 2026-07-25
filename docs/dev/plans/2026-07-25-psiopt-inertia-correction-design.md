# PSIOPT inertia correction: IPOPT Algorithm IC (δ_w + δ_c) — design

**Date:** 2026-07-25
**Branch:** `fix/psiopt-inertia-correction` (off `main` at `c9e8ddd`)
**Status:** approved design, pre-implementation
**Review flag:** PSIOPT optimizer internals — requires explicit human review before merge
(CLAUDE.md policy).

## Problem

`PSIOPT::factor_impl` (src/solvers/psiopt.cpp:1638) corrects the KKT inertia only when
there are **excess negative eigenvalues** (`neigs − m > 0`). A factorization that is
**rank-deficient** (`peigs + neigs < kkt_dim`) or has **too few negatives** (`neigs < m`)
is accepted: rank deficiency prints a warning and nothing else. The singular solve then
returns a zero component in the null space, the iterate never moves, and the solver
crawls to `max_iters`.

Diagnosed end-to-end in the PR #88 investigation
(`docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification-RESULTS.md`):

- **Failing case (red on macOS today):** `DivergencePersistence.MaratosCorpusConvergesAtDefaults`.
  At the start point the least-squares multiplier init makes ∇²L exactly zero; the KKT
  matrix `[[0,0,0],[0,0,2],[0,2,0]]` has true inertia (1,1,1). Accelerate LDLT-TPP
  reports it honestly (`status=OK`, `p/n/z = 1/1/1`); PSIOPT accepts (`IncEigs = 0`),
  solves a zero step, and stalls for 500 identical iterations.
- **Why Linux never saw it:** MKL Pardiso automatically perturbs near-zero pivots
  (`iparm(10)`), so reported inertia always sums to n — exact rank deficiency never
  surfaces on MKL; it manifests as wrong inertia instead, which *does* enter the
  existing correction loop. Verified: Apple's `SparseNumericFactorOptions` has **no
  perturbation option** (`pivotTolerance` selects pivots, clamped [0, 0.5];
  `zeroTolerance` classifies values as zero — SDK `Sparse/Solve.h`). Accelerate's
  contract is honest inertia; the optimizer is the correct consumer of it.
- **Out of scope:** docking `Form2` fails for a different reason — instrumentation shows
  `zeigs = 0` on all ~200 of its factorizations (n=23077); it is a backend-numerics
  convergence difference, not rank deficiency. It remains red after this fix and stays a
  separate campaign item.

## Design decisions (settled with Grant)

1. **Full IPOPT Algorithm IC**, not a minimal primal-only patch. Rationale: δ_w provably
   cannot fix a rank-deficient constraint Jacobian (the m − rank(J) zero eigenvalues
   survive any primal perturbation), and redundant equality constraints are a routine
   user modeling error that works on MKL today; a primal-only fix would leave that class
   macOS-broken — the exact platform-parity gap the campaign exists to close.
2. **On exhaustion, fail the step (IPOPT-faithful)** — route into the existing recovery
   chain rather than today's warn-and-proceed.

## Algorithm (Wächter & Biegler Algorithm IC, adapted)

Let `m = equal_cons_ + inequal_cons_`, `n_kkt = kkt_dim_`. Expected inertia of the KKT
matrix is `(n_kkt − m, m, 0)`.

```
factor_impl(docompute, Zfac, ipurt, incpurt0, incpurt, mu, &finalpert, &cumpert, &exhausted):
    wrong()        := (neigs != m) || (peigs + neigs != n_kkt)   // full IC condition
    singular()     := (peigs + neigs != n_kkt)
    engage_dc()    := if !dc_applied: perturb_kkt_d_diags(-kDualRegKappaC * pow(mu, kDualRegExponent));
                      dc_applied = true                          // 1e-8 * mu^0.25 (Ipopt defaults)

    if (Zfac || docompute):                                      // gating exactly as today —
        factor (Compute or Refactor as today); CheckInfo; RankDef // the cycling heuristic may
        if singular(): engage_dc()                               // skip the initial factor
        if !wrong(): return 0                                    // fast path unchanged
    p = ipurt                                                    // delta_w schedule UNCHANGED
    for i in 0 .. max_refac_-1:
        Perturb(p)                                               // existing primal machinery
        Refactor; CheckInfo; RankDef
        if singular(): engage_dc()
        if !wrong(): return i + 1
        p escalation exactly as today (incpurt0 / incpurt, finalpert bookkeeping)
    exhausted = true                                             // NEW distinct signal
    warn (existing message, extended with the inertia triple)
    return max_refac_
```

Notes:

- The δ_w loop body, escalation constants, `finalpert`/`Hpert0` warm-start, and
  `cumpert` display accounting are **byte-identical to today**. The changes are the
  acceptance predicate, the δ_c step, and the exhaustion signal.
- δ_c is applied **at most once per factor_impl call** (Ipopt: δ_c fixed within an IC
  episode), the first time singularity is observed in that call — whether at the initial
  factorization or mid-loop after a δ_w perturbation (relevant when `Zfac`'s cycling
  heuristic skipped the initial factor, or when the matrix goes singular only under a
  particular δ_w). Normal problems never see it: δ_c stays 0 whenever
  `peigs + neigs == n_kkt` throughout. A refactor after engage_dc() picks the new
  constraint-diagonal values up from the same matrix the δ_w Perturb writes to.
- Haynsworth: with J full-rank, once `H + δ_wI ≻ 0` the inertia is exactly
  `(n_kkt − m, m, 0)` — so δ_w alone resolves every non-singular wrong-inertia case, and
  δ_c resolves the `m − rank(J)` zero eigenvalues; the two knobs jointly cover the full
  condition.
- The KKT values are re-assembled from the NLP every iteration, so neither perturbation
  leaks across iterations (same property the existing δ_w relies on).

## Component changes

### 1. `NonLinearProgram::perturb_kkt_d_diags` (new; non_linear_program.h)

Sibling of `perturb_kkt_p_diags` (non_linear_program.h:277). The constraint-row diagonal
entries are already solver-owned KKT elements with precomputed locations — the
"solver equal pivots" and "solver inequal pivots" blocks (`e_pivot_data_start_`,
`i_pivot_data_start_`, contiguous; src/solvers/non_linear_program.cpp:242-244):

```cpp
void perturb_kkt_d_diags(double pert, Eigen::SparseMatrix<double, Eigen::RowMajor> &mat) {
    int ofs = this->e_pivot_data_start_ + this->num_user_kkt_elems_;
    for (int i = 0; i < this->equal_cons_ + this->inequal_cons_; i++) {
        mat.valuePtr()[this->kkt_locations_[ofs + i]] += pert;   // pert = -delta_c < 0
    }
}
```

Implementation must confirm what `assign`-time values the *inequality* pivot block
carries in the slack-complementarity formulation (equality pivots are the plain zeros);
δ_c adds to whatever is there, matching Ipopt's (2,2)-block treatment of both row types.

### 2. `PSIOPT::factor_impl` (signature + body; psiopt.h:979, psiopt.cpp:1638)

- Signature gains `double mu` and `bool &exhausted` (out, set only on exhaustion).
  Return value semantics unchanged (attempt count) — today `max_refac_` is ambiguous
  between success-on-last-attempt and exhaustion; `exhausted` disambiguates without
  disturbing `Citer.h_facs_` consumers (display, cycling heuristic).
- Constants (file-scope, psiopt.cpp): `kDualRegKappaC = 1e-8`,
  `kDualRegExponent = 0.25`. Not user settings (YAGNI; Ipopt defaults, documented).

### 3. Call site (psiopt.cpp:2219) — exhaustion routes to recovery

The comment block at psiopt.cpp:2220-2223 documents today's proceed-anyway policy; it is
replaced by the approved policy:

- `bool kkt_exhausted = false;` threaded into the `factor_impl` call (with `mu`, which
  is in scope; the one-iteration-stale μ is immaterial at δ_c's magnitude).
- The iteration proceeds through the solve and line search **unchanged**, but when
  `kkt_exhausted` is set, the line-search verdict is forced to *rejected*, so control
  flows into the existing `recovery_->on_step_rejected(...)` dispatch (psiopt.cpp:2411)
  with every precondition satisfied by construction:
  - `kSwitchToFeasibility` → existing `enter_feasibility_restoration()` (entry-permitted
    guards intact),
  - `kGiveUp` → existing abort flagging,
  - other actions per the chain's normal semantics.
- Rejected alternative: calling `on_step_rejected` directly while skipping the solve —
  it would synthesize a post-line-search context the chain was not designed to receive.
  Cost of the chosen route is one wasted solve on a near-unreachable path.

### 4. Backends — no interface changes

The predicate uses only `peigs()`/`neigs()`, present on both `AccelerateLDLTTPP` and
`PardisoLDLT`. On MKL, `peigs + neigs == n` always (internal pivot perturbation), so the
`singular` branch is near-unfireable there; on Accelerate it is the fix.

## Risk

**The one non-MKL-inert change:** today PSIOPT silently accepts `neigs < m`; the full IC
condition corrects it instead. `neigs < m` can occur transiently on MKL (perturbed
pivots), so some Linux problems may now take perturbed refactorizations where they
previously proceeded. This is the literature-correct behavior; the full Linux corpus run
(CI on the PR) gates the merge per existing policy. Everything else is provably inert or
confined: the new predicate is a strict superset of today's, so every case corrected
today is corrected identically; the only additional cases are `neigs < m` (the
MKL-reachable one above) and rank deficiency (near-unreachable on MKL, the fix on
Accelerate).

## Tests (red → green on macOS/Accelerate; Linux corpus via CI)

Location: `tests/cpp/solvers/test_divergence_persistence.cpp` (existing file) plus a new
leaf file if size warrants, following the existing Layer-2 public-API style.

1. **Primary (exists, red today):** `DivergencePersistence.MaratosCorpusConvergesAtDefaults`
   — must converge to obj −1.0 in ≤ 60 iterations.
2. **New — δ_c path:** duplicate-equality-constraint problem. Same Maratos objective and
   constraint, plus the *identical* constraint added a second time → rank(J) = 1 < m = 2
   → the m − r = 1 zero eigenvalue is δ_w-unfixable. Red on macOS today (stall), green
   after; passes on MKL both before and after (masked), hence portable. Assert
   CONVERGED, obj ≈ −1.0, bounded iterations.
3. **New — exhaustion routing:** Maratos problem with `settings_.max_refac_ = 0` so
   every wrong-inertia factorization exhausts immediately (if a validating setter
   forbids 0, write the settings member directly with a comment, as
   `test_event_refinement_coverage.cpp` does for its unsatisfiable tolerance). Assert
   the solve terminates promptly via the recovery path (restoration entry or a
   non-CONVERGED flag within a small iteration budget) — i.e., **not** a 500-iteration
   stall. Public API plus at most that one settings write.
4. **Regressions:** `GenuineDivergenceStillAborts` (persistence window must still abort
   genuinely divergent problems); full `ctest` (docking Form2 stays red — documented,
   out of scope); 34 Python examples; benchmarks (`bench_track.sh`). The in-process
   Ipopt reference backend (#107, on main) is available as a behavioral cross-check on
   the corpus if attribution questions arise.

Verification tooling: the instrumented-header workflow from the #88 investigation
(shadow-`-I` debug prints in `accelerate_interface.h`, ~20 s solver-TU rebuilds via
`compile_commands.json`) is the fast probe for confirming inertia sequences during
development, before any full build.

## Delivery

Single PR from `fix/psiopt-inertia-correction`. Commits per component (NLP method;
factor_impl; call-site routing; tests). Pre-merge gates per CLAUDE.md (ctest, examples,
brachistochrone, benchmarks) on macOS, plus Linux CI corpus. Explicit human review
required (PSIOPT internals).
