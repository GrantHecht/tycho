# PSIOPT inertia correction: full IPOPT IC condition on the classic ladder — design

**Date:** 2026-07-25 (rev 2 — re-grounded against `main` at `c9e8ddd`, which includes
PR #103's proximal primal-dual regularization mode)
**Branch:** `fix/psiopt-inertia-correction` (off `main` at `c9e8ddd`)
**Status:** approved design, pre-implementation
**Review flag:** PSIOPT optimizer internals — requires explicit human review before merge
(CLAUDE.md policy).

## Problem

`PSIOPT::factor_impl` (src/solvers/psiopt.cpp:1639) accepts a factorization whenever
there are no **excess negative eigenvalues** (`IncEigs <= 0`). A factorization that is
**rank-deficient** (`peigs + neigs < kkt_dim`) or has **too few negatives** (`neigs < m`)
is accepted: rank deficiency prints a warning and nothing else. The singular solve then
returns a zero component in the null space, the iterate never moves, and the solver
crawls to `max_iters`. This holds on the default `classic` inertia branch
(psiopt.cpp:1719-1730) and on the shared perturbation ladder's exit (psiopt.cpp:1746).

Diagnosed end-to-end in the PR #88 investigation
(`docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification-RESULTS.md`):

- **Failing case (red on macOS at defaults, verified on `c9e8ddd`):**
  `DivergencePersistence.MaratosCorpusConvergesAtDefaults` — flag=2, 500 iterations.
  At the start point the least-squares multiplier init makes ∇²L exactly zero; the KKT
  matrix `[[0,0,0],[0,0,2],[0,2,0]]` has true inertia (1,1,1). Accelerate LDLT-TPP
  reports it honestly (`status=OK`, `p/n/z = 1/1/1`); classic accepts (`IncEigs = 0`),
  solves a zero step, and stalls.
- **Why Linux never saw it:** MKL Pardiso automatically perturbs near-zero pivots, so
  reported inertia always sums to n — exact rank deficiency never surfaces on MKL.
  Verified: Apple's `SparseNumericFactorOptions` has **no perturbation option**
  (`pivotTolerance` selects pivots; `zeroTolerance` classifies values as zero — SDK
  `Sparse/Solve.h`). Accelerate's contract is honest inertia; the optimizer is the
  correct consumer of it.
- **Out of scope:** docking `Form2` fails for a different reason — instrumentation shows
  `zeigs = 0` on all its factorizations (n=23077); a backend-numerics convergence
  difference, not rank deficiency. Stays red after this fix; separate campaign item.

## What PR #103 already provides (reused, not duplicated)

The `proximal_regularization` inertia mode (opt-in; default remains `classic`,
psiopt.h:342) landed the machinery this fix needs:

- `NonLinearProgram::perturb_kkt_c_diags(double, mat&)` — writes onto the
  constraint-row diagonals (the solver-owned `e_pivot`/`i_pivot` blocks).
- `tycho::solvers::dual_regularization(mu)` = `kDualRegScale · μ^kDualRegExponent`
  (1e-8, 0.25 — Ipopt `jacobian_regularization_value`/`_exponent`, pinned commit in
  `globalization/inertia_regularization.h`).
- `factor_impl` already takes `base_prox`/`dual_shift`; the proximal branch applies
  both up-front and treats a singular base as ladder-entry (psiopt.cpp:1711-1718),
  explicitly disclosing that classic keeps the warn-and-proceed gap.
- Nested-restoration suppression rationale (inertia_regularization.h:61-71): while a
  nested l1 restoration phase is active, δ_c must NOT be applied — the elastic pivots
  own the constraint-row diagonals (~1/μ) and the condensed elastic recovery assumes
  the (y,y) diagonal equals the elastic pivot exactly. (This also means constraint-block
  singularity cannot arise during nested restoration.)

**Probe finding (recorded, out of scope):** the proximal mode does NOT fix the Maratos
case — it **diverges** (flag=3, obj ≈ 7.8e17, 4 iterations, verified on `c9e8ddd`).
With ∇²L ≡ 0 the ρ = 1e-10 floor yields a correct-inertia but catastrophically
ill-conditioned system; the ~g/ρ step blows up before the divergence window trips. So
"flip the default to proximal" is not an alternative to this fix, and the proximal
mode's behavior on exactly-singular Hessians is its own campaign follow-up (file an
issue; do not fix here).

## Design decisions (settled with Grant)

1. **Full IPOPT Algorithm IC condition** on the classic branch and the shared ladder —
   not a minimal primal-only patch. δ_w provably cannot fix a rank-deficient constraint
   Jacobian (the m − rank(J) zero eigenvalues survive any primal perturbation);
   redundant equality constraints are a routine user modeling error that MKL masks
   today. δ_c comes from the existing #103 machinery, engaged **on demand**.
2. **On exhaustion, fail the step (IPOPT-faithful)** — route into the existing recovery
   chain rather than today's warn-and-proceed.
3. **Reuse #103's pieces verbatim** — no new NLP methods, no new constants.

## Algorithm

Let `m = equal_cons_ + inequal_cons_`, `n_kkt = kkt_dim_`. Expected inertia is
`(n_kkt − m, m, 0)`.

```
factor_impl(docompute, Zfac, ipurt, incpurt0, incpurt, &finalpert, &cumpert,
            base_prox, dual_shift, &exhausted):        // signature: only &exhausted new
    wrong()    := (neigs != m) || (peigs + neigs != n_kkt)   // full IC condition
    singular() := (peigs + neigs != n_kkt)
    dc_applied = false
    engage_dc():= if (!dc_applied && dual_shift != 0.0):
                      PerturbC(-dual_shift); dc_applied = true

    if proximal mode:                                   // branch structure as today
        Perturb(base_prox)
        if dual_shift != 0.0: { PerturbC(-dual_shift); dc_applied = true }  // as today
        factor; CheckInfo; RankDef; bookkeeping as today
        if singular(): engage_dc()                      // no-op here (dc_applied)
        if !wrong(): return 0
    else if Zfac || docompute:                          // classic branch
        factor; CheckInfo; RankDef; bookkeeping as today
        if singular(): engage_dc()                      // δ_c on demand — the fix
        if !wrong(): return 0                           // was: IncEigs <= 0
    p = ipurt
    for i in 0 .. max_refac_-1:                         // shared ladder
        Perturb(p); cumpert += p; Refactor; CheckInfo; RankDef; finalpert = p
        if singular(): engage_dc()                      // first observation mid-ladder
        if !wrong(): return i + 1                       // was: IncEigs <= 0
        p escalation exactly as today (incpurt0 / incpurt)
    exhausted = true                                    // NEW distinct signal
    warn (existing message, extended with the inertia triple)
    return max_refac_
```

- **`dual_shift` semantics change:** it becomes "the δ_c magnitude available to this
  call" for BOTH modes. `alg_impl` hoists the existing computation
  (`nested_active ? 0.0 : dual_regularization(mu)`, currently proximal-only at
  psiopt.cpp:2280-2286) out of the mode conditional so classic passes it too. Proximal
  applies it up-front exactly as today; classic applies it only on observed
  singularity. The nested-restoration suppression is inherited from the hoisted
  computation — under nested phases `dual_shift == 0.0` and `engage_dc()` is inert.
- An `engage_dc()` mid-call takes effect at the next `Refactor()` (same matrix the δ_w
  `Perturb` writes to), so handling a singular base costs one ladder rung — a small
  δ_w rides along with δ_c, matching Ipopt (which raises δ_w alongside δ_c on
  singularity).
- The δ_w loop body, escalation constants, `finalpert`/`Hpert0` warm-start, `cumpert`
  display accounting, and the return-value (`h_facs_`) semantics are byte-identical to
  today. The changes are the two acceptance predicates, `engage_dc()`, and `exhausted`.
- The strengthened ladder exit applies to the proximal branch too — coherent with its
  own design (its base already treats singular as ladder-entry; today its ladder could
  still exit singular).

## Component changes

### 1. `PSIOPT::factor_impl` (psiopt.h declaration ~:979-982; psiopt.cpp:1639)

- Signature gains `bool &exhausted` (out; set only on exhaustion). Today's return value
  `max_refac_` is ambiguous between success-on-last-attempt and exhaustion; `exhausted`
  disambiguates without disturbing `Citer.h_facs_` consumers (HPert display, Zfac
  cycling heuristic).
- Body per the pseudocode above. The exhaustion warning gains the observed inertia
  triple (`peigs/neigs/zeigs` equivalent via `peigs()`, `neigs()`, and
  `kkt_dim_ − peigs − neigs`).

### 2. Call site (psiopt.cpp:2269-2293) — hoist + route

- Hoist `nested_active`/`dual_shift` computation out of the
  `InertiaModes::proximal_regularization` conditional (classic now passes a nonzero
  `dual_shift` too; `base_prox` stays 0.0 on classic — classic must NOT apply a base
  primal shift). The proximal-only display/decay block (psiopt.cpp:2305-2319) is
  unchanged.
- `bool kkt_exhausted = false;` threaded into the call. The stale policy comment at
  psiopt.cpp:2290-2293 is replaced by the new policy:
- **Exhaustion routing:** the iteration proceeds through the solve and line search
  unchanged, but when `kkt_exhausted` is set the line-search verdict is forced to
  *rejected*, so control flows into the existing
  `recovery_->on_step_rejected(...)` dispatch (psiopt.cpp:~2411 region) with every
  precondition satisfied by construction: `kSwitchToFeasibility` → existing
  `enter_feasibility_restoration()` (entry-permitted guards intact); `kGiveUp` →
  existing abort flagging; other actions per the chain's normal semantics. Rejected
  alternative: calling `on_step_rejected` directly while skipping the solve — it would
  synthesize a post-line-search context the chain was not designed to receive. Cost of
  the chosen route is one wasted solve on a near-unreachable path.

### 3. Backends — no interface changes

The predicates use only `peigs()`/`neigs()`, present on both `AccelerateLDLTTPP` and
`PardisoLDLT`. On MKL, `peigs + neigs == n` always (internal pivot perturbation), so
`singular()` is near-unfireable there; on Accelerate it is the fix.

## Risk

- **The one MKL-reachable behavior change:** today classic silently accepts
  `neigs < m`; the full IC condition corrects it instead. `neigs < m` can occur
  transiently on MKL (perturbed pivots), so some Linux problems may now take perturbed
  refactorizations where they previously proceeded. Literature-correct; the full Linux
  corpus run (CI on the PR) gates the merge. Otherwise the new predicate is a strict
  superset of today's — every case corrected today is corrected identically.
- **Shared-ladder exit affects the proximal mode** (strengthening: it can no longer
  exit still-singular). #103's tests (a) `ProximalRegularizationConvergesOnRankDeficientKkt`,
  (c) `WellConditionedParityAcrossModes`, (d) the nested-restoration composition test
  must stay green (they should be inert to this change: (a) is nonsingular after its
  base δ_c; (c) never triggers correction; (d) runs under nested suppression).
- δ_c applied on demand changes no solve where `singular()` never fires — the exact
  condition that is unreachable in ordinary operation on both backends.

## Tests (red → green on macOS/Accelerate at `c9e8ddd`; Linux corpus via CI)

1. **Primary (exists, red today):** `DivergencePersistence.MaratosCorpusConvergesAtDefaults`
   (tests/cpp/solvers/test_divergence_persistence.cpp:209) — must converge to obj −1.0
   in ≤ 60 iterations at defaults (classic).
2. **Upgrade (exists as conditional):**
   `InertiaRegularizationSolve.ClassicOnRankDeficientKktDocumented`
   (tests/cpp/solvers/test_inertia_regularization.cpp:151) currently documents classic
   behavior with `if (flag == CONVERGED)` guards; upgrade to assert `CONVERGED`,
   obj ≈ 0.5, primals ≈ (0.5, 0.5) unconditionally, using the existing
   `build_duplicated_equality_nlp()` (rank(J) = 1 < m = 2 — the δ_c path). Red on
   macOS today; Linux may already pass via Pardiso's rescue — portable either way.
   Rename to drop "Documented".
3. **New — exhaustion routing:** Maratos problem with `max_refac_` forced to 0 so every
   wrong-inertia factorization exhausts immediately (write the settings member directly
   with a comment if a validating setter forbids 0, as
   `test_event_refinement_coverage.cpp` does for its unsatisfiable tolerance). Assert
   the solve terminates promptly via the recovery path (restoration entry or a
   non-CONVERGED flag within a small iteration budget) — **not** a 500-iteration stall.
4. **Regressions:** `DivergencePersistence.GenuineDivergenceStillAborts`; all
   `InertiaRegularizationSolve` tests; full `ctest` (docking Form2 stays red —
   documented, out of scope); 34 Python examples; brachistochrone; benchmarks. The
   in-process Ipopt reference backend (#107) is available as a behavioral cross-check
   if attribution questions arise.

Verification tooling: the probe pipeline from the #88 investigation — recompile the 7
solver TUs (now including `ipopt_backend_stub.cpp`) with flags from
`compile_commands.json`, relink the standalone Maratos probe (~30 s total) — is the
fast red/green check before paying for the real build. `psiopt.h` is in the PCH
include chain, so the signature change costs one full rebuild; sequence all header
changes into the first implementation task so that cost is paid once.

## Delivery

Single PR from `fix/psiopt-inertia-correction`. Commits per component (factor_impl +
call-site hoist; exhaustion routing; tests). File a GH issue for the proximal-Maratos
divergence finding. Pre-merge gates per CLAUDE.md (ctest, examples, brachistochrone,
benchmarks) on macOS, plus Linux CI corpus. Explicit human review required (PSIOPT
internals).
