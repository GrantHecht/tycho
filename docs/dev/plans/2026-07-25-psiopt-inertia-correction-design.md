# PSIOPT inertia correction: full IPOPT IC condition on the classic ladder — design

**Date:** 2026-07-25 (rev 3, 2026-07-27 — re-anchored after merging main through #114
`e2314a3e`; degeneracy latch added per the lean-toward-IPOPT decision)
**Branch:** `fix/psiopt-inertia-correction` (main merged through `e2314a3e`)
**Status:** implemented on fix/psiopt-inertia-correction, PR pending, awaiting human review
**Review flag:** PSIOPT optimizer internals + a Python-visible enum addition — requires
explicit human review before merge (CLAUDE.md policy).

## Problem

`PSIOPT::factor_impl` (src/solvers/psiopt.cpp:1062) accepts a factorization whenever
there are no **excess negative eigenvalues** (`IncEigs <= 0`). A factorization that is
**rank-deficient** (`peigs + neigs < kkt_dim`) or has **too few negatives** (`neigs < m`)
is accepted: rank deficiency prints a warning and nothing else. The singular solve then
returns a zero component in the null space, the iterate never moves, and the solver
crawls to `max_iters`. This holds on the default `classic` branch (psiopt.cpp:1142-1153)
and on the shared perturbation ladder's exit (psiopt.cpp:1169).

Diagnosed end-to-end in the PR #88 investigation
(`docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification-RESULTS.md`):

- **Failing case (red on macOS at defaults; probe re-verified on the merged tree,
  2026-07-27):** `DivergencePersistence.MaratosCorpusConvergesAtDefaults` — flag=2,
  500 iterations, bit-identical across `c9e8ddd` and the #108-#114 merge. At the start
  point the least-squares multiplier init makes ∇²L exactly zero; the KKT matrix
  `[[0,0,0],[0,0,2],[0,2,0]]` has true inertia (1,1,1). Accelerate LDLT-TPP reports it
  honestly; classic accepts (`IncEigs = 0`), solves a zero step, and stalls.
- **Why Linux never saw it:** MKL Pardiso automatically perturbs near-zero pivots, so
  reported inertia always sums to n. Verified: Apple's `SparseNumericFactorOptions` has
  **no perturbation option** (`pivotTolerance` selects pivots; `zeroTolerance`
  classifies values as zero — SDK `Sparse/Solve.h`). Accelerate's contract is honest
  inertia; the optimizer is the correct consumer of it.
- **Confirmed unaddressed by #108-#114** (reviewed 2026-07-27): the classic predicate
  is textually unchanged; the default stays `InertiaModes::classic` (psiopt.h:342
  region); #114's presets are opt-in (`classic` = stock `Settings{}`); #110's stall
  detector is feasibility-stage-only and restoration-gated (Maratos stalls in the
  optimality stage with feasibility exactly satisfied); #109 handles trial-evaluation
  *exceptions*, which this solve never throws. The campaign's own
  `globalization-refinement-areas.md` records "no constraint-side escalation" in the
  ladder and "rank-deficient CONVERGED pin untested on Apple Accelerate" as open
  follow-ups — this design is that follow-up.
- **Out of scope:** docking `Form2` (zeigs = 0 on all its factorizations — a
  backend-numerics convergence difference, not rank deficiency).

## What PR #103 already provides (reused, not duplicated)

- `NonLinearProgram::perturb_kkt_c_diags(double, mat&)` — writes onto the
  constraint-row diagonals.
- `tycho::solvers::dual_regularization(mu)` = `kDualRegScale · μ^kDualRegExponent`
  (1e-8, 0.25 — Ipopt `jacobian_regularization_value`/`_exponent`, pinned commit in
  `globalization/inertia_regularization.h`).
- `factor_impl` takes `base_prox`/`dual_shift`; the proximal branch applies both
  up-front and treats a singular base as ladder-entry (psiopt.cpp:1134-1141),
  disclosing that classic keeps the warn-and-proceed gap.
- Nested-restoration suppression rationale (inertia_regularization.h:61-71): δ_c must
  not be applied while a nested l1 phase owns the constraint-row diagonals.

**Probe findings (recorded, out of scope):** the proximal mode does NOT fix Maratos —
it **diverges** (flag=3, obj ≈ 7.8e17, 4 iterations; re-verified on the merged tree
2026-07-27). With ∇²L ≡ 0 the ρ = 1e-10 floor yields a correct-inertia but
catastrophically ill-conditioned system. "Flip the default to proximal" is not an
alternative; file the finding as a GH issue (delivery section).

## Design decisions (settled with Grant)

1. **Full IPOPT Algorithm IC condition** on the classic branch and the shared ladder.
   δ_w provably cannot fix a rank-deficient constraint Jacobian; δ_c comes from the
   existing #103 machinery, engaged **on demand**.
2. **On exhaustion, fail the step (IPOPT-faithful)** — route through the existing
   recovery chain; an unresolved rejection aborts the phase as `SINGULAR_KKT`.
3. **Reuse #103's pieces verbatim** — no new NLP methods, no new constants.
4. **Degeneracy latch (rev 3, lean-toward-IPOPT):** adapt IPOPT's
   `hess_degenerate_`/`jac_degenerate_` memory. Once δ_c has been engaged, subsequent
   `factor_impl` calls pre-apply it at the classic base attempt instead of
   re-discovering the singularity, eliminating the wasted singular factorization per
   iteration on persistently rank-deficient problems (the exact cost
   `globalization-refinement-areas.md` records). Simplification vs IPOPT: the latch is
   **sticky per phase** (reset at each `alg_impl` phase init; IPOPT can un-diagnose).
   Residual cost of a stale latch is a μ-vanishing 1e-8·μ^0.25 shift.
5. **Deliberate non-parity, justified elsewhere:** PSIOPT's δ_w escalation constants
   stay native (corpus-validated; adopting IPOPT's schedule is a campaign measurement
   item via the #108 sweep driver, not this PR). Restoration stays opt-in (campaign
   no-flip recommendation); the `SINGULAR_KKT` abort is IPOPT's own
   restoration-unavailable branch, and the routing automatically becomes full IPOPT
   behavior whenever restoration is enabled.

## Algorithm

Let `m = equal_cons_ + inequal_cons_`, `n_kkt = kkt_dim_`. Expected inertia is
`(n_kkt − m, m, 0)`.

```
factor_impl(docompute, Zfac, ipurt, incpurt0, incpurt, &finalpert, &cumpert,
            base_prox, dual_shift, &exhausted):        // signature: only &exhausted new
    Singular()  := (peigs + neigs != n_kkt)
    wrong()     := (Inertia() != 0) || Singular()       // Inertia() = neigs - m, existing
    dc_applied = false
    EngageDualReg() := if (!dc_applied && dual_shift != 0.0):
                           PerturbC(-dual_shift); dc_applied = true
                           this->dc_latched_ = true     // degeneracy latch (sticky/phase)

    if proximal mode:                                   // branch as today
        Perturb(base_prox)
        if dual_shift != 0.0: { PerturbC(-dual_shift); dc_applied = true }
        factor; bookkeeping as today
        if !wrong(): return 0
    else if Zfac || docompute:                          // classic branch
        if this->dc_latched_: EngageDualReg()           // latch pre-application
        factor; bookkeeping as today
        if Singular(): EngageDualReg()                  // δ_c on demand — the fix
        if !wrong(): return 0                           // was: IncEigs <= 0
    p = ipurt
    for i in 0 .. max_refac_-1:                         // shared ladder
        Perturb(p); cumpert += p; Refactor; bookkeeping as today
        if Singular(): EngageDualReg()
        if !wrong(): return i + 1                       // was: IncEigs <= 0
        p escalation exactly as today
    exhausted = true
    warn (existing message, extended with the inertia triple)
    return max_refac_
```

- **`dual_shift` semantics change:** "the δ_c magnitude available to this call" for
  BOTH modes. `alg_impl` hoists the existing computation
  (`nested_active ? 0.0 : dual_regularization(mu)`, currently proximal-only at
  psiopt.cpp:1823-1830) out of the mode conditional. Nested-restoration suppression is
  inherited (`dual_shift == 0.0` ⇒ `EngageDualReg` inert, latch or not).
- Latch pre-application happens BEFORE the base factorization (mirrors the proximal
  base), so a latched problem factors correctly on the first attempt — 1 factorization
  per iteration instead of 1 + ladder.
- `dc_latched_` is a PSIOPT member, reset in `alg_impl`'s per-phase init (next to the
  `result_.last_kkt_info_` reset, psiopt.cpp:1227).
- The δ_w loop body, escalation, warm-start, `cumpert`, and return-value (`h_facs_`)
  semantics are byte-identical to today.
- The strengthened ladder exit applies to the proximal branch too (its base already
  treats singular as ladder-entry; today its ladder can still exit singular).

## Exhaustion routing (re-anchored to the post-#109/#110 loop shape)

The loop tail now has an established idiom for exactly this terminal — two precedents:
`exit_at_acceptable` (#109) and `exit_stage_stalled` (#110). The commit
`XSL += alpha * DXSL;` (psiopt.cpp:2295) executes AFTER the exit decision, so a
terminating iteration never commits; and `!GoodStep` already exits as `DIVERGING`
(psiopt.cpp:2257-2258), which covers the exhausted-plus-non-finite corner (the
non-finite verdict dominates; no special case needed).

1. Force the rejection before the dispatch gate (psiopt.cpp:2054
   `should_dispatch_recovery(GoodStep, Citer)` = `good_step && !accepted_`):
   `if (kkt_exhausted) Citer.accepted_ = false;`
2. In `kAcceptAsIs` (psiopt.cpp:2062), when `kkt_exhausted` and `resolved_depth`
   is still `kRecoveryDepthUnresolved`: `alpha = 0.0; singular_abort = true;`
   (a nested re-center stamps `kRecoveryDepthRestoration` and is a resolution;
   `kRetry`/`kSwitchToFeasibility`/`kSoftFeasibilityStep` are resolutions).
3. `bool singular_abort = false;` declared beside `exit_at_acceptable`
   (psiopt.cpp:2000). After the `exit_at_acceptable` upgrade (psiopt.cpp:2269-2270):
   `if (singular_abort) ExitCode = ConvergenceFlags::SINGULAR_KKT;` — SINGULAR_KKT is
   decisive over the acceptable-upgrade (an IC failure is a step-computation error,
   Ipopt `Error_In_Step_Computation`; composing it with an acceptable-stop heuristic
   is a possible future refinement, noted for review).
4. Add `ExitCode == ConvergenceFlags::SINGULAR_KKT ||` to the exit disjunction
   (psiopt.cpp:2281-2283).

New flag: `ConvergenceFlags::SINGULAR_KKT = 4` (psiopt_fwd.h:26-33; severity comment
extends to `... < DIVERGING < SINGULAR_KKT`). Consumers: psiopt_print.cpp:211 chain
(print "KKT System Persistently Singular"), jet.h:200 switch (count with NumDiv),
psiopt_bind.cpp:605 (`.value("SINGULAR_KKT", ...)`; regenerate the stub snapshot).

## Component changes (anchors at the #114 merge)

1. `factor_impl` definition psiopt.cpp:1062-1183 (declaration psiopt.h:987-989, doc
   comment above it rewritten for the new `dual_shift` semantics and `exhausted`).
   `PerturbC` moves from the proximal branch (:1120-1122) to function scope.
2. Call site psiopt.cpp:1823-1834: hoist `dual_shift`, thread `bool kkt_exhausted`.
   The stale proceed-anyway comment below the call is replaced by the new policy.
3. New PSIOPT member `bool dc_latched_ = false;` + friend-test access via the existing
   gtest-friend pattern (psiopt.h:54-61 forward decls, :763-770 friend block).
4. Recovery-dispatch edits per the routing section.
5. No solver-interface (backend) changes: predicates use `peigs()`/`neigs()` only.

## Risk

- **One MKL-reachable behavior change:** classic no longer silently accepts
  `neigs < m` (transiently possible on MKL). Literature-correct; Linux corpus CI gates
  the merge. Otherwise the new predicate is a strict superset of today's.
- **Shared-ladder exit strengthens the proximal mode too.** #103's tests
  (`ProximalRegularizationConvergesOnRankDeficientKkt`,
  `WellConditionedParityAcrossModes`, the nested composition test) must stay green.
- **Latch:** only activates on problems that already exhibited singularity; a stale
  latch costs a μ-vanishing δ_c. Sticky-per-phase (documented deviation from IPOPT's
  un-diagnose logic).
- δ_c on demand changes no solve where `Singular()` never fires.

## Tests (red → green on macOS/Accelerate at the merged tree; Linux corpus via CI)

1. **Primary (exists, red):** `DivergencePersistence.MaratosCorpusConvergesAtDefaults`
   — converge to obj −1.0 in ≤ 60 iterations at defaults.
2. **Upgrade (exists as conditional):**
   `InertiaRegularizationSolve.ClassicOnRankDeficientKktDocumented`
   (test_inertia_regularization.cpp:165) → assert `CONVERGED`, obj ≈ 0.5, primals
   ≈ (0.5, 0.5) unconditionally; rename to `ClassicConvergesOnRankDeficientKkt`. Uses
   the existing builder, renamed by #113 to `build_inertia_duplicated_equality_nlp`.
3. **New — degeneracy latch (friend test):** after solving the duplicated-equality
   problem under classic, the optimizer's `dc_latched_` is true; after the
   well-conditioned problem it is false.
4. **New — exhaustion routing:** Maratos with `settings().max_refac_ = 0` (public
   member, no validating setter) → `SINGULAR_KKT` within ~10 iterations.
5. **Regressions:** `GenuineDivergenceStillAborts`; all `InertiaRegularizationSolve`
   tests; full ctest (docking Form2 stays red — documented); 34 examples;
   brachistochrone; benchmarks; Linux CI corpus. The #107 in-process Ipopt backend is
   available as a behavioral cross-check.

Verification tooling: the probe pipeline — compile every `src/solvers/*.cpp` except
`ipopt_tnlp_adapter.cpp` (real-Ipopt TU; the stub provides `ipopt_backend::solve`)
with flags from `compile_commands.json` (fall back to psiopt.cpp's flags for TUs the
stale JSON lacks, e.g. `psiopt_settings.cpp`), recompile the probe TU, link against
the prebuilt non-solver archives (~1 min total). Verified working on the merged tree
2026-07-27.

## Delivery

Single PR from `fix/psiopt-inertia-correction`. Commits per component (factor_impl +
latch + call-site hoist; SINGULAR_KKT + routing; tests ride with their component).
File the proximal-Maratos divergence GH issue and a campaign-measurement issue for
adopting IPOPT's δ_w schedule via the #108 sweep driver. Pre-merge gates per CLAUDE.md
on macOS plus Linux CI corpus. Explicit human review required.

## Deviations from this design (as-built)

This document was approved pre-implementation (rev 3) and is linked from the PR body
as the authoritative spec. The final whole-branch review (F2) found five places where
the as-built tree on `fix/psiopt-inertia-correction` diverges from the text above —
all correct, intentional refinements that landed during the two round-of-review fix
passes (task-1-review.md, task-2-review.md), not regressions. Recorded here so the
spec is current with the merged tree rather than silently stale:

1. **δ_c engagement trigger widened.** The Algorithm block and Design decision 1
   above describe engagement as `if Singular(): EngageDualReg()`. As built, the
   trigger is `SingularitySignal() = Singular() || IncEigs < 0`
   (psiopt.cpp:1144 region) — a round-2 review finding (I1), not an original design
   decision. Rationale: by Gould's inertia theorem, `In(KKT) = In(Z^T H Z) + (m, m, 0)`
   for a full-rank constraint Jacobian, so `neigs < m` with `zeigs == 0` cannot occur
   for a full-rank `J`; it can only be a masked rank-deficiency report from a
   pivot-perturbing backend (e.g. MKL Pardiso's static pivot perturbation), and
   `IncEigs < 0` is exactly that signal. `delta_w` cannot correct this case (Weyl's
   inequality: `+p*I` on the primal diagonal is PSD, so `neigs` is non-increasing in
   `p`), so it is routed to `delta_c` instead.
2. **The Risk section's δ_c-inertness claim is backend-conditional.** "δ_c on demand
   changes no solve where `Singular()` never fires" holds only on honest-inertia
   backends (Apple Accelerate). On a pivot-perturbing backend (MKL Pardiso), the
   widened trigger from point 1 can engage δ_c — and set the degeneracy latch — on a
   masked `neigs < m` report where `Singular()` never fires. That is the trigger's
   purpose, not a side effect, and it is a real MKL-reachable behavioral delta that
   the Linux CI corpus run exercises.
3. **`Settings::validate()` relaxed to `max_refac_ >= 0`.** The Tests section (item 4)
   says "public member, no validating setter" — stale. `validate()` already pinned
   `max_refac_ >= 1` (added by #113's settings-contract sweep); this branch had to
   relax that bound to `>= 0` (src/solvers/psiopt_settings.cpp) so the exhaustion-
   routing test's `settings().max_refac_ = 0` is a legal configuration, and re-pinned
   the settings-contract test to assert both directions (0 valid, -1 throws). A real
   settings-contract change the original spec denies exists.
4. **The two backend-dependent tests are gated `#ifdef USE_ACCELERATE_SPARSE`,** not
   merely documented as platform-dependent: the `EXPECT_TRUE` half of
   `InertiaRegularizationSolve.ClassicDegeneracyLatchTracksSingularity`
   (test_inertia_regularization.cpp) and the whole of
   `DivergencePersistence.ExhaustedInertiaCorrectionAbortsAsSingularKkt`
   (test_divergence_persistence.cpp) compile out entirely on MKL Pardiso builds,
   because a pivot-perturbing backend can mask the premise (an honest wrong-inertia/
   singular report) outright rather than merely produce a flaky assertion.
5. **`run_phase_sequence` gained a new severity consumer.** The exhaustion-routing
   consumer list (Component changes item 4 / the `SINGULAR_KKT` paragraph) omits that
   `run_phase_sequence`'s inter-phase skip guard was widened from `== DIVERGING` to
   `>= DIVERGING` (psiopt.cpp:2767) — a round-2 review finding (I4) — so that a later
   phase's own `converge_flag_` can never silently overwrite the more-severe
   `SINGULAR_KKT` verdict left by an earlier phase; this is necessary for the new
   severity ordering (`... < DIVERGING < SINGULAR_KKT`) to mean anything across a
   multi-phase solve.

6. **Exhaustion no longer consults the merit-retry recovery links.** A limitation
   was originally recorded here ("resolution outranks abort": a configured recovery
   link could resolve the exhaustion-forced rejection, so a step vetted by recovery
   might still be committed from a wrong-inertia factorization). Post-merge review
   found that every merit-retry link (SOC, extended backtracking, watchdog
   relaxation, the soft feasibility pre-stage) can only re-test or relax acceptance
   of the very direction the never-correct-inertia factorization produced — which is
   not a resolution of the underlying step-computation error, and could commit a
   false-convergence step (a merit-decreasing Newton direction at a saddle). As
   built, `kkt_exhausted` now dispatches directly, bypassing the chain: nested
   elastic re-center when a nested l1 phase is active, else direct restoration entry
   when configured, inactive, and entry-permitted (skipping the soft pre-stage,
   whose trial would be the untrusted direction itself), else the `SINGULAR_KKT`
   abort — mirroring the un-evaluable-step routing and Ipopt's
   `Error_In_Step_Computation`. Regression-pinned by the backend-portable
   `DivergencePersistence.ExhaustedInertiaCorrectionIsNotResolvedByExtendedBacktracking`
   (wrong inertia by excess negative-eigenvalue count, which pivot-perturbing
   backends report honestly).
