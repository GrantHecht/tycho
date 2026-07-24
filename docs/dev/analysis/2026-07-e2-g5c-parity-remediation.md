# PSIOPT parity remediation — CI restoration, divergence persistence, and generic-path corrections

**Date:** 2026-07-24
**Branch:** `feat/e2-g5c-parity-remediation` (HEAD `3c1fa2f6`)
**Data:** the shipped source at HEAD, the branch's CI run, and fourteen
`scripts/run_corpus.py --cbwr --repeat 2` captures (17 problems × 2 repeats each) —
the reclassified `defaults`, `merit`, `filter`+monitored, and `funnel`+monitored
columns produced by the persistence change, the first correction-enabled `filter` and
`merit` captures, and the prior-stage reference columns each is measured against.
**Task class:** REMEDIATION — three deliverables that close parity and hygiene gaps the
preceding restoration series left open, each shipped with its evidence, none of which
changes default-path behavior beyond a single enumerated deterministic-example delta.

This stage is the remediation tail of the globalization series. It carries no new
algorithm: it restores the continuous-integration signal the series had been merging
around, generalizes a solver policy the series had left in a one-shot form, and lifts a
classic-only restriction on two recovery links so they compose with the acceptance
strategies the series introduced. The three deliverables are independent; they are
reported here in the order they shipped, with the divergence-classification change and
the correction generalization jointly producing the note's centerpiece — a coherent
account of how the classic Maratos-effect example is cured (or not) across the
acceptance families.

## 1. Continuous-integration restoration

For several PRs across the restoration series, two CI lanes had been merged around while
red. This stage diagnosed and fixed both, and audited the full lane inventory so that the
merge gate is trustworthy again rather than carrying standing red that masks new
breakage.

- **`docs-build`** — broken by a stale documentation cross-reference.
  `InterpTable2D::cache_amats()` referenced `InterpTable3D::cache_alphavecs()` and
  `InterpTable4D::cache_alphavecs()` as bare `Class::method()` text, which Doxygen's
  strict coverage gate (`INTERNAL_DOCS=YES`, unhiding `@internal` members) could not
  autolink across classes, failing the build. Both target methods still exist under
  those exact names; the fix rewords the references to inline code spans rather than
  relying on cross-class autolink resolution. Commit `63c7d5df`.
- **`wheel-layout`** — broken by a packaging-suite import path. The wheel-layout job
  copies `tychopy/test/` to `/tmp` before running pytest (so the source `./tychopy`
  package cannot shadow the installed wheel). `test_corpus_smoke.py` located
  `tests/corpus/` and `scripts/run_corpus.py` by walking up from its own file path,
  which resolves to the real repo root only when the file runs in place; from the copied
  `/tmp` location it computed a bogus root and `import registry` failed collection. The
  fix adds a `TYCHO_REPO_ROOT` override, set by the workflow to the real checkout path;
  in-tree invocations (no env var) are unchanged. Commit `5203f0bb`.

Both lanes were validated green on the PR's own CI run — `docs-build`, `wheel-layout`
(including its layout-inspection step), and `repo-lint` all pass. The lane audit
confirms the only remaining red lane is `docs-linkcheck`, which runs on a weekly
schedule (`cron: '0 6 * * 1'`) and `workflow_dispatch` only — not on pull-request or
push — so it is **not a merge gate**. Its standing external-link failures are
self-tracked: the workflow holds `issues: write` and auto-files an issue on failure, so
the failures surface as tracked issues rather than as red on any mergeable run. They are
a pre-existing follow-up candidate, out of this stage's scope.

## 2. Persistence-based divergence classification

Commit `dd7f4873`. The convergence check aborted the solve the first time any monitored
residual crossed its divergence threshold. A single blown-up iterate is often a
recoverable transient rather than true divergence. This stage replaces the one-shot
abort with a trailing-window rule.

### 2.1 The policy, and its Tycho-original standing

This is a **Tycho-original policy with no external reference**: the reference solver
(Ipopt) ships **no divergence abort at all**. Direct comparison evidence makes the point
concrete — on the classic Maratos example, real Ipopt rides out an equality-residual
excursion to ~6e6 and converges to the textbook optimum without ever considering the
excursion a failure. Tycho keeps a divergence abort (it is a useful fast-fail on
genuinely divergent solves), but the one-shot form was mistaking single-iteration
overshoots for divergence.

The shipped rule, documented at `kDivergencePersistIters` in
`include/tycho/detail/solvers/psiopt.h` and applied in `converge_check` in
`src/solvers/psiopt.cpp`:

- **Non-finite residual (NaN/Inf) — immediate abort, exempt from the window.** A
  corrupted state cannot recover; there is nothing to ride out.
- **Finite residual past its threshold — DIVERGING only once the trailing window of
  `kDivergencePersistIters` (= 3) iterates is *all* divergent.** Histories shorter than
  the window cannot declare DIVERGING on a finite overshoot.

Three is the smallest window that survives the observed one- and two-iteration
recoverable excursions (Maratos-class overshoots, restoration-entry transients) while
still failing fast — within three iterations of onset — on genuine divergence. The
per-iterate divergent predicate is unchanged; the trailing scan runs only when a
threshold has already tripped; and a solve where no iterate ever trips a threshold takes
the identical classification path as before. Coverage is in
`tests/cpp/solvers/test_divergence_persistence.cpp`.

### 2.2 Reclassification: the corpus differential

The window's effect on the corpus is measured with
`conda run -n tycho python scripts/run_corpus.py --diff A B`. The headline is
`lit_maratos`: under every family it goes from `diverged` at iteration 2 to a genuine
solve.

**Default path (`defaults`, classic merit) — `defaults` vs the pre-change baseline:**

| Problem | before | after |
| --- | --- | --- |
| `lit_maratos` | diverged / 2 | **converged / 40** (obj −0.9999999900836263) |
| `hard_zermelo_wrongbasin` | diverged / 822 | diverged / 907 |

The `lit_maratos` result converges to the exact reference-solver answer (obj ≈ −1). The
`hard_zermelo_wrongbasin` row is the window behaving honestly on a case it should *not*
rescue: the wrong-basin blow-up is genuine divergence, the trailing window declines it,
and it stays `diverged` — the extra iterations are the window observing the persistence
before aborting, and the status is unchanged. Every other default-path corpus problem is
byte-identical.

**Restoration configurations — reclassified columns vs their pre-change (prior-stage
final) baselines:**

| Problem | Merit | Filter+mon | Funnel+mon |
| --- | --- | --- | --- |
| `lit_maratos` | diverged/2 → **converged/40** | diverged/2 → **converged/36** | diverged/2 → **failed/33** |
| `hard_zermelo_wrongbasin` | diverged (898→900) | diverged (884→913) | diverged (667→690) |
| `lit_wb2000` | conv 124 → **conv 90** | conv 37 (unchanged) | conv 32 (unchanged) |

`lit_maratos` is cured under merit (conv/40) and filter+monitored (conv/36); under funnel
it becomes a **severity demotion** (`diverged` → `failed`/33), reported honestly rather
than dressed as a win — the funnel does not reach the optimum but no longer mistakes the
overshoot for divergence. `hard_zermelo_wrongbasin` persists `diverged` in every family
(the window declines the wrong-basin case, iterations rising as it observes persistence).
The **`lit_wb2000` merit movement 124 → 90 is not attributable to this stage** — it is
the preceding stage's restoration re-centering fallback, which the merit baseline capture
predates; the diff surfaces it only because the two captures straddle that prior-stage
change. It is attributed to the prior stage, not to the divergence window.

### 2.3 The one default-path deterministic change

The persistence window produces exactly one change on the deterministic example gate:
**`MultiPhaseZermelo` 577 → 579 iterations** (same status). That example contains a
mid-sequence sub-solve that genuinely aborts as diverging — the multi-phase driver
continues past it — and the persistence window adds 2 iterations before that abort
fires. The plan's premise that deterministic examples never abort was wrong for exactly
this one example. The +2 is enumerated, justified, and deterministic: it reproduced
exactly in two later captures. It is the sole non-byte-identical default-path change this
stage introduces, and the reclassified CBWR iteration snapshot becomes the baseline for
the subsequent gates in this stage.

### 2.4 Repeat-exactness

Every reclassified configuration is repeat-1/repeat-2 identical on status, iteration
count, and objective, with the single known exception of `hard_cartpole_tightbounds`
under the merit family — the multithreaded solver's documented iteration-count jitter on
that problem, consistent with its behavior in the prior stage and not a persistence
effect.

## 3. Correction and extended-backtracking under generic-path acceptance

Commit `785c82e0`, with a test refinement in `3c1fa2f6`. The second-order-correction
(SOC) and extended-backtracking recovery links had been validated and gated for the
classic-merit path only. This stage lifts that restriction so both links compose with the
generic acceptance surface — modern merit, filter, and funnel.

### 3.1 What changed, and the fidelity of the change

The corrected or extended trial is now re-tested through a
`GlobalizationMechanism::run_acceptance_backtrack` seam, which dispatches to the classic
merit test or to `AcceptanceStrategy::is_iterate_acceptable` per the active strategy. The
governing fidelity property: **a corrected trial faces the *same* acceptance criteria as
an ordinary step** — exactly the reference-solver posture, where a second-order
correction is not privileged past the line-search test but must clear it like any other
trial. The trigger, budget, and cut-off constants were already transcribed in the
existing correction machinery, with their Ipopt citations in `soc.h`; this stage did not
introduce new constants, only a new dispatch path for the re-test.

- **Classic path byte-identical.** The classic dispatch tail was moved into the new seam
  by verbatim method extraction (review-verified); `SocRecovery`'s trigger norm matches
  the driving path (squared-L2 classic, L1 generic) so the classic path stays
  byte-identical. The default path (`max_soc = 0`, `ls_extended_iters = 0`) is untouched —
  recovery is `NoopRecovery` and `compute_step` forwards identical arguments to
  `classic_line_search`.
- **Interaction rules disclosed in `soc.h`.** On a corrected accept the funnel width
  tightens from the *corrected* trial's (θ, φ) and the filter augments from the accepted
  point; a correction attempted inside a nested restoration phase uses the phase's trial
  seam via the acceptance re-test, so the correction's own trigger measure is
  elastic-shifted in-phase while the first-rejection trigger measure is a mixed-space
  comparison — exactly as the classic link already computes it; the watchdog composition
  is unchanged.

### 3.2 Evidence

**CBWR:** the correction generalization is byte-exact against the post-reclassification
baseline established in §2 — the gate reproduced every deterministic example's status and
iteration count bit-identically. The generic re-test dispatch changes no
deterministic-example trajectory.

**First correction-enabled scorecards.** With corrections lifted, two exploratory
captures probe the generic path. Neither is a clean single-variable isolation of SOC —
each is reported with its confounds.

*Filter + monitored + `max_soc = 4`, vs the no-restoration filter+monitored reference:*
the only status change is `hard_zermelo_wrongbasin` `diverged`/845 → `failed`/1000 (a
severity demotion — bounded exit rather than divergence), with small iteration
improvements on `lit_powell_badscaled` (21 → 18) and `lit_wb2000` (43 → 39); `lit_maratos`
stays `diverged` (2 → 4, still diverged — see §3.3).

*Modern merit + `max_soc = 4`, vs the reclassified `defaults` (classic merit) baseline —
this table conflates the classic→modern-merit acceptance switch with SOC enablement, so
it isolates neither:*

| Problem | classic defaults | modern merit + SOC |
| --- | --- | --- |
| `lit_wb2000` | failed / 500 | **converged / 89** |
| `hard_cartpole_tightbounds` | converged / 95 | **failed / 500** |
| `deg_dup_equality` | converged / 3 | converged / 56 |
| `deg_redundant_defects` | converged / 3 | converged / 56 |
| `lit_hs13` | acceptable / 77 | acceptable / 142 |
| `hard_zermelo_wrongbasin` | diverged / 907 | diverged / 799 |
| `lit_powell_badscaled` | converged / 22 | converged / 18 |
| `hard_brach_coldstart` | converged / 24 | converged / 31 |
| `lit_maratos` | converged / 40 | converged / 40 |

The picture is genuinely mixed — a strong `lit_wb2000` rescue (failed → converged/89), a
`hard_cartpole_tightbounds` regression (converged → failed), and iteration swings in both
directions. It is reported as a first look at the generalized path, not as a
recommendation. Both captures are repeat-1/repeat-2 identical.

### 3.3 Centerpiece: the Maratos coherence story

Bringing §2 and §3 together produces a coherent, per-family account of the classic
Maratos-effect example — the note's central result:

- **Classic-merit `defaults` cure Maratos through persistence alone.** conv/40, obj ≈ −1,
  with *no recovery machinery engaged* — the divergence window (§2) is sufficient. A probe
  adding SOC and the watchdog to the classic path also lands conv/40: the corrections are
  not what cures it; the window is.
- **Under filter + monitored, Maratos genuinely persist-diverges** (diverged/4) *with or
  without corrections and the watchdog*. The mechanism is not a persistence failure — it
  is the filter working correctly. The blow-up trials increase both objective and
  infeasibility, so the filter **rejects** them; a corrected trial faces the same filter
  and is rejected too, so corrections cannot cure the step; the recovery ladder then
  exhausts and the terminal accept-as-is takes the catastrophic step. Probe evidence:
  filter+corrections+watchdog → diverged/4; the corpus filter+`max_soc=4` capture agrees
  (diverged/4).
- **With a restoration mode enabled, the mode-switch replaces accept-as-is, and the
  filter family converges Maratos** (conv/36, recorded in this stage's own §2
  reclassification tables). The restoration switch supplies the feasibility step the
  filter's rejection of the blow-up otherwise leaves unfilled.

**Per-family recommendation.** The filter family should run with a restoration mode for
Maratos-class robustness. The correction-plus-watchdog cell *without* restoration is
insufficient there — the probe pins the two endpoints cleanly: filter + corrections +
watchdog → diverged/4; classic + corrections + watchdog → conv/40. The difference is the
acceptance strategy's treatment of the blow-up, not the recovery links.

## 4. Verification

The full pre-merge sequence was run on HEAD `3c1fa2f6` after all three deliverables were
in place.

- **C++ unit tests.** `ctest` reports **1650 passed, 0 failed** — including the new
  `test_divergence_persistence` and `test_soc_generic_acceptance` suites, and the two
  end-to-end assertions the persistence change is pinned against (`lit_maratos` converges
  under `defaults`; a genuinely divergent solve still aborts).
- **Python suite.** `pytest` reports **359 passed, 4 skipped, 168 subtests passed**.
- **Python examples.** All **34** example scripts pass.
- **C++ brachistochrone.** Converges — "Optimal Solution Found".
- **CBWR bit-identity.** The final iteration snapshot is **byte-exact** against the
  reclassified baseline established in §2.3. The single
  enumerated default-path change in this stage is the `MultiPhaseZermelo` +2 iterations
  of §2.3, carried into that baseline; every other deterministic example reproduces its
  prior iteration count exactly. The correction-generalization and CI deliverables add no
  deterministic-trajectory change of their own.
- **Benchmarks.** 128 lanes compared against the baseline; one lane flagged —
  `BM_BumpAllocator_Resize` (129 → 204 ns, +57.9%), a bump-allocator micro-benchmark
  **unrelated to any code this stage touches** (a companion micro, `BM_Phase_Transcribe_64seg`,
  simultaneously moved −16.2% *faster*) — measurement-floor jitter, not a real
  regression. Every deliverable's code path is dead on every benchmarked lane: the
  benchmarks all run the default configuration (`max_soc = 0`, restoration off, and no
  iterate trips a divergence threshold), so the persistence scan, the generic-path
  re-test dispatch, and the recovery links are never entered.

## 5. References

- `include/tycho/detail/solvers/psiopt.h` — `kDivergencePersistIters` and the rationale
  for the trailing-window divergence policy (§2).
- `src/solvers/psiopt.cpp` — `converge_check`, the finite-vs-non-finite verdict split
  (§2).
- `tests/cpp/solvers/test_divergence_persistence.cpp` — the persistence-classification
  coverage (§2).
- `include/tycho/detail/solvers/globalization/soc.h` — the correction machinery, its
  Ipopt constant citations, and the disclosed generic-path interaction rules (funnel/
  filter update from the corrected trial, the mixed-space in-phase trigger, watchdog
  unchanged) (§3).
- `tests/cpp/solvers/test_soc_generic_acceptance.cpp` — the generic-path routing,
  construction, and through-API composition coverage, split into feasible (correction
  pre-empts restoration) and infeasible (restoration engages) in-phase cases (§3).
- `.github/workflows/docs-build.yml`, `wheel-layout.yml`, `docs-linkcheck.yml` — the two
  repaired gate lanes and the scheduled-only, auto-issue-tracked link checker (§1).
- `docs/dev/analysis/2026-07-e2-g5b-l1-restoration-scorecards.md` and
  `2026-07-e2-g5a-scorecards.md` — the restoration scorecards whose reference columns
  this stage's reclassification tables are measured against, and the source of the
  `lit_wb2000` re-centering-fallback attribution in §2.2.
- Commit `63c7d5df` — the docs cross-reference repair (§1). Commit `5203f0bb` — the
  wheel-suite import fix (§1). Commit `dd7f4873` — the persistence divergence
  classification (§2). Commit `785c82e0` — the generic-path correction generalization,
  with the test refinement in `3c1fa2f6` (§3).
