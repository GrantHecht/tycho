# Globalization refinement areas from the evaluation campaign

Date: 2026-07-25. Source evidence: `docs/dev/analysis/2026-07-e2-g8-campaign.md`
and the committed artifacts under `tests/corpus/campaign/`. Status: recorded
work items — none scheduled until the campaign evidence PR and the presets/docs
stage land.

## 1. Trial-evaluation exceptions abort the solve instead of rejecting the step

**Evidence:** both example-arm "new failures" are evaluation-domain exceptions,
not solver regressions: `InterpTable2D: query x=1.80012 outside table x range
[0, 1.8]` (MinimumTimeToClimb under prox+ℓ1 — the iterate grazes 0.007% past
the aero-table edge) and `ParallelDriver: step size underflowed ... non-finite
derivatives` (Betts central shooting under the filter/monitored/ℓ1
configuration). The globalization mechanisms legitimately visit trial points
the classic path never did; the evaluation layer converts an out-of-domain
state into a process-killing exception that bypasses the step-rejection
machinery entirely.

**Reference behavior:** Ipopt treats a failed evaluation as a rejected trial
point and backtracks — and Tycho's own Ipopt adapter already implements
exactly that (evaluator exceptions latch and return false). The built-in
solver has no equivalent.

**Refinement:** catch evaluation exceptions at trial-point evaluation inside
the acceptance backtrack, treat as a rejected trial (bounded retries), latch
the message into diagnostics; a committed-point evaluation failure remains
fatal. PSIOPT step-path change: own CA-gated PR, CBWR bit-identity on the
default path required (the classic path should be unaffected unless an
exception occurs, which today is fatal anyway — byte-identity expected by
construction).

## 2. Regularization-mode shift double-memory (post-episode over-shift)

**Evidence:** the review of the regularization mode flagged that after a ladder
episode the successful shift is remembered twice — the persistent shift carries
its decayed value as the next base AND the ladder warm-start seeds from the
same delta — so the first rung of a subsequent episode lands at roughly twice
the intended shift. The campaign attached measured costs consistent with that
mechanism: with the mode as default, MinimumTimeToClimb +137%, OptimalDocking
+99%, MultiSpacecraftOptimization 1535 → 11969 iterations (+680%), while the
median stays +0.00%.

**A/B RESULT (2026-07-25): the double-memory hypothesis is mostly refuted and
the shipped behavior stands.** Variant B (ladder warm-start neutralized once
the persistent shift absorbs an episode) leaves the dominant tails unchanged
(MinimumTimeToClimb +138%, OrbitContinuation +34% — their cost comes from the
persistent-shift path itself), improves only OptimalDocking (+99% → −58%,
faster than stock) and MultiSpacecraft partially (+680% → +548%), worsens
BettsLowThrust (+23% → +45%) — and **destroys the flagship wb2000 rescue**
(converged @95 → failed @500): the warm-start memory is load-bearing for
re-entering the ladder high. Conclusion: no implementation-vs-intention
deviation; the shipped trade is correct as designed. Any future tuning is a
genuine algorithm study (e.g. per-problem adaptive warm-start decay), for
which this A/B (captures `critb-proxonly-varB.csv` / `corpus-prox-varB.csv`,
session records summarized here) is the first data point.

## 3. Acceptable-tier semantics on structurally meaningless points

**Evidence:** the campaign winner's `hard_brach_illscaled` exit is an
ACCEPTABLE at objective −2.410 — a negative transfer time — where Ipopt
converges to 1.801 on the identical NLP.

**RESOLVED (2026-07-25 deep-dive): formulation admits it; no solver change
warranted.** The point is feasible to 3.1e-8 (inside even the strict equality
tolerance); the NLP is unbounded below via time-reversal symmetry with
delta-time unbounded; the acceptable tier applied its tolerances exactly as
written; and Ipopt applied no scaling on this problem (its 1.801 exit is basin
selection). Remaining action: an unboundedness note in the corpus module
docstring only.

## 4. Feasibility-only stage has no globalization (deep-dive finding)

**Evidence:** `solve()` / the feasibility stage of `solve_optimize` force the
objective scale to zero, under which every trial step is accepted — the
backtracking line search, recovery chain, and restoration are structurally
unreachable from that path (zermelo: 0 backtracks, 0 restoration entries,
empty filter across a 500-iteration stall while the equality residual grew
1.9×; the same problem run as a single `optimize()` with merit + nested-ℓ1
converges to Ipopt's exact point, 2.4e-15 relative agreement). The
globalization program scoped the optimize path; the feasibility path never
gained any of it.

**Refinement:** extend acceptance/recovery dispatch to the feasibility stage
(a feasibility-norm acceptance test is well-defined without an objective; the
candidate designs are in the deep-dive report). Likely the highest-value
remaining robustness item — it affects every `solve()`/`solve_optimize` user.

## 5. Harness follow-ups (deep-dive finding)

The corpus driver compares mismatched call shapes across backends (psiopt runs
each module's `SOLVE_MODE`; the ipopt backend always runs a single solve) —
add a matched-call option before quoting cross-backend rows for
non-`optimize` modules. (`run_corpus.py --config`'s repeated-flag handling was
fixed on the campaign branch after the deep-dive hit it.)

## 6. Smaller deferred items (consolidated from review notes and PR records)

Solver/adapter (from the Ipopt-backend branch's final review; apply when the
respective files are next touched, or batch into a maintenance pass):
- Jet × ipopt-backend concurrency is dispatchable but unconsidered (Ipopt is
  not reliably re-entrant): add a guard or an explicit unsupported note in
  `jet_run`.
- `solve()` under the ipopt backend runs the objective-bearing NLP (no
  feasibility-only analog): one sentence in the `nlp_solver` docstring at the
  next stubs regeneration.
- Adapter callbacks latch `std::exception` only; add `catch (...)` for
  contract completeness.
- `prob.ipopt_options` returns a copy (in-place item assignment silently
  no-ops): docstring note at next binding touch.
- `run_nlp_solver` branches `== ipopt` while the call-impls branch
  `== psiopt`: unify on `!= psiopt` whenever a third backend enumerator
  appears (hazard is zero until then).
- RPATH ordering (conda toolchain's rpath precedes the Ipopt lib dir, so a
  same-soname conda package shadows a custom Ipopt): needs a principled CMake
  answer if the backend is ever promoted user-facing; the dev-grade answer
  (no conda ipopt installed) is in effect.
- `capture_example_iters.py` has no per-config surface — measuring a
  non-default configuration "as default" requires the temporary-flip rebuild
  procedure used by the campaign (recorded in the campaign note §6).

Regularization-mode test follow-ups (from its review): direct tests asserting
the nested-restoration dual-shift suppression and the singular-base ladder
upgrade; the rank-deficient CONVERGED pin is untested on Apple Accelerate.
Also recorded there: a persistently singular base spends 1 + max_refac
factorizations per iteration with no constraint-side escalation.

Corpus/docs hygiene: internal planning labels persist in files predating the
no-labels rule (`tests/corpus/registry.py` section comments,
`tests/corpus/README.md` header, `scripts/run_corpus.py` instrument comment,
`test_corpus_smoke.py` docstring) — one dedicated cleanup commit; the
brach_illscaled module needs its unboundedness docstring note (area 3); the
docs-site linkcheck false positives (bot-blocked hosts) still want
`linkcheck_ignore` patterns in `docs/source/conf.py`.

## Deep-dive resolution record

The addendum's original two "Ipopt-only successes" both dissolved on
investigation: zermelo is solved by the stack under a matched call
(restoration head-to-head agreeing with Ipopt to machine precision — the
strongest correctness evidence produced for the restoration implementation),
and brach_illscaled is an unbounded formulation where Ipopt's advantage is
basin selection, not scaling (the earlier native-scaling attribution is
retracted). The implementations behaved as designed in both cases; the real
findings were the un-globalized feasibility stage (area 4) and the harness
call-shape confound (area 5).
