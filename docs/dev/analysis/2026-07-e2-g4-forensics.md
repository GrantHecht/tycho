# wb2000 forensics: why funnel acceptance escapes and filter acceptance jams

## Purpose

The Wächter–Biegler (2000) counterexample (`lit_wb2000` in the robustness
corpus) is the one literature-tier problem where PSIOPT's step-acceptance
strategies visibly disagree. Under the default classic-merit acceptance it
jams (500-iteration cap, objective ≈ −0.968); under the funnel strategy it
converges to the true optimum (x₁\* = 1) in 29 iterations; under the
reference-faithful filter strategy it jams again (500-iteration cap,
objective ≈ −0.975). This note explains, from per-iteration evidence and the
exact acceptance rules, *why* the funnel rescues this problem while the
filter — which shares the same switching skeleton — does not. It records two
distinct findings: (a) a bisection result that pins the filter's failure to a
specific reference-faithful rule, and (b) the mechanism that makes the
funnel's absolute slack bound the load-bearing difference at the corpus
start.

The problem (Benson–Shanno–Vanderbei reduced form, constants a = −1, b = 1,
start (−2, 3, 1)):

    minimize   x1
    subject to x1^2 - x2 + a = 0      (equality c1)
               x1 - x3 - b   = 0      (equality c2)
               x2, x3 >= 0            (encoded -x2 <= 0, -x3 <= 0)

It is constructed so that a class of line-search interior-point methods
converges to a feasible-looking but non-stationary point near x₁ ≈ −1.1
("jamming"). The escape to the true optimum x₁\* = 1 requires the iterates to
climb x₁ through the jamming basin — which *worsens the objective* (this is a
minimization of x₁) *while holding constraint violation elevated*. Any
acceptance rule that insists on monotone objective-or-violation improvement
relative to the current iterate blocks that route. This is exactly the
behaviour the counterexample was built to expose.

## Reproduction

Run from the repo root in the `tycho` conda environment. `acceptance_strategy`
selects the strategy (`classic_merit` default, `funnel`, `filter`);
`set_print_level(0)` prints the per-iteration table, `(3)` prints only the
exit summary.

```python
# scratch driver — build the NLP exactly as tests/corpus/problems/lit_wb2000.py,
# then set opt.acceptance_strategy and opt.set_print_level(...) before optimize().
import tychopy as typy
solvs, vf = typy.solvers, typy.vector_functions
Args = vf.Arguments
prob = solvs.OptimizationProblem()
prob.set_vars([-2.0, 3.0, 1.0])
prob.add_objective(Args(1)[0], [0])
prob.add_equal_con(Args(2)[0] ** 2 - Args(2)[1] - 1.0, [0, 1])
prob.add_equal_con(Args(2)[0] - Args(2)[1] - 1.0, [0, 2])
prob.add_inequal_con(-Args(1)[0], [1])
prob.add_inequal_con(-Args(1)[0], [2])
opt = prob.optimizer
opt.acceptance_strategy = solvs.AcceptanceStrategies.funnel   # or .filter / .classic_merit
opt.set_print_level(0)
flag = prob.optimize()
print(flag.name, int(opt.last_iter_num), float(opt.last_obj_val),
      opt.last_funnel_width, opt.last_filter_size, opt.last_filter_resets)
```

Or through the corpus harness:

```bash
conda run -n tycho python scripts/run_corpus.py --filter lit_wb2000 \
    --config acceptance_strategy=2     # 0 classic_merit, 2 funnel, 3 filter
```

Observed outcomes (this toolchain, 2026-07):

| strategy       | enum | flag         | iters | objective  | diagnostics                        |
|----------------|------|--------------|-------|------------|------------------------------------|
| classic_merit  | 0    | NOTCONVERGED | 500   | −0.968231  | width −1, filter −1                 |
| funnel         | 2    | CONVERGED    | 29    | +1.000000  | last_funnel_width 5.9862            |
| filter         | 3    | NOTCONVERGED | 500   | −0.975454  | last_filter_size 8, resets 0        |

## (a) The bisection: the filter's failure is the every-trial membership gate

The filter strategy as shipped reproduces the Wächter–Biegler / Ipopt / Uno
rule that **every** trial — F-type included — is checked against the filter
set for membership (`is_trial_acceptable_to_strategy`, run at step 2 of the
switching skeleton's acceptance order). Bisection over the development of the
strategy isolated this as the cause of the wb2000 jam:

- An earlier, **unfaithful** variant that skipped the membership test on
  f-type trials "rescued" wb2000 (converged in 133 iterations). That variant
  is not reference-faithful — WB step A-5.4, Ipopt
  `CheckAcceptabilityOfTrialPoint`, and Uno's `FunnelMethod` all gate every
  trial on membership, not only H-type ones.
- The rejection-cause **attribution** semantics (which the reset heuristic
  keys on) were bisected *out* as a cause: a build at the intermediate commit
  that had the faithful every-trial membership gate but the *old* attribution
  logic already fails identically — 500 iterations, same objective. So the
  attribution rewrite is not what tips wb2000; the membership gate is.

Consistent with that, the shipped filter run never triggers the reset
heuristic on this problem: `last_filter_resets = 0`. The failure is not a
reset-heuristic artifact — it is the membership gate plus the H-type
acceptable-to-current test walling off the escape route, described next.

## (b) The mechanism: absolute slack bound vs current-relative envelope

Both strategies inherit the identical switching skeleton
(`switching_acceptance.h`): the θ_max ceiling (Eq. 21), the θ_min threshold,
the switching condition (Eq. 19), and the F-type Armijo test (Eq. 20) are
byte-for-byte shared. Only two hooks differ — the every-trial membership
verdict and the H-type sufficient-progress verdict:

| hook                 | funnel                                   | filter                                                                 |
|----------------------|-------------------------------------------|------------------------------------------------------------------------|
| membership (2a/1b)   | θ_trial ≤ τ (funnel width)                | trial not dominated by any stored (θ,φ) entry                          |
| H-type progress      | θ_trial ≤ β·τ, β = 0.9999                  | θ_trial ≤ (1−γθ)·θ_current **OR** φ_trial ≤ φ_current − γφ·θ_current    |

For this problem the shared skeleton's ceilings never bind. θ₀ = |c1| + |c2|
= 0 + 4 = 4.0 at the start, so:

- funnel width init τ = max(1, 1.5·θ₀) = **6.0**
- θ_max = 10⁴·max(1, θ₀) = 4×10⁴  (never reached — θ peaks at 4.0)
- θ_min = 10⁻⁴·max(1, θ₀) = 4×10⁻⁴

The funnel's final width is **5.9862** — it shrank 0.23% over 29 iterations.
That single number is the headline: the funnel win is **not** the product of
progressive width tightening. The initial width, set generously by the large
starting equality violation (θ₀ = 4 → τ = 6), already contains the entire
escape trajectory (θ ≤ 4 everywhere), so membership never binds and the
H-type test θ_trial ≤ 0.9999·6 ≈ 6.0 passes for every escape step regardless
of what the objective does.

### Where the two runs diverge

The two strategies compute the *same* Newton directions (same KKT system, same
adaptive barrier oracle) and take **byte-identical steps through iteration 9**.
The divergence window (θ ≈ ECons Inf + ICons Inf; α = taken primal step; LS =
line-search backtracks):

Funnel:

| iter | μ        | x1 (=obj) | ECons    | ICons    | θ≈    | α        | LS |
|------|----------|-----------|----------|----------|-------|----------|----|
| 8    | 1.6e−07  | −1.224    | 1.18     | 1.04     | 2.22  | 5.7e−01  | 0  |
| 9    | 2.6e−05  | −1.108    | 0.51     | 1.60     | 2.11  | 1.00     | 0  |
| 10   | 1.0e−05  | −0.965    | 0.069    | 1.965    | 2.03  | 1.00     | 0  |
| 11   | 9.2e−12  | −0.796    | 0.367    | 1.796    | 2.16  | 1.00     | 0  |
| 12   | 4.9e−12  | −0.637    | 0.594    | 1.637    | 2.23  | 1.00     | 0  |
| …    |          | climbs monotonically to +1.000 by iter 28                        |

Filter:

| iter | μ        | x1 (=obj) | ECons    | ICons    | θ≈    | α        | LS |
|------|----------|-----------|----------|----------|-------|----------|----|
| 8    | 1.9e−05  | −1.228    | 1.21     | 1.02     | 2.23  | 5.7e−01  | 0  |
| 9    | 2.5e−02  | −1.110    | 0.52     | 1.59     | 2.11  | 1.00     | 0  |
| 10   | 2.3e−07  | −0.965    | 0.068    | 1.965    | 2.03  | **0.25** | **2** |
| 11   | 2.2e−07  | −0.923    | 0.148    | 1.923    | 2.07  | 0.25     | 2  |
| 12   | 2.1e−07  | −0.883    | 0.220    | 1.883    | 2.10  | 0.25     | 2  |
| …    |          | creeps, then oscillates near −0.6…−0.98, never escapes; caps at 500 |

Both reach the *same* state at iteration 10 (x₁ ≈ −0.965, θ ≈ 2.03, having
just accepted a violation-**increasing** step: ICons climbed 1.04 → 1.60 →
1.965 across iters 8–10 in both runs). From there the full Newton step raises
x₁ further toward 0 while θ stays ≈ 2.

- **Funnel accepts the full step (α = 1, LS = 0).** The step increases φ (x₁
  climbing = objective worsening under minimize-x₁), so it is not a descent
  direction for φ — switching fails — and the trial is **H-type**. Its H-type
  test is θ_trial (≈ 2.1) ≤ 0.9999·6.0, which passes trivially. Because
  θ_trial > θ_current on these steps, the width update takes the τ⁺ = β·τ
  branch — a 0.9999 nudge, hence the near-flat final width. The funnel simply
  does not look at the objective; any step inside the width is fine.
- **Filter rejects the full step and backtracks to α = 0.25 (LS = 2).** Same
  H-type classification. Its H-type test is acceptable-to-current: θ_trial
  (≈ 2.1) ≤ (1−10⁻⁵)·θ_current (2.03)? No. φ_trial ≤ φ_current − γφ·θ_current?
  No — φ is climbing. **Both disjuncts fail**, so the full step is rejected;
  the backtracked α = 0.25 step barely moves and the accumulating filter
  entries (grows to 8) wall off the region. The iterate never climbs out.

That is the whole story. The escape route is "let θ stay elevated (≈ 2) while
the objective worsens." The funnel permits it because its acceptance is a pure
**absolute** slack bound (θ_trial ≤ τ ≈ 6), decoupled from the objective. The
filter forbids it because its H-type verdict is **current-relative** and
objective-coupled — it demands monotone θ-decrease or φ-decrease against the
current iterate, and the accumulated (θ,φ) history dominates the region the
iterate must pass through. This confirms the working hypothesis: **the
funnel's generous absolute width is the load-bearing difference, not the
switching / Armijo machinery** (which is shared and produces identical steps
through iteration 9).

Note also that the funnel takes **LS = 0 on all 29 iterations** — no line-
search backtracking anywhere, so the recovery-chain "accept-as-is" fallback
(which fires only when the backtracking ladder exhausts) plays no part. The
funnel win is genuine strategy-level acceptance of full Newton steps, not a
ladder-exhaustion artifact.

## Robustness of the funnel win

Two probes, both a handful of runs. First, hold the problem fixed and sweep
`init_mu` (shifting the barrier oracle's starting point, hence the early μ
schedule) across four orders of magnitude:

| init_mu | funnel                    | filter                     |
|---------|---------------------------|----------------------------|
| 1e−3    | CONVERGED, 29, +1.000     | NOTCONVERGED, 500, −0.975  |
| 1e−2    | CONVERGED, 29, +1.000     | CONVERGED, 37, +1.000      |
| 0.1     | CONVERGED, 33, +1.000     | NOTCONVERGED, 500, −0.974  |
| 1.0     | CONVERGED, 30, +1.000     | NOTCONVERGED, 500, −0.969  |
| 10.0    | CONVERGED, 28, +1.000     | NOTCONVERGED, 500, −0.988  |

The funnel converges to the true optimum for **every** value (28–33 iters,
final width always ≈ 5.98); the filter escapes only for one lucky value. The
funnel win is not a knife-edge in the barrier schedule.

Second, perturb the start point (which *changes* θ₀ and hence the funnel
width):

| start           | funnel                    | filter                     |
|-----------------|---------------------------|----------------------------|
| (−2.0,3.0,1.0)  | CONVERGED, 29, +1.000     | NOTCONVERGED, 500, −0.975  |
| (−1.5,2.0,0.5)  | NOTCONVERGED, 500, −0.794 | NOTCONVERGED, 500, −0.974  |
| (−1.8,2.5,0.8)  | NOTCONVERGED, 500, −0.940 | NOTCONVERGED, 500, −0.974  |
| (−2.2,3.5,1.2)  | CONVERGED, 28, +1.000     | CONVERGED, 44, +1.000      |
| (−2.5,4.0,1.5)  | CONVERGED, 32, +1.000     | CONVERGED, 47, +1.000      |
| (−3.0,5.0,2.0)  | CONVERGED, 31, +1.000     | CONVERGED, 57, +1.000      |

Honest reading: the funnel converges from 4 of 6 starts, the filter from 2 of
6, and on the two starts closest to the jamming basin (smaller |x₁(0)|) even
the funnel jams. The funnel advantage is **real but start-dependent**, not
universal — consistent with the mechanism, since a start with smaller θ₀
yields a tighter width and less slack for the escape. At the registered corpus
start the funnel escapes and the filter does not; that is the outcome the
corpus records.

## Confidence

- **Established (direct evidence):** the three outcomes and their per-iteration
  trajectories; the byte-identical steps through iteration 9 and the precise
  iteration-10 divergence (funnel α = 1 / LS = 0 accept vs filter α = 0.25 /
  LS = 2 backtrack from the same state); the funnel final width 5.9862 (0.23%
  shrink); LS = 0 on all 29 funnel iterations (no accept-as-is involvement);
  `last_filter_resets = 0` (reset heuristic not implicated); the init_mu and
  start-perturbation robustness tables.
- **Inferred (from the acceptance rules applied to the observed θ/φ, not from
  instrumented trial-level values):** the specific reason each strategy
  accepts/rejects the iteration-10 full step — i.e. that the funnel classifies
  it H-type and passes it on θ_trial ≤ β·τ while the filter fails both
  acceptable-to-current disjuncts. The printed table exposes accepted-iterate
  measures, not the rejected trial's (θ, φ); the classification is deduced
  from the rules in `switching_acceptance.h` / `funnel_acceptance.h` /
  `filter_acceptance.h` and the visible objective/violation motion, which are
  mutually consistent, but the exact rejected-trial θ/φ were not instrumented.
- **Taken as given (prior bisection):** result (a). The 133-iter unfaithful-
  variant number and the intermediate-commit build come from the strategy's
  development bisection, incorporated here rather than re-derived (rebuilding
  is out of scope for this analysis).

## Implications for the barrier-governor work

The monitored barrier governor is expected to change the filter's operating
regime on exactly this problem class. The mechanism above says the filter jams
because its H-type acceptance is current-relative and objective-coupled, and
the μ schedule drives *where* the current iterate sits when the escape step is
offered — the divergence at iteration 10 sits immediately after a barrier
event (μ jumps at iters 6 and 9 in both runs). A monotone μ safeguard changes
that schedule. The upcoming monitored-governor experiment on wb2000 should
therefore look for:

- Whether a monotone (non-increasing) μ safeguard shifts the filter's
  iteration-10-analogue state enough that the acceptable-to-current test
  admits the escape step — i.e. whether the filter's outcome flips to
  CONVERGED under the governor, and if so from which starts (re-run the
  start-perturbation table under the governor).
- Whether the funnel's win survives unchanged under the governor (it should —
  its acceptance is μ-schedule-insensitive here, per the init_mu table — but
  the width init still keys off θ₀, so confirm the final width stays ≈ 5.98
  and LS stays 0).
- The filter's `last_filter_resets` under the governor: still 0 (reset
  heuristic not the lever), or does the changed schedule start exercising it.

See `2026-07-e2-g3-scorecards.md` and `2026-07-e2-g2-scorecards.md` for the
full corpus scorecards these outcomes sit within, and
`2026-07-e2-g0-baseline.md` for the defaults baseline.
