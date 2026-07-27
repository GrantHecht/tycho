# PSIOPT performance backlog from the solver review

Date: 2026-07-27. Source: a line-by-line review of `src/solvers/psiopt.cpp`,
`src/solvers/psiopt_globalization.cpp`, and `src/solvers/psiopt_print.cpp`
looking for wasted work in the solver's per-iteration bookkeeping.

**Standing bar:** every entry below requires dedicated benchmark and
CBWR-pinned iteration evidence in its own PR before being acted on. None of
them met the >90%-confidence-of->5%-improvement bar for the maintenance
batch this note accompanies, so nothing perf-motivated shipped in that batch.

## Why this backlog is mostly not worth chasing

Two structural facts bound how much any of this can matter. First, sparse
factorization and NLP evaluation (objective/gradient/Jacobian/Hessian) plus
KKT assembly — including the unavoidable `std::fill_n` zeroing pass over
`KKTmat.nonZeros()` at the top of `PSIOPT::eval_nlp` (`psiopt.cpp`) — are two
to four orders of magnitude more expensive than anything in `alg_impl`'s
surrounding bookkeeping. Second, `PSIOPT::SolveResult::misc_time()`
(`psiopt.h`) is the direct empirical upper bound on the bookkeeping-only
items below (everything except the first two entries: the terminating-iterate
cost lives in evaluation time, and the print path has its own `print_time_`
metric); read it on the `PolarLT_256seg` benchmark before spending any
effort here — it already tells you the ceiling bookkeeping-only fixes can
recover.

## Terminating iterate evaluates the full KKT system it will not use

**Mechanism:** `PSIOPT::alg_impl`'s per-iteration loop calls `eval_nlp`
(which computes objective, gradient, Jacobian, *and* the Lagrangian Hessian,
then zeroes and scatters the entire `KKTmat` value array) before running its
pre-factorization convergence check (`fill_residual_info` followed by
`converge_check`, both called ahead of the KKT factor/solve step). A
terminating iterate — one that trips `CONVERGED`, `ACCEPTABLE`, or
`DIVERGING` at this check — only ever needed `prim_grad()`, `eq_cons()`,
`iq_cons()`, and the slack/multiplier norms to reach that verdict; the
Hessian assembly and the KKT scatter it just paid for are dead work on that
iterate. This ordering itself is intentional and already documented as
bit-identical to the historical post-line-search check position (see the
comment above `fill_residual_info` in `alg_impl`) — the finding is only that
the *evaluation mode* underneath it does more than the check needs.

**Workload:** once per solve phase (twice for a typical `solve_optimize`
call), on whichever iterate happens to terminate the phase.

**Estimated improvement:** the Hessian-plus-assembly share of `eval_kkt` is
roughly 0.5-0.7 of its cost; dividing that by the phase's iteration count
gives approximately 0.7-1.0% of solve time, rising to about 2% on a short
phase (N of roughly 12-15 iterations). This is the largest single item in
the review.

**FP-neutrality:** bit-identical, if a fix is achievable at all.

**Confidence:** medium on the estimate itself (the Hessian-cost share is
assumed from `eval_kkt`'s structure, not measured directly). High confidence
that the naive fix — evaluate residuals first, cheaply, and only pay for the
rest on a non-terminating iterate — is net-negative: `eval_rhs` (the
residuals-only path used elsewhere) still recomputes the full gradient and
Jacobian, so splitting the evaluation this way would duplicate that work on
every one of the `N-1` non-terminating iterations to save it on the one that
terminates. A real fix needs an NLP-layer evaluation mode that returns
residuals without gradient/Jacobian/Hessian, which does not exist today, and
PSIOPT internals are gated for human review regardless.

## Per-iteration console print does 25 locked, unbuffered writes

**Mechanism:** `PSIOPT::print_last_iterate` (`src/solvers/psiopt_print.cpp`),
called from three sites in `alg_impl` (`psiopt.cpp`), builds one table row
out of roughly 25 independent `fmt::print` calls — twelve data fields plus
about thirteen separator and color-escape prints — each of which takes the
stdout lock and runs its own format pass, with the row terminated by `\n`
under line buffering (one `write(2)` plus roughly ten SGR color escapes).
Collapsing this into a single `fmt::memory_buffer` plus one `fwrite` would
preserve byte-identical output while cutting the lock/format/write overhead
to one pass.

**Workload:** every iteration, but only when `settings_.print_level_ == 0` —
which is the shipped default (`int print_level_ = 0;` in `psiopt.h`).

**Estimated improvement:** zero on the benchmarked configuration (the
benchmark harness runs at a higher print level, so `bench_track.sh` cannot
validate a fix either way). Plausibly negligible (roughly 0.3-0.5%) on
`PolarLT_256seg`, but plausibly 3-10% of wall time on small (Brach-scale)
problems run with a terminal attached, where per-iteration overhead is a much
larger fraction of total solve time.

**FP-neutrality:** not FP-relevant; any coalescing must preserve
byte-identical console output (including the color escapes).

**Confidence:** medium — the mechanism and the default-print-level exposure
are both real, but the magnitude is unmeasured and configuration-dependent.
Any attempt at this should gate on `print_time_` versus `SolveResult`'s total
time on a small problem (e.g. `Brach_16seg`) at `print_level=0`, not on
`bench_track.sh`, since the benchmark suite's fixed print level cannot see
this cost at all.

## A cluster of sub-0.5% findings

Each of the following is real but individually too small to justify a PR on
its own; several only exist on off-default configurations.

- **Redundant residual/complementarity recomputation.** `alg_impl` calls
  `fill_residual_info` twice per iteration — once before factorization
  (through the pre-factorization convergence check) and once after the
  barrier-parameter update, via `fill_iter_info` — and nothing writes any
  block the second call reads in between (the barrier update only touches
  `dual_grad()`, which is disjoint from `prim_grad()`/`all_cons()`). Because
  `fill_residual_info` itself calls `PSIOPT::complementarity`, and `alg_impl`
  also calls `complementarity` directly earlier in the same iteration to feed
  the barrier-parameter oracle, `complementarity` runs three times per
  iteration on the same `(slacks, iq_lmults)` pair, recomputing nine
  reductions bit-for-bit. Two off-default call paths would need their own
  guard if this is ever fixed: `WatchdogRecovery::on_step_rejected`'s
  `kTrialRevert` outcome and `PSIOPT::enter_feasibility_restoration`
  (`psiopt_globalization.cpp`, `psiopt.cpp`). Estimated at roughly
  2.5×`kkt_dim` doubles of redundant work per iteration — about 0.05% on
  `PolarLT_256seg` up to about 0.2% on `Brach_16seg`. Bit-identical either
  way (same function, same inputs). High confidence on the mechanism and
  neutrality, high confidence the payoff stays under 0.5%. The fix shape
  would be to thread the already-computed `avgcomp`/`maxcomp` into the second
  call, or to skip it entirely behind a "point unchanged since the first
  call" flag.

- **Recovery re-drives recompute invariant current-point measures.**
  `ClassicMeritAcceptance::compute_penalties` (`psiopt_globalization.cpp`) is
  called once per line-search re-drive to get the current-point penalty
  terms (`init`), even on backtracking rungs and SOC corrections where the
  current point has not changed since the previous re-drive — only the
  generic `dirderiv` genuinely changes for SOC, since its direction is
  `dxsl_soc`. Estimated at under 0.1% even with SOC enabled and a high
  rejection rate, and only reachable when SOC or extended backtracking is
  configured (both default off). Bit-identical if hoisted. High confidence.

- **Fresh heap allocations in the rejection recovery path.**
  `SocRecovery::on_step_rejected` allocates four fresh `Eigen::VectorXd`
  buffers per call (two of them `kkt_dim`-sized: `rhs_soc`, `dxsl_soc`), and
  `ExtendedBacktrackRecovery::on_step_rejected` allocates one
  (`dxsl_ext`) — both in `psiopt_globalization.cpp`. This is inconsistent
  with the rest of the solver's `*_scratch_`-member discipline (compare
  `BacktrackingLineSearch`'s `resto_eq_shift_scratch_`, which follows the
  precedent correctly). Dead by default (`max_soc_ = 0`,
  `ls_extended_iters_ = 0`); under 0.2% of a SOC-enabled solve, 0% by
  default. Bit-identical. High confidence. Worth doing opportunistically the
  next time SOC code is touched, for scratch-discipline consistency rather
  than for the performance win.

- **Materialized temporary in the proximal-restoration objective.**
  `ProximalSwitchRestoration::proximal_objective` (`psiopt_globalization.cpp`)
  materializes a `primal_vars_`-sized `Eigen::VectorXd delta` purely to feed
  `diagonal_.dot(delta.cwiseProduct(delta))`; its sibling
  `add_proximal_gradient` right below it already uses the no-temporary
  expression form (`diagonal_.cwiseProduct(primals - x_r_)`). The function
  has four call sites — one per iteration from the `eval_nlp` seam
  (`psiopt.cpp`) plus three per line-search rung
  (`psiopt_globalization.cpp`) — so the allocation recurs often when
  proximal restoration is active, but the mode is off by default
  (`restoration_mode_` unset). Under 0.2% even of a restoration-heavy solve,
  0% by default. FP-neutrality is unverified rather than assumed bit-identical:
  Eigen's packet traversal for an expression operand versus a materialized
  vector was not checked, so a fix would need to gate on a CBWR
  iteration-count comparison, not just an eyeball diff. High confidence the
  allocation is real; medium confidence on bit-identity.

- **Standalone negation pass after the KKT solve.** `alg_impl` and
  `ClassicAdaptiveGovernor::update_barrier` both do `DXSL =
  kkt_sol_.solve(RHS); DXSL = -DXSL;` — the split correctly avoids the
  `EvalBeforeNestingBit` temporary Eigen would otherwise force, but the
  negation remains its own full pass over the vector. It could in principle
  fold into the fraction-to-boundary block scalings
  (`BacktrackingLineSearch::max_step_to_boundary`,
  `psiopt_globalization.cpp`), except that routine reads *signed* `dSLI`, so
  folding the negation in would require flipping both the comparison and the
  ratio in the fraction-to-boundary rule. One `kkt_dim` read+write per
  iteration, roughly 128 KB on `PolarLT_256seg` — under 0.05%.
  Bit-identical in principle (negation is exact); the risk is entirely in
  the sign-flip rewrite of the fraction-to-boundary rule, not in the
  arithmetic. Medium confidence on the fold's bit-identity, high confidence
  the payoff is under 0.1% regardless — not worth the sign-bug risk at this
  size.

- **Complementarity min/max could fuse with the product pass.**
  `PSIOPT::complementarity` and its token-identical copy
  `ClassicAdaptiveGovernor::complementarity` (`psiopt_globalization.cpp`)
  compute `stli_scratch_ = S.cwiseProduct(LI)` and then take `minCoeff()`,
  `maxCoeff()`, and `sum()` as three separate passes where two would do. The
  `sum()` reduction must not be fused with the others — it feeds
  `mpc_mu`/`loqo_mu`, and a ULP-level reordering there moves `mu` on
  subsequent iterations, per the in-file warning at both copies — but
  min/max could fold into the product pass. Under 0.03% per call. Min/max
  fusion is bit-identical for finite data, but `Eigen::minCoeff`/`maxCoeff`
  and a hand-written loop disagree on NaN propagation, and a NaN slack
  product is exactly the signal `converge_check`'s NaN/Inf branch watches
  for. High confidence; dominated by the redundant-recomputation item above,
  since it runs three times per iteration for the same reason.

- **`converge_check` copies `IterateInfo` by value.**
  `PSIOPT::converge_check` (`psiopt.cpp`) takes a `std::vector<IterateInfo>&`
  but immediately copies `IterateInfo last = iters.back();` (roughly 232
  bytes) and is called twice per iteration, where a const reference would
  do; the per-iteration `Citer` is also pushed, popped, and re-pushed on some
  exit paths for a few more ~232-byte copies. Roughly 1 KB of memcpy per
  iteration — under 0.01%. Bit-identical. High confidence; recorded mainly
  because the history vector's allocation discipline was in scope for the
  review and, unlike the copies, is already correct:
  `iters.reserve(settings_.max_iters_)` is set once up front, so there is no
  reallocation churn from repeated growth.

- **An always-allocated but conditionally-dead scratch vector.** `alg_impl`
  allocates `Temp` (`kkt_dim`-sized, `psiopt.cpp`) every phase and threads it
  through `update_barrier`'s signature, but it is written and read by nobody
  under the default `LOQO` barrier mode
  (`opt_bar_mode_ = BarrierModes::LOQO` in `psiopt.h`) — only the `PROBE`
  predictor path uses it. It cannot be deleted without an interface change
  to `update_barrier`. Under 0.01% per solve (two to three allocations per
  solve). Bit-identical. High confidence; this is noise, and the real cost
  of touching it is the interface change, not the allocation.

## Resolved by removal: dead diagnostic norms

Four write-only diagnostic L2 norms plus `barr_norm_err_` in `alg_impl`
(`psiopt.cpp`) had one writer each and zero readers; `max_e_mult_` and
`max_i_mult_` are read only by the console print path
(`psiopt_print.cpp`), and only when `wide_console_` and `print_level_ == 0`
are both set. This was confirmed by grep, not inference, and estimated at
roughly 3.4×`kkt_dim` doubles of dead read/write traffic per iteration —
about 0.05% on `PolarLT_256seg` up to about 0.2% on `Brach_16seg`,
bit-identical either way. Because this was a pure dead-code removal rather
than a behavior-preserving optimization requiring its own benchmark gate, it
was already deleted earlier on this branch; the estimate above documents why
the removal was worth doing; it is not itself the justification (the
justification is that the fields had no reader).

## Non-finding: component extraction has no measurable runtime cost

The globalization refactor that split `alg_impl`'s inline logic out into
named strategy objects (`GlobalizationMechanism`, `AcceptanceStrategy`,
`BarrierGovernor`, `RecoveryChain`, and their concrete implementations in
`psiopt_globalization.cpp`) was reviewed for indirect-call overhead. On the
default path this comes to roughly five to eight indirect calls per
iteration, on the order of 40 ns total — about 0.0002% of a `PolarLT`
iteration. The `restoration_` nullptr checks throughout `alg_impl`
(`this->restoration_ && this->restoration_->is_active()`, e.g. around line
1826 and 1903) short-circuit on `nullptr` and never reach a vtable;
`SolverContext` is constructed once per `alg_impl` call (i.e. once per
phase), not once per inner call; and `KKTVector` accessors — used
pervasively, e.g. by `ClassicMeritAcceptance::classic_line_search`
(`psiopt_globalization.cpp`) on every call — are free after inlining. Some
of the helper bodies duplicated across `PSIOPT` and the extracted governor
classes (the `complementarity` copies discussed above are the clearest
example) are a *source* duplication, not a runtime one — each copy compiles
to its own inlined code at its own call site either way. **No runtime cost
is attributable to the component extraction itself.** This is worth stating
explicitly because several of the smaller findings above predate the
extraction and are sometimes mistaken for a consequence of it; they are not.
