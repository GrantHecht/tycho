# Trust-region acceptance mechanism — investigation, decision to cut, and the composite-step successor path

Date: 2026-07-24. Status: **cut without implementation** (decision recorded here so it
is not re-litigated from scratch; reversal condition below). This was the planned
next stage of the PSIOPT globalization series, spec-marked "explicitly experimental:
no validated TR×IPM preset exists anywhere; the corpus decides whether it stays;
cuttable without losing foundations."

## 1. What was investigated

The proposed mechanism: an ℓ∞ trust region on the primal Newton step behind the
existing `GlobalizationMechanism` interface — effective step cap = min(TR radius,
fraction-to-boundary step), with Uno's radius machinery (initial radius 10,
shrink/grow ×2, activity tolerance 1e-6, minimum radius 1e-12, re-entry floor 1e-4).

Two source-fidelity investigations were run before implementation (Uno pinned at
`7481abe4`; PSIOPT at `287d71e9`):

1. **Uno's trust region is not a step cap.** `TrustRegionStrategy` bounds the step by
   tightening the *subproblem's box constraints* (`Subproblem::set_variables_bounds`:
   lower = max(−radius, lb−x), upper = min(radius, ub−x)) and **re-solving** on
   rejection — the direction genuinely changes as the radius shrinks. There is no
   post-hoc clipping of a computed direction anywhere in the TR path.
2. **Uno never pairs TR with its interior-point method.** Its `ipopt` preset is
   LS+IPM; its three TR presets (`filtersqp`, `funnelsqp`, `filterslp`) all use the
   active-set inequality-handling method. The two direct-KKT subproblem solvers used
   by the IPM route (`EQPSolver`, `WoodburyEQPSolver`) both contain an explicit
   runtime guard: `throw std::runtime_error("The direct linear solver does not
   support a trust region")` on any finite radius. The TR×IPM combination is
   reachable only via raw options, silently reroutes the barrier subproblem to an
   active-set QP backend, and shows no evidence of ever having been exercised.
3. **The only architecture compatible with PSIOPT's direct-LDLT core is step
   clipping**: keep the Newton direction, scale its primal block onto the radius.
   The full integration map for that variant was worked out (single seam in the
   fraction-to-boundary scaling so all eight step consumers — main commit, barrier
   predictor, SOC, soft-feasibility, elastic lockstep, extended backtrack, merit
   directional derivative, watchdog — inherit the cap consistently). It is
   implementable at low cost; that is not why it was cut.

## 2. Why it was cut

- **Clipping removes the ingredient that makes trust regions robust.** A real TR's
  power is that the direction bends toward the steepest-descent/Cauchy direction as
  the radius shrinks — a bad Newton direction gets *replaced*. Clipping keeps the
  same direction at every scale; a bad direction stays bad in smaller bites. What
  remains is functionally a backtracking line search with persistent step-length
  memory and a cruder (full-step-or-shrink) acceptance rule. This is exactly why
  line-search IPMs historically skipped TR.
- **Its robustness niche is already served.** The proximal regularization mode is
  the implicit trust region (Levenberg–Marquardt correspondence, persistent shift
  memory), and it delivered the degenerate-tier corpus rescues (wb2000,
  near_infeasible). Restoration provides the infeasibility certificate; SOC and the
  watchdog cover rejection recovery. The marginal class clipping-TR targets —
  repeated full-step rejection — overlaps all of these.
- **The pairing has no external validation anywhere** (Uno's source actively fences
  it off; no published benchmark of TR-as-acceptance × IPM was found), so the prior
  on corpus wins is weak, and under the performance-parity standard the likely
  "mostly wash" outcome would mean cutting it after paying for its implementation,
  gate, corpus campaign, and permanent settings/docs/evaluation-matrix surface.
- **Precedent:** the elastic/penalty relaxation was likewise cut without
  implementation once indirect corpus evidence showed its tier well-served, with a
  named reconsideration checkpoint. This decision follows the same pattern.

**The honest counterarguments, recorded so the cut is not one-sided:** (a) the
persistent radius memory is the same design ingredient (cross-iteration persistence)
that made the proximal mode's Tycho-original dynamics win, applied to the
step-length axis where nothing in the current stack holds memory; (b) the funnel
acceptance strategy's dedicated global-convergence theory is proven for its
*trust-region* variant, so funnel×TR was the one theory-supported pairing. Neither
outweighed the reasons above, but a future reader should know the case for the
mechanism was real.

## 3. The successor path: the Waltz-hybrid composite-step architecture

**This is the design to reach for if trust-region-grade robustness is ever actually
needed — not a revival of step clipping.**

Reference: Waltz, Morales, Nocedal, Orban, *An interior algorithm for nonlinear
optimization that combines line search and trust region steps*, Math. Program.
107:391–408, 2006 (= the Knitro Interior/Direct ↔ Interior/CG hybrid). Component
machinery: Byrd–Hribar–Nocedal, SIAM J. Optim. 9(4):877–900, 1999 and
Byrd–Gilbert–Nocedal, Math. Program. 89:149–185, 2000 (the Byrd–Omojokun
composite-step TR-IPM = Knitro Interior/CG).

The architecture, and why it fits PSIOPT:

- **Line-search direct steps are always tried first** (PSIOPT's existing path,
  unchanged). The steplengths are monitored; when they fall below a threshold, or
  negative curvature / factorization trouble appears, the line-search iteration is
  **discarded and replaced by one composite-step trust-region iteration**:
  - *Normal component*: reduce ‖c‖ within ξΔ (ξ≈0.8) — handles rank-deficient /
    inconsistent constraint linearizations by construction.
  - *Tangential component*: reduce the objective model in the null space of the
    constraint Jacobian via projected CG (Steihaug/Lanczos termination) — sees
    negative curvature directly instead of regularizing it away.
  - Acceptance by an ℓ1/ℓ2 merit ratio test; fraction-to-boundary folded into the
    subproblems (κ≈0.005 in the Nocedal–Öztoprak–Waltz description).
  - Guaranteed Cauchy decrease for both feasibility and optimality; the hybrid's
    global convergence properties come from the TR component.
- **In PSIOPT terms this is a recovery-chain link, not a co-equal mode**: the
  natural dock is the existing recovery dispatch (after SOC / extended backtrack /
  watchdog, before or replacing the feasibility switch), triggered by exactly the
  signals the chain already observes (rejection-ladder exhaustion, inertia-ladder
  distress). The direct-LDLT fast path stays untouched.
- **Honest cost assessment**: this is architecturally a second step engine
  (projections onto the Jacobian null space, a CG loop, its own convergence
  behavior to validate). It is a program-scale effort, not a stage-scale one, and
  its per-iteration cost is substantially higher than a direct solve — which is
  precisely why the hybrid pattern (direct steps normally, composite steps as
  rescue) is the production-validated form and a permanent-TR mode is not.

## 4. Activation / reversal condition

The evaluation campaign's shared-transcription reference harness (same sparse NLP
handed to both PSIOPT and Ipopt) is the designated checkpoint. **If it exposes a
systematic robustness gap — a problem class where steps keep failing despite
regularization + restoration + SOC + watchdog — schedule the composite-step
recovery engine as its own program, scoped from §3.** If no such gap appears, the
matter stays closed. Step clipping does not return in either branch; if new
evidence ever suggests a cheap step-cap has value after all, the burden is a
corpus differential demonstrating a win the existing stack cannot produce.
