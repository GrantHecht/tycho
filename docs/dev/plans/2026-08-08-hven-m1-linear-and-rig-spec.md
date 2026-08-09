# hven M1 — Spec, Part 1: the `linear/` Surface and the Golden-Rig Build

**Status: BOTH PARTS FROZEN.** **Part B [FROZEN as amended]** (SQP freeze
review `tycho_sqp/docs/notes/2026-08-08-hven-m1-freeze-review-sqp.md`: T8
amendment accepted; B.2 pin-mechanism tightening). **Part A [FROZEN as
amended]** (revision 1 — epoch-based co-owning handles, partial-solve
surface + phase-33x rule, A.5 migration note — stamped in the review's
addendum, which also contributed the `supports_partial_solve()`
amendment, the T2b trace, and the T4b epoch sharpener, all absorbed
below). M1 is fully specified; execution gates on the Appendix-A close
sign-off. Post-freeze changes ride the freeze gate (small, fast, never
silent). Everything else about M1 lives in the companion plan
(`2026-08-08-hven-m1-plan.md`) and needs no SQP review.

---

# Part A — the `hven::linear` surface

## A.1 Scope and shape

One namespace, `hven::linear`, replacing two seams: psiopt's Eigen-style
`PardisoLU/LLT/LDLT` + `AccelerateLDLT` classes, and sqp's phase-numbered
`KktSystem` pair. Two backends behind one surface: MKL Pardiso
(Linux/Windows) and Apple Accelerate + the LAPACKE shim (macOS), selected
at configure time exactly as both codebases do today. The Schur-border
stack (border ops, bordered EQP) is OUT of scope — it composes this
surface from `kkt/` (rev B §6.6).

Two components:

1. **`SymmetricFactor`** — sparse symmetric-indefinite LDLT (the
   workhorse; Pardiso mtype = -2 / Accelerate LDLTTPP), plus the LU/LLT
   variants psiopt's presets can select.
2. **`DenseSymmetricFactor`** — the small dense LDLT (LAPACKE
   `dsytrf`/`dsytrs`; Accelerate via the migrated shim) that the border
   stack layers on the sparse oracle. Named explicitly per SQP response
   §3.6: it is a second, small backend surface with the same evidence
   discipline, not an implementation detail.

## A.2 Core types

```cpp
namespace hven::linear {

using Index = std::int64_t;   // one index type across the library (core/)

enum class FactorKind { kLDLT, kLLT, kLU };

// Backend-honest inertia + perturbation evidence (rev B §6.3).
struct InertiaEvidence {
    enum class State {
        kObserved,      // counts below are valid
        kQueryFailed,   // backend query ran and failed — counts INVALID
        kUnavailable    // backend/factor-kind cannot provide inertia
    };
    State state = State::kUnavailable;
    Index n_pos = -1, n_neg = -1, n_zero = -1;   // valid iff kObserved
    bool zero_is_derived = false;   // MKL: n_zero = dim − n_pos − n_neg,
                                    // marked derived; Accelerate: native
    // Perturbation evidence is a backend-qualified OPTIONAL:
    // present with Pardiso semantics on MKL; nullopt on Accelerate —
    // ABSENT, never zero-filled. (Fixes fabrication defect 1.)
    std::optional<Index> perturbed_pivots;
};

struct FactorizeOutcome {
    enum class Status { kOk, kBackendError };
    Status status = Status::kBackendError;
    int backend_code = 0;           // raw backend error, 0 on kOk
    InertiaEvidence inertia;        // kQueryFailed is representable and
                                    // is NOT a factorize failure by itself
};

struct SolveInfo {
    // Populated where the backend reports it; absence is honest.
    std::optional<Index> refinement_iters;
};
```

**Verdict discipline:** the surface reports evidence; the *drivers* derive
verdicts (singular, wrong inertia, IC acceptance) exactly as both engines
do today. `kQueryFailed` is distinguishable from "zero class present" by
construction, which is the driver-side obligation P5 asserts. (Fixes
fabrication defect 2: today's `SparseGetInertia`-failure zero-fill.)

## A.3 `SymmetricFactor` — lifecycle and contract

```cpp
class SymmetricFactor {
  public:
    struct Options {
        FactorKind kind = FactorKind::kLDLT;
        int num_threads = 0;          // 0 = backend default; see A.6
        int pivot_perturb_exp = 8;    // Pardiso 10^-k static pivot
                                      // perturbation (psiopt default);
                                      // Accelerate: pivot tolerance
                                      // mapping documented per backend
        int max_refinement_iters = 0; // backend iterative refinement
        // The full named-option set is the union both engines actually
        // consume, finalized by the M1 consumed-surface audit (B.4).
        // There is NO raw-iparm escape hatch: any need to touch an
        // unlisted iparm is a surface change and rides the standing
        // MKL-iparm human-review gate.
        //
        // SQP-side consumed-surface list (freeze review §1, from
        // source; the audit's checklist input): writes iparm[34]=1
        // (zero-based CSR — backend-adapter internal, not an option)
        // and iparm[7] (covered by max_refinement_iters, PLUS the
        // phase-33x force-zero-restore rule in solve_partial below);
        // reads iparm[13], iparm[21]/[22] (covered by InertiaEvidence/
        // SolveInfo); rides the default on iparm[9] (psiopt sets it;
        // the named option covers both). Confirmed: no escape hatch
        // needed by the walk, SSN kernel, or border stack.

        // POST-FREEZE AMENDMENT (implementation-discovered, ruled by
        // Grant's delegate 2026-08-08; rides the freeze gate): the IPM
        // engine's proven configuration also sets Pardiso ordering
        // (iparm[1] = nested-dissection METIS) and weighted matching
        // (iparm[12] = on), which the SQP arm's pins were derived
        // WITHOUT. Both become named options with don't-write-by-
        // default semantics — at the default, hven does not touch the
        // iparm entry at all, so pardisoinit's value survives exactly
        // (maximally pin-preserving for the SQP arm):
        // Enum locked backend-neutral after an independent design
        // check: kMinimumDegree added NOW because the IPM engine
        // exposes it as a live user-facing choice (its MINDEG mode) —
        // omitting it contradicted this spec's own finalized-union
        // claim and would have reopened the public enum later.
        enum class Ordering { kBackendDefault, kMinimumDegree,
                              kNestedDissection,
                              kParallelNestedDissection };
        Ordering ordering = Ordering::kBackendDefault;
        // Locked mapping (Mac hardware verification + the availability
        // guard are the ONLY deferred pieces):
        //   kBackendDefault          MKL: don't write iparm[1]
        //                            Accelerate: SparseOrderDefault
        //                            (documented as AMD for symmetric)
        //   kMinimumDegree           MKL: iparm[1]=0
        //                            Accelerate: SparseOrderAMD
        //   kNestedDissection        MKL: iparm[1]=2
        //                            Accelerate: SparseOrderMetis
        //   kParallelNestedDissection MKL: iparm[1]=3
        //                            Accelerate: SparseOrderMTMetis,
        //                            downgraded to SparseOrderMetis
        //                            where the OS lacks it (the IPM
        //                            engine's proven guard; passing it
        //                            unsupported raises a backend
        //                            parameter error)
        // MIGRATION HAZARD (recorded for the IPM adapter): the IPM
        // engine's Accelerate default is Metis, but Accelerate's own
        // default is AMD — the adapter must explicitly request
        // kNestedDissection or migration silently changes the ordering.
        bool weighted_matching = false;                  // iparm[12]: 1 when true
        // Accelerate semantics — REVISED after the consumed-surface
        // audit falsified this block's original premise (the IPM
        // engine's Accelerate path calls set_order() on EVERY
        // invocation, so throw-on-non-default would have broken the
        // Mac migration): the ORDERING option maps to Accelerate's
        // own ordering-method control (the audit shows the seam
        // already exercises it); the exact enum-to-SparseOrder mapping
        // is fixed at M2 against the IPM adapter's need and verified
        // on the Mac leg. weighted_matching keeps THROW on non-default
        // (Accelerate has no matching analogue — never silently
        // ignored). psiopt's iparm[4] (store_perm) is deliberately NOT
        // carried: its only consumer is dead code, audit-confirmed.
    };

    explicit SymmetricFactor(Options opts);

    // --- lifecycle (SQP req. 1, 2) ---
    // Pattern-bound symbolic analysis. Pattern is upper-triangle CSR
    // (matching the model contract's Hessian convention); the pattern
    // hash (the library-wide FNV-1a) is captured here and becomes this
    // instance's structural key.
    void analyze(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);

    // Numeric factorization into the EXISTING symbolic. Guaranteed to
    // never re-analyze (T3/P1 pin this: re-analysis count observable
    // via counters()). Throws std::invalid_argument if A's pattern hash
    // differs from the analyzed key (contract violation, not a numeric
    // outcome).
    FactorizeOutcome factorize(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);

    // Unlimited solves with the existing factorization, freely
    // interleaved with other instances' operations (T2: the border
    // stack interleaves the sparse oracle with a dense border factor).
    SolveInfo solve(ConstEigenRef<Eigen::VectorXd> rhs,
                    EigenRef<Eigen::VectorXd> x) const;
    SolveInfo solve(ConstEigenRef<Eigen::MatrixXd> RHS,    // multi-RHS
                    EigenRef<Eigen::MatrixXd> X) const;

    // Partial (phase-split) solves — the border stack's oracle surface
    // (rev B §6.1). INTERNAL CORRECTNESS RULE (from the SQP freeze review §1,
    // kkt_system.h:327-335): MKL phase-33x partial solves require
    // iparm[7] == 0 — the implementation saves/zeroes/restores it
    // around every partial solve, unconditionally. This is a Pardiso
    // correctness constraint, not an option; refinement configuration
    // never leaks into partial solves.
    // Phase set CONFIRMED at the freeze review: {forward (331),
    // diagonal (332), backward (333)}, composed
    // backward(diagonal(forward(rhs))) — exactly the split the border
    // stack consumes (kkt_system.h:53-66, schur_complement.h:55-56).
    enum class SolvePhase { kForward, kDiagonal, kBackward };
    SolveInfo solve_partial(SolvePhase phase,
                            ConstEigenRef<Eigen::VectorXd> rhs,
                            EigenRef<Eigen::VectorXd> x) const;

    // Composability predicate (freeze-review addendum amendment,
    // absorbed verbatim — guards a documented silent-wrong-answer
    // hazard):
    // `bool supports_partial_solve() const;` — true iff the CURRENT
    // factorization's partial solves compose to the full solve. MKL:
    // requires perturbed_pivots == 0 (composition under perturbation
    // silently diverges by O(1e8) with no error raised; kkt_system.h:
    // 68-79, derivation in the SQP 2026-07-27 pardiso-inertia findings
    // note). Accelerate: perturbation evidence is ABSENT, so
    // composability is unverifiable — the predicate returns FALSE
    // (conservative; the border stack degrades to full solves on Apple
    // until an Accelerate-side verification exists — a named checklist
    // item, never a fabricated `true`). Matching/scaling caveat carries
    // verbatim: composition under active matching/scaling is
    // unexercised at scale.
    bool supports_partial_solve() const;

    InertiaEvidence inertia() const;   // evidence of the LAST factorize

    // --- counters (core/ counters contract; the rig's asserted currency) ---
    struct Counters {
        Index analyze_count = 0;      // T1/T3/P1 assert this
        Index factorize_count = 0;
        Index solve_count = 0;
    };
    const Counters& counters() const;

    // --- shared handles (SQP req. 1; T4/T4b) ---
    // REVISED per the SQP freeze review §2 (share-as-handoff contradicted
    // the shipped kHot lifecycle, qp_engine.h:2093-2115): share() returns
    // a CO-OWNING, STALE-DETECTABLE handle and does NOT empty this
    // engine. The originator keeps working — continuation's hot path
    // reuses the same engine and factorization after emission, paying
    // nothing.
    //
    // Staleness is detected, not prevented, via an EPOCH (the surface's
    // name for the shipped generation stamp):
    //   - SymmetricFactor::epoch() is live — bumped by every successful
    //     factorize().
    //   - Factorization::epoch() is fixed at emission, and share()
    //     records the emitter's COMMITTED epoch, never a live/
    //     mid-operation read (the forged-handle lesson,
    //     qp_engine.h:2307-2333).
    //   - A handle's solve() reflects the CURRENT numeric state of the
    //     shared backend session; contract: handle solves are only
    //     meaningful while no co-owner has refactorized. Consumers
    //     requiring emission-time numerics compare epochs first.
    // The handle co-owns all backend resources: it remains valid after
    // the emitting engine is destroyed or idle; release on last owner.
    // T4 pins the 0-ULP no-interleaving case; T4b pins the staleness
    // contract (emit -> refactorize -> adopt refuses numeric reuse).
    std::shared_ptr<const Factorization> share();
    std::uint64_t epoch() const;   // live committed epoch

    // errors: T5/T6 rules — misuse (solve before factorize, size
    // mismatch, pattern-hash mismatch) THROWS with a formatted message;
    // numeric outcomes are returned, never thrown.
};

// The shared, read-only factorization: owns its backend session.
class Factorization {
  public:
    SolveInfo solve(ConstEigenRef<Eigen::VectorXd> rhs,
                    EigenRef<Eigen::VectorXd> x) const;
    SolveInfo solve(ConstEigenRef<Eigen::MatrixXd> RHS,
                    EigenRef<Eigen::MatrixXd> X) const;
    InertiaEvidence inertia() const;
    std::uint64_t pattern_hash() const;   // the structural key it was
                                          // built under (WarmStart's
                                          // structure_hash currency)
    std::uint64_t epoch() const;          // fixed at emission = the
                                          // emitter's committed epoch
    // Thread-safety: solves on one Factorization are NOT internally
    // synchronized (Pardiso handle semantics); concurrent use requires
    // external serialization. Documented, asserted nowhere.
};

} // namespace hven::linear
```

**Adoption note (kHot ingest):** `SymmetricFactor::adopt(
std::shared_ptr<const Factorization>)` (static factory) validates **both
conjuncts before any reuse** — pattern hash AND epoch (the analogue of
the shipped reuse conjuncts) — and **degrades on mismatch, never lies**:

- hash match + epoch match → full reuse: solves route through the
  handle; a later `factorize()` reuses the symbolic (no re-analysis).
- hash match + epoch MISMATCH → symbolic-only reuse: numeric reuse is
  REFUSED (the handle's numerics are stale); the adopter's first
  `factorize()` reuses the symbolic without re-analysis.
- hash mismatch → fresh `analyze()`; the handle contributes nothing.

T3+T4+T4b jointly pin this ladder: adopt-then-solve is 0-ULP the
handle's solve (T4, no interleaving); adopt-then-factorize does not
re-analyze (T3-class); emit → originator refactorizes → adopt refuses
numeric reuse on the epoch mismatch while symbolic reuse stays legal
(T4b — the scenario the shipped generation stamp exists for).

## A.4 `DenseSymmetricFactor`

```cpp
class DenseSymmetricFactor {   // LAPACKE dsytrf/dsytrs; Accelerate shim
  public:
    void factorize(ConstEigenRef<Eigen::MatrixXd> A);   // small border blocks
    void solve(ConstEigenRef<Eigen::MatrixXd> RHS, EigenRef<Eigen::MatrixXd> X) const;
    // Same error discipline; no inertia surface (the border stack does
    // not consume one today — added only if the consumed-surface audit
    // finds a reader).
};
```

## A.5 Per-backend semantics table (normative, shipped as a doc page)

| Field / behavior | MKL Pardiso | Accelerate |
|---|---|---|
| inertia counts | `iparm[21]/[22]`; zero class DERIVED (`dim − p − n`), `zero_is_derived = true` | `SparseGetInertia` native 3-way (LDLTTPP) |
| inertia on non-LDLT kinds | `kUnavailable` | `kUnavailable` |
| inertia query failure | n/a (fields always populated by Pardiso on success) | `kQueryFailed` — counts invalid, NEVER zero-filled |
| `perturbed_pivots` | present (Pardiso perturbed-pivot counter) | **absent** (`nullopt`) — Accelerate has no counter; absence is the honest state |
| `refinement_iters` | present (Pardiso counter) | per what Accelerate honestly reports; absent otherwise |
| degradation direction | reference semantics | documented per row above; a Mac reading can only be *less* informative, never differently-valued |

This table is the contract the rig's Mac arms assert (T5-Mac asserts
`perturbed_pivots` ABSENT — the assertion today's `ppivs() == 0` would
fail; P4/P5 are the docket members).

**A.5 is NEW LAW, not a description of today (freeze review §3):** the
SQP side's current Accelerate shim maps Apple's ZERO-pivot count into
`num_perturbed_pivots()` — a *differently-valued* reading that this
table's degradation rule ("less informative, never differently-valued")
newly forbids. Migration obligation carried to the M3 plan by name: at
the M3 retarget, both verdict copies (`detail::inertia_verdict` and
`detail::ssn_inertia_verdict`, which test perturbed-pivots FIRST) must
be rewritten to consume the optional's absence, with checklist (h)'s
re-derived degradation analysis as input. T5-Mac asserts the surface
half; the verdict-consumer half is an M3 test obligation.

## A.6 Threading (SQP req. 5)

`Options::num_threads` is per-instance, applied at call scope on MKL
(`mkl_set_num_threads_local` around backend calls — never the global
env/process setting). On Accelerate: CORRECTED by the consumed-surface
audit — thread control EXISTS (`BLASSetThreading`, already exercised by
the IPM engine's Accelerate path); it is process-scoped, not
per-instance, so the option maps to it with process-scope semantics
documented per the A.5 honesty discipline (scope difference stated,
never papered over). The exact mapping is fixed at M2 and verified on
the Mac leg. Measurement discipline and
the rig's pinning policy are in B.2. The merged-bench caveat carries:
concurrent MKL instances spin-wait catastrophically; sweep runners
serialize suites.

## A.7 What the old seams have that this surface drops

- psiopt's Eigen `SparseSolverBase` inheritance (solve-expression API):
  dropped; both engines call solve directly.
- psiopt's `mem_/flops_` performance metrics: kept as optional
  informational getters if the consumed-surface audit finds a reader;
  dropped otherwise.
- `KktSystem`'s Pardiso-phase numbering in its public face: subsumed by
  the analyze/factorize/solve lifecycle; the phase discipline survives as
  the implementation.
- Raw iparm access: dropped (A.3 Options note; human-review gate).

Anything else found in use by the audit gets a named option or a named
rejection in the freeze review — nothing silently.

---

# Part B — the golden-rig build plan

## B.1 Harness (recap of the agreed shape, now concrete)

GTest suite `hven_golden_rig` in `hven/tests/golden_rig/`; a
`SeamUnderTest` concept mirroring A.3; three adapters — `hven::linear`
native, psiopt old seam (`HVEN_RIG_PSIOPT_SEAM=<tycho checkout>`), sqp
old seam (`HVEN_RIG_SQP_SEAM=<tycho_sqp archive checkout>`); old-seam
adapters are test-only, deleted after M3. Expected tables at
`tests/golden_rig/expected/<trace>.csv`: counters exact; values with
stated tolerance; one row per backend arm; provenance columns (machine,
backend+version, thread pin, commit, date); Mac slots literal
`UNOBSERVED` until filled from hardware. `hven_golden_rig_report` dumps
observed-vs-expected for derivation and docket entries.

## B.2 Comparison policy (NORMATIVE — updated for the thread-count evidence)

The SQP side demonstrated a committed test whose float residuals at nine
digits are run-to-run nondeterministic under multithreaded MKL, while
integer counters always reproduce. Policy, now evidence-backed:

1. **Counters: exact integers, per backend, always.**
2. **Float values: compared per-backend against own-backend expectations
   at the trace's stated tolerance, and every asserted run pins threads
   to 1 by the mechanism the seam under test possesses** — per-instance
   (`Options::num_threads = 1`, A.6) on `hven::linear`; process-env
   (`MKL_NUM_THREADS=1`) on the old-seam adapters, which have no
   per-instance control. Expected tables record WHICH mechanism and its
   value in their provenance banner; a table row without a thread pin
   is invalid. (Tightened per the SQP freeze review §5.)
3. **0-ULP assertions** (T4-vs-T1 class) are additionally valid only
   within a single pinned-thread process run — never across runs with
   different thread settings.
4. **Multithreaded runs are permitted as unasserted smoke only** (crash/
   hang detection), clearly labeled.
5. **Presence/absence of evidence fields is part of the expectation**
   (T5-Mac asserts absence; zero fails).
6. **Failure states asserted as states**, never sentinel values.
7. Cross-backend comparison only at trace-stated analytic tolerances.

**T8 amendment — RESOLVED (accepted verbatim at the freeze review §4):**
T8 asserts counters identical and values within stated tolerance across
thread settings; the bitwise claim is retained only if the M1 derivation
run demonstrates it for T8's specific matrix; otherwise the observed
cross-thread deviation is recorded as documentation.

**T4b — NEW trace (SQP-proposed at the freeze review §2, accepted;
sharpened per the addendum):** emit handle → originator refactorizes
with new values → `adopt()` must REFUSE symbolic-plus-numeric reuse on
the epoch mismatch (symbolic-only reuse on the hash match stays legal).
Observables: counters (no re-analysis, one numeric refactorize on the
adopter) AND the epoch discriminator — `handle.epoch()` remains at its
emission value while the adopter's live `epoch()` advances past it,
which distinguishes "reuse refused, fresh numeric" from "quietly
re-shared the mutated session" in a way counter deltas alone cannot.
Pins the A.3 staleness ladder — the scenario the shipped generation
stamp exists for.

**T2b — NEW trace (the addendum's predicate observable):** a
factorization with perturbed pivots must report
`supports_partial_solve() == false`, and the trace asserts the border
stack's full-solve fallback fires (counter-visible: solve-count
parity). The Mac arm of T2b asserts the predicate is false under
absent perturbation evidence — the conservative rung, never a
fabricated true.

## B.3 Derivation procedure (M1, after `linear/` lands, before M2)

1. Build the two old-seam adapters against the pinned checkouts (tycho
   at the M1-start commit; tycho_sqp at the archive tag — the tag is a
   Phase-7-close deliverable).
2. Run every P/T trace (T1–T8 + T2b + T4b, P1–P6) on the old seams, MKL,
   thread-pinned per B.2,
   recording counters + values via `hven_golden_rig_report`.
3. Write expected tables from those recordings (this is pin derivation:
   the numbers come from observation of the seams the engines trust
   today, with provenance). Mac slots stay `UNOBSERVED`.
4. **P4/P5 run fail-by-design on the psiopt old seam** → docket entries
   created with today's fabricated behaviors recorded as the failing
   state and A.5's semantics as the expected state.
5. Run the full suite against `hven::linear`; every trace must pass
   (including P4/P5 against the new semantics). A NEW-side failure
   blocks M1 close.
6. One Mac hardware leg (Grant's machine): fill Mac slots for every
   trace with a Mac arm; T5-Mac/P4 absence assertions run here for the
   first time.

## B.4 Consumed-surface audit (M1, feeds A.3's option set and A.7)

Static + runtime pass over both consumers: grep the psiopt and sqp
sources for every backend touchpoint (iparm reads/writes, Pardiso phase
calls, Accelerate calls, LAPACKE calls, thread-control calls); then run
the 17-problem corpus (psiopt) and the SQP latch/HS fixtures under a
recording shim confirming which touchpoints execute. Output: the
audited option/field union — reconciled against the freeze review §1
list — plus any additional trace the audit shows is needed (rev B's
"coverage claim is observed, not assumed").

**Pre-registered coverage test (freeze review §6):** the phase-33x
`iparm[7]` force-zero-restore rule (`kkt_system.h:327-335`) is NOT an
option and a consumed-*options* grep will not find it. The audit must
find it independently through its runtime shim; if it does not, the
audit has a demonstrated coverage gap and its method gets fixed before
its output is trusted.

## B.5 Rig deliverables checklist (M1 exit)

- [ ] Harness + three adapters build; suite runs on all three seams.
- [ ] Expected tables for T1–T8 + T2b + T4b and P1–P6, MKL-observed,
      thread-pinned (mechanism recorded), provenance-complete; Mac
      slots filled from one hardware leg.
- [ ] P4/P5 docket entries (old-seam failures) filed.
- [ ] Consumed-surface audit report committed; A.3 option set finalized
      against it.
- [ ] Full suite green on `hven::linear` (both backends; Mac leg).
