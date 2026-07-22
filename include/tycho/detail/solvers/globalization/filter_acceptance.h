// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FilterAcceptance — the concrete Wächter–Biegler (θ, φ)-pair FILTER strategy
// on the shared switching skeleton (SwitchingAcceptance). Where the funnel
// companion collapses the acceptance history into ONE scalar width, the filter
// keeps a SET of dominating (θ, φ) pairs: a trial is acceptable only if it is
// not dominated (within a margin) by any stored pair. This class supplies ONLY
// the filter-specific H-type verdict and the augmentation bookkeeping through
// the base's subclass hooks; the θ_min/θ_max ceiling, the switching condition,
// and the F-type Armijo test all live in SwitchingAcceptance.
//
// Opt-in via Settings::acceptance_strategy_ == filter; the default
// classic_merit path stays bit-identical.
//
// =============================================================================
// FORMULATION — derived from the two primary sources (fetched + read, not from
// memory):
//   [WB]  Wächter & Biegler, "On the implementation of an interior-point
//         filter line-search algorithm for large-scale nonlinear programming",
//         Math. Program. 106(1):25-57 (2006). Filter-acceptability margins
//         Eqs. (18a)/(18b); the filter is augmented (Eq. (17)) only when the
//         switching condition fails or Armijo was not the acceptor.
//   [Ipopt] The COIN-OR Ipopt reference implementation,
//         src/Algorithm/IpFilterLSAcceptor.cpp and src/Algorithm/IpFilter.{cpp,hpp}.
//         The constants below are Ipopt's shipped option defaults (cited by
//         option name); each rule cites the Ipopt function that implements it.
//         Ipopt fills the practical details the paper leaves open (the
//         barrier-objective ceiling test, the filter-reset heuristic, and the
//         exact dominance comparison), transcribed rule-by-rule below.
// =============================================================================
//
// ProgressMeasures mapping (same convention as switching_acceptance.h):
//   current.infeasibility / trial.infeasibility = θ = ‖c‖ at z_k / z_k + α·d
//     (Ipopt reference_theta_ / trial_theta)
//   φ(pt) = pt.objective + pt.auxiliary = the barrier objective
//     (Ipopt reference_barr_ / trial_barr)
//
// (1) Acceptance verdicts, split across the base's two hooks to match Ipopt's
//     CheckAcceptabilityOfTrialPoint, which checks filter membership for EVERY
//     trial (F-type included), not only H-type ones. The base runs (1b) as the
//     membership test on every trial (step 2 of its order) and (1a) as the
//     H-type sufficient-progress test:
//
//   (1a) Acceptable to the current iterate (Ipopt
//        FilterLSAcceptor::IsAcceptableToCurrentIterate) — the H-type progress
//        verdict. Two parts:
//
//        • Barrier-objective ceiling (the "obj_max_inc" test). Ipopt rejects a
//          trial whose barrier objective grows by more than ~5 orders of
//          magnitude — a LOG10-SCALED comparison, NOT a ratio. Verbatim Ipopt
//          (only evaluated when φ_trial > φ_current):
//              Number basval = 1.;
//              if (std::abs(reference_barr_) > 10.)
//                 basval = std::log10(std::abs(reference_barr_));
//              if (std::log10(trial_barr - reference_barr_) > obj_max_inc_ + basval)
//                 return false;
//          with obj_max_inc_ = kFilterObjMaxInc (option "obj_max_inc" = 5.0).
//
//        • The two-condition margin test (WB Eqs. (18a)/(18b)); Ipopt returns
//              Compare_le(trial_theta, (1.-gamma_theta_)*reference_theta_, ...)
//           || Compare_le(trial_barr - reference_barr_, -gamma_phi_*reference_theta_, ...)
//          i.e. θ_trial ≤ (1−γ_θ)·θ_current  OR  φ_trial ≤ φ_current − γ_φ·θ_current,
//          with γ_θ = kFilterGammaTheta (option "gamma_theta" = 1e-5) and
//          γ_φ = kFilterGammaPhi (option "gamma_phi" = 1e-8). (Ipopt's Compare_le
//          is a machine-epsilon-scaled ≤; this implementation uses a plain ≤ —
//          see the divergence note below.) Note BOTH margins scale by θ_current,
//          matching Ipopt exactly.
//
//   (1b) Acceptable to the filter (Ipopt FilterLSAcceptor::IsAcceptableToCurrentFilter
//        → Filter::Acceptable → FilterEntry::Acceptable) — the MEMBERSHIP test,
//        run for every trial. The trial is acceptable
//        w.r.t. a stored entry (φ_j, θ_j) iff φ_trial ≤ φ_j OR θ_trial ≤ θ_j
//        (Ipopt FilterEntry::Acceptable: `for i: if (vals[i] <= vals_[i]) return
//        true;` — a per-coordinate ≤, so an entry BLOCKS the trial only when
//        φ_trial > φ_j AND θ_trial > θ_j). The trial is acceptable to the filter
//        iff it is acceptable w.r.t. EVERY entry (Filter::Acceptable). No γ margin
//        is re-applied here — the margin is baked into the stored pair at
//        augmentation time (2), matching Ipopt.
//
// (2) Augmentation on an ACCEPTED H-type step (Ipopt FilterLSAcceptor::
//     AugmentFilter, run from the H-type acceptance branch only; WB Eq. (17)).
//     The base runs register_accepted_step() on every accept; the filter
//     augments only when its h_type flag is set. The stored pair is the CURRENT
//     (reference) iterate's, pushed IN from both margins:
//         φ_add = φ_current − γ_φ·θ_current  (Ipopt reference_barr_ − gamma_phi_·reference_theta_)
//         θ_add = (1 − γ_θ)·θ_current        (Ipopt (1.−gamma_theta_)·reference_theta_)
//     Adding a pair prunes every stored entry it DOMINATES (Ipopt
//     Filter::AddEntry → FilterEntry::Dominated: entry (φ_j, θ_j) is dominated
//     by the new (φ_add, θ_add) iff φ_add ≤ φ_j AND θ_add ≤ θ_j — Ipopt
//     `for i: if (vals[i] > vals_[i]) return false; ... return true;`).
//
// (3) When the filter is augmented: ONLY on an accepted H-type step. An F-type
//     accept (switching + Armijo, handled entirely by the base) never augments
//     — WB augment the filter only when the switching condition fails or Armijo
//     was not the acceptor. This matches Ipopt, where AugmentFilter is called
//     from the single H-type acceptance branch of CheckAcceptabilityOfTrialPoint.
//
// (4) Filter-reset heuristic (Ipopt FilterLSAcceptor::CheckAcceptabilityOfTrialPoint,
//     the "Filter reset heuristic" block; options "max_filter_resets" = 5 and
//     "filter_reset_trigger" = 5). Ipopt distinguishes WHY the last trial was
//     rejected: last_rejection_due_to_filter_ is set TRUE only for a filter-
//     membership rejection, FALSE for a current-iterate/Armijo rejection, and is
//     left UNCHANGED by a θ_max-ceiling rejection (Ipopt returns on "Tmax"
//     before touching the flag). The heuristic runs ONCE PER ACCEPT (the block
//     sits at the tail of CheckAcceptabilityOfTrialPoint, past both early
//     returns): a filter-caused LAST rejection advances the successive-iteration
//     counter (clearing the filter at the trigger), any other last rejection
//     zeroes it. This implementation mirrors that exactly via the base's two
//     bookkeeping hooks — notify_trial_rejected() records the last cause into
//     last_rejection_was_filter_, and register_accepted_step() runs the block
//     below on each accept. Ipopt code, verbatim:
//         if (max_filter_resets_ > 0) {
//            if (n_filter_resets_ < max_filter_resets_) {
//               if (last_rejection_due_to_filter_) {
//                  count_successive_filter_rejections_++;
//                  if (count_successive_filter_rejections_ >= filter_reset_trigger_) {
//                     Reset();   // Filter::Clear + counters
//                  }
//               } else {
//                  count_successive_filter_rejections_ = 0;
//               }
//            }
//         }
//     Because the counter advances only on ACCEPT (not per rejected trial), a
//     single line search backtracking through N filter-blocked trials before its
//     accept counts ONE increment, not N — exactly Ipopt's per-iteration
//     granularity. n_filter_resets_ caps the number of clears; all reset state
//     is cleared by reset_bounds() (the per-phase hook), so the cap is per-phase.
//
// Divergences from the sources, disclosed (consequence stated):
//   • Membership-check ORDER (no longer a rejection-attribution divergence).
//     The shared skeleton still checks the filter membership (1b) at step 2,
//     before the switching/Armijo/(1a) tests at step 3 — the unified order
//     Uno's funnel also uses, the reverse of Ipopt's own (1a)-then-(1b) order.
//     Acceptance outcomes are unaffected by this (a trial must pass ALL
//     tests, and AND does not depend on order). Rejection-cause ATTRIBUTION
//     now reproduces Ipopt's T1-then-filter semantics exactly despite the
//     reversed check order: a kMembership rejection speculatively evaluates
//     the type-appropriate T1 (Armijo for an f-type trial, the current-
//     iterate test otherwise — side-effect-free) and is attributed to the
//     filter (last_rejection_was_filter_ = true) only when that speculative
//     T1 passes; if T1 would also have failed, the rejection is attributed to
//     that failure instead (last_rejection_was_filter_ = false), exactly as
//     Ipopt's last_rejection_due_to_filter_ would read. See
//     SwitchingAcceptance::is_iterate_acceptable's membership-reject branch
//     and notify_trial_rejected() below. The θ_max-ceiling and current-
//     iterate/Armijo mappings match Ipopt exactly (a ceiling rejection leaves
//     the flag unchanged; an (1a)/Armijo rejection zeroes it on the next
//     accept).
//   • n_filter_resets_ increment. In current Ipopt master n_filter_resets_ is
//     initialized to 0 but never incremented (the "maximal number of resets
//     already exceeded" branch is unreachable), so the max_filter_resets cap is
//     not actually enforced upstream. This implementation increments the reset
//     counter at each clear and honours the kFilterMaxResets cap — the behavior
//     the option name and the "F-" branch document, so a sixth reset never
//     happens.
//   • No θ_max filter seed. Ipopt's per-phase reset (InitializeImpl → Reset)
//     Clear()s the filter to EMPTY; θ_max is a separate scalar ceiling (owned by
//     the base here), not a seeded filter entry. This implementation likewise
//     starts each phase with an empty filter — consequence: the first accepted
//     H-type step is what first populates the filter, exactly as in Ipopt.
//   • Compare_le → plain ≤. Ipopt's Compare_le applies a machine-epsilon-scaled
//     rounding tolerance to the ≤ comparisons in (1a). This implementation uses
//     a plain ≤; the tolerance only affects trials sitting within a few ULP of a
//     margin, and the plain form matches the WB paper's stated inequalities.
//
// Ownership: like the other generic strategies, no SolverContext reference and
// no NLP eval — a pure function of its ProgressMeasures arguments plus the
// filter/counter state. reset() (via the base) re-arms the lazy θ₀ init and
// calls reset_bounds(), which empties the filter and zeroes the counters.
//
// Definitions live in src/solvers/psiopt_globalization.cpp.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/switching_acceptance.h"

namespace tycho::solvers {

// =============================================================================
// Filter constants (kPascalCase; each cites its source equation + Ipopt option).
// =============================================================================

// (18a) constraint-violation margin γ_θ (WB Eq. (18a); Ipopt option
// "gamma_theta" = 1e-5).
inline constexpr double kFilterGammaTheta = 1.0e-5;
// (18b) barrier-objective margin γ_φ (WB Eq. (18b); Ipopt option
// "gamma_phi" = 1e-8).
inline constexpr double kFilterGammaPhi = 1.0e-8;
// Barrier-objective ceiling: reject a trial whose barrier objective grows by
// more than this many orders of magnitude (log10-scaled — see (1a); Ipopt
// option "obj_max_inc" = 5.0).
inline constexpr double kFilterObjMaxInc = 5.0;
// Filter-reset heuristic: clear the filter after this many successive
// filter-caused H-type rejections (Ipopt option "filter_reset_trigger" = 5).
inline constexpr int kFilterResetTrigger = 5;
// Filter-reset heuristic: cap on the number of clears per phase (Ipopt option
// "max_filter_resets" = 5).
inline constexpr int kFilterMaxResets = 5;

// =============================================================================
// Filter — the (θ, φ)-pair set value type. A plain std::vector of margined
// entries (corpus problems are small-iteration-count; no capacity scheme until
// profiling demands one). Mirrors Ipopt's Filter/FilterEntry with the same
// per-coordinate ≤ dominance semantics; entries store the ALREADY-margined pair
// so the acceptability test is a plain dominance check (see (1b)/(2)).
// =============================================================================
class Filter {
  public:
    // (1b) Acceptable w.r.t. every entry (Ipopt Filter::Acceptable +
    // FilterEntry::Acceptable). An entry (φ_j, θ_j) blocks the trial only when
    // φ > φ_j AND θ > θ_j (the negation of the per-coordinate ≤).
    bool acceptable(double phi, double theta) const {
        for (const Entry &e : entries_) {
            if (phi > e.phi && theta > e.theta)
                return false;
        }
        return true;
    }

    // (2) Add the margined pair and prune every entry it dominates (Ipopt
    // Filter::AddEntry + FilterEntry::Dominated). The caller passes the CURRENT
    // (reference) iterate's (φ, θ); the margins are applied here.
    void augment(double phi_current, double theta_current) {
        const double phi_add = phi_current - kFilterGammaPhi * theta_current;
        const double theta_add = (1.0 - kFilterGammaTheta) * theta_current;
        // Prune entries dominated by the new pair: (φ_j, θ_j) is dominated iff
        // φ_add ≤ φ_j AND θ_add ≤ θ_j.
        std::erase_if(entries_, [&](const Entry &e) {
            return phi_add <= e.phi && theta_add <= e.theta;
        });
        entries_.push_back({phi_add, theta_add});
    }

    void clear() { entries_.clear(); }

    std::size_t size() const { return entries_.size(); }

    // Diagnostics (unit tests): the i-th stored margined (φ, θ) pair.
    std::pair<double, double> entry(std::size_t i) const {
        return {entries_[i].phi, entries_[i].theta};
    }

  private:
    struct Entry {
        double phi;
        double theta;
    };
    std::vector<Entry> entries_;
};

// =============================================================================
// FilterAcceptance — (θ, φ)-pair filter H-type strategy on the switching
// skeleton.
// =============================================================================
class FilterAcceptance final : public SwitchingAcceptance {
  public:
    // Number of stored filter entries (diagnostics + unit tests). Zero before
    // the first accepted H-type step and after any reset.
    std::size_t filter_size() const { return filter_.size(); }

    // Diagnostics (unit tests): the i-th stored margined (φ, θ) pair.
    std::pair<double, double> filter_entry(std::size_t i) const { return filter_.entry(i); }

    // Diagnostics (unit tests): reset-heuristic counters (see (4)).
    int successive_filter_rejections() const { return successive_filter_rejections_; }
    int filter_resets() const { return n_filter_resets_; }

    // Solver-level observability hook (see AcceptanceStrategy::
    // append_diagnostics): reports filter_size() into
    // SolveResult::last_filter_size_ and filter_resets() (n_filter_resets_,
    // the per-phase reset-heuristic total — see (4) and reset_bounds()) into
    // SolveResult::last_filter_resets_.
    void append_diagnostics(PSIOPT::SolveResult &result) const override;

  protected:
    // --- SwitchingAcceptance hooks (see the file-top formulation) ---
    // Start the phase with an empty filter (Ipopt Reset; no θ_max seed).
    void initialize_bounds(double theta_0) override;
    // Empty the filter and zero the reset-heuristic state.
    void reset_bounds() override;
    // (1b) membership: acceptable to the current filter (every trial).
    bool is_trial_acceptable_to_strategy(const ProgressMeasures &current,
                                         const ProgressMeasures &trial) override;
    // (1a) H-type sufficient progress: acceptable to the current iterate.
    bool is_h_type_progress_acceptable(const ProgressMeasures &current,
                                       const ProgressMeasures &trial) override;
    // (2)/(4): augment the filter on an H-type accept, and run the per-iteration
    // reset heuristic (which reads the last rejection's cause) on any accept.
    void register_accepted_step(const ProgressMeasures &current, const ProgressMeasures &trial,
                                bool h_type) override;
    // (4): record the last rejection's cause for the per-iteration reset
    // heuristic. trial_passed_progress_test is the base's speculative T1
    // result, meaningful only when cause == kMembership.
    void notify_trial_rejected(RejectionCause cause, bool trial_passed_progress_test) override;

  private:
    // (1a): acceptable to the current iterate — the barrier-objective ceiling
    // AND the two-condition margin test.
    static bool is_acceptable_to_current(double phi_trial, double theta_trial, double phi_current,
                                         double theta_current);

    Filter filter_;
    // (4) reset-heuristic state; all zeroed by reset_bounds().
    int successive_filter_rejections_ = 0;   // Ipopt count_successive_filter_rejections_.
    int n_filter_resets_ = 0;                // Ipopt n_filter_resets_ (capped by kFilterMaxResets).
    bool last_rejection_was_filter_ = false; // Ipopt last_rejection_due_to_filter_.
};

} // namespace tycho::solvers
