// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FeasibilityStallDetector — windowed sustained-worsening detector for the
// feasibility-only stage.
//
// Under its default no-line-search configuration the feasibility stage
// accepts every fraction-to-boundary step, so the rejected-trial gate that
// dispatches the recovery chain — and through it feasibility restoration —
// is never consulted from that stage. (A user-configured stage line search
// runs a zero-objective merit test that can reject growth steps, making the
// gate reachable; the detector below is useful either way, since a stage whose
// violation runs away under either configuration otherwise burns its budget.)
// Such a stage therefore burns its whole iteration budget with no mechanism
// ever consulted. This detector supplies the missing dispatch
// signal: it watches the L1 constraint violation once per feasibility-stage
// iteration and reports a worsening stage when that violation has sat a full
// window of consecutive iterations at or above a fixed multiple of the best
// value the stage has ever held.
//
// Provenance: Tycho-original. The reference interior-point method (Ipopt) has
// no feasibility-only stage and therefore no analog of this detector; its
// restoration dispatch rides the line-search failure path, which the
// zero-objective stage never takes.
//
// Constant sizing, against the recorded corpus evidence:
//
//   * The detector fires only on a stage whose violation has sat at least 25%
//     above its own best for 50 consecutive iterations — genuine, sustained
//     worsening. A stage flat at its best (a plateau) and a stage improving at
//     any rate (a crawl, however slow) never fire, and both keep burning their
//     iteration budget exactly as they did before this detector existed. That
//     narrowing is deliberate: worsening is the only class in which a
//     dispatched episode has demonstrated value. Dispatching into a plateaued
//     stage only saved iterations on problems that were failing anyway, while
//     an episode injected into a quietly succeeding stage steered one corpus
//     problem off the acceptable exit it otherwise reaches.
//   * The motivating stalled trace sets the price of that narrowing. Its worst
//     single residual GREW from 1.106 to 2.602 with nothing to contest it, but
//     the growth is one jump across the opening iterations followed by some 495
//     flat ones, and in the L1 measure watched here that plateau stays inside
//     the 25% margin — so that stage keeps its old burn-the-budget behaviour.
//     Ceding it is the right trade: the episodes it used to draw only shortened
//     a solve that diverges under every configuration, while the same episodes
//     elsewhere in the corpus cost a verdict.
//   * Jitter robustness is by construction. A 25% margin is some eleven orders
//     of magnitude above the relative rounding noise of a threaded sparse
//     factorization, so no amount of run-to-run FP drift can move an
//     observation across the elevation mark and reshape the dispatch schedule.
//   * The window is sized so a real worsening is caught promptly while nothing
//     transient is mistaken for one: a stage has to hold an elevated violation
//     for 50 straight iterations, and a single dip back to (or below) its best
//     starts the count over.
// =============================================================================

#pragma once

#include <limits>

namespace tycho::solvers {

// Consecutive feasibility-stage iterations at an elevated violation before the
// stage is declared worsening.
inline constexpr int kFeasStallWindow = 50;

// Multiple of the best-seen L1 constraint violation at or above which an
// observation counts as elevated. The violation has to sit a full 25% above
// the best ground the stage has held — far outside any floating-point noise —
// for the window to accumulate.
inline constexpr double kFeasStallGrowthFactor = 1.25;

// Relative floor for the caller's net-progress test against the violation
// recorded at the last restoration dispatch: a rounding-noise floor, not a
// progress standard, so an ulp of drift at a flat stage does not read as
// ground gained. The detector itself does not use it — best-seen tracking
// takes any strict decrease.
inline constexpr double kFeasStallMinRelImprovement = 1.0e-12;

// Per-phase value type: alg_impl owns one instance per phase and calls
// observe() once per feasibility-stage iteration. Per-phase freshness needs no
// explicit clear — the detector is an alg_impl local, so every phase begins
// with a default-constructed one.
//
// reset_window() re-arms the elevation window after a dispatched restoration
// episode, so the resumed stage measures its worsening against the
// post-restoration point; it deliberately preserves the last-dispatch
// violation, which is the reference the caller compares against to decide
// whether the stage has gained anything since recovery last handed it back.
struct FeasibilityStallDetector {
    // Smallest L1 constraint violation observed since the window was armed.
    double best_theta_ = std::numeric_limits<double>::infinity();

    // Consecutive observations at or above kFeasStallGrowthFactor * best_theta_.
    int iters_elevated_ = 0;

    // L1 constraint violation at this phase's MOST RECENT restoration dispatch,
    // or infinity if the phase has never dispatched. Rewritten by every
    // note_dispatch() and preserved across reset_window(), so it always marks
    // the point at which recovery last handed the stage back.
    double theta_at_last_dispatch_ = std::numeric_limits<double>::infinity();

    // Returns true once the stage has spent kFeasStallWindow consecutive
    // observations at or above kFeasStallGrowthFactor times its best-seen
    // violation. Any observation below that mark breaks the run — and, when it
    // is also a new low, records it. A stage sitting exactly at its best, or
    // improving at any rate, therefore never accumulates a window.
    bool observe(double theta) {
        if (theta >= kFeasStallGrowthFactor * best_theta_) {
            ++iters_elevated_;
            return iters_elevated_ >= kFeasStallWindow;
        }
        if (theta < best_theta_)
            best_theta_ = theta;
        iters_elevated_ = 0;
        return false;
    }

    // Records the violation at which this phase entered restoration. Every
    // dispatch overwrites the reference: the question it answers is whether the
    // stage has gained anything since recovery LAST handed it back, not since
    // the first episode of the phase.
    void note_dispatch(double theta) { theta_at_last_dispatch_ = theta; }

    // Re-arms the elevation window only; theta_at_last_dispatch_ survives.
    void reset_window() {
        best_theta_ = std::numeric_limits<double>::infinity();
        iters_elevated_ = 0;
    }
};

} // namespace tycho::solvers
