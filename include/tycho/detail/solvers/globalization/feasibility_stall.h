// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FeasibilityStallDetector — windowed no-progress detector for the
// feasibility-only stage.
//
// The feasibility stage accepts every fraction-to-boundary step (the line
// search runs with the objective scaled to zero), so the rejected-trial gate
// that dispatches the recovery chain — and through it feasibility restoration
// — is structurally unreachable from that stage. A stage that stops making
// feasibility progress therefore burns its whole iteration budget with no
// mechanism ever consulted. This detector supplies the missing dispatch
// signal: it watches the L1 constraint violation once per feasibility-stage
// iteration and reports a stall when the best-seen violation has not improved
// AT ALL — beyond relative rounding noise — for a full window of consecutive
// iterations.
//
// Provenance: Tycho-original. The reference interior-point method (Ipopt) has
// no feasibility-only stage and therefore no analog of this detector; its
// restoration dispatch rides the line-search failure path, which the
// zero-objective stage never takes.
//
// Constant sizing, against the recorded corpus evidence:
//
//   * The threshold is a rounding-noise floor, not a progress standard. A
//     stage only has to make the best-seen violation strictly smaller by a
//     relative 1e-12 to keep its window open, so the detector fires on a
//     genuine plateau or on growth and never on a productive crawl, however
//     slow. An earlier 1% threshold did cut productive crawls: the stiff
//     hypersensitive corpus problem, which grinds its violation down by a few
//     parts in ten thousand per iteration, lost the acceptable exit it reaches
//     under this threshold.
//   * The window is sized so a real stall is caught promptly while a crawl is
//     never mistaken for one. The motivating stalled trace GREW its residual
//     for hundreds of uncontested iterations, so it fires within a single
//     window; a stage crawling along its violation floor improves the
//     best-seen value on essentially every iteration and never accumulates
//     one.
// =============================================================================

#pragma once

#include <limits>

namespace tycho::solvers {

// Consecutive feasibility-stage iterations without improvement of the
// best-seen violation before the stage is declared stalled.
inline constexpr int kFeasStallWindow = 50;

// Relative improvement of the best-seen L1 constraint violation that keeps the
// window open. A rounding-noise floor, not a progress standard: any genuine
// decrease clears it.
inline constexpr double kFeasStallMinRelImprovement = 1.0e-12;

// Per-phase value type: alg_impl owns one instance per phase and calls
// observe() once per feasibility-stage iteration.
//
// Two levels of re-arming, because the two pieces of state have different
// lifetimes. reset_window() re-arms the no-progress window after a dispatched
// restoration episode, so the resumed stage restarts its window from the
// post-restoration point; it deliberately preserves the first-dispatch
// violation, which is the reference the caller compares against to decide
// whether recovery has bought the stage any ground at all. reset() clears
// everything, for a fresh phase.
struct FeasibilityStallDetector {
    // Smallest L1 constraint violation observed since the window was armed.
    double best_theta_ = std::numeric_limits<double>::infinity();

    // Consecutive observations since best_theta_ last improved.
    int iters_without_improvement_ = 0;

    // L1 constraint violation at the first restoration dispatch of this phase,
    // or infinity if the phase has never dispatched. Recorded once, by
    // note_dispatch(), and preserved across reset_window().
    double theta_at_first_dispatch_ = std::numeric_limits<double>::infinity();

    // Returns true when the stage has gone kFeasStallWindow consecutive
    // observations without improving best_theta_ by at least
    // kFeasStallMinRelImprovement (relative). An improving observation
    // records the new best and restarts the window.
    bool observe(double theta) {
        if (theta < (1.0 - kFeasStallMinRelImprovement) * best_theta_) {
            best_theta_ = theta;
            iters_without_improvement_ = 0;
            return false;
        }
        ++iters_without_improvement_;
        return iters_without_improvement_ >= kFeasStallWindow;
    }

    // Records the violation at which this phase first entered restoration.
    // Later dispatches are ignored, so the recorded value always marks where
    // recovery began.
    void note_dispatch(double theta) {
        if (theta_at_first_dispatch_ == std::numeric_limits<double>::infinity())
            theta_at_first_dispatch_ = theta;
    }

    // Re-arms the no-progress window only; theta_at_first_dispatch_ survives.
    void reset_window() {
        best_theta_ = std::numeric_limits<double>::infinity();
        iters_without_improvement_ = 0;
    }

    // Full per-phase re-arm, including the first-dispatch reference.
    void reset() {
        reset_window();
        theta_at_first_dispatch_ = std::numeric_limits<double>::infinity();
    }
};

} // namespace tycho::solvers
