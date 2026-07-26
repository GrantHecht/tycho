// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FeasibilityStallDetector — windowed no-progress detector for the
// feasibility-only stage.
//
// Under its default no-line-search configuration the feasibility stage
// accepts every fraction-to-boundary step, so the rejected-trial gate that
// dispatches the recovery chain — and through it feasibility restoration —
// is never consulted from that stage. (A user-configured stage line search
// runs a zero-objective merit test that can reject growth steps, making the
// gate reachable; the detector below is useful either way, since a stalled
// stage under either configuration otherwise burns its budget.) A stage that stops making
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
// observe() once per feasibility-stage iteration. Per-phase freshness needs no
// explicit clear — the detector is an alg_impl local, so every phase begins
// with a default-constructed one.
//
// reset_window() re-arms the no-progress window after a dispatched restoration
// episode, so the resumed stage restarts its window from the post-restoration
// point; it deliberately preserves the last-dispatch violation, which is the
// reference the caller compares against to decide whether the stage has gained
// anything since recovery last handed it back.
struct FeasibilityStallDetector {
    // Smallest L1 constraint violation observed since the window was armed.
    double best_theta_ = std::numeric_limits<double>::infinity();

    // Consecutive observations since best_theta_ last improved.
    int iters_without_improvement_ = 0;

    // L1 constraint violation at this phase's MOST RECENT restoration dispatch,
    // or infinity if the phase has never dispatched. Rewritten by every
    // note_dispatch() and preserved across reset_window(), so it always marks
    // the point at which recovery last handed the stage back.
    double theta_at_last_dispatch_ = std::numeric_limits<double>::infinity();

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

    // Records the violation at which this phase entered restoration. Every
    // dispatch overwrites the reference: the question it answers is whether the
    // stage has gained anything since recovery LAST handed it back, not since
    // the first episode of the phase.
    void note_dispatch(double theta) { theta_at_last_dispatch_ = theta; }

    // Re-arms the no-progress window only; theta_at_last_dispatch_ survives.
    void reset_window() {
        best_theta_ = std::numeric_limits<double>::infinity();
        iters_without_improvement_ = 0;
    }
};

} // namespace tycho::solvers
