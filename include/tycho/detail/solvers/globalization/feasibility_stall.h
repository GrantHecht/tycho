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
// by a minimum relative amount for a full window of consecutive iterations.
//
// Provenance: Tycho-original. The reference interior-point method (Ipopt) has
// no feasibility-only stage and therefore no analog of this detector; its
// restoration dispatch rides the line-search failure path, which the
// zero-objective stage never takes. Constants were sized against the recorded
// stall evidence: a corpus feasibility stage that burned 500 uncontested
// iterations while its equality residual grew 1.9x (fires here within one
// window), versus healthy stages, which improve the best-seen violation by
// far more than the threshold every few iterations and never accumulate a
// stalled window.
// =============================================================================

#pragma once

#include <limits>

namespace tycho::solvers {

// Consecutive feasibility-stage iterations without sufficient improvement of
// the best-seen violation before the stage is declared stalled.
inline constexpr int kFeasStallWindow = 25;

// Minimum relative improvement of the best-seen L1 constraint violation that
// resets the window (1%).
inline constexpr double kFeasStallMinRelImprovement = 1.0e-2;

// Per-phase value type: alg_impl owns one instance per phase and calls
// observe() once per feasibility-stage iteration. reset() re-arms after a
// dispatched restoration episode so the resumed stage restarts its window
// from the post-restoration point.
struct FeasibilityStallDetector {
    double best_theta_ = std::numeric_limits<double>::infinity();
    int iters_without_improvement_ = 0;

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

    void reset() {
        best_theta_ = std::numeric_limits<double>::infinity();
        iters_without_improvement_ = 0;
    }
};

} // namespace tycho::solvers
