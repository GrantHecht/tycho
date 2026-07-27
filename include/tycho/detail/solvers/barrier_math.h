// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Extracted the slack-reset and log-barrier objective/gradient kernels from
//     PSIOPT (psiopt.cpp), where the globalization component extraction had left
//     one verbatim copy per component
// =============================================================================
//
// Three tiny pure kernels the barrier machinery evaluates everywhere: the slack
// reset that completes a raw inequality residual, the log-barrier objective, and
// its dual gradient. They depend on nothing but their arguments, which is why
// each extracted component (ClassicMeritAcceptance, ClassicAdaptiveGovernor,
// MonitoredBarrierGovernor) could carry its own verbatim copy reading through a
// SolverContext instead of a PSIOPT member. This header is the single home; each
// former member is now a one-line forwarder, so the arithmetic exists once.
//
// The bodies below are token-identical to the copies they replace, with the
// former member reads (slack_vars_ / inequal_cons_ / neg_slack_reset_) turned
// into explicit parameters. That matters: this arithmetic is on the iterate
// path, and the merge gate for the component extraction was a bit-identical
// iteration-count comparison.
//
// PSIOPT::complementarity is deliberately NOT here. Its .sum() reduction feeds
// mu, so any change to how that sum is formed can move iterates by a ULP under
// fast-math; unifying it needs its own evidence and is out of scope for a
// move-neutral extraction.

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace tycho::solvers::detail {

// Completes the raw inequality residual g(x) into g(x) + s and repairs
// non-positive slacks in place: a slack below `neg_slack_reset` is treated as
// `neg_slack_reset`, and a negative residual resets the pair instead of adding.
// `slack_vars` and `neg_slack_reset` are the solver's slack_vars_ /
// settings_.neg_slack_reset_.
inline void apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> FXI,
                               int slack_vars, double neg_slack_reset) {
    for (int i = 0; i < slack_vars; i++) {
        double fxi = FXI[i];
        double si = S[i];
        if (si < neg_slack_reset) {
            si = neg_slack_reset;
        }

        if (fxi < 0.0) {
            FXI[i] = 0.0;
            S[i] = std::max(std::abs(fxi), neg_slack_reset);
        } else {
            FXI[i] += si;
        }
    }
}

// Log-barrier objective -mu * sum(log s_i) over the first `inequal_cons` slacks.
inline double barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu, int inequal_cons) {
    double psi = 0;
    for (int i = 0; i < inequal_cons; i++) {
        psi += -mu * std::log(S[i]);
    }
    return psi;
}

// Dual gradient of the log-barrier objective: lambda_i - mu / s_i, written into
// the KKT vector's dual-gradient block.
inline void barrier_gradient(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI,
                             double mu, Eigen::Ref<Eigen::VectorXd> AGS) {
    AGS = LI - mu * (S.cwiseInverse());
}

} // namespace tycho::solvers::detail
