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
//   - Added the primal variable-bound barrier kernels (objective, the two
//     gradient forms, and the condensed sigma diagonal)
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
// The four bound kernels at the bottom are the same idea for barrier terms on
// PRIMAL VARIABLE BOUNDS rather than on inequality slacks. They walk a BoundSet
// (index/value pairs, reduced-space indices), so a problem with no variable
// bounds gives them no trip count at all and every caller guards on an empty
// set anyway. Two of them are gradient accumulators that look interchangeable
// and are NOT -- see the mu-form / z-form note on the pair below.
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

#include "tycho/detail/solvers/bound_set.h"
#include "tycho/detail/typedefs/eigen_types.h"

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

// =============================================================================
// Primal variable-bound barrier kernels.
//
// For the bounds recorded in `b` (reduced-space indices; a two-sided variable
// appears in both lists) the barrier objective is
//
//     phi_mu(x) = f(x) - mu * sum ln(x_i - l_i) - mu * sum ln(u_i - x_i)
//
// and its primal gradient contribution is -mu/(x_i - l_i) + mu/(u_i - x_i).
// Every caller is responsible for keeping x strictly inside the recorded
// bounds; the interior push at solve entry and the fraction-to-boundary rule
// are what guarantee it, and none of these kernels re-checks.
// =============================================================================

// -mu * [ sum ln(x_i - l_i) + sum ln(u_i - x_i) ] over the bound set, plus the
// one-sided damping term kappa_d * mu * sum(distance) over the entries whose
// variable is bounded on that side only.
//
// The damping is Ipopt's (IpIpoptCalculatedQuantities::CalcBarrierTerm adds
// `kappa_d * mu * slack.Dot(dampind)` per side). Without it the log barrier
// gives a variable with only one finite bound nothing to push back against in
// its unbounded direction, and a barrier subproblem can drive it arbitrarily
// far out. It belongs to the barrier OBJECTIVE and its mu-form gradient only --
// see kKappaD's note in bound_set.h for the seams it must stay out of.
inline double bound_barrier_objective(ConstEigenRef<Eigen::VectorXd> x, const BoundSet &b,
                                      double mu) {
    const int nl = static_cast<int>(b.lower_idx_.size());
    const int nu = static_cast<int>(b.upper_idx_.size());
    double psi = 0.0;
    for (int k = 0; k < nl; k++) {
        const double d = x[b.lower_idx_[k]] - b.lower_val_[k];
        psi += -mu * std::log(d) + kKappaD * mu * b.lower_damp_[k] * d;
    }
    for (int k = 0; k < nu; k++) {
        const double d = b.upper_val_[k] - x[b.upper_idx_[k]];
        psi += -mu * std::log(d) + kKappaD * mu * b.upper_damp_[k] * d;
    }
    return psi;
}

// mu-FORM primal gradient terms: gx_i += -mu/(x_i - l_i) and += +mu/(u_i - x_i),
// plus the derivative of the one-sided damping term, += +kappa_d*mu on a
// lower-only entry and -= kappa_d*mu on an upper-only entry.
//
// This is the gradient of the barrier objective above, and it is what the
// CONDENSED NEWTON RIGHT-HAND SIDE carries. Eliminating the bound-multiplier
// rows from the primal-dual system turns the primal row's right-hand side
// (grad f + J'lambda - z_L + z_U) into exactly grad phi_mu + J'lambda: the
// -z_L + z_U cancels against the multiplier steps that were substituted in.
// Ipopt keeps the same split, damping and all: its Newton RHS reads
// curr_grad_lag_WITH_DAMPING_x while its optimality error reads the undamped
// curr_grad_lag_x. Never use this form for a residual a convergence test
// consumes.
inline void accumulate_bound_barrier_gradient(ConstEigenRef<Eigen::VectorXd> x, const BoundSet &b,
                                              double mu, EigenRef<Eigen::VectorXd> gx) {
    const int nl = static_cast<int>(b.lower_idx_.size());
    const int nu = static_cast<int>(b.upper_idx_.size());
    for (int k = 0; k < nl; k++) {
        const int i = b.lower_idx_[k];
        gx[i] += -mu / (x[i] - b.lower_val_[k]) + kKappaD * mu * b.lower_damp_[k];
    }
    for (int k = 0; k < nu; k++) {
        const int i = b.upper_idx_[k];
        gx[i] += mu / (b.upper_val_[k] - x[i]) - kKappaD * mu * b.upper_damp_[k];
    }
}

// z-FORM primal gradient terms: gx_i += -zL_i and += +zU_i.
//
// This is the DUAL INFEASIBILITY contribution -- the residual whose norm the
// convergence check consumes, grad f + J'lambda - z_L + z_U. It agrees with the
// mu-form only at a point that satisfies the bound complementarity exactly
// (z_L = mu/(x-l), z_U = mu/(u-x)); away from the central path the two differ,
// which is the whole reason both exist. Never use this form for a Newton
// right-hand side.
inline void accumulate_bound_dual_terms(const BoundSet &b, const BoundDualState &z,
                                        EigenRef<Eigen::VectorXd> gx) {
    const int nl = static_cast<int>(b.lower_idx_.size());
    const int nu = static_cast<int>(b.upper_idx_.size());
    for (int k = 0; k < nl; k++) {
        gx[b.lower_idx_[k]] += -z.z_lower_[k];
    }
    for (int k = 0; k < nu; k++) {
        gx[b.upper_idx_[k]] += z.z_upper_[k];
    }
}

// Condensed bound curvature: sigma_i += zL_i/(x_i - l_i) and += zU_i/(u_i - x_i).
//
// The Sigma that eliminating the bound-multiplier rows leaves on the Hessian
// (1,1) diagonal. Accumulated onto whatever base the solver already writes to
// the primal-diagonal slots, so it composes with the restoration and proximal
// bases instead of replacing them, and it does not grow the KKT system.
inline void accumulate_bound_sigma(ConstEigenRef<Eigen::VectorXd> x, const BoundSet &b,
                                   const BoundDualState &z, EigenRef<Eigen::VectorXd> sigma) {
    const int nl = static_cast<int>(b.lower_idx_.size());
    const int nu = static_cast<int>(b.upper_idx_.size());
    for (int k = 0; k < nl; k++) {
        const int i = b.lower_idx_[k];
        sigma[i] += z.z_lower_[k] / (x[i] - b.lower_val_[k]);
    }
    for (int k = 0; k < nu; k++) {
        const int i = b.upper_idx_[k];
        sigma[i] += z.z_upper_[k] / (b.upper_val_[k] - x[i]);
    }
}

} // namespace tycho::solvers::detail
