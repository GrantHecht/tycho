// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// Classification products for primal variable bounds: the set of finite bounds
// the solver must keep barrier terms for, and the dual state that will be
// attached to them.
// =============================================================================

#pragma once

#include <Eigen/Core>

namespace tycho::solvers {

/// <summary>
/// The finite variable bounds a solve has to honor, split into a lower-bound
/// list and an upper-bound list and stored as index/value pairs rather than as
/// dense per-variable vectors: the barrier and fraction-to-boundary loops walk
/// only the bounded variables, and on a problem with no variable bounds both
/// lists are empty so those loops have no trip count at all.
///
/// Produced by NonLinearProgram::configure_variable_treatment and owned by the
/// NonLinearProgram (see NonLinearProgram::variable_bound_set()). Variables
/// eliminated by the fixed-variable treatment never appear here — a variable
/// pinned at l == u has no barrier term, its bound is enforced exactly. A
/// two-sided variable appears in BOTH lists.
///
/// Index space: entries are primal-variable indices in the REDUCED space — the
/// space the solver iterates in, which is the problem's own space minus the
/// variables the fixed-variable treatment eliminated. They index the solver's
/// primal vectors and the KKT primal block directly, with no translation.
/// NonLinearProgram::reduced_to_full() maps them back to the problem's own
/// numbering for a caller that needs to name the variable a bound came from.
///
/// Values are the RELAXED bounds: configure_variable_treatment widens each
/// finite bound by bound_relax_factor * max(1, |bound|) before recording it, so
/// consumers do not have to re-apply the relaxation. A relax factor of 0.0
/// records the declared bounds verbatim.
/// </summary>
struct BoundSet {
    Eigen::VectorXi lower_idx_; ///< Variable indices carrying a finite lower bound.
    Eigen::VectorXd lower_val_; ///< Relaxed lower bound, parallel to lower_idx_.
    Eigen::VectorXi upper_idx_; ///< Variable indices carrying a finite upper bound.
    Eigen::VectorXd upper_val_; ///< Relaxed upper bound, parallel to upper_idx_.

    /// Damping indicators, parallel to the corresponding index list: 1.0 when
    /// that entry's variable is bounded on THIS side only, 0.0 when it is
    /// two-sided. They are the classification's answer to "does this bound
    /// leave the variable free to run to infinity in the other direction",
    /// which only the classifier can answer cheaply -- it sees both endpoints
    /// of a variable at once, while the two lists above no longer pair them.
    ///
    /// Recorded here rather than derived at use because the consumer is the
    /// barrier objective and its gradient, on the per-iteration path: pairing
    /// the lists there would cost a dense lookup per evaluation. Mirrors
    /// Ipopt's dampind_x_L / dampind_x_U vectors, which are likewise
    /// materialized once (IpIpoptCalculatedQuantities::ComputeDampingIndicators).
    Eigen::VectorXd lower_damp_;
    Eigen::VectorXd upper_damp_;

    /// True iff at least one variable carries a finite bound the solver must
    /// keep a barrier term for. False on an unbounded problem AND on a problem
    /// whose only bounds fixed their variables (those are eliminated, not
    /// barriered).
    bool any() const { return this->lower_idx_.size() > 0 || this->upper_idx_.size() > 0; }

    /// Drops every recorded bound, leaving the set in its unbounded state.
    void clear() {
        this->lower_idx_.resize(0);
        this->lower_val_.resize(0);
        this->upper_idx_.resize(0);
        this->upper_val_.resize(0);
        this->lower_damp_.resize(0);
        this->upper_damp_.resize(0);
    }
};

/// <summary>
/// Bound-multiplier state for the variables in a BoundSet: the multipliers
/// themselves and the Newton step computed for them each iteration. Parallel to
/// the corresponding BoundSet list (z_lower_ / dz_lower_ index-align with
/// BoundSet::lower_idx_, z_upper_ / dz_upper_ with BoundSet::upper_idx_).
///
/// Solver-owned rather than NLP-owned: unlike the bound set itself, which is a
/// property of the problem, these are iterate state that belongs to whichever
/// solve is running. PSIOPT sizes and seeds them at solve entry (the interior
/// push), fills the step vectors from each Newton solve, and commits the
/// multipliers once per accepted iterate.
/// </summary>
struct BoundDualState {
    Eigen::VectorXd z_lower_;  ///< Lower-bound multipliers, parallel to BoundSet::lower_idx_.
    Eigen::VectorXd z_upper_;  ///< Upper-bound multipliers, parallel to BoundSet::upper_idx_.
    Eigen::VectorXd dz_lower_; ///< Newton step for z_lower_.
    Eigen::VectorXd dz_upper_; ///< Newton step for z_upper_.
};

/// Cap on how far a bound multiplier may drift from the primal-dual central
/// path before it is projected back (Ipopt's kappa_sigma default): z is clamped
/// into [mu / (kKappaSigma * d), kKappaSigma * mu / d] for bound distance d.
/// Applied once per accepted iterate, against the distances at the new point.
inline constexpr double kKappaSigma = 1.0e10;

/// Cap applied when initializing a bound multiplier from the barrier parameter
/// and the initial bound distance, so a point started very close to a bound
/// cannot seed an enormous multiplier.
inline constexpr double kBoundMultInitCap = 1.0e3;

/// Cap on the barrier parameter the kappa_sigma clamp uses in FREE-mu mode,
/// where that parameter is the average complementarity at the new point rather
/// than the barrier parameter itself (Ipopt IpIpoptAlg.cpp
/// correct_bound_multiplier: `mu = Min(trial_avrg_compl, 1e3)` under
/// FreeMuMode, `mu = curr_mu` otherwise).
inline constexpr double kFreeModeClipMuCap = 1.0e3;

/// Coefficient of the linear damping term the barrier objective carries for
/// variables bounded on ONE side only (Ipopt's kappa_d default). Its purpose is
/// to stop such a variable running to infinity in its unbounded direction,
/// which the log barrier alone does nothing to discourage. It enters phi_mu as
/// `+ kKappaD * mu * distance` and the mu-form gradient as `+/- kKappaD * mu`,
/// on one-sided entries only -- and NOWHERE else: Ipopt keeps it out of the
/// dual-infeasibility residual (curr_grad_lag_x) and out of sigma, so the
/// convergence account and the condensed curvature stay undamped.
inline constexpr double kKappaD = 1.0e-5;

} // namespace tycho::solvers
