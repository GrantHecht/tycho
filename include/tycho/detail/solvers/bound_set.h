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
/// Index space: entries are primal-variable indices in the space the solver
/// iterates over. Under the MakeParameter treatment as implemented today, the
/// solver's primal space is still the full primal space (eliminated variables
/// remain as pinned, decoupled coordinates — see the reduction note on
/// NonLinearProgram::is_reduced), so these are full-space indices and
/// NonLinearProgram::full_to_reduced() maps them into the compacted numbering
/// if a caller needs it.
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
/// solve is running.
///
/// Declared here alongside the set it indexes; the barrier terms that fill it
/// are not part of the fixed-variable reduction and no code writes these
/// vectors yet.
/// </summary>
struct BoundDualState {
    Eigen::VectorXd z_lower_;  ///< Lower-bound multipliers, parallel to BoundSet::lower_idx_.
    Eigen::VectorXd z_upper_;  ///< Upper-bound multipliers, parallel to BoundSet::upper_idx_.
    Eigen::VectorXd dz_lower_; ///< Newton step for z_lower_.
    Eigen::VectorXd dz_upper_; ///< Newton step for z_upper_.
};

/// Cap on how far a bound multiplier may drift from the primal-dual central
/// path before it is projected back (Ipopt's kappa_sigma): z is clamped into
/// [mu / (kKappaSigma * d), kKappaSigma * mu / d] for bound distance d. Not
/// read yet — it belongs to the bound-multiplier contract this header defines.
inline constexpr double kKappaSigma = 1.0e10;

/// Cap applied when initializing a bound multiplier from the barrier parameter
/// and the initial bound distance, so a point started very close to a bound
/// cannot seed an enormous multiplier. Not read yet — same contract note as
/// kKappaSigma.
inline constexpr double kBoundMultInitCap = 1.0e3;

} // namespace tycho::solvers
