// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// ProximalSwitchRestoration — a RestorationStrategy that keeps the SAME
// barrier algorithm running on globalization failure, but swaps the true
// objective for a proximal term pulling the primal variables back toward the
// point restoration was entered from. This is the first of the
// feasibility-restoration trio (restoration.h); the nested l1 proximal
// restoration (l1_restoration.h) is the second, and an elastic/penalty
// relaxation is expected to follow as a separate strategy. Constructed by
// rebuild_globalization_components when Settings::restoration_mode_ selects
// the proximal switch (see restoration.h's file docstring for the wiring
// overview).
//
// =============================================================================
// FORMULATION — derived from the two primary sources (fetched + read, not
// from memory):
//   [Uno]   Vanaret's Uno solver, commit 7481abe47cec45e0e91ac73ccc2461c17e68f84c.
//           uno/ingredients/subproblem/interior_point_methods/InteriorPointMethod.hpp
//           (proximal_coefficient()) and
//           uno/ingredients/globalization_strategies/feasibility_restoration/
//           FeasibilityRestoration.cpp (switch_to_feasibility_problem, the ONE
//           call site that sets the coefficient) define the coefficient and its
//           set-once-at-switch-time timing; uno/optimization/l1RelaxedProblem.cpp
//           defines the per-coordinate scaling and the resulting proximal
//           term/gradient.
//   [Ipopt] coin-or/Ipopt, commit 72a29c9aab198afa0dbb940339022a22c415a4eb.
//           src/Algorithm/IpOptionsList / registered option "resto_proximity_weight"
//           supplies the weight default; src/Algorithm/IpBacktrackingLineSearch.cpp
//           supplies the near-feasible entry guard this class's entry_permitted()
//           adapts.
//   [Knitro] The mode-switch CONCEPT (same barrier algorithm, objective swapped
//           for a scalar-weighted proximal term rather than a nested solve)
//           follows Knitro's bar_switchrule/bar_switchobj=scalarprox feature
//           (documented behavior only — Knitro is closed-source, so entry/exit
//           mechanics here are the Uno/Ipopt-derived ones above).
//
// (1) Proximal coefficient ζ, [Uno] InteriorPointMethod::proximal_coefficient():
//
//       ζ = resto_proximity_weight · sqrt(μ)
//
//     set ONCE, at the moment restoration is entered ([Uno]
//     FeasibilityRestoration::switch_to_feasibility_problem is the single call
//     site) — NOT re-derived from μ on every subsequent iteration while
//     restoration remains active. enter_restoration() below is this component's
//     equivalent single call site: it reads the `mu` argument passed to it
//     (the barrier parameter live at that instant) and freezes ζ_ from it;
//     nothing else ever reassigns ζ_ except a later enter_restoration() call
//     (a fresh episode) or reset(). kRestoProximityWeight = 1.0 is Ipopt's
//     shipped default for the "resto_proximity_weight" option — Uno's
//     proximal_coefficient() takes this weight as a caller-supplied parameter
//     rather than hardcoding it, so citing Ipopt's option default for the
//     constant (rather than inventing a Uno-side default) is the source-
//     faithful choice here.
//
// (2) Per-coordinate scaling d_i and the proximal term, [Uno]
//     l1RelaxedProblem.cpp:
//
//       d_i = min( 1, 1 / |x_R_i| )                              (scaling)
//       P(x) = (ζ/2) · Σ_i d_i² · (x_i − x_R_i)²                 (term)
//       ∂P/∂x_i = ζ · d_i² · (x_i − x_R_i)                        (gradient)
//
//     with x_R the primal snapshot at entry (this class's `x_r_`). The
//     min(1, ·) branch means: a coordinate with |x_R_i| < 1 gets d_i = 1
//     (unscaled — proximal_center components inside the unit ball are not
//     shrunk); |x_R_i| > 1 gets d_i = 1/|x_R_i| < 1 (shrunk in proportion to
//     the center's own magnitude, so the proximal term's curvature stays
//     bounded near large-magnitude centers). |x_R_i| == 0 lands in the first
//     branch (1/0 = +∞, min(1, +∞) = 1) without special-casing division by
//     zero. `proximal_diagonal()` below returns ζ·d_i² directly (the primal-
//     diagonal Hessian block of P), so `proximal_objective` and
//     `add_proximal_gradient` are both expressed in terms of it.
//
// (3) Near-feasible entry guard, adapted from [Ipopt]
//     IpBacktrackingLineSearch.cpp: Ipopt refuses to enter restoration at an
//     almost-feasible point, testing constraint violation against a SCALED
//     tolerance (1e-2 · tol) AND an UNSCALED one (1e-1 · constr_viol_tol)
//     together. Tycho's ProgressMeasures/SolverContext carry a single
//     constraint-violation measure and a single econ_tol_, not Ipopt's
//     scaled/unscaled pair, so this is a single-measure ADAPTATION, not a
//     transcription:
//
//       refuse entry iff constraint_violation <= kNearFeasibleGuardFactor · econ_tol_
//
//     with kNearFeasibleGuardFactor = 0.1, chosen to match the UNSCALED member
//     of Ipopt's pair (1e-1 · constr_viol_tol) since Tycho's econ_tol_ is
//     itself unscaled. DISCLOSED CONSEQUENCE: because Tycho tests one measure
//     against one threshold instead of Ipopt's two-measure AND, this guard can
//     refuse entry slightly earlier or later than Ipopt's dual test would at
//     any given point where the scaled and unscaled tests would have
//     disagreed — this adaptation trades that boundary-exactness for a guard
//     that fits the single-measure ProgressMeasures/SolverContext shape
//     already established by the other globalization components.
//
// (4) Budget guard: entry_permitted() ALSO refuses once this phase's entry
//     count has reached ctx.settings_.max_feas_rest_ (Settings, psiopt.h) —
//     an int Tycho-side budget with no direct Uno/Ipopt analog (both those
//     solvers gate re-entry through their own restoration-specific state
//     machines instead of a flat per-phase counter). max_feas_rest_ == 0
//     means the budget is exhausted before the first entry, so entry is
//     always refused; the guard in (3) and the budget in (4) are independent
//     — either one refusing is enough to refuse entry_permitted() as a whole.
//
// Ownership: the ONLY state this class caches across calls is the entry
// snapshot (x_r_/d_/diagonal_/ζ_/reference_) plus the per-phase diagnostic
// counters (entries_/iterations_in_mode_) — exactly the mode's defining
// state, per restoration.h's ownership rule. No SolverContext reference or
// NLP handle is retained beyond a single call.
//
// Definitions live in src/solvers/psiopt_globalization.cpp.

#pragma once

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/restoration.h"
#include "tycho/detail/solvers/globalization/solver_context.h"

namespace tycho::solvers {

// (1) ζ weight; Ipopt option "resto_proximity_weight" shipped default 1.0.
inline constexpr double kRestoProximityWeight = 1.0;

// (3)'s guard factor kNearFeasibleGuardFactor, and the near_feasible() test
// built on it, live in restoration.h — shared by both concrete strategies and
// by the feasibility-stage stall seam. The provenance disclosure for the
// factor stays in this file's docstring, item (3) above.

// Failure-classification threshold for a restoration STALL (distinct from the
// ENTRY guard above, and three orders of magnitude looser). When the proximal
// feasibility subproblem converges/stalls while restoration is active, Ipopt
// classifies the outcome by comparing the true primal infeasibility against
// resto_failure_feasibility_threshold, default 1e2 · tol
// (IpRestoMinC_1Nrm.cpp, commit 72a29c9a): at or below it the restoration is
// treated as having reached a (near-)feasible point — a soft, recoverable
// outcome — and only above it is the problem declared locally infeasible.
// Tycho applies the same 1e2 factor to its unscaled econ_tol_ (same
// single-measure adaptation as the entry guard above). Using the entry-guard
// factor here instead would misclassify every stall with violation in
// (0.1·tol, 1e2·tol] — points Ipopt considers feasible-enough to continue
// from — as local infeasibility on feasible problems.
inline constexpr double kRestoFailureFeasibilityFactor = 1.0e2;

// =============================================================================
// ProximalSwitchRestoration — concrete proximal feasibility mode-switch.
// See the file docstring for the full formulation and citations.
// =============================================================================
class ProximalSwitchRestoration final : public RestorationStrategy {
  public:
    void enter_restoration(const ProgressMeasures &reference,
                            const Eigen::Ref<const Eigen::VectorXd> &primals,
                            double mu) override;

    void exit_restoration() override { active_ = false; }

    bool is_active() const override { return active_; }

    void reset() override;

    double proximal_objective(const Eigen::Ref<const Eigen::VectorXd> &primals) const override;

    void add_proximal_gradient(const Eigen::Ref<const Eigen::VectorXd> &primals,
                                Eigen::Ref<Eigen::VectorXd> grad_out) const override;

    const Eigen::VectorXd &proximal_diagonal() const override { return diagonal_; }

    bool entry_permitted(double constraint_violation, const SolverContext &ctx) const override;

    const ProgressMeasures &reference() const override { return reference_; }

    void note_iteration() override { ++iterations_in_mode_; }

    // Reports entries_/iterations_in_mode_ into SolveResult::
    // last_feas_rest_entries_/last_feas_rest_iters_ (psiopt.h) — see
    // RestorationStrategy::append_diagnostics() for the call-site contract
    // this overrides. Written unconditionally (0/0 is a correct report for a
    // strategy that was constructed but never entered).
    void append_diagnostics(PSIOPT::SolveResult &result) const override;

    // --- Test/diagnostic observers ---
    double zeta() const { return zeta_; }
    const Eigen::VectorXd &scaling() const { return d_; }
    const Eigen::VectorXd &snapshot() const { return x_r_; }
    int entries() const { return entries_; }
    int iterations_in_mode() const { return iterations_in_mode_; }

  private:
    bool active_ = false;
    ProgressMeasures reference_;

    // Entry snapshot — the mode's defining state (see the ownership note
    // above). x_r_ is the primal center; d_ the per-coordinate scaling (2);
    // diagonal_ = ζ·d_i² is cached so proximal_objective/add_proximal_gradient
    // never recompute it.
    Eigen::VectorXd x_r_;
    Eigen::VectorXd d_;
    Eigen::VectorXd diagonal_;
    double zeta_ = 0.0;

    // Per-phase diagnostics (write-only, see append_diagnostics()).
    int entries_ = 0;
    int iterations_in_mode_ = 0;
};

} // namespace tycho::solvers
