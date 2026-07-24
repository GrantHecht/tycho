// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// NestedL1Restoration — a RestorationStrategy that solves the l1 elastic
// feasibility reformulation as a CONDENSED in-place phase, reusing the outer
// barrier algorithm's KKT system rather than spinning up a separate nested
// solver instance. Second of the feasibility-restoration trio (restoration.h);
// the proximal mode-switch (proximal_restoration.h) precedes it and an
// elastic/penalty relaxation is expected to follow. Constructed by
// rebuild_globalization_components when Settings::restoration_mode_ selects the
// nested l1 mode; the evaluation/step seam and the entry/exit orchestration
// live in src/solvers/psiopt.cpp, definitions in
// src/solvers/psiopt_globalization.cpp.
//
// =============================================================================
// FORMULATION — transcribed from pinned source (fetched + read, not memory):
//   [Ipopt] coin-or/Ipopt, commit 72a29c9aab198afa0dbb940339022a22c415a4eb.
//     src/Algorithm/IpRestoIpoptNLP.{hpp,cpp}          — the restoration NLP,
//         the reference-point scaling D_R, and Eta(mu).
//     src/Algorithm/IpRestoIterateInitializer.cpp      — the closed-form
//         elastic slack initialization and entry barrier parameter.
//     src/Algorithm/IpRestoMinC_1Nrm.cpp               — the penalty parameter
//         and bound-multiplier reset threshold defaults.
//
// The elastic reformulation, per constraint row with residual value c
// (equality rows: c = h_j(x); inequality rows: c = g_j(x) + s_j):
//
//     row:        c + n − p = 0,   n, p ≥ 0
//                 (Ipopt sign convention c(x) − p_c + n_c = 0, IpRestoIpoptNLP.cpp)
//     objective:  ρ·Σ(n + p) + (η(μ)/2)·‖D_R (x − x_R)‖²
//
// (1) Penalty parameter ρ = kRestoPenaltyParameter = 1e3, Ipopt option
//     "resto_penalty_parameter" default (IpRestoIpoptNLP.cpp RegisterOptions).
//
// (2) Proximity weight η(μ) = kRestoProximityWeight · sqrt(μ). Ipopt's
//     RestoIpoptNLP::Eta(mu) = eta_factor_ · mu^0.5 with eta_factor_ the option
//     "resto_proximity_weight" (default 1.0) and the exponent hardcoded 0.5 in
//     the constructor. η is a METHOD of μ, recomputed on every evaluation — see
//     disclosure (2) below.
//
// (3) Reference scaling D_R = diag(1 / max(1, |x_R_i|)) (IpRestoIpoptNLP.cpp
//     InitializeStructures). dr2_ caches d_i² for the gradient/Hessian pieces.
//     Identical to the proximal-switch per-coordinate scaling.
//
// (4) Closed-form elastic slack initialization (IpRestoIterateInitializer.cpp
//     SetInitialIterates / solve_quadratic). Trust the code body, NOT the header
//     doc comment — the header's stated quadratic sign is inconsistent with the
//     implemented arithmetic (verified). For each row with residual c and the
//     restoration barrier parameter resto_mu:
//
//         k   = resto_mu / (2 ρ)
//         a   = k − c/2
//         b   = c · k
//         n   = a + sqrt(a² + b)      (positive root of n² − 2 a n − b = 0)
//         p   = c + n
//         z_n = resto_mu / n,   z_p = resto_mu / p
//
//     resto_mu = max(μ_outer, ‖h‖∞, ‖g+s‖∞) at entry (SetInitialIterates); this
//     is also the phase's starting barrier parameter (entry_mu()). The c = 0
//     edge gives a = resto_mu/(2ρ) > 0, b = 0, so n = 2a = p (no division by
//     zero). n, p > 0 for mixed-sign c (l1_elastic_slack_init verifies this).
//
// (5) Per-iteration block condensation (the elastic pair (n,p) and their bound
//     multipliers (z_n,z_p) are eliminated analytically per row; verified
//     against a dense enlarged-vs-condensed solve). With μ the LIVE barrier
//     parameter, y the current constraint multiplier, and the current n,p,z:
//
//         pivot   = n/z_n + p/z_p                      (POSITIVE; the seam
//                     negates it into the KKT (y,y) diagonal entry)
//         r̃      = (c + n − p) + μ/z_n − (n/z_n)(ρ+y)
//                     − μ/z_p + (p/z_p)(ρ−y)           (condensed row RHS)
//
//     Step recovery after the condensed solve (Δy the row's multiplier step):
//
//         Δn   = μ/z_n − (n/z_n)(ρ+y) − (n/z_n)·Δy
//         Δp   = μ/z_p − (p/z_p)(ρ−y) + (p/z_p)·Δy
//         Δz_n = Δy + ρ + y − z_n
//         Δz_p = −Δy + ρ − y − z_p
//
//     Fraction-to-boundary: n,p join the primal cap, z_n,z_p the dual cap, via
//     the standard τ rule (max α ∈ (0,1] keeping each positive variable v with
//     v + αΔv ≥ (1−τ)v).
//
// =============================================================================
// DESIGN DECISIONS (disclosed deviations from a literal Ipopt restoration solve,
// stated with their consequences):
//
//   (a) Condensed in-place, not a literal second solver. Ipopt builds a
//       separate 5-block NLP and runs a nested algorithm instance; this
//       component solves the SAME restoration problem with the same per-step
//       algebra (proven by block-elimination equivalence) inside the outer
//       loop. Consequence: the phase shares the outer machinery rather than a
//       fresh inner algorithm; iteration-level trajectories differ from Ipopt's
//       even where the per-step algebra matches, and there is no inner/outer
//       iteration-count split — the phase counts iterations with the in-mode
//       counter. A further structural consequence is a CONDITIONING asymmetry:
//       the elimination concentrates the elastic barrier curvature into a
//       single constraint-row pivot (n/z_n + p/z_p, which grows like 1/μ as
//       the barrier parameter falls) where Ipopt's enlarged system carries the
//       same information in separate bounded diagonal blocks. The condensed
//       KKT is therefore more sensitive to a barrier parameter that runs ahead
//       of the elastic complementarity — the reason the phase's barrier
//       schedule feeds the elastic pairs to the oracle and runs the monotone
//       safeguard (see the barrier-schedule notes in psiopt.cpp); the
//       asymmetry itself is inherent to the condensation and is the price of
//       avoiding the enlarged system.
//
//   (b) η(μ) is recomputed live on every evaluation (η = kRestoProximityWeight·
//       sqrt(μ)), matching Ipopt's Eta(mu) method — UNLIKE the proximal-switch
//       mode, which freezes its proximal coefficient ζ at entry. The difference
//       stays internal to each mode.
//
//   (c) Constraint multipliers on exit are reset to zero, not recomputed by a
//       least-squares estimate. Ipopt's re-entry path calls least_square_mults
//       with the reset threshold at its shipped default 0, whose body only
//       computes the LSQ estimate when the threshold is > 0 — at the default it
//       sets the multipliers to zero. This component transcribes the shipped-
//       default behavior (multipliers ← 0) and does not implement the dormant
//       LSQ branch (no knob exposes it). Entry likewise starts the phase with
//       multipliers at zero.
//
//   (d) Single-measure floors. Ipopt carries separate scaled/unscaled
//       tolerances; Tycho carries one econ_tol_. The entry guard and the stall
//       failure classification both use econ_tol_ (the same single-measure
//       adaptation as the proximal switch, whose constants this component
//       reuses). Consequence: boundary behavior can differ from Ipopt where its
//       two tolerances would diverge.
//
//   (e) No separate restoration iteration budget. Ipopt's max_resto_iter default
//       is effectively unbounded; the outer iteration limit already bounds this
//       in-place phase, so no new knob is introduced. The per-solve ENTRY budget
//       remains the shared max_feas_rest_ (Settings), gated by entry_permitted()
//       exactly as the proximal switch gates it.
//
// Ownership: the only state cached across calls is the entry snapshot
// (x_r_/dr2_/resto_mu_), the live elastic state (n,p,z per channel), the last
// recovered steps (Δn,Δp,Δz per channel), the cached pivots, and the per-phase
// diagnostic counters — exactly the mode's defining state, per restoration.h's
// ownership rule. No SolverContext reference or NLP handle is retained across
// calls.

#pragma once

#include <Eigen/Core>

#include <cmath>

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/proximal_restoration.h"
#include "tycho/detail/solvers/globalization/restoration.h"
#include "tycho/detail/solvers/globalization/solver_context.h"

namespace tycho::solvers {

// (1) Penalty parameter ρ; Ipopt option "resto_penalty_parameter" default 1e3.
inline constexpr double kRestoPenaltyParameter = 1.0e3;

// Threshold above which the re-entry bound-multiplier update resets all slack
// multipliers to 1; Ipopt option "bound_mult_reset_threshold" default 1e3
// (IpRestoMinC_1Nrm.cpp). Consumed by the re-entry sequence (later wiring); the
// component exposes it here so both live at the same literature default.
inline constexpr double kBoundMultResetThreshold = 1.0e3;

// kRestoProximityWeight, kNearFeasibleGuardFactor, and
// kRestoFailureFeasibilityFactor are shared with the proximal switch
// (proximal_restoration.h) — included above; not redefined here.

// =============================================================================
// Closed-form elastic slack initialization for one constraint row (4).
// Exposed as a free function so it is directly unit-testable at a chosen
// resto_mu independent of the entry max-rule.
// =============================================================================
struct ElasticSlackInit {
    double n;  ///< positive slack, root of n² − 2 a n − b = 0.
    double p;  ///< paired slack, p = c + n.
    double zn; ///< bound multiplier resto_mu / n.
    double zp; ///< bound multiplier resto_mu / p.
};

inline ElasticSlackInit l1_elastic_slack_init(double c, double resto_mu, double rho) {
    const double k = resto_mu / (2.0 * rho);
    const double a = k - 0.5 * c;
    const double b = c * k;
    const double n = a + std::sqrt(a * a + b);
    const double p = c + n;
    return ElasticSlackInit{n, p, resto_mu / n, resto_mu / p};
}

// =============================================================================
// NestedL1Restoration — condensed l1 elastic feasibility restoration.
// See the file docstring for the full formulation, citations, and disclosures.
// =============================================================================
class NestedL1Restoration final : public RestorationStrategy {
  public:
    // --- Base RestorationStrategy surface ---

    // Guard-only entry: records the reference point and proximal center for the
    // entry-permission counters/budget. The full elastic initialization runs in
    // enter_nested(), which the nested solver path calls instead.
    void enter_restoration(const ProgressMeasures &reference,
                           const Eigen::Ref<const Eigen::VectorXd> &primals, double mu) override;

    void exit_restoration() override { active_ = false; }

    bool is_active() const override { return active_; }

    void reset() override;

    // The nested mode does not use the frozen-ζ proximal trio (it has its own
    // live-μ nested_* pieces); reaching these marks a wiring bug — they throw.
    double proximal_objective(const Eigen::Ref<const Eigen::VectorXd> &primals) const override;
    void add_proximal_gradient(const Eigen::Ref<const Eigen::VectorXd> &primals,
                               Eigen::Ref<Eigen::VectorXd> grad_out) const override;
    const Eigen::VectorXd &proximal_diagonal() const override;

    bool entry_permitted(double constraint_violation, const SolverContext &ctx) const override;

    const ProgressMeasures &reference() const override { return reference_; }

    void note_iteration() override { ++iterations_in_mode_; }

    void append_diagnostics(PSIOPT::SolveResult &result) const override;

    // --- Nested restoration surface (see restoration.h for the contract) ---

    bool is_nested() const override { return true; }

    void enter_nested(const ProgressMeasures &reference,
                      const Eigen::Ref<const Eigen::VectorXd> &primals,
                      const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
                      const Eigen::Ref<const Eigen::VectorXd> &iq_residuals,
                      double outer_mu) override;

    double entry_mu() const override { return resto_mu_; }

    const Eigen::VectorXd &e_pivots() const override { return e_pivots_; }
    const Eigen::VectorXd &i_pivots() const override { return i_pivots_; }

    void nested_complementarity(double &sum, double &min_comp, double &max_comp,
                                int &count) const override;

    void condensed_residuals(double mu, const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
                             const Eigen::Ref<const Eigen::VectorXd> &iq_residuals,
                             const Eigen::Ref<const Eigen::VectorXd> &eq_lmults,
                             const Eigen::Ref<const Eigen::VectorXd> &iq_lmults,
                             Eigen::Ref<Eigen::VectorXd> eq_rtilde_out,
                             Eigen::Ref<Eigen::VectorXd> iq_rtilde_out) const override;

    double nested_objective(double mu,
                            const Eigen::Ref<const Eigen::VectorXd> &primals) const override;
    void add_nested_gradient(double mu, const Eigen::Ref<const Eigen::VectorXd> &primals,
                             Eigen::Ref<Eigen::VectorXd> grad_out) const override;
    void nested_primal_diagonal(double mu, Eigen::Ref<Eigen::VectorXd> diag_out) const override;

    void recover_elastic_steps(double mu, const Eigen::Ref<const Eigen::VectorXd> &eq_lmults,
                               const Eigen::Ref<const Eigen::VectorXd> &iq_lmults,
                               const Eigen::Ref<const Eigen::VectorXd> &eq_dy,
                               const Eigen::Ref<const Eigen::VectorXd> &iq_dy) override;

    double primal_boundary_alpha(double tau) const override;
    double dual_boundary_alpha(double tau) const override;
    void apply_elastic_step(double alpha_primal, double alpha_dual) override;

    double trial_objective(double mu, double alpha,
                           const Eigen::Ref<const Eigen::VectorXd> &trial_primals) const override;
    void trial_residual_shift(double alpha, Eigen::Ref<Eigen::VectorXd> eq_shift_out,
                              Eigen::Ref<Eigen::VectorXd> iq_shift_out) const override;

    // --- Test/diagnostic observers ---
    const Eigen::VectorXd &dr2() const { return dr2_; }
    const Eigen::VectorXd &snapshot() const { return x_r_; }
    double resto_mu() const { return resto_mu_; }
    int entries() const { return entries_; }
    int iterations_in_mode() const { return iterations_in_mode_; }

    const Eigen::VectorXd &ec_n() const { return n_e_; }
    const Eigen::VectorXd &ec_p() const { return p_e_; }
    const Eigen::VectorXd &ec_zn() const { return z_ne_; }
    const Eigen::VectorXd &ec_zp() const { return z_pe_; }
    const Eigen::VectorXd &ic_n() const { return n_i_; }
    const Eigen::VectorXd &ic_p() const { return p_i_; }
    const Eigen::VectorXd &ic_zn() const { return z_ni_; }
    const Eigen::VectorXd &ic_zp() const { return z_pi_; }

    const Eigen::VectorXd &ec_dn() const { return dn_e_; }
    const Eigen::VectorXd &ec_dp() const { return dp_e_; }
    const Eigen::VectorXd &ec_dzn() const { return dzn_e_; }
    const Eigen::VectorXd &ec_dzp() const { return dzp_e_; }
    const Eigen::VectorXd &ic_dn() const { return dn_i_; }
    const Eigen::VectorXd &ic_dp() const { return dp_i_; }
    const Eigen::VectorXd &ic_dzn() const { return dzn_i_; }
    const Eigen::VectorXd &ic_dzp() const { return dzp_i_; }

  private:
    // Initializes one channel's elastic state from residual values.
    void init_channel(const Eigen::Ref<const Eigen::VectorXd> &residuals, Eigen::VectorXd &n,
                      Eigen::VectorXd &p, Eigen::VectorXd &zn, Eigen::VectorXd &zp) const;
    // Recomputes a channel's pivot vector from its live elastic state.
    static void update_pivots(const Eigen::VectorXd &n, const Eigen::VectorXd &p,
                              const Eigen::VectorXd &zn, const Eigen::VectorXd &zp,
                              Eigen::VectorXd &pivots);

    bool active_ = false;
    ProgressMeasures reference_;

    // Entry snapshot (proximal center and its cached D_R²) and entry barrier.
    Eigen::VectorXd x_r_;
    Eigen::VectorXd dr2_;
    double resto_mu_ = 0.0;

    // Live elastic state — equality channel (EC) and inequality channel (IC).
    Eigen::VectorXd n_e_, p_e_, z_ne_, z_pe_;
    Eigen::VectorXd n_i_, p_i_, z_ni_, z_pi_;

    // Last recovered steps per channel (set by recover_elastic_steps).
    Eigen::VectorXd dn_e_, dp_e_, dzn_e_, dzp_e_;
    Eigen::VectorXd dn_i_, dp_i_, dzn_i_, dzp_i_;

    // Cached pivot vectors (recomputed at entry and after apply_elastic_step).
    Eigen::VectorXd e_pivots_, i_pivots_;

    // Per-phase diagnostics (write-only, see append_diagnostics()).
    int entries_ = 0;
    int iterations_in_mode_ = 0;
};

} // namespace tycho::solvers
