// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Proximal primal-dual KKT regularization — constants and pure-logic helpers
// for the `proximal_regularization` inertia mode (PSIOPT::Settings::
// inertia_mode_). This is an alternative to the classic on-demand inertia
// ladder that lives inline in PSIOPT::factor_impl: instead of first attempting
// an unperturbed factorization every iteration and only shifting the Hessian
// diagonal when the factorization reports wrong inertia, the mode carries a
// small persistent primal shift AND an always-on barrier-scaled dual shift into
// the base matrix, then lets the same ladder escalate on top when needed.
// (The classic ladder also borrows δ_c below on demand — engaged at most once
// per phase when a factorization reports the singularity signal, see
// PSIOPT::factor_impl — so δ_c's constants are shared by both modes; what is
// exclusive to this mode is applying both shifts always-on, in the base
// matrix.)
//
// The two shifts and their references:
//
//   * Dual shift δ_c (always on, out of restoration). Every iteration a
//     −δ_c is applied to all constraint-row diagonals, with
//     δ_c = kDualRegScale · μ^kDualRegExponent (μ the barrier parameter the
//     iteration's KKT assembly used). This is exactly Ipopt's
//     `perturb_always_cd` semantics — the one persistent-regularization knob
//     Ipopt ships — with `jacobian_regularization_value` (1e-8) as the
//     prefactor and `jacobian_regularization_exponent` (0.25) as the μ
//     exponent (coin-or/Ipopt at 72a29c9aab198afa0dbb940339022a22c415a4eb,
//     IpPDPerturbationHandler.cpp delta_cd(); Wächter & Biegler,
//     Math. Program. 106(1):25-57, 2006). The negative sign on the constraint
//     block is the augmented-system convention (W + δ_w·I on the Hessian block,
//     −δ_c·I on the constraint block): it keeps one negative eigenvalue per
//     constraint row, so the target inertia count is unchanged, while making a
//     rank-deficient constraint Jacobian factorizable. δ_c shrinks to zero with
//     μ, so it never masks non-convergence (the convergence residuals are read
//     from the raw evaluations, never from the shifted matrix).
//
//   * Primal base shift ρ_k (persistent, decaying). A proximal-point /
//     primal-regularization term +ρ_k on the Hessian diagonal, in the
//     Cipolla–Gondzio / Friedlander–Orban lineage of proximal-stabilized
//     interior-point methods (S. Cipolla & J. Gondzio, arXiv:2205.01775, 2022 /
//     JOTA 197 (2023) 1061-1103; M. P. Friedlander & D. Orban, Math. Prog.
//     Comp. 4 (2012) 71-107). Those methods hold ρ fixed (CG) or decrease it
//     geometrically per outer proximal step with a floor (FO: ρ₀=1, ÷10/step,
//     floor 1e-8). ρ is the Lagrange multiplier of an implicit trust region on
//     the primal step: larger ρ ⇔ shorter, more conservative steps; ρ→0 ⇔ the
//     full Newton step. Here ρ_k is initialized at kProxRegFloor (the
//     Cipolla–Gondzio 1e-10 floor) and, each iteration, decays by the ladder's
//     own decrease factor toward that floor.
//
// The ρ_k dynamics are a Tycho-original composition and carry NO direct
// literature reference. The convex-QP proximal literature holds ρ fixed within
// an inner solve (CG) or decreases it only per OUTER proximal step (FO); there
// is no published always-on, per-iteration decaying-primal-shift rule for
// nonconvex NLP, and Ipopt ships no always-on primal (Hessian-block) mode at
// all — only the constraint-block `perturb_always_cd`. The design instead
// mirrors the classic ladder's own warm-start memory: on a healthy problem ρ_k
// sits at the floor (numerically negligible, parity by construction); on
// curvature-troubled stretches the informed base attempt replaces the wasted
// unperturbed attempt the classic ladder pays for (the very waste the Zfac
// cycling heuristic exists to avoid — which is why that heuristic is bypassed
// under this mode). Its acceptance evidence is the solver corpus A/B, not a
// convergence theorem: the maximal-monotonicity argument behind the proximal
// outer rate holds only for convex problems.
//
// Nested-restoration suppression (disclosed consequence). While a nested l1
// restoration phase is active the dual shift δ_c is NOT applied. The elastic
// pivots already occupy and regularize the constraint-row diagonals with
// magnitude ~1/μ (dwarfing δ_c ~ 1e-8·μ^0.25), and the condensed elastic
// step-recovery algebra assumes the (y,y) diagonal equals the elastic pivot
// EXACTLY — stacking −δ_c on top would make the recovered elastic steps
// inconsistent with the solved system. The primal base shift ρ_k stays on
// throughout (it touches only the Hessian diagonal, which the nested phase does
// not condense onto the constraint rows), and the proximal mode-switch
// restoration touches only the primal diagonal too, so δ_c stays on under it —
// no slot conflict there.
//
// Orthogonality to Pardiso static pivoting. δ_c is a problem-level constraint
// regularization; the MKL Pardiso static tiny-pivot perturbation
// (Settings::qp_pivot_perturb_) is a factorization-internal guard on
// near-zero pivots. They address different failure modes and neither subsumes
// the other; this mode does not adjust the Pardiso pivot exponent.

#pragma once

#include <algorithm>
#include <cmath>

namespace tycho::solvers {

// Floor for the persistent primal base shift ρ_k, and its initial value ρ₀.
// The Cipolla–Gondzio regularization floor (arXiv:2205.01775, eq. (19):
// ρ = δ = max{1/max(‖A‖∞,‖H‖∞), 1e-10}).
inline constexpr double kProxRegFloor = 1.0e-10;

// Prefactor of the barrier-scaled dual shift δ_c = kDualRegScale · μ^kDualRegExponent.
// Ipopt `jacobian_regularization_value` (bar-delta_c) default, at pinned commit
// 72a29c9aab198afa0dbb940339022a22c415a4eb.
inline constexpr double kDualRegScale = 1.0e-8;

// μ exponent of the dual shift. Ipopt `jacobian_regularization_exponent`
// (kappa_c) default, same commit. δ_c therefore shrinks to zero as μ → 0.
inline constexpr double kDualRegExponent = 0.25;

// Barrier-scaled dual shift magnitude δ_c(μ) = kDualRegScale · μ^kDualRegExponent
// (Ipopt delta_cd()). Returned as a non-negative magnitude; the negative sign
// on the constraint block is imposed by the KKT assembly (−δ_c on the
// constraint-row diagonals).
inline double dual_regularization(double mu) {
    return kDualRegScale * std::pow(mu, kDualRegExponent);
}

// One decay step of the persistent primal base shift toward its floor:
// ρ_{k+1} = max(kProxRegFloor, applied_total · decr). `applied_total` is the
// primal shift actually carried by the factorization this iteration — ρ_k alone
// when the base attempt sufficed, or ρ_k plus the last successful ladder delta
// when the ladder fired — so the decayed total successful shift persists into
// the next iteration.
inline double prox_reg_decay(double applied_total, double decr) {
    return std::max(kProxRegFloor, applied_total * decr);
}

} // namespace tycho::solvers
