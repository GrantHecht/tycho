// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction. Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3
// ("Component architecture (G1)"). Ground-truth recon: state inventory table,
// docs/superpowers/plans/2026-07-16-e2-g1-dossier.md §6.
//
// G1 (this file): declares SolverContext, a references-only aggregate that
// bundles the persistent PSIOPT state the globalization components need to
// read (and, for a few scratch buffers, write). NOT filled in with usage
// yet — no component reads from a SolverContext instance until Task 2+
// wires the call sites. The member list here is exactly the dossier §6
// state-inventory rows whose "prospective owner" is one of the globalization
// components (Acceptance / Globalization / BarrierGovernor / RecoveryChain)
// or "shared". Members NOT needed by any of the seven Task-1 interfaces
// (e.g. result_.*, qp_analyzed_, callbacks) are intentionally omitted; the
// brief documents that this list "grows per task" as later tasks wire real
// call sites — do not treat this as the final member set.
//
// Ownership rule: SolverContext owns nothing. Every member is a reference or
// a non-owning pointer into the live PSIOPT instance; a SolverContext must
// not outlive the PSIOPT it was built from (same lifetime discipline as
// PSIOPT::KKTVector, psiopt.h). It is constructed fresh (as a temporary) at
// each call site that needs it — never stored across iterations by a
// component. Components hold NO solver state themselves; SolverContext (plus
// the explicit per-iteration transient parameters threaded through each
// interface method, e.g. mu/prim_obj/barr_obj) is the only channel by which
// they observe or mutate PSIOPT's persistent state.

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

// SolverContext exposes PSIOPT::Settings by reference and the concrete KKT
// solver type: both are only visible once the PSIOPT class itself has been
// parsed (Settings is a nested struct; the KKT solver type mirrors
// PSIOPT::kkt_sol_'s declaration exactly, macro-guarded the same way psiopt.h
// guards it). This mirrors the existing include discipline used by other
// downstream consumers of the PSIOPT class (see
// include/tycho/detail/solvers/optimization_problem_base.h, which also
// #includes psiopt.h directly) — psiopt.h itself does NOT include this
// directory (see psiopt.cpp for where these headers are pulled into the
// build instead); that one-directional arrangement is what keeps every
// header below self-sufficient/standalone-compilable without a fragile
// circular-include trick. See the Task 1 report for the full rationale.
#include "tycho/detail/solvers/psiopt.h"

#ifdef USE_ACCELERATE_SPARSE
#include "tycho/detail/solvers/linear/accelerate_interface.h"
#else
#include "tycho/detail/solvers/linear/pardiso_interface.h"
#endif

namespace tycho::solvers {

// The concrete sparse KKT factorization type, mirroring PSIOPT::kkt_sol_'s
// declaration (psiopt.h) exactly — same macro guard, same template
// arguments. Components never choose or construct this type; they only ever
// see it through SolverContext::kkt_solver, driving solves/refactors that
// PSIOPT itself still owns.
#ifdef USE_ACCELERATE_SPARSE
using KktSolverType =
    Eigen::AccelerateLDLTTPP<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper>;
#else
using KktSolverType =
    Eigen::PardisoLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper>;
#endif

// =============================================================================
// SolverContext — references-only view into the live PSIOPT instance.
//
// Every member below is documented with which dossier §6 row it corresponds
// to and which component(s) read (or read+write) it today per the recon;
// this is the "grows per task" list the brief calls for.
// =============================================================================
struct SolverContext {
    // --- NLP / KKT solve machinery ---
    // dossier §6: `nlp_` — "eval_*, barrier_hessian, perturb, diags" — read by
    // Acceptance (eval_trial_point_occ/eval_rhs) and BarrierGovernor
    // (barrier_hessian) call sites once wired. Raw, non-owning pointer:
    // PSIOPT retains the owning std::shared_ptr<NonLinearProgram>.
    NonLinearProgram *nlp;

    // dossier §6: `kkt_sol_` — "factor_impl, all solves, ppivs/neigs/peigs" —
    // read (solve) by BarrierGovernor's PROBE predictor solve and by the
    // (not-yet-extracted) main step solve; written (factor/refactor) by
    // RecoveryChain's future inertia-ladder dispatch. Reference: PSIOPT
    // retains ownership of the actual Eigen solver object.
    KktSolverType &kkt_solver;

    // dossier §6: `settings_.*` rows — every globalization component reads a
    // disjoint subset (max_ls_iters_/alpha_red_ -> Acceptance;
    // bound_fraction_/pd_step_strategy_ -> Globalization;
    // opt_bar_mode_/soe_bar_mode_/init_mu_/min_mu_/max_mu_ -> BarrierGovernor;
    // delta_h_/incr_h_/decr_h_/max_refac_ -> RecoveryChain;
    // econ_tol_/icon_tol_ and the div_*/acc_*/kkt_tol_ family -> shared with
    // convergence checking). Bundled as one const reference rather than
    // split per-component to match the "no wiring yet" scope of Task 1 — no
    // component reads through it until Task 2+.
    const PSIOPT::Settings &settings;

    // --- Problem dimensions (dossier §6: "shared (immutable during solve)") ---
    // References (not copies) into PSIOPT's own dimension members: they are
    // fixed for the lifetime of a solve, but SolverContext never takes a
    // snapshot — it always observes the live value.
    const int &primal_vars;
    const int &slack_vars;
    const int &equal_cons;
    const int &inequal_cons;
    const int &kkt_dim;

    // --- Reusable scratch buffers (dossier §6, "prospective owner" column) ---
    // stli_scratch_ / hp_scratch_: BarrierGovernor (complementarity() and
    // barrier_hessian()'s internal buffers, both read+write by the function
    // that owns them today; PSIOPT keeps the backing storage to avoid
    // per-call heap allocation, see psiopt.h:404-411).
    Eigen::VectorXd &stli_scratch;
    Eigen::VectorXd &hp_scratch;

    // best_xsl_scratch_ / best_rhs_scratch_: RecoveryChain (the return_best_
    // snapshot/restore machinery, psiopt.h:413-419). Read+write once the
    // best-iterate blocks are extracted (not in Task 1).
    Eigen::VectorXd &best_xsl_scratch;
    Eigen::VectorXd &best_rhs_scratch;
};

} // namespace tycho::solvers
