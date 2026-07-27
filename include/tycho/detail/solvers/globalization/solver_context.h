// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this header declares
// SolverContext, a references-only aggregate that bundles the persistent
// PSIOPT state the globalization components need to read (and, for a few
// scratch buffers, write).
//
// This file: declares SolverContext. The member list here is exactly the
// state inventory rows whose prospective owner is one of the globalization
// components (Acceptance / Globalization / BarrierGovernor / RecoveryChain /
// RestorationStrategy) or "shared". Members not needed by any of the five
// component interfaces or their sixteen shipped concrete strategies (e.g.
// result_.*, qp_analyzed_, callbacks) are intentionally omitted; this list is
// expected to grow as later work wires real call sites — do not treat this
// as the final member set.
//
// Ownership rule: SolverContext owns nothing. Every member is a reference or
// a non-owning pointer into the live PSIOPT instance; a SolverContext must
// not outlive the PSIOPT it was built from (same lifetime discipline as
// PSIOPT::KKTVector, psiopt.h). Most call sites construct one fresh (as a
// temporary) for the duration of a single call. One exception:
// ClassicMeritAcceptance holds a SolverContext by value as a private member
// (merit_acceptance.h) instead of re-threading it through every call — that
// copy is rebuilt by rebuild_globalization_components() on every solve entry
// and must not outlive that rebuild, the same lifetime bound as any other
// SolverContext. Components hold no OTHER solver state themselves;
// SolverContext (plus the explicit per-iteration transient parameters
// threaded through each interface method, e.g. mu/prim_obj/barr_obj) is the
// only channel by which they observe or mutate PSIOPT's persistent state.

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
// circular-include trick.
#include "tycho/detail/solvers/psiopt.h"

#ifdef USE_ACCELERATE_SPARSE
#include "tycho/detail/solvers/linear/accelerate_interface.h"
#else
#include "tycho/detail/solvers/linear/pardiso_interface.h"
#endif

namespace tycho::solvers {

// Forward declaration only: SolverContext carries a non-owning pointer to the
// active RestorationStrategy (complete type in globalization/restoration.h,
// which includes THIS header — so a plain include here would be circular). The
// pointer is null whenever feasibility restoration is off (the default), which
// is exactly when every restoration branch that consults it is provably dead.
class RestorationStrategy;

// The concrete sparse KKT factorization type, mirroring PSIOPT::kkt_sol_'s
// declaration (psiopt.h) exactly — same macro guard, same template
// arguments. Components never choose or construct this type; they only ever
// see it through SolverContext::kkt_solver_, driving solves/refactors that
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
// Every member below is documented with which state-inventory row it
// corresponds to and which component(s) read (or read+write) it today; this
// list is expected to grow as later globalization work lands.
// =============================================================================
struct SolverContext {
    // --- NLP / KKT solve machinery ---
    // `nlp_` — "eval_*, barrier_hessian, perturb, diags" — read by
    // Acceptance (eval_trial_point_occ/eval_rhs) and BarrierGovernor
    // (barrier_hessian) call sites once wired. Raw, non-owning pointer:
    // PSIOPT retains the owning std::shared_ptr<NonLinearProgram>.
    NonLinearProgram *nlp_;

    // `kkt_sol_` — "factor_impl, all solves, ppivs/neigs/peigs" —
    // read (solve) by BarrierGovernor's PROBE predictor solve and by the
    // (not-yet-extracted) main step solve; written (factor/refactor) by
    // RecoveryChain's future inertia-ladder dispatch. Reference: PSIOPT
    // retains ownership of the actual Eigen solver object.
    KktSolverType &kkt_solver_;

    // `settings_.*` rows — every globalization component reads a
    // disjoint subset (max_ls_iters_/alpha_red_ -> Acceptance;
    // bound_fraction_/pd_step_strategy_ -> Globalization;
    // opt_bar_mode_/soe_bar_mode_/init_mu_/min_mu_/max_mu_ -> BarrierGovernor;
    // delta_h_/incr_h_/decr_h_/max_refac_ -> RecoveryChain;
    // econ_tol_/icon_tol_ and the div_*/acc_*/kkt_tol_ family -> shared with
    // convergence checking). Bundled as one const reference rather than
    // split per-component: every live call site reads it as
    // ctx_.settings_./ctx.settings_. (e.g. ClassicMeritAcceptance's exit-test
    // helpers, BacktrackingLineSearch::compute_step,
    // ClassicAdaptiveGovernor::update_barrier), so one reference covers every
    // component's disjoint subset without per-field plumbing.
    const PSIOPT::Settings &settings_;

    // --- Problem dimensions (shared, immutable during solve) ---
    // References (not copies) into PSIOPT's own dimension members: they are
    // fixed for the lifetime of a solve, but SolverContext never takes a
    // snapshot — it always observes the live value.
    const int &primal_vars_;
    const int &slack_vars_;
    const int &equal_cons_;
    const int &inequal_cons_;
    const int &kkt_dim_;

    // --- Reusable scratch buffers ---
    // stli_scratch_: read+write by ClassicAdaptiveGovernor::complementarity(),
    // which is a moved copy of PSIOPT::complementarity and uses the SAME
    // PSIOPT-owned buffer to avoid per-call heap allocation (see
    // PSIOPT::stli_scratch_).
    //
    // This is the only scratch buffer any component reaches through the context.
    // hp_scratch_ (barrier_hessian's buffer) and best_xsl_scratch_/
    // best_rhs_scratch_ (the return_best_ snapshots) used to be here too, on the
    // expectation that a future BarrierGovernor/RecoveryChain would need them;
    // no component ever read one, and both remained PSIOPT-internal (the
    // best-iterate bookkeeping is PSIOPT::track_best_iterate). They were dropped
    // rather than carried forward as three unused references passed at every
    // construction site.
    Eigen::VectorXd &stli_scratch_;

    // --- Feasibility restoration (optional; null when off) ---
    // Non-owning pointer to the active RestorationStrategy, or nullptr when
    // restoration is off (the default). Consulted by the classic and generic
    // trial-point evaluators (ClassicMeritAcceptance::ls_* / modern_eval_trial_
    // point) to add the proximal objective φ_prox to a trial's objective value
    // while restoration is active, and by FeasibilitySwitchRecovery to test
    // entry permission. A default member initializer of nullptr keeps this an
    // aggregate whose existing braced-init call sites (which omit this trailing
    // member) still compile and default it to nullptr — so those sites, and the
    // whole default solve path, remain restoration-free and bit-identical.
    const RestorationStrategy *restoration_ = nullptr;

    // --- Trial-evaluation exception log (optional; null in isolation) ---
    // Non-owning pointer to the live PSIOPT's EvalErrorLog. The wrapped
    // trial-evaluation sites record through it; a defaulted nullptr keeps
    // existing braced-init call sites (unit tests constructing a bare
    // SolverContext) compiling, and every recording site null-guards.
    EvalErrorLog *eval_errors_ = nullptr;
};

} // namespace tycho::solvers
