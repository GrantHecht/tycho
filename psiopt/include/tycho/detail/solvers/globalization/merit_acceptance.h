// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this is the line search &
// merit acceptance component.
//
// ClassicMeritAcceptance implements AcceptanceStrategy::classic_line_search by
// hosting today's PSIOPT::ls_impl dispatcher plus the ls_lang / ls_l1 /
// ls_auglang merit variants and their eval_trial_point_occ / compute_penalties
// / secondary_accept helpers, moved VERBATIM from src/solvers/psiopt.cpp (the
// merge gate is a bit-identical CBWR iteration-count comparison, so operand
// order and statement order are preserved exactly; the only edits are
// context-plumbing renames — member reads such as settings_/equal_cons_/nlp_
// now come through the SolverContext reference `ctx_`). Definitions live in
// src/solvers/psiopt_globalization.cpp.
//
// Byte-identity design note (references-only channel):
//   The moved merit bodies call four tiny PSIOPT barrier/eval helpers
//   (eval_rhs, apply_reset_slacks, barrier_objective, barrier_gradient) and
//   use the compound-KKT segment view + a PenaltyTerms value type. Per this
//   component architecture (acceptance_strategy.h and solver_context.h) a
//   non-member, non-friend AcceptanceStrategy may reach PSIOPT state ONLY
//   through SolverContext (nlp_/settings_/dims/scratch references), so each
//   helper survives here as a private method reading through `ctx_`.
//   apply_reset_slacks / barrier_objective / barrier_gradient are now one-line
//   forwarders into detail/solvers/barrier_math.h, the single home for those
//   three kernels; eval_rhs remains a local copy (it slices the KKT layout for
//   an NLP call rather than doing barrier arithmetic). The segment view is
//   tycho::solvers::KKTVector (detail/solvers/kkt_vector.h), shared with PSIOPT
//   and the sibling components — this header's former verbatim copy of it, and
//   the identical copies in BacktrackingLineSearch and ClassicAdaptiveGovernor,
//   were consolidated there. PenaltyTerms stays a private nested type (it has
//   one user). This is the "reconstructs KKTVector-equivalent segment views
//   internally from SolverContext's dims" path acceptance_strategy.h
//   anticipates.

#pragma once

#include <cassert>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"
#include "tycho/detail/solvers/kkt_vector.h"
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// =============================================================================
// ClassicMeritAcceptance — the classic backtracking merit line search.
//
// Holds a SolverContext BY VALUE (references-only aggregate; cheap to copy,
// cannot dangle because PSIOPT owns the acceptance_ unique_ptr and therefore
// outlives it, and every SolverContext member refers to a stable PSIOPT
// member). Rebuilt by PSIOPT::rebuild_globalization_components() at the start
// of every solve invocation, so the captured nlp_ raw pointer never goes
// stale (dims are captured by reference and track the live members
// regardless; nlp_ and the dims are themselves only ever written by
// set_nlp(), which always precedes any solve).
// =============================================================================
class ClassicMeritAcceptance : public AcceptanceStrategy {
  public:
    explicit ClassicMeritAcceptance(const SolverContext &ctx) : ctx_(ctx) {}

    // The classic strategy is the one (and only) fused-path driver.
    bool drives_classic_path() const override { return true; }

    // --- Generic interface ---
    // is_iterate_acceptable: the classic acceptance test is fused inside
    // classic_line_search's backtracking loop, so the generic (θ, f) hook is
    // never driven on the classic path. Reaching it is a wiring bug, so it
    // throws (T6: never a silent wrong return) rather than fabricate an answer.
    bool is_iterate_acceptable(const ProgressMeasures &current, const ProgressMeasures &trial,
                               const ProgressMeasures &predicted_reduction,
                               double objective_multiplier, double step_length) override;
    // is_infeasibility_sufficiently_reduced: the restoration-exit test, driven
    // by alg_impl while the classic strategy runs in feasibility mode. There
    // is no Uno counterpart — Uno pairs restoration with its own
    // filter/funnel strategies, not a monolithic merit line search — so the
    // Ipopt IpRestoConvCheck relative-reduction shape is the reference (see the
    // definition in psiopt_globalization.cpp for the term-for-term mapping and
    // the single-tolerance floor adaptation).
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &reference,
                                               const ProgressMeasures &trial) const override;

    // μ-event / phase-change hook — no-op: the classic merit test carries no
    // persistent state across iterations (see acceptance_strategy.h).
    void reset() override {}

    // --- Classic fused entry point (verbatim today's PSIOPT::ls_impl) ---
    double classic_line_search(PSIOPT::LineSearchModes lsmode, double obj_scale, double mu,
                               double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                               Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
                               Eigen::VectorXd &RHS2, IterateInfo &Citer,
                               const std::vector<IterateInfo> &iters) override;

  private:
    SolverContext ctx_;

    // The compound-KKT segment view (tycho::solvers::KKTVector,
    // detail/solvers/kkt_vector.h) is shared with PSIOPT and the sibling
    // components, so the moved merit bodies keep their exact
    // `xsl.primals()`/`rhs.all_cons()` named-segment accessors unchanged. Only
    // the factory below is per-component: the dimensions come from ctx_.

    /// Create a KKTVector view over a VectorXd using the context's dimensions.
    KKTVector kkt_view(Eigen::VectorXd &v) {
        return KKTVector(v, ctx_.primal_vars_, ctx_.slack_vars_, ctx_.equal_cons_,
                         ctx_.inequal_cons_);
    }

    // Moved verbatim from PSIOPT (psiopt.h), which no longer has one: the merit
    // penalty triple is used only by the moved merit bodies below.
    struct PenaltyTerms {
        double l1_, l2_, linf_;
    };

    // --- Merit variants + shared helpers (moved verbatim from psiopt.cpp) ---
    double ls_lang(double obj_scale, double mu, double prim_obj, double barr_obj, KKTVector &xsl,
                   KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                   IterateInfo &citer);
    double ls_l1(double obj_scale, double mu, double prim_obj, double barr_obj, KKTVector &xsl,
                 KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                 IterateInfo &citer);
    double ls_auglang(double obj_scale, double mu, double prim_obj, double barr_obj, KKTVector &xsl,
                      KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                      IterateInfo &citer);

    void eval_trial_point_occ(double obj_scale, double mu, double alpha, KKTVector &xsl,
                              KKTVector &dxsl, KKTVector &xsl2, KKTVector &rhs2, double &ptest,
                              double &btest);
    PenaltyTerms compute_penalties(KKTVector &xsl, KKTVector &rhs) const;
    bool secondary_accept(double ptest, double prim_obj, const PenaltyTerms &test,
                          const PenaltyTerms &init) const;

    // --- Barrier/eval helpers ---
    //     apply_reset_slacks/barrier_objective/barrier_gradient forward to the
    //     shared inline kernels in barrier_math.h, as do PSIOPT's own members
    //     of the same name. eval_rhs stays a real body here (it forwards to
    //     ctx_.nlp_->eval_rhs with the segment plumbing this component needs);
    //     it has no shared-header counterpart.
    void eval_rhs(double obj_scale, const Eigen::Ref<const Eigen::VectorXd> &XSL, double &val,
                  Eigen::Ref<Eigen::VectorXd> GX, Eigen::Ref<Eigen::VectorXd> AGXS_FX);
    void apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> FXI) const;
    double barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu) const;
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu,
                          Eigen::Ref<Eigen::VectorXd> AGS) const;

    // Nested-restoration trial-path scratch (dead unless a nested restoration
    // strategy is active). The trial-point evaluators add the elastic residual
    // shift (n+αΔn)−(p+αΔp) to a trial's raw constraint residuals so the merit
    // sees the restoration subproblem's infeasibility; these back that shift
    // without per-backtrack heap allocation. mutable so the const-facing trial
    // helpers may fill them.
    mutable Eigen::VectorXd resto_eq_shift_scratch_;
    mutable Eigen::VectorXd resto_iq_shift_scratch_;
};

} // namespace tycho::solvers
