// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction (Task 3). Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3.
// Ground-truth recon: docs/superpowers/plans/2026-07-16-e2-g1-dossier.md §2
// ("Fraction-to-boundary & step coupling") and §8 ("Riskiest seam" callout).
//
// BacktrackingLineSearch implements GlobalizationMechanism::compute_step by
// hosting today's PSIOPT::max_primal_dual_step fraction-to-boundary scaling
// plus its max_step_to_boundary scalar helper, moved VERBATIM from
// src/solvers/psiopt.cpp (statement order and operand order preserved exactly
// — the merge gate is a bit-identical CBWR iteration-count comparison). The
// only edits are context-plumbing renames: former PSIOPT member reads
// (settings_.pd_step_strategy_, inequal_cons_, equal_cons_) now go through the
// SolverContext reference passed to the call. Definitions live in
// src/solvers/psiopt_globalization.cpp.
//
// Riskiest-seam design note (dossier §2/§8):
//   max_primal_dual_step SCALES DXSL's primal/slack/multiplier blocks IN PLACE
//   between the KKT solve+negate and the merit backtrack; the backtrack's trial
//   points (xsl + alpha*dxsl) and the eventual commit (XSL += alpha*DXSL) both
//   operate on that SAME, already-scaled DXSL. compute_step therefore FUSES
//   "scale DXSL by the fraction-to-boundary step (alphap/alphad)" and "run the
//   AcceptanceStrategy backtrack on the scaled DXSL" into one call — the
//   in-place block-scaling statements (dxsl.primals() *= primstep; …) are
//   preserved with the exact same statement order, operand order, and
//   pd_step_strategy_ switch as the original, so no component boundary reorders
//   those multiplications. DXSL absolutely IS mutated in place.
//
// Byte-identity design note (references-only channel):
//   Like ClassicMeritAcceptance (merit_acceptance.h), this component reaches
//   PSIOPT state ONLY through SolverContext, and reconstructs a KKTVector view
//   over the raw XSL/DXSL blocks internally (PSIOPT::KKTVector is private and
//   not name-accessible from this non-member, non-friend type). The nested
//   KKTVector below is a VERBATIM copy of PSIOPT::KKTVector; a later G-task that
//   consolidates the globalization helpers should share one copy.
//
// Ownership rule: BacktrackingLineSearch holds NO solver state (matches the
// GlobalizationMechanism ownership rule). Every quantity it needs is either an
// explicit per-call parameter (obj_scale/mu/prim_obj/barr_obj, the working
// vectors) or reached through the SolverContext reference passed to the call
// (settings_/dims) — never cached across calls. reset() is a no-op (classic
// backtracking carries no persistent state across iterations).

#pragma once

#include <cassert>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// =============================================================================
// BacktrackingLineSearch — the classic globalization mechanism: fraction-to-
// boundary primal/dual step-length scaling followed by an AcceptanceStrategy
// backtrack on the scaled search direction.
//
// Stateless (holds NO solver state, per GlobalizationMechanism's ownership
// rule). Constructed once by PSIOPT::set_nlp; every call receives the live
// SolverContext view of the solver as an explicit parameter.
// =============================================================================
class BacktrackingLineSearch : public GlobalizationMechanism {
  public:
    BacktrackingLineSearch() = default;

    // Fused fraction-to-boundary scaling + acceptance backtrack — see the
    // riskiest-seam note above and GlobalizationMechanism::compute_step.
    double compute_step(PSIOPT::LineSearchModes lsmode, double obj_scale, double mu,
                        double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                        Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
                        Eigen::VectorXd &RHS2, AcceptanceStrategy &acceptance, double &alphap,
                        double &alphad, IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                        SolverContext &ctx) override;

    // μ-event / phase-change hook — no-op: classic backtracking carries no
    // persistent state across iterations (see the ownership-rule note above).
    void reset() override {}

    // Fraction-to-boundary primal/dual step (verbatim today's
    // PSIOPT::max_primal_dual_step). Public so the PROBE barrier block's
    // predictor call site can drive it directly on the predictor DXSL (dossier
    // §3); builds the KKTVector view over the raw XSL/DXSL blocks internally
    // and MUTATES DXSL in place. bfrac / dims / pd_step_strategy_ are read
    // through `ctx`.
    void max_primal_dual_step(Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, double bfrac,
                              double &alphap, double &alphad, const SolverContext &ctx) override;

  private:
    // =========================================================================
    // KKTVector — VERBATIM copy of PSIOPT::KKTVector (psiopt.h). Reproduced
    // here (rather than reached through PSIOPT, which is private/non-friend) so
    // the moved step body keeps its exact named-segment accessors
    // (dxsl.primals()/dxsl.slacks()/…) unchanged. Non-owning view over the
    // compound KKT layout [primals | slacks | eq_lmults | iq_lmults]; must not
    // outlive the referenced VectorXd.
    // =========================================================================
    class KKTVector {
      public:
        KKTVector(Eigen::VectorXd &data, int pv, int sv, int ec, int ic)
            : data_(data), pv_(pv), sv_(sv), ec_(ec), ic_(ic) {
            assert(pv >= 0 && sv >= 0 && ec >= 0 && ic >= 0);
            assert(data.size() >= pv + sv + ec + ic);
        }

        auto primals() { return data_.head(pv_); }
        auto primals() const { return std::as_const(data_).head(pv_); }
        auto slacks() { return data_.segment(pv_, sv_); }
        auto slacks() const { return std::as_const(data_).segment(pv_, sv_); }
        auto primals_slacks() { return data_.head(pv_ + sv_); }
        auto primals_slacks() const { return std::as_const(data_).head(pv_ + sv_); }

        auto eq_lmults() { return data_.segment(pv_ + sv_, ec_); }
        auto eq_lmults() const { return std::as_const(data_).segment(pv_ + sv_, ec_); }
        auto iq_lmults() { return data_.tail(ic_); }
        auto iq_lmults() const { return std::as_const(data_).tail(ic_); }
        auto lmults() { return data_.tail(ec_ + ic_); }
        auto lmults() const { return std::as_const(data_).tail(ec_ + ic_); }

        auto prim_grad() { return data_.head(pv_); }
        auto prim_grad() const { return std::as_const(data_).head(pv_); }
        auto dual_grad() { return data_.segment(pv_, sv_); }
        auto dual_grad() const { return std::as_const(data_).segment(pv_, sv_); }
        auto prim_dual_grad() { return data_.head(pv_ + sv_); }
        auto prim_dual_grad() const { return std::as_const(data_).head(pv_ + sv_); }
        auto eq_cons() { return data_.segment(pv_ + sv_, ec_); }
        auto eq_cons() const { return std::as_const(data_).segment(pv_ + sv_, ec_); }
        auto iq_cons() { return data_.tail(ic_); }
        auto iq_cons() const { return std::as_const(data_).tail(ic_); }
        auto all_cons() { return data_.tail(ec_ + ic_); }
        auto all_cons() const { return std::as_const(data_).tail(ec_ + ic_); }

        Eigen::VectorXd &data() { return data_; }
        const Eigen::VectorXd &data() const { return data_; }

      private:
        Eigen::VectorXd &data_;
        int pv_, sv_, ec_, ic_;
    };

    /// Create a KKTVector view over a VectorXd using the context's dimensions.
    static KKTVector kkt_view(Eigen::VectorXd &v, const SolverContext &ctx) {
        return KKTVector(v, ctx.primal_vars_, ctx.slack_vars_, ctx.equal_cons_, ctx.inequal_cons_);
    }

    // Scalar fraction-to-boundary step for one slack/multiplier block (verbatim
    // today's PSIOPT::max_step_to_boundary); reads inequal_cons_ through `ctx`.
    double max_step_to_boundary(Eigen::Ref<Eigen::VectorXd> SLI, Eigen::Ref<Eigen::VectorXd> dSLI,
                                double bfrac, const SolverContext &ctx) const;
};

} // namespace tycho::solvers
