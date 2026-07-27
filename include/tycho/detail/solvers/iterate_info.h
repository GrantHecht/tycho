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
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

namespace tycho::solvers {

struct IterateInfo {

    int iter_ = 0;

    double mu_ = 0;
    double prim_obj_ = 0;
    double barr_obj_ = 0;
    double kkt_inf_ = 0;
    double barr_inf_ = 0;
    double econ_inf_ = 0;
    double icon_inf_ = 0;

    double pen_par1_ = 0.0;
    double pen_par2_ = 0.0;

    int ls_iters_ = 0;
    double alpha_p_ = 1.0;
    double alpha_d_ = 1.0;
    double alpha_t_ = 1.0;

    double h_pert_ = 0;
    int h_facs_ = 0;

    // PSIOPT 2.4 (display-only carve-out): the running total of every inertia-
    // perturbation delta applied to the KKT diagonal during this iteration's
    // factor_impl() call (i.e. the actual cumulative perturbation, as opposed to
    // h_pert_ above, which is only the LAST delta). Split into its own field so the
    // HPert table column can show the cumulative total without touching h_pert_'s
    // existing meaning or its Hpert0 warm-start producer/consumer in alg_impl().
    double h_pert_cum_ = 0;

    // Proximal primal-dual regularization shifts applied this iteration (written
    // only when Settings::inertia_mode_ == proximal_regularization; sentinel -1
    // on the classic path, which applies neither). prox_reg_primal_ is the
    // persistent primal base shift ρ_k added to the Hessian diagonal;
    // prox_reg_dual_ is the barrier-scaled dual shift δ_c subtracted from the
    // constraint-row diagonals (0 when suppressed inside a nested l1 restoration
    // phase). Both are >= 0 when active; a negative value means "mode off". Not
    // printed (the iteration table formats an explicit field list), so adding
    // them leaves console output byte-identical.
    double prox_reg_primal_ = -1.0;
    double prox_reg_dual_ = -1.0;

    int p_pivots_ = 0;
    double max_e_mult_ = 0;
    double max_i_mult_ = 0;
    double merit_val_ = 0.0;

    // Line-search acceptance outcome. Write-only diagnostic signal recorded by
    // the merit line search at the point it decides accept vs. reject; not
    // consumed on the classic solve path except by the recovery-dispatch gate
    // in alg_impl (which reads accepted_). These fields are NOT printed — the
    // iteration table formats an explicit field list (see psiopt_print.cpp) —
    // so adding them leaves console output byte-identical.
    //   accepted_               — did the merit test accept a step this line
    //                             search (false if every backtrack was
    //                             rejected, i.e. the search exhausted).
    //   first_rejection_iter_   — backtracking index of the first rejected
    //                             trial (-1 if the step was accepted with no
    //                             rejection).
    //   theta_at_first_rejection_ — constraint infeasibility at that first
    //                             rejected trial. The NORM CONVENTION differs
    //                             by acceptance path: the classic merit
    //                             variants (ls_l1/ls_auglang) store the
    //                             SQUARED L2 norm of the full constraint
    //                             block (all_cons().squaredNorm()), while the
    //                             modern merit path (modern_merit.h) stores
    //                             the L1 norm (all_cons().lpNorm<1>()).
    //                             Consumers must NOT assume a specific norm
    //                             across strategies — compare this field only
    //                             against another reading taken under the
    //                             SAME acceptance path. The one live consumer
    //                             today, soc_should_trigger() (globalization/
    //                             soc.h), still compares like-with-like even
    //                             though SocRecovery composes with every
    //                             acceptance strategy (the recovery hook in
    //                             alg_impl dispatches to it regardless of
    //                             which strategy is active): its caller
    //                             (SocRecovery::on_step_rejected) selects the
    //                             comparison norm via
    //                             AcceptanceStrategy::drives_classic_path() —
    //                             squared-L2 when true (ClassicMeritAcceptance),
    //                             L1 when false (the generic path: modern
    //                             merit, filter, funnel) — so the two
    //                             readings it compares are always the same
    //                             norm, just not always squared-L2. -1.0
    //                             means UNAVAILABLE: no rejection was
    //                             recorded, or the LANG variant ran (it
    //                             materializes no infeasibility scalar). A
    //                             real infeasibility is always >= 0, so
    //                             consumers (e.g. a second-order correction
    //                             trigger) must treat any negative value as
    //                             "skip theta-based logic" rather than as a
    //                             feasible reading.
    bool accepted_ = false;
    int first_rejection_iter_ = -1;
    double theta_at_first_rejection_ = -1.0;

    // Number of trial-point evaluations that threw during this iteration's
    // acceptance attempts (line-search rungs, SOC/extended-backtrack trials,
    // soft-feasibility trial). 0 on the overwhelmingly common no-exception
    // path. Not printed in the iteration table. The `iters` vector alg_impl
    // accumulates is solve-local, but PSIOPT::LateCallBackType hands each
    // completed IterateInfo (iters.back()) to a registered C++ callback once
    // per iteration; no Python surface exposes it.
    int eval_exceptions_ = 0;
};

} // namespace tycho::solvers
