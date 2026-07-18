// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

namespace tycho::solvers {

struct IterateInfo {

    int iter = 0;

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

    double kkt_norm_err_ = 0;
    double barr_norm_err_ = 0;
    double econ_norm_err_ = 0;
    double icon_norm_err_ = 0;
    double all_con_norm_err_ = 0;

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
    //                             soc.h), only ever compares like-with-like:
    //                             SocRecovery is reachable only through
    //                             ClassicMeritAcceptance::classic_line_search
    //                             (ModernMeritAcceptance's override throws),
    //                             so its trigger always compares two
    //                             squared-L2 readings, never a mix. -1.0
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
};

} // namespace tycho::solvers
