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
};

} // namespace tycho::solvers
