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
//   - Extracted the compound-KKT segment view from PSIOPT (psiopt.h), where it
//     was a private nested class that the globalization components could not
//     name and therefore each reproduced verbatim
// =============================================================================
//
// The solver's compound KKT vectors are plain Eigen::VectorXd of length
// primal_vars + slack_vars + equal_cons + inequal_cons. KKTVector is the
// non-owning view that gives those four blocks names. It is deliberately a
// standalone header with no PSIOPT dependency: PSIOPT and every globalization
// component (ClassicMeritAcceptance, BacktrackingLineSearch,
// ClassicAdaptiveGovernor, ...) builds its views from the SAME type, so the
// segment expressions that encode the layout exist once in the codebase.
//
// Each component supplies its own kkt_view() factory, since the dimensions come
// from different places (PSIOPT's own members; a SolverContext's dimension
// references).

#pragma once

#include <cassert>
#include <utility>

#include <Eigen/Core>

namespace tycho::solvers {

// =============================================================================
// KKTVector — lightweight non-owning view over compound KKT layout
//   [primals | slacks | eq_lmults | iq_lmults]
// Used for both the iterate vector (x, s, lambda_e, lambda_i) and the
// RHS/gradient vector (grad_x, grad_s, c_eq, c_iq). The two accessor
// groups provide semantic names for each interpretation.
//
// const-correctness: const overloads use std::as_const(data_) to force
// Eigen's .head()/.segment()/.tail() to return immutable segment
// expressions. Without this, calling .head() on the non-const VectorXd&
// member would return a mutable expression even from a const method.
// Lifetime: must not outlive the referenced VectorXd.
// =============================================================================
class KKTVector {
  public:
    KKTVector(Eigen::VectorXd &data, int pv, int sv, int ec, int ic)
        : data_(data), pv_(pv), sv_(sv), ec_(ec), ic_(ic) {
        assert(pv >= 0 && sv >= 0 && ec >= 0 && ic >= 0);
        assert(data.size() >= pv + sv + ec + ic);
    }

    // --- Primal/slack segments ---
    auto primals() { return data_.head(pv_); }
    auto primals() const { return std::as_const(data_).head(pv_); }
    auto slacks() { return data_.segment(pv_, sv_); }
    auto slacks() const { return std::as_const(data_).segment(pv_, sv_); }
    auto primals_slacks() { return data_.head(pv_ + sv_); }
    auto primals_slacks() const { return std::as_const(data_).head(pv_ + sv_); }

    // --- Multiplier segments ---
    auto eq_lmults() { return data_.segment(pv_ + sv_, ec_); }
    auto eq_lmults() const { return std::as_const(data_).segment(pv_ + sv_, ec_); }
    auto iq_lmults() { return data_.tail(ic_); }
    auto iq_lmults() const { return std::as_const(data_).tail(ic_); }
    auto lmults() { return data_.tail(ec_ + ic_); }
    auto lmults() const { return std::as_const(data_).tail(ec_ + ic_); }

    // --- Gradient/constraint segments (intentional aliases) ---
    // Same memory layout as the primal/multiplier accessors above, but with
    // names matching the RHS/gradient interpretation: the primal block holds
    // the objective gradient, the slack block holds the dual gradient, and
    // the multiplier blocks hold constraint values.
    // These are intentional aliases: prim_grad() == primals(),
    // dual_grad() == slacks(), eq_cons() == eq_lmults(), iq_cons() == iq_lmults().
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

    // --- Full vector access ---
    Eigen::VectorXd &data() { return data_; }
    const Eigen::VectorXd &data() const { return data_; }

  private:
    Eigen::VectorXd &data_;
    int pv_, sv_, ec_, ic_;
};

} // namespace tycho::solvers
