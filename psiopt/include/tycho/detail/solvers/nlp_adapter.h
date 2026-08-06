// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <fmt/format.h>

#include "tycho/detail/solvers/constraint_function.h"
#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/solvers/objective_function.h"
#include "tycho/detail/solvers/threading_flags.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/solvers/nlp_problem.h"

namespace tycho::solvers {

/// How one user constraint row is realized in the solver's row spaces.
enum class NLPRowKind { Equality, UpperBounded, LowerBounded, Range, Free };

/// Setup-time classification of the user's constraint rows against their
/// bounds: which solver rows each user row becomes, in declaration order.
/// A Range row (two finite, unequal bounds) becomes two inequality rows,
/// the upper part first, then the negated lower part.
struct NLPRowClassification {
    std::vector<NLPRowKind> kinds_; ///< one entry per user row
    Eigen::VectorXi eq_row_;        ///< user row -> solver equality row, -1 if none
    Eigen::VectorXi iq_upper_row_;  ///< user row -> solver inequality row (upper part), -1
    Eigen::VectorXi iq_lower_row_;  ///< user row -> solver inequality row (negated lower), -1
    int num_eq_ = 0;
    int num_iq_ = 0;

    static NLPRowClassification classify(ConstEigenRef<Eigen::VectorXd> gl,
                                         ConstEigenRef<Eigen::VectorXd> gu);
};

/// One Jacobian value scatter: user triplet slot -> this piece's local
/// constraint row, with the residual's sign.
struct NLPPieceJacEntry {
    int user_slot_;
    int local_row_;
    double sign_;
};

/// One residual: local row k of a piece evaluates
/// sign_ * (g_user[user_row_] - shift_).
struct NLPPieceResEntry {
    int user_row_;
    double sign_;
    double shift_;
};

/// Shared state behind the adapter pieces: the user problem, its validated
/// structures, the row classification, per-piece scatter tables, the
/// per-iterate evaluation cache, and the per-assembly multiplier records.
/// Shared by every piece copy via shared_ptr; safe without locking because the
/// adapter always runs single-partition, so every access is serial.
struct NLPAdapterCore {
    std::shared_ptr<NLPProblem> problem_;
    int n_ = 0, m_ = 0, jac_nnz_ = 0, hess_nnz_ = 0;

    Eigen::VectorXd x_lower_, x_upper_; ///< variable bounds, as declared
    NLPRowClassification rows_;
    Eigen::VectorXi jac_rows_, jac_cols_;   ///< user Jacobian structure
    Eigen::VectorXi hess_rows_, hess_cols_; ///< user Hessian structure (lower triangle)

    std::vector<NLPPieceJacEntry> eq_jac_, iq_jac_;
    std::vector<NLPPieceResEntry> eq_res_, iq_res_;

    /// Which piece scatters the (monolithic) Lagrangian Hessian: the last one
    /// the evaluation order reaches, decided once here.
    enum class HessOwner { Objective, EqPiece, IqPiece };
    HessOwner hess_owner_ = HessOwner::Objective;

    // --- Per-iterate cache (callbacks are pure; keyed on the iterate value) ---
    Eigen::VectorXd x_cache_;
    bool g_valid_ = false;
    bool jac_valid_ = false;
    Eigen::VectorXd g_cache_, jac_cache_;

    // --- Per-assembly consume-once records. The objective piece records the
    // objective scale when its Hessian-bearing method runs; the equality piece
    // records its multipliers likewise; the Hessian owner consumes both. A
    // chain that skips the objective (the solver's no-objective KKT mode)
    // leaves no record, and the owner correctly uses scale 0. ---
    std::optional<double> pending_obj_scale_;
    std::optional<Eigen::VectorXd> pending_le_;

    // Scratch (sized once at setup; reused every call)
    Eigen::VectorXd grad_scratch_, lambda_user_, hess_vals_;

    explicit NLPAdapterCore(std::shared_ptr<NLPProblem> problem);

    void refresh_g(ConstEigenRef<Eigen::VectorXd> x);
    void refresh_jac(ConstEigenRef<Eigen::VectorXd> x);

    /// Fills lambda_user_ from solver-space multiplier vectors (either may be
    /// empty, meaning zero).
    void compose_user_lambda(ConstEigenRef<Eigen::VectorXd> le, ConstEigenRef<Eigen::VectorXd> li);

    /// compose_user_lambda + the single eval_hess call into hess_vals_.
    void eval_hessian_values(ConstEigenRef<Eigen::VectorXd> x, double obj_factor,
                             ConstEigenRef<Eigen::VectorXd> le, ConstEigenRef<Eigen::VectorXd> li);

  private:
    void sync_x(ConstEigenRef<Eigen::VectorXd> x);
};

} // namespace tycho::solvers
