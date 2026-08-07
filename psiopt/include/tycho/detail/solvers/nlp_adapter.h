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

#include "tycho/detail/solvers/constraint_function.h"
#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/solvers/objective_function.h"
#include "tycho/detail/solvers/threading_flags.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/solvers/nlp_problem.h"

namespace tycho::solvers {

struct NonLinearProgram;

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

/// Claims the Lagrangian-Hessian block: one KKT slot per user Hessian triplet,
/// at the (reduced) variable coordinates. The reduced renumbering is monotone,
/// so a lower-triangle structure stays lower-triangle.
inline void nlp_claim_hessian_block(const NLPAdapterCore &core, EigenRef<Eigen::VectorXi> KKTrows,
                                    EigenRef<Eigen::VectorXi> KKTcols, int &freeloc,
                                    const SolverIndexingData &data) {
    for (int k = 0; k < core.hess_nnz_; k++) {
        const int r = data.v_scatter_loc(core.hess_rows_[k], 0);
        const int c = data.v_scatter_loc(core.hess_cols_[k], 0);
        if (r < 0 || c < 0) {
            KKTrows[freeloc] = -1;
            KKTcols[freeloc] = -1;
        } else {
            KKTrows[freeloc] = r;
            KKTcols[freeloc] = c;
        }
        freeloc++;
    }
}

/// Sums the freshly evaluated Hessian values into the KKT matrix through the
/// claims recorded at @p claim_start. Lock keying follows the shared protocol
/// (kkt_canonical_lock_col); with the adapter's single partition no slot is
/// ever contested, but the discipline is kept so the code stays correct if
/// that ever changes.
inline void nlp_scatter_hessian_block(const NLPAdapterCore &core, int claim_start,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                      EigenRef<Eigen::VectorXi> KKTLocs,
                                      EigenRef<Eigen::VectorXi> VarClashes,
                                      std::vector<std::mutex> &ClashLocks,
                                      const SolverIndexingData &data) {
    if (claim_start < 0) {
        throw std::logic_error("NLP adapter: Hessian scatter before its KKT space was claimed");
    }
    int cursor = claim_start;
    for (int k = 0; k < core.hess_nnz_; k++, cursor++) {
        const int r = data.v_scatter_loc(core.hess_rows_[k], 0);
        const int c = data.v_scatter_loc(core.hess_cols_[k], 0);
        if (r < 0 || c < 0) {
            continue;
        }
        const int lockcol = kkt_canonical_lock_col(r, c);
        const bool lock_column = (VarClashes[lockcol] != -1);
        if (lock_column) {
            ClashLocks[VarClashes[lockcol]].lock();
        }
        KKTmat.valuePtr()[KKTLocs.data()[cursor]] += core.hess_vals_[k];
        if (lock_column) {
            ClashLocks[VarClashes[lockcol]].unlock();
        }
    }
}

/// The user problem's objective, as a solver objective. Also the Hessian owner
/// when the problem has no constraints.
struct NLPObjectivePiece {
    std::shared_ptr<NLPAdapterCore> core_;
    int hess_claim_start_ = -1; ///< recorded per partition copy during get_kkt_space

    NLPObjectivePiece() = default;
    explicit NLPObjectivePiece(std::shared_ptr<NLPAdapterCore> core) : core_(std::move(core)) {}

    std::string name() const { return core_->problem_->name() + " (objective)"; }
    int input_rows() const { return core_->n_; }
    int output_rows() const { return 1; }
    bool thread_safe() const { return false; }
    bool owns_hessian() const { return core_->hess_owner_ == NLPAdapterCore::HessOwner::Objective; }

    void objective(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                   const SolverIndexingData &data) const {
        (void)data;
        double f = 0.0;
        core_->problem_->eval_f(X, f);
        Val += ObjScale * f;
    }

    void objective_gradient(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                            EigenRef<Eigen::VectorXd> GX, const SolverIndexingData &data) const {
        this->objective(ObjScale, X, Val, data);
        core_->problem_->eval_grad_f(X, core_->grad_scratch_);
        GX.segment(data.inner_gradient_starts_[0], core_->n_) = ObjScale * core_->grad_scratch_;
    }

    void objective_gradient_hessian(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                                    EigenRef<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    EigenRef<Eigen::VectorXi> KKTLocations,
                                    EigenRef<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const SolverIndexingData &data) const {
        // A user callback further down this same assembly (an eq/iq piece's
        // eval_g/eval_jac/eval_hess) can throw after this method has already
        // recorded pending_obj_scale_. An aborted assembly must not leave that
        // record behind for a later, unrelated chain (PSIOPT's restoration
        // entry runs eval_kkt_no next; a caller can also retry solve() after a
        // propagated exception) to read as if it were this chain's own -- so
        // any exception escaping this method clears both consume-once records
        // before propagating. A completed chain is unaffected: the records are
        // still set/consumed exactly as before.
        try {
            this->objective_gradient(ObjScale, X, Val, GX, data);
            if (this->owns_hessian()) {
                core_->eval_hessian_values(X, ObjScale, Eigen::VectorXd(), Eigen::VectorXd());
                nlp_scatter_hessian_block(*core_, hess_claim_start_, KKTmat, KKTLocations,
                                          KKTClashes, KKTLocks, data);
            } else {
                core_->pending_obj_scale_ = ObjScale;
            }
        } catch (...) {
            core_->pending_obj_scale_.reset();
            core_->pending_le_.reset();
            throw;
        }
    }

    // A constraint interface is part of the objective concept because the KKT
    // sizing pass is shared; only the two structure methods are ever invoked on
    // an objective. The evaluation methods below are unreachable and say so.
    void constraints(ConstEigenRef<Eigen::VectorXd>, EigenRef<Eigen::VectorXd>,
                     const SolverIndexingData &) const {
        throw std::logic_error("NLPObjectivePiece: constraint evaluation on an objective");
    }
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd>, ConstEigenRef<Eigen::VectorXd>,
                                     EigenRef<Eigen::VectorXd>, EigenRef<Eigen::VectorXd>,
                                     const SolverIndexingData &) const {
        throw std::logic_error("NLPObjectivePiece: constraint evaluation on an objective");
    }
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd>, EigenRef<Eigen::VectorXd>,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &,
                              EigenRef<Eigen::VectorXi>, EigenRef<Eigen::VectorXi>,
                              std::vector<std::mutex> &, const SolverIndexingData &) const {
        throw std::logic_error("NLPObjectivePiece: constraint evaluation on an objective");
    }
    void constraints_jacobian_adjointgradient(ConstEigenRef<Eigen::VectorXd>,
                                              ConstEigenRef<Eigen::VectorXd>,
                                              EigenRef<Eigen::VectorXd>, EigenRef<Eigen::VectorXd>,
                                              Eigen::SparseMatrix<double, Eigen::RowMajor> &,
                                              EigenRef<Eigen::VectorXi>, EigenRef<Eigen::VectorXi>,
                                              std::vector<std::mutex> &,
                                              const SolverIndexingData &) const {
        throw std::logic_error("NLPObjectivePiece: constraint evaluation on an objective");
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd>, ConstEigenRef<Eigen::VectorXd>, EigenRef<Eigen::VectorXd>,
        EigenRef<Eigen::VectorXd>, Eigen::SparseMatrix<double, Eigen::RowMajor> &,
        EigenRef<Eigen::VectorXi>, EigenRef<Eigen::VectorXi>, std::vector<std::mutex> &,
        const SolverIndexingData &) const {
        throw std::logic_error("NLPObjectivePiece: constraint evaluation on an objective");
    }

    void get_kkt_space(EigenRef<Eigen::VectorXi> KKTrows, EigenRef<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       SolverIndexingData &data) {
        (void)conoffset;
        (void)dojac;
        data.inner_kkt_starts_.resize(1);
        data.inner_kkt_starts_[0] = freeloc;
        if (dohess && this->owns_hessian()) {
            hess_claim_start_ = freeloc;
            nlp_claim_hessian_block(*core_, KKTrows, KKTcols, freeloc, data);
        }
    }
    int num_kkt_elements(bool dojac, bool dohess) const {
        (void)dojac;
        return (dohess && this->owns_hessian()) ? core_->hess_nnz_ : 0;
    }
};

/// One block of the user problem's constraint rows — the equalities or the
/// inequalities — as a solver constraint. The last-evaluated piece also owns
/// the Lagrangian Hessian.
struct NLPConstraintPiece {
    std::shared_ptr<NLPAdapterCore> core_;
    bool is_inequality_ = false;
    int hess_claim_start_ = -1;

    NLPConstraintPiece() = default;
    NLPConstraintPiece(std::shared_ptr<NLPAdapterCore> core, bool is_inequality)
        : core_(std::move(core)), is_inequality_(is_inequality) {}

    std::string name() const {
        return core_->problem_->name() + (is_inequality_ ? " (inequalities)" : " (equalities)");
    }
    int input_rows() const { return core_->n_; }
    int output_rows() const { return is_inequality_ ? core_->rows_.num_iq_ : core_->rows_.num_eq_; }
    bool thread_safe() const { return false; }
    bool owns_hessian() const {
        return is_inequality_ ? core_->hess_owner_ == NLPAdapterCore::HessOwner::IqPiece
                              : core_->hess_owner_ == NLPAdapterCore::HessOwner::EqPiece;
    }
    const std::vector<NLPPieceJacEntry> &jac_table() const {
        return is_inequality_ ? core_->iq_jac_ : core_->eq_jac_;
    }
    const std::vector<NLPPieceResEntry> &res_table() const {
        return is_inequality_ ? core_->iq_res_ : core_->eq_res_;
    }

    void constraints(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {
        core_->refresh_g(X);
        this->write_residuals(FX, data);
    }
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {
        core_->refresh_g(X);
        core_->refresh_jac(X);
        this->write_residuals(FX, data);
        this->write_adjoint_gradient(L, AGX, data);
    }
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              EigenRef<Eigen::VectorXi> KKTLocations,
                              EigenRef<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const {
        core_->refresh_g(X);
        core_->refresh_jac(X);
        this->write_residuals(FX, data);
        this->scatter_jacobian(KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        core_->refresh_g(X);
        core_->refresh_jac(X);
        this->write_residuals(FX, data);
        this->write_adjoint_gradient(L, AGX, data);
        this->scatter_jacobian(KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        // Same exception-safety concern as NLPObjectivePiece::objective_gradient_hessian
        // above: this piece's own eval_g/eval_jac/eval_hess can throw either
        // before or after it has recorded pending_le_, and the owner (this
        // piece or the one after it) may never run to consume a record either
        // piece left behind. Clear both consume-once records on the way out
        // through any exception so an aborted assembly can never leak a stale
        // objective scale or equality multiplier into a later, unrelated
        // chain.
        try {
            this->constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations,
                                                       KKTClashes, KKTLocks, data);
            if (!is_inequality_ && !this->owns_hessian()) {
                // The inequality piece runs after this one in every Hessian-bearing
                // chain; leave it these multipliers.
                core_->pending_le_ = Eigen::VectorXd(L);
            }
            if (this->owns_hessian()) {
                const double obj_factor = core_->pending_obj_scale_.value_or(0.0);
                core_->pending_obj_scale_.reset();
                if (is_inequality_) {
                    const Eigen::VectorXd le =
                        core_->pending_le_ ? *core_->pending_le_ : Eigen::VectorXd();
                    core_->pending_le_.reset();
                    core_->eval_hessian_values(X, obj_factor, le, L);
                } else {
                    core_->eval_hessian_values(X, obj_factor, L, Eigen::VectorXd());
                }
                nlp_scatter_hessian_block(*core_, hess_claim_start_, KKTmat, KKTLocations,
                                          KKTClashes, KKTLocks, data);
            }
        } catch (...) {
            core_->pending_obj_scale_.reset();
            core_->pending_le_.reset();
            throw;
        }
    }

    void get_kkt_space(EigenRef<Eigen::VectorXi> KKTrows, EigenRef<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       SolverIndexingData &data) {
        data.inner_kkt_starts_.resize(1);
        data.inner_kkt_starts_[0] = freeloc;
        if (dojac) {
            for (const auto &e : this->jac_table()) {
                const int col = data.v_scatter_loc(core_->jac_cols_[e.user_slot_], 0);
                KKTrows[freeloc] = (col < 0) ? -1 : data.c_loc(e.local_row_, 0) + conoffset;
                KKTcols[freeloc] = col;
                freeloc++;
            }
        }
        if (dohess && this->owns_hessian()) {
            hess_claim_start_ = freeloc;
            nlp_claim_hessian_block(*core_, KKTrows, KKTcols, freeloc, data);
        }
    }
    int num_kkt_elements(bool dojac, bool dohess) const {
        return (dojac ? static_cast<int>(this->jac_table().size()) : 0) +
               ((dohess && this->owns_hessian()) ? core_->hess_nnz_ : 0);
    }

  private:
    void write_residuals(EigenRef<Eigen::VectorXd> FX, const SolverIndexingData &data) const {
        const auto &tab = this->res_table();
        const int start = data.inner_constraint_starts_[0];
        for (int k = 0; k < static_cast<int>(tab.size()); k++) {
            FX[start + k] = tab[k].sign_ * (core_->g_cache_[tab[k].user_row_] - tab[k].shift_);
        }
    }
    void write_adjoint_gradient(ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> AGX,
                                const SolverIndexingData &data) const {
        const int gstart = data.inner_gradient_starts_[0];
        for (const auto &e : this->jac_table()) {
            AGX[gstart + core_->jac_cols_[e.user_slot_]] +=
                e.sign_ * core_->jac_cache_[e.user_slot_] * L[data.c_loc(e.local_row_, 0)];
        }
    }
    void scatter_jacobian(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                          EigenRef<Eigen::VectorXi> KKTLocs, EigenRef<Eigen::VectorXi> VarClashes,
                          std::vector<std::mutex> &ClashLocks,
                          const SolverIndexingData &data) const {
        int cursor = data.inner_kkt_starts_[0];
        for (const auto &e : this->jac_table()) {
            const int col = data.v_scatter_loc(core_->jac_cols_[e.user_slot_], 0);
            if (col < 0) {
                cursor++;
                continue;
            }
            const bool lock_column = !data.unique_constraints_ && (VarClashes[col] != -1);
            if (lock_column) {
                ClashLocks[VarClashes[col]].lock();
            }
            KKTmat.valuePtr()[KKTLocs.data()[cursor]] += e.sign_ * core_->jac_cache_[e.user_slot_];
            if (lock_column) {
                ClashLocks[VarClashes[col]].unlock();
            }
            cursor++;
        }
    }
};

/// Builds the single-partition NonLinearProgram for an adapter core: the
/// objective piece, the constraint pieces the row counts call for, the staged
/// variable bounds, and the layout. The one production path AND the one the
/// tests assemble through.
std::shared_ptr<NonLinearProgram> make_nlp_program(const std::shared_ptr<NLPAdapterCore> &core);

} // namespace tycho::solvers
