// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers/nlp_adapter.h"

#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "tycho/detail/solvers/non_linear_program.h"

namespace tycho::solvers {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
} // namespace

NLPRowClassification NLPRowClassification::classify(ConstEigenRef<Eigen::VectorXd> gl,
                                                    ConstEigenRef<Eigen::VectorXd> gu) {
    if (gl.size() != gu.size()) {
        throw std::invalid_argument(
            fmt::format("NLP row classification: g_lower has {} rows but g_upper has {}", gl.size(),
                        gu.size()));
    }
    const int m = static_cast<int>(gl.size());
    NLPRowClassification rc;
    rc.kinds_.resize(m);
    rc.eq_row_ = Eigen::VectorXi::Constant(m, -1);
    rc.iq_upper_row_ = Eigen::VectorXi::Constant(m, -1);
    rc.iq_lower_row_ = Eigen::VectorXi::Constant(m, -1);
    for (int r = 0; r < m; r++) {
        const double lo = gl[r], up = gu[r];
        if (std::isnan(lo) || std::isnan(up)) {
            throw std::invalid_argument(
                fmt::format("constraint row {}: bound is NaN (lower={}, upper={})", r, lo, up));
        }
        if (lo > up) {
            throw std::invalid_argument(
                fmt::format("constraint row {}: lower bound {} exceeds upper bound {}", r, lo, up));
        }
        if (lo == up) {
            if (!std::isfinite(lo)) {
                throw std::invalid_argument(fmt::format(
                    "constraint row {}: both bounds are {} — an equality at infinity", r, lo));
            }
            rc.kinds_[r] = NLPRowKind::Equality;
            rc.eq_row_[r] = rc.num_eq_++;
        } else if (lo == -kInf && up == kInf) {
            rc.kinds_[r] = NLPRowKind::Free;
        } else if (lo == -kInf) {
            rc.kinds_[r] = NLPRowKind::UpperBounded;
            rc.iq_upper_row_[r] = rc.num_iq_++;
        } else if (up == kInf) {
            rc.kinds_[r] = NLPRowKind::LowerBounded;
            rc.iq_lower_row_[r] = rc.num_iq_++;
        } else {
            rc.kinds_[r] = NLPRowKind::Range;
            rc.iq_upper_row_[r] = rc.num_iq_++;
            rc.iq_lower_row_[r] = rc.num_iq_++;
        }
    }
    return rc;
}

NLPAdapterCore::NLPAdapterCore(std::shared_ptr<NLPProblem> problem) : problem_(std::move(problem)) {
    if (!problem_) {
        throw std::invalid_argument("NLPAdapterCore: the problem pointer is null");
    }
    n_ = problem_->num_vars();
    m_ = problem_->num_cons();
    jac_nnz_ = problem_->num_jac_nonzeros();
    hess_nnz_ = problem_->num_hess_nonzeros();
    if (n_ <= 0) {
        throw std::invalid_argument(
            fmt::format("{}: num_vars() must be positive, got {}", problem_->name(), n_));
    }
    if (m_ < 0 || jac_nnz_ < 0 || hess_nnz_ < 0) {
        throw std::invalid_argument(fmt::format(
            "{}: num_cons()={}, num_jac_nonzeros()={}, num_hess_nonzeros()={} — all must be "
            "non-negative",
            problem_->name(), m_, jac_nnz_, hess_nnz_));
    }
    if (m_ == 0 && jac_nnz_ != 0) {
        throw std::invalid_argument(
            fmt::format("{}: {} Jacobian nonzeros declared for a problem with no constraints",
                        problem_->name(), jac_nnz_));
    }

    x_lower_.resize(n_);
    x_upper_.resize(n_);
    Eigen::VectorXd gl(m_), gu(m_);
    problem_->bounds(x_lower_, x_upper_, gl, gu);
    for (int i = 0; i < n_; i++) {
        if (std::isnan(x_lower_[i]) || std::isnan(x_upper_[i])) {
            throw std::invalid_argument(
                fmt::format("{}: variable {} bound is NaN", problem_->name(), i));
        }
        if (x_lower_[i] > x_upper_[i]) {
            throw std::invalid_argument(
                fmt::format("{}: variable {} lower bound {} exceeds upper bound {}",
                            problem_->name(), i, x_lower_[i], x_upper_[i]));
        }
    }
    rows_ = NLPRowClassification::classify(gl, gu);

    jac_rows_.resize(jac_nnz_);
    jac_cols_.resize(jac_nnz_);
    if (jac_nnz_ > 0) {
        problem_->jac_structure(jac_rows_, jac_cols_);
    }
    for (int s = 0; s < jac_nnz_; s++) {
        if (jac_rows_[s] < 0 || jac_rows_[s] >= m_ || jac_cols_[s] < 0 || jac_cols_[s] >= n_) {
            throw std::invalid_argument(
                fmt::format("{}: Jacobian slot {} is ({}, {}), outside the {}x{} constraint "
                            "Jacobian",
                            problem_->name(), s, jac_rows_[s], jac_cols_[s], m_, n_));
        }
    }
    hess_rows_.resize(hess_nnz_);
    hess_cols_.resize(hess_nnz_);
    if (hess_nnz_ > 0) {
        problem_->hess_structure(hess_rows_, hess_cols_);
    }
    for (int s = 0; s < hess_nnz_; s++) {
        if (hess_rows_[s] < 0 || hess_rows_[s] >= n_ || hess_cols_[s] < 0 || hess_cols_[s] >= n_) {
            throw std::invalid_argument(
                fmt::format("{}: Hessian slot {} is ({}, {}), outside the {}x{} Hessian",
                            problem_->name(), s, hess_rows_[s], hess_cols_[s], n_, n_));
        }
        if (hess_rows_[s] < hess_cols_[s]) {
            throw std::invalid_argument(fmt::format(
                "{}: Hessian slot {} is ({}, {}) — above the diagonal. Declare the LOWER "
                "triangle of the Lagrangian Hessian (row >= col)",
                problem_->name(), s, hess_rows_[s], hess_cols_[s]));
        }
    }

    // Residual tables, in solver-row order. Classification assigned solver rows
    // in user-row order (a range row's upper part before its lower part), and
    // the loops here follow the same order, so entry k of each table IS local
    // solver row k of its piece.
    for (int r = 0; r < m_; r++) {
        switch (rows_.kinds_[r]) {
        case NLPRowKind::Equality:
            eq_res_.push_back({r, 1.0, gl[r]});
            break;
        case NLPRowKind::UpperBounded:
            iq_res_.push_back({r, 1.0, gu[r]});
            break;
        case NLPRowKind::LowerBounded:
            iq_res_.push_back({r, -1.0, gl[r]});
            break;
        case NLPRowKind::Range:
            iq_res_.push_back({r, 1.0, gu[r]});
            iq_res_.push_back({r, -1.0, gl[r]});
            break;
        case NLPRowKind::Free:
            break;
        }
    }

    for (int s = 0; s < jac_nnz_; s++) {
        const int r = jac_rows_[s];
        switch (rows_.kinds_[r]) {
        case NLPRowKind::Equality:
            eq_jac_.push_back({s, rows_.eq_row_[r], 1.0});
            break;
        case NLPRowKind::UpperBounded:
            iq_jac_.push_back({s, rows_.iq_upper_row_[r], 1.0});
            break;
        case NLPRowKind::LowerBounded:
            iq_jac_.push_back({s, rows_.iq_lower_row_[r], -1.0});
            break;
        case NLPRowKind::Range:
            iq_jac_.push_back({s, rows_.iq_upper_row_[r], 1.0});
            iq_jac_.push_back({s, rows_.iq_lower_row_[r], -1.0});
            break;
        case NLPRowKind::Free:
            break;
        }
    }

    hess_owner_ = (rows_.num_iq_ > 0)   ? HessOwner::IqPiece
                  : (rows_.num_eq_ > 0) ? HessOwner::EqPiece
                                        : HessOwner::Objective;

    g_cache_.resize(m_);
    jac_cache_.resize(jac_nnz_);
    grad_scratch_.resize(n_);
    lambda_user_.resize(m_);
    hess_vals_.resize(hess_nnz_);
}

void NLPAdapterCore::sync_x(ConstEigenRef<Eigen::VectorXd> x) {
    if (x.size() != n_) {
        throw std::invalid_argument(
            fmt::format("{}: the solver handed a {}-element iterate to a {}-variable problem",
                        problem_->name(), x.size(), n_));
    }
    if (x_cache_.size() != n_ || x_cache_ != x) {
        x_cache_ = x;
        g_valid_ = false;
        jac_valid_ = false;
    }
}

void NLPAdapterCore::refresh_g(ConstEigenRef<Eigen::VectorXd> x) {
    sync_x(x);
    if (!g_valid_) {
        if (m_ > 0) {
            problem_->eval_g(x, g_cache_);
        }
        g_valid_ = true;
    }
}

void NLPAdapterCore::refresh_jac(ConstEigenRef<Eigen::VectorXd> x) {
    sync_x(x);
    if (!jac_valid_) {
        if (jac_nnz_ > 0) {
            problem_->eval_jac(x, jac_cache_);
        }
        jac_valid_ = true;
    }
}

void NLPAdapterCore::compose_user_lambda(ConstEigenRef<Eigen::VectorXd> le,
                                         ConstEigenRef<Eigen::VectorXd> li) {
    for (int r = 0; r < m_; r++) {
        switch (rows_.kinds_[r]) {
        case NLPRowKind::Equality:
            lambda_user_[r] = (le.size() > 0) ? le[rows_.eq_row_[r]] : 0.0;
            break;
        case NLPRowKind::UpperBounded:
            lambda_user_[r] = (li.size() > 0) ? li[rows_.iq_upper_row_[r]] : 0.0;
            break;
        case NLPRowKind::LowerBounded:
            lambda_user_[r] = (li.size() > 0) ? -li[rows_.iq_lower_row_[r]] : 0.0;
            break;
        case NLPRowKind::Range:
            lambda_user_[r] =
                (li.size() > 0) ? li[rows_.iq_upper_row_[r]] - li[rows_.iq_lower_row_[r]] : 0.0;
            break;
        case NLPRowKind::Free:
            lambda_user_[r] = 0.0;
            break;
        }
    }
}

void NLPAdapterCore::eval_hessian_values(ConstEigenRef<Eigen::VectorXd> x, double obj_factor,
                                         ConstEigenRef<Eigen::VectorXd> le,
                                         ConstEigenRef<Eigen::VectorXd> li) {
    compose_user_lambda(le, li);
    if (hess_nnz_ > 0) {
        problem_->eval_hess(x, obj_factor, lambda_user_, hess_vals_);
    }
}

std::shared_ptr<NonLinearProgram> make_nlp_program(const std::shared_ptr<NLPAdapterCore> &core) {
    const int n = core->n_;
    Eigen::MatrixXi vindex(n, 1);
    for (int i = 0; i < n; i++) {
        vindex(i, 0) = i;
    }

    auto nlp = std::make_shared<NonLinearProgram>(1);

    ObjectiveFunction obj(ObjectiveInterface(NLPObjectivePiece(core)), vindex);
    obj.thread_mode_ = ThreadingFlags::MainThread;
    nlp->objectives_.push_back(obj);

    if (core->rows_.num_eq_ > 0) {
        Eigen::MatrixXi cindex(core->rows_.num_eq_, 1);
        for (int k = 0; k < core->rows_.num_eq_; k++) {
            cindex(k, 0) = k;
        }
        ConstraintFunction eq(ConstraintInterface(NLPConstraintPiece(core, false)), vindex, cindex);
        eq.thread_mode_ = ThreadingFlags::MainThread;
        nlp->equality_constraints_.push_back(eq);
    }
    if (core->rows_.num_iq_ > 0) {
        Eigen::MatrixXi cindex(core->rows_.num_iq_, 1);
        for (int k = 0; k < core->rows_.num_iq_; k++) {
            cindex(k, 0) = k;
        }
        ConstraintFunction iq(ConstraintInterface(NLPConstraintPiece(core, true)), vindex, cindex);
        iq.thread_mode_ = ThreadingFlags::MainThread;
        nlp->inequality_constraints_.push_back(iq);
    }

    for (int i = 0; i < n; i++) {
        if (std::isfinite(core->x_lower_[i]) || std::isfinite(core->x_upper_[i])) {
            nlp->set_variable_bound(i, core->x_lower_[i], core->x_upper_[i]);
        }
    }

    nlp->make_nlp(n, core->rows_.num_eq_, core->rows_.num_iq_);
    return nlp;
}

} // namespace tycho::solvers
