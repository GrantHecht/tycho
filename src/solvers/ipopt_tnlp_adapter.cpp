// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Ipopt backend: the TNLP adapter's implementation plus the backend entry
// points the solve dispatch seam calls. Only compiled in builds configured with
// Ipopt support; the fallback translation unit takes its place otherwise.

#include "tycho/detail/solvers/ipopt/tnlp_adapter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/core.h>
#include <fmt/format.h>

#include <IpIpoptApplication.hpp>
#include <IpRegOptions.hpp>
#include <IpSolveStatistics.hpp>

#include "tycho/detail/solvers/ipopt_backend.h"
#include "tycho/detail/solvers/optimization_problem_base.h"

namespace tycho::solvers {

namespace {

/// Ipopt treats bounds at or beyond this magnitude as infinite (its documented
/// nlp_lower_bound_inf / nlp_upper_bound_inf defaults).
constexpr double kIpoptInfinity = 1.0e19;

/// Latched when a TNLP callback is unwound by something that does not derive
/// from std::exception. Every callback below is a C++ boundary Ipopt calls
/// through its own stack, so nothing may escape it -- the catch-all keeps that
/// contract complete and still reports the failure instead of swallowing it.
constexpr const char *kUnknownCallbackError =
    "unknown exception (not derived from std::exception) escaped an NLP evaluation callback";

/// Spelling of an ApplicationReturnStatus, reported verbatim in the run info so
/// an unmapped outcome is still identifiable.
std::string application_status_name(Ipopt::ApplicationReturnStatus status) {
    switch (status) {
    case Ipopt::Solve_Succeeded:
        return "Solve_Succeeded";
    case Ipopt::Solved_To_Acceptable_Level:
        return "Solved_To_Acceptable_Level";
    case Ipopt::Infeasible_Problem_Detected:
        return "Infeasible_Problem_Detected";
    case Ipopt::Search_Direction_Becomes_Too_Small:
        return "Search_Direction_Becomes_Too_Small";
    case Ipopt::Diverging_Iterates:
        return "Diverging_Iterates";
    case Ipopt::User_Requested_Stop:
        return "User_Requested_Stop";
    case Ipopt::Feasible_Point_Found:
        return "Feasible_Point_Found";
    case Ipopt::Maximum_Iterations_Exceeded:
        return "Maximum_Iterations_Exceeded";
    case Ipopt::Restoration_Failed:
        return "Restoration_Failed";
    case Ipopt::Error_In_Step_Computation:
        return "Error_In_Step_Computation";
    case Ipopt::Maximum_CpuTime_Exceeded:
        return "Maximum_CpuTime_Exceeded";
    case Ipopt::Maximum_WallTime_Exceeded:
        return "Maximum_WallTime_Exceeded";
    case Ipopt::Not_Enough_Degrees_Of_Freedom:
        return "Not_Enough_Degrees_Of_Freedom";
    case Ipopt::Invalid_Problem_Definition:
        return "Invalid_Problem_Definition";
    case Ipopt::Invalid_Option:
        return "Invalid_Option";
    case Ipopt::Invalid_Number_Detected:
        return "Invalid_Number_Detected";
    case Ipopt::Unrecoverable_Exception:
        return "Unrecoverable_Exception";
    case Ipopt::NonIpopt_Exception_Thrown:
        return "NonIpopt_Exception_Thrown";
    case Ipopt::Insufficient_Memory:
        return "Insufficient_Memory";
    case Ipopt::Internal_Error:
        return "Internal_Error";
    }
    return "Unknown_Status";
}

/// Normalized outcome name plus the convergence flag the solve dispatch returns.
std::pair<std::string, ConvergenceFlags> normalize_status(Ipopt::ApplicationReturnStatus status) {
    switch (status) {
    case Ipopt::Solve_Succeeded:
    case Ipopt::Feasible_Point_Found:
        return {"converged", ConvergenceFlags::CONVERGED};
    case Ipopt::Solved_To_Acceptable_Level:
        return {"acceptable", ConvergenceFlags::ACCEPTABLE};
    case Ipopt::Infeasible_Problem_Detected:
        return {"infeasible", ConvergenceFlags::NOTCONVERGED};
    case Ipopt::Diverging_Iterates:
        return {"diverged", ConvergenceFlags::DIVERGING};
    default:
        return {"failed", ConvergenceFlags::NOTCONVERGED};
    }
}

/// Apply one user option, routed by the type the option was registered with
/// (rather than by how the value happens to parse): an integer-typed option
/// goes through the integer setter, a number-typed option through the
/// numeric setter, and everything else (including string-typed options)
/// through the string setter. This avoids the parse-and-retry approach's
/// failure mode, where a valid option still triggers one or two rejected
/// Ipopt::SetIntegerValue/SetNumericValue attempts that Ipopt logs verbosely
/// to stdout before the setter that actually applies the option is tried.
void apply_user_option(Ipopt::IpoptApplication &app, const std::string &key,
                       const std::string &value) {
    Ipopt::SmartPtr<Ipopt::OptionsList> options = app.Options();
    Ipopt::SmartPtr<const Ipopt::RegisteredOption> reg_option = app.RegOptions()->GetOption(key);
    if (Ipopt::IsNull(reg_option)) {
        throw std::runtime_error(fmt::format("no Ipopt option named '{0}'", key));
    }

    bool applied = false;
    switch (reg_option->Type()) {
    case Ipopt::OT_Integer: {
        long long as_int = 0;
        bool parsed = false;
        try {
            std::size_t pos = 0;
            as_int = std::stoll(value, &pos);
            parsed = (pos == value.size());
        } catch (const std::exception &) {
            parsed = false;
        }
        const long long lo = static_cast<long long>(std::numeric_limits<Ipopt::Index>::min());
        const long long hi = static_cast<long long>(std::numeric_limits<Ipopt::Index>::max());
        if (!parsed || as_int < lo || as_int > hi) {
            throw std::runtime_error(
                fmt::format("Ipopt option '{0}' expects an integer value, got '{1}'", key, value));
        }
        applied = options->SetIntegerValue(key, static_cast<Ipopt::Index>(as_int), true, true);
        break;
    }
    case Ipopt::OT_Number: {
        double as_num = 0.0;
        bool parsed = false;
        try {
            std::size_t pos = 0;
            as_num = std::stod(value, &pos);
            parsed = (pos == value.size());
        } catch (const std::exception &) {
            parsed = false;
        }
        if (!parsed) {
            throw std::runtime_error(
                fmt::format("Ipopt option '{0}' expects a numeric value, got '{1}'", key, value));
        }
        applied = options->SetNumericValue(key, as_num, true, true);
        break;
    }
    default:
        applied = options->SetStringValue(key, value, true, true);
        break;
    }

    if (!applied) {
        throw std::runtime_error(fmt::format("Ipopt rejected option '{0}' = '{1}'", key, value));
    }
}

} // namespace

// -----------------------------------------------------------------------------
// TychoTNLP
// -----------------------------------------------------------------------------

TychoTNLP::TychoTNLP(std::shared_ptr<NonLinearProgram> nlp, Eigen::VectorXd x0, double obj_scale)
    : nlp_(std::move(nlp)), x0_(std::move(x0)), obj_scale_(obj_scale) {

    if (!nlp_) {
        throw std::invalid_argument("TychoTNLP: NonLinearProgram pointer must not be null");
    }

    primal_vars_ = nlp_->primal_vars_;
    slack_vars_ = nlp_->slack_vars_;
    equal_cons_ = nlp_->equal_cons_;
    inequal_cons_ = nlp_->inequal_cons_;

    if (primal_vars_ < 1) {
        throw std::invalid_argument(fmt::format(
            "TychoTNLP: NLP has {0} primal variables; at least one is required", primal_vars_));
    }
    if (nlp_->kkt_dim_ != primal_vars_ + slack_vars_ + equal_cons_ + inequal_cons_) {
        throw std::logic_error(
            fmt::format("TychoTNLP: NLP kkt_dim ({0}) != primal_vars ({1}) + slack_vars ({2}) + "
                        "equal_cons ({3}) + inequal_cons ({4})",
                        nlp_->kkt_dim_, primal_vars_, slack_vars_, equal_cons_, inequal_cons_));
    }
    if (x0_.size() != primal_vars_) {
        throw std::invalid_argument(
            fmt::format("TychoTNLP: starting point has {0} entries, expected {1} (primal_vars)",
                        x0_.size(), primal_vars_));
    }

    x_cache_.setZero(primal_vars_);
    pgx_cache_.setZero(primal_vars_);
    fxe_cache_.setZero(equal_cons_);
    fxi_cache_.setZero(inequal_cons_);

    pgx_scratch_.setZero(primal_vars_);
    agx_scratch_.setZero(primal_vars_);
    fxe_scratch_.setZero(equal_cons_);
    fxi_scratch_.setZero(inequal_cons_);
    le_scratch_.setZero(equal_cons_);
    li_scratch_.setZero(inequal_cons_);

    // A run that never reaches finalize_solution (an early Ipopt abort) still
    // hands the caller a well-formed iterate: the point it started from.
    x_final_ = x0_;
    eq_lmults_final_.setZero(equal_cons_);
    iq_lmults_final_.setZero(inequal_cons_);

    // Same matrix preparation the built-in solver performs when it takes
    // ownership of an NLP: size the KKT matrix, then let the NLP compute its
    // sparsity pattern and scatter locations into it.
    kkt_.resize(nlp_->kkt_dim_, nlp_->kkt_dim_);
    nlp_->analyze_sparsity(kkt_);

    build_slot_maps();
}

void TychoTNLP::build_slot_maps() {
    // The assembled KKT matrix stores one triangle: every entry sits at
    // (min(row, col), max(row, col)) over the [primals | slacks | equalities |
    // inequalities] index layout. Classifying by that pair splits the stored
    // entries into the Hessian block (both endpoints primal), the constraint
    // Jacobian (one primal endpoint, one constraint endpoint), and the solver's
    // own slack/pivot bookkeeping, which is not part of the NLP Ipopt sees.
    const int cons_start = primal_vars_ + slack_vars_;

    jac_slots_.clear();
    hess_slots_.clear();
    jac_rows_.clear();
    jac_cols_.clear();
    hess_rows_.clear();
    hess_cols_.clear();

    // Every stored KKT entry is classified into at most one of these six
    // vectors, so nonZeros() is a safe (if loose, since it also bounds the
    // solver-bookkeeping entries none of these vectors receive) upper bound
    // that avoids reallocation growth during the walk below.
    const std::size_t nnz = static_cast<std::size_t>(kkt_.nonZeros());
    jac_slots_.reserve(nnz);
    hess_slots_.reserve(nnz);
    jac_rows_.reserve(nnz);
    jac_cols_.reserve(nnz);
    hess_rows_.reserve(nnz);
    hess_cols_.reserve(nnz);

    const auto *outer = kkt_.outerIndexPtr();
    const auto *inner = kkt_.innerIndexPtr();

    for (int r = 0; r < kkt_.outerSize(); ++r) {
        for (int k = outer[r]; k < outer[r + 1]; ++k) {
            const int c = static_cast<int>(inner[k]);
            const int lo = std::min(r, c);
            const int hi = std::max(r, c);

            if (hi < primal_vars_) {
                // Hessian of the Lagrangian. Ipopt takes the lower triangle
                // (row >= col), so the larger index is the row.
                hess_slots_.push_back(k);
                hess_rows_.push_back(static_cast<Index>(hi));
                hess_cols_.push_back(static_cast<Index>(lo));
            } else if (lo < primal_vars_ && hi >= cons_start) {
                // Constraint Jacobian. Constraint rows run [equalities;
                // inequalities] immediately after the slack block, which is the
                // order eval_g reports them in.
                jac_slots_.push_back(k);
                jac_rows_.push_back(static_cast<Index>(hi - cons_start));
                jac_cols_.push_back(static_cast<Index>(lo));
            }
            // Everything else is solver bookkeeping (slack Jacobian, slack
            // Hessian diagonal, constraint-row pivots) and has no counterpart
            // in the NLP handed to Ipopt.
        }
    }
}

void TychoTNLP::prepare_kkt_assembly() {
    std::fill_n(kkt_.valuePtr(), kkt_.nonZeros(), 0.0);

    pgx_scratch_.setZero();
    agx_scratch_.setZero();
    fxe_scratch_.setZero();
    fxi_scratch_.setZero();

    // Every KKT assembly also scatters the NLP's solver-supplied coefficients.
    // Those are zero at rest, but the primal-diagonal block lands on Hessian
    // slots this adapter gathers from, so the resting state is asserted rather
    // than assumed.
    nlp_->set_primal_diags(0.0);
    nlp_->set_e_pivots(0.0);
    nlp_->set_i_pivots(0.0);
}

void TychoTNLP::refresh_point(const Number *x, bool new_x) {
    Eigen::Map<const Eigen::VectorXd> xv(x, primal_vars_);

    if (point_valid_ && !new_x && (x_cache_.array() == xv.array()).all()) {
        return;
    }

    x_cache_ = xv;
    obj_cache_ = 0.0;
    pgx_cache_.setZero();
    fxe_cache_.setZero();
    fxi_cache_.setZero();
    point_valid_ = false;

    nlp_->eval_ogc(obj_scale_, x_cache_, obj_cache_, pgx_cache_, fxe_cache_, fxi_cache_);

    point_valid_ = true;
}

double TychoTNLP::constraint_violation(const Eigen::VectorXd &x) {
    if (x.size() != primal_vars_) {
        throw std::invalid_argument(fmt::format(
            "TychoTNLP::constraint_violation: point has {0} entries, expected {1} (primal_vars)",
            x.size(), primal_vars_));
    }

    double val = 0.0;
    fxe_scratch_.setZero();
    fxi_scratch_.setZero();
    nlp_->eval_occ(obj_scale_, x, val, fxe_scratch_, fxi_scratch_);

    double violation = 0.0;
    if (equal_cons_ > 0) {
        violation = std::max(violation, fxe_scratch_.cwiseAbs().maxCoeff());
    }
    if (inequal_cons_ > 0) {
        violation = std::max(violation, fxi_scratch_.maxCoeff());
    }
    return std::max(violation, 0.0);
}

bool TychoTNLP::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
                             IndexStyleEnum &index_style) {
    try {
        n = static_cast<Index>(primal_vars_);
        m = static_cast<Index>(equal_cons_ + inequal_cons_);
        nnz_jac_g = static_cast<Index>(jac_slots_.size());
        nnz_h_lag = static_cast<Index>(hess_slots_.size());
        index_style = TNLP::C_STYLE;
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

bool TychoTNLP::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l,
                                Number *g_u) {
    try {
        if (n != static_cast<Index>(primal_vars_) ||
            m != static_cast<Index>(equal_cons_ + inequal_cons_)) {
            throw std::logic_error(
                fmt::format("get_bounds_info called with n={0}, m={1}; expected n={2}, m={3}", n, m,
                            primal_vars_, equal_cons_ + inequal_cons_));
        }

        // Forward the NLP's dense variable bounds, clamped to the magnitude
        // Ipopt treats as infinite. nlp_->x_lower_/x_upper_ are materialized
        // by make_nlp from whatever the transcription layer staged via
        // set_variable_bound; where nothing was staged for a variable, both
        // vectors carry +-inf and this loop reproduces the previous
        // unconditionally-unbounded behavior.
        for (Index i = 0; i < n; ++i) {
            x_l[i] = std::max(nlp_->x_lower_[i], -kIpoptInfinity);
            x_u[i] = std::min(nlp_->x_upper_[i], kIpoptInfinity);
        }

        // Equalities h(x) = 0 first, then inequalities g(x) <= 0.
        for (int i = 0; i < equal_cons_; ++i) {
            g_l[i] = 0.0;
            g_u[i] = 0.0;
        }
        for (int i = 0; i < inequal_cons_; ++i) {
            g_l[equal_cons_ + i] = -kIpoptInfinity;
            g_u[equal_cons_ + i] = 0.0;
        }
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

bool TychoTNLP::get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *z_L,
                                   Number *z_U, Index m, bool init_lambda, Number *lambda) {
    try {
        if (n != static_cast<Index>(primal_vars_) ||
            m != static_cast<Index>(equal_cons_ + inequal_cons_)) {
            throw std::logic_error(
                fmt::format("get_starting_point called with n={0}, m={1}; expected n={2}, m={3}", n,
                            m, primal_vars_, equal_cons_ + inequal_cons_));
        }

        if (init_x) {
            Eigen::Map<Eigen::VectorXd>(x, primal_vars_) = x0_;
        }
        // No multiplier estimate is carried across the seam, so a
        // warm-started run gets neutral duals rather than a stale guess.
        if (init_z) {
            std::fill_n(z_L, primal_vars_, 1.0);
            std::fill_n(z_U, primal_vars_, 1.0);
        }
        if (init_lambda) {
            std::fill_n(lambda, equal_cons_ + inequal_cons_, 0.0);
        }
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

bool TychoTNLP::eval_f(Index n, const Number *x, bool new_x, Number &obj_value) {
    try {
        if (n != static_cast<Index>(primal_vars_)) {
            throw std::logic_error(
                fmt::format("eval_f called with n={0}, expected {1}", n, primal_vars_));
        }
        refresh_point(x, new_x);
        obj_value = obj_cache_;
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

bool TychoTNLP::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f) {
    try {
        if (n != static_cast<Index>(primal_vars_)) {
            throw std::logic_error(
                fmt::format("eval_grad_f called with n={0}, expected {1}", n, primal_vars_));
        }
        refresh_point(x, new_x);
        Eigen::Map<Eigen::VectorXd>(grad_f, primal_vars_) = pgx_cache_;
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

bool TychoTNLP::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) {
    try {
        if (n != static_cast<Index>(primal_vars_) ||
            m != static_cast<Index>(equal_cons_ + inequal_cons_)) {
            throw std::logic_error(fmt::format("eval_g called with n={0}, m={1}; expected n={2}, "
                                               "m={3}",
                                               n, m, primal_vars_, equal_cons_ + inequal_cons_));
        }
        refresh_point(x, new_x);
        if (equal_cons_ > 0) {
            Eigen::Map<Eigen::VectorXd>(g, equal_cons_) = fxe_cache_;
        }
        if (inequal_cons_ > 0) {
            Eigen::Map<Eigen::VectorXd>(g + equal_cons_, inequal_cons_) = fxi_cache_;
        }
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

bool TychoTNLP::eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac,
                           Index *iRow, Index *jCol, Number *values) {
    (void)new_x;
    try {
        if (n != static_cast<Index>(primal_vars_) ||
            m != static_cast<Index>(equal_cons_ + inequal_cons_) ||
            nele_jac != static_cast<Index>(jac_slots_.size())) {
            throw std::logic_error(fmt::format(
                "eval_jac_g called with n={0}, m={1}, nele_jac={2}; expected n={3}, "
                "m={4}, nele_jac={5}",
                n, m, nele_jac, primal_vars_, equal_cons_ + inequal_cons_, jac_slots_.size()));
        }

        if (values == nullptr) {
            std::copy(jac_rows_.begin(), jac_rows_.end(), iRow);
            std::copy(jac_cols_.begin(), jac_cols_.end(), jCol);
            return true;
        }

        Eigen::Map<const Eigen::VectorXd> xv(x, primal_vars_);
        prepare_kkt_assembly();
        double val = 0.0;
        nlp_->eval_soe(obj_scale_, xv, le_scratch_, li_scratch_, val, pgx_scratch_, agx_scratch_,
                       fxe_scratch_, fxi_scratch_, kkt_);

        const double *kkt_values = kkt_.valuePtr();
        for (std::size_t i = 0; i < jac_slots_.size(); ++i) {
            values[i] = kkt_values[jac_slots_[i]];
        }
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

bool TychoTNLP::eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m,
                       const Number *lambda, bool new_lambda, Index nele_hess, Index *iRow,
                       Index *jCol, Number *values) {
    (void)new_x;
    (void)new_lambda;
    try {
        if (n != static_cast<Index>(primal_vars_) ||
            m != static_cast<Index>(equal_cons_ + inequal_cons_) ||
            nele_hess != static_cast<Index>(hess_slots_.size())) {
            throw std::logic_error(fmt::format(
                "eval_h called with n={0}, m={1}, nele_hess={2}; expected n={3}, "
                "m={4}, nele_hess={5}",
                n, m, nele_hess, primal_vars_, equal_cons_ + inequal_cons_, hess_slots_.size()));
        }

        if (values == nullptr) {
            std::copy(hess_rows_.begin(), hess_rows_.end(), iRow);
            std::copy(hess_cols_.begin(), hess_cols_.end(), jCol);
            return true;
        }

        // Multiplier convention: Ipopt's Lagrangian is
        // obj_factor*f + sum_i lambda_i*g_i, and the NLP's KKT Hessian block is
        // ObjScale*grad^2 f + sum_i L_i*grad^2 c_i with the same sign on the
        // multipliers, so lambda maps onto (LE, LI) unchanged.
        if (equal_cons_ > 0) {
            le_scratch_ = Eigen::Map<const Eigen::VectorXd>(lambda, equal_cons_);
        }
        if (inequal_cons_ > 0) {
            li_scratch_ = Eigen::Map<const Eigen::VectorXd>(lambda + equal_cons_, inequal_cons_);
        }

        Eigen::Map<const Eigen::VectorXd> xv(x, primal_vars_);
        prepare_kkt_assembly();
        double val = 0.0;
        nlp_->eval_kkt(obj_factor * obj_scale_, xv, le_scratch_, li_scratch_, val, pgx_scratch_,
                       agx_scratch_, fxe_scratch_, fxi_scratch_, kkt_);

        const double *kkt_values = kkt_.valuePtr();
        for (std::size_t i = 0; i < hess_slots_.size(); ++i) {
            values[i] = kkt_values[hess_slots_[i]];
        }
        return true;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
        return false;
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
        return false;
    }
}

void TychoTNLP::finalize_solution(Ipopt::SolverReturn status, Index n, const Number *x,
                                  const Number *z_L, const Number *z_U, Index m, const Number *g,
                                  const Number *lambda, Number obj_value,
                                  const Ipopt::IpoptData *ip_data,
                                  Ipopt::IpoptCalculatedQuantities *ip_cq) {
    (void)status;
    (void)z_L;
    (void)z_U;
    (void)g;
    (void)ip_data;
    (void)ip_cq;

    // finalize_solution returns void, so an evaluation error latches into
    // latched_error_ the same as the bool-returning callbacks rather than
    // unwinding through Ipopt's C++ stack; a bad_alloc from the Eigen::Map
    // assignments below must not cross the Ipopt ABI either.
    try {
        if (x != nullptr && n == static_cast<Index>(primal_vars_)) {
            x_final_ = Eigen::Map<const Eigen::VectorXd>(x, primal_vars_);
        } else if (x != nullptr) {
            throw std::logic_error(
                fmt::format("finalize_solution called with n={0}, expected {1}", n, primal_vars_));
        }
        if (lambda != nullptr && m == static_cast<Index>(equal_cons_ + inequal_cons_)) {
            // Same sign mapping as eval_h: no negation between the two conventions.
            if (equal_cons_ > 0) {
                eq_lmults_final_ = Eigen::Map<const Eigen::VectorXd>(lambda, equal_cons_);
            }
            if (inequal_cons_ > 0) {
                iq_lmults_final_ =
                    Eigen::Map<const Eigen::VectorXd>(lambda + equal_cons_, inequal_cons_);
            }
        } else if (lambda != nullptr) {
            throw std::logic_error(fmt::format("finalize_solution called with m={0}, expected {1}",
                                               m, equal_cons_ + inequal_cons_));
        }
        obj_final_ = obj_value;
    } catch (const std::exception &e) {
        if (latched_error_.empty()) {
            latched_error_ = e.what();
        }
    } catch (...) {
        if (latched_error_.empty()) {
            latched_error_ = kUnknownCallbackError;
        }
    }
}

// -----------------------------------------------------------------------------
// Backend entry points
// -----------------------------------------------------------------------------

namespace ipopt_backend {

bool available() { return true; }

OptimizationProblemBase::NlpSolveOutput solve(OptimizationProblemBase &prob,
                                              OptimizationProblemBase::JetJobModes mode,
                                              const Eigen::VectorXd &input) {
    // The staged feasibility/optimality modes are an artifact of the built-in
    // solver's phase sequencing and have no Ipopt analog: every mode runs the
    // one NLP solve from the given starting point.
    (void)mode;

    if (!prob.nlp_) {
        throw std::runtime_error("no transcribed NLP on this problem; transcribe before "
                                 "dispatching a solve to the ipopt backend");
    }
    if (!prob.optimizer_) {
        throw std::runtime_error("no optimizer on this problem; the ipopt backend reads its "
                                 "termination tolerances from the built-in solver's settings");
    }

    const PSIOPT::Settings &settings = prob.optimizer_->settings();

    Ipopt::SmartPtr<TychoTNLP> adapter = new TychoTNLP(prob.nlp_, input, settings.obj_scale_);

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    Ipopt::SmartPtr<Ipopt::OptionsList> options = app->Options();

    // Matched-tolerance baseline: the same termination thresholds the built-in
    // solver would apply to this problem, so a backend comparison differs by
    // algorithm rather than by stopping rule.
    auto set_baseline_numeric = [&](const char *key, double value) {
        if (!options->SetNumericValue(key, value)) {
            throw std::runtime_error(
                fmt::format("Ipopt refused the baseline option '{0}' = {1}", key, value));
        }
    };
    auto set_baseline_integer = [&](const char *key, int value) {
        if (!options->SetIntegerValue(key, value)) {
            throw std::runtime_error(
                fmt::format("Ipopt refused the baseline option '{0}' = {1}", key, value));
        }
    };

    set_baseline_numeric("tol", settings.kkt_tol_);
    set_baseline_numeric("constr_viol_tol", std::max(settings.econ_tol_, settings.icon_tol_));
    set_baseline_numeric("acceptable_tol", settings.acc_kkt_tol_);
    set_baseline_numeric("acceptable_constr_viol_tol",
                         std::max(settings.acc_econ_tol_, settings.acc_icon_tol_));
    set_baseline_integer("max_iter", settings.max_iters_);
    set_baseline_integer("print_level", 0);
    // Banner suppression is an advanced Ipopt option; a build that does not
    // register it simply prints its banner.
    (void)options->SetStringValue("sb", "yes", true, true);

    for (const auto &[key, value] : prob.ipopt_options_) {
        apply_user_option(*app, key, value);
    }

    const Ipopt::ApplicationReturnStatus init_status = app->Initialize();
    if (init_status != Ipopt::Solve_Succeeded) {
        throw std::runtime_error(
            fmt::format("Ipopt initialization failed ({0}); check the options set through "
                        "ipopt_options_",
                        application_status_name(init_status)));
    }

    const auto wall_start = std::chrono::steady_clock::now();
    Ipopt::SmartPtr<Ipopt::TNLP> tnlp = Ipopt::GetRawPtr(adapter);
    const Ipopt::ApplicationReturnStatus status = app->OptimizeTNLP(tnlp);
    const double wall_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    if (!adapter->latched_error().empty()) {
        throw std::runtime_error(
            fmt::format("Ipopt run aborted by evaluation error: {0}", adapter->latched_error()));
    }

    const auto [normalized, flag] = normalize_status(status);

    IpoptRunInfo info;
    info.ran_ = true;
    info.status_ = application_status_name(status);
    info.normalized_ = normalized;
    info.converge_flag_ = flag;
    info.objective_ = adapter->final_objective();
    info.wall_time_s_ = wall_time;

    Ipopt::SmartPtr<Ipopt::SolveStatistics> stats = app->Statistics();
    if (Ipopt::IsValid(stats)) {
        info.iterations_ = stats->IterationCount();
        Ipopt::Number dual_inf = 0.0;
        Ipopt::Number constr_viol = 0.0;
        Ipopt::Number varbounds_viol = 0.0;
        Ipopt::Number complementarity = 0.0;
        Ipopt::Number kkt_error = 0.0;
        stats->Infeasibilities(dual_inf, constr_viol, varbounds_viol, complementarity, kkt_error);
        info.constraint_violation_ = constr_viol;
    } else {
        // An early abort leaves no statistics object; the iteration count keeps
        // its sentinel and the violation is measured directly at the iterate.
        info.constraint_violation_ = adapter->constraint_violation(adapter->solution());
    }

    prob.last_ipopt_result_ = info;

    OptimizationProblemBase::NlpSolveOutput out;
    out.variables_ = adapter->solution();
    out.eq_lmults_ = adapter->eq_lmults();
    out.iq_lmults_ = adapter->iq_lmults();
    out.flag_ = flag;
    return out;
}

} // namespace ipopt_backend
} // namespace tycho::solvers
