// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Ipopt::TNLP adapter over a transcribed NonLinearProgram.
//
// This header is only compiled in builds configured with Ipopt support: it
// includes Ipopt headers and is reachable only from the adapter translation
// unit and its build-gated tests. Nothing here is exposed to Python.
//
// Contracts this adapter maintains:
//
//   Identical NLP. Ipopt is handed exactly the problem the built-in solver
//   sees: n = primal_vars_ variables with no bounds, m = equal_cons_ +
//   inequal_cons_ general constraints ordered [equalities; inequalities],
//   h(x) = 0 and g(x) <= 0. Nothing is lifted, split, or re-expressed as a
//   variable bound, so the two backends solve the same NLP and their results
//   are directly comparable.
//
//   Shared KKT structure. The adapter owns its own KKT matrix and prepares it
//   exactly the way the built-in solver prepares its own — one
//   NonLinearProgram::analyze_sparsity call on that matrix. analyze_sparsity is
//   a pure function of the NLP's triplet bookkeeping, so the adapter's matrix
//   gets the identical CSR structure and the identical scatter locations, and
//   the Jacobian/Hessian entries Ipopt asks for are a flat gather through slot
//   maps built once from that structure.
//
//   Threading. All callbacks run on the thread that dispatched the solve. The
//   wrapped NonLinearProgram must not be evaluated concurrently from anywhere
//   else for the duration of a run: the adapter mutates the NLP's shared
//   scatter buffers on every evaluation, exactly as the built-in solver does.
//   The one place this adapter writes into the shared NLP itself (rather than
//   into its own KKT matrix or scratch buffers) is prepare_kkt_assembly,
//   which resets the NLP's set_primal_diags/e_pivots/i_pivots coefficient
//   counters before every Jacobian/Hessian assembly; this is safe because the
//   built-in solver re-seeds those same counters at the start of every
//   factorization, so nothing downstream of this adapter observes a stale
//   value.
//
//   Exception latching. A Tycho evaluation error cannot be allowed to unwind
//   through Ipopt's C++ stack, so every callback catches, stores the message,
//   and returns false. The backend entry point re-raises the latched message
//   as a std::runtime_error once Ipopt has returned.

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

// Ipopt's pkg-config Cflags point at the coin-or include directory itself, so
// its headers are included unprefixed (this is also how Ipopt's own headers
// include one another).
#include <IpTNLP.hpp>

#include "tycho/detail/solvers/non_linear_program.h"

namespace tycho::solvers {

/// Ipopt::TNLP view of a transcribed NonLinearProgram. Construct one per solve;
/// it caches per-point evaluations and stores the final iterate for the caller.
class TychoTNLP final : public Ipopt::TNLP {
  public:
    using Index = Ipopt::Index;
    using Number = Ipopt::Number;

    /// @param nlp        Transcribed NLP to wrap (must be non-null).
    /// @param x0         Starting point, sized primal_vars_.
    /// @param obj_scale  Objective scale factor applied to the objective and
    ///                   all of its derivatives, matching the built-in solver's
    ///                   Settings::obj_scale_.
    TychoTNLP(std::shared_ptr<NonLinearProgram> nlp, Eigen::VectorXd x0, double obj_scale);

    TychoTNLP(const TychoTNLP &) = delete;
    TychoTNLP &operator=(const TychoTNLP &) = delete;

    // --- Ipopt::TNLP interface ---

    bool get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
                      IndexStyleEnum &index_style) override;

    bool get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l,
                         Number *g_u) override;

    bool get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *z_L, Number *z_U,
                            Index m, bool init_lambda, Number *lambda) override;

    bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value) override;

    bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f) override;

    bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) override;

    bool eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac, Index *iRow,
                    Index *jCol, Number *values) override;

    bool eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m,
                const Number *lambda, bool new_lambda, Index nele_hess, Index *iRow, Index *jCol,
                Number *values) override;

    void finalize_solution(Ipopt::SolverReturn status, Index n, const Number *x, const Number *z_L,
                           const Number *z_U, Index m, const Number *g, const Number *lambda,
                           Number obj_value, const Ipopt::IpoptData *ip_data,
                           Ipopt::IpoptCalculatedQuantities *ip_cq) override;

    // --- Results and diagnostics ---

    const Eigen::VectorXd &solution() const { return x_final_; }
    const Eigen::VectorXd &eq_lmults() const { return eq_lmults_final_; }
    const Eigen::VectorXd &iq_lmults() const { return iq_lmults_final_; }
    double final_objective() const { return obj_final_; }

    /// Message of the first evaluation error latched during the run, empty if
    /// no callback failed.
    const std::string &latched_error() const { return latched_error_; }

    /// max(|h(x)|_inf, max(g(x), 0)_inf) at @p x, for callers that need a
    /// constraint-violation measure Ipopt did not report.
    double constraint_violation(const Eigen::VectorXd &x);

    int primal_vars() const { return primal_vars_; }
    int equal_cons() const { return equal_cons_; }
    int inequal_cons() const { return inequal_cons_; }

  private:
    /// Classify every stored KKT entry once and record the value-array position
    /// each Jacobian / Hessian triplet gathers from.
    void build_slot_maps();

    /// Evaluate objective value, objective gradient, and constraints at @p x
    /// unless the cache already holds that point.
    void refresh_point(const Number *x, bool new_x);

    /// Zero the KKT values and the scatter accumulators before an assembly.
    void prepare_kkt_assembly();

    std::shared_ptr<NonLinearProgram> nlp_;

    /// Adapter-owned KKT matrix, structurally identical to the built-in
    /// solver's.
    Eigen::SparseMatrix<double, Eigen::RowMajor> kkt_;

    Eigen::VectorXd x0_;
    double obj_scale_ = 1.0;

    int primal_vars_ = 0;
    int slack_vars_ = 0;
    int equal_cons_ = 0;
    int inequal_cons_ = 0;

    // Slot maps: triplet structure plus the KKT value-array position each
    // triplet reads, so per-iteration extraction is a flat gather.
    std::vector<int> jac_slots_;
    std::vector<int> hess_slots_;
    std::vector<Index> jac_rows_;
    std::vector<Index> jac_cols_;
    std::vector<Index> hess_rows_;
    std::vector<Index> hess_cols_;

    // Per-point cache (objective value, objective gradient, constraints).
    Eigen::VectorXd x_cache_;
    bool point_valid_ = false;
    double obj_cache_ = 0.0;
    Eigen::VectorXd pgx_cache_;
    Eigen::VectorXd fxe_cache_;
    Eigen::VectorXd fxi_cache_;

    // Scratch for the derivative assemblies (the NLP scatters into these).
    Eigen::VectorXd pgx_scratch_;
    Eigen::VectorXd agx_scratch_;
    Eigen::VectorXd fxe_scratch_;
    Eigen::VectorXd fxi_scratch_;
    Eigen::VectorXd le_scratch_;
    Eigen::VectorXd li_scratch_;

    // Final iterate, filled by finalize_solution.
    Eigen::VectorXd x_final_;
    Eigen::VectorXd eq_lmults_final_;
    Eigen::VectorXd iq_lmults_final_;
    // NaN rather than 0.0 so a run that never reaches finalize_solution (an
    // early abort) reports an implausible sentinel instead of a plausible
    // objective value; IpoptRunInfo::ran_ is the authoritative "did this run"
    // flag, but final_objective() should not silently look like a converged
    // zero-cost solution on the no-run path.
    double obj_final_ = std::numeric_limits<double>::quiet_NaN();

    std::string latched_error_;
};

} // namespace tycho::solvers
