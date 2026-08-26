// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Selection of the NLP solver backend a problem is dispatched to.
//
// The solver library tycho builds on carries no interface to a foreign solver:
// its problem base runs the built-in interior-point solver and nothing else.
// The selector, the options forwarded to a foreign solver, the diagnostics of
// the last foreign run, and the dispatch that routes a solve to it therefore
// live on tycho's side of the boundary — here.
//
// This header deliberately contains no Ipopt includes and no Ipopt types, so it
// is safe to include in every build regardless of how the project was
// configured. The Ipopt dependency is confined to the translation unit that
// implements the backend.
//
// BackendProblemBase owns the NLP/solver handles and the partitioned-evaluation
// settings directly rather than inheriting them from hven's
// OptimizationProblemBase. hven's base pairs those handles with a NON-VIRTUAL
// run_nlp_solver(), and tycho needs to intercept that exact entry point to add
// backend selection; a derived class cannot override a non-virtual member, only
// hide it, and which of the two ran would then depend on the STATIC type of the
// object a call is made through rather than on what the object actually is.
// Owning the members locally turns that into one member with an internal
// branch instead of two members in a hide relationship. hven's own base
// (dep/hven/include/hven/drivers/optimization_problem_base.h) is unchanged and
// still used as-is by anything on the solver-library side of the boundary
// (e.g. NLPSolver).

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#include <fmt/format.h>

#include <hven/detail/drivers/interior_point_solver_fwd.h>
#include <hven/detail/interior/utils/get_core_count.h>
#include <hven/detail/interior/utils/thread_pool.h>
#include <hven/drivers/interior_point_solver.h>
#include <hven/model/non_linear_program.h>

#include "tycho/detail/hven_namespaces.h"

namespace tycho::solvers {

/// The transcribed problem published as a claim-stream-bearing provider. Named
/// here and defined in solvers_vf/transcribed_aggregate.h, so that a problem
/// can hold one without every consumer of this header seeing the layout it
/// publishes.
class TranscribedAggregate;

/// NLP solver backend selector (BackendProblemBase::nlp_solver_).
/// interior_point is the built-in interior-point solver (default). ipopt hands the
/// identical transcribed NLP to a linked Ipopt installation for reference
/// comparison; dispatching it requires a build configured with Ipopt support.
enum class NLPSolvers { interior_point = 0, ipopt = 1 };

/// Outcome of one Ipopt run on the transcribed NLP. Sentinel values (-1 /
/// empty / ran_ == false) mean no Ipopt solve has run on this problem object.
struct IpoptRunInfo {
    bool ran_ = false;
    std::string status_ = "";     ///< Raw Ipopt ApplicationReturnStatus name.
    std::string normalized_ = ""; ///< converged/acceptable/infeasible/failed/diverged.
    ConvergenceFlags converge_flag_ = ConvergenceFlags::NOTCONVERGED;
    int iterations_ = -1;
    double objective_ = 0.0;
    double constraint_violation_ = -1.0;
    double wall_time_s_ = -1.0;
};

/// Every tycho optimization problem — OptimizationProblem, ODEPhaseBase and
/// OptimalControlProblemBase alike — derives from this rather than from the
/// solver library's OptimizationProblemBase directly, so that backend
/// selection is available wherever a problem is. It is the class the Python
/// binding exposes under the name "OptimizationProblemBase"; the distinct C++
/// name keeps the two apart at every unqualified use site.
struct BackendProblemBase {

    /// @brief Which solve-mode entry point jet_run() dispatches to.
    enum class JetJobModes {
        NotSet,
        /// Parsed by strto_jet_job_mode, but dispatched by nothing: both
        /// jet_run() and run_nlp_solver() reject it with
        /// std::invalid_argument.
        DoNothing,
        /// @brief Dispatches solve().
        Solve,
        /// @brief Dispatches optimize().
        Optimize,
        /// @brief Dispatches solve_optimize().
        SolveOptimize,
        /// @brief Dispatches solve_optimize_solve().
        SolveOptimizeSolve,
        /// @brief Dispatches optimize_solve().
        OptimizeSolve
    };

    /// @brief Number of evaluation partitions the NLP is split over.
    int num_partitions_ = 1;
    /// @brief The mode jet_run() dispatches on.
    JetJobModes jet_job_mode_ = JetJobModes::NotSet;

    /// @brief The problem's program (built by the derived class's make_nlp()).
    std::shared_ptr<NonLinearProgram> nlp_;
    /// @brief The shared interior-point solver instance.
    std::shared_ptr<InteriorPointSolver> optimizer_;

    /// NLP solver backend for the solve/optimize entry points. interior_point (default)
    /// is the built-in path, byte-identical to previous behavior. ipopt runs
    /// the identical transcribed NLP through a linked Ipopt installation
    /// (requires a build with Ipopt support; throws std::runtime_error
    /// otherwise). Not usable in a Jet batch run — see the guard in jet_run().
    NLPSolvers nlp_solver_ = NLPSolvers::interior_point;

    /// String key/value options forwarded verbatim to Ipopt (e.g.
    /// {"linear_solver", "pardisomkl"}). Applied after the matched-tolerance
    /// baseline, so user entries win. Ignored by the interior-point backend.
    std::map<std::string, std::string> ipopt_options_;

    /// Diagnostics of the most recent ipopt-backend run on this problem.
    IpoptRunInfo last_ipopt_result_;

    /// The transcribed problem published as a claim-stream-bearing provider:
    /// the declaration this problem was laid from, and the per-slot coordinates
    /// a consumer needs in order to lay a destination of its own. Replaced by
    /// every transcription, and null until the first one runs.
    std::shared_ptr<TranscribedAggregate> provider_;

    virtual ~BackendProblemBase() = default;

    /// @brief Constructs the solver and applies the default partitioning.
    BackendProblemBase() {
        this->optimizer_ = std::make_shared<InteriorPointSolver>();
        this->init_partitions();
    }

    /// Runs the feasibility (SOE-mode) phase sequence for the derived
    /// problem's NLP.
    virtual ConvergenceFlags solve() = 0;
    /// Runs the optimality (OPT-mode) phase sequence for the derived
    /// problem's NLP.
    virtual ConvergenceFlags optimize() = 0;
    /// Runs the SOE-mode phase sequence, then the OPT-mode one. Both always
    /// run.
    virtual ConvergenceFlags solve_optimize() = 0;
    /// Runs SOE, then OPT, then SOE again. The trailing SOE phase is
    /// conditional: it is skipped when OPT reported
    /// ConvergenceFlags::CONVERGED.
    virtual ConvergenceFlags solve_optimize_solve() = 0;
    /// Runs the OPT-mode phase sequence, then the SOE-mode one. The trailing
    /// SOE phase is conditional: it is skipped when OPT reported
    /// ConvergenceFlags::CONVERGED.
    virtual ConvergenceFlags optimize_solve() = 0;

    /// Compute default partition count from the global thread budget.
    /// Over-partitions by 4x so the work-stealing pool can smooth out
    /// unequal partition costs.
    ///
    /// @return 1 on a single-thread budget, else 4x the thread count.
    static int default_num_partitions() {
        int nt = hven::utils::get_num_threads();
        if (nt <= 1)
            return 1;
        return nt * 4;
    }

    /// Applies the default partitioning: num_partitions_ from
    /// default_num_partitions(), and the solver's QP thread count capped at
    /// the physical core count.
    virtual void init_partitions() {
        this->num_partitions_ = default_num_partitions();
        this->optimizer_->set_qp_threads(
            std::min(HVEN_DEFAULT_QP_THREADS, hven::utils::get_core_count()));
    }

    /// @brief Sets the number of evaluation partitions the problem is split over.
    /// @param num_partitions Partition count; must be positive.
    ///
    /// Partition count and QP thread count are independent settings and are set
    /// independently: the solver's own QP thread count is
    /// `optimizer_->set_qp_threads(n)`.
    ///
    /// @throws std::invalid_argument if `num_partitions < 1`.
    virtual void set_num_partitions(int num_partitions) {
        if (num_partitions < 1) {
            throw std::invalid_argument("Number of partitions must be positive");
        }
        this->num_partitions_ = num_partitions;
    }

    /// Prepares the problem for inline (non-partitioned) evaluation inside
    /// jet_run(); must leave num_partitions_ == 1.
    virtual void jet_initialize() = 0;

    /// @brief Releases whatever jet_initialize() acquired.
    virtual void jet_release() = 0;

    /// Runs the configured job mode between jet_initialize()/jet_release(),
    /// returning the dispatched mode's convergence flag.
    ///
    /// Concurrency guard, checked before jet_initialize() mutates this problem
    /// and before any solve work begins. Jet::map builds each problem inside
    /// its worker job, so this entry point is the first place the selected
    /// backend is observable; rejecting here means no batch element ever
    /// reaches an Ipopt solve. Ipopt is not reliably re-entrant, so running
    /// several of its solves concurrently is unsupported rather than merely
    /// untested.
    ///
    /// IMPORTANT: jet_run() is called from Jet::map() on pool worker threads.
    /// If jet_initialize() did NOT set num_partitions_=1, the NLP eval methods
    /// would call parallel_sequence/parallel_task from a pool worker,
    /// triggering the nested-dispatch guard (std::logic_error).
    /// jet_initialize() MUST set num_partitions_=1 so NLP eval methods run
    /// inline.
    ///
    /// @throws std::invalid_argument if nlp_solver_ is ipopt, or if
    /// jet_job_mode_ is NotSet or otherwise unrecognized.
    ConvergenceFlags jet_run() {
        if (this->nlp_solver_ == NLPSolvers::ipopt) {
            throw std::invalid_argument(
                "nlp_solver=ipopt cannot be used in a Jet batch run: Ipopt is not reliably "
                "re-entrant, so concurrent solves through it are unsupported. Run the ipopt "
                "backend one solve at a time (solve/optimize on a single problem), or set "
                "nlp_solver=interior_point for the batch.");
        }

        this->jet_initialize();

        ConvergenceFlags flag;

        switch (this->jet_job_mode_) {
        case JetJobModes::Solve: {
            flag = this->solve();
            break;
        }
        case JetJobModes::Optimize: {
            flag = this->optimize();
            break;
        }
        case JetJobModes::SolveOptimize: {
            flag = this->solve_optimize();
            break;
        }
        case JetJobModes::SolveOptimizeSolve: {
            flag = this->solve_optimize_solve();
            break;
        }
        case JetJobModes::OptimizeSolve: {
            flag = this->optimize_solve();
            break;
        }
        case JetJobModes::NotSet: {
            throw ::std::invalid_argument("jet_job_mode_ not set");
        }
        default:
            throw std::invalid_argument("Unrecognized jet_job_mode");
        }

        this->jet_release();
        return flag;
    }

    /// Uniform output of one solve: the updated variable vector, the
    /// constraint multipliers, and the convergence flag.
    struct NlpSolveOutput {
        /// @brief Updated variable vector from the solve.
        Eigen::VectorXd variables_;
        /// @brief Equality-constraint multipliers from the final result.
        Eigen::VectorXd eq_lmults_;
        /// @brief Inequality-constraint multipliers from the final result.
        Eigen::VectorXd iq_lmults_;
        /// @brief Convergence flag from the final result.
        ConvergenceFlags flag_ = ConvergenceFlags::NOTCONVERGED;
    };

    /// Single backend dispatch point for the five solve modes, mapping each
    /// onto the matching InteriorPointSolver entry point and collecting the
    /// uniform result above (interior-point branch), or running a single
    /// Ipopt solve on the transcribed NLP (ipopt branch — the
    /// feasibility-then-optimize staging has no Ipopt analog).
    ///
    /// Defined out of line below, after the ipopt_backend::solve declaration it
    /// calls (a member function body defined in the class cannot name a
    /// namespace-scope function declared later in the file).
    ///
    /// @throws std::invalid_argument if `mode` is NotSet, DoNothing, or any
    /// other value with no entry point.
    NlpSolveOutput run_nlp_solver(JetJobModes mode, const Eigen::VectorXd &input);

    /// Parses a job-mode name into its enum value. Accepted spellings:
    /// "solve"/"Solve", "optimize"/"Optimize",
    /// "solve_optimize"/"SolveOptimize"/"Solve_Optimize",
    /// "solve_optimize_solve"/"SolveOptimizeSolve"/"Solve_Optimize_Solve",
    /// "optimize_solve"/"OptimizeSolve"/"Optimize_Solve",
    /// "DoNothing"/"do_nothing"/"Do_Nothing".
    ///
    /// @throws std::invalid_argument on any other spelling.
    static JetJobModes strto_jet_job_mode(const std::string &str) {

        if (str == "solve" || str == "Solve")
            return JetJobModes::Solve;
        else if (str == "optimize" || str == "Optimize")
            return JetJobModes::Optimize;
        else if (str == "solve_optimize" || str == "SolveOptimize" || str == "Solve_Optimize")
            return JetJobModes::SolveOptimize;
        else if (str == "solve_optimize_solve" || str == "SolveOptimizeSolve" ||
                 str == "Solve_Optimize_Solve")
            return JetJobModes::SolveOptimizeSolve;
        else if (str == "optimize_solve" || str == "OptimizeSolve" || str == "Optimize_Solve")
            return JetJobModes::OptimizeSolve;
        else if (str == "DoNothing" || str == "do_nothing" || str == "Do_Nothing")
            return JetJobModes::DoNothing;
        else {
            auto msg = fmt::format("Unrecognized jet_job_mode: {0}\n", str);
            throw std::invalid_argument(msg);
        }
    }

    /// @brief Sets the mode jet_run() dispatches on (enum overload).
    void set_jet_job_mode(JetJobModes m) { this->jet_job_mode_ = m; }

    /// Sets the mode jet_run() dispatches on, parsing the same spellings
    /// strto_jet_job_mode accepts.
    void set_jet_job_mode(const std::string &str) {
        this->set_jet_job_mode(strto_jet_job_mode(str));
    }
};

namespace ipopt_backend {

/// True when this build was configured with Ipopt support linked in.
bool available();

/// Run Ipopt on the problem's transcribed NLP. A real implementation is linked
/// only in builds configured with Ipopt support; the stub throws
/// std::runtime_error.
BackendProblemBase::NlpSolveOutput
solve(BackendProblemBase &prob, BackendProblemBase::JetJobModes mode, const Eigen::VectorXd &input);

} // namespace ipopt_backend

inline BackendProblemBase::NlpSolveOutput
BackendProblemBase::run_nlp_solver(JetJobModes mode, const Eigen::VectorXd &input) {
    // This site branches on `== NLPSolvers::ipopt`, while the call-impl sites
    // in ode_phase_base.cpp / optimal_control_problem.cpp branch on
    // `== NLPSolvers::interior_point` (or its negation) to make the same backend
    // choice. Both predicates are equivalent today (only two backends exist),
    // but if a third backend enumerator is ever added, all of these sites
    // should be unified on `!= NLPSolvers::interior_point` ("anything that isn't interior_point
    // uses the generic NLP-solver path") rather than each needing its own added
    // `== <new-backend>` branch.
    if (this->nlp_solver_ == NLPSolvers::ipopt) {
        return ipopt_backend::solve(*this, mode, input);
    }

    NlpSolveOutput out;
    switch (mode) {
    case JetJobModes::Solve:
        out.variables_ = this->optimizer_->solve(input);
        break;
    case JetJobModes::Optimize:
        out.variables_ = this->optimizer_->optimize(input);
        break;
    case JetJobModes::SolveOptimize:
        out.variables_ = this->optimizer_->solve_optimize(input);
        break;
    case JetJobModes::SolveOptimizeSolve:
        out.variables_ = this->optimizer_->solve_optimize_solve(input);
        break;
    case JetJobModes::OptimizeSolve:
        out.variables_ = this->optimizer_->optimize_solve(input);
        break;
    default:
        throw std::invalid_argument("Unrecognized NLP solve mode");
    }
    out.eq_lmults_ = this->optimizer_->result().eq_lmults_;
    out.iq_lmults_ = this->optimizer_->result().iq_lmults_;
    out.flag_ = this->optimizer_->result().converge_flag_;
    return out;
}

} // namespace tycho::solvers
