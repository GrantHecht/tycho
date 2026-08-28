// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// The solve-call value types: what a `solve()` call hands back, independent of
// which engine ran it or how many stages it took.
//
// A SolveResult is engine-neutral by construction: it carries the declared-
// space warm-start currency (hven::solvers::WarmStartData) and the
// declaration-identity stamp that currency was taken under
// (decltype(WarmStartData::structure_key_), i.e. hven's DeclarationKey --
// spelled via decltype rather than named directly so this header tracks
// whatever type that field actually is), plus one StageResult per solver
// stage that ran (presolve?, main, polish?) and one PhaseResult per OCP phase
// (empty for a bare VF problem). `flag_` mirrors the final -- i.e. last in
// `stages_` -- stage's convergence flag: that stage is the one whose result
// the caller actually gets.

#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <hven/warmstart/warm_start_data.h>

#include "tycho/detail/hven_namespaces.h"

namespace tycho::solvers {

/// @brief Which objective a solve call pursued: drive to optimality, or only
///        to feasibility.
enum class Mode {
    Optimal = 0,
    Feasible = 1,
};

/// @brief `Mode` spelled for display -- "Optimal" / "Feasible".
const char *to_string(Mode m);

/// @brief `Mode` parsed from a case-insensitive string -- "optimal" /
///        "feasible" map to `Mode::Optimal` / `Mode::Feasible`.
/// @throws std::invalid_argument naming the unrecognized string and both
///         legal values, for anything else.
Mode mode_from_string(const std::string &s);

/// @brief One solver stage's outcome: which stage it was, which engine ran
///        it, and the numbers that describe how it finished.
///
/// `engine_details_`/`engine_notes_` are an additive annex -- engine-specific
/// numeric and string diagnostics that do not warrant a named field here --
/// kept as plain maps rather than one opaque blob so they stay trivially
/// picklable and additive.
///
/// The three residual fields default to NaN, not 0.0: NaN is this project's
/// cross-engine "unmeasured" value (the same convention hven's own solution
/// records carry), and every engine writes NaN into a residual it did not
/// measure. A 0.0 default would make a stage that reported nothing read as
/// converged to machine precision.
struct StageResult {
    std::string role_;        ///< "presolve" | "main" | "polish".
    std::string engine_name_; ///< Engine class name, e.g. "InteriorPointSolver".
    tycho::ConvergenceFlags flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
    int iterations_ = 0;
    double objective_ = 0.0;                                         ///< Caller's scale.
    double kkt_residual_ = std::numeric_limits<double>::quiet_NaN(); ///< NaN: unmeasured.
    double eq_violation_ = std::numeric_limits<double>::quiet_NaN(); ///< Max-norm; NaN: unmeasured.
    double iq_violation_ = std::numeric_limits<double>::quiet_NaN(); ///< Max-norm; NaN: unmeasured.
    double wall_time_s_ = 0.0;
    std::map<std::string, double> engine_details_;    ///< Engine-specific numeric annex.
    std::map<std::string, std::string> engine_notes_; ///< Engine-specific string annex.
};

/// @brief One OCP phase's slice of a solve, keyed the same way as the OCP
///        itself (`index_ == 0` for a single Phase, no OCP).
///
/// Every field is a snapshot taken at solve time -- sliced out of the full
/// primal/dual solution once, never re-derived lazily -- so a `PhaseResult`
/// stays valid after the problem it came from is re-transcribed or destroyed.
struct PhaseResult {
    int index_ = 0;                     ///< OCP phase index; 0 for a single Phase.
    int var_start_ = 0, var_count_ = 0; ///< This phase's slice of the full primal space.
    int eq_start_ = 0, eq_count_ = 0;   ///< Rows in the full equality space.
    int iq_start_ = 0, iq_count_ = 0;   ///< Rows in the full inequality space.
    Eigen::VectorXd eq_lmults_;
    Eigen::VectorXd iq_lmults_;
    Eigen::VectorXd bound_lmults_; ///< Declared-space signed z = zL - zU slice.
};

/// @brief What a `solve()` call hands back: the deciding convergence flag,
///        every stage that ran, every OCP phase's slice (empty for a bare VF
///        problem), and the declared-space warm-start currency taken from the
///        final deciding stage.
struct SolveResult {
    tycho::ConvergenceFlags flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
    std::vector<StageResult> stages_;   ///< Run order: presolve?, main, polish?.
    std::vector<PhaseResult> phases_;   ///< Index-keyed like the OCP; empty for the VF problem.
    hven::solvers::WarmStartData warm_; ///< Declared-space payload from the final deciding stage.
    decltype(hven::solvers::WarmStartData::structure_key_) structure_key_;

    /// @brief True for CONVERGED or ACCEPTABLE -- ACCEPTABLE is convergence to
    ///        the acceptable tolerance ladder, still a caller-usable answer.
    bool converged() const {
        return flag_ == tycho::ConvergenceFlags::CONVERGED ||
               flag_ == tycho::ConvergenceFlags::ACCEPTABLE;
    }

    explicit operator bool() const { return converged(); }

    /// @throws std::logic_error if `stages_` is empty -- there is no stage to
    ///         report on.
    const StageResult &final_stage() const;

    double objective() const { return final_stage().objective_; }
    int iterations() const { return final_stage().iterations_; }
};

} // namespace tycho::solvers
