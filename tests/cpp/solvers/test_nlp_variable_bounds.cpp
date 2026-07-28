///////////////////////////////////////////////////////////////////////////////
// NLP variable-bound contract: NonLinearProgram::set_variable_bound,
// clear_variable_bounds, has_variable_bounds, and the x_lower_/x_upper_
// vectors make_nlp materializes from the staged declarations.
//
// The first group hand-builds a NonLinearProgram directly (no PSIOPT, no
// Phase/transcription) with empty objective/constraint lists, so make_nlp
// only has to run its own bookkeeping over primal variables.
//
// The second group covers the fixed-variable treatment: classification of the
// materialized bounds, the full<->reduced index maps, and end-to-end solves of
// hand-built QPs where a variable is pinned by equal bounds. Those solves drive
// a PSIOPT instance directly against the NLP an OptimizationProblem
// transcribed, so the bounds can be declared on the NLP between transcription
// and the solve.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/optimization_problem.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

using tycho::solvers::FixedVariableTreatments;
using tycho::solvers::NonLinearProgram;

namespace {
constexpr double kNlpVarBoundsInf = std::numeric_limits<double>::infinity();
}

TEST(NlpVariableBounds, DefaultsAreUnboundedAndHasNoBounds) {
    NonLinearProgram nlp(1);
    nlp.make_nlp(4, 0, 0);

    ASSERT_EQ(nlp.x_lower_.size(), 4);
    ASSERT_EQ(nlp.x_upper_.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(nlp.x_lower_[i], -kNlpVarBoundsInf);
        EXPECT_EQ(nlp.x_upper_[i], kNlpVarBoundsInf);
    }
    EXPECT_FALSE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, SingleSetRoundTrips) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(2, -1.5, 3.5);
    nlp.make_nlp(5, 0, 0);

    EXPECT_DOUBLE_EQ(nlp.x_lower_[2], -1.5);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[2], 3.5);

    // Every other index is left unbounded.
    for (int i = 0; i < 5; ++i) {
        if (i == 2)
            continue;
        EXPECT_EQ(nlp.x_lower_[i], -kNlpVarBoundsInf);
        EXPECT_EQ(nlp.x_upper_[i], kNlpVarBoundsInf);
    }
    EXPECT_TRUE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, TightestWinsOverOverlappingDeclarations) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, 0.0, 10.0);
    nlp.set_variable_bound(1, 2.0, 6.0);
    nlp.make_nlp(3, 0, 0);

    EXPECT_DOUBLE_EQ(nlp.x_lower_[1], 2.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[1], 6.0);
}

TEST(NlpVariableBounds, ConflictingBoundsThrowWithIndexInMessage) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(3, 5.0, 10.0);
    nlp.set_variable_bound(3, 20.0, 30.0); // merged: lower=max(5,20)=20, upper=min(10,30)=10

    try {
        nlp.make_nlp(6, 0, 0);
        FAIL() << "expected std::invalid_argument for a lower > upper conflict";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("3"), std::string::npos) << msg;
        EXPECT_NE(msg.find("20"), std::string::npos) << msg;
        EXPECT_NE(msg.find("10"), std::string::npos) << msg;
    }
}

TEST(NlpVariableBounds, EqualBoundsAreAcceptedAsAFixedVariable) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(0, 4.0, 4.0);
    nlp.make_nlp(2, 0, 0);

    EXPECT_DOUBLE_EQ(nlp.x_lower_[0], 4.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[0], 4.0);
}

TEST(NlpVariableBounds, OutOfRangeIndexThrows) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(10, -1.0, 1.0); // primal_vars_ will only be 3

    EXPECT_THROW(nlp.make_nlp(3, 0, 0), std::invalid_argument);
}

TEST(NlpVariableBounds, ReMakeNlpPreservesStagedBounds) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, -2.0, 2.0);

    nlp.make_nlp(4, 0, 0);
    EXPECT_DOUBLE_EQ(nlp.x_lower_[1], -2.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[1], 2.0);

    // Re-transcription: make_nlp runs again without an intervening
    // clear_variable_bounds() call. The staged declaration must still apply.
    nlp.make_nlp(4, 0, 0);
    EXPECT_DOUBLE_EQ(nlp.x_lower_[1], -2.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[1], 2.0);
    EXPECT_TRUE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, ClearVariableBoundsDropsStagedRecords) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, -2.0, 2.0);
    nlp.make_nlp(4, 0, 0);
    ASSERT_TRUE(nlp.has_variable_bounds());

    nlp.clear_variable_bounds();
    EXPECT_FALSE(nlp.has_variable_bounds());

    nlp.make_nlp(4, 0, 0);
    EXPECT_FALSE(nlp.has_variable_bounds());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(nlp.x_lower_[i], -kNlpVarBoundsInf);
        EXPECT_EQ(nlp.x_upper_[i], kNlpVarBoundsInf);
    }
}

TEST(NlpVariableBounds, BothBoundsInfiniteIsANoOp) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(0, -kNlpVarBoundsInf, kNlpVarBoundsInf);
    EXPECT_TRUE(nlp.staged_variable_bounds_.empty());

    nlp.make_nlp(2, 0, 0);
    EXPECT_FALSE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, NanBoundThrowsImmediately) {
    NonLinearProgram nlp(1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(nlp.set_variable_bound(0, nan, 1.0), std::invalid_argument);
    EXPECT_THROW(nlp.set_variable_bound(0, -1.0, nan), std::invalid_argument);
    EXPECT_TRUE(nlp.staged_variable_bounds_.empty());
}

///////////////////////////////////////////////////////////////////////////////
// Fixed-variable treatment: classification, index maps, reduction seam.
///////////////////////////////////////////////////////////////////////////////

namespace {

// An NLP with no user functions at all: make_nlp still lays out the solver's
// own KKT elements (one diagonal per primal variable), which is everything
// configure_variable_treatment needs to address the eliminated rows. Keeps the
// classification tests free of any function-evaluation machinery.
struct NlpVarBoundsBareNlp {
    NonLinearProgram nlp_{1};
    Eigen::SparseMatrix<double, Eigen::RowMajor> kkt_;

    void build(int num_vars) {
        this->nlp_.make_nlp(num_vars, 0, 0);
        this->nlp_.analyze_sparsity(this->kkt_);
    }
};

// Relaxed form of a finite bound, mirroring the widening
// configure_variable_treatment applies.
double nlp_var_bounds_relax(double bound, double factor, bool upper) {
    const double delta = factor * std::max(1.0, std::abs(bound));
    return upper ? bound + delta : bound - delta;
}

// Objective ||x||^2 over every variable. Deliberately depends on the pinned
// variable too: its own stationarity row is then 2*x_fixed != 0, so a solve
// that failed to drop that row would stall its KKT residual there, and one that
// failed to pin its KKT column would drag the variable toward zero. Both
// failures are visible in the assertions below.
template <int N>
void nlp_var_bounds_add_squared_norm_objective(tycho::solvers::OptimizationProblem &prob) {
    Eigen::VectorXi indices(N);
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }
    auto args = tycho::vf::Arguments<N>();
    auto obj = args.squared_norm();
    prob.add_objective(tycho::vf::GenericFunction<-1, 1>(obj), indices);
}

// Adds the equality constraint sum(x[indices]) == target over exactly two
// variables.
void nlp_var_bounds_add_pair_sum_con(tycho::solvers::OptimizationProblem &prob, int i0, int i1,
                                     double target) {
    auto args = tycho::vf::Arguments<2>();
    auto con = args.coeff<0>() + args.coeff<1>() - target;
    prob.add_equal_con(tycho::vf::GenericFunction<-1, -1>(con),
                       (Eigen::VectorXi(2) << i0, i1).finished());
}

// Same, over exactly three variables.
void nlp_var_bounds_add_triple_sum_con(tycho::solvers::OptimizationProblem &prob, int i0, int i1,
                                       int i2, double target) {
    auto args = tycho::vf::Arguments<3>();
    auto con = args.coeff<0>() + args.coeff<1>() + args.coeff<2>() - target;
    prob.add_equal_con(tycho::vf::GenericFunction<-1, -1>(con),
                       (Eigen::VectorXi(3) << i0, i1, i2).finished());
}

// Transcribes the problem and hands back the NLP it built, detached from the
// OptimizationProblem so bounds can be declared on it and the tests can drive
// their own PSIOPT against it.
std::shared_ptr<NonLinearProgram>
nlp_var_bounds_transcribe(tycho::solvers::OptimizationProblem &prob, int num_vars) {
    prob.set_vars(Eigen::VectorXd::Zero(num_vars));
    prob.transcribe();
    return prob.nlp_;
}

// Re-materializes the bounds staged on an already-transcribed NLP (make_nlp
// reallocates every KKT array, so the solver has to be re-pointed at it, which
// is what recomputes the storage locations the reduction addresses).
void nlp_var_bounds_rebuild(NonLinearProgram &nlp) {
    nlp.make_nlp(nlp.primal_vars_, nlp.equal_cons_, nlp.inequal_cons_);
}

} // namespace

// --- Classification -------------------------------------------------------

// Identity path: no declared bounds at all. Nothing is eliminated, the reduced
// width is the full width, the bound set is empty, and the maps stay empty --
// the state every evaluation-path guard reads to skip the reduction entirely.
TEST(NlpVarBoundsReduction, UnboundedProblemStaysUnreduced) {
    NlpVarBoundsBareNlp problem;
    problem.build(4);
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8);

    EXPECT_FALSE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 4);
    EXPECT_FALSE(problem.nlp_.variable_bound_set().any());
    EXPECT_EQ(problem.nlp_.fixed_variable_indices().size(), 0);
    EXPECT_EQ(problem.nlp_.full_to_reduced().size(), 0);
    EXPECT_EQ(problem.nlp_.reduced_to_full().size(), 0);

    // Both maps are pass-through copies on this path.
    Eigen::VectorXd full(4);
    full << 1.0, 2.0, 3.0, 4.0;
    Eigen::VectorXd reduced(4);
    problem.nlp_.gather_reduced_x(full, reduced);
    EXPECT_TRUE(reduced.isApprox(full));

    Eigen::VectorXd back(4);
    back.setZero();
    problem.nlp_.scatter_full_x(reduced, back);
    EXPECT_TRUE(back.isApprox(full));
}

// Every bound kind lands where it belongs: lower-only and upper-only variables
// appear in exactly one list, two-sided in both, fixed in neither (they are
// eliminated, not barriered), free in neither.
TEST(NlpVarBoundsReduction, ClassificationSplitsBoundKinds) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(0, -2.0, kNlpVarBoundsInf); // lower only
    problem.nlp_.set_variable_bound(1, -kNlpVarBoundsInf, 5.0); // upper only
    problem.nlp_.set_variable_bound(2, -1.0, 1.0);              // two sided
    problem.nlp_.set_variable_bound(3, 4.0, 4.0);               // fixed
    problem.build(5);                                           // index 4 stays free

    const double factor = 1.0e-8;
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, factor);

    const auto &bounds = problem.nlp_.variable_bound_set();
    ASSERT_EQ(bounds.lower_idx_.size(), 2);
    EXPECT_EQ(bounds.lower_idx_[0], 0);
    EXPECT_EQ(bounds.lower_idx_[1], 2);
    ASSERT_EQ(bounds.upper_idx_.size(), 2);
    EXPECT_EQ(bounds.upper_idx_[0], 1);
    EXPECT_EQ(bounds.upper_idx_[1], 2);
    EXPECT_TRUE(bounds.any());

    // Values are recorded already relaxed outward.
    EXPECT_DOUBLE_EQ(bounds.lower_val_[0], nlp_var_bounds_relax(-2.0, factor, false));
    EXPECT_DOUBLE_EQ(bounds.upper_val_[0], nlp_var_bounds_relax(5.0, factor, true));
    EXPECT_DOUBLE_EQ(bounds.lower_val_[1], nlp_var_bounds_relax(-1.0, factor, false));
    EXPECT_DOUBLE_EQ(bounds.upper_val_[1], nlp_var_bounds_relax(1.0, factor, true));

    // The fixed variable is eliminated, not bounded.
    ASSERT_EQ(problem.nlp_.fixed_variable_indices().size(), 1);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[0], 3);
    EXPECT_DOUBLE_EQ(problem.nlp_.fixed_variable_values()[0], 4.0);
    EXPECT_TRUE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 4);
}

// A zero relax factor records the declared bounds verbatim.
TEST(NlpVarBoundsReduction, ZeroRelaxFactorRecordsDeclaredBounds) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(1, -3.0, 7.0);
    problem.build(3);
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 0.0);

    const auto &bounds = problem.nlp_.variable_bound_set();
    ASSERT_EQ(bounds.lower_idx_.size(), 1);
    EXPECT_DOUBLE_EQ(bounds.lower_val_[0], -3.0);
    EXPECT_DOUBLE_EQ(bounds.upper_val_[0], 7.0);
}

// Two fixed variables interleaved with free ones: the maps must renumber the
// survivors in order and leave a hole at each eliminated index.
TEST(NlpVarBoundsReduction, InterleavedFixedVariablesMapCorrectly) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(1, -2.0, -2.0);
    problem.nlp_.set_variable_bound(3, 10.0, 10.0);
    problem.build(5);
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8);

    ASSERT_TRUE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 3);

    const auto &full_to_reduced = problem.nlp_.full_to_reduced();
    ASSERT_EQ(full_to_reduced.size(), 5);
    EXPECT_EQ(full_to_reduced[0], 0);
    EXPECT_EQ(full_to_reduced[1], -1);
    EXPECT_EQ(full_to_reduced[2], 1);
    EXPECT_EQ(full_to_reduced[3], -1);
    EXPECT_EQ(full_to_reduced[4], 2);

    const auto &reduced_to_full = problem.nlp_.reduced_to_full();
    ASSERT_EQ(reduced_to_full.size(), 3);
    EXPECT_EQ(reduced_to_full[0], 0);
    EXPECT_EQ(reduced_to_full[1], 2);
    EXPECT_EQ(reduced_to_full[2], 4);

    ASSERT_EQ(problem.nlp_.fixed_variable_indices().size(), 2);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[0], 1);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[1], 3);
    EXPECT_DOUBLE_EQ(problem.nlp_.fixed_variable_values()[0], -2.0);
    EXPECT_DOUBLE_EQ(problem.nlp_.fixed_variable_values()[1], 10.0);

    // Compact, then expand: the survivors round-trip and the eliminated
    // coordinates come back at their pinned values regardless of what the
    // full-space buffer held.
    Eigen::VectorXd full(5);
    full << 1.0, 99.0, 2.0, 99.0, 3.0;
    Eigen::VectorXd reduced(3);
    problem.nlp_.gather_reduced_x(full, reduced);
    EXPECT_DOUBLE_EQ(reduced[0], 1.0);
    EXPECT_DOUBLE_EQ(reduced[1], 2.0);
    EXPECT_DOUBLE_EQ(reduced[2], 3.0);

    Eigen::VectorXd back = Eigen::VectorXd::Constant(5, -7.0);
    problem.nlp_.scatter_full_x(reduced, back);
    EXPECT_DOUBLE_EQ(back[0], 1.0);
    EXPECT_DOUBLE_EQ(back[1], -2.0);
    EXPECT_DOUBLE_EQ(back[2], 2.0);
    EXPECT_DOUBLE_EQ(back[3], 10.0);
    EXPECT_DOUBLE_EQ(back[4], 3.0);

    // pin_fixed_variables touches only the eliminated coordinates.
    Eigen::VectorXd guess = Eigen::VectorXd::Constant(5, 0.5);
    problem.nlp_.pin_fixed_variables(guess);
    EXPECT_DOUBLE_EQ(guess[0], 0.5);
    EXPECT_DOUBLE_EQ(guess[1], -2.0);
    EXPECT_DOUBLE_EQ(guess[2], 0.5);
    EXPECT_DOUBLE_EQ(guess[3], 10.0);
    EXPECT_DOUBLE_EQ(guess[4], 0.5);

    // clear_fixed_variable_rows does the same to a primal-row vector.
    Eigen::VectorXd rows = Eigen::VectorXd::Constant(5, 3.0);
    problem.nlp_.clear_fixed_variable_rows(rows);
    EXPECT_DOUBLE_EQ(rows[0], 3.0);
    EXPECT_DOUBLE_EQ(rows[1], 0.0);
    EXPECT_DOUBLE_EQ(rows[2], 3.0);
    EXPECT_DOUBLE_EQ(rows[3], 0.0);
    EXPECT_DOUBLE_EQ(rows[4], 3.0);
}

// Re-entrancy: a repeat call with the same arguments and the same bound state
// is a no-op, and a call after the bounds changed re-classifies. The second
// half is what a solver instance solving twice against different bounds relies
// on.
TEST(NlpVarBoundsReduction, ReconfigureAfterBoundChangeReclassifies) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(2, 4.0, 4.0);
    problem.build(4);
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8);
    ASSERT_TRUE(problem.nlp_.is_reduced());
    ASSERT_EQ(problem.nlp_.fixed_variable_indices()[0], 2);
    EXPECT_DOUBLE_EQ(problem.nlp_.fixed_variable_values()[0], 4.0);

    // Same treatment, same factor, same bounds: idempotent.
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8);
    EXPECT_TRUE(problem.nlp_.is_reduced());
    ASSERT_EQ(problem.nlp_.fixed_variable_indices().size(), 1);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[0], 2);

    // Different bounds, re-materialized and re-analyzed: a different variable
    // is now the fixed one.
    problem.nlp_.clear_variable_bounds();
    problem.nlp_.set_variable_bound(0, -1.0, -1.0);
    problem.build(4);
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8);
    ASSERT_TRUE(problem.nlp_.is_reduced());
    ASSERT_EQ(problem.nlp_.fixed_variable_indices().size(), 1);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[0], 0);
    EXPECT_DOUBLE_EQ(problem.nlp_.fixed_variable_values()[0], -1.0);

    // Bounds dropped entirely: back to the identity path.
    problem.nlp_.clear_variable_bounds();
    problem.build(4);
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8);
    EXPECT_FALSE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 4);
}

// --- Rejected configurations ----------------------------------------------

TEST(NlpVarBoundsReduction, UnavailableTreatmentsThrow) {
    NlpVarBoundsBareNlp problem;
    problem.build(2);

    try {
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeConstraint, 1.0e-8);
        FAIL() << "expected std::invalid_argument for an unavailable treatment";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("make_constraint"), std::string::npos) << msg;
        EXPECT_NE(msg.find("make_parameter"), std::string::npos) << msg;
    }
    EXPECT_THROW(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::RelaxBounds, 1.0e-8),
        std::invalid_argument);
}

TEST(NlpVarBoundsReduction, InvalidRelaxFactorThrows) {
    NlpVarBoundsBareNlp problem;
    problem.build(2);
    EXPECT_THROW(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, -1.0),
        std::invalid_argument);
    EXPECT_THROW(problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter,
                                                           kNlpVarBoundsInf),
                 std::invalid_argument);
}

// Eliminating a variable addresses KKT storage locations, so it needs the
// sparsity pattern. Without a fixed variable there is nothing to address and
// the call succeeds.
TEST(NlpVarBoundsReduction, FixedVariableWithoutSparsityAnalysisThrows) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, 2.0, 2.0);
    nlp.make_nlp(3, 0, 0);

    try {
        nlp.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8);
        FAIL() << "expected std::invalid_argument without an analyzed sparsity pattern";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find("analyze_sparsity"), std::string::npos) << e.what();
    }

    NonLinearProgram unbounded(1);
    unbounded.make_nlp(3, 0, 0);
    EXPECT_NO_THROW(
        unbounded.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8));
    EXPECT_FALSE(unbounded.is_reduced());
}

TEST(NlpVarBoundsReduction, InfiniteFixingValueThrows) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(0, kNlpVarBoundsInf, kNlpVarBoundsInf);
    problem.build(2);
    EXPECT_THROW(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8),
        std::invalid_argument);
}

// --- End-to-end solves ----------------------------------------------------

// Reference: min ||x||^2 over three variables s.t. x0 + x2 = 6, no bounds.
// The optimum is (3, 0, 3). Pins the identity path end to end.
TEST(NlpVarBoundsReduction, UnboundedQpSolvesWithoutReduction) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_squared_norm_objective<3>(prob);
    nlp_var_bounds_add_pair_sum_con(prob, 0, 2, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);

    tycho::solvers::PSIOPT opt;
    opt.set_print_level(3);
    opt.set_nlp(nlp);
    Eigen::VectorXd solution = opt.optimize(Eigen::VectorXd::Zero(3));

    EXPECT_EQ(opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_FALSE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), nlp->primal_vars_);
    EXPECT_NEAR(solution[0], 3.0, 1e-6);
    EXPECT_NEAR(solution[1], 0.0, 1e-6);
    EXPECT_NEAR(solution[2], 3.0, 1e-6);
}

// Same problem with x1 pinned by equal bounds. The remaining variables must
// reach the SAME optimum as the hand-eliminated two-variable problem
// min ||y||^2 s.t. y0 + y1 = 6, and the pinned value must appear exactly in the
// returned full-space solution even though the initial guess disagreed with it.
TEST(NlpVarBoundsReduction, FixedVariableQpMatchesHandEliminatedProblem) {
    // Hand-eliminated reference.
    tycho::solvers::OptimizationProblem reference_prob;
    nlp_var_bounds_add_squared_norm_objective<2>(reference_prob);
    nlp_var_bounds_add_pair_sum_con(reference_prob, 0, 1, 6.0);
    auto reference_nlp = nlp_var_bounds_transcribe(reference_prob, 2);

    tycho::solvers::PSIOPT reference_opt;
    reference_opt.set_print_level(3);
    reference_opt.set_nlp(reference_nlp);
    Eigen::VectorXd reference = reference_opt.optimize(Eigen::VectorXd::Zero(2));
    ASSERT_EQ(reference_opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);

    // Full problem with the middle variable fixed.
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_squared_norm_objective<3>(prob);
    nlp_var_bounds_add_pair_sum_con(prob, 0, 2, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    tycho::solvers::PSIOPT opt;
    opt.set_print_level(3);
    opt.set_nlp(nlp);
    Eigen::VectorXd guess(3);
    guess << 0.0, -5.0, 0.0; // deliberately disagrees with the fixed value
    Eigen::VectorXd solution = opt.optimize(guess);

    EXPECT_EQ(opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 2);

    // The pinned value is exact, not merely converged to.
    EXPECT_DOUBLE_EQ(solution[1], 7.0);
    EXPECT_NEAR(solution[0], reference[0], 1e-6);
    EXPECT_NEAR(solution[2], reference[1], 1e-6);
    EXPECT_NEAR(solution[0], 3.0, 1e-6);
    EXPECT_NEAR(solution[2], 3.0, 1e-6);
}

// Two fixed variables interleaved with three free ones, with the constraint
// spanning only the free ones: index bookkeeping under interleaving, end to
// end. min ||x||^2 over five variables s.t. x0 + x2 + x4 = 6, x1 = -2, x3 = 10,
// whose free-block optimum is (2, 2, 2).
TEST(NlpVarBoundsReduction, TwoInterleavedFixedVariablesSolve) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_squared_norm_objective<5>(prob);
    nlp_var_bounds_add_triple_sum_con(prob, 0, 2, 4, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 5);
    nlp->set_variable_bound(1, -2.0, -2.0);
    nlp->set_variable_bound(3, 10.0, 10.0);
    nlp_var_bounds_rebuild(*nlp);

    tycho::solvers::PSIOPT opt;
    opt.set_print_level(3);
    opt.set_nlp(nlp);
    Eigen::VectorXd solution = opt.optimize(Eigen::VectorXd::Zero(5));

    EXPECT_EQ(opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 3);

    EXPECT_DOUBLE_EQ(solution[1], -2.0);
    EXPECT_DOUBLE_EQ(solution[3], 10.0);
    EXPECT_NEAR(solution[0], 2.0, 1e-6);
    EXPECT_NEAR(solution[2], 2.0, 1e-6);
    EXPECT_NEAR(solution[4], 2.0, 1e-6);
}

// One solver instance, two solves, different bounds in between: the second
// solve must re-classify rather than reuse the first solve's elimination.
TEST(NlpVarBoundsReduction, SecondSolveAfterBoundChangeUsesTheNewValue) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_squared_norm_objective<3>(prob);
    nlp_var_bounds_add_pair_sum_con(prob, 0, 2, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    tycho::solvers::PSIOPT opt;
    opt.set_print_level(3);
    opt.set_nlp(nlp);
    Eigen::VectorXd first = opt.optimize(Eigen::VectorXd::Zero(3));
    ASSERT_EQ(opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_DOUBLE_EQ(first[1], 7.0);

    nlp->clear_variable_bounds();
    nlp->set_variable_bound(1, 9.0, 9.0);
    nlp_var_bounds_rebuild(*nlp);
    opt.set_nlp(nlp);
    Eigen::VectorXd second = opt.optimize(Eigen::VectorXd::Zero(3));

    EXPECT_EQ(opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_DOUBLE_EQ(second[1], 9.0);
    EXPECT_NEAR(second[0], 3.0, 1e-6);
    EXPECT_NEAR(second[2], 3.0, 1e-6);

    // And dropping the bound entirely returns the problem to the identity path.
    nlp->clear_variable_bounds();
    nlp_var_bounds_rebuild(*nlp);
    opt.set_nlp(nlp);
    Eigen::VectorXd third = opt.optimize(Eigen::VectorXd::Zero(3));
    EXPECT_EQ(opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_FALSE(nlp->is_reduced());
    EXPECT_NEAR(third[1], 0.0, 1e-6);
}
