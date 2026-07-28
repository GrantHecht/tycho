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
// own KKT elements (one diagonal per primal variable), so the reduced layout is
// observable through kkt_dim_ without any function-evaluation machinery.
struct NlpVarBoundsBareNlp {
    NonLinearProgram nlp_{1};

    void build(int num_vars) { this->nlp_.make_nlp(num_vars, 0, 0); }
};

// Relaxed form of a finite bound, mirroring the widening
// configure_variable_treatment applies.
double nlp_var_bounds_relax(double bound, double factor, bool upper) {
    const double delta = factor * std::max(1.0, std::abs(bound));
    return upper ? bound + delta : bound - delta;
}

// Adds the objective term (x[index] - center)^2.
//
// Every solve below is built from SHIFTED squares rather than a plain ||x||^2,
// and started away from the minimizer, so that neither the objective value nor
// its gradient is identically zero at the initial point. A problem whose
// gradient vanishes at the start point is a degenerate input to the barrier /
// merit machinery and does not exercise anything this suite is about.
void nlp_var_bounds_add_shifted_square(tycho::solvers::OptimizationProblem &prob, int index,
                                       double center) {
    auto args = tycho::vf::Arguments<1>();
    auto term = (args.coeff<0>() - center).squared_norm();
    prob.add_objective(tycho::vf::GenericFunction<-1, 1>(term),
                       (Eigen::VectorXi(1) << index).finished());
}

// Adds one shifted square per variable: sum_i (x_i - centers[i])^2.
void nlp_var_bounds_add_separable_objective(tycho::solvers::OptimizationProblem &prob,
                                            const std::vector<double> &centers) {
    for (int i = 0; i < static_cast<int>(centers.size()); i++) {
        nlp_var_bounds_add_shifted_square(prob, i, centers[i]);
    }
}

// Adds the objective term (x[i0] + x[i1])^2 -- a genuine Hessian cross-term
// between two variables, which is what makes the elimination observable when
// one of them is fixed.
void nlp_var_bounds_add_pair_square(tycho::solvers::OptimizationProblem &prob, int i0, int i1) {
    auto args = tycho::vf::Arguments<2>();
    auto term = (args.coeff<0>() + args.coeff<1>()).squared_norm();
    prob.add_objective(tycho::vf::GenericFunction<-1, 1>(term),
                       (Eigen::VectorXi(2) << i0, i1).finished());
}

// Adds the equality constraint x[i0] == target.
void nlp_var_bounds_add_single_sum_con(tycho::solvers::OptimizationProblem &prob, int i0,
                                       double target) {
    auto args = tycho::vf::Arguments<1>();
    auto con = args.coeff<0>() - target;
    prob.add_equal_con(tycho::vf::GenericFunction<-1, -1>(con),
                       (Eigen::VectorXi(1) << i0).finished());
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

struct NlpVarBoundsSolveOutcome {
    Eigen::VectorXd solution_;
    tycho::ConvergenceFlags flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
};

// Drives a fresh silent PSIOPT against an NLP. Returns the flag alongside the
// solution rather than asserting internally, so every caller can make the
// convergence check a hard precondition (a reference problem that failed to
// solve must never be able to masquerade as agreement).
NlpVarBoundsSolveOutcome nlp_var_bounds_solve(const std::shared_ptr<NonLinearProgram> &nlp,
                                              const Eigen::VectorXd &guess) {
    tycho::solvers::PSIOPT opt;
    opt.set_print_level(3);
    opt.set_nlp(nlp);
    NlpVarBoundsSolveOutcome outcome;
    outcome.solution_ = opt.optimize(guess);
    outcome.flag_ = opt.result().converge_flag_;
    return outcome;
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
    EXPECT_EQ(problem.nlp_.kkt_dim_, 4); // unchanged: nothing was eliminated
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
    EXPECT_EQ(problem.nlp_.kkt_dim_, 4); // one column and row fewer than the five declared
}

// The bound set is indexed in the space the solver iterates in, so a bound on a
// variable that sits AFTER an eliminated one is renumbered. Index 0 is fixed and
// indices 1 and 3 are bounded, which the reduced space calls 0 and 2.
TEST(NlpVarBoundsReduction, BoundSetIndicesAreInTheReducedSpace) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(0, 1.0, 1.0);  // eliminated
    problem.nlp_.set_variable_bound(1, -3.0, 3.0); // reduced index 0
    problem.nlp_.set_variable_bound(3, 0.0, 8.0);  // reduced index 2
    problem.build(4);
    problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 0.0);

    ASSERT_TRUE(problem.nlp_.is_reduced());
    const auto &bounds = problem.nlp_.variable_bound_set();
    ASSERT_EQ(bounds.lower_idx_.size(), 2);
    EXPECT_EQ(bounds.lower_idx_[0], 0);
    EXPECT_EQ(bounds.lower_idx_[1], 2);
    EXPECT_DOUBLE_EQ(bounds.lower_val_[0], -3.0);
    EXPECT_DOUBLE_EQ(bounds.lower_val_[1], 0.0);
    ASSERT_EQ(bounds.upper_idx_.size(), 2);
    EXPECT_EQ(bounds.upper_idx_[0], 0);
    EXPECT_EQ(bounds.upper_idx_[1], 2);

    // And they name the variables the caller declared them on.
    EXPECT_EQ(problem.nlp_.reduced_to_full()[bounds.lower_idx_[0]], 1);
    EXPECT_EQ(problem.nlp_.reduced_to_full()[bounds.lower_idx_[1]], 3);
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
    EXPECT_EQ(problem.nlp_.kkt_dim_, 3); // two of the five columns and rows are gone

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

// Configuration owns the whole layout rebuild, so it needs nothing to have been
// prepared for it: a bare make_nlp is enough, and the KKT dimensions come back
// narrowed on the spot.
TEST(NlpVarBoundsReduction, ConfigurationRebuildsTheLayoutOnItsOwn) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, 2.0, 2.0);
    nlp.make_nlp(3, 0, 0);
    ASSERT_EQ(nlp.kkt_dim_, 3);

    EXPECT_TRUE(nlp.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8));
    EXPECT_TRUE(nlp.is_reduced());
    EXPECT_EQ(nlp.reduced_primal_vars(), 2);
    EXPECT_EQ(nlp.kkt_dim_, 2);

    // A repeat call rebuilds nothing and says so.
    EXPECT_FALSE(nlp.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8));

    NonLinearProgram unbounded(1);
    unbounded.make_nlp(3, 0, 0);
    // Nothing to eliminate: no rebuild, and the layout is untouched.
    EXPECT_FALSE(
        unbounded.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8));
    EXPECT_FALSE(unbounded.is_reduced());
    EXPECT_EQ(unbounded.kkt_dim_, 3);
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
//
// Every problem here is a shifted-square QP started away from its minimizer, so
// the objective and its gradient are both non-zero at the initial point, and
// every problem carries at least one equality constraint. Each equivalence test
// solves the hand-eliminated problem FIRST and asserts, as a hard precondition,
// both that the reference converged and that it reached its own closed-form
// optimum -- a reference that failed to solve can then never masquerade as
// agreement with the eliminated problem.

// Identity path end to end: min sum (x_i - c_i)^2 over three variables with
// c = (1, 2, 3), s.t. x0 + x2 = 6, no bounds. Stationarity gives
// x0 = 1 - l/2, x2 = 3 - l/2 with 4 - l = 6, so l = -2 and the optimum is
// (2, 2, 4).
TEST(NlpVarBoundsReduction, UnboundedQpSolvesWithoutReduction) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0, 3.0});
    nlp_var_bounds_add_pair_sum_con(prob, 0, 2, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_FALSE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), nlp->primal_vars_);
    EXPECT_EQ(nlp->kkt_dim_, nlp->primal_vars_ + nlp->equal_cons_);
    EXPECT_NEAR(outcome.solution_[0], 2.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[1], 2.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[2], 4.0, 1e-6);
}

// Same problem with x1 pinned by equal bounds. The remaining variables must
// reach the SAME optimum as the hand-eliminated two-variable problem
// min (y0-1)^2 + (y1-3)^2 s.t. y0 + y1 = 6, and the pinned value must appear
// exactly in the returned full-space solution even though the initial guess
// disagreed with it.
TEST(NlpVarBoundsReduction, FixedVariableQpMatchesHandEliminatedProblem) {
    // Hand-eliminated reference, solved and validated first.
    tycho::solvers::OptimizationProblem reference_prob;
    nlp_var_bounds_add_separable_objective(reference_prob, {1.0, 3.0});
    nlp_var_bounds_add_pair_sum_con(reference_prob, 0, 1, 6.0);
    auto reference_nlp = nlp_var_bounds_transcribe(reference_prob, 2);

    auto reference = nlp_var_bounds_solve(reference_nlp, Eigen::VectorXd::Zero(2));
    ASSERT_EQ(reference.flag_, tycho::ConvergenceFlags::CONVERGED)
        << "the hand-eliminated reference problem must solve before it can be compared against";
    ASSERT_NEAR(reference.solution_[0], 2.0, 1e-6);
    ASSERT_NEAR(reference.solution_[1], 4.0, 1e-6);

    // Full problem with the middle variable fixed.
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0, 3.0});
    nlp_var_bounds_add_pair_sum_con(prob, 0, 2, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    Eigen::VectorXd guess(3);
    guess << 0.0, -5.0, 0.0; // deliberately disagrees with the fixed value
    auto outcome = nlp_var_bounds_solve(nlp, guess);

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 2);
    // One variable eliminated: the factorized system is exactly one row and
    // column narrower than the declared problem's.
    EXPECT_EQ(nlp->kkt_dim_, (nlp->primal_vars_ - 1) + nlp->equal_cons_);
    // ... and the solution still comes back in the caller's own space.
    EXPECT_EQ(outcome.solution_.size(), nlp->primal_vars_);

    // The pinned value is exact, not merely converged to.
    EXPECT_DOUBLE_EQ(outcome.solution_[1], 7.0);
    EXPECT_NEAR(outcome.solution_[0], reference.solution_[0], 1e-6);
    EXPECT_NEAR(outcome.solution_[2], reference.solution_[1], 1e-6);
}

// The eliminated variable is coupled to a free one through a genuine objective
// HESSIAN cross-term, so the elimination is load-bearing rather than incidental:
//
//   min (x0 + x1)^2 + (x0 - 1)^2 + (x2 - 1)^2   s.t. x2 = 1,   x1 fixed at 7
//
// The full Hessian carries d2f/dx0dx1 = 2. Eliminating x1 leaves
// min (y0 + 7)^2 + (y0 - 1)^2, whose optimum is y0 = -3. If the cross-term
// survived into the solved system, the eliminated coordinate's Newton step
// would be -dx0 rather than 0 -- x1 would drift off 7 AND x0 would land
// somewhere else -- so both assertions below fail loudly on a broken
// elimination. (Under a separable objective, by contrast, the cross-terms are
// numerically zero and the elimination cannot be observed at all.)
TEST(NlpVarBoundsReduction, FixedVariableCoupledThroughObjectiveHessian) {
    // Hand-eliminated reference: min (y0+7)^2 + (y0-1)^2 + (y1-1)^2 s.t. y1 = 1.
    tycho::solvers::OptimizationProblem reference_prob;
    nlp_var_bounds_add_shifted_square(reference_prob, 0, -7.0);
    nlp_var_bounds_add_shifted_square(reference_prob, 0, 1.0);
    nlp_var_bounds_add_shifted_square(reference_prob, 1, 1.0);
    nlp_var_bounds_add_single_sum_con(reference_prob, 1, 1.0);
    auto reference_nlp = nlp_var_bounds_transcribe(reference_prob, 2);

    auto reference = nlp_var_bounds_solve(reference_nlp, Eigen::VectorXd::Zero(2));
    ASSERT_EQ(reference.flag_, tycho::ConvergenceFlags::CONVERGED)
        << "the hand-eliminated reference problem must solve before it can be compared against";
    ASSERT_NEAR(reference.solution_[0], -3.0, 1e-6);
    ASSERT_NEAR(reference.solution_[1], 1.0, 1e-6);

    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_pair_square(prob, 0, 1);
    nlp_var_bounds_add_shifted_square(prob, 0, 1.0);
    nlp_var_bounds_add_shifted_square(prob, 2, 1.0);
    nlp_var_bounds_add_single_sum_con(prob, 2, 1.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 2);
    EXPECT_EQ(nlp->kkt_dim_, (nlp->primal_vars_ - 1) + nlp->equal_cons_);
    EXPECT_EQ(outcome.solution_.size(), nlp->primal_vars_);

    EXPECT_DOUBLE_EQ(outcome.solution_[1], 7.0);
    EXPECT_NEAR(outcome.solution_[0], reference.solution_[0], 1e-6);
    EXPECT_NEAR(outcome.solution_[2], reference.solution_[1], 1e-6);
    EXPECT_NEAR(outcome.solution_[0], -3.0, 1e-6);
}

// The eliminated variable sits inside a constraint's JACOBIAN row:
//
//   min sum (x_i - c_i)^2, c = (1, 2, 3)   s.t. x0 + x1 = 3,   x1 fixed at 7
//
// Two independent things have to happen. The Jacobian entry in the eliminated
// column must leave the solved system (otherwise the eliminated coordinate
// picks up a step through the constraint row), and the pinned value must still
// reach the constraint RESIDUAL through the full-space evaluation (otherwise x0
// solves x0 = 3 instead of x0 = 3 - 7 = -4). The two failure modes are
// distinguishable in the assertions: the first moves x1, the second moves x0 to
// 3.
TEST(NlpVarBoundsReduction, FixedVariableInsideConstraintJacobian) {
    // Hand-eliminated reference: min (y0-1)^2 + (y1-3)^2 s.t. y0 = -4.
    tycho::solvers::OptimizationProblem reference_prob;
    nlp_var_bounds_add_separable_objective(reference_prob, {1.0, 3.0});
    nlp_var_bounds_add_single_sum_con(reference_prob, 0, -4.0);
    auto reference_nlp = nlp_var_bounds_transcribe(reference_prob, 2);

    auto reference = nlp_var_bounds_solve(reference_nlp, Eigen::VectorXd::Zero(2));
    ASSERT_EQ(reference.flag_, tycho::ConvergenceFlags::CONVERGED)
        << "the hand-eliminated reference problem must solve before it can be compared against";
    ASSERT_NEAR(reference.solution_[0], -4.0, 1e-6);
    ASSERT_NEAR(reference.solution_[1], 3.0, 1e-6);

    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0, 3.0});
    nlp_var_bounds_add_pair_sum_con(prob, 0, 1, 3.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 2);
    EXPECT_EQ(nlp->kkt_dim_, (nlp->primal_vars_ - 1) + nlp->equal_cons_);
    EXPECT_EQ(outcome.solution_.size(), nlp->primal_vars_);

    EXPECT_DOUBLE_EQ(outcome.solution_[1], 7.0);
    EXPECT_NEAR(outcome.solution_[0], reference.solution_[0], 1e-6);
    EXPECT_NEAR(outcome.solution_[2], reference.solution_[1], 1e-6);
    EXPECT_NEAR(outcome.solution_[0], -4.0, 1e-6);
}

// Two fixed variables interleaved with three free ones, with the constraint
// spanning only the free ones: index bookkeeping end to end.
// min sum (x_i - c_i)^2, c = (1..5), s.t. x0 + x2 + x4 = 6, x1 = -2, x3 = 10.
// On the free block x_i = c_i - l/2 with 9 - 3l/2 = 6, so l = 2 and the free
// optimum is (0, 2, 4).
TEST(NlpVarBoundsReduction, TwoInterleavedFixedVariablesSolve) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0, 3.0, 4.0, 5.0});
    nlp_var_bounds_add_triple_sum_con(prob, 0, 2, 4, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 5);
    nlp->set_variable_bound(1, -2.0, -2.0);
    nlp->set_variable_bound(3, 10.0, 10.0);
    nlp_var_bounds_rebuild(*nlp);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(5));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 3);
    // Two eliminated: two rows and columns fewer than the declared problem's.
    EXPECT_EQ(nlp->kkt_dim_, (nlp->primal_vars_ - 2) + nlp->equal_cons_);
    EXPECT_EQ(outcome.solution_.size(), nlp->primal_vars_);

    EXPECT_DOUBLE_EQ(outcome.solution_[1], -2.0);
    EXPECT_DOUBLE_EQ(outcome.solution_[3], 10.0);
    EXPECT_NEAR(outcome.solution_[0], 0.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[2], 2.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[4], 4.0, 1e-6);
}

// One solver instance, three solves, different bounds in between: each solve
// must re-classify rather than reuse the previous solve's elimination.
TEST(NlpVarBoundsReduction, SecondSolveAfterBoundChangeUsesTheNewValue) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0, 3.0});
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
    EXPECT_EQ(nlp->kkt_dim_, (nlp->primal_vars_ - 1) + nlp->equal_cons_);
    EXPECT_DOUBLE_EQ(second[1], 9.0);
    EXPECT_NEAR(second[0], 2.0, 1e-6);
    EXPECT_NEAR(second[2], 4.0, 1e-6);

    // And dropping the bound entirely returns the problem to the identity path.
    nlp->clear_variable_bounds();
    nlp_var_bounds_rebuild(*nlp);
    opt.set_nlp(nlp);
    Eigen::VectorXd third = opt.optimize(Eigen::VectorXd::Zero(3));
    EXPECT_EQ(opt.result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_FALSE(nlp->is_reduced());
    // Back to the full layout, and the solver re-analyzed for it.
    EXPECT_EQ(nlp->kkt_dim_, nlp->primal_vars_ + nlp->equal_cons_);
    EXPECT_NEAR(third[1], 2.0, 1e-6);
}
