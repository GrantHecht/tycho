///////////////////////////////////////////////////////////////////////////////
// NLP variable-bound contract: NonLinearProgram::set_variable_bound,
// clear_variable_bounds, has_variable_bounds, and the x_lower_/x_upper_
// vectors make_nlp materializes from the staged declarations.
//
// The first group hand-builds a NonLinearProgram directly (no InteriorPointSolver, no
// Phase/transcription) with empty objective/constraint lists, so make_nlp
// only has to run its own bookkeeping over primal variables.
//
// The second group covers the fixed-variable treatment: classification of the
// materialized bounds, the full<->reduced index maps, and end-to-end solves of
// hand-built QPs where a variable is pinned by equal bounds. Those solves drive
// a InteriorPointSolver instance directly against the NLP an OptimizationProblem
// transcribed, so the bounds can be declared on the NLP between transcription
// and the solve.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/hven_namespaces.h"
#include <hven/drivers/non_linear_program.h>
#include "tycho/detail/solvers_vf/optimization_problem.h"

#include <gtest/gtest.h>

#include <cassert>
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
// Most solves below are built from SHIFTED squares rather than a plain ||x||^2,
// and started away from the minimizer, so that neither the objective value nor
// its gradient is identically zero at the initial point -- which keeps them about
// the bound classification and the reduction rather than about what the solver
// does on a stationary start.
//
// The stationary start is not itself a hazard any more. It used to be, and not for
// a barrier or merit reason: squared_norm formed its derivative coefficient as
// 2 n^2 / n^2, which is 0/0 at the centre of the norm, so a variable sitting
// exactly on its own objective centre put a NaN on its KKT diagonal.
// ObjectiveStartedAtItsOwnCentreConverges pins the fixed behaviour.
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

// Adds the inequality constraint x[i0] <= target. InteriorPointSolver's inequalities are
// g(x) <= 0, slack-completed as g(x) + s == 0 with s >= 0.
void nlp_var_bounds_add_upper_iq_con(tycho::solvers::OptimizationProblem &prob, int i0,
                                     double target) {
    auto args = tycho::vf::Arguments<1>();
    auto con = args.coeff<0>() - target;
    prob.add_inequal_con(tycho::vf::GenericFunction<-1, -1>(con),
                         (Eigen::VectorXi(1) << i0).finished());
}

// A quadratic whose Hessian is genuinely block sparse: the third argument does
// not couple to the first two. It declares that structure through
// hessian_elem_is_nonzero and honors it in add_hessian_elem -- the pairing every
// sparsity-aware function in the tree has to keep, and the only case where the
// fill cursor's advance predicate is not a compile-time tautology. Written the
// way the transcription defects write it, so elimination meets the same shape
// here that it meets in the motivating workload.
//
// f(u) = (u0 + u1)^2 + (u0 - 1)^2 + (u2 - 1)^2
struct NlpVarBoundsSparseHessianQuad
    : tycho::vf::VectorFunction<NlpVarBoundsSparseHessianQuad, 3, 1,
                                tycho::vf::DenseDerivativeMode::FDiffFwd,
                                tycho::vf::DenseDerivativeMode::FDiffFwd> {
    using Base = tycho::vf::VectorFunction<NlpVarBoundsSparseHessianQuad, 3, 1,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    // The two blocks are {0, 1} and {2}: an element couples only within a block.
    static bool couples(int row, int col) { return (row < 2) == (col < 2); }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx[0] = (x[0] + x[1]) * (x[0] + x[1]) + (x[0] - 1.0) * (x[0] - 1.0) +
                (x[2] - 1.0) * (x[2] - 1.0);
    }

    inline bool hessian_elem_is_nonzero(int row, int col) const { return couples(row, col); }

    inline void add_hessian_elem(double v, int row, int col, double *mpt, const int *lpt,
                                 int &freeloc) const {
        if (couples(row, col)) {
            // Same slot-validity tripwire the base and the transcription defects
            // carry: this override is exactly where the cursor contract is a
            // runtime predicate and can drift, which is what this test exists to
            // catch. Debug-only; compiled out under NDEBUG.
            assert(lpt[freeloc] >= 0 &&
                   "KKT Hessian fill cursor landed on an eliminated element's slot");
            mpt[lpt[freeloc]] += v;
            freeloc++;
        }
    }
};

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
// their own InteriorPointSolver against it.
std::shared_ptr<NonLinearProgram>
nlp_var_bounds_transcribe(tycho::solvers::OptimizationProblem &prob, int num_vars) {
    prob.set_vars(Eigen::VectorXd::Zero(num_vars));
    prob.transcribe();
    return prob.nlp_;
}

// Re-materializes the bounds staged on an already-transcribed NLP (make_nlp
// reallocates every KKT array, so the solver has to be re-pointed at it, which
// is what recomputes the storage locations the reduction addresses).
//
// user_equal_cons_, not equal_cons_: the latter counts the internal fixing rows
// the MakeConstraint treatment installs, and make_nlp takes the user's own row
// count. The two are the same number under every other treatment.
void nlp_var_bounds_rebuild(NonLinearProgram &nlp) {
    nlp.make_nlp(nlp.primal_vars_, nlp.user_equal_cons_, nlp.inequal_cons_);
}

struct NlpVarBoundsSolveOutcome {
    Eigen::VectorXd solution_;
    tycho::ConvergenceFlags flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
};

// Drives a fresh silent InteriorPointSolver against an NLP. Returns the flag alongside the
// solution rather than asserting internally, so every caller can make the
// convergence check a hard precondition (a reference problem that failed to
// solve must never be able to masquerade as agreement).
NlpVarBoundsSolveOutcome nlp_var_bounds_solve(const std::shared_ptr<NonLinearProgram> &nlp,
                                              const Eigen::VectorXd &guess) {
    tycho::solvers::InteriorPointSolver opt;
    opt.set_print_level(3);
    opt.set_nlp(nlp);
    NlpVarBoundsSolveOutcome outcome;
    outcome.solution_ = opt.optimize(guess);
    outcome.flag_ = opt.result().converge_flag_;
    return outcome;
}

// The same, under a chosen fixed-variable treatment. The settings go on before
// set_nlp so that the analysis set_nlp performs is the one the solve's own
// classification will confirm rather than replace.
NlpVarBoundsSolveOutcome nlp_var_bounds_solve_under(const std::shared_ptr<NonLinearProgram> &nlp,
                                                    const Eigen::VectorXd &guess,
                                                    FixedVariableTreatments treatment,
                                                    double relax_factor) {
    tycho::solvers::InteriorPointSolver opt;
    opt.set_print_level(3);
    opt.set_fixed_variable_treatment(treatment);
    opt.set_bound_relax_factor(relax_factor);
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

// --- The treatments that keep the variable ---------------------------------

// MakeConstraint hands the fixed variable to an internal equality row instead of
// eliminating it: the variable keeps its column (so a bound declared on a LATER
// variable is not renumbered), carries no bound of its own, and the row space
// grows by one row per fixed variable -- appended after every row the caller
// declared, which is what keeps the user's own row indices theirs.
TEST(NlpVarBoundsReduction, MakeConstraintAppendsOneEqualityRowPerFixedVariable) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(1, 2.0, 2.0);   // fixed
    problem.nlp_.set_variable_bound(3, -4.0, -4.0); // fixed
    problem.nlp_.set_variable_bound(4, 0.0, 8.0);   // ordinary two-sided
    problem.build(5);
    ASSERT_EQ(problem.nlp_.kkt_dim_, 5);
    ASSERT_EQ(problem.nlp_.user_equal_cons_, 0);

    EXPECT_TRUE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeConstraint, 1.0e-8));

    // Nothing eliminated: the primal block is the declared one, and the maps the
    // reduction would need stay empty.
    EXPECT_FALSE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 5);
    EXPECT_EQ(problem.nlp_.full_to_reduced().size(), 0);
    EXPECT_EQ(problem.nlp_.reduced_to_full().size(), 0);

    // Two internal rows, on top of the caller's own count, and the system is
    // wider by exactly that many.
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 2);
    EXPECT_EQ(problem.nlp_.equal_cons_, problem.nlp_.user_equal_cons_ + 2);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 5 + 2);

    // The fixed variables are reported, and carry no barrier bound: the rows hold
    // them. The one genuinely bounded variable is the only entry in the set, at
    // index 4 -- the index it was declared on, because no column was dropped
    // ahead of it.
    ASSERT_EQ(problem.nlp_.fixed_variable_indices().size(), 2);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[0], 1);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[1], 3);
    EXPECT_DOUBLE_EQ(problem.nlp_.fixed_variable_values()[0], 2.0);
    EXPECT_DOUBLE_EQ(problem.nlp_.fixed_variable_values()[1], -4.0);
    const auto &bounds = problem.nlp_.variable_bound_set();
    ASSERT_EQ(bounds.lower_idx_.size(), 1);
    ASSERT_EQ(bounds.upper_idx_.size(), 1);
    EXPECT_EQ(bounds.lower_idx_[0], 4);
    EXPECT_EQ(bounds.upper_idx_[0], 4);

    // A repeat call is still idempotent, and does not append a second copy.
    EXPECT_FALSE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeConstraint, 1.0e-8));
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 2);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 5 + 2);
}

// RelaxBounds leaves the layout alone entirely and records the fixed variable as
// an ordinary two-sided bound whose endpoints have been pushed apart. The widening
// is applied ONCE, by the same shared path that widens every other finite bound --
// there is no separate separation step, which is also what the reference does (its
// adapter enters a relaxed variable in both bound maps at the declared value and
// does not count it among the fixed ones, so the only widening it ever gets is the
// universal relax factor).
TEST(NlpVarBoundsReduction, RelaxBoundsRecordsATwoSidedPairAroundTheFixedValue) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(1, 4.0, 4.0);
    problem.build(3);

    const double factor = 1.0e-3;
    // Nothing structural changed, so there is nothing for the caller to re-read.
    EXPECT_FALSE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::RelaxBounds, factor));
    EXPECT_FALSE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 3);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 3);
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 0);

    const auto &bounds = problem.nlp_.variable_bound_set();
    ASSERT_EQ(bounds.lower_idx_.size(), 1);
    ASSERT_EQ(bounds.upper_idx_.size(), 1);
    EXPECT_EQ(bounds.lower_idx_[0], 1);
    EXPECT_EQ(bounds.upper_idx_[0], 1);

    // One application of the widening, and it is the same helper every other
    // bound's expected value in this file is computed with.
    EXPECT_DOUBLE_EQ(bounds.lower_val_[0], nlp_var_bounds_relax(4.0, factor, false));
    EXPECT_DOUBLE_EQ(bounds.upper_val_[0], nlp_var_bounds_relax(4.0, factor, true));
    EXPECT_LT(bounds.lower_val_[0], 4.0);
    EXPECT_GT(bounds.upper_val_[0], 4.0);
    // The box is 2 * factor * max(1, |value|) wide -- the width every consumer of
    // this treatment has to plan around. Compared with a tolerance rather than
    // exactly: the two endpoints are each rounded once, so their difference is not
    // required to reproduce the product bit for bit (it does not, at this factor).
    EXPECT_NEAR(bounds.upper_val_[0] - bounds.lower_val_[0],
                2.0 * factor * std::max(1.0, std::abs(4.0)), 1.0e-15);

    // Two-sided, so neither entry carries the one-sided damping indicator.
    EXPECT_DOUBLE_EQ(bounds.lower_damp_[0], 0.0);
    EXPECT_DOUBLE_EQ(bounds.upper_damp_[0], 0.0);

    // And it is still reported as a fixed variable, which is what the treatment
    // acted on.
    ASSERT_EQ(problem.nlp_.fixed_variable_indices().size(), 1);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[0], 1);
}

// One NLP, every treatment in turn. Each configuration must derive its own answer
// from the pristine state rather than compound the previous one -- which for
// MakeConstraint means its rows are gone the moment another treatment takes over,
// and for MakeParameter means the elimination is gone again when it does not.
TEST(NlpVarBoundsReduction, SwitchingTreatmentReclassifiesFromScratch) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(2, 4.0, 4.0);
    problem.build(4);

    EXPECT_TRUE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8));
    EXPECT_TRUE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.kkt_dim_, 3);
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 0);

    EXPECT_TRUE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeConstraint, 1.0e-8));
    EXPECT_FALSE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 4);
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 1);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 5);

    // Back to a treatment that adds no row: the row space returns to the
    // caller's own, and the system with it.
    EXPECT_TRUE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::RelaxBounds, 1.0e-8));
    EXPECT_FALSE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 0);
    EXPECT_EQ(problem.nlp_.equal_cons_, problem.nlp_.user_equal_cons_);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 4);
    EXPECT_TRUE(problem.nlp_.variable_bound_set().any());

    // And back to the elimination, which must reach exactly the state it reached
    // the first time.
    EXPECT_TRUE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8));
    EXPECT_TRUE(problem.nlp_.is_reduced());
    EXPECT_EQ(problem.nlp_.reduced_primal_vars(), 3);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 3);
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 0);
    EXPECT_FALSE(problem.nlp_.variable_bound_set().any());
    ASSERT_EQ(problem.nlp_.fixed_variable_indices().size(), 1);
    EXPECT_EQ(problem.nlp_.fixed_variable_indices()[0], 2);
}

// A problem all of whose variables are fixed has nothing left to solve for only
// under the treatment that eliminates them. The other two keep every variable, so
// the same declaration is an ordinary square system of internal rows in one case
// and a fully boxed problem in the other.
TEST(NlpVarBoundsReduction, AnAllFixedProblemIsRejectedOnlyByTheElimination) {
    NlpVarBoundsBareNlp problem;
    problem.nlp_.set_variable_bound(0, 1.0, 1.0);
    problem.nlp_.set_variable_bound(1, 2.0, 2.0);
    problem.build(2);

    EXPECT_THROW(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8),
        std::invalid_argument);

    EXPECT_TRUE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::MakeConstraint, 1.0e-8));
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 2);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 4);

    EXPECT_TRUE(
        problem.nlp_.configure_variable_treatment(FixedVariableTreatments::RelaxBounds, 1.0e-8));
    EXPECT_EQ(problem.nlp_.internal_fixed_constraints(), 0);
    EXPECT_EQ(problem.nlp_.kkt_dim_, 2);
    EXPECT_EQ(problem.nlp_.variable_bound_set().lower_idx_.size(), 2);
}

// --- Rejected configurations ----------------------------------------------

// The relaxation is the whole mechanism by which RelaxBounds gives the variable
// somewhere to sit, so a zero factor is not a conservative choice -- it leaves an
// interval of zero width, which the interior push has no interior to land in and
// the barrier divides by. Rejected at configuration rather than as a non-finite
// objective a few evaluations later. With nothing fixed there is no pair to
// separate and the same factor is fine.
TEST(NlpVarBoundsReduction, RelaxBoundsRejectsAZeroRelaxFactorOnAFixedVariable) {
    NlpVarBoundsBareNlp fixed_problem;
    fixed_problem.nlp_.set_variable_bound(0, 3.0, 3.0);
    fixed_problem.build(2);

    try {
        fixed_problem.nlp_.configure_variable_treatment(FixedVariableTreatments::RelaxBounds, 0.0);
        FAIL() << "expected std::invalid_argument for a zero relax factor on a fixed variable";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("relax_bounds"), std::string::npos) << msg;
        EXPECT_NE(msg.find("make_parameter"), std::string::npos) << msg;
    }
    EXPECT_FALSE(fixed_problem.nlp_.variable_bound_set().any());

    NlpVarBoundsBareNlp bounded_problem;
    bounded_problem.nlp_.set_variable_bound(0, -1.0, 1.0);
    bounded_problem.build(2);
    EXPECT_NO_THROW(bounded_problem.nlp_.configure_variable_treatment(
        FixedVariableTreatments::RelaxBounds, 0.0));
    EXPECT_TRUE(bounded_problem.nlp_.variable_bound_set().any());
}

// A rejected configuration arriving at an NLP that has internal fixing rows
// installed must leave neither the rows nor the layout they were laid out for
// behind. This is the MakeConstraint half of the restore: the rows are dropped
// before the new classification runs, so a classification that then rejects has
// to have re-derived the element counts, the work partitioning and the layout
// over the caller's own row space -- and the rejected configuration must not be
// remembered as done.
TEST(NlpVarBoundsReduction, ARejectedTreatmentSwitchLeavesNoInternalRowBehind) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0, 3.0});
    nlp_var_bounds_add_pair_sum_con(prob, 0, 2, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    const int user_eq = nlp->user_equal_cons_;
    ASSERT_GT(user_eq, 0);
    ASSERT_TRUE(nlp->configure_variable_treatment(FixedVariableTreatments::MakeConstraint, 1.0e-8));
    ASSERT_EQ(nlp->internal_fixed_constraints(), 1);
    ASSERT_EQ(nlp->kkt_dim_, nlp->primal_vars_ + user_eq + 1);

    // The switch this NLP cannot honor: RelaxBounds with nothing to separate the
    // fixed pair by.
    EXPECT_THROW(nlp->configure_variable_treatment(FixedVariableTreatments::RelaxBounds, 0.0),
                 std::invalid_argument);

    EXPECT_EQ(nlp->internal_fixed_constraints(), 0);
    EXPECT_EQ(nlp->equal_cons_, user_eq);
    EXPECT_FALSE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), nlp->primal_vars_);
    EXPECT_EQ(nlp->kkt_dim_, nlp->primal_vars_ + user_eq);

    // Not remembered as done: the same call fails the same way instead of
    // reporting "no change". This is the sharpest form of that requirement --
    // unlike every other rejection in this suite, this one arrives at an NLP a
    // previous call had SUCCESSFULLY configured, and the treatment and factor it
    // was called with are already recorded. If the restore left the validity stamp
    // standing, this second call would match the idempotence shortcut exactly and
    // report "no change" on a problem the restore has just put back to
    // unconfigured -- and a solve would then run with the fixing bound ignored.
    EXPECT_THROW(nlp->configure_variable_treatment(FixedVariableTreatments::RelaxBounds, 0.0),
                 std::invalid_argument);

    // ... and the next VALID configuration is a real one, not a shortcut: it
    // reports that it rebuilt the structures, which a stamp left standing would
    // have suppressed.
    EXPECT_TRUE(nlp->configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8));
    EXPECT_TRUE(nlp->is_reduced());

    // And the NLP is still solvable, on the layout the restore left: the
    // elimination reaches the same answer as the identity-path problem's, with the
    // pinned value exact.
    // min sum (x_i - c_i)^2, c = (1, 2, 3), s.t. x0 + x2 = 6, x1 fixed at 7.
    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));
    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_DOUBLE_EQ(outcome.solution_[1], 7.0);
    EXPECT_NEAR(outcome.solution_[0], 2.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[2], 4.0, 1e-6);
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
//
// The two degenerate-start solves that open the section are the deliberate
// exception: they start ON the objective's centre, which is the case
// squared_norm's 0/0 derivative coefficient used to turn into a DIVERGING verdict.

// min sum_i x_i^2 started at the origin -- which is the minimizer, so the solve is
// over as soon as the convergence test is applied, and every coordinate sits on
// its own objective centre when it is.
TEST(NlpVarBoundsReduction, ObjectiveStartedAtItsOwnCentreConverges) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {0.0, 0.0, 0.0});
    auto nlp = nlp_var_bounds_transcribe(prob, 3);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(outcome.solution_[i], 0.0, 1e-9) << "coordinate " << i;
    }
}

// The same degeneracy with a step still required, which is the harder half: two
// coordinates have to move to satisfy x0 + x1 = 4, so the iteration cannot finish
// before a Newton solve, and the third coordinate is on its own objective centre
// at every iterate of it. The NaN this used to put on that coordinate's KKT
// diagonal was invisible in the residual table -- an infinity-norm reduction drops
// it -- and surfaced only as a non-finite step and a DIVERGING verdict.
TEST(NlpVarBoundsReduction, ObjectiveAtItsOwnCentreConvergesWithAStepStillRequired) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {0.0, 0.0, 0.0});
    nlp_var_bounds_add_pair_sum_con(prob, 0, 1, 4.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    // Closed form: the row splits evenly between the two coordinates it couples,
    // and the third is already at its minimum and must stay there.
    EXPECT_NEAR(outcome.solution_[0], 2.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[1], 2.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[2], 0.0, 1e-9);
}

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

// Same coupled problem as above, but the objective comes from a function that
// declares a SPARSE Hessian: it tells the fill machinery that its third argument
// does not couple to the first two, and honors that in its own element scatter.
// That is the only case in which the fill cursor's advance predicate is not a
// compile-time tautology -- and it is the shape the motivating workload always
// produces, since the transcription defects declare their Hessian sparsity the
// same way. If the reduced fill arm's step-over disagreed with the claim-side
// predicate by even one element, the cursor would drift and every element after
// it would be written to the wrong slot (or, on an eliminated element, through a
// -1 location). The solve either matches the hand-eliminated reference or it
// does not.
TEST(NlpVarBoundsReduction, SparseHessianFunctionUnderElimination) {
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
    prob.add_objective(tycho::vf::GenericFunction<-1, 1>(NlpVarBoundsSparseHessianQuad()),
                       (Eigen::VectorXi(3) << 0, 1, 2).finished());
    nlp_var_bounds_add_single_sum_con(prob, 2, 1.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 2);
    EXPECT_EQ(nlp->kkt_dim_, (nlp->primal_vars_ - 1) + nlp->equal_cons_);

    EXPECT_DOUBLE_EQ(outcome.solution_[1], 7.0);
    // Derivatives here are finite-difference, so this is compared a little
    // looser than the analytic problems above.
    EXPECT_NEAR(outcome.solution_[0], reference.solution_[0], 1e-5);
    EXPECT_NEAR(outcome.solution_[2], reference.solution_[1], 1e-5);
}

// Elimination alongside an ACTIVE inequality, which is what puts the slack and
// barrier block behind the narrowed primal width: the slack Jacobian and slack
// diagonal that finalize_data lays down, get_mat_space's inequality row offset,
// and every segment(primal_vars_, slack_vars_) in the solver all have to agree
// on the reduced width at once.
//
//   min sum (x_i - c_i)^2, c = (1, 2, 3)   s.t. x0 + x1 = 3, x2 <= 1,
//                                               x1 fixed at 7
//
// The equality forces x0 = 3 - 7 = -4 (which needs the pinned value to reach the
// residual), and x2's unconstrained optimum of 3 is cut off by the inequality,
// so the inequality is active at the solution and its slack goes to zero.
TEST(NlpVarBoundsReduction, FixedVariableWithActiveInequality) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0, 3.0});
    nlp_var_bounds_add_pair_sum_con(prob, 0, 1, 3.0);
    nlp_var_bounds_add_upper_iq_con(prob, 2, 1.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 3);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    ASSERT_EQ(nlp->inequal_cons_, 1);
    ASSERT_EQ(nlp->slack_vars_, 1);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(3));

    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 2);
    // The slack, equality and inequality blocks all sit above the narrowed
    // primal block.
    EXPECT_EQ(nlp->kkt_dim_,
              (nlp->primal_vars_ - 1) + nlp->slack_vars_ + nlp->equal_cons_ + nlp->inequal_cons_);
    EXPECT_EQ(outcome.solution_.size(), nlp->primal_vars_);

    EXPECT_DOUBLE_EQ(outcome.solution_[1], 7.0);
    EXPECT_NEAR(outcome.solution_[0], -4.0, 1e-6);
    EXPECT_NEAR(outcome.solution_[2], 1.0, 1e-5); // approached from below through the barrier
}

// A configuration that throws must not be remembered as done. If it were, the
// next call would take the idempotence shortcut, report that nothing changed,
// and the solve would quietly proceed on the UNREDUCED problem with every
// fixing bound ignored -- the worst possible failure for this feature, because
// it produces a plausible answer to the wrong problem.
TEST(NlpVarBoundsReduction, AThrownConfigurationIsNotCommitted) {
    tycho::solvers::OptimizationProblem prob;
    nlp_var_bounds_add_separable_objective(prob, {1.0, 2.0});
    nlp_var_bounds_add_pair_sum_con(prob, 0, 1, 6.0);
    auto nlp = nlp_var_bounds_transcribe(prob, 2);

    // Both variables fixed: nothing would be left to solve for.
    nlp->set_variable_bound(0, 4.0, 4.0);
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    EXPECT_THROW(nlp->configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8),
                 std::invalid_argument);

    // Nothing was committed, and the NLP is still coherent: it describes the
    // full problem rather than a half-reduced one.
    EXPECT_FALSE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 2);
    EXPECT_EQ(nlp->kkt_dim_, 2 + nlp->equal_cons_);

    // Above all it does not consider itself configured: repeating the call
    // re-attempts and fails the same way instead of returning "no change".
    EXPECT_THROW(nlp->configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8),
                 std::invalid_argument);

    // ... which is what stops a solve from silently running the unreduced
    // problem: the solve re-attempts the configuration and fails loudly.
    {
        tycho::solvers::InteriorPointSolver opt;
        opt.set_print_level(3);
        opt.set_nlp(nlp);
        EXPECT_THROW(opt.optimize(Eigen::VectorXd::Zero(2)), std::invalid_argument);
    }

    // Correcting the bounds and configuring again reduces properly and solves:
    // min (y-1)^2 + (7-2)^2 s.t. y + 7 = 6, so the free variable is -1.
    nlp->clear_variable_bounds();
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    auto outcome = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(2));
    EXPECT_EQ(outcome.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 1);
    EXPECT_EQ(nlp->kkt_dim_, 1 + nlp->equal_cons_);
    EXPECT_DOUBLE_EQ(outcome.solution_[1], 7.0);
    EXPECT_NEAR(outcome.solution_[0], -1.0, 1e-6);

    // Second phase: the same rejection arriving at an NLP that is ALREADY
    // reduced. This is the sharper case. A rejection that escaped the restore
    // here would leave is_reduced() true and the old reduced width in place
    // while the maps behind them had been rewritten for the rejected
    // classification -- so the next expansion of a primal vector would index off
    // the end of a shorter map, silently, since Eigen's bounds checks are
    // compiled out in release builds.
    ASSERT_TRUE(nlp->is_reduced());
    ASSERT_EQ(nlp->reduced_primal_vars(), 1);

    // No public route reaches this state: make_nlp is the only thing that
    // re-materializes bounds, and it also resets the reduction. The failing
    // input is therefore set up directly on the materialized vectors and the
    // revision counter that drives re-classification. What is under test is the
    // restore, not the route in.
    nlp->x_lower_[0] = 4.0;
    nlp->x_upper_[0] = 4.0;
    nlp->bounds_revision_++;

    EXPECT_THROW(nlp->configure_variable_treatment(FixedVariableTreatments::MakeParameter, 1.0e-8),
                 std::invalid_argument);

    // Restored to a coherent pass-through state, not left half-reduced.
    EXPECT_FALSE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), nlp->primal_vars_);
    EXPECT_EQ(nlp->kkt_dim_, nlp->primal_vars_ + nlp->equal_cons_);
    EXPECT_EQ(nlp->full_to_reduced().size(), 0);
    EXPECT_EQ(nlp->reduced_to_full().size(), 0);

    // Which is what makes the expansion helpers safe: they take their
    // pass-through branch and read no map at all.
    Eigen::VectorXd solver_space(nlp->primal_vars_);
    solver_space << 1.0, 2.0;
    Eigen::VectorXd caller_space(nlp->primal_vars_);
    caller_space.setZero();
    EXPECT_NO_THROW(nlp->scatter_full_x(solver_space, caller_space));
    EXPECT_DOUBLE_EQ(caller_space[0], 1.0);
    EXPECT_DOUBLE_EQ(caller_space[1], 2.0);

    // Still not marked configured, so a solve fails loudly rather than running
    // the unreduced problem.
    {
        tycho::solvers::InteriorPointSolver opt;
        opt.set_print_level(3);
        opt.set_nlp(nlp);
        EXPECT_THROW(opt.optimize(Eigen::VectorXd::Zero(2)), std::invalid_argument);
    }

    // And it is still recoverable: correcting the bounds reduces and solves again.
    nlp->clear_variable_bounds();
    nlp->set_variable_bound(1, 7.0, 7.0);
    nlp_var_bounds_rebuild(*nlp);

    auto recovered = nlp_var_bounds_solve(nlp, Eigen::VectorXd::Zero(2));
    EXPECT_EQ(recovered.flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(nlp->is_reduced());
    EXPECT_EQ(nlp->reduced_primal_vars(), 1);
    EXPECT_DOUBLE_EQ(recovered.solution_[1], 7.0);
    EXPECT_NEAR(recovered.solution_[0], -1.0, 1e-6);
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

    tycho::solvers::InteriorPointSolver opt;
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
