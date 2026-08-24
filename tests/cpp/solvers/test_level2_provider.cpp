///////////////////////////////////////////////////////////////////////////////
// The transcribed problem as a Level 2 provider.
//
// A transcription declares its pieces, their thread modes, its variable bounds
// and the partition count it wants into one declaration, and the layout is a
// pure function of that declaration and the partition count actually adopted.
// This file pins the parts of that arrangement a consumer can see: what the
// declaration carries after a transcription, what the published claim stream
// looks like, and what the provider does with a request it cannot serve.
//
// Nothing here asserts an ANSWER. The equivalence suite next door already says
// the doors agree on the optimum and the multipliers; what is pinned here is
// structural, and structural facts are pinned exactly rather than to a
// tolerance.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers_vf/optimization_problem.h"
#include "tycho/detail/solvers_vf/transcribed_aggregate.h"

#include <hven/model/candidate_point.h>
#include <hven/model/claim_space.h>
#include <hven/model/non_linear_program.h>

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

using tycho::solvers::OptimizationProblem;
using tycho::solvers::TranscribedAggregate;
using tycho::vf::Arguments;
using tycho::vf::GenericFunction;

namespace {

constexpr double kL2ProviderInf = std::numeric_limits<double>::infinity();

/// One small problem, built the same way every time it is asked for.
///
///   min  (x0 - 1)^2 + (x1 - 2)^2 + (x2 - 3)^2
///   s.t. x0 + x1 + x2 - 3 = 0
///        x0 - x1 - 0.5   >= 0   (as the library's inequality sense)
///        x0 >= 0.25
///
/// Three variables, one equality row and one inequality row: enough for every
/// domain of the claim stream to be non-empty, and small enough that a
/// slot-by-slot assertion is readable.
void l2_provider_build(OptimizationProblem &prob) {
    prob.set_vars((Eigen::VectorXd(3) << 0.5, 1.5, 1.0).finished());

    const Eigen::VectorXi all3 = (Eigen::VectorXi(3) << 0, 1, 2).finished();
    {
        auto args = Arguments<3>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        auto x2 = args.coeff<2>();
        prob.add_objective(GenericFunction<-1, 1>((x0 - 1.0) * (x0 - 1.0) +
                                                  (x1 - 2.0) * (x1 - 2.0) +
                                                  (x2 - 3.0) * (x2 - 3.0)),
                           all3);
    }
    {
        auto args = Arguments<3>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        auto x2 = args.coeff<2>();
        prob.add_equal_con(GenericFunction<-1, -1>(x0 + x1 + x2 - 3.0), all3);
    }
    {
        auto args = Arguments<3>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_inequal_con(GenericFunction<-1, -1>(0.5 + x1 - x0), all3);
    }
    prob.add_variable_bound(0, 0.25, kL2ProviderInf);
}

/// The problem above, transcribed, with its provider on hand.
struct L2ProviderFixture {
    OptimizationProblem prob_;

    L2ProviderFixture() {
        l2_provider_build(prob_);
        prob_.transcribe();
    }

    TranscribedAggregate &provider() { return *prob_.provider_; }
};

} // namespace

///////////////////////////////////////////////////////////////////////////////
// What the declaration carries.
///////////////////////////////////////////////////////////////////////////////

// The transcription's dimensions, thread modes, bound records and partition
// count are all readable off the declaration afterwards, which is what makes
// the declaration -- rather than the sequence of calls that filled it -- the
// thing the layout is a function of.
TEST(Level2Provider, TheDeclarationCarriesWhatTheTranscriptionDeclared) {
    L2ProviderFixture fixture;
    const auto &declared = fixture.provider().declaration();

    EXPECT_EQ(declared.primal_vars_, 3);
    EXPECT_EQ(declared.equality_rows_, 1);
    EXPECT_EQ(declared.inequality_rows_, 1);
    EXPECT_EQ(declared.fixing_rows_, 0);
    EXPECT_EQ(declared.partition_count_, 1);

    ASSERT_EQ(declared.objectives_.size(), 1u);
    ASSERT_EQ(declared.equality_constraints_.size(), 1u);
    ASSERT_EQ(declared.inequality_constraints_.size(), 1u);

    // The policy the transcription picks: a function that is not thread safe
    // stays on the calling thread, and one that is goes round-robin where it has
    // a single application. Every piece here has one application, so the mode is
    // decided by the function's own answer -- read off the declared piece rather
    // than assumed, so this pins the rule and not a property of the fixture.
    const auto expected = [](bool thread_safe) {
        return thread_safe ? tycho::solvers::ThreadingFlags::RoundRobin
                           : tycho::solvers::ThreadingFlags::MainThread;
    };
    EXPECT_EQ(declared.objectives_[0].get_thread_mode(),
              expected(declared.objectives_[0].function_.thread_safe()));
    EXPECT_EQ(declared.equality_constraints_[0].get_thread_mode(),
              expected(declared.equality_constraints_[0].function_.thread_safe()));
    EXPECT_EQ(declared.inequality_constraints_[0].get_thread_mode(),
              expected(declared.inequality_constraints_[0].function_.thread_safe()));

    ASSERT_EQ(declared.variable_bounds_.size(), 1u);
    EXPECT_EQ(declared.variable_bounds_[0].index_, 0);
    EXPECT_EQ(declared.variable_bounds_[0].lower_, 0.25);
    EXPECT_EQ(declared.variable_bounds_[0].upper_, kL2ProviderInf);
}

// The thread mode is declaration data, and the layout freezes it: writing it on
// a laid piece is refused, and the message names the piece. Reaching a new mode
// means declaring it and laying again.
TEST(Level2Provider, AThreadModeCannotBeWrittenOntoALaidPiece) {
    L2ProviderFixture fixture;
    EXPECT_THROW(fixture.prob_.nlp_->equality_constraints_[0].set_thread_mode(
                     tycho::solvers::ThreadingFlags::MainThread),
                 std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// The claim stream.
///////////////////////////////////////////////////////////////////////////////

// The three domains occupy three contiguous slot ranges, in the published
// order, and together they are exactly the slots the layout claimed: no gap,
// no overlap, nothing outside them.
TEST(Level2Provider, TheClaimStreamIsThreeContiguousDomainRuns) {
    L2ProviderFixture fixture;
    TranscribedAggregate &provider = fixture.provider();

    const auto hessian = provider.hessian_claims();
    const auto equality = provider.equality_jacobian_claims();
    const auto inequality = provider.inequality_jacobian_claims();

    EXPECT_EQ(hessian.start_, 0);
    EXPECT_EQ(equality.start_, hessian.start_ + hessian.count_);
    EXPECT_EQ(inequality.start_, equality.start_ + equality.count_);

    EXPECT_GT(hessian.count_, 0);
    EXPECT_GT(equality.count_, 0);
    EXPECT_GT(inequality.count_, 0);

    const int total = inequality.start_ + inequality.count_;
    EXPECT_EQ(provider.kkt_claim_rows().size(), total);
    EXPECT_EQ(provider.kkt_claim_cols().size(), total);
}

// Each run's coordinates are the ones its domain names, in the assembled space
// the claim-stream contract fixes: the Hessian on the upper triangle of the
// primal block, an equality Jacobian row at n + r, an inequality Jacobian row
// at n + me + r, and every column a declared variable.
TEST(Level2Provider, EveryClaimNamesACoordinateOfItsOwnDomain) {
    L2ProviderFixture fixture;
    TranscribedAggregate &provider = fixture.provider();

    const auto &declared = provider.declaration();
    const int primal = declared.primal_vars_;
    const int equality_rows = declared.equality_rows_;
    const int inequality_rows = declared.inequality_rows_;
    EXPECT_EQ(provider.kkt_dimension(), primal + equality_rows + inequality_rows);

    const Eigen::VectorXi rows = provider.kkt_claim_rows();
    const Eigen::VectorXi cols = provider.kkt_claim_cols();

    const auto hessian = provider.hessian_claims();
    for (int slot = hessian.start_; slot < hessian.start_ + hessian.count_; slot++) {
        EXPECT_GE(rows[slot], 0);
        EXPECT_LT(rows[slot], primal);
        EXPECT_GE(cols[slot], rows[slot]);
        EXPECT_LT(cols[slot], primal);
    }

    const auto equality = provider.equality_jacobian_claims();
    for (int slot = equality.start_; slot < equality.start_ + equality.count_; slot++) {
        EXPECT_GE(rows[slot], primal);
        EXPECT_LT(rows[slot], primal + equality_rows);
        EXPECT_GE(cols[slot], 0);
        EXPECT_LT(cols[slot], primal);
    }

    const auto inequality = provider.inequality_jacobian_claims();
    for (int slot = inequality.start_; slot < inequality.start_ + inequality.count_; slot++) {
        EXPECT_GE(rows[slot], primal + equality_rows);
        EXPECT_LT(rows[slot], primal + equality_rows + inequality_rows);
        EXPECT_GE(cols[slot], 0);
        EXPECT_LT(cols[slot], primal);
    }
}

// Every claim slot belongs to exactly one partition, and the slots of one
// partition are contiguous within each domain run: the stream is issued
// serially, one partition at a time, in partition-index order.
//
// This is the disjointness that holds of a transcribed problem, and it is a
// property of SLOTS. It is deliberately not stated of COORDINATES: several
// pieces contributing to one matrix entry hold several slots naming it, which
// is how overlapping pieces compose, and the layout marks the contested columns
// so their writers serialize.
TEST(Level2Provider, EachClaimSlotBelongsToExactlyOnePartitionInIndexOrder) {
    L2ProviderFixture fixture;
    TranscribedAggregate &provider = fixture.provider();

    const int adopted = provider.declaration().partition_count_;
    const Eigen::VectorXi partitions = provider.kkt_claim_partitions();
    ASSERT_EQ(partitions.size(), provider.kkt_claim_rows().size());

    const std::vector<hven::solvers::ClaimBlock> runs = {provider.hessian_claims(),
                                                         provider.equality_jacobian_claims(),
                                                         provider.inequality_jacobian_claims()};
    for (const auto &run : runs) {
        int previous = -1;
        std::set<int> seen;
        for (int slot = run.start_; slot < run.start_ + run.count_; slot++) {
            const int partition = partitions[slot];
            EXPECT_GE(partition, 0);
            EXPECT_LT(partition, adopted);
            if (partition != previous) {
                // A partition's slots within one run are one contiguous stretch,
                // so a partition index is never revisited after it is left.
                EXPECT_EQ(seen.count(partition), 0u);
                seen.insert(partition);
                previous = partition;
            }
        }
    }
}

// The whole point of the declaration being a value: two constructions of the
// same problem lay the same layout. Same claim stream, slot for slot, same
// domain runs, and the same structural key.
TEST(Level2Provider, TwoConstructionsOfOneProblemLayTheSameStream) {
    L2ProviderFixture first;
    L2ProviderFixture second;

    TranscribedAggregate &a = first.provider();
    TranscribedAggregate &b = second.provider();

    EXPECT_EQ(a.model_structure_key(), b.model_structure_key());

    EXPECT_EQ(a.hessian_claims(), b.hessian_claims());
    EXPECT_EQ(a.equality_jacobian_claims(), b.equality_jacobian_claims());
    EXPECT_EQ(a.inequality_jacobian_claims(), b.inequality_jacobian_claims());

    const Eigen::VectorXi a_rows = a.kkt_claim_rows();
    const Eigen::VectorXi b_rows = b.kkt_claim_rows();
    const Eigen::VectorXi a_cols = a.kkt_claim_cols();
    const Eigen::VectorXi b_cols = b.kkt_claim_cols();
    ASSERT_EQ(a_rows.size(), b_rows.size());
    EXPECT_TRUE(a_rows == b_rows);
    EXPECT_TRUE(a_cols == b_cols);

    const Eigen::VectorXi a_grad = a.objective_gradient_claim_rows();
    const Eigen::VectorXi b_grad = b.objective_gradient_claim_rows();
    ASSERT_EQ(a_grad.size(), b_grad.size());
    EXPECT_TRUE(a_grad == b_grad);
}

// The objective-gradient arena's claims name declared variables, one slot per
// row a piece will sum into.
TEST(Level2Provider, TheObjectiveGradientClaimsNameDeclaredVariables) {
    L2ProviderFixture fixture;
    TranscribedAggregate &provider = fixture.provider();

    const int primal = provider.declaration().primal_vars_;
    const Eigen::VectorXi rows = provider.objective_gradient_claim_rows();
    EXPECT_GT(rows.size(), 0);
    for (int slot = 0; slot < rows.size(); slot++) {
        EXPECT_GE(rows[slot], -1);
        EXPECT_LT(rows[slot], primal);
    }
}

///////////////////////////////////////////////////////////////////////////////
// What the provider evaluates, and what it refuses.
///////////////////////////////////////////////////////////////////////////////

// A candidate evaluation is in declaration space on both sides of this view, so
// it forwards to the program unchanged and the two answer identically.
TEST(Level2Provider, ACandidateEvaluationForwardsToTheProgram) {
    L2ProviderFixture fixture;
    TranscribedAggregate &provider = fixture.provider();

    const Eigen::VectorXd x = (Eigen::VectorXd(3) << 0.4, 1.1, 1.5).finished();
    const Eigen::VectorXd empty = Eigen::VectorXd::Zero(0);
    const hven::solvers::CandidatePoint point{x, empty, empty, 1.0};

    double through_view = 0.0;
    Eigen::VectorXd view_equality(1);
    Eigen::VectorXd view_inequality(1);
    provider.evaluate_candidate_values(point, {through_view, view_equality, view_inequality});

    double through_program = 0.0;
    Eigen::VectorXd program_equality(1);
    Eigen::VectorXd program_inequality(1);
    fixture.prob_.nlp_->evaluate_candidate_values(
        point, {through_program, program_equality, program_inequality});

    EXPECT_EQ(through_view, through_program);
    EXPECT_EQ(view_equality[0], program_equality[0]);
    EXPECT_EQ(view_inequality[0], program_inequality[0]);
}

// An assembly against a destination the caller laid is refused, by name and
// with the shape it was asked for. The program behind this view binds the value
// array its location tables address, so a fill can only land there; publishing
// the claim stream is what lets a consumer lay a destination, and filling one
// waits on the engine-side path that consumes a foreign aggregate.
TEST(Level2Provider, AnAssemblyAgainstACallerLaidDestinationIsRefused) {
    L2ProviderFixture fixture;
    TranscribedAggregate &provider = fixture.provider();

    const Eigen::VectorXd x = (Eigen::VectorXd(3) << 0.4, 1.1, 1.5).finished();
    const Eigen::VectorXd empty = Eigen::VectorXd::Zero(0);
    const hven::solvers::CandidatePoint point{x, empty, empty, 1.0};

    double objective = 0.0;
    hven::solvers::RhsScatterView rhs;
    rhs.objective_ = &objective;

    // A request that names no KKT-bearing output and no arena, so every entry
    // check passes and the refusal that fires is this view's own.
    EXPECT_THROW(provider.assemble(point, hven::solvers::EvalRequest::kObjectiveValue, {}, rhs),
                 std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// What the declaration refuses.
///////////////////////////////////////////////////////////////////////////////

// A bound that cannot narrow anything is refused where it is declared, with the
// variable named -- not later, where only the merged interval is left to name.
TEST(Level2Provider, AnInvertedBoundIsRefusedWhereItIsDeclared) {
    OptimizationProblem prob;
    l2_provider_build(prob);
    EXPECT_THROW(prob.add_variable_bound(1, 2.0, 1.0), std::invalid_argument);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(prob.add_variable_bound(1, nan, 1.0), std::invalid_argument);
    EXPECT_THROW(prob.add_variable_bound(-1, 0.0, 1.0), std::invalid_argument);
}

// Two records on one variable intersect tightest-wins, and an intersection that
// is empty is refused when the declaration is laid.
TEST(Level2Provider, AnEmptyBoundIntersectionIsRefusedAtTheLayout) {
    OptimizationProblem prob;
    l2_provider_build(prob);
    prob.add_variable_bound(1, -1.0, 0.0);
    prob.add_variable_bound(1, 1.0, 2.0);
    EXPECT_THROW(prob.transcribe(), std::invalid_argument);
}
