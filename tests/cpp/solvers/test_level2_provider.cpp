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

#include "tycho/detail/solvers_vf/transcription_declaration.h"

#include <hven/detail/interior/constraint_function.h>
#include <hven/detail/interior/objective_function.h>
#include <hven/model/candidate_point.h>
#include <hven/model/claim_space.h>
#include <hven/model/non_linear_program.h>

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>

using tycho::solvers::ConstraintFunction;
using tycho::solvers::NonLinearProgram;
using tycho::solvers::ObjectiveFunction;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::ThreadingFlags;
using tycho::solvers::TranscribedAggregate;
using tycho::solvers::TranscriptionDeclaration;
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

/// A problem big enough that the layout's element-per-partition floor lets it
/// adopt more than one partition: 120 applications of an eight-variable
/// equality constraint, each contributing a dense 8x8 upper-triangle Hessian
/// plus its Jacobian row, which is thousands of claim slots rather than the
/// dozen the small fixture has.
constexpr int kL2ProviderGroups = 120;
constexpr int kL2ProviderGroupSize = 8;

void l2_provider_build_wide(OptimizationProblem &prob, int requested_partitions) {
    const int variables = kL2ProviderGroups * kL2ProviderGroupSize;
    prob.set_vars(Eigen::VectorXd::LinSpaced(variables, 0.1, 1.0));

    std::vector<Eigen::VectorXi> groups;
    groups.reserve(kL2ProviderGroups);
    for (int g = 0; g < kL2ProviderGroups; g++) {
        groups.push_back(Eigen::VectorXi::LinSpaced(kL2ProviderGroupSize, g * kL2ProviderGroupSize,
                                                    (g + 1) * kL2ProviderGroupSize - 1));
    }

    {
        auto args = Arguments<kL2ProviderGroupSize>();
        prob.add_equal_con(GenericFunction<-1, -1>(args.squared_norm() - 1.0), groups);
    }
    {
        auto args = Arguments<kL2ProviderGroupSize>();
        prob.add_inequal_con(GenericFunction<-1, -1>(args.squared_norm() - 0.5), groups[0]);
    }
    {
        auto args = Arguments<kL2ProviderGroupSize>();
        prob.add_objective(GenericFunction<-1, 1>(args.squared_norm()), groups[0]);
    }
    prob.set_num_partitions(requested_partitions);
}

/// A one-row constraint piece over the first two variables, applied
/// @p applications times, every application writing constraint row @p row.
///
/// With one application it is an ordinary piece. With more, every application
/// sums into the same row -- what an accumulating integrand declares, and what
/// the declaration boundary's equality row-sum conjunct cannot describe.
ConstraintFunction l2_provider_row_piece(int applications, int row) {
    Eigen::MatrixXi vindex(2, applications);
    Eigen::MatrixXi cindex(1, applications);
    for (int a = 0; a < applications; a++) {
        vindex(0, a) = 0;
        vindex(1, a) = 1;
        cindex(0, a) = row;
    }
    auto args = Arguments<2>();
    return ConstraintFunction(GenericFunction<-1, -1>(args.squared_norm() - 1.0), vindex, cindex);
}

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
    OptimizationProblem prob;
    l2_provider_build_wide(prob, 4);
    prob.transcribe();
    TranscribedAggregate &provider = *prob.provider_;

    const int adopted = provider.declaration().partition_count_;
    // The pin is about MULTI-partition order, so the fixture has to reach more
    // than one partition or the body below asserts nothing.
    ASSERT_GT(adopted, 1);

    const Eigen::VectorXi partitions = provider.kkt_claim_partitions();
    ASSERT_EQ(partitions.size(), provider.kkt_claim_rows().size());

    const std::vector<hven::solvers::ClaimBlock> runs = {provider.hessian_claims(),
                                                         provider.equality_jacobian_claims(),
                                                         provider.inequality_jacobian_claims()};
    std::set<int> occupied;
    for (const auto &run : runs) {
        int previous = -1;
        for (int slot = run.start_; slot < run.start_ + run.count_; slot++) {
            const int partition = partitions[slot];
            EXPECT_GE(partition, 0);
            EXPECT_LT(partition, adopted);
            // Partition-INDEX order, asserted as such: the tag never decreases
            // along a run, so 2, 0, 1 fails here rather than passing a
            // never-revisited test.
            EXPECT_GE(partition, previous);
            previous = partition;
            occupied.insert(partition);
        }
    }

    // And the spread is real: an all-claims-in-one-partition layout would make
    // the ordering assertion above vacuous even at an adopted count above one.
    EXPECT_GT(occupied.size(), 1u);
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

///////////////////////////////////////////////////////////////////////////////
// What a fixed-variable treatment may and may not move.
///////////////////////////////////////////////////////////////////////////////

// A treatment that ELIMINATES variables re-lays the program in a narrower
// space. That is the engine's own reduction and not declaration data, so the
// published stream -- which is stated in DECLARED identities -- must come back
// unchanged, coordinate for coordinate, and must never carry the negative
// placeholder the narrower layout records for an eliminated coordinate.
TEST(Level2Provider, AnEliminatingTreatmentDoesNotMoveTheClaimStream) {
    OptimizationProblem prob;
    l2_provider_build(prob);
    // A variable whose declared bounds coincide is what a treatment acts on.
    prob.add_variable_bound(2, 1.25, 1.25);
    prob.transcribe();
    TranscribedAggregate &provider = *prob.provider_;

    const Eigen::VectorXi rows_before = provider.kkt_claim_rows();
    const Eigen::VectorXi cols_before = provider.kkt_claim_cols();
    const Eigen::VectorXi gradient_before = provider.objective_gradient_claim_rows();
    const auto hessian_before = provider.hessian_claims();
    const auto equality_before = provider.equality_jacobian_claims();
    const auto inequality_before = provider.inequality_jacobian_claims();
    const auto epoch_before = provider.structure_epoch();

    ASSERT_TRUE(prob.nlp_->configure_variable_treatment(
        hven::solvers::FixedVariableTreatments::MakeParameter, 0.0));
    ASSERT_TRUE(prob.nlp_->is_reduced());
    EXPECT_FALSE(provider.structure_epoch() == epoch_before);

    const Eigen::VectorXi rows_after = provider.kkt_claim_rows();
    const Eigen::VectorXi cols_after = provider.kkt_claim_cols();
    const Eigen::VectorXi gradient_after = provider.objective_gradient_claim_rows();

    ASSERT_EQ(rows_after.size(), rows_before.size());
    EXPECT_TRUE(rows_after == rows_before);
    EXPECT_TRUE(cols_after == cols_before);
    ASSERT_EQ(gradient_after.size(), gradient_before.size());
    EXPECT_TRUE(gradient_after == gradient_before);

    EXPECT_EQ(provider.hessian_claims(), hessian_before);
    EXPECT_EQ(provider.equality_jacobian_claims(), equality_before);
    EXPECT_EQ(provider.inequality_jacobian_claims(), inequality_before);

    // No eliminated-coordinate placeholder reaches the published stream, so
    // every slot still names a coordinate of the assembled space and no
    // dropped Jacobian slot can be counted in the Hessian run.
    const int dimension = provider.kkt_dimension();
    for (int slot = 0; slot < rows_after.size(); slot++) {
        EXPECT_GE(rows_after[slot], 0);
        EXPECT_LT(rows_after[slot], dimension);
        EXPECT_GE(cols_after[slot], 0);
        EXPECT_LT(cols_after[slot], dimension);
    }
}

// The companion: a treatment that keeps every variable and one that eliminates
// them produce the SAME stream from the same declaration. The claim stream is a
// function of the declaration and the adopted partition count, and the
// treatment is neither.
TEST(Level2Provider, TheClaimStreamIsTheSameUnderEveryTreatmentThatKeepsTheDeclaration) {
    auto stream_under = [](hven::solvers::FixedVariableTreatments treatment, double relax) {
        auto prob = std::make_shared<OptimizationProblem>();
        l2_provider_build(*prob);
        prob->add_variable_bound(2, 1.25, 1.25);
        prob->transcribe();
        prob->nlp_->configure_variable_treatment(treatment, relax);
        Eigen::VectorXi rows = prob->provider_->kkt_claim_rows();
        Eigen::VectorXi cols = prob->provider_->kkt_claim_cols();
        return std::make_tuple(prob, rows, cols);
    };

    auto [eliminated, eliminated_rows, eliminated_cols] =
        stream_under(hven::solvers::FixedVariableTreatments::MakeParameter, 0.0);
    auto [relaxed, relaxed_rows, relaxed_cols] =
        stream_under(hven::solvers::FixedVariableTreatments::RelaxBounds, 1.0e-8);

    ASSERT_TRUE(eliminated->nlp_->is_reduced());
    ASSERT_FALSE(relaxed->nlp_->is_reduced());

    ASSERT_EQ(eliminated_rows.size(), relaxed_rows.size());
    EXPECT_TRUE(eliminated_rows == relaxed_rows);
    EXPECT_TRUE(eliminated_cols == relaxed_cols);
}

///////////////////////////////////////////////////////////////////////////////
// Re-laying under a bound consumer.
///////////////////////////////////////////////////////////////////////////////

// Adopting a partition count re-lays the program, which empties the location
// table every scatter addresses and unbinds the destination those offsets
// described. The program re-analyses itself only when a fixed-variable
// treatment reports a change, so on a problem with no fixed variable the
// emptied table would survive into the next solve. Refused while a consumer is
// bound, and the refusal names the program's own binding as the reason.
TEST(Level2Provider, NegotiatingIsRefusedWhileAConsumerIsBoundToTheProgram) {
    L2ProviderFixture fixture;
    // transcribe() hands the program to the optimizer, whose sparsity analysis
    // binds its own destination.
    ASSERT_NE(fixture.prob_.nlp_->bound_kkt_destination(), nullptr);
    EXPECT_THROW(fixture.provider().negotiate_partition_count(1), std::invalid_argument);
    EXPECT_THROW(fixture.provider().negotiate_partition_count(4), std::invalid_argument);

    // And the refusal did not re-lay anything on the way out.
    EXPECT_NE(fixture.prob_.nlp_->bound_kkt_destination(), nullptr);
}

// The guard is the binding and nothing else: a program no consumer has analysed
// negotiates normally, and the stream is re-read against the count adopted.
TEST(Level2Provider, NegotiatingIsServedWhileNoConsumerIsBound) {
    auto host = std::make_shared<NonLinearProgram>(1);
    {
        TranscriptionDeclaration declaration(1);
        Eigen::MatrixXi vindex(2, 1);
        vindex << 0, 1;
        Eigen::MatrixXi cindex(1, 1);
        cindex << 0;
        auto args = Arguments<2>();
        declaration.add_equality(
            ConstraintFunction(GenericFunction<-1, -1>(args.squared_norm() - 1.0), vindex, cindex),
            ThreadingFlags::RoundRobin);
        declaration.lay(*host, 2, 1, 0);
    }
    ASSERT_EQ(host->bound_kkt_destination(), nullptr);

    TranscribedAggregate provider(host);
    EXPECT_EQ(provider.negotiate_partition_count(4), 1);
    EXPECT_EQ(provider.declaration().partition_count_, 1);
    EXPECT_GT(provider.kkt_claim_rows().size(), 0);
}

///////////////////////////////////////////////////////////////////////////////
// What the shared-row layout route still refuses.
///////////////////////////////////////////////////////////////////////////////

// A declaration whose equality pieces share a constraint row cannot be counted
// by the declaration boundary's equality row-sum rule, so it is laid by a
// different entry. That entry stands down THAT conjunct and no other: an
// inequality row count its pieces do not sum to is still refused, and refused
// before anything of the program is written.
TEST(Level2Provider, TheSharedRowRouteStillRefusesTheRestOfTheDeclaration) {
    auto host = std::make_shared<NonLinearProgram>(1);
    TranscriptionDeclaration declaration(1);

    const int shared =
        declaration.add_equality(l2_provider_row_piece(3, 0), ThreadingFlags::MainThread);
    declaration.equality(shared).index_data_.unique_constraints_ = false;

    // One inequality piece claiming one row, against a declaration that states
    // two of them.
    Eigen::MatrixXi vindex(2, 1);
    vindex << 0, 1;
    Eigen::MatrixXi cindex(1, 1);
    cindex << 0;
    auto args = Arguments<2>();
    declaration.add_inequality(
        ConstraintFunction(GenericFunction<-1, -1>(args.squared_norm() - 0.5), vindex, cindex),
        ThreadingFlags::MainThread);

    EXPECT_THROW(declaration.lay(*host, 2, 1, 2), std::invalid_argument);
    // Refused before the program was touched.
    EXPECT_EQ(host->primal_vars_, 0);
    EXPECT_TRUE(host->equality_constraints_.empty());
}

// The same route, with a declaration that is otherwise sound: it lays, and the
// shared row is laid once rather than once per application.
TEST(Level2Provider, TheSharedRowRouteLaysASoundDeclaration) {
    auto host = std::make_shared<NonLinearProgram>(1);
    TranscriptionDeclaration declaration(1);

    const int shared =
        declaration.add_equality(l2_provider_row_piece(3, 0), ThreadingFlags::MainThread);
    declaration.equality(shared).index_data_.unique_constraints_ = false;

    declaration.lay(*host, 2, 1, 0);

    EXPECT_EQ(host->primal_vars_, 2);
    EXPECT_EQ(host->equal_cons_, 1);
    EXPECT_EQ(host->user_equal_cons_, 1);
    EXPECT_EQ(host->internal_fixed_cons_, 0);
    EXPECT_EQ(host->equality_constraints_.size(), 1u);
}

// The same route onto a program that is not fresh: one already laid, and
// carrying an internal fixing row of its own from a treatment configured over
// it. Replacing the three piece lists leaves that row's bookkeeping describing
// functions that are gone, and the layout call discards internal rows by that
// recorded count before it lays -- so a stale count deletes the piece just
// installed, and a stale row space trips the layout's own invariant.
//
// The fresh-host pin above cannot see any of that: on a program straight from
// its constructor the three counters are already zero, so it passes whether the
// bookkeeping is reset or not. This one fails if any one of the three resets is
// dropped.
TEST(Level2Provider, TheSharedRowRouteLaysOnAHostThatCarriedAnInternalFixingRow) {
    auto host = std::make_shared<NonLinearProgram>(1);

    // First layout: an ordinary declaration over two variables, one of them
    // declared fixed, then the treatment that hands a fixed variable an
    // internal equality row of its own.
    {
        TranscriptionDeclaration declaration(1);
        declaration.add_equality(l2_provider_row_piece(1, 0), ThreadingFlags::MainThread);
        declaration.set_variable_bound(1, 0.5, 0.5);
        declaration.lay(*host, 2, 1, 0);
    }
    ASSERT_TRUE(host->configure_variable_treatment(
        hven::solvers::FixedVariableTreatments::MakeConstraint, 0.0));
    // Exactly one internal row, which is the case that fails silently without
    // the resets: the discard finds a list long enough to truncate, so it
    // deletes the piece just installed instead of refusing.
    ASSERT_EQ(host->internal_fixed_cons_, 1);
    ASSERT_EQ(host->equality_constraints_.size(), 2u);
    ASSERT_EQ(host->equal_cons_, 2);
    ASSERT_EQ(host->user_equal_cons_, 1);

    // Second layout, on that same program, through the shared-row route.
    TranscriptionDeclaration declaration(1);
    const int shared =
        declaration.add_equality(l2_provider_row_piece(3, 0), ThreadingFlags::MainThread);
    declaration.equality(shared).index_data_.unique_constraints_ = false;
    declaration.lay(*host, 2, 1, 0);

    // The declared piece is on the program, laid once, and the internal-row
    // bookkeeping describes the new layout rather than the old one.
    EXPECT_EQ(host->equality_constraints_.size(), 1u);
    EXPECT_EQ(host->equal_cons_, 1);
    EXPECT_EQ(host->user_equal_cons_, 1);
    EXPECT_EQ(host->internal_fixed_cons_, 0);
    EXPECT_EQ(host->primal_vars_, 2);
}

// The view reads a layout's coordinates and states them as DECLARED
// identities, which it may do only while the two spaces are the same one. A
// program whose treatment has already eliminated variables lays a narrower
// space, so a view cannot be built over one: refused at construction, which is
// the entry with no previously published stream to fall back on.
//
// Two shapes of the same case, because they fail differently without the
// refusal. When the eliminated variable appears in a piece, its claims carry a
// negative coordinate and the read trips the negative-coordinate refusal on the
// way through -- so the refusal is only pinned by requiring it to be THIS one,
// naming the elimination rather than a layout that reports none. When the
// eliminated variable appears in no piece, nothing is negative and nothing is
// out of band: the read runs to completion and publishes a stream whose columns
// name the wrong declared variables, so there the refusal is pinned by there
// being one at all.
TEST(Level2Provider, AViewCannotBeBuiltOverAProgramWithVariablesEliminated) {
    {
        OptimizationProblem prob;
        l2_provider_build(prob);
        // x2 is fixed, and it appears in the objective and the equality row.
        prob.add_variable_bound(2, 1.25, 1.25);
        prob.transcribe();

        ASSERT_TRUE(prob.nlp_->configure_variable_treatment(
            hven::solvers::FixedVariableTreatments::MakeParameter, 0.0));
        ASSERT_TRUE(prob.nlp_->is_reduced());

        bool refused = false;
        try {
            TranscribedAggregate view(prob.nlp_);
            (void)view;
        } catch (const std::invalid_argument &refusal) {
            refused = true;
            EXPECT_NE(std::string(refusal.what()).find("has eliminated variables"),
                      std::string::npos)
                << "the refusal names something other than the elimination: " << refusal.what();
        }
        EXPECT_TRUE(refused);

        // The view the transcription built is unaffected: it read its stream
        // before the treatment ran, and keeps publishing it.
        EXPECT_GT(prob.provider_->kkt_claim_rows().size(), 0);
    }

    {
        // Four variables, of which the fixed one takes part in nothing, so its
        // elimination leaves no negative coordinate anywhere in the layout --
        // it only renumbers the variable declared after it.
        OptimizationProblem prob;
        prob.set_vars((Eigen::VectorXd(4) << 0.5, 1.5, 1.0, 2.0).finished());
        const Eigen::VectorXi used = (Eigen::VectorXi(3) << 0, 1, 3).finished();
        {
            auto args = Arguments<3>();
            prob.add_objective(GenericFunction<-1, 1>(args.squared_norm()), used);
        }
        {
            auto args = Arguments<3>();
            prob.add_equal_con(GenericFunction<-1, -1>(args.squared_norm() - 3.0), used);
        }
        prob.add_variable_bound(2, 2.0, 2.0);
        prob.transcribe();

        ASSERT_TRUE(prob.nlp_->configure_variable_treatment(
            hven::solvers::FixedVariableTreatments::MakeParameter, 0.0));
        ASSERT_TRUE(prob.nlp_->is_reduced());

        EXPECT_THROW(
            {
                TranscribedAggregate view(prob.nlp_);
                (void)view;
            },
            std::invalid_argument);
    }
}
