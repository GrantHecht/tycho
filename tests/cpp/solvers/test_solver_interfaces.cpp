///////////////////////////////////////////////////////////////////////////////
// Solver interface tests (ConstraintInterface / ObjectiveInterface)
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
// Not reachable through the tycho.h umbrella; included directly so the
// registered family below is named together with its registration.
#include "tycho/detail/optimal_control/transcription/mesh_spacing_constraints.h"
#include <gtest/gtest.h>
#include <typeinfo>

using namespace tycho;
using TychoTest::SolverTest;

// ConstraintInterface/ObjectiveInterface live in tycho::solvers; this file
// previously relied on the TychoTest -> tycho::solvers using-directive leak
// (fixed in solver_test_utils.h) to see them unqualified.
using tycho::solvers::ConstraintInterface;
using tycho::solvers::ConstraintModel;
using tycho::solvers::ObjectiveInterface;
using tycho::solvers::ObjectiveModel;

TEST_F(SolverTest, ConstraintInterfaceFromScalar) {
    auto args = Arguments<3>();
    auto n = args.norm();
    GenericFunction<-1, 1> gf(n);
    ConstraintInterface ci(gf);
    EXPECT_EQ(ci.input_rows(), 3);
    EXPECT_EQ(ci.output_rows(), 1);
}

TEST_F(SolverTest, ObjectiveInterfaceFromScalar) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    GenericFunction<-1, 1> gf(sn);
    ObjectiveInterface oi(gf);
    EXPECT_EQ(oi.input_rows(), 3);
    EXPECT_EQ(oi.output_rows(), 1);
}

TEST_F(SolverTest, ConstraintInterfaceCopy) {
    auto args = Arguments<4>();
    auto f = 2.0 * args;
    GenericFunction<-1, -1> gf(f);
    ConstraintInterface ci1(gf);
    ConstraintInterface ci2(ci1);
    EXPECT_EQ(ci2.input_rows(), 4);
    EXPECT_EQ(ci2.output_rows(), 4);
}

///////////////////////////////////////////////////////////////////////////////
// What actually gets stored
//
// Sizes and numerics cannot tell one virtual dispatch from two -- both routes
// evaluate the same function and return the same numbers. The stored dynamic
// type can: a GenericFunction stored AS ITSELF would show up as
// ConstraintModel<GenericFunction<...>> and cost a second dispatch on every
// solver evaluation, silently. These pin the concrete payload instead.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, GenericFunctionStoresItsWrappedFunctionAsAConstraint) {
    auto args = Arguments<3>();
    auto expr = 2.0 * args;
    using ConcreteExpr = decltype(expr);

    GenericFunction<-1, -1> gf(expr);
    ConstraintInterface ci(gf);

    EXPECT_EQ(typeid(ci.storage_.get()), typeid(ConstraintModel<ConcreteExpr>));
    EXPECT_NE(typeid(ci.storage_.get()), typeid(ConstraintModel<GenericFunction<-1, -1>>));
    EXPECT_EQ(ci.input_rows(), 3);
    EXPECT_EQ(ci.output_rows(), 3);
}

TEST_F(SolverTest, GenericFunctionStoresItsWrappedFunctionAsAnObjective) {
    auto args = Arguments<3>();
    auto expr = args.squared_norm();
    using ConcreteExpr = decltype(expr);

    GenericFunction<-1, 1> gf(expr);
    ObjectiveInterface oi(gf);

    EXPECT_EQ(typeid(oi.storage_.get()), typeid(ObjectiveModel<ConcreteExpr>));
    EXPECT_NE(typeid(oi.storage_.get()), typeid(ObjectiveModel<GenericFunction<-1, 1>>));
    EXPECT_EQ(oi.input_rows(), 3);
    EXPECT_EQ(oi.output_rows(), 1);
}

// The sanctioned route for an entry site whose function is an expression built
// in place (the Kepler shooter): wrapping in a GenericFunction stores exactly
// what a family registration would have stored.
TEST_F(SolverTest, WrappingAComposedExpressionStoresTheExpressionItself) {
    auto args = Arguments<4>();
    auto expr = args.head<2>() - args.tail<2>();
    using ConcreteExpr = decltype(expr);

    ConstraintInterface ci{GenericFunction<-1, -1>(expr)};

    EXPECT_EQ(typeid(ci.storage_.get()), typeid(ConstraintModel<ConcreteExpr>));
    EXPECT_NE(typeid(ci.storage_.get()), typeid(ConstraintModel<GenericFunction<-1, -1>>));
}

// The same hatch on the objective side: this is the route the phase-integral
// ladder takes, where one erasure serves both interfaces.
TEST_F(SolverTest, WrappingAComposedExpressionStoresTheExpressionItselfAsAnObjective) {
    auto args = Arguments<4>();
    auto expr = args.head<2>().squared_norm();
    using ConcreteExpr = decltype(expr);

    ObjectiveInterface oi{GenericFunction<-1, 1>(expr)};

    EXPECT_EQ(typeid(oi.storage_.get()), typeid(ObjectiveModel<ConcreteExpr>));
    EXPECT_NE(typeid(oi.storage_.get()), typeid(ObjectiveModel<GenericFunction<-1, 1>>));
    EXPECT_EQ(oi.output_rows(), 1);
}

// A registered family goes in directly -- no wrapper, no second erasure.
TEST_F(SolverTest, RegisteredFamilyIsStoredAsItself) {
    ConstraintInterface ci(oc::SingleMeshSpacing(0.5));

    EXPECT_EQ(typeid(ci.storage_.get()), typeid(ConstraintModel<oc::SingleMeshSpacing>));
    EXPECT_EQ(ci.input_rows(), 3);
    EXPECT_EQ(ci.output_rows(), 1);
}
