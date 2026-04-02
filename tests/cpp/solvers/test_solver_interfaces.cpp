///////////////////////////////////////////////////////////////////////////////
// Solver interface tests (ConstraintInterface / ObjectiveInterface)
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

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
