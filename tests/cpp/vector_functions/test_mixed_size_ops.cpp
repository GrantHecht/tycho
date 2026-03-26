///////////////////////////////////////////////////////////////////////////////
// Mixed static/dynamic size operator tests and pack() tests
//
// Validates that VF DSL operators compose correctly when one operand has
// dynamic sizes (IRC/ORC = -1) and the other has static sizes.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

class MixedSizeOpsTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// operator+ with mixed ORC (the core pain point)
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, DynamicSegmentPlusStaticCrossProduct) {
    // This is the exact scenario from PAIN_POINTS.md #1:
    // ODEArguments<-1,-1,-1> segment (ORC=-1) + CrossProduct (ORC=3)
    auto args = ODEArguments(7, 3, 0);
    auto R = args.segment(0, 3);  // Segment<-1,-1,-1>, ORC=-1
    auto V = args.segment(3, 3);  // Segment<-1,-1,-1>, ORC=-1

    Eigen::Vector3d omega_val(0.0, 0.0, 1.0);
    auto omega = Constant<-1, 3>(11, omega_val);  // ORC=3

    // This is the line that previously failed to compile
    auto Vr = V + R.cross(omega);

    EXPECT_EQ(Vr.IRows(), 11);
    EXPECT_EQ(Vr.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(11, 100, -5.0, 5.0);
    verify_jacobian_fd(Vr, x);
}

TEST_F(MixedSizeOpsTest, StaticSegmentPlusStaticCrossProduct) {
    // Template segment accessors: ORC=3 + ORC=3 (should still work)
    auto args = ODEArguments(7, 3, 0);
    auto R = args.head<3>();       // ORC=3
    auto V = args.segment<3>(3);   // ORC=3

    Eigen::Vector3d omega_val(0.0, 0.0, 1.0);
    auto omega = Constant<-1, 3>(11, omega_val);

    auto Vr = V + R.cross(omega);
    EXPECT_EQ(Vr.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(11, 101, -5.0, 5.0);
    verify_jacobian_fd(Vr, x);
}

TEST_F(MixedSizeOpsTest, DynamicSegmentMinusStaticCrossProduct) {
    auto args = ODEArguments(7, 3, 0);
    auto R = args.segment(0, 3);
    auto V = args.segment(3, 3);

    Eigen::Vector3d omega_val(0.0, 0.0, 1.0);
    auto omega = Constant<-1, 3>(11, omega_val);

    auto Vr = V - R.cross(omega);
    EXPECT_EQ(Vr.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(11, 102, -5.0, 5.0);
    verify_jacobian_fd(Vr, x);
}

///////////////////////////////////////////////////////////////////////////////
// operator* and operator/ with mixed IRC
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, DynamicVectorTimesStaticScalar) {
    // Vector (IRC=-1) * Scalar (IRC=3) — mixed IRC
    auto dyn_args = Arguments<-1>(6);
    auto vec = dyn_args.segment(0, 3);   // IRC=-1, ORC=-1 (3 at runtime)

    auto stat_args = Arguments<6>();
    auto scl = stat_args.coeff<5>();      // IRC=6, ORC=1

    // Both have the same runtime IRows (6) but different compile-time IRC
    auto result = vec * scl;
    EXPECT_EQ(result.IRows(), 6);
    EXPECT_EQ(result.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 103, 0.5, 5.0);
    verify_jacobian_fd(result, x);
}

TEST_F(MixedSizeOpsTest, DynamicVectorDividedByStaticScalar) {
    auto dyn_args = Arguments<-1>(6);
    auto vec = dyn_args.segment(0, 3);

    auto stat_args = Arguments<6>();
    auto scl = stat_args.coeff<5>();

    auto result = vec / scl;
    EXPECT_EQ(result.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 104, 0.5, 5.0);
    verify_jacobian_fd(result, x);
}

///////////////////////////////////////////////////////////////////////////////
// Full ODE-like expression with dynamic args + vector operations
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, FullDynamicODEExpression) {
    // Simulate a simplified rocket ODE with dynamic args
    auto args = ODEArguments(7, 3, 0);

    // Use dynamic segments (ORC=-1) — the relaxed operators allow this
    auto R = args.segment(0, 3);
    auto V = args.segment(3, 3);
    auto m = args.coeff(6);

    Eigen::Vector3d omega_val(0.0, 0.0, 0.1);
    auto omega = Constant<-1, 3>(11, omega_val);

    auto Vr = V + R.cross(omega);         // dynamic + cross -> mixed ORC
    auto grav = (-1.0) * R.normalized();   // normalized preserves dynamic ORC
    auto Rdot = V;
    auto Vdot = grav + Vr / m;            // mixed operations

    auto ode = StackedOutputs{Rdot, Vdot};
    EXPECT_EQ(ode.IRows(), 11);
    EXPECT_EQ(ode.ORows(), 6);

    Eigen::VectorXd x = deterministic_random_vector(11, 105, 1.0, 10.0);
    // Ensure position vector norm > 0 for normalized()
    x[0] = 5.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[6] = 1.0;  // mass > 0
    verify_jacobian_fd(ode, x, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// pack() tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, PackPreservesValue) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto cross_expr = a.cross(b);

    auto packed = cross_expr.pack();  // GenericFunction<6, 3>
    EXPECT_EQ(packed.IRows(), 6);
    EXPECT_EQ(packed.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 106, -5.0, 5.0);
    Eigen::VectorXd fx_orig(3), fx_packed(3);
    fx_orig.setZero();
    fx_packed.setZero();
    cross_expr.compute(x, fx_orig);
    packed.compute(x, fx_packed);

    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(fx_orig[i], fx_packed[i]);
}

TEST_F(MixedSizeOpsTest, PackPreservesJacobian) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto cross_expr = a.cross(b);

    auto packed = cross_expr.pack();

    Eigen::VectorXd x = deterministic_random_vector(6, 107, -5.0, 5.0);
    verify_jacobian_fd(packed, x);
}

TEST_F(MixedSizeOpsTest, PackedExpressionComposable) {
    // pack() result can be composed further
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();

    auto cross_packed = a.cross(b).pack();    // GenericFunction<6, 3>
    auto result = cross_packed + a;           // should compile: both ORC=3

    EXPECT_EQ(result.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 108, -5.0, 5.0);
    verify_jacobian_fd(result, x);
}

TEST_F(MixedSizeOpsTest, PackDynamicSizes) {
    // pack() on a dynamic expression preserves dynamic IR/OR
    auto args = Arguments<-1>(6);
    auto seg = args.segment(0, 3);
    auto packed = seg.pack();  // GenericFunction<-1, -1>
    EXPECT_EQ(packed.IRows(), 6);
    EXPECT_EQ(packed.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(6, 109, -5.0, 5.0);
    Eigen::VectorXd fx(3);
    fx.setZero();
    packed.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], x[0]);
    EXPECT_DOUBLE_EQ(fx[1], x[1]);
    EXPECT_DOUBLE_EQ(fx[2], x[2]);
}

TEST_F(MixedSizeOpsTest, PackAdjointConsistency) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto packed = (a.cross(b) + 2.0 * a).pack();

    Eigen::VectorXd x = deterministic_random_vector(6, 110, -5.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 111, -1.0, 1.0);
    verify_adjoint_consistency(packed, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Regression: static-static mismatch still caught at compile time
// (these are compile-time checks — we just verify the valid cases work)
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, StaticMatchStillWorks) {
    auto args = Arguments<6>();
    auto a = args.head<3>();
    auto b = args.tail<3>();
    auto sum = a + b;  // ORC=3 + ORC=3 — was and still is valid
    EXPECT_EQ(sum.ORows(), 3);
}

///////////////////////////////////////////////////////////////////////////////
// Comparison operators with mixed IRC
///////////////////////////////////////////////////////////////////////////////

TEST_F(MixedSizeOpsTest, ComparisonMixedIR) {
    // Dynamic scalar (IRC=-1) compared with static scalar (IRC=6)
    // ConditionalStatement is not a DenseFunctionBase — it's a boolean
    // used inside IfElseFunction.  We verify the comparison compiles
    // and then use it in an IfElseFunction to confirm runtime behavior.
    auto dyn_args = Arguments<-1>(6);
    auto dyn_scl = dyn_args.coeff(0);         // IRC=-1, ORC=1

    auto stat_args = Arguments<6>();
    auto stat_scl = stat_args.coeff<5>();      // IRC=6, ORC=1

    auto cond = dyn_scl > stat_scl;           // mixed IR comparison compiles
    EXPECT_EQ(cond.IRC, -1);                  // SZ_MAX<-1, 6> = -1

    // Use the conditional in an IfElseFunction to verify runtime behavior
    auto true_branch = dyn_scl;               // returns x[0]
    auto false_branch = stat_scl;             // returns x[5]
    auto ite = IfElseFunction(cond, true_branch, false_branch);

    Eigen::VectorXd x = deterministic_random_vector(6, 112, -5.0, 5.0);
    Eigen::VectorXd fx(1);
    fx.setZero();
    ite.compute(x, fx);
    double expected = (x[0] > x[5]) ? x[0] : x[5];
    EXPECT_DOUBLE_EQ(fx[0], expected);
}
