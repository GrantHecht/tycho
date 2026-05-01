///////////////////////////////////////////////////////////////////////////////
// ODE definition syntax prototypes
//
// Tests the Impl+macro ODE definition pattern with:
//   - ODEArguments with offset-aware XVar/UVar tags
//   - FD macro variants for reduced compile time
//   - Hessian consistency through FD wrappers
//   - Dynamic-size ODE wrapping via StaticODE_DerivModeWrapper
//
// Both features are complementary: tags provide semantic access to ODE
// variables, while FD macros avoid expensive Jacobian/Hessian
// template instantiation.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::vf;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// ODEArguments + XVar/UVar tags — Impl+macro pattern
///////////////////////////////////////////////////////////////////////////////
namespace {

// Brachistochrone — single-parameter (analytic Jacobian)
struct Brachistochrone_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g_) {
        auto args = ODEArguments<3, 1, 0>();
        auto v = args[XVar<2>];
        auto theta = args[UVar<0>]; // offset-aware: resolves to index 4
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0), g_ * cos(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION(Brachistochrone, Brachistochrone_Impl, double);

// Multi-parameter variant (drag brachistochrone)
struct SyntaxDragBrach_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g, double drag) {
        auto args = ODEArguments<3, 1, 0>();
        auto v = args[XVar<2>];
        auto theta = args[UVar<0>];
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0),
                              g * cos(theta) - drag * v};
    }
};
BUILD_ODE_FROM_EXPRESSION(SyntaxDragBrach, SyntaxDragBrach_Impl, double, double);
BUILD_ODE_FROM_EXPRESSION_FD(SyntaxDragBrachFD, SyntaxDragBrach_Impl, double, double);

///////////////////////////////////////////////////////////////////////////////
// FD macro variants — same Impl+macro pattern but avoids expensive
// Jacobian/Hessian template instantiation
///////////////////////////////////////////////////////////////////////////////

// FD variant
struct BrachistochroneFD_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = ODEArguments<3, 1, 0>();
        auto v = args[XVar<2>];
        auto theta = args[UVar<0>];
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION_FD(BrachistochroneFD, BrachistochroneFD_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Dynamic-size ODE for testing StaticODE_DerivModeWrapper with IRC == -1
///////////////////////////////////////////////////////////////////////////////

struct DynamicBrach : StaticODE<DynamicBrach, -1, -1, -1> {
    DynamicBrach() { set_ode_size(3, 1, 0); }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x,
                             CVecRef<OutType> fx_) const {
        auto &fx = const_cast<Eigen::MatrixBase<OutType> &>(fx_);
        auto v = x[2];
        auto theta = x[4];
        using std::cos;
        using std::sin;
        fx[0] = sin(theta) * v;
        fx[1] = cos(theta) * v * (-1.0);
        fx[2] = 9.81 * cos(theta);
    }
};

struct DynamicBrachFD
    : StaticODE_DerivModeWrapper<DynamicBrachFD, DynamicBrach, DenseDerivativeMode::FDiffFwd,
                                 DenseDerivativeMode::FDiffFwd> {
    using Base =
        StaticODE_DerivModeWrapper<DynamicBrachFD, DynamicBrach, DenseDerivativeMode::FDiffFwd,
                                   DenseDerivativeMode::FDiffFwd>;
    using Base::Base;
};

} // anonymous namespace

///////////////////////////////////////////////////////////////////////////////
// Tests — analytic Jacobian
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, Brachistochrone_Basic) {
    Brachistochrone ode(9.81);
    EXPECT_EQ(ode.input_rows(), 5);
    EXPECT_EQ(ode.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    ode.compute(x, fx);

    double v = 5.0, theta = std::numbers::pi / 4.0;
    EXPECT_NEAR(fx[0], std::sin(theta) * v, 1e-12);
    EXPECT_NEAR(fx[1], -std::cos(theta) * v, 1e-12);
    EXPECT_NEAR(fx[2], 9.81 * std::cos(theta), 1e-12);
}

TEST_F(VFCompositionTest, MultiParam_AdjointConsistency) {
    SyntaxDragBrach ode(9.81, 0.1);
    EXPECT_EQ(ode.input_rows(), 5);
    EXPECT_EQ(ode.output_rows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(5, 400, 0.1, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 401, -1.0, 1.0);
    verify_adjoint_consistency(ode, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, Brachistochrone_Integrator) {
    Brachistochrone ode(9.81);
    auto integ = ode.integrator(0.01);
    EXPECT_EQ(integ.input_rows(), 6);
    EXPECT_EQ(integ.output_rows(), 5);
}

///////////////////////////////////////////////////////////////////////////////
// Tests — FD Jacobian
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, FD_ComputeMatchesAnalytic) {
    Brachistochrone analytic(9.81);
    BrachistochroneFD fd(9.81);

    EXPECT_EQ(fd.input_rows(), 5);
    EXPECT_EQ(fd.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx_a(3), fx_fd(3);
    fx_a.setZero();
    fx_fd.setZero();
    analytic.compute(x, fx_a);
    fd.compute(x, fx_fd);

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(fx_a[i], fx_fd[i], 1e-14);
}

TEST_F(VFCompositionTest, FD_JacobianMatchesAnalytic) {
    Brachistochrone analytic(9.81);
    BrachistochroneFD fd(9.81);

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;

    // Analytic Jacobian
    Eigen::VectorXd fx_a(3);
    Eigen::MatrixXd jx_a(3, 5);
    fx_a.setZero();
    jx_a.setZero();
    analytic.compute_jacobian(x, fx_a, jx_a);

    // FD Jacobian
    Eigen::VectorXd fx_fd(3);
    Eigen::MatrixXd jx_fd(3, 5);
    fx_fd.setZero();
    jx_fd.setZero();
    fd.compute_jacobian(x, fx_fd, jx_fd);

    // FD approximation should be close to analytic
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_NEAR(jx_a(i, j), jx_fd(i, j), 1e-6)
                << "Jacobian mismatch at (" << i << "," << j << ")";
}

TEST_F(VFCompositionTest, FD_MultiParam_JacobianMatchesAnalytic) {
    SyntaxDragBrach analytic(9.81, 0.1);
    SyntaxDragBrachFD fd(9.81, 0.1);
    EXPECT_EQ(fd.input_rows(), 5);
    EXPECT_EQ(fd.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;

    Eigen::VectorXd fx_a(3);
    Eigen::MatrixXd jx_a(3, 5);
    fx_a.setZero();
    jx_a.setZero();
    analytic.compute_jacobian(x, fx_a, jx_a);

    Eigen::VectorXd fx_fd(3);
    Eigen::MatrixXd jx_fd(3, 5);
    fx_fd.setZero();
    jx_fd.setZero();
    fd.compute_jacobian(x, fx_fd, jx_fd);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_NEAR(jx_a(i, j), jx_fd(i, j), 1e-6)
                << "Multi-param FD Jacobian mismatch at (" << i << "," << j << ")";
}

///////////////////////////////////////////////////////////////////////////////
// Tests — Hessian consistency through FD wrappers
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, FD_HessianConsistency) {
    BrachistochroneFD fd(9.81);

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;
    Eigen::VectorXd lm = deterministic_random_vector(3, 500, -1.0, 1.0);

    // FD Hessian: check symmetry and adjoint gradient consistency
    verify_hessian_consistency(fd, x, lm, 1e-5);
}

///////////////////////////////////////////////////////////////////////////////
// Tests — Dynamic-size ODE wrapping (IRC == -1)
//
// Regression test for the FD step re-initialization fix. When
// StaticODE_DerivModeWrapper wraps a dynamic-size ODE (IRC == -1), the FD step
// vectors must be reinitialized after set_io_rows. The macros always produce
// compile-time-sized inner ODEs, so this test manually constructs a wrapper
// around a dynamic-size ODE to exercise that path.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, DynamicODE_FD_ComputeCorrect) {
    DynamicBrachFD fd;
    EXPECT_EQ(fd.input_rows(), 5);
    EXPECT_EQ(fd.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    fd.compute(x, fx);

    double v = 5.0, theta = std::numbers::pi / 4.0;
    EXPECT_NEAR(fx[0], std::sin(theta) * v, 1e-12);
    EXPECT_NEAR(fx[1], -std::cos(theta) * v, 1e-12);
    EXPECT_NEAR(fx[2], 9.81 * std::cos(theta), 1e-12);
}

TEST_F(VFCompositionTest, DynamicODE_FD_JacobianMatchesAnalytic) {
    Brachistochrone analytic(9.81);
    DynamicBrachFD fd;

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;

    // Analytic Jacobian from expression-based ODE
    Eigen::VectorXd fx_a(3);
    Eigen::MatrixXd jx_a(3, 5);
    fx_a.setZero();
    jx_a.setZero();
    analytic.compute_jacobian(x, fx_a, jx_a);

    // FD Jacobian from dynamic-size wrapper
    Eigen::VectorXd fx_fd(3);
    Eigen::MatrixXd jx_fd(3, 5);
    fx_fd.setZero();
    jx_fd.setZero();
    fd.compute_jacobian(x, fx_fd, jx_fd);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_NEAR(jx_a(i, j), jx_fd(i, j), 1e-6)
                << "Dynamic FD Jacobian mismatch at (" << i << "," << j << ")";
}

TEST_F(VFCompositionTest, DynamicODE_FD_HessianConsistency) {
    DynamicBrachFD fd;
    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;
    Eigen::VectorXd lm = deterministic_random_vector(3, 502, -1.0, 1.0);
    verify_hessian_consistency(fd, x, lm, 1e-5);
}

