///////////////////////////////////////////////////////////////////////////////
// ODE definition syntax prototypes
//
// Tests the Impl+macro ODE definition pattern with:
//   - ODEArguments with offset-aware XVar/UVar tags
//   - FD/FWAD macro variants for reduced compile time
//   - Hessian consistency through FD/FWAD wrappers
//   - Dynamic-size ODE wrapping via ODE_DerivModeWrapper
//
// Both features are complementary: tags provide semantic access to ODE
// variables, while FD/FWAD macros avoid expensive Jacobian/Hessian
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
// FD/FWAD macro variants — same Impl+macro pattern but avoids expensive
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

// Forward-AD variant
struct BrachistochroneFWAD_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = ODEArguments<3, 1, 0>();
        auto v = args[XVar<2>];
        auto theta = args[UVar<0>];
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION_FWAD(BrachistochroneFWAD, BrachistochroneFWAD_Impl, double);
BUILD_ODE_FROM_EXPRESSION_FWAD(SyntaxDragBrachFWAD, SyntaxDragBrach_Impl, double, double);

///////////////////////////////////////////////////////////////////////////////
// Dynamic-size ODE for testing ODE_DerivModeWrapper with IRC == -1
///////////////////////////////////////////////////////////////////////////////

struct DynamicBrach : ODE<DynamicBrach, -1, -1, -1> {
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
    : ODE_DerivModeWrapper<DynamicBrachFD, DynamicBrach, DenseDerivativeMode::FDiffFwd,
                           DenseDerivativeMode::FDiffFwd> {
    using Base = ODE_DerivModeWrapper<DynamicBrachFD, DynamicBrach, DenseDerivativeMode::FDiffFwd,
                                      DenseDerivativeMode::FDiffFwd>;
    using Base::Base;
};

struct DynamicBrachFWAD
    : ODE_DerivModeWrapper<DynamicBrachFWAD, DynamicBrach, DenseDerivativeMode::AutodiffFwd,
                           DenseDerivativeMode::AutodiffFwd> {
    using Base =
        ODE_DerivModeWrapper<DynamicBrachFWAD, DynamicBrach, DenseDerivativeMode::AutodiffFwd,
                             DenseDerivativeMode::AutodiffFwd>;
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
// Tests — FD/FWAD Jacobian
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

TEST_F(VFCompositionTest, FWAD_ComputeMatchesAnalytic) {
    Brachistochrone analytic(9.81);
    BrachistochroneFWAD fwad(9.81);

    EXPECT_EQ(fwad.input_rows(), 5);
    EXPECT_EQ(fwad.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx_a(3), fx_fwad(3);
    fx_a.setZero();
    fx_fwad.setZero();
    analytic.compute(x, fx_a);
    fwad.compute(x, fx_fwad);

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(fx_a[i], fx_fwad[i], 1e-14);
}

TEST_F(VFCompositionTest, FWAD_JacobianMatchesAnalytic) {
    Brachistochrone analytic(9.81);
    BrachistochroneFWAD fwad(9.81);

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;

    Eigen::VectorXd fx_a(3);
    Eigen::MatrixXd jx_a(3, 5);
    fx_a.setZero();
    jx_a.setZero();
    analytic.compute_jacobian(x, fx_a, jx_a);

    Eigen::VectorXd fx_fwad(3);
    Eigen::MatrixXd jx_fwad(3, 5);
    fx_fwad.setZero();
    jx_fwad.setZero();
    fwad.compute_jacobian(x, fx_fwad, jx_fwad);

    // Forward-AD should be more accurate than FD
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_NEAR(jx_a(i, j), jx_fwad(i, j), 1e-13)
                << "FWAD Jacobian mismatch at (" << i << "," << j << ")";
}

TEST_F(VFCompositionTest, FWAD_MultiParam_JacobianMatchesAnalytic) {
    SyntaxDragBrach analytic(9.81, 0.1);
    SyntaxDragBrachFWAD fwad(9.81, 0.1);
    EXPECT_EQ(fwad.input_rows(), 5);
    EXPECT_EQ(fwad.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;

    Eigen::VectorXd fx_a(3);
    Eigen::MatrixXd jx_a(3, 5);
    fx_a.setZero();
    jx_a.setZero();
    analytic.compute_jacobian(x, fx_a, jx_a);

    Eigen::VectorXd fx_fwad(3);
    Eigen::MatrixXd jx_fwad(3, 5);
    fx_fwad.setZero();
    jx_fwad.setZero();
    fwad.compute_jacobian(x, fx_fwad, jx_fwad);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_NEAR(jx_a(i, j), jx_fwad(i, j), 1e-13)
                << "Multi-param FWAD Jacobian mismatch at (" << i << "," << j << ")";
}

///////////////////////////////////////////////////////////////////////////////
// Tests — Hessian consistency through FD/FWAD wrappers
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, FD_HessianConsistency) {
    BrachistochroneFD fd(9.81);

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;
    Eigen::VectorXd lm = deterministic_random_vector(3, 500, -1.0, 1.0);

    // FD Hessian: check symmetry and adjoint gradient consistency
    verify_hessian_consistency(fd, x, lm, 1e-5);
}

TEST_F(VFCompositionTest, FWAD_HessianConsistency) {
    BrachistochroneFWAD fwad(9.81);

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;
    Eigen::VectorXd lm = deterministic_random_vector(3, 501, -1.0, 1.0);

    // Forward-AD Hessian: tighter tolerance than FD
    verify_hessian_consistency(fwad, x, lm, 1e-10);
}

///////////////////////////////////////////////////////////////////////////////
// Tests — Dynamic-size ODE wrapping (IRC == -1)
//
// Regression test for the FD step re-initialization fix. When
// ODE_DerivModeWrapper wraps a dynamic-size ODE (IRC == -1), the FD step
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

///////////////////////////////////////////////////////////////////////////////
// Tests — Dynamic-size ODE wrapping with AutodiffFwd (IRC == -1)
//
// Exercises the AutodiffFwd path through ODE_DerivModeWrapper with a
// dynamic-size inner ODE. The FD path is tested above; this verifies
// that set_io_rows works correctly for the autodiff derivative mode too.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, DynamicODE_FWAD_ComputeCorrect) {
    DynamicBrachFWAD fwad;
    EXPECT_EQ(fwad.input_rows(), 5);
    EXPECT_EQ(fwad.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    fwad.compute(x, fx);

    double v = 5.0, theta = std::numbers::pi / 4.0;
    EXPECT_NEAR(fx[0], std::sin(theta) * v, 1e-12);
    EXPECT_NEAR(fx[1], -std::cos(theta) * v, 1e-12);
    EXPECT_NEAR(fx[2], 9.81 * std::cos(theta), 1e-12);
}

TEST_F(VFCompositionTest, DynamicODE_FWAD_JacobianMatchesAnalytic) {
    Brachistochrone analytic(9.81);
    DynamicBrachFWAD fwad;

    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;

    Eigen::VectorXd fx_a(3);
    Eigen::MatrixXd jx_a(3, 5);
    fx_a.setZero();
    jx_a.setZero();
    analytic.compute_jacobian(x, fx_a, jx_a);

    Eigen::VectorXd fx_fwad(3);
    Eigen::MatrixXd jx_fwad(3, 5);
    fx_fwad.setZero();
    jx_fwad.setZero();
    fwad.compute_jacobian(x, fx_fwad, jx_fwad);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_NEAR(jx_a(i, j), jx_fwad(i, j), 1e-13)
                << "Dynamic FWAD Jacobian mismatch at (" << i << "," << j << ")";
}

TEST_F(VFCompositionTest, DynamicODE_FWAD_HessianConsistency) {
    DynamicBrachFWAD fwad;
    Eigen::VectorXd x(5);
    x << 0.5, 10, 5, 0.1, std::numbers::pi / 4.0;
    Eigen::VectorXd lm = deterministic_random_vector(3, 503, -1.0, 1.0);
    verify_hessian_consistency(fwad, x, lm, 1e-10);
}
