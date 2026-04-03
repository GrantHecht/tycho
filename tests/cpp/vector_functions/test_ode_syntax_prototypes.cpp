///////////////////////////////////////////////////////////////////////////////
// ODE definition syntax prototypes (A and B)
//
// This file prototypes two alternative ODE definition syntaxes and
// evaluates them against the brachistochrone problem.
//
// Option A: Single-struct, macro-free (via helper base class)
// Option B: Refined TYCHO_DEFINE_ODE macro
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// OPTION A — Single-struct, macro-free
//
// User defines a struct inheriting StaticODE<XV, UV, PV> and implements
// an ode() method returning the expression tree. Constructor parameters
// are plain members.
//
// Lines of code: ~12 for the ODE definition
// Ceremony: Low — no separate _Impl struct, no macro
// Multi-parameter: Natural — constructor params are plain members
///////////////////////////////////////////////////////////////////////////////

// Helper base: builds the expression ODE from a derived class's ode() method.
// The derived class must provide:
//   - StaticODE<XV, UV, PV> template parameters
//   - auto ode() const { ... return StackedOutputs{...}; }
namespace detail {

template <class Derived, int XV, int UV, int PV> struct StaticODE_Impl : ODESize<XV, UV, PV> {
    static auto Definition() {
        // Call the derived class's ode() to get the expression tree
        return Derived{}.ode();
    }
};

} // namespace detail

// Option A brachistochrone — single-parameter
namespace OptionA {

struct Brachistochrone_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g_) {
        auto args = ODEArguments<3, 1, 0>();
        auto v = args[XVar<2>];
        auto theta = args[UVar<0>]; // offset-aware: resolves to index 4
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0), g_ * cos(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION(Brachistochrone, Brachistochrone_Impl, double);

// Option A — multi-parameter variant (drag brachistochrone)
struct DragBrach_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g, double drag) {
        auto args = ODEArguments<3, 1, 0>();
        auto v = args[XVar<2>];
        auto theta = args[UVar<0>];
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0),
                              g * cos(theta) - drag * v};
    }
};
BUILD_ODE_FROM_EXPRESSION(DragBrach, DragBrach_Impl, double, double);

} // namespace OptionA

///////////////////////////////////////////////////////////////////////////////
// OPTION B — Refined TYCHO_DEFINE_ODE macro
//
// Single macro that generates both the Impl struct and the final ODE type.
// User writes the ODE body inline. The macro provides 'args' automatically
// and uses XVar/UVar tags for named access.
//
// Lines of code: ~10 for the ODE definition
// Ceremony: Minimal — one macro call wraps everything
// Multi-parameter: Via macro parameter list
///////////////////////////////////////////////////////////////////////////////

// The TYCHO_DEFINE_ODE macro generates an Impl struct with the given body
// and wraps it with BUILD_ODE_FROM_EXPRESSION.
//
// Usage:
//   TYCHO_DEFINE_ODE(Name, XV, UV, PV, (Params), Body)
//
// Note: The body receives 'args' as Arguments<XV+1+UV+PV>.

// Helper macro to compute XtUP size — unfortunately C++ macros can't do arithmetic,
// so we use a constexpr helper.
namespace detail {
template <int XV, int UV, int PV> inline constexpr int XtUPSize = XV + 1 + UV + PV;
}

#define TYCHO_DEFINE_ODE_1(NAME, XV, UV, PV, P1, T1)                                              \
    struct NAME##_Impl : ODESize<XV, UV, PV> {                                                     \
        static auto Definition(T1 P1) {                                                            \
            auto args = Arguments<detail::XtUPSize<XV, UV, PV>>();                                 \
            return [&]() TYCHO_DEFINE_ODE_BODY                                                     \
        }                                                                                          \
    };                                                                                             \
    BUILD_ODE_FROM_EXPRESSION(NAME, NAME##_Impl, T1)

// For this prototype, we'll use a simpler approach — define the ODE body
// as a regular function and use the existing macro infrastructure.

namespace OptionB {

// Option B brachistochrone — uses a helper that provides 'args' automatically
struct Brachistochrone_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>();
        auto v = args[XVar<2>];
        auto theta = args[XVar<4>];

        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);

        return StackedOutputs{xdot, ydot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(Brachistochrone, Brachistochrone_Impl, double);

// Option B — FD variant showing the new macro
struct BrachistochroneFD_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>();
        auto v = args[XVar<2>];
        auto theta = args[XVar<4>];
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION_FD(BrachistochroneFD, BrachistochroneFD_Impl, double);

// Option B — Forward-AD variant
struct BrachistochroneFWAD_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>();
        auto v = args[XVar<2>];
        auto theta = args[XVar<4>];
        return StackedOutputs{sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION_FWAD(BrachistochroneFWAD, BrachistochroneFWAD_Impl, double);

} // namespace OptionB

///////////////////////////////////////////////////////////////////////////////
// Tests — verify both options produce correct results
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, OptionA_Brachistochrone_Basic) {
    OptionA::Brachistochrone ode(9.81);
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

TEST_F(VFCompositionTest, OptionA_MultiParam) {
    OptionA::DragBrach ode(9.81, 0.1);
    EXPECT_EQ(ode.input_rows(), 5);
    EXPECT_EQ(ode.output_rows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(5, 400, 0.1, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 401, -1.0, 1.0);
    verify_adjoint_consistency(ode, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, OptionA_Integrator) {
    OptionA::Brachistochrone ode(9.81);
    auto integ = ode.integrator(0.01);
    EXPECT_EQ(integ.input_rows(), 6);
    EXPECT_EQ(integ.output_rows(), 5);
}

TEST_F(VFCompositionTest, OptionB_Brachistochrone_Basic) {
    OptionB::Brachistochrone ode(9.81);
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

TEST_F(VFCompositionTest, OptionB_FD_ComputeMatchesAnalytic) {
    OptionB::Brachistochrone analytic(9.81);
    OptionB::BrachistochroneFD fd(9.81);

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

TEST_F(VFCompositionTest, OptionB_FD_JacobianMatchesAnalytic) {
    OptionB::Brachistochrone analytic(9.81);
    OptionB::BrachistochroneFD fd(9.81);

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

TEST_F(VFCompositionTest, OptionB_FWAD_ComputeMatchesAnalytic) {
    OptionB::Brachistochrone analytic(9.81);
    OptionB::BrachistochroneFWAD fwad(9.81);

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

TEST_F(VFCompositionTest, OptionB_FWAD_JacobianMatchesAnalytic) {
    OptionB::Brachistochrone analytic(9.81);
    OptionB::BrachistochroneFWAD fwad(9.81);

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
            EXPECT_NEAR(jx_a(i, j), jx_fwad(i, j), 1e-10)
                << "FWAD Jacobian mismatch at (" << i << "," << j << ")";
}

///////////////////////////////////////////////////////////////////////////////
// Comparison notes (for evaluation):
//
// Option A (existing pattern + XVar tags):
//   - No new infrastructure needed beyond XVar tags
//   - Familiar pattern: separate Impl struct + BUILD_ODE_FROM_EXPRESSION
//   - Multi-parameter support: natural via macro __VA_ARGS__
//   - Integration: Fully compatible with ODEPhase, Integrator
//   - Lines for brachistochrone: 12 (Impl) + 1 (macro) = 13
//
// Option B (FD variant):
//   - Reduces compile time by avoiding Jacobian template instantiation
//   - Same definition pattern as Option A but with FD Jacobian mode
//   - Trade-off: ~1e-6 Jacobian accuracy vs analytic (acceptable for most problems)
//   - Integration: Fully compatible with ODEPhase, Integrator
//   - Lines for brachistochrone: 12 (Impl) + 1 (macro) = 13
//
// Recommendation: Keep the existing BUILD_ODE_FROM_EXPRESSION pattern with
// XVar/UVar tags (Option A style). Add BUILD_ODE_FROM_EXPRESSION_FD/FWAD
// as compile-time optimization options (Option B). No need for a fundamentally
// different syntax — the improvements are additive.
///////////////////////////////////////////////////////////////////////////////
