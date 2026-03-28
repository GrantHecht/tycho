///////////////////////////////////////////////////////////////////////////////
// Shared test utilities for the Tycho C++ test suite
//
// Reusable helpers: deterministic RNG, VectorFunction fixture base,
// derivative cross-checks (analytical jacobian, adjoint consistency,
// hessian symmetry).
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>
#include <gtest/gtest.h>
#include <numbers>
#include <random>

// Bring all sub-namespaces into file scope so every test translation unit
// that includes this header can use unqualified names.
using namespace tycho;
using namespace tycho::utils;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;

namespace TychoTest {

///////////////////////////////////////////////////////////////////////////////
// Deterministic pseudo-random vector
///////////////////////////////////////////////////////////////////////////////

inline Eigen::VectorXd deterministic_random_vector(int n, unsigned seed = 42, double lo = -10.0,
                                                   double hi = 10.0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i)
        v[i] = dist(rng);
    return v;
}

///////////////////////////////////////////////////////////////////////////////
// VectorFunction test fixture — initialises BumpAllocator once
///////////////////////////////////////////////////////////////////////////////

class VectorFunctionFixture : public ::testing::Test {
  protected:
    static void SetUpTestSuite() { BumpAllocator::resize(256, 256); }
};

///////////////////////////////////////////////////////////////////////////////
// RAII guard to restore global thread count on scope exit.
// Ensures test failures don't leave stale thread counts for subsequent tests.
///////////////////////////////////////////////////////////////////////////////

struct ScopedThreadCount {
    int prev;
    explicit ScopedThreadCount(int n) : prev(tycho::utils::get_num_threads()) {
        tycho::utils::set_num_threads(n);
    }
    ~ScopedThreadCount() { tycho::utils::set_num_threads(prev); }
    ScopedThreadCount(const ScopedThreadCount &) = delete;
    ScopedThreadCount &operator=(const ScopedThreadCount &) = delete;
};

///////////////////////////////////////////////////////////////////////////////
// Derivative cross-check helpers
///////////////////////////////////////////////////////////////////////////////

/// Compare compute_jacobian output against a hand-computed analytical jacobian.
template <class Func>
void verify_jacobian_analytical(Func &func, const Eigen::VectorXd &x,
                                const Eigen::MatrixXd &expected_jx, double tol = 1e-12) {
    const int ir = func.input_rows();
    const int or_ = func.output_rows();
    Eigen::VectorXd fx(or_);
    Eigen::MatrixXd jx(or_, ir);
    fx.setZero();
    jx.setZero();
    func.compute_jacobian(x, fx, jx);

    ASSERT_EQ(jx.rows(), expected_jx.rows());
    ASSERT_EQ(jx.cols(), expected_jx.cols());
    for (int i = 0; i < or_; ++i) {
        for (int j = 0; j < ir; ++j) {
            EXPECT_NEAR(jx(i, j), expected_jx(i, j), tol)
                << "Jacobian mismatch at (" << i << "," << j << ")";
        }
    }
}

/// Cross-check two independent AD paths: forward-mode jacobian vs reverse-mode
/// adjoint gradient.  Verifies gx == jx^T * lm.
template <class Func>
void verify_adjoint_consistency(Func &func, const Eigen::VectorXd &x, const Eigen::VectorXd &lm,
                                double tol = 1e-12) {
    const int ir = func.input_rows();
    const int or_ = func.output_rows();

    // Forward: compute jacobian
    Eigen::VectorXd fx1(or_);
    Eigen::MatrixXd jx(or_, ir);
    fx1.setZero();
    jx.setZero();
    func.compute_jacobian(x, fx1, jx);

    // Reverse: compute adjoint gradient
    Eigen::VectorXd fx2(or_);
    Eigen::VectorXd gx(ir);
    fx2.setZero();
    gx.setZero();
    func.compute_adjointgradient(x, fx2, gx, lm);

    // Expected: gx_expected = J^T * lm
    Eigen::VectorXd gx_expected = jx.transpose() * lm;

    for (int j = 0; j < ir; ++j) {
        EXPECT_NEAR(gx[j], gx_expected[j], tol) << "Adjoint gradient mismatch at element " << j;
    }
}

/// Verify hessian symmetry and adjoint gradient consistency.
/// Calls compute_jacobian_adjointgradient_adjointhessian, then checks:
///  1) hx == hx^T  (symmetry)
///  2) gx == jx^T * lm  (adjoint consistency)
template <class Func>
void verify_hessian_consistency(Func &func, const Eigen::VectorXd &x, const Eigen::VectorXd &lm,
                                double tol = 1e-12) {
    const int ir = func.input_rows();
    const int or_ = func.output_rows();

    Eigen::VectorXd fx(or_);
    Eigen::MatrixXd jx(or_, ir);
    Eigen::VectorXd gx(ir);
    Eigen::MatrixXd hx(ir, ir);
    fx.setZero();
    jx.setZero();
    gx.setZero();
    hx.setZero();

    func.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

    // Check symmetry
    for (int i = 0; i < ir; ++i) {
        for (int j = i + 1; j < ir; ++j) {
            EXPECT_NEAR(hx(i, j), hx(j, i), tol)
                << "Hessian not symmetric at (" << i << "," << j << ")";
        }
    }

    // Check adjoint gradient consistency: gx == jx^T * lm
    Eigen::VectorXd gx_expected = jx.transpose() * lm;
    for (int j = 0; j < ir; ++j) {
        EXPECT_NEAR(gx[j], gx_expected[j], tol)
            << "Hessian adjoint gradient mismatch at element " << j;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Shared Brachistochrone ODE — used across multiple test files
///////////////////////////////////////////////////////////////////////////////

struct BrachODE_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>(); // [x, y, v, t, theta]
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();
        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);
        return StackedOutputs{xdot, ydot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(BrachODE, BrachODE_Impl, double);

} // namespace TychoTest
