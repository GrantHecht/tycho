///////////////////////////////////////////////////////////////////////////////
// Error norm unit tests
//
// Validates scaled_residuals() and error_norm() against Julia's:
//   residual_i = ũ_i / (α_i + max(|u₀_i|,|u₁_i|) · ρ_i)
//   RMS norm = sqrt(sum(res_i²) / n)
//   MAX norm = max_i |res_i|
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/integrators/error_norm.h"
#include <Eigen/Core>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using tycho::integrators::error_norm;
using tycho::integrators::ErrorNormType;
using tycho::integrators::scaled_residuals;

TEST(ErrorNormTest, JuliaResidualUsesMaxOfBeforeAfter) {
    // u0 = 10, u1 = 5 → max = 10 (before-step is larger)
    Eigen::VectorXd u0(1), u1(1), utilde(1), atol(1), rtol(1);
    u0 << 10.0;
    u1 << 5.0;
    utilde << 0.01;
    atol << 1e-6;
    rtol << 1e-3;
    Eigen::VectorXd res = scaled_residuals(utilde, u0, u1, atol, rtol);
    // Expected: 0.01 / (1e-6 + 10 * 1e-3) = 0.01 / 0.010001 ≈ 0.99990001
    EXPECT_NEAR(res[0], 0.01 / (1e-6 + 10.0 * 1e-3), 1e-15);
}

TEST(ErrorNormTest, JuliaResidualBeforeSmallerAfterLarger) {
    Eigen::VectorXd u0(1), u1(1), utilde(1), atol(1), rtol(1);
    u0 << 5.0;
    u1 << 10.0;
    utilde << 0.01;
    atol << 1e-6;
    rtol << 1e-3;
    Eigen::VectorXd res = scaled_residuals(utilde, u0, u1, atol, rtol);
    // Expected: 0.01 / (1e-6 + 10 * 1e-3) — max(5,10)=10
    EXPECT_NEAR(res[0], 0.01 / (1e-6 + 10.0 * 1e-3), 1e-15);
}

TEST(ErrorNormTest, ZeroStateUsesAtolOnly) {
    Eigen::VectorXd u0(1), u1(1), utilde(1), atol(1), rtol(1);
    u0 << 0.0;
    u1 << 0.0;
    utilde << 1e-9;
    atol << 1e-6;
    rtol << 1e-3;
    Eigen::VectorXd res = scaled_residuals(utilde, u0, u1, atol, rtol);
    EXPECT_NEAR(res[0], 1e-9 / 1e-6, 1e-15);
}

TEST(ErrorNormTest, RMSOverComponents) {
    Eigen::VectorXd res(3);
    res << 1.0, 2.0, 2.0;
    // RMS = sqrt((1 + 4 + 4) / 3) = sqrt(3)
    double n = error_norm(res, ErrorNormType::RMS);
    EXPECT_NEAR(n, std::sqrt(3.0), 1e-15);
}

TEST(ErrorNormTest, MaxOverComponents) {
    Eigen::VectorXd res(3);
    res << 1.0, -3.0, 2.0;
    double n = error_norm(res, ErrorNormType::MAX);
    EXPECT_NEAR(n, 3.0, 1e-15);
}

TEST(ErrorNormTest, VectorizedPerComponentTolerances) {
    Eigen::VectorXd u0(3), u1(3), utilde(3), atol(3), rtol(3);
    u0 << 1.0, 2.0, 3.0;
    u1 << 1.1, 2.2, 3.3;
    utilde << 0.01, 0.02, 0.03;
    atol << 1e-8, 1e-9, 1e-10;
    rtol << 1e-4, 1e-5, 1e-6;
    Eigen::VectorXd res = scaled_residuals(utilde, u0, u1, atol, rtol);
    for (int i = 0; i < 3; ++i) {
        double denom = atol[i] + std::max(std::abs(u0[i]), std::abs(u1[i])) * rtol[i];
        EXPECT_NEAR(res[i], utilde[i] / denom, 1e-15) << "i=" << i;
    }
}

// MAX branch must not call maxCoeff() on an empty vector (UB / assert) —
// INTEGRATORS_REVIEW §3.5.
TEST(ErrorNormTest, MaxBranchEmptyVectorReturnsZero) {
    Eigen::VectorXd empty(0);
    EXPECT_DOUBLE_EQ(error_norm(empty, ErrorNormType::MAX), 0.0);
}

// MAX branch must PROPAGATE a NaN, not silently drop it. Eigen's maxCoeff()
// returns whichever operand is not NaN by argument order (like C++ std::max),
// so a NaN at a non-first index would be laundered to a finite value — the exact
// Julia-vs-C++ trap (Julia max(x,NaN) === NaN). The explicit isNaN().any() guard
// in error_norm.h must force NaN out regardless of position.
TEST(ErrorNormTest, MaxBranchPropagatesNaN) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd tail_nan(2);
    tail_nan << 1.0, nan; // NaN at the LAST index — the case maxCoeff drops
    EXPECT_TRUE(std::isnan(error_norm(tail_nan, ErrorNormType::MAX)));

    Eigen::VectorXd head_nan(2);
    head_nan << nan, 1.0; // NaN at the FIRST index
    EXPECT_TRUE(std::isnan(error_norm(head_nan, ErrorNormType::MAX)));
}

// Inf must survive as Inf (a large-but-real error), NOT be turned into NaN or
// dropped. Distinguishes the NaN guard from a blanket non-finite coercion.
TEST(ErrorNormTest, MaxBranchKeepsInfAsInf) {
    Eigen::VectorXd res(2);
    res << std::numeric_limits<double>::infinity(), 1.0;
    EXPECT_TRUE(std::isinf(error_norm(res, ErrorNormType::MAX)));
}

// RMS must also propagate a NaN (squaredNorm: NaN² → NaN → sqrt → NaN). Pins the
// companion guarantee the drivers rely on ("one scalar isfinite subsumes both").
TEST(ErrorNormTest, RMSPropagatesNaN) {
    Eigen::VectorXd res(2);
    res << 1.0, std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(error_norm(res, ErrorNormType::RMS)));
}
