// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Zero-interval integration tests: pin the contract for integrate(x0, t0),
// integrate_dense, integrate_stm, and batch integrate when tf == t0.
// Without the scalar/SIMD short-circuit in integrate_impl / integrate_impl_vectorized,
// h==0 reaches stepper_compute and divides by zero in the FSAL / midpoint
// derivative reconstruction, producing NaN in `derivs`.

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

namespace {

Eigen::Vector3d sho_x0() {
    Eigen::Vector3d x;
    x << 1.0, 0.0, 0.0; // x=1, v=0, t=0
    return x;
}

} // namespace

// -----------------------------------------------------------------------------
// integrate(x, x[t_var]) returns x bit-identically across all 8 methods.
// -----------------------------------------------------------------------------
class ZeroIntervalIntegrate : public VectorFunctionFixture,
                              public ::testing::WithParamInterface<IVPAlg> {};

TEST_P(ZeroIntervalIntegrate, ReturnsInputState) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, GetParam(), 0.01);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    auto x0 = sho_x0();
    auto xf = integ.integrate(x0, x0[2]);

    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(xf[i], x0[i]) << "component " << i;
        EXPECT_TRUE(std::isfinite(xf[i])) << "component " << i << " not finite";
    }
}

INSTANTIATE_TEST_SUITE_P(AllMethods, ZeroIntervalIntegrate,
                         ::testing::Values(IVPAlg::DOPRI54, IVPAlg::DOPRI87, IVPAlg::Tsit5,
                                           IVPAlg::BS3, IVPAlg::BS5, IVPAlg::Vern7, IVPAlg::Vern8,
                                           IVPAlg::Vern9));

// -----------------------------------------------------------------------------
// integrate_dense produces a single-point trajectory with finite derivatives
// (no NaN leaked from the h==0 FSAL path).
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, ZeroInterval_DenseReturnsSinglePoint) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-12);

    auto x0 = sho_x0();
    auto xs = integ.integrate_dense(x0, x0[2]);

    ASSERT_GE(xs.size(), 1u);
    for (const auto &s : xs) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_DOUBLE_EQ(s[i], x0[i]);
            EXPECT_TRUE(std::isfinite(s[i]));
        }
    }
}

// -----------------------------------------------------------------------------
// integrate_stm(x, x[t_var]) returns the identity STM (preserves pre-existing
// STMIdentityAtZeroTime contract while eliminating the underlying NaN).
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, ZeroInterval_STMIsIdentity) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-12);

    auto x0 = sho_x0();
    auto [xf, stm] = integ.integrate_stm(x0, x0[2]);

    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(xf[i], x0[i]);
    }
    EXPECT_NEAR(stm(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(stm(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(stm(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(stm(1, 0), 0.0, 1e-12);
    EXPECT_TRUE(stm.allFinite()) << "STM must not contain NaN/Inf";
}

// -----------------------------------------------------------------------------
// SIMD batch path: a mix of zero-interval and normal lanes. Zero-interval lanes
// must return their input state; normal lanes must integrate correctly in the
// same batch.
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, ZeroInterval_BatchMixedLanes) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    std::vector<Eigen::Vector3d> xs = {sho_x0(), sho_x0(), sho_x0()};
    xs[1][2] = 0.5; // lane 1 starts at t=0.5 (still H != 0 below)
    Eigen::VectorXd tfs(3);
    tfs << 0.0, 0.5, 1.0; // lane 0: H==0; lane 1: H==0; lane 2: H==1

    auto results = integ.integrate(xs, tfs);

    ASSERT_EQ(results.size(), 3u);
    // Lane 0: zero-interval — expect exact input
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(results[0][i], xs[0][i]) << "lane 0 component " << i;
    }
    // Lane 1: zero-interval — expect exact input
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(results[1][i], xs[1][i]) << "lane 1 component " << i;
    }
    // Lane 2: normal SHO integration to t=1
    EXPECT_NEAR(results[2][0], std::cos(1.0), 1e-9);
    EXPECT_NEAR(results[2][1], -std::sin(1.0), 1e-9);
    EXPECT_NEAR(results[2][2], 1.0, 1e-12);
}

// -----------------------------------------------------------------------------
// SIMD batch path where ALL lanes are zero-interval — exercises the
// numrunning==0 exit at the top of the SIMD loop.
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, ZeroInterval_BatchAllZeroLanes) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::Vern7, 0.01);

    std::vector<Eigen::Vector3d> xs = {sho_x0(), sho_x0()};
    Eigen::VectorXd tfs(2);
    tfs << 0.0, 0.0;

    auto results = integ.integrate(xs, tfs);
    ASSERT_EQ(results.size(), 2u);
    for (size_t k = 0; k < 2; ++k) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_DOUBLE_EQ(results[k][i], xs[k][i]);
            EXPECT_TRUE(std::isfinite(results[k][i]));
        }
    }
}
