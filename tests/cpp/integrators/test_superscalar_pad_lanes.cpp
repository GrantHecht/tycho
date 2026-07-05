///////////////////////////////////////////////////////////////////////////////
// SuperScalar pad-lane time slots (INTEGRATORS_REVIEW §3.4). With t0 == 0 and a
// batch size that forces pad lanes (ntrajs not a multiple of the SIMD width),
// pad lanes must not poison the run. The effect is lane-local (Eigen per-lane
// arithmetic), so this is a consistency + finiteness smoke test, not red-green:
// it exercises the vectorized ParallelDriver (integrate) and StmDriver
// (integrate_stm) pad paths with t0 == 0 and asserts the vectorized result is
// finite and matches the scalar path.
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <numbers>
#include <vector>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

class SuperScalarPadLaneTest : public VectorFunctionFixture {};

// ParallelDriver pad lanes (integrate batch): single trajectory -> pad lanes.
TEST_F(SuperScalarPadLaneTest, VectorizedBatchIntegrateMatchesScalar) {
    SHO ode(0.0);
    std::vector<Eigen::Vector3d> x0s(1, Eigen::Vector3d(1.0, 0.0, 0.0)); // t0 == 0
    Eigen::VectorXd tfs(1);
    tfs << 3.14159;

    Integrator<SHO> scalar_integ(ode, IVPAlg::DOPRI87, 0.01);
    scalar_integ.set_abs_tol(1e-12);
    scalar_integ.set_rel_tol(1e-12);
    scalar_integ.vectorize_batch_calls_ = false;
    auto scalar_res = scalar_integ.integrate(x0s, tfs);

    Integrator<SHO> vec_integ(ode, IVPAlg::DOPRI87, 0.01);
    vec_integ.set_abs_tol(1e-12);
    vec_integ.set_rel_tol(1e-12);
    vec_integ.vectorize_batch_calls_ = true; // default; forces the SIMD/pad path
    auto vec_res = vec_integ.integrate(x0s, tfs);

    ASSERT_EQ(scalar_res.size(), 1u);
    ASSERT_EQ(vec_res.size(), 1u);
    EXPECT_TRUE(vec_res[0].allFinite());
    EXPECT_NEAR((vec_res[0] - scalar_res[0]).norm(), 0.0, 1e-8)
        << "Vectorized (pad-lane) integrate must match the scalar path.";
}

// StmDriver pad lanes (integrate_stm batch): single trajectory -> pad lanes.
TEST_F(SuperScalarPadLaneTest, VectorizedBatchIntegrateStmMatchesScalar) {
    SHO ode(0.0);
    std::vector<Eigen::Vector3d> x0s(1, Eigen::Vector3d(1.0, 0.0, 0.0)); // t0 == 0
    Eigen::VectorXd tfs(1);
    tfs << 3.14159;

    Integrator<SHO> scalar_integ(ode, IVPAlg::DOPRI87, 0.01);
    scalar_integ.set_abs_tol(1e-12);
    scalar_integ.set_rel_tol(1e-12);
    scalar_integ.vectorize_batch_calls_ = false;
    auto scalar_res = scalar_integ.integrate_stm(x0s, tfs);

    Integrator<SHO> vec_integ(ode, IVPAlg::DOPRI87, 0.01);
    vec_integ.set_abs_tol(1e-12);
    vec_integ.set_rel_tol(1e-12);
    vec_integ.vectorize_batch_calls_ = true;
    auto vec_res = vec_integ.integrate_stm(x0s, tfs);

    ASSERT_EQ(scalar_res.size(), 1u);
    ASSERT_EQ(vec_res.size(), 1u);
    const auto &[xf_s, J_s] = scalar_res[0];
    const auto &[xf_v, J_v] = vec_res[0];
    EXPECT_TRUE(xf_v.allFinite());
    EXPECT_TRUE(J_v.allFinite());
    EXPECT_NEAR((xf_v - xf_s).norm(), 0.0, 1e-8);
    EXPECT_NEAR((J_v - J_s).norm(), 0.0, 1e-6)
        << "Vectorized (pad-lane) STM must match the scalar path.";
}
