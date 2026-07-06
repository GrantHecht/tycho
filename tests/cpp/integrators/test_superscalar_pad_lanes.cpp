///////////////////////////////////////////////////////////////////////////////
// SuperScalar pad-lane time slots (INTEGRATORS_REVIEW §3.4). With t0 == 0 and a
// batch size that forces pad lanes (ntrajs not a multiple of the SIMD width),
// an unpadded pad lane would see h == 0 and feed Inf/NaN into a `1/h` scaling.
// The effect is genuinely lane-local (Eigen per-lane arithmetic; pad lanes
// [Vmax, width) are never read back), so this cannot be a red-green output test.
// To make it meaningful we deliberately choose configurations that DO route the
// padded h through a division: an FSAL method (DOPRI54 executes k_fsal_ = k*1/h
// in step<false>) for plain integrate, and integrate_stm2 (whose adjoint-Hessian
// path has the only STM-side 1/h) — then assert the real-lane result stays
// finite and matches the scalar path.
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <vector>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

class SuperScalarPadLaneTest : public VectorFunctionFixture {};

// ParallelDriver pad lanes, FSAL method (routes padded h through the 1/h FSAL
// cache): single trajectory -> (width - 1) pad lanes.
TEST_F(SuperScalarPadLaneTest, VectorizedBatchIntegrateMatchesScalar) {
    SHO ode(0.0);
    std::vector<Eigen::Vector3d> x0s(1, Eigen::Vector3d(1.0, 0.0, 0.0)); // t0 == 0
    Eigen::VectorXd tfs(1);
    tfs << 3.14159;

    Integrator<SHO> scalar_integ(ode, IVPAlg::DOPRI54, 0.01);
    scalar_integ.set_abs_tol(1e-12);
    scalar_integ.set_rel_tol(1e-12);
    scalar_integ.vectorize_batch_calls_ = false;
    auto scalar_res = scalar_integ.integrate(x0s, tfs);

    Integrator<SHO> vec_integ(ode, IVPAlg::DOPRI54, 0.01);
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

// StmDriver pad lanes via integrate_stm2 (the adjoint-Hessian path is the only
// STM-side 1/h division): single trajectory -> pad lanes.
TEST_F(SuperScalarPadLaneTest, VectorizedBatchIntegrateStm2MatchesScalar) {
    SHO ode(0.0);
    std::vector<Eigen::Vector3d> x0s(1, Eigen::Vector3d(1.0, 0.0, 0.0)); // t0 == 0
    Eigen::VectorXd tfs(1);
    tfs << 3.14159;
    std::vector<Eigen::Vector3d> lfs(1, Eigen::Vector3d(1.0, 1.0, 0.0)); // adjoint seed

    Integrator<SHO> scalar_integ(ode, IVPAlg::DOPRI54, 0.01);
    scalar_integ.set_abs_tol(1e-12);
    scalar_integ.set_rel_tol(1e-12);
    scalar_integ.vectorize_batch_calls_ = false;
    auto scalar_res = scalar_integ.integrate_stm2(x0s, tfs, lfs);

    Integrator<SHO> vec_integ(ode, IVPAlg::DOPRI54, 0.01);
    vec_integ.set_abs_tol(1e-12);
    vec_integ.set_rel_tol(1e-12);
    vec_integ.vectorize_batch_calls_ = true;
    auto vec_res = vec_integ.integrate_stm2(x0s, tfs, lfs);

    ASSERT_EQ(scalar_res.size(), 1u);
    ASSERT_EQ(vec_res.size(), 1u);
    const auto &[xf_s, J_s, H_s] = scalar_res[0];
    const auto &[xf_v, J_v, H_v] = vec_res[0];
    EXPECT_TRUE(xf_v.allFinite());
    EXPECT_TRUE(J_v.allFinite());
    EXPECT_TRUE(H_v.allFinite());
    EXPECT_NEAR((xf_v - xf_s).norm(), 0.0, 1e-8);
    EXPECT_NEAR((J_v - J_s).norm(), 0.0, 1e-6);
    EXPECT_NEAR((H_v - H_s).norm(), 0.0, 1e-6)
        << "Vectorized (pad-lane) STM2 adjoint-Hessian must match the scalar path.";
}
