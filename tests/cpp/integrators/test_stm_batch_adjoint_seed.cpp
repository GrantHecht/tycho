///////////////////////////////////////////////////////////////////////////////
// Batch STM adjoint seeding tests
//
// Regression: the batch (SuperScalar) STM path must seed the FULL adjoint,
// including time/control/parameter slots (INTEGRATORS_REVIEW 1.1).
// Pre-fix the seed loop ran to ode.output_rows() (= x_vars), silently
// zeroing adjoint components in [x_vars, input_rows) — batch-only.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(IntegratorTest, BatchStm2AdjointSeedMatchesScalar) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.adaptive_ = false;  // identical step sequences scalar vs batch

    using OState = Integrator<SHO>::IntegRet;

    // N > SuperScalar width to cover full packs + remainder; varied tfs
    // exercise the heterogeneous-length tail-fusion path.
    const int N = 5;
    std::vector<OState> x0s(N), lfs(N);
    Eigen::VectorXd tfs(N);
    for (int i = 0; i < N; ++i) {
        OState x0;
        x0 << std::cos(0.3 * i), std::sin(0.3 * i), 0.0;
        x0s[i] = x0;
        tfs[i] = 1.0 + 0.25 * i;
        OState lf;
        lf << 0.7 + 0.1 * i, -0.4, 1.3;  // lf[t_var] nonzero — the bug's blind spot
        lfs[i] = lf;
    }

    integ.vectorize_batch_calls_ = false;
    auto scalar_res = integ.integrate_stm2(x0s, tfs, lfs);
    integ.vectorize_batch_calls_ = true;
    auto batch_res = integ.integrate_stm2(x0s, tfs, lfs);

    ASSERT_EQ(scalar_res.size(), batch_res.size());
    for (int i = 0; i < N; ++i) {
        const auto &[xf_s, J_s, H_s] = scalar_res[i];
        const auto &[xf_b, J_b, H_b] = batch_res[i];
        for (int r = 0; r < xf_s.size(); ++r)
            EXPECT_NEAR(xf_s[r], xf_b[r], 1e-8) << "traj " << i << " xf row " << r;
        for (int r = 0; r < J_s.rows(); ++r)
            for (int c = 0; c < J_s.cols(); ++c)
                EXPECT_NEAR(J_s(r, c), J_b(r, c), 1e-8)
                    << "traj " << i << " J(" << r << "," << c << ")";
        for (int r = 0; r < H_s.rows(); ++r)
            for (int c = 0; c < H_s.cols(); ++c)
                EXPECT_NEAR(H_s(r, c), H_b(r, c), 1e-8)
                    << "traj " << i << " H(" << r << "," << c << ")";
    }
}
