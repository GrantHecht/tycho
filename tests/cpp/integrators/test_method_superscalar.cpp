///////////////////////////////////////////////////////////////////////////////
// SuperScalar (SIMD batch) path coverage for SP2 methods
//
// Verifies Tsit5, BS3, BS5, Vern7, Vern8, Vern9 all pass through the
// integrate_impl_vectorized path with matching results vs the scalar path.
///////////////////////////////////////////////////////////////////////////////

#include "regression/regression_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

namespace {

template <IVPAlg Alg>
void run_batch_vs_scalar(double atol, double rtol, double tol) {
    using State = typename Integrator<Kepler>::IntegRet;
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, Alg, 10.0);
    integ.set_abs_tol(atol);
    integ.set_rel_tol(rtol);

    double r0 = 7000.0;
    double v0 = std::sqrt(MU_EARTH / r0);
    std::vector<State> x0s;
    Eigen::VectorXd tfs(4);
    for (int i = 0; i < 4; ++i) {
        State x0;
        x0 << r0 + 100.0 * i, 0, 0, 0, v0, 0, 0;
        x0s.push_back(x0);
        tfs[i] = 2000.0 + 100.0 * i;
    }

    integ.vectorize_batch_calls_ = true;
    auto xfs_vec = integ.integrate(x0s, tfs);

    integ.vectorize_batch_calls_ = false;
    for (size_t i = 0; i < x0s.size(); ++i) {
        auto xf_scalar = integ.integrate(x0s[i], tfs[i]);
        for (int j = 0; j < xf_scalar.size(); ++j) {
            EXPECT_NEAR(xfs_vec[i][j], xf_scalar[j], tol)
                << "SIMD vs scalar mismatch: method=" << int(Alg) << " traj=" << i
                << " j=" << j;
        }
    }
}

} // namespace

class SuperScalarMethodTest : public VectorFunctionFixture {};

TEST_F(SuperScalarMethodTest, Tsit5) {
    run_batch_vs_scalar<IVPAlg::Tsit5>(1e-10, 1e-11, 1e-7);
}

TEST_F(SuperScalarMethodTest, BS3) {
    run_batch_vs_scalar<IVPAlg::BS3>(1e-6, 1e-7, 1e-4);
}

TEST_F(SuperScalarMethodTest, BS5) {
    run_batch_vs_scalar<IVPAlg::BS5>(1e-10, 1e-11, 1e-7);
}

TEST_F(SuperScalarMethodTest, Vern7) {
    run_batch_vs_scalar<IVPAlg::Vern7>(1e-11, 1e-12, 1e-8);
}

TEST_F(SuperScalarMethodTest, Vern8) {
    run_batch_vs_scalar<IVPAlg::Vern8>(1e-12, 1e-13, 1e-9);
}

TEST_F(SuperScalarMethodTest, Vern9) {
    run_batch_vs_scalar<IVPAlg::Vern9>(1e-12, 1e-13, 1e-9);
}
