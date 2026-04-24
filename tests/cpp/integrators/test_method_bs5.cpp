///////////////////////////////////////////////////////////////////////////////
// BS5 method validation against OrdinaryDiffEq.jl reference
///////////////////////////////////////////////////////////////////////////////

#include "regression/regression_utils.h"
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

#ifndef JULIA_REFERENCE_DIR
#error "JULIA_REFERENCE_DIR must be defined at compile time"
#endif

namespace {

struct JuliaRef {
    Eigen::VectorXd uf;
    int nsteps;
    double walltime;
};

JuliaRef read_julia_ref(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("Cannot open Julia ref: " + path);
    int32_t n;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    Eigen::VectorXd uf(n);
    f.read(reinterpret_cast<char *>(uf.data()), n * sizeof(double));
    int32_t nsteps;
    f.read(reinterpret_cast<char *>(&nsteps), sizeof(nsteps));
    double walltime;
    f.read(reinterpret_cast<char *>(&walltime), sizeof(walltime));
    return {uf, nsteps, walltime};
}

std::string julia_ref_path(const std::string &filename) {
    return std::string(JULIA_REFERENCE_DIR) + "/test/reference_outputs/" + filename;
}

} // namespace

class BS5Test : public VectorFunctionFixture {};

TEST_F(BS5Test, TwoBodyMatchesJulia) {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::BS5, 10.0);
    integ.set_abs_tol(1e-10);
    integ.set_rel_tol(1e-11);

    auto x0 = leo_circular_x0();
    double tf = leo_period() / 4.0;
    auto xf = integ.integrate(x0, tf);

    auto ref = read_julia_ref(julia_ref_path("bs5_twobody.bin"));

    // 5th-order method at atol=1e-10, rtol=1e-11: expect ~1e-8 agreement.
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(xf[i], ref.uf[i], 1e-6)
            << "BS5 two-body state[" << i << "] mismatch: Tycho=" << xf[i]
            << " Julia=" << ref.uf[i];
    }
}

TEST_F(BS5Test, CR3BPMatchesJulia) {
    CR3BP_ODE ode(MU_CR3BP);
    Integrator<CR3BP_ODE> integ(ode, IVPAlg::BS5, 0.01);
    integ.set_abs_tol(1e-10);
    integ.set_rel_tol(1e-11);

    auto x0 = cr3bp_l1_x0();
    double tf = 2.0;
    auto xf = integ.integrate(x0, tf);

    auto ref = read_julia_ref(julia_ref_path("bs5_crtbp.bin"));

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(xf[i], ref.uf[i], 1e-6)
            << "BS5 CR3BP state[" << i << "] mismatch: Tycho=" << xf[i]
            << " Julia=" << ref.uf[i];
    }
}
