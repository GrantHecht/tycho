// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// Phase 1 Enzyme Jacobian unit test.  Sanity-check the active
// DenseFirstDerivatives<..., EnzymeAD> specialization against central
// finite differences on the Brachistochrone ODE.
// =============================================================================

#include <gtest/gtest.h>
#include "enzyme_test_dynamics.h"

namespace {

// Sanity: the Enzyme variant produces a Jacobian consistent with central FD.
TEST(EnzymeJacobian, BrachODEEnzymeVsCentralFD) {
    tycho_enzyme_test::BrachEnzymeAD f(32.2);
    Eigen::Matrix<double, 5, 1> x;
    x << 1.0, 2.0, 3.0, 0.0, 0.5;

    Eigen::Matrix<double, 3, 1> fx;
    Eigen::Matrix<double, 3, 5> jx;
    fx.setZero();
    jx.setZero();
    f.compute_jacobian(x, fx, jx);

    // Central FD reference.
    constexpr double h = 1e-6;
    Eigen::Matrix<double, 3, 5> jx_fd;
    for (int i = 0; i < 5; i++) {
        Eigen::Matrix<double, 5, 1> xp = x, xm = x;
        xp[i] += h;
        xm[i] -= h;
        Eigen::Matrix<double, 3, 1> fp, fm;
        fp.setZero();
        fm.setZero();
        f.compute(xp, fp);
        f.compute(xm, fm);
        jx_fd.col(i) = (fp - fm) / (2.0 * h);
    }
    EXPECT_TRUE(jx.isApprox(jx_fd, 1e-7))
        << "Enzyme Jacobian:\n" << jx << "\nCentral FD:\n" << jx_fd;
}

} // namespace
