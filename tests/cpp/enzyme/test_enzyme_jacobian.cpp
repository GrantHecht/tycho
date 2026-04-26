// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// Phase 1 Enzyme Jacobian unit-test matrix.
//
// Compares the active DenseFirstDerivatives<..., EnzymeAD> specialization
// against:
//   - central finite differences (sanity baseline, BrachODE),
//   - autodiff forward-mode (lane-for-lane reference, BrachODE/CR3BP/MEE),
//   - mixed-mode pairing <EnzymeAD, AutodiffFwd> versus pure autodiff,
//   - GenericFunction<-1, -1> type erasure of an EnzymeAD VF.
// =============================================================================
//
// Numerical input choices:
//   - BrachODE is well-conditioned for any finite (x, y, v, t, theta), so a
//     hand-picked tuple is fine.
//   - CR3BP has singular regions where dvec.norm() or rvec.norm() vanishes
//     (the two primary bodies sit at x = -mu and x = 1 - mu).  We pick a
//     point well away from both to keep the Jacobian numerically stable.
//   - MEE has singularities at x[0] = 0 (sqrt) and the MEE state has a
//     denominator (1 + x1*cos(x5) + x2*sin(x5)).  We pick a point with x[0]
//     comfortably positive and small e1/e2 so the denominator stays away
//     from zero.
//
// All comparisons require 1e-12 absolute (isApprox) agreement.  The point of
// the matrix is to lock in lane-for-lane Enzyme/autodiff parity: do not relax
// these tolerances.
// =============================================================================

#include <gtest/gtest.h>

#include <tycho/tycho.h>

#include "enzyme_test_dynamics.h"

namespace {

// -----------------------------------------------------------------------------
// BrachODE: Enzyme vs central FD (kept as the original sanity baseline).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// BrachODE: Enzyme vs AutodiffFwd lane-for-lane.
// -----------------------------------------------------------------------------
TEST(EnzymeJacobian, BrachODEEnzymeVsAutodiff) {
    tycho_enzyme_test::BrachEnzymeAD fE(32.2);
    tycho_enzyme_test::BrachAutodiff fA(32.2);
    Eigen::Matrix<double, 5, 1> x;
    x << 1.0, 2.0, 3.0, 0.0, 0.5;

    Eigen::Matrix<double, 3, 1> fxE, fxA;
    Eigen::Matrix<double, 3, 5> jxE, jxA;
    fxE.setZero();
    fxA.setZero();
    jxE.setZero();
    jxA.setZero();
    fE.compute_jacobian(x, fxE, jxE);
    fA.compute_jacobian(x, fxA, jxA);

    EXPECT_TRUE(jxE.isApprox(jxA, 1e-12))
        << "Enzyme Jac:\n" << jxE << "\nAutodiff Jac:\n" << jxA;
    EXPECT_TRUE(fxE.isApprox(fxA, 1e-15))
        << "Primal mismatch.  Enzyme: " << fxE.transpose()
        << "  Autodiff: " << fxA.transpose();
}

// -----------------------------------------------------------------------------
// CR3BP: Enzyme vs AutodiffFwd.  Deterministic input, picked away from both
// primaries (x = -mu and x = 1 - mu) so |dvec| and |rvec| stay O(1).
// -----------------------------------------------------------------------------
TEST(EnzymeJacobian, CR3BPVsAutodiff) {
    constexpr double mu = 0.0123;
    tycho_enzyme_test::CR3BPEnzymeAD fE(mu);
    tycho_enzyme_test::CR3BPAutodiff fA(mu);

    Eigen::Matrix<double, 7, 1> x;
    // Position around L4-ish (well away from -mu and 1-mu primaries),
    // velocity small but nonzero, time arbitrary.
    x << 0.5, 0.7, 0.1,    // X
         0.05, -0.03, 0.02, // V
         0.0;              // t
    Eigen::Matrix<double, 6, 1> fxE, fxA;
    Eigen::Matrix<double, 6, 7> jxE, jxA;
    fxE.setZero();
    fxA.setZero();
    jxE.setZero();
    jxA.setZero();
    fE.compute_jacobian(x, fxE, jxE);
    fA.compute_jacobian(x, fxA, jxA);

    EXPECT_TRUE(jxE.isApprox(jxA, 1e-12))
        << "Enzyme Jac:\n" << jxE << "\nAutodiff Jac:\n" << jxA;
    EXPECT_TRUE(fxE.isApprox(fxA, 1e-15));
}

// -----------------------------------------------------------------------------
// MEE: Enzyme vs AutodiffFwd.  Deterministic input chosen to stay away from
// MEE singularities: x[0] (semi-latus-rectum-like) > 0, |e1|, |e2| small so
// (1 + x1*cos(L) + x2*sin(L)) stays well away from zero.
// -----------------------------------------------------------------------------
TEST(EnzymeJacobian, MEEVsAutodiff) {
    constexpr double mu = 1.0;
    tycho_enzyme_test::MEEEnzymeAD fE(mu);
    tycho_enzyme_test::MEEAutodiff fA(mu);

    Eigen::Matrix<double, 9, 1> x;
    // p, f, g, h, k, L, then three control-like inputs (u1, u2, u3).
    x << 1.2, 0.05, -0.03, 0.02, 0.01, 0.4, 0.1, 0.2, 0.3;

    Eigen::Matrix<double, 6, 1> fxE, fxA;
    Eigen::Matrix<double, 6, 9> jxE, jxA;
    fxE.setZero();
    fxA.setZero();
    jxE.setZero();
    jxA.setZero();
    fE.compute_jacobian(x, fxE, jxE);
    fA.compute_jacobian(x, fxA, jxA);

    EXPECT_TRUE(jxE.isApprox(jxA, 1e-12))
        << "Enzyme Jac:\n" << jxE << "\nAutodiff Jac:\n" << jxA;
    EXPECT_TRUE(fxE.isApprox(fxA, 1e-15));
}

// -----------------------------------------------------------------------------
// Mixed mode <EnzymeAD, AutodiffFwd>: Phase 1 only differentiates to Jacobian
// via Enzyme (Hessian path stays AutodiffFwd until Phase 2).  This test exists
// to lock in that the mixed-mode pairing's compute_jacobian agrees with the
// pure-autodiff baseline; the Hessian path is exercised in Phase 2 tasks.
// -----------------------------------------------------------------------------
TEST(EnzymeJacobian, MixedMode) {
    using MixedJac = tycho_enzyme_test::BrachODEModed<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::AutodiffFwd>;
    using PureAutodiff = tycho_enzyme_test::BrachAutodiff;

    MixedJac fM(32.2);
    PureAutodiff fA(32.2);

    Eigen::Matrix<double, 5, 1> x;
    x << 0.4, 1.1, 2.5, 0.0, -0.3;

    Eigen::Matrix<double, 3, 1> fxM, fxA;
    Eigen::Matrix<double, 3, 5> jxM, jxA;
    fxM.setZero();
    fxA.setZero();
    jxM.setZero();
    jxA.setZero();
    fM.compute_jacobian(x, fxM, jxM);
    fA.compute_jacobian(x, fxA, jxA);

    EXPECT_TRUE(jxM.isApprox(jxA, 1e-12))
        << "Mixed-mode Jac:\n" << jxM << "\nPure autodiff Jac:\n" << jxA;
    EXPECT_TRUE(fxM.isApprox(fxA, 1e-15));
}

// -----------------------------------------------------------------------------
// Type erasure: a BrachEnzymeAD must round-trip through GenericFunction<-1,-1>
// preserving both compute() and compute_jacobian() lane-for-lane.
// -----------------------------------------------------------------------------
TEST(EnzymeJacobian, GenericFunctionTypeErasure) {
    tycho_enzyme_test::BrachEnzymeAD f(32.2);
    tycho::vf::GenericFunction<-1, -1> erased(f);

    EXPECT_EQ(erased.input_rows(), 5);
    EXPECT_EQ(erased.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 1.0, 2.0, 3.0, 0.0, 0.5;

    Eigen::VectorXd fx_direct(3), fx_erased(3);
    fx_direct.setZero();
    fx_erased.setZero();
    f.compute(x, fx_direct);
    erased.compute(x, fx_erased);
    EXPECT_TRUE(fx_direct.isApprox(fx_erased, 1e-15))
        << "Direct primal: " << fx_direct.transpose()
        << "\nErased primal: " << fx_erased.transpose();

    Eigen::VectorXd fxj_direct(3), fxj_erased(3);
    Eigen::MatrixXd jx_direct(3, 5), jx_erased(3, 5);
    fxj_direct.setZero();
    fxj_erased.setZero();
    jx_direct.setZero();
    jx_erased.setZero();
    f.compute_jacobian(x, fxj_direct, jx_direct);
    erased.compute_jacobian(x, fxj_erased, jx_erased);

    EXPECT_TRUE(jx_direct.isApprox(jx_erased, 1e-15))
        << "Direct Jac:\n" << jx_direct
        << "\nErased Jac:\n" << jx_erased;
    EXPECT_TRUE(fxj_direct.isApprox(fxj_erased, 1e-15));
}

} // namespace
