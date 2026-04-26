// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// Phase 2 Enzyme Hessian unit-test matrix.
//
// Compares <EnzymeAD, EnzymeAD> against <AutodiffFwd, AutodiffFwd> for the
// brachistochrone, CR3BP, and MEE test dynamics. Tests run under whichever
// nested-Enzyme strategy (FoR or FoF) is selected at configure time via
// TYCHO_ENZYME_HESSIAN_STRATEGY.
//
// Tolerances per spec §4.4 / plan Phase 2 gate:
//   - Jacobian / adjoint gradient agree to 1e-12 (lane-for-lane parity).
//   - Adjoint Hessian agrees to 1e-10 (Enzyme's IR-level diff and autodiff's
//     dual arithmetic both compute the same mathematical quantity, but with
//     slightly different floating-point rounding paths).
//   - Hessian is symmetric to 1e-12 (we explicitly symmetrize).
// =============================================================================

#include <gtest/gtest.h>

#include <tycho/tycho.h>

#include "enzyme_test_dynamics.h"

namespace {

using tycho::vf::DenseDerivativeMode;

// Convenience aliases for the two-mode pairings exercised in this matrix.
using BrachAutodiffFull = tycho_enzyme_test::BrachODEModed<
    DenseDerivativeMode::AutodiffFwd, DenseDerivativeMode::AutodiffFwd>;
using BrachEnzymeFull = tycho_enzyme_test::BrachODEModed<
    DenseDerivativeMode::EnzymeAD, DenseDerivativeMode::EnzymeAD>;

using CR3BPAutodiffFull = tycho_enzyme_test::CR3BPModed<
    DenseDerivativeMode::AutodiffFwd, DenseDerivativeMode::AutodiffFwd>;
using CR3BPEnzymeFull = tycho_enzyme_test::CR3BPModed<
    DenseDerivativeMode::EnzymeAD, DenseDerivativeMode::EnzymeAD>;

using MEEAutodiffFull = tycho_enzyme_test::MEEModed<
    DenseDerivativeMode::AutodiffFwd, DenseDerivativeMode::AutodiffFwd>;
using MEEEnzymeFull = tycho_enzyme_test::MEEModed<
    DenseDerivativeMode::EnzymeAD, DenseDerivativeMode::EnzymeAD>;

// -----------------------------------------------------------------------------
// Brachistochrone — simplest dynamics; exercises core pipeline correctness.
// -----------------------------------------------------------------------------
TEST(EnzymeHessian, BrachistochroneVsAutodiff) {
    BrachEnzymeFull fE;
    BrachAutodiffFull fA;

    Eigen::Matrix<double, 5, 1> x;
    x << 1.0, 2.0, 3.0, 0.0, 0.5;
    Eigen::Matrix<double, 3, 1> lam;
    lam << 0.7, -1.1, 0.3;

    Eigen::Matrix<double, 3, 1> fxE, fxA;
    Eigen::Matrix<double, 3, 5> jxE, jxA;
    Eigen::Matrix<double, 5, 1> gE, gA;
    Eigen::Matrix<double, 5, 5> hE, hA;
    fxE.setZero(); fxA.setZero();
    jxE.setZero(); jxA.setZero();
    gE.setZero();  gA.setZero();
    hE.setZero();  hA.setZero();

    fE.compute_jacobian_adjointgradient_adjointhessian(x, fxE, jxE, gE, hE, lam);
    fA.compute_jacobian_adjointgradient_adjointhessian(x, fxA, jxA, gA, hA, lam);

    EXPECT_TRUE(fxE.isApprox(fxA, 1e-12)) << "fx mismatch";
    EXPECT_TRUE(jxE.isApprox(jxA, 1e-12)) << "Jacobian mismatch";
    EXPECT_TRUE(gE.isApprox(gA, 1e-12))   << "adjoint gradient mismatch";
    EXPECT_TRUE(hE.isApprox(hA, 1e-10))   << "adjoint Hessian mismatch";

    // Hessian symmetry (we explicitly symmetrize, so this is tight).
    EXPECT_TRUE(hE.isApprox(hE.transpose(), 1e-12)) << "Hessian not symmetric";
}

// -----------------------------------------------------------------------------
// CR3BP — represents the typical astrodynamics workload (7 inputs, 6 outputs,
// dense second derivatives).  Pick a point well away from the primary bodies
// to avoid the 1/r^3 singularities.
// -----------------------------------------------------------------------------
TEST(EnzymeHessian, CR3BPVsAutodiff) {
    CR3BPEnzymeFull fE(0.0123);
    CR3BPAutodiffFull fA(0.0123);

    Eigen::Matrix<double, 7, 1> x;
    x << 0.5, 0.7, 0.3, 0.4, -0.2, 0.6, 1.0;
    Eigen::Matrix<double, 6, 1> lam;
    lam << 0.2, -0.5, 0.7, 0.1, -0.3, 0.4;

    Eigen::Matrix<double, 6, 1> fxE, fxA;
    Eigen::Matrix<double, 6, 7> jxE, jxA;
    Eigen::Matrix<double, 7, 1> gE, gA;
    Eigen::Matrix<double, 7, 7> hE, hA;
    fxE.setZero(); fxA.setZero();
    jxE.setZero(); jxA.setZero();
    gE.setZero();  gA.setZero();
    hE.setZero();  hA.setZero();

    fE.compute_jacobian_adjointgradient_adjointhessian(x, fxE, jxE, gE, hE, lam);
    fA.compute_jacobian_adjointgradient_adjointhessian(x, fxA, jxA, gA, hA, lam);

    EXPECT_TRUE(jxE.isApprox(jxA, 1e-12)) << "CR3BP Jacobian mismatch";
    EXPECT_TRUE(gE.isApprox(gA, 1e-12))   << "CR3BP adjoint gradient mismatch";
    EXPECT_TRUE(hE.isApprox(hA, 1e-10))   << "CR3BP adjoint Hessian mismatch";
    EXPECT_TRUE(hE.isApprox(hE.transpose(), 1e-12))
        << "CR3BP Hessian not symmetric";
}

// -----------------------------------------------------------------------------
// MEE — 9 inputs, 6 outputs; the largest input dimension in the test set.
// Pick the point so the 1/x[0] and 1/(1+x1*cos(x5)+x2*sin(x5)) denominators
// are well away from zero.
// -----------------------------------------------------------------------------
TEST(EnzymeHessian, MEEVsAutodiff) {
    MEEEnzymeFull fE(1.0);
    MEEAutodiffFull fA(1.0);

    Eigen::Matrix<double, 9, 1> x;
    x << 1.5, 0.05, 0.02, 0.1, 0.05, 0.3, 0.0, 0.7, 0.4;
    Eigen::Matrix<double, 6, 1> lam;
    lam << 0.3, -0.4, 0.2, 0.7, -0.1, 0.5;

    Eigen::Matrix<double, 6, 1> fxE, fxA;
    Eigen::Matrix<double, 6, 9> jxE, jxA;
    Eigen::Matrix<double, 9, 1> gE, gA;
    Eigen::Matrix<double, 9, 9> hE, hA;
    fxE.setZero(); fxA.setZero();
    jxE.setZero(); jxA.setZero();
    gE.setZero();  gA.setZero();
    hE.setZero();  hA.setZero();

    fE.compute_jacobian_adjointgradient_adjointhessian(x, fxE, jxE, gE, hE, lam);
    fA.compute_jacobian_adjointgradient_adjointhessian(x, fxA, jxA, gA, hA, lam);

    EXPECT_TRUE(jxE.isApprox(jxA, 1e-12)) << "MEE Jacobian mismatch";
    EXPECT_TRUE(gE.isApprox(gA, 1e-12))   << "MEE adjoint gradient mismatch";
    EXPECT_TRUE(hE.isApprox(hA, 1e-10))   << "MEE adjoint Hessian mismatch";
    EXPECT_TRUE(hE.isApprox(hE.transpose(), 1e-12))
        << "MEE Hessian not symmetric";
}

// -----------------------------------------------------------------------------
// Adjoint consistency: g = J^T λ, computed two different ways through the
// EnzymeAD path, must agree to 1e-12.  Cross-checks that the adjoint gradient
// returned by compute_jacobian_adjointgradient_adjointhessian is consistent
// with the Jacobian it returned in the same call.
// -----------------------------------------------------------------------------
TEST(EnzymeHessian, AdjointConsistencyCR3BP) {
    CR3BPEnzymeFull fE(0.0123);

    Eigen::Matrix<double, 7, 1> x;
    x << 0.5, 0.7, 0.3, 0.4, -0.2, 0.6, 1.0;
    Eigen::Matrix<double, 6, 1> lam;
    lam << 0.2, -0.5, 0.7, 0.1, -0.3, 0.4;

    Eigen::Matrix<double, 6, 1> fx;
    Eigen::Matrix<double, 6, 7> jx;
    Eigen::Matrix<double, 7, 1> g;
    Eigen::Matrix<double, 7, 7> h;
    fx.setZero(); jx.setZero(); g.setZero(); h.setZero();

    fE.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, g, h, lam);

    Eigen::Matrix<double, 7, 1> g_from_jx = jx.transpose() * lam;
    EXPECT_TRUE(g.isApprox(g_from_jx, 1e-12))
        << "Adjoint gradient inconsistent with returned Jacobian";
}

} // namespace
