// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// Phase 2 Enzyme Hessian unit-test matrix.
//
// Compares <EnzymeAD, EnzymeAD> against <FDiffCentArray, FDiffCentArray> for
// the brachistochrone, CR3BP, and MEE test dynamics. Tests run under whichever
// nested-Enzyme strategy (FoR or FoF) is selected at configure time via
// TYCHO_ENZYME_HESSIAN_STRATEGY.
//
// Tolerances:
//   - Jacobian / adjoint gradient agree to 1e-5 (FDiffCentArray
//     central-difference floor ~1e-5 with default step 1e-5).
//   - Adjoint Hessian agrees to 1e-3 (FDiffCentArray Hessian floor ~1e-3;
//     higher than Jacobian floor due to second-difference compounding).
//   - Hessian is symmetric to 1e-12 (we explicitly symmetrize).
// =============================================================================

#include <gtest/gtest.h>

#include <tycho/tycho.h>

#include "enzyme_test_dynamics.h"

namespace {

using tycho::vf::DenseDerivativeMode;

// Convenience aliases for the two-mode pairings exercised in this matrix.
// FDiff reference: FDiffCentArray Jacobian + FDiffFwd Hessian.
// (These mirror the *FDiff aliases in enzyme_test_dynamics.h.)
using BrachFDiffFull  = tycho_enzyme_test::BrachFDiff;
using BrachEnzymeFull = tycho_enzyme_test::BrachEnzymeFull;

using CR3BPFDiffFull  = tycho_enzyme_test::CR3BPFDiff;
using CR3BPEnzymeFull = tycho_enzyme_test::CR3BPEnzymeFull;

using MEEFDiffFull  = tycho_enzyme_test::MEEFDiff;
using MEEEnzymeFull = tycho_enzyme_test::MEEEnzymeFull;

// -----------------------------------------------------------------------------
// Brachistochrone — simplest dynamics; exercises core pipeline correctness.
// -----------------------------------------------------------------------------
TEST(EnzymeHessian, BrachistochroneVsFDiff) {
    BrachEnzymeFull fE;
    BrachFDiffFull fA;

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
    EXPECT_TRUE(jxE.isApprox(jxA, 1e-5)) << "Jacobian mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(gE.isApprox(gA, 1e-5))   << "adjoint gradient mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(hE.isApprox(hA, 1e-3))   << "adjoint Hessian mismatch"; // FDiffCentArray Hessian floor (~1e-3); higher than Jacobian floor due to second-difference compounding

    // Hessian symmetry (we explicitly symmetrize, so this is tight).
    EXPECT_TRUE(hE.isApprox(hE.transpose(), 1e-12)) << "Hessian not symmetric";
}

// -----------------------------------------------------------------------------
// CR3BP — represents the typical astrodynamics workload (7 inputs, 6 outputs,
// dense second derivatives).  Pick a point well away from the primary bodies
// to avoid the 1/r^3 singularities.
// -----------------------------------------------------------------------------
TEST(EnzymeHessian, CR3BPVsFDiff) {
    CR3BPEnzymeFull fE(0.0123);
    CR3BPFDiffFull fA(0.0123);

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

    EXPECT_TRUE(jxE.isApprox(jxA, 1e-5)) << "CR3BP Jacobian mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(gE.isApprox(gA, 1e-5))   << "CR3BP adjoint gradient mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(hE.isApprox(hA, 1e-3))   << "CR3BP adjoint Hessian mismatch"; // FDiffCentArray Hessian floor (~1e-3); higher than Jacobian floor due to second-difference compounding
    EXPECT_TRUE(hE.isApprox(hE.transpose(), 1e-12))
        << "CR3BP Hessian not symmetric";
}

// -----------------------------------------------------------------------------
// MEE — 9 inputs, 6 outputs; the largest input dimension in the test set.
// Pick the point so the 1/x[0] and 1/(1+x1*cos(x5)+x2*sin(x5)) denominators
// are well away from zero.
// -----------------------------------------------------------------------------
TEST(EnzymeHessian, MEEVsFDiff) {
    MEEEnzymeFull fE(1.0);
    MEEFDiffFull fA(1.0);

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

    EXPECT_TRUE(jxE.isApprox(jxA, 1e-5)) << "MEE Jacobian mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(gE.isApprox(gA, 1e-5))   << "MEE adjoint gradient mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(hE.isApprox(hA, 1e-3))   << "MEE adjoint Hessian mismatch"; // FDiffCentArray Hessian floor (~1e-3); higher than Jacobian floor due to second-difference compounding
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
