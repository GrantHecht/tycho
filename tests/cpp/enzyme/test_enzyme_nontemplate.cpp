// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// Phase 4: non-templated compute_impl with EnzymeAD.
//
// Verifies that a user dynamics struct can declare compute_impl with concrete
// double-typed arguments instead of the templated <class InType, class OutType>
// form, and still works under <EnzymeAD, EnzymeAD>.
// =============================================================================

#include <gtest/gtest.h>

#include "enzyme_test_dynamics.h"

namespace {

TEST(EnzymeNonTemplate, ComputeJacobianMatchesAnalytic) {
    tycho_enzyme_test::BrachEnzymeNonTemplate f(32.2);
    Eigen::Matrix<double, 5, 1> x;
    x << 1.0, 2.0, 3.0, 0.0, 0.5;

    Eigen::Matrix<double, 3, 1> fx;
    Eigen::Matrix<double, 3, 5> jx;
    fx.setZero();
    jx.setZero();
    f.compute_jacobian(x, fx, jx);

    // Reference values from analytical brachistochrone dynamics at the test point.
    EXPECT_NEAR(fx[0], std::sin(0.5) * 3.0, 1e-12);
    EXPECT_NEAR(fx[1], -std::cos(0.5) * 3.0, 1e-12);
    EXPECT_NEAR(fx[2], 32.2 * std::cos(0.5), 1e-12);
}

TEST(EnzymeNonTemplate, JacobianMatchesTemplatedReference) {
    tycho_enzyme_test::BrachEnzymeNonTemplate fNT(32.2);
    tycho_enzyme_test::BrachFDiff fA(32.2);
    Eigen::Matrix<double, 5, 1> x;
    x << 1.0, 2.0, 3.0, 0.0, 0.5;

    Eigen::Matrix<double, 3, 1> fxNT, fxA;
    Eigen::Matrix<double, 3, 5> jxNT, jxA;
    fxNT.setZero(); fxA.setZero();
    jxNT.setZero(); jxA.setZero();

    fNT.compute_jacobian(x, fxNT, jxNT);
    fA.compute_jacobian(x, fxA, jxA);

    EXPECT_TRUE(jxNT.isApprox(jxA, 1e-5)) // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
        << "Non-templated Enzyme Jacobian disagrees with FDiffCentArray reference";
}

TEST(EnzymeNonTemplate, HessianMatchesTemplatedReference) {
    tycho_enzyme_test::BrachEnzymeNonTemplate fNT(32.2);
    tycho_enzyme_test::BrachFDiff fA(32.2);
    Eigen::Matrix<double, 5, 1> x;
    x << 1.0, 2.0, 3.0, 0.0, 0.5;
    Eigen::Matrix<double, 3, 1> lam;
    lam << 0.7, -1.1, 0.3;

    Eigen::Matrix<double, 3, 1> fxNT, fxA;
    Eigen::Matrix<double, 3, 5> jxNT, jxA;
    Eigen::Matrix<double, 5, 1> gNT, gA;
    Eigen::Matrix<double, 5, 5> hNT, hA;
    fxNT.setZero(); fxA.setZero();
    jxNT.setZero(); jxA.setZero();
    gNT.setZero();  gA.setZero();
    hNT.setZero();  hA.setZero();

    fNT.compute_jacobian_adjointgradient_adjointhessian(x, fxNT, jxNT, gNT, hNT, lam);
    fA.compute_jacobian_adjointgradient_adjointhessian(x, fxA, jxA, gA, hA, lam);

    EXPECT_TRUE(jxNT.isApprox(jxA, 1e-5))  << "Jacobian mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(gNT.isApprox(gA, 1e-5))    << "Adjoint gradient mismatch"; // FDiffCentArray central-difference floor (~1e-5 with default step 1e-5)
    EXPECT_TRUE(hNT.isApprox(hA, 1e-3))    << "Adjoint Hessian mismatch"; // FDiffCentArray Hessian floor (~1e-3); higher than Jacobian floor due to second-difference compounding
}

// Documents the dispatch constraint: SuperScalar lane-batched compute requires
// templated compute_impl. Non-templated structs cannot opt into Vectorizable.
TEST(EnzymeNonTemplate, NotVectorizable) {
    static_assert(!tycho::vf::Vectorizable<tycho_enzyme_test::BrachEnzymeNonTemplate>,
        "BrachEnzymeNonTemplate must NOT be marked Vectorizable: SuperScalar "
        "VectorImpl path requires templated compute_impl.");
}

} // namespace
