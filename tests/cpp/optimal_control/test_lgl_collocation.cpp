///////////////////////////////////////////////////////////////////////////////
// LGL collocation tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, LGLCoeffsIntegralWeights) {
    // LGL3 (CS=2): Cardinal integral weights + interior integral weights should
    // sum to 2.0 (the Simpson's 1/3 rule weights on [0,1] with midpoint)
    using Coeffs = LGLCoeffs<2>;

    // Cardinal spacings
    EXPECT_DOUBLE_EQ(Coeffs::CardinalSpacings[0], 0.0);
    EXPECT_DOUBLE_EQ(Coeffs::CardinalSpacings[1], 1.0);

    // Interior spacing at midpoint
    EXPECT_DOUBLE_EQ(Coeffs::InteriorSpacings[0], 0.5);

    // Integral weights: Simpson's 1/3 rule gives 1/3 + 4/3 + 1/3 = 2
    double sum = Coeffs::Cardinal_Integral_Weights[0] + Coeffs::Interior_Integral_Weights[0] +
                 Coeffs::Cardinal_Integral_Weights[1];
    EXPECT_NEAR(sum, 2.0, 1e-14);

    // Cardinal weights are symmetric
    EXPECT_DOUBLE_EQ(Coeffs::Cardinal_Integral_Weights[0], Coeffs::Cardinal_Integral_Weights[1]);
}

TEST_F(OptimalControlTest, LGLDefectsDimensions) {
    // BrachODE: XV=3, UV=1, PV=0
    // For CS=2: input = 2 * XtUVars + PVars = 2 * 5 + 0 = 10
    //           output = ORows * (CS-1) = 3 * 1 = 3
    BrachODE ode(9.81);
    LGLDefects<BrachODE, 2> defects(ode);
    EXPECT_EQ(defects.input_rows(), 10);
    EXPECT_EQ(defects.output_rows(), 3);
}

TEST_F(OptimalControlTest, LGLDefectsAdjointConsistency) {
    BrachODE ode(9.81);
    LGLDefects<BrachODE, 2> defects(ode);

    // Random input: 2 cardinal states of [x, y, v, t, theta], no params
    Eigen::VectorXd x = deterministic_random_vector(10, 300, 0.1, 5.0);
    // Ensure time is increasing: t0 < t1
    x[3] = 0.0; // t0
    x[8] = 1.0; // t1

    Eigen::VectorXd lm = deterministic_random_vector(3, 301, -1.0, 1.0);
    verify_adjoint_consistency(defects, x, lm, 1e-10);
}
