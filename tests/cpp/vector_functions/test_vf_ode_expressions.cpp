///////////////////////////////////////////////////////////////////////////////
// ODE expression tests
//
// Extracted from test_vector_function_composition.cpp — ODE expression tests
// section.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// ODE expression tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, BrachistochroneAdjointConsistency) {
    BrachODE ode(9.81);
    Eigen::VectorXd x = deterministic_random_vector(5, 200, 0.1, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 201, -1.0, 1.0);
    verify_adjoint_consistency(ode, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, BrachistochroneGenericODEErasure) {
    BrachODE ode(9.81);
    GenericFunction<-1, -1> gf(ode);
    EXPECT_EQ(gf.input_rows(), 5);
    EXPECT_EQ(gf.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx_orig(3), fx_gf(3);
    fx_orig.setZero();
    fx_gf.setZero();
    ode.compute(x, fx_orig);
    gf.compute(x, fx_gf);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(fx_orig[i], fx_gf[i], 1e-14);
    }
}

TEST_F(VFCompositionTest, KeplerODEDimensions) {
    Kepler kep(398600.4418);
    EXPECT_EQ(kep.input_rows(), 7); // [x,y,z,vx,vy,vz,t]
    EXPECT_EQ(kep.output_rows(), 6); // [dx,dy,dz,dvx,dvy,dvz]
}

TEST_F(VFCompositionTest, KeplerODEAdjointConsistency) {
    Kepler kep(398600.4418);
    Eigen::VectorXd x(7);
    x << 7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0;
    Eigen::VectorXd lm = deterministic_random_vector(6, 202, -1.0, 1.0);
    verify_adjoint_consistency(kep, x, lm, 1e-8);
}

TEST_F(VFCompositionTest, CRTBPDynamicsAdjointConsistency) {
    double mu = 0.012150585; // Earth-Moon
    astro::CRTBPDynamics dyn(mu);
    EXPECT_EQ(dyn.input_rows(), 9);  // 6 state + 3 extra-acceleration
    EXPECT_EQ(dyn.output_rows(), 6);

    Eigen::VectorXd x(9);
    x << 0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd lm = deterministic_random_vector(6, 203, -1.0, 1.0);
    verify_adjoint_consistency(dyn, x, lm, 1e-10);
}
