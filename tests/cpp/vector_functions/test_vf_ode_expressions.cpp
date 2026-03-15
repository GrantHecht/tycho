///////////////////////////////////////////////////////////////////////////////
// ODE expression tests
//
// Extracted from test_vector_function_composition.cpp — ODE expression tests
// section.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
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
    EXPECT_EQ(gf.IRows(), 5);
    EXPECT_EQ(gf.ORows(), 3);

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
    EXPECT_EQ(kep.IRows(), 7); // [x,y,z,vx,vy,vz,t]
    EXPECT_EQ(kep.ORows(), 6); // [dx,dy,dz,dvx,dvy,dvz]
}

TEST_F(VFCompositionTest, KeplerODEAdjointConsistency) {
    Kepler kep(398600.4418);
    Eigen::VectorXd x(7);
    x << 7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0;
    Eigen::VectorXd lm = deterministic_random_vector(6, 202, -1.0, 1.0);
    verify_adjoint_consistency(kep, x, lm, 1e-8);
}

TEST_F(VFCompositionTest, CR3BPODEAdjointConsistency) {
    double mu = 0.012150585; // Earth-Moon
    CR3BP cr3bp(mu);
    EXPECT_EQ(cr3bp.IRows(), 7);
    EXPECT_EQ(cr3bp.ORows(), 6);

    Eigen::VectorXd x(7);
    x << 0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0;
    Eigen::VectorXd lm = deterministic_random_vector(6, 203, -1.0, 1.0);
    verify_adjoint_consistency(cr3bp, x, lm, 1e-10);
}
