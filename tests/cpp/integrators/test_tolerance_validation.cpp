// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Tolerance-validation tests: pin the contract that the adaptive path throws
// with an actionable message when abs_tol and rel_tol vanish together, rather
// than silently producing NaN error norms and looping until max_steps.

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

// Both setters accept 0 individually; only the joint (both-zero) invariant
// is enforced at integrate entry.
TEST_F(IntegratorTest, TolValidation_ScalarSetterAcceptsZero) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    EXPECT_NO_THROW(integ.set_abs_tol(0.0));
    EXPECT_NO_THROW(integ.set_rel_tol(0.0));
}

// Vector setter rejects negative entries with a component-indexed message.
TEST_F(IntegratorTest, TolValidation_VectorSetterRejectsNegative) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);

    Eigen::VectorXd bad(2);
    bad << 1e-6, -1e-3;
    EXPECT_THROW(integ.set_abs_tols(bad), std::invalid_argument);
    EXPECT_THROW(integ.set_rel_tols(bad), std::invalid_argument);
}

// Both tolerances zero — adaptive integrate must throw at the boundary with a
// message naming the offending component, NOT silently loop to max_steps.
TEST_F(IntegratorTest, TolValidation_BothZeroThrowsOnIntegrate) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(0.0);
    integ.set_rel_tol(0.0);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    try {
        integ.integrate(x0, 1.0);
        FAIL() << "Expected invalid_argument for both-zero tolerance";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("abs_tol + rel_tol"), std::string::npos)
            << "Error message should explain the invariant; got: " << msg;
    }
}

// Mixed abs-only + rel-only per-component tolerances are legal.
TEST_F(IntegratorTest, TolValidation_MixedComponentsAccepted) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);

    Eigen::VectorXd abs_t(2);
    abs_t << 1e-6, 0.0; // component 0 has abs, component 1 none
    Eigen::VectorXd rel_t(2);
    rel_t << 0.0, 1e-6; // component 1 has rel, component 0 none
    integ.set_abs_tols(abs_t);
    integ.set_rel_tols(rel_t);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.5, 0.0;
    EXPECT_NO_THROW(integ.integrate(x0, 0.1));
}

// Fixed-step path (non-adaptive) does NOT consult tolerances; misconfigured
// zero-zero tols should be accepted here without triggering the check.
TEST_F(IntegratorTest, TolValidation_FixedStepBypassesCheck) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.adaptive_ = false;
    integ.set_abs_tol(0.0);
    integ.set_rel_tol(0.0);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    EXPECT_NO_THROW(integ.integrate(x0, 0.1));
}

// +inf in a component is a silent correctness hazard: it zeros that
// component's scaled residual. Reject at the setter boundary, matching the
// scalar set_abs_tol / set_rel_tol counterparts.
TEST_F(IntegratorTest, TolValidation_VectorSetterRejectsInfinite) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);

    Eigen::VectorXd prior_abs = integ.get_abs_tols();
    Eigen::VectorXd prior_rel = integ.get_rel_tols();

    Eigen::VectorXd bad(2);
    bad << 1e-6, std::numeric_limits<double>::infinity();
    EXPECT_THROW(integ.set_abs_tols(bad), std::invalid_argument);
    EXPECT_THROW(integ.set_rel_tols(bad), std::invalid_argument);

    // State must be unchanged after the throw — validation is atomic.
    EXPECT_EQ(integ.get_abs_tols(), prior_abs);
    EXPECT_EQ(integ.get_rel_tols(), prior_rel);
}

// Non-finite initial step sizes (NaN or +inf) must fail at the API boundary,
// not be stored and surface later as a generic divide-by-zero.
TEST_F(IntegratorTest, TolValidation_InitialStepSizeRejectsNonFinite) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);

    EXPECT_THROW(integ.set_initial_step_size(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(integ.set_initial_step_size(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(integ.set_initial_step_size(0.0), std::invalid_argument);
    EXPECT_THROW(integ.set_initial_step_size(-1.0), std::invalid_argument);
}

// Constructor-path defstep shares the same validator via set_method; same
// guarantees expected.
TEST_F(IntegratorTest, TolValidation_ConstructorRejectsNonFiniteDefstep) {
    SHO ode(0.0);
    EXPECT_THROW((Integrator<SHO>{ode, IVPAlg::DOPRI87,
                                  std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((Integrator<SHO>{ode, IVPAlg::DOPRI87,
                                  std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
}
