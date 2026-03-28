///////////////////////////////////////////////////////////////////////////////
// Shared utilities for optimal control tests
//
// Provides the OptimalControlTest fixture and a standard Brachistochrone
// phase builder for reuse across OC and solver test files.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace TychoTest {

using namespace tycho;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class OptimalControlTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Helper: build a standard Brachistochrone phase
///////////////////////////////////////////////////////////////////////////////

inline std::shared_ptr<ODEPhase<BrachODE>> make_brach_phase(int n_pts = 100, int n_defects = 32) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase =
        std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, n_defects);

    // Front boundary
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    // Back boundary
    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    // Control bounds
    phase->add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);

    // Objective
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);

    return phase;
}

} // namespace TychoTest
