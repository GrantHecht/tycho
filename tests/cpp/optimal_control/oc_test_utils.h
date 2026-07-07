///////////////////////////////////////////////////////////////////////////////
// Shared utilities for optimal control tests
//
// Provides the OptimalControlTest fixture and a standard Brachistochrone
// phase builder for reuse across OC and solver test files.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <tycho/tycho.h>
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

inline std::shared_ptr<ODEPhase<BrachODE>>
make_brach_phase(int n_pts = 100, int n_defects = 32,
                 TranscriptionModes mode = TranscriptionModes::LGL3) {
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
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, mode, traj, n_defects);

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
    phase->add_delta_time_objective(1.0, ScaleModes::AUTO);

    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// Helper: build a non-even-data LGLInterpTable via load_exact_data
//
// load_exact_data() is the loader used by the phase path (ode_phase_base.cpp);
// unlike the even-data loaders it never sets delta_t_ (stays 0.0). Used to
// reproduce OC review §1.5: InterpFunction's non-LGL3 FD-Hessian branch
// historically derived its step directly from delta_t_.
///////////////////////////////////////////////////////////////////////////////

/// Build a single-state (no controls) LGLInterpTable in the given transcription
/// mode from a short analytic trajectory (x(t) = sin(t), xdot = cos(t)) loaded
/// via load_exact_data(), so delta_t_ stays 0.0 as it does on the phase path.
inline std::shared_ptr<LGLInterpTable> make_exact_lgl_table(TranscriptionModes method) {
    constexpr int x_vars = 1;
    // 6 == (num_nodes - 1) is divisible by (block_size_ - 1) for LGL3/5/7 (1, 2, 3).
    constexpr int num_nodes = 7;
    constexpr double t_final = 1.0;

    auto tab = std::make_shared<LGLInterpTable>(x_vars, /*uv=*/0, method);

    std::vector<Eigen::VectorXd> xtudat;
    std::vector<Eigen::VectorXd> xdotdat;
    xtudat.reserve(num_nodes);
    xdotdat.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        double t = t_final * static_cast<double>(i) / (num_nodes - 1);
        Eigen::VectorXd node(2); // [x, t] -- axis_ == x_vars == 1
        node << std::sin(t), t;
        xtudat.push_back(node);
        Eigen::VectorXd deriv(1);
        deriv << std::cos(t);
        xdotdat.push_back(deriv);
    }

    tab->load_exact_data(xtudat, xdotdat);
    return tab;
}

/// Indices of all state variables (excluding the time/control columns) of a table.
inline Eigen::VectorXi all_state_vars(const std::shared_ptr<LGLInterpTable> &tab) {
    Eigen::VectorXi v(tab->x_vars_);
    for (int i = 0; i < tab->x_vars_; ++i)
        v[i] = i;
    return v;
}

} // namespace TychoTest
