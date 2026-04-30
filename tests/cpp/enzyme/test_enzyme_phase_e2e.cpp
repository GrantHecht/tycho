// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// End-to-end PSIOPT test exercising the EnzymeAD Jacobian path through a
// realistic optimal-control problem (brachistochrone).  Confirms Phase 1's
// scalar Jacobian implementation is correct enough for PSIOPT to converge
// on a real direct-collocation problem.
//
// Mirror of examples/cpp_examples/static/brachistochrone/main.cpp, with the
// expression-DSL ODE replaced by tycho_enzyme_test::BrachEnzymeAD wrapped in
// a GenericFunction<-1,-1>.  The dynamics are identical; only the derivative-
// mode plumbing changes, so any deviation from the reference optimal time
// (~1.8013 s) flags an Enzyme-path correctness regression.
// =============================================================================

#include <gtest/gtest.h>

#include <tycho/tycho.h>

#include "enzyme_test_dynamics.h"

#include <Eigen/Core>
#include <utility>
#include <vector>

namespace {

TEST(EnzymePhaseE2E, BrachistochroneEnzymeFwdFDiff) {
    using namespace tycho;
    using namespace tycho::oc;
    using namespace tycho::solvers;

    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0;
    constexpr double theta_guess = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_defects = 32;

    // Linearly-interpolated initial guess: identical to the reference example.
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5); // [x, y, v, t, theta]
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    // Wrap the typed BrachEnzymeAD VF in a GenericFunction<-1, -1>.  Type
    // erasure preserves the underlying derivative path (verified
    // lane-for-lane in EnzymeJacobian.GenericFunctionTypeErasure), so calls
    // through this wrapper hit the EnzymeAD scalar Jacobian.
    auto erased = tycho::vf::GenericFunction<-1, -1>(
        tycho_enzyme_test::BrachEnzymeAD(g));
    ODE ode(std::move(erased), 3, 1, 0);
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, n_defects);

    // Front boundary: x(0)=x0, y(0)=y0, v(0)=v0, t(0)=t0.
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase.add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val,
                             ScaleModes::AUTO);

    // Back boundary: x(tf)=xf, y(tf)=yf.
    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase.add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val,
                             ScaleModes::AUTO);

    // Control bounds: theta in [-0.1, 2.0].
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);

    // Minimum-time objective.
    phase.add_delta_time_objective(1.0, ScaleModes::AUTO);

    const auto status = phase.solve_optimize();
    ASSERT_LE(static_cast<int>(status),
              static_cast<int>(PSIOPT::ConvergenceFlags::ACCEPTABLE))
        << "PSIOPT did not converge; status=" << static_cast<int>(status);

    const auto result = phase.return_traj();
    const double tf_opt = result.back()[3];
    EXPECT_NEAR(tf_opt, 1.8013, 1e-4)
        << "Optimal time mismatch (Enzyme path); got " << tf_opt;
}

// -----------------------------------------------------------------------------
// Full Enzyme pipeline: <EnzymeAD, EnzymeAD>.  Same problem, same expected
// optimal time, but the second-order derivatives PSIOPT consumes also flow
// through Enzyme (Phase 2's nested-AD strategy — FoR is the only cmake-
// selectable strategy; FoF preserved as archived research in dense_enzyme.h).
// Convergence here is the integration test that validates Phase 2 across the
// full direct-collocation stack.
// -----------------------------------------------------------------------------
TEST(EnzymePhaseE2E, BrachistochroneFullEnzymePipeline) {
    using namespace tycho;
    using namespace tycho::oc;
    using namespace tycho::solvers;

    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0;
    constexpr double theta_guess = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_defects = 32;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    auto erased = tycho::vf::GenericFunction<-1, -1>(
        tycho_enzyme_test::BrachEnzymeFull(g));
    ODE ode(std::move(erased), 3, 1, 0);
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, n_defects);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase.add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val,
                             ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase.add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val,
                             ScaleModes::AUTO);

    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase.add_delta_time_objective(1.0, ScaleModes::AUTO);

    const auto status = phase.solve_optimize();
    ASSERT_LE(static_cast<int>(status),
              static_cast<int>(PSIOPT::ConvergenceFlags::ACCEPTABLE))
        << "PSIOPT did not converge under <EnzymeAD, EnzymeAD>; status="
        << static_cast<int>(status);

    const auto result = phase.return_traj();
    const double tf_opt = result.back()[3];
    EXPECT_NEAR(tf_opt, 1.8013, 1e-4)
        << "Optimal time mismatch (full Enzyme pipeline); got " << tf_opt;
}

} // namespace
