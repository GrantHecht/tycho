///////////////////////////////////////////////////////////////////////////////
// Phase-level value-lock / sub_variables regression test
//
// Exercises Phase::add_value_lock + sub_variables through a full Phase solve
// so that the SubstitutedVarPlaceholder contract (zero residual, identity
// Jacobian) is verified end-to-end, not just at the VectorFunction unit level.
// This closes the gap between the internal placeholder pin in
// test_vf_value_lock.cpp and the user-facing Phase API.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/phase_wrapper.h>
#include <tycho/detail/optimal_control/builder/runtime_ode.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

class SubVariablesPhaseTest : public OptimalControlTest {};

namespace {

// Trivial 2-state 2-control ODE: xdot = u1, ydot = u2. Independent controls
// let us specify x(0), y(0), x(1), y(1) feasibly.
ODE make_trivial_ode() {
    return ODEBuilder(2, 2)
        .var_names({{"x", 0}, {"y", 1}, {"t", 2}, {"u1", 3}, {"u2", 4}})
        .define([](auto &args) {
            auto u1 = args.u_var(0);
            auto u2 = args.u_var(1);
            return stack(u1, u2);
        })
        .build();
}

std::vector<Eigen::VectorXd> make_linear_guess(double x0, double y0, double xf, double yf,
                                               int n_pts = 20) {
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s; // x
        pt[1] = y0 + (yf - y0) * s; // y
        pt[2] = s;                  // t
        pt[3] = (xf - x0);          // u1
        pt[4] = (yf - y0);          // u2
        traj.push_back(pt);
    }
    return traj;
}

} // namespace

// Solve the trivial problem with a free initial state, then lock it to a
// specific value via add_value_lock + sub_variables and verify the locked
// value is preserved at the solution.
TEST_F(SubVariablesPhaseTest, AddValueLockPinsInitialState) {
    auto ode = make_trivial_ode();

    const double locked_x0 = 0.25;
    const double locked_y0 = 0.75;

    auto traj_ig = make_linear_guess(locked_x0, locked_y0, 1.0, 1.0);
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, 16);

    // Lock the front-state x and y via the placeholder mechanism
    phase.add_value_lock(PhaseRegionFlags::Front, {"x", "y"});

    // Substitute the concrete values
    Eigen::VectorXi front_xy(2);
    front_xy << 0, 1;
    Eigen::Vector2d locked_vals(locked_x0, locked_y0);
    phase.sub_variables(PhaseRegionFlags::Front, front_xy, locked_vals);

    // Fix initial and final time so the feasibility problem has a unique solution
    phase.add_boundary_value(PhaseRegionFlags::Front, {"t"}, Eigen::Matrix<double, 1, 1>(0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"t"}, Eigen::Matrix<double, 1, 1>(1.0));

    // Back boundary: force a known target
    Eigen::Vector2d back_val(1.0, 1.0);
    phase.add_boundary_value(PhaseRegionFlags::Back, front_xy, back_val);

    // Bound both controls and add a quadratic sum-of-squares objective
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 3, -5.0, 5.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -5.0, 5.0);
    phase.add_integral_objective(GenericFunction<-1, 1>(Arguments<2>().squared_norm()),
                                 {"u1", "u2"});

    phase.optimizer().set_print_level(0);

    auto status = phase.solve_optimize();
    ASSERT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE)
        << "Phase with add_value_lock did not converge";

    auto result = phase.return_traj();
    ASSERT_FALSE(result.empty());

    // The front state must still be the locked value — the placeholder's
    // zero-residual Jacobian paired with sub_variables guarantees this.
    EXPECT_NEAR(result.front()[0], locked_x0, 1e-8)
        << "Locked x0 was perturbed by the solve; residual is not structurally zero";
    EXPECT_NEAR(result.front()[1], locked_y0, 1e-8)
        << "Locked y0 was perturbed by the solve; residual is not structurally zero";
}

// Pin the named-overload sub_variables path (string resolution via registry)
// — the full Phase solve verifies that the resolver returns the same index
// the int-based overload would use.
TEST_F(SubVariablesPhaseTest, NamedOverloadLocksFrontState) {
    auto ode = make_trivial_ode();

    const double locked_x0 = 0.1;
    const double locked_y0 = 0.9;

    auto traj_ig = make_linear_guess(locked_x0, locked_y0, 1.0, 1.0);
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, 16);

    phase.add_value_lock(PhaseRegionFlags::Front, {"x", "y"});
    const std::vector<std::string> names_xy{"x", "y"};
    Eigen::Vector2d locked_vals(locked_x0, locked_y0);
    phase.sub_variables(PhaseRegionFlags::Front, names_xy, locked_vals);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"t"}, Eigen::Matrix<double, 1, 1>(0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"t"}, Eigen::Matrix<double, 1, 1>(1.0));

    Eigen::VectorXi front_xy(2);
    front_xy << 0, 1;
    Eigen::Vector2d back_val(1.0, 1.0);
    phase.add_boundary_value(PhaseRegionFlags::Back, front_xy, back_val);

    phase.add_lu_var_bound(PhaseRegionFlags::Path, 3, -5.0, 5.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -5.0, 5.0);
    phase.add_integral_objective(GenericFunction<-1, 1>(Arguments<2>().squared_norm()),
                                 {"u1", "u2"});

    phase.optimizer().set_print_level(0);
    auto status = phase.solve_optimize();
    ASSERT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase.return_traj();
    EXPECT_NEAR(result.front()[0], locked_x0, 1e-8);
    EXPECT_NEAR(result.front()[1], locked_y0, 1e-8);
}

// sub_variables with an unknown name must surface a diagnostic rather than
// silently resolve to an invalid index — pins the resolve_for_region throw.
TEST_F(SubVariablesPhaseTest, UnknownNameThrows) {
    auto ode = make_trivial_ode();
    auto traj_ig = make_linear_guess(0.0, 0.0, 1.0, 1.0);
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, 8);

    const std::vector<std::string> bad{"not_a_variable"};
    Eigen::Matrix<double, 1, 1> val(0.0);
    EXPECT_THROW(phase.sub_variables(PhaseRegionFlags::Front, bad, val), std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// OC review §1.18: ODEPhaseBase::sub_variables() had no size-match or
// bounds-check guard, so a mismatched or out-of-range `indices` vector
// silently indexed past the end of `active_traj_[i]` / `active_static_params_`
// (or read stale/garbage `vals` entries). The Phase-wrapper overloads
// (phase_wrapper.h) forward directly to the base and need the same size
// guard so the diagnostic fires before any name resolution work is done.
///////////////////////////////////////////////////////////////////////////////

// Base-level (ODEPhaseBase::sub_variables) size-mismatch guard, exercised
// through the raw ODEPhase<BrachODE> pointer make_brach_phase() returns
// (bypasses the Phase wrapper entirely).
TEST_F(SubVariablesPhaseTest, BaseMismatchedSizesThrow) {
    auto phase = make_brach_phase(10);
    Eigen::VectorXi idx(2);
    idx << 0, 1;
    Eigen::VectorXd vals(1);
    vals << 3.0;
    EXPECT_THROW(phase->sub_variables(PhaseRegionFlags::Front, idx, vals), std::invalid_argument);
}

// Base-level bounds check: an index past the end of the trajectory row
// (BrachODE: x_vars=3, u_vars=1, p_vars=0 -> row width xtu_p_vars()==5) must
// throw rather than write out of bounds.
TEST_F(SubVariablesPhaseTest, BaseOutOfRangeIndexThrowsFront) {
    auto phase = make_brach_phase(10);
    Eigen::VectorXi idx(1);
    idx << 999;
    Eigen::VectorXd vals(1);
    vals << 1.0;
    EXPECT_THROW(phase->sub_variables(PhaseRegionFlags::Front, idx, vals), std::invalid_argument);
}

TEST_F(SubVariablesPhaseTest, BaseOutOfRangeIndexThrowsBack) {
    auto phase = make_brach_phase(10);
    Eigen::VectorXi idx(1);
    idx << -1;
    Eigen::VectorXd vals(1);
    vals << 1.0;
    EXPECT_THROW(phase->sub_variables(PhaseRegionFlags::Back, idx, vals), std::invalid_argument);
}

// Trajectory-less phase guard: ODEPhase(ode, Tmode) (ode_phase.h) is
// constructible -- and Python-reachable -- without ever setting a trajectory,
// leaving active_traj_ empty. sub_variables(Front/Back/Path, ...) on such a
// phase must throw a diagnostic pointing at set_traj, not dereference the
// empty vector.
TEST_F(SubVariablesPhaseTest, BaseTrajectorylessPhaseThrows) {
    BrachODE ode(9.81);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3);
    Eigen::VectorXi idx(1);
    idx << 0;
    Eigen::VectorXd vals(1);
    vals << 1.0;
    EXPECT_THROW(phase->sub_variables(PhaseRegionFlags::Front, idx, vals), std::invalid_argument);
    EXPECT_THROW(phase->sub_variables(PhaseRegionFlags::Back, idx, vals), std::invalid_argument);
    EXPECT_THROW(phase->sub_variables(PhaseRegionFlags::Path, idx, vals), std::invalid_argument);
    // Even the previously-no-op empty call must throw: there is no trajectory
    // for the substitution to (not) apply to.
    EXPECT_THROW(
        phase->sub_variables(PhaseRegionFlags::Front, Eigen::VectorXi(0), Eigen::VectorXd(0)),
        std::invalid_argument);
}

// StaticParams bounds check: make_brach_phase() never registers static
// params, so active_static_params_ is empty and even index 0 is out of range.
TEST_F(SubVariablesPhaseTest, BaseOutOfRangeIndexThrowsStaticParams) {
    auto phase = make_brach_phase(10);
    Eigen::VectorXi idx(1);
    idx << 0;
    Eigen::VectorXd vals(1);
    vals << 1.0;
    EXPECT_THROW(phase->sub_variables(PhaseRegionFlags::StaticParams, idx, vals),
                 std::invalid_argument);
}

// Phase-wrapper index-overload guard (phase_wrapper.h): the wrapper must
// reject a vars/vals size mismatch itself, before ever forwarding to the
// base. The "Phase::sub_variables" message prefix pins that the throw came
// from the wrapper layer, not from ODEPhaseBase.
TEST_F(SubVariablesPhaseTest, WrapperIndexOverloadMismatchedSizesThrow) {
    auto ode = make_trivial_ode();
    auto traj_ig = make_linear_guess(0.0, 0.0, 1.0, 1.0);
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, 8);

    Eigen::VectorXi idx(2);
    idx << 0, 1;
    Eigen::VectorXd vals(1);
    vals << 1.0;
    try {
        phase.sub_variables(PhaseRegionFlags::Front, idx, vals);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_EQ(std::string(e.what()).rfind("Phase::sub_variables", 0), 0u)
            << "throw did not come from the wrapper layer: " << e.what();
    }
}

// Phase-wrapper named-overload guard (phase_wrapper.h): same check, through
// the string-name resolution path; the size guard must fire before any name
// resolution work is done.
TEST_F(SubVariablesPhaseTest, WrapperNamedOverloadMismatchedSizesThrow) {
    auto ode = make_trivial_ode();
    auto traj_ig = make_linear_guess(0.0, 0.0, 1.0, 1.0);
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, 8);

    const std::vector<std::string> names{"x", "y"};
    Eigen::VectorXd vals(1);
    vals << 1.0;
    try {
        phase.sub_variables(PhaseRegionFlags::Front, names, vals);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_EQ(std::string(e.what()).rfind("Phase::sub_variables", 0), 0u)
            << "throw did not come from the wrapper layer: " << e.what();
    }
}
