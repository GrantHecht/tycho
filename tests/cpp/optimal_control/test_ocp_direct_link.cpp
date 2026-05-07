// OptimalControlProblem::add_direct_link_equal_con overload coverage —
// pins all four variants (int+int, Phase&+Phase&, int+names, Phase&+names)
// plus the out-of-range throw for the int-indexed name path that goes
// through phases_.at().

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/phase_wrapper.h>
#include <tycho/tycho.h>

#include <vector>

using namespace tycho;
using namespace TychoTest;

namespace {

// Trivial 2-state 1-control ODE: x' = v, v' = u.
// Gives us two named state variables plus a control so every link-constraint
// overload has something meaningful to reference.
ODE make_direct_link_ode() {
    return ODEBuilder(2, 1)
        .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
        .define([](auto &args) {
            auto v = args.x_var(1);
            auto u = args.u_var(0);
            return stack(v, u);
        })
        .build();
}

std::vector<Eigen::VectorXd> make_linear_guess_dl(double x0, double xf, int n = 10) {
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n);
    for (int i = 0; i < n; ++i) {
        double s = static_cast<double>(i) / (n - 1);
        Eigen::VectorXd pt(4);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = (xf - x0);
        pt[2] = s;
        pt[3] = 0.0;
        traj.push_back(pt);
    }
    return traj;
}

} // namespace

class OcpDirectLinkTest : public OptimalControlTest {};

TEST_F(OcpDirectLinkTest, IntIndexedOverloadAcceptsConstraint) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);

    OptimalControlProblem ocp;
    int idx0 = ocp.add_phase(p0);
    int idx1 = ocp.add_phase(p1);
    EXPECT_EQ(idx0, 0);
    EXPECT_EQ(idx1, 1);

    Eigen::VectorXi vars_a(1);
    vars_a << 0; // x
    Eigen::VectorXi vars_b(1);
    vars_b << 0;

    // Should accept without throwing.
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, vars_a, 1,
                                                  PhaseRegionFlags::Front, vars_b));
}

TEST_F(OcpDirectLinkTest, PhaseRefIndexedOverloadAcceptsConstraint) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);

    Eigen::VectorXi vars_a(1);
    vars_a << 1; // v
    Eigen::VectorXi vars_b(1);
    vars_b << 1;

    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(p0, PhaseRegionFlags::Back, vars_a, p1,
                                                  PhaseRegionFlags::Front, vars_b));
}

TEST_F(OcpDirectLinkTest, IntIndexedNamedOverloadResolvesViaPhasesVector) {
    // Exercises the phases_.at(int_idx) path — the specific reason
    // phases_ exists on the wrapper at all.
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);

    const std::vector<std::string> names_xv{"x", "v"};
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, names_xv, 1,
                                                  PhaseRegionFlags::Front, names_xv));
}

TEST_F(OcpDirectLinkTest, PhaseRefNamedOverloadResolvesViaRegistry) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);

    const std::vector<std::string> names_x{"x"};
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(p0, PhaseRegionFlags::Back, names_x, p1,
                                                  PhaseRegionFlags::Front, names_x));
}

TEST_F(OcpDirectLinkTest, IntIndexedNamedOverloadThrowsOnBadIndex) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);

    const std::vector<std::string> names_x{"x"};
    // Only one phase registered — index 7 must throw via phases_.at().
    EXPECT_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, names_x, 7,
                                               PhaseRegionFlags::Front, names_x),
                 std::out_of_range);
}

// ---------------------------------------------------------------------------
// Regression: pre-validate iphase/fphase (and p0/p1) before any constraint
// insertion. The loop-based add_forward_link_equal_con and
// add_param_link_equal_con call add_func_impl per iteration, so a mid-loop
// validation failure (e.g. fphase out of range) used to leave earlier
// link-equality constraints already inserted in link_equalities_, and a
// retry would duplicate them.
// ---------------------------------------------------------------------------

TEST_F(OcpDirectLinkTest, ForwardLinkRejectsBadEndingPhaseWithoutPartialMutation) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);
    auto p2 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(2.0, 3.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);
    ocp.add_phase(p2);

    Eigen::VectorXi vars(1);
    vars << 0;

    const auto before = ocp.base().link_equalities_.size();

    // fphase=99 is out of range; pre-fix this would have inserted link
    // constraints for iterations i=0 and i=1 before the i=2 access of
    // phases[3] threw inside make_func_impl.
    EXPECT_THROW(ocp.base().add_forward_link_equal_con(0, 99, vars, ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before)
        << "Failed forward-link insertion must leave the constraint map untouched";

    // Retry must also not insert anything (pre-fix: would have duplicated 0,1).
    EXPECT_THROW(ocp.base().add_forward_link_equal_con(0, 99, vars, ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}

TEST_F(OcpDirectLinkTest, DirectLinkRejectsBadSecondPhaseWithoutPartialMutation) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);

    Eigen::VectorXi vars(1);
    vars << 0;

    const auto before = ocp.base().link_equalities_.size();

    // p1=99 is out of range; pre-fix only p0 was checked up front and the
    // bad p1 was diagnosed deeper in the call stack.
    EXPECT_THROW(ocp.base().add_direct_link_equal_con(0, PhaseRegionFlags::Back, vars, 99,
                                                      PhaseRegionFlags::Front, vars,
                                                      ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}

TEST_F(OcpDirectLinkTest, ParamLinkRejectsBadEndingPhaseWithoutPartialMutation) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);
    auto p2 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(2.0, 3.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);
    ocp.add_phase(p2);

    Eigen::VectorXi vars(0); // empty — never reached due to fphase check

    const auto before = ocp.base().link_equalities_.size();

    // Reg passes the ODEParams/StaticParams check; bad fphase must short-circuit
    // before the per-iteration add_direct_link_equal_con call inserts anything.
    EXPECT_THROW(ocp.base().add_param_link_equal_con(0, 99, PhaseRegionFlags::StaticParams, vars,
                                                     ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}
