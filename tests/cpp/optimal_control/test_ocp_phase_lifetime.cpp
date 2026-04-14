// Regression pin for OCP phases_ lifetime (Batch 2 B-1).
//
// OptimalControlProblem::add_phase takes a Phase by value and stores it as
// shared_ptr<Phase>, so the OCP owns its own copy and is independent of the
// caller's container. Prior raw-pointer storage required the caller to
// pre-reserve() a container or risk UB on reallocation — this test reallocates
// the caller's vector after add_phase and confirms name resolution still works.
//
// Without the shared_ptr fix, `IntIndexedNamedOverloadResolvesViaPhasesVector`
// here would read a dangling pointer after the push_back reallocation.

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/phase_wrapper.h>
#include <tycho/tycho.h>

#include <vector>

using namespace tycho;
using namespace TychoTest;

namespace {

ODE make_lifetime_ode() {
    return ODEBuilder(2, 1)
        .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
        .define([](auto &args) {
            auto v = args.x_var(1);
            auto u = args.u_var(0);
            return stack(v, u);
        })
        .build();
}

std::vector<Eigen::VectorXd> make_linear_guess_lt(double x0, double xf, int n = 10) {
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

class OcpPhaseLifetimeTest : public OptimalControlTest {};

// Push phases into the OCP, then force the caller's container to reallocate
// and append more phases afterwards. The int+named overload must still resolve
// names against the OCP's own copies, not the (now-dangling) original
// addresses.
TEST_F(OcpPhaseLifetimeTest, CallerReallocAfterAddPhaseDoesNotDangle) {
    auto ode = make_lifetime_ode();

    // Intentionally start with capacity=1 so the second push_back reallocates.
    std::vector<Phase> caller_phases;
    caller_phases.reserve(1);
    caller_phases.push_back(ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(0.0, 1.0), 8));

    OptimalControlProblem ocp;
    ocp.add_phase(caller_phases.front());

    // Reallocate caller container: adds two more phases, grows past capacity 1.
    caller_phases.push_back(ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(1.0, 2.0), 8));
    caller_phases.push_back(ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(2.0, 3.0), 8));

    // Register the new ones with the OCP.
    ocp.add_phase(caller_phases[1]);
    ocp.add_phase(caller_phases[2]);

    // Name resolution uses the OCP-owned copies; reallocation of caller_phases
    // must not invalidate anything inside the OCP.
    const std::vector<std::string> names_x{"x"};
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, names_x, 1,
                                                  PhaseRegionFlags::Front, names_x));
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(1, PhaseRegionFlags::Back, names_x, 2,
                                                  PhaseRegionFlags::Front, names_x));
}

// After add_phase, destroy the caller's original Phase entirely. The OCP must
// still resolve names because it holds its own shared_ptr<Phase>.
TEST_F(OcpPhaseLifetimeTest, CallerPhaseDestroyedAfterAddPhase) {
    auto ode = make_lifetime_ode();

    OptimalControlProblem ocp;
    {
        auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(0.0, 1.0), 8);
        auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(1.0, 2.0), 8);
        ocp.add_phase(p0);
        ocp.add_phase(p1);
    } // p0, p1 destroyed here

    const std::vector<std::string> names_x{"x"};
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, names_x, 1,
                                                  PhaseRegionFlags::Front, names_x));
}
