// Contract pin for OCP phases_ lifetime (post-Batch 2 revert).
//
// OptimalControlProblem::add_phase takes a Phase by reference and stores a
// raw Phase* into phases_. The caller must keep each added Phase alive (and
// not relocated) for the OCP's lifetime. In exchange, name-resolution reads
// the caller's live VarRegistry and sp-name map, so names registered on the
// Phase after add_phase are visible to the OCP.
//
// This file pins both halves of that contract:
//   - stable caller storage works across multiple phases and name resolution,
//   - static-param names added AFTER add_phase are seen by the OCP's int +
//     named add_direct_link_equal_con overload.

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/phase_wrapper.h>
#include <tycho/tycho.h>

#include <memory>
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

// Stable storage (heap-allocated Phase wrappers) is the supported way to keep
// the caller's Phase alive while the OCP holds raw pointers to it. This test
// pins that contract: three heap-allocated Phases, added in order, resolving
// names via the int + named overload that goes through phases_.at().
TEST_F(OcpPhaseLifetimeTest, StableHeapStorageSupportsIntNamedResolution) {
    auto ode = make_lifetime_ode();

    std::vector<std::unique_ptr<Phase>> caller_phases;
    caller_phases.push_back(std::make_unique<Phase>(
        ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(0.0, 1.0), 8)));
    caller_phases.push_back(std::make_unique<Phase>(
        ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(1.0, 2.0), 8)));
    caller_phases.push_back(std::make_unique<Phase>(
        ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(2.0, 3.0), 8)));

    OptimalControlProblem ocp;
    for (auto &up : caller_phases)
        ocp.add_phase(*up);

    const std::vector<std::string> names_x{"x"};
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, names_x, 1,
                                                  PhaseRegionFlags::Front, names_x));
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(1, PhaseRegionFlags::Back, names_x, 2,
                                                  PhaseRegionFlags::Front, names_x));
}

// Post-add_phase static-param name additions on the caller's Phase must be
// visible to the OCP's int + named add_direct_link_equal_con overload. Under
// the Batch 2 shared_ptr-copy design this silently failed because the OCP
// queried a stale snapshot; with raw pointers the OCP reads the live sp map.
TEST_F(OcpPhaseLifetimeTest, PostAddPhaseStaticParamNamesVisible) {
    auto ode = make_lifetime_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_lt(1.0, 2.0), 8);

    // Give each phase a single static parameter slot so a name added after
    // add_phase has something to bind to. set_static_params sizes the SP
    // vector; the name registry is still empty at this point.
    Eigen::VectorXd sp(1);
    sp << 0.0;
    p0.set_static_params(sp);
    p1.set_static_params(sp);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);

    // Register the SP name AFTER add_phase. The OCP must see it.
    p0.add_static_param_name("sp0", 0);
    p1.add_static_param_name("sp0", 0);

    const std::vector<std::string> names_sp{"sp0"};
    EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::StaticParams, names_sp, 1,
                                                  PhaseRegionFlags::StaticParams, names_sp));
}
