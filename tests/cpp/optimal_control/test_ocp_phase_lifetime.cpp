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
    int link_01 = -1;
    int link_12 = -1;
    ASSERT_NO_THROW(link_01 = ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, names_x, 1,
                                                            PhaseRegionFlags::Front, names_x));
    ASSERT_NO_THROW(link_12 = ocp.add_direct_link_equal_con(1, PhaseRegionFlags::Back, names_x, 2,
                                                            PhaseRegionFlags::Front, names_x));
    // Pin that the OCP actually committed both link constraints — a stale
    // pointer that segfaulted on access would surface here, and a silent
    // resolution failure would return a sentinel/-1 instead of a valid idx.
    EXPECT_GE(link_01, 0);
    EXPECT_GE(link_12, 0);
    EXPECT_NE(link_01, link_12);
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
    int sp_link = -1;
    ASSERT_NO_THROW(sp_link = ocp.add_direct_link_equal_con(0, PhaseRegionFlags::StaticParams,
                                                            names_sp, 1,
                                                            PhaseRegionFlags::StaticParams,
                                                            names_sp));
    EXPECT_GE(sp_link, 0);
}

// ---------------------------------------------------------------------------
// OC review §1.9: OptimalControlProblemBase::remove_phase() must (a) reject
// removal of a phase still referenced by a link function, (b) bounds-check
// its index argument (and phase()'s), and (c) shift the phase indices
// recorded on every remaining link function down by one wherever they
// referenced a phase after the removed slot.
// ---------------------------------------------------------------------------

TEST_F(OcpPhaseLifetimeTest, RemovePhaseShiftsLinkIndices) {
    auto ocp = make_three_phase_ocp();       // phases 0,1,2
    add_forward_link(ocp, /*a=*/1, /*b=*/2); // link on {1,2}
    ocp.remove_phase(0);
    // remaining phases now {0,1}; the link must target {0,1} after shift.
    EXPECT_EQ(first_link_phases(ocp), (std::vector<int>{0, 1}));
}

TEST_F(OcpPhaseLifetimeTest, RemoveReferencedPhaseThrows) {
    auto ocp = make_three_phase_ocp();
    add_forward_link(ocp, 1, 2);
    EXPECT_THROW(ocp.remove_phase(1), std::invalid_argument); // 1 is referenced
}

TEST_F(OcpPhaseLifetimeTest, RemovePhaseAndPhaseOutOfRangeThrow) {
    auto ocp = make_three_phase_ocp();
    EXPECT_THROW(ocp.remove_phase(10), std::invalid_argument);
    EXPECT_THROW(ocp.remove_phase(-5), std::invalid_argument);
    EXPECT_THROW((void)ocp.phase(10), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// OC review §3.8: LinkFunction's "state-bindings-only" convenience
// constructors size their broadcast xtv/empty vectors off PTL[0].size()
// *before* delegating to init() (which only later checks nappl == 0). With
// an empty PTL that PTL[0] access is undefined behavior. Each of the four
// overload families exercised below must now throw std::invalid_argument
// instead. (The fully-bound constructors that route straight through init()
// already throw cleanly via its pre-existing nappl == 0 check; not
// re-tested here.)
// ---------------------------------------------------------------------------

TEST_F(OcpPhaseLifetimeTest, LinkFunctionRegFlagsStateBindingsRejectsEmptyPTL) {
    auto args = Arguments<1>();
    GenericFunction<-1, -1> identity(args);

    Eigen::Matrix<PhaseRegionFlags, -1, 1> reg_flags(2);
    reg_flags << PhaseRegionFlags::Back, PhaseRegionFlags::Front;

    std::vector<Eigen::VectorXi> empty_ptl;
    Eigen::VectorXi idx(1);
    idx << 0;
    std::vector<Eigen::VectorXi> xtv{idx};

    EXPECT_THROW(
        (tycho::oc::LinkFunction<GenericFunction<-1, -1>>(identity, reg_flags, empty_ptl, xtv)),
        std::invalid_argument);
}

TEST_F(OcpPhaseLifetimeTest, LinkFunctionLinkFlagStateBindingsRejectsEmptyPTL) {
    auto args = Arguments<1>();
    GenericFunction<-1, -1> identity(args);

    std::vector<Eigen::VectorXi> empty_ptl;
    Eigen::VectorXi idx(1);
    idx << 0;
    std::vector<Eigen::VectorXi> xtv{idx};

    EXPECT_THROW((tycho::oc::LinkFunction<GenericFunction<-1, -1>>(identity, LinkFlags::BackToFront,
                                                                   empty_ptl, xtv)),
                 std::invalid_argument);
}

TEST_F(OcpPhaseLifetimeTest, LinkFunctionRegFlagsSharedBindingRejectsEmptyPTL) {
    auto args = Arguments<1>();
    GenericFunction<-1, -1> identity(args);

    Eigen::Matrix<PhaseRegionFlags, -1, 1> reg_flags(2);
    reg_flags << PhaseRegionFlags::Back, PhaseRegionFlags::Front;

    std::vector<Eigen::VectorXi> empty_ptl;
    Eigen::VectorXi xtv(1);
    xtv << 0;

    EXPECT_THROW(
        (tycho::oc::LinkFunction<GenericFunction<-1, -1>>(identity, reg_flags, empty_ptl, xtv)),
        std::invalid_argument);
}

TEST_F(OcpPhaseLifetimeTest, LinkFunctionLinkFlagSharedBindingRejectsEmptyPTL) {
    auto args = Arguments<1>();
    GenericFunction<-1, -1> identity(args);

    std::vector<Eigen::VectorXi> empty_ptl;
    Eigen::VectorXi xtv(1);
    xtv << 0;

    EXPECT_THROW((tycho::oc::LinkFunction<GenericFunction<-1, -1>>(identity, LinkFlags::BackToFront,
                                                                   empty_ptl, xtv)),
                 std::invalid_argument);
}

TEST_F(OcpPhaseLifetimeTest, LinkFunctionLinkFlagNameSharedBindingRejectsEmptyPTL) {
    auto args = Arguments<1>();
    GenericFunction<-1, -1> identity(args);

    std::vector<Eigen::VectorXi> empty_ptl;
    Eigen::VectorXi xtv(1);
    xtv << 0;

    EXPECT_THROW((tycho::oc::LinkFunction<GenericFunction<-1, -1>>(
                     identity, std::string("BackToFront"), empty_ptl, xtv)),
                 std::invalid_argument);
}
