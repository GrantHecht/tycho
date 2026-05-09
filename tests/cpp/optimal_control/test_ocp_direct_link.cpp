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

TEST_F(OcpDirectLinkTest, ForwardLinkRejectsBadStartingPhaseWithoutPartialMutation) {
    // Symmetric coverage to the bad-fphase case: a bad iphase must throw
    // before any insertion. Pre-fix the fphase pre-check existed but the
    // iphase pre-check was added in the same commit; this test pins that the
    // iphase branch is reachable and equally exception-safe.
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);

    Eigen::VectorXi vars(1);
    vars << 0;

    const auto before = ocp.base().link_equalities_.size();

    EXPECT_THROW(ocp.base().add_forward_link_equal_con(99, 1, vars, ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}

TEST_F(OcpDirectLinkTest, ForwardLinkRejectsIphaseGreaterEqualFphase) {
    // iphase >= fphase used to be a silent no-op: the for-loop ran zero
    // iterations and returned an empty std::vector<int>, so the user got no
    // continuity constraints with no diagnostic. Now it must throw.
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

    // iphase > fphase: misordered.
    EXPECT_THROW(ocp.base().add_forward_link_equal_con(2, 0, vars, ScaleModes::AUTO),
                 std::invalid_argument);
    // iphase == fphase: linking a phase to itself.
    EXPECT_THROW(ocp.base().add_forward_link_equal_con(1, 1, vars, ScaleModes::AUTO),
                 std::invalid_argument);

    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}

TEST_F(OcpDirectLinkTest, ParamLinkRejectsIphaseGreaterEqualFphase) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);
    auto p2 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(2.0, 3.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);
    ocp.add_phase(p2);

    Eigen::VectorXi vars(0);

    const auto before = ocp.base().link_equalities_.size();

    EXPECT_THROW(ocp.base().add_param_link_equal_con(2, 0, PhaseRegionFlags::StaticParams, vars,
                                                     ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_THROW(ocp.base().add_param_link_equal_con(1, 1, PhaseRegionFlags::StaticParams, vars,
                                                     ScaleModes::AUTO),
                 std::invalid_argument);

    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}

TEST_F(OcpDirectLinkTest, AddLinkEqualConPartialMutationOnVarsSizeMismatch) {
    // Pins add_func_impl's validate-before-mutate ordering. Pre-fix:
    // map[index] = func ran, then check_function_size threw, leaving the
    // bad entry behind in link_equalities_. Now check_function_size runs
    // first and the map is untouched on throw.
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);

    // Functional with IRows = 4, but indexing implies IRows = 2 (one var per
    // side). check_function_size's PhaseToPhase IRows check (the
    // "Input size of {} (IRows = {}) does not match that implied by indexing
    // parameters (IRows = {})." branch) must fire.
    auto args = Arguments<-1>(4);
    auto bad_expr = args.head<-1>(2) - args.tail<-1>(2);
    GenericFunction<-1, -1> bad_func(bad_expr);

    Eigen::VectorXi v_one(1);
    v_one << 0;

    const auto before = ocp.base().link_equalities_.size();

    EXPECT_THROW(ocp.base().add_link_equal_con(bad_func, 0, PhaseRegionFlags::Back, v_one, 1,
                                               PhaseRegionFlags::Front, v_one, ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before)
        << "Failed insertion must leave the constraint map untouched";

    // Retry: pre-fix the first insert succeeded into the map and only the
    // size-check threw, so this would have produced two duplicate entries.
    EXPECT_THROW(ocp.base().add_link_equal_con(bad_func, 0, PhaseRegionFlags::Back, v_one, 1,
                                               PhaseRegionFlags::Front, v_one, ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}

// Heterogeneous-phase loop-pattern coverage. The per-call atomicity guaranteed
// by add_func_impl's validate-before-mutate ordering covers each iteration of
// the add_forward_link_equal_con loop. This test pins that guarantee for a
// case that the existing single-insertion AddLinkEqualConPartialMutationOnVarsSizeMismatch
// does not exercise: iter 0 succeeds (insert), iter 1 throws inside
// check_function_size. Pre-fix, iter 1 would have ALSO inserted before throwing,
// producing 2 entries instead of 1.
//
// NOTE: this test pins per-call atomicity, NOT loop-level rollback. A
// successful iter 0 insertion is preserved on iter 1 throw — the loop is not
// transactional. Body failures inside the loop on heterogeneous phases are
// acknowledged out-of-scope per the comment in add_forward_link_equal_con.
TEST_F(OcpDirectLinkTest, ForwardLinkPerIterationAtomicityOnHeterogeneousPhases) {
    // Three phases with the same ODE but different named-group definitions.
    // "g" resolves to size 1 in phases 0 and 1, size 2 in phase 2.
    auto ode_g1 = ODEBuilder(2, 1)
                      .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
                      .var_group("g", 0, 1)
                      .define([](auto &args) {
                          auto v = args.x_var(1);
                          auto u = args.u_var(0);
                          return stack(v, u);
                      })
                      .build();
    auto ode_g2 = ODEBuilder(2, 1)
                      .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
                      .var_group("g", 0, 2)
                      .define([](auto &args) {
                          auto v = args.x_var(1);
                          auto u = args.u_var(0);
                          return stack(v, u);
                      })
                      .build();

    auto p0 = ode_g1.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode_g1.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);
    auto p2 = ode_g2.phase(TranscriptionModes::LGL3, make_linear_guess_dl(2.0, 3.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);
    ocp.add_phase(p2);

    const auto before = ocp.base().link_equalities_.size();

    // vsize is computed from phase 0's "g" → size 1, so func.IRows = 2.
    // Iter 0 (link p0,p1): xtu sizes 1,1 → isize=2 == irows. OK, inserts.
    // Iter 1 (link p1,p2): xtu sizes 1,2 → isize=3 != irows=2. Throws.
    EXPECT_THROW(ocp.base().add_forward_link_equal_con(0, 2, std::string("g"),
                                                       ScaleModes::AUTO),
                 std::invalid_argument);

    // Iter 0's successful insertion is preserved (per-call atomicity, not
    // loop atomicity). Iter 1's failed insertion did NOT mutate, so the
    // total is exactly +1, not +2.
    EXPECT_EQ(ocp.base().link_equalities_.size(), before + 1)
        << "Iter 0 must remain inserted (per-call atomicity); iter 1 must not "
           "have mutated before throwing (validate-before-mutate ordering).";
}

// Region-rejection coverage for add_param_link_equal_con. Pre-validation at
// optimal_control_problem.h:657-659 rejects regions other than ODEParams or
// StaticParams before any mutation.
TEST_F(OcpDirectLinkTest, ParamLinkRejectsNonParamRegion) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);

    Eigen::VectorXi vars(0);

    const auto before = ocp.base().link_equalities_.size();

    // Front is not a param region — must be rejected up-front before any
    // mutation. Pre-validation lives at the top of add_param_link_equal_con.
    EXPECT_THROW(ocp.base().add_param_link_equal_con(0, 1, PhaseRegionFlags::Front, vars,
                                                     ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}

// Symmetric coverage to ParamLinkRejectsBadEndingPhaseWithoutPartialMutation.
// Pre-fix, the iphase pre-check existed for add_forward_link_equal_con but
// ParamLinkRejectsBadStartingPhase was not exercised. This test pins that
// the iphase branch is reachable and equally exception-safe.
TEST_F(OcpDirectLinkTest, ParamLinkRejectsBadStartingPhaseWithoutPartialMutation) {
    auto ode = make_direct_link_ode();
    auto p0 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(0.0, 1.0), 8);
    auto p1 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(1.0, 2.0), 8);
    auto p2 = ode.phase(TranscriptionModes::LGL3, make_linear_guess_dl(2.0, 3.0), 8);

    OptimalControlProblem ocp;
    ocp.add_phase(p0);
    ocp.add_phase(p1);
    ocp.add_phase(p2);

    Eigen::VectorXi vars(0);

    const auto before = ocp.base().link_equalities_.size();

    EXPECT_THROW(ocp.base().add_param_link_equal_con(99, 1, PhaseRegionFlags::StaticParams,
                                                     vars, ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_EQ(ocp.base().link_equalities_.size(), before);
}
