///////////////////////////////////////////////////////////////////////////////
// Native variable-bound path on a phase: ODEPhaseBase records bound
// declarations as VarBoundRecords and transcribe_var_bounds() resolves them to
// NLP variable indices, staging them through
// NonLinearProgram::set_variable_bound instead of lowering them to inequality
// constraints.
//
// TEST STRUCTURE
// ------------------------------------------------------------------------
//   * The recording and transcription machinery (record_var_bounds,
//     transcribe_var_bounds, the indexer resolution, the intersection and
//     conflict rules, multi-phase offsets) is exercised DIRECTLY through a
//     derived phase that reaches the protected members, so each rule can be
//     driven without going through a public declaration call.
//   * The public declaration sites (add_lu_var_bound / add_lower_var_bound /
//     add_upper_var_bound) are covered separately at the end of the file:
//     those tests pin that a declaration lands on the bound record list and
//     on the NLP's variable-bound contract, and that it produces no
//     inequality constraint.
//
// Hand-computed indexer layout used by the assertions below
// --------------------------------------------------------
// TychoTest::LinearODE has x_vars=2, u_vars=0, p_vars=0, no static params, so
// xtu_vars = 3 ([x, v, t]). Built LGL3 (2 cardinal states per defect) with 2
// defects and a non-blocked control mode:
//     num_states       = (2 - 1) * 2 + 1 = 3
//     num_phase_vars   = 3 * 3           = 9
//     state i occupies NLP indices [3i, 3i + 3)
// so for phase-local variable v:
//     Front        -> {v}
//     Back         -> {6 + v}
//     Path         -> {v, 3 + v, 6 + v}
//     FrontandBack -> {v, 6 + v}
//
// File-local symbol prefix (unity-build hygiene): NativeVarBounds*.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kNativeVarBoundsInf = std::numeric_limits<double>::infinity();

/// Number of NLP decision variables of a phase built by
/// make_native_var_bounds_phase(), per the layout note in the file header.
constexpr int kNativeVarBoundsPhaseVars = 9;

/// A LinearODE phase that exposes ODEPhaseBase's protected variable-bound
/// machinery, so the recording and transcription steps can be driven without
/// going through a public declaration call.
struct NativeVarBoundsPhase : ODEPhase<TychoTest::LinearODE> {
    using ODEPhase<TychoTest::LinearODE>::ODEPhase;

    /// Append a bound record directly, as record_var_bounds() would.
    void push_var_bound(PhaseRegionFlags reg, int var, double lower, double upper) {
        this->user_var_bounds_.push_back(VarBoundRecord{reg, var, lower, upper});
        this->reset_transcription();
    }

    /// Drive the declaration-site recording path irrespective of the switch.
    int record_bounds(PhaseRegionFlags reg, int var, double lower, double upper) {
        return this->record_var_bounds(reg, var, lower, upper);
    }
    int record_bounds(PhaseRegionFlags reg, const Eigen::VectorXi &vars, double lower,
                      double upper) {
        return this->record_var_bounds(reg, vars, lower, upper);
    }

    std::size_t num_var_bound_records() const { return this->user_var_bounds_.size(); }
    std::size_t num_inequalities() const { return this->user_inequalities_.size(); }

    int record_var(std::size_t i) const { return this->user_var_bounds_.at(i).var_; }
    PhaseRegionFlags record_region(std::size_t i) const {
        return this->user_var_bounds_.at(i).region_;
    }
    double record_lower(std::size_t i) const { return this->user_var_bounds_.at(i).lower_; }
    double record_upper(std::size_t i) const { return this->user_var_bounds_.at(i).upper_; }
};

/// Build a NativeVarBoundsPhase with the exact linear trajectory guess used by
/// TychoTest::make_linear_phase (x0=0, v0=1, t in [0, 1]) on 2 defects, giving
/// the 9-variable layout documented in the file header.
std::shared_ptr<NativeVarBoundsPhase> make_native_var_bounds_phase() {
    constexpr double x0 = 0.0, v0 = 1.0, t0 = 0.0, tf = 1.0;
    constexpr int n_pts = 5;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        double t = t0 + (tf - t0) * s;
        Eigen::VectorXd pt(3);
        pt[0] = x0 + v0 * (t - t0);
        pt[1] = v0;
        pt[2] = t;
        traj.push_back(pt);
    }

    TychoTest::LinearODE ode;
    return std::make_shared<NativeVarBoundsPhase>(ode, TranscriptionModes::LGL3, traj, 2);
}

/// A scalar function of one region variable, for the func-bound tests.
GenericFunction<-1, 1> native_var_bounds_scalar_fn() {
    return GenericFunction<-1, 1>(Arguments<1>().coeff<0>());
}

///////////////////////////////////////////////////////////////////////////////
// A LinearODE variant carrying one ODE parameter, for the ODEParams arm of the
// scaled-bound unit lookup.
//
// TychoTest::LinearODE has p_vars == 0, so no fixture built on it can reach that
// arm at all. Here x_vars=2, u_vars=0, p_vars=1, giving the input layout
// [x, v, t, p]: xtu_vars() == 3 and xtu_p_vars() == 4, so the ODE parameter's
// scaling unit is the LAST entry of xtup_units_ and is reached only through the
// + xtu_vars() offset. Dynamics are xdot = v, vdot = p (the parameter is a
// constant acceleration) -- the tests below never solve, so only the sizes and
// the parameter's presence in the index space matter.
///////////////////////////////////////////////////////////////////////////////

struct NativeVarBoundsParamODE_Impl : ODESize<2, 0, 1> {
    static auto Definition() {
        auto args = Arguments<4>(); // [x, v, t, p]
        auto v = args.coeff<1>();
        auto p = args.coeff<3>();
        return StackedOutputs{v, p};
    }
};
BUILD_ODE_FROM_EXPRESSION(NativeVarBoundsParamODE, NativeVarBoundsParamODE_Impl);

/// The NativeVarBoundsPhase equivalent for the parameter-bearing ODE: exposes
/// the protected bound-record list so a declaration can be staged without
/// depending on the process-wide declaration-site switch.
struct NativeVarBoundsParamPhase : ODEPhase<NativeVarBoundsParamODE> {
    using ODEPhase<NativeVarBoundsParamODE>::ODEPhase;

    void push_var_bound(PhaseRegionFlags reg, int var, double lower, double upper) {
        this->user_var_bounds_.push_back(VarBoundRecord{reg, var, lower, upper});
        this->reset_transcription();
    }
};

/// Build a NativeVarBoundsParamPhase on the same 2-defect LGL3 mesh as
/// make_native_var_bounds_phase(), with the ODE-parameter column held constant.
std::shared_ptr<NativeVarBoundsParamPhase> make_native_var_bounds_param_phase() {
    constexpr double x0 = 0.0, v0 = 1.0, t0 = 0.0, tf = 1.0, p0 = 0.0;
    constexpr int n_pts = 5;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        double t = t0 + (tf - t0) * s;
        Eigen::VectorXd pt(4); // [x, v, t, p]
        pt[0] = x0 + v0 * (t - t0);
        pt[1] = v0;
        pt[2] = t;
        pt[3] = p0;
        traj.push_back(pt);
    }

    NativeVarBoundsParamODE ode;
    return std::make_shared<NativeVarBoundsParamPhase>(ode, TranscriptionModes::LGL3, traj, 2);
}

/// The index of the single bounded variable in @p nlp, asserting there is
/// exactly one. Used where the NLP slot of a parameter is not worth hardcoding.
int native_var_bounds_only_bounded_index(
    const std::shared_ptr<tycho::solvers::NonLinearProgram> &nlp) {
    int bounded = -1;
    for (int i = 0; i < nlp->x_lower_.size(); ++i) {
        if (nlp->x_lower_[i] != -kNativeVarBoundsInf || nlp->x_upper_[i] != kNativeVarBoundsInf) {
            EXPECT_EQ(bounded, -1) << "more than one variable was bounded, at " << i;
            bounded = i;
        }
    }
    return bounded;
}

/// Assert that every NLP variable except those in @p bounded is unbounded.
void native_var_bounds_expect_only(const std::shared_ptr<tycho::solvers::NonLinearProgram> &nlp,
                                   const std::vector<int> &bounded) {
    for (int i = 0; i < nlp->x_lower_.size(); ++i) {
        if (std::find(bounded.begin(), bounded.end(), i) != bounded.end()) {
            continue;
        }
        EXPECT_EQ(nlp->x_lower_[i], -kNativeVarBoundsInf) << "index " << i;
        EXPECT_EQ(nlp->x_upper_[i], kNativeVarBoundsInf) << "index " << i;
    }
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Region resolution: recorded bounds land on the indices the phase indexer
// assigns to that region's applications of the variable.
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBounds, FrontRegionResolvesToTheFirstStateBlock) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.5, 3.5);
    phase->transcribe();

    ASSERT_EQ(phase->nlp_->x_lower_.size(), kNativeVarBoundsPhaseVars);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[0], -1.5);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[0], 3.5);
    native_var_bounds_expect_only(phase->nlp_, {0});
    EXPECT_TRUE(phase->nlp_->has_variable_bounds());
}

TEST(PhaseNativeVarBounds, BackRegionResolvesToTheLastStateBlock) {
    auto phase = make_native_var_bounds_phase();
    // Variable 2 is the time column; Back puts it at 6 + 2 == 8.
    phase->push_var_bound(PhaseRegionFlags::Back, /*var=*/2, 0.5, 2.0);
    phase->transcribe();

    ASSERT_EQ(phase->nlp_->x_lower_.size(), kNativeVarBoundsPhaseVars);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[8], 0.5);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[8], 2.0);
    native_var_bounds_expect_only(phase->nlp_, {8});
}

TEST(PhaseNativeVarBounds, PathRegionExpandsToEveryNodeFromOneRecord) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -4.0, 4.0);
    ASSERT_EQ(phase->num_var_bound_records(), 1u)
        << "Path must stay one record before transcription";

    phase->transcribe();

    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -4.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 4.0) << "index " << gidx;
    }
    native_var_bounds_expect_only(phase->nlp_, {1, 4, 7});
}

TEST(PhaseNativeVarBounds, FrontAndBackRegionCoversBothEndpoints) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::FrontandBack, /*var=*/1, -2.0, 2.0);
    phase->transcribe();

    for (int gidx : {1, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -2.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 2.0) << "index " << gidx;
    }
    native_var_bounds_expect_only(phase->nlp_, {1, 7});
}

TEST(PhaseNativeVarBounds, LowerOnlyAndUpperOnlyRecordsLeaveTheOtherSideOpen) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0, kNativeVarBoundsInf);
    phase->push_var_bound(PhaseRegionFlags::Back, /*var=*/0, -kNativeVarBoundsInf, 7.0);
    phase->transcribe();

    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[0], -1.0);
    EXPECT_EQ(phase->nlp_->x_upper_[0], kNativeVarBoundsInf);
    EXPECT_EQ(phase->nlp_->x_lower_[6], -kNativeVarBoundsInf);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[6], 7.0);
}

///////////////////////////////////////////////////////////////////////////////
// Intersection and conflict rules.
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBounds, OverlappingDeclarationsIntersectToTheTightestInterval) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, 0.0, 10.0);
    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, 2.0, 6.0);
    phase->transcribe();

    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], 2.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 6.0) << "index " << gidx;
    }
}

TEST(PhaseNativeVarBounds, PathAndFrontDeclarationsIntersectOnlyWhereTheyOverlap) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, 0.0, 10.0);
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/1, 3.0, 4.0);
    phase->transcribe();

    // The Front declaration only tightens the first node.
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[1], 3.0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[1], 4.0);
    for (int gidx : {4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], 0.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 10.0) << "index " << gidx;
    }
}

TEST(PhaseNativeVarBounds, ConflictingDeclarationsThrowNamingThePhaseAndVariable) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/2, 5.0, 10.0);
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/2, 20.0, 30.0);

    try {
        phase->transcribe();
        FAIL() << "expected std::invalid_argument for an empty bound intersection";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("phase 0"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Front"), std::string::npos) << msg;
        EXPECT_NE(msg.find("variable index 2"), std::string::npos) << msg;
    }
}

TEST(PhaseNativeVarBounds, EqualBoundsAreAcceptedAsAFixedVariable) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/0, 4.0, 4.0);
    EXPECT_NO_THROW(phase->transcribe());

    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[0], 4.0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[0], 4.0);
}

TEST(PhaseNativeVarBounds, OutOfRangeRecordedIndexThrowsWithPhaseContext) {
    auto phase = make_native_var_bounds_phase();
    // xtu_vars == 3, so index 7 has no state/time/control slot.
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/7, -1.0, 1.0);

    try {
        phase->transcribe();
        FAIL() << "expected std::invalid_argument for an out-of-range bound index";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("phase 0"), std::string::npos) << msg;
        EXPECT_NE(msg.find("7"), std::string::npos) << msg;
    }
}

TEST(PhaseNativeVarBounds, NoRecordsLeavesEveryVariableUnbounded) {
    auto phase = make_native_var_bounds_phase();
    phase->transcribe();

    ASSERT_EQ(phase->nlp_->x_lower_.size(), kNativeVarBoundsPhaseVars);
    EXPECT_FALSE(phase->nlp_->has_variable_bounds());
}

TEST(PhaseNativeVarBounds, ReTranscriptionReproducesTheSameBounds) {
    auto phase = make_native_var_bounds_phase();
    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -3.0, 3.0);

    phase->transcribe();
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[4], -3.0);

    // Each transcribe() installs a fresh NonLinearProgram, so the staged
    // declarations never accumulate across calls.
    phase->transcribe();
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[4], -3.0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[4], 3.0);
    EXPECT_EQ(phase->nlp_->staged_variable_bounds_.size(), 3u);
}

///////////////////////////////////////////////////////////////////////////////
// Auto-scaling: a declared bound is in physical units, but the NLP decision
// variable is the physical value divided by that variable's scaling unit
// (make_solver_input divides on the way in, collect_solver_output multiplies on
// the way out). The inequality lowering this path replaces kept comparing
// physical values -- add_inequal_con wrapped the bound expression in IOScaled
// with get_input_scale's units, undoing the packing before the comparison -- so
// a bound staged straight onto the NLP variable has to carry the same division.
//
// LinearODE has x_vars=2, u_vars=0, p_vars=0, so xtu_p_vars == 3 and set_units
// takes [x, v, t].
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBoundsScaling, StagedBoundsAreDividedByTheVariableUnit) {
    auto phase = make_native_var_bounds_phase();
    Eigen::VectorXd units(3);
    units << 1.0, 4.0, 2.0;
    phase->set_auto_scaling(true);
    phase->set_units(units);

    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -3.0, 5.0);
    phase->transcribe();

    // v carries unit 4, so the physical box [-3, 5] is the scaled box
    // [-0.75, 1.25] on every node the Path region touches.
    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -0.75) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 1.25) << "index " << gidx;
    }
    native_var_bounds_expect_only(phase->nlp_, {1, 4, 7});
}

TEST(PhaseNativeVarBoundsScaling, EachVariableIsDividedByItsOwnUnit) {
    auto phase = make_native_var_bounds_phase();
    Eigen::VectorXd units(3);
    units << 1.0, 4.0, 2.0;
    phase->set_auto_scaling(true);
    phase->set_units(units);

    // x (unit 1) is untouched by the division; t (unit 2) is halved.
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.5, 3.5);
    phase->push_var_bound(PhaseRegionFlags::Back, /*var=*/2, 1.0, 9.0);
    phase->transcribe();

    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[0], -1.5);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[0], 3.5);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[8], 0.5);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[8], 4.5);
    native_var_bounds_expect_only(phase->nlp_, {0, 8});
}

TEST(PhaseNativeVarBoundsScaling, UnitUnitsLeaveTheDeclaredBoundUnchanged) {
    auto phase = make_native_var_bounds_phase();
    phase->set_auto_scaling(true);
    phase->set_units(Eigen::VectorXd::Ones(3));

    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -3.0, 5.0);
    phase->transcribe();

    // The all-ones default must divide out exactly, not scale twice.
    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -3.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 5.0) << "index " << gidx;
    }
}

TEST(PhaseNativeVarBoundsScaling, UnitsAreIgnoredWhenAutoScalingIsOff) {
    auto phase = make_native_var_bounds_phase();
    Eigen::VectorXd units(3);
    units << 1.0, 4.0, 2.0;
    // Units are set but never consumed: without auto-scaling make_solver_input
    // packs physical values, so the staged bound must stay physical too.
    phase->set_units(units);

    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -3.0, 5.0);
    phase->transcribe();

    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -3.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 5.0) << "index " << gidx;
    }
}

TEST(PhaseNativeVarBoundsScaling, OpenBoundSidesSurviveTheUnitDivision) {
    auto phase = make_native_var_bounds_phase();
    Eigen::VectorXd units(3);
    units << 1.0, 4.0, 2.0;
    phase->set_auto_scaling(true);
    phase->set_units(units);

    phase->push_var_bound(PhaseRegionFlags::Back, /*var=*/2, -kNativeVarBoundsInf, 9.0);
    phase->push_var_bound(PhaseRegionFlags::Front, /*var=*/1, -6.0, kNativeVarBoundsInf);
    phase->transcribe();

    EXPECT_EQ(phase->nlp_->x_lower_[8], -kNativeVarBoundsInf);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[8], 4.5);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[1], -1.5);
    EXPECT_EQ(phase->nlp_->x_upper_[1], kNativeVarBoundsInf);
}

TEST(PhaseNativeVarBoundsScaling, StaticParamBoundsUseTheStaticParamUnits) {
    auto phase = make_native_var_bounds_phase();
    Eigen::VectorXd parm(1);
    parm << 3.0;
    Eigen::VectorXd sp_units(1);
    sp_units << 5.0;
    phase->set_static_params(parm, sp_units);
    phase->set_auto_scaling(true);
    phase->set_units(Eigen::VectorXd::Ones(3));

    phase->push_var_bound(PhaseRegionFlags::StaticParams, /*var=*/0, -10.0, 20.0);
    phase->transcribe();

    // The static-parameter index is drawn from sp_units_, not xtup_units_ (all
    // ones here), so a scaled bound proves the right unit array was consulted.
    // Located by scan rather than by a hardcoded index: this is the only bounded
    // variable, and its NLP slot sits past the phase's own variable block.
    int bounded = native_var_bounds_only_bounded_index(phase->nlp_);
    ASSERT_GE(bounded, kNativeVarBoundsPhaseVars);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[bounded], -2.0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[bounded], 4.0);
}

TEST(PhaseNativeVarBoundsScaling, ODEParamBoundsUseTheOffsetParameterUnit) {
    auto phase = make_native_var_bounds_param_phase();
    // [x, v, t, p]: every state/time slot is one, so ONLY the parameter's own
    // unit can scale the bound. A lookup that dropped the + xtu_vars() offset
    // would read xtup_units_[0] == 1 and stage the declared numbers unscaled;
    // a lookup that reached for sp_units_ would find an empty array. Either way
    // the assertion below fails, which is the point of the all-ones padding.
    Eigen::VectorXd units(4);
    units << 1.0, 1.0, 1.0, 5.0;
    phase->set_auto_scaling(true);
    phase->set_units(units);

    phase->push_var_bound(PhaseRegionFlags::ODEParams, /*var=*/0, -10.0, 20.0);
    phase->transcribe();

    int bounded = native_var_bounds_only_bounded_index(phase->nlp_);
    ASSERT_GE(bounded, 0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[bounded], -2.0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[bounded], 4.0);
}

TEST(PhaseNativeVarBoundsScaling, ODEParamUnitsAreIgnoredWhenAutoScalingIsOff) {
    auto phase = make_native_var_bounds_param_phase();
    Eigen::VectorXd units(4);
    units << 1.0, 1.0, 1.0, 5.0;
    phase->set_units(units);

    phase->push_var_bound(PhaseRegionFlags::ODEParams, /*var=*/0, -10.0, 20.0);
    phase->transcribe();

    int bounded = native_var_bounds_only_bounded_index(phase->nlp_);
    ASSERT_GE(bounded, 0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[bounded], -10.0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[bounded], 20.0);
}

TEST(PhaseNativeVarBoundsScaling, EachPhaseUsesItsOwnUnitsInAMultiPhaseProblem) {
    auto phase0 = make_native_var_bounds_phase();
    auto phase1 = make_native_var_bounds_phase();

    // Same declaration in both phases, different unit on the bounded variable:
    // the two staged boxes must differ, which they can only do if the divisor is
    // read per phase. Auto-scaling is set on the phases rather than on the
    // problem because that is the flag the packing reads too --
    // OptimalControlProblemBase::make_solver_input delegates to each phase's own
    // make_solver_input, so each phase's variables are scaled by its own units.
    Eigen::VectorXd units0(3);
    units0 << 1.0, 4.0, 2.0;
    Eigen::VectorXd units1(3);
    units1 << 1.0, 0.5, 2.0;
    phase0->set_auto_scaling(true);
    phase0->set_units(units0);
    phase1->set_auto_scaling(true);
    phase1->set_units(units1);

    phase0->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -3.0, 5.0);
    phase1->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -3.0, 5.0);

    OptimalControlProblemBase ocp;
    ocp.add_phase(phase0);
    ocp.add_phase(phase1);
    ocp.transcribe();

    ASSERT_EQ(ocp.nlp_->x_lower_.size(), 2 * kNativeVarBoundsPhaseVars);

    // Phase 0 owns indices [0, 9) and divides by 4; phase 1 owns [9, 18) and
    // divides by 0.5.
    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(ocp.nlp_->x_lower_[gidx], -0.75) << "index " << gidx;
        EXPECT_DOUBLE_EQ(ocp.nlp_->x_upper_[gidx], 1.25) << "index " << gidx;
    }
    for (int gidx : {9 + 1, 9 + 4, 9 + 7}) {
        EXPECT_DOUBLE_EQ(ocp.nlp_->x_lower_[gidx], -6.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(ocp.nlp_->x_upper_[gidx], 10.0) << "index " << gidx;
    }
    native_var_bounds_expect_only(ocp.nlp_, {1, 4, 7, 10, 13, 16});
}

TEST(PhaseNativeVarBoundsScaling, SetUnitsRetranscriptionRestagesAgainstTheNewUnits) {
    auto phase = make_native_var_bounds_phase();
    phase->set_auto_scaling(true);

    Eigen::VectorXd first(3);
    first << 1.0, 4.0, 2.0;
    phase->set_units(first);
    phase->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -3.0, 5.0);
    phase->transcribe();
    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -0.75) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 1.25) << "index " << gidx;
    }

    // set_units() resets the transcription; the records outlive it, and the next
    // transcribe() must re-divide them by the NEW units rather than carrying the
    // values staged against the old ones.
    Eigen::VectorXd second(3);
    second << 1.0, 0.5, 2.0;
    phase->set_units(second);
    phase->transcribe();

    EXPECT_EQ(phase->num_var_bound_records(), 1u) << "set_units must not drop the declaration";
    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -6.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 10.0) << "index " << gidx;
    }
    native_var_bounds_expect_only(phase->nlp_, {1, 4, 7});
}

TEST(PhaseNativeVarBoundsScaling, NonPositiveUnitUnderAutoScalingThrowsWithContext) {
    auto phase = make_native_var_bounds_phase();
    Eigen::VectorXd units(3);
    units << 1.0, 1.0, -2.0;
    phase->set_auto_scaling(true);
    phase->set_units(units);

    phase->push_var_bound(PhaseRegionFlags::Back, /*var=*/2, 1.0, 9.0);

    try {
        phase->transcribe();
        FAIL() << "expected std::invalid_argument for a non-positive scaling unit";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("phase 0"), std::string::npos) << msg;
        EXPECT_NE(msg.find("variable index 2"), std::string::npos) << msg;
        EXPECT_NE(msg.find("finite and positive"), std::string::npos) << msg;
    }
}

TEST(PhaseNativeVarBoundsScaling, NonFiniteUnitUnderAutoScalingThrowsWithContext) {
    // set_units() validates only the vector's length, so this guard -- not an
    // upstream check -- is what stops a non-finite unit. The NaN leg matters
    // independently of the infinity leg: NaN fails every ordered comparison, so
    // `unit <= 0.0` is false for it and only the !isfinite clause catches it.
    const double bad_units[] = {kNativeVarBoundsInf, -kNativeVarBoundsInf,
                                std::numeric_limits<double>::quiet_NaN()};

    for (double bad : bad_units) {
        auto phase = make_native_var_bounds_phase();
        Eigen::VectorXd units(3);
        units << 1.0, 1.0, bad;
        phase->set_auto_scaling(true);
        phase->set_units(units);

        phase->push_var_bound(PhaseRegionFlags::Back, /*var=*/2, 1.0, 9.0);

        try {
            phase->transcribe();
            FAIL() << "expected std::invalid_argument for a non-finite scaling unit (" << bad
                   << ")";
        } catch (const std::invalid_argument &e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("phase 0"), std::string::npos) << msg;
            EXPECT_NE(msg.find("variable index 2"), std::string::npos) << msg;
            EXPECT_NE(msg.find("finite and positive"), std::string::npos) << msg;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Multi-phase: the same resolution picks up the phase's variable offset.
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBounds, TwoPhaseProblemOffsetsGlobalIndices) {
    auto phase0 = make_native_var_bounds_phase();
    auto phase1 = make_native_var_bounds_phase();
    phase0->push_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0, 1.0);
    phase1->push_var_bound(PhaseRegionFlags::Path, /*var=*/1, -5.0, 5.0);

    OptimalControlProblemBase ocp;
    ocp.add_phase(phase0);
    ocp.add_phase(phase1);
    ocp.transcribe();

    ASSERT_EQ(ocp.nlp_->x_lower_.size(), 2 * kNativeVarBoundsPhaseVars);

    // Phase 0 owns indices [0, 9); phase 1 owns [9, 18).
    EXPECT_DOUBLE_EQ(ocp.nlp_->x_lower_[0], -1.0);
    EXPECT_DOUBLE_EQ(ocp.nlp_->x_upper_[0], 1.0);
    for (int gidx : {9 + 1, 9 + 4, 9 + 7}) {
        EXPECT_DOUBLE_EQ(ocp.nlp_->x_lower_[gidx], -5.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(ocp.nlp_->x_upper_[gidx], 5.0) << "index " << gidx;
    }
    native_var_bounds_expect_only(ocp.nlp_, {0, 10, 13, 16});
}

TEST(PhaseNativeVarBounds, SecondPhaseConflictNamesItsOwnPhaseNumber) {
    auto phase0 = make_native_var_bounds_phase();
    auto phase1 = make_native_var_bounds_phase();
    phase1->push_var_bound(PhaseRegionFlags::Back, /*var=*/0, 5.0, 10.0);
    phase1->push_var_bound(PhaseRegionFlags::Back, /*var=*/0, 20.0, 30.0);

    OptimalControlProblemBase ocp;
    ocp.add_phase(phase0);
    ocp.add_phase(phase1);

    try {
        ocp.transcribe();
        FAIL() << "expected std::invalid_argument for an empty bound intersection";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("phase 1"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Back"), std::string::npos) << msg;
    }
}

///////////////////////////////////////////////////////////////////////////////
// record_var_bounds(): the declaration-site recording step itself.
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBounds, RecordVarBoundsAppendsOneRecordPerResolvedVariable) {
    auto phase = make_native_var_bounds_phase();

    int first = phase->record_bounds(PhaseRegionFlags::Path, /*var=*/1, -1.0, 1.0);
    EXPECT_EQ(first, 0);
    ASSERT_EQ(phase->num_var_bound_records(), 1u);
    EXPECT_EQ(phase->record_region(0), PhaseRegionFlags::Path);
    EXPECT_EQ(phase->record_var(0), 1);
    EXPECT_DOUBLE_EQ(phase->record_lower(0), -1.0);
    EXPECT_DOUBLE_EQ(phase->record_upper(0), 1.0);

    Eigen::VectorXi vars(2);
    vars << 0, 2;
    int second = phase->record_bounds(PhaseRegionFlags::Front, vars, -2.0, 2.0);
    EXPECT_EQ(second, 1) << "handle must point at the first record of the declaration";
    ASSERT_EQ(phase->num_var_bound_records(), 3u);
    EXPECT_EQ(phase->record_var(1), 0);
    EXPECT_EQ(phase->record_var(2), 2);

    // Recording never touches the inequality-constraint store.
    EXPECT_EQ(phase->num_inequalities(), 0u);
}

TEST(PhaseNativeVarBounds, RecordVarBoundsRejectsUnusableRegions) {
    auto phase = make_native_var_bounds_phase();

    // Params cannot be split into ODE- and static-parameter indices.
    EXPECT_THROW(phase->record_bounds(PhaseRegionFlags::Params, 0, -1.0, 1.0),
                 std::invalid_argument);
    // Internal sentinels are not user-selectable regions.
    EXPECT_THROW(phase->record_bounds(PhaseRegionFlags::BlockDefectPath, 0, -1.0, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(phase->record_bounds(PhaseRegionFlags::NotSet, 0, -1.0, 1.0),
                 std::invalid_argument);

    // A rejected declaration leaves nothing behind.
    EXPECT_EQ(phase->num_var_bound_records(), 0u);
}

TEST(PhaseNativeVarBounds, RecordVarBoundsRejectsAnEmptySelector) {
    auto phase = make_native_var_bounds_phase();
    Eigen::VectorXi empty(0);
    EXPECT_THROW(phase->record_bounds(PhaseRegionFlags::Front, empty, -1.0, 1.0),
                 std::invalid_argument);
    EXPECT_EQ(phase->num_var_bound_records(), 0u);
}

// Bound records and inequality constraints live in separate stores with
// separate handle spaces, so a recorded bound must not disturb the
// inequality-index operations. Nothing translates between the two handle
// spaces, which is exactly why a bound-record handle must never be passed to
// these entry points.
TEST(PhaseNativeVarBounds, InequalityIndexOperationsAreUnaffectedByRecordedBounds) {
    auto phase = make_native_var_bounds_phase();
    // A genuine inequality constraint, added the ordinary way.
    int handle = phase->add_lu_func_bound(PhaseRegionFlags::Front, native_var_bounds_scalar_fn(), 0,
                                          -1.0, 1.0, 1.0, 1.0, ScaleModes::AUTO);
    ASSERT_EQ(phase->num_inequalities(), 1u);

    phase->record_bounds(PhaseRegionFlags::Front, 0, -1.0, 1.0);
    ASSERT_EQ(phase->num_var_bound_records(), 1u);

    // The inequality's own index still reads and removes its own constraint.
    EXPECT_NO_THROW(phase->return_inequal_con_scales(handle));
    EXPECT_NO_THROW(phase->remove_inequal_con(handle));
    EXPECT_EQ(phase->num_inequalities(), 0u);
    EXPECT_EQ(phase->num_var_bound_records(), 1u) << "removal must not touch the bound records";
}

///////////////////////////////////////////////////////////////////////////////
// Bound kinds that are constraints on expressions, not on variables: function
// bounds, norm bounds, and delta-variable bounds. They stay inequality
// constraints and never produce a bound record.
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBounds, FuncNormAndDeltaBoundsRemainInequalityConstraints) {
    auto phase = make_native_var_bounds_phase();

    phase->add_lu_func_bound(PhaseRegionFlags::Front, native_var_bounds_scalar_fn(), 0, -1.0, 1.0,
                             1.0, 1.0, ScaleModes::AUTO);
    EXPECT_EQ(phase->num_inequalities(), 1u);

    Eigen::VectorXi norm_vars(2);
    norm_vars << 0, 1;
    phase->add_lu_norm_bound(PhaseRegionFlags::Front, norm_vars, 0.0, 5.0, 1.0, 1.0,
                             ScaleModes::AUTO);
    EXPECT_EQ(phase->num_inequalities(), 2u);

    phase->add_lower_delta_var_bound(/*var=*/2, 0.1, 1.0, ScaleModes::AUTO);
    EXPECT_EQ(phase->num_inequalities(), 3u);

    phase->add_upper_delta_var_bound(/*var=*/2, 10.0, 1.0, ScaleModes::AUTO);
    EXPECT_EQ(phase->num_inequalities(), 4u);

    // None of them produced a bound record.
    EXPECT_EQ(phase->num_var_bound_records(), 0u);

    phase->transcribe();
    EXPECT_FALSE(phase->nlp_->has_variable_bounds());
}

///////////////////////////////////////////////////////////////////////////////
// The public declaration sites. Everything above drives the protected
// machinery directly; these pin that add_lu_var_bound / add_lower_var_bound /
// add_upper_var_bound reach it -- a declaration lands on the bound record list
// and on the NLP's variable-bound contract, and produces no inequality
// constraint.
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBounds, LuVarBoundDeclarationLandsOnTheNlpBoundContract) {
    auto phase = make_native_var_bounds_phase();
    int handle = phase->add_lu_var_bound(PhaseRegionFlags::Path, /*var=*/1, -1.0, 2.0);

    EXPECT_EQ(handle, 0);
    EXPECT_EQ(phase->num_inequalities(), 0u);
    ASSERT_EQ(phase->num_var_bound_records(), 1u);
    EXPECT_EQ(phase->record_region(0), PhaseRegionFlags::Path);
    EXPECT_DOUBLE_EQ(phase->record_lower(0), -1.0);
    EXPECT_DOUBLE_EQ(phase->record_upper(0), 2.0);

    phase->transcribe();
    ASSERT_TRUE(phase->nlp_->has_variable_bounds());
    native_var_bounds_expect_only(phase->nlp_, {1, 4, 7});
    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -1.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 2.0) << "index " << gidx;
    }
}

// add_lu_var_bounds (plural) accepts a multi-variable selector, something the
// old inequality-row lowering never supported. It must resolve to one bound
// record per variable, not one record spanning the whole selector, and each
// record's box must reach every NLP index the Path region expands to.
TEST(PhaseNativeVarBounds, MultiVariableSelectorRecordsOneBoundPerVariable) {
    auto phase = make_native_var_bounds_phase();

    Eigen::VectorXi vars(2);
    vars << 0, 1;
    Eigen::VectorXi handles = phase->add_lu_var_bounds(PhaseRegionFlags::Path, vars, -3.0, 3.0);

    EXPECT_EQ(handles.size(), 2);
    EXPECT_EQ(phase->num_inequalities(), 0u);
    ASSERT_EQ(phase->num_var_bound_records(), 2u)
        << "one record per resolved variable, not one record for the whole selector";
    EXPECT_EQ(phase->record_var(0), 0);
    EXPECT_EQ(phase->record_var(1), 1);
    for (std::size_t i = 0; i < 2u; ++i) {
        EXPECT_EQ(phase->record_region(i), PhaseRegionFlags::Path);
        EXPECT_DOUBLE_EQ(phase->record_lower(i), -3.0);
        EXPECT_DOUBLE_EQ(phase->record_upper(i), 3.0);
    }

    phase->transcribe();
    ASSERT_TRUE(phase->nlp_->has_variable_bounds());
    // Path expands var 0 to {0, 3, 6} and var 1 to {1, 4, 7}.
    for (int gidx : {0, 1, 3, 4, 6, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -3.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 3.0) << "index " << gidx;
    }
    native_var_bounds_expect_only(phase->nlp_, {0, 1, 3, 4, 6, 7});
}

TEST(PhaseNativeVarBounds, LowerAndUpperVarBoundDeclarationsLeaveTheOtherSideOpen) {
    auto phase = make_native_var_bounds_phase();
    phase->add_lower_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0);
    phase->add_upper_var_bound(PhaseRegionFlags::Back, /*var=*/0, 7.0);

    ASSERT_EQ(phase->num_var_bound_records(), 2u);
    EXPECT_EQ(phase->num_inequalities(), 0u);
    EXPECT_DOUBLE_EQ(phase->record_lower(0), -1.0);
    EXPECT_EQ(phase->record_upper(0), kNativeVarBoundsInf);
    EXPECT_EQ(phase->record_lower(1), -kNativeVarBoundsInf);
    EXPECT_DOUBLE_EQ(phase->record_upper(1), 7.0);

    phase->transcribe();
    ASSERT_TRUE(phase->nlp_->has_variable_bounds());
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[0], -1.0);
    EXPECT_EQ(phase->nlp_->x_upper_[0], kNativeVarBoundsInf);
    EXPECT_EQ(phase->nlp_->x_lower_[6], -kNativeVarBoundsInf);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[6], 7.0);
}

TEST(PhaseNativeVarBounds, LuVarBoundDeclarationRejectsAnInvertedInterval) {
    auto phase = make_native_var_bounds_phase();

    EXPECT_THROW(phase->add_lu_var_bound(PhaseRegionFlags::Front, /*var=*/0, 2.0, 1.0),
                 std::invalid_argument);

    // A rejected declaration leaves nothing behind, in either store.
    EXPECT_EQ(phase->num_var_bound_records(), 0u);
    EXPECT_EQ(phase->num_inequalities(), 0u);
}
