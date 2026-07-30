///////////////////////////////////////////////////////////////////////////////
// Native variable-bound path on a phase: ODEPhaseBase records bound
// declarations as VarBoundRecords and transcribe_var_bounds() resolves them to
// NLP variable indices, staging them through
// NonLinearProgram::set_variable_bound instead of lowering them to inequality
// constraints.
//
// TEST STRUCTURE — how the environment-dependent tests stay unconditional
// ------------------------------------------------------------------------
// The declaration sites (add_lu_var_bound / add_lower_var_bound /
// add_upper_var_bound) choose between the old inequality lowering and the
// native recording path through a file-local helper in
// src/optimal_control/ode_phase_base.cpp that reads TYCHO_DEV_NATIVE_BOUNDS
// fresh on every call — it is a setup-path check (bound declaration time),
// not a hot path, so there is no process-wide caching to work around. Each
// test that needs a specific switch state sets TYCHO_DEV_NATIVE_BOUNDS
// in-fixture (PhaseNativeVarBoundsSwitchOn / PhaseNativeVarBoundsSwitchOff
// below) and restores the prior environment (including "was unset") in
// TearDown, so every test in this file runs unconditionally in the default
// tycho_tests binary — none of them skip.
//
// The one rule this relies on: do not toggle TYCHO_DEV_NATIVE_BOUNDS between
// a phase's bound declarations and that phase's transcription. Declarations
// record into user_var_bounds_ (native) or user_inequalities_ (lowered)
// under whichever state was active when they were made; transcribe_phase
// only ever consults those two stores, never the switch itself. Toggling
// mid-phase would not corrupt anything, but it would produce a phase with a
// mix of records and inequalities that no test here intends to exercise, so
// each fixture holds the switch fixed for its own test's whole body.
//
// Consequently:
//   * The recording and transcription machinery (record_var_bounds,
//     transcribe_var_bounds, the indexer resolution, the intersection and
//     conflict rules, multi-phase offsets) is exercised DIRECTLY through a
//     derived phase that reaches the protected members, independent of the
//     switch.
//   * The routing question — "does add_lu_var_bound record instead of lower
//     when the switch is on (or keep lowering when it's off)?" — is covered
//     by the PhaseNativeVarBoundsSwitchOn / PhaseNativeVarBoundsSwitchOff
//     fixtures below, which set/clear TYCHO_DEV_NATIVE_BOUNDS for their
//     duration.
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
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kNativeVarBoundsInf = std::numeric_limits<double>::infinity();
constexpr const char *kNativeVarBoundsEnvVar = "TYCHO_DEV_NATIVE_BOUNDS";

/// Number of NLP decision variables of a phase built by
/// make_native_var_bounds_phase(), per the layout note in the file header.
constexpr int kNativeVarBoundsPhaseVars = 9;

void native_var_bounds_set_env(const char *value) {
#ifdef _WIN32
    _putenv_s(kNativeVarBoundsEnvVar, value);
#else
    setenv(kNativeVarBoundsEnvVar, value, 1);
#endif
}

void native_var_bounds_clear_env() {
#ifdef _WIN32
    _putenv_s(kNativeVarBoundsEnvVar, "");
#else
    unsetenv(kNativeVarBoundsEnvVar);
#endif
}

/// A LinearODE phase that exposes ODEPhaseBase's protected native
/// variable-bound machinery, so the recording and transcription steps can be
/// driven without depending on the process-wide declaration-site switch.
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
    int bounded = -1;
    for (int i = 0; i < phase->nlp_->x_lower_.size(); ++i) {
        if (phase->nlp_->x_lower_[i] != -kNativeVarBoundsInf ||
            phase->nlp_->x_upper_[i] != kNativeVarBoundsInf) {
            EXPECT_EQ(bounded, -1) << "more than one variable was bounded, at " << i;
            bounded = i;
        }
    }
    ASSERT_GE(bounded, kNativeVarBoundsPhaseVars);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[bounded], -2.0);
    EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[bounded], 4.0);
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

TEST(PhaseNativeVarBounds, RemovingAnInequalityByExplicitIndexIsRejectedWhileBoundsAreRecorded) {
    auto phase = make_native_var_bounds_phase();
    // A genuine inequality constraint, added the ordinary way.
    int handle = phase->add_lu_func_bound(PhaseRegionFlags::Front, native_var_bounds_scalar_fn(), 0,
                                          -1.0, 1.0, 1.0, 1.0, ScaleModes::AUTO);
    ASSERT_EQ(phase->num_inequalities(), 1u);
    EXPECT_NO_THROW(phase->remove_inequal_con(handle));

    handle = phase->add_lu_func_bound(PhaseRegionFlags::Front, native_var_bounds_scalar_fn(), 0,
                                      -1.0, 1.0, 1.0, 1.0, ScaleModes::AUTO);
    phase->record_bounds(PhaseRegionFlags::Front, 0, -1.0, 1.0);
    // An explicit index is not unambiguously an inequality-constraint index
    // once bound records exist.
    EXPECT_THROW(phase->remove_inequal_con(handle), std::invalid_argument);

    // -1 ("the most recently added") resolves within user_inequalities_
    // itself and is unambiguous regardless of any recorded bounds.
    EXPECT_NO_THROW(phase->remove_inequal_con(-1));
    EXPECT_EQ(phase->num_inequalities(), 0u);
}

TEST(PhaseNativeVarBounds, InequalityScaleAccessorIsRejectedWhileBoundsAreRecorded) {
    auto phase = make_native_var_bounds_phase();
    phase->add_lu_func_bound(PhaseRegionFlags::Front, native_var_bounds_scalar_fn(), 0, -1.0, 1.0,
                             1.0, 1.0, ScaleModes::AUTO);
    EXPECT_NO_THROW(phase->return_inequal_con_scales(0));

    phase->record_bounds(PhaseRegionFlags::Front, 0, -1.0, 1.0);
    EXPECT_THROW(phase->return_inequal_con_scales(0), std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// Bound kinds that must keep lowering to inequality constraints under BOTH
// switch states: function bounds, norm bounds, and delta-variable bounds never
// consult the switch.
///////////////////////////////////////////////////////////////////////////////

TEST(PhaseNativeVarBounds, FuncNormAndDeltaBoundsStillLowerToInequalities) {
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
// Declaration-site routing under TYCHO_DEV_NATIVE_BOUNDS. Both fixtures below
// set/restore the environment variable per test (see the file header), so
// every test here always runs — none of them skip.
///////////////////////////////////////////////////////////////////////////////

// Defined at namespace scope (not in the anonymous namespace above) so the
// TEST_F-generated classes do not derive from an internal-linkage base. The
// names are unique across tests/cpp/, which is what the unity build requires.
class PhaseNativeVarBoundsSwitchOn : public ::testing::Test {
  protected:
    void SetUp() override {
        const char *prior = std::getenv(kNativeVarBoundsEnvVar);
        had_prior_ = prior != nullptr;
        if (had_prior_) {
            prior_ = prior;
        }
        native_var_bounds_set_env("1");
    }

    void TearDown() override {
        if (had_prior_) {
            native_var_bounds_set_env(prior_.c_str());
        } else {
            native_var_bounds_clear_env();
        }
    }

    bool had_prior_ = false;
    std::string prior_;
};

// The mirror-image fixture: forces the switch OFF for its duration, so the
// old lowering behavior can be pinned regardless of what the environment
// looked like before this test ran.
class PhaseNativeVarBoundsSwitchOff : public ::testing::Test {
  protected:
    void SetUp() override {
        const char *prior = std::getenv(kNativeVarBoundsEnvVar);
        had_prior_ = prior != nullptr;
        if (had_prior_) {
            prior_ = prior;
        }
        native_var_bounds_clear_env();
    }

    void TearDown() override {
        if (had_prior_) {
            native_var_bounds_set_env(prior_.c_str());
        } else {
            native_var_bounds_clear_env();
        }
    }

    bool had_prior_ = false;
    std::string prior_;
};

TEST_F(PhaseNativeVarBoundsSwitchOn, LuVarBoundRecordsInsteadOfLowering) {
    auto phase = make_native_var_bounds_phase();
    int handle = phase->add_lu_var_bound(PhaseRegionFlags::Path, /*var=*/1, -1.0, 2.0, 1.0,
                                         ScaleModes::AUTO);

    EXPECT_EQ(handle, 0);
    EXPECT_EQ(phase->num_inequalities(), 0u);
    ASSERT_EQ(phase->num_var_bound_records(), 1u);
    EXPECT_EQ(phase->record_region(0), PhaseRegionFlags::Path);
    EXPECT_DOUBLE_EQ(phase->record_lower(0), -1.0);
    EXPECT_DOUBLE_EQ(phase->record_upper(0), 2.0);

    phase->transcribe();
    for (int gidx : {1, 4, 7}) {
        EXPECT_DOUBLE_EQ(phase->nlp_->x_lower_[gidx], -1.0) << "index " << gidx;
        EXPECT_DOUBLE_EQ(phase->nlp_->x_upper_[gidx], 2.0) << "index " << gidx;
    }
}

TEST_F(PhaseNativeVarBoundsSwitchOn, LowerAndUpperVarBoundsLeaveTheOtherSideOpen) {
    auto phase = make_native_var_bounds_phase();
    phase->add_lower_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0, 1.0, ScaleModes::AUTO);
    phase->add_upper_var_bound(PhaseRegionFlags::Back, /*var=*/0, 7.0, 1.0, ScaleModes::AUTO);

    ASSERT_EQ(phase->num_var_bound_records(), 2u);
    EXPECT_EQ(phase->num_inequalities(), 0u);
    EXPECT_DOUBLE_EQ(phase->record_lower(0), -1.0);
    EXPECT_EQ(phase->record_upper(0), kNativeVarBoundsInf);
    EXPECT_EQ(phase->record_lower(1), -kNativeVarBoundsInf);
    EXPECT_DOUBLE_EQ(phase->record_upper(1), 7.0);
}

TEST_F(PhaseNativeVarBoundsSwitchOn, ValidationMatchesTheInequalityPath) {
    auto phase = make_native_var_bounds_phase();

    // Lower > upper is rejected exactly as the lowering path rejects it.
    EXPECT_THROW(phase->add_lu_var_bound(PhaseRegionFlags::Front, /*var=*/0, 2.0, 1.0, 1.0, 1.0,
                                         ScaleModes::AUTO),
                 std::invalid_argument);
    // Non-positive bound scales still throw, even though the recorded bound
    // does not use their values.
    EXPECT_THROW(phase->add_lu_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0, 1.0, 0.0, 1.0,
                                         ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_THROW(phase->add_lu_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0, 1.0, 1.0, -1.0,
                                         ScaleModes::AUTO),
                 std::invalid_argument);
    EXPECT_THROW(
        phase->add_lower_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0, 0.0, ScaleModes::AUTO),
        std::invalid_argument);
    EXPECT_THROW(
        phase->add_upper_var_bound(PhaseRegionFlags::Front, /*var=*/0, 1.0, 0.0, ScaleModes::AUTO),
        std::invalid_argument);

    EXPECT_EQ(phase->num_var_bound_records(), 0u);
    EXPECT_EQ(phase->num_inequalities(), 0u);
}

TEST_F(PhaseNativeVarBoundsSwitchOn, FuncAndDeltaBoundsStillLowerToInequalities) {
    auto phase = make_native_var_bounds_phase();
    phase->add_lu_func_bound(PhaseRegionFlags::Front, native_var_bounds_scalar_fn(), 0, -1.0, 1.0,
                             1.0, 1.0, ScaleModes::AUTO);
    phase->add_lower_delta_var_bound(/*var=*/2, 0.1, 1.0, ScaleModes::AUTO);

    Eigen::VectorXi norm_vars(2);
    norm_vars << 0, 1;
    phase->add_lu_norm_bound(PhaseRegionFlags::Front, norm_vars, 0.0, 5.0, 1.0, 1.0,
                             ScaleModes::AUTO);

    EXPECT_EQ(phase->num_inequalities(), 3u);
    EXPECT_EQ(phase->num_var_bound_records(), 0u);
}

TEST_F(PhaseNativeVarBoundsSwitchOff, AddLuVarBoundStillLowersToInequality) {
    auto phase = make_native_var_bounds_phase();
    int handle = phase->add_lu_var_bound(PhaseRegionFlags::Front, /*var=*/0, -1.0, 1.0, 1.0,
                                         ScaleModes::AUTO);

    EXPECT_EQ(handle, 0);
    EXPECT_EQ(phase->num_inequalities(), 1u);
    EXPECT_EQ(phase->num_var_bound_records(), 0u);
}
