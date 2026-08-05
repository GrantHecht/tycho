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

/// @brief Build the standard Brachistochrone phase.
/// @param n_pts      Number of initial-guess trajectory points.
/// @param n_defects  Number of collocation segments.
/// @param mode       Transcription scheme.
/// @param control_bound_as_inequality  Carry the control box as a pair of
///        inequality-constraint rows (``-0.1 - u <= 0`` and ``u - 2 <= 0``)
///        instead of as a variable bound. The two formulations describe the
///        same feasible set through different NLP structures, which is what
///        lets a test compare them side by side; every other caller wants the
///        default.
inline std::shared_ptr<ODEPhase<BrachODE>>
make_brach_phase(int n_pts = 100, int n_defects = 32,
                 TranscriptionModes mode = TranscriptionModes::LGL3,
                 bool control_bound_as_inequality = false) {
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
    constexpr double u_lower = -0.1, u_upper = 2.0;
    if (control_bound_as_inequality) {
        auto u = vf::Arguments<1>();
        Eigen::VectorXi u_idx(1);
        u_idx << 4;
        phase->add_inequal_con(PhaseRegionFlags::Path,
                               vf::StackedOutputs{u_lower - u, u - u_upper}, u_idx,
                               ScaleModes::AUTO);
    } else {
        phase->add_lu_var_bound(PhaseRegionFlags::Path, 4, u_lower, u_upper);
    }

    // Objective
    phase->add_delta_time_objective(1.0, ScaleModes::AUTO);

    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// Helper: non-unit packing units for a Brachistochrone phase, for OC review
// §1.12 (set_units() must trigger reset_transcription()) tests.
///////////////////////////////////////////////////////////////////////////////

/// @brief Non-unit `[x, y, v, t, theta]` scaling units matching the
/// `xtu_p_vars()` layout of `make_brach_phase()`'s `BrachODE` (x_vars=3,
/// u_vars=1, p_vars=0). Only consumed by the transcription when auto-scaling
/// is enabled (`set_auto_scaling(true)`); deliberately far from the all-ones
/// default so re-transcribing under these units changes the NLP the solver
/// sees. Values are rough characteristic scales of the Brachistochrone
/// solution (x, y ~ 10 m, v ~ 5 m/s, t ~ 2 s, theta ~ 0.5 rad).
inline Eigen::VectorXd brach_nonunit_units() {
    Eigen::VectorXd units(5);
    units << 10.0, 10.0, 5.0, 2.0, 0.5;
    return units;
}

///////////////////////////////////////////////////////////////////////////////
// Helper: build a phase with exactly-representable linear dynamics, for OC
// review §1.7 / §3.4 mesh-refinement robustness tests.
//
// State: [x, v, t], x_vars=2, u_vars=0, p_vars=0. ODE: dx/dt = v, dv/dt = 0
// (constant velocity), so the analytic solution x(t) = x0 + v0*t is a linear
// polynomial in t -- exactly representable by any LGL collocation scheme.
// All trajectory values (x, v, t on a power-of-two grid) and the LGL3
// cardinal power weights (lgl_coeffs.h:80-85, small exact-binary rationals)
// are exact in binary floating point, so the de Boor derivative estimate
// cancels to bit-exact zero: the mesh error/density is identically 0.0,
// reproducing the zero-density mesh that pre-fix produced 0/0 (NaN) bins
// (§3.4).
//
// NOTE: nsegs >= 2 is required at construction -- the validated set_traj
// overload (ode_phase_base.cpp:499-501) throws for DPB.sum() < 2. The
// num_blocks == 1 state that under-ran the de Boor stencil (§1.7) is only
// reachable through mesh refinement: update_mesh() -> refine_traj_manual()
// has no segment-count guard and the min_segments_ clamp permits n == 1.
// See MeshRobustness.RefinementToSingleSegmentNoCrashConverges.
///////////////////////////////////////////////////////////////////////////////

struct LinearODE_Impl : ODESize<2, 0, 0> {
    static auto Definition() {
        auto args = Arguments<3>(); // [x, v, t]
        auto v = args.coeff<1>();
        auto xdot = v;
        auto vdot = 0.0 * v; // constant velocity
        return StackedOutputs{xdot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(LinearODE, LinearODE_Impl);

/// @brief Build an ODEPhase<LinearODE> with an exact linear trajectory guess:
/// x0=0, v0=1, t in [0, 1].
/// @param nsegs  Number of defect intervals (segments) to construct with;
///               must be >= 2 (set_traj throws below that).
inline std::shared_ptr<ODEPhase<LinearODE>> make_linear_phase(int nsegs = 2) {
    constexpr double x0 = 0.0, v0 = 1.0, t0 = 0.0, tf = 1.0;
    constexpr int n_pts = 5;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        double t = t0 + (tf - t0) * s;
        Eigen::VectorXd pt(3);
        pt[0] = x0 + v0 * (t - t0); // x(t)
        pt[1] = v0;                 // v(t) (constant)
        pt[2] = t;                  // t
        traj.push_back(pt);
    }

    LinearODE ode;
    auto phase = std::make_shared<ODEPhase<LinearODE>>(ode, TranscriptionModes::LGL3, traj, nsegs);

    // Front boundary: fix x0, v0, t0.
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(3, 0, 2);
    Eigen::VectorXd front_val(3);
    front_val << x0, v0, t0;
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    // Back boundary: fix only tf, leaving xf/vf to be pinned down by the
    // (exact) defect equations alone.
    Eigen::VectorXi back_idx(1);
    back_idx << 2;
    Eigen::VectorXd back_val(1);
    back_val << tf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

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

/// Build a state+control LGLInterpTable analogous to @ref make_exact_lgl_table
/// (short analytic trajectory loaded via load_exact_data(), so delta_t_ stays
/// 0.0), but with @p u_vars control columns appended after the time column:
/// layout `[x_0..x_{x_vars-1}, t, u_0..u_{u_vars-1}]` (axis_ == x_vars). Used
/// by OC review §1.6: pairing a fixed compile-time-OR InterpFunction<OR>
/// (whose scratch is sized for exactly OR+1 columns) with a CONTROL-bearing
/// table -- controls push xtu_vars_ above OR+1 whenever u_vars is large enough.
inline std::shared_ptr<LGLInterpTable> make_exact_lgl_table_with_controls(TranscriptionModes method,
                                                                          int x_vars, int u_vars) {
    // 6 == (num_nodes - 1) is divisible by (block_size_ - 1) for LGL3/5/7 (1, 2, 3).
    constexpr int num_nodes = 7;
    constexpr double t_final = 1.0;

    auto tab = std::make_shared<LGLInterpTable>(x_vars, u_vars, method);

    std::vector<Eigen::VectorXd> xtudat;
    std::vector<Eigen::VectorXd> xdotdat;
    xtudat.reserve(num_nodes);
    xdotdat.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        double t = t_final * static_cast<double>(i) / (num_nodes - 1);
        Eigen::VectorXd node(x_vars + u_vars + 1); // [x..., t, u...] -- axis_ == x_vars
        for (int j = 0; j < x_vars; ++j)
            node[j] = std::sin(t * (j + 1));
        node[x_vars] = t;
        for (int j = 0; j < u_vars; ++j)
            node[x_vars + 1 + j] = std::cos(t * (j + 1));
        xtudat.push_back(node);

        Eigen::VectorXd deriv(x_vars);
        for (int j = 0; j < x_vars; ++j)
            deriv[j] = (j + 1) * std::cos(t * (j + 1));
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

///////////////////////////////////////////////////////////////////////////////
// Helpers: LGLInterpTable::load_even_data() rejection paths, for OC review
// §3.2 (LGLInterpTable robustness -- throw not exit(1), validate inputs).
//
// All helpers build a no-ODE table (x_vars=1, u_vars=0, so xtu_vars_ == 2,
// axis_ == 1) and call load_even_data() directly with deliberately malformed
// input, exercising the throwing path pre-fix reached via std::cout + exit(1).
///////////////////////////////////////////////////////////////////////////////

/// @brief load_even_data() with nodes one column short of xtu_vars_ (2).
inline void load_even_wrong_dim() {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL3);
    std::vector<Eigen::VectorXd> xtudat;
    for (int i = 0; i < 3; ++i) {
        Eigen::VectorXd node(1); // wrong: table expects xtu_vars_ == 2 ([x, t])
        node << double(i);
        xtudat.push_back(node);
    }
    tab.load_even_data(xtudat);
}

/// @brief load_even_data() with two nodes sharing the same time coordinate.
inline void load_even_duplicate_times() {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL3);
    Eigen::VectorXd n0(2), n1(2), n2(2);
    n0 << 0.0, 0.0;
    n1 << 1.0, 0.0; // duplicate time (0.0) vs n0
    n2 << 2.0, 1.0;
    std::vector<Eigen::VectorXd> xtudat{n0, n1, n2};
    tab.load_even_data(xtudat);
}

/// @brief load_even_data() with a node count incompatible with the table's block
/// size: LGL5 has block_size_ == 3 (block_size_ - 1 == 2), so num_states_ - 1 must
/// be a multiple of 2; 4 nodes gives num_states_ - 1 == 3, which is not.
inline void load_even_missized_blocks() {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL5);
    std::vector<Eigen::VectorXd> xtudat;
    constexpr int num_nodes = 4;
    for (int i = 0; i < num_nodes; ++i) {
        Eigen::VectorXd node(2);
        node << double(i), double(i);
        xtudat.push_back(node);
    }
    tab.load_even_data(xtudat);
}

/// @brief load_even_data() with an empty node vector.
inline void load_even_empty() {
    LGLInterpTable tab(/*xv=*/1, /*uv=*/0, TranscriptionModes::LGL3);
    std::vector<Eigen::VectorXd> xtudat;
    tab.load_even_data(xtudat);
}

///////////////////////////////////////////////////////////////////////////////
// Helper: build a phase with a hand-set, large-magnitude control trajectory
// for calc_switches() unit tests (OC review §1.4).
//
// calc_switches() only reads the t_var()/xt_vars()/u_vars() columns of
// active_traj_ (see ode_phase_base.cpp); it doesn't need a solved, physically
// consistent trajectory. BrachSwitchTestPhase exposes a setter that writes
// active_traj_ directly -- it's `protected` on ODEPhaseBase, so a derived
// class may write it -- bypassing the mesh-resampling machinery in
// set_traj() so the exact node values driving calc_switches() are known
// precisely (no interpolation/resampling to reason about).
///////////////////////////////////////////////////////////////////////////////

struct BrachSwitchTestPhase : ODEPhase<BrachODE> {
    using ODEPhase<BrachODE>::ODEPhase;

    /// @brief Install a hand-crafted discretized trajectory directly, bypassing
    /// set_traj()'s mesh-resampling machinery.
    void set_active_traj_for_test(const std::vector<Eigen::VectorXd> &traj) {
        this->active_traj_ = traj;
    }
};

/// Normalized switch time produced by make_bangbang_phase(): the midpoint
/// between its t=0.25 and t=0.5 nodes, where the large control jump sits.
inline constexpr double kBangBangSwitchTime = 0.375;

/// @brief Build a BrachSwitchTestPhase with a 5-node, large-magnitude
/// ("bang-bang"-like) theta-control trajectory: one true switch (a jump of
/// 1000 raw units at t=[0.25, 0.5], scaled by @p u_scale) plus several
/// smaller-but-still-large raw steps (500, 400, 100 * u_scale/1000 units)
/// that must NOT be flagged as switches.
///
/// Reproduces OC review §1.4: pre-fix, calc_switches()'s `unddiff` reused the
/// raw (un-normalized) diff instead of the normalized `und` matrix, so the
/// "relative" tolerance check degenerated into another absolute-scale check
/// once the control isn't O(1) -- every step below gets (mis)flagged as a
/// switch, since all four raw diffs (500, 1000, 400, 100) exceed
/// rel_switch_tol_ (0.3) trivially at this scale. Post-fix, only the true
/// jump (normalized diff ~0.4997 > 0.3) is flagged; the others normalize to
/// ~0.2499, ~0.1999, ~0.0500, all below the 0.3 relative threshold.
inline std::shared_ptr<BrachSwitchTestPhase> make_bangbang_phase(double u_scale = 1000.0) {
    constexpr double g = 9.81;
    BrachODE ode(g);

    // [x, y, v, t, theta]; only t (idx 3) and theta (idx 4) matter to calc_switches().
    constexpr double thetas[5] = {-1.0, -0.5, 0.5, 0.9, 1.0};
    constexpr double times[5] = {0.0, 0.25, 0.5, 0.75, 1.0};

    std::vector<Eigen::VectorXd> traj(5);
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXd pt = Eigen::VectorXd::Zero(5);
        pt[3] = times[i];
        pt[4] = thetas[i] * u_scale;
        traj[i] = pt;
    }

    auto phase = std::make_shared<BrachSwitchTestPhase>(ode, TranscriptionModes::LGL3);
    phase->set_active_traj_for_test(traj);
    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// Helpers: OptimalControlProblemBase phase-lifetime tests (OC review §1.9 —
// remove_phase link-index remap/guard + bounds-checked phase()/remove_phase()).
///////////////////////////////////////////////////////////////////////////////

/// @brief Build an OptimalControlProblemBase with three independent
/// LinearODE phases (indices 0, 1, 2), for remove_phase() tests.
inline OptimalControlProblemBase make_three_phase_ocp() {
    OptimalControlProblemBase ocp;
    ocp.add_phase(make_linear_phase());
    ocp.add_phase(make_linear_phase());
    ocp.add_phase(make_linear_phase());
    return ocp;
}

/// @brief Add a direct link-equality constraint on the "x" state (index 0)
/// linking phase @p a's Back region to phase @p b's Front region.
/// @return The index assigned to the constraint.
inline int add_forward_link(OptimalControlProblemBase &ocp, int a, int b) {
    Eigen::VectorXi vars(1);
    vars << 0;
    return ocp.add_direct_link_equal_con(a, PhaseRegionFlags::Back, vars, b,
                                         PhaseRegionFlags::Front, vars, ScaleModes::AUTO);
}

/// @brief The phase indices (PhasePack order) of the first link-equality
/// constraint in @p ocp, as a plain `vector<int>` for `EXPECT_EQ` comparisons.
inline std::vector<int> first_link_phases(const OptimalControlProblemBase &ocp) {
    const auto &ptl = ocp.link_equalities_.begin()->second.phases_to_link_.front();
    return std::vector<int>(ptl.data(), ptl.data() + ptl.size());
}

} // namespace TychoTest
