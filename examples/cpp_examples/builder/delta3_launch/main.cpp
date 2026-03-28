///////////////////////////////////////////////////////////////////////////////
// Delta III Launch Vehicle — C++ example (Builder API)
//
// Same problem as the static DSL version.  Demonstrates:
//   - var_group() for named groups (R, V, u)
//   - from() with ODEArguments for vector ODE expressions
//   - OCP wrapper for named-variable link constraints (no base_ptr())
//   - Named variables for all constraint/objective calls
//
// State  : [Rx, Ry, Rz, Vx, Vy, Vz, m]   (7)
// Control: [ux, uy, uz]                     (3)
//
// Phase vector layout: [R(3), V(3), m, t, u(3)]
//                       0-2   3-5   6  7  8-10
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;

///////////////////////////////////////////////////////////////////////////////
// Physical constants (non-dimensionalised)
///////////////////////////////////////////////////////////////////////////////

static constexpr double Lstar = 6378145.0;
static constexpr double Tstar = 961.0;
static constexpr double Mstar = 301454.0;

static const double Astar = Lstar / (Tstar * Tstar);
static const double Vstar = Lstar / Tstar;
static const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
static const double Mustar = (Lstar * Lstar * Lstar) / (Tstar * Tstar);
static const double Fstar = Astar * Mstar;

static const double mu = 3.986012e14 / Mustar;
static const double Re = 6378145.0 / Lstar;
static const double We = 7.29211585e-5 * Tstar;

static const double RhoAir = 1.225 / Rhostar;
static const double h_scale = 7200.0 / Lstar;
static const double g0 = 9.80665;
static const double g = g0 / Astar;

static constexpr double CD = 0.5;
static const double S = 4.0 * M_PI / (Lstar * Lstar);

// Engine parameters
static const double TS = 628500.0 / Fstar;
static const double T1 = 1083100.0 / Fstar;
static const double T2 = 110094.0 / Fstar;

static const double IS = 283.33364 / Tstar;
static const double I1 = 301.68 / Tstar;
static const double I2 = 467.21 / Tstar;

static const double tS = 75.2 / Tstar;
static const double t1 = 261.0 / Tstar;
static const double t2 = 700.0 / Tstar;

// Stage masses
static const double TMS = 19290.0 / Mstar;
static const double TM1 = 104380.0 / Mstar;
static const double TM2 = 19300.0 / Mstar;
static const double TMPay = 4164.0 / Mstar;

static const double PMS = 17010.0 / Mstar;
static const double PM1 = 95550.0 / Mstar;
static const double PM2 = 16820.0 / Mstar;

static const double SMS = TMS - PMS;
static const double SM1 = TM1 - PM1;

// Phase thrust and mass flow rates
static const double T_phase1 = 6.0 * TS + T1;
static const double T_phase2 = 3.0 * TS + T1;
static const double T_phase3 = T1;
static const double T_phase4 = T2;

static const double mdot_phase1 = (6.0 * TS / IS + T1 / I1) / g;
static const double mdot_phase2 = (3.0 * TS / IS + T1 / I1) / g;
static const double mdot_phase3 = T1 / (g * I1);
static const double mdot_phase4 = T2 / (g * I2);

// Phase times
static const double tf_phase1 = tS;
static const double tf_phase2 = 2.0 * tS;
static const double tf_phase3 = t1;
static const double tf_phase4 = t1 + t2;

// Phase masses
static const double m0_phase1 = 9.0 * TMS + TM1 + TM2 + TMPay;
static const double mf_phase1 = m0_phase1 - 6.0 * PMS - (tS / t1) * PM1;

static const double m0_phase2 = mf_phase1 - 6.0 * SMS;
static const double mf_phase2 = m0_phase2 - 3.0 * PMS - (tS / t1) * PM1;

static const double m0_phase3 = mf_phase2 - 3.0 * SMS;
static const double mf_phase3 = m0_phase3 - (1.0 - 2.0 * tS / t1) * PM1;

static const double m0_phase4 = mf_phase3 - SM1;
static const double mf_phase4 = m0_phase4 - PM2;

///////////////////////////////////////////////////////////////////////////////
// RocketODE factory — parameterised by (T, mdot)
///////////////////////////////////////////////////////////////////////////////

RuntimeODE make_rocket_ode(double T, double mdot) {
    // Dynamic args with template segment accessors: dynamic shell, static core.
    // head<3>() / segment<3>() produce ORC=3, enabling cross/normalized_power.
    auto args = ODEArguments(7, 3, 0);
    auto R = args.head<3>();
    auto V = args.segment<3>(3);
    auto m = args.coeff(6);
    auto u = args.tail<3>().normalized();

    auto h = R.norm() - Re;

    // Exponential atmosphere model: density decay factor exp(-h / h_scale)
    auto exp_neg_h = exp(h * (-1.0 / h_scale));

    // Relative velocity: Vr = V + R x [0, 0, We]
    Eigen::Vector3d omega_val(0.0, 0.0, We);
    auto omega = Constant<-1, 3>(11, omega_val);
    auto Vr = V + R.cross(omega);

    // Drag
    const double drag_rho = -0.5 * CD * S * RhoAir;
    auto D = (drag_rho * exp_neg_h) * (Vr * Vr.norm());

    auto Rdot = V;
    auto Vdot = (-mu) * R.normalized_power<3>() + (T * u + D) / m;

    // Constant mass flow rate
    Eigen::Matrix<double, 1, 1> neg_mdot_val;
    neg_mdot_val[0] = -mdot;
    auto neg_mdot_vf = Constant<-1, 1>(11, neg_mdot_val);

    auto ode_expr = StackedOutputs{Rdot, Vdot, neg_mdot_vf};

    return ODEBuilder(7, 3)
        .from(ode_expr)
        .var_names({{"m", 6}, {"t", 7}})
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_group("u", 8, 3)
        .build();
}

///////////////////////////////////////////////////////////////////////////////
// TargetOrbit — equality constraint on classical orbital elements
///////////////////////////////////////////////////////////////////////////////

auto MakeTargetOrbit(double at, double et, double it, double Ot, double Wt) {
    auto args = Arguments<6>();
    auto R = args.head<3>();
    auto V = args.tail<3>();

    auto r = R.norm();

    auto hvec = R.cross(V);

    Eigen::Vector3d z_hat(0.0, 0.0, 1.0);
    auto z_const = Constant<6, 3>(6, z_hat);
    auto nvec = z_const.cross(hvec);

    auto eps = V.squared_norm() * 0.5 - mu / r;
    auto a = -0.5 * mu / eps;

    auto evec = V.cross(hvec) / mu - R.normalized();
    auto e = evec.norm();

    auto inc = acos(hvec.normalized().coeff<2>());

    auto O_raw = acos(nvec.normalized().coeff<0>());
    auto O = IfElseFunction(nvec.coeff<1>() > 0.0, O_raw, 2.0 * M_PI - O_raw);

    auto W_raw = acos(nvec.normalized().dot(evec.normalized()));
    auto W = IfElseFunction(evec.coeff<2>() > 0.0, W_raw, 2.0 * M_PI - W_raw);

    return StackedOutputs{a - at, e - et, inc - it, O - Ot, W - Wt};
}

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    // Target orbital elements
    const double at = 24361140.0 / Lstar;
    const double et = 0.7308;
    const double Ot = 269.8 * M_PI / 180.0;
    const double Wt = 130.5 * M_PI / 180.0;
    const double istart = 28.5 * M_PI / 180.0;

    // Initial state: launch site on rotating Earth
    Eigen::VectorXd y0(6);
    y0.setZero();
    y0[0] = std::cos(istart) * Re;
    y0[2] = std::sin(istart) * Re;

    Eigen::Vector3d R0 = y0.head<3>();
    Eigen::Vector3d omega_earth(0.0, 0.0, We);
    Eigen::Vector3d V0 = -R0.cross(omega_earth);
    V0[0] += 0.00001 / Vstar;
    y0[3] = V0[0];
    y0[4] = V0[1];
    y0[5] = V0[2];

    // Target state from classical orbital elements
    Vector6<double> oelems;
    oelems << at, et, istart, Ot, Wt, -0.05;
    auto yf_vec = classic_to_cartesian(oelems, mu);
    Eigen::VectorXd yf = yf_vec;

    // Generate initial guesses for each phase
    const int nPts = 1000;
    std::vector<Eigen::VectorXd> IG1, IG2, IG3, IG4;

    for (int i = 0; i < nPts; ++i) {
        const double t = tf_phase4 * static_cast<double>(i) / (nPts - 1);
        Eigen::VectorXd X(11);
        X.setZero();

        Eigen::VectorXd rv = y0 + (yf - y0) * (t / tf_phase4);
        X.head<6>() = rv;
        X[7] = t;
        X[9] = 1.0; // u_y = 1

        if (t < tf_phase1) {
            X[6] = m0_phase1 + (mf_phase1 - m0_phase1) * (t / tf_phase1);
            IG1.push_back(X);
        } else if (t < tf_phase2) {
            X[6] = m0_phase2 +
                   (mf_phase2 - m0_phase2) * ((t - tf_phase1) / (tf_phase2 - tf_phase1));
            IG2.push_back(X);
        } else if (t < tf_phase3) {
            X[6] = m0_phase3 +
                   (mf_phase3 - m0_phase3) * ((t - tf_phase2) / (tf_phase3 - tf_phase2));
            IG3.push_back(X);
        } else {
            X[6] = m0_phase4 +
                   (mf_phase4 - m0_phase4) * ((t - tf_phase3) / (tf_phase4 - tf_phase3));
            IG4.push_back(X);
        }
    }

    // ── Build ODEs for each phase ───────────────────────────────────────
    auto ode1 = make_rocket_ode(T_phase1, mdot_phase1);
    auto ode2 = make_rocket_ode(T_phase2, mdot_phase2);
    auto ode3 = make_rocket_ode(T_phase3, mdot_phase3);
    auto ode4 = make_rocket_ode(T_phase4, mdot_phase4);

    constexpr int nSegs = 40;
    const auto tmode = TranscriptionModes::LGL3;

    ///////////////////////////////////////////////////////////////////////////
    // Phase 1: 6 SRBs + core
    ///////////////////////////////////////////////////////////////////////////
    auto phase1 = ode1.phase(tmode, IG1, nSegs);
    phase1.set_control_mode(ControlModes::HighestOrderSpline);

    phase1.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.5, 1.5);

    // Front BC: full state [R, V, m, t]
    Eigen::VectorXd front_val(8);
    front_val << y0[0], y0[1], y0[2], y0[3], y0[4], y0[5], m0_phase1, 0.0;
    phase1.add_boundary_value(PhaseRegionFlags::Front, {"R", "V", "m", "t"}, front_val);

    // Altitude constraint
    phase1.add_lower_norm_bound(PhaseRegionFlags::Path, {"R"}, Re * 0.999999);

    // Back BC: final time
    phase1.add_boundary_value(PhaseRegionFlags::Back, "t", tf_phase1);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 2: 3 SRBs + core
    ///////////////////////////////////////////////////////////////////////////
    auto phase2 = ode2.phase(tmode, IG2, nSegs);
    phase2.set_control_mode(ControlModes::HighestOrderSpline);

    phase2.add_lower_norm_bound(PhaseRegionFlags::Path, {"R"}, Re);
    phase2.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.5, 1.5);

    phase2.add_boundary_value(PhaseRegionFlags::Front, "m", m0_phase2);
    phase2.add_boundary_value(PhaseRegionFlags::Back, "t", tf_phase2);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 3: core only
    ///////////////////////////////////////////////////////////////////////////
    auto phase3 = ode3.phase(tmode, IG3, nSegs);
    phase3.set_control_mode(ControlModes::HighestOrderSpline);

    phase3.add_lower_norm_bound(PhaseRegionFlags::Path, {"R"}, Re);
    phase3.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.5, 1.5);

    phase3.add_boundary_value(PhaseRegionFlags::Front, "m", m0_phase3);
    phase3.add_boundary_value(PhaseRegionFlags::Back, "t", tf_phase3);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 4: upper stage
    ///////////////////////////////////////////////////////////////////////////
    auto phase4 = ode4.phase(tmode, IG4, nSegs);
    phase4.set_control_mode(ControlModes::HighestOrderSpline);

    phase4.add_lower_norm_bound(PhaseRegionFlags::Path, {"R"}, Re);
    phase4.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.5, 1.5);

    phase4.add_boundary_value(PhaseRegionFlags::Front, "m", m0_phase4);

    // Upper bound on final time
    phase4.add_upper_var_bound(PhaseRegionFlags::Back, "t", tf_phase4);

    // Terminal orbital element constraint
    auto target_orbit_fn = MakeTargetOrbit(at, et, istart, Ot, Wt);
    phase4.add_equal_con(PhaseRegionFlags::Back, GenericFunction<-1, -1>(target_orbit_fn),
                       {"R", "V"});

    // Maximise final mass (minimise -m)
    phase4.add_value_objective(PhaseRegionFlags::Back, "m", -1.0);

    ///////////////////////////////////////////////////////////////////////////
    // Optimal Control Problem — link phases via OCP wrapper
    ///////////////////////////////////////////////////////////////////////////
    OCP ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);
    ocp.add_phase(phase3);
    ocp.add_phase(phase4);

    // Continuity in R, V, t, u across consecutive phase pairs (1->2, 2->3, 3->4);
    // mass is discontinuous at staging events.
    ocp.add_forward_link_equal_con(phase1, phase4, {"R", "V", "t", "u"});

    ocp.optimizer().set_OptLSMode("L1");
    ocp.optimizer().set_SoeLSMode("L1");
    ocp.optimizer().set_MaxLSIters(2);
    ocp.optimizer().set_PrintLevel(1);

    ///////////////////////////////////////////////////////////////////////////
    // Solve
    ///////////////////////////////////////////////////////////////////////////
    std::cout << "Solving Delta III launch vehicle problem ...\n" << std::flush;

    const auto status = ocp.solve_optimize();

    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "Delta3Launch (builder): FAILED (status " << static_cast<int>(status) << ")\n";
        return 1;
    }

    auto traj4 = phase4.return_traj();
    if (traj4.empty()) {
        std::cerr << "Delta3Launch (builder): solver converged but trajectory is empty\n";
        return 1;
    }

    const double final_mass_kg = traj4.back()[6] * Mstar;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Delta3Launch (builder): Final mass = " << final_mass_kg << " kg\n";

    return 0;
}
