///////////////////////////////////////////////////////////////////////////////
// Delta III Launch Vehicle — C++ example (static DSL)
//
// Maximum payload-to-orbit for a multi-stage rocket.  Four phases model the
// staged burn sequence (6 SRBs + core, 3 SRBs + core, core only, upper stage).
// The control vector u(3) is the thrust direction; it is normalised inside the
// ODE so no unit-norm path constraint is needed.
//
// State  : [Rx, Ry, Rz, Vx, Vy, Vz, m]   (7)
// Control: [ux, uy, uz]                     (3)
//
// Phase vector layout: [R(3), V(3), m, t, u(3)]
//                       0-2   3-5   6  7  8-10
//
// Corresponds to the Python example in examples/Delta3Launch.py.
//
// === PAIN POINTS (for Phase 7 static DSL improvements) ===
// 1. No composable ODE: each thrust/mdot combination requires a separate
//    ODE type in the static DSL.  Python just passes (T, mdot) at runtime.
//    Workaround: BUILD_ODE_FROM_EXPRESSION with (double, double) works here
//    because both parameters are the same type.  Heterogeneous parameter packs
//    would need a struct wrapper.
// 2. Constant vectors in expressions: R.cross([0,0,We]) requires explicit
//    Constant<IR,3> construction.  Python accepts numpy arrays directly.
// 3. TargetOrbit constraint function requires manual construction of
//    orbital elements from VF primitives (acos, cross, ifelse, normalized).
//    Verbose but functional.
// 4. Manual index arrays for every addBoundaryValue / addEqualCon call.
// 5. Arguments<11> is expensive to compile (~heavy template instantiation).
// 6. No ODEArguments equivalent — must track [x,t,u] layout manually.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Tycho;

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
// RocketODE — parameterised by (thrust, mass_flow_rate)
///////////////////////////////////////////////////////////////////////////////

struct RocketODE_Impl : ODESize<7, 3, 0> {
    static auto Definition(double T, double mdot) {
        auto args = Arguments<11>(); // [R(3), V(3), m, t, u(3)]
        auto R = args.head<3>();
        auto V = args.segment<3, 3>();
        auto m = args.coeff<6>();
        auto u = args.tail<3>().normalized();

        auto h = R.norm() - Re;

        // Atmospheric density: rho = RhoAir * exp(-h / h_scale)
        // NOTE: Combine RhoAir with drag coefficient to avoid double-scaling
        //       the expression (double * Scaled<...> has a bug in OperatorOverloads.h
        //       where it accesses func.Scaled_func instead of func.func).
        auto exp_neg_h = exp(h * (-1.0 / h_scale));

        // Relative velocity: Vr = V + R × [0, 0, We]
        Eigen::Vector3d omega_val(0.0, 0.0, We);
        auto omega = Constant<11, 3>(11, omega_val);
        auto Vr = V + R.cross(omega);

        // Drag: D = -0.5 * CD * S * rho * ||Vr|| * Vr
        // Combine all scalar constants into one to avoid double-Scaled wrapping
        const double drag_rho = -0.5 * CD * S * RhoAir;
        auto D = (drag_rho * exp_neg_h) * (Vr * Vr.norm());

        auto Rdot = V;
        auto Vdot = (-mu) * R.normalized_power<3>() + (T * u + D) / m;

        // Constant mass flow rate as a VF (must match IR=11)
        Eigen::Matrix<double, 1, 1> neg_mdot_val;
        neg_mdot_val[0] = -mdot;
        auto neg_mdot_vf = Constant<11, 1>(11, neg_mdot_val);

        return StackedOutputs{Rdot, Vdot, neg_mdot_vf};
    }
};
BUILD_ODE_FROM_EXPRESSION(RocketODE, RocketODE_Impl, double, double);

///////////////////////////////////////////////////////////////////////////////
// TargetOrbit — equality constraint on classical orbital elements
///////////////////////////////////////////////////////////////////////////////

auto MakeTargetOrbit(double at, double et, double it, double Ot, double Wt) {
    auto args = Arguments<6>();
    auto R = args.head<3>();
    auto V = args.tail<3>();

    auto r = R.norm();

    // Angular momentum: h = R × V
    auto hvec = R.cross(V);

    // Node vector: n = [0,0,1] × h
    Eigen::Vector3d z_hat(0.0, 0.0, 1.0);
    auto z_const = Constant<6, 3>(6, z_hat);
    auto nvec = z_const.cross(hvec);

    // Specific energy: eps = 0.5 * v^2 - mu/r
    auto eps = V.squared_norm() * 0.5 - mu / r;

    // Semi-major axis
    auto a = -0.5 * mu / eps;

    // Eccentricity vector: e_vec = (V × h)/mu - R_hat
    auto evec = V.cross(hvec) / mu - R.normalized();
    auto e = evec.norm();

    // Inclination
    auto inc = acos(hvec.normalized().coeff<2>());

    // RAAN
    auto O_raw = acos(nvec.normalized().coeff<0>());
    // Quadrant check: if nvec[1] > 0, O = O_raw; else O = 2*pi - O_raw
    auto O = IfElseFunction(nvec.coeff<1>() > 0.0, O_raw, 2.0 * M_PI - O_raw);

    // Argument of periapse
    auto W_raw = acos(nvec.normalized().dot(evec.normalized()));
    // Quadrant check: if evec[2] > 0, W = W_raw; else W = 2*pi - W_raw
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
    y0[1] = 0.0;
    y0[2] = std::sin(istart) * Re;

    // Velocity from Earth rotation: V = -R × omega
    Eigen::Vector3d R0 = y0.head<3>();
    Eigen::Vector3d omega_earth(0.0, 0.0, We);
    Eigen::Vector3d V0 = -R0.cross(omega_earth);
    V0[0] += 0.00001 / Vstar; // small perturbation to avoid zero Vr norm
    y0[3] = V0[0];
    y0[4] = V0[1];
    y0[5] = V0[2];

    // Target state from classical orbital elements
    Vector6<double> oelems;
    oelems << at, et, istart, Ot, Wt, -0.05; // last element = mean anomaly
    auto yf_vec = classic_to_cartesian(oelems, mu);
    Eigen::VectorXd yf = yf_vec;

    // Generate initial guesses for each phase
    const int nPts = 1000;
    std::vector<Eigen::VectorXd> IG1, IG2, IG3, IG4;

    for (int i = 0; i < nPts; ++i) {
        const double t = tf_phase4 * static_cast<double>(i) / (nPts - 1);
        Eigen::VectorXd X(11);
        X.setZero();

        // Linearly interpolate position/velocity
        Eigen::VectorXd rv = y0 + (yf - y0) * (t / tf_phase4);
        X.head<6>() = rv;
        X[7] = t;
        X[8] = 0.0;
        X[9] = 1.0;
        X[10] = 0.0;

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

    // ODEs for each phase
    RocketODE ode1(T_phase1, mdot_phase1);
    RocketODE ode2(T_phase2, mdot_phase2);
    RocketODE ode3(T_phase3, mdot_phase3);
    RocketODE ode4(T_phase4, mdot_phase4);

    constexpr int nSegs = 40;
    const auto tmode = TranscriptionModes::LGL3;

    // Helper for VectorXi construction
    auto make_idx = [](std::initializer_list<int> vals) {
        Eigen::VectorXi v(vals.size());
        int i = 0;
        for (auto val : vals) v[i++] = val;
        return v;
    };

    // Phase index arrays used repeatedly
    auto pos_idx = make_idx({0, 1, 2});
    auto ctrl_idx = make_idx({8, 9, 10});
    auto rv_idx = make_idx({0, 1, 2, 3, 4, 5});

    ///////////////////////////////////////////////////////////////////////////
    // Phase 1: 6 SRBs + core
    ///////////////////////////////////////////////////////////////////////////
    auto phase1 = std::make_shared<ODEPhase<RocketODE>>(ode1, tmode);
    phase1->setTraj(IG1, nSegs);
    phase1->setControlMode(ControlModes::HighestOrderSpline);

    phase1->addLUNormBound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0,
                           ScaleModes::AUTO);

    // Front BC: full state [R, V, m, t]
    auto front_idx_8 = make_idx({0, 1, 2, 3, 4, 5, 6, 7});
    Eigen::VectorXd front_val_8(8);
    front_val_8 << y0[0], y0[1], y0[2], y0[3], y0[4], y0[5], m0_phase1, 0.0;
    phase1->addBoundaryValue(PhaseRegionFlags::Front, front_idx_8, front_val_8,
                             ScaleModes::AUTO);

    // Altitude above (slightly relaxed) Earth radius
    phase1->addLowerNormBound(PhaseRegionFlags::Path, pos_idx, Re * 0.999999, 1.0,
                              ScaleModes::AUTO);

    // Back BC: final time
    Eigen::VectorXi t_idx(1);
    t_idx << 7;
    Eigen::VectorXd tf1_val(1);
    tf1_val << tf_phase1;
    phase1->addBoundaryValue(PhaseRegionFlags::Back, t_idx, tf1_val, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 2: 3 SRBs + core
    ///////////////////////////////////////////////////////////////////////////
    auto phase2 = std::make_shared<ODEPhase<RocketODE>>(ode2, tmode);
    phase2->setTraj(IG2, nSegs);
    phase2->setControlMode(ControlModes::HighestOrderSpline);

    phase2->addLowerNormBound(PhaseRegionFlags::Path, pos_idx, Re, 1.0, ScaleModes::AUTO);
    phase2->addLUNormBound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0,
                           ScaleModes::AUTO);

    Eigen::VectorXi m_idx(1);
    m_idx << 6;
    Eigen::VectorXd m0_p2_val(1);
    m0_p2_val << m0_phase2;
    phase2->addBoundaryValue(PhaseRegionFlags::Front, m_idx, m0_p2_val, ScaleModes::AUTO);

    Eigen::VectorXd tf2_val(1);
    tf2_val << tf_phase2;
    phase2->addBoundaryValue(PhaseRegionFlags::Back, t_idx, tf2_val, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 3: core only
    ///////////////////////////////////////////////////////////////////////////
    auto phase3 = std::make_shared<ODEPhase<RocketODE>>(ode3, tmode);
    phase3->setTraj(IG3, nSegs);
    phase3->setControlMode(ControlModes::HighestOrderSpline);

    phase3->addLowerNormBound(PhaseRegionFlags::Path, pos_idx, Re, 1.0, ScaleModes::AUTO);
    phase3->addLUNormBound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0,
                           ScaleModes::AUTO);

    Eigen::VectorXd m0_p3_val(1);
    m0_p3_val << m0_phase3;
    phase3->addBoundaryValue(PhaseRegionFlags::Front, m_idx, m0_p3_val, ScaleModes::AUTO);

    Eigen::VectorXd tf3_val(1);
    tf3_val << tf_phase3;
    phase3->addBoundaryValue(PhaseRegionFlags::Back, t_idx, tf3_val, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 4: upper stage
    ///////////////////////////////////////////////////////////////////////////
    auto phase4 = std::make_shared<ODEPhase<RocketODE>>(ode4, tmode);
    phase4->setTraj(IG4, nSegs);
    phase4->setControlMode(ControlModes::HighestOrderSpline);

    phase4->addLowerNormBound(PhaseRegionFlags::Path, pos_idx, Re, 1.0, ScaleModes::AUTO);
    phase4->addLUNormBound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0,
                           ScaleModes::AUTO);

    Eigen::VectorXd m0_p4_val(1);
    m0_p4_val << m0_phase4;
    phase4->addBoundaryValue(PhaseRegionFlags::Front, m_idx, m0_p4_val, ScaleModes::AUTO);

    // Upper bound on final time
    phase4->addUpperVarBound(PhaseRegionFlags::Back, 7, tf_phase4, 1.0, ScaleModes::AUTO);

    // Terminal orbital element constraint
    auto target_orbit_fn = MakeTargetOrbit(at, et, istart, Ot, Wt);
    phase4->addEqualCon(PhaseRegionFlags::Back,
                        GenericFunction<-1, -1>(target_orbit_fn), rv_idx,
                        ScaleModes::AUTO);

    // Maximize final mass (minimise -m)
    phase4->addValueObjective(PhaseRegionFlags::Back, 6, -1.0, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Optimal Control Problem — link phases
    ///////////////////////////////////////////////////////////////////////////
    OptimalControlProblem ocp;
    ocp.addPhase(phase1);
    ocp.addPhase(phase2);
    ocp.addPhase(phase3);
    ocp.addPhase(phase4);

    // Continuity in everything except mass (var 6)
    auto link_vars = make_idx({0, 1, 2, 3, 4, 5, 7, 8, 9, 10});
    ocp.addForwardLinkEqualCon(phase1, phase4, link_vars);

    ocp.optimizer->set_OptLSMode("L1");
    ocp.optimizer->set_SoeLSMode("L1");
    ocp.optimizer->set_MaxLSIters(2);
    ocp.optimizer->set_PrintLevel(1);

    ///////////////////////////////////////////////////////////////////////////
    // Solve
    ///////////////////////////////////////////////////////////////////////////
    std::cout << "Solving Delta III launch vehicle problem ...\n" << std::flush;

    const auto status = ocp.solve_optimize();

    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "Delta3Launch: FAILED (status " << static_cast<int>(status) << ")\n";
        return 1;
    }

    auto traj4 = phase4->returnTraj();
    if (traj4.empty()) {
        std::cerr << "Delta3Launch: solver converged but trajectory is empty\n";
        return 1;
    }

    const double final_mass_kg = traj4.back()[6] * Mstar;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Delta3Launch: Final mass = " << final_mass_kg << " kg\n";

    return 0;
}
