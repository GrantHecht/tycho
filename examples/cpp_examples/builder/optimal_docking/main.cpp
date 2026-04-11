///////////////////////////////////////////////////////////////////////////////
// Optimal Docking — C++ example (Builder API)
//
// Ported from examples/python_examples/OptimalDocking.py
// Source: J. Michael et al., IFAC ACA2013
//
// CW (Clohessy-Wiltshire) proximity rendezvous with body-frame thrust.
//
// This script solves the problem in two formulations:
//
// Form1: Both servicer (13 states) and target (7 states) are integrated
//   together in one ODE (20 states, 6 controls). The rendezvous constraint
//   (RendCon) takes the full 20-variable state.
//
// Form2: The target trajectory is pre-computed via a torque-free ODE and
//   stored in an LGLInterpTable. The phase ODE only has servicer dynamics
//   (13 states, 6 controls). The rendezvous constraint (RendCon2) looks up
//   the target state from the table at the current time.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

///////////////////////////////////////////////////////////////////////////////
// Physical constants (non-dimensionalised)
///////////////////////////////////////////////////////////////////////////////

static constexpr double Lstar = 10.0;
static constexpr double Tstar = 30.0;
static constexpr double Mstar_val = 10.0;

static const double Astar = Lstar / (Tstar * Tstar);
static const double Fstar = Astar * Mstar_val;
static const double Istar = Mstar_val * Lstar * Lstar;

static const double a = 7071000.0 / Lstar;
static const double mu = 3.986e14 / (Lstar * Lstar * Lstar / (Tstar * Tstar));
static const double n = std::sqrt(mu / (a * a * a));
static const double m_sc = 100.0 / Mstar_val;

static const double MaxThrust = 0.1 / Fstar;
static const double MaxTorque = 1.0 / (Fstar * Lstar);

static const double Srad = 1.0 / Lstar;

///////////////////////////////////////////////////////////////////////////////
// Shared helper: build the RendCon function (20 inputs -> 6 outputs)
//
// Inputs: [X(3), V(3), q(4), w(3), p(4), phi(3)] = 20
// Outputs: position + velocity error at docking port
///////////////////////////////////////////////////////////////////////////////
static GenericFunction<-1, -1> make_rendcon(const Eigen::Vector3d &ud) {
    auto rc_args = Arguments<20>();

    auto X = rc_args.head<3>();
    auto V = rc_args.segment<3, 3>();
    auto q_s = rc_args.segment<4, 6>().normalized();
    auto w_s = rc_args.segment<3, 10>();
    auto p_t = rc_args.segment<4, 13>().normalized();
    auto phi_t = rc_args.segment<3, 17>();

    // Servicer docking port contribution
    auto Xdq = quat_rotate(q_s, ud);
    auto vdq = quat_rotate(q_s, w_s);
    auto Vdq = cross_product(vdq, Xdq);

    // Target docking port contribution
    auto Xdp = quat_rotate(p_t, ud);
    auto vdp = quat_rotate(p_t, phi_t);
    auto Vdp = cross_product(vdp, Xdp);

    return GenericFunction<-1, -1>(
        StackedOutputs{X + Xdq - Xdp, V + Vdq - Vdp});
}

///////////////////////////////////////////////////////////////////////////////
// Form2: RendCon2 (14 inputs -> 6 outputs)
//
// Inputs: [X(3), V(3), q(4), w(3), t(1)] = 14
// Looks up target state [p(4), phi(3)] from LGLInterpTable at time t,
// then delegates to RendCon.
///////////////////////////////////////////////////////////////////////////////
static GenericFunction<-1, -1>
make_rendcon2(const Eigen::Vector3d &ud,
              const std::shared_ptr<LGLInterpTable> &tab) {
    // Variable indices for all 7 target state variables
    Eigen::VectorXi target_vars(7);
    for (int i = 0; i < 7; ++i)
        target_vars[i] = i;

    auto rc2_args = Arguments<14>();
    auto X = rc2_args.head<3>();
    auto V = rc2_args.segment<3, 3>();
    auto q_s = rc2_args.segment<4, 6>();
    auto w_s = rc2_args.segment<3, 10>();
    auto t = rc2_args.coeff<13>();

    // Look up target state at time t
    auto target_at_t = lgl_interp(tab, target_vars, t);

    // Compose with RendCon(ud): feed [X, V, q, w, target_at_t] (20 inputs)
    auto rendcon = make_rendcon(ud);
    return GenericFunction<-1, -1>(
        rendcon.eval(StackedOutputs{X, V, q_s, w_s, target_at_t}));
}

///////////////////////////////////////////////////////////////////////////////
// Build servicer-only ODE (13 states, 6 controls) — used by both forms
///////////////////////////////////////////////////////////////////////////////
static ODE build_servicer_ode(const Eigen::Vector3d &Ivec) {
    Eigen::Vector3d Ivec_inv = Ivec.cwiseInverse();

    auto args = ODEArguments(13, 6, 0);

    auto X = args.head<3>();
    auto V = args.segment<3>(3);
    auto q = args.segment<4>(6).normalized();
    auto w = args.segment<3>(10);
    auto Thrust = args.segment<3>(14);
    auto Torque = args.segment<3>(17);

    auto Xdot = V;
    auto Vdot_cw = StackedOutputs{
        2.0 * n * V.coeff<1>() + 3.0 * n * n * X.coeff<0>(),
        (-2.0) * n * V.coeff<0>(),
        (-1.0) * n * n * X.coeff<2>()};

    auto Thrust_global = quat_rotate(q, Thrust);
    auto Vdot = Vdot_cw + Thrust_global / m_sc;

    auto qdot = quat_product(q, w.padded_lower<1>()) / 2.0;
    auto L1 = cwise_product(w, Ivec);
    auto wdot = cwise_product(cross_product(L1, w) + Torque, Ivec_inv);

    auto ode_expr = StackedOutputs{Xdot, Vdot, qdot, wdot};

    return ODE(ode_expr, 13, 6)
        .var_group("X", 0, 3)
        .var_group("V", 3, 3)
        .var_group("q", 6, 4)
        .var_group("w", 10, 3)
        .var_names({{"t", 13}})
        .var_group("Thrust", 14, 3)
        .var_group("Torque", 17, 3);
}

///////////////////////////////////////////////////////////////////////////////
// Build full ODE (20 states, 6 controls) — Form1 only
///////////////////////////////////////////////////////////////////////////////
static ODE build_full_ode(const Eigen::Vector3d &I1, const Eigen::Vector3d &I2) {
    Eigen::Vector3d I1_inv = I1.cwiseInverse();
    Eigen::Vector3d I2_inv = I2.cwiseInverse();

    auto args = ODEArguments(20, 6, 0);

    // Servicer
    auto X = args.head<3>();
    auto V = args.segment<3>(3);
    auto q = args.segment<4>(6).normalized();
    auto w = args.segment<3>(10);

    // Target
    auto p = args.segment<4>(13).normalized();
    auto phi = args.segment<3>(17);

    // Controls
    auto Thrust = args.segment<3>(21);
    auto Torque = args.segment<3>(24);

    // Translational dynamics
    auto Xdot = V;
    auto Vdot_cw = StackedOutputs{
        2.0 * n * V.coeff<1>() + 3.0 * n * n * X.coeff<0>(),
        (-2.0) * n * V.coeff<0>(),
        (-1.0) * n * n * X.coeff<2>()};

    auto Thrust_global = quat_rotate(q, Thrust);
    auto Vdot = Vdot_cw + Thrust_global / m_sc;

    // Servicer attitude dynamics
    auto qdot = quat_product(q, w.padded_lower<1>()) / 2.0;
    auto L1 = cwise_product(w, I1);
    auto wdot = cwise_product(cross_product(L1, w) + Torque, I1_inv);

    // Target attitude dynamics (torque-free)
    auto pdot = quat_product(p, phi.padded_lower<1>()) / 2.0;
    auto L2 = cwise_product(phi, I2);
    auto phidot = cwise_product(cross_product(L2, phi), I2_inv);

    auto ode_expr = StackedOutputs{Xdot, Vdot, qdot, wdot, pdot, phidot};

    return ODE(ode_expr, 20, 6)
        .var_group("X", 0, 3)
        .var_group("V", 3, 3)
        .var_group("q", 6, 4)
        .var_group("w", 10, 3)
        .var_group("p", 13, 4)
        .var_group("phi", 17, 3)
        .var_names({{"t", 20}})
        .var_group("Thrust", 21, 3)
        .var_group("Torque", 24, 3);
}

///////////////////////////////////////////////////////////////////////////////
// Build torque-free target ODE (7 states, 0 controls)
///////////////////////////////////////////////////////////////////////////////
static ODE build_torquefree_ode(const Eigen::Vector3d &I2) {
    Eigen::Vector3d I2_inv = I2.cwiseInverse();

    auto args = ODEArguments(7, 0, 0);

    auto p = args.head<4>().normalized();
    auto phi = args.segment<3>(4);

    auto pdot = quat_product(p, phi.padded_lower<1>()) / 2.0;
    auto L2 = cwise_product(phi, I2);
    auto phidot = cwise_product(cross_product(L2, phi), I2_inv);

    auto ode_expr = StackedOutputs{pdot, phidot};

    return ODE(ode_expr, 7, 0);
}

///////////////////////////////////////////////////////////////////////////////
// Form2 — Servicer-only ODE, target via LGLInterpTable
///////////////////////////////////////////////////////////////////////////////
static bool run_form2() {
    std::cout << "=== OptimalDocking Form2: LGLInterpTable target ===\n\n";

    Eigen::Vector3d Ivec;
    Ivec << 1000.0 / Istar, 2000.0 / Istar, 1000.0 / Istar;

    Eigen::Vector3d Udvec;
    Udvec << 0.0, 1.01 / Lstar, 0.0;

    // ── Build torque-free ODE and integrate target trajectory ──────────
    auto torquefree_ode = build_torquefree_ode(Ivec);
    auto integ_tf = torquefree_ode.integrator(0.01);
    integ_tf.adaptive_ = true;

    const double SimTime = 600.0 / Tstar;

    // Target initial state: [qx, qy, qz, qw, phix, phiy, phiz, t]
    Eigen::VectorXd target_IS(8);
    target_IS.setZero();
    target_IS[0] = 0.05;                                     // q_x
    target_IS[3] = std::sqrt(1.0 - 0.05 * 0.05);            // q_w
    target_IS[5] = 0.0349 * Tstar;                           // phi_y
    target_IS[6] = 0.017453 * Tstar;                         // phi_z
    // target_IS[7] = 0.0 (time)

    auto target_traj = integ_tf.integrate_dense(target_IS, SimTime, 2000);
    std::cout << "  Target trajectory: " << target_traj.size() << " points\n";

    // Create LGLInterpTable from torque-free ODE + trajectory
    auto target_tab =
        std::make_shared<LGLInterpTable>(torquefree_ode.function(), 7, 0, target_traj);

    // ── Build servicer ODE ────────────────────────────────────────────
    auto ode = build_servicer_ode(Ivec);

    // ── Initial state (13 states + 1 time + 6 controls = 20) ──────────
    Eigen::VectorXd X0(20);
    X0.setZero();
    X0[1] = -10.0 / Lstar;     // approach from -Y
    X0[9] = 1.0;               // servicer quat = identity [0,0,0,1]
    X0[14] = -MaxThrust;       // initial thrust guess
    X0[15] = MaxThrust;
    X0[19] = -MaxTorque / 4.0;

    // ── Integrate initial guess ────────────────────────────────────────
    auto integ = ode.integrator(0.01);
    integ.adaptive_ = true;
    auto trajIG = integ.integrate_dense(X0, 200.0 / Tstar, 1000);
    std::cout << "  Initial guess: " << trajIG.size() << " points\n";

    // ── Construct phase ────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, trajIG, 384);
    phase.set_control_mode(ControlModes::BlockConstant);

    // Front BC: fix all 13 states + time
    phase.add_boundary_value(PhaseRegionFlags::Front,
                             {"X", "V", "q", "w", "t"},
                             X0.head(14));

    // Thrust bounds
    for (int i = 14; i <= 16; ++i)
        phase.add_lu_var_bound(PhaseRegionFlags::Path, i, -MaxThrust, MaxThrust, 0.1);
    // Torque bounds
    for (int i = 17; i <= 19; ++i)
        phase.add_lu_var_bound(PhaseRegionFlags::Path, i, -MaxTorque, MaxTorque, 1.0);

    // Safety sphere constraint
    Eigen::VectorXi pos_vars(3);
    pos_vars << 0, 1, 2;
    phase.add_lower_norm_bound(PhaseRegionFlags::Path, pos_vars, 2.0 * Srad, 1.0);

    // Terminal rendezvous constraint using RendCon2 (looks up target from table)
    auto rendcon2 = make_rendcon2(Udvec, target_tab);
    Eigen::VectorXi back_vars(14);
    for (int i = 0; i < 14; ++i)
        back_vars[i] = i;
    phase.add_equal_con(PhaseRegionFlags::Back, rendcon2, back_vars);

    // Upper bound on total time
    phase.add_upper_delta_time_bound(SimTime);

    // Minimum time objective
    phase.add_delta_time_objective(1.0);

    // Solver settings
    phase.optimizer().set_bound_fraction(0.995);
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    auto flag = phase.optimize();

    auto traj = phase.return_traj();
    double tf_final = traj.back()[13] * Tstar;

    std::cout << "\n=== Form2 Results ===\n";
    std::cout << "  Final Time: " << std::fixed << std::setprecision(2) << tf_final
              << " s\n";

    if (flag <= PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cout << "  Form2: PASSED\n\n";
        return true;
    } else {
        std::cerr << "  Form2: FAILED (convergence flag = " << static_cast<int>(flag)
                  << ")\n\n";
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Form1 — Full 20-state ODE (servicer + target)
///////////////////////////////////////////////////////////////////////////////
static bool run_form1() {
    std::cout << "=== OptimalDocking Form1: Full 20-state ODE ===\n\n";

    Eigen::Vector3d Ivec;
    Ivec << 1000.0 / Istar, 2000.0 / Istar, 1000.0 / Istar;

    Eigen::Vector3d Udvec;
    Udvec << 0.0, 1.01 / Lstar, 0.0;

    // ── Build full ODE ────────────────────────────────────────────────
    auto ode = build_full_ode(Ivec, Ivec);

    // ── Initial state (20 states + 1 time + 6 controls = 27) ──────────
    Eigen::VectorXd X0(27);
    X0.setZero();
    X0[1] = -10.0 / Lstar;     // approach from -Y
    X0[9] = 1.0;               // servicer quat = identity

    // Target initial conditions
    X0[13] = 0.05;                                    // p_x
    X0[16] = std::sqrt(1.0 - X0[13] * X0[13]);       // p_w
    X0[18] = 0.0349 * Tstar;                          // phi_y
    X0[19] = 0.017453 * Tstar;                        // phi_z

    // Initial thrust guess
    X0[21] = -MaxThrust;
    X0[22] = MaxThrust;
    X0[26] = -MaxTorque / 4.0;

    // ── Integrate initial guess ────────────────────────────────────────
    auto integ = ode.integrator(0.01);
    auto trajIG = integ.integrate_dense(X0, 200.0 / Tstar, 1000);
    std::cout << "  Initial guess: " << trajIG.size() << " points\n";

    // ── Construct phase ────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, trajIG, 384);
    phase.set_control_mode(ControlModes::BlockConstant);

    // Front BC: fix all 20 states + time
    phase.add_boundary_value(PhaseRegionFlags::Front,
                             {"X", "V", "q", "w", "p", "phi", "t"},
                             X0.head(21));

    // Thrust bounds
    for (int i = 21; i <= 23; ++i)
        phase.add_lu_var_bound(PhaseRegionFlags::Path, i, -MaxThrust, MaxThrust, 0.1);
    // Torque bounds
    for (int i = 24; i <= 26; ++i)
        phase.add_lu_var_bound(PhaseRegionFlags::Path, i, -MaxTorque, MaxTorque, 1.0);

    // Safety sphere constraint
    Eigen::VectorXi pos_vars(3);
    pos_vars << 0, 1, 2;
    phase.add_lower_norm_bound(PhaseRegionFlags::Path, pos_vars, 2.0 * Srad, 1.0);

    // Terminal rendezvous constraint using RendCon (full 20-state)
    auto rendcon = make_rendcon(Udvec);
    Eigen::VectorXi back_vars(20);
    for (int i = 0; i < 20; ++i)
        back_vars[i] = i;
    phase.add_equal_con(PhaseRegionFlags::Back, rendcon, back_vars);

    // Minimum time objective
    phase.add_delta_time_objective(1.0);

    // Solver settings
    phase.optimizer().set_bound_fraction(0.995);
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    auto flag = phase.optimize();

    auto traj = phase.return_traj();
    double tf_final = traj.back()[20] * Tstar;

    std::cout << "\n=== Form1 Results ===\n";
    std::cout << "  Final Time: " << std::fixed << std::setprecision(2) << tf_final
              << " s\n";

    if (flag <= PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cout << "  Form1: PASSED\n\n";
        return true;
    } else {
        std::cerr << "  Form1: FAILED (convergence flag = " << static_cast<int>(flag)
                  << ")\n\n";
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////////
int main() {
    bool ok2 = run_form2();
    bool ok1 = run_form1();
    return (ok1 && ok2) ? 0 : 1;
}
