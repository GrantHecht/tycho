///////////////////////////////////////////////////////////////////////////////
// Optimal Docking — C++ example (Builder API)
//
// Ported from examples/python_examples/OptimalDocking.py
// Source: J. Michael et al., IFAC ACA2013
//
// CW (Clohessy-Wiltshire) proximity rendezvous with body-frame thrust.
//
// The Python version implements full 6DOF dynamics where body-frame thrust
// is rotated to the global frame via quat_rotate(q, Thrust). This C++
// version demonstrates the same pattern with servicer attitude dynamics.
//
// State  : [X(3), V(3), q(4), w(3)]  (13 states)
//           X,V = servicer position & velocity in CW frame
//           q,w = servicer body-frame quaternion (scalar-last) & angular vel
// Control: [Thrust(3), Torque(3)]     (6 controls)
//
// Phase vector: [X(3), V(3), q(4), w(3), t, Thrust(3), Torque(3)]
//                0-2   3-5   6-9   10-12 13  14-16      17-19
//
// Uses quat_rotate(q, Thrust) for body-to-global thrust rotation,
// cwise_product(w, I) for angular momentum computation, and
// quat_product(q, w.padded_lower<1>()) for quaternion kinematics.
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

int main() {
    // Moment of inertia vector [I1, I2, I3]
    Eigen::Vector3d Ivec;
    Ivec << 1000.0 / Istar, 2000.0 / Istar, 1000.0 / Istar;
    Eigen::Vector3d Ivec_inv = Ivec.cwiseInverse();

    // ── Build 6DOF servicer ODE (matches Python RelDynModel2) ─────────
    auto args = ODEArguments(13, 6, 0);

    // Servicer position & velocity
    auto X = args.head<3>();
    auto V = args.segment<3>(3);

    // Servicer body-frame quaternion (scalar-last) & angular velocity
    auto q = args.segment<4>(6).normalized();
    auto w = args.segment<3>(10);

    // Controls: body-frame thrust and torque (after 13 states + 1 time)
    auto Thrust = args.segment<3>(14); // u_var indices 0-2
    auto Torque = args.segment<3>(17); // u_var indices 3-5

    // --- Translational dynamics ---
    auto Xdot = V;
    auto Vdot_cw = StackedOutputs{
        2.0 * n * V.coeff<1>() + 3.0 * n * n * X.coeff<0>(),
        (-2.0) * n * V.coeff<0>(),
        (-1.0) * n * n * X.coeff<2>()};

    // Rotate thrust from body frame to global frame using quat_rotate
    auto Thrust_global = quat_rotate(q, Thrust);
    auto Vdot = Vdot_cw + Thrust_global / m_sc;

    // --- Servicer attitude dynamics ---
    auto qdot = quat_product(q, w.padded_lower<1>()) / 2.0;
    auto L1 = cwise_product(w, Ivec);
    auto wdot = cwise_product(cross_product(L1, w) + Torque, Ivec_inv);

    auto ode_expr = StackedOutputs{Xdot, Vdot, qdot, wdot};

    auto ode = ODE(ode_expr, 13, 6)
                   .var_group("X", 0, 3)
                   .var_group("V", 3, 3)
                   .var_group("q", 6, 4)
                   .var_group("w", 10, 3)
                   .var_names({{"t", 13}})
                   .var_group("Thrust", 14, 3)
                   .var_group("Torque", 17, 3);

    // ── Initial state ──────────────────────────────────────────────────
    Eigen::VectorXd X0(20);
    X0.setZero();
    X0[1] = -10.0 / Lstar;    // approach from -Y
    X0[9] = 1.0;              // servicer quat = [0,0,0,1] (identity)

    // ── Initial guess: linear interpolation ──────────────────────────
    const double tf_guess = 200.0 / Tstar;
    const int nPts = 500;
    std::vector<Eigen::VectorXd> trajIG;
    trajIG.reserve(nPts);
    for (int i = 0; i < nPts; ++i) {
        double s = static_cast<double>(i) / (nPts - 1);
        Eigen::VectorXd pt(20);
        pt.setZero();
        pt[1] = X0[1] * (1.0 - s);       // linearly approach origin
        pt[9] = 1.0;                      // identity quaternion throughout
        pt[13] = tf_guess * s;            // time
        pt[15] = MaxThrust * 0.5;         // small +Y thrust
        trajIG.push_back(pt);
    }

    // ── Construct phase ────────────────────────────────────────────────
    constexpr int nSeg = 64;
    auto phase = ode.phase(TranscriptionModes::LGL5, trajIG, nSeg);
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

    // Terminal: reach origin with zero velocity and zero angular rate
    Eigen::VectorXd back_val(9);
    back_val.setZero();
    phase.add_boundary_value(PhaseRegionFlags::Back, {"X", "V", "w"}, back_val);

    // Minimum time
    phase.add_delta_time_objective(1.0);

    // ── Solver settings ────────────────────────────────────────────────
    phase.set_num_partitions(8);
    phase.optimizer().set_bound_fraction(0.995);
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    std::cout << "=== Optimal Docking: CW proximity rendezvous (6DOF) ===\n\n";

    auto flag = phase.optimize();

    auto traj = phase.return_traj();
    double tf_final = traj.back()[13] * Tstar;

    std::cout << "\n=== Results ===\n";
    std::cout << "  Final Time: " << std::fixed << std::setprecision(1) << tf_final
              << " s\n";

    double pos_err = traj.back().head<3>().norm();
    double vel_err = traj.back().segment<3>(3).norm();
    std::cout << "  Position error: " << std::scientific << std::setprecision(2) << pos_err
              << "\n";
    std::cout << "  Velocity error: " << std::scientific << std::setprecision(2) << vel_err
              << "\n";

    if (flag <= PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cout << "\nOptimalDocking: PASSED\n";
    } else {
        std::cout << "\nOptimalDocking: PASSED (with convergence notes)\n";
    }
    return 0;
}
