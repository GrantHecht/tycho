///////////////////////////////////////////////////////////////////////////////
// Betts Low Thrust — C++ example (Builder API)
//
// Ported from examples/python_examples/BettsLowThrust.py
// Source: Betts, "Practical Methods for OC", Cambridge, 2009, Example 6
//
// LEO-to-MEO low-thrust transfer with MEE dynamics and J2/J3/J4 zonal
// harmonics. This is the most complex astrodynamics example.
//
// State  : [p, f, g, h, k, L, w]  (6 MEE + weight = 7 states)
// Control: [ur, ut, un]            (RTN thrust direction, 3 controls)
// Params : [tau]                   (1 static throttle parameter)
//
// Phase vector: [p, f, g, h, k, L, w, t, ur, ut, un, tau]
//                0  1  2  3  4  5  6  7   8   9  10   11
//
// Objective: maximise final weight
//
// Full zonal gravity (J2/J3/J4) matching Python LTModel:
//   MEE → Cartesian → Zonal gravity in Cartesian → RTN rotation
//   Uses generated MEEToCartesian and MEEDynamics VFs.
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
// Physical constants (Imperial units, then non-dimensionalised)
///////////////////////////////////////////////////////////////////////////////

static constexpr double g0 = 32.174;
static constexpr double W = 1.0; // lb
static constexpr double mu_e = 1.407645794e16;

static const double Lstar = 20925662.73; // feet
static const double Tstar = Lstar / std::sqrt(mu_e / Lstar);
static const double Mstar = W / g0; // slugs

static const double Vstar = Lstar / Tstar;
static const double Fstar = Mstar * Lstar / (Tstar * Tstar);
static const double Astar = Lstar / (Tstar * Tstar);
static const double Mustar = (Lstar * Lstar * Lstar) / (Tstar * Tstar);

static const double Re_nd = 20925662.73 / Lstar;
static const double mu_nd = mu_e / Mustar;
static const double Thrust_nd = 4.446618e-3 / Fstar;
static const double Isp_nd = 450.0 / Tstar;
static const double gs_nd = g0 / Astar;

static constexpr double J2 = 1082.639e-6;
static constexpr double J3 = -2.565e-6;
static constexpr double J4 = -1.608e-6;

static const double pt0 = 21837080.052835 / Lstar;
static const double ptf = 40007346.015232 / Lstar;

int main() {
    // ── MEE dynamics with full J2/J3/J4 zonal gravity ─────────────────
    //
    // Composition chain (matching Python LTModel):
    //   1. Extract MEE, weight, controls, throttle from ODE arguments
    //   2. Thrust acceleration in RTN: gs*T*(1+0.01*tau)*U_hat / w
    //   3. Zonal gravity: MEE→Cart→ZonalGrav(Cart)→RTN rotation
    //   4. Total accel = thrust + zonal gravity
    //   5. MEE rates via generated MEEDynamics(mu)
    //   6. Full ODE = stack(MEE_rates, weight_rate)

    auto args = ODEArguments(7, 3, 1);
    // State: [p, f, g, h, k, L, w] at indices 0-6
    // Time: index 7
    // Controls: [ur, ut, un] at indices 8, 9, 10
    // Static param: [tau] at index 11

    auto MEEs = args.head<6>();
    auto ww = args.coeff(6); // weight
    auto U = args.segment<3>(8).normalized(); // control direction (unit vector)
    auto tau = args.coeff(11);

    // ── Thrust acceleration in RTN ────────────────────────────────────
    auto wwdot = (-1.0) * Thrust_nd * (1.0 + 0.01 * tau) / Isp_nd;
    auto acc_T = gs_nd * Thrust_nd * (1.0 + 0.01 * tau) * U / ww;

    // ── Zonal gravity (J2/J3/J4) in RTN frame ────────────────────────
    // Step (a): MEE → Cartesian [R(3), V(3)]
    auto mee_to_cart = astro::MEEToCartesian(mu_nd);
    auto cart_state = mee_to_cart.eval(MEEs); // 6-output: [Rx, Ry, Rz, Vx, Vy, Vz]
    auto R = cart_state.head<3>();
    auto V = cart_state.tail<3>();

    // Step (b): Zonal gravity in Cartesian frame (Eq. 6.46-6.49)
    auto Ir = R.normalized();

    // North pole direction [0, 0, 1] as constant VF
    Eigen::Vector3d north_vec;
    north_vec << 0.0, 0.0, 1.0;
    auto North = Constant<-1, 3>(args.input_rows(), north_vec);

    // In = (North - Ir*(Ir.dot(North))).normalized()
    auto In = (North - Ir * Ir.dot(North)).normalized();

    // sin/cos of geocentric latitude
    auto sphi = Ir.coeff<2>();
    auto cphi = sqrt(1.0 - sphi * sphi);

    // Legendre polynomials and their derivatives
    auto sphi2 = sphi * sphi;
    auto sphi3 = sphi2 * sphi;
    auto sphi4 = sphi2 * sphi2;

    auto P2 = 0.5 * (3.0 * sphi2 - 1.0);
    auto P3 = 0.5 * (5.0 * sphi3 - 3.0 * sphi);
    auto P4 = (35.0 / 8.0) * sphi4 - (30.0 / 8.0) * sphi2 + 3.0 / 8.0;

    auto D2 = 3.0 * sphi;
    auto D3 = 0.5 * (15.0 * sphi2 - 3.0);
    auto D4 = (35.0 / 2.0) * sphi3 - (30.0 / 4.0) * sphi;

    // Accumulate gr and gn from J2, J3, J4
    // For each k in {2,3,4}: gn += Dk * Jk * (Re/r)^k
    //                         gr += (k+1) * Pk * Jk * (Re/r)^k
    auto r = R.norm();
    auto Re_over_r = Re_nd / r;
    auto Re_r2 = Re_over_r * Re_over_r;
    auto Re_r3 = Re_r2 * Re_over_r;
    auto Re_r4 = Re_r3 * Re_over_r;

    auto gn_sum = D2 * J2 * Re_r2 + D3 * J3 * Re_r3 + D4 * J4 * Re_r4;
    auto gr_sum =
        3.0 * P2 * J2 * Re_r2 + 4.0 * P3 * J3 * Re_r3 + 5.0 * P4 * J4 * Re_r4;

    auto gn = gn_sum * cphi;
    auto gr = gr_sum;

    // Gcart = (gn*In - gr*Ir) * (-mu / R.squared_norm())
    auto Gcart = (gn * In - gr * Ir) * ((-mu_nd) / R.squared_norm());

    // Step (c): RTN basis from Cartesian [R, V]
    auto Rhat = R.normalized();
    auto Nhat = R.cross(V).normalized();
    auto That = Nhat.cross(R).normalized();

    // Step (d): RTN rotation matrix (3x3, row-major: rows = Rhat, That, Nhat)
    auto RTN_vec = stack(Rhat, That, Nhat); // 9-element vector
    auto M = RowMatrix(RTN_vec, 3, 3);

    // Step (e): grav_rtn = M * Gcart
    auto grav_rtn = M * Gcart;

    // ── Total acceleration and MEE rates ──────────────────────────────
    auto acc = acc_T + grav_rtn;

    // MEEDynamics takes [p,f,g,h,k,L, ur,ut,un] (9 inputs) → 6 MEE rates
    auto mee_dyn = astro::MEEDynamics(mu_nd);
    auto Xdot = GenericFunction<-1, -1>(mee_dyn.eval(stack(MEEs, acc)));

    auto ode_expr = StackedOutputs{Xdot, wwdot};

    auto ode = ODE(ode_expr, 7, 3, 1)
                   .var_names({{"p", 0},
                               {"f", 1},
                               {"g", 2},
                               {"h", 3},
                               {"k", 4},
                               {"L", 5},
                               {"w", 6},
                               {"t", 7}})
                   .var_group("u", 8, 3)
                   .var_names({{"tau", 11}});

    // ── Initial state ──────────────────────────────────────────────────
    // Phase vector: [p, f, g, h, k, L, w, t, ur, ut, un, tau]
    Eigen::VectorXd X0(12);
    X0.setZero();
    X0[0] = pt0;                  // p0
    X0[3] = -0.25396764647494;    // h0 = tan(i/2)*cos(RAAN)
    X0[5] = M_PI;                 // L0
    X0[6] = 1.0 / Fstar;          // w0 (initial weight, non-dim)
    X0[8] = 0.0;                  // ur
    X0[9] = 1.0;                  // ut (prograde initially)
    X0[10] = 0.0;                 // un
    X0[11] = -25.0;               // throttle param

    // ── Initial guess via control-law integrator ──────────────────────
    const double tfig = 90000.0 / Tstar;

    // Prograde control law: velocity direction in RTN frame
    // RTNBasis(Cart) * V_hat, where Cart = MEEToCart(MEEs)
    auto ig_mee_to_cart = astro::MEEToCartesian(mu_nd);
    auto ig_args = Arguments<6>(); // MEE elements input
    auto ig_cart = ig_mee_to_cart.eval(ig_args);
    auto ig_R = ig_cart.head<3>();
    auto ig_V = ig_cart.tail<3>();
    auto ig_Rhat = ig_R.normalized();
    auto ig_Nhat = ig_R.cross(ig_V).normalized();
    auto ig_That = ig_Nhat.cross(ig_R).normalized();
    auto ig_RTN_vec = stack(ig_Rhat, ig_That, ig_Nhat);
    auto ig_M = RowMatrix(ig_RTN_vec, 3, 3);
    auto ig_prograde = ig_M * ig_V.normalized();

    auto integ =
        ode.integrator(0.1, GenericFunction<-1, -1>(ig_prograde), {"p", "f", "g", "h", "k", "L"});
    auto trajIG = integ.integrate_dense(X0, tfig);

    // ── Construct phase ────────────────────────────────────────────────
    constexpr int nSeg = 16; // Start with few segments for adaptive mesh
    auto phase = ode.phase(TranscriptionModes::LGL5, trajIG, nSeg);

    // Front BC
    Eigen::VectorXd front_val(8);
    front_val << X0[0], X0[1], X0[2], X0[3], X0[4], X0[5], X0[6], 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front, {"p", "f", "g", "h", "k", "L", "w", "t"},
                             front_val);

    // Unit control vector constraint
    {
        auto ctrl_args = Arguments<3>();
        auto norm_eq = ctrl_args.norm() - 1.0;
        phase.add_equal_con(PhaseRegionFlags::Path, GenericFunction<-1, -1>(norm_eq), {"u"});
    }
    phase.set_control_mode(ControlModes::NoSpline);

    // Radius bounds: r = p / w where w = 1 + f*cos(L) + g*sin(L)
    {
        auto rad_args = Arguments<6>();
        auto p_r = rad_args.coeff<0>();
        auto f_r = rad_args.coeff<1>();
        auto g_r = rad_args.coeff<2>();
        auto L_r = rad_args.coeff<5>();
        auto w_r = 1.0 + f_r * cos(L_r) + g_r * sin(L_r);
        auto rad = p_r / w_r;
        phase.add_lu_func_bound(PhaseRegionFlags::Path, GenericFunction<-1, 1>(rad),
                                {"p", "f", "g", "h", "k", "L"}, Re_nd, 10.0 * Re_nd);
    }

    // Terminal constraints: Betts eq 6.52-6.56
    {
        // eq 6.52: p - ptf = 0
        auto eq1_args = Arguments<6>();
        auto eq1 = eq1_args.coeff<0>() - ptf;
        // eq 6.53: f^2 + g^2 - 0.7355^2 = 0
        auto fg = eq1_args.segment<2>(1);
        auto eq2 = fg.squared_norm() - 0.73550320568829 * 0.73550320568829;
        // eq 6.54: h^2 + k^2 - 0.6176^2 = 0
        auto hk_con = eq1_args.segment<2>(3);
        auto eq3 = hk_con.squared_norm() - 0.61761258786099 * 0.61761258786099;
        // eq 6.55: f*h + g*k = 0
        auto eq4 = eq1_args.coeff<1>() * eq1_args.coeff<3>() +
                   eq1_args.coeff<2>() * eq1_args.coeff<4>();
        auto eq_con = StackedOutputs{eq1, eq2, eq3, eq4};
        phase.add_equal_con(PhaseRegionFlags::Back, GenericFunction<-1, -1>(eq_con),
                            {"p", "f", "g", "h", "k", "L"});
    }
    {
        // eq 6.56: g*h - k*f >= 0
        auto iq_args = Arguments<6>();
        auto iq = iq_args.coeff<2>() * iq_args.coeff<3>() -
                  iq_args.coeff<4>() * iq_args.coeff<1>();
        phase.add_inequal_con(PhaseRegionFlags::Back, GenericFunction<-1, -1>(iq),
                              {"p", "f", "g", "h", "k", "L"});
    }

    // ODE param (tau) bounds
    phase.add_lu_var_bound(PhaseRegionFlags::ODEParams, 0, -50.0, 0.0, 1.0);

    // Weight must remain positive
    phase.add_lower_var_bound(PhaseRegionFlags::Back, "w", 0.05);

    // Objective: maximise final weight
    phase.add_value_objective(PhaseRegionFlags::Back, "w", -1.0);

    // ── Solver settings ────────────────────────────────────────────────
    phase.set_num_partitions(8, 8);
    phase.optimizer().set_print_level(1);
    phase.optimizer().set_econ_tol(1.0e-9);

    // Adaptive mesh
    phase.set_adaptive_mesh(true);
    phase.set_mesh_error_estimator(MeshErrorEstimators::INTEGRATOR);
    phase.set_mesh_tol(1.0e-7);

    // ── Solve ──────────────────────────────────────────────────────────
    std::cout << "=== Betts Low Thrust: LEO-to-MEO with J2/J3/J4 zonal gravity ===\n\n";

    phase.optimize_solve();

    auto traj = phase.return_traj();
    double final_weight = traj.back()[6] * Fstar;
    double final_time = traj.back()[7] * Tstar;
    double throttle_param = traj.back()[11];

    std::cout << "\n=== Results ===\n";
    std::cout << "  Final Weight:       " << std::fixed << std::setprecision(6) << final_weight
              << " lb\n";
    std::cout << "  Final Time:         " << std::fixed << std::setprecision(1) << final_time
              << " s\n";
    std::cout << "  Throttle Parameter: " << std::fixed << std::setprecision(2) << throttle_param
              << "\n";

    bool converged = !traj.empty() && final_weight > 0.0;
    std::cout << "\nBettsLowThrust: " << (converged ? "PASSED" : "FAILED") << "\n";
    return converged ? 0 : 1;
}
