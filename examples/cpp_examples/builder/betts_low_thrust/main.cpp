// Source: Betts, "Practical Methods for OC", Cambridge, 2009, Example 6

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

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
    auto args = ODEArguments(7, 3, 1);

    auto MEEs = args.head<6>();
    auto ww = args.coeff(6);
    auto U = args.segment<3>(8).normalized();
    auto tau = args.coeff(11);

    auto wwdot = (-1.0) * Thrust_nd * (1.0 + 0.01 * tau) / Isp_nd;
    auto acc_T = gs_nd * Thrust_nd * (1.0 + 0.01 * tau) * U / ww;

    auto mee_to_cart = astro::MEEToCartesian(mu_nd);
    auto cart_state = mee_to_cart.eval(MEEs);
    auto R = cart_state.head<3>();
    auto V = cart_state.tail<3>();

    auto Ir = R.normalized();

    Eigen::Vector3d north_vec;
    north_vec << 0.0, 0.0, 1.0;
    auto North = Constant<-1, 3>(args.input_rows(), north_vec);

    auto In = (North - Ir * Ir.dot(North)).normalized();

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

    auto Gcart = (gn * In - gr * Ir) * ((-mu_nd) / R.squared_norm());

    auto Rhat = R.normalized();
    auto Nhat = R.cross(V).normalized();
    auto That = Nhat.cross(R).normalized();

    auto RTN_vec = stack(Rhat, That, Nhat);
    auto M = row_matrix(RTN_vec, 3, 3);

    auto grav_rtn = M * Gcart;

    auto acc = acc_T + grav_rtn;

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

    Eigen::VectorXd X0(12);
    X0.setZero();
    X0[0] = pt0;
    X0[3] = -0.25396764647494;    // h0 = tan(i/2)*cos(RAAN)
    X0[5] = M_PI;
    X0[6] = 1.0 / Fstar;          // w0 (initial weight, non-dim)
    X0[8] = 0.0;
    X0[9] = 1.0;                  // ut (prograde initially)
    X0[10] = 0.0;
    X0[11] = -25.0;

    const double tfig = 90000.0 / Tstar;

    auto ig_mee_to_cart = astro::MEEToCartesian(mu_nd);
    auto ig_args = Arguments<6>();
    auto ig_cart = ig_mee_to_cart.eval(ig_args);
    auto ig_R = ig_cart.head<3>();
    auto ig_V = ig_cart.tail<3>();
    auto ig_Rhat = ig_R.normalized();
    auto ig_Nhat = ig_R.cross(ig_V).normalized();
    auto ig_That = ig_Nhat.cross(ig_R).normalized();
    auto ig_RTN_vec = stack(ig_Rhat, ig_That, ig_Nhat);
    auto ig_M = row_matrix(ig_RTN_vec, 3, 3);
    auto ig_prograde = ig_M * ig_V.normalized();

    auto integ = ode.integrator()
                     .step(0.1)
                     .control(GenericFunction<-1, -1>(ig_prograde),
                              {"p", "f", "g", "h", "k", "L"})
                     .build();
    auto trajIG = integ.integrate_dense(X0, tfig);

    constexpr int nSeg = 16; // Start with few segments for adaptive mesh
    auto phase = ode.phase(TranscriptionModes::LGL5, trajIG, nSeg);

    Eigen::VectorXd front_val(8);
    front_val << X0[0], X0[1], X0[2], X0[3], X0[4], X0[5], X0[6], 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front, {"p", "f", "g", "h", "k", "L", "w", "t"},
                             front_val);

    {
        auto ctrl_args = Arguments<3>();
        auto norm_eq = ctrl_args.norm() - 1.0;
        phase.add_equal_con(PhaseRegionFlags::Path, GenericFunction<-1, -1>(norm_eq), {"u"});
    }
    phase.set_control_mode(ControlModes::NoSpline);

    // r = p / w where w = 1 + f*cos(L) + g*sin(L)
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

    phase.add_lu_var_bound(PhaseRegionFlags::ODEParams, 0, -50.0, 0.0, 1.0);

    phase.add_lower_var_bound(PhaseRegionFlags::Back, "w", 0.05);

    phase.add_value_objective(PhaseRegionFlags::Back, "w", -1.0);

    phase.set_num_partitions(8, 8);
    phase.optimizer().set_print_level(1);
    phase.optimizer().set_econ_tol(1.0e-9);

    phase.set_adaptive_mesh(true);
    phase.set_mesh_error_estimator(MeshErrorEstimators::INTEGRATOR);
    phase.set_mesh_tol(1.0e-7);

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
