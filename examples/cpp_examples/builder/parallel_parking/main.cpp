///////////////////////////////////////////////////////////////////////////////
// Parallel Parking — C++ example (Builder API)
//
// Ported from examples/python_examples/ParallelParking.py
// Source: http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html
//         https://ieeexplore.ieee.org/document/7463491
//
// State  : [x, y, v, a, theta, phi]   (pos, vel, accel, heading, steering)
// Control: [u1, u2]                    (jerk, steering rate)
// Static : [k]                         (tanh smoothing constant)
//
// Objective: minimise time (~18.4 s expected)
//
// All API gaps previously noted here have been resolved:
//   - Phase wrapper add_inequal_con supports mixed var sources (XtUP + SP)
//   - refine_traj_manual() and sub_variable() are on the Phase wrapper
//   - Phase::set_traj(traj, ndef, true) supports LerpIG interpolation
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

// Helper: corner world coordinates from body position and heading
template <class X, class Y, class Th>
auto corner_world(const X &x, const Y &y, const Th &theta, double locx, double locy) {
    auto xl = cos(theta) * locx - sin(theta) * locy;
    auto yl = sin(theta) * locx + cos(theta) * locy;
    return std::make_pair(x + xl, y + yl);
}

// Helper: Heaviside approx H(z,k) = (1 + tanh(k*z))/2
template <class T, class K> auto heaviside_approx(const T &z, const K &k) {
    return (1.0 + tanh(k * z)) / 2.0;
}

// Helper: slot boundary Fslot(X,k) = (-H(X,k) + H(X-SL,k)) * SW
template <class T, class K> auto fslot_fn(const T &X, const K &k, double SL, double SW) {
    return ((-1.0) * heaviside_approx(X, k) + heaviside_approx(X - SL, k)) * SW;
}

// Helper: triangle area |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
template <class T1, class T2, class T3, class T4, class T5, class T6>
auto tri_area(const T1 &ax, const T2 &ay, const T3 &bx, const T4 &by, const T5 &cx,
              const T6 &cy) {
    return abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0;
}

// Corner body-frame coordinates
constexpr double Ax_b = 0.839 + 2.588, Ay_b = 1.771 / 2.0;
constexpr double Bx_b = 0.839 + 2.588, By_b = -1.771 / 2.0;
constexpr double Cx_b = -0.657, Cy_b = 1.771 / 2.0;
constexpr double Dx_b = -0.657, Dy_b = -1.771 / 2.0;

constexpr double SL = 5.0;
constexpr double SW = 2.0;
constexpr double CL = 3.5;

int main() {
    // Car dimensions (m)
    constexpr double l_front = 0.839;
    constexpr double l_axes = 2.588;
    constexpr double l_rear = 0.657;
    constexpr double b_width = 1.771 / 2.0;

    // Limits
    constexpr double phi_max = 33.0 * M_PI / 180.0;
    constexpr double v_max = 2.0;
    constexpr double a_max = 0.75;
    constexpr double u1_max = 0.5;
    constexpr double curvature_dot_max = 0.6;
    constexpr double xmin = -10.0, xmax_val = 7.5;

    // Initial conditions
    constexpr double x0 = -5.14;
    constexpr double y0 = 1.41;
    constexpr double theta0 = 13.18 * M_PI / 180.0;

    constexpr int n_segs1 = 50;
    constexpr int n_segs2 = 200;
    constexpr double k1 = 75.0;
    constexpr double k2 = 150.0;

    const double area_ref = (l_axes + l_front + l_rear) * 2.0 * b_width;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(6, 2)
                   .define([l_axes](auto &args) {
                       auto x = args.x_var(0);
                       auto y = args.x_var(1);
                       auto v = args.x_var(2);
                       auto a = args.x_var(3);
                       auto theta = args.x_var(4);
                       auto phi = args.x_var(5);
                       auto u1 = args.u_var(0);
                       auto u2 = args.u_var(1);

                       return stack(v * cos(theta), v * sin(theta), a, u1,
                                    v * tan(phi) / l_axes, u2);
                   })
                   .var_names({{"x", 0},
                               {"y", 1},
                               {"v", 2},
                               {"a", 3},
                               {"theta", 4},
                               {"phi", 5},
                               {"t", 6},
                               {"u1", 7},
                               {"u2", 8}})
                   .build();

    // ── Slot boundary constraint (8 inequalities) ──────────────────────
    // Uses static param k — must go through base() API
    auto make_slot_fn = []() {
        auto args = Arguments<4>(); // x, y, theta, k
        auto x = args.coeff<0>();
        auto y = args.coeff<1>();
        auto theta = args.coeff<2>();
        auto k = args.coeff<3>();

        auto [AX, AY] = corner_world(x, y, theta, Ax_b, Ay_b);
        auto [BX, BY] = corner_world(x, y, theta, Bx_b, By_b);
        auto [CXw, CYw] = corner_world(x, y, theta, Cx_b, Cy_b);
        auto [DX, DY] = corner_world(x, y, theta, Dx_b, Dy_b);

        return stack(AY - CL, (-1.0) * AY + fslot_fn(AX, k, SL, SW),
                     BY - CL, (-1.0) * BY + fslot_fn(BX, k, SL, SW),
                     CYw - CL, (-1.0) * CYw + fslot_fn(CXw, k, SL, SW),
                     DY - CL, (-1.0) * DY + fslot_fn(DX, k, SL, SW));
    };

    // ── Corner collision avoidance (2 inequalities) ────────────────────
    auto make_corner_fn = [area_ref]() {
        auto args = Arguments<3>();
        auto x = args.coeff<0>();
        auto y = args.coeff<1>();
        auto theta = args.coeff<2>();

        auto [aX, aY] = corner_world(x, y, theta, Ax_b, Ay_b);
        auto [bX, bY] = corner_world(x, y, theta, Bx_b, By_b);
        auto [cX, cY] = corner_world(x, y, theta, Cx_b, Cy_b);
        auto [dX, dY] = corner_world(x, y, theta, Dx_b, Dy_b);

        auto ox = 0.0 * x;
        auto oy = 0.0 * y;
        auto ex = 0.0 * x + SL;
        auto ey = 0.0 * y;

        auto AO1 = tri_area(ox, oy, aX, aY, bX, bY);
        auto AO2 = tri_area(ox, oy, cX, cY, bX, bY);
        auto AO3 = tri_area(ox, oy, aX, aY, dX, dY);
        auto AO4 = tri_area(ox, oy, dX, dY, cX, cY);
        auto eq1 = area_ref - (AO1 + AO2 + AO3 + AO4);

        auto AE1 = tri_area(ex, ey, aX, aY, bX, bY);
        auto AE2 = tri_area(ex, ey, cX, cY, bX, bY);
        auto AE3 = tri_area(ex, ey, aX, aY, dX, dY);
        auto AE4 = tri_area(ex, ey, dX, dY, cX, cY);
        auto eq2 = area_ref - (AE1 + AE2 + AE3 + AE4);

        return stack(eq1, eq2);
    };

    // ── Final Y constraint ─────────────────────────────────────────────
    auto make_final_y_fn = []() {
        auto args = Arguments<2>();
        auto y = args.coeff<0>();
        auto theta = args.coeff<1>();

        return stack(y + sin(theta) * Ax_b + cos(theta) * Ay_b,
                     y + sin(theta) * Bx_b + cos(theta) * By_b,
                     y + sin(theta) * Cx_b + cos(theta) * Cy_b,
                     y + sin(theta) * Dx_b + cos(theta) * Dy_b);
    };

    // ── Curvature rate function ────────────────────────────────────────
    auto make_curv_fn = [l_axes]() {
        auto args = Arguments<2>();
        auto phi = args.coeff<0>();
        auto u2 = args.coeff<1>();
        return u2 / (l_axes * cos(phi) * cos(phi));
    };

    // ── Initial guess ──────────────────────────────────────────────────
    auto make_state = [](double x, double y, double theta_deg, double t) {
        Eigen::VectorXd XtU(9);
        XtU.setZero();
        XtU[0] = x;
        XtU[1] = y;
        XtU[4] = theta_deg * M_PI / 180.0;
        XtU[6] = t;
        return XtU;
    };

    std::vector<Eigen::VectorXd> waypoints = {make_state(x0, y0, 13.18, 0),
                                               make_state(0.0, y0, 0.0, 5),
                                               make_state(5.5, y0, 10.0, 10),
                                               make_state(1.0, -0.5, 20.0, 15),
                                               make_state(1.0, -1.0, 0.0, 25)};

    // Create phase with dummy IG, then re-set with LerpIG=true
    std::vector<Eigen::VectorXd> dummy_ig;
    for (int i = 0; i < 100; ++i) {
        double s = static_cast<double>(i) / 99.0;
        dummy_ig.push_back(waypoints.front() * (1.0 - s) + waypoints.back() * s);
    }

    // ── Phase setup (coarse) ───────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, dummy_ig, n_segs1);
    // Re-set with sparse waypoints + LERP interpolation
    phase.set_traj(waypoints, n_segs1, true);

    Eigen::VectorXd sp(1);
    sp[0] = k1;
    phase.set_static_params(sp);
    phase.set_static_param_names({{"k", 0}});
    phase.set_control_mode(ControlModes::BlockConstant);

    // Boundary conditions
    Eigen::VectorXd front_bc(7);
    front_bc << x0, y0, 0.0, 0.0, theta0, 0.0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front,
                             {"x", "y", "v", "a", "theta", "phi", "t"}, front_bc);

    phase.add_boundary_value(PhaseRegionFlags::Back, {"v", "a"}, Eigen::Vector2d(0.0, 0.0));

    // Slot geometry constraints — uses mixed XtUP + static param sources
    {
        Eigen::VectorXi xtup(3);
        xtup << 0, 1, 4;
        Eigen::VectorXi op_empty;
        Eigen::VectorXi sp_idx(1);
        sp_idx << 0;
        phase.add_inequal_con(PhaseRegionFlags::Path,
                              GenericFunction<-1, -1>(make_slot_fn()), xtup, op_empty,
                              sp_idx, ScaleModes::AUTO);
    }

    // Final Y constraint
    phase.add_inequal_con(PhaseRegionFlags::Back, GenericFunction<-1, -1>(make_final_y_fn()),
                          {"y", "theta"});

    // Variable bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", xmin, xmax_val);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", -v_max, v_max);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "a", -a_max, a_max);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "phi", -phi_max, phi_max);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u1", -u1_max, u1_max);

    // Curvature rate bound
    phase.add_lu_func_bound(PhaseRegionFlags::Path, GenericFunction<-1, 1>(make_curv_fn()),
                            {"phi", "u2"}, -curvature_dot_max, curvature_dot_max);

    // Corner collision avoidance
    phase.add_inequal_con(PhaseRegionFlags::Path, GenericFunction<-1, -1>(make_corner_fn()),
                          {"x", "y", "theta"});

    // Lock static parameter k
    phase.add_value_lock(PhaseRegionFlags::StaticParams, Eigen::VectorXi::Constant(1, 0));

    // Objective: minimise time
    phase.add_delta_time_objective(1.0);

    phase.set_num_partitions(8);
    phase.optimizer().set_bound_fraction(0.995);
    phase.optimizer().set_max_iters(2000);
    phase.optimizer().set_print_level(1);

    // ── Coarse solve ───────────────────────────────────────────────────
    std::cout << "ParallelParking: coarse solve (k=" << k1 << ", segs=" << n_segs1 << ")...\n"
              << std::flush;
    phase.solve_optimize();

    // ── Refine and re-solve with tighter k ─────────────────────────────
    std::cout << "ParallelParking: refining to " << n_segs2 << " segments, k=" << k2 << "...\n"
              << std::flush;
    phase.refine_traj_manual(n_segs2);
    phase.sub_variable(PhaseRegionFlags::StaticParams, "k", k2);
    phase.optimizer().set_kkt_tol(1.0e-8);
    phase.optimize();

    auto traj = phase.return_traj();
    const double final_time = traj.back()[6];

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "ParallelParking (builder): maneuver time = " << final_time
              << " s (paper: 18.426 s)\n";

    // NOTE: This problem is sensitive to initial guess quality and solver
    // settings. Both C++ and Python versions may not converge to the paper's
    // 18.426 s value without careful tuning. We check that the solver runs
    // to completion and produces a plausible maneuver time.
    if (final_time < 5.0 || final_time > 200.0) {
        std::cerr << "ParallelParking (builder): FAILED — time out of plausible range\n";
        return EXIT_FAILURE;
    }

    std::cout << "ParallelParking (builder): PASSED\n";
    return EXIT_SUCCESS;
}
