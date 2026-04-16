#include <tycho/tycho.h>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

// Earth-Moon system constants
static constexpr double MuEarth = 3.986004418e14;
static constexpr double MuMoon = 4.9048695e12;
static const double mu = MuMoon / (MuEarth + MuMoon);
static constexpr double LD = 384400000.0; // Earth-Moon distance (m)

static Eigen::Vector3d normalize_vec(const Eigen::Vector3d &v) {
    return v / v.norm();
}

// Co-linear Lagrange point solver (L1 with sign=-1, L2 with sign=+1).
// Iterates the standard CR3BP fixed-point relation; sign factors out the
// only difference between the two cases.
static Eigen::Vector3d compute_lagrange_colinear(int sign, double mu_val) {
    double gamma0 = std::pow((mu_val * (1.0 - mu_val)) / 3.0, 1.0 / 3.0);
    double guess = gamma0 + 1.0;
    while (std::abs(guess - gamma0) > 1.0e-14) {
        gamma0 = guess;
        guess = std::pow(
            (mu_val * (gamma0 + sign) * (gamma0 + sign)) /
                (3.0 - 2.0 * mu_val + sign * gamma0 * (3.0 - mu_val + sign * gamma0)),
            1.0 / 3.0);
    }
    return Eigen::Vector3d(1.0 - mu_val + sign * guess, 0.0, 0.0);
}

static void cr3bp_accel_partials(const Eigen::Vector3d &pos, double mu_val,
                                  double &Oxx, double &Oyy, double &Ozz) {
    double x = pos[0], y = pos[1], z = pos[2];
    double r1 = std::sqrt((x + mu_val) * (x + mu_val) + y * y + z * z);
    double r2 = std::sqrt((x - 1.0 + mu_val) * (x - 1.0 + mu_val) + y * y + z * z);

    double r1_3 = r1 * r1 * r1;
    double r2_3 = r2 * r2 * r2;
    double r1_5 = r1_3 * r1 * r1;
    double r2_5 = r2_3 * r2 * r2;

    // Partial derivatives of the pseudo-potential Omega
    // Omega = 0.5*(x^2 + y^2) + (1-mu)/r1 + mu/r2
    Oxx = 1.0 - (1.0 - mu_val) / r1_3 - mu_val / r2_3 +
          3.0 * (1.0 - mu_val) * (x + mu_val) * (x + mu_val) / r1_5 +
          3.0 * mu_val * (x - 1.0 + mu_val) * (x - 1.0 + mu_val) / r2_5;
    Oyy = 1.0 - (1.0 - mu_val) / r1_3 - mu_val / r2_3 +
          3.0 * (1.0 - mu_val) * y * y / r1_5 +
          3.0 * mu_val * y * y / r2_5;
    Ozz = -(1.0 - mu_val) / r1_3 - mu_val / r2_3 +
          3.0 * (1.0 - mu_val) * z * z / r1_5 +
          3.0 * mu_val * z * z / r2_5;
}

std::vector<Eigen::VectorXd> gen_lissajous(const Eigen::Vector3d &lp,
                                            double xnd, double znd,
                                            double phideg, double psideg,
                                            double nplanrev, int npo) {
    double Oxx, Oyy, Ozz;
    cr3bp_accel_partials(lp, mu, Oxx, Oyy, Ozz);

    constexpr double pi = 3.14159265358979323846;
    double b1 = 2.0 - (Oxx + Oyy) / 2.0;
    double b2sq = -Oxx * Oyy;
    double s = std::sqrt(b1 + std::sqrt(b1 * b1 + b2sq));
    double b3 = (s * s + Oxx) / (2.0 * s);
    double pp = 2.0 * pi / s;
    double nu = std::sqrt(std::abs(Ozz));

    double dtr = pi / 180.0;
    double phi = phideg * dtr;
    double psi = psideg * dtr;
    double ynd = xnd * b3;
    double tt = nplanrev * pp;
    double dt = tt / static_cast<double>(npo - 1);

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(npo);

    for (int i = 0; i < npo; ++i) {
        double ti = static_cast<double>(i) * dt;
        Eigen::VectorXd st = Eigen::VectorXd::Zero(7);
        st[0] = -(ynd / b3) * std::cos(s * ti + phi);
        st[1] = ynd * std::sin(s * ti + phi);
        st[2] = znd * std::sin(nu * ti + psi);
        st[3] = (ynd / b3) * s * std::sin(s * ti + phi);
        st[4] = ynd * s * std::cos(s * ti + phi);
        st[5] = znd * nu * std::cos(nu * ti + psi);
        st[6] = ti;
        st.head<3>() += lp;
        traj.push_back(st);
    }

    return traj;
}

ODE make_cr3bp_ode() {
    auto args = ODEArguments(6, 0, 0);
    auto state = args.head<6>();

    Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
    auto zero_accel = Constant<7, 3>(7, zero3);

    auto dyn = astro::CRTBPDynamics(mu);
    auto ode_expr = GenericFunction<-1, -1>(dyn.eval(stack(state, zero_accel)));

    return ODE(ode_expr, 6, 0)
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_names({{"t", 6}});
}

auto make_jacobi_func() {
    auto args = Arguments<6>();
    auto r = args.head<3>();
    auto v = args.tail<3>();

    Eigen::Vector3d p1loc;
    p1loc << -mu, 0.0, 0.0;
    Eigen::Vector3d p2loc;
    p2loc << 1.0 - mu, 0.0, 0.0;

    auto p1c = Constant<6, 3>(6, p1loc);
    auto p2c = Constant<6, 3>(6, p2loc);

    auto gt1 = (r - p1c).inverse_norm() * (1.0 - mu);
    auto gt2 = (r - p2c).inverse_norm() * mu;

    return r.head<2>().squared_norm() + (gt1 + gt2) * 2.0 - v.squared_norm();
}

std::vector<Eigen::VectorXd> make_orbit(const ODE &ode,
                                         const std::vector<Eigen::VectorXd> &orbit_ig,
                                         double Jconst, int nsegs = 100) {
    auto phase = ode.phase(TranscriptionModes::LGL5, orbit_ig, nsegs);

    // Planar perpendicular crossing
    phase.add_boundary_value(PhaseRegionFlags::Front, 1, 0.0); // y=0
    phase.add_boundary_value(PhaseRegionFlags::Front, 2, 0.0); // z=0
    phase.add_boundary_value(PhaseRegionFlags::Front, 3, 0.0); // vx=0
    phase.add_boundary_value(PhaseRegionFlags::Front, 5, 0.0); // vz=0
    phase.add_boundary_value(PhaseRegionFlags::Front, 6, 0.0); // t=0
    phase.add_boundary_value(PhaseRegionFlags::Back, 1, 0.0);  // y=0
    phase.add_boundary_value(PhaseRegionFlags::Back, 3, 0.0);  // vx=0

    // Jacobi constant constraint
    auto jfunc = make_jacobi_func() - Jconst;
    phase.add_equal_con(PhaseRegionFlags::Front, GenericFunction<-1, -1>(jfunc), {"R", "V"});

    phase.optimizer().set_econ_tol(1.0e-12);
    phase.optimizer().set_print_level(1);
    auto flag = phase.solve();
    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        return {}; // caller checks for empty
    }

    return phase.return_traj();
}

using Trajectory = std::vector<Eigen::VectorXd>;

std::vector<Trajectory> get_manifold(const ODE &ode, const Trajectory &orbit_in,
                                     double dx, double dt, int nman = 100,
                                     bool stable = true) {
    auto integ = ode.integrator().method(IVPAlg::DOPRI87).step(0.01).build();
    integ.set_abs_tol(1.0e-13);

    double period = orbit_in.back()[6];
    auto orbit = integ.integrate_dense(orbit_in[0], period, nman);

    Eigen::VectorXd times(orbit.size());
    for (int i = 0; i < static_cast<int>(orbit.size()); ++i) {
        times[i] = orbit[i][6] + period;
    }

    const int ncores = 8;
    auto stm_results = integ.integrate_stm_parallel(orbit, times, ncores);

    std::vector<Eigen::VectorXd> eig_igs;
    eig_igs.reserve(2 * orbit.size());

    for (int i = 0; i < static_cast<int>(stm_results.size()); ++i) {
        auto &[xf, jac] = stm_results[i];

        Eigen::MatrixXd stm = jac.block(0, 0, 6, 6);
        Eigen::EigenSolver<Eigen::MatrixXd> solver(stm);
        auto vals = solver.eigenvalues();
        auto vecs = solver.eigenvectors();

        std::vector<int> idxs(6);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            return std::abs(vals[a]) < std::abs(vals[b]);
        });

        // Monodromy convention: smallest |λ| is the stable direction,
        // largest is the unstable direction.
        int eig_idx = stable ? idxs[0] : idxs[5];
        Eigen::Vector3d vec = vecs.col(eig_idx).head<3>().real();
        vec = normalize_vec(vec);

        Eigen::VectorXd xp = orbit[i];
        xp.head<3>() += vec * dx;
        eig_igs.push_back(xp);

        Eigen::VectorXd xm = orbit[i];
        xm.head<3>() -= vec * dx;
        eig_igs.push_back(xm);
    }

    double prop_dt = stable ? -dt : dt;
    Eigen::VectorXd ts(eig_igs.size());
    for (int i = 0; i < static_cast<int>(eig_igs.size()); ++i) {
        ts[i] = eig_igs[i][6] + prop_dt;
    }

    auto cross_moon_args = Arguments<7>();
    auto cross_moon_func = GenericFunction<-1, 1>(cross_moon_args.coeff<0>() - (1.0 - mu));

    auto cull_args = Arguments<7>();
    Eigen::Vector3d p2_loc;
    p2_loc << 1.0 - mu, 0.0, 0.0;
    auto cull_p2c = Constant<7, 3>(7, p2_loc);
    auto alt = (cull_args.head<3>() - cull_p2c).norm() - 0.015;
    auto y_term = (cull_args.coeff<1>() - 0.15) * (cull_args.coeff<1>() + 0.15);
    auto cull_func = GenericFunction<-1, 1>(alt * y_term);

    using EventPack = typename decltype(integ)::EventPack;
    std::vector<EventPack> events = {
        {cross_moon_func, 0, 1},
        {cull_func, 0, 1}
    };

    auto results = integ.integrate_dense_parallel(eig_igs, ts, events, ncores);

    std::vector<Trajectory> manifolds;
    for (auto &[traj, eventlocs] : results) {
        if (eventlocs[0].size() == 1 && eventlocs[1].empty()) {
            traj.pop_back();
            traj.push_back(eventlocs[0][0]);
            manifolds.push_back(std::move(traj));
        }
    }

    return manifolds;
}

std::pair<Trajectory, Trajectory>
find_closest_connection(const std::vector<Trajectory> &orbs1,
                        const std::vector<Trajectory> &orbs2) {
    double best_dist = std::numeric_limits<double>::max();
    int best_i = 0, best_j = 0;

    for (int i = 0; i < static_cast<int>(orbs1.size()); ++i) {
        for (int j = 0; j < static_cast<int>(orbs2.size()); ++j) {
            double dist = (orbs1[i].back().head<6>() - orbs2[j].back().head<6>()).norm();
            if (dist < best_dist) {
                best_dist = dist;
                best_i = i;
                best_j = j;
            }
        }
    }

    return {orbs1[best_i], orbs2[best_j]};
}

std::pair<Trajectory, Trajectory>
make_heteroclinic(const ODE &ode, const Trajectory &man1, const Trajectory &man2,
                  const Trajectory &l1_orbit, const Trajectory &l2_orbit) {

    auto orbit_tab1 = std::make_shared<LGLInterpTable>(l1_orbit);
    orbit_tab1->make_periodic();

    auto orbit_tab2 = std::make_shared<LGLInterpTable>(l2_orbit);
    orbit_tab2->make_periodic();

    Eigen::VectorXi pos_vars(3);
    pos_vars[0] = 0; pos_vars[1] = 1; pos_vars[2] = 2;
    Eigen::VectorXi vel_vars(3);
    vel_vars[0] = 3; vel_vars[1] = 4; vel_vars[2] = 5;

    auto make_pos_con = [&](const std::shared_ptr<LGLInterpTable> &tab) {
        auto rt_args = Arguments<4>();
        auto R = rt_args.head<3>();
        auto t = rt_args.coeff<3>();
        return GenericFunction<-1, -1>(R - lgl_interp(tab, pos_vars, t));
    };

    auto make_dv_obj = [&](const std::shared_ptr<LGLInterpTable> &tab) {
        auto vt_args = Arguments<4>();
        auto V = vt_args.head<3>();
        auto t = vt_args.coeff<3>();
        return GenericFunction<-1, 1>((V - lgl_interp(tab, vel_vars, t)).squared_norm());
    };

    Eigen::VectorXi pos_t_idx(4);
    pos_t_idx[0] = 0; pos_t_idx[1] = 1; pos_t_idx[2] = 2; pos_t_idx[3] = 6;
    Eigen::VectorXi vel_t_idx(4);
    vel_t_idx[0] = 3; vel_t_idx[1] = 4; vel_t_idx[2] = 5; vel_t_idx[3] = 6;

    Trajectory man1_traj(man1.begin() + 1, man1.end());
    auto phase1 = ode.phase(TranscriptionModes::LGL7, man1_traj, 50);

    double l1_period = l1_orbit.back()[6];
    double l2_period = l2_orbit.back()[6];

    phase1.add_lower_var_bound(PhaseRegionFlags::Front, 6, -l1_period);
    phase1.add_upper_var_bound(PhaseRegionFlags::Front, 6, 2.0 * l1_period);

    phase1.add_equal_con(PhaseRegionFlags::Front, make_pos_con(orbit_tab1), pos_t_idx);
    phase1.add_state_objective(PhaseRegionFlags::Front, make_dv_obj(orbit_tab1), vel_t_idx);

    Trajectory man2_traj(man2.begin(), man2.end() - 1);
    auto phase2 = ode.phase(TranscriptionModes::LGL7, man2_traj, 50);

    phase2.add_equal_con(PhaseRegionFlags::Back, make_pos_con(orbit_tab2), pos_t_idx);
    phase2.add_state_objective(PhaseRegionFlags::Back, make_dv_obj(orbit_tab2), vel_t_idx);

    phase1.add_lower_var_bound(PhaseRegionFlags::Back, 6, -l2_period);
    phase1.add_upper_var_bound(PhaseRegionFlags::Back, 6, 2.0 * l2_period);

    auto ocp = OptimalControlProblem();
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);

    Eigen::VectorXi link_vars(6);
    link_vars[0] = 0; link_vars[1] = 1; link_vars[2] = 2;
    link_vars[3] = 3; link_vars[4] = 4; link_vars[5] = 5;
    ocp.add_forward_link_equal_con(phase1, phase2, link_vars);
    ocp.set_adaptive_mesh(true);

    ocp.optimizer().set_econ_tol(1.0e-9);
    ocp.optimizer().set_opt_ls_mode("L1");
    auto flag = ocp.optimize();
    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        return {}; // caller checks for empty
    }

    auto traj1 = phase1.return_traj();
    auto traj2 = phase2.return_traj();

    double t1_dep = traj1.front()[6];
    auto orb1_state = orbit_tab1->interpolate(t1_dep);
    double dv1 = (traj1.front().segment<3>(3) - orb1_state.segment<3>(3)).norm();

    double t2_arr = traj2.back()[6];
    auto orb2_state = orbit_tab2->interpolate(t2_arr);
    double dv2 = (traj2.back().segment<3>(3) - orb2_state.segment<3>(3)).norm();

    double vstar = std::sqrt((MuEarth + MuMoon) / LD);
    double total_dv = (dv1 + dv2) * vstar;
    std::cout << "  Total DV: " << std::fixed << std::setprecision(4) << total_dv
              << " m/s\n";

    return {traj1, traj2};
}

int main() {
    auto ode = make_cr3bp_ode();

    const double Jconst = 3.15;
    const double dx = 1.0e-5;
    const double dt = 12.0;
    const int nman = 100;
    const int nsegs = 100;

    std::cout << "=== Heteroclinic Connection in Earth-Moon CR3BP ===\n\n";

    auto L1 = compute_lagrange_colinear(-1, mu);
    auto L2 = compute_lagrange_colinear(+1, mu);
    std::cout << "L1 = " << std::fixed << std::setprecision(6) << L1[0] << "\n";
    std::cout << "L2 = " << std::fixed << std::setprecision(6) << L2[0] << "\n\n";

    // Generate Lissajous initial guesses (matching Python: xnd=0.03, znd=0)
    auto L1ig = gen_lissajous(L1, 0.03, 0.0, 180.0, 0.0, 1.0, 100);
    auto L2ig = gen_lissajous(L2, 0.03, 0.0, 0.0, 0.0, 1.0, 100);

    std::cout << "Computing L1 Lyapunov orbit...\n";
    auto L1Orbit = make_orbit(ode, L1ig, Jconst, nsegs);

    if (L1Orbit.empty()) {
        std::cerr << "  Failed to converge L1 orbit\n";
        return 1;
    }
    std::cout << "  L1 orbit: x0=" << std::fixed << std::setprecision(6) << L1Orbit[0][0]
              << ", period=" << std::setprecision(4) << L1Orbit.back()[6] << "\n";

    std::cout << "\nComputing L2 Lyapunov orbit...\n";
    auto L2Orbit = make_orbit(ode, L2ig, Jconst, nsegs);

    if (L2Orbit.empty()) {
        std::cerr << "  Failed to converge L2 orbit\n";
        return 1;
    }
    std::cout << "  L2 orbit: x0=" << std::fixed << std::setprecision(6) << L2Orbit[0][0]
              << ", period=" << std::setprecision(4) << L2Orbit.back()[6] << "\n";

    std::cout << "\nComputing unstable L1 manifold...\n";
    auto unstable_l1 = get_manifold(ode, L1Orbit, dx, dt, nman, false);
    std::cout << "  Unstable L1 manifold: " << unstable_l1.size() << " arms\n";

    std::cout << "Computing stable L2 manifold...\n";
    auto stable_l2 = get_manifold(ode, L2Orbit, dx, dt, nman, true);
    std::cout << "  Stable L2 manifold: " << stable_l2.size() << " arms\n";

    if (unstable_l1.empty() || stable_l2.empty()) {
        std::cerr << "  No valid manifold arms found\n";
        return 1;
    }

    std::cout << "\nFinding closest connection...\n";
    auto [traj1_ig, traj2_ig] = find_closest_connection(unstable_l1, stable_l2);
    double conn_dist = (traj1_ig.back().head<6>() - traj2_ig.back().head<6>()).norm();
    std::cout << "  Connection distance: " << std::scientific << std::setprecision(4)
              << conn_dist << "\n";

    std::reverse(traj2_ig.begin(), traj2_ig.end());

    std::cout << "\nOptimizing heteroclinic connection...\n";
    auto [traj1, traj2] = make_heteroclinic(ode, traj1_ig, traj2_ig, L1Orbit, L2Orbit);

    // ── Verification ───────────────────────────────────────────────────

    double link_err = (traj1.back().head<6>() - traj2.front().head<6>()).norm();

    std::cout << "\n=== Results ===\n";
    std::cout << "  L1 orbit period: " << std::fixed << std::setprecision(4)
              << L1Orbit.back()[6] << "\n";
    std::cout << "  L2 orbit period: " << std::fixed << std::setprecision(4)
              << L2Orbit.back()[6] << "\n";
    std::cout << "  Manifold arms: " << unstable_l1.size() << " unstable L1, "
              << stable_l2.size() << " stable L2\n";
    std::cout << "  Link continuity error: " << std::scientific << std::setprecision(2)
              << link_err << "\n";

    bool ok = link_err < 1.0e-6;
    if (ok) {
        std::cout << "\nHeteroclinic: PASSED\n";
    } else {
        std::cout << "\nHeteroclinic: FAILED (link error too large)\n";
    }

    return ok ? 0 : 1;
}
