///////////////////////////////////////////////////////////////////////////////
// Golden regression data generator
//
// Generates golden binary files for all regression test cases.
// Run once to capture current integrator behavior as the oracle.
///////////////////////////////////////////////////////////////////////////////

#include "regression_utils.h"
#include <filesystem>
#include <iostream>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Case 1: Two-body DOPRI54 — LEO circular orbit, quarter period
///////////////////////////////////////////////////////////////////////////////
void generate_ivp_01() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = leo_circular_x0();
    double tf = leo_period() / 4.0;
    auto xf = integ.integrate(x0, tf);

    std::ofstream f(golden_path("ivp_01_twobody_dopri54.bin"), std::ios::binary);
    write_vector(f, xf);
    std::cout << "  ivp_01: xf = [" << xf.transpose() << "]\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 2: Two-body DOPRI87 — Same orbit, same duration
///////////////////////////////////////////////////////////////////////////////
void generate_ivp_02() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = leo_circular_x0();
    double tf = leo_period() / 4.0;
    auto xf = integ.integrate(x0, tf);

    std::ofstream f(golden_path("ivp_02_twobody_dopri87.bin"), std::ios::binary);
    write_vector(f, xf);
    std::cout << "  ivp_02: xf = [" << xf.transpose() << "]\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 3: CR3BP substitute DOPRI87 — Kepler with mu=1.0 (normalized)
///////////////////////////////////////////////////////////////////////////////
void generate_ivp_03() {
    CR3BP_Substitute ode(MU_CR3BP_SUBSTITUTE);
    Integrator<CR3BP_Substitute> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = cr3bp_l1_x0();
    double tf = 2.0; // ~2 time units
    auto xf = integ.integrate(x0, tf);

    std::ofstream f(golden_path("ivp_03_cr3bp_dopri87.bin"), std::ios::binary);
    write_vector(f, xf);
    std::cout << "  ivp_03: xf = [" << xf.transpose() << "]\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 4: Forced control DOPRI54 — BrachODE with constant control
///////////////////////////////////////////////////////////////////////////////
void generate_ivp_04() {
    BrachODE ode(9.81);
    Eigen::VectorXd u(1);
    u << std::numbers::pi / 4.0;
    Integrator<BrachODE> integ(ode, 0.01, u);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    Eigen::VectorXd x0(5);
    x0 << 0.0, 10.0, 0.1, 0.0, std::numbers::pi / 4.0;
    double tf = 1.0;

    auto xf = integ.integrate(x0, tf);

    std::ofstream f(golden_path("ivp_04_brach_forced.bin"), std::ios::binary);
    write_vector(f, xf);
    std::cout << "  ivp_04: xf = [" << xf.transpose() << "]\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 5: Altitude crossing event DOPRI54 — Eccentric orbit
///////////////////////////////////////////////////////////////////////////////
void generate_ivp_05() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = eccentric_orbit_x0();
    double tf = eccentric_orbit_period(); // Full period

    // Event: radial distance crossing at r = 7000 km
    auto args = Arguments<7>();
    auto r = args.head<3>().norm();
    auto event_func = r + (-7000.0);
    GenericFunction<-1, 1> gf(event_func);
    std::vector<Integrator<Kepler>::EventPack> events;
    events.push_back({gf, 0, 0}); // direction=0 (both), index=0

    auto [xf, eventlocs] = integ.integrate(x0, tf, events);

    // Write final state
    std::ofstream f(golden_path("ivp_05_event_crossing.bin"), std::ios::binary);
    write_vector(f, xf);

    // Write number of event groups then each group. The golden file format
    // predates the std::optional API — only resolved crossings round-trip,
    // and an assert fires if refinement failed (the golden would be
    // non-reproducible otherwise).
    int32_t ngroups = static_cast<int32_t>(eventlocs.size());
    write_int32(f, ngroups);
    for (const auto &group : eventlocs) {
        std::vector<Eigen::VectorXd> dyn_group(group.size());
        for (size_t e = 0; e < group.size(); ++e) {
            if (!group[e].has_value()) {
                throw std::runtime_error(
                    "generate_golden: event refinement failed — cannot serialise a reproducible golden file.");
            }
            dyn_group[e] = *group[e];
        }
        write_vectors(f, dyn_group);
    }

    std::cout << "  ivp_05: xf = [" << xf.transpose() << "]\n";
    std::cout << "          " << ngroups << " event group(s)\n";
    for (int g = 0; g < ngroups; ++g) {
        std::cout << "          group " << g << ": " << eventlocs[g].size() << " events\n";
        for (const auto &ev : eventlocs[g]) {
            if (!ev.has_value()) {
                std::cout << "            (refinement failed)\n";
                continue;
            }
            double r_ev = ev->head(3).norm();
            std::cout << "            t=" << (*ev)[6] << " r=" << r_ev << "\n";
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Case 6: Batch N=8 trajectories DOPRI87 — SuperScalar batch
///////////////////////////////////////////////////////////////////////////////
void generate_ivp_06() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);
    integ.vectorize_batch_calls_ = true;

    constexpr int N = 8;
    using KeplerState = Integrator<Kepler>::IntegRet;
    std::vector<KeplerState> x0s(N);
    Eigen::VectorXd tfs(N);

    // 8 different circular orbits at different altitudes
    for (int i = 0; i < N; ++i) {
        double r0 = 6800.0 + 200.0 * i; // 6800 to 8200 km
        double v0 = std::sqrt(MU_EARTH / r0);
        x0s[i] << r0, 0.0, 0.0, 0.0, v0, 0.0, 0.0;
        double T = 2.0 * std::numbers::pi * std::sqrt(r0 * r0 * r0 / MU_EARTH);
        tfs[i] = T / 4.0; // quarter period
    }

    auto xfs = integ.integrate(x0s, tfs);

    // Convert to VectorXd for serialization
    std::vector<Eigen::VectorXd> xfs_dyn(N);
    for (int i = 0; i < N; ++i) {
        xfs_dyn[i] = xfs[i];
    }

    std::ofstream f(golden_path("ivp_06_batch_dopri87.bin"), std::ios::binary);
    write_vectors(f, xfs_dyn);

    std::cout << "  ivp_06: " << N << " batch trajectories\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "    orbit " << i << ": xf = [" << xfs[i].transpose() << "]\n";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Case 7: Backward integration DOPRI54 — LEO orbit forward then backward
///////////////////////////////////////////////////////////////////////////////
void generate_ivp_07() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = leo_circular_x0();
    double tf_fwd = leo_period() / 4.0;

    // Forward
    auto xf = integ.integrate(x0, tf_fwd);
    // Backward to t=0
    auto xr = integ.integrate(xf, 0.0);

    std::ofstream f(golden_path("ivp_07_backward.bin"), std::ios::binary);
    write_vector(f, xf);
    write_vector(f, xr);

    std::cout << "  ivp_07: forward  xf = [" << xf.transpose() << "]\n";
    std::cout << "          backward xr = [" << xr.transpose() << "]\n";
    std::cout << "          roundtrip error = " << (xr.head(6) - x0.head(6)).norm() << "\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 8: Two-body Jacobian via integrate_stm
///////////////////////////////////////////////////////////////////////////////
void generate_trans_08() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    auto x0 = leo_circular_x0();
    double tf = leo_period() / 4.0;

    auto [xf, jx] = integ.integrate_stm(x0, tf);

    std::ofstream f(golden_path("trans_08_jacobian.bin"), std::ios::binary);
    write_vector(f, xf);
    write_matrix(f, jx);

    std::cout << "  trans_08: Jacobian size = " << jx.rows() << "x" << jx.cols() << "\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 9: Two-body Jacobian+Hessian via integrate_stm2
///////////////////////////////////////////////////////////////////////////////
void generate_trans_09() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);
    integ.vectorize_batch_calls_ = false; // Use scalar path

    auto x0 = leo_circular_x0();
    double tf = leo_period() / 4.0;

    // Deterministic adjoint vector
    using KeplerState = Integrator<Kepler>::IntegRet;
    KeplerState lf;
    auto lf_dyn = deterministic_random_vector(integ.output_rows(), 42, -1.0, 1.0);
    for (int i = 0; i < integ.output_rows(); ++i)
        lf[i] = lf_dyn[i];

    std::vector<KeplerState> x0s = {x0};
    Eigen::VectorXd tfs(1);
    tfs[0] = tf;
    std::vector<KeplerState> lfs = {lf};

    auto results = integ.integrate_stm2(x0s, tfs, lfs);
    auto &[xf, jx, hx] = results[0];

    std::ofstream f(golden_path("trans_09_jacobian_hessian.bin"), std::ios::binary);
    write_vector(f, xf);
    write_matrix(f, jx);
    write_matrix(f, hx);

    std::cout << "  trans_09: Jacobian size = " << jx.rows() << "x" << jx.cols()
              << ", Hessian size = " << hx.rows() << "x" << hx.cols() << "\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 10: Batch Jacobians via batch integrate_stm
///////////////////////////////////////////////////////////////////////////////
void generate_trans_10() {
    CR3BP_Substitute ode(MU_CR3BP_SUBSTITUTE);
    Integrator<CR3BP_Substitute> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);
    integ.vectorize_batch_calls_ = true;

    constexpr int N = 4;
    using State = Integrator<CR3BP_Substitute>::IntegRet;
    std::vector<State> x0s(N);
    Eigen::VectorXd tfs(N);

    auto base_x0 = cr3bp_l1_x0();
    for (int i = 0; i < N; ++i) {
        x0s[i] = base_x0;
        x0s[i][0] += 0.01 * i;          // Slightly different x positions
        x0s[i][4] += 0.02 * (i - 1.5);  // Slightly different vy
        tfs[i] = 0.5 + 0.1 * i;         // Different durations
    }

    auto stm_results = integ.integrate_stm(x0s, tfs);

    std::vector<Eigen::MatrixXd> jxs(N);
    for (int i = 0; i < N; ++i) {
        jxs[i] = std::get<1>(stm_results[i]);
    }

    std::ofstream f(golden_path("trans_10_batch_jacobians.bin"), std::ios::binary);
    write_matrices(f, jxs);

    std::cout << "  trans_10: " << N << " batch Jacobians\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "    traj " << i << ": Jacobian " << jxs[i].rows() << "x" << jxs[i].cols()
                  << "\n";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Case 11: Two-body single-step Jacobian via integrate_stm (DOPRI54)
///////////////////////////////////////////////////////////////////////////////
void generate_trans_11() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);
    integ.adaptive_ = false; // Fixed step for exactly one step

    auto x0 = leo_circular_x0();
    double tf = 10.0; // One step

    auto [xf, jx] = integ.integrate_stm(x0, tf);

    std::ofstream f(golden_path("trans_11_single_step_jac.bin"), std::ios::binary);
    write_vector(f, xf);
    write_matrix(f, jx);

    std::cout << "  trans_11: single-step Jacobian " << jx.rows() << "x" << jx.cols() << "\n";
}

///////////////////////////////////////////////////////////////////////////////
// Case 12: Two-body single-step Jacobian+Hessian via integrate_stm2 (DOPRI87)
///////////////////////////////////////////////////////////////////////////////
void generate_trans_12() {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);
    integ.adaptive_ = false; // Fixed step for exactly one step
    integ.vectorize_batch_calls_ = false;

    auto x0 = leo_circular_x0();
    double tf = 10.0; // One step

    using KeplerState = Integrator<Kepler>::IntegRet;
    KeplerState lf;
    auto lf_dyn = deterministic_random_vector(integ.output_rows(), 99, -1.0, 1.0);
    for (int i = 0; i < integ.output_rows(); ++i)
        lf[i] = lf_dyn[i];

    std::vector<KeplerState> x0s = {x0};
    Eigen::VectorXd tfs(1);
    tfs[0] = tf;
    std::vector<KeplerState> lfs = {lf};

    auto results = integ.integrate_stm2(x0s, tfs, lfs);
    auto &[xf, jx, hx] = results[0];

    std::ofstream f(golden_path("trans_12_single_step_jac_hess.bin"), std::ios::binary);
    write_vector(f, xf);
    write_matrix(f, jx);
    write_matrix(f, hx);

    std::cout << "  trans_12: single-step Jacobian " << jx.rows() << "x" << jx.cols()
              << ", Hessian " << hx.rows() << "x" << hx.cols() << "\n";
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////
int main() {
    BumpAllocator::resize(256, 256);

    // Ensure golden data directory exists
    std::filesystem::create_directories(golden_path("").substr(0, golden_path("").size() - 1));

    std::cout << "Generating golden regression data...\n\n";
    std::cout << "IVP Cases:\n";

    generate_ivp_01();
    generate_ivp_02();
    generate_ivp_03();
    generate_ivp_04();
    generate_ivp_05();
    generate_ivp_06();
    generate_ivp_07();

    std::cout << "\nTranscription Cases:\n";

    generate_trans_08();
    generate_trans_09();
    generate_trans_10();
    generate_trans_11();
    generate_trans_12();

    std::cout << "\nAll golden data generated successfully.\n";
    return 0;
}
