///////////////////////////////////////////////////////////////////////////////
// STM / shooting-defect benchmarks -- 2026-07 review series PR 8 coverage
// foundation for INTEGRATORS §2.1 ★ / §2.3-2.5 and OC §2.1 ★.
//
// Prior to this file, no benchmark anywhere in bench/cpp/ exercised
// integrate_stm, integrate_stm_parallel, or CentralShootingDefect -- see the
// PR 8 OC/INTEGRATORS fact dossier ("no benchmark in the entire bench/cpp/
// tree exercises integrate_stm, integrate_stm2, integrate_stm_parallel,
// STMDriver, CentralShootingDefect, ..."). These benches exist to gate the
// upcoming win-or-drop perf changes (dense-core swap in
// integrate_stm_parallel, shooting-defect arena allocation, STM-driver
// ping-pong buffers) against a real baseline.
///////////////////////////////////////////////////////////////////////////////

#include "../bench_odes.h"
#include <benchmark/benchmark.h>
#include <tycho/integrators.h>

using namespace tycho::integrators;

///////////////////////////////////////////////////////////////////////////////
// integrate_stm / integrate_stm_parallel -- PolarLTODE (7-state, nonlinear
// rational dynamics; the dossier's own suggested stand-in for a CR3BP-class
// ODE, INTEGRATORS §2.1 ★). Exercises STMDriver's per-step jxall/hxall chain
// products (§2.3) and the dense-core-vs-stm-core main-thread handoff (§2.1).
///////////////////////////////////////////////////////////////////////////////

static void BM_IntegrateSTM_Serial(benchmark::State &state) {
    PolarLTODE ode(0.01);
    Eigen::VectorXd x0(7);
    x0 << 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.3; // [r, theta, vr, vt, t, u, alpha]
    double tf = 2.0;

    for (auto _ : state) {
        Integrator<PolarLTODE> integ(ode, tycho::IVPAlg::DOPRI87, 0.05);
        auto [xf, stm] = integ.integrate_stm(x0, tf);
        benchmark::DoNotOptimize(xf);
        benchmark::DoNotOptimize(stm);
    }
}
BENCHMARK(BM_IntegrateSTM_Serial);

// Single-trajectory integrate_stm_parallel(x0, tf, n_parts) -- INTEGRATORS
// §2.1 ★. The main thread currently re-propagates each segment via
// integrate_stm_core (integrate_dense_core + a discarded calculate_jacobian)
// solely to hand worker i+1 a bit-identical start state, serially performing
// nearly the full STM workload the workers are doing in parallel. This
// benchmark is the only regression gate for that fix (swap to
// integrate_dense_core at the handoff call site) -- run pre-fix to establish
// the (expected-broken) baseline scaling across n_parts, then post-fix to
// confirm the restored speedup.
static void BM_IntegrateSTM_Parallel(benchmark::State &state) {
    PolarLTODE ode(0.01);
    Eigen::VectorXd x0(7);
    x0 << 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.3;
    double tf = 2.0;
    const int n_parts = static_cast<int>(state.range(0));

    for (auto _ : state) {
        Integrator<PolarLTODE> integ(ode, tycho::IVPAlg::DOPRI87, 0.05);
        auto [xf, stm] = integ.integrate_stm_parallel(x0, tf, n_parts);
        benchmark::DoNotOptimize(xf);
        benchmark::DoNotOptimize(stm);
    }
}
BENCHMARK(BM_IntegrateSTM_Parallel)->Arg(2)->Arg(4);

///////////////////////////////////////////////////////////////////////////////
// CentralShootingDefect KKT evaluation -- OC §2.1 ★, the O(segments x PSIOPT
// iterations) hot loop: extract_scalar_inputs/extract_scalar_lmults/
// get_input_states_tfs/get_lmults/compute_all_impl_v each build fresh
// std::vectors per call (shooting_defects.h). Constructed directly (not via
// a full phase transcribe -- transcribe_dynamics() for CentralShooting mode
// only wires the constraint into the indexer; it never evaluates it) so the
// benchmark actually drives the flagged allocation path every iteration --
// mirrors the OptimalControlTest.ShootingHessianSparsityHookDispatch
// construction (test_shooting_hessian_sparsity.cpp).
///////////////////////////////////////////////////////////////////////////////

static void BM_ShootingDefect_Transcribe(benchmark::State &state) {
    constexpr double g = 9.81;
    BrachODE ode(g);
    Integrator<BrachODE> integ(ode, 0.01);
    CentralShootingDefect defect(ode, integ);

    const int ir = defect.input_rows();
    const int or_ = defect.output_rows();

    // Two endpoints from a physically reasonable Brachistochrone guess
    // trajectory (same shape as bench_phases.h's make_brach_phase), packed
    // as the shooting-defect's [x1(xtu), x2(xtu)] input layout (p_vars=0
    // for BrachODE, so no trailing parameter block).
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;
    auto point = [&](double s) {
        Eigen::VectorXd pt(5); // [x, y, v, t, theta]
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        return pt;
    };
    Eigen::VectorXd x(ir);
    x.head(5) = point(0.0);
    x.tail(5) = point(0.5);

    Eigen::VectorXd lm = Eigen::VectorXd::Constant(or_, 0.1);
    Eigen::VectorXd fx(or_);
    Eigen::MatrixXd jx(or_, ir);
    Eigen::VectorXd adjgrad(ir);
    Eigen::MatrixXd adjhess(ir, ir);

    for (auto _ : state) {
        // Hessian accumulation is +=-style (add_hessian_elem); outputs must
        // start zeroed every call, exactly as PSIOPT's KKT fill does.
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();
        adjhess.setZero();
        defect.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, lm);
        benchmark::DoNotOptimize(adjhess);
    }
}
BENCHMARK(BM_ShootingDefect_Transcribe);
