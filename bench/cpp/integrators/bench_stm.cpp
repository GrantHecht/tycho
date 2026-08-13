///////////////////////////////////////////////////////////////////////////////
// STM / shooting-defect benchmarks -- 2026-07 review series PR 8 coverage
// foundation for INTEGRATORS §2.1 ★ / §2.3 and OC §2.1 ★.
//
// Prior to this file, no benchmark anywhere in bench/cpp/ exercised
// integrate_stm, integrate_stm2, integrate_stm_parallel, or
// CentralShootingDefect -- see the PR 8 OC/INTEGRATORS fact dossier ("no
// benchmark in the entire bench/cpp/ tree exercises integrate_stm,
// integrate_stm2, integrate_stm_parallel, STMDriver, CentralShootingDefect,
// ..."). These benches exist to gate the upcoming win-or-drop perf changes
// (dense-core swap in integrate_stm_parallel, shooting-defect arena
// allocation, STM-driver ping-pong buffers) against a real baseline.
//
// Coverage map for STMDriver (§2.3), verified by reading stm_driver.h's
// dispatch directly, not inferred:
//   - BM_IntegrateSTM_Serial / BM_IntegrateSTM_Parallel drive integrate_stm /
//     integrate_stm_parallel, which chain only first-order Jacobians via
//     STMDriver::calculate_jacobian(s) -- a jxall-only self-aliasing chain
//     product (jxall.topRows(...) = stepper_jacobian * jxall). Neither path
//     ever calls STMDriver::calculate_jacobian_hessian, so neither touches
//     the adjoint-Hessian hxall self-aliasing chain.
//   - BM_IntegrateSTM2_Serial drives integrate_stm2 (vectorize_batch_calls_
//     forced false so the scalar per-trajectory loop -- and hence
//     STMDriver::calculate_jacobian_hessian -- is guaranteed to run). This is
//     the only entry point that reaches calculate_jacobian_hessian's *both*
//     jxall and hxall self-aliasing chain products (jxall = jxall * jtwist;
//     hxall = jtwist.transpose() * hxall * jtwist) -- the full §2.3 claim.
///////////////////////////////////////////////////////////////////////////////

#include "../bench_odes.h"
#include <benchmark/benchmark.h>
#include <tycho/integrators.h>

using namespace tycho::integrators;

///////////////////////////////////////////////////////////////////////////////
// integrate_stm / integrate_stm_parallel -- PolarLTODE (7-state, nonlinear
// rational dynamics; the dossier's own suggested stand-in for a CR3BP-class
// ODE, INTEGRATORS §2.1 ★). Exercises STMDriver::calculate_jacobian(s)'s
// per-step jxall-only self-aliasing chain product and the dense-core-vs-
// stm-core main-thread handoff (§2.1). Neither path calls
// STMDriver::calculate_jacobian_hessian -- see BM_IntegrateSTM2_Serial below
// for jxall+hxall adjoint-Hessian chain coverage.
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

// Single-trajectory integrate_stm2(x0s, tfs, lfs) -- INTEGRATORS §2.3. This
// is the only entry point that reaches STMDriver::calculate_jacobian_hessian
// (integrator.h:1690), the adjoint-Hessian chain with *both* self-aliasing
// products: jxall = jxall * jtwist and hxall = jtwist.transpose() * hxall *
// jtwist (stm_driver.h:189/194). integrate_stm2 has no scalar single-
// trajectory overload, so this drives the batch API with a length-1 batch
// and forces vectorize_batch_calls_ = false to guarantee the scalar
// per-trajectory loop (integrator.h:1687-1692) is taken -- and hence that
// calculate_jacobian_hessian, not the SuperScalar-batched
// calculate_jacobians_hessians, is what actually runs. Call shape
// (length-1 batch vectors, ODEState-typed lf seeded across all slots
// including t/u) follows test_stm_batch_adjoint_seed.cpp's
// BatchStm2AdjointSeedMatchesScalar scalar-vs-batch equivalence test.
static void BM_IntegrateSTM2_Serial(benchmark::State &state) {
    using OState = Integrator<PolarLTODE>::IntegRet; // fixed-size (7) ODE state type

    PolarLTODE ode(0.01);
    OState x0;
    x0 << 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.3; // [r, theta, vr, vt, t, u, alpha]
    double tf = 2.0;
    OState lf;
    lf << 0.3, -0.2, 0.5, 0.1, 0.4, -0.3, 0.2; // nonzero across all slots incl. t/u

    std::vector<OState> x0s{x0};
    Eigen::VectorXd tfs(1);
    tfs << tf;
    std::vector<OState> lfs{lf};

    for (auto _ : state) {
        Integrator<PolarLTODE> integ(ode, tycho::IVPAlg::DOPRI87, 0.05);
        integ.vectorize_batch_calls_ = false;
        auto results = integ.integrate_stm2(x0s, tfs, lfs);
        benchmark::DoNotOptimize(results);
    }
}
BENCHMARK(BM_IntegrateSTM2_Serial);

///////////////////////////////////////////////////////////////////////////////
// CentralShootingDefect KKT evaluation -- OC §2.1 ★, the O(segments x InteriorPointSolver
// iterations) hot loop: extract_scalar_inputs/extract_scalar_lmults/
// get_input_states_tfs/get_lmults/compute_all_impl_v each build fresh
// std::vectors per call (shooting_defects.h). Constructed directly (not via
// a full phase transcribe -- transcribe_dynamics() for CentralShooting mode
// only wires the constraint into the indexer; it never evaluates it) so the
// benchmark actually drives the flagged allocation path every iteration --
// mirrors the OptimalControlTest.ShootingHessianSparsityHookDispatch
// construction (test_shooting_hessian_sparsity.cpp).
///////////////////////////////////////////////////////////////////////////////

static void BM_ShootingDefect_Eval(benchmark::State &state) {
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
    constexpr double x0 = 0.0, y0 = 10.0, t0 = 0.0;
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
        // start zeroed every call, exactly as InteriorPointSolver's KKT fill does.
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();
        adjhess.setZero();
        defect.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, lm);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::DoNotOptimize(adjgrad);
        benchmark::DoNotOptimize(adjhess);
    }
}
BENCHMARK(BM_ShootingDefect_Eval);
