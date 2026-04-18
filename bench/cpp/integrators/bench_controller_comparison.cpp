///////////////////////////////////////////////////////////////////////////////
// Per-controller benchmark comparison
//
// Compares I vs PI vs PID step controllers on a Kepler LEO half-period
// propagation for each of the 8 runtime-selectable methods, at 1e-9/1e-10
// tolerance. Reports per-iteration walltime.
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>
#include <tycho/tycho.h>

#include <cmath>
#include <numbers>

using namespace tycho::integrators;
using tycho::astro::Kepler;

namespace {

constexpr double MU_EARTH = 398600.4418;
constexpr double LEO_R0 = 7000.0;

double leo_period() {
    return 2.0 * std::numbers::pi * std::sqrt(LEO_R0 * LEO_R0 * LEO_R0 / MU_EARTH);
}

Eigen::Matrix<double, 7, 1> leo_x0() {
    Eigen::Matrix<double, 7, 1> x0;
    double v0 = std::sqrt(MU_EARTH / LEO_R0);
    x0 << LEO_R0, 0.0, 0.0, 0.0, v0, 0.0, 0.0;
    return x0;
}

void BM_Controller(benchmark::State &state, tycho::integrators::IVPAlg alg,
                   IVPController ctrl) {
    Kepler kep(MU_EARTH);
    const double tf = leo_period() / 2.0;
    for (auto _ : state) {
        tycho::integrators::Integrator<Kepler> integ(kep, alg, 10.0);
        integ.set_controller(ctrl);
        integ.set_abs_tol(1e-9);
        integ.set_rel_tol(1e-10);
        auto xf = integ.integrate(leo_x0(), tf);
        benchmark::DoNotOptimize(xf);
    }
}

} // namespace

#define REG(name, alg, ctrl)                                                                       \
    BENCHMARK_CAPTURE(BM_Controller, name, tycho::integrators::IVPAlg::alg, IVPController::ctrl)

REG(DOPRI54_I, DOPRI54, I);
REG(DOPRI54_PI, DOPRI54, PI);
REG(DOPRI54_PID, DOPRI54, PID);

REG(DOPRI87_I, DOPRI87, I);
REG(DOPRI87_PI, DOPRI87, PI);

REG(Tsit5_PI, Tsit5, PI);
REG(Tsit5_I, Tsit5, I);
REG(Vern7_PI, Vern7, PI);
REG(Vern7_I, Vern7, I);
REG(Vern8_PI, Vern8, PI);
REG(Vern9_PI, Vern9, PI);

#undef REG
