///////////////////////////////////////////////////////////////////////////////
// Per-method benchmark comparison
//
// Compares all 8 runtime-selectable RK methods (DOPRI54, DOPRI87, Tsit5,
// BS3, BS5, Vern7, Vern8, Vern9) on a Kepler two-body LEO propagation
// (quarter period) at three tolerance tiers (1e-6, 1e-9, 1e-12).
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>
#include <tycho/astro.h>
#include <tycho/integrators.h>

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

void BM_Kepler_Method(benchmark::State &state, tycho::IVPAlg alg, double tol) {
    Kepler kep(MU_EARTH);
    const double tf = leo_period() / 4.0;
    for (auto _ : state) {
        Integrator<Kepler> integ(kep, alg, 10.0);
        integ.set_abs_tol(tol);
        integ.set_rel_tol(tol);
        auto xf = integ.integrate(leo_x0(), tf);
        benchmark::DoNotOptimize(xf);
    }
}

} // namespace

#define REGISTER(name, alg, tol_tag, tol_val)                                            \
    BENCHMARK_CAPTURE(BM_Kepler_Method, name##_##tol_tag, tycho::IVPAlg::alg, tol_val)

// Low-tolerance tier (1e-6) — exercises all methods including lowest-order BS3.
REGISTER(DOPRI54, DOPRI54, tol1e6, 1e-6);
REGISTER(DOPRI87, DOPRI87, tol1e6, 1e-6);
REGISTER(Tsit5,   Tsit5,   tol1e6, 1e-6);
REGISTER(BS3,     BS3,     tol1e6, 1e-6);
REGISTER(BS5,     BS5,     tol1e6, 1e-6);
REGISTER(Vern7,   Vern7,   tol1e6, 1e-6);
REGISTER(Vern8,   Vern8,   tol1e6, 1e-6);
REGISTER(Vern9,   Vern9,   tol1e6, 1e-6);

// Mid-tolerance tier (1e-9) — excludes BS3 (order 3 saturates at ~1e-6 practical).
REGISTER(DOPRI54, DOPRI54, tol1e9, 1e-9);
REGISTER(DOPRI87, DOPRI87, tol1e9, 1e-9);
REGISTER(Tsit5,   Tsit5,   tol1e9, 1e-9);
REGISTER(BS5,     BS5,     tol1e9, 1e-9);
REGISTER(Vern7,   Vern7,   tol1e9, 1e-9);
REGISTER(Vern8,   Vern8,   tol1e9, 1e-9);
REGISTER(Vern9,   Vern9,   tol1e9, 1e-9);

// High-tolerance tier (1e-12) — excludes BS3, BS5 (orders too low to benefit).
REGISTER(DOPRI54, DOPRI54, tol1e12, 1e-12);
REGISTER(DOPRI87, DOPRI87, tol1e12, 1e-12);
REGISTER(Tsit5,   Tsit5,   tol1e12, 1e-12);
REGISTER(Vern7,   Vern7,   tol1e12, 1e-12);
REGISTER(Vern8,   Vern8,   tol1e12, 1e-12);
REGISTER(Vern9,   Vern9,   tol1e12, 1e-12);

#undef REGISTER
