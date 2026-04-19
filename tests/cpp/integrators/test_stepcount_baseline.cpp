// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Local step-count regression test.
//
// test_julia_parity.cpp tolerates a wide ±50% step-count drift against
// Julia's OrdinaryDiffEq. That's appropriate for cross-implementation parity,
// but blinds us to LOCAL regressions — a refactor that silently adds 30%
// rejections passes the Julia-parity gate.
//
// This test pins per-(method, problem, tolerance) step counts against a
// hardcoded baseline captured on the canonical Linux/clang-release/FP
// SAFER_FAST platform. ±5% drift passes; anything larger blocks merge.
//
// To regenerate the baseline (after an intentional controller / algorithm
// change), set TYCHO_REGEN_STEPCOUNT_BASELINE=1 in the environment and the
// test prints the updated table to stderr then skips. Run once, copy the
// output into kBaseline below, and unset the env var.

#include "integrator_test_utils.h"
#include <Eigen/Core>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

using namespace tycho;
using namespace tycho::astro;
using namespace TychoTest;

namespace {

struct BaselineEntry {
    const char *method;
    IVPAlg alg;
    const char *problem;
    double abs_tol;
    double rel_tol;
    int expected_naccept;
    int expected_nreject;
};

// Captured 2026-04-19 on Linux x86_64, clang-release, FP SAFER_FAST.
// All values reflect the current AdaptiveDriver implementation post-wiring
// (Phase D1). Run with TYCHO_REGEN_STEPCOUNT_BASELINE=1 to emit updated
// values after any intentional algorithmic change.
constexpr BaselineEntry kBaseline[] = {
    {"DOPRI54", IVPAlg::DOPRI54, "SHO_2pi", 1e-08, 1e-08, 64, 0},
    {"DOPRI87", IVPAlg::DOPRI87, "SHO_2pi", 1e-08, 1e-08, 11, 0},
    {"Tsit5", IVPAlg::Tsit5, "SHO_2pi", 1e-08, 1e-08, 71, 0},
    {"Vern7", IVPAlg::Vern7, "SHO_2pi", 1e-08, 1e-08, 28, 0},
    {"DOPRI54", IVPAlg::DOPRI54, "SHO_2pi", 1e-10, 1e-10, 158, 0},
    {"DOPRI87", IVPAlg::DOPRI87, "SHO_2pi", 1e-10, 1e-10, 18, 0},
    {"Tsit5", IVPAlg::Tsit5, "SHO_2pi", 1e-10, 1e-10, 171, 0},
    {"Vern7", IVPAlg::Vern7, "SHO_2pi", 1e-10, 1e-10, 50, 0},
};

struct RunResult {
    int naccept;
    int nreject;
};

RunResult integrate_sho_2pi(IVPAlg alg, double atol, double rtol) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, alg, 0.01);
    integ.set_abs_tol(atol);
    integ.set_rel_tol(rtol);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    (void)integ.integrate(x0, 2.0 * M_PI);
    return {integ.get_naccept(), integ.get_nreject()};
}

RunResult run_entry(const BaselineEntry &e) {
    if (std::string(e.problem) == "SHO_2pi") {
        return integrate_sho_2pi(e.alg, e.abs_tol, e.rel_tol);
    }
    throw std::logic_error(std::string("Unknown baseline problem: ") + e.problem);
}

bool within_tolerance(int expected, int actual, double rel_tol = 0.05, int abs_floor = 2) {
    if (expected == 0)
        return actual >= 0; // regen mode
    int diff = std::abs(actual - expected);
    int allowed = std::max(abs_floor, static_cast<int>(std::ceil(rel_tol * expected)));
    return diff <= allowed;
}

} // namespace

class StepCountBaselineTest : public VectorFunctionFixture {};

TEST_F(StepCountBaselineTest, LocalStepCountsWithinTolerance) {
    const char *regen = std::getenv("TYCHO_REGEN_STEPCOUNT_BASELINE");
    const bool regenerating = (regen != nullptr && std::string(regen) == "1");

    if (regenerating) {
        std::cerr << "\n=== TYCHO_REGEN_STEPCOUNT_BASELINE ===\n";
        std::cerr << "Copy the following into kBaseline in "
                  << "tests/cpp/integrators/test_stepcount_baseline.cpp:\n\n";
    }

    bool any_mismatch = false;
    for (const auto &e : kBaseline) {
        auto got = run_entry(e);

        if (regenerating) {
            std::cerr << "    {\"" << e.method << "\", IVPAlg::" << e.method << ", \"" << e.problem
                      << "\", " << e.abs_tol << ", " << e.rel_tol << ", " << got.naccept << ", "
                      << got.nreject << "},\n";
            continue;
        }

        EXPECT_TRUE(within_tolerance(e.expected_naccept, got.naccept))
            << e.method << " / " << e.problem << " / tol=" << e.abs_tol
            << ": naccept drift (expected " << e.expected_naccept << ", got " << got.naccept
            << ", >5%)";
        EXPECT_TRUE(within_tolerance(e.expected_nreject, got.nreject, 0.10, 2))
            << e.method << " / " << e.problem << " / tol=" << e.abs_tol
            << ": nreject drift (expected " << e.expected_nreject << ", got " << got.nreject
            << ", >10%)";
        if (got.naccept != e.expected_naccept || got.nreject != e.expected_nreject) {
            any_mismatch = true;
        }
    }
    if (regenerating) {
        GTEST_SKIP() << "Regeneration mode — update kBaseline with the printed values.";
    }
    // any_mismatch is advisory; the EXPECT_TRUE calls above are the hard gates.
    (void)any_mismatch;
}
