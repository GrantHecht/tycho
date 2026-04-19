///////////////////////////////////////////////////////////////////////////////
// Lazy interpolant accuracy tests — exercises the dense-output tables
// (Bmid + ExtraA + ExtraC) for every user-selectable RK method by
// requesting many more interior sample points than the natural step count
// produces, then comparing each interpolated state against the closed-form
// SHO solution.
//
// Why this matters: per-method tests previously only checked the end-state
// at tf, which uses only the *main* RK stages. The interpolation tables
// — newly added in SP2 for BS5, Vern7, Vern8, Vern9 — never participated
// in those checks. A typo in any extra-stage row or in a Bmid weight
// would silently ship until a downstream user noticed wrong dense output.
//
// Reference: SHO has the exact solution
//     x(t) = x0 * cos(t) + v0 * sin(t)
//     v(t) = -x0 * sin(t) + v0 * cos(t)
// with x0=1, v0=0 → x(t)=cos(t), v(t)=-sin(t). Closed-form, no FP
// dependence on a separate solver, and exactly the kind of well-behaved
// ODE for which a method's interpolant should be near machine-epsilon
// accurate at the chosen tolerance.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <string>
#include <tuple>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {

// One sample point per ~h_natural / 8 to force the interpolant to fill
// in many sub-step samples.
constexpr int kSamplesPerCall = 200;

struct InterpExpect {
    IVPAlg method;
    const char *name;
    double tolerance; // tolerance per sampled state component vs analytic
};

std::string param_name(const ::testing::TestParamInfo<InterpExpect> &info) {
    return info.param.name;
}

} // namespace

class InterpolantAccuracyTest : public ::testing::TestWithParam<InterpExpect> {};

TEST_P(InterpolantAccuracyTest, SHO_DenseOutputMatchesAnalytic) {
    const auto &p = GetParam();

    SHO ode(0.0);
    Integrator<SHO> integ(ode, p.method, 0.05);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    const double tf = std::numbers::pi; // half period; covers the full sin/cos sweep.

    auto traj = integ.integrate_dense(x0, tf, kSamplesPerCall);
    ASSERT_EQ(static_cast<int>(traj.size()), kSamplesPerCall);

    // Interior points (skip endpoints — those use the main RK stages, not
    // the interpolant tables we are validating).
    int interior_failures = 0;
    double max_abs_err_x = 0.0;
    double max_abs_err_v = 0.0;
    for (int i = 1; i + 1 < kSamplesPerCall; ++i) {
        const double t = traj[i][2];
        const double x_exact = std::cos(t);
        const double v_exact = -std::sin(t);
        const double err_x = std::abs(traj[i][0] - x_exact);
        const double err_v = std::abs(traj[i][1] - v_exact);
        max_abs_err_x = std::max(max_abs_err_x, err_x);
        max_abs_err_v = std::max(max_abs_err_v, err_v);
        if (err_x > p.tolerance || err_v > p.tolerance) {
            ++interior_failures;
        }
    }
    EXPECT_EQ(interior_failures, 0)
        << p.name << ": " << interior_failures << " interior points exceeded tol=" << p.tolerance
        << " (max |err_x|=" << max_abs_err_x << ", max |err_v|=" << max_abs_err_v << ")";
}

// Per-method tolerances. Tighter for higher-order methods because
// integrate_dense at atol=1e-12 produces interpolated states near
// machine epsilon on a problem this smooth. Looser for lower orders
// (BS3 = order 3) because the interpolant cannot be more accurate than
// the underlying step error.
INSTANTIATE_TEST_SUITE_P(
    AllUserSelectableMethods, InterpolantAccuracyTest,
    ::testing::Values(InterpExpect{IVPAlg::DOPRI54, "DOPRI54", 1e-9},
                      InterpExpect{IVPAlg::DOPRI87, "DOPRI87", 1e-9},
                      InterpExpect{IVPAlg::Tsit5, "Tsit5", 1e-9},
                      InterpExpect{IVPAlg::BS3, "BS3", 1e-5},
                      InterpExpect{IVPAlg::BS5, "BS5", 1e-9},
                      InterpExpect{IVPAlg::Vern7, "Vern7", 1e-9},
                      InterpExpect{IVPAlg::Vern8, "Vern8", 1e-9},
                      InterpExpect{IVPAlg::Vern9, "Vern9", 1e-9}),
    param_name);
