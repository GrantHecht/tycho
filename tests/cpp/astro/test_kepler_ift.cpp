///////////////////////////////////////////////////////////////////////////////
// Kepler IFT-composed Jacobian tests
//
// Verifies kepler_propagate_jacobian<double> against central finite difference
// on five fixtures: LEO circular, MEO eccentric, hyperbolic, multi-period
// (5 LEO orbits), and near-parabolic (a=1e7, e=0.99 — the same lane-3 fixture
// used in Task 1 to exercise small but well-conditioned α).
///////////////////////////////////////////////////////////////////////////////
#include "astro_test_utils.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <tycho/detail/astro/kepler_propagator_ift.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace tycho::astro;
using namespace tycho::astro::detail;

namespace {

// Compare analytic Jacobian against central FD on the 7-vector y = (R0, V0, dt).
// Tolerance is checked element-wise as |a - fd| <= rel_tol * max(1, |fd|).
//
// `cross_check_propagate_cartesian` toggles the bit-equivalence check against
// the existing `propagate_cartesian<double>` reference.  That reference is
// known to be broken on hyperbolic orbits (Task 1 / test_kepler_lcd.cpp's
// MatchesPropagateCartesianHyperbolic comment); for those fixtures we instead
// validate the IFT-layer primal via orbit-energy and angular-momentum
// invariants, then rely on the FD Jacobian agreement as the acceptance gate.
void check_jacobian_fd(const Vector3<double> &R0, const Vector3<double> &V0, double dt, double mu,
                       double rel_tol, bool cross_check_propagate_cartesian = true) {
    Vector6<double> xf_a;
    Eigen::Matrix<double, 6, 7> jac_a;
    kepler_propagate_jacobian<double>(R0, V0, dt, mu, xf_a, jac_a);

    Vector6<double> rv0;
    rv0.head<3>() = R0;
    rv0.tail<3>() = V0;
    if (cross_check_propagate_cartesian) {
        // Acceptance criterion #1: primal output bit-equivalent to the reference.
        Vector6<double> xf_ref = propagate_cartesian<double>(rv0, dt, mu);
        for (int i = 0; i < 6; ++i)
            EXPECT_NEAR(xf_a[i], xf_ref[i], 1e-13 * std::max(1.0, std::abs(xf_ref[i])))
                << "primal comp " << i;
    } else {
        // Validate via orbit invariants on hyperbolic fixtures where
        // propagate_cartesian is broken.
        const Vector3<double> R1 = xf_a.head<3>();
        const Vector3<double> V1 = xf_a.tail<3>();
        const double E0 = 0.5 * V0.squaredNorm() - mu / R0.norm();
        const double E1 = 0.5 * V1.squaredNorm() - mu / R1.norm();
        EXPECT_NEAR(E0, E1, 1e-9 * std::abs(E0)) << "energy not conserved";
        const Vector3<double> H0 = R0.cross(V0);
        const Vector3<double> H1 = R1.cross(V1);
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(H0[i], H1[i], 1e-9 * std::max(1.0, std::abs(H0[i])))
                << "angular momentum comp " << i;
    }

    auto eval = [&](const Eigen::Matrix<double, 7, 1> &y) {
        Vector6<double> xf;
        kepler_propagate<double>(y.head<3>(), y.segment<3>(3), y[6], mu, xf);
        return xf;
    };
    Eigen::Matrix<double, 7, 1> y0;
    y0.head<3>() = R0;
    y0.segment<3>(3) = V0;
    y0[6] = dt;

    // Central FD step: eps ≈ eps_machine^(1/3) · max(1, |y|).  For LEO scale
    // (R ~ 7000, V ~ 7) this puts eps in the 4e-3 / 4e-6 range, balancing
    // truncation O(eps²) against the cancellation noise eps_machine·|f|/eps.
    Eigen::Matrix<double, 6, 7> jac_fd;
    constexpr double cbrt_epsm = 6.0554544523933395e-6; // ~ eps_machine^(1/3)
    for (int i = 0; i < 7; ++i) {
        const double eps = cbrt_epsm * std::max(1.0, std::abs(y0[i]));
        Eigen::Matrix<double, 7, 1> yp = y0, ym = y0;
        yp[i] += eps;
        ym[i] -= eps;
        const Vector6<double> fp = eval(yp);
        const Vector6<double> fm = eval(ym);
        jac_fd.col(i) = (fp - fm) / (2.0 * eps);
    }
    // Use row-norm scaling: small in-row entries are dominated by central-FD
    // round-off (ε_machine · |f| / eps ~ 1e-5 km on LEO scale), so they cannot
    // be checked at the per-element absolute level — but they are well-checked
    // relative to the row's overall magnitude.  This is the standard Jacobian-
    // FD comparison rule.
    for (int r = 0; r < 6; ++r) {
        const double row_scale = std::max(1.0, jac_fd.row(r).cwiseAbs().maxCoeff());
        for (int c = 0; c < 7; ++c) {
            EXPECT_NEAR(jac_a(r, c), jac_fd(r, c), rel_tol * row_scale)
                << "row " << r << " col " << c;
        }
    }
}

} // namespace

TEST(KeplerIFT_Jacobian, LEOCircular) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, MEOEccentric) {
    Vector6<double> oe;
    oe << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 600.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, Hyperbolic) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    // The existing propagate_cartesian<double> is broken on this fixture (see
    // test_kepler_lcd.cpp, MatchesPropagateCartesianHyperbolic comment); skip
    // the bit-equivalence cross-check and rely on orbit invariants + FD.
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH, 5e-7,
                      /*cross_check_propagate_cartesian=*/false);
}

TEST(KeplerIFT_Jacobian, MultiPeriod) {
    auto oe = TychoTest::leoClassic();
    const double a = oe[0];
    const double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / TychoTest::MU_EARTH);
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 5.0 * T, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, NearParabolic) {
    // a=1e7 km, e=0.99 ⇒ α = 1/a ≈ 1e-7 (well-conditioned for the iteration kernel
    // but small enough to exercise the recursion → Taylor crossover region; α stays
    // above α_tol=1e-12 here, so the recursion branch is active and the test
    // verifies its numerical conditioning rather than the Taylor fallback).
    Vector6<double> oe;
    oe << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, 5e-6);
}

///////////////////////////////////////////////////////////////////////////////
// IFT-composed adjoint Hessian tests
//
// Verifies kepler_propagate_adjoint_hessian<double> against central FD of the
// adjoint gradient (jacᵀ·λ) on the same five fixtures used for the Jacobian
// tests.  The adjoint Hessian H = ∇²(λᵀ·xf) is symmetric; FD's mirror Hessian
// has comparable noise on either side of the diagonal so we symmetrize for
// the comparison.
///////////////////////////////////////////////////////////////////////////////

namespace {

// Step-size selection notes:
//
// FD here is the y-derivative of an analytic gradient — sensitivity to step
// size depends on (a) the analytic Jacobian's own FP precision (which sets
// the floor on FD noise) and (b) the third-derivative size (which sets the
// truncation error).  These two compete.  For LEO/MEO/Hyp the analytic Jac
// is ~1e-12 accurate, so cbrt(eps_machine) is fine.  For NearParabolic the
// analytic Jac itself is only 5e-6 accurate (per Task 3), and a larger step
// is required to keep FD noise below the per-row scale.  For MultiPeriod the
// Hessian magnitude is O(1e3-1e4) so a smaller step would pick up truncation;
// cbrt_eps is ideal there.
//
// Solution: accept fd_eps_scale as a per-fixture parameter (1e-3 floor for
// NearParabolic, cbrt(eps_machine) ~ 6e-6 elsewhere).
void check_adjoint_hessian_fd(const Vector3<double> &R0, const Vector3<double> &V0, double dt,
                              double mu, const Vector6<double> &lm, double rel_tol,
                              double fd_eps_scale = 6.0554544523933395e-6) {
    Vector6<double> xf_a;
    Eigen::Matrix<double, 6, 7> jac_a;
    Vector7<double> grad_a;
    Eigen::Matrix<double, 7, 7> hess_a;
    kepler_propagate_adjoint_hessian<double>(R0, V0, dt, mu, lm, xf_a, jac_a, grad_a, hess_a);

    // Acceptance #1: total xf and jac match the Jacobian-only path bit-for-bit
    // (within 1e-12 rel — reflects only differences in expression order,
    // not algorithmic divergence).
    {
        Vector6<double> xf_j;
        Eigen::Matrix<double, 6, 7> jac_j;
        kepler_propagate_jacobian<double>(R0, V0, dt, mu, xf_j, jac_j);
        for (int i = 0; i < 6; ++i)
            EXPECT_NEAR(xf_a[i], xf_j[i], 1e-12 * std::max(1.0, std::abs(xf_j[i])))
                << "xf comp " << i;
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 7; ++c)
                EXPECT_NEAR(jac_a(r, c), jac_j(r, c), 1e-12 * std::max(1.0, std::abs(jac_j(r, c))))
                    << "jac (" << r << "," << c << ")";
    }

    // Acceptance #2: adjgrad = jacᵀ·λ exactly (1e-13 tolerance for FP noise).
    {
        const Vector7<double> jac_t_lm = jac_a.transpose() * lm;
        for (int i = 0; i < 7; ++i)
            EXPECT_NEAR(grad_a[i], jac_t_lm[i], 1e-13 * std::max(1.0, std::abs(jac_t_lm[i])))
                << "adjgrad comp " << i;
    }

    // FD-of-adjoint-gradient.  Each column of hess_fd is the FD of grad(y) in
    // direction y_i; symmetrize for comparison.
    auto eval_adjgrad = [&](const Eigen::Matrix<double, 7, 1> &y) {
        Vector6<double> xf_;
        Eigen::Matrix<double, 6, 7> jac_;
        kepler_propagate_jacobian<double>(y.head<3>(), y.segment<3>(3), y[6], mu, xf_, jac_);
        return Vector7<double>(jac_.transpose() * lm);
    };

    Eigen::Matrix<double, 7, 1> y0;
    y0.head<3>() = R0;
    y0.segment<3>(3) = V0;
    y0[6] = dt;

    Eigen::Matrix<double, 7, 7> hess_fd;
    for (int i = 0; i < 7; ++i) {
        const double eps = fd_eps_scale * std::max(1.0, std::abs(y0[i]));
        Eigen::Matrix<double, 7, 1> yp = y0, ym = y0;
        yp[i] += eps;
        ym[i] -= eps;
        const Vector7<double> gp = eval_adjgrad(yp);
        const Vector7<double> gm = eval_adjgrad(ym);
        hess_fd.col(i) = (gp - gm) / (2.0 * eps);
    }
    const Eigen::Matrix<double, 7, 7> hess_fd_sym = 0.5 * (hess_fd + hess_fd.transpose());

    // Row-norm scaling — same convention as the Jacobian FD.  Small in-row
    // entries are dominated by central-FD round-off (eps_machine·|grad|/eps);
    // checking them at the row's overall magnitude keeps the test stable.
    for (int r = 0; r < 7; ++r) {
        const double row_scale = std::max(1.0, hess_fd_sym.row(r).cwiseAbs().maxCoeff());
        for (int c = 0; c < 7; ++c) {
            EXPECT_NEAR(hess_a(r, c), hess_fd_sym(r, c), rel_tol * row_scale)
                << "row " << r << " col " << c;
        }
    }
}

} // namespace

TEST(KeplerIFT_Hessian, LEOCircular) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 401, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, MEOEccentric) {
    Vector6<double> oe;
    oe << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 402, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 600.0, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, Hyperbolic) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 403, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, MultiPeriod) {
    auto oe = TychoTest::leoClassic();
    const double a = oe[0];
    const double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / TychoTest::MU_EARTH);
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 404, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 5.0 * T, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, NearParabolic) {
    // Same fixture as IFT_Jacobian.NearParabolic — a=1e7, e=0.99, dt=200s.
    // FD step bumped to 1e-3 (vs cbrt_eps default ~6e-6) because the analytic
    // Jacobian on this fixture is itself only ~5e-6 accurate (per Task 3),
    // and FD-of-Jacobian amplifies that floor by 1/eps.  A larger step trades
    // a small amount of truncation error for a big reduction in noise floor.
    Vector6<double> oe;
    oe << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 405, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, lm, 5e-5,
                             /*fd_eps_scale=*/1e-3);
}
