///////////////////////////////////////////////////////////////////////////////
// Kepler IFT-composed Jacobian tests
//
// Verifies kepler_propagate_jacobian<double> against central finite difference
// on five fixtures: LEO circular, MEO eccentric, hyperbolic, multi-period
// (5 LEO orbits), and near-parabolic (a=1e7, e=0.99 — small but well-
// conditioned α exercising the recursion → Taylor crossover region).
///////////////////////////////////////////////////////////////////////////////
#include "astro_test_utils.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <tycho/detail/astro/kepler/kepler_propagator_ift.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace tycho::astro;
using namespace tycho::astro::detail;

namespace {

// Compare analytic Jacobian against central FD on the 7-vector y = (R0, V0, dt).
// Tolerance is checked element-wise as |a - fd| <= rel_tol * max(1, |fd|).
//
// Acceptance criterion #1: the IFT-layer primal output matches propagate_cartesian
// to bit-near tolerance.  Both share the LCD kernel for X*/U_n; they then assemble
// xf via algebraically equivalent Goodyear f-g formulas (codegen vs inline), so any
// disagreement signals a bug in one of the assemblies.
void check_jacobian_fd(const Vector3<double> &R0, const Vector3<double> &V0, double dt, double mu,
                       double rel_tol) {
    KeplerPrimal_VF primal(mu);
    KeplerResidual_VF residual(mu);
    Vector6<double> xf_a;
    Eigen::Matrix<double, 6, 7> jac_a;
    kepler_propagate_jacobian<double>(R0, V0, dt, primal, residual, xf_a, jac_a);

    Vector6<double> rv0;
    rv0.head<3>() = R0;
    rv0.tail<3>() = V0;
    Vector6<double> xf_ref = propagate_cartesian<double>(rv0, dt, mu);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(xf_a[i], xf_ref[i], 1e-13 * std::max(1.0, std::abs(xf_ref[i])))
            << "primal comp " << i;

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
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH, 5e-7);
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

TEST(KeplerIFT_Jacobian, HighEccentricity_e0p999) {
    // Higher-eccentricity ellipse than NearParabolic (e=0.99) — within the
    // domain where the recursion form is well-conditioned but the orbit's
    // periapsis sensitivity to V0 is amplified (1/(1-e) ≈ 1000), making it
    // an effective fence against sign errors in the IFT chain rule that
    // NearParabolic's milder sensitivity might not catch.
    Vector6<double> oe;
    oe << 7000.0, 0.999, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH, 5e-5);
}

TEST(KeplerIFT_Jacobian, HighEccentricity_e0p9999) {
    // Long-period-comet-style fixture: e = 0.9999 with a = 1.5e9 km
    // (periapsis r_p = a·(1-e) = 1.5e5 km, ~24 Earth radii).  This is
    // physical (real long-period comets sit at e > 0.99 with periapsis
    // around 1 AU), but FD-vs-analytic comparison is fundamentally
    // limited at near-parabolic eccentricity: the orbit's specific
    // energy |E| = μ/(2a) shrinks as a grows, so the FD perturbation
    // dE = V·dV must satisfy dE << |E| to keep the perturbed orbit
    // elliptic.  At a = 1.5e9 km, μ_Earth: |E| ≈ 1.3e-4 km²/s²,
    // v_p ≈ 2.3 km/s, dV ≈ cbrt(eps_m)·v_p ≈ 1.4e-5 km/s,
    // dE ≈ 3.2e-5 km²/s² ≪ |E| — FD stays well clear of the
    // parabolic boundary.  Tolerance is wider than NearParabolic's
    // 5e-6 because the analytic Jacobian's 1/(1-e) ≈ 1e4 conditioning
    // amplifier interacts with FD's truncation at this regime.
    Vector6<double> oe;
    oe << 1.5e9, 0.9999, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 1000.0, TychoTest::MU_EARTH, 5e-4);
}

TEST(KeplerIFT_Jacobian, DeepHyperbolic_e3) {
    // e = 3.0 hyperbolic — deeper than the e=1.5 Hyperbolic fixture; tests
    // the Stumpff cosh/sinh recursion at larger sqma·X (well-conditioned but
    // past the e=1.5 regime that the existing tests reach).
    Vector6<double> oe;
    oe << -10000.0, 3.0, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, 5e-6);
}

TEST(KeplerIFT_Jacobian, NegativeDtNearParabolic) {
    // Negative-dt near-parabolic — closes the gap that the test analyzer
    // flagged where U_partials_alpha's Taylor coefficients have an
    // X^{n+1}/(n+1)! factor that is odd in X (and hence in dt for parabolic).
    // A sign error there would slip past positive-dt tests if it cancels
    // with another sign; this test catches it.  Distinct from Commit 4's
    // NegativeDtTrueParabolic which exercises α ≡ 0 exactly.
    Vector6<double> oe;
    oe << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), -200.0, TychoTest::MU_EARTH, 5e-6);
}

TEST(KeplerIFT_Jacobian, AlphaBandRecursionVsTaylor) {
    // Probes the band roughly 1e-12 < |α| < 1e-9 where the IFT layer's
    // U_partials_alpha recursion form (which carries 1/(2α) factors) is
    // numerically suspect but the kernel's Stumpff y→0 Taylor branch
    // (kStumpffTaylorEps = 1e-8) is not yet active.  The existing
    // NearParabolic fixture sits at α ≈ 1e-7; this fixture pushes to
    // α ≈ 1e-11 to fence the recursion-branch precision in the band
    // that matters.  The kIFTAlphaTaylorEps = 1e-8 cutoff in
    // kepler_propagator_ift.h is precisely what makes this fixture
    // converge: at α = 1e-11 < 1e-8, the IFT layer takes the Taylor
    // branch and avoids the catastrophic 1/(2α) blow-up.  Without that
    // cutoff (i.e. if the IFT used the kernel's alpha_tol = 1e-12), the
    // recursion path would hit the suspect band and the FD comparison
    // would diverge.
    //
    // Construction: a = 1e11 km (massive semi-major axis ⇒ α = 1e-11),
    // e = 0.99, periapsis at t=0.  Tolerance is wider than NearParabolic
    // because the FD truncation error at this scale (cbrt_eps · |y|
    // ≈ 0.04 over a 1e11-km orbit) is itself in the 1e-4 regime, well
    // above the analytic Jacobian's actual precision.
    Vector6<double> oe;
    oe << 1.0e11, 0.99, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, 1e-4);
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
// analytic Jac itself is only ~5e-6 accurate, and a larger step is required to
// keep FD noise below the per-row scale.  For MultiPeriod the
// Hessian magnitude is O(1e3-1e4) so a smaller step would pick up truncation;
// cbrt_eps is ideal there.
//
// Solution: accept fd_eps_scale as a per-fixture parameter (1e-3 floor for
// NearParabolic, cbrt(eps_machine) ~ 6e-6 elsewhere).
void check_adjoint_hessian_fd(const Vector3<double> &R0, const Vector3<double> &V0, double dt,
                              double mu, const Vector6<double> &lm, double rel_tol,
                              double fd_eps_scale = 6.0554544523933395e-6) {
    KeplerPrimal_VF primal(mu);
    KeplerResidual_VF residual(mu);
    Vector6<double> xf_a;
    Eigen::Matrix<double, 6, 7> jac_a;
    Vector7<double> grad_a;
    Eigen::Matrix<double, 7, 7> hess_a;
    kepler_propagate_adjoint_hessian<double>(R0, V0, dt, primal, residual, lm, xf_a, jac_a, grad_a,
                                             hess_a);

    // Acceptance #1: total xf and jac match the Jacobian-only path bit-for-bit
    // (within 1e-12 rel — reflects only differences in expression order,
    // not algorithmic divergence).
    {
        Vector6<double> xf_j;
        Eigen::Matrix<double, 6, 7> jac_j;
        kepler_propagate_jacobian<double>(R0, V0, dt, primal, residual, xf_j, jac_j);
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
        kepler_propagate_jacobian<double>(y.head<3>(), y.segment<3>(3), y[6], primal, residual,
                                          xf_, jac_);
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
    // Jacobian on this fixture is itself only ~5e-6 accurate, and FD-of-Jacobian
    // amplifies that floor by 1/eps.  A larger step trades a small amount of
    // truncation error for a big reduction in noise floor.
    Vector6<double> oe;
    oe << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 405, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, lm, 5e-5,
                             /*fd_eps_scale=*/1e-3);
}

///////////////////////////////////////////////////////////////////////////////
// Negative-dt FD Jacobian: backward propagation is a routine PSIOPT pattern.
// The column-6 (∂xf/∂dt) sign is the most likely place a sign bug hides.
///////////////////////////////////////////////////////////////////////////////
TEST(KeplerIFT_Jacobian, NegativeDtLEO) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), -300.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, NegativeDtHyperbolic) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), -100.0, TychoTest::MU_EARTH, 5e-7);
}

///////////////////////////////////////////////////////////////////////////////
// SuperScalar IFT FD checks
//
// is_vectorizable=true on KeplerPropagator routes through Eigen::Array<double,W,1>
// SS Scalar.  These tests exercise the SS Jacobian and adjoint-Hessian paths via
// the public KeplerPropagator VF API and per-lane FD-cross-check against the
// scalar IFT path on the same single-lane state.
///////////////////////////////////////////////////////////////////////////////
namespace {

void check_ss_jacobian_per_lane(const Vector6<double> rvs[4], const double dts[4], double mu,
                                double rel_tol) {
    using SS = Eigen::Array<double, 4, 1>;

    // Pack lanes
    Eigen::Matrix<SS, 7, 1> x_ss;
    for (int i = 0; i < 3; ++i) {
        SS Ri, Vi;
        for (int lane = 0; lane < 4; ++lane) {
            Ri[lane] = rvs[lane][i];
            Vi[lane] = rvs[lane][i + 3];
        }
        x_ss[i] = Ri;
        x_ss[i + 3] = Vi;
    }
    {
        SS dt_ss;
        for (int lane = 0; lane < 4; ++lane)
            dt_ss[lane] = dts[lane];
        x_ss[6] = dt_ss;
    }

    // SS analytic Jacobian via the public VF.
    KeplerPropagator vf{mu};
    Eigen::Matrix<SS, 6, 1> xf_ss;
    Eigen::Matrix<SS, 6, 7> jac_ss;
    vf.compute_jacobian_impl(x_ss, xf_ss, jac_ss);

    // Scalar reference per lane via the same VF.
    for (int lane = 0; lane < 4; ++lane) {
        Eigen::Matrix<double, 7, 1> x_d;
        x_d.head<3>() = rvs[lane].head<3>();
        x_d.segment<3>(3) = rvs[lane].tail<3>();
        x_d[6] = dts[lane];

        Eigen::Matrix<double, 6, 1> xf_d;
        Eigen::Matrix<double, 6, 7> jac_d;
        vf.compute_jacobian_impl(x_d, xf_d, jac_d);

        for (int r = 0; r < 6; ++r) {
            EXPECT_NEAR(xf_ss[r][lane], xf_d[r],
                        rel_tol * std::max(1.0, std::abs(xf_d[r])))
                << "lane " << lane << " xf row " << r;
            for (int c = 0; c < 7; ++c) {
                EXPECT_NEAR(jac_ss(r, c)[lane], jac_d(r, c),
                            rel_tol * std::max(1.0, std::abs(jac_d(r, c))))
                    << "lane " << lane << " jac (" << r << "," << c << ")";
            }
        }
    }
}

void check_ss_adjoint_hessian_per_lane(const Vector6<double> rvs[4], const double dts[4], double mu,
                                       const Vector6<double> &lm_d, double rel_tol) {
    using SS = Eigen::Array<double, 4, 1>;

    Eigen::Matrix<SS, 7, 1> x_ss;
    for (int i = 0; i < 3; ++i) {
        SS Ri, Vi;
        for (int lane = 0; lane < 4; ++lane) {
            Ri[lane] = rvs[lane][i];
            Vi[lane] = rvs[lane][i + 3];
        }
        x_ss[i] = Ri;
        x_ss[i + 3] = Vi;
    }
    {
        SS dt_ss;
        for (int lane = 0; lane < 4; ++lane)
            dt_ss[lane] = dts[lane];
        x_ss[6] = dt_ss;
    }

    Eigen::Matrix<SS, 6, 1> lm_ss;
    for (int i = 0; i < 6; ++i)
        lm_ss[i] = SS::Constant(lm_d[i]);

    KeplerPropagator vf{mu};
    Eigen::Matrix<SS, 6, 1> xf_ss;
    Eigen::Matrix<SS, 6, 7> jac_ss;
    Eigen::Matrix<SS, 7, 1> grad_ss;
    Eigen::Matrix<SS, 7, 7> hess_ss;
    vf.compute_jacobian_adjointgradient_adjointhessian_impl(x_ss, xf_ss, jac_ss, grad_ss, hess_ss,
                                                            lm_ss);

    for (int lane = 0; lane < 4; ++lane) {
        Eigen::Matrix<double, 7, 1> x_d;
        x_d.head<3>() = rvs[lane].head<3>();
        x_d.segment<3>(3) = rvs[lane].tail<3>();
        x_d[6] = dts[lane];

        Eigen::Matrix<double, 6, 1> xf_d;
        Eigen::Matrix<double, 6, 7> jac_d;
        Eigen::Matrix<double, 7, 1> grad_d;
        Eigen::Matrix<double, 7, 7> hess_d;
        vf.compute_jacobian_adjointgradient_adjointhessian_impl(x_d, xf_d, jac_d, grad_d, hess_d,
                                                                lm_d);

        for (int i = 0; i < 7; ++i) {
            EXPECT_NEAR(grad_ss[i][lane], grad_d[i],
                        rel_tol * std::max(1.0, std::abs(grad_d[i])))
                << "lane " << lane << " grad " << i;
        }
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                EXPECT_NEAR(hess_ss(r, c)[lane], hess_d(r, c),
                            rel_tol * std::max(1.0, std::abs(hess_d(r, c))))
                    << "lane " << lane << " hess (" << r << "," << c << ")";
            }
        }
    }
}

} // namespace

// SS-vs-scalar tolerance: the SIMD path re-orders sums vs scalar (FMA grouping
// across lane components), producing sub-ULP differences in the kernel's X*/U_n
// output.  Through the IFT chain rule (1/r divisions, U-recursion compositions),
// these amplify to O(1e-9) absolute in the Hessian.  This is the natural
// precision floor for SS-vs-scalar parity, comparable to the FD-vs-analytic
// floor on the scalar tests (~5e-6 with row-norm scaling).
TEST(KeplerIFT_Jacobian, SimdMixedRegimes) {
    // Pack four orbit regimes into one SS call: LEO / MEO / Hyperbolic / NearParabolic.
    // The kernel takes the per_lane fallback path (mixed orbit types) and the IFT
    // layer's <Scalar=SS> instantiation must agree with the scalar IFT per lane.
    auto rv0 = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    Vector6<double> oe1, oe2, oe3;
    oe1 << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oe2 << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oe3 << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    Vector6<double> rvs[4] = {
        rv0,
        classic_to_cartesian<double>(oe1, TychoTest::MU_EARTH),
        classic_to_cartesian<double>(oe2, TychoTest::MU_EARTH),
        classic_to_cartesian<double>(oe3, TychoTest::MU_EARTH),
    };
    const double dts[4] = {300.0, 600.0, 100.0, 200.0};
    check_ss_jacobian_per_lane(rvs, dts, TychoTest::MU_EARTH, 1e-9);
}

TEST(KeplerIFT_Jacobian, SimdUniformElliptic) {
    // All elliptic + all nonzero dt → routes to kepler_lcd_iterate_simd_ellipse.
    Vector6<double> oes[4];
    oes[0] << 7000.0, 0.01, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oes[1] << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    oes[2] << 9000.0, 0.2, 0.3, 0.4, 0.5, 0.6;
    oes[3] << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    Vector6<double> rvs[4];
    for (int lane = 0; lane < 4; ++lane)
        rvs[lane] = classic_to_cartesian<double>(oes[lane], TychoTest::MU_EARTH);
    const double dts[4] = {100.0, 600.0, 250.0, 200.0};
    check_ss_jacobian_per_lane(rvs, dts, TychoTest::MU_EARTH, 1e-9);
}

TEST(KeplerIFT_Hessian, SimdMixedRegimes) {
    auto rv0 = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    Vector6<double> oe1, oe2, oe3;
    oe1 << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oe2 << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oe3 << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    Vector6<double> rvs[4] = {
        rv0,
        classic_to_cartesian<double>(oe1, TychoTest::MU_EARTH),
        classic_to_cartesian<double>(oe2, TychoTest::MU_EARTH),
        classic_to_cartesian<double>(oe3, TychoTest::MU_EARTH),
    };
    const double dts[4] = {300.0, 600.0, 100.0, 200.0};
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 501, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_ss_adjoint_hessian_per_lane(rvs, dts, TychoTest::MU_EARTH, lm, 1e-9);
}

TEST(KeplerIFT_Hessian, SimdUniformElliptic) {
    Vector6<double> oes[4];
    oes[0] << 7000.0, 0.01, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oes[1] << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    oes[2] << 9000.0, 0.2, 0.3, 0.4, 0.5, 0.6;
    oes[3] << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    Vector6<double> rvs[4];
    for (int lane = 0; lane < 4; ++lane)
        rvs[lane] = classic_to_cartesian<double>(oes[lane], TychoTest::MU_EARTH);
    const double dts[4] = {100.0, 600.0, 250.0, 200.0};
    Eigen::VectorXd lm_d = TychoTest::deterministic_random_vector(6, 502, -1.0, 1.0);
    Vector6<double> lm = lm_d;
    check_ss_adjoint_hessian_per_lane(rvs, dts, TychoTest::MU_EARTH, lm, 1e-9);
}

///////////////////////////////////////////////////////////////////////////////
// NaN-poisoning end-to-end: PSIOPT relies on the IFT layer's bailout NaN
// signal to detect failed steps.  Verify that on kernel non-convergence, every
// entry of xf, jac, adjgrad, and adjhess is non-finite — not just a subset.
///////////////////////////////////////////////////////////////////////////////
TEST(KeplerIFT, NaNPoisoningEndToEnd) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    const double nan_dt = std::numeric_limits<double>::quiet_NaN();
    const Vector3<double> R0 = rv.head<3>();
    const Vector3<double> V0 = rv.tail<3>();

    Vector6<double> xf;
    kepler_propagate<double>(R0, V0, nan_dt, TychoTest::MU_EARTH, xf);
    for (int i = 0; i < 6; ++i)
        EXPECT_FALSE(std::isfinite(xf[i])) << "primal xf[" << i << "]";

    KeplerPrimal_VF primal(TychoTest::MU_EARTH);
    KeplerResidual_VF residual(TychoTest::MU_EARTH);

    Vector6<double> xf_j;
    Eigen::Matrix<double, 6, 7> jac;
    kepler_propagate_jacobian<double>(R0, V0, nan_dt, primal, residual, xf_j, jac);
    for (int i = 0; i < 6; ++i)
        EXPECT_FALSE(std::isfinite(xf_j[i])) << "jac-xf[" << i << "]";
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 7; ++c)
            EXPECT_FALSE(std::isfinite(jac(r, c))) << "jac(" << r << "," << c << ")";

    Vector6<double> xf_h;
    Eigen::Matrix<double, 6, 7> jac_h;
    Vector7<double> grad;
    Eigen::Matrix<double, 7, 7> hess;
    Vector6<double> lm = TychoTest::deterministic_random_vector(6, 601, -1.0, 1.0);
    kepler_propagate_adjoint_hessian<double>(R0, V0, nan_dt, primal, residual, lm, xf_h, jac_h,
                                             grad, hess);
    for (int i = 0; i < 6; ++i)
        EXPECT_FALSE(std::isfinite(xf_h[i])) << "hess-xf[" << i << "]";
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 7; ++c)
            EXPECT_FALSE(std::isfinite(jac_h(r, c))) << "hess-jac(" << r << "," << c << ")";
    for (int i = 0; i < 7; ++i)
        EXPECT_FALSE(std::isfinite(grad[i])) << "adjgrad[" << i << "]";
    for (int r = 0; r < 7; ++r)
        for (int c = 0; c < 7; ++c)
            EXPECT_FALSE(std::isfinite(hess(r, c))) << "adjhess(" << r << "," << c << ")";
}

///////////////////////////////////////////////////////////////////////////////
// True-parabolic orbit (alpha == 0 exactly).  Drives the |alpha| <= alpha_tol
// Taylor branches in U_partials_alpha and compute_U_second_partials, which the
// existing NearParabolic fixtures (alpha ~ 1e-7) do not exercise.
///////////////////////////////////////////////////////////////////////////////
TEST(KeplerIFT_Jacobian, TrueParabolic) {
    // Periapsis radius / parabolic-velocity construction yields alpha = 0 exactly.
    // FD step bumped to 1e-3: parabolic conditioning amplifies analytic-Jac round-off,
    // matching the NearParabolic / Hessian convention.
    const double mu = TychoTest::MU_EARTH;
    const double r0 = 7000.0;
    Vector3<double> R0(r0, 0.0, 0.0);
    Vector3<double> V0(0.0, std::sqrt(2.0 * mu / r0), 0.0);
    check_jacobian_fd(R0, V0, 100.0, mu, 5e-6);
}

TEST(KeplerIFT_Jacobian, NegativeDtTrueParabolic) {
    // Closes a coverage gap flagged by the multi-agent review: the parabolic
    // initial-X seed in compute_universal_functions's |α| ≤ alpha_tol branch
    // uses the closed-form atan/cbrt/tan composition that depends on sgn(dt)
    // through atan(3·sqrt(mu/p^3)·dt).  A sign error in the seed (or in any
    // odd-in-X term in the IFT layer's α=0 Taylor partials, e.g.
    //   ∂U_n/∂α ~ -X^{n+2}/(n+2)!  )
    // would slip past the positive-dt fixture if it cancels with another
    // sign.  FD-vs-analytic at dt < 0 catches that class.
    const double mu = TychoTest::MU_EARTH;
    const double r0 = 7000.0;
    Vector3<double> R0(r0, 0.0, 0.0);
    Vector3<double> V0(0.0, std::sqrt(2.0 * mu / r0), 0.0);
    check_jacobian_fd(R0, V0, -100.0, mu, 5e-6);
}

TEST(KeplerIFT_Hessian, TrueParabolicWellFormed) {
    // FD-vs-analytic comparison on a true-parabolic orbit is dominated by the
    // third-derivative magnitude of the propagation (which blows up at α=0,
    // making any FD step size simultaneously truncation- and noise-limited).
    // Instead, verify the analytic Hessian is well-formed: finite, symmetric,
    // and consistent with the Jacobian-only path.  This exercises the Taylor
    // branch in compute_U_second_partials (|α| ≤ alpha_tol) which the existing
    // NearParabolic fixture (α ≈ 1e-7) does not reach.
    const double mu = TychoTest::MU_EARTH;
    const double r0 = 7000.0;
    Vector3<double> R0(r0, 0.0, 0.0);
    Vector3<double> V0(0.0, std::sqrt(2.0 * mu / r0), 0.0);
    Vector6<double> lm = TychoTest::deterministic_random_vector(6, 406, -1.0, 1.0);

    KeplerPrimal_VF primal(mu);
    KeplerResidual_VF residual(mu);

    Vector6<double> xf;
    Eigen::Matrix<double, 6, 7> jac;
    Vector7<double> grad;
    Eigen::Matrix<double, 7, 7> hess;
    kepler_propagate_adjoint_hessian<double>(R0, V0, 100.0, primal, residual, lm, xf, jac, grad,
                                             hess);

    for (int i = 0; i < 6; ++i)
        EXPECT_TRUE(std::isfinite(xf[i]));
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 7; ++c)
            EXPECT_TRUE(std::isfinite(jac(r, c)));
    for (int i = 0; i < 7; ++i)
        EXPECT_TRUE(std::isfinite(grad[i]));
    for (int r = 0; r < 7; ++r)
        for (int c = 0; c < 7; ++c)
            EXPECT_TRUE(std::isfinite(hess(r, c))) << "hess(" << r << "," << c << ")";

    // Symmetry check: the Taylor-branch second partials are symmetric in (X, α);
    // a sign or index error in compute_U_second_partials would surface here.
    for (int r = 0; r < 7; ++r)
        for (int c = r + 1; c < 7; ++c)
            EXPECT_NEAR(hess(r, c), hess(c, r),
                        1e-12 * std::max(1.0, std::abs(hess(r, c))))
                << "asymmetry at (" << r << "," << c << ")";

    // Cross-check primal + Jacobian against the Jacobian-only path (acceptance #1
    // from check_adjoint_hessian_fd).  This is independent of FD step choice and
    // catches divergence between the two IFT routes.
    Vector6<double> xf_j;
    Eigen::Matrix<double, 6, 7> jac_j;
    kepler_propagate_jacobian<double>(R0, V0, 100.0, primal, residual, xf_j, jac_j);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(xf[i], xf_j[i], 1e-12 * std::max(1.0, std::abs(xf_j[i])));
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 7; ++c)
            EXPECT_NEAR(jac(r, c), jac_j(r, c),
                        1e-12 * std::max(1.0, std::abs(jac_j(r, c))));

    // Adjoint gradient must equal jacᵀ·λ.
    const Vector7<double> jac_t_lm = jac.transpose() * lm;
    for (int i = 0; i < 7; ++i)
        EXPECT_NEAR(grad[i], jac_t_lm[i], 1e-13 * std::max(1.0, std::abs(jac_t_lm[i])));
}
