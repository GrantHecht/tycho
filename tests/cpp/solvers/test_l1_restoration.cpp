///////////////////////////////////////////////////////////////////////////////
// Unit tests for NestedL1Restoration — the nested l1 proximal feasibility
// restoration (second of the feasibility-restoration trio), with elastic-slack
// condensation.
//
// No solver wiring exists for this component yet (see restoration.h /
// l1_restoration.h file docstrings), so every test drives the class directly:
// the closed-form elastic slack initialization, the per-iteration condensation
// outputs (pivots, condensed residuals, proximal Hessian/gradient/objective),
// the elastic step recovery, the fraction-to-boundary caps, and the entry
// guard / diagnostics surface.
//
// The centerpiece is a DENSE ORACLE: the enlarged primal-dual Newton system in
// (dx, dn, dp, dzn, dzp, dy) is assembled and solved with dense Eigen, then the
// condensed system in (dx, dy) is assembled from the component's own pivot /
// condensed-residual / proximal outputs and solved; all six recovered blocks
// must match the enlarged reference to 1e-9. This pins the condensation algebra
// and its sign conventions against a fully independent linear solve.
//
// UNITY RULE: the unity build defeats anonymous namespaces for ODR, so every
// file-local helper here is prefixed L1Resto* to stay globally unique across
// tests/cpp/ (grep-confirmed no other "L1Resto" symbol exists).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/l1_restoration.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>

namespace {

using tycho::solvers::ElasticSlackInit;
using tycho::solvers::kBoundMultResetThreshold;
using tycho::solvers::kNearFeasibleGuardFactor;
using tycho::solvers::kRestoPenaltyParameter;
using tycho::solvers::kRestoProximityWeight;
using tycho::solvers::l1_elastic_slack_init;
using tycho::solvers::NestedL1Restoration;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;
using tycho::solvers::SolverContext;

// Builds a minimal all-zero-dimension SolverContext with the given settings --
// entry_permitted only reads ctx.settings_.econ_tol_/max_feas_rest_, so the
// rest of the aggregate is inert (same pattern as
// test_proximal_restoration.cpp's ProxRestoContext fixture).
SolverContext L1RestoContext(PSIOPT::Settings &settings, tycho::solvers::KktSolverType &solver,
                             Eigen::VectorXd &scratch, int &zero) {
    return SolverContext{nullptr, solver,  settings, zero,    zero,    zero,
                         zero,    zero,    scratch};
}

// =============================================================================
// (a) Slack-init truth table.
//
// Closed form (Ipopt RestoIterateInitializer, trust the code body):
//   k = resto_mu / (2 rho),  a = k - c/2,  b = c*k,
//   n = a + sqrt(a^2 + b),   p = c + n,   z_n = resto_mu/n,  z_p = resto_mu/p.
// The positive root n satisfies the quadratic  n^2 - 2 a n - b = 0 (independent
// of the sqrt form used to compute it), so we verify that identity rather than
// re-deriving n via the same expression. Every case must satisfy c + n - p = 0
// and n, p > 0 (including the mixed-sign residuals).
// =============================================================================

TEST(L1RestoSlackInit, TruthTablePositiveNegativeZeroResiduals) {
    const double rho = kRestoPenaltyParameter;
    const double resto_mu = 0.1;
    const double residuals[] = {2.0, -1.5, 0.0};

    for (double c : residuals) {
        const ElasticSlackInit s = l1_elastic_slack_init(c, resto_mu, rho);

        const double k = resto_mu / (2.0 * rho);
        const double a = k - 0.5 * c;
        const double b = c * k;

        // Defining quadratic (independent of the sqrt form): n^2 - 2 a n - b = 0.
        EXPECT_NEAR(s.n * s.n - 2.0 * a * s.n - b, 0.0, 1e-12) << "c = " << c;

        // Sign convention c + n - p = 0 (p = c + n), holds for mixed-sign c.
        EXPECT_NEAR(c + s.n - s.p, 0.0, 1e-12) << "c = " << c;

        // Strict positivity of both elastic slacks in every case.
        EXPECT_GT(s.n, 0.0) << "c = " << c;
        EXPECT_GT(s.p, 0.0) << "c = " << c;

        // Bound multipliers from the exact complementarity z*slack = resto_mu.
        EXPECT_NEAR(s.zn * s.n, resto_mu, 1e-12) << "c = " << c;
        EXPECT_NEAR(s.zp * s.p, resto_mu, 1e-12) << "c = " << c;
    }
}

TEST(L1RestoSlackInit, ZeroResidualEdgeExactValues) {
    // c = 0: a = resto_mu/(2 rho) > 0, b = 0, so n = a + sqrt(a^2) = 2a, p = n.
    // With resto_mu = 0.1, rho = 1e3: a = 5e-5, n = p = 1e-4, z = 1e3.
    const ElasticSlackInit s = l1_elastic_slack_init(0.0, 0.1, kRestoPenaltyParameter);
    EXPECT_DOUBLE_EQ(s.n, 1.0e-4);
    EXPECT_DOUBLE_EQ(s.p, 1.0e-4);
    EXPECT_DOUBLE_EQ(s.zn, 1.0e3);
    EXPECT_DOUBLE_EQ(s.zp, 1.0e3);
}

// =============================================================================
// entry_mu: resto_mu = max(outer_mu, ||eq||_inf, ||iq||_inf).
// =============================================================================

TEST(L1RestoEntryMu, MaxOfOuterMuAndInfinityNormResiduals) {
    NestedL1Restoration r;
    Eigen::VectorXd x(2);
    x << 1.0, 2.0;
    Eigen::VectorXd eq(2), iq(1);
    eq << 0.3, -0.8; // ||.||_inf = 0.8
    iq << 0.2;       // ||.||_inf = 0.2
    const ProgressMeasures ref{0.8, 1.0, 0.0};

    // outer_mu below both norms -> resto_mu = max(0.05, 0.8, 0.2) = 0.8.
    r.enter_nested(ref, x, eq, iq, /*outer_mu=*/0.05);
    EXPECT_DOUBLE_EQ(r.entry_mu(), 0.8);

    // outer_mu dominating -> resto_mu = outer_mu.
    NestedL1Restoration r2;
    r2.enter_nested(ref, x, eq, iq, /*outer_mu=*/5.0);
    EXPECT_DOUBLE_EQ(r2.entry_mu(), 5.0);
}

// =============================================================================
// (b) Dense oracle: condensation equivalence, equality and inequality channels.
//
// Assembles the enlarged (dx, dn, dp, dzn, dzp, dy) primal-dual Newton system
// with dense Eigen and solves it; then assembles the condensed (dx, dy) system
// from the component's e_pivots/i_pivots + condensed_residuals +
// nested_primal_diagonal/add_nested_gradient, solves it, recovers the elastic
// steps via recover_elastic_steps, and compares all six blocks to the enlarged
// reference. Mirrors the numpy verification (formulas independently checked to
// 1e-13). One instance drives the equality channel, one the inequality channel.
// =============================================================================

void L1RestoRunDenseOracle(bool use_inequality) {
    const int npv = 3;
    const int nec = 2;
    const double rho = kRestoPenaltyParameter;
    const double mu_live = 0.05; // barrier parameter live at the solve
    const double outer_mu = 0.05;

    // Deterministic, well-conditioned data (SPD-ish W, mixed-sign residuals).
    Eigen::MatrixXd W(npv, npv);
    W << 2.0, 0.3, 0.1, 0.3, 3.0, 0.2, 0.1, 0.2, 2.5;
    Eigen::MatrixXd Ae(nec, npv);
    Ae << 1.0, 0.5, -0.3, 0.2, -1.0, 0.4;
    Eigen::VectorXd x(npv);
    x << 0.7, -1.2, 2.3;
    Eigen::VectorXd xr(npv);
    xr << 0.9, -1.0, 2.0;
    Eigen::VectorXd cvec(nec);
    cvec << 0.8, -0.6;
    Eigen::VectorXd yvec(nec);
    yvec << 0.3, -0.4;

    const Eigen::VectorXd empty(0);

    NestedL1Restoration r;
    const ProgressMeasures ref{cvec.lpNorm<Eigen::Infinity>(), 0.0, 0.0};
    if (use_inequality) {
        r.enter_nested(ref, xr, empty, cvec, outer_mu);
    } else {
        r.enter_nested(ref, xr, cvec, empty, outer_mu);
    }

    // Pull the initialized elastic state for the channel under test.
    const Eigen::VectorXd n = use_inequality ? r.ic_n() : r.ec_n();
    const Eigen::VectorXd p = use_inequality ? r.ic_p() : r.ec_p();
    const Eigen::VectorXd zn = use_inequality ? r.ic_zn() : r.ec_zn();
    const Eigen::VectorXd zp = use_inequality ? r.ic_zp() : r.ec_zp();

    // Proximal Hessian diagonal and gradient (x rows) from the component, live mu.
    Eigen::VectorXd hd(npv);
    r.nested_primal_diagonal(mu_live, hd); // eta(mu) * d_i^2
    Eigen::VectorXd gprox = Eigen::VectorXd::Zero(npv);
    r.add_nested_gradient(mu_live, x, gprox); // eta(mu) * D^2 * (x - xr)

    // ---- Enlarged system, order [dx, dn, dp, dzn, dzp, dy] ----
    const int dim = npv + 5 * nec;
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(dim);
    const int ix = 0, in = npv, ip = npv + nec, izn = npv + 2 * nec, izp = npv + 3 * nec,
              iy = npv + 4 * nec;
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nec, nec);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(nec);

    K.block(ix, ix, npv, npv) = W + hd.asDiagonal().toDenseMatrix();
    K.block(ix, iy, npv, nec) = Ae.transpose();
    rhs.segment(ix, npv) = -(gprox + Ae.transpose() * yvec);

    K.block(in, iy, nec, nec) = I; // dy - dzn = -(rho + y - zn)
    K.block(in, izn, nec, nec) = -I;
    rhs.segment(in, nec) = -(rho * ones + yvec - zn);

    K.block(ip, iy, nec, nec) = -I; // -dy - dzp = -(rho - y - zp)
    K.block(ip, izp, nec, nec) = -I;
    rhs.segment(ip, nec) = -(rho * ones - yvec - zp);

    K.block(izn, in, nec, nec) = zn.asDiagonal(); // Zn dn + N dzn = mu - n zn
    K.block(izn, izn, nec, nec) = n.asDiagonal();
    rhs.segment(izn, nec) = mu_live * ones - n.cwiseProduct(zn);

    K.block(izp, ip, nec, nec) = zp.asDiagonal(); // Zp dp + P dzp = mu - p zp
    K.block(izp, izp, nec, nec) = p.asDiagonal();
    rhs.segment(izp, nec) = mu_live * ones - p.cwiseProduct(zp);

    K.block(iy, ix, nec, npv) = Ae; // Ae dx + dn - dp = -(c + n - p)
    K.block(iy, in, nec, nec) = I;
    K.block(iy, ip, nec, nec) = -I;
    rhs.segment(iy, nec) = -(cvec + n - p);

    const Eigen::VectorXd sol = K.fullPivLu().solve(rhs);
    const Eigen::VectorXd dx_ref = sol.segment(ix, npv);
    const Eigen::VectorXd dn_ref = sol.segment(in, nec);
    const Eigen::VectorXd dp_ref = sol.segment(ip, nec);
    const Eigen::VectorXd dzn_ref = sol.segment(izn, nec);
    const Eigen::VectorXd dzp_ref = sol.segment(izp, nec);
    const Eigen::VectorXd dy_ref = sol.segment(iy, nec);

    // ---- Condensed system [dx, dy] from the component's outputs ----
    const Eigen::VectorXd piv = use_inequality ? r.i_pivots() : r.e_pivots();
    ASSERT_EQ(piv.size(), nec);

    Eigen::VectorXd eq_rt(use_inequality ? 0 : nec);
    Eigen::VectorXd iq_rt(use_inequality ? nec : 0);
    if (use_inequality) {
        r.condensed_residuals(mu_live, empty, cvec, empty, yvec, eq_rt, iq_rt);
    } else {
        r.condensed_residuals(mu_live, cvec, empty, yvec, empty, eq_rt, iq_rt);
    }
    const Eigen::VectorXd rtilde = use_inequality ? iq_rt : eq_rt;

    Eigen::MatrixXd Kc = Eigen::MatrixXd::Zero(npv + nec, npv + nec);
    Kc.block(0, 0, npv, npv) = W + hd.asDiagonal().toDenseMatrix();
    Kc.block(0, npv, npv, nec) = Ae.transpose();
    Kc.block(npv, 0, nec, npv) = Ae;
    Kc.block(npv, npv, nec, nec) = (-piv).asDiagonal(); // seam negates the pivot
    Eigen::VectorXd rc(npv + nec);
    rc.head(npv) = -(gprox + Ae.transpose() * yvec);
    rc.tail(nec) = -rtilde;

    const Eigen::VectorXd solc = Kc.fullPivLu().solve(rc);
    const Eigen::VectorXd dx = solc.head(npv);
    const Eigen::VectorXd dy = solc.tail(nec);

    // ---- Recover the elastic steps from dy ----
    if (use_inequality) {
        r.recover_elastic_steps(mu_live, empty, yvec, empty, dy);
    } else {
        r.recover_elastic_steps(mu_live, yvec, empty, dy, empty);
    }
    const Eigen::VectorXd dn = use_inequality ? r.ic_dn() : r.ec_dn();
    const Eigen::VectorXd dp = use_inequality ? r.ic_dp() : r.ec_dp();
    const Eigen::VectorXd dzn = use_inequality ? r.ic_dzn() : r.ec_dzn();
    const Eigen::VectorXd dzp = use_inequality ? r.ic_dzp() : r.ec_dzp();

    const double tol = 1e-9;
    EXPECT_LT((dx - dx_ref).cwiseAbs().maxCoeff(), tol);
    EXPECT_LT((dy - dy_ref).cwiseAbs().maxCoeff(), tol);
    EXPECT_LT((dn - dn_ref).cwiseAbs().maxCoeff(), tol);
    EXPECT_LT((dp - dp_ref).cwiseAbs().maxCoeff(), tol);
    EXPECT_LT((dzn - dzn_ref).cwiseAbs().maxCoeff(), tol);
    EXPECT_LT((dzp - dzp_ref).cwiseAbs().maxCoeff(), tol);
}

TEST(L1RestoDenseOracle, EqualityChannelCondensationMatchesEnlargedSolve) {
    L1RestoRunDenseOracle(/*use_inequality=*/false);
}

TEST(L1RestoDenseOracle, InequalityChannelCondensationMatchesEnlargedSolve) {
    L1RestoRunDenseOracle(/*use_inequality=*/true);
}

// =============================================================================
// (c) eta(mu) live recompute: nested_objective differs across mu by exactly the
// proximal term (eta(mu1) - eta(mu2))/2 * ||D(x - xr)||^2 (the rho*sum(n+p)
// term is mu-independent).
// =============================================================================

TEST(L1RestoLiveEta, ObjectiveDeltaAcrossMuIsProximalTermOnly) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(3);
    xr << 0.5, 2.0, 4.0; // d = [1, 0.5, 0.25] -> dr2 = [1, 0.25, 0.0625]
    Eigen::VectorXd eq(2);
    eq << 0.7, -0.9;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.9, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, /*outer_mu=*/0.1);

    Eigen::VectorXd x(3);
    x << 1.5, 3.0, 5.0; // x - xr = [1, 1, 1]

    const double mu1 = 0.04, mu2 = 0.16;
    const double obj1 = r.nested_objective(mu1, x);
    const double obj2 = r.nested_objective(mu2, x);

    // ||D(x - xr)||^2 = sum_i dr2_i * (x_i - xr_i)^2 = 1 + 0.25 + 0.0625 = 1.3125.
    const double dnorm2 = 1.0 + 0.25 + 0.0625;
    const double eta1 = kRestoProximityWeight * std::sqrt(mu1);
    const double eta2 = kRestoProximityWeight * std::sqrt(mu2);
    const double expected = (eta1 - eta2) / 2.0 * dnorm2;
    EXPECT_NEAR(obj1 - obj2, expected, 1e-12);
}

// =============================================================================
// (d) D_R truth table: d_i = 1/max(1, |xr_i|), dr2_i = d_i^2, for
// |xr| < 1, > 1, = 0.
// =============================================================================

TEST(L1RestoScaling, DrSquaredTruthTable) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(4);
    xr << 0.5, 4.0, 0.0, -8.0; // max(1,|.|) = [1, 4, 1, 8]; d = [1, 0.25, 1, 0.125]
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.0, 0.0, 0.0};
    r.enter_nested(ref, xr, empty, empty, /*outer_mu=*/1.0);

    ASSERT_EQ(r.dr2().size(), 4);
    EXPECT_DOUBLE_EQ(r.dr2()[0], 1.0);          // |xr| < 1 -> d = 1
    EXPECT_DOUBLE_EQ(r.dr2()[1], 0.0625);       // |xr| > 1 -> d = 1/4
    EXPECT_DOUBLE_EQ(r.dr2()[2], 1.0);          // |xr| = 0 -> d = 1 (no divide-by-zero)
    EXPECT_DOUBLE_EQ(r.dr2()[3], 0.015625);     // |xr| = 8 -> d = 1/8
}

// =============================================================================
// (e) Fraction-to-boundary caps: the recovered steps are verified by the dense
// oracle; here we verify that primal_boundary_alpha / dual_boundary_alpha apply
// the standard tau rule to those steps. The expected cap is recomputed
// independently from the component's own recovered steps:
//   max alpha in (0,1] with  v + alpha*dv >= (1 - tau) v  for each positive v.
// =============================================================================

double L1RestoTauCap(const Eigen::VectorXd &v, const Eigen::VectorXd &dv, double tau) {
    double alpha = 1.0;
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (dv[i] < 0.0) {
            alpha = std::min(alpha, -tau * v[i] / dv[i]);
        }
    }
    return alpha;
}

TEST(L1RestoFractionToBoundary, PrimalAndDualCapsMatchTauRule) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(2);
    xr << 1.0, 1.0;
    Eigen::VectorXd eq(2);
    // A large positive residual drives n tiny -> z_n large (~2000), so the dual
    // ftb cap has a chance to bind on Delta z_n; the negative residual keeps the
    // channel mixed-sign.
    eq << 2.0, -0.5;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{2.0, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, /*outer_mu=*/0.1);

    const double mu_live = 0.1;
    // Constraint multipliers y, a distinct quantity from the residuals c above:
    // recover_elastic_steps reads them as the (rho +/- y) terms of the step
    // formulas, so feeding the residual vector here would pin the seam with
    // semantically wrong data. Mixed sign, order 1 (rho = 1e3 dominates either
    // way), and different from both `eq` and `dy`.
    Eigen::VectorXd lmults(2);
    lmults << 0.3, -0.7;
    Eigen::VectorXd dy(2);
    dy << 0.5, -0.3;
    r.recover_elastic_steps(mu_live, lmults, empty, dy, empty);

    const double tau = 0.99;

    // Independent tau-rule cap over the component's recovered primal steps.
    Eigen::VectorXd vp(4), dvp(4);
    vp << r.ec_n(), r.ec_p();
    dvp << r.ec_dn(), r.ec_dp();
    const double expected_primal = L1RestoTauCap(vp, dvp, tau);

    Eigen::VectorXd vd(4), dvd(4);
    vd << r.ec_zn(), r.ec_zp();
    dvd << r.ec_dzn(), r.ec_dzp();
    const double expected_dual = L1RestoTauCap(vd, dvd, tau);

    EXPECT_NEAR(r.primal_boundary_alpha(tau), expected_primal, 1e-14);
    EXPECT_NEAR(r.dual_boundary_alpha(tau), expected_dual, 1e-14);

    // With rho = 1e3 the primal step Delta n is strongly negative, so the primal
    // cap must actually bind below 1 (exercises the tau rule, not just the 1.0
    // default). The "cap is a valid fraction in (0, 1]" contract itself lives in
    // CapNeverExceedsUnity below, where the cap does NOT bind.
    EXPECT_LT(r.primal_boundary_alpha(tau), 1.0);
}

// The complement of the test above: a channel on which no positive variable's
// step is large enough to bind, so both caps stay at their unit default. Pinned
// exactly (not just <= 1) -- an implementation that started the cap anywhere but
// 1.0, or that let a non-binding step scale it, would fail here.
TEST(L1RestoFractionToBoundary, CapNeverExceedsUnity) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(1);
    xr << 1.0;
    Eigen::VectorXd eq(1);
    eq << 0.0; // n = p = 1e-4, z_n = z_p = 1e3 at resto_mu = 0.1
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.0, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, /*outer_mu=*/0.1);

    // Multipliers y (NOT the residuals): at |y| of order 1 the (rho +/- y) terms
    // cancel mu/z to within ~1e-7 * |y|, so |Delta n|, |Delta p| ~ 1e-11 against
    // n = p = 1e-4 and |Delta z| ~ |y| against z = 1e3 -- three orders too small
    // for the tau rule to bind on either channel.
    Eigen::VectorXd lmults(1);
    lmults << 0.4;
    // A zero multiplier-step recovery.
    Eigen::VectorXd dy(1);
    dy << 0.0;
    r.recover_elastic_steps(0.1, lmults, empty, dy, empty);

    EXPECT_DOUBLE_EQ(r.primal_boundary_alpha(0.99), 1.0);
    EXPECT_DOUBLE_EQ(r.dual_boundary_alpha(0.99), 1.0);
}

// =============================================================================
// apply_elastic_step + trial_residual_shift + trial_objective consistency.
// =============================================================================

TEST(L1RestoStepApplication, ApplyMovesSlacksAndUpdatesPivots) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(2);
    xr << 1.0, 1.0;
    Eigen::VectorXd eq(2);
    eq << 0.6, -0.4;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.6, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, /*outer_mu=*/0.1);

    const Eigen::VectorXd n0 = r.ec_n();
    const Eigen::VectorXd p0 = r.ec_p();
    const Eigen::VectorXd zn0 = r.ec_zn();
    const Eigen::VectorXd zp0 = r.ec_zp();

    // Constraint multipliers y, distinct from the residuals `eq` (see the note
    // in PrimalAndDualCapsMatchTauRule).
    Eigen::VectorXd lmults(2);
    lmults << 0.3, -0.7;
    Eigen::VectorXd dy(2);
    dy << 0.2, -0.1;
    r.recover_elastic_steps(0.1, lmults, empty, dy, empty);
    const Eigen::VectorXd dn = r.ec_dn();
    const Eigen::VectorXd dp = r.ec_dp();
    const Eigen::VectorXd dzn = r.ec_dzn();
    const Eigen::VectorXd dzp = r.ec_dzp();

    const double ap = 0.3, ad = 0.4;
    r.apply_elastic_step(ap, ad);

    // 1e-12, not 1e-14: apply_elastic_step's internal x + a*dx and this
    // expression contract FMAs differently per compiler/ISA (observed up to
    // 6.8e-14 on arm64 vs 1e-14 passing on x86). The bound stays far below
    // any behavioral-change signal.
    EXPECT_NEAR((r.ec_n() - (n0 + ap * dn)).cwiseAbs().maxCoeff(), 0.0, 1e-12);
    EXPECT_NEAR((r.ec_p() - (p0 + ap * dp)).cwiseAbs().maxCoeff(), 0.0, 1e-12);
    EXPECT_NEAR((r.ec_zn() - (zn0 + ad * dzn)).cwiseAbs().maxCoeff(), 0.0, 1e-12);
    EXPECT_NEAR((r.ec_zp() - (zp0 + ad * dzp)).cwiseAbs().maxCoeff(), 0.0, 1e-12);

    // pivot = n/z_n + p/z_p recomputed from the moved state.
    Eigen::VectorXd expected_piv =
        r.ec_n().cwiseQuotient(r.ec_zn()) + r.ec_p().cwiseQuotient(r.ec_zp());
    EXPECT_NEAR((r.e_pivots() - expected_piv).cwiseAbs().maxCoeff(), 0.0, 1e-12);
}

TEST(L1RestoStepApplication, TrialResidualShiftIsAlphaBlendedSlackDifference) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(2);
    xr << 1.0, 1.0;
    Eigen::VectorXd eq(2);
    eq << 0.6, -0.4;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.6, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, /*outer_mu=*/0.1);

    // Constraint multipliers y, distinct from the residuals `eq`.
    Eigen::VectorXd lmults(2);
    lmults << 0.3, -0.7;
    Eigen::VectorXd dy(2);
    dy << 0.2, -0.1;
    r.recover_elastic_steps(0.1, lmults, empty, dy, empty);

    const double alpha = 0.35;
    Eigen::VectorXd eq_shift(2), iq_shift(0);
    r.trial_residual_shift(alpha, eq_shift, iq_shift);

    // shift = (n + alpha*dn) - (p + alpha*dp).
    const Eigen::VectorXd expected =
        (r.ec_n() + alpha * r.ec_dn()) - (r.ec_p() + alpha * r.ec_dp());
    EXPECT_NEAR((eq_shift - expected).cwiseAbs().maxCoeff(), 0.0, 1e-14);
}

TEST(L1RestoStepApplication, TrialObjectiveBlendsSlacksAndProximalTerm) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(2);
    xr << 0.5, 2.0; // dr2 = [1, 0.25]
    Eigen::VectorXd eq(2);
    eq << 0.6, -0.4;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.6, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, /*outer_mu=*/0.1);

    // Constraint multipliers y, distinct from the residuals `eq`.
    Eigen::VectorXd lmults(2);
    lmults << 0.3, -0.7;
    Eigen::VectorXd dy(2);
    dy << 0.2, -0.1;
    r.recover_elastic_steps(0.1, lmults, empty, dy, empty);

    const double mu_live = 0.09; // eta = 0.3
    const double alpha = 0.5;
    Eigen::VectorXd xt(2);
    xt << 1.0, 3.0; // trial primal

    const double rho = kRestoPenaltyParameter;
    const Eigen::VectorXd nt = r.ec_n() + alpha * r.ec_dn();
    const Eigen::VectorXd pt = r.ec_p() + alpha * r.ec_dp();
    const double eta = kRestoProximityWeight * std::sqrt(mu_live);
    const Eigen::VectorXd dx = xt - xr;
    const double prox = 0.5 * eta * r.dr2().dot(dx.cwiseProduct(dx));
    const double expected = rho * (nt.sum() + pt.sum()) + prox;

    // Relative tolerance: the recovered steps put |expected| near 4e5, where one
    // double ULP is ~6e-11 — an absolute 1e-12 bound is tighter than representable.
    EXPECT_NEAR(r.trial_objective(mu_live, alpha, xt), expected, 1e-12 * std::abs(expected));
}

// =============================================================================
// Second-level elastic re-centering fallback (recenter_elastics). Re-solving the
// separable elastic subproblem at a NEW barrier parameter and NEW residuals must
// match the entry-init closed form on both channels, hold c + n − p = 0 (mixed
// sign), pair n·z = μ / p·z = μ exactly, and refresh the pivots. The one-shot
// guard itself lives in alg_impl and is exercised by the wiring suite.
// =============================================================================

TEST(L1RestoRecenter, ClosedFormBothChannelsAtNewMuAndResiduals) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(2);
    xr << 1.0, 2.0;
    // Entry residuals differ from the re-center residuals, and the entry μ differs
    // from the re-center μ, so a stale-state re-center would be caught.
    Eigen::VectorXd eq0(2), iq0(1);
    eq0 << 0.6, -0.4;
    iq0 << 0.3;
    const ProgressMeasures ref{0.6, 0.0, 0.0};
    r.enter_nested(ref, xr, eq0, iq0, /*outer_mu=*/0.1);
    ASSERT_EQ(r.recenter_calls(), 0);

    // Re-center at a new μ with fresh mixed-sign residuals on both channels.
    const double mu = 0.037;
    Eigen::VectorXd eq(2), iq(1);
    eq << 1.25, -0.8;
    iq << -0.5;
    r.recenter_elastics(mu, eq, iq);
    EXPECT_EQ(r.recenter_calls(), 1);

    const double rho = kRestoPenaltyParameter;
    auto check_channel = [&](const Eigen::VectorXd &c, const Eigen::VectorXd &n,
                             const Eigen::VectorXd &p, const Eigen::VectorXd &zn,
                             const Eigen::VectorXd &zp) {
        ASSERT_EQ(n.size(), c.size());
        for (Eigen::Index i = 0; i < c.size(); ++i) {
            const ElasticSlackInit s = l1_elastic_slack_init(c[i], mu, rho);
            EXPECT_DOUBLE_EQ(n[i], s.n) << "row " << i;
            EXPECT_DOUBLE_EQ(p[i], s.p) << "row " << i;
            EXPECT_DOUBLE_EQ(zn[i], s.zn) << "row " << i;
            EXPECT_DOUBLE_EQ(zp[i], s.zp) << "row " << i;
            // c + n − p = 0 exactly (p = c + n), mixed sign.
            EXPECT_NEAR(c[i] + n[i] - p[i], 0.0, 1e-12) << "row " << i;
            // Complementarity pairing n·z_n = μ, p·z_p = μ.
            EXPECT_NEAR(n[i] * zn[i], mu, 1e-12) << "row " << i;
            EXPECT_NEAR(p[i] * zp[i], mu, 1e-12) << "row " << i;
            EXPECT_GT(n[i], 0.0) << "row " << i;
            EXPECT_GT(p[i], 0.0) << "row " << i;
        }
    };
    check_channel(eq, r.ec_n(), r.ec_p(), r.ec_zn(), r.ec_zp());
    check_channel(iq, r.ic_n(), r.ic_p(), r.ic_zn(), r.ic_zp());

    // Pivots refreshed from the re-centered state: pivot = n/z_n + p/z_p.
    const Eigen::VectorXd e_piv_expected =
        r.ec_n().cwiseQuotient(r.ec_zn()) + r.ec_p().cwiseQuotient(r.ec_zp());
    const Eigen::VectorXd i_piv_expected =
        r.ic_n().cwiseQuotient(r.ic_zn()) + r.ic_p().cwiseQuotient(r.ic_zp());
    ASSERT_EQ(r.e_pivots().size(), 2);
    ASSERT_EQ(r.i_pivots().size(), 1);
    for (Eigen::Index i = 0; i < 2; ++i)
        EXPECT_DOUBLE_EQ(r.e_pivots()[i], e_piv_expected[i]) << "eq pivot " << i;
    EXPECT_DOUBLE_EQ(r.i_pivots()[0], i_piv_expected[0]);

    // A second re-center is a fresh full re-solve (the one-shot guard is an
    // alg_impl-level concern; the component method itself always re-centers).
    r.recenter_elastics(mu, eq, iq);
    EXPECT_EQ(r.recenter_calls(), 2);
}

// reset() clears the re-center observer alongside the other diagnostics.
TEST(L1RestoRecenter, ResetClearsRecenterCount) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(1);
    xr << 1.0;
    Eigen::VectorXd eq(1);
    eq << 0.5;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.5, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, 0.1);
    r.recenter_elastics(0.05, eq, empty);
    ASSERT_EQ(r.recenter_calls(), 1);
    r.reset();
    EXPECT_EQ(r.recenter_calls(), 0);
}

// =============================================================================
// (f) Budget / near-feasible guard parity with the shipped entry-permitted
// semantics (same constants as the proximal switch).
// =============================================================================

TEST(L1RestoEntryPermitted, NearFeasibleGuardBoundary) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1e-6;
    settings.max_feas_rest_ = 2;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    const SolverContext ctx = L1RestoContext(settings, solver, scratch, zero);

    NestedL1Restoration r;
    const double threshold = kNearFeasibleGuardFactor * settings.econ_tol_; // 1e-7
    EXPECT_FALSE(r.entry_permitted(threshold, ctx));       // "<=" refuses at boundary
    EXPECT_FALSE(r.entry_permitted(threshold * 0.5, ctx)); // below -> refused
    EXPECT_TRUE(r.entry_permitted(threshold * 2.0, ctx));  // above -> permitted
    EXPECT_TRUE(r.entry_permitted(1.0, ctx));
}

TEST(L1RestoEntryPermitted, BudgetExhaustionCountsNestedEntries) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1e-6;
    settings.max_feas_rest_ = 2;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    const SolverContext ctx = L1RestoContext(settings, solver, scratch, zero);

    NestedL1Restoration r;
    Eigen::VectorXd xr(1);
    xr << 1.0;
    Eigen::VectorXd eq(1);
    eq << 0.5;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.5, 0.0, 0.0};

    const double violation = 1.0;
    EXPECT_TRUE(r.entry_permitted(violation, ctx));
    r.enter_nested(ref, xr, eq, empty, 0.1);
    EXPECT_EQ(r.entries(), 1);
    EXPECT_TRUE(r.entry_permitted(violation, ctx));
    r.enter_nested(ref, xr, eq, empty, 0.1);
    EXPECT_EQ(r.entries(), 2);
    EXPECT_FALSE(r.entry_permitted(violation, ctx)); // 2 >= max_feas_rest_
}

TEST(L1RestoEntryPermitted, ZeroBudgetAlwaysRefuses) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1e-6;
    settings.max_feas_rest_ = 0;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    const SolverContext ctx = L1RestoContext(settings, solver, scratch, zero);

    NestedL1Restoration r;
    EXPECT_FALSE(r.entry_permitted(1e6, ctx));
    EXPECT_EQ(r.entries(), 0);
}

// =============================================================================
// is_nested identification and the proximal-trio guard (nested mode does not
// implement the frozen-zeta proximal surface).
// =============================================================================

TEST(L1RestoIdentity, IsNestedTrue) {
    NestedL1Restoration r;
    EXPECT_TRUE(r.is_nested());
}

TEST(L1RestoIdentity, ProximalTrioThrowsForNestedMode) {
    NestedL1Restoration r;
    Eigen::VectorXd x(1);
    x << 1.0;
    Eigen::VectorXd grad(1);
    grad << 0.0;
    EXPECT_THROW(r.proximal_objective(x), std::logic_error);
    EXPECT_THROW(r.add_proximal_gradient(x, grad), std::logic_error);
    EXPECT_THROW(r.proximal_diagonal(), std::logic_error);
}

// =============================================================================
// (g) reset / exit lifecycle clears elastic state; diagnostics report through
// append_diagnostics.
// =============================================================================

TEST(L1RestoLifecycle, EnterActivatesExitDeactivates) {
    NestedL1Restoration r;
    EXPECT_FALSE(r.is_active());
    Eigen::VectorXd xr(1);
    xr << 2.0;
    Eigen::VectorXd eq(1);
    eq << 0.5;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.5, 0.0, 0.0};
    r.enter_nested(ref, xr, eq, empty, 0.1);
    EXPECT_TRUE(r.is_active());
    r.exit_restoration();
    EXPECT_FALSE(r.is_active());
}

TEST(L1RestoReset, ClearsAllElasticState) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(2);
    xr << 1.0, 2.0;
    Eigen::VectorXd eq(2), iq(1);
    eq << 0.6, -0.4;
    iq << 0.2;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.7, 0.3, 0.1};
    r.enter_nested(ref, xr, eq, iq, 0.1);
    r.note_iteration();
    r.note_iteration();
    ASSERT_TRUE(r.is_active());
    ASSERT_EQ(r.entries(), 1);
    ASSERT_EQ(r.iterations_in_mode(), 2);
    ASSERT_GT(r.ec_n().size(), 0);
    ASSERT_GT(r.ic_n().size(), 0);
    ASSERT_GT(r.dr2().size(), 0);
    ASSERT_GT(r.e_pivots().size(), 0);

    r.reset();
    EXPECT_FALSE(r.is_active());
    EXPECT_EQ(r.entries(), 0);
    EXPECT_EQ(r.iterations_in_mode(), 0);
    EXPECT_EQ(r.ec_n().size(), 0);
    EXPECT_EQ(r.ec_p().size(), 0);
    EXPECT_EQ(r.ic_n().size(), 0);
    EXPECT_EQ(r.dr2().size(), 0);
    EXPECT_EQ(r.e_pivots().size(), 0);
    EXPECT_EQ(r.i_pivots().size(), 0);
    EXPECT_DOUBLE_EQ(r.entry_mu(), 0.0);
    EXPECT_DOUBLE_EQ(r.reference().infeasibility, 0.0);
}

TEST(L1RestoDiagnostics, NeverEnteredReportsZeroZero) {
    NestedL1Restoration r;
    PSIOPT::SolveResult result;
    r.append_diagnostics(result);
    EXPECT_EQ(result.last_feas_rest_entries_, 0);
    EXPECT_EQ(result.last_feas_rest_iters_, 0);
}

TEST(L1RestoDiagnostics, ReportsEntriesAndIterationsInMode) {
    NestedL1Restoration r;
    Eigen::VectorXd xr(1);
    xr << 1.0;
    Eigen::VectorXd eq(1);
    eq << 0.5;
    const Eigen::VectorXd empty(0);
    const ProgressMeasures ref{0.5, 0.0, 0.0};

    r.enter_nested(ref, xr, eq, empty, 0.1);
    r.note_iteration();
    r.note_iteration();
    r.note_iteration();
    r.exit_restoration();
    r.enter_nested(ref, xr, eq, empty, 0.2);
    r.note_iteration();

    PSIOPT::SolveResult result;
    r.append_diagnostics(result);
    EXPECT_EQ(result.last_feas_rest_entries_, 2);
    EXPECT_EQ(result.last_feas_rest_iters_, 4);
}

// Reference the reset-threshold constant so the header's declaration is
// exercised at its literature default (Ipopt bound_mult_reset_threshold).
TEST(L1RestoConstants, BoundMultResetThresholdDefault) {
    EXPECT_DOUBLE_EQ(kBoundMultResetThreshold, 1.0e3);
}

} // namespace
