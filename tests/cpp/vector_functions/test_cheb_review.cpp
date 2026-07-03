// ChebTable / ChebFunction review-response tests: input-validation error paths,
// dimension guards, out-of-domain policy (Clamp + Periodic), bounds accessors /
// contains(), the order==1 (linear / zero-2nd-derivative) path, and the
// ChebFunction<IR> compile-time/runtime dimension-mismatch guard.

#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

static double Tk(int k, double x) { return std::cos(k * std::acos(std::clamp(x, -1.0, 1.0))); }

class ChebReviewTest : public VectorFunctionFixture {};

// --- Build helpers -----------------------------------------------------------

static oc::ChebTable make_1d(int n = 6, double lb = -1.0, double ub = 1.0) {
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    oc::ChebTable::MatType vals(n + 1, 1);
    for (int j = 0; j <= n; ++j)
        vals(j, 0) = Tk(3, pts[j]);
    return oc::ChebTable::from_values(vals, lb, ub, n);
}

static oc::ChebTable make_2d(const std::vector<int> &orders, const Eigen::VectorXd &lb,
                             const Eigen::VectorXd &ub,
                             std::vector<oc::ChebTable::OutOfDomain> oob = {}) {
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            vals(i * ny + j, 0) = Tk(2, nodes[0][i]) * Tk(1, nodes[1][j]);
    return oc::ChebTable::from_values(vals, lb, ub, orders, 1, std::move(oob));
}

// --- Input-validation error paths (1-D) -------------------------------------

TEST_F(ChebReviewTest, ConstructionValidation1D) {
    oc::ChebTable::MatType vals(5, 1); // order 4 -> 5 rows (nodes)
    vals.setOnes();
    EXPECT_THROW(oc::ChebTable::from_values(vals, 0.0, 1.0, /*order=*/0), std::invalid_argument);
    // wrong row count vs order
    EXPECT_THROW(oc::ChebTable::from_values(vals, 0.0, 1.0, /*order=*/6), std::invalid_argument);
    // ub <= lb
    EXPECT_THROW(oc::ChebTable::from_values(vals, 1.0, 1.0, /*order=*/4), std::invalid_argument);
    EXPECT_THROW(oc::ChebTable::from_values(vals, 2.0, 1.0, /*order=*/4), std::invalid_argument);
}

// --- Input-validation error paths (N-D) -------------------------------------

TEST_F(ChebReviewTest, ConstructionValidationND) {
    std::vector<int> orders{3, 3};
    Eigen::VectorXd lb(2), ub(2);
    lb << 0, 0;
    ub << 1, 1;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 1);
    vals.setOnes();

    // lb/ub length mismatch
    Eigen::VectorXd lb1(1);
    lb1 << 0;
    EXPECT_THROW(oc::ChebTable::from_values(vals, lb1, ub, orders), std::invalid_argument);
    // wrong row count
    oc::ChebTable::MatType bad(nx * ny + 1, 1);
    bad.setOnes();
    EXPECT_THROW(oc::ChebTable::from_values(bad, lb, ub, orders), std::invalid_argument);
    // ub <= lb per axis
    Eigen::VectorXd ubbad(2);
    ubbad << 1, 0;
    EXPECT_THROW(oc::ChebTable::from_values(vals, lb, ubbad, orders), std::invalid_argument);
    // out_of_domain length mismatch
    std::vector<oc::ChebTable::OutOfDomain> oob1{oc::ChebTable::OutOfDomain::Clamp};
    EXPECT_THROW(oc::ChebTable::from_values(vals, lb, ub, orders, 1, oob1), std::invalid_argument);
}

// --- Dimension guards --------------------------------------------------------

TEST_F(ChebReviewTest, ScalarEvalOnNdTableThrows) {
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto tab = make_2d({4, 3}, lb, ub);
    EXPECT_THROW(tab.eval(0.5), std::invalid_argument);
    EXPECT_THROW(tab.eval_deriv1(0.5), std::invalid_argument);
    EXPECT_THROW(tab.eval_deriv2(0.5), std::invalid_argument);
}

TEST_F(ChebReviewTest, WrongLengthQueryThrows) {
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto tab = make_2d({4, 3}, lb, ub);
    Eigen::VectorXd q3(3);
    q3 << 0.1, 0.2, 0.3;
    EXPECT_THROW(tab.eval(q3), std::invalid_argument);
    EXPECT_THROW(tab.eval_jacobian(q3), std::invalid_argument);
    EXPECT_THROW(tab.eval_hessian(q3), std::invalid_argument);
}

TEST_F(ChebReviewTest, DefaultConstructedTableThrows) {
    oc::ChebTable empty;
    EXPECT_THROW(empty.eval(0.5), std::invalid_argument);
    Eigen::VectorXd q(1);
    q << 0.5;
    EXPECT_THROW(empty.eval(q), std::invalid_argument);
}

// --- Bounds accessors + contains() ------------------------------------------

TEST_F(ChebReviewTest, BoundsAndContains) {
    auto tab = make_1d(6, -2.0, 3.0);
    ASSERT_EQ(tab.lb().size(), 1);
    EXPECT_NEAR(tab.lb()[0], -2.0, 1e-15);
    EXPECT_NEAR(tab.ub()[0], 3.0, 1e-15);
    EXPECT_TRUE(tab.contains(0.0));
    EXPECT_FALSE(tab.contains(-3.0));
    EXPECT_FALSE(tab.contains(4.0));

    Eigen::VectorXd lb(2), ub(2);
    lb << -1, 0;
    ub << 1, 2;
    auto t2 = make_2d({4, 3}, lb, ub);
    Eigen::VectorXd inside(2), outside(2);
    inside << 0.0, 1.0;
    outside << 0.0, 3.0;
    EXPECT_TRUE(t2.contains(inside));
    EXPECT_FALSE(t2.contains(outside));
}

// --- Out-of-domain clamp (default) ------------------------------------------

TEST_F(ChebReviewTest, ClampMatchesEndpoint1D) {
    auto tab = make_1d(8, 0.0, 2.0);
    // A query past ub returns exactly the endpoint value (no extrapolation).
    EXPECT_NEAR(tab.eval(2.0 + 5.0)[0], tab.eval(2.0)[0], 1e-14);
    EXPECT_NEAR(tab.eval(0.0 - 5.0)[0], tab.eval(0.0)[0], 1e-14);
}

TEST_F(ChebReviewTest, ClampPerAxisND) {
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto tab = make_2d({5, 5}, lb, ub);
    // Clamp axis 0 past ub while axis 1 stays interior; must match evaluating at
    // the clamped point (ub on axis 0).
    Eigen::VectorXd q(2), qc(2);
    q << 5.0, 0.3;
    qc << 1.0, 0.3;
    EXPECT_NEAR(tab.eval(q)[0], tab.eval(qc)[0], 1e-13);
}

// --- Out-of-domain periodic -------------------------------------------------

TEST_F(ChebReviewTest, PeriodicWraps1D) {
    // f(t) = cos(t) on [0, 2pi] is genuinely periodic: f(lb) == f(ub).
    const int n = 24;
    const double lb = 0.0, ub = 2.0 * std::numbers::pi;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    oc::ChebTable::MatType vals(n + 1, 1);
    for (int j = 0; j <= n; ++j)
        vals(j, 0) = std::cos(pts[j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, n, 1, oc::ChebTable::OutOfDomain::Periodic);
    // A query one full period past the domain equals the in-domain query.
    const double t = 1.3;
    EXPECT_NEAR(tab.eval(t + 2.0 * (ub - lb))[0], tab.eval(t)[0], 1e-11);
    EXPECT_NEAR(tab.eval(t - (ub - lb))[0], tab.eval(t)[0], 1e-11);
    // And it tracks the true periodic function.
    EXPECT_NEAR(tab.eval(t + (ub - lb))[0], std::cos(t), 1e-9);
}

TEST_F(ChebReviewTest, PeriodicPerAxisND) {
    // f(x,y) = cos(x) * y-poly; axis 0 periodic on [0, 2pi], axis 1 clamp on [-1,1].
    std::vector<int> orders{20, 3};
    Eigen::VectorXd lb(2), ub(2);
    lb << 0.0, -1.0;
    ub << 2.0 * std::numbers::pi, 1.0;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            vals(i * ny + j, 0) = std::cos(nodes[0][i]) * Tk(1, nodes[1][j]);
    std::vector<oc::ChebTable::OutOfDomain> oob{oc::ChebTable::OutOfDomain::Periodic,
                                                oc::ChebTable::OutOfDomain::Clamp};
    auto tab = oc::ChebTable::from_values(vals, lb, ub, orders, 1, oob);

    Eigen::VectorXd q(2), qw(2);
    q << 1.1, 0.4;
    qw << 1.1 + (ub[0] - lb[0]), 0.4; // wrapped on axis 0
    EXPECT_NEAR(tab.eval(qw)[0], tab.eval(q)[0], 1e-10);
    EXPECT_EQ(tab.out_of_domain().size(), 2u);
    EXPECT_EQ(tab.out_of_domain()[0], oc::ChebTable::OutOfDomain::Periodic);
    EXPECT_EQ(tab.out_of_domain()[1], oc::ChebTable::OutOfDomain::Clamp);
}

// --- order == 1 (linear / zero second derivative) ---------------------------

TEST_F(ChebReviewTest, OrderOneLinearZeroSecondDeriv) {
    // order 1 -> 2 nodes -> exact linear interpolant; d2/dt2 == 0.
    const int n = 1;
    const double lb = 0.0, ub = 4.0;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    auto f = [](double t) { return 2.0 * t - 1.0; };
    oc::ChebTable::MatType vals(n + 1, 1);
    for (int j = 0; j <= n; ++j)
        vals(j, 0) = f(pts[j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, n);
    for (double t : {0.5, 1.7, 3.2}) {
        auto [v, dv, d2] = tab.eval_deriv2(t);
        EXPECT_NEAR(v[0], f(t), 1e-12) << "t=" << t;
        EXPECT_NEAR(dv[0], 2.0, 1e-10) << "t=" << t;
        EXPECT_NEAR(d2[0], 0.0, 1e-12) << "t=" << t;
    }
}

// --- Out-of-domain derivative contract (zero outside Clamp domain) ----------

// Smooth non-saturating 1-D table on [0, 2] (Tk clamps its arg to [-1,1], so it
// is unsuitable here — use a genuine sinusoid with nonzero endpoint slope).
static oc::ChebTable make_smooth_1d(int n, double lb, double ub) {
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    oc::ChebTable::MatType vals(n + 1, 1);
    for (int j = 0; j <= n; ++j)
        vals(j, 0) = std::sin(1.5 * pts[j]);
    return oc::ChebTable::from_values(vals, lb, ub, n);
}

TEST_F(ChebReviewTest, ClampDerivativesZeroOutside1D) {
    // Outside [lb, ub] under Clamp the value is a flat endpoint constant, so the
    // analytic first/second derivatives must be zero (consistent with that flat
    // value) — not the nonzero endpoint slope.
    auto tab = make_smooth_1d(24, 0.0, 2.0);
    for (double t : {2.0 + 3.0, 0.0 - 3.0}) {
        auto [v, dv] = tab.eval_deriv1(t);
        auto [v2, dv2, d2] = tab.eval_deriv2(t);
        EXPECT_NEAR(dv[0], 0.0, 1e-14) << "t=" << t;
        EXPECT_NEAR(dv2[0], 0.0, 1e-14) << "t=" << t;
        EXPECT_NEAR(d2[0], 0.0, 1e-14) << "t=" << t;
        // Value is flat there, so a central difference of eval agrees (~0).
        const double e = 1e-4;
        double fd = (tab.eval(t + e)[0] - tab.eval(t - e)[0]) / (2 * e);
        EXPECT_NEAR(fd, 0.0, 1e-10) << "FD t=" << t;
    }
}

TEST_F(ChebReviewTest, ClampBoundaryRetainsEndpointSlope1D) {
    // Exactly at the boundary the query is in-domain: the one-sided endpoint
    // slope is retained (not zeroed), matching a one-sided finite difference.
    auto tab = make_smooth_1d(24, 0.0, 2.0);
    auto [v, dv] = tab.eval_deriv1(2.0);
    const double e = 1e-6;
    double fd = (tab.eval(2.0)[0] - tab.eval(2.0 - e)[0]) / e; // one-sided from interior
    EXPECT_NEAR(dv[0], fd, 1e-4);
    EXPECT_GT(std::abs(dv[0]), 1e-6); // genuinely nonzero endpoint slope (1.5*cos(3)≈-1.48)
}

TEST_F(ChebReviewTest, ClampDerivativesZeroOutsideND) {
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto tab = make_2d({5, 5}, lb, ub);
    Eigen::VectorXd q(2), qc(2);
    q << 5.0, 0.3;  // axis 0 clamped (out of domain), axis 1 interior
    qc << 1.0, 0.3; // the clamped point (axis 0 at its ub boundary)
    auto Jout = tab.eval_jacobian(q);
    auto Jbnd = tab.eval_jacobian(qc);
    // Clamped axis 0 -> zero partial; interior axis 1 unchanged vs the boundary point.
    EXPECT_NEAR(Jout(0, 0), 0.0, 1e-13);
    EXPECT_NEAR(Jout(0, 1), Jbnd(0, 1), 1e-12);
    EXPECT_GT(std::abs(Jbnd(0, 0)), 1e-6); // boundary retains the endpoint slope
    // Hessian: any entry touching the clamped axis is zero; pure-axis-1 entry is not.
    auto Hout = tab.eval_hessian(q);
    EXPECT_NEAR(Hout[0](0, 0), 0.0, 1e-13);
    EXPECT_NEAR(Hout[0](0, 1), 0.0, 1e-13);
    EXPECT_NEAR(Hout[0](1, 0), 0.0, 1e-13);
    // Value is flat along axis 0 outside, so its FD partial there is ~0.
    const double e = 1e-4;
    Eigen::VectorXd qp = q, qm = q;
    qp[0] += e;
    qm[0] -= e;
    double fd0 = (tab.eval(qp)[0] - tab.eval(qm)[0]) / (2 * e);
    EXPECT_NEAR(fd0, 0.0, 1e-10);
}

TEST_F(ChebReviewTest, PeriodicDerivativeWraps1D) {
    // On a periodic axis the derivative at a wrapped query equals the derivative
    // at its in-domain equivalent (the wrap is a pure translation).
    const int n = 24;
    const double lb = 0.0, ub = 2.0 * std::numbers::pi;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    oc::ChebTable::MatType vals(n + 1, 1);
    for (int j = 0; j <= n; ++j)
        vals(j, 0) = std::cos(pts[j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, n, 1, oc::ChebTable::OutOfDomain::Periodic);
    const double t = 1.3, span = ub - lb;
    auto [v0, d0] = tab.eval_deriv1(t);
    auto [v1, d1] = tab.eval_deriv1(t + span);
    EXPECT_NEAR(d1[0], d0[0], 1e-9);
    // And it tracks the true derivative -sin(t).
    EXPECT_NEAR(d0[0], -std::sin(t), 1e-7);
}

// --- cheb_points N-D length guard + order() accessor guard -------------------

TEST_F(ChebReviewTest, ChebPointsNDLengthGuard) {
    std::vector<int> orders{5, 5, 5};
    Eigen::VectorXd lb2(2), ub2(2);
    lb2 << 0, 0;
    ub2 << 1, 1;
    EXPECT_THROW(oc::ChebTable::cheb_points(orders, lb2, ub2), std::invalid_argument);
}

TEST_F(ChebReviewTest, OrderAccessorThrowsOnND) {
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto tab = make_2d({4, 3}, lb, ub);
    EXPECT_THROW(tab.order(), std::invalid_argument);
    EXPECT_EQ(make_1d(6).order(), 6); // 1-D still works
}

// --- Threaded vs serial construction equivalence (large transform) ----------

TEST_F(ChebReviewTest, ThreadedConstructionEquivalence) {
    // A large enough 1-D DCT to let pocketfft actually consider a thread pool;
    // threaded and serial coefficients must produce identical evaluations.
    const int n = 256;
    const double lb = -3.0, ub = 4.0;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    auto f = [](double t) { return std::sin(1.3 * t) * std::exp(-0.05 * t * t); };
    oc::ChebTable::MatType vals(n + 1, 1);
    for (int j = 0; j <= n; ++j)
        vals(j, 0) = f(pts[j]);
    auto serial = oc::ChebTable::from_values(vals, lb, ub, n, /*nthreads=*/1);
    auto threaded = oc::ChebTable::from_values(vals, lb, ub, n, /*nthreads=*/4);
    for (double t : {-2.5, -1.1, 0.4, 2.2, 3.7})
        EXPECT_NEAR(serial.eval(t)[0], threaded.eval(t)[0], 1e-13) << "t=" << t;
}

// --- ChebFunction<IR> compile-time/runtime mismatch guard -------------------

TEST_F(ChebReviewTest, ChebFunctionFixedIrMismatchThrows) {
    auto tab1d = std::make_shared<oc::ChebTable>(make_1d(6));
    // Correct IR is fine.
    EXPECT_NO_THROW((oc::ChebFunction<1>(tab1d)));
    // Fixed IR != table input dim must throw.
    EXPECT_THROW((oc::ChebFunction<2>(tab1d)), std::invalid_argument);
    EXPECT_THROW((oc::ChebFunction<3>(tab1d)), std::invalid_argument);
    // Null table throws.
    std::shared_ptr<oc::ChebTable> null_tab;
    EXPECT_THROW((oc::ChebFunction<-1>(null_tab)), std::invalid_argument);
}
