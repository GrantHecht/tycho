///////////////////////////////////////////////////////////////////////////////
// Norms unit tests
//
// Extracted from test_common_functions.cpp — Norms section.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <string>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Norms
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, NormValue) {
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    n.compute(x, fx);
    EXPECT_NEAR(fx[0], 5.0, 1e-14);
}

TEST_F(CommonFunctionsTest, NormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    // J = x^T / ||x|| = [3/5, 4/5, 0]
    double norm_val = 5.0;
    Eigen::MatrixXd expected(1, 3);
    expected << x[0] / norm_val, x[1] / norm_val, x[2] / norm_val;
    verify_jacobian_analytical(n, x, expected, 1e-12);
}

TEST_F(CommonFunctionsTest, SquaredNormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    // d(x^T x)/dx = 2*x^T
    Eigen::MatrixXd expected(1, 3);
    expected << 2.0, 4.0, 6.0;
    verify_jacobian_analytical(sn, x, expected, 1e-12);
}

namespace {

// Fills every derivative output of a scalar-valued VectorFunction in one call.
struct NormsDerivativeOutputs {
    Eigen::VectorXd fx_, adjgrad_;
    Eigen::MatrixXd jx_, adjhess_;
};

template <class Func>
NormsDerivativeOutputs norms_all_derivatives(Func &f, const Eigen::VectorXd &x, double lm0) {
    const int n = static_cast<int>(x.size());
    NormsDerivativeOutputs out;
    out.fx_ = Eigen::VectorXd::Zero(1);
    out.adjgrad_ = Eigen::VectorXd::Zero(n);
    out.jx_ = Eigen::MatrixXd::Zero(1, n);
    out.adjhess_ = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd lm(1);
    lm << lm0;
    f.compute_jacobian_adjointgradient_adjointhessian(x, out.fx_, out.jx_, out.adjgrad_,
                                                      out.adjhess_, lm);
    return out;
}

// The exact derivatives of ||x||^2 under an adjoint seed: J = 2 x^T, adjoint
// gradient 2 lm x, adjoint Hessian 2 lm I.
void norms_expect_squared_norm_derivatives(const NormsDerivativeOutputs &out,
                                           const Eigen::VectorXd &x, double lm0) {
    const int n = static_cast<int>(x.size());
    for (int j = 0; j < n; j++) {
        EXPECT_DOUBLE_EQ(out.jx_(0, j), 2.0 * x[j]) << "Jacobian column " << j;
        EXPECT_DOUBLE_EQ(out.adjgrad_[j], 2.0 * lm0 * x[j]) << "adjoint gradient row " << j;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            EXPECT_DOUBLE_EQ(out.adjhess_(i, j), i == j ? 2.0 * lm0 : 0.0)
                << "adjoint Hessian (" << i << "," << j << ")";
        }
    }
}

} // namespace

// The squared norm is the one norm power whose argument routinely reaches zero:
// a least-squares term evaluated at its own minimizer. Its derivatives there are
// entirely ordinary -- J = 2 x^T = 0 and adjoint Hessian 2 lm I -- and they come
// out exact rather than merely close, so these are equality assertions. They used
// to be NaN, because the shared coefficient quotient p n^p / n^2 that serves
// every integral power evaluates 0/0 at the centre of the norm.
TEST_F(CommonFunctionsTest, SquaredNormDerivativesAtItsCentreAreExact) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    const Eigen::VectorXd x = Eigen::VectorXd::Zero(3);
    const double lm0 = 1.75;

    const auto out = norms_all_derivatives(sn, x, lm0);

    EXPECT_DOUBLE_EQ(out.fx_[0], 0.0);
    norms_expect_squared_norm_derivatives(out, x, lm0);
}

// The same, through the shape a user objective actually writes: the norm as the
// outer node of a composition, evaluated exactly at the shift. This is the
// expression whose NaN reached a solver's KKT matrix.
TEST_F(CommonFunctionsTest, ShiftedSquaredNormDerivativesAtItsCentreAreExact) {
    auto args = Arguments<2>();
    const Eigen::Vector2d center(3.0, -1.5);
    auto term = (args - center).squared_norm();
    Eigen::VectorXd x = center; // exactly the centre, to the last bit
    const double lm0 = 1.75;

    const auto out = norms_all_derivatives(term, x, lm0);

    EXPECT_DOUBLE_EQ(out.fx_[0], 0.0);
    // The composition is a unit shift, so the chain rule leaves the inner
    // derivatives unchanged and the argument seen by the norm is the zero vector.
    norms_expect_squared_norm_derivatives(out, Eigen::VectorXd::Zero(2), lm0);
}

// The quotient degenerates for an argument that is merely small, too: ||x||^2
// underflows to exactly zero well before ||x|| does, taking the coefficient's
// denominator with it. The derivatives stay exact there even though the value
// itself has already underflowed.
TEST_F(CommonFunctionsTest, SquaredNormDerivativesSurviveAnUnderflowingArgument) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    Eigen::VectorXd x(3);
    x << 1.0e-200, -2.0e-200, 3.0e-200;
    const double lm0 = 1.75;
    ASSERT_DOUBLE_EQ(x.squaredNorm(), 0.0) << "premise: the squared norm underflows to zero here";

    const auto out = norms_all_derivatives(sn, x, lm0);

    norms_expect_squared_norm_derivatives(out, x, lm0);
}

// Away from the centre the derivatives are unchanged, checked both against the
// closed form across five decades of argument scale and against a central
// difference at a moderate one.
TEST_F(CommonFunctionsTest, SquaredNormDerivativesAwayFromItsCentreAreUnchanged) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    const double lm0 = 1.75;
    for (double scale : {1.0e-2, 0.25, 1.0, 7.25, 1.0e2}) {
        Eigen::VectorXd x(3);
        x << scale, -1.25 * scale, 1.5 * scale;
        SCOPED_TRACE("argument scale " + std::to_string(scale));
        norms_expect_squared_norm_derivatives(norms_all_derivatives(sn, x, lm0), x, lm0);
    }

    Eigen::VectorXd x(3);
    x << 1.0, -1.25, 1.5;
    Eigen::VectorXd lm(1);
    lm << lm0;
    verify_jacobian_fd(sn, x);
    verify_adjoint_hessian_fd(sn, x, lm);
}

TEST_F(CommonFunctionsTest, InverseNormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto inv_n = args.inverse_norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    double norm_val = 5.0;
    // d(1/||x||)/dx = -x^T / ||x||^3
    Eigen::MatrixXd expected(1, 3);
    double n3 = norm_val * norm_val * norm_val;
    expected << -x[0] / n3, -x[1] / n3, -x[2] / n3;
    verify_jacobian_analytical(inv_n, x, expected, 1e-12);
}

TEST_F(CommonFunctionsTest, InverseSquaredNormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto isn = args.inverse_squared_norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    double n2 = x.squaredNorm();
    // d(1/||x||^2)/dx = -2*x^T / ||x||^4
    Eigen::MatrixXd expected(1, 3);
    double n4 = n2 * n2;
    expected << -2.0 * x[0] / n4, -2.0 * x[1] / n4, -2.0 * x[2] / n4;
    verify_jacobian_analytical(isn, x, expected, 1e-12);
}

TEST_F(CommonFunctionsTest, NormPowerValue) {
    auto args = Arguments<3>();
    auto np = args.norm_power<3>();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    np.compute(x, fx);
    EXPECT_NEAR(fx[0], 125.0, 1e-10); // 5^3 = 125
}

TEST_F(CommonFunctionsTest, NormPowerAdjointConsistency) {
    auto args = Arguments<3>();
    auto np = args.norm_power<3>();
    Eigen::VectorXd x = deterministic_random_vector(3, 50, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(np, x, lm);
}

TEST_F(CommonFunctionsTest, NormAdjointConsistencyRandomInput) {
    auto args = Arguments<5>();
    auto n = args.norm();
    Eigen::VectorXd x = deterministic_random_vector(5, 51, 1.0, 10.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(n, x, lm);
}
