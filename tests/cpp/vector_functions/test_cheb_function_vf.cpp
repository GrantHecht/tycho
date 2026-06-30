#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>
#include <cmath>

using namespace tycho;
using namespace TychoTest;

class ChebFunctionTest : public VectorFunctionFixture {};

static std::shared_ptr<oc::ChebTable> make_table(int n, double lb, double ub) {
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    oc::ChebTable::MatType vals(1, n + 1);
    for (int j = 0; j <= n; ++j) vals(0, j) = std::sin(0.7 * pts[j]) + 0.3 * pts[j];
    return std::make_shared<oc::ChebTable>(oc::ChebTable::from_values(vals, lb, ub, n));
}

TEST_F(ChebFunctionTest, JacobianMatchesAnalytic) {
    auto tab = make_table(28, 0.0, 6.0);
    oc::ChebFunction<1> f(tab);
    for (double t : {0.5, 2.3, 4.7}) {
        Eigen::VectorXd x(1);
        x << t;
        auto [v, dv] = tab->eval_deriv1(t);
        Eigen::MatrixXd expected(1, 1);
        expected(0, 0) = dv[0];
        verify_jacobian_analytical(f, x, expected, 1e-9);
    }
}

TEST_F(ChebFunctionTest, HessianConsistency) {
    auto tab = make_table(28, 0.0, 6.0);
    oc::ChebFunction<1> f(tab);
    Eigen::VectorXd x(1);
    x << 2.3;
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(f, x, lm, 1e-9);
}

TEST_F(ChebFunctionTest, ComposesWithArguments) {
    auto tab = make_table(20, -1.0, 1.0);
    auto g = vf::GenericFunction<-1, -1>(oc::ChebFunction<-1>(tab).eval(Arguments<1>()));
    Eigen::VectorXd x(1);
    x << 0.25;
    Eigen::VectorXd out(1);
    out.setZero();
    g.compute(x, out);
    EXPECT_NEAR(out[0], tab->eval(0.25)[0], 1e-12);
}
