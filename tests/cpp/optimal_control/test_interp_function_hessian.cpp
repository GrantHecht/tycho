///////////////////////////////////////////////////////////////////////////////
// InterpFunction adjoint-Hessian finiteness (non-LGL3 FD branch)
//
// Regression for OC review §1.5: the FD step for the non-LGL3 adjoint-Hessian
// branch was `h = delta_t_ / 10.0`, but delta_t_ is only set by the even-data
// loaders -- the phase path uses load_exact_data(), which leaves delta_t_ ==
// 0.0, so h == 0 and the FD divide injected NaN into the KKT for every
// non-LGL3 table (LGL5, LGL7, Trapezoidal, CentralShooting).
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST(InterpFunctionHessian, NonLGL3AdjointHessianIsFinite) {
    auto tab = make_exact_lgl_table(TranscriptionModes::LGL5);
    ASSERT_DOUBLE_EQ(tab->delta_t_, 0.0)
        << "test precondition: load_exact_data leaves delta_t_ == 0";

    InterpFunction<-1> ifn(tab, all_state_vars(tab));

    Eigen::VectorXd x(1);
    x << 0.5 * tab->total_t_; // interior query time

    const int or_ = ifn.output_rows();
    Eigen::VectorXd fx(or_);
    Eigen::MatrixXd jx(or_, 1);
    Eigen::VectorXd adjgrad(1);
    Eigen::MatrixXd adjhess(1, 1);
    Eigen::VectorXd adjvars = Eigen::VectorXd::Ones(or_);
    fx.setZero();
    jx.setZero();
    adjgrad.setZero();
    adjhess.setZero();

    ifn.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, adjvars);

    EXPECT_TRUE(std::isfinite(adjhess(0, 0)))
        << "adjoint Hessian is NaN -- FD step h derived from delta_t_ == 0";
}
