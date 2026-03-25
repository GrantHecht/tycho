///////////////////////////////////////////////////////////////////////////////
// Shared VectorFunction test fixtures and helpers
//
// Fixture classes that inherit VectorFunctionFixture and are used
// across multiple VF test files, plus finite-difference Jacobian
// verification utilities.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "test_utils.h"

namespace TychoTest {

class CommonFunctionsTest : public VectorFunctionFixture {};
class GenericFunctionTest : public VectorFunctionFixture {};
class VFCompositionTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Finite-difference Jacobian and cross-check helpers
///////////////////////////////////////////////////////////////////////////////

/// Central-difference Jacobian approximation for any VectorFunction-like object.
template <class Func>
Eigen::MatrixXd fd_jacobian(Func &f, const Eigen::VectorXd &x, double eps = 1e-7) {
    int nr = f.ORows(), nc = f.IRows();
    Eigen::MatrixXd jac(nr, nc);
    Eigen::VectorXd xp = x, fp(nr), fm(nr);
    for (int j = 0; j < nc; ++j) {
        xp[j] = x[j] + eps;
        f.compute(xp, fp);
        xp[j] = x[j] - eps;
        f.compute(xp, fm);
        jac.col(j) = (fp - fm) / (2.0 * eps);
        xp[j] = x[j];
    }
    return jac;
}

/// Verify compute_jacobian output matches finite-difference approximation.
template <class Func>
void verify_jacobian_fd(Func &f, const Eigen::VectorXd &x, double tol = 1e-5) {
    int nr = f.ORows(), nc = f.IRows();
    Eigen::VectorXd fx(nr);
    Eigen::MatrixXd jx(nr, nc);
    fx.setZero();
    jx.setZero();
    f.compute_jacobian(x, fx, jx);
    Eigen::MatrixXd jx_fd = fd_jacobian(f, x);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
            EXPECT_NEAR(jx(i, j), jx_fd(i, j), tol)
                << "Jacobian mismatch at (" << i << "," << j << ")";
}

/// Same check but through GenericFunction (virtual dispatch path).
template <int IR, int OR, class Func>
void verify_gf_jacobian_fd(Func &f, const Eigen::VectorXd &x, double tol = 1e-5) {
    Tycho::GenericFunction<IR, OR> gf(f);
    Eigen::MatrixXd jx = gf.jacobian(x);
    Eigen::MatrixXd jx_fd = fd_jacobian(gf, x);
    for (int i = 0; i < gf.ORows(); ++i)
        for (int j = 0; j < gf.IRows(); ++j)
            EXPECT_NEAR(jx(i, j), jx_fd(i, j), tol)
                << "GF Jacobian mismatch at (" << i << "," << j << ")";
}

} // namespace TychoTest
