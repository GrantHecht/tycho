///////////////////////////////////////////////////////////////////////////////
// Jacobian Aliasing Diagnostic
//
// Reproduces the TwoFunctionSum Jacobian aliasing bug documented in
// bug-fix/ANALYSIS.md
//
// Tests several VectorFunction expression patterns and checks whether
// their Jacobians are computed correctly. Prints pointer diagnostics
// to help identify aliasing.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Tycho;

// Finite-difference Jacobian for verification
template <int IR, int OR>
Eigen::MatrixXd fd_jacobian(GenericFunction<IR, OR> &gf, const Eigen::VectorXd &x,
                            double eps = 1e-7) {
    int nr = gf.ORows();
    int nc = gf.IRows();
    Eigen::MatrixXd jac(nr, nc);
    Eigen::VectorXd xp = x;
    Eigen::VectorXd fp(nr), fm(nr);
    for (int j = 0; j < nc; ++j) {
        xp[j] = x[j] + eps;
        gf.compute(xp, fp);
        xp[j] = x[j] - eps;
        gf.compute(xp, fm);
        jac.col(j) = (fp - fm) / (2.0 * eps);
        xp[j] = x[j];
    }
    return jac;
}

bool check_jacobian(const char *label, const Eigen::MatrixXd &analytic,
                    const Eigen::MatrixXd &fd, double tol = 1e-5) {
    double err = (analytic - fd).norm();
    bool pass = err < tol;
    std::cout << (pass ? "PASS" : "FAIL") << "  " << label << "\n";
    if (!pass) {
        std::cout << "  Analytic: " << analytic << "\n";
        std::cout << "  FD:       " << fd << "\n";
        std::cout << "  Error:    " << err << "\n";
    }
    return pass;
}

int main() {
    std::cout << "===== Jacobian Aliasing Diagnostic =====\n\n";

    int failures = 0;

    // Test point
    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;

    // ---- Test 1: Simple sum x + y (should work) ----
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto f = x + y;

        GenericFunction<2, 1> gf(f);
        Eigen::MatrixXd jac = gf.jacobian(tp);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("x + y", jac, jac_fd))
            ++failures;
    }

    // ---- Test 2: Scaled sum 2*(x+y) (should work) ----
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto f = 2.0 * (x + y);

        GenericFunction<2, 1> gf(f);
        Eigen::MatrixXd jac = gf.jacobian(tp);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("2*(x+y)", jac, jac_fd))
            ++failures;
    }

    // ---- Test 3: cos(2*(x+y)) — the minimal reproducer ----
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto ang = 2.0 * (x + y);
        auto f = cos(ang);

        GenericFunction<2, 1> gf(f);
        Eigen::MatrixXd jac = gf.jacobian(tp);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("cos(2*(x+y))", jac, jac_fd))
            ++failures;
    }

    // ---- Test 4: sin(x) * cos(2*(x+y)) — complex expression ----
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto r = sqrt(x * x + y * y);
        auto ang = 2.0 * (x + y);
        auto f = sin(r) * cos(ang);

        GenericFunction<2, 1> gf(f);
        Eigen::MatrixXd jac = gf.jacobian(tp);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("sin(r)*cos(2*(x+y))", jac, jac_fd))
            ++failures;
    }

    // ---- Test 5: cos(2*(x+y)) via two-arg jacobian (uninitialized jac) ----
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto ang = 2.0 * (x + y);
        auto f = cos(ang);

        GenericFunction<2, 1> gf(f);
        Eigen::MatrixXd jac(1, 2);
        // Deliberately NOT zeroing jac — tests whether DirectAssignment
        // correctly initializes all columns
        gf.jacobian(tp, jac);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("cos(2*(x+y)) [2-arg, uninitialized]", jac, jac_fd))
            ++failures;
    }

    // ---- Test 6: cos(2*(x+y)) via two-arg jacobian (zeroed jac) ----
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto ang = 2.0 * (x + y);
        auto f = cos(ang);

        GenericFunction<2, 1> gf(f);
        Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(1, 2);
        gf.jacobian(tp, jac);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("cos(2*(x+y)) [2-arg, zeroed]", jac, jac_fd))
            ++failures;
    }

    // ---- Test 7: Stacked cos(2*(x+y)); sin(x+y) ----
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto ang = 2.0 * (x + y);
        auto f1 = cos(ang);
        auto f2 = sin(x + y);
        auto f = stack(f1, f2);

        GenericFunction<2, 2> gf(f);
        Eigen::MatrixXd jac = gf.jacobian(tp);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("stack(cos(2*(x+y)), sin(x+y))", jac, jac_fd))
            ++failures;
    }

    // ---- Test 8: Direct (non-GenericFunction) Jacobian ----
    // This tests whether the bug manifests without virtual dispatch
    {
        auto a = Arguments<2>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto ang = 2.0 * (x + y);
        auto f = cos(ang);

        Eigen::Matrix<double, 1, 1> fx;
        Eigen::Matrix<double, 1, 2> jx;
        jx.setZero();
        f.compute_jacobian(tp, fx, jx);

        // Compute FD for comparison
        GenericFunction<2, 1> gf(f);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp);

        if (!check_jacobian("cos(2*(x+y)) [direct, no GF]", jx, jac_fd))
            ++failures;
    }

    // ---- Test 9: 3-variable sum inside nested function ----
    {
        auto a = Arguments<3>();
        auto x = a.coeff<0>();
        auto y = a.coeff<1>();
        auto z = a.coeff<2>();
        auto f = cos(x + y + z);

        GenericFunction<3, 1> gf(f);
        Eigen::VectorXd tp3(3);
        tp3 << 0.2, -0.5, 0.3;
        Eigen::MatrixXd jac = gf.jacobian(tp3);
        Eigen::MatrixXd jac_fd = fd_jacobian(gf, tp3);

        if (!check_jacobian("cos(x+y+z)", jac, jac_fd))
            ++failures;
    }

    std::cout << "\n===== Summary: " << failures << " failure(s) =====\n";
    return failures > 0 ? 1 : 0;
}
