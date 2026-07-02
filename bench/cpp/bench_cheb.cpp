///////////////////////////////////////////////////////////////////////////////
// Chebyshev table micro-benchmark.
//
// Covers:
//   - ChebTable::eval / eval_jacobian / eval_hessian at 1-D, 2-D, 3-D.
//   - ChebFunction<-1> N-D hot path:
//       compute_jacobian_adjointgradient_adjointhessian (the PSIOPT per-iteration
//       call) at 2-D and 3-D, to assess allocation overhead in the N-D Hessian
//       path (O(D^2) clenshaw_nd calls).
//
// Review item I2: compare BM_ChebFunction2D_AdjHess and
// BM_ChebFunction3D_AdjHess against BM_ChebTable2D_Hess and
// BM_ChebTable3D_Hess respectively.  If ChebFunction dominates,
// a preallocated-scratch clenshaw_nd is warranted as a Phase-3 follow-up.
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>
#include <tycho/optimal_control.h>
#include <tycho/vector_functions.h>
#include <cmath>

namespace {

// ---------------------------------------------------------------------------
// Helper: build a 1-D ChebTable of f(t) = sin(3t) on [0, 2pi], order n.
// ---------------------------------------------------------------------------
static tycho::oc::ChebTable make_cheb_1d(int order) {
    const double lb = 0.0, ub = 6.283185307179586;
    Eigen::VectorXd pts = tycho::oc::ChebTable::cheb_points(order, lb, ub);
    tycho::oc::ChebTable::MatType vals(1, order + 1);
    for (int k = 0; k <= order; ++k) vals(0, k) = std::sin(3.0 * pts[k]);
    return tycho::oc::ChebTable::from_values(vals, lb, ub, order);
}

// ---------------------------------------------------------------------------
// Helper: build a 2-D ChebTable of f(x,y) = sin(x)*cos(y) on [0,pi]^2.
// ---------------------------------------------------------------------------
static std::shared_ptr<tycho::oc::ChebTable> make_cheb_2d(int order) {
    Eigen::VectorXd lb(2), ub(2);
    lb << 0.0, 0.0;
    ub << 3.141592653589793, 3.141592653589793;
    std::vector<int> orders = {order, order};
    auto nodes = tycho::oc::ChebTable::cheb_points(orders, lb, ub);
    const int n0 = order + 1, n1 = order + 1;
    tycho::oc::ChebTable::MatType vals(n0 * n1, 1);
    for (int i = 0; i < n0; ++i)
        for (int j = 0; j < n1; ++j)
            vals(i * n1 + j, 0) = std::sin(nodes[0][i]) * std::cos(nodes[1][j]);
    return std::make_shared<tycho::oc::ChebTable>(
        tycho::oc::ChebTable::from_values(vals, lb, ub, orders));
}

// ---------------------------------------------------------------------------
// Helper: build a 3-D ChebTable of f(x,y,z) = exp(-0.5*(x^2+y^2+z^2)) on [-2,2]^3.
// ---------------------------------------------------------------------------
static std::shared_ptr<tycho::oc::ChebTable> make_cheb_3d(int order) {
    Eigen::VectorXd lb(3), ub(3);
    lb << -2.0, -2.0, -2.0;
    ub << 2.0, 2.0, 2.0;
    std::vector<int> orders = {order, order, order};
    auto nodes = tycho::oc::ChebTable::cheb_points(orders, lb, ub);
    const int n = order + 1;
    tycho::oc::ChebTable::MatType vals(n * n * n, 1);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                double x = nodes[0][i], y = nodes[1][j], z = nodes[2][k];
                vals((i * n + j) * n + k, 0) = std::exp(-0.5 * (x * x + y * y + z * z));
            }
    return std::make_shared<tycho::oc::ChebTable>(
        tycho::oc::ChebTable::from_values(vals, lb, ub, orders));
}

// ---------------------------------------------------------------------------
// 1-D eval
// ---------------------------------------------------------------------------
static void BM_ChebTable1D_Eval(benchmark::State &state) {
    auto tab = make_cheb_1d(32);
    double t = 1.23;
    for (auto _ : state) {
        auto v = tab.eval(t);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_ChebTable1D_Eval);

static void BM_ChebTable1D_Jacobian(benchmark::State &state) {
    auto tab = make_cheb_1d(32);
    double t = 1.23;
    for (auto _ : state) {
        auto [v, dv] = tab.eval_deriv1(t);
        benchmark::DoNotOptimize(v);
        benchmark::DoNotOptimize(dv);
    }
}
BENCHMARK(BM_ChebTable1D_Jacobian);

static void BM_ChebTable1D_Hessian(benchmark::State &state) {
    auto tab = make_cheb_1d(32);
    double t = 1.23;
    for (auto _ : state) {
        auto [v, dv, d2v] = tab.eval_deriv2(t);
        benchmark::DoNotOptimize(v);
        benchmark::DoNotOptimize(d2v);
    }
}
BENCHMARK(BM_ChebTable1D_Hessian);

// ---------------------------------------------------------------------------
// 2-D eval / Jacobian / Hessian (ChebTable data methods)
// ---------------------------------------------------------------------------
static void BM_ChebTable2D_Eval(benchmark::State &state) {
    auto tab = make_cheb_2d(12);
    Eigen::VectorXd x(2);
    x << 1.0, 0.5;
    for (auto _ : state) {
        auto v = tab->eval(x);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_ChebTable2D_Eval);

static void BM_ChebTable2D_Jacobian(benchmark::State &state) {
    auto tab = make_cheb_2d(12);
    Eigen::VectorXd x(2);
    x << 1.0, 0.5;
    for (auto _ : state) {
        auto J = tab->eval_jacobian(x);
        benchmark::DoNotOptimize(J);
    }
}
BENCHMARK(BM_ChebTable2D_Jacobian);

static void BM_ChebTable2D_Hess(benchmark::State &state) {
    auto tab = make_cheb_2d(12);
    Eigen::VectorXd x(2);
    x << 1.0, 0.5;
    for (auto _ : state) {
        auto H = tab->eval_hessian(x);
        benchmark::DoNotOptimize(H);
    }
}
BENCHMARK(BM_ChebTable2D_Hess);

// ---------------------------------------------------------------------------
// 3-D eval / Jacobian / Hessian (ChebTable data methods)
// ---------------------------------------------------------------------------
static void BM_ChebTable3D_Eval(benchmark::State &state) {
    auto tab = make_cheb_3d(10);
    Eigen::VectorXd x(3);
    x << 0.5, -0.3, 1.0;
    for (auto _ : state) {
        auto v = tab->eval(x);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_ChebTable3D_Eval);

static void BM_ChebTable3D_Jacobian(benchmark::State &state) {
    auto tab = make_cheb_3d(10);
    Eigen::VectorXd x(3);
    x << 0.5, -0.3, 1.0;
    for (auto _ : state) {
        auto J = tab->eval_jacobian(x);
        benchmark::DoNotOptimize(J);
    }
}
BENCHMARK(BM_ChebTable3D_Jacobian);

static void BM_ChebTable3D_Hess(benchmark::State &state) {
    auto tab = make_cheb_3d(10);
    Eigen::VectorXd x(3);
    x << 0.5, -0.3, 1.0;
    for (auto _ : state) {
        auto H = tab->eval_hessian(x);
        benchmark::DoNotOptimize(H);
    }
}
BENCHMARK(BM_ChebTable3D_Hess);

// ---------------------------------------------------------------------------
// ChebFunction<-1> N-D hot path (I2 review item):
// compute_jacobian_adjointgradient_adjointhessian — the PSIOPT per-iteration
// call.  Allocates O(D^2) clenshaw_nd work for the Hessian.
// ---------------------------------------------------------------------------
static void BM_ChebFunction2D_AdjHess(benchmark::State &state) {
    auto tab = make_cheb_2d(12);
    tycho::oc::ChebFunction<-1> f(tab);
    Eigen::VectorXd x(2);
    x << 1.0, 0.5;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, 2);
    Eigen::VectorXd adjgrad(2);
    Eigen::MatrixXd adjhess(2, 2);
    Eigen::VectorXd adjvars(1);
    adjvars << 1.0;
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();
        adjhess.setZero();
        f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, adjvars);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(adjhess);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ChebFunction2D_AdjHess);

static void BM_ChebFunction3D_AdjHess(benchmark::State &state) {
    auto tab = make_cheb_3d(10);
    tycho::oc::ChebFunction<-1> f(tab);
    Eigen::VectorXd x(3);
    x << 0.5, -0.3, 1.0;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, 3);
    Eigen::VectorXd adjgrad(3);
    Eigen::MatrixXd adjhess(3, 3);
    Eigen::VectorXd adjvars(1);
    adjvars << 1.0;
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();
        adjhess.setZero();
        f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, adjvars);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(adjhess);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ChebFunction3D_AdjHess);

} // namespace
