# Kepler Propagator LCD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `KeplerPropagator` (DSL) and `propagate_cartesian` (Newton) with a single canonical EMTG-style Laguerre-Conway-Der implementation. Hand-rolled iteration kernel + SymPy-codegen'd closed-form primal/residual + small IFT composition layer.

**Architecture:** Five files, one canonical algorithm. Iteration kernel (hand-written, scalar/SuperScalar templated) drives F→0. SymPy script emits `KeplerPrimal` and `KeplerResidual` with CSE'd partials of the post-converged closed forms. IFT layer composes total Jac/adjoint-Hess via `dX*/dy = −F_y/F_X` and the U-recursion chain rule. Public `KeplerPropagator` VF and `propagate_cartesian` both delegate to the kernel. Per-lane SuperScalar in this PR; true SIMD blend deferred.

**Tech Stack:** C++20, Eigen 5, SymPy (existing `utils/CodeGen.py`), Google Test, Google Benchmark.

**Spec:** `doc/plans/2026-05-02-kepler-propagator-lcd-design.md`

**Branch:** `feat/kepler-propagator-lcd` (already created off main).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `include/tycho/detail/astro/kepler_lcd_iterate.h` | NEW | Iteration kernel; `kepler_lcd_iterate<Scalar>` + `KeplerLCDResult<Scalar>` |
| `utils/codegen_kepler_propagator.py` | NEW | SymPy script that emits the closed-form headers |
| `include/tycho/detail/astro/kepler_primal.h` | NEW (generated) | `KeplerPrimal` — internal helper |
| `include/tycho/detail/astro/kepler_residual.h` | NEW (generated) | `KeplerResidual` — internal helper |
| `include/tycho/detail/astro/kepler_propagator_ift.h` | NEW | IFT composition: `kepler_propagate`, `kepler_propagate_jacobian`, `kepler_propagate_adjoint_hessian` |
| `include/tycho/detail/astro/kepler_propagator.h` | REWRITTEN | Public `KeplerPropagator` VF; delegates to IFT layer |
| `include/tycho/detail/astro/kepler_utils.h` | MODIFIED | `propagate_cartesian` body replaced; rest unchanged |
| `tests/cpp/astro/test_kepler_lcd.cpp` | NEW | Iteration-kernel tests (scalar + per-lane SS) |
| `tests/cpp/astro/test_kepler_ift.cpp` | NEW | IFT layer tests (Jac/Hess vs FD across orbit fixtures) |
| `tests/cpp/astro/test_kepler_propagation.cpp` | unchanged | Existing tests must continue to pass |
| `tests/cpp/CMakeLists.txt` | MODIFIED | Register new test files |
| `bench/cpp/kepler/bench_kepler.cpp` | MODIFIED | Add 4 VF benchmarks |

---

## Sequencing

```
0. Iteration kernel (scalar Scalar=double)
1. Iteration kernel (per-lane SuperScalar)
2. Codegen script + generated header  
3. IFT layer — Jacobian
4. IFT layer — adjoint Hessian
5. Rewrite propagate_cartesian (consumes kernel)
6. Rewrite KeplerPropagator VF (consumes IFT layer)
7. Add benchmarks
8. Pre-merge verification gate
```

Tasks 0–4 are pure additions (no existing file modified). Task 5 modifies `kepler_utils.h`. Task 6 rewrites `kepler_propagator.h`. Tasks 7–8 are bench/test wiring.

---

## Task 0: Iteration kernel (scalar)

**Goal:** Header-only `kepler_lcd_iterate<double>` that drives the universal-variable residual to zero via Laguerre-Conway-Der at order N=1..10. Returns `KeplerLCDResult<double>` with X*, α, r₀, σ₀, r, σ, U₀..U₃, and `converged`.

**Files:**
- Create: `include/tycho/detail/astro/kepler_lcd_iterate.h`
- Create: `tests/cpp/astro/test_kepler_lcd.cpp`
- Modify: `tests/cpp/CMakeLists.txt` (register new test)

**Acceptance Criteria:**
- [ ] `kepler_lcd_iterate<double>(R0, V0, dt, mu)` returns identical state to existing `propagate_cartesian<double>` (within 1e-10 rel) for LEO circular, MEO e=0.5, hyperbolic e=1.5.
- [ ] δt=0 returns X=0, U_n at X=0, `converged=true`.
- [ ] Hyperbolic asymptote-guard hit (very large δt on hyperbolic) returns `converged=false`.
- [ ] Multi-period (5T) round-trip on LEO matches existing test tolerance (1e-4 km).

**Verify:** `cd build && ninja -j6 tycho_tests && ctest --test-dir tests/cpp/astro -R KeplerLCDKernel --output-on-failure`

**Steps:**

- [ ] **Step 1: Create header skeleton with options + result types**

```cpp
// include/tycho/detail/astro/kepler_lcd_iterate.h
#pragma once
#include "tycho/detail/typedefs/eigen_types.h"
#include <cmath>

namespace tycho::astro::detail {

struct KeplerLCDOptions {
    double Xtol = 1.0e-12;
    double alpha_tol = 1.0e-12;
    int max_order = 10;
    int iters_per_order = 10;
    double hyp_guard = 30.0;
};

template <class Scalar>
struct KeplerLCDResult {
    Scalar X;
    Scalar alpha;
    Scalar r0;
    Scalar sigma0;
    Scalar r;
    Scalar sigma;
    Scalar U0, U1, U2, U3;
    bool converged;
};

// compute_universal_functions: orbit-type branch lives here.
// alpha > alpha_tol  -> ellipse via Stumpff
// alpha < -alpha_tol -> hyperbola via cosh/sinh
// |alpha| <= alpha_tol -> parabola
template <class Scalar>
inline void compute_universal_functions(Scalar alpha, Scalar X, double alpha_tol,
                                        Scalar& U0, Scalar& U1, Scalar& U2, Scalar& U3);

template <class Scalar>
KeplerLCDResult<Scalar>
kepler_lcd_iterate(const Vector3<Scalar>& R0,
                   const Vector3<Scalar>& V0,
                   Scalar dt, double mu,
                   const KeplerLCDOptions& opts = {});

} // namespace tycho::astro::detail
```

- [ ] **Step 2: Implement `compute_universal_functions` for `Scalar = double`**

Mirrors `emtg_kepler.cpp:117-146` exactly:

```cpp
namespace tycho::astro::detail {

template <>
inline void compute_universal_functions<double>(double alpha, double X, double alpha_tol,
                                                double& U0, double& U1, double& U2, double& U3) {
    using std::cos; using std::sin; using std::cosh; using std::sinh; using std::sqrt;
    if (alpha > alpha_tol) {                         // ellipse
        const double y = alpha * X * X;
        const double sq = sqrt(y);
        const double C = (1.0 - cos(sq)) / y;
        const double S = (sq - sin(sq)) / (sq * y);   // = (sqrt(y) - sin(sqrt(y))) / sqrt(y^3)
        U1 = X * (1.0 - y * S);
        U2 = X * X * C;
        U3 = X * X * X * S;
        U0 = 1.0 - alpha * U2;
    } else if (alpha < -alpha_tol) {                 // hyperbola
        const double sqma = sqrt(-alpha);
        const double sqmaX = sqma * X;
        U0 = cosh(sqmaX);
        U1 = sinh(sqmaX) / sqma;
        U2 = (1.0 - U0) / alpha;
        U3 = (X - U1) / alpha;
    } else {                                          // parabola
        U0 = 1.0;
        U1 = X;
        U2 = U1 * X / 2.0;
        U3 = U2 * X / 3.0;
    }
}

} // namespace tycho::astro::detail
```

- [ ] **Step 3: Implement `kepler_lcd_iterate<double>` body**

Mirror EMTG's iteration loop, with the changes from spec §3.1.

```cpp
template <>
inline KeplerLCDResult<double>
kepler_lcd_iterate<double>(const Vector3<double>& R0,
                            const Vector3<double>& V0,
                            double dt, double mu,
                            const KeplerLCDOptions& opts) {
    using std::sqrt; using std::fabs; using std::log;
    KeplerLCDResult<double> r;
    r.converged = true;
    const double sqmu = sqrt(mu);
    const double r0 = R0.norm();
    const double v2 = V0.squaredNorm();
    const double sigma0 = R0.dot(V0) / sqmu;
    const double alpha = 2.0 / r0 - v2 / mu;
    r.r0 = r0;
    r.sigma0 = sigma0;
    r.alpha = alpha;

    if (dt == 0.0) {
        r.X = 0.0;
        compute_universal_functions<double>(alpha, 0.0, opts.alpha_tol, r.U0, r.U1, r.U2, r.U3);
        r.r = r0;
        r.sigma = sigma0;
        return r;
    }

    // Initial guess (from existing propagate_cartesian)
    double X;
    if (alpha > opts.alpha_tol) {
        X = sqmu * dt * alpha;
    } else if (alpha < -opts.alpha_tol) {
        const double a = 1.0 / alpha;
        const double sgn_dt = dt >= 0.0 ? 1.0 : -1.0;
        X = sgn_dt * sqrt(-a) *
            log(fabs((-2.0 * mu * alpha * dt) /
                     (R0.dot(V0) + sgn_dt * sqrt(-mu * a) * (1.0 - r0 * alpha))));
    } else {
        // parabolic seed (from kepler_utils.h:550-556)
        const Vector3<double> h = R0.cross(V0);
        const double hmag = h.norm();
        const double p = hmag * hmag / mu;
        const double s = (M_PI / 2.0 - std::atan(3.0 * sqrt(mu / (p * p * p)) * dt)) / 2.0;
        const double w = std::atan(std::cbrt(std::tan(s)));
        X = sqrt(p) * 2.0 / std::tan(2.0 * w);
    }

    double X_new = X;
    int N = 1;
    int iters_this_N = 0;
    double dX = 1e+100;
    double U0 = 0, U1 = 0, U2 = 0, U3 = 0;
    double rad = r0, sig = sigma0;
    double F = 0;

    while (fabs(dX) > opts.Xtol && N <= opts.max_order) {
        ++iters_this_N;
        if (iters_this_N >= opts.iters_per_order) {
            ++N;
            iters_this_N = 0;
            if (N > opts.max_order) {
                r.converged = false;
                break;
            }
        }
        X = X_new;

        // Hyperbolic asymptote guard
        if (alpha < -opts.alpha_tol) {
            const double sqma = sqrt(-alpha);
            if (fabs(sqma * X) > opts.hyp_guard) {
                r.converged = false;
                break;
            }
        }

        compute_universal_functions<double>(alpha, X, opts.alpha_tol, U0, U1, U2, U3);
        rad = r0 * U0 + sigma0 * U1 + U2;
        sig = sigma0 * U0 + (1.0 - alpha * r0) * U1;
        F = r0 * U1 + sigma0 * U2 + U3 - sqmu * dt;
        const double F1 = rad;
        const double F2 = sig;

        const double sgn = F1 >= 0 ? 1.0 : -1.0;
        const double disc = (N - 1) * (N - 1) * F1 * F1 - N * (N - 1) * F * F2;
        const double denom = fabs(disc);
        if (denom > 0.0)
            dX = N * F / (F1 + sgn * sqrt(denom));
        else
            dX = F / F1;
        X_new = X - dX;
    }

    // Final state at converged X (or last iterate if not converged)
    X = X_new;
    compute_universal_functions<double>(alpha, X, opts.alpha_tol, U0, U1, U2, U3);
    rad = r0 * U0 + sigma0 * U1 + U2;
    sig = sigma0 * U0 + (1.0 - alpha * r0) * U1;

    r.X = X;
    r.U0 = U0; r.U1 = U1; r.U2 = U2; r.U3 = U3;
    r.r = rad;
    r.sigma = sig;
    return r;
}
```

- [ ] **Step 4: Write failing tests**

```cpp
// tests/cpp/astro/test_kepler_lcd.cpp
#include <tycho/detail/astro/kepler_lcd_iterate.h>
#include <tycho/detail/astro/kepler_utils.h>     // for ground-truth propagate_cartesian
#include "astro_test_utils.h"                    // leoClassic, MU_EARTH
#include <gtest/gtest.h>
#include <numbers>

using namespace tycho;
using namespace tycho::astro;
using namespace tycho::astro::detail;

namespace {
Vector6<double> apply_fg(const Vector6<double>& RV, const KeplerLCDResult<double>& k,
                        double mu, double dt) {
    const double sqmu = std::sqrt(mu);
    const double aF  = 1.0 - k.U2 / k.r0;
    const double aG  = (k.r0 * k.U1 + k.sigma0 * k.U2) / sqmu;
    const double aFt = -sqmu / (k.r0 * k.r) * k.U1;
    const double aGt = 1.0 - k.U2 / k.r;
    Vector6<double> out;
    out.head<3>() = aF  * RV.head<3>() + aG  * RV.tail<3>();
    out.tail<3>() = aFt * RV.head<3>() + aGt * RV.tail<3>();
    return out;
}
} // anon

TEST(KeplerLCDKernel, MatchesPropagateCartesianLEO) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv_ref = propagate_cartesian<double>(rv, 300.0, TychoTest::MU_EARTH);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH, 300.0);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_lcd[i], rv_ref[i], 1e-10 * std::max(1.0, std::abs(rv_ref[i])))
            << "comp " << i;
}

TEST(KeplerLCDKernel, MatchesPropagateCartesianMEO) {
    Vector6<double> oe;
    oe << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 600.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv_ref = propagate_cartesian<double>(rv, 600.0, TychoTest::MU_EARTH);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH, 600.0);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_lcd[i], rv_ref[i], 1e-10 * std::max(1.0, std::abs(rv_ref[i])));
}

TEST(KeplerLCDKernel, MatchesPropagateCartesianHyperbolic) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv_ref = propagate_cartesian<double>(rv, 100.0, TychoTest::MU_EARTH);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH, 100.0);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_lcd[i], rv_ref[i], 1e-10 * std::max(1.0, std::abs(rv_ref[i])));
}

TEST(KeplerLCDKernel, ZeroDtFastPath) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 0.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    EXPECT_DOUBLE_EQ(k.X, 0.0);
    EXPECT_DOUBLE_EQ(k.U0, 1.0);
    EXPECT_DOUBLE_EQ(k.U1, 0.0);
    EXPECT_DOUBLE_EQ(k.U2, 0.0);
    EXPECT_DOUBLE_EQ(k.U3, 0.0);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH, 0.0);
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(rv_lcd[i], rv[i], 1e-13);
}

TEST(KeplerLCDKernel, HyperbolicAsymptoteGuard) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    // 1e8 s on a hyperbolic with v ~ 10 km/s drives X past the |sqma*X|>30 guard.
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 1.0e8, TychoTest::MU_EARTH);
    EXPECT_FALSE(k.converged);
}

TEST(KeplerLCDKernel, MultiPeriodReturn) {
    auto oe = TychoTest::leoClassic();
    double a = oe[0];
    double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / TychoTest::MU_EARTH);
    auto rv0 = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv0.head<3>(), rv0.tail<3>(), 5.0 * T, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv5 = apply_fg(rv0, k, TychoTest::MU_EARTH, 5.0 * T);
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(rv0[i], rv5[i], 1e-4);
}
```

- [ ] **Step 5: Register the new test in CMakeLists.txt**

Edit `tests/cpp/CMakeLists.txt` — append `test_kepler_lcd.cpp` to the same list that contains `test_kepler_propagation.cpp`. Locate the pattern that registers astro tests and add the new file alongside.

- [ ] **Step 6: Build and run tests**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 tycho_tests \
  && ctest --test-dir tests/cpp/astro -R KeplerLCDKernel --output-on-failure
```

Expected: all 6 `KeplerLCDKernel.*` tests pass.

- [ ] **Step 7: Commit**

```bash
git add include/tycho/detail/astro/kepler_lcd_iterate.h \
        tests/cpp/astro/test_kepler_lcd.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(astro): scalar Laguerre-Conway-Der Kepler iteration kernel

New kepler_lcd_iterate.h implements the EMTG-style universal-variable
LCD iteration as a hand-written kernel returning X*, U0..U3, r, sigma,
plus a converged flag. Scalar (double) specialization only at this stage;
SuperScalar follows in next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Iteration kernel (per-lane SuperScalar)

**Goal:** Specialize `kepler_lcd_iterate<Eigen::Array<double, W, 1>>` to a per-lane scalar dispatch loop. Closed-form post-processing remains naturally packed across lanes.

**Files:**
- Modify: `include/tycho/detail/astro/kepler_lcd_iterate.h` (add SS template specialization)
- Modify: `tests/cpp/astro/test_kepler_lcd.cpp` (add SS tests)

**Acceptance Criteria:**
- [ ] `kepler_lcd_iterate<Eigen::Array<double,4,1>>` returns per-lane results equal to four `kepler_lcd_iterate<double>` calls.
- [ ] Mixed-orbit-type lanes (lane 0 LEO, lane 1 MEO, lane 2 hyperbolic, lane 3 parabolic) all converge correctly per-lane.
- [ ] Compiles with `is_vectorizable=true` exposure (smoke test by including from a downstream TU).

**Verify:** `ctest --test-dir tests/cpp/astro -R KeplerLCDKernelSS --output-on-failure`

**Steps:**

- [ ] **Step 1: Add per-lane specialization**

Append to `kepler_lcd_iterate.h`:

```cpp
namespace tycho::astro::detail {

template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>>
kepler_lcd_iterate_per_lane(const Vector3<Eigen::Array<double, W, 1>>& R0,
                            const Vector3<Eigen::Array<double, W, 1>>& V0,
                            const Eigen::Array<double, W, 1>& dt, double mu,
                            const KeplerLCDOptions& opts) {
    using SS = Eigen::Array<double, W, 1>;
    KeplerLCDResult<SS> out;
    out.converged = true;
    for (int lane = 0; lane < W; ++lane) {
        Vector3<double> R0_l, V0_l;
        for (int i = 0; i < 3; ++i) {
            R0_l[i] = R0[i][lane];
            V0_l[i] = V0[i][lane];
        }
        auto k = kepler_lcd_iterate<double>(R0_l, V0_l, dt[lane], mu, opts);
        out.X[lane]      = k.X;
        out.alpha[lane]  = k.alpha;
        out.r0[lane]     = k.r0;
        out.sigma0[lane] = k.sigma0;
        out.r[lane]      = k.r;
        out.sigma[lane]  = k.sigma;
        out.U0[lane] = k.U0; out.U1[lane] = k.U1;
        out.U2[lane] = k.U2; out.U3[lane] = k.U3;
        out.converged = out.converged && k.converged;
    }
    return out;
}

template <int W>
inline KeplerLCDResult<Eigen::Array<double, W, 1>>
kepler_lcd_iterate(const Vector3<Eigen::Array<double, W, 1>>& R0,
                   const Vector3<Eigen::Array<double, W, 1>>& V0,
                   const Eigen::Array<double, W, 1>& dt, double mu,
                   const KeplerLCDOptions& opts = {}) {
    return kepler_lcd_iterate_per_lane<W>(R0, V0, dt, mu, opts);
}

} // namespace tycho::astro::detail
```

Note: the SS overload is a separate template (different parameter types — `const Eigen::Array<...>& dt` vs `Scalar dt`), so it does not require `template<>` explicit specialization syntax. It overload-resolves when the caller passes Array Scalars.

- [ ] **Step 2: Add SS test**

Append to `test_kepler_lcd.cpp`:

```cpp
TEST(KeplerLCDKernelSS, MixedOrbitsFourLanes) {
    using SS = Eigen::Array<double, 4, 1>;
    Vector3<SS> R0, V0;
    SS dt;

    // Lane 0: LEO
    auto rv0 = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    // Lane 1: MEO e=0.5
    Vector6<double> oe1; oe1 << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    auto rv1 = classic_to_cartesian<double>(oe1, TychoTest::MU_EARTH);
    // Lane 2: hyperbolic
    Vector6<double> oe2; oe2 << -10000.0, 1.5, 0.2, 0.0, 0.0, 0.0;
    auto rv2 = classic_to_cartesian<double>(oe2, TychoTest::MU_EARTH);
    // Lane 3: near-parabolic
    Vector6<double> oe3; oe3 << 1.0e7, 0.999999, 0.1, 0.0, 0.0, 0.0;
    auto rv3 = classic_to_cartesian<double>(oe3, TychoTest::MU_EARTH);

    for (int i = 0; i < 3; ++i) {
        R0[i] << rv0[i],     rv1[i],     rv2[i],     rv3[i];
        V0[i] << rv0[i + 3], rv1[i + 3], rv2[i + 3], rv3[i + 3];
    }
    dt << 100.0, 600.0, 50.0, 200.0;

    auto k_ss = kepler_lcd_iterate<Eigen::Array<double, 4, 1>>(R0, V0, dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(k_ss.converged);

    Vector6<double> rv_arr[4] = { rv0, rv1, rv2, rv3 };
    double dt_arr[4] = { 100.0, 600.0, 50.0, 200.0 };
    for (int lane = 0; lane < 4; ++lane) {
        auto k = kepler_lcd_iterate<double>(rv_arr[lane].head<3>(),
                                            rv_arr[lane].tail<3>(),
                                            dt_arr[lane], TychoTest::MU_EARTH);
        EXPECT_NEAR(k_ss.X[lane],     k.X,     1e-12) << "X lane " << lane;
        EXPECT_NEAR(k_ss.U0[lane],    k.U0,    1e-12) << "U0 lane " << lane;
        EXPECT_NEAR(k_ss.U1[lane],    k.U1,    1e-12) << "U1 lane " << lane;
        EXPECT_NEAR(k_ss.U2[lane],    k.U2,    1e-12) << "U2 lane " << lane;
        EXPECT_NEAR(k_ss.U3[lane],    k.U3,    1e-12) << "U3 lane " << lane;
        EXPECT_NEAR(k_ss.r[lane],     k.r,     1e-12) << "r lane " << lane;
        EXPECT_NEAR(k_ss.sigma[lane], k.sigma, 1e-12) << "sigma lane " << lane;
    }
}
```

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 tycho_tests \
  && ctest --test-dir tests/cpp/astro -R KeplerLCDKernelSS --output-on-failure
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/astro/kepler_lcd_iterate.h tests/cpp/astro/test_kepler_lcd.cpp
git commit -m "$(cat <<'EOF'
feat(astro): per-lane SuperScalar dispatch for Kepler LCD kernel

Eigen::Array<double, W, 1> overload runs the scalar iteration once per
lane and packs results back. Closed-form post-processing remains naturally
SIMD-packed since it is branch-free. Phase 2 (true SIMD masked-blend
iteration) is deferred per design spec section 3.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Codegen script + generated header

**Goal:** SymPy script `utils/codegen_kepler_propagator.py` emits `include/tycho/detail/astro/kepler_propagator_closed_form.h` containing two `TychoHeaderGen`-produced structs: `KeplerPrimal` (S, 11→6) and `KeplerResidual` (F, 10→1). Both with full Jac+adjoint-Hess.

**Files:**
- Create: `utils/codegen_kepler_propagator.py`
- Create (generated): `include/tycho/detail/astro/kepler_propagator_closed_form.h`
- Create: smoke test in `tests/cpp/astro/test_kepler_lcd.cpp` (extend) — verify the generated header compiles and that `KeplerPrimal::compute_impl` reproduces the expected closed-form output for a known fixture.

**Acceptance Criteria:**
- [ ] `conda run -n tycho python utils/codegen_kepler_propagator.py` exits 0 and writes `kepler_propagator_closed_form.h`.
- [ ] Generated header has no `pow(` calls in compute bodies (TychoHeaderGen guard).
- [ ] Generated header compiles when included from a TU.
- [ ] `KeplerPrimal{mu}.compute_impl(input, out)` for a hand-crafted input vector matches the same closed form computed inline in C++ to 1e-13 relative.

**Verify:** `ctest --test-dir tests/cpp/astro -R KeplerClosedForm --output-on-failure`

**Steps:**

- [ ] **Step 1: Write the codegen script**

```python
# utils/codegen_kepler_propagator.py
"""Generate KeplerPrimal and KeplerResidual closed-form helpers.

Usage:
    cd utils && conda run -n tycho python codegen_kepler_propagator.py
"""
import os
import sympy as sp
from CodeGen import TychoHeaderGen


def _common_symbols():
    R0 = sp.symbols("R0_:3")
    V0 = sp.symbols("V0_:3")
    dt, X = sp.symbols("dt X")
    U0, U1, U2, U3 = sp.symbols("U0 U1 U2 U3")
    mu = sp.symbols("mu")
    return R0, V0, dt, X, U0, U1, U2, U3, mu


def _kepler_primal():
    """S(R0, V0, dt, X, U0, U1, U2) -> (rf, vf). 11 inputs, 6 outputs."""
    R0, V0, dt, X, U0, U1, U2, _U3, mu = _common_symbols()
    r0 = sp.sqrt(R0[0]**2 + R0[1]**2 + R0[2]**2)
    sigma0 = (R0[0]*V0[0] + R0[1]*V0[1] + R0[2]*V0[2]) / sp.sqrt(mu)
    r = r0*U0 + sigma0*U1 + U2

    aF  = 1 - U2/r0
    aG  = (r0*U1 + sigma0*U2) / sp.sqrt(mu)
    aFt = -sp.sqrt(mu)/(r0*r) * U1
    aGt = 1 - U2/r

    rf = sp.Matrix([aF*R0[i] + aG*V0[i] for i in range(3)])
    vf = sp.Matrix([aFt*R0[i] + aGt*V0[i] for i in range(3)])

    Xs = sp.Matrix(list(R0) + list(V0) + [dt, X, U0, U1, U2])
    Func = sp.Matrix.vstack(rf, vf)
    return TychoHeaderGen(
        "KeplerPrimal", Func, Xs,
        [(mu, "Gravitational parameter", "mu > 0.0")],
        docstr="Kepler primal map post-converged: (R0,V0,dt,X*,U0..U2) -> (rf,vf)",
        gen_build_method=False,
        is_vectorizable=True,
    )


def _kepler_residual():
    """F(R0, V0, dt, U1, U2, U3) -> [F]. 10 inputs, 1 output. F linear in U_n."""
    R0, V0, dt, _X, _U0, U1, U2, U3, mu = _common_symbols()
    r0 = sp.sqrt(R0[0]**2 + R0[1]**2 + R0[2]**2)
    sigma0 = (R0[0]*V0[0] + R0[1]*V0[1] + R0[2]*V0[2]) / sp.sqrt(mu)
    F = r0*U1 + sigma0*U2 + U3 - sp.sqrt(mu)*dt

    Xs = sp.Matrix(list(R0) + list(V0) + [dt, U1, U2, U3])
    return TychoHeaderGen(
        "KeplerResidual", sp.Matrix([F]), Xs,
        [(mu, "Gravitational parameter", "mu > 0.0")],
        docstr="Kepler universal-variable residual F = r0*U1 + sigma0*U2 + U3 - sqrt(mu)*dt",
        gen_build_method=False,
        is_vectorizable=True,
    )


def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )

    # TychoHeaderGen.make_header() writes one file per call, named after the
    # struct in snake_case. We want both structs in the same header, so we
    # write them sequentially into a combined output path.
    primal = _kepler_primal()
    residual = _kepler_residual()

    # Build the combined text manually using TychoHeaderGen internals.
    primal_text   = primal._build_text(script_name="utils/codegen_kepler_propagator.py")
    residual_text = residual._build_text(script_name="utils/codegen_kepler_propagator.py")

    # Strip the residual's preamble (#pragma once / includes / namespace open)
    # so we can append it after the primal's namespace open. TychoHeaderGen
    # emits each struct as a full file; we splice them.
    import re
    # Find namespace block content in residual
    ns_match = re.search(r"namespace tycho::astro \{(.*)\} // namespace tycho::astro",
                         residual_text, re.DOTALL)
    if not ns_match:
        raise RuntimeError("Could not locate namespace block in generated residual text")
    residual_body = ns_match.group(1)

    # Drop the using-decls header from residual_body (already in primal's block)
    using_pattern = re.compile(r"// Import cross-namespace types.*?(?=struct )", re.DOTALL)
    residual_body = using_pattern.sub("", residual_body, count=1)

    # Insert residual_body before the closing `} // namespace tycho::astro` of primal
    combined = primal_text.replace(
        "}; \n\n} // namespace tycho::astro\n",
        "}; \n\n" + residual_body + "\n} // namespace tycho::astro\n",
    )
    # Defensive: if the exact closer is not found, fall back to inserting
    # before the LAST `} // namespace tycho::astro` and warn loudly.
    if "}; \n\n" + residual_body[:30] not in combined:
        # try alternative — locate '}\n\n} // namespace tycho::astro\n' more loosely
        m = re.search(r"\};\s*\n\s*\n\} // namespace tycho::astro\s*\n", primal_text)
        if not m:
            raise RuntimeError("Could not splice residual into primal namespace block")
        combined = primal_text[:m.start()] + "};\n\n" + residual_body + "\n} // namespace tycho::astro\n"

    out_path = os.path.join(output_dir, "kepler_propagator_closed_form.h")
    with open(out_path + ".tmp", "w") as f:
        f.write(combined)
    os.replace(out_path + ".tmp", out_path)
    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
```

Note: the splice approach above is brittle — see Step 1b for a cleaner alternative if it fails.

- [ ] **Step 1b (fallback if splice fails): emit two separate headers and have the IFT layer include both**

If the splice in Step 1 proves fragile, replace `main()` with:

```python
def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    _kepler_primal().make_header(
        output_dir=output_dir,
        script_name="utils/codegen_kepler_propagator.py",
    )
    _kepler_residual().make_header(
        output_dir=output_dir,
        script_name="utils/codegen_kepler_propagator.py",
    )
    # Produces kepler_primal_v_f.h and kepler_residual_v_f.h
    print(f"Generated kepler_primal_v_f.h, kepler_residual_v_f.h in {output_dir}")
```

In that case the IFT layer includes both files and there is no `kepler_propagator_closed_form.h`. **Decision rule:** try Step 1 first; if the namespace splice fails or the resulting header doesn't compile cleanly, fall back to Step 1b. Document the choice in the commit message.

- [ ] **Step 2: Run the codegen**

```bash
cd /home/ghecht/Projects/tycho/utils && conda run -n tycho python codegen_kepler_propagator.py
```

Expected: exit 0, writes `kepler_propagator_closed_form.h` (or the two-file variant) under `include/tycho/detail/astro/`.

- [ ] **Step 3: Add closed-form smoke test**

Append to `tests/cpp/astro/test_kepler_lcd.cpp`:

```cpp
#include <tycho/detail/astro/kepler_propagator_closed_form.h>
// (if Step 1b path was taken: include the two split headers instead)

TEST(KeplerClosedForm, PrimalMatchesInlineFG) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH);
    ASSERT_TRUE(k.converged);

    // Pack codegen input: R0(3) + V0(3) + dt + X + U0..U2 = 11
    Eigen::Matrix<double, 11, 1> in;
    in.head<3>() = rv.head<3>();
    in.segment<3, 3>(3) = rv.tail<3>();
    in[6] = 300.0;
    in[7] = k.X;
    in[8] = k.U0;
    in[9] = k.U1;
    in[10] = k.U2;

    KeplerPrimal primal{TychoTest::MU_EARTH};
    Vector6<double> out;
    primal.compute_impl(in, out);

    Vector6<double> expected = apply_fg(rv, k, TychoTest::MU_EARTH, 300.0);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-13 * std::max(1.0, std::abs(expected[i])));
}

TEST(KeplerClosedForm, ResidualVanishesAtConverged) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH);
    ASSERT_TRUE(k.converged);

    // F input: R0(3) + V0(3) + dt + U1 + U2 + U3 = 10
    Eigen::Matrix<double, 10, 1> in;
    in.head<3>() = rv.head<3>();
    in.segment<3, 3>(3) = rv.tail<3>();
    in[6] = 300.0;
    in[7] = k.U1;
    in[8] = k.U2;
    in[9] = k.U3;

    KeplerResidual residual{TychoTest::MU_EARTH};
    Eigen::Matrix<double, 1, 1> F_val;
    residual.compute_impl(in, F_val);
    EXPECT_NEAR(F_val[0], 0.0, 1e-9);  // |F| at converged X* should be at noise floor
}
```

The `using` introduced by the generated header puts `KeplerPrimal` and `KeplerResidual` in the `tycho::astro` namespace; the `using namespace tycho::astro;` already at the top of the test file resolves them.

- [ ] **Step 4: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 tycho_tests \
  && ctest --test-dir tests/cpp/astro -R KeplerClosedForm --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/codegen_kepler_propagator.py \
        include/tycho/detail/astro/kepler_propagator_closed_form.h \
        tests/cpp/astro/test_kepler_lcd.cpp
# (or the two split headers if Step 1b was taken)
git commit -m "$(cat <<'EOF'
feat(astro): SymPy codegen for Kepler primal + residual closed forms

utils/codegen_kepler_propagator.py emits kepler_propagator_closed_form.h
containing KeplerPrimal (11->6, post-converged Goodyear f/g/fdot/gdot
applied to (R0,V0,dt,X,U0..U2)) and KeplerResidual (10->1, the
universal-variable residual F linear in U1..U3). Both with TychoHeaderGen
analytic Jacobian and adjoint-Hessian.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: IFT layer — Jacobian

**Goal:** `kepler_propagator_ift.h::kepler_propagate_jacobian<Scalar>` returns total `(rf, vf)` and 6×7 Jacobian using the iteration kernel + codegen partials + IFT composition. No Hessian yet.

**Files:**
- Create: `include/tycho/detail/astro/kepler_propagator_ift.h`
- Create: `tests/cpp/astro/test_kepler_ift.cpp`
- Modify: `tests/cpp/CMakeLists.txt` (register new test)

**Acceptance Criteria:**
- [ ] `kepler_propagate_jacobian<double>` reproduces (rf, vf) bit-equivalent to `propagate_cartesian<double>` (modulo CSE order, target 1e-13 rel).
- [ ] 6×7 Jacobian matches central finite difference to 5e-7 relative on each of:
  - LEO circular (e=0.01, dt=300s)
  - MEO eccentric (e=0.5, dt=600s)
  - hyperbolic (e=1.5, dt=100s)
  - multi-rev (5 periods, LEO)
- [ ] Parabolic-α fixture (`α ≈ 1e-10`): Jacobian matches FD to 5e-6 rel (looser tolerance acceptable).

**Verify:** `ctest --test-dir tests/cpp/astro -R KeplerIFT_Jacobian --output-on-failure`

**Steps:**

- [ ] **Step 1: Write the IFT header skeleton**

```cpp
// include/tycho/detail/astro/kepler_propagator_ift.h
#pragma once
#include "tycho/detail/astro/kepler_lcd_iterate.h"
#include "tycho/detail/astro/kepler_primal.h"
#include "tycho/detail/astro/kepler_residual.h"
#include "tycho/detail/typedefs/eigen_types.h"

namespace tycho::astro::detail {

template <class Scalar>
inline void kepler_propagate(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                             Scalar dt, double mu,
                             Vector6<Scalar>& xf);

template <class Scalar>
inline void kepler_propagate_jacobian(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                                      Scalar dt, double mu,
                                      Vector6<Scalar>& xf,
                                      Eigen::Matrix<Scalar, 6, 7>& jac);

template <class Scalar>
inline void kepler_propagate_adjoint_hessian(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                                             Scalar dt, double mu,
                                             const Vector6<Scalar>& adjvars,
                                             Vector6<Scalar>& xf,
                                             Eigen::Matrix<Scalar, 6, 7>& jac,
                                             Vector7<Scalar>& adjgrad,
                                             Eigen::Matrix<Scalar, 7, 7>& adjhess);

} // namespace tycho::astro::detail
```

- [ ] **Step 2: Implement the U-derivative recursion helpers**

Helper functions, parameterized on Scalar. The α-recursion uses a Taylor fallback when `|α| < α_tol`.

```cpp
namespace tycho::astro::detail {

// ∂U_n/∂X = U_{n-1}, with ∂U_0/∂X = -alpha * U_1.
template <class Scalar>
inline void U_partials_X(Scalar alpha, const KeplerLCDResult<Scalar>& k,
                         Scalar& dU0_dX, Scalar& dU1_dX,
                         Scalar& dU2_dX, Scalar& dU3_dX) {
    dU0_dX = -alpha * k.U1;
    dU1_dX = k.U0;
    dU2_dX = k.U1;
    dU3_dX = k.U2;
}

// ∂U_n/∂α via the recursion (X*U_{n-1} - n*U_n)/(2 alpha) for n >= 1;
// ∂U_0/∂α = -X * U_1 / 2 (no 1/alpha singularity).
// Taylor fallback near alpha=0:  U_n = X^n/n! - alpha X^{n+2}/(n+2)! + O(alpha^2)
//   => ∂U_n/∂α |_{α=0} = -X^{n+2} / (n+2)!
template <class Scalar>
inline void U_partials_alpha(Scalar alpha, Scalar X,
                             const KeplerLCDResult<Scalar>& k, double alpha_tol,
                             Scalar& dU0_da, Scalar& dU1_da,
                             Scalar& dU2_da, Scalar& dU3_da) {
    using std::abs;
    dU0_da = Scalar(-0.5) * X * k.U1;  // regular at alpha=0
    if (abs_scalar(alpha) > Scalar(alpha_tol)) {
        const Scalar inv2a = Scalar(0.5) / alpha;
        dU1_da = inv2a * (X * k.U0 - Scalar(1) * k.U1);
        dU2_da = inv2a * (X * k.U1 - Scalar(2) * k.U2);
        dU3_da = inv2a * (X * k.U2 - Scalar(3) * k.U3);
    } else {
        // Taylor leading-order
        Scalar X2 = X * X;
        Scalar X3 = X2 * X;
        Scalar X4 = X2 * X2;
        Scalar X5 = X3 * X2;
        // -X^3/6, -X^4/24, -X^5/120
        dU1_da = -X3 * Scalar(1.0 / 6.0);
        dU2_da = -X4 * Scalar(1.0 / 24.0);
        dU3_da = -X5 * Scalar(1.0 / 120.0);
    }
}

// abs_scalar: branch-aware absolute value usable for both double and Eigen::Array<double,W,1>.
// For SS Scalar we want a per-lane abs that produces a Scalar; Eigen::Array provides .abs().
template <class S> inline auto abs_scalar(const S& x) { return std::abs(x); }
template <int W>
inline auto abs_scalar(const Eigen::Array<double, W, 1>& x) { return x.abs(); }
// Note: comparison `> Scalar(alpha_tol)` for Eigen::Array yields an array of bools.
// For per-lane SS dispatch, Task 1's structure ensures U_partials_alpha is called from
// scalar lane code. The Eigen::Array overload exists for the IFT-layer path that
// runs natively packed (Tasks 3-4 use it on closed-form partials which are branch-free
// in alpha but the Taylor fallback IS alpha-branched). For now, the SS-packed IFT
// layer falls through to the regular branch using `(abs_scalar(alpha) > alpha_tol).all()`
// guarded with Eigen::select; see Step 2b.
} // namespace tycho::astro::detail
```

- [ ] **Step 2b: Refine the SS Taylor fallback**

The naive `if` above doesn't compile when `alpha` is `Eigen::Array<double, W, 1>`. Use `Eigen::Array::select` for per-lane blending:

```cpp
template <int W>
inline void U_partials_alpha(const Eigen::Array<double, W, 1>& alpha,
                             const Eigen::Array<double, W, 1>& X,
                             const KeplerLCDResult<Eigen::Array<double, W, 1>>& k,
                             double alpha_tol,
                             Eigen::Array<double, W, 1>& dU0_da,
                             Eigen::Array<double, W, 1>& dU1_da,
                             Eigen::Array<double, W, 1>& dU2_da,
                             Eigen::Array<double, W, 1>& dU3_da) {
    using SS = Eigen::Array<double, W, 1>;
    const SS abs_a = alpha.abs();
    const auto is_para = abs_a <= SS::Constant(alpha_tol);
    // Safe alpha for the recursion path (clamp to alpha_tol*sign to avoid div-by-zero in lanes
    // where Taylor wins). The select() picks the Taylor result for those lanes after.
    const SS safe_a = is_para.select(SS::Constant(alpha_tol), alpha);
    const SS inv2a = SS::Constant(0.5) / safe_a;

    dU0_da = SS::Constant(-0.5) * X * k.U1;
    SS rec1 = inv2a * (X * k.U0 - k.U1);
    SS rec2 = inv2a * (X * k.U1 - SS::Constant(2.0) * k.U2);
    SS rec3 = inv2a * (X * k.U2 - SS::Constant(3.0) * k.U3);

    SS X2 = X * X, X3 = X2 * X, X4 = X3 * X, X5 = X4 * X;
    SS tay1 = SS::Constant(-1.0 / 6.0)   * X3;
    SS tay2 = SS::Constant(-1.0 / 24.0)  * X4;
    SS tay3 = SS::Constant(-1.0 / 120.0) * X5;

    dU1_da = is_para.select(tay1, rec1);
    dU2_da = is_para.select(tay2, rec2);
    dU3_da = is_para.select(tay3, rec3);
}
```

- [ ] **Step 3: Implement `kepler_propagate` (primal-only)**

```cpp
template <class Scalar>
inline void kepler_propagate(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                             Scalar dt, double mu, Vector6<Scalar>& xf) {
    auto k = kepler_lcd_iterate<Scalar>(R0, V0, dt, mu);
    using std::sqrt;
    Scalar sqmu = sqrt(Scalar(mu));
    Scalar aF  = Scalar(1) - k.U2 / k.r0;
    Scalar aG  = (k.r0 * k.U1 + k.sigma0 * k.U2) / sqmu;
    Scalar aFt = -sqmu / (k.r0 * k.r) * k.U1;
    Scalar aGt = Scalar(1) - k.U2 / k.r;
    xf.template head<3>() = aF  * R0 + aG  * V0;
    xf.template tail<3>() = aFt * R0 + aGt * V0;
}
```

- [ ] **Step 4: Implement `kepler_propagate_jacobian`**

This is the IFT composition. Spelling out step-by-step per spec §3.5:

```cpp
template <class Scalar>
inline void kepler_propagate_jacobian(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                                      Scalar dt, double mu,
                                      Vector6<Scalar>& xf,
                                      Eigen::Matrix<Scalar, 6, 7>& jac) {
    KeplerLCDOptions opts;
    auto k = kepler_lcd_iterate<Scalar>(R0, V0, dt, mu, opts);

    // Pack S input: R0(3), V0(3), dt, X, U0, U1, U2  -> 11
    Eigen::Matrix<Scalar, 11, 1> sin;
    sin.template head<3>() = R0;
    sin.template segment<3, 3>(3) = V0;
    sin[6]  = dt;
    sin[7]  = k.X;
    sin[8]  = k.U0;
    sin[9]  = k.U1;
    sin[10] = k.U2;

    // Codegen primal partials
    KeplerPrimal primal{mu};
    Vector6<Scalar> sout;
    Eigen::Matrix<Scalar, 6, 11> S_jac;
    primal.compute_jacobian_impl(sin, sout, S_jac);
    xf = sout;

    // Pack F input: R0(3), V0(3), dt, U1, U2, U3 -> 10
    Eigen::Matrix<Scalar, 10, 1> fin;
    fin.template head<3>() = R0;
    fin.template segment<3, 3>(3) = V0;
    fin[6] = dt;
    fin[7] = k.U1;
    fin[8] = k.U2;
    fin[9] = k.U3;

    KeplerResidual residual{mu};
    Eigen::Matrix<Scalar, 1, 1> F_val;
    Eigen::Matrix<Scalar, 1, 10> F_jac;
    residual.compute_jacobian_impl(fin, F_val, F_jac);

    // U-recursions
    Scalar dU0_dX, dU1_dX, dU2_dX, dU3_dX;
    U_partials_X<Scalar>(k.alpha, k, dU0_dX, dU1_dX, dU2_dX, dU3_dX);
    Scalar dU0_da, dU1_da, dU2_da, dU3_da;
    U_partials_alpha(k.alpha, k.X, k, opts.alpha_tol, dU0_da, dU1_da, dU2_da, dU3_da);

    // dα/dy:  α = 2/r0 - v0²/μ.   y = (R0_x, R0_y, R0_z, V0_x, V0_y, V0_z, dt)
    Vector7<Scalar> dalpha_dy = Vector7<Scalar>::Zero();
    Scalar r0_3 = k.r0 * k.r0 * k.r0;
    for (int i = 0; i < 3; ++i) {
        dalpha_dy[i]     = -Scalar(2) * R0[i] / r0_3;
        dalpha_dy[i + 3] = -Scalar(2) * V0[i] / Scalar(mu);
    }
    // dalpha_dy[6] (dt) = 0

    // Compose total dF/dy  (F has no explicit X dependence)
    // dF/dy_total = F_struct_y + sum_n F_struct_{Un} * dU_n/dα * dα/dy
    // F input layout maps as: F_jac[0..5] = ∂F/∂(R0_x..V0_z); F_jac[6] = ∂F/∂dt;
    //                         F_jac[7..9] = ∂F/∂(U1, U2, U3).
    Vector7<Scalar> dF_dy_total;
    for (int i = 0; i < 7; ++i) {
        Scalar dF_struct_y = F_jac(0, i);  // first 7 entries cover R0,V0,dt
        Scalar dF_chain = F_jac(0, 7) * dU1_da * dalpha_dy[i]
                        + F_jac(0, 8) * dU2_da * dalpha_dy[i]
                        + F_jac(0, 9) * dU3_da * dalpha_dy[i];
        dF_dy_total[i] = dF_struct_y + dF_chain;
    }
    // total dF/dX = r (from kernel)
    Scalar F_X_total = k.r;

    // dX*/dy = -dF/dy_total / F_X_total
    Vector7<Scalar> dX_dy;
    for (int i = 0; i < 7; ++i) dX_dy[i] = -dF_dy_total[i] / F_X_total;

    // Total dS/dy = S_struct_y + S_struct_X * dX/dy + sum_{n=0,1,2} S_struct_{Un} * (dU_n/dX * dX/dy + dU_n/dα * dα/dy)
    // S input layout: S_jac.col(0..5) = ∂S/∂(R0,V0); .col(6) = ∂S/∂dt;
    //                 .col(7) = ∂S/∂X;  .col(8) = ∂S/∂U0;  .col(9) = ∂S/∂U1;  .col(10) = ∂S/∂U2
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
            Scalar dS_struct_y = S_jac(row, col);
            Scalar dS_X_path = S_jac(row, 7) * dX_dy[col];
            Scalar dS_U0_path = S_jac(row, 8) * (dU0_dX * dX_dy[col] + dU0_da * dalpha_dy[col]);
            Scalar dS_U1_path = S_jac(row, 9) * (dU1_dX * dX_dy[col] + dU1_da * dalpha_dy[col]);
            Scalar dS_U2_path = S_jac(row, 10) * (dU2_dX * dX_dy[col] + dU2_da * dalpha_dy[col]);
            jac(row, col) = dS_struct_y + dS_X_path + dS_U0_path + dS_U1_path + dS_U2_path;
        }
    }
}
```

- [ ] **Step 5: Add Hessian stub (full impl in Task 4)**

```cpp
template <class Scalar>
inline void kepler_propagate_adjoint_hessian(...) {
    static_assert(sizeof(Scalar) == 0,
        "kepler_propagate_adjoint_hessian: implemented in Task 4 of the LCD plan");
}
```

- [ ] **Step 6: Write Jacobian FD tests**

```cpp
// tests/cpp/astro/test_kepler_ift.cpp
#include <tycho/detail/astro/kepler_propagator_ift.h>
#include "astro_test_utils.h"
#include <gtest/gtest.h>
#include <numbers>

using namespace tycho;
using namespace tycho::astro;
using namespace tycho::astro::detail;

namespace {

void check_jacobian_fd(const Vector3<double>& R0, const Vector3<double>& V0,
                      double dt, double mu, double rel_tol) {
    Vector6<double> xf_a;
    Eigen::Matrix<double, 6, 7> jac_a;
    kepler_propagate_jacobian<double>(R0, V0, dt, mu, xf_a, jac_a);

    auto eval = [&](const Eigen::Matrix<double, 7, 1>& y) {
        Vector6<double> xf;
        kepler_propagate<double>(y.head<3>(), y.segment<3, 3>(3), y[6], mu, xf);
        return xf;
    };
    Eigen::Matrix<double, 7, 1> y0;
    y0.head<3>() = R0; y0.segment<3, 3>(3) = V0; y0[6] = dt;

    Eigen::Matrix<double, 6, 7> jac_fd;
    for (int i = 0; i < 7; ++i) {
        double eps = std::max(1e-7, 1e-7 * std::abs(y0[i]));
        Eigen::Matrix<double, 7, 1> yp = y0, ym = y0;
        yp[i] += eps; ym[i] -= eps;
        Vector6<double> fp = eval(yp);
        Vector6<double> fm = eval(ym);
        jac_fd.col(i) = (fp - fm) / (2.0 * eps);
    }
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 7; ++c) {
            double scale = std::max(1.0, std::abs(jac_fd(r, c)));
            EXPECT_NEAR(jac_a(r, c), jac_fd(r, c), rel_tol * scale)
                << "row " << r << " col " << c;
        }
    }
}

} // anon

TEST(KeplerIFT_Jacobian, LEOCircular) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, MEOEccentric) {
    Vector6<double> oe; oe << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 600.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, Hyperbolic) {
    Vector6<double> oe; oe << -10000.0, 1.5, 0.2, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, MultiPeriod) {
    auto oe = TychoTest::leoClassic();
    double a = oe[0];
    double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / TychoTest::MU_EARTH);
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 5.0 * T, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, NearParabolic) {
    // alpha ≈ 1e-10: choose a so that 2/r - v²/mu is small.
    // Easy way: pick e=0.999999 with large a; the resulting alpha sweeps near 0.
    Vector6<double> oe; oe << 1.0e7, 0.999999, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, 5e-6);
}
```

- [ ] **Step 7: Register the new test in CMakeLists.txt**

Add `test_kepler_ift.cpp` to `tests/cpp/CMakeLists.txt` next to `test_kepler_lcd.cpp`.

- [ ] **Step 8: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 tycho_tests \
  && ctest --test-dir tests/cpp/astro -R KeplerIFT_Jacobian --output-on-failure
```

Expected: all 5 tests pass. If `NearParabolic` fails, raise the Taylor-fallback threshold above `α_tol = 1e-12` (try `1e-8`); document the choice in the source comment.

- [ ] **Step 9: Commit**

```bash
git add include/tycho/detail/astro/kepler_propagator_ift.h \
        tests/cpp/astro/test_kepler_ift.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(astro): IFT-composed Jacobian for Kepler propagator

kepler_propagator_ift.h::kepler_propagate_jacobian composes the codegen'd
structural Jacobians of S and F with the U-derivative recursions
(∂U_n/∂X = U_{n-1}, ∂U_n/∂α = (X·U_{n-1} - n·U_n)/(2α)) and applies the
implicit-function theorem dX*/dy = -F_y/F_X. Verified against central FD
across LEO, MEO, hyperbolic, multi-period, and near-parabolic fixtures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: IFT layer — adjoint Hessian

**Goal:** Implement `kepler_propagate_adjoint_hessian`. The function returns total `(rf, vf)`, the 6×7 Jacobian, the 7-vector adjoint gradient `Jᵀλ`, and the 7×7 adjoint Hessian. Uses the second-derivative codegen entry points `compute_jacobian_adjointgradient_adjointhessian_impl` on both `KeplerPrimal` and `KeplerResidual`, and applies second-order IFT.

**Files:**
- Modify: `include/tycho/detail/astro/kepler_propagator_ift.h` (replace stub from Task 3 with full body)
- Modify: `tests/cpp/astro/test_kepler_ift.cpp` (add Hessian FD tests)

**Acceptance Criteria:**
- [ ] `kepler_propagate_adjoint_hessian<double>` produces `xf` and `jac` identical to `kepler_propagate_jacobian<double>` (within 1e-12 rel).
- [ ] `adjgrad = jacᵀ · adjvars` to 1e-13 rel.
- [ ] `adjhess` matches central-FD adjoint Hessian to 5e-6 rel on all 5 fixtures (LEO, MEO, hyperbolic, multi-period, near-parabolic; the latter at 5e-5).
- [ ] Existing `KeplerPropagatorAdjointConsistency` test (`test_kepler_propagation.cpp:58`) — once Task 6 wires up the VF — continues to pass at its current 1e-6 tolerance.

**Verify:** `ctest --test-dir tests/cpp/astro -R KeplerIFT_Hessian --output-on-failure`

**Steps:**

- [ ] **Step 1: Derive the second-order IFT formula concretely**

The needed pieces (all evaluated at the converged `(X*, U_n)`):

1. Structural Jacobians and Hessians of S and F from the codegen call `compute_jacobian_adjointgradient_adjointhessian_impl`.
2. U-derivative recursions, applied twice for second derivatives:
   - `∂²U_n/∂X² = ∂U_{n-1}/∂X` (so `∂²U_n/∂X² = U_{n-2}` for `n ≥ 2`; `∂²U_1/∂X² = ∂U_0/∂X = -α U_1`; `∂²U_0/∂X² = -α U_0`).
   - `∂²U_n/(∂X∂α) = ∂(U_{n-1})/∂α` (recurses one level down).
   - `∂²U_n/∂α² = ∂[(X·U_{n-1} - n·U_n)/(2α)]/∂α`. Expand:
     `= (X·∂U_{n-1}/∂α - n·∂U_n/∂α)/(2α) - (X·U_{n-1} - n·U_n)/(2α²)`
     `= (X·∂U_{n-1}/∂α - n·∂U_n/∂α)/(2α) - ∂U_n/∂α / α`.
3. `∂²α/∂y∂y'`:
   - `∂²α/∂R0_i∂R0_j = -2 (δ_ij r0² - 3 R0_i R0_j) / r0⁵` (from `α = 2/r0 - v²/μ`, `r0 = ||R0||`).
   - `∂²α/∂V0_i∂V0_j = -2 δ_ij / μ`.
   - `∂²α/∂R0∂V0 = 0`, `∂²α/∂dt∂· = 0`.
4. Second-order IFT for `X*`:
   - `X*''_{ij} = -(F_yy + F_yX·X*'_j + F_yX·X*'_i + F_XX·X*'_i·X*'_j) / F_X`
   - `F_X` total = `r` (from kernel). `F_XX` total = `σ` (from kernel; equals F''(X*)).
5. Total second derivative of S:
   - `S''_{ij} = S_yy + S_yX(·X*'_j) + S_yX(·X*'_i) + S_XX X*'_i X*'_j + S_X X*''_{ij}`
   - plus the U_n chain through (X, α(y)) — both first- and second-order.

This formula is long. Implementation is mechanical loop-and-sum.

- [ ] **Step 2: Implement helper for second-order U partials**

```cpp
template <class Scalar>
struct U_second_partials {
    Scalar dU0_dX2, dU1_dX2, dU2_dX2, dU3_dX2;     // ∂²U_n/∂X²
    Scalar dU0_dXda, dU1_dXda, dU2_dXda, dU3_dXda; // ∂²U_n/∂X∂α
    Scalar dU0_da2, dU1_da2, dU2_da2, dU3_da2;     // ∂²U_n/∂α²
};

template <class Scalar>
inline U_second_partials<Scalar>
compute_U_second_partials(Scalar alpha, Scalar X,
                          const KeplerLCDResult<Scalar>& k,
                          double alpha_tol,
                          Scalar dU0_dX, Scalar dU1_dX, Scalar dU2_dX, Scalar dU3_dX,
                          Scalar dU0_da, Scalar dU1_da, Scalar dU2_da, Scalar dU3_da) {
    using std::abs;
    U_second_partials<Scalar> p;
    // ∂²U_n/∂X² = ∂(dU_{n-1}/dX)/dX
    p.dU0_dX2 = -alpha * k.U0;            // d(-alpha U1)/dX = -alpha U0
    p.dU1_dX2 = -alpha * k.U1;            // d(U0)/dX = -alpha U1
    p.dU2_dX2 = k.U0;                     // d(U1)/dX = U0
    p.dU3_dX2 = k.U1;                     // d(U2)/dX = U1

    // ∂²U_n/∂X∂α = ∂(dU_{n-1}/dα)/dX  (since U-recursion in α involves X through U_{n-1})
    // Using the X-recursion structure: ∂/∂X (dU_n/dα) = derivatives of the alpha recursion w.r.t X
    // For n ≥ 1: dU_n/dα = (X U_{n-1} - n U_n)/(2α) → ∂/∂X = (U_{n-1} + X dU_{n-1}/dX - n dU_n/dX)/(2α)
    // For n = 0: dU_0/dα = -X U_1/2 → ∂/∂X = -U_1/2 - X dU_1/dX/2
    p.dU0_dXda = Scalar(-0.5) * (k.U1 + X * dU1_dX);
    if (abs_scalar(alpha) > Scalar(alpha_tol)) {
        const Scalar inv2a = Scalar(0.5) / alpha;
        p.dU1_dXda = inv2a * (k.U0 + X * dU0_dX - Scalar(1) * dU1_dX);
        p.dU2_dXda = inv2a * (k.U1 + X * dU1_dX - Scalar(2) * dU2_dX);
        p.dU3_dXda = inv2a * (k.U2 + X * dU2_dX - Scalar(3) * dU3_dX);

        // ∂²U_n/∂α² = (X·dU_{n-1}/dα - n·dU_n/dα)/(2α) - dU_n/dα / α
        p.dU0_da2 = Scalar(-0.5) * X * dU1_da;  // direct from -X U_1/2 → ∂/∂α (-X dU_1/dα/2)
        p.dU1_da2 = inv2a * (X * dU0_da - Scalar(1) * dU1_da) - dU1_da / alpha;
        p.dU2_da2 = inv2a * (X * dU1_da - Scalar(2) * dU2_da) - dU2_da / alpha;
        p.dU3_da2 = inv2a * (X * dU2_da - Scalar(3) * dU3_da) - dU3_da / alpha;
    } else {
        // Taylor-leading terms for the second derivatives.
        // ∂²U_n/∂α² |_{α=0} = 2 X^{n+4} / (n+4)!  (next coefficient in the series)
        Scalar X2 = X*X, X3 = X2*X, X4 = X2*X2, X5 = X3*X2, X6 = X3*X3, X7 = X4*X3;
        p.dU1_dXda = Scalar(-1.0/2.0) * X2; // ∂/∂X (-X^3/6)
        p.dU2_dXda = Scalar(-1.0/6.0) * X3; // ∂/∂X (-X^4/24)
        p.dU3_dXda = Scalar(-1.0/24.0) * X4;// ∂/∂X (-X^5/120)
        p.dU0_da2 = Scalar(2.0 / 24.0)  * X4;  // 2 X^4 / 4!
        p.dU1_da2 = Scalar(2.0 / 120.0) * X5;
        p.dU2_da2 = Scalar(2.0 / 720.0) * X6;
        p.dU3_da2 = Scalar(2.0 / 5040.0)* X7;
    }
    return p;
}
```

- [ ] **Step 3: Implement the full `kepler_propagate_adjoint_hessian` body**

Replace the Task-3 stub with the full implementation. The implementation computes:

1. Iteration kernel call (same as Jacobian).
2. Codegen calls to both VFs' `compute_jacobian_adjointgradient_adjointhessian_impl` — produces structural Jacobians, structural adjoint-Hessians, and structural adjoint-gradients of S and F.
3. The full first-order chain (same as Task 3 — total Jac, dX*/dy, etc).
4. Second-order chain — computes `X*''_{ij}` from second-order IFT, then composes total `S''_{ij}` and contracts with `adjvars` (the 6-vector λ for S; F is single-output, so its adjoint multiplier is implicit in the chain).

```cpp
template <class Scalar>
inline void kepler_propagate_adjoint_hessian(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                                             Scalar dt, double mu,
                                             const Vector6<Scalar>& adjvars,
                                             Vector6<Scalar>& xf,
                                             Eigen::Matrix<Scalar, 6, 7>& jac,
                                             Vector7<Scalar>& adjgrad,
                                             Eigen::Matrix<Scalar, 7, 7>& adjhess) {
    KeplerLCDOptions opts;
    auto k = kepler_lcd_iterate<Scalar>(R0, V0, dt, mu, opts);

    // === Step A: codegen call for S ===
    Eigen::Matrix<Scalar, 11, 1> sin;
    sin.template head<3>() = R0;
    sin.template segment<3, 3>(3) = V0;
    sin[6]  = dt;     sin[7]  = k.X;
    sin[8]  = k.U0;   sin[9]  = k.U1;  sin[10] = k.U2;

    KeplerPrimal primal{mu};
    Vector6<Scalar> sout;
    Eigen::Matrix<Scalar, 6, 11> S_jac;
    Vector<Scalar, 11> S_grad;             // structural adjoint-grad of S
    Eigen::Matrix<Scalar, 11, 11> S_hess;  // structural adjoint-Hess of S
    primal.compute_jacobian_adjointgradient_adjointhessian_impl(
        sin, sout, S_jac, S_grad, S_hess, adjvars);
    xf = sout;

    // === Step B: codegen call for F ===
    Eigen::Matrix<Scalar, 10, 1> fin;
    fin.template head<3>() = R0;
    fin.template segment<3, 3>(3) = V0;
    fin[6] = dt;
    fin[7] = k.U1;  fin[8] = k.U2;  fin[9] = k.U3;

    KeplerResidual residual{mu};
    Eigen::Matrix<Scalar, 1, 1> F_val;
    Eigen::Matrix<Scalar, 1, 10> F_jac;
    Vector<Scalar, 10> F_grad;
    Eigen::Matrix<Scalar, 10, 10> F_hess;
    Vector<Scalar, 1> F_lm; F_lm[0] = Scalar(1);
    residual.compute_jacobian_adjointgradient_adjointhessian_impl(
        fin, F_val, F_jac, F_grad, F_hess, F_lm);
    // F_grad[i] = dF/d(input[i]); F_hess(i,j) = d²F/(d input_i d input_j)
    // (F is single-output and adjvars[0]=1, so adjoint-Hessian == raw Hessian.)

    // === Step C: U partials (1st and 2nd order) ===
    Scalar dU0_dX, dU1_dX, dU2_dX, dU3_dX;
    U_partials_X<Scalar>(k.alpha, k, dU0_dX, dU1_dX, dU2_dX, dU3_dX);
    Scalar dU0_da, dU1_da, dU2_da, dU3_da;
    U_partials_alpha(k.alpha, k.X, k, opts.alpha_tol, dU0_da, dU1_da, dU2_da, dU3_da);
    auto p2 = compute_U_second_partials<Scalar>(k.alpha, k.X, k, opts.alpha_tol,
                                                 dU0_dX, dU1_dX, dU2_dX, dU3_dX,
                                                 dU0_da, dU1_da, dU2_da, dU3_da);

    // === Step D: alpha partials (1st and 2nd order) ===
    Vector7<Scalar> dalpha_dy = Vector7<Scalar>::Zero();
    Scalar r0 = k.r0, r0_3 = r0*r0*r0, r0_5 = r0_3*r0*r0;
    for (int i = 0; i < 3; ++i) {
        dalpha_dy[i]     = -Scalar(2) * R0[i] / r0_3;
        dalpha_dy[i + 3] = -Scalar(2) * V0[i] / Scalar(mu);
    }
    Eigen::Matrix<Scalar, 7, 7> d2alpha_dy2 = Eigen::Matrix<Scalar, 7, 7>::Zero();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        Scalar delta_ij = (i == j ? Scalar(1) : Scalar(0));
        d2alpha_dy2(i, j) = -Scalar(2) * (delta_ij * r0 * r0 - Scalar(3) * R0[i] * R0[j]) / r0_5;
    }
    for (int i = 0; i < 3; ++i)
        d2alpha_dy2(i + 3, i + 3) = -Scalar(2) / Scalar(mu);

    // === Step E: first-order IFT (same as Task 3) ===
    // [Replicate the dX_dy / jac computation from kepler_propagate_jacobian here so
    //  this function is self-contained — see Task 3 Step 4 for the loop body.]
    Vector7<Scalar> dF_dy_total;
    for (int i = 0; i < 7; ++i) {
        Scalar dF_struct_y = F_jac(0, i);
        Scalar dF_chain = F_jac(0, 7) * dU1_da * dalpha_dy[i]
                        + F_jac(0, 8) * dU2_da * dalpha_dy[i]
                        + F_jac(0, 9) * dU3_da * dalpha_dy[i];
        dF_dy_total[i] = dF_struct_y + dF_chain;
    }
    Scalar F_X_total = k.r;
    Vector7<Scalar> dX_dy;
    for (int i = 0; i < 7; ++i) dX_dy[i] = -dF_dy_total[i] / F_X_total;

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
            Scalar dS_struct_y = S_jac(row, col);
            Scalar dS_X_path = S_jac(row, 7) * dX_dy[col];
            Scalar dS_U0_path = S_jac(row, 8) * (dU0_dX * dX_dy[col] + dU0_da * dalpha_dy[col]);
            Scalar dS_U1_path = S_jac(row, 9) * (dU1_dX * dX_dy[col] + dU1_da * dalpha_dy[col]);
            Scalar dS_U2_path = S_jac(row, 10) * (dU2_dX * dX_dy[col] + dU2_da * dalpha_dy[col]);
            jac(row, col) = dS_struct_y + dS_X_path + dS_U0_path + dS_U1_path + dS_U2_path;
        }
    }

    // adjgrad = jacᵀ · adjvars
    adjgrad = jac.transpose() * adjvars;

    // === Step F: second-order IFT — adjhess ===
    //
    // Decomposition into helpers (write each one and unit-test mentally, then compose):
    //
    // (F.1) Total first derivatives of U_n (n = 0..3) w.r.t. y_i.
    //       dU[n][i] := ∂U_n/∂X · dX_dy[i] + ∂U_n/∂α · dalpha_dy[i]
    Eigen::Matrix<Scalar, 4, 7> dU;          // rows = n, cols = y-index
    {
        Scalar dUdX[4]  = { dU0_dX, dU1_dX, dU2_dX, dU3_dX };
        Scalar dUda[4]  = { dU0_da, dU1_da, dU2_da, dU3_da };
        for (int n = 0; n < 4; ++n)
            for (int i = 0; i < 7; ++i)
                dU(n, i) = dUdX[n] * dX_dy[i] + dUda[n] * dalpha_dy[i];
    }

    // (F.2) Total second derivatives of U_n w.r.t. (y_i, y_j) — symmetric in (i, j).
    //       d²U_n/(dy_i dy_j) = ∂²U_n/∂X² · X'_i X'_j
    //                         + ∂²U_n/∂X∂α · (X'_i α'_j + X'_j α'_i)
    //                         + ∂²U_n/∂α² · α'_i α'_j
    //                         + ∂U_n/∂X · X''_{ij}    (depends on X''_{ij}, computed below)
    //                         + ∂U_n/∂α · α''_{ij}
    //
    // *** WARNING — IMPLEMENTATION NOTE FROM TASK 4 EXECUTION ***
    // The plan-as-originally-written labels the F_y* terms below as
    // "F_y*_total" suggesting total y-derivatives.  This is WRONG for the
    // IFT formula `0 = F̃_yy + F̃_yX·X' + F̃_XX·X'·X' + F̃_X·X''`, which
    // requires PARTIALS at fixed X.  Using totals here introduced O(1e-3)
    // errors on every Hessian fixture during execution.  The corrected
    // formulas (also implemented in the committed code in
    // kepler_propagator_ift.h) are partials at fixed X:
    //   • F̃_yX(i)|_fixed_X has NO ∂²U_n/∂X²·X'_i contribution
    //     (that term belongs to a total-derivative expansion only).
    //   • F̃_yy(i,j)|_fixed_X uses ∂U_n/∂α · α'_i (the α-only path of dU),
    //     and explicitly includes F_struct_{Un} · ∂²U_n/∂α²·α'·α' and
    //     F_struct_{Un} · ∂U_n/∂α · α''_{ij} terms that the original plan
    //     code below omits.
    // The labels "F_yy_total" / "F_yX_total" in the variable names below
    // are retained only for diff readability; the actual *quantities*
    // computed in the committed implementation are the fixed-X partials.
    //
    // We need X''_{ij} first.  Since X''_{ij} itself depends on dU values (through F_yy_total
    // and F_yX_total below), proceed in this order:
    //   (a) compute F_XX_total, F_yX_total, F_yy_total (all use dU but NOT d²U).
    //   (b) compute X''_{ij} from second-order IFT.
    //   (c) compute d²U_n/(dy_i dy_j) using X''_{ij} from (b) and α''_{ij} from Step D.
    //   (d) compute S''_{ij} including the contributions from S_struct_X·X''_{ij} and
    //       S_struct_{U_n}·d²U_n/(dy_i dy_j).
    //   (e) contract: adjhess(i, j) = sum_row adjvars[row] * S''_{row, i, j}.

    // (a.i) F_XX_total = ∂²F_total/∂X² = sigma (from kernel, by F linear in U_n + the X-recursion).
    Scalar F_XX_total = k.sigma;

    // (a.ii) F_yX_total[i] = ∂²F_total/∂X∂y_i.
    //        F_struct linear in U_n => ∂²F_struct/∂U_n∂U_m = 0.
    //        F_yX_total[i]
    //          = ∂/∂y_i (sum_n F_struct_{Un} · ∂U_n/∂X)
    //          = sum_n [ (∂F_struct_{Un}/∂y_i) · ∂U_n/∂X
    //                  + F_struct_{Un} · (∂²U_n/∂X² · X'_i + ∂²U_n/∂X∂α · α'_i) ]
    //
    //   ∂F_struct_{Un}/∂y_i is the row of F's structural Hessian indexed by (U_n, y_i).
    //   Layout: F input (R0_x..V0_z, dt, U1, U2, U3) — y indices 0..6, U-indices 7..9.
    Eigen::Matrix<Scalar, 1, 7> F_yX_total;
    {
        // F has U1, U2, U3 only (n = 1, 2, 3).
        const int U_start = 7;  // F_hess column for U1
        Scalar dUdX_n[3]  = { dU1_dX, dU2_dX, dU3_dX };
        // ∂²U_n/∂X² and ∂²U_n/∂X∂α already in p2.
        Scalar d2U_dX2_n[3]  = { p2.dU1_dX2,  p2.dU2_dX2,  p2.dU3_dX2  };
        Scalar d2U_dXda_n[3] = { p2.dU1_dXda, p2.dU2_dXda, p2.dU3_dXda };
        for (int i = 0; i < 7; ++i) {
            Scalar acc = Scalar(0);
            for (int n = 0; n < 3; ++n) {
                Scalar partial_F_Un_yi = F_hess(U_start + n, i);  // ∂²F/(∂U_n ∂y_i)
                acc += partial_F_Un_yi * dUdX_n[n]
                     + F_jac(0, U_start + n) * (d2U_dX2_n[n] * dX_dy[i] + d2U_dXda_n[n] * dalpha_dy[i]);
            }
            F_yX_total(0, i) = acc;
        }
    }

    // (a.iii) F_yy_total(i, j) = ∂²F_total/(∂y_i ∂y_j) symmetric.
    //         Combines structural F_hess in (y, y) and (y, U_n) blocks with the U-chain.
    Eigen::Matrix<Scalar, 7, 7> F_yy_total;
    {
        const int U_start = 7;
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j <= i; ++j) {
                Scalar val = F_hess(i, j);  // pure structural y-y block
                // cross-y/U: 2 · ∂²F/(∂y · ∂U_n) · ∂U_n/∂y'
                for (int n = 0; n < 3; ++n) {
                    val += F_hess(i, U_start + n) * dU(n + 1, j)
                         + F_hess(j, U_start + n) * dU(n + 1, i);
                }
                // U-U block: zero (F linear).
                // (No ∂U_n/∂y · ∂U_m/∂y · ∂²F/(∂U_n∂U_m) term.)
                // 1st-order chain through 2nd-order y-derivative of U_n:
                //   F_struct_{U_n} · d²U_n/(dy_i dy_j) — but d²U_n depends on X''_{ij} (cycle).
                // Defer that piece to (c) below; contribution recorded after X''_{ij} is known.
                F_yy_total(i, j) = val;
                F_yy_total(j, i) = val;
            }
        }
    }

    // (b) X''_{ij} from second-order IFT (without the deferred d²U term yet).
    //     Differentiating F_total(X*(y), y) = 0 twice:
    //       0 = F_yy + F_yX·X' (sym) + F_XX·X'·X' + F_X·X''
    //     => X'' = -(F_yy + F_yX·X' + (F_yX·X')^T + F_XX·X'·X') / F_X
    Eigen::Matrix<Scalar, 7, 7> X_pp;  // X''_{ij}
    Scalar F_X_total_inv = Scalar(1) / F_X_total;
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar v = F_yy_total(i, j)
                     + F_yX_total(0, i) * dX_dy[j]
                     + F_yX_total(0, j) * dX_dy[i]
                     + F_XX_total * dX_dy[i] * dX_dy[j];
            X_pp(i, j) = -v * F_X_total_inv;
            X_pp(j, i) = X_pp(i, j);
        }
    }
    // Note: the deferred "F_struct_{Un} · d²U_n" contribution to F_yy_total includes a term
    // proportional to ∂U_n/∂X · X''_{ij}. We absorbed those by noting the recursion
    //   d²U_n/(dy_i dy_j) (from X-path) contributes F_struct_{Un} · ∂U_n/∂X · X''_{ij}
    //   to F_yy_total, which in turn contributes -F_X^{-1} · F_struct_{Un} · ∂U_n/∂X · X''_{ij}
    //   to X''_{ij} — but sum_n F_struct_{Un} · ∂U_n/∂X = F_X_total = r, so this term is
    //   X''_{ij} multiplied by -1, i.e. it cancels. The expression above is therefore correct.
    // (Sanity: the "F linear in U_n" structure makes this fixed-point trivially solvable.)

    // (c) Now compose d²U_n_total[n][i][j] using X_pp.
    Eigen::Matrix<Scalar, 4, 49> d2U_flat;  // n in [0..3], (i*7+j) for symmetric matrix
    auto d2U_idx = [](int i, int j) { return i * 7 + j; };
    {
        Scalar d2UdX2_n[4]  = { p2.dU0_dX2,  p2.dU1_dX2,  p2.dU2_dX2,  p2.dU3_dX2  };
        Scalar d2UdXda_n[4] = { p2.dU0_dXda, p2.dU1_dXda, p2.dU2_dXda, p2.dU3_dXda };
        Scalar d2Uda2_n[4]  = { p2.dU0_da2,  p2.dU1_da2,  p2.dU2_da2,  p2.dU3_da2  };
        Scalar dUdX_n[4]    = { dU0_dX, dU1_dX, dU2_dX, dU3_dX };
        Scalar dUda_n[4]    = { dU0_da, dU1_da, dU2_da, dU3_da };
        for (int n = 0; n < 4; ++n) {
            for (int i = 0; i < 7; ++i) for (int j = 0; j <= i; ++j) {
                Scalar v = d2UdX2_n[n] * dX_dy[i] * dX_dy[j]
                         + d2UdXda_n[n] * (dX_dy[i] * dalpha_dy[j] + dX_dy[j] * dalpha_dy[i])
                         + d2Uda2_n[n] * dalpha_dy[i] * dalpha_dy[j]
                         + dUdX_n[n] * X_pp(i, j)
                         + dUda_n[n] * d2alpha_dy2(i, j);
                d2U_flat(n, d2U_idx(i, j)) = v;
                d2U_flat(n, d2U_idx(j, i)) = v;
            }
        }
    }

    // (d) S''_{row, i, j} for row in [0..5]. Compose total second derivative of S, then contract.
    //     S input layout: cols 0..6 = y; col 7 = X; cols 8..10 = U0..U2.
    //     S''_{ij} = S_yy_struct(i,j)
    //              + S_yX_struct(j,i_or_swap blah) ... actually compose mechanically:
    //
    //     dS_total/dy_i = S_struct_y[i] + S_struct_X · X'_i
    //                   + sum_{n=0..2} S_struct_{Un} · dU[n][i]
    //
    //     ∂²S_total/∂y_i ∂y_j is the y-derivative of the above. Derivatives:
    //
    //       ∂(S_struct_y[i])/∂y_j         = S_yy_struct(i, j)
    //                                        + sum_n S_yU_struct(i, n) · dU[n][j]
    //                                        + S_yX_struct(i) · X'_j
    //       ∂(S_struct_X)/∂y_j            = S_XX_struct · X'_j
    //                                        + sum_n S_XU_struct(n) · dU[n][j]
    //                                        + S_yX_struct(j)            (mixed)
    //       ∂(S_struct_{Un})/∂y_j         = S_yU_struct(j, n)
    //                                        + S_XU_struct(n) · X'_j
    //                                        + sum_m S_UU_struct(n, m) · dU[m][j]
    //
    //     plus second-order chain through X'' and d²U:
    //       + S_struct_X · X''_{ij}
    //       + sum_{n=0..2} S_struct_{Un} · d²U_n/(dy_i dy_j)
    //
    //     Implementation: pull S_jac (6×11), S_grad (11), S_hess (11×11) — note S_grad and
    //     S_hess are ALREADY adjvar-contracted (S.compute_jacobian_adjointgradient_adjointhessian
    //     contracts with adjvars internally).  So S_grad = S_jac^T · adjvars  (we can verify),
    //     and S_hess = sum_row adjvars[row] · ∂²S_row/(∂x_i ∂x_j).  This is what the codegen
    //     emits and is what we want — adjhess is the y-y restriction of S_hess plus the chain
    //     contributions.
    //
    //     Compute adjhess(i, j) directly using S_hess (already contracted with adjvars):
    //
    //       adjhess(i, j) = S_hess(i, j)                                  // y-y block
    //                     + S_hess(7, i) · X'_j + S_hess(7, j) · X'_i     // y-X cross
    //                     + S_hess(7, 7) · X'_i · X'_j                    // X-X
    //                     + sum_{n=0..2} [ S_hess(8+n, i) · dU[n][j]
    //                                    + S_hess(8+n, j) · dU[n][i]
    //                                    + S_hess(7, 8+n) · (X'_i·dU[n][j] + X'_j·dU[n][i])
    //                                    + S_jac_contracted_X · X''_{ij}                  // see below
    //                                    + S_jac_contracted_Un · d²U_n/(dy_i dy_j) ]
    //                     + sum_{n,m} S_hess(8+n, 8+m) · dU[n][i] · dU[m][j]
    //
    //     where S_jac_contracted_X = sum_row adjvars[row] · S_jac(row, 7)
    //         S_jac_contracted_Un = sum_row adjvars[row] · S_jac(row, 8+n)
    //
    //     Note the "contracted" quantities can be derived from S_jac and adjvars at runtime.
    Scalar adj_S_X = Scalar(0);
    for (int row = 0; row < 6; ++row) adj_S_X += adjvars[row] * S_jac(row, 7);
    Scalar adj_S_U[3];
    for (int n = 0; n < 3; ++n) {
        adj_S_U[n] = Scalar(0);
        for (int row = 0; row < 6; ++row) adj_S_U[n] += adjvars[row] * S_jac(row, 8 + n);
    }

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar v = S_hess(i, j)
                     + S_hess(7, i) * dX_dy[j] + S_hess(7, j) * dX_dy[i]
                     + S_hess(7, 7) * dX_dy[i] * dX_dy[j]
                     + adj_S_X * X_pp(i, j);
            for (int n = 0; n < 3; ++n) {
                v += S_hess(8 + n, i) * dU(n, j) + S_hess(8 + n, j) * dU(n, i)
                   + S_hess(7, 8 + n) * (dX_dy[i] * dU(n, j) + dX_dy[j] * dU(n, i))
                   + adj_S_U[n] * d2U_flat(n, d2U_idx(i, j));
            }
            for (int n = 0; n < 3; ++n)
                for (int m = 0; m < 3; ++m)
                    v += S_hess(8 + n, 8 + m) * dU(n, i) * dU(m, j);
            adjhess(i, j) = v;
            adjhess(j, i) = v;
        }
    }
}
```

> **Note for the implementer:** The second-order body is mechanical but voluminous. After writing it, run the Hessian FD test in Step 5 and iterate until it passes the 5e-6 tolerance. If a bug surfaces, debug by checking each chain piece in isolation: (a) zero-adjvars except one entry → reduces to a single-row Hessian; (b) zero δt-perturbations → checks only `(R0, V0)` block; (c) compare against a numerical second-order chain rule applied to `kepler_propagate_jacobian` outputs (FD of analytical Jacobian).

- [ ] **Step 4: Add Vector<Scalar, N> typedef alias if missing**

`Vector<Scalar, N> = Eigen::Matrix<Scalar, N, 1>` is already defined in `eigen_types.h:23`. No change needed.

- [ ] **Step 5: Add Hessian FD tests**

Append to `tests/cpp/astro/test_kepler_ift.cpp`:

```cpp
namespace {
void check_adjoint_hessian_fd(const Vector3<double>& R0, const Vector3<double>& V0,
                              double dt, double mu, const Vector6<double>& lm,
                              double rel_tol) {
    Vector6<double> xf;
    Eigen::Matrix<double, 6, 7> jac;
    Vector7<double> grad;
    Eigen::Matrix<double, 7, 7> hess_a;
    kepler_propagate_adjoint_hessian<double>(R0, V0, dt, mu, lm, xf, jac, grad, hess_a);

    auto eval_adjgrad = [&](const Eigen::Matrix<double, 7, 1>& y) {
        Vector6<double> xf_;
        Eigen::Matrix<double, 6, 7> jac_;
        kepler_propagate_jacobian<double>(y.head<3>(), y.segment<3, 3>(3), y[6], mu, xf_, jac_);
        return Vector7<double>(jac_.transpose() * lm);
    };

    Eigen::Matrix<double, 7, 1> y0;
    y0.head<3>() = R0; y0.segment<3, 3>(3) = V0; y0[6] = dt;

    Eigen::Matrix<double, 7, 7> hess_fd;
    for (int i = 0; i < 7; ++i) {
        double eps = std::max(1e-6, 1e-6 * std::abs(y0[i]));
        Eigen::Matrix<double, 7, 1> yp = y0, ym = y0;
        yp[i] += eps; ym[i] -= eps;
        Vector7<double> gp = eval_adjgrad(yp);
        Vector7<double> gm = eval_adjgrad(ym);
        hess_fd.col(i) = (gp - gm) / (2.0 * eps);
    }
    // Symmetrize the FD Hessian for comparison
    Eigen::Matrix<double, 7, 7> hess_fd_sym = 0.5 * (hess_fd + hess_fd.transpose());
    for (int r = 0; r < 7; ++r) for (int c = 0; c < 7; ++c) {
        double scale = std::max(1.0, std::abs(hess_fd_sym(r, c)));
        EXPECT_NEAR(hess_a(r, c), hess_fd_sym(r, c), rel_tol * scale)
            << "row " << r << " col " << c;
    }
    EXPECT_NEAR((jac.transpose() * lm - grad).norm(), 0.0, 1e-13);
}
} // anon

TEST(KeplerIFT_Hessian, LEOCircular) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    Vector6<double> lm = TychoTest::deterministic_random_vector(6, 401, -1.0, 1.0);
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, MEOEccentric) {
    Vector6<double> oe; oe << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Vector6<double> lm = TychoTest::deterministic_random_vector(6, 402, -1.0, 1.0);
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 600.0, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, Hyperbolic) {
    Vector6<double> oe; oe << -10000.0, 1.5, 0.2, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Vector6<double> lm = TychoTest::deterministic_random_vector(6, 403, -1.0, 1.0);
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, MultiPeriod) {
    auto oe = TychoTest::leoClassic();
    double T = 2.0 * std::numbers::pi * std::sqrt(oe[0]*oe[0]*oe[0] / TychoTest::MU_EARTH);
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Vector6<double> lm = TychoTest::deterministic_random_vector(6, 404, -1.0, 1.0);
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 5.0 * T, TychoTest::MU_EARTH, lm, 5e-6);
}

TEST(KeplerIFT_Hessian, NearParabolic) {
    Vector6<double> oe; oe << 1.0e7, 0.999999, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    Vector6<double> lm = TychoTest::deterministic_random_vector(6, 405, -1.0, 1.0);
    check_adjoint_hessian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, lm, 5e-5);
}
```

- [ ] **Step 6: Build and iterate until passing**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 tycho_tests \
  && ctest --test-dir tests/cpp/astro -R KeplerIFT_Hessian --output-on-failure
```

Iterate on the body of `kepler_propagate_adjoint_hessian` until all 5 tests pass at the stated tolerances.

- [ ] **Step 7: Commit**

```bash
git add include/tycho/detail/astro/kepler_propagator_ift.h tests/cpp/astro/test_kepler_ift.cpp
git commit -m "$(cat <<'EOF'
feat(astro): IFT-composed adjoint Hessian for Kepler propagator

kepler_propagate_adjoint_hessian completes the IFT layer with second-order
chain rule through U_n(X, alpha(y)) and second-order IFT for X*. Verified
against central FD across LEO, MEO, hyperbolic, multi-period, and
near-parabolic fixtures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Rewrite `propagate_cartesian`

**Goal:** Replace the body of `propagate_cartesian<Scalar>` in `kepler_utils.h:527-600` with a kernel call + closed-form post-processing. No DSL changes; just a body rewrite. Public API and template signature unchanged.

**Files:**
- Modify: `include/tycho/detail/astro/kepler_utils.h` (lines 526-600)

**Acceptance Criteria:**
- [ ] Existing `KeplerPropagation.MultiPeriodReturn` and `KeplerPropagation.HyperbolicTrajectory` (`tests/cpp/astro/test_kepler_propagation.cpp`) pass unchanged.
- [ ] `bench/cpp/kepler/bench_kepler.cpp::BM_PropagateCartesian` runs and produces a result; numerical sanity-check vs LEO ground truth within 1e-10.
- [ ] No template-instantiation regressions in the rest of the codebase that uses `propagate_cartesian` (none currently in `src/`, but examples and bindings should still compile).

**Verify:**
```
ctest --test-dir tests/cpp/astro -R KeplerPropagation --output-on-failure
```

**Steps:**

- [ ] **Step 1: Replace the body**

Open `include/tycho/detail/astro/kepler_utils.h`. Replace `propagate_cartesian` (lines 526-600) with:

```cpp
template <class Scalar>
Vector6<Scalar> propagate_cartesian(const Vector6<Scalar> &RV, Scalar dt, Scalar mu) {
    using std::sqrt;
    if (dt == Scalar(0)) return RV;

    auto k = ::tycho::astro::detail::kepler_lcd_iterate<Scalar>(
        RV.template head<3>(), RV.template tail<3>(), dt, double(mu));

    Scalar sqmu = sqrt(mu);
    Scalar aF  = Scalar(1) - k.U2 / k.r0;
    Scalar aG  = (k.r0 * k.U1 + k.sigma0 * k.U2) / sqmu;
    Scalar aFt = -sqmu / (k.r0 * k.r) * k.U1;
    Scalar aGt = Scalar(1) - k.U2 / k.r;

    Vector6<Scalar> fx;
    fx.template head<3>() = aF  * RV.template head<3>() + aG  * RV.template tail<3>();
    fx.template tail<3>() = aFt * RV.template head<3>() + aGt * RV.template tail<3>();
    return fx;
}
```

- [ ] **Step 2: Add include**

At the top of `kepler_utils.h`, alongside the existing includes, add:

```cpp
#include "tycho/detail/astro/kepler_lcd_iterate.h"
```

- [ ] **Step 3: Build and run existing tests**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 tycho_tests \
  && ctest --test-dir tests/cpp/astro -R KeplerPropagation --output-on-failure
```

Expected: existing 2 tests in `KeplerPropagation` pass.

- [ ] **Step 4: Build benchmarks (smoke test only — no comparison yet)**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 bench_all \
  && ./bench/cpp/bench_all --benchmark_filter='BM_PropagateCartesian'
```

Expected: runs and produces a number.

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/astro/kepler_utils.h
git commit -m "$(cat <<'EOF'
refactor(astro): propagate_cartesian uses LCD kernel instead of inline Newton

Body of propagate_cartesian now delegates to kepler_lcd_iterate + the
post-converged Goodyear closed form. Public API and template signature
unchanged. Existing KeplerPropagation tests cover the public behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Rewrite `KeplerPropagator` VF

**Goal:** Replace `kepler_propagator.h` with a hand-written VF that delegates `compute_*_impl` to the IFT layer. Delete the giant commented-out hand-rolled block (lines 333-555 of the current file). DSL machinery (`GenericFunction`, `ScalarRootFinder`, `IfElseFunction`, etc.) usage in this file is removed.

**Files:**
- Rewrite: `include/tycho/detail/astro/kepler_propagator.h`
- (Verify only) `tests/cpp/astro/test_kepler_propagation.cpp:58` — `KeplerPropagatorAdjointConsistency` continues to pass.
- (Verify only) `tychopy/test/test_OptimalControl/test_Integrators.py:264, :504` — Python tests using `Astro.Kepler.KeplerPropagator` continue to pass.

**Acceptance Criteria:**
- [ ] `KeplerPropagator` is a `VectorFunction<7, 6, Analytic, Analytic>` with `is_vectorizable = true`, member `mu_`, two constructors (default, `(double)`).
- [ ] `compute_impl`, `compute_jacobian_impl`, `compute_jacobian_adjointgradient_adjointhessian_impl` all delegate to the IFT layer.
- [ ] Existing `KeplerPropagatorAdjointConsistency` test passes at 1e-6 tolerance.
- [ ] Python integrator tests pass.
- [ ] No remaining commented-out hand-rolled code in `kepler_propagator.h`.

**Verify:**
```
ctest --test-dir tests/cpp/astro -R KeplerPropagation --output-on-failure
```
plus
```
conda run -n tycho python -m pytest tychopy/test/test_OptimalControl/test_Integrators.py -v
```

**Steps:**

- [ ] **Step 1: Write the new file**

Replace the entire contents of `include/tycho/detail/astro/kepler_propagator.h` with:

```cpp
// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Reimplemented as a hand-written VF backed by kepler_lcd_iterate +
//     codegen'd primal/residual + IFT composition layer (no DSL).
// =============================================================================
#pragma once
#include "tycho/detail/astro/kepler_propagator_ift.h"
#include "tycho/vector_functions.h"

namespace tycho::astro {

using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::MatRef;
using vf::VecRef;
using vf::VectorFunction;

struct KeplerPropagator
    : VectorFunction<KeplerPropagator, 7, 6,
                     DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<KeplerPropagator, 7, 6,
                                DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    double mu_ = 1.0;
    static constexpr bool is_vectorizable = true;

    KeplerPropagator() { this->set_io_rows(7, 6); }
    explicit KeplerPropagator(double mu) : mu_(mu) { this->set_io_rows(7, 6); }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3, 3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        detail::kepler_propagate<Scalar>(R0, V0, dt, mu_, out);
        fx = out;
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x,
                                      CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3, 3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        Eigen::Matrix<Scalar, 6, 7> jac;
        detail::kepler_propagate_jacobian<Scalar>(R0, V0, dt, mu_, out, jac);
        fx = out;
        jx = jac;
    }

    template <class InType, class OutType, class JacType,
              class AdjGradType, class AdjHessType, class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> ag = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> ah = adjhess_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3, 3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        Eigen::Matrix<Scalar, 6, 7> jac;
        Vector7<Scalar> grad;
        Eigen::Matrix<Scalar, 7, 7> hess;
        Vector6<Scalar> lm = adjvars;
        detail::kepler_propagate_adjoint_hessian<Scalar>(R0, V0, dt, mu_, lm, out, jac, grad, hess);
        fx = out;
        jx = jac;
        ag = grad;
        ah = hess;
    }
};

} // namespace tycho::astro
```

- [ ] **Step 2: Build C++ tests and check the existing astro tests still pass**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 tycho_tests \
  && ctest --test-dir tests/cpp/astro --output-on-failure
```

Expected: all astro tests pass, including the existing `KeplerPropagatorAdjointConsistency`.

- [ ] **Step 3: Build the Python extension and run the integrator tests**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 _tychopy \
  && cd /home/ghecht/Projects/tycho \
  && conda run -n tycho python -m pytest tychopy/test/test_OptimalControl/test_Integrators.py -v
```

Expected: all integrator tests pass.

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/astro/kepler_propagator.h
git commit -m "$(cat <<'EOF'
refactor(astro): KeplerPropagator VF without the DSL

Reimplements KeplerPropagator as a hand-written VectorFunction whose
compute_impl / compute_jacobian_impl /
compute_jacobian_adjointgradient_adjointhessian_impl delegate directly
to the kepler_propagate / _jacobian / _adjoint_hessian functions in
kepler_propagator_ift.h. The previous DSL-based GenericFunction +
ScalarRootFinder construction is removed. Old commented-out hand-rolled
implementation block is also deleted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add benchmarks

**Goal:** Add 4 new VF-path benchmarks to `bench/cpp/kepler/bench_kepler.cpp` to quantify the DSL-removal win and to give Phase 2 (true SIMD) something to compare against later.

**Files:**
- Modify: `bench/cpp/kepler/bench_kepler.cpp` (append to existing file)

**Acceptance Criteria:**
- [ ] 4 new benchmarks compile and run: `BM_KeplerPropagator_VF_Compute`, `BM_KeplerPropagator_VF_Jacobian`, `BM_KeplerPropagator_VF_AdjointHessian`, `BM_KeplerPropagator_VF_Compute_SS4`.
- [ ] Existing `BM_PropagateCartesian` / `_Moderate` / `_Hyperbolic` continue to run.
- [ ] `bench/bench_track.sh record` produces a recording without errors.

**Verify:**
```
cd build && ./bench/cpp/bench_all --benchmark_filter='BM_KeplerPropagator'
```

**Steps:**

- [ ] **Step 1: Append the new benchmarks**

Append to `bench/cpp/kepler/bench_kepler.cpp`:

```cpp
///////////////////////////////////////////////////////////////////////////////
// KeplerPropagator VF benchmarks (post-LCD-rewrite)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/detail/astro/kepler_propagator.h>

static void BM_KeplerPropagator_VF_Compute(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    Eigen::Matrix<double, 7, 1> in;
    in.head<6>() = rv;
    in[6] = 300.0;
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<double, 6, 1> out;
    for (auto _ : state) {
        prop.compute_impl(in, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_Compute);

static void BM_KeplerPropagator_VF_Jacobian(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    Eigen::Matrix<double, 7, 1> in;
    in.head<6>() = rv;
    in[6] = 300.0;
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<double, 6, 1> out;
    Eigen::Matrix<double, 6, 7> jac;
    for (auto _ : state) {
        prop.compute_jacobian_impl(in, out, jac);
        benchmark::DoNotOptimize(jac);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_Jacobian);

static void BM_KeplerPropagator_VF_AdjointHessian(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    Eigen::Matrix<double, 7, 1> in;
    in.head<6>() = rv;
    in[6] = 300.0;
    Eigen::Matrix<double, 6, 1> lm = Eigen::Matrix<double, 6, 1>::Constant(0.5);
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<double, 6, 1> out;
    Eigen::Matrix<double, 6, 7> jac;
    Eigen::Matrix<double, 7, 1> grad;
    Eigen::Matrix<double, 7, 7> hess;
    for (auto _ : state) {
        prop.compute_jacobian_adjointgradient_adjointhessian_impl(
            in, out, jac, grad, hess, lm);
        benchmark::DoNotOptimize(hess);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_AdjointHessian);

static void BM_KeplerPropagator_VF_Compute_SS4(benchmark::State &state) {
    using SS = Eigen::Array<double, 4, 1>;
    auto rv = classic_to_cartesian<double>(leoClassic(), MU_EARTH);
    Eigen::Matrix<SS, 7, 1> in;
    for (int i = 0; i < 6; ++i) in[i] = SS::Constant(rv[i]);
    in[6] = SS::Constant(300.0);
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<SS, 6, 1> out;
    for (auto _ : state) {
        prop.compute_impl(in, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_Compute_SS4);
```

- [ ] **Step 2: Build and smoke-run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j6 bench_all \
  && ./bench/cpp/bench_all --benchmark_filter='BM_KeplerPropagator'
```

Expected: all 4 new benchmarks run and produce numbers.

- [ ] **Step 3: Commit**

```bash
git add bench/cpp/kepler/bench_kepler.cpp
git commit -m "$(cat <<'EOF'
bench(astro): VF-path benchmarks for KeplerPropagator

Adds BM_KeplerPropagator_VF_{Compute, Jacobian, AdjointHessian, Compute_SS4}
to quantify the DSL-removal speedup and give a baseline for the future
true-SIMD iteration kernel (Phase 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Pre-merge verification gate

**Goal:** Run the full pre-merge verification sequence per CLAUDE.md §Testing. Document any benchmark regressions or improvements in the eventual PR description.

**Files:** none modified — this is a verification task that just runs commands and records outputs.

**Acceptance Criteria:**
- [ ] All C++ unit tests pass: `ctest --output-on-failure` (no skip, no fail).
- [ ] All 38 Python examples pass: `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"`.
- [ ] C++ brachistochrone converges to obj ≈ 1.8013 s.
- [ ] `bench/bench_track.sh compare` shows no unexplained regressions; document VF benchmark improvements.

**Verify:** all four sequences in CLAUDE.md §Pre-Merge Verification Sequence pass.

**Steps:**

- [ ] **Step 1: Record bench baseline against `main`**

```bash
cd /home/ghecht/Projects/tycho
git stash
git checkout main
cmake --preset linux-clang-release -DBUILD_CPP_BENCHMARKS=ON
cd build && ninja -j6 bench_all
cd .. && bench/bench_track.sh baseline
git checkout feat/kepler-propagator-lcd
git stash pop || true
```

- [ ] **Step 2: Run the four-step pre-merge sequence on the feature branch**

```bash
# 1. C++ unit tests
cd build && ninja -j6 tycho_tests tycho_tests_light \
  && ctest --output-on-failure

# 2. Python examples
cd .. && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"

# 3. C++ brachistochrone
cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j6 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp | grep "Optimal Solution"

# 4. Benchmarks (record + compare)
ninja -j6 bench_all
cd .. && bench/bench_track.sh record && bench/bench_track.sh compare
```

- [ ] **Step 3: Document results**

For each benchmark in the kepler suite (existing + new), record `head` vs `baseline` numbers from `bench_track.sh compare` output. Save to `doc/plans/2026-05-02-kepler-propagator-lcd-bench.md` as a small results-only doc, suitable for inclusion in the eventual PR description.

- [ ] **Step 4: Commit the results doc**

```bash
git add doc/plans/2026-05-02-kepler-propagator-lcd-bench.md
git commit -m "$(cat <<'EOF'
docs(astro): bench results for LCD Kepler propagator rewrite

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Spec coverage check

| Spec section | Implementing task |
|---|---|
| §3.1 iteration kernel | Tasks 0, 1 |
| §3.2 SuperScalar (Phase 1) | Task 1 |
| §3.3 codegen script | Task 2 |
| §3.4 generated header | Task 2 |
| §3.5 IFT composition (Jac) | Task 3 |
| §3.5 IFT composition (Hess) | Task 4 |
| §3.6 KeplerPropagator VF | Task 6 |
| §3.7 propagate_cartesian | Task 5 |
| §4 data flow | Tasks 3, 4, 5, 6 (verified by tests) |
| §5 numerical contracts | Tasks 0, 3, 4 (FD checks); Task 8 (multi-period) |
| §6 error handling | Task 0 (asymptote guard, dt=0); Tasks 3, 4 (parabolic Taylor) |
| §7.1 existing tests | Tasks 5, 6, 8 |
| §7.2 new unit tests | Tasks 0, 1, 2, 3, 4 |
| §7.3 new benchmarks | Task 7 |
| §7.4 pre-merge gate | Task 8 |
| §8 cleanup | Task 6 (deletes commented-out block) |

No spec section is unaddressed.
