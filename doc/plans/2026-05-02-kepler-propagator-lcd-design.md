# Kepler propagator — Laguerre-Conway-Der + codegen partials

**Status:** design only. No code change yet.

**Branch:** to be created off `main`. Working name: `feat/kepler-propagator-lcd`.

**Goal:** Replace the DSL-based `KeplerPropagator` and the universal-variable Newton in `propagate_cartesian` with a single canonical EMTG-style Laguerre-Conway-Der implementation, backed by a hand-written iteration kernel and a codegen-generated closed-form layer. Recover analytic Jacobian and adjoint-Hessian via the implicit-function theorem.

**Tech stack:** C++20, Eigen 5, SymPy (existing `utils/CodeGen.py`), Google Test, Google Benchmark.

---

## 1. Context

### 1.1 What exists today

Two parallel implementations of universal-variable Newton, both correct but neither ideal:

| Path | File:Line | Derivatives | Used by |
| --- | --- | --- | --- |
| `KeplerPropagator` (VF) | `include/tycho/detail/astro/kepler_propagator.h:236-267` | analytic via VF DSL (`ScalarRootFinder` differentiates through Newton) | `KeplerPhase::make_shooter()` (`kepler_model.h:57`); Python `KeplerPropagator` binding |
| `propagate_cartesian<Scalar>` | `include/tycho/detail/astro/kepler_utils.h:527-600` | none | benchmarks; potential integrator hot paths |

The DSL path produces correct partials but pays the type-erased `GenericFunction<-1,-1>` overhead and forces every operation through the expression-tree machinery, which (per the user's framing) caps the achievable top speed. The scalar path is straight-line C++ but exposes no derivatives.

`kepler_propagator.h:333-555` carries a large commented-out hand-rolled implementation from earlier work — to be deleted as part of this work.

### 1.2 EMTG reference

`https://github.com/nasa/EMTG/blob/master/src/Propagation/KeplerPropagatorTimeDomain.cpp` (downloaded to `/tmp/emtg_kepler.cpp` during design exploration).

Key properties of the EMTG implementation:

- **Iteration:** Laguerre-Conway-Der on the universal-variable residual `F(X) = r₀·U₁ + σ₀·U₂ + U₃ − √μ·δt`. Order `N` promoted from 1 to 10 on stagnation; max 10 iterations per order.
- **Tolerances:** `Xtol = 1e-12`, `α_tol = 1e-12`.
- **Branching:** orbit-type branch (`α > α_tol` ellipse / `α < −α_tol` hyperbola / parabolic) lives **only** in the per-iteration evaluation of `U₀..U₃` (`emtg_kepler.cpp:117-146`). Everything downstream — `r = r₀·U₀ + σ₀·U₁ + U₂`, `σ = σ₀·U₀ + (1 − α·r₀)·U₁`, the Goodyear `aF, aG, aFt, aGt` formulas, the STM blocks — is unbranched closed form in `(R₀, V₀, δt, μ, X, U_n)`.
- **STM:** closed-form post-convergence using `U₄, U₅` and Battin's `C` coefficient (`emtg_kepler.cpp:230-293`). Computed only when requested.
- **Hyperbolic asymptote guard:** `|√(−α)·X| < 30` (EMTG throws on violation).

### 1.3 Why codegen at all

The closed-form post-converged algebra is large enough to be tedious and error-prone to hand-derive — particularly the **adjoint-Hessian**, which EMTG does not supply. CSE'd codegen of the partial Jacobian and Hessian of the primal map `S` and the residual `F` (treating `X` and `U_n` as independent symbolic inputs) gets us:

- A small, mechanical SymPy script.
- Existing `utils/CodeGen.py` infrastructure with no extensions required (the orbit-type branch is *not* in the codegen target).
- Verifiable, reproducible derivation of all partials.
- Cheap future extensions (e.g., `∂/∂μ` partials).

The implicit-function-theorem composition that folds the implicit `X*(R₀, V₀, δt, μ)` dependence back in is hand-written — it is short, textbook, and unit-testable on its own.

### 1.4 Derivative-order budget

PSIOPT consumes adjoint-Hessian. Both the first-order IFT (`X*' = −F_y / F_X`) and the second-order IFT (`X*''_{ij} = −(F_yy + F_yX·X*' + F_yX·X*' + F_XX·X*'·X*') / F_X`) need only **first and second** partials of `S` and `F`. F is *linear* in `(U₁, U₂, U₃)`, which collapses several cross terms.

The U-derivative recursions

```
∂U_n/∂X = U_{n-1}                              (with ∂U₀/∂X = −α·U₁)
∂U_n/∂α = (X·U_{n-1} − n·U_n) / (2α)            for n ≥ 1
∂U_0/∂α = −X·U_1 / 2                            (no 1/α singularity)
```

apply *twice* for the Hessian path. The α-recursion's index moves **down**, not up — and the highest-index U_n ever referenced is **U₃** (used in F's chain rule). U₄ and U₅, which EMTG computes for its closed-form STM expression (Battin's C coefficient), are *not* needed by the codegen-IFT path.

**No third-order derivatives are required anywhere.**

---

## 2. Architecture

```
                     ┌──────────────────────────────────┐
 SymPy at dev time   │ utils/codegen_kepler_propagator.py│
                     └─────────────────┬────────────────┘
                                       │ emits
                     ┌─────────────────▼─────────────────────────┐
                     │ kepler_propagator_closed_form.h            │
                     │   • struct KeplerPrimal_VF (S → rf,vf)     │
                     │   • struct KeplerResidual_VF (F)           │
                     │     each with TychoHeaderGen Jac + adjHess │
                     └─────────────────┬─────────────────────────┘
                                       │ used by
       ┌───────────────────────────────┴───────────────────────────────┐
       │                                                               │
   ┌───▼─────────────────────────────┐               ┌─────────────────▼─────────────────┐
   │ kepler_lcd_iterate.h             │               │ kepler_propagator_ift.h           │
   │   template <Scalar>              │◄──────────────┤   IFT composition layer (≈80 LOC)│
   │   KeplerLCDResult<Scalar>        │ used by       │   • Jac via 1st-order IFT         │
   │   kepler_lcd_iterate(...)        │               │   • adjH via 2nd-order IFT        │
   │   (orbit-type branch lives here) │               │   • parabolic-α Taylor fallback   │
   └───┬─────────────────────────────┘               └─────────────────┬─────────────────┘
       │                                                               │
       │ scalar fast path                                              │ VF analytic path
       │                                                               │
   ┌───▼─────────────────────────┐                       ┌─────────────▼─────────────┐
   │ propagate_cartesian<Scalar> │                       │ KeplerPropagator (VF<7,6>)│
   │   in kepler_utils.h         │                       │   in kepler_propagator.h  │
   │   (rewritten — no DSL)      │                       │   compute_impl,           │
   │                             │                       │   compute_jacobian_impl,  │
   │                             │                       │   compute_..._hessian_impl│
   └─────────────────────────────┘                       └───────────────────────────┘
```

Five files, **one canonical algorithm**: the iteration kernel. Both consumer paths (scalar fast, VF analytic) call the same kernel; the closed-form generated header and the IFT composition layer are only invoked by the VF path.

---

## 3. Components

### 3.1 `kepler_lcd_iterate.h` — iteration kernel (hand-written)

Header-only, scalar/SuperScalar templated.

```cpp
template <class Scalar>
struct KeplerLCDResult {
    Scalar X;                        // converged universal anomaly X*
    Scalar alpha;                    // 2/r0 - v0²/μ
    Scalar r0;                       // ||R0||
    Scalar sigma0;                   // R0·V0/√μ
    Scalar r;                        // r at X* (= F'(X*))
    Scalar sigma;                    // σ at X* (= F''(X*))
    Scalar U0, U1, U2, U3;           // U_4 / U_5 not needed by IFT path; not computed
    bool converged;                  // false on asymptote-guard or max-iter exhaustion
};

struct KeplerLCDOptions {
    double Xtol      = 1.0e-12;
    double alpha_tol = 1.0e-12;
    int max_order    = 10;
    int iters_per_order = 10;
    double hyp_guard = 30.0;         // |√(-α)·X| limit
};

template <class Scalar>
KeplerLCDResult<Scalar>
kepler_lcd_iterate(const Eigen::Matrix<Scalar, 3, 1>& R0,
                   const Eigen::Matrix<Scalar, 3, 1>& V0,
                   Scalar dt, double mu,
                   const KeplerLCDOptions& opts = {});
```

**Body:** essentially line-for-line `emtg_kepler.cpp:46-180` with two changes:

1. Initial guess for `X` matches `propagate_cartesian` (mixed elliptic/hyperbolic/parabolic seed at `kepler_utils.h:548-562`) — slightly different from EMTG's seed but already battle-tested in tycho.
2. Asymptote guard sets `converged = false` and exits cleanly (no throw — see §6).

**δt = 0 fast path:** preserved as in `propagate_cartesian` (early return without iteration; `X = 0`, `U_n` set to their X=0 values).

### 3.2 SuperScalar / SIMD strategy

**Phase 1 (this PR):** per-lane scalar dispatch when `Scalar = Eigen::Array<double, W, 1>`. The kernel template specializes to a `for (lane = 0; lane < W; ++lane)` loop that runs the scalar iteration once per lane and packs the results back. The closed-form post-processing (codegen) and the IFT composition layer remain natively packed across lanes, since both are branch-free.

**Phase 2 (deferred follow-up):** true SIMD masked-blend iteration. The structural requirement on Phase 1 to keep this open: the orbit-type branch and the U_n evaluation are isolated in a single function (`compute_universal_functions(alpha, X) → (U0, U1, U2, U3)`) so swapping in a packet-mask blend implementation is a localized change, not a kernel rewrite.

### 3.3 `utils/codegen_kepler_propagator.py` — SymPy script (hand-written)

Mirrors the structure of `codegen_mee_dynamics.py` and `codegen_cartesian_to_mee.py`. Two `TychoHeaderGen` invocations into the same output header:

```python
# inputs: R0 (3) + V0 (3) + dt + X + U0..U2 = 11 scalars
#   S depends on U0, U1, U2 (via r, aF, aG, aFt, aGt). U3 appears only in F.
# output: rf (3), vf (3) — primal map S
def KeplerPrimal():
    R0 = sp.symbols("R0_:3")
    V0 = sp.symbols("V0_:3")
    dt, X = sp.symbols("dt X")
    U0, U1, U2 = sp.symbols("U0 U1 U2")
    mu = sp.symbols("mu")

    r0 = sp.sqrt(R0[0]**2 + R0[1]**2 + R0[2]**2)
    sigma0 = (R0[0]*V0[0] + R0[1]*V0[1] + R0[2]*V0[2]) / sp.sqrt(mu)
    r = r0*U0 + sigma0*U1 + U2          # = F'(X)

    aF  = 1 - U2/r0
    aG  = (r0*U1 + sigma0*U2) / sp.sqrt(mu)
    aFt = -sp.sqrt(mu)/(r0*r) * U1
    aGt = 1 - U2/r

    rf = sp.Matrix([aF*R0[i] + aG*V0[i] for i in range(3)])
    vf = sp.Matrix([aFt*R0[i] + aGt*V0[i] for i in range(3)])

    Xs = sp.Matrix(list(R0) + list(V0) + [dt, X, U0, U1, U2])
    Func = sp.Matrix.vstack(rf, vf)
    return TychoHeaderGen("KeplerPrimal_VF", Func, Xs,
                          [(mu, "Gravitational parameter", "mu > 0.0")],
                          docstr="Kepler primal map (post-converged)",
                          gen_build_method=False, is_vectorizable=True)

# inputs: R0, V0, dt, U1, U2, U3 = 10 scalars
#   F has no explicit X dependence — only through U_n, which are independent inputs here.
#   U_0 is not used in F.
# output: F (1) — residual
def KeplerResidual(): ...
```

**No changes to `utils/CodeGen.py`.** The codegen tool already supports everything we need: scalar parameter (`mu`), `precompute_params=True` to factor `sqrt(mu)`, `1/mu`, `is_vectorizable=True` for SuperScalar dispatch, full Jac + adjoint-Hess.

The two generated structs (`KeplerPrimal_VF` and `KeplerResidual_VF`) are emitted into the same file `kepler_propagator_closed_form.h` and are **not** registered as Python-visible VFs — they are internal helpers. We invoke their `compute_*_impl` methods directly from the IFT layer.

### 3.4 `kepler_propagator_closed_form.h` — generated (do not edit manually)

Output of §3.3. Header marker: `// Generated by utils/codegen_kepler_propagator.py — do not edit manually.` (TychoHeaderGen already emits this preamble.) Two struct definitions: `KeplerPrimal_VF` and `KeplerResidual_VF`. Both have `compute_impl`, `compute_jacobian_impl`, `compute_jacobian_adjointgradient_adjointhessian_impl` from the existing codegen.

### 3.5 `kepler_propagator_ift.h` — IFT composition (hand-written)

Pure C++, ≈80 LOC. Templated on `Scalar` (matches the kernel). Provides three free functions:

```cpp
namespace tycho::astro::detail {

template <class Scalar>
void kepler_propagate(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                      Scalar dt, double mu,
                      Vector6<Scalar>& xf);                                   // primal only

template <class Scalar>
void kepler_propagate_jacobian(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                               Scalar dt, double mu,
                               Vector6<Scalar>& xf,
                               Eigen::Matrix<Scalar, 6, 7>& jac);              // adds Jac

template <class Scalar>
void kepler_propagate_adjoint_hessian(const Vector3<Scalar>& R0, const Vector3<Scalar>& V0,
                                      Scalar dt, double mu,
                                      const Vector6<Scalar>& adjvars,
                                      Vector6<Scalar>& xf,
                                      Eigen::Matrix<Scalar, 6, 7>& jac,
                                      Vector7<Scalar>& adjgrad,
                                      Eigen::Matrix<Scalar, 7, 7>& adjhess);

} // namespace tycho::astro::detail
```

**Body of `kepler_propagate_jacobian`:**

1. `auto k = kepler_lcd_iterate(R0, V0, dt, mu);`
2. Pack `(R0, V0, dt, X*, U0, U1, U2)` into a 11-vector input. (S depends on U₀, U₁, U₂ only — see §3.3.)
3. Call `KeplerPrimal_VF{mu}.compute_jacobian_impl(input, xf, S_jac)` — `S_jac` is the 6×11 *structural* Jacobian (treats X and U_n as independent inputs).
4. Pack `(R0, V0, dt, U1, U2, U3)` into a 10-vector input. (F has no explicit X dependence — see §3.3.)
5. Call `KeplerResidual_VF{mu}.compute_jacobian_impl(input, F_val, F_jac)` — the 1×10 row of structural F-partials.
6. U-recursions: `∂U_n/∂X = U_{n-1}` for `n ≥ 1`; `∂U_0/∂X = −α·U_1`; `∂U_n/∂α = (X·U_{n-1} − n·U_n)/(2α)` for `n ≥ 1`; `∂U_0/∂α = −X·U_1/2`. Parabolic guard (`|α| < α_tol`) uses leading-order Taylor `U_n(α≈0, X) ≈ X^n / n!` and its α-derivative `∂U_n/∂α|_{α=0} = −X^{n+2}/(n+2)!` (n ≥ 0).
7. α-chain: `∂α/∂R0_i = −2·R0_i/r0³`, `∂α/∂V0_i = −2·V0_i/μ`, `∂α/∂δt = 0`.
8. Compose IFT. F has no explicit X dependence (`F_struct_X = 0`), so the *total* `dF/dX = Σ_n F_struct_{U_n}·∂U_n/∂X = r₀·U₀ + σ₀·U₁ + U₂ = r` — already in `k.r` from the kernel, no extra computation needed. The remaining chain:
   ```
   dF/dy_total = F_struct_y + Σ_{n=1,2,3} F_struct_{U_n} · ∂U_n/∂α · ∂α/∂y
   dX*/dy      = −(dF/dy_total) / k.r
   ```
   (here F_struct_y is the 1×9 partial in (R0, V0, dt) from the codegen; ∂F/∂U_n was returned as part of F_jac).
9. Same chain for S (which *does* have explicit X dependence — `S_jac` includes `∂S/∂X`):
   ```
   dS/dy_total = S_struct_y
               + S_struct_X · dX*/dy
               + Σ_{n=0,1,2} S_struct_{U_n} · (∂U_n/∂X · dX*/dy + ∂U_n/∂α · ∂α/∂y)
   ```
   Result is the 6×7 total Jacobian.

**Body of `kepler_propagate_adjoint_hessian`:** same shape, second-order IFT formula, second derivatives of U_n via the α-recursion applied twice (still index-down, so the highest U_n needed remains U₃).

The IFT layer is the **only** hand-derived non-trivial math in the system. It has its own targeted unit tests (§7.2).

### 3.6 `KeplerPropagator` (rebuilt, `kepler_propagator.h`)

```cpp
struct KeplerPropagator
    : VectorFunction<KeplerPropagator, 7, 6,
                     DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<KeplerPropagator, 7, 6,
                                DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    double mu_ = 1.0;
    static constexpr bool is_vectorizable = true;

    KeplerPropagator() = default;
    explicit KeplerPropagator(double mu) : mu_(mu) {}

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx) const {
        detail::kepler_propagate<Scalar>(x.template head<3>(), x.template segment<3,3>(),
                                         x[6], mu_, fx.const_cast_derived());
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(/* ... */) const {
        detail::kepler_propagate_jacobian<Scalar>(/* ... */);
    }
    template <class InType, class OutType, class JacType,
              class AdjGradType, class AdjHessType, class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(/* ... */) const {
        detail::kepler_propagate_adjoint_hessian<Scalar>(/* ... */);
    }
};
```

DSL machinery, `GenericFunction<-1,-1>`, `ScalarRootFinder`, the giant commented-out block — all gone.

### 3.7 `propagate_cartesian` (rebuilt, `kepler_utils.h`)

```cpp
template <class Scalar>
Vector6<Scalar> propagate_cartesian(const Vector6<Scalar>& RV, Scalar dt, Scalar mu) {
    if (dt == Scalar(0)) return RV;
    auto k = kepler_lcd_iterate<Scalar>(
        RV.template head<3>(), RV.template tail<3>(), dt, mu);
    Scalar aF  = Scalar(1) - k.U2/k.r0;
    Scalar aG  = (k.r0*k.U1 + k.sigma0*k.U2) / sqrt(mu);
    Scalar aFt = -sqrt(mu)/(k.r0*k.r) * k.U1;
    Scalar aGt = Scalar(1) - k.U2/k.r;
    Vector6<Scalar> fx;
    fx.template head<3>() = aF *RV.template head<3>() + aG *RV.template tail<3>();
    fx.template tail<3>() = aFt*RV.template head<3>() + aGt*RV.template tail<3>();
    return fx;
}
```

The closed-form `aF, aG, aFt, aGt` are duplicated here (also in the codegen output) — five lines, deemed worth the redundancy to keep the scalar fast path independent of the codegen output.

---

## 4. Data flow

**Scalar primal call** (`propagate_cartesian` or `KeplerPropagator::compute_impl`):

```
(R0, V0, dt, mu) ──► kepler_lcd_iterate ──► (X*, U0..U3, r0, sigma0, r) ──► closed-form aF/aG/aFt/aGt ──► (rf, vf)
```

**VF Jacobian call** (`KeplerPropagator::compute_jacobian_impl`):

```
(R0, V0, dt, mu) ──► kepler_lcd_iterate ──► (X*, U0..U3, r0, sigma0, r, sigma)
                                          ──► KeplerPrimal_VF::compute_jacobian_impl    ──► dS/d(R0,V0,dt,X,U_n)
                                          ──► KeplerResidual_VF::compute_jacobian_impl  ──► dF/d(R0,V0,dt,U_n)
                                          ──► IFT layer ──► (rf, vf, J ∈ R^{6×7})
```

**VF adjoint-Hessian call:** same kernel call (no extra U-values needed); the second-derivative codegen entry points (`compute_jacobian_adjointgradient_adjointhessian_impl`) on both `KeplerPrimal_VF` and `KeplerResidual_VF` are invoked. IFT layer applies the second-order chain rule (α-recursion still goes index-down).

---

## 5. Numerical contracts

| Property | Target | Verified by |
| --- | --- | --- |
| Round-trip (5 periods, LEO circular) | per-component error ≤ 1e-4 km | existing `KeplerPropagation.MultiPeriodReturn` |
| Hyperbolic propagation (e=1.5) | finite, monotonic radius | existing `KeplerPropagation.HyperbolicTrajectory` |
| Adjoint consistency (LEO circular, dt=100s) | matches FD adjoint to 1e-6 | existing `KeplerPropagatorAdjointConsistency` |
| **(new)** STM vs central FD across {LEO, MEO e=0.5, near-parabolic e=0.999, hyperbolic e=1.5, multi-rev} × {1s, 100s, 0.5T, 1.5T} dt fixtures | per-element ≤ 5e-7 relative | new `STMConsistency_*` tests |
| **(new)** adjoint-Hess vs central FD on same fixtures | per-element ≤ 5e-6 relative | new `AdjointHessianConsistency_*` tests |
| **(new)** scalar-path / VF-path numerical equality | ≤ 1e-13 relative (bit-equivalent expected modulo CSE order) | new `ScalarVFEquivalence` test |

---

## 6. Error handling

- **`δt = 0`**: early-return `(X=0, U_n at X=0)` directly from the kernel; no iteration.
- **Hyperbolic asymptote guard** (`|√(−α)·X| > 30`): kernel sets `converged = false` and returns the last iterate. **No throw** — propagators called inside PSIOPT shooting loops must not unwind a long-running optimization on a transient bad iterate. Downstream callers may inspect `converged` if they need to fail loudly; in `KeplerPropagator::compute_impl`, the result is returned as-is (NaN propagation if the iterate diverged).
- **Max-order exhaustion** (`N = 10`, `iters_per_order = 10`, no convergence): same — `converged = false`, return last iterate.
- **Parabolic singularity in IFT** (`|α| < α_tol`): the U-derivative recursion `(X·U_{n-1} − n·U_n)/(2α)` would divide by zero. IFT layer falls back to leading-order Taylor expansion of `U_n(α, X)` around `α = 0`:
  - `U_n(0, X) = X^n / n!`
  - `∂U_n/∂α |_{α=0} = −X^{n+2} / (n+2)!` (next-order coefficient in the universal-function series `U_n = X^n/n! − α X^{n+2}/(n+2)! + α² X^{n+4}/(n+4)! − …`).
  - `∂U_0/∂α = −X·U_1/2` is regular at α=0 and does not need a fallback.
- **`mu_ ≤ 0`**: rejected at construction time by `TychoHeaderGen`'s constraint mechanism (already supported via the `(mu, "Gravitational parameter", "mu > 0.0")` entry in the codegen script).

---

## 7. Testing

### 7.1 Existing tests (must continue to pass)

- `tests/cpp/astro/test_kepler_propagation.cpp` (all three tests).
- `tychopy/test/test_OptimalControl/test_Integrators.py:264, :504` (`KeplerPropagator` used as ground truth in integrator tests).
- All 38 examples in `examples/python_examples/` (per CLAUDE.md merge policy).

### 7.2 New unit tests

**`tests/cpp/astro/test_kepler_lcd.cpp`:**
- Iteration-kernel correctness: scalar Scalar = double, fixtures spanning ellipse / near-parabolic / hyperbolic / multi-rev, comparing against `propagate_cartesian` (current implementation, which becomes obsolete after this PR but is kept as a one-shot reference snapshot in the test file as `propagate_cartesian_legacy`).
- Convergence flag: forced asymptote-guard hit (very large hyperbolic dt) sets `converged = false`.
- δt = 0 round-trip equality.
- SuperScalar per-lane: `Scalar = Eigen::Array<double, 4, 1>` with mixed-orbit-type lanes returns the same as 4 scalar invocations.

**`tests/cpp/astro/test_kepler_ift.cpp`:**
- Total Jacobian (6×7) vs. central FD on each fixture, tolerance 5e-7 relative.
- Adjoint-Hessian (7×7) vs. central FD on each fixture, tolerance 5e-6 relative.
- Parabolic-α path: dedicated fixture with `|α| < 1e-10`, exercising the Taylor fallback.
- Adjoint-gradient consistency: `λᵀ·J` matches `adjgrad` to machine precision.

### 7.3 New benchmarks

**`bench/cpp/kepler/bench_kepler.cpp`** (extending existing file):
- `BM_KeplerPropagator_VF_Compute` — primal only, scalar Scalar.
- `BM_KeplerPropagator_VF_Jacobian` — full 6×7 Jacobian, scalar.
- `BM_KeplerPropagator_VF_AdjointHessian` — full 7×7 adjoint-Hess, scalar.
- `BM_KeplerPropagator_VF_Compute_SS4` — primal, `Eigen::Array<double, 4, 1>` SuperScalar (per-lane fallback).

The existing `BM_PropagateCartesian` / `_Moderate` / `_Hyperbolic` benchmarks will run against the rewritten `propagate_cartesian` and form the baseline-vs-head comparison. `bench/bench_track.sh compare` is the gate per CLAUDE.md.

### 7.4 Pre-merge gate (per CLAUDE.md §Testing)

1. `ctest --output-on-failure` — all C++ tests pass.
2. `MPLBACKEND=Agg python scripts/run_examples.py` — all 38 examples exit 0.
3. C++ brachistochrone — `Optimal Solution Found`, obj ≈ 1.8013 s.
4. `bench/bench_track.sh compare` — VF benchmarks must improve materially; scalar benchmarks should not regress beyond noise; any regression justified in PR description.

---

## 8. Migration / cleanup

- **Delete** `kepler_propagator.h:333-555` (commented-out hand-rolled implementation).
- **Audit** `ScalarRootFinder` consumers: `grep -r "ScalarRootFinder" include/ src/`. If Kepler was the only user, mark for follow-up cleanup PR (out of scope here).
- **Generated header lifecycle:** `kepler_propagator_closed_form.h` is committed to git like `mee_dynamics.h` / `crtbp_dynamics.h`. The codegen script is run manually (per existing pattern); no CMake-time regeneration.

---

## 9. Out of scope

- Partials w.r.t. μ (`∂(rf,vf)/∂μ`). Trivial to add later (one-line change to the SymPy script and the IFT layer); not needed for current consumers.
- True SIMD masked-blend iteration kernel (Phase 2 follow-up — see §3.2).
- Removing `ScalarRootFinder` from the codebase.
- Changes to `KeplerPhase`, `Kepler` ODE, or any non-`KeplerPropagator` astro types.
- The 8-input variant of `KeplerPropagator` (with μ as input). Stays 7-input.

---

## 10. Risks

- **IFT layer correctness.** Hand-derived second-order chain rule with the U-recursion is the part most likely to harbor subtle bugs. Mitigation: dedicated FD-comparison tests (§7.2) on multiple orbit types including near-parabolic.
- **Parabolic-α precision.** The `(X·U_{n-1} − n·U_n)/(2α)` recursion is numerically delicate when α is small but nonzero (catastrophic cancellation in the numerator). Tolerance threshold for the Taylor fallback may need to be larger than `α_tol = 1e-12` — perhaps `1e-8`. Test fixtures should sweep across α ∈ {0, 1e-14, 1e-10, 1e-8, 1e-6, 1e-3, 1e-1}.
- **Per-lane SuperScalar perf.** Phase 1's per-lane loop may not beat the current DSL's lane-blended path on some workloads. If so, Phase 2 (true SIMD iteration) becomes blocking before merge — to be assessed in §7.3 benchmark results.
- **Codegen output size.** The 6×12 Hessian (~78 unique entries by symmetry, each potentially several FMAs) may produce a large generated TU. Existing `heavy_compile` ninja pool handles this, but compile time should be measured.
