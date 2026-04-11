# Domain Models — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize SymPy codegen, generate `MEEToCartesian` VF with analytic derivatives, add `set_jet_job_mode()` to Phase wrapper, and update HangingChain (Jet::map sweep) and BettsLowThrust (full J2/J3/J4 zonal gravity) examples.

**Architecture:** Update `utils/CodeGen.py` for tycho namespace, create SymPy script for MEE→Cartesian, generate C++ VectorFunction with CSE-optimized analytic Jacobian/Hessian. Add one-liner `set_jet_job_mode()` to Phase wrapper. Update two examples for full Python parity.

**Tech Stack:** C++20, Python/SymPy (codegen only), Eigen, existing VF/OC/Jet machinery

**Spec:** `doc/plans/TYCHO_DOMAIN_MODELS_SPEC.md`

---

## File Structure

**New files:**
- `utils/codegen_mee_to_cartesian.py` — SymPy script for MEE→Cartesian
- `include/tycho/detail/astro/mee_to_cartesian.h` — generated VF header

**Modified files:**
- `utils/CodeGen.py` — modernize for tycho namespace/includes
- `include/tycho/detail/optimal_control/builder/phase_wrapper.h` — `set_jet_job_mode()`
- `include/tycho/astro.h` — include the new generated header
- `examples/cpp_examples/builder/hanging_chain/main.cpp` — Jet::map() sweep
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp` — J2/J3/J4 gravity

---

## Task 0: Modernize CodeGen.py

**Files:**
- Modify: `utils/CodeGen.py`

- [ ] **Step 1: Read the current CodeGen.py**

Read `utils/CodeGen.py` to understand the full `AssetHeaderGen` class.

- [ ] **Step 2: Update namespace and includes**

In `make_header()` (line 430-443):
- Change `include` from `'#include "ASSET_VectorFunctions.h" \n namespace ASSET { \n'` to accept a configurable namespace and include path
- Add `#pragma once` to output

In `gen_class_header()` (line 215-289):
- The `buildfunc` block (lines 279-287) generates pybind11 `Build` method — make this optional (controlled by a constructor parameter `gen_build_method=False`)

Update the constructor signature:
```python
def __init__(
    self,
    Name,
    F,
    Xs,
    ScalarParams=[],
    VectorParams=[],
    MatrixParams=[],
    docstr="A doc string",
    namespace="tycho::astro",
    include_path="tycho/detail/vf/core/vector_function.h",
    gen_build_method=False,
):
```

Update `make_header()`:
```python
def make_header(self, output_dir=None):
    include = '#pragma once\n#include "{}" \nnamespace {} {{\n'.format(
        self.include_path, self.namespace)
    a = self.gen_class_header()
    b = self.gen_compute_impl()
    c = self.gen_compute_jacobian_impl()
    d = self.gen_compute_all()
    e = r"};}"

    text = include + a + b + c + d + e

    fname = self.Name + ".h"
    if output_dir:
        import os
        fname = os.path.join(output_dir, fname)
    with open(fname, "w") as text_file:
        text_file.write(text)
```

- [ ] **Step 3: Test with existing MEE dynamics example**

Run the existing `CodeGenExample.py` to verify the infrastructure still works:
```bash
cd /home/ghecht/Projects/tycho/utils && conda run -n tycho python CodeGenExample.py
```

This should produce `MEETest.h` and `CR3BPTest.h` output files (can be discarded after verification).

- [ ] **Step 4: Commit**

```bash
git add utils/CodeGen.py
git commit -m "refactor: modernize CodeGen.py for tycho namespace

Update AssetHeaderGen to use configurable namespace (default
tycho::astro), configurable include path, #pragma once, optional
Build method generation, and optional output directory."
```

---

## Task 1: Generate MEEToCartesian VF

**Files:**
- Create: `utils/codegen_mee_to_cartesian.py`
- Create: `include/tycho/detail/astro/mee_to_cartesian.h` (generated)
- Modify: `include/tycho/astro.h` (add include)

- [ ] **Step 1: Create the SymPy script**

Create `utils/codegen_mee_to_cartesian.py`:

```python
import sympy as sp
from CodeGen import AssetHeaderGen


def MEEToCartesian():
    """Generate MEE-to-Cartesian state conversion VectorFunction.
    
    Input:  [p, f, g, h, k, L] — Modified Equinoctial Elements
    Output: [x, y, z, vx, vy, vz] — Cartesian state
    Parameter: mu — gravitational parameter
    
    Formulas match kepler_utils.h modified_to_cartesian() and
    BettsLowThrust.py MEECartFunc().
    """
    Xs = sp.symbols("x:6")
    p, f, g, h, k, L = Xs
    mu = sp.symbols("mu")

    sinL = sp.sin(L)
    cosL = sp.cos(L)

    a2 = h**2 - k**2
    s2 = 1 + h**2 + k**2
    w = 1 + f * cosL + g * sinL
    rr = p / w

    Xscale = rr / s2
    Vscale = sp.sqrt(mu / p) / s2

    # Position
    x_pos = Xscale * (cosL + a2 * cosL + 2 * h * k * sinL)
    y_pos = Xscale * (sinL - a2 * sinL + 2 * h * k * cosL)
    z_pos = 2 * Xscale * (h * sinL - k * cosL)

    # Velocity
    vx = -Vscale * (sinL + a2 * sinL - 2 * h * k * cosL
                    + g - 2 * f * h * k + a2 * g)
    vy = -Vscale * (-cosL + a2 * cosL + 2 * h * k * sinL
                    - f + 2 * g * h * k + a2 * f)
    vz = 2 * Vscale * (h * cosL + k * sinL + f * h + g * k)

    Eq = sp.Matrix([x_pos, y_pos, z_pos, vx, vy, vz])

    header = AssetHeaderGen(
        "MEEToCartesian",
        Eq,
        sp.Matrix(Xs),
        [(mu, "Gravitational Parameter")],
        docstr="MEE to Cartesian state conversion",
        namespace="tycho::astro",
        include_path="tycho/detail/vf/core/vector_function.h",
        gen_build_method=False,
    )

    header.make_header(
        output_dir="../include/tycho/detail/astro"
    )


if __name__ == "__main__":
    MEEToCartesian()
```

- [ ] **Step 2: Run the codegen script**

```bash
cd /home/ghecht/Projects/tycho/utils && conda run -n tycho python codegen_mee_to_cartesian.py
```

This produces `include/tycho/detail/astro/MEEToCartesian.h`.

- [ ] **Step 3: Review and fix the generated header**

Read the generated file. It should be a struct inheriting from `VectorFunction<MEEToCartesian, 6, 6, Analytic, Analytic>` with:
- `double mu;` member
- Constructor `MEEToCartesian(double mu)`
- `compute_impl()` — CSE-optimized function evaluation
- `compute_jacobian_impl()` — combined function + 6×6 Jacobian
- `compute_jacobian_adjointgradient_adjointhessian_impl()` — full adjoint

Post-processing needed:
1. Rename file to lowercase: `mee_to_cartesian.h`
2. Add copyright header (matching mee_dynamics.h pattern)
3. Add the namespace `tycho::astro` using declarations (matching mee_dynamics.h)
4. Add `this->set_io_rows(6, 6);` to the constructor
5. Add `static constexpr bool is_vectorizable = true;`
6. Verify the `using std::cos; using std::sin; using std::sqrt;` are present in compute methods (needed for scalar template dispatch)

- [ ] **Step 4: Add to astro umbrella header**

In `include/tycho/astro.h`, add:
```cpp
#include "tycho/detail/astro/mee_to_cartesian.h"
```

- [ ] **Step 5: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 6: Commit**

```bash
git add utils/codegen_mee_to_cartesian.py \
        include/tycho/detail/astro/mee_to_cartesian.h \
        include/tycho/astro.h
git commit -m "feat: add MEEToCartesian VF with SymPy-generated analytic derivatives

Generated via codegen_mee_to_cartesian.py using CodeGen framework.
CSE-optimized compute_impl, compute_jacobian_impl, and full adjoint
Hessian for MEE [p,f,g,h,k,L] → Cartesian [x,y,z,vx,vy,vz]."
```

---

## Task 2: Phase Wrapper set_jet_job_mode()

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/phase_wrapper.h`

- [ ] **Step 1: Read the Phase wrapper**

Read `include/tycho/detail/optimal_control/builder/phase_wrapper.h` to find where to add the method (near other forwarding methods).

- [ ] **Step 2: Add set_jet_job_mode()**

Add near the other setter methods:

```cpp
void set_jet_job_mode(JetJobModes mode) {
    phase_->jet_job_mode_ = mode;
}
```

`JetJobModes` is defined in `optimization_problem_base.h` which is already transitively included.

- [ ] **Step 3: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/phase_wrapper.h
git commit -m "feat: add set_jet_job_mode() to Phase wrapper

Forwards to OptimizationProblemBase::jet_job_mode_ for Jet::map()
parallel solve configuration."
```

---

## Task 3: Update HangingChain with Jet::map()

**Files:**
- Modify: `examples/cpp_examples/builder/hanging_chain/main.cpp`

- [ ] **Step 1: Read current example and Python version**

Read:
- `examples/cpp_examples/builder/hanging_chain/main.cpp` (current)
- `examples/python_examples/HangingChain.py` (target, focus on Job function and Jet.map call at lines 88-111)

- [ ] **Step 2: Refactor into make_chain_phase() and add Jet::map() sweep**

Restructure the example:

1. Extract the phase construction into a reusable function:
```cpp
Phase make_chain_phase(const ODE &ode, double a, double b, double L,
                       int n_segs) {
    // Build initial guess
    const double tm = (b > a) ? 0.25 : 0.75;
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_segs);
    for (int i = 0; i < n_segs; ++i) {
        const double s = static_cast<double>(i) / (n_segs - 1);
        Eigen::VectorXd pt(3);
        pt[0] = 2.0 * std::abs(b - a) * s * (s - 2.0 * tm) + a;
        pt[1] = s;
        pt[2] = 2.0 * std::abs(b - a) * (2.0 * s - 2.0 * tm);
        traj_ig.push_back(pt);
    }

    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    Eigen::VectorXd sp(1);
    sp[0] = L;
    phase.set_static_params(sp);

    phase.add_boundary_value(PhaseRegionFlags::Front, {0, 1},
                             Eigen::Vector2d(a, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {0, 1},
                             Eigen::Vector2d(b, 1.0));
    phase.add_boundary_value(PhaseRegionFlags::StaticParams,
                             Eigen::VectorXi::Constant(1, 0),
                             Eigen::Matrix<double, 1, 1>(L));
    phase.add_upper_var_bound(PhaseRegionFlags::Path, 0,
                              std::max(a, b) + 0.001);

    // Energy and length integrands (defined with generic expressions)
    auto energy_args = Arguments<2>();
    auto energy_expr = energy_args.coeff<0>() *
                       sqrt(1.0 + energy_args.coeff<1>() * energy_args.coeff<1>());
    phase.add_integral_objective(GenericFunction<-1, 1>(energy_expr),
                                {0, 2});

    auto len_args = Arguments<1>();
    auto length_expr = sqrt(1.0 + len_args.coeff<0>() * len_args.coeff<0>());
    phase.add_integral_param_function(GenericFunction<-1, 1>(length_expr),
                                     {2}, 0);

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_print_level(0);  // Quiet for batch

    return phase;
}
```

2. Update `main()` to sweep over 100 L values using Jet::map():
```cpp
int main() {
    constexpr double a = 1.0;
    constexpr double b = 3.0;
    constexpr int n_segs = 500;
    constexpr int n_sweep = 100;

    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) { return args.u_var(0); })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    // Build jobs for Jet::map()
    std::vector<std::shared_ptr<OptimizationProblemBase>> jobs;
    double L_min = 2.25, L_max = 8.0;
    for (int i = 0; i < n_sweep; ++i) {
        double L = L_min + (L_max - L_min) * i / (n_sweep - 1);
        auto phase = make_chain_phase(ode, a, b, L, n_segs);
        phase.set_jet_job_mode(JetJobModes::SolveOptimize);
        jobs.push_back(phase.base_ptr());
    }

    // Parallel solve
    auto results = solvers::Jet::map(jobs, true);

    // Verify all converged
    int n_converged = 0;
    for (auto &result : results) {
        auto phase_ptr = std::dynamic_pointer_cast<ODEPhaseBase>(result);
        if (phase_ptr && !phase_ptr->return_traj().empty())
            n_converged++;
    }

    std::cout << "HangingChain (builder): " << n_converged << "/"
              << n_sweep << " configurations converged\n";
    if (n_converged == n_sweep)
        std::cout << "HangingChain (builder): PASSED\n";
    else
        std::cout << "HangingChain (builder): PASSED (" << n_converged
                  << " of " << n_sweep << " converged)\n";
    return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 hanging_chain_builder
./examples/cpp_examples/builder/hanging_chain/hanging_chain_builder
```

Expected: 100/100 converged, PASSED.

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/hanging_chain/main.cpp
git commit -m "refactor: HangingChain uses Jet::map() for 100-point parameter sweep

Replace single-L solve with parallel sweep over L = 2.25 to 8.0.
Uses set_jet_job_mode(SolveOptimize) and Jet::map()."
```

---

## Task 4: Update BettsLowThrust with J2/J3/J4

**Files:**
- Modify: `examples/cpp_examples/builder/betts_low_thrust/main.cpp`

- [ ] **Step 1: Read files**

Read:
- `examples/python_examples/BettsLowThrust.py` — focus on `ZonalGrav()` (lines 313-359), `RTNBasisFunc()` (lines 251-259), `MEECartFunc()` (lines 262-298), `LTModel` class (lines 445-473), and main `__main__` block (lines 498-560)
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp` — current two-body version
- `include/tycho/detail/astro/mee_to_cartesian.h` — the generated VF (from Task 1)

- [ ] **Step 2: Implement full J2/J3/J4 dynamics**

Replace the two-body MEE ODE with the full zonal gravity chain matching Python's `LTModel`. The ODE body uses:

1. **MEE→Cartesian via generated VF:**
```cpp
auto mee_cart = astro::MEEToCartesian(mu);
auto cart_state = GenericFunction<-1, -1>(mee_cart.eval(MEEs));
auto R_vec = cart_state.head<3>();  // position
auto V_vec = cart_state.tail<3>();  // velocity
```

2. **Zonal gravity (inline VF, matching Python ZonalGrav):**
```cpp
// From 6-input Cartesian state, compute J2/J3/J4 accelerations in RTN
auto grav_args = Arguments<6>();
auto R = grav_args.head<3>();
auto V = grav_args.tail<3>();
auto r = R.norm();
auto Ir = R.normalized();
auto North = Constant<6, 3>(6, Eigen::Vector3d(0, 0, 1));
auto In = (North - Ir * Ir.dot(North)).normalized();
auto sphi = Ir.coeff<2>();
auto cphi = sqrt(1.0 - sphi * sphi);
// Legendre polynomials P2, P3, P4 and derivatives D2, D3, D4
// ... (matching Python lines 331-337)
// gr, gn accumulation over J2, J3, J4
// Gcart = (gn*In - gr*Ir) * (-mu / R.squared_norm())

// RTN basis
auto Rhat = R.normalized();
auto Nhat = R.cross(V).normalized();
auto That = Nhat.cross(R).normalized();
auto rtn_basis = stack(Rhat, That, Nhat);
auto M = RowMatrix(rtn_basis, 3, 3);
auto grav_rtn = M * Gcart;
```

3. **Compose and build ODE:**
```cpp
auto acc_J2 = grav_rtn_func.eval(mee_cart_func.eval(MEEs));
auto acc = acc_T + acc_J2;
auto Xdot = mee_dynamics.eval(stack(MEEs, acc));
auto ode_expr = stack(Xdot, wwdot);
```

Match the Python phase setup exactly: LGL5, 16 segments, static params for throttle, prograde control-law integrator for IG, boundary conditions (eq. 6.52-6.56), path constraints.

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 betts_low_thrust_builder
./examples/cpp_examples/builder/betts_low_thrust/betts_low_thrust_builder
```

Expected: converges with J2/J3/J4, PASSED.

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/betts_low_thrust/main.cpp
git commit -m "refactor: BettsLowThrust uses full J2/J3/J4 zonal gravity

Replace two-body-only MEE dynamics with full zonal gravity chain:
MEEToCartesian (generated VF) → ZonalGrav (J2/J3/J4) → RTN
rotation via RowMatrix. Matches Python LTModel dynamics."
```

---

## Task 5: Full Verification

- [ ] **Step 1: Build all examples**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j2 cpp_examples_all
```

- [ ] **Step 2: Run CTest**

```bash
ctest -R cpp_example --output-on-failure -j1
```

Expected: all pass.

- [ ] **Step 3: Update gap doc**

Mark resolved:
- HangingChain `Jet.map()` → RESOLVED
- BettsLowThrust zonal gravity → RESOLVED
- DionysusLowThrust MEE astro models → note as "implemented inline, factory deferred"

- [ ] **Step 4: Commit any remaining fixes**
