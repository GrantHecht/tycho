# Tycho — TODO

## Betts Low-Thrust C++ Benchmark + C++ ODE API Improvements

### Background

PR #20 added Accelerate sparse solver improvements and PSIOPT benchmarks.
During regression investigation, we identified the need for a large-scale
benchmark based on the Betts (2009) low-thrust orbit transfer (example 6).
The Python version exists at `examples/python_examples/BettsLowThrust.py`
and exercises the full solver stack: MEE dynamics with J2/J3/J4, LGL5
collocation, adaptive mesh refinement, unit-vector control constraints.

Implementing this in C++ exposed fundamental scalability issues with the
static VectorFunction DSL used by `BUILD_ODE_FROM_EXPRESSION`.

### Problem: Template Instantiation Cost

The MEE low-thrust ODE has 7 states, 3 controls, 1 parameter (12-element
input vector, 7 complex outputs involving trig functions, divisions, and
products). When defined via the static VF DSL (`Arguments<12>`, trig
compositions, `StackedOutputs{7 items}`), clang consumes **11+ GB of RAM**
per translation unit during template instantiation.

For comparison:
- `BrachODE` (3 states, 1 control → `Arguments<5>`, 3 outputs): ~2 GB
- `PolarLTODE` (4 states, 2 controls → `Arguments<7>`, 4 outputs): ~8 GB
- `MEELTODE` (7 states, 3 controls, 1 param → `Arguments<12>`, 7 outputs): **11+ GB**

The cost grows super-linearly with expression tree depth because the Jacobian
and VJP template expansions create deeply nested types. Placing a complex ODE
in a shared header (`bench_common.h`) multiplies this cost across every TU
that includes it.

### Additional Issues Discovered

1. **`BUILD_ODE_FROM_EXPRESSION` only supports single-type constructor args.**
   Multi-parameter ODEs require workarounds (struct wrapper or hardcoded
   constants). The struct approach fails because `VectorExpression<Derived,
   Impl, StructType>` doesn't generate the `Input<Scalar>`/`Output<Scalar>`
   type aliases that `Integrator<ODE>` requires.

2. **No C++ equivalent of Python's `ODEBase` + named variable groups.**
   The Python API (`oc.ODEBase` with `Vgroups`, `make_input()`, `make_units()`)
   provides a clean interface for complex ODEs with named states/controls.
   The C++ API requires manual index tracking for boundary conditions,
   constraints, and objectives.

3. **No C++ control-law integration API.**
   Python's `ode.integrator(step, control_law, var_group)` allows generating
   initial guesses with a control law applied during integration. The C++
   `Integrator<ODE>` only supports autonomous integration — there's no way
   to inject a control law for initial guess generation.

4. **VF DSL double-scaling limitation.**
   Expressions like `scalar * (1.0 + 0.01 * vf_expr) / scalar` produce
   `Scaled<Scaled<...>>` nesting that triggers missing `Scaled_func` errors.
   Workaround: pre-compute scalar coefficients before multiplying VF exprs.

### Implementation Plan

#### Phase 1: Dedicated BettsLT Benchmark TU (near-term)

Add the MEE ODE and BettsLT benchmarks in their own source file to avoid
polluting other TUs with the template cost:

```
bench/cpp/solvers/bench_betts_lt.cpp   — dedicated TU for BettsLT benchmarks
```

This file would include `bench_common.h` for runtime init but define the
`MEELT_Impl` ODE locally. Only the solver benchmark TU pays the 11 GB cost.

The CMake target should compile this file with reduced parallelism (`-j1`)
to avoid OOM on machines with <16 GB RAM.

**Benchmarks to add:**
- `BM_PSIOPT_BettsLT_32seg` — fixed mesh, 32 segments
- `BM_PSIOPT_BettsLT_64seg` — fixed mesh, 64 segments
- `BM_PSIOPT_BettsLT_MeshRefine` — adaptive mesh starting from 16 segments
  (LGL5, mesh tol 1e-7) — this is the key benchmark that grows the NLP to
  realistic sizes via refinement

**Open questions:**
- Linear interpolation initial guess likely won't converge for this problem.
  Need to either implement C++ control-law integration or pre-compute and
  embed an initial trajectory (from the Python integrator).
- The `std::make_shared<ODEPhase<MEELTODE>>` call fails with libc++ due to
  `enable_shared_from_this` vtable issues. Workaround: use
  `std::shared_ptr<ODEPhase<MEELTODE>>(new ODEPhase<MEELTODE>(...))`.

#### Phase 2: C++ ODE API Improvements (medium-term)

Improve the C++ API to match Python's usability for complex problems:

1. **Multi-parameter ODE support.** Fix `BUILD_ODE_FROM_EXPRESSION` to work
   with struct constructor parameters, or provide an alternative macro that
   accepts a parameter struct and generates the correct `Input`/`Output`
   type aliases.

2. **Named variable interface.** Add a C++ equivalent of Python's `Vgroups`
   / `make_input()` for ODE state/control/parameter naming. This would
   enable `phase->addBoundaryValue("Front", {"MEEs", "w", "t"}, vals)`
   instead of manual index vectors.

3. **Control-law integration.** Extend `Integrator<ODE>` to accept a
   control law VectorFunction, matching Python's
   `ode.integrator(step, control_law, var_group)`.

#### Phase 3: VF DSL Compile-Time Scalability (longer-term)

The exponential template depth growth is the root cause of the memory issue.
Potential approaches:

1. **Expression tree simplification pass.** Detect and collapse redundant
   nesting (e.g., `Scaled<Scaled<...>>` → single `Scaled` with combined
   coefficient).

2. **Runtime VF fallback.** For ODEs above a certain complexity threshold,
   build the expression DAG at runtime (like Python's `ODEBase` does) instead
   of at compile time. This trades some evaluation speed for dramatically
   better compile times and memory.

3. **Explicit Jacobian ODEs.** Allow users to provide hand-coded Jacobians
   for complex ODEs, bypassing the automatic differentiation template
   expansion entirely. This is common practice for production astrodynamics
   codes.

### Current State

The MEE ODE definition and `make_betts_lt_phase()` builder are written and
`#if 0`'d in `bench/cpp/bench_common.h` with comments explaining the issue.
The code is functional but disabled due to memory cost. The Python example
(`examples/python_examples/BettsLowThrust.py`) serves as the reference
implementation.

### MT-METIS Tracking Context

The BettsLT benchmark is also important for tracking MT-METIS performance.
Current findings (macOS 26, Apple Silicon):
- MT-METIS has ~5 ms per-call thread coordination overhead on small graphs
- Overhead is amortized at large problem sizes (neutral at 128+ segments)
- Serial METIS is the default; MT-METIS available via `PARMETIS`/`MTMETIS`
  ordering mode
- MT-METIS comparison benchmarks already exist for Brach and PolarLT
- Apple may improve MT-METIS in future macOS releases — benchmarks track this
