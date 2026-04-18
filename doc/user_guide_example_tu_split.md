# Building Tycho Programs: Splitting Your ODE Into Its Own Translation Unit

This guide shows how to structure a Tycho C++ program so its concrete
VectorFunction / ODE definitions live in a dedicated translation unit (TU),
behind a type-erased factory. The pattern is what the `examples/cpp_examples/static/*`
sources now use, and it matters for **build performance and memory footprint**
— not for runtime speed.

## Why split?

Tycho's VectorFunction DSL builds very expressive expression trees at compile
time. That expressiveness cashes out in **runtime** speed (all specialisation is
inlined and optimised), but the same templates are **expensive to instantiate**:

- In Tycho's post-SP3 baseline build, ~88 % of total compile time is template
  instantiation. A single example `main.cpp` that constructs an `ODEPhase<T>`
  over a custom ODE routinely takes **120–180 seconds** to compile on a modern
  workstation and holds **6–8 GB resident memory** at peak, because every
  `compute_jacobian_adjointgradient_adjointhessian` specialisation the DSL
  synthesises is fully processed inside that one TU.
- On a 31 GB / 16-core Linux host those 8 GB peaks cap safe parallelism at
  roughly `-j2`. Two example TUs in flight is already enough to edge toward
  swap.

**Splitting each example into `main.cpp` + `<example>_ode.cpp`, with a
`GenericFunction<-1,-1>` factory boundary between them, halves the peak RSS
and lets the two TUs compile in parallel.** Tycho users building their own
programs against libtycho get the same improvement on their own hosts/CI.

Measured impact on Tycho's four static examples (2026-04-18 baseline, clang
21.1.8, Linux 31 GB, data: `bench/build_perf/2026-04-18-phaseA*`):

| Example           | Old wall | Old peak RSS | New wall (@-j2) | New peak RSS | RSS ratio |
| ----------------- | -------: | -----------: | --------------: | -----------: | --------: |
| brachistochrone   |    134 s |      7.11 GB |            41 s |      4.32 GB |    60.8 % |
| hypersens         |    115 s |      6.56 GB |            41 s |      4.22 GB |    64.3 % |
| delta3_launch     |    143 s |      7.67 GB |            48 s |      4.72 GB |    61.5 % |
| zermelo (4 ODEs)  |    176 s |      8.19 GB |            43 s |      4.45 GB |    54.3 % |

**Peak RSS across the set** drops from 8.19 GB → 4.72 GB, doubling the safe
`-j` value that fits in 32 GB of RAM.

This pattern does **not** reduce serial CPU-time meaningfully on its own — the
total template-instantiation work is roughly conserved. The gain is
**parallelism** (two TUs instead of one) and **lower RSS per TU** (so more TUs
fit under the memory budget concurrently).

## The pattern

Your program starts with a `main.cpp` that does roughly this:

```cpp
// main.cpp — single TU; instantiates everything.
#include <tycho/tycho.h>

struct MyODE_Impl : tycho::oc::ODESize<NX, NU, NP> {
    static auto Definition(/* model parameters */) {
        auto args = tycho::oc::ODEArguments<NX, NU, NP>();
        // …expression tree using the VectorFunction DSL…
        return tycho::vf::StackedOutputs{ … };
    }
};
BUILD_ODE_FROM_EXPRESSION(MyODE, MyODE_Impl, /* param types */);

int main() {
    MyODE ode(/* args */);
    auto phase = std::make_shared<tycho::oc::ODEPhase<MyODE>>(
        ode, tycho::oc::TranscriptionModes::LGL3, traj, nSegs);
    // …constraints, objective, phase->solve_optimize()…
}
```

Split that file into **three** pieces:

### `my_ode.h`

Just the factory declaration. `main.cpp` never sees `MyODE`.

```cpp
#pragma once
#include <tycho/tycho.h>

namespace my_program {
tycho::vf::GenericFunction<-1, -1> make_my_ode(/* model parameters */);
}
```

### `my_ode.cpp`

Concrete VectorFunction and factory definition. All the heavy
`compute_jacobian_adjointgradient_adjointhessian` instantiations emit once —
in this TU.

```cpp
#include "my_ode.h"

namespace my_program {
namespace {
using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

struct MyODE_Impl : ODESize<NX, NU, NP> {
    static auto Definition(/* model parameters */) {
        auto args = ODEArguments<NX, NU, NP>();
        // …expression tree…
        return StackedOutputs{ … };
    }
};
BUILD_ODE_FROM_EXPRESSION(MyODE, MyODE_Impl, /* param types */);
} // namespace

tycho::vf::GenericFunction<-1, -1> make_my_ode(/* args */) {
    return tycho::vf::GenericFunction<-1, -1>(MyODE(/* args */));
}
} // namespace my_program
```

### `main.cpp`

Includes the header, pulls the erased ODE from the factory, wraps in the
runtime `tycho::ODE` (the same type-erased wrapper the `ODEBuilder` API uses
internally), and drives the solver through the `Phase` returned by
`ode.phase(...)`. No concrete VectorFunction type appears here.

```cpp
#include "my_ode.h"
#include <tycho/tycho.h>

int main() {
    auto erased = my_program::make_my_ode(/* args */);
    tycho::ODE ode(std::move(erased), NX, NU, NP);
    auto phase = ode.phase(tycho::oc::TranscriptionModes::LGL3, traj, nSegs);

    phase.add_boundary_value(/* … */);
    phase.add_delta_time_objective(1.0);
    const auto status = phase.solve_optimize();
    // …
}
```

The `Phase` returned by `ODE::phase(...)` is the same builder-API wrapper used
throughout the `examples/cpp_examples/builder/*` programs. Its member functions
(name-based and index-based `add_boundary_value`, `add_lu_var_bound`,
`add_equal_con`, `set_adaptive_mesh`, `optimize_solve`, …) are exactly what you
would call on the old `ODEPhase<MyODE>` shared pointer; `phase.` replaces
`phase->`, and the optimiser is reached via `phase.optimizer()` instead of
`phase->optimizer_`.

### CMake

One line per target:

```cmake
# Before
add_executable(my_program main.cpp)
# After
add_executable(my_program main.cpp my_ode.cpp)
```

## Multi-ODE programs

If a program has several ODEs (for example, staged-launch phases, or a set of
alternative dynamical models compared against each other), expose one factory
per ODE:

```cpp
// my_odes.h
tycho::vf::GenericFunction<-1,-1> make_phase1_ode(/* params */);
tycho::vf::GenericFunction<-1,-1> make_phase2_ode(/* params */);
tycho::vf::GenericFunction<-1,-1> make_phase3_ode(/* params */);
```

`main.cpp` builds each `tycho::ODE` from the matching factory, then links the
resulting phases through `tycho::OptimalControlProblem`
(`add_phase`, `add_forward_link_equal_con`, `solve_optimize`, …). See
`examples/cpp_examples/static/delta3_launch/` for a concrete four-phase
example.

If `main.cpp` had a `template<typename ODE> solve(...)` helper that was
instantiated once per ODE type, **also erase that helper**. Rewrite it as a
non-template function taking `const tycho::vf::GenericFunction<-1,-1>&`; the
phase/solver machinery inside the helper now instantiates exactly once across
the whole program. The `examples/cpp_examples/static/zermelo/` example
demonstrates this — a single `navigate` function handles all four wind models.

## When NOT to split

- **Tiny single-purpose demos** — if `main.cpp` only contains a 10-line ODE
  and a 30-line driver, the split costs more readability than it saves in
  build time.
- **ODEs that are already type-erased** — programs built through the
  `ODEBuilder` fluent API (`examples/cpp_examples/builder/`) already live
  behind a `GenericFunction<-1,-1>` boundary; splitting further delivers little.
- **Python-bound ODEs** — when a problem crosses into Python via the Tycho
  Python bindings, the ODE is already erased. The split pattern only helps the
  C++-native programs.

## Verifying the impact

To reproduce the numbers above on your own host:

```bash
# Measure peak RSS + wall time for a specific TU
python3 - <<'PY'
import json, subprocess, os, time
with open('build/compile_commands.json') as f:
    data = json.load(f)
tgts = [e for e in data if 'my_program' in e['file']]
os.chdir('build')
for e in tgts:
    op = os.path.abspath(e['output'])
    if os.path.exists(op): os.remove(op)
    t0 = time.time()
    subprocess.run(['/usr/bin/time', '-v', '-o', '/tmp/_t.txt', 'bash', '-c', e['command']])
    dt = time.time() - t0
    rss_kb = int([l for l in open('/tmp/_t.txt') if 'Maximum resident' in l][0].split(':')[-1])
    print(f"{e['output']:<60} wall={dt:6.1f}s  peak_rss={rss_kb/1024/1024:5.2f} GB")
PY
```

The critical-path wall time at `-j<N>` is roughly
`max(main.cpp, <example>_ode.cpp) + link_time` when `N ≥ 2`. Prior to
splitting, the critical path was the single `main.cpp` wall time.

## See also

- `doc/build_performance_report_2026-04-18.md` — full build-perf analysis that
  motivated this guide.
- `doc/build_performance_analysis.md` — procedure for producing the
  measurements in that report.
- `bench/build_perf/2026-04-18-baseline/` and `2026-04-18-phaseA{1,2,3}/` —
  raw measurement artefacts.
- `examples/cpp_examples/static/brachistochrone/` — minimal single-ODE example.
- `examples/cpp_examples/static/delta3_launch/` — multi-phase example with
  four `tycho::ODE` instances under one `OptimalControlProblem`.
- `examples/cpp_examples/static/zermelo/` — multi-ODE comparison with a
  non-template navigator shared across four dynamical models.
