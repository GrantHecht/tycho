# C++ Static DSL Pain Points

Documented during Phase 4 C++ example implementation. These findings inform the
design priorities for phases 5-7 (ODEArguments, GenericODE, improved static DSL).

## Compile Metrics

| Example         | Wall Time | Peak Memory | Notes                          |
|-----------------|-----------|-------------|--------------------------------|
| brachistochrone | 1m41s     | 6.9 GB      | Arguments<4>, 2 states 1 ctrl  |
| zermelo         | 2m13s     | 7.3 GB      | Arguments<4>, 4 ODE types      |
| hypersens       | 1m32s     | 6.3 GB      | Arguments<3>, simplest ODE     |
| delta3_launch   | 2m11s     | 8.6 GB      | Arguments<11>, 7 states 3 ctrl |

All measurements: single-threaded (-j1), ccache cold, Clang 21, -O3, Linux x86_64.

## Pain Point 1: No composable ODE functions

**Severity: High** — each wind model / engine configuration requires its own ODE
struct. In Python, the ODE constructor takes callables (wind functions, engine params)
at runtime. In the static DSL, the expression template type changes with every
structural variation, requiring a separate `BUILD_ODE_FROM_EXPRESSION` per model.

**Zermelo impact:** 4 separate ODE structs for what Python does with one class and
a composable wind function.

**Delta3Launch impact:** One ODE struct works (same structure, different params), but
only because `BUILD_ODE_FROM_EXPRESSION(RocketODE, RocketODE_Impl, double, double)`
accepts multiple constructor args of the same type. Heterogeneous parameter packs
would require a struct wrapper.

## Pain Point 2: No unary negation on VF expressions

**Severity: Medium** — `-expr` does not compile. Must rewrite as `expr * (-1.0)` or
restructure (e.g., `Re - R.norm()` instead of `-(R.norm() - Re)`).

## Pain Point 3: double * Scaled<...> operator bug

**Severity: Medium** — `OperatorOverloads.h` lines 62-66 access `func.Scaled_func`
but the member in `Scaled_Impl` is `func`. This means you cannot scale an
already-scaled expression: `a * (b * expr)` compiles but `(a * b_scaled_expr)` where
`b_scaled_expr = b * expr` does not. Workaround: pre-combine all scalar constants
into a single double and apply one scaling pass. The `/` operator (line 95-96)
correctly uses `func.func`.

## Pain Point 4: Manual index tracking

**Severity: High** — every `addBoundaryValue`, `addEqualCon`, `addLUNormBound` call
requires constructing `Eigen::VectorXi` and `Eigen::VectorXd` arrays manually.
Python uses plain lists. The C++ API also requires an explicit `ScaleType` argument
that Python infers.

```cpp
// C++: 6 lines for what Python does in 1
Eigen::VectorXi front_idx(2); front_idx << 0, 1;
Eigen::VectorXd front_val(2); front_val << xt0, 0.0;
phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

# Python
phase.addBoundaryValue("First", [0, 1], [xt0, 0])
```

## Pain Point 5: No ODEArguments in static DSL

**Severity: Medium** — `Arguments<N>` requires the user to compute the total phase
vector size (states + 1 time + controls) and track all indices manually. Python's
`ODEArguments(nX, nU)` auto-computes the layout and provides `XVar()`, `UVar()`,
`XVec()`, `UVec()` accessors.

## Pain Point 6: Constant vectors in expressions

**Severity: Medium** — expressions like `R.cross([0, 0, We])` or
`vf.cross([0, 0, 1], hvec)` require explicit `Constant<IR, OR>` construction with
matching input size. Python accepts numpy arrays directly.

```cpp
// C++: must match IR=11 of surrounding expressions
Eigen::Vector3d omega_val(0.0, 0.0, We);
auto omega = Constant<11, 3>(11, omega_val);
auto Vr = V + R.cross(omega);

# Python
Vr = V + R.cross(np.array([0, 0, We]))
```

## Pain Point 7: BUILD_ODE_FROM_EXPRESSION requires at least one type arg

**Severity: Low** — the macro uses `__VA_ARGS__` which leaves a trailing comma when
empty. ODEs with zero constructor parameters need a dummy `double` arg.

## Pain Point 8: Template compilation cost

**Severity: Medium** — even the simplest ODE (HyperSens: 1 state, 1 control) takes
1.5 minutes and 6.3 GB. The most complex (Delta3Launch: Arguments<11>) takes 2.2
minutes and 8.6 GB. This is a barrier to iteration speed during development.
