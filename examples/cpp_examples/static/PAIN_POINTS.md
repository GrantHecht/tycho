# C++ Static DSL Pain Points

Documented during Phase 4 C++ example implementation, updated in Phase 8 with
post-improvement status.

## Compile Metrics (Phase 4 Baseline)

| Example         | Wall Time | Peak Memory | Notes                          |
|-----------------|-----------|-------------|--------------------------------|
| brachistochrone | 1m41s     | 6.9 GB      | Arguments<5>, 3 states 1 ctrl  |
| zermelo         | 2m13s     | 7.3 GB      | Arguments<4>, 4 ODE types      |
| hypersens       | 1m32s     | 6.3 GB      | Arguments<3>, simplest ODE     |
| delta3_launch   | 2m11s     | 8.6 GB      | Arguments<11>, 7 states 3 ctrl |

All measurements: single-threaded (-j1), ccache cold, Clang 21, -O3, Linux x86_64.

## Pain Point 1: No composable ODE functions

**Severity: High** | **Status: Open**

Each wind model / engine configuration requires its own ODE struct. In Python, the
ODE constructor takes callables (wind functions, engine params) at runtime. In the
static DSL, the expression template type changes with every structural variation,
requiring a separate `BUILD_ODE_FROM_EXPRESSION` per model.

This is a fundamental limitation of the expression-template approach and remains
open for future work (runtime-polymorphic GenericODE is the alternative).

## Pain Point 2: No unary negation on VF expressions

**Severity: Medium** | **Status: Open**

`-expr` does not compile. Must rewrite as `expr * (-1.0)` or restructure.

## Pain Point 3: double * Scaled<...> operator bug

**Severity: Medium** | **Status: Open**

Nested scaling (`a * (b * expr)`) produces `Scaled<Scaled<...>>` types that fail
to compile. Workaround: pre-combine scalar constants into a single factor.

## Pain Point 4: Manual index tracking

**Severity: High** | **Status: Open**

Boundary conditions, constraints, etc. still require `Eigen::VectorXi`/`VectorXd`
construction. Future work could add initializer_list overloads.

## Pain Point 5: No ODEArguments in static DSL

**Severity: Medium** | **Status: RESOLVED (Phase 7/8)**

`ODEArguments<XV, UV, PV>` now auto-computes the phase vector layout and provides
semantic variable tags: `XVar<I>`, `UVar<I>`, `TVar`, `PVar<I>`, `XVec`, `UVec`,
`PVec`, `XSeg<Start, Size>`, `USeg<Start, Size>`, `PSeg<Start, Size>`.

All four static examples updated to use `ODEArguments` + tags in Phase 8.

## Pain Point 6: Constant vectors in expressions

**Severity: Medium** | **Status: Open**

Still requires `Constant<IR, OR>` construction. Future work could add implicit
conversion from Eigen vectors within expression contexts.

## Pain Point 7: BUILD_ODE_FROM_EXPRESSION requires at least one type arg

**Severity: Low** | **Status: RESOLVED (Phase 7)**

The macros now use `__VA_OPT__` (C++20) to handle zero variadic arguments cleanly.
HyperSens example uses zero-arg `Definition()` and `BUILD_ODE_FROM_EXPRESSION`
with no type arguments.

## Pain Point 8: Template compilation cost

**Severity: Medium** | **Status: PARTIALLY RESOLVED (Phase 7/8)**

`BUILD_ODE_FROM_EXPRESSION_FD` avoids instantiating the expression tree's
Jacobian and Hessian templates, using finite differences instead. Delta3Launch
uses this variant. The simpler ODEs (Brachistochrone, Zermelo, HyperSens) keep
analytic derivatives since their expression trees are small enough that compile
cost is acceptable.
