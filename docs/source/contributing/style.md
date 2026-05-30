# Style Guide

This guide is the single source of truth for code style in Tycho. The
PR template references this page; every PR is reviewed against the rules
on this page.

## Formatting

All source code is formatted by automated tools. **Run the formatter
before committing.** Tycho's CI does not auto-format your PR — it
fails when the diff is not clean.

### C++ — clang-format

The C++ format config is `.clang-format` at the repo root (LLVM base,
4-space indent, 100-column limit). Apply it via the CMake target:

```bash
cd build && ninja clang-format
```

Do not add or modify nested `.clang-format` files inside `src/` or
`include/` — the root config is authoritative. Do not reformat
vendored sources under `dep/`.

### Python — ruff

Python uses `ruff format` (Black-compatible) and `ruff check --select I`
(isort import ordering):

```bash
conda run -n tycho ruff format .
conda run -n tycho ruff check --select I --fix .
```

Configuration is in `pyproject.toml`. Line length is 88 (Black default).
`dep/` and `build/` are excluded automatically.

## Naming Conventions

| Symbol kind                         | Convention             | Example                                |
| ----------------------------------- | ---------------------- | -------------------------------------- |
| Types and classes (C++ and Python)  | `PascalCase`           | `DenseFunction`, `ODEPhase`            |
| Free functions, member functions    | `snake_case`           | `set_io_rows`, `add_equal_con`         |
| Member variables (C++)              | `snake_case_` (trailing `_`) | `num_defects_`, `max_iter_`      |
| Compile-time constants              | `kPascalCase`          | `kRKWeightTol`, `kStumpffTaylorEps`    |
| Macros                              | `TYCHO_UPPER_SNAKE`    | `TYCHO_ALWAYS_INLINE`, `TYCHO_NOINLINE`|

The Python API exposes the C++ member-function names verbatim — so a
C++ method `add_boundary_value(...)` becomes
`phase.add_boundary_value(...)` in Python. Grandfathered exceptions
exist for legacy nanobind methods named `adjointgradient`,
`adjointhessian`, and `computeall`; do not introduce new exceptions.

## C++ Docstring Style (Doxygen JavaDoc)

New and public code uses **JavaDoc `@`-style** Doxygen comments —
`@brief`, `@param`, `@return`, etc. — in preference to the older
backslash style (`\brief`, etc.); residual backslash-style comments are
being migrated as files are touched.
**Every public symbol declared in `include/tycho/`** must carry a
docstring with at least `@brief`. Internal helpers in
`include/tycho/detail/` should still be documented; private
implementation details that should not appear in the rendered docs are
wrapped in `@internal ... @endinternal`.

### Required tags by symbol type

| Symbol kind                          | Required                                                    | Optional                                                 |
| ------------------------------------ | ----------------------------------------------------------- | -------------------------------------------------------- |
| Function or method                   | `@brief`, `@param` per arg, `@return` if non-void           | `@throws`, `@pre`, `@post`, `@note`, `@code`             |
| Class or struct                      | `@brief`                                                    | `@tparam`, `@ingroup`, `@note`, `@code`                  |
| Template function or class           | All of the above, plus `@tparam` per template parameter     | —                                                        |
| Member variable                      | `@brief` (`///<` inline comment is allowed)                 | `@note`, `@warning`                                      |
| Enum                                 | `@brief` on the enum; `///<` inline on each value           | —                                                        |
| Type alias (`using`, `typedef`)      | `@brief`                                                    | —                                                        |

### Example: public class

```cpp
/// @brief Sparse interior-point optimizer for large-scale NLPs.
///
/// PSIOPT solves problems of the form
/// @f[
///   \min_{x} f(x) \quad \text{s.t.}\quad g(x) = 0,\; h(x) \le 0
/// @f]
/// using a Mehrotra-style predictor-corrector with Pardiso /
/// Accelerate-Sparse linear solves.
///
/// @note PSIOPT is **not** thread-safe; one instance per solve.
/// @see  tycho::Phase, tycho::vf::VectorFunction
/// @ingroup solvers
class PSIOPT {
 public:
    /// @brief Set the maximum number of major iterations.
    /// @param n  Iteration cap (must be > 0). Defaults to 500.
    void set_max_iterations(int n);

    /// @brief Run the optimizer.
    /// @param  x0  Initial decision vector, shape (nx).
    /// @return     Exit code; ::ConvergedOptimal on success.
    /// @throws std::runtime_error if the linear-solver factorization fails.
    ExitCode solve(Eigen::Ref<const Eigen::VectorXd> x0);

 private:
    int max_iter_ = 500;   ///< @brief Major-iteration cap.
};
```

### Example: templated CRTP method

Templated methods on a CRTP base require `@tparam` for every template
parameter:

```cpp
/// @brief Evaluate the function: `fx = f(x)`.
///
/// CRTP entry point — dispatches to `Derived::compute_impl()`.
/// SuperScalar `Eigen::Array` packs evaluate `W` trajectories per call
/// when @p Derived sets `is_vectorizable = true`.
///
/// @tparam InType   Concrete Eigen expression type for the input vector.
/// @tparam OutType  Concrete Eigen expression type for the output buffer.
/// @param  x        Input vector (size = `InputRows()`).
/// @param  fx_      Output buffer (size = `OutputRows()`).
template <class InType, class OutType>
inline void compute(CVecRef<InType> x, CVecRef<OutType> fx_) const;
```

### Conventions

- One-line `@brief`; second sentence onward is detail. `@brief` ends at
  the first period.
- Math: `@f$ ... @f$` inline, `@f[ ... @f]` block. Rendered via MathJax.
- Examples: `@code{.cpp} ... @endcode` blocks.
- Cross-references: `@see` — never bare-link to a symbol in prose.
- Deprecation: `@deprecated Use foo() instead. Removed in v0.X.`
- Internal-only context: `@internal ... @endinternal` (visible in
  source, hidden from rendered docs by default).
- **No `@author`, `@date`, `@version` tags** — git history is
  authoritative.

Tycho ships a small set of custom Doxygen aliases for recurring
patterns; see the `ALIASES` block in `docs/Doxyfile.in`.

## Python Docstring Style (NumPy + napoleon)

The Python side uses **NumPy-style** docstrings, rendered by the
Sphinx [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
extension. Static type information lives in the `.pyi` stubs and is
**not** repeated in the prose docstring — docstrings document semantic
type info (shape, dtype, sign, units) only.

### The two-docstring rule

```{important}
**nanobind bindings have two docstrings — keep them in sync.**

A change to one must be mirrored to the other.
```

For every Python-facing symbol that originates from C++ (i.e., every
public nanobind binding in `src/bindings/`), there are *two* separate
docstrings that document the same logical function:

1. The **C++ Doxygen docstring** in `include/tycho/...` — describes
   the symbol for C++ users browsing the C++ reference.
2. The **Python NumPy-style docstring** passed as the
   `R"doc(...)doc"` argument inside the `.def(..., R"doc(...)doc")`
   call in `src/bindings/...` — describes the same symbol for Python
   users browsing the Python reference.

These document the *same logical function* but for different audiences
and type systems. If you edit one, edit the other. The PR template
includes an explicit checkbox for this; reviewers check the binding
source against the public header.

Pure-Python modules under `tychopy/` do not have this dual-authoring
problem — their NumPy-style docstring lives inline in the `.py` file.

### Example: nanobind binding

```cpp
nb::class_<PSIOPT>(m, "PSIOPT",
    R"doc(Sparse interior-point optimizer for large-scale NLPs.

    Notes
    -----
    PSIOPT is **not** thread-safe; use one instance per solve.
    )doc")
    .def("solve", &PSIOPT::solve, "x0"_a,
         R"doc(Run the optimizer.

         Parameters
         ----------
         x0 : numpy.ndarray, shape (n,), dtype=float64
             Initial decision vector.

         Returns
         -------
         ExitCode
             ``ExitCode.ConvergedOptimal`` on success; see :class:`ExitCode`
             for the full set.

         Raises
         ------
         RuntimeError
             If the linear-solver factorization fails.

         Examples
         --------
         >>> opt = PSIOPT()
         >>> opt.set_max_iterations(1000)
         >>> code = opt.solve(x0)
         )doc");
```

### Example: pure-Python module

```python
def make_kepler_phase(mu: float, t0: float, t1: float) -> ODEPhase:
    """Build an ODE phase governed by two-body Keplerian dynamics.

    Constructs a ready-to-link :class:`ODEPhase` whose right-hand side
    is the classical two-body acceleration around a primary of
    gravitational parameter ``mu``.

    Parameters
    ----------
    mu : float
        Gravitational parameter of the central body, in consistent
        units with the integration state.
    t0, t1 : float
        Initial and final phase times. ``t1 > t0`` is required; the
        phase is forward-integrated.

    Returns
    -------
    ODEPhase
        A configured phase; no boundary values or controls are set.

    See Also
    --------
    make_cr3bp_phase : Same idea for the circular restricted 3-body problem.

    Examples
    --------
    >>> phase = make_kepler_phase(mu=1.0, t0=0.0, t1=2 * 3.14159)
    >>> phase.add_boundary_value("Front", [0, 1, 2, 3, 4, 5], r0_v0)
    """
```

### Section order

When a docstring uses multiple sections, they appear in this fixed
order so napoleon renders predictable HTML:

> Summary → Extended description → Parameters → Returns → Yields →
> Raises → See Also → Notes → References → Examples.

### Conventions

- Static types live in the `.pyi` stubs (canonical) — do not repeat
  them in the docstring. Document semantic type info (shape, dtype,
  sign, units) inside the prose.
- Math: `.. math::` directive inside the docstring (rendered via
  MathJax).
- Cross-references: `:class:`, `:func:`, `:mod:` roles — Sphinx
  auto-resolves them.
- Doctest examples (`>>> `): `conf.py`'s `doctest_global_setup`
  pre-imports common modules so example blocks don't repeat the same
  imports.
- Deprecation: `.. deprecated:: 0.X` directive at the top of the
  docstring; emit a `DeprecationWarning` from the function body via
  `warnings.warn(...)`.

## Stub regeneration

Tycho ships a committed snapshot of the nanobind-generated `.pyi`
stubs at `tychopy/_stubs/`. The snapshot is what IDEs see for
auto-complete. **CI fails if the snapshot drifts from what
`nanobind_add_stub` produces against a freshly built `_tychopy`.**

If you change any binding signature, or the Python-side
`R"doc(...)"` docstring, regenerate the snapshot and commit it:

```bash
cd build && ninja tychopy_stubs_snapshot
git add tychopy/_stubs/
git commit -m "build: refresh _tychopy stubs"
```

## Commit messages

Use descriptive commit subjects with one of these conventional
prefixes:

| Prefix       | Use for                                                 |
| ------------ | ------------------------------------------------------- |
| `feat:`      | New features or capabilities                            |
| `fix:`       | Bug fixes                                               |
| `refactor:`  | Code restructuring without behavior change              |
| `docs:`      | Documentation-only changes                              |
| `chore:`     | Build system, CI, dependency, or tooling changes        |
| `build:`     | Changes to the CMake build, presets, or scikit-build    |
| `ci:`        | Changes to GitHub Actions workflows                     |
| `test:`      | New or updated tests                                    |

Subjects under ~70 characters; longer rationale belongs in the body.
