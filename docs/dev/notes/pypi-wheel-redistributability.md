# Making the Linux wheel redistributable for PyPI

**Status:** planning note — no build behavior changes yet.
**Owner decision required:** yes (Intel MKL dependency strategy; PSIOPT-adjacent).

## Problem

The Linux `_tychopy` wheel is **not redistributable** today. `cmake/FindMKL.cmake`
(lines ~245–264) appends the **absolute** build-time MKL and Intel-OpenMP
directories to `CMAKE_INSTALL_RPATH`:

```cmake
list(APPEND CMAKE_INSTALL_RPATH "${MKL_RPATH_DIR}")      # e.g. /opt/intel/oneapi/mkl/<ver>/lib
list(APPEND CMAKE_INSTALL_RPATH "${MKL_OMP_RPATH_DIR}")  # e.g. /opt/intel/oneapi/compiler/<ver>/lib
```

So the shipped `.so` hard-codes the *build machine's* library paths and assumes
`libmkl_rt.so` / `libiomp5.so` physically exist there at import time. The baked
RPATH observed in CI:

```
[/usr/share/miniconda/envs/tycho/lib : /opt/intel/oneapi/mkl/2026.0/lib : /opt/intel/oneapi/compiler/2026.0/lib]
```

This works today only because every consumer (dev machines, all CI lanes) has
oneAPI + the `tycho` conda env at those exact absolute paths. The first clean
environment fails immediately with:

```
ImportError: libiomp5.so: cannot open shared object file: No such file or directory
```

That is exactly the failure that forced `wheel-layout.yml`'s import/test steps
to run **on the build runner** instead of bare runners (see the job-layout
comment in that workflow). The CI constraint is a symptom of this same root
cause. A `pip install tychopy` from PyPI onto any other machine would fail the
same way.

The MKL dependency is **Pardiso** — `Eigen::PardisoLDLT` in
`include/tycho/detail/solvers/psiopt.h` (KKT factorization on PSIOPT's hot
path), via `include/tycho/detail/solvers/linear/pardiso_interface.h`. macOS
already uses Apple Accelerate (`accelerate_interface.h`); MKL is the
Linux/Windows sparse backend. So MKL is load-bearing, not peripheral.

## Decision: Option A — MKL as a runtime dependency, NOT vendored

We will **not** bundle Intel binaries (`libmkl_rt`, `libiomp5`) into the wheel,
to avoid taking on Intel's redistribution terms (CLAUDE.md flags MKL
redistribution as human-review-sensitive). Intel publishes these as
redistributable packages on **PyPI** (`mkl`, `intel-openmp`, `mkl-devel`) and on
**conda-forge** (`mkl`, `intel-openmp`); the user (or their package manager)
pulls them — we are not the redistributor.

Rejected alternatives:
- **Vendor MKL via auditwheel** — solves portability but makes us the Intel
  redistributor. Rejected on licensing grounds (the point of this note).
- **OpenBLAS-backed wheel** (BSD, freely vendorable, fully self-contained) —
  would require replacing Pardiso with an OpenBLAS-compatible sparse direct
  solver (Eigen `SimplicialLDLT`, or SuiteSparse/CHOLMOD). That is a real
  solver swap on the optimizer hot path with convergence/performance
  implications — a separate engineering project, not a packaging change. Keep
  as a possible future "pure-permissive-stack" build, not the PyPI path.

## Implementation sketch (when scheduled)

The mechanical pieces are independent of the licensing question and can be
designed now; only "where does `libmkl_rt` come from at the user's runtime" is
the decided part (Option A: a PyPI/conda dependency).

1. **Relative RPATH.** Change `FindMKL.cmake` so the installed `.so` uses an
   `$ORIGIN`-relative RPATH instead of absolute build-time dirs. (Keep the
   absolute `CMAKE_BUILD_RPATH` entries — they are needed so the *build-tree*
   binary imports without `LD_LIBRARY_PATH` for nanobind stubgen and the docs
   autodoc/doctest import, per the existing comment. Only `CMAKE_INSTALL_RPATH`
   needs to change.)
2. **`auditwheel repair` with exclusions.** Post-build, run
   `auditwheel repair --exclude libmkl_rt.so.2 --exclude libiomp5.so` (exact
   sonames TBD). This rewrites our own RPATH to `$ORIGIN/..`-relative and
   produces a `manylinux_x_y` tag, while **leaving the Intel libs out** of the
   wheel — so no Intel binaries ship inside it.
3. **Runtime dependency.** Add `mkl` (and `intel-openmp` if needed) to
   `[project].dependencies` in `pyproject.toml` (PyPI install), and document the
   conda-forge equivalent for conda users. The Intel pip package installs
   `libmkl_rt` into the env's `lib/`; the loader resolves it via the env, with
   `$ORIGIN`-relative RPATH as the fallback.
4. **Build under manylinux.** Build the PyPI wheel in a manylinux container (or
   via `cibuildwheel`) for a correct glibc/ABI floor, against `mkl-devel` from
   PyPI rather than the oneAPI apt install.
5. **CI follow-through.** Once the wheel is self-sufficient (Intel libs resolved
   via the runtime dependency), the import + pytest jobs in `wheel-layout.yml`
   can move back onto bare runners (install the wheel + its deps, no oneAPI/conda
   toolchain). That reverses the "must run on the build runner" workaround and
   is the canary that the redistributability fix actually worked.

## Cross-references

- `cmake/FindMKL.cmake` — current absolute-RPATH logic (~245–264).
- `include/tycho/detail/solvers/psiopt.h`,
  `include/tycho/detail/solvers/linear/pardiso_interface.h` — the Pardiso/MKL
  dependency being satisfied.
- `.github/workflows/wheel-layout.yml` — import/test pinned to the build runner
  as a workaround for this issue.
- CLAUDE.md → "Things to Flag for Human Review" (Intel MKL redistribution;
  PyPI/packaging) — both apply here.
