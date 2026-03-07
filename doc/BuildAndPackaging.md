# Build and Packaging

## Development builds

The primary development workflow uses CMake and ninja directly.
`config_and_build.sh` (macOS/Linux) or `build.ps1` (Windows) handle environment
setup and invoke the build:

```bash
# macOS / Linux — first time
bash config_and_build.sh

# Subsequent C++ rebuilds
cd build && ninja -j8 all
```

```powershell
# Windows — every build
powershell -File build.ps1
```

When `PYTHON_LOCAL_INSTALL_DIR` is set (resolved automatically by the build
scripts to the active conda environment's site-packages), a CMake POST_BUILD
command copies `_tycho` and the `tycho/` package there immediately after
compilation. No separate install step is required during daily development.

## Distribution wheels

Wheel building uses **scikit-build-core** as the PEP 517 build backend, declared
in `pyproject.toml`. It drives the CMake build internally and collects the
compiled extension and the pure-Python package into a platform-tagged wheel.

```bash
pip install scikit-build-core build
python -m build --wheel
```

The resulting wheel is written to `dist/`.

### Windows: bundling DLLs with delvewheel

On Windows, `_tycho.pyd` depends on `libiomp5md.dll` (Intel OpenMP, used by
MKL) and potentially other non-system DLLs. End users will not have these on
their search path. After building the wheel, run **delvewheel** to copy the
required DLLs into the wheel and patch the load path:

```bash
pip install delvewheel
delvewheel repair dist/tycho-*.whl
```

The repaired wheel in `dist/repaired/` is self-contained and suitable for
upload to PyPI.

## Why these steps are kept separate from the CMake build

`python -m build` calls scikit-build-core which internally invokes CMake.
Embedding it inside an existing CMake build would create a circular dependency
and force a full recompile from scratch, wasting build artifacts. Similarly,
`delvewheel repair` is a post-processing step on a finished `.whl` file — it
has no role during compilation.

The conventional separation used by the scientific Python ecosystem is:

| Step | Tool | When |
|---|---|---|
| C++ + dev install | `cmake --build` / `ninja` | Daily development |
| Source distribution | `python -m build --sdist` | Release |
| Binary wheel | `python -m build --wheel` | CI / release |
| Bundle DLLs (Windows) | `delvewheel repair dist/*.whl` | CI / release, after wheel |
| Publish | `twine upload dist/repaired/*.whl` | Release |

In CI (e.g. GitHub Actions) these are natural sequential steps within a job.

## Publishing to PyPI

```bash
pip install twine
twine upload dist/repaired/*.whl   # Windows
twine upload dist/*.whl            # macOS / Linux
```
