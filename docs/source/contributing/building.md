# Building from Source

Tycho is a CMake + nanobind project. A successful build produces:

- `_tychopy.cpython-<ver>-<platform>.so` — the C++ extension
- `tychopy/` — a pure-Python package that wraps it
- Optionally: HTML documentation, C++ examples, C++ tests, benchmarks

Both `_tychopy.so` and the `tychopy/` package are installed into the
active Python environment's `site-packages` by the build step, so you
can `import tychopy` immediately after building.

## Prerequisites

Tycho assumes a [conda](https://docs.conda.io/) environment named
`tycho`. The build picks up the active Python interpreter from `$PATH`,
so the env must be active before you configure.

```bash
conda create -n tycho python=3.13
conda activate tycho
pip install numpy scipy matplotlib spiceypy seaborn
```

For all 32 Python example scripts to run (some need extra packages):

```bash
conda install -n tycho -c conda-forge basemap
```

## Source checkout

Tycho vendors three submodules (`dep/eigen`, `dep/fmt`, `dep/nanobind`).
Either clone with `--recurse-submodules`:

```bash
git clone --recurse-submodules https://github.com/GrantHecht/tycho.git
```

…or initialize after cloning:

```bash
git clone https://github.com/GrantHecht/tycho.git
cd tycho
git submodule update --init --recursive
```

(The CMake helpers under `cmake/git-submodule-*.cmake` will also
initialize submodules automatically on the first configure if you
forget.)

## Configure and build

Tycho ships CMake presets in `CMakePresets.json`. **Always use the
preset for your platform** rather than running `cmake` with manual
flags.

| Platform               | Configure preset            |
| ---------------------- | --------------------------- |
| macOS (Apple Silicon)  | `macos-llvm-release`        |
| Linux / WSL2           | `linux-clang-conda`         |
| Windows x64            | `x64-Clang-Release`         |

Each platform has its own setup steps below; the configure-and-build
pair at the end is identical:

```bash
cmake --preset <preset-name>
cd build && ninja -j6 all
```

`ninja -j6` is the upper bound on a 32 GB system. Use `-j4` on a 16 GB
system. Tycho's `heavy_compile` ninja job pool auto-throttles RAM-heavy
translation units (1 slot per 10 GB of system RAM), so light TUs
continue to fill the remaining job slots — you can safely use higher
`-j` values without OOMing.

### macOS (Apple Silicon)

```bash
brew install llvm ninja ccache jq
```

The sparse linear solver on macOS is **Apple Accelerate** (ships with
the OS, detected automatically).

```bash
mkdir build
conda activate tycho
cmake --preset macos-llvm-release
cd build && ninja -j6 all
```

### Linux / WSL2

The recommended Linux preset (`linux-clang-conda`) uses the conda-forge
clang toolchain — the resulting `.so` links against conda's libstdc++ so
it is import-compatible with the conda env regardless of how new the
host distro's GCC is.

Install the conda toolchain into the `tycho` env:

```bash
conda install -n tycho -c conda-forge \
  clangxx_linux-64 clang_linux-64 \
  libstdcxx-devel_linux-64 sysroot_linux-64
```

Install system Ninja and ccache:

```bash
sudo apt install ninja-build ccache    # Debian/Ubuntu
sudo dnf install ninja-build ccache    # Fedora
```

The sparse linear solver on Linux is **Intel oneAPI MKL**. Install
oneAPI MKL via the Intel apt repository (Debian/Ubuntu instructions —
see [the Intel oneAPI installation
guide](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html)
for other distros):

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install intel-oneapi-mkl-devel
source /opt/intel/oneapi/setvars.sh       # add to ~/.bashrc
```

```bash
mkdir build
conda activate tycho
cmake --preset linux-clang-conda
cd build && ninja -j6 all
```

A `linux-clang-release` preset using the system clang at `/usr/bin/clang++`
is also available; use it if you need system libstdc++ symbols or are
building with `ENABLE_ENZYME_AD=ON` against a specific system-clang
version. On most distros the two presets produce equivalent wheels.

### Windows x64

The Windows build uses LLVM/Clang with the `clang-cl` frontend.
Install LLVM (with `clang-cl` and `lld-link`), Ninja, and Intel oneAPI
MKL. The preset hard-codes paths under `C:/Program Files/LLVM/bin/` —
see `CMakePresets.json` for the exact files used.

```bat
cmake --preset x64-Clang-Release
cd build & ninja -j6 all
```

## Building a wheel

Tycho ships with a `pyproject.toml` driven by
[scikit-build-core](https://scikit-build-core.readthedocs.io/). To
produce a Python wheel:

```bash
conda activate tycho
pip wheel . --no-deps -w dist/
```

The resulting wheel contains the C++ extension, its type stubs, and the
`tychopy/` Python package. On Linux, the wheel is built with whichever
compiler is on `$CC`/`$CXX` in the active env — the conda toolchain
installed above provides a binary that imports cleanly under the conda
env.

## Building the documentation

The docs build is opt-in via the `BUILD_SPHINX_DOCS` CMake option.

```bash
pip install -e ".[docs]"                       # Sphinx + theme + extensions (also compiles _tychopy)
cmake --preset <preset-name> -DBUILD_SPHINX_DOCS=ON
cd build && ninja tycho_docs
xdg-open docs/html/index.html                  # or `open` on macOS
```

Because the build backend is scikit-build-core, the editable
`pip install -e ".[docs]"` also compiles the `_tychopy` extension — it is
not a docs-tooling-only install, so expect a multi-minute native build the
first time and make sure the conda toolchain is active (see the platform
sections above).

Additional docs targets:

- `tycho_doxygen` — Doxygen XML output only (fast, no Sphinx)
- `tycho_docs_doctest` — runs `>>> ` doctest blocks
- `tycho_docs_linkcheck` — checks every external URL
- `tycho_docs_serve` — Sphinx live-reload server on port 8000
- `tycho_docs_clean` — wipes the HTML and Doxygen output trees

If you edit any binding signature or Python-side docstring inside a
nanobind `.def(...)` call, regenerate the stub snapshot and commit it:

```bash
cd build && ninja tychopy_stubs_snapshot
git add tychopy/_stubs/
```

CI fails if the committed snapshot drifts from what `nanobind_add_stub`
produces against the freshly built `_tychopy`.

## CMake configuration options

The most commonly overridden CMake variables:

| Variable                      | Purpose                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| `BUILD_SPHINX_DOCS`           | `ON` to build documentation (needs the `docs` optional extra installed).  |
| `BUILD_CPP_EXAMPLES`          | `ON` to build C++ examples under `examples/cpp_examples/`.                |
| `BUILD_CPP_TESTS`             | `ON` to build C++ unit tests (Google Test via FetchContent).              |
| `BUILD_CPP_BENCHMARKS`        | `ON` to build benchmarks (Google Benchmark via FetchContent).             |
| `TYCHO_FP_MODE`               | Floating-point mode: `STRICT`, `SAFER_FAST` (default), or `FAST`.         |
| `TYCHO_HEAVY_COMPILE_JOBS`    | Override the auto-detected heavy-TU job-pool depth (default: 1 per 10 GB).|

Pass with `-D`, e.g.

```bash
cmake --preset linux-clang-conda -DBUILD_CPP_TESTS=ON -DBUILD_CPP_BENCHMARKS=ON
```

## Subsequent builds

After the first configure, rebuilds are just:

```bash
cd build && ninja -j6
```

If you change `CMakeLists.txt` or any of the preset cache variables,
re-run the configure step before rebuilding. Ninja itself will pick up
most other source-tree changes automatically.
