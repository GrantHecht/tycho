(getting-started-install)=
# Installing Tycho

Tycho is currently distributed as source and builds into a
[conda](https://docs.conda.io/) environment — there is no PyPI package yet, so
"installing" means building the `_tychopy` extension into your environment. The
build is driven by CMake + Ninja and takes a few minutes the first time.

This page is the quick path to a working `import tychopy`. For wheels, the
documentation build, and the full list of CMake options, see
{doc}`Building from Source </contributing/building>`.

## 1. Create the conda environment

The build picks up the active interpreter from `$PATH`, so create and activate
the `tycho` environment first:

```bash
conda create -n tycho python=3.13
conda activate tycho
pip install numpy scipy matplotlib spiceypy
```

## 2. Get the source

Tycho vendors three submodules (`dep/eigen`, `dep/fmt`, `dep/nanobind`), so clone
recursively:

```bash
git clone --recurse-submodules https://github.com/GrantHecht/tycho.git
cd tycho
```

(If you already cloned without `--recurse-submodules`, run
`git submodule update --init --recursive`. The CMake helpers also initialize them
on the first configure.)

## 3. Build for your platform

Each platform has a CMake preset and a sparse linear solver. **Always use the
preset for your platform.** The configure-and-build pair is the same everywhere;
the per-platform setup differs.

::::{tab-set}
:::{tab-item} macOS (Apple Silicon)
System tools via Homebrew; the sparse solver is **Apple Accelerate** (ships with
macOS, detected automatically):

```bash
brew install llvm ninja ccache jq
mkdir build
cmake --preset macos-llvm-release
cd build && ninja -j6 all
```
:::
:::{tab-item} Linux / WSL2
The recommended preset uses the conda-forge clang toolchain, so the extension
links against conda's libstdc++ and imports cleanly regardless of the host
distro. The sparse solver is **Intel oneAPI MKL**.

```bash
# conda-forge clang toolchain (into the tycho env)
conda install -n tycho -c conda-forge \
  clangxx_linux-64 clang_linux-64 libstdcxx-devel_linux-64 sysroot_linux-64

# system Ninja + ccache
sudo apt install ninja-build ccache      # Debian/Ubuntu
# sudo dnf install ninja-build ccache    # Fedora

# Intel oneAPI MKL (see the building guide for the apt-repo setup),
# then source it (add to your shell rc):
source /opt/intel/oneapi/setvars.sh

mkdir build
cmake --preset linux-clang-conda
cd build && ninja -j6 all
```
:::
:::{tab-item} Windows x64
The Windows build uses LLVM/Clang with the `clang-cl` frontend and **Intel
oneAPI MKL**. Install LLVM (with `clang-cl` and `lld-link`), Ninja, and MKL; the
preset hard-codes the LLVM paths (see `CMakePresets.json`). This preset builds
into `out/build/x64-Clang-Release/`, so build it through the preset rather than
`cd`-ing into a directory:

```bat
cmake --preset x64-Clang-Release
cmake --build --preset x64-Clang-Release
```
:::
::::

:::{note}
`ninja -j6` is the upper bound on a 32 GB machine; use `-j4` on 16 GB. Tycho's
`heavy_compile` job pool auto-throttles RAM-heavy translation units, so lighter
files keep filling the remaining slots.
:::

The build installs both `_tychopy.*.so` and the `tychopy/` package directly into
your environment's `site-packages`.

## 4. Verify the install

```bash
python -c "from tychopy import vector_functions as vf; print(vf.Arguments(3).compute([1.0, 2.0, 3.0]))"
```

You should see `[1. 2. 3.]` — the identity VectorFunction evaluated at
$(1, 2, 3)$.

## Next steps

- {doc}`Your first Tycho program <first_program>` — the five-minute path from a
  working import to a runnable, self-differentiating dynamics model.
- {doc}`The first-VectorFunction tutorial </user_guide/first_vectorfunction>`
  — build and evaluate VectorFunctions in depth, working up to real dynamics.
- {doc}`Building from Source </contributing/building>` — wheels, the docs build,
  and every CMake option.
