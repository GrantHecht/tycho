# Tycho

Tycho is a modular, extensible library for trajectory design and optimal control.
It uses a custom implementation of vector math formalisms to enable rapid 
implementation of dynamical systems and automatic differentiation. The phase object 
is the core of the optimal control functionality, and by linking multiple phases 
together, the user can construct scenarios of arbitrary complexity. A high performance 
interior-point optimizer (PSIOPT) is included with the library, which enables quick 
turnaround from concept to solution.

## Origin

Tycho is an independently maintained fork of 
[ASSET (Astrodynamics Software and Science Enabling Toolkit)](https://github.com/AlabamaASRL/asset_asrl), 
originally developed by the Astrodynamics and Space Research Laboratory at the 
University of Alabama. Full credit and thanks are owed to the original authors 
for building the foundation this project is built upon.

Original ASSET development was funded by NASA under Grant No. 80NSSC19K1643.

## Building from Source

Tycho is a C++20 / Python library built with CMake and nanobind. The build
produces a native extension module (`_tycho`) and installs it alongside the
pure-Python `tycho` package into your active Python environment.

### Prerequisites

| Requirement | macOS (Apple Silicon) | Linux / WSL2 (Ubuntu) | Windows x64 |
|---|---|---|---|
| C++ compiler | LLVM/Clang via Homebrew | Clang 22+ | LLVM/Clang (clang-cl) |
| Build tools | `brew install ninja ccache` | `sudo apt install ninja-build ccache` | Ninja, CMake |
| Sparse solver | Apple Accelerate (built-in) | Intel MKL (via oneAPI) | Intel MKL |
| Python | 3.12+ (conda or venv) | 3.12+ (system, conda, or venv) | 3.12+ |

**Python packages** (all platforms):
```bash
pip install numpy scipy matplotlib spiceypy
```

### macOS (Apple Silicon)

Install LLVM and build tools:
```bash
brew install llvm ninja ccache
```

Set up a Python environment:
```bash
conda create -n tycho python=3.13
conda activate tycho
pip install numpy scipy matplotlib spiceypy
```

Build:
```bash
git clone git@github.com:GrantHecht/tycho.git
cd tycho
mkdir build
cmake --preset macos-llvm-release
cd build && ninja -j2 all
```

The preset uses the stable Homebrew symlink `/opt/homebrew/opt/llvm` for the
compiler, so it does not need updating when Homebrew upgrades LLVM. Python is
auto-detected from the active conda/venv environment.

### Linux / WSL2 (Ubuntu)

Install compiler toolchain:
```bash
sudo apt install clang-22 llvm-22 llvm-22-dev libomp-22-dev lld-22 ninja-build ccache
```

Install Intel MKL:
```bash
# Add Intel APT repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] \
  https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install intel-oneapi-mkl-devel

# Set MKLROOT (add to ~/.bashrc for persistence)
source /opt/intel/oneapi/setvars.sh
```

Set up a Python environment (system Python 3.12+, or use conda/venv):
```bash
pip install numpy scipy matplotlib spiceypy
```

Build:
```bash
git clone git@github.com:GrantHecht/tycho.git
cd tycho
mkdir build
cmake --preset linux-clang-release
cd build && ninja -j8 all
```

Python is auto-detected from `$PATH`. To target a specific interpreter, pass
`-DPython_EXECUTABLE=/path/to/python` during the configure step.

### Windows x64

Uses LLVM/Clang with the clang-cl frontend. See `CMakePresets.json` for
compiler paths. Sparse solver: Intel MKL.

```bash
git clone git@github.com:GrantHecht/tycho.git
cd tycho
cmake --preset x64-Clang-Release
cmake --build build --config Release -j8
```

### Rebuilding After Changes

After modifying C++ source files, rebuild from the `build` directory:
```bash
cd build && ninja -j<N> all    # N = 2 on macOS, 8 on Linux/Windows
```

Only changed files will be recompiled. Pure-Python changes in `tycho/` take
effect immediately (the build step copies the package to site-packages).

### CMake Options

These can be passed as `-D<VARIABLE>=<VALUE>` during the configure step:

| Variable | Default | Purpose |
|---|---|---|
| `Python_EXECUTABLE` | auto-detected | Path to Python interpreter |
| `PYTHON_LOCAL_INSTALL_DIR` | `python -m site --user-site` | Site-packages install directory |
| `TYCHO_FP_MODE` | `SAFER_FAST` | Floating-point mode: `STRICT`, `SAFER_FAST`, or `FAST` |
| `BUILD_CPP_TESTS` | `OFF` | Build C++ unit tests (Google Test, fetched via FetchContent) |
| `BUILD_CPP_BENCHMARKS` | `OFF` | Build C++ benchmarks (Google Benchmark, fetched via FetchContent) |
| `BUILD_CPP_EXAMPLES` | `ON` | Build standalone C++ example programs |

### Verifying the Build

```bash
# Quick smoke test — import the module
python -c "import tycho; print('tycho loaded successfully')"

# Run the C++ brachistochrone example (should print "Optimal Solution Found")
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp

# Run the Python example suite
python scripts/run_examples.py
```

## Documentation

Documentation is coming soon.

## Citation

If you use Tycho in a published work, please cite the original ASSET paper that 
this project is based upon:
```bibtex
@inproceedings{Pezent2022,
        author = {James B. Pezent and Jared Sikes and William Ledbetter and Rohan Sood and Kathleen C. Howell and Jeffrey R. Stuart},
        title = {ASSET: Astrodynamics Software and Science Enabling Toolkit},
        booktitle = {AIAA SCITECH 2022 Forum},
        pages = {AIAA 2022-1131},
        year={2022},
        doi = {10.2514/6.2022-1131}
}
```