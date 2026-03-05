#!/usr/bin/env bash
#
# bench_compile.sh — time a clean Ninja build of Tycho
#
# Wipes the build directory, runs CMake configuration, then times
# `ninja -j6 all` for N trials (using `ninja -t clean` between trials
# to avoid re-running CMake each time).
#
# Usage:
#   bash bench_compile.sh [--label LABEL] [--output FILE] [--trials N]
#
# Options:
#   --label  LABEL   Name for this build (default: current git branch)
#   --output FILE    Write results JSON to FILE (default: print to stdout)
#   --trials N       Number of build trials (default: 3)
#
# Prerequisites: same as config_and_build.sh
#   - conda env "tycho" must exist
#   - Homebrew LLVM installed (see LLVM_VERSION below)
#   - Ninja installed via Homebrew
#
# NOTE: This script wipes the build directory. Any existing build artifacts
# (including the installed _tycho extension) will be rebuilt from scratch.
# Run your runtime benchmarks before running this script, or rebuild
# afterwards with: cd build && ninja -j6 all
#
# NOTE: OS page-cache warming between trials
#   `ninja -t clean` removes compiled artifacts but NOT source files.  macOS
#   keeps recently-read source files in the page cache, so trials 2+ read from
#   RAM instead of disk.  Observed effect: trial 1 ≈737s, trial 3 ≈410s on
#   Apple M-series (3-trial nanobind run, 2026-03-02).
#
#   Recommended practice for cross-branch comparisons:
#     - Use --trials 1 (cold-cache, most reproducible across machines / CI)
#     - Or run `sudo purge` before each trial for true cold-cache isolation
#       (requires sudo; very slow; better for rigorous measurement)
#     - Multi-trial runs with this script measure warm-cache build time, which
#       is useful for developer-workflow comparisons but not cold-start CI time.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_VERSION=22.1.0
CONDA_ENV=tycho
REPO_DIR="$(dirname "${SCRIPT_DIR}")"  # one level up from bench/
BUILD_DIR="${REPO_DIR}/build"

LABEL=""
OUTPUT=""
TRIALS=3

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)   LABEL="$2";  shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --trials)  TRIALS="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^#/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Default label to git branch
if [[ -z "$LABEL" ]]; then
    LABEL=$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
fi

# ---------------------------------------------------------------------------
# Resolve conda environment paths (mirrors config_and_build.sh)
# ---------------------------------------------------------------------------
PYTHON=$(conda run -n "${CONDA_ENV}" which python)
SITE_PACKAGES=$(conda run -n "${CONDA_ENV}" python -c "import site; print(site.getsitepackages()[0])")

echo "=== Tycho Compilation Benchmark ==="
echo "Label:        ${LABEL}"
echo "Trials:       ${TRIALS}"
echo "Build dir:    ${BUILD_DIR}"
echo "Python:       ${PYTHON}"
echo ""

# ---------------------------------------------------------------------------
# Clean build directory and run CMake configuration
# ---------------------------------------------------------------------------
echo "Wiping build directory..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

echo "Running CMake configuration..."
/opt/homebrew/bin/cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_C_COMPILER:FILEPATH=/opt/homebrew/Cellar/llvm/${LLVM_VERSION}/bin/clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=/opt/homebrew/Cellar/llvm/${LLVM_VERSION}/bin/clang++ \
    -DPython_EXECUTABLE="${PYTHON}" \
    -DPYTHON_LOCAL_INSTALL_DIR="${SITE_PACKAGES}" \
    --no-warn-unused-cli \
    -S "${REPO_DIR}" -B "${BUILD_DIR}" -G Ninja
echo ""

# ---------------------------------------------------------------------------
# Build trials
# ---------------------------------------------------------------------------
echo "Running ${TRIALS} build trial(s)..."
TIMES=()

for i in $(seq 1 "${TRIALS}"); do
    echo "--- Trial ${i}/${TRIALS} ---"

    # Clean compiled outputs between trials (preserves CMake/Ninja config)
    if [[ $i -gt 1 ]]; then
        ninja -C "${BUILD_DIR}" -t clean
    fi

    # Time the build.  Use two Python calls against the system monotonic clock
    # (time.monotonic is system-wide on macOS) to get sub-second precision
    # without any external timing tool.
    T_START=$("${PYTHON}" -c "import time; print(time.monotonic())")
    ninja -C "${BUILD_DIR}" -j6 all
    T_END=$("${PYTHON}" -c "import time; print(time.monotonic())")
    ELAPSED=$("${PYTHON}" -c "print(${T_END} - ${T_START})")

    TIMES+=("${ELAPSED}")
    echo "Trial ${i} time: ${ELAPSED}s"
    echo ""
done

# ---------------------------------------------------------------------------
# Emit JSON (mean ± std over trials)
# ---------------------------------------------------------------------------
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

JSON=$("${PYTHON}" - "${LABEL}" "${TIMESTAMP}" "${TIMES[@]}" <<'PYEOF'
import sys, json, math

label     = sys.argv[1]
timestamp = sys.argv[2]
times     = [float(x) for x in sys.argv[3:]]

mean = sum(times) / len(times)
variance = sum((t - mean) ** 2 for t in times) / len(times)
std = math.sqrt(variance)

result = {
    "label":     label,
    "timestamp": timestamp,
    "trials":    times,
    "mean_s":    round(mean, 3),
    "std_s":     round(std, 3),
}
print(json.dumps(result, indent=2))
PYEOF
)

echo "=== Results ==="
echo "${JSON}"

if [[ -n "${OUTPUT}" ]]; then
    echo "${JSON}" > "${OUTPUT}"
    echo ""
    echo "Results saved to ${OUTPUT}"
fi
