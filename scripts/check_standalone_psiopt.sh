#!/usr/bin/env bash
# Proves the psiopt project configures, builds, and tests standalone.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
BUILD=build-psiopt-standalone
cmake -S psiopt -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="${CC:?activate the tycho conda env}" \
      -DCMAKE_CXX_COMPILER="${CXX:?activate the tycho conda env}"
cmake --build "$BUILD" -j2
ctest --test-dir "$BUILD" --output-on-failure
