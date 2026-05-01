#!/usr/bin/env bash
# Tycho upstream Enzyme/Eigen canary.
#
# Two compile-time tests detect when upstream gains support for SIMD trig
# differentiation under Enzyme.  Both fail today; either passing is an
# actionable signal.
#
#   Test A — Eigen SIMD pcos<Packet4d> differentiable by Enzyme.
#            If passes: drop the per-lane scalar libm wrappers in
#            include/tycho/detail/utils/simd_math.h and call
#            x.cos() / x.sin() directly.
#
#   Test B — Custom Enzyme derivative composes with enzyme_width > 1.
#            If passes: an alternative architecture becomes viable —
#            keep Eigen's SIMD packet trig, register a custom derivative,
#            and let Phase 3 batching amortise across BW tangents.
#
# Usage:
#   scripts/upstream_canary/check.sh
#
# Environment overrides:
#   CXX            — C++ compiler (default: clang++)
#   ENZYME_PLUGIN  — path to ClangEnzyme-*.so
#                    (default: $HOME/Software/Enzyme/install/lib/ClangEnzyme-21.so)
#   EIGEN_INCLUDE  — path to Eigen include root
#                    (default: <repo>/dep/eigen)
#
# Exit code: 0 if everything is unchanged (both still fail). Non-zero if
# either test compiles or passes — meaning upstream has changed and we
# should investigate.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CXX="${CXX:-clang++}"
ENZYME_PLUGIN="${ENZYME_PLUGIN:-${HOME}/Software/Enzyme/install/lib/ClangEnzyme-21.so}"
EIGEN_INCLUDE="${EIGEN_INCLUDE:-${REPO_ROOT}/dep/eigen}"

echo "==== Tycho upstream Enzyme/Eigen canary ===="
echo "Date            : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CXX             : ${CXX} ($(${CXX} --version | head -1))"
echo "Enzyme plugin   : ${ENZYME_PLUGIN}"
if [[ -f "${ENZYME_PLUGIN}" ]]; then
    plugin_size=$(stat -c%s "${ENZYME_PLUGIN}" 2>/dev/null || stat -f%z "${ENZYME_PLUGIN}")
    plugin_date=$(stat -c%y "${ENZYME_PLUGIN}" 2>/dev/null || stat -f%Sm "${ENZYME_PLUGIN}")
    echo "  size=${plugin_size} bytes, mtime=${plugin_date}"
else
    echo "  ERROR: plugin not found"
    exit 2
fi
echo "Eigen include   : ${EIGEN_INCLUDE}"
EIGEN_VERSION_FILE=""
for candidate in "${EIGEN_INCLUDE}/Eigen/Version" \
                 "${EIGEN_INCLUDE}/Eigen/src/Core/util/Macros.h"; do
    if [[ -f "${candidate}" ]] && grep -q 'EIGEN_WORLD_VERSION' "${candidate}"; then
        EIGEN_VERSION_FILE="${candidate}"
        break
    fi
done
if [[ -n "${EIGEN_VERSION_FILE}" ]]; then
    eigen_str=$(awk -F'"' '/#define EIGEN_VERSION_STRING/ {print $2}' "${EIGEN_VERSION_FILE}")
    if [[ -n "${eigen_str}" ]]; then
        echo "  Eigen ${eigen_str}"
    else
        eigen_major=$(awk '/#define EIGEN_MAJOR_VERSION/ {print $3}' "${EIGEN_VERSION_FILE}")
        eigen_minor=$(awk '/#define EIGEN_MINOR_VERSION/ {print $3}' "${EIGEN_VERSION_FILE}")
        eigen_patch=$(awk '/#define EIGEN_PATCH_VERSION/ {print $3}' "${EIGEN_VERSION_FILE}")
        echo "  Eigen ${eigen_major}.${eigen_minor}.${eigen_patch}"
    fi
fi
echo "============================================="
echo

WORK_DIR=$(mktemp -d -t tycho_canary.XXXXXX)
trap 'rm -rf "${WORK_DIR}"' EXIT

run_test() {
    local label="$1" src="$2" extra_flags="$3"
    local out="${WORK_DIR}/${label}"
    local log="${WORK_DIR}/${label}.log"

    echo "----- Test ${label}: ${src} -----"
    if "${CXX}" -std=c++20 -O3 -march=native -ffast-math -fno-finite-math-only \
        -fplugin="${ENZYME_PLUGIN}" \
        ${extra_flags} \
        "${SCRIPT_DIR}/${src}" -o "${out}" 2>"${log}"; then
        echo "COMPILE: succeeded"
        if "${out}" >"${log}.run" 2>&1; then
            echo "RUN:     succeeded"
            cat "${log}.run"
            return 0
        else
            echo "RUN:     FAILED (compile passed but runtime mismatched)"
            cat "${log}.run"
            return 2
        fi
    else
        echo "COMPILE: failed (expected today — upstream unchanged)"
        echo "---- compile log (last 15 lines) ----"
        tail -15 "${log}"
        echo "-------------------------------------"
        return 1
    fi
}

set +e
run_test test_a test_a_eigen_simd_trig.cpp "-isystem ${EIGEN_INCLUDE}"
RESULT_A=$?
echo
run_test test_b test_b_custom_deriv_batched.cpp ""
RESULT_B=$?
set -e

echo
echo "==== Summary ===="
case ${RESULT_A} in
    0) echo "Test A:  PASS — upstream now differentiates Eigen SIMD trig."
       echo "         ACTION: simplify simd_math.h to call x.cos() / x.sin() directly." ;;
    1) echo "Test A:  fail (status quo — upstream unchanged)" ;;
    2) echo "Test A:  COMPILES BUT WRONG RESULT — upstream regressed.  Investigate." ;;
esac
case ${RESULT_B} in
    0) echo "Test B:  PASS — custom derivative now composes with enzyme_width > 1."
       echo "         ACTION: consider register-custom-derivative + Phase 3 batching arch." ;;
    1) echo "Test B:  fail (status quo — upstream unchanged)" ;;
    2) echo "Test B:  COMPILES BUT WRONG RESULT — upstream regressed.  Investigate." ;;
esac

# Exit code: 0 only if both still fail (status quo).
if [[ ${RESULT_A} -ne 1 || ${RESULT_B} -ne 1 ]]; then
    echo
    echo "STATE CHANGED — see docs/superpowers/specs/2026-04-26-enzyme-trig-spike.md and CLAUDE.md for the dependent code."
    exit 1
fi
exit 0
