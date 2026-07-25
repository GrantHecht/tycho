#!/usr/bin/env bash
# Tycho Accelerate header hygiene canary.
#
# accelerate_interface.h includes Eigen's DisableStupidWarnings.h, which issues
# `#pragma clang diagnostic push`. If the matching ReenableStupidWarnings.h
# include is ever dropped again, Eigen's suppressions -- including
# -Wimplicit-int-float-conversion -- silently leak into the remainder of every
# TU that includes the header, which is most of the solver TUs plus the PCH.
# That is the warning class that masked the P3/P4 narrowing bugs.
#
# Two checks:
#   1. LEAK  -- a deliberate int->double conversion AFTER the include must warn.
#              Silence means the suppression leaked.
#   2. CLEAN -- accelerate_interface.h AND accelerate_utils.h (which it
#              includes, and whose diagnostics would otherwise be reported
#              under a different file path and slip past a narrower grep)
#              must emit no warnings of their own.
#
# Usage: scripts/check_accelerate_warnings.sh
#
# Environment overrides:
#   CXX -- C++ compiler (default: /opt/homebrew/opt/llvm/bin/clang++, then clang++)
#
# Exit code: 0 if both checks pass. Non-zero otherwise.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "SKIP: Apple-only header (USE_ACCELERATE_SPARSE); nothing to check on $(uname -s)."
    exit 0
fi

if [[ -z "${CXX:-}" ]]; then
    if [[ -x /opt/homebrew/opt/llvm/bin/clang++ ]]; then
        CXX=/opt/homebrew/opt/llvm/bin/clang++
    else
        CXX=clang++
    fi
fi

echo "==== Tycho Accelerate header hygiene canary ===="
echo "CXX : ${CXX} ($(${CXX} --version | head -1))"
echo "================================================"

WORK_DIR=$(mktemp -d -t tycho_accel_canary.XXXXXX)
trap 'rm -rf "${WORK_DIR}"' EXIT

cat >"${WORK_DIR}/canary.cpp" <<'EOF'
#include "tycho/detail/solvers/linear/accelerate_interface.h"

#include <Eigen/Sparse>

// Must warn under -Wimplicit-int-float-conversion. Silence => Eigen's
// suppression leaked past accelerate_interface.h.
//
// NOTE: deliberately `long long`, not `int`. Clang's
// -Wimplicit-int-float-conversion only fires when the source integer type's
// range exceeds what the destination float type's mantissa can represent
// exactly. A 32-bit int always fits exactly in a double's 53-bit mantissa,
// so int -> double never warns under this diagnostic regardless of
// suppression state (verified directly against both Homebrew LLVM clang++ 22
// and Apple clang++ 21). long long -> double can lose precision and reliably
// triggers it.
double tycho_accel_leak_canary(long long i) { return i; }

// Instantiate the solver so the header's own templates are compiled: a
// deprecation or narrowing warning inside them only fires on instantiation.
int tycho_accel_instantiate_canary() {
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    Eigen::AccelerateLDLTTPP<SpMat, Eigen::Upper> s;
    SpMat A(2, 2);
    A.insert(0, 0) = 2.0;
    A.insert(0, 1) = -1.0;
    A.insert(1, 1) = 2.0;
    A.makeCompressed();
    s.compute(A);
    Eigen::VectorXd b(2);
    b << 1.0, 1.0;
    Eigen::VectorXd x = s.solve(b);
    s.release();
    return (s.info() == Eigen::Success) ? 0 : static_cast<int>(x.size());
}
EOF

LOG="${WORK_DIR}/compile.log"
set +e
"${CXX}" -std=c++20 -O2 -DNDEBUG -DFMT_HEADER_ONLY -DUSE_ACCELERATE_SPARSE \
    -Wimplicit-int-float-conversion \
    -I "${REPO_ROOT}/include" -I "${REPO_ROOT}/dep/eigen" \
    -I "${REPO_ROOT}/dep/fmt/include" \
    -c "${WORK_DIR}/canary.cpp" -o "${WORK_DIR}/canary.o" 2>"${LOG}"
COMPILE_STATUS=$?
set -e

if [[ ${COMPILE_STATUS} -ne 0 ]]; then
    echo "FAIL: canary TU did not compile."
    tail -30 "${LOG}"
    exit 2
fi

RESULT=0

if grep -q 'tycho_accel_leak_canary' "${LOG}"; then
    echo "CHECK 1 (leak) : PASS -- suppression does not leak past the header."
else
    echo "CHECK 1 (leak) : FAIL -- int->double conversion after the include did NOT warn."
    echo "                 Eigen's diagnostic push is unbalanced: is"
    echo "                 ReenableStupidWarnings.h still included at the end of"
    echo "                 accelerate_interface.h?"
    RESULT=1
fi

if grep -Eq '(accelerate_interface|accelerate_utils)\.h.*warning:' "${LOG}"; then
    echo "CHECK 2 (clean): FAIL -- accelerate_interface.h/accelerate_utils.h emit their own warnings:"
    grep -E '(accelerate_interface|accelerate_utils)\.h.*warning:' "${LOG}" | sort -u | sed 's/^/                 /'
    RESULT=1
else
    echo "CHECK 2 (clean): PASS -- accelerate_interface.h and accelerate_utils.h emit no warnings."
fi

if [[ ${RESULT} -eq 0 ]]; then
    echo "==== OK ===="
else
    echo "==== CHANGED -- see docs/dev/plans/2026-07-24-accelerate-interface-overhaul-design.md I2 ===="
fi
exit ${RESULT}
