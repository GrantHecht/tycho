#!/usr/bin/env bash
# check_abs_sweep.sh — CI lint: bare abs()/pow() calls on floating-point values.
#
# Background (PR7 T1): `abs(x)`/`pow(x, y)` written without qualification
# resolve via ordinary/ADL lookup, and on some toolchains or include orders
# only the integer `::abs(int)` overload (from <cstdlib>) is visible,
# silently truncating a floating-point argument to zero-or-one. This class of
# bug was swept once across ~28 sites in the 2026-07 review series (the
# `using std::abs;` ADL pattern in kepler_utils.h / lambert_solvers.h /
# root_finder.h is the sanctioned fix shape); this script is the CI guard
# that keeps it from being silently reintroduced. -Wabsolute-value in
# CMakeLists.txt is the compile-time half of the same guard.
#
# Invariant enforced, for each of `abs` and `pow` independently: a bare call
# `abs(` / `pow(` (regex `(?<![\w:.>])abs\(` — i.e. not already qualified by
# `::`, `.`, or a preceding word character, so `std::abs(`, `x.abs(`,
# `foo_abs(` are all correctly excluded) found in include/tycho, src, or
# extensions (*.h, *.cpp) is allowed only if:
#   (a) the containing file also contains a top-level `using std::abs;` /
#       `using std::pow;` — the sanctioned ADL-guard pattern used throughout
#       the astro/kepler and root_finder code, or
#   (b) the exact file:line is on one of the explicit allowlists below.
#
# KNOWN LIMITATION (accepted, not a bug in this script): the `using std::abs`
# grant in (a) is FILE-level, not function-scope-level. A file could contain
# the using-declaration inside one function and an unrelated unguarded bare
# `abs(` in a different function, and this script would not catch it. Being
# function-scope-aware would require real C++ parsing, which is out of scope
# for a bash+grep repo-hygiene lint — this is a documented trade-off, not an
# oversight. Treat a green run here as "no *newly introduced, unguarded*
# bare abs/pow", not as a proof of correctness.
#
# Also note: this script treats any line whose content (after stripping
# leading whitespace) starts with `//` as a comment and skips it, so doc
# comments that merely *mention* `abs(`/`pow(` don't need allowlist entries.
# This is a line-oriented heuristic, not a real comment parser: it will not
# catch a trailing `// abs(...)` comment appended after real code on the same
# line (no such case exists in the tree today), nor `abs(`/`pow(` mentioned
# inside a raw string literal (e.g. a nanobind docstring) — the latter is
# handled via an explicit allowlist entry instead.
#
# Usage: scripts/check_abs_sweep.sh
# Exit code: 0 if clean, non-zero (with offending file:line(s) printed) if
# any unguarded bare abs()/pow() call sites are found.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SCAN_DIRS=(include/tycho psiopt/include/tycho psiopt/src src extensions)

# ---------------------------------------------------------------------------
# Allowlists — exact "path:line" entries. Update the line number if the
# allowlisted line moves; that keeps the allowlist honest (a stale entry
# silently stops suppressing anything, so the check re-flags the site until
# the entry is fixed, rather than silently covering the wrong line forever).
# ---------------------------------------------------------------------------

# abs() — function *definitions* named `abs` (the naive regex also matches
# the declaration itself, not just call sites).
ABS_ALLOWLIST=(
    # numext::abs ADL overload for SuperScalar (Eigen::Array<double, W, 1>) packs.
    "psiopt/include/tycho/detail/typedefs/super_scalar_traits.h:130"
    # VectorFunction `abs(f)` expression-operator definition (builds a CwiseAbs node).
    "include/tycho/detail/vf/operators/math_overloads.h:199"
    # nanobind docstring raw string literal ("...(``abs(self)``)."), not code;
    # not caught by the `//`-comment filter above since it isn't `//`-prefixed.
    "src/bindings/vf/dense_function_base_bind.h:913"
)

# pow() — no expression-operator `pow(...)` *definition* exists in
# math_overloads.h today (checked; `CwisePow` in cwise_operators.h calls the
# qualified `x.array().pow(...)` member form, which the bare-call regex does
# not match) — so there is currently no analogous definition-site entry here.
# If one is added in the future, list it here with the same rationale shape
# as the abs entries above.
#
# The lgl_control_splines.h / lambert_solvers.h gap tracked here previously
# (bare pow() call sites with no `using std::pow;` guard) was closed by
# adding function-scope `using std::pow;` declarations to each enclosing
# function, so the allowlist is intentionally empty.
POW_ALLOWLIST=(
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Drop grep -n output lines ("path:line:content") whose content, after
# stripping leading whitespace, starts with `//`.
strip_comment_lines() {
    awk '{
        content = $0
        sub(/^[^:]*:[^:]*:/, "", content)
        gsub(/^[ \t]+/, "", content)
        if (content !~ /^\/\//) print $0
    }'
}

# Drop grep -n output lines whose file is in a newline-separated list of
# filenames (stdin: candidate lines; $1: newline-separated file list).
strip_files() {
    local files="$1" line path found f
    while IFS= read -r line; do
        path="${line%%:*}"
        found=0
        while IFS= read -r f; do
            [[ -z "${f}" ]] && continue
            if [[ "${path}" == "${f}" ]]; then
                found=1
                break
            fi
        done <<<"${files}"
        (( found == 0 )) && printf '%s\n' "${line}"
    done
}

# Drop grep -n output lines whose exact "path:line" is in the named allowlist
# array (nameref, $1 = array name).
strip_allowlist() {
    local -n _allow="$1"
    local line path rest ln pl entry matched
    while IFS= read -r line; do
        path="${line%%:*}"
        rest="${line#*:}"
        ln="${rest%%:*}"
        pl="${path}:${ln}"
        matched=0
        for entry in "${_allow[@]}"; do
            if [[ "${pl}" == "${entry}" ]]; then
                matched=1
                break
            fi
        done
        (( matched == 0 )) && printf '%s\n' "${line}"
    done
}

# ---------------------------------------------------------------------------
# Run one function's check: $1 = function name (abs|pow), $2 = allowlist
# array name. Prints any remaining violations and returns their count.
# ---------------------------------------------------------------------------
check_function() {
    local fn="$1" allowlist_name="$2"
    local raw guarded_files filtered remaining count

    raw="$(grep -rnP "(?<![\\w:.>])${fn}\\(" "${SCAN_DIRS[@]}" --include='*.h' --include='*.cpp' 2>/dev/null || true)"
    [[ -z "${raw}" ]] && return 0

    guarded_files="$(grep -rl "using std::${fn}" "${SCAN_DIRS[@]}" --include='*.h' --include='*.cpp' 2>/dev/null || true)"

    filtered="$(printf '%s\n' "${raw}" | strip_comment_lines)"
    [[ -n "${guarded_files}" ]] && filtered="$(printf '%s\n' "${filtered}" | strip_files "${guarded_files}")"
    filtered="$(printf '%s\n' "${filtered}" | strip_allowlist "${allowlist_name}")"

    remaining="$(printf '%s\n' "${filtered}" | sed '/^$/d')"
    if [[ -z "${remaining}" ]]; then
        return 0
    fi

    count="$(printf '%s\n' "${remaining}" | wc -l | tr -d ' ')"
    echo "check_abs_sweep.sh: ${count} unqualified bare ${fn}() call site(s) found:"
    echo "${remaining}"
    echo
    echo "  Fix: qualify as std::${fn}(...), or add a top-level 'using std::${fn};'"
    echo "  in the enclosing function/file (ADL-guard pattern used in kepler_utils.h,"
    echo "  lambert_solvers.h, root_finder.h, norms.h, normalized.h, etc.), or, if this"
    echo "  is a genuine new definition/false-positive, add an explicit allowlist entry"
    echo "  in scripts/check_abs_sweep.sh with a comment explaining why."
    return 1
}

fail=0
check_function abs ABS_ALLOWLIST || fail=1
check_function pow POW_ALLOWLIST || fail=1

if [[ "${fail}" -ne 0 ]]; then
    exit 1
fi

echo "check_abs_sweep.sh: clean — no unqualified bare abs()/pow() call sites found."
exit 0
