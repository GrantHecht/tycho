#!/usr/bin/env bash
# check_crtp_hooks.sh — CI lint: CRTP hook near-misses in the VectorFunction /
# optimal-control dispatch hierarchy.
#
# Background (PR7 T4): `ComputableBase`/`DenseFunctionBase` (the CRTP core of
# the VectorFunction system; see
# include/tycho/detail/vf/core/computable_base.h, "Canonical CRTP hook
# inventory") dispatch to a derived class exclusively through unqualified
# `this->derived().X(...)` calls. There is no `virtual`/`override` in this
# hierarchy, so a hook name that is even one character off from the
# canonical name (e.g. `hessian_elem_is_non_zero` instead of
# `hessian_elem_is_nonzero`, or `compute` shadowing `compute_impl`) is not a
# compile error — it silently compiles as an unused method, and the base's
# default behavior (or a pure-hook link error) applies instead. This script
# is the CI guard for the two concrete near-miss shapes documented in the
# PR7 review series; see include/tycho/detail/vf/core/computable_base.h's
# "Canonical CRTP hook inventory" doc comment for the full, authoritative
# hook list.
#
# Checks:
#   (a) Any occurrence of `_is_non_zero` anywhere in include/tycho/detail or
#       src/ — the known misspelling class of `hessian_elem_is_nonzero` /
#       `jacobian_elem_is_nonzero`. Comment-only lines are skipped (see the
#       comment-line heuristic note below); a real occurrence anywhere else
#       (a method definition, a call site, an override) fails the check.
#   (b) Any `inline void compute(` or `inline auto compute(` *definition*
#       inside include/tycho/detail/vf or include/tycho/detail/optimal_control
#       that isn't on the allowlist below. `compute` (not `compute_impl`) is
#       a legitimate override point for exactly one known class today
#       (`CwiseOperator`); anywhere else, a derived class defining `compute`
#       directly instead of `compute_impl` silently never gets called by the
#       base's dispatch machinery.
#
# KNOWN LIMITATION (accepted, not a bug in this script): this is a bash+grep
# heuristic, not a real C++ parser. It cannot verify that a flagged `compute`
# definition is actually inside a class deriving from `ComputableBase` /
# `DenseFunctionBase` / `VectorFunction` (a signature-only check, as noted for
# `conditional.h`'s unrelated bool-returning `compute` predicate, would need
# real type information) — legitimate exceptions are handled via the explicit
# allowlist instead of narrower pattern matching. Likewise `_is_non_zero`
# is a plain substring search, not scoped to a class hierarchy.
#
# Usage: scripts/check_crtp_hooks.sh
# Exit code: 0 if clean, non-zero (with offending file:line(s) printed) if any
# near-misses are found.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HOOK_INVENTORY_POINTER="include/tycho/detail/vf/core/computable_base.h (see the \"Canonical CRTP hook inventory\" doc comment on ComputableBase)"

# ---------------------------------------------------------------------------
# Allowlist for check (b): files where a direct `compute(` definition
# (instead of `compute_impl`) is intentional, not a near-miss.
# ---------------------------------------------------------------------------
COMPUTE_FILE_ALLOWLIST=(
    # The CRTP base itself: this IS the canonical `compute` hook definition
    # every other class in the hierarchy dispatches through (it calls
    # `derived().compute_impl(...)`). Not a near-miss by definition.
    "include/tycho/detail/vf/core/computable_base.h"
    # CwiseOperator: a documented, intentional second CRTP tier. Its
    # subclasses implement `cwise_compute`/`cwise_compute_jacobian`, and
    # CwiseOperator itself is the one sanctioned place that overrides
    # `compute`/`compute_jacobian`/`compute_jacobian_adjointgradient_adjointhessian`
    # directly and applies those hooks to the input. See the class doc
    # comment in this file and the "Known-legitimate direct overrider" note
    # in computable_base.h.
    "include/tycho/detail/vf/operators/cwise_operators.h"
    # conditional.h's `compute(x) -> bool` is an unrelated predicate family
    # (ConditionalNode-style branch predicates), not a ComputableBase/
    # DenseFunctionBase override — it doesn't inherit that hierarchy and its
    # `compute` returns bool, not void. Allowlisted for documentation/
    # future-proofing even though the current `inline (void|auto) compute(`
    # pattern doesn't match its `inline bool compute(` signature today.
    "include/tycho/detail/vf/type_erasure/conditional.h"
)

# ---------------------------------------------------------------------------
# Helpers (mirrors scripts/check_abs_sweep.sh)
# ---------------------------------------------------------------------------

strip_comment_lines() {
    awk '{
        content = $0
        sub(/^[^:]*:[^:]*:/, "", content)
        gsub(/^[ \t]+/, "", content)
        if (content !~ /^\/\//) print $0
    }'
}

strip_file_allowlist() {
    local -n _allow="$1"
    local line path entry matched
    while IFS= read -r line; do
        path="${line%%:*}"
        matched=0
        for entry in "${_allow[@]}"; do
            if [[ "${path}" == "${entry}" ]]; then
                matched=1
                break
            fi
        done
        (( matched == 0 )) && printf '%s\n' "${line}"
    done
}

fail=0

# --- Check (a): `_is_non_zero` misspelling class ---------------------------
raw_a="$(grep -rn "_is_non_zero" include/tycho/detail src 2>/dev/null || true)"
if [[ -n "${raw_a}" ]]; then
    filtered_a="$(printf '%s\n' "${raw_a}" | strip_comment_lines | sed '/^$/d')"
    if [[ -n "${filtered_a}" ]]; then
        count_a="$(printf '%s\n' "${filtered_a}" | wc -l | tr -d ' ')"
        echo "check_crtp_hooks.sh: ${count_a} '_is_non_zero' occurrence(s) found (misspelling of *_is_nonzero):"
        echo "${filtered_a}"
        echo
        echo "  This is the known hessian_elem_is_nonzero / jacobian_elem_is_nonzero"
        echo "  misspelling class — the base dispatches via this->derived().X(...) with"
        echo "  no virtual/override, so a misspelled hook silently compiles as an unused"
        echo "  method instead of overriding the sparsity predicate. Rename to the exact"
        echo "  canonical spelling. Canonical hook names: ${HOOK_INVENTORY_POINTER}."
        echo
        fail=1
    fi
fi

# --- Check (b): direct `compute(` definitions outside the allowlist --------
raw_b="$(grep -rnE 'inline (void|auto) compute\(' include/tycho/detail/vf include/tycho/detail/optimal_control 2>/dev/null || true)"
if [[ -n "${raw_b}" ]]; then
    filtered_b="$(printf '%s\n' "${raw_b}" | strip_file_allowlist COMPUTE_FILE_ALLOWLIST | sed '/^$/d')"
    if [[ -n "${filtered_b}" ]]; then
        count_b="$(printf '%s\n' "${filtered_b}" | wc -l | tr -d ' ')"
        echo "check_crtp_hooks.sh: ${count_b} unexpected 'compute(' definition(s) found (expected 'compute_impl'):"
        echo "${filtered_b}"
        echo
        echo "  A derived VectorFunction should implement compute_impl(x, fx), not"
        echo "  compute(x, fx) directly — compute() is the base's dispatch entry point"
        echo "  (ComputableBase::compute calls derived().compute_impl(...)). Defining"
        echo "  compute() directly on a derived class silently shadows nothing and is"
        echo "  never called by the dispatch machinery, unless this is a genuinely new,"
        echo "  intentional second-tier override — in which case add it to"
        echo "  COMPUTE_FILE_ALLOWLIST in scripts/check_crtp_hooks.sh with a comment."
        echo "  Canonical hook names: ${HOOK_INVENTORY_POINTER}."
        echo
        fail=1
    fi
fi

if [[ "${fail}" -ne 0 ]]; then
    exit 1
fi

echo "check_crtp_hooks.sh: clean — no CRTP hook near-misses found."
exit 0
