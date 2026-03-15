#!/usr/bin/env bash
###############################################################################
# bench_track.sh — Local benchmark tracking for Tycho
#
# Usage:
#   bench/bench_track.sh baseline [--reps N] [--threshold PCT]
#   bench/bench_track.sh record   [--reps N]
#   bench/bench_track.sh compare  [--threshold PCT] [commit_a] [commit_b]
#   bench/bench_track.sh list
#   bench/bench_track.sh help
###############################################################################
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_BINARY="${REPO_DIR}/build/bench/cpp/bench_all"
RESULTS_DIR="${REPO_DIR}/bench/results"
DEFAULT_REPS=5
DEFAULT_THRESHOLD=10.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

check_deps() {
    command -v jq  >/dev/null 2>&1 || die "jq not found. Install with: brew install jq"
    command -v git >/dev/null 2>&1 || die "git not found"
}

get_commit_hash() { git -C "$REPO_DIR" rev-parse HEAD; }
get_commit_short() { git -C "$REPO_DIR" rev-parse --short=8 HEAD; }
get_branch_name() { git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD; }
is_dirty() { ! git -C "$REPO_DIR" diff-index --quiet HEAD -- 2>/dev/null; }

# Convert time to nanoseconds given a time_unit string
to_ns() {
    local val="$1" unit="$2"
    case "$unit" in
        ns) echo "$val" ;;
        us) echo "$val * 1000"       | bc -l ;;
        ms) echo "$val * 1000000"    | bc -l ;;
        s)  echo "$val * 1000000000" | bc -l ;;
        *)  echo "$val" ;;
    esac
}

# Format nanoseconds into a human-friendly string
fmt_time() {
    local ns="$1"
    # Use awk for floating-point comparison
    awk -v ns="$ns" 'BEGIN {
        if (ns < 0) { ns = -ns; sign = "-" } else { sign = "" }
        if (ns < 1000)          printf "%s%.2f ns\n", sign, ns
        else if (ns < 1000000)  printf "%s%.2f us\n", sign, ns/1000
        else if (ns < 1e9)      printf "%s%.2f ms\n", sign, ns/1000000
        else                    printf "%s%.2f s\n",  sign, ns/1e9
    }'
}

# ---------------------------------------------------------------------------
# record / baseline
# ---------------------------------------------------------------------------
do_record() {
    local reps="$DEFAULT_REPS"
    local make_baseline=false
    if [[ "${1:-}" == "--baseline" ]]; then
        make_baseline=true
        shift
    fi

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --reps) reps="$2"; shift 2 ;;
            *)      die "Unknown flag: $1" ;;
        esac
    done

    [[ -x "$BENCH_BINARY" ]] || die "bench_all not found at $BENCH_BINARY
Build it with: cmake --preset macos-llvm-release -DBUILD_CPP_BENCHMARKS=ON && cd build && ninja -j6 bench_all"

    mkdir -p "$RESULTS_DIR"

    local commit_hash commit_short branch dirty_flag suffix
    commit_hash="$(get_commit_hash)"
    commit_short="$(get_commit_short)"
    branch="$(get_branch_name)"
    dirty_flag=false
    suffix=""
    if is_dirty; then
        dirty_flag=true
        suffix="-dirty"
        warn "Working tree has uncommitted changes — results tagged as dirty"
    fi

    local outfile="${RESULTS_DIR}/${commit_short}${suffix}.json"
    local tmpfile
    tmpfile="$(mktemp)"
    trap "rm -f '$tmpfile'" EXIT

    info "Running benchmarks (${reps} repetitions)..."
    "$BENCH_BINARY" \
        --benchmark_repetitions="$reps" \
        --benchmark_report_aggregates_only=true \
        --benchmark_out="$tmpfile" \
        --benchmark_out_format=json \
        2>/dev/null

    # Wrap in metadata envelope
    jq -n \
        --arg commit "$commit_hash" \
        --arg commit_short "${commit_short}${suffix}" \
        --arg branch "$branch" \
        --arg timestamp "$(date -Iseconds)" \
        --argjson dirty "$dirty_flag" \
        --slurpfile gbench "$tmpfile" \
        '{
            metadata: {
                commit: $commit,
                commit_short: $commit_short,
                branch: $branch,
                timestamp: $timestamp,
                dirty: $dirty
            },
            gbench: $gbench[0]
        }' > "$outfile"

    local count
    count="$(jq '[.gbench.benchmarks[] | select(.aggregate_name == "median")] | length' "$outfile")"
    info "Saved ${count} benchmark medians to ${outfile}"

    if $make_baseline; then
        ln -sf "$(basename "$outfile")" "${RESULTS_DIR}/baseline.json"
        info "Updated baseline symlink -> $(basename "$outfile")"
    fi
}

# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
do_compare() {
    local threshold="$DEFAULT_THRESHOLD"
    local positional=()

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --threshold) threshold="$2"; shift 2 ;;
            -*)          die "Unknown flag: $1" ;;
            *)           positional+=("$1"); shift ;;
        esac
    done

    local file_old file_new
    case "${#positional[@]}" in
        0)
            # Compare baseline vs HEAD
            file_old="${RESULTS_DIR}/baseline.json"
            [[ -e "$file_old" ]] || die "No baseline found. Run: bench/bench_track.sh baseline"
            local head_short
            head_short="$(get_commit_short)"
            file_new="$(find_result "$head_short")"
            [[ -n "$file_new" ]] || die "No results for HEAD (${head_short}). Run: bench/bench_track.sh record"
            ;;
        1)
            # Compare given commit vs HEAD
            file_old="$(find_result "${positional[0]}")"
            [[ -n "$file_old" ]] || die "No results for ${positional[0]}"
            local head_short
            head_short="$(get_commit_short)"
            file_new="$(find_result "$head_short")"
            [[ -n "$file_new" ]] || die "No results for HEAD (${head_short}). Run: bench/bench_track.sh record"
            ;;
        2)
            file_old="$(find_result "${positional[0]}")"
            [[ -n "$file_old" ]] || die "No results for ${positional[0]}"
            file_new="$(find_result "${positional[1]}")"
            [[ -n "$file_new" ]] || die "No results for ${positional[1]}"
            ;;
        *)
            die "Too many arguments. Usage: bench_track.sh compare [commit_a] [commit_b]"
            ;;
    esac

    local label_old label_new
    label_old="$(jq -r '.metadata.commit_short' "$file_old")"
    label_new="$(jq -r '.metadata.commit_short' "$file_new")"

    echo ""
    echo "Benchmark Comparison: ${label_old} -> ${label_new}"
    echo "Threshold: ${threshold}% (regressions flagged with !!)"
    echo ""

    # Do the entire join + comparison in jq, output TSV rows
    local comparison
    comparison="$(jq -r --argjson threshold "$threshold" '
        # Build lookup tables keyed by run_name
        def to_ns:
            if .time_unit == "ns" then .real_time
            elif .time_unit == "us" then .real_time * 1000
            elif .time_unit == "ms" then .real_time * 1000000
            elif .time_unit == "s"  then .real_time * 1000000000
            else .real_time end;

        def fmt_ns:
            if . < 0 then "-" + (-(.) | fmt_ns)
            elif . < 1000       then "\(. * 100 | round / 100) ns"
            elif . < 1000000    then "\(. / 1000 * 100 | round / 100) us"
            elif . < 1000000000 then "\(. / 1000000 * 100 | round / 100) ms"
            else                     "\(. / 1000000000 * 100 | round / 100) s"
            end;

        (.[0].gbench.benchmarks
            | map(select(.aggregate_name == "median"))
            | map({key: .run_name, value: {real_time, time_unit}})
            | from_entries) as $old |
        (.[1].gbench.benchmarks
            | map(select(.aggregate_name == "median"))
            | map({key: .run_name, value: {real_time, time_unit}})
            | from_entries) as $new |

        ($old | keys | sort[]) as $name |
        ($old[$name] | to_ns) as $old_ns |
        (if $new[$name] then ($new[$name] | to_ns) else null end) as $new_ns |
        select($new_ns != null) |
        (($new_ns - $old_ns) / $old_ns * 100) as $pct |
        (if $pct < 0 then -$pct else $pct end) as $abs_pct |
        (if $abs_pct > $threshold then
            if $pct > 0 then "!! REGRESSION" else "++ faster" end
         else "ok" end) as $status |
        (if $pct > 0 then "+" else "" end) as $sign |

        [$name, ($old_ns | fmt_ns), ($new_ns | fmt_ns),
         "\($sign)\($pct * 10 | round / 10)%", $status] | @tsv
    ' <(jq -s '.' "$file_old" "$file_new"))"

    # Print formatted table
    printf "%-42s %12s %12s %9s   %s\n" "Benchmark" "Old" "New" "Change" "Status"
    printf "%s\n" "$(printf '%0.s-' {1..95})"

    local total=0 regressions=0
    while IFS=$'\t' read -r name old_fmt new_fmt change status; do
        [[ -z "$name" ]] && continue
        printf "%-42s %12s %12s %9s   %s\n" "$name" "$old_fmt" "$new_fmt" "$change" "$status"
        total=$((total + 1))
        [[ "$status" == "!! REGRESSION" ]] && regressions=$((regressions + 1))
    done <<< "$comparison"

    echo ""
    if [[ $regressions -gt 0 ]]; then
        echo "Summary: ${total} benchmarks compared, ${regressions} regression(s) detected"
        exit 2
    else
        echo "Summary: ${total} benchmarks compared, no regressions"
        exit 0
    fi
}

# Find a result file by commit prefix (supports short hashes and -dirty suffix)
find_result() {
    local prefix="$1"
    local match
    # Try exact match first, then prefix glob
    for pattern in "${RESULTS_DIR}/${prefix}.json" "${RESULTS_DIR}/${prefix}-dirty.json" "${RESULTS_DIR}/${prefix}"*.json; do
        match="$(ls $pattern 2>/dev/null | head -1 || true)"
        [[ -n "$match" ]] && echo "$match" && return 0
    done
    echo ""
}

# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
do_list() {
    [[ -d "$RESULTS_DIR" ]] || die "No results directory. Run a benchmark first."

    local count=0
    printf "%-12s %-40s %-24s %s\n" "Commit" "Branch" "Date" "Baseline"
    printf "%s\n" "$(printf '%0.s-' {1..85})"

    local baseline_target=""
    if [[ -L "${RESULTS_DIR}/baseline.json" ]]; then
        baseline_target="$(readlink "${RESULTS_DIR}/baseline.json")"
    fi

    for f in "${RESULTS_DIR}"/*.json; do
        [[ -L "$f" ]] && continue  # skip symlinks
        [[ -f "$f" ]] || continue

        local short branch ts marker=""
        short="$(jq -r '.metadata.commit_short' "$f")"
        branch="$(jq -r '.metadata.branch' "$f")"
        ts="$(jq -r '.metadata.timestamp' "$f")"

        if [[ "$(basename "$f")" == "$baseline_target" ]]; then
            marker="<-- baseline"
        fi

        printf "%-12s %-40s %-24s %s\n" "$short" "$branch" "$ts" "$marker"
        count=$((count + 1))
    done

    echo ""
    echo "${count} result(s) stored"
}

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
do_help() {
    cat <<'HELP'
bench_track.sh — Local benchmark tracking for Tycho

Commands:
  baseline [--reps N]                 Record benchmarks and set as baseline
  record   [--reps N]                 Record benchmarks for current commit
  compare  [--threshold P] [a] [b]    Compare two result sets
  list                                List all stored results
  help                                Show this message

Compare modes:
  compare                             baseline vs HEAD
  compare <commit>                    <commit> vs HEAD
  compare <commit_a> <commit_b>       <commit_a> vs <commit_b>

Options:
  --reps N          Repetitions per benchmark (default: 5)
  --threshold P     Regression threshold percentage (default: 10.0)

Exit codes:
  0    Success / no regressions
  1    Usage error or missing files
  2    Regressions detected above threshold

Examples:
  bench/bench_track.sh baseline
  bench/bench_track.sh record
  bench/bench_track.sh compare
  bench/bench_track.sh compare --threshold 5.0
  bench/bench_track.sh compare abc12345 def67890
HELP
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
check_deps

cmd="${1:-help}"
shift || true

case "$cmd" in
    baseline) do_record --baseline "$@" ;;
    record)   do_record "$@" ;;
    compare)  do_compare "$@" ;;
    list)     do_list ;;
    help)     do_help ;;
    *)        die "Unknown command: $cmd. Run with 'help' for usage." ;;
esac
