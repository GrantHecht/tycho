#!/usr/bin/env bash
###############################################################################
# bench_track.sh — Local benchmark tracking for Tycho
#
# Usage:
#   bench/bench_track.sh baseline [--reps N] [--force]
#   bench/bench_track.sh record   [--reps N] [--force]
#   bench/bench_track.sh compare  [--threshold PCT] [--force] [commit_a] [commit_b]
#   bench/bench_track.sh compare  --alternate OTHER_BENCH_ALL [--reps N] [--threshold PCT]
#   bench/bench_track.sh list
#   bench/bench_track.sh help
#
# THE IDLE-BOX GATE. A measurement taken while something else is using the
# machine is not a measurement of the code, and this suite has benchmarks whose
# timing is bimodal enough that a busy box can invent a double-digit
# "regression" out of nothing. Every command that MEASURES therefore refuses to
# start above a one-minute load average of IDLE_GATE, and every recording
# carries the load it was taken at, so a later comparison can refuse a pair of
# files that were not both taken on a quiet box. --force overrides either
# refusal, and marks the result as forced so it cannot be mistaken later for a
# clean one.
#
# ALTERNATED REPS. Recording arm A to completion and then arm B leaves each arm
# sitting in whatever thermal and allocator state its own run produced, and a
# bimodal benchmark can settle into a different mode per arm. `compare
# --alternate` instead interleaves them -- one repetition of the old binary,
# one of the new, N times round -- so both arms see the same conditions and a
# mode flip lands in both rather than in the difference between them.
###############################################################################
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_BINARY="${REPO_DIR}/build/bench/cpp/bench_all"
RESULTS_DIR="${REPO_DIR}/bench/results"
DEFAULT_REPS=5
DEFAULT_THRESHOLD=10.0

# One-minute load average at or above which a measuring command refuses to run.
# Chosen well below one busy core: this box's own editors and language servers
# idle under it, and anything doing real work sits above it.
IDLE_GATE="${TYCHO_BENCH_IDLE_GATE:-0.6}"
# Validated here rather than where it is used: it goes into result metadata via
# jq --argjson, which fails hard on anything jq will not read as a number (".6",
# a trailing space, an empty string), and that failure would land in the middle
# of a recording rather than before one.
case "$IDLE_GATE" in
    [0-9]|[0-9].[0-9]*|[0-9][0-9]*|[0-9][0-9]*.[0-9]*) ;;
    *) echo "ERROR: TYCHO_BENCH_IDLE_GATE must be a plain non-negative decimal number (got '${IDLE_GATE}')" >&2
       exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# An explicit template, because a bare `mktemp` (and `mktemp -d`) is a usage
# error on the BSD mktemp macOS ships -- it wants a template or -t.
make_temp_file() { mktemp "${TMPDIR:-/tmp}/tycho_bench.XXXXXX"; }
make_temp_dir()  { mktemp -d "${TMPDIR:-/tmp}/tycho_bench.XXXXXX"; }

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

check_deps() {
    command -v jq  >/dev/null 2>&1 || die "jq not found. Install with: brew install jq"
    command -v git >/dev/null 2>&1 || die "git not found"
}

# One-minute load average, as a decimal string. Linux reads it from /proc;
# everything else parses uptime, whose "load average(s):" tail is the one field
# both BSD and macOS spell the same way.
read_loadavg() {
    if [[ -r /proc/loadavg ]]; then
        cut -d' ' -f1 /proc/loadavg
    else
        # The first field only, with a trailing list comma stripped and a
        # comma DECIMAL separator converted rather than deleted -- a blanket
        # comma-to-space would turn a comma-decimal locale's "0,49" into "0"
        # and let every load pass the gate.
        uptime | sed -e 's/.*[Ll]oad [Aa]verages\{0,1\}: *//' |
            awk '{v=$1; sub(/,$/, "", v); gsub(/,/, ".", v); print v}'
    fi
}

# Refuses to measure on a box that is not idle. $1 names the command for the
# message; $2 is "true" when --force was given.
require_idle_box() {
    local what="$1" forced="$2" load
    load="$(read_loadavg)"
    if awk -v l="$load" -v g="$IDLE_GATE" 'BEGIN { exit !(l >= g) }'; then
        if [[ "$forced" == true ]]; then
            warn "${what}: load average ${load} is at or above the ${IDLE_GATE} gate — forced"
        else
            die "${what}: load average is ${load}, at or above the ${IDLE_GATE} idle gate.
Wait for the box to go quiet, or pass --force to measure anyway (the result is
then marked forced and a later compare will refuse it without --force)."
        fi
    fi
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
    local forced=false
    if [[ "${1:-}" == "--baseline" ]]; then
        make_baseline=true
        shift
    fi

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --reps)  reps="$2"; shift 2 ;;
            --force) forced=true; shift ;;
            *)       die "Unknown flag: $1" ;;
        esac
    done

    [[ -x "$BENCH_BINARY" ]] || die "bench_all not found at $BENCH_BINARY
Build it with: cmake --preset macos-llvm-release -DBUILD_CPP_BENCHMARKS=ON && cd build && ninja -j6 bench_all"

    require_idle_box "record" "$forced"
    local load_before
    load_before="$(read_loadavg)"

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
    tmpfile="$(make_temp_file)"
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
        --argjson reps "$reps" \
        --argjson load_before "$load_before" \
        --argjson load_after "$(read_loadavg)" \
        --argjson idle_gate "$IDLE_GATE" \
        --argjson forced "$forced" \
        --slurpfile gbench "$tmpfile" \
        '{
            metadata: {
                commit: $commit,
                commit_short: $commit_short,
                branch: $branch,
                timestamp: $timestamp,
                dirty: $dirty,
                reps: $reps,
                protocol: "sequential",
                load_before: $load_before,
                load_after: $load_after,
                idle_gate: $idle_gate,
                forced: $forced
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
# Refuses a stored result that was not taken on a quiet box. A file predating
# the recorded-load metadata cannot be judged either way, so it warns instead.
check_recorded_conditions() {
    local file="$1" forced_ok="$2" label load recorded_forced
    label="$(jq -r '.metadata.commit_short' "$file")"
    # has() rather than //: jq's alternative operator treats `false` as absent,
    # so `.metadata.forced // "unknown"` would report every CLEAN recording as
    # forced=unknown -- in exactly the message a reader needs to understand.
    load="$(jq -r 'if (.metadata|has("load_before")) then .metadata.load_before else "unknown" end' "$file")"
    recorded_forced="$(jq -r 'if (.metadata|has("forced")) then .metadata.forced else "unknown" end' "$file")"

    if [[ "$load" == "unknown" ]]; then
        warn "${label}: recorded before the idle gate existed — its box conditions are unknown"
        return 0
    fi
    if [[ "$recorded_forced" == "true" ]] || awk -v l="$load" -v g="$IDLE_GATE" \
            'BEGIN { exit !(l >= g) }'; then
        if [[ "$forced_ok" == true ]]; then
            warn "${label}: recorded at load ${load} (forced=${recorded_forced}) — comparing anyway"
        else
            die "${label} was recorded at load average ${load} (forced=${recorded_forced}), at or
above the ${IDLE_GATE} idle gate. A comparison is only as good as the recordings
behind it: re-record it on a quiet box, or pass --force to compare anyway."
        fi
    fi
}

# One alternated measurement round-trip: reps rounds, each running the old
# binary once and then the new one once. Writes an envelope per arm, in the
# shape do_compare's reporting half reads.
# One arm's single repetition. Stdout is the human table, which the JSON output
# makes redundant, so it goes away; STDERR IS KEPT and surfaced on a nonzero
# exit -- discarding it turns a wrong flag, a missing shared library or a filter
# matching nothing into set -e killing the script with no message at all.
run_alternated_arm() {
    local bin="$1" out="$2" errfile="$3"
    shift 3
    # "$@" is used rather than an array expanded at the call site: bash 3.2
    # through 4.3 -- which is what macOS ships -- abort under set -u on
    # "${arr[@]}" when arr is EMPTY, and the filter array usually is.
    if ! "$bin" --benchmark_repetitions=1 "$@" \
            --benchmark_out="$out" --benchmark_out_format=json \
            >/dev/null 2>"$errfile"; then
        warn "benchmark run failed: $bin"
        cat "$errfile" >&2
        return 1
    fi
}

run_alternated() {
    local old_bin="$1" new_bin="$2" reps="$3" out_old="$4" out_new="$5" workdir="$6"
    local filter="${7:-}"
    local filter_arg=()
    [[ -n "$filter" ]] && filter_arg=(--benchmark_filter="$filter")

    info "Alternating ${reps} repetitions: $(basename "$(dirname "$old_bin")")/$(basename "$old_bin") vs current build"
    local i
    for ((i = 1; i <= reps; i++)); do
        info "  round ${i}/${reps}"
        # Same empty-array guard at the call site, for the same bash versions.
        run_alternated_arm "$old_bin" "${workdir}/old-${i}.json" "${workdir}/stderr.txt" \
            ${filter_arg[@]+"${filter_arg[@]}"}
        run_alternated_arm "$new_bin" "${workdir}/new-${i}.json" "${workdir}/stderr.txt" \
            ${filter_arg[@]+"${filter_arg[@]}"}
    done

    # Per benchmark, the median over the rounds. Emitted with the aggregate
    # name the reporting half selects on, so an alternated run and a recorded
    # one are read by exactly the same code.
    local median_prog='
        def to_ns:
            if .time_unit == "ns" then .real_time
            elif .time_unit == "us" then .real_time * 1000
            elif .time_unit == "ms" then .real_time * 1000000
            elif .time_unit == "s"  then .real_time * 1000000000
            else .real_time end;
        def median: sort | (length as $n |
            if $n % 2 == 1 then .[($n - 1) / 2]
            else (.[$n / 2 - 1] + .[$n / 2]) / 2 end);
        [ .[].benchmarks[] | select(.run_type == "iteration" or (.run_type | not)) ]
        | group_by(.run_name)
        | map({ run_name: .[0].run_name,
                name: .[0].run_name,
                aggregate_name: "median",
                time_unit: "ns",
                real_time: (map(to_ns) | median) })'

    local arm
    for arm in old new; do
        local out
        [[ "$arm" == old ]] && out="$out_old" || out="$out_new"
        jq -n \
            --arg label "$arm" \
            --arg timestamp "$(date -Iseconds)" \
            --argjson reps "$reps" \
            --argjson load_before "$(read_loadavg)" \
            --argjson idle_gate "$IDLE_GATE" \
            --slurpfile marks <(jq -s "$median_prog" "${workdir}/${arm}"-*.json) \
            '{
                metadata: {
                    commit: "alternated",
                    commit_short: $label,
                    branch: "alternated",
                    timestamp: $timestamp,
                    dirty: false,
                    reps: $reps,
                    protocol: "alternated",
                    load_before: $load_before,
                    idle_gate: $idle_gate,
                    forced: false
                },
                gbench: { benchmarks: $marks[0] }
            }' > "$out"
    done
}

do_compare() {
    local threshold="$DEFAULT_THRESHOLD"
    local reps="$DEFAULT_REPS"
    local forced=false
    local alternate=""
    local filter=""
    local positional=()

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --threshold) threshold="$2"; shift 2 ;;
            --reps)      reps="$2"; shift 2 ;;
            --alternate) alternate="$2"; shift 2 ;;
            --filter)    filter="$2"; shift 2 ;;
            --force)     forced=true; shift ;;
            -*)          die "Unknown flag: $1" ;;
            *)           positional+=("$1"); shift ;;
        esac
    done

    if [[ -n "$alternate" ]]; then
        [[ "${#positional[@]}" -eq 0 ]] ||
            die "compare --alternate takes no commit arguments: it measures both arms itself"
        [[ -x "$alternate" ]] || die "not an executable benchmark binary: $alternate"
        [[ -x "$BENCH_BINARY" ]] || die "bench_all not found at $BENCH_BINARY"
        require_idle_box "compare --alternate" "$forced"

        local alt_old alt_new alt_dir
        alt_old="$(make_temp_file)"
        alt_new="$(make_temp_file)"
        alt_dir="$(make_temp_dir)"
        # One trap for all three: run_alternated can exit through set -e on a
        # failed benchmark run, so nothing may rely on reaching the end of it.
        trap "rm -f '$alt_old' '$alt_new'; rm -rf '$alt_dir'" EXIT
        run_alternated "$alternate" "$BENCH_BINARY" "$reps" "$alt_old" "$alt_new" "$alt_dir" \
            "$filter"
        report_comparison "$alt_old" "$alt_new" "$threshold"
        return
    fi

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

    check_recorded_conditions "$file_old" "$forced"
    check_recorded_conditions "$file_new" "$forced"
    report_comparison "$file_old" "$file_new" "$threshold"
}

# The reporting half, shared by the stored-file and alternated paths.
report_comparison() {
    local file_old="$1" file_new="$2" threshold="$3"

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
  baseline [--reps N] [--force]       Record benchmarks and set as baseline
  record   [--reps N] [--force]       Record benchmarks for current commit
  compare  [--threshold P] [a] [b]    Compare two stored result sets
  compare  --alternate BIN [--reps N] Measure both arms alternately and compare
  list                                List all stored results
  help                                Show this message

Compare modes:
  compare                             baseline vs HEAD
  compare <commit>                    <commit> vs HEAD
  compare <commit_a> <commit_b>       <commit_a> vs <commit_b>
  compare --alternate <bench_all>     that binary vs the current build, run
                                      one repetition each, alternately, N times

Options:
  --reps N          Repetitions per benchmark (default: 5). Under --alternate,
                    the number of alternating rounds.
  --threshold P     Regression threshold percentage (default: 10.0)
  --filter REGEX    Under --alternate, restrict to matching benchmarks
  --force           Measure (or compare) despite the idle-box gate

  TYCHO_BENCH_IDLE_GATE in the environment moves the gate (default 0.6), for a
  box whose quiet load sits somewhere else.

The idle-box gate:
  A command that measures refuses to start when the one-minute load average is
  at or above 0.6, and every recording carries the load it was taken at, so a
  comparison refuses a pair that was not both taken quiet. --force overrides
  either refusal and marks the recording as forced.

Alternated reps:
  Recording one arm to completion and then the other lets a bimodal benchmark
  settle into a different mode per arm, which shows up as a difference between
  the arms rather than as the noise it is. --alternate interleaves them
  instead, so both arms see the same conditions.

Exit codes:
  0    Success / no regressions
  1    Usage error, missing files, or a refused idle-box gate
  2    Regressions detected above threshold

Examples:
  bench/bench_track.sh baseline
  bench/bench_track.sh record
  bench/bench_track.sh compare
  bench/bench_track.sh compare --threshold 5.0
  bench/bench_track.sh compare abc12345 def67890
  bench/bench_track.sh compare --alternate /path/to/old/build/bench/cpp/bench_all --reps 7
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
