# Build Performance Analysis Procedure

This document describes how to run a complete build performance analysis of the
Tycho project. Follow it end-to-end to produce a quantitative report identifying
the heaviest translation units, the most expensive template instantiations, symbol
duplication across TUs, and peak memory consumption.

The analysis produces data for three questions:

1. **Where is build time going?** Per-TU timing, category breakdown, and
   compilation phase split (template instantiation vs parsing vs optimization).
2. **What specific templates are expensive?** Ranked by aggregate compilation cost,
   instantiation count, and per-instantiation overhead.
3. **Where is memory going?** Peak RSS per TU, and cross-TU symbol duplication
   that inflates both compile memory and link time.

---

## Prerequisites

### Required tools

All tools are standard Linux/Clang utilities. Nothing extra needs to be installed.

| Tool | Purpose | Typical location |
|------|---------|------------------|
| `clang++` | Compiler (must support `-ftime-trace`) | System or LLVM install |
| `ninja` | Build system | System package |
| `/usr/bin/time` | Peak RSS measurement (GNU time, **not** the shell builtin) | coreutils |
| `nm` | Symbol listing from object files | binutils |
| `c++filt` | Demangle C++ symbols | binutils |
| `awk` | Text processing for ninja log parsing | System |
| Python 3 | Analysis scripts (json, pathlib — stdlib only) | conda `tycho` env |

### Build state

You must have a **complete, successful build** before starting. The analysis reads
`.ninja_log`, `compile_commands.json`, and existing `.o` files from the build
directory. If you are analyzing a fresh configuration:

```bash
conda activate tycho
cmake --preset <your-preset>       # e.g. linux-clang-release
cd build && ninja -j4 all          # full build — tests, benchmarks, examples
```

The `all` target must include tests, benchmarks, and examples for the analysis to
cover the full build. If you built only a subset, the analysis will only reflect
that subset.

---

## Step 1: Parse the Ninja Build Log (Per-TU Timing)

The `.ninja_log` file in the build directory records wall-clock start/end
timestamps for every compilation step. Parse it to produce a ranked list of TUs
by build time.

### 1a. Get the authoritative list of current targets

Stale entries from previous builds (e.g., pre-rename paths) accumulate in
`.ninja_log`. Filter to only current targets:

```bash
cd build
ninja -t targets all | grep '\.cpp\.o:' | sed 's/:.*//' > /tmp/current_targets.txt
wc -l /tmp/current_targets.txt    # should match your expected TU count
```

### 1b. Extract per-TU build times (deduplicated)

```bash
cd build
awk -F'\t' 'NR>1 && NF>=4 && $4 ~ /\.cpp\.o$/ {
  key=$4; time=($2-$1)/1000
  if(time > best[key]) best[key]=time
}
END {
  for(k in best) printf "%.1f\t%s\n", best[k], k
}' .ninja_log | grep -Ff /tmp/current_targets.txt | sort -rn > /tmp/tu_timing.tsv
```

Inspect the top 30:

```bash
head -30 /tmp/tu_timing.tsv
```

### 1c. Category breakdown

```bash
cat /tmp/tu_timing.tsv | awk -F'\t' '{
  t=$1; k=$2
  if (k ~ /^src\/bindings\//) cat="bindings"
  else if (k ~ /^src\//) cat="core_lib"
  else if (k ~ /^tests\//) cat="tests"
  else if (k ~ /^bench\//) cat="benchmarks"
  else if (k ~ /^examples\//) cat="examples"
  else cat="other"
  sum[cat] += t; cnt[cat]++
}
END {
  total_t=0; total_c=0
  for(c in sum) { total_t+=sum[c]; total_c+=cnt[c] }
  for(c in sum) printf "%7.0f s  %5.1f%%  (%2d TUs, avg %5.1fs)  %s\n", \
    sum[c], sum[c]/total_t*100, cnt[c], sum[c]/cnt[c], c
  printf "\n%7.0f s  100.0%%  (%d TUs total)\n", total_t, total_c
}' | sort -rn
```

**What to look for:** Which category dominates? What is the per-TU average?
Compare these numbers across runs to measure improvement.

---

## Step 2: Measure Peak RSS and Collect `-ftime-trace` Data

This step recompiles the heaviest TUs one at a time, measuring peak RSS via
`/usr/bin/time -v` and generating Clang `-ftime-trace` JSON files for template
instantiation profiling.

### 2a. Run the profiler script

```bash
mkdir -p /tmp/tycho_traces
conda run -n tycho python scripts/measure_tu_profile.py \
    --top 25 \
    --trace-dir /tmp/tycho_traces \
    --output /tmp/tycho_tu_profile.json
```

**Arguments:**

| Flag | Purpose |
|------|---------|
| `--top N` | Number of heaviest TUs to profile (default 20) |
| `--trace-dir DIR` | Directory to collect `-ftime-trace` JSON files |
| `--output FILE` | Save structured results as JSON |
| `--stats` | Also collect `-Xclang -print-stats` (verbose compiler internals) |

**How it works:** For each TU, the script:
1. Looks up the compile command from `compile_commands.json` (falls back to `ninja -t commands`)
2. Deletes the `.o` file to force recompilation
3. Prepends `/usr/bin/time -v` and appends `-ftime-trace` to the compile command
4. Runs the compilation, captures peak RSS and wall time from `/usr/bin/time` stderr
5. Moves the generated `.json` trace file to the trace directory

**Runtime:** Each TU takes 60-180 seconds. Profiling the top 25 takes roughly
40-70 minutes total. TUs are compiled sequentially — do not run anything else
that compiles in parallel.

**Known limitation:** TUs with the same source filename (e.g., multiple
`main.cpp` files from different example directories) will produce trace files
with the same name, and later ones overwrite earlier ones. The RSS and wall-time
measurements in the JSON output are still correct for all TUs; only some trace
files are lost. If you need all traces, rename the `.json` files between runs
or modify the script to use the full target path as the trace filename.

### 2b. Inspect RSS results

The script prints a summary table sorted by peak RSS. You can also read the JSON:

```bash
python3 -c "
import json
with open('/tmp/tycho_tu_profile.json') as f:
    data = json.load(f)
print(f\"{'TU':<45} {'Cat':<8} {'RSS (GB)':>8} {'Wall (s)':>8}\")
print('-' * 75)
for r in sorted(data, key=lambda x: -(x.get('peak_rss_kb') or 0)):
    name = r['short_name'][:44]
    cat = r.get('category', '?')
    rss = f\"{r['peak_rss_gb']:.2f}\" if r.get('peak_rss_gb') else 'N/A'
    wall = f\"{r['wall_time_s']:.1f}\" if r.get('wall_time_s') else 'N/A'
    print(f'{name:<45} {cat:<8} {rss:>8} {wall:>8}')
"
```

**What to look for:** The highest-RSS TUs determine the maximum safe `-j` value.
On a system with `M` GB of RAM, the practical maximum parallelism is roughly
`floor(M / max_rss_gb)`. For example, if the heaviest TU uses 8.7 GB, a 32 GB
machine can safely run `-j3`.

---

## Step 3: Analyze Template Instantiation Profiles

### 3a. Run the trace analyzer

```bash
conda run -n tycho python scripts/analyze_ftime_trace.py /tmp/tycho_traces/
```

This reads all `.json` trace files in the directory and produces a report with
these sections:

1. **Compilation phase breakdown** — What fraction of time is template
   instantiation (function + class), frontend parsing, backend optimization,
   and code generation.
2. **Per-TU breakdown** — Phase split for each individual TU.
3. **Top 30 class template instantiations** — Ranked by aggregate time.
4. **Top 30 function template instantiations** — Ranked by aggregate time.
5. **Top 20 headers by parse time** — Which headers cost the most to parse
   (only visible for TUs that don't use a PCH, since the PCH absorbs parsing).
6. **Top 20 code generation costs** — Which functions produce the most LLVM IR.
7. **Derivative mode instantiations** — How many `DenseFirstDerivatives` and
   `DenseSecondDerivatives` variants are instantiated.
8. **Template instantiation time by tycho class family** — Aggregates all
   instantiations by their outermost `tycho::` namespace class (e.g., all
   `DenseFunctionBase` methods summed together).

### 3b. Save the output

```bash
conda run -n tycho python scripts/analyze_ftime_trace.py /tmp/tycho_traces/ \
    > /tmp/tycho_trace_analysis.txt 2>&1
```

### 3c. Interpreting the results

**Phase breakdown:** The `InstantiateFunction` + `InstantiateClass` percentages
tell you how much of the build is template instantiation. In the baseline
analysis (2026-04-06), this was 91.9%. If a code change reduces this
significantly, the change is having the intended effect.

**Per-TU breakdown:** Compare `Frontend` vs `Backend` time for each TU. TUs
with high frontend and low backend are dominated by template instantiation.
TUs with significant backend time have substantial LLVM optimization cost
(usually because they generate large amounts of code).

**Function instantiations:** The top entries here are your highest-leverage
improvement targets. The key columns are:
- **Time** — total seconds spent instantiating this function across all TUs
- **Count** — how many unique instantiations exist
- **TUs** — how many TUs instantiate it (higher = more cross-TU duplication)

A function that appears in many TUs with high aggregate time is a strong
candidate for `extern template` or refactoring to reduce instantiation count.

**Class family aggregation:** This is the most actionable section. It answers
"which tycho subsystem is costing the most?" by rolling up all methods and
nested types under their parent class.

---

## Step 4: Object File and Symbol Analysis

### 4a. Object file sizes by category

```bash
cd build
cat /tmp/current_targets.txt | while read f; do
  [ -f "$f" ] && stat -c "%s $f" "$f"
done | awk '{
  size=$1; f=$2
  if (f ~ /^src\/bindings\//) cat="bindings"
  else if (f ~ /^src\//) cat="core"
  else if (f ~ /^tests\//) cat="tests"
  else if (f ~ /^bench\//) cat="bench"
  else if (f ~ /^examples\//) cat="examples"
  else cat="other"
  sum[cat]+=size; cnt[cat]++
}
END {
  total=0
  for(c in sum) total+=sum[c]
  for(c in sum) printf "%6.1f MB  %5.1f%%  %3d TUs  %s\n", \
    sum[c]/1048576, sum[c]/total*100, cnt[c], c
  printf "\n%6.1f MB total\n", total/1048576
}' | sort -rn
```

### 4b. Largest object files

```bash
cd build
cat /tmp/current_targets.txt | while read f; do
  [ -f "$f" ] && stat -c "%s $f" "$f"
done | sort -rn | head -20 | awk '{printf "%.1f MB\t%s\n", $1/1048576, $2}'
```

### 4c. Symbol analysis for a specific TU

Replace `<path_to_o>` with the path to an object file from the list above:

```bash
# Total text symbol count and size
nm --size-sort --radix=d -C <path_to_o> | grep ' [tTW] ' | \
  awk '{total+=$1; count++} END {printf "Symbols: %d, Code size: %.1f MB\n", count, total/1048576}'

# Top 15 largest symbols (truncated to 120 chars)
nm --size-sort --radix=d -C <path_to_o> | grep ' [tTW] ' | tail -15 | \
  awk '{printf "%6.1f KB  %s\n", $1/1024, substr($0, index($0,$3), 120)}'

# Symbol count by tycho class family
nm --size-sort --radix=d -C <path_to_o> | grep ' [tTW] ' | \
  grep -oP 'tycho::\w+::\w+' | sort | uniq -c | sort -rn | head -20
```

### 4d. Count GFModel type instantiations per TU

```bash
nm --defined-only -C <path_to_o> | grep -oP 'GFModel<[^>]+>' | sort -u | wc -l
```

To see the actual types:

```bash
nm --defined-only -C <path_to_o> | grep -oP 'GFModel<[^>]+>' | sort -u
```

---

## Step 5: Cross-TU Duplicate Symbol Analysis

### 5a. Total weak symbol count

```bash
cd build
total=0
while read f; do
  [ -f "$f" ] && count=$(nm --defined-only "$f" 2>/dev/null | grep -c ' W ') && total=$((total + count))
done < /tmp/current_targets.txt
echo "Total weak symbols (before linker dedup): $total"
```

Each weak symbol represents a template instantiation that the compiler fully
processed but may be identical to the same instantiation in other TUs. The linker
keeps only one copy, but every TU pays the full compilation cost.

### 5b. Most-duplicated symbols across binding TUs

```bash
cd build
find . -path "*/bindings/*" -name "*.cpp.o" | grep -Ff /tmp/current_targets.txt | \
  while read f; do
    nm --defined-only -C "$f" 2>/dev/null | grep ' W ' | awk '{print $3}'
  done | sort | uniq -c | sort -rn | awk '$1 > 3' | head -20
```

### 5c. Most-duplicated symbols across ALL TUs

```bash
cd build
cat /tmp/current_targets.txt | while read f; do
  [ -f "$f" ] && nm --defined-only -C "$f" 2>/dev/null | grep ' W ' | awk '{print $3}'
done | sort | uniq -c | sort -rn | awk '$1 > 3' | head -20
```

**What to look for:** Symbols with hundreds or thousands of occurrences are
strong candidates for `extern template` declarations. The number before each
symbol is the number of TUs that independently instantiate it.

---

## Step 6: Include Dependency Analysis (Optional)

This step is less critical since template instantiation dominates (not parsing),
but it is useful for verifying that the PCH is effective.

### 6a. Generate an include tree for a specific TU

Extract the compile command and run with `-H -fsyntax-only`:

```bash
cd build
# Get the compile command for a specific TU
CMD=$(ninja -t commands <target_path.o>)

# Add -H and -fsyntax-only, run it
$CMD -H -fsyntax-only 2>&1 | head -300
```

Each line is indented with `.` characters to show include depth. Count unique
headers and group by source:

```bash
$CMD -H -fsyntax-only 2>&1 | grep '^\.' | sed 's/^\.* //' | sort -u | awk -F'/' '{
  if ($0 ~ /tycho\/include/) print "tycho_public"
  else if ($0 ~ /tycho\/dep\/eigen/) print "eigen"
  else if ($0 ~ /tycho\/dep\/fmt/) print "fmt"
  else if ($0 ~ /tycho\/dep\/nanobind/) print "nanobind"
  else if ($0 ~ /intel\/oneapi/) print "mkl"
  else if ($0 ~ /benchmark/) print "benchmark"
  else if ($0 ~ /include\/c\+\+/) print "libstdc++"
  else if ($0 ~ /clang/) print "clang_builtins"
  else print "system"
}' | sort | uniq -c | sort -rn
```

**Note:** If the TU uses a PCH, most headers will be absorbed by the PCH and
won't appear in the `-H` output. This is expected and confirms the PCH is
working.

---

## Comparing Before and After

When you run this analysis after making build-performance improvements, compare:

### Key metrics to track

| Metric | How to get it | What improvement looks like |
|--------|---------------|---------------------------|
| Total serial build time | Step 1b — sum column 1 of `/tmp/tu_timing.tsv` | Lower total |
| Heaviest TU time | Step 1b — first line of `/tmp/tu_timing.tsv` | Lower peak |
| Peak RSS (heaviest TU) | Step 2a — first row of output | Lower peak |
| Template instantiation % | Step 3a — `InstantiateFunction` + `InstantiateClass` | Lower % |
| Top function template time | Step 3a — first entry in function instantiation table | Lower time |
| Top class family time | Step 3a — first entry in class family table | Lower time |
| Total weak symbols | Step 5a | Fewer symbols |
| Top duplicate count | Step 5b — first entry | Lower count |
| Total .o file size | Step 4a — bottom line | Smaller |

### Quick comparison script

After running the full analysis on both the baseline and the updated code, you
can diff the key numbers:

```bash
# Compare timing summaries
echo "=== Before ==="
awk '{sum+=$1} END {printf "Total: %.0fs (%.1f min)\n", sum, sum/60}' /tmp/tu_timing_before.tsv
head -5 /tmp/tu_timing_before.tsv

echo ""
echo "=== After ==="
awk '{sum+=$1} END {printf "Total: %.0fs (%.1f min)\n", sum, sum/60}' /tmp/tu_timing_after.tsv
head -5 /tmp/tu_timing_after.tsv
```

### Saving a baseline

Before making changes, save the analysis artifacts so you can compare later:

```bash
mkdir -p /tmp/tycho_baseline
cp /tmp/tu_timing.tsv /tmp/tycho_baseline/
cp /tmp/tycho_tu_profile.json /tmp/tycho_baseline/
cp -r /tmp/tycho_traces /tmp/tycho_baseline/traces
conda run -n tycho python scripts/analyze_ftime_trace.py /tmp/tycho_traces/ \
    > /tmp/tycho_baseline/trace_analysis.txt 2>&1
```

---

## Full End-to-End Example

From a clean, complete build:

```bash
# 0. Ensure full build is complete
conda activate tycho
cd build && ninja -j4 all

# 1. Parse build log
ninja -t targets all | grep '\.cpp\.o:' | sed 's/:.*//' > /tmp/current_targets.txt
awk -F'\t' 'NR>1 && NF>=4 && $4 ~ /\.cpp\.o$/ {
  key=$4; time=($2-$1)/1000
  if(time > best[key]) best[key]=time
} END {
  for(k in best) printf "%.1f\t%s\n", best[k], k
}' .ninja_log | grep -Ff /tmp/current_targets.txt | sort -rn > /tmp/tu_timing.tsv
echo "Top 10 TUs by build time:"
head -10 /tmp/tu_timing.tsv

# 2. Profile top 25 TUs (RSS + ftime-trace) — takes 40-70 min
cd ..
mkdir -p /tmp/tycho_traces
conda run -n tycho python scripts/measure_tu_profile.py \
    --top 25 \
    --trace-dir /tmp/tycho_traces \
    --output /tmp/tycho_tu_profile.json

# 3. Analyze template instantiation profiles
conda run -n tycho python scripts/analyze_ftime_trace.py /tmp/tycho_traces/ \
    | tee /tmp/tycho_trace_analysis.txt

# 4. Object file sizes
cd build
cat /tmp/current_targets.txt | while read f; do
  [ -f "$f" ] && stat -c "%s $f" "$f"
done | sort -rn | head -20 | awk '{printf "%.1f MB\t%s\n", $1/1048576, $2}'

# 5. Cross-TU duplicate symbols
total=0
while read f; do
  [ -f "$f" ] && count=$(nm --defined-only "$f" 2>/dev/null | grep -c ' W ') \
    && total=$((total + count))
done < /tmp/current_targets.txt
echo "Total weak symbols: $total"

# 6. Save as baseline for future comparison
mkdir -p /tmp/tycho_baseline
cp /tmp/tu_timing.tsv /tmp/tycho_baseline/
cp /tmp/tycho_tu_profile.json /tmp/tycho_baseline/
cp /tmp/tycho_trace_analysis.txt /tmp/tycho_baseline/
echo "Baseline saved to /tmp/tycho_baseline/"
```

---

## Reference: Scripts

### `scripts/measure_tu_profile.py`

Measures peak RSS and wall time for the heaviest TUs. Optionally generates
`-ftime-trace` JSON files and collects `-Xclang -print-stats` output.

Reads `.ninja_log` and `compile_commands.json` from the `build/` directory.
Recompiles each TU individually (sequentially) with `/usr/bin/time -v`.

### `scripts/analyze_ftime_trace.py`

Parses `-ftime-trace` Chrome trace JSON files and produces an aggregate report.
Accepts a directory path; reads all `*.json` files in it.

Reports compilation phase breakdown, per-TU breakdown, top template
instantiations (class and function), header parse costs, code generation costs,
derivative mode analysis, and tycho class family aggregation.
