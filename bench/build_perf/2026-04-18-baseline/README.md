# Build-perf baseline — 2026-04-18

Frozen snapshot of Tycho build performance data captured on branch
`ivp-solver-modernization` at commit `a3a8133` (post-SP3). See `env.txt` for the
host/toolchain configuration. See `report.md` for the narrative analysis.

## Files

| File | Contents |
|---|---|
| `tu_timing.tsv` | Per-TU serial compile time from `.ninja_log`, 142 rows (see `report.md` §"Top 10 TUs by wall time") |
| `tycho_tu_profile.json` | Peak RSS + wall time for the top 25 heaviest TUs (`scripts/measure_tu_profile.py` output) |
| `tycho_trace_analysis.txt` | Aggregate ftime-trace breakdown: phase split, top class/function template instantiations, class family roll-up (`scripts/analyze_ftime_trace.py` output) |
| `obj_category.txt` | Object file size totals per category (bindings/examples/tests/bench/core) |
| `obj_top25.txt` | 25 largest `.cpp.o` files by bytes |
| `dup_tycho.txt` | Most-duplicated `tycho::`-prefixed weak symbols across all TUs |
| `dup_symbols_all.txt` | Most-duplicated weak symbols across all TUs (includes fmt, stdlib boilerplate) |
| `dup_symbols_bindings.txt` | Most-duplicated weak symbols within `src/bindings/*.cpp.o` only |
| `report.md` | Narrative `docs/dev/notes/build_performance_report_2026-04-18.md` (copy) |
| `env.txt` | Host, compiler, build options at snapshot time |

## Regenerating this baseline or a comparison snapshot

Run the procedure in `docs/dev/notes/build_performance_analysis.md`. Copy the `/tmp/` outputs
into a sibling directory named `bench/build_perf/YYYY-MM-DD-<phase>/`. The key
metrics to diff are phase-specific (see `/home/ghecht/.claude/plans/kind-prancing-crown.md`
or the refactor plan's commit messages):

- **Example TU-split phases (A.1–A.3):** compare parallel wall-clock at a fixed
  `-j` and peak per-TU RSS (`tycho_tu_profile.json`). Do not gate on serial
  CPU-time (`sum(tu_timing.tsv)`); split refactors target parallelism, not work
  reduction.
- **Extern-template phase (C):** compare `sum(tu_timing.tsv)` serial CPU-time
  and the count of duplicate `Integrator<Kepler>::init_stepper_and_controller`
  symbols in `dup_symbols_bindings.txt`.
