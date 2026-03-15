---
name: pre-merge
description: "Run the full pre-merge validation procedure: build, tests, benchmarks, format check, and post results as a PR comment."
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
  - Agent
---

# Pre-Merge Validation

Run the complete pre-merge checklist as defined in the Development Workflow
section of CLAUDE.md. After all checks complete, post a summary comment on
the current PR.

The PR number can be passed as $ARGUMENTS, or auto-detected from the current
branch.

## Procedure

### 1. Detect PR

If $ARGUMENTS contains a PR number, use that. Otherwise, detect from the
current branch:

```bash
gh pr view --json number -q .number
```

If no PR exists for the current branch, warn the user and continue with
checks (skip the comment at the end).

### 2. Build

Build everything (library, tests, benchmarks) with `-j2`:

```bash
cd build && ninja -j2 all bench_all
```

If the build fails, stop and report the error.

### 3. C++ unit tests

```bash
cd build && ctest --output-on-failure
```

Record: total tests, passed, failed.

### 4. Python integration tests

```bash
conda activate tycho && python run_examples.py
```

Record: total examples, passed, failed.

### 5. C++ brachistochrone example

```bash
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Verify output contains "Optimal Solution Found".

### 6. Benchmarks

Record benchmarks for the current commit and compare against baseline:

```bash
bench/bench_track.sh record
bench/bench_track.sh compare
```

Capture the full comparison table output. Note: exit code 2 from compare
means regressions were detected — this is not a build failure, but must
be reported.

### 7. Format check

```bash
# C++ — check only, do not modify
find src extensions bench tests \( -name "*.cpp" -o -name "*.h" \) -print0 \
  | xargs -0 /opt/homebrew/opt/llvm/bin/clang-format -style=file --dry-run -Werror 2>&1

# Python
ruff format --check .
ruff check --select I .
```

### 8. Post PR comment

If a PR was detected in step 1, post a comment summarizing all results
using `gh pr comment`. Use this format:

```
## Pre-Merge Validation

| Check | Result |
|-------|--------|
| Build | PASS |
| C++ unit tests | PASS (N/N passed) |
| Python examples | PASS (38/38 passed) |
| Brachistochrone | PASS (converged) |
| Benchmarks | PASS (no regressions) |
| Format (C++) | PASS |
| Format (Python) | PASS |

<details>
<summary>Benchmark comparison</summary>

```
<paste bench_track.sh compare output here>
```

</details>
```

If any check failed, mark it as FAIL with a brief explanation.

### 9. Report to user

Print a concise summary of all results. If any check failed, clearly
state which ones and why.
