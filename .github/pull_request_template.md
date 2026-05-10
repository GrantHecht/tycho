## Summary

<!-- 1-3 bullets describing what changed and why. -->

## Test plan

- [ ] C++ unit tests pass (`ctest --output-on-failure`)
- [ ] Python examples pass (`MPLBACKEND=Agg python scripts/run_examples.py`)
- [ ] Benchmarks show no unexplained regressions (`bench/bench_track.sh compare`)

## Documentation

- [ ] If this PR touches a public C++ symbol in `include/tycho/`, the
      Doxygen docstring is present and follows the JavaDoc `@`-style
      conventions in `docs/source/contributing/style.md`.
- [ ] If this PR touches a Python binding signature or its `R"doc(...)"`
      docstring, the **Python** NumPy-style docstring is updated alongside
      the **C++** Doxygen docstring (the two-docstring rule).
- [ ] If this PR changes any binding signature, the stub snapshot was
      regenerated (`cmake --build build --target tychopy_stubs_snapshot`)
      and committed. CI will flag drift otherwise.
