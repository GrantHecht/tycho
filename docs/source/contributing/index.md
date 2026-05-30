# Contributing to Tycho

Welcome — thanks for your interest in contributing.

Tycho is a C++/Python library for trajectory design and optimal control,
maintained on GitHub at [GrantHecht/tycho](https://github.com/GrantHecht/tycho).
Contributions of all sizes are welcome: bug reports, fixes, new features,
documentation improvements, examples, and tests.

```{toctree}
:maxdepth: 1

building
style
testing
```

## Where to start

- **Building from source** — set up a development environment and produce
  a working `_tychopy` extension. Start with [Building](building.md).
- **Style conventions** — the C++ and Python docstring rules every PR is
  checked against, plus naming and formatting. Start with the
  [Style guide](style.md).
- **Testing** — how to run Tycho's Python unit tests, example suite, C++
  unit tests, and benchmarks. Start with [Testing](testing.md).

For internal engineering notes (build-performance analyses, AD design
docs, ongoing work plans), see
[`docs/dev/notes/`](https://github.com/GrantHecht/tycho/tree/main/docs/dev/notes)
in the repository. These are raw markdown and are not rendered into this
site.

## Reporting issues

File issues on the
[GitHub issue tracker](https://github.com/GrantHecht/tycho/issues).
For bugs, please include:

- The exact `cmake --preset` you used and the host OS / compiler version.
- A minimal reproducer (Python script or C++ snippet) plus the observed
  output and what you expected.
- Whether the issue reproduces on the latest `main`.
