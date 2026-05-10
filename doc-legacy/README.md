# Archived asset_asrl Documentation

This directory contains the original asset_asrl Sphinx documentation as
inherited by Tycho. It is retained as a structural reference for the
ongoing Tycho documentation effort but is **not built or rendered**.

**Do not edit these files.** When porting content into the new
`docs/source/` tree, copy and rewrite rather than modify in place. Many
code samples here reference the pre-snake_case Tycho Python API and will
not run as-is.

The `.bak`-suffixed files (`CMakeLists.txt.bak`, `Doxyfile.in.bak`,
`conf.py.in.bak`) are kept as reference for the new `docs/` pipeline.

## Archived CI Scaffolding

The following asset_asrl-era files were also archived here because they
reference the now-removed CMake `sphinx` target and the old `build/doc/sphinx`
output path:

- `.readthedocs.yml.bak` — Read the Docs config. Replaced by the GitHub
  Pages hosting strategy in `docs/superpowers/specs/`.
- `github-workflows-archive/fast-docs.yml` — Manual docs build workflow.
  Replaced by `.github/workflows/docs-build.yml` (added in Task 6).
- `github-workflows-archive/pip-ubuntu-22.yml` — Scheduled pip-install + docs
  build. Replaced by `.github/workflows/docs-build.yml` and the
  `.github/workflows/pypi-wheels.yml` flow.
