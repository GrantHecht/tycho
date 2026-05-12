# Documentation pipeline — one-time GitHub setup

This note documents the manual GitHub configuration required for the
docs CI pipeline to work end-to-end. The pipeline itself
(`docs-build.yml`, `docs-linkcheck.yml`, and `stubs-drift.yml`) lives in
`.github/workflows/`. The deploy step is the second job of
`docs-build.yml` (gated on `push: main`); there is no separate
`docs-deploy.yml`.

## GitHub Pages: source = "GitHub Actions"

The `deploy` job in `docs-build.yml` uses
[`actions/deploy-pages@v4`](https://github.com/actions/deploy-pages),
which requires the repository's Pages source to be set to **GitHub
Actions** (NOT "Deploy from a branch").

**Manual configuration (one-time):**

1. Go to the repository's **Settings** → **Pages**.
2. Under **Source**, select **GitHub Actions**.
3. Custom domain: leave empty initially. The default URL is
   `https://<owner>.github.io/<repo>/`.

If the first deploy run fails with a "Pages site not found" error, this
setting is the most likely cause.

## Deploy permissions

The `deploy` job in `docs-build.yml` requests `pages: write` and
`id-token: write`. These are job-scoped permissions — they're claimed
only by the deploy job on `push: main` events, not by PR runs (the
`deploy` job's `if:` guard skips it entirely on PRs).

## CI matrix

- **`docs-build.yml`** — runs on every PR + push to main. Two jobs:
  - `build` — builds `_tychopy` and the docs site; uploads the
    `github-pages` artifact on main and a per-PR `docs-html-NNN`
    artifact on PRs.
  - `deploy` — depends on `build`, gated on `push:main`. Uses
    `configure-pages` + `deploy-pages` to publish the artifact from
    the same workflow run.
- **`docs-linkcheck.yml`** — weekly cron (Mondays 06:00 UTC) plus
  manual dispatch.
- **`stubs-drift.yml`** — runs on PRs touching bindings, headers, or
  the committed stub snapshot; fails if regenerated stubs differ from
  the committed snapshot.

## Local validation

Before opening a PR that touches bindings or headers:

```bash
cd build
ninja -j6 _tychopy
ninja tycho_docs                  # build the site locally
ninja tychopy_stubs_snapshot      # regenerate committed snapshot
cd ..
git status tychopy/_stubs/        # commit any changes
```

## Nitpicky mode + content authoring

`docs/source/conf.py` sets `nitpicky = True` and CI invokes Sphinx with
`-W --keep-going -n`, so every undocumented cross-reference becomes a
build failure. The current `nitpick_ignore` list is intentionally narrow
(only `tycho` and `tycho::integrators` are exempted). As public-API
docstring coverage expands, expect to either:

- extend `nitpick_ignore` with each new namespace component that is
  referenced before it's fully documented, or
- relax to `nitpicky = False` once cross-reference coverage is broad
  enough that nitpicky's signal-to-noise is no longer worth the
  maintenance burden.

This is a content-authoring concern, not an infrastructure concern —
revisit when the first PR adding substantive C++ docstrings runs into
the strict mode wall.
