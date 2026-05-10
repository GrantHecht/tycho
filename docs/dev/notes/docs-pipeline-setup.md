# Documentation pipeline — one-time GitHub setup

This note documents the manual GitHub configuration required for the
docs CI pipeline to work end-to-end. The pipeline itself
(`docs-build.yml`, `docs-deploy.yml`, `docs-linkcheck.yml`, and
`stubs-drift.yml`) lives in `.github/workflows/`.

## GitHub Pages: source = "GitHub Actions"

The `docs-deploy.yml` workflow uses
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

`docs-deploy.yml` requests `pages: write` and `id-token: write`.
These are workflow-scoped permissions — they're claimed only by the
deploy workflow on `push: main` events, not by PR runs.

## CI matrix

- **`docs-build.yml`** — runs on every PR + push to main; builds
  `_tychopy` and the docs site; uploads `github-pages` artifact on main
  and a per-PR `docs-html-NNN` artifact on PRs.
- **`docs-deploy.yml`** — runs only when `docs-build` succeeds on main;
  pulls the artifact and deploys.
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
