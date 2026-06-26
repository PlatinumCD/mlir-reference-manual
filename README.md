# MLIR Reference Manual

A beginner-focused Quarto book about MLIR dialects, operations,
transformations, conversions, and their roles in compiler pipelines.

## Local Preview

Install Quarto, then run:

```bash
quarto preview --host 0.0.0.0 --port 8811 --no-browser
```

The site will be available at:

```text
http://localhost:8811/
```

## Render

To build the static site locally:

```bash
quarto render
```

The generated site is written to `_book/`. That directory is ignored by git
because GitHub Actions renders the site during deployment.

## GitHub Pages

The workflow in `.github/workflows/pages.yml` publishes the Quarto book to
GitHub Pages whenever changes are pushed to `master`. It can also be run
manually from the GitHub Actions tab.

In the repository settings, configure Pages to use **GitHub Actions** as the
source.
