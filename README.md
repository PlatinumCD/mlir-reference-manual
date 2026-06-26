# MLIR Reference Manual

A beginner-focused Quarto book about MLIR dialects, operations,
transformations, conversions, and their roles in compiler pipelines.

The book starts with MLIR foundations, then moves through upstream MLIR
dialects and third-party MLIR ecosystems such as ONNX-MLIR, torch-mlir,
StableHLO, IREE, CIRCT, IMEX, and DaCe.

## Repository Structure

```text
.
├── _quarto.yml
├── index.qmd
├── chapters/
├── styles.css
├── README.md
└── .github/workflows/pages.yml
```

`_quarto.yml` defines the book metadata, chapter order, and HTML output
settings. `index.qmd` is the opening About page. All book chapters live in
`chapters/`.

## Book Structure

The book is organized into these sections:

1. About, introduction, and MLIR foundations
2. Core IR Structure
3. Basic Computation
4. Tensor, Shape, and Memory
5. Structured Computation and Optimization
6. ML and Model-Level IR
7. Parallelism, Accelerators, and Distributed Compute
8. Rewrite, Transform, and Metaprogramming
9. Target and Lowering Dialects
10. GPU Vendor and Hardware Dialects
11. CPU and Architecture-Specific Dialects
12. Third-Party MLIR Ecosystem
13. Frontend and Model Dialects
14. IREE Compiler and Runtime Dialects
15. Hardware and Circuit Dialects
16. Array, Data-Centric, and HPC Extensions

## Local Preview

Install Quarto, then run:

```bash
quarto preview
```

Quarto will print the local preview URL.

## Render

To build the static site locally:

```bash
quarto render
```

The generated site is written to `_book/`. Generated Quarto output is ignored
by git.

## Publishing

The workflow in `.github/workflows/pages.yml` publishes the site whenever
changes are pushed to `master`.
