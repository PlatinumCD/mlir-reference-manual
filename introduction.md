# Introduction

This book is a beginner's guide to MLIR dialects and the passes that move
programs between them. It is written for readers who know some programming and
may know the broad idea of a compiler, but who do not yet have a confident
mental model for MLIR.

MLIR can feel large because it is not one intermediate representation with one
fixed set of instructions. It is a framework for building many related IRs.
Those IRs are called dialects. Each dialect has its own operations, types,
attributes, verification rules, transformations, and lowering paths. This book
organizes that surface area into chapters you can read one dialect at a time.

The main goal is practical understanding. After reading a chapter, you should
know what the dialect is for, when you are likely to see it, what its important
operations mean, what passes usually transform it, and where it fits in a real
compiler pipeline.

## Who This Book Is For

This book is for:

- compiler beginners who want a map of MLIR instead of only reference pages;
- engineers reading MLIR dumps and trying to understand what they mean;
- people writing frontends, transformations, or lowering pipelines;
- readers who want to know which dialect to use for a specific job;
- readers who want source paths into LLVM so they can keep learning from the
  implementation.

You do not need to already know all of LLVM. It helps to know that compilers
usually translate programs through a series of intermediate forms before
generating executable code, but the next chapter introduces the MLIR-specific
terms used throughout the book.

## How The Book Is Organized

The book begins with two foundation chapters:

- `introduction.md`, this chapter, explains the purpose and reading strategy.
- `what-is-mlir.md` explains MLIR's core ideas: operations, values, types,
  attributes, regions, blocks, dialects, passes, transformations, conversions,
  and lowering.

The rest of the book is organized by dialect. Every dialect chapter is named
after the dialect namespace, for example `arith.md`, `linalg.md`, `gpu.md`,
and `llvm.md`.

The first half of the book focuses on upstream MLIR and LLVM dialects. Those
chapters teach the common vocabulary that most MLIR projects share. The second
half expands into third-party MLIR projects that define their own dialects on
top of the same infrastructure.

The upstream dialect chapters are grouped by domain:

- Core IR structure: `builtin`, `func`, `cf`, `scf`, `index`.
- Basic computation: `arith`, `math`, `complex`, `ub`.
- Tensor, shape, and memory modeling: `tensor`, `memref`, `bufferization`,
  `shape`, `sparse_tensor`, `quant`, `ptr`, `dlti`.
- Structured computation and optimization: `affine`, `linalg`, `vector`.
- Machine learning and model-level IR: `tosa`, `ml_program`.
- Parallelism, accelerators, and distributed compute: `async`, `gpu`, `acc`,
  `omp`, `mpi`, `shard`.
- Rewrite, transform, and metaprogramming: `transform`, `pdl`, `pdl_interp`,
  `irdl`, `smt`.
- Target and lowering dialects: `llvm`, `spirv`, `emitc`, `wasmssa`.
- GPU vendor and hardware dialects: `nvgpu`, `nvvm`, `amdgpu`, `rocdl`,
  `xegpu`, `xevm`.
- CPU and architecture-specific dialects: `x86`, `arm_neon`, `arm_sve`,
  `arm_sme`.

The third-party chapters are grouped by project domain:

- Frontend and model dialects: ONNX-MLIR, torch-mlir, and StableHLO.
- IREE compiler and runtime dialects: IREE's flow, stream, HAL, VM, codegen,
  target, and runtime-support dialects.
- Hardware and circuit dialects: CIRCT's structural, behavioral, simulation,
  verification, and emission dialects.
- Array, data-centric, and HPC extensions: IMEX and DaCe.

That order is also a good beginner reading order. Start with the dialects that
describe ordinary program shape, then move toward tensors, memory, structured
computation, parallelism, and finally target-specific lowering.

After the upstream MLIR chapters, read `external-ecosystem.md` before jumping
into third-party dialects. It explains how external projects extend MLIR, why
their dialects are often project-specific, and how they still connect back to
the same core concepts.

## How To Read A Dialect Chapter

Each dialect chapter follows the same structure.

### Beginner Summary

This section gives the short version: what the dialect is and what kind of
program concept it models.

For example, `arith` is about scalar integer and floating-point arithmetic.
`memref` is about shaped memory buffers. `gpu` is about a generic GPU execution
model. `llvm` is about representing LLVM IR inside MLIR.

### Why This Dialect Exists

MLIR dialects exist because different compiler stages need different kinds of
information. A high-level tensor operation is useful for optimization, but it
does not directly describe machine memory. A target-specific GPU intrinsic is
useful near code generation, but it is too low-level for most frontend work.

This section explains the gap the dialect fills.

### When It Matters

This section explains where the dialect normally appears in a pipeline. Some
dialects are common near the beginning of compilation. Some are used in the
middle for optimization. Some appear only near a target backend.

### When To Use It

This section is guidance for compiler authors. It answers questions like:

- Should a frontend emit this dialect?
- Should an optimization create this dialect?
- Is this dialect usually temporary?
- Is this dialect intended as a target of lowering?
- Is it better to stay at a higher level and let existing passes lower later?

### Core Concepts

Many dialects introduce ideas that are bigger than a list of operations. The
chapter calls those out before the operation reference.

Examples include:

- shaped tensor and memref types;
- affine maps;
- loop-carried values;
- GPU launch grids;
- vector masks;
- memory spaces;
- data layout metadata;
- transformation handles;
- target-specific intrinsics.

### Operations

Operations are the verbs of MLIR. A dialect chapter lists the operations in the
dialect and groups them by purpose.

For small dialects, the operation section can explain every op directly. For
large dialects, the chapter separates essential beginner operations from the
full operation inventory. The goal is to make the chapter useful as both a
learning guide and a reference map.

### Transformations

A transformation changes IR while staying at roughly the same level of
abstraction. Examples include canonicalization, simplification, loop
normalization, tiling, fusion, vector transfer rewriting, or removing
redundant operations.

Transformations are often implemented as passes, rewrite patterns, folders, or
canonicalization patterns. The chapter explains the important transformations
associated with the dialect and points to the source files where they live.

### Conversions And Lowering Paths

A conversion changes the legal dialects or types in the IR. Lowering is the
common compiler direction where a program moves from higher-level operations to
lower-level operations.

For example:

- tensor-style IR may lower through bufferization into `memref`;
- structured loops may lower toward `cf`;
- `linalg` may lower to loops, vectors, or library calls;
- `gpu` may lower toward `nvvm`, `rocdl`, `spirv`, or other targets;
- many dialects eventually lower toward `llvm` or another final emission
  target.

This section shows what commonly lowers into the dialect and what the dialect
commonly lowers into.

### Example IR

Each chapter includes small MLIR examples. The examples are intentionally small:
the point is to learn what the IR means, not to show a complete production
compiler.

Where useful, a chapter includes a before-and-after pass example. That is often
the easiest way to see why a dialect exists.

### Mental Model

This is the short phrase to keep in your head while reading or writing the
dialect. For example:

- `tensor` models immutable shaped values.
- `memref` models addressable shaped buffers.
- `scf` models structured control flow.
- `cf` models explicit block-to-block control flow.
- `llvm` models code that is close to LLVM IR.

The mental model is not a complete specification. It is a starting point for
reading IR without getting lost.

### Gotchas

MLIR has many details that are easy to misunderstand at first. Gotchas include
things like:

- a dialect name is not the same thing as a complete compiler level;
- tensors and memrefs are not interchangeable;
- a pass may require dialects to be loaded before it can create their ops;
- canonicalization is useful but should not be required for correctness;
- conversion success depends on legality, not only on whether a rewrite exists;
- target-specific dialects often assume hardware or backend constraints.

### Source Map

The source map points into the local LLVM checkout. It usually includes:

- dialect TableGen definitions under `mlir/include/mlir/Dialect/...`;
- operation implementation files under `mlir/lib/Dialect/...`;
- documentation under `mlir/docs/...`;
- tests under `mlir/test/...`;
- conversion code under `mlir/lib/Conversion/...`;
- transform code under `mlir/lib/Dialect/.../Transforms`;
- tool behavior exposed through `mlir-opt` and related utilities.

The source map matters because MLIR changes over time. This book explains the
concepts, but the implementation is the final authority for the exact operation
set, verifier rules, pass options, and conversion behavior in your checkout.

## Suggested Beginner Path

If you are new to MLIR, read in this order:

1. `what-is-mlir.md`
2. `builtin.md`
3. `func.md`
4. `arith.md`
5. `cf.md` and `scf.md`
6. `tensor.md`
7. `memref.md`
8. `bufferization.md`
9. `linalg.md`
10. `vector.md`
11. `gpu.md`
12. `llvm.md`

After that, choose by interest. If you care about ML models, read `tosa` and
`ml_program`. If you care about GPU lowering, read `nvgpu`, `nvvm`, `amdgpu`,
`rocdl`, `xegpu`, `xevm`, and `spirv`. If you care about writing rewrite
systems, read `pdl`, `pdl_interp`, and `transform`.

Once the upstream dialects make sense, read `external-ecosystem.md`. From
there, choose a project path: ONNX-MLIR, torch-mlir, and StableHLO for ML
frontends and model portability; IREE for runtime-oriented ML compilation;
CIRCT for hardware and circuit compilation; IMEX and DaCe for array,
distributed, and data-centric compilation.

## What To Expect From Examples

The examples in this book favor clarity over completeness. They are not meant
to be optimized kernels or full compiler pipelines. A good beginner example
should answer one question: "What does this operation or pass mean?"

When an example leaves out production details, the chapter says so. MLIR is
precise IR, and examples should not teach misleading shortcuts.

## Source Map

Useful upstream files for this chapter:

- `mlir/docs/LangRef.md`
- `mlir/docs/Tutorials/UnderstandingTheIRStructure.md`
- `mlir/docs/PassManagement.md`
- `mlir/docs/DialectConversion.md`
- `mlir/docs/Canonicalization.md`
- `mlir/lib/RegisterAllDialects.cpp`
