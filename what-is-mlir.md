# What Is MLIR?

MLIR stands for Multi-Level Intermediate Representation. It is a compiler IR
framework designed to represent programs at many levels of abstraction, from
high-level tensor graphs down to low-level target-specific code.

The important word is "multi-level." A traditional compiler often has one main
IR at a time. MLIR instead lets a compiler use many dialects together. Each
dialect defines a vocabulary of operations, types, attributes, and rules for a
specific domain. A compiler pipeline can start with high-level dialects, apply
optimizations while information is still rich, then gradually lower into
lower-level dialects that are closer to hardware or an output format.

## Why Intermediate Representations Exist

A compiler does not usually translate source code directly to machine code in
one step. It translates through intermediate representations, or IRs.

An IR gives the compiler a form of the program that is easier to analyze and
change than the original source language. Different IRs are good for different
jobs:

- a high-level IR can preserve domain concepts like tensors, maps, kernels, or
  parallel loops;
- a mid-level IR can expose memory, control flow, and optimization structure;
- a low-level IR can represent target-specific instructions, calling
  conventions, address spaces, and runtime details.

MLIR gives compiler authors a common infrastructure for building those levels
without inventing a new IR system every time.

## The Shape Of MLIR

MLIR is built from a small number of core concepts:

- operations;
- values;
- types;
- attributes;
- blocks;
- regions;
- modules;
- dialects.

Once these ideas are clear, most MLIR syntax becomes much less mysterious.

## Operations

An operation is the central unit of MLIR. Operations are what the program does.

Examples:

```mlir
%c0 = arith.constant 0 : index
%sum = arith.addi %lhs, %rhs : i32
func.return %sum : i32
```

Each operation has a name. Most operation names are written as
`dialect.operation`, such as `arith.addi`, `func.return`, or `memref.load`.
The dialect prefix tells you which vocabulary the operation belongs to.

An operation may have:

- operands, which are input values;
- results, which are output values;
- attributes, which are compile-time metadata;
- regions, which contain nested blocks and operations;
- successors, which point to other blocks for control flow;
- traits and interfaces, which describe reusable behavior and constraints.

Some operations look like familiar instructions. Others represent large
semantic concepts. For example, `linalg.matmul` can represent matrix
multiplication at a level where the compiler still knows it is a structured
linear algebra operation.

## Values

Values are the edges between operations. An operation result is a value, and a
block argument is also a value.

MLIR is usually read in SSA style: a value is defined once and then used by
operations that come later in the valid scope.

```mlir
%a = arith.constant 4 : i32
%b = arith.constant 5 : i32
%c = arith.addi %a, %b : i32
```

Here `%a`, `%b`, and `%c` are values. `%a` and `%b` are results of constant
operations. `%c` is the result of an addition.

The printed names are for human readability. The exact names are not the
semantic identity of the values.

## Types

Every value has a type. Types describe what kind of value an operation consumes
or produces.

Examples:

```mlir
i32
f32
index
tensor<4x8xf32>
memref<?x?xf32>
vector<16xf32>
```

Types are one of the main ways dialects communicate. `tensor<4x8xf32>` says
"an immutable shaped tensor value." `memref<?x?xf32>` says "a shaped reference
to memory, with dynamic dimensions." Those are not the same concept, even if
they describe the same element type and rank.

## Attributes

Attributes are compile-time data attached to operations. They are not SSA
values. They are constants in the IR that describe properties of an operation.

Examples include integer attributes, string attributes, affine maps, array
attributes, symbol names, memory spaces, and target metadata.

```mlir
%0 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
```

The dense constant payload is an attribute. It is part of the operation, not a
runtime value computed by another operation.

## Blocks

A block is a sequence of operations with optional block arguments. In
control-flow-oriented IR, blocks are the basic blocks that branches jump to. In
structured IR, blocks often appear inside regions to represent the body of a
loop, function, or transformation.

Example:

```mlir
^bb0(%arg0: i32):
  %c1 = arith.constant 1 : i32
  %next = arith.addi %arg0, %c1 : i32
  cf.br ^bb1(%next : i32)
```

`%arg0` is a block argument. It behaves like a value defined at the beginning
of the block.

## Regions

A region contains one or more blocks. Operations can own regions, which is how
MLIR represents nesting.

Examples:

- `func.func` owns a region for the function body.
- `scf.for` owns a region for the loop body.
- `scf.if` owns regions for the then and else bodies.
- `gpu.launch` owns a region for the GPU kernel body.

This nesting is one of MLIR's most important design choices. Instead of forcing
every concept into flat basic blocks immediately, MLIR can keep structure while
it is useful.

## Modules

A module is the usual top-level container for MLIR. It is itself an operation,
normally from the `builtin` dialect.

```mlir
module {
  func.func @add_one(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1 : i32
    return %0 : i32
  }
}
```

A module can contain functions, global objects, nested modules, and other
top-level operations depending on the dialects involved.

## Dialects

A dialect is a namespace and semantic package for MLIR IR.

A dialect may define:

- operations;
- types;
- attributes;
- custom syntax;
- verification rules;
- canonicalization patterns;
- folding behavior;
- interfaces;
- transformations;
- conversions to and from other dialects.

The `arith` dialect defines arithmetic operations. The `tensor` dialect defines
operations on immutable tensor values. The `memref` dialect defines operations
on shaped memory references. The `gpu` dialect defines a generic GPU execution
model. The `llvm` dialect defines operations that map closely to LLVM IR.

Dialects are the reason MLIR can represent different compiler levels in one
system. A compiler can mix dialects in the same module when that is useful.

## Dialects Compose

MLIR programs commonly contain more than one dialect at the same time.

```mlir
func.func @add(%lhs: i32, %rhs: i32) -> i32 {
  %sum = arith.addi %lhs, %rhs : i32
  return %sum : i32
}
```

This tiny function uses:

- `func` for the function and return structure;
- `arith` for arithmetic;
- `builtin` implicitly for the surrounding module if one is printed.

Larger examples may combine `linalg`, `tensor`, `memref`, `scf`, `vector`,
`gpu`, and target dialects in one pipeline. The key is that each dialect
contributes the vocabulary for one level or concern.

## How External Dialects Connect

MLIR's dialect system is open-ended. A dialect does not need to live in the
LLVM repository to be part of an MLIR-based compiler. External projects can
define their own operations, types, attributes, passes, and conversions while
still using MLIR's parser, verifier, pass manager, rewrite infrastructure, and
dialect conversion framework.

That is how projects such as ONNX-MLIR, torch-mlir, StableHLO, IREE, CIRCT,
IMEX, and DaCe fit into the MLIR world. They reuse the same core ideas
described in this chapter, but they add dialects for their own domains:
framework import, model portability, runtime execution, hardware design,
distributed arrays, and data-centric compilation.

External dialects usually connect to upstream MLIR in one of three ways:

- they lower into upstream dialects such as `tensor`, `linalg`, `memref`,
  `scf`, `vector`, `gpu`, `llvm`, or `spirv`;
- they consume upstream dialects and add project-specific runtime or backend
  meaning;
- they sit at a tool boundary where another framework, runtime, hardware
  backend, or exchange format expects a specific representation.

The practical lesson is that upstream MLIR teaches the grammar, while
third-party dialects show how real compiler projects specialize that grammar
for a domain. When you read an ecosystem dialect chapter later in the book,
look for the same questions: what information is this dialect preserving, what
does it lower from, what does it lower to, and what tool or runtime depends on
it?

## Passes

A pass is a compiler action that runs on IR. Some passes analyze the IR. Most
passes transform it.

Examples of pass jobs:

- remove unused operations;
- canonicalize operations into simpler forms;
- convert one dialect into another;
- tile loops;
- vectorize computation;
- bufferize tensors into memory;
- lower GPU operations to target-specific dialects.

In MLIR, passes run on operations. A pass might run on a whole module, on each
function, or on another operation type. Pass pipelines are sequences of passes.

## Transformations

A transformation changes IR while usually staying at the same conceptual level.
The output may use the same dialects as the input, or it may introduce a small
amount of related IR.

Examples:

- canonicalizing `x + 0` to `x`;
- folding constants;
- simplifying an `scf.if` with a known condition;
- tiling a `linalg` operation;
- cleaning up redundant tensor casts;
- rewriting vector transfer operations.

Transformations are often local or structural improvements. They make later
analysis, optimization, or lowering easier.

## Canonicalization

Canonicalization is a shared MLIR mechanism for applying operation-specific
simplifications. Dialects register canonicalization and folding behavior on
their operations. The canonicalizer pass applies loaded dialect patterns
greedily until it reaches a fixpoint or configured limits.

The practical beginner rule is:

Canonicalization makes IR cleaner and easier to optimize, but a correct
pipeline should not depend on canonicalization for basic correctness.

That distinction matters. A conversion pass should not only work after the
canonicalizer happened to hide a malformed assumption.

## Conversions

A conversion changes IR from one legal vocabulary to another. It is usually
more formal than a general transformation because it has a legality target:
after conversion, certain operations or dialects are legal, illegal, or
dynamically legal.

For example, a conversion might say:

- `linalg` ops are illegal and must become loops;
- `gpu` ops are illegal and must become target-specific GPU dialect ops;
- only `llvm` and a small set of helper ops may remain;
- some operations are legal only if their types have already been converted.

Conversions often use rewrite patterns and type converters. Type conversion is
important because lowering frequently changes both operations and types.

## Lowering

Lowering is the process of moving from a higher-level representation to a
lower-level one.

A rough tensor-to-CPU path might look like this:

```text
tosa or linalg on tensors
  -> tensor/memref plus bufferization
  -> scf or affine loops
  -> vector operations
  -> llvm dialect
  -> LLVM IR or object code
```

A rough GPU path might look like this:

```text
linalg or tensor computation
  -> loops and vector operations
  -> gpu dialect
  -> nvgpu/nvvm, amdgpu/rocdl, spirv, or xegpu/xevm
  -> target backend
```

Real pipelines vary. The important idea is that lowering gradually removes
high-level abstractions and replaces them with more explicit implementation
details.

## Verification

Verification checks whether IR obeys the rules of the operations and dialects
it uses. An operation can have custom verifier logic. Traits and interfaces can
also contribute constraints.

Examples of things verification may check:

- operand and result types match;
- a region has the expected number of blocks;
- a terminator operation is present where required;
- an attribute has a valid value;
- a symbol reference points to something valid;
- an operation is used only in the right parent context.

Verification is one reason dialects are powerful. They do not just give names
to operations; they define what valid IR means for a domain.

## Traits And Interfaces

Traits and interfaces let MLIR describe operation behavior in reusable ways.

A trait is a reusable property or constraint. An interface is a reusable API
that analyses and transformations can query.

This matters because a generic pass cannot understand every operation in every
dialect by hard-coding it. Instead, it can ask whether an operation implements
an interface or has a trait. That lets transformations work across dialects
when the operations expose the right semantics.

## Reading MLIR Syntax

A useful beginner strategy is to read an operation in this order:

1. Identify the dialect prefix.
2. Identify the operation name.
3. Read the operands.
4. Read the result types.
5. Look for attributes.
6. Look for nested regions.
7. Ask what dialect owns the surrounding operation.

For example:

```mlir
%0 = arith.addi %lhs, %rhs : i32
```

This means:

- dialect: `arith`;
- operation: `addi`;
- operands: `%lhs`, `%rhs`;
- result: `%0`;
- type: `i32`;
- meaning: integer addition.

For a nested operation:

```mlir
scf.if %cond {
  func.return
}
```

This means:

- dialect: `scf`;
- operation: `if`;
- operand: `%cond`;
- nested region: the body between braces;
- meaning: structured conditional control flow.

## What "Important Dialect" Means

A dialect can be important in different ways.

Some dialects are important because almost every MLIR program uses them:
`builtin`, `func`, `arith`, `scf`, `cf`.

Some are important because they bridge major compiler levels:
`tensor`, `memref`, `bufferization`, `linalg`, `vector`, `llvm`.

Some are important for specific domains:
`gpu`, `omp`, `acc`, `mpi`, `tosa`, `sparse_tensor`.

Some are important near a backend:
`nvvm`, `rocdl`, `spirv`, `x86`, `arm_sve`, `emitc`.

When this book says a dialect matters, it explains which kind of importance is
in play.

## A Small End-To-End Mental Model

Imagine a compiler for tensor programs.

At the start, the program might be represented with high-level tensor or ML
operations. This is useful because the compiler can still see whole operations
like matrix multiplication, reshape, or elementwise maps.

Then the compiler may lower into structured computation dialects such as
`linalg`, where optimization passes can tile, fuse, or restructure work.

Next, bufferization may replace immutable tensor values with explicit memory
buffers in `memref`.

Then loops, vector operations, and target-specific operations make the program
closer to executable code.

Finally, the program lowers to a target dialect such as `llvm`, `spirv`,
`nvvm`, or `rocdl`.

MLIR's value is that those stages can live in one extensible framework instead
of being separate compiler worlds.

## Source Map

Useful upstream files for this chapter:

- `mlir/docs/LangRef.md`
- `mlir/docs/Tutorials/UnderstandingTheIRStructure.md`
- `mlir/docs/PassManagement.md`
- `mlir/docs/Passes.md`
- `mlir/docs/DialectConversion.md`
- `mlir/docs/Canonicalization.md`
- `mlir/docs/PatternRewriter.md`
- `mlir/include/mlir/IR/`
- `mlir/include/mlir/Pass/`
- `mlir/lib/IR/`
- `mlir/lib/Pass/`
