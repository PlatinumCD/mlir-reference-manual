# SCF Dialect

## Beginner Summary

The `scf` dialect is MLIR's structured control flow dialect. It represents
loops, conditionals, switches, and structured parallel regions as nested MLIR
regions instead of raw blocks and branches.

For a beginner, the main idea is this: `scf` is where a compiler can still see
that a program has a loop or an if statement. That structure makes
optimization easier. Later, when the compiler needs a lower-level form, `scf`
can be converted to branch-based `cf`, GPU launches, OpenMP, SPIR-V structured
control flow, or EmitC.

## Why This Dialect Exists

MLIR has both structured and unstructured control flow.

`cf` models low-level control flow with branches between blocks. That is close
to a traditional control-flow graph. It is flexible, but the original source
structure can be hard to recover.

`scf` keeps common control-flow shapes explicit:

- `scf.for` is a counted loop.
- `scf.if` is a conditional with then and optional else regions.
- `scf.while` is a loop with explicit condition and body regions.
- `scf.parallel` and `scf.forall` express parallel iteration spaces.
- `scf.index_switch` expresses a structured switch over an `index` value.

This is useful in the middle of a compiler pipeline. Dialects such as `affine`,
`linalg`, `vector`, `tosa`, and hardware-specific dialects often create or use
SCF before lowering to target-specific IR.

## When It Matters

SCF matters whenever the compiler needs to preserve loop and branch structure.

Use it when:

- You need readable loop and if constructs in MLIR.
- You are lowering from higher-level tensor, affine, vector, or model dialects.
- You want to apply loop transformations such as peeling, tiling, fusion,
  specialization, unrolling, or software pipelining.
- You need a target-independent representation before choosing CPU, GPU,
  OpenMP, SPIR-V, EmitC, or LLVM lowering.
- You want parallel loops that are not yet committed to a concrete runtime or
  hardware mapping.

Avoid using it when:

- You are already at low-level CFG form and do not need structured analysis.
- You need arbitrary branches that do not fit structured regions.
- You are encoding target intrinsics or ABI details. Those belong in target
  dialects such as `llvm`, `gpu`, `spirv`, `omp`, or `emitc`.

## What It Means

An SCF operation usually owns one or more regions. A region is a nested body of
IR with its own block arguments and terminator. This is how SCF represents
structured control flow without exposing arbitrary branch edges.

SCF also uses SSA values to model values carried through loops and conditionals:

- Loop-carried values enter `scf.for` through `iter_args`.
- `scf.yield` passes values from a region back to the parent op.
- `scf.condition` decides whether an `scf.while` continues.
- `scf.reduce` and `scf.reduce.return` describe reductions for
  `scf.parallel`.
- `scf.forall.in_parallel` describes how parallel `shared_outs` are combined.

The important implication is that control-flow structure and value flow are
part of the IR, not just comments around blocks.

## Core Concepts

### Structured Regions

Most SCF operations contain nested regions. For example, an `scf.if` contains a
then region and may contain an else region. An `scf.for` contains the loop body.
An `scf.while` contains a before region and an after region.

The region shape is part of the operation verifier. This is why invalid loop
or branch structure is usually caught early.

### Terminators

SCF region terminators are also SCF operations.

- `scf.yield` terminates regions of `scf.for`, `scf.if`,
  `scf.index_switch`, `scf.while`, and `scf.execute_region`.
- `scf.condition` terminates the before region of `scf.while`.
- `scf.reduce` terminates `scf.parallel`.
- `scf.reduce.return` terminates a reduction combiner region.
- `scf.forall.in_parallel` terminates `scf.forall`.

Some terminators can be omitted in custom syntax when no values are returned.
The parser or builder inserts them implicitly.

### Loop-Carried Values

`scf.for` and `scf.while` can carry values from one iteration to the next.
This is the structured equivalent of updating variables in a source-language
loop.

The types must line up:

- Initial values define the loop-carried value types.
- Region arguments receive the current iteration values.
- `scf.yield` provides the next iteration values.
- The parent operation results provide the final values.

### Parallel Semantics

`scf.parallel` means the iteration space may execute in any order and in
parallel. If there are data races, behavior is undefined. Reductions are
explicit.

`scf.forall` is a newer target-independent parallel region construct. It
models a set of virtual threads and can carry mapping attributes that later
lowerings interpret for a concrete target such as GPU.

## Operations

SCF defines 12 operations.

### Basic Structured Control Flow

- `scf.if`: A structured if-then-else operation. It takes an `i1` condition and
  has a then region plus an optional else region. If it returns values, both
  branches must yield matching values.
- `scf.index_switch`: A structured switch on an `index` value. It contains one
  default region and zero or more case regions.
- `scf.execute_region`: Executes a region exactly once. It is useful when an
  operation normally allows only a single block but a temporary multi-block
  region is needed. The optional `no_inline` unit attribute asks canonicalizers
  to preserve the op until an explicit lowering step.

### Sequential Loops

- `scf.for`: A counted loop with lower bound, upper bound, and positive step.
  Bounds are half-open: the lower bound is included and the upper bound is
  excluded. It may have loop-carried values through `iter_args`. The
  `unsignedCmp` unit attribute makes integer loop comparisons unsigned.
- `scf.while`: A general structured while or do-while loop. It has a before
  region terminated by `scf.condition` and an after region terminated by
  `scf.yield`.
- `scf.condition`: The terminator of the before region of `scf.while`. Its
  first operand is the continuation condition. If true, execution enters the
  after region; otherwise the `scf.while` exits.
- `scf.yield`: The common terminator for many SCF regions. It returns values
  from the region to the parent SCF operation.

### Parallel Loops

- `scf.parallel`: A multi-dimensional parallel loop. Its iteration order is
  unspecified, and the body may execute in parallel. It can return reduction
  results through `scf.reduce`.
- `scf.reduce`: The terminator of `scf.parallel`. It lists values to reduce
  and owns one combiner region per reduced value.
- `scf.reduce.return`: The terminator inside each `scf.reduce` combiner
  region. It returns the combined value.
- `scf.forall`: A target-independent multi-dimensional parallel region. It
  represents virtual threads and optional shared tensor outputs. It can carry a
  `mapping` attribute array whose elements implement the device mapping
  interface.
- `scf.forall.in_parallel`: The terminator of `scf.forall`. Its nested region
  contains aggregation actions, commonly tensor parallel insert operations for
  shared outputs. For a `forall` without outputs, the region may be empty.

## Attributes And Types

SCF does not define custom types. It uses standard MLIR types such as `index`,
`i1`, integer and floating-point types, tensors, memrefs, and vectors.

SCF has operation-specific attributes:

- `scf.for` has `unsignedCmp`.
- `scf.forall` has `staticLowerBound`, `staticUpperBound`, `staticStep`, and
  `mapping`.
- `scf.index_switch` has `cases`.
- `scf.execute_region` has `no_inline`.

The `mapping` attribute on `scf.forall` is not an SCF-specific mapping format.
It accepts attributes that implement the device mapping interface, such as GPU
mapping attributes.

## Transformations

SCF has a large set of loop and structure transformations. Some are normal
passes; others are transform dialect ops for scriptable transformations.

### SCF Passes

- `scf-for-loop-canonicalization`: Canonicalizes operations inside `scf.for`
  loop bodies. It currently focuses on affine min/max computations involving
  induction variables, loop bounds, and loop steps.
- `scf-for-loop-peeling`: Peels `scf.for` loops at their bounds. Options:
  `peel-front` and `skip-partial`.
- `scf-for-loop-range-folding`: Folds add/mul operations into loop ranges.
- `scf-for-loop-specialization`: Specializes `scf.for` loops for
  vectorization.
- `scf-for-to-while`: Rewrites `scf.for` loops as `scf.while` loops.
- `scf-forall-to-for`: Converts `scf.forall` loops to nested `scf.for` loops.
- `scf-forall-to-parallel`: Converts `scf.forall` loops to `scf.parallel`
  loops.
- `scf-parallel-for-to-nested-fors`: Converts `scf.parallel` loops to nested
  `scf.for` loops.
- `scf-parallel-loop-fusion`: Fuses adjacent `scf.parallel` loops.
- `scf-parallel-loop-specialization`: Specializes `scf.parallel` loops for
  vectorization.
- `scf-parallel-loop-tiling`: Tiles `scf.parallel` loops. Options:
  `parallel-loop-tile-sizes` and `no-min-max-bounds`.
- `test-scf-parallel-loop-collapsing`: A test pass for the
  `scf::collapseParallelLoops` utility. It is not a production pipeline pass,
  but it documents the collapse utility used by SCF tests.

### Transform Dialect Hooks

The SCF transform extension exposes these transform ops:

- `transform.apply_patterns.scf.for_loop_canonicalization`: Collects SCF
  for-loop canonicalization patterns.
- `transform.apply_conversion_patterns.scf.scf_to_control_flow`: Collects
  patterns that lower SCF to branch-based control flow.
- `transform.apply_conversion_patterns.scf.structural_conversions`: Collects
  SCF structural type conversion patterns.
- `transform.loop.coalesce`: Coalesces a perfect loop nest.
- `transform.loop.coalesce_nested`: Coalesces nested loops, including
  imperfectly nested loops.
- `transform.loop.forall_to_for`: Converts one `scf.forall` to nested
  `scf.for` loops.
- `transform.loop.forall_to_parallel`: Converts one `scf.forall` to
  `scf.parallel`.
- `transform.loop.fuse_sibling`: Fuses sibling `scf.for` or `scf.forall`
  loops when the caller guarantees independence and the loop shapes match.
- `transform.loop.outline`: Outlines a loop-like payload operation to a new
  function and replaces it with a call.
- `transform.loop.parallel_for_to_nested_fors`: Converts one `scf.parallel`
  to nested `scf.for` loops.
- `transform.loop.peel`: Peels the first or last iteration of `scf.for`.
- `transform.loop.pipeline`: Applies software pipelining to `scf.for`.
- `transform.loop.promote_if_one_iteration`: Removes a loop when it has a
  single iteration.
- `transform.loop.unroll`: Unrolls `scf.for` or `affine.for`.
- `transform.loop.unroll_and_jam`: Applies unroll-and-jam to `scf.for` or
  `affine.for`.
- `transform.scf.take_assumed_branch`: Replaces an `scf.if` with a selected
  branch when the transform script asserts that branch is safe to take.

### Library-Level Utilities And Patterns

SCF also exposes C++ utilities and pattern population APIs used by other
passes:

- `forallToForLoop`
- `forallToParallelLoop`
- `parallelForToNestedFors`
- `peelForLoopAndSimplifyBounds`
- `peelForLoopFirstIteration`
- `tileParallelLoop`
- `pipelineForLoop`
- `populateSCFStructuralTypeConversions`
- `populateSCFStructuralTypeConversionTarget`
- `populateSCFLoopPipeliningPatterns`
- `populateSCFForLoopCanonicalizationPatterns`
- `populateSCFRotateWhileLoopPatterns`

These names matter when you read pass implementations: many "dialect passes"
are thin wrappers around these reusable utilities.

## Conversions And Lowering Paths

SCF sits in the middle of many MLIR pipelines.

### Common Inputs To SCF

- `lower-affine`: Lowers `affine.for` to `scf.for`, `affine.if` to `scf.if`,
  and affine expressions to arithmetic.
- `lift-cf-to-scf`: Lifts some branch-based `cf` control flow back to SCF. It
  is named "lift" because it is not guaranteed to eliminate all `cf` ops.
- `convert-vector-to-scf`: Lowers selected vector operations, especially
  higher-rank vector transfers, to loops and conditionals using SCF.
- `tosa-to-scf`: Converts TOSA control-flow operations to equivalent SCF
  operations.
- `convert-arm-sme-to-scf`: Lowers selected ArmSME operations into SCF plus
  supporting dialects.
- `convert-openacc-to-scf`: Converts some OpenACC constructs to OpenACC plus
  SCF.
- Linalg tiling and fusion utilities commonly generate `scf.for` or
  `scf.forall` loops as the loop structure around tiled computations.

### SCF To Other Dialects

- `convert-scf-to-cf`: Lowers structured SCF operations to the `cf` dialect.
  This is the usual path before lowering to LLVM-style low-level control flow.
- `convert-scf-to-emitc`: Lowers SCF to EmitC while preserving structured
  control flow for C/C++ emission.
- `convert-scf-to-openmp`: Converts `scf.parallel` to OpenMP parallel and
  workshare constructs. Option: `num-threads`.
- `convert-scf-to-spirv`: Converts SCF to SPIR-V structured control flow. When
  SCF ops yield values, the pass uses SPIR-V variables and loads/stores because
  SPIR-V structured control-flow ops do not yield SSA values directly.
- `convert-parallel-loops-to-gpu`: Converts mapped `scf.parallel` operations
  to `gpu.launch` operations.
- `convert-affine-for-to-gpu`: An adjacent conversion in the same conversion
  area. It converts top-level `affine.for` loops to GPU kernels, not SCF ops.

### Internal SCF Normalization Paths

- `scf-forall-to-for`: Turn target-independent virtual threads into sequential
  loops.
- `scf-forall-to-parallel`: Turn `scf.forall` into `scf.parallel`.
- `scf-parallel-for-to-nested-fors`: Turn parallel loops into sequential nested
  loops.
- `scf-for-to-while`: Express counted loops as `scf.while`.

These are useful when a downstream target cannot handle the richer SCF form
directly.

## Example IR

### Counted Loop With Loop-Carried State

This example uses `scf.for`, `scf.if`, and `scf.yield`.

```mlir
module {
  func.func @sum_if_enabled(%n: index, %enabled: i1) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32

    %sum = scf.for %i = %c0 to %n step %c1
        iter_args(%acc = %zero) -> (f32) {
      %candidate = arith.addf %acc, %one : f32
      %next = scf.if %enabled -> (f32) {
        scf.yield %candidate : f32
      } else {
        scf.yield %acc : f32
      }
      scf.yield %next : f32
    }

    return %sum : f32
  }
}
```

Read it as: start with `%zero`; for each iteration, either add one or keep the
current value; yield the next accumulator value.

### While Loop, Switch, And Execute Region

This example uses `scf.while`, `scf.condition`, `scf.index_switch`,
`scf.execute_region`, and `scf.yield`.

```mlir
module {
  func.func @choose_after_counting(%n: index, %which: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %count = scf.while (%i = %c0) : (index) -> index {
      %keep_going = arith.cmpi slt, %i, %n : index
      scf.condition(%keep_going) %i : index
    } do {
    ^bb0(%j: index):
      %next = arith.addi %j, %c1 : index
      scf.yield %next : index
    }

    %chosen = scf.index_switch %which -> i32
    case 0 {
      %ten = arith.constant 10 : i32
      scf.yield %ten : i32
    }
    case 1 {
      %twenty = arith.constant 20 : i32
      scf.yield %twenty : i32
    }
    default {
      %thirty = arith.constant 30 : i32
      scf.yield %thirty : i32
    }

    %result = scf.execute_region -> i32 {
      %unused = arith.index_cast %count : index to i32
      scf.yield %chosen : i32
    }

    return %result : i32
  }
}
```

`scf.execute_region` is not changing the value here. It shows how a nested
single-execution region can return values to its parent.

### Parallel Reduction

This example uses `scf.parallel`, `scf.reduce`, and `scf.reduce.return`.

```mlir
module {
  func.func @parallel_count(%n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32

    %sum = scf.parallel (%i) = (%c0) to (%n) step (%c1)
        init (%zero) -> f32 {
      scf.reduce(%one : f32) {
      ^bb0(%lhs: f32, %rhs: f32):
        %next = arith.addf %lhs, %rhs : f32
        scf.reduce.return %next : f32
      }
    }

    return %sum : f32
  }
}
```

The reduction combiner receives two values and returns one combined value. For
parallel execution, reduction functions should be associative and commutative if
you need deterministic results.

### Forall Virtual Threads

This example uses `scf.forall` and `scf.forall.in_parallel`.

```mlir
module {
  func.func @virtual_threads(%num_threads: index) {
    scf.forall (%thread_id) in (%num_threads) {
      scf.forall.in_parallel {
      }
    }
    return
  }
}
```

This is a minimal `forall` with no shared outputs. Real `forall` loops often
use `shared_outs` and terminator actions such as tensor parallel insert
operations to combine per-thread tensor slices.

## How To Use It

For hand-written or generated MLIR, start with the simplest SCF operation that
matches the source construct:

- Use `scf.if` for structured conditional values.
- Use `scf.for` for counted loops with known lower bound, upper bound, and
  positive step.
- Use `scf.while` when the loop condition is not naturally a counted range.
- Use `scf.parallel` for unordered parallel iteration with explicit
  reductions.
- Use `scf.forall` for target-independent virtual-thread distribution.
- Use `scf.index_switch` for structured dispatch on an `index` value.

For compiler pipelines, a common path is:

```text
higher-level dialects
  -> affine/linalg/vector/tosa transformations
  -> scf
  -> loop optimizations and normalization
  -> cf, gpu, omp, spirv, emitc, or target-specific lowering
```

For CPU lowering to LLVM, SCF usually does not go directly to LLVM. A typical
route is `scf` to `cf`, then `cf` to `llvm`, with arithmetic, memref, func, and
other dialects lowered along the way.

## Gotchas

- `scf.for` uses a half-open range: it includes the lower bound and excludes
  the upper bound.
- The `scf.for` step must be strictly positive.
- `scf.for` has a no-overflow semantic requirement for the final induction
  value computation. Do not rely on wraparound.
- If an SCF op returns values, the region terminator must yield the same number
  and types of values.
- `scf.parallel` does not make side effects ordered. Data races are undefined.
- `scf.reduce` order is unspecified. Use associative and commutative
  reductions when result stability matters.
- `scf.forall` represents virtual threads, not a concrete hardware launch by
  itself. Mapping attributes need a lowering that understands them.
- Lowering to `cf` loses high-level structure. Run loop transformations before
  `convert-scf-to-cf` when those transformations need structured loops.
- `lift-cf-to-scf` is useful but not magic. Some CFG shapes cannot be fully
  lifted back into SCF.
- `scf.execute_region` is a bridge construct. If it remains late in a pipeline,
  check whether a lowering or inlining step is missing.

## What It Implies In A Compiler Pipeline

Seeing SCF in IR usually means the compiler is still in a structured middle-end
phase. Analyses can reason about loops and branches through operation
interfaces such as `LoopLikeOpInterface` and `RegionBranchOpInterface`.

As the pipeline moves toward code generation, SCF must either be normalized to a
form the target supports or lowered away:

- Sequential CPU-style code usually lowers through `cf`.
- C/C++ emission can preserve structure through EmitC.
- OpenMP lowering consumes `scf.parallel`.
- GPU lowering can consume mapped `scf.parallel`.
- SPIR-V lowering maps SCF to SPIR-V structured control-flow operations.

## Source Map

- `mlir/include/mlir/Dialect/SCF/IR/SCFOps.td`: Dialect definition and native
  operation definitions.
- `mlir/lib/Dialect/SCF/IR/SCF.cpp`: Parser, printer, verifier, folding, and
  canonicalization implementation.
- `mlir/include/mlir/Dialect/SCF/Transforms/Passes.td`: SCF pass definitions.
- `mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h`: Reusable SCF
  transformation utilities.
- `mlir/include/mlir/Dialect/SCF/Transforms/Patterns.h`: SCF pattern
  population APIs.
- `mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td`: Transform
  dialect extension for SCF and loop transformations.
- `mlir/include/mlir/Conversion/Passes.td`: SCF-related conversion pass
  definitions.
- `mlir/include/mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h`: SCF to
  branch-based `cf` conversion patterns.
- `mlir/include/mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h`: CFG to
  SCF lifting interface and pass declaration.
- `mlir/include/mlir/Conversion/VectorToSCF/VectorToSCF.h`: Vector transfer to
  SCF conversion patterns and options.
- `mlir/include/mlir/Conversion/SCFToGPU/SCFToGPUPass.h`: SCF parallel loop to
  GPU launch conversion pass declarations.
- `mlir/include/mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h`: SCF to SPIR-V
  pass declaration.
- `mlir/include/mlir/Conversion/SCFToEmitC/SCFToEmitC.h`: SCF to EmitC
  conversion patterns.
- `mlir/include/mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h`: SCF parallel loop
  to OpenMP conversion patterns.
