# Affine Dialect

## Beginner Summary

The `affine` dialect represents loops, conditionals, and memory accesses whose
indexing can be described with affine expressions.

An affine expression is a restricted integer expression built from loop
induction variables, symbols, constants, addition, subtraction,
multiplication by constants, floor division by constants, ceiling division by
constants, and modulo by constants.

For a beginner, the practical idea is:

- `affine.for` is a loop with analyzable bounds.
- `affine.if` is a branch with analyzable integer constraints.
- `affine.load` and `affine.store` are memory accesses with analyzable
  subscripts.
- `affine.apply`, `affine.min`, and `affine.max` compute analyzable index
  expressions.
- Affine loop passes can tile, fuse, unroll, vectorize, parallelize, and
  simplify loop nests because the compiler can reason about their bounds and
  memory accesses.

Think of `affine` as MLIR's "structured loop math" dialect. It is more
restricted than `scf`, but those restrictions are exactly what make stronger
loop and memory optimizations possible.

## Why This Dialect Exists

Many compiler optimizations need to answer questions such as:

- How many times does this loop run?
- Which memory element does this iteration access?
- Do two loop nests access the same element?
- Can these loops be fused?
- Can this loop be tiled?
- Is this loop parallel?
- What are the exact bounds of this slice?

General-purpose control flow makes those questions difficult. A loop whose
bound is computed by arbitrary code may be impossible to analyze precisely.

The `affine` dialect solves this by using a restricted but useful model:

```text
loop bounds and memory indices are affine functions of loop IVs and symbols
```

That restriction gives MLIR a strong middle-level representation for
polyhedral-style loop transformations, dependence analysis, locality
improvement, and exact index simplification.

The cost is that not every program fits. Data-dependent loops, arbitrary
branches, indirect indexing, and pointer-heavy control flow often belong in
other dialects such as `scf`, `cf`, `memref`, or `llvm`.

## When It Matters

The `affine` dialect matters when a compiler still wants to transform loop
nests aggressively.

It commonly appears in pipelines for:

- Dense linear algebra kernels.
- Stencil computations.
- Tensor-to-loop lowering.
- Locality optimization.
- Loop tiling.
- Loop fusion.
- Loop unrolling and unroll-and-jam.
- Scalar replacement of temporary memref values.
- Raising simple `memref.load` and `memref.store` operations into analyzable
  affine accesses.
- Generating explicit copies into faster memory spaces.
- Super-vectorization from loop nests to `vector` operations.
- Mapping simple top-level affine loops to GPU kernels.

A typical flow looks like this:

```text
linalg / tensor / structured computation
  -> affine.for, affine.load, affine.store, affine.apply
  -> affine tiling, fusion, simplification, scalar replacement
  -> affine vectorization or parallelization
  -> lower-affine
  -> scf / arith / memref / vector
  -> target-specific lowering
```

The key point is that `affine` is usually a transformation dialect. You use it
while the compiler still benefits from exact loop and index structure, then you
lower it away before final code generation.

## When To Use It

Use the `affine` dialect when loop bounds and memory indices are affine
functions of known induction variables and symbols.

Use it for:

- Counted loops with statically analyzable bounds.
- Loop nests over dense arrays.
- Index computations that are linear plus constants, constant divisions, or
  constant modulo.
- Memref accesses where each subscript is analyzable.
- Tiling and fusion pipelines.
- Dependence-sensitive loop optimization.
- Generating local buffers or explicit DMA-like transfers.
- Expressing structured parallel loop bands with affine bounds.

Do not force code into `affine` when the computation is not affine.

Use `scf` instead for:

- Data-dependent `while` loops.
- Bounds computed by arbitrary operations.
- Branches that depend on arbitrary boolean values.
- Irregular algorithms.
- Control flow where analysis precision is less important than generality.

Use `memref.load` and `memref.store` instead of `affine.load` and
`affine.store` when the access indices are not valid affine expressions.

## Core Concepts

### Affine Expressions

An affine expression is an integer index expression that MLIR can analyze.

Examples:

```mlir
#m0 = affine_map<(d0) -> (d0 + 4)>
#m1 = affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>
#m2 = affine_map<(d0) -> (d0 floordiv 4)>
#m3 = affine_map<(d0) -> (d0 mod 8)>
```

In these examples:

- `d0` and `d1` are dimensions.
- `s0` is a symbol.
- Constants can appear directly.
- Multiplication is allowed by constants, not by arbitrary runtime values.

The restrictions are intentional. They let MLIR compose, simplify, compare,
and bound the expressions.

### Dimensions And Symbols

Affine maps use dimensions and symbols:

```mlir
#map = affine_map<(d0, d1)[s0] -> (d0 + 2 * d1 + s0)>
```

Dimensions are usually loop induction variables or values that vary inside the
current affine scope.

Symbols are values that are treated as invariant for the affine expression,
such as function arguments, constants, or values defined outside the relevant
affine scope.

The difference matters because Affine verification depends on whether each
operand is a valid dimension or a valid symbol at the operation's location.

In custom syntax, dimension operands are written in parentheses and symbol
operands are written in square brackets:

```mlir
%j = affine.apply #map(%i, %k)[%offset]
```

Here `%i` and `%k` are dimension operands and `%offset` is a symbol operand.

### Affine Maps And Integer Sets

An affine map computes one or more index results:

```mlir
#map = affine_map<(d0)[s0] -> (d0 + s0, d0 + 4)>
```

An integer set describes constraints:

```mlir
#set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>
```

`affine.apply`, `affine.min`, `affine.max`, loop bounds, and memory accesses
use affine maps. `affine.if` uses integer sets.

### Half-Open Loops

`affine.for` uses half-open loop ranges:

```text
for i = lower_bound; i < upper_bound; i += step
```

The step is a positive constant. Lower and upper bounds are affine maps. If a
bound map has multiple results, the lower bound is interpreted as a maximum
and the upper bound is interpreted as a minimum.

This is why custom syntax can use:

```mlir
affine.for %i = max #lb(%n) to min #ub(%n) {
  ...
}
```

### Analyzable Memory Access

`affine.load` and `affine.store` are like `memref.load` and `memref.store`,
but their subscripts must be affine expressions.

```mlir
%v = affine.load %A[%i, %j + 1] : memref<64x64xf32>
affine.store %v, %B[%i, %j] : memref<64x64xf32>
```

This gives MLIR enough information for dependence analysis and transformations
such as fusion, scalar replacement, and local buffer generation.

### Affine Is Not The Final Form

Backends generally do not consume Affine operations directly.

After Affine-specific transformations are done, `lower-affine` converts the
dialect to more primitive dialects:

```text
affine.for       -> scf.for
affine.if        -> scf.if
affine.apply     -> arith index arithmetic
affine.load      -> memref.load
affine.store     -> memref.store
affine.parallel  -> scf.parallel
affine.vector_*  -> vector.load / vector.store
```

This is a common MLIR pattern: use a structured dialect while it helps
optimization, then lower it to more general dialects.

## Operations

### Index Operations

`affine.apply`
: Applies a single-result affine map to `index` operands and returns an
  `index`. It is the basic operation for naming an affine index computation in
  SSA form.

```mlir
#map = affine_map<(d0)[s0] -> (d0 + s0)>
%j = affine.apply #map(%i)[%offset]
```

Use `affine.apply` when an index expression needs to be reused or passed to
another operation.

`affine.min`
: Computes the minimum of all results of an affine map.

```mlir
#ub = affine_map<(d0) -> (d0, 64)>
%limit = affine.min #ub(%n)
```

This is useful for upper bounds, edge tiles, and clamping computations.

`affine.max`
: Computes the maximum of all results of an affine map.

```mlir
#lb = affine_map<(d0) -> (0, d0 - 8)>
%start = affine.max #lb(%n)
```

This is useful for lower bounds and guarded slices.

`affine.linearize_index`
: Converts a multi-dimensional index into one linear index using a basis.

```mlir
%linear = affine.linearize_index [%i, %j] by (4, 8) : index
```

With a basis like `(4, 8)`, the first basis element is an outer bound. It can
help analysis but is not used in the arithmetic. The operation can also carry
the `disjoint` property, which asserts that the component indices fit within
their basis ranges.

`affine.delinearize_index`
: Converts one linear index into multiple indices using a basis.

```mlir
%i, %j = affine.delinearize_index %linear into (4, 8) : index, index
```

The basis elements must be strictly positive. Dynamic basis values that are
zero or negative cause undefined behavior. Lowerings may also assume the index
arithmetic does not overflow in the signed sense.

### Control-Flow Operations

`affine.for`
: A counted loop with affine lower and upper bounds and a positive constant
  step.

```mlir
affine.for %i = 0 to 64 {
  ...
}
```

`affine.for` can carry loop-carried values through `iter_args`, similar to
`scf.for`. If the loop produces results, the body must end with
`affine.yield`.

`affine.if`
: A structured conditional controlled by an affine integer set.

```mlir
#set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>
affine.if #set(%i)[%n] {
  ...
}
```

Like `scf.if`, it can produce values from `then` and `else` regions.

`affine.parallel`
: A parallel loop band with affine bounds.

```mlir
affine.parallel (%i) = (0) to (16) {
  ...
}
```

It can also express reductions with reduction kinds such as `"addf"`. The
execution order is unspecified, so the body must be valid for parallel
execution.

`affine.yield`
: Terminates the region of `affine.for`, `affine.if`, or `affine.parallel`.
  It yields zero or more values to the parent operation.

When the parent operation has no results, custom syntax can omit the
`affine.yield`.

### Memory Operations

`affine.load`
: Loads one element from a memref using affine indices.

```mlir
%v = affine.load %A[%i, %j + 1] : memref<64x64xf32>
```

The result type is the memref element type.

`affine.store`
: Stores one element to a memref using affine indices.

```mlir
affine.store %v, %A[%i, %j + 1] : memref<64x64xf32>
```

The stored value type must match the memref element type.

`affine.prefetch`
: Describes a prefetch of a memref element or region using affine indices.
  It carries whether the prefetch is for a read or write, a locality hint, and
  whether the target is the data cache or instruction cache.

Use it when a pipeline wants to preserve prefetch intent before lowering to
target-specific code.

`affine.vector_load`
: Loads a contiguous slice from a memref into a non-zero-rank vector using
  affine indices.

```mlir
%v = affine.vector_load %A[%i] : memref<64xf32>, vector<4xf32>
```

The memref element type and vector element type must match.

`affine.vector_store`
: Stores a non-zero-rank vector into a contiguous memref slice using affine
  indices.

```mlir
affine.vector_store %v, %A[%i] : memref<64xf32>, vector<4xf32>
```

These vector operations are still affine memory accesses. The vector type
describes the slice shape.

### DMA Operations

`affine.dma_start`
: Starts a non-blocking DMA transfer from a source memref to a destination
  memref. Source, destination, and tag indices all use affine maps.

The operation records:

- Source memref and source indices.
- Destination memref and destination indices.
- Tag memref and tag indices.
- Number of elements.
- Optional stride and elements-per-stride values.

`affine.dma_wait`
: Waits for the DMA transfer associated with a tag memref element to complete.

The DMA operations are useful for explicitly managed memory hierarchies, such
as moving data between slower and faster memory spaces. They are also used by
Affine data-copy and pipelining transformations.

## Transformations

### Canonicalization And Simplification

Most Affine ops have verifiers, folders, or canonicalization patterns.

Useful simplification passes include:

- `affine-simplify-structures`: simplifies affine expressions in maps and sets
  and normalizes memrefs.
- `affine-simplify-min-max`: simplifies `affine.min`, `affine.max`, and
  `affine.apply` using canonicalization and value-bounds reasoning.
- `affine-simplify-with-bounds`: simplifies `affine.delinearize_index`,
  `affine.linearize_index`, and `affine.for` using value bounds analysis.
- `affine-expand-index-ops`: lowers affine index operations such as
  `affine.linearize_index` and `affine.delinearize_index` into more
  fundamental operations.
- `affine-expand-index-ops-as-affine`: lowers those index operations into
  `affine.apply` operations when affine form should be preserved longer.

Use these passes after transformations introduce complex maps, min/max
expressions, or linearized index calculations.

### Loop Transformations

The Affine dialect has a large loop-optimization surface:

- `affine-loop-fusion`: fuses affine loop nests, including producer-consumer
  and sibling fusion cases.
- `affine-loop-tile`: tiles affine loop nests.
- `affine-loop-unroll`: unrolls affine loops.
- `affine-loop-unroll-jam`: unrolls an outer loop and jams the copies of the
  inner loop body together.
- `affine-loop-invariant-code-motion`: hoists loop-invariant operations out of
  affine loops.
- `affine-loop-normalize`: normalizes affine loop-like ops, including promotion
  of single-iteration loops when enabled.
- `affine-loop-coalescing`: coalesces nested loops with independent bounds into
  one loop.

These are the main reason to use Affine in the first place. They are much more
powerful when the compiler can see exact loop bounds and memory subscripts.

### Memory And Locality Transformations

Affine also has memory-specific transformations:

- `affine-data-copy-generate`: creates explicit copies for affine memory
  operations, optionally using DMA and faster memory spaces.
- `affine-scalrep`: performs scalar replacement by forwarding stores to loads
  and eliminating redundant affine loads.
- `affine-raise-from-memref`: turns supported `memref.load` and
  `memref.store` operations into `affine.load` and `affine.store`.
- `affine-fold-memref-alias-ops`: folds `memref.subview`,
  `memref.expand_shape`, and `memref.collapse_shape` into affine loads and
  stores when possible.
- `affine-pipeline-data-transfer`: overlaps non-blocking DMA transfers with
  computation through double buffering.

These passes are especially useful when a compiler is trying to reduce memory
traffic, improve locality, or target a memory hierarchy with explicit fast and
slow spaces.

### Parallelization And Vectorization

Affine can expose parallelism and vector structure:

- `affine-parallelize`: converts suitable `affine.for` loops into
  one-dimensional `affine.parallel` operations.
- `affine-super-vectorize`: vectorizes affine loops to target-independent
  `vector` operations.

`affine.parallel` still belongs to the Affine dialect, so it can remain
analyzable before later lowering to `scf.parallel` or GPU-oriented forms.

### Transform Dialect Controls

Affine also exposes some transformations through the Transform dialect:

- `transform.affine.simplify_bounded_affine_ops`
- `transform.affine.simplify_min_max_affine_ops`
- `transform.affine.super_vectorize`

These are not Affine dialect operations. They are Transform dialect operations
that target Affine operations from a transform script.

Use them when a pipeline wants to control Affine simplification or
vectorization declaratively through Transform IR.

## Conversions And Lowering Paths

### `lower-affine`

The main lowering pass is `lower-affine`.

It lowers Affine operations to a combination of:

- `arith`
- `scf`
- `memref`
- `vector`

Important rewrites include:

- `affine.for` to `scf.for`
- `affine.if` to `scf.if`
- `affine.parallel` to `scf.parallel`
- `affine.yield` to `scf.yield`
- `affine.apply`, `affine.min`, and `affine.max` to primitive `arith`
  operations on `index`
- `affine.load` to `memref.load`
- `affine.store` to `memref.store`
- `affine.prefetch` to `memref.prefetch`
- `affine.dma_start` to `memref.dma_start`
- `affine.dma_wait` to `memref.dma_wait`
- `affine.vector_load` to `vector.load`
- `affine.vector_store` to `vector.store`
- `affine.linearize_index` and `affine.delinearize_index` to lower-level
  index arithmetic

This is usually the point where Affine's high-level restrictions are no longer
needed.

### Affine To GPU

`convert-affine-for-to-gpu` converts top-level `affine.for` operations to GPU
kernels.

The pass has options for the number of GPU block dimensions and thread
dimensions:

- `gpu-block-dims`
- `gpu-thread-dims`

Use this when affine loop nests are simple enough to map directly to GPU launch
structure. More complex GPU pipelines may first go through `scf`, `gpu`,
`linalg`, or `vector` transformations.

### Raising From MemRef

`affine-raise-from-memref` goes in the opposite direction from lowering.

It tries to turn supported:

```mlir
memref.load
memref.store
```

into:

```mlir
affine.load
affine.store
```

This is useful when earlier passes produced ordinary memref operations but the
indices are still affine and later Affine passes can optimize them.

## Example IR

### Shifted Copy

This example uses `affine.for`, `affine.apply`, `affine.load`, and
`affine.store`.

```mlir
#shift = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @shifted_copy(%src: memref<128xf32>,
                        %dst: memref<128xf32>,
                        %offset: index) {
  affine.for %i = 0 to 64 {
    %j = affine.apply #shift(%i)[%offset]
    %v = affine.load %src[%j] : memref<128xf32>
    affine.store %v, %dst[%i] : memref<128xf32>
  }
  return
}
```

The loop is analyzable because the bounds are constants and the memory
subscripts are affine expressions of `%i` and `%offset`.

### Guarded Read

This example uses `affine.if` with an integer set.

```mlir
#set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>

func.func @guarded_read(%A: memref<?xf32>, %i: index, %n: index) -> f32 {
  %zero = arith.constant 0.0 : f32
  %v = affine.if #set(%i)[%n] -> f32 {
    %x = affine.load %A[%i] : memref<?xf32>
    affine.yield %x : f32
  } else {
    affine.yield %zero : f32
  }
  return %v : f32
}
```

The integer set means:

```text
0 <= i && i <= n - 1
```

Because that condition is affine, MLIR can reason about it together with other
affine constraints.

### Parallel Reduction

This example uses `affine.parallel` and `affine.yield`.

```mlir
func.func @sum16(%A: memref<16xf32>) -> f32 {
  %sum = affine.parallel (%i) = (0) to (16) reduce ("addf") -> f32 {
    %v = affine.load %A[%i] : memref<16xf32>
    affine.yield %v : f32
  }
  return %sum : f32
}
```

The operation says the iterations can run in parallel and their yielded values
are combined with a floating-point add reduction.

### Linearize And Delinearize

```mlir
func.func @linearize(%i: index, %j: index) -> (index, index, index) {
  %lin = affine.linearize_index [%i, %j] by (4, 8) : index
  %d0, %d1 = affine.delinearize_index %lin into (4, 8) : index, index
  return %lin, %d0, %d1 : index, index, index
}
```

Linearization converts a multi-index into one index. Delinearization converts
one index back into coordinates. These operations are common around flattening,
tiling, and mapping multi-dimensional loops to one-dimensional work.

### Vector Access

```mlir
func.func @vector_access(%A: memref<64xf32>,
                         %v: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %loaded = affine.vector_load %A[%c0] : memref<64xf32>, vector<4xf32>
  affine.vector_store %v, %A[%c0] : memref<64xf32>, vector<4xf32>
  return %loaded : vector<4xf32>
}
```

The access starts at an affine index, while the vector type describes the shape
of the slice.

## Mental Model

Treat the `affine` dialect as a contract with the optimizer.

When you write:

```mlir
affine.for %i = 0 to 64 {
  %v = affine.load %A[%i] : memref<64xf32>
}
```

you are not just saying "run a loop and load memory." You are also saying:

```text
the loop range and memory access are structured enough for MLIR to analyze
```

That contract enables stronger transformations.

The tradeoff is that Affine is less general than `scf`. If the compiler cannot
prove that a value is a valid dimension or symbol, or if the expression is not
affine, the IR should use a more general dialect.

Good beginner rule:

```text
Use affine when the loop math is simple and important.
Use scf when the control flow is general.
```

## Gotchas

- Affine expressions are not arbitrary arithmetic. Multiplication by a runtime
  value is not affine.
- Dimension and symbol placement matters. A value that is valid as a symbol in
  one location may not be valid as a symbol somewhere else.
- `affine.for` has a positive constant step.
- `affine.for` ranges are half-open. The upper bound is not included.
- Multi-result lower-bound maps are interpreted as `max`; multi-result
  upper-bound maps are interpreted as `min`.
- `affine.load` and `affine.store` require affine subscripts. If indexing is
  indirect or data-dependent, use `memref.load` and `memref.store`.
- `affine.parallel` has unspecified iteration order. Only use it when the body
  is safe for parallel execution or expresses reductions correctly.
- Dynamic basis elements for `affine.delinearize_index` must be strictly
  positive.
- `affine.linearize_index` and `affine.delinearize_index` lowerings may assume
  signed `index` arithmetic does not overflow.
- Affine is usually not a final backend dialect. Most pipelines eventually run
  `lower-affine`.

## Source Map

Primary definitions:

- `mlir/include/mlir/Dialect/Affine/IR/AffineOps.td`
- `mlir/include/mlir/Dialect/Affine/IR/AffineOps.h`
- `mlir/lib/Dialect/Affine/IR/AffineOps.cpp`
- `mlir/include/mlir/Dialect/Affine/Transforms/Passes.td`
- `mlir/lib/Dialect/Affine/Transforms/`
- `mlir/include/mlir/Dialect/Affine/TransformOps/AffineTransformOps.td`
- `mlir/lib/Conversion/AffineToStandard/AffineToStandard.cpp`
- `mlir/include/mlir/Conversion/Passes.td`

All Affine dialect operations covered in this chapter:

- `affine.apply`
- `affine.delinearize_index`
- `affine.dma_start`
- `affine.dma_wait`
- `affine.for`
- `affine.if`
- `affine.linearize_index`
- `affine.load`
- `affine.max`
- `affine.min`
- `affine.parallel`
- `affine.prefetch`
- `affine.store`
- `affine.vector_load`
- `affine.vector_store`
- `affine.yield`

Affine-specific passes covered in this chapter:

- `affine-data-copy-generate`
- `affine-loop-fusion`
- `affine-loop-invariant-code-motion`
- `affine-loop-tile`
- `affine-loop-unroll`
- `affine-loop-unroll-jam`
- `affine-pipeline-data-transfer`
- `affine-scalrep`
- `affine-super-vectorize`
- `affine-parallelize`
- `affine-loop-normalize`
- `affine-loop-coalescing`
- `affine-raise-from-memref`
- `affine-simplify-structures`
- `affine-simplify-min-max`
- `affine-simplify-with-bounds`
- `affine-expand-index-ops`
- `affine-expand-index-ops-as-affine`
- `affine-fold-memref-alias-ops`

Conversion passes and paths covered in this chapter:

- `lower-affine`
- `convert-affine-for-to-gpu`
- Affine to `arith`
- Affine to `scf`
- Affine to `memref`
- Affine to `vector`
- Raising supported `memref.load` and `memref.store` operations back to Affine
  with `affine-raise-from-memref`
