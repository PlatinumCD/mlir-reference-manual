# Vector Dialect

## Beginner Summary

The `vector` dialect is MLIR's target-independent dialect for SIMD-style and
SIMT-style vector computation.

It gives the compiler a way to describe:

- Fixed-width vector values such as `vector<4xf32>`.
- Multi-dimensional vector values such as `vector<4x8xf32>`.
- Scalable vector values whose runtime length depends on the target.
- Vector reads and writes from memory or tensors.
- Vector masks.
- Vector reductions, contractions, transposes, shuffles, and scans.
- Late lowering decisions before LLVM, GPU, SPIR-V, Arm SME, XeGPU, or AMX.

For beginners, `vector` is the dialect that appears after loops or tensor
programs have been vectorized, but before the compiler has committed to one
exact machine instruction sequence.

## Why This Dialect Exists

Many programs contain regular computation over adjacent values:

```text
load several values
compute on them together
store several values
```

CPUs, GPUs, and accelerators all have ways to execute that kind of work, but
their instruction sets are different. MLIR therefore needs an intermediate
level that is more explicit than scalar loops but less target-specific than
LLVM intrinsics, GPU MMA operations, Arm SME tiles, or AMX tiles.

The `vector` dialect fills that level.

It lets earlier passes express useful structure:

- This operation is a vector transfer from memory.
- This operation is a matrix-like contraction.
- This reduction happens across lanes.
- This access is masked.
- This vector shape can be lowered in several different ways.

Later passes choose how to lower that structure. The same `vector.contract` may
be lowered to dot products, outer products, LLVM matrix intrinsics, GPU MMA
operations, Arm SME operations, AMX operations, or scalar/vector arithmetic,
depending on the target and pass pipeline.

## When It Matters

The `vector` dialect matters when a compiler is moving from high-level
structured computation toward code generation.

You usually see it after:

- `linalg` operations have been tiled and vectorized.
- Tensor operations have been bufferized or are being lowered through
  `vector.transfer_read` and `vector.transfer_write`.
- Loops contain small fixed-size chunks of work.
- A GPU or CPU pipeline is preparing matmul-like operations for target-specific
  instructions.
- A backend needs masks for boundary tiles or predicated execution.

You usually see it before:

- `convert-vector-to-scf`
- `convert-vector-to-llvm`
- `convert-vector-to-gpu`
- `convert-vector-to-arm-sme`
- `convert-vector-to-spirv`
- `convert-vector-to-xegpu`
- `convert-vector-to-amx`

A common beginner pipeline shape is:

```text
linalg / scf / tensor / memref
  -> vectorize loops or structured ops
  -> simplify vector shape, transfer, mask, contraction, and reduction forms
  -> lower high-level vector ops to lower-level vector ops or loops
  -> convert vector ops to LLVM, GPU, SPIR-V, Arm SME, XeGPU, or AMX
```

## When To Use It

Use `vector` when the IR should expose lane-level computation while still
remaining target-independent.

Good uses include:

- Representing SIMD arithmetic before choosing exact LLVM operations.
- Representing multi-dimensional tile fragments from `linalg`.
- Representing full and partial vector transfers at tile boundaries.
- Keeping masks explicit while deciding whether to use predication, scalar
  control flow, or masked target operations.
- Modeling contractions before selecting dot-product, outer-product, MMA, SME,
  or AMX forms.
- Expressing shuffles, transposes, interleaves, scans, and reductions in a form
  rewrite patterns can still understand.

Avoid using `vector` as a replacement for all loops. It is best when the
compiler already knows the chunk size and wants vector-level semantics. If the
program still needs arbitrary iteration, keep loops in `scf`, `affine`, `gpu`,
or another control-flow dialect and use `vector` inside those loops.

## Core Concepts

### Vector Types

MLIR vector types describe shaped SSA values:

```mlir
vector<4xf32>
vector<4x8xf32>
vector<[4]xf32>
vector<2x[4]xf32>
```

The first two are fixed-size vectors. The forms with brackets contain scalable
dimensions. A scalable dimension has a minimum size known in IR, multiplied by a
runtime scale factor known to the target.

The `vector` dialect can represent more than one dimension. That is important
because MLIR often vectorizes structured operations such as matrix multiplication
before flattening them to the one-dimensional vectors expected by many targets.

### Values, Not Memory

Most `vector` operations manipulate SSA vector values. For example,
`vector.broadcast`, `vector.shuffle`, `vector.transpose`, and
`vector.reduction` do not access memory.

Memory access is explicit:

- `vector.load` and `vector.store` are simple contiguous vector accesses.
- `vector.transfer_read` and `vector.transfer_write` are higher-level transfers
  that can describe permutation maps, padding, tensor operands, and partial
  boundary behavior.
- `vector.gather` and `vector.scatter` use index vectors.
- `vector.maskedload`, `vector.maskedstore`, `vector.expandload`, and
  `vector.compressstore` model masked memory forms.

### Transfers

`vector.transfer_read` and `vector.transfer_write` are some of the most
important operations in this dialect.

They describe moving a shaped vector between an SSA value and a slice of a
memref or ranked tensor. They can represent out-of-bounds behavior through
padding or masks, and they preserve layout information long enough for later
passes to choose a lowering strategy.

For beginners, think of a transfer as:

```text
read or write a rectangular vector tile
starting at these indices
with this boundary behavior
```

### Masks

Vector masks are vectors of `i1`.

The dialect has two styles:

- Mask values built by `vector.create_mask` and `vector.constant_mask`.
- Region-based masking through `vector.mask`.

`vector.mask` wraps one maskable operation in a region. The mask controls which
lanes are active. A passthrough value may provide results for inactive lanes
when the masked operation supports it.

### Contractions And Reductions

The dialect distinguishes several levels of reduction-like work:

- `vector.reduction` reduces a one-dimensional vector to a scalar.
- `vector.multi_reduction` reduces selected dimensions of a multi-dimensional
  vector.
- `vector.contract` represents generalized multiply-and-accumulate behavior
  with explicit indexing maps and iterator types.
- `vector.outerproduct` is a simpler vector outer-product operation.
- `vector.scan` computes prefix-style results along one dimension.

This distinction lets a pipeline preserve high-level structure and then choose a
lowering form late.

## Operations

The current Vector dialect in this LLVM checkout defines 39 generated
operations.

### Shape, Construction, And Casting

`vector.broadcast`
: Broadcasts a scalar or lower-rank vector to a larger vector shape.

`vector.bitcast`
: Reinterprets vector element type and minor vector size while preserving the
  total bitwidth constraints required by the op.

`vector.shape_cast`
: Changes vector shape without changing the number of elements, element type, or
  scalable dimensions.

`vector.from_elements`
: Builds a vector from scalar elements in row-major order.

`vector.to_elements`
: Decomposes a vector into scalar results in row-major order.

`vector.vscale`
: Produces the runtime vector scale used for scalable vector lengths.

`vector.step`
: Produces a one-dimensional index vector containing `0, 1, ..., N - 1`.

`vector.scalable.extract`
: Extracts a fixed or scalable subvector from a rank-1 scalable vector.

`vector.scalable.insert`
: Inserts a fixed or scalable subvector into a rank-1 scalable vector.

### Element Movement Within Vectors

`vector.extract`
: Extracts a scalar or subvector from a vector at static and/or dynamic
  positions.

`vector.insert`
: Inserts a scalar or subvector into a vector at static and/or dynamic
  positions.

`vector.extract_strided_slice`
: Extracts a strided subvector using static offsets, sizes, and strides.

`vector.insert_strided_slice`
: Inserts a strided subvector into a larger vector.

`vector.shuffle`
: Forms a one-dimensional vector by selecting elements from one or two input
  vectors according to a mask.

`vector.transpose`
: Permutes the dimensions of an n-D vector.

`vector.interleave`
: Interleaves elements from the trailing dimension of two vectors.

`vector.deinterleave`
: Splits even and odd elements from the trailing dimension into two result
  vectors.

### Vector Computation

`vector.fma`
: Computes fused vector multiply-add: `lhs * rhs + acc`.

`vector.reduction`
: Horizontally reduces a one-dimensional vector to a scalar, with an optional
  accumulator.

`vector.multi_reduction`
: Reduces selected dimensions of an n-D vector to a lower-rank vector or scalar.

`vector.contract`
: Represents generalized contraction, commonly used for matrix-like multiply
  and accumulate.

`vector.outerproduct`
: Builds a two-dimensional outer product from two one-dimensional vectors, with
  optional fused accumulation.

`vector.scan`
: Computes inclusive or exclusive scan results along one dimension.

### Masks

`vector.create_mask`
: Builds a runtime mask from dynamic dimension sizes.

`vector.constant_mask`
: Builds a compile-time rectangular mask from static dimension sizes.

`vector.mask`
: Applies an `i1` vector mask to one maskable operation inside a region.

`vector.yield`
: Terminates regions owned by Vector dialect operations, especially
  `vector.mask`.

### Memory And Tensor Access

`vector.load`
: Reads an n-D slice of a memref into an n-D vector.

`vector.store`
: Writes an n-D vector to an n-D slice of a memref.

`vector.transfer_read`
: Reads a vector tile from a memref or ranked tensor, with padding, maps, masks,
  and boundary semantics.

`vector.transfer_write`
: Writes a vector tile to a memref or ranked tensor, with maps, masks, and
  boundary semantics.

`vector.maskedload`
: Loads vector elements from memory according to a mask, using a passthrough
  value for inactive lanes.

`vector.maskedstore`
: Stores vector elements to memory according to a mask.

`vector.expandload`
: Loads selected memory elements and expands them into vector lanes according to
  a mask.

`vector.compressstore`
: Compresses active vector lanes and stores them contiguously according to a
  mask.

`vector.gather`
: Loads vector lanes from indexed memory or tensor locations according to an
  index vector and mask.

`vector.scatter`
: Stores vector lanes to indexed memory or tensor locations according to an
  index vector and mask.

`vector.type_cast`
: Reinterprets a scalar-element memref as a memref whose element is a vector,
  copying the memref shape into the vector layout.

`vector.print`
: Prints vector values or punctuation for testing and debugging.

## Transformations

The Vector dialect is transformation-heavy. Many operations are deliberately
high-level and are expected to be rewritten before final lowering.

### Native Vector Passes

`lower-vector-mask`
: Lowers `vector.mask` operations, especially masks around side-effecting vector
  operations.

`lower-vector-multi-reduction`
: Lowers `vector.multi_reduction`. It supports `inner-parallel` and
  `inner-reduction` strategies.

`lower-vector-to-from-elements-to-shuffle-tree`
: Lowers `vector.to_elements` and `vector.from_elements` to a tree of
  `vector.shuffle` operations.

### Important Rewrite Pattern Families

The Vector transform library exposes many pattern-population entry points. In a
normal compiler pipeline these appear either through C++ pass construction or
through transform dialect operations.

Important families include:

- Bitcast lowering: `vector.bitcast` to finer vector primitives.
- Broadcast lowering: `vector.broadcast` to inserts, shuffles, or simpler forms.
- Contraction lowering: `vector.contract` to dot products, outer products,
  matrix intrinsics, or elementwise arithmetic.
- Outer-product lowering: `vector.outerproduct` to simpler vector primitives.
- Multi-reduction reorder, flattening, and unrolling.
- Mask materialization: `vector.create_mask` and `vector.constant_mask` to
  arithmetic or comparison operations.
- Masked transfer lowering: masked `vector.transfer_read`,
  `vector.transfer_write`, and `vector.gather` forms.
- Gather lowering: `vector.gather` to conditional loads or lower-level forms.
- Transfer permutation lowering: arbitrary transfer maps to minor-identity
  transfers plus `vector.transpose`.
- Transfer splitting: full-tile and partial-tile paths.
- Transfer-to-SCF: large or high-rank transfers to loops around lower-rank
  transfers.
- Shape-cast and unit-dimension cleanup.
- Transpose lowering: elementwise, LLVM intrinsic, one-dimensional shuffle, or
  16x16 shuffle strategies.
- Interleave/deinterleave lowering or rewriting to `vector.shuffle`.
- Narrow type emulation for targets that need wider legal element types.
- Sink patterns that move vector ops across elementwise or memory operations.
- Linearization and flattening of vector transfers.
- Vector unrolling through `vector.to_elements` and `vector.from_elements`.

### Transform Dialect Hooks

The Vector dialect also registers transform-dialect pattern descriptors. These
do not run by themselves; they are handles that a transform script can use to
populate rewrite or conversion patterns.

Conversion pattern hook:

- `apply_conversion_patterns.vector.vector_to_llvm`

Rewrite pattern hooks:

- `apply_patterns.vector.cast_away_vector_leading_one_dim`
- `apply_patterns.vector.rank_reducing_subview_patterns`
- `apply_patterns.vector.drop_unit_dims_with_shape_cast`
- `apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops`
- `apply_patterns.vector.transfer_permutation_patterns`
- `apply_patterns.vector.lower_bitcast`
- `apply_patterns.vector.lower_broadcast`
- `apply_patterns.vector.lower_contraction`
- `apply_patterns.vector.lower_create_mask`
- `apply_patterns.vector.lower_masks`
- `apply_patterns.vector.lower_masked_transfers`
- `apply_patterns.vector.materialize_masks`
- `apply_patterns.vector.reorder_multi_reduction_dims`
- `apply_patterns.vector.multi_reduction_flattening`
- `apply_patterns.vector.multi_reduction_unrolling`
- `apply_patterns.vector.lower_outerproduct`
- `apply_patterns.vector.lower_gather`
- `apply_patterns.vector.unroll_from_elements`
- `apply_patterns.vector.unroll_to_elements`
- `apply_patterns.vector.lower_scan`
- `apply_patterns.vector.lower_shape_cast`
- `apply_patterns.vector.lower_transfer`
- `apply_patterns.vector.lower_transpose`
- `apply_patterns.vector.lower_interleave`
- `apply_patterns.vector.interleave_and_deinterleave_to_shuffle`
- `apply_patterns.vector.rewrite_narrow_types`
- `apply_patterns.vector.split_transfer_full_partial`
- `apply_patterns.vector.transfer_to_scf`
- `apply_patterns.vector.fold_arith_extension`
- `apply_patterns.vector.elementwise_to_vector`
- `apply_patterns.vector.reduction_to_contract`
- `apply_patterns.vector.sink_ops`
- `apply_patterns.vector.sink_mem_ops`
- `apply_patterns.vector.flatten_vector_transfer_ops`

The beginner takeaway is that Vector lowering is intentionally staged. A pass
pipeline often applies several small vector rewrites before asking a target
conversion pass to finish the job.

## Conversions And Lowering Paths

### To SCF

`convert-vector-to-scf` lowers a subset of vector operations, especially vector
transfers, into `scf` loops and smaller vector operations.

Important options:

- `full-unroll`
- `target-rank`
- `lower-tensors`
- `lower-scalable`

Use this when high-rank or partially out-of-bounds transfers need to become
explicit control flow before lower-level code generation.

### To LLVM

`convert-vector-to-llvm` lowers supported Vector dialect operations to the LLVM
dialect.

It directly contains conversion patterns for operations such as:

- `vector.reduction`
- `vector.create_mask`
- `vector.load`
- `vector.store`
- `vector.maskedload`
- `vector.maskedstore`
- `vector.gather`
- `vector.scatter`
- `vector.bitcast`
- `vector.shuffle`
- `vector.extract`
- `vector.insert`
- `vector.fma`
- `vector.print`
- `vector.type_cast`
- `vector.vscale`
- `vector.expandload`
- `vector.compressstore`
- `vector.broadcast`
- `vector.scalable.insert`
- `vector.scalable.extract`
- `vector.mask`
- `vector.interleave`
- `vector.deinterleave`
- `vector.from_elements`
- `vector.to_elements`
- `vector.step`

Important options:

- `reassociate-fp-reductions`
- `force-32bit-vector-indices`
- `use-vector-alignment`
- `enable-gep-inbounds-nuw`
- `enable-arm-neon`
- `enable-arm-sve`
- `enable-arm-i8mm`
- `enable-arm-bf16`
- `enable-x86`
- `vector-contract-lowering`
- `vector-transpose-lowering`

`convert-to-llvm` can also use the Vector dialect's conversion interface when
the dialect is loaded with the generic conversion driver.

### To GPU And NVGPU

`convert-vector-to-gpu` lowers selected vector operations toward GPU dialect
matrix forms. With `use-nvgpu`, it can choose NVGPU operations instead of only
generic GPU operations.

This path is mainly about recognizing vector contractions, transfer reads,
transfer writes, broadcasts, transposes, and slices that fit GPU MMA-style
execution.

### To Arm SME

`convert-vector-to-arm-sme` lowers supported Vector operations to Arm SME and
Arm SVE dialect operations.

The implementation includes patterns for operations such as transfer reads,
transfer writes, loads, stores, broadcasts, transposes, outer products,
extracts, inserts, masks, and scalable-vector-related forms when they match the
Arm SME model.

### To SPIR-V

`convert-vector-to-spirv` converts supported Vector operations to the SPIR-V
dialect. SPIR-V has stricter type and capability rules than generic MLIR
vectors, so pipelines usually simplify vector IR before this pass.

The Vector-to-SPIR-V library also provides patterns for vector reductions that
can map to SPIR-V dot-product forms.

### To XeGPU

`convert-vector-to-xegpu` lowers selected Vector operations into the XeGPU
dialect for Intel GPU-oriented pipelines.

Use it when the pipeline wants to preserve GPU-tile semantics in XeGPU before
continuing to XeVM or LLVM.

### To AMX

`convert-vector-to-amx` lowers selected Vector operations to X86 AMX dialect
operations.

This path is mainly useful for tile-shaped contractions and transfers that match
AMX-style matrix computation.

### Lowering Strategy

A practical lowering strategy is:

```text
1. Keep high-level vector ops while optimizing structured computation.
2. Normalize transfers, masks, transposes, reductions, and contractions.
3. Lower high-rank or high-level vector forms to smaller vector forms or SCF.
4. Use the target conversion pass that matches the backend.
```

Trying to run a target conversion too early often leaves illegal Vector ops in
the IR. That usually means the pipeline still needs a Vector rewrite pass or a
target-specific preparation step.

## Example IR

### Basic Vector Computation

```mlir
func.func @shape_ops(%x: f32, %y: vector<4xf32>) -> vector<4xf32> {
  %b = vector.broadcast %x : f32 to vector<4xf32>
  %z = vector.fma %b, %y, %y : vector<4xf32>
  %r = vector.reduction <add>, %z : vector<4xf32> into f32
  %out = vector.broadcast %r : f32 to vector<4xf32>
  func.return %out : vector<4xf32>
}
```

This example starts with one scalar, broadcasts it to four lanes, computes a
fused multiply-add, reduces the lanes, and broadcasts the scalar result back to
a vector.

### Transfer Read And Transfer Write

```mlir
func.func @memory_ops(
    %A: memref<?x?xf32>,
    %i: index,
    %j: index,
    %v: vector<4x8xf32>) -> vector<4x8xf32> {
  %pad = arith.constant 0.0 : f32
  %read = vector.transfer_read %A[%i, %j], %pad
      {in_bounds = [true, true]} : memref<?x?xf32>, vector<4x8xf32>
  vector.transfer_write %v, %A[%i, %j]
      {in_bounds = [true, true]} : vector<4x8xf32>, memref<?x?xf32>
  func.return %read : vector<4x8xf32>
}
```

The transfer operations describe a two-dimensional vector tile moving between a
memref and an SSA value.

### Masked Transfer

```mlir
func.func @mask_ops(
    %A: memref<?xf32>,
    %i: index,
    %n: index) -> vector<4xf32> {
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %n : vector<4xi1>
  %read = vector.mask %mask {
    vector.transfer_read %A[%i], %pad : memref<?xf32>, vector<4xf32>
  } : vector<4xi1> -> vector<4xf32>
  func.return %read : vector<4xf32>
}
```

This example models a vector read where only the first `%n` lanes are active.

### Multi-Dimensional Reduction

```mlir
func.func @multi_reduce(%v: vector<4x8xf32>) -> vector<4xf32> {
  %zero = arith.constant dense<0.0> : vector<4xf32>
  %r = vector.multi_reduction <add>, %v, %zero [1]
      : vector<4x8xf32> to vector<4xf32>
  func.return %r : vector<4xf32>
}
```

This reduces dimension 1 of a `4x8` vector, producing one result per row.

## Mental Model

Think of the `vector` dialect as a box of partially lowered computation.

It is no longer just loops or tensors. The compiler has already decided to work
on chunks of multiple elements at once.

It is also not final machine IR. The compiler has not yet decided whether a
chunk becomes:

- LLVM vector operations.
- Scalar loops.
- GPU MMA operations.
- NVGPU operations.
- Arm SME tiles.
- SPIR-V vector operations.
- XeGPU operations.
- X86 AMX operations.

The dialect keeps enough structure for good late decisions.

For a beginner, the main question is:

```text
Is this vector op still describing useful structure,
or is it ready to be lowered to the target?
```

If it still describes structure, use Vector rewrite passes. If it is ready for
the target, use one of the conversion passes.

## Gotchas

`vector.transfer_read` is not the same as `vector.load`.

`vector.load` is a simpler memref vector load. `vector.transfer_read` is a
higher-level transfer that can carry padding, permutation maps, tensor access,
in-bounds information, and boundary semantics.

`vector.store` is not the same as `vector.transfer_write`.

The same distinction applies on the write side. Transfer writes are usually
better while a compiler is still preserving tile structure.

`vector.contract` is not always a hardware matrix instruction.

It is a semantic contraction. The target pipeline decides whether it becomes
dot products, outer products, LLVM matrix intrinsics, GPU MMA, SME, AMX, or
ordinary arithmetic.

Multi-dimensional vectors often need lowering before LLVM conversion.

LLVM itself mainly wants one-dimensional vector types. MLIR's LLVM conversion
has support for representing n-D vectors, but many high-level n-D Vector ops
need rewrite patterns before they become legal or efficient LLVM dialect IR.

Masks are values, not just attributes.

Runtime masks are SSA values such as `vector<4xi1>`. A mask can be built with
`vector.create_mask`, materialized with patterns, or wrapped around a maskable
operation with `vector.mask`.

Scalable vectors need target awareness.

`vector.vscale`, `vector.step`, `vector.scalable.extract`, and
`vector.scalable.insert` are useful for scalable-vector targets, but they also
make legality more target-dependent.

Target conversion passes are partial.

`convert-vector-to-gpu`, `convert-vector-to-arm-sme`, `convert-vector-to-xegpu`,
and `convert-vector-to-amx` are not general "lower every vector op" passes. They
look for patterns that match target-specific capabilities. Unsupported Vector
ops need earlier canonicalization, lowering, or a different path.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/Vector/IR/Vector.td`
- `mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`
- `mlir/lib/Dialect/Vector/IR/VectorOps.cpp`
- `mlir/include/mlir/Dialect/Vector/Interfaces/MaskableOpInterface.td`
- `mlir/include/mlir/Dialect/Vector/Interfaces/MaskingOpInterface.td`
- `mlir/include/mlir/Dialect/Vector/Transforms/Passes.td`
- `mlir/include/mlir/Dialect/Vector/Transforms/VectorTransformsBase.td`
- `mlir/include/mlir/Dialect/Vector/TransformOps/VectorTransformOps.td`
- `mlir/lib/Dialect/Vector/Transforms/`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp`
- `mlir/lib/Conversion/VectorToSCF/VectorToSCF.cpp`
- `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp`
- `mlir/lib/Conversion/VectorToArmSME/VectorToArmSME.cpp`
- `mlir/lib/Conversion/VectorToSPIRV/VectorToSPIRV.cpp`
- `mlir/lib/Conversion/VectorToXeGPU/VectorToXeGPU.cpp`
- `mlir/lib/Conversion/VectorToAMX/VectorToAMX.cpp`

Generated op documentation source:

```text
mlir-tblgen --gen-op-doc -dialect=vector \
  mlir/include/mlir/Dialect/Vector/IR/VectorOps.td
```
