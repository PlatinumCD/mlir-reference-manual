# Tensor Dialect

## Beginner Summary

The `tensor` dialect contains core operations for creating, reshaping,
querying, slicing, and assembling tensor values.

The tensor type itself is a builtin MLIR type, not a type owned by the
`tensor` dialect. The dialect provides operations that work on those tensor
values.

The key beginner idea is that tensors are value-like and immutable. A
`tensor.insert` or `tensor.insert_slice` does not mutate an existing tensor in
place at the IR level. It returns a new tensor value that represents the update.
Later bufferization may prove that an update can be implemented in-place, but
that is an optimization and lowering decision.

Use the `tensor` dialect when you need to describe tensor shape and tensor
value manipulation without committing to a particular memory layout or buffer.

## Why This Dialect Exists

Many MLIR programs need tensors before they need buffers.

At a high level, tensors are useful because they represent shaped values without
forcing the compiler to decide:

- Where the data lives in memory.
- Which allocation owns the data.
- Whether an update can happen in-place.
- How a slice maps to a strided view.
- Which target-specific layout will be used.

The `tensor` dialect exists to hold operations that are broadly meaningful for
many element types and domains:

- Create tensor values.
- Query tensor dimensions and rank.
- Extract or insert scalar elements.
- Extract or insert tensor slices.
- Pad, concatenate, gather, and scatter values.
- Reshape tensors without changing the logical elements.
- Provide destination-style tensor values for later bufferization.

More domain-specific tensor computation belongs in dialects such as `linalg`,
`tosa`, `mhlo`-style dialects, or `sparse_tensor`. The `tensor` dialect is the
small common layer underneath those higher-level tensor programs.

## When It Matters

The `tensor` dialect matters in the middle of many MLIR pipelines.

Typical pipeline shape:

```text
frontend / tosa / linalg / shape-producing IR
  -> tensor values, slices, reshapes, pads, and dimensions
  -> tiling, fusion, canonicalization, destination-style rewrites
  -> bufferization
  -> memref, vector, gpu, spirv, llvm, or other target dialects
```

It is especially important before bufferization. At that stage, the compiler can
still reason about whole tensor values and can rewrite slices, pads, reshapes,
and destinations without committing to physical buffers.

## When To Use It

Use `tensor` when your IR needs shaped values and pure tensor manipulation.

Use it for:

- Creating a tensor with known or dynamic shape.
- Asking for a dynamic dimension or rank.
- Building a tensor from scalar values.
- Broadcasting a scalar into a tensor.
- Extracting or inserting one tensor element.
- Extracting or inserting a slice.
- Padding a tensor.
- Concatenating tensors.
- Representing gather and scatter at tensor level.
- Expanding, collapsing, reshaping, casting, or bitcasting tensor values.
- Supplying destination tensors to destination-style operations.

Do not use it for:

- Explicit memory allocation and deallocation. Use bufferization and `memref`
  after tensor lowering.
- Rich numerical algorithms by itself. Use `linalg`, `tosa`, or another compute
  dialect to describe the algorithm.
- Sparse storage behavior. Use `sparse_tensor` when the storage format matters.

## Core Concepts

### Tensor Values Are Immutable

At the tensor IR level, an operation that "updates" a tensor returns a new
value.

For example, `tensor.insert` returns a tensor value with one element replaced.
It does not mean the original SSA value was mutated. This makes tensor IR easier
to reason about and easier to transform.

Bufferization is where the compiler decides whether the new tensor value can be
implemented by reusing an existing buffer.

### Shape And Data Are Related But Separate

A tensor type may contain static and dynamic dimensions:

```text
tensor<4x?xf32>
```

The `4` is statically known. The `?` is dynamic and must be carried by SSA
values or derived from another tensor. Operations such as `tensor.dim`,
`tensor.empty`, `tensor.generate`, `tensor.splat`, `tensor.pad`, and
`tensor.expand_shape` are often where dynamic sizes become explicit.

### Destination-Style Programming

Some tensor operations use a destination tensor to express the shape and
potential storage of the result. `tensor.insert_slice` is a simple example: it
takes a source slice and a destination tensor, then returns a new destination
value with that slice inserted.

This style is important because bufferization can often map destination-style
IR to efficient in-place writes.

### Offset, Size, And Stride Triples

Slice operations use the same mental model as a strided view:

- Offsets say where the slice starts.
- Sizes say how much to take or insert.
- Strides say how to step through each dimension.

`tensor.extract_slice`, `tensor.insert_slice`, and
`tensor.parallel_insert_slice` all use this idea.

### Regions That Yield Elements

`tensor.generate` and `tensor.pad` contain regions. The region computes an
element value and terminates with `tensor.yield`.

This lets tensor IR describe element-wise construction or padding without
choosing a loop nest or buffer layout yet.

## Operations

### Creation Operations

- `tensor.empty` creates a tensor value of a specified shape with unspecified
  contents. It is commonly used as a destination for destination-style ops.
- `tensor.from_elements` builds a statically shaped tensor from scalar element
  operands.
- `tensor.generate` creates a tensor by running a region for each element.
- `tensor.splat` broadcasts one scalar value to all elements of a tensor.

### Shape And Type Operations

- `tensor.dim` returns the size of one tensor dimension.
- `tensor.rank` returns the rank of a tensor.
- `tensor.cast` changes the amount of static shape information without changing
  element values.
- `tensor.bitcast` changes the element interpretation when element bitwidths
  are compatible.
- `tensor.reshape` reshapes using a runtime shape tensor.
- `tensor.expand_shape` expands rank using reassociation groups.
- `tensor.collapse_shape` collapses rank using reassociation groups.

### Element And Slice Operations

- `tensor.extract` reads one scalar element from a ranked tensor.
- `tensor.insert` returns a new tensor value with one scalar element inserted.
- `tensor.extract_slice` extracts a subtensor using offsets, sizes, and strides.
- `tensor.insert_slice` inserts a subtensor into a destination tensor and
  returns the updated tensor value.
- `tensor.parallel_insert_slice` is used inside a parent parallel combining op
  to describe one thread's slice update.
- `tensor.pad` pads a tensor with a region-computed padding value.
- `tensor.yield` yields a value from `tensor.generate` and `tensor.pad`.

### Combining And Indexed Access Operations

- `tensor.concat` concatenates tensors along a static dimension.
- `tensor.gather` extracts multiple elements or slices from source coordinates.
- `tensor.scatter` inserts multiple elements or slices at destination
  coordinates. It requires the `unique` marker because duplicate coordinates
  would make the result undefined.

These operations are still tensor-level. Lowering and bufferization decide how
to implement them with memory, loops, vector transfers, or target-specific
operations.

## Transformations

Native tensor passes:

- `fold-tensor-subset-ops`: folds tensor subset operations into producer or
  consumer operations. In this tree it handles patterns such as
  `tensor.extract_slice` into `vector.transfer_read` and
  `vector.transfer_write` into `tensor.insert_slice`.
- `scalarize-single-element-tensor-return`: rewrites private functions that
  return a one-element statically shaped ranked tensor so they return the
  element type directly, when all transitive users can be updated safely.

Tensor transform dialect pattern descriptors:

- `transform.apply_patterns.tensor.decompose_concat`: decomposes
  `tensor.concat` into `tensor.insert_slice` chains.
- `transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion`:
  drops redundant rank expansion/reduction patterns around insert and extract
  slices.
- `transform.apply_patterns.tensor.fold_tensor_empty`: folds
  `tensor.extract_slice` and reassociative reshapes into `tensor.empty`.
- `transform.apply_patterns.tensor.fold_tensor_subset_ops`: folds
  `tensor.empty` with `tensor.extract_slice`, `tensor.expand_shape`, and
  `tensor.collapse_shape`.
- `transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers`:
  folds tensor subset ops into vector transfer reads and writes.
- `transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice`:
  merges consecutive extract/insert slice chains.
- `transform.apply_patterns.tensor.reassociative_reshape_folding`: folds
  `tensor.collapse_shape` and `tensor.expand_shape` with inverse rank-changing
  slice patterns.
- `transform.apply_patterns.tensor.bubble_up_extract_slice`: swaps
  `tensor.extract_slice` with producers so the producer can operate on a
  smaller slice.
- `transform.apply_patterns.tensor.rewrite_as_constant`: rewrites tensor ops
  such as `tensor.generate` to `arith.constant` when possible.
- `transform.tensor.make_loop_independent`: rewrites targeted `tensor.empty` or
  `tensor.pad` ops so selected dimensions no longer depend on enclosing
  `scf.for` induction variables, often enabling hoisting.
- `transform.type_conversion.tensor.cast_shape_dynamic_dims`: adds type
  converter materializations for shape-compatible `tensor.cast` operations.

## Conversions And Lowering Paths

Common paths into `tensor`:

- `tosa-to-tensor`: lowers supported TOSA operations to Tensor dialect
  operations.
- Many frontends and high-level dialect conversions directly create tensor
  values, `tensor.empty`, `tensor.extract_slice`, `tensor.insert_slice`,
  `tensor.pad`, and shape queries.

Common paths out of `tensor`:

- `convert-tensor-to-linalg`: converts some Tensor dialect operations to
  Linalg dialect operations.
- `convert-tensor-to-spirv`: converts supported Tensor dialect operations to
  SPIR-V, with options for emulating narrow scalar and unsupported floating
  types.
- Bufferization, especially one-shot bufferization, lowers tensor values and
  destination-style tensor operations toward `bufferization`, `memref`, and
  eventually lower-level target dialects.

Typical relationships:

```text
tensor.empty + destination-style operation
  -> bufferization.alloc_tensor or reusable buffer
  -> memref allocation/view/update

tensor.extract_slice / tensor.insert_slice
  -> bufferization decisions
  -> memref.subview plus loads/stores or vector transfers

tensor.generate / tensor.pad
  -> loops, linalg, or constants when foldable
  -> bufferized element writes

tensor.reshape / tensor.expand_shape / tensor.collapse_shape
  -> shape-only rewrites when possible
  -> memref reshape/view operations after bufferization
```

The implication is that `tensor` is usually not the final target. It is the
value-semantic layer that lets the compiler delay memory-layout decisions.

## Example IR

### Shape Queries And Element Extraction

```mlir
func.func @shape_and_element(%input: tensor<?x4xf32>, %i: index) -> (index, index, f32) {
  %c0 = arith.constant 0 : index
  %dim0 = tensor.dim %input, %c0 : tensor<?x4xf32>
  %rank = tensor.rank %input : tensor<?x4xf32>
  %value = tensor.extract %input[%i, %c0] : tensor<?x4xf32>
  return %dim0, %rank, %value : index, index, f32
}
```

`tensor.dim` asks for a dynamic dimension, `tensor.rank` asks for rank, and
`tensor.extract` reads one element.

### Element Insertion

```mlir
func.func @insert_element(%input: tensor<?x4xf32>, %i: index, %value: f32) -> tensor<?x4xf32> {
  %c0 = arith.constant 0 : index
  %updated = tensor.insert %value into %input[%i, %c0] : tensor<?x4xf32>
  return %updated : tensor<?x4xf32>
}
```

The result is a new tensor value. The original `%input` is not mutated at the
tensor IR level.

### Slice Update

```mlir
func.func @slice_update(%src: tensor<8x16x4xf32>, %dest: tensor<16x32x8xf32>) -> tensor<16x32x8xf32> {
  %slice = tensor.extract_slice %src[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>
  %updated = tensor.insert_slice %slice into %dest[0, 0, 0][4, 4, 4][1, 1, 1]
    : tensor<4x4x4xf32> into tensor<16x32x8xf32>
  return %updated : tensor<16x32x8xf32>
}
```

The offsets, sizes, and strides describe the subset relationship.

### Generating A Tensor

```mlir
func.func @generate(%m: index) -> tensor<?x4xf32> {
  %result = tensor.generate %m {
  ^bb0(%i: index, %j: index):
    %zero = arith.constant 0.0 : f32
    tensor.yield %zero : f32
  } : tensor<?x4xf32>
  return %result : tensor<?x4xf32>
}
```

The region computes each element and yields it with `tensor.yield`.

### Padding

```mlir
func.func @pad(%input: tensor<2x3xf32>) -> tensor<4x5xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %input low[1, 1] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %zero : f32
  } : tensor<2x3xf32> to tensor<4x5xf32>
  return %padded : tensor<4x5xf32>
}
```

The low and high lists say how much padding is added before and after each
dimension.

### Expanding Shape

```mlir
func.func @reshape(%input: tensor<?x?xf32>, %d0: index, %d1: index, %d2: index)
    -> tensor<5x?x?x?xf32> {
  %expanded = tensor.expand_shape %input [[0, 1], [2, 3]] output_shape [5, %d0, %d1, %d2]
    : tensor<?x?xf32> into tensor<5x?x?x?xf32>
  return %expanded : tensor<5x?x?x?xf32>
}
```

Reassociation groups explain how lower-rank dimensions correspond to
higher-rank dimensions.

### Constructing Small Tensors

```mlir
func.func @construct(%x: f32, %y: f32) -> (tensor<2xf32>, tensor<2xf32>) {
  %from = tensor.from_elements %x, %y : tensor<2xf32>
  %splat = tensor.splat %x : tensor<2xf32>
  return %from, %splat : tensor<2xf32>, tensor<2xf32>
}
```

`tensor.from_elements` builds from explicit elements. `tensor.splat` broadcasts
one value.

## Mental Model

The `tensor` dialect means:

```text
The compiler is still reasoning about whole shaped values, not concrete buffers.
```

Read tensor IR as a functional description of shaped data:

- This tensor has this shape.
- This dimension is dynamic.
- This value is a slice of that value.
- This result is that destination with a logical update applied.
- This pad or generate region describes element values.

Later passes decide whether those values become allocations, views, loops,
vector transfers, or target-specific memory operations.

## Gotchas

- The tensor type is builtin; the `tensor` dialect provides operations on it.
- Tensor values are immutable in the IR. Insertions return new values.
- `tensor.empty` has unspecified contents. It is a shape/destination carrier,
  not a zero-initialized tensor.
- `tensor.extract_slice` and `tensor.insert_slice` may rank-reduce or
  rank-expand unit dimensions. This is useful but can be confusing.
- Dynamic dimensions must be supplied where the result type contains `?`.
- `tensor.pad` and `tensor.generate` require `tensor.yield` in their regions.
- `tensor.scatter` requires unique coordinates; incorrect uniqueness is
  undefined behavior.
- Many tensor rewrites are sensitive to bufferization. A rewrite that looks
  algebraically harmless may affect whether later bufferization can be in-place.
- Tensor IR is usually an intermediate layer. Expect it to lower into `linalg`,
  `bufferization`, `memref`, `vector`, `spirv`, or target dialects.

## Source Map

Primary source files in the LLVM tree:

- `mlir/include/mlir/Dialect/Tensor/IR/TensorBase.td`
- `mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td`
- `mlir/include/mlir/Dialect/Tensor/TransformOps/TensorTransformOps.td`
- `mlir/include/mlir/Dialect/Tensor/Transforms/Passes.td`
- `mlir/lib/Dialect/Tensor/IR/`
- `mlir/lib/Dialect/Tensor/Transforms/`
- `mlir/lib/Dialect/Tensor/TransformOps/`
- `mlir/lib/Conversion/TensorToLinalg/`
- `mlir/lib/Conversion/TensorToSPIRV/`

Useful tests:

- `mlir/test/Dialect/Tensor/`
- `mlir/test/Conversion/TensorToLinalg/`
- `mlir/test/Conversion/TensorToSPIRV/`
