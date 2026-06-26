# Sparse Tensor Dialect

## Beginner Summary

The `sparse_tensor` dialect is MLIR's dialect for tensors where most elements are implicit, usually zero, and only selected elements are stored. It lets a compiler keep sparse tensors as normal tensor values for as long as possible, while attaching a sparsity encoding that describes how the tensor should be stored and iterated.

For a beginner, the main idea is:

- A tensor type such as `tensor<100x100xf32, #CSR>` still means a 100 by 100 tensor of `f32` values.
- The `#CSR` sparse encoding says that the tensor is stored with a particular level layout, such as dense rows and compressed columns.
- High-level computations are usually written in `linalg`.
- Sparse lowering uses the encoding to generate loops, storage accesses, conversions, runtime calls, or direct buffer code.

Most users do not write many `sparse_tensor` operations by hand. They describe the sparse format with `#sparse_tensor.encoding`, write tensor or `linalg` code, and let sparse passes generate the lower-level IR.

## Why This Dialect Exists

Dense tensor IR assumes every element is present. Sparse tensor programs need something different: the compiler must understand which coordinates are stored, which coordinates are implicit, how positions and coordinate arrays are arranged, and how computations should co-iterate over several sparse operands.

The `sparse_tensor` dialect exists to make sparsity a first-class property of tensor types instead of an external library detail. It provides:

- Sparse tensor encodings attached to ranked tensor types.
- Storage concepts such as positions, coordinates, values, and storage specifiers.
- High-level sparse operations such as conversion, concatenation, assembly, disassembly, and foreach iteration.
- Operations used inside `linalg.generic` to express sparse set-like behavior.
- Lower-level sparse iteration operations used by experimental or specialized sparse lowering.
- Passes that turn sparse tensor computations into SCF, memref, LLVM, runtime-library calls, vector code, or GPU-oriented code.

Without this dialect, every frontend or optimization would need to invent its own sparse representation, and the compiler would lose the ability to reason across sparse operations.

## When It Matters

The dialect matters when the amount of stored data is much smaller than the logical tensor shape, or when the storage order is important for performance.

Common cases include:

- Sparse vectors and matrices, such as CSR, CSC, COO, DCSR, and block sparse formats.
- Linear algebra kernels where dense iteration would waste work.
- Machine learning workloads with pruned weights, embedding-like accesses, graph adjacency data, or sparse activations.
- Scientific computing kernels that need sparse matrix-vector or sparse matrix-matrix computations.
- Compiler pipelines that must interoperate with external sparse formats from Python, PyTorch, JAX, or a custom runtime.

It also matters in lowering pipelines because sparse tensors cannot be lowered like dense tensors. A dense tensor can become a contiguous buffer. A sparse tensor usually becomes a group of buffers plus metadata, or an opaque runtime object.

## When To Use It

Use `sparse_tensor` when:

- A ranked tensor's storage format is part of the program semantics or performance contract.
- You want `linalg` operations to compile into sparse loops instead of dense loops.
- You need to convert between dense and sparse tensor representations.
- You need explicit access to sparse storage arrays, such as positions, coordinates, or values.
- You are building a pipeline that lowers sparse tensors to runtime-library calls or direct buffer code.
- You need an ABI wrapper for passing sparse tensors into or out of compiled MLIR code.

Do not use it just because a value might contain zeros. Use it when zeros are implicit in the representation or when the compiler should exploit a sparse storage scheme.

## Core Concepts

### Sparse Encoding

The main attribute is `#sparse_tensor.encoding`. It is attached to a ranked tensor type:

```text
tensor<?x?xf64, #CSR>
```

The encoding describes how semantic tensor dimensions map to storage levels. A dimension is an axis in the mathematical tensor. A level is an axis in the physical sparse storage format.

For example, CSR can be described as dense rows and compressed columns:

```text
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>
```

Important level formats include:

- `dense`: every coordinate along this level is stored.
- `batch`: every coordinate is stored, but the level is not linearized in the same way as `dense`.
- `compressed`: only stored coordinates appear, represented with position and coordinate arrays.
- `loose_compressed`: like compressed, but with extra room or non-contiguous intervals.
- `singleton`: one coordinate per parent position, commonly used in COO-like formats.
- `structured[n, m]`: stores structured n-out-of-m sparsity.

Important level properties include:

- `nonunique`: duplicate coordinates may appear.
- `nonordered`: coordinates may appear out of sorted order.
- `soa`: singleton coordinates are stored structure-of-arrays style instead of array-of-structures style.

Optional encoding fields include `posWidth`, `crdWidth`, `explicitVal`, and `implicitVal`. Position and coordinate widths let the storage use narrower integer types. Explicit and implicit values support patterns such as binary-valued sparse tensors.

### Positions, Coordinates, And Values

Sparse storage is usually represented with:

- Position arrays: ranges into a child level.
- Coordinate arrays: explicit level coordinates.
- Value arrays: stored element values.
- Metadata: level sizes, dimension sizes, slice information, and buffer sizes.

`sparse_tensor.positions`, `sparse_tensor.coordinates`, `sparse_tensor.coordinates_buffer`, and `sparse_tensor.values` expose these arrays after sparse lowering has made storage visible.

### Storage Specifiers

`!sparse_tensor.storage_specifier<#encoding>` is a structured metadata type for the chosen sparse layout. It carries information such as dimension sizes, level sizes, position buffer sizes, coordinate buffer sizes, and slice metadata.

The type is intentionally low level. Beginner-facing sparse code usually starts from tensor encodings, not storage specifiers.

### Sparse Iteration

The dialect has two iteration models:

- `sparse_tensor.foreach` iterates over stored elements of a sparse tensor and is easier to read.
- `sparse_tensor.extract_iteration_space`, `sparse_tensor.iterate`, `sparse_tensor.coiterate`, and `sparse_tensor.extract_value` represent sparse iteration spaces and iterators more explicitly.

The main sparsifier can emit either direct SCF-style loops or an intermediate iterator form depending on pass options.

### Sparse Set Operations In Linalg

Inside `linalg.generic`, operations such as `sparse_tensor.binary`, `sparse_tensor.unary`, `sparse_tensor.reduce`, and `sparse_tensor.select` express behavior that depends on whether sparse operands are present or absent at a coordinate. These operations are not general-purpose arithmetic ops. They describe sparse set semantics for the sparsifier.

## Operations

The dialect operation surface falls into several groups.

### Tensor Construction, Conversion, And I/O

- `sparse_tensor.new` creates a sparse tensor from an external source, usually an opaque pointer or reader-like object.
- `sparse_tensor.assemble` builds a sparse tensor from explicit level arrays and a values tensor.
- `sparse_tensor.disassemble` copies a sparse tensor's level arrays and values into caller-provided output tensors.
- `sparse_tensor.convert` converts between dense and sparse tensors or between sparse tensors with different encodings.
- `sparse_tensor.reinterpret_map` reinterprets a sparse tensor using a different dimension/level map when the physical storage can be viewed through another encoding.
- `sparse_tensor.concatenate` concatenates ranked tensors along one dimension, preserving or producing a sparse result when appropriate.
- `sparse_tensor.load` finalizes pending insertions and returns the loaded tensor value. With `hasInserts`, it tells the compiler that insertions occurred before the load.
- `sparse_tensor.out` writes a sparse tensor to an external destination.
- `sparse_tensor.print` prints a sparse tensor, mainly for debugging or test-style usage.
- `sparse_tensor.has_runtime_library` returns whether the sparse runtime library is enabled in the current lowering configuration.

### Storage Access

- `sparse_tensor.positions` extracts the positions array for a level.
- `sparse_tensor.coordinates` extracts the coordinates array for a level.
- `sparse_tensor.coordinates_buffer` extracts a linear coordinates buffer, useful for COO-style layouts.
- `sparse_tensor.values` extracts the stored values array.
- `sparse_tensor.number_of_entries` returns the number of explicitly stored entries.
- `sparse_tensor.slice.offset` reads the offset for a sparse tensor slice dimension.
- `sparse_tensor.slice.stride` reads the stride for a sparse tensor slice dimension.
- `sparse_tensor.storage_specifier.init` creates a storage specifier, optionally from another compatible specifier.
- `sparse_tensor.storage_specifier.get` reads metadata from a storage specifier, such as `lvl_sz`, `dim_offset`, `dim_stride`, or buffer sizes.
- `sparse_tensor.storage_specifier.set` returns an updated storage specifier with one metadata field changed.
- `sparse_tensor.lvl` returns the size of a storage level, which can differ from a semantic dimension size when the map remaps dimensions to levels.
- `sparse_tensor.crd_translate` translates coordinates between dimension space and level space for an encoding.

### Workspace And Buffer Helpers

- `sparse_tensor.expand` creates a temporary one-dimensional workspace for sparse output assembly.
- `sparse_tensor.compress` compresses a workspace back into a sparse tensor level.
- `sparse_tensor.push_back` appends one or more values into a one-dimensional memref-like buffer, growing it if needed unless marked `inbounds`.
- `sparse_tensor.sort` sorts coordinate arrays, optionally sorting other arrays jointly.
- `sparse_tensor.reorder_coo` reorders an unordered COO tensor into the ordering required by another COO-like encoding.

These operations are normally introduced by sparse lowering or sparse conversion passes. They are implementation tools, not the preferred source-level interface.

### Sparse Semantics Inside Linalg

- `sparse_tensor.binary` describes a binary sparse set operation with separate overlap, left-only, and right-only regions.
- `sparse_tensor.unary` describes present and absent behavior for a unary sparse operation.
- `sparse_tensor.reduce` defines a custom reduction over sparse values.
- `sparse_tensor.select` keeps or drops a sparse value based on a predicate.
- `sparse_tensor.yield` terminates the regions of sparse set operations, sparse foreach, sparse iterate, and sparse coiterate.

These operations are important because sparse elementwise behavior is not always the same as dense elementwise behavior. For example, "missing" may mean "use zero", "skip this coordinate", or "copy the present side unchanged", depending on the operation.

### Element And Iterator Traversal

- `sparse_tensor.foreach` iterates over stored elements of a tensor, passing coordinates, values, and optional loop-carried state.
- `sparse_tensor.extract_iteration_space` extracts an abstract sparse iteration space for a range of levels.
- `sparse_tensor.iterate` iterates over one sparse iteration space.
- `sparse_tensor.coiterate` co-iterates multiple sparse iteration spaces and has cases for which spaces are present.
- `sparse_tensor.extract_value` reads the tensor value at a sparse iterator.

These operations make sparse traversal explicit. They are useful for sparse lowering, experiments with sparse iterators, and compiler tests.

## Transformations

The main transformations are registered as sparse tensor passes.

- `sparse-assembler`: adds `sparse_tensor.assemble` and `sparse_tensor.disassemble` around public entry points so external code can pass sparse tensors as arrays instead of relying on an unstable internal ABI. The `direct-out` option can return output buffers directly.
- `sparse-reinterpret-map`: rewrites sparse tensor mappings so later sparsification can work in level-coordinate space. It can run on all applicable ops, only `linalg.generic`, or everything except generic ops, and it has loop-ordering strategies such as `default`, `dense-outer`, and `sparse-outer`.
- `pre-sparsification-rewrite`: applies cleanup and canonical sparse rewrites before the main sparsifier.
- `sparsification`: the central pass. It converts `linalg` operations over sparse tensor types into explicit sparse loops and storage logic. Options control parallelization, whether to emit functional SCF-like loops or sparse iterator IR, and whether runtime library support is enabled.
- `stage-sparse-ops`: decomposes complex sparse operations into simpler stages, such as converting CSR to CSC through COO plus sorting.
- `lower-sparse-ops-to-foreach`: lowers higher-level sparse operations to `sparse_tensor.foreach`.
- `lower-sparse-foreach-to-scf`: lowers `sparse_tensor.foreach` to `scf`.
- `sparse-tensor-conversion`: converts sparse tensor primitives to runtime-library calls and represents sparse tensors as opaque pointers.
- `sparse-tensor-codegen`: lowers sparse tensors to compiler-visible buffers and metadata instead of opaque runtime objects.
- `sparse-buffer-rewrite`: rewrites sparse primitives on buffers into concrete MLIR code, including implementations for operations such as sorting.
- `sparse-vectorization`: vectorizes loops produced after sparsification, using the `vector` dialect. Options include `vl`, `enable-vla-vectorization`, and `enable-simd-index32`.
- `sparse-gpu-codegen`: enables GPU-oriented sparse code generation or library-call generation. If the thread count is zero, it can choose direct library calls such as cuSPARSE-like paths.
- `sparse-storage-specifier-to-llvm`: lowers sparse storage specifier operations and types to LLVM dialect structures.
- `sparsification-and-bufferization`: a mini-pipeline that combines sparse reinterpretation, sparsification, staged sparse operations, sparse operation lowering, vectorization, runtime conversion or direct codegen, and buffer rewrites.
- `sparse-space-collapse`: experimental pass that collapses consecutive sparse iteration spaces extracted from the same tensor.
- `lower-sparse-iteration-to-scf`: lowers `sparse_tensor.iterate` and `sparse_tensor.coiterate` to `scf`.

The Transform dialect also has a sparse matcher, `transform.sparse_tensor.match.sparse_inout`, which matches payload operations that have sparse inputs or outputs.

## Conversions/Lowering Paths

The most common lowering path is:

1. Start with tensor or `linalg` IR whose tensor types carry sparse encodings.
2. Run sparse reinterpretation and pre-sparsification rewrites when needed.
3. Run `sparsification` to turn sparse tensor algebra into explicit sparse iteration.
4. Lower high-level sparse ops to `sparse_tensor.foreach` or sparse iterator IR if the selected strategy uses them.
5. Lower sparse iteration to `scf`.
6. Choose either runtime-library conversion or direct codegen.
7. Continue through bufferization, memref, vector, LLVM, GPU, or target-specific lowering.

There are two important implementation choices:

- Runtime-library path: `sparse-tensor-conversion` turns sparse tensors into opaque pointers and calls helper functions for storage operations. This is simpler and often useful for interoperability.
- Direct-codegen path: `sparse-tensor-codegen` and `sparse-buffer-rewrite` expose the buffers and metadata in MLIR. This can enable more optimization because the compiler sees the storage operations.

The dialect also supports ABI and data movement paths:

- External array input can use `sparse-assembler` or direct `sparse_tensor.assemble`.
- External array output can use `sparse_tensor.disassemble`.
- Data import/export can use `sparse_tensor.new`, `sparse_tensor.out`, and runtime-library support.
- Format conversion uses `sparse_tensor.convert`, `stage-sparse-ops`, sorting, COO reordering, and storage-level lowering.

## Example IR

This example attaches a CSR-style encoding to a dense tensor, converts the value, and exposes storage arrays.

```mlir
#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

func.func @convert_and_access(%dense: tensor<4x4xf32>) -> (tensor<4x4xf32, #CSR>, memref<?xindex>, memref<?xf32>) {
  %s = sparse_tensor.convert %dense : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
  %pos = sparse_tensor.positions %s {level = 1 : index} : tensor<4x4xf32, #CSR> to memref<?xindex>
  %vals = sparse_tensor.values %s : tensor<4x4xf32, #CSR> to memref<?xf32>
  return %s, %pos, %vals : tensor<4x4xf32, #CSR>, memref<?xindex>, memref<?xf32>
}
```

This example assembles a sparse vector from external arrays.

```mlir
#SV = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed), posWidth=32, crdWidth=32}>

func.func @assemble_vector(%pos: tensor<2xi32>, %crd: tensor<3x1xi32>, %values: tensor<3xf64>) -> tensor<10xf64, #SV> {
  %s = sparse_tensor.assemble (%pos, %crd), %values : (tensor<2xi32>, tensor<3x1xi32>), tensor<3xf64> to tensor<10xf64, #SV>
  return %s : tensor<10xf64, #SV>
}
```

This example uses `sparse_tensor.foreach` to visit only stored values and sum them.

```mlir
#DCSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : compressed)}>

func.func @sum_stored(%arg0: tensor<2x4xf64, #DCSR>, %init: f64) -> f64 {
  %ret = sparse_tensor.foreach in %arg0 init(%init) : tensor<2x4xf64, #DCSR>, f64 -> f64 do {
    ^bb0(%i: index, %j: index, %v: f64, %acc: f64):
      %next = arith.addf %acc, %v : f64
      sparse_tensor.yield %next : f64
  }
  return %ret : f64
}
```

This example shows the lower-level sparse iterator model.

```mlir
#COO = #sparse_tensor.encoding<{ map = (i, j) -> (i : compressed(nonunique), j : singleton(soa)) }>

func.func @iterate_rows(%sp : tensor<4x8xf32, #COO>, %init : index) -> index {
  %space = sparse_tensor.extract_iteration_space %sp lvls = 0 : tensor<4x8xf32, #COO> -> !sparse_tensor.iter_space<#COO, lvls = 0>
  %ret = sparse_tensor.iterate %it in %space at (%row) iter_args(%acc = %init) : !sparse_tensor.iter_space<#COO, lvls = 0 to 1> -> index {
    sparse_tensor.yield %acc : index
  }
  return %ret : index
}
```

## Mental Model

Think of the dialect as a contract between high-level tensor math and low-level sparse storage.

At the top of the pipeline, a sparse tensor is still a tensor. The type says "this is a logical tensor, but it has a storage encoding." At the bottom of the pipeline, that value becomes buffers, metadata, loops, or runtime calls.

The encoding is the bridge. It tells the compiler:

- Which logical dimensions exist.
- Which storage levels exist.
- How dimensions map to levels.
- Which levels are dense, compressed, singleton, structured, ordered, unique, or nonunique.
- Which integer widths and implicit/explicit value conventions to use.

The sparsifier then uses the encoding and the `linalg` indexing maps to build sparse iteration order. It decides how to co-iterate operands, when to skip missing coordinates, when to materialize output, and when temporary workspaces are needed.

For a beginner, a good rule is:

- Write computations in `linalg` when possible.
- Use `#sparse_tensor.encoding` to describe storage.
- Let sparse passes introduce most `sparse_tensor` operations.
- Inspect `sparse_tensor` IR to understand what the compiler generated.

## Gotchas

- Sparse tensor encodings describe storage, not just optimization hints. Changing an encoding can change lowering behavior and ABI.
- Dimension order and level order are different concepts. A tensor can have dimensions `(i, j)` but store levels in another order or split them into more levels.
- `compressed`, `singleton`, `nonunique`, and `nonordered` have concrete implications for whether loops can assume sorted unique coordinates.
- `sparse_tensor.binary`, `sparse_tensor.unary`, `sparse_tensor.reduce`, and `sparse_tensor.select` are sparse semantic operations for `linalg.generic`; they are not replacements for normal arithmetic operations.
- Runtime-library lowering is easier to integrate but hides storage from later compiler optimizations.
- Direct sparse codegen exposes more IR but can make the pipeline more complex.
- `sparse_tensor.positions`, `sparse_tensor.coordinates`, and `sparse_tensor.values` expose implementation storage arrays. Use them only when you are intentionally working below the tensor abstraction.
- `sparse_tensor.assemble` trusts the provided arrays. The compiler does not prove that external sparse data is sorted, unique, or structurally valid.
- Slices carry offset, size, and stride metadata. Slice encodings can affect both iteration and storage specifier handling.
- Some sparse iterator passes are marked experimental or not yet stabilized in the source. Treat `sparse_tensor.iterate`, `sparse_tensor.coiterate`, and `sparse-space-collapse` as lower-level compiler-building tools.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorBase.td`
- `mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td`
- `mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorTypes.td`
- `mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorOps.td`
- `mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.td`
- `mlir/include/mlir/Dialect/SparseTensor/Transforms/Passes.td`
- `mlir/include/mlir/Dialect/SparseTensor/Pipelines/Passes.h`
- `mlir/include/mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.td`
- `mlir/lib/Dialect/SparseTensor/Transforms/`
- `mlir/lib/Dialect/SparseTensor/Pipelines/SparseTensorPipelines.cpp`
- `mlir/test/Dialect/SparseTensor/roundtrip.mlir`
- `mlir/test/Dialect/SparseTensor/sparse_iteration_to_scf.mlir`

Useful entry points:

- Start with `SparseTensorBase.td` for the dialect purpose and conceptual background.
- Read `SparseTensorAttrDefs.td` for encoding syntax and dimension-to-level mapping.
- Read `SparseTensorOps.td` for operation semantics.
- Read `Passes.td` and `SparseTensorPipelines.cpp` for the expected lowering pipelines.
- Read the tests under `mlir/test/Dialect/SparseTensor/` for runnable examples of encodings, conversion, foreach, iterator lowering, and storage access.
