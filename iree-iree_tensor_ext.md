# IREE TensorExt Dialect

The IREE `iree_tensor_ext` dialect contains tensor operations and types that IREE needs but that do not fit cleanly in the upstream MLIR `tensor` dialect. It is a compiler-internal dialect, not a frontend tensor library. Its operations appear when IREE is forming dispatches, moving tensors across dispatch boundaries, preserving shape information, calculating workgroup counts, or experimenting with sparse and ragged tensor layouts.

For a beginner, the main idea is that TensorExt gives IREE a few precise handles on tensors at points where ordinary tensor IR would be too vague. A dispatch body does not receive ordinary tensors directly. It receives special placeholder values of type `!iree_tensor_ext.dispatch.tensor<...>`, then uses `iree_tensor_ext.dispatch.tensor.load` to get an SSA tensor value and `iree_tensor_ext.dispatch.tensor.store` to write a result. TensorExt also supplies marker operations, workgroup-count helpers, and ragged-shape casts.

## When TensorExt Is Important

Use this dialect as a reading tool when inspecting IREE IR around `flow.dispatch.workgroups`, dispatch creation, tensor slicing, unsupported element-type handling, split reductions, and sparse or ragged tensor experiments. If you see `iree_tensor_ext` operations, the compiler is no longer just representing mathematical tensor expressions. It is starting to expose dispatch boundaries and implementation constraints.

The dialect is important because IREE needs a transition point between high-level tensor semantics and executable code generation. Upstream MLIR tensor operations are value-semantic: they describe immutable tensor values. IREE dispatches, however, need to describe reads and writes against dispatch arguments, dynamic dimensions, workgroup-count logic, and target-specific element-type support. TensorExt gives those concerns explicit IR.

You normally do not write this dialect by hand as an application author. It is mostly introduced by IREE passes such as `iree-dispatch-creation-convert-dispatch-regions-to-workgroups`, `iree-dispatch-creation-materialize-default-workgroup-count-region`, `iree-dispatch-creation-insert-tensor-barriers`, and encoding/materialization paths in Stream and Flow lowering.

## Why It Is Needed

Without TensorExt, IREE would have to overload ordinary tensor operations with dispatch-specific meaning. For example, a dispatch input is not just a tensor value. It is a placeholder for a buffer-like binding with known access permissions, shape metadata, possible dynamic dimensions, and a region-local load/store protocol. Encoding that as a plain `tensor<?xf32>` would lose information that later passes need.

TensorExt also helps keep transformations controlled. The compute barrier operations are identity operations, but their presence tells reshape propagation and related passes where movement is allowed or blocked. The workgroup-count operations are similarly small, but they let IREE materialize a count region in a structured way and preserve a relationship between captured workload operands and values inside the dispatch body.

The sparse and ragged pieces solve a different problem. A ragged tensor has a logical shape, but one dimension has row-dependent size. That shape cannot be described by a normal dense tensor shape alone. TensorExt uses sparse shape attributes and cast operations to keep that logical shape visible while still lowering through tensor or memref types.

## Type Inventory

The main TensorExt type is `!iree_tensor_ext.dispatch.tensor`. It is a placeholder for a dispatch region input or output operand. The type carries an access mode and a bound type:

```mlir
!iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>
!iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>
!iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>
```

The access mode can be `readonly`, `writeonly`, or `readwrite`. Readable dispatch tensors can be used by `iree_tensor_ext.dispatch.tensor.load`. Writable dispatch tensors can be used by `iree_tensor_ext.dispatch.tensor.store`. The bound type is either a ranked tensor type or a scalar integer/float type. The dispatch tensor type also exposes rank, shape, dynamic dimension count, element type, and whole-slice checks to compiler code.

## Attribute and Interface Inventory

The dialect defines two attributes for sparse or ragged shape work.

- `#iree_tensor_ext.ragged_shape<n>` marks a shaped type whose dimensions `n` and `n + 1` represent ragged rows. The `n + 1` dimension is logically dynamic because each row can have a different length.
- `#iree_tensor_ext.sparse_iteration_dims<[...]>` records sparse iteration dimensions on loops such as `scf.forall`. It lets tiling and distribution logic know which loop dimensions correspond to sparse data-space dimensions.

TensorExt also defines sparse interfaces. `SparseShapeAttrInterface` lets an attribute report which dimensions are sparse. `SparseCastOpInterface` is implemented by operations that cast a dense shaped value into a sparse or ragged shaped value. The interface supplies methods for lowering sparse loop ranges, estimating loop ranges, resolving a sparse slice back to the source tensor, and reporting which sparse dimensions can be distributed across workgroups.

## Operation Inventory

The local TensorExt dialect defines ten operations.

### `iree_tensor_ext.bitcast`

`iree_tensor_ext.bitcast` reinterprets a ranked tensor with a new shape or element type without changing the serialized contents. It is similar in spirit to `flow.tensor.bitcast`, but it is allowed to be cloned into dispatch regions and supports transformations needed by executable translation and bufferization. IREE uses it when a tensor's element type must be changed for target support while the underlying bytes stay the same size.

### `iree_tensor_ext.dispatch.tensor.load`

`iree_tensor_ext.dispatch.tensor.load` reads a tensor or subtensor from a dispatch tensor placeholder. It takes offsets, sizes, and strides, so it can load the whole tensor or a slice. This is the operation that turns a `!iree_tensor_ext.dispatch.tensor<readonly:...>` block argument into an ordinary ranked tensor SSA value that compute operations can use.

### `iree_tensor_ext.dispatch.tensor.store`

`iree_tensor_ext.dispatch.tensor.store` writes a ranked tensor value into a writable dispatch tensor placeholder. Like the load operation, it takes offsets, sizes, and strides, so it can write a whole output or a slice. If multiple workgroups write overlapping regions, behavior is undefined, so later compiler stages must ensure the tiling and distribution are valid.

### `iree_tensor_ext.dispatch.workgroup_count_from_slice`

`iree_tensor_ext.dispatch.workgroup_count_from_slice` is a placeholder for the default workgroup-count calculation. It returns the three index values used as the x, y, and z workgroup counts. The operation assumes that the captured index workload values and a program slice inside the dispatch are enough to compute the count.

### `iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier`

`iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier` adjusts a default workgroup count for split reductions. A split reduction introduces an `scf.forall` representing partial reductions. This operation combines the inner workgroup count with the workload values so the final count covers both the normal parallel work and the split-reduction dimension.

### `iree_tensor_ext.dispatch.workload.ordinal`

`iree_tensor_ext.dispatch.workload.ordinal` marks a value inside a dispatch body as corresponding to one captured workload operand. The body signature of `flow.dispatch.workgroups` is not preserved all the way through IREE's compilation stack, so this operation gives backends a way to recover which internal value maps to workload operand 0, 1, and so on.

### `iree_tensor_ext.compute_barrier.start`

`iree_tensor_ext.compute_barrier.start` is an identity operation on a tensor value. Its purpose is not computation. It prevents certain transformations, especially reshape propagation, from moving operations downward across the marked boundary.

### `iree_tensor_ext.compute_barrier.end`

`iree_tensor_ext.compute_barrier.end` is the matching identity marker for the other direction. It prevents transformations from moving operations upward across the boundary. The start and end operations let IREE temporarily fence off compute regions, perform controlled rewrite steps, and then remove the markers later.

### `iree_tensor_ext.cast_to_ragged_shape`

`iree_tensor_ext.cast_to_ragged_shape` reinterprets a tensor or memref as a ragged shaped value. It takes a source, a `ragged_dim`, a `column_lengths` value, and optionally the number of ragged rows. The `column_lengths` input behaves like the row pointer array in CSR sparse storage: row `i` has length `column_lengths[i + 1] - column_lengths[i]`. The result type carries `#iree_tensor_ext.ragged_shape<...>`.

### `iree_tensor_ext.linearize_ragged_dims`

`iree_tensor_ext.linearize_ragged_dims` turns the contiguous sparse dimensions of a ragged tensor or memref into one dense dimension. It removes the ragged encoding or layout and preserves the non-sparse dimensions. This is useful when later lowering wants a linear dense representation of the actual stored elements.

## Transformations and Conversions

TensorExt has one pass registered directly under its own transform directory: `iree-tensor-ext-test-sparse-op-interface-methods`. It is a test pass for `SparseCastOpInterface` methods, not a normal production lowering pass.

Most practical TensorExt transformations live in other IREE pipelines. `iree-dispatch-creation-insert-tensor-barriers` inserts `iree_tensor_ext.compute_barrier.start` and `iree_tensor_ext.compute_barrier.end` around compute regions. `iree-dispatch-creation-fold-reshapes-into-tensor-barriers` moves `tensor.expand_shape` and `tensor.collapse_shape` through those markers in a controlled way. `iree-dispatch-creation-remove-tensor-barriers` erases the markers once they have done their job.

`iree-dispatch-creation-bitcast-unsupported-element-types` uses `iree_tensor_ext.bitcast` when tensor element types are unsupported by the HAL or backend path. Inside a dispatch, TensorExt bitcasts are useful because they can participate in dispatch-local transformations.

`iree-dispatch-creation-convert-dispatch-regions-to-workgroups` is one of the most important conversion points. It converts `flow.dispatch.region` operations into `flow.dispatch.workgroups` operations. During that rewrite, ranked tensor operands become `!iree_tensor_ext.dispatch.tensor` block arguments. The body uses `iree_tensor_ext.dispatch.tensor.load` to read inputs and `iree_tensor_ext.dispatch.tensor.store` to write results.

`iree-dispatch-creation-materialize-default-workgroup-count-region` creates the default workgroup-count region. It inserts `iree_tensor_ext.dispatch.workgroup_count_from_slice`, may wrap it with `iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier`, and adds `iree_tensor_ext.dispatch.workload.ordinal` operations inside the workgroup body to preserve workload operand positions.

`iree-dispatch-creation-convert-tensor-to-flow` folds ordinary `tensor.extract_slice` and `tensor.insert_slice` operations into TensorExt dispatch tensor loads and stores. The folding patterns combine nested slice offsets, sizes, and strides so the dispatch load or store directly describes the slice that crosses the dispatch boundary.

Stream and encoding passes also depend on TensorExt. `iree-stream-materialize-encodings` emits dispatches that use TensorExt dispatch tensor types, loads, stores, and workload ordinals. `iree-stream-encode-host-tensors` and `iree-stream-encode-device-tensors` update tensor storage and dispatch tensor element types when encoded or packed layouts are materialized.

## What It Implies

When you see `!iree_tensor_ext.dispatch.tensor`, read it as a dispatch binding placeholder, not as a normal tensor value. A load operation is needed before computation can use the data, and a store operation is needed to publish results. The access mode tells you whether the binding is an input, output, or both.

When you see compute barriers, do not look for runtime behavior. They are compile-time markers. They imply that the compiler is managing where reshapes and related transformations may move.

When you see `#iree_tensor_ext.ragged_shape`, the shaped value has a logical sparse or ragged structure. Its printed rank is not enough to understand iteration. You also need the ragged row dimension and the `column_lengths` data that describes row sizes.

## How To Read TensorExt IR

Start at dispatch boundaries. If a block argument has type `!iree_tensor_ext.dispatch.tensor<readonly:...>`, find its `iree_tensor_ext.dispatch.tensor.load` operations. If an argument is `writeonly` or `readwrite`, find its `iree_tensor_ext.dispatch.tensor.store` operations. The load and store offsets, sizes, and strides tell you which part of the logical tensor the dispatch touches.

Then inspect workgroup-count logic. `iree_tensor_ext.dispatch.workgroup_count_from_slice` tells you the compiler is deriving the count from workload values. A split-reduction modifier means the dispatch has extra partial-reduction parallelism.

Finally, look for barrier and sparse operations. Barriers explain why reshape movement stops at a certain point. Ragged casts and linearization tell you that the program is modeling irregular rows rather than a fully dense rectangular tensor.

## Minimal Example

This simplified sketch shows how a dispatch body can load from and store to TensorExt dispatch tensor placeholders:

```mlir
%input = iree_tensor_ext.dispatch.tensor.load %arg0,
    offsets = [0, 0], sizes = [%m, %n], strides = [1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %n}
      -> tensor<?x?xf32>

%result = "some.compute"(%input) : (tensor<?x?xf32>) -> tensor<?x?xf32>

iree_tensor_ext.dispatch.tensor.store %result, %arg1,
    offsets = [0, 0], sizes = [%m, %n], strides = [1, 1]
    : tensor<?x?xf32>
      -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m, %n}
```

The mental model is direct: TensorExt makes dispatch-local tensor access explicit. Loads bring data from dispatch placeholders into ordinary tensor SSA form. Stores write tensor SSA results back to dispatch placeholders. The rest of the dialect supplies the markers and shape tools IREE needs to make that lowering precise.
