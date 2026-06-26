# torch-mlir TMTensor Dialect

The torch-mlir `tm_tensor` dialect is a temporary tensor computation dialect used inside torch-mlir. The `tm` stands for torch-mlir. It holds tensor operations that are common and useful, but that do not yet have a good direct representation in upstream MLIR dialects.

For a beginner, the most useful mental model is this: `tm_tensor` is a staging dialect between Torch/ATen semantics and lower-level structured IR. It names operations such as scan, scatter, sort, attention, and top-k while they are still recognizable. Later passes can bufferize them, lower them to loops, or convert them into another project's extension dialect such as IREE LinalgExt.

You usually do not write `tm_tensor` by hand. You inspect it when a Torch model contains operations that are too structured to be arbitrary loops but too specialized for plain upstream `linalg`.

## When TMTensor Is Important

`tm_tensor` is important when torch-mlir lowers recognized PyTorch operations that need special tensor semantics. Examples include `aten.cumsum`, `aten.cumprod`, `aten.sort`, `aten.scatter`, `aten.scatter_add`, `aten.scatter_reduce`, `aten.bincount`, `aten.scaled_dot_product_attention`, `aten.kthvalue`, `aten.index_put`, and `aten.max_pool2d_with_indices_backward`.

Those operations can be expressed as loops, but lowering them too early would hide useful intent. A sort is not just a loop nest; it has a comparison dimension and a comparator. A scatter is not just stores; it has an update tensor, an index tensor, a destination tensor, a dimension map, and possible repeated indices. Attention is not just several matmuls; it is a high-level operation that backends may want to lower with a specialized algorithm.

Use this dialect as a reading tool when debugging torch-mlir conversion pipelines. If you see `tm_tensor`, torch-mlir has already left the high-level Torch dialect for that operation, but it has not yet fully committed to low-level loops or backend-specific codegen.

## Why It Is Needed

The core problem is representation. Torch has many tensor operations whose behavior is well understood but awkward to represent cleanly using only existing upstream MLIR operations. If torch-mlir lowers these directly to `scf`, `memref`, `tensor`, and `arith`, the compiler loses the operation name and much of the shape/indexing intent. If it keeps them as Torch operations, downstream projects cannot easily treat them as structured tensor computations.

`tm_tensor` sits in the middle. Its operations are inspired by MLIR `linalg` structured operations and by IREE's `iree_linalg_ext` dialect, but the dialect is meant to be more of an interchange/interface layer than an opinionated codegen layer. The source describes it as temporary: the desired long-term outcome is either upstream MLIR support or replacement by upstream functionality.

The implication is that `tm_tensor` is a compiler cooperation point. It gives torch-mlir and downstream consumers stable names for common tensor operations without requiring every backend to understand arbitrary Torch semantics directly.

## Types and Interfaces

The dialect does not define custom data types. Its operations use normal MLIR tensor and memref types, plus standard scalar and index types. The dialect does define a helper constraint for operands that can be ranked tensors or memrefs.

The important abstractions are interfaces.

`TMTensorInterface` is a structured-op-style interface. It exposes an operation's input operands, output operands, tensor operands, buffer operands, output types, and tied destination operands. This is why the dialect looks similar to destination-style `linalg`: tensor forms produce result tensors, while buffer forms write into output buffers.

`ScalarLoopOpInterface` lets an operation describe how to lower itself into scalar loops. It provides destination operands, iterator types, loop ranges, and a `generateScalarImplementation` hook. The loop lowering pass uses this interface generically rather than hard-coding a separate pass pattern for every operation.

This design is important. It means the operation name preserves high-level meaning, while the interfaces still give generic passes enough structure to lower, bufferize, and analyze the operation.

## Operation Inventory

The local `tm_tensor` dialect defines six operations.

### `tm_tensor.scan`

`tm_tensor.scan` computes an inclusive or exclusive scan along one dimension. It has one input, two outputs, a `dimension` attribute, an `inclusive` attribute, and a region that combines the running value with the next input value.

The first output is the scan result. The second output is an accumulator whose rank is one less than the input rank because it removes the scan dimension. The verifier checks that the input, output, and accumulator have compatible shapes and element types.

For an inclusive scan, the first result along the scan dimension comes from the input, and later elements combine the previous output with the current input. For an exclusive scan, the first result comes from the accumulator, and later elements combine the previous output with the previous input.

Torch conversions use this operation for cumulative operations. `aten.cumsum` becomes a `tm_tensor.scan` with an addition region. `aten.cumprod` becomes a `tm_tensor.scan` with a multiplication region.

### `tm_tensor.scatter`

`tm_tensor.scatter` updates an output tensor or buffer according to an index tensor. It takes two inputs, `updates` and `indices`, and one output, the original destination. Its region receives an update element and the current destination element, then yields the replacement value.

The operation uses a `dimension_map` attribute to say which destination dimensions are indexed by the index tensor. The `unique_indices` attribute records whether updates are known to target distinct locations. If indices are not unique, the first loop is treated as a reduction because multiple updates may write the same destination element.

The verifier is strict. The index tensor must be rank 2 with integer elements, and its index depth must be static. The dimension map length must match the index depth. Update shape, original shape, region argument types, and yielded type all have to agree.

Torch conversions use `tm_tensor.scatter` for `aten.scatter`, `aten.scatter_add`, `aten.scatter_reduce`, `aten.bincount`, `aten.index_put`, and parts of `aten.max_pool2d_with_indices_backward`. The combining region explains whether the scatter overwrites, adds, multiplies, takes min/max, or counts.

### `tm_tensor.sort`

`tm_tensor.sort` sorts one or more output tensors along a selected dimension. Its region is a comparator. For each sorted tensor, the region receives the left and right element values. The region must yield one `i1` predicate.

The operation has no `ins` operands in the current verifier; the tensors being sorted are passed as `outs`. This follows the destination-style pattern used by the dialect. Every output tensor must have the same rank and shape, and the region must have exactly two block arguments per output tensor.

Torch conversion uses `tm_tensor.sort` for `aten.sort`. The converter also creates an indices tensor, so the sort returns both sorted values and sorted indices. The comparator is built from the requested descending flag.

In loop lowering, `tm_tensor.sort` currently lowers through a simple bubble-sort-like scalar implementation over the selected dimension. That is a reference lowering, not a claim that production backends should use bubble sort.

### `tm_tensor.attention`

`tm_tensor.attention` represents scaled dot-product-style attention without carrying every PyTorch option directly. It takes query, key, value, and optionally a mask. Its output shape follows the usual attention structure.

The TableGen description presents the computation as:

```text
matmul(softmax(matmul(Q, transpose(K)) + M), V)
```

The verifier checks that query, key, value, mask, and output shapes agree on their batch dimensions, sequence dimensions, and head dimensions. The implementation's scalar reference path builds matmul, masking, softmax-like normalization, and the value projection using lower-level operations after bufferization.

Torch conversion uses this operation for `aten.scaled_dot_product_attention` when the supported constraints are met. The converter rejects unsupported dropout, handles causal mask creation, can broadcast masks, checks scale compatibility, and handles grouped-query attention preprocessing in supported static cases.

This operation is especially important because downstream backends often care about attention as an algorithmic unit. Lowering it immediately to generic loops would erase information that a backend could use for tiled or online attention.

### `tm_tensor.topk`

`tm_tensor.topk` selects the top or bottom K values along a dimension. It accepts a values tensor and optionally an input indices tensor. It produces two outputs: selected values and selected indices. If no input indices are provided, indices are inferred from the selected dimension.

The comparator region receives the next input candidate and an existing K output candidate, then yields an `i1`. If the predicate is true, the candidate should replace the existing K value. Equal values keep the first occurrence by comparing indices as a tie breaker in the scalar implementation.

The verifier checks that the operation has one or two inputs, exactly two outputs, a valid dimension, matching value element types, `i32` indices when indices are provided, compatible input/output ranks, and an `i1`-yielding comparator region.

Torch conversion uses this operation as part of `aten.kthvalue`. The conversion builds a min-k style `tm_tensor.topk`, then uses additional `linalg.generic` operations to find the kth value and translate the selected index back to the original input index.

### `tm_tensor.yield`

`tm_tensor.yield` terminates regions inside the other `tm_tensor` operations. It is the dialect's region terminator, similar in role to `linalg.yield` for Linalg operations.

The yielded values are not incidental. They define the scalar combining rule for scan, scatter, sort, and top-k.

## Transformations and Conversions

The dialect has two direct dialect passes and one important frontend conversion pass.

`convert-torch-to-tmtensor` converts recognized Torch ATen operations to a mixture of `tm_tensor` and ordinary `linalg`/`tensor`/`arith` IR. It is similar to torch-mlir's Torch-to-Linalg conversion, but it is allowed to introduce `tm_tensor` operations when no good direct Linalg form exists. Its `allow-non-finites` option controls whether some generated initial values may use infinities or must use the closest finite value for a dtype.

The local pass marks these Torch operations illegal and rewrites them when patterns match: `aten.bincount`, `aten.index_put`, `aten.max_pool2d_with_indices_backward`, `aten.scatter_reduce`, `aten.sort`, `aten.cumsum`, `aten.cumprod`, `aten.scaled_dot_product_attention`, `aten.scatter`, `aten.scatter_add`, and `aten.kthvalue`.

`tm-tensor-bufferize` converts tensor-based `tm_tensor` operations to buffer-based operations. It converts ranked tensors to memrefs, unranked tensors to unranked memrefs, materializes tensor/buffer boundary casts, and rewrites each `TMTensorInterface` operation generically. For each tensor result, it either allocates a fresh result buffer or clones an output buffer when the operation's payload reads the destination value.

`tm-tensor-to-loops` lowers `ScalarLoopOpInterface` operations into `scf.for` loops plus supporting dialects such as `arith`, `math`, `memref`, and `linalg`. The pass recursively builds one loop per iteration range, calls the operation's scalar implementation at the innermost point, then erases the original operation. This is the reference escape path when the dialect needs to disappear into ordinary loop IR.

There is also an important downstream conversion outside torch-mlir's core TMTensor dialect. IREE's torch integration has a `torch-iree-tm-tensor-to-linalg-ext` path that converts selected `tm_tensor` operations into `iree_linalg_ext`, including scatter and attention. That shows the intended role of `tm_tensor`: it can be an interchange dialect between torch-mlir and another MLIR-based compiler.

## What It Implies

Seeing `tm_tensor.scan` implies that a cumulative operation is still visible as a prefix computation. Look at the `dimension`, `inclusive`, accumulator shape, and region body to understand whether it is sum, product, or another associative-like computation.

Seeing `tm_tensor.scatter` implies irregular indexed writes. The `dimension_map` explains how columns of the index tensor map to destination dimensions. The `unique_indices` flag tells you whether repeated writes are expected. The region tells you whether the operation overwrites, adds, reduces, or performs some custom update.

Seeing `tm_tensor.sort` implies order-sensitive behavior along one dimension. The region is the comparator, and multiple outputs usually mean values and companion data such as indices are being sorted together.

Seeing `tm_tensor.attention` implies that torch-mlir has preserved attention as a recognizable operation. That is useful for backends that can choose a specialized lowering instead of accepting a fully expanded matmul/softmax/matmul sequence too early.

Seeing `tm_tensor.topk` implies a partial selection operation. It is not a full sort. The output dimension is reduced to K, and the second result tracks where the selected values came from.

## How To Read TMTensor IR

Start with the operation name, then inspect `ins`, `outs`, attributes, and the region.

The `outs` operands are destination-style operands. In tensor form, their types usually determine the result types. In memref form, they are the buffers that will be updated.

For region operations, read `tm_tensor.yield` as the semantic payload. In scatter, the yielded value is the new destination element. In scan, it is the new running accumulator/output value. In sort and top-k, it is the comparison predicate.

Finally, check whether the operation is still tensor-based or already bufferized. Tensor-based operations return ranked tensor results. Bufferized operations write into memrefs and are ready for `tm-tensor-to-loops`.

## Minimal Example

This simplified scan computes a cumulative sum along dimension 0:

```mlir
%result = tm_tensor.scan dimension(0) inclusive(true)
    ins(%input : tensor<4xi32>)
    outs(%output, %acc : tensor<4xi32>, tensor<i32>) {
  ^bb0(%previous: i32, %next: i32):
    %sum = arith.addi %previous, %next : i32
    tm_tensor.yield %sum : i32
} -> tensor<4xi32>
```

The operation says "perform a scan along dimension 0." The region says "combine values with integer addition." A later pass can bufferize this operation and lower it into ordinary loops, but until then the IR still preserves the fact that it is a scan.
