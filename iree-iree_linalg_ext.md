# IREE LinalgExt Dialect

The IREE `iree_linalg_ext` dialect contains extensions around MLIR's `linalg` dialect. It is used for operations that IREE wants to tile, fuse, distribute, bufferize, and lower like structured tensor computations, but that are not represented directly or conveniently by upstream `linalg` operations.

For a beginner, the useful mental model is this: `linalg` is good at perfectly structured loop nests such as matmul, convolution, fill, elementwise generic ops, and reductions. Real model frontends also produce operations such as attention, top-k, scatter, gather, scan, FFT, Winograd convolution transforms, and im2col. These operations have enough structure to be optimized, but they often need special indexing rules or decomposition strategies. `iree_linalg_ext` gives IREE a place to hold those operations before lowering them to ordinary `linalg`, `scf`, `tensor`, `vector`, or target-specific forms.

## When LinalgExt Is Important

Use `iree_linalg_ext` when reading IREE compiler IR after frontend import and before final code generation. It commonly appears in paths from StableHLO, TOSA, and torch-mlir into IREE's flow and codegen pipelines.

The dialect is important when a program contains operations that should not be immediately flattened into generic loops. For example, attention should preserve the fact that it is scaled dot-product attention long enough for IREE to choose online attention or tiled attention strategies. Convolution may be converted to im2col or Winograd forms before becoming matmuls. Top-k and arg-style reductions need dedicated partial-reduction behavior. Gather, scatter, and map-store need irregular indexing while still participating in tiling and fusion.

You normally do not write this dialect as application code. It is an internal compiler staging dialect. You inspect it to understand how IREE is preserving or lowering higher-level tensor operations.

## Why It Is Needed

Without `iree_linalg_ext`, IREE would have two poor choices. It could lower these operations too early into generic loops, losing the high-level structure needed for better transformations. Or it could keep them in frontend dialects such as StableHLO, TOSA, or Torch, which are not designed to participate directly in IREE's structured codegen pipeline.

`iree_linalg_ext` sits between those extremes. Its operations implement interfaces such as destination-style operation interfaces, tiling interfaces, fusion interfaces, and aggregated-op decomposition interfaces. That means a pass can treat them like structured computations, tile them, fuse producers, split reductions, and eventually decompose them into lower-level IR.

The implication is that LinalgExt is not just a bag of random custom ops. It is a bridge from frontend semantics to IREE's structured lowering strategy.

## Type and Attribute Inventory

The local dialect does not define standalone custom types. Its operations use standard MLIR tensor, memref, scalar, index, vector, affine-map, and dictionary attributes.

The dialect defines two important attributes:

- `#iree_linalg_ext.iterator_type<parallel>` and `#iree_linalg_ext.iterator_type<reduction>` describe loop iterator kinds for LinalgExt operations that look like structured operations. They play the same conceptual role as iterator types on `linalg.generic`.
- `#iree_linalg_ext.split_reduction_mapping<N>` is a device-mapping attribute used after split-reduction transformations. It marks `scf.forall` dimensions that represent parallel partial reductions.

Several operations also use ordinary MLIR attributes with dialect-specific meaning:

- `indexing_maps` describes how loop or tile dimensions index operands and results.
- `iterator_types` distinguishes parallel and reduction iteration spaces.
- `dimension`, `dimension_map`, `unique_indices`, and `is_sorted` control gather, scatter, sort, scan, and top-k behavior.
- `decomposition_config` is a dictionary used by attention-like ops to pass attributes into decomposed matmul operations, such as `qk_attrs`, `pv_attrs`, and `use_exp2`.
- Winograd and im2col operations carry layout and convolution metadata such as `strides`, `dilations`, `kernel_size`, `image_dimensions`, `kernel_dimensions`, `batch_pos`, `m_pos`, `k_pos`, `input_k_perm`, and `output_perm`.

## Interfaces

`LinalgExtInterface` is a small structured-op-style interface. It exposes shape information for operands, supports result shape reification, and provides helper methods for updating segmented input and output operands.

`LinalgFusionInterface` lets LinalgExt and Linalg operations participate in fusion. It exposes indexing maps for operands and results, loop counts, parallel loop counts, and static loop ranges.

Many concrete operations also implement upstream interfaces such as `TilingInterface`, `DestinationStyleOpInterface`, `ReifyRankedShapedTypeOpInterface`, `IndexingMapOpInterface`, `PartialReductionOpInterface`, and `AggregatedOpInterface`. Those interfaces are the reason the dialect can be progressively lowered rather than immediately erased.

## Operation Inventory

The local `iree_linalg_ext` dialect defines 20 operations.

### Utility Operations

`iree_linalg_ext.index` is a region-local index operation. It mirrors `linalg.index`, but verifies that it appears in supported LinalgExt parent operations such as `iree_linalg_ext.custom_op`, `iree_linalg_ext.attention`, or `iree_linalg_ext.online_attention`.

`iree_linalg_ext.yield` terminates regions inside LinalgExt operations. It is the dialect's counterpart to `linalg.yield`.

### Irregular Indexing and Movement

`iree_linalg_ext.scatter` scatters update slices into an original tensor or memref according to an indices tensor and `dimension_map`. It has an optional mask and a `unique_indices` attribute. If indices are not unique, the batch loops are treated as reductions because multiple updates may target the same destination element.

`iree_linalg_ext.gather` is the opposite direction: it gathers slices from a source into an output according to indices and `dimension_map`. Like scatter, it can use an optional mask.

`iree_linalg_ext.map_load` loads from a source into an output using an index transformation region. The region maps output indices to source indices and yields a padding value for out-of-bounds reads. This is useful for explicit layout transforms, padding-aware loads, and vectorized mapped reads.

`iree_linalg_ext.map_store` stores from an input into an output using an index transformation region. The region maps input indices to output indices and yields a mask deciding whether the write happens. Later decomposition can turn vectorized, bufferized forms into `vector.store` sequences or `vector.scatter`.

### Ordering, Selection, and Reductions

`iree_linalg_ext.sort` sorts one or more shaped values along a selected dimension using a comparator region. Its semantics follow the XLA-style sort model.

`iree_linalg_ext.topk` is the legacy top-k operation. It reduces one dimension from size N to K and produces output values and output indices. Its comparator region uses swap-style semantics.

`iree_linalg_ext.topk_v2` is the newer top-k form. It has optional output indices, supports input indices of any integer type, and uses a pure ordering predicate in its comparator region. The `is_sorted` marker requests sorted top-k output.

`iree_linalg_ext.arg_compare` performs an arg-style reduction with a user-defined comparator. It can compute implicit local indices, implicit indices with an `index_base` for tiled reductions, or explicit indices supplied as an input tensor. This makes it useful for argmax, argmin, and partial-reduction merging.

`iree_linalg_ext.scan` computes an inclusive or exclusive scan along a dimension. The local StableHLO conversion path notes that the operation supports purely inclusive prefix-scan cases.

`iree_linalg_ext.exp_reduction` is a restricted reduction form for exponential reductions, especially numerically stable online normalization. It represents computations such as softmax-like running maximum and running sum patterns that are hard to infer safely from arbitrary `linalg.generic` regions.

### Attention

`iree_linalg_ext.attention` represents scaled dot-product attention:

```text
softmax(Q @ K.T * scale) @ V
```

It can also take a mask that is added into the score matrix before softmax. It carries indexing maps for query, key, value, scale, optional mask, and output. It preserves the high-level operation long enough for IREE to choose a useful attention lowering strategy.

`iree_linalg_ext.online_attention` represents a tiled, online-normalized form of attention. Instead of materializing a complete softmax matrix, it tracks output, running maximum, and running sum. This makes the attention reduction dimension tileable and mergeable. After the full reduction is complete, the output is normalized by the final sum.

### FFT

`iree_linalg_ext.fft` represents a 1D iterative FFT over the innermost dimension. The source expects the FFT length to be a power of two and assumes bit-reversal has already been handled. It may carry coefficient tensors or buffers. StableHLO FFT conversion and Torch RFFT conversion use helper rewrites that create LinalgExt FFT-related IR.

### Convolution Preparation

`iree_linalg_ext.im2col` transforms convolution input data into a GEMM-compatible layout. It carries detailed convolution metadata, dimension mappings, output delinearization sizes, layout permutations, and optional input/output padding. This operation lets IREE expose convolution as matmul-like work without losing all indexing structure too early.

`iree_linalg_ext.winograd.input_transform` computes the input-side Winograd transform for convolution. It transforms image tiles before the Winograd-domain multiplication.

`iree_linalg_ext.winograd.filter_transform` computes the filter-side Winograd transform. It transforms convolution filters into the Winograd domain.

`iree_linalg_ext.winograd.output_transform` converts Winograd-domain results back to the normal output image domain.

These Winograd operations are normally temporary. After tiling, `iree-linalg-ext-decompose-winograd` lowers them into matrix multiplications and related tensor operations.

### Custom Tiled Computation

`iree_linalg_ext.custom_op` represents a prescribed tile-level computation. It is similar in spirit to `linalg.generic`, with `indexing_maps`, `iterator_types`, inputs, outputs, and a region, but its region receives slices instead of just scalar elements. It exists for cases where IREE wants a fused tiled computation that current automatic fusion cannot discover or express cleanly.

## Transformations and Conversions

The dialect has internal lowering and cleanup passes.

`iree-linalg-ext-to-loops` converts LinalgExt operations to loops and Linalg operations. This is the broad escape path when an operation needs to become ordinary structured control flow.

`iree-linalg-pad-contraction-to-block-size` pads supported matmul-style contractions to row and column block-size alignments.

`iree-linalg-ext-topk-split-reduction` splits top-k into a map-reduce style computation with parallel partial reductions and a final merge.

`iree-linalg-ext-decompose-aggregated-ops` calls the `AggregatedOpInterface` decomposition on selected operations, such as attention-like or custom aggregated operations. It can be filtered by fully qualified operation name.

`iree-linalg-ext-decompose-im2col` decomposes `iree_linalg_ext.im2col` into lower-level slice, insert, loop, or async-copy-like forms. It can optionally unroll the resulting loop nest.

`iree-linalg-ext-decompose-map-store` decomposes vectorized, bufferized `iree_linalg_ext.map_store` operations into vector operations. It chooses direct load/store-style sequences when the mapping is unit-stride-friendly and `vector.scatter` otherwise.

`iree-linalg-ext-decompose-winograd` lowers tiled Winograd transform operations into matrix multiplications, fills, slices, pads, and tensor operations.

`iree-linalg-ext-convert-conv-to-im2col-op` rewrites compatible Linalg convolution operations into an im2col plus GEMM-style implementation.

`iree-linalg-ext-convert-conv2d-to-winograd` rewrites compatible 2D convolutions into Winograd input/filter/output transform structure. By default, it targets annotated convolutions, with an option to replace all compatible convolutions.

`iree-linalg-ext-decompose-attention` decomposes online attention into lower-level Linalg operations, with an option to use `exp2` rather than `exp`.

`iree-linalg-ext-convert-attention-to-online-attention` rewrites normal attention into online attention, adding running max and running sum outputs and a final normalization step.

`iree-linalg-ext-fold-unit-extent-dims` folds unit dimensions in LinalgExt operations, using reshapes or slices depending on the option.

`iree-linalg-ext-test-reshape-fusion` is a test pass for reshape fusion patterns around LinalgExt operations.

The dialect also has related transform-dialect operations: `transform.iree.decompose_aggregate_op` and `transform.iree.convert_to_online_attention`. They are not `iree_linalg_ext` operations, but they let transform dialect scripts target LinalgExt attention and aggregate decomposition behavior.

## Frontend Conversion Paths

LinalgExt is created by multiple frontend conversion paths.

`iree-stablehlo-to-linalg-ext` converts selected StableHLO and CHLO patterns to LinalgExt. Local source patterns include StableHLO sort to `iree_linalg_ext.sort`, StableHLO scatter to `iree_linalg_ext.scatter`, StableHLO FFT forms to FFT rewrites, StableHLO scan to `iree_linalg_ext.scan`, and CHLO top-k to `iree_linalg_ext.topk`. Elementwise operations inside LinalgExt regions are also converted so the regions can use IREE-compatible operations and `iree_linalg_ext.yield`.

`iree-tosa-to-linalg-ext` converts TOSA scatter-style behavior to LinalgExt scatter, including materializing batch-index information that LinalgExt's scatter expects explicitly.

`torch-iree-tm-tensor-to-linalg-ext` converts torch-mlir `tm_tensor` operations into LinalgExt forms. The local source maps at least TMTensor scatter and scaled dot-product attention into `iree_linalg_ext.scatter` and `iree_linalg_ext.attention`.

`torch-iree-torch-unstructured-to-linalg-ext` handles unstructured Torch operations that benefit from LinalgExt staging. The local source includes conversions for FFT/RFFT, scaled dot-product attention to `iree_linalg_ext.online_attention`, and arg-style reductions to `iree_linalg_ext.arg_compare`.

These conversions explain why LinalgExt appears even if a source model never mentioned IREE. It is a compiler-selected intermediate form.

## What It Implies

When you see `iree_linalg_ext.attention`, IREE has not yet committed to a final attention implementation. It still has the chance to switch to online attention, tile along the softmax reduction dimension, fuse transposes, or pass attributes into decomposed QK and PV matmuls.

When you see `iree_linalg_ext.im2col`, a convolution is being prepared for a GEMM-like lowering. The operation's attributes explain how output coordinates map back to input image and filter coordinates.

When you see Winograd transform ops, IREE is using a convolution algorithm choice, not just a layout transform. Those ops should disappear after decomposition.

When you see `iree_linalg_ext.map_store` or `iree_linalg_ext.map_load`, look at the transformation region. The region is the real indexing rule. It explains where each element is loaded from or stored to, and whether padding or masks affect the operation.

When you see `iree_linalg_ext.custom_op`, treat it as a deliberately prescribed tiled fusion. The region works over slices, so it is closer to a tile program than to scalar `linalg.generic`.

## How To Read LinalgExt IR

Start with the operation name and decide whether it is preserving source semantics, choosing an algorithm, or expressing irregular indexing.

Then inspect the destination-style operands. Most important LinalgExt operations have `ins` and `outs`, and tensor forms return new result tensors while memref forms mutate destinations.

Next inspect regions. Comparator, reduction, attention mask, map-load, map-store, and custom tiled computations all use regions. The operation name tells you the framework; the region tells you the scalar or tile-level rule.

Finally inspect attributes. `dimension`, `dimension_map`, `indexing_maps`, `iterator_types`, and convolution metadata are not decoration. They define the iteration space and the mapping between logical operation semantics and actual tensor coordinates.

## Minimal Example

This simplified scatter shows the style of the dialect:

```mlir
%result = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%updates, %indices : tensor<4xf32>, tensor<4x1xi32>)
    outs(%original : tensor<16xf32>) {
  ^bb0(%update: f32, %old: f32):
    iree_linalg_ext.yield %update : f32
} -> tensor<16xf32>
```

The operation says "scatter updates into this output according to indices." The region says how to combine an update with the existing value. Later IREE passes can tile, fuse, or lower this operation because it still exposes structured interfaces.
