# IREE `iree_vector_ext` Dialect

The IREE `iree_vector_ext` dialect is a small codegen dialect for vector operations that IREE needs before those ideas are available, or convenient, in upstream MLIR's `vector` dialect. It is not a general replacement for `vector`. It is a staging area for IREE-specific vector layout, distributed-vector materialization, indexed gather/scatter transfers, and vectorized arg-reductions.

For a beginner, the key idea is that IREE codegen often needs to talk about two related but different things at once: the mathematical vector shape of a value, and the way that vector is mapped onto hardware execution resources such as subgroups, threads, and per-thread elements. The standard `vector` dialect gives a broad vocabulary for vector computation. `iree_vector_ext` adds the extra vocabulary IREE needs while lowering those computations toward target-specific code.

The dialect depends on MLIR's `affine` and `vector` dialects. That is visible in the operation design: `iree_vector_ext.transfer_gather` and `iree_vector_ext.transfer_scatter` use affine maps to describe indexing, and the lowering passes eventually rewrite custom operations into ordinary `vector` plus `scf` and `arith` operations where possible.

## Why This Dialect Exists

The main reason for `iree_vector_ext` is codegen experimentation without forcing every idea into upstream MLIR immediately. IREE needs to represent layout conversions, SIMT/SIMD boundary materializations, vector gathers and scatters with richer indexing maps, and arg-reduction patterns. Some of those constructs are temporary; others encode IREE-specific hardware mapping knowledge.

This matters because vector codegen is not only about element type and shape. On GPU-like targets, a vector may be distributed across subgroups and threads. The compiler needs to remember how a logical value is packed, distributed, and later reassembled. The `#iree_vector_ext.nested_layout` attribute is the main layout object for that purpose. It describes a vector through levels such as subgroup tile, batch tile, outer tile, thread tile, element tile, subgroup strides, and thread strides.

The dialect is also useful because it delays a lowering decision. For example, `iree_vector_ext.transfer_gather` can describe a gather from a tensor or memref where some dimensions are contiguous, some are gathered by index vectors, and some are broadcast. Later, a pass can rewrite the simple cases to `vector.gather`, `vector.transfer_read`, or smaller unrolled forms.

## When To Use It

Use `iree_vector_ext` when reading IREE's codegen pipeline around vectorization, GPU distribution, or vector transfer lowering. You will usually encounter it after tensor and linalg-level code has been tiled or vectorized, but before final target-specific vector lowering.

It is especially important when a design needs explicit vector layouts. `iree_vector_ext.to_layout` says that a shaped value should be interpreted with a particular layout. The layout may describe how an undistributed vector maps to subgroups and threads, or whether a conversion requires shared memory.

It is also important around indexed access patterns. Standard `vector.transfer_read` and `vector.transfer_write` cover many contiguous and permuted transfers, and standard `vector.gather` and `vector.scatter` cover some indexed cases. `iree_vector_ext.transfer_gather` and `iree_vector_ext.transfer_scatter` give IREE a richer intermediate form while it decides whether an access can lower to one of those standard operations.

Avoid using this dialect as a beginner-facing source IR. It is meant for compiler internals. If you are writing high-level programs, you should expect IREE to introduce these operations for you.

## Core Concepts

The first concept is layout. A `#iree_vector_ext.nested_layout` attribute maps a vector shape to a hierarchy inspired by GPU execution. It can talk about subgroups per workgroup, threads per subgroup, and elements per thread. The layout is not just a comment; `iree_vector_ext.to_layout` verifies that the layout is valid for the shaped value.

The second concept is conversion between execution views. `iree_vector_ext.to_simd` and `iree_vector_ext.to_simt` are temporary materialization operations for conversions between distributed and non-distributed vectors. They preserve element type, and their folders cancel back-to-back inverse conversions.

The third concept is indexed vector transfer. `iree_vector_ext.transfer_gather` reads a vector from a shaped source. `iree_vector_ext.transfer_scatter` writes a vector into a shaped destination. Both use an `indexing_maps` array. The first map describes how vector dimensions and index-vector symbols map to the base tensor or memref. Later maps describe how each index vector is indexed from the vector iteration space. An optional final map describes the mask.

The fourth concept is vectorized arg-reduction. `iree_vector_ext.arg_compare` reduces a vector along one dimension and returns both the selected value and the selected index. The user supplies a comparator region, and that region must yield one `i1` using `iree_vector_ext.yield`.

## Operation Inventory

| Operation | What It Means |
| --- | --- |
| `iree_vector_ext.to_layout` | Converts a shaped value to a requested vector layout. The operand and result types match, but the layout attribute records the intended distribution or packing. An optional `shared_memory_conversion` attribute marks conversions that must be materialized through shared memory. |
| `iree_vector_ext.to_simd` | Materializes a conversion from a SIMT/distributed vector view to a SIMD/non-distributed vector view. It is a temporary conversion marker used during type conversion. |
| `iree_vector_ext.to_simt` | Materializes a conversion from a SIMD/non-distributed vector view to a SIMT/distributed vector view. It is the inverse marker for `iree_vector_ext.to_simd`. |
| `iree_vector_ext.transfer_gather` | Reads a supervector from a tensor or memref. Each base dimension can be contiguous, gathered through an index vector, or broadcast, as described by affine indexing maps. |
| `iree_vector_ext.transfer_scatter` | Writes a supervector into a tensor or memref. It is the write-side counterpart of `iree_vector_ext.transfer_gather`; for tensor bases it returns the updated tensor, and for memref bases it writes in place and has no result. |
| `iree_vector_ext.arg_compare` | Performs a vectorized arg-reduction along one dimension. It returns the chosen values and the corresponding indices. The comparator region decides whether the new candidate wins over the current accumulator. |
| `iree_vector_ext.yield` | Terminates the comparator region inside `iree_vector_ext.arg_compare`. It must yield exactly one `i1` predicate. |

## Transformations And Conversions

| Pass or Pattern | Role |
| --- | --- |
| `iree-vector-ext-fold-unit-extent-dims` | Folds unit dimensions around `iree_vector_ext.to_layout` for tensor values. It uses rank-reducing tensor slices, projects the layout, applies the reduced layout conversion, and expands the result back to the original shape. |
| `iree-vector-ext-lower-transfer-gather-scatter-to-vector` | Lowers supported `iree_vector_ext.transfer_gather` and `iree_vector_ext.transfer_scatter` operations to `vector.gather` and `vector.scatter`. The current lowering expects exactly one index vector, constants for non-gathered dimensions, and a pure symbol expression in the final gathered dimension. |
| `iree-vector-ext-lower-arg-compare-to-vector` | Lowers `iree_vector_ext.arg_compare` to `vector`, `scf`, and `arith` operations. It clones the comparator region, loops over the reduction dimension, and updates value/index accumulators with `arith.select`. |
| `populateVectorTransferGatherScatterLoweringPatterns` | Pattern helper, not a standalone pass name. It unrolls high-rank transfer gather/scatter operations along dimension 0 until they are easier to lower. |
| `populateLowerTransferGatherScatterToVectorPatterns` | Pattern helper used by the gather/scatter lowering pass. It rewrites eligible transfer operations to standard vector gather/scatter operations. |
| `populateLowerArgCompareToVectorPatterns` | Pattern helper used by the arg-compare lowering pass. It rewrites arg-compare operations into standard vector, SCF, and arithmetic IR. |

## Indexed Transfer Semantics

The most important operations for this dialect are the indexed transfers. Their `indexing_maps` attribute is the part that beginners should learn slowly.

For `iree_vector_ext.transfer_gather`, the base operand is the shaped source. The offsets give the starting point. The result is a vector. The first affine map maps result-vector dimensions and index-vector symbols to source dimensions. A dimension expression means "walk this source dimension along with the vector." A symbol expression means "look up this source dimension through an index vector." A constant zero means "broadcast this source dimension from the base offset."

For `iree_vector_ext.transfer_scatter`, the same structure applies, but the base is the destination and the vector operand is the data being written. The scatter indices are expected to be unique. If multiple vector elements map to the same destination location, the operation has undefined behavior because the write has a data race.

The verifier checks that the number of maps matches the number of index vectors and optional mask, that all maps use the vector rank as their dimension count, and that symbol counts match the number of index vectors. Index-vector maps and mask maps must describe shapes compatible with their operands.

Several canonicalizations simplify transfers before lowering. All-true masks can be removed. All-false gather masks can become a broadcast of the padding value. All-false scatter masks can become the original tensor or disappear for memref writes. Contiguous gathers can become `vector.transfer_read`; contiguous scatters can become `vector.transfer_write`.

## Arg Compare Semantics

`iree_vector_ext.arg_compare` is IREE's vectorized form of an arg-reduction. It takes an input value vector, optional explicit index vector, initial value accumulator, initial index accumulator, an optional `index_base`, and a reduction dimension. It returns two vectors: the reduced values and the selected indices.

In implicit-index mode, the operation computes candidate indices from the reduction loop induction variable. If `index_base` is present, it adds that base to the computed index. In explicit-index mode, the operation reads candidate indices from the `input_index` vector, and `index_base` is not allowed.

The comparator region receives two scalar values of the input element type. It yields an `i1`. When the predicate is true, the candidate value wins; otherwise the accumulator remains. The verifier requires the comparator region to have exactly two arguments of the input element type, to yield one `i1`, and to contain only pure operations.

The lowering to `vector`, `scf`, and `arith` moves the reduction dimension to the last position when needed, fully unrolls parallel dimensions at compile time, uses an `scf.for` for the reduction dimension, and updates both value and index accumulators with `arith.select`.

## Layout Semantics

The `#iree_vector_ext.nested_layout` attribute is not an operation, but it is central to the dialect. It describes how an undistributed vector shape is packed into a hierarchy. The key fields are subgroup tile, batch tile, outer tile, thread tile, element tile, subgroup strides, and thread strides.

This layout matters for GPU codegen because a logical vector may be split across subgroups and threads. The compiler needs to know which pieces each lane owns and how to recombine layouts after transformations. The `VectorLayoutInterface` exposes operations such as validation, permutation, projection, reshape, and recombination from operand layouts and indexing maps.

`iree_vector_ext.to_layout` is the operation that attaches this layout to a shaped value. If the value has tensor semantics, `iree-vector-ext-fold-unit-extent-dims` can temporarily drop unit dimensions, project the layout, and then rebuild the original shape. This is useful because many vector layouts become simpler after dimensions of size one are removed.

## How To Read It In IREE IR

Start by finding `iree_vector_ext.to_layout`. That tells you where IREE is assigning a hardware-aware vector layout. Read the layout attribute before reading the surrounding arithmetic; otherwise the vector shape alone may be misleading.

Next, look for `iree_vector_ext.to_simd` and `iree_vector_ext.to_simt`. These are usually materialization artifacts around type conversion. If they appear back to back as inverse conversions, canonicalization can often fold them away.

Then inspect transfer operations. If a transfer has no index vectors and only contiguous maps, expect it to become `vector.transfer_read` or `vector.transfer_write`. If it has one suitable index vector, expect the named lowering pass to target `vector.gather` or `vector.scatter`. If it has multiple index vectors or higher-rank structure, expect unrolling and other cleanup patterns first.

Finally, inspect `iree_vector_ext.arg_compare` regions as reductions. The operation is not just comparing two vectors. It reduces along a selected dimension, carries an index accumulator, and uses the region as a scalar comparator.

## What It Implies

`iree_vector_ext` implies that the compiler is in a late, codegen-oriented part of the pipeline. It is no longer just proving tensor semantics; it is deciding how vector work maps onto target execution. Seeing this dialect usually means layout, gather/scatter legality, masks, and target lowering constraints are now relevant.

It also shows a common MLIR design pattern: use a project dialect to carry precise intermediate meaning, then lower back to standard dialects when the operation becomes expressible there. In this case, the final destination is often standard `vector`, `scf`, and `arith` IR, followed by the normal IREE target codegen pipeline.

