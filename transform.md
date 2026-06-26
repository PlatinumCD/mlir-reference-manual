# Transform Dialect

## Beginner Summary

The `transform` dialect is a dialect for controlling compiler transformations.

Most MLIR dialects describe the program being compiled. The Transform dialect
describes how to transform that program. The program being transformed is called
the payload IR. The Transform dialect script is called transform IR.

A transform script can find operations, keep handles to them, apply patterns,
run focused transformations, compose alternatives, and call named transform
sequences. It is useful when a normal pass is too broad and a rewrite pattern is
too local.

## Why This Dialect Exists

MLIR already has passes and rewrite patterns. The Transform dialect fills the
space between them.

- A rewrite pattern is usually local: match one operation shape and rewrite it.
- A pass is usually broad: walk an IR scope and transform everything matching
  its rules.
- Transform IR is programmable orchestration: pick specific payload operations,
  transform them, use the new handles, and continue.

This is especially important for optimization pipelines where order and target
selection matter. For example, a transform script can match one `linalg.matmul`,
tile it, vectorize the tiled body, apply cleanup patterns, and then lower only
the selected pieces.

## When It Matters

The Transform dialect matters when:

- You need to script a sequence of transformations in IR.
- You want transformations to be reproducible and inspectable.
- You need fine-grained control over which payload ops are transformed.
- You are tuning linear algebra, vector, GPU, loop, or bufferization pipelines.
- You need a reusable transform library with named transform sequences.
- You want to debug why a transform handle became invalid.

It is often used in advanced MLIR pipelines, especially for structured
operations and hardware-specific lowering strategies.

## When To Use It

Use Transform IR when a pass pipeline does not give you enough control. It is a
good fit when you want to say:

1. Find these operations.
2. Apply this transformation to exactly those operations.
3. Use the produced handles for later transformations.
4. Try another plan if this plan fails.

Do not use it as a replacement for the pass manager or pattern infrastructure.
Transform ops usually call into pass, pattern, analysis, or utility code. The
dialect provides the control layer, not every transformation implementation by
itself.

## Core Concepts

### Payload IR And Transform IR

Payload IR is the IR being transformed. Transform IR is the script that controls
the transformation.

The interpreter starts with a payload root operation and a top-level transform
operation. The first block argument of that transform operation is associated
with the payload root.

### Handles

Transform operation handles point to payload operations. A handle can point to
zero, one, or many payload operations.

Important handle types include:

- `!transform.any_op`: handle to any payload operation.
- `!transform.op<"dialect.op">`: handle restricted to a specific payload op
  name.
- `!transform.normalized_op<...>`: handle with additional normal-form
  guarantees.

Value handles point to payload SSA values:

- `!transform.any_value`: handle to payload values.

### Parameters

Parameters are values known to the transform interpreter rather than ordinary
payload SSA values. They are represented as attributes under the hood.

Important parameter types include:

- `!transform.any_param`: a parameter of any supported kind.
- `!transform.param<i64>` or another integer type: typed integer parameters.
- `!transform.affine_map`: affine map parameters.
- `!transform.type`: type parameters.

### Named Sequences

`transform.named_sequence` defines a callable transform function. The default
entry point is named `@__transform_main`.

A module containing named sequences is marked with:

```text
transform.with_named_sequence
```

The interpreter pass looks for the entry point and applies it to the payload IR.

### Failure Model

Transform application has three outcomes:

- Success.
- Recoverable, or silenceable, failure.
- Irrecoverable failure.

Container operations such as `transform.sequence` and
`transform.alternatives` decide whether recoverable failures are propagated,
suppressed, or used to try another region.

### Handle Invalidation

A transform op that mutates payload IR often consumes the handles it uses. After
a handle is consumed, other handles that may point into the modified or erased
payload subtree can become invalid. Using an invalidated handle is like using a
dangling pointer.

This is why many transform ops return new handles. Use the returned handles for
later steps instead of reusing consumed handles.

### Extensions

The core Transform dialect intentionally does not depend on every dialect it
can transform. Other dialects inject transform ops through the
`TransformDialectExtension` mechanism.

That is why this chapter has two layers:

- Core Transform dialect ops such as `transform.sequence`,
  `transform.apply_patterns`, and `transform.select`.
- Extension ops such as `transform.structured.tile_using_for`,
  `transform.loop.unroll`, and `transform.apply_patterns.vector.lower_transfer`.

## Core Operation Groups

The core Transform dialect has 37 operations.

### Control And Structure

| Operation | Purpose |
| --- | --- |
| `transform.sequence` | Applies nested transform ops in order. |
| `transform.alternatives` | Tries alternative transform regions until one succeeds. |
| `transform.foreach` | Iterates over associations in handles or params. |
| `transform.foreach_match` | Applies actions for matching named sequences. |
| `transform.named_sequence` | Defines a callable named transform sequence. |
| `transform.include` | Includes/calls another named sequence. |
| `transform.yield` | Yields values from transform regions. |

### Finding And Navigating Payload IR

| Operation | Purpose |
| --- | --- |
| `transform.select` | Selects nested payload operations by operation name. |
| `transform.collect_matching` | Collects payload ops accepted by a matcher sequence. |
| `transform.match.operation_name` | Checks that a handle is associated with operations with given names. |
| `transform.match.operation_empty` | Checks that a payload operation is empty. |
| `transform.get_parent_op` | Gets parent payload operations. |
| `transform.get_defining_op` | Gets defining operations for payload values. |
| `transform.get_consumers_of_result` | Gets consumers of a payload operation result. |
| `transform.get_producer_of_operand` | Gets a producer of a payload operation operand. |
| `transform.get_operand` | Gets payload operands as value handles. |
| `transform.get_result` | Gets payload results as value handles. |
| `transform.get_type` | Gets type parameters from payload values. |

### Handle And Parameter Utilities

| Operation | Purpose |
| --- | --- |
| `transform.cast` | Refines or relaxes a transform handle type. |
| `transform.merge_handles` | Merges multiple handles, optionally deduplicating. |
| `transform.split_handle` | Splits one handle into multiple handles. |
| `transform.replicate` | Replicates associations to match another handle. |
| `transform.payload` | Gets payload information from transform IR. |
| `transform.num_associations` | Counts associations in a handle. |
| `transform.param.constant` | Creates a constant transform parameter. |
| `transform.match.param.cmpi` | Compares integer parameters. |

### Applying Transformations

| Operation | Purpose |
| --- | --- |
| `transform.apply_patterns` | Applies a set of rewrite pattern descriptors. |
| `transform.apply_patterns.canonicalization` | Adds canonicalization patterns to an apply-patterns region. |
| `transform.apply_conversion_patterns` | Applies dialect conversion patterns. |
| `transform.apply_conversion_patterns.dialect_to_llvm` | Adds dialect-to-LLVM conversion patterns through an interface. |
| `transform.apply_cse` | Runs CSE on the target. |
| `transform.apply_dce` | Runs dead code elimination on the target. |
| `transform.apply_licm` | Runs loop-invariant code motion on the target. |
| `transform.apply_registered_pass` | Runs a registered pass by name. |
| `transform.annotate` | Adds an attribute to target payload operations. |
| `transform.verify` | Verifies the target payload operation. |
| `transform.print` | Prints transform or payload information for debugging. |

## Extension Operation Inventory

This checkout exposes 233 unique `transform.*` operations across the core
dialect and its registered extensions. Core ops were grouped above. The
extension ops are grouped below by the TableGen source that defines them.

### Debug, Tune, SMT, PDL, IRDL

- Debug: `transform.debug.emit_param_as_remark`,
  `transform.debug.emit_remark_at`.
- Tune: `transform.tune.alternatives`, `transform.tune.knob`.
- SMT: `transform.smt.constrain_params`.
- PDL: `transform.pdl_match`, `transform.with_pdl_patterns`.
- IRDL: `transform.irdl.collect_matching`.

### Affine And DLTI

- Affine: `transform.affine.simplify_bounded_affine_ops`,
  `transform.affine.simplify_min_max_affine_ops`,
  `transform.affine.super_vectorize`.
- DLTI: `transform.dlti.query`.

### Arm Neon, Arm SVE, And X86 Patterns

- Arm Neon: `transform.apply_patterns.arm_neon.vector_contract_to_bfmmla`,
  `transform.apply_patterns.arm_neon.vector_contract_to_i8mm`.
- Arm SVE: `transform.apply_patterns.arm_sve.vector_contract_to_bfmmla`,
  `transform.apply_patterns.arm_sve.vector_contract_to_i8mm`.
- X86: `transform.apply_patterns.x86.shuffle_vector_fma_ops`,
  `transform.apply_patterns.x86.sink_vector_producer_ops`,
  `transform.apply_patterns.x86.vector_contract_bf16_to_fma`,
  `transform.apply_patterns.x86.vector_contract_to_amx_dot_product`,
  `transform.apply_patterns.x86.vector_contract_to_fma`,
  `transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product`.

### Bufferization

- `transform.bufferization.buffer_loop_hoisting`,
  `transform.bufferization.eliminate_empty_tensors`,
  `transform.bufferization.empty_tensor_to_alloc_tensor`,
  `transform.bufferization.one_shot_bufferize`.

### Func

- `transform.apply_conversion_patterns.func.func_to_llvm`,
  `transform.func.cast_and_call`,
  `transform.func.deduplicate_func_args`,
  `transform.func.replace_func_signature`.

### GPU And NVGPU

- GPU conversion/pattern ops:
  `transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm`,
  `transform.apply_conversion_patterns.gpu.gpu_to_nvvm`,
  `transform.apply_conversion_patterns.gpu.gpu_to_rocdl`,
  `transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm`,
  `transform.apply_patterns.gpu.eliminate_barriers`,
  `transform.apply_patterns.gpu.gpu_rewrite_patterns`,
  `transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu`,
  `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma`.
- GPU mapping ops: `transform.gpu.map_forall_to_blocks`,
  `transform.gpu.map_nested_forall_to_threads`.
- NVGPU: `transform.apply_conversion_patterns.nvgpu.nvgpu_to_nvvm`,
  `transform.nvgpu.create_async_groups`,
  `transform.nvgpu.pipeline_shared_memory_copies`,
  `transform.nvgpu.rewrite_copy_as_tma`,
  `transform.nvgpu.rewrite_matmul_as_mma_sync`.

### Linalg Match Operations

- `transform.match.structured`, `transform.match.structured.body`,
  `transform.match.structured.classify_contraction_dims`,
  `transform.match.structured.classify_convolution_dims`,
  `transform.match.structured.dim`,
  `transform.match.structured.elemental_bitwidth`,
  `transform.match.structured.init`, `transform.match.structured.input`,
  `transform.match.structured.num_inits`,
  `transform.match.structured.num_inputs`, `transform.match.structured.rank`,
  `transform.match.structured.result`, `transform.match.structured.yield`.

### Linalg Structured Transform Operations

- Pattern descriptors: `transform.apply_patterns.linalg.data_layout_propagation`,
  `transform.apply_patterns.linalg.decompose_pack_unpack`,
  `transform.apply_patterns.linalg.decompose_pad`,
  `transform.apply_patterns.linalg.erase_unnecessary_inputs`,
  `transform.apply_patterns.linalg.extract_slice_sinking`,
  `transform.apply_patterns.linalg.fold_add_into_dest`,
  `transform.apply_patterns.linalg.fold_pack_unpack_into_empty`,
  `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes`,
  `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices`,
  `transform.apply_patterns.linalg.pad_vectorization`,
  `transform.apply_patterns.linalg.tiling_canonicalization`,
  `transform.apply_patterns.tensor.fold_into_pack_and_unpack`.
- Structured transformations: `transform.structured.bufferize_to_allocation`,
  `transform.structured.continuous_tile_sizes`,
  `transform.structured.convert_conv2d_to_img2col`,
  `transform.structured.convert_to_loops`, `transform.structured.decompose`,
  `transform.structured.decompose_interface`,
  `transform.structured.decompose_winograd_op`,
  `transform.structured.eliminate_empty_tensors`,
  `transform.structured.flatten_elementwise`, `transform.structured.fuse`,
  `transform.structured.fuse_into_containing_op`,
  `transform.structured.generalize`,
  `transform.structured.gpu.map_copy_to_threads`,
  `transform.structured.hoist_pad`,
  `transform.structured.hoist_pad.build_packing_loop_nest`,
  `transform.structured.hoist_redundant_vector_broadcasts`,
  `transform.structured.hoist_redundant_vector_transfers`,
  `transform.structured.insert_slice_to_copy`,
  `transform.structured.interchange`,
  `transform.structured.linalg_copy_to_memref`,
  `transform.structured.lower_pack`, `transform.structured.lower_unpack`,
  `transform.structured.match`, `transform.structured.multitile_sizes`,
  `transform.structured.pack`, `transform.structured.pack_greedily`,
  `transform.structured.pack_transpose`, `transform.structured.pad`,
  `transform.structured.pad_tiling_interface`,
  `transform.structured.promote`, `transform.structured.promote_tensor`,
  `transform.structured.replace`,
  `transform.structured.rewrite_in_destination_passing_style`,
  `transform.structured.scalarize`, `transform.structured.specialize`,
  `transform.structured.split`, `transform.structured.split_reduction`,
  `transform.structured.tile_reduction_using_for`,
  `transform.structured.tile_reduction_using_forall`,
  `transform.structured.tile_using_for`,
  `transform.structured.tile_using_forall`,
  `transform.structured.transpose_conv2d`,
  `transform.structured.transpose_matmul`,
  `transform.structured.vectorize`,
  `transform.structured.vectorize_children_and_apply_patterns`,
  `transform.structured.winograd_conv2d`.

### Loop And SCF

- Loop extension: `transform.loop.hoist_loop_invariant_subsets`.
- SCF conversion/pattern ops:
  `transform.apply_conversion_patterns.scf.scf_to_control_flow`,
  `transform.apply_conversion_patterns.scf.structural_conversions`,
  `transform.apply_patterns.scf.for_loop_canonicalization`.
- Loop transforms: `transform.loop.coalesce`,
  `transform.loop.coalesce_nested`, `transform.loop.forall_to_for`,
  `transform.loop.forall_to_parallel`, `transform.loop.fuse_sibling`,
  `transform.loop.outline`, `transform.loop.parallel_for_to_nested_fors`,
  `transform.loop.peel`, `transform.loop.pipeline`,
  `transform.loop.promote_if_one_iteration`, `transform.loop.unroll`,
  `transform.loop.unroll_and_jam`.
- SCF transform: `transform.scf.take_assumed_branch`.

### MemRef

- Conversion and patterns:
  `transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter`,
  `transform.apply_patterns.memref.alloc_to_alloca`,
  `transform.apply_patterns.memref.expand_ops`,
  `transform.apply_patterns.memref.expand_strided_metadata`,
  `transform.apply_patterns.memref.extract_address_computations`,
  `transform.apply_patterns.memref.fold_memref_alias_ops`,
  `transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims`.
- MemRef transforms: `transform.memref.alloca_to_global`,
  `transform.memref.erase_dead_alloc_and_stores`,
  `transform.memref.make_loop_independent`, `transform.memref.multibuffer`.

### Sparse Tensor And Tensor

- Sparse tensor: `transform.sparse_tensor.match.sparse_inout`.
- Tensor patterns: `transform.apply_patterns.tensor.bubble_up_extract_slice`,
  `transform.apply_patterns.tensor.decompose_concat`,
  `transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion`,
  `transform.apply_patterns.tensor.fold_tensor_empty`,
  `transform.apply_patterns.tensor.fold_tensor_subset_ops`,
  `transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers`,
  `transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice`,
  `transform.apply_patterns.tensor.reassociative_reshape_folding`,
  `transform.apply_patterns.tensor.rewrite_as_constant`.
- Tensor transforms: `transform.tensor.make_loop_independent`,
  `transform.type_conversion.tensor.cast_shape_dynamic_dims`.

### Vector

- Conversion: `transform.apply_conversion_patterns.vector.vector_to_llvm`.
- Pattern descriptors:
  `transform.apply_patterns.vector.cast_away_vector_leading_one_dim`,
  `transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops`,
  `transform.apply_patterns.vector.drop_unit_dims_with_shape_cast`,
  `transform.apply_patterns.vector.elementwise_to_vector`,
  `transform.apply_patterns.vector.flatten_vector_transfer_ops`,
  `transform.apply_patterns.vector.fold_arith_extension`,
  `transform.apply_patterns.vector.interleave_and_deinterleave_to_shuffle`,
  `transform.apply_patterns.vector.lower_bitcast`,
  `transform.apply_patterns.vector.lower_broadcast`,
  `transform.apply_patterns.vector.lower_contraction`,
  `transform.apply_patterns.vector.lower_create_mask`,
  `transform.apply_patterns.vector.lower_gather`,
  `transform.apply_patterns.vector.lower_interleave`,
  `transform.apply_patterns.vector.lower_masked_transfers`,
  `transform.apply_patterns.vector.lower_masks`,
  `transform.apply_patterns.vector.lower_outerproduct`,
  `transform.apply_patterns.vector.lower_scan`,
  `transform.apply_patterns.vector.lower_shape_cast`,
  `transform.apply_patterns.vector.lower_transfer`,
  `transform.apply_patterns.vector.lower_transpose`,
  `transform.apply_patterns.vector.materialize_masks`,
  `transform.apply_patterns.vector.multi_reduction_flattening`,
  `transform.apply_patterns.vector.multi_reduction_unrolling`,
  `transform.apply_patterns.vector.rank_reducing_subview_patterns`,
  `transform.apply_patterns.vector.reduction_to_contract`,
  `transform.apply_patterns.vector.reorder_multi_reduction_dims`,
  `transform.apply_patterns.vector.rewrite_narrow_types`,
  `transform.apply_patterns.vector.sink_mem_ops`,
  `transform.apply_patterns.vector.sink_ops`,
  `transform.apply_patterns.vector.split_transfer_full_partial`,
  `transform.apply_patterns.vector.transfer_permutation_patterns`,
  `transform.apply_patterns.vector.transfer_to_scf`,
  `transform.apply_patterns.vector.unroll_from_elements`,
  `transform.apply_patterns.vector.unroll_to_elements`.

### XeGPU

- `transform.xegpu.convert_layout`, `transform.xegpu.get_load_op`,
  `transform.xegpu.insert_prefetch`, `transform.xegpu.set_anchor_layout`,
  `transform.xegpu.set_gpu_launch_threads`.

## Transformations

The Transform dialect is itself the transformation control layer. Its important
runtime transformation is interpretation:

```text
transform IR + payload IR
  -> transform interpreter
  -> modified payload IR
```

The main transform passes are:

| Pass | Purpose |
| --- | --- |
| `-transform-interpreter` | Runs a transform entry point, defaulting to `@__transform_main`. |
| `-transform-dialect-check-uses` | Warns about potential use-after-free of transform handles. |
| `-transform-infer-effects` | Infers side-effect attributes on transform named sequence arguments. |
| `-transform-preload-library` | Loads transform library modules for later interpreter use. |

Important `-transform-interpreter` options include:

- `entry-point`: choose a named sequence other than `@__transform_main`.
- `debug-payload-root-tag`: choose the payload root with a
  `transform.target_tag` attribute.
- `debug-bind-trailing-args`: bind extra entry point arguments for debugging.
- `disable-expensive-checks`: skip expensive interpreter checks for speed.

## Conversions And Lowering Paths

The Transform dialect is not normally lowered to target code. It is consumed by
the transform interpreter.

That said, Transform IR can drive conversions of payload IR:

- `transform.apply_conversion_patterns` runs dialect conversion on a payload
  target using conversion pattern descriptor ops.
- `transform.apply_conversion_patterns.dialect_to_llvm` contributes
  dialect-to-LLVM patterns for dialects that implement the relevant interface.
- Extension ops provide conversion pattern descriptors for Func, GPU, MemRef,
  NVGPU, SCF, Vector, and other domains.

So the transform IR is not the object being converted. The payload IR is.

## Example IR

### A Simple Sequence

```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %funcs = transform.select "func.func" in %root
    : (!transform.any_op) -> !transform.any_op
  transform.apply_cse to %funcs : !transform.any_op
}
```

This sequence selects all nested `func.func` operations under the payload root
and runs CSE inside them.

### A Named Entry Point

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %funcs = transform.select "func.func" in %root
      : (!transform.any_op) -> !transform.any_op
    transform.print %funcs {name = "functions"} : !transform.any_op
    transform.yield
  }
}
```

This is the shape expected by the default interpreter entry point.

### Applying Pattern Descriptors

```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  transform.apply_patterns to %root {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
```

The nested operation describes which patterns to populate. The outer
`transform.apply_patterns` applies them to the payload target.

### Trying Alternatives

```mlir
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  %chosen = transform.alternatives %root
    : !transform.any_op -> !transform.any_op {
  ^bb1(%scope: !transform.any_op):
    %loops = transform.select "scf.for" in %scope
      : (!transform.any_op) -> !transform.any_op
    transform.yield %loops : !transform.any_op
  }, {
  ^bb1(%scope: !transform.any_op):
    transform.yield %scope : !transform.any_op
  }
  transform.print %chosen : !transform.any_op
}
```

The first alternative tries to select loops. The second alternative falls back
to the original scope.

### Using A Linalg Extension Op

```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op
  %tiled:3 = transform.structured.tile_using_for %matmuls
    tile_sizes [8, 8]
    : (!transform.any_op) -> (!transform.any_op,
                              !transform.any_op,
                              !transform.any_op)
}
```

This is a typical Transform dialect pattern: match a structured payload op and
then use an extension op to transform it.

## Mental Model

Think of Transform IR as a script over handles:

1. The interpreter binds a handle to the payload root.
2. Match/navigation ops create more handles.
3. Transform ops consume handles and mutate payload IR.
4. Transform ops return new handles for follow-up transformations.
5. The script either succeeds, produces a recoverable failure that a container
   may handle, or fails irrecoverably.

The payload program does not execute the transform IR. The compiler executes
the transform IR to rewrite the payload program.

## Gotchas

- A handle may refer to multiple payload objects, so many transform ops execute
  in batches.
- A handle may refer to zero payload objects; navigation failure is often not an
  interpreter failure.
- Mutating transforms can invalidate handles. Prefer returned handles after a
  mutating operation.
- The core dialect is small compared with its extension ecosystem. If an op
  name starts with `transform.structured`, `transform.loop`,
  `transform.vector`, or another domain prefix, it likely comes from an
  extension.
- `transform.apply_registered_pass` is powerful but can reduce the benefit of
  fine-grained transform scripting if used as a broad pass pipeline escape
  hatch.
- `transform.with_named_sequence`, `transform.readonly`,
  `transform.consumed`, and `transform.target_tag` are important attributes for
  interpreter setup and verification.
- Transform scripts are compiler control IR. They are not part of the final
  executable program.

## Source Map

Important source files in the LLVM tree:

- `mlir/docs/Dialects/Transform.md` is the main conceptual documentation.
- `mlir/include/mlir/Dialect/Transform/IR/TransformDialect.td` defines the
  dialect and key dialect attributes.
- `mlir/include/mlir/Dialect/Transform/IR/TransformTypes.td` defines transform
  handle, value-handle, and parameter types.
- `mlir/include/mlir/Dialect/Transform/IR/TransformOps.td` defines the core
  operations.
- `mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.td`
  defines the execution and handle interfaces.
- `mlir/include/mlir/Dialect/Transform/Transforms/Passes.td` declares the
  interpreter and analysis passes.
- `mlir/lib/Dialect/Transform/Transforms/InterpreterPass.cpp` implements the
  interpreter pass.
- `mlir/lib/Dialect/Transform/IR/TransformOps.cpp` implements core operation
  behavior.
- `mlir/include/mlir/Dialect/*/TransformOps/*.td` defines dialect-specific
  transform extensions.
- `mlir/include/mlir/Dialect/Transform/*Extension/*Ops.td` defines generic
  Transform dialect extensions.
- `mlir/test/Dialect/Transform/` contains core interpreter, verification, and
  extension tests.
