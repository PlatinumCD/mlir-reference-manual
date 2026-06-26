# IREE Codegen Dialect

The IREE `iree_codegen` dialect is the shared control dialect for IREE code generation. It does not model a user program by itself. Instead, it records the decisions that drive lowering from dispatch-level tensor code into CPU, GPU, SPIR-V, VMVX, or other backend-specific forms.

For a beginner, the key idea is that code generation has two different kinds of IR. The first kind is computation: `linalg.matmul`, `vector.transfer_read`, `scf.forall`, `memref.load`, and so on. The second kind is policy: which pipeline should lower this dispatch, what tile sizes should be used, what workgroup size should be requested, whether a microkernel should be called, and what swizzle or tuning hints should guide later passes. `iree_codegen` mostly represents the second kind.

This dialect is especially important in IREE because backend heuristics are intentionally staged. A target may select a lowering configuration early, attach it as attributes, and then later passes simply follow those attributes. That makes the compiler easier to debug and makes external tuning possible: changing an attribute can change the generated kernel without rewriting the whole pipeline.

## When Codegen Is Important

Use `iree_codegen` when reading IR inside `hal.executable.variant`, executable entry points, dispatch workgroup functions, or backend-specific lowering tests. If a `linalg` op has a `lowering_config`, or an export has `translation_info`, the compiler has already made important lowering choices.

This dialect is important when you need to answer questions such as:

- Which pass pipeline will lower this dispatch?
- What tile sizes does the compiler intend to use?
- What workgroup size and subgroup size are expected?
- Does this operation lower to a microkernel?
- Are workgroup ids or local ids mapped to loop dimensions?
- Is a swizzle, index hint, or SMT tuning constraint influencing lowering?

You usually do not write this dialect as application IR. You see it after IREE has formed dispatches and selected target-specific codegen strategies.

## Why It Is Needed

Without `iree_codegen`, every backend pass would have to rediscover lowering decisions from scratch. That would make the pipeline harder to reason about and harder to tune. IREE instead records decisions in attributes such as `#iree_codegen.translation_info`, `#iree_codegen.lowering_config`, and `#iree_codegen.compilation_info`.

The dialect also gives common names to codegen helper operations. A backend can emit `iree_codegen.inner_tiled` to represent an operation organized around an intrinsic tile. It can use `iree_codegen.workgroup_count_hint` to communicate desired workgroup counts. It can wrap a microkernel call with `iree_codegen.ukernel.generic` before lowering that wrapper to an actual function call.

The result is a separation of concerns. Selection passes decide. Lowering passes implement. Tuning tools can override or verify the decisions.

## Type Inventory

The dialect defines one type:

- `!iree_codegen.null_pointer` is a pseudo null-pointer type. It is meant only for microkernel arguments and lowers to a null pointer.

The corresponding operation is `iree_codegen.null_pointer`, which produces a value of this type.

## Attribute Inventory

The dialect contains many attributes because most of its meaning is configuration.

- `#iree_codegen.workgroup_mapping<x>`, `#iree_codegen.workgroup_mapping<y>`, and `#iree_codegen.workgroup_mapping<z>` map distributed loop dimensions to HAL workgroup ids. The `z` mapping can be delinearized with a secondary dimension.
- `#iree_codegen.local_mapping<n>` maps an `scf.forall` dimension to sequential local execution.
- `#iree_codegen.workgroup_local` is a memory space for per-workgroup local memory. CPU lowering can pack these allocations and report the required local memory size.
- `#iree_codegen.simple_target` is a basic target-info attribute, including optional maximum workgroup count.
- `#iree_codegen.vmvx_pipeline`, `#iree_codegen.transform_dialect_codegen`, `#iree_codegen.no_pipeline`, and `#iree_codegen.pass_pipeline<"...">` are pipeline selectors. The last one carries an MLIR textual pass pipeline.
- `#iree_codegen.translation_info` is attached to executable exports and says which pipeline, optional transform dialect spec, workgroup size, subgroup size, and configuration should be used.
- `#iree_codegen.lowering_config_level` describes one tiling level. `#iree_codegen.lowering_config_levels` groups levels. `#iree_codegen.lowering_config` carries the tiling configuration for an operation.
- `#iree_codegen.compilation_info` combines a lowering config and translation info for frontend or root operations before the data is propagated to executable exports.
- `#iree_codegen.export_config` lets pre-formed dispatches request a workgroup size.
- `#iree_codegen.rotate_rows` and `#iree_codegen.xor_shuffle` describe memory access swizzles used by `iree_codegen.swizzle_hint`.
- `#iree_codegen.symbolic_ukernel_provider` resolves microkernel implementations from nearby symbols.
- `#iree_codegen.ukernel_descriptor` identifies a microkernel by name and says whether it integrates at tensor, memref, or bitcode level.
- `#iree_codegen.denormal_fp_math` records denormal floating-point behavior such as none, preserve-sign flush, or positive-zero flush.
- `#iree_codegen.workgroup_scope` represents parallel execution across workgroups and can request linearized workgroup ids.
- `#iree_codegen.root_op` groups root operations for tuning constraints.
- `#iree_codegen.smt.int_knob` and `#iree_codegen.smt.one_of_knob` are placeholders for tunable parameters used by SMT-based tuning flows.

The dialect also defines the discardable operation attribute `iree_codegen.specialization_ranges`, which lets a root operation describe workload ranges for specialized kernel variants.

## Operation Inventory

The local `iree_codegen` dialect defines seventeen operations.

### `iree_codegen.query_tile_sizes`

`iree_codegen.query_tile_sizes` yields runtime tile sizes for a tensor type. It exists for targets where tile sizes cannot be fully resolved at compile time. The source notes VMVX as the current motivating case.

### `iree_codegen.extract_strided_metadata`

`iree_codegen.extract_strided_metadata` extracts a base buffer, offset, sizes, and strides from a strided memref. It is similar to upstream `memref.extract_strided_metadata`, but it deliberately does not fold away static offset or stride information. IREE keeps that link intact because later buffer binding optimizations may still need to update metadata consistently.

### `iree_codegen.fusion_barrier`

`iree_codegen.fusion_barrier` is a pure identity operation on a ranked tensor. It prevents fusion through the value. If a producer and consumer should not be fused, this operation gives the compiler an explicit boundary.

### `iree_codegen.swizzle_hint`

`iree_codegen.swizzle_hint` attaches a swizzle attribute to a tensor or memref view. It is best effort and affects direct users such as immediate loads or stores. It is not correctness-critical; if the swizzle cannot be applied, the compiler may effectively leave accesses unchanged.

### `iree_codegen.null_pointer`

`iree_codegen.null_pointer` creates a `!iree_codegen.null_pointer` value. It is intended for microkernel calls that need a null pointer argument.

### `iree_codegen.load_from_buffer`

`iree_codegen.load_from_buffer` loads a ranked tensor from a compatible strided memref. It is used around bufferization and microkernel paths where the compiler needs an explicit bridge from buffer storage back to tensor semantics.

### `iree_codegen.store_to_buffer`

`iree_codegen.store_to_buffer` stores a ranked tensor into a compatible strided memref. It is the matching bridge in the other direction.

### `iree_codegen.inner_tiled`

`iree_codegen.inner_tiled` represents an operation whose operands are viewed as outer loop dimensions plus inner intrinsic tiles. It is destination-style and can model tensor or vector semantics. The `kind` attribute describes the intrinsic tile, while the `semantics` attribute describes how strict the tile shape rules are and whether the operation has been distributed.

This operation is central for intrinsic-first lowering. For example, CPU or GPU encoding models can turn a matmul-like operation into `iree_codegen.inner_tiled` so later passes know the exact per-intrinsic tile structure.

### `iree_codegen.workgroup_count_hint`

`iree_codegen.workgroup_count_hint` records values that should influence the dispatch workgroup count. If several hints reach the same entry point, IREE uses the elementwise maximum across the hints. If fewer than three sizes are provided, the remaining dimensions default to one.

### `iree_codegen.index_hint`

`iree_codegen.index_hint` is a pure pass-through operation for index values. The hint attribute describes semantic behavior, such as an index being constant across a group of GPU lanes or incrementing across lanes. Optimization passes can use this information and then erase the operation.

### `iree_codegen.smt.constraints`

`iree_codegen.smt.constraints` holds SMT constraints for codegen configuration tuning. It targets a set of root operations, names a pipeline, carries a knobs dictionary, and has a region over problem dimensions represented as `!smt.int` block arguments. It has no runtime semantics and should be erased before normal lowering.

### `iree_codegen.smt.assert`

`iree_codegen.smt.assert` appears inside `iree_codegen.smt.constraints`. It asserts that an SMT boolean condition holds and carries a human-readable diagnostic string with optional integer arguments.

### `iree_codegen.dispatch_config`

`iree_codegen.dispatch_config` is a module-level operation that stores dispatch metadata and the workgroup-count computation for a function. It references the corresponding `func.func`, may record workgroup size, subgroup size, and workgroup local memory, and contains a region that yields exactly three index values for x, y, and z workgroup counts.

### `iree_codegen.smt.knob`

`iree_codegen.smt.knob` declares a named SMT integer constant inside a constraints region. Its name must match a knob attribute in the enclosing constraints dictionary.

### `iree_codegen.smt.lookup`

`iree_codegen.smt.lookup` performs an integer lookup from a sparse key-value table in SMT constraint IR. It is useful for deriving numeric values from enumerated choices, such as selecting dimensions from an MMA-shape index.

### `iree_codegen.yield`

`iree_codegen.yield` is the terminator used in regions owned by `iree_codegen` operations. In `iree_codegen.dispatch_config`, it yields the three workgroup-count values.

### `iree_codegen.ukernel.generic`

`iree_codegen.ukernel.generic` wraps computation that should be forwarded to a microkernel. It has input operands, output operands, optional additional operands, optional function definition attributes, and optional stride-dimension controls. After bufferization, memref operands lower to microkernel ABI arguments such as base pointer, offset, and strides.

## Interfaces

Several interfaces make the dialect extensible.

`PipelineAttrInterface` lets a pipeline attribute build an MLIR pass pipeline. This is how `#iree_codegen.pass_pipeline<"...">` and marker pipeline attributes participate in lowering.

`LoweringConfigAttrInterface` lets passes query tile sizes and tiling levels without depending on one concrete attribute implementation.

`TargetInfoAttrInterface` exposes target limits, such as maximum workgroup count.

`SwizzleAttrInterface` lets a swizzle attribute rewrite access offsets and report access element counts.

`UKernelProviderInterface` provides microkernel implementations. This can be symbolic through `#iree_codegen.symbolic_ukernel_provider` or backend-specific in other dialects.

`InnerTileDescAttrInterface` and `InnerTiledSemanticsAttrInterface` describe intrinsic tile layouts for `iree_codegen.inner_tiled`. They separate the invariant intrinsic description from lowering state such as opaque shape handling or distribution.

## Transformations and Conversions

The IREE codegen tree defines many passes. The most important beginner point is that most of them do not create a new user-level abstraction. They read `iree_codegen` attributes and operations, perform a lowering step, and leave the IR closer to vectors, memrefs, GPU/SPIR-V/LLVM dialects, or runtime ABI calls.

Configuration passes include `iree-codegen-materialize-user-configs`, `iree-codegen-lowering-config-interpreter`, `iree-codegen-reconcile-translation-info`, `iree-codegen-propagate-dispatch-config`, and `iree-codegen-strip-compilation-info`. These passes attach, interpret, propagate, reconcile, or remove configuration attributes such as `#iree_codegen.compilation_info`, `#iree_codegen.lowering_config`, and `#iree_codegen.translation_info`.

Dispatch configuration passes include `iree-codegen-create-dispatch-config`, `iree-codegen-resolve-workgroup-count-hints`, `iree-codegen-propagate-dispatch-size-bounds`, and `iree-codegen-specialize-exports`. These are the passes to inspect when `iree_codegen.dispatch_config` or `iree_codegen.workgroup_count_hint` affects the final executable export.

Tiling and vectorization passes include `iree-codegen-tile-and-distribute-to-workgroups-using-forall-op`, `iree-codegen-generic-vectorization`, `iree-codegen-materialize-vector-masking`, `iree-codegen-materialize-vector-tile-sizes`, `iree-codegen-vector-transfer-lowering`, and `iree-codegen-vectorize-memref-copy`. Backend-specific variants such as `iree-codegen-gpu-tile`, `iree-codegen-gpu-distribute`, `iree-llvmcpu-tile`, and `iree-llvmgpu-vector-distribute` consume the same style of configuration but lower toward different targets.

Bufferization and memref conversion passes include `iree-codegen-bufferize-dispatch-tensor-load-store`, `iree-codegen-iree-comprehensive-bufferize`, `iree-codegen-expand-strided-metadata`, `iree-codegen-fold-memref-alias-ops`, `iree-codegen-bufferize-copy-only-dispatches`, and `iree-codegen-convert-unsupported-float-to-int-buffers`. These are the passes where `iree_codegen.load_from_buffer`, `iree_codegen.store_to_buffer`, and `iree_codegen.extract_strided_metadata` become relevant.

Swizzle and hint passes include `iree-codegen-resolve-swizzle-hints`, `iree-codegen-absorb-swizzle-hint-to-alloc`, `iree-codegen-flatten-swizzle-hint-allocs`, `iree-codegen-reinsert-swizzle-hints`, and `iree-codegen-remove-index-hints`. They either apply hints to memory accesses or erase hints after the information has been consumed.

Microkernel passes include `iree-codegen-cpu-prepare-ukernels`, `iree-codegen-cpu-lower-to-ukernels`, `iree-codegen-lower-tensor-ukernels`, `iree-codegen-lower-memref-ukernels`, `iree-codegen-lower-bitcode-ukernels`, and `iree-codegen-lower-ukernel-ops-to-calls`. The path generally starts from a descriptor such as `#iree_codegen.ukernel_descriptor`, rewrites suitable operations into `iree_codegen.ukernel.generic`, then lowers that wrapper to calls or linked bitcode.

SMT and tuning passes include `iree-codegen-insert-smt-constraints`, `iree-codegen-convert-constraints-to-smt`, `iree-codegen-verify-smt-constraints`, `iree-codegen-link-tuning-specs`, and `iree-codegen-materialize-tuning-specs`. These passes use `iree_codegen.smt.constraints`, `iree_codegen.smt.knob`, `iree_codegen.smt.lookup`, and `iree_codegen.smt.assert` to describe and apply tunable codegen choices.

Target conversion passes include `iree-convert-to-llvm`, `iree-convert-to-nvvm`, `iree-convert-to-rocdl`, and `iree-convert-to-spirv`, along with target families such as `iree-llvmcpu-*`, `iree-llvmgpu-*`, and `iree-spirv-*`. Those are not all part of the Codegen dialect itself, but they are where Codegen's attributes and helper ops finally turn into target-specific IR.

## What It Implies

When you see `#iree_codegen.translation_info`, the dispatch is no longer relying on generic lowering defaults. The export has a selected pipeline and often a chosen workgroup or subgroup size.

When you see `#iree_codegen.lowering_config`, an operation has tile-size instructions. Later tiling, distribution, vectorization, and bufferization passes are expected to follow that information.

When you see `iree_codegen.inner_tiled`, the compiler is exposing intrinsic tile structure. This is deeper than ordinary `linalg` lowering: the operation is already shaped around the native or modeled inner tile of the target computation.

When you see `iree_codegen.ukernel.generic`, codegen has chosen a library-style implementation path. The operation is a temporary wrapper around a function-call ABI, not a final runtime operation.

## How To Read Codegen IR

Start with the executable export. Look for `#iree_codegen.translation_info`, workgroup size, subgroup size, and any referenced transform dialect codegen spec.

Then inspect the root compute operations. Look for `#iree_codegen.lowering_config`, `#iree_codegen.compilation_info`, `#iree_codegen.ukernel_descriptor`, or `iree_codegen.specialization_ranges`. These attributes tell you why later passes tile or specialize the kernel the way they do.

Next, scan helper operations. `iree_codegen.workgroup_count_hint` affects dispatch counts. `iree_codegen.swizzle_hint` affects memory access layout. `iree_codegen.index_hint` guides lane-aware optimization. `iree_codegen.load_from_buffer` and `iree_codegen.store_to_buffer` mark tensor-buffer bridges. SMT operations mean you are looking at tuning or constraint-generation IR rather than normal executable code.

## Minimal Example

This simplified sketch shows the policy role of the dialect:

```mlir
#pipeline = #iree_codegen.pass_pipeline<"builtin.module(func.func(canonicalize))">
#translation = #iree_codegen.translation_info<
  pipeline = #pipeline
  workgroup_size = [64, 1, 1]
  subgroup_size = 32>
#level0 = #iree_codegen.lowering_config_level<tile_sizes = [16, 16, 4]>
#config = #iree_codegen.lowering_config<tile_sizes = [#level0]>

hal.executable.export @matmul {
  translation_info = #translation
}

%0 = linalg.matmul
    ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%c : tensor<?x?xf32>)
    {lowering_config = #config}
    -> tensor<?x?xf32>
```

The important part is not the exact syntax. The mental model is that `translation_info` controls the dispatch-level pipeline, while `lowering_config` controls how the root operation should be tiled and lowered inside that pipeline.
