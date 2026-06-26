# IREE `iree_cpu` Dialect

The IREE `iree_cpu` dialect is a CPU code generation support dialect. It does not model ordinary program computation with its own operation set. Instead, it gives IREE's CPU and VMVX lowering paths a precise way to attach CPU-specific lowering choices, packed-data layouts, matrix-multiply intrinsic choices, encoding resolvers, and microkernel providers to the rest of the compiler IR.

For a beginner, the useful mental model is this: `iree_cpu` is a control panel for IREE CPU lowering. Most of the computation is still represented by other dialects, such as `linalg`, `tensor`, `vector`, `iree_codegen`, `hal`, and LLVM dialects. The `iree_cpu` attributes tell those transformations which CPU strategy, tiling levels, data layout, and intrinsic family to use.

The local checkout defines 0 `iree_cpu` operations, 8 `iree_cpu` attributes, 28 `MMAIntrinsic` enum values including `None`, 7 CPU lowering pipeline enum values, and 28 named CPU-adjacent transformation or conversion passes.

## When IREE CPU Is Important

`iree_cpu` is important whenever IREE is compiling work for CPU-like targets, especially the LLVMCPU target and the VMVX path. You will see it when a dispatch has moved past generic tensor-level program structure and the compiler must answer CPU-specific questions:

- Which lowering pipeline should lower this dispatch?
- What tile sizes should be used at distribution, cache, reduction, and vector levels?
- Should a matmul use packed data tiling and CPU MMA-like intrinsic lowering?
- Which x86 AVX, x86 AVX-512, Arm SVE, or generic scalar strategy matches this workload?
- Should an operation become an IREE CPU microkernel call?
- Which encoding resolver should materialize layouts for CPU or VMVX execution?

You usually do not author this dialect directly in frontend IR. You inspect it while debugging IREE CPU code generation, lowering strategy selection, materialized encodings, packed matmul layouts, vector lowering, or LLVM conversion.

## Why It Is Needed

MLIR's standard computational dialects intentionally avoid baking in one backend's CPU strategy. A `linalg.matmul` describes a mathematical operation. It does not say whether to use AVX-512 VNNI, Arm SVE FMLA, a generic scalar fallback, a packed microkernel, or a particular sequence of cache and vector tiling steps.

IREE also needs a representation that survives across several codegen stages. Early passes may choose a pipeline and lowering config, middle passes may introduce `iree_codegen.inner_tiled` operations, and late passes may lower those inner tiles to vector operations, LLVM intrinsics, or microkernel calls. `iree_cpu` attributes carry those choices without turning the whole program into target-specific operations too early.

The implication is that `iree_cpu` is mostly metadata, but it is not passive documentation. Its attributes implement IREE codegen interfaces. Passes query those interfaces to build pass pipelines, read tile sizes, materialize encodings, lower inner-tiled MMA operations, and replace selected regions with microkernel calls.

## Operations

This checkout defines no dialect-owned `iree_cpu.*` operations.

That absence is important. The dialect TableGen description says the dialect provides operations and attributes for CPU-focused IREE code generation, but the current local IR surface is attribute-only. CPU-specific behavior appears through attributes attached to other operations, especially `iree_codegen.translation_info`, `iree_codegen.inner_tiled`, `hal.executable.target`, and operations carrying `lowering_config` attributes.

When reading IR, do not search for `iree_cpu.some_op`. Search for `#iree_cpu...` attributes.

## Attributes

The local checkout defines these 8 `iree_cpu` attributes.

```text
#iree_cpu.mma_intrinsic<...>
#iree_cpu.mma_semantics<>
#iree_cpu.data_tiled_mma_layout<...>
#iree_cpu.ukernel_provider
#iree_cpu.pipeline<...>
#iree_cpu.lowering_config<...>
#iree_cpu.cpu_encoding_resolver<...>
#iree_cpu.vmvx_encoding_resolver<...>
```

`#iree_cpu.mma_intrinsic<...>` is the enum attribute wrapper for the CPU `MMAIntrinsic` enum. In practice, you most often see the enum value nested inside `#iree_cpu.data_tiled_mma_layout`, for example `intrinsic = MMA_X86_AVX512_1x16x1_F32_F32`.

`#iree_cpu.mma_semantics<>` describes CPU semantics for `iree_codegen.inner_tiled`. In this checkout it is currently a unit attribute: CPU tiles are undistributed and expanded, so there are no parameters to print. It still matters because it implements the inner-tiled semantics interface expected by codegen.

`#iree_cpu.data_tiled_mma_layout<...>` describes a CPU data-tiled MMA layout for `iree_codegen.inner_tiled`. It carries an `intrinsic` enum value, optional unrolling factors `intrinsics_m`, `intrinsics_n`, and `intrinsics_k`, and optional `lhs_type`, `rhs_type`, and `acc_type` parameters used by the generic scalar fallback. This is the key attribute for packed CPU matmul lowering.

`#iree_cpu.ukernel_provider` implements IREE's microkernel provider interface for LLVMCPU. It can rewrite an `iree_codegen.inner_tiled` operation with a CPU data-tiled MMA layout and an `iree_codegen.ukernel` descriptor into an `iree_codegen.ukernel.generic` call, then attach matching CPU bitcode as a `hal.executable_object`.

`#iree_cpu.pipeline<...>` identifies the CPU lowering pipeline to build. It implements the IREE pipeline attribute interface by delegating to the CPU pipeline builder registered by LLVMCPU codegen.

`#iree_cpu.lowering_config<...>` carries CPU tiling and vectorization configuration. It is a dictionary-based attribute whose known keys include `distribution`, `cache_parallel`, `cache_reduction`, `vector_common_parallel`, `vector_reduction`, and `vector_inner_parallel`. Values are tile-size lists, and some vector levels may use scalable-vector notation such as `[[4], [4], 0]`.

`#iree_cpu.cpu_encoding_resolver<...>` is the encoding resolver for CPU backends. It participates in layout serialization and materialization through external interface models.

`#iree_cpu.vmvx_encoding_resolver<...>` is the analogous resolver for VMVX. It lets the same encoding materialization infrastructure use VMVX-specific layout decisions.

Some older prose in the checkout mentions `#iree_cpu.cpu_encoding_layout`; the TableGen in this checkout defines resolver attributes, not a standalone `cpu_encoding_layout` attribute.

## MMA Intrinsic Values

The `MMAIntrinsic` enum names CPU matrix-multiply-accumulate strategies. Some correspond closely to hardware intrinsics. Others are virtual descriptions that tell lowering to cast, widen, broadcast, or fall back to generic scalar code.

The exact enum values in this checkout are:

```text
None
MMA_X86_AVX2_FMA_1x8x1_F32_F32
MMA_X86_AVX2_FMA_8x1x1_F32_F32
MMA_X86_AVX512_1x8x1_F64_F64
MMA_X86_AVX512_8x1x1_F64_F64
MMA_X86_AVX512_1x16x1_F32_F32
MMA_X86_AVX512_16x1x1_F32_F32
MMA_X86_AVX512_1x16x1_F32_F16_CASTF32
MMA_X86_AVX512_16x1x1_F32_F16_CASTF32
MMA_X86_AVX512FP16_1x32x1_F16_F16
MMA_X86_AVX512FP16_32x1x1_F16_F16
MMA_X86_AVX512BF16_1x16x2_F32_BF16
MMA_X86_AVX512BF16_16x1x2_F32_BF16
MMA_X86_AVX512_1x16x2_I32_I16
MMA_X86_AVX512_16x1x2_I32_I16
MMA_X86_AVX512VNNI_1x16x2_I32_I16
MMA_X86_AVX512VNNI_16x1x2_I32_I16
MMA_X86_AVX512_1x16x2_I32_I8_CASTI16
MMA_X86_AVX512_16x1x2_I32_I8_CASTI16
MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16
MMA_X86_AVX512VNNI_16x1x2_I32_I8_CASTI16
MMA_X86_AVX512VNNI_1x16x4_I32_UI8_I8
MMA_X86_AVX512VNNI_16x1x4_I32_I8_UI8
MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16
MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32
MMA_ARM_SVE_FMLA_4VLx1x1_F32_F32
MMA_GENERIC_SCALAR_1x1x1_REG8
MMA_GENERIC_SCALAR_1x1x1_REG16
```

The name shape is meaningful. `MMA_X86_AVX512_1x16x1_F32_F32` says the architecture family is x86 AVX-512, the conceptual tile is `1x16x1`, and the accumulator/input type family is `f32`. Suffixes such as `CASTF32` and `CASTI16` tell lowering that widening is part of the strategy. The `UI8_I8` and `I8_UI8` forms preserve signedness intent even though IR storage may use signless integer types.

The generic scalar values are type-polymorphic fallbacks. They use the `lhs_type`, `rhs_type`, and `acc_type` parameters on `#iree_cpu.data_tiled_mma_layout` because the enum value alone does not determine the element types.

## Pipeline Values

`#iree_cpu.pipeline<...>` uses this `LoweringPipeline` enum:

```text
Default
DoubleTilingExpert
ConvTileAndDecomposeExpert
Mmt4dTilingExpert
BufferOpsTileAndVectorize
DataTiling
LinalgExtTileAndVectorize
```

`Default` is the general CPU lowering route. `DoubleTilingExpert` is used for double-tiling strategies, often around matmul-like operations. `ConvTileAndDecomposeExpert` targets convolution lowering. `Mmt4dTilingExpert` is for packed matrix multiplication paths. `BufferOpsTileAndVectorize` handles buffer-level tiling and vectorization. `DataTiling` is the packed-data tiling route that pairs naturally with `#iree_cpu.data_tiled_mma_layout`. `LinalgExtTileAndVectorize` covers IREE's linalg-extension operations.

The pipeline attribute is usually nested in `#iree_codegen.translation_info`, for example:

```mlir
#iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
```

Read this as "lower the dispatch using IREE's CPU double-tiling pipeline."

## Transformations And Conversions

The `iree_cpu` dialect itself does not define a dialect-owned pass table in this checkout. The transformations that matter live in IREE's common CPU and LLVMCPU codegen pass sets. They are the passes that read or create `#iree_cpu.pipeline`, `#iree_cpu.lowering_config`, `#iree_cpu.data_tiled_mma_layout`, and related attributes.

The exact named CPU-adjacent passes covered by this chapter are:

```text
iree-codegen-cpu-lower-to-ukernels
iree-codegen-cpu-prepare-ukernels
iree-codegen-cpu-propagate-data-layout
iree-convert-to-llvm
iree-llvmcpu-2d-scalable-to-1d-scalable
iree-llvmcpu-assign-constant-ordinals
iree-llvmcpu-assign-import-ordinals
iree-llvmcpu-assign-workgroup-local-memory
iree-llvmcpu-check-ir-before-llvm-conversion
iree-llvmcpu-emit-vectorization-remarks
iree-llvmcpu-expand-f16-op-to-f32
iree-llvmcpu-link-executables
iree-llvmcpu-lower-executable-target
iree-llvmcpu-mmt4d-vector-lowering
iree-llvmcpu-peel
iree-llvmcpu-select-lowering-strategy
iree-llvmcpu-split-reduction
iree-llvmcpu-synchronize-symbol-visibility
iree-llvmcpu-tile
iree-llvmcpu-tile-and-fuse-producer-consumer
iree-llvmcpu-tile-to-vector-size
iree-llvmcpu-unfuse-fma-pass
iree-llvmcpu-vector-contract-custom-kernels
iree-llvmcpu-vector-shape-cast-lowering
iree-llvmcpu-vector-transpose-lowering
iree-llvmcpu-verify-linalg-transform-legality
iree-llvmcpu-verify-vector-size-legality
iree-llvmcpu-virtual-vector-lowering
```

`iree-llvmcpu-select-lowering-strategy` is the pass that chooses CPU lowering strategy information for executable variants. It is where you expect to see `#iree_cpu.pipeline` and `#iree_cpu.lowering_config` appear or be refined.

`iree-llvmcpu-lower-executable-target` consumes a selected pipeline attribute and runs the target lowering sequence. Internally, `#iree_cpu.pipeline<...>` dispatches to the registered CPU pipeline builder, which maps enum values such as `Default`, `Mmt4dTilingExpert`, or `DataTiling` to concrete pass pipelines.

`iree-llvmcpu-tile`, `iree-llvmcpu-tile-and-fuse-producer-consumer`, and `iree-llvmcpu-tile-to-vector-size` are the tiling passes that use the tiling levels stored in `#iree_cpu.lowering_config`.

`iree-llvmcpu-split-reduction` handles reduction splitting. `iree-llvmcpu-peel` performs loop peeling on non-distributed loops. `iree-llvmcpu-2d-scalable-to-1d-scalable` replaces unsupported scalable dimensions with loops, currently important for Arm SME-style constraints.

`iree-codegen-cpu-propagate-data-layout` propagates pack, unpack, and reshape operations so a whole dispatch uses a consistent layout. `iree-codegen-cpu-prepare-ukernels` rank-reduces operations to fit existing microkernel requirements. `iree-codegen-cpu-lower-to-ukernels` separates out IR that should become a microkernel call.

`iree-llvmcpu-mmt4d-vector-lowering`, `iree-llvmcpu-virtual-vector-lowering`, `iree-llvmcpu-vector-transpose-lowering`, `iree-llvmcpu-vector-shape-cast-lowering`, and `iree-llvmcpu-vector-contract-custom-kernels` are vector-level lowering passes. They are where packed matmul, vector contracts, transposes, shape casts, and target custom kernels become lower-level vector or LLVM constructs.

`iree-llvmcpu-expand-f16-op-to-f32` widens selected f16 operations to f32 and then narrows results as needed. `iree-llvmcpu-unfuse-fma-pass` converts LLVM FMA operations into separate multiply and add operations when the target path wants unfused arithmetic.

`iree-llvmcpu-assign-workgroup-local-memory` lays out marked workgroup-local allocations and records the required local memory. `iree-llvmcpu-assign-constant-ordinals` and `iree-llvmcpu-assign-import-ordinals` assign executable ordinals for constants and imports. `iree-llvmcpu-synchronize-symbol-visibility` keeps LLVM linkage consistent with MLIR symbol visibility.

`iree-llvmcpu-check-ir-before-llvm-conversion`, `iree-llvmcpu-verify-vector-size-legality`, and `iree-llvmcpu-verify-linalg-transform-legality` are validation passes. They make sure the IR handed to late CPU lowering is legal for the backend.

`iree-convert-to-llvm` is the final CPU-side conversion path from higher-level IR such as Linalg, HAL, Shape, Vector, and standard MLIR dialects to LLVM dialect. `iree-llvmcpu-link-executables` links LLVMCPU HAL executables in the top-level module.

There is also an important transform dialect pattern used in tests:

```mlir
transform.apply_patterns.iree.lower_inner_tiled
```

This is not an `iree_cpu` operation. It is a transform pattern that lowers `iree_codegen.inner_tiled` operations using `#iree_cpu.data_tiled_mma_layout` and `#iree_cpu.mma_semantics<>`. Depending on the intrinsic, the result may include broadcasts, widening casts, `vector.contract`, LLVM intrinsic calls such as FMA or x86 dot-product intrinsics, and accumulator updates.

## How To Read IREE CPU IR

Start with the translation info. If you see:

```mlir
#iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DataTiling>>
```

then the dispatch is being lowered through the CPU data-tiling path. The pipeline value tells you which family of CPU transformations should run.

Next, inspect `lowering_config` attributes. A config like:

```mlir
#iree_cpu.lowering_config<
  distribution = [128, 128, 0],
  cache_parallel = [64, 64, 0],
  cache_reduction = [0, 0, 16],
  vector_common_parallel = [4, 4, 0],
  vector_reduction = [0, 0, 4],
  vector_inner_parallel = [0, 0, 0]
>
```

describes a staged lowering plan. The operation is first tiled for distribution, then cache behavior, then vectorization. Zeros generally mean that dimension is not tiled at that level.

Then look for `iree_codegen.inner_tiled`. A CPU data-tiled MMA example has this shape:

```mlir
%0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
  kind = #iree_cpu.data_tiled_mma_layout<
    intrinsic = MMA_X86_AVX512_1x16x1_F32_F32,
    intrinsics_m = 2>,
  semantics = #iree_cpu.mma_semantics<>
} : vector<2x1x2x1xf32>, vector<1x1x1x16xf32> into vector<2x1x2x16xf32>
```

Read this as "this inner tile should be lowered as an AVX-512 f32 MMA-style tile, unrolled twice in M, using CPU semantics." Later lowering can turn that into vector operations and LLVM intrinsics.

Finally, inspect executable target configuration. If a `hal.executable.target` carries `iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>`, materialized encodings should use CPU layout logic. If it carries `#iree_cpu.vmvx_encoding_resolver<>`, it is using the VMVX resolver. If it carries `iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider`, selected packed operations can become CPU microkernel calls.

## What It Implies

Seeing `#iree_cpu.pipeline<...>` means the compiler has chosen a CPU lowering strategy. If the choice is wrong, look at lowering strategy selection and tuning input.

Seeing `#iree_cpu.lowering_config<...>` means the operation is no longer just mathematical IR. It carries concrete tiling intent that later passes will interpret.

Seeing `#iree_cpu.data_tiled_mma_layout<...>` means a matmul-like inner tile is being lowered using CPU packed-data and MMA-intrinsic rules. The intrinsic name tells you the target family and the expected widening or signedness behavior.

Seeing `#iree_cpu.mma_semantics<>` means the inner-tiled operation is using CPU semantics: no GPU-style thread distribution, no subgroups, and expanded tile values.

Seeing `#iree_cpu.ukernel_provider` means IREE may replace selected inner-tiled computation with a CPU microkernel call and attach the required bitcode object.

Seeing `#iree_cpu.cpu_encoding_resolver<>` or `#iree_cpu.vmvx_encoding_resolver<>` means tensor encodings have not just arbitrary layouts; they will be materialized according to the selected CPU-like backend.

The main beginner lesson is that `iree_cpu` is not a frontend dialect. It is a backend coordination dialect. It records decisions that let IREE move from portable tensor and linalg IR to CPU-specific tiled, vectorized, LLVM, and microkernel code.
