# IREE GPU Dialect

The `iree_gpu` dialect is part of IREE's compiler code generation layer. It is
not a general-purpose GPU programming dialect like MLIR's upstream `gpu`
dialect. Instead, it is a compact set of helper operations and attributes that
let IREE represent GPU-specific codegen facts while it is still transforming
tensors, vectors, tiled operations, and structured parallel loops.

For a beginner, the most useful way to understand this dialect is to think of it
as a bridge. Earlier compiler stages describe computation in higher-level
dialects such as `linalg`, `tensor`, `scf`, `vector`, and IREE's own codegen
dialect. Later compiler stages need target-specific GPU code for backends such
as SPIR-V or LLVM-based GPU pipelines. The `iree_gpu` dialect sits in the
middle. It records synchronization, subgroup behavior, DMA-style movement, and
GPU memory-resource information in a form that is still flexible enough for
IREE's lowering pipeline to rewrite.

## Why This Dialect Exists

GPU lowering has several concepts that are awkward to express with only generic
tensor or loop IR. Threads may cooperate on a shared tile. A value may need to
cross a synchronization point without implying that data was copied. A subgroup
scan may need to preserve its combining region until a target-specific lowering
is chosen. A memory access may need to be recognized as a coalesced gather or an
asynchronous DMA movement before it becomes low-level buffer code.

The `iree_gpu` dialect gives IREE places to put those ideas.

This matters because lowering too early can erase information that later
optimizations need. For example, lowering a producer-consumer tensor pattern
directly to allocations, reads, writes, and barriers can make it harder to
reason about where synchronization belongs. Keeping a structured operation such
as `iree_gpu.barrier_region` allows the compiler to preserve the dependency
shape for a longer part of the pipeline.

The dialect is also deliberately target-independent at the operation level. Some
operations are motivated by hardware behavior, and some have lowering paths that
are only profitable or legal on certain GPU targets, but the dialect itself is
used before the final choice of SPIR-V, LLVM, ROCDL, or other backend-specific
IR.

## When It Is Important

You usually see `iree_gpu` when reading IREE compiler IR rather than when writing
frontend MLIR by hand. It becomes important in GPU codegen passes that:

- distribute tiled work to GPU lanes or subgroups,
- preserve synchronization around tensor or vector values,
- model subgroup-level collectives,
- prepare coalesced memory movement,
- bridge from `scf.forall` GPU mappings into IREE's parallel-control-flow model,
- lower GPU codegen helper operations after bufferization.

Use this dialect mentally as a sign that the compiler is past the purely
algorithmic part of lowering and is now making GPU execution structure explicit.
It is still not the final machine-level representation. It is a staging dialect
for IREE's GPU lowering pipeline.

## Core Ideas

The first core idea is value-semantics synchronization. MLIR tensor and vector
values are SSA values, not mutable memory locations. A synchronization point in
this world should not automatically mean "copy this data" or "write this memory
now." Operations such as `iree_gpu.value_barrier` and
`iree_gpu.barrier_region` represent ordering among workers while preserving
value semantics.

The second core idea is subgroup and lane structure. GPUs execute work in groups
of lanes, and many efficient operations are expressed at subgroup granularity.
The dialect represents subgroup scans directly with
`iree_gpu.subgroup_scan`, and its passes can convert GPU-thread-mapped
`scf.forall` operations into nested `pcf.generic` operations with subgroup and
lane scopes.

The third core idea is late memory movement. Operations such as
`iree_gpu.async_dma` and `iree_gpu.coalesced_gather_dma` let the compiler keep a
high-level representation of cooperative data movement while it still decides
which target-specific instruction sequence or fallback lowering should be used.

## Operation Inventory

The current `iree_gpu` dialect defines eight operations.

| Operation | Role |
| --- | --- |
| `iree_gpu.barrier_region` | Wraps a region of shared code whose inputs and results require worker synchronization. |
| `iree_gpu.value_barrier` | Synchronizes tensor or vector SSA values without giving the operation copy or data-movement semantics. |
| `iree_gpu.yield` | Terminates regions used by `iree_gpu.barrier_region` and `iree_gpu.subgroup_scan`. |
| `iree_gpu.subgroup_scan` | Performs an inclusive or exclusive prefix scan across lanes in a subgroup cluster and also returns the cluster total. |
| `iree_gpu.buffer_resource_cast` | Marks a tensor as using an AMDGPU-style buffer resource memory space before bufferization when that lowering path applies. |
| `iree_gpu.coalesced_gather_dma` | Represents cooperative gather-style movement designed for coalesced GPU access. |
| `iree_gpu.async_dma` | Represents asynchronous movement between memories, either in tensor value-semantics form or memref buffer-semantics form. |
| `iree_gpu.global_subgroup_barrier` | Represents a synchronization-only barrier across all subgroups in a workgroup. |

## Synchronization Operations

`iree_gpu.barrier_region` is a structured synchronization operation. It contains
a region, takes optional inputs, and yields results through `iree_gpu.yield`.
The source comments describe it as arising naturally when producer and consumer
`scf.forall` regions with the same mapping are fused together. In that situation
the compiler may need a barrier between the producer side and the consumer side,
but it still wants to keep the combined computation in a structured form.

This operation can later lower to more concrete barriers around its operands and
results. The important beginner point is that `iree_gpu.barrier_region` is not
just a low-level fence. It is a way to keep a synchronized region visible while
the compiler continues to transform the code.

`iree_gpu.value_barrier` is smaller. It takes tensor or vector inputs and returns
equivalent results. It does not copy data. It does not perform a transfer. In a
parallel context, it means that workers synchronize around the value. Outside a
parallel context, it is effectively a no-op. That makes it useful as a
value-semantics marker that can survive until the compiler has enough context to
place lower-level synchronization correctly.

`iree_gpu.global_subgroup_barrier` is a synchronization-only barrier across all
subgroups in a workgroup. The important limitation is that it has no memory fence
semantics. Memory operations may still be reordered around it unless a separate
fence or memory-ordering mechanism is present. A reader should therefore avoid
interpreting this operation as a complete memory synchronization primitive.

## Subgroup Operation

`iree_gpu.subgroup_scan` performs a prefix scan across lanes in a subgroup
cluster. The operation has a region that defines the combiner. The combiner
takes two values of the scan type and yields one value with `iree_gpu.yield`.

The scan can be inclusive or exclusive. In inclusive mode, each lane receives the
combination of values up to and including its own lane. In exclusive mode, each
lane receives the combination of earlier lanes, with an identity value supplied
for the first lane. The operation also returns a second result: the total
reduction for the cluster.

The combiner must be associative, because the compiler and hardware may group
the computation in different ways. It does not have to be commutative, so the
order of values still matters. That distinction is important when reading
lowering code: associativity gives the compiler freedom to reassociate the
grouping, but not to arbitrarily reorder operands.

## Memory Movement Operations

`iree_gpu.async_dma` represents asynchronous data movement. It can appear in a
tensor form, where the result aliases the destination value and can be combined
with `iree_gpu.value_barrier`, or in a memref form, where it writes directly into
the destination buffer. The operation carries a transfer type, source and
destination indices, and optional permutation and in-bounds metadata. Source
indices can also describe gather-like access.

This operation is useful because GPU code often needs to stage data into a
faster memory space while other work continues. Keeping that movement explicit
as `iree_gpu.async_dma` lets the compiler reason about the transfer before it
chooses a specific backend implementation.

`iree_gpu.coalesced_gather_dma` represents cooperative gather movement where
lanes work together to produce a coalesced access pattern. It has both
tensor-style and buffer-style forms. The operation is designed to live in a
parallel combining context such as `scf.forall.in_parallel`, and it can lower to
an AMDGPU gather-to-LDS form when requirements are met. Otherwise, it can lower
through a more general vector gather path.

`iree_gpu.buffer_resource_cast` is a specialized marker used before
bufferization. It nominally casts a tensor to an AMDGPU buffer-resource memory
space. If the value eventually bufferizes into the expected storage-buffer form,
the cast information applies. If not, the operation is dropped without changing
program meaning. That makes it a hint for a particular lowering path, not a
frontend-level semantic conversion.

## Transformations And Conversions

The dialect's pass file defines five passes.

| Pass | What it does |
| --- | --- |
| `iree-gpu-distribute-inner-tiled-to-lanes` | Distributes `iree_codegen.inner_tiled` operations to GPU lanes and performs cleanup and producer fusion around the resulting lane-level loop. |
| `iree-gpu-expand-undistributed-inner-tiles` | Expands inner dimensions of undistributed `iree_codegen.inner_tiled` operations so their shapes match the thread layout. |
| `iree-gpu-lower-ops` | Performs post-bufferization lowering of `iree_gpu` operations before later backend-specific lowerings. |
| `iree-gpu-unroll-to-intrinsics` | Unrolls IREE codegen inner-tiled operations toward intrinsic-sized forms, then folds unit dimensions and removes hoistable conversions. |
| `iree-gpu-convert-forall-to-generic-nest` | Converts `scf.forall` operations with GPU thread mappings into nested `pcf.generic` operations using subgroup and lane scopes. |

These passes show how `iree_gpu` fits into the broader IREE pipeline. Some of
them do not primarily rewrite `iree_gpu` operations. For example,
`iree-gpu-distribute-inner-tiled-to-lanes`,
`iree-gpu-expand-undistributed-inner-tiles`, and
`iree-gpu-unroll-to-intrinsics` mainly operate on IREE codegen tiled operations.
They are still part of the GPU dialect's transform package because they prepare
GPU-specific structure for the same lowering flow.

`iree-gpu-lower-ops` is the most direct lowering pass for the dialect. Its
implementation populates patterns for `iree_gpu.value_barrier` and IREE codegen
inner-tiled operations, applies them greedily, and eliminates hoistable
conversions. This is a post-bufferization stage, so it is close to the point
where value-semantics helper operations need to disappear or become lower-level
IR.

`iree-gpu-convert-forall-to-generic-nest` is a conversion pass. It looks for
`scf.forall` operations whose mapping attributes are GPU thread mappings. It
then converts them into nested `pcf.generic` operations. The outer scope is a
subgroup scope and the inner scope is a lane scope. This is a good example of
IREE using a custom dialect to encode parallel control flow more explicitly than
plain `scf.forall` can.

## How To Read This Dialect In IR

When you see `iree_gpu` IR, first ask what execution concept is being preserved.
If the operation is `iree_gpu.barrier_region` or `iree_gpu.value_barrier`, the
compiler is preserving synchronization while still using tensor or vector SSA
values. If the operation is `iree_gpu.subgroup_scan`, the compiler is preserving
a subgroup collective with an explicit combiner. If the operation is
`iree_gpu.async_dma` or `iree_gpu.coalesced_gather_dma`, the compiler is keeping
memory movement visible enough to choose a better target lowering later.

Then look at the surrounding dialects. `scf.forall` often tells you that the
compiler is still representing parallel work structurally. `pcf.generic` means
IREE has moved toward its own parallel-control-flow representation. `vector`,
`memref`, and `bufferization` operations show how close the code is to
target-level lowering. The same `iree_gpu` operation can be easier or harder to
understand depending on which of those neighbors are present.

Finally, avoid reading every operation as a runtime API call. These operations
are compiler IR. Some lower to real hardware instructions or barriers. Some
lower to combinations of other operations. Some disappear after they have guided
the compiler. Their meaning is in the lowering pipeline, not in a stable source
programming interface.

## What It Implies

The presence of `iree_gpu` implies that IREE has committed to a GPU-oriented
codegen path, but it has not necessarily committed to the final backend
instruction sequence. It also implies that the compiler is carrying information
that generic MLIR dialects either cannot express directly or would express too
early in a less optimizable form.

For optimization, this is useful because the compiler can delay decisions about
barriers, subgroup execution, lane distribution, and memory movement. For
debugging, it means that `iree_gpu` IR is a middle-stage artifact. If a value
barrier, DMA operation, or subgroup scan remains too late, the issue is probably
in the GPU lowering pipeline. If it appears early, it is usually because an IREE
codegen pass has already recognized a GPU-specific execution pattern.

The main pitfall is overinterpreting the dialect. `iree_gpu.value_barrier` does
not copy data. `iree_gpu.global_subgroup_barrier` is not a memory fence.
`iree_gpu.buffer_resource_cast` is a lowering hint that may be dropped. The
passes named under the GPU dialect package may transform nearby IREE codegen IR
without adding new `iree_gpu` operations. Understanding those distinctions makes
the dialect much easier to read.

## Summary

The `iree_gpu` dialect is a focused IREE codegen dialect for GPU lowering. It
models synchronization, subgroup scans, cooperative memory movement, and
GPU-specific resource information while preserving enough structure for later
optimization. It is important when reading IREE's GPU compiler pipeline because
it marks the point where generic tensor, vector, and loop IR has started to gain
explicit GPU execution meaning, but before that meaning has been fully lowered
to backend-specific code.
