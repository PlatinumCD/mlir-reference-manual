# IREE `flow` Dialect

The IREE `flow` dialect is the compiler layer where high-level tensor programs become explicit units of executable work. It is not a source language dialect like Torch or StableHLO, and it is not yet the low-level runtime dialect. Instead, it is the middle layer that says: these tensor operations belong together, this is the data that crosses the boundary, this is the symbolic workload, and this is the executable entry point that later compiler phases will schedule and allocate.

For a beginner, the most useful way to read Flow is as the "dispatch planning" dialect in IREE. Earlier compiler stages may still look like ordinary tensor, linalg, or model-level operations. Later stages such as Stream and HAL need explicit resources, command ordering, devices, and executables. Flow is the bridge between those worlds. It keeps value-semantic tensors long enough for MLIR analyses and fusion to work, while adding explicit dispatch boundaries so the compiler can decide what work should run together.

## When Flow Is Important

Flow matters when you want to understand how IREE turns tensor code into device work. If you are debugging why two `linalg` operations fused, why a dispatch was outlined, why a tensor copy appeared, or why a dynamic shape was captured, Flow is usually one of the first dialects to inspect.

Use Flow when you are working inside the IREE compiler after input conversion and dispatch creation. It is useful for compiler developers writing fusion, dispatch formation, tracing, executable outlining, tensor-shape cleanup, or flow-to-stream conversion logic. It is usually not the dialect an application author writes by hand. An application generally sends StableHLO, TOSA, Torch, or another input form to IREE and lets the compiler create Flow.

Flow is needed because ordinary tensor IR does not say enough about dispatch execution. A `linalg.matmul` on tensors tells us the computation, but it does not by itself say what device dispatch contains it, how many workgroups it needs, which tensor dimensions must be carried across the boundary, which output aliases an input, or where the dispatch executable lives. Flow adds those missing compiler-level contracts.

## How It Fits In IREE

In IREE's VM transform pipeline, the broad order is:

1. Input conversion and preprocessing legalize frontend IR.
2. Dispatch creation forms `flow.dispatch.region` and lowers it to `flow.dispatch.workgroups`.
3. The Flow pipeline outlines dispatch workgroups into `flow.executable` operations and replaces the inline work with `flow.dispatch`.
4. The Stream pipeline starts with `iree-stream-conversion`, which converts Flow and other supported input dialects into Stream.
5. HAL and VM lowering continue toward runtime commands, executable targets, and bytecode.

This means Flow is where the compiler still thinks in tensors, but has already started making runtime-shaped decisions. Workloads are symbolic, device choices are not final, and buffer allocation has not happened yet.

## Core Ideas

Flow is based on value semantics at the boundaries. A `flow.dispatch` consumes tensor values and produces tensor values. Inside a `flow.dispatch.workgroups` body, those tensors are represented as dispatch tensors that can be loaded and stored by workgroups. This split lets the outer program remain SSA-like and easy to analyze while the inner dispatch body can model tiled parallel execution.

Dynamic dimensions are explicit. Many Flow ops carry dynamic dimension operands next to tensor values. This is why you will often see syntax like `tensor<?xf32>{%d0}`. The tensor type says the rank and static parts; the dynamic dimension operands say the actual runtime sizes.

Workgroup count is also explicit but not always fully resolved in Flow. A dispatch has a workload list, and an executable export can have a workgroup-count region that computes the final X, Y, Z grid. Backends may later specialize or lower that computation according to device information.

Flow also represents memory and placement hints at a tensor level. Operations such as `flow.tensor.barrier`, `flow.tensor.transfer`, and `flow.tensor.encode` do not allocate buffers directly, but they communicate scheduling, affinity, and layout constraints that later stages must respect.

## Operation Inventory

Flow has 39 operations in the local IREE source tree.

### Dispatch and Executable Ops

- `flow.dispatch.region` groups operations before final dispatch outlining. It is a lightweight fusion container that can capture values from the surrounding scope.
- `flow.dispatch.workgroups` represents executable work over a three-dimensional workgroup grid. It is isolated from above, so its inputs and outputs are explicit.
- `flow.dispatch` calls one or more outlined executable entry points from the host-side Flow program.
- `flow.dispatch.workgroup.id` reads the current workgroup's X, Y, or Z index inside a dispatch body.
- `flow.dispatch.workgroup.count` reads the total number of workgroups in a grid dimension.
- `flow.dispatch.workgroup.size` reads the symbolic workgroup size for a grid dimension.
- `flow.dispatch.tie_shape` ties runtime dynamic dimensions to a dispatch tensor argument.
- `flow.return` terminates `flow.dispatch.region` and `flow.dispatch.workgroups` regions.
- `flow.executable` contains outlined dispatch code that can later be lowered to target-specific executable forms.
- `flow.executable.export` defines a callable entry point inside a `flow.executable`, including the workgroup-count calculation.
- `flow.executable_end` is the terminator for a `flow.executable` body.

These are the operations that make Flow recognizable. A program with only tensor utilities is not really in the dispatch-planning stage yet. A program with `flow.dispatch.region`, `flow.dispatch.workgroups`, or `flow.dispatch` is showing how IREE is partitioning work.

### Streamable Call Ops

- `flow.func` declares an external streamable function with tensor-shaped inputs and results.
- `flow.call` calls a `flow.func` while preserving shape and tied-result information.

These operations model host or external calls that fit into the asynchronous stream model. They are less common than dispatches in ordinary compute-heavy code, but they matter when the compiler must call an external implementation while keeping tensor dataflow explicit.

### Tensor Ops

- `flow.tensor.constant` creates a tensor constant.
- `flow.tensor.dynamic_constant` creates a constant-like tensor with a dynamic result shape, mainly for testing and benchmarking dynamic shape behavior.
- `flow.tensor.tie_shape` ties dynamic dimensions to a tensor value.
- `flow.tensor.reshape` changes tensor shape without changing contents.
- `flow.tensor.bitcast` reinterprets tensor contents with a different element type or shape, while preserving total serialized size.
- `flow.tensor.load` reads an element or vector from a tensor.
- `flow.tensor.store` returns a tensor with an element updated.
- `flow.tensor.alloca` creates a transient tensor allocation with undefined contents. The source notes that this disables many allocation optimizations and usually indicates an undesirable lowering.
- `flow.tensor.empty` creates a tensor with undefined contents but without forcing the same allocation behavior as `flow.tensor.alloca`.
- `flow.tensor.splat` fills a tensor with one scalar value.
- `flow.tensor.clone` clones a tensor.
- `flow.tensor.encode` changes a tensor to an encoded layout or representation.
- `flow.tensor.barrier` prevents fusion or scheduling across a target affinity boundary.
- `flow.tensor.transfer` transfers a tensor to a target, inserting copies later if required.
- `flow.tensor.slice` extracts a tensor subregion.
- `flow.tensor.update` updates a tensor subregion with another tensor.
- `flow.tensor.trace` emits runtime tracing for tensor values.

These ops are the value-semantic tensor vocabulary of Flow. They are close to MLIR's tensor dialect, but they carry the extra shape, affinity, and IREE lowering semantics needed by the dispatch pipeline.

### Parameter Ops

- `flow.parameter.load` loads a tensor from a parameter archive or parameter source.
- `flow.parameter.write` writes a tensor to a parameter archive or parameter target and returns the tensor to preserve dataflow dependencies.

These are important for model weights and other externally stored parameter data. They let IREE keep parameter movement in the IR instead of treating it as an invisible runtime side effect.

### Channel and Collective Ops

- `flow.channel.default` gets the default collective communication channel.
- `flow.channel.split` splits a channel into subgroups.
- `flow.channel.rank` returns the local rank in a channel.
- `flow.channel.count` returns the number of participants in a channel.
- `flow.collective.all_gather` gathers values from all ranks and concatenates them.
- `flow.collective.all_reduce` reduces values across ranks.
- `flow.collective.all_to_all` exchanges values across ranks.
- `flow.collective.reduce_scatter` reduces and scatters results back to ranks.
- `flow.collective.send_recv` models grouped send and receive communication.

These operations show that Flow is not only about single-device kernels. It can also represent distributed or multi-participant tensor communication before later lowering maps those operations to stream/runtime mechanisms.

## Transformations and Conversions

The local source separates Flow-related transformations into two important sets: DispatchCreation passes and Flow passes.

DispatchCreation creates and prepares dispatches. It starts from tensor/linalg-style computation and forms Flow dispatch regions and workgroups.

- `iree-dispatch-creation-fusion-preprocessing` prepares IR for fusion.
- `iree-dispatch-creation-bitcast-unsupported-element-types` inserts Flow bitcasts around dispatch boundaries for element types unsupported by HAL.
- `iree-dispatch-creation-bubble-up-expand-shapes` moves reshape operations to improve fusion opportunities.
- `iree-dispatch-creation-insert-tensor-barriers` inserts compute-region barriers.
- `iree-dispatch-creation-remove-tensor-barriers` removes those barriers after they have served their purpose.
- `iree-dispatch-creation-elementwise-op-fusion` fuses elementwise tensor computations.
- `iree-dispatch-creation-fold-reshapes-into-tensor-barriers` controls reshape propagation around barriers.
- `iree-dispatch-creation-fold-unit-extent-dims` folds unit dimensions at module scope.
- `iree-dispatch-creation-fold-unit-extent-dims-for-func` folds unit dimensions inside a function.
- `iree-dispatch-creation-fuse-horizontal-contractions` combines related contraction operations that share a compatible left-hand side.
- `iree-dispatch-creation-fuse-multi-use-elementwise-producer` fuses elementwise producers even when they have multiple uses.
- `iree-dispatch-creation-sink-reshapes` moves reshapes toward consumers to enable fusion.
- `iree-dispatch-creation-split-reduction-ops` splits reductions to increase parallelism.
- `iree-dispatch-creation-tensor-pad-to-tensor-insert-slice` rewrites `tensor.pad` into fill plus insert-slice form.
- `iree-dispatch-creation-transpose-generic-ops` transposes generic linalg loop structure.
- `iree-dispatch-creation-form-scalar-dispatches` forms dispatch regions for scalar computations.
- `iree-dispatch-creation-form-dispatch-regions` creates `flow.dispatch.region` around root tensor computations.
- `iree-dispatch-creation-clone-producers-into-dispatch-regions` clones producers into dispatch regions before isolation.
- `iree-dispatch-creation-collapse-dimensions` collapses dimensions inside dispatch regions and hoists reshape work.
- `iree-dispatch-creation-hoist-uniform-scalar-compute` hoists scalar computation out of dispatch regions when it is uniform.
- `iree-dispatch-creation-set-split-reduction-sizes` annotates split-reduction choices.
- `iree-dispatch-creation-form-split-reduction-dispatches` builds dispatches for split reductions.
- `iree-dispatch-creation-annotate-data-tiling-hints` adds data-tiling hints to eligible linalg operations.
- `iree-dispatch-creation-set-encoding` introduces tensor encodings for Flow dispatch regions.
- `iree-dispatch-creation-hoist-encoding-ops` hoists encoding operations out of Flow dispatch regions.
- `iree-dispatch-creation-fuse-encoding-ops-into-dispatch-regions-pass` fuses encoding operations into producer dispatch regions or forms new dispatches.
- `iree-dispatch-creation-propagate-encodings` propagates encodings through other tensor operations.
- `iree-dispatch-creation-convert-encoding-to-flow` converts top-level encoding operations to Flow operations.
- `iree-dispatch-creation-convert-dispatch-regions-to-workgroups` rewrites `flow.dispatch.region` into `flow.dispatch.workgroups`.
- `iree-dispatch-creation-convert-tensor-to-flow` converts tensor operations to `flow.tensor.*` operations.
- `iree-dispatch-creation-materialize-default-workgroup-count-region` creates default workgroup-count regions for dispatch workgroups.

The Flow transformation pipeline then cleans up, outlines, annotates, and prepares the IR for Stream lowering.

- `iree-verify-input-legality` checks that IR entering Flow is legal for the pipeline.
- `iree-top-level-scf-to-cfg` converts top-level SCF constructs to CFG form where needed.
- `iree-flow-initialize-empty-tensors` initializes empty tensors to either zero or uninitialized allocations.
- `iree-flow-capture-dynamic-dims` captures dynamic dimensions required by dispatch operands, results, and control flow.
- `iree-flow-canonicalize` applies Flow-specific canonicalization.
- `iree-flow-convert-to-flow` is a test-oriented conversion pass that applies tensor-to-Flow patterns.
- `iree-flow-outline-dispatch-externs` outlines external dispatches into executable structures.
- `iree-flow-outline-dispatch-regions` outlines `flow.dispatch.workgroups` into `flow.executable` and replaces them with `flow.dispatch`.
- `iree-flow-annotate-dispatches` annotates executable dispatches based on their contents.
- `iree-flow-insert-debug-target-at-ordinal` inserts debug or trace targets by dispatch ordinal.
- `iree-flow-insert-debug-target-at-symbol` inserts debug or trace targets by dispatch symbol.
- `iree-flow-deduplicate-executables` deduplicates equivalent executables.
- `iree-flow-export-benchmark-funcs-pass` exports benchmark functions for entry points and dispatch executables.
- `iree-flow-inject-dispatch-tracing` traces dispatch inputs and outputs.
- `iree-flow-inject-tensor-tracing` inserts tensor tracing for annotated operations.
- `iree-flow-cleanup-tensor-shapes` removes remaining Flow tensor shape metadata after lowering.
- `iree-flow-outline-constants` outlines tensor constants into module-level globals.
- `iree-flow-replicate-globals-per-affinity` specializes globals for different affinities.
- `iree-flow-dump-dispatch-graph-pass` dumps a Graphviz view of dispatch dataflow and control flow.

The downstream conversion is handled by the Stream phase. The pass `iree-stream-conversion` converts supported input dialects, including `flow`, into the `stream` dialect. The implementation populates Flow-to-Stream conversion patterns for Flow tensor operations, parameters, channels, collectives, dispatches, executable ops, calls, workgroup information ops, and returns.

Flow also provides two transform-dialect extension operations:

- `transform.iree.forall_to_flow` rewrites `scf.forall` to `flow.dispatch.workgroups`.
- `transform.iree.region_to_workgroups` rewrites `flow.dispatch.region` to `flow.dispatch.workgroups`.

These are not `flow.*` operations, but they are important if you are using MLIR's transform dialect to script or control IREE dispatch formation.

## What Flow Implies

Seeing Flow in IR implies that IREE has crossed from "what tensor computation exists" into "how the compiler wants to package computation." A `flow.dispatch.region` implies a candidate fused computation. A `flow.dispatch.workgroups` implies that the candidate has been turned into an isolated workgroup dispatch body. A `flow.dispatch` implies that the body has been outlined and is now called as an executable entry point.

Flow also implies that later stages still have major work to do. Buffers are not fully allocated, device command buffers are not fully scheduled, and target executables are not fully lowered. Workgroup size and workload information may still be symbolic. Tied operands communicate aliasing and in-place update opportunities, but the actual memory plan belongs to later Stream and HAL lowering.

This is why Flow is a good dialect for learning IREE's compiler architecture. It exposes the shape of execution without hiding everything inside runtime calls, but it is still high-level enough that you can read tensor values, dispatch boundaries, dynamic dimensions, and fusion decisions directly.

## How To Read Flow IR

Start by finding `flow.dispatch.region`, `flow.dispatch.workgroups`, and `flow.dispatch`. These tell you where the compiler has partitioned computation.

Then inspect the operands and result types. Dynamic dimensions in braces are part of the ABI between the host program and dispatch body. If a result is written as tied to an input, the compiler is preserving an in-place or aliasing relationship.

Inside `flow.dispatch.workgroups`, look for `flow.dispatch.workgroup.id`, `flow.dispatch.workgroup.count`, and `flow.dispatch.workgroup.size`. These tell you whether the body is written in terms of the symbolic grid. You should not expect final device-specific launch details here.

Finally, look for `flow.tensor.barrier`, `flow.tensor.transfer`, and `flow.tensor.encode`. These are clues about placement, scheduling, copy boundaries, and layout. They often explain why two operations did not fuse or why a copy appears later.

## Minimal Example

This simplified fragment shows the two main shapes of Flow. First, work is represented inline as a dispatch region:

```mlir
%result = flow.dispatch.region -> (tensor<4xf32>) {
  %empty = flow.tensor.empty : tensor<4xf32>
  flow.return %empty : tensor<4xf32>
}
```

After dispatch-region conversion, the work becomes an isolated workgroup dispatch:

```mlir
%result = flow.dispatch.workgroups[%n](%input)
    : (tensor<?xf32>{%n}) -> tensor<?xf32>{%n} = {
  %x = flow.dispatch.workgroup.id[0] : index
  %count = flow.dispatch.workgroup.count[0] : index
  flow.return
}
```

After outlining, the body is moved into a `flow.executable`, an entry point is exposed with `flow.executable.export`, and the host-side program calls it with `flow.dispatch`.

## Beginner Checklist

When studying Flow, ask these questions:

- What dispatch regions did the compiler form?
- Which operands cross each dispatch boundary?
- Which dynamic dimensions are carried explicitly?
- Are results tied to operands?
- Did tensor ops become `flow.tensor.*` operations?
- Were encodings, barriers, or transfers inserted?
- Did `flow.dispatch.workgroups` get outlined into `flow.executable` plus `flow.dispatch`?
- Is the next phase expected to convert this to Stream with `iree-stream-conversion`?

If you can answer those questions, you understand the main job of the Flow dialect: it turns high-level tensor dataflow into explicit dispatch dataflow that IREE can lower into asynchronous runtime execution.
