# `acc` Dialect

## Beginner Summary

`acc` is the MLIR dialect for OpenACC.

OpenACC is a directive-based programming model. A C, C++, or Fortran frontend
can use it to say "this region can run in parallel", "this data should be
available on the device", or "this loop has gang, worker, or vector
parallelism" without immediately choosing a final GPU or CPU implementation.

The `acc` dialect preserves those OpenACC meanings in MLIR. It is not just a
runtime-call dialect. It models high-level OpenACC constructs, decomposed data
clauses, loop parallelism, atomic operations, routines, privatization recipes,
and lower-level code-generation forms used on the path toward GPU and LLVM IR.

You usually see `acc` after a frontend has parsed OpenACC directives and before
the compiler has fully lowered offload regions to GPU, SCF, LLVM, and runtime
calls.

## Why This Dialect Exists

OpenACC carries source-level intent that would be lost if a frontend lowered
directly to runtime calls.

For example, these source concepts need to remain visible to the compiler:

- compute constructs such as `parallel`, `kernels`, and `serial`;
- loop parallelism clauses such as `gang`, `worker`, `vector`, `seq`,
  `independent`, `auto`, `collapse`, and `tile`;
- structured and unstructured data lifetimes;
- entry and exit behavior for data clauses such as `copy`, `copyin`,
  `copyout`, `create`, `present`, `deviceptr`, `attach`, and `delete`;
- implicit data mapping rules;
- privatization, firstprivate, and reduction semantics;
- `routine` directives for functions called from accelerator code;
- host fallback behavior when an `if` clause disables device execution.

The dialect makes those semantics inspectable and transformable. That lets MLIR
run analysis, verification, canonicalization, region rewriting, GPU preparation,
and LLVM IR translation without throwing away OpenACC structure too early.

## When It Matters

`acc` matters when compiling OpenACC-enabled source code.

You are likely to see it when:

- a frontend lowers OpenACC pragmas or directives to MLIR;
- data clauses are decomposed into explicit entry and exit operations;
- implicit data attributes need to be generated;
- OpenACC compute regions are prepared for device execution;
- OpenACC loops are converted to `scf.for` or `scf.parallel`;
- routines are specialized for accelerator execution;
- host fallback code is created for OpenACC `if` clauses;
- OpenACC data operations are translated to LLVM IR runtime mapper calls.

It is less relevant for programs that do not use OpenACC. It is also not the
final GPU dialect. In a full offload pipeline, `acc` cooperates with `scf`,
`gpu`, `memref`, `llvm`, and target-specific dialects.

## When To Use It

Use `acc` when the source program has OpenACC semantics that should survive into
MLIR.

This includes:

- frontend lowering from C, C++, or Fortran OpenACC;
- compiler passes that reason about OpenACC data mapping or offload regions;
- transformations that prepare OpenACC loops for GPU-like parallel execution;
- lowering flows that must preserve source-level directive information for
  diagnostics or later decisions.

Do not use `acc` as a generic parallel programming dialect. If your input is
already target-independent loop IR, use `scf`, `affine`, `linalg`, or `gpu` as
appropriate. Use `acc` when the OpenACC programming model itself is part of the
meaning.

## Core Concepts

### High-Level Constructs

The dialect has operations that directly correspond to OpenACC constructs:

```mlir
acc.parallel
acc.kernels
acc.serial
acc.loop
acc.data
acc.enter_data
acc.exit_data
acc.host_data
acc.update
acc.init
acc.shutdown
acc.set
acc.wait
acc.routine
```

These are useful early because they look like the original program. A beginner
should read them as "OpenACC syntax, represented in MLIR form."

### Decomposed Data Clauses

OpenACC clauses often describe two phases: what happens when a region begins
and what happens when it ends. The `acc` dialect decomposes those phases into
separate operations.

For example, an OpenACC `copy(x)` clause is not a single `acc.copy` op.
Instead it is represented as:

- `acc.copyin` at region entry;
- `acc.copyout` at region exit;
- a `dataClause` attribute that remembers the original user spelling was
  `acc_copy`.

This makes dataflow explicit. The result of an entry operation is an accelerator
view of the value, and that result is threaded into compute or data constructs.

### Bounds

OpenACC data clauses can map array slices. `acc.bounds` records normalized
zero-based bounds information so the dialect does not need to understand every
source language's array model.

The type for a bounds value is:

```mlir
!acc.data_bounds_ty
```

Accessor operations expose parts of a bounds value:

```mlir
acc.get_lowerbound
acc.get_upperbound
acc.get_extent
acc.get_stride
```

### Structured And Unstructured Data

Structured data operations are tied to a lexical region, such as `acc.data` or
data clauses on `acc.parallel`. Unstructured data operations are standalone
directives such as `acc.enter_data` and `acc.exit_data`.

Many data operations carry a `structured` flag because OpenACC distinguishes
structured and dynamic reference counters. That distinction affects runtime
mapping behavior.

### Device Types

Several clauses can be specialized for particular OpenACC device types. The
main device-type values are:

- `none`
- `star`
- `default`
- `host`
- `multicore`
- `nvidia`
- `radeon`

Device-type-aware operands appear on operations such as `acc.parallel`,
`acc.kernels`, `acc.serial`, `acc.loop`, and data operations with `async` or
`wait` behavior.

### Parallel Levels

OpenACC uses a hierarchy of parallelism:

- gang;
- worker;
- vector;
- sequential.

The dialect represents high-level clauses on `acc.loop` and later codegen
parallel dimensions with attributes such as:

```mlir
#acc.par_dim<thread_x>
#acc.par_dims[block_x, thread_x]
```

The codegen mapping uses these to attach GPU-style execution dimensions to
SCF loops and compute regions.

### Recipes

Privatization, firstprivate, and reduction can require source-language-specific
initialization, copying, combining, or destruction. The `acc` dialect stores
those rules in recipe operations:

```mlir
acc.private.recipe
acc.firstprivate.recipe
acc.reduction.recipe
```

Recipe materialization later clones those regions into the construct that uses
them. This keeps the high-level OpenACC operation clean while still allowing
complex C++, Fortran, or frontend-specific behavior.

### Compute Lowering

High-level compute constructs can be lowered to:

```mlir
acc.kernel_environment
acc.compute_region
acc.par_width
```

`acc.kernel_environment` captures the data environment. `acc.compute_region`
contains the code intended for device execution. `acc.par_width` describes a
known or unknown launch width for a parallel dimension.

This is an intermediate OpenACC code-generation form. It is closer to GPU
execution than `acc.parallel`, but it is still in the `acc` dialect.

## Operations

The `acc` dialect defines 64 generated operations.

### Compute And Control Constructs

| Operation | Purpose |
|---|---|
| `acc.parallel` | Structured OpenACC parallel construct. Runs one region with programmer-directed parallelism. |
| `acc.kernels` | Structured OpenACC kernels construct. Lets the compiler identify and generate kernels from a region. |
| `acc.serial` | Structured OpenACC serial construct. Represents accelerator execution with serial semantics. |
| `acc.loop` | OpenACC loop construct. Carries loop bounds and clauses such as gang, worker, vector, seq, auto, independent, collapse, tile, cache, private, firstprivate, and reduction. |
| `acc.data` | Structured data region. Threads decomposed data-clause operands into a region. |
| `acc.host_data` | Host-data region, commonly used for `use_device` behavior. |
| `acc.enter_data` | Standalone unstructured data-entry directive. |
| `acc.exit_data` | Standalone unstructured data-exit directive. |
| `acc.update` | Standalone update directive. Uses decomposed `update_device` or `update_host` data operands. |
| `acc.init` | Runtime initialization directive. |
| `acc.shutdown` | Runtime shutdown directive. |
| `acc.set` | Runtime set directive, for example default device configuration. |
| `acc.wait` | Wait directive for asynchronous activity queues. |
| `acc.routine` | Represents an OpenACC routine directive for functions callable from accelerator code. |
| `acc.declare` | Implicit declare region. |
| `acc.declare_enter` | Entry side of an implicit declare data region. |
| `acc.declare_exit` | Exit side of an implicit declare data region. |
| `acc.global_ctor` | Holds global construction operations associated with OpenACC declare behavior. |
| `acc.global_dtor` | Holds global destruction operations associated with OpenACC declare behavior. |

### Data Entry Operations

These operations usually produce an accelerator-side value. That result is then
used by an `acc.data`, `acc.parallel`, `acc.kernels`, `acc.serial`,
`acc.enter_data`, `acc.update`, or related operation.

| Operation | Purpose |
|---|---|
| `acc.copyin` | Entry side of `copyin` and the entry side of `copy`. Copies host data to the device if needed. |
| `acc.create` | Entry side of `create` and `copyout`. Creates device storage without necessarily copying host contents. |
| `acc.present` | Requires that data is already present on the device. |
| `acc.nocreate` | Uses data if present without creating a new mapping. |
| `acc.deviceptr` | Says the variable is already a device pointer. |
| `acc.getdeviceptr` | Gets the device address for a variable, often when an exit operation has no visible matching entry operation. |
| `acc.attach` | Attaches a pointer field by updating the device copy with the device address of its pointee. |
| `acc.use_device` | Models `host_data use_device`. |
| `acc.update_device` | Entry-style operation for updating device data from host data. |
| `acc.declare_device_resident` | Data action for `declare device_resident`. |
| `acc.declare_link` | Data action for `declare link`. |
| `acc.cache` | Cache directive associated with a loop. |
| `acc.private` | Data operation for private variables. Usually references a private recipe. |
| `acc.firstprivate` | Data operation for firstprivate variables. Usually references a firstprivate recipe. |
| `acc.firstprivate_map` | Helper produced during recipe materialization so the firstprivate initial value is available on the device. |
| `acc.reduction` | Data operation for reduction variables. Usually references a reduction recipe. |

### Data Exit Operations

These operations consume an accelerator-side value and describe the exit action.

| Operation | Purpose |
|---|---|
| `acc.copyout` | Exit side of `copyout`, `copy`, and related clauses. Copies data back to the host. |
| `acc.delete` | Exit side of `create` or delete-like behavior. Removes a device mapping. |
| `acc.detach` | Reverse of `attach`. |
| `acc.update_host` | Exit-style operation for updating host data from device data. Also represents `update self`. |

### Bounds Operations

| Operation | Purpose |
|---|---|
| `acc.bounds` | Records normalized lower bound, upper bound, extent, stride, and start index for a data clause. |
| `acc.get_lowerbound` | Extracts a lower bound from `!acc.data_bounds_ty`. Missing lower bound means zero. |
| `acc.get_upperbound` | Extracts or computes an upper bound from bounds information. |
| `acc.get_extent` | Extracts or computes an extent from bounds information. |
| `acc.get_stride` | Extracts a stride. Missing stride means one. |

### Atomic Operations

| Operation | Purpose |
|---|---|
| `acc.atomic.read` | OpenACC atomic read. |
| `acc.atomic.write` | OpenACC atomic write. |
| `acc.atomic.update` | OpenACC atomic update with a region that computes the new value. |
| `acc.atomic.capture` | OpenACC atomic capture, combining update and read/capture behavior. |

### Recipe Operations

| Operation | Purpose |
|---|---|
| `acc.private.recipe` | Symbol operation defining how to initialize and optionally destroy a private value. |
| `acc.firstprivate.recipe` | Symbol operation defining how to initialize, copy, and optionally destroy a firstprivate value. |
| `acc.reduction.recipe` | Symbol operation defining reduction initialization, combination, and optional destruction. |
| `acc.yield` | Yields values from OpenACC regions and terminates many `acc` regions. |
| `acc.terminator` | Terminates OpenACC regions that do not yield values. |

### Code-Generation Helper Operations

These are lower-level than the frontend-like constructs.

| Operation | Purpose |
|---|---|
| `acc.kernel_environment` | Captures the data environment around a lowered compute region. |
| `acc.compute_region` | Isolated region prepared for device execution. Carries launch arguments and captured inputs. |
| `acc.par_width` | Represents the launch width for a GPU parallel dimension such as `thread_x` or `block_x`. |
| `acc.predicate_region` | Groups operations at intermediate loop-nest points where predication or synchronization may be needed. |
| `acc.privatize` | Creates a private handle for codegen. |
| `acc.private_local` | Materializes local storage for an `!acc.private_type` handle in the current parallel context. |
| `acc.unwrap_private` | Gets the underlying value from a private handle. |
| `acc.reduction_init` | Codegen operation for initializing a reduction value. |
| `acc.reduction_combine_region` | Codegen operation holding the inlined combiner region. |
| `acc.reduction_combine` | Combines a private reduction value into a shared reduction value. |
| `acc.reduction_accumulate` | Accumulates a partial value into a private reduction value. |

## Attributes And Types

### Types

| Type | Meaning |
|---|---|
| `!acc.data_bounds_ty` | Opaque bounds value produced by `acc.bounds`. |
| `!acc.declare_token` | Token returned by `acc.declare_enter` and consumed by `acc.declare_exit`. |
| `!acc.private_type<T>` | Codegen handle for privatized storage whose underlying type is `T`. |

`acc.compute_region` can also use `!gpu.async.token` as an optional stream
operand. That type comes from the `gpu` dialect, not from `acc`.

### Important Attributes And Enums

| Attribute or enum | Important values |
|---|---|
| `DeviceType` | `none`, `star`, `default`, `host`, `multicore`, `nvidia`, `radeon`. |
| `DataClause` | `acc_copyin`, `acc_copyin_readonly`, `acc_copy`, `acc_copyout`, `acc_copyout_zero`, `acc_present`, `acc_create`, `acc_create_zero`, `acc_delete`, `acc_attach`, `acc_detach`, `acc_no_create`, `acc_private`, `acc_firstprivate`, `acc_deviceptr`, `acc_getdeviceptr`, `acc_update_host`, `acc_update_self`, `acc_update_device`, `acc_use_device`, `acc_reduction`, `acc_declare_device_resident`, `acc_declare_link`, `acc_cache`, `acc_cache_readonly`. |
| `DataClauseModifier` | `none`, `zero`, `readonly`, `alwaysin`, `alwaysout`, `always`, `capture`. |
| `ReductionOperator` | `none`, `add`, `mul`, `max`, `min`, `iand`, `ior`, `xor`, `eqv`, `neqv`, `land`, `lor`, `maximum`, `minimum`, `maxnum`, `minnum`. |
| `VariableTypeCategory` | `uncategorized`, `scalar`, `array`, `composite`, `nonscalar`, `aggregate`. |
| `LoopParMode` | `seq`, `auto`, `independent`. |
| `ParLevel` | `seq`, `gang_dim1`, `gang_dim2`, `gang_dim3`, `worker`, `vector`. |
| `GangArgType` | `Num`, `Dim`, `Static`. |
| `CombinedConstructsType` | `kernels_loop`, `parallel_loop`, `serial_loop`. |
| `Construct` | IDs for OpenACC constructs such as parallel, kernels, loop, data, enter data, exit data, host data, atomic, declare, init, shutdown, set, update, routine, wait, runtime API, and serial. |
| `#acc.par_dim<...>` | One GPU parallel dimension used during codegen. |
| `#acc.par_dims[...]` | Ordered list of GPU parallel dimensions. |
| `#acc.var_name<...>` | Carries a source variable name for diagnostics and analysis. |
| `#acc.declare<...>` and `#acc.declare_action<...>` | Carry declare metadata on globals and declare actions. |

## Transformations

### Data And Semantic Completion

| Pass | Purpose |
|---|---|
| `acc-implicit-data` | Generates implicit data operations for variables used in OpenACC compute constructs. |
| `acc-implicit-declare` | Adds implicit `acc.declare` metadata for globals referenced in compute or routine regions. |
| `acc-implicit-routine` | Creates implicit `acc.routine` operations for functions called from accelerator regions. |
| `openacc-legalize-data-values` | Rewrites uses in compute regions to use the accelerator values produced by data-clause operations. |
| `offload-target-verifier` | Verifies that values and symbols used in offload regions are legal for the chosen execution model. |

The implicit-data pass is especially important for beginners. OpenACC lets many
variables get data attributes implicitly. In MLIR, those implicit choices become
explicit `acc.copyin`, `acc.firstprivate`, `acc.present`, or related operations.

### Region And Loop Rewriting

| Pass | Purpose |
|---|---|
| `acc-legalize-serial` | Rewrites `acc.serial` as `acc.parallel` with one gang, one worker, and vector length one. |
| `acc-loop-tiling` | Applies OpenACC `tile` clauses by creating tiled loop nests. |
| `acc-if-clause-lowering` | Splits compute constructs with `if` clauses into device and host paths using `scf.if`. |
| `offload-livein-value-canonicalization` | Sinks or rematerializes live-in values for regions that will be outlined. |
| `acc-emit-remarks-loop` | Emits optimization remarks explaining loop parallelism mapping. |

### Recipe And Privatization Lowering

| Pass | Purpose |
|---|---|
| `acc-recipe-materialization` | Clones private, firstprivate, and reduction recipe regions into the construct that uses them. |

This pass turns symbolic recipe references into actual IR. It may introduce
operations such as `acc.firstprivate_map`, `acc.reduction_init`, and
`acc.reduction_combine_region`.

### Compute And Device Specialization

| Pass | Purpose |
|---|---|
| `acc-compute-lowering` | Replaces `acc.parallel`, `acc.kernels`, and `acc.serial` with `acc.kernel_environment` plus `acc.compute_region`; lowers `acc.loop` to SCF loops with parallel-dimension attributes. |
| `acc-specialize-for-device` | Strips or inlines host-side OpenACC constructs when compiling code that is already specialized for device execution. |
| `acc-specialize-for-host` | Converts OpenACC operations to host-compatible forms, including atomic and loop lowering. |
| `acc-routine-lowering` | Creates specialized device versions of functions marked by `acc.routine`. |
| `acc-routine-to-gpu-func` | Moves routine functions into a GPU module as `gpu.func` operations. |
| `acc-bind-routine` | Rewrites calls in offload regions to use `bind(name)` targets from `acc.routine`. |
| `acc-declare-gpu-module-insertion` | Copies globals marked with `acc.declare` into the GPU module. |

`acc-compute-lowering` is the main structural lowering pass for compute
constructs. It is where high-level OpenACC compute regions become the
intermediate `acc.compute_region` representation.

## Conversions / Lowering Paths

### `convert-openacc-to-scf`

`convert-openacc-to-scf` is a conversion pass for conditional standalone data
operations.

It handles:

- `acc.enter_data` with an `if` condition;
- `acc.exit_data` with an `if` condition;
- `acc.update` with an `if` condition.

If the condition is dynamic, the operation is moved into an `scf.if` then
region and the `if` operand is removed from the cloned OpenACC operation. If
the condition is a constant true, the condition is removed. If it is a constant
false, the operation is erased.

### OpenACC Compute Lowering

`acc-compute-lowering` is not named as a conversion pass, but conceptually it is
the key OpenACC compute lowering step.

It rewrites:

- `acc.parallel`, `acc.kernels`, and `acc.serial` to `acc.kernel_environment`
  containing `acc.compute_region`;
- launch-related clauses to `acc.par_width`;
- `acc.loop` to `scf.parallel`, `scf.for`, or `scf.execute_region` depending
  on its context and parallelism mode;
- loop parallelism to `#acc.par_dims[...]` attributes.

This is where the IR stops looking like source-level OpenACC and starts looking
like a compiler-managed offload region.

### Host And Device Specialization

OpenACC `if` clauses and separate host/device compilation require
specialization.

Important paths:

- `acc-if-clause-lowering` builds explicit device and host paths for compute
  constructs with `if`.
- `acc-specialize-for-host` lowers OpenACC constructs to host-compatible IR for
  fallback execution.
- `acc-specialize-for-device` removes host-only OpenACC structure from device
  code.

These passes are why the dialect has both high-level operations and helper
operations. The compiler needs a place to stand while it splits one source
construct into different execution paths.

### Routine And GPU Lowering

OpenACC routines are handled by a sequence of transforms:

1. `acc-implicit-routine` creates routine declarations when calls from offload
   regions require them.
2. `acc-routine-lowering` creates specialized device function bodies wrapped in
   `acc.compute_region`.
3. `acc-bind-routine` applies `bind(name)` call targets.
4. `acc-routine-to-gpu-func` moves device routine functions into a GPU module
   as `gpu.func`.

### LLVM IR Translation

The OpenACC LLVM IR translation interface lowers a focused subset of `acc`:

- `acc.data`;
- `acc.enter_data`;
- `acc.exit_data`;
- `acc.update`;
- `acc.yield` and `acc.terminator`;
- helper data operations such as `acc.create`, `acc.copyin`, `acc.copyout`,
  `acc.delete`, `acc.update_device`, and `acc.getdeviceptr` as no-op carriers
  consumed by their parent data operation.

The translation emits OpenACC runtime mapper calls through LLVM's OpenMP IR
builder support. Other OpenACC operations must be lowered, specialized, or
removed before final LLVM IR translation. If an unsupported `acc` operation
reaches this translation path, the interface reports an error.

### Common Pipeline Shape

A simplified OpenACC lowering path looks like this:

```text
frontend OpenACC constructs
  -> acc dialect high-level constructs and data-clause operations
  -> implicit data / implicit declare / implicit routine
  -> recipe materialization
  -> if-clause lowering and host/device specialization as needed
  -> acc-compute-lowering
  -> loop/offload canonicalization and verification
  -> routine-to-gpu and GPU lowering where applicable
  -> convert-openacc-to-scf for standalone data/update if clauses
  -> LLVM dialect and OpenACC LLVM IR translation for data/runtime operations
```

Real pipelines vary. The main principle is stable: preserve OpenACC meaning
early, make implicit clauses explicit, lower compute regions structurally, then
specialize for host/device and translate runtime data operations late.

## Example IR

### A Simple Parallel Region With A Loop

```mlir
module {
  func.func @parallel_loop(%n : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    acc.parallel {
      acc.loop gang vector control(%i : index) =
          (%c0 : index) to (%n : index) step (%c1 : index) {
        acc.yield
      } attributes {independent = [#acc.device_type<none>]}
      acc.yield
    }
    return
  }
}
```

This preserves the source idea: a parallel region contains a loop with gang and
vector parallelism. Later passes decide how that maps to SCF and GPU dimensions.

### A Structured Data Region

```mlir
module {
  func.func @data_region(%a : memref<f32>) {
    %copy = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
    acc.data dataOperands(%copy : memref<f32>) {
      acc.terminator
    }
    acc.delete accPtr(%copy : memref<f32>)
    return
  }
}
```

`acc.copyin` creates the accelerator-side value. `acc.data` uses that value in
its data operand list. `acc.delete` models the exit action.

### A Lowered Compute Region

```mlir
module {
  func.func @compute_region(%data : memref<1024xf32>, %width : index) {
    %w = acc.par_width %width {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%tid = %w)
        ins(%arg0 = %data) : (memref<1024xf32>) {
      acc.yield
    } {origin = "acc.parallel"}
    return
  }
}
```

This is no longer a source-like `acc.parallel`. It is an isolated offload region
with an explicit launch-width operand and explicit captured inputs.

### Standalone Data With An `if` Clause

```mlir
module {
  func.func @enter_data_if(%a : memref<f32>, %cond : i1) {
    %create = acc.create varPtr(%a : memref<f32>) -> memref<f32>
    acc.enter_data if(%cond) dataOperands(%create : memref<f32>)
    return
  }
}
```

`convert-openacc-to-scf` can lower this to an `scf.if` containing an
unconditional `acc.enter_data`.

## Mental Model

Think of `acc` as a staged representation of OpenACC.

At the beginning, it looks like the source program:

```text
parallel, kernels, serial, loop, data, update, routine
```

In the middle, it makes hidden OpenACC semantics explicit:

```text
copyin, create, present, copyout, delete, bounds, recipes, implicit clauses
```

Later, it becomes an offload-oriented form:

```text
kernel_environment, compute_region, par_width, par_dims, reduction helpers
```

At the end, the remaining OpenACC data/runtime operations are translated to
LLVM IR runtime calls, while compute code is handled through SCF, GPU, and
target-specific lowering.

## Gotchas

- `acc` is semantic IR, not just syntax. A simple source clause may become
  multiple operations.
- Data clauses are decomposed. There is no single `acc.copy` operation.
- The result of data-entry operations is important. It represents the value to
  use for accelerator-side access.
- `acc.bounds` values are normalized and source-language agnostic. Frontends
  must provide the needed bounds information when the type system cannot.
- `device_type` operands and attributes must stay consistent. Many operations
  have helper APIs because manually editing the operand lists is error-prone.
- Recipes are symbolic until materialized. Seeing `acc.private.recipe` does not
  mean privatization has already been inlined into the compute region.
- `acc.compute_region` is isolated from above. Values used inside must be passed
  as launch or input arguments, or rematerialized inside.
- Direct OpenACC LLVM IR translation supports only a subset of operations.
  High-level compute constructs must be lowered or removed first.
- Host fallback and device specialization can produce very different IR from
  the same source directive.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/OpenACC/OpenACCBase.td`
- `mlir/include/mlir/Dialect/OpenACC/OpenACCOps.td`
- `mlir/include/mlir/Dialect/OpenACC/OpenACCCGOps.td`
- `mlir/include/mlir/Dialect/OpenACC/OpenACCOpsTypes.td`
- `mlir/include/mlir/Dialect/OpenACC/OpenACCAttributes.td`
- `mlir/include/mlir/Dialect/OpenACC/OpenACCCGAttributes.td`
- `mlir/include/mlir/Dialect/OpenACC/OpenACCOpsInterfaces.td`
- `mlir/include/mlir/Dialect/OpenACC/OpenACCTypeInterfaces.td`
- `mlir/lib/Dialect/OpenACC/IR/OpenACC.cpp`
- `mlir/lib/Dialect/OpenACC/IR/OpenACCCG.cpp`

Transform source files:

- `mlir/include/mlir/Dialect/OpenACC/Transforms/Passes.td`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCImplicitData.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCImplicitDeclare.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCImplicitRoutine.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCRecipeMaterialization.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCComputeLowering.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCLegalizeSerial.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCLoopTiling.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCIfClauseLowering.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCSpecializeForDevice.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCSpecializeForHost.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCRoutineLowering.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCRoutineToGPUFunc.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCBindRoutine.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCDeclareGPUModuleInsertion.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/ACCEmitRemarksLoop.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/OffloadLiveInValueCanonicalization.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/OffloadTargetVerifier.cpp`
- `mlir/lib/Dialect/OpenACC/Transforms/LegalizeDataValues.cpp`

Conversion and translation source files:

- `mlir/include/mlir/Conversion/OpenACCToSCF/ConvertOpenACCToSCF.h`
- `mlir/lib/Conversion/OpenACCToSCF/OpenACCToSCF.cpp`
- `mlir/include/mlir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h`
- `mlir/lib/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.cpp`

Tests:

- `mlir/test/Dialect/OpenACC`
- `mlir/test/Conversion/OpenACCToSCF`
- `mlir/test/lib/Dialect/OpenACC`
- `mlir/unittests/Dialect/OpenACC`
