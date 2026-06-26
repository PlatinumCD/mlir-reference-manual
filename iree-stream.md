# IREE Stream Dialect

## Beginner Summary

The IREE `stream` dialect is the compiler stage where tensor programs become
explicitly scheduled asynchronous programs. It sits between IREE `flow` and the
lower-level hardware abstraction layer, usually called HAL:

```text
flow.* -> stream.* -> hal.*
```

For a beginner, the central idea is that `stream` explains how tensor values
turn into byte-addressed resources, how those resources move between host and
device memory, how work is dispatched, and how asynchronous dependencies are
tracked. Earlier dialects can pretend that tensors are immutable SSA values.
The Stream dialect makes the hard execution questions explicit: where data
lives, how large it is, what lifetime it has, what device or affinity should
own it, what operations may run concurrently, and what timepoint says a result
is ready.

## Why This Dialect Exists

IREE compiles machine-learning programs to many runtime targets. A tensor
program may start as a clean mathematical dataflow graph, but a runtime cannot
execute abstract tensors directly. It needs buffers, command streams, device
queues, dispatches, synchronization objects, parameter loads, file reads, and
memory transfers.

The `stream` dialect exists to bridge that gap without immediately committing to
one device API. It is lower level than tensor IR, but still higher level than a
particular HAL backend. It gives IREE a place to decide scheduling and resource
management before the program is converted to HAL Inline or HAL Loader forms.

## When This Dialect Matters

Stream matters when you are reading the middle and late IREE compiler pipeline.
If you see `stream.tensor.*`, the compiler is still carrying logical tensor
information while preparing resource lowering. If you see `stream.async.*`, the
program has mostly moved to asynchronous resource operations. If you see
`stream.cmd.*`, the compiler is issuing more explicit command-style operations
that are close to HAL lowering.

This dialect is important for understanding memory use, transfer behavior,
dispatch boundaries, asynchronous ordering, concurrency, and target-specific
layout/encoding decisions. Many performance questions in IREE eventually pass
through Stream: unnecessary copies, poor resource lifetimes, over-serialization,
too many dispatch bindings, or tensors encoded in layouts that do not match the
target.

## When To Use It

Most users do not write Stream IR by hand. You use it when inspecting compiler
output, debugging IREE lowering, authoring IREE compiler passes, or adding a
new backend path.

Use Stream concepts when you need to answer questions such as:

- Has this tensor become a `!stream.resource` yet?
- Is this operation still in the tensor phase, async phase, or command phase?
- What timepoint protects this resource?
- Is the resource external, transient, variable, constant, staging, or unknown?
- Which dispatch executable receives this binding?
- Which pass is responsible for removing a copy or assigning a lifetime?

## Core Types

`!stream.timepoint` represents a point in the execution timeline. If a resource
is produced with a timepoint, it is not safe to use the resource until that
timepoint has been reached. Timepoints make ordering explicit while still
allowing the compiler to move waits, join dependencies, and preserve
concurrency.

`!stream.resource<...>` represents managed storage. The lifetime parameter is
critical:

| Lifetime | Meaning |
| --- | --- |
| `!stream.resource<*>` | Lifetime has not been analyzed yet. |
| `!stream.resource<external>` | Storage is externally managed, often at an ABI boundary. |
| `!stream.resource<staging>` | Staging storage used for upload/download paths. |
| `!stream.resource<transient>` | Short-lived storage used across stream operations. |
| `!stream.resource<variable>` | Long-lived mutable storage. |
| `!stream.resource<constant>` | Immutable long-lived storage, often program constants. |

`!stream.channel` represents a participant in a collective communication group.
`!stream.file` represents a file handle for async file I/O into or out of
resources. `!stream.binding` represents a resource binding visible inside an
executable dispatch function. `!stream.test.fence` is test-only infrastructure
for Stream timeline tests.

## Core Attributes

Important attributes include `#stream.collective`, which describes collective
operations such as all-gather, all-reduce, broadcast, send, and receive;
`#stream.partitioning_config`, which guides partitioning tradeoffs such as
debuggability, peak memory, or concurrency; `#stream.resource_config`, which
captures resource constraints such as allocation limits, alignment, range
limits, index bit width, aliasing policy, and memory model; and
`#stream.parameter.named`, which names externally defined parameter storage.

These attributes are not decorative. They guide how Stream chooses memory
layouts, dispatch boundaries, transfer behavior, and target-specific lowering.

## The Phase Model

The easiest way to understand Stream is as a sequence of lowering phases.

`stream.tensor.*` operations still talk about tensors, shapes, encodings, and
tensor imports/exports. This phase preserves high-level information needed to
choose storage sizes and layouts.

`stream.async.*` operations work on `!stream.resource` values and represent
asynchronous work. They model allocation, copies, fills, updates, dispatches,
collectives, function calls, and timepoint-producing execution regions.

`stream.cmd.*` operations are later command-phase operations. They are closer to
what HAL command buffers and runtime issue paths need: flushes, invalidates,
copies, fills, dispatches, serial/concurrent command regions, and command-side
parameter I/O.

Timepoint operations cut across the phases. They let Stream represent when work
is complete without forcing every operation to become a blocking host wait.

## Operation Inventory: Resources, Files, And Tensors

| Operation | Meaning |
| --- | --- |
| `stream.context.resolve` | Resolves low-level context resources for an affinity. |
| `stream.resource.alloc` | Allocates persistent resource storage. |
| `stream.resource.alloca` | Allocates transient asynchronous resource storage. |
| `stream.resource.dealloca` | Frees transient storage after a timepoint. |
| `stream.resource.retain` | Retains ownership of a resource. |
| `stream.resource.release` | Releases an ownership claim and reports terminal ownership. |
| `stream.resource.is_terminal` | Checks whether the current owner is the last owner. |
| `stream.resource.size` | Returns resource storage size in bytes. |
| `stream.resource.try_map` | Tries to map host memory into a resource. |
| `stream.resource.load` | Loads a value from a resource byte range. |
| `stream.resource.store` | Stores a value into a resource byte range. |
| `stream.resource.pack` | Packs multiple resource slices into a larger resource layout. |
| `stream.resource.constants` | Represents constant resource storage. |
| `stream.resource.subview` | Creates a resource subview. |
| `stream.resource.transients` | Represents user-provided transient storage. |
| `stream.file.constant` | Creates or references a file handle constant. |
| `stream.file.read` | Reads file data into a resource. |
| `stream.file.write` | Writes resource data to a file. |
| `stream.tensor.import` | Imports an external tensor-like value into Stream. |
| `stream.tensor.export` | Exports a Stream tensor/resource value outward. |
| `stream.tensor.sizeof` | Computes the encoded storage size of a tensor. |
| `stream.tensor.empty` | Creates an uninitialized tensor value. |
| `stream.tensor.constant` | Creates a tensor from constant or parameter data. |
| `stream.tensor.splat` | Creates a tensor filled with one scalar value. |
| `stream.tensor.clone` | Clones a tensor value. |
| `stream.tensor.encode` | Applies an encoding transformation to a tensor. |
| `stream.tensor.slice` | Produces a tensor slice. |
| `stream.tensor.fill` | Fills a tensor range. |
| `stream.tensor.update` | Updates part of a tensor with another tensor. |
| `stream.tensor.load` | Loads a scalar or vector from a tensor. |
| `stream.tensor.store` | Stores a scalar or vector into a tensor. |
| `stream.tensor.trace` | Emits tensor tracing/debug information. |
| `stream.tensor.dispatch` | Dispatches executable work over tensor operands. |
| `stream.tensor.parameter.load` | Loads tensor data from a parameter scope. |
| `stream.tensor.parameter.write` | Writes tensor data to a parameter scope. |

## Operation Inventory: Async Operations

| Operation | Meaning |
| --- | --- |
| `stream.async.alloca` | Allocates an async resource. |
| `stream.async.constant` | Creates an async constant resource. |
| `stream.async.splat` | Creates an async resource filled with one value. |
| `stream.async.clone` | Clones an async resource. |
| `stream.async.slice` | Creates an async resource slice. |
| `stream.async.fill` | Fills an async resource range. |
| `stream.async.update` | Updates a target resource with another resource. |
| `stream.async.copy` | Copies bytes between async resources. |
| `stream.async.collective` | Performs a collective operation over resources. |
| `stream.async.barrier` | Creates a dependency barrier for async resources. |
| `stream.async.transfer` | Transfers a resource between compatible lifetimes or affinities. |
| `stream.async.cast` | Casts a resource lifetime/type view. |
| `stream.async.load` | Loads a value from an async resource. |
| `stream.async.store` | Stores a value into an async resource. |
| `stream.async.dispatch` | Dispatches executable work over async resources. |
| `stream.async.func` | Declares an async streamable function. |
| `stream.async.call` | Calls an async streamable function. |
| `stream.async.execute` | Groups async work into a timepoint-producing execution region. |
| `stream.async.concurrent` | Groups async work that may run concurrently. |
| `stream.async.parameter.load` | Loads async resources from parameters. |
| `stream.async.parameter.read` | Reads a resource from a parameter scope. |
| `stream.async.parameter.write` | Writes a resource to a parameter scope. |
| `stream.async.parameter.gather` | Gathers multiple parameter resources. |
| `stream.async.parameter.scatter` | Scatters multiple resources to parameters. |

## Operation Inventory: Command Operations

| Operation | Meaning |
| --- | --- |
| `stream.cmd.flush` | Flushes resource writes for command execution. |
| `stream.cmd.invalidate` | Invalidates resource ranges before reads. |
| `stream.cmd.discard` | Discards resource contents before overwrite. |
| `stream.cmd.fill` | Fills a resource range. |
| `stream.cmd.copy` | Copies a resource range. |
| `stream.cmd.collective` | Issues a command-phase collective. |
| `stream.cmd.dispatch` | Issues a command-phase executable dispatch. |
| `stream.cmd.func` | Declares a command-phase streamable function. |
| `stream.cmd.call` | Calls a command-phase streamable function. |
| `stream.cmd.execute` | Executes a dependency-aware command sequence. |
| `stream.cmd.serial` | Executes nested command ops serially. |
| `stream.cmd.concurrent` | Executes nested command ops concurrently. |
| `stream.cmd.parameter.load` | Loads command-phase resources from parameters. |
| `stream.cmd.parameter.read` | Reads a parameter resource. |
| `stream.cmd.parameter.write` | Writes a parameter resource. |
| `stream.cmd.parameter.gather` | Gathers parameter resources. |
| `stream.cmd.parameter.scatter` | Scatters resources to parameters. |

## Operation Inventory: Synchronization, Dispatch, And Utilities

| Operation | Meaning |
| --- | --- |
| `stream.timepoint.immediate` | Produces an already-reached timepoint. |
| `stream.timepoint.import` | Imports an external synchronization object as a timepoint. |
| `stream.timepoint.export` | Exports a timepoint to an external type. |
| `stream.timepoint.chain_external` | Chains a timepoint through an external synchronization value. |
| `stream.timepoint.join` | Joins multiple timepoints. |
| `stream.timepoint.barrier` | Produces a timepoint for resource availability. |
| `stream.timepoint.await` | Waits for a timepoint before exposing resources. |
| `stream.channel.create` | Creates a collective communication channel. |
| `stream.channel.split` | Splits a channel by color and key. |
| `stream.channel.rank` | Returns the local rank in a channel. |
| `stream.channel.count` | Returns the participant count in a channel. |
| `stream.executable` | Defines a generic executable module. |
| `stream.executable.end` | Terminates a `stream.executable`. |
| `stream.executable.export` | Defines a dispatch entry point. |
| `stream.binding.subspan` | Creates an alias to subspan data in an executable binding. |
| `stream.dispatch.workgroup.id` | Returns the current workgroup id dimension. |
| `stream.dispatch.workgroup.count` | Returns the workgroup count dimension. |
| `stream.dispatch.workgroup.size` | Returns the workgroup size dimension. |
| `stream.return` | Returns from Stream regions/functions. |
| `stream.yield` | Yields from Stream execution regions. |
| `stream.test.timeline_op` | Test-only timeline interface operation. |
| `stream.test.timeline_aware` | Test-only timeline-aware operation. |

## Transformation Inventory

| Pass | Role |
| --- | --- |
| `iree-stream-conversion` | Converts supported Flow, tensor, util, HAL ABI, and control-flow inputs into Stream. |
| `iree-stream-split-parameter-encoder` | Splits compatible parameter encoding work into a separate module. |
| `iree-stream-encode-host-tensors` | Lowers host-side tensor encodings to async resource forms. |
| `iree-stream-encode-device-tensors` | Encodes executable binding tensors for device-side expectations. |
| `iree-stream-materialize-builtins` | Replaces unsupported operations with builtin dispatches. |
| `iree-stream-materialize-copy-on-write` | Expands implicit immutable-resource semantics into explicit clones or rematerialization. |
| `iree-stream-materialize-encodings` | Turns `stream.tensor.encode` operations into dispatches and executables. |
| `iree-stream-clone-to-consumers` | Clones eligible operations per consumer affinity. |
| `iree-stream-elide-async-copies` | Removes async clones, transfers, or slices that do no useful work. |
| `iree-stream-emplace-allocations` | Places results directly into existing resources when safe. |
| `iree-stream-refine-usage` | Assigns fixed resource lifetimes and inserts transfers when needed. |
| `iree-stream-schedule-execution` | Groups async operations into executable regions. |
| `iree-stream-schedule-concurrency` | Groups operations inside execution regions into concurrent streams. |
| `iree-stream-sync-initializers` | Converts initializer-produced timepoints into synchronous waits. |
| `iree-stream-propagate-timepoints` | Moves timepoints through calls, globals, and control flow to avoid unnecessary waits. |
| `iree-stream-elide-timepoints` | Removes waits covered by dependent timepoints. |
| `iree-stream-schedule-allocation` | Converts implicit async resource management into explicit command-style allocation and deallocation. |
| `iree-stream-automatic-reference-counting` | Inserts retains, releases, and deallocations for async resources. |
| `iree-stream-pack-constants` | Packs constant resources and materializes initialization operations. |
| `iree-stream-layout-slices` | Lays out packed resource slices with target-aware alignment and offsets. |
| `iree-stream-reuse-allocations` | Reuses compatible transient allocations when lifetime ordering permits. |
| `iree-stream-emplace-transients` | Places transient allocations into user-provided storage buffers. |
| `iree-stream-materialize-transient-size-queries` | Generates query functions for transient storage size requirements. |
| `iree-stream-annotate-constant-transient-size` | Adds reflection metadata for constant transient sizes. |
| `iree-stream-fold-uniform-operands` | Folds uniformly passed dispatch operands. |
| `iree-stream-fuse-dispatch-bindings` | Fuses dispatch bindings that share underlying storage. |
| `iree-stream-specialize-dispatches` | Specializes executables based on dispatch-site operand patterns. |
| `iree-stream-unify-encoding-for-globals` | Chooses one encoding for shared immutable global data when possible. |
| `iree-stream-specialize-encodings` | Duplicates and specializes executables for resolved encoding layouts. |
| `iree-stream-annotate-dispatch-arguments` | Annotates dispatch operands and bindings with value/alignment facts. |
| `iree-stream-annotate-dispatch-assumptions` | Inserts executable-local assumptions derived from dispatch sites. |
| `iree-stream-pack-dispatch-operands` | Packs dispatch operands into i32 push constants. |
| `iree-stream-annotate-affinities` | Adds affinity annotations for debugging. |
| `iree-stream-dump-statistics` | Dumps Stream dialect statistics. |
| `iree-stream-verify-input` | Verifies that input dialects are supported by Stream lowering. |
| `iree-stream-verify-affinities` | Checks that operations have affinities assigned directly or indirectly. |
| `iree-stream-verify-lowering-to-tensors` | Checks that inputs have reached `stream.tensor.*` form. |
| `iree-stream-verify-lowering-to-async-resources` | Checks that tensor ops/types have lowered to async resource ops. |
| `iree-stream-verify-lowering-to-async` | Checks that tensor ops are gone and resource lifetimes are assigned. |
| `iree-stream-verify-async-access-ranges` | Checks async resource ranges are in bounds where possible. |
| `iree-stream-verify-lowering-to-cmd` | Checks that async ops/types have lowered to command-phase ops. |

## Conversion Paths

The entry conversion is `iree-stream-conversion`. It uses conversion patterns
from Flow-to-Stream, Util-to-Stream, Standard-to-Stream, and HAL-to-Stream
paths. Its job is not just to rename operations. It changes the representation
from implicit tensor dataflow to resource-aware Stream IR with symbolic storage
sizes.

The outbound conversions are usually owned by HAL module passes. The
`iree-hal-inline-conversion` pass uses Stream-to-HAL-Inline patterns, converting
Stream resource, file, tensor boundary, command, and timepoint operations into
HAL Inline or simpler surrounding IR. The `iree-hal-loader-conversion` pass
combines Stream-to-HAL-Inline patterns with Stream-to-HAL-Loader patterns,
especially for executable loading and dispatch through the loader module.

In other words, Stream is the middle layer: it receives high-level tensor work,
turns it into scheduled resource work, and then hands that work to HAL lowering.

## How To Read Stream IR

A typical Stream resource type includes both lifetime and size information:

```mlir
%r = stream.async.constant ... : !stream.resource<constant>{%size}
```

Read this as: `%r` is immutable storage with a known or symbolic byte size.

A timepoint-producing operation means the resource is not immediately safe to
use:

```mlir
%r, %t = stream.resource.alloca uninitialized : !stream.resource<transient>{%n}
       => !stream.timepoint
```

Read this as: the allocation has been scheduled, and `%t` tells later work when
the resource is available.

A dispatch connects host-side resource state to executable code:

```mlir
%r2 = stream.async.dispatch @executable::@entry(%r[%c0 to %n for %n])
    : (!stream.resource<*>{%n}) -> %r{%n}
```

Read this as: Stream is scheduling executable work over a resource slice and
tracking the resulting resource.

Inside executable code, bindings are made visible through
`stream.binding.subspan`, and workgroup metadata comes from
`stream.dispatch.workgroup.id`, `stream.dispatch.workgroup.count`, and
`stream.dispatch.workgroup.size`.

## What It Implies

Seeing Stream IR means the compiler is no longer only reasoning about tensor
values. It is reasoning about execution. A `stream.tensor.clone` may still look
tensor-like, but it is on a path toward resource copies. A `stream.async.copy`
has concurrency and lifetime implications. A `stream.timepoint.await` may force
a wait unless later propagation or elision can move it. A
`stream.resource<*>` means lifetime inference is still pending, while
`stream.resource<transient>` or `stream.resource<constant>` means the compiler
has made a resource classification decision.

The dialect also implies that target details are beginning to matter. Affinity,
resource constraints, encodings, memory model, and dispatch binding layout all
influence the final runtime behavior.

## Common Pitfalls

Do not treat `stream.tensor.*`, `stream.async.*`, and `stream.cmd.*` as
interchangeable spellings. They are different lowering phases.

Do not ignore timepoints. Using a resource before its timepoint is reached is
undefined in the Stream model.

Do not assume `!stream.resource<*>` is final. It means the lifetime is unknown
and a later pass such as `iree-stream-refine-usage` must resolve it.

Do not assume every copy is waste. Some copies are needed for correctness,
lifetime isolation, memory compatibility, or concurrency. The purpose of passes
such as `iree-stream-materialize-copy-on-write` and
`iree-stream-elide-async-copies` is to make required copies explicit and remove
the redundant ones.

Do not debug backend performance without checking Stream. Many performance
problems are visible here before HAL lowering: excessive timepoint waits,
unfused dispatch bindings, poor resource reuse, unnecessary transfers, or
encoding choices that force extra dispatches.

