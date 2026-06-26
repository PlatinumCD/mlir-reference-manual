# IREE IO Parameters Dialect

The `io_parameters` dialect is an IREE module dialect for external parameter
resource management. It represents runtime reads and writes of parameter data,
such as model weights or other large constants, through device-aware
asynchronous I/O operations.

For a beginner, the main idea is that not every constant in a compiled program
has to be embedded directly in the executable. Large model parameters can live
outside the program in a parameter archive, memory-mapped file, cache, shared
storage object, or other provider. The compiled program then needs IR that says
"load this parameter by name" or "gather these parameter slices into this
buffer." The `io_parameters` dialect is where IREE represents those operations
after stream-level parameter commands have been lowered to HAL buffers and
fences, but before final VM calls are emitted.

## Why This Dialect Exists

IREE targets environments where model data may be large, shared, cached,
memory-mapped, sharded, or loaded directly by device-aware runtime APIs. Keeping
all of that data inline in the compiled module can make binaries large and can
prevent the runtime from using efficient storage and transfer strategies.

The `io_parameters` dialect exists to give the compiler an explicit IR layer for
parameter I/O. It can name parameters by scope and key, issue asynchronous loads
on a device timeline, batch many reads into one gather operation, and batch many
writes into one scatter operation. That gives backends and runtime providers a
chance to handle parameter access efficiently.

The dialect also separates two related concerns. Archive passes can export and
import parameter contents at compile time. Runtime operations can load, gather,
and scatter parameter contents at execution time. Both are part of the same
external-parameter story, but they happen at different points in the pipeline.

## When It Is Important

You will usually encounter `io_parameters` when working with IREE models that
externalize constants or weights. It is important when:

- large constants are stored in `.irpa` parameter archives instead of embedded
  directly in the compiled module,
- a program needs to read parameter data asynchronously on a HAL device queue,
- many parameter slices should be batched into one runtime call,
- parameter data should be loaded into buffers with specific memory types and
  usage flags,
- Stream dialect parameter commands are being lowered to HAL and VM level IR,
- the final IREE VM program needs imports for runtime parameter provider calls.

You generally do not write this dialect by hand. It is a compiler-internal
lowering dialect. You read it when debugging IREE parameter loading, archive
import/export behavior, or the lowering path from Stream to VM.

## Core Model

A parameter is identified by a key and, optionally, a scope. The key names the
specific parameter. The scope distinguishes groups of parameters, which matters
when multiple model parts or parameter sets are compiled together. The dialect
allows unscoped parameters, but the source description strongly recommends
scopes to avoid collisions.

The runtime operations are asynchronous and device-aware. Each operation carries
a HAL device, queue affinity, wait fence, and signal fence. The wait fence says
what work must complete before the parameter operation starts. The signal fence
says what later work can wait on after the parameter operation is queued or
completed according to runtime semantics.

The data itself is expressed in terms of HAL buffers and byte-like spans.
`io_parameters.load` creates result buffers. `io_parameters.gather` copies
parameter spans into an existing target buffer. `io_parameters.scatter` copies
spans from an existing source buffer into parameters.

## Operation Inventory

The current `io_parameters` dialect defines three operations.

| Operation | Role |
| --- | --- |
| `io_parameters.load` | Asynchronously reads one or more parameters and returns HAL buffers containing the results. |
| `io_parameters.gather` | Asynchronously gathers one or more parameter spans into a single target HAL buffer. |
| `io_parameters.scatter` | Asynchronously scatters spans from one source HAL buffer into one or more parameters. |

The operation set is small because the dialect is focused. It does not model
general file I/O, general tensors, or arbitrary resource management. It models
parameter access in the form needed by IREE's HAL and VM lowering pipeline.

## `io_parameters.load`

`io_parameters.load` reads one or more parameters from an external parameter
provider and returns one HAL buffer per requested parameter span. It takes a
device, queue affinity, wait fence, signal fence, optional source scope, source
keys, source offsets, memory type requirements, buffer usage requirements, and
result lengths.

This operation is useful when the program wants the runtime to allocate or
produce buffers for parameter data. Depending on the provider and target memory
requirements, a result may alias cached storage, map directly to the parameter
origin, or be produced by an allocate-and-read style operation. The IR leaves
that choice to the lowering and runtime implementation.

The verifier requires the source keys, source offsets, and result sizes to be
one-to-one. In other words, every requested key has a matching source offset and
requested length. Its custom assembly groups each request as a parameter
reference plus offset, result type, and length.

Use `io_parameters.load` as the mental model for "give me buffers containing
these parameters."

## `io_parameters.gather`

`io_parameters.gather` reads one or more parameter spans into a single existing
target buffer. It takes the same kind of device, queue, and fence operands as
`io_parameters.load`, plus an optional source scope, source keys, source
offsets, a target buffer, target offsets, and target lengths.

This operation is equivalent to many parameter reads into different locations of
one buffer, but it exposes the batch to the compiler and runtime. If a provider
can combine lookups, reads, or copies, a gather gives it the chance to do that
with less overhead than a long sequence of individual calls.

The verifier requires source keys, source offsets, target offsets, and target
lengths to be one-to-one. The parser also enforces that all rows in a single
gather use the same scope and target resource. That is why the operation is a
single batch into one target buffer rather than a general list of unrelated
copies.

Use `io_parameters.gather` as the mental model for "read these parameter slices
into this buffer."

## `io_parameters.scatter`

`io_parameters.scatter` is the write-side counterpart to gather. It takes spans
from a single source HAL buffer and writes them into one or more parameters. Its
operands include the device, queue affinity, wait and signal fences, source
buffer, source offsets, source lengths, optional target scope, target keys, and
target offsets.

This is useful for runtime-updated parameters, caches, checkpoints, or any flow
where the program writes parameter storage rather than only reading from it. As
with gather, the point is batching: many logical writes can be expressed as one
operation.

The verifier requires source offsets, source lengths, target keys, and target
offsets to be one-to-one. The custom parser enforces that all rows use the same
source resource and scope. This keeps the scatter operation shaped like one
batched transfer from one buffer to one parameter scope.

Use `io_parameters.scatter` as the mental model for "write these source buffer
slices back to these parameters."

## Archive Passes

The module defines three named passes for parameter archive management.

| Pass | What it does |
| --- | --- |
| `iree-io-export-parameters` | Exports large serializable global constants to a parameter archive and replaces their initial values with named parameter references. |
| `iree-io-import-parameters` | Imports selected named parameters from archives back into global initial values when the type and size rules allow it. |
| `iree-io-generate-splat-parameter-archive` | Creates a `.irpa` archive with zero-splat entries for all parameters found in the module. |

`iree-io-export-parameters` scans `util.global` operations. If a global has a
serializable initial value and meets the configured minimum size, the pass adds
it to an archive builder. Immutable splat values can become splat archive
entries. Other serializable values become data entries. The pass then changes
the global initializer to a `flow` named parameter attribute containing the
scope and key.

`iree-io-import-parameters` goes the other direction. It opens one or more
archives, grouped by optional scope, and looks for globals initialized by named
parameter attributes. It imports a parameter when its key is explicitly listed
or when its storage size is below the configured maximum. Currently, import is
limited to shaped integer or floating-point element types with common storage
widths, and packed types are not yet supported.

`iree-io-generate-splat-parameter-archive` is a utility pass. It finds all named
parameters referenced by globals and flow tensor constants, then writes a
parameter archive with default zero-splat entries. This is useful for producing
a placeholder archive that has the right parameter names and sizes.

## Conversions

The dialect registers two conversion interfaces rather than standalone pass
names in this directory.

| Conversion | Direction | What it does |
| --- | --- | --- |
| Stream to IO Parameters | `stream.cmd.parameter.*` to `io_parameters.*` | Converts Stream parameter commands into HAL-buffer-level `io_parameters` operations with devices, queue affinities, and fences. |
| IO Parameters to VM | `io_parameters.*` to VM calls | Converts `io_parameters.load`, `io_parameters.gather`, and `io_parameters.scatter` into calls to VM imports from `io_parameters.imports.mlir`. |

The Stream conversion maps `stream.cmd.parameter.load` to
`io_parameters.load`. It maps parameter read and gather commands to
`io_parameters.gather`, and parameter write and scatter commands to
`io_parameters.scatter`. During that lowering it also resolves device and queue
affinity, creates or forwards fences, and chooses memory properties for loaded
buffers.

The VM conversion makes the runtime ABI explicit. It lowers the three dialect
operations to VM calls named `io_parameters.load`, `io_parameters.gather`, and
`io_parameters.scatter`. It builds key tables and key data buffers for parameter
names. When all keys are compile-time constants, the conversion uses inline
rodata tables. When keys are dynamic, it constructs the key table and key data
at runtime. It also builds an indirect spans buffer containing the parameter
offsets, buffer offsets, and lengths.

This conversion also handles the optional scope. A missing scope becomes a zero
VM buffer reference. A present scope is passed through as the scope buffer.

## How To Read This Dialect In IR

Start by identifying the operation shape. If it returns buffers, it is
`io_parameters.load`. If it writes into an existing target buffer, it is
`io_parameters.gather`. If it reads from an existing source buffer and writes to
parameters, it is `io_parameters.scatter`.

Then read the synchronization operands. The device and queue affinity say where
the operation is queued. The wait fence tells you what the operation depends on.
The signal fence is the value later operations should wait on.

Next, read the parameter references. A reference may look like a scope and key
pair, or just a key. The bracketed value after the key is the offset into the
parameter. For gather and scatter, the other side of the arrow gives the buffer
offset and length.

Finally, look at surrounding dialects. If the IR still contains Stream
operations, parameter access has not yet been lowered to `io_parameters`. If the
IR is mostly VM operations and calls to `io_parameters.load`,
`io_parameters.gather`, or `io_parameters.scatter`, the dialect operations have
already been lowered to runtime imports.

## What It Implies

Seeing `io_parameters` in IR implies that parameter storage is externalized or
at least represented as externalizable runtime I/O. It also implies that the
compiler is preserving asynchronous device execution semantics. These
operations are not simple host-side file reads. They are queued with HAL devices
and fences so they can compose with dispatches and other device work.

The dialect also implies that batching may matter. `io_parameters.gather` and
`io_parameters.scatter` are not just convenience operations. They make many
logical parameter reads or writes visible as one operation, which can reduce
runtime call overhead and enable providers to optimize storage access.

A common pitfall is confusing compile-time archive manipulation with runtime
parameter I/O. The archive passes rewrite globals and files. The dialect ops
represent runtime actions. Another pitfall is treating scope as unimportant.
Unscoped parameters are legal, but scoped parameters are safer when multiple
parameter sets could otherwise reuse the same key names.

## Summary

The `io_parameters` dialect is IREE's compact IR layer for external parameter
I/O. It has three operations: `io_parameters.load` for producing buffers from
parameters, `io_parameters.gather` for reading parameter spans into a target
buffer, and `io_parameters.scatter` for writing source buffer spans back to
parameters. Its archive passes manage `.irpa` files and named parameter
attributes, while its conversion interfaces lower Stream parameter commands into
device-aware parameter ops and then lower those ops into VM runtime imports.
