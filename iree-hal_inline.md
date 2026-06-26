# IREE HAL Inline Dialect

The IREE `hal_inline` dialect is a low-level module dialect for a restricted, in-process version of IREE's Hardware Abstraction Layer. It exists for configurations where the compiler wants ABI compatibility with HAL buffer and buffer view concepts, but does not want the full asynchronous, multi-device, queue-based HAL runtime model.

The source description is deliberately narrow: `hal_inline` only operates synchronously, single-threaded, and on host-local buffers. That is the most important fact for beginners. The dialect is not a faster or simpler spelling of the full `hal` dialect. It is a constrained replacement for cases where inline execution is acceptable.

You will usually see `hal_inline` with `!hal.buffer`, `!hal.buffer_view`, and `!util.buffer`. It wraps host byte buffers as HAL buffers, creates buffer views for tensor-like metadata, performs queries and assertions, and then lowers to VM imports from an embedded `hal_inline` runtime module.

## Why This Dialect Exists

The full IREE HAL is designed to model real devices, command buffers, dispatches, queues, synchronization, memory types, buffer usage flags, executable variants, and other device-facing concerns. That model is necessary for serious GPU and accelerator execution, but it is heavier than needed for some standalone or host-local configurations.

The `hal_inline` dialect exists to preserve the ABI shape of HAL values while using host-local execution. It can wrap a `!util.buffer` in a `!hal.buffer`, create `!hal.buffer_view` values, query simple runtime configuration values, and map those operations to VM imports such as `hal_inline.buffer.allocate`, `hal_inline.buffer_view.create`, and `hal_inline.device.query.i64`.

This makes `hal_inline` useful when IREE wants to accept or produce functions whose public ABI uses HAL buffers or buffer views, but internally everything can run inline on the host. It is also intended to work with the `hal_loader` dialect, which carries similar usage restrictions.

## When To Use It

Use `hal_inline` when reading IREE modules built for inline HAL execution. It matters when stream resources are converted to host buffers, dispatch bodies are inlined into the host program, and asynchronous timepoints collapse into immediate resolved values.

It is most important in standalone or embedded-style compilation flows where the runtime does not need a full HAL device stack. It lets the compiler keep HAL-like buffer and buffer view ABI boundaries while avoiding full device scheduling.

Do not use `hal_inline` when you need real asynchronous device execution, multi-threaded command scheduling, non-host-local memory, external timepoints across the ABI, or full HAL resource lifetime tracking. The conversion code intentionally erases or simplifies many of those concepts.

## Core Model

The dialect has three operation groups:

| Group | Meaning |
| --- | --- |
| Buffer ops | Allocate, wrap, slice, measure, and expose backing storage for `!hal.buffer`. |
| Buffer view ops | Create and inspect `!hal.buffer_view` values, assert shape/type compatibility, and trace views. |
| Device ops | Query small runtime configuration values by category and key. |

A `!hal.buffer` in this dialect is usually backed by a `!util.buffer`. Some ops return both values. For example, `hal_inline.buffer.allocate` returns a HAL buffer and the underlying util storage buffer. `hal_inline.buffer.storage` recovers that host storage when later lowering wants byte-level access.

A `!hal.buffer_view` is a buffer plus metadata: source buffer, offset, length, element type, encoding type, and shape. The inline dialect can create views and query their metadata, but it does not imply real device-side tensor execution by itself.

## Operation Inventory

| Operation | What It Means |
| --- | --- |
| `hal_inline.buffer.allocate` | Allocates a new HAL buffer of at least the requested allocation size and returns both the HAL buffer and its `!util.buffer` storage. |
| `hal_inline.buffer.allocate.initialized` | Allocates a HAL buffer initialized from a slice of an existing `!util.buffer`. |
| `hal_inline.buffer.wrap` | Wraps host memory backed by a `!util.buffer` as a `!hal.buffer`. |
| `hal_inline.buffer.subspan` | Creates a HAL buffer reference to a subspan of another HAL buffer. |
| `hal_inline.buffer.length` | Returns the logical byte length of a HAL buffer. This may be smaller than the full underlying allocation. |
| `hal_inline.buffer.storage` | Returns the host backing storage of a HAL buffer as a `!util.buffer` subspan matching the HAL buffer's logical range. |
| `hal_inline.buffer_view.create` | Creates a `!hal.buffer_view` from a buffer, source offset, source length, element type, encoding type, and shape. |
| `hal_inline.buffer_view.assert` | Asserts that a buffer view has compatible tensor metadata, including shape, element type, and encoding. |
| `hal_inline.buffer_view.buffer` | Returns the HAL buffer backing a buffer view. |
| `hal_inline.buffer_view.element_type` | Returns the buffer view element type value. |
| `hal_inline.buffer_view.encoding_type` | Returns the buffer view encoding type value. |
| `hal_inline.buffer_view.rank` | Returns the rank of a buffer view. |
| `hal_inline.buffer_view.dim` | Returns one dimension value from a buffer view shape. |
| `hal_inline.buffer_view.trace` | Sends buffer views to a runtime trace sink under a string key. |
| `hal_inline.device.query` | Queries a runtime configuration value by category and key, returning an `ok` flag and a typed value. |

## Transformations And Passes

| Pass or Pattern | Role |
| --- | --- |
| `iree-hal-inline-executables` | Inlines translated executable functions into the host module. It extracts executable variant functions, builds dispatch wrapper functions, erases executable ops, and annotates stream dispatches with `hal_inline.target`. |
| `iree-hal-inline-conversion` | Converts supported stream, util, standard, and full HAL operations into the `hal_inline` dialect and supporting `util`, `arith`, `affine`, `func`, and `scf` IR. |
| `populateStreamToHALInlinePatterns` | Converts stream resources to `!hal.buffer` or `!util.buffer`, files to `!util.buffer`, and timepoints to `i64`. It also rewrites stream commands into direct buffer operations or direct calls. |
| `populateHALToHALInlinePatterns` | Converts full HAL buffer and buffer view operations into corresponding `hal_inline` operations or lower-level `util.buffer` operations. |
| `populateHALInlineToVMPatterns` | Lowers `hal_inline` operations to VM imports from the embedded `hal_inline` runtime module. |

## Conversion From Stream

The stream-to-inline conversion is the clearest place to see the dialect's restrictions.

Stream resources become buffers. External stream resources become `!hal.buffer` because they cross the public ABI boundary. Internal resources become `!util.buffer` because inline execution can use host storage directly. Resource allocation becomes `hal_inline.buffer.allocate`, while loads and stores become `util.buffer.load` and `util.buffer.store` against the resource's host storage.

Timepoints are simplified aggressively. Immediate and joined timepoints become constant zero `i64` values. Barriers and awaits simply return resources and resolved timepoints. Importing, exporting, or externally chaining timepoints is rejected because inline execution does not support timepoints across the ABI.

Command operations also become direct host operations. Flush, invalidate, and discard are erased. Fill and copy become `util.buffer.fill` and `util.buffer.copy`. Serial and concurrent command regions are inlined. Dispatch becomes a `util.call` to the function annotated by `iree-hal-inline-executables`.

Tensor import and export use buffer and buffer view operations. Importing from a HAL buffer can directly use the buffer. Exporting to a buffer view creates a `hal_inline.buffer_view.create` with element type, encoding type, and flattened shape operands.

## Conversion From Full HAL

The HAL-to-inline conversion preserves the parts of full HAL that make sense for inline execution. HAL element type, encoding type, memory type, and buffer usage constants become integer constants. HAL buffer subspan and length become `hal_inline.buffer.subspan` and `hal_inline.buffer.length`.

Full HAL buffer loads and stores do not remain HAL operations. The conversion obtains `hal_inline.buffer.storage`, obtains `hal_inline.buffer.length`, computes element byte size with `util.sizeof`, and uses `util.buffer.load` or `util.buffer.store`.

HAL buffer view creation, assertion, metadata accessors, dimension queries, and trace operations map directly to their `hal_inline.buffer_view.*` counterparts. This is the main ABI-preserving part of the dialect.

## Conversion To VM

When lowered to VM, each `hal_inline` op becomes a VM import call. The embedded `hal_inline.imports.mlir` module declares imports for buffer allocation, initialized allocation, wrapping, subspan, length, storage, buffer view creation, assertions, metadata queries, tracing, and device query.

Most op names map directly to import names, such as `hal_inline.buffer.allocate`, `hal_inline.buffer_view.rank`, and `hal_inline.buffer_view.trace`. `hal_inline.device.query` maps to the `hal_inline.device.query.i64` import. This import returns a pair: whether the query was recognized and the queried value.

This VM-import lowering is why the dialect is a module dialect rather than just a set of helper rewrites. It defines a runtime ABI surface.

## Canonicalization And Verification

`hal_inline.buffer.length` can fold by finding an existing size-aware value for the buffer. `hal_inline.buffer.storage` can fold through `hal_inline.buffer.allocate` and `hal_inline.buffer.allocate.initialized` to return the already-known storage result.

`hal_inline.buffer_view.create` canonicalizes through `hal_inline.buffer.subspan` by folding the subspan offset into the view creation offset and using the original source buffer. `hal_inline.buffer_view.buffer` can fold away when the buffer view was created in the same scope, returning the original source buffer directly.

`hal_inline.device.query` verifies that a default value, when provided as a typed attribute, has the same type as the query value result. This matters because device query keys are target-dependent and the fallback value must match the expected result type.

## How To Read HAL Inline IR

Start with buffer ownership. If an operation returns both `!hal.buffer` and `!util.buffer`, the HAL value is the ABI-facing handle and the util value is the host storage. Reads, writes, copies, and fills usually happen through the util buffer after storage extraction.

Next, inspect buffer views. A buffer view is metadata over a buffer, not a tensor computation. Check the element type, encoding type, offset, length, and shape operands. Assertions are runtime checks on that metadata.

Then look for dispatch calls. After `iree-hal-inline-executables`, stream dispatches should have been mapped to wrapper functions. After conversion, dispatches become direct `util.call` operations. That is the core "inline" behavior.

Finally, watch for erased synchronization. If you expected asynchronous behavior, `hal_inline` is the wrong model. Timepoints collapse, command regions inline, and external timepoint ABI operations are unsupported.

## What It Implies

Seeing `hal_inline` means IREE has chosen a host-local, synchronous execution path. It implies fewer runtime moving parts, but also fewer capabilities. There is no full device queue model, no real async timepoint boundary, and no general memory placement policy.

The dialect is valuable because it keeps HAL-shaped ABI values while simplifying execution to direct host operations and VM imports. For a beginner, it is a concrete example of a dialect that is defined as much by what it refuses to model as by what operations it contains.

