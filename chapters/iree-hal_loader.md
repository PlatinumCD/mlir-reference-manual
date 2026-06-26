# IREE `hal_loader` Dialect

The IREE `hal_loader` dialect is a low-level compiler dialect for dynamically loading executable binaries and dispatching work through IREE's inline HAL loader runtime module.

For a beginner, the useful mental model is this: the full IREE HAL is the general hardware abstraction layer, while `hal_loader` is a much smaller runtime bridge for inline executable loading. It keeps just enough IR to ask whether an executable format is supported, load a selected executable binary, look up an entry point, and dispatch work with push constants and buffer bindings.

The local checkout defines 6 `hal_loader` operations and 3 `hal_loader` passes.

## When HAL Loader Is Important

`hal_loader` is important in IREE configurations that use inline HAL behavior and want a lightweight executable-loading path. The dialect description in the local source says it is low-level, dynamically loads executables, dispatches work, and only operates synchronously, single-threaded, and on host-local buffers. For all other cases, IREE's full HAL should be used.

Use this dialect when you need to answer questions like:

- How does an inline HAL dispatch become a runtime call?
- Where is an embedded `hal.executable` loaded?
- How does the compiler choose a supported executable binary format?
- Where does an executable export name become a runtime function id?
- How are workgroup counts, push constants, and buffer binding ranges passed to the runtime?
- Has the compiler already lowered `hal_loader` operations to VM imports?

You usually do not write this dialect by hand. You inspect it when debugging IREE's inline HAL loader path, runtime import generation, executable materialization, or conversion to the VM dialect.

## Why It Is Needed

IREE has a rich HAL dialect for general device abstraction, queues, asynchronous execution, memory management, and device-specific behavior. The `hal_loader` dialect intentionally avoids that full model. It exists for a restricted environment: synchronous, single-threaded dispatch using host-local buffers.

That restricted path still needs a compiler representation for executable loading. The compiler may have serialized executable binaries in different formats. At runtime, the loader must ask which format is supported, load the first supported binary, look up a callable entry point, and dispatch work using dense constants and explicit buffer ranges.

The `hal_loader` dialect keeps those operations explicit until VM lowering. This makes the runtime boundary visible and lets IREE materialize executable globals and initializer code before replacing operations with calls to the `hal_loader` VM import module.

The implication is that `hal_loader` IR is not general GPU scheduling or full HAL command buffer code. It is a compact dynamic-loader layer.

## Types And Attributes

`hal_loader` does not define its own custom types in this checkout. It borrows types from IREE HAL and Util dialects.

The most important borrowed type is `!hal.executable`, represented in TableGen as `HAL_Executable`. It is the runtime executable handle produced by `hal_loader.executable.load`, cached by materialization, and consumed by function lookup and dispatch.

`hal_loader.executable.lookup.function` returns a `HAL_FunctionId`, which is an alias for `i64`. This is an opaque runtime function id. It is meaningful only for the executable it was looked up from.

Workgroup dimensions use `HAL_Dim`, an alias for index. Binding offsets and lengths use `HAL_DeviceSize`, also index-shaped in this IR. Executable binary data and dispatch binding buffers use `!util.buffer`.

String attributes are important. Executable formats are string attributes, and export references are symbol references. The compiler preserves symbol references long enough to resolve ordinals or names, even though later executable serialization may remove the original export declarations.

## Operation Inventory

This checkout defines 6 `hal_loader` operations. All of them are executable operations.

### Executable Query And Load

```text
hal_loader.executable.query_support
hal_loader.executable.load
```

`hal_loader.executable.query_support` asks the runtime whether a particular executable format is supported. It returns an `i1`. A true result means the loader supports that format, not that every binary using that format is guaranteed to load successfully.

`hal_loader.executable.load` creates, loads, and dynamically links an executable. It takes a format string, a `!util.buffer` containing executable data, and optional `i32` specialization constants. It returns `!hal.executable`.

These two operations usually appear in initializer code created by the materialization pass. The initializer tries available executable binaries, queries support for each format, loads a supported binary, and stores the result in a global.

### Executable Lookup

```text
hal_loader.executable.lookup
hal_loader.executable.export.ordinal
hal_loader.executable.lookup.function
```

`hal_loader.executable.lookup` is a pseudo-operation used during conversion as a placeholder for a globally cached executable. It references a `hal.executable` symbol and returns `!hal.executable`. After executable materialization, this lookup is replaced by a load from the generated executable global.

`hal_loader.executable.export.ordinal` resolves an executable export symbol reference to an index value once export ordinals have been assigned. The `iree-hal-loader-resolve-export-ordinals` pass replaces it with an `arith.constant` index.

`hal_loader.executable.lookup.function` maps an executable export name to a runtime function id. It takes an executable handle and an entry-point symbol reference, and returns an opaque `i64` function id. During VM lowering, this becomes a call to the `hal_loader.executable.lookup.function` runtime import.

### Executable Dispatch

```text
hal_loader.executable.dispatch
```

`hal_loader.executable.dispatch` performs the inline executable dispatch. It takes:

- an executable handle,
- a function id,
- three workgroup counts,
- optional `i32` push constants,
- binding buffers,
- binding offsets,
- binding lengths.

The operation verifies that the number of binding buffers, offsets, and lengths match. It also has a canonicalization pattern that folds `util.buffer.subspan` into the dispatch binding by replacing the subspan buffer with its source buffer and adding the subspan offset to the binding offset.

During VM lowering, this operation becomes a variadic call to the `hal_loader.executable.dispatch` runtime import. The variadic call packs constants and bindings in the import format expected by the runtime module.

### Complete Op List

For reference, the exact `hal_loader` operation names in this checkout are:

```text
hal_loader.executable.dispatch, hal_loader.executable.export.ordinal, hal_loader.executable.load, hal_loader.executable.lookup, hal_loader.executable.lookup.function, hal_loader.executable.query_support
```

## Transformations And Conversions

The local checkout defines 3 named `hal_loader` passes.

`iree-hal-loader-conversion` converts selected input dialect operations to the inline HAL and HAL Loader dialects. The conversion pass allows common MLIR dialects such as `func`, `scf`, `arith`, and `affine` to remain legal, converts stream operations mostly to `hal_inline`, and overrides executable-related stream dispatch handling with `hal_loader` patterns. A stream dispatch becomes an executable lookup, a function lookup, and a `hal_loader.executable.dispatch`. Full `hal` operations are converted into inline HAL where applicable.

`iree-hal-loader-materialize-executables` materializes executable globals and loader code. It walks `hal.executable` operations, creates a private `!hal.executable` global for each one, builds a Util initializer, and emits a chain of query/load blocks. Each executable binary gets a query block using `hal_loader.executable.query_support` and a load block using `hal_loader.executable.load`. If no binary format is supported, the initializer reports an unavailable status. Existing `hal_loader.executable.lookup` operations are then replaced with loads from the generated globals.

`iree-hal-loader-resolve-export-ordinals` resolves `hal_loader.executable.export.ordinal` operations. It finds the referenced `hal.executable.export`, reads its assigned ordinal attribute, replaces the pseudo-op with an `arith.constant` index, and erases the lookup operation.

The conversion from `hal_loader` to VM is registered as a dialect conversion interface on the dialect itself. That interface parses and inserts the embedded `hal_loader.imports.mlir` module, marks the `hal_loader` dialect illegal for VM conversion, and populates patterns that lower loader ops to VM calls.

At the VM boundary:

- `hal_loader.executable.query_support` calls `hal_loader.executable.query_support`.
- `hal_loader.executable.load` calls `hal_loader.executable.load` with a rodata format string, executable data buffer, and packed constants buffer.
- `hal_loader.executable.lookup.function` calls `hal_loader.executable.lookup.function` with the executable handle and a rodata function name.
- `hal_loader.executable.dispatch` calls `hal_loader.executable.dispatch` as a variadic VM call carrying executable, function id, workgroup counts, constants, and binding tuples.

The exact pass names covered in this chapter are:

```text
iree-hal-loader-conversion, iree-hal-loader-materialize-executables, iree-hal-loader-resolve-export-ordinals
```

## What It Implies

Seeing `hal_loader.executable.query_support` means executable format selection is still explicit in IR.

Seeing `hal_loader.executable.load` means executable binary data has not yet become a runtime import call. The compiler is still carrying a load operation that returns a `!hal.executable`.

Seeing `hal_loader.executable.lookup` means the executable has not yet been materialized into a global load. After `iree-hal-loader-materialize-executables`, these lookup operations should be gone.

Seeing `hal_loader.executable.lookup.function` means the runtime function id has not yet been requested from the VM import layer. The symbolic export reference is still visible.

Seeing `hal_loader.executable.export.ordinal` means export ordinals have not yet been resolved to constants.

Seeing `hal_loader.executable.dispatch` means dispatch is still represented as a HAL Loader operation. After VM conversion, it should become a variadic call to the runtime import.

## How To Read HAL Loader IR

Start by finding the executable handle. It may come from `hal_loader.executable.lookup` before materialization, or from `hal_loader.executable.load` inside an initializer.

Next, find `hal_loader.executable.lookup.function`. That operation tells you which export name will be used as the dispatch entry point and where the opaque function id comes from.

Then read `hal_loader.executable.dispatch`. The workgroup triple tells you how many workgroups will be launched. The constants list is the densely packed push-constant payload. The bindings list gives each `!util.buffer` plus its offset and length.

Finally, check the pipeline stage. If `hal_loader` ops are still present, VM lowering has not consumed them yet. If the embedded `hal_loader` VM import module and VM calls are present, the dialect has done its job.

## Minimal Example

This simplified fragment shows the dispatch shape:

```mlir
%exe = hal_loader.executable.lookup executable(@my_executable) : !hal.executable
%fn = hal_loader.executable.lookup.function
    target(%exe : !hal.executable)
    function(@my_executable::@variant::@main) : i64

hal_loader.executable.dispatch
  executable(%exe : !hal.executable)[%fn]
  workgroups([%x, %y, %z])
  constants([%c0, %c1])
  bindings([
    (%buffer : !util.buffer)[%offset, %length]
  ])
```

Read this as "get the loaded executable, look up the runtime function id for one export, then call that function over a workgroup grid with constants and buffer slices." Later VM conversion replaces this with calls into the runtime `hal_loader` module.
