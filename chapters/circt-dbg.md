# CIRCT `dbg` Dialect

The CIRCT `dbg` dialect represents source-language debug information as MLIR operations. It lets CIRCT carry names, scopes, source aggregates, source-language type annotations, and enum interpretations alongside the hardware IR being optimized and lowered.

For a beginner, the useful mental model is this: `dbg` is a map from the user's original design concepts to the lower-level values that remain after compilation. A frontend may split a source struct into scalar wires, inline a module, fold a constant, or lower arrays into separate ports. The debug dialect records enough information for tools to reconstruct the source-facing view later.

The local checkout defines 6 `dbg` operations, 5 `dbg` types, no dialect-owned lowering pass, and several adjacent analyses, passes, and translations that create, inspect, strip, or emit debug information.

## When Debug Is Important

`dbg` is important when a hardware compiler flow needs debuggable output. It is used to preserve a human-readable relationship between source constructs and generated hardware. This matters for waveform debugging, debug-info emission, bounded model checking traces, simulator views, and source-level inspection of optimized hardware.

Use this dialect when you need to answer questions like:

- Which source variable does this lowered IR value correspond to?
- How can a source struct or array be reconstructed from scalar hardware values?
- Where did the contents of an inlined module go?
- Which constants and parameters should remain visible to the user?
- Is a value only present for debug information?
- Can a debug-info emitter produce HGLDD or a human-readable dump from this IR?

Debug information in CIRCT is best effort. It is meant to help humans, not to guarantee perfect preservation of every source-language detail through every optimization. It can also affect optimization because debug operations introduce uses of values that might otherwise be removed.

## Why It Is Needed

MLIR locations can track source positions, but source-level hardware debugging often needs more than file and line information. A hardware frontend may need to preserve the name of a parameter, the fields of a source bundle, an array shape, a source enum, or an inlined instance scope.

The `dbg` dialect uses operations rather than hidden metadata. This makes debug information first-class IR. Debug operations can refer directly to SSA values, survive ordinary IR rewriting, and make it clear which values are needed only for debug reconstruction.

The tradeoff is that debug operations may influence optimization because they add uses. CIRCT handles this with analysis and debug-aware consumers. For example, `DebugAnalysis` can identify operations and values that are debug-only, and debug-info emission can look through debug operations to find the real emitted values.

## Types

The exact `dbg` types in this checkout are:

```text
!dbg.scope
!dbg.struct
!dbg.array
!dbg.value
!dbg.enum
```

`!dbg.scope` is the result of `dbg.scope`. It represents an explicit debug hierarchy level, often an inlined source module or source-language scope.

`!dbg.struct` is the result of `dbg.struct`. It represents a debug aggregate with named fields. The type itself does not precisely encode the field layout; the structure is carried by the operation operands and field names.

`!dbg.array` is the result of `dbg.array`. It represents a debug aggregate with ordered elements. As with structs, the useful structure is in the operation.

`!dbg.value` is the result of `dbg.value`. It wraps another value with source-language metadata such as a type name and constructor parameters.

`!dbg.enum` is the result of `dbg.enum`. It wraps an integer tag with a source-language enum name and variant map.

The important design point is that the debug aggregate types are intentionally lightweight. Consumers reconstruct details by walking debug operations, not by relying on rich type parameters.

## Operation Inventory

The exact `dbg` operation names in this checkout are:

```text
dbg.scope
dbg.variable
dbg.value
dbg.enum
dbg.struct
dbg.array
```

### Scopes

`dbg.scope` defines an explicit debug scope.

```mlir
%scope = dbg.scope "bar", "Bar"
%nested = dbg.scope "inner", "InnerModule" scope %scope
```

It takes an instance name, a module name, and an optional parent scope. Operations such as `hw.module` already introduce implicit scopes. `dbg.scope` is for cases where the source hierarchy no longer exists directly in the IR, especially after inlining.

The `scope` operand used by debug operations must be produced locally by a `dbg.scope` operation. It cannot currently be a block argument. This keeps scope resolution simple for the debug-info analysis.

### Variables

`dbg.variable` maps a source-language name to an IR value.

```mlir
dbg.variable "Depth", %c12_i32 : i32
dbg.variable "req", %req : !dbg.struct
dbg.variable "x", %value scope %scope : i42
```

This is the central debug operation. It can represent ports, constants, parameters, local variables, nodes, aliases, registers, or aggregate source values. It has a string name, a value of any type, and an optional scope.

The value may be a normal hardware value, a `!dbg.struct`, a `!dbg.array`, a `!dbg.value`, or a `!dbg.enum`. Consumers use the variable as an anchor when reconstructing what a user should see.

### Metadata Values

`dbg.value` wraps a value with source-language metadata.

```mlir
%v = dbg.value %raw typeName "MyState" : !dbg.enum
dbg.variable "state", %v : !dbg.value
```

It can carry an optional `typeName` string and optional `params` array. The result is a `!dbg.value` wrapper that may be used by `dbg.variable`, `dbg.struct`, or `dbg.array`.

The verifier enforces that surviving users of `dbg.value` are debug-dialect aggregate or variable users. This keeps the wrapper from becoming part of normal hardware computation.

### Enums

`dbg.enum` interprets an integer value as a source-language enum tag.

```mlir
%e = dbg.enum %tag, "MyState", {Idle = 0 : i64, Run = 1 : i64}
    fqn "pkg.MyState" : i2
dbg.variable "state", %e : !dbg.enum
```

It takes the raw integer tag, a source enum type name, a dictionary mapping variant names to integer tags, and an optional fully qualified name. The `fqn` provides a stable key that consumers can use to deduplicate enum type definitions.

The enum is tag-only. The variants map has no place for per-variant payload types. The verifier requires the variants map to be nonempty, requires every value to be an integer attribute, requires signless integer tag values, and rejects duplicate enum tag values.

### Structs

`dbg.struct` builds a named-field debug aggregate.

```mlir
%req = dbg.struct {"data": %data, "valid": %valid, "ready": %ready}
    : i42, i1, i1
dbg.variable "req", %req : !dbg.struct
```

It has variadic field operands and a string array of field names. The verifier requires the number of names and fields to match. It is useful for source structs, bundles, interfaces, classes, unions, and similar named-field constructs.

The operation is pure. It exists to describe a source-level aggregate; it does not compute hardware logic.

### Arrays

`dbg.array` builds an ordered debug aggregate.

```mlir
%resps = dbg.array [%resp0, %resp1] : !dbg.struct
dbg.variable "resps", %resps : !dbg.array
```

It has variadic element operands and requires all element operands to have the same type. Element 0 is the first source element, and the last operand is the highest array index. It can represent source arrays, packed and unpacked arrays, queues, FIFOs, channels, vectors, or other ordered containers.

Empty arrays are allowed. Debug-info emitters can still represent them in a fallback form even though there are no element values to inspect.

## Transformations, Analyses, And Emission

The `dbg` dialect itself does not define a pass that lowers `dbg` operations to hardware. In normal flows, debug operations are side-channel information. They are produced by frontends or materialization passes, preserved through transformations as far as practical, analyzed, and consumed by debug-info targets.

The named passes and translations most relevant to this dialect are:

```text
firrtl-materialize-debug-info
materialize-debug-variables
strip-debuginfo-with-pred
dump-di
emit-hgldd
emit-split-hgldd
```

`firrtl-materialize-debug-info` creates debug operations for FIRRTL modules. It materializes `dbg.variable` anchors for FIRRTL ports, nodes, wires, registers, and reset registers. For FIRRTL aggregates, it decomposes bundles and vectors into `dbg.struct` and `dbg.array` values so the original source aggregate can be reconstructed after lowering.

`materialize-debug-variables` is used by CIRCT's BMC tooling. It creates missing `dbg.variable` anchors for hardware module inputs and sequence registers. It skips outputs, clocks, unnamed ports, and values that already have a debug variable.

`strip-debuginfo-with-pred` selectively strips MLIR location debug info from operations based on a predicate, with a testing option to drop file locations by suffix. This pass is about operation location metadata rather than deleting every `dbg` operation, but it belongs in the same debug-info maintenance space.

`DebugInfo` is an analysis, not a pass. It gathers module debug information by reading `dbg.variable` and `dbg.scope` operations. If a module has no explicit debug variables, it falls back to module ports and instances.

`DebugAnalysis` is also an analysis. It identifies debug-only operations, values, and operands. It starts from debug dialect operations and debug dialect typed values, then propagates through eligible HW and Comb operations when all uses or all operands are debug-only.

`dump-di` is a `circt-translate` translation that prints collected debug information in a human-readable form.

`emit-hgldd` emits HGLDD debug information. `emit-split-hgldd` emits HGLDD debug information as separate files. These translations register the `dbg`, `hw`, `comb`, `seq`, `sv`, `emit`, and `om` dialects needed for debug-info extraction.

## How To Read Debug IR

Start with `dbg.variable`. It tells you which user-facing name is being tracked and which IR value currently represents it.

```mlir
dbg.variable "outB", %sum : i32
```

This means the source-visible name `outB` should be reconstructed from `%sum`.

If the value is an aggregate, read the aggregate producer:

```mlir
%0 = dbg.struct {"result": %r, "done": %d} : i42, i1
dbg.variable "resp", %0 : !dbg.struct
```

This says the source variable `resp` has fields `result` and `done`, even if the hardware IR stores them as separate scalar signals.

For arrays, read operand order as source order:

```mlir
%0 = dbg.array [%resp0, %resp1] : !dbg.struct
dbg.variable "resps", %0 : !dbg.array
```

This reconstructs a two-element source array from two debug struct values.

For inlined hierarchy, follow the optional `scope` operand:

```mlir
%s = dbg.scope "bar", "Bar"
dbg.variable "x", %a scope %s : i42
```

This preserves that `x` came from the source instance `bar` of module `Bar`, even though the `hw.instance` may no longer exist.

For source type metadata, look for `dbg.value` and `dbg.enum`. A `dbg.enum` gives the tag map. A `dbg.value` can carry a source type name and parameters without overloading `dbg.variable` itself.

## What It Implies

Seeing `dbg.variable` means the compiler is deliberately keeping a user-facing source name alive.

Seeing `dbg.scope` means some hierarchy or source scope is being represented explicitly, often because inlining or lowering removed the original structural operation.

Seeing `dbg.struct` or `dbg.array` means a source aggregate has been broken into IR pieces but can still be reconstructed.

Seeing `dbg.value` means there is source-language type metadata attached to an underlying value.

Seeing `dbg.enum` means an integer should be interpreted as a named enum variant by debug consumers.

The main beginner lesson is that `dbg` operations are not hardware behavior. They are the compiler's breadcrumb trail from transformed hardware IR back to the concepts a designer wrote and wants to debug.
