# WasmSSA Dialect

## Beginner Summary

The `wasmssa` dialect represents WebAssembly modules in SSA form.

WebAssembly is normally a stack-machine format: instructions push and pop
values on an implicit operand stack. The `wasmssa` dialect makes those values
explicit as MLIR SSA values. That makes imported WebAssembly easier to inspect,
verify, and transform inside MLIR.

Think of `wasmssa` as:

```text
WebAssembly structure and instructions, but rewritten into explicit SSA values.
```

This dialect is useful when the input or target concept is WebAssembly itself,
not when you are writing normal high-level tensor, loop, or GPU compiler IR.

## Why This Dialect Exists

Raw WebAssembly is compact and well specified, but its stack-machine execution
model is awkward for many compiler analyses.

The `wasmssa` dialect exists to:

- Represent WebAssembly functions, globals, memories, tables, imports, and
  exports in MLIR.
- Replace implicit stack operands with explicit SSA operands and results.
- Model WebAssembly local variables as typed local references.
- Model WebAssembly structured control flow with blocks, loops, labels, and
  branch levels.
- Preserve WebAssembly instruction-level meaning while giving MLIR passes a
  structured IR to work with.
- Serve as the output of the local WebAssembly importer,
  `mlir-translate --import-wasm`.

The dialect is intentionally close to WebAssembly. Most operations map
one-to-one to WebAssembly instructions, except that values are SSA values
instead of stack entries.

## When It Matters

The `wasmssa` dialect matters when MLIR is consuming or analyzing WebAssembly.

Typical local flow in this checkout:

```text
WebAssembly binary
  -> mlir-translate --import-wasm
  -> wasmssa module/function/global/memory/table operations
  -> MLIR inspection, verification, and possible downstream analysis
```

This dialect is less like `linalg`, `tensor`, or `scf`, which are used to build
portable compiler pipelines. It is closer to an imported target-level IR for
WebAssembly.

## When To Use It

Use `wasmssa` when:

- You are importing WebAssembly into MLIR.
- You need to inspect WebAssembly instructions in MLIR form.
- You want explicit SSA operands for WebAssembly arithmetic, local, global, and
  control-flow operations.
- You need to represent WebAssembly module concepts such as imports, globals,
  memories, tables, and function references.
- You are building analyses or transformations that operate directly on
  WebAssembly semantics.

Do not use `wasmssa` as a general-purpose frontend dialect. If your source
program is not WebAssembly, higher-level dialects are usually a better starting
point.

## Core Concepts

### WebAssembly Stack Values Become SSA Values

In WebAssembly, an addition conceptually pops two values from the stack and
pushes one result. In `wasmssa`, those inputs and outputs are explicit:

```mlir
%a = wasmssa.const 8 : i32
%b = wasmssa.const 12 : i32
%sum = wasmssa.add %a %b : i32
```

That is the main idea behind the dialect name: WebAssembly in SSA form.

### Value Types

The dialect recognizes WebAssembly value types:

- Integer values: `i32`, `i64`.
- Floating-point values: `f32`, `f64`.
- Vector-like value: `i128` for the current WasmSSA vector type constraint.
- Reference values: `!wasmssa.funcref` and `!wasmssa.externref`.

The dialect also defines supporting types:

- `!wasmssa<local ref to T>` for local variable references.
- `!wasmssa<limit[min:max]>` for memory and table limits.
- `!wasmssa<tabletype REF LIMIT>` for tables.

### Functions And Locals

`wasmssa.func` represents a WebAssembly function.

WebAssembly function arguments and locals are both accessed with local
instructions. WasmSSA models this by wrapping function arguments as local
references, such as:

```text
!wasmssa<local ref to i32>
```

Local operations include `wasmssa.local`, `wasmssa.local_get`,
`wasmssa.local_set`, and `wasmssa.local_tee`.

### Imports, Globals, Memories, And Tables

The dialect models WebAssembly module-level objects:

- `wasmssa.import_func` and `wasmssa.call` for functions.
- `wasmssa.import_global`, `wasmssa.global`, and `wasmssa.global_get` for
  globals.
- `wasmssa.memory` and `wasmssa.import_mem` for linear memories.
- `wasmssa.table` and `wasmssa.import_table` for tables.

Exported functions, globals, memories, and tables use the `exported` marker.
Imported objects carry an import module name and import name.

### Label-Based Control Flow

WebAssembly structured control flow uses nesting levels and labels. The dialect
models this with:

- `wasmssa.block`
- `wasmssa.loop`
- `wasmssa.if`
- `wasmssa.block_return`
- `wasmssa.branch_if`

The important beginner idea is that branch levels are not arbitrary symbolic
labels in the source format. They refer to WebAssembly nesting levels. WasmSSA
uses interfaces to map those levels onto MLIR blocks and successors.

## Operations

### Module, Function, Import, And Return Operations

- `wasmssa.func` defines a WebAssembly function.
- `wasmssa.import_func` imports a function from another WebAssembly module.
- `wasmssa.call` calls a defined or imported function.
- `wasmssa.return` returns from a function or initializer region.
- `wasmssa.const` creates a numeric constant.

### Globals, Memories, Tables, And Locals

- `wasmssa.global` defines a WebAssembly global with an initializer region.
- `wasmssa.import_global` imports a global.
- `wasmssa.global_get` reads a global.
- `wasmssa.memory` defines a linear memory.
- `wasmssa.import_mem` imports a linear memory.
- `wasmssa.table` defines a table.
- `wasmssa.import_table` imports a table.
- `wasmssa.local` declares a local variable.
- `wasmssa.local_get` reads a local.
- `wasmssa.local_set` writes a local.
- `wasmssa.local_tee` writes a local and returns the written value.

### Control-Flow Operations

- `wasmssa.block` creates a nested block with an exit label.
- `wasmssa.loop` creates a nested loop with an entry label.
- `wasmssa.if` represents a WebAssembly conditional with optional else region.
- `wasmssa.block_return` exits the current block-like construct.
- `wasmssa.branch_if` conditionally branches to a nesting level.

### Numeric Operations

Integer and floating-point arithmetic:

- `wasmssa.add`
- `wasmssa.sub`
- `wasmssa.mul`
- `wasmssa.div`
- `wasmssa.div_si`
- `wasmssa.div_ui`
- `wasmssa.rem_si`
- `wasmssa.rem_ui`
- `wasmssa.neg`
- `wasmssa.abs`
- `wasmssa.sqrt`
- `wasmssa.min`
- `wasmssa.max`
- `wasmssa.copysign`

Bit operations:

- `wasmssa.and`
- `wasmssa.or`
- `wasmssa.xor`
- `wasmssa.shl`
- `wasmssa.shr_s`
- `wasmssa.shr_u`
- `wasmssa.rotl`
- `wasmssa.rotr`
- `wasmssa.clz`
- `wasmssa.ctz`
- `wasmssa.popcnt`

Comparisons:

- `wasmssa.eq`
- `wasmssa.ne`
- `wasmssa.eqz`
- `wasmssa.lt`
- `wasmssa.le`
- `wasmssa.gt`
- `wasmssa.ge`
- `wasmssa.lt_si`
- `wasmssa.lt_ui`
- `wasmssa.le_si`
- `wasmssa.le_ui`
- `wasmssa.gt_si`
- `wasmssa.gt_ui`
- `wasmssa.ge_si`
- `wasmssa.ge_ui`

Rounding and conversion:

- `wasmssa.ceil`
- `wasmssa.floor`
- `wasmssa.trunc`
- `wasmssa.nearest`
- `wasmssa.convert_s`
- `wasmssa.convert_u`
- `wasmssa.trunc_si`
- `wasmssa.trunc_ui`
- `wasmssa.extend_i32_s`
- `wasmssa.extend_i32_u`
- `wasmssa.extend`
- `wasmssa.demote`
- `wasmssa.promote`
- `wasmssa.wrap`
- `wasmssa.reinterpret`

### Complete Generated Operation Inventory

The generated operation list in this LLVM checkout is:

`wasmssa.abs`, `wasmssa.add`, `wasmssa.and`, `wasmssa.block`, `wasmssa.block_return`
`wasmssa.branch_if`, `wasmssa.call`, `wasmssa.ceil`, `wasmssa.clz`, `wasmssa.const`
`wasmssa.convert_s`, `wasmssa.convert_u`, `wasmssa.copysign`, `wasmssa.ctz`, `wasmssa.demote`
`wasmssa.div`, `wasmssa.div_si`, `wasmssa.div_ui`, `wasmssa.eq`, `wasmssa.eqz`, `wasmssa.extend`
`wasmssa.extend_i32_s`, `wasmssa.extend_i32_u`, `wasmssa.floor`, `wasmssa.func`, `wasmssa.ge`
`wasmssa.ge_si`, `wasmssa.ge_ui`, `wasmssa.global`, `wasmssa.global_get`, `wasmssa.gt`
`wasmssa.gt_si`, `wasmssa.gt_ui`, `wasmssa.if`, `wasmssa.import_func`, `wasmssa.import_global`
`wasmssa.import_mem`, `wasmssa.import_table`, `wasmssa.le`, `wasmssa.le_si`, `wasmssa.le_ui`
`wasmssa.local`, `wasmssa.local_get`, `wasmssa.local_set`, `wasmssa.local_tee`, `wasmssa.loop`
`wasmssa.lt`, `wasmssa.lt_si`, `wasmssa.lt_ui`, `wasmssa.max`, `wasmssa.memory`, `wasmssa.min`
`wasmssa.mul`, `wasmssa.ne`, `wasmssa.nearest`, `wasmssa.neg`, `wasmssa.or`, `wasmssa.popcnt`
`wasmssa.promote`, `wasmssa.reinterpret`, `wasmssa.rem_si`, `wasmssa.rem_ui`, `wasmssa.return`
`wasmssa.rotl`, `wasmssa.rotr`, `wasmssa.shl`, `wasmssa.shr_s`, `wasmssa.shr_u`, `wasmssa.sqrt`
`wasmssa.sub`, `wasmssa.table`, `wasmssa.trunc`, `wasmssa.trunc_si`, `wasmssa.trunc_ui`
`wasmssa.wrap`, `wasmssa.xor`

## Transformations

This LLVM checkout does not define WasmSSA-specific transform passes under
`mlir/include/mlir/Dialect/WasmSSA/Transforms`, and it does not register a
standard `convert-*-wasmssa` pass in `mlir/include/mlir/Conversion/Passes.td`.

The transformations that matter most for this dialect are structural:

- The WebAssembly importer converts stack-machine bytecode into explicit SSA
  operations.
- WasmSSA verifier logic checks details such as function argument local-ref
  types, local set/get type consistency, global initializer restrictions, and
  branch-level structure.
- Generic MLIR infrastructure can still parse, print, walk, and analyze the
  operations once they are in MLIR.

For this reason, a beginner should think of WasmSSA less as an optimization
dialect and more as an import/inspection dialect for WebAssembly semantics.

## Conversions And Lowering Paths

The main path into this dialect is:

- `mlir-translate --import-wasm`: translates a WebAssembly binary into MLIR
  using the `wasmssa` dialect.

The corresponding source code is in `mlir/lib/Target/Wasm/TranslateFromWasm.cpp`
and the registration name is defined in
`mlir/lib/Target/Wasm/TranslateRegistration.cpp`.

There is no general WasmSSA exporter or `convert-wasmssa-to-*` pass visible in
this checkout's standard pass registry. If a project wants to lower WasmSSA to
another dialect or emit WebAssembly from another dialect, that would require an
additional conversion pipeline beyond the dialect definitions shown here.

Typical local flow:

```text
wasm binary
  -> mlir-translate --import-wasm
  -> wasmssa.func / wasmssa.global / wasmssa.memory / wasmssa.table
  -> analysis or custom transformations
```

The implication is that WasmSSA is a faithful representation layer for imported
WebAssembly, not a general target-independent lowering layer.

## Example IR

### Function And Arithmetic

```mlir
module {
  wasmssa.func @add_i32() -> i32 {
    %a = wasmssa.const 8 : i32
    %b = wasmssa.const 12 : i32
    %sum = wasmssa.add %a %b : i32
    wasmssa.return %sum : i32
  }
}
```

The WebAssembly stack operands are explicit SSA operands to `wasmssa.add`.

### Locals

```mlir
module {
  wasmssa.func @local_example() -> i32 {
    %local = wasmssa.local of type i32
    %value = wasmssa.const 7 : i32
    wasmssa.local_set %local : ref to i32 to %value : i32
    %out = wasmssa.local_get %local : ref to i32
    wasmssa.return %out : i32
  }
}
```

Locals are represented by `!wasmssa<local ref to T>` values. `local_set`
updates the local, and `local_get` reads it.

### Imports, Memory, Table, And Global

```mlir
module {
  wasmssa.import_func "print" from "env" as @print {sym_visibility = "nested", type = (i32) -> ()}
  wasmssa.import_global "seed" from "env" as @seed : i32
  wasmssa.memory exported @mem !wasmssa<limit[1:10]>
  wasmssa.table @tab !wasmssa<tabletype !wasmssa.funcref [1:]>

  wasmssa.global @answer i32 : {
    %c42 = wasmssa.const 42 : i32
    wasmssa.return %c42 : i32
  }
}
```

This models the WebAssembly module-level surface: imported functions and
globals, a linear memory, a table, and a global initializer.

### If

```mlir
module {
  wasmssa.func @if_example(%arg0 : !wasmssa<local ref to i32>) -> f32 {
    %cond = wasmssa.local_get %arg0 : ref to i32
    wasmssa.if %cond : {
      %then_value = wasmssa.const 5.000000e-01 : f32
      wasmssa.block_return %then_value : f32
    } else {
      %else_value = wasmssa.const 2.500000e-01 : f32
      wasmssa.block_return %else_value : f32
    } >^after
  ^after(%result: f32):
    wasmssa.return %result : f32
  }
}
```

The `wasmssa.if` operation transfers the selected result to the successor
block through `wasmssa.block_return`.

### Loop

```mlir
module {
  wasmssa.func @loop_example() {
    wasmssa.loop : {
      wasmssa.block_return
    } > ^after
  ^after:
    wasmssa.return
  }
}
```

A `wasmssa.loop` creates a WebAssembly nesting level. Branching operations use
nesting levels to decide which label they target.

## Mental Model

The `wasmssa` dialect means:

```text
This is WebAssembly, but the operand stack has been made explicit as SSA values.
```

When reading WasmSSA, ask:

- Which WebAssembly module-level object is being represented?
- Which values came from locals, globals, constants, calls, or arithmetic?
- Which operations correspond directly to WebAssembly instructions?
- Which block, loop, or if operation defines the branch nesting level?
- Is this IR meant for import/analysis, or is there a custom downstream
  conversion pipeline?

## Gotchas

- WasmSSA is close to WebAssembly semantics. It is not a high-level portable
  optimization dialect.
- WebAssembly value types are limited. Do not expect arbitrary MLIR types.
- Function arguments are local references, not ordinary raw value parameters.
- `wasmssa.global` initializers must be WebAssembly constant expressions.
- Control flow uses WebAssembly nesting levels. This differs from ordinary MLIR
  branch labels.
- `wasmssa.branch_if` tests whether an `i32` condition is nonzero.
- The local checkout exposes an importer, `mlir-translate --import-wasm`, but no
  standard WasmSSA-to-LLVM, WasmSSA-to-CF, or WasmSSA exporter pass in the
  pass registry.

## Source Map

Primary source files in the LLVM tree:

- `mlir/include/mlir/Dialect/WasmSSA/IR/WasmSSABase.td`
- `mlir/include/mlir/Dialect/WasmSSA/IR/WasmSSAOps.td`
- `mlir/include/mlir/Dialect/WasmSSA/IR/WasmSSATypes.td`
- `mlir/include/mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.td`
- `mlir/lib/Dialect/WasmSSA/IR/`
- `mlir/include/mlir/Target/Wasm/WasmImporter.h`
- `mlir/lib/Target/Wasm/TranslateFromWasm.cpp`
- `mlir/lib/Target/Wasm/TranslateRegistration.cpp`

Useful tests:

- `mlir/test/Dialect/WasmSSA/`
- `mlir/test/Dialect/WasmSSA/custom_parser/`
- `mlir/test/Target/Wasm/`
