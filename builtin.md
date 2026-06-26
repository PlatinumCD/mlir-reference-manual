# `builtin` Dialect

## Beginner Summary

The `builtin` dialect contains the core pieces that make MLIR IR possible:
the top-level `module`, the standard type system, standard attributes, and
source locations.

Most dialects describe a particular domain: arithmetic, functions, tensors,
GPU programs, loops, or target-specific instructions. The `builtin` dialect is
different. It is the common foundation that all of those dialects use.

You see `builtin` every time you read MLIR:

```mlir
module {
  func.func @id(%arg0: i32) -> i32 {
    func.return %arg0 : i32
  }
}
```

The `module` operation, the `i32` type, the function type syntax, the attribute
syntax, symbol references such as `@id`, tensor and memref types, vector types,
and source locations all come from the builtin layer.

For beginners, the key idea is this: `builtin` is not a computation dialect.
It is the IR vocabulary that lets other dialects talk to each other.

## Why This Dialect Exists

MLIR is built to host many dialects at once. Those dialects need shared
concepts:

- A container that can hold nested operations.
- A way to name symbols.
- A way to represent scalar, shaped, function, tuple, and opaque types.
- A way to attach compile-time metadata to operations.
- A way to attach source locations for diagnostics and debugging.
- A temporary cast operation used while converting between dialect type systems.

The `builtin` dialect provides those shared concepts. Without it, every dialect
would need to invent its own module operation, integer type, tensor type,
dictionary attribute, symbol reference, and location model.

The dialect exists so that MLIR has one common substrate while still allowing
domain dialects to define their own operations and abstractions.

## When It Matters

`builtin` matters in every MLIR program, but it is especially visible when you
are learning the IR syntax.

You care about it when:

- You are reading or writing the top-level `module`.
- You are trying to understand MLIR types such as `i32`, `index`,
  `tensor<4xf32>`, `memref<?xf32>`, `vector<8xi16>`, or `(i32) -> f32`.
- You are reading attributes in braces, such as `{sym_visibility = "private"}`.
- You are debugging source locations such as `loc("file.mlir":10:5)`.
- You are looking at a partially converted program containing
  `builtin.unrealized_conversion_cast`.
- You are writing passes that need to traverse nested modules, symbol tables,
  attributes, types, or locations.

It is less useful to think of `builtin` as something you "lower" like
`linalg` or `scf`. It is more like the grammar and shared data model of MLIR.

## When To Use It

Use builtin constructs whenever you need MLIR's common IR infrastructure.

Good uses:

- Put a program inside `module`.
- Use builtin scalar types such as `i1`, `i32`, `index`, `f32`, and `bf16`.
- Use shaped types such as `tensor`, `memref`, and `vector`.
- Attach attributes such as strings, integers, arrays, dictionaries, affine
  maps, symbol references, and dense constants.
- Preserve source information with locations.
- Use `builtin.unrealized_conversion_cast` only as a temporary bridge during
  dialect conversion.

Avoid using `builtin` as if it were a domain language. It does not describe
loops, arithmetic, memory allocation, function calls, GPU kernels, tensor
programs, or target instructions. Other dialects do that.

## Core Concepts

### Builtin Is Always Around

MLIR parsers and printers treat the builtin dialect specially. You often see
the operations printed without a `builtin.` prefix:

```text
module
unrealized_conversion_cast
```

Their registered operation names are:

```text
builtin.module
builtin.unrealized_conversion_cast
```

Similarly, builtin types and attributes usually have compact syntax:

```text
i32
index
tensor<4xf32>
memref<?xf32>
dense<[1, 2, 3]> : tensor<3xi32>
"hello"
@symbol_name
```

### Module Is The Default Top-Level Container

`builtin.module` is a symbol table and an isolated container. It usually appears
as `module`.

Important traits:

| Trait | Meaning |
| --- | --- |
| `SymbolTable` | Nested symbol operations can be looked up by name. |
| `Symbol` | A module may itself have a symbol name. |
| `IsolatedFromAbove` | Operations inside cannot implicitly capture SSA values defined outside the module. |
| `AffineScope` | Affine constructs may use the module as an affine scope. |

The module region is a graph region with one block and no terminator.

### Builtin Types Are Shared By All Dialects

When `arith.addi` takes `i32`, `tensor.extract` reads from
`tensor<4xf32>`, or `memref.load` reads from `memref<?xi8>`, those types are
builtin types.

This is why builtin is a foundation dialect: many operation dialects reuse the
same type language.

### Attributes Are Compile-Time Values

Attributes are immutable compile-time values attached to operations, types, or
other attributes. They are not SSA values and they do not represent runtime
computation.

Examples:

```text
42 : i32
"private"
[1, 2, 3]
{sym_name = "main", sym_visibility = "public"}
dense<[1, 2, 3]> : tensor<3xi32>
@callee
```

### Locations Are For Diagnostics

Locations say where an operation came from. They can be file locations, names,
call sites, fused locations, opaque frontend locations, or unknown locations.

Locations are not program semantics. A pass may strip them with
`strip-debuginfo`, and the executable meaning of the program should not change.

### Unrealized Casts Are Temporary Glue

`builtin.unrealized_conversion_cast` is a temporary operation used by dialect
conversion. It lets one part of a conversion proceed even when not all producer
and consumer types have been converted yet.

It should not survive as a meaningful operation in final IR. The
`reconcile-unrealized-casts` pass cleans up cast chains that cancel out.

## Operations

The current local LLVM checkout defines two builtin operations.

| Operation | Printed form | What it does |
| --- | --- | --- |
| `builtin.module` | `module` | Top-level container operation, optional symbol, symbol table, isolated region. |
| `builtin.unrealized_conversion_cast` | `unrealized_conversion_cast` | Temporary variadic cast used while intermixing type systems during conversion. |

### `builtin.module`

`builtin.module` holds the main IR body. It contains one region with one block.
The region has no terminator. Modules may be named:

```mlir
module @named_module {
  func.func @id(%arg0: i32) -> i32 {
    func.return %arg0 : i32
  }
}
```

A named module is also a symbol. Many compiler pipelines run module-level
passes on this operation.

### `builtin.unrealized_conversion_cast`

This operation can convert from zero or more input values to zero or more
result values. It has no intended runtime behavior. It represents "the
conversion infrastructure has not resolved this boundary yet."

```mlir
module {
  func.func @cast_example(%arg0: i32) -> index {
    %0 = builtin.unrealized_conversion_cast %arg0 : i32 to index
    func.return %0 : index
  }
}
```

If a final lowering pipeline still contains this operation, that usually means
some conversion pattern or cleanup pass is missing.

## Types

The builtin dialect defines the common MLIR type language. The current local
checkout generated these builtin type definitions.

### Integer And Index Types

| Type | Syntax | Meaning |
| --- | --- | --- |
| `IntegerType` | `i32`, `si32`, `ui32`, `i1`, `i64` | Integer with arbitrary precision up to MLIR's fixed limit. Signedness may be signless, signed, or unsigned. |
| `IndexType` | `index` | Machine-word-like integer used for dimensions, loop indices, and memory indexing. Its concrete width is target-dependent. |

The most common beginner mistake is treating `index` as just another spelling
for `i64`. It is not. It is an abstract index type that later lowerings map to
a concrete integer width.

### Floating-Point Types

| Type | Syntax | Meaning |
| --- | --- | --- |
| `BFloat16Type` | `bf16` | 16-bit bfloat type. |
| `Float16Type` | `f16` | IEEE-style 16-bit float. |
| `FloatTF32Type` | `tf32` | TensorFloat-32 style float. |
| `Float32Type` | `f32` | 32-bit float. |
| `Float64Type` | `f64` | 64-bit float. |
| `Float80Type` | `f80` | 80-bit float. |
| `Float128Type` | `f128` | 128-bit float. |

Builtin also includes small floating-point formats used by accelerator and ML
pipelines:

| Type | Syntax |
| --- | --- |
| `Float4E2M1FNType` | `f4E2M1FN` |
| `Float6E2M3FNType` | `f6E2M3FN` |
| `Float6E3M2FNType` | `f6E3M2FN` |
| `Float8E3M4Type` | `f8E3M4` |
| `Float8E4M3Type` | `f8E4M3` |
| `Float8E4M3FNType` | `f8E4M3FN` |
| `Float8E4M3FNUZType` | `f8E4M3FNUZ` |
| `Float8E4M3B11FNUZType` | `f8E4M3B11FNUZ` |
| `Float8E5M2Type` | `f8E5M2` |
| `Float8E5M2FNUZType` | `f8E5M2FNUZ` |
| `Float8E8M0FNUType` | `f8E8M0FNU` |

These are builtin because many dialects need to agree on the same scalar
element types even when only some hardware targets can lower them directly.

### Composite And Function-Like Types

| Type | Syntax | Meaning |
| --- | --- | --- |
| `ComplexType` | `complex<f32>` | Complex number with integer or floating element type. |
| `TupleType` | `tuple<i32, f32>` | Fixed-sized collection of other types. |
| `FunctionType` | `(i32, f32) -> i1` | Function signature type. |
| `GraphType` | implementation-facing syntax | Function-like graph type used by MLIR internals. |
| `NoneType` | `none` | Unit-like absence type. |
| `TokenType` | `token` | Token for dependency-style ordering. |
| `OpaqueType` | `!dialect.mnemonic<...>` style opaque form | Type from an unregistered or opaque dialect. |

Function types describe signatures. Function operations themselves come from
the `func` dialect, not from `builtin`.

### Shaped Types

| Type | Syntax | Meaning |
| --- | --- | --- |
| `RankedTensorType` | `tensor<4x?xf32>` | Tensor with known rank and possibly dynamic dimensions. |
| `UnrankedTensorType` | `tensor<*xf32>` | Tensor with unknown rank. |
| `MemRefType` | `memref<4x?xf32>` | Shaped reference to memory, with optional layout and memory space. |
| `UnrankedMemRefType` | `memref<*xf32>` | Memref with unknown rank. |
| `VectorType` | `vector<8xf32>` | Fixed or scalable SIMD-style vector type. |

Shaped types use `?` for dynamic dimensions:

```text
tensor<4x?xf32>
memref<?x?xi32>
```

Memrefs may also carry layout and memory-space information:

```text
memref<4x?xf32, strided<[?, 1], offset: ?>>
```

## Attributes

Attributes are part of the builtin dialect even when they appear inside
operations from other dialects.

The current local checkout defines these ordinary builtin attributes.

| Attribute | Common syntax | What it represents |
| --- | --- | --- |
| `AffineMapAttr` | `affine_map<(d0) -> (d0 + 1)>` | An affine map value. |
| `ArrayAttr` | `[1, "x", i32]` | Ordered list of attributes. |
| `DenseArrayAttr` | `array<i32: 1, 2, 3>` | Flat dense array of primitive integer or float data. |
| `DenseTypedElementsAttr` | `dense<[1, 2]> : tensor<2xi32>` | Dense shaped elements attribute. |
| `DenseStringElementsAttr` | dense string shaped data | Dense shaped string elements. |
| `DenseResourceElementsAttr` | resource-backed dense data | Dense shaped elements stored in an external resource blob. |
| `DictionaryAttr` | `{name = "x", value = 1 : i32}` | Sorted dictionary of named attributes. |
| `FloatAttr` | `1.000000e+00 : f32` | Floating-point attribute with type. |
| `IntegerAttr` | `42 : i32` | Integer attribute with type. |
| `IntegerSetAttr` | `affine_set<(d0) : (d0 >= 0)>` | Integer set used by affine-style constraints. |
| `OpaqueAttr` | `#dialect<...>` | Opaque attribute from another dialect. |
| `SparseElementsAttr` | `sparse<...> : tensor<...>` | Sparse shaped elements attribute. |
| `StringAttr` | `"hello"` | String attribute. |
| `SymbolRefAttr` | `@foo` or `@foo::@bar` | Reference to a symbol operation. |
| `TypeAttr` | `i32`, `tensor<4xf32>` as an attribute value | Attribute wrapping a type. |
| `UnitAttr` | keyword-only unit attribute | Presence marker with no payload. |
| `StridedLayoutAttr` | `strided<[?, 1], offset: ?>` | Strided layout for shaped types such as memrefs. |

### Dense Elements

Dense elements are common in constants:

```mlir
module {
  func.func @dense_values() -> tensor<4xi32> {
    %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    func.return %0 : tensor<4xi32>
  }
}
```

The attribute is compile-time data. The `arith.constant` operation gives that
attribute an SSA value.

### Dictionaries

Operation attributes are stored as dictionary attributes. In assembly, they
usually appear in braces:

```text
{sym_name = "main", sym_visibility = "private"}
```

Many operations also have custom syntax that hides part of this dictionary.

### Symbol References

Symbol references point to symbol operations by name. A function call uses a
symbol reference to name the callee:

```mlir
module {
  func.func private @callee(%arg0: i32) -> i32

  func.func @caller(%arg0: i32) -> i32 {
    %0 = func.call @callee(%arg0) : (i32) -> i32
    func.return %0 : i32
  }
}
```

The symbol reference `@callee` is an attribute, not an SSA operand.

## Location Attributes

Location attributes are builtin attributes used specially as operation
locations.

| Location | Syntax idea | Meaning |
| --- | --- | --- |
| `FileLineColRange` | `"file.mlir":10:5` | File, line, column, and optional range. |
| `NameLoc` | `"name"` | Named location, optionally wrapping a child location. |
| `CallSiteLoc` | `callsite("callee" at "caller.mlir":3:7)` | Call stack style location. |
| `FusedLoc` | `fused["a", "b"]` | Multiple locations merged into one. |
| `OpaqueLoc` | frontend-specific opaque location | Location owned by another frontend or tool. |
| `UnknownLoc` | `unknown` | No known source location. |

Example:

```mlir
module {
  func.func @locations() {
    %c0 = arith.constant 0 : index loc("frontend.name")
    %c1 = arith.constant 1 : index loc("book.mlir":12:3)
    %c2 = arith.addi %c0, %c1 : index loc(fused["folded", "book.mlir":13:5])
    func.return loc(unknown)
  }
}
```

Locations explain diagnostics and debugging output. They should not be used as
semantic program data.

## Transformations

The builtin dialect does not have a large set of domain-specific optimization
passes. Its operations and attributes are affected by MLIR's general
infrastructure passes.

Important cross-cutting passes:

| Pass | Why it matters for builtin |
| --- | --- |
| `canonicalize` | Applies canonicalization patterns across loaded dialects and simplifies regions. It may fold `builtin.unrealized_conversion_cast` when patterns are available. |
| `cse` | Eliminates duplicate side-effect-free operations using MLIR's side-effect interfaces. |
| `trivial-dce` | Removes trivially dead operations and blocks. |
| `sccp` | Performs sparse conditional constant propagation over MLIR regions. |
| `inline` | Uses symbol tables, call interfaces, and regions to inline calls. |
| `strip-debuginfo` | Replaces operation locations with `unknown`. |
| `symbol-dce` | Deletes unreachable private symbols from symbol tables. |
| `symbol-privatize` | Marks top-level symbols private except excluded symbols. |
| `print-ir` | Prints IR, exposing builtin types, attributes, locations, and module structure. |
| `print-op-stats` | Counts operations, often useful for seeing how much builtin and non-builtin IR remains. |

These are not "builtin-only" passes. They are shared MLIR passes that operate
over the IR structure provided by builtin concepts.

## Conversions And Lowering Paths

`builtin` is not usually lowered wholesale. Instead, builtin pieces participate
in many conversions:

- Builtin types such as `index`, `memref`, `tensor`, and `vector` are converted
  by dialect-specific or target-specific conversion passes.
- Builtin attributes are interpreted by the operations that own them.
- Builtin locations may be preserved, transformed, or stripped.
- `builtin.unrealized_conversion_cast` is introduced by partial conversions and
  cleaned up later.

The most directly relevant conversion cleanup pass is
`reconcile-unrealized-casts`.

### `reconcile-unrealized-casts`

This pass simplifies and removes chains of `builtin.unrealized_conversion_cast`
operations that convert a value back to a compatible original type.

Example pattern:

```text
!type.A -> !type.B -> !type.C -> !type.A
```

If the conversion chain returns to the same type, the casts can often be
removed and uses can be rewired to the original value.

A healthy final lowering pipeline should not leave meaningful unrealized casts
behind.

## Example IR

### Minimal Module

```mlir
module {
  func.func @id(%arg0: i32) -> i32 {
    func.return %arg0 : i32
  }
}
```

This example contains one `builtin.module`, one `func.func`, and several
builtin types.

### Named Module And Symbol Visibility

```mlir
module @book_module {
  func.func private @helper(%arg0: i32) -> i32 {
    func.return %arg0 : i32
  }

  func.func @entry(%arg0: i32) -> i32 {
    %0 = func.call @helper(%arg0) : (i32) -> i32
    func.return %0 : i32
  }
}
```

The module and functions are symbols. The call names `@helper` with a symbol
reference attribute.

### Builtin Types In Ordinary Operations

```mlir
module {
  func.func @types(
      %arg0: index,
      %arg1: tensor<4x?xf32>,
      %arg2: memref<?xi8>,
      %arg3: vector<8xi16>) -> tuple<index, i32> {
    %c0 = arith.constant 0 : index
    %i = arith.index_cast %c0 : index to i32
    %t = builtin.unrealized_conversion_cast %c0, %i
        : index, i32 to tuple<index, i32>
    func.return %t : tuple<index, i32>
  }
}
```

This is not a recommended final program because it returns a tuple through an
unrealized cast. It is a compact way to show that builtin types are shared by
operations from other dialects.

### Dense Attribute As A Constant

```mlir
module {
  func.func @dense_tensor() -> tensor<4xi32> {
    %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    func.return %0 : tensor<4xi32>
  }
}
```

The dense elements attribute is compile-time data. The operation turns it into
an SSA result.

### Locations

```mlir
module {
  func.func @with_locations() {
    %c0 = arith.constant 0 : index loc("frontend.zero")
    %c1 = arith.constant 1 : index loc("source.mlir":4:9)
    %c2 = arith.addi %c0, %c1 : index loc(fused["sum", "source.mlir":5:11])
    func.return loc(unknown)
  }
}
```

The location syntax is builtin. The arithmetic operation is not.

## How To Read Builtin IR

When reading MLIR, separate builtin structure from domain behavior.

| If you see | Read it as |
| --- | --- |
| `module` | A top-level isolated container and symbol table. |
| `i32`, `index`, `f32` | Shared scalar types. |
| `tensor<...>`, `memref<...>`, `vector<...>` | Shared shaped types used by many dialects. |
| `{...}` | Dictionary of operation attributes. |
| `@name` | Symbol reference or symbol definition, depending on context. |
| `loc(...)` | Source/debug location metadata. |
| `builtin.unrealized_conversion_cast` | Temporary conversion glue, not final semantics. |

This distinction helps when learning new dialects. In
`arith.addi %a, %b : i32`, the operation is from `arith`, but the type `i32` is
from `builtin`.

## Gotchas

- `builtin` is not optional. Even when no operation is printed with a
  `builtin.` prefix, builtin types and attributes are everywhere.
- `index` is target-dependent. Do not assume it is always `i64`.
- Attributes are not SSA values. An operation may use an attribute to produce an
  SSA value, but the attribute itself is compile-time metadata.
- Symbol references are attributes, not SSA operands. Renaming symbols requires
  updating symbol references.
- `module` is isolated from above. Nested operations cannot implicitly capture
  values outside the module.
- `builtin.unrealized_conversion_cast` is temporary. If it remains after final
  lowering, inspect conversion completeness and run `reconcile-unrealized-casts`
  when appropriate.
- Locations are diagnostic metadata. They should not affect program meaning.
- Some builtin syntax is compact or implicit. For example, operation attribute
  dictionaries often print as braces, and builtin operation names may be printed
  without the `builtin.` prefix.

## What It Implies In A Compiler Pipeline

The builtin dialect shapes every pipeline:

- Pass managers commonly run on `builtin.module`.
- Symbol passes depend on module symbol tables and symbol reference attributes.
- Conversion passes use builtin type conversion rules and may introduce
  `builtin.unrealized_conversion_cast`.
- Lowering to LLVM, SPIR-V, EmitC, or target dialects must decide how builtin
  types such as `index`, `memref`, `tensor`, and `vector` map to target forms.
- Debug and diagnostic behavior depends on builtin location attributes.
- Constant data is frequently carried in builtin dense, sparse, string, integer,
  float, array, and dictionary attributes.

For a beginner, the important mental model is that `builtin` is the shared
grammar of MLIR. Other dialects give the program meaning; builtin gives them a
common place to live and a common set of types, attributes, symbols, and
locations.

## Source Map

Primary source files in the local LLVM checkout:

| File | What to look for |
| --- | --- |
| `mlir/include/mlir/IR/BuiltinDialect.td` | Dialect definition and purpose. |
| `mlir/include/mlir/IR/BuiltinOps.td` | `builtin.module` and `builtin.unrealized_conversion_cast`. |
| `mlir/include/mlir/IR/BuiltinTypes.td` | Builtin scalar, shaped, function-like, tuple, token, and opaque types. |
| `mlir/include/mlir/IR/BuiltinAttributes.td` | Builtin ordinary attributes. |
| `mlir/include/mlir/IR/BuiltinLocationAttributes.td` | Builtin location attributes. |
| `mlir/lib/IR/BuiltinDialect.cpp` | Dialect registration and parsing/printer hooks. |
| `mlir/lib/IR/BuiltinTypes.cpp` | Type parsing, printing, verification, and helpers. |
| `mlir/lib/IR/BuiltinAttributes.cpp` | Attribute parsing, printing, verification, and helpers. |
| `mlir/include/mlir/Transforms/Passes.td` | Cross-cutting passes such as `canonicalize`, `cse`, `strip-debuginfo`, `symbol-dce`, and `symbol-privatize`. |
| `mlir/include/mlir/Conversion/Passes.td` | `reconcile-unrealized-casts`. |
| `mlir/test/IR/` | Parser, printer, verifier, attribute, type, location, and symbol tests. |
| `mlir/test/Transforms/` | General transforms that act on builtin module structure. |
