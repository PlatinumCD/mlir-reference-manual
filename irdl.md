# `irdl` Dialect

## Beginner Summary

`irdl` stands for Intermediate Representation Definition Language. It is a
dialect for defining other MLIR dialects using MLIR itself.

Most MLIR dialects are defined ahead of time with TableGen and C++. IRDL gives
MLIR an IR-level way to describe dialects, operations, attributes, types, and
their verifier constraints. That means a dialect definition can be parsed,
inspected, transformed, loaded dynamically, or translated into C++.

IRDL is not a computation dialect. It does not represent the user's program.
It represents the definition of an IR.

## Why This Dialect Exists

MLIR is extensible, but normal dialect development requires C++ and TableGen.
That is powerful, but it is not always portable or dynamic.

IRDL exists to make dialect definitions themselves into MLIR data. This helps
with:

- portable dialect descriptions;
- runtime loading of dialects;
- dialect analysis and simplification;
- generating C++ dialect definitions from IR;
- testing dialect ideas without writing full C++ infrastructure;
- tools that reason about verifier constraints, such as fuzzers or SMT-based
  analyses.

The key idea is that a dialect definition can be represented as structured IR
instead of only as compiler source code.

## When It Matters

IRDL matters when you want to define or inspect the structure of dialects.

You are likely to see it when:

- loading a dynamic dialect with `mlir-opt --irdl-file=...`;
- writing tests for dynamically defined operations or types;
- translating IRDL definitions to C++ with `mlir-translate --irdl-to-cpp`;
- generating IRDL from TableGen with `tblgen-to-irdl`;
- using transform dialect support to match payload operations against an IRDL
  definition;
- experimenting with dialect definitions before committing to C++/TableGen.

You are less likely to see IRDL in a normal lowering pipeline from tensor code
to machine code.

## When To Use It

Use IRDL when the thing you need to represent is a dialect definition.

Good uses:

- define a small dynamic dialect for tests or experiments;
- ship a self-contained dialect description;
- generate dialect metadata for tools;
- check whether an operation matches a declarative definition;
- translate supported IRDL dialect definitions into C++ declarations and
  definitions.

Avoid using IRDL to model normal program semantics. If you are trying to
represent arithmetic, tensors, control flow, memory, GPU code, or LLVM-like
code, use the relevant payload dialect instead.

## Core Concepts

### IRDL Definitions Are MLIR Operations

An IRDL dialect definition is written with ordinary MLIR operations:

```mlir
irdl.dialect @cmath {
  irdl.type @complex {
    %f32 = irdl.is f32
    %f64 = irdl.is f64
    %elem = irdl.any_of(%f32, %f64)
    irdl.parameters(elem: %elem)
  }

  irdl.operation @mul {
    %f32 = irdl.is f32
    %f64 = irdl.is f64
    %elem = irdl.any_of(%f32, %f64)
    %complex = irdl.parametric @cmath::@complex<%elem>
    irdl.operands(lhs: %complex, rhs: %complex)
    irdl.results(res: %complex)
  }
}
```

This defines a dialect named `cmath`, a type named `complex`, and an operation
named `mul`.

### Constraint Variables

IRDL constraints produce SSA values. Reusing the same constraint value means
the matched attribute or type must be the same within that verification.

Example:

```mlir
%t = irdl.any
irdl.operands(lhs: %t, rhs: %t)
irdl.results(res: %t)
```

This does not merely say that all three positions can be anything. It says that
they must agree on the same thing.

### Types Are Wrapped As Attributes

IRDL variables are handles to `mlir::Attribute`. MLIR types are represented
through `TypeAttr`. This simplifies IRDL's internal model: constraints operate
on attributes, and types appear as attributes that wrap a type.

### Runtime Loading

IRDL definitions can be loaded into an MLIR context. `mlir-opt` exposes this
with:

```text
mlir-opt --irdl-file=some-dialect.irdl.mlir input.mlir
```

The IRDL file is parsed, and dialects defined by `irdl.dialect` are registered
dynamically.

### IRDL-To-C++

IRDL can also be translated to C++:

```text
mlir-translate --irdl-to-cpp dialect.irdl.mlir
```

This is not a replacement for all hand-written dialect code, but it is an
important path from declarative IRDL definitions to compiled dialect support.

## Operations

IRDL defines 17 core operations.

### Definition Operations

| Operation | Purpose |
|---|---|
| `irdl.dialect` | Defines a new dialect and contains type, attribute, and operation definitions. |
| `irdl.type` | Defines a new type in the enclosing `irdl.dialect`. |
| `irdl.attribute` | Defines a new attribute in the enclosing `irdl.dialect`. |
| `irdl.operation` | Defines a new operation in the enclosing `irdl.dialect`. |

### Type And Attribute Shape Operations

| Operation | Purpose |
|---|---|
| `irdl.parameters` | Defines parameters of an `irdl.type` or `irdl.attribute`. |

### Operation Shape Operations

| Operation | Purpose |
|---|---|
| `irdl.operands` | Defines the operands of an `irdl.operation`. |
| `irdl.results` | Defines the results of an `irdl.operation`. |
| `irdl.attributes` | Defines required attributes of an `irdl.operation`. |
| `irdl.region` | Creates a region constraint, including optional entry-block argument and block-count constraints. |
| `irdl.regions` | Attaches named region constraints to an `irdl.operation`. |

### Constraint Operations

| Operation | Purpose |
|---|---|
| `irdl.is` | Matches exactly one concrete attribute or type instance. |
| `irdl.base` | Matches a type or attribute base, either by symbol reference or name. |
| `irdl.parametric` | Matches a type or attribute base plus parameter constraints. |
| `irdl.any` | Matches any type or attribute. |
| `irdl.any_of` | Matches if any operand constraint matches. |
| `irdl.all_of` | Matches only if all operand constraints match. |
| `irdl.c_pred` | Uses a C++ predicate as a constraint. |

### Related Transform Operation

The Transform dialect has an IRDL extension op:

| Operation | Dialect | Purpose |
|---|---|---|
| `transform.irdl.collect_matching` | `transform` | Finds payload operations that match an embedded IRDL operation definition, without registering that dialect. |

`transform.irdl.collect_matching` is not a payload `irdl` operation. It is a
transform op that uses IRDL definitions as matching criteria.

## Types And Attributes

IRDL defines two types:

| Type | Purpose |
|---|---|
| `!irdl.attribute` | A handle to an MLIR attribute; also used for types wrapped as `TypeAttr`. |
| `!irdl.region` | A handle to a region constraint produced by `irdl.region`. |

IRDL also defines variadicity attributes:

| Attribute | Purpose |
|---|---|
| `#irdl.variadicity` | Represents whether an operand/result definition is `single`, `optional`, or `variadic`. |
| `#irdl.variadicity_array` | Array form used by operand/result definitions. |

Most users see variadicity through assembly keywords:

```mlir
irdl.operands(lhs: single %t, extras: variadic %t, maybe: optional %t)
```

## Transformations

IRDL has fewer optimization-style transformations than computation dialects.
Its important "transformations" are definition-oriented.

### Dynamic Dialect Loading

`irdl::loadDialects` loads `irdl.dialect` definitions from a module into an
MLIR context. This creates dynamic dialect/type/attribute/operation
definitions that can verify later IR.

In `mlir-opt`, this is exposed through:

```text
--irdl-file=<filename>
```

### IRDL Matching In Transform Dialect

`transform.irdl.collect_matching` embeds an IRDL operation definition inside a
transform script and returns handles to payload operations that satisfy it.

Example:

```mlir
%matched = transform.irdl.collect_matching in %root
  : (!transform.any_op) -> !transform.any_op {
  irdl.dialect @test {
    irdl.operation @whatever {
      %i32 = irdl.is i32
      %i64 = irdl.is i64
      %t = irdl.any_of(%i32, %i64)
      irdl.results(res: %t)
    }
  }
}
```

This can match `test.whatever` operations with either `i32` or `i64` results,
without registering the `test` dialect globally.

### TableGen To IRDL

The `tblgen-to-irdl` tool can generate IRDL definitions from TableGen dialect
definitions in supported cases. This is useful when moving between existing
ODS/TableGen dialect definitions and IRDL's self-describing form.

## Conversions And Lowering Paths

IRDL is not normally lowered to machine code. Its conversions are about moving
between dialect-definition formats.

Common paths:

```text
IRDL file
  -> mlir-opt --irdl-file
  -> dynamic dialect registration
  -> verification of payload IR
```

```text
IRDL dialect definition
  -> mlir-translate --irdl-to-cpp
  -> C++ dialect/type/op definitions
```

```text
TableGen dialect definition
  -> tblgen-to-irdl
  -> IRDL dialect definition
```

IRDL may also be used as input to analysis tools because verifier constraints
are explicit IR.

## Example IR

### A Small Type And Operation Definition

```mlir
irdl.dialect @cmath {
  irdl.type @complex {
    %f32 = irdl.is f32
    %f64 = irdl.is f64
    %elem = irdl.any_of(%f32, %f64)
    irdl.parameters(elem: %elem)
  }

  irdl.operation @norm {
    %any = irdl.any
    %complex = irdl.parametric @cmath::@complex<%any>
    irdl.operands(input: %complex)
    irdl.results(output: %any)
  }
}
```

This defines:

- a `cmath.complex` type parameterized by either `f32` or `f64`;
- a `cmath.norm` operation that consumes a complex value and returns its
  element type.

### Variadic Operands

```mlir
irdl.dialect @example {
  irdl.operation @join {
    %t = irdl.any
    irdl.operands(first: %t, rest: variadic %t)
    irdl.results(out: %t)
  }
}
```

This defines an operation with one required operand and then any number of
additional operands matching the same constraint variable.

### Region Constraints

```mlir
irdl.dialect @example {
  irdl.operation @with_region {
    %i32 = irdl.is i32
    %body = irdl.region(%i32) with size 1
    irdl.regions(body: %body)
  }
}
```

This defines an operation with one region, whose entry block takes one `i32`
argument and whose region has one block.

## Mental Model

Think of IRDL as "dialect definitions as IR."

Normal dialects describe programs. IRDL describes the rules for dialects. Its
operations do not perform computation; they define names, parameters, operands,
results, attributes, regions, and constraints.

## Gotchas

- IRDL is meant to be generated and analyzed; it is not optimized for pleasant
  hand-writing.
- Reusing the same constraint SSA value enforces equality of the matched
  attribute/type.
- IRDL variables are attributes; MLIR types are represented through `TypeAttr`.
- `irdl.c_pred` uses C++ predicates, so definitions using it are not fully
  runtime-portable.
- Dynamic dialects are useful, but they do not automatically provide all the
  custom behavior a hand-written dialect might have.
- Multiple optional or variadic operand/result groups may require segment-size
  attributes such as `operandSegmentSizes` or `resultSegmentSizes` on payload
  operations.
- `transform.irdl.collect_matching` currently supports a restricted embedded
  IRDL shape: one dialect with exactly one operation and no IRDL types or
  attributes.

## Source Map

Primary dialect files:

- `mlir/docs/Dialects/IRDL.md`
- `mlir/include/mlir/Dialect/IRDL/IR/IRDL.td`
- `mlir/include/mlir/Dialect/IRDL/IR/IRDLOps.td`
- `mlir/include/mlir/Dialect/IRDL/IR/IRDLTypes.td`
- `mlir/include/mlir/Dialect/IRDL/IR/IRDLAttributes.td`
- `mlir/include/mlir/Dialect/IRDL/IR/IRDLInterfaces.td`
- `mlir/lib/Dialect/IRDL/IR/IRDL.cpp`
- `mlir/lib/Dialect/IRDL/IR/IRDLOps.cpp`

Dynamic loading and verification:

- `mlir/include/mlir/Dialect/IRDL/IRDLLoading.h`
- `mlir/lib/Dialect/IRDL/IRDLLoading.cpp`
- `mlir/include/mlir/Dialect/IRDL/IRDLVerifiers.h`
- `mlir/lib/Dialect/IRDL/IRDLVerifiers.cpp`
- `mlir/include/mlir/Dialect/IRDL/IRDLSymbols.h`
- `mlir/lib/Dialect/IRDL/IRDLSymbols.cpp`
- `mlir/lib/Tools/mlir-opt/MlirOptMain.cpp`

Transform extension:

- `mlir/include/mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.td`
- `mlir/lib/Dialect/Transform/IRDLExtension/IRDLExtensionOps.cpp`

Translation:

- `mlir/include/mlir/Target/IRDLToCpp/IRDLToCpp.h`
- `mlir/lib/Target/IRDLToCpp/IRDLToCpp.cpp`
- `mlir/lib/Target/IRDLToCpp/TranslationRegistration.cpp`

Tests:

- `mlir/test/Dialect/IRDL/`
- `mlir/test/Dialect/Transform/test-interpreter-irdl.mlir`
- `mlir/test/tblgen-to-irdl/`
- `mlir/test/mlir-irdl-to-cpp/`
