# PDLInterp Dialect

## Beginner Summary

The `pdl_interp` dialect is the executable form of PDL patterns.

PDL lets compiler developers describe rewrites in a declarative, pattern-shaped
IR. `pdl_interp` lowers those patterns into smaller interpreter operations that
can be compiled into MLIR's pattern bytecode. A beginner should think of it as
the "compiled pattern program" that MLIR runs when applying PDL rewrite
patterns.

Most users do not write `pdl_interp` by hand. They write PDL, C++ rewrite
patterns, or Transform dialect scripts. `pdl_interp` matters when you want to
understand how PDL is executed, why pattern matching can be optimized, and how
declarative rewrites become real matcher and rewriter code.

## Why This Dialect Exists

PDL is intentionally high level. It describes ideas such as "match an operation
with this name", "check this operand type", and "rewrite the root operation".
That form is pleasant for humans and tools, but it is not the most direct form
for an interpreter.

`pdl_interp` exists to make PDL execution explicit:

- It separates matching from rewriting.
- It turns high-level PDL constraints into explicit predicate checks.
- It uses branches, switches, and loops to encode matcher control flow.
- It records successful matches with enough metadata to invoke the right
  rewriter later.
- It gives MLIR's rewrite engine a compact form that can be compiled into
  pattern bytecode.

The source dialect summary calls it an "Interpreted pattern execution dialect".
That phrase is literal: the dialect is built for running patterns.

## When It Matters

`pdl_interp` matters when you are studying or debugging:

- PDL pattern lowering.
- Pattern bytecode generation.
- Why a PDL pattern matches or does not match.
- How native constraints and native rewrites connect to PDL.
- Multi-root or multi-operation patterns.
- Pattern benefit, root operation selection, and generated operation metadata.

It is especially important if you are implementing infrastructure around
`RewritePatternSet`, `FrozenRewritePatternSet`, or tools that generate PDL
patterns.

## When To Use It

Use `pdl_interp` directly when:

- You are testing PDL interpreter internals.
- You are reading the output of `-convert-pdl-to-pdl-interp`.
- You are debugging the lowering of a PDL pattern.
- You are working on MLIR's rewrite infrastructure.

Do not usually use it as the authoring dialect for compiler rewrites. For normal
work, write patterns in PDL or C++, then let MLIR lower PDL into `pdl_interp`.

## Core Concepts

### Matcher Function

The lowering creates a matcher function named `matcher`. It accepts the current
root operation as `!pdl.operation` and branches through a decision tree of
checks. If a pattern matches, the matcher emits `pdl_interp.record_match`.

### Rewriter Module

The lowering creates a nested module named `rewriters`. That module contains
`pdl_interp.func` functions that perform the rewrite after a match is selected.

### PDL Values

The dialect depends on the PDL dialect and uses PDL handle types:

- `!pdl.operation` is a handle to an operation.
- `!pdl.value` is a handle to an SSA value.
- `!pdl.type` is a handle to a type.
- `!pdl.attribute` is a handle to an attribute.
- `!pdl.range<...>` is a range of PDL handles.

These are not normal runtime program values. They are interpreter handles used
while matching and rewriting MLIR IR.

### Predicate Control Flow

Many `pdl_interp` operations combine a predicate with branches. For example,
`pdl_interp.check_operation_name` both checks an operation name and branches to
the success or failure block. This is different from a normal IR style where a
compare operation produces an `i1` and a separate branch consumes it.

That fusion is deliberate. It reduces interpreter work.

### Null Handles

Several query operations can return a null PDL handle. For example,
`pdl_interp.get_operand 3 of %op` returns null if `%op` has no operand at index
3. Matchers usually follow those queries with `pdl_interp.is_not_null`.

### Match Recording

`pdl_interp.record_match` does not rewrite immediately. It records that a
pattern matched, which rewriter function should run, what benefit the pattern
has, which operations contributed to the match location, and what values should
be passed into the rewriter.

## Operations

The dialect has 39 operations.

### Function And Control Operations

| Operation | Meaning |
| --- | --- |
| `pdl_interp.func` | Defines an interpreter function for matcher or rewriter code. |
| `pdl_interp.finalize` | Terminates a matcher or rewriter sequence. |
| `pdl_interp.branch` | Unconditionally branches to another block. |
| `pdl_interp.foreach` | Iterates over a range of PDL entities. |
| `pdl_interp.continue` | Continues the nearest `pdl_interp.foreach` loop. |
| `pdl_interp.record_match` | Records a successful pattern match and its rewriter metadata. |

### Predicate And Check Operations

| Operation | Meaning |
| --- | --- |
| `pdl_interp.apply_constraint` | Calls a registered native constraint and branches on success or failure. |
| `pdl_interp.are_equal` | Checks equality between two PDL entities or ranges. |
| `pdl_interp.check_attribute` | Checks an attribute against a constant attribute. |
| `pdl_interp.check_operand_count` | Checks an operation operand count, either exact or at least a minimum. |
| `pdl_interp.check_operation_name` | Checks an operation name. |
| `pdl_interp.check_result_count` | Checks an operation result count, either exact or at least a minimum. |
| `pdl_interp.check_type` | Checks a type handle against a known type. |
| `pdl_interp.check_types` | Checks a range of type handles against a known type list. |
| `pdl_interp.is_not_null` | Checks that a PDL entity or range handle exists. |

### Switch Operations

| Operation | Meaning |
| --- | --- |
| `pdl_interp.switch_attribute` | Dispatches based on an attribute value. |
| `pdl_interp.switch_operand_count` | Dispatches based on an operation's operand count. |
| `pdl_interp.switch_operation_name` | Dispatches based on an operation name. |
| `pdl_interp.switch_result_count` | Dispatches based on an operation's result count. |
| `pdl_interp.switch_type` | Dispatches based on one type value. |
| `pdl_interp.switch_types` | Dispatches based on a range of type values. |

Switch operations are useful when multiple PDL patterns share the same query but
have different constant answers. Instead of emitting repeated checks, lowering
can build a compact dispatch.

### IR Query Operations

| Operation | Meaning |
| --- | --- |
| `pdl_interp.extract` | Extracts one item from a PDL range. |
| `pdl_interp.get_attribute` | Gets a named attribute from an operation. |
| `pdl_interp.get_attribute_type` | Gets the type associated with an attribute. |
| `pdl_interp.get_defining_op` | Gets the defining operation for a value or value range. |
| `pdl_interp.get_operand` | Gets one indexed operand from an operation. |
| `pdl_interp.get_operands` | Gets one operand group, or all operands, from an operation. |
| `pdl_interp.get_result` | Gets one indexed result from an operation. |
| `pdl_interp.get_results` | Gets one result group, or all results, from an operation. |
| `pdl_interp.get_users` | Gets users of a value or value range. |
| `pdl_interp.get_value_type` | Gets the type of a value, or the type range for a value range. |

These operations navigate the IR being matched. They are the interpreter-level
equivalent of calling APIs such as `Operation::getOperand`,
`Operation::getResult`, `Value::getDefiningOp`, and `Value::getUsers`.

### Rewrite And Construction Operations

| Operation | Meaning |
| --- | --- |
| `pdl_interp.apply_rewrite` | Calls a registered native rewrite function. |
| `pdl_interp.create_attribute` | Creates a PDL handle for a constant attribute. |
| `pdl_interp.create_operation` | Creates an MLIR operation from PDL handles. |
| `pdl_interp.create_range` | Creates a PDL range from PDL entities or ranges. |
| `pdl_interp.create_type` | Creates a PDL handle for a constant type. |
| `pdl_interp.create_types` | Creates a PDL range handle for constant types. |
| `pdl_interp.erase` | Erases an operation through the pattern rewriter. |
| `pdl_interp.replace` | Replaces an operation through the pattern rewriter. |

These operations are normally found inside the generated rewriter functions.

## Transformations

`pdl_interp` is not a general-purpose program optimization dialect. It does not
have a large suite of standalone optimization passes like `linalg`, `scf`, or
`vector`.

The important transformation is:

```text
PDL patterns -> PDLInterp matcher/rewriter IR -> PDL bytecode
```

The lowering process performs important pattern-compiler work:

- It creates the `matcher` function.
- It creates the `rewriters` module.
- It orders predicates into a matcher decision tree.
- It reuses shared constraint checks when possible.
- It lowers PDL rewrites into explicit create, erase, replace, and native
  rewrite calls.
- It emits loops such as `pdl_interp.foreach` for multi-root or user-walking
  patterns.
- It emits switch operations when several patterns can share one dispatch.

Inside the rewrite engine, `FrozenRewritePatternSet` can lower PDL patterns with
the `convert-pdl-to-pdl-interp` pass and then build a `PDLByteCode` object from
the resulting module.

## Conversions And Lowering Paths

### Main Pass

The main conversion pass is:

```text
-convert-pdl-to-pdl-interp
```

It is a module pass. It converts `pdl.pattern` operations into `pdl_interp`
matcher and rewriter operations.

### Normal Pipeline Position

The usual path is:

```text
pdl.pattern
  -> -convert-pdl-to-pdl-interp
  -> pdl_interp.func @matcher plus module @rewriters
  -> PDL bytecode inside the rewrite engine
  -> PatternApplicator runs matches and rewrites
```

`pdl_interp` is usually not lowered to LLVM IR, SPIR-V, or another target
dialect. It is consumed by MLIR's rewrite infrastructure.

### What Gets Lowered From PDL

Common PDL constructs lower as follows:

| PDL idea | Typical `pdl_interp` form |
| --- | --- |
| Match operation name | `pdl_interp.check_operation_name` or `pdl_interp.switch_operation_name` |
| Match operand count | `pdl_interp.check_operand_count` or `pdl_interp.switch_operand_count` |
| Match result count | `pdl_interp.check_result_count` or `pdl_interp.switch_result_count` |
| Match type | `pdl_interp.check_type`, `pdl_interp.check_types`, `pdl_interp.switch_type`, or `pdl_interp.switch_types` |
| Native constraint | `pdl_interp.apply_constraint` |
| Native rewrite | `pdl_interp.apply_rewrite` |
| PDL rewrite region | `pdl_interp.create_operation`, `pdl_interp.erase`, `pdl_interp.replace`, and helpers |
| Successful pattern | `pdl_interp.record_match` |

## Example IR

### A Small Matcher And Rewriter

This example matches a root operation named `foo.op` with zero operands. On
success, it records a match that invokes `@rewriters::@rewrite`.

```mlir
module {
  pdl_interp.func @matcher(%root: !pdl.operation) {
    pdl_interp.check_operation_name of %root is "foo.op" -> ^matched, ^failed

  ^failed:
    pdl_interp.finalize

  ^matched:
    pdl_interp.check_operand_count of %root is 0 -> ^record, ^failed

  ^record:
    pdl_interp.record_match @rewriters::@rewrite(%root : !pdl.operation) :
      benefit(1), loc([%root]), root("foo.op") -> ^failed
  }

  module @rewriters {
    pdl_interp.func @rewrite(%root: !pdl.operation) {
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}
```

The key point is that the matcher does not erase `%root` directly. It records a
match. The rewriter function performs the erase after the rewrite engine chooses
that match.

### Creating And Replacing Operations

```mlir
module {
  pdl_interp.func @rewriter(%input: !pdl.value,
                            %attribute: !pdl.attribute,
                            %type: !pdl.type) {
    %created = pdl_interp.create_operation "foo.op"(%input : !pdl.value)
      {"attr" = %attribute} -> (%type : !pdl.type)
    %result = pdl_interp.get_result 0 of %created
    pdl_interp.replace %created with (%result : !pdl.value)
    pdl_interp.finalize
  }
}
```

This shows how construction and replacement are expressed with PDL handles. In
real generated code, the replaced operation is usually the matched root, not the
newly created operation in the same snippet.

### Iterating Over Users

```mlir
module {
  pdl_interp.func @walk_users(%value: !pdl.value) {
    %users = pdl_interp.get_users of %value : !pdl.value
    pdl_interp.foreach %user : !pdl.operation in %users {
      %operand = pdl_interp.get_operand 0 of %user
      pdl_interp.continue
    } -> ^done

  ^done:
    pdl_interp.finalize
  }
}
```

`foreach` is useful for patterns that need to look away from a single root and
inspect related operations.

### Dispatching On Operation Name

```mlir
module {
  pdl_interp.func @switch_on_root(%root: !pdl.operation) {
    pdl_interp.switch_operation_name of %root to ["foo.op", "bar.op"](^foo, ^bar) -> ^unknown

  ^foo:
    pdl_interp.finalize

  ^bar:
    pdl_interp.finalize

  ^unknown:
    pdl_interp.finalize
  }
}
```

This is the compact form of several operation-name checks sharing the same
queried root.

## Mental Model

Think of PDLInterp as a small instruction set for pattern matching:

1. Start with a candidate operation.
2. Query facts from that operation and nearby IR.
3. Branch based on names, counts, types, attributes, equality, and native
   constraints.
4. Record every successful pattern match.
5. Later, invoke the recorded rewriter to create, erase, replace, or call
   native rewrite code.

That mental model explains why the dialect has many operations that look like
basic C++ rewrite APIs. The dialect is not modeling the user's program. It is
modeling the pattern engine's work.

## Gotchas

- `pdl_interp` is internal-facing. If you are just writing a compiler rewrite,
  start with PDL or C++ patterns.
- Query operations can return null handles. A missing `is_not_null` can make a
  matcher behave differently than expected.
- `record_match` records a possible rewrite; it does not perform the rewrite.
- `benefit` belongs to the match metadata and is used when choosing among
  matches.
- `root("...")` metadata helps connect a match to a root operation kind.
- Native constraints and rewrites must be registered with the interpreter
  environment. The IR only stores their names and arguments.
- `pdl_interp` functions are not normal application functions. They are
  interpreter functions over PDL handles.
- The dialect depends on PDL types. Understanding `!pdl.operation`,
  `!pdl.value`, `!pdl.type`, `!pdl.attribute`, and `!pdl.range<...>` is
  necessary before reading complex examples.

## Source Map

Important source files in the LLVM tree:

- `mlir/include/mlir/Dialect/PDLInterp/IR/PDLInterpOps.td` defines the dialect
  and all operations.
- `mlir/lib/Dialect/PDLInterp/IR/PDLInterp.cpp` implements parsing, printing,
  verification, and helpers.
- `mlir/include/mlir/Conversion/Passes.td` declares
  `convert-pdl-to-pdl-interp`.
- `mlir/lib/Conversion/PDLToPDLInterp/PDLToPDLInterp.cpp` lowers PDL patterns
  into `pdl_interp`.
- `mlir/lib/Conversion/PDLToPDLInterp/Predicate*.{h,cpp}` and
  `RootOrdering.*` support matcher planning and predicate ordering.
- `mlir/lib/Rewrite/FrozenRewritePatternSet.cpp` shows how PDL patterns are
  lowered before bytecode construction.
- `mlir/lib/Rewrite/ByteCode.cpp` consumes `pdl_interp` operations and builds
  the bytecode interpreter program.
- `mlir/test/Dialect/PDLInterp/ops.mlir` contains parser and printer examples.
- `mlir/test/Conversion/PDLToPDLInterp/` contains examples of PDL lowering into
  `pdl_interp`.
