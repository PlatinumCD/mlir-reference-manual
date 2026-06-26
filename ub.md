# `ub` Dialect

## Beginner Summary

The `ub` dialect represents undefined behavior in MLIR. It is a small dialect,
but it teaches an important compiler idea: sometimes the compiler needs an IR
value that means "using this value can trigger undefined behavior" or a
terminator that means "execution cannot legally continue here".

The dialect has two operations:

| Operation | Meaning |
| --- | --- |
| `ub.poison` | Produces a poisoned value. This is deferred undefined behavior. |
| `ub.unreachable` | Terminates control flow with immediate undefined behavior if reached. |

Most programmers do not write this dialect by hand. It appears in compiler
pipelines when an operation needs a placeholder value with undefined semantics,
or when lowering must preserve the fact that a control-flow path is impossible.

## Why This Dialect Exists

MLIR needs a target-independent way to represent undefined behavior before IR is
lowered to a concrete target such as LLVM IR or SPIR-V.

Without `ub`, passes would have to choose target-specific operations too early,
such as `llvm.mlir.poison`, `llvm.unreachable`, `spirv.Undef`, or
`spirv.Unreachable`. The `ub` dialect gives MLIR a small shared vocabulary for
these concepts:

- `ub.poison` says "there is a value here, but the value is poison".
- `ub.unreachable` says "if execution reaches this point, the program has
  immediate undefined behavior".

This keeps undefined-behavior modeling separate from a particular backend.

## When It Matters

`ub` matters when a pass needs to represent undefined semantics explicitly.

Common situations:

- A rewrite needs to create a value for an impossible or don't-care case.
- A conversion needs a placeholder value that will become a backend poison or
  undef value later.
- A pass proves a control-flow path cannot be reached.
- A frontend or optimizer wants to preserve language undefined behavior without
  lowering directly to LLVM or SPIR-V.
- A conversion target requires every branch to end in a terminator, and the only
  meaningful terminator is "this path is impossible".

It also matters when reading lowered IR. Seeing `ub.poison` tells you that the
IR has intentionally stopped promising a concrete value at that point.

## When To Use It

Use `ub` when you need a portable undefined-behavior marker.

Good uses:

- Materializing a poison value in target-independent MLIR.
- Representing a proven-impossible path before converting to a backend.
- Writing tests for conversion to LLVM or SPIR-V.
- Implementing a pass that needs to match poison constants with
  `ub::m_Poison()`.

Avoid direct use when:

- You mean "some arbitrary valid value". Poison is stronger than "unknown".
- You need a normal default value. Use an explicit constant instead.
- You are already writing final LLVM IR and specifically need
  `llvm.mlir.poison` or `llvm.unreachable`.
- You are already writing final SPIR-V and specifically need `spirv.Undef` or
  `spirv.Unreachable`.

## Core Concepts

### Deferred Undefined Behavior

`ub.poison` creates a poison value. The operation itself does not immediately
stop execution. The undefined behavior is deferred until the poison value is
used in a way that makes the behavior observable according to the target's
semantics.

This is why `ub.poison` is a value-producing operation:

```mlir
%x = ub.poison : i32
return %x : i32
```

The IR still has a value `%x`, but the value carries poison semantics.

### Immediate Undefined Behavior

`ub.unreachable` is a terminator. If control flow reaches it, the program has
immediate undefined behavior:

```mlir
func.func @bad_path() {
  ub.unreachable
}
```

This is useful for impossible paths, failed assumptions, or regions where the
compiler has proven there is no legal continuation.

### Poison Attributes

`ub.poison` carries a poison attribute. The default is `#ub.poison`, printed
implicitly:

```mlir
%a = ub.poison : i32
%b = ub.poison <#ub.poison> : i32
```

Both forms mean the same thing for the built-in poison attribute. The dialect
also defines `PoisonAttrInterface`, which allows attributes to model more
specific poison semantics, such as a future partially poisoned vector attribute.

### Constant-Like Folding

`ub.poison` is `ConstantLike` and `Pure`. Its folder returns its poison
attribute, so canonicalization can merge identical poison constants:

```mlir
%a = ub.poison : i32
%b = ub.poison : i32
```

After canonicalization, both uses can share one poison operation.

### Inlining

The dialect defines an inliner interface, and UB operations can be inlined. A
function returning `ub.poison` can be inlined into its caller without changing
the meaning of the poison value.

## Operations

### `ub.poison`

`ub.poison` materializes a compile-time poisoned constant value.

Syntax:

```mlir
%0 = ub.poison : i32
%1 = ub.poison <#ub.poison> : vector<4xi64>
```

Properties:

| Property | Meaning |
| --- | --- |
| Result | Any MLIR type. Tests cover integers, complex values, vectors, and tensors. |
| Attribute | A `PoisonAttrInterface` value, defaulting to `#ub.poison`. |
| Traits | `ConstantLike`, `Pure`. |
| Folding | Folds to its poison attribute. |

Use it when the IR needs a value but the value is intentionally undefined if
observed.

### `ub.unreachable`

`ub.unreachable` is a terminator that represents immediate undefined behavior.

Syntax:

```mlir
ub.unreachable
```

Properties:

| Property | Meaning |
| --- | --- |
| Result | No results. |
| Trait | `Terminator`. |
| Control flow | Ends the current block. |

Use it for paths that cannot be legally executed.

## Attributes And Interfaces

The dialect defines one built-in attribute and one attribute interface:

| Name | Meaning |
| --- | --- |
| `PoisonAttr` / `#ub.poison` | The default poison value attribute. |
| `PoisonAttrInterface` | Interface implemented by poison attributes. |

The helper matcher `ub::m_Poison()` can match poison attributes, poison
constants, and operations that fold to poison attributes. This is useful in C++
rewrite patterns:

```c++
matchPattern(value, ub::m_Poison());
```

## Transformations

The dialect has a small amount of local behavior:

| Transformation | Effect |
| --- | --- |
| Canonicalization / folding | Identical `ub.poison` constants can be merged because the op folds to its poison attribute. |
| Inlining | UB operations are legal to inline. |
| Generic `convert-to-llvm` interface | The dialect registers conversion patterns for the generic LLVM conversion flow. |

There are no large optimization pipelines specific to `ub`. Its main role is to
be consumed by backend conversions.

## Conversions And Lowering Paths

The important lowering paths are:

| Pass | Input | Output |
| --- | --- | --- |
| `convert-ub-to-llvm` | `ub.poison`, `ub.unreachable` | `llvm.mlir.poison`, `llvm.unreachable` |
| `convert-ub-to-spirv` | `ub.poison`, `ub.unreachable` | `spirv.Undef`, `spirv.Unreachable` |
| `convert-to-llvm` with UB enabled | `ub` ops | Uses the UB dialect's LLVM conversion interface. |

`convert-ub-to-llvm` has an `index-bitwidth` option. If it is `0`, the pass
derives the index bitwidth from the data layout. Otherwise, it uses the
specified bitwidth for index type conversion.

LLVM example:

```mlir
%0 = ub.poison : index
ub.unreachable
```

can lower to:

```mlir
%0 = llvm.mlir.poison : i64
llvm.unreachable
```

SPIR-V example:

```mlir
%0 = ub.poison : vector<4xf32>
```

can lower to:

```mlir
%0 = spirv.Undef : vector<4xf32>
```

## Example IR

This example uses `ub.poison` to return a value whose behavior is intentionally
undefined if observed:

```mlir
func.func @poison_value(%cond: i1) -> i32 {
  %bad = ub.poison : i32
  return %bad : i32
}
```

This example uses `ub.unreachable` for an impossible branch:

```mlir
func.func @only_true(%cond: i1) {
  cf.cond_br %cond, ^ok, ^bad

^ok:
  return

^bad:
  ub.unreachable
}
```

After LLVM lowering, the second example's bad block ends in
`llvm.unreachable`.

## Mental Model

Think of `ub` as the dialect for "the compiler is no longer promising a normal
program value or a normal continuation."

`ub.poison` is still a value. It can flow through SSA and be converted later.
`ub.unreachable` is not a value; it is a hard end to a control-flow path.

The distinction matters. Poison says "this value is invalid if used in the
wrong way." Unreachable says "execution cannot legally get past this point."

## Gotchas

- Poison is not the same as zero, null, or a harmless placeholder.
- Poison is not the same as a runtime error. It gives the optimizer freedom
  because the program has undefined behavior if the poison becomes relevant.
- `ub.unreachable` is a terminator. You cannot put ordinary operations after it
  in the same block.
- `ub.poison` can have any result type, but backend conversion still has to be
  able to convert that type.
- `convert-ub-to-llvm` only converts the built-in `#ub.poison` attribute. A
  custom `PoisonAttrInterface` attribute may need custom lowering.
- `spirv.Undef` is the SPIR-V lowering for `ub.poison`; do not read that as a
  normal deterministic value.

## Source Map

Use these files in the LLVM repo when you need exact behavior:

| Topic | Files |
| --- | --- |
| Dialect, ops, and attribute definitions | `mlir/include/mlir/Dialect/UB/IR/UBOps.td` |
| Poison attribute interface | `mlir/include/mlir/Dialect/UB/IR/UBOpsInterfaces.td` |
| Matchers | `mlir/include/mlir/Dialect/UB/IR/UBMatchers.h` |
| Dialect implementation | `mlir/lib/Dialect/UB/IR/UBOps.cpp` |
| LLVM conversion | `mlir/include/mlir/Conversion/UBToLLVM/UBToLLVM.h`, `mlir/lib/Conversion/UBToLLVM/UBToLLVM.cpp` |
| SPIR-V conversion | `mlir/include/mlir/Conversion/UBToSPIRV/UBToSPIRV.h`, `mlir/lib/Conversion/UBToSPIRV/UBToSPIRV.cpp` |
| Pass declarations | `mlir/include/mlir/Conversion/Passes.td` |
| Parser/printer tests | `mlir/test/Dialect/UB/ops.mlir` |
| Canonicalization and inlining tests | `mlir/test/Dialect/UB/canonicalize.mlir`, `mlir/test/Dialect/UB/inlining.mlir` |
| Conversion tests | `mlir/test/Conversion/UBToLLVM/ub-to-llvm.mlir`, `mlir/test/Conversion/UBToSPIRV/ub-to-spirv.mlir`, `mlir/test/Conversion/ConvertToSPIRV/ub.mlir` |
