# ControlFlow Dialect

## Beginner Summary

The `cf` dialect is MLIR's low-level control-flow dialect.

It defines unstructured control flow over SSA blocks:

- `cf.br`: unconditional branch.
- `cf.cond_br`: conditional branch.
- `cf.switch`: multi-way branch.
- `cf.assert`: runtime assertion.

For beginners, `cf` is the dialect you see when structured control flow such as
`scf.if` or `scf.for` has been lowered toward a control-flow graph.

## Why This Dialect Exists

MLIR has both structured and unstructured control flow.

Structured control flow uses operations with regions:

```text
scf.if
scf.for
scf.while
```

Unstructured control flow uses blocks and branches:

```text
^bb0:
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
```

The `cf` dialect exists for the unstructured form. This is useful because many
low-level targets and compiler analyses operate on control-flow graphs rather
than nested structured regions.

The dialect is still target-independent. A `cf.cond_br` is lower than `scf.if`,
but it is not yet an LLVM branch, a SPIR-V branch, or machine code.

## When It Matters

The `cf` dialect matters when a pipeline crosses the boundary between structured
MLIR and CFG-style IR.

You often see it:

- After `convert-scf-to-cf`.
- Before `convert-cf-to-llvm`.
- Before some SPIR-V lowering paths.
- In low-level code that already looks like basic blocks.
- In canonicalization or CFG cleanup.
- Around code that needs block arguments and explicit branch operands.

You may also see it temporarily when a lowering pass needs to express control
flow that no longer fits a structured `scf` operation.

## When To Use It

Use `cf` when the IR needs explicit block-to-block jumps.

Good uses include:

- Lowering `scf.if`, `scf.while`, or `scf.for` toward a CFG.
- Representing arbitrary branches that are not naturally structured.
- Passing values to successor blocks through block arguments.
- Modeling low-level switch dispatch.
- Preserving target-independent control flow before LLVM or SPIR-V conversion.
- Emitting runtime checks with `cf.assert`.

Prefer `scf` when the control flow is naturally structured and you still want
loop and region structure for analysis, transformations, or readability.

## Core Concepts

### Blocks And Terminators

`cf.br`, `cf.cond_br`, and `cf.switch` are terminators. They end a block and
choose the next block.

In MLIR, successor blocks may have block arguments. Branch operands provide the
values for those block arguments.

Example:

```text
cf.br ^done(%x : i32)
^done(%y: i32):
```

The branch operand `%x` becomes the block argument `%y` in `^done`.

### Structured Versus Unstructured

Structured control flow nests regions:

```text
scf.if %cond {
  ...
} else {
  ...
}
```

Unstructured control flow links blocks:

```text
cf.cond_br %cond, ^then, ^else
```

Both can represent conditionals, but they are useful at different stages.
Structured control flow is easier for many high-level transformations.
Unstructured control flow is closer to LLVM IR and many target CFGs.

### Branch Interfaces

The branch operations implement MLIR branch interfaces so generic analyses can
ask:

- Which successors can this operation branch to?
- What operands are passed to each successor?
- Which successor is chosen for known operands?

`cf.cond_br` also implements weighted branch behavior through optional branch
weights.

## Operations

The current ControlFlow dialect in this LLVM checkout defines four generated
operations.

### `cf.assert`

`cf.assert` checks a one-bit integer condition at runtime.

If the condition is true, execution continues. If the condition is false, the
program aborts. The operation carries a string message for diagnostics.

Use it for runtime assumptions that should fail loudly when violated.

### `cf.br`

`cf.br` is an unconditional branch to one successor block.

It may pass operands to the destination block. The operand count and types must
match the destination block arguments.

Use it when control always continues at the same successor.

### `cf.cond_br`

`cf.cond_br` branches on an `i1` condition.

If the condition is true, it jumps to the true successor. Otherwise, it jumps to
the false successor. Each successor may receive a different operand list.

The operation may also carry branch weights.

Use it for low-level if/else control flow.

### `cf.switch`

`cf.switch` branches on a signless integer value.

It has:

- A default destination.
- Zero or more integer case values.
- One destination per case.
- Optional operands for each destination.

Use it for low-level multi-way dispatch.

## Transformations

The `cf` dialect does not have a large standalone transform-pass library like
`vector` or `memref`. Its main transformations are conversions to or from other
control-flow representations.

### Canonicalization

The operations provide canonicalization hooks:

- `cf.assert` has canonicalization.
- `cf.br` has canonicalization.
- `cf.cond_br` has canonicalization.
- `cf.switch` has canonicalization and verification.

Typical cleanups include simplifying branches whose conditions are known,
removing unnecessary branch operands, or simplifying switches when cases become
unreachable or redundant.

### Bufferization And Type Conversion Support

The ControlFlow dialect has support files for:

- Buffer deallocation interfaces.
- Bufferizable operation interfaces.
- Structural type conversions.

This matters because branch operands and block arguments form type boundaries.
If a conversion changes the type of a value, it must update both the branch
operand and the successor block argument consistently.

## Conversions And Lowering Paths

### From SCF

`convert-scf-to-cf` lowers structured control flow to ControlFlow dialect
branches.

This is the usual path from structured loops and conditionals to an explicit
CFG. It replaces structured `scf` operations with blocks, `cf.br`, and
`cf.cond_br`.

The pass has an `allow-pattern-rollback` option.

### To SCF

`lift-cf-to-scf` attempts to lift ControlFlow dialect operations back into
structured `scf` operations.

The pass is named "lift" because it is not guaranteed to replace every
ControlFlow operation. If a region has a single kind of return-like operation,
the pass can replace all ControlFlow operations successfully. Otherwise, it may
leave a `cf.switch` that branches to one block per return-like operation kind.

This path is useful when a CFG can be recovered into structured control flow for
further high-level optimization.

### To LLVM

`convert-cf-to-llvm` lowers ControlFlow operations to the LLVM dialect.

The conversion includes patterns for:

- `cf.br`
- `cf.cond_br`
- `cf.switch`
- `cf.assert`

`cf.assert` lowers through an abort-style runtime path by default.

The pass has an `index-bitwidth` option. The generic `convert-to-llvm` driver can
also use the ControlFlow dialect's LLVM conversion interface.

### To SPIR-V

`convert-cf-to-spirv` lowers supported ControlFlow operations to the SPIR-V
dialect.

The direct pattern set covers:

- `cf.br` to `spirv.Branch`
- `cf.cond_br` to `spirv.BranchConditional`

This path is useful after the surrounding IR has been shaped into forms SPIR-V
can represent legally.

## Example IR

### Direct Branch

```mlir
func.func @direct(%x: i32) -> i32 {
  cf.br ^done(%x : i32)
^done(%y: i32):
  func.return %y : i32
}
```

The value `%x` is passed as the block argument `%y`.

### Conditional Branch

```mlir
func.func @select_cf(%a: i32, %b: i32, %cond: i1) -> i32 {
  cf.cond_br %cond, ^true(%a : i32), ^false(%b : i32)
^true(%x: i32):
  cf.br ^done(%x : i32)
^false(%y: i32):
  cf.br ^done(%y : i32)
^done(%z: i32):
  func.return %z : i32
}
```

This is the CFG version of a simple if/else select.

### Switch

```mlir
func.func @switch_cf(%flag: i32, %a: i32, %b: i32, %c: i32) -> i32 {
  cf.switch %flag : i32, [
    default: ^done(%a : i32),
    42: ^done(%b : i32),
    43: ^done(%c : i32)
  ]
^done(%x: i32):
  func.return %x : i32
}
```

All cases branch to the same block here, but they pass different values.

### Runtime Assert

```mlir
func.func @assert_cf(%ok: i1) {
  cf.assert %ok, "expected ok"
  func.return
}
```

This keeps a runtime check in the IR until it is canonicalized away or lowered.

## Mental Model

Think of `cf` as "MLIR basic-block control flow."

It is lower-level than `scf`, because it talks directly about branches and
successor blocks.

It is higher-level than LLVM IR, because it still uses MLIR block arguments,
MLIR types, and dialect conversion infrastructure.

The main question is:

```text
Do I want structured regions or explicit CFG edges?
```

If you want structured regions, use `scf`. If you want explicit edges between
basic blocks, use `cf`.

## Gotchas

`cf` branches are terminators.

After `cf.br`, `cf.cond_br`, or `cf.switch`, the current block is done.

Branch operands must match successor block arguments.

If `cf.br ^bb1(%x : i32)` targets `^bb1(%y: f32)`, verification fails because
the types do not match.

`cf.cond_br` cannot target the entry block of a region.

This restriction prevents invalid control-flow shapes inside MLIR regions.

`lift-cf-to-scf` is best-effort.

Not every CFG has clean structured control flow. The pass can leave some
ControlFlow operations behind when the region cannot be fully lifted.

`cf.assert` has runtime behavior.

It is not just a compile-time hint. If the condition is false at runtime, the
lowered program is expected to abort.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td`
- `mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp`
- `mlir/include/mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h`
- `mlir/lib/Dialect/ControlFlow/Transforms/StructuralTypeConversions.cpp`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.cpp`
- `mlir/lib/Conversion/ControlFlowToSCF/ControlFlowToSCF.cpp`
- `mlir/lib/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.cpp`
- `mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp`

Generated op documentation source:

```text
mlir-tblgen --gen-op-doc -dialect=cf \
  mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td
```
