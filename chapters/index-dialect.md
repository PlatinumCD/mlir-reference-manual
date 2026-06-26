# Index Dialect

## Beginner Summary

The `index` dialect contains operations for arithmetic on MLIR's builtin
`index` type.

The builtin `index` type is used for loop bounds, induction variables, tensor
dimensions, memref subscripts, offsets, sizes, and strides. It is similar to a
machine pointer-sized integer, like `intptr_t`, but MLIR often manipulates it
before the final target pointer width is known.

The `index` dialect exists for that exact problem. It lets the compiler express
index arithmetic while preserving target-independent folding and analysis.

Think of it as the dialect for scalar shape, loop, and subscript math when the
values have MLIR's `index` type and the pipeline should not prematurely commit
to `i32` or `i64`.

## Why This Dialect Exists

The `index` type is target-dependent. On one target it may lower to 32 bits; on
another, it may lower to 64 bits.

That creates a subtle compiler problem:

```text
index.constant 3000000000
```

This value has a different fixed-width interpretation depending on whether the
eventual index width is 32 or 64 bits. A compiler still wants to fold,
canonicalize, and reason about index expressions before that choice is final.

The `index` dialect provides operations whose folding rules are aware of this.
Constants are stored internally at the maximum supported index bitwidth, 64
bits, but folders only fold when the result is valid for both 32-bit and 64-bit
index interpretations.

That makes the dialect safer than blindly lowering index math to ordinary
fixed-width integer math too early.

## When It Matters

The `index` dialect matters whenever a pipeline is doing arithmetic on:

- Loop bounds and induction variables.
- Tensor or memref dimensions.
- Linearized offsets.
- Sizes, strides, and layout calculations.
- Shape-related scalar math that has already become `index`.
- Target-independent analysis of pointer-width-sized values.

It commonly appears near structured control flow, tensor, memref, affine, and
shape-related IR:

```text
tensor/memref/scf/affine/shape-like calculations
  -> index arithmetic and canonicalization
  -> convert-index-to-llvm or convert-index-to-spirv
  -> target-width integer operations
```

It is especially important in libraries or compiler pipelines that must support
both 32-bit and 64-bit targets.

## When To Use It

Use the `index` dialect when you are computing scalar values of builtin
`index` type and want the operation semantics to remain target-independent.

Use it for:

- Adding, subtracting, and multiplying index values.
- Signed or unsigned division and remainder on index values.
- Ceil and floor division for bounds and tiling math.
- Signed or unsigned min/max on index values.
- Bitwise operations and shifts on index values.
- Comparisons between index values.
- Casts between `index` and fixed-width integer types.
- Materializing `index` constants and querying the eventual index bitwidth.

Do not use it for vector or tensor elementwise arithmetic. The dialect is
scalar-only. If you need elementwise tensor or vector arithmetic, use a dialect
that models those container types and lower to scalar index operations only when
appropriate.

## Core Concepts

### Builtin `index`, Dialect `index`

There are two related ideas:

- `index` is a builtin MLIR type.
- `index.*` operations are operations in the Index dialect.

The type appears throughout MLIR, even in dialects that do not use Index dialect
operations. The dialect provides a dedicated arithmetic vocabulary for that
type.

### Target-Independent Folding

Index constants are stored with 64-bit internal width. Folding also computes
with 64-bit arithmetic, but it is careful about 32-bit targets.

Some operations can always be folded because truncating the folded 64-bit result
matches folding after truncating the operands to 32 bits. Addition is the
standard example:

```text
trunc(add64(a, b)) == add32(trunc(a), trunc(b))
```

Other operations are more dangerous. Division, remainder, right shifts, signed
and unsigned min/max, and comparisons can be affected by high bits. Those folds
are accepted only when the 32-bit and 64-bit interpretations agree for the
specific operands.

This is the most important reason the dialect exists.

### Signless Index Values

The builtin `index` type is treated as signless.

For operations where signedness matters, the dialect provides separate signed
and unsigned forms:

- `index.divs` and `index.divu`
- `index.rems` and `index.remu`
- `index.maxs` and `index.maxu`
- `index.mins` and `index.minu`
- `index.shrs` and `index.shru`
- signed and unsigned predicates on `index.cmp`

The `s` and `u` suffixes are part of the operation name because the type itself
does not carry signedness.

### Width Is Chosen At Conversion Time

The dialect does not decide whether `index` is 32 or 64 bits. That decision is
made when lowering.

For LLVM lowering, `convert-index-to-llvm` has an `index-bitwidth` option. A
value of `0` means derive the width from the data layout or machine word size.

For SPIR-V lowering, `convert-index-to-spirv` has a `use-64bit-index` option.
If it is false, index values lower as 32-bit integers.

### Scalar Only

Every operation in the dialect works on scalar `index` values, except casts
which convert between scalar `index` and scalar fixed-width integer types.

There are no `vector<...xindex>` Index dialect ops and no tensor Index dialect
ops.

## Types And Attributes

The dialect does not define a new type. It operates on MLIR's builtin `index`
type.

It defines a comparison predicate attribute for `index.cmp`. In assembly, the
predicate is written as a keyword:

```mlir
%is_less = index.cmp slt(%a, %b)
```

The supported predicates are:

| Predicate | Meaning |
| --- | --- |
| `eq` | Equal. |
| `ne` | Not equal. |
| `slt` | Signed less than. |
| `sle` | Signed less than or equal. |
| `sgt` | Signed greater than. |
| `sge` | Signed greater than or equal. |
| `ult` | Unsigned less than. |
| `ule` | Unsigned less than or equal. |
| `ugt` | Unsigned greater than. |
| `uge` | Unsigned greater than or equal. |

## Operations

The dialect has 26 operations.

### Constants And Size

| Operation | Purpose |
| --- | --- |
| `index.constant` | Creates an `index` constant. |
| `index.bool.constant` | Creates an `i1` constant, mainly for folded `index.cmp` results. |
| `index.sizeof` | Produces the eventual index bitwidth as an `index` value. |

### Arithmetic

| Operation | Purpose |
| --- | --- |
| `index.add` | Adds two index values. |
| `index.sub` | Subtracts the second index value from the first. |
| `index.mul` | Multiplies two index values. |
| `index.divs` | Signed division, rounding toward zero. |
| `index.divu` | Unsigned division, rounding toward zero. |
| `index.ceildivs` | Signed ceil division, rounding toward positive infinity. |
| `index.ceildivu` | Unsigned ceil division, rounding toward positive infinity. |
| `index.floordivs` | Signed floor division, rounding toward negative infinity. |
| `index.rems` | Signed remainder. |
| `index.remu` | Unsigned remainder. |

### Min, Max, Shifts, And Bitwise Operations

| Operation | Purpose |
| --- | --- |
| `index.maxs` | Signed maximum. |
| `index.maxu` | Unsigned maximum. |
| `index.mins` | Signed minimum. |
| `index.minu` | Unsigned minimum. |
| `index.shl` | Shift left. |
| `index.shrs` | Signed shift right. |
| `index.shru` | Unsigned shift right. |
| `index.and` | Bitwise and. |
| `index.or` | Bitwise or. |
| `index.xor` | Bitwise xor. |

### Comparisons And Casts

| Operation | Purpose |
| --- | --- |
| `index.cmp` | Compares two index values and returns `i1`. |
| `index.casts` | Casts between `index` and fixed-width integers using signed extension when widening. |
| `index.castu` | Casts between `index` and fixed-width integers using zero extension when widening. |

### Arithmetic Operations

Basic index arithmetic looks like ordinary integer arithmetic, but the operand
and result types are `index`.

```mlir
func.func @linearize(%i: index, %j: index, %cols: index) -> index {
  %row = index.mul %i, %cols
  %linear = index.add %row, %j
  return %linear : index
}
```

Signedness matters for division, remainder, min/max, and right shifts. Choose
the operation that matches the interpretation you need.

### Division Operations

The dialect has several division forms because compiler index math often needs
different rounding behavior.

| Operation | Rounding behavior |
| --- | --- |
| `index.divs` | Signed divide, rounds toward zero. |
| `index.divu` | Unsigned divide, rounds toward zero. |
| `index.ceildivs` | Signed divide, rounds toward positive infinity. |
| `index.ceildivu` | Unsigned divide, rounds toward positive infinity. |
| `index.floordivs` | Signed divide, rounds toward negative infinity. |

`index.ceildivu` is common in tiling and chunking:

```mlir
func.func @num_tiles(%n: index, %tile: index) -> index {
  %tiles = index.ceildivu %n, %tile
  return %tiles : index
}
```

Division by zero is undefined behavior. Signed division overflow is also
undefined for the signed division operations.

### Shift Operations

`index.shl`, `index.shrs`, and `index.shru` use an index shift amount.

The right-hand side is treated as unsigned. If the shift amount is equal to or
greater than the eventual index bitwidth, the result is poison. Because the
target width may be 32 or 64 bits, folders avoid folding shifts whose shift
amount would already be too large for a 32-bit target.

### Comparisons

`index.cmp` returns `i1`.

```mlir
func.func @in_bounds(%i: index, %n: index) -> i1 {
  %ok = index.cmp ult(%i, %n)
  return %ok : i1
}
```

Use signed predicates when the mathematical meaning is signed, and unsigned
predicates for sizes, offsets, and counts that cannot be negative after
interpretation.

### Cast Operations

`index.casts` and `index.castu` convert between `index` and concrete integer
types.

```mlir
func.func @casts(%idx: index, %x: i32) -> (i64, index) {
  %as_i64 = index.casts %idx : index to i64
  %as_index = index.castu %x : i32 to index
  return %as_i64, %as_index : i64, index
}
```

When widening:

- `index.casts` sign-extends.
- `index.castu` zero-extends.

When narrowing, both forms truncate.

## Transformations

The Index dialect has no large dialect-specific optimization pass. Its
transformations are mostly operation-local.

### Folding

Most operations have folders.

Important folds include:

- Constant folding for arithmetic, bitwise operations, comparisons, and casts.
- Identity folds such as `index.add %x, 0`, `index.sub %x, 0`, and
  `index.mul %x, 1`.
- Zero multiplication such as `index.mul %x, 0`.
- Comparison folding, including `index.cmp` on identical operands.
- Materializing folded comparison results with `index.bool.constant`.

The distinctive part is the width check. If a fold would produce different
results on 32-bit and 64-bit index targets, the folder leaves the operation in
the IR.

### Canonicalization

Several operations have canonicalization patterns.

Examples include:

- Combining constants through associative and commutative operations such as
  `index.add`, `index.mul`, `index.and`, `index.or`, and `index.xor`.
- Similar constant-combining for `index.maxs`, `index.maxu`, `index.mins`, and
  `index.minu`.
- Rewriting comparisons against subtraction, such as comparing `x - y` with
  zero, into a direct comparison between `x` and `y`.

These patterns make index expressions easier for later passes to analyze.

### Integer Range Inference

All Index dialect operations implement integer range inference.

This lets analyses reason about possible values of index expressions without
fully lowering them. The implementation considers both 64-bit storage and the
32-bit minimum index width where needed, so the inferred range remains useful
before the final target width is known.

## Conversions And Lowering Paths

The two main conversion passes are:

| Pass | Target | Important option |
| --- | --- | --- |
| `convert-index-to-llvm` | LLVM dialect | `index-bitwidth`, where `0` derives from data layout or machine word size. |
| `convert-index-to-spirv` | SPIR-V dialect | `use-64bit-index`, false by default. |

### Lowering To LLVM

`convert-index-to-llvm` lowers index operations to LLVM dialect integer
operations.

Most operations lower one-to-one:

| Index op | LLVM-style target |
| --- | --- |
| `index.add` | `llvm.add` |
| `index.sub` | `llvm.sub` |
| `index.mul` | `llvm.mul` |
| `index.divs` | `llvm.sdiv` |
| `index.divu` | `llvm.udiv` |
| `index.rems` | `llvm.srem` |
| `index.remu` | `llvm.urem` |
| `index.shl` | `llvm.shl` |
| `index.shrs` | `llvm.ashr` |
| `index.shru` | `llvm.lshr` |

The exotic divide operations `index.ceildivs`, `index.ceildivu`, and
`index.floordivs` expand to multiple LLVM operations because LLVM does not have
single instructions with those exact semantics.

`index.casts`, `index.castu`, `index.constant`, and `index.sizeof` lower using
the chosen index bitwidth.

### Lowering To SPIR-V

`convert-index-to-spirv` lowers index operations to SPIR-V integer operations.

Most operations lower one-to-one to SPIR-V arithmetic, bitwise, comparison, or
conversion operations. As with LLVM, ceil and floor division expand to operation
sequences.

The `use-64bit-index` option controls whether `index` becomes 32-bit or 64-bit
SPIR-V integer operations.

## Example IR

### Shape And Offset Math

```mlir
func.func @shape_math(%rows: index, %cols: index,
                      %i: index, %j: index) -> (index, i1) {
  %c4 = index.constant 4
  %row_offset = index.mul %i, %cols
  %linear = index.add %row_offset, %j
  %rounded = index.ceildivu %linear, %c4
  %total = index.mul %rows, %cols
  %in_bounds = index.cmp ult(%linear, %total)
  return %rounded, %in_bounds : index, i1
}
```

### Casts And Target Width

```mlir
func.func @cast_example(%idx: index, %signed: i32,
                        %unsigned: i32) -> (i64, index, index) {
  %wide = index.casts %idx : index to i64
  %from_signed = index.casts %signed : i32 to index
  %from_unsigned = index.castu %unsigned : i32 to index
  return %wide, %from_signed, %from_unsigned : i64, index, index
}
```

### Querying Index Size

```mlir
func.func @index_bits() -> index {
  %bits = index.sizeof
  return %bits : index
}
```

## Mental Model

Treat the Index dialect as arithmetic on "future pointer-width integers."

While the IR is still target-independent:

- The values have builtin type `index`.
- Operations stay in the `index` dialect.
- Folding and range inference avoid choices that would be wrong on a different
  index width.

When the target becomes concrete:

- `index` lowers to `i32`, `i64`, or the configured pointer-width integer type.
- Index operations lower to LLVM or SPIR-V integer operations.
- Constants are truncated or extended according to the chosen width.

## Gotchas

### `index` Is Not Always 64-Bit

Do not assume `index` means `i64`. The dialect stores constants with 64-bit
internal precision so it can reason about the maximum supported width, but the
lowered program may use 32-bit index values.

### Signedness Is In The Operation

The type is signless. If signedness matters, choose the signed or unsigned
operation explicitly.

For example, `index.maxs` and `index.maxu` can produce different results for
the same bit pattern.

### Some Folds Are Intentionally Missing

If a constant expression does not fold, that may be deliberate. The folder may
be preserving correctness across both 32-bit and 64-bit index targets.

### Shifts Can Produce Poison

A shift amount greater than or equal to the eventual index bitwidth is poison.
Because that bitwidth may not be known yet, avoid generating questionable shift
amounts in target-independent IR.

### Casts Need A Signedness Choice

`index.casts` and `index.castu` differ only when widening. Pick the one that
matches the intended interpretation of the source value.

### The Dialect Is Scalar

The Index dialect is not a replacement for tensor, vector, or arithmetic
dialects. It is for scalar builtin `index` values.

## Source Map

Use these source files when you want to inspect or update the dialect:

| Area | File |
| --- | --- |
| Dialect definition | `mlir/include/mlir/Dialect/Index/IR/IndexDialect.td` |
| Operations | `mlir/include/mlir/Dialect/Index/IR/IndexOps.td` |
| Comparison predicates | `mlir/include/mlir/Dialect/Index/IR/IndexEnums.td` |
| Operation implementation and folding | `mlir/lib/Dialect/Index/IR/IndexOps.cpp` |
| Integer range inference | `mlir/lib/Dialect/Index/IR/InferIntRangeInterfaceImpls.cpp` |
| LLVM conversion | `mlir/lib/Conversion/IndexToLLVM/IndexToLLVM.cpp` |
| SPIR-V conversion | `mlir/lib/Conversion/IndexToSPIRV/IndexToSPIRV.cpp` |
| Dialect tests | `mlir/test/Dialect/Index/` |
| LLVM conversion tests | `mlir/test/Conversion/IndexToLLVM/` |
| SPIR-V conversion tests | `mlir/test/Conversion/IndexToSPIRV/` |
