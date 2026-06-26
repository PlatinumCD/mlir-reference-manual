# CIRCT `hwarith` Dialect

The CIRCT `hwarith` dialect models bitwidth-aware integer arithmetic for hardware frontends. Its main purpose is to make arithmetic signedness and result-width inference explicit before the design is lowered into CIRCT's lower-level hardware dialects.

For a beginner, the useful mental model is this: `hwarith` is a short-lived, strongly typed arithmetic layer between frontend expression construction and core RTL-style IR. It lets a frontend say "this value is unsigned" or "this value is signed" and then lets the dialect infer the safe result width for operations such as addition, subtraction, multiplication, division, and comparison.

The local checkout defines 7 `hwarith` operations, one signedness-aware integer type constraint, one comparison predicate enum attribute, one canonicalization pattern, and one named conversion pass.

## When HWArith Is Important

`hwarith` is important when a hardware frontend needs arithmetic that behaves like hardware arithmetic but wants those rules to be visible and checked. It is especially useful for frontends such as PyCDE that generate CIRCT IR from a high-level hardware construction API.

Use this dialect when you need to answer questions like:

- Is this operand signed or unsigned?
- How wide should the result of this arithmetic expression be?
- Did an addition or subtraction need an extra bit for overflow?
- Did a mixed signed/unsigned expression become signed?
- Is a cast doing sign extension, zero extension, truncation, or only a signedness reinterpretation?
- Which `comb` operation will this eventually become?

You usually do not keep `hwarith` until final RTL emission. The CIRCT rationale describes it as a frontend target with a short lifetime. It should be lowered early to `hw` and `comb` once the signedness-sensitive decisions have been made explicit.

## Why It Is Needed

Core hardware dialects such as `hw` and `comb` operate close to RTL structure. They are good at representing wires, constants, bit operations, and combinational logic, but they do not by themselves preserve a frontend's signedness intent in every arithmetic expression.

SystemVerilog has width and signedness rules, but those rules can be subtle. CIRCT's `hwarith` dialect provides a strongly typed alternative: every arithmetic operator documents and verifies its result type. The frontend still provides concrete result types in IR, but the operations implement MLIR type inference and verification so mistakes are caught.

The key design choice is that `hwarith` arithmetic operations do not accept signless integer operands or signless arithmetic results. They use MLIR integer types with signedness semantics:

```text
uiN  unsigned N-bit integer
siN  signed N-bit integer
```

The implementation accepts only nonzero-width signed or unsigned integer types for the `HWArithIntegerType` constraint. Signless `iN` values can appear at the boundary through `hwarith.cast`, and `hwarith.icmp` returns signless `i1` because its result is a boolean.

## Types And Attributes

`hwarith` does not introduce a new printed type syntax. It constrains MLIR integer types to require signedness semantics. A value of type `ui8` is an 8-bit unsigned integer, while `si8` is an 8-bit signed integer. The sign is part of the type-level contract used by the operation verifier and by lowering.

The dialect defines one operation attribute enum for comparisons:

```text
eq
ne
lt
ge
le
gt
```

These are the allowed `hwarith.icmp` predicates. During lowering, `eq` and `ne` map directly to equality and inequality comparisons. Ordered predicates such as `lt`, `ge`, `le`, and `gt` lower to signed or unsigned `comb.icmp` predicates depending on the common comparison type inferred from the operands.

## Operation Inventory

The exact `hwarith` operation names in this checkout are:

```text
hwarith.constant
hwarith.add
hwarith.sub
hwarith.mul
hwarith.div
hwarith.cast
hwarith.icmp
```

### Constants

`hwarith.constant` produces a signedness-aware integer constant.

```mlir
%0 = hwarith.constant 22 : ui5
%1 = hwarith.constant -1 : si2
```

The result must be a nonzero-width signed or unsigned integer type. A signless result such as `i1` is rejected, and a zero-width result such as `ui0` is rejected. The operation folds to its raw integer attribute, and the conversion pass lowers it to `hw.constant`.

### Addition

`hwarith.add` is bitwidth-aware integer addition. It is binary and commutative.

```mlir
%0 = hwarith.add %a, %b : (ui3, ui4) -> ui5
%1 = hwarith.add %c, %d : (si3, si3) -> si4
%2 = hwarith.add %e, %f : (ui3, si4) -> si5
```

For same-signed operands, the result width is `max(lhs_width, rhs_width) + 1` and the result keeps that signedness. For mixed signedness, the result is signed. If the unsigned operand is at least as wide as the signed operand, the result needs one more bit than the usual max-plus-one rule, because the unsigned range needs an extra signed bit.

Use `hwarith.add` when overflow must be represented instead of silently truncated.

### Subtraction

`hwarith.sub` is bitwidth-aware integer subtraction.

```mlir
%0 = hwarith.sub %a, %b : (ui3, ui4) -> si5
%1 = hwarith.sub %c, %d : (si3, si3) -> si4
```

Subtraction can produce negative results even when both inputs are unsigned, so its result is always signed. Its width calculation follows the addition width rule, then forces signedness on the result.

This is a useful distinction from plain bit-vector subtraction. In `hwarith`, `ui8 - ui8` does not produce `ui8`; it produces a signed result wide enough to represent the possible negative value.

### Multiplication

`hwarith.mul` is bitwidth-aware integer multiplication. It is binary and commutative.

```mlir
%0 = hwarith.mul %a, %b : (ui3, ui4) -> ui7
%1 = hwarith.mul %c, %d : (si3, si3) -> si6
%2 = hwarith.mul %e, %f : (si3, ui5) -> si8
```

The result width is `lhs_width + rhs_width`. If both operands are unsigned, the result is unsigned. If either operand is signed, the result is signed.

This makes multiplication predictable: the value range of a product is preserved by growing the bitwidth to the sum of operand widths.

### Division

`hwarith.div` is bitwidth-aware integer division.

```mlir
%0 = hwarith.div %a, %b : (ui3, ui4) -> ui3
%1 = hwarith.div %c, %d : (si3, si3) -> si4
%2 = hwarith.div %e, %f : (ui3, si4) -> si4
%3 = hwarith.div %g, %h : (si4, ui6) -> si4
```

The result width is based on the dividend. Unsigned divided by unsigned returns `ui<lhs_width>`. If the divisor is signed, the result width grows by one bit. The result signedness follows the rule that signed operands dominate: same signedness keeps that signedness, and mixed signedness produces signed output.

Lowering chooses `comb.divs` or `comb.divu`. Signed division may require both operands to be extended to a common width, and the result may then be truncated back to the expected `hwarith.div` result width.

### Cast

`hwarith.cast` is the boundary operation between sign-aware `hwarith` values and signless hardware bit vectors.

```mlir
%0 = hwarith.cast %a : (ui3) -> si5
%1 = hwarith.cast %b : (si3) -> si4
%2 = hwarith.cast %c : (si7) -> ui4
%3 = hwarith.cast %d : (i7) -> si5
%4 = hwarith.cast %e : (si14) -> i4
```

At least one side of the cast must be sign-aware. A signless-to-signless cast is rejected.

If the input is unsigned and the target is wider, lowering zero-extends. If the input is signed and the target is wider, lowering sign-extends. If the target is narrower, lowering truncates by extracting low bits. If the input is signless, the cast may only truncate or reinterpret into a same-width or narrower sign-aware type; widening a signless input is rejected because the compiler cannot know whether to sign-extend or zero-extend.

`hwarith.cast` has one canonicalization pattern, `EliminateCast`: a cast whose input and output have identical types is eliminated.

### Integer Comparison

`hwarith.icmp` compares two sign-aware integer operands and returns `i1`.

```mlir
%0 = hwarith.icmp eq %a, %b : ui2, si9
%1 = hwarith.icmp lt %c, %d : si3, ui6
```

The operands must be signed or unsigned `hwarith` integer values. The result is signless `i1` because it is a boolean. Before lowering, the operation determines a common comparison type that can represent both operand ranges. For same-signed operands, the comparison width is the max operand width. For mixed signedness, the comparison type is signed and may need an extra bit to safely represent the unsigned operand.

The predicates are `eq`, `ne`, `lt`, `ge`, `le`, and `gt`. Lowering maps ordered comparisons to signed or unsigned `comb.icmp` predicates according to the inferred comparison type.

## Transformations And Conversions

The local checkout defines one named `hwarith` conversion pass:

```text
lower-hwarith-to-hw
```

`lower-hwarith-to-hw` lowers HWArith to HW and Comb. The pass runs on `mlir::ModuleOp`, makes the `hwarith` dialect illegal, and applies a full dialect conversion. That means all `hwarith` operations must be removed, and other operations that still carry signedness-aware types must be converted.

The conversion has two parts.

First, it rewrites each `hwarith` operation:

- `hwarith.constant` becomes `hw.constant`.
- `hwarith.add` becomes `comb.add` after operands are sign- or zero-extended to the result width.
- `hwarith.sub` becomes `comb.sub` after operands are extended to the result width.
- `hwarith.mul` becomes `comb.mul` after operands are extended to the product width.
- `hwarith.div` becomes `comb.divs` or `comb.divu`, with extra extension and final extraction when required.
- `hwarith.cast` becomes either the original value, a zero/sign extension built from `hw.constant`, `comb.replicate`, and `comb.concat`, or a truncation built from `comb.extract`.
- `hwarith.icmp` becomes `comb.icmp`, with operands first extended to the inferred comparison type.

Second, it removes signedness from surrounding types. The type converter maps `uiN` and `siN` to signless `iN`, and recursively removes signedness from `hw.array`, `hw.unpacked_array`, `hw.struct`, `hw.union`, `hw.inout`, and `hw.typealias` element types. Generic type-conversion patterns rebuild other operations with converted operand and result types when they are otherwise legal.

The pass depends on the `hw`, `comb`, and `sv` dialects because the lowering may introduce `hw.constant`, `comb.extract`, `comb.concat`, `comb.replicate`, and other signless hardware operations while preserving surrounding SystemVerilog-facing constructs.

## How To Read HWArith IR

Start by looking at every `uiN` and `siN` type. The signedness is not decorative. It controls both result inference and lowering.

In this example:

```mlir
%0 = hwarith.add %a, %b : (ui8, si8) -> si10
```

the result is signed because one operand is signed, and it is 10 bits wide because the unsigned 8-bit operand can represent values up to 255, which need 9 bits to fit in a signed type before addition overflow is considered.

For subtraction:

```mlir
%1 = hwarith.sub %a, %b : (ui8, ui8) -> si9
```

the result is signed even though both inputs are unsigned, because `0 - 255` is negative.

For casts, read the source type first:

```mlir
%2 = hwarith.cast %x : (si5) -> ui8
```

This widens from 5 to 8 bits using sign extension, then treats the result as unsigned. The target signedness does not decide the extension behavior; the source signedness does.

For comparisons:

```mlir
%3 = hwarith.icmp lt %a, %b : ui5, si7
```

the operation first finds a common signed comparison type that can represent both inputs, then lowers to a signed `comb.icmp slt`.

## What It Implies

Seeing `hwarith.constant` means the constant still carries signedness semantics. After lowering, it should be a signless `hw.constant`.

Seeing `hwarith.add`, `hwarith.sub`, `hwarith.mul`, or `hwarith.div` means overflow-preserving result width has not yet been lowered into explicit bit padding and core combinational operations.

Seeing `hwarith.cast` means the IR is crossing between sign-aware arithmetic and signless hardware bit vectors, or adjusting width while preserving signedness intent.

Seeing `hwarith.icmp` means signedness-sensitive comparison has not yet been reduced to a concrete signed or unsigned `comb.icmp` predicate.

Seeing `uiN` or `siN` types outside `hwarith` after `lower-hwarith-to-hw` would indicate that conversion did not fully remove signedness-aware types from the lowered hardware IR.

The main beginner lesson is that `hwarith` is about making arithmetic intent explicit. It is not the final RTL representation. Its job is to preserve frontend arithmetic semantics long enough for CIRCT to lower them into plain signless hardware operations with the correct extensions, truncations, and comparison predicates already inserted.
