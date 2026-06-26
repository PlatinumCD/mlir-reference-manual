# CIRCT Datapath Dialect

The CIRCT `datapath` dialect models arithmetic datapath structures used during hardware synthesis. It is about how to build efficient arithmetic circuits, not just what arithmetic result should be computed.

For a beginner, the central idea is carry propagation. A normal addition eventually needs carries to ripple or propagate through bit positions. That can be expensive if a design repeatedly commits to carry-propagate adders too early. The `datapath` dialect exposes intermediate arithmetic structures such as partial products and compressor trees so the compiler can defer carry propagation and build better arithmetic hardware.

The local checkout defines three `datapath` operations: `datapath.compress`, `datapath.partial_product`, and `datapath.pos_partial_product`. The dialect has no custom types or attributes in this checkout. Its operations use hardware integer bitvectors from the surrounding CIRCT type system.

## When Datapath Is Important

`datapath` is important in synthesis-oriented flows, especially when arithmetic expressions should become efficient gate-level structures instead of naive chains of `comb.add` and `comb.mul`.

Use this dialect when you need to answer questions like:

- Has a multi-input addition been converted into a compressor tree?
- Has a multiplication been split into partial products?
- Is a result still in carry-save form, with final carry propagation delayed?
- Can adjacent arithmetic operations be fused into a larger datapath block?
- Is the compiler optimizing for shorter arithmetic delay, possibly at the cost of more area?
- Has datapath IR been lowered back to ordinary `comb` gates or to SMT constraints?

You usually do not write `datapath` by hand. You inspect it when a synthesis flow has converted ordinary combinational arithmetic into a more implementation-aware form.

## Why It Is Needed

The `comb` dialect can say "add these values" or "multiply these values." That is the right abstraction for logical hardware behavior, but it does not expose all implementation choices. A multiplier, for example, is often implemented in three conceptual stages:

1. Generate partial products.
2. Reduce those partial products with a compressor tree into a carry-save representation.
3. Use a final carry-propagate adder.

If the compiler lowers every multiply and add independently, it may insert carry-propagate adders between operations that could have stayed in carry-save form. The `datapath` dialect makes those intermediate forms explicit so optimizations can combine them.

The implication is that `datapath` operations are closer to generators or arithmetic implementation promises than to ordinary user-level math operations. They preserve enough structure for later lowering to choose compressor trees, AND arrays, Booth partial-product arrays, or SMT encodings.

## Types And Attributes

The dialect does not define custom types or attributes in this checkout.

All three operations use `HWIntegerType`, meaning hardware integer bitvectors such as `i8`, `i16`, or `i32`. The operations generally require operands and results to have the same type. That uniform width matters because compressor trees and partial product rows are represented as rows of same-width bitvectors.

## Operation Inventory

The local `datapath` dialect defines three operations.

### `datapath.compress`

`datapath.compress` reduces several same-width bitvectors to a smaller number of same-width bitvectors. The result is a redundant carry-save style representation: summing the results should equal summing the inputs, but the results have not yet been collapsed into one carry-propagated value.

Example:

```mlir
%r:2 = datapath.compress %a, %b, %c : i16 [3 -> 2]
```

Read this as "compress three 16-bit rows into two 16-bit rows." The verifier requires at least three inputs, requires the operation to reduce the number of rows, and requires at least two results. If there are only two inputs, an ordinary `comb.add` is the right operation.

This operation is the heart of the dialect. It lets the compiler say "these rows need to be summed eventually, but do not build the final carry-propagate adder yet."

### `datapath.partial_product`

`datapath.partial_product` generates partial product rows for multiplying two same-width operands. Summing all result rows gives the product of the two operands, truncated or represented at the operation's result width.

Example:

```mlir
%pp:4 = datapath.partial_product %a, %b
  : (i4, i4) -> (i4, i4, i4, i4)
```

Read this as "create four partial-product rows for a 4-bit multiply." The operation does not force one exact multiplier implementation. Later lowering may choose an AND array, an optimized square array, or a Booth-style array depending on size and pass options.

### `datapath.pos_partial_product`

`datapath.pos_partial_product` generates partial product rows for `(a + b) * c`. It is useful when the compiler can avoid first computing `a + b` with a full carry-propagate adder.

Example:

```mlir
%pp:3 = datapath.pos_partial_product %a, %b, %c
  : (i3, i3, i3) -> (i3, i3, i3)
```

Read this as "produce partial products for multiplying a carry-save-like sum by another operand." The implementation can encode `(a + b)` using carry and save terms and then build partial products from that representation.

## Canonicalization And Local Transformations

The dialect has several canonicalization patterns. These are not separate named passes, but they run when canonicalization or greedy rewriting asks the operations for their simplifications.

For `datapath.compress`, canonicalization can fold nested compressors into a larger compressor when all compressor results are being summed together. It can also fold a wide `comb.add` that already contains datapath values into a new compressor, fold constants, drop zero rows, and rewrite sign-extension or leading-one-extension patterns into compressor-friendly correction terms.

For `datapath.partial_product`, canonicalization can reduce the number of rows when known leading-zero bits prove that fewer partial products are needed. It can also rewrite selected signed-extension multiplication cases using Baugh-Wooley-style corrections. If one operand is a two-input `comb.add`, it can replace `datapath.partial_product` with `datapath.pos_partial_product`.

For `datapath.pos_partial_product`, canonicalization can reduce the number of generated rows when known bits prove that fewer rows are needed.

These patterns show the purpose of the dialect: it is not just a container for arithmetic nodes. It is a place where arithmetic structure can be reorganized before final gate-level lowering.

## Transformations And Conversions

The Datapath dialect has one dialect-owned transform pass and three important conversion passes.

`datapath-reduce-delay` rewrites arithmetic patterns to reduce delay, potentially increasing area. It can replicate logic to expose carry-save values. The local implementation includes patterns for nested adds, muxes whose arms contain adds, and unsigned comparisons over add expressions. For example, it can turn an add of nested adders into a `datapath.compress` followed by a two-input `comb.add`. It can also restructure a mux of additions into additions of muxes, making more operands available to a compressor.

`convert-comb-to-datapath` lowers selected `comb` arithmetic into `datapath`. A multi-input `comb.add` becomes `datapath.compress` plus a final two-input `comb.add`. A two-input `comb.mul` becomes `datapath.partial_product` followed by a sum of the partial product rows. The pass also uses the standard `comb` subtraction-to-add conversion, so subtraction can feed compressor-style arithmetic. Two-input `comb.add` operations remain legal because they represent the final carry-propagate adder.

`convert-datapath-to-comb` lowers `datapath` operations back to ordinary `comb` and `hw` operations. `datapath.compress` can lower either to a simple variadic `comb.add` plus zeros, if the `lower-compress-to-add` option is set, or to a concrete compressor tree built from gate-level `comb` operations. The compressor-tree path can use timing information when the `timing-aware` option is enabled. `datapath.partial_product` lowers to an AND array, a square-optimized AND array, or a Booth array. The `lower-partial-product-to-booth` option forces Booth lowering. `datapath.pos_partial_product` lowers through an AND-array-style construction for `(a + b) * c`.

`convert-datapath-to-smt` lowers Datapath operations into SMT constraints. It does not need to construct the exact gate implementation. Instead, it creates fresh SMT variables for the result rows and asserts the mathematical relationship: compressor results sum to compressor inputs, partial-product rows sum to `a * b`, and positive partial-product rows sum to `(a + b) * c`. This is useful for verification-oriented flows.

## What It Implies

Seeing `datapath.compress` means the IR is carrying a multi-row sum in a redundant representation. The rows still need to be reduced or finally added, but the compiler has not committed to one carry-propagate result yet.

Seeing `datapath.partial_product` means a multiplication has been split into rows. The next thing to look for is whether those rows feed `datapath.compress` or a wide `comb.add`.

Seeing `datapath.pos_partial_product` means the compiler has recognized a multiply by a sum and is trying to keep the addend side in a form that avoids early carry propagation.

Seeing `datapath-reduce-delay` in a pipeline means the compiler is willing to trade area for shorter delay. That pass may replicate logic that a pure area-oriented flow would prefer to keep shared.

Seeing `convert-datapath-to-comb` means Datapath is being erased into implementable hardware operations. Seeing `convert-datapath-to-smt` means the compiler is preserving Datapath semantics for proof rather than producing gates.

## How To Read Datapath IR

Start with row counts. In `datapath.compress`, the syntax `[N -> M]` says how many rows enter and how many rows leave. A `[5 -> 2]` compressor is reducing five rows to two carry-save rows.

Next, trace whether the rows are immediately summed by `comb.add`. A common pattern is:

```mlir
%c:2 = datapath.compress %a, %b, %d, %e : i8 [4 -> 2]
%sum = comb.add %c#0, %c#1 : i8
```

The `comb.add` is the final carry-propagate step. If more arithmetic can be folded into the compressor before that point, the compiler may still improve the datapath.

For multiplication, read `datapath.partial_product` as row generation and `datapath.compress` as row reduction. The final `comb.add` converts the carry-save result into the normal value.

## Minimal Example

This is the example from the Datapath rationale: multiply and add without committing to a carry-propagate adder between the two operations.

```mlir
%pp:4 = datapath.partial_product %a, %b
  : (i4, i4) -> (i4, i4, i4, i4)
%cs:2 = datapath.compress %pp#0, %pp#1, %pp#2, %pp#3, %c : i4 [5 -> 2]
%out = comb.add %cs#0, %cs#1 : i4
```

Read it as "generate partial products for `a * b`, include `c` in the same compressor tree, and only then perform the final add." The important optimization is that the compiler avoids computing `a * b` as a completed carry-propagated result before adding `c`.

