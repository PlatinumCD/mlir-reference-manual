# StableHLO CHLO Dialect

## Beginner Summary

The StableHLO `chlo` dialect is the Client HLO dialect.

CHLO exists for frontend ergonomics. It models operations that are convenient
for client libraries to emit, even when those operations are not the canonical
StableHLO form. The most common example is broadcasting. In StableHLO, binary
elementwise operations normally expect operands to already have compatible
shapes. In CHLO, operations such as `chlo.broadcast_add` and
`chlo.broadcast_multiply` can carry NumPy-style or explicit broadcast behavior
directly.

For a beginner, think of CHLO as a short-lived helper dialect:

```text
frontend or XLA-builder-like API
  -> chlo convenience operations
  -> chlo-legalize-to-stablehlo
  -> stablehlo + shape operations
```

CHLO is not meant to be the final portable model format. It is a friendly input
layer that lowers into StableHLO once broadcasting, special math functions, and
other client conveniences have been made explicit.

## Why This Dialect Exists

Frontend APIs often expose operations at a higher level than StableHLO wants to
store permanently.

For example, a frontend user might write:

```text
x + y
```

where `x` has shape `tensor<4x1xf32>` and `y` has shape `tensor<1x8xf32>`.
Client libraries commonly expect NumPy-like broadcasting to produce
`tensor<4x8xf32>`. StableHLO can represent that, but it usually wants the
broadcasts to be explicit.

The `chlo` dialect exists so frontends can emit the convenient operation first
and let a compiler pass materialize the exact StableHLO operations later.

It also captures special functions and API-level operations that are easier to
describe as one operation at the frontend boundary, such as `chlo.top_k`,
`chlo.ragged_dot`, `chlo.bessel_i1e`, `chlo.erf_inv`, `chlo.polygamma`, and
`chlo.zeta`.

## When It Matters

CHLO matters early in a StableHLO pipeline.

You will usually see it before the program is fully canonical StableHLO:

```text
frontend import
  -> chlo operations for convenient API semantics
  -> chlo-legalize-to-stablehlo
  -> stablehlo + shape
  -> StableHLO cleanup, compatibility, optimization, and target lowering
```

If you are debugging why a frontend expression did not lower as expected, CHLO
is often the place to check. It shows the operation the client intended before
the compiler expands it into broadcasts, comparisons, slices, loops, or math
approximations.

## When To Use It

Use `chlo` when building or importing frontend ML IR and you want to preserve
client API semantics briefly.

Use it for:

- binary operations with implicit or explicit broadcasting;
- frontend-level special math functions;
- constants shaped like another tensor;
- Top-K selection;
- ragged dot products;
- scan-style computations;
- compatibility with XLA Builder style APIs.

Do not use CHLO as a stable long-term program representation. Once the
convenience semantics are made explicit, lower to StableHLO.

Do not use CHLO for target-specific implementation details. Those belong later,
after StableHLO lowering chooses Linalg, TOSA, loops, buffers, or backend code.

## Core Concepts

### Client Convenience

The name "Client HLO" is literal. CHLO is aligned with client APIs. It models
what the frontend user or frontend library asked for, not necessarily the most
canonical compiler form.

This is why CHLO contains operations like `chlo.broadcast_select` and
`chlo.constant_like`. They are easier for client code to emit than a longer
sequence of shape queries, broadcasts, and StableHLO operations.

### Broadcasting

Broadcasting is the main CHLO idea.

Most `chlo.broadcast_*` operations take two operands and an optional
`broadcast_dimensions` attribute. If the operands already have the same shape,
the legalization can become a direct StableHLO binary operation. If they need
NumPy-style broadcasting, the legalization inserts `stablehlo.broadcast_in_dim`
or `stablehlo.dynamic_broadcast_in_dim` style logic before the final operation.

For dynamic shapes, CHLO legalization may also use the Shape dialect to assert
that operands are broadcastable.

### Special Function Decomposition

Several CHLO operations are mathematical functions that StableHLO does not
always keep as single primitive operations. The legalization pipeline expands
them into StableHLO arithmetic, comparisons, selects, constants, and helper
operations.

Examples include inverse trigonometric functions, hyperbolic functions, error
functions, Bessel approximations, `polygamma`, and `zeta`.

### Higher-Level Operations

CHLO also has operations that are more structural than elementwise math.

`chlo.top_k` lowers to StableHLO operations such as `iota`, `sort`, and
`slice`. `chlo.scan` has a dedicated conversion that builds a StableHLO
`while`-style loop with slicing and dynamic update operations. `chlo.ragged_dot`
has custom lowering paths for its different ragged-dimension modes.

The important lesson is that CHLO operations may lower to one StableHLO op, but
they may also lower to a substantial StableHLO subgraph.

## Types And Attributes

The inspected CHLO source does not define custom CHLO types. CHLO operations
mostly use ordinary MLIR tensor types.

CHLO does define attributes and enums used by specific operations:

| Attribute or enum | Used for |
| --- | --- |
| `comparison_direction` | Direction for `chlo.broadcast_compare`: `EQ`, `NE`, `GE`, `GT`, `LE`, or `LT`. |
| `comparison_type` | Comparison interpretation: `NOTYPE`, `FLOAT`, `TOTALORDER`, `SIGNED`, or `UNSIGNED`. |
| `precision` and precision config | Backend-specific precision hints for `chlo.ragged_dot`. |
| `#chlo.ragged_dot` | Dimension numbering for ragged dot: batching, contracting, ragged, and group dimensions. |
| `broadcast_dimensions` | Optional explicit broadcast mapping on broadcast binary operations. |

## Operation Inventory

The CHLO source currently defines 50 operations.

### Broadcast Binary And Ternary Operations

| Operation | Purpose |
| --- | --- |
| `chlo.broadcast_add` | Add with CHLO broadcasting semantics. |
| `chlo.broadcast_and` | Logical or bitwise and with broadcasting. |
| `chlo.broadcast_atan2` | Two-argument arctangent with broadcasting. |
| `chlo.broadcast_compare` | Compare with broadcasting and CHLO comparison attributes. |
| `chlo.broadcast_complex` | Build complex values from real and imaginary operands with broadcasting. |
| `chlo.broadcast_divide` | Divide with broadcasting. |
| `chlo.broadcast_maximum` | Maximum with broadcasting. |
| `chlo.broadcast_minimum` | Minimum with broadcasting. |
| `chlo.broadcast_multiply` | Multiply with broadcasting. |
| `chlo.broadcast_next_after` | Next representable floating-point value with broadcasting. |
| `chlo.broadcast_or` | Logical or bitwise or with broadcasting. |
| `chlo.broadcast_polygamma` | Polygamma with broadcasting. |
| `chlo.broadcast_power` | Power with broadcasting. |
| `chlo.broadcast_remainder` | Remainder with broadcasting. |
| `chlo.broadcast_select` | Select between two tensors using a predicate, with broadcasting. |
| `chlo.broadcast_shift_left` | Left shift with broadcasting. |
| `chlo.broadcast_shift_right_arithmetic` | Arithmetic right shift with broadcasting. |
| `chlo.broadcast_shift_right_logical` | Logical right shift with broadcasting. |
| `chlo.broadcast_subtract` | Subtract with broadcasting. |
| `chlo.broadcast_xor` | Logical or bitwise xor with broadcasting. |
| `chlo.broadcast_zeta` | Hurwitz zeta with broadcasting. |

### Unary Math And Predicates

| Operation | Purpose |
| --- | --- |
| `chlo._asin_acos_kernel` | Internal helper used by inverse trigonometric decompositions. |
| `chlo.acos` | Elementwise arccosine. |
| `chlo.acosh` | Elementwise inverse hyperbolic cosine. |
| `chlo.asin` | Elementwise arcsine. |
| `chlo.asinh` | Elementwise inverse hyperbolic sine. |
| `chlo.atan` | Elementwise arctangent. |
| `chlo.atanh` | Elementwise inverse hyperbolic tangent. |
| `chlo.bessel_i1e` | Exponentially scaled modified Bessel function of order one. |
| `chlo.conj` | Complex conjugate, or identity for non-complex values. |
| `chlo.cosh` | Elementwise hyperbolic cosine. |
| `chlo.digamma` | Elementwise digamma function. |
| `chlo.erf` | Elementwise error function. |
| `chlo.erf_inv` | Elementwise inverse error function. |
| `chlo.erfc` | Elementwise complementary error function. |
| `chlo.is_inf` | Test for positive or negative infinity. |
| `chlo.is_neg_inf` | Test for negative infinity. |
| `chlo.is_pos_inf` | Test for positive infinity. |
| `chlo.lgamma` | Elementwise log gamma function. |
| `chlo.sinh` | Elementwise hyperbolic sine. |
| `chlo.square` | Elementwise square, with complex-aware semantics. |
| `chlo.tan` | Elementwise tangent. |

### Non-Broadcast Binary Special Functions

| Operation | Purpose |
| --- | --- |
| `chlo.next_after` | Next representable floating-point value in the direction of another value. |
| `chlo.polygamma` | Polygamma function. |
| `chlo.zeta` | Hurwitz zeta function. |

### Constants And Higher-Level Operations

| Operation | Purpose |
| --- | --- |
| `chlo.constant` | CHLO constant, lowered directly to StableHLO constant. |
| `chlo.constant_like` | Create a splat constant with the same shape as another tensor. |
| `chlo.ragged_dot` | Matrix multiplication with one ragged dimension and optional group dimension. |
| `chlo.scan` | Apply a region along one dimension, producing scan outputs and carries. |
| `chlo.top_k` | Return the top `k` values and indices along the last dimension. |

## Transformations And Conversions

CHLO has one main named pass in the inspected StableHLO source:

| Pass | What it does |
| --- | --- |
| `chlo-legalize-to-stablehlo` | Lowers CHLO operations to StableHLO and Shape operations. |

The pass is backed by rewrite-pattern APIs:

| API or pattern family | What it handles |
| --- | --- |
| `populateChloToStablehloPatterns` | Registers the full CHLO-to-StableHLO legalization pattern set. |
| `populateChloConstantLikePattern` | Registers the `chlo.constant_like` lowering pattern. |
| Broadcasting patterns | Lower `chlo.broadcast_*` operations either directly to StableHLO binary ops or through explicit StableHLO broadcasting. |
| Generated CHLO decomposition patterns | Lower mathematical CHLO functions such as `asin`, `acos`, `acosh`, `asinh`, complex `atan`, complex `atanh`, and `square`. |
| Custom decomposition patterns | Lower functions such as `bessel_i1e`, `cosh`, `digamma`, `erf`, `erfc`, `erf_inv`, `lgamma`, `next_after`, `polygamma`, `sinh`, and `zeta`. |
| Structural conversion patterns | Lower `chlo.top_k`, `chlo.ragged_dot`, and `chlo.scan` using StableHLO subgraphs. |

Important examples:

- `chlo.broadcast_add` becomes explicit broadcasting plus
  `stablehlo.add`.
- `chlo.broadcast_compare` becomes explicit broadcasting plus
  `stablehlo.compare`, translating CHLO comparison attributes to StableHLO
  comparison attributes.
- `chlo.constant_like` becomes a `stablehlo.constant` for static shapes, or a
  constant plus dynamic broadcast for dynamic shapes.
- `chlo.top_k` becomes a sequence based on `stablehlo.iota`,
  `stablehlo.sort`, and `stablehlo.slice`.
- `chlo.scan` becomes a loop-like StableHLO form using `stablehlo.while`,
  slicing, and `stablehlo.dynamic_update_slice`.

There is no separate CHLO-to-Linalg or CHLO-to-TOSA conversion path in the
inspected source. The normal route is:

```text
chlo
  -> chlo-legalize-to-stablehlo
  -> stablehlo + shape
  -> stablehlo-legalize-to-linalg or stablehlo-legalize-to-tosa
```

## How To Read CHLO IR

When you see CHLO, ask two questions.

First: what convenience is being preserved? For `chlo.broadcast_*`, the answer
is usually implicit broadcasting. For `chlo.constant_like`, it is "make a
constant shaped like this operand." For `chlo.top_k`, it is a frontend API that
will expand into sorting and slicing.

Second: what StableHLO will this become? CHLO is useful because it delays
tedious expansion, but the compiler eventually needs explicit StableHLO.

Example:

```mlir
%0 = chlo.broadcast_add %lhs, %rhs
  : (tensor<4x1xf32>, tensor<1x8xf32>) -> tensor<4x8xf32>
```

This means "add these tensors using broadcasting." After legalization, the
program should contain explicit broadcasts and a `stablehlo.add`.

## What It Implies

Choosing CHLO implies that the program is still near the frontend boundary.

It also implies that some semantics are intentionally not expanded yet:

- broadcast rules may still be implicit;
- special functions may still be single operations;
- constants may depend on another tensor's shape;
- higher-level operations such as Top-K, scan, and ragged dot may still need
  custom expansion.

That is useful early, but it should not persist deep into a compiler pipeline.
Once CHLO is legalized, later passes can reason about ordinary StableHLO
operations, Shape constraints, and explicit tensor transformations.

## Source Files Inspected

This chapter was written from the local StableHLO source:

- `stablehlo/stablehlo/dialect/ChloOps.td`
- `stablehlo/stablehlo/dialect/ChloEnums.td`
- `stablehlo/docs/generated/chlo.md`
- `stablehlo/stablehlo/transforms/Passes.td`
- `stablehlo/stablehlo/transforms/Passes.h`
- `stablehlo/stablehlo/transforms/ChloLegalizeToStablehlo.cpp`
- `stablehlo/stablehlo/transforms/ChloDecompositionPatterns.td`
- `stablehlo/stablehlo/transforms/ChloDecompositionPatternsMath.td`
- `stablehlo/stablehlo/transforms/StablehloBroadcastLowering.h`
