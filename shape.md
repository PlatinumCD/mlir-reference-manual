# Shape Dialect

## Beginner Summary

The `shape` dialect is MLIR's dialect for reasoning about tensor and memref
shapes as first-class IR values.

It gives the compiler explicit operations for:

- Asking for the shape, rank, or extent of a value.
- Building shapes from scalar extents.
- Computing shape algebra such as broadcast, concat, split, min, max, and
  number of elements.
- Representing unknown, invalid, or partially known shape information.
- Expressing shape constraints as compiler-visible witness values.
- Keeping shape computation separate from payload computation.

Think of `shape` as the dialect for the question "what will the shape of this
value be?" It is not a tensor compute dialect. It is a dialect for describing,
checking, simplifying, and lowering shape calculations.

## Why This Dialect Exists

Dynamic shapes are one of the hardest parts of a compiler IR.

Consider a tensor operation whose result shape depends on input dimensions:

```text
result shape = broadcast(shape_of(lhs), shape_of(rhs))
```

The compiler may need to know that result shape before it has lowered the
actual tensor computation. It may also need to prove that the broadcast is
valid, preserve a shape formula across dialect conversion, or lower the formula
to runtime code later.

The `shape` dialect provides a common language for that work. Without it, every
high-level dialect would need its own private representation for ranks,
dimensions, shape functions, dynamic shape checks, and invalid shape
propagation.

The dialect is also useful because it distinguishes several related ideas that
beginners often collapse into one:

- A `!shape.shape` can be unknown, partially known, or invalid.
- A `!shape.size` is a dimension-like scalar that can also be unknown or
  invalid.
- A `tensor<?xindex>` extent tensor is a concrete runtime list of extents and
  is assumed to be error-free.
- A `!shape.witness` is not data. It is a proof-like token that carries the
  fact that a constraint is satisfied.
- A `!shape.value_shape` couples a payload value with shape knowledge about
  that value.

Those distinctions let MLIR model shape reasoning before deciding how much of
it becomes runtime code.

## When It Matters

The `shape` dialect matters whenever a pipeline has dynamic dimensions and
needs shape information to stay explicit.

It commonly appears around:

- Shape inference for tensor-producing operations.
- Runtime checks for shape compatibility.
- Broadcasting and elementwise operation result-shape calculation.
- Reified result shapes from higher-level dialects.
- Dynamic rank or dynamic extent manipulation.
- Lowering shape calculations to `arith`, `tensor`, `scf`, `cf`, and asserts.
- Temporarily outlining shape computation so other passes can operate on the
  main payload IR.

A typical high-level flow looks like this:

```text
higher-level tensor dialect
  -> reify result shapes with shape.shape_of / shape.with_shape
  -> simplify and canonicalize shape computations
  -> optionally outline-shape-computation
  -> shape-to-shape-lowering
  -> convert-shape-to-std / convert-shape-constraints
  -> arith + tensor + scf + cf
```

The important point is that `shape` often lives between semantic tensor
dialects and executable control-flow or scalar dialects.

## When To Use It

Use the `shape` dialect when the IR needs to compute, compare, refine, or check
shapes explicitly.

Use it for:

- Querying the shape of a tensor or other shaped value.
- Computing extents, ranks, and element counts.
- Representing shape formulas before lowering them to loops or scalar code.
- Expressing broadcasting rules.
- Capturing runtime shape assertions.
- Associating computed shape information with a value.
- Writing shape functions that describe the result shape of another operation.

Do not use it as the primary representation for tensor element computation.
`shape` tells you about sizes and compatibility. Tensor contents still belong
in dialects such as `tensor`, `linalg`, `tosa`, `arith`, or target-specific
dialects.

Also do not keep `shape` too late in a pipeline if you are producing executable
code. Witnesses, constraints, and shape functions are compiler abstractions;
they must eventually fold away, become assertions, or lower to ordinary
runtime computations.

## Core Concepts

### Shape Values

`!shape.shape` represents a shape as a compiler value.

It can represent:

- An unranked shape.
- A ranked shape with known and unknown extents.
- A fully static shape.
- A scalar shape with rank zero.
- An invalid shape.

The Shape dialect documentation describes printed examples like:

```text
[*]
[?, 2]
[3, 4]
[]
[1]
[invalid]
```

The key beginner idea is that `!shape.shape` is richer than a runtime tensor of
extents. It can carry unknown and invalid states directly.

### Size Values

`!shape.size` is a non-negative dimension-like scalar.

It is similar to `index`, but it can also represent unknown and invalid shape
states. Shape arithmetic operations such as `shape.add`, `shape.mul`,
`shape.div`, `shape.min`, and `shape.max` understand that behavior.

If both operands are plain `index`, these operations can lower to ordinary
index arithmetic. If a `!shape.size` participates, error and unknown behavior
must be preserved.

### Extent Tensors

An extent tensor is a one-dimensional tensor of index values:

```mlir
tensor<?xindex>
```

It represents a runtime list of extents. Unlike `!shape.shape`, it is
guaranteed to be error-free.

Many Shape operations accept both `!shape.shape` and extent tensors. This lets a
pipeline move between abstract shape reasoning and concrete runtime extent
lists.

### Witness Values

`!shape.witness` is a proof-like token used to order shape assumptions and
constraints.

Constraint operations such as `shape.cstr_eq`, `shape.cstr_broadcastable`, and
`shape.cstr_require` produce witnesses. The `shape.assuming` family consumes
those witnesses and marks code that may rely on them.

A witness is not physical runtime data. It is a compiler representation of
"this constraint has been established." Later passes either prove it, erase it,
or lower it to side-effecting checks such as `cf.assert`.

### Value Shapes

`!shape.value_shape` conceptually pairs a value with shape information about
that value.

`shape.with_shape` creates or refines that pairing. `shape.value_of` extracts
the payload value. This is useful when a compiler has computed shape
information for an operation result and wants to keep that information attached
without immediately changing the payload computation.

### Unknown Versus Invalid

Unknown and invalid are different.

An unknown dimension means the compiler does not know the value yet. It may be
runtime-dependent and still valid.

An invalid shape means a shape computation found an impossible condition, such
as incompatible static dimensions. Most Shape operations that return shapes
propagate invalid operands to avoid producing many redundant diagnostics for
one root shape error.

### Shape Functions

`shape.func` describes a shape transfer function: a function whose job is to
compute a shape, not to compute tensor payload data.

`shape.function_library` groups shape functions and maps them to operations.
This gives a pipeline a way to preserve reusable shape semantics separately
from the main computation.

The `outline-shape-computation` pass uses this idea to pull shape calculations
out of the main IR and record how they relate to payload values.

## Operations

### Constants And Construction

`shape.const_shape` creates a constant shape or constant extent tensor.

```mlir
%s0 = shape.const_shape [] : !shape.shape
%s1 = shape.const_shape [1, 2, 3] : !shape.shape
%e0 = shape.const_shape [4, 5, 6] : tensor<3xindex>
```

`shape.const_size` creates a constant `!shape.size`.

`shape.const_witness` creates a statically known witness value, usually `true`
or `false`.

`shape.from_extents` builds a `!shape.shape` from SSA extent values.

```mlir
%shape = shape.from_extents %m, %n : index, index
```

`shape.from_extent_tensor` converts a one-dimensional `tensor<...xindex>` into
a `!shape.shape`.

`shape.to_extent_tensor` converts a shape-like value into a one-dimensional
index tensor. If the input is an invalid `!shape.shape`, the behavior is
undefined because extent tensors cannot carry the invalid state.

`shape.index_to_size` converts a builtin `index` to `!shape.size`.

`shape.size_to_index` converts `!shape.size` back to `index`. Its behavior is
undefined for unknown or invalid `!shape.size` values.

`shape.value_as_shape` interprets a value, usually a one-dimensional integer or
index tensor, as a shape.

### Query Operations

`shape.shape_of` returns the shape of a shaped value or a `!shape.value_shape`.
It may return either `!shape.shape` or an extent tensor, depending on the result
type requested.

`shape.dim` returns one extent from the shape of a shaped value. It is a
convenience form of `shape.get_extent(shape.shape_of(value), index)`.

`shape.get_extent` extracts one extent from a shape or extent tensor.

`shape.rank` returns the number of extents in a shape or extent tensor.

`shape.num_elements` returns the product of all extents.

### Shape Algebra

`shape.add`, `shape.mul`, and `shape.div` perform arithmetic on `index` and
`!shape.size` values while respecting shape error propagation rules.

`shape.min` and `shape.max` compute elementwise minimum or maximum on sizes or
same-rank shapes.

`shape.broadcast` computes the result shape of broadcasting two or more input
shapes. Smaller ranks are padded with leading ones, and each aligned dimension
must either be equal or one side must be one.

`shape.is_broadcastable` returns an `i1` predicate describing whether
`shape.broadcast` would succeed.

`shape.concat` concatenates two shapes.

`shape.split_at` splits one shape into a head and tail at an index. Negative
indices count from the back.

`shape.any` returns an arbitrary compatible shape from its inputs. It is useful
when multiple shapes are known to be compatible and the compiler only needs one
representative.

`shape.meet` computes the least general shape or size of its operands. It
refines unknown information when possible and produces invalid shape
information when operands contradict each other.

`shape.shape_eq` returns an `i1` indicating whether one or more shapes or
extent tensors are equal.

### Constraints And Assumptions

`shape.cstr_eq` creates a witness that all input shapes are equal.

`shape.cstr_broadcastable` creates a witness that input shapes can be
broadcasted.

`shape.cstr_require` creates a witness from an arbitrary `i1` predicate and an
error message.

`shape.assuming_all` combines multiple witnesses with logical AND.

`shape.assuming` executes a region under a witness. The region may use facts
guaranteed by that witness.

`shape.assuming_yield` terminates a `shape.assuming` region and yields its
results.

These operations are not intended to survive to final executable IR. They are a
structured way to say "this code depends on these shape facts."

### Shape Functions And Regions

`shape.func` defines a shape function.

`shape.return` returns values from a `shape.func`.

`shape.function_library` groups shape functions and maps them to the operations
whose shape behavior they describe.

`shape.reduce` iterates over the extents of a shape or extent tensor and
reduces them with a region.

`shape.yield` yields values from `shape.reduce` or `shape.function_library`.

### Value And Shape Pairing

`shape.with_shape` returns a `!shape.value_shape` by pairing an operand value
with explicit shape information.

`shape.value_of` extracts the payload value from a `!shape.value_shape`.

This pair is often used after a shape calculation has been reified. The payload
operation can stay in the original dialect while the shape formula is kept
available to the compiler.

### Debugging

`shape.debug_print` prints a shape or size and returns it. It is intended for
testing and debugging, not as a stable part of production lowering.

## Transformations

### Canonicalization And Folding

Many Shape operations have folders or canonicalizers.

Common simplifications include:

- Folding constant shapes and sizes.
- Simplifying known-true or known-false witnesses.
- Combining `shape.assuming_all` values.
- Removing redundant casts between shape and extent-tensor forms.
- Simplifying query operations on constant or statically shaped values.

Canonicalization is especially important for the Shape dialect because many
shape checks become trivial once constants, static ranks, or static dimensions
are visible.

### `outline-shape-computation`

`outline-shape-computation` is a module pass.

It finds shape computations introduced by shape reification, especially around
`shape.with_shape`, and outlines them into `shape.func` functions. The pass
also populates shape mapping analysis so the compiler knows which shape
function describes which payload value.

Use this when later passes should operate on the main payload IR without being
confused by inline shape computation, or when reifying shapes again after a
conversion would be difficult.

### `remove-shape-constraints`

`remove-shape-constraints` is a function pass.

It replaces `cstr_` operations with a true witness. This is useful in pipelines
that want to erase shape constraints rather than lower them to runtime checks.

This is a semantic choice. Removing constraints means the compiler is no
longer preserving those runtime assertions.

### `shape-to-shape-lowering`

`shape-to-shape-lowering` is a function pass.

It rewrites higher-level Shape dialect constructs into forms that are easier to
convert to ordinary scalar and control-flow dialects, especially Arith-related
lowering. It is a preparation pass for later conversion, not a final lowering
by itself.

## Conversions And Lowering Paths

### `convert-shape-to-std`

`convert-shape-to-std` lowers many Shape operations to standard MLIR dialects,
mainly `arith`, `tensor`, and `scf`.

Examples include:

- `shape.add` and `shape.mul` on plain `index` values lower to `arith.addi` and
  `arith.muli`.
- `shape.rank` on an extent tensor lowers to `tensor.dim`.
- `shape.get_extent` on an extent tensor lowers to `tensor.extract`.
- `shape.const_shape` lowers to `tensor.from_elements`.
- `shape.reduce` lowers to `scf.for`.
- `shape.shape_of` on tensors lowers to tensor-rank, tensor-dim, or constant
  extent construction when possible.
- `shape.broadcast`, `shape.is_broadcastable`, `shape.shape_eq`, and
  `shape.split_at` lower to combinations of `arith`, `tensor`, and `scf` when
  the operands are extent tensors.

The conversion cannot blindly lower every Shape operation in every type form.
For example, operations involving `!shape.shape` or `!shape.size` may remain
when their invalid or unknown behavior still matters.

### `convert-shape-constraints`

`convert-shape-constraints` lowers shape constraints to eager runtime checking.

For example:

- `shape.cstr_broadcastable` can become a broadcastability predicate plus
  `cf.assert`.
- `shape.cstr_eq` can become a shape equality predicate plus `cf.assert`.
- `shape.cstr_require` can become `cf.assert` directly.

This pass is separate from `convert-shape-to-std` because a pipeline may want
to lower computations and constraints at different times.

### Typical Lowering Strategy

A practical lowering sequence often looks like this:

```text
canonicalize shape computations
  -> optionally outline-shape-computation
  -> shape-to-shape-lowering
  -> convert-shape-constraints or remove-shape-constraints
  -> convert-shape-to-std
  -> lower arith/tensor/scf/cf further
```

The exact order depends on whether constraints should become runtime asserts or
be erased.

## Example IR

### Querying Rank And Extent

```mlir
func.func @rank_and_extent(%arg0: tensor<?x4xf32>, %i: index)
    -> (index, index) {
  %shape = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
  %rank = shape.rank %shape : tensor<2xindex> -> index
  %extent = shape.get_extent %shape, %i
      : tensor<2xindex>, index -> index
  return %rank, %extent : index, index
}
```

This example asks MLIR to materialize the tensor shape as an extent tensor,
then query the rank and one selected extent.

### Broadcasting Under A Witness

```mlir
func.func @broadcast(%a: tensor<2xindex>, %b: tensor<2xindex>)
    -> tensor<?xindex> {
  %ok = shape.cstr_broadcastable %a, %b
      : tensor<2xindex>, tensor<2xindex>
  %result = shape.assuming %ok -> tensor<?xindex> {
    %broadcast = shape.broadcast %a, %b
        : tensor<2xindex>, tensor<2xindex> -> tensor<?xindex>
    shape.assuming_yield %broadcast : tensor<?xindex>
  }
  return %result : tensor<?xindex>
}
```

The witness says the broadcast is valid. The region computes the broadcasted
shape under that assumption.

### Attaching A Computed Shape To A Value

```mlir
func.func @attach(%value: tensor<?xf32>, %n: index) -> tensor<?xf32> {
  %shape = shape.from_extents %n : index
  %vs = shape.with_shape %value, %shape : tensor<?xf32>, !shape.shape
  %out = shape.value_of %vs : tensor<?xf32>
  return %out : tensor<?xf32>
}
```

This does not compute tensor elements. It attaches the computed one-dimensional
shape `%shape` to `%value`, then extracts the original payload value again.

### Reducing Over Extents

```mlir
func.func @num_elements(%shape: tensor<?xindex>) -> index {
  %one = arith.constant 1 : index
  %count = shape.reduce(%shape, %one) : tensor<?xindex> -> index {
    ^bb0(%i: index, %extent: index, %acc: index):
      %next = arith.muli %acc, %extent : index
      shape.yield %next : index
  }
  return %count : index
}
```

This is the explicit form of multiplying all extents in a shape. The
`convert-shape-to-std` pass lowers this structure to an `scf.for` loop.

## Mental Model

The Shape dialect is a small symbolic system embedded in MLIR.

It lets the compiler say:

```text
Here is the shape formula.
Here are the constraints that make the formula valid.
Here is where payload IR depends on those shape facts.
Later, decide whether to fold, assert, erase, outline, or lower it.
```

For beginners, the most useful split is:

- `shape.shape_of`, `shape.rank`, `shape.get_extent`, and `shape.dim` ask
  questions about shapes.
- `shape.from_extents`, `shape.const_shape`, and `shape.to_extent_tensor`
  convert between shape representations.
- `shape.broadcast`, `shape.concat`, `shape.split_at`, `shape.meet`, and
  `shape.any` compute new shapes.
- `shape.cstr_*`, `shape.assuming*`, and `!shape.witness` express facts that
  must hold.
- `shape.with_shape`, `shape.value_of`, `shape.func`, and
  `shape.function_library` preserve shape knowledge across larger compiler
  transformations.

## Gotchas

`!shape.shape` and `tensor<?xindex>` are not the same thing. An extent tensor is
an ordinary runtime tensor of extents and cannot represent invalid shape
states. A `!shape.shape` can.

`!shape.size` and `index` are also not the same thing. `index` is a builtin
machine-sized integer-like type. `!shape.size` carries shape-specific unknown
and invalid behavior.

Witnesses are not booleans. A witness is a compiler token for an established
constraint. Use `shape.is_broadcastable` or `shape.shape_eq` when you need an
ordinary `i1` predicate.

`shape.assuming` is not a runtime `if`. It represents code that may assume a
witness is true. Later passes decide whether the associated constraint becomes
an assertion, folds away, or is removed.

`shape.any` is intentionally under-specified when inputs disagree. It should be
used only when prior reasoning makes the candidate shapes compatible enough for
the pipeline's purposes.

`shape.value_of` has undefined behavior for unknown or invalid
`!shape.value_shape` operands. It is not a dynamic verifier by itself.

`remove-shape-constraints` erases checks by replacing constraints with true
witnesses. Use `convert-shape-constraints` instead when runtime assertions must
be preserved.

## Source Map

Important local source files:

- `mlir/include/mlir/Dialect/Shape/IR/ShapeBase.td` defines the dialect and
  core types: `!shape.shape`, `!shape.size`, `!shape.value_shape`, extent
  tensors, and `!shape.witness`.
- `mlir/include/mlir/Dialect/Shape/IR/ShapeOps.td` defines the Shape dialect
  operations.
- `mlir/lib/Dialect/Shape/IR/Shape.cpp` implements parsing, verification,
  folding, canonicalization hooks, and operation behavior.
- `mlir/lib/Dialect/Shape/IR/ShapeCanonicalization.td` contains declarative
  canonicalization patterns.
- `mlir/include/mlir/Dialect/Shape/Transforms/Passes.td` declares
  `outline-shape-computation`, `remove-shape-constraints`, and
  `shape-to-shape-lowering`.
- `mlir/lib/Dialect/Shape/Transforms/OutlineShapeComputation.cpp` implements
  shape-computation outlining.
- `mlir/lib/Dialect/Shape/Transforms/RemoveShapeConstraints.cpp` implements
  constraint removal.
- `mlir/lib/Dialect/Shape/Transforms/ShapeToShapeLowering.cpp` implements
  Shape-to-Shape preparation for later lowering.
- `mlir/include/mlir/Conversion/Passes.td` declares `convert-shape-to-std` and
  `convert-shape-constraints`.
- `mlir/lib/Conversion/ShapeToStandard/ShapeToStandard.cpp` implements general
  Shape-to-standard lowering.
- `mlir/lib/Conversion/ShapeToStandard/ConvertShapeConstraints.cpp` implements
  shape constraint lowering to assertions.
- `mlir/test/Dialect/Shape/` contains parser, verifier, canonicalization, and
  transform tests for the dialect.
- `mlir/test/Conversion/ShapeToStandard/` contains conversion tests.

Generated operation documentation for this checkout lists these 40 operations:

```text
shape.add
shape.any
shape.assuming
shape.assuming_all
shape.assuming_yield
shape.broadcast
shape.concat
shape.const_shape
shape.const_size
shape.const_witness
shape.cstr_broadcastable
shape.cstr_eq
shape.cstr_require
shape.debug_print
shape.dim
shape.div
shape.from_extent_tensor
shape.from_extents
shape.func
shape.function_library
shape.get_extent
shape.index_to_size
shape.is_broadcastable
shape.max
shape.meet
shape.min
shape.mul
shape.num_elements
shape.rank
shape.reduce
shape.return
shape.shape_eq
shape.shape_of
shape.size_to_index
shape.split_at
shape.to_extent_tensor
shape.value_as_shape
shape.value_of
shape.with_shape
shape.yield
```
