# `quant` Dialect

## Beginner Summary

The `quant` dialect represents quantized values.

Quantization is the compiler technique of replacing high precision values,
usually floating-point values, with smaller storage values, usually integers or
compact floating-point encodings. A model might compute as if it has `f32`
values, but store some tensors as `i8`, `u8`, `i16`, `f8`, or a table-based
format. The compiler needs to know both views:

- the expressed value, such as `f32`
- the stored value, such as `i8`
- the scale and zero point that relate the two
- the axis or block where those parameters apply

`quant` is the dialect that carries that information in MLIR.

Unlike many dialects, `quant` is mostly type-centered. It has only three
operations:

- `quant.qcast`, which converts expressed values to quantized values
- `quant.dcast`, which converts quantized values back to expressed values
- `quant.scast`, which reinterprets quantized values as their storage type, or
  storage values as their quantized type

The main idea is simple:

```text
expressed_value = (stored_value - zero_point) * scale
```

The type tells the compiler what `stored_value`, `zero_point`, and `scale` mean.
The ops say where a value crosses between the expressed, quantized, and raw
storage views.

## Why This Dialect Exists

MLIR needs a way to represent quantized tensors before they become ordinary
integer tensors.

Without a quant dialect, a compiler would have to choose between two bad
representations:

- keep quantized values as floats and lose the storage layout that a real
  deployment target will use
- lower immediately to integers and lose the numerical meaning of the original
  model

`quant` keeps both pieces of information available. A tensor can have an element
type like this:

```mlir
!quant.uniform<i8:f32, 2.0:10>
```

That type says:

- storage is 8-bit signed integer
- the value being approximated is `f32`
- the scale is `2.0`
- the zero point is `10`

That is enough information for later passes to generate integer arithmetic,
runtime calls, target-specific instructions, or storage-only ABI boundaries.

The dialect is especially important for machine-learning compiler pipelines.
Quantized neural networks often store weights and activations in low precision
formats while preserving enough metadata to dequantize, requantize, validate,
or transform the computation.

## When It Matters

The `quant` dialect matters when a compiler pipeline needs to preserve
quantization semantics.

Common situations include:

- importing a quantized ML model
- representing quantized tensor types in TOSA or other model-level IR
- lowering fake-quantized training artifacts into deployment quantization
- changing a function ABI from quantized types to storage types
- expanding quantize and dequantize operations into arithmetic
- deciding whether a tensor is per-tensor, per-axis, or blockwise quantized
- preserving calibration ranges before final quantization choices are made
- representing lookup-table based compressed formats such as quantile encodings

It matters most before the final low-level code generation stage. Once the
compiler commits to ordinary integer or floating-point operations, the high
level quantization type usually disappears.

## When To Use It

Use `quant` when a value is not merely an integer, but an integer with numerical
meaning relative to some expressed type.

Good uses:

- a tensor of `i8` values that represents `f32` model activations
- a tensor with one scale per channel
- a blockwise-quantized tensor where each block has separate parameters
- a function boundary that still uses quantized tensor types
- an explicit point where the IR quantizes or dequantizes a value

Avoid using `quant` when:

- the integer is just an integer and does not approximate another type
- the scale and zero point are already irrelevant
- the target ABI requires plain integer tensors only
- the computation has already been lowered to arithmetic that no longer needs
  quantized element types

In a typical MLIR pipeline, `quant` is a middle-level representation. It is
high enough to preserve model semantics, but low enough that lowering passes can
turn casts into `arith`, `linalg`, `shape`, and `tensor` operations.

## Core Concepts

### Expressed Type And Storage Type

A quantized type always relates two views of a value.

The expressed type is the type the original computation thinks in. In ML
models, this is commonly `f32`, `f16`, `bf16`, or another floating-point type.

The storage type is the type used to store the quantized value. Common examples
are `i8`, `u8`, `i16`, `u16`, `i32`, and compact floating-point or quantile
storage types.

For example:

```mlir
!quant.uniform<i8:f32, 0.5>
```

This says that values are stored using `i8`, but they express `f32` values.

### Scale And Zero Point

Uniform affine quantization uses this relationship:

```text
expressed_value = (stored_value - zero_point) * scale
```

The scale controls the spacing between representable expressed values. The zero
point tells which storage value represents expressed zero.

For this type:

```mlir
!quant.uniform<i8:f32, 2.0:10>
```

the storage value `10` represents expressed `0.0`, storage value `11`
represents `2.0`, and storage value `9` represents `-2.0`.

If the zero point is omitted, it defaults to zero.

### Storage Bounds

Quantized types can include explicit storage bounds:

```mlir
!quant.uniform<i8<-5:10>:f32, 2.0>
```

The storage type is still `i8`, but the valid quantized range is restricted to
`-5` through `10`. When `lower-quant-ops` lowers a `quant.qcast`, it emits
clamping logic when these bounds are narrower than the default storage range.

### Per-Tensor Quantization

Per-tensor, also called per-layer, quantization uses one scale and zero point
for the whole value:

```mlir
!quant.uniform<i8:f32, 0.25:3>
```

Every element uses the same parameters.

This is the easiest form to understand and the easiest form to lower. For a
ranked tensor, the lowering can splat the scalar scale and zero point across
the tensor shape.

### Per-Axis Quantization

Per-axis, also called per-channel, quantization uses a different scale and zero
point for each slice along one tensor dimension:

```mlir
tensor<2x3x!quant.uniform<i8:f32:1, {0.5:0, 0.25:0, 0.125:0}>>
```

The `:1` says that dimension 1 is the quantized axis. Since dimension 1 has
size 3, the type provides three scale and zero-point entries.

This form is common for neural network weights, where each output channel may
have a different numerical range.

### Sub-Channel Quantization

Sub-channel quantization, also called blockwise quantization, generalizes
per-tensor and per-axis quantization. Instead of assigning parameters to a
whole tensor or a whole channel, it assigns parameters to blocks.

Example:

```mlir
tensor<2x4x!quant.uniform<u8:f32:{0:1, 1:2},
    {{2.0:120, 3.0:127}, {4.0, 5.0}}>>
```

The `{0:1, 1:2}` part says that axis 0 has block size 1 and axis 1 has block
size 2. The nested scale and zero-point list then gives parameters for the
resulting block grid.

This matters for more aggressive compression schemes, especially for weights.

### Calibrated Quantized Types

A calibrated type records an observed expressed range:

```mlir
!quant.calibrated<f32<-1.0:1.0>>
```

This type does not say exactly how to store the value. Instead, it records that
the value is an `f32` whose expected range is `[-1.0, 1.0]`.

Calibration is useful before the compiler has committed to a concrete storage
type, scale, and zero point.

### Any Quantized Types

`!quant.any` is a generic quantized type:

```mlir
!quant.any<i8:f32>
!quant.any<i8>
!quant.any<i8<-8:7>:f32>
```

It can describe storage and optional expressed type information without
committing to a uniform scale or zero point. It is useful as a flexible
stand-in in generic quantization workflows, but most lowering paths require
more specific types such as `!quant.uniform`.

### Quantile Types

`!quant.quantile` represents a lookup-table based quantized storage format:

```mlir
!quant.quantile<ui4:f16, {-1.0, -0.5, 0.0, 0.5, 1.0}, <0:4>>
```

Instead of describing values with a linear scale and zero point, a quantile type
uses a table of floating-point values. The stored value indexes the table.

This is useful for formats such as NF4-style weight compression.

### Casts Are Semantic Boundaries

The three operations in this dialect mark view changes:

```text
f32 value           -- quant.qcast -->  !quant.uniform value
!quant.uniform      -- quant.dcast -->  f32 value
!quant.uniform      -- quant.scast -->  raw storage integer
raw storage integer -- quant.scast -->  !quant.uniform value
```

`quant.scast` is intentionally a storage reinterpretation. It does not apply
the scale or zero point. `quant.qcast` and `quant.dcast` are the operations that
perform numerical conversion.

## Operations

The `quant` dialect defines three operations.

### Cast Operations

| Operation | Meaning |
| --- | --- |
| `quant.qcast` | Quantize cast. Converts a floating-point scalar or tensor to a scalar or tensor with `!quant.uniform` element type. |
| `quant.dcast` | Dequantize cast. Converts a scalar or tensor with `!quant.uniform` element type back to a floating-point scalar or tensor. |
| `quant.scast` | Storage cast. Reinterprets a quantized value as its signless integer storage type, or reinterprets storage values as the corresponding quantized type. |

### `quant.qcast`

`quant.qcast` converts expressed values to quantized values.

Conceptually, it does this:

```text
scaled = expressed_value / scale
stored_float = scaled + zero_point
stored_integer = convert_float_to_integer(stored_float)
stored_integer = clamp(stored_integer, storage_min, storage_max)
quantized_value = reinterpret_as_quantized_type(stored_integer)
```

The operation syntax is:

```mlir
%q = quant.qcast %x : f32 to !quant.uniform<i8:f32, 2.0:10>
```

The input must be floating-point. The result must be a quantized scalar or a
tensor with a quantized element type. For tensors, the input and result must
have the same shape.

### `quant.dcast`

`quant.dcast` converts quantized values back to expressed floating-point values.

Conceptually, it does this:

```text
stored_integer = reinterpret_as_storage_type(quantized_value)
stored_float = convert_integer_to_float(stored_integer)
expressed_value = (stored_float - zero_point) * scale
```

The operation syntax is:

```mlir
%y = quant.dcast %q : !quant.uniform<i8:f32, 2.0:10> to f32
```

The input must be quantized. The result must match the expressed type encoded
in the quantized type.

### `quant.scast`

`quant.scast` moves between the quantized type and the raw storage type.

Example:

```mlir
%q = quant.scast %storage : i8 to !quant.uniform<i8:f32, 2.0>
%i = quant.scast %q : !quant.uniform<i8:f32, 2.0> to i8
```

This is not quantization or dequantization. It is a bit-level or storage-level
view change. The width of the integer storage type must match the storage width
encoded in the quantized type.

`quant.scast` is often left behind by `lower-quant-ops` as the boundary between
quantized semantic values and plain storage values.

## Attributes And Types

`quant` is mostly about types. The operation inventory is small because the
type inventory carries most of the information.

### Type Families

| Type form | Meaning |
| --- | --- |
| `!quant.any<...>` | Generic quantized type with storage type, optional expressed type, and optional storage bounds. |
| `!quant.uniform<storage:expressed, scale[:zeroPoint]>` | Per-tensor uniform affine quantization. |
| `!quant.uniform<storage:expressed:axis, {scale[:zeroPoint], ...}>` | Per-axis uniform affine quantization. |
| `!quant.uniform<storage:expressed:{axis:block, ...}, {{...}}>` | Sub-channel or blockwise uniform affine quantization. |
| `!quant.calibrated<expressed<min:max>>` | Expressed type with observed calibration range. |
| `!quant.quantile<storage:float, {lut}, <min:max>>` | Lookup-table quantized storage format. |

### Storage Type Spelling

Quant storage can use integer storage spelling such as:

```text
i8, u8, i16, u16, i32, u32, si8, ui8
```

It can also use supported low-precision floating-point storage types and
quantile storage types when those implement MLIR's quant storage type
interface.

Storage bounds can narrow the valid storage range:

```mlir
!quant.uniform<i8<-8:7>:f32, 0.25>
```

### Uniform Type Variants

Per-tensor:

```mlir
!quant.uniform<i8:f32, 0.25:3>
```

Per-axis:

```mlir
tensor<2x3x!quant.uniform<i8:f32:1, {0.5:0, 0.25:0, 0.125:0}>>
```

Sub-channel:

```mlir
tensor<2x4x!quant.uniform<u8:f32:{0:1, 1:2},
    {{2.0:120, 3.0:127}, {4.0, 5.0}}>>
```

All three are printed as `!quant.uniform`, but they map to different internal
C++ type classes.

## Transformations

### `lower-quant-ops`

`lower-quant-ops` lowers `quant.qcast` and `quant.dcast` inside `func.func`.

It expands the numerical conversion into core dialect operations:

- `arith` for constants, integer/float casts, add, subtract, multiply, divide,
  min, and max
- `tensor` for shape-aware tensor construction and reshaping
- `shape` for unranked tensor shape handling
- `linalg` for per-axis and sub-channel elementwise lowering

The pass keeps `quant.scast` legal. That means it removes the high-level
numerical casts but still uses storage casts as explicit boundaries between
quantized values and their raw storage types.

For per-tensor quantization, the lowering can use scalar constants or tensor
splats.

For per-axis quantization, the pass materializes scale and zero-point tensors
and emits a `linalg.generic` that indexes the scale and zero point by the
channel dimension.

For sub-channel quantization, the pass materializes multi-dimensional parameter
tensors and emits a `linalg.generic` using affine maps that divide element
indices by block sizes.

### `normalize-quant-types`

`normalize-quant-types` rewrites sub-channel quantized tensor types to simpler
uniform variants when possible.

It performs two important simplifications:

- a sub-channel type with a single scale and zero point becomes a per-tensor
  uniform type
- a sub-channel type whose scale tensor has only one non-one dimension becomes
  a per-axis uniform type

This is useful because per-tensor and per-axis forms are easier and cheaper to
lower than general blockwise forms.

The pass uses dialect conversion. It rewrites function signatures and generic
operations so operands and results use the normalized types.

### `strip-func-quant-types`

`strip-func-quant-types` removes quantized types from function boundaries.

It rewrites function arguments, results, calls, and returns so function
signatures use storage types instead of quantized types. It inserts
`quant.scast` where needed inside the function body to preserve the original
semantic view.

Example intent:

```text
func @f(tensor<4x!quant.uniform<i8:f32, 0.5>>) -> tensor<4x!quant.uniform<i8:f32, 0.5>>
```

becomes a function boundary using:

```text
tensor<4xi8>
```

with `quant.scast` operations at the places where the function body still needs
the quantized type.

This is useful near ABI boundaries, where callers and callees must exchange
ordinary storage values rather than MLIR quantized element types.

## Conversions / Lowering Paths

### Quantized Types To Storage Types

`QuantizedType` provides helpers that convert between:

- expressed types and quantized types
- quantized types and storage types
- expressed types and storage types through a quantized type

For example, a type like:

```mlir
tensor<4x!quant.uniform<i8:f32, 1.0>>
```

can be viewed as:

```mlir
tensor<4xf32>
```

or as:

```mlir
tensor<4xi8>
```

depending on whether the compiler is reasoning about expressed values or raw
storage.

### Lowering `quant.qcast`

The lowering of `quant.qcast` depends on the type:

- per-tensor quantization lowers to scalar or tensor arithmetic
- per-axis quantization lowers through `linalg.generic` indexed by channel
- sub-channel quantization lowers through `linalg.generic` indexed by block

For signed storage, the pass uses signed float/integer conversions and signed
clamping. For unsigned storage, it uses unsigned conversions and unsigned
clamping.

### Lowering `quant.dcast`

`quant.dcast` begins by inserting a `quant.scast` from the quantized type to the
storage type. It then expands the dequantization arithmetic:

```text
stored_float = integer_to_float(storage)
adjusted = stored_float - zero_point
result = adjusted * scale
```

For a zero zero-point, the lowering skips the subtraction.

### Lowering Function Boundaries

`strip-func-quant-types` is the dedicated function-boundary lowering path.

It changes function signatures, calls, and returns to storage types while
inserting `quant.scast` materializations as needed. This pass does not lower
all quant operations to arithmetic. Its job is ABI cleanup.

### Relationship To Other Dialects

`quant` frequently appears next to ML dialects such as `tosa`, but it is not
limited to one source dialect. Its lowering passes depend on core MLIR dialects
because the final arithmetic is expressed with normal compiler IR:

```text
quant.qcast / quant.dcast
  -> arith + tensor + shape + linalg + quant.scast
  -> later tensor/linalg/arith lowerings
  -> target-specific IR
```

## Example IR

### Scalar Quantize And Dequantize

```mlir
!qalias = !quant.uniform<i8:f32, 2.0:10>

func.func @round_trip(%x : f32) -> f32 {
  %q = quant.qcast %x : f32 to !qalias
  %y = quant.dcast %q : !qalias to f32
  return %y : f32
}
```

`quant.qcast` converts `%x` from the expressed `f32` view to the quantized
storage-backed view. `quant.dcast` converts it back to `f32`.

### Per-Axis Tensor Quantization

```mlir
!qaxis = !quant.uniform<i8:f32:1, {0.5:0, 0.25:0, 0.125:0}>

func.func @per_axis(%x : tensor<2x3xf32>) -> tensor<2x3x!qaxis> {
  %q = quant.qcast %x : tensor<2x3xf32> to tensor<2x3x!qaxis>
  return %q : tensor<2x3x!qaxis>
}
```

The `:1` selects dimension 1 as the channel dimension. Since the tensor shape
is `2x3`, the channel dimension has size 3, so the type provides three scales.

### Storage Cast At A Boundary

```mlir
!qalias = !quant.uniform<i8:f32, 0.25>

func.func @storage_boundary(%storage : tensor<4xi8>) -> tensor<4x!qalias> {
  %q = quant.scast %storage : tensor<4xi8> to tensor<4x!qalias>
  return %q : tensor<4x!qalias>
}
```

This does not perform numerical dequantization. It says that the incoming
`tensor<4xi8>` should now be viewed as a quantized tensor with the metadata in
`!qalias`.

### Calibrated Type

```mlir
!cal = !quant.calibrated<f32<-1.0:1.0>>

func.func @calibrated_value(%x : !cal) {
  return
}
```

This records a floating-point range without yet choosing storage type, scale,
or zero point.

### Sub-Channel Tensor Quantization

```mlir
!qblock = !quant.uniform<u8:f32:{0:1, 1:2},
    {{2.0:120, 3.0:127}, {4.0, 5.0}}>

func.func @sub_channel(%x : tensor<2x4xf32>) -> tensor<2x4x!qblock> {
  %q = quant.qcast %x : tensor<2x4xf32> to tensor<2x4x!qblock>
  return %q : tensor<2x4x!qblock>
}
```

This assigns quantization parameters by block. Axis 0 uses block size 1 and
axis 1 uses block size 2.

## Mental Model

Think of `quant` as metadata plus crossing points.

The metadata is in the type:

```text
storage type
expressed type
scale
zero point
axis or block layout
optional storage bounds
```

The crossing points are the ops:

```text
quant.qcast: expressed -> quantized
quant.dcast: quantized -> expressed
quant.scast: quantized <-> storage
```

A beginner mistake is to treat a quantized tensor as just an integer tensor.
That loses the numerical contract. Another mistake is to treat it as just a
float tensor. That loses the storage contract.

`quant` exists so MLIR can keep both contracts until the pipeline is ready to
lower one or both away.

## Gotchas

- `quant.scast` is not `quant.dcast`. Storage casting does not multiply by
  scale or subtract zero point.
- `quant.qcast` and `quant.dcast` only operate on `!quant.uniform` types, not
  every quantized type family.
- Per-axis quantization must be wrapped in a tensor type. A scalar cannot have
  a channel axis.
- The number of per-axis scales must match the size of the selected channel
  dimension when that dimension is statically known.
- Sub-channel quantization can be normalized to simpler forms. Do not assume a
  blockwise type will remain blockwise for the whole pipeline.
- `lower-quant-ops` leaves `quant.scast` operations behind. This is expected.
- Rounding behavior is not fully specified by `quant.qcast` itself. The final
  behavior depends on the lower-level operations or target chosen by the
  pipeline.
- Storage bounds are part of the type. Narrow bounds can add clamp operations
  during lowering.
- Quantized types are usually not a final ABI. Use `strip-func-quant-types`
  when function boundaries need plain storage types.
- `!quant.calibrated` records ranges. It is not the same as a final uniform
  quantization decision.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/Quant/IR/QuantBase.td`
- `mlir/include/mlir/Dialect/Quant/IR/QuantOps.td`
- `mlir/include/mlir/Dialect/Quant/IR/QuantTypes.h`
- `mlir/include/mlir/Dialect/Quant/IR/QuantDialectBytecode.td`
- `mlir/lib/Dialect/Quant/IR/QuantOps.cpp`
- `mlir/lib/Dialect/Quant/IR/QuantTypes.cpp`
- `mlir/lib/Dialect/Quant/IR/TypeParser.cpp`
- `mlir/lib/Dialect/Quant/IR/QuantDialectBytecode.cpp`

Transform source files:

- `mlir/include/mlir/Dialect/Quant/Transforms/Passes.td`
- `mlir/lib/Dialect/Quant/Transforms/LowerQuantOps.cpp`
- `mlir/lib/Dialect/Quant/Transforms/NormalizeQuantTypes.cpp`
- `mlir/lib/Dialect/Quant/Transforms/StripFuncQuantTypes.cpp`

Utility source files:

- `mlir/include/mlir/Dialect/Quant/Utils/UniformSupport.h`
- `mlir/include/mlir/Dialect/Quant/Utils/FakeQuantSupport.h`
- `mlir/lib/Dialect/Quant/Utils/UniformSupport.cpp`
- `mlir/lib/Dialect/Quant/Utils/FakeQuantSupport.cpp`

Useful tests:

- `mlir/test/Dialect/Quant/ops.mlir`
- `mlir/test/Dialect/Quant/lower-quant-ops.mlir`
- `mlir/test/Dialect/Quant/normalize-quant-types.mlir`
- `mlir/test/Dialect/Quant/strip-func-quant-types.mlir`
- `mlir/test/Dialect/Quant/parse-any.mlir`
- `mlir/test/Dialect/Quant/parse-uniform.mlir`
- `mlir/test/Dialect/Quant/parse-calibrated.mlir`
- `mlir/test/Dialect/Quant/quantile-types.mlir`
