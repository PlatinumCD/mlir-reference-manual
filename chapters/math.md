# `math` Dialect

## Beginner Summary

The `math` dialect contains mathematical operations that are more specialized
than ordinary arithmetic.

The `arith` dialect handles basic arithmetic such as add, subtract, multiply,
divide, compare, and casts. The `math` dialect handles operations such as:

- Trigonometric functions like `math.sin`, `math.cos`, and `math.atan2`.
- Exponentials and logarithms like `math.exp`, `math.log`, and `math.powf`.
- Rounding operations like `math.ceil`, `math.floor`, and `math.roundeven`.
- Fused and compound operations like `math.fma`, `math.sincos`, and
  `math.clampf`.
- Floating-point classification like `math.isnan` and `math.isfinite`.
- Integer bit operations like `math.ctlz`, `math.cttz`, and `math.ctpop`.

Most operations work on scalars, vectors, and tensors. Vector and tensor forms
are elementwise unless an operation says otherwise.

Think of `math` as MLIR's portable vocabulary for elementary math functions
before the compiler has chosen a final implementation strategy.

## Why This Dialect Exists

Elementary math is deceptively target-dependent.

The expression:

```text
sin(x)
```

could lower to:

- An LLVM intrinsic.
- A `libm` call.
- A CUDA libdevice call.
- An AMD OCML or ROCDL call.
- A SPIR-V GLSL or OpenCL extended instruction.
- An EmitC `call_opaque`.
- A compiler-generated software helper function.
- A polynomial approximation.

Those choices depend on the target, precision requirements, available runtime
libraries, vector support, fast-math flags, and whether the operation appears
on scalar, vector, or tensor types.

The `math` dialect keeps the source-level mathematical intent visible while
the compiler is still making those choices.

It also separates ordinary arithmetic from math-library semantics. For example,
`arith.mulf` is a multiplication operation. `math.fma` is a fused
multiply-add. `math.exp` is an exponential function. `math.powf` carries the
semantics of floating-point exponentiation, not merely a sequence of multiplies.

## When It Matters

The `math` dialect matters when a compiler needs to preserve a mathematical
operation long enough to choose the right target implementation.

It commonly appears in pipelines for:

- Numerical kernels.
- ML model lowering.
- GPU code generation.
- Vectorized CPU code generation.
- Scientific and signal-processing code.
- Low-precision floating-point legalization.
- Runtime-library lowering.
- Fast-math optimization.

A typical flow looks like this:

```text
higher-level tensor/vector/scalar IR
  -> math.sin, math.exp, math.fma, math.isfinite, ...
  -> canonicalize and math-specific transforms
  -> choose target lowering
  -> LLVM / libm / NVVM / ROCDL / SPIR-V / XeVM / EmitC / helper funcs
```

The key point is that `math` is often a middle-level dialect. It is specific
enough for optimization, but not yet tied to one runtime library or target ISA.

## When To Use It

Use the `math` dialect when your IR needs a named mathematical operation whose
semantics should remain visible.

Use it for:

- Elementary floating-point functions.
- Integer absolute value and bit-counting operations.
- Floating-point classification predicates.
- Fused multiply-add.
- Combined sine/cosine computation.
- Power operations.
- Rounding operations that are distinct from casts.
- Elementwise math on vectors and tensors.
- Lowering frontends that have source-level math functions.

Do not use `math` for basic arithmetic when `arith` directly expresses the
operation. Use `arith.addf`, `arith.mulf`, `arith.divf`, `arith.addi`, and
similar operations for ordinary arithmetic. Use `math` when the operation has a
library-like, intrinsic-like, or special semantic meaning.

## Core Concepts

### Elementwise Semantics

Math operations can apply to:

- Scalar values such as `f32`, `f64`, or `i32`.
- Vector values such as `vector<4xf32>`.
- Tensor values such as `tensor<?xf32>`.

For vectors and tensors, operations are elementwise unless explicitly stated
otherwise.

```mlir
%s = math.sin %x : f32
%v = math.sin %vec : vector<4xf32>
%t = math.sin %tensor : tensor<?xf32>
```

The operation is the same mathematical function. The container type controls
how many elements are mapped.

### Math Versus Arith

The `arith` dialect owns primitive arithmetic. The `math` dialect owns
elementary math functions and related operations.

Examples:

- Use `arith.addf` for `x + y`.
- Use `arith.mulf` for `x * y`.
- Use `math.fma` when the operation must be fused as `x * y + z`.
- Use `math.exp` for the exponential function.
- Use `math.sqrt` for square root.
- Use `math.ctlz` for count-leading-zeros.

This split helps the compiler keep simple arithmetic simple while still
tracking operations that may require target intrinsics or runtime calls.

### Fast-Math Flags

Many floating-point Math operations implement MLIR's fast-math interface.

Fast-math flags communicate which IEEE-style constraints a compiler may relax.
For example, an operation may carry `fastmath<contract>` or
`fastmath<afn>`.

These flags matter for transformations and target lowering:

- `math-uplift-to-fma` requires fast-math flags that allow contraction.
- `convert-math-to-xevm` lowers selected operations to native OpenCL math
  intrinsics only when `afn` is present, unless configured otherwise.
- LLVM lowering carries fast-math information into LLVM operations and
  intrinsics.

Fast-math flags are not decoration. They are part of the optimization contract.

### Rounding Mode

Most Math operations do not carry an explicit rounding mode.

The dialect documentation says the default rounding mode used internally for
constant folding and canonicalization is round-to-nearest, ties-to-even. Runtime
behavior without an explicit rounding mode is deferred to the backend.

`math.fma` is special: it can carry an optional explicit IEEE-754 rounding mode.
When no rounding mode is present, LLVM lowering uses the ordinary LLVM FMA
intrinsic. When a rounding mode is present, LLVM lowering uses the constrained
FMA intrinsic.

### Low-Precision Floating Point

Targets often lack direct support for math library functions on types smaller
than `f32`, such as `f16`, `bf16`, or 8-bit float types.

The `math-extend-to-supported-types` pass handles this by extending unsupported
inputs to a supported type, applying the math operation there, and truncating
the result back to the original type. By default, `f32` and `f64` are treated
as supported.

`math.fma` is intentionally excluded from this pass because low-precision FMA
is often directly supported by hardware.

### Approximation And Target Choice

Some lowering paths preserve precise library semantics. Others intentionally
choose approximate or native target functions.

The `math` dialect itself records the mathematical operation. Later passes
decide whether to lower to a precise function, an approximate implementation,
or a target-native instruction. Fast-math flags and pass options are often what
make that decision legal.

## Operations

### Floating-Point Unary Functions

These operations take one floating-point-like operand and return the same type:

- `math.absf`: floating-point absolute value.
- `math.acos`: inverse cosine.
- `math.acosh`: inverse hyperbolic cosine.
- `math.asin`: inverse sine.
- `math.asinh`: inverse hyperbolic sine.
- `math.atan`: inverse tangent.
- `math.atanh`: inverse hyperbolic tangent.
- `math.cbrt`: cube root.
- `math.ceil`: ceiling.
- `math.cos`: cosine.
- `math.cosh`: hyperbolic cosine.
- `math.erf`: error function.
- `math.erfc`: complementary error function.
- `math.exp`: base-e exponential.
- `math.exp2`: base-2 exponential.
- `math.expm1`: `exp(x) - 1`, represented directly for accuracy.
- `math.floor`: floor.
- `math.log`: natural logarithm.
- `math.log10`: base-10 logarithm.
- `math.log1p`: `log(1 + x)`, represented directly for accuracy.
- `math.log2`: base-2 logarithm.
- `math.round`: round halfway cases away from zero.
- `math.roundeven`: round halfway cases to even.
- `math.rsqrt`: reciprocal square root.
- `math.sin`: sine.
- `math.sinh`: hyperbolic sine.
- `math.sqrt`: square root.
- `math.tan`: tangent.
- `math.tanh`: hyperbolic tangent.
- `math.trunc`: round toward zero in floating-point format.

Most of these operations can fold constants and carry fast-math flags.

### Floating-Point Binary And Ternary Functions

`math.atan2` computes the two-argument arctangent.

`math.copysign` returns the magnitude of the first operand with the sign of the
second operand.

`math.powf` raises a floating-point base to a floating-point exponent.

`math.fpowi` raises a floating-point base to a signed integer exponent. The
base and result have floating-point type; the exponent has integer or index-like
type with the same scalar/vector/tensor shape.

`math.fma` computes fused multiply-add. It takes three floating-point operands
and returns one floating-point result of the same type. It may carry an
explicit rounding mode.

`math.clampf` clamps a value to a floating-point range:

```text
clampf(value, min, max) = maxf(minf(value, max), min)
```

If `min > max`, the result is poison.

### Combined Sine And Cosine

`math.sincos` computes sine and cosine together and returns two results.

```mlir
%sin, %cos = math.sincos %x : f32
```

This can be more efficient than computing `math.sin` and `math.cos`
separately when both values are needed. The `math-sincos-fusion` pass can fuse
matching `math.sin` and `math.cos` operations into this operation.

### Integer Math

`math.absi` computes integer absolute value.

`math.ipowi` raises a signed integer base to an integer power.

`math.ctlz` counts leading zero bits.

`math.cttz` counts trailing zero bits.

`math.ctpop` counts set bits.

These operations work on integer scalars, vectors, or tensors.

### Floating-Point Classification

These operations take a floating-point-like operand and return an `i1`-like
result with the same shape:

- `math.isfinite`: true for normal, subnormal, or zero values; false for
  infinities and NaNs.
- `math.isinf`: true for positive or negative infinity.
- `math.isnan`: true for NaN.
- `math.isnormal`: true for normal finite values, excluding zero and subnormal
  values.

For example, a `vector<4xf32>` input produces a `vector<4xi1>` result.

## Transformations

### Canonicalization And Folding

Many Math operations fold constants and simplify obvious identities.

Canonicalization can simplify operations such as:

- Constant `math.powf`, `math.exp2`, `math.expm1`, and `math.rsqrt`.
- `math.sincos` of constants.
- Floating-point classification on constant values.
- Integer bit-counting on constant values.

The dialect also provides algebraic simplification patterns in the source tree.
These can rewrite selected math expressions when the required fast-math
conditions are met. For example, powers with special exponents can simplify to
more direct operations.

### `math-uplift-to-fma`

`math-uplift-to-fma` rewrites a multiply-add pattern into `math.fma` when
fast-math flags allow contraction.

This matters because a fused multiply-add has different semantics from a
separate multiply followed by add. The pass is conservative and uses fast-math
information to decide when that change is legal.

### `math-extend-to-supported-types`

`math-extend-to-supported-types` legalizes math operations on low-precision
floating-point types.

It inserts `arith.extf` before the math operation and `arith.truncf` after it.
The pass has options for:

- `target-type`, defaulting to `f32`.
- `extra-types`, for target-supported types beyond the implicit `f32` and
  `f64`.

This pass does not rewrite `math.fma`.

### `math-expand-ops`

`math-expand-ops` expands selected Math operations into more fundamental
operations.

The local implementation registers expansions for:

- `math.ctlz`
- `math.sinh`
- `math.cosh`
- `math.tan`
- `math.tanh`
- `math.asinh`
- `math.acosh`
- `math.atanh`
- `math.fma`
- `math.ceil`
- `math.exp2`
- `math.powf`
- `math.fpowi`
- `math.round`
- `math.roundeven`
- `math.rsqrt`
- `math.clampf`

The pass accepts an `ops` option so a pipeline can request only selected
expansions, such as `ops=tanh,tan`.

### `math-sincos-fusion`

`math-sincos-fusion` fuses matching `math.sin` and `math.cos` operations into
`math.sincos`.

This is useful when the target has a combined sin/cos instruction or library
call, or when computing both together is cheaper than two separate calls.

### Polynomial Approximation Patterns

The Math transform library also contains polynomial approximation patterns used
by tests and target-specific pipelines.

These approximate selected `f32` math functions using combinations of
arithmetic and `math.fma`. They are useful when a compiler wants an inline
approximation instead of a library call, but they are a deliberate numerical
choice and should not be treated as the default lowering for every target.

## Conversions And Lowering Paths

### `convert-math-to-llvm`

`convert-math-to-llvm` lowers supported Math operations to LLVM dialect
operations and intrinsics.

The local conversion includes direct LLVM lowering for operations such as
`math.absf`, `math.absi`, `math.ceil`, `math.copysign`, `math.cos`,
`math.exp`, `math.exp2`, `math.floor`, `math.fma`, `math.fpowi`,
`math.log`, `math.log2`, `math.log10`, `math.powf`, `math.round`,
`math.roundeven`, `math.sin`, `math.sqrt`, `math.tan`, `math.tanh`,
`math.trunc`, `math.ctlz`, `math.cttz`, `math.ctpop`, and selected
classification operations.

Some operations lower by expansion during LLVM conversion. For example:

- `math.expm1` lowers as `exp(x) - 1`.
- `math.log1p` may lower as `log(1 + x)` when `approximate-log1p` is enabled.
- `math.rsqrt` lowers as `1 / sqrt(x)`.
- `math.sincos` lowers to an LLVM sincos intrinsic followed by value
  extraction.

### `convert-math-to-libm`

`convert-math-to-libm` lowers supported floating-point Math operations to
`libm` function calls.

Examples include:

- `math.sin` to `sinf` or `sin`.
- `math.exp` to `expf` or `exp`.
- `math.log` to `logf` or `log`.
- `math.powf` to `powf` or `pow`.
- `math.sqrt` to `sqrtf` or `sqrt`.
- `math.fma` to `fmaf` or `fma`.

This path is appropriate for CPU-style lowering when the final program can
link against the math library.

### `convert-math-to-apfloat`

`convert-math-to-apfloat` lowers supported Math operations to APFloat-based
runtime library calls.

APFloat is a software floating-point implementation. This path is useful when
the target needs software emulation instead of native hardware or ordinary
`libm` behavior.

### `convert-math-to-funcs`

`convert-math-to-funcs` lowers supported Math operations to calls to
compiler-generated helper functions.

In this checkout, the pass is especially relevant for `math.fpowi` and
optionally `math.ctlz`. It has options controlling:

- The minimum exponent integer width for converting `math.fpowi`.
- Whether to convert `math.ctlz` to a software implementation.

The generated functions use ordinary MLIR control flow and arithmetic, with
LLVM linkage attributes used for generated helper linkage.

### GPU And Accelerator Lowerings

`convert-math-to-nvvm` lowers supported Math operations to CUDA libdevice
calls, and in some cases NVVM-specific behavior. It handles scalarization of
vector math where needed.

`convert-math-to-rocdl` lowers supported Math operations to AMDGPU/ROCDL
library calls, including OCML-style function names. It has a `chipset` option
for AMDGPU architecture-specific patterns.

`convert-math-to-spirv` lowers supported Math operations to SPIR-V operations
or extended instructions.

`convert-math-to-xevm` lowers supported operations to native XeVM/SPIR-V or
OpenCL-style math intrinsics. Its default behavior is precision-aware: native
OpenCL `native_` intrinsics are selected for supported Math ops marked with the
`afn` fast-math flag, unless the pass is configured to convert all supported
ops to OpenCL intrinsics.

### `convert-math-to-emitc`

`convert-math-to-emitc` lowers supported Math operations to EmitC
`call_opaque` operations targeting libc or libm functions.

This is different from `convert-math-to-funcs`: EmitC lowering preserves calls
that the C or C++ output can print as external library calls, with an option to
select the language target.

### Choosing A Lowering

There is no single correct Math lowering path.

Use:

- `convert-math-to-llvm` when LLVM intrinsics are the desired representation.
- `convert-math-to-libm` when linking against libm is appropriate.
- `convert-math-to-apfloat` when software floating-point emulation is needed.
- `convert-math-to-funcs` when compiler-generated helper implementations are
  preferred.
- `convert-math-to-nvvm` for CUDA/NVVM pipelines.
- `convert-math-to-rocdl` for AMDGPU/ROCDL pipelines.
- `convert-math-to-spirv` for SPIR-V pipelines.
- `convert-math-to-xevm` for XeVM/OpenCL-style native math lowering.
- `convert-math-to-emitc` for C/C++ emission through EmitC.

## Example IR

### Elementwise Scalar, Vector, And Tensor Math

```mlir
func.func @elementwise(%x: f32, %v: vector<4xf32>, %t: tensor<?xf32>)
    -> (f32, vector<4xf32>, tensor<?xf32>) {
  %s = math.exp %x : f32
  %vv = math.sin %v : vector<4xf32>
  %tt = math.sqrt %t : tensor<?xf32>
  return %s, %vv, %tt : f32, vector<4xf32>, tensor<?xf32>
}
```

The same dialect operation family works on scalar, vector, and tensor shapes.
The vector and tensor cases are elementwise.

### FMA, Sincos, Fast-Math, And Rounding

```mlir
func.func @fused(%a: f32, %b: f32, %c: f32) -> (f32, f32, f32) {
  %fma = math.fma %a, %b, %c to_nearest_even fastmath<contract> : f32
  %sin, %cos = math.sincos %a fastmath<contract> : f32
  return %fma, %sin, %cos : f32, f32, f32
}
```

This example shows two important target-lowering signals: a fused operation and
fast-math flags.

### Classification And Integer Bit Counting

```mlir
func.func @classify_and_count(%x: f32, %bits: i32) -> (i1, i1, i32) {
  %nan = math.isnan %x : f32
  %finite = math.isfinite %x : f32
  %leading = math.ctlz %bits : i32
  return %nan, %finite, %leading : i1, i1, i32
}
```

The `math` dialect is not only transcendental floating-point functions. It also
contains classification predicates and integer bit-counting operations.

### Low-Precision Math Before Legalization

```mlir
func.func @low_precision(%x: f16) -> f16 {
  %y = math.exp %x : f16
  return %y : f16
}
```

If the target does not support `exp` directly on `f16`, the
`math-extend-to-supported-types` pass can rewrite this through a wider
supported type, commonly `f32`.

## Mental Model

The Math dialect says what mathematical function is intended, not exactly how
the machine will compute it.

For beginners, group it this way:

- `math.sin`, `math.exp`, `math.log`, `math.sqrt`, and similar ops are
  portable elementary functions.
- `math.fma`, `math.sincos`, and `math.clampf` expose compound operations that
  targets often implement specially.
- `math.isnan`, `math.isfinite`, `math.isinf`, and `math.isnormal` answer
  questions about floating-point classes.
- `math.ctlz`, `math.cttz`, `math.ctpop`, `math.absi`, and `math.ipowi` cover
  integer math that is not ordinary arithmetic.
- Transform passes decide whether to keep, combine, expand, approximate, or
  legalize operations.
- Conversion passes decide which runtime, intrinsic set, or target dialect will
  implement them.

## Gotchas

Vector and tensor Math operations are elementwise. They do not imply a
reduction, map loop, or library call by themselves.

Do not assume every target supports every operation natively. Many Math ops
need lowering to runtime calls, expansions, or approximations.

Do not treat `math.fma` as syntactic sugar for `arith.mulf` plus `arith.addf`.
FMA is fused and has different numerical behavior.

Fast-math flags affect legality. A pass that is valid under
`fastmath<contract>` or `fastmath<afn>` may be invalid without those flags.

`math.expm1` and `math.log1p` exist for numerical accuracy. Expanding them to
`exp(x) - 1` or `log(1 + x)` can lose accuracy near zero unless the pipeline
has explicitly chosen that tradeoff.

`math-extend-to-supported-types` is a legalization pass, not a precision
improvement pass. It allows lowering through a supported type and then truncates
back to the original type.

The Math dialect depends on `arith`, but it is not a replacement for `arith`.
Use each dialect at the right semantic level.

## Source Map

Important local source files:

- `mlir/include/mlir/Dialect/Math/IR/MathBase.td` defines the dialect,
  elementwise intent, default rounding note, and `arith` dependency.
- `mlir/include/mlir/Dialect/Math/IR/MathOps.td` defines the Math operations.
- `mlir/lib/Dialect/Math/IR/MathOps.cpp` implements parsing, folding,
  canonicalization, and related operation behavior.
- `mlir/include/mlir/Dialect/Math/Transforms/Passes.td` declares
  `math-uplift-to-fma`, `math-extend-to-supported-types`, `math-expand-ops`,
  and `math-sincos-fusion`.
- `mlir/lib/Dialect/Math/Transforms/ExpandOps.cpp` implements Math expansion
  patterns.
- `mlir/lib/Dialect/Math/Transforms/ExtendToSupportedTypes.cpp` implements
  low-precision extension/truncation legalization.
- `mlir/lib/Dialect/Math/Transforms/UpliftToFMA.cpp` implements FMA uplift.
- `mlir/lib/Dialect/Math/Transforms/SincosFusion.cpp` implements sin/cos
  fusion.
- `mlir/lib/Dialect/Math/Transforms/PolynomialApproximation.cpp` implements
  polynomial approximation pattern utilities.
- `mlir/include/mlir/Conversion/Passes.td` declares the Math conversion passes.
- `mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp` implements Math-to-LLVM
  lowering.
- `mlir/lib/Conversion/MathToLibm/MathToLibm.cpp` implements Math-to-libm
  lowering.
- `mlir/lib/Conversion/ArithAndMathToAPFloat/MathToAPFloat.cpp` implements
  Math-to-APFloat runtime lowering.
- `mlir/lib/Conversion/MathToFuncs/MathToFuncs.cpp` implements lowering to
  generated helper functions.
- `mlir/lib/Conversion/MathToNVVM/MathToNVVM.cpp` implements Math-to-CUDA
  libdevice/NVVM lowering.
- `mlir/lib/Conversion/MathToROCDL/MathToROCDL.cpp` implements Math-to-ROCDL
  lowering.
- `mlir/lib/Conversion/MathToSPIRV/MathToSPIRV.cpp` implements Math-to-SPIR-V
  lowering.
- `mlir/lib/Conversion/MathToXeVM/MathToXeVM.cpp` implements Math-to-XeVM and
  OpenCL-style lowering.
- `mlir/lib/Conversion/MathToEmitC/MathToEmitC.cpp` implements Math-to-EmitC
  pattern lowering.
- `mlir/test/Dialect/Math/` contains parser, canonicalization, transform, and
  approximation tests.
- `mlir/test/Conversion/MathTo*/` contains conversion tests for the target
  lowering families.

Generated operation documentation for this checkout lists these 46 operations:

```text
math.absf
math.absi
math.acos
math.acosh
math.asin
math.asinh
math.atan
math.atan2
math.atanh
math.cbrt
math.ceil
math.clampf
math.copysign
math.cos
math.cosh
math.ctlz
math.ctpop
math.cttz
math.erf
math.erfc
math.exp
math.exp2
math.expm1
math.floor
math.fma
math.fpowi
math.ipowi
math.isfinite
math.isinf
math.isnan
math.isnormal
math.log
math.log10
math.log1p
math.log2
math.powf
math.round
math.roundeven
math.rsqrt
math.sin
math.sincos
math.sinh
math.sqrt
math.tan
math.tanh
math.trunc
```
