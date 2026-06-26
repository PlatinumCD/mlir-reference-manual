# `complex` Dialect

## Beginner Summary

The `complex` dialect contains operations for creating, inspecting, computing
with, and lowering complex numbers.

In MLIR, the type syntax for a complex number is the built-in type
`complex<T>`, for example:

```mlir
complex<f32>
complex<f64>
```

The `complex` dialect is the operation dialect that makes those values useful:
it has ops like `complex.create`, `complex.re`, `complex.im`, `complex.add`,
`complex.mul`, `complex.div`, `complex.exp`, `complex.log`, and
`complex.sqrt`.

The important beginner idea is that complex arithmetic is not primitive on
most final targets. It usually needs to be expanded into real arithmetic,
math-library calls, LLVM struct operations, or target-specific calls.

## Why This Dialect Exists

Complex numbers appear in numerical computing, signal processing, FFTs, quantum
simulation, scientific computing, and some ML workloads. Source languages and
libraries often treat complex numbers as first-class values, but lower-level
IRs and hardware usually treat them as pairs of real values.

The `complex` dialect gives MLIR a middle level where complex values are still
explicit and understandable. A compiler can optimize or reason about complex
operations before choosing a final representation.

Without this dialect, a frontend would usually lower complex values immediately
to pairs, structs, or library calls. That loses intent early and makes it harder
to choose the right lowering path later.

## When It Matters

The `complex` dialect matters when a compiler needs to preserve complex-number
semantics across part of the MLIR pipeline.

It is especially useful for:

- mathematical source languages with complex scalar types;
- tensor programs whose element type is complex;
- frontend lowering before final target decisions are known;
- staged lowering to `arith` and `math`;
- lowering to LLVM-compatible aggregate representation;
- lowering selected operations to libm or AMD OCML calls;
- lowering basic complex operations to SPIR-V-compatible vector operations.

It is usually not the final form. Most backends want real arithmetic, calls, or
target-specific operations by the end of the pipeline.

## When To Use It

Use `complex` when the IR should still say "this is a complex number" rather
than "this is a pair of floats that happen to be related."

Good uses include:

- constructing a complex value from real and imaginary parts with
  `complex.create`;
- extracting parts with `complex.re` and `complex.im`;
- representing high-level complex arithmetic;
- keeping complex transcendental functions visible until a math-library or
  target-specific lowering decision is made;
- using `convert-complex-to-standard`, `convert-complex-to-llvm`,
  `convert-complex-to-libm`, `convert-complex-to-spirv`, or
  `convert-complex-to-rocdl-library-calls`.

Avoid using it after the target ABI has already decided how complex numbers
must be represented. At that stage, use the target dialect or runtime-call form
expected by the backend.

## Core Concepts

### Complex Type

The type is `complex<T>`. Most operations in this dialect require `T` to be a
floating-point type:

```mlir
%z = complex.create %real, %imag : complex<f32>
```

`complex.constant` can also build integer-element complex constants, but the
arithmetic and transcendental operations are defined for complex values with
floating-point elements.

### Real And Imaginary Parts

A complex number is conceptually:

```text
real + imaginary * i
```

Use `complex.re` and `complex.im` when you need the individual floating-point
components:

```mlir
%real = complex.re %z : complex<f32>
%imag = complex.im %z : complex<f32>
```

The dialect has canonicalizations that understand these relationships. For
example, `complex.re(complex.create(%r, %i))` can fold to `%r`.

### Fast Math

Many `complex` operations implement `ArithFastMathInterface` and can carry
`fastmath` flags. Those flags matter because complex lowering often expands one
operation into several floating-point operations. A flag such as `nnan` or
`ninf` may change which simplifications or lowerings are valid.

### Built-In Type, Dialect Operations

The complex type itself is not a type defined by this dialect; it is a built-in
MLIR type. The dialect owns the operations and the `#complex.number` attribute.

That means you can see `complex<f32>` in IR even while the operation producing
or consuming it is not from the `complex` dialect.

## Operations

The local LLVM checkout defines 29 `complex` operations.

### Construction, Constants, And Access

| Operation | Purpose |
| --- | --- |
| `complex.constant` | Creates a complex constant from two attributes. |
| `complex.create` | Creates a complex value from real and imaginary operands. |
| `complex.re` | Extracts the real part. |
| `complex.im` | Extracts the imaginary part. |
| `complex.bitcast` | Bitcasts between one complex value and one equal-width scalar integer or float value. |

`complex.bitcast` is intentionally narrow. One side must be complex and the
total bitwidths must match, such as `complex<f32>` to `i64`.

### Basic Arithmetic

| Operation | Purpose |
| --- | --- |
| `complex.add` | Complex addition. |
| `complex.sub` | Complex subtraction. |
| `complex.mul` | Complex multiplication. |
| `complex.div` | Complex division. |
| `complex.neg` | Complex negation. |
| `complex.conj` | Complex conjugate. |
| `complex.abs` | Magnitude, returning the element floating-point type. |
| `complex.angle` | Argument or phase angle, returning the element floating-point type. |
| `complex.sign` | Complex sign, `z / abs(z)` for nonzero `z`, otherwise zero. |

`complex.div` is one of the most important operations for lowering quality.
The conversion passes expose a `complex-range` option because a naive algebraic
formula can overflow or underflow more easily than a range-reduced algorithm.

### Comparisons

| Operation | Purpose |
| --- | --- |
| `complex.eq` | Returns true when both real and imaginary parts are equal. |
| `complex.neq` | Returns true when either real or imaginary part differs. |

These return `i1`, not `complex<i1>`.

### Exponential, Logarithmic, Power, And Roots

| Operation | Purpose |
| --- | --- |
| `complex.exp` | Complex exponential. |
| `complex.expm1` | `exp(z) - 1`. |
| `complex.log` | Natural logarithm. |
| `complex.log1p` | `log(1 + z)`. |
| `complex.pow` | Complex base raised to a complex exponent. |
| `complex.powi` | Complex base raised to a signed integer exponent. |
| `complex.sqrt` | Complex square root. |
| `complex.rsqrt` | Reciprocal square root. |

`complex.pow` and `complex.powi` commonly lower through `log`, `mul`, and
`exp`, or through library calls on targets that provide them.

### Trigonometric And Hyperbolic Operations

| Operation | Purpose |
| --- | --- |
| `complex.sin` | Complex sine. |
| `complex.cos` | Complex cosine. |
| `complex.tan` | Complex tangent. |
| `complex.tanh` | Complex hyperbolic tangent. |
| `complex.atan2` | Complex two-argument arctangent, expressed through log/div/sqrt during lowering. |

The dialect does not try to make these operations look like primitive target
instructions. They are high-level mathematical operations that are later
expanded or converted to calls.

## Attributes

### `#complex.number`

The dialect defines a custom typed attribute for complex numbers:

```mlir
#complex.number<:f64 1.0, 2.0>
```

It stores:

- a real `APFloat`;
- an imaginary `APFloat`;
- the complex type.

Most day-to-day IR examples use `complex.constant [real, imag] :
complex<...>` rather than spelling this attribute directly, but it is part of
the dialect's attribute model.

### `ComplexRangeFlags`

`ComplexRangeFlags` is an enum used by conversion passes to control complex
division lowering. The values are:

| Value | Meaning |
| --- | --- |
| `improved` | Use Smith-style range reduction for better behavior around overflow, underflow, infinities, and NaNs. |
| `basic` | Use the direct algebraic formula. |
| `none` | Use the direct algebraic formula, like `basic`, without the improved range-reduction path. |

The pass defaults differ:

- `convert-complex-to-standard` defaults to `improved`;
- `convert-complex-to-llvm` defaults to `basic`.

## Transformations

The Complex dialect does not have a standalone "optimize-complex" pass in this
checkout. Its main transformations are canonicalizations, folders, and
conversion passes.

### Folding And Canonicalization

The dialect implementation includes local folds and canonicalizations such as:

- `complex.constant` folds to its attribute value;
- `complex.create(complex.re(%z), complex.im(%z))` folds to `%z`;
- `complex.re(complex.create(%r, %i))` folds to `%r`;
- `complex.im(complex.create(%r, %i))` folds to `%i`;
- nested `complex.bitcast` and mixed `complex.bitcast` / `arith.bitcast`
  chains can be merged;
- `complex.add(complex.sub(a, b), b)` folds to `a`;
- `complex.sub(complex.add(a, b), b)` folds to `a`;
- adding or subtracting complex zero can fold away;
- `complex.neg(complex.neg(a))` folds to `a`;
- `complex.log(complex.exp(a))` and `complex.exp(complex.log(a))` fold back
  to `a`;
- `complex.conj(complex.conj(a))` folds to `a`;
- multiplying by complex one can fold away;
- dividing by complex one can fold away when NaN behavior is safe.

These folds usually run through regular MLIR cleanup passes such as
`canonicalize` and `cse`.

### Inlining And Constants

The dialect has an inliner interface that permits Complex dialect operations to
be inlined. It also has a constant materializer for `complex.constant`.

## Conversions And Lowering Paths

### To Standard, Arith, And Math

`convert-complex-to-standard` lowers complex operations into lower-level
operations, mostly `arith`, `math`, and simpler `complex` construction and
extraction operations that can be further rewritten.

The local pattern list covers:

- `complex.abs`;
- `complex.angle`;
- `complex.atan2`;
- `complex.add`;
- `complex.sub`;
- `complex.eq`;
- `complex.neq`;
- `complex.conj`;
- `complex.cos`;
- `complex.exp`;
- `complex.expm1`;
- `complex.log`;
- `complex.log1p`;
- `complex.mul`;
- `complex.neg`;
- `complex.sign`;
- `complex.sin`;
- `complex.sqrt`;
- `complex.tan`;
- `complex.tanh`;
- `complex.pow`;
- `complex.powi`;
- `complex.rsqrt`;
- `complex.div`.

Use this when you want portable scalar formulas expressed in MLIR rather than
calls to a C or device math library.

### To LLVM

`convert-complex-to-llvm` lowers core complex representation and arithmetic to
the LLVM dialect. A complex value becomes an LLVM-compatible aggregate with real
and imaginary fields.

The local LLVM conversion patterns cover:

- `complex.abs`;
- `complex.add`;
- `complex.constant`;
- `complex.create`;
- `complex.re`;
- `complex.im`;
- `complex.mul`;
- `complex.sub`;
- `complex.div`.

Use this path when the complex operations you still have are in that core set.
For broader math such as `complex.sin` or `complex.exp`, first lower through
standard/math or library-call conversions.

### To libm Calls

`convert-complex-to-libm` rewrites supported scalar complex operations to libm
function calls. In this checkout, it covers:

| Operation | f32 call | f64 call |
| --- | --- | --- |
| `complex.pow` | `cpowf` | `cpow` |
| `complex.sqrt` | `csqrtf` | `csqrt` |
| `complex.tanh` | `ctanhf` | `ctanh` |
| `complex.cos` | `ccosf` | `ccos` |
| `complex.sin` | `csinf` | `csin` |
| `complex.conj` | `conjf` | `conj` |
| `complex.log` | `clogf` | `clog` |
| `complex.abs` | `cabsf` | `cabs` |
| `complex.angle` | `cargf` | `carg` |
| `complex.tan` | `ctanf` | `ctan` |

Use this when the target runtime provides the C complex math functions and you
prefer calls over expanded formulas.

### To ROCDL Library Calls

`convert-complex-to-rocdl-library-calls` rewrites supported complex operations
to AMD device-library calls. In this checkout it covers:

- `complex.abs` to `__ocml_cabs_f32` or `__ocml_cabs_f64`;
- `complex.cos` to `__ocml_ccos_f32` or `__ocml_ccos_f64`;
- `complex.exp` to `__ocml_cexp_f32` or `__ocml_cexp_f64`;
- `complex.log` to `__ocml_clog_f32` or `__ocml_clog_f64`;
- `complex.sin` to `__ocml_csin_f32` or `__ocml_csin_f64`;
- `complex.sqrt` to `__ocml_csqrt_f32` or `__ocml_csqrt_f64`;
- `complex.tan` to `__ocml_ctan_f32` or `__ocml_ctan_f64`;
- `complex.tanh` to `__ocml_ctanh_f32` or `__ocml_ctanh_f64`;
- `complex.pow`, by lowering through `log`, `mul`, and `exp`;
- `complex.powi`, by converting the integer exponent to a complex value and
  lowering through `log`, `mul`, and `exp`.

Use this path for AMD GPU pipelines where OCML calls are the intended math
implementation.

### To SPIR-V

`convert-complex-to-spirv` lowers a focused subset of complex operations using
SPIR-V-compatible representations and operations. The local pattern list covers:

- `complex.constant`;
- `complex.create`;
- `complex.re`;
- `complex.im`;
- `complex.add`;
- `complex.sub`;
- `complex.mul`;
- `complex.div`;
- `complex.neg`;
- `complex.conj`;
- `complex.abs`.

The conversion represents complex values through converted types such as
two-element vectors and uses SPIR-V arithmetic, GLSL sqrt, or OpenCL sqrt
depending on the available target environment.

## Example IR

This example creates a complex number and extracts its parts:

```mlir
func.func @make_and_split(%r: f32, %i: f32) -> (f32, f32) {
  %z = complex.create %r, %i : complex<f32>
  %real = complex.re %z : complex<f32>
  %imag = complex.im %z : complex<f32>
  return %real, %imag : f32, f32
}
```

This example performs basic arithmetic:

```mlir
func.func @multiply_add(%a: complex<f32>, %b: complex<f32>,
                        %c: complex<f32>) -> complex<f32> {
  %prod = complex.mul %a, %b : complex<f32>
  %sum = complex.add %prod, %c : complex<f32>
  return %sum : complex<f32>
}
```

This example uses a higher-level complex math operation:

```mlir
func.func @phase_and_magnitude(%z: complex<f64>) -> (f64, f64) {
  %mag = complex.abs %z : complex<f64>
  %angle = complex.angle %z : complex<f64>
  return %mag, %angle : f64, f64
}
```

## Mental Model

Think of `complex` as a mathematical holding area.

At the frontend side, it keeps complex intent visible: a value is not just two
unrelated floats; it is one complex number.

At the backend side, the compiler must eventually choose a representation:
expanded real arithmetic, LLVM aggregate values, libm calls, OCML calls, or
SPIR-V-compatible vector operations.

The dialect is useful because it delays that representation decision until the
compiler has enough target information.

## Gotchas

- `complex<T>` is a built-in type, not a type defined in `ComplexOps.td`.
- Most arithmetic and math ops require floating-point complex element types.
- `complex.constant` can represent integer-element complex constants, but that
  does not mean every operation works on integer complex values.
- `complex.bitcast` requires matching total bitwidth and exactly one complex
  side, except for same-type folds.
- Complex division lowering is subtle. The `complex-range` option changes the
  algorithm used by `convert-complex-to-standard` and
  `convert-complex-to-llvm`.
- Not every conversion pass covers every complex op. Pick the conversion path
  based on the operations that remain in the IR.
- Library-call conversions assume the target has the named functions with the
  expected ABI.
- Fast-math flags propagate into expanded floating-point operations and can
  affect correctness around NaNs and infinities.

## Source Map

Use these files in the LLVM repo when you need exact behavior:

| Topic | Files |
| --- | --- |
| Dialect base and range enum | `mlir/include/mlir/Dialect/Complex/IR/ComplexBase.td` |
| Operation definitions | `mlir/include/mlir/Dialect/Complex/IR/ComplexOps.td` |
| Attribute definitions | `mlir/include/mlir/Dialect/Complex/IR/ComplexAttributes.td` |
| Dialect and attribute implementation | `mlir/lib/Dialect/Complex/IR/ComplexDialect.cpp` |
| Operation folding and canonicalization | `mlir/lib/Dialect/Complex/IR/ComplexOps.cpp` |
| Conversion pass declarations | `mlir/include/mlir/Conversion/Passes.td` |
| Shared division lowering | `mlir/lib/Conversion/ComplexCommon/DivisionConverter.cpp` |
| Standard conversion | `mlir/lib/Conversion/ComplexToStandard/ComplexToStandard.cpp` |
| LLVM conversion | `mlir/lib/Conversion/ComplexToLLVM/ComplexToLLVM.cpp` |
| libm conversion | `mlir/lib/Conversion/ComplexToLibm/ComplexToLibm.cpp` |
| ROCDL library conversion | `mlir/lib/Conversion/ComplexToROCDLLibraryCalls/ComplexToROCDLLibraryCalls.cpp` |
| SPIR-V conversion | `mlir/lib/Conversion/ComplexToSPIRV` |
| Dialect tests | `mlir/test/Dialect/Complex` |
| Conversion tests | `mlir/test/Conversion/ComplexToLLVM`, `mlir/test/Conversion/ComplexToStandard`, `mlir/test/Conversion/ComplexToLibm`, `mlir/test/Conversion/ComplexToSPIRV`, `mlir/test/Conversion/ComplexToROCDLLibraryCalls` |
