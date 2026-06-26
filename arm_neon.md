# `arm_neon` Dialect

## Beginner Summary

`arm_neon` is an Arm CPU target dialect for NEON SIMD operations. It represents
operations that map closely to AArch64 NEON LLVM intrinsics.

You usually do not start a compiler pipeline in this dialect. Instead, you
start with target-independent dialects such as `linalg`, `vector`, `arith`,
and `scf`, then lower selected vector operations into `arm_neon` when you know
the target is Arm NEON and the operation shape matches a useful hardware
instruction.

The dialect is small and specialized. Its main job is to expose efficient NEON
dot-product and matrix multiply-accumulate instructions that generic vector IR
cannot name directly.

## Why This Dialect Exists

The `vector` dialect is target-independent. That is useful for most
optimization, but eventually a compiler may want to select a specific hardware
instruction.

Arm NEON has instructions such as:

- signed multiply long;
- signed dot product;
- signed and unsigned 8-bit matrix multiply-accumulate;
- BF16 matrix multiply-accumulate.

These instructions have strict vector shapes and element types. Representing
them as explicit `arm_neon` operations gives the compiler a clear bridge from
generic vector computation to LLVM AArch64 NEON intrinsics.

## When It Matters

`arm_neon` matters late in a CPU lowering pipeline for AArch64 targets.

You are likely to see it when:

- lowering `vector.contract` to Arm FEAT_I8MM instructions;
- lowering BF16 vector contractions to Arm FEAT_BF16 instructions;
- using `convert-vector-to-llvm` with `enable-arm-neon`;
- using transform dialect pattern ops that request Arm NEON contraction
  lowering;
- translating MLIR to LLVM IR and expecting AArch64 NEON intrinsics.

It is not a general-purpose vector dialect. Use `vector` for target-independent
vector IR. Use `arm_neon` when you want specific Arm NEON instructions.

## When To Use It

Use `arm_neon` when all of these are true:

- the target is an Arm CPU with the relevant NEON feature;
- the computation matches one of the dialect's supported instruction shapes;
- you are late enough in lowering that target-specific IR is acceptable;
- you want MLIR-to-LLVM translation to emit the corresponding AArch64
  intrinsic.

Do not use it as an early frontend dialect. Early IR should usually stay in
`linalg`, `tensor`, `memref`, or `vector` so generic optimizations can still
work.

## Core Concepts

### NEON Is Fixed-Width SIMD

The operations in this dialect use fixed vector types such as:

```mlir
vector<8xi8>
vector<16xi8>
vector<4xi32>
vector<8xbf16>
vector<4xf32>
```

They are not scalable vector operations. If the target path needs scalable
vectors, look at `arm_sve` instead.

### Intrinsic-Like Operations

Most `arm_neon` ops are named `arm_neon.intr.*`. These are close to LLVM
AArch64 NEON intrinsics. They are not abstract math operations; they already
encode target instruction choices.

### Matrix Multiply-Accumulate

Several ops model small matrix multiply-accumulate instructions. The result is
usually a 2x2 logical tile packed into a one-dimensional vector:

```mlir
vector<4xi32>
vector<4xf32>
```

This is why some operations take flattened vector operands even though the
mental model is matrix multiplication.

### Feature-Specific Lowering

Some ops require specific Arm architectural features:

- I8 matrix multiply patterns correspond to FEAT_I8MM.
- BF16 matrix multiply patterns correspond to FEAT_BF16.

The MLIR op can exist in IR, but generating correct runnable code requires a
target that actually supports the corresponding feature.

## Operations

The `arm_neon` dialect defines seven payload operations.

| Operation | Purpose |
|---|---|
| `arm_neon.intr.smull` | Signed multiply long. Multiplies smaller signed integer vector lanes and produces wider integer lanes. |
| `arm_neon.intr.sdot` | Signed dot product. Computes grouped 8-bit dot products and accumulates into 32-bit lanes. |
| `arm_neon.intr.smmla` | Signed 8-bit matrix multiply-accumulate into 32-bit lanes. |
| `arm_neon.intr.ummla` | Unsigned 8-bit matrix multiply-accumulate into 32-bit lanes. |
| `arm_neon.intr.usmmla` | Mixed unsigned-by-signed 8-bit matrix multiply-accumulate into 32-bit lanes. |
| `arm_neon.intr.bfmmla` | BF16 matrix multiply-accumulate into 32-bit float lanes. |
| `arm_neon.2d.sdot` | Structured 2D signed dot-product form that lowers to flattened `arm_neon.intr.sdot`. |

### `arm_neon.intr.smull`

Signed multiply long. The operands have the same vector type, and the result
has the same number of lanes with twice the element bitwidth.

Supported forms:

```mlir
%0 = arm_neon.intr.smull %a, %b : vector<8xi8> to vector<8xi16>
%1 = arm_neon.intr.smull %c, %d : vector<4xi16> to vector<4xi32>
%2 = arm_neon.intr.smull %e, %f : vector<2xi32> to vector<2xi64>
```

### `arm_neon.intr.sdot`

Signed dot product over groups of four `i8` elements, accumulating into `i32`
lanes.

Supported forms:

```mlir
%0 = arm_neon.intr.sdot %acc, %lhs, %rhs
  : vector<8xi8>, vector<8xi8> to vector<2xi32>

%1 = arm_neon.intr.sdot %acc4, %lhs16, %rhs16
  : vector<16xi8>, vector<16xi8> to vector<4xi32>
```

### `arm_neon.intr.smmla`

Signed 8-bit matrix multiply-accumulate.

```mlir
%0 = arm_neon.intr.smmla %acc, %lhs, %rhs
  : vector<16xi8> to vector<4xi32>
```

The accumulator and result are `vector<4xi32>`. The inputs are
`vector<16xi8>`.

### `arm_neon.intr.ummla`

Unsigned 8-bit matrix multiply-accumulate.

```mlir
%0 = arm_neon.intr.ummla %acc, %lhs, %rhs
  : vector<16xi8> to vector<4xi32>
```

This has the same shape as `smmla`, but the input interpretation is unsigned.

### `arm_neon.intr.usmmla`

Mixed unsigned-by-signed 8-bit matrix multiply-accumulate.

```mlir
%0 = arm_neon.intr.usmmla %acc, %lhs, %rhs
  : vector<16xi8> to vector<4xi32>
```

Lowering patterns may swap operands to express the signed-by-unsigned case
using this instruction form.

### `arm_neon.intr.bfmmla`

BF16 matrix multiply-accumulate into `f32`.

```mlir
%0 = arm_neon.intr.bfmmla %acc, %lhs, %rhs
  : vector<8xbf16> to vector<4xf32>
```

The accumulator and result are `vector<4xf32>`. The inputs are
`vector<8xbf16>`.

### `arm_neon.2d.sdot`

A structured 2D dot-product form. It uses 2D input vectors and lowers to
`arm_neon.intr.sdot` by flattening the input vectors.

```mlir
%0 = arm_neon.2d.sdot %acc, %lhs, %rhs
  : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
```

Supported shapes include:

```mlir
vector<2xi32>, vector<2x4xi8>, vector<2x4xi8> -> vector<2xi32>
vector<4xi32>, vector<4x4xi8>, vector<4x4xi8> -> vector<4xi32>
```

## Transformations

### Vector Contract To I8MM Patterns

The Arm NEON transform extension provides:

```mlir
transform.apply_patterns.arm_neon.vector_contract_to_i8mm
```

This indicates that matching `vector.contract` operations should be lowered to
`arm_neon` operations that map to FEAT_I8MM instructions.

The implementation can produce:

- `arm_neon.intr.smmla` for signed-by-signed inputs;
- `arm_neon.intr.ummla` for unsigned-by-unsigned inputs;
- `arm_neon.intr.usmmla` for mixed signedness.

The patterns handle compatible matrix-multiply-style contractions, including
some larger shapes by tiling/unrolling into sub-tiles that match NEON
instruction shapes.

### Vector Contract To BFMMLA Patterns

The transform extension also provides:

```mlir
transform.apply_patterns.arm_neon.vector_contract_to_bfmmla
```

This lowers compatible BF16 `vector.contract` operations to
`arm_neon.intr.bfmmla`, corresponding to FEAT_BF16.

### `arm-neon-2d-to-intr`

The pass:

```text
arm-neon-2d-to-intr
```

lowers `arm_neon.2d.sdot` to `arm_neon.intr.sdot`. It inserts
`vector.shape_cast` operations to flatten the 2D `i8` vectors into the 1D
vectors required by the intrinsic form.

### `convert-vector-to-llvm` Options

The `convert-vector-to-llvm` pass can use Arm-specific dialects when options
are enabled:

```text
convert-vector-to-llvm='enable-arm-neon enable-arm-i8mm'
convert-vector-to-llvm='enable-arm-neon enable-arm-bf16'
```

These options allow vector lowering to use `arm_neon` as an intermediate
target-specific dialect before final LLVM IR translation.

## Conversions And Lowering Paths

Common lowering paths:

```text
vector.contract
  -> arm_neon.intr.smmla / ummla / usmmla / bfmmla
  -> LLVM AArch64 NEON intrinsic calls
```

```text
arm_neon.2d.sdot
  -> vector.shape_cast + arm_neon.intr.sdot
  -> LLVM AArch64 NEON intrinsic call
```

```text
generic vector IR
  -> convert-vector-to-llvm with enable-arm-neon
  -> arm_neon plus llvm dialect
  -> LLVM IR
```

The final MLIR-to-LLVM translation registers an Arm NEON dialect translation
interface. That interface emits LLVM AArch64 NEON intrinsics such as
`llvm.aarch64.neon.sdot`, `llvm.aarch64.neon.smmla`, and related intrinsic
forms.

## Example IR

### Direct Intrinsic Form

```mlir
func.func @dot(%acc: vector<4xi32>,
               %lhs: vector<16xi8>,
               %rhs: vector<16xi8>) -> vector<4xi32> {
  %0 = arm_neon.intr.sdot %acc, %lhs, %rhs
    : vector<16xi8>, vector<16xi8> to vector<4xi32>
  return %0 : vector<4xi32>
}
```

### 2D Form Lowering To Intrinsic Form

Before:

```mlir
func.func @dot2d(%acc: vector<4xi32>,
                 %lhs: vector<4x4xi8>,
                 %rhs: vector<4x4xi8>) -> vector<4xi32> {
  %0 = arm_neon.2d.sdot %acc, %lhs, %rhs
    : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
  return %0 : vector<4xi32>
}
```

After `arm-neon-2d-to-intr`, conceptually:

```mlir
%lhs1d = vector.shape_cast %lhs : vector<4x4xi8> to vector<16xi8>
%rhs1d = vector.shape_cast %rhs : vector<4x4xi8> to vector<16xi8>
%0 = arm_neon.intr.sdot %acc, %lhs1d, %rhs1d
  : vector<16xi8>, vector<16xi8> to vector<4xi32>
```

### Matrix Multiply-Accumulate

```mlir
func.func @i8mm(%acc: vector<4xi32>,
                %lhs: vector<16xi8>,
                %rhs: vector<16xi8>) -> vector<4xi32> {
  %0 = arm_neon.intr.smmla %acc, %lhs, %rhs
    : vector<16xi8> to vector<4xi32>
  return %0 : vector<4xi32>
}
```

## Mental Model

Think of `arm_neon` as the named NEON-instruction layer between generic MLIR
vector code and LLVM AArch64 intrinsics.

Use `vector` while you still want portable optimization. Use `arm_neon` when
you are selecting concrete Arm NEON SIMD instructions.

## Gotchas

- The dialect is target-specific. IR containing `arm_neon` is no longer
  portable vector IR.
- The ops require exact fixed vector shapes and element types.
- The matrix multiply ops produce packed logical 2x2 tiles in one-dimensional
  vectors.
- `arm_neon` is not `arm_sve` or `arm_sme`. SVE is scalable-vector oriented;
  SME is matrix/tile oriented.
- The lowering patterns reject scalable vectors.
- I8MM and BF16 paths require target feature support.
- `arm_neon.2d.sdot` is not the final intrinsic form; it must lower to
  `arm_neon.intr.sdot`.
- `convert-vector-to-llvm` will not use Arm NEON unless the relevant options
  are enabled.

## Source Map

Primary dialect files:

- `mlir/include/mlir/Dialect/ArmNeon/ArmNeon.td`
- `mlir/include/mlir/Dialect/ArmNeon/ArmNeonDialect.h`
- `mlir/lib/Dialect/ArmNeon/IR/ArmNeonDialect.cpp`

Transforms:

- `mlir/include/mlir/Dialect/ArmNeon/Transforms.h`
- `mlir/lib/Dialect/ArmNeon/Transforms/LowerContractToNeonPatterns.cpp`
- `mlir/include/mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.td`
- `mlir/lib/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.cpp`

Conversions and translation:

- `mlir/lib/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.cpp`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp`
- `mlir/include/mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h`
- `mlir/lib/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.cpp`

Tests:

- `mlir/test/Dialect/ArmNeon/roundtrip.mlir`
- `mlir/test/Dialect/ArmNeon/invalid.mlir`
- `mlir/test/Dialect/ArmNeon/lower-to-arm-neon.mlir`
- `mlir/test/Dialect/ArmNeon/vector-bfmmla.mlir`
- `mlir/test/Target/LLVMIR/arm-neon.mlir`
- `mlir/test/Target/LLVMIR/arm-neon-2d.mlir`
- `mlir/test/Integration/Dialect/Vector/CPU/ArmNeon/`
