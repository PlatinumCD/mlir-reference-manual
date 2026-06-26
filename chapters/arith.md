# `arith` Dialect

## Beginner Summary

The `arith` dialect is MLIR's basic arithmetic dialect. It holds integer arithmetic, floating-point arithmetic, bitwise operations, comparisons, casts, constants, and selects.

For a beginner, `arith` is the dialect you expect to see almost everywhere. It is the scalar math layer that other dialects build on:

- `linalg` regions use `arith.addf`, `arith.mulf`, `arith.cmpi`, and friends to describe element computations.
- `scf` loops use `arith.constant`, `arith.addi`, and `arith.cmpi` for loop math and conditions.
- `memref`, `tensor`, and lowering pipelines use `arith.index_cast`, integer arithmetic, and constants for indexing.
- Target conversions lower `arith` into LLVM, SPIR-V, EmitC, AMDGPU, ArmSME, or runtime calls.

Most `arith` operations work on scalars, vectors, and tensors. On vectors and tensors, the operation applies elementwise.

## Why This Dialect Exists

MLIR needs a shared, target-independent way to represent basic math. Without `arith`, every dialect would need its own addition, comparison, cast, and constant operations. That would make optimization and lowering much harder.

The `arith` dialect exists to provide:

- A common arithmetic vocabulary for integer, index, and floating-point values.
- Elementwise arithmetic on vectors and tensors.
- Explicit signed versus unsigned integer behavior.
- Explicit floating-point comparison predicates, fast-math flags, and rounding modes.
- Casts between integer, index, and floating-point types.
- A stable source for canonicalization, constant folding, range analysis, and target conversion.

The dialect assumes integers are bitvectors using two's complement representation. Unless an operation says otherwise, poison inputs produce poison outputs, and vector or tensor poison behavior is elementwise.

## When It Matters

The `arith` dialect matters in almost every nontrivial MLIR pipeline:

- In frontend lowering, it expresses literal constants and basic scalar computation.
- In high-level tensor code, it appears inside `linalg.generic` bodies.
- In loop code, it computes induction-derived values and conditions.
- In buffer and index code, it casts between `index` and fixed-width integers.
- In optimization, it enables constant folding, canonicalization, integer range analysis, and narrowing.
- In target lowering, it is one of the main dialects converted to LLVM, SPIR-V, EmitC, or target-specific dialects.

## When To Use It

Use `arith` when you need target-independent scalar or elementwise math:

- Use integer ops for fixed-width integer and `index` arithmetic.
- Use floating-point ops for target-independent float math.
- Use comparison ops to produce `i1`, vector-of-`i1`, or tensor-of-`i1` conditions.
- Use cast ops to change between integer widths, index, and floating-point types.
- Use `arith.select` for value selection without control-flow branching.

Avoid using `arith` for:

- Memory access. Use `memref`, `tensor`, or bufferization dialects.
- Structured loops. Use `scf`, `affine`, or another control-flow dialect.
- Target intrinsics. Lower to target dialects such as LLVM, SPIR-V, AMDGPU, or ArmSME when target details matter.
- High-level algebraic structure. Use `linalg`, `tensor`, `vector`, or domain dialects before reducing everything to scalar math.

## Core Concepts

### Elementwise Semantics

Many `arith` operations accept scalar, vector, and tensor operands. For vectors and tensors, the operation is applied lane-by-lane or element-by-element:

```text
arith.addi : vector<4xi32> means four independent i32 additions.
```

This is why `arith` appears both in scalar loop bodies and in vectorized IR.

### Signed Versus Unsigned

Integer values in MLIR are signless by default, so the operation chooses signed or unsigned interpretation:

- Signed examples: `arith.divsi`, `arith.remsi`, `arith.maxsi`, `arith.minsi`, signed `arith.cmpi` predicates like `slt`.
- Unsigned examples: `arith.divui`, `arith.remui`, `arith.maxui`, `arith.minui`, unsigned `arith.cmpi` predicates like `ult`.
- Bitwise operations such as `arith.andi`, `arith.ori`, and `arith.xori` do not need signedness.

### Overflow And Exact Flags

Some integer operations can carry overflow flags:

- `overflow<nsw>` means no signed wrap.
- `overflow<nuw>` means no unsigned wrap.
- `overflow<nsw, nuw>` means both assumptions hold.

Some division and shift operations can carry `exact`, meaning the operation has no discarded nonzero remainder or shifted-out bits. These flags are promises to the optimizer. If the promise is false, the result can become poison or undefined in later lowering.

### Floating-Point Flags And Rounding

Floating-point operations may carry:

- Rounding modes such as `to_nearest_even`, `downward`, `upward`, `toward_zero`, and `to_nearest_away`.
- Fast-math flags such as `nnan`, `ninf`, `nsz`, `reassoc`, `arcp`, `contract`, `afn`, or grouped `fast`.

When no explicit rounding mode is present, Arith uses round-to-nearest ties-to-even for internal constant folding and canonicalization. Runtime behavior without an explicit rounding mode is deferred to the target backend.

### Poison

The dialect documentation says Arith generally propagates poison. This matters because flags such as `nsw`, `nuw`, `exact`, and fast-math assumptions can give optimizers permission to replace code based on facts the program producer promised.

### Attributes

Important Arith attributes include:

- `CmpIPredicate`: integer comparison predicates `eq`, `ne`, `slt`, `sle`, `sgt`, `sge`, `ult`, `ule`, `ugt`, `uge`.
- `CmpFPredicate`: floating comparison predicates such as `oeq`, `ogt`, `olt`, `ord`, `ueq`, `uno`, plus always-true and always-false predicates.
- `FastMathFlags`: floating-point optimization flags.
- `IntegerOverflowFlags`: `nsw` and `nuw`.
- `RoundingMode`: explicit floating-point rounding mode.
- `AtomicRMWKind`: shared arithmetic names used by atomic read-modify-write style operations in other dialects.

## Operations

### Constants And Selection

- `arith.constant` creates an integer, index, floating-point, vector, tensor, or other typed constant value.
- `arith.select` chooses between two values using an `i1`, vector-of-`i1`, or tensor-of-`i1` condition.

### Integer Arithmetic

- `arith.addi` adds integers or indices and may carry overflow flags.
- `arith.subi` subtracts integers or indices and may carry overflow flags.
- `arith.muli` multiplies integers or indices and may carry overflow flags.
- `arith.divsi` performs signed integer division and may be marked `exact`.
- `arith.divui` performs unsigned integer division and may be marked `exact`.
- `arith.ceildivsi` performs signed integer division rounded toward positive infinity.
- `arith.ceildivui` performs unsigned integer division rounded upward.
- `arith.floordivsi` performs signed integer division rounded toward negative infinity.
- `arith.remsi` computes signed integer remainder.
- `arith.remui` computes unsigned integer remainder.

### Extended Integer Arithmetic

- `arith.addui_extended` computes unsigned addition and returns both the sum and overflow bit.
- `arith.subui_extended` computes unsigned subtraction and returns both the difference and borrow bit.
- `arith.mulsi_extended` computes signed multiplication and returns low and high halves.
- `arith.mului_extended` computes unsigned multiplication and returns low and high halves.

### Bitwise And Shift Operations

- `arith.andi` computes bitwise and.
- `arith.ori` computes bitwise or.
- `arith.xori` computes bitwise xor.
- `arith.shli` shifts left and may carry overflow flags.
- `arith.shrsi` performs signed arithmetic right shift and may be marked `exact`.
- `arith.shrui` performs unsigned logical right shift and may be marked `exact`.

### Floating-Point Arithmetic

- `arith.negf` negates a floating-point value.
- `arith.addf` adds floating-point values and may carry rounding and fast-math metadata.
- `arith.subf` subtracts floating-point values and may carry rounding and fast-math metadata.
- `arith.mulf` multiplies floating-point values and may carry rounding and fast-math metadata.
- `arith.divf` divides floating-point values and may carry rounding and fast-math metadata.
- `arith.remf` computes floating-point remainder.
- `arith.maximumf` computes a maximum with NaN propagation semantics.
- `arith.minimumf` computes a minimum with NaN propagation semantics.
- `arith.maxnumf` computes a maximum-number style operation that handles NaN differently from `maximumf`.
- `arith.minnumf` computes a minimum-number style operation that handles NaN differently from `minimumf`.
- `arith.flush_denormals` flushes denormal floating-point values to zero.

### Integer Min And Max

- `arith.maxsi` computes signed integer maximum.
- `arith.maxui` computes unsigned integer maximum.
- `arith.minsi` computes signed integer minimum.
- `arith.minui` computes unsigned integer minimum.

### Casts Between Integer, Index, And Float

- `arith.extsi` sign-extends an integer to a wider integer type.
- `arith.extui` zero-extends an integer to a wider integer type and can carry `nneg`.
- `arith.trunci` truncates an integer to a narrower integer type and may carry overflow flags.
- `arith.index_cast` casts between `index` and integer types using signed interpretation.
- `arith.index_castui` casts between `index` and integer types using unsigned interpretation and can carry `nneg`.
- `arith.sitofp` converts signed integer to floating-point.
- `arith.uitofp` converts unsigned integer to floating-point and can carry `nneg`.
- `arith.fptosi` converts floating-point to signed integer.
- `arith.fptoui` converts floating-point to unsigned integer.
- `arith.extf` extends a floating-point value to a wider floating-point type and may carry fast-math metadata.
- `arith.truncf` truncates a floating-point value to a narrower floating-point type and may carry rounding and fast-math metadata.
- `arith.convertf` converts between floating-point types of the same bit width and may carry rounding and fast-math metadata.
- `arith.bitcast` reinterprets bits between equal-bit-width types.
- `arith.scaling_extf` upcasts input floats using scale values, useful for scaled low-precision floating formats.
- `arith.scaling_truncf` downcasts floating-point values using scale values, useful for scaled low-precision floating formats.

### Comparisons

- `arith.cmpi` compares integer, index, vector, or tensor values using integer predicates.
- `arith.cmpf` compares floating-point, vector, or tensor values using ordered or unordered floating predicates.

## Transformations

Native Arith transformations include:

- `arith-expand`: legalizes selected Arith ops into simpler Arith and vector operations that are easier to convert to LLVM. Options include `include-bf16`, `include-f8e8m0`, `include-f4e2m1`, and `include-flush-denormals`.
- `arith-unsigned-when-equivalent`: uses integer range analysis to replace signed operations with unsigned equivalents when operands and results are proven non-negative.
- `int-range-optimizations`: runs integer range analysis and folds or rewrites operations based on known ranges, such as known-constant comparisons or trivial remainders.
- `arith-int-range-narrowing`: narrows integer operations to supported bit widths when range analysis proves it is safe. The `int-bitwidths-supported` option names the allowed widths.
- `arith-emulate-unsupported-floats`: wraps unsupported floating-point operations with `extf` and `truncf` to compute in a supported target type. Options include `source-types` and `target-type`.
- `arith-emulate-wide-int`: splits too-wide integer operations into operations on narrower integer halves. The `widest-int-supported` option controls the target integer width.

Arith also participates in canonicalization, constant folding, integer range inference, value bounds inference, bufferization interfaces, and sharding interfaces.

## Conversions/Lowering Paths

Important conversion passes include:

- `convert-arith-to-llvm`: lowers supported Arith operations to LLVM dialect. The `index-bitwidth` option controls the bit width used for `index`.
- `convert-arith-to-spirv`: lowers Arith operations to SPIR-V. It can emulate sub-32-bit scalar types and unsupported float types.
- `convert-arith-to-emitc`: lowers Arith operations to EmitC source-level operations.
- `convert-arith-to-amdgpu`: lowers selected Arith operations, currently focused on 8-bit float extension/truncation patterns, to AMDGPU-specific operations.
- `convert-arith-to-arm-sme`: lowers supported Arith operations to ArmSME dialect.
- `convert-arith-to-apfloat`: lowers supported Arith operations to APFloat runtime library calls for software floating-point behavior.

Arith is also a common lowering target:

- `tosa-to-arith` lowers selected TOSA operations to Arith.
- Affine lowering often produces Arith plus SCF.
- Many dialects use Arith inside generated loop bodies after lowering.

A typical lowering flow is:

1. High-level dialects lower computation into `linalg`, `scf`, `vector`, and `arith`.
2. Canonicalization and range-based Arith passes simplify scalar math.
3. Expansion or emulation passes rewrite operations the target cannot directly support.
4. Target conversion lowers Arith to LLVM, SPIR-V, EmitC, AMDGPU, ArmSME, or runtime calls.

## Example IR

This example shows integer arithmetic with overflow and exactness flags.

```mlir
func.func @integer_ops(%a: i32, %b: i32) -> (i32, i1, i32) {
  %sum = arith.addi %a, %b overflow<nsw, nuw> : i32
  %quot = arith.divsi %sum, %b exact : i32
  %cmp = arith.cmpi sgt, %sum, %quot : i32
  %chosen = arith.select %cmp, %sum, %quot : i32
  return %sum, %cmp, %chosen : i32, i1, i32
}
```

This example shows floating-point rounding and fast-math metadata.

```mlir
func.func @float_ops(%x: f32, %y: f32) -> (f32, i1, f32) {
  %prod = arith.mulf %x, %y upward fastmath<contract> : f32
  %mx = arith.maximumf %prod, %x : f32
  %cmp = arith.cmpf ogt, %mx, %y : f32
  %out = arith.select %cmp, %mx, %y : f32
  return %prod, %cmp, %out : f32, i1, f32
}
```

This example shows casts between integer, index, and floating-point types.

```mlir
func.func @casts(%i: i32, %f: f32, %idx: index) -> (i64, f64, index, i32) {
  %wide_i = arith.extsi %i : i32 to i64
  %wide_f = arith.extf %f fastmath<contract> : f32 to f64
  %idx_from_i = arith.index_castui %i nneg : i32 to index
  %i_from_idx = arith.index_cast %idx : index to i32
  return %wide_i, %wide_f, %idx_from_i, %i_from_idx : i64, f64, index, i32
}
```

This example shows vector elementwise arithmetic and selection.

```mlir
func.func @vector_ops(%a: vector<4xi32>, %b: vector<4xi32>, %mask: vector<4xi1>) -> vector<4xi32> {
  %sum = arith.addi %a, %b : vector<4xi32>
  %zero = arith.constant dense<0> : vector<4xi32>
  %selected = arith.select %mask, %sum, %zero : vector<4xi1>, vector<4xi32>
  return %selected : vector<4xi32>
}
```

## Mental Model

Think of `arith` as MLIR's target-independent calculator. It does not allocate memory, own control flow, or describe high-level algorithms. It computes values.

The dialect is small in concept but precise in semantics. The hard parts are not addition or multiplication themselves. The hard parts are:

- Whether an integer operation is signed or unsigned.
- Whether overflow is allowed to happen.
- Whether division or shift is exact.
- Whether floating-point operations can ignore NaNs, signed zero, infinities, or reassociation rules.
- Whether a cast preserves sign, zero-extends, truncates, rounds, or reinterprets bits.

If you keep those details straight, Arith IR is usually straightforward to read.

## Gotchas

- Signless integer types do not imply signed behavior. The op or predicate chooses signed versus unsigned interpretation.
- Overflow flags are promises. Do not add `nsw`, `nuw`, or `exact` unless the producer can prove them.
- `arith.maximumf` and `arith.maxnumf` are not identical; likewise `arith.minimumf` and `arith.minnumf` differ in NaN handling.
- Floating-point fast-math flags can make transformations legal that would otherwise be invalid under strict IEEE behavior.
- Rounding modes affect constant folding and canonicalization.
- `index` is not a fixed-width integer in the source IR. Target conversions decide its bit width, often through data layout or pass options.
- `arith.index_cast` and `arith.index_castui` differ by signed versus unsigned interpretation.
- `arith.bitcast` requires equal total bit width; it does not numerically convert.
- Most operations work elementwise on vectors and tensors, so a scalar-looking op name can still represent many lane operations.
- The dialect does not support manipulating `i0` values.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/Arith/IR/ArithBase.td`
- `mlir/include/mlir/Dialect/Arith/IR/ArithOps.td`
- `mlir/include/mlir/Dialect/Arith/IR/ArithOpsInterfaces.td`
- `mlir/lib/Dialect/Arith/IR/ArithOps.cpp`
- `mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td`
- `mlir/include/mlir/Dialect/Arith/Transforms/Passes.td`
- `mlir/lib/Dialect/Arith/Transforms/`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/ArithToLLVM/`
- `mlir/lib/Conversion/ArithToSPIRV/`
- `mlir/lib/Conversion/ArithToEmitC/`
- `mlir/lib/Conversion/ArithToAMDGPU/`
- `mlir/lib/Conversion/ArithToArmSME/`
- `mlir/lib/Conversion/ArithToAPFloat/`
- `mlir/lib/Conversion/TosaToArith/`

Useful tests:

- `mlir/test/Dialect/Arith/ops.mlir`
- `mlir/test/Dialect/Arith/canonicalize.mlir`
- `mlir/test/Dialect/Arith/constant-fold.mlir`
- `mlir/test/Dialect/Arith/expand-ops.mlir`
- `mlir/test/Dialect/Arith/int-range-opts.mlir`
- `mlir/test/Dialect/Arith/emulate-wide-int.mlir`
- `mlir/test/Dialect/Arith/emulate-unsupported-floats.mlir`
- `mlir/test/Conversion/ArithToLLVM/`
- `mlir/test/Conversion/ArithToSPIRV/`
- `mlir/test/Conversion/ArithToEmitC/`
