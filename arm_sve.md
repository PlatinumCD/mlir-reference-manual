# Arm SVE Dialect

## Beginner Summary

The `arm_sve` dialect models operations for Arm SVE, the Arm Scalable Vector
Extension. SVE is different from fixed-width SIMD because the hardware vector
length is not baked into the program. The same program can run on machines with
different SVE vector lengths.

In MLIR, SVE values are usually represented with scalable vector types such as
`vector<[4]xf32>` or `vector<[16]xi8>`. The square brackets mean the dimension
is scalable. For example, `vector<[4]xf32>` means "4 times vscale lanes of
f32", where the runtime hardware determines `vscale`.

The `arm_sve` dialect is not a general-purpose vector dialect. It is a
target-specific dialect used when the compiler has decided that certain vector
operations should map to Arm SVE instructions or Arm SVE LLVM intrinsics.

## Why This Dialect Exists

The generic `vector` dialect can express scalable vectors, transfers, masks,
and contractions without committing to one hardware instruction set. That is
good for target-independent optimization, but eventually a compiler must choose
target instructions.

The `arm_sve` dialect exists for that target-specific stage. It gives MLIR
operations for SVE concepts that are awkward or too specific to keep in the
generic `vector` dialect:

- Masked scalable arithmetic where inactive lanes keep the first operand.
- SVE predicate storage rules around `svbool`.
- Integer and BF16 matrix multiply-accumulate instructions.
- Predicate selection and multi-vector zip operations used by SVE and SME
  lowerings.
- LLVM-intrinsic-like Arm SVE operations used near LLVM export.

## When It Matters

Arm SVE matters when compiling vector code for AArch64 targets with scalable
vector support.

Use `arm_sve` when:

- You are lowering generic `vector` IR for an SVE-capable Arm target.
- You need SVE-specific masked arithmetic semantics.
- You need I8MM or BF16 matrix multiply-accumulate instructions.
- You need to legalize SVE predicate loads, stores, and stack allocations before
  LLVM lowering.
- You are reading late-stage MLIR close to LLVM export and see
  `arm_sve.intr.*` operations.

Avoid writing it too early when:

- The IR can still be optimized in target-independent `vector` form.
- The target may not be Arm SVE.
- The operation is better represented as `arith`, `math`, `vector`, `linalg`,
  or `arm_sme` until a lower stage.

## What It Means

Seeing `arm_sve` means the compiler is specializing vector IR for Arm scalable
vector hardware.

The main implication is that vector length is runtime-scalable. Code should not
assume a fixed number of lanes. Instead, it should use scalable vector types and
predicate masks.

There are two layers of operations:

- User-facing `arm_sve.*` operations such as `arm_sve.sdot`,
  `arm_sve.masked.addf`, and `arm_sve.convert_to_svbool`.
- Lower-level `arm_sve.intr.*` operations that map closely to LLVM Arm SVE
  intrinsics.

Most compiler pipelines prefer the first layer until LLVM export. The
intrinsic layer is the bridge to LLVM.

## Core Concepts

### Scalable Vector Types

SVE values use scalable vector types from the builtin/vector type system:

- `vector<[16]xi8>`: scalable vector of i8 with base size 16.
- `vector<[8]xi16>`: scalable vector of i16 with base size 8.
- `vector<[4]xf32>`: scalable vector of f32 with base size 4.
- `vector<[2]xi64>`: scalable vector of i64 with base size 2.

The bracketed dimension is multiplied by `vscale` at runtime.

### Predicates And svbool

SVE predicates are masks. MLIR models SVE predicate masks as scalable vectors of
`i1`.

The full SVE predicate register storage type is called `svbool`. In MLIR, this
is represented as:

```text
vector<[16]xi1>
```

Smaller predicate masks are legal SVE predicate values, for example:

```text
vector<[1]xi1>
vector<[2]xi1>
vector<[4]xi1>
vector<[8]xi1>
```

However, smaller predicates cannot always be loaded and stored directly by the
LLVM backend. The dialect therefore has conversion operations:

- `arm_sve.convert_to_svbool`
- `arm_sve.convert_from_svbool`

These are critical for legal memory representation of predicate values.

### Friendly Ops And Intrinsic Ops

Many `arm_sve` operations have a friendlier operation and a corresponding
intrinsic operation:

- `arm_sve.sdot` lowers toward `arm_sve.intr.sdot`.
- `arm_sve.masked.addi` lowers toward `arm_sve.intr.add`.
- `arm_sve.convert_to_svbool` lowers toward
  `arm_sve.intr.convert.to.svbool`.
- `arm_sve.zip.x2` lowers toward `arm_sve.intr.zip.x2`.

This distinction is useful when reading IR. If you see `arm_sve.intr.*`, the
IR is already close to LLVM translation.

## Operations

Arm SVE defines 42 operations.

### Predicate And Storage Operations

- `arm_sve.convert_to_svbool`: Converts an SVE predicate mask such as
  `vector<[4]xi1>` or `vector<2x[8]xi1>` to a full `svbool` shape with trailing
  scalable dimension `[16]`.
- `arm_sve.convert_from_svbool`: Converts a full `svbool` mask back to a
  smaller SVE predicate type.
- `arm_sve.psel`: Selects predicate `p1` or an all-false predicate based on one
  bit of another predicate. This requires SME or SVE2.1 target support.
- `arm_sve.dupq_lane`: Broadcasts one indexed 128-bit segment of a scalable
  vector across the result vector.

### Masked Arithmetic Operations

These operations perform arithmetic on active lanes and keep the first operand
on inactive lanes.

- `arm_sve.masked.addf`: Masked floating-point add.
- `arm_sve.masked.subf`: Masked floating-point subtract.
- `arm_sve.masked.mulf`: Masked floating-point multiply.
- `arm_sve.masked.divf`: Masked floating-point divide.
- `arm_sve.masked.addi`: Masked integer add.
- `arm_sve.masked.subi`: Masked integer subtract.
- `arm_sve.masked.muli`: Masked integer multiply.
- `arm_sve.masked.divi_signed`: Masked signed integer divide.
- `arm_sve.masked.divi_unsigned`: Masked unsigned integer divide.

These are SVE-specific because inactive lane behavior is part of the operation.

### Dot Product And Matrix Multiply Operations

- `arm_sve.sdot`: Signed integer dot product and accumulate. It interprets
  signless integer operands as signed.
- `arm_sve.udot`: Unsigned integer dot product and accumulate. It interprets
  signless integer operands as unsigned.
- `arm_sve.smmla`: Signed i8 matrix multiply-accumulate.
- `arm_sve.ummla`: Unsigned i8 matrix multiply-accumulate.
- `arm_sve.usmmla`: Unsigned-by-signed i8 matrix multiply-accumulate.
- `arm_sve.intr.bfmmla`: BF16 matrix multiply-accumulate into f32
  accumulators.

The integer MMLA operations are important for Arm FEAT_I8MM. The BF16 operation
is important for Arm FEAT_BF16.

### Multi-Vector Operations

- `arm_sve.zip.x2`: Interleaves two scalable vectors and returns two scalable
  vectors.
- `arm_sve.zip.x4`: Interleaves four scalable vectors and returns four
  scalable vectors.

These operations map to SME2-style multi-vector zip instructions and are useful
in tiling and matrix lowering code.

### Intrinsic Operations

The `arm_sve.intr.*` operations are LLVM-intrinsic-facing operations. They are
usually produced by Arm SVE legalization patterns rather than hand-written in
high-level IR.

- `arm_sve.intr.add`: Masked integer add intrinsic.
- `arm_sve.intr.sub`: Masked integer subtract intrinsic.
- `arm_sve.intr.mul`: Masked integer multiply intrinsic.
- `arm_sve.intr.sdiv`: Masked signed integer divide intrinsic.
- `arm_sve.intr.udiv`: Masked unsigned integer divide intrinsic.
- `arm_sve.intr.fadd`: Masked floating-point add intrinsic.
- `arm_sve.intr.fsub`: Masked floating-point subtract intrinsic.
- `arm_sve.intr.fmul`: Masked floating-point multiply intrinsic.
- `arm_sve.intr.fdiv`: Masked floating-point divide intrinsic.
- `arm_sve.intr.sdot`: Signed dot product intrinsic.
- `arm_sve.intr.udot`: Unsigned dot product intrinsic.
- `arm_sve.intr.smmla`: Signed integer matrix multiply-accumulate intrinsic.
- `arm_sve.intr.ummla`: Unsigned integer matrix multiply-accumulate intrinsic.
- `arm_sve.intr.usmmla`: Unsigned-by-signed matrix multiply-accumulate
  intrinsic.
- `arm_sve.intr.bfmmla`: BF16 matrix multiply-accumulate intrinsic.
- `arm_sve.intr.convert.to.svbool`: Converts a predicate to `svbool` for LLVM
  lowering.
- `arm_sve.intr.convert.from.svbool`: Converts from `svbool` for LLVM lowering.
- `arm_sve.intr.psel`: Predicate select intrinsic.
- `arm_sve.intr.whilelt`: Creates a predicate for lanes below a bound.
- `arm_sve.intr.dupq_lane`: 128-bit segment broadcast intrinsic.
- `arm_sve.intr.zip.x2`: Two-way zip intrinsic.
- `arm_sve.intr.zip.x4`: Four-way zip intrinsic.

## Types And Attributes

The dialect does not define custom MLIR types or attributes.

Instead, it constrains ordinary MLIR vector types:

- SVE data vectors use a single scalable dimension and element sizes that match
  SVE vector shapes.
- SVE predicates use scalable `i1` vectors with base sizes `1`, `2`, `4`, `8`,
  or `16`.
- `svbool` is represented as `vector<[16]xi1>`.

Operation-specific attributes include:

- `arm_sve.dupq_lane` and `arm_sve.intr.dupq_lane` have a `lane` integer
  attribute.

Most other operation properties are encoded in operand and result types.

## Transformations

### Arm SVE Passes

- `arm-sve-legalize-vector-storage`: Ensures SVE vector loads, stores, and
  allocations are legal for LLVM lowering.

This pass runs at the memref level and should run before lowering all the way
to LLVM. It addresses two practical backend issues:

- Smaller SVE predicates such as `vector<[4]xi1>` are widened to
  `vector<[16]xi1>` for storage, with `arm_sve.convert_to_svbool` and
  `arm_sve.convert_from_svbool` inserted around stores and loads.
- SVE vector stack allocations get relaxed alignment so LLVM does not request
  an unsupported over-aligned stack slot.

### Transform Dialect Hooks

Arm SVE also provides transform dialect pattern descriptors:

- `transform.apply_patterns.arm_sve.vector_contract_to_i8mm`: Requests
  lowering of matching `vector.contract` operations to Arm SVE operations that
  map to FEAT_I8MM instructions.
- `transform.apply_patterns.arm_sve.vector_contract_to_bfmmla`: Requests
  lowering of matching `vector.contract` operations to Arm SVE operations that
  map to FEAT_BF16 instructions.

These are not normal IR operations. They are transform script operations that
collect rewrite patterns for the transform interpreter.

### Pattern APIs

The C++ transformation entry points are:

- `populateArmSVELegalizeForLLVMExportPatterns`: Rewrites user-facing Arm SVE
  ops to LLVM-intrinsic-facing Arm SVE ops and lowers suitable
  `vector.create_mask` operations to `arm_sve.intr.whilelt`.
- `configureArmSVELegalizeForExportTarget`: Marks the user-facing Arm SVE ops
  illegal and the intrinsic-facing Arm SVE ops legal for LLVM export.
- `populateLowerContractionToSVEI8MMPatterns`: Lowers suitable
  `vector.contract` operations to SVE I8MM operations.
- `populateLowerContractionToSVEBFMMLAPatterns`: Lowers suitable
  `vector.contract` operations to SVE BF16 matrix multiply operations.
- `populateLegalizeVectorStoragePatterns`: Populates the storage legalization
  patterns used by `arm-sve-legalize-vector-storage`.

## Conversions And Lowering Paths

Arm SVE does not have a standalone `convert-arm-sve-to-llvm` pass. Its LLVM
lowering path is integrated with vector-to-LLVM conversion.

### Producing Arm SVE

Common producers include:

- `convert-vector-to-llvm` with `enable-arm-sve`: Enables Arm SVE dialect use
  while lowering the `vector` dialect.
- `convert-vector-to-llvm` with `enable-arm-i8mm`: Enables Arm FEAT_I8MM
  lowering for suitable vector contractions.
- `convert-vector-to-llvm` with `enable-arm-bf16`: Enables Arm FEAT_BF16
  lowering for suitable vector contractions.
- `convert-vector-to-arm-sme`: Lowers vector operations to ArmSME and may also
  use ArmSVE operations such as `arm_sve.psel`.
- Transform dialect pattern collection through
  `transform.apply_patterns.arm_sve.vector_contract_to_i8mm` and
  `transform.apply_patterns.arm_sve.vector_contract_to_bfmmla`.

### Lowering Arm SVE

The usual path is:

```text
vector/linalg/tensor-level IR
  -> vector operations with scalable vectors
  -> Arm SVE-specific operations
  -> arm-sve-legalize-vector-storage
  -> convert-vector-to-llvm with enable-arm-sve
  -> Arm SVE intrinsic ops and LLVM dialect
  -> LLVM IR translation
```

In practical `mlir-opt` form, a late pipeline may contain pieces like:

```text
arm-sve-legalize-vector-storage
convert-vector-to-llvm{enable-arm-sve}
convert-func-to-llvm
convert-arith-to-llvm
reconcile-unrealized-casts
```

The exact pipeline depends on the surrounding dialects.

## Example IR

### Signed Dot Product

```mlir
module {
  func.func @signed_dot(%a: vector<[16]xi8>,
                        %b: vector<[16]xi8>,
                        %acc: vector<[4]xi32>) -> vector<[4]xi32> {
    %result = arm_sve.sdot %acc, %a, %b
        : vector<[16]xi8> to vector<[4]xi32>
    return %result : vector<[4]xi32>
  }
}
```

This operation accumulates signed dot products from i8 lanes into i32 lanes.

### Masked Floating-Point Arithmetic

```mlir
module {
  func.func @masked_float(%mask: vector<[4]xi1>,
                          %a: vector<[4]xf32>,
                          %b: vector<[4]xf32>,
                          %c: vector<[4]xf32>) -> vector<[4]xf32> {
    %sum = arm_sve.masked.addf %mask, %a, %b
        : vector<[4]xi1>, vector<[4]xf32>
    %product = arm_sve.masked.mulf %mask, %sum, %c
        : vector<[4]xi1>, vector<[4]xf32>
    return %product : vector<[4]xf32>
  }
}
```

Inactive lanes keep the first arithmetic operand, not an arbitrary passthrough
value.

### Predicate Conversion For Storage

```mlir
module {
  func.func @predicate_round_trip(%mask: vector<[4]xi1>) -> vector<[4]xi1> {
    %svbool = arm_sve.convert_to_svbool %mask : vector<[4]xi1>
    %restored = arm_sve.convert_from_svbool %svbool : vector<[4]xi1>
    return %restored : vector<[4]xi1>
  }
}
```

This is the key idea behind legalizing smaller SVE predicates for memory.

### Zip And Predicate Select

```mlir
module {
  func.func @zip_and_select(%a: vector<[8]xi16>,
                            %b: vector<[8]xi16>,
                            %p1: vector<[4]xi1>,
                            %p2: vector<[8]xi1>,
                            %index: index)
      -> (vector<[8]xi16>, vector<[4]xi1>) {
    %lo, %hi = arm_sve.zip.x2 %a, %b : vector<[8]xi16>
    %pred = arm_sve.psel %p1, %p2[%index]
        : vector<[4]xi1>, vector<[8]xi1>
    return %lo, %pred : vector<[8]xi16>, vector<[4]xi1>
  }
}
```

`arm_sve.psel` is useful in lowerings that need to select a predicate slice.

### BF16 Matrix Multiply-Accumulate Intrinsic

```mlir
module {
  func.func @bf16_mmla(%a: vector<[8]xbf16>,
                       %b: vector<[8]xbf16>,
                       %acc: vector<[4]xf32>) -> vector<[4]xf32> {
    %result = arm_sve.intr.bfmmla %acc, %a, %b
        : vector<[8]xbf16> to vector<[4]xf32>
    return %result : vector<[4]xf32>
  }
}
```

This op is already in the intrinsic-facing layer.

## How To Use It

As a beginner, do not start a compiler pipeline by inventing `arm_sve` IR. Start
from `linalg`, `vector`, or another target-independent dialect. Let target
selection and lowering decide when Arm SVE is appropriate.

Use direct Arm SVE ops when writing tests or target-specific lowering patterns:

- Use `arm_sve.sdot` or `arm_sve.udot` for SVE dot-product instructions.
- Use `arm_sve.smmla`, `arm_sve.ummla`, or `arm_sve.usmmla` for I8MM matrix
  instructions.
- Use `arm_sve.intr.bfmmla` for BF16 matrix multiply-accumulate.
- Use `arm_sve.masked.*` operations when inactive lanes should preserve the
  first operand.
- Use `arm_sve.convert_to_svbool` and `arm_sve.convert_from_svbool` around
  predicate storage.

When lowering to LLVM, enable Arm SVE support through vector-to-LLVM options
and run storage legalization before final LLVM conversion if predicates or
SVE-vector memory are involved.

## Gotchas

- `vector<[4]xf32>` is not four lanes. It is `4 * vscale` lanes.
- `vector<[16]xi1>` has special meaning as `svbool`, the full predicate
  storage shape.
- Smaller predicates such as `vector<[4]xi1>` are valid predicate values but
  need legalization for memory.
- Only the trailing dimension may be scalable for the predicate conversion
  operations.
- `arm_sve.masked.*` inactive lanes keep the first operand.
- `arm_sve.sdot` and `arm_sve.smmla` interpret signless operands as signed.
- `arm_sve.udot` and `arm_sve.ummla` interpret signless operands as unsigned.
- `arm_sve.usmmla` is unsigned-by-signed.
- `arm_sve.psel` is in the ArmSVE dialect but requires SME or SVE2.1 target
  features.
- `arm_sve.zip.x2` and `arm_sve.zip.x4` require SME2 target support.
- `arm_sve.intr.*` ops are late-lowering artifacts. Seeing them early usually
  means the IR has become target-specific too soon.

## What It Implies In A Compiler Pipeline

Arm SVE indicates target commitment. Once IR contains `arm_sve`, generic vector
optimization may no longer apply cleanly because the operations encode specific
Arm hardware semantics.

The best mental model is:

```text
generic vector shape and math
  -> target-specific Arm SVE operation choice
  -> SVE storage and predicate legalization
  -> LLVM intrinsic-facing Arm SVE ops
  -> LLVM IR
```

If a pass fails after `arm_sve` appears, check target features and storage
legality first. Many failures are not about the arithmetic operation itself,
but about whether the predicate or scalable vector type can be represented
legally at the next lowering stage.

## Source Map

- `mlir/include/mlir/Dialect/ArmSVE/IR/ArmSVE.td`: Dialect definition,
  constraints, user-facing ops, and intrinsic-facing ops.
- `mlir/lib/Dialect/ArmSVE/IR/ArmSVEDialect.cpp`: Dialect registration and
  generated op/type hookup.
- `mlir/include/mlir/Dialect/ArmSVE/Transforms/Passes.td`: Arm SVE pass
  definitions.
- `mlir/lib/Dialect/ArmSVE/Transforms/LegalizeVectorStorage.cpp`: Predicate and
  scalable-vector storage legalization.
- `mlir/lib/Dialect/ArmSVE/Transforms/LegalizeForLLVMExport.cpp`: Rewrites from
  user-facing Arm SVE ops to intrinsic-facing ops for LLVM export.
- `mlir/lib/Dialect/ArmSVE/Transforms/LowerContractToSVEPatterns.cpp`:
  `vector.contract` lowering to I8MM and BF16 SVE operations.
- `mlir/include/mlir/Dialect/ArmSVE/Transforms/Transforms.h`: Pattern
  population and LLVM export target configuration APIs.
- `mlir/include/mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.td`:
  Transform dialect pattern descriptor operations.
- `mlir/include/mlir/Conversion/Passes.td`: `convert-vector-to-llvm`,
  `convert-vector-to-arm-sme`, and Arm feature options that interact with
  ArmSVE lowering.
