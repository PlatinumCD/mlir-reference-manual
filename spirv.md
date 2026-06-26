# SPIR-V Dialect

## Beginner Summary

The `spirv` dialect is MLIR's representation of SPIR-V.

SPIR-V is a binary intermediate language used for graphics shaders and compute
kernels in Khronos ecosystems such as Vulkan, OpenCL, and OpenGL. In MLIR, the
`spirv` dialect gives that world a structured textual IR with MLIR parsing,
verification, transformation, and conversion support.

Think of `spirv` as the point where a compiler has stopped describing a generic
loop, tensor, GPU kernel, or vector operation and has started describing
something that should map closely to SPIR-V instructions.

The dialect is target-facing. It is still MLIR, but it is much closer to a
portable GPU/shader binary format than dialects like `linalg`, `tensor`, `scf`,
`vector`, or `gpu`.

## Why This Dialect Exists

MLIR needs a way to lower generic program structure into SPIR-V without losing
all compiler structure at once.

The `spirv` dialect exists to:

- Model SPIR-V instructions as MLIR operations.
- Model SPIR-V modules, functions, globals, entry points, execution modes,
  memory models, storage classes, decorations, and capabilities.
- Keep SPIR-V-specific validation visible in MLIR instead of waiting until
  binary serialization.
- Support transformations that are specific to Vulkan, OpenCL, WebGPU, GLSL,
  and SPIR-V extensions.
- Serialize to and deserialize from SPIR-V binary form.
- Provide a target for conversions from `arith`, `cf`, `func`, `gpu`, `index`,
  `math`, `memref`, `scf`, `tensor`, `vector`, `ub`, `complex`, and TOSA-related
  dialects.

The dialect mostly tracks the semantic level of the SPIR-V specification. Some
operations differ representationally where MLIR regions, types, attributes, or
symbols make the IR easier to analyze and rewrite.

## When It Matters

The `spirv` dialect matters late in compilation for SPIR-V targets.

Typical Vulkan or compute pipeline shape:

```text
linalg / tensor / scf / vector / gpu
  -> tiling, fusion, bufferization, and GPU mapping
  -> arith/cf/func/gpu/index/math/memref/scf/vector conversions
  -> spirv.module with storage classes, entry points, and capabilities
  -> SPIR-V-specific cleanup, ABI lowering, layout decoration, and VCE update
  -> SPIR-V binary serialization or further lowering when appropriate
```

It also matters when debugging why a Vulkan, OpenCL, or WebGPU target accepts or
rejects an operation. In SPIR-V, type bitwidths, storage classes, capabilities,
extensions, addressing model, memory model, and client API rules are part of
the contract.

## When To Use It

Use `spirv` when your compiler pipeline is intentionally targeting SPIR-V or a
SPIR-V-based environment.

Use it for:

- Device shader or compute code for Vulkan, OpenCL, OpenGL, or WebGPU-related
  flows.
- Inspecting the result of GPU-to-SPIR-V or vector-to-SPIR-V lowering.
- Representing SPIR-V memory, pointer, global variable, entry point, execution
  mode, and storage class decisions.
- Representing SPIR-V arithmetic, logical, bit, control-flow, atomic, image,
  group, subgroup, cooperative matrix, and extension operations.
- Running SPIR-V-specific cleanup passes before serialization.
- Lowering SPIR-V into LLVM dialect for supported host/runtime-oriented flows.

Do not use `spirv` as the first IR for normal compiler optimization. For most
programs, start in higher-level dialects and let conversion passes introduce
SPIR-V when the target constraints are known.

## Core Concepts

### Modules, Addressing Model, And Memory Model

A full SPIR-V program is represented with `spirv.module`.

The module carries the addressing model and memory model, for example:

```mlir
spirv.module Logical GLSL450 {
  // SPIR-V functions, globals, entry points, and execution modes live here.
}
```

These choices are not decoration. They affect what pointers, storage classes,
memory operations, and capabilities are legal.

### Target Environment And Availability

SPIR-V has versions, capabilities, extensions, and client API limits. MLIR
models this through target environment attributes such as `spirv.target_env`.

Many operations and enum cases carry availability rules. An operation may be
valid for one SPIR-V version or capability set and invalid for another. The
`spirv-update-vce` pass deduces minimal version, capability, and extension
requirements for a `spirv.module` under a target environment.

### Storage Classes

SPIR-V pointers include a storage class:

```text
!spirv.ptr<f32, Function>
!spirv.ptr<!spirv.struct<(f32, i32)>, StorageBuffer>
```

Storage classes describe where an object lives and how it is accessed. They are
central to `spirv.Load`, `spirv.Store`, `spirv.AccessChain`,
`spirv.GlobalVariable`, and ABI lowering.

The `map-memref-spirv-storage-class` pass maps numeric MLIR memref spaces to
SPIR-V storage classes for a client API such as Vulkan.

### Types

The `spirv` dialect uses MLIR builtin scalar and vector types where possible,
plus SPIR-V-specific types:

- `!spirv.ptr<type, StorageClass>` for typed SPIR-V pointers.
- `!spirv.array<N x T>` for fixed-size arrays, optionally with layout stride.
- `!spirv.rtarray<T>` for runtime arrays.
- `!spirv.struct<(...)>` for literal structs and identified structs.
- `!spirv.image<...>` for images.
- `!spirv.sampled_image<...>` and `!spirv.sampler` for sampled image flows.
- `!spirv.named_barrier` for named barriers.
- `!spirv.matrix<...>` for SPIR-V matrix values.
- `!spirv.coopmatrix<...>` for cooperative matrix extension values.
- `!spirv.tensorArm<...>` for ARM graph/tensor extension flows.

Types are constrained by SPIR-V rules. For example, vectors have allowed
lengths, integer and floating-point bitwidths depend on capabilities, and some
types are only legal in specific storage classes or extensions.

### Structured Control Flow

SPIR-V has structured control-flow requirements. MLIR represents some SPIR-V
structure directly with operations such as `spirv.mlir.selection`,
`spirv.mlir.loop`, `spirv.mlir.merge`, and `spirv.mlir.yield`.

Lowering from `scf` or `cf` to `spirv` must respect these rules. `scf` ops that
yield values may need SPIR-V variables plus loads and stores because SPIR-V
structured control flow does not yield values the same way MLIR `scf` does.

### ABI And Interface Variables

SPIR-V entry functions cannot use ordinary function parameters in the same way
as high-level MLIR functions. For GPU lowering, resource arguments often become
global variables with descriptor sets and bindings.

MLIR uses attributes such as `spirv.interface_var_abi` and
`spirv.entry_point_abi` during lowering. The `spirv-lower-abi-attrs` pass turns
those attributes into `spirv.GlobalVariable`, `spirv.EntryPoint`, and
`spirv.ExecutionMode` operations.

## Operations

The `spirv` dialect is large because SPIR-V is large. It includes core
instructions, extended instruction sets, and vendor or extension operations.

The most useful beginner grouping is:

- Structure and module operations: `spirv.module`, `spirv.func`,
  `spirv.EntryPoint`, `spirv.ExecutionMode`, `spirv.ExecutionModeId`,
  `spirv.FunctionCall`, `spirv.Return`, `spirv.ReturnValue`, and
  `spirv.Unreachable`.
- Constants and composites: `spirv.Constant`, `spirv.SpecConstant`,
  `spirv.SpecConstantComposite`, `spirv.SpecConstantOperation`,
  `spirv.CompositeConstruct`, `spirv.CompositeExtract`, and
  `spirv.CompositeInsert`.
- Arithmetic, bit, logical, and conversion operations: integer, floating-point,
  comparison, bitfield, cast, dot-product, and select operations.
- Memory and pointer operations: `spirv.Variable`, `spirv.GlobalVariable`,
  `spirv.Load`, `spirv.Store`, `spirv.CopyMemory`, `spirv.AccessChain`,
  `spirv.PtrAccessChain`, `spirv.InBoundsPtrAccessChain`,
  `spirv.mlir.addressof`, and `spirv.mlir.referenceof`.
- Atomics and synchronization: `spirv.Atomic*`, `spirv.ControlBarrier`,
  `spirv.MemoryBarrier`, named-barrier ops, and Intel split barrier ops.
- Group and subgroup operations: `spirv.Group*` and
  `spirv.GroupNonUniform*`.
- Image and sampling operations: `spirv.Image`, `spirv.SampledImage`,
  `spirv.ImageRead`, `spirv.ImageWrite`, `spirv.ImageFetch`, image sampling,
  gathering, and query operations.
- Vector and matrix operations: vector dynamic extract/insert/shuffle, matrix
  multiply variants, transposition, outer product, cooperative matrix KHR ops,
  and integer dot-product ops.
- Extended instruction set operations: `spirv.GL.*` for GLSL extended
  instructions and `spirv.CL.*` for OpenCL extended instructions.
- ML and graph extension operations: `spirv.Tosa.*`, `spirv.ARM.*`, and
  `spirv.ExperimentalML.Call`.
- Vendor and extension operations: `spirv.EXT.*`, `spirv.KHR.*`, and
  `spirv.INTEL.*`.

### Complete Generated Operation Inventory

The generated operation list in this LLVM checkout is:

`spirv.AccessChain`, `spirv.All`, `spirv.Any`, `spirv.ARM.Graph`, `spirv.ARM.GraphConstant`
`spirv.ARM.GraphEntryPoint`, `spirv.ARM.GraphOutputs`, `spirv.AtomicAnd`
`spirv.AtomicCompareExchange`, `spirv.AtomicCompareExchangeWeak`, `spirv.AtomicExchange`
`spirv.AtomicIAdd`, `spirv.AtomicIDecrement`, `spirv.AtomicIIncrement`, `spirv.AtomicISub`
`spirv.AtomicLoad`, `spirv.AtomicOr`, `spirv.AtomicSMax`, `spirv.AtomicSMin`
`spirv.AtomicStore`, `spirv.AtomicUMax`, `spirv.AtomicUMin`, `spirv.AtomicXor`, `spirv.Bitcast`
`spirv.BitCount`, `spirv.BitFieldInsert`, `spirv.BitFieldSExtract`, `spirv.BitFieldUExtract`
`spirv.BitReverse`, `spirv.BitwiseAnd`, `spirv.BitwiseOr`, `spirv.BitwiseXor`, `spirv.Branch`
`spirv.BranchConditional`, `spirv.CL.acos`, `spirv.CL.acosh`, `spirv.CL.asin`, `spirv.CL.asinh`
`spirv.CL.atan`, `spirv.CL.atan2`, `spirv.CL.atanh`, `spirv.CL.cbrt`, `spirv.CL.ceil`
`spirv.CL.clz`, `spirv.CL.cos`, `spirv.CL.cosh`, `spirv.CL.erf`, `spirv.CL.erfc`, `spirv.CL.exp`
`spirv.CL.exp10`, `spirv.CL.exp2`, `spirv.CL.expm1`, `spirv.CL.fabs`, `spirv.CL.floor`
`spirv.CL.fma`, `spirv.CL.fmax`, `spirv.CL.fmin`, `spirv.CL.ldexp`, `spirv.CL.log`
`spirv.CL.log10`, `spirv.CL.log1p`, `spirv.CL.log2`, `spirv.CL.mix`, `spirv.CL.pow`
`spirv.CL.pown`, `spirv.CL.printf`, `spirv.CL.rint`, `spirv.CL.rootn`, `spirv.CL.round`
`spirv.CL.rsqrt`, `spirv.CL.s_abs`, `spirv.CL.s_max`, `spirv.CL.s_min`, `spirv.CL.sin`
`spirv.CL.sinh`, `spirv.CL.sqrt`, `spirv.CL.tan`, `spirv.CL.tanh`, `spirv.CL.trunc`
`spirv.CL.u_max`, `spirv.CL.u_min`, `spirv.CompositeConstruct`, `spirv.CompositeExtract`
`spirv.CompositeInsert`, `spirv.Constant`, `spirv.ControlBarrier`, `spirv.ConvertFToS`
`spirv.ConvertFToU`, `spirv.ConvertPtrToU`, `spirv.ConvertSToF`, `spirv.ConvertUToF`
`spirv.ConvertUToPtr`, `spirv.CopyMemory`, `spirv.Dot`, `spirv.EmitVertex`, `spirv.EndPrimitive`
`spirv.EntryPoint`, `spirv.ExecutionMode`, `spirv.ExecutionModeId`, `spirv.ExperimentalML.Call`
`spirv.EXT.AtomicFAdd`, `spirv.EXT.ConstantCompositeReplicate`, `spirv.EXT.EmitMeshTasks`
`spirv.EXT.SetMeshOutputs`, `spirv.EXT.SpecConstantCompositeReplicate`, `spirv.FAdd`
`spirv.FConvert`, `spirv.FDiv`, `spirv.FMod`, `spirv.FMul`, `spirv.FNegate`, `spirv.FOrdEqual`
`spirv.FOrdGreaterThan`, `spirv.FOrdGreaterThanEqual`, `spirv.FOrdLessThan`
`spirv.FOrdLessThanEqual`, `spirv.FOrdNotEqual`, `spirv.FRem`, `spirv.FSub`, `spirv.func`
`spirv.FunctionCall`, `spirv.FUnordEqual`, `spirv.FUnordGreaterThan`
`spirv.FUnordGreaterThanEqual`, `spirv.FUnordLessThan`, `spirv.FUnordLessThanEqual`
`spirv.FUnordNotEqual`, `spirv.GenericCastToPtr`, `spirv.GenericCastToPtrExplicit`
`spirv.GL.Acos`, `spirv.GL.Acosh`, `spirv.GL.Asin`, `spirv.GL.Asinh`, `spirv.GL.Atan`
`spirv.GL.Atan2`, `spirv.GL.Atanh`, `spirv.GL.Ceil`, `spirv.GL.Cos`, `spirv.GL.Cosh`
`spirv.GL.Cross`, `spirv.GL.Degrees`, `spirv.GL.Distance`, `spirv.GL.Exp`, `spirv.GL.Exp2`
`spirv.GL.FAbs`, `spirv.GL.FClamp`, `spirv.GL.FindILsb`, `spirv.GL.FindSMsb`
`spirv.GL.FindUMsb`, `spirv.GL.Floor`, `spirv.GL.Fma`, `spirv.GL.FMax`, `spirv.GL.FMin`
`spirv.GL.FMix`, `spirv.GL.Fract`, `spirv.GL.FrexpStruct`, `spirv.GL.FSign`
`spirv.GL.InverseSqrt`, `spirv.GL.Ldexp`, `spirv.GL.Length`, `spirv.GL.Log`, `spirv.GL.Log2`
`spirv.GL.Normalize`, `spirv.GL.PackHalf2x16`, `spirv.GL.PackSnorm4x8`, `spirv.GL.Pow`
`spirv.GL.Radians`, `spirv.GL.Reflect`, `spirv.GL.Round`, `spirv.GL.RoundEven`, `spirv.GL.SAbs`
`spirv.GL.SClamp`, `spirv.GL.Sin`, `spirv.GL.Sinh`, `spirv.GL.SMax`, `spirv.GL.SMin`
`spirv.GL.Sqrt`, `spirv.GL.SSign`, `spirv.GL.Tan`, `spirv.GL.Tanh`, `spirv.GL.Trunc`
`spirv.GL.UClamp`, `spirv.GL.UMax`, `spirv.GL.UMin`, `spirv.GL.UnpackHalf2x16`
`spirv.GL.UnpackSnorm4x8`, `spirv.GlobalVariable`, `spirv.GroupBroadcast`, `spirv.GroupFAdd`
`spirv.GroupFMax`, `spirv.GroupFMin`, `spirv.GroupIAdd`, `spirv.GroupNonUniformAll`
`spirv.GroupNonUniformAllEqual`, `spirv.GroupNonUniformAny`, `spirv.GroupNonUniformBallot`
`spirv.GroupNonUniformBallotBitCount`, `spirv.GroupNonUniformBallotFindLSB`
`spirv.GroupNonUniformBallotFindMSB`, `spirv.GroupNonUniformBitwiseAnd`
`spirv.GroupNonUniformBitwiseOr`, `spirv.GroupNonUniformBitwiseXor`
`spirv.GroupNonUniformBroadcast`, `spirv.GroupNonUniformBroadcastFirst`
`spirv.GroupNonUniformElect`, `spirv.GroupNonUniformFAdd`, `spirv.GroupNonUniformFMax`
`spirv.GroupNonUniformFMin`, `spirv.GroupNonUniformFMul`, `spirv.GroupNonUniformIAdd`
`spirv.GroupNonUniformIMul`, `spirv.GroupNonUniformLogicalAnd`, `spirv.GroupNonUniformLogicalOr`
`spirv.GroupNonUniformLogicalXor`, `spirv.GroupNonUniformQuadSwap`
`spirv.GroupNonUniformRotateKHR`, `spirv.GroupNonUniformShuffle`
`spirv.GroupNonUniformShuffleDown`, `spirv.GroupNonUniformShuffleUp`
`spirv.GroupNonUniformShuffleXor`, `spirv.GroupNonUniformSMax`, `spirv.GroupNonUniformSMin`
`spirv.GroupNonUniformUMax`, `spirv.GroupNonUniformUMin`, `spirv.GroupSMax`, `spirv.GroupSMin`
`spirv.GroupUMax`, `spirv.GroupUMin`, `spirv.IAdd`, `spirv.IAddCarry`, `spirv.IEqual`
`spirv.Image`, `spirv.ImageDrefGather`, `spirv.ImageFetch`, `spirv.ImageQuerySize`
`spirv.ImageRead`, `spirv.ImageSampleExplicitLod`, `spirv.ImageSampleImplicitLod`
`spirv.ImageSampleProjDrefImplicitLod`, `spirv.ImageWrite`, `spirv.IMul`
`spirv.InBoundsPtrAccessChain`, `spirv.INotEqual`, `spirv.INTEL.ControlBarrierArrive`
`spirv.INTEL.ControlBarrierWait`, `spirv.INTEL.ConvertBF16ToF`, `spirv.INTEL.ConvertFToBF16`
`spirv.INTEL.MaskedGather`, `spirv.INTEL.MaskedScatter`, `spirv.INTEL.RoundFToTF32`
`spirv.INTEL.SubgroupBlockRead`, `spirv.INTEL.SubgroupBlockWrite`, `spirv.IsFinite`
`spirv.IsInf`, `spirv.IsNan`, `spirv.IsNormal`, `spirv.ISub`, `spirv.ISubBorrow`
`spirv.KHR.AssumeTrue`, `spirv.KHR.CooperativeMatrixLength`, `spirv.KHR.CooperativeMatrixLoad`
`spirv.KHR.CooperativeMatrixMulAdd`, `spirv.KHR.CooperativeMatrixStore`, `spirv.KHR.Expect`
`spirv.KHR.GroupFMul`, `spirv.KHR.GroupIMul`, `spirv.KHR.SubgroupBallot`, `spirv.Kill`
`spirv.Load`, `spirv.LogicalAnd`, `spirv.LogicalEqual`, `spirv.LogicalNot`
`spirv.LogicalNotEqual`, `spirv.LogicalOr`, `spirv.MatrixTimesMatrix`, `spirv.MatrixTimesScalar`
`spirv.MatrixTimesVector`, `spirv.MemoryBarrier`, `spirv.MemoryNamedBarrier`
`spirv.mlir.addressof`, `spirv.mlir.loop`, `spirv.mlir.merge`, `spirv.mlir.referenceof`
`spirv.mlir.selection`, `spirv.mlir.yield`, `spirv.module`, `spirv.NamedBarrierInitialize`
`spirv.Not`, `spirv.Ordered`, `spirv.OuterProduct`, `spirv.PtrAccessChain`
`spirv.PtrCastToGeneric`, `spirv.Return`, `spirv.ReturnValue`, `spirv.SampledImage`
`spirv.SConvert`, `spirv.SDiv`, `spirv.SDot`, `spirv.SDotAccSat`, `spirv.Select`
`spirv.SGreaterThan`, `spirv.SGreaterThanEqual`, `spirv.ShiftLeftLogical`
`spirv.ShiftRightArithmetic`, `spirv.ShiftRightLogical`, `spirv.SLessThan`
`spirv.SLessThanEqual`, `spirv.SMod`, `spirv.SMulExtended`, `spirv.SNegate`
`spirv.SpecConstant`, `spirv.SpecConstantComposite`, `spirv.SpecConstantOperation`, `spirv.SRem`
`spirv.Store`, `spirv.SUDot`, `spirv.SUDotAccSat`, `spirv.Switch`, `spirv.Tosa.Abs`
`spirv.Tosa.Add`, `spirv.Tosa.ArgMax`, `spirv.Tosa.ArithmeticRightShift`, `spirv.Tosa.AvgPool2D`
`spirv.Tosa.BitwiseAnd`, `spirv.Tosa.BitwiseNot`, `spirv.Tosa.BitwiseOr`
`spirv.Tosa.BitwiseXor`, `spirv.Tosa.Cast`, `spirv.Tosa.Ceil`, `spirv.Tosa.Clamp`
`spirv.Tosa.Clz`, `spirv.Tosa.Concat`, `spirv.Tosa.Conv2D`, `spirv.Tosa.Conv3D`
`spirv.Tosa.Cos`, `spirv.Tosa.DepthwiseConv2D`, `spirv.Tosa.Equal`, `spirv.Tosa.Erf`
`spirv.Tosa.Exp`, `spirv.Tosa.FFT2D`, `spirv.Tosa.Floor`, `spirv.Tosa.Gather`
`spirv.Tosa.Greater`, `spirv.Tosa.GreaterEqual`, `spirv.Tosa.IntDiv`, `spirv.Tosa.Log`
`spirv.Tosa.LogicalAnd`, `spirv.Tosa.LogicalLeftShift`, `spirv.Tosa.LogicalNot`
`spirv.Tosa.LogicalOr`, `spirv.Tosa.LogicalRightShift`, `spirv.Tosa.LogicalXor`
`spirv.Tosa.MatMul`, `spirv.Tosa.Maximum`, `spirv.Tosa.MaxPool2D`, `spirv.Tosa.Minimum`
`spirv.Tosa.Mul`, `spirv.Tosa.Negate`, `spirv.Tosa.Pad`, `spirv.Tosa.Pow`
`spirv.Tosa.Reciprocal`, `spirv.Tosa.ReduceAll`, `spirv.Tosa.ReduceAny`, `spirv.Tosa.ReduceMax`
`spirv.Tosa.ReduceMin`, `spirv.Tosa.ReduceProduct`, `spirv.Tosa.ReduceSum`, `spirv.Tosa.Rescale`
`spirv.Tosa.Reshape`, `spirv.Tosa.Resize`, `spirv.Tosa.Reverse`, `spirv.Tosa.RFFT2D`
`spirv.Tosa.Rsqrt`, `spirv.Tosa.Scatter`, `spirv.Tosa.Select`, `spirv.Tosa.Sigmoid`
`spirv.Tosa.Sin`, `spirv.Tosa.Slice`, `spirv.Tosa.Sub`, `spirv.Tosa.Table`, `spirv.Tosa.Tanh`
`spirv.Tosa.Tile`, `spirv.Tosa.Transpose`, `spirv.Tosa.TransposeConv2D`, `spirv.Transpose`
`spirv.UConvert`, `spirv.UDiv`, `spirv.UDot`, `spirv.UDotAccSat`, `spirv.UGreaterThan`
`spirv.UGreaterThanEqual`, `spirv.ULessThan`, `spirv.ULessThanEqual`, `spirv.UMod`
`spirv.UMulExtended`, `spirv.Undef`, `spirv.Unordered`, `spirv.Unreachable`, `spirv.Variable`
`spirv.VectorExtractDynamic`, `spirv.VectorInsertDynamic`, `spirv.VectorShuffle`
`spirv.VectorTimesMatrix`, `spirv.VectorTimesScalar`

## Transformations

SPIR-V-specific transformation passes:

- `spirv-canonicalize-gl`: runs canonicalization patterns involving GLSL
  extended instruction ops.
- `decorate-spirv-composite-type-layout`: attaches layout information to
  composite types used by storage classes such as `StorageBuffer`,
  `PhysicalStorageBuffer`, `Uniform`, and `PushConstant`. The current pass is
  aimed at Vulkan layout rules.
- `spirv-lower-abi-attrs`: lowers SPIR-V ABI attributes into global variables,
  entry points, and execution modes.
- `spirv-rewrite-inserts`: rewrites sequential `spirv.CompositeInsert` chains
  into `spirv.CompositeConstruct` operations.
- `spirv-unify-aliased-resource`: rewrites access to multiple aliased resources
  so they access one unified resource.
- `spirv-update-vce`: deduces and attaches minimal version, capability, and
  extension requirements to `spirv.module` ops.
- `spirv-webgpu-prepare`: prepares SPIR-V for WebGPU by expanding unsupported
  operations and replacing them with supported forms.
- `spirv-promote-to-replicated-constants`: converts splat composite constants
  and spec constants to replicated composite extension ops.

These passes are usually used after conversion into `spirv`, before binary
serialization or target-specific handoff.

## Conversions And Lowering Paths

Common paths into `spirv`:

- `convert-arith-to-spirv`: lowers arithmetic operations, with options to
  emulate narrow scalar and unsupported floating-point types.
- `convert-complex-to-spirv`: lowers supported complex operations.
- `convert-cf-to-spirv`: lowers ControlFlow dialect branch operations.
- `convert-func-to-spirv`: lowers Func dialect functions and calls.
- `convert-gpu-to-spirv`: lowers supported GPU device ops to SPIR-V. It does
  not lower GPU host ops. Resource arguments can become SPIR-V global variables
  with descriptor set and binding assignments.
- `convert-index-to-spirv`: lowers Index dialect ops using 32-bit or 64-bit
  integer representation.
- `convert-math-to-spirv`: lowers supported Math dialect ops to core SPIR-V,
  GLSL, or OpenCL extended instructions depending on patterns and target
  support.
- `map-memref-spirv-storage-class`: maps numeric memref memory spaces to
  SPIR-V storage classes for a client API.
- `convert-memref-to-spirv`: lowers MemRef dialect memory operations to SPIR-V
  pointer, load, store, variable, and access-chain forms.
- `convert-scf-to-spirv`: lowers structured control flow to SPIR-V structured
  control flow.
- `convert-tensor-to-spirv`: lowers supported tensor constants and tensor
  operations.
- `convert-ub-to-spirv`: lowers supported UB dialect operations.
- `convert-vector-to-spirv`: lowers supported vector operations, including
  vector reductions and dot-product-related patterns when legal.
- `tosa-to-spirv-tosa-mark-graph-constants`: marks large TOSA constants for
  SPIR-V Graph/TOSA lowering.
- `tosa-to-spirv-tosa`: lowers TOSA programs to SPIR-V Graph/TOSA operations
  such as `spirv.ARM.Graph` and `spirv.Tosa.*`.

Common paths out of `spirv`:

- SPIR-V binary serialization from `spirv.module`.
- `convert-spirv-to-llvm`: lowers supported SPIR-V dialect operations to LLVM
  dialect. It has a `client-api` option for deriving storage-class-to-address
  space mappings for APIs such as Vulkan, OpenCL, Metal, and WebGPU.

The main implication is that `spirv` is both a lowering target and a validation
surface. If a conversion fails, the reason is often not just "the pattern is
missing"; it may be that the target environment does not allow the operation,
type, storage class, extension, or capability.

## Example IR

### Arithmetic

```mlir
func.func @spirv_arithmetic(%x: i32, %y: i32) -> i32 {
  %sum = spirv.IAdd %x, %y : i32
  %twice = spirv.IMul %sum, %sum : i32
  return %twice : i32
}
```

This looks similar to ordinary arithmetic, but the operations are SPIR-V
instructions with SPIR-V type legality rules.

### Composite Construction

```mlir
func.func @spirv_composite(%x: f32, %y: f32, %z: f32) -> vector<3xf32> {
  %v = spirv.CompositeConstruct %x, %y, %z : (f32, f32, f32) -> vector<3xf32>
  return %v : vector<3xf32>
}
```

Composite operations build, extract, and update vectors, arrays, structs,
matrices, and other aggregate values.

### Function Memory

```mlir
func.func @spirv_memory(%value: f32) -> f32 {
  %ptr = spirv.Variable : !spirv.ptr<f32, Function>
  spirv.Store "Function" %ptr, %value : f32
  %loaded = spirv.Load "Function" %ptr : f32
  return %loaded : f32
}
```

The pointer type includes the `Function` storage class, and the load/store ops
spell out the storage class too.

### Branching

```mlir
func.func @spirv_branch() -> () {
  %true = spirv.Constant true
  spirv.BranchConditional %true, ^then, ^else
^then:
  spirv.Return
^else:
  spirv.Return
}
```

SPIR-V control flow has its own terminators and structured-control-flow rules.

### Module, Global Variable, And Access Chain

```mlir
spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
  spirv.func @access_chain() -> () "None" {
    %c1 = spirv.Constant 1 : i32
    %addr = spirv.mlir.addressof @var : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
    %field = spirv.AccessChain %addr[%c1, %c1] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>, i32, i32 -> !spirv.ptr<f32, Input>
    spirv.Return
  }
}
```

`spirv.AccessChain` computes a pointer into a composite object. This is common
when lowering structured memory references into SPIR-V pointer operations.

## Mental Model

The `spirv` dialect means:

```text
The compiler is now speaking in SPIR-V concepts: modules, storage classes,
capabilities, extensions, entry points, pointer types, and SPIR-V instructions.
```

A beginner can read `spirv` IR by asking:

- What target environment is this module meant for?
- What addressing and memory model does the module use?
- Which storage class does each pointer refer to?
- Are the integer, float, vector, image, matrix, and pointer types legal for the
  target?
- Which capabilities and extensions are implied by the operations?
- Which high-level dialects have already been lowered away?

## Gotchas

- `spirv` is target-facing. It is not a general-purpose high-level GPU dialect.
- Target environment matters. The same op may be legal under one version,
  capability, extension, or client API and illegal under another.
- Storage classes are part of pointer types. Ignoring them leads to incorrect
  memory lowering.
- SPIR-V entry points do not behave like ordinary MLIR functions with arbitrary
  arguments. ABI lowering often creates global interface variables.
- Some high-level MLIR types need emulation or rewriting because a target does
  not support narrow integers, unsupported float types, or certain vector forms.
- `convert-gpu-to-spirv` handles GPU device code, not host-side launch logic.
- Structured control flow has different constraints than `scf` or `cf`.
- Extensions such as GLSL, OpenCL, KHR, EXT, INTEL, ARM, and TOSA/Graph are not
  universally available.
- Layout decoration matters for Vulkan interfaces. A program can be structurally
  valid MLIR but still need layout decoration before it is acceptable SPIR-V for
  a client API.
- `convert-spirv-to-llvm` is useful for supported lowering paths, but SPIR-V is
  often intended to serialize to SPIR-V binary rather than pass through LLVM.

## Source Map

Primary source files in the LLVM tree:

- `mlir/include/mlir/Dialect/SPIRV/IR/SPIRVBase.td`
- `mlir/include/mlir/Dialect/SPIRV/IR/SPIRVOps.td`
- `mlir/include/mlir/Dialect/SPIRV/IR/SPIRVAttributes.td`
- `mlir/include/mlir/Dialect/SPIRV/IR/SPIRVTypes.h`
- `mlir/include/mlir/Dialect/SPIRV/Transforms/Passes.td`
- `mlir/lib/Dialect/SPIRV/IR/`
- `mlir/lib/Dialect/SPIRV/Transforms/`
- `mlir/lib/Dialect/SPIRV/Linking/`
- `mlir/lib/Conversion/*ToSPIRV/`
- `mlir/lib/Conversion/SPIRVToLLVM/`

Useful tests:

- `mlir/test/Dialect/SPIRV/IR/`
- `mlir/test/Dialect/SPIRV/Transforms/`
- `mlir/test/Conversion/*ToSPIRV/`
- `mlir/test/Conversion/SPIRVToLLVM/`
