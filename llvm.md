# `llvm` Dialect

## Beginner Summary

The `llvm` dialect is MLIR's representation of LLVM IR. It is the main target
dialect used when lowering MLIR programs to code that LLVM can optimize and
compile.

Most MLIR pipelines do not start in the `llvm` dialect. They start in
higher-level dialects such as `func`, `arith`, `memref`, `scf`, `cf`, `vector`,
`math`, `complex`, `gpu`, or domain-specific dialects. Those dialects are then
progressively converted into the `llvm` dialect and related LLVM-intrinsic
dialects such as `rocdl`, `nvvm`, `x86`, `arm_neon`, `arm_sve`, and `arm_sme`.

For a beginner, the `llvm` dialect means:

- You are close to the backend.
- Types and operations should now look like LLVM IR.
- `func.func` has usually become `llvm.func`.
- Structured control flow has usually become branches and blocks.
- `memref` values have usually become descriptors, pointers, and explicit
  loads/stores.
- Translation to real LLVM IR is now mostly mechanical.

## Why This Dialect Exists

MLIR and LLVM IR have different structure. MLIR uses dialects, regions, blocks,
block arguments, attributes, and operations. LLVM IR has functions, basic
blocks, instructions, metadata, global values, target triples, and data layouts.

The `llvm` dialect exists to bridge those worlds without immediately depending
on LLVM IR objects. It gives MLIR a thread-safe, structured way to represent
LLVM IR concepts before exporting them to an `llvm::Module`.

It also lets MLIR perform lowering in stages:

```text
high-level dialects
  -> mid-level MLIR dialects
  -> llvm dialect plus target-specific LLVM dialects
  -> LLVM IR
  -> object code
```

That staged design matters because many conversions are not one operation at a
time. For example, a `memref` type becomes a descriptor struct; a function with
multiple results may become an LLVM function returning a struct; structured
loops and conditionals must first become block-based control flow.

## When It Matters

The `llvm` dialect matters when:

- You are building a CPU lowering pipeline.
- You are lowering host-side GPU launch code.
- You are lowering `func`, `cf`, `arith`, `memref`, `vector`, `math`, `complex`,
  `index`, `ub`, OpenMP, or SPIR-V into LLVM-compatible IR.
- You need to reason about ABI details, pointer address spaces, function
  calling conventions, linkage, visibility, globals, or data layout.
- You need to inspect the last MLIR form before `mlir-translate --mlir-to-llvmir`.
- You are debugging failed LLVM export, usually from illegal types, unsupported
  operations, bad metadata, or a missing legalization pass.

It is less useful as a first IR for a frontend. If your source language has
loops, arrays, tensors, or structured control flow, you usually get better
optimization by starting higher up and lowering later.

## When To Use It

Use the `llvm` dialect when you need LLVM IR semantics in MLIR.

Good uses:

- Final CPU lowering.
- Host-side lowering for accelerators.
- Interfacing with external C or LLVM functions.
- Writing tests for conversion and export.
- Modeling LLVM-specific details: atomics, linkage, calling conventions,
  global variables, pointer arithmetic, exception handling, inline assembly,
  and metadata.
- Writing backend passes that need to run before LLVM IR export but after most
  MLIR abstraction has been removed.

Avoid using it too early when:

- You still need structured loop, tensor, vector, or memory-layout
  transformations.
- You want target-independent IR.
- You can express the program in `func`, `arith`, `memref`, `scf`, `cf`,
  `vector`, or `math`.
- You are tempted to manually build memref descriptors instead of using the
  memref-to-LLVM conversion.

## Core Concepts

### LLVM Dialect Mirrors LLVM IR

The dialect documentation states that LLVM dialect operation semantics should
match LLVM IR instruction semantics. If an LLVM dialect operation and its LLVM
IR counterpart disagree, that is treated as a bug.

Examples:

| LLVM dialect op | LLVM IR idea |
| --- | --- |
| `llvm.add` | Integer add |
| `llvm.fadd` | Floating-point add |
| `llvm.br` | Unconditional branch |
| `llvm.cond_br` | Conditional branch |
| `llvm.load` | Load from pointer |
| `llvm.store` | Store to pointer |
| `llvm.call` | Function call |
| `llvm.return` | Return instruction |
| `llvm.mlir.global` | Global variable |
| `llvm.func` | Function |

### MLIR Block Arguments Replace LLVM PHI Nodes

LLVM IR has `phi` instructions. MLIR uses block arguments instead. The `llvm`
dialect therefore has no `llvm.phi` operation.

```mlir
^then:
  llvm.br ^merge(%x : i32)

^else:
  llvm.br ^merge(%y : i32)

^merge(%v : i32):
  llvm.return %v : i32
```

The block argument `%v` plays the role that a PHI node would play in LLVM IR.

### `llvm.mlir.*` Operations Are MLIR Modeling Helpers

Some LLVM IR values are context-level things in LLVM IR, such as constants,
undef, poison, null/zero values, and global addresses. MLIR models them as
ordinary SSA-producing operations. These helper operations are prefixed with
`llvm.mlir.`:

```mlir
%c42 = llvm.mlir.constant(42 : i32) : i32
%p = llvm.mlir.poison : i64
%u = llvm.mlir.undef : !llvm.struct<(i32, f32)>
%z = llvm.mlir.zero : !llvm.ptr
%addr = llvm.mlir.addressof @global : !llvm.ptr
```

They are not normal LLVM IR instruction names; they are MLIR's way to model
LLVM IR values safely.

### Data Layout And Target Triple

LLVM modules can carry target information:

```mlir
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  // llvm.func, globals, metadata, and other operations
}
```

These attributes affect type conversion, especially `index` width, pointer
layout, memref descriptor layout, and target ABI assumptions.

### LLVM-Compatible Types

The LLVM dialect uses built-in MLIR types when they already match LLVM IR:

| Compatible built-in type | Example |
| --- | --- |
| Signless integers | `i1`, `i32`, `i64` |
| Floating-point types | `f16`, `bf16`, `f32`, `f64`, `f80`, `f128` |
| One-dimensional vectors | `vector<4xi32>`, `vector<[4]xf32>` |
| Tokens | MLIR token type |

Additional LLVM dialect types cover LLVM concepts that built-in MLIR types do
not cover:

| Type | Meaning |
| --- | --- |
| `!llvm.ptr` / `!llvm.ptr<addrspace>` | Opaque pointer, optionally with address space. |
| `!llvm.array<N x T>` | Fixed-size LLVM array. |
| `!llvm.func<result (args...)>` | LLVM function type. |
| `!llvm.struct<(...)>` | Literal struct type. |
| `!llvm.struct<"name", (...)>` | Identified struct type. |
| `!llvm.struct<"name", opaque>` | Opaque identified struct type. |
| `!llvm.target<...>` | LLVM target extension type. |
| `!llvm.x86_amx` | x86 AMX tile type. |
| `!llvm.ppc_fp128` | PowerPC 128-bit floating type. |
| `!llvm.byte` | LLVM byte type used by the dialect. |
| `!llvm.metadata` | Metadata value type for rare metadata-as-value cases. |
| `!llvm.void` | Void result type for LLVM function types. |

## Operations

The generated operation documentation in this checkout contains 83 base
`llvm` operations and 212 `llvm.intr.*` intrinsic operations.

### Essential Operation Groups

| Group | Operations |
| --- | --- |
| Integer arithmetic | `llvm.add`, `llvm.sub`, `llvm.mul`, `llvm.sdiv`, `llvm.udiv`, `llvm.srem`, `llvm.urem`, `llvm.and`, `llvm.or`, `llvm.xor`, `llvm.shl`, `llvm.ashr`, `llvm.lshr` |
| Floating arithmetic | `llvm.fadd`, `llvm.fsub`, `llvm.fmul`, `llvm.fdiv`, `llvm.frem`, `llvm.fneg` |
| Comparisons and select | `llvm.icmp`, `llvm.fcmp`, `llvm.select` |
| Casts | `llvm.trunc`, `llvm.zext`, `llvm.sext`, `llvm.fptrunc`, `llvm.fpext`, `llvm.fptosi`, `llvm.fptoui`, `llvm.sitofp`, `llvm.uitofp`, `llvm.ptrtoint`, `llvm.ptrtoaddr`, `llvm.inttoptr`, `llvm.bitcast`, `llvm.addrspacecast` |
| Memory | `llvm.alloca`, `llvm.load`, `llvm.store`, `llvm.getelementptr`, `llvm.fence`, `llvm.atomicrmw`, `llvm.cmpxchg` |
| Control flow | `llvm.br`, `llvm.cond_br`, `llvm.switch`, `llvm.indirectbr`, `llvm.return`, `llvm.unreachable` |
| Calls and exceptions | `llvm.call`, `llvm.invoke`, `llvm.resume`, `llvm.landingpad`, `llvm.call_intrinsic`, `llvm.inline_asm`, `llvm.va_arg` |
| Aggregate/vector | `llvm.extractvalue`, `llvm.insertvalue`, `llvm.extractelement`, `llvm.insertelement`, `llvm.shufflevector` |
| Functions/globals/symbols | `llvm.func`, `llvm.mlir.global`, `llvm.mlir.addressof`, `llvm.mlir.alias`, `llvm.mlir.ifunc`, `llvm.blockaddress`, `llvm.blocktag`, `llvm.dso_local_equivalent` |
| Metadata and module entities | `llvm.module_flags`, `llvm.named_metadata`, `llvm.linker_options`, `llvm.mlir.metadata_as_value`, `llvm.comdat`, `llvm.comdat_selector`, `llvm.mlir.global_ctors`, `llvm.mlir.global_dtors` |
| MLIR modeling helpers | `llvm.mlir.constant`, `llvm.mlir.undef`, `llvm.mlir.poison`, `llvm.mlir.zero`, `llvm.mlir.none` |

### Base Operation Inventory

```text
llvm.add
llvm.addrspacecast
llvm.alloca
llvm.and
llvm.ashr
llvm.atomicrmw
llvm.bitcast
llvm.blockaddress
llvm.blocktag
llvm.br
llvm.call
llvm.call_intrinsic
llvm.cmpxchg
llvm.comdat
llvm.comdat_selector
llvm.cond_br
llvm.dso_local_equivalent
llvm.extractelement
llvm.extractvalue
llvm.fadd
llvm.fcmp
llvm.fdiv
llvm.fence
llvm.fmul
llvm.fneg
llvm.fpext
llvm.fptosi
llvm.fptoui
llvm.fptrunc
llvm.freeze
llvm.frem
llvm.fsub
llvm.func
llvm.getelementptr
llvm.icmp
llvm.indirectbr
llvm.inline_asm
llvm.insertelement
llvm.insertvalue
llvm.inttoptr
llvm.invoke
llvm.landingpad
llvm.linker_options
llvm.load
llvm.lshr
llvm.mlir.addressof
llvm.mlir.alias
llvm.mlir.constant
llvm.mlir.global
llvm.mlir.global_ctors
llvm.mlir.global_dtors
llvm.mlir.ifunc
llvm.mlir.metadata_as_value
llvm.mlir.none
llvm.mlir.poison
llvm.mlir.undef
llvm.mlir.zero
llvm.module_flags
llvm.mul
llvm.named_metadata
llvm.or
llvm.ptrtoaddr
llvm.ptrtoint
llvm.resume
llvm.return
llvm.sdiv
llvm.select
llvm.sext
llvm.shl
llvm.shufflevector
llvm.sitofp
llvm.srem
llvm.store
llvm.sub
llvm.switch
llvm.trunc
llvm.udiv
llvm.uitofp
llvm.unreachable
llvm.urem
llvm.va_arg
llvm.xor
llvm.zext
```

### Intrinsic Operation Inventory

The dialect also defines wrappers for LLVM IR intrinsics. These are still in
the `llvm` dialect, unlike target-specific intrinsics that live in dialects
such as `rocdl`, `nvvm`, `x86`, or `arm_neon`.

```text
llvm.intr.abs
llvm.intr.acos
llvm.intr.annotation
llvm.intr.asin
llvm.intr.assume
llvm.intr.atan
llvm.intr.atan2
llvm.intr.bitreverse
llvm.intr.bswap
llvm.intr.ceil
llvm.intr.copysign
llvm.intr.coro.align
llvm.intr.coro.begin
llvm.intr.coro.end
llvm.intr.coro.free
llvm.intr.coro.id
llvm.intr.coro.promise
llvm.intr.coro.resume
llvm.intr.coro.save
llvm.intr.coro.size
llvm.intr.coro.suspend
llvm.intr.cos
llvm.intr.cosh
llvm.intr.ctlz
llvm.intr.ctpop
llvm.intr.cttz
llvm.intr.dbg.declare
llvm.intr.dbg.label
llvm.intr.dbg.value
llvm.intr.debugtrap
llvm.intr.eh.typeid.for
llvm.intr.exp
llvm.intr.exp10
llvm.intr.exp2
llvm.intr.expect
llvm.intr.expect.with.probability
llvm.intr.experimental.constrained.fadd
llvm.intr.experimental.constrained.fdiv
llvm.intr.experimental.constrained.fma
llvm.intr.experimental.constrained.fmul
llvm.intr.experimental.constrained.fmuladd
llvm.intr.experimental.constrained.fpext
llvm.intr.experimental.constrained.fptrunc
llvm.intr.experimental.constrained.frem
llvm.intr.experimental.constrained.fsub
llvm.intr.experimental.constrained.sitofp
llvm.intr.experimental.constrained.uitofp
llvm.intr.experimental.noalias.scope.decl
llvm.intr.experimental.vp.strided.load
llvm.intr.experimental.vp.strided.store
llvm.intr.fabs
llvm.intr.fake.use
llvm.intr.floor
llvm.intr.fma
llvm.intr.fmuladd
llvm.intr.frexp
llvm.intr.fshl
llvm.intr.fshr
llvm.intr.get.active.lane.mask
llvm.intr.invariant.end
llvm.intr.invariant.start
llvm.intr.is.constant
llvm.intr.is.fpclass
llvm.intr.launder.invariant.group
llvm.intr.ldexp
llvm.intr.lifetime.end
llvm.intr.lifetime.start
llvm.intr.llrint
llvm.intr.llround
llvm.intr.log
llvm.intr.log10
llvm.intr.log2
llvm.intr.lrint
llvm.intr.lround
llvm.intr.masked.compressstore
llvm.intr.masked.expandload
llvm.intr.masked.gather
llvm.intr.masked.load
llvm.intr.masked.scatter
llvm.intr.masked.store
llvm.intr.matrix.column.major.load
llvm.intr.matrix.column.major.store
llvm.intr.matrix.multiply
llvm.intr.matrix.transpose
llvm.intr.maximum
llvm.intr.maxnum
llvm.intr.memcpy
llvm.intr.memcpy.inline
llvm.intr.memmove
llvm.intr.memset
llvm.intr.memset.inline
llvm.intr.minimum
llvm.intr.minnum
llvm.intr.nearbyint
llvm.intr.pow
llvm.intr.powi
llvm.intr.prefetch
llvm.intr.ptr.annotation
llvm.intr.ptrmask
llvm.intr.rint
llvm.intr.round
llvm.intr.roundeven
llvm.intr.sadd.sat
llvm.intr.sadd.with.overflow
llvm.intr.scmp
llvm.intr.sin
llvm.intr.sincos
llvm.intr.sinh
llvm.intr.smax
llvm.intr.smin
llvm.intr.smul.with.overflow
llvm.intr.sqrt
llvm.intr.ssa.copy
llvm.intr.sshl.sat
llvm.intr.ssub.sat
llvm.intr.ssub.with.overflow
llvm.intr.stackrestore
llvm.intr.stacksave
llvm.intr.stepvector
llvm.intr.strip.invariant.group
llvm.intr.tan
llvm.intr.tanh
llvm.intr.threadlocal.address
llvm.intr.trap
llvm.intr.trunc
llvm.intr.uadd.sat
llvm.intr.uadd.with.overflow
llvm.intr.ubsantrap
llvm.intr.ucmp
llvm.intr.umax
llvm.intr.umin
llvm.intr.umul.with.overflow
llvm.intr.ushl.sat
llvm.intr.usub.sat
llvm.intr.usub.with.overflow
llvm.intr.vacopy
llvm.intr.vaend
llvm.intr.var.annotation
llvm.intr.vastart
llvm.intr.vector.deinterleave2
llvm.intr.vector.extract
llvm.intr.vector.insert
llvm.intr.vector.interleave2
llvm.intr.vector.reduce.add
llvm.intr.vector.reduce.and
llvm.intr.vector.reduce.fadd
llvm.intr.vector.reduce.fmax
llvm.intr.vector.reduce.fmaximum
llvm.intr.vector.reduce.fmin
llvm.intr.vector.reduce.fminimum
llvm.intr.vector.reduce.fmul
llvm.intr.vector.reduce.mul
llvm.intr.vector.reduce.or
llvm.intr.vector.reduce.smax
llvm.intr.vector.reduce.smin
llvm.intr.vector.reduce.umax
llvm.intr.vector.reduce.umin
llvm.intr.vector.reduce.xor
llvm.intr.vp.add
llvm.intr.vp.and
llvm.intr.vp.ashr
llvm.intr.vp.fadd
llvm.intr.vp.fdiv
llvm.intr.vp.fma
llvm.intr.vp.fmul
llvm.intr.vp.fmuladd
llvm.intr.vp.fneg
llvm.intr.vp.fpext
llvm.intr.vp.fptosi
llvm.intr.vp.fptoui
llvm.intr.vp.fptrunc
llvm.intr.vp.frem
llvm.intr.vp.fsub
llvm.intr.vp.inttoptr
llvm.intr.vp.load
llvm.intr.vp.lshr
llvm.intr.vp.merge
llvm.intr.vp.mul
llvm.intr.vp.or
llvm.intr.vp.ptrtoint
llvm.intr.vp.reduce.add
llvm.intr.vp.reduce.and
llvm.intr.vp.reduce.fadd
llvm.intr.vp.reduce.fmax
llvm.intr.vp.reduce.fmin
llvm.intr.vp.reduce.fmul
llvm.intr.vp.reduce.mul
llvm.intr.vp.reduce.or
llvm.intr.vp.reduce.smax
llvm.intr.vp.reduce.smin
llvm.intr.vp.reduce.umax
llvm.intr.vp.reduce.umin
llvm.intr.vp.reduce.xor
llvm.intr.vp.sdiv
llvm.intr.vp.select
llvm.intr.vp.sext
llvm.intr.vp.shl
llvm.intr.vp.sitofp
llvm.intr.vp.smax
llvm.intr.vp.smin
llvm.intr.vp.srem
llvm.intr.vp.store
llvm.intr.vp.sub
llvm.intr.vp.trunc
llvm.intr.vp.udiv
llvm.intr.vp.uitofp
llvm.intr.vp.umax
llvm.intr.vp.umin
llvm.intr.vp.urem
llvm.intr.vp.xor
llvm.intr.vp.zext
llvm.intr.vscale
```

## Attributes, Types, And Enums

The LLVM dialect has many attributes because LLVM IR uses metadata and flags
heavily. Beginners should first recognize these categories:

| Category | Examples |
| --- | --- |
| ABI and symbol behavior | `CConvAttr`, `LinkageAttr`, `Visibility`, `UnnamedAddr`, `TailCallKindAttr`, `UWTableKindAttr` |
| Atomics and memory | `AtomicOrdering`, `AtomicBinOp`, `MemoryEffectsAttr`, `DereferenceableAttr`, `AccessGroupAttr` |
| Arithmetic flags | `FastmathFlags`, `IntegerOverflowFlags`, `GEPNoWrapFlags` |
| Comparisons | `ICmpPredicate`, `FCmpPredicate` |
| Loop metadata | `LoopAnnotationAttr`, `LoopVectorizeAttr`, `LoopUnrollAttr`, `LoopInterleaveAttr`, `LoopPipelineAttr`, `LoopLICMAttr` |
| Debug information | `DICompileUnitAttr`, `DISubprogramAttr`, `DILocalVariableAttr`, `DIExpressionAttr`, `DIFileAttr`, and related `DI*` attributes |
| TBAA and aliasing | `TBAARootAttr`, `TBAATypeDescriptorAttr`, `TBAATagAttr`, `AliasScopeAttr`, `AliasScopeDomainAttr` |
| Module metadata | `ModuleFlagAttr`, `ModuleFlagCGProfileEntryAttr`, `ModuleFlagProfileSummaryAttr`, `DependentLibrariesAttr`, `TargetAttr` |
| MLIR helper values | `PoisonAttr`, `UndefAttr`, `ZeroAttr`, `MDNodeAttr`, `MDStringAttr`, `MDConstantAttr` |

Generated attribute definitions in this checkout:

```text
CConvAttr
ComdatAttr
FramePointerKindAttr
AccessGroupAttr
AddressSpaceAttr
AliasScopeAttr
AliasScopeDomainAttr
BlockAddressAttr
BlockTagAttr
ConstantRangeAttr
DIAnnotationAttr
DIBasicTypeAttr
DICommonBlockAttr
DICompileUnitAttr
DICompositeTypeAttr
DIDerivedTypeAttr
DIExpressionAttr
DIExpressionElemAttr
DIFileAttr
DIGenericSubrangeAttr
DIGlobalVariableAttr
DIGlobalVariableExpressionAttr
DIImportedEntityAttr
DILabelAttr
DILexicalBlockAttr
DILexicalBlockFileAttr
DILocalVariableAttr
DIModuleAttr
DINamespaceAttr
DINullTypeAttr
DIStringTypeAttr
DISubprogramAttr
DISubrangeAttr
DISubroutineTypeAttr
DSOLocalEquivalentAttr
DenormalFPEnvAttr
DependentLibrariesAttr
DereferenceableAttr
MDConstantAttr
MDFuncAttr
MDNodeAttr
MDStringAttr
MMRATagAttr
MemoryEffectsAttr
PoisonAttr
TBAAMemberAttr
TBAARootAttr
TBAATagAttr
TBAATypeDescriptorAttr
TargetAttr
TargetFeaturesAttr
UndefAttr
VScaleRangeAttr
VecTypeHintAttr
ZeroAttr
LinkageAttr
LoopAnnotationAttr
LoopDistributeAttr
LoopInterleaveAttr
LoopLICMAttr
LoopPeeledAttr
LoopPipelineAttr
LoopUnrollAndJamAttr
LoopUnrollAttr
LoopUnswitchAttr
LoopVectorizeAttr
ModuleFlagAttr
ModuleFlagCGProfileEntryAttr
ModuleFlagProfileSummaryAttr
ModuleFlagProfileSummaryDetailedAttr
TailCallKindAttr
UWTableKindAttr
WorkgroupAttributionAttr
```

Generated enum families include:

```text
AsmDialect
AtomicBinOp
AtomicOrdering
CConv
Comdat
DIFlags
DISubprogramFlags
DenormalModeKind
FCmpPredicate
FPExceptionBehavior
FastmathFlags
FramePointerKind
GEPNoWrapFlags
ICmpPredicate
IntegerOverflowFlags
DIEmissionKind
DINameTableKind
ProfileSummaryFormatKind
Linkage
ModFlagBehavior
ModRefInfo
RoundingMode
TailCallKind
UWTableKind
UnnamedAddr
Visibility
```

## Transformations

There are two kinds of transformations to know: transformations that produce
the LLVM dialect and transformations that clean up or prepare LLVM dialect IR
for export.

### Producing LLVM Dialect IR

| Pass | Purpose |
| --- | --- |
| `convert-to-llvm` | Generic conversion pass that asks dialects for their `ConvertToLLVMPatternInterface` patterns. |
| `convert-func-to-llvm` | Converts `func.func`, calls, and returns to `llvm.func`, `llvm.call`, and `llvm.return`. |
| `convert-cf-to-llvm` | Converts ControlFlow dialect branch operations to LLVM branch operations. |
| `convert-arith-to-llvm` | Converts supported `arith` operations to LLVM dialect operations. |
| `convert-index-to-llvm` | Lowers `index` dialect operations to LLVM operations. |
| `finalize-memref-to-llvm` | Converts MemRef operations and descriptors to LLVM dialect operations. |
| `convert-vector-to-llvm` | Lowers vector operations to LLVM dialect operations or LLVM intrinsics. |
| `convert-math-to-llvm` | Converts supported `math` operations to LLVM dialect operations or intrinsics. |
| `convert-complex-to-llvm` | Converts complex-number operations to LLVM-compatible forms. |
| `convert-ub-to-llvm` | Converts `ub.poison` and `ub.unreachable` to LLVM dialect operations. |
| `gpu-to-llvm` and `lower-host-to-llvm` | Lower host-side GPU runtime calls and launch code. |
| `convert-spirv-to-llvm` | Converts SPIR-V dialect to LLVM dialect. |
| `convert-openmp-to-llvm` | Converts OpenMP operations to OpenMP operations using LLVM dialect values. |

### Preparing LLVM Dialect IR

| Pass | Purpose |
| --- | --- |
| `set-llvm-module-datalayout` | Attaches an LLVM data layout string to a module. |
| `reconcile-unrealized-casts` | Removes no-op `unrealized_conversion_cast` operations left by staged conversions. |
| `llvm-legalize-for-export` | Rewrites LLVM dialect IR into a form that can be translated to LLVM IR. |
| `llvm-add-comdats` | Adds COMDATs to linkonce/linkonce_odr functions. |
| `llvm-request-c-wrappers` | Marks builtin functions so conversion emits C wrappers. |
| `llvm-use-default-visibility` | Updates default symbol visibility to hidden or protected. |
| `ensure-debug-info-scope-on-llvm-func` | Adds debug-info subprogram attributes for line-table emission. |
| `mem2reg` | Promotes eligible LLVM `alloca` memory slots to SSA values through memory slot interfaces. |
| `sroa` | Breaks aggregate LLVM memory slots into smaller slots when safe. |

## Conversions And Lowering Paths

A common CPU lowering path looks like this:

```text
func + arith + memref + scf + cf + vector + math
  -> lower-affine / convert-scf-to-cf / expand-strided-metadata as needed
  -> convert-arith-to-llvm
  -> convert-index-to-llvm
  -> convert-cf-to-llvm
  -> convert-func-to-llvm
  -> finalize-memref-to-llvm
  -> convert-vector-to-llvm
  -> convert-math-to-llvm
  -> reconcile-unrealized-casts
  -> llvm-legalize-for-export
  -> mlir-translate --mlir-to-llvmir
  -> LLVM optimization and code generation
```

The exact order depends on the source dialects and target. The key idea is that
conversion is progressive: one pass can convert some operations while leaving
others in place, using temporary `unrealized_conversion_cast` operations between
old and new type systems.

Important type conversions:

| Source type | LLVM dialect form |
| --- | --- |
| `index` | Integer whose width comes from data layout or pass option. |
| `complex<T>` | `!llvm.struct<(T, T)>` after element conversion. |
| `memref<...>` | Descriptor struct containing allocated pointer, aligned pointer, offset, sizes, and strides. |
| `memref<*xT>` | Unranked descriptor with rank and pointer to ranked descriptor. |
| Multiple function results | Packed into an LLVM struct result. |
| No function result | `!llvm.void` in the LLVM function type. |

Important outgoing conversion:

```shell
mlir-translate --mlir-to-llvmir input.mlir
```

The translation registers the LLVM dialect to LLVM IR translation interface,
then builds an LLVM `Module`.

## Example IR

This is a small LLVM dialect function:

```mlir
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  llvm.func @add_one(%arg0: i32) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %sum = llvm.add %arg0, %c1 : i32
    llvm.return %sum : i32
  }
}
```

This example shows a global and an address-of helper:

```mlir
llvm.mlir.global internal @counter(0 : i32) : i32

llvm.func @load_counter() -> i32 {
  %addr = llvm.mlir.addressof @counter : !llvm.ptr
  %value = llvm.load %addr : !llvm.ptr -> i32
  llvm.return %value : i32
}
```

This example shows block arguments acting like PHI nodes:

```mlir
llvm.func @select_branch(%cond: i1, %a: i32, %b: i32) -> i32 {
  llvm.cond_br %cond, ^then(%a : i32), ^else(%b : i32)

^then(%x: i32):
  llvm.br ^merge(%x : i32)

^else(%y: i32):
  llvm.br ^merge(%y : i32)

^merge(%v: i32):
  llvm.return %v : i32
}
```

## Mental Model

Think of the `llvm` dialect as LLVM IR in MLIR clothing.

It still uses MLIR syntax, MLIR attributes, MLIR regions, and MLIR block
arguments, but its concepts are LLVM concepts. This dialect is where the
compiler stops carrying high-level source structure and starts carrying ABI,
pointer, memory, calling convention, control-flow, and backend semantics.

The most important beginner rule is: lower to `llvm` late enough that you have
already used the useful MLIR optimizations, but early enough that final backend
translation has the exact LLVM semantics it needs.

## Gotchas

- There is no `llvm.phi`. Use block arguments on destination blocks.
- `llvm.mlir.constant`, `llvm.mlir.undef`, `llvm.mlir.poison`, and
  `llvm.mlir.zero` are MLIR modeling operations, not ordinary LLVM IR
  instructions.
- `!llvm.ptr` is opaque; pointee type information belongs on operations such as
  `llvm.load`, `llvm.store`, and `llvm.getelementptr`.
- Type conversion must be consistent. Mixing different index bitwidth choices
  in one lowering pipeline is a common source of broken IR.
- `memref` lowering creates descriptors. Do not hand-write descriptors unless
  you are deliberately working at ABI level.
- Staged conversions may leave `unrealized_conversion_cast` operations. Run
  `reconcile-unrealized-casts` once enough of the IR has been converted.
- `llvm-legalize-for-export` is often needed before translating to LLVM IR.
- LLVM dialect supports metadata as structured MLIR attributes. Incorrect debug
  info, TBAA, alias scopes, or module flags can make export fail.
- Target-specific intrinsics usually do not belong in the base `llvm` dialect.
  Look for `rocdl`, `nvvm`, `x86`, `arm_neon`, `arm_sve`, or `arm_sme`.

## Source Map

Use these files in the LLVM repo when you need exact behavior:

| Topic | Files |
| --- | --- |
| Dialect definition | `mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.td` |
| Base operations | `mlir/include/mlir/Dialect/LLVMIR/LLVMOps.td` |
| Intrinsic operations | `mlir/include/mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td` |
| Operation base classes | `mlir/include/mlir/Dialect/LLVMIR/LLVMOpBase.td` |
| Types | `mlir/include/mlir/Dialect/LLVMIR/LLVMTypes.td`, `mlir/lib/Dialect/LLVMIR/IR/LLVMTypes.cpp`, `mlir/lib/Dialect/LLVMIR/IR/LLVMTypeSyntax.cpp` |
| Attributes and enums | `mlir/include/mlir/Dialect/LLVMIR/LLVMAttrDefs.td`, `mlir/include/mlir/Dialect/LLVMIR/LLVMEnums.td`, `mlir/lib/Dialect/LLVMIR/IR/LLVMAttrs.cpp` |
| Dialect implementation | `mlir/lib/Dialect/LLVMIR/IR/LLVMDialect.cpp` |
| Memory slot support | `mlir/lib/Dialect/LLVMIR/IR/LLVMMemorySlot.cpp` |
| LLVM dialect transforms | `mlir/include/mlir/Dialect/LLVMIR/Transforms/Passes.td`, `mlir/lib/Dialect/LLVMIR/Transforms` |
| Conversion pass declarations | `mlir/include/mlir/Conversion/Passes.td` |
| LLVM IR export | `mlir/include/mlir/Target/LLVMIR/Export.h`, `mlir/lib/Target/LLVMIR/ModuleTranslation.cpp` |
| LLVM dialect translation | `mlir/include/mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h`, `mlir/lib/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.cpp` |
| LLVM IR import | `mlir/include/mlir/Target/LLVMIR/Import.h`, `mlir/lib/Target/LLVMIR/ModuleImport.cpp` |
| Main docs | `mlir/docs/Dialects/LLVM.md`, `mlir/docs/TargetLLVMIR.md` |
| Dialect tests | `mlir/test/Dialect/LLVMIR` |
| Conversion tests | `mlir/test/Conversion/*ToLLVM`, `mlir/test/Conversion/LLVMCommon` |
