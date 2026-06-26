# XeVM Dialect

## Beginner Summary

The `xevm` dialect models low-level Intel GPU hardware features as an
extension of the LLVM dialect.

It is used for operations that ordinary LLVM dialect does not describe well by
itself:

- Intel GPU subgroup block loads and stores.
- 2D block memory operations for matrix tiles.
- Cache-control decorations for loads, stores, and prefetches.
- Subgroup matrix multiply-add operations.
- MX-style scaled matrix multiply-add.
- Low-precision float packing and unpacking.
- Intel GPU execution IDs such as local ID, group ID, lane ID, and subgroup ID.
- GPU target attributes for Intel Xe compilation.

Think of `xevm` as the Intel Xe GPU "near LLVM" dialect. It sits below
higher-level GPU dialects such as `gpu` and `xegpu`, and above final LLVM IR or
target-specific builtin calls.

## Why This Dialect Exists

Intel GPU code generation needs to represent operations that are not generic
LLVM operations.

For example, an Intel GPU has hardware-oriented operations such as:

- Subgroup block reads and writes.
- 2D matrix tile loads and stores.
- Cache policy controls for different cache levels.
- Subgroup matrix multiply-add instructions.
- Special registers for lane and workgroup information.
- Packed low-precision matrix formats such as f8, bf8, and e2m1.

The generic LLVM dialect can model calls and low-level scalar/vector
operations, but it does not by itself carry the higher-level intent of an Intel
GPU block load or matrix instruction. The `xevm` dialect keeps that intent
visible long enough for MLIR conversion and LLVM IR translation to choose the
right target builtin, calling convention, metadata, or intrinsic.

The dialect is deliberately target-specific. It is not a portable GPU dialect.
It exists so lowerings that already know they are targeting Intel Xe hardware
can preserve the hardware operation accurately.

## When It Matters

The `xevm` dialect matters late in Intel GPU lowering pipelines.

It commonly appears in flows like:

```text
gpu / xegpu / math / vector / memref
  -> convert-xegpu-to-xevm and convert-math-to-xevm
  -> xevm.blockload2d, xevm.mma, xevm.extf, xevm.local_id.x, ...
  -> convert-xevm-to-llvm
  -> LLVM dialect calls, intrinsics, metadata, and target-specific builtins
  -> LLVM IR translation for XeVM
```

It is especially important for:

- Intel GPU matrix kernels.
- Subgroup cooperative memory operations.
- Tiled matrix loads and stores.
- Cache-aware memory access.
- Low-precision GPU matrix formats.
- Device code that needs Intel GPU special registers.
- Attaching Intel Xe target information to GPU modules.

## When To Use It

Use the `xevm` dialect when a pipeline is already targeting Intel Xe GPUs and
needs explicit hardware-level operations.

Use it for:

- Lowering `xegpu` tile operations to Intel GPU block memory operations.
- Expressing 1D and 2D subgroup block loads, stores, and prefetches.
- Carrying cache-control decisions into lowering.
- Representing subgroup matrix multiply-add.
- Representing MX scaled matrix multiply-add.
- Converting between f16/bf16 and narrow packed formats such as f8, bf8, and
  e2m1.
- Reading Intel GPU execution IDs.
- Lowering math operations to native XeVM or OpenCL-style calls.

Do not use `xevm` as a beginner-level GPU programming model. If your IR still
needs portability, use `gpu`, `scf`, `vector`, `linalg`, or `xegpu` first.
Move to `xevm` when the pipeline has committed to Intel GPU target behavior.

## Core Concepts

### XeVM Extends LLVM

In this checkout, XeVM lives under the LLVMIR dialect area:

```text
mlir/include/mlir/Dialect/LLVMIR/XeVMOps.td
```

The dialect depends on `LLVM::LLVMDialect`, and its operations use LLVM-like
types such as `!llvm.ptr<1>`.

That placement matters. XeVM is not a high-level tensor dialect. It is a
low-level target dialect used close to LLVM lowering.

### XeGPU Versus XeVM

`xegpu` is a higher-level Intel GPU dialect that describes tiled GPU
operations in a form that is still easier for MLIR to transform.

`xevm` is lower-level. It exposes operations that are much closer to target
builtins and LLVM translation.

A common path is:

```text
xegpu.create_nd_tdesc / xegpu.load_nd / xegpu.dpas
  -> convert-xegpu-to-xevm
  -> xevm.blockload2d / xevm.mma / xevm.blockstore2d
  -> convert-xevm-to-llvm
```

### Subgroups And Lanes

Many XeVM operations are subgroup operations. A subgroup is a set of lanes that
cooperate on memory or matrix operations.

`xevm.blockload`, `xevm.blockstore`, `xevm.blockload2d`, `xevm.blockstore2d`,
`xevm.mma`, and `xevm.mma_mx` describe work performed cooperatively by the
subgroup, not independent scalar work from one ordinary CPU thread.

That is why the dialect has special register operations such as `xevm.lane_id`,
`xevm.subgroup_id`, and `xevm.subgroup_size`.

### Cache Control

XeVM memory operations can carry cache-control attributes.

Load cache controls include combinations for L1, L2, and L3 behavior, plus an
invalidate-read option. Store cache controls include uncached, write-back,
write-through, and streaming-style choices across cache levels.

These attributes are not general memory effects. They are target-specific hints
or controls that lower to metadata, builtin choices, or target-specific call
forms.

### Memory Scope And Address Space

`xevm.memfence` uses memory scopes such as:

- `lane`
- `subgroup`
- `workgroup`
- `cluster`
- `device`
- `system`

It also uses address spaces such as:

- `private`
- `global`
- `constant`
- `shared`
- `generic`

The operation says which other work-items must observe prior memory operations
and which address space is affected.

### Target Attribute

XeVM defines the `#xevm.target` attribute for GPU modules.

The attribute carries target compilation information such as optimization
level, target triple, chip, target-specific flags, and link files. The
`xevm-attach-target` pass can attach this attribute to matching `gpu.module`
operations.

Example shape:

```mlir
gpu.module @kernel_module [#xevm.target<chip = "pvc">] {
}
```

## Operations

### 1D Block Memory Operations

`xevm.blockload` performs a subgroup block load from a pointer.

The loaded values are strided by subgroup local invocation ID. Conceptually,
each lane participates in reading a block from memory, and a result may be a
scalar or a one-dimensional vector.

`xevm.blockstore` performs the matching subgroup block store.

These operations accept optional cache-control attributes.

### 2D Block Memory Operations

`xevm.blockload2d` loads a two-dimensional matrix tile from a larger matrix in
global memory.

Important operands and properties include:

- Base pointer.
- Base width, height, and pitch.
- Tile starting coordinates `x` and `y`.
- Element size in bits.
- Tile width and height.
- Number of vertical blocks.
- Optional transpose.
- Optional packed-register layout.
- Optional load cache control.

If the tile contains out-of-bounds elements, the operation fills those elements
with zero.

`xevm.blockstore2d` stores a two-dimensional matrix tile back to memory.

`xevm.blockprefetch2d` prefetches a two-dimensional tile into the cache
subsystem.

These operations are central to Intel GPU tiled matrix kernels.

### Prefetch And Fence

`xevm.prefetch` prefetches data from global or generic memory into a cache
subsystem. It accepts load cache-control attributes.

`xevm.memfence` enforces memory visibility for prior memory accesses from the
current work-item. It is parameterized by memory scope and address space.

### Matrix Multiply-Add

`xevm.mma` is the subgroup matrix multiply-add operation.

It represents:

```text
D = C + A x B
```

The operation carries:

- Vector fragments for A and B.
- An optional vector fragment for C.
- An MMA shape attribute with `m`, `n`, and `k`.
- An MMA type attribute with D, A, B, and optionally C element types.

`xevm.mma_mx` is similar, but adds scale operands for A and B. It is used for
MX-style scaled matrix multiplication over narrow formats.

### Low-Precision Conversion

`xevm.truncf` truncates floating-point values from `f16` or `bf16` to narrow
packed formats:

- `f8`
- `bf8`
- `e2m1`

`xevm.extf` extends narrow packed formats back to `f16` or `bf16`.

These operations are important for matrix kernels that use compact storage or
MX-scaled formats but accumulate in wider types.

### Execution ID Operations

XeVM provides operations for reading Intel GPU execution IDs.

Local work-item ID:

- `xevm.local_id.x`
- `xevm.local_id.y`
- `xevm.local_id.z`

Local workgroup size:

- `xevm.local_size.x`
- `xevm.local_size.y`
- `xevm.local_size.z`

Workgroup ID:

- `xevm.group_id.x`
- `xevm.group_id.y`
- `xevm.group_id.z`

Grid group count:

- `xevm.group_count.x`
- `xevm.group_count.y`
- `xevm.group_count.z`

Subgroup and lane values:

- `xevm.lane_id`
- `xevm.subgroup_id`
- `xevm.subgroup_size`

These operations return `i32` or `i64` and can carry an optional LLVM constant
range attribute.

## Transformations

### Canonicalization And Verification

XeVM is a low-level dialect, so its most important local behavior is
verification.

For example, the verifier checks restrictions on 2D block load/store shapes,
element sizes, tile widths, tile heights, vertical block counts, transpose, and
packed-register combinations. `xevm.mma` and `xevm.mma_mx` also verify that
fragment vector types, shape attributes, and element type attributes are
consistent.

These checks are important because XeVM operations correspond to constrained
hardware builtins. Invalid combinations should be rejected before final target
lowering.

### `xevm-attach-target`

`xevm-attach-target` attaches a `#xevm.target` attribute to matching
`gpu.module` operations.

The pass has options for:

- Matching module names with a regex.
- Target triple.
- Target chip.
- Optimization level.
- Link libraries.
- Downstream command options.

Use this when a module must be marked for Intel XeVM target compilation.

### `convert-xegpu-to-xevm`

`convert-xegpu-to-xevm` lowers higher-level `xegpu` operations into XeVM.

Examples include:

- `xegpu.dpas` to `xevm.mma`.
- `xegpu` 2D tile loads/stores to `xevm.blockload2d` and
  `xevm.blockstore2d`.
- `xegpu` prefetch to `xevm.blockprefetch2d`.
- Micro-scaling float conversions to `xevm.truncf` and `xevm.extf`.
- XeGPU cache hints to XeVM cache-control attributes.

The pass has a `use-64bit-index` option that controls how index values are
converted.

### `convert-math-to-xevm`

`convert-math-to-xevm` lowers supported `math` operations to native XeVM or
SPIR-V/OpenCL-style equivalents.

The pass is precision-aware. By default, it lowers supported math operations to
native OpenCL `native_` intrinsics only when the operation carries an `afn`
fast-math flag. With the `convert-to-ocl` option, it can lower supported Math
ops to OpenCL math intrinsics more broadly.

This pass can also convert selected `arith` operations, such as `arith.divf`,
unless `convert-arith=false` is specified.

## Conversions And Lowering Paths

### `convert-xevm-to-llvm`

`convert-xevm-to-llvm` lowers XeVM operations to LLVM dialect.

This conversion commonly produces:

- LLVM dialect calls to Intel GPU builtins.
- SPIR calling convention on generated functions or calls.
- LLVM metadata and annotations for cache controls.
- LLVM intrinsics for special register queries.
- Helper calls for low-precision conversion.
- Ordinary LLVM vector, pointer, load, store, alloca, and bitcast operations
  around target-specific calls.

For example, a 2D block load lowers to a call to a mangled Intel subgroup block
read builtin, with temporary storage used to receive the tile result.

### LLVM IR Translation

The XeVM dialect also has LLVM IR translation support under:

```text
mlir/lib/Target/LLVMIR/Dialect/XeVM/
```

That translation is the bridge from MLIR's LLVM dialect plus XeVM target
metadata into final LLVM IR suitable for the Intel GPU path.

### Typical Lowering Strategy

A practical Intel GPU lowering sequence looks like:

```text
gpu.module + xegpu/math/vector/memref IR
  -> xevm-attach-target
  -> convert-xegpu-to-xevm
  -> convert-math-to-xevm
  -> convert-xevm-to-llvm
  -> LLVM dialect and XeVM LLVM IR translation
```

The exact order depends on the surrounding pipeline, especially when math,
vector, memref, and GPU dialect conversions are scheduled.

## Example IR

### Reading Execution IDs

```mlir
llvm.func @ids() -> i32 {
  %lid = xevm.local_id.x : i32
  %gid = xevm.group_id.x : i32
  %lane = xevm.lane_id : i32
  llvm.return %lane : i32
}
```

This reads a work-item local ID, workgroup ID, and subgroup lane ID. The example
returns only one value because `llvm.func` supports zero or one result.

### Loading A 2D Tile

```mlir
llvm.func @tile_load(%ptr: !llvm.ptr<1>, %width: i32, %height: i32,
                     %pitch: i32, %x: i32, %y: i32) -> vector<8xi16> {
  %tile = xevm.blockload2d %ptr, %width, %height, %pitch, %x, %y
    <{elem_size_in_bits = 16 : i32, tile_width = 16 : i32,
      tile_height = 8 : i32, v_blocks = 1 : i32,
      transpose = false, pack_register = false,
      cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>}>
    : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return %tile : vector<8xi16>
}
```

This loads an 8-row by 16-column tile of 16-bit elements. The result type is the
per-subgroup vector fragment produced by the hardware operation.

### Subgroup Matrix Multiply-Add

```mlir
llvm.func @mma(%a: vector<8xf16>, %b: vector<16xf16>,
               %c: vector<8xf32>) -> vector<8xf32> {
  %d = xevm.mma %a, %b, %c
      {shape = <m = 8, n = 16, k = 16>,
       types = <d = f32, a = f16, b = f16, c = f32>}
      : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
  llvm.return %d : vector<8xf32>
}
```

The fragments are not full matrices. They are the per-subgroup vector pieces
that the Intel GPU matrix operation consumes and produces.

### Narrow Float Conversion Round Trip

```mlir
llvm.func @narrow(%src: vector<16xf16>) -> vector<16xf16> {
  %packed = xevm.truncf %src {src_etype = f16, dst_etype = f8}
      : (vector<16xf16>) -> vector<16xi8>
  %wide = xevm.extf %packed {src_etype = f8, dst_etype = f16}
      : (vector<16xi8>) -> vector<16xf16>
  llvm.return %wide : vector<16xf16>
}
```

This shows how XeVM represents conversion between ordinary half precision and
packed f8 storage.

### Fence And Prefetch

```mlir
llvm.func @fence_and_prefetch(%ptr: !llvm.ptr<1>) {
  xevm.memfence <{addrspace = #xevm.addr_space<global>,
                  scope = #xevm.mem_scope<workgroup>}>
  xevm.prefetch %ptr
      <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>}>
      : (!llvm.ptr<1>)
  llvm.return
}
```

The fence controls visibility of prior global memory accesses at workgroup
scope. The prefetch requests a target-specific cache behavior.

## Mental Model

The XeVM dialect says:

```text
This is no longer portable GPU IR.
This is Intel Xe GPU IR close to LLVM lowering.
Preserve the hardware operation, cache choice, tile shape, or special register
until it can become the right LLVM call, intrinsic, or target annotation.
```

For beginners, the useful split is:

- `xevm.blockload`, `xevm.blockstore`, `xevm.blockload2d`,
  `xevm.blockstore2d`, `xevm.prefetch`, and `xevm.blockprefetch2d` are memory
  operations.
- `xevm.mma` and `xevm.mma_mx` are subgroup matrix operations.
- `xevm.truncf` and `xevm.extf` handle narrow packed float formats.
- `xevm.memfence` handles memory visibility.
- `xevm.local_id.*`, `xevm.group_id.*`, `xevm.group_count.*`,
  `xevm.local_size.*`, `xevm.lane_id`, `xevm.subgroup_id`, and
  `xevm.subgroup_size` expose execution IDs.
- `#xevm.target` tells the GPU module what Intel target it is compiled for.

## Gotchas

XeVM is target-specific. Do not use it if the IR is still meant to be portable
across GPU vendors.

XeVM is low-level. It uses LLVM-style pointer types and often appears inside
`llvm.func`, not only inside high-level `func.func` or tensor code.

The vector types on `xevm.mma` and block operations are hardware fragments, not
source-language vectors in the usual beginner sense.

Cache controls are target controls, not generic memory semantics. They matter
for codegen, but they do not replace memory ordering.

`xevm.memfence` is scoped. The memory scope and address space determine what
kind of visibility is requested.

The verifier is strict because the hardware operation has strict shape and type
constraints. Tile sizes, element sizes, transpose, and packed-register options
must be compatible.

`convert-math-to-xevm` is controlled by fast-math flags and pass options. A
math op without `afn` may intentionally remain unchanged.

## Source Map

Important local source files:

- `mlir/include/mlir/Dialect/LLVMIR/XeVMOps.td` defines the XeVM dialect,
  attributes, operations, and `#xevm.target`.
- `mlir/include/mlir/Dialect/LLVMIR/XeVMDialect.h` declares the dialect C++
  interface.
- `mlir/lib/Dialect/LLVMIR/IR/XeVMDialect.cpp` implements dialect
  registration, verification, and target attribute verification.
- `mlir/include/mlir/Conversion/XeGPUToXeVM/XeGPUToXeVM.h` and
  `mlir/lib/Conversion/XeGPUToXeVM/XeGPUToXeVM.cpp` implement XeGPU-to-XeVM
  conversion.
- `mlir/include/mlir/Conversion/XeVMToLLVM/XeVMToLLVM.h` and
  `mlir/lib/Conversion/XeVMToLLVM/XeVMToLLVM.cpp` implement XeVM-to-LLVM
  conversion.
- `mlir/include/mlir/Conversion/MathToXeVM/MathToXeVM.h` and
  `mlir/lib/Conversion/MathToXeVM/MathToXeVM.cpp` implement Math-to-XeVM
  lowering.
- `mlir/lib/Dialect/GPU/Transforms/XeVMAttachTarget.cpp` implements
  `xevm-attach-target`.
- `mlir/lib/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.cpp` implements
  LLVM IR translation support.
- `mlir/include/mlir/Target/LLVM/XeVM/Target.h` and
  `mlir/lib/Target/LLVM/XeVM/Target.cpp` implement XeVM target support.
- `mlir/test/Conversion/XeGPUToXeVM/` contains conversion tests from higher
  Intel GPU IR to XeVM.
- `mlir/test/Conversion/XeVMToLLVM/` contains XeVM-to-LLVM conversion tests.
- `mlir/test/Conversion/MathToXeVM/` contains Math-to-XeVM conversion tests.
- `mlir/test/Integration/Dialect/XeVM/GPU/` contains GPU integration tests for
  XeVM operations.

Generated operation documentation for this checkout lists these 26 operations:

```text
xevm.blockload
xevm.blockload2d
xevm.blockprefetch2d
xevm.blockstore
xevm.blockstore2d
xevm.extf
xevm.group_count.x
xevm.group_count.y
xevm.group_count.z
xevm.group_id.x
xevm.group_id.y
xevm.group_id.z
xevm.lane_id
xevm.local_id.x
xevm.local_id.y
xevm.local_id.z
xevm.local_size.x
xevm.local_size.y
xevm.local_size.z
xevm.memfence
xevm.mma
xevm.mma_mx
xevm.prefetch
xevm.subgroup_id
xevm.subgroup_size
xevm.truncf
```
