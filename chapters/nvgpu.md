# NVGPU Dialect

## Beginner Summary

The `nvgpu` dialect is MLIR's structured NVIDIA GPU dialect.

It sits between portable GPU/vector IR and the lower-level `nvvm` dialect. It
represents NVIDIA-specific GPU concepts such as asynchronous global-to-shared
copies, Tensor Core MMA instructions, Tensor Memory Accelerator operations,
memory barriers, and warpgroup MMA while still using MLIR types such as
`memref` and `vector`.

Think of `nvgpu` as the stage where a compiler is still using understandable
MLIR data structures, but it has already decided that the target is NVIDIA
hardware.

Most users should not start by writing `nvgpu` directly. A typical pipeline
starts with higher-level dialects such as `linalg`, `tensor`, `scf`, `gpu`, and
`vector`, then lowers selected parts into `nvgpu` when NVIDIA-specific hardware
features are needed.

## Why This Dialect Exists

Portable GPU IR needs to describe ideas such as threads, blocks, memory
transfers, vector operations, and matrix multiplication without committing to
one vendor.

NVIDIA GPUs expose features that are more specific than that:

- `ldmatrix` loads for matrix fragments in shared memory.
- Warp-level `mma.sync` and sparse `mma.sp.sync` Tensor Core operations.
- Device-side asynchronous copies from global memory to shared memory.
- Shared-memory memory barriers, or mbarriers, used by newer async operations.
- Tensor Memory Accelerator, or TMA, descriptors, prefetches, loads, and stores.
- Warpgroup MMA, or WGMMA, where multiple warps cooperate on matrix multiply.
- NVIDIA-specific reciprocal lowering details.

The `nvgpu` dialect exists so MLIR can represent those choices before dropping
all the way to `nvvm` intrinsics and PTX-like operations. This middle level is
useful because it keeps the IR structured enough for compiler transformations
while exposing enough target information to select real NVIDIA instructions.

## When It Matters

The `nvgpu` dialect matters in NVIDIA GPU code generation after the compiler has
already selected tiling, memory placement, vector shapes, and GPU mapping.

A common pipeline shape is:

```text
linalg / tensor / scf / vector
  -> tiling, fusion, vectorization, and bufferization
  -> gpu modules, kernels, blocks, and threads
  -> nvgpu for NVIDIA-specific async copy, MMA, TMA, and WGMMA operations
  -> convert-nvgpu-to-nvvm
  -> convert-gpu-to-nvvm and convert-math-to-nvvm as needed
  -> convert-nvvm-to-llvm
  -> LLVM/NVPTX/PTX lowering
```

It is especially important when optimizing matrix-heavy kernels, shared-memory
data movement, or kernels that use Hopper-era features such as TMA and WGMMA.

## When To Use It

Use `nvgpu` when the IR is already targeting NVIDIA GPUs and the compiler needs
to express NVIDIA-specific hardware operations without immediately losing MLIR's
structured `memref` and `vector` information.

Use it for:

- Turning global-to-shared vector transfers into asynchronous device copies.
- Representing `ldmatrix` loads from shared memory into register fragments.
- Representing warp-level dense and sparse Tensor Core MMA.
- Modeling mbarrier allocation, initialization, arrival, and waiting.
- Building TMA descriptors and issuing TMA loads, stores, fences, and prefetches.
- Representing warpgroup descriptors, accumulators, WGMMA execution, and stores.
- Preparing NVIDIA-specific operations for lowering into `nvvm`.

Do not use it as a portable GPU abstraction. Code that must target NVIDIA,
AMD, Intel, CPU vector units, and SPIR-V backends should stay in higher-level
dialects until the target-specific split is intentional.

## Core Concepts

### NVGPU Is Not NVVM

`nvgpu` is NVIDIA-specific, but it is still higher-level than `nvvm`.

The `nvvm` dialect is closer to NVVM IR, LLVM intrinsics, and PTX instructions.
The `nvgpu` dialect keeps operations closer to MLIR's structured world. For
example, `nvgpu.ldmatrix` still reads from a `memref` and returns a `vector`,
while the corresponding lower-level operation is tied more directly to NVVM/PTX
instruction forms.

This makes `nvgpu` a useful compiler staging point.

### Memory Spaces Matter

Many `nvgpu` operations care deeply about memory space.

Shared memory is represented either by the numeric NVVM shared memory space
`3` or by `#gpu.address_space<workgroup>`. Global memory commonly uses numeric
space `1`. Async copies usually move from global memory to shared memory. TMA
loads also move data into shared memory, and mbarriers live in shared memory.

If a memref is in the wrong memory space, the IR may parse but fail verification
or fail to lower into the intended NVIDIA instruction.

### Async Copy Tokens And Groups

`nvgpu.device_async_copy` starts a device-side asynchronous copy and returns a
`!nvgpu.device.async.token`.

`nvgpu.device_async_create_group` groups one or more copy tokens. The group
token is then passed to `nvgpu.device_async_wait`. This token chain makes the
ordering explicit in SSA form.

The important beginner idea is that the copy does not block immediately. You
must group it and wait at a point where the copied data is required.

### Matrix Fragments

`nvgpu.ldmatrix`, `nvgpu.mma.sync`, and `nvgpu.mma.sp.sync` represent
warp-level matrix operations. Their operands are not whole logical matrices in
the high-level sense. They are per-thread fragments that collectively represent
a warp-level operation.

This is why the vector shapes can look small. A type such as `vector<4x2xf16>`
is the fragment owned by one thread, not the full matrix tile handled by the
warp.

### Mbarriers

An mbarrier is a shared-memory synchronization object used by newer NVIDIA
asynchronous mechanisms. The dialect models groups of mbarriers with
`!nvgpu.mbarrier.group<memorySpace = ...>`, individual barrier actions, and
tokens such as `!nvgpu.mbarrier.token`.

You create the group, initialize a barrier with an expected thread count,
arrive at it, and test or wait for phase completion.

### Tensor Memory Accelerator

TMA uses a tensor map descriptor to describe tiled memory movement. In MLIR this
is represented by `!nvgpu.tensormap.descriptor<...>`.

The descriptor carries the tensor memref type and layout controls:

- `swizzle`: `none`, `swizzle_32b`, `swizzle_64b`, or `swizzle_128b`.
- `l2promo`: `none`, `l2promo_64b`, `l2promo_128b`, or `l2promo_256b`.
- `oob`: `zero` or `nan` for out-of-bounds fill behavior.
- `interleave`: `none`, `interleave_16b`, or `interleave_32b`.

TMA loads use mbarriers for completion tracking.

### Warpgroup MMA

Warpgroup MMA, or WGMMA, is larger than normal warp-level MMA. It represents a
cooperative matrix multiply across a warpgroup.

The dialect models WGMMA with:

- `!nvgpu.warpgroup.descriptor<tensor = ...>` for shared-memory matrix
  descriptors.
- `!nvgpu.warpgroup.accumulator<fragmented = vector<...>>` for distributed
  accumulator fragments.
- Operations that initialize, execute, and store the accumulator.

## Operations

### Device Async Copy Operations

- `nvgpu.device_async_copy` starts an asynchronous device-side copy from source
  memory to destination memory, usually global to shared. It returns a
  `!nvgpu.device.async.token`.
- `nvgpu.device_async_create_group` groups one or more async copy tokens into a
  group token.
- `nvgpu.device_async_wait` waits for one or more async groups. Its optional
  `numGroups` attribute can express a wait that allows a bounded number of
  groups to remain in flight.

These operations usually appear after vector transfers have been mapped to a
GPU memory movement pattern.

### Matrix Fragment And MMA Operations

- `nvgpu.ldmatrix` loads a matrix fragment from shared memory into vector
  registers. It is the structured NVGPU counterpart to NVIDIA `ldmatrix`.
- `nvgpu.mma.sync` represents warp-level dense matrix multiply-accumulate.
- `nvgpu.mma.sp.sync` represents warp-level sparse matrix multiply-accumulate
  with structured sparse metadata.

These operations are important for Tensor Core lowering. Their `mmaShape`
attributes describe the warp-level MMA shape, while their vector operands
describe per-thread fragments.

### Mbarrier Operations

- `nvgpu.mbarrier.create` creates one or more shared-memory mbarrier objects.
- `nvgpu.mbarrier.get` gets a pointer-like value for a specific barrier in a
  barrier group.
- `nvgpu.mbarrier.init` initializes a barrier with an expected participant
  count.
- `nvgpu.mbarrier.arrive` records arrival and returns an
  `!nvgpu.mbarrier.token`.
- `nvgpu.mbarrier.arrive.nocomplete` records arrival without completing the
  current phase.
- `nvgpu.mbarrier.arrive.expect_tx` increases the transaction count expected by
  a barrier.
- `nvgpu.mbarrier.test.wait` tests whether the current barrier phase has
  completed.
- `nvgpu.mbarrier.try_wait.parity` waits for a phase parity with a tick limit.

Mbarriers are low-level synchronization objects. They are easy to misuse if the
program does not maintain the expected phase and transaction discipline.

### Tensor Memory Accelerator Operations

- `nvgpu.tma.create.descriptor` creates a tensor map descriptor for a tiled
  memory region.
- `nvgpu.tma.fence.descriptor` fences a tensor map descriptor before use.
- `nvgpu.tma.prefetch.descriptor` prefetches the descriptor cache line.
- `nvgpu.tma.async.load` performs a TMA asynchronous load into shared memory and
  uses an mbarrier for completion.
- `nvgpu.tma.async.store` performs a TMA asynchronous store through a tensor map
  descriptor.

TMA is useful for large structured memory transfers that the hardware can
perform efficiently.

### Warpgroup Operations

- `nvgpu.warpgroup.generate.descriptor` builds a WGMMA matrix descriptor from a
  shared-memory tensor and tensor map.
- `nvgpu.warpgroup.mma.init.accumulator` initializes a distributed accumulator.
- `nvgpu.warpgroup.mma` performs warpgroup-level matrix multiply-accumulate.
- `nvgpu.warpgroup.mma.store` stores a fragmented warpgroup accumulator to a
  memref.

These operations are target-specific and require shapes and memory layouts that
match NVIDIA WGMMA constraints.

### Reciprocal Operation

- `nvgpu.rcp` represents reciprocal calculation for `vector` values of `f32`.
  It supports attributes such as `approx`, `ftz`, and the rounding mode used by
  lowering.

This operation gives the lowering pipeline an NVIDIA-aware reciprocal operation
instead of leaving the choice entirely to generic math lowering.

## Transformations

The native NVGPU pass is:

- `nvgpu-optimize-shared-memory`: optimizes accesses to shared-memory memrefs to
  reduce bank conflicts. It depends on the `memref` and `vector` dialects.

The NVGPU transform dialect operations are:

- `transform.nvgpu.create_async_groups`: looks for global-to-shared vector
  transfer patterns, converts them to `nvgpu.device_async_copy`, groups
  consecutive copies, and inserts waits. It can add the `bypassL1` hint through
  its `bypass_l1` attribute when the transfer size allows it.
- `transform.nvgpu.pipeline_shared_memory_copies`: software-pipelines an
  `scf.for` loop to overlap shared-memory copies with the rest of the loop. It
  uses attributes such as `depth`, `peel_epilogue`, and
  `failure_propagation_mode`.
- `transform.nvgpu.rewrite_matmul_as_mma_sync`: rewrites suitable memref-based
  `linalg` matmul operations to `nvgpu.mma.sync` operations, inserting required
  memory copies where possible.
- `transform.nvgpu.rewrite_copy_as_tma`: rewrites suitable memref copy
  operations into TMA operations that move data through shared memory.
- `transform.apply_conversion_patterns.nvgpu.nvgpu_to_nvvm`: collects
  conversion patterns that lower NVGPU operations to NVVM operations. These
  patterns require an LLVM type converter.

These transformations are structural. They work only when the surrounding IR has
the expected loops, memory spaces, vector transfer shapes, and matmul forms.

## Conversions And Lowering Paths

The main conversion pass is:

- `convert-nvgpu-to-nvvm`: converts supported `nvgpu` operations to `nvvm`
  dialect intrinsics and related LLVM dialect operations.

There is also an important path into `nvgpu`:

- `convert-vector-to-gpu` has a `use-nvgpu` option. When enabled, eligible
  vector operations can lower toward NVGPU operations instead of only generic
  GPU operations.

Typical lowering relationships:

```text
vector.transfer_read from shared memory
  -> nvgpu.ldmatrix
  -> nvvm.ldmatrix-style operation
  -> LLVM/NVPTX/PTX

vector.contract or linalg.matmul after GPU tiling
  -> nvgpu.mma.sync or nvgpu.mma.sp.sync
  -> nvvm.mma.sync-style operation
  -> LLVM/NVPTX/PTX

global-to-shared vector transfers
  -> nvgpu.device_async_copy/group/wait
  -> nvvm cp.async-style operations
  -> LLVM/NVPTX/PTX

structured tile copies for supported NVIDIA targets
  -> nvgpu.tma.* and nvgpu.mbarrier.*
  -> nvvm TMA and mbarrier operations
  -> LLVM/NVPTX/PTX

warpgroup matrix multiply
  -> nvgpu.warpgroup.*
  -> nvvm.wgmma.* operations
  -> LLVM/NVPTX/PTX
```

The implication is that `nvgpu` is not usually the end of the pipeline. It is a
target-aware staging dialect that lets the compiler express NVIDIA GPU choices
before final lowering.

## Example IR

### Loading A Matrix Fragment

```mlir
func.func @ldmatrix(%arg0: memref<?x?xf16, 3>, %x: index, %y: index) {
  %l = nvgpu.ldmatrix %arg0[%x, %y] {numTiles = 4 : i32, transpose = false} :
    memref<?x?xf16, 3> -> vector<4x2xf16>
  return
}
```

This example loads a matrix fragment from shared memory address space `3`.

### Warp-Level MMA

```mlir
func.func @mma_sync(%arg0: vector<4x2xf16>,
                    %arg1: vector<2x2xf16>,
                    %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  %d = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} :
    (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}
```

The `mmaShape` is the warp-level operation shape. The vector types describe the
per-thread fragments.

### Sparse MMA

```mlir
func.func @mma_sp_sync(%a: vector<4x2xf16>,
                       %b: vector<4x2xf16>,
                       %c: vector<2x2xf16>,
                       %meta: vector<2xi16>) -> vector<2x2xf16> {
  %d = nvgpu.mma.sp.sync(%a, %b, %c) metadata(%meta) {mmaShape = [16, 8, 32]} :
    (vector<4x2xf16>, vector<4x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}
```

Sparse MMA adds metadata that describes the structured sparse layout.

### Async Copy Group

```mlir
func.func @async_copy(%dst : memref<2x7x5xf32, 3>, %src : memref<4x5xf32>) {
  %c0 = arith.constant 0 : index
  %copy = nvgpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4
    : memref<4x5xf32> to memref<2x7x5xf32, 3>
  %group = nvgpu.device_async_create_group %copy
  nvgpu.device_async_wait %group {numGroups = 1 : i32}
  return
}
```

The copy is started, grouped, and then waited on before the function returns.

### Mbarrier Lifecycle

```mlir
!barrierType = !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

func.func @mbarrier_example() {
  %threads = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %barrier = nvgpu.mbarrier.create -> !barrierType
  nvgpu.mbarrier.init %barrier[%c0], %threads : !barrierType
  %token = nvgpu.mbarrier.arrive %barrier[%c0] : !barrierType -> !tokenType
  %done = nvgpu.mbarrier.test.wait %barrier[%c0], %token : !barrierType, !tokenType
  return
}
```

This shows the basic pattern: create, initialize, arrive, and test completion.

### TMA Load

```mlir
!tensorMap1d = !nvgpu.tensormap.descriptor<tensor = memref<128xf32,3>, swizzle=none, l2promo = none, oob = nan, interleave = none>
!mbarrier = !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>>

func.func @tma_load(%tensorMap: !tensorMap1d, %buffer: memref<128xf32,3>, %mbarrier: !mbarrier) {
  %c0 = arith.constant 0 : index
  nvgpu.tma.async.load %tensorMap[%c0], %mbarrier[%c0] to %buffer
    : !tensorMap1d, !mbarrier -> memref<128xf32,3>
  return
}
```

The TMA load uses a tensor map descriptor and an mbarrier for asynchronous
completion tracking.

### Warpgroup MMA

```mlir
func.func @warpgroup_mma(%descA: !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>,
                         %descB: !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>,
                         %dst: memref<128x128xf32, 3>) {
  %acc = nvgpu.warpgroup.mma.init.accumulator
    -> !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
  %out = nvgpu.warpgroup.mma %descA, %descB, %acc {transposeB} :
    !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>,
    !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>,
    !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
    -> !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
  nvgpu.warpgroup.mma.store %out, %dst :
    !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
    to memref<128x128xf32, 3>
  return
}
```

The accumulator is a fragmented value collectively owned by the warpgroup.

### Reciprocal

```mlir
func.func @rcp(%in: vector<32x16xf32>) -> vector<32x16xf32> {
  %out = nvgpu.rcp %in {approx = true, ftz = true} : vector<32x16xf32>
  return %out : vector<32x16xf32>
}
```

The reciprocal operation is specialized for NVIDIA lowering and currently
targets `f32` vector values.

## Mental Model

The `nvgpu` dialect means:

```text
The compiler has chosen NVIDIA GPU behavior, but it still wants structured MLIR
operations before final NVVM/PTX lowering.
```

A beginner can read `nvgpu` IR as a set of explicit hardware commitments:

- This data lives in global or shared memory.
- This copy can run asynchronously.
- This wait or mbarrier controls when async work is complete.
- This vector fragment participates in a Tensor Core operation.
- This descriptor tells NVIDIA hardware how to move or multiply a tile.

Those commitments make the IR less portable but more useful for generating fast
NVIDIA GPU code.

## Gotchas

- `nvgpu` is NVIDIA-specific. Do not expect it to lower to non-NVIDIA GPU
  backends.
- `nvgpu` is not as low-level as `nvvm`. If you are looking for final PTX-like
  operations, inspect the IR after `convert-nvgpu-to-nvvm`.
- Memory spaces are part of the contract. Shared-memory operations need shared
  memory memrefs.
- Matrix operations use per-thread fragments. The vector shape is not the full
  logical matrix tile.
- `mmaShape`, TMA descriptor parameters, and WGMMA descriptor types must match
  hardware-supported forms.
- Async copies require correct grouping and waiting. Starting a copy is not the
  same as making the data available.
- Mbarriers require correct phase, count, and transaction handling.
- TMA operations require descriptor setup and target support.
- Transform ops are pattern-based. If the IR shape does not match their
  preconditions, they may leave operations unchanged or report a failure.
- Shared-memory bank conflicts are performance bugs, not type errors. The
  `nvgpu-optimize-shared-memory` pass exists because valid shared-memory IR can
  still be slow.

## Source Map

Primary source files in the LLVM tree:

- `mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td`
- `mlir/include/mlir/Dialect/NVGPU/IR/NVGPUOps.td`
- `mlir/include/mlir/Dialect/NVGPU/IR/NVGPUTypes.td`
- `mlir/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td`
- `mlir/include/mlir/Dialect/NVGPU/Transforms/Passes.td`
- `mlir/lib/Dialect/NVGPU/Transforms/`
- `mlir/lib/Dialect/NVGPU/TransformOps/`
- `mlir/lib/Conversion/NVGPUToNVVM/`
- `mlir/lib/Conversion/VectorToGPU/`

Useful tests:

- `mlir/test/Dialect/NVGPU/`
- `mlir/test/Conversion/NVGPUToNVVM/`
