# `nvvm` Dialect

## Beginner Summary

The `nvvm` dialect is MLIR's NVIDIA-specific GPU target dialect.

It models operations that are close to NVIDIA's NVVM IR, LLVM NVPTX intrinsics,
and PTX instructions. Generic GPU concepts such as blocks, threads, barriers,
warp collectives, async copies, tensor-core instructions, and special registers
become concrete NVIDIA operations in this dialect.

Think of `nvvm` as the point where the compiler has stopped saying "some GPU"
and has started saying "NVIDIA GPU with a particular PTX/NVVM feature set."

Most users should not start a pipeline by writing NVVM. Higher-level IR usually
flows through `gpu`, `nvgpu`, `vector`, `math`, and `llvm` first, then lowers to
`nvvm` before final LLVM/PTX emission.

## Why This Dialect Exists

Generic GPU IR is intentionally portable. For example, `gpu.thread_id x` means
"the current thread's x coordinate" without committing to CUDA, HIP, SPIR-V, or
any specific target.

NVIDIA GPUs have many target-specific features that need a lower-level
representation:

- PTX special registers such as thread id, block id, lane id, warp id, timers,
  and cluster ids.
- NVIDIA-specific warp collectives such as shuffle, vote, match, elect, and
  redux.
- CTA, warp, cluster, proxy, and memory barrier instructions.
- `cp.async`, TMA, and mbarrier operations.
- WMMA, MMA, WGMMA, sparse MMA, block-scale MMA, and fifth-generation tensor
  core operations.
- NVVM address spaces and kernel attributes.
- Inline PTX escape hatches for instructions that are not represented directly.

The `nvvm` dialect exists so MLIR can represent those NVIDIA-specific decisions
while still keeping the IR typed, verifiable, and connected to the rest of the
MLIR lowering pipeline.

## When It Matters

The `nvvm` dialect matters late in a CUDA/NVIDIA GPU compilation pipeline.

Typical pipeline shape:

```text
linalg / tensor / scf / vector
  -> tiling, mapping, vectorization, bufferization
  -> gpu.module / gpu.func / gpu.launch_func
  -> nvgpu for NVIDIA-specific structured GPU operations
  -> convert-gpu-to-nvvm
  -> convert-nvgpu-to-nvvm
  -> convert-math-to-nvvm
  -> llvm-optimize-for-nvvm-target
  -> convert-nvvm-to-llvm
  -> LLVM dialect translation and NVPTX/PTX code generation
```

It is also the dialect you inspect when debugging whether GPU lowering selected
the right NVIDIA primitive: the right special register, memory space, barrier,
async copy, or tensor-core instruction should be visible here.

## When To Use It

Use `nvvm` when your IR already targets NVIDIA GPUs and must express a PTX or
NVVM-specific operation.

Use it for:

- Reading NVIDIA special registers.
- Representing CUDA/NVIDIA warp-level collectives.
- Expressing CTA, warp, cluster, memory, proxy, and mbarrier synchronization.
- Modeling asynchronous copies and tensor memory accelerator operations.
- Lowering NVIDIA matrix instructions such as WMMA, MMA, WGMMA, and tcgen05.
- Carrying NVVM address spaces, target attributes, and kernel metadata.
- Emitting inline PTX for NVIDIA instructions that need a direct escape hatch.

Do not use `nvvm` as the first representation of portable GPU work. Start with
`gpu`, `scf`, `vector`, `linalg`, or `nvgpu` where possible, then lower to
`nvvm` when the target is known.

## Core Concepts

### NVVM Is Target-Specific

`nvvm` is not a portable GPU abstraction. It represents NVIDIA semantics.

That has two consequences:

- Many ops map directly to NVVM intrinsics or PTX instructions.
- Some ops require a minimum SM architecture, such as SM 80, SM 90, or newer
  architecture families.

The `#nvvm.target` attribute can verify whether the operations in a GPU module
are compatible with the selected chip.

### It Depends On The LLVM Dialect

The NVVM dialect depends on the LLVM dialect. You will frequently see:

- `llvm.func` and `llvm.return`.
- LLVM pointer types such as `!llvm.ptr<1>` and `!llvm.ptr<3>`.
- LLVM struct types for multi-result fragments.
- NVVM operations mixed with LLVM dialect operations.

This is normal. NVVM is a target dialect near LLVM lowering, not a high-level
GPU programming dialect.

### Address Spaces Matter

NVVM uses NVIDIA memory spaces through LLVM pointer address spaces:

| Address space | Meaning |
| --- | --- |
| `0` | Generic memory. |
| `1` | Global memory. |
| `3` | Shared memory. |
| `4` | Constant memory. |
| `5` | Local memory. |
| `6` | Tensor memory. |
| `7` | Shared cluster memory. |

You may also see the attribute form, such as `#nvvm.memory_space<global>` or
`#nvvm.memory_space<shared>`, when memory space metadata is carried as an
attribute.

### Kernel And Target Metadata

The dialect defines attributes used to mark NVIDIA kernels and tune launches:

| Attribute | Meaning |
| --- | --- |
| `nvvm.kernel` | Marks an LLVM function as an NVVM kernel. |
| `nvvm.maxntid` | Maximum threads per CTA. |
| `nvvm.reqntid` | Required threads per CTA. |
| `nvvm.cluster_dim` | Required CTAs per cluster. |
| `nvvm.cluster_max_blocks` | Maximum CTAs per cluster. |
| `nvvm.minctasm` | Minimum CTAs per SM. |
| `nvvm.maxnreg` | Maximum registers per thread. |
| `nvvm.grid_constant` | Marks a kernel argument as a grid constant. |
| `nvvm.blocksareclusters` | Says the grid launch shape is specified in clusters. |
| `nvvm.managed` | Marks a global as managed memory. |

The target attribute is attached to `gpu.module`:

```mlir
gpu.module @kernels [#nvvm.target<chip = "sm_90", flags = {fast}>] {
}
```

Important `#nvvm.target` fields include optimization level `O`, target triple,
chip, features, flags, linked files, and `verifyTarget`.

### SM Requirements

Many NVVM operations have minimum SM requirements. For example, some cluster
and tensor-memory operations require SM 90 or newer, and tcgen05 operations
target newer architecture families.

If the `#nvvm.target` attribute enables target verification, the verifier can
reject operations that do not match the selected chip.

### Inline PTX Escape Hatch

`nvvm.inline_ptx` lets the IR contain direct PTX text.

It is useful for target features that need an exact PTX instruction before a
structured MLIR op exists. It is also a tradeoff: inline PTX is less portable,
harder for MLIR to analyze, and usually belongs at the very end of lowering.

## Types And Attributes

The dialect does not introduce a large standalone type system. It mostly uses
LLVM dialect types and NVVM attributes.

Important attribute families include:

| Family | Examples |
| --- | --- |
| Floating-point modes | `#nvvm.fp_rnd_mode<rn>`, `#nvvm.sat_mode<satfinite>`. |
| Memory metadata | `#nvvm.memory_space<shared>`, `#nvvm.mem_scope<gpu>`, `#nvvm.mem_order<release>`. |
| Reductions | `#nvvm.redux_kind<add>`, `#nvvm.reduction<popc>`. |
| Warp collectives | shuffle, vote, and match kind attributes. |
| Cache and prefetch | cache eviction and prefetch-level attributes. |
| MMA metadata | shapes, layouts, operand types, fragment kinds, overflow modes. |
| TMA metadata | TMA load/store modes, CTA group, reduction kind. |
| WGMMA metadata | scale-in, scale-out, and WGMMA operand type attributes. |
| Tcgen05 metadata | tcgen05 copy, load/store, wait, MMA, collector, and block-scale attributes. |
| Tensor maps | tensor-map field, element type, swizzle, interleave, atomicity, and fill-mode attributes. |
| Target metadata | `#nvvm.target<...>`. |

Beginners should not try to memorize these. Read the operation syntax and then
look up the attribute family that operation uses.

## Operations

The dialect has 208 operations. The catalog below groups every operation by
feature family.

### Special Register Reads

Thread, block, grid, lane, warp, SM, cluster, timer, shared-memory-size, and
environment-register reads:

- `nvvm.read.ptx.sreg.tid.x`, `nvvm.read.ptx.sreg.tid.y`,
  `nvvm.read.ptx.sreg.tid.z`
- `nvvm.read.ptx.sreg.ntid.x`, `nvvm.read.ptx.sreg.ntid.y`,
  `nvvm.read.ptx.sreg.ntid.z`
- `nvvm.read.ptx.sreg.ctaid.x`, `nvvm.read.ptx.sreg.ctaid.y`,
  `nvvm.read.ptx.sreg.ctaid.z`
- `nvvm.read.ptx.sreg.nctaid.x`, `nvvm.read.ptx.sreg.nctaid.y`,
  `nvvm.read.ptx.sreg.nctaid.z`
- `nvvm.read.ptx.sreg.laneid`, `nvvm.read.ptx.sreg.warpsize`,
  `nvvm.read.ptx.sreg.warpid`, `nvvm.read.ptx.sreg.nwarpid`
- `nvvm.read.ptx.sreg.smid`, `nvvm.read.ptx.sreg.nsmid`,
  `nvvm.read.ptx.sreg.gridid`
- `nvvm.read.ptx.sreg.lanemask.eq`, `nvvm.read.ptx.sreg.lanemask.le`,
  `nvvm.read.ptx.sreg.lanemask.lt`, `nvvm.read.ptx.sreg.lanemask.ge`,
  `nvvm.read.ptx.sreg.lanemask.gt`
- `nvvm.read.ptx.sreg.clusterid.x`, `nvvm.read.ptx.sreg.clusterid.y`,
  `nvvm.read.ptx.sreg.clusterid.z`
- `nvvm.read.ptx.sreg.nclusterid.x`, `nvvm.read.ptx.sreg.nclusterid.y`,
  `nvvm.read.ptx.sreg.nclusterid.z`
- `nvvm.read.ptx.sreg.cluster.ctaid.x`,
  `nvvm.read.ptx.sreg.cluster.ctaid.y`,
  `nvvm.read.ptx.sreg.cluster.ctaid.z`,
  `nvvm.read.ptx.sreg.cluster.ctarank`
- `nvvm.read.ptx.sreg.cluster.nctaid.x`,
  `nvvm.read.ptx.sreg.cluster.nctaid.y`,
  `nvvm.read.ptx.sreg.cluster.nctaid.z`,
  `nvvm.read.ptx.sreg.cluster.nctarank`
- `nvvm.read.ptx.sreg.total.smem.size`,
  `nvvm.read.ptx.sreg.dynamic.smem.size`,
  `nvvm.read.ptx.sreg.aggr.smem.size`
- `nvvm.read.ptx.sreg.clock`, `nvvm.read.ptx.sreg.clock64`,
  `nvvm.read.ptx.sreg.globaltimer`,
  `nvvm.read.ptx.sreg.globaltimer.lo`
- `nvvm.read.ptx.sreg.envreg0`, `nvvm.read.ptx.sreg.envreg1`,
  `nvvm.read.ptx.sreg.envreg2`, `nvvm.read.ptx.sreg.envreg3`,
  `nvvm.read.ptx.sreg.envreg4`, `nvvm.read.ptx.sreg.envreg5`,
  `nvvm.read.ptx.sreg.envreg6`, `nvvm.read.ptx.sreg.envreg7`,
  `nvvm.read.ptx.sreg.envreg8`, `nvvm.read.ptx.sreg.envreg9`,
  `nvvm.read.ptx.sreg.envreg10`, `nvvm.read.ptx.sreg.envreg11`,
  `nvvm.read.ptx.sreg.envreg12`, `nvvm.read.ptx.sreg.envreg13`,
  `nvvm.read.ptx.sreg.envreg14`, `nvvm.read.ptx.sreg.envreg15`,
  `nvvm.read.ptx.sreg.envreg16`, `nvvm.read.ptx.sreg.envreg17`,
  `nvvm.read.ptx.sreg.envreg18`, `nvvm.read.ptx.sreg.envreg19`,
  `nvvm.read.ptx.sreg.envreg20`, `nvvm.read.ptx.sreg.envreg21`,
  `nvvm.read.ptx.sreg.envreg22`, `nvvm.read.ptx.sreg.envreg23`,
  `nvvm.read.ptx.sreg.envreg24`, `nvvm.read.ptx.sreg.envreg25`,
  `nvvm.read.ptx.sreg.envreg26`, `nvvm.read.ptx.sreg.envreg27`,
  `nvvm.read.ptx.sreg.envreg28`, `nvvm.read.ptx.sreg.envreg29`,
  `nvvm.read.ptx.sreg.envreg30`, `nvvm.read.ptx.sreg.envreg31`

### Floating-Point Math And Format Conversion

Floating-point arithmetic and approximations:

- `nvvm.addf`, `nvvm.subf`, `nvvm.divf`, `nvvm.fma`
- `nvvm.sqrt`, `nvvm.sqrt.approx`, `nvvm.rsqrt`,
  `nvvm.rcp.approx.ftz.f`
- `nvvm.sin`, `nvvm.cos`, `nvvm.log2`, `nvvm.ex2`
- `nvvm.convert.float.to.tf32`

Packed low-precision conversions:

- `nvvm.convert.f32x2.to.f16x2`, `nvvm.convert.f32x2.to.bf16x2`
- `nvvm.convert.f32x2.to.f4x2`, `nvvm.convert.f32x2.to.f6x2`,
  `nvvm.convert.f32x2.to.f8x2`, `nvvm.convert.f32x2.to.s2f6x2`
- `nvvm.convert.f32x4.to.f4x4`, `nvvm.convert.f32x4.to.f6x4`,
  `nvvm.convert.f32x4.to.f8x4`
- `nvvm.convert.f16x2.to.f4x2`, `nvvm.convert.f16x2.to.f6x2`,
  `nvvm.convert.f16x2.to.f8x2`
- `nvvm.convert.bf16x2.to.f4x2`, `nvvm.convert.bf16x2.to.f6x2`,
  `nvvm.convert.bf16x2.to.f8x2`, `nvvm.convert.bf16x2.to.s2f6x2`
- `nvvm.convert.f4x2.to.f16x2`, `nvvm.convert.f4x2.to.bf16x2`
- `nvvm.convert.f6x2.to.f16x2`, `nvvm.convert.f6x2.to.bf16x2`
- `nvvm.convert.f8x2.to.f16x2`, `nvvm.convert.f8x2.to.bf16x2`
- `nvvm.convert.s2f6x2.to.bf16x2`

### Warp, Vote, Reduction, And Control Operations

Warp-level collectives and small control operations:

- `nvvm.shfl.sync`
- `nvvm.vote.sync`
- `nvvm.match.sync`
- `nvvm.redux.sync`
- `nvvm.elect.sync`
- `nvvm.bar.warp.sync`
- `nvvm.prmt`
- `nvvm.dot.accumulate.2way`, `nvvm.dot.accumulate.4way`
- `nvvm.nanosleep`
- `nvvm.pmevent`
- `nvvm.griddepcontrol`
- `nvvm.setmaxregister`
- `nvvm.exit`
- `nvvm.breakpoint`
- `nvvm.inline_ptx`

### Barriers, Fences, And Cluster Synchronization

CTA, warp, cluster, memory, and proxy synchronization:

- `nvvm.barrier`, `nvvm.barrier.arrive`, `nvvm.barrier.reduction`
- `nvvm.cluster.arrive`, `nvvm.cluster.arrive.relaxed`,
  `nvvm.cluster.wait`
- `nvvm.memory.barrier`
- `nvvm.fence.sc.cluster`
- `nvvm.fence.sync_restrict`
- `nvvm.fence.mbarrier.init`
- `nvvm.fence.proxy`, `nvvm.fence.proxy.acquire`,
  `nvvm.fence.proxy.release`, `nvvm.fence.proxy.sync_restrict`
- `nvvm.clusterlaunchcontrol.try.cancel`,
  `nvvm.clusterlaunchcontrol.query.cancel`

### MBarrier Operations

Memory barrier object operations:

- `nvvm.mbarrier.init`
- `nvvm.mbarrier.inval`
- `nvvm.mbarrier.expect_tx`
- `nvvm.mbarrier.complete_tx`
- `nvvm.mbarrier.arrive`
- `nvvm.mbarrier.arrive.nocomplete`
- `nvvm.mbarrier.arrive.expect_tx`
- `nvvm.mbarrier.arrive_drop`
- `nvvm.mbarrier.arrive_drop.nocomplete`
- `nvvm.mbarrier.arrive_drop.expect_tx`
- `nvvm.mbarrier.test.wait`
- `nvvm.mbarrier.try_wait`
- `nvvm.mbarrier.try_wait.parity`

### Async Copy, TMA, And Memory Movement

Asynchronous copy and bulk memory operations:

- `nvvm.cp.async.shared.global`
- `nvvm.cp.async.commit.group`
- `nvvm.cp.async.wait.group`
- `nvvm.cp.async.mbarrier.arrive`
- `nvvm.cp.async.bulk.commit.group`
- `nvvm.cp.async.bulk.wait_group`
- `nvvm.cp.async.bulk.global.shared.cta`
- `nvvm.cp.async.bulk.shared.cluster.global`
- `nvvm.cp.async.bulk.shared.cluster.shared.cta`
- `nvvm.cp.async.bulk.prefetch`
- `nvvm.cp.async.bulk.tensor.global.shared.cta`
- `nvvm.cp.async.bulk.tensor.shared.cluster.global`
- `nvvm.cp.async.bulk.tensor.prefetch`
- `nvvm.cp.async.bulk.tensor.reduce`
- `nvvm.prefetch`
- `nvvm.st.bulk`
- `nvvm.mapa`
- `nvvm.tensormap.replace`

### Matrix And Tensor-Core Operations

Matrix load/store and movement:

- `nvvm.ldmatrix`
- `nvvm.stmatrix`
- `nvvm.movmatrix`

Classic WMMA and MMA operations:

- `nvvm.wmma.load`
- `nvvm.wmma.mma`
- `nvvm.wmma.store`
- `nvvm.mma.sync`
- `nvvm.mma.sp.sync`
- `nvvm.mma.block_scale`
- `nvvm.mma.sp.block_scale`

Warp-group MMA operations:

- `nvvm.wgmma.fence.aligned`
- `nvvm.wgmma.mma_async`
- `nvvm.wgmma.commit.group.sync.aligned`
- `nvvm.wgmma.wait.group.sync.aligned`

Fifth-generation tensor-core operations:

- `nvvm.tcgen05.alloc`
- `nvvm.tcgen05.dealloc`
- `nvvm.tcgen05.relinquish_alloc_permit`
- `nvvm.tcgen05.fence`
- `nvvm.tcgen05.wait`
- `nvvm.tcgen05.commit`
- `nvvm.tcgen05.shift`
- `nvvm.tcgen05.cp`
- `nvvm.tcgen05.ld`
- `nvvm.tcgen05.ld.red`
- `nvvm.tcgen05.st`
- `nvvm.tcgen05.mma_smem_desc`
- `nvvm.tcgen05.mma`
- `nvvm.tcgen05.mma.sp`
- `nvvm.tcgen05.mma.block_scale`
- `nvvm.tcgen05.mma.sp.block_scale`
- `nvvm.tcgen05.mma.ws`
- `nvvm.tcgen05.mma.ws.sp`

## Transformations

### Target Attachment

`nvvm-attach-target` attaches `#nvvm.target` to matching `gpu.module`
operations.

Important options include:

| Option | Meaning |
| --- | --- |
| `module` | Regex used to select GPU modules. |
| `triple` | Target triple, defaulting to `nvptx64-nvidia-cuda`. |
| `chip` | Target chip, defaulting to `sm_75`. |
| `features` | Target feature string. |
| `O` | Optimization level. |
| `fast` | Enable fast math mode in target flags. |
| `ftz` | Enable flush-to-zero mode in target flags. |
| `l` | Extra bitcode libraries to link. |
| `ptxas-cmd-options` | Extra downstream compiler options. |
| `verify-target-arch` | Enable or disable target architecture verification. |

### Producing NVVM

| Pass | Role |
| --- | --- |
| `convert-gpu-to-nvvm` | Lowers GPU dialect device operations inside a `gpu.module` to NVVM operations. |
| `convert-nvgpu-to-nvvm` | Lowers NVGPU operations to NVVM intrinsics and PTX-like ops. |
| `convert-math-to-nvvm` | Converts supported Math dialect operations to CUDA libdevice calls. |

`convert-gpu-to-nvvm` handles generic device concepts such as special
registers and barriers. `convert-nvgpu-to-nvvm` handles NVIDIA-specific
structured operations such as async copies, mbarriers, and matrix instructions.

### Optimizing For NVVM

`llvm-optimize-for-nvvm-target` currently optimizes LLVM/NVVM-shaped IR for the
NVVM target. One important pattern expands `llvm.fdiv` on `f16` into a faster
sequence using `nvvm.rcp.approx.ftz.f`, fp32 arithmetic, and one Newton-style
refinement before truncating back to fp16.

This pass is target-aware cleanup near the LLVM/NVVM boundary.

## Conversions And Lowering Paths

The final direct conversion pass is:

| Pass | Target |
| --- | --- |
| `convert-nvvm-to-llvm` | Converts NVVM operations that build PTX through `BasicPtxBuilderInterface` into LLVM dialect inline assembly. |

Not every NVVM operation becomes inline assembly through this pass. Many NVVM
operations translate to LLVM NVVM intrinsics during LLVM IR translation. The
important idea is that after NVVM lowering, the compiler has enough information
to emit NVIDIA-specific LLVM IR or PTX.

A common lowering path is:

```text
gpu.func / gpu.thread_id / gpu.barrier / nvgpu.*
  -> convert-gpu-to-nvvm
  -> convert-nvgpu-to-nvvm
  -> nvvm.read.ptx.sreg.*, nvvm.barrier, nvvm.cp.async.*, nvvm.mma.*
  -> convert-nvvm-to-llvm
  -> llvm.inline_asm for PTX-builder ops plus LLVM/NVVM intrinsics
  -> LLVM NVPTX backend / PTX tooling
```

## Example IR

### Reading Thread Coordinates

```mlir
func.func @thread_x() -> i32 {
  %tid = nvvm.read.ptx.sreg.tid.x : i32
  return %tid : i32
}
```

### Barrier And Warp Vote

```mlir
llvm.func @warp_vote(%mask: i32, %pred: i1) -> i32 {
  nvvm.barrier
  %vote = nvvm.vote.sync ballot %mask, %pred -> i32
  llvm.return %vote : i32
}
```

### Floating-Point NVVM Operation

```mlir
func.func @fast_add(%x: f32, %y: f32) -> f32 {
  %sum = nvvm.addf %x, %y : f32
  %root = nvvm.sqrt.approx %sum : f32
  return %root : f32
}
```

### Async Copy From Global To Shared

```mlir
func.func @async_copy(%dst: !llvm.ptr<3>, %src: !llvm.ptr<1>) {
  nvvm.cp.async.shared.global %dst, %src, 16, cache = ca
      : !llvm.ptr<3>, !llvm.ptr<1>
  nvvm.cp.async.commit.group
  nvvm.cp.async.wait.group 0
  return
}
```

## Mental Model

Read NVVM IR as "NVIDIA GPU assembly-level intent, still in MLIR form."

- `gpu` decides that code runs on a GPU.
- `nvgpu` expresses NVIDIA-specific structured GPU operations.
- `nvvm` expresses the final NVIDIA primitive: the special register, barrier,
  warp operation, async copy, or tensor-core instruction.
- LLVM translation and NVPTX code generation turn those primitives into LLVM
  intrinsics, inline PTX, and final PTX/cubin artifacts.

The closer you are to `nvvm`, the less portable the IR is and the more precise
the target contract becomes.

## Gotchas

### NVVM Is Not Portable GPU IR

If you want one pipeline that can target NVIDIA, AMD, SPIR-V, and other GPU
backends, keep most transformations above NVVM. Use NVVM only after selecting
the NVIDIA path.

### SM Requirements Are Real

Some operations are not legal on older NVIDIA architectures. Attach and verify
`#nvvm.target` early enough to catch mismatches before code generation.

### Address Spaces Must Match The Instruction

Many operations require pointers in a specific address space. For example,
`nvvm.cp.async.shared.global` copies from global memory to shared memory and
uses pointer types such as `!llvm.ptr<1>` and `!llvm.ptr<3>`.

### Inline PTX Is Powerful But Opaque

`nvvm.inline_ptx` can express almost anything PTX can express, but it hides
semantics from MLIR analyses and transformations. Prefer structured NVVM ops
when they exist.

### Many Ops Are Generated By Lowering

Most programmers should not hand-write `nvvm.mma.*`, `nvvm.cp.async.*`, or
`nvvm.tcgen05.*` from scratch. These are usually produced by `nvgpu`, `gpu`,
or target-specific lowering passes.

### NVVM Uses LLVM Types

Seeing LLVM structs, LLVM pointer address spaces, and `llvm.func` around NVVM
ops is expected. This dialect is already near the LLVM lowering boundary.

## Source Map

Use these source files when you want to inspect or update the dialect:

| Area | File |
| --- | --- |
| Dialect definition | `mlir/include/mlir/Dialect/LLVMIR/NVVMDialect.td` |
| Operations and attributes | `mlir/include/mlir/Dialect/LLVMIR/NVVMOps.td` |
| Enum attributes | `mlir/include/mlir/Dialect/LLVMIR/NVVMEnums.td` |
| SM requirement traits | `mlir/include/mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.td` |
| Dialect implementation | `mlir/lib/Dialect/LLVMIR/IR/NVVMDialect.cpp` |
| SM trait implementation | `mlir/lib/Dialect/LLVMIR/IR/NVVMRequiresSMTraits.cpp` |
| NVVM optimization pass | `mlir/lib/Dialect/LLVMIR/Transforms/OptimizeForNVVM.cpp` |
| GPU-to-NVVM conversion tests | `mlir/test/Conversion/GPUToNVVM/` |
| NVGPU-to-NVVM conversion tests | `mlir/test/Conversion/NVGPUToNVVM/` |
| NVVM-to-LLVM conversion tests | `mlir/test/Conversion/NVVMToLLVM/` |
| Dialect tests | `mlir/test/Dialect/LLVMIR/nvvm*.mlir` and `mlir/test/Dialect/LLVMIR/nvvm/` |
