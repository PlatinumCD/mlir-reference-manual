# `rocdl` Dialect

## Beginner Summary

The `rocdl` dialect is MLIR's AMD GPU LLVM dialect. It represents AMDGPU
LLVM intrinsics, AMD GPU kernel attributes, and the `#rocdl.target` attribute
used by MLIR's GPU compilation pipeline.

Most beginners should not write `rocdl` directly. You usually write higher-level
IR in dialects such as `linalg`, `vector`, `gpu`, `amdgpu`, `math`, or
`complex`, and then a lowering pipeline produces `rocdl` near the end. When you
see `rocdl`, the compiler has moved from "portable GPU program" to "AMD GPU
backend details".

`rocdl` is important because it is the bridge between MLIR GPU code and AMDGPU
LLVM IR. It names concrete hardware concepts: workitem IDs, workgroup IDs,
wavefront operations, barriers, raw buffers, LDS movement, matrix instructions,
low-precision conversion instructions, AMD device-library calls, and AMD GPU
target options.

## Why This Dialect Exists

MLIR has a general `llvm` dialect for LLVM IR concepts. Some targets also need
target-specific LLVM intrinsics. AMD GPUs have many of these intrinsics, and
they are not portable across targets. `rocdl` gives those AMD-specific
intrinsics a typed MLIR form.

The dialect exists so that MLIR can:

- Lower `gpu` operations to AMDGPU intrinsics.
- Lower `amdgpu` operations to final AMDGPU LLVM intrinsic wrappers.
- Preserve AMD-specific kernel metadata and target information.
- Serialize a `gpu.module` to AMDGPU object code through `#rocdl.target`.
- Keep target-specific IR separate from portable GPU, vector, and math IR.

The source dialect documentation states the inclusion rule plainly: operations
in `rocdl` should be wrappers around AMD-specific LLVM intrinsics. New
high-level AMD GPU abstractions should usually live in `amdgpu`, not `rocdl`.

## When It Matters

`rocdl` matters when a pipeline targets AMD GPUs through ROCm or the AMDGPU LLVM
backend.

You care about this dialect when:

- You are lowering `gpu.module` code for an AMD target such as `gfx90a`,
  `gfx942`, `gfx1100`, `gfx1200`, or `gfx1250`.
- You need to inspect the final GPU-side MLIR before translation to LLVM IR.
- You are debugging why a `gpu.thread_id`, `gpu.block_id`, subgroup operation,
  barrier, or memory operation lowered the way it did.
- You are working on AMD-specific matrix instructions such as MFMA, WMMA,
  SMFMAC, or SWMMAC.
- You need AMD-specific low-precision conversion behavior for fp8, bf8, fp6,
  bf6, or fp4 data.
- You need to attach or inspect an AMD GPU compilation target on a `gpu.module`.

It usually does not matter in the early design of a frontend. At that level,
choosing `gpu`, `vector`, `linalg`, and `amdgpu` correctly is more important
than emitting `rocdl` directly.

## When To Use It

Use `rocdl` directly only when you are already at the AMDGPU backend boundary.

Good uses:

- Writing a test for AMDGPU lowering.
- Emitting a specific LLVM AMDGPU intrinsic that has no higher-level wrapper.
- Checking the output of `convert-gpu-to-rocdl` or `convert-amdgpu-to-rocdl`.
- Building a target-specific GPU backend pass.
- Attaching `#rocdl.target` to a `gpu.module` before binary serialization.

Avoid direct use when:

- The operation has a portable `gpu`, `vector`, `memref`, `arith`, or `math`
  representation.
- The operation is an AMD GPU concept but still higher level than an LLVM
  intrinsic. In that case prefer the `amdgpu` dialect.
- You want target-independent IR. `rocdl` commits the IR to AMDGPU semantics.

## Core Concepts

### ROCDL Is An LLVM Intrinsic Dialect

Most `rocdl` operations map to LLVM intrinsic names beginning with
`llvm.amdgcn.*`. For example, `rocdl.workitem.id.x` maps to an AMDGPU intrinsic
that reads the hardware workitem ID register, and `rocdl.mbcnt.lo` /
`rocdl.mbcnt.hi` map to masked bit-count intrinsics used to compute lane IDs.

This is different from a dialect such as `gpu`, where operations describe a
portable GPU program. `rocdl` is closer to assembly-level intent.

### ROCDL Depends On LLVM Types

The dialect depends on the `llvm` dialect. You will see LLVM pointer address
spaces and LLVM-style function operations around `rocdl` operations. Common AMD
GPU address spaces include:

| Address space | Meaning in this chapter |
| --- | --- |
| `1` | Global memory |
| `3` | Shared/workgroup memory, also called LDS |
| `7` | AMDGPU fat raw buffer pointer |
| `8` | AMDGPU buffer resource pointer |
| `9` | AMDGPU fat strided buffer pointer |

### `#rocdl.target`

`#rocdl.target` is a GPU target attribute used by MLIR's GPU offloading
infrastructure. It records how to compile a `gpu.module` for AMDGPU:

```mlir
gpu.module @kernels [#rocdl.target<chip = "gfx90a", flags = {fast, no_wave64}>] {
  // GPU functions lower here.
}
```

Important parameters are:

| Parameter | Meaning |
| --- | --- |
| `O` | Optimization level, default `2`. |
| `triple` | Target triple, default `amdgcn-amd-amdhsa`. |
| `chip` | AMDGPU processor name, default `gfx900`. |
| `features` | Target feature string. |
| `abi` | AMDGPU ABI version, default `600`. |
| `flags` | Target flags such as `wave64`, `no_wave64`, `fast`, `daz`, `finite_only`, `unsafe_math`, and `unsafe_sqrt`. |
| `link` | Extra bitcode files to link. |

### Kernel Attributes

ROCDL also owns AMD-specific function and module attributes. Common examples:

| Attribute | Meaning |
| --- | --- |
| `rocdl.kernel` | Marks an LLVM function as an AMD GPU kernel. |
| `rocdl.flat_work_group_size` | Encodes the flat workgroup size metadata. |
| `rocdl.reqd_work_group_size` | Encodes required x/y/z workgroup size metadata. |
| `rocdl.uniform_work_group_size` | Overrides the default uniform-workgroup assumption. |
| `rocdl.max_flat_work_group_size` | Encodes maximum flat workgroup size metadata. |
| `rocdl.waves_per_eu` | Requests a wave-per-execution-unit constraint. |
| `rocdl.unsafe_fp_atomics` | Enables AMD metadata for unsafe floating-point atomics. |

## Operations

The current local LLVM checkout generated 373 registered `rocdl` operations.
The dialect also declares support for unknown operations because not every
possible ROCDL operation is registered.

For a beginner, the essential groups are:

| Group | Representative ops | What they mean |
| --- | --- | --- |
| Hardware IDs | `rocdl.workitem.id.x`, `rocdl.workgroup.id.x`, `rocdl.wave.id`, `rocdl.wavefrontsize` | Read AMDGPU execution identifiers. |
| Wave operations | `rocdl.ballot`, `rocdl.mbcnt.lo`, `rocdl.mbcnt.hi`, `rocdl.readlane`, `rocdl.ds_bpermute` | Communicate or compute within a wavefront. |
| Barriers and waits | `rocdl.s.barrier`, `rocdl.s.barrier.signal`, `rocdl.s.waitcnt`, `rocdl.wave.barrier` | Coordinate lanes, waves, or memory counters. |
| Memory movement | `rocdl.load.to.lds`, `rocdl.global.load.async.to.lds.b128`, `rocdl.tensor.load.to.lds` | Move data between global memory, LDS, and tensor/LDS forms. |
| Buffers and atomics | `rocdl.make.buffer.rsrc`, `rocdl.raw.ptr.buffer.load`, `rocdl.raw.buffer.atomic.fadd` | Use AMD raw-buffer memory operations. |
| Matrix instructions | `rocdl.mfma.*`, `rocdl.wmma.*`, `rocdl.smfmac.*`, `rocdl.swmmac.*` | Target AMD matrix/tensor-core-like instructions. |
| Low-precision conversion | `rocdl.cvt.*` | Convert and pack fp8, bf8, fp6, bf6, fp4, fp16, bf16, f32 forms. |
| Math intrinsics | `rocdl.sin`, `rocdl.exp2`, `rocdl.sqrt`, `rocdl.fmed3` | AMDGPU math intrinsics and chipset-specific math lowering results. |

### Hardware IDs

These operations read special hardware registers:

```text
rocdl.workitem.id.x, rocdl.workitem.id.y, rocdl.workitem.id.z
rocdl.workgroup.id.x, rocdl.workgroup.id.y, rocdl.workgroup.id.z
rocdl.cluster.id.x, rocdl.cluster.id.y, rocdl.cluster.id.z
rocdl.cluster.workgroup.id.x, rocdl.cluster.workgroup.id.y, rocdl.cluster.workgroup.id.z
rocdl.wave.id, rocdl.wavefrontsize
```

These can carry optional `range` information:

```mlir
%tid = rocdl.workitem.id.x range <i32, 0, 256> : i32
```

Range information is useful because later LLVM optimization can prove bounds on
thread IDs and workgroup IDs.

### Wave-Level Operations

Wave operations communicate between lanes or describe lane masks:

```text
rocdl.ballot
rocdl.mbcnt.lo, rocdl.mbcnt.hi
rocdl.readfirstlane, rocdl.readlane
rocdl.ds_swizzle, rocdl.ds_bpermute
rocdl.update.dpp
rocdl.permlanex16, rocdl.permlanex16.var
rocdl.permlane16.swap, rocdl.permlane16.var
rocdl.permlane32.swap
```

`rocdl.mbcnt.lo` and `rocdl.mbcnt.hi` are often used together to compute a
lane index within a wave64. `rocdl.ballot` creates a mask from a lane predicate.
`rocdl.readfirstlane` broadcasts the first active lane's value. `rocdl.readlane`
reads from a selected lane.

### Synchronization, Waits, And Scheduling

These operations synchronize work or control wait counters:

```text
rocdl.barrier
rocdl.s.barrier
rocdl.wave.barrier
rocdl.s.barrier.init
rocdl.s.barrier.signal
rocdl.s.barrier.signal.var
rocdl.s.barrier.signal.isfirst
rocdl.s.barrier.join
rocdl.s.barrier.leave
rocdl.s.barrier.wait
rocdl.s.get.barrier.state
rocdl.s.get.named.barrier.state
rocdl.s.wakeup.barrier
rocdl.s.waitcnt
rocdl.s.wait.dscnt
rocdl.s.wait.loadcnt
rocdl.s.wait.storecnt
rocdl.s.wait.expcnt
rocdl.s.wait.asynccnt
rocdl.s.wait.tensorcnt
rocdl.asyncmark
rocdl.wait.asyncmark
rocdl.s.sleep
rocdl.s.nop
rocdl.s.setprio
rocdl.sched.barrier
rocdl.sched.group.barrier
rocdl.iglp.opt
```

`rocdl.barrier` has the same expansion as HIP's `__syncthreads()` but is
deprecated in favor of using `gpu.barrier` and letting lowering produce the
right target sequence. `rocdl.s.barrier` is a lower-level barrier intrinsic.
On newer gfx12+ targets, lowering may use `rocdl.s.barrier.signal` plus
`rocdl.s.barrier.wait` instead.

### Memory Movement And LDS

These operations express AMD-specific movement between global memory, LDS, DS,
cluster memory paths, and tensor/LDS forms:

```text
rocdl.load.to.lds
rocdl.load.async.to.lds
rocdl.global.load.lds
rocdl.global.load.async.lds
rocdl.global.load.async.to.lds.b8
rocdl.global.load.async.to.lds.b32
rocdl.global.load.async.to.lds.b64
rocdl.global.load.async.to.lds.b128
rocdl.global.store.async.from.lds.b8
rocdl.global.store.async.from.lds.b32
rocdl.global.store.async.from.lds.b64
rocdl.global.store.async.from.lds.b128
rocdl.cluster.load.async.to.lds.b8
rocdl.cluster.load.async.to.lds.b32
rocdl.cluster.load.async.to.lds.b64
rocdl.cluster.load.async.to.lds.b128
rocdl.global.load.tr.b64
rocdl.global.load.tr.b128
rocdl.global.load.tr4.b64
rocdl.global.load.tr6.b96
rocdl.ds.load.tr4.b64
rocdl.ds.load.tr8.b64
rocdl.ds.load.tr6.b96
rocdl.ds.load.tr16.b128
rocdl.ds.read.tr4.b64
rocdl.ds.read.tr8.b64
rocdl.ds.read.tr6.b96
rocdl.ds.read.tr16.b64
rocdl.ds.atomic.barrier.arrive.rtn.b64
rocdl.ds.atomic.async.barrier.arrive.b64
rocdl.tensor.load.to.lds
rocdl.tensor.store.from.lds
rocdl.global.prefetch
rocdl.flat.prefetch
```

These are not general-purpose memory abstractions. They are AMDGPU instruction
interfaces. A beginner should read them as "the compiler has selected a very
specific AMD memory path."

### Raw Buffers, Resources, And Atomics

Raw-buffer operations build and use AMD buffer resources:

```text
rocdl.make.buffer.rsrc
rocdl.raw.buffer.load
rocdl.raw.buffer.store
rocdl.raw.buffer.atomic.cmpswap
rocdl.raw.buffer.atomic.fadd
rocdl.raw.buffer.atomic.fmax
rocdl.raw.buffer.atomic.smax
rocdl.raw.buffer.atomic.umin
rocdl.raw.ptr.buffer.load
rocdl.raw.ptr.buffer.load.lds
rocdl.raw.ptr.buffer.load.async.lds
rocdl.raw.ptr.buffer.store
rocdl.raw.ptr.buffer.atomic.cmpswap
rocdl.raw.ptr.buffer.atomic.fadd
rocdl.raw.ptr.buffer.atomic.fmax
rocdl.raw.ptr.buffer.atomic.smax
rocdl.raw.ptr.buffer.atomic.umin
```

`amdgpu` dialect buffer operations often lower into these operations. This is
where higher-level AMD buffer concepts become LLVM-intrinsic-shaped.

### Dot Product Operations

Dot-product operations target small vector dot instructions:

```text
rocdl.fdot2
rocdl.fdot2.f16.f16
rocdl.fdot2.bf16.bf16
rocdl.fdot2.f32.bf16
rocdl.sdot2, rocdl.sdot4, rocdl.sdot8
rocdl.udot2, rocdl.udot4, rocdl.udot8
rocdl.sudot4, rocdl.sudot8
rocdl.dot4.f32.fp8.fp8
rocdl.dot4.f32.fp8.bf8
rocdl.dot4.f32.bf8.fp8
rocdl.dot4.f32.bf8.bf8
```

### MFMA Operations

MFMA operations represent AMD matrix fused multiply-add instructions. They are
usually produced from `amdgpu` or vector/matrix lowering, not written by
frontends.

```text
rocdl.mfma.f32.16x16x16bf16.1k
rocdl.mfma.f32.16x16x16f16
rocdl.mfma.f32.16x16x1f32
rocdl.mfma.f32.16x16x2bf16
rocdl.mfma.f32.16x16x32.bf16
rocdl.mfma.f32.16x16x32.bf8.bf8
rocdl.mfma.f32.16x16x32.bf8.fp8
rocdl.mfma.f32.16x16x32.f16
rocdl.mfma.f32.16x16x32.fp8.bf8
rocdl.mfma.f32.16x16x32.fp8.fp8
rocdl.mfma.f32.16x16x4bf16.1k
rocdl.mfma.f32.16x16x4f16
rocdl.mfma.f32.16x16x4f32
rocdl.mfma.f32.16x16x8.xf32
rocdl.mfma.f32.16x16x8bf16
rocdl.mfma.f32.32x32x16.bf16
rocdl.mfma.f32.32x32x16.bf8.bf8
rocdl.mfma.f32.32x32x16.bf8.fp8
rocdl.mfma.f32.32x32x16.f16
rocdl.mfma.f32.32x32x16.fp8.bf8
rocdl.mfma.f32.32x32x16.fp8.fp8
rocdl.mfma.f32.32x32x1f32
rocdl.mfma.f32.32x32x2bf16
rocdl.mfma.f32.32x32x2f32
rocdl.mfma.f32.32x32x4.xf32
rocdl.mfma.f32.32x32x4bf16
rocdl.mfma.f32.32x32x4bf16.1k
rocdl.mfma.f32.32x32x4f16
rocdl.mfma.f32.32x32x8bf16.1k
rocdl.mfma.f32.32x32x8f16
rocdl.mfma.f32.4x4x1f32
rocdl.mfma.f32.4x4x2bf16
rocdl.mfma.f32.4x4x4bf16.1k
rocdl.mfma.f32.4x4x4f16
rocdl.mfma.f64.16x16x4f64
rocdl.mfma.f64.4x4x4f64
rocdl.mfma.i32.16x16x16i8
rocdl.mfma.i32.16x16x32.i8
rocdl.mfma.i32.16x16x4i8
rocdl.mfma.i32.16x16x64.i8
rocdl.mfma.i32.32x32x16.i8
rocdl.mfma.i32.32x32x32.i8
rocdl.mfma.i32.32x32x4i8
rocdl.mfma.i32.32x32x8i8
rocdl.mfma.i32.4x4x4i8
rocdl.mfma.scale.f32.16x16x128.f8f6f4
rocdl.mfma.scale.f32.32x32x64.f8f6f4
```

### Sparse MFMA, WMMA, And SWMMAC Operations

Sparse MFMA operations:

```text
rocdl.smfmac.f32.16x16x128.bf8.bf8
rocdl.smfmac.f32.16x16x128.bf8.fp8
rocdl.smfmac.f32.16x16x128.fp8.bf8
rocdl.smfmac.f32.16x16x128.fp8.fp8
rocdl.smfmac.f32.16x16x32.bf16
rocdl.smfmac.f32.16x16x32.f16
rocdl.smfmac.f32.16x16x64.bf16
rocdl.smfmac.f32.16x16x64.bf8.bf8
rocdl.smfmac.f32.16x16x64.bf8.fp8
rocdl.smfmac.f32.16x16x64.f16
rocdl.smfmac.f32.16x16x64.fp8.bf8
rocdl.smfmac.f32.16x16x64.fp8.fp8
rocdl.smfmac.f32.32x32x16.bf16
rocdl.smfmac.f32.32x32x16.f16
rocdl.smfmac.f32.32x32x32.bf16
rocdl.smfmac.f32.32x32x32.bf8.bf8
rocdl.smfmac.f32.32x32x32.bf8.fp8
rocdl.smfmac.f32.32x32x32.f16
rocdl.smfmac.f32.32x32x32.fp8.bf8
rocdl.smfmac.f32.32x32x32.fp8.fp8
rocdl.smfmac.f32.32x32x64.bf8.bf8
rocdl.smfmac.f32.32x32x64.bf8.fp8
rocdl.smfmac.f32.32x32x64.fp8.bf8
rocdl.smfmac.f32.32x32x64.fp8.fp8
rocdl.smfmac.i32.16x16x128.i8
rocdl.smfmac.i32.16x16x64.i8
rocdl.smfmac.i32.32x32x32.i8
rocdl.smfmac.i32.32x32x64.i8
```

WMMA operations:

```text
rocdl.wmma.bf16.16x16x16.bf16
rocdl.wmma.bf16.16x16x32.bf16
rocdl.wmma.bf16f32.16x16x32.bf16
rocdl.wmma.f16.16x16x128.bf8_bf8
rocdl.wmma.f16.16x16x128.bf8_fp8
rocdl.wmma.f16.16x16x128.fp8_bf8
rocdl.wmma.f16.16x16x128.fp8_fp8
rocdl.wmma.f16.16x16x16.f16
rocdl.wmma.f16.16x16x32.f16
rocdl.wmma.f16.16x16x64.bf8_bf8
rocdl.wmma.f16.16x16x64.bf8_fp8
rocdl.wmma.f16.16x16x64.fp8_bf8
rocdl.wmma.f16.16x16x64.fp8_fp8
rocdl.wmma.f32.16x16x128.bf8_bf8
rocdl.wmma.f32.16x16x128.bf8_fp8
rocdl.wmma.f32.16x16x128.fp8_bf8
rocdl.wmma.f32.16x16x128.fp8_fp8
rocdl.wmma.f32.16x16x16.bf16
rocdl.wmma.f32.16x16x16.bf8_bf8
rocdl.wmma.f32.16x16x16.bf8_fp8
rocdl.wmma.f32.16x16x16.f16
rocdl.wmma.f32.16x16x16.fp8_bf8
rocdl.wmma.f32.16x16x16.fp8_fp8
rocdl.wmma.f32.16x16x32.bf16
rocdl.wmma.f32.16x16x32.f16
rocdl.wmma.f32.16x16x4.f32
rocdl.wmma.f32.16x16x64.bf8_bf8
rocdl.wmma.f32.16x16x64.bf8_fp8
rocdl.wmma.f32.16x16x64.fp8_bf8
rocdl.wmma.f32.16x16x64.fp8_fp8
rocdl.wmma.i32.16x16x16.iu4
rocdl.wmma.i32.16x16x16.iu8
rocdl.wmma.i32.16x16x32.iu4
rocdl.wmma.i32.16x16x64.iu8
rocdl.wmma.scale.f32.16x16x128.f8f6f4
rocdl.wmma.scale.f32.32x16x128.f4
rocdl.wmma.scale16.f32.16x16x128.f8f6f4
rocdl.wmma.scale16.f32.32x16x128.f4
```

SWMMAC operations:

```text
rocdl.swmmac.bf16.16x16x32.bf16
rocdl.swmmac.bf16.16x16x64.bf16
rocdl.swmmac.bf16f32.16x16x64.bf16
rocdl.swmmac.f16.16x16x128.bf8.bf8
rocdl.swmmac.f16.16x16x128.bf8.fp8
rocdl.swmmac.f16.16x16x128.fp8.bf8
rocdl.swmmac.f16.16x16x128.fp8.fp8
rocdl.swmmac.f16.16x16x32.f16
rocdl.swmmac.f16.16x16x64.f16
rocdl.swmmac.f32.16x16x128.bf8.bf8
rocdl.swmmac.f32.16x16x128.bf8.fp8
rocdl.swmmac.f32.16x16x128.fp8.bf8
rocdl.swmmac.f32.16x16x128.fp8.fp8
rocdl.swmmac.f32.16x16x32.bf16
rocdl.swmmac.f32.16x16x32.bf8.bf8
rocdl.swmmac.f32.16x16x32.bf8.fp8
rocdl.swmmac.f32.16x16x32.f16
rocdl.swmmac.f32.16x16x32.fp8.bf8
rocdl.swmmac.f32.16x16x32.fp8.fp8
rocdl.swmmac.f32.16x16x64.bf16
rocdl.swmmac.f32.16x16x64.f16
rocdl.swmmac.i32.16x16x128.iu8
rocdl.swmmac.i32.16x16x32.iu4
rocdl.swmmac.i32.16x16x32.iu8
rocdl.swmmac.i32.16x16x64.iu4
```

### Low-Precision Conversion Operations

The `rocdl.cvt.*` family handles AMD low-precision packing, unpacking, scaling,
and stochastic rounding forms:

```text
rocdl.cvt.f32.bf8
rocdl.cvt.f32.fp8
rocdl.cvt.pk.bf8.f32
rocdl.cvt.pk.f32.bf8
rocdl.cvt.pk.f32.fp8
rocdl.cvt.pk.fp8.f32
rocdl.cvt.pkrtz
rocdl.cvt.sr.bf8.f32
rocdl.cvt.sr.fp8.f32
rocdl.cvt.scale.pk16.bf16.bf6
rocdl.cvt.scale.pk16.bf16.fp6
rocdl.cvt.scale.pk16.f16.bf6
rocdl.cvt.scale.pk16.f16.fp6
rocdl.cvt.scale.pk16.f32.bf6
rocdl.cvt.scale.pk16.f32.fp6
rocdl.cvt.scale.pk8.bf16.bf8
rocdl.cvt.scale.pk8.bf16.fp4
rocdl.cvt.scale.pk8.bf16.fp8
rocdl.cvt.scale.pk8.f16.bf8
rocdl.cvt.scale.pk8.f16.fp4
rocdl.cvt.scale.pk8.f16.fp8
rocdl.cvt.scale.pk8.f32.bf8
rocdl.cvt.scale.pk8.f32.fp4
rocdl.cvt.scale.pk8.f32.fp8
rocdl.cvt.scalef32.2xpk16.bf6.f32
rocdl.cvt.scalef32.2xpk16.fp6.f32
rocdl.cvt.scalef32.f16.bf8
rocdl.cvt.scalef32.f16.fp8
rocdl.cvt.scalef32.f32.bf8
rocdl.cvt.scalef32.f32.fp8
rocdl.cvt.scalef32.pk.bf16.bf8
rocdl.cvt.scalef32.pk.bf16.fp4
rocdl.cvt.scalef32.pk.bf16.fp8
rocdl.cvt.scalef32.pk.bf8.bf16
rocdl.cvt.scalef32.pk.bf8.f16
rocdl.cvt.scalef32.pk.bf8.f32
rocdl.cvt.scalef32.pk.f16.bf8
rocdl.cvt.scalef32.pk.f16.fp4
rocdl.cvt.scalef32.pk.f16.fp8
rocdl.cvt.scalef32.pk.f32.bf8
rocdl.cvt.scalef32.pk.f32.fp4
rocdl.cvt.scalef32.pk.f32.fp8
rocdl.cvt.scalef32.pk.fp4.bf16
rocdl.cvt.scalef32.pk.fp4.f16
rocdl.cvt.scalef32.pk.fp4.f32
rocdl.cvt.scalef32.pk.fp8.bf16
rocdl.cvt.scalef32.pk.fp8.f16
rocdl.cvt.scalef32.pk.fp8.f32
rocdl.cvt.scalef32.pk16.bf6.bf16
rocdl.cvt.scalef32.pk16.bf6.f16
rocdl.cvt.scalef32.pk16.bf6.f32
rocdl.cvt.scalef32.pk16.fp6.bf16
rocdl.cvt.scalef32.pk16.fp6.f16
rocdl.cvt.scalef32.pk16.fp6.f32
rocdl.cvt.scalef32.pk32.bf16.bf6
rocdl.cvt.scalef32.pk32.bf16.fp6
rocdl.cvt.scalef32.pk32.bf6.bf16
rocdl.cvt.scalef32.pk32.bf6.f16
rocdl.cvt.scalef32.pk32.f16.bf6
rocdl.cvt.scalef32.pk32.f16.fp6
rocdl.cvt.scalef32.pk32.f32.bf6
rocdl.cvt.scalef32.pk32.f32.fp6
rocdl.cvt.scalef32.pk32.fp6.bf16
rocdl.cvt.scalef32.pk32.fp6.f16
rocdl.cvt.scalef32.pk8.bf8.bf16
rocdl.cvt.scalef32.pk8.bf8.f16
rocdl.cvt.scalef32.pk8.bf8.f32
rocdl.cvt.scalef32.pk8.fp4.bf16
rocdl.cvt.scalef32.pk8.fp4.f16
rocdl.cvt.scalef32.pk8.fp4.f32
rocdl.cvt.scalef32.pk8.fp8.bf16
rocdl.cvt.scalef32.pk8.fp8.f16
rocdl.cvt.scalef32.pk8.fp8.f32
rocdl.cvt.scalef32.sr.bf8.bf16
rocdl.cvt.scalef32.sr.bf8.f16
rocdl.cvt.scalef32.sr.bf8.f32
rocdl.cvt.scalef32.sr.fp8.bf16
rocdl.cvt.scalef32.sr.fp8.f16
rocdl.cvt.scalef32.sr.fp8.f32
rocdl.cvt.scalef32.sr.pk.fp4.bf16
rocdl.cvt.scalef32.sr.pk.fp4.f16
rocdl.cvt.scalef32.sr.pk.fp4.f32
rocdl.cvt.scalef32.sr.pk16.bf6.bf16
rocdl.cvt.scalef32.sr.pk16.bf6.f16
rocdl.cvt.scalef32.sr.pk16.bf6.f32
rocdl.cvt.scalef32.sr.pk16.fp6.bf16
rocdl.cvt.scalef32.sr.pk16.fp6.f16
rocdl.cvt.scalef32.sr.pk16.fp6.f32
rocdl.cvt.scalef32.sr.pk32.bf6.bf16
rocdl.cvt.scalef32.sr.pk32.bf6.f16
rocdl.cvt.scalef32.sr.pk32.bf6.f32
rocdl.cvt.scalef32.sr.pk32.fp6.bf16
rocdl.cvt.scalef32.sr.pk32.fp6.f16
rocdl.cvt.scalef32.sr.pk32.fp6.f32
rocdl.cvt.scalef32.sr.pk8.bf8.bf16
rocdl.cvt.scalef32.sr.pk8.bf8.f16
rocdl.cvt.scalef32.sr.pk8.bf8.f32
rocdl.cvt.scalef32.sr.pk8.fp4.bf16
rocdl.cvt.scalef32.sr.pk8.fp4.f16
rocdl.cvt.scalef32.sr.pk8.fp4.f32
rocdl.cvt.scalef32.sr.pk8.fp8.bf16
rocdl.cvt.scalef32.sr.pk8.fp8.f16
rocdl.cvt.scalef32.sr.pk8.fp8.f32
```

### Math Operations

The scalar/vector math operations are:

```text
rocdl.sin
rocdl.cos
rocdl.tanh
rocdl.exp
rocdl.exp2
rocdl.log
rocdl.sqrt
rocdl.rsq
rocdl.rcp
rocdl.fmed3
```

These are AMDGPU math intrinsics or AMD-specific math lowering results. Use
portable `math` operations until the target-specific lowering point.

## Attributes And Enums

ROCDL defines attributes that model target options, cache policies, buffer
out-of-bounds modes, and matrix-instruction modifiers.

Important attributes:

| Attribute | Meaning |
| --- | --- |
| `ROCDLTargetAttr` / `#rocdl.target` | AMDGPU compilation target for `gpu.module`. |
| `BufferOOBModeAttr` | Buffer out-of-bounds mode attribute. |
| `BufferOOBModeModuleFlagAttr` | Module flag form for AMDGPU buffer OOB mode. |
| `TBufferOOBModeModuleFlagAttr` | Module flag form for tbuffer OOB mode. |
| `PreGfx12CachePolicyAttr` | Cache policy for pre-gfx12 operations. |
| `Gfx942CachePolicyAttr` | Cache policy for gfx942 operations. |
| `Gfx12CachePolicyAttr` | Cache policy for gfx12 operations. |
| `Gfx12AtomicCachePolicyAttr` | Cache policy for gfx12 atomic operations. |
| `MFMAPermBAttr` | MFMA B operand permutation modifier. |
| `MFMANegModifierAttr` | MFMA negation modifier. |
| `MatrixFormatAttr` | Matrix layout/format modifier. |
| `WMMAMatrixScaleAttr` | WMMA matrix scaling modifier. |
| `WMMAMatrixScaleFormatAttr` | WMMA scaling format modifier. |
| `WMMACModifierAttr` | WMMA C modifier attribute. |
| `SchedGroupMaskAttr` | Scheduling barrier mask. |

Important enum families:

```text
BufferOOBMode
PreGfx12CachePolicy
Gfx942CachePolicy
Gfx12CachePolicy
Gfx12AtomicCachePolicy
MFMAPermB
MFMANegModifier
MatrixFormat
WMMAMatrixScale
WMMAMatrixScaleFormat
WMMACModifier
SchedGroupMask
```

## Transformations

`rocdl` itself is not a high-level optimization dialect. The important
transformations are the passes that create `rocdl`, attach targets, or serialize
GPU modules.

| Pass or pipeline | Role |
| --- | --- |
| `convert-gpu-to-rocdl` | Converts `gpu` operations inside a `gpu.module` to ROCDL, LLVM, AMDGPU, CF, and memref support operations. |
| `convert-amdgpu-to-rocdl` | Converts supported `amdgpu` operations to ROCDL intrinsics. |
| `convert-math-to-rocdl` | Converts supported `math` operations to ROCDL operations or AMD device-library calls. |
| `convert-complex-to-rocdl-library-calls` | Converts supported `complex` operations to AMD device-library calls. |
| `rocdl-attach-target` | Attaches `#rocdl.target` to matching `gpu.module` operations. |
| `gpu-module-to-binary` | Uses `#rocdl.target` and target interfaces to serialize GPU modules to binary objects. |
| `gpu-lower-to-rocdl-pipeline` | Default pipeline that lowers common MLIR GPU code to ROCDL and then lowers host code. |

Common options:

| Pass | Notable options |
| --- | --- |
| `convert-gpu-to-rocdl` | `chipset`, `index-bitwidth`, `use-bare-ptr-memref-call-conv`, `runtime`, `allowed-dialects`. |
| `convert-amdgpu-to-rocdl` | `chipset`. |
| `convert-math-to-rocdl` | `chipset`. Empty chipset disables chipset-dependent patterns. |
| `rocdl-attach-target` | `module`, `triple`, `chip`, `features`, `abi`, `O`, `wave64`, `fast`, `daz`, `finite-only`, `unsafe-math`, `correct-sqrt`, `-l`. |

Example target attachment:

```shell
mlir-opt input.mlir --rocdl-attach-target='module=kernel.* chip=gfx90a'
```

Example full lowering pipeline:

```shell
mlir-opt input.mlir --gpu-lower-to-rocdl-pipeline='chip=gfx90a'
```

## Conversions And Lowering Paths

The typical lowering path is:

```text
linalg / vector / scf / memref / arith / math
  -> gpu and amdgpu
  -> rocdl plus llvm
  -> LLVM IR with AMDGPU intrinsics
  -> AMDGPU ISA/object code through ROCm tooling
```

Important incoming conversions:

- `gpu.thread_id` and `gpu.block_id` lower to `rocdl.workitem.id.*` and
  `rocdl.workgroup.id.*`.
- `gpu.subgroup_id`, lane-ID computations, shuffles, ballots, and subgroup
  broadcasts lower to wave-level `rocdl` operations.
- `gpu.barrier` lowers to the right AMD barrier sequence for the target
  chipset.
- `amdgpu.raw_buffer_*` operations lower to `rocdl.raw.*buffer.*` operations.
- `amdgpu.mfma`, `amdgpu.wmma`, and related matrix operations lower to ROCDL
  matrix intrinsics.
- Some `math` operations lower to AMDGPU math intrinsics or device-library
  calls.
- Some `complex` operations lower to AMD device-library calls.

Important outgoing conversions:

- There is no normal "convert ROCDL to LLVM dialect" pass in the way that
  higher-level dialects convert to `llvm`. `rocdl` is already in the LLVMIR
  dialect family.
- `mlir-translate` lowers ROCDL operations to LLVM IR by registering the ROCDL
  LLVM IR translation.
- The GPU binary pipeline uses the `#rocdl.target` target interface to compile
  translated LLVM IR to AMDGPU object code.

## Example IR

This example shows a GPU module already committed to an AMDGPU target:

```mlir
gpu.module @kernels [#rocdl.target<chip = "gfx90a", flags = {no_wave64}>] {
  llvm.func @kernel(%out: !llvm.ptr) attributes {
    rocdl.kernel,
    rocdl.reqd_work_group_size = array<i32: 256, 1, 1>
  } {
    %tid = rocdl.workitem.id.x range <i32, 0, 256> : i32
    %wave = rocdl.wavefrontsize : i32
    rocdl.s.barrier
    llvm.return
  }
}
```

This is not the form a beginner should start from. A higher-level source might
use `gpu.thread_id x` and `gpu.barrier`; `convert-gpu-to-rocdl` would produce
the target-specific `rocdl` operations.

Another small example computes a lane index in a wave64:

```mlir
%all = llvm.mlir.constant(-1 : i32) : i32
%zero = llvm.mlir.constant(0 : i32) : i32
%lo = rocdl.mbcnt.lo %all, %zero : (i32, i32) -> i32
%lane = rocdl.mbcnt.hi %all, %lo : (i32, i32) -> i32
```

## Mental Model

Think of `rocdl` as "AMDGPU LLVM intrinsic MLIR".

If `gpu` says "this code runs on a GPU", `rocdl` says "this exact AMDGPU
intrinsic or AMDGPU metadata is needed". Once IR reaches `rocdl`, portability is
mostly gone. That is not a problem; it is the point of this dialect. The
pipeline needs a precise target-specific language before translation to LLVM IR
and object code.

## Gotchas

- `rocdl` is low level. Prefer `gpu`, `vector`, `math`, and `amdgpu` until the
  final AMDGPU lowering stage.
- Many operations are only valid or profitable on certain chipsets. Always
  check the `chipset` or `#rocdl.target<chip = "...">` being used.
- `rocdl.barrier` is deprecated for direct use; prefer `gpu.barrier` and let
  lowering choose the right AMD sequence.
- Some barrier and matrix operations are gfx12 or gfx1250 specific.
- Cache policy attributes differ by architecture family. A policy that makes
  sense for gfx12 may not be valid for pre-gfx12.
- `rocdl` supports unknown operations, so seeing a `rocdl.*` operation that is
  not in the generated registered list does not necessarily mean the parser
  rejected it.
- The semantics of many operations are the semantics of the corresponding LLVM
  AMDGPU intrinsic. The MLIR op documentation is not always the full hardware
  manual.
- AMDGPU lowering often combines dialects. You may see `rocdl`, `llvm`,
  `amdgpu`, `cf`, and `memref` together during staged conversion.

## Source Map

Use these files in the LLVM repo when you need exact behavior:

| Topic | Files |
| --- | --- |
| Dialect definition | `mlir/include/mlir/Dialect/LLVMIR/ROCDLDialect.td` |
| Operation definitions | `mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td` |
| Attributes | `mlir/include/mlir/Dialect/LLVMIR/ROCDLAttrs.td` |
| Enums | `mlir/include/mlir/Dialect/LLVMIR/ROCDLEnums.td` |
| Dialect implementation | `mlir/lib/Dialect/LLVMIR/IR/ROCDLDialect.cpp` |
| LLVM IR translation | `mlir/lib/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.cpp` |
| ROCDL target interface | `mlir/include/mlir/Target/LLVM/ROCDL/Target.h`, `mlir/lib/Target/LLVM/ROCDL/Target.cpp` |
| ROCDL target utilities | `mlir/include/mlir/Target/LLVM/ROCDL/Utils.h`, `mlir/lib/Target/LLVM/ROCDL/Utils.cpp` |
| GPU to ROCDL conversion | `mlir/include/mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h`, `mlir/lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp` |
| AMDGPU to ROCDL conversion | `mlir/include/mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h`, `mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp` |
| Math and complex conversions | `mlir/lib/Conversion/MathToROCDL/MathToROCDL.cpp`, `mlir/lib/Conversion/ComplexToROCDLLibraryCalls/ComplexToROCDLLibraryCalls.cpp` |
| GPU target attachment | `mlir/lib/Dialect/GPU/Transforms/ROCDLAttachTarget.cpp` |
| Full pipeline | `mlir/lib/Dialect/GPU/Pipelines/GPUToROCDLPipeline.cpp` |
| Parser/printer tests | `mlir/test/Dialect/LLVMIR/rocdl.mlir` |
| LLVM IR translation tests | `mlir/test/Target/LLVMIR/rocdl.mlir` |
| Conversion tests | `mlir/test/Conversion/GPUToROCDL`, `mlir/test/Conversion/AMDGPUToROCDL`, `mlir/test/Conversion/MathToROCDL`, `mlir/test/Conversion/ComplexToROCDLLibraryCalls` |
