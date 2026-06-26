# `amdgpu` Dialect

## Beginner Summary

The `amdgpu` dialect is MLIR's higher-level AMD GPU helper dialect. It is used
when a compiler already knows it is targeting AMD hardware, but still wants
operations that are more pleasant than raw LLVM AMDGPU intrinsics.

Think of it as the layer between portable GPU/vector/memref IR and final
`rocdl` intrinsic IR:

```text
linalg/vector/gpu/memref/arith
  -> AMD-specific choices appear
amdgpu
  -> final AMDGPU LLVM intrinsic form
rocdl/llvm
```

Most beginners should not start a program in `amdgpu`. You usually begin with
portable dialects such as `linalg`, `vector`, `memref`, `scf`, and `gpu`.
The `amdgpu` dialect becomes important when the program needs a concrete AMD
GPU feature: MFMA or WMMA matrix instructions, raw buffer memory operations,
LDS movement, wave-lane movement, AMD-specific low-precision float conversion,
or hardware synchronization behavior.

`amdgpu` is not the same as `rocdl`. `rocdl` is close to LLVM intrinsics.
`amdgpu` exists when a direct intrinsic wrapper would be too low level, too
chipset-specific, or awkward to express with MLIR types.

## Why This Dialect Exists

AMD GPUs have many target-specific operations. Some are naturally represented
by portable MLIR dialects. Others are naturally represented by final LLVM
intrinsics in `rocdl`. A third group needs a better MLIR-level abstraction:
the operation is AMD-specific, but the compiler still benefits from `memref`,
`vector`, MLIR small-float types, attributes, and verification.

The `amdgpu` dialect exists for that third group.

It is needed because:

- AMD matrix instructions have many variants and depend on the target `gfx`
  architecture.
- Some AMD intrinsics use packed integer operands even when the programmer is
  really thinking about vectors of small float values.
- Raw buffer operations need MLIR-friendly memref forms.
- Some hardware controls are clearer as enums or named attributes than as
  immediate "magic constants".
- Chipset-specific behavior can be abstracted behind one operation until the
  `convert-amdgpu-to-rocdl` lowering chooses the final intrinsic.
- A pass pipeline sometimes needs AMD-specific operations before final LLVM
  lowering, especially for fp8/fp6/fp4 conversion and AMD GPU matrix code.

The dialect documentation gives a useful rule of thumb: if lowering an
operation to `rocdl` would only replace it with a single intrinsic wrapper, it
probably belongs in `rocdl`, not `amdgpu`. An `amdgpu` operation should buy the
compiler something: better types, better verification, better syntax, hardware
generation abstraction, or integration with existing MLIR abstractions.

## When It Matters

The `amdgpu` dialect matters when a pipeline targets AMD GPUs and portable IR
is no longer enough.

You care about it when:

- You are lowering GPU kernels for AMD targets such as `gfx908`, `gfx90a`,
  `gfx942`, `gfx950`, `gfx1100`, `gfx1200`, or `gfx1250`.
- You need CDNA or RDNA matrix instructions such as MFMA, WMMA, sparse MFMA, or
  sparse WMMA.
- You need AMD raw buffer loads, stores, atomics, prefetches, or fat raw buffer
  memrefs.
- You are optimizing movement between global memory and LDS, also called
  workgroup memory.
- You need AMD wave-lane movement operations such as DPP, swizzle, or permlane.
- You are converting `arith` fp8/fp6/fp4 style operations to AMD-specific
  packed forms before lowering to `rocdl`.
- You are debugging why an AMD GPU lowering pipeline selected or rejected a
  specific intrinsic for a specific chipset.

It usually does not matter for CPU-only code, portable GPU frontend IR, or
target-independent optimization. In those layers, keep the program in dialects
such as `linalg`, `vector`, `gpu`, `memref`, `arith`, and `scf`.

## When To Use It

Use `amdgpu` when the IR is intentionally AMD-specific but should still look
like MLIR instead of raw LLVM intrinsics.

Good uses:

- Representing AMD matrix instructions before final lowering to `rocdl`.
- Using raw-buffer memory features while still working with memrefs.
- Describing low-precision float packing or extension with MLIR float and
  vector types.
- Inserting lane movement operations that are specific to AMD wave execution.
- Writing AMDGPU lowering tests.
- Building an AMD GPU backend pass that runs before LLVM dialect lowering.

Avoid it when:

- A portable `gpu`, `vector`, `memref`, `arith`, or `math` operation expresses
  the same thing.
- You need target-independent IR.
- You only need a direct LLVM AMDGPU intrinsic wrapper. That usually belongs in
  `rocdl`.
- You are describing host-side ROCm APIs. `amdgpu` is an MLIR device-code IR
  dialect, not a HIP runtime abstraction.

## Core Concepts

### AMDGPU Sits Above ROCDL

`rocdl` wraps AMDGPU LLVM intrinsics. `amdgpu` wraps AMD-specific concepts in a
way that works better with MLIR.

For example, a final AMDGPU intrinsic may want a packed 32-bit integer plus
several immediate fields. An `amdgpu` operation can instead accept
`vector<4xf8E4M3FN>`, a named lane selection, and attributes that are verified
before lowering.

This makes `amdgpu` useful in the middle of a lowering pipeline. It has already
given up portability, but it has not given up MLIR structure.

### GFX Names Are Architecture Names

This dialect uses LLVM's AMDGPU processor names, such as `gfx90a` or
`gfx1100`, rather than product names. Many operations are only legal on some
chipsets. The conversion pass often needs a `chipset` option so it can choose
or reject the final intrinsic.

The beginner rule is simple: if an `amdgpu` operation fails to lower, check the
chipset first. The IR may be well-formed MLIR but unsupported for the selected
AMD GPU generation.

### LDS Means Workgroup Memory

AMD documentation often uses the term LDS for the low-latency memory shared by
threads in a workgroup. In MLIR GPU address-space terms, this usually appears
as `#gpu.address_space<workgroup>`.

Several `amdgpu` operations move data between global memory and LDS or
synchronize work around LDS operations.

### Raw Buffers Are AMD-Specific Memory Views

AMD raw buffer instructions can do things that ordinary pointer operations do
not expose directly, such as hardware bounds behavior and buffer descriptor
controls. The `amdgpu` dialect provides both direct raw buffer operations and a
fat raw buffer address space:

```text
#amdgpu.address_space<fat_raw_buffer>
```

A memref in this address space behaves like a memref in MLIR, but lowers
toward AMD raw buffer machinery.

### Small Floats Are First-Class MLIR Types

The AMDGPU hardware and intrinsics support packed low-precision formats such as
fp8, bf8, fp6, bf6, and fp4. LLVM-level intrinsics may expose these as packed
integer forms. `amdgpu` operations can expose them as MLIR element types such
as:

```text
f8E4M3FN
f8E5M2
f8E4M3FNUZ
f8E5M2FNUZ
f6E2M3FN
f6E3M2FN
f4E2M1FN
```

That is one of the major reasons this dialect is useful for machine learning
and GPU matrix kernels.

### Matrix Ops Are Hardware Instructions

Operations such as `amdgpu.mfma`, `amdgpu.wmma`, `amdgpu.sparse_mfma`, and
`amdgpu.sparse_wmma` are not generic matrix multiply operations like
`linalg.matmul`. They represent AMD hardware instructions or instruction
families.

A beginner should read them as "the compiler has already selected a specific
AMD GPU matrix execution strategy." Earlier IR decides what computation is
being done; `amdgpu` records how the AMD GPU should perform an important part
of it.

## Types

The dialect defines four types.

| Type | Meaning |
| --- | --- |
| `!amdgpu.tdm_base<T>` | Opaque pair of LDS and global base addresses used by tensor data movement descriptors. |
| `!amdgpu.tdm_gather_base<T, I>` | Gather-mode variant of `!amdgpu.tdm_base` with an element type and index type. |
| `!amdgpu.tdm_descriptor` | Opaque descriptor group used by tensor load/store operations. |
| `!amdgpu.ds_barrier_state` | In-LDS barrier state used by gfx1250 and later barrier instructions. |

The `tdm` types are not general-purpose program data types. They are carrier
types for tensor data movement operations. The barrier state type is stored in
LDS and encodes pending arrivals, phase, and initialization count.

## Attributes

### Address Spaces

The dialect defines AMDGPU-specific address spaces:

| Attribute | Meaning |
| --- | --- |
| `#amdgpu.address_space<fat_raw_buffer>` | Address space 7. Represents a buffer resource plus offset as a memref-compatible fat pointer. |
| `#amdgpu.address_space<buffer_rsrc>` | Address space 8. Represents a buffer resource pointer. It is not suitable for memrefs because it does not support ordinary indexing. |
| `#amdgpu.address_space<fat_structured_buffer>` | Address space 9. Represents a structured buffer fat pointer, mainly relevant to graphics-style structured indexing. |

### DPP Permutations

`#amdgpu.dpp_perm<...>` and the matching DPP enum describe data-parallel
permutation modes used by `amdgpu.dpp`.

Available modes:

```text
quad_perm
row_shl
row_shr
row_ror
wave_shl
wave_shr
wave_ror
wave_rol
row_mirror
row_half_mirror
row_bcast_15
row_bcast_31
```

### Cache And Temporal Hints

`amdgpu.global_prefetch` uses temporal and cache-scope attributes.

Temporal hints:

```text
RT
NT
HT
LU
NT_RT
RT_NT
NT_HT
```

Cache scopes:

```text
WGP
SE
DEV
SYS
```

These names are target concepts. They are not portable GPU cache abstractions.

## Operations

The current local LLVM checkout defines 47 `amdgpu` operations. They are best
learned by purpose rather than alphabetically.

### Low-Precision Packing And Extension

These operations convert between packed low-precision AMD GPU forms and wider
floating-point values.

| Operation | What it does |
| --- | --- |
| `amdgpu.ext_packed_fp8` | Extends an fp8 value, or a selected pair from a packed fp8 vector, to `f32` or `vector<2xf32>`. |
| `amdgpu.scaled_ext_packed` | Extends packed fp8/fp6/fp4 style values with a scale operand. |
| `amdgpu.scaled_ext_packed_matrix` | Extends a wave-wide matrix of packed values using scale data. |
| `amdgpu.packed_trunc_2xfp8` | Rounds two `f32` values into a packed vector of 8-bit floats. |
| `amdgpu.packed_scaled_trunc` | Rounds scaled floating-point values into a packed low-precision vector. |
| `amdgpu.packed_stoch_round_fp8` | Stochastically rounds a float into a packed fp8 vector. |

These operations commonly appear after `convert-arith-to-amdgpu`. They are
useful because the source IR may express extension or truncation as normal
`arith` operations, while AMD hardware wants packed target-specific operations.

### Raw Buffer And Addressing Operations

These operations expose AMD raw-buffer behavior while preserving MLIR memref
structure where possible.

| Operation | What it does |
| --- | --- |
| `amdgpu.fat_raw_buffer_cast` | Casts a memref to a memref in `#amdgpu.address_space<fat_raw_buffer>`. |
| `amdgpu.raw_buffer_load` | Loads through AMD raw buffer machinery. |
| `amdgpu.raw_buffer_store` | Stores through AMD raw buffer machinery. |
| `amdgpu.raw_buffer_atomic_cmpswap` | Performs raw-buffer compare-and-swap. |
| `amdgpu.raw_buffer_atomic_fadd` | Performs raw-buffer floating-point atomic add. |
| `amdgpu.raw_buffer_atomic_fmax` | Performs raw-buffer floating-point atomic max. |
| `amdgpu.raw_buffer_atomic_smax` | Performs raw-buffer signed integer atomic max. |
| `amdgpu.raw_buffer_atomic_umin` | Performs raw-buffer unsigned integer atomic min. |
| `amdgpu.global_prefetch` | Prefetches global memory data to AMD GPU caches. |

Raw buffer indices are expressed in element units at the MLIR level. Lowering
converts them to the byte-oriented form expected by the underlying hardware
intrinsics. Bounds behavior is hardware-specific and may depend on whether the
target is CDNA or RDNA.

### Lane And Wave Data Movement

These operations move data between lanes in AMD wave execution.

| Operation | What it does |
| --- | --- |
| `amdgpu.dpp` | Data Parallel Primitive operation for lane movement and broadcast patterns. |
| `amdgpu.swizzle_bitmode` | Wrapper around AMD `ds_swizzle` bitmode behavior. |
| `amdgpu.permlane_swap` | Per-lane swap operation. |
| `amdgpu.permlane_var` | Variable-selector permlane operation for gfx12 and later. |

These are not ordinary vector shuffles. They model hardware wave-lane
communication. They matter in hand-tuned GPU kernels and in lowerings that
select wave-level data movement.

### Barriers, Counters, And Scheduling

These operations coordinate memory effects, backend scheduling, or in-LDS
barrier state.

| Operation | What it does |
| --- | --- |
| `amdgpu.lds_barrier` | Barrier that includes a wait for LDS memory operations. |
| `amdgpu.sched_barrier` | Limits how far the backend scheduler may move instructions. |
| `amdgpu.memory_counter_wait` | Waits for specified AMD GPU hardware counters. |
| `amdgpu.ds_barrier_init` | Initializes an in-LDS barrier state. |
| `amdgpu.ds_barrier_poll_state` | Atomically reads an in-LDS barrier state. |
| `amdgpu.ds_async_barrier_arrive` | Asynchronously arrives at an in-LDS barrier. |
| `amdgpu.ds_barrier_arrive` | Arrives at an in-LDS barrier and returns the old state. |
| `amdgpu.ds_barrier_state_phase` | Extracts the phase field from a barrier state. |
| `amdgpu.ds_barrier_state_pending_count` | Extracts the pending-count field from a barrier state. |
| `amdgpu.ds_barrier_state_init_count` | Extracts the init-count field from a barrier state. |
| `amdgpu.ds_barrier_state_phase_parity` | Extracts phase parity from a barrier state. |

The `ds_barrier_*` operations are tied to hardware support introduced for
newer AMD GPUs. They are much more specific than `gpu.barrier`.

### Matrix, Dot, And Sparse Matrix Operations

These operations represent AMD hardware matrix and dot-product instruction
families.

| Operation | What it does |
| --- | --- |
| `amdgpu.mfma` | CDNA MFMA matrix instruction wrapper. |
| `amdgpu.scaled_mfma` | CDNA scaled MFMA wrapper. |
| `amdgpu.sparse_mfma` | CDNA sparse MFMA, also called SMFMAC, wrapper. |
| `amdgpu.wmma` | WMMA instruction wrapper. |
| `amdgpu.scaled_wmma` | Scaled WMMA instruction wrapper. |
| `amdgpu.sparse_wmma` | gfx12 and later sparse WMMA wrapper. |
| `amdgpu.dot` | AMDGPU `v_dot*` intrinsic wrapper. |

Use these only after the compiler has committed to AMD hardware. They encode
tile sizes, source and accumulator types, sparsity behavior, or instruction
shape details that are not portable.

### Global, LDS, And Tensor Data Movement

These operations move data between global memory and LDS, or build descriptors
for tensor data movement.

| Operation | What it does |
| --- | --- |
| `amdgpu.gather_to_lds` | CDNA gather-to-LDS operation. |
| `amdgpu.global_load_async_to_lds` | Asynchronously loads global memory into LDS while bypassing VGPRs. |
| `amdgpu.transpose_load` | CDNA transpose load operation. |
| `amdgpu.global_transpose_load` | Global memory transpose load operation. |
| `amdgpu.make_dma_base` | Builds the base-address pair for tensor data movement. |
| `amdgpu.make_gather_dma_base` | Builds the gather-mode base-address pair for tensor data movement. |
| `amdgpu.make_dma_descriptor` | Builds descriptor groups for tensor load/store operations. |
| `amdgpu.make_gather_dma_descriptor` | Builds gather-mode descriptor groups for tensor load/store operations. |
| `amdgpu.tensor_load_to_lds` | Loads tensors from global memory to LDS. |
| `amdgpu.tensor_store_from_lds` | Stores tensors from LDS to global memory. |

These are advanced operations. A beginner should connect them to the common GPU
optimization pattern: stage data from global memory into shared/workgroup memory
so a matrix or tiled computation can reuse it efficiently.

## Transformations

The AMDGPU dialect owns three public transformation passes in this checkout.

### `amdgpu-emulate-atomics`

This pass rewrites unsupported AMDGPU atomic operations into compare-and-swap
loops for a selected chipset.

Important point: "unsupported" depends on hardware. The pass has a `chipset`
option, defaulting to `gfx000`, and uses that chipset to decide which atomics
need emulation.

Use it when a program contains AMDGPU atomic operations that are not directly
available on the target GPU.

### `amdgpu-maskedload-to-load`

This pass lowers `vector.maskedload`-style behavior toward a form based on
`vector.load`, `arith.select`, and related operations. It is useful because the
result can lower more effectively toward AMD buffer loads with bounds checking.

It is especially relevant when working with AMD raw buffer address spaces.

### `amdgpu-resolve-strided-metadata`

This pass rewrites `memref.extract_strided_metadata` patterns that target
AMDGPU casts, especially around `amdgpu.fat_raw_buffer_cast`.

It is meant to be used near strided metadata expansion. The pass description in
the source notes that it may need to run alongside `expand-strided-metadata`,
and simple pipelines may need another strided metadata expansion after it.

## Conversions And Lowering Paths

### `convert-arith-to-amdgpu`

This conversion pass rewrites selected `arith` operations to AMDGPU-specific
implementations.

In this checkout it focuses on extension and truncation involving 8-bit float
types, producing `amdgpu` operations rather than lowering directly from
`arith` to `rocdl`. This keeps the lowering split into understandable stages:

```text
arith
  -> amdgpu
  -> rocdl/llvm
```

Important options:

| Option | Meaning |
| --- | --- |
| `chipset` | AMD GPU processor name used for target-specific choices. |
| `saturate-fp8-truncf` | Uses saturating truncation for 8-bit float types. |
| `allow-packed-f16-round-to-zero` | Allows packed `f32` to `f16` round-to-zero conversion. |

### `convert-amdgpu-to-rocdl`

This pass converts supported `amdgpu` operations to `rocdl` intrinsics.

It also has a `chipset` option. That option is central: many AMDGPU operations
are only available on particular AMD GPU generations, or lower to different
intrinsics depending on the target.

The normal backend direction is:

```text
amdgpu.mfma
amdgpu.raw_buffer_load
amdgpu.dpp
amdgpu.ext_packed_fp8
  -> convert-amdgpu-to-rocdl
rocdl.* / llvm.*
```

### Related GPU Lowering

In a full AMD GPU pipeline, `amdgpu` usually appears next to other conversions:

| Conversion or pipeline | Role |
| --- | --- |
| `convert-gpu-to-rocdl` | Lowers generic `gpu` operations to ROCDL and LLVM forms. |
| `convert-math-to-rocdl` | Lowers math operations to AMDGPU-compatible ROCDL forms where applicable. |
| `gpu-lower-to-rocdl-pipeline` | A broader pipeline for lowering GPU modules toward ROCDL. |
| `rocdl-attach-target` | Attaches an AMDGPU target attribute to a `gpu.module`. |

The exact pipeline is project-specific, but the conceptual order is stable:
portable computation first, AMD-specific choices in the middle, ROCDL/LLVM near
the end.

## Example IR

### Packed fp8 Extension

This example reads one element position from a packed fp8 vector and extends it
to `f32`.

```mlir
func.func @ext_packed_fp8_s(%v: vector<4xf8E4M3FNUZ>) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E4M3FNUZ> to f32
  func.return %ret : f32
}
```

The important idea is that the IR says "packed fp8 vector" directly. It does
not force the frontend to describe the value as a generic packed integer.

### Raw Buffer Load

This example uses a raw buffer load from a memref. The index is an `i32`,
matching the raw buffer operation's indexing form.

```mlir
func.func @raw_buffer_load(%buf: memref<64xi32>, %idx: i32) -> i32 {
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx]
      : memref<64xi32>, i32 -> i32
  func.return %0 : i32
}
```

At this level the operation still carries a memref type. Later lowering builds
the AMD buffer resource details.

### Fat Raw Buffer Cast

This example casts a global memref into the AMDGPU fat raw buffer address
space.

```mlir
func.func @fat_raw_buffer_cast(%buf: memref<8xi32, #gpu.address_space<global>>)
    -> memref<8xi32, #amdgpu.address_space<fat_raw_buffer>> {
  %ret = amdgpu.fat_raw_buffer_cast %buf
      : memref<8xi32, #gpu.address_space<global>>
      to memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
  func.return %ret : memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
}
```

This is a good example of why `amdgpu` exists: the operation is AMD-specific,
but the result is still a memref-like value that other MLIR code can reason
about.

### Dot Product

This example represents an AMD dot-product intrinsic family.

```mlir
func.func @dot_i8(%a: vector<4xi8>, %b: vector<4xi8>, %c: i32) -> i32 {
  %r = amdgpu.dot %a * %b + %c : vector<4xi8>, vector<4xi8>, i32
  func.return %r : i32
}
```

This is not a generic linear algebra operation. It is already a hardware-aware
selection.

### Asynchronous Global-To-LDS Load

This example loads one `f32` from global memory to LDS with an optional mask.

```mlir
func.func @global_load_async_to_lds(
    %src: memref<16xf32, #gpu.address_space<global>>,
    %dst: memref<16xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  amdgpu.global_load_async_to_lds %src[%c0], %dst[%c0], %true
      : f32, memref<16xf32, #gpu.address_space<global>>,
        memref<16xf32, #gpu.address_space<workgroup>>
  func.return
}
```

This shows the common AMD GPU optimization pattern: move data from global
memory into LDS so the workgroup can reuse it.

### DPP Lane Movement

This example shifts a value across lanes using a DPP row shift.

```mlir
func.func @dpp_shift(%a: i32, %old: i32) -> i32 {
  %0 = amdgpu.dpp %a %old row_shl ( 0x1 : i32 )
      { row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true } : i32
  func.return %0 : i32
}
```

This is hardware lane movement. It should not be confused with a normal vector
shuffle in portable IR.

## How To Read AMDGPU IR

When you see an `amdgpu` operation, ask four questions:

1. What portable operation did this come from?
2. Which AMD hardware feature has now been selected?
3. Which `gfx` target is this intended for?
4. What final `rocdl` intrinsic family will it lower to?

For example:

| If you see | Read it as |
| --- | --- |
| `amdgpu.mfma` | The compiler selected an AMD CDNA matrix instruction shape. |
| `amdgpu.raw_buffer_load` | The compiler is using AMD raw-buffer memory access, not ordinary pointer load. |
| `amdgpu.fat_raw_buffer_cast` | A memref is being reinterpreted as an AMD raw-buffer-compatible memref. |
| `amdgpu.dpp` | Data is moving between lanes in an AMD wave. |
| `amdgpu.global_load_async_to_lds` | Data is being staged from global memory into LDS using AMD async hardware support. |
| `amdgpu.ext_packed_fp8` | Packed fp8 data is being unpacked through AMD-specific lowering. |

This reading style keeps the dialect from feeling like a random bag of
intrinsics. Each operation is a sign that the pipeline has crossed into an AMD
GPU-specific decision.

## Gotchas

- `amdgpu` is target-specific. Once you introduce it, the IR is no longer
  portable to NVIDIA, CPU, or SPIR-V backends without a separate rewrite.
- The selected `chipset` matters. Some valid-looking operations cannot lower on
  older or different AMD GPU generations.
- `amdgpu` and `rocdl` are related but not interchangeable. Prefer `amdgpu`
  when MLIR types, memrefs, enums, or chipset abstraction help. Prefer `rocdl`
  for direct intrinsic wrappers.
- LDS means workgroup memory. Many operations require
  `#gpu.address_space<workgroup>` destinations or barrier state.
- Raw buffer behavior is not the same as ordinary `memref.load` and
  `memref.store`. Bounds behavior and partial vector behavior can be
  hardware-dependent.
- Matrix ops such as `amdgpu.mfma` are not high-level matrix multiplication.
  They are hardware instruction selections.
- Atomic support varies by chipset. Run or inspect `amdgpu-emulate-atomics`
  when targeting a GPU that lacks the exact atomic operation.
- The dialect is not a frontend modeling language. It is most useful in backend
  and late-middle-end GPU lowering.

## What It Implies In A Compiler Pipeline

Introducing `amdgpu` implies that the compiler has made a target decision:
this code is for AMD GPU hardware.

That has practical consequences:

- The pipeline should carry a real AMD GPU chipset choice.
- Verification and conversion may reject operations unsupported by that
  chipset.
- Later conversion should include `convert-amdgpu-to-rocdl`.
- The surrounding GPU lowering will likely include `convert-gpu-to-rocdl`,
  `convert-math-to-rocdl`, LLVM conversion, and GPU module serialization.
- Optimization is now allowed to use AMD-specific memory, wave, and matrix
  behavior.

For a beginner, the important conceptual shift is this: before `amdgpu`, the IR
can still describe a mostly portable GPU program. After `amdgpu`, the IR is
describing how AMD hardware should execute important pieces of that program.

## Source Map

Primary source files in the local LLVM checkout:

| File | What to look for |
| --- | --- |
| `mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPUBase.td` | Dialect purpose, design rules, dependent dialects. |
| `mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPUOps.td` | Operation definitions, syntax, summaries, and examples. |
| `mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPUTypes.td` | AMDGPU-specific type definitions. |
| `mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPUAttrs.td` | AMDGPU-specific attributes. |
| `mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPUEnums.td` | Enums used by operations and attributes. |
| `mlir/include/mlir/Dialect/AMDGPU/Transforms/Passes.td` | AMDGPU dialect pass declarations. |
| `mlir/lib/Dialect/AMDGPU/IR/AMDGPU.cpp` | Dialect implementation, verification, parsing, printing, canonicalization. |
| `mlir/lib/Dialect/AMDGPU/Transforms/` | AMDGPU transformation pass implementations. |
| `mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp` | Lowering from `amdgpu` to `rocdl`. |
| `mlir/lib/Conversion/ArithToAMDGPU/ArithToAMDGPU.cpp` | Lowering from selected `arith` operations to `amdgpu`. |
| `mlir/test/Dialect/AMDGPU/` | Parser, verifier, canonicalization, and transform tests. |
| `mlir/test/Conversion/AMDGPUToROCDL/` | Examples of final AMDGPU lowering behavior. |
