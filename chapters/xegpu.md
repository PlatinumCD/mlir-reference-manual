# `xegpu` Dialect

## Beginner Summary

The `xegpu` dialect models Intel Xe GPU operations inside MLIR. It is a
target-specific GPU dialect used for high-performance kernels, especially GEMM
and tile-based matrix code.

Most beginners should not start with `xegpu`. You usually begin with portable
IR such as `linalg`, `vector`, `memref`, `scf`, `arith`, and `gpu`. The
`xegpu` dialect appears when the compiler chooses Intel Xe GPU concepts such as
DPAS matrix instructions, tensor descriptors, block loads/stores, scattered
loads/stores, subgroup distribution, lane distribution, and Xe memory behavior.

The practical lowering shape is:

```text
linalg / vector / memref / gpu
  -> convert-vector-to-xegpu or explicit XeGPU lowering
xegpu
  -> convert-xegpu-to-xevm
xevm / spirv-like target forms
  -> convert-xevm-to-llvm or GPU binary pipeline
```

For a beginner, the main idea is that `xegpu` is the Intel GPU bridge between
portable vector-like GPU computation and lower-level XeVM/SPIR-V/LLVM target
code.

## Why This Dialect Exists

Intel Xe GPUs have hardware operations that are not naturally modeled by
portable MLIR operations alone. The dialect documentation calls out two
important examples: DPAS matrix instructions and 2-D block load/store.

The dialect exists so MLIR can:

- Represent Intel Xe GPU tensor descriptors.
- Express block memory operations that operate on tiles of memory.
- Express scattered load/store and prefetch operations.
- Express DPAS and scaled DPAS matrix computations.
- Carry layout information for workgroup, subgroup, instruction, and lane
  distribution.
- Lower vector operations to Intel GPU-specific operations before final target
  lowering.
- Support high-performance GEMM code generation with tile-level structure.

`xegpu` still uses MLIR `memref` and `vector` types around its operations. That
makes it more structured than final target IR, but less portable than generic
GPU/vector IR.

## When It Matters

`xegpu` matters when your compiler pipeline targets Intel Xe GPUs.

You care about it when:

- You are generating Intel GPU GEMM kernels.
- You need DPAS or scaled DPAS instructions.
- You need 2-D or N-D block load/store/prefetch.
- You need shared local memory descriptors.
- You need explicit workgroup, subgroup, instruction, or lane distribution.
- You are moving from `vector.contract` or `vector.transfer_*` into Intel GPU
  hardware operations.
- You are debugging `convert-vector-to-xegpu`, `xegpu-propagate-layout`, or
  `convert-xegpu-to-xevm`.
- You are using XeGPU Transform dialect operations to set layouts, insert
  prefetches, or find load operations.

It usually does not matter for target-independent frontend IR. At that level,
prefer `linalg`, `vector`, `tensor`, `memref`, `scf`, and `gpu`.

## When To Use It

Use `xegpu` when the program is intentionally Intel Xe GPU-specific and needs
operations that portable dialects do not model.

Good uses:

- Lowering vector contractions to DPAS operations.
- Describing N-D tensor descriptors for block memory access.
- Describing shared local memory through `!xegpu.mem_desc`.
- Applying layout propagation and workgroup/subgroup/lane distribution passes.
- Inserting block prefetches for Xe GPU memory access.
- Writing tests for XeGPU-to-XeVM lowering.

Avoid it when:

- The program should remain portable across GPU vendors.
- `vector.transfer_read`, `vector.transfer_write`, or `vector.contract` still
  express the computation adequately.
- You do not yet know the Intel GPU execution level you want to target:
  workgroup, subgroup, or lane.
- You are modeling host-side runtime APIs rather than device IR.

## Core Concepts

### XeGPU Is Tile-Oriented

The dialect is designed around tile-level code generation. A GEMM kernel is
decomposed into blocks that map onto Xe GPU hardware instructions and register
tiles.

The most important operation family is DPAS:

```text
xegpu.dpas
xegpu.dpas_mx
```

These are matrix multiply-accumulate operations for Intel GPU hardware, not
generic linear algebra operations.

### Tensor Descriptors Describe Memory Regions

`!xegpu.tensor_desc<...>` describes a region of memory and how hardware should
access it. It is metadata for block load/store/prefetch operations; it does not
own the data.

Typical flow:

```text
memref or pointer
  -> xegpu.create_nd_tdesc
!xegpu.tensor_desc<...>
  -> xegpu.load_nd / xegpu.store_nd / xegpu.prefetch_nd
```

### Mem Descriptors Describe Shared Local Memory

`!xegpu.mem_desc<...>` describes data in shared local memory. It is used by
matrix load/store forms and can carry a memory layout attribute:

```text
!xegpu.mem_desc<16x64xf16>
!xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<stride = [1, 16]>>
!xegpu.mem_desc<64xf16, #xegpu.mem_layout<block = [16]>>
```

### Layout Attributes Drive Distribution

`#xegpu.layout<...>` carries information about workgroup, subgroup,
instruction, and lane distribution. The layout passes propagate and refine this
information so operations can be mapped to the right execution granularity.

Common levels:

| Level | Meaning |
| --- | --- |
| Workgroup | A larger tile distributed across subgroups. |
| Subgroup | A tile distributed across lanes or instructions. |
| Instruction | How much data one instruction covers. |
| Lane | How each lane participates in the operation. |

### XeGPU Lowers To XeVM

The target-specific path is not directly "XeGPU to LLVM" in one step. The
important conversion is:

```text
xegpu
  -> xevm
  -> llvm or target binary pipeline
```

`xevm` is another target dialect in this checkout and represents the lower
level Intel GPU target boundary.

## Types

The current local LLVM checkout defines three XeGPU types.

| Type | Syntax | Meaning |
| --- | --- | --- |
| `TensorDescType` | `!xegpu.tensor_desc<8x16xf16>` | Descriptor for an interested data region, used by N-D block memory ops. |
| `MemDescType` | `!xegpu.mem_desc<16x64xf16>` | Descriptor for data in shared local memory. |
| `NbarrierType` | `!xegpu.nbarrier` | Named barrier value used by XeGPU barrier operations. |

### `TensorDescType`

A tensor descriptor carries shape, element type, optional block descriptor
encoding, and optional layout information.

Examples:

```text
!xegpu.tensor_desc<8x16xf32>
!xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
!xegpu.tensor_desc<32xf32, #xegpu.layout<lane_layout = [16], lane_data = [2]>>
```

### `MemDescType`

A mem descriptor describes data in shared local memory and can carry layout:

```text
!xegpu.mem_desc<16x64xf16>
!xegpu.mem_desc<16x64xf16, #xegpu.mem_layout<stride = [1, 16]>>
!xegpu.mem_desc<64xf16, #xegpu.mem_layout<block = [16]>>
```

### `NbarrierType`

`!xegpu.nbarrier` represents a named barrier. It is produced by
`xegpu.init_nbarrier` and consumed by `xegpu.nbarrier_arrive` and
`xegpu.nbarrier_wait`.

## Attributes

The current local LLVM checkout defines eight XeGPU attributes.

| Attribute | Meaning |
| --- | --- |
| `BlockTensorDescAttr` | Encoding for `TensorDescType`; records memory space, array length, and boundary-check behavior. |
| `MemorySpaceAttr` | Memory space enum, currently global device memory or SLM shared local memory. |
| `CachePolicyAttr` | Cache hint for load, store, and prefetch operations. |
| `FenceScopeAttr` | Fence scope, either workgroup or GPU. |
| `LayoutAttr` | Distribution layout across subgroup, instruction, and lane levels. |
| `SliceAttr` | Layout wrapper that applies a parent layout to selected dimensions. |
| `RangeAttr` | Half-open range attribute. |
| `MemLayoutAttr` | Memory layout metadata such as stride and block layout for `MemDescType`. |

Important enum values:

```text
memory_space: global, slm
cache_hint: cached, uncached, streaming, read_invalidate, write_back, write_through
fence_scope: workgroup, gpu
```

## Operations

The current local LLVM checkout defines 20 `xegpu` operations.

### Tensor Descriptor And N-D Block Memory

| Operation | What it does |
| --- | --- |
| `xegpu.create_nd_tdesc` | Creates an N-D tensor descriptor from a memref or pointer-like base. |
| `xegpu.prefetch_nd` | Prefetches an N-D block described by a tensor descriptor. |
| `xegpu.load_nd` | Loads an N-D block from memory through a tensor descriptor. |
| `xegpu.store_nd` | Stores an N-D block back through a tensor descriptor. |

These operations are the main block-memory path. They are used for dense tiles
and multidimensional blocks.

### Scatter/Gather Memory

| Operation | What it does |
| --- | --- |
| `xegpu.prefetch` | Prefetches scattered data points to cache. |
| `xegpu.load` | Loads scattered data points from memory. |
| `xegpu.store` | Stores scattered data points to memory. |
| `xegpu.atomic_rmw` | Performs atomic read-modify-write through a tensor descriptor. |

Scatter/gather forms work with offset vectors or scalar offsets and masks.

### Matrix And DPAS Operations

| Operation | What it does |
| --- | --- |
| `xegpu.dpas` | Matrix multiply-accumulate using Intel Xe DPAS behavior. |
| `xegpu.dpas_mx` | Scaled DPAS for MX-style low-precision formats with scale operands. |
| `xegpu.truncf` | Floating-point truncation to lower precision. |

These operations are central to ML and GEMM kernels on Intel GPUs.

### Named Barriers And Fences

| Operation | What it does |
| --- | --- |
| `xegpu.alloc_nbarrier` | Allocates a set of named barriers. |
| `xegpu.init_nbarrier` | Assigns a named barrier to the current thread. |
| `xegpu.nbarrier_arrive` | Signals arrival at a named barrier. |
| `xegpu.nbarrier_wait` | Waits for a named barrier. |
| `xegpu.fence` | Synchronizes memory accesses with a selected memory kind and scope. |

These operations coordinate memory and execution around Xe GPU hardware
features.

### Layout And Shared-Memory Matrix Operations

| Operation | What it does |
| --- | --- |
| `xegpu.convert_layout` | Converts the layout of a vector value. |
| `xegpu.create_mem_desc` | Creates a shared-local-memory descriptor. |
| `xegpu.load_matrix` | Loads matrix data through a `!xegpu.mem_desc`. |
| `xegpu.store_matrix` | Stores matrix data through a `!xegpu.mem_desc`. |

These operations are useful after data has been staged into shared local
memory.

## Transformations

XeGPU has several public transformation passes in this checkout.

| Pass | Purpose |
| --- | --- |
| `xegpu-propagate-layout` | Propagates and assigns layout information across XeGPU ops. |
| `xegpu-wg-to-sg-distribute` | Distributes workgroup-level XeGPU code to subgroup-level code. |
| `xegpu-blocking` | Partitions large-shape XeGPU ops into smaller hardware-friendly operations using layout `inst_data`. |
| `xegpu-vector-linearize` | Linearizes N-D vectors to 1-D vectors for XeVM lowering. |
| `xegpu-optimize-peephole` | Rewrites XeGPU block-load operations into more optimal forms, including transpose load improvements. |
| `xegpu-sg-to-lane-distribute` | Distributes subgroup-level XeGPU ops to lane-level code. |

The most important beginner point is that layout propagation and distribution
are part of the meaning of the lowering pipeline. The same high-level tile
shape can be transformed through workgroup, subgroup, instruction, and lane
levels before final target lowering.

### Layout Propagation Options

`xegpu-propagate-layout` has a `layout-kind` option:

| Value | Meaning |
| --- | --- |
| `inst` | Propagate instruction-level data layout. |
| `lane` | Propagate lane layout and lane data. |
| `subgroup` | Propagate subgroup layout and subgroup data. |

It also has `print-analysis-only` and `index-bitwidth` options.

## Transform Dialect Operations

The XeGPU Transform dialect extension defines five operations.

| Transform op | Purpose |
| --- | --- |
| `transform.xegpu.get_load_op` | Finds an `xegpu.load_nd` or `xegpu.load` in the producer chain of a value. |
| `transform.xegpu.set_anchor_layout` | Sets an `xegpu.layout` anchor layout on an operation. |
| `transform.xegpu.set_gpu_launch_threads` | Overrides x/y/z thread operands of a `gpu.launch`. |
| `transform.xegpu.insert_prefetch` | Inserts `xegpu.prefetch_nd` operations for an `xegpu.load_nd` inside a loop. |
| `transform.xegpu.convert_layout` | Inserts an `xegpu.convert_layout` operation before first use of a value. |

These are useful when a transform script controls XeGPU layout and prefetch
decisions explicitly.

## Conversions And Lowering Paths

### `convert-vector-to-xegpu`

This pass lowers selected `vector` operations into the XeGPU dialect. It is the
main bridge from portable vector IR to Intel GPU-specific IR.

Typical direction:

```text
vector.transfer_read
vector.transfer_write
vector.contract
  -> xegpu.create_nd_tdesc
  -> xegpu.load_nd / xegpu.store_nd
  -> xegpu.dpas
```

### `convert-xegpu-to-xevm`

This pass lowers XeGPU operations to the XeVM dialect. It depends on `xevm`,
`vector`, `memref`, `arith`, `llvm`, `index`, `gpu`, and `scf`.

It has a `use-64bit-index` option, defaulting to true, to control how index
types are converted.

### `convert-xevm-to-llvm`

After XeGPU has been lowered to XeVM, XeVM can be lowered to LLVM dialect using
`convert-xevm-to-llvm`.

### GPU Pipeline

The help output also exposes `gpu-lower-to-xevm-pipeline`. This is a broader
GPU pipeline that can attach XeVM targets and lower GPU modules. It has an
`xegpu-op-level` option that selects the intended XeGPU operation granularity:

```text
workgroup
subgroup
lane
```

## Example IR

### Tensor Descriptor And Block Load

```mlir
gpu.module @book {
  gpu.func @load_block(%src: memref<24x32xf16>) {
    %tdesc = xegpu.create_nd_tdesc %src
        : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
    %tile = xegpu.load_nd %tdesc[0, 0]
        <{l1_hint = #xegpu.cache_hint<cached>,
          l2_hint = #xegpu.cache_hint<uncached>}>
        : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    gpu.return
  }
}
```

The tensor descriptor describes the block. The `xegpu.load_nd` operation turns
that block into a vector value.

### Block Store

```mlir
gpu.module @book {
  gpu.func @store_block(%dst: memref<24x32xf16>, %value: vector<8x16xf16>) {
    %tdesc = xegpu.create_nd_tdesc %dst
        : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
    xegpu.store_nd %value, %tdesc[0, 0]
        <{l1_hint = #xegpu.cache_hint<write_back>,
          l2_hint = #xegpu.cache_hint<uncached>}>
        : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
    gpu.return
  }
}
```

### DPAS

```mlir
gpu.module @book {
  gpu.func @dpas(%a: vector<8x16xf16>, %b: vector<16x16xf16>) {
    %0 = xegpu.dpas %a, %b
        : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    gpu.return
  }
}
```

This is the XeGPU matrix multiply-accumulate operation. It is already a
hardware-specific choice.

### Scaled DPAS

```mlir
gpu.module @book {
  gpu.func @dpas_mx(
      %a: vector<8x32xf8E5M2>,
      %b: vector<32x16xf8E5M2>,
      %acc: vector<8x16xbf16>,
      %a_scale: vector<8x1xf8E8M0FNU>,
      %b_scale: vector<1x16xf8E8M0FNU>) {
    %0 = xegpu.dpas_mx %a, %b, %acc scale_a = %a_scale scale_b = %b_scale
        : (vector<8x32xf8E5M2>, vector<32x16xf8E5M2>,
           vector<8x16xbf16>, vector<8x1xf8E8M0FNU>,
           vector<1x16xf8E8M0FNU>) -> vector<8x16xbf16>
    gpu.return
  }
}
```

Scaled DPAS is used for MX-style low-precision computation.

### Named Barrier And Fence

```mlir
gpu.module @book {
  gpu.func @barrier_and_fence() {
    xegpu.alloc_nbarrier 8
    %id = arith.constant 1 : i8
    %threads = arith.constant 16 : i8
    %barrier = xegpu.init_nbarrier %id, %threads : i8, i8 -> !xegpu.nbarrier
    xegpu.nbarrier_arrive %barrier : !xegpu.nbarrier
    xegpu.nbarrier_wait %barrier : !xegpu.nbarrier
    xegpu.fence memory_kind = global, fence_scope = workgroup
    gpu.return
  }
}
```

### Shared Memory Descriptor And Matrix Load

```mlir
gpu.module @book {
  gpu.func @load_matrix(%mem: !xegpu.mem_desc<16x64xf16>) {
    %data = xegpu.load_matrix %mem[8, 8]
        : !xegpu.mem_desc<16x64xf16> -> vector<8x16xf16>
    gpu.return
  }
}
```

`!xegpu.mem_desc` describes shared local memory rather than global memory.

## How To Read XeGPU IR

When reading `xegpu`, ask:

1. Is this global-memory block access, shared-memory access, scatter/gather, or
   matrix compute?
2. What descriptor type is being used: `tensor_desc`, `mem_desc`, or
   `nbarrier`?
3. Which layout level is represented: workgroup, subgroup, instruction, or
   lane?
4. Which conversion stage produced this IR?
5. Is the next lowering stage XeVM?

Examples:

| If you see | Read it as |
| --- | --- |
| `xegpu.create_nd_tdesc` | A memory region is being packaged as a tensor descriptor for block access. |
| `xegpu.load_nd` | A block is loaded from descriptor-backed memory. |
| `xegpu.store_nd` | A vector block is stored through a tensor descriptor. |
| `xegpu.dpas` | Intel Xe matrix instruction semantics have been selected. |
| `xegpu.dpas_mx` | Scaled low-precision matrix instruction semantics have been selected. |
| `xegpu.convert_layout` | A value is being adapted between distribution layouts. |
| `xegpu.nbarrier_wait` | The kernel is synchronizing with a named barrier. |

## Gotchas

- `xegpu` is Intel GPU-specific. It is not portable GPU IR.
- Descriptor types are not data containers. They describe how memory is
  accessed.
- Layout attributes are part of the lowering strategy. Missing or conflicting
  layouts can block later passes.
- Workgroup, subgroup, and lane are different levels of distribution. Do not
  treat them as interchangeable.
- DPAS is not a generic matrix multiply. It is a hardware instruction family.
- Cache hints and memory spaces are target-specific.
- `convert-xegpu-to-xevm` is the main target conversion. If you expect direct
  LLVM lowering, you are skipping a major stage of the Intel GPU path.
- `xegpu.load` and `xegpu.load_nd` are different: one is scatter/gather-like,
  the other is block descriptor based.

## What It Implies In A Compiler Pipeline

Introducing `xegpu` means the compiler has committed to an Intel Xe GPU path.

That implies:

- The pipeline should know whether it is targeting workgroup, subgroup, or lane
  XeGPU granularity.
- Layout propagation and distribution passes become important, not optional
  cleanup.
- Further lowering should include `convert-xegpu-to-xevm`.
- Block memory operations should be matched with descriptor creation and layout
  information.
- DPAS operations should eventually lower to target instructions through XeVM.

For beginners, the mental model is: `xegpu` is where portable vector/GPU code
turns into Intel Xe GPU tile, memory, and matrix instructions while still
remaining in MLIR.

## Source Map

Primary source files in the local LLVM checkout:

| File | What to look for |
| --- | --- |
| `mlir/include/mlir/Dialect/XeGPU/IR/XeGPUDialect.td` | Dialect purpose and dependent dialects. |
| `mlir/include/mlir/Dialect/XeGPU/IR/XeGPUOps.td` | XeGPU operation definitions and examples. |
| `mlir/include/mlir/Dialect/XeGPU/IR/XeGPUTypes.td` | `TensorDescType`, `MemDescType`, and `NbarrierType`. |
| `mlir/include/mlir/Dialect/XeGPU/IR/XeGPUAttrs.td` | Layout, memory, cache, fence, range, slice, and descriptor attributes. |
| `mlir/include/mlir/Dialect/XeGPU/Transforms/Passes.td` | XeGPU transformation pass declarations. |
| `mlir/include/mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.td` | Transform dialect operations for XeGPU layout and prefetch control. |
| `mlir/lib/Dialect/XeGPU/IR/` | Operation, type, attribute, parser, printer, verifier, and layout behavior. |
| `mlir/lib/Dialect/XeGPU/Transforms/` | Layout propagation, blocking, distribution, linearization, and peephole passes. |
| `mlir/lib/Conversion/VectorToXeGPU/` | Lowering from vector operations to XeGPU. |
| `mlir/lib/Conversion/XeGPUToXeVM/` | Lowering from XeGPU to XeVM. |
| `mlir/test/Dialect/XeGPU/` | Parser, verifier, transform, and pass tests. |
| `mlir/test/Conversion/VectorToXeGPU/` | Vector-to-XeGPU conversion tests. |
| `mlir/test/Conversion/XeGPUToXeVM/` | XeGPU-to-XeVM conversion tests. |
| `mlir/test/Integration/Dialect/XeGPU/` | Integration tests for workgroup, subgroup, and lane XeGPU flows. |
