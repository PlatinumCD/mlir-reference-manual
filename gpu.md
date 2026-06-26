# GPU Dialect

## Beginner Summary

The `gpu` dialect is MLIR's target-independent model of GPU programming.

It describes the ideas that most GPU systems share: kernels, grids, blocks,
threads, subgroups, barriers, device memory operations, kernel launches, and
separate device modules. It does not mean "CUDA only" or "ROCm only". Instead,
it gives the compiler a common GPU layer that can later lower to target-specific
dialects such as `nvvm`, `rocdl`, `spirv`, or `xevm`.

Think of it as the point where a compiler pipeline starts saying: this part of
the program runs on an accelerator, this is the launch shape, these values are
thread ids, and these operations need GPU synchronization or GPU runtime calls.

## Why This Dialect Exists

GPU code is split across two worlds:

- Host code decides what kernel to run and with what arguments.
- Device code runs many work items in parallel.

Without a common dialect, every frontend and every target would need to invent
its own representation of launches, thread ids, barriers, GPU memory copies,
and device modules. The `gpu` dialect provides that shared representation.

It deliberately sits above vendor-specific details:

- `gpu.thread_id x` means the current thread's x id, not a particular CUDA,
  HIP, or SPIR-V builtin.
- `gpu.launch_func` means launch this kernel, not a direct call to a specific
  runtime API.
- `gpu.module` holds device code before it is serialized to a target object.
- `gpu.binary` holds serialized target objects after device compilation.

This lets transformations optimize GPU-shaped IR before committing to one
hardware backend.

## When It Matters

The `gpu` dialect matters in pipelines that lower parallel work to accelerators.

It commonly appears after higher-level parallelism has been exposed from
dialects such as `linalg`, `scf`, `affine`, or `vector`, and before the program
is lowered to target-specific GPU IR.

Typical pipeline shape:

```text
linalg/tensor/scf/vector
  -> loop tiling, mapping, bufferization, vectorization
  -> gpu.launch or gpu.func/gpu.module
  -> gpu-kernel-outlining
  -> convert-gpu-to-nvvm / convert-gpu-to-rocdl / convert-gpu-to-spirv
  -> gpu-module-to-binary
  -> gpu-to-llvm for host runtime calls
```

It is also important for compiler debugging. If a launch shape, address space,
barrier, or host/device boundary is wrong, the problem is often visible first in
the `gpu` dialect.

## When To Use It

Use `gpu` when your compiler needs a portable GPU execution model.

Use it for:

- Representing kernels and kernel launches.
- Mapping loops or `scf.forall` work to blocks, threads, warps, or lanes.
- Expressing GPU builtins such as block ids, thread ids, subgroup ids, and
  global ids.
- Modeling device memory allocation, copy, memset, wait, and host registration.
- Representing barriers, subgroup collectives, ballots, shuffles, and simple
  matrix-fragment operations.
- Separating host IR from device IR before target-specific lowering.

Do not use it when you need a vendor intrinsic directly. Use `nvvm`, `nvgpu`,
`rocdl`, `amdgpu`, `spirv`, `xegpu`, or `xevm` after the generic GPU structure
has been chosen.

## Core Concepts

### Host And Device Split

Host-side IR contains operations such as `gpu.launch_func`, `gpu.alloc`,
`gpu.memcpy`, `gpu.wait`, and `gpu.dealloc`. These operations describe runtime
actions: allocate device memory, copy memory, launch a kernel, and wait for
work to finish.

Device-side IR is usually inside `gpu.module` and `gpu.func`. A kernel function
is marked with the `kernel` keyword. Inside that function, operations such as
`gpu.thread_id`, `gpu.block_id`, `gpu.barrier`, and `gpu.subgroup_reduce`
describe what each work item does.

### Launches

There are two main launch forms:

- `gpu.launch` contains an inline kernel body as a region.
- `gpu.launch_func` launches a kernel function stored in a `gpu.module` or a
  serialized `gpu.binary`.

The `gpu-kernel-outlining` pass turns `gpu.launch` regions into `gpu.module`
and `gpu.func` definitions, then replaces the inline launch with
`gpu.launch_func`.

### Grid, Blocks, Threads, Clusters, And Subgroups

The dialect uses three dimensions: `x`, `y`, and `z`.

- The grid is made of blocks.
- A block is made of threads.
- A cluster is a group of thread blocks on targets that support clustered
  launches.
- A subgroup is a warp/wave-style group of lanes inside a workgroup.

The dialect has builtin operations to read ids and sizes at each level:
`gpu.block_id`, `gpu.thread_id`, `gpu.grid_dim`, `gpu.block_dim`,
`gpu.global_id`, `gpu.subgroup_id`, `gpu.lane_id`, and related ops.

### GPU Modules And Binaries

`gpu.module` is a device compilation unit. It can contain GPU kernel functions
and target attributes such as `#nvvm.target`, `#rocdl.target`, `#spirv.target`,
or `#xevm.target`.

`gpu.binary` is created later, after `gpu-module-to-binary` serializes a
`gpu.module` into target objects. Host launches may then refer to that binary.

### Async Tokens

Many host-side GPU ops support async dependencies. The type is
`!gpu.async.token`.

If an op is written with the `async` keyword, it returns a token instead of
blocking. Later GPU ops can depend on that token, and `gpu.wait` can block on
one or more tokens.

### Address Spaces

The dialect defines GPU memory address space attributes:

| Attribute | Meaning |
| --- | --- |
| `#gpu.address_space<global>` | Device/global memory visible across workgroups. |
| `#gpu.address_space<workgroup>` | Shared/workgroup memory visible within a workgroup. |
| `#gpu.address_space<private>` | Per-thread private memory. |
| `#gpu.address_space<constant>` | Constant memory. |

These are most often seen on `memref` types or barrier memory fences.

### Mapping Attributes

The dialect defines attributes used by loop and `scf.forall` mapping:

| Attribute family | Meaning |
| --- | --- |
| `#gpu.block<x>`, `#gpu.block<y>`, `#gpu.block<z>` | Map work to block ids. |
| `#gpu.thread<x>`, `#gpu.thread<y>`, `#gpu.thread<z>` | Map work to thread ids. |
| `#gpu.warpgroup<...>` | Map work to warpgroup-level parallelism. |
| `#gpu.warp<...>` | Map work to warp-level parallelism. |
| `#gpu.lane<linear_dim_0>` and related linear ids | Map work to lanes or linearized dimensions. |
| `#gpu.memory_space<workgroup>` and related spaces | Request placement in a GPU memory hierarchy. |
| `#gpu.loop_dim_map<...>` | Legacy-style mapping metadata for `scf.parallel` to GPU launch conversion. |

These attributes do not execute anything by themselves. They are instructions to
lowering and transform passes.

### Compilation Attributes

The dialect also defines attributes used after device compilation starts:

| Attribute | Meaning |
| --- | --- |
| `#gpu.kernel_metadata<...>` | Metadata for a compiled kernel. |
| `#gpu.kernel_table<...>` | A searchable list of kernel metadata. |
| `#gpu.object<...>` | A serialized GPU object plus target/properties metadata. |
| `#gpu.select_object<...>` | Offloading handler that chooses which object to embed. |

These are mainly relevant to `gpu.module-to-binary`, offloading, and runtime
translation.

### Important Types

| Type | Meaning |
| --- | --- |
| `!gpu.async.token` | Async dependency token for host/device GPU work. |
| `!gpu.mma_matrix<MxNxtype, "AOp">` | Opaque subgroup matrix fragment for an A operand. |
| `!gpu.mma_matrix<MxNxtype, "BOp">` | Opaque subgroup matrix fragment for a B operand. |
| `!gpu.mma_matrix<MxNxtype, "COp">` | Opaque subgroup matrix fragment for an accumulator/result. |
| `!gpu.named_barrier` | Handle for named barrier synchronization. |
| `!gpu.sparse.dntensor_handle` | Handle for a dense tensor descriptor used by sparse routines. |
| `!gpu.sparse.spmat_handle` | Handle for a sparse matrix descriptor. |
| `!gpu.sparse.spgemmop_handle` | Handle for a sparse matrix-matrix multiply descriptor. |

The MMA matrix type is intentionally opaque. Do not treat it as an ordinary
tensor or vector; it represents a target-lowered matrix fragment distributed
across a subgroup.

## Operations

The dialect has 68 operations.

### Essential Beginner Operations

Most beginners should understand these first:

| Operation | Purpose |
| --- | --- |
| `gpu.launch` | Inline GPU kernel body. |
| `gpu.launch_func` | Host-side launch of a kernel symbol or binary symbol. |
| `gpu.module` | Device compilation unit. |
| `gpu.func` | Device function; a kernel if marked `kernel`. |
| `gpu.return` | Terminator for `gpu.func`. |
| `gpu.terminator` | Terminator for `gpu.launch`. |
| `gpu.thread_id` | Current thread id in x/y/z. |
| `gpu.block_id` | Current block id in x/y/z. |
| `gpu.block_dim` | Block size in x/y/z. |
| `gpu.grid_dim` | Grid size in x/y/z. |
| `gpu.global_id` | Linearized global thread id in x/y/z. |
| `gpu.barrier` | Synchronize work items within a scope. |
| `gpu.alloc` | Allocate GPU-accessible memory. |
| `gpu.dealloc` | Free memory allocated by `gpu.alloc`. |
| `gpu.memcpy` | Copy between memrefs, usually host/device or device/device. |
| `gpu.memset` | Fill a GPU-accessible buffer. |
| `gpu.wait` | Wait on GPU async tokens or produce a joined token. |
| `gpu.binary` | Store serialized GPU object data. |

### Program Structure And Launch Operations

| Operation | Purpose |
| --- | --- |
| `gpu.module` | Device compilation unit and symbol table for GPU code. |
| `gpu.func` | Device function inside a `gpu.module`; `kernel` functions can be launched. |
| `gpu.return` | Return from a `gpu.func`. |
| `gpu.launch` | Inline GPU launch with a region body. |
| `gpu.launch_func` | Host-side launch of a kernel in a `gpu.module` or `gpu.binary`. |
| `gpu.terminator` | Ends a `gpu.launch` body. |
| `gpu.yield` | Returns values from nested GPU regions such as reductions or warp regions. |
| `gpu.binary` | Stores serialized objects produced from GPU modules. |
| `gpu.printf` | Device-side debug printing. |
| `gpu.wait` | Synchronizes async GPU work. |

`gpu.launch` is convenient while building and transforming kernels. It keeps the
kernel body next to the host code. `gpu.launch_func` is closer to what a real
runtime needs: the host launches a separately compiled kernel symbol.

### Execution Geometry Operations

| Operation | Purpose |
| --- | --- |
| `gpu.thread_id` | Thread id within a block. |
| `gpu.block_id` | Block id within the grid. |
| `gpu.block_dim` | Number of threads in a block dimension. |
| `gpu.grid_dim` | Number of blocks in a grid dimension. |
| `gpu.global_id` | Current global thread id. |
| `gpu.cluster_id` | Cluster id within the grid. |
| `gpu.cluster_dim` | Number of clusters in a grid dimension. |
| `gpu.cluster_block_id` | Block id within a cluster. |
| `gpu.cluster_dim_blocks` | Number of blocks in a cluster dimension. |
| `gpu.lane_id` | Lane id inside the current subgroup. |
| `gpu.subgroup_id` | Subgroup id inside the workgroup. |
| `gpu.num_subgroups` | Number of subgroups in the workgroup. |
| `gpu.subgroup_size` | Number of lanes in a subgroup. |

Several of these ops accept an optional `upper_bound`. Bounds are useful to
analyses, but they are also promises. If the runtime value exceeds the bound,
the lowered program has undefined behavior.

### Memory, Runtime, And Synchronization Operations

| Operation | Purpose |
| --- | --- |
| `gpu.alloc` | Allocates GPU memory; may be async and may be `host_shared`. |
| `gpu.dealloc` | Frees GPU memory; may be async. |
| `gpu.memcpy` | Copies between source and destination memrefs; may be async. |
| `gpu.memset` | Sets a destination memref to a scalar value; may be async. |
| `gpu.host_register` | Registers host memory for device access. |
| `gpu.host_unregister` | Unregisters previously registered host memory. |
| `gpu.set_default_device` | Sets the current default GPU device by index. |
| `gpu.dynamic_shared_memory` | Gets the dynamic workgroup/shared memory buffer for a kernel launch. |
| `gpu.barrier` | Synchronizes work items and optionally fences selected address spaces. |
| `gpu.initialize_named_barrier` | Creates a named barrier handle with a participant count. |

`gpu.barrier` defaults to workgroup scope. It can be restricted with
`memfence [#gpu.address_space<...>]`, and it can use `scope <subgroup>` or
`scope <cluster>` where supported. Named barriers require care: all relevant
participants must reach the matching dynamic barrier instance.

### Collective, Subgroup, And Lane Operations

| Operation | Purpose |
| --- | --- |
| `gpu.all_reduce` | Reduces one value across a workgroup. |
| `gpu.subgroup_reduce` | Reduces values across a subgroup. |
| `gpu.subgroup_broadcast` | Broadcasts a value from a selected subgroup lane. |
| `gpu.ballot` | Collects predicate bits from subgroup lanes. |
| `gpu.shuffle` | Moves values between subgroup lanes using xor/up/down/idx modes. |
| `gpu.rotate` | Rotates values across lanes. |
| `gpu.warp_execute_on_lane_0` | Runs a region only on lane 0 while bridging vector-style and SPMD-style code. |

These operations are target-independent descriptions of common GPU subgroup
communication. Later conversions map them to NVVM, ROCDL, or SPIR-V
instructions when supported.

### Subgroup MMA Operations

| Operation | Purpose |
| --- | --- |
| `gpu.subgroup_mma_load_matrix` | Collectively loads a matrix fragment into `!gpu.mma_matrix`. |
| `gpu.subgroup_mma_store_matrix` | Stores a matrix fragment from `!gpu.mma_matrix`. |
| `gpu.subgroup_mma_compute` | Performs subgroup matrix multiply-accumulate. |
| `gpu.subgroup_mma_constant_matrix` | Creates a matrix fragment filled with a scalar. |
| `gpu.subgroup_mma_elementwise` | Applies an elementwise operation to MMA matrix fragments. |
| `gpu.subgroup_mma_extract_thread_local` | Extracts the current thread's local element from a matrix fragment. |
| `gpu.subgroup_mma_insert_thread_local` | Inserts a thread-local scalar into a matrix fragment. |

These operations are for warp/subgroup-level matrix instructions. They are not
the same abstraction level as `linalg.matmul`. A high-level matmul may become
vector operations, then GPU/NVGPU/XeGPU matrix-fragment operations, and only
later target-specific instructions.

### Sparse Runtime Handle Operations

The sparse operations model calls into GPU sparse libraries through handles.
They are host-side descriptor and library-call operations, not a general sparse
IR for algebraic transformation. The TableGen comments note that this path
currently lowers to cuSparse for CUDA.

| Operation | Purpose |
| --- | --- |
| `gpu.create_dn_tensor` | Creates a dense tensor descriptor handle. |
| `gpu.destroy_dn_tensor` | Destroys a dense tensor descriptor. |
| `gpu.create_coo` | Creates a COO sparse matrix descriptor from separate row/column buffers. |
| `gpu.create_coo_aos` | Creates a COO sparse matrix descriptor from AoS index storage. |
| `gpu.create_csr` | Creates a CSR sparse matrix descriptor. |
| `gpu.create_csc` | Creates a CSC sparse matrix descriptor. |
| `gpu.create_bsr` | Creates a BSR sparse matrix descriptor. |
| `gpu.create_2to4_spmat` | Creates a 2:4 sparse matrix descriptor. |
| `gpu.destroy_sp_mat` | Destroys a sparse matrix descriptor. |
| `gpu.spmv_buffer_size` | Computes temporary buffer size for sparse matrix-vector multiply. |
| `gpu.spmv` | Runs sparse matrix-vector multiply. |
| `gpu.spmm_buffer_size` | Computes temporary buffer sizes for sparse matrix-dense matrix multiply. |
| `gpu.spmm` | Runs sparse matrix-dense matrix multiply. |
| `gpu.sddmm_buffer_size` | Computes temporary buffer size for sampled dense-dense matrix multiply. |
| `gpu.sddmm` | Runs sampled dense-dense matrix multiply. |
| `gpu.spgemm_create_descr` | Creates a descriptor for sparse matrix-matrix multiply. |
| `gpu.spgemm_destroy_descr` | Destroys a sparse matrix-matrix descriptor. |
| `gpu.spgemm_work_estimation_or_compute` | Runs SpGEMM work estimation or compute step. |
| `gpu.spgemm_copy` | Copies the SpGEMM result into a sparse matrix descriptor. |
| `gpu.spmat_get_size` | Reads rows, columns, and nonzero count from a sparse matrix descriptor. |
| `gpu.set_csr_pointers` | Updates CSR pointers/indices/values on an existing sparse matrix descriptor. |

## Transformations

### GPU Dialect Passes

The GPU dialect defines these dedicated passes:

| Pass | Purpose |
| --- | --- |
| `-gpu-launch-sink-index-computations` | Sinks index computations into `gpu.launch` bodies. |
| `-gpu-kernel-outlining` | Outlines `gpu.launch` bodies into `gpu.module`/`gpu.func` and emits `gpu.launch_func`. |
| `-gpu-async-region` | Makes GPU ops async inside `func.func` regions. |
| `-gpu-map-parallel-loops` | Greedily maps parallel loops to GPU workgroup/local dimensions. |
| `-gpu-eliminate-barriers` | Removes barriers that do not enforce needed memory-effect dependencies. |
| `-gpu-decompose-memrefs` | Makes memref sizes/strides explicit for targets such as SPIR-V. |
| `-gpu-module-to-binary` | Serializes nested `gpu.module` ops using attached target attributes. |
| `-nvvm-attach-target` | Attaches `#nvvm.target` to matching `gpu.module` ops. |
| `-rocdl-attach-target` | Attaches `#rocdl.target` to matching `gpu.module` ops. |
| `-spirv-attach-target` | Attaches SPIR-V target environment attributes to matching GPU modules. |
| `-xevm-attach-target` | Attaches `#xevm.target` to matching `gpu.module` ops. |

Important options:

- `gpu-map-parallel-loops` has `mapping-policy=outermost-first` or
  `mapping-policy=innermost-first`.
- `gpu-module-to-binary` supports output formats such as `offloading`,
  `assembly`, `binary`, and `fatbinary`.
- Target attach passes accept module matching and target configuration such as
  chip, triple, features, optimization level, and tool options.

### GPU Lowering Pipelines

The dialect registers these named pass pipelines:

| Pipeline | Purpose |
| --- | --- |
| `gpu-lower-to-nvvm-pipeline` | Lowers common dialects plus `gpu`/`nvgpu` toward NVVM, serializes device code, and lowers host code. |
| `gpu-lower-to-rocdl-pipeline` | Lowers common dialects plus `gpu` toward ROCDL/AMDGPU, serializes device code, and lowers host code. |
| `gpu-lower-to-xevm-pipeline` | Lowers GPU code toward XeVM, serializes device code, and lowers host code. |

These pipelines are useful examples of how much work is involved after a
program reaches the generic `gpu` dialect: device code and host code lower on
different tracks, then meet again at the launch/runtime boundary.

### Transform Dialect GPU Ops

The GPU dialect also contributes Transform dialect operations:

| Transform op | Purpose |
| --- | --- |
| `transform.gpu.map_forall_to_blocks` | Maps top-level `scf.forall` work to GPU blocks, optionally generating a `gpu.launch`. |
| `transform.gpu.map_nested_forall_to_threads` | Maps nested `scf.forall` work inside a `gpu.launch` to GPU threads. |
| `transform.apply_patterns.gpu.gpu_rewrite_patterns` | Applies GPU rewrite patterns such as all-reduce/global-id/shuffle rewrites. |
| `transform.apply_patterns.gpu.eliminate_barriers` | Applies barrier elimination patterns. |
| `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma` | Unrolls vector ops for a target subgroup MMA shape. |
| `transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu` | Promotes `gpu.shuffle` to AMDGPU-specific forms where possible. |
| `transform.apply_conversion_patterns.gpu.gpu_to_nvvm` | Adds GPU-to-NVVM conversion patterns. |
| `transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm` | Adds GPU WMMA-to-NVVM conversion patterns. |
| `transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm` | Adds GPU subgroup-reduce-to-NVVM conversion patterns. |
| `transform.apply_conversion_patterns.gpu.gpu_to_rocdl` | Adds GPU-to-ROCDL conversion patterns for a chipset. |

These are useful when a pipeline is driven by Transform dialect scripts instead
of a fixed command-line pass sequence.

## Conversions And Lowering Paths

### Into `gpu`

Common paths into `gpu` include:

| Source | Path |
| --- | --- |
| `affine.for` | `convert-affine-for-to-gpu` can convert top-level affine loops to GPU kernels. |
| Mapped `scf.parallel` | `convert-parallel-loops-to-gpu` converts mapped parallel loops to `gpu.launch`. |
| `scf.forall` | Transform ops map forall work to `gpu.block_id` and `gpu.thread_id`. |
| `vector` | `convert-vector-to-gpu` can move vector forms toward GPU/NVGPU matrix operations. |
| Hand-built GPU IR | Frontends may emit `gpu.launch`, `gpu.module`, or `gpu.launch_func` directly. |

The important beginner rule is: the compiler needs explicit parallel structure
before it can produce useful GPU IR. A sequential loop does not magically become
a GPU kernel unless a pass or transform maps its iteration space.

### Out Of `gpu`

Common lowering paths out of `gpu` include:

| Target path | Main passes |
| --- | --- |
| Host runtime calls | `gpu-to-llvm` lowers host GPU ops such as `gpu.launch_func`, `gpu.alloc`, `gpu.memcpy`, `gpu.wait`, and `gpu.dealloc` to LLVM dialect plus runtime-wrapper calls. |
| NVIDIA device code | `convert-gpu-to-nvvm`, often with `nvvm-attach-target` and `gpu-module-to-binary`. |
| AMD device code | `convert-gpu-to-rocdl`, often with `rocdl-attach-target` and `gpu-module-to-binary`. |
| SPIR-V device code | `convert-gpu-to-spirv`, sometimes with `gpu-decompose-memrefs` and `spirv-attach-target`. |
| LLVM SPIR-V backend path | `convert-gpu-to-llvm-spv` for LLVM operations intended for a SPIR-V backend. |
| Intel Xe path | `gpu-lower-to-xevm-pipeline` and related XeGPU/XeVM conversions. |

There are two different lowering problems:

- Device lowering turns `gpu.func`, `gpu.thread_id`, `gpu.barrier`, subgroup
  ops, and memory address spaces into target code.
- Host lowering turns allocation, copies, launches, waits, and binaries into
  runtime calls and object references.

Keeping those separate makes GPU pipelines easier to reason about.

## Example IR

### Inline Launch

This is the early, inline form. The kernel body lives inside the host function.

```mlir
module attributes {gpu.container_module} {
  func.func @scale(%buffer: memref<?xf32, 1>, %factor: f32) {
    %one = arith.constant 1 : index
    %block = arith.constant 128 : index

    gpu.launch blocks(%bx, %by, %bz) in
                 (%grid_x = %one, %grid_y = %one, %grid_z = %one)
               threads(%tx, %ty, %tz) in
                 (%block_x = %block, %block_y = %one, %block_z = %one) {
      %value = memref.load %buffer[%tx] : memref<?xf32, 1>
      %scaled = arith.mulf %value, %factor : f32
      memref.store %scaled, %buffer[%tx] : memref<?xf32, 1>
      gpu.terminator
    }

    return
  }
}
```

The launch names `%bx`, `%by`, `%bz`, `%tx`, `%ty`, and `%tz` are region
arguments with GPU meaning. They are not arbitrary loop variables.

### Outlined Kernel And Launch Function

After outlining, device code is in a `gpu.module` and host code uses
`gpu.launch_func`.

```mlir
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @scale_kernel(%buffer: memref<?xf32, 1>, %factor: f32) kernel {
      %tid = gpu.thread_id x
      %value = memref.load %buffer[%tid] : memref<?xf32, 1>
      %scaled = arith.mulf %value, %factor : f32
      memref.store %scaled, %buffer[%tid] : memref<?xf32, 1>
      gpu.return
    }
  }

  func.func @host(%buffer: memref<?xf32, 1>, %factor: f32) {
    %one = arith.constant 1 : index
    %block = arith.constant 128 : index
    gpu.launch_func @kernels::@scale_kernel
        blocks in (%one, %one, %one)
        threads in (%block, %one, %one)
        args(%buffer : memref<?xf32, 1>, %factor : f32)
    return
  }
}
```

This is closer to the final shape of GPU offloading: host code launches a
separately compiled kernel.

### GPU Memory Operations

```mlir
func.func @copy_to_device(%src: memref<?xf32>, %n: index) {
  %dst = gpu.alloc (%n) : memref<?xf32>
  gpu.memcpy %dst, %src : memref<?xf32>, memref<?xf32>
  gpu.dealloc %dst : memref<?xf32>
  return
}
```

The async form adds `async` and `!gpu.async.token` dependencies. Without
`async`, the operation is modeled as blocking.

### Subgroup Communication

```mlir
gpu.module @kernels {
  gpu.func @reduce_lane_value(%x: f32) kernel {
    %sum = gpu.subgroup_reduce add %x : (f32) -> f32
    %lane = gpu.lane_id
    %mask = arith.constant 1 : i32
    %width = arith.constant 32 : i32
    %other, %valid = gpu.shuffle xor %sum, %mask, %width : f32
    gpu.return
  }
}
```

This says "reduce and communicate inside a subgroup" without choosing the final
NVVM, ROCDL, or SPIR-V instruction.

## Mental Model

The `gpu` dialect is the compiler's portable GPU contract.

It answers four questions:

- What code runs on the device?
- How is the parallel work mapped to blocks, threads, clusters, and subgroups?
- What synchronization and memory movement must the runtime perform?
- Which target-specific lowering path will turn the device module into an
  object the host can launch?

Once you see those four pieces, most GPU IR becomes easier to read.

## Gotchas

`gpu.launch` and `gpu.launch_func` are different stages. `gpu.launch` is an
inline region form that is useful during transformation. `gpu.launch_func`
launches a symbol and is the form expected by host runtime lowering.

GPU launch sizes are always three-dimensional. For 1-D or 2-D kernels, unused
dimensions must still be provided, usually as `1`.

Bounds and known-size attributes are promises. `upper_bound`,
`known_block_size`, `known_grid_size`, and `known_cluster_size` enable stronger
reasoning, but violating them makes execution undefined after lowering.

Barriers are convergence-sensitive. Threads in the same synchronization scope
must reach the same dynamic barrier instance. Divergent control flow around
barriers is a common source of invalid GPU programs.

Address spaces matter. A `memref` in workgroup memory is not the same as a
global-memory `memref`, and some conversions expect address spaces to already
match target constraints.

`gpu.module` is not enough by itself. It needs target attributes before
`gpu-module-to-binary` can serialize it.

The sparse handle ops are runtime/library abstractions. They are not the same
as the `sparse_tensor` dialect, which models sparse tensor structure and
lowering more generally.

Subgroup and MMA ops are portable only at the MLIR abstraction level. Each
backend still decides which shapes, element types, scopes, and instructions are
actually legal.

## Source Map

Primary definitions:

- `mlir/include/mlir/Dialect/GPU/IR/GPUBase.td`
- `mlir/include/mlir/Dialect/GPU/IR/GPUOps.td`
- `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrs.td`
- `mlir/include/mlir/Dialect/GPU/IR/GPUDeviceMappingAttr.td`
- `mlir/include/mlir/Dialect/GPU/IR/ParallelLoopMapperAttr.td`

Implementation:

- `mlir/lib/Dialect/GPU/IR/GPUDialect.cpp`
- `mlir/lib/Dialect/GPU/Transforms/`
- `mlir/lib/Dialect/GPU/Pipelines/`
- `mlir/lib/Dialect/GPU/TransformOps/`

Passes and conversions:

- `mlir/include/mlir/Dialect/GPU/Transforms/Passes.td`
- `mlir/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/GPUCommon/`
- `mlir/lib/Conversion/GPUToNVVM/`
- `mlir/lib/Conversion/GPUToROCDL/`
- `mlir/lib/Conversion/GPUToSPIRV/`
- `mlir/lib/Conversion/GPUToLLVMSPV/`
- `mlir/lib/Conversion/SCFToGPU/`
- `mlir/lib/Conversion/VectorToGPU/`

Tests:

- `mlir/test/Dialect/GPU/`
- `mlir/test/Conversion/GPUCommon/`
- `mlir/test/Conversion/GPUToNVVM/`
- `mlir/test/Conversion/GPUToROCDL/`
- `mlir/test/Conversion/GPUToSPIRV/`
