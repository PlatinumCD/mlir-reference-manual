# `shard` Dialect

## Beginner Summary

The `shard` dialect describes how tensor values are split, replicated, moved,
and synchronized across a logical grid of devices or processes.

It is easiest to think of `shard` as the MLIR dialect for distributed tensor
placement. A compiler can use it to say:

- this tensor is split across these device-grid axes;
- this operation should run in SPMD form;
- this result needs an all-gather, all-reduce, all-to-all, halo exchange, or
  other communication step;
- this distributed program can later be lowered to a message-passing runtime
  such as MPI.

The dialect is inspired by GSPMD-style compilation. The important beginner
idea is that `shard` does not start from explicit send and receive buffers. It
starts from tensor sharding and lets compiler passes derive the communication
needed to make the program correct on many devices.

## Why This Dialect Exists

Large tensor programs often cannot run efficiently on one device. They need
model parallelism, data parallelism, tensor parallelism, or domain
decomposition. Those strategies all require the compiler to know where pieces
of a tensor live and how operations communicate between devices.

Without a dialect like `shard`, this information tends to be hidden in runtime
calls, framework-specific metadata, or opaque attributes. That makes it hard for
MLIR passes to propagate layouts, change layouts, optimize collectives, or
lower distributed work in stages.

The `shard` dialect gives the compiler first-class IR for:

- device/process grids;
- tensor sharding annotations;
- current process indices;
- distributed tensor shapes;
- collective communication;
- halo exchange;
- SPMD partitioning;
- lowering to MPI.

## When It Matters

The `shard` dialect matters when a program is being compiled for multiple
devices or processes and the tensor distribution is part of the compiler's job.

It is especially important for:

- distributed ML workloads where tensors are partitioned across accelerators;
- SPMD lowering, where every process runs the same program on a local shard;
- resharding between different tensor layouts;
- reductions that cross device boundaries;
- halo exchange in stencil-like or domain-decomposed computations;
- lowering high-level tensor communication into MPI operations.

It is less relevant for a single-device compiler pipeline. A single GPU kernel,
for example, usually uses `gpu`, `nvgpu`, `nvvm`, `amdgpu`, `rocdl`, `xegpu`,
or `xevm` rather than `shard`.

## When To Use It

Use `shard` when the IR still talks about tensors, but those tensors already
have distributed placement semantics.

Typical uses are:

- represent a logical device grid with `shard.grid`;
- annotate tensor values with `shard.sharding` and `shard.shard`;
- propagate those annotations through supported operations with
  `sharding-propagation`;
- partition a tensor function into SPMD form with `shard-partition`;
- simplify communication with `shard-simplify`;
- lower implemented communication patterns with `convert-shard-to-mpi`.

Do not use `shard` as a generic replacement for `memref` or `mpi`. `shard`
explains distributed tensor intent. `mpi` models the lower-level communication
runtime. `memref` models concrete buffers.

## Core Concepts

### Device Grids

A `shard.grid` operation defines a logical grid of devices or processes:

```mlir
shard.grid @grid(shape = 2x4)
```

This means there are two grid axes. Axis 0 has size 2, and axis 1 has size 4.
Dynamic grid dimensions are allowed with `?`:

```mlir
shard.grid @dynamic_grid(shape = 2x?)
```

Most other `shard` operations refer to a grid by symbol, for example `@grid`.

### Grid Axes And Device Groups

Communication operations use `grid_axes` to say which axes participate in a
collective. Devices with the same coordinates outside those axes form one
communication group.

For example, on a grid with shape `2x3x4`, `grid_axes = [1]` forms groups along
axis 1 while fixing axes 0 and 2. `grid_axes = [0, 2]` forms groups over axes 0
and 2 while fixing axis 1.

The order of axes can matter for operations such as `all_to_all`.

### Sharding Values

`!shard.sharding` is a value type that describes a tensor placement. It is
usually produced by `shard.sharding`:

```mlir
%s = shard.sharding @grid split_axes = [[0], []] : !shard.sharding
```

Here, the first tensor dimension is split across grid axis 0. The second tensor
dimension is replicated with respect to the grid.

The `split_axes` list is indexed by tensor dimension. An empty inner list means
that dimension is not split.

### Shard Annotations

`shard.shard` attaches a sharding value to a ranked tensor:

```mlir
%local = shard.shard %arg0 to %s : tensor<8x16xf32>
```

Without `annotate_for_users`, the annotation describes the defining value. With
`annotate_for_users`, it describes how following users should see the value:

```mlir
%for_users = shard.shard %local to %s annotate_for_users : tensor<8x16xf32>
```

This distinction is important during propagation and partitioning because one
value may need different source-side and use-side layout information.

### SPMD Meaning

The execution model is SPMD: every process runs the same program, but operations
act on the local shard owned by that process. Collective operations require all
processes in their group to participate consistently.

That is why the dialect has both high-level tensor communication ops such as
`shard.all_reduce` and process-query ops such as
`shard.process_multi_index`.

### Halos And Uneven Shards

`shard.sharding` can include `halo_sizes` for data that overlaps neighboring
shards:

```mlir
%s = shard.sharding @grid split_axes = [[0]]
     halo_sizes = [1, 2] : !shard.sharding
```

It can also include `sharded_dims_offsets` to describe uneven shard boundaries:

```mlir
%s = shard.sharding @grid split_axes = [[0]]
     sharded_dims_offsets = [0, 3, 7, 10] : !shard.sharding
```

Those two forms are mutually exclusive in the same `shard.sharding`.

## Operations

The local LLVM checkout defines 22 `shard` operations.

### Placement And Grid Operations

| Operation | Purpose |
| --- | --- |
| `shard.grid` | Defines a named device/process grid. |
| `shard.grid_shape` | Returns selected grid dimension sizes as `index` values. |
| `shard.process_linear_index` | Returns the current process as one linear index in the grid. |
| `shard.process_multi_index` | Returns the current process coordinates over selected grid axes. |
| `shard.neighbors_linear_indices` | Returns previous and next neighbor process indices along split axes. |

Use these operations when the compiler needs to materialize process identity,
grid sizes, or neighbor relationships in SPMD IR.

### Sharding Metadata Operations

| Operation | Purpose |
| --- | --- |
| `shard.sharding` | Builds a `!shard.sharding` value from a grid, split axes, optional halos, and optional offsets. |
| `shard.shard` | Annotates a ranked tensor with a sharding, either for the value or for its users. |
| `shard.get_sharding` | Reads the sharding associated with a tensor value. |
| `shard.shard_shape` | Computes the shape of the shard for a given global shape, sharding, and device. |

These operations are mostly about compiler knowledge. They describe placement
and local shapes rather than performing a numerical computation.

### Collective Communication Operations

| Operation | Purpose |
| --- | --- |
| `shard.all_gather` | Concatenates pieces from all devices in a group and replicates the gathered result. |
| `shard.all_reduce` | Reduces values across a group and gives every participant the reduced result. |
| `shard.all_slice` | Slices a replicated value according to the process position; this has no inter-device communication. |
| `shard.all_to_all` | Splits each participant's input, exchanges pieces, and concatenates received pieces. |
| `shard.broadcast` | Copies the root device's value to the rest of the group. |
| `shard.gather` | Gathers pieces to the root device; non-root results are undefined. |
| `shard.reduce` | Reduces values to the root device; non-root results are undefined. |
| `shard.reduce_scatter` | Reduces across the group and scatters pieces of the reduced value. |
| `shard.scatter` | Splits a root value and distributes pieces to the group. |
| `shard.shift` | Shifts tensor values along a grid axis, optionally rotating. |

The reduction kind for `all_reduce`, `reduce`, and `reduce_scatter` is one of:
`sum`, `max`, `min`, `product`, `average`, `bitwise_and`, `bitwise_or`,
`bitwise_xor`, or `generic`.

### Point-To-Point And Halo Operations

| Operation | Purpose |
| --- | --- |
| `shard.send` | Sends a tensor to a destination in-group device. |
| `shard.recv` | Receives a tensor from a source in-group device. |
| `shard.update_halo` | Exchanges halo regions with neighboring shards. |

`send` and `recv` are not collective pure operations. They model explicit
point-to-point movement. `update_halo` is a higher-level halo exchange that the
MPI lowering expands into neighbor communication and subview copies.

## Attributes, Types, And Interfaces

### Type

| Type | Meaning |
| --- | --- |
| `!shard.sharding` | A sharding definition value produced by `shard.sharding`. |

### Attributes

| Attribute family | Meaning |
| --- | --- |
| `DenseI16ArrayAttr` grid axes | Stores lists such as `grid_axes = [0, 1]`. |
| `#shard.axisarray` / `GridAxesArrayAttr` | Stores nested split-axis arrays such as `[[0], []]`. |
| `ReductionKindAttr` | Stores reduction kinds such as `sum`, `max`, and `generic`. |
| `DenseI64ArrayAttr` shapes, halos, offsets, roots | Stores grid shapes, halo sizes, sharded dimension offsets, and root coordinates. |

### ShardingInterface

`ShardingInterface` lets operations from other dialects participate in sharding
propagation and partitioning. An operation implementing the interface can tell
the Shard transforms:

- its loop iterator types;
- which reduction iterators it has;
- its indexing maps;
- how to infer missing sharding annotations;
- how to add annotations;
- how to partition itself into SPMD form.

The local source includes ShardingInterface implementations or integrations for
some tensor and TOSA operations. Linalg also has sharding-related interface
support in its transform code.

## Transformations

### `sharding-propagation`

`sharding-propagation` is a function-level interface pass. It propagates
sharding information through operations that implement `ShardingInterface`.

Its traversal option can be:

| Option | Meaning |
| --- | --- |
| `forward` | Propagate from operands/results in forward order only. |
| `backward` | Propagate backward only. |
| `forward-backward` | Run a forward pass followed by a backward pass. |
| `backward-forward` | Run a backward pass followed by a forward pass; this is the default in the pass definition. |

After this pass, supported operations should have enough `shard.shard`
annotations for partitioning.

### `shard-partition`

`shard-partition` turns a fully annotated tensor function into SPMD form. The
input must have sharding annotations on ranked tensor operands, results, and
block arguments. Direct descendant operations must either implement
`ShardingInterface` or be fully replicated for their ranked tensor operands and
results.

The pass changes types to local shard types and inserts communication for
resharding. In the local implementation, the resharding patterns include:

- split a newly sharded axis with `shard.all_slice`;
- unsplit axes with `shard.all_gather`;
- move a split axis with `shard.all_to_all`;
- update halo regions with `shard.update_halo`.

### `shard-simplify`

`shard-simplify` applies simplification patterns for Shard operations. The pass
definition and implementation include:

- folding static `shard.grid_shape` results to constants;
- simplifying all-reduce endomorphisms such as all-reduce around compatible
  add/min/max operations;
- folding `shard.all_slice(shard.all_reduce(...))` into
  `shard.reduce_scatter` when grid and axes match;
- folding inverse `shard.all_gather` and `shard.all_slice` pairs when legal.

### Lowering Patterns Used By Passes

The Shard transform library also provides rewrite patterns, used by tests and
by the MPI conversion:

| Pattern group | Purpose |
| --- | --- |
| `populateProcessMultiIndexOpLoweringPatterns` | Rewrites `shard.process_multi_index` using `shard.process_linear_index`, `shard.grid_shape`, and affine delinearization. |
| `populateAllSliceOpLoweringPatterns` | Rewrites `shard.all_slice` into process-index queries, shape checks, and tensor slicing. |
| `populateAllOpLoweringPatterns` | Adds both of the above groups. |

## Conversions And Lowering Paths

### Shard To MPI

The main conversion path in this checkout is:

```text
shard tensor IR
  -> sharding-propagation
  -> shard-partition
  -> shard-simplify
  -> convert-shard-to-mpi
```

The `convert-shard-to-mpi` pass lowers implemented Shard operations to MPI and
supporting dialects such as `arith`, `scf`, `cf`, `affine`, `tensor`, `memref`,
`bufferization`, and `linalg`.

The conversion pass directly includes patterns for:

- `shard.get_sharding`;
- `shard.sharding`;
- `shard.process_linear_index`;
- `shard.neighbors_linear_indices`;
- `shard.shard_shape`;
- `shard.all_gather`;
- `shard.all_reduce`;
- `shard.reduce_scatter`;
- `shard.update_halo`.

It also invokes Shard lowering patterns for:

- `shard.process_multi_index`;
- `shard.all_slice`.

The pass keeps `shard.grid` and `shard.grid_shape` legal during conversion so
that global grid declarations and foldable grid-shape queries can remain during
the staged lowering.

Do not assume every operation in the Shard operation inventory has a direct MPI
conversion in this pass. Some operations are high-level modeling operations,
some are produced by partitioning, and some require custom or later lowering in
a complete pipeline.

### Rank Specialization With DLTI

If a module contains the DLTI attribute `MPI:comm_world-rank`, the
`convert-shard-to-mpi` pass can use that integer value as the current MPI rank
instead of emitting an `MPI_Comm_rank` query. That can expose constants for
shape propagation and fusion.

### After MPI

After `convert-shard-to-mpi`, the next stage is usually the MPI dialect's own
lowering path, eventually reaching LLVM-compatible IR and runtime calls. That
belongs to the `mpi` chapter, not the `shard` chapter.

## Example IR

This example declares a 1D grid, says that the first dimension of a tensor is
split across that grid, and annotates the tensor:

```mlir
module {
  shard.grid @grid_1d(shape = 4)

  func.func @annotate(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %s = shard.sharding @grid_1d split_axes = [[0], []] : !shard.sharding
    %sharded = shard.shard %arg0 to %s : tensor<8x16xf32>
    return %sharded : tensor<8x16xf32>
  }
}
```

This example shows communication over the same grid:

```mlir
module {
  shard.grid @grid_1d(shape = 4)

  func.func @communicate(%arg0: tensor<2x16xf32>) -> tensor<8x16xf32> {
    %gathered = shard.all_gather %arg0 on @grid_1d
      grid_axes = [0] gather_axis = 0
      : tensor<2x16xf32> -> tensor<8x16xf32>
    return %gathered : tensor<8x16xf32>
  }
}
```

This example computes the current process coordinate:

```mlir
module {
  shard.grid @grid_2d(shape = 2x4)

  func.func @where_am_i() -> (index, index) {
    %i, %j = shard.process_multi_index on @grid_2d : index, index
    return %i, %j : index, index
  }
}
```

## Mental Model

Think of `shard` as a contract between tensor IR and distributed execution.

At the top of the contract, tensors have global shapes and sharding annotations.
The compiler can propagate those annotations, reason about how operations map
to local shards, and insert communication when layouts do not line up.

At the bottom of the contract, communication becomes explicit: MPI ranks,
communicators, buffers, reductions, gathers, and halo sends/receives.

The dialect is valuable because it keeps distribution visible long enough for
compiler transformations to optimize it.

## Gotchas

- `shard.shard` is an annotation operation, not a data copy by itself.
- `annotate_for_users` changes whether a sharding describes a value's producer
  side or consumer side.
- `split_axes` is organized by tensor dimension, while `grid_axes` is organized
  by device-grid dimension. Mixing those up is the most common reading error.
- `grid_axes` defines communication groups by fixing all other grid coordinates.
- Collective ops assume SPMD participation. Removing, duplicating, or moving a
  collective on only some paths can make the runtime program invalid.
- `shard.all_slice` is named like a collective, but it does not communicate; it
  slices based on process position.
- `halo_sizes` and `sharded_dims_offsets` are mutually exclusive in
  `shard.sharding`.
- `convert-shard-to-mpi` currently covers a specific set of Shard operations.
  Unsupported high-level communication may need partitioning, simplification, or
  additional lowering first.
- Dynamic grid sizes and uneven shards often require shape computations to stay
  in the IR until enough constants are available.

## Source Map

Use these files in the LLVM repo when you need exact behavior:

| Topic | Files |
| --- | --- |
| Dialect base, type, attributes, reduction enum | `mlir/include/mlir/Dialect/Shard/IR/ShardBase.td` |
| Operation definitions | `mlir/include/mlir/Dialect/Shard/IR/ShardOps.td` |
| Operation implementation | `mlir/lib/Dialect/Shard/IR/ShardOps.cpp` |
| Main dialect documentation | `mlir/docs/Dialects/Shard.md` |
| Sharding interface | `mlir/include/mlir/Dialect/Shard/Interfaces/ShardingInterface.td`, `mlir/lib/Dialect/Shard/Interfaces/ShardingInterface.cpp` |
| Transform pass definitions | `mlir/include/mlir/Dialect/Shard/Transforms/Passes.td` |
| Partition pass and resharding patterns | `mlir/lib/Dialect/Shard/Transforms/Partition.cpp`, `mlir/include/mlir/Dialect/Shard/Transforms/ReshardingPartitionDoc.md` |
| Sharding propagation | `mlir/lib/Dialect/Shard/Transforms/ShardingPropagation.cpp` |
| Simplification patterns | `mlir/lib/Dialect/Shard/Transforms/Simplify.cpp` |
| Lowering helper patterns | `mlir/lib/Dialect/Shard/Transforms/Transforms.cpp` |
| Shard to MPI pass declaration | `mlir/include/mlir/Conversion/Passes.td` |
| Shard to MPI implementation | `mlir/lib/Conversion/ShardToMPI/ShardToMPI.cpp` |
| Dialect tests | `mlir/test/Dialect/Shard` |
| Conversion tests | `mlir/test/Conversion/ShardToMPI` |
