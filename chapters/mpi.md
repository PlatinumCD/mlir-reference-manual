# `mpi` Dialect

## Beginner Summary

The `mpi` dialect models Message Passing Interface operations in MLIR.

MPI is a standard library interface for distributed-memory parallel programs.
Instead of one process sharing memory with all other workers, each process has
its own address space and communicates by sending and receiving messages.

The `mpi` dialect gives MLIR operations for:

- Initializing and finalizing MPI.
- Getting `MPI_COMM_WORLD`.
- Querying process rank and world size.
- Splitting communicators.
- Blocking send and receive.
- Nonblocking send and receive handles.
- Waiting on requests.
- Barriers.
- Collective communication such as all-gather, all-reduce, and
  reduce-scatter-block.
- Checking MPI return values.

Think of `mpi` as MLIR's distributed-process communication dialect. It sits
between higher-level distributed IR, such as `shard`, and lower-level calls to
an actual MPI implementation.

## Why This Dialect Exists

MPI is a C library API. Raw MPI calls expose ABI details:

- How `MPI_Comm` is represented.
- How predefined datatypes are represented.
- How predefined reduction operations are represented.
- How `MPI_COMM_WORLD` is accessed.
- How Open MPI and MPICH differ.
- How memref buffers become raw pointers and element counts.

The `mpi` dialect hides those ABI details behind MLIR types and operations.

For example, a pass can create:

```mlir
mpi.allreduce(%send, %recv, MPI_SUM, %comm)
  : memref<16xf32>, memref<16xf32>
```

without immediately deciding whether the final target uses MPICH-style integer
handles or Open MPI global symbols.

This lets higher-level distributed transformations describe communication in a
target-independent way, then lower to concrete MPI calls later.

## When It Matters

The `mpi` dialect matters when an MLIR pipeline represents communication
between distributed processes.

It commonly appears in flows like:

```text
shard distributed operations
  -> convert-shard-to-mpi
  -> mpi.comm_world, mpi.comm_rank, mpi.allreduce, mpi.send, mpi.recv, ...
  -> convert-to-llvm
  -> LLVM dialect calls to MPI_Init, MPI_Comm_rank, MPI_Allreduce, ...
  -> native code linked with an MPI implementation
```

It is especially relevant for:

- Distributed tensor and array programs.
- SPMD-style process grids.
- Lowering `shard` communication to MPI.
- Modeling point-to-point communication.
- Modeling collective communication.
- Preserving MPI concepts before choosing an ABI.
- Generating code for MPICH-compatible or Open MPI-compatible environments.

The dialect is still described by its own source as under active development.
Treat it as useful compiler infrastructure, not as a fully complete MPI 4.0
surface.

## When To Use It

Use the `mpi` dialect when your IR needs explicit distributed communication.

Use it for:

- Communicator handles.
- Rank and size queries.
- Blocking sends and receives.
- Nonblocking sends and receives.
- Collectives over memref buffers.
- Communicator splits for subgroups.
- Translating `shard` operations into concrete communication.
- Lowering to MPI library calls through LLVM.

Do not use it for ordinary threading or shared-memory parallelism. Use `scf`,
`omp`, `gpu`, `async`, or other parallel dialects when the execution model is
not distributed MPI.

Do not use it as a high-level distributed tensor language. If the program still
talks in terms of process grids, tensor sharding, halo exchange, or
partitioned tensor values, start with `shard` and lower to `mpi` when the
pipeline is ready to express concrete communication.

## Core Concepts

### Processes, Rank, And Size

MPI programs run as multiple processes.

Each process has:

- A rank: its integer identity within a communicator.
- A communicator size: the number of processes in that communicator.

In the `mpi` dialect:

```mlir
%comm = mpi.comm_world : !mpi.comm
%rank = mpi.comm_rank(%comm) : i32
%size = mpi.comm_size(%comm) : i32
```

`%rank` and `%size` are ordinary `i32` values. The communicator itself has type
`!mpi.comm`.

### Communicators

A communicator identifies a group of processes that can communicate.

`mpi.comm_world` returns the predefined world communicator:

```mlir
%comm = mpi.comm_world : !mpi.comm
```

`mpi.comm_split` partitions a communicator into sub-communicators:

```mlir
%new = mpi.comm_split(%comm, %color, %key) : !mpi.comm
```

The color chooses the subgroup. The key chooses ordering within the new
communicator.

### Memrefs As MPI Buffers

MPI sends and collectives operate on buffers.

The MLIR dialect uses memrefs rather than raw pointers:

```mlir
mpi.send(%buf, %tag, %dest, %comm) : memref<16xf32>, i32, i32
```

During LLVM lowering, the memref descriptor is converted into:

- A data pointer.
- An element count.
- An MPI datatype derived from the element type.

This is why the dialect is easier to use from MLIR than raw C calls. It keeps
buffer shape and element type visible until lowering.

### Optional Return Values

Most MPI dialect operations can optionally return `!mpi.retval`.

With no return value:

```mlir
mpi.send(%buf, %tag, %dest, %comm) : memref<16xf32>, i32, i32
```

With a return value:

```mlir
%err = mpi.send(%buf, %tag, %dest, %comm)
  : memref<16xf32>, i32, i32 -> !mpi.retval
```

The return value represents the integer status returned by the underlying MPI
call. It can be checked with `mpi.retval_check` or mapped to an error class
with `mpi.error_class`.

In this checkout, the MPI-to-LLVM patterns for `mpi.init` and `mpi.finalize`
replace the op with an MPI call returning `i32`, so the best-tested lowering
form captures their optional return value.

### Blocking And Nonblocking Communication

Blocking point-to-point communication:

```mlir
mpi.send(%buf, %tag, %dest, %comm) : memref<16xf32>, i32, i32
mpi.recv(%buf, %tag, %source, %comm) : memref<16xf32>, i32, i32
```

Nonblocking point-to-point communication:

```mlir
%req = mpi.isend(%buf, %tag, %dest, %comm)
  : memref<16xf32>, i32, i32 -> !mpi.request
mpi.wait(%req) : !mpi.request
```

The request handle has type `!mpi.request`.

The nonblocking operations are present in the dialect. The current MPI-to-LLVM
pattern registration in this checkout does not include lowering patterns for
`mpi.isend`, `mpi.irecv`, or `mpi.wait`.

### Collectives

Collectives involve all processes in a communicator.

The dialect currently includes:

- `mpi.allgather`
- `mpi.allreduce`
- `mpi.reduce_scatter_block`
- `mpi.barrier`

These are useful targets for lowering higher-level distributed operations such
as all-reduce or shard gather/scatter operations.

### Implementation Choice Through DLTI

The MPI-to-LLVM lowering has implementation traits for MPICH-compatible MPI and
Open MPI.

The module can carry:

```mlir
module attributes {dlti.map = #dlti.map<"MPI:Implementation" = "OpenMPI">} {
  ...
}
```

Recognized values include:

- `"MPICH"`
- `"OpenMPI"`

If no implementation is specified, or an unknown value is used, the lowering
defaults to MPICH behavior.

There is also an `mpi.dlti` attribute convention used by canonicalization and
Shard-to-MPI:

```mlir
module attributes {
  mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 5,
                       "MPI:comm_world_size" = 12>
} {
  ...
}
```

When `mpi.comm_rank` or `mpi.comm_size` uses `mpi.comm_world`, canonicalization
can fold those queries to constants if these DLTI values are present.

## Operations

### Lifecycle Operations

`mpi.init`
: Initializes the MPI library, equivalent to `MPI_Init(NULL, NULL)`. Passing
  `argc` and `argv` is not currently supported. It may return `!mpi.retval`.

`mpi.finalize`
: Finalizes the MPI library, equivalent to `MPI_Finalize()`. After finalization
  most MPI calls must not be made. It may return `!mpi.retval`.

### Communicator Operations

`mpi.comm_world`
: Returns the predefined world communicator as `!mpi.comm`.

`mpi.comm_rank`
: Returns the rank of the current process in a communicator as `i32`. It may
  also return `!mpi.retval`.

`mpi.comm_size`
: Returns the communicator size as `i32`. It may also return `!mpi.retval`.

`mpi.comm_split`
: Splits a communicator into sub-communicators using `color` and `key` `i32`
  values. It returns a new `!mpi.comm` and may also return `!mpi.retval`.

### Point-To-Point Operations

`mpi.send`
: Blocking send of a memref buffer to a destination rank with a tag and
  communicator. It may return `!mpi.retval`.

`mpi.recv`
: Blocking receive into a memref buffer from a source rank with a tag and
  communicator. The current operation ignores MPI status. It may return
  `!mpi.retval`.

`mpi.isend`
: Begins a nonblocking send and returns an `!mpi.request`. It may also return
  `!mpi.retval`.

`mpi.irecv`
: Begins a nonblocking receive and returns an `!mpi.request`. It may also
  return `!mpi.retval`.

`mpi.wait`
: Waits for an `!mpi.request` to complete. The current operation ignores MPI
  status. It may return `!mpi.retval`.

### Collective Operations

`mpi.barrier`
: Blocks until all processes in the communicator reach the barrier. It may
  return `!mpi.retval`.

`mpi.allgather`
: Collects a contribution from every process and stores the gathered result in
  each process's receive buffer. It may return `!mpi.retval`.

`mpi.allreduce`
: Reduces values across all processes and stores the result in the receive
  buffer of every process. The reduction operation is an MPI reduction enum
  such as `MPI_SUM`, `MPI_MAX`, or `MPI_MIN`.

`mpi.reduce_scatter_block`
: Reduces values across all processes, then scatters equal-sized result blocks
  to each process. The send and receive buffers must have the same element
  type.

### Error Operations

`mpi.retval_check`
: Compares an `!mpi.retval` against an MPI error class attribute such as
  `<MPI_SUCCESS>` and returns `i1`.

`mpi.error_class`
: Maps an MPI return value to a known MPI error class, equivalent to
  `MPI_Error_class`.

### MPI Types

The dialect defines:

- `!mpi.comm`: communicator handle.
- `!mpi.request`: asynchronous request handle.
- `!mpi.retval`: MPI function return value or error code.
- `!mpi.status`: receive status handle. The current receive and wait ops still
  use `MPI_STATUS_IGNORE` in their descriptions/lowerings rather than exposing
  full status handling.

### MPI Attributes

MPI error classes are represented by `#mpi.errclass` attributes and used by
`mpi.retval_check`.

MPI reduction operations are represented by `MPI_ReductionOpEnum` values used
by collectives. The defined reduction names include:

- `MPI_OP_NULL`
- `MPI_MAX`
- `MPI_MIN`
- `MPI_SUM`
- `MPI_PROD`
- `MPI_LAND`
- `MPI_BAND`
- `MPI_LOR`
- `MPI_BOR`
- `MPI_LXOR`
- `MPI_BXOR`
- `MPI_MINLOC`
- `MPI_MAXLOC`
- `MPI_REPLACE`

## Transformations

### Canonicalization

The MPI dialect has canonicalization patterns for:

- `mpi.comm_rank`
- `mpi.comm_size`
- `mpi.send`
- `mpi.recv`
- `mpi.isend`
- `mpi.irecv`

`mpi.comm_rank` and `mpi.comm_size` can fold to constants when:

- The communicator comes from `mpi.comm_world`.
- The module has `mpi.dlti` entries for `MPI:comm_world_rank` or
  `MPI:comm_world_size`.

Example:

```mlir
module attributes {
  mpi.dlti = #dlti.map<"MPI:comm_world_size" = 12,
                       "MPI:comm_world_rank" = 5>
} {
  ...
}
```

The send and receive canonicalizers can fold certain `memref.cast` operations
away when a dynamic-shape memref cast has a static-shape source.

### Shard To MPI

The main producer-side conversion is:

```text
convert-shard-to-mpi
```

This pass lowers supported `shard` communication operations to MPI operations.

It can introduce:

- `mpi.comm_world`
- `mpi.comm_rank`
- `mpi.comm_size`
- `mpi.comm_split`
- `mpi.allreduce`
- `mpi.allgather`
- `mpi.reduce_scatter_block`
- `mpi.send`
- `mpi.recv`
- `affine.delinearize_index`
- `memref.alloc`
- `bufferization.to_buffer`
- `bufferization.to_tensor`
- `linalg.copy`
- `scf` and `cf` control flow

The pass uses `MPI:comm_world-rank` from `mpi.dlti` when available. That lets
the conversion materialize process coordinates as constants in cases where the
rank is known statically.

For some tensor and non-leading-axis cases, the pass creates intermediate
buffers or transposes/copies data so the MPI collective sees a contiguous
memref layout.

## Conversions And Lowering Paths

### MPI To LLVM Through `convert-to-llvm`

There is no standalone `convert-mpi-to-llvm` pass name in this checkout.

Instead, MPI registers a conversion interface used by:

```text
convert-to-llvm
```

The registered MPI-to-LLVM patterns lower these operations:

- `mpi.init`
- `mpi.finalize`
- `mpi.comm_world`
- `mpi.comm_rank`
- `mpi.comm_size`
- `mpi.comm_split`
- `mpi.send`
- `mpi.recv`
- `mpi.allgather`
- `mpi.allreduce`
- `mpi.reduce_scatter_block`

The lowering emits LLVM dialect calls such as:

- `MPI_Init`
- `MPI_Finalize`
- `MPI_Comm_rank`
- `MPI_Comm_size`
- `MPI_Comm_split`
- `MPI_Send`
- `MPI_Recv`
- `MPI_Allgather`
- `MPI_Allreduce`
- `MPI_Reduce_scatter_block`

### ABI Handling

The lowering has implementation traits for MPICH-compatible MPI and Open MPI.

MPICH-style lowering uses integer constants for predefined handles such as
communicators, datatypes, and reduction operations.

Open MPI-style lowering uses external global symbols such as:

- `ompi_mpi_comm_world`
- `ompi_mpi_float`
- `ompi_mpi_double`
- `ompi_mpi_sum`

The selected implementation changes the LLVM types and calls that are emitted.

### Current Lowering Gaps

The dialect contains more operations than the current MPI-to-LLVM pattern set
lowers.

In this checkout, registered MPI-to-LLVM patterns do not include:

- `mpi.isend`
- `mpi.irecv`
- `mpi.wait`
- `mpi.barrier`
- `mpi.retval_check`
- `mpi.error_class`

Those operations are still valid dialect operations. They just need additional
lowering support before a complete LLVM conversion pipeline can consume them.

Also note that `mpi.init` and `mpi.finalize` are defined with optional return
values, but the local conversion pattern replaces them with LLVM calls that
return `i32`. Capturing their `!mpi.retval` form is the safer path for current
LLVM lowering.

## Example IR

### Rank And Point-To-Point Exchange

```mlir
func.func @rank_and_exchange(%buf: memref<16xf32>) {
  %err = mpi.init : !mpi.retval
  %comm = mpi.comm_world : !mpi.comm
  %rank = mpi.comm_rank(%comm) : i32
  %size = mpi.comm_size(%comm) : i32
  mpi.send(%buf, %rank, %rank, %comm) : memref<16xf32>, i32, i32
  mpi.recv(%buf, %rank, %rank, %comm) : memref<16xf32>, i32, i32
  mpi.finalize
  return
}
```

This is a minimal shape of an MPI program in the dialect:

```text
initialize
get world communicator
query rank and size
communicate through memref buffers
finalize
```

The example sends to and receives from the same rank only to keep the IR small.
Real programs usually compute destination and source ranks from the algorithm.

### Collectives

```mlir
func.func @collectives(%send: memref<16xf32>, %recv: memref<16xf32>) {
  %comm = mpi.comm_world : !mpi.comm
  mpi.allgather(%send, %recv, %comm) : memref<16xf32>, memref<16xf32>
  mpi.allreduce(%send, %recv, MPI_SUM, %comm)
    : memref<16xf32>, memref<16xf32>
  mpi.reduce_scatter_block(%send, %recv, MPI_MAX, %comm)
    : memref<16xf32>, memref<16xf32>
  mpi.barrier(%comm)
  return
}
```

Collectives use a communicator and memref buffers. Reduction collectives also
carry the MPI reduction operation.

### Nonblocking Requests

```mlir
func.func @nonblocking(%buf: memref<16xf32>) {
  %comm = mpi.comm_world : !mpi.comm
  %rank = mpi.comm_rank(%comm) : i32
  %send_req = mpi.isend(%buf, %rank, %rank, %comm)
    : memref<16xf32>, i32, i32 -> !mpi.request
  %recv_req = mpi.irecv(%buf, %rank, %rank, %comm)
    : memref<16xf32>, i32, i32 -> !mpi.request
  mpi.wait(%send_req) : !mpi.request
  mpi.wait(%recv_req) : !mpi.request
  return
}
```

The nonblocking operations model request-based MPI communication. They are
useful at the dialect level, even though current MPI-to-LLVM lowering does not
yet include them.

### Known Rank And Size

```mlir
module attributes {
  mpi.dlti = #dlti.map<"MPI:comm_world_size" = 12,
                       "MPI:comm_world_rank" = 5>
} {
  func.func @known_world() -> (i32, i32) {
    %comm = mpi.comm_world : !mpi.comm
    %size = mpi.comm_size(%comm) : i32
    %rank = mpi.comm_rank(%comm) : i32
    return %size, %rank : i32, i32
  }
}
```

After canonicalization, `%size` can become `arith.constant 12 : i32` and
`%rank` can become `arith.constant 5 : i32`.

## Mental Model

The `mpi` dialect is a bridge.

At the top, distributed abstractions may talk about process grids, shards, and
tensor partitions.

At the bottom, executable code must call an MPI library with pointers, counts,
datatypes, communicators, and ABI-specific predefined values.

The `mpi` dialect sits in the middle:

```text
distributed intent
  -> mpi operations over memrefs and communicator types
  -> LLVM calls using MPICH or Open MPI ABI details
```

For beginners, the most useful way to read an MPI op is:

```text
this is an MPI library call, but still in MLIR form
```

`!mpi.comm`, `!mpi.request`, and `!mpi.retval` keep MPI concepts explicit
without forcing every pass to understand C ABI details.

## Gotchas

- The dialect is under active development and does not cover all of MPI 4.0.
- `mpi.init` currently models `MPI_Init(NULL, NULL)` only; passing `argc` and
  `argv` is not supported.
- Most return values are optional in the IR, but current lowering is best
  exercised when `mpi.init` and `mpi.finalize` return `!mpi.retval`.
- `mpi.recv` and `mpi.wait` currently use `MPI_STATUS_IGNORE`; full status
  handling is not exposed in the tested operations.
- Nonblocking ops are in the dialect, but `mpi.isend`, `mpi.irecv`, and
  `mpi.wait` are not in the current registered MPI-to-LLVM lowering pattern
  set.
- `mpi.barrier`, `mpi.retval_check`, and `mpi.error_class` also do not appear
  in the current MPI-to-LLVM pattern set.
- MPI collectives operate on memrefs. Shape, contiguity, and element type
  matter when lowering to raw MPI calls.
- MPI-to-LLVM lowering supports a limited set of element types for MPI datatype
  mapping, including common floats and 8/16/32/64-bit integer types.
- MPICH and Open MPI use different ABI representations. Set
  `"MPI:Implementation"` in DLTI when the target implementation matters.
- `convert-shard-to-mpi` may allocate and copy buffers to satisfy MPI layout
  requirements.

## Source Map

Primary definitions:

- `mlir/include/mlir/Dialect/MPI/IR/MPI.td`
- `mlir/include/mlir/Dialect/MPI/IR/MPIOps.td`
- `mlir/include/mlir/Dialect/MPI/IR/MPITypes.td`
- `mlir/include/mlir/Dialect/MPI/IR/Utils.h`
- `mlir/lib/Dialect/MPI/IR/MPI.cpp`
- `mlir/lib/Dialect/MPI/IR/MPIOps.cpp`
- `mlir/include/mlir/Conversion/ShardToMPI/ShardToMPI.h`
- `mlir/lib/Conversion/ShardToMPI/ShardToMPI.cpp`
- `mlir/include/mlir/Conversion/MPIToLLVM/MPIToLLVM.h`
- `mlir/lib/Conversion/MPIToLLVM/MPIToLLVM.cpp`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/test/Dialect/MPI/`
- `mlir/test/Conversion/MPIToLLVM/`
- `mlir/test/Conversion/ShardToMPI/`

All MPI dialect operations covered in this chapter:

- `mpi.allgather`
- `mpi.allreduce`
- `mpi.barrier`
- `mpi.comm_rank`
- `mpi.comm_size`
- `mpi.comm_split`
- `mpi.comm_world`
- `mpi.error_class`
- `mpi.finalize`
- `mpi.init`
- `mpi.irecv`
- `mpi.isend`
- `mpi.recv`
- `mpi.reduce_scatter_block`
- `mpi.retval_check`
- `mpi.send`
- `mpi.wait`

MPI-related conversion paths covered:

- `convert-shard-to-mpi`
- `convert-to-llvm`
- Shard to MPI communication lowering.
- MPI to LLVM dialect calls.
- MPICH and Open MPI ABI selection through DLTI.
