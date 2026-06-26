# ONNX-MLIR `krnl` Dialect

## Beginner Summary

The ONNX-MLIR `krnl` dialect is the main lowering dialect used by the classic
ONNX-MLIR compiler path. It sits between ONNX graph operations and low-level
MLIR such as Affine, SCF, MemRef, Vector, OpenMP, and LLVM.

For a beginner, the useful mental model is:

```text
ONNX tensor graph
  -> convert-onnx-to-krnl
  -> explicit loops, buffers, copies, calls, runtime hooks
  -> convert-krnl-to-affine
  -> affine/scf/memref/vector/openmp
  -> convert-krnl-to-llvm
  -> LLVM plus ONNX-MLIR runtime calls
```

Krnl is not a frontend dialect and not a final machine dialect. It is a
compiler engineering dialect. It gives ONNX-MLIR a place to express loop nests,
loop scheduling decisions, tiled matrix multiplication, sequence storage,
runtime entry points, string helpers, and fallback library calls before the IR
is fully lowered.

## Why This Dialect Exists

ONNX operations are graph-level tensor operations. A compiler eventually needs
concrete memory, loops, scalar arithmetic, runtime calls, and ABI handling.
Lowering directly from every ONNX op to LLVM would be difficult to maintain and
would hide important optimization choices.

The `krnl` dialect exists as a structured midpoint. ONNX lowering patterns can
emit Krnl loop references, loads, stores, buffer copies, specialized kernels,
and runtime helper operations without immediately committing to final loop
syntax or LLVM ABI details. Later passes interpret those Krnl operations and
replace them with standard MLIR loops, memory operations, OpenMP constructs, or
LLVM operations.

This split is especially important for machine learning workloads. Matrix
multiplication, convolution lowering, RNNs, reductions, elementwise operations,
dynamic shapes, optional values, and sequences all need different lowering
strategies. Krnl gives those strategies a common implementation language.

## When It Matters

You will see `krnl` when debugging the normal ONNX-MLIR lowering pipeline. It
matters when:

- an ONNX operator has already been lowered out of the ONNX dialect;
- loop bounds, tiling, unrolling, SIMD, or parallelization choices are visible;
- a tensor has become a memref and accesses are explicit;
- sequence operations need an implementation in terms of memrefs;
- constants, entry point signatures, or runtime calls are being prepared;
- an op is being lowered through an external library call instead of expanded
  inline;
- the compiler is about to move from structured Krnl loops into Affine or LLVM.

Use Krnl when implementing or debugging ONNX-MLIR lowering. Most users should
not write Krnl IR by hand, but reading it is one of the best ways to understand
what ONNX-MLIR will actually execute.

## Core Types

The dialect defines two custom types:

```text
!krnl.loop
!krnl.string
```

`!krnl.loop` is a symbolic loop handle. It is produced by
`krnl.define_loops`, transformed by scheduling operations such as `krnl.block`,
`krnl.permute`, `krnl.unroll`, and `krnl.parallel`, and consumed by
`krnl.iterate`. The handle is not the final induction variable. It is a
compiler token that lets Krnl describe a loop schedule before final Affine or
SCF loops exist.

`!krnl.string` represents string data used by operations such as `krnl.strlen`,
`krnl.strncmp`, and `krnl.find_index`. During LLVM lowering it is converted to
an LLVM-compatible representation.

## Complete Operation Inventory

The current `krnl` dialect defines 50 operations:

```text
krnl.acos
krnl.acosh
krnl.asin
krnl.asinh
krnl.atan
krnl.atanh
krnl.block
krnl.call
krnl.copy_from_tile_buffer
krnl.copy_to_tile_buffer
krnl.define_loops
krnl.entry_point
krnl.erf
krnl.find_index
krnl.get_induction_var_value
krnl.get_linear_offset_index
krnl.global
krnl.isinf
krnl.isnan
krnl.iterate
krnl.load
krnl.matmul
krnl.memcpy
krnl.memset
krnl.movable
krnl.noValue
krnl.parallel
krnl.parallel_clause
krnl.permute
krnl.prefetch
krnl.print
krnl.print_tensor
krnl.random_normal
krnl.region
krnl.round_even
krnl.runtime_instrument
krnl.runtime_instrument_init
krnl.seqalloc
krnl.seqdealloc
krnl.seqextract
krnl.seqstore
krnl.specialized_kernel
krnl.store
krnl.strlen
krnl.strncmp
krnl.tan
krnl.terminate
krnl.unroll
krnl.vector_type_cast
krnl.yield
```

## Loop And Schedule Operations

The loop model begins with `krnl.define_loops`, which creates one or more
`!krnl.loop` handles. Those handles can be transformed before they are lowered.

`krnl.block` splits a loop into a block loop and a local loop. This is the Krnl
way to describe tiling. `krnl.permute` reorders loop handles. `krnl.unroll`
marks a loop for full unrolling. `krnl.parallel` marks one or more Krnl loops as
parallel and may carry a requested thread count or processor binding.

`krnl.iterate` is the region operation that turns scheduled loop handles into a
nested loop computation. It records original loops, optimized loops, lower
bounds, upper bounds, and optional loop-carried values. Its body is terminated
by `krnl.yield`.

`krnl.get_induction_var_value` maps symbolic loop handles back to concrete
induction values inside the body. This is important after blocking or
permutation, because the loop handle may no longer correspond directly to a
simple source index.

`krnl.movable` is an internal helper used during Krnl-to-Affine lowering. It
temporarily holds operations that must move under newly materialized loops when
the loop nest is not perfectly nested yet.

## Memory And Buffer Operations

Krnl uses memrefs heavily. `krnl.load` and `krnl.store` are Krnl-level memory
access operations. They make memory effects explicit while still participating
in Krnl-specific lowering.

`krnl.memcpy` copies a number of elements from one memref to another using
linear offsets. `krnl.memset` fills a buffer with a value and has a `delayed`
mode for normalized memrefs. `krnl.get_linear_offset_index` computes a linear
offset from a multidimensional memref index. `krnl.prefetch` records a prefetch
request with locality and cache attributes.

`krnl.vector_type_cast` views a memref with scalar element type as a memref with
vector element type. This supports SIMD lowering without changing the underlying
storage.

`krnl.global` holds constant data. It can carry a name, dense value, offset, and
alignment. During LLVM lowering, globals can become embedded data, large
read-only data, or external constant-file data depending on compiler options.

## Tiled Kernel Operations

`krnl.matmul` is a specialized tiled matrix multiplication operation. It
represents a local matrix multiply over A, B, and C buffers or original memrefs,
with global start indices, global upper bounds, optional tile-size attributes,
and optimization flags such as `simdize`, `unroll`, and `overcompute`.

`krnl.copy_to_tile_buffer` fills a tile buffer from source memory, optionally
padding or transposing. `krnl.copy_from_tile_buffer` writes a tile buffer back to
destination memory. These operations let ONNX-MLIR describe cache tiling around
matrix multiplication and related kernels before expanding to affine loops.

`krnl.specialized_kernel` is an interface-bearing operation for specialized
kernel lowering. It is a hook point rather than an ordinary ONNX computation
operation.

## Sequence Operations

ONNX sequences are implemented using memrefs of memrefs in this path. Krnl has
four sequence operations:

```text
krnl.seqalloc
krnl.seqdealloc
krnl.seqextract
krnl.seqstore
```

`krnl.seqalloc` allocates sequence storage. `krnl.seqdealloc` performs deep
deallocation of the sequence and its elements. `krnl.seqextract` reads an
element and usually copies it so ownership and deallocation remain correct.
`krnl.seqstore` stores an element into existing sequence storage.

The `convert-seq-to-memref` pass lowers these operations to memref and standard
operations. In the normal compiler pipeline this pass is currently guarded by
pipeline choices, but it is the dedicated lowering path for Krnl sequence ops.

## Runtime, Calls, And Utility Operations

`krnl.call` is a generic external-call operation. It lets an ONNX op lower to a
runtime or library call at Krnl level. Its operands include output and input
memrefs, and attributes can be copied from the original ONNX operation.

`krnl.entry_point` records the ONNX model entry point, number of inputs,
number of outputs, and JSON-like signature data. `convert-krnl-to-llvm` uses it
to generate runtime entry point and signature functions for the compiled model.

`krnl.noValue` represents absence of a value, usually for optional ONNX
operands. It lowers to a null-like representation when appropriate.

`krnl.runtime_instrument` and `krnl.runtime_instrument_init` support profiling
or debugging instrumentation. `krnl.print` and `krnl.print_tensor` lower to
runtime printing helpers.

String and helper operations include `krnl.strlen`, `krnl.strncmp`,
`krnl.find_index`, and `krnl.random_normal`. Math helper operations include
`krnl.round_even`, `krnl.erf`, `krnl.isinf`, `krnl.isnan`, `krnl.acos`,
`krnl.acosh`, `krnl.asin`, `krnl.asinh`, `krnl.atan`, `krnl.atanh`, and
`krnl.tan`.

## Region And Terminator Operations

`krnl.region` is an affine-scope boundary. It exists so Krnl loops inside it can
be converted to affine loops when their bounds are defined at the right scope.
It does not make the enclosed loops affine by itself. After affine lowering has
run, `lower-krnl-region` moves the operations out and erases the region.

`krnl.terminate` is the terminator used by some Krnl regions. `krnl.yield`
terminates `krnl.iterate` bodies and returns loop-carried values to the parent
operation.

## Transformations And Conversions

The most important Krnl-related passes are:

```text
convert-onnx-to-krnl
convert-krnl-to-affine
lower-krnl-region
convert-seq-to-memref
process-krnl-parallel-clause
scf-parallel-private
buffer-omploop-hoisting
instrument
instrument-cleanup
convert-krnl-to-llvm
```

`convert-onnx-to-krnl` lowers ONNX operations into Krnl and supporting standard
dialects. It has options for tiling, SIMD, parallelization, fast math,
intermediate IR emission for tests, and forcing selected ops to become
`krnl.call`.

`convert-krnl-to-affine` interprets Krnl loop operations. It lowers
`krnl.define_loops`, `krnl.iterate`, `krnl.block`, `krnl.permute`,
`krnl.unroll`, `krnl.get_induction_var_value`, `krnl.matmul`,
`krnl.copy_to_tile_buffer`, `krnl.copy_from_tile_buffer`, `krnl.memset`, and
related operations into Affine, Vector, MemRef, and standard MLIR operations.
With parallelization enabled, it also carries parallel loop information forward.

`lower-krnl-region` erases `krnl.region` after affine lowering has made the
scope wrapper unnecessary.

`process-krnl-parallel-clause` migrates information from Krnl parallel clauses
to OpenMP parallel operations after OpenMP lowering. `scf-parallel-private` and
`buffer-omploop-hoisting` are supporting passes for safe and efficient OpenMP
parallel memory behavior.

`convert-krnl-to-llvm` completes lowering to LLVM. It handles remaining Krnl
operations such as entry points, globals, external calls, string helpers,
printing, instrumentation, random normal generation, `krnl.noValue`, and the
runtime ABI. It also emits model signature functions, compilation metadata,
constant-file loading helpers when requested, C wrapper attributes, and runtime
calls used by the ONNX-MLIR execution library.

## What The Dialect Implies

Seeing Krnl means the model has moved out of pure ONNX graph form. Tensor
semantics have been turned into explicit control flow, memory access, buffer
management, or runtime calls.

It also implies that the final target is still not fixed. A `krnl.matmul` might
lower to scalar affine loops, vectorized loops, or unrolled tiled code depending
on options and shapes. A `krnl.parallel` marker may become OpenMP only if the
parallel path is enabled. A `krnl.call` may keep a computation as a library
call instead of expanding it inline.

The tradeoff is complexity. Krnl is easier to optimize than raw ONNX, but it is
less declarative. When reading it, you are looking at compiler strategy, not
just model math.

## How To Use It In Practice

When reading Krnl IR, first find the `krnl.define_loops` and `krnl.iterate`
operations. They show the loop structure. Then read scheduling markers such as
`krnl.block`, `krnl.permute`, `krnl.unroll`, and `krnl.parallel` before
assuming what the loop nest will look like after lowering.

Next, inspect memory operations. `krnl.load`, `krnl.store`, `krnl.memset`,
tile-buffer copies, and `krnl.global` show where data is read, written,
initialized, and stored.

Finally, look for boundary operations. `krnl.entry_point` explains how the model
will be exposed through the runtime ABI. `krnl.call` means part of the model is
delegated to an external function. `krnl.noValue` usually corresponds to an
optional ONNX operand.

## Source Pointers

The core definitions are in `src/Dialect/Krnl/Krnl.td`,
`src/Dialect/Krnl/KrnlOps.cpp`, and `src/Dialect/Krnl/KrnlTypes.*`. The ONNX
lowering patterns live under `src/Conversion/ONNXToKrnl`. Krnl-to-Affine
lowering is in `src/Conversion/KrnlToAffine`, sequence lowering is in
`src/Conversion/KrnlSeqToMemref`, and final LLVM/runtime lowering is in
`src/Conversion/KrnlToLLVM`.
