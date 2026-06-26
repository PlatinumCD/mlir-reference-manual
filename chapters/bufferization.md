# `bufferization` Dialect

## Beginner Summary

The `bufferization` dialect supports the move from tensor IR to buffer IR.

In MLIR, tensor values describe mathematical values. They are often immutable,
SSA-style, and convenient for optimization. Real machine code eventually needs
memory: allocations, reads, writes, copies, and deallocations. Bufferization is
the process that maps tensor computations onto buffers, usually represented with
`memref` types and `memref` operations.

The dialect itself is small: it has seven operations. The surrounding
infrastructure is large: One-Shot Bufferize, bufferization interfaces,
deallocation passes, allocation hoisting passes, and Transform dialect ops.

## Why This Dialect Exists

Tensor IR is good for analysis because tensor values behave like immutable SSA
values. Buffer IR is good for execution because it talks about memory. A
compiler needs a controlled bridge between the two.

The `bufferization` dialect exists to make that bridge explicit:

- It marks tensor values that should become fresh allocations.
- It materializes boundaries between tensor and buffer worlds.
- It records where a tensor should be written into a destination buffer.
- It represents ownership-aware deallocation before lowering to raw
  `memref.dealloc`.
- It provides interfaces that tell One-Shot Bufferize how an op reads, writes,
  aliases, and allocates buffers.

The dialect description states the central idea directly: bufferization is the
process of converting `tensor` types to `memref` types.

## When It Matters

Bufferization matters whenever an MLIR pipeline starts with tensor-level IR and
needs to reach executable lower-level IR.

It is common in:

- Linalg-on-tensors pipelines.
- Tensor programs that eventually lower to loops, LLVM, GPU code, or another
  low-level target.
- Pipelines that need to decide whether tensor results can reuse destination
  buffers.
- Pipelines that must insert deallocations and avoid memory leaks.
- Compiler debugging sessions where unexpected copies or allocations appear.

If you are learning MLIR lowering, bufferization is one of the key transitions:
it is where value semantics meet memory semantics.

## When To Use It

Use bufferization infrastructure when:

- Your IR has tensors and you need memrefs.
- Your dialect implements tensor operations and should participate in
  One-Shot Bufferize.
- You need to preserve partial tensor/buffer boundaries with `to_tensor` or
  `to_buffer`.
- You need to control where a tensor result materializes.
- You need automatic deallocation after bufferization.

Do not use it as a replacement for `memref` once your program is fully in
buffer form. After bufferization and deallocation cleanup, most
`bufferization` ops should disappear or lower to `memref`/control-flow ops.

## Core Concepts

### Tensor Values Versus Buffers

A tensor SSA value represents a value. A memref represents storage. Bufferization
asks: which tensor values can share the same storage, and where must the
compiler allocate or copy?

For example, an operation in destination-passing style already receives a
destination tensor. If analysis proves it is safe, the result can reuse the
destination buffer. If not, the compiler inserts a new allocation and copy.

### One-Shot Bufferize

One-Shot Bufferize is the main analysis and rewrite pass. It analyzes SSA
use-def chains of tensor values, decides which operands can bufferize in place,
and rewrites tensor IR to buffer IR.

The important beginner idea is "in-place versus out-of-place":

- In-place means a tensor result reuses an existing destination buffer.
- Out-of-place means a new buffer allocation and copy are needed.

Read-after-write conflicts are the classic reason an operand cannot bufferize
in place.

### Destination-Passing Style

One-Shot Bufferize is designed around ops that carry their destination as an
operand. `tensor.insert` is the simplest example: the inserted element updates a
destination tensor and returns an updated tensor value. Many `linalg` ops also
have `outs` operands and fit this model well.

### Bufferization Boundaries

`bufferization.to_tensor` and `bufferization.to_buffer` are boundary ops. They
keep IR valid while part of a program is still tensor-based and another part is
buffer-based.

They are usually temporary. A complete bufferization pipeline should fold or
lower them away unless partial bufferization is intentionally being used.

### Ownership And Deallocation

Allocating buffers is only half the problem. A compiler also needs to deallocate
them exactly once and not before their last use.

The ownership-based deallocation pipeline inserts ownership information and
`bufferization.dealloc` ops. Later passes simplify those ops and lower them to
`memref.dealloc`, `scf.if`, helper functions, and related IR.

### Interfaces

The bufferization infrastructure is interface-driven:

- `BufferizableOpInterface` tells One-Shot Bufferize how an op reads, writes,
  aliases, and bufferizes tensor operands/results.
- `AllocationOpInterface` lets allocation-like ops build compatible dealloc and
  clone operations.
- `BufferDeallocationOpInterface` and `BufferViewFlowOpInterface` support
  ownership and alias reasoning.
- `TensorLikeType` and `BufferLikeType` type interfaces allow custom tensor-like
  and buffer-like types to participate in the machinery.

This is why many dialects can participate in bufferization without being in the
`bufferization` dialect.

## Operations

The dialect has seven operations.

| Operation | Purpose |
| --- | --- |
| `bufferization.alloc_tensor` | Creates a tensor value that bufferizes to a fresh buffer allocation. |
| `bufferization.clone` | Clones a memref-like buffer; later lowers to allocation plus copy. |
| `bufferization.dealloc` | Ownership-aware deallocation of memrefs with retained-alias checks. |
| `bufferization.dealloc_tensor` | Tensor-level deallocation marker, especially relevant for sparse storage. |
| `bufferization.materialize_in_destination` | Requires a source tensor to materialize in a destination tensor or memref. |
| `bufferization.to_buffer` | Returns the future buffer of a tensor-like value. |
| `bufferization.to_tensor` | Wraps a buffer-like value as a tensor-like value. |

### `bufferization.alloc_tensor`

`alloc_tensor` creates a tensor value that always bufferizes to a new buffer
allocation. It is not a normal tensor compute op; it is a bufferization anchor.

It is useful when you need to force a fresh use-def chain so One-Shot Bufferize
does not try to reuse some other buffer.

Important details:

- Dynamic sizes are listed as operands.
- A `copy(...)` operand can initialize the new allocation from an existing
  tensor.
- `memory_space` can control the memory space of the eventual buffer.
- `size_hint` is used for sparse tensors.
- If there is no `copy`, reading the result before writing to it observes
  undefined contents.

### `bufferization.clone`

`clone` takes a memref and returns another memref. Valid implementations may
alias the input and output or make an actual copy, so mutating either side after
the clone has undefined behavior.

The cleanup conversion `-convert-bufferization-to-memref` lowers surviving
`bufferization.clone` ops to `memref.alloc` plus `memref.copy`.

### `bufferization.dealloc`

`dealloc` is not just `memref.dealloc` with a different name. It can take:

- Memrefs to deallocate.
- Boolean ownership conditions.
- Retained memrefs that must stay alive.

It returns updated ownership conditions for retained memrefs. This lets the
deallocation pipeline model aliasing and avoid both leaks and double frees.

### `bufferization.dealloc_tensor`

`dealloc_tensor` is a tensor-world deallocation marker. In dense cases it lowers
toward `memref.dealloc` during bufferization. For sparse tensors it releases
the underlying sparse storage format.

Most users do not need to insert it when using One-Shot Bufferize plus the
deallocation pipeline.

### `bufferization.materialize_in_destination`

This op says that the data of a source tensor must materialize in a particular
destination. The destination can be another tensor or a memref.

With a tensor destination, the op returns an updated destination tensor. With a
memref destination, it has no result and writes into the buffer.

This op is useful when a transformation knows where a result should go and wants
to preserve that placement through bufferization.

### `bufferization.to_tensor`

`to_tensor` creates a tensor-like value from a buffer-like value. It is often
used at a boundary where buffer IR is made visible to tensor IR.

Important attributes:

- `restrict` promises that this is the only tensor access path to that buffer or
  an alias of it. One-Shot Bufferize requires `restrict` for supported
  `to_tensor` uses.
- `writable` marks the resulting tensor as writable. Without it, writes through
  that tensor generally bufferize out of place.

Incorrect `restrict` usage can lead to incorrect bufferization.

### `bufferization.to_buffer`

`to_buffer` returns the future buffer of a tensor-like value. It is a specialized
materialization boundary, similar in spirit to an `unrealized_conversion_cast`
but with bufferization-specific meaning.

The optional `read_only` attribute tells bufferization that aliases of the
returned buffer will not be written.

## Transformations

### Main Bufferization Passes

| Pass | Purpose |
| --- | --- |
| `-one-shot-bufferize` | Analyze tensor SSA use-def chains and rewrite bufferizable tensor IR to buffer IR. |
| `-empty-tensor-to-alloc-tensor` | Replace `tensor.empty` with `bufferization.alloc_tensor`. |
| `-eliminate-empty-tensors` | Try to remove `tensor.empty` by reusing destination subsets. |
| `-ownership-based-buffer-deallocation` | Insert ownership-aware deallocation operations for allocated buffers. |
| `-buffer-deallocation-simplification` | Use static alias information to simplify `bufferization.dealloc`. |
| `-bufferization-lower-deallocations` | Lower `bufferization.dealloc` to `memref.dealloc` plus control flow/helper code. |
| `-optimize-allocation-liveness` | Move deallocations closer to the last use of an allocation. |

### Allocation Placement And Calling Convention Passes

| Pass | Purpose |
| --- | --- |
| `-buffer-hoisting` | Move allocations upward into common dominators and out of nested regions. |
| `-buffer-loop-hoisting` | Move allocations out of loop nests. |
| `-promote-buffers-to-stack` | Convert selected heap allocations to stack allocations. |
| `-buffer-results-to-out-params` | Convert memref function results to output parameters. |
| `-drop-equivalent-buffer-results` | Remove memref results that are equivalent to function block arguments. |

### Deallocation Pipeline

The registered pipeline is:

```text
-buffer-deallocation-pipeline
```

The default pipeline expands reallocs, canonicalizes, runs ownership-based
buffer deallocation, canonicalizes again, simplifies deallocations, lowers
deallocations, runs CSE, and canonicalizes again.

Conceptually:

```text
bufferized memref IR
  -> ownership-based-buffer-deallocation
  -> buffer-deallocation-simplification
  -> bufferization-lower-deallocations
  -> memref.dealloc and helper control flow
```

One-Shot Bufferize does not deallocate the buffers it allocates. A realistic
pipeline usually runs deallocation after One-Shot Bufferize.

### Transform Dialect Operations

The bufferization extension to the Transform dialect provides:

| Transform op | Purpose |
| --- | --- |
| `transform.bufferization.buffer_loop_hoisting` | Apply buffer loop hoisting to targeted payload IR. |
| `transform.bufferization.eliminate_empty_tensors` | Apply empty tensor elimination to targeted payload IR. |
| `transform.bufferization.empty_tensor_to_alloc_tensor` | Rewrite targeted `tensor.empty` ops to `bufferization.alloc_tensor`. |
| `transform.bufferization.one_shot_bufferize` | Run configurable One-Shot Bufferize from a transform script. |

These are not operations in the `bufferization` dialect. They are Transform
dialect operations that control bufferization passes.

## Conversions And Lowering Paths

### Tensor IR To Buffer IR

The main path is:

```text
tensor/linalg/vector/etc.
  -> -one-shot-bufferize
  -> memref operations plus temporary bufferization boundaries
```

Only operations that implement `BufferizableOpInterface` are bufferized. If
`allow-unknown-ops` is enabled, unknown operations are wrapped with
`bufferization.to_buffer` and `bufferization.to_tensor` at boundaries.

### Bufferization Dialect To MemRef

The cleanup conversion is:

```text
-convert-bufferization-to-memref
```

This pass converts remaining bufferization operations into the MemRef dialect.
In the current implementation, it handles surviving `bufferization.clone` and
`bufferization.dealloc` operations. It is a cleanup pass; it does not perform
memory analysis or fix leaks.

### Deallocation Lowering

The deallocation path is:

```text
bufferization.dealloc
  -> -buffer-deallocation-simplification
  -> -bufferization-lower-deallocations
  -> memref.dealloc, scf.if, arith checks, and helper functions
```

The key idea is that `bufferization.dealloc` can express alias-aware ownership
logic. `memref.dealloc` is the final primitive deallocation operation.

## Example IR

### Tensor And Buffer Boundary

```mlir
func.func @tensor_buffer_boundary(%m: memref<4xf32>,
                                  %f: f32,
                                  %i: index) -> tensor<4xf32> {
  %t = bufferization.to_tensor %m restrict writable
    : memref<4xf32> to tensor<4xf32>
  %updated = tensor.insert %f into %t[%i] : tensor<4xf32>
  return %updated : tensor<4xf32>
}
```

The `restrict writable` boundary tells One-Shot Bufferize that `%t` is the
tensor view of `%m` and can be written through.

### Forcing A Fresh Tensor Allocation

```mlir
func.func @allocate_tensor(%d0: index,
                           %src: tensor<?xf32>) -> tensor<?xf32> {
  %scratch = bufferization.alloc_tensor(%d0) : tensor<?xf32>
  %copy = bufferization.alloc_tensor() copy(%src) : tensor<?xf32>
  return %scratch : tensor<?xf32>
}
```

`%scratch` bufferizes to a new uninitialized buffer. `%copy` bufferizes to a new
buffer initialized from `%src`.

### Materializing Into A Destination Buffer

```mlir
func.func @materialize(%src: tensor<?xf32>, %dest: memref<?xf32>) {
  bufferization.materialize_in_destination %src in restrict writable %dest
    : (tensor<?xf32>, memref<?xf32>) -> ()
  return
}
```

This says that `%src` must be materialized in `%dest`.

### Ownership-Aware Deallocation

```mlir
func.func @dealloc_example(%a: memref<4xf32>,
                           %retain: memref<4xf32>,
                           %own: i1) -> i1 {
  %new_own = bufferization.dealloc (%a : memref<4xf32>) if (%own)
    retain (%retain : memref<4xf32>)
  return %new_own : i1
}
```

The deallocation happens only if `%own` is true and `%a` is not retained through
an alias. The returned value is the updated ownership condition for `%retain`.

### Clone Cleanup

```mlir
func.func @clone_example(%m: memref<?xf32>) -> memref<?xf32> {
  %copy = bufferization.clone %m : memref<?xf32> to memref<?xf32>
  return %copy : memref<?xf32>
}
```

After `-convert-bufferization-to-memref`, this kind of clone becomes an
allocation plus `memref.copy`.

## A Practical Pipeline Shape

A typical tensor-to-buffer pipeline has this shape:

```text
canonicalize/cse
empty-tensor elimination or empty-tensor-to-alloc-tensor
one-shot-bufferize
canonicalize/cse
buffer-deallocation-pipeline
convert-bufferization-to-memref
canonicalize/cse
lower memref/control-flow/etc. toward the target
```

The exact order varies by project, but two rules are stable:

- Run the main bufferization step before expecting tensor IR to become memref
  IR.
- Run deallocation after bufferization if One-Shot Bufferize introduced
  allocations.

## Mental Model

Think of bufferization as answering three questions:

1. What buffer represents each tensor value?
2. Which tensor results can reuse an existing buffer?
3. Who owns each allocated buffer, and where is it deallocated?

The `bufferization` dialect provides the markers and boundary operations. The
interfaces and passes perform most of the analysis.

## Gotchas

- One-Shot Bufferize does not insert final deallocations. Use the deallocation
  pipeline after it.
- `tensor.empty` has no contents. It must be eliminated or turned into
  `bufferization.alloc_tensor` before it can be fully bufferized.
- `to_tensor restrict` is a promise. If the same buffer is exposed through
  another aliasing `to_tensor restrict`, bufferization can be wrong.
- `writable` affects in-place decisions. Without it, writes through a tensor
  boundary may be forced out of place.
- `bufferization.dealloc` can be expensive before simplification because it may
  encode runtime alias checks.
- Bufferization is interface-driven. If a dialect op does not implement
  `BufferizableOpInterface`, One-Shot Bufferize either fails or leaves boundary
  ops when `allow-unknown-ops` is enabled.
- Function boundary bufferization is useful but constrained. Recursive call
  graphs, external tensor-returning functions, and complex multi-block cases
  are not generally supported by the experimental path.

## Source Map

Important source files in the LLVM tree:

- `mlir/include/mlir/Dialect/Bufferization/IR/BufferizationBase.td` defines the
  dialect and dialect-level attributes.
- `mlir/include/mlir/Dialect/Bufferization/IR/BufferizationOps.td` defines the
  seven core operations.
- `mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td`
  defines the core One-Shot Bufferize interface.
- `mlir/include/mlir/Dialect/Bufferization/IR/AllocationOpInterface.td`
  defines allocation-related hooks used by deallocation and hoisting.
- `mlir/include/mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.td`
  defines tensor-like and buffer-like type interfaces.
- `mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.td` declares the
  bufferization and deallocation passes.
- `mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp` implements
  One-Shot analysis.
- `mlir/lib/Dialect/Bufferization/Transforms/Bufferize.cpp` implements common
  bufferization rewriting support.
- `mlir/lib/Dialect/Bufferization/Transforms/OwnershipBasedBufferDeallocation.cpp`
  implements ownership-based deallocation.
- `mlir/lib/Dialect/Bufferization/Pipelines/BufferizationPipelines.cpp`
  registers `buffer-deallocation-pipeline`.
- `mlir/include/mlir/Conversion/Passes.td` declares
  `convert-bufferization-to-memref`.
- `mlir/include/mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.td`
  defines the Transform dialect operations for bufferization.
- `mlir/test/Dialect/Bufferization/` contains operation and pass tests.
