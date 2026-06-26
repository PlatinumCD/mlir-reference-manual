# IREE `pcf` Dialect

## Beginner Summary

The IREE `pcf` dialect is a parallel control-flow dialect used inside IREE
code generation.

PCF stands for parallel control flow. It is not a frontend dialect and it is not
the final target representation. It is an intermediate layer for saying:
"these workers execute this region, these values are shared through scoped
references, and these writes need well-defined synchronization before the IR
turns into ordinary control flow and memory operations."

The central idea is that parallel code needs more information than a plain loop.
An ordinary `scf.for` or `scf.forall` can describe iteration, but it does not by
itself say which hardware scope owns memory, when workers must synchronize, or
how a parallel write to a result tensor becomes a concrete buffer update. PCF
adds that missing structure.

## Why This Dialect Exists

IREE lowers tensor programs to executable code for CPUs, GPUs, and other
accelerators. During that lowering, the compiler must represent nested
parallelism, temporary buffers, thread-local or workgroup-local memory, result
updates, and synchronization.

The `pcf` dialect exists to hold those ideas before the compiler commits to a
lower-level representation.

It fills a gap between:

- high-level tensor and loop IR, such as `linalg`, `tensor`, and `scf.forall`;
- low-level memory and control-flow IR, such as `memref`, `scf`, `cf`, and
  target-specific GPU dialects.

PCF is especially useful because it separates three concerns that are often
mixed together too early:

- how work is split among workers;
- what memory is visible at a given parallel scope;
- when writes to shared data are synchronized.

That separation gives IREE room to fuse producers and consumers, bufferize
parallel results, resolve synchronization tokens, and only then lower to
concrete loops, branches, barriers, and memrefs.

## When It Matters

You will usually see `pcf` in the middle of IREE codegen, after high-level
parallel work has been exposed and before final structural lowering.

A typical path looks like this:

```text
linalg/tensor/scf
  -> tiling and distribution
  -> scf.forall or similar parallel structure
  -> pcf.generic / pcf.loop with pcf.sref values
  -> fusion and write composition
  -> resolve PCF tokens
  -> convert pcf.sref to memref
  -> lower structural PCF to scf/cf
  -> target-specific codegen
```

For a beginner, the important thing is that PCF is not a general-purpose
parallel programming dialect. It is an IREE compiler-internal dialect for
lowering parallel tensor work while preserving enough information about
parallel scopes and synchronization.

## When To Use It

Use `pcf` when you are working inside IREE codegen and need to preserve
parallel execution structure across bufferization and fusion.

Use it for:

- representing a region executed by a hardware or logical parallel scope;
- representing explicit parallel loops with scoped worker IDs;
- modeling result tensors as scoped references during bufferization;
- expressing reads and writes to slices of scoped references;
- carrying synchronization intent until a later pass resolves it;
- lowering from `scf.forall` toward ordinary `scf` and `cf`.

Do not use `pcf` as a frontend IR. A frontend should normally emit higher-level
tensor, linalg, stablehlo, torch, or input dialect IR and let IREE's pipelines
introduce PCF when it is useful.

Do not use it as final target IR either. After the PCF lowering passes run, the
remaining program should be expressed with concrete control-flow and memory
dialects.

## Core Concepts

### Parallel Scopes

PCF operations carry a scope attribute. A scope says what kind of workers are
executing the operation and how to query worker IDs, worker counts, barriers,
allocation memory space, and preferred allocation alignment.

The dialect defines a `ScopeAttrInterface` so different IREE targets or
codegen layers can provide their own scope semantics. The built-in
`#pcf.sequential` scope is the simplest case: it represents a single worker and
returns worker count `1` and worker ID `0`.

### Scoped References

The main PCF type is `!pcf.sref`, short for shaped reference.

A shaped reference is like a shaped handle to memory whose physical layout is
not yet fixed. It carries:

- a shape;
- an element type;
- an allocation scope;
- an optional synchronization scope.

This lets PCF talk about memory before it is committed to a particular
`memref` layout or memory space.

Example shape:

```mlir
!pcf.sref<?x?xf32, #pcf.sequential>
```

The type means "a ranked reference to `f32` elements with dynamic two-dimensional
shape, allocated at the sequential PCF scope." Later, `iree-pcf-convert-sref-to-
memref` turns this into concrete `memref` types.

### Synchronization Scopes

PCF also defines a `SyncScopeAttrInterface`. Synchronization scope attributes
describe what concrete synchronization state is needed for a shaped reference.

The built-in `#pcf.sync_on_return` attribute means that the shaped reference is
synchronized when control returns from workers at the same scope. In practical
terms, it preserves the idea that writes become visible after the parallel
region finishes and any required scope barrier has happened.

### Structural Parallel Operations

PCF has two main structural operations:

- `pcf.generic`: run a region over workers provided by a scope.
- `pcf.loop`: run a region over an explicit iteration count, mapped through a
  scope.

Both operations can produce tensor or memref results. Inside their bodies,
result values are represented as `pcf.sref` block arguments. Body code reads
from and writes to those references with PCF slice operations.

### Results And Tied Initial Values

PCF results can be tied to input values. This is similar in spirit to
destination-passing style: an initial tensor or memref provides the starting
contents, and the PCF operation returns the final state.

If a result is not tied to an input, the PCF operation can allocate it itself,
using dynamic size operands when the result shape has dynamic dimensions.

This is why PCF is useful during bufferization. It gives the compiler an
explicit place to reason about whether a parallel result aliases an existing
buffer or needs new storage.

## Operations

The IREE PCF dialect currently defines these operations:

| Operation | Purpose |
| --- | --- |
| `pcf.alloc` | Allocate a `pcf.sref` value with static and/or dynamic shape. |
| `pcf.generic` | Execute a region across workers from a scope. |
| `pcf.loop` | Execute a region over an explicit iteration space mapped to a scope. |
| `pcf.br.cond_return` | Branch to another block or conditionally return from the parent PCF region. |
| `pcf.write_slice` | Write a tensor, vector, or memref slice into a `pcf.sref`. |
| `pcf.read_slice` | Read a tensor or vector slice from a `pcf.sref`. |
| `pcf.get_memref` | Extract a memref view from a `pcf.sref`, intentionally breaking synchronization guarantees. |
| `pcf.yield` | Yield values from a `pcf.generic` initializer region. |
| `pcf.return` | Return from a worker region without directly yielding parent results. |

### `pcf.alloc`

`pcf.alloc` creates a `pcf.sref`.

Use it when a PCF region needs scoped temporary storage. The result type
contains the shaped reference type. Dynamic dimensions in that type must be
provided as operands.

Example:

```mlir
%ref = pcf.alloc(%m, %n)
  : !pcf.sref<?x?xf32, #pcf.sequential>
```

Later conversion turns this allocation into a concrete `memref.alloc` with
layout and memory space derived from the scope.

### `pcf.generic`

`pcf.generic` executes a region across the workers of a scope. The region
receives worker ID and worker count arguments. It may also receive `pcf.sref`
arguments corresponding to tied or allocated results.

Example shape:

```mlir
%out = pcf.generic scope(#pcf.sequential)
  execute(%ref = %init)[%id: index, %count: index]
       : (!pcf.sref<?xf32, #pcf.sequential>) -> (tensor<?xf32>) {
  pcf.return
}
```

The operation is useful when the number of workers comes from the scope rather
than from an explicit loop bound. It also supports an optional `initialize`
region. The initializer runs once and can yield values that become leading
block arguments in the execute region.

### `pcf.loop`

`pcf.loop` executes a region for each point in an explicit iteration space. The
`count` operands define that space. The scope decides how those iterations are
distributed across available workers.

Example shape:

```mlir
%out = pcf.loop scope(#pcf.sequential) count(%n)
  execute(%ref = %init)[%i: index]
       : (!pcf.sref<?xf32, #pcf.sequential>) -> (tensor<?xf32>) {
  pcf.return
}
```

Use `pcf.loop` when the IR still needs a structured loop-like operation but
also needs PCF's scoped memory and synchronization model.

### `pcf.br.cond_return`

`pcf.br.cond_return` is a control-flow helper used inside `pcf.generic`. It
branches to a destination block when control should continue, or returns from
the parent PCF region when the condition requests it.

This matters for lowering because PCF regions may need structured early exit
semantics before they become ordinary `cf.cond_br` and return blocks.

### `pcf.write_slice`

`pcf.write_slice` writes a shaped value into a slice of a `pcf.sref`.

The source may be a ranked tensor, vector, or memref. The destination is a
`pcf.sref`. Like MLIR slice operations in other dialects, it uses offsets,
sizes, and strides. These can be static, dynamic, or mixed.

Conceptually:

```mlir
pcf.write_slice %tile into %ref[%offset] [%size] [1]
  : tensor<16xf32> into !pcf.sref<?xf32, #pcf.sequential>
```

This is how body code publishes a computed tile into the parent parallel
result.

### `pcf.read_slice`

`pcf.read_slice` reads a tensor or vector slice from a `pcf.sref`.

It is the inverse companion to `pcf.write_slice`: it provides a shaped value
that can be used by vector, linalg, arithmetic, or other code inside the PCF
region. When the result is a vector and the requested sizes are smaller than
the vector type, out-of-bounds vector elements are undefined.

### `pcf.get_memref`

`pcf.get_memref` extracts a memref view from a `pcf.sref`.

This is an escape hatch. The source comments are explicit that it breaks the
synchronization guarantees of the source shaped reference. Use it only when
the later conversion and codegen path can reason about the resulting memref
view.

The returned memref is expected to have a maximally dynamic layout and no
memory space before conversion. The conversion pass fills in the real layout
and memory space.

### `pcf.yield`

`pcf.yield` yields values from the initializer region of `pcf.generic`. Those
values become leading arguments in the execute region.

It is not the operation that produces the final results of `pcf.generic`; it is
only for initializer-to-execute data flow.

### `pcf.return`

`pcf.return` returns from a worker region in `pcf.generic` or `pcf.loop`.

It takes no operands. The parent operation is responsible for final result
state through tied or allocated `pcf.sref` values. This is why PCF writes use
`pcf.write_slice` rather than returning tensor values directly from each
worker.

## Attributes And Interfaces

PCF's attributes are small, but they are central to the dialect.

| Attribute or interface | Meaning |
| --- | --- |
| `#pcf.sequential` | A scope with one worker. It is useful for tests and sequentialized lowering paths. |
| `#pcf.test_scope` | A test-only scope attribute whose interface methods intentionally fail if queried. |
| `#pcf.sync_on_return` | A sync scope saying the reference is synchronized when the worker returns at the same scope. |
| `ScopeAttrInterface` | Lets scope attributes provide worker IDs, worker counts, barriers, memory spaces, and alignments. |
| `SyncScopeAttrInterface` | Lets sync attributes expand to concrete state and enqueue synchronization behavior for writes. |

The important beginner lesson is that PCF does not hard-code one hardware
scope. It asks scope attributes how to materialize worker IDs, counts, barriers,
and memory spaces. That is what lets the same structural dialect serve CPU,
GPU, and accelerator lowering paths.

## Transformations And Conversions

PCF has both transformations that optimize PCF IR and conversions that lower it
away.

### `iree-pcf-fuse-consumers`

This pass fuses consumers of `pcf.generic` and `pcf.loop` results back into the
PCF region.

It looks for consumers that implement MLIR's `TilingInterface`, and it also
handles `tensor.extract_slice`. The transformed IR clones or tiles the consumer
inside the PCF region and replaces the external use with new PCF writes. This
keeps producer and consumer computation close together, which is important for
tile-level codegen.

### `iree-pcf-fuse-producers`

This pass fuses destination-passing-style producers into `pcf.generic` or
`pcf.loop` through tied init arguments.

For example, a `linalg.fill` that initializes a tensor before PCF can often be
folded into the PCF region. The pass replaces `pcf.read_slice` uses with tiled
producer computations and updates the PCF init value to the producer's own DPS
init.

### `iree-pcf-fuse-pcf-writes`

This pass composes `pcf.write_slice` operations with nested `scf.forall`
producers.

Instead of writing the result of an entire inner parallel operation and then
writing that into a PCF destination, it moves the PCF write into the inner
parallel body and composes the offsets, sizes, and strides. This makes later
lowering see the write at the right granularity.

### `iree-pcf-resolve-tokens`

This pass resolves synchronization scopes attached to `pcf.sref` types.

It runs before `iree-pcf-convert-sref-to-memref`. A shaped reference with a sync
scope can expand into a shaped reference without that sync scope plus any
concrete state required by the synchronization attribute. For
`#pcf.sync_on_return`, the pass marks the parent `pcf.generic` or `pcf.loop`
with `sync_on_return`, so structural lowering knows to insert a barrier if the
scope can provide one.

### `iree-pcf-convert-sref-to-memref`

This is the main memory conversion pass.

It converts `pcf.sref` types to concrete `memref` types. It also lowers PCF
memory operations:

- `pcf.alloc` becomes `memref.alloc`;
- `pcf.write_slice` becomes subview/copy or vector transfer write logic;
- `pcf.read_slice` becomes subview plus vector transfer read logic;
- `pcf.get_memref` becomes a concrete memref view;
- PCF region arguments are rewritten from shaped refs to tied memref values.

After this pass, PCF's abstract shaped references should be gone.

### `iree-pcf-lower-structural-pcf`

This is the final structural lowering pass for the dialect.

It lowers:

- `pcf.generic` to `scf.execute_region`;
- `pcf.loop` to `scf.forall`;
- `pcf.return` to `scf.yield` or `scf.forall.in_parallel`, depending on parent;
- `pcf.br.cond_return` to `cf.cond_br` and a return block.

The pass expects PCF operations to no longer have tied results, because
`iree-pcf-convert-sref-to-memref` should already have handled memory results.
If `sync_on_return` is set, the pass asks the scope to insert a barrier after
the lowered operation.

### Test-Only Conversion Passes

The source tree also defines test passes:

- `iree-pcf-test-convert-forall-to-loops`;
- `iree-pcf-test-fold-forall-into-pcf-loop`;
- `iree-pcf-test-convert-forall-to-generic-nest`.

These expose reusable conversion helpers for tests and development. They are
useful for understanding the intended lowering patterns from `scf.forall` to
PCF, but they should not be confused with stable user-facing pipeline entry
points.

## Conversions In The Pipeline

A useful way to remember PCF lowering is:

```text
scf.forall / tensor parallel writes
  -> pcf.loop or pcf.generic
  -> pcf.sref values for scoped result storage
  -> token resolution
  -> memref conversion
  -> structural lowering to scf/cf
```

The order matters. If structural PCF were lowered before tokens and shaped
refs were resolved, the compiler would lose the structured information needed
to decide where synchronization and memory conversion belong.

## What It Implies

Seeing `pcf` in IR implies that the program is in an IREE-specific codegen
stage. The compiler is no longer just optimizing abstract tensors, but it has
not yet fully committed to ordinary loops and concrete buffers.

It also implies that result values may be represented indirectly through
`pcf.sref` block arguments. Beginners often expect region terminators to return
results directly. PCF usually does not work that way. Workers write slices into
scoped references, and the parent operation snapshots the final state.

Finally, PCF implies that synchronization is still a first-class concern. A
plain `memref` cannot say "this buffer is synchronized when all workers of this
scope return." A `pcf.sref` can carry that information until the relevant pass
turns it into concrete barrier or memory behavior.

## Common Mistakes

Do not treat `pcf.sref` as just another spelling of `memref`. It is a temporary
scoped reference with unresolved layout, memory space, and synchronization.

Do not expect `pcf.return` to carry values. Parallel results are expressed
through writes to result references.

Do not lower structural PCF too early. Token resolution and shaped-reference
conversion need to happen first.

Do not assume every scope means GPU threads. The dialect is designed around an
interface so different scopes can model sequential execution, CPU-style
parallelism, GPU workgroups, or other accelerator levels.

## Source Map

| Topic | Source path |
| --- | --- |
| Dialect, `pcf.sref`, and attributes | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFBase.td` |
| Operations | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td` |
| Scope and sync interfaces | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.td` |
| Operation implementation | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp` |
| Pass declarations | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.td` |
| Reusable transform helpers | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h` |
| Token resolution | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/ResolveTokens.cpp` |
| Shaped-reference conversion | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/ConvertSRefToMemRef.cpp` |
| Structural lowering | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/LowerStructuralPCF.cpp` |
| Bufferization external models | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/BufferizationExternalModels.cpp` |
| Tests | `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test` and `IREE/iree/compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test` |
