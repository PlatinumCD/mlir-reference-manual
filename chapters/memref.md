# `memref` Dialect

## Beginner Summary

The `memref` dialect is MLIR's core dialect for memory references.

It defines operations for:

- Allocating and freeing memory.
- Stack allocation scopes.
- Loading and storing scalar elements.
- Copying between buffers.
- Creating views into existing buffers.
- Querying dimensions and rank.
- Representing globals.
- Describing metadata such as offset, sizes, strides, layout, and memory space.
- Lowering memory operations toward LLVM, EmitC, or SPIR-V.

For beginners, `memref` is the dialect that usually appears after tensor values
have been bufferized. Tensor IR says "this is a value." MemRef IR says "this is
a reference to storage."

## Why This Dialect Exists

MLIR needs a target-independent memory model before it lowers to target ABI
details.

The `memref` dialect provides that model.

A memref type carries more information than a raw pointer:

```text
element type
shape
dynamic sizes
layout map or strided layout
memory space
```

That information lets MLIR optimize memory accesses while still knowing enough
to lower later to:

- LLVM pointers and memref descriptors.
- C or C++ allocation and subscript code through EmitC.
- SPIR-V storage classes and access chains.
- GPU memory spaces and target-specific memory operations.

Without `memref`, a compiler would have to choose raw pointer-level details too
early. That would make shape, layout, aliasing, and boundary reasoning much
harder.

## When It Matters

The `memref` dialect matters whenever the IR is about storage instead of pure
values.

It appears in pipelines that:

- Bufferize tensor programs.
- Lower `linalg` operations to loops over buffers.
- Represent stack or heap allocation.
- Represent globals and constant buffers.
- Lower memory accesses to LLVM or EmitC.
- Map memory spaces for GPU or SPIR-V targets.
- Preserve view metadata through `memref.subview`,
  `memref.extract_strided_metadata`, `memref.collapse_shape`, and
  `memref.expand_shape`.
- Normalize layouts before affine or LLVM lowering.

A common flow is:

```text
tensor / linalg / scf
  -> bufferization
  -> memref allocation, views, loads, stores, copies
  -> memref metadata simplification and alias folding
  -> convert memref to LLVM, EmitC, or SPIR-V
```

## When To Use It

Use `memref` when the IR needs to talk about memory identity and memory effects.

Good uses include:

- Allocating a temporary buffer with `memref.alloc`.
- Allocating stack storage with `memref.alloca`.
- Reading or writing one scalar element with `memref.load` and `memref.store`.
- Passing buffers between functions after bufferization.
- Representing a slice with `memref.subview`.
- Representing a reshaped or collapsed view without copying data.
- Asking for a runtime dimension with `memref.dim`.
- Describing a global buffer with `memref.global` and `memref.get_global`.
- Making layout metadata explicit before low-level conversion.

Avoid using `memref` when the program is still best described as immutable
tensor values. Moving to `memref` introduces aliasing, ownership, lifetime, and
mutation concerns. That is useful, but it is also a different level of the
compiler.

## Core Concepts

### MemRef Types

A memref type is a shaped memory reference:

```mlir
memref<4xf32>
memref<?x?xf32>
memref<8x8xi32, strided<[8, 1]>>
memref<16xf32, affine_map<(i) -> (i floordiv 4, i mod 4)>>
```

The `?` dimensions are dynamic. Their concrete sizes are carried at runtime.

The layout can be identity, strided, or an affine map. The memory space can
identify target-specific storage such as GPU workgroup memory, SPIR-V storage
classes, or custom backend spaces.

### MemRef Values Are References

A memref SSA value is not the contents of the buffer. It is a reference to a
buffer plus metadata.

That means two memref values can refer to overlapping storage. For example,
`memref.subview` creates a new memref value that aliases the original buffer.

This is one of the biggest differences from tensor IR:

```text
tensor value: immutable shaped value
memref value: mutable reference to storage
```

### Ownership And Lifetime

`memref.alloc` allocates heap-like memory that must normally be paired with
`memref.dealloc`.

`memref.alloca` allocates stack-like memory that is automatically released when
control exits the nearest automatic allocation scope. `memref.alloca_scope`
makes that scope explicit.

`memref.realloc` may allocate a new buffer, copy from the old buffer, and free
the old storage depending on the requested size and lowering choices.

### Views And Metadata

Many MemRef ops do not move data. They only create a new view:

- `memref.subview`
- `memref.cast`
- `memref.reshape`
- `memref.reinterpret_cast`
- `memref.transpose`
- `memref.collapse_shape`
- `memref.expand_shape`
- `memref.view`
- `memref.memory_space_cast`

These operations are important because they let high-level transformations
change how memory is addressed without immediately materializing pointer
arithmetic.

### Metadata Lowering

Low-level backends eventually need explicit pointer, offset, size, and stride
values. `memref.extract_strided_metadata` exposes that metadata. The
`expand-strided-metadata` pass rewrites metadata-changing MemRef ops into
explicit pieces that later passes can analyze and lower more easily.

## Operations

The current MemRef dialect in this LLVM checkout defines 32 generated
operations.

### Allocation And Lifetime

`memref.alloc`
: Allocates a heap-like memory region described by a memref type. Dynamic
  dimensions are operands.

`memref.dealloc`
: Frees memory that was allocated by `memref.alloc`.

`memref.alloca`
: Allocates stack-like memory that is automatically released at scope exit.

`memref.alloca_scope`
: Creates an explicit scope for stack allocations.

`memref.alloca_scope.return`
: Returns values from a `memref.alloca_scope` region.

`memref.realloc`
: Changes the size of a memory region, possibly allocating, copying, and
  deallocating.

### Basic Memory Access

`memref.load`
: Reads one element from a memref at the given indices.

`memref.store`
: Writes one element to a memref at the given indices.

`memref.copy`
: Copies data from one memref to another compatible memref.

`memref.prefetch`
: Emits a prefetch hint for a memory location, with read/write, locality, and
  cache attributes.

### Atomics

`memref.atomic_rmw`
: Performs a built-in atomic read-modify-write operation at one memref element.

`memref.generic_atomic_rmw`
: Performs a custom atomic read-modify-write using a region.

`memref.atomic_yield`
: Yields the computed value from a `memref.generic_atomic_rmw` region.

### DMA

`memref.dma_start`
: Starts a non-blocking DMA transfer between source and destination memrefs
  using a tag memref.

`memref.dma_wait`
: Waits for a DMA transfer associated with a tag element to complete.

These operations are older but still part of the dialect surface. They matter
when a pipeline models explicit asynchronous memory transfers.

### Shape And Queries

`memref.dim`
: Returns the size of one dimension of a ranked memref.

`memref.rank`
: Returns the rank of a memref.

### Globals

`memref.global`
: Declares or defines a named global memref.

`memref.get_global`
: Produces the memref value for a named global.

### Views, Casts, And Metadata

`memref.cast`
: Casts between compatible memref types, often static and dynamic forms of the
  same storage.

`memref.subview`
: Creates a view into a source memref using offsets, sizes, and strides.

`memref.view`
: Creates an N-D contiguous memref view from a one-dimensional byte buffer.

`memref.reshape`
: Reinterprets a memref with a runtime shape memref. It does not copy data.

`memref.reinterpret_cast`
: Creates a memref view with explicitly provided offset, sizes, and strides.

`memref.transpose`
: Creates a metadata-only transposed strided memref.

`memref.collapse_shape`
: Produces a lower-rank view by reassociating dimensions.

`memref.expand_shape`
: Produces a higher-rank view by reassociating dimensions and output sizes.

`memref.extract_strided_metadata`
: Splits a strided memref into base buffer, offset, sizes, and strides.

`memref.extract_aligned_pointer_as_index`
: Extracts the underlying aligned pointer as an `index` value for low-level
  lowering paths.

`memref.memory_space_cast`
: Casts a memref between memory spaces while preserving shape and element type.

### Assumptions

`memref.assume_alignment`
: Attaches alignment information to a memref SSA value.

`memref.distinct_objects`
: States that a list of memrefs never alias each other and returns equivalent
  memref values carrying that assumption.

## Transformations

MemRef transformations usually simplify ownership, aliasing, shape metadata, or
layout before conversion.

### Native MemRef Passes

`memref-elide-reinterpret-cast`
: Rewrites redundant `memref.reinterpret_cast` users so the IR is easier to
  convert to EmitC.

`memref-expand`
: Legalizes selected MemRef operations into forms that are easier to convert to
  LLVM. It includes rewrites such as expanding some atomic RMW cases and
  replacing statically shaped `memref.reshape` with `memref.reinterpret_cast`.

`fold-memref-alias-ops`
: Folds aliasing view ops, such as `memref.subview`, into consumer load/store
  style operations.

`memref-emulate-wide-int`
: Emulates memory operations on too-wide integer element types by splitting
  values into supported narrower integer pieces.

`normalize-memrefs`
: Normalizes memref types with non-identity layout maps into identity-layout
  memrefs, updating function signatures, call sites, and normalizable users.

`resolve-ranked-shaped-type-result-dims`
: Resolves `memref.dim` of ranked-shaped operation results by reifying result
  dimensions from operands.

`resolve-shaped-type-result-dims`
: Similar to the ranked pass, but works with operations implementing shaped type
  inference or reification interfaces.

`reify-result-shapes`
: Reifies selected tensor result shapes, currently for tensor pad and concat
  cases, inserting casts when result types become more static.

`expand-strided-metadata`
: Expands metadata-changing MemRef ops into explicit base, offset, size, and
  stride computations. Supported ops include `memref.collapse_shape`,
  `memref.expand_shape`, `memref.extract_aligned_pointer_as_index`,
  `memref.extract_strided_metadata`, and `memref.subview`.

`expand-realloc`
: Expands `memref.realloc` into simpler allocation, copy, deallocation, and
  conditional control-flow pieces. It has an `emit-deallocs` option.

`flatten-memref`
: Flattens multi-dimensional memrefs to one-dimensional memrefs.

### Transform Dialect Hooks

The MemRef dialect exposes transform-dialect hooks for conversion and rewrite
pattern control.

Conversion type-converter hook:

- `apply_conversion_patterns.memref.memref_to_llvm_type_converter`

Rewrite pattern hooks:

- `apply_patterns.memref.alloc_to_alloca`
- `apply_patterns.memref.expand_ops`
- `apply_patterns.memref.expand_strided_metadata`
- `apply_patterns.memref.extract_address_computations`
- `apply_patterns.memref.fold_memref_alias_ops`
- `apply_patterns.memref.resolve_ranked_shaped_type_result_dims`

Concrete transform operations:

- `memref.alloca_to_global`
- `memref.multibuffer`
- `memref.erase_dead_alloc_and_stores`
- `memref.make_loop_independent`

These are useful when a transform script wants to control memory optimization
explicitly. For example, `memref.multibuffer` expands an allocation by a factor
to break loop-carried dependencies through a temporary buffer, while
`memref.alloca_to_global` can move stack allocations into a module-level global.

## Conversions And Lowering Paths

### From Bufferization

`convert-bufferization-to-memref` converts bufferization dialect operations into
MemRef operations.

For example, it can lower allocation-like bufferization operations into
`memref.alloc` and `memref.copy` forms. This is one bridge from bufferization
IR into ordinary MemRef IR.

### To LLVM

`finalize-memref-to-llvm` finalizes conversion from MemRef to the LLVM dialect.

The conversion contains direct patterns for:

- `memref.alloc`
- `memref.alloca`
- `memref.alloca_scope`
- `memref.assume_alignment`
- `memref.atomic_rmw`
- `memref.cast`
- `memref.collapse_shape`
- `memref.copy`
- `memref.dealloc`
- `memref.dim`
- `memref.distinct_objects`
- `memref.expand_shape`
- `memref.extract_aligned_pointer_as_index`
- `memref.extract_strided_metadata`
- `memref.generic_atomic_rmw`
- `memref.get_global`
- `memref.global`
- `memref.load`
- `memref.memory_space_cast`
- `memref.prefetch`
- `memref.rank`
- `memref.reinterpret_cast`
- `memref.reshape`
- `memref.store`
- `memref.subview`
- `memref.transpose`
- `memref.view`

Important options:

- `use-aligned-alloc`
- `index-bitwidth`
- `use-generic-functions`

The pass description explicitly notes that some complex MemRef operations are
not converted directly and should be prepared with `expand-strided-metadata`
first.

The generic `convert-to-llvm` driver can also use the MemRef dialect's
conversion interface when it is registered.

### To EmitC

`convert-memref-to-emitc` converts supported MemRef operations to the EmitC
dialect. It has a `lower-to-cpp` option.

The direct conversion pattern set covers:

- `memref.alloca`
- `memref.alloc`
- `memref.copy`
- `memref.dealloc`
- `memref.global`
- `memref.get_global`
- `memref.load`
- `memref.store`

The generic `convert-to-emitc` driver can also use the MemRef dialect's EmitC
conversion interface.

### To SPIR-V

`map-memref-spirv-storage-class` maps numeric MemRef memory spaces to SPIR-V
storage classes. Its `client-api` option defaults to Vulkan mappings.

`convert-memref-to-spirv` converts supported MemRef operations to the SPIR-V
dialect. Important options include:

- `bool-num-bits`
- `use-64bit-index`

The direct SPIR-V pattern set covers:

- `memref.alloca`
- `memref.alloc`
- `memref.atomic_rmw`
- `memref.dealloc`
- `memref.load`
- `memref.store`
- `memref.memory_space_cast`
- `memref.reinterpret_cast`
- `memref.cast`
- `memref.extract_aligned_pointer_as_index`

SPIR-V has stricter storage-class and type rules than generic MemRef IR, so
memory-space mapping and type legality matter more on this path.

## Example IR

### Heap Allocation, Load, Store, And Deallocation

```mlir
func.func @basic(%v: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %A = memref.alloc() : memref<4xf32>
  memref.store %v, %A[%c0] : memref<4xf32>
  %x = memref.load %A[%c0] : memref<4xf32>
  memref.dealloc %A : memref<4xf32>
  func.return %x : f32
}
```

This is the simplest owned-buffer pattern: allocate, write, read, deallocate.

### Scoped Stack Allocation

```mlir
func.func @stack_scope(%v: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %r = memref.alloca_scope -> f32 {
    %A = memref.alloca() : memref<4xf32>
    memref.store %v, %A[%c0] : memref<4xf32>
    %x = memref.load %A[%c0] : memref<4xf32>
    memref.alloca_scope.return %x : f32
  }
  func.return %r : f32
}
```

`memref.alloca` storage is released when control leaves the automatic allocation
scope.

### Subview

```mlir
func.func @views(%A: memref<8x8xf32>)
    -> memref<4x4xf32, strided<[8, 1]>> {
  %sub = memref.subview %A[0, 0] [4, 4] [1, 1]
      : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1]>>
  func.return %sub : memref<4x4xf32, strided<[8, 1]>>
}
```

`memref.subview` creates a view. It does not copy the elements.

### Metadata Extraction

```mlir
func.func @metadata(%A: memref<8x8xf32>) -> index {
  %sub = memref.subview %A[0, 0] [4, 4] [1, 1]
      : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1]>>
  %base, %offset, %size0, %size1, %stride0, %stride1 =
      memref.extract_strided_metadata %sub
      : memref<4x4xf32, strided<[8, 1]>>
        -> memref<f32>, index, index, index, index, index
  func.return %stride0 : index
}
```

This makes the view metadata explicit. Later lowering can turn that metadata
into pointer arithmetic and descriptor fields.

### Global MemRef

```mlir
memref.global "private" constant @weights : memref<4xf32> = dense<1.0>

func.func @use_global() -> f32 {
  %c0 = arith.constant 0 : index
  %g = memref.get_global @weights : memref<4xf32>
  %x = memref.load %g[%c0] : memref<4xf32>
  func.return %x : f32
}
```

`memref.global` defines the storage. `memref.get_global` produces a memref value
that can be loaded from.

## Mental Model

Think of `memref` as "pointer plus shape metadata."

The pointer tells the compiler where the storage begins. The metadata tells it
how to interpret indices into that storage.

For a ranked memref, an indexed access conceptually needs:

```text
base pointer
offset
sizes
strides
indices
element type
memory space
```

High-level MemRef IR keeps those details structured. Low-level lowering turns
them into descriptors, pointer arithmetic, target storage classes, or C-like
subscripts.

The most important beginner shift is this:

```text
Tensor transformations reason about values.
MemRef transformations reason about storage, aliases, and effects.
```

## Gotchas

Views alias their source.

`memref.subview`, `memref.cast`, `memref.reinterpret_cast`, `memref.transpose`,
`memref.collapse_shape`, and `memref.expand_shape` usually produce new memref
values that refer to the same underlying storage. Do not assume they copy data.

`memref.alloc` needs lifetime management.

Unless another pass owns deallocation, an allocated buffer should eventually
have a matching `memref.dealloc`. Missing deallocation is a memory management
bug in lowered IR.

`memref.alloca` has scope-based lifetime.

It is easier to use for temporary stack-like storage, but its lifetime is tied
to control flow. Returning or storing an escaping view of stack storage is not a
safe lowering strategy.

Layouts are part of the type.

Changing from `memref<8x8xf32>` to `memref<8x8xf32, strided<[8, 1]>>` is not
just cosmetic. It changes how indices map to memory metadata.

`memref.dim` may be dynamic or static.

For static dimensions, canonicalization can often fold the result. For dynamic
dimensions, the size is runtime metadata.

`finalize-memref-to-llvm` is late.

Run MemRef cleanup first when the IR still contains complex metadata operations.
The pass description specifically calls out `expand-strided-metadata` as a
preparation step for complex MemRef lowering.

Memory spaces are target-sensitive.

The MemRef dialect itself does not define all target memory-space semantics.
Passes such as `map-memref-spirv-storage-class` interpret them for a particular
target family.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/MemRef/IR/MemRefBase.td`
- `mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td`
- `mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp`
- `mlir/include/mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.td`
- `mlir/include/mlir/Dialect/MemRef/Transforms/Passes.td`
- `mlir/include/mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.td`
- `mlir/lib/Dialect/MemRef/Transforms/`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp`
- `mlir/lib/Conversion/MemRefToEmitC/MemRefToEmitC.cpp`
- `mlir/lib/Conversion/MemRefToSPIRV/MemRefToSPIRV.cpp`
- `mlir/lib/Conversion/BufferizationToMemRef/BufferizationToMemRef.cpp`

Generated op documentation source:

```text
mlir-tblgen --gen-op-doc -dialect=memref \
  mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td
```
