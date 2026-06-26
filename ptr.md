# Ptr Dialect

## Beginner Summary

The `ptr` dialect gives MLIR an opaque pointer type and pointer operations.

Its main type is `!ptr.ptr<memory-space>`. This represents a pointer-like
handle to memory without baking in an element type. The dialect also provides
operations for pointer constants, loads, stores, pointer arithmetic,
masked/vector memory access, and conversions between opaque pointers and other
pointer-like types such as `memref`.

Think of `ptr` as a target-independent pointer vocabulary that sits between
higher-level memory abstractions and lower-level LLVM pointer operations.

## Why This Dialect Exists

MLIR has several ways to talk about memory:

- `memref` describes shaped memory with layout metadata.
- `llvm.ptr` is the LLVM dialect pointer type near LLVM lowering.
- GPU and target dialects carry target-specific address spaces.

The `ptr` dialect provides a common opaque pointer layer. It can represent a
raw pointer value while still keeping memory-space information, data-layout
queries, and metadata conversions explicit.

This is useful because pointer-like objects often have two parts:

- The actual address.
- Metadata needed to reconstruct a richer object, such as a memref descriptor.

The dialect separates those ideas with `ptr.to_ptr`, `ptr.get_metadata`, and
`ptr.from_ptr`.

## When It Matters

The `ptr` dialect matters in pipelines that need to manipulate pointer values
without immediately committing to LLVM IR.

It appears around:

- Lowering memrefs or other pointer-like objects.
- Target-independent pointer arithmetic.
- Address constants and null pointers.
- Memory-space-aware loads and stores.
- Masked loads/stores and gather/scatter operations.
- Data-layout-driven pointer sizes, alignments, and type offsets.
- Conversion paths that eventually produce LLVM pointer operations.

It is especially important when a compiler wants an opaque pointer type but
still wants memory space attributes and pointer-like metadata to remain visible.

## When To Use It

Use `ptr` when your IR needs explicit pointer values.

Use it for:

- Representing null or raw-address pointer constants.
- Performing byte-based pointer arithmetic.
- Loading from or storing to an opaque pointer.
- Computing pointer differences.
- Modeling masked loads/stores and gather/scatter memory operations.
- Converting a pointer-like object to an opaque pointer.
- Reconstructing a pointer-like object from a pointer plus metadata.
- Querying the byte offset of one element of a type.

Do not use `ptr` as a replacement for `memref` when shape, layout, and bounds
are still semantically important. Use `memref` while the compiler benefits from
structured memory information, and lower toward `ptr` when raw pointer behavior
is the useful abstraction.

## Core Concepts

### Opaque Pointers

The main type is:

```mlir
!ptr.ptr<#ptr.generic_space>
```

The pointer is opaque. It has a memory space, but it does not have an element
type. Loads and stores specify the value type at the operation.

Example:

```mlir
%x = ptr.load %p : !ptr.ptr<#ptr.generic_space> -> f32
```

The pointer says where to load from. The operation says the loaded value type is
`f32`.

### Memory Spaces

A pointer is parameterized by a memory-space attribute.

The built-in generic memory space is:

```mlir
#ptr.generic_space
```

Other dialects can provide memory-space attributes by implementing
`MemorySpaceAttrInterface`. For example, LLVM address spaces can be used:

```mlir
!ptr.ptr<#llvm.address_space<1>>
```

Memory spaces define whether loads, stores, atomics, address-space casts, and
pointer/integer casts are valid.

### Pointer Metadata

Some pointer-like types need metadata in addition to the raw pointer.

For example, a memref descriptor includes shape, stride, offset, and allocation
information. The Ptr dialect represents that metadata with:

```mlir
!ptr.ptr_metadata<memref<?xf32, #ptr.generic_space>>
```

The usual flow is:

```text
memref-like value
  -> ptr.to_ptr       extracts the raw pointer
  -> ptr.get_metadata extracts metadata
  -> ptr.from_ptr     rebuilds the memref-like value
```

### Byte-Based Pointer Arithmetic

`ptr.ptr_add` adds an integer-like offset to a pointer. The offset is in bytes.

This is different from LLVM GEP syntax, where indices are scaled by an element
type. In Ptr, if you want to move by one `f32`, compute the byte offset with
`ptr.type_offset f32`.

### Data Layout

The `ptr` type implements data-layout interfaces. Pointer size, ABI alignment,
preferred alignment, and index bitwidth can be described with `#ptr.spec`.

Example:

```mlir
#ptr.spec<size = 64, abi = 64, preferred = 128, index = 64>
```

Sizes and alignments are stored in bits and must be divisible by 8.

## Types And Attributes

| Type or attribute | Meaning |
| --- | --- |
| `!ptr.ptr<#ptr.generic_space>` | Opaque pointer in the generic memory space. |
| `!ptr.ptr_metadata<T>` | Opaque metadata needed to reconstruct pointer-like type `T`. |
| `#ptr.generic_space` | Generic memory-space attribute. |
| `#ptr.null` | Null pointer attribute for `ptr.constant`. |
| `#ptr.address<...>` | Raw byte address attribute for `ptr.constant`. |
| `#ptr.spec<...>` | Data-layout specification for pointer size, alignment, and index width. |

The dialect also defines enum properties for:

- Atomic ordering: `not_atomic`, `unordered`, `monotonic`, `acquire`,
  `release`, `acq_rel`, and `seq_cst`.
- Pointer add flags: `none`, `nusw`, `nuw`, and `inbounds`.
- Pointer difference flags: `none`, `nuw`, and `nsw`.

## Operations

The dialect has 13 operations.

| Operation | Purpose |
| --- | --- |
| `ptr.constant` | Creates a null or raw-address pointer constant. |
| `ptr.load` | Loads a value from an opaque pointer. |
| `ptr.store` | Stores a value through an opaque pointer. |
| `ptr.masked_load` | Conditionally loads a shaped value using a mask and passthrough. |
| `ptr.masked_store` | Conditionally stores shaped values using a mask. |
| `ptr.gather` | Loads from a shaped collection of pointers using a mask. |
| `ptr.scatter` | Stores to a shaped collection of pointers using a mask. |
| `ptr.ptr_add` | Adds a byte offset to one or more pointers. |
| `ptr.ptr_diff` | Computes the byte difference between pointers. |
| `ptr.type_offset` | Computes the byte offset of one element of a type. |
| `ptr.to_ptr` | Extracts an opaque pointer from a pointer-like value. |
| `ptr.get_metadata` | Extracts pointer metadata from a pointer-like value. |
| `ptr.from_ptr` | Reconstructs a pointer-like value from a pointer and optional metadata. |

### Constants

`ptr.constant` materializes pointer constants.

```mlir
%null = ptr.constant #ptr.null : !ptr.ptr<#ptr.generic_space>
%addr = ptr.constant #ptr.address<0x1000> : !ptr.ptr<#ptr.generic_space>
```

The attribute is typed by the result pointer type.

### Loads And Stores

`ptr.load` and `ptr.store` are scalar memory operations.

```mlir
%x = ptr.load %p : !ptr.ptr<#ptr.generic_space> -> f32
ptr.store %x, %p : f32, !ptr.ptr<#ptr.generic_space>
```

They support memory modifiers:

- `volatile`
- `nontemporal`
- `invariant` on loads
- `invariant_group`
- `alignment = ...`
- `atomic ...` with optional `syncscope("...")`

Atomic loads and stores require explicit alignment and support only valid atomic
value types for the memory space.

### Masked And Vectorized Memory Ops

The dialect supports shaped memory access:

- `ptr.masked_load` uses one base pointer plus a mask and passthrough value.
- `ptr.masked_store` uses one base pointer plus a shaped value and mask.
- `ptr.gather` loads from a shaped collection of pointers.
- `ptr.scatter` stores to a shaped collection of pointers.

The mask shape must match the result or value shape. Gather and scatter also
require pointer and value shapes to be compatible.

### Pointer Arithmetic

`ptr.ptr_add` supports scalar and shaped operands.

```mlir
%elem_size = ptr.type_offset f32 : index
%q = ptr.ptr_add inbounds %p, %elem_size
    : !ptr.ptr<#ptr.generic_space>, index
```

The offset is byte-based. Flags communicate no-wrap or in-bounds assumptions:

| Flag | Meaning |
| --- | --- |
| `none` | No additional assumption. |
| `nusw` | No unsigned signed wrap style flag used for LLVM GEP lowering. |
| `nuw` | No unsigned wrap. |
| `inbounds` | In-bounds pointer arithmetic assumption. |

`ptr.ptr_diff` computes a byte difference and returns an integer-like value.
Its flags are `none`, `nuw`, and `nsw`.

### Pointer-Like Casts

`ptr.to_ptr` extracts the raw pointer from another pointer-like value:

```mlir
%ptr = ptr.to_ptr %m
    : memref<?xf32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
```

`ptr.get_metadata` extracts the metadata:

```mlir
%md = ptr.get_metadata %m : memref<?xf32, #ptr.generic_space>
```

`ptr.from_ptr` reconstructs the pointer-like value:

```mlir
%m2 = ptr.from_ptr %ptr metadata %md
    : !ptr.ptr<#ptr.generic_space> -> memref<?xf32, #ptr.generic_space>
```

These operations are pure casts. They require compatible memory spaces.

## Transformations

The Ptr dialect has operation-local folding and canonicalization.

Important folds include:

- `ptr.constant` folds to its typed attribute.
- `ptr.ptr_add` with zero offset folds back to the original pointer.
- `ptr.to_ptr` and `ptr.from_ptr` chains fold when the pointer value and
  metadata relationship is provably preserved.
- Cast chains involving memrefs can fold away when the metadata comes from the
  same original pointer-like value.

The metadata rule is deliberately conservative. A raw pointer plus arbitrary
metadata is not always equivalent to the original pointer-like object.

## Conversions And Lowering Paths

Ptr-to-LLVM lowering is exposed through the generic `convert-to-llvm` pass by a
Ptr dialect conversion interface. There is no standalone `convert-ptr-to-llvm`
pass in the local pass registry inspected for this chapter.

The current Ptr-to-LLVM conversion patterns cover:

| Ptr construct | LLVM lowering |
| --- | --- |
| `!ptr.ptr<#ptr.generic_space>` | `!llvm.ptr` in address space 0. |
| `ptr.constant #ptr.null` | LLVM zero/null pointer. |
| `ptr.constant #ptr.address<...>` | Integer constant plus `llvm.inttoptr`. |
| `ptr.ptr_add` | `llvm.getelementptr` using `i8` byte offsets. |
| `ptr.type_offset` | Zero pointer GEP by one element plus `llvm.ptrtoint`. |
| `ptr.to_ptr` on memrefs | Extracts the aligned pointer from the memref descriptor. |
| `ptr.get_metadata` on memrefs | Builds a compact LLVM struct containing memref metadata. |
| `ptr.from_ptr` to memrefs | Reconstructs a memref descriptor from pointer plus metadata. |

The source inspected here does not provide dedicated Ptr-to-LLVM patterns for
every Ptr memory operation such as `ptr.load`, `ptr.store`, `ptr.gather`, or
`ptr.scatter`. Those operations still define dialect semantics, but a complete
lowering pipeline must ensure they are handled by the appropriate conversion or
legalization path for the target.

## Example IR

### Null Pointer And Byte Offset

```mlir
func.func @offset(%p: !ptr.ptr<#ptr.generic_space>, %i: index)
    -> !ptr.ptr<#ptr.generic_space> {
  %null = ptr.constant #ptr.null : !ptr.ptr<#ptr.generic_space>
  %elem = ptr.type_offset f32 : index
  %bytes = index.mul %i, %elem
  %q = ptr.ptr_add inbounds %p, %bytes
      : !ptr.ptr<#ptr.generic_space>, index
  return %q : !ptr.ptr<#ptr.generic_space>
}
```

### Load And Store

```mlir
func.func @load_store(%p: !ptr.ptr<#ptr.generic_space>, %v: f32) -> f32 {
  ptr.store %v, %p : f32, !ptr.ptr<#ptr.generic_space>
  %x = ptr.load %p : !ptr.ptr<#ptr.generic_space> -> f32
  return %x : f32
}
```

### Masked Load

```mlir
func.func @masked(%p: !ptr.ptr<#ptr.generic_space>,
                  %mask: vector<4xi1>,
                  %passthrough: vector<4xf32>) -> vector<4xf32> {
  %x = ptr.masked_load %p, %mask, %passthrough alignment = 16
      : !ptr.ptr<#ptr.generic_space> -> vector<4xf32>
  return %x : vector<4xf32>
}
```

### Memref Pointer And Metadata Round Trip

```mlir
func.func @round_trip(%m: memref<?xf32, #ptr.generic_space>)
    -> memref<?xf32, #ptr.generic_space> {
  %p = ptr.to_ptr %m
      : memref<?xf32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  %md = ptr.get_metadata %m : memref<?xf32, #ptr.generic_space>
  %r = ptr.from_ptr %p metadata %md
      : !ptr.ptr<#ptr.generic_space> -> memref<?xf32, #ptr.generic_space>
  return %r : memref<?xf32, #ptr.generic_space>
}
```

## Mental Model

The easiest mental model is:

- `memref` is structured memory with shape and layout.
- `ptr` is a raw address plus a memory-space contract.
- `ptr_metadata` is the extra information needed to rebuild structured memory.
- `llvm.ptr` is the final LLVM pointer representation after lowering.

So a lowering can peel a memref into:

```text
raw pointer: ptr.to_ptr
metadata:    ptr.get_metadata
```

and later rebuild it with:

```text
ptr.from_ptr
```

## Gotchas

### Pointer Arithmetic Is In Bytes

`ptr.ptr_add` offsets are byte offsets. Use `ptr.type_offset` if you want to
advance by one element of a type.

### `ptr` Has No Element Type

The pointer type is opaque. The element type belongs to the load, store, or
type-offset operation.

### Metadata Is Not Optional For Every Reconstruction

Some pointer-like types need metadata. Rebuilding a memref from a pointer
without correct metadata can be invalid or semantically wrong.

### Memory Spaces Define Legality

The memory-space attribute can decide whether a load, store, atomic operation,
or cast is legal. `#ptr.generic_space` is permissive, but target memory spaces
may not be.

### Conversion Is Partial

Do not assume every Ptr operation disappears after `convert-to-llvm`. Check the
target pipeline. In the inspected source, the Ptr conversion interface handles
types, constants, pointer arithmetic, type offsets, and memref pointer/metadata
casts, but not every memory operation.

## Source Map

Use these source files when you want to inspect or update the dialect:

| Area | File |
| --- | --- |
| Dialect and types | `mlir/include/mlir/Dialect/Ptr/IR/PtrDialect.td` |
| Operations | `mlir/include/mlir/Dialect/Ptr/IR/PtrOps.td` |
| Attributes | `mlir/include/mlir/Dialect/Ptr/IR/PtrAttrDefs.td` |
| Enums | `mlir/include/mlir/Dialect/Ptr/IR/PtrEnums.td` |
| Memory-space interface | `mlir/include/mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td` |
| Dialect implementation | `mlir/lib/Dialect/Ptr/IR/PtrDialect.cpp` |
| Type implementation | `mlir/lib/Dialect/Ptr/IR/PtrTypes.cpp` |
| Attribute implementation | `mlir/lib/Dialect/Ptr/IR/PtrAttrs.cpp` |
| LLVM conversion | `mlir/lib/Conversion/PtrToLLVM/PtrToLLVM.cpp` |
| Dialect tests | `mlir/test/Dialect/Ptr/` |
| LLVM conversion tests | `mlir/test/Conversion/PtrToLLVM/` |
