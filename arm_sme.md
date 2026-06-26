# `arm_sme` Dialect

## Beginner Summary

`arm_sme` is an Arm CPU target dialect for the Arm Scalable Matrix Extension
and its ZA matrix storage.

The dialect is used when a compiler wants to represent operations that fit the
Arm SME programming model: scalable vectors, scalable matrix tiles, tile slices,
and matrix outer-product instructions. It is more target-specific than
`linalg` or `vector`, but more structured than raw LLVM IR intrinsics.

You normally do not start a program in `arm_sme`. A typical path starts with
portable IR such as `linalg`, `vector`, `memref`, `scf`, and `arith`, then
lowers matching vector operations to `arm_sme` once the compiler knows it is
targeting an Arm CPU with SME support.

The important idea is simple: `arm_sme` is the bridge from portable matrix and
vector IR to Arm SME hardware tiles and AArch64 SME intrinsics.

## Why This Dialect Exists

Generic MLIR dialects intentionally avoid committing to one hardware target.
That is useful for optimization, but Arm SME has concepts that generic IR
cannot name precisely:

- ZA, the architectural matrix tile storage used by SME.
- Streaming SVE mode, controlled by the processor state bit `PSTATE.SM`.
- Scalable 2-D tile values such as `vector<[4]x[4]xf32>`.
- Horizontal and vertical tile slices.
- Matrix outer-product instructions such as MOPA and MOPS.
- Tile allocation, where virtual tile SSA values are assigned hardware tile
  IDs before lowering to intrinsics.

The `arm_sme` dialect makes those concepts explicit. That lets MLIR optimize
and verify SME-shaped code before the final conversion to LLVM-compatible
intrinsic operations.

## When It Matters

`arm_sme` matters when a compiler is lowering matrix or vector code for Arm
SME-capable AArch64 targets.

You are likely to see it when:

- lowering `linalg.matmul` or `vector.outerproduct` toward Arm SME outer
  product instructions;
- converting `vector.transfer_read`, `vector.transfer_write`, `vector.load`,
  or `vector.store` on SME-sized scalable tiles;
- legalizing scalable vectors that are larger than one SME tile;
- lowering full tile loads and stores into loops over tile slices;
- attaching function attributes that describe streaming mode and ZA usage;
- translating to LLVM IR that contains AArch64 SME intrinsics.

It is not a general matrix dialect. Use `linalg` for target-independent matrix
algorithms and `vector` for target-independent vectorization. Use `arm_sme`
when you have crossed into the Arm SME lowering path.

## When To Use It

Use `arm_sme` when all of these are true:

- the target is AArch64 with the required SME feature set;
- the computation maps naturally to SME tiles, tile slices, or outer products;
- the IR is late enough in the pipeline for target-specific choices;
- you want the final lowering to use Arm SME LLVM intrinsic operations.

Avoid using it as an early frontend dialect. Early IR should usually preserve
portable structure in `linalg`, `tensor`, `memref`, `scf`, `arith`, and
`vector` so generic transformations still apply.

## Core Concepts

### SME Is About Scalable Matrix Tiles

Arm SME extends the Arm scalable-vector model with matrix-oriented storage and
instructions. The key storage is ZA, which can be viewed as a collection of
hardware tiles. In MLIR, `arm_sme` models these tiles with 2-D scalable vector
types:

```mlir
vector<[16]x[16]xi8>
vector<[8]x[8]xi16>
vector<[4]x[4]xi32>
vector<[2]x[2]xi64>
vector<[1]x[1]xi128>
vector<[8]x[8]xf16>
vector<[8]x[8]xbf16>
vector<[4]x[4]xf32>
vector<[2]x[2]xf64>
```

The square brackets mean the dimensions are scalable. The physical number of
lanes depends on the runtime vector length, not a fixed compile-time width.

### SVE Vectors Feed SME Tiles

Many `arm_sme` operations take 1-D scalable vectors:

```mlir
vector<[16]xi8>
vector<[8]xi16>
vector<[4]xi32>
vector<[2]xi64>
vector<[1]xi128>
vector<[8]xf16>
vector<[8]xbf16>
vector<[4]xf32>
vector<[2]xf64>
```

Predicate operands use matching scalable `i1` vectors:

```mlir
vector<[16]xi1>
vector<[8]xi1>
vector<[4]xi1>
vector<[2]xi1>
vector<[1]xi1>
```

This is why `arm_sme` sits near `vector` and `arm_sve` in lowering pipelines:
SME tile operations are built out of scalable-vector rows, columns, masks, and
outer products.

### Virtual Tiles And Tile IDs

Most high-level `arm_sme` tile operations implement `ArmSMETileOpInterface`.
They operate on virtual tile SSA values first. Later, tile allocation assigns a
`tile_id` attribute that selects which ZA tile the operation uses.

Tile ID availability depends on element width:

| Tile type | Possible tile IDs |
|---|---|
| `vector<[16]x[16]xi8>` | `0` |
| `vector<[8]x[8]xi16>`, `vector<[8]x[8]xf16>`, `vector<[8]x[8]xbf16>` | `0`, `1` |
| `vector<[4]x[4]xi32>`, `vector<[4]x[4]xf32>` | `0` through `3` |
| `vector<[2]x[2]xi64>`, `vector<[2]x[2]xf64>` | `0` through `7` |
| `vector<[1]x[1]xi128>` | `0` through `15` |

The programmer usually does not write `tile_id` by hand. The
`convert-arm-sme-to-llvm` pass runs tile allocation before lowering tile ops to
intrinsic ops.

### Tile Slices

SME tiles can be accessed one scalable vector slice at a time. A slice can be
horizontal or vertical:

```mlir
layout<horizontal>
layout<vertical>
```

The default is horizontal. Full tile loads and stores can be lowered into loops
over slice loads and stores.

### Outer Products

The central compute operation is an outer product. `arm_sme.outerproduct` is
the generic form. It consumes two 1-D scalable vectors and produces a 2-D SME
tile. It can optionally accumulate into an existing tile and optionally use
operand masks.

SME also has fused widening outer-product instructions. The dialect represents
them with operations such as `arm_sme.fmopa_2way`,
`arm_sme.smopa_4way`, and `arm_sme.usmops_4way`.

In the names:

- `mopa` means outer product and accumulate.
- `mops` means outer product and subtract.
- `f` means floating point.
- `s` means signed integer.
- `u` means unsigned integer.
- `sum` means signed by unsigned.
- `usm` means unsigned by signed.
- `2way` and `4way` describe how many widened outer products are fused.

### Streaming Mode And ZA Mode

SME code may require Arm streaming SVE mode. The `enable-arm-streaming` pass
annotates `func.func` operations with attributes that describe how streaming
mode and ZA storage are managed.

The streaming mode choices are:

- `disabled`
- `streaming`
- `streaming-locally`
- `streaming-compatible`

The ZA mode choices are:

- `disabled`
- `new-za`
- `in-za`
- `out-za`
- `inout-za`
- `preserves-za`

These are ABI-level choices. They affect how callers and callees agree about
`PSTATE.SM` and ZA state, so they should be introduced deliberately and late in
the target lowering path.

## Operations

The `arm_sme` dialect defines 25 regular operations and 43 intrinsic wrapper
operations.

### Tile State Operations

| Operation | Purpose |
|---|---|
| `arm_sme.get_tile` | Creates a virtual tile with undefined contents. |
| `arm_sme.zero` | Creates a zero-initialized virtual tile. |
| `arm_sme.copy_tile` | Copies a virtual tile SSA value, mainly to normalize IR before tile allocation. |
| `arm_sme.streaming_vl` | Returns the streaming vector length as an `index` for a selected element size. |

`arm_sme.get_tile` is useful when an operation needs an existing tile value but
the contents are not meaningful yet.

```mlir
%tile = arm_sme.get_tile : vector<[4]x[4]xf32>
```

`arm_sme.zero` creates a tile whose contents are known to be zero.

```mlir
%tile = arm_sme.zero : vector<[4]x[4]xf32>
```

`arm_sme.copy_tile` preserves tile dataflow while giving tile allocation a
separate SSA value to reason about.

```mlir
%copy = arm_sme.copy_tile %tile : vector<[4]x[4]xf32>
```

`arm_sme.streaming_vl` queries the streaming vector length for an element size.

```mlir
%svl = arm_sme.streaming_vl <word>
```

The element-size attribute is one of `byte`, `half`, `word`, or `double`.

### Full Tile Memory Operations

| Operation | Purpose |
|---|---|
| `arm_sme.tile_load` | Loads a full 2-D SME tile from a 2-D memref. |
| `arm_sme.tile_store` | Stores a full 2-D SME tile to a 2-D memref. |

These are structured operations. They describe a full tile transfer, but they
can be lowered to loops over tile slices by `convert-arm-sme-to-scf`.

```mlir
%tile = arm_sme.tile_load %src[%i, %j]
  : memref<?x?xf32>, vector<[4]x[4]xf32>

arm_sme.tile_store %tile, %dst[%i, %j]
  : memref<?x?xf32>, vector<[4]x[4]xf32>
```

Both operations accept an optional `layout<horizontal>` or `layout<vertical>`.
`arm_sme.tile_load` can also take padding and a mask together. `arm_sme.tile_store`
can take a mask.

### Tile Slice Operations

| Operation | Purpose |
|---|---|
| `arm_sme.load_tile_slice` | Loads one scalable vector slice from memory into a tile. |
| `arm_sme.store_tile_slice` | Stores one scalable vector slice from a tile to memory. |
| `arm_sme.insert_tile_slice` | Inserts a 1-D scalable vector into a tile slice. |
| `arm_sme.extract_tile_slice` | Extracts a 1-D scalable vector from a tile slice. |

Tile-slice ops are the lower-level form of tile memory and tile manipulation.
They make the slice index, predicate, and layout explicit.

```mlir
%updated = arm_sme.insert_tile_slice %row, %tile[%k]
  : vector<[4]xf32> into vector<[4]x[4]xf32>

%row2 = arm_sme.extract_tile_slice %updated[%k]
  : vector<[4]xf32> from vector<[4]x[4]xf32>
```

Slice load and store operations use an explicit scalable predicate:

```mlir
%next = arm_sme.load_tile_slice %src[%i, %j], %mask, %tile, %k
  : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>

arm_sme.store_tile_slice %next, %k, %mask, %dst[%i, %j]
  : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
```

### Generic Outer Product

| Operation | Purpose |
|---|---|
| `arm_sme.outerproduct` | Computes an SME-sized outer product, optionally with masks, an accumulator, and add/sub behavior. |

The generic outer product is close to `vector.outerproduct`, but its masking is
on the operands. That matches how SME instructions use predicates.

```mlir
%tile = arm_sme.outerproduct %lhs, %rhs
  : vector<[4]xf32>, vector<[4]xf32>
```

An existing accumulator tile can be provided:

```mlir
%tile2 = arm_sme.outerproduct %lhs, %rhs acc(%tile)
  : vector<[4]xf32>, vector<[4]xf32>
```

The optional combining kind is `kind<add>` or `kind<sub>`. Add corresponds to
MOPA-style accumulation. Sub corresponds to MOPS-style subtraction.

### Fused Widening Outer Product Operations

| Operation | Input | Result | Meaning |
|---|---|---|---|
| `arm_sme.fmopa_2way` | `vector<[8]xf16>` or `vector<[8]xbf16>` | `vector<[4]x[4]xf32>` | Floating-point sum of two widened outer products and accumulate. |
| `arm_sme.fmops_2way` | `vector<[8]xf16>` or `vector<[8]xbf16>` | `vector<[4]x[4]xf32>` | Floating-point sum of two widened outer products and subtract. |
| `arm_sme.smopa_2way` | `vector<[8]xi16>` | `vector<[4]x[4]xi32>` | Signed integer sum of two widened outer products and accumulate. |
| `arm_sme.smops_2way` | `vector<[8]xi16>` | `vector<[4]x[4]xi32>` | Signed integer sum of two widened outer products and subtract. |
| `arm_sme.umopa_2way` | `vector<[8]xi16>` | `vector<[4]x[4]xi32>` | Unsigned integer sum of two widened outer products and accumulate. |
| `arm_sme.umops_2way` | `vector<[8]xi16>` | `vector<[4]x[4]xi32>` | Unsigned integer sum of two widened outer products and subtract. |
| `arm_sme.smopa_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Signed integer sum of four widened outer products and accumulate. |
| `arm_sme.smops_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Signed integer sum of four widened outer products and subtract. |
| `arm_sme.umopa_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Unsigned integer sum of four widened outer products and accumulate. |
| `arm_sme.umops_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Unsigned integer sum of four widened outer products and subtract. |
| `arm_sme.sumopa_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Signed-by-unsigned sum of four widened outer products and accumulate. |
| `arm_sme.sumops_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Signed-by-unsigned sum of four widened outer products and subtract. |
| `arm_sme.usmopa_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Unsigned-by-signed sum of four widened outer products and accumulate. |
| `arm_sme.usmops_4way` | `vector<[16]xi8>` or `vector<[8]xi16>` | `vector<[4]x[4]xi32>` or `vector<[2]x[2]xi64>` | Unsigned-by-signed sum of four widened outer products and subtract. |

These operations are usually produced by `arm-sme-outer-product-fusion`, not by
frontends. They are close to hardware instructions and lower directly to SME
intrinsic wrappers.

### LLVM Intrinsic Wrapper Operations

The `arm_sme.intr.*` operations are low-level wrappers around AArch64 SME LLVM
intrinsics. They are still MLIR operations, but they are already close to LLVM
IR and backend instruction selection.

Count and zero operations:

| Operation | Purpose |
|---|---|
| `arm_sme.intr.cntsd` | Counts streaming doubleword elements and is used to compute streaming vector length. |
| `arm_sme.intr.zero` | Zeroes ZA tiles selected by an immediate tile mask. |

Memory load operations:

| Operation | Purpose |
|---|---|
| `arm_sme.intr.ld1b.horiz` | Load byte elements into a horizontal tile slice. |
| `arm_sme.intr.ld1b.vert` | Load byte elements into a vertical tile slice. |
| `arm_sme.intr.ld1h.horiz` | Load halfword elements into a horizontal tile slice. |
| `arm_sme.intr.ld1h.vert` | Load halfword elements into a vertical tile slice. |
| `arm_sme.intr.ld1w.horiz` | Load word elements into a horizontal tile slice. |
| `arm_sme.intr.ld1w.vert` | Load word elements into a vertical tile slice. |
| `arm_sme.intr.ld1d.horiz` | Load doubleword elements into a horizontal tile slice. |
| `arm_sme.intr.ld1d.vert` | Load doubleword elements into a vertical tile slice. |
| `arm_sme.intr.ld1q.horiz` | Load quadword elements into a horizontal tile slice. |
| `arm_sme.intr.ld1q.vert` | Load quadword elements into a vertical tile slice. |

Memory store operations:

| Operation | Purpose |
|---|---|
| `arm_sme.intr.st1b.horiz` | Store byte elements from a horizontal tile slice. |
| `arm_sme.intr.st1b.vert` | Store byte elements from a vertical tile slice. |
| `arm_sme.intr.st1h.horiz` | Store halfword elements from a horizontal tile slice. |
| `arm_sme.intr.st1h.vert` | Store halfword elements from a vertical tile slice. |
| `arm_sme.intr.st1w.horiz` | Store word elements from a horizontal tile slice. |
| `arm_sme.intr.st1w.vert` | Store word elements from a vertical tile slice. |
| `arm_sme.intr.st1d.horiz` | Store doubleword elements from a horizontal tile slice. |
| `arm_sme.intr.st1d.vert` | Store doubleword elements from a vertical tile slice. |
| `arm_sme.intr.st1q.horiz` | Store quadword elements from a horizontal tile slice. |
| `arm_sme.intr.st1q.vert` | Store quadword elements from a vertical tile slice. |
| `arm_sme.intr.str` | Store ZA-related data using the SME STR intrinsic form. |

Tile slice read/write operations:

| Operation | Purpose |
|---|---|
| `arm_sme.intr.read.horiz` | Read a horizontal tile slice into an SVE vector. |
| `arm_sme.intr.read.vert` | Read a vertical tile slice into an SVE vector. |
| `arm_sme.intr.write.horiz` | Write an SVE vector into a horizontal tile slice. |
| `arm_sme.intr.write.vert` | Write an SVE vector into a vertical tile slice. |

Outer-product intrinsic operations:

| Operation | Purpose |
|---|---|
| `arm_sme.intr.mopa` | Non-widening outer product and accumulate. |
| `arm_sme.intr.mops` | Non-widening outer product and subtract. |
| `arm_sme.intr.mopa.wide` | Widening floating-point outer product and accumulate. |
| `arm_sme.intr.mops.wide` | Widening floating-point outer product and subtract. |
| `arm_sme.intr.smopa.wide` | Widening signed integer outer product and accumulate. |
| `arm_sme.intr.smops.wide` | Widening signed integer outer product and subtract. |
| `arm_sme.intr.umopa.wide` | Widening unsigned integer outer product and accumulate. |
| `arm_sme.intr.umops.wide` | Widening unsigned integer outer product and subtract. |
| `arm_sme.intr.sumopa.wide` | Widening signed-by-unsigned outer product and accumulate. |
| `arm_sme.intr.sumops.wide` | Widening signed-by-unsigned outer product and subtract. |
| `arm_sme.intr.usmopa.wide` | Widening unsigned-by-signed outer product and accumulate. |
| `arm_sme.intr.usmops.wide` | Widening unsigned-by-signed outer product and subtract. |
| `arm_sme.intr.smopa.za32` | Signed 2-way outer product using a ZA32 tile. |
| `arm_sme.intr.smops.za32` | Signed 2-way outer product/subtract using a ZA32 tile. |
| `arm_sme.intr.umopa.za32` | Unsigned 2-way outer product using a ZA32 tile. |
| `arm_sme.intr.umops.za32` | Unsigned 2-way outer product/subtract using a ZA32 tile. |

Most users should read intrinsic ops as "the target-specific form after
lowering." They are important for backend correctness, but not the best
teaching entry point for SME programming in MLIR.

## Transformations

### `enable-arm-streaming`

`enable-arm-streaming` runs on `func.func`. It annotates functions with Arm
streaming-mode and ZA-mode attributes.

Important options:

| Option | Meaning |
|---|---|
| `streaming-mode=disabled` | Do not enable streaming mode. |
| `streaming-mode=streaming` | Streaming mode is part of the function ABI; the caller manages `PSTATE.SM`. |
| `streaming-mode=streaming-locally` | The callee enters and exits streaming mode internally. |
| `streaming-mode=streaming-compatible` | The function can be entered in either streaming or non-streaming mode. |
| `za-mode=disabled` | Do not enable ZA state. |
| `za-mode=new-za` | The function creates ZA state on entry and destroys it on exit. |
| `za-mode=in-za` | ZA is an input to the function. |
| `za-mode=out-za` | ZA is an output of the function. |
| `za-mode=inout-za` | ZA is both input and output. |
| `za-mode=preserves-za` | The function returns with shared ZA unchanged. |
| `if-required-by-ops` | Apply the selected modes only when the function contains ArmSME tile-interface ops. |
| `if-scalable-and-supported` | Apply the selected modes only for supported scalable-vector operations. |

This pass is about function boundaries and ABI behavior. It does not by itself
turn a matrix multiply into SME instructions.

### `arm-sme-outer-product-fusion`

`arm-sme-outer-product-fusion` runs on `func.func`. It finds chains of
`arm_sme.outerproduct` operations connected through an accumulator and fuses
them into 2-way or 4-way widening outer-product operations.

For example, two `f16` outer products widened to `f32` can become one
`arm_sme.fmopa_2way`. Four `i8` or `i16` outer products can become one of the
4-way integer operations.

This pass matters because the fused operations are closer to the hardware
instructions and have direct lowering support in `convert-arm-sme-to-llvm`.

### `arm-sme-vector-legalization`

`arm-sme-vector-legalization` runs on `module`. It rewrites vector operations
so they can be lowered to ArmSME. The main job is decomposing vector types that
are larger than a single SME tile into multiple tile-sized operations.

The current decomposition is limited to vector types that are exact multiples
of SME tiles. That means both scalable dimensions must divide cleanly into the
SME tile shape for the element type.

### Tile Allocation

Tile allocation assigns concrete `tile_id` attributes to operations that
implement `ArmSMETileOpInterface`.

There is a `test-arm-sme-tile-allocation` pass for tests and debugging, but
normal pipelines rely on `convert-arm-sme-to-llvm`, which runs tile allocation
internally before converting tile operations to intrinsic operations.

Tile allocation can introduce spills and fills when the virtual tile live
ranges do not fit directly in available hardware tile IDs.

### Canonicalization

Standard canonicalization patterns apply to some ArmSME operations. For
example, tests cover simplifying redundant tile slice extraction and insertion
patterns.

Canonicalization is useful before tile allocation because it can shorten live
ranges and remove unnecessary tile operations.

## Conversions / Lowering Paths

### `convert-arith-to-arm-sme`

This pass converts `arith.constant` operations that produce valid SME tile
vectors.

The important case is a zero dense constant:

```mlir
%0 = arith.constant dense<0.0> : vector<[4]x[4]xf32>
```

It lowers to:

```mlir
%0 = arm_sme.zero : vector<[4]x[4]xf32>
```

Nonzero dense tile constants are expanded by building tile slices, using
`arm_sme.get_tile` and `arm_sme.insert_tile_slice`.

### `convert-vector-to-arm-sme`

This pass converts matching `vector` operations to `arm_sme`.

Important patterns include:

| Vector operation pattern | ArmSME result |
|---|---|
| 2-D `vector.transfer_read` of a valid SME tile type | `arm_sme.tile_load` |
| 2-D `vector.transfer_write` of a valid SME tile type | `arm_sme.tile_store` |
| `vector.load` of a valid SME tile type | `arm_sme.tile_load` |
| `vector.store` of a valid SME tile type | `arm_sme.tile_store` |
| `vector.outerproduct` with an SME-sized result | `arm_sme.outerproduct` |
| `vector.broadcast` to an SME tile | A loop that inserts slices into a tile. |
| `vector.transpose` of an SME tile | Store/reload or folding into transfer-read layout when possible. |
| `vector.extract` from an SME tile | `arm_sme.extract_tile_slice` plus scalar/vector extraction. |
| `vector.insert` into an SME tile | Slice extraction/insertion and `arm_sme.insert_tile_slice`. |
| `vector.print` of an SME tile | A loop that extracts and prints tile slices. |

Not every vector operation lowers. The source must have a supported scalable
tile shape, supported element type, and compatible indexing or masking form.

### `convert-arm-sme-to-scf`

This pass lowers structured full-tile memory operations to loops over tile
slices.

Typical rewrites:

| Input | Lowered form |
|---|---|
| `arm_sme.tile_load` | `arm_sme.get_tile` or `arm_sme.zero`, an `scf.for` loop, and `arm_sme.load_tile_slice`. |
| `arm_sme.tile_store` | An `scf.for` loop and `arm_sme.store_tile_slice`. |

This pass is useful because the final intrinsic lowering operates naturally on
tile slices.

### `convert-arm-sme-to-llvm`

This pass lowers ArmSME tile operations to LLVM-compatible intrinsic wrapper
operations.

It performs tile allocation first. Then it lowers:

- `arm_sme.zero` to `arm_sme.intr.zero`;
- `arm_sme.load_tile_slice` to an `arm_sme.intr.ld1*` op;
- `arm_sme.store_tile_slice` to an `arm_sme.intr.st1*` op;
- `arm_sme.insert_tile_slice` to `arm_sme.intr.write.*`;
- `arm_sme.extract_tile_slice` to `arm_sme.intr.read.*`;
- `arm_sme.outerproduct` to `arm_sme.intr.mopa` for supported non-widening
  floating-point add cases;
- fused widening outer products to `arm_sme.intr.*mopa*` or
  `arm_sme.intr.*mops*` forms;
- `arm_sme.streaming_vl` to `arm_sme.intr.cntsd`, index casts, and scaling.

The pass also handles tile spills and fills when virtual tile allocation needs
temporary memory. If an operation still has an SME tile type after conversion
and is not one of the allowed pseudo operations, the pass reports an error.

### Common Pipeline Shape

A simplified ArmSME lowering path looks like this:

```text
linalg / vector / arith
  -> vectorization and vector cleanup
  -> arm-sme-vector-legalization
  -> convert-arith-to-arm-sme
  -> convert-vector-to-arm-sme
  -> arm-sme-outer-product-fusion
  -> convert-arm-sme-to-scf
  -> enable-arm-streaming
  -> convert-arm-sme-to-llvm
  -> LLVM lowering and translation
```

Real pipelines may interleave canonicalization, CSE, bufferization, and other
target-specific steps. The important conceptual order is: keep IR portable
early, introduce SME tile operations when the shapes match, lower structured
tile operations to slice operations, then allocate tiles and emit intrinsics.

## Example IR

### Zero A Tile

```mlir
module {
  func.func @zero_tile() -> vector<[4]x[4]xf32> {
    %0 = arm_sme.zero : vector<[4]x[4]xf32>
    return %0 : vector<[4]x[4]xf32>
  }
}
```

This creates a virtual `f32` tile with zero contents. Before LLVM conversion,
it is still an SSA value. During `convert-arm-sme-to-llvm`, tile allocation
chooses a ZA tile ID and the operation lowers to `arm_sme.intr.zero`.

### Load And Store A Tile

```mlir
module {
  func.func @tile_load_store(
      %src: memref<?x?xf32>, %dst: memref<?x?xf32>,
      %i: index, %j: index) {
    %tile = arm_sme.tile_load %src[%i, %j]
      : memref<?x?xf32>, vector<[4]x[4]xf32>

    arm_sme.tile_store %tile, %dst[%i, %j]
      : memref<?x?xf32>, vector<[4]x[4]xf32>

    return
  }
}
```

This is the structured form of a full tile memory transfer. A later
`convert-arm-sme-to-scf` pass can turn it into a loop over tile slices.

### Compute An Outer Product

```mlir
module {
  func.func @outer(
      %lhs: vector<[4]xf32>,
      %rhs: vector<[4]xf32>) -> vector<[4]x[4]xf32> {
    %tile = arm_sme.outerproduct %lhs, %rhs
      : vector<[4]xf32>, vector<[4]xf32>
    return %tile : vector<[4]x[4]xf32>
  }
}
```

This is the direct SME-shaped representation of an outer product. If an
accumulator is supplied, the result represents an accumulated update to that
tile.

### Use A Fused Widening Outer Product

```mlir
module {
  func.func @fused_f16_outer(
      %lhs: vector<[8]xf16>,
      %rhs: vector<[8]xf16>) -> vector<[4]x[4]xf32> {
    %tile = arm_sme.fmopa_2way %lhs, %rhs
      : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
    return %tile : vector<[4]x[4]xf32>
  }
}
```

This is already close to an SME widening outer-product instruction. It is the
kind of operation `arm-sme-outer-product-fusion` tries to create from generic
outer-product chains.

### Query Streaming Vector Length

```mlir
module {
  func.func @streaming_words() -> index {
    %svl = arm_sme.streaming_vl <word>
    return %svl : index
  }
}
```

This returns the number of bytes, halfwords, words, or doublewords in the
streaming vector length depending on the `TypeSize` attribute.

## Mental Model

Think of `arm_sme` as the point where vectorized matrix code becomes "real Arm
SME code."

Before `arm_sme`, the compiler mostly describes mathematical structure:
matrix multiplication, vector transfer, vector outer products, loops, and
memory.

Inside `arm_sme`, the compiler describes SME hardware structure:
virtual ZA tiles, scalable tile slices, streaming vector length, predicates,
and outer-product instructions.

After `arm_sme`, the compiler describes LLVM-compatible intrinsic operations
and eventually machine instructions.

The dialect is therefore not just a bag of intrinsics. Its higher-level ops are
there so MLIR can still reason about tile dataflow before final instruction
selection.

## Gotchas

- `arm_sme` is target-specific. Do not introduce it before generic
  transformations have had a chance to run.
- Tile vector types must be valid SME tile shapes. A random 2-D scalable vector
  is not automatically legal.
- Most tile ops need tile allocation before LLVM conversion. Missing `tile_id`
  attributes are expected before allocation and an error after allocation is
  required.
- `arm_sme.tile_load` and `arm_sme.tile_store` are structured full-tile ops.
  The lower-level intrinsic path is slice-based.
- `arm_sme.outerproduct` masks operands, not result elements. This differs from
  how many beginners first think about matrix masks.
- Direct LLVM conversion of generic `arm_sme.outerproduct` currently supports
  non-widening floating-point add cases. Integer and subtract forms usually
  need earlier rewriting or fusion into supported widening operations.
- The fused 2-way and 4-way operations encode packed input layouts. They are
  not just spelling variants of generic outer products.
- Streaming mode and ZA mode are ABI-level choices. Adding them changes the
  function contract.
- Intrinsic wrapper ops are not portable IR. They should appear late, close to
  LLVM conversion.
- Some lowering patterns only handle exact tile shapes, exact multiples of tile
  shapes, or specific indexing maps.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/ArmSME/IR/ArmSME.td`
- `mlir/include/mlir/Dialect/ArmSME/IR/ArmSMEOps.td`
- `mlir/include/mlir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.td`
- `mlir/include/mlir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.h`
- `mlir/include/mlir/Dialect/ArmSME/IR/ArmSMEEnums.h`
- `mlir/lib/Dialect/ArmSME/IR/ArmSME.cpp`
- `mlir/lib/Dialect/ArmSME/Utils/Utils.cpp`

Transform source files:

- `mlir/include/mlir/Dialect/ArmSME/Transforms/Passes.td`
- `mlir/lib/Dialect/ArmSME/Transforms/EnableArmStreaming.cpp`
- `mlir/lib/Dialect/ArmSME/Transforms/OuterProductFusion.cpp`
- `mlir/lib/Dialect/ArmSME/Transforms/TileAllocation.cpp`
- `mlir/lib/Dialect/ArmSME/Transforms/VectorLegalization.cpp`

Conversion source files:

- `mlir/lib/Conversion/ArithToArmSME/ArithToArmSME.cpp`
- `mlir/lib/Conversion/VectorToArmSME/VectorToArmSME.cpp`
- `mlir/lib/Conversion/ArmSMEToSCF/ArmSMEToSCF.cpp`
- `mlir/lib/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.cpp`
- `mlir/include/mlir/Conversion/Passes.td`

Tests:

- `mlir/test/Dialect/ArmSME`
- `mlir/test/Conversion/ArithToArmSME`
- `mlir/test/Conversion/VectorToArmSME`
- `mlir/test/Conversion/ArmSMEToSCF`
- `mlir/test/Conversion/ArmSMEToLLVM`
- `mlir/test/Integration/Dialect/Linalg/CPU/ArmSME`
- `mlir/test/Integration/Dialect/Vector/CPU/ArmSME`
