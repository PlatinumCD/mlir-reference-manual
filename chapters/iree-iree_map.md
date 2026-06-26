# IREE `iree_map` Dialect

The IREE `iree_map` dialect is a small code generation dialect for map algebra. Unlike most MLIR dialects in this book, it does not define ordinary operations. Its public surface is made of attributes that describe hierarchical coordinate mappings. Those attributes are then used by IREE vector layout and GPU distribution code to decide which data coordinates are owned by which thread and which values remain local inside each thread.

For a beginner, the key idea is that `iree_map` describes layout as a mathematical object. A layout answers a question like: given coordinates in a vector or tensor, what one-dimensional index, lane, or thread ID do they map to? The dialect gives IREE a compact way to write row-major layouts, column-major layouts, strided layouts, broadcasts, and nested layouts that resemble the structure needed by vector and GPU code generation.

This dialect is important because layout is not just metadata in a GPU compiler. It determines what each lane computes, how a vector is distributed across lanes, where broadcast values live, when a layout conversion needs shared memory, and how operations such as transpose or shape cast can be rewritten after distribution. IREE uses `iree_map` to make those decisions explicit and algebraic instead of scattering layout reasoning across ad hoc integer arithmetic.

## Operation Inventory

The `iree_map` dialect defines no operations in this checkout. There are no `iree_map.*` operation names to list.

Its inventory is attribute-based:

```text
#iree_map.pack_map
#iree_map.pack_layout
```

This is a useful dialect-design lesson. MLIR dialects can define types, attributes, interfaces, and operations. `iree_map` is a dialect whose main job is to define attributes and attach behavior to them.

## Pack Maps

`#iree_map.pack_map` is the base attribute. It represents a mapping from coordinates to a one-dimensional index using a shape and a stride. The syntax is:

```mlir
#iree_map.pack_map<(shape) : (stride)>
```

For a flat two-dimensional layout such as `#iree_map.pack_map<(4, 8) : (8, 1)>`, a coordinate `(c0, c1)` maps to `c0 * 8 + c1 * 1`. That is the ordinary row-major layout for a 4-by-8 space. A column-major version would be `#iree_map.pack_map<(4, 8) : (1, 4)>`, where the first coordinate changes fastest.

Stride zero means broadcast. For example, `#iree_map.pack_map<(4, 8) : (0, 1)>` ignores the first coordinate when computing the one-dimensional index. All rows at the same column map to the same index. This is how the dialect can say that one dimension is replicated rather than uniquely owned.

Pack maps can be hierarchical. The shape and stride are not just flat lists of integers; they are `IntTuple` trees. A shape like `((2, 4), 8)` has a nested first mode. The corresponding stride must have the same tree structure, such as `((16, 1), 4)`. The verifier requires shape and stride to be congruent, shape leaves to be positive, and stride leaves to be non-negative.

The dialect uses lexicographic, row-major ordering as the default convention. `PackMapAttr::makeIdentity` builds the identity row-major map for a flat shape: for shape `(M, N, K)`, it produces strides `(N*K, K, 1)`.

## Pack Layouts

`#iree_map.pack_layout` wraps a pack map for vector layout distribution. Its syntax looks like a pack map:

```mlir
#iree_map.pack_layout<((32, 4)) : ((1, 0))>
```

The semantic difference is that `pack_layout` maps data coordinates to thread IDs. Leaves with positive stride contribute to the thread mapping. Leaves with stride zero are broadcast or per-thread value dimensions. After removing broadcast leaves, the remaining thread mapping must be injective: two different thread coordinates must not map to the same thread ID.

This attribute decomposes a layout into two ideas. The thread mapping tells which thread owns a coordinate. The value mapping tells where, inside one thread's local tile, the coordinate falls. For example, a layout like `((4, 2), (4, 8)) : ((1, 0), (0, 4))` has positive-stride leaves for thread ownership and zero-stride leaves for per-thread values. That is exactly the kind of fact a vector distribution pass needs when replacing a large SIMD vector with smaller per-thread SIMT values.

`pack_layout` also normalizes the underlying map by coalescing inside each top-level mode. This preserves the rank that must match the vector shape while simplifying nested leaves that are contiguous inside a mode.

## Map Algebra

`PackMapAttr` is more than a printed pair of shape and stride. The implementation provides a closed algebra over maps. Important methods include `evaluate`, `coalesce`, `coalesceModes`, `flatten`, `compose`, `complement`, `logicalDivide`, `logicalProduct`, `filter`, `rightInverse`, `leftInverse`, `tiledDivide`, `tiledProduct`, `permute`, `project`, and `makeIdentity`.

`evaluate` computes the index for a coordinate. If a flat index is supplied for a multi-leaf shape, the implementation first converts it to natural coordinates using row-major mixed-radix rules, then applies the stride inner product.

`coalesce` merges adjacent leaves that cover a contiguous range, even across top-level mode boundaries. `coalesceModes` performs the same kind of simplification but preserves each top-level mode boundary. `flatten` removes hierarchy and keeps only the leaf sequence.

`compose` is functional composition: one map is applied after another. This is what lets a reshape or layout conversion be expressed algebraically instead of manually rebuilding every index expression.

`complement` finds the missing mapping needed to cover a target range with no gaps. This matters when a thread mapping covers only part of a thread grid and must be implicitly replicated across the rest.

`logicalDivide` and `logicalProduct` express tiling-style transformations. A layout can be split into inner and outer pieces, or replicated according to a tiler pattern. `tiledDivide` and `tiledProduct` are variants that flatten outer or replicated modes into top-level modes.

`rightInverse` and `leftInverse` are used when a compiler needs to go backwards through a layout. A right inverse recovers coordinates from an injective contiguous mapping. A left inverse can recover original positions even when the output range has gaps by using a complement layout.

For beginners, the implication is simple: `iree_map` is not just naming a layout. It gives IREE a vocabulary for transforming layouts and proving that the transformed layout still means the right thing.

## Vector Layout Interface

The dialect registers an external implementation of IREE's `VectorLayoutInterface` for `PackLayoutAttr`. This is where `pack_layout` becomes useful to the rest of codegen.

The interface can report the layout rank, the undistributed shape, and the distributed shape. The undistributed shape is the original vector shape described by the mode sizes. The distributed shape is built from stride-zero value leaves; if a mode has no value leaves, the distributed dimension is `1`. In other words, positive-stride leaves are assigned to threads, while zero-stride leaves remain as per-thread vector dimensions.

The interface also validates that a layout matches a shaped type. For vectors, the layout rank must match the vector rank, and static dimensions must match the layout's undistributed shape. Ranked tensors are allowed to have layouts that exceed the tensor size for padding or masking.

Interface methods also support layout permutation, projection, affine-map application, reshape, recombination, and deciding whether a conversion between two layouts needs shared memory. If the coalesced source and target pack layouts differ, the conversion is treated as needing shared memory. That is a practical code generation decision encoded through the layout abstraction.

## Transformations And Conversions

The `iree_map` directory does not register a standalone pass like `iree-map-lower`. Its transformations are exposed as pattern population functions and external interfaces used by IREE codegen.

`registerIREEMapExternalInterfaces` registers the `PackLayoutAttr` vector layout model. This makes map layouts visible to vector layout analysis and distribution infrastructure.

`populateMapDistributeGenericPatterns` adds PackLayout-specific distribution patterns for generic vector operations. In this checkout, it covers:

```text
vector.shape_cast
vector.broadcast
vector.transpose
vector.step
```

For `vector.shape_cast`, the pattern rewrites the operation on the distributed vector shape derived from the destination `PackLayoutAttr`. For `vector.broadcast`, scalar sources are broadcast directly, while vector sources are first distributed according to their source layout. For `vector.transpose`, the pattern expands the permutation over the distributed value leaves, because one original vector dimension can become several per-thread distributed dimensions. For `vector.step`, the pattern builds a per-thread vector of indices by combining a constant value-offset vector with a thread-offset value computed from the layout.

The helper `buildThreadOffsets` is the key lowering utility. It collects positive-stride thread leaves, delinearizes a flat thread ID into thread-leaf coordinates with `affine.delinearize_index`, and then re-linearizes those coordinates into per-dimension data-space offsets with `affine.linearize_index`. The result is a base offset for each original vector dimension.

The test transform operation `transform.iree.test_gpu_vector_distribution` wires these patterns into vector distribution for tests. It creates a lane ID from `gpu.thread_id x`, populates general GPU distribution patterns, nested-layout distribution patterns, and the `iree_map` generic patterns, then runs vector distribution. Production pipelines can use the same underlying interfaces and pattern helpers without treating `iree_map` as a separate lowering pass.

## When To Use It

Use `iree_map` when you need to describe how logical vector or tensor coordinates map to a linear lane or thread space. It is especially useful in GPU codegen, SIMT distribution, vector layout conversion, and layouts with nested tiling or broadcasted per-thread values.

Do not use it to represent computation. There are no arithmetic, memory, or control-flow operations in this dialect. Computation remains in dialects such as `vector`, `arith`, `linalg`, `iree_vector_ext`, GPU, and LLVM-related dialects. `iree_map` describes the layout facts that those computations rely on.

When reading IR, look for `#iree_map.pack_layout` attached to `iree_vector_ext.to_layout` operations or attributes. Positive strides tell you which coordinates are distributed across lanes. Zero strides tell you which coordinates are broadcast or retained inside the per-thread value. Nested tuples tell you that one logical dimension has been split into smaller layout factors.

The dialect implies that IREE is doing layout-aware code generation. A pack layout is not decorative metadata; it can change how vector operations are distributed, whether a layout conversion needs shared memory, how transposes expand in per-thread space, and how `vector.step` is reconstructed after distribution.
