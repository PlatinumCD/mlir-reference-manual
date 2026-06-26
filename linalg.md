# Linalg Dialect

## Beginner Summary

`linalg` is MLIR's dialect for structured linear algebra and tensor
computation.

It represents computations such as elementwise maps, reductions, matrix
multiplication, convolution, pooling, packing, transposition, and more. The
important word is structured: a Linalg operation describes a regular iteration
space, how operands are indexed in that space, and what scalar computation runs
at each point.

A beginner can think of Linalg as this idea:

```text
for every point in a structured iteration space:
  read scalar elements from inputs
  read scalar elements from outputs when needed
  run a small scalar body
  write yielded scalar values to outputs
```

The dialect is used heavily in MLIR because it is high-level enough for tensor
and machine-learning transformations, but regular enough to lower to loops,
vectors, buffers, library calls, or target-specific code.

The most important op is `linalg.generic`. Named ops such as `linalg.matmul`,
`linalg.add`, or `linalg.conv_2d_nhwc_hwcf` are specialized forms that make
common computations easier to recognize. Many passes convert between those
forms.

## Why This Dialect Exists

Compilers need a representation that sits between high-level math and low-level
loops.

If a compiler lowers directly from a model operation to loops, it loses useful
structure. A loop nest can compute a matrix multiplication, but the compiler may
no longer know that it is a matrix multiplication. That makes tiling,
fusion, vectorization, library-call replacement, and layout changes harder.

If a compiler stays at the model level too long, it cannot express general
low-level transformations. A source op such as "convolution" or "add" is often
too coarse for hardware-specific scheduling.

`linalg` solves this by giving computations a structured shape:

- operands are tensors or memrefs
- each operand has an affine indexing map
- each loop dimension has an iterator type such as `parallel` or `reduction`
- scalar computation lives in a region
- outputs are explicit destinations

That structure lets MLIR run transformations while still understanding the
computation.

## When It Matters

`linalg` matters in tensor, ML, and accelerator pipelines.

You are likely to see it when:

- TOSA or another ML dialect lowers into tensor computations
- tensor padding, elementwise math, matmul, convolution, or pooling must be
  optimized before code generation
- a pipeline wants to tile, fuse, vectorize, or distribute computation
- tensor IR is being prepared for bufferization
- bufferized computation is being lowered to `scf` or `affine` loops
- a known operation should become a library call
- a generic computation should be recognized as a named operation

It is one of the central bridge dialects in MLIR. Many pipelines flow through
Linalg even if they do not start or end there.

## When To Use It

Use `linalg` when the computation has a regular iteration space over tensors or
memrefs.

Good uses:

- elementwise tensor math
- reductions
- matrix multiplication and other contractions
- convolution and pooling
- layout packing and unpacking
- transposes and broadcasts that move or materialize data
- tensor-level intermediate IR before bufferization
- memref-level loop generation after bufferization

Avoid using `linalg` when:

- the control flow is irregular and not naturally an affine-indexed iteration
  space
- the computation is mostly scalar control flow
- the operation is only a metadata view change, such as `memref.transpose`
- the target representation already needs explicit low-level instructions
- the source semantics require a specialized dialect that has not yet been
  lowered

In practice, a common path is:

```text
model dialect -> tensor/linalg -> bufferization -> linalg on memrefs -> loops/vector/gpu/llvm
```

## Core Concepts

### Structured Operations

A Linalg structured operation describes:

- inputs
- output destinations
- an iteration space
- indexing maps
- iterator types
- a scalar body

The scalar body works on individual elements, not whole tensors. Linalg uses the
operation metadata to decide which elements those scalar arguments correspond
to.

### Destination-Passing Style

Linalg uses destination-passing style.

An op does not simply say "produce a result". It also says where the result is
initialized or written.

For tensors:

```mlir
%empty = tensor.empty() : tensor<4x16xf32>
%result = linalg.fill ins(%cst : f32)
                      outs(%empty : tensor<4x16xf32>)
                      -> tensor<4x16xf32>
```

The output tensor operand is the destination. The op returns a new tensor value.

For memrefs:

```mlir
linalg.fill ins(%cst : f32) outs(%buffer : memref<4x16xf32>)
```

The destination is written in place and the operation has no tensor result.

This distinction is important:

- tensor semantics are value semantics
- memref semantics are buffer mutation semantics

### Inputs, Outputs, And Inits

Linalg often calls its destination operands `outs` or `inits`.

An output operand may be used in the scalar body. For example, matrix
multiplication accumulates into the current output element:

```text
C[m, n] = C[m, n] + A[m, k] * B[k, n]
```

That means the output is both the destination and the initial accumulator.

Other ops, such as a pure elementwise map, may ignore the previous output value
and use the destination only for shape and storage.

### Indexing Maps

Indexing maps connect loop dimensions to operand dimensions.

For matrix multiplication, the logical loops are `(m, n, k)`:

```mlir
affine_map<(m, n, k) -> (m, k)>  // A
affine_map<(m, n, k) -> (k, n)>  // B
affine_map<(m, n, k) -> (m, n)>  // C
```

The maps explain:

- how to read `A`
- how to read `B`
- how to read or write `C`

This is the core of `linalg.generic`. The scalar body does the math, while the
indexing maps describe how that math is applied over tensors or buffers.

### Iterator Types

Each loop dimension has an iterator type.

The most important iterator types are:

- `parallel`: iterations are independent
- `reduction`: iterations contribute to an accumulated value
- `window`: used by windowed operations such as convolution and pooling

For matmul:

```text
m: parallel
n: parallel
k: reduction
```

This tells transformations which loops can be parallelized and which loops must
combine values.

### Regions And `linalg.yield`

`linalg.generic`, `linalg.map`, `linalg.reduce`, and many named structured ops
contain regions.

The region receives scalar block arguments. It must end with `linalg.yield`,
which returns scalar values for the output destinations.

Example:

```mlir
^bb0(%a: f32, %b: f32, %out: f32):
  %sum = arith.addf %a, %b : f32
  linalg.yield %sum : f32
```

`linalg.yield` is not a function return. It returns one scalar iteration result
to the enclosing Linalg op.

### Named, Category, And Generic Forms

Linalg operations come in several levels of specificity.

`linalg.generic` is the most general structured form. It explicitly stores
indexing maps, iterator types, and a scalar region.

Category ops are structured but more constrained. Examples include:

- `linalg.elementwise`
- `linalg.contract`
- `linalg.map`
- `linalg.reduce`
- `linalg.broadcast`
- `linalg.transpose`

Named ops are the most recognizable forms:

- `linalg.matmul`
- `linalg.conv_2d_nhwc_hwcf`
- `linalg.add`
- `linalg.pooling_nhwc_max`

The dialect has passes that convert between these forms. Generalization goes
from named or category ops toward `linalg.generic`. Specialization tries to
recognize a generic op as a category or named op.

### Tensor And MemRef Semantics

The same Linalg computation can exist over tensors or memrefs.

Tensor form:

```text
ins(tensor values) outs(init tensor) -> result tensor
```

Memref form:

```text
ins(memrefs) outs(output memref)
```

Tensor form is better before bufferization. Memref form is better when lowering
to explicit loops and stores.

### Library Calls

Some Linalg ops can lower to external library calls. The generic path is
`convert-linalg-to-std`, which turns a Linalg op with a library call name into a
`func.call`.

This is not the same as loop lowering. Loop lowering expands the operation into
explicit loads, scalar computation, and stores. Library-call lowering delegates
the computation to an external implementation.

## Operations

The generated Linalg dialect operation inventory has 99 operations.

### Structural And Support Operations

| Operation | Meaning |
| --- | --- |
| `linalg.generic` | Fully explicit structured op with indexing maps, iterator types, inputs, outputs, and scalar region. |
| `linalg.yield` | Terminator for Linalg regions. Yields scalar values to the enclosing structured op. |
| `linalg.index` | Reads the current loop induction value for a dimension inside a Linalg structured op. |

### Category Operations

| Operation | Meaning |
| --- | --- |
| `linalg.map` | Elementwise map with a user-provided scalar mapper region. |
| `linalg.reduce` | Reduction over selected dimensions with a combiner region. |
| `linalg.transpose` | Materializing transpose. Moves data according to a permutation. |
| `linalg.broadcast` | Materializing broadcast into a larger destination shape. |
| `linalg.elementwise` | Category op for unary, binary, or ternary elementwise functions. |
| `linalg.contract` | General contraction op with explicit indexing maps. |

### Elementwise Named Operations

Unary named elementwise ops:

| Operation | Meaning |
| --- | --- |
| `linalg.abs` | Elementwise absolute value. |
| `linalg.ceil` | Elementwise ceiling. |
| `linalg.erf` | Elementwise error function. |
| `linalg.exp` | Elementwise exponential. |
| `linalg.floor` | Elementwise floor. |
| `linalg.log` | Elementwise logarithm. |
| `linalg.negf` | Elementwise floating-point negation. |
| `linalg.reciprocal` | Elementwise reciprocal. |
| `linalg.round` | Elementwise round. |
| `linalg.rsqrt` | Elementwise reciprocal square root. |
| `linalg.sqrt` | Elementwise square root. |
| `linalg.square` | Elementwise square. |
| `linalg.tanh` | Elementwise hyperbolic tangent. |

Binary and ternary named elementwise ops:

| Operation | Meaning |
| --- | --- |
| `linalg.add` | Elementwise add. |
| `linalg.sub` | Elementwise subtract. |
| `linalg.mul` | Elementwise multiply. |
| `linalg.div` | Elementwise signed or floating division depending on type. |
| `linalg.div_unsigned` | Elementwise unsigned integer division. |
| `linalg.max` | Elementwise max. |
| `linalg.min` | Elementwise min. |
| `linalg.powf` | Elementwise floating-point power. |
| `linalg.select` | Elementwise select. |

### Fill, Copy, And Random Fill

| Operation | Meaning |
| --- | --- |
| `linalg.fill` | Fill an output tensor or memref with a scalar value. |
| `linalg.copy` | Copy from one shaped value to another. |
| `linalg.fill_rng_2d` | Fill a 2D output with pseudo-random values in a range. |

### Contractions And Matrix-Like Operations

| Operation | Meaning |
| --- | --- |
| `linalg.matmul` | Matrix multiplication. |
| `linalg.batch_matmul` | Batched matrix multiplication. |
| `linalg.batch_reduce_matmul` | Batched/reduced matrix multiplication form. |
| `linalg.matvec` | Matrix-vector multiplication. |
| `linalg.vecmat` | Vector-matrix multiplication. |
| `linalg.batch_matvec` | Batched matrix-vector multiplication. |
| `linalg.batch_vecmat` | Batched vector-matrix multiplication. |
| `linalg.dot` | Dot product. |
| `linalg.mmt4d` | 4D packed matrix multiplication micro-kernel style operation. |
| `linalg.batch_mmt4d` | Batched `mmt4d`. |
| `linalg.quantized_matmul` | Quantized matrix multiplication with zero-point operands. |
| `linalg.quantized_batch_matmul` | Batched quantized matrix multiplication. |

### Convolution Operations

Generic convolution forms:

| Operation | Meaning |
| --- | --- |
| `linalg.conv_1d` | Generic 1D convolution. |
| `linalg.conv_2d` | Generic 2D convolution. |
| `linalg.conv_3d` | Generic 3D convolution. |

Named convolution layout forms:

| Operation |
| --- |
| `linalg.conv_1d_ncw_fcw` |
| `linalg.conv_1d_nwc_wcf` |
| `linalg.conv_2d_nchw_fchw` |
| `linalg.conv_2d_nchw_fchw_q` |
| `linalg.conv_2d_ngchw_fgchw` |
| `linalg.conv_2d_ngchw_gfchw` |
| `linalg.conv_2d_ngchw_gfchw_q` |
| `linalg.conv_2d_nhwc_fhwc` |
| `linalg.conv_2d_nhwc_fhwc_q` |
| `linalg.conv_2d_nhwc_hwcf` |
| `linalg.conv_2d_nhwc_hwcf_q` |
| `linalg.conv_2d_nhwgc_gfhwc` |
| `linalg.conv_2d_nhwgc_gfhwc_q` |
| `linalg.conv_3d_ncdhw_fcdhw` |
| `linalg.conv_3d_ndhwc_dhwcf` |
| `linalg.conv_3d_ndhwc_dhwcf_q` |

The suffix describes the layout of input and filter tensors. For example,
`nhwc_hwcf` indicates input layout `N,H,W,C` and filter layout `H,W,C,F`.
The `_q` variants represent quantized convolution forms.

### Depthwise Convolution Operations

| Operation |
| --- |
| `linalg.depthwise_conv_1d_ncw_cw` |
| `linalg.depthwise_conv_1d_nwc_wc` |
| `linalg.depthwise_conv_1d_nwc_wcm` |
| `linalg.depthwise_conv_2d_nchw_chw` |
| `linalg.depthwise_conv_2d_nhwc_hwc` |
| `linalg.depthwise_conv_2d_nhwc_hwc_q` |
| `linalg.depthwise_conv_2d_nhwc_hwcm` |
| `linalg.depthwise_conv_2d_nhwc_hwcm_q` |
| `linalg.depthwise_conv_3d_ncdhw_cdhw` |
| `linalg.depthwise_conv_3d_ndhwc_dhwc` |
| `linalg.depthwise_conv_3d_ndhwc_dhwcm` |

Depthwise convolution is separated because its channel semantics differ from
ordinary convolution.

### Pooling Operations

1D pooling:

| Operation |
| --- |
| `linalg.pooling_ncw_max` |
| `linalg.pooling_ncw_sum` |
| `linalg.pooling_nwc_max` |
| `linalg.pooling_nwc_max_unsigned` |
| `linalg.pooling_nwc_min` |
| `linalg.pooling_nwc_min_unsigned` |
| `linalg.pooling_nwc_sum` |

2D pooling:

| Operation |
| --- |
| `linalg.pooling_nchw_max` |
| `linalg.pooling_nchw_sum` |
| `linalg.pooling_nhwc_max` |
| `linalg.pooling_nhwc_max_unsigned` |
| `linalg.pooling_nhwc_min` |
| `linalg.pooling_nhwc_min_unsigned` |
| `linalg.pooling_nhwc_sum` |

3D pooling:

| Operation |
| --- |
| `linalg.pooling_ndhwc_max` |
| `linalg.pooling_ndhwc_min` |
| `linalg.pooling_ndhwc_sum` |

### Relayout Operations

| Operation | Meaning |
| --- | --- |
| `linalg.pack` | Convert a tensor or memref into a tiled packed layout, optionally with padding and outer-dimension permutation. |
| `linalg.unpack` | Convert a packed layout back to an unpacked destination layout. |

### Aggregate And Special Operations

| Operation | Meaning |
| --- | --- |
| `linalg.softmax` | Aggregate softmax op. Decomposes to smaller structured operations. |
| `linalg.winograd_filter_transform` | Filter transform for Winograd Conv2D lowering. |
| `linalg.winograd_input_transform` | Input transform for Winograd Conv2D lowering. |
| `linalg.winograd_output_transform` | Output transform for Winograd Conv2D lowering. |

## Attributes And Interfaces

### Iterator Type Attribute

Linalg iterator types describe loop dimensions:

```mlir
iterator_types = ["parallel", "parallel", "reduction"]
```

They are represented by the dialect's iterator type enum attribute.

### Elementwise And Function Attributes

The dialect defines attributes for generated region builders and category ops:

- `#linalg.elementwise_kind<...>`
- `#linalg.unary_fn<...>`
- `#linalg.binary_fn<...>`
- `#linalg.ternary_fn<...>`
- `#linalg.type_fn<...>`

Beginners usually encounter these indirectly through named ops and category ops.

### `LinalgOp` Interface

Most structured Linalg ops implement the Linalg structured interface. This lets
passes ask common questions:

- how many loops does the op have?
- which loops are parallel?
- which loops are reductions?
- what are the indexing maps?
- which operands are inputs?
- which operands are destinations?
- does the payload use a destination value?

This interface is why generic transformations can work on many different
Linalg ops.

### Contraction And Convolution Interfaces

Contraction ops such as `linalg.matmul`, `linalg.batch_matmul`, and
`linalg.contract` implement contraction-related interfaces. This is important
for vectorization and matmul-specific rewrites.

Convolution ops implement convolution-related interfaces. This lets passes
recognize image/filter/output structure without hard-coding every operation
name.

## Transformations

### Form Conversion

Linalg has several ways to represent the same computation:

```text
named op <-> category op <-> linalg.generic
```

Important passes:

| Pass | Meaning |
| --- | --- |
| `linalg-morph-ops` | Converts Linalg ops between named, category, and generic forms. |
| `linalg-generalize-named-ops` | Deprecated compatibility path from named ops to `linalg.generic`. |
| `linalg-specialize-generic-ops` | Deprecated compatibility path that tries to recognize `linalg.generic` as named ops. |

Generalization is reliable because every named structured op can be described
as a generic structured op. Specialization is best-effort because not every
generic op matches a known named op.

### Loop Lowering

Linalg can lower to loops once it has buffer semantics.

| Pass | Meaning |
| --- | --- |
| `convert-linalg-to-loops` | Lowers Linalg ops to `scf.for` loops. |
| `convert-linalg-to-parallel-loops` | Lowers parallel iterator dimensions to `scf.parallel` where possible. |
| `convert-linalg-to-affine-loops` | Lowers Linalg ops to `affine.for` style loops. |

The important precondition is buffer semantics. Tensor operands and tensor
results usually need bufferization before loop lowering.

### Elementwise Conversion And Fusion

| Pass | Meaning |
| --- | --- |
| `convert-elementwise-to-linalg` | Converts ranked tensor elementwise ops with the `ElementwiseMappable` trait to Linalg. |
| `linalg-fuse-elementwise-ops` | Fuses elementwise Linalg ops on tensors. |
| `linalg-fold-into-elementwise` | Folds `linalg.transpose` and `linalg.broadcast` producers into `linalg.elementwise` indexing maps when legal. |
| `linalg-inline-scalar-operands` | Inlines scalar operands into `linalg.generic` bodies. |

These passes are common in tensor pipelines where many small elementwise
operations should be combined before bufferization or vectorization.

### Shape And Layout Simplification

| Pass | Meaning |
| --- | --- |
| `linalg-fold-unit-extent-dims` | Removes unit-extent dimensions from Linalg ops on tensors. |
| `linalg-block-pack-matmul` | Packs matmul into blocked layouts, uses `linalg.mmt4d`, then unpacks back. |
| `simplify-depthwise-conv` | Simplifies depthwise convolution forms. |

`linalg.pack` and `linalg.unpack` are central to layout changes. They make
tiled layouts explicit so later transformations can reason about them.

### Aggregate And Decomposition Patterns

Some ops are intentionally aggregate forms. For example:

- `linalg.softmax`
- Winograd transform ops
- selected pack/unpack patterns
- selected convolution rewrites

These are usually decomposed by pattern-driven transformations or transform
dialect ops rather than a single public pass with the exact op name.

### Transform Dialect Extension

Linalg also provides many transform dialect operations under names such as:

- `transform.structured.match`
- `transform.structured.tile_using_for`
- `transform.structured.tile_using_forall`
- `transform.structured.fuse`
- `transform.structured.generalize`
- `transform.structured.specialize`
- `transform.structured.vectorize`
- `transform.structured.pack`
- `transform.structured.pad`
- `transform.structured.lower_pack`
- `transform.structured.lower_unpack`
- `transform.structured.split_reduction`
- `transform.structured.convert_conv2d_to_img2col`
- `transform.structured.winograd_conv2d`

These are not `linalg` dialect operations. They are transform dialect ops that
target Linalg payload operations. They matter when a pipeline uses explicit
transform scripts instead of only pass pipelines.

## Conversions / Lowering Paths

### Into Linalg

Common paths into Linalg:

| Pass | Direction |
| --- | --- |
| `convert-elementwise-to-linalg` | Ranked tensor elementwise ops to `linalg.generic`. |
| `convert-tensor-to-linalg` | Some Tensor dialect ops, notably tensor padding patterns, to Linalg-oriented IR. |
| `tosa-to-linalg` | TOSA ops to tensor and Linalg operations. |
| `tosa-to-linalg-named` | TOSA ops to Linalg named operations where possible. |

This is why Linalg often appears after model-level dialects.

### Linalg To Loops

After bufferization, Linalg can lower to loops:

```text
linalg.matmul on memrefs
  -> scf.for / scf.parallel / affine.for
  -> scalar arith + memref.load + memref.store
```

The lowering inlines the Linalg region into the innermost loop and replaces
`linalg.index` with the corresponding induction variable.

### Linalg To Library Calls

`convert-linalg-to-std` lowers Linalg ops with library call names to `func.call`.

The conversion canonicalizes memref layouts in function signatures because
library-call ABIs do not use MLIR's full layout information directly.

This path is different from loop lowering:

- loop lowering expands the computation
- library-call lowering delegates the computation

### Linalg To Vector And GPU-Oriented Forms

Vectorization, tiling, distribution, and GPU mapping are mostly implemented as
patterns and transform dialect operations rather than one simple public
conversion pass.

Conceptually:

```text
linalg structured op
  -> tiled linalg/scf form
  -> vector operations
  -> gpu or target-specific dialects
```

The reason this works is that Linalg exposes iterator types, indexing maps, and
contraction/convolution interfaces.

## Example IR

### Tensor Matmul

```mlir
func.func @matmul_tensor(%a : tensor<4x8xf32>,
                         %b : tensor<8x16xf32>) -> tensor<4x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x16xf32>
  %init = linalg.fill ins(%zero : f32)
                      outs(%empty : tensor<4x16xf32>)
                      -> tensor<4x16xf32>
  %result = linalg.matmul
      ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%init : tensor<4x16xf32>)
      -> tensor<4x16xf32>
  return %result : tensor<4x16xf32>
}
```

This is tensor-value Linalg. The destination tensor initializes the result, and
the op returns a new tensor value.

### Generic Elementwise Add

```mlir
#id2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @generic_add(%lhs : tensor<?x?xf32>,
                       %rhs : tensor<?x?xf32>,
                       %out : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.generic {
      indexing_maps = [#id2, #id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%out : tensor<?x?xf32>) {
    ^bb0(%a : f32, %b : f32, %old : f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
```

The indexing maps are identity maps, so every result element reads the matching
element from both inputs.

### Buffer Matmul

```mlir
func.func @matmul_buffer(%a : memref<4x8xf32>,
                         %b : memref<8x16xf32>,
                         %c : memref<4x16xf32>) {
  linalg.matmul
      ins(%a, %b : memref<4x8xf32>, memref<8x16xf32>)
      outs(%c : memref<4x16xf32>)
  return
}
```

This is buffer Linalg. It can lower directly to loops because the destination is
a memref that can be loaded and stored.

### Packing A Tensor

```mlir
func.func @pack_tensor(%src : tensor<16x32xf32>,
                       %dst : tensor<2x4x8x8xf32>)
    -> tensor<2x4x8x8xf32> {
  %packed = linalg.pack %src
      inner_dims_pos = [0, 1]
      inner_tiles = [8, 8]
      into %dst : tensor<16x32xf32> -> tensor<2x4x8x8xf32>
  return %packed : tensor<2x4x8x8xf32>
}
```

This makes a blocked layout explicit. The original `16x32` shape becomes
`2x4x8x8`, where the last two dimensions are tile dimensions.

### Softmax

```mlir
func.func @softmax(%input : tensor<2x4xf32>,
                   %out : tensor<2x4xf32>) -> tensor<2x4xf32> {
  %result = linalg.softmax dimension(1)
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
```

`linalg.softmax` is an aggregate op. It represents a useful high-level pattern,
but it can be decomposed into smaller structured operations.

## Mental Model

Read a Linalg op in this order:

1. Look at `ins` and `outs`.
2. Ask whether the op is tensor-value form or memref-buffer form.
3. Look at iterator types to see which loops are parallel and which reduce.
4. Look at indexing maps to see how each loop indexes each operand.
5. Look at the scalar region or named op to see the element computation.

For named ops, the first pass is easier:

```text
linalg.matmul means a contraction
linalg.add means elementwise add
linalg.pooling_nhwc_max means max pooling in NHWC layout
```

For optimization, the named op can be generalized:

```text
linalg.matmul -> linalg.generic
```

Then later it can be tiled, fused, vectorized, bufferized, or lowered to loops.

## Gotchas

- `linalg.transpose` moves data. `memref.transpose` is a metadata view change.
- Tensor Linalg and memref Linalg have different lowering options. Loop lowering
  expects buffer semantics.
- `outs` are not just outputs. They are destination operands and may be read as
  initial accumulator values.
- `linalg.generic` is powerful, but not every generic op can be recognized as a
  named op.
- Indexing maps are part of the computation. Changing a map can change the
  meaning as much as changing the scalar body.
- Iterator types matter. Marking a reduction dimension as parallel is a
  semantic bug.
- `convert-linalg-to-std` is for library-call lowering, not the normal
  structured loop expansion path.
- Named op layout suffixes are meaningful. `nchw_fchw` and `nhwc_hwcf` are
  different data layouts.
- Pack/unpack can imply padding behavior. Missing padding values can be
  undefined behavior when tile sizes do not divide dimensions evenly.
- The transform dialect operations that target Linalg are not themselves
  Linalg dialect ops.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/Linalg/IR/LinalgBase.td`
- `mlir/include/mlir/Dialect/Linalg/IR/LinalgDoc.td`
- `mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td`
- `mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td`
- `mlir/include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml`
- `mlir/include/mlir/Dialect/Linalg/IR/LinalgRelayoutOps.td`
- `mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.td`
- `mlir/include/mlir/Dialect/Linalg/IR/LinalgEnums.td`
- `mlir/lib/Dialect/Linalg/IR/LinalgDialect.cpp`
- `mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp`
- `mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp`

Transform source files:

- `mlir/include/mlir/Dialect/Linalg/Passes.td`
- `mlir/lib/Dialect/Linalg/Transforms/Loops.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/Generalization.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/FoldIntoElementwise.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/DropUnitDims.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/BlockPackMatmul.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/SimplifyDepthwiseConv.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/Vectorization.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp`
- `mlir/lib/Dialect/Linalg/Transforms/Fusion.cpp`

Conversion source files:

- `mlir/include/mlir/Conversion/LinalgToStandard/LinalgToStandard.h`
- `mlir/lib/Conversion/LinalgToStandard/LinalgToStandard.cpp`
- `mlir/include/mlir/Conversion/TensorToLinalg/TensorToLinalg.h`
- `mlir/lib/Conversion/TensorToLinalg/TensorToLinalg.cpp`
- `mlir/lib/Conversion/TensorToLinalg/TensorToLinalgPass.cpp`
- `mlir/include/mlir/Conversion/TosaToLinalg/TosaToLinalg.h`
- `mlir/lib/Conversion/TosaToLinalg/TosaToLinalg.cpp`
- `mlir/lib/Conversion/TosaToLinalg/TosaToLinalgNamed.cpp`
- `mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp`

Transform dialect extension files:

- `mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td`
- `mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.td`
- `mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp`
- `mlir/lib/Dialect/Linalg/TransformOps/LinalgMatchOps.cpp`

Useful tests:

- `mlir/test/Dialect/Linalg/roundtrip.mlir`
- `mlir/test/Dialect/Linalg/named-ops.mlir`
- `mlir/test/Dialect/Linalg/generalize-named-ops.mlir`
- `mlir/test/Dialect/Linalg/linalg-morph-category-ops.mlir`
- `mlir/test/Dialect/Linalg/loops.mlir`
- `mlir/test/Dialect/Linalg/convert-elementwise-to-linalg.mlir`
- `mlir/test/Dialect/Linalg/decompose-ops.mlir`
- `mlir/test/Dialect/Linalg/simplify-depthwise-conv.mlir`
- `mlir/test/Dialect/Linalg/block-pack-matmul.mlir`
- `mlir/test/Dialect/Linalg/transform-ops.mlir`
- `mlir/test/Conversion/TensorToLinalg/tensor-ops-to-linalg.mlir`
- `mlir/test/Conversion/TosaToLinalg/tosa-to-linalg.mlir`
- `mlir/test/Conversion/TosaToLinalg/tosa-to-linalg-named.mlir`
