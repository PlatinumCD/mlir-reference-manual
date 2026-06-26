# StableHLO stablehlo Dialect

## Beginner Summary

The StableHLO `stablehlo` dialect is a portable tensor IR for machine learning
programs.

StableHLO is based on XLA HLO, but its purpose is different from an internal
optimizer IR. It is meant to be a stable contract between ML frameworks and ML
compilers. A frontend can lower a TensorFlow, JAX, PyTorch, or other ML graph
into StableHLO, serialize it, version it, and hand it to a compiler without
requiring that compiler to understand the original framework.

For a beginner, the key idea is:

```text
framework tensor program
  -> StableHLO portability layer
  -> compiler-specific lowering pipeline
  -> Linalg, TOSA, GPU, CPU, accelerator, or runtime code
```

StableHLO is important because ML graphs need more than generic arithmetic.
They need tensor shapes, broadcasts, convolutions, reductions, collectives,
random-number generation, quantization, side-effect tokens, custom calls, and
version compatibility. The dialect collects those ideas in one operation set.

## Why This Dialect Exists

Many ML frameworks can express similar computations, but they do not use the
same IR internally. Many compilers can optimize and execute those computations,
but they do not want to maintain a full importer for every framework.

The `stablehlo` dialect exists as the middle layer.

It gives frontends a place to express:

- elementwise tensor math;
- reshaping, slicing, gathering, scattering, padding, and broadcasting;
- reductions and higher-order region operations;
- common ML kernels such as convolution, dot, FFT, batch normalization, and
  triangular solve;
- communication across replicas and partitions;
- tokens for ordered side effects;
- custom calls for runtime-specific behavior;
- quantized tensor math;
- compatibility and versioning through VHLO.

It fills the gap between framework-specific graph IR and lower-level compiler
dialects such as `linalg`, `tensor`, `scf`, `arith`, `math`, `tosa`, `memref`,
and target-specific codegen dialects.

## When It Matters

StableHLO matters whenever the compiler boundary needs to be stable.

You will see it in import/export paths, model portability flows, ahead-of-time
compilation pipelines, and compiler test cases that want to describe ML
semantics without depending on a frontend. It is especially common around
OpenXLA, IREE, torch-mlir, and other MLIR-based ML compiler stacks.

A typical flow looks like this:

```text
framework graph
  -> CHLO or frontend-specific helper ops
  -> stablehlo
  -> StableHLO cleanup, shape refinement, compatibility, and quantization passes
  -> linalg, tosa, or compiler-specific tensor IR
  -> bufferization, loops, vectorization, target codegen
```

StableHLO is not usually the final execution dialect. It preserves ML semantics
long enough for a compiler to choose the right lowering strategy.

## When To Use It

Use `stablehlo` when you need a framework-neutral representation of tensor ML
computation.

Use it for:

- interchange between ML frameworks and ML compilers;
- model serialization with stable semantics;
- compiler pipelines that begin from XLA-style tensor operations;
- preserving high-level tensor operations before choosing a target lowering;
- testing ML compiler transformations without depending on frontend syntax;
- representing collectives, quantized operations, or custom calls that do not
  map cleanly to plain arithmetic.

Do not use it as a general replacement for all MLIR tensor code. If a program
is already in a compiler-internal phase where loops, buffers, memory spaces, or
target intrinsics matter more than framework portability, lower to the dialects
that model those concepts directly.

## Core Concepts

### Stable Tensor Semantics

Most StableHLO values are tensors. Operations describe what computation should
happen, not how to schedule it. For example, `stablehlo.add` means elementwise
addition over compatible tensor shapes; it does not say whether the target will
use a vector instruction, a GPU kernel, or a library call.

This makes the dialect a good boundary format. The receiving compiler can pick
the implementation strategy later.

### Static And Dynamic Shapes

StableHLO supports both static and dynamic tensor shapes. Some operations have
static forms, such as `stablehlo.reshape`, `stablehlo.slice`, and
`stablehlo.broadcast_in_dim`. Others carry dynamic shape operands, such as
`stablehlo.dynamic_reshape`, `stablehlo.dynamic_slice`,
`stablehlo.dynamic_pad`, and `stablehlo.dynamic_broadcast_in_dim`.

The transform pipeline can canonicalize dynamic operations to static operations
when shape operands are known constants.

### Regions

Several StableHLO operations contain regions. A region is a nested block of MLIR
that describes a computation used by the parent operation.

Examples:

- `stablehlo.reduce` has a reducer region;
- `stablehlo.reduce_window` has a reducer region;
- `stablehlo.sort` has a comparator region;
- `stablehlo.map` applies a region elementwise;
- `stablehlo.if`, `stablehlo.case`, and `stablehlo.while` hold control-flow
  bodies.

Inside these regions, `stablehlo.return` terminates the StableHLO region and
returns values to the parent operation.

### Tokens And Side Effects

Most tensor math is pure, but ML programs sometimes need ordered side effects:
input feeds, output feeds, sends, receives, and runtime calls. StableHLO models
ordering with `!stablehlo.token` and operations such as `stablehlo.after_all`,
`stablehlo.infeed`, `stablehlo.outfeed`, `stablehlo.send`, and
`stablehlo.recv`.

The token is not data being computed by the model. It is an ordering handle
that tells later passes which side effects must happen before others.

### Collectives And SPMD Meaning

StableHLO includes collective operations for distributed computation:
`stablehlo.all_reduce`, `stablehlo.all_gather`, `stablehlo.all_to_all`,
`stablehlo.reduce_scatter`, `stablehlo.collective_broadcast`, and
`stablehlo.collective_permute`.

These operations imply a parallel execution model. They do not merely compute a
local tensor result; they communicate between replicas or partitions according
to attributes such as replica groups, channels, and partition IDs.

### Compatibility And VHLO

StableHLO evolves. To keep old serialized programs usable and new serialized
programs understandable, the project also uses VHLO, a versioned form of HLO
operations. StableHLO can be legalized to VHLO for serialization, converted
between VHLO versions, and legalized back to StableHLO.

For users, this means the `stablehlo` dialect is both an operation set and a
compatibility promise.

## Types

StableHLO mostly uses MLIR tensor types, including quantized element types. The
dialect itself adds two important non-tensor types:

| Type | Meaning |
| --- | --- |
| `!stablehlo.token` | Ordering token for side-effecting operations. |
| `!stablehlo.future<...>` | Handle for async StableHLO results. |

## Important Attributes

Many StableHLO operations are compact because their detailed layout or algorithm
information is stored in attributes.

| Attribute family | Used for |
| --- | --- |
| `#stablehlo.gather` | Dimension numbering for `stablehlo.gather` and `stablehlo.dynamic_gather`. |
| `#stablehlo.scatter` | Dimension numbering for `stablehlo.scatter`. |
| `#stablehlo.dot` | Batching and contracting dimensions for `stablehlo.dot_general`. |
| `#stablehlo.dot_algorithm` | Precision and accumulation constraints for dot-like operations. |
| `#stablehlo.conv` | Input, kernel, and output dimension numbering for `stablehlo.convolution` and `stablehlo.dynamic_conv`. |
| `#stablehlo.channel_handle` | Communication channel identity for send, receive, and collective operations. |
| `#stablehlo.output_operand_alias` | Output-to-operand aliasing on `stablehlo.custom_call`. |
| `#stablehlo.type_extensions` | Bounds and StableHLO-specific tensor type properties. |
| `#stablehlo.result_accuracy` | Accuracy requests for transcendental math operations. |
| Mesh and axis attributes | Replica group and mesh-axis modeling for collective operations. |
| Enum attributes | Precision, FFT type, RNG distribution, RNG algorithm, comparison direction, comparison type, transpose flag, and custom-call API version. |

## Operation Inventory

The StableHLO source currently defines 114 `stablehlo` operations.

### Constants, Control, Tokens, Calls, And Tuples

| Operation | Purpose |
| --- | --- |
| `stablehlo.constant` | Create a tensor value from a literal attribute. |
| `stablehlo.create_token` | Create a token value; this is compatibility-era behavior. |
| `stablehlo.after_all` | Join zero or more tokens into one ordered token. |
| `stablehlo.optimization_barrier` | Prevent certain optimizations from moving values across the barrier. |
| `stablehlo.tuple` | Pack multiple values into a tuple. |
| `stablehlo.get_tuple_element` | Extract one element from a tuple. |
| `stablehlo.return` | Terminator for StableHLO regions. |
| `stablehlo.if` | Select between true and false regions. |
| `stablehlo.case` | Select one region from multiple branches by index. |
| `stablehlo.while` | Loop with condition and body regions. |
| `stablehlo.custom_call` | Call runtime- or backend-defined functionality. |
| `stablehlo.composite` | Wrap a higher-level composite operation with a decomposition function. |
| `stablehlo.infeed` | Read values from an external input feed, ordered by token. |
| `stablehlo.outfeed` | Write values to an external output feed, ordered by token. |
| `stablehlo.send` | Send values through a channel. |
| `stablehlo.recv` | Receive values through a channel. |
| `stablehlo.async_start` | Start an async computation and produce a future. |
| `stablehlo.async_done` | Resolve a future back to a tensor result. |

### Elementwise And Scalar-Like Tensor Math

| Operation | Purpose |
| --- | --- |
| `stablehlo.abs` | Elementwise absolute value. |
| `stablehlo.add` | Elementwise addition. |
| `stablehlo.and` | Elementwise bitwise or logical and. |
| `stablehlo.atan2` | Elementwise two-argument arctangent. |
| `stablehlo.bitcast_convert` | Reinterpret bits while converting tensor element type. |
| `stablehlo.cbrt` | Elementwise cube root. |
| `stablehlo.ceil` | Elementwise ceiling. |
| `stablehlo.clamp` | Clamp each element between minimum and maximum tensors. |
| `stablehlo.compare` | Elementwise comparison using a direction and comparison type. |
| `stablehlo.complex` | Build complex values from real and imaginary tensors. |
| `stablehlo.convert` | Convert each element to a different element type. |
| `stablehlo.cosine` | Elementwise cosine. |
| `stablehlo.count_leading_zeros` | Count leading zero bits elementwise. |
| `stablehlo.divide` | Elementwise division. |
| `stablehlo.exponential` | Elementwise exponential. |
| `stablehlo.exponential_minus_one` | Elementwise exp(x) - 1. |
| `stablehlo.floor` | Elementwise floor. |
| `stablehlo.imag` | Extract the imaginary part of complex values. |
| `stablehlo.is_finite` | Test whether floating-point values are finite. |
| `stablehlo.log` | Elementwise natural logarithm. |
| `stablehlo.log_plus_one` | Elementwise log(1 + x). |
| `stablehlo.logistic` | Elementwise logistic sigmoid. |
| `stablehlo.maximum` | Elementwise maximum. |
| `stablehlo.minimum` | Elementwise minimum. |
| `stablehlo.multiply` | Elementwise multiplication. |
| `stablehlo.negate` | Elementwise negation. |
| `stablehlo.not` | Elementwise bitwise or logical not. |
| `stablehlo.or` | Elementwise bitwise or logical or. |
| `stablehlo.popcnt` | Count set bits elementwise. |
| `stablehlo.power` | Elementwise power. |
| `stablehlo.real` | Extract the real part of complex values. |
| `stablehlo.reduce_precision` | Reduce floating-point mantissa and exponent precision. |
| `stablehlo.remainder` | Elementwise remainder. |
| `stablehlo.round_nearest_afz` | Round to nearest with away-from-zero tie behavior. |
| `stablehlo.round_nearest_even` | Round to nearest with ties to even. |
| `stablehlo.rsqrt` | Elementwise reciprocal square root. |
| `stablehlo.select` | Elementwise choose between two values using a predicate. |
| `stablehlo.shift_left` | Elementwise left shift. |
| `stablehlo.shift_right_arithmetic` | Elementwise arithmetic right shift. |
| `stablehlo.shift_right_logical` | Elementwise logical right shift. |
| `stablehlo.sign` | Elementwise sign. |
| `stablehlo.sine` | Elementwise sine. |
| `stablehlo.sqrt` | Elementwise square root. |
| `stablehlo.subtract` | Elementwise subtraction. |
| `stablehlo.tan` | Elementwise tangent. |
| `stablehlo.tanh` | Elementwise hyperbolic tangent. |
| `stablehlo.xor` | Elementwise bitwise or logical exclusive-or. |

### Shape, Indexing, Layout, And Tensor Movement

| Operation | Purpose |
| --- | --- |
| `stablehlo.broadcast` | Broadcast by adding dimensions at the front. |
| `stablehlo.broadcast_in_dim` | Broadcast into an explicitly mapped result shape. |
| `stablehlo.dynamic_broadcast_in_dim` | Broadcast to a shape provided by an operand. |
| `stablehlo.concatenate` | Concatenate tensors along one dimension. |
| `stablehlo.iota` | Create a tensor filled with increasing indices along one dimension. |
| `stablehlo.dynamic_iota` | Create an iota tensor with runtime-provided shape. |
| `stablehlo.reshape` | Change tensor shape without changing element order. |
| `stablehlo.dynamic_reshape` | Reshape using a runtime-provided shape. |
| `stablehlo.transpose` | Permute tensor dimensions. |
| `stablehlo.reverse` | Reverse elements along selected dimensions. |
| `stablehlo.slice` | Extract a static slice. |
| `stablehlo.dynamic_slice` | Extract a slice with runtime start indices. |
| `stablehlo.real_dynamic_slice` | Slice using start, limit, and stride operands. |
| `stablehlo.dynamic_update_slice` | Update a slice at runtime start indices. |
| `stablehlo.pad` | Pad a tensor with static edge and interior padding. |
| `stablehlo.dynamic_pad` | Pad a tensor with runtime padding values. |
| `stablehlo.gather` | Gather slices from an operand using index tensors. |
| `stablehlo.dynamic_gather` | Gather with runtime-provided slice sizes. |
| `stablehlo.scatter` | Scatter updates into an operand using index tensors. |
| `stablehlo.get_dimension_size` | Return one dimension size as a tensor scalar. |
| `stablehlo.set_dimension_size` | Refine or attach a dynamic dimension size. |
| `stablehlo.sort` | Sort tensors along a dimension using a comparator region. |
| `stablehlo.torch_index_select` | Compatibility operation for Torch-style index selection. |

### Reductions, Higher-Order Tensor Ops, And Windows

| Operation | Purpose |
| --- | --- |
| `stablehlo.map` | Apply a region elementwise to one or more tensors. |
| `stablehlo.reduce` | Reduce one or more tensors across selected dimensions. |
| `stablehlo.reduce_window` | Reduce values over sliding windows. |
| `stablehlo.select_and_scatter` | Select values from windows and scatter updates back. |

### ML Kernels, Linear Algebra, And Numeric Primitives

| Operation | Purpose |
| --- | --- |
| `stablehlo.batch_norm_grad` | Batch-normalization backward computation. |
| `stablehlo.batch_norm_inference` | Batch-normalization inference computation. |
| `stablehlo.batch_norm_training` | Batch-normalization training computation. |
| `stablehlo.cholesky` | Cholesky factorization. |
| `stablehlo.convolution` | General N-dimensional convolution. |
| `stablehlo.dynamic_conv` | Convolution with dynamic window or padding information. |
| `stablehlo.dot` | Basic dot product. |
| `stablehlo.dot_general` | Dot product with explicit batching and contracting dimensions. |
| `stablehlo.einsum` | Einstein-summation style tensor contraction. |
| `stablehlo.unary_einsum` | Unary einsum-style tensor transform. |
| `stablehlo.fft` | Fast Fourier transform. |
| `stablehlo.rng` | Random values from a distribution. |
| `stablehlo.rng_bit_generator` | Random bits and updated RNG state. |
| `stablehlo.triangular_solve` | Solve triangular linear systems. |

### Distribution And Collectives

| Operation | Purpose |
| --- | --- |
| `stablehlo.replica_id` | Return the current replica ID. |
| `stablehlo.partition_id` | Return the current partition ID. |
| `stablehlo.all_gather` | Gather values from participants into larger tensors. |
| `stablehlo.all_reduce` | Reduce values across participants. |
| `stablehlo.reduce_scatter` | Reduce across participants and scatter slices of the result. |
| `stablehlo.all_to_all` | Exchange split tensor pieces between participants. |
| `stablehlo.collective_broadcast` | Broadcast from source participants to others. |
| `stablehlo.collective_permute` | Send values according to source-target participant pairs. |
| `stablehlo.cross-replica-sum` | Compatibility operation for cross-replica summation. |

### Quantization

| Operation | Purpose |
| --- | --- |
| `stablehlo.uniform_quantize` | Convert floating-point tensor values to uniform quantized values. |
| `stablehlo.uniform_dequantize` | Convert uniform quantized values back to floating-point tensor values. |

## Transformations And Conversions

StableHLO has two kinds of compiler support:

- transformations that keep the program in StableHLO or VHLO while improving,
  checking, refining, or versioning it;
- conversions that lower StableHLO to another dialect or legalize another
  dialect into StableHLO.

### StableHLO Transform Passes

| Pass | What it does |
| --- | --- |
| `chlo-legalize-to-stablehlo` | Lowers CHLO helper operations to StableHLO and Shape operations. |
| `shape-legalize-to-stablehlo` | Legalizes shape-related operations into StableHLO. |
| `stablehlo-canonicalize-dynamism` | Rewrites dynamic StableHLO ops to static forms when dynamic operands are constants. |
| `stablehlo-check-shape-assertions` | Validates and removes `custom_call @shape_assertion` operations when shape constraints hold. |
| `stablehlo-compatibility-expander` | Decomposes newer StableHLO features into forms supported by an older target version. |
| `stablehlo-complex-math-expander` | Expands complex math operations into real-valued StableHLO math. |
| `stablehlo-convert-to-signless` | Converts signed or unsigned integer IR toward signless integer types. |
| `stablehlo-legalize-composite-to-call` | Replaces `stablehlo.composite` with calls to their decomposition functions. |
| `stablehlo-legalize-deprecated-ops` | Rewrites deprecated operations to long-term supported forms when possible. |
| `stablehlo-legalize-qdq-to-quantized-op` | Fuses dequantize-operation-quantize patterns into quantized StableHLO operations. |
| `stablehlo-legalize-quantized-op-to-qdq` | Decomposes quantized StableHLO operations into dequantize, floating-point operation, and quantize. |
| `stablehlo-legalize-quant-to-math` | Rewrites quantized operations into primitive integer and floating-point math. |
| `stablehlo-legalize-to-vhlo` | Converts StableHLO to versioned VHLO for compatibility and serialization workflows. |
| `stablehlo-refine-arguments` | Refines `main` function argument shapes using a provided type signature. |
| `stablehlo-refine-shapes` | Propagates refined shapes through a StableHLO module. |
| `vhlo-legalize-to-stablehlo` | Converts versioned VHLO back to StableHLO. |
| `vhlo-to-version` | Converts VHLO between target versions. |
| `stablehlo-wrap-in-composite` | Wraps selected StableHLO ops in `stablehlo.composite` with generated decomposition functions. |

### Optimization Passes

| Pass | What it does |
| --- | --- |
| `stablehlo-aggressive-folder` | Folds StableHLO operations into constants when profitable and allowed by element limits. |
| `stablehlo-aggressive-simplification` | Applies algebraic and structural simplifications such as removing identity broadcasts, reshapes, tuple round-trips, zero arithmetic, and unused reduce inputs. |
| `stablehlo-target-independent-optimization` | Runs the preferred combined target-independent StableHLO simplification and folding pipeline. |

### Conversion Passes

| Pass | What it does |
| --- | --- |
| `stablehlo-legalize-to-linalg` | Lowers StableHLO to Linalg, plus supporting dialects such as `bufferization`, `complex`, `math`, `memref`, `scf`, `shape`, and `sparse_tensor`. |
| `stablehlo-legalize-to-tosa` | Lowers StableHLO to TOSA. |
| `stablehlo-prepare-for-tosa` | Rewrites StableHLO into forms that are easier to lower to TOSA, such as simplifying some `dot_general` cases. |
| `stablehlo-quant-legalize-to-tosa-rescale` | Converts StableHLO quantized patterns to TOSA rescale-based forms. |
| `tosa-rescale-legalize-to-stablehlo` | Rewrites TOSA rescale operations into StableHLO primitive math. |

## How To Read StableHLO IR

Start with the result type. StableHLO operations are usually easier to
understand if you first ask, "what tensor shape and element type comes out?"

Then read the operation family:

- elementwise operations preserve shape and apply a scalar operation per
  element;
- shape movement operations preserve values but rearrange or select elements;
- reductions and windows contain regions that define the combining logic;
- collectives imply communication between participants;
- token-carrying operations imply ordered side effects;
- custom calls and composites mean some meaning is intentionally delegated.

Example:

```mlir
func.func @add_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<1.0> : tensor<4xf32>
  %0 = stablehlo.add %arg0, %cst : tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

This says the program creates a constant tensor, adds it elementwise to the
argument, and returns the result. It says nothing about where the tensor is
stored or what target instruction implements the addition.

## What It Implies

Choosing StableHLO implies that portability and stable semantics matter more
than immediate target details.

It also implies that later compiler stages must still make major decisions:

- how to lower tensor operations to loops, libraries, or kernels;
- how to specialize or preserve dynamic shapes;
- how to implement collectives on the target runtime;
- how to lower custom calls;
- whether quantized operations should remain quantized or decompose to math;
- whether VHLO versioning is needed for serialization.

StableHLO is therefore a boundary dialect. It is high-level enough to preserve
ML meaning and low-level enough for compiler pipelines to reason about concrete
tensor operations.

## Source Files Inspected

This chapter was written from the local StableHLO source:

- `stablehlo/stablehlo/dialect/StablehloOps.td`
- `stablehlo/stablehlo/dialect/StablehloAttrs.td`
- `stablehlo/stablehlo/dialect/StablehloEnums.td`
- `stablehlo/stablehlo/dialect/StablehloTypes.td`
- `stablehlo/stablehlo/transforms/Passes.td`
- `stablehlo/stablehlo/transforms/optimization/Passes.td`
- `stablehlo/stablehlo/conversions/linalg/transforms/Passes.td`
- `stablehlo/stablehlo/conversions/tosa/transforms/Passes.td`
