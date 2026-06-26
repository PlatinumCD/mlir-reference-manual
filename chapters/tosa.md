# TOSA Dialect

## Beginner Summary

The `tosa` dialect implements the Tensor Operator Set Architecture in MLIR.

TOSA is a portable operator set for machine-learning graphs. It represents
whole-tensor operations such as elementwise arithmetic, convolutions, pooling,
matmul, reshapes, reductions, control flow, variables, and quantization-related
scaling.

For beginners, `tosa` is a model-level dialect. It is usually higher level than
`linalg`, `tensor`, `scf`, `arith`, and `memref`. A compiler can import a neural
network into TOSA, validate it against a target profile, run shape and
decomposition passes, and then lower it into implementation dialects.

## Why This Dialect Exists

Machine-learning frameworks all have their own graph formats and operator
sets. Backends also differ: CPUs, GPUs, NPUs, DSPs, and embedded accelerators
support different subsets and data types.

TOSA gives MLIR a standardized middle layer:

```text
framework graph
  -> TOSA operators with defined semantics
  -> validation, shape inference, decomposition, quantization preparation
  -> linalg/tensor/arith/scf or SPIR-V Graph/TOSA
```

The key goal is portability. TOSA defines what an operation means before the
compiler chooses a target-specific implementation.

## When It Matters

The `tosa` dialect matters when MLIR is compiling a neural-network style graph.

It appears when:

- Importing models from ML frameworks.
- Representing portable whole-tensor ML operations.
- Preserving quantization information.
- Validating that a model fits a TOSA profile, level, or extension set.
- Inferring and cleaning up tensor shapes.
- Decomposing high-level TOSA ops into simpler TOSA ops before lowering.
- Lowering to `linalg`, `tensor`, `arith`, `scf`, `ml_program`, or
  SPIR-V Graph/TOSA.

It usually disappears before final code generation. Backends typically do not
execute generic TOSA directly; they lower it to dialects closer to loops,
buffers, target graphs, or hardware instructions.

## When To Use It

Use `tosa` when the IR should describe an ML graph in portable operator terms.

Good uses include:

- Frontend import for neural-network models.
- Graph-level validation against TOSA profiles and extensions.
- Keeping convolution, pooling, matmul, reshape, and quantization semantics
  explicit before lowering.
- Expressing persistent model variables before conversion to `ml_program`.
- Preserving shape operations in `!tosa.shape` form.
- Lowering a graph to SPIR-V Graph/TOSA instead of immediately lowering to
  loops.

Avoid using TOSA for arbitrary numerical programs that are not naturally
operator-graph programs. If the program is already loop-level or buffer-level,
`linalg`, `scf`, `vector`, and `memref` are usually better fits.

## Core Concepts

### Whole-Tensor Operations

Most TOSA ops operate on entire tensors:

```text
tosa.add: tensor + tensor -> tensor
tosa.conv2d: input, weights, bias -> output
tosa.reshape: tensor, shape -> tensor
tosa.reduce_sum: tensor -> reduced tensor
```

This is different from `arith`, which is scalar or elementwise at a lower level,
and different from `linalg`, which describes computation with indexing maps and
iteration spaces.

### Shapes Are First-Class

TOSA has a shape type, `!tosa.shape<N>`, and a family of shape operations.

Examples include:

- `tosa.const_shape`
- `tosa.add_shape`
- `tosa.mul_shape`
- `tosa.slice_shape`
- `tosa.assert_equal_shape`

These are used by shape inference and dynamic-shape lowering.

### Quantization Is Explicit

TOSA models quantized computation with explicit scaling and zero-point-related
attributes or helper ops.

Important examples:

- `tosa.apply_scale`
- `tosa.rescale`
- convolution and matmul quantization attributes
- padding quantization attributes

This is one reason TOSA is useful for ML compilers: numerical behavior is part
of the IR, not just a backend accident.

### Profiles, Levels, And Extensions

TOSA validation is target-aware. A module can carry a TOSA target environment
describing:

- Specification version.
- Level.
- Profiles.
- Extensions.

The `tosa-attach-target` pass attaches this information. The `tosa-validate`
pass checks operations against it.

## Operations

The current TOSA dialect in this LLVM checkout defines 102 generated operations.

### Constants, Identity, Casting, And Custom Hooks

These ops create constants, preserve values, convert element types, or represent
backend-specific hooks:

- `tosa.const`
- `tosa.const_shape`
- `tosa.identity`
- `tosa.cast`
- `tosa.custom`

Block-scaled casting forms:

- `tosa.cast_to_block_scaled`
- `tosa.cast_from_block_scaled`

### Elementwise Numeric Operations

Unary and binary numeric tensor operations include:

- `tosa.abs`
- `tosa.add`
- `tosa.sub`
- `tosa.mul`
- `tosa.negate`
- `tosa.intdiv`
- `tosa.pow`
- `tosa.reciprocal`
- `tosa.rsqrt`
- `tosa.exp`
- `tosa.log`
- `tosa.sin`
- `tosa.cos`
- `tosa.tanh`
- `tosa.sigmoid`
- `tosa.erf`
- `tosa.ceil`
- `tosa.floor`
- `tosa.clamp`
- `tosa.clz`

Quantized scaling:

- `tosa.apply_scale`
- `tosa.rescale`

### Bitwise, Logical, Comparison, And Select

Integer and boolean-style operations include:

- `tosa.bitwise_and`
- `tosa.bitwise_or`
- `tosa.bitwise_xor`
- `tosa.bitwise_not`
- `tosa.logical_and`
- `tosa.logical_or`
- `tosa.logical_xor`
- `tosa.logical_not`
- `tosa.logical_left_shift`
- `tosa.logical_right_shift`
- `tosa.arithmetic_right_shift`
- `tosa.equal`
- `tosa.greater`
- `tosa.greater_equal`
- `tosa.maximum`
- `tosa.minimum`
- `tosa.select`

### Tensor Data Movement And Shape-Changing Ops

These ops rearrange or select tensor data:

- `tosa.concat`
- `tosa.reshape`
- `tosa.pad`
- `tosa.slice`
- `tosa.tile`
- `tosa.transpose`
- `tosa.reverse`
- `tosa.resize`
- `tosa.gather`
- `tosa.scatter`
- `tosa.row_gather`

Block-scaled movement forms:

- `tosa.reshape_block_scaled`
- `tosa.row_gather_block_scaled`

Shape query:

- `tosa.dim`

### Reductions And Argmax

Reduction operations collapse one tensor dimension while preserving rank in the
TOSA style:

- `tosa.reduce_all`
- `tosa.reduce_any`
- `tosa.reduce_max`
- `tosa.reduce_min`
- `tosa.reduce_product`
- `tosa.reduce_sum`
- `tosa.argmax`

### Neural-Network Layer Operations

Convolution, pooling, matmul, and transform-style ML operations include:

- `tosa.conv2d`
- `tosa.conv3d`
- `tosa.depthwise_conv2d`
- `tosa.transpose_conv2d`
- `tosa.conv2d_block_scaled`
- `tosa.matmul`
- `tosa.matmul_t`
- `tosa.matmul_t_block_scaled`
- `tosa.avg_pool2d`
- `tosa.max_pool2d`
- `tosa.avg_pool2d_adaptive`
- `tosa.max_pool2d_adaptive`

Frequency-domain operations:

- `tosa.fft2d`
- `tosa.rfft2d`

Lookup/table operation:

- `tosa.table`

### Shape Operations

TOSA shape operations work on `!tosa.shape` values:

- `tosa.add_shape`
- `tosa.sub_shape`
- `tosa.mul_shape`
- `tosa.div_ceil_shape`
- `tosa.div_floor_shape`
- `tosa.mod_shape`
- `tosa.max_shape`
- `tosa.min_shape`
- `tosa.concat_shape`
- `tosa.slice_shape`
- `tosa.exp2_shape`
- `tosa.log2_ceil_shape`
- `tosa.log2_floor_shape`
- `tosa.assert_equal_shape`

These are important for dynamic-shape models and shape inference.

### Variables And Control Flow

TOSA includes graph-level mutable state and structured control flow:

- `tosa.variable`
- `tosa.variable_read`
- `tosa.variable_write`
- `tosa.cond_if`
- `tosa.while_loop`
- `tosa.yield`

These are not the same as low-level `memref` mutation or `cf` branches. They are
model-level state and control constructs.

## Transformations

TOSA has several native passes that prepare, validate, and simplify model-level
IR before conversion.

### Native TOSA Passes

`tosa-layerwise-constant-fold`
: Folds whole-layer operations on constant tensors. It has an
  `aggressive-reduce-constant` option.

`tosa-infer-shapes`
: Propagates shapes across TOSA operations and can legalize rankless or dynamic
  shapes toward static shapes. Options include `fold-shape-expressions` and
  `convert-function-boundaries`.

`tosa-make-broadcastable`
: Inserts reshapes that prepend unit dimensions so operands can broadcast.

`tosa-optional-decompositions`
: Applies optional decompositions exposed by the TOSA transform library.

`tosa-validate`
: Validates TOSA operations against specification-related criteria such as
  profile, level, and datatype combinations.

`tosa-reduce-transposes`
: Pushes and folds `tosa.transpose` through operation chains to remove layout
  churn, especially common framework layout conversions.

`tosa-arith-const-to-tosa-const`
: Converts tensor-valued `arith.constant` operations to `tosa.const`.

`tosa-convert-integer-type-to-signless`
: Converts signed or unsigned integer tensor types to signless integer types.

`tosa-attach-target`
: Attaches a `tosa.target_env` module attribute describing specification
  version, level, profiles, and extensions.

`tosa-downgrade-1-1-to-1-0`
: Best-effort downgrade from TOSA 1.1 constructs to TOSA 1.0 equivalents.

`tosa-narrow-i64-to-i32`
: Rewrites 64-bit integer TOSA tensor operations to 32-bit integer operations
  for targets without the relevant extension.

`tosa-narrow-f64-to-f32`
: Rewrites 64-bit floating-point TOSA tensor operations to 32-bit floating-point
  operations.

`tosa-experimental-input-shape`
: Overrides dynamic function argument shapes with specified static shapes.

### Common Preparation Strategy

A practical TOSA preparation sequence often looks like:

```text
tosa-attach-target
tosa-validate
tosa-infer-shapes
tosa-make-broadcastable
tosa-optional-decompositions
tosa-layerwise-constant-fold
tosa-reduce-transposes
```

The exact order depends on the frontend and backend. The important idea is that
TOSA is usually cleaned up and validated before it is converted away.

## Conversions And Lowering Paths

### To Arith

`tosa-to-arith` lowers selected TOSA operations to `arith`.

It is mainly useful for scalar-like helper operations and scaling forms. Options
include:

- `include-apply-rescale`
- `use-32-bit`

### To Linalg

`tosa-to-linalg` lowers TOSA operations to `linalg` on tensors.

This is the main path for turning whole-tensor TOSA graph ops into structured
tensor computation that can later be tiled, bufferized, vectorized, and lowered.

Options include:

- `disable-tosa-decompositions`
- `aggressive-reduce-constant`

The registered `tosa-to-linalg-pipeline` composes several TOSA preparation
passes with TOSA-to-Linalg conversion.

### To Linalg Named Ops

`tosa-to-linalg-named` lowers selected neural-network operations to named
`linalg` operations.

The pass targets operations such as:

- `tosa.conv2d`
- `tosa.conv3d`
- `tosa.depthwise_conv2d`
- `tosa.max_pool2d`
- `tosa.avg_pool2d`
- `tosa.matmul`
- `tosa.transpose`

It has a `prefer-conv2d-kernel-layout-hwcf` option for convolution layout
choice.

### To Tensor

`tosa-to-tensor` lowers TOSA shape/data movement ops to the `tensor` dialect.

The pass targets operations such as:

- `tosa.concat`
- `tosa.reshape`
- `tosa.slice`
- `tosa.pad`

This is useful because these are structural tensor transformations rather than
numeric kernels.

### To SCF

`tosa-to-scf` lowers TOSA control-flow operations to `scf`.

It is the path for model-level control flow such as `tosa.cond_if` and
`tosa.while_loop`.

### To MLProgram

`tosa-to-mlprogram` lowers TOSA variable operations to the `ml_program` dialect.

It targets:

- `tosa.variable`
- `tosa.variable_read`
- `tosa.variable_write`

Use this when persistent model state should become MLProgram globals and
load/store-like operations.

### To SPIR-V Graph/TOSA

`tosa-to-spirv-tosa` lowers TOSA IR to the SPIR-V Graph/TOSA representation.

It wraps converted functions in SPIR-V graph structure, lowers supported TOSA
ops to `spirv.Tosa.*`, and rewrites TOSA tensor and shape types to SPIR-V ARM
tensor types.

Related pass:

- `tosa-to-spirv-tosa-mark-graph-constants`

That pass marks large `tosa.const` and `tosa.const_shape` operations so the
SPIR-V Graph/TOSA conversion lowers them as graph constants instead of inlining
them as ordinary SPIR-V constants.

## Example IR

### Elementwise Tensor Operation

```mlir
func.func @elementwise(
    %a: tensor<4xf32>,
    %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.add %a, %b
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
```

This is whole-tensor addition. It does not say how to loop over the elements.

### Constant Shape And Reshape

```mlir
func.func @const_and_reshape() -> tensor<2x2xf32> {
  %cst = "tosa.const"() {
    values = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  } : () -> tensor<4xf32>
  %shape = "tosa.const_shape"() {
    values = dense<[2, 2]> : tensor<2xindex>
  } : () -> !tosa.shape<2>
  %r = tosa.reshape %cst, %shape
      : (tensor<4xf32>, !tosa.shape<2>) -> tensor<2x2xf32>
  func.return %r : tensor<2x2xf32>
}
```

TOSA reshape takes an explicit shape value.

### Reduction

```mlir
func.func @reduce(%x: tensor<2x4xf32>) -> tensor<2x1xf32> {
  %r = tosa.reduce_sum %x {axis = 1 : i32}
      : (tensor<2x4xf32>) -> tensor<2x1xf32>
  func.return %r : tensor<2x1xf32>
}
```

TOSA reductions keep the result ranked.

## Mental Model

Think of TOSA as "portable ML graph IR."

It is not a loop dialect. It is not a buffer dialect. It is not a hardware
instruction dialect.

It says:

```text
this model performs these tensor operations
with these shapes, data types, and quantization semantics
under these profile and target constraints
```

Lowering decides how to implement those operations.

## Gotchas

TOSA ops are whole-tensor ops.

Do not expect `tosa.add` to expose loops. Lower to `linalg` or another
implementation dialect when loop structure is needed.

TOSA shape values use `!tosa.shape`.

Some operations take shape operands instead of shape attributes. Shape inference
and shape folding passes are important before lowering.

Not every op lowers through the same path.

Data movement may lower to `tensor`, numeric ops may lower to `linalg` or
`arith`, control flow may lower to `scf`, variables may lower to `ml_program`,
and graph deployment may lower to SPIR-V Graph/TOSA.

Validation matters.

A TOSA module may be syntactically valid MLIR but invalid for a specific TOSA
profile, level, extension set, or datatype combination. Use `tosa-attach-target`
and `tosa-validate` when target conformance matters.

Some ops carry quantization-specific operands or attributes.

Do not assume that an op's apparent mathematical name fully explains its
operands. In this checkout, several operations carry explicit quantization or
scale-related information.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/Tosa/IR/TosaOpBase.td`
- `mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td`
- `mlir/include/mlir/Dialect/Tosa/IR/TosaShapeOps.td`
- `mlir/include/mlir/Dialect/Tosa/IR/TosaUtilOps.td`
- `mlir/include/mlir/Dialect/Tosa/IR/TosaInterfaces.td`
- `mlir/lib/Dialect/Tosa/IR/TosaOps.cpp`
- `mlir/lib/Dialect/Tosa/IR/TosaCanonicalizations.cpp`
- `mlir/include/mlir/Dialect/Tosa/Transforms/Passes.td`
- `mlir/lib/Dialect/Tosa/Transforms/`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/TosaToArith/`
- `mlir/lib/Conversion/TosaToLinalg/`
- `mlir/lib/Conversion/TosaToMLProgram/`
- `mlir/lib/Conversion/TosaToSCF/`
- `mlir/lib/Conversion/TosaToSPIRVTosa/`
- `mlir/lib/Conversion/TosaToTensor/`

Generated op documentation source:

```text
mlir-tblgen --gen-op-doc -dialect=tosa \
  mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td
```
