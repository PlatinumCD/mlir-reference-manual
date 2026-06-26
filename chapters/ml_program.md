# `ml_program` Dialect

## Beginner Summary

The `ml_program` dialect provides structural IR for compiled machine-learning
programs.

It does not define tensor math operations like convolution, matmul, or
activation functions. Instead, it defines program-level structure: functions,
graph subgraphs, globals, global loads and stores, returns, outputs, and tokens
for ordering side effects in graph regions.

Think of it as a model/program container dialect for ML compiler pipelines.

## Why This Dialect Exists

ML frameworks often have concepts that are bigger than a single tensor
operation:

- Program entry points.
- Externally supplied constants or parameters.
- Mutable model state.
- Global tensors.
- Graph-style regions where SSA dominance is different from ordinary CFG
  regions.
- Explicit ordering constraints for side effects inside graph regions.

The `ml_program` dialect gives those concepts a common MLIR representation
without baking in TensorFlow, PyTorch, JAX, or another frontend's exact
semantics.

The dialect source describes it as structural operations and types for defining
a compiled machine-learning program. It also notes that the dialect is under
active development, so users should treat it as less stable than older core
dialects.

## When It Matters

`ml_program` matters when the compiler needs to preserve model-level structure:

- Frontends importing ML programs with global parameters or state.
- Pipelines that need module-level tensors or external weights.
- Graph import paths where ordering is not implicit in block order.
- ML runtime integration where some values are resolved externally.
- Bufferization of global tensor state into `memref.global`.

It is less important in a low-level loop pipeline that has already erased all
model-level structure.

## When To Use It

Use `ml_program` when you need program structure for an ML model but do not want
to define a full frontend-specific dialect.

Use it for:

- ML entry-point functions.
- Graph-like callable regions.
- Immutable and mutable global values.
- Externally resolved global contents.
- Ordered global loads/stores inside graph regions.

Do not use it for the math inside the model. Use dialects such as `linalg`,
`tensor`, `tosa`, `arith`, or frontend-specific dialects for computation.

## Core Concepts

### SSACFG Functions

`ml_program.func` is a function-like op with an SSACFG region. It behaves like a
simple callable symbol and terminates with `ml_program.return`.

Use it when ordinary SSA dominance and control-flow structure are appropriate.

### Graph Subgraphs

`ml_program.subgraph` is also function-like, but its body is a single-block
Graph region. It terminates with `ml_program.output`.

Graph regions are useful for ML graph representations where operation ordering
is not purely textual. Side effects in graph regions need explicit ordering.

### Globals

`ml_program.global` declares a module-level value. A global can be immutable or
mutable. Immutable globals must have an initial value. Mutable globals may have
an initial value, an external value marker, or be left undefined.

Globals are symbols, and load/store operations refer to them by symbol.

### Extern Attribute

`#ml_program.extern` marks a global whose actual value should be resolved
externally. The global symbol name is the lookup key.

This is useful for weights, parameters, or runtime-provided state.

### Tokens

`!ml_program.token` and `ml_program.token` are used to order side-effecting
operations in graph regions. Graph variants of load/store consume and produce
tokens so dependencies are explicit.

## Operations

The dialect has 11 operations.

| Operation | Purpose |
| --- | --- |
| `ml_program.func` | Function-like callable with an SSACFG region. |
| `ml_program.return` | Terminator for `ml_program.func`. |
| `ml_program.subgraph` | Function-like callable with a Graph region. |
| `ml_program.output` | Terminator for `ml_program.subgraph`. |
| `ml_program.global` | Declares an immutable or mutable module-level value. |
| `ml_program.global_load` | Side-effecting load from a mutable global in ordinary regions. |
| `ml_program.global_load_const` | Pure load from an immutable global. |
| `ml_program.global_store` | Side-effecting store to a mutable global in ordinary regions. |
| `ml_program.global_load_graph` | Token-ordered load from a mutable global in graph regions. |
| `ml_program.global_store_graph` | Token-ordered store to a mutable global in graph regions. |
| `ml_program.token` | Produces a token for graph-side-effect ordering. |

### Function And Subgraph Ops

`ml_program.func` and `ml_program.subgraph` both implement function-like
interfaces and are symbols.

The difference is region kind:

- `ml_program.func` contains an SSACFG region and uses `ml_program.return`.
- `ml_program.subgraph` contains a Graph region and uses `ml_program.output`.

Both terminators verify that operand count and operand types match the callable
signature.

### Global Ops

`ml_program.global` has:

- A symbol name.
- A type.
- Optional `mutable`.
- Optional initial value.
- Optional symbol visibility.

Use `ml_program.global_load_const` only for immutable globals. Use
`ml_program.global_load` and `ml_program.global_store` for mutable globals in
ordinary regions.

### Graph Global Ops

`ml_program.global_load_graph` and `ml_program.global_store_graph` exist because
ordinary side-effecting loads and stores may not be sufficiently ordered inside
Graph regions.

The graph forms consume zero or more tokens and produce a token. This makes
ordering dependencies explicit.

## Types And Attributes

| Name | Kind | Meaning |
| --- | --- | --- |
| `!ml_program.token` | Type | Ordering token for side-effecting graph operations. |
| `#ml_program.extern` | Attribute | Externally resolved global value marker. |

Example external global:

```mlir
ml_program.global private mutable @weights(#ml_program.extern : tensor<4xi32>)
  : tensor<?xi32>
```

## Transformations

The dialect defines one dedicated pass:

```text
-mlprogram-pipeline-globals
```

This pass optimizes `ml_program` global loads and stores. It can remove
redundant loads and stores when the value is already known in IR. The pass is
designed to handle nested regions and function calls safely.

Common effects include:

- Reusing a previous `ml_program.global_load` when the global has not been
  overwritten.
- Removing redundant consecutive stores of the same known value.
- Preserving correctness when an intervening region or call may affect global
  state.

## Conversions And Lowering Paths

`ml_program` does not have a broad "convert-ml-program-to-X" pass in this
checkout.

The important lowering path is through One-Shot Bufferize. The dialect provides
bufferization interface implementations so tensor globals can lower toward
memref globals and accesses:

```text
ml_program.global tensor<T>
  -> memref.global

ml_program.global_load
  -> memref.get_global plus memref.load/copy behavior

ml_program.global_store
  -> memref.get_global plus memref.copy/store behavior
```

The exact IR depends on tensor shape, read/write conflicts, and the surrounding
bufferization analysis.

## Example IR

### Globals And SSACFG Functions

```mlir
ml_program.global private @weights(dense<4> : tensor<4xi32>) : tensor<4xi32>
ml_program.global private mutable @state : tensor<?xi32>

ml_program.func @read_weights() -> tensor<4xi32> {
  %w = ml_program.global_load_const @weights : tensor<4xi32>
  ml_program.return %w : tensor<4xi32>
}

ml_program.func @round_trip_state() {
  %s = ml_program.global_load @state : tensor<?xi32>
  ml_program.global_store @state = %s : tensor<?xi32>
  ml_program.return
}
```

`@weights` is immutable and can be loaded with `global_load_const`. `@state` is
mutable and must use the side-effecting load/store ops.

### Graph Subgraph With Tokens

```mlir
ml_program.global private mutable @state(dense<0> : tensor<i64>) : tensor<i64>

ml_program.subgraph @ordered_state() -> (tensor<i64>, !ml_program.token) {
  %t0 = ml_program.token
  %v, %t1 = ml_program.global_load_graph @state
    ordering(%t0 -> !ml_program.token) : tensor<i64>
  %t2 = ml_program.global_store_graph @state = %v
    ordering(%t1 -> !ml_program.token) : tensor<i64>
  ml_program.output %v, %t2 : tensor<i64>, !ml_program.token
}
```

The token chain says that the store depends on the load.

### Global Pipeline Optimization

Before `-mlprogram-pipeline-globals`, a function may load the same global twice:

```mlir
ml_program.global private mutable @state(dense<4> : tensor<4xi32>)
  : tensor<4xi32>

func.func @global_double_load() {
  %0 = ml_program.global_load @state : tensor<4xi32>
  %1 = ml_program.global_load @state : tensor<4xi32>
  %2 = "test.combine"(%0, %1)
    : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  ml_program.global_store @state = %2 : tensor<4xi32>
  return
}
```

After the pass, the second load can be replaced by the first load if nothing in
between may overwrite the global.

## Mental Model

Think of `ml_program` as the model shell around tensor computation:

1. Globals represent weights, parameters, or state.
2. Functions and subgraphs represent callable model entry points.
3. Loads and stores connect computation to global model state.
4. Tokens make graph-side effects explicit.
5. Bufferization later maps tensor globals and global accesses to memory.

## Gotchas

- The dialect is explicitly under active development.
- `ml_program.func` and `ml_program.subgraph` are both callable, but their
  region kinds differ.
- `ml_program.global_load_const` is only legal for immutable globals.
- Storing to an immutable global is invalid.
- Load/store result types must match the referenced global type.
- Graph regions need token-ordered global side effects.
- The `#ml_program.extern` attribute does not load data by itself; it marks that
  external resolution is required by some implementation-specific mechanism.
- One-Shot Bufferize can lower MLProgram tensor globals, but this relies on
  bufferization interface models being registered.

## Source Map

Important source files in the LLVM tree:

- `mlir/include/mlir/Dialect/MLProgram/IR/MLProgramBase.td` defines the dialect.
- `mlir/include/mlir/Dialect/MLProgram/IR/MLProgramOps.td` defines the 11 ops.
- `mlir/include/mlir/Dialect/MLProgram/IR/MLProgramTypes.td` defines
  `!ml_program.token`.
- `mlir/include/mlir/Dialect/MLProgram/IR/MLProgramAttributes.td` defines
  `#ml_program.extern`.
- `mlir/lib/Dialect/MLProgram/IR/MLProgramOps.cpp` implements custom parsing,
  printing, and verification.
- `mlir/include/mlir/Dialect/MLProgram/Transforms/Passes.td` declares
  `mlprogram-pipeline-globals`.
- `mlir/lib/Dialect/MLProgram/Transforms/PipelineGlobalOps.cpp` implements the
  global load/store optimization pass.
- `mlir/lib/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.cpp`
  implements One-Shot Bufferize integration.
- `mlir/test/Dialect/MLProgram/` contains parser, verifier, pipeline, and
  bufferization tests.
