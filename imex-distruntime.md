# Intel IMEX DistRuntime Dialect

The Intel IMEX `distruntime` dialect represents distributed-runtime operations for SPMD-style programs. It is meant to sit between higher-level distributed array semantics and lower-level runtime library calls.

For a beginner, the useful mental model is this: `distruntime` describes operations that require coordination between members of a distributed team. In the local checkout, the implemented surface is focused on asynchronous redistribution copies for ranked tensors: one operation for reshaping distributed data, one operation for permuting distributed data, and one operation for waiting on the asynchronous result.

The local checkout defines 3 `distruntime` operations, one custom async-handle type, one async-operation interface, and 1 DistRuntime lowering pass.

## When DistRuntime Is Important

`distruntime` is important when IMEX is compiling code that manipulates distributed tensors or arrays and needs a runtime library to move data between team members. The dialect is especially relevant near the boundary between distributed tensor IR, NDArray-like representations, and calls into Intel's distributed tensor runtime.

Use this dialect when you need to answer questions like:

- Is this operation local tensor math, or does it require distributed communication?
- What team owns the distributed array?
- What global shape and local offset describe the current local shard?
- What local output shard should be produced after a reshape or permutation?
- Is communication asynchronous, and where is it waited on?
- Has the operation been lowered to an IDTR runtime call?

You usually inspect this dialect after distributed array lowering has introduced runtime-facing operations, and before `lower-distruntime-to-idtr` converts those operations into `func.call` operations.

## Why It Is Needed

Distributed array compilers often need operations that are neither ordinary tensor math nor low-level MPI calls. A reshape or permutation of a distributed tensor may require moving data between processes. Encoding that directly as low-level runtime calls too early would hide important shape, offset, team, and dependency information.

The `distruntime` dialect keeps this information in MLIR. Its operations say which team participates, what the global shape is, where the local shard starts, what the new local shape and offset should be, and what tensor value represents the local result.

The dialect also models asynchronous communication. Asynchronous operations return `!distruntime.asynchandle`. The user must wait on that handle before consuming dependent results. This makes communication/computation overlap visible to the compiler while still allowing later lowering to a concrete runtime.

The implication is that `distruntime` is a runtime-boundary dialect. It is higher-level than raw calls and lower-level than a distributed tensor algebra dialect.

## Implemented Scope Versus RFC Scope

The local repository includes an RFC describing a broader DistRuntime dialect with operations such as team size, team member, allreduce, and halo exchange. Those are useful design context, but they are not all implemented in the local TableGen operation inventory.

The implemented local dialect surface contains:

- `distruntime.copy_reshape`
- `distruntime.copy_permute`
- `distruntime.wait`

This chapter describes the implemented dialect, while using the RFC only to explain the intended role of the dialect.

## Types And Interfaces

`!distruntime.asynchandle` is the custom type used to represent an asynchronous operation or communication handle. The type is opaque: the dialect does not define how it is implemented. The runtime lowering gives it concrete meaning.

`AsyncOpInterface` marks asynchronous operations and exposes the values that depend on completion. In this checkout, `distruntime.copy_reshape` and `distruntime.copy_permute` implement the interface and report their output tensor as dependent.

The main data payloads are ranked tensors. Shape and offset information is passed as variadic `index` operands. The `team` is an attribute, commonly shown as an integer attribute in tests, such as `{team = 22 : i64}`. It is up to the lowering and runtime to interpret the team identifier.

## Operation Inventory

This checkout defines 3 `distruntime` operations.

### `distruntime.copy_reshape`

`distruntime.copy_reshape` copies the necessary data from the local part of a distributed input tensor into the locally owned part of a reshaped distributed output tensor. The local input data is not modified.

The operation takes:

- `team`, the distributed team owning the array,
- `lArray`, the locally owned input tensor,
- `gShape`, the global input shape,
- `lOffsets`, the input local offsets in the global input,
- `ngShape`, the new global output shape,
- `nlOffsets`, the local offsets in the new output,
- `nlShape`, the local output shape.

It returns an async handle and the new local tensor:

```text
!distruntime.asynchandle, tensor<...>
```

Read this as "start the distributed copy needed for reshape, give me a handle for completion, and give me the local output value that depends on that completion."

The operation has a canonicalization pattern that can refine dynamically shaped result tensors to more static shapes when `nlShape` provides that information. It inserts a new operation with the refined type and casts back to the original type if needed.

### `distruntime.copy_permute`

`distruntime.copy_permute` copies the necessary data from the local part of a distributed input tensor into the locally owned part of a permuted distributed output tensor. The local input data is not modified.

The operation takes:

- `team`, the distributed team owning the array,
- `lArray`, the locally owned input tensor,
- `gShape`, the global input shape,
- `lOffsets`, the input local offsets,
- `nlOffsets`, the output local offsets,
- `nlShape`, the output local shape,
- `axes`, a dense integer array attribute describing the permutation.

It returns an async handle and the new local tensor.

`distruntime.copy_permute` has two important canonicalization patterns. One refines dynamically shaped result types using the known output local shape. The other folds a producer `tensor.cast` into the operation when the cast can be folded into the consumer, reducing unnecessary cast noise before lowering.

### `distruntime.wait`

`distruntime.wait` waits for an asynchronous DistRuntime operation to finish. It accepts a `!distruntime.asynchandle` and returns no results.

The semantic rule is simple: results of an asynchronous operation must be preceded by a wait before first use or consumption. In practice, this is the operation that makes async dependencies explicit in the IR.

### Complete Op List

For reference, the exact implemented operation names in this checkout are:

```text
distruntime.copy_permute, distruntime.copy_reshape, distruntime.wait
```

## Transformations And Conversions

The local checkout defines 1 DistRuntime pass.

`lower-distruntime-to-idtr` lowers DistRuntime operations to calls into IDTR, the Intel Distributed Tensor Runtime. The pass also inserts private runtime function declarations at module scope.

For `distruntime.wait`, the lowering creates a call to `_idtr_wait`.

For `distruntime.copy_reshape`, the lowering:

- creates an output tensor with `tensor.empty`,
- converts shape and offset operands into unranked memref descriptors,
- converts the input and output tensors into unranked memref descriptors using IMEX NDArray utilities,
- selects the element-typed IDTR function name such as `_idtr_copy_reshape_i64`,
- calls that runtime function,
- replaces the operation with the returned handle and the new output tensor.

For `distruntime.copy_permute`, the lowering is similar. It creates the output tensor, converts shape, offset, input, output, and axes data into runtime-call operands, selects a typed function name such as `_idtr_copy_permute_i64`, and replaces the operation with the runtime handle and output tensor.

The pass inserts declarations for several runtime entry points, including `_idtr_nprocs`, `_idtr_prank`, typed `_idtr_reduce_all_*` declarations, typed `_idtr_copy_reshape_*` declarations, `_idtr_update_halo_*` declarations, `_idtr_wait`, and typed `_idtr_copy_permute_*` declarations. Some of those declarations support the broader runtime interface even if the local implemented operation set only lowers reshape, permute, and wait.

The exact pass name covered in this chapter is:

```text
lower-distruntime-to-idtr
```

## What It Implies

Seeing `distruntime.copy_reshape` means a reshape of distributed data is not purely local. The compiler expects a runtime-mediated redistribution to produce the local result shard.

Seeing `distruntime.copy_permute` means a dimension permutation of distributed data is not purely local. The `axes` attribute describes the logical permutation, and the runtime is responsible for moving the necessary data.

Seeing `!distruntime.asynchandle` means an operation has started but may not be complete. Look for `distruntime.wait` before trusting dependent results.

Seeing `distruntime.wait` means the compiler is making an asynchronous dependency explicit. After `lower-distruntime-to-idtr`, this becomes a call to `_idtr_wait`.

Seeing `lower-distruntime-to-idtr` in a pipeline means the compiler is leaving the DistRuntime abstraction and committing to IDTR function calls and unranked memref descriptors.

## How To Read DistRuntime IR

Start with the team attribute. That identifies the group of processes or participants involved in the distributed operation.

Next, read the shape operands. `gShape` describes the global input tensor, `lOffsets` describes where the local input shard sits, and `nlShape` plus `nlOffsets` describe the local output shard.

Then inspect the async handle. The first result is the handle, and the second result is the local output tensor. A correct consumer should wait on the handle before using the output value.

Finally, check whether the op has already been lowered. Runtime-call IR will contain typed `_idtr_copy_reshape_*`, `_idtr_copy_permute_*`, or `_idtr_wait` calls instead of `distruntime` operations.

## Minimal Example

This simplified example starts a distributed reshape copy, waits for completion, and returns the local output tensor:

```mlir
%handle, %out = distruntime.copy_reshape %local
  g_shape %g0, %g1
  l_offs %o0, %o1
  to n_g_shape %ng0
  n_offs %no0
  n_shape %ns0
  {team = 22 : i64}
  : (tensor<?x?xi64>, index, index, index, index, index, index, index)
  -> (!distruntime.asynchandle, tensor<?xi64>)

distruntime.wait %handle : !distruntime.asynchandle
```

Read this as "for team 22, copy the local shard of a globally shaped input into the local shard of a reshaped output, then wait for the asynchronous runtime operation to finish."
