# torch-mlir torch_c Dialect

The torch-mlir `torch_c` dialect is the TorchConversion dialect. It is a small bridge dialect used when torch-mlir moves from Torch types to backend-friendly MLIR types. Its name is short for "torch conversion"; the TableGen file notes that `torch_conversion` would be too verbose.

For a beginner, the important idea is that `torch_c` is not the dialect that represents PyTorch operators. That is the much larger `torch` dialect. The `torch_c` dialect lives at the boundary between Torch IR and backend IR. It says, explicitly and temporarily, "this value is crossing from a Torch type to a builtin MLIR type" or "this value is crossing back."

This matters because torch-mlir is organized around a backend contract. The frontend imports and normalizes PyTorch into the `torch` dialect. Backends such as Linalg-on-Tensors, TOSA, and StableHLO want builtin tensors and ordinary MLIR scalar types. `torch_c` provides the materialization operations that make this type conversion legal, inspectable, and removable.

## Operation Inventory

The `torch_c` dialect defines these operations:

```text
torch_c.to_builtin_tensor
torch_c.from_builtin_tensor
torch_c.to_i1
torch_c.from_i1
torch_c.to_i64
torch_c.from_i64
torch_c.to_f64
torch_c.from_f64
torch_c.i64_to_generator
torch_c.generator_to_i64
torch_c.get_next_seed
```

The dialect does not define its own types in this checkout. Instead, it converts between existing Torch types and builtin MLIR types:

```text
!torch.vtensor      <-> tensor
!torch.bool         <-> i1
!torch.int          <-> i64
!torch.float        <-> f64
!torch.Generator    <-> i64
```

Most `torch_c` operations are pure. `torch_c.get_next_seed` is intentionally not pure because it models advancing a global random seed.

## Tensor Boundary Ops

`torch_c.to_builtin_tensor` converts a value-semantic Torch tensor, `!torch.vtensor`, into a builtin MLIR `tensor`. It only accepts `!torch.vtensor`, not the non-value-semantic Torch tensor type. That distinction matters: `!torch.vtensor` is immutable and non-aliased, so it can safely map to MLIR's value-semantics tensor world.

`torch_c.from_builtin_tensor` does the opposite conversion. It wraps a builtin `tensor` back into a `!torch.vtensor` when a conversion pattern still needs to return a Torch-typed value.

Both operations verify that the operand and result agree on rank, shape, and element type. For integer tensors, the verifier compares integer bit width but does not require signedness to match exactly. This is deliberate because different backends have different expectations about signed, unsigned, and signless integer tensors.

For example:

```mlir
%0 = torch_c.to_builtin_tensor %arg0
  : !torch.vtensor<[3,?],si8> -> tensor<3x?xi8>

%1 = torch_c.from_builtin_tensor %0
  : tensor<3x?xi8> -> !torch.vtensor<[3,?],si8>
```

These operations also fold constants. A `torch.vtensor.literal` can fold through `to_builtin_tensor` into an `arith.constant`, and an `arith.constant` tensor can fold through `from_builtin_tensor` into a Torch value tensor literal when the types line up.

## Scalar Boundary Ops

`torch_c.to_i1` converts `!torch.bool` to `i1`, and `torch_c.from_i1` converts `i1` back to `!torch.bool`.

`torch_c.to_i64` converts `!torch.int` to signless `i64`, and `torch_c.from_i64` converts `i64` back to `!torch.int`.

`torch_c.to_f64` converts `!torch.float` to `f64`, and `torch_c.from_f64` converts `f64` back to `!torch.float`.

`torch_c.generator_to_i64` converts a Torch generator handle to an `i64`, and `torch_c.i64_to_generator` reconstructs a `!torch.Generator` from that integer representation. This is a backend-boundary representation choice, not a general semantic claim that every PyTorch generator is just an integer.

The bool, int, and float conversions implement folders for constant values. In simple cases, canonicalization can remove a round trip such as `from_i64` followed by `to_i64`, or fold a Torch constant through the conversion into a builtin MLIR constant.

## Random Seed Op

`torch_c.get_next_seed` returns an `i64` seed and has side effects. It is used by random lowering paths when torch-mlir needs an explicit seed value for backend IR.

The operation itself is intentionally abstract. It says "advance and return the next seed" without committing the `torch_c` dialect to a specific storage mechanism. A later conversion pass, `convert-torch-conversion-to-mlprogram`, lowers it to `ml_program` state. That pass creates or reuses a mutable private global named `global_seed`, loads it, applies a linear congruential update, stores the updated scalar back into the global, and replaces `torch_c.get_next_seed` with the new `i64` value.

This is a good example of why bridge dialects exist. A random op lowered to Linalg may still need a stateful seed operation. The `torch_c` dialect can carry that seed operation until a backend-specific state representation is chosen.

## Backend Type Conversion

The core helper is `setupBackendTypeConversion`. It configures an MLIR `TypeConverter` and `ConversionTarget` for the ordinary backend type boundary.

The default conversion maps:

```text
!torch.vtensor<[...],dtype> -> tensor<...xdtype>
!torch.bool                 -> i1
!torch.int                  -> i64
!torch.float                -> f64
!torch.Generator            -> i64
```

For value tensors, the default path converts integer element types to signless builtin integers. The StableHLO-specific setup is slightly different: it preserves unsigned integer element types but converts signed integer element types to signless. That distinction reflects backend expectations rather than a difference in PyTorch semantics.

The type converter also registers materializations. If a converted function body still has a producer of `!torch.vtensor` where the new type should be a builtin tensor, the converter can insert `torch_c.to_builtin_tensor`. If a remaining Torch operation needs a `!torch.vtensor` but the surrounding value has already become a builtin `tensor`, it can insert `torch_c.from_builtin_tensor`. The same idea applies to bool, int, float, and generator conversions.

## Transformations And Passes

The main TorchConversion passes are:

```text
torch-func-backend-type-conversion
torch-func-backend-type-conversion-for-stablehlo
torch-finalizing-backend-type-conversion
torch-finalizing-backend-type-conversion-for-stablehlo
torch-verify-linalg-on-tensors-backend-contract
torch-verify-tosa-backend-contract
torch-verify-stablehlo-backend-contract
torch-unpack-quant-tensor
torch-convert-custom-quant-op
```

The StableHLO and TOSA passes are conditionally compiled when those integrations are enabled.

`torch-func-backend-type-conversion` is a module pass. It converts function signatures, calls, returns, and control-flow block arguments to backend types. It uses standard MLIR dialect conversion patterns for functions, calls, branch-like operations, and returns. When an operation body is not converted yet, the pass inserts `torch_c` materializations at the boundary.

For example, a function returning `!torch.vtensor<[],f32>` can become a function returning `tensor<f32>`. If an unknown operation inside the body still produces `!torch.vtensor<[],f32>`, the conversion inserts `torch_c.to_builtin_tensor` before the return.

`torch-func-backend-type-conversion-for-stablehlo` is the same idea with StableHLO-specific integer signedness rules.

`torch-finalizing-backend-type-conversion` runs after the program has been fully converted to backend types. At that point, all `torch_c` materialization operations should be identities. The pass marks the materialization ops illegal and replaces each one with its converted operand. It finalizes:

```text
torch_c.to_builtin_tensor
torch_c.from_builtin_tensor
torch_c.to_i1
torch_c.from_i1
torch_c.to_i64
torch_c.from_i64
torch_c.to_f64
torch_c.from_f64
torch_c.i64_to_generator
torch_c.generator_to_i64
```

It also strips function dialect attributes whose names start with `torch.` because those attributes are no longer used after conversion out of Torch. The non-StableHLO finalizer additionally runs a small cleanup pattern that removes `arith.extf` followed by a matching `arith.truncf`.

`torch-finalizing-backend-type-conversion-for-stablehlo` performs the corresponding finalization for the StableHLO type conversion setup.

## Backend Contract Verification

The verification passes check that a module has reached the intended backend contract.

`torch-verify-linalg-on-tensors-backend-contract` accepts structural operations such as modules, functions, and returns when their types are legal. It accepts scalar elementwise work in dialects such as `arith`, `math`, `func`, and `complex`, and tensor computation in dialects such as `linalg`, `tensor`, `sparse_tensor`, `affine`, `cf`, `scf`, `ml_program`, and TMTensor. It also allows `torch_c.get_next_seed` when its types are legal, because seed state can be lowered later.

`torch-verify-tosa-backend-contract` checks the TOSA backend shape of the program. It primarily allows TOSA plus a small amount of tensor and arithmetic support with legal tensor element types.

`torch-verify-stablehlo-backend-contract` checks the StableHLO backend shape of the program. It accepts StableHLO, CHLO, tensor, arithmetic, math, and shape dialect operations when types satisfy the backend's element type rules.

These verifier passes are not lowering passes. They are contract checks. Their job is to catch a pipeline that claims to have produced backend IR but still contains illegal Torch-style types or unsupported operations.

## Quantization Helper Passes

`torch-unpack-quant-tensor` is a specialized rewrite for the custom operator named `quant.matmul_rhs_group_quant`. It looks for packed integer weight literals used as the RHS operand, computes the pack ratio from the storage bit width and unpacked bit width, expands the last tensor dimension, and replaces the literal with an unpacked `torch.vtensor.literal`.

`torch-convert-custom-quant-op` is another specialized pass for the same custom operator. It converts the operator into backend IR using tensor and linalg operations. The conversion expands the LHS and RHS shapes, creates a dequantized RHS with `linalg.generic`, then emits another `linalg.generic` for the grouped matmul. It uses backend type conversion so Torch tensor values become builtin tensors at the boundary.

The source comments describe these as one-off conversions that should not be treated as default lowering flow until they are more mature. For readers, they are still useful examples of how `torch_c` helps a conversion pattern move from Torch operator space to backend tensor space.

## Conversion To MLProgram

The pass `convert-torch-conversion-to-mlprogram` handles recognized TorchConversion operations that need `ml_program` support. In this checkout, the important case is `torch_c.get_next_seed`.

The pass makes `torch_c.get_next_seed` illegal, adds a conversion pattern, and applies it inside functions. The replacement uses:

```text
ml_program.global
ml_program.global_load
tensor.extract
arith.muli
arith.addi
tensor.insert
ml_program.global_store
```

The resulting program has explicit mutable seed state in MLProgram form. That is much closer to what a backend can lower or interpret than an abstract `torch_c.get_next_seed` operation.

## When To Use It

Use `torch_c` when writing or debugging torch-mlir lowering passes that sit at the boundary between the `torch` dialect and backend dialects. If you see a `torch_c.to_builtin_tensor`, the program is moving from `!torch.vtensor` to builtin `tensor`. If you see `torch_c.from_builtin_tensor`, some remaining Torch-typed operation still expects a Torch value tensor.

Do not use `torch_c` to model PyTorch computation. Operations such as `aten.add`, `aten.matmul`, list construction, module state, and PyTorch control flow belong in the `torch` dialect and its lowering passes. The `torch_c` dialect is for type and state materialization during conversion.

The main implication is that `torch_c` operations should usually disappear before final backend IR. They are helpful while the program is partially converted, but a completed backend pipeline should either erase them with finalizing backend type conversion or lower special cases such as `get_next_seed` to a concrete state dialect. When `torch_c` remains late in a pipeline, it usually means the conversion is incomplete or a backend boundary has not been finalized.
