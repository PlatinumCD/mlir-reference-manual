# torch-mlir `torch` Dialect

The torch-mlir `torch` dialect is the main bridge between PyTorch programs and MLIR. It represents PyTorch's TorchScript-level semantics inside MLIR so that a PyTorch model can be analyzed, normalized, and lowered using MLIR passes.

For a beginner, the useful mental model is this: the `torch` dialect is PyTorch before it has fully become normal compiler tensor IR. It still knows about Torch tensors, Torch scalar types, lists, tuples, dictionaries, `torch.nn.Module` objects, object attributes, control flow, ATen operators, and PyTorch-style mutation. Later passes gradually remove those features or translate them into lower-level dialects such as `arith`, `scf`, `tensor`, `linalg`, `tm_tensor`, `tosa`, `stablehlo`, or `ml_program`.

The local checkout defines 833 `torch` operations. Most of them are generated ATen operator overloads. The important lesson is not to memorize all of them, but to understand the role of the dialect: it keeps PyTorch meaning explicit long enough for torch-mlir to make careful lowering choices.

## When The Torch Dialect Is Important

The `torch` dialect is important whenever the compiler still needs PyTorch semantics. You will see it near the front of a torch-mlir pipeline, after import from TorchScript, FX, ExportedProgram, ONNX-fronted Torch paths, or other PyTorch-flavored sources.

Use this dialect when you need to answer questions like:

- Which PyTorch operator did this value come from?
- Is this tensor still potentially mutable or aliased?
- Are shapes, ranks, or dtypes still being inferred from Torch semantics?
- Has object-oriented `torch.nn.Module` structure been flattened yet?
- Has an in-place ATen operation been converted to a value-semantic form?
- Is a control-flow operation still a Torch `prim` operation, or has it become `scf`?
- Which backend conversion is expected to consume this operation?

The dialect is especially important for debugging PyTorch model compilation. If a model fails to lower, the first useful question is often whether the failing operation is still a `torch.aten.*` op, an object-graph op, a Torch control-flow op, or already a lower-level MLIR op.

## Why It Is Needed

PyTorch programs are not just tensor algebra. They can contain Python-like containers, optional values, module attributes, mutable tensors, in-place updates, object construction, dynamic control flow, symbolic shapes, dtype calculations, random-number behavior, and many ATen overloads with subtly different schemas.

Lowering that directly to low-level MLIR would either lose information or require every backend to understand PyTorch. The `torch` dialect gives torch-mlir a place to preserve PyTorch meaning while it simplifies the program.

The dialect's own TableGen description says it maintains a fairly isomorphic representation with TorchScript and provides transforms that lower to the "Torch backend contract." That contract is the simplified form presented to later conversions. In that form, the TorchScript object graph has been flattened, most tensor operations use value semantics, operation variants have been reduced, and tensor sizes, ranks, and dtypes have been propagated as far as possible.

The implication is that `torch` is not the final destination. It is the normalization layer that makes PyTorch precise enough for compiler lowering.

## Type Inventory

The two most important tensor types are `!torch.tensor` and `!torch.vtensor`.

`!torch.tensor` models a general PyTorch tensor. It can represent aliasing, mutation, and other behavior that does not fit pure SSA value semantics. If a value has this type, you should not assume that it behaves like a plain MLIR `tensor`.

`!torch.vtensor` models a value-semantic Torch tensor. It carries the same kind of size and dtype information as `!torch.tensor`, but it promises that the tensor is being used in the value-semantic world. When its dtype is known, it can be converted to a builtin MLIR `tensor` type.

Both tensor forms use Torch-style static information:

```text
!torch.tensor<*,unk>
!torch.vtensor<[?,3,224,224],f32>
!torch.vtensor<[2,?],si64>
```

The `*` means unknown rank and `unk` means unknown dtype. A list such as `[?,3,224,224]` means known rank with some dynamic and some static sizes.

The dialect also defines scalar and object-like types:

- `!torch.bool`, `!torch.int`, `!torch.float`, `!torch.number`, `!torch.str`, and `!torch.none` model Torch scalar and singleton values.
- `!torch.Device` and `!torch.Generator` model device and random-generator objects.
- `!torch.nn.Module` models a module instance.
- `!torch.list<T>`, `!torch.optional<T>`, `!torch.tuple<...>`, `!torch.dict`, and `!torch.union<...>` model Python/Torch container typing.
- `!torch.any` is the escape hatch when a value is not statically known more precisely.
- `!torch.qint8`, `!torch.qint16`, `!torch.quint8`, and `!torch.qint32` model quantized element types.
- `!torch.LinearParams` models packed quantized linear parameters.

There is also an important argument attribute, `torch.type_bound`. It can be attached to function arguments to record a more precise Torch type bound. The `torch-adjust-calling-conventions` pass incorporates those bounds into function signatures when it normalizes public function conventions.

## Operation Inventory

The local `torch` dialect defines 833 operations. They fall into three broad groups.

First, there are 742 exact `torch.aten.*` overload operations. These represent PyTorch ATen operator schemas. After stripping overload suffixes, those 742 exact operation names correspond to 568 ATen operation families.

Second, there are TorchScript and torch-mlir infrastructure operations. These cover module objects, class metadata, slots, globals, constants, control flow, container construction, value-semantics conversion, shape and dtype calculation, symbolic shapes, quantization helpers, and special frontend placeholders.

Third, there are a few namespace-specific helpers such as `torch.torchvision.*`, `torch.onnx.rotary_embedding`, and `torch.hop_flex_attention`.

### ATen Operation Families

The `torch.aten.*` operations are generated from PyTorch operator schemas. They are the largest part of the dialect. Exact overload names matter because PyTorch has multiple schemas for a concept such as add, reshape, mean, or index.

Examples include scalar and tensor arithmetic such as `torch.aten.add.Tensor`, `torch.aten.sub.Tensor`, `torch.aten.mul.Tensor`, `torch.aten.div.Tensor`, `torch.aten.pow.Tensor_Tensor`, and comparison operators such as `torch.aten.eq.Tensor`, `torch.aten.lt.Tensor`, `torch.aten.ge.Tensor`, and `torch.aten.ne.Tensor`.

Shape and view operations include families such as `torch.aten.view`, `torch.aten.reshape`, `torch.aten.flatten`, `torch.aten.squeeze`, `torch.aten.unsqueeze`, `torch.aten.transpose`, `torch.aten.permute`, `torch.aten.expand`, `torch.aten.repeat`, `torch.aten.slice.Tensor`, `torch.aten.select.int`, and `torch.aten.as_strided`.

Tensor construction and conversion operations include families such as `torch.aten.tensor`, `torch.aten.empty`, `torch.aten.zeros`, `torch.aten.ones`, `torch.aten.full`, `torch.aten.arange`, `torch.aten.eye`, `torch.aten.to.dtype`, `torch.aten.type_as`, and `torch.aten.detach`.

Indexing and data-movement operations include `torch.aten.index.Tensor`, `torch.aten.index_put`, `torch.aten.gather`, `torch.aten.scatter`, `torch.aten.scatter_add`, `torch.aten.masked_fill`, `torch.aten.where`, `torch.aten.cat`, `torch.aten.stack`, `torch.aten.split`, and `torch.aten.chunk`.

Reductions and statistics include `torch.aten.sum`, `torch.aten.mean`, `torch.aten.prod`, `torch.aten.max`, `torch.aten.min`, `torch.aten.argmax`, `torch.aten.argmin`, `torch.aten.var`, `torch.aten.std`, `torch.aten.any`, and `torch.aten.all`.

Neural-network and numerical families include convolution, pooling, normalization, dropout, softmax, log-softmax, losses, linear algebra, FFT, random-number operations, and quantized operators. Examples include `torch.aten.convolution`, `torch.aten.max_pool2d`, `torch.aten.avg_pool2d`, `torch.aten.batch_norm`, `torch.aten.layer_norm`, `torch.aten.softmax.int`, `torch.aten.log_softmax.int`, `torch.aten.mm`, `torch.aten.matmul`, `torch.aten.bmm`, `torch.aten.linalg_vector_norm`, and `torch.aten.scaled_dot_product_attention`.

The beginner rule is to read the `aten` name as the PyTorch operation and the suffix as the overload. A conversion pass may support one overload but not another, so `torch.aten.add.Tensor` and `torch.aten.add.Scalar` are not interchangeable compiler facts.

### Module, Class, And Slot Operations

These operations preserve TorchScript object structure before it is flattened:

- `torch.nn_module`
- `torch.nn_module_terminator`
- `torch.class_type`
- `torch.class_type_terminator`
- `torch.method`
- `torch.attr`
- `torch.slot`

`torch.nn_module` and `torch.class_type` declare module or class structure. `torch.method` and `torch.attr` record methods and attributes. `torch.slot` represents a named storage slot on a class or module. These operations are important early in the pipeline because PyTorch models often carry parameters, buffers, and submodules through object fields rather than as plain SSA arguments.

### Global Slot Operations

Object graph lowering uses global slots:

- `torch.global_slot`
- `torch.global_slot.module_initializer`
- `torch.initialize.global_slots`
- `torch.global_slot.init`
- `torch.global_slot.get`
- `torch.global_slot.set`

These operations are the intermediate form used after object graph globalizing begins. They let the compiler represent parameters and attributes as named global storage before later passes inline them, erase module initializers, or convert state into function arguments and constants.

### Constants, Containers, And TorchScript Prim Ops

Constants are represented by:

- `torch.constant.none`
- `torch.constant.str`
- `torch.constant.device`
- `torch.constant.int`
- `torch.constant.float`
- `torch.constant.number`
- `torch.constant.bool`

Container and object construction uses:

- `torch.prim.ListConstruct`
- `torch.prim.ListUnpack`
- `torch.prim.TupleConstruct`
- `torch.prim.TupleIndex`
- `torch.prim.TupleUnpack`
- `torch.prim.DictConstruct`
- `torch.prim.CreateObject`
- `torch.prim.GetAttr`
- `torch.prim.SetAttr`
- `torch.prim.CallMethod`

Other generated or hand-written `prim` helpers include:

- `torch.prim.Enter`
- `torch.prim.Exit`
- `torch.prim.Load`
- `torch.prim.Store`
- `torch.prim.Print`
- `torch.prim.RaiseException`
- `torch.prim.Uninitialized`
- `torch.prim.NumToTensor.Scalar`
- `torch.prim.abs.Scalar`
- `torch.prim.device`
- `torch.prim.dtype`
- `torch.prim.layout`
- `torch.prim.max.int`
- `torch.prim.max.self_int`
- `torch.prim.min.int`
- `torch.prim.min.self_int`
- `torch.prim.tolist`
- `torch.prim.unchecked_cast`

These are TorchScript operations rather than ATen tensor kernels. They are common before torch-mlir has simplified Python-like program structure.

### Torch Control Flow

Torch control flow uses:

- `torch.prim.If`
- `torch.prim.If.yield`
- `torch.prim.Loop`
- `torch.prim.Loop.condition`

`torch.prim.If` is similar in spirit to `scf.if`, but it uses Torch boolean values and always carries TorchScript semantics. `torch.prim.Loop` combines loop-count and loop-condition behavior from TorchScript. The `convert-torch-to-scf` pass lowers these into standard MLIR structured control flow when the surrounding values are legal.

### Value Semantics, Copies, And Type Refinement

These operations mark boundaries between Torch's mutable tensor world and MLIR's value world:

- `torch.copy.to_tensor`
- `torch.copy.to_vtensor`
- `torch.overwrite.tensor.contents`
- `torch.tensor_static_info_cast`
- `torch.derefine`
- `torch.tensor.literal`
- `torch.vtensor.literal`
- `torch.operator`
- `torch.operator_terminator`

`torch.copy.to_vtensor` and `torch.copy.to_tensor` move between value-semantic and non-value tensor representations. `torch.overwrite.tensor.contents` represents a content update to a non-value tensor. `torch.tensor_static_info_cast` changes the static shape or dtype information attached to a tensor type without changing the runtime value. `torch.derefine` moves from a more precise type to a less precise supertype, which is sometimes needed to model TorchScript subtyping.

`torch.operator` is an opaque fallback for an unregistered Torch JIT operator. It preserves the call structure when torch-mlir cannot give the operator a normal first-class op yet.

### Shape, Dtype, Symbolic, And Primitive Helpers

Shape and dtype abstract-interpretation operations are:

- `torch.shape.calculate`
- `torch.shape.calculate.yield`
- `torch.shape.calculate.yield.shapes`
- `torch.dtype.calculate`
- `torch.dtype.calculate.yield`
- `torch.dtype.calculate.yield.dtypes`

Primitive tensor helpers are:

- `torch.prims.collapse`
- `torch.prims.convert_element_type`
- `torch.prims.iota`
- `torch.prims.prod`
- `torch.prims.split_dim`
- `torch.prims.sqrt`
- `torch.prims.squeeze`
- `torch.prims.sum`
- `torch.prims.var`
- `torch.prims.view_of`

Symbolic-shape operations are:

- `torch.symbolic_int`
- `torch.bind_symbolic_shape`

These operations appear when the compiler is representing shape or dtype reasoning directly in the IR. `torch.shape.calculate` and `torch.dtype.calculate` are not ordinary runtime computation in the same sense as ATen tensor kernels; they are reified calculations used to explain or simplify inferred metadata.

### Quantization, TorchVision, ONNX, And HOP Helpers

Quantization helpers include:

- `torch.linear_params.create`
- `torch.per_tensor_affine.create`
- `torch.quantized.linear`
- `torch.promote_dtypes`

TorchVision custom operations include:

- `torch.torchvision.deform_conv2d`
- `torch.torchvision.nms`
- `torch.torchvision.roi_align`
- `torch.torchvision.roi_pool`

Other special frontend helpers include:

- `torch.runtime.assert`
- `torch.onnx.rotary_embedding`
- `torch.hop_flex_attention`
- `torch.valsem.aten.bernoulli.float`

`torch.runtime.assert` keeps a runtime check in the Torch dialect until it can be converted. `torch.onnx.rotary_embedding` represents an ONNX-sourced rotary embedding operation. `torch.hop_flex_attention` represents a higher-order PyTorch FlexAttention placeholder before the regular JIT registry path can cover it. `torch.valsem.aten.bernoulli.float` is a value-semantic random helper for Bernoulli.

## Transformations

Torch-owned transformation passes prepare the dialect for backend lowering.

`torch-prepare-for-globalize-object-graph` prepares object-graph IR so it can be globalized. It normalizes call-like patterns and invariants that the object graph globalizer expects.

`torch-globalize-object-graph` flattens the TorchScript object graph into global-like declarations and functions. This is where `torch.nn_module`, slots, methods, attributes, and reachable object paths start turning into a compiler-friendly representation.

`torch-adjust-calling-conventions` rewrites public function conventions. It incorporates `torch.type_bound`, removes Python-style `None` returns, and turns tuple returns into multiple MLIR results where appropriate.

`torch-inline-global-slots` inlines `torch.global_slot` uses where safe. This helps remove object storage indirection after object graph lowering has exposed parameters or attributes.

`torch-reduce-op-variants` reduces the number of ATen forms the backend has to understand. It rewrites in-place variants and scalar variants toward more canonical value-semantic tensor forms when possible.

`torch-maximize-value-semantics` is one of the key Torch passes. It tries to move tensor operations from `!torch.tensor` into `!torch.vtensor`, roughly like converting mutable tensor state into SSA value flow.

`torch-refine-public-return` refines public function return types using intraprocedural shape and dtype information.

`torch-decompose-complex-ops` lowers complex Torch operations into simpler Torch operations when the decomposition is legal for the selected backend. For example, a high-level operator may become smaller arithmetic, reduction, or indexing pieces.

`torch-recompose-complex-ops` does the opposite in selected cases. It recognizes lower-level patterns produced by frontend decomposition and rebuilds a higher-level Torch operation when that is more useful for lowering.

`torch-fuse-quantized-ops` fuses quantize/dequantize style patterns into quantized Torch operations.

`torch-match-quantized-custom-ops` maps recognized quantization custom-op namespaces to known ATen quantized forms.

`torch-scalarize-shapes` rewrites shape calculations into scalar form so static shape information can propagate more effectively.

`torch-reify-shape-calculations` inserts explicit `torch.shape.calculate` operations for shape abstract interpretation, and `torch-simplify-shape-calculations` simplifies those calculations.

`torch-reify-dtype-calculations` inserts explicit `torch.dtype.calculate` operations for dtype abstract interpretation, and `torch-simplify-dtype-calculations` simplifies those calculations.

`torch-drop-abstract-interp-calculations` removes reified shape and dtype calculations after their information is no longer needed.

`torch-erase-module-initializer` removes `torch.global_slot.module_initializer` operations when the module initializer has become unnecessary or illegal for the target stage.

`torch-restructure-non-constant-axes` rewrites some reduction-style operations around non-constant axes so later conversion patterns that require constant axes can fire.

`torch-lower-to-backend-contract` runs the main simplification pipeline that tries to produce the Torch backend contract. It uses decomposition, shape/dtype refinement, value-semantics maximization, and operation-variant reduction to make the IR easier for backend conversions to consume.

`torch-verify-backend-contract-no-decompositions` checks that the IR satisfies the backend contract under a mode that does not assume further decompositions will happen.

## Conversions

The `torch` dialect has several direct conversion paths.

`convert-torch-to-arith` lowers Torch scalar arithmetic, boolean logic, constants, tensor literals, and `torch.runtime.assert`-style checks into `arith` and related standard operations when the values are no longer Torch-specific.

`convert-torch-to-scf` lowers `torch.prim.If`, `torch.prim.If.yield`, `torch.prim.Loop`, and `torch.prim.Loop.condition` into `scf` control-flow operations.

`convert-torch-to-linalg` lowers supported Torch tensor operations to `linalg`, `tensor`, `arith`, `math`, and related dialects. It covers categories such as tensor-scalar interop, linear algebra, pooling, random operations, reductions, data movement, indirect data movement, and tensor constructors. It may insert checks such as `cf.assert` for shape mismatches because PyTorch often defines safe runtime behavior where lower-level Linalg would otherwise have undefined behavior.

`convert-torch-to-tensor` lowers selected Torch operations to the upstream `tensor` dialect, including operations such as `torch.aten.item`, `torch.aten.tensor`, and shape-as-tensor forms.

`convert-torch-to-tmtensor` lowers selected Torch operations to torch-mlir's `tm_tensor` dialect. This is used for structured operations such as scan, scatter, sort, top-k, and attention when a plain Linalg form would lose too much meaning.

`convert-torch-to-tosa` lowers supported Torch operations to TOSA. This path is useful for backends that consume the TOSA tensor operator set.

`convert-torch-to-stablehlo` lowers supported Torch operations to StableHLO. It has options for static-shape behavior, index-width behavior, and non-finite constants. This path is important when torch-mlir is used as a PyTorch frontend for StableHLO-based compiler stacks.

`convert-torch-conversion-to-mlprogram` is technically for the `torch_conversion` helper dialect rather than the main `torch` dialect, but it belongs to the same conversion layer. It converts torch-mlir's conversion support constructs into `ml_program` globals and related IR.

The beginner rule is that conversion passes are not all interchangeable. `convert-torch-to-linalg` is a general structured-tensor lowering path. `convert-torch-to-tmtensor` preserves some higher-level tensor algorithms. `convert-torch-to-stablehlo` and `convert-torch-to-tosa` target external ML operator sets. `convert-torch-to-arith`, `convert-torch-to-scf`, and `convert-torch-to-tensor` peel away specific standard pieces.

## What It Implies

Seeing `torch.aten.*` means the compiler is still carrying a PyTorch operator schema. Check the exact overload suffix before assuming the semantics. An `aten` op may have a direct conversion, may require decomposition, or may be unsupported for the selected backend.

Seeing `!torch.tensor` means aliasing or mutation may still matter. Seeing `!torch.vtensor` means the value-semantics pass has proven or enforced a cleaner SSA-like tensor interpretation.

Seeing `torch.prim.*` means TorchScript structure is still present. `torch.prim.If` and `torch.prim.Loop` are not yet standard MLIR control flow. Container and object `prim` ops mean lists, tuples, dictionaries, or module attributes have not been fully simplified.

Seeing `torch.global_slot.*` means object graph lowering has started but storage has not been fully inlined or erased.

Seeing `torch.shape.calculate` or `torch.dtype.calculate` means shape or dtype abstract interpretation has been made explicit. These operations are compiler reasoning artifacts, not user-level model layers.

Seeing `torch.copy.to_vtensor`, `torch.copy.to_tensor`, or `torch.overwrite.tensor.contents` means you are near the boundary between PyTorch mutation semantics and value-semantic tensor IR.

## How To Read Torch IR

Start with the operation namespace. `torch.aten` usually means a PyTorch tensor or scalar operator. `torch.prim` usually means TorchScript program structure. `torch.global_slot`, `torch.nn_module`, `torch.class_type`, `torch.method`, `torch.attr`, and `torch.slot` usually mean object graph or module metadata. `torch.shape`, `torch.dtype`, and `torch.symbolic_int` usually mean compiler-side reasoning about metadata.

Next, inspect tensor types. If a tensor is `!torch.tensor<*,unk>`, very little is known. If it is `!torch.vtensor<[1,3,224,224],f32>`, the compiler knows value semantics, rank, sizes, and dtype.

Then check whether the IR has reached the backend contract. A backend-contract-like program should have fewer object operations, fewer mutable tensor operations, more value-semantic tensors, and more refined ranks and dtypes.

Finally, match unsupported operations to conversion paths. If an op is unsupported by `convert-torch-to-linalg`, it may need `torch-decompose-complex-ops`, `torch-reduce-op-variants`, `convert-torch-to-tmtensor`, or a backend-specific legality list.

## Minimal Example

This simplified fragment adds two value-semantic tensors with an ATen overload:

```mlir
%alpha = torch.constant.int 1
%result = torch.aten.add.Tensor %lhs, %rhs, %alpha
  : !torch.vtensor<[?,128],f32>, !torch.vtensor<[?,128],f32>, !torch.int
    -> !torch.vtensor<[?,128],f32>
```

Read it as "call PyTorch's tensor overload of `aten.add` with alpha equal to 1." While it remains in the `torch` dialect, the compiler still knows this is PyTorch addition. After lowering, it may become `linalg`, `stablehlo`, `tosa`, or lower-level arithmetic depending on the chosen pipeline.

