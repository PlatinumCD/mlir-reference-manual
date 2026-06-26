# IREE `iree_encoding` Dialect

The IREE `iree_encoding` dialect represents tensor layout intent. It is small in operation count but important in meaning: it lets IREE say that a tensor should be interpreted, tiled, padded, packed, or specialized in a particular way without changing the tensor's logical rank or shape.

For a beginner, the key idea is that an encoding is not the tensor data itself. It is metadata attached to a ranked tensor type, usually through an attribute such as `#iree_encoding.encoding` or `#iree_encoding.layout`. The compiler then uses that metadata to decide how to move from logical tensors to target-specific physical storage and dispatch code.

This dialect is central to IREE's data-tiling path. The compiler can insert `iree_encoding.set_encoding` before a compute operation, run the operation on encoded tensors, and then insert `iree_encoding.unset_encoding` to return to the default logical layout.

## When Encoding Is Important

Use `iree_encoding` when you need to understand why IREE changed tensor layouts around matmuls, scaled matmuls, convolutions, packed sub-byte data, padding, or target-specific dispatch layouts. If a tensor type looks like `tensor<?x?xf32, #iree_encoding.encoding<...>>`, the type is telling you that the tensor has the same logical shape but a non-default layout contract.

The dialect is important for performance. A matrix multiplication may run much better if its operands are tiled or packed in a way that matches the target backend. Rather than immediately materializing copies, IREE first expresses that layout choice as an encoding. Later passes decide how to specialize, materialize, or erase the encoding depending on the target and the surrounding dispatches.

You normally do not write this dialect as an application author. It is mostly produced by IREE compiler passes such as `iree-dispatch-creation-set-encoding`. Compiler developers use it when implementing layout selection, data tiling, operand packing, or lowering from logical tensor computation to physical memory forms.

## Why It Is Needed

Without a dialect like `iree_encoding`, a layout change would need to be represented too early as physical operations such as pack, unpack, pad, copy, or bitcast. That would make fusion harder and would force layout decisions before the compiler knows enough about the target device.

Encoding gives IREE a staged representation:

1. A verbose encoding records why the tensor needs a layout, such as `matmul` operand 0 with a certain indexing map.
2. A resolver specializes that encoding using backend and affinity information.
3. A serialized layout carries enough information to compute storage sizes and update stream/dispatch signatures.
4. Later lowering materializes physical operations and types.

This staged form lets IREE keep logical tensor semantics while still making layout decisions visible and optimizable.

## Operation Inventory

The local IREE Encoding dialect defines two operations.

### `iree_encoding.set_encoding`

`iree_encoding.set_encoding` adds an encoding attribute to a ranked tensor value. It preserves logical rank and logical extent. The source tensor must not already have a tensor encoding, and the result must have a valid IREE encoding.

The optional `encoding_dims` operands carry dynamic values needed by the encoding. For matmul-style encodings, those dimensions may be the problem dimensions such as M, N, and K. These values allow runtime-sensitive layout selection while still keeping the IR statically structured.

Example shape:

```mlir
#enc = #iree_encoding.encoding<
  operand_index = 0 : i64,
  op_type = matmul,
  element_types = [f32, f32, f32]>

%encoded = iree_encoding.set_encoding %arg0
    : tensor<?x?xf32> -> tensor<?x?xf32, #enc>
```

### `iree_encoding.unset_encoding`

`iree_encoding.unset_encoding` removes an encoding attribute and returns a default-layout tensor. In IREE's current model, the default layout is row-major. Like `set_encoding`, it preserves logical rank and shape. Dynamic result dimensions are passed explicitly when needed.

This operation usually appears after an encoded compute operation. For example, IREE may encode the lhs, rhs, and output tensors for a `linalg.matmul`, run the matmul with encoded tensor types, and then unset the result encoding so later IR sees an ordinary logical tensor.

## Attribute Inventory

Most of the dialect's meaning is in attributes, not ops.

- `#iree_encoding.encoding` records data-tiling information for an operand or result of an operation. It stores an `operand_index`, an operation type, element types, optional original element type, optional user indexing maps, and optional iteration sizes.
- `#iree_encoding.layout` wraps one or more serialized layout attributes. It is used when the compiler has resolved possible target layouts.
- `#iree_encoding.packed_storage` marks packed storage for sub-byte element types such as 4-bit values. It is only valid for sub-byte power-of-two bitwidth element types.
- `#iree_encoding.padding` records per-dimension padding. Padding may be static, zero, or dynamic with `?`.
- `#iree_encoding.identity` represents an identity encoding, meaning the encoded type is effectively the same as the unencoded type.
- `#iree_encoding.identity_resolver` is a resolver that maps an encoding to identity behavior.
- `#iree_encoding.unsupported_resolver` is a resolver that intentionally fails layout resolution, mainly for default or testing paths.
- `#iree_encoding.testing` is a testing attribute that can carry optional layouts and original element type information.
- `#iree_encoding.unknown` is a testing-oriented unknown encoding attribute.
- `#iree_encoding.specialization_resolver` is a testing resolver that produces specialized layouts from a seed value.
- `#iree_encoding.specialized` is a serialized testing layout identified by a seed and optional type.

The `#iree_encoding.encoding` operation type enum currently includes `matmul`, `scaled_matmul`, and `conv`. These tags tell the compiler what kind of operation the encoding came from, which is necessary because the same tensor shape can need different layouts depending on the consuming computation.

## Resolver and Interface Ideas

Encoding attributes fall into two broad roles.

Encoding type attributes are attached to tensor types. They describe the intended layout transformation at a high level. `#iree_encoding.encoding`, `#iree_encoding.padding`, and `#iree_encoding.packed_storage` are examples.

Encoding resolver attributes know how to turn abstract layout intent into something target-specific. The important interfaces are:

- `LayoutResolverAttr`, which converts verbose encodings into target-informed serialized encodings.
- `SerializableAttr`, which represents serialized encodings and can answer questions such as storage size.
- `LayoutMaterializerAttr`, which knows how to lower encoded tensors into physical operations and types.
- `EncodingPropagationAttrInterface` and `EncodingPropagationOpInterface`, which let encodings move through compatible operations.
- `ContractionEncodingAttrInterface`, which exposes information such as reduction dimensions for contraction-style encodings.

The beginner takeaway is that the dialect separates "what layout do we want" from "how does this exact backend implement it." This is why the same `#iree_encoding.encoding` can later resolve differently depending on affinity, target, or executable specialization.

## Transformations and Conversions

The first relevant pass is `iree-dispatch-creation-annotate-data-tiling-hints`. It adds hints to eligible linalg operations. By default, the pass targets matmul and scaled matmul families, and it can be configured for convolution as well.

The pass `iree-dispatch-creation-set-encoding` introduces encoding operations. It reads the tiling hints, creates `iree_encoding.set_encoding` for inputs and init tensors, clones the compute operation on encoded tensors, and then creates `iree_encoding.unset_encoding` on the result. This is the main pass that makes encodings visible in the IR.

The pass `iree-dispatch-creation-hoist-encoding-ops` moves encoding operations out of dispatch regions when possible. It can bubble `iree_encoding.set_encoding` upward through compatible producers, such as certain bit-extension or broadcast-like `linalg.generic` operations, so encodings can be placed at more useful boundaries.

The pass `iree-dispatch-creation-fuse-encoding-ops-into-dispatch-regions-pass` does the opposite when that is profitable: it fuses a top-level `iree_encoding.set_encoding` with the producer dispatch region or forms a dispatch around it. This helps avoid unnecessary standalone layout work.

The pass `iree-dispatch-creation-propagate-encodings` propagates encodings across operations that implement the encoding propagation interfaces. In the current source, one visible pattern swaps `tensor.cast` with `iree_encoding.set_encoding` when the encoding can be bubbled through the cast.

The pass `iree-dispatch-creation-convert-encoding-to-flow` converts top-level Encoding ops outside `flow.dispatch.region` into `flow.tensor.encode`. This is the bridge from the Encoding dialect to the Flow dialect. Inside dispatch regions, Encoding ops may remain because dispatch-level lowering can still reason about them.

The pass `iree-stream-unify-encoding-for-globals` is a Stream-level optimization that can unify multiple encodings of the same immutable global, reducing memory use when the same source parameter would otherwise be encoded several different ways.

The pass `iree-stream-specialize-encodings` resolves serializable encodings based on Stream affinity analysis. This is where verbose encodings become layouts attached to Stream tensor operations and executable bindings. If a dispatch executable can run on multiple devices with different layout interpretations, the pass can duplicate executables and update dispatches so each device gets the right binding types.

The pass `iree-stream-materialize-encodings` materializes `stream.tensor.encode` operations as dispatches to generated executables. The passes `iree-stream-encode-host-tensors` and `iree-stream-encode-device-tensors` lower tensor-like objects into storage/binary forms and resolve encoding-dependent storage details such as sub-byte packing.

## What It Implies

An encoded tensor type implies that the tensor's logical shape has not changed, but its layout contract has. The verifier enforces this for `iree_encoding.set_encoding` and `iree_encoding.unset_encoding`: they cannot change rank or logical shape. If the IR changes rank or shape at the same time, it is not a valid encoding boundary.

Encodings also imply that later passes must understand storage size. A padded tensor may need more memory than its logical element count suggests. A packed sub-byte tensor may need fewer bytes than a normal element-width calculation. This is why `SerializableAttr` and Stream specialization matter.

Finally, encodings imply target dependence. The same logical matmul encoding may resolve differently for different devices. That is a feature, not a bug. IREE keeps the encoding abstract until it has enough target and affinity information to make a backend-specific choice.

## How To Read Encoding IR

Start with `iree_encoding.set_encoding` and `iree_encoding.unset_encoding`. These mark where the compiler enters and exits an encoded layout.

Then inspect the tensor type after `set_encoding`. If it has `#iree_encoding.encoding`, look at `operand_index`, `op_type`, `element_types`, `user_indexing_maps`, and `iteration_sizes`. Those fields tell you which operation and operand the layout was chosen for.

If you see `#iree_encoding.layout`, the encoding is already more specialized. Look inside the layout list for attributes such as `#iree_encoding.padding` or `#iree_encoding.packed_storage`. Those are closer to physical storage decisions.

If you see testing attributes such as `#iree_encoding.testing`, `#iree_encoding.specialization_resolver`, or `#iree_encoding.specialized`, remember that they are mainly used to exercise the dialect and transformation machinery without depending on a real backend layout resolver.

## Minimal Example

This simplified example shows the core pattern:

```mlir
#lhs = #iree_encoding.encoding<
  operand_index = 0 : i64,
  op_type = matmul,
  element_types = [f32, f32, f32]>
#rhs = #iree_encoding.encoding<
  operand_index = 1 : i64,
  op_type = matmul,
  element_types = [f32, f32, f32]>
#out = #iree_encoding.encoding<
  operand_index = 2 : i64,
  op_type = matmul,
  element_types = [f32, f32, f32]>

%a_enc = iree_encoding.set_encoding %a
    : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
%b_enc = iree_encoding.set_encoding %b
    : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
%c_enc = iree_encoding.set_encoding %c
    : tensor<?x?xf32> -> tensor<?x?xf32, #out>
%r_enc = linalg.matmul
    ins(%a_enc, %b_enc : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
    outs(%c_enc : tensor<?x?xf32, #out>)
    -> tensor<?x?xf32, #out>
%r = iree_encoding.unset_encoding %r_enc
    : tensor<?x?xf32, #out> -> tensor<?x?xf32>{%m, %n}
```

The logical operation is still matrix multiplication. The encoding dialect adds a layout conversation around it so later passes can pick and materialize a better physical representation.
