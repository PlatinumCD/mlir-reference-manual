# Intel IMEX `ndarray` Dialect

The Intel IMEX `ndarray` dialect is a high-level array dialect for Python-like and NumPy-like tensor programming. Its purpose is to keep array semantics visible in MLIR long enough for the compiler to reason about mutation, views, device placement, and distributed placement before lowering into `linalg`, `tensor`, `memref`, `bufferization`, Region, and runtime-oriented dialects.

In this checkout, the implemented dialect is deliberately small. The RFC describes a broader `NDArray` design with many Array API operations and a logical `ndarray.ndarray` type. The checked TableGen dialect currently defines ranked-tensor based operations plus the `#ndarray.envs<...>` attribute used as a tensor encoding to attach placement environments such as `#region.gpu_env<...>`. That difference matters: when reading this repository, expect to see ordinary `tensor<...>` types, sometimes annotated with `#ndarray.envs<...>`, rather than a large standalone array type vocabulary.

## When NDArray Is Important

Use `ndarray` when the IR needs array semantics that ordinary MLIR tensor operations do not express well. MLIR tensors are value-semantic and immutable. That is ideal for many compiler optimizations, but it is not enough to model Python array behavior such as in-place slice assignment or guaranteed views. The `ndarray` dialect gives those cases explicit operations.

This dialect is important in IMEX because it helps preserve "compute follows data" information. If an array is associated with a GPU environment or a distributed mesh/team environment, compiler passes can wrap work in Region dialect operations, propagate sharding, or lower the work in a way that respects where the data lives.

You would use this dialect when building a frontend for Array API, NumPy-like, Numba-like, or distributed array code in IMEX. You would not usually use it as a final target. Its main job is to be lowered once array-specific semantics have been made explicit.

## Why It Is Needed

The key problem is that arrays are not just mathematical values. They may be mutable, they may have views that alias existing storage, and they may live on a device or across a distributed team. If all of that is erased too early into plain tensors, later passes need expensive or impossible analysis to reconstruct it.

`ndarray.subview` is the clearest example. MLIR `tensor.extract_slice` has value semantics: it describes a slice value. NDArray's subview is intended to describe a view, closer to `memref.subview`. That guarantee affects aliasing and mutation. Similarly, `ndarray.insert_slice` is meant to update the destination array in place. That is not the same as a pure tensor update returning an unrelated value.

The dialect therefore acts as a bridge: high-level enough for Python-like array semantics, but close enough to ranked tensors that it can still lower into existing MLIR infrastructure.

## Operation Inventory

The local `NDArrayOps.td` defines seven `ndarray` operations.

### `ndarray.copy`

`ndarray.copy` creates a new array with the same shape and element type as the source. During `convert-ndarray-to-linalg`, it lowers through `memref.alloc`, `bufferization.to_buffer`, `memref.copy`, and `bufferization.to_tensor`. The lowering wraps the copy in a Region operation with a `"protect_copy_op"` environment marker so later region-sensitive transformations do not accidentally move or reinterpret the copy.

Use this operation when the program needs a distinct array allocation rather than another view of the same data.

### `ndarray.delete`

`ndarray.delete` explicitly frees an array. It declares a memory free effect and lowers to `memref.dealloc` after converting the input tensor view to a memref. It must be the last use of the input array.

This operation is unusual in a tensor-style IR because tensors normally do not have explicit lifetime management. It exists because NDArray is modeling array allocation and mutation more directly than pure tensor dialects.

### `ndarray.subview`

`ndarray.subview` takes a source ranked tensor plus offsets, sizes, and strides, and returns a reduced view. It implements offset/size/stride and view-like interfaces. The operation can use static entries, dynamic index operands, or a mixture of both.

The important semantic point is that this operation is a guaranteed view. The conversion code contains a memref-subview style lowering path, and the dialect has bufferization support for this operation. In the current `convert-ndarray-to-linalg` pass, `ndarray.subview` is treated as dynamically legal during partial conversion so that this view meaning is preserved for later handling instead of being erased too early.

Use it to represent array slicing when aliasing must be preserved.

### `ndarray.insert_slice`

`ndarray.insert_slice` copies values from a source array into a slice of a destination array. It is a destination-style operation and returns the destination type as its result, but the intended semantic is in-place update.

The conversion code contains a memref-based lowering path that converts the source and destination to memrefs, creates a `memref.subview` of the destination, and then emits `memref.copy`. If the source is a scalar tensor, that path uses a `linalg.generic` to broadcast the scalar into the destination slice. In the current partial conversion pass, `ndarray.insert_slice` is also treated as dynamically legal, because preserving its in-place update semantics is important for later bufferization and sharding logic.

Use this operation for Python-style slice assignment.

### `ndarray.linspace`

`ndarray.linspace` creates evenly spaced values between `start` and `stop`. It takes `start`, `stop`, `num`, and an `endpoint` unit attribute. The implementation can canonicalize constant `num` into a more precise static result shape.

Lowering computes a step value using arithmetic operations, creates an empty tensor of the result shape, and fills it with a parallel `linalg.generic` using `linalg.index` to compute each element.

Use it for Array API-style array creation.

### `ndarray.reshape`

`ndarray.reshape` changes the shape of an array without changing its data. It accepts a source ranked tensor, variadic index shape operands, and an optional `copy` attribute.

If `copy` is set, the lowering first emits `ndarray.copy`. It then builds a `tensor.from_elements` shape tensor and lowers to `tensor.reshape`. This keeps the high-level copy-vs-view intent explicit until conversion.

Use it when the frontend needs Array API reshape semantics, including the ability to request a copy.

### `ndarray.cast_elemtype`

`ndarray.cast_elemtype` casts an array from one element type to another. It has an optional `copy` attribute. Canonicalization can replace dynamic result shapes with static ones when the input shape is known, and can fold through a preceding tensor cast.

Lowering handles no-op casts specially. If the source and destination types are identical and `copy` is false, the op is erased. If the types are identical and `copy` is true, it becomes `ndarray.copy`. Otherwise it creates an empty tensor and emits a parallel `linalg.generic` with the appropriate arithmetic cast in the body, such as float extension/truncation or integer/float conversion.

Use it when Array API dtype conversion semantics should remain explicit before lowering.

## Attributes and Placement

The dialect defines `#ndarray.envs<...>`, an attribute that stores an array of environment attributes. In tests, GPU placement appears as a ranked tensor encoding:

```mlir
#GPUENV = #ndarray.envs<#region.gpu_env<device = "XeGPU">>
tensor<33xi64, #GPUENV>
```

The helper APIs in `NDArrayOps.h` inspect ranked tensor encodings and detect GPU environments with functions such as `hasGPUEnv` and `getGPUEnv`. This is what lets the `add-gpu-regions` pass identify NDArray computations that should be wrapped in GPU-marked Region dialect operations.

The RFC also discusses distributed environments and team information. In this source tree, the relevant implemented interaction is with MLIR's `shard` dialect through the NDArray mesh sharding extensions and the `coalesce-shard-ops` pass.

## Transformations and Conversions

The NDArray-specific transformation pass is `add-gpu-regions`. It scans operations and, when an operation works on or returns a ranked tensor annotated with a GPU environment, wraps that operation in `region.env_region`. The wrapped operation is cloned inside the region and the region yields the cloned results. This preserves device placement as an explicit region structure for later passes.

The second NDArray transformation pass is `coalesce-shard-ops`. It works with Mesh/Sharding annotations and NDArray view/update operations. Its goal is to reduce redundant sharding or resharding operations. It computes local bounding boxes for multiple uses of repartitioned arrays, updates `ndarray.subview` and `ndarray.insert_slice` users, and treats `ndarray.insert_slice` as a mutation barrier because it writes data.

The main conversion pass is `convert-ndarray-to-linalg`. It converts most NDArray operations into lower MLIR dialects:

- `ndarray.copy` lowers to allocation, bufferization, `memref.copy`, and tensor conversion.
- `ndarray.delete` lowers to `memref.dealloc`.
- `ndarray.linspace` lowers to arithmetic plus `linalg.generic`.
- `ndarray.reshape` lowers to `tensor.from_elements` plus `tensor.reshape`, optionally after a copy.
- `ndarray.cast_elemtype` lowers to a parallel `linalg.generic` or folds away when no conversion is needed.
- `ndarray.subview` and `ndarray.insert_slice` are handled specially because they model views and mutation. The pass marks them dynamically legal during partial conversion; conversion utilities and bufferization support preserve their memref-like behavior for later stages.

Related passes outside the NDArray directory often appear in the same IMEX pipeline. `drop-regions` removes Region dialect operations, `convert-region-parallel-loops-to-gpu` lowers mapped parallel loops in GPU regions to GPU launches, and general GPU memory movement passes such as `insert-gpu-allocs` and `insert-gpu-copy` may run after NDArray has expressed device placement. Those are not NDArray ops, but they are part of the practical path from high-level array code to GPU-oriented IR.

## What It Implies

When you see `ndarray` in IR, assume the compiler is still preserving source-level array behavior. In-place updates, explicit deletion, copy requests, view guarantees, and placement environments are still meant to be visible.

This has two important implications. First, not every operation is purely functional even when it uses tensor types. `ndarray.insert_slice` and `ndarray.delete` carry mutation or lifetime meaning. Second, an annotated tensor type may guide where computation should run. A tensor with `#ndarray.envs<#region.gpu_env<...>>` is not just a tensor value; it is a tensor value whose data placement should influence surrounding transformation choices.

The dialect also implies that lowering will soon leave the purely high-level array world. After `convert-ndarray-to-linalg`, most operations become linalg, tensor, memref, bufferization, and region operations. At that point, standard MLIR optimizations and target-specific IMEX passes can take over.

## How To Read NDArray IR

Start with the operations that can affect aliasing: `ndarray.subview`, `ndarray.insert_slice`, `ndarray.copy`, and `ndarray.delete`. These tell you whether a value is a view, an in-place update, a new allocation, or the end of an array's lifetime.

Then check tensor encodings. If you see `#ndarray.envs<...>`, inspect the contained environment attributes. A GPU environment usually means `add-gpu-regions` can introduce Region dialect wrappers around related operations.

Finally, look for conversion boundaries. If `convert-ndarray-to-linalg` has not run, array semantics are still explicit. If it has run, expect to reason in terms of `linalg.generic`, `tensor.reshape`, `memref.subview`, `memref.copy`, `memref.dealloc`, and bufferization casts instead.

## Minimal Example

This example shows a GPU-annotated array creation and view:

```mlir
#GPUENV = #ndarray.envs<#region.gpu_env<device = "XeGPU">>

func.func @example(%start: i64, %stop: i64, %count: i64) -> tensor<33xi64, #GPUENV> {
  %arr = ndarray.linspace %start %stop %count false
      : (i64, i64, i64) -> tensor<33xi64, #GPUENV>
  return %arr : tensor<33xi64, #GPUENV>
}
```

After `add-gpu-regions`, an operation like the `ndarray.linspace` can be wrapped in a `region.env_region` carrying the GPU environment. After `convert-ndarray-to-linalg`, the linspace computation becomes arithmetic plus a parallel `linalg.generic`.

The mental model is simple: `ndarray` keeps array intent alive until the compiler has enough structure to lower it without losing mutation, view, and placement meaning.
